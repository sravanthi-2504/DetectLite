import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VOCDataset, collate_fn
from models.detector import MobileViTDetector
from utils.boxes import decode_predictions
from evaluation.metrics import precision_recall_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", default="results/metrics.csv")
    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ds = VOCDataset(args.data, split=args.split, image_size=224)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    model = MobileViTDetector(num_classes=20).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    totals = {"tp": 0, "fp": 0, "fn": 0}

    for images, targets in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)

        boxes, scores, labels = decode_predictions(outputs)

        for i, target in enumerate(targets):
            m = precision_recall_f1(
                boxes[i], labels[i], scores[i],
                target["boxes"], target["labels"]
            )
            for k in totals:
                totals[k] += m[k]

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result.keys())
        w.writeheader()
        w.writerow(result)

    print(result)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
