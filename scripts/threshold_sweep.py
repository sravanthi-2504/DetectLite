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
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.05, 0.10, 0.15, 0.20, 0.25])
    p.add_argument("--out", default="results/threshold_sweep.csv")
    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    ds = VOCDataset(args.data, split=args.split, image_size=224)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    model = MobileViTDetector(num_classes=20).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    rows = []
    for threshold in args.thresholds:
        totals = {"tp": 0, "fp": 0, "fn": 0}
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=f"threshold={threshold:.2f}"):
                images = images.to(device)
                outputs = model(images)
                boxes, scores, labels = decode_predictions(
                    outputs, score_threshold=threshold
                )
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
        row = {"threshold": threshold, "precision": precision,
               "recall": recall, "f1": f1, **totals}
        rows.append(row)
        print(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
