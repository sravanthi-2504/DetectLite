import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VOCDataset, collate_fn
from models.detector import MobileViTDetector
from utils.boxes import decode_predictions


def iou_one_to_many(box, boxes):
    if boxes.numel() == 0:
        return torch.empty(0)

    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)

    inter = wh[:, 0] * wh[:, 1]

    a = (
        (box[2] - box[0]).clamp(min=0)
        * (box[3] - box[1]).clamp(min=0)
    )

    b = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )

    return inter / (a + b - inter).clamp(min=1e-12)


def evaluate(detections, gt, threshold):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in range(20):
        n_gt = sum(len(v) for v in gt[cls].values())

        if n_gt == 0:
            continue

        preds = [
            p for p in detections[cls]
            if p[0] >= threshold
        ]

        preds.sort(key=lambda x: x[0], reverse=True)

        matched = {
            k: torch.zeros(len(v), dtype=torch.bool)
            for k, v in gt[cls].items()
        }

        tp = 0
        fp = 0

        for _, pred_box, image_id in preds:
            gt_boxes = (
                torch.stack(gt[cls][image_id])
                if gt[cls].get(image_id)
                else torch.empty((0, 4))
            )

            if len(gt_boxes) == 0:
                fp += 1
                continue

            ious = iou_one_to_many(pred_box, gt_boxes)
            best, idx = ious.max(0)
            idx = int(idx)

            if (
                float(best) >= 0.50
                and not matched[image_id][idx]
            ):
                tp += 1
                matched[image_id][idx] = True
            else:
                fp += 1

        fn = n_gt - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1, total_tp, total_fp, total_fn


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")

    args = parser.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    dataset = VOCDataset(
        args.data,
        split=args.split,
        image_size=224
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = MobileViTDetector(
        num_classes=20
    ).to(device).eval()

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(checkpoint["model"])

    detections = defaultdict(list)
    gt = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, targets in tqdm(
            loader,
            desc="Collecting detections"
        ):
            images = images.to(device)

            outputs = model(images)

            boxes, scores, labels = decode_predictions(
                outputs,
                score_threshold=0.01
            )

            for i, target in enumerate(targets):
                image_id = target["image_id"]

                for box, label in zip(
                    target["boxes"],
                    target["labels"]
                ):
                    gt[int(label)][image_id].append(
                        box.cpu()
                    )

                for box, score, label in zip(
                    boxes[i],
                    scores[i],
                    labels[i]
                ):
                    detections[int(label)].append(
                        (
                            float(score),
                            box.cpu(),
                            image_id
                        )
                    )

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50
    ]

    print()
    print("=" * 80)
    print("PRECISION / RECALL / F1 THRESHOLD SWEEP")
    print("=" * 80)
    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F1':>12} "
        f"{'TP':>8} "
        f"{'FP':>8} "
        f"{'FN':>8}"
    )
    print("-" * 80)

    best_f1 = (-1, None)

    for threshold in thresholds:
        precision, recall, f1, tp, fp, fn = evaluate(
            detections,
            gt,
            threshold
        )

        print(
            f"{threshold:>10.2f} "
            f"{precision:>12.4f} "
            f"{recall:>12.4f} "
            f"{f1:>12.4f} "
            f"{tp:>8} "
            f"{fp:>8} "
            f"{fn:>8}"
        )

        if f1 > best_f1[0]:
            best_f1 = (f1, threshold)

    print()
    print(
        f"Best F1: {best_f1[0]:.4f} "
        f"at threshold {best_f1[1]:.2f}"
    )


if __name__ == "__main__":
    main()
