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


def ap_from_pr(rec, prec):
    mrec = torch.cat([
        torch.tensor([0.0]),
        rec,
        torch.tensor([1.0])
    ])

    mpre = torch.cat([
        torch.tensor([0.0]),
        prec,
        torch.tensor([0.0])
    ])

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.nonzero(
        mrec[1:] != mrec[:-1]
    ).flatten() + 1

    return float(
        torch.sum(
            (mrec[idx] - mrec[idx - 1]) * mpre[idx]
        )
    )


def evaluate_at_iou(detections, gt, iou_threshold):
    aps = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in range(20):
        n_gt = sum(len(v) for v in gt[cls].values())

        if n_gt == 0:
            continue

        preds = sorted(
            detections[cls],
            key=lambda x: x[0],
            reverse=True
        )

        matched = {
            k: torch.zeros(
                len(v),
                dtype=torch.bool
            )
            for k, v in gt[cls].items()
        }

        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))

        for j, (_, pred_box, image_id) in enumerate(preds):
            gt_boxes = (
                torch.stack(gt[cls][image_id])
                if gt[cls].get(image_id)
                else torch.empty((0, 4))
            )

            if len(gt_boxes) == 0:
                fp[j] = 1
                continue

            ious = iou_one_to_many(
                pred_box,
                gt_boxes
            )

            best, idx = ious.max(0)
            idx = int(idx)

            if (
                float(best) >= iou_threshold
                and not matched[image_id][idx]
            ):
                tp[j] = 1
                matched[image_id][idx] = True
            else:
                fp[j] = 1

        if len(preds):
            tc = torch.cumsum(tp, 0)
            fc = torch.cumsum(fp, 0)

            rec = tc / n_gt
            prec = tc / (tc + fc).clamp(min=1e-12)

            ap = ap_from_pr(rec, prec)
        else:
            ap = 0.0

        aps.append(ap)

        total_tp += int(tp.sum())
        total_fp += int(fp.sum())
        total_fn += n_gt - int(tp.sum())

    map_value = sum(aps) / max(len(aps), 1)

    precision = (
        total_tp / max(total_tp + total_fp, 1)
    )

    recall = (
        total_tp / max(total_tp + total_fn, 1)
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return (
        map_value,
        precision,
        recall,
        f1,
        total_tp,
        total_fp,
        total_fn
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05
    )

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

    model.load_state_dict(
        checkpoint["model"]
    )

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
                score_threshold=args.score_threshold
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

    print()
    print("=" * 60)
    print("FINAL DETECTION EVALUATION")
    print("=" * 60)
    print(f"Score threshold: {args.score_threshold}")
    print()

    iou_thresholds = [
        round(x, 2)
        for x in torch.arange(
            0.50,
            0.951,
            0.05
        ).tolist()
    ]

    map_values = []

    for iou_threshold in iou_thresholds:
        result = evaluate_at_iou(
            detections,
            gt,
            iou_threshold
        )

        map_value = result[0]
        map_values.append(map_value)

        print(
            f"mAP@{iou_threshold:.2f}: "
            f"{map_value:.4f}"
        )

    map_50 = map_values[0]
    map_50_95 = sum(map_values) / len(map_values)

    (
        _,
        precision,
        recall,
        f1,
        tp,
        fp,
        fn
    ) = evaluate_at_iou(
        detections,
        gt,
        0.50
    )

    print()
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)

    print(f"mAP@0.50:       {map_50:.4f}")
    print(f"mAP@0.50:0.95:  {map_50_95:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-score:       {f1:.4f}")
    print()
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")


if __name__ == "__main__":
    main()
