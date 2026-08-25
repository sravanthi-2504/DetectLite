import torch
from torchvision.ops import box_iou


def precision_recall_f1(
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_boxes,
    gt_labels,
    iou_threshold=0.5,
):
    """
    Compute image-level detection precision, recall, and F1.

    Predictions and ground truth are moved to the same device before
    IoU calculation.
    """

    # ---------------------------------------------------------
    # Make sure everything is on the same device
    # ---------------------------------------------------------

    if isinstance(pred_boxes, torch.Tensor):
        device = pred_boxes.device
    else:
        device = gt_boxes.device

    pred_boxes = pred_boxes.to(device)
    pred_labels = pred_labels.to(device)
    pred_scores = pred_scores.to(device)

    gt_boxes = gt_boxes.to(device)
    gt_labels = gt_labels.to(device)

    # ---------------------------------------------------------
    # No ground-truth objects
    # ---------------------------------------------------------

    if len(gt_boxes) == 0:

        return {
            "tp": 0,
            "fp": len(pred_boxes),
            "fn": 0,
        }

    # ---------------------------------------------------------
    # No predictions
    # ---------------------------------------------------------

    if len(pred_boxes) == 0:

        return {
            "tp": 0,
            "fp": 0,
            "fn": len(gt_boxes),
        }

    # ---------------------------------------------------------
    # Compute IoU
    # ---------------------------------------------------------

    ious = box_iou(
        pred_boxes,
        gt_boxes,
    )

    # ---------------------------------------------------------
    # Match predictions to ground truth
    #
    # Each ground-truth object can be matched only once.
    # ---------------------------------------------------------

    matched_gt = set()

    tp = 0
    fp = 0

    # Process highest-confidence predictions first
    order = torch.argsort(
        pred_scores,
        descending=True,
    )

    for pred_idx in order:

        pred_idx = int(pred_idx)

        pred_label = int(
            pred_labels[pred_idx]
        )

        best_iou = 0.0
        best_gt = -1

        for gt_idx in range(len(gt_boxes)):

            if gt_idx in matched_gt:
                continue

            gt_label = int(
                gt_labels[gt_idx]
            )

            # Class must match
            if pred_label != gt_label:
                continue

            iou = float(
                ious[pred_idx, gt_idx]
            )

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_idx

        if (
            best_gt >= 0
            and best_iou >= iou_threshold
        ):
            tp += 1
            matched_gt.add(best_gt)

        else:
            fp += 1

    # Every unmatched ground-truth object is a false negative
    fn = len(gt_boxes) - len(matched_gt)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
