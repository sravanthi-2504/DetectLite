import torch
from torchvision.ops import box_iou


def precision_recall_f1(pred_boxes, pred_labels, pred_scores,
                        gt_boxes, gt_labels, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        tp = 0
        fp = 0
    else:
        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        matched = set()
        tp = 0
        fp = 0
        ious = box_iou(pred_boxes, gt_boxes) if len(gt_boxes) else torch.zeros((len(pred_boxes), 0))

        for i in range(len(pred_boxes)):
            candidates = [
                j for j in range(len(gt_boxes))
                if j not in matched and pred_labels[i].item() == gt_labels[j].item()
            ]
            if candidates:
                best_j = max(candidates, key=lambda j: float(ious[i, j]))
                if float(ious[i, best_j]) >= iou_threshold:
                    tp += 1
                    matched.add(best_j)
                else:
                    fp += 1
            else:
                fp += 1

    fn = max(len(gt_boxes) - tp, 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}
