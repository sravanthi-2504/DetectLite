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
    a = (box[2]-box[0]).clamp(min=0) * (box[3]-box[1]).clamp(min=0)
    b = (boxes[:,2]-boxes[:,0]).clamp(min=0) * (boxes[:,3]-boxes[:,1]).clamp(min=0)
    return inter / (a + b - inter).clamp(min=1e-12)

def ap_from_pr(rec, prec):
    mrec = torch.cat([torch.tensor([0.]), rec, torch.tensor([1.])])
    mpre = torch.cat([torch.tensor([0.]), prec, torch.tensor([0.])])
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = torch.maximum(mpre[i-1], mpre[i])
    idx = torch.nonzero(mrec[1:] != mrec[:-1]).flatten() + 1
    return float(torch.sum((mrec[idx]-mrec[idx-1]) * mpre[idx]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--score-threshold", type=float, default=0.05)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    ds = VOCDataset(args.data, split=args.split, image_size=224)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)
    model = MobileViTDetector(num_classes=20).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    detections = defaultdict(list)
    gt = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting detections"):
            images = images.to(device)
            outputs = model(images)
            boxes, scores, labels = decode_predictions(
                outputs, score_threshold=args.score_threshold
            )
            for i, t in enumerate(targets):
                image_id = t["image_id"]
                for b, l in zip(t["boxes"], t["labels"]):
                    gt[int(l)][image_id].append(b.cpu())
                for b, s, l in zip(boxes[i], scores[i], labels[i]):
                    detections[int(l)].append((float(s), b.cpu(), image_id))

    aps = []
    for cls in range(20):
        n_gt = sum(len(v) for v in gt[cls].values())
        if n_gt == 0:
            continue
        preds = sorted(detections[cls], key=lambda x: x[0], reverse=True)
        matched = {k: torch.zeros(len(v), dtype=torch.bool) for k, v in gt[cls].items()}
        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))

        for j, (_, pb, image_id) in enumerate(preds):
            g = torch.stack(gt[cls].get(image_id, [])) if gt[cls].get(image_id) else torch.empty((0,4))
            if len(g) == 0:
                fp[j] = 1
                continue
            ious = iou_one_to_many(pb, g)
            best, idx = ious.max(0)
            idx = int(idx)
            if float(best) >= args.iou_threshold and not matched[image_id][idx]:
                tp[j] = 1
                matched[image_id][idx] = True
            else:
                fp[j] = 1

        if len(preds):
            tc, fc = torch.cumsum(tp,0), torch.cumsum(fp,0)
            rec = tc / n_gt
            prec = tc / (tc+fc).clamp(min=1e-12)
            ap = ap_from_pr(rec, prec)
        else:
            ap = 0.0
        aps.append(ap)
        print(f"class {cls:2d}: AP={ap:.4f}")

    print(f"mAP@{args.iou_threshold:.2f}: {sum(aps)/max(len(aps),1):.4f}")

if __name__ == "__main__":
    main()
