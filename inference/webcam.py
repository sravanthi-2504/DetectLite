import argparse
import cv2
import torch
from PIL import Image

from models.detector import MobileViTDetector
from utils.boxes import decode_predictions
from data import VOC_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.35)
    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = MobileViTDetector(num_classes=20).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((224, 224))

        x = torch.from_numpy(__import__("numpy").array(pil)).float()
        x = x.permute(2, 0, 1) / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / \
            torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
        boxes, scores, labels = decode_predictions(
            outputs, score_threshold=args.threshold
        )

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            sx = frame.shape[1] / 224
            sy = frame.shape[0] / 224
            x1, x2 = int(x1*sx), int(x2*sx)
            y1, y2 = int(y1*sy), int(y2*sy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{VOC_CLASSES[label]} {float(score):.2f}",
                        (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        cv2.imshow("MobileViT Object Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
