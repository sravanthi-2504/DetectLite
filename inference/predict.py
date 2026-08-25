import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from data import VOC_CLASSES
from models.detector import MobileViTDetector
from utils.boxes import decode_predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="results/predictions/result.jpg")
    p.add_argument("--threshold", type=float, default=0.25)
    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MobileViTDetector(num_classes=20).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    original = Image.open(args.image).convert("RGB")
    image = original.resize((224, 224))

    x = transforms.ToTensor()(image)
    x = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )(x).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)

    boxes, scores, labels = decode_predictions(
        outputs, score_threshold=args.threshold
    )
    boxes, scores, labels = boxes[0], scores[0], labels[0]

    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)),
                  f"{VOC_CLASSES[label]} {float(score):.2f}",
                  fill="red")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
