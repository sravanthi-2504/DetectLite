import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import MobileViTDetector
from utils.boxes import decode_predictions


DATASET = PROJECT_ROOT / "datasets/VOCdevkit/VOC2012"
CHECKPOINT = PROJECT_ROOT / "checkpoints/mobilevit_xs_pretrained_10epoch.pt"

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def main():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = MobileViTDetector(num_classes=20).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    val_file = DATASET / "ImageSets/Main/val.txt"

    image_ids = [
        x.strip()
        for x in val_file.read_text().splitlines()
        if x.strip()
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    results = []

    for i, image_id in enumerate(image_ids, 1):

        image_path = DATASET / "JPEGImages" / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)

        boxes, scores, labels = decode_predictions(
            outputs,
            score_threshold=0.15,
            nms_iou=0.5
        )

        scores = scores[0].detach().cpu()
        labels = labels[0].detach().cpu()

        if len(scores) < 2:
            continue

        predictions = []

        for score, label in zip(scores.tolist(), labels.tolist()):
            predictions.append(
                (
                    VOC_CLASSES[int(label)],
                    float(score)
                )
            )

        # Keep only the strongest prediction for each class.
        best_by_class = {}

        for class_name, score in predictions:
            if (
                class_name not in best_by_class
                or score > best_by_class[class_name]
            ):
                best_by_class[class_name] = score

        # Need at least two different predicted classes.
        if len(best_by_class) < 2:
            continue

        best = sorted(
            best_by_class.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Consider the strongest two classes.
        top_two = best[:2]

        combined = top_two[0][1] + top_two[1][1]

        # Reward additional classes, but prioritize confidence.
        quality = (
            combined * 100
            + min(len(best), 4) * 10
        )

        results.append(
            (
                quality,
                image_id,
                best
            )
        )

        if i % 500 == 0:
            print(f"Checked {i}/{len(image_ids)}...")

    results.sort(reverse=True)

    print()
    print("=" * 85)
    print("BEST MULTI-CLASS PREDICTIONS")
    print("=" * 85)
    print()

    for i, (_, image_id, predictions) in enumerate(
        results[:40],
        1
    ):
        text = ", ".join(
            f"{name} {score * 100:.1f}%"
            for name, score in predictions[:5]
        )

        print(
            f"{i:2d}. {image_id} | {text}"
        )

    print()
    print("Total multi-class images:", len(results))


if __name__ == "__main__":
    main()
