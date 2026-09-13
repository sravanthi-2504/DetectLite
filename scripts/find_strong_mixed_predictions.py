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

CONFIDENCE = 0.40


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

        strong = scores >= CONFIDENCE

        if strong.sum() < 2:
            continue

        strong_scores = scores[strong]
        strong_labels = labels[strong]

        pairs = []

        for score, label in zip(
            strong_scores.tolist(),
            strong_labels.tolist()
        ):
            pairs.append(
                (
                    VOC_CLASSES[int(label)],
                    float(score)
                )
            )

        unique_classes = set(
            label for label, score in pairs
        )

        if len(unique_classes) < 2:
            continue

        # Prefer multiple different classes and high confidence.
        mean_conf = sum(
            score for label, score in pairs
        ) / len(pairs)

        quality = (
            len(unique_classes) * 100
            + len(pairs) * 20
            + mean_conf * 100
        )

        results.append(
            (
                quality,
                image_id,
                pairs
            )
        )

        if i % 500 == 0:
            print(
                f"Checked {i}/{len(image_ids)}..."
            )

    results.sort(reverse=True)

    print()
    print("=" * 80)
    print("STRONG MULTI-CLASS MODEL PREDICTIONS")
    print("=" * 80)
    print()

    if not results:
        print(
            "No validation images contain at least "
            "2 different predicted classes with confidence >= 40%."
        )
        return

    for i, (_, image_id, pairs) in enumerate(
        results[:30],
        1
    ):
        prediction_text = ", ".join(
            f"{label} {score * 100:.1f}%"
            for label, score in pairs
        )

        print(
            f"{i:2d}. {image_id} | "
            f"{prediction_text}"
        )

    print()
    print("Total strong mixed predictions:", len(results))


if __name__ == "__main__":
    main()
