import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import MobileViTDetector
from utils.boxes import decode_predictions


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


DATASET = PROJECT_ROOT / "datasets/VOCdevkit/VOC2012"
CHECKPOINT = PROJECT_ROOT / "checkpoints/mobilevit_xs_pretrained_10epoch.pt"
OUTPUT = PROJECT_ROOT / "results/demo_candidates"

THRESHOLD = 0.15
MIN_CONFIDENCE = 0.60


def get_device():
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    device = get_device()
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
        line.strip()
        for line in val_file.read_text().splitlines()
        if line.strip()
    ]

    print("Validation images:", len(image_ids))
    print("Searching for high-confidence detections...")
    print()

    candidates = []

    for index, image_id in enumerate(image_ids, 1):
        image_path = DATASET / "JPEGImages" / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        original = Image.open(image_path).convert("RGB")
        image = original.resize((224, 224))

        from torchvision import transforms

        x = transforms.ToTensor()(image)

        x = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )(x).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)

        boxes, scores, labels = decode_predictions(
            outputs,
            score_threshold=THRESHOLD,
            nms_iou=0.5
        )

        boxes = boxes[0].detach().cpu()
        scores = scores[0].detach().cpu()
        labels = labels[0].detach().cpu()

        if len(scores) == 0:
            continue

        high = scores >= MIN_CONFIDENCE

        high_scores = scores[high]
        high_labels = labels[high]

        if len(high_scores) == 0:
            continue

        # Prefer images with multiple strong detections,
        # but avoid images overloaded with duplicate detections.
        strong_count = len(high_scores)
        max_conf = float(high_scores.max())

        unique_classes = len(set(int(x) for x in high_labels.tolist()))

        duplicate_penalty = max(
            0,
            strong_count - unique_classes
        )

        quality = (
            strong_count * 100
            + unique_classes * 50
            + max_conf * 100
            - duplicate_penalty * 40
        )

        candidates.append(
            (
                quality,
                image_id,
                strong_count,
                unique_classes,
                max_conf
            )
        )

        if index % 100 == 0:
            print(
                f"Checked {index}/{len(image_ids)} images..."
            )

    candidates.sort(reverse=True)

    print()
    print("=" * 70)
    print("TOP DEMO CANDIDATES")
    print("=" * 70)

    selected_classes = set()
    selected = []

    for candidate in candidates:
        quality, image_id, count, unique_classes, max_conf = candidate

        image_path = DATASET / "JPEGImages" / f"{image_id}.jpg"

        # Run detection again so we can report the classes.
        original = Image.open(image_path).convert("RGB")
        image = original.resize((224, 224))

        from torchvision import transforms

        x = transforms.ToTensor()(image)

        x = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )(x).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)

        boxes, scores, labels = decode_predictions(
            outputs,
            score_threshold=THRESHOLD,
            nms_iou=0.5
        )

        labels = labels[0].detach().cpu()
        scores = scores[0].detach().cpu()

        strong = scores >= MIN_CONFIDENCE

        classes = [
            VOC_CLASSES[int(label)]
            for label in labels[strong]
        ]

        # Prefer different object categories across demo images.
        new_classes = set(classes) - selected_classes

        if not new_classes and len(selected) < 3:
            continue

        selected.append(candidate)
        selected_classes.update(classes)

        print(
            f"{len(selected)}. {image_id} | "
            f"objects={count} | "
            f"classes={', '.join(sorted(set(classes)))} | "
            f"max_conf={max_conf * 100:.1f}%"
        )

        if len(selected) >= 10:
            break

    print()
    print("Selected image IDs:")

    for candidate in selected:
        print(candidate[1])

    print()
    print("These are candidate images for the final presentation.")
    print("Use the generated prediction script to render them.")
    print()
    print("Output directory:", OUTPUT)


if __name__ == "__main__":
    main()
