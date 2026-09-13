import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.voc_dataset import VOC_CLASSES
from models.detector import MobileViTDetector
from utils.boxes import decode_predictions


CLASS_COLORS = {
    "aeroplane": (30, 144, 255),
    "bicycle": (255, 140, 0),
    "bird": (50, 205, 50),
    "boat": (138, 43, 226),
    "bottle": (220, 20, 60),
    "bus": (0, 128, 128),
    "car": (255, 69, 0),
    "cat": (218, 112, 214),
    "chair": (160, 82, 45),
    "cow": (70, 130, 180),
    "diningtable": (184, 134, 11),
    "dog": (34, 139, 34),
    "horse": (139, 69, 19),
    "motorbike": (255, 20, 147),
    "person": (0, 120, 215),
    "pottedplant": (46, 139, 87),
    "sheep": (105, 105, 105),
    "sofa": (128, 0, 128),
    "train": (255, 99, 71),
    "tvmonitor": (0, 100, 0),
}


def get_device():
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def remove_redundant_detections(
    boxes,
    scores,
    labels,
    max_detections=6,
):
    if len(scores) == 0:
        return boxes, scores, labels

    # Keep only the strongest prediction for each class.
    best = {}

    for i in range(len(scores)):
        label = int(labels[i])
        score = float(scores[i])

        if label not in best or score > float(scores[best[label]]):
            best[label] = i

    indices = sorted(
        best.values(),
        key=lambda i: float(scores[i]),
        reverse=True
    )

    indices = indices[:max_detections]

    indices = torch.tensor(
        indices,
        dtype=torch.long,
        device=boxes.device
    )

    return (
        boxes[indices],
        scores[indices],
        labels[indices]
    )


def main():
    p = argparse.ArgumentParser(
        description="DetectLite object detection visualization"
    )

    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--out",
        default="results/predictions/result.jpg"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.15
    )
    p.add_argument(
        "--max-detections",
        type=int,
        default=6
    )

    args = p.parse_args()

    device = get_device()

    print("Device:", device)

    model = MobileViTDetector(
        num_classes=20
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    original = Image.open(
        args.image
    ).convert("RGB")

    original_width, original_height = original.size

    resized = original.resize(
        (224, 224)
    )

    x = transforms.ToTensor()(resized)

    x = transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )(x).unsqueeze(0).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(2):
            model(x)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(x)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter() - start
    ) * 1000

    boxes, scores, labels = decode_predictions(
        outputs,
        score_threshold=args.threshold,
        nms_iou=0.5
    )

    boxes = boxes[0].detach().cpu()
    scores = scores[0].detach().cpu()
    labels = labels[0].detach().cpu()

    # Visualization-only filtering.
    # The actual evaluation code is unchanged.
    boxes, scores, labels = remove_redundant_detections(
        boxes,
        scores,
        labels,
        max_detections=args.max_detections
    )

    # Convert 224x224 coordinates back to original image size.
    scale_x = original_width / 224.0
    scale_y = original_height / 224.0

    if len(boxes) > 0:
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y

    draw = ImageDraw.Draw(original)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc",
            17
        )
    except Exception:
        font = ImageFont.load_default()

    try:
        info_font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc",
            15
        )
    except Exception:
        info_font = ImageFont.load_default()

    line_width = max(
        2,
        round(min(original_width, original_height) / 180)
    )

    # Draw detections.
    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):
        x1, y1, x2, y2 = box.tolist()

        class_name = VOC_CLASSES[int(label)]
        confidence = float(score)

        color = CLASS_COLORS.get(
            class_name,
            (255, 0, 0)
        )

        # Bounding box.
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width
        )

        text = (
            f"{class_name} "
            f"{confidence * 100:.1f}%"
        )

        text_box = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = (
            text_box[2] - text_box[0]
        )

        text_height = (
            text_box[3] - text_box[1]
        )

        padding_x = 6
        padding_y = 4

        label_width = (
            text_width + padding_x * 2
        )

        label_height = (
            text_height + padding_y * 2
        )

        # Prefer placing label above the box.
        label_x = max(
            0,
            min(
                x1,
                original_width - label_width
            )
        )

        label_y = y1 - label_height - 2

        # If there isn't enough room above,
        # put the label just inside the box.
        if label_y < 0:
            label_y = y1 + 2

        draw.rounded_rectangle(
            (
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height
            ),
            radius=4,
            fill=color
        )

        draw.text(
            (
                label_x + padding_x,
                label_y + padding_y - 1
            ),
            text,
            fill="white",
            font=font
        )

    # Information panel placed in bottom-left
    # so it does not cover the top detections.
    info = (
        f"DetectLite  |  "
        f"{len(boxes)} objects  |  "
        f"{inference_time:.2f} ms"
    )

    info_box = draw.textbbox(
        (0, 0),
        info,
        font=info_font
    )

    info_width = (
        info_box[2] - info_box[0]
    )

    info_height = (
        info_box[3] - info_box[1]
    )

    margin = 10
    padding = 8

    info_x = margin
    info_y = (
        original_height
        - info_height
        - padding * 2
        - margin
    )

    draw.rounded_rectangle(
        (
            info_x,
            info_y,
            info_x
            + info_width
            + padding * 2,
            info_y
            + info_height
            + padding * 2
        ),
        radius=5,
        fill=(0, 0, 0)
    )

    draw.text(
        (
            info_x + padding,
            info_y + padding - 1
        ),
        info,
        fill="white",
        font=info_font
    )

    Path(args.out).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    original.save(
        args.out,
        quality=95
    )

    print()
    print("Detections:", len(boxes))

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):
        print(
            f"  {VOC_CLASSES[int(label)]}: "
            f"{float(score) * 100:.1f}%"
        )

    print(
        f"Inference time: "
        f"{inference_time:.2f} ms"
    )

    print(
        "Visualization threshold:",
        args.threshold
    )

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
