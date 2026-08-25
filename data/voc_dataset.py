import os
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
    """
    Pascal VOC 2007/2012 XML loader.

    Expected:
      root/
        JPEGImages/
        Annotations/
        ImageSets/Main/train.txt
    """
    def __init__(self, root, split="train", image_size=224):
        self.root = Path(root)
        self.image_size = image_size
        ids_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
        if not ids_file.exists():
            raise FileNotFoundError(f"Missing split file: {ids_file}")
        self.ids = [x.strip() for x in ids_file.read_text().splitlines() if x.strip()]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_path = self.root / "JPEGImages" / f"{image_id}.jpg"
        ann_path = self.root / "Annotations" / f"{image_id}.xml"

        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size

        root = ET.parse(ann_path).getroot()
        boxes, labels = [], []

        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in CLASS_TO_ID:
                continue
            difficult = obj.findtext("difficult", default="0")
            if difficult == "1":
                continue

            bb = obj.find("bndbox")
            x1 = float(bb.findtext("xmin"))
            y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax"))
            y2 = float(bb.findtext("ymax"))

            sx = self.image_size / original_w
            sy = self.image_size / original_h
            boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
            labels.append(CLASS_TO_ID[name])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": image_id,
            "original_size": (original_h, original_w),
        }
        return self.transform(image), target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)
