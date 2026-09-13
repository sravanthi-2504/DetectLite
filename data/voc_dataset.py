import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


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

    Training-time augmentation (random resized crop, horizontal flip,
    color jitter) is applied only when `augment=True`. It defaults to
    on for split == "train" and off otherwise, so val/test stay
    deterministic for fair comparison across epochs/checkpoints.
    """
    def __init__(self, root, split="train", image_size=224, augment=None):
        self.root = Path(root)
        self.image_size = image_size
        ids_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
        if not ids_file.exists():
            raise FileNotFoundError(f"Missing split file: {ids_file}")
        self.ids = [x.strip() for x in ids_file.read_text().splitlines() if x.strip()]

        self.augment = (split == "train") if augment is None else augment

        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        )
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.ids)

    def _random_resized_crop(self, image, boxes, labels, scale=(0.7, 1.0), tries=10):
        """Crop a random sub-region and drop/clip boxes accordingly."""
        w, h = image.size
        area = w * h

        for _ in range(tries):
            target_area = random.uniform(*scale) * area
            aspect = random.uniform(0.8, 1.25)
            crop_w = int(round((target_area * aspect) ** 0.5))
            crop_h = int(round((target_area / aspect) ** 0.5))
            if crop_w <= 0 or crop_h <= 0 or crop_w > w or crop_h > h:
                continue

            x0 = random.randint(0, w - crop_w)
            y0 = random.randint(0, h - crop_h)

            if boxes.numel() == 0:
                return image.crop((x0, y0, x0 + crop_w, y0 + crop_h)), boxes, labels

            new_boxes = boxes.clone()
            new_boxes[:, [0, 2]] = (new_boxes[:, [0, 2]] - x0).clamp(0, crop_w)
            new_boxes[:, [1, 3]] = (new_boxes[:, [1, 3]] - y0).clamp(0, crop_h)

            keep = (
                (new_boxes[:, 2] - new_boxes[:, 0] > 4)
                & (new_boxes[:, 3] - new_boxes[:, 1] > 4)
            )
            if keep.sum() == 0:
                continue

            cropped = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
            return cropped, new_boxes[keep], labels[keep]

        # Fell through every attempt (e.g. tiny/unlucky boxes): skip cropping.
        return image, boxes, labels

    def _random_hflip(self, image, boxes, p=0.5):
        if random.random() >= p:
            return image, boxes
        w, _ = image.size
        image = TF.hflip(image)
        if boxes.numel():
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
        return image, boxes

    def _resize(self, image, boxes):
        w, h = image.size
        sx = self.image_size / w
        sy = self.image_size / h
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        if boxes.numel():
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        return image, boxes

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

            # Kept in original pixel coordinates here; scaled to
            # image_size at the very end (after any augmentation).
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_ID[name])

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.augment:
            image, boxes, labels = self._random_resized_crop(image, boxes, labels)
            image, boxes = self._random_hflip(image, boxes)
            image = self.color_jitter(image)

        image, boxes = self._resize(image, boxes)

        image_t = TF.to_tensor(image)
        image_t = self.normalize(image_t)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "original_size": (original_h, original_w),
        }
        return image_t, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)
