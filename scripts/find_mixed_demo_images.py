import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATASET = PROJECT_ROOT / "datasets/VOCdevkit/VOC2012"
VAL_FILE = DATASET / "ImageSets/Main/val.txt"
ANNOTATIONS = DATASET / "Annotations"

image_ids = [
    line.strip()
    for line in VAL_FILE.read_text().splitlines()
    if line.strip()
]

candidates = []

for image_id in image_ids:
    xml_path = ANNOTATIONS / f"{image_id}.xml"

    if not xml_path.exists():
        continue

    root = ET.parse(xml_path).getroot()

    classes = []

    for obj in root.findall("object"):
        difficult = obj.findtext("difficult", "0")

        if difficult == "1":
            continue

        name = obj.findtext("name")

        if name:
            classes.append(name)

    unique_classes = sorted(set(classes))

    if len(unique_classes) >= 2:
        candidates.append(
            (
                len(unique_classes),
                len(classes),
                image_id,
                unique_classes
            )
        )

# Prefer images with several different object classes
candidates.sort(
    key=lambda x: (x[0], x[1]),
    reverse=True
)

print("=" * 75)
print("MIXED-OBJECT VOC VALIDATION CANDIDATES")
print("=" * 75)
print()

for i, (unique_count, object_count, image_id, classes) in enumerate(
    candidates[:30],
    1
):
    print(
        f"{i:2d}. {image_id} | "
        f"objects={object_count} | "
        f"classes={', '.join(classes)}"
    )

print()
print("Total mixed-object validation images:", len(candidates))
