from pathlib import Path
from collections import Counter

root = Path("data/garbage_yolo_merged/labels")

names = [
    "biodegradable",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic"
]

for split in ["train", "val", "test"]:

    object_counts = Counter()
    image_counts = Counter()

    files = list((root / split).glob("*.txt"))

    for f in files:

        classes_in_image = set()

        for line in f.read_text().splitlines():

            if not line.strip():
                continue

            cls = int(line.split()[0])

            object_counts[cls] += 1
            classes_in_image.add(cls)

        for cls in classes_in_image:
            image_counts[cls] += 1

    print()
    print("=" * 50)
    print(split.upper())
    print("=" * 50)

    print(f"{'Class':<18} {'Images':>8} {'Objects':>10}")
    print("-" * 50)

    for i, name in enumerate(names):
        print(
            f"{name:<18} "
            f"{image_counts[i]:>8} "
            f"{object_counts[i]:>10}"
        )
