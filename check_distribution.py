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
    counter = Counter()

    files = list((root / split).glob("*.txt"))

    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                cls = int(line.split()[0])
                counter[cls] += 1

    print()
    print(split.upper())
    print("-" * 30)

    for i, name in enumerate(names):
        print(f"{name:<15} {counter[i]:>6}")

    print(f"{'TOTAL':<15} {sum(counter.values()):>6}")
