from pathlib import Path
from collections import Counter

root = Path("datasets/RDD_SPLIT")

for split in ["train", "val", "test"]:
    counts = Counter()

    for file in (root / split / "labels").glob("*.txt"):
        for line in file.read_text().splitlines():
            parts = line.split()
            if parts:
                counts[parts[0]] += 1

    print(f"{split}: {dict(counts)}")