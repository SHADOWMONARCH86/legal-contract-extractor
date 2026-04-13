"""
split_data.py — Split annotations into train / val / test sets.

Usage:
    poetry run python split_data.py

Reads:  data/annotations/train.jsonl  (all annotated documents)
Writes: data/annotations/train.jsonl  (80%)
        data/annotations/val.jsonl    (10%)
        data/annotations/test.jsonl   (10%)
"""

import json
import random
from pathlib import Path

ANNOTATIONS_DIR = Path("data/annotations")
SOURCE_FILE = ANNOTATIONS_DIR / "train.jsonl"


def main():
    if not SOURCE_FILE.exists():
        print(f"❌ No annotations found at {SOURCE_FILE}")
        print("   Run annotate.py first to create some annotations.")
        return

    # Load all records
    records = []
    with open(SOURCE_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if total < 10:
        print(f"⚠️  Only {total} annotated documents found.")
        print("   You need at least 10 to split (ideally 50+).")
        print("   Keep annotating with annotate.py and come back.")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(records)

    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        "train.jsonl": records[:train_end],
        "val.jsonl":   records[train_end:val_end],
        "test.jsonl":  records[val_end:],
    }

    for filename, split_records in splits.items():
        out_path = ANNOTATIONS_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record) + "\n")
        print(f"✅ {filename}: {len(split_records)} documents")

    print(f"\nTotal: {total} documents split into train/val/test.")
    print("You can now run training:")
    print("  poetry run python training/train.py")


if __name__ == "__main__":
    main()