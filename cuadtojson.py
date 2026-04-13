"""
cuad_to_jsonl.py — Converts CUAD dataset to clean NER training format.

Key improvements over v1:
- Only uses questions with consistently short, clean answers
- Filters out full-sentence answers that pollute NER training
- Extracts actual monetary amounts from long money sentences
- Validates all spans before saving

Usage:
    poetry run python cuad_to_jsonl.py --input CUADv1.json --output data/annotations

Output:
    data/annotations/train.jsonl  (80%)
    data/annotations/val.jsonl    (10%)
    data/annotations/test.jsonl   (10%)
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional, Tuple

# ── Answer length limits per question type ────────────────────────────────────
# Based on analysis of CUAD answer length distributions:
# Q2  Agreement Date  → avg 18 chars  → keep answers < 60 chars
# Q3  Effective Date  → avg 62 chars  → keep answers < 60 chars only
# Q4  Expiration Date → avg 231 chars → skip (too noisy)
# Q1  Parties         → avg 24 chars  → keep answers < 80 chars
# Q15 Termination     → avg 222 chars → keep all (termination clauses ARE sentences)
# Q21 Money           → avg 353 chars → extract amounts only

# Question index → (label, max_answer_length)
# max_answer_length=None means keep all lengths
QUESTION_CONFIG = {
    1:  ("PARTY",       80),   # Parties — short names only
    2:  ("DATE",        60),   # Agreement Date — clean dates
    3:  ("DATE",        60),   # Effective Date — short only
    5:  ("DATE",        60),   # Renewal Term — short only
    15: ("TERMINATION", None), # Termination For Convenience — keep full clauses
    17: ("TERMINATION", None), # Change Of Control
    32: ("TERMINATION", 400),  # Post-Termination Services
}

# Regex to extract monetary amounts from long sentences
_MONEY_PATTERN = re.compile(
    r"""
    (?:USD|EUR|GBP|INR|AUD|CAD)?   # optional currency code
    \s*[\$€£₹]?\s*                  # optional symbol
    \d{1,3}(?:,\d{3})*(?:\.\d+)?  # number with commas
    (?:\s*(?:million|billion|thousand|mn|bn|k))? # optional multiplier
    (?:\s*(?:USD|EUR|GBP|INR|AUD|CAD))?  # optional trailing code
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Date patterns to extract from long date sentences
_DATE_PATTERN = re.compile(
    r"""
    \b(?:
        (?:January|February|March|April|May|June|July|
           August|September|October|November|December|
           Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)
        \s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}   # Month DD, YYYY
        |\d{1,2}(?:st|nd|rd|th)?\s+
        (?:January|February|March|April|May|June|July|
           August|September|October|November|December)
        \s+\d{4}                                  # DD Month YYYY
        |\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}       # MM/DD/YYYY
        |\d{4}[\/\-]\d{2}[\/\-]\d{2}             # YYYY-MM-DD
        |\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+
        (?:January|February|March|April|May|June|July|
           August|September|October|November|December),?\s+\d{4}
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def find_span(context: str, answer_text: str, answer_start: int) -> Tuple[Optional[int], Optional[int]]:
    """Find exact character span of answer in context."""
    # Try exact position
    end = answer_start + len(answer_text)
    if context[answer_start:end] == answer_text:
        return answer_start, end

    # Search nearby
    search_start = max(0, answer_start - 20)
    search_end = min(len(context), answer_start + len(answer_text) + 20)
    idx = context.find(answer_text, search_start, search_end)
    if idx != -1:
        return idx, idx + len(answer_text)

    # Full context search
    idx = context.find(answer_text)
    if idx != -1:
        return idx, idx + len(answer_text)

    return None, None


def extract_money_spans(context: str, long_answer: str, answer_start: int):
    """
    Extract just the monetary amount substrings from a long money sentence.
    Returns list of (start, end, label) tuples.
    """
    spans = []
    # Find the long answer in context first
    start, end = find_span(context, long_answer, answer_start)
    if start is None:
        return spans

    # Search within the answer for monetary patterns
    answer_slice = context[start:end]
    for m in _MONEY_PATTERN.finditer(answer_slice):
        match_text = m.group().strip()
        # Must contain a digit and be meaningful length
        if not any(c.isdigit() for c in match_text) or len(match_text) < 2:
            continue
        # Must contain a currency signal OR be a large number
        has_currency = any(sym in match_text for sym in ['$', '€', '£', '₹', 'USD', 'EUR', 'GBP', 'INR'])
        # Parse numeric value
        numeric = re.sub(r'[^\d.]', '', match_text.replace(',', ''))
        try:
            value = float(numeric) if numeric else 0
        except ValueError:
            value = 0
        if has_currency or value >= 1000:
            span_start = start + m.start()
            span_end = start + m.end()
            spans.append((span_start, span_end, "MONEY"))

    return spans


def extract_date_spans(context: str, long_answer: str, answer_start: int):
    """Extract clean date substrings from a long effective/expiration date sentence."""
    spans = []
    start, end = find_span(context, long_answer, answer_start)
    if start is None:
        return spans

    answer_slice = context[start:end]
    for m in _DATE_PATTERN.finditer(answer_slice):
        match_text = m.group().strip()
        if len(match_text) < 6:  # too short to be a real date
            continue
        span_start = start + m.start()
        span_end = start + m.end()
        spans.append((span_start, span_end, "DATE"))

    return spans


def convert_cuad(input_path: Path, output_dir: Path, seed: int = 42):
    print(f"Loading {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        cuad = json.load(f)

    contracts = cuad["data"]
    print(f"Found {len(contracts)} contracts.")

    records = []
    total_labels = 0
    skipped_length = 0
    skipped_span = 0

    for contract in contracts:
        full_context = ""
        all_labels = []

        for para in contract["paragraphs"]:
            context = para["context"]
            para_offset = len(full_context)

            for q_idx, qa in enumerate(para["qas"]):
                config = QUESTION_CONFIG.get(q_idx)
                if config is None:
                    continue

                label, max_len = config

                for answer in qa["answers"]:
                    answer_text = answer["text"].strip()
                    answer_start = answer["answer_start"]

                    if not answer_text or len(answer_text) < 2:
                        continue

                    # ── Handle long answers ───────────────────────────────────
                    if max_len and len(answer_text) > max_len:
                        # For dates: extract clean date substrings
                        if label == "DATE":
                            extracted = extract_date_spans(context, answer_text, answer_start)
                            for s, e, l in extracted:
                                all_labels.append([para_offset + s, para_offset + e, l])
                                total_labels += 1
                        else:
                            skipped_length += 1
                        continue

                    # ── Normal short answer — find exact span ─────────────────
                    start, end = find_span(context, answer_text, answer_start)
                    if start is None:
                        skipped_span += 1
                        continue

                    all_labels.append([para_offset + start, para_offset + end, label])
                    total_labels += 1

            # ── Extract monetary values from full contract text ────────────────
            # Scan the paragraph directly for money patterns
            for m in _MONEY_PATTERN.finditer(context):
                match_text = m.group().strip()
                if not any(c.isdigit() for c in match_text) or len(match_text) < 3:
                    continue
                has_currency = any(sym in match_text for sym in ['$', '€', '£', '₹', 'USD', 'EUR', 'GBP'])
                numeric = re.sub(r'[^\d.]', '', match_text.replace(',', ''))
                try:
                    value = float(numeric) if numeric else 0
                except ValueError:
                    value = 0
                if has_currency and value >= 100:
                    all_labels.append([
                        para_offset + m.start(),
                        para_offset + m.end(),
                        "MONEY"
                    ])
                    total_labels += 1

            full_context += context + "\n\n"

        if not all_labels:
            continue

        # Deduplicate overlapping labels
        all_labels.sort(key=lambda x: x[0])
        deduped = []
        last_end = -1
        for label in all_labels:
            if label[0] >= last_end:
                deduped.append(label)
                last_end = label[1]

        records.append({
            "text": full_context.strip(),
            "label": deduped,
        })

    print(f"\nConversion complete:")
    print(f"  Contracts:      {len(records)}")
    print(f"  Total labels:   {total_labels}")
    print(f"  Skipped (long): {skipped_length}")
    print(f"  Skipped (span): {skipped_span}")

    # Label distribution
    label_counts = {}
    for r in records:
        for _, _, l in r["label"]:
            label_counts[l] = label_counts.get(l, 0) + 1
    print(f"\nLabel distribution:")
    for l, c in sorted(label_counts.items()):
        print(f"  {l}: {c}")

    # Split
    random.seed(seed)
    random.shuffle(records)
    total = len(records)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        "train.jsonl": records[:train_end],
        "val.jsonl":   records[train_end:val_end],
        "test.jsonl":  records[val_end:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_dir}:")
    for filename, split_records in splits.items():
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record) + "\n")
        print(f"  {filename}: {len(split_records)} contracts")

    print(f"\n✅ Done! Now run:")
    print(f"  poetry run python training/train.py --base_model nlpaueb/legal-bert-base-uncased")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("CUADv1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}")
        return

    convert_cuad(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()