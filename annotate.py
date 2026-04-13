"""
annotate.py — Simple CLI annotation tool for Windows.
No Doccano needed. Run this, paste contract text, highlight entities interactively.

Usage:
    poetry run python annotate.py
    
Output:
    data/annotations/train.jsonl  (appends each annotated contract)
"""

import json
import re
from pathlib import Path

ANNOTATIONS_FILE = Path("data/annotations/train.jsonl")
ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

LABELS = {
    "1": "DATE",
    "2": "PARTY",
    "3": "MONEY",
    "4": "TERMINATION",
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(
                p.extract_text() for p in pdf.pages if p.extract_text()
            )
    except Exception as e:
        print(f"Could not read PDF: {e}")
        return ""


def find_span(text: str, phrase: str):
    """Find start/end position of a phrase in text (case-insensitive)."""
    match = re.search(re.escape(phrase), text, re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None, None


def annotate_document(text: str) -> dict:
    """Interactively annotate a document."""
    labels = []
    print("\n" + "="*60)
    print("DOCUMENT TEXT:")
    print("="*60)
    # Print with line numbers for easy reference
    for i, line in enumerate(text.split("\n"), 1):
        if line.strip():
            print(f"{i:3}: {line}")
    print("="*60)

    print("\nLabel types:")
    for key, label in LABELS.items():
        print(f"  {key} = {label}")

    print("\nInstructions:")
    print("  Type the EXACT text you want to label, then choose its type.")
    print("  Press Enter with empty text when done with this document.\n")

    while True:
        phrase = input("Enter text to label (or press Enter to finish): ").strip()
        if not phrase:
            break

        start, end = find_span(text, phrase)
        if start is None:
            print(f"  ⚠️  Could not find '{phrase}' in document. Check spelling.\n")
            continue

        print(f"  Found: '{text[start:end]}'")
        label_key = input(f"  Label type (1=DATE, 2=PARTY, 3=MONEY, 4=TERMINATION): ").strip()

        if label_key not in LABELS:
            print("  ⚠️  Invalid choice. Skipping.\n")
            continue

        label = LABELS[label_key]
        labels.append([start, end, label])
        print(f"  ✅ Saved: [{start}:{end}] '{phrase}' → {label}\n")

    return {"text": text, "label": labels}


def main():
    print("\n" + "="*60)
    print("  Legal Contract Annotation Tool")
    print("="*60)
    print("\nChoose input method:")
    print("  1 = Load from PDF file")
    print("  2 = Paste text manually")

    choice = input("\nChoice (1 or 2): ").strip()

    if choice == "1":
        pdf_path = input("Enter PDF file path (e.g. test_contract.pdf): ").strip()
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print("Failed to extract text. Try option 2.")
            return
    else:
        print("\nPaste your contract text below.")
        print("When done, type END on a new line and press Enter:\n")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        text = "\n".join(lines)

    if not text.strip():
        print("No text provided. Exiting.")
        return

    # Annotate
    record = annotate_document(text)

    if not record["label"]:
        print("\n⚠️  No labels added. Document not saved.")
        return

    # Save to JSONL
    with open(ANNOTATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    count = sum(1 for _ in open(ANNOTATIONS_FILE, encoding="utf-8"))
    print(f"\n✅ Saved! Total annotated documents: {count}")
    print(f"   File: {ANNOTATIONS_FILE}")
    print("\nRun this script again to annotate another document.")
    print("You need ~50 annotated documents before training.\n")


if __name__ == "__main__":
    main()