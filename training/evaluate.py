"""
evaluate.py — Evaluate a trained NER model on a held-out test set.

Usage:
    python training/evaluate.py \
        --model_path models/legal-ner-bert-v1/final \
        --test_file data/annotations/test.jsonl \
        --output_dir reports/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate(model_path: Path, test_file: Path, output_dir: Path):
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        import evaluate as hf_evaluate
    except ImportError:
        logger.error("transformers and evaluate packages required.")
        sys.exit(1)

    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )

    # Load test records
    records = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Evaluating on {len(records)} test documents...")

    seqeval = hf_evaluate.load("seqeval")
    all_true_labels, all_pred_labels = [], []

    for record in records:
        text = record["text"]
        true_spans = record.get("label", [])  # [[start, end, label], ...]

        # Build character-level true labels
        char_true = ["O"] * len(text)
        for start, end, label in true_spans:
            label = label.upper()
            char_true[start] = f"B-{label}"
            for i in range(start + 1, end):
                char_true[i] = f"I-{label}"

        # Run model
        try:
            predictions = ner_pipeline(text[:512])
        except Exception as e:
            logger.warning(f"Inference failed for record: {e}")
            predictions = []

        # Build character-level pred labels
        char_pred = ["O"] * len(text)
        for pred in predictions:
            start, end = pred["start"], pred["end"]
            label = pred["entity_group"].upper()
            if start < len(char_pred):
                char_pred[start] = f"B-{label}"
            for i in range(start + 1, min(end, len(char_pred))):
                char_pred[i] = f"I-{label}"

        # Tokenize for seqeval (word-level)
        tokens = text.split()
        true_seq, pred_seq = [], []
        idx = 0
        for token in tokens:
            if idx < len(char_true):
                true_seq.append(char_true[idx])
                pred_seq.append(char_pred[idx])
            idx += len(token) + 1

        all_true_labels.append(true_seq)
        all_pred_labels.append(pred_seq)

    results = seqeval.compute(predictions=all_pred_labels, references=all_true_labels)

    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    for key, val in results.items():
        logger.info(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    logger.info("=" * 60)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report saved to: {report_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained NER model")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--test_file", type=Path, default=Path("data/annotations/test.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model_path, args.test_file, args.output_dir)
