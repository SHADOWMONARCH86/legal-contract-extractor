"""
train.py — Fine-tune a BERT-based NER model on annotated legal contracts.

Usage:
    python training/train.py --data_dir data/annotations --output_dir models/legal-ner-bert-v1

Annotation format expected: Doccano JSONL export.
Model: dslim/bert-base-NER as base → fine-tuned with custom legal NER labels.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── NER Label Schema ──────────────────────────────────────────────────────────

LABEL_LIST = [
    "O",
    "B-DATE", "I-DATE",
    "B-PARTY", "I-PARTY",
    "B-MONEY", "I-MONEY",
    "B-TERM", "I-TERM",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_doccano_jsonl(file_path: Path) -> List[Dict]:
    """
    Load Doccano JSONL export.
    Each line: {"text": "...", "label": [[start, end, "ENTITY_TYPE"], ...]}
    """
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} annotated documents from {file_path}")
    return records


def convert_to_hf_dataset(records: List[Dict], tokenizer):
    """Convert Doccano annotations to HuggingFace token-classification dataset."""
    from datasets import Dataset

    all_tokens, all_labels = [], []

    for record in records:
        text = record["text"]
        span_labels = record.get("label", [])

        # Build character-level label array
        char_labels = ["O"] * len(text)
        for start, end, entity_type in span_labels:
            hf_type = _map_entity_type(entity_type)
            if hf_type:
                char_labels[start] = f"B-{hf_type}"
                for i in range(start + 1, end):
                    char_labels[i] = f"I-{hf_type}"

        # Tokenize and align labels
        encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
        offset_mapping = encoding["offset_mapping"]
        token_labels = []
        for offset_start, offset_end in offset_mapping:
            if offset_start == offset_end:  # Special tokens
                token_labels.append(-100)
            else:
                cl = char_labels[offset_start]
                token_labels.append(LABEL2ID.get(cl, LABEL2ID["O"]))

        all_tokens.append(encoding["input_ids"])
        all_labels.append(token_labels)

    return Dataset.from_dict({"input_ids": all_tokens, "labels": all_labels})


def _map_entity_type(raw_type: str) -> Optional[str]:
    mapping = {
        "DATE": "DATE",
        "PARTY": "PARTY",
        "ORG": "PARTY",
        "MONEY": "MONEY",
        "MONETARY": "MONEY",
        "TERMINATION": "TERM",
        "TERM": "TERM",
    }
    return mapping.get(raw_type.upper())


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_dir: Path,
    output_dir: Path,
    base_model: str = "dslim/bert-base-NER",
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    mlflow_uri: Optional[str] = None,
):
    # ── Import seqeval directly — avoids evaluate/transformers namespace conflict
    try:
        from seqeval.metrics import (
            precision_score,
            recall_score,
            f1_score,
            accuracy_score,
        )
    except ImportError:
        logger.error("seqeval not installed. Run: pip install seqeval")
        sys.exit(1)

    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    # ── Detect device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🚀 GPU detected: {device_name} ({vram:.1f}GB VRAM)")
        use_fp16 = True
        use_gpu = True
    else:
        logger.info("💻 No GPU detected — training on CPU (will be slow)")
        use_fp16 = False
        use_gpu = False

    # ── MLflow tracking ──────────────────────────────────────────────────────
    if mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("legal-ner")
            mlflow.transformers.autolog()
            logger.info(f"MLflow tracking enabled: {mlflow_uri}")
        except ImportError:
            logger.warning("mlflow not installed; skipping experiment tracking.")

    # ── Load tokenizer and model ─────────────────────────────────────────────
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ── Load and split data ──────────────────────────────────────────────────
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        sys.exit(1)

    train_records = load_doccano_jsonl(train_file)
    val_records = (
        load_doccano_jsonl(val_file)
        if val_file.exists()
        else train_records[:max(1, len(train_records) // 10)]
    )

    logger.info("Converting training data to HuggingFace dataset format...")
    train_dataset = convert_to_hf_dataset(train_records, tokenizer)
    val_dataset = convert_to_hf_dataset(val_records, tokenizer)
    logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # ── Training arguments ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to=[],
        fp16=use_fp16,
        dataloader_pin_memory=use_gpu,
        no_cuda=not use_gpu,
    )

    # ── Metrics using seqeval directly ───────────────────────────────────────
    def compute_metrics(eval_pred):
        import numpy as np
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = [
            [ID2LABEL[l] for l in label if l != -100]
            for label in labels
        ]
        true_predictions = [
            [ID2LABEL[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    # ── Trainer ──────────────────────────────────────────────────────────────
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    logger.info(f"  Epochs:     {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  FP16:       {use_fp16}")
    logger.info(f"  Device:     {'GPU' if use_gpu else 'CPU'}")
    trainer.train()

    # ── Save final model ─────────────────────────────────────────────────────
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"✅ Model saved to: {final_path}")

    # ── Evaluate on validation set ───────────────────────────────────────────
    metrics = trainer.evaluate()
    logger.info(f"Final validation metrics: {metrics}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT NER model for legal contracts")
    parser.add_argument("--data_dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--output_dir", type=Path, default=Path("models/legal-ner-bert-v1"))
    parser.add_argument("--base_model", default="dslim/bert-base-NER")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mlflow_uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mlflow_uri=args.mlflow_uri,
    )