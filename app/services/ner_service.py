"""
NERService — Singleton wrapper around the NER model.

Strategy:
- Tries to load a fine-tuned HuggingFace model first (set MODEL_PATH in config).
- Falls back to spaCy if HF model not found.
- Falls back to rule-based extraction if spaCy not installed.
- Rule-based extraction ALWAYS runs as a supplement to catch what BERT misses.
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class RawEntity:
    text: str
    label: str
    start: int
    end: int
    score: float = 1.0


@dataclass
class RawExtractionResult:
    dates: List[RawEntity] = field(default_factory=list)
    parties: List[RawEntity] = field(default_factory=list)
    monetary: List[RawEntity] = field(default_factory=list)
    termination_spans: List[RawEntity] = field(default_factory=list)


class NERService:
    """Thread-safe singleton NER service."""

    _instance: Optional["NERService"] = None
    _lock = threading.Lock()

    # Chunk size for transformer inference
    _CHUNK_SIZE = 512
    _CHUNK_OVERLAP = 50

    def __init__(self):
        self._pipeline: Any = None
        self._backend: str = "rule_based"
        self.model_version: str = "rule-based-v1"
        self._load_model()

    @classmethod
    def get_instance(cls) -> "NERService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Model Loading ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._try_load_huggingface():
            return
        if self._try_load_spacy():
            return
        logger.warning("⚠️  No NER model found. Falling back to rule-based extraction.")
        self._backend = "rule_based"
        self.model_version = "rule-based-v1"

    def _try_load_huggingface(self) -> bool:
        model_path = settings.model_path
        if not model_path:
            return False
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            # Auto-detect GPU
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("💻 Using CPU for inference")

            logger.info(f"Loading HuggingFace NER model from: {model_path}")
            self._pipeline = hf_pipeline(
                "ner",
                model=model_path,
                aggregation_strategy="simple",
                device=device,
            )
            self._backend = "huggingface"
            self.model_version = f"hf-{model_path.split('/')[-1]}"
            logger.info(f"✅ HuggingFace model loaded: {self.model_version}")
            return True
        except Exception as e:
            logger.warning(f"HuggingFace model load failed: {e}")
            return False

    def _try_load_spacy(self) -> bool:
        try:
            import spacy
            model_name = settings.spacy_model or "en_core_web_sm"
            logger.info(f"Loading spaCy model: {model_name}")
            self._pipeline = spacy.load(model_name)
            self._backend = "spacy"
            self.model_version = f"spacy-{model_name}"
            logger.info(f"✅ spaCy model loaded: {self.model_version}")
            return True
        except Exception as e:
            logger.warning(f"spaCy model load failed: {e}")
            return False

    # ── Extraction Dispatch ──────────────────────────────────────────────────

    def extract(self, text: str) -> RawExtractionResult:
        if self._backend == "huggingface":
            return self._extract_huggingface(text)
        if self._backend == "spacy":
            return self._extract_spacy(text)
        return self._extract_rule_based(text)

    # ── HuggingFace NER ──────────────────────────────────────────────────────

    def _extract_huggingface(self, text: str) -> RawExtractionResult:
        result = RawExtractionResult()
        chunks = self._chunk_text(text)

        char_offset = 0
        for chunk_text, chunk_start in chunks:
            try:
                predictions = self._pipeline(chunk_text)
            except Exception as e:
                logger.error(f"HF NER inference error: {e}")
                predictions = []

            for pred in predictions:
                entity = RawEntity(
                    # Restore original case from source text using char positions
                    text=text[chunk_start + pred["start"]: chunk_start + pred["end"]],
                    label=pred["entity_group"],
                    start=chunk_start + pred["start"],
                    end=chunk_start + pred["end"],
                    score=round(pred["score"], 4),
                )
                self._route_entity(entity, result)

        # Always supplement BERT with rule-based to catch what model misses
        self._supplement_with_rules(text, result)
        return result

    # ── spaCy NER ───────────────────────────────────────────────────────────

    def _extract_spacy(self, text: str) -> RawExtractionResult:
        result = RawExtractionResult()
        doc = self._pipeline(text[:1_000_000])
        for ent in doc.ents:
            entity = RawEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                score=0.85,
            )
            self._route_entity(entity, result)
        self._supplement_with_rules(text, result)
        return result

    # ── Rule-based Fallback ──────────────────────────────────────────────────

    def _extract_rule_based(self, text: str) -> RawExtractionResult:
        result = RawExtractionResult()
        self._detect_dates_rule(text, result)
        self._detect_parties_rule(text, result)
        self._detect_money_rule(text, result)
        self._detect_termination_clauses(text, result)
        return result

    def _supplement_with_rules(self, text: str, result: RawExtractionResult):
        """
        Run rule-based detection and add findings not already covered by BERT/spaCy.
        This ensures dates, money and termination clauses are caught even when
        the model is uncertain.
        """
        # Track existing spans to avoid duplicates
        existing_spans = set()
        for bucket in [result.dates, result.parties, result.monetary, result.termination_spans]:
            for ent in bucket:
                existing_spans.add((ent.start, ent.end))

        def _add_if_new(entity: RawEntity, bucket: List):
            # Check if this span overlaps with any existing entity
            for s, e in existing_spans:
                if not (entity.end <= s or entity.start >= e):
                    return  # overlaps — skip
            existing_spans.add((entity.start, entity.end))
            bucket.append(entity)

        # Supplement dates
        rule_dates = RawExtractionResult()
        self._detect_dates_rule(text, rule_dates)
        for ent in rule_dates.dates:
            _add_if_new(ent, result.dates)

        # Supplement parties
        rule_parties = RawExtractionResult()
        self._detect_parties_rule(text, rule_parties)
        for ent in rule_parties.parties:
            _add_if_new(ent, result.parties)

        # Supplement money
        rule_money = RawExtractionResult()
        self._detect_money_rule(text, rule_money)
        for ent in rule_money.monetary:
            _add_if_new(ent, result.monetary)

        # Supplement termination clauses
        rule_term = RawExtractionResult()
        self._detect_termination_clauses(text, rule_term)
        for ent in rule_term.termination_spans:
            _add_if_new(ent, result.termination_spans)

    # ── Entity Routing ───────────────────────────────────────────────────────

    # Maps both our custom fine-tuned labels AND standard spaCy/HF labels
    LABEL_MAP: Dict[str, str] = {
        # ── Our fine-tuned model labels ──
        "DATE":         "dates",
        "PARTY":        "parties",
        "MONEY":        "monetary",
        "TERM":         "termination_spans",
        # ── Standard spaCy / generic BERT labels ──
        "TIME":         "dates",
        "ORG":          "parties",
        "PER":          "parties",
        "PERSON":       "parties",
        "CARDINAL":     "monetary",
        "GPE":          "parties",
        # ── dslim/bert-base-NER labels ──
        "B-DATE":       "dates",
        "I-DATE":       "dates",
        "B-PARTY":      "parties",
        "I-PARTY":      "parties",
        "B-MONEY":      "monetary",
        "I-MONEY":      "monetary",
        "B-TERM":       "termination_spans",
        "I-TERM":       "termination_spans",
        "B-ORG":        "parties",
        "I-ORG":        "parties",
        "B-PER":        "parties",
        "I-PER":        "parties",
        "B-MISC":       "parties",
    }

    def _route_entity(self, entity: RawEntity, result: RawExtractionResult):
        bucket = self.LABEL_MAP.get(entity.label.upper())
        if bucket == "dates":
            result.dates.append(entity)
        elif bucket == "parties":
            result.parties.append(entity)
        elif bucket == "monetary":
            result.monetary.append(entity)
        elif bucket == "termination_spans":
            result.termination_spans.append(entity)
        else:
            logger.debug(f"Unmapped label: {entity.label} — text: {entity.text[:30]}")

    # ── Rule-based Helpers ────────────────────────────────────────────────────

    _DATE_PATTERN = re.compile(
        r"\b(?:"
        r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
        r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
        r"|\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}"
        r"|\d{4}[\/\-]\d{2}[\/\-]\d{2}"
        r")\b",
        re.IGNORECASE,
    )

    _MONEY_PATTERN = re.compile(
        r"(?:USD|EUR|GBP|INR|AUD|CAD)?\s*[\$€£₹]\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
        r"(?:\s*(?:million|billion|thousand|mn|bn|k))?",
        re.IGNORECASE,
    )

    _PARTY_PATTERN = re.compile(
        r"(?:between|by and between|entered into by)\s+"
        r"([A-Z][A-Za-z\s,\.&]+?(?:Inc|LLC|Ltd|Corp|Co|LLP|LP|PLC|Pvt|GmbH|BV|AG)\.?)"
        r"(?:\s*\(|,|\s+and\b)",
        re.IGNORECASE,
    )

    # Also catch "hereinafter referred to as" pattern
    _PARTY_ALIAS_PATTERN = re.compile(
        r'([A-Z][A-Za-z\s,\.&]+?(?:Inc|LLC|Ltd|Corp|Co|LLP|LP|PLC|Pvt|GmbH|BV|AG)\.?)'
        r'\s*\((?:hereinafter\s+)?(?:referred\s+to\s+as\s+)?["\']?(?:the\s+)?'
        r'(?:Company|Client|Vendor|Employer|Employee|Party\s+[AB]|Contractor|Supplier)["\']?\)',
        re.IGNORECASE,
    )

    _TERMINATION_PATTERN = re.compile(
        r"(?:(?:this\s+)?(?:agreement|contract)\s+(?:may|shall|can|will)\s+(?:be\s+)?terminated"
        r"|termination\s+(?:of\s+(?:this\s+)?(?:agreement|contract)|for\s+(?:cause|convenience))"
        r"|either\s+party\s+may\s+terminate"
        r"|right\s+to\s+terminate"
        r"|may\s+terminate\s+this\s+agreement)",
        re.IGNORECASE,
    )

    def _detect_dates_rule(self, text: str, result: RawExtractionResult):
        for m in self._DATE_PATTERN.finditer(text):
            result.dates.append(RawEntity(
                text=m.group(), label="DATE",
                start=m.start(), end=m.end(), score=0.75
            ))

    def _detect_parties_rule(self, text: str, result: RawExtractionResult):
        # Pattern 1: "between X and Y"
        for m in self._PARTY_PATTERN.finditer(text):
            name = m.group(1).strip().rstrip(",")
            if name and len(name) > 2:
                result.parties.append(RawEntity(
                    text=name, label="PARTY",
                    start=m.start(1), end=m.end(1), score=0.75
                ))
        # Pattern 2: "X (hereinafter referred to as 'Party A')"
        for m in self._PARTY_ALIAS_PATTERN.finditer(text):
            name = m.group(1).strip().rstrip(",")
            if name and len(name) > 2:
                result.parties.append(RawEntity(
                    text=name, label="PARTY",
                    start=m.start(1), end=m.end(1), score=0.78
                ))

    def _detect_money_rule(self, text: str, result: RawExtractionResult):
        for m in self._MONEY_PATTERN.finditer(text):
            val = m.group().strip()
            if any(c.isdigit() for c in val) and len(val) > 2:
                result.monetary.append(RawEntity(
                    text=val, label="MONEY",
                    start=m.start(), end=m.end(), score=0.80
                ))

    def _detect_termination_clauses(self, text: str, result: RawExtractionResult):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        char_offset = 0
        for sentence in sentences:
            if self._TERMINATION_PATTERN.search(sentence):
                result.termination_spans.append(RawEntity(
                    text=sentence.strip(),
                    label="TERM",
                    start=char_offset,
                    end=char_offset + len(sentence),
                    score=0.80,
                ))
            char_offset += len(sentence) + 1

    # ── Chunking ─────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[tuple]:
        """
        Split text into (chunk_text, chunk_start_in_original) pairs.
        Uses overlap to avoid missing entities at boundaries.
        Returns list of (chunk_text, start_offset) tuples.
        """
        chunks = []
        step = self._CHUNK_SIZE - self._CHUNK_OVERLAP
        for i in range(0, len(text), step):
            chunk = text[i: i + self._CHUNK_SIZE]
            chunks.append((chunk, i))
            if i + self._CHUNK_SIZE >= len(text):
                break
        return chunks or [(text, 0)]