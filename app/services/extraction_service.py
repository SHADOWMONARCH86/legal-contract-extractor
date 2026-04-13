"""
ExtractionService — Orchestrates the full pipeline:
PDF → OCR (if needed) → Text Cleaning → NER → Post-processing → JSON
"""

import time
import uuid
from pathlib import Path

from app.schemas.extraction import (
    EntityBundle,
    ExtractionResponse,
    ProcessingMetadata,
)
from app.services.ner_service import NERService
from app.services.postprocessor import PostProcessor
from app.services.text_cleaner import TextCleaner
from app.utils.logger import get_logger
from app.utils.pdf_utils import PDFUtils

logger = get_logger(__name__)


class ExtractionService:
    """
    Stateless orchestrator. A new instance is created per request,
    but NERService is a singleton loaded once at startup.
    """

    def __init__(self):
        self.ner = NERService.get_instance()
        self.cleaner = TextCleaner()
        self.postprocessor = PostProcessor()

    def process(self, pdf_path: Path, original_filename: str = "document.pdf") -> ExtractionResponse:
        t_start = time.perf_counter()
        document_id = str(uuid.uuid4())

        # ── Step 1: Detect whether OCR is needed ────────────────────────────
        raw_text, page_count, ocr_applied = PDFUtils.extract_text(pdf_path)

        if not raw_text.strip():
            raise ValueError(
                "Could not extract any text from the document. "
                "The PDF may be corrupt, password-protected, or fully image-based without OCR support."
            )

        logger.info(
            f"[{document_id}] Extracted {len(raw_text)} chars from {page_count} pages "
            f"(OCR={'yes' if ocr_applied else 'no'})"
        )

        # ── Step 2: Clean noisy text ─────────────────────────────────────────
        clean_text = self.cleaner.clean(raw_text, ocr_applied=ocr_applied)

        # ── Step 3: Run NER ──────────────────────────────────────────────────
        raw_entities = self.ner.extract(clean_text)

        # ── Step 4: Post-process and validate ────────────────────────────────
        entity_bundle: EntityBundle = self.postprocessor.process(raw_entities, clean_text)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return ExtractionResponse(
            document_id=document_id,
            metadata=ProcessingMetadata(
                filename=original_filename,
                file_size_bytes=pdf_path.stat().st_size,
                ocr_applied=ocr_applied,
                pages_processed=page_count,
                processing_time_ms=round(elapsed_ms, 2),
                model_version=self.ner.model_version,
            ),
            entities=entity_bundle,
        )
