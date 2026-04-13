"""
PDFUtils — Smart PDF text extraction.

Decision flow:
1. Attempt direct digital text extraction via pdfplumber.
2. If extracted text is below the "digital threshold" (too little text → likely scanned),
   fall back to OCR via the ocr_pipeline module.
"""

from pathlib import Path
from typing import Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)

# If fewer than this many characters are extracted digitally per page,
# we treat the PDF as scanned and run OCR.
_DIGITAL_TEXT_THRESHOLD = 50  # chars per page


def _is_text_sufficient(text: str, pages: int) -> bool:
    if pages == 0:
        return False
    return (len(text.strip()) / pages) >= _DIGITAL_TEXT_THRESHOLD


class PDFUtils:
    @staticmethod
    def extract_text(pdf_path: Path) -> Tuple[str, int, bool]:
        """
        Extract text from a PDF file.

        Returns:
            (text, page_count, ocr_applied)
        """
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
                pages_text = []
                for page in pdf.pages:
                    pt = page.extract_text()
                    if pt:
                        pages_text.append(pt)
                digital_text = "\n\n".join(pages_text)

            if _is_text_sufficient(digital_text, page_count):
                logger.info(f"Digital PDF — {page_count} pages, {len(digital_text)} chars extracted.")
                return digital_text, page_count, False
            else:
                logger.info(
                    f"Insufficient digital text ({len(digital_text)} chars / {page_count} pages). "
                    "Switching to OCR."
                )
                return PDFUtils._ocr_fallback(pdf_path, page_count)

        except ImportError:
            logger.warning("pdfplumber not installed. Attempting OCR directly.")
            return PDFUtils._ocr_fallback(pdf_path, 0)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError(f"Could not read PDF: {e}") from e

    @staticmethod
    def _ocr_fallback(pdf_path: Path, known_pages: int) -> Tuple[str, int, bool]:
        """Delegate to OCR pipeline."""
        try:
            from ocr.ocr_pipeline import OCRPipeline

            pipeline = OCRPipeline()
            text, page_count = pipeline.extract(pdf_path)
            page_count = page_count or known_pages
            return text, page_count, True
        except Exception as e:
            logger.error(f"OCR pipeline failed: {e}")
            raise ValueError(f"OCR extraction failed: {e}") from e
