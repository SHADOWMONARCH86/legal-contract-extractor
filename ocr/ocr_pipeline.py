"""
OCRPipeline — Converts scanned PDF pages to text using Tesseract OCR.

Pipeline per page:
  1. Render PDF page to high-res PIL image (pdf2image / poppler)
  2. Pre-process image (deskew, denoise, binarise) for better OCR accuracy
  3. Run Tesseract via pytesseract
  4. Aggregate page texts
"""

import os
from pathlib import Path
from typing import Tuple

from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class OCRPipeline:
    def __init__(self):
        self._configure_tesseract()

    def _configure_tesseract(self):
        if settings.tesseract_cmd:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            except ImportError:
                pass

    def extract(self, pdf_path: Path) -> Tuple[str, int]:
        """
        Extract text from a scanned PDF via OCR.

        Returns:
            (full_text, page_count)
        """
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise RuntimeError(
                "pdf2image is not installed. Install it with: pip install pdf2image\n"
                "Also ensure poppler is installed on your system."
            ) from e

        try:
            import pytesseract
        except ImportError as e:
            raise RuntimeError(
                "pytesseract is not installed. Install it with: pip install pytesseract\n"
                "Also ensure Tesseract OCR engine is installed on your system."
            ) from e

        logger.info(f"Running OCR on {pdf_path.name} at {settings.ocr_dpi} DPI...")

        images = convert_from_path(
            str(pdf_path),
            dpi=settings.ocr_dpi,
            fmt="jpeg",
        )
        page_count = len(images)
        page_texts = []

        for i, image in enumerate(images, start=1):
            logger.debug(f"  OCR processing page {i}/{page_count}...")
            processed = self._preprocess_image(image)
            text = pytesseract.image_to_string(
                processed,
                lang=settings.ocr_language,
                config="--psm 6",  # Assume a single uniform block of text
            )
            page_texts.append(text)

        full_text = "\n\n".join(page_texts)
        logger.info(f"OCR complete: {page_count} pages, {len(full_text)} chars extracted.")
        return full_text, page_count

    @staticmethod
    def _preprocess_image(image):
        """
        Apply image preprocessing to improve OCR accuracy:
        - Convert to grayscale
        - Apply adaptive thresholding (binarisation)
        - Basic noise removal

        Requires Pillow and optionally OpenCV for advanced ops.
        """
        try:
            import cv2
            import numpy as np

            img = np.array(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Adaptive thresholding — handles uneven lighting in scans
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=31,
                C=10,
            )
            # Mild denoising
            denoised = cv2.fastNlMeansDenoising(binary, h=10)
            from PIL import Image as PILImage
            return PILImage.fromarray(denoised)

        except ImportError:
            # OpenCV not available — return image as-is (still functional)
            logger.debug("OpenCV not available; skipping advanced image preprocessing.")
            return image
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}. Using raw image.")
            return image
