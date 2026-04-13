"""
TextCleaner — Normalises raw extracted text before NER.

Handles:
- Noisy OCR artefacts (ligatures, broken hyphens, spurious newlines)
- Unicode normalisation
- Whitespace compaction
- Common OCR character substitutions (0→O, l→1 in numeric contexts)
"""

import re
import unicodedata

from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    # Common OCR ligature replacements
    _LIGATURES = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2013": "-",   # en-dash
        "\u2014": "-",   # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u00a0": " ",   # non-breaking space
        "\u200b": "",    # zero-width space
    }

    def clean(self, text: str, ocr_applied: bool = False) -> str:
        text = self._normalise_unicode(text)
        text = self._replace_ligatures(text)
        text = self._fix_line_breaks(text)
        text = self._compact_whitespace(text)
        if ocr_applied:
            text = self._fix_ocr_artefacts(text)
        text = self._remove_control_chars(text)
        logger.debug(f"Cleaned text length: {len(text)} chars")
        return text.strip()

    def _normalise_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def _replace_ligatures(self, text: str) -> str:
        for char, replacement in self._LIGATURES.items():
            text = text.replace(char, replacement)
        return text

    def _fix_line_breaks(self, text: str) -> str:
        # Remove hyphenation at line breaks (word-\nword → wordword)
        text = re.sub(r"-\n(\w)", r"\1", text)
        # Join lines that are part of the same sentence (don't end with punctuation)
        text = re.sub(r"(?<![.!?:\n])\n(?=[a-z])", " ", text)
        # Preserve paragraph breaks (double newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _compact_whitespace(self, text: str) -> str:
        # Collapse multiple spaces/tabs into one
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    def _fix_ocr_artefacts(self, text: str) -> str:
        """Apply OCR-specific corrections for legal documents."""
        # Fix common OCR mistakes in numeric/currency contexts
        # e.g., "USO 1OO,OOO" → "USD 100,000"
        text = re.sub(r"\bUSO\b", "USD", text)
        text = re.sub(r"\bEUR0\b", "EURO", text)
        # Scanned docs often drop spaces around parentheses
        text = re.sub(r"(\w)\(", r"\1 (", text)
        text = re.sub(r"\)(\w)", r") \1", text)
        # Fix digit-letter confusion in purely numeric sequences
        # e.g., "l00,000" → "100,000" (lowercase-L as digit-1)
        text = re.sub(r"(?<!\w)l(\d)", r"1\1", text)
        text = re.sub(r"(\d)O(\d)", r"\g<1>0\2", text)  # O (letter) as 0 in digit strings
        # Remove isolated single characters that are likely OCR noise
        text = re.sub(r"(?<!\w)[^A-Za-z0-9\s.,$%@#&*()\-+]{1}(?!\w)", " ", text)
        return text

    def _remove_control_chars(self, text: str) -> str:
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
