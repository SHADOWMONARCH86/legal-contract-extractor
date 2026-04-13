"""
Unit tests for the text cleaning and post-processing pipeline.
"""

import pytest

from app.services.text_cleaner import TextCleaner
from app.services.postprocessor import PostProcessor
from app.services.ner_service import RawEntity, RawExtractionResult


# ── TextCleaner ───────────────────────────────────────────────────────────────

class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_ligature_replacement(self):
        text = "\ufb01rst class \ufb02ight"
        result = self.cleaner.clean(text)
        assert "fi" in result
        assert "fl" in result

    def test_em_dash_normalisation(self):
        result = self.cleaner.clean("Party A\u2014Party B")
        assert "-" in result

    def test_non_breaking_space(self):
        result = self.cleaner.clean("USD\u00a0150,000")
        assert "\u00a0" not in result

    def test_hyphenated_line_break_fixed(self):
        text = "agree-\nment between parties"
        result = self.cleaner.clean(text)
        assert "agreement" in result

    def test_compact_whitespace(self):
        result = self.cleaner.clean("hello    world")
        assert "hello world" in result

    def test_ocr_usd_fix(self):
        result = self.cleaner.clean("USO 100,000", ocr_applied=True)
        assert "USD" in result

    def test_control_chars_removed(self):
        result = self.cleaner.clean("hello\x00world\x1f")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_empty_string(self):
        result = self.cleaner.clean("")
        assert result == ""


# ── PostProcessor ─────────────────────────────────────────────────────────────

class TestPostProcessor:
    def setup_method(self):
        self.pp = PostProcessor()

    def _make_raw(self, **kwargs) -> RawExtractionResult:
        defaults = dict(dates=[], parties=[], monetary=[], termination_spans=[])
        defaults.update(kwargs)
        return RawExtractionResult(**defaults)

    # Dates
    def test_date_normalisation_iso(self):
        raw = self._make_raw(dates=[RawEntity("2024-01-15", "DATE", 0, 10, 0.9)])
        bundle = self.pp.process(raw, "")
        assert bundle.dates[0].value == "2024-01-15"

    def test_date_normalisation_long_format(self):
        raw = self._make_raw(dates=[RawEntity("January 15, 2024", "DATE", 0, 16, 0.9)])
        bundle = self.pp.process(raw, "")
        assert bundle.dates[0].value == "2024-01-15"

    def test_date_deduplication(self):
        entity = RawEntity("January 15, 2024", "DATE", 0, 16, 0.9)
        raw = self._make_raw(dates=[entity, entity])
        bundle = self.pp.process(raw, "")
        assert len(bundle.dates) == 1

    def test_low_confidence_date_dropped(self):
        raw = self._make_raw(dates=[RawEntity("March 1, 2023", "DATE", 0, 13, 0.2)])
        bundle = self.pp.process(raw, "")
        assert len(bundle.dates) == 0

    # Monetary
    def test_monetary_usd_parsing(self):
        raw = self._make_raw(monetary=[RawEntity("USD 150,000", "MONEY", 0, 11, 0.95)])
        bundle = self.pp.process(raw, "USD 150,000")
        assert bundle.monetary_values[0].amount == 150000.0
        assert bundle.monetary_values[0].currency == "USD"

    def test_monetary_symbol_parsing(self):
        raw = self._make_raw(monetary=[RawEntity("$50,000.00", "MONEY", 0, 10, 0.95)])
        bundle = self.pp.process(raw, "$50,000.00")
        assert bundle.monetary_values[0].amount == 50000.0
        assert bundle.monetary_values[0].currency == "USD"

    def test_monetary_million_multiplier(self):
        raw = self._make_raw(monetary=[RawEntity("USD 2 million", "MONEY", 0, 13, 0.9)])
        bundle = self.pp.process(raw, "USD 2 million")
        assert bundle.monetary_values[0].amount == 2_000_000.0

    # Termination
    def test_termination_for_cause_classification(self):
        text = "Either party may terminate this Agreement for cause upon 30 days written notice."
        raw = self._make_raw(termination_spans=[RawEntity(text, "TERM", 0, len(text), 0.85)])
        bundle = self.pp.process(raw, text)
        clause = bundle.termination_clauses[0]
        assert clause.clause_type == "TERMINATION_FOR_CAUSE"
        assert clause.notice_period == "30 days"

    def test_termination_for_convenience_classification(self):
        text = "Either party may terminate this Agreement for convenience with 60 days written notice."
        raw = self._make_raw(termination_spans=[RawEntity(text, "TERM", 0, len(text), 0.85)])
        bundle = self.pp.process(raw, text)
        assert bundle.termination_clauses[0].clause_type == "TERMINATION_FOR_CONVENIENCE"

    def test_party_name_cleaning(self):
        raw = self._make_raw(parties=[RawEntity("Acme Corporation, Inc. (the \"Vendor\")", "ORG", 0, 37, 0.88)])
        bundle = self.pp.process(raw, "")
        assert "Acme" in bundle.parties[0].name
        assert '"' not in bundle.parties[0].name

    def test_party_deduplication(self):
        entity = RawEntity("Acme Corp", "ORG", 0, 9, 0.88)
        raw = self._make_raw(parties=[entity, entity])
        bundle = self.pp.process(raw, "")
        assert len(bundle.parties) == 1
