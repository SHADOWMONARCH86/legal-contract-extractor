"""
PostProcessor — Converts raw NER output into validated, structured schemas.

Responsibilities:
- Deduplicate overlapping entities
- Normalise date formats to ISO-8601
- Validate and parse currency amounts
- Classify termination clause types
- Assign confidence scores
- Attach contextual snippets to monetary entities
"""

import re
from datetime import datetime
from typing import List, Optional, Tuple

from app.schemas.extraction import (
    DateEntity,
    EntityBundle,
    MonetaryEntity,
    PartyEntity,
    TerminationClause,
)
from app.services.ner_service import RawEntity, RawExtractionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Context window (chars) to extract around a monetary value
_CONTEXT_WINDOW = 80

# Minimum confidence threshold below which entities are dropped
_MIN_CONFIDENCE = 0.40

# FIX 2: Minimum amount to be considered a real monetary value
# Filters out bare integers like "1", "3", "4" that spaCy tags as CARDINAL
_MIN_MONETARY_AMOUNT = 100.0

# FIX 2: Words that look like numbers but are not monetary values
_MONETARY_NOISE_WORDS = {
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "one", "zero", "first", "second", "third",
}

# FIX 4: Patterns that look like dates but are actually durations — skip these
_DURATION_PATTERN = re.compile(
    r"^\d+-(?:day|month|week|year|hour)s?$"      # e.g. "30-day", "6-month"
    r"|^\d+\s+(?:days?|months?|weeks?|years?)$",  # e.g. "30 days", "6 months"
    re.IGNORECASE,
)


class PostProcessor:
    def process(self, raw: RawExtractionResult, full_text: str) -> EntityBundle:
        return EntityBundle(
            dates=self._process_dates(raw.dates),
            parties=self._process_parties(raw.parties),
            monetary_values=self._process_monetary(raw.monetary, full_text),
            termination_clauses=self._process_termination(raw.termination_spans),
        )

    # ── Dates ────────────────────────────────────────────────────────────────

    def _process_dates(self, entities: List[RawEntity]) -> List[DateEntity]:
        seen: set = set()
        results: List[DateEntity] = []
        for ent in entities:
            if ent.score < _MIN_CONFIDENCE:
                continue

            # FIX 4: Skip durations like "30-day", "6-month"
            if _DURATION_PATTERN.match(ent.text.strip()):
                logger.debug(f"Skipping duration pattern as date: '{ent.text}'")
                continue

            norm = self._normalise_date(ent.text)
            key = norm or ent.text.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(
                DateEntity(
                    value=norm or ent.text,
                    raw_text=ent.text,
                    label=self._classify_date_label(ent.text),
                    confidence=round(ent.score, 4),
                )
            )
        return results

    # FIX 1: Added the missing date formats for "12 March 2024" style dates
    _DATE_FORMATS = [
        "%B %d, %Y",    # January 15, 2024
        "%B %d %Y",     # January 15 2024
        "%d %B %Y",     # 15 January 2024  ← NEW
        "%d %b %Y",     # 15 Jan 2024       ← NEW
        "%d %B, %Y",    # 15 January, 2024  ← NEW
        "%d/%m/%Y",     # 15/01/2024
        "%m/%d/%Y",     # 01/15/2024
        "%Y-%m-%d",     # 2024-01-15
        "%d-%m-%Y",     # 15-01-2024
        "%d.%m.%Y",     # 15.01.2024
        "%B %Y",        # January 2024
        "%b %d, %Y",    # Jan 15, 2024
        "%b %Y",        # Jan 2024          ← NEW
    ]

    def _normalise_date(self, raw: str) -> Optional[str]:
        cleaned = raw.strip().rstrip(".")
        for fmt in self._DATE_FORMATS:
            try:
                dt = datetime.strptime(cleaned, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    _DATE_CONTEXT_KEYWORDS = {
        "effective": "EFFECTIVE_DATE",
        "commence": "COMMENCEMENT_DATE",
        "expir": "EXPIRY_DATE",
        "terminat": "TERMINATION_DATE",
        "execut": "EXECUTION_DATE",
        "sign": "SIGNING_DATE",
        "deliver": "DELIVERY_DATE",
    }

    def _classify_date_label(self, raw: str) -> str:
        lower = raw.lower()
        for keyword, label in self._DATE_CONTEXT_KEYWORDS.items():
            if keyword in lower:
                return label
        return "CONTRACT_DATE"

    # ── Parties ──────────────────────────────────────────────────────────────

    _ROLE_KEYWORDS = {
        "vendor": "VENDOR",
        "supplier": "VENDOR",
        "service provider": "VENDOR",
        "client": "CLIENT",
        "customer": "CLIENT",
        "buyer": "CLIENT",
        "employer": "EMPLOYER",
        "employee": "EMPLOYEE",
        "contractor": "CONTRACTOR",
        "lessor": "LESSOR",
        "lessee": "LESSEE",
        "licensor": "LICENSOR",
        "licensee": "LICENSEE",
    }

    def _process_parties(self, entities: List[RawEntity]) -> List[PartyEntity]:
        seen: set = set()
        results: List[PartyEntity] = []
        for ent in entities:
            if ent.score < _MIN_CONFIDENCE:
                continue
            name = self._clean_party_name(ent.text)
            if not name or name.lower() in seen:
                continue
            # Skip clearly bad party names (contain newlines, too short, purely numeric)
            if "\n" in name or len(name.strip()) < 3 or name.strip().isdigit():
                continue
            seen.add(name.lower())
            results.append(
                PartyEntity(
                    name=name,
                    role=self._infer_role(ent.text),
                    raw_text=ent.text,
                    confidence=round(ent.score, 4),
                )
            )
        return results

    def _clean_party_name(self, raw: str) -> str:
        # Remove common trailing noise from NER output
        name = re.sub(r"\s*\(.*?\)\s*$", "", raw).strip()
        name = re.sub(
            r",?\s*(Inc|LLC|Ltd|Corp|Co|LLP|LP|PLC|GmbH|BV|AG|Pvt\.?\s*Ltd|Pvt)\.?\s*$",
            r" \1.",
            name,
            flags=re.IGNORECASE,
        )
        # Remove trailing newlines and section numbers like ".\n2"
        name = re.sub(r"\s*[\.\n]+\s*\d+\s*$", "", name)
        return name.strip(" ,;")

    def _infer_role(self, raw: str) -> Optional[str]:
        lower = raw.lower()
        for keyword, role in self._ROLE_KEYWORDS.items():
            if keyword in lower:
                return role
        return None

    # ── Monetary ─────────────────────────────────────────────────────────────

    # FIX 3: Added $ symbol lookup BEFORE stripping it, so currency is captured
    _CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
    _CURRENCY_CODES = re.compile(r"\b(USD|EUR|GBP|INR|AUD|CAD|JPY|CHF|CNY)\b", re.IGNORECASE)
    _MULTIPLIERS = {
        "million": 1_000_000, "mn": 1_000_000,
        "billion": 1_000_000_000, "bn": 1_000_000_000,
        "thousand": 1_000, "k": 1_000,
    }

    def _process_monetary(self, entities: List[RawEntity], full_text: str) -> List[MonetaryEntity]:
        seen: set = set()
        results: List[MonetaryEntity] = []
        for ent in entities:
            if ent.score < _MIN_CONFIDENCE:
                continue
            raw = ent.text.strip()
            if raw in seen:
                continue

            # FIX 2: Skip plain noise words like "two", "three"
            if raw.lower() in _MONETARY_NOISE_WORDS:
                logger.debug(f"Skipping noise word as monetary: '{raw}'")
                continue

            seen.add(raw)
            amount, currency = self._parse_monetary(raw)

            # Check tight context (5 chars) immediately around the entity
            # for a currency symbol — e.g. "$25,000" where spaCy gives us "25,000"
            tight_start = max(0, ent.start - 5)
            tight_snippet = full_text[tight_start:ent.end + 5]
            for symbol, code in self._CURRENCY_SYMBOLS.items():
                if symbol in tight_snippet and currency is None:
                    currency = code
                    break
            if currency is None:
                tight_code = self._CURRENCY_CODES.search(tight_snippet)
                if tight_code:
                    currency = tight_code.group(1).upper()

            # Skip bare small integers with no currency — clause numbers like "3.", "4."
            if amount is not None:
                if amount <= 0:
                    continue
                if amount < _MIN_MONETARY_AMOUNT and currency is None:
                    logger.debug(f"Skipping low-value non-currency entity: '{raw}'")
                    continue

            context = self._extract_context(full_text, ent.start, ent.end)
            results.append(
                MonetaryEntity(
                    amount=amount,
                    currency=currency,
                    raw_text=raw,
                    context=context,
                    confidence=round(ent.score, 4),
                )
            )
        return results

    def _parse_monetary(self, raw: str) -> Tuple[Optional[float], Optional[str]]:
        currency: Optional[str] = None
        text = raw.strip()

        # FIX 3: Detect currency symbol FIRST before any stripping
        for symbol, code in self._CURRENCY_SYMBOLS.items():
            if symbol in text:
                currency = code          # capture currency before removing symbol
                text = text.replace(symbol, "")
                break  # only match first symbol found

        # Detect currency code (may override symbol if more specific)
        code_match = self._CURRENCY_CODES.search(text)
        if code_match:
            currency = code_match.group(1).upper()
            text = self._CURRENCY_CODES.sub("", text)

        # Detect multiplier
        multiplier = 1
        for word, mult in self._MULTIPLIERS.items():
            if re.search(rf"\b{word}\b", text, re.IGNORECASE):
                multiplier = mult
                text = re.sub(rf"\b{word}\b", "", text, flags=re.IGNORECASE)

        # Extract numeric part
        numeric = re.sub(r"[^\d.]", "", text.replace(",", ""))
        try:
            amount = float(numeric) * multiplier if numeric else None
        except ValueError:
            amount = None

        return amount, currency

    def _extract_context(self, text: str, start: int, end: int) -> Optional[str]:
        ctx_start = max(0, start - _CONTEXT_WINDOW)
        ctx_end = min(len(text), end + _CONTEXT_WINDOW)
        snippet = text[ctx_start:ctx_end].replace("\n", " ").strip()
        return snippet if snippet else None

    # ── Termination Clauses ──────────────────────────────────────────────────

    _CLAUSE_PATTERNS = {
        "TERMINATION_FOR_CAUSE": re.compile(
            r"terminat\w*\s+for\s+cause|material\s+breach|default", re.IGNORECASE
        ),
        "TERMINATION_FOR_CONVENIENCE": re.compile(
            r"terminat\w*\s+for\s+convenience|without\s+cause|at\s+(any\s+)?will|written\s+notice",
            re.IGNORECASE,
        ),
        "TERMINATION_BY_MUTUAL_AGREEMENT": re.compile(
            r"mutual\s+(agreement|consent)|by\s+agreement\s+of\s+(both|all)\s+parties", re.IGNORECASE
        ),
        "AUTOMATIC_TERMINATION": re.compile(
            r"automatically\s+terminat|shall\s+terminat\w*\s+(?:upon|immediately)", re.IGNORECASE
        ),
        "INSOLVENCY_TERMINATION": re.compile(
            r"insolvency|bankrupt|liquidat|receivership|winding.up", re.IGNORECASE
        ),
    }

    # FIX: Also catch "X-day written notice" pattern for notice period extraction
    _NOTICE_PATTERN = re.compile(
        r"(\d+)[\s\-]*(days?|months?|weeks?)\s*(?:written|prior)?\s*notice",
        re.IGNORECASE,
    )

    def _process_termination(self, entities: List[RawEntity]) -> List[TerminationClause]:
        seen: set = set()
        results: List[TerminationClause] = []
        for ent in entities:
            if ent.score < _MIN_CONFIDENCE:
                continue
            text = ent.text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            clause_type = self._classify_termination(text)
            notice = self._extract_notice_period(text)
            results.append(
                TerminationClause(
                    text=text,
                    clause_type=clause_type,
                    notice_period=notice,
                    confidence=round(ent.score, 4),
                )
            )
        return results

    def _classify_termination(self, text: str) -> str:
        for label, pattern in self._CLAUSE_PATTERNS.items():
            if pattern.search(text):
                return label
        return "TERMINATION_GENERAL"

    def _extract_notice_period(self, text: str) -> Optional[str]:
        m = self._NOTICE_PATTERN.search(text)
        if m:
            return f"{m.group(1)} {m.group(2).lower()}"
        return None