"""Detectors for identity and financial numbers with a fixed, checkable format."""

import re
from collections.abc import Callable

from leakguard.detect.checksums import luhn_valid, verhoeff_valid
from leakguard.detect.spans import Span
from leakguard.entities import EntityType

# (?<!\d) / (?!\d) mean "not touching another digit", so we never match a slice of a
# longer number. \1 makes the separator consistent: "2345 6789 0124" or
# "2345-6789-0124" or "234567890124". First digit 2-9: Aadhaar never starts with 0 or 1.
#
# The extra lookarounds stop a match inside a longer *grouped* number: not preceded by
# "dddd " and not followed by " dddd" (a fourth group). Without them, the first 12 digits
# of a card like "4023-5925-6341-6859" can pass Verhoeff (1 in 10), and since AADHAAR
# outranks CARD, the card's last four digits would leak.
_AADHAAR = re.compile(
    r"(?<!\d)(?<!\d{4}[ -])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?!\d)(?!\1\d{4}(?!\d))")
# Words that mark a number as a reference (order, invoice...) rather than an Aadhaar.
# "id" is deliberately left out: "my id is ..." often *is* an Aadhaar.
_REFERENCE_CONTEXT = re.compile(
    r"\b(?:order|ord|invoice|inv|ticket|txn|transaction|ref|reference|tracking|awb)\b",
    re.IGNORECASE)
_AADHAAR_CONTEXT = re.compile(r"\b(?:aadhaa?r|adhaa?r|uid|uidai)\b", re.IGNORECASE)
_REFERENCE_BEFORE = 20  # characters
_AADHAAR_KEYWORD_BEFORE = 40  # characters

# PAN: 5 letters, 4 digits, 1 letter. The 4th letter is the holder type:
# P person, C company, H HUF, F firm, A AOP, T trust, B BOI, L local authority,
# J artificial juridical person, G government.
_PAN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{3}[ABCFGHJLPT][A-Za-z]\d{4}[A-Za-z](?![A-Za-z0-9])",
                  re.IGNORECASE)

# Cards: 13-19 digits, optionally grouped with single spaces or hyphens.
_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
_CARD_CONTEXT = re.compile(
    r"\b(?:card|credit|debit|visa|master\s?card|rupay|amex|diners|maestro)\b", re.IGNORECASE)
_CARD_CONTEXT_BEFORE = 30  # characters

# IFSC: 4-letter bank code, a literal 0 (reserved), 6-character branch code.
IFSC_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{4}0[A-Za-z0-9]{6}(?![A-Za-z0-9])")


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _find(
    pattern: re.Pattern, text: str, entity: EntityType, detector: str,
    valid: Callable[[str], bool] | None = None,
) -> list[Span]:
    return [
        Span(m.start(), m.end(), entity, detector=detector)
        for m in pattern.finditer(text)
        if valid is None or valid(m.group())
    ]


def _is_reference_number(text: str, start: int) -> bool:
    """'order 234567890124' is an order number even if it passes Verhoeff (1 in 10 do),
    unless an Aadhaar keyword says otherwise."""
    if not _REFERENCE_CONTEXT.search(text[max(0, start - _REFERENCE_BEFORE):start]):
        return False
    return not _AADHAAR_CONTEXT.search(text[max(0, start - _AADHAAR_KEYWORD_BEFORE):start])


def detect_aadhaar(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), EntityType.AADHAAR, detector="aadhaar")
            for m in _AADHAAR.finditer(text)
            if verhoeff_valid(_digits(m.group())) and not _is_reference_number(text, m.start())]


def detect_pan(text: str) -> list[Span]:
    return _find(_PAN, text, EntityType.PAN, "pan")


def detect_card(text: str) -> list[Span]:
    """Luhn-valid 13-19 digit numbers. Indian cards are almost always 15 (Amex) or 16
    digits; other lengths only count with a card word just before them, because about
    1 in 10 random 14-digit ticket or order numbers also pass Luhn."""
    spans = []
    for m in _CARD.finditer(text):
        digits = _digits(m.group())
        if not luhn_valid(digits):
            continue
        common_length = len(digits) in (15, 16)
        has_context = _CARD_CONTEXT.search(text[max(0, m.start() - _CARD_CONTEXT_BEFORE):m.start()])
        if common_length or has_context:
            spans.append(Span(m.start(), m.end(), EntityType.CARD, detector="card"))
    return spans


def detect_ifsc(text: str) -> list[Span]:
    return _find(IFSC_PATTERN, text, EntityType.IFSC, "ifsc")
