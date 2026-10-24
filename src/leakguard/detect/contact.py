"""Detectors for contact and payment details: phone, UPI, email, bank account."""

import re
from importlib.resources import files

from leakguard.detect.ids import IFSC_PATTERN
from leakguard.detect.spans import Span
from leakguard.entities import EntityType


def _load_upi_handles() -> frozenset[str]:
    lines = files("leakguard").joinpath("data/upi_handles.txt").read_text(encoding="utf-8")
    return frozenset(
        line.strip().lower() for line in lines.splitlines()
        if line.strip() and not line.startswith("#")
    )


UPI_HANDLES = _load_upi_handles()

# Indian mobile: optional +91 / 91 / 0 prefix, then 10 digits starting 6-9, allowing a
# single space or hyphen between digits ("98765 43210", "+91-9876543210").
_PHONE = re.compile(r"(?<![\d+])(?:(?:\+91|91)[ -]?|0)?[6-9](?:[ -]?\d){9}(?!\d)")

# Every pattern below starts with a "not preceded by a token character" lookbehind, so
# a failed match is not retried from every position inside a long token (that retrying
# is what makes inputs like "aaaa...a@" take quadratic time — a ReDoS risk).
# The handle may be followed by a sentence-ending "." but not by ".com"-style text.
_UPI = re.compile(
    r"(?<![A-Za-z0-9._@-])[A-Za-z0-9._-]{2,256}@([A-Za-z]{2,64})(?![A-Za-z0-9@-])(?!\.[A-Za-z0-9])")
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
    r"\.[A-Za-z]{2,24}(?![A-Za-z0-9-])"
)

# Bank accounts have no checksum, so a 9-18 digit run only counts with context.
_DIGIT_RUN = re.compile(r"(?<!\d)\d(?:[ -]?\d){8,17}(?!\d)")
_ACCOUNT_CONTEXT = re.compile(
    r"\ba/c\b|\bac\s*no\b|\bacc(?:ount)?s?\b|\bacct\b|\bkhata\b|\bbank\b", re.IGNORECASE)
_CONTEXT_BEFORE = 40  # characters before the number to look for "a/c", "account", ...
_IFSC_NEARBY = 50  # characters around the number to look for an IFSC code


def detect_phone(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), EntityType.PHONE, detector="phone")
            for m in _PHONE.finditer(text)]


def detect_upi(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), EntityType.UPI, detector="upi")
            for m in _UPI.finditer(text) if m.group(1).lower() in UPI_HANDLES]


def detect_email(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), EntityType.EMAIL, detector="email")
            for m in _EMAIL.finditer(text)]


def _has_account_context(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - _CONTEXT_BEFORE):start]
    if _ACCOUNT_CONTEXT.search(before):
        return True
    around = text[max(0, start - _IFSC_NEARBY):end + _IFSC_NEARBY]
    return IFSC_PATTERN.search(around) is not None


def detect_bank_account(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), EntityType.BANK_ACCOUNT, detector="bank_account")
            for m in _DIGIT_RUN.finditer(text) if _has_account_context(text, m.start(), m.end())]
