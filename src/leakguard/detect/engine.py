"""Runs every detector over a text and merges the results."""

from collections.abc import Callable

from leakguard.detect.contact import detect_bank_account, detect_email, detect_phone, detect_upi
from leakguard.detect.ids import detect_aadhaar, detect_card, detect_ifsc, detect_pan
from leakguard.detect.secrets import detect_secrets
from leakguard.detect.spans import Span, merge_spans

Detector = Callable[[str], list[Span]]

# Phase 4 adds the name model and Phase 5 the address detector to this tuple.
DETECTORS: tuple[Detector, ...] = (
    detect_aadhaar,
    detect_pan,
    detect_card,
    detect_ifsc,
    detect_phone,
    detect_upi,
    detect_email,
    detect_bank_account,
    detect_secrets,
)


def detect_all(text: str) -> list[Span]:
    """All sensitive spans in `text`, non-overlapping, in order of position."""
    return merge_spans(span for detector in DETECTORS for span in detector(text))
