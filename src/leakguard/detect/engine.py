"""Runs every detector over a text and merges the results."""

from collections.abc import Callable

from leakguard.detect.contact import detect_bank_account, detect_email, detect_phone, detect_upi
from leakguard.detect.ids import detect_aadhaar, detect_card, detect_ifsc, detect_pan
from leakguard.detect.name_cues import detect_name_cues
from leakguard.detect.secrets import detect_secrets
from leakguard.detect.spans import Span, merge_spans

Detector = Callable[[str], list[Span]]

# Phase 5 adds the address detector to this tuple. Names come from a model that has
# to be loaded first, so they are added by build_detector() instead.
RULE_DETECTORS: tuple[Detector, ...] = (
    detect_aadhaar,
    detect_pan,
    detect_card,
    detect_ifsc,
    detect_phone,
    detect_upi,
    detect_email,
    detect_bank_account,
    detect_secrets,
    detect_name_cues,  # lowercase / ALL CAPS names after cues like "mera naam"
)


def detect_rules(text: str) -> list[Span]:
    """Spans found by the rule detectors, non-overlapping, in order of position."""
    return merge_spans(span for detector in RULE_DETECTORS for span in detector(text))


def build_detector(name_model: Detector | None = None) -> Detector:
    """The full detector: rules plus, if given, a name model (e.g. NameModel.detect).
    NAME has the lowest priority, so an email or UPI ID containing a name stays whole."""
    if name_model is None:
        return detect_rules

    def detect(text: str) -> list[Span]:
        return merge_spans([*detect_rules(text), *name_model(text)])

    return detect
