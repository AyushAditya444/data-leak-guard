"""Runs every detector over a text and merges the results."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from leakguard.detect.address import detect_address
from leakguard.detect.contact import detect_bank_account, detect_email, detect_phone, detect_upi
from leakguard.detect.ids import detect_aadhaar, detect_card, detect_ifsc, detect_pan
from leakguard.detect.name_cues import detect_name_cues
from leakguard.detect.secrets import detect_secrets
from leakguard.detect.spans import Span, merge_spans
from leakguard.entities import EntityType

if TYPE_CHECKING:  # the NER package needs onnxruntime; rules alone must not
    from leakguard.ner.model import Entities

Detector = Callable[[str], list[Span]]
EntityModel = Callable[[str], "Entities"]  # e.g. NameModel.analyse: names + locations

# Names come from a model that has to be loaded first, so they are added by
# build_detector(). Addresses run after these detectors (see detect_rules).
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


def detect_rules(text: str, locations: Sequence[tuple[int, int]] = ()) -> list[Span]:
    """Spans found by the rule detectors, non-overlapping, in order of position.
    `locations` (e.g. NER location tags) are extra place signals for addresses.

    The address detector runs last and may not overlap what the others found (except
    names, which an address absorbs). Merging keeps the higher-priority span of two that
    overlap, so an address that swallowed a phone number would be dropped WHOLE.
    """
    others = [span for detector in RULE_DETECTORS for span in detector(text)]
    stops = merge_spans(s for s in others if s.type is not EntityType.NAME)
    return merge_spans([*others, *detect_address(text, stops=stops, locations=locations)])


def build_detector(name_model: Detector | None = None,
                   ner: EntityModel | None = None) -> Detector:
    """The full detector: rules plus, if given, a name model (e.g. NameModel.detect).
    NAME has the lowest priority, so an email or UPI ID containing a name stays whole.

    `ner` (e.g. NameModel.analyse) replaces `name_model`: one model run gives the names
    and the location tags, which the address detector uses as extra place signals."""
    if ner is not None:
        def detect_with_locations(text: str) -> list[Span]:
            found = ner(text)
            return merge_spans([*detect_rules(text, found.locations), *found.names])

        return detect_with_locations
    if name_model is None:
        return detect_rules

    def detect(text: str) -> list[Span]:
        return merge_spans([*detect_rules(text), *name_model(text)])

    return detect
