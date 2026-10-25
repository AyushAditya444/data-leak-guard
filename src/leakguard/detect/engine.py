"""Runs every detector over a text and merges the results."""

from bisect import bisect_left
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from leakguard.detect.address import DEFAULT_GAP, detect_address
from leakguard.detect.contact import detect_bank_account, detect_email, detect_phone, detect_upi
from leakguard.detect.ids import detect_aadhaar, detect_card, detect_ifsc, detect_pan
from leakguard.detect.name_cues import detect_name_cues
from leakguard.detect.secrets import detect_secrets
from leakguard.detect.spans import Intervals, Span, merge_spans
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


def _merge(spans: Iterable[Span]) -> list[Span]:
    """merge_spans, after letting each ADDRESS take in the names it overlaps.

    A surname can also be a village name ("Sharma"), so an address may start inside a
    name; merging alone would keep the ADDRESS and drop the whole NAME, leaking the first
    name. A name that also overlaps another value (a name inside an email) is left alone:
    widening into the email would get the address dropped instead.
    """
    spans = list(spans)
    names = sorted((s for s in spans if s.type is EntityType.NAME), key=lambda s: s.start)
    others = Intervals((s.start, s.end) for s in spans
                       if s.type not in (EntityType.NAME, EntityType.ADDRESS))
    if not names or len(names) == len(spans):
        return merge_spans(spans)
    starts = [n.start for n in names]
    longest = max(n.end - n.start for n in names)
    out = []
    for span in spans:
        if span.type is EntityType.ADDRESS:
            lo, hi = bisect_left(starts, span.start - longest), bisect_left(starts, span.end)
            for name in names[lo:hi]:
                if name.overlaps(span) and not others.overlaps(name.start, name.end):
                    span = replace(span, start=min(span.start, name.start),
                                   end=max(span.end, name.end))
        out.append(span)
    return merge_spans(out)


def detect_rules(text: str, locations: Sequence[tuple[int, int]] = (),
                 address_gap: int = DEFAULT_GAP) -> list[Span]:
    """Spans found by the rule detectors, non-overlapping, in order of position.
    `locations` (e.g. NER location tags) are extra place signals for addresses;
    `address_gap` exists so the dev-split tuning can try other values.

    The address detector runs last and may not overlap what the others found (except
    names, which an address absorbs). Merging keeps the higher-priority span of two that
    overlap, so an address that swallowed a phone number would be dropped WHOLE.
    """
    others = [span for detector in RULE_DETECTORS for span in detector(text)]
    stops = merge_spans(s for s in others if s.type is not EntityType.NAME)
    addresses = detect_address(text, stops=stops, locations=locations, gap=address_gap)
    return _merge([*others, *addresses])


def build_detector(name_model: Detector | None = None, ner: EntityModel | None = None,
                   address_gap: int = DEFAULT_GAP) -> Detector:
    """The full detector: rules plus, if given, a name model (e.g. NameModel.detect).
    NAME has the lowest priority, so an email or UPI ID containing a name stays whole.

    `ner` (e.g. NameModel.analyse) replaces `name_model`: one model run gives the names
    and the location tags, which the address detector uses as extra place signals."""
    if ner is not None:
        def detect_with_locations(text: str) -> list[Span]:
            found = ner(text)
            rules = detect_rules(text, found.locations, address_gap)
            return _merge([*rules, *found.names])

        return detect_with_locations
    if name_model is None:
        if address_gap == DEFAULT_GAP:
            return detect_rules
        return lambda text: detect_rules(text, address_gap=address_gap)

    def detect(text: str) -> list[Span]:
        return _merge([*detect_rules(text, address_gap=address_gap), *name_model(text)])

    return detect
