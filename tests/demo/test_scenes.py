"""The demo's scenes: the fake data must be detected (or left alone) the way the narration
says, using the real detector. Names need the model, so they are checked in test_record.py."""

import re

from examples.demo.scenes import SCENES, scene
from leakguard.detect.checksums import verhoeff_valid
from leakguard.detect.engine import build_detector
from leakguard.entities import EntityType
from leakguard.masking import mask_messages
from leakguard.policy import load_policy

detect = build_detector(None)  # rules only: no name model in the plain CI job


def _text(scene_id: str) -> str:
    return scene(scene_id).messages[-1]["content"]


def test_scene_order_and_ids():
    assert [s.id for s in SCENES] == ["title", "one_line", "core", "stream", "secret",
                                      "lookalike", "log", "results", "outro"]


def test_core_values_are_detected():
    core = scene("core")
    text = _text("core")
    spans = detect(text)
    for value in core.highlight[1:]:  # [0] is the name: model-only
        start = text.index(value)
        end = start + len(value)
        assert any(s.start <= start and end <= s.end for s in spans), value
    aadhaar = next(v for v in core.highlight if re.fullmatch(r"[\d ]{14}", v))
    assert verhoeff_valid(aadhaar.replace(" ", ""))


def test_secret_is_blocked():
    result = mask_messages(list(scene("secret").messages), load_policy(None), detect)
    assert EntityType.SECRET in result.blocked


def test_secret_has_no_provider_prefix():
    for s in SCENES:
        for message in s.messages:
            for prefix in ("sk_", "sk-", "gsk_", "ghp_", "xox", "AKIA"):
                assert prefix not in message["content"], (s.id, prefix)


def test_lookalike_is_untouched():
    assert detect(_text("lookalike")) == []


def test_request_scenes_have_messages():
    for s in SCENES:
        assert (s.expect != "none") == bool(s.messages), s.id
