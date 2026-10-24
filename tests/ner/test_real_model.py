"""Smoke tests against the real exported model. Skipped unless it has been exported
(python -m leakguard.ner.export); the CI job `ner-model` exports it and runs these."""

import os
import time
from pathlib import Path

import pytest

from leakguard.entities import EntityType as T
from leakguard.ner.model import NameModel

MODEL_DIR = Path(os.environ.get("LEAKGUARD_NER_MODEL_DIR", "models/ner"))
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "model.onnx").is_file(),
                                reason=f"no exported NER model in {MODEL_DIR}")


@pytest.fixture(scope="module")
def model():
    return NameModel.load(MODEL_DIR)


def test_finds_a_full_name(model):
    text = "Hi, I'm Priya Nair, call me back"
    assert [(s.type, s.text(text)) for s in model.detect(text)] == [(T.NAME, "Priya Nair")]


def test_otp_is_not_a_name(model):
    assert model.detect("Your OTP is 482913") == []


def test_long_message_is_fast_enough(model):
    text = ("The parcel was delayed again and nobody called back. " * 60)[:3000]
    started = time.perf_counter()
    model.detect(text + " Regards, Priya Nair")
    assert time.perf_counter() - started < 2.0
