"""The demo recorder, run against the mock: the CI smoke test that keeps the demo working.
It runs the real guard, so it needs the exported name model (CI's ner-model job has it)."""

import json
import os
from pathlib import Path

import httpx
import pytest

from examples.demo.record import record_session
from examples.demo.scenes import SCENES
from examples.demo.session import verify_session

MODEL_DIR = Path(os.environ.get("LEAKGUARD_NER_MODEL_DIR", "models/ner"))
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "model.onnx").is_file(),
                                reason=f"no exported NER model in {MODEL_DIR}")


def test_mock_session_passes_verification(tmp_path):
    out = tmp_path / "s.json"
    session = record_session("mock", out)
    assert verify_session(session) == []
    assert json.loads(out.read_text(encoding="utf-8")) == session
    assert session["meta"]["upstream"] == "mock"
    assert list(session["scenes"]) == [s.id for s in SCENES]


def test_record_refuses_to_write_key(tmp_path):
    # A provider whose error message echoes the request's Authorization header.
    def echo_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": request.headers["authorization"]}})

    out = tmp_path / "s.json"
    with pytest.raises(RuntimeError, match="API key"):
        record_session("mock", out, inner=httpx.MockTransport(echo_auth),
                       api_key="fake-key-0123456789abcdef")
    assert not out.exists()
