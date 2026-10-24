"""The guard with name detection: masking names, failing closed, not blocking the loop."""

import asyncio
import json
import logging
import time

import httpx
import pytest

from leakguard.app import create_app
from leakguard.config import Settings
from leakguard.detect.spans import Span
from leakguard.entities import EntityType as T
from leakguard.ner.model import NameModelError

CHAT = "/v1/chat/completions"


def _req(content):
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


def _fake_names(text):
    start = text.find("Rahul")
    return [] if start == -1 else [Span(start, start + 5, T.NAME, 0.9, "ner")]


def _echo(seen):
    def handler(request):
        seen["sent"] = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": "Hello " + seen["sent"].split()[-1]}}]})
    return handler


def test_name_is_masked_and_restored(make_guard):
    seen = {}
    guard = make_guard(transport=httpx.MockTransport(_echo(seen)), name_model=_fake_names)
    r = guard.post(CHAT, json=_req("I am Rahul"))
    assert seen["sent"] == "I am [NAME_1]"
    assert r.json()["choices"][0]["message"]["content"] == "Hello Rahul"


def test_name_model_error_fails_closed(make_guard):
    def broken(text):
        raise RuntimeError("onnxruntime exploded")

    calls = []
    guard = make_guard(transport=httpx.MockTransport(lambda req: calls.append(1)),
                       name_model=broken)
    r = guard.post(CHAT, json=_req("I am Rahul"))
    assert r.status_code == 500 and r.json()["error"]["code"] == "detection_failed"
    assert calls == []


def test_incomplete_model_dir_fails_at_startup(tmp_path):
    settings = Settings(ner_enabled=True, ner_model_dir=str(tmp_path))
    with pytest.raises(NameModelError, match="LEAKGUARD_NER=off"):
        create_app(settings)


def test_ner_off_starts_without_model(tmp_path, caplog):
    settings = Settings(ner_enabled=False, ner_model_dir=str(tmp_path / "nothing-here"))
    with caplog.at_level(logging.WARNING, logger="leakguard"):
        create_app(settings)
    assert any("ner_disabled" in record.message for record in caplog.records)


@pytest.mark.anyio
async def test_slow_detection_does_not_block_other_requests():
    def slow_names(text):
        time.sleep(0.5)  # stands in for a long CPU-bound inference
        return []

    app = create_app(Settings(upstream_base_url="http://mock/v1", ner_enabled=False),
                     httpx.MockTransport(_echo({})), name_model=slow_names)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://guard") as client:
            # The clock starts with the chat request: if detection blocked the event loop,
            # the 0.5 s would land somewhere before /healthz could answer.
            started = time.perf_counter()
            chat = asyncio.create_task(client.post(CHAT, json=_req("hello there")))
            await asyncio.sleep(0.05)  # let the chat request reach detection
            health = await client.get("/healthz")
            elapsed = time.perf_counter() - started
            assert (await chat).status_code == 200
    assert health.status_code == 200 and elapsed < 0.25
