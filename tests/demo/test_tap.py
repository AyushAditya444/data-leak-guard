"""The demo's wire tap records what the guard sent upstream and what came back, and never
the caller's API key."""

import json

import httpx
import pytest

from examples.demo.tap import WireTap
from leakguard.mock_llm import create_mock_app

URL = "http://upstream/v1/chat/completions"
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi [PHONE_1]"}]}


def _tap() -> WireTap:
    return WireTap(httpx.ASGITransport(create_mock_app(chunk_size=3)))


@pytest.mark.anyio
async def test_tap_records_messages_and_stream_chunks():
    tap = _tap()
    async with (httpx.AsyncClient(transport=tap) as client,
                client.stream("POST", URL, json={**BODY, "stream": True}) as resp):
        async for _ in resp.aiter_bytes():
            pass
    [record] = tap.records
    assert record["messages"] == BODY["messages"]
    assert record["status"] == 200
    times = [t for t, _ in record["chunks"]]
    assert len(times) >= 3 and times == sorted(times)
    assert "".join(text for _, text in record["chunks"]) == "You said: hi [PHONE_1]"


@pytest.mark.anyio
async def test_tap_records_non_stream_reply_content():
    tap = _tap()
    async with httpx.AsyncClient(transport=tap) as client:
        await client.post(URL, json=BODY)
    [record] = tap.records
    assert [text for _, text in record["chunks"]] == ["You said: hi [PHONE_1]"]


@pytest.mark.anyio
async def test_tap_never_records_authorization():
    tap = _tap()
    async with httpx.AsyncClient(transport=tap) as client:
        await client.post(URL, json=BODY, headers={"Authorization": "Bearer test-key-123456789"})
    assert "test-key-123456789" not in json.dumps(tap.records)


@pytest.mark.anyio
async def test_tap_records_error_status():
    inner = httpx.MockTransport(lambda request: httpx.Response(503, text="busy"))
    tap = WireTap(inner)
    async with httpx.AsyncClient(transport=tap) as client:
        await client.post(URL, json=BODY)
    [record] = tap.records
    assert record["status"] == 503
    assert record["chunks"][0][1] == "busy"


@pytest.mark.anyio
async def test_tap_clear():
    tap = _tap()
    async with httpx.AsyncClient(transport=tap) as client:
        await client.post(URL, json=BODY)
    tap.clear()
    assert tap.records == []
