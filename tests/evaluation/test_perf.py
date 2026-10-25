import asyncio

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from leakguard.entities import EntityType
from leakguard.evaluation.perf import forwardable_texts, load, machine_info, measure_overhead
from leakguard.evaluation.samples import Sample, parse_marked

TEXTS = ["call 9876543210 please", "hello there", "my PAN is ABCPS1234K"]


def _sample(marked, sid):
    text, spans = parse_marked(marked)
    return Sample(sid, "support", "tpl", text, spans)


def test_forwardable_texts_drops_secrets():
    samples = [_sample("call [[PHONE:9876543210]]", "a"),
               _sample("key [[SECRET:sk_live_abcdefghijklmnop]]", "b")]
    assert [s.spans[0].type for s in samples] == [EntityType.PHONE, EntityType.SECRET]
    assert forwardable_texts(samples) == ["call 9876543210"]


def test_measure_overhead_shapes_and_ttft(guard, mock_app):
    with TestClient(mock_app) as direct:
        plain = measure_overhead(direct, guard, TEXTS, stream=False, repeats=1)
        streamed = measure_overhead(direct, guard, TEXTS, stream=True, repeats=1)
    assert plain["n"] == 3 and "ttft" not in plain
    assert streamed["n"] == 3 and streamed["ttft"]["n"] == 3
    assert streamed["p95_ms"] >= streamed["p50_ms"]


def _run_load(app, seconds=0.3):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await load(client, TEXTS, concurrency=2, seconds=seconds)
    return asyncio.run(go())


def test_load_counts_requests(mock_app):
    result = _run_load(mock_app)
    assert result["requests"] > 0 and result["errors"] == 0
    assert result["msgs_per_s"] > 0 and result["concurrency"] == 2


def test_load_counts_errors():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def broken():
        return JSONResponse({"error": "boom"}, status_code=500)

    result = _run_load(app)
    assert result["requests"] > 0 and result["errors"] == result["requests"]
    assert result["msgs_per_s"] == 0


def test_machine_info_without_docker():
    info = machine_info(docker=False)
    assert info["logical_cores"] >= 1 and info["python"] and info["os"]
    assert info["cpu"] and info["docker"] is None
