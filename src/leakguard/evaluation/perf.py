"""Client-side performance measurements: what the guard adds to a request, and how many
requests it serves per second.

The functions take ready-made httpx clients, so the same code measures real processes
(perf_run.py) and in-process apps (the tests). Also a tiny `load` command, run inside a
container next to the guard:
    python -m leakguard.evaluation.perf load --url http://guard:8080 --concurrency 8 --seconds 30
"""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence

import httpx

from leakguard.entities import EntityType
from leakguard.evaluation.report import _percentile
from leakguard.evaluation.samples import Sample

CHAT = "/v1/chat/completions"


def forwardable_texts(samples: Iterable[Sample]) -> list[str]:
    """Messages the guard would forward: those with a secret are blocked (a fast 400), which
    would make the guard look quicker than it is."""
    return [s.text for s in samples if not any(g.type is EntityType.SECRET for g in s.spans)]


def _body(text: str, stream: bool) -> dict:
    return {"model": "perf", "messages": [{"role": "user", "content": text}], "stream": stream}


def _first_content(line: str) -> bool:
    """Is this SSE line a chunk carrying reply text?"""
    if not line.startswith("data: ") or line == "data: [DONE]":
        return False
    choices = json.loads(line[6:]).get("choices") or [{}]
    return bool((choices[0].get("delta") or {}).get("content"))


def _timed(client: httpx.Client, text: str, stream: bool) -> tuple[float, float | None]:
    """(total ms, ms to the first reply text or None). Raises on a non-200 reply."""
    started = time.perf_counter()
    if not stream:
        client.post(CHAT, json=_body(text, False)).raise_for_status()
        return (time.perf_counter() - started) * 1000, None
    first = None
    with client.stream("POST", CHAT, json=_body(text, True)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if first is None and _first_content(line):
                first = (time.perf_counter() - started) * 1000
    return (time.perf_counter() - started) * 1000, first


def _summary(values: list[float]) -> dict:
    return {"n": len(values), "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95)}


def measure_overhead(direct: httpx.Client, guard: httpx.Client, texts: Sequence[str],
                     stream: bool, repeats: int = 3) -> dict:
    """Per message: best-of-`repeats` through the guard minus best-of-`repeats` straight to
    the same upstream. Best-of removes one-off OS hiccups from both sides."""
    totals: list[float] = []
    firsts: list[float] = []
    for text in texts:
        runs = {name: [_timed(client, text, stream) for _ in range(repeats)]
                for name, client in (("direct", direct), ("guard", guard))}
        totals.append(min(t for t, _ in runs["guard"]) - min(t for t, _ in runs["direct"]))
        if stream:
            firsts.append(min(f for _, f in runs["guard"] if f is not None)
                          - min(f for _, f in runs["direct"] if f is not None))
    result = _summary(totals)
    if stream:
        result["ttft"] = _summary(firsts)
    return result


async def load(client: httpx.AsyncClient, texts: Sequence[str], concurrency: int,
               seconds: float) -> dict:
    """`concurrency` clients sending non-streaming requests back to back for `seconds`.
    Errors (non-200, timeouts) are counted, and only successes count towards throughput."""
    latencies: list[float] = []
    counts = {"requests": 0, "errors": 0}
    deadline = time.perf_counter() + seconds
    next_text = iter(range(sys.maxsize))

    async def worker() -> None:
        while time.perf_counter() < deadline:
            text = texts[next(next_text) % len(texts)]
            started = time.perf_counter()
            counts["requests"] += 1
            try:
                response = await client.post(CHAT, json=_body(text, False))
                ok = response.status_code == 200
            except httpx.HTTPError:
                ok = False
            if ok:
                latencies.append((time.perf_counter() - started) * 1000)
            else:
                counts["errors"] += 1

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - started
    return {"concurrency": concurrency, "seconds": seconds, **counts,
            "msgs_per_s": len(latencies) / elapsed,
            "p50_ms": _percentile(latencies, 0.50) if latencies else None,
            "p95_ms": _percentile(latencies, 0.95) if latencies else None}


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _docker_info() -> dict | None:
    try:
        out = subprocess.run(
            ["docker", "info", "--format",
             '{"server": "{{.ServerVersion}}", "cpus": {{.NCPU}}, "memory": {{.MemTotal}}}'],
            capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    info = json.loads(out)
    return {"server": info["server"], "cpus": info["cpus"],
            "memory_gb": round(info["memory"] / 2**30, 1)}


def _ram_gb() -> float | None:
    try:
        import psutil  # [perf] extra only
    except ImportError:
        return None
    return round(psutil.virtual_memory().total / 2**30, 1)


def machine_info(docker: bool = True) -> dict:
    """Every performance number is printed next to this."""
    return {"cpu": _cpu_model(), "logical_cores": os.cpu_count(), "ram_gb": _ram_gb(),
            "os": platform.platform(), "python": platform.python_version(),
            "docker": _docker_info() if docker else None}


def main() -> None:
    from leakguard.evaluation.loading import load_split

    parser = argparse.ArgumentParser(prog="python -m leakguard.evaluation.perf")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("load", help="load-test a guard; prints JSON")
    run.add_argument("--url", required=True, help="guard base URL, e.g. http://guard:8080")
    run.add_argument("--concurrency", type=int, required=True)
    run.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    texts = forwardable_texts(load_split("test"))

    async def go() -> dict:
        async with httpx.AsyncClient(base_url=args.url, timeout=60) as client:
            return await load(client, texts, args.concurrency, args.seconds)
    print(json.dumps(asyncio.run(go())))


if __name__ == "__main__":
    main()
