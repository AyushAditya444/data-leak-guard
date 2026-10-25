"""Record the demo session: run every scene through the real guard and save what happened.

    python -m examples.demo.record --upstream groq   # the recording the video replays
    python -m examples.demo.record --upstream mock   # free, no key; what CI runs

The guard runs in this process on a local port, exactly as `create_app` builds it, with the
wire tap around its upstream connection. Your app's side is the official OpenAI SDK with
`base_url` pointed at the guard. The Groq key comes from GROQ_API_KEY and is checked not to
appear anywhere in the session before it is written.
"""

import argparse
import json
import logging
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Literal

import httpx
import openai
import uvicorn

from examples.demo.scenes import SCENES, SYSTEM_PROMPT, Scene
from examples.demo.session import save_session, verify_session
from examples.demo.tap import WireTap
from leakguard.app import create_app
from leakguard.config import Settings
from leakguard.evaluation.presidio_report import both_support_summary
from leakguard.evaluation.restore_run import GROQ_BASE_URL, api_key_from_env
from leakguard.mock_llm import create_mock_app

DEFAULT_OUT = Path("examples/demo/session.json")
PRESIDIO_RESULTS = Path("data/eval/presidio-test.json")
GROQ_MODEL = "openai/gpt-oss-120b"
MOCK_KEY = "not-needed-for-mock"
MAX_ATTEMPTS = 3
LOG_WAIT_S = 2.0  # a streamed request is logged just after its last chunk


def serve_in_thread(app) -> tuple[str, Callable[[], None]]:
    """Serve `app` on a free local port. Returns its OpenAI-style base URL and a stop function."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("the guard failed to start (see the error above)")
        time.sleep(0.05)

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=10)

    return f"http://127.0.0.1:{port}/v1", stop


class _RequestLog(logging.Handler):
    """Collects the guard's one-line request logs (counts only, never values)."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            data = json.loads(record.getMessage())
        except ValueError:
            return
        if data.get("event") == "request":
            self.lines.append(data)

    def wait_for(self, count: int) -> dict | None:
        deadline = time.monotonic() + LOG_WAIT_S
        while len(self.lines) < count and time.monotonic() < deadline:
            time.sleep(0.02)
        return self.lines[count - 1] if len(self.lines) >= count else None


def _error_text(error: openai.APIStatusError) -> str:
    try:
        return json.loads(error.response.text)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return error.response.text


def _call(client: openai.OpenAI, model: str, extra: dict, scene: Scene,
          messages: list[dict]) -> dict:
    """One request as the app makes it; returns {"status", "text", "chunks"}."""
    started = time.perf_counter()
    if not scene.stream:
        reply = client.chat.completions.create(model=model, messages=messages, **extra)
        return {"status": 200, "text": reply.choices[0].message.content or "", "chunks": []}
    chunks = []
    for chunk in client.chat.completions.create(model=model, messages=messages, stream=True,
                                                **extra):
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append([round(time.perf_counter() - started, 3),
                           chunk.choices[0].delta.content])
    return {"status": 200, "text": "".join(text for _, text in chunks), "chunks": chunks}


def _run_scene(client, model, extra, scene: Scene, tap: WireTap) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *scene.messages]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        tap.clear()
        try:
            got = _call(client, model, extra, scene, messages)
        except openai.APIStatusError as error:
            retry = error.status_code == 429 or error.status_code >= 500
            if retry and attempt < MAX_ATTEMPTS:
                time.sleep(float(error.response.headers.get("retry-after", 2 ** attempt)))
                continue
            got = {"status": error.status_code, "text": _error_text(error), "chunks": []}
        break
    return {"app_sent": messages, "stream": scene.stream, "wire": list(tap.records),
            "app_got": got}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record_session(upstream: Literal["groq", "mock"], out: Path, *,
                   inner: httpx.AsyncBaseTransport | None = None,
                   api_key: str | None = None) -> dict:
    """Run every scene through the guard and write the session to `out`. `inner` replaces
    the real upstream connection (tests); `api_key` replaces GROQ_API_KEY / the mock's."""
    if upstream == "groq":
        base_url, model = GROQ_BASE_URL, GROQ_MODEL
        extra = {"max_tokens": 600, "extra_body": {"reasoning_effort": "low"}}
        inner = inner or httpx.AsyncHTTPTransport()
        api_key = api_key or api_key_from_env()
    else:
        base_url, model, extra = "http://mock-llm/v1", "mock", {}
        inner = inner or httpx.ASGITransport(create_mock_app())
        api_key = api_key or MOCK_KEY

    tap = WireTap(inner)
    log = _RequestLog()
    logger = logging.getLogger("leakguard")
    logger.addHandler(log)
    logger.setLevel(logging.INFO)
    guard_url, stop = serve_in_thread(
        create_app(Settings(upstream_base_url=base_url, ner_enabled=True), transport=tap))
    try:
        client = openai.OpenAI(base_url=guard_url, api_key=api_key, max_retries=0)
        scenes = {}
        for scene in SCENES:
            show = {"title": scene.title, "highlight": list(scene.highlight),
                    "beats": scene.beats, "expect": scene.expect}
            scenes[scene.id] = {"show": show}
            if scene.messages:
                logged = len(log.lines)
                scenes[scene.id] |= _run_scene(client, model, extra, scene, tap)
                # The last line is the final attempt's (a retried attempt logs one too).
                log.wait_for(logged + 1)
                scenes[scene.id]["log"] = log.lines[-1] if len(log.lines) > logged else None
    finally:
        stop()
        logger.removeHandler(log)

    results = json.loads(PRESIDIO_RESULTS.read_text(encoding="utf-8"))
    session = {"meta": {"recorded": date.today().isoformat(), "upstream": upstream,
                        "model": model, "guard_commit": _git_commit()},
               "scenes": scenes, "results": both_support_summary(results)}
    if api_key in json.dumps(session, ensure_ascii=False):
        raise RuntimeError("refusing to write: the API key appears in the session")
    save_session(session, out)
    return session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", choices=("groq", "mock"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    session = record_session(args.upstream, args.out)
    problems = verify_session(session)
    for problem in problems:
        print("claim does not hold:", problem, file=sys.stderr)
    print(f"wrote {args.out} ({'NOT ok: record again' if problems else 'all claims hold'})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
