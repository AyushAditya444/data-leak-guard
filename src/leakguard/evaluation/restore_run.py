"""How often do real LLMs' replies come back fully restored? Measured on Groq's free tier.

Each conversation is a support-desk message from the evaluation set plus an instruction that
makes the model repeat the details back (left alone, models often don't, and there would be
nothing to measure). It is sent through the real guard, in process, once streaming and once
not, to each model. Per reply we record:

- how the model wrote our placeholders, read from the RAW upstream reply (exact, case,
  separator, other brackets, bracketless, unclosed), via a recording transport;
- the guard's own `restored` / `unrestored` counts, read from its log line.

The results file holds counts only: no reply text, no values, no key. The key is read from
GROQ_API_KEY and only ever sent to Groq.
"""

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from leakguard.app import create_app
from leakguard.config import Settings
from leakguard.detect import build_detector
from leakguard.detect.engine import Detector
from leakguard.entities import EntityType
from leakguard.evaluation.loading import load_split
from leakguard.masking import PlaceholderMap, mask_messages
from leakguard.placeholders import loose_mentions
from leakguard.policy import DEFAULT_POLICY
from leakguard.restore import TEXT_FIELDS
from leakguard.sse import SSEParser

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b")
SIZES = {"dev": 100, "test": 300}
SEEDS = {"dev": 1, "test": 2}
MAX_TOKENS = 600
MAX_RETRIES = 5
LOW_TOKENS = 1500  # pause until the per-minute token budget resets below this
TOOL_SHARE = 5  # every 5th conversation uses the forced tool: 20%

SYSTEM_PROMPT = ("You are a customer support agent for an Indian company. Help the customer "
                 "with their message.")
INSTRUCTIONS = (
    "Please confirm all my details back to me exactly as I gave them.",
    "Can you repeat my details back so I know you noted them correctly?",
    "Draft a short reply to this customer that restates every detail they shared.",
    ("Write a one-paragraph summary of this message for the support ticket, keeping every "
     "detail exactly as given."),
    "Mujhe mere saare details wapas confirm kar dijiye, bilkul waise hi jaise maine diye.",
)
TOOL = {"type": "function", "function": {
    "name": "create_ticket",
    "description": "Create a support ticket holding the customer's details.",
    "parameters": {"type": "object", "properties": {
        "name": {"type": "string"}, "phone": {"type": "string"},
        "email": {"type": "string"}, "address": {"type": "string"},
        "notes": {"type": "string", "description": "every other detail, exactly as given"},
    }, "required": ["notes"]}}}
TOOL_CHOICE = {"type": "function", "function": {"name": "create_ticket"}}


@dataclass(frozen=True)
class Conversation:
    id: str
    sample_id: str
    messages: list[dict]
    tools: list[dict] | None
    tool_choice: dict | None


def build_conversations(split: str, n: int, seed: int) -> list[Conversation]:
    """`n` conversations from this split's samples that carry sensitive data the default
    policy masks: no secrets (they are blocked, so nothing comes back), no negatives, and no
    `hard` samples (formats the detector is known to miss, so often nothing is masked)."""
    eligible = [s for s in load_split(split)
                if s.spans and s.category not in ("hard", "negative")
                and EntityType.SECRET not in {span.type for span in s.spans}]
    rng = random.Random(seed)
    conversations = []
    for i, sample in enumerate(rng.sample(eligible, n)):
        uses_tool = i % TOOL_SHARE == 0
        instruction = "Open a ticket for this customer." if uses_tool else rng.choice(INSTRUCTIONS)
        conversations.append(Conversation(
            id=f"{split}-{i:03d}", sample_id=sample.id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"{sample.text}\n\n{instruction}"}],
            tools=[TOOL] if uses_tool else None,
            tool_choice=TOOL_CHOICE if uses_tool else None))
    return conversations


def api_key_from_env() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit("GROQ_API_KEY is not set. Create a free key at console.groq.com and "
                         "set it as an environment variable (never in a file in this repo).")
    return key


def parse_reset(text: str) -> float:
    """Groq's reset headers look like "7.66s", "2m59.5s", "120ms"; seconds, 0 if unreadable."""
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?", text.strip())
    if not text.strip() or not m:
        return 0.0
    h, mins, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mins * 60 + s + ms / 1000


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Wraps the real transport and keeps the raw bytes of the last upstream reply, so the
    harness sees what the model actually wrote, before the guard restored anything."""

    def __init__(self, inner: httpx.AsyncBaseTransport):
        self._inner = inner
        self.last = bytearray()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.headers["accept-encoding"] = "identity"  # record plain bytes, not gzip
        response = await self._inner.handle_async_request(request)
        self.last = recorded = bytearray()
        upstream = response.stream

        class Recorded(httpx.AsyncByteStream):
            async def __aiter__(self):
                async for chunk in upstream:
                    recorded.extend(chunk)
                    yield chunk

            async def aclose(self):
                await upstream.aclose()

        return httpx.Response(response.status_code, headers=response.headers,
                              stream=Recorded(), extensions=response.extensions)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _GuardLog(logging.Handler):
    """Collects the guard's request log lines by request id."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.records: dict[str, dict] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            data = json.loads(record.getMessage())
        except ValueError:
            return
        if data.get("event") == "request":
            self.records[data["request_id"]] = data


def _reply_texts(raw: bytes, stream: bool) -> list[str]:
    """Every text field of the raw upstream reply (content, reasoning, tool arguments...)."""
    texts: dict[tuple, str] = {}

    def add(key: tuple, value: object) -> None:
        if isinstance(value, str):
            texts[key] = texts.get(key, "") + value

    def collect(index: object, part: dict) -> None:
        for name in TEXT_FIELDS:
            add((index, name), part.get(name))
        for call in part.get("tool_calls") or []:
            if isinstance(call, dict) and isinstance(call.get("function"), dict):
                add((index, "tool", call.get("index", 0)), call["function"].get("arguments"))

    if stream:
        parser = SSEParser()
        events = parser.feed(bytes(raw)) + parser.close()
        objects = [json.loads(e.data) for e in events if e.data and e.data.startswith("{")]
        parts = [(c.get("index"), c.get("delta")) for o in objects for c in o.get("choices", [])]
    else:
        try:
            body = json.loads(raw)
        except ValueError:
            return []
        parts = [(c.get("index"), c.get("message")) for c in body.get("choices", [])]
    for index, part in parts:
        if isinstance(part, dict):
            collect(index, part)
    return list(texts.values())


def _mentions(texts: list[str], mapping: PlaceholderMap) -> Counter:
    """Placeholder mentions that point at one of ours, counted by how they were written."""
    forms: Counter = Counter()
    for text in texts:
        for m in loose_mentions(text, mapping.entities()):
            if mapping.value_for(m.entity, m.number) is not None:
                forms[m.form] += 1
    return forms


def _body(conversation: Conversation, model: str, stream: bool, max_tokens: int) -> dict:
    body = {"model": model, "messages": conversation.messages, "max_tokens": max_tokens,
            "stream": stream}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"  # its reasoning tokens count against the quota
    if conversation.tools:
        body |= {"tools": conversation.tools, "tool_choice": conversation.tool_choice}
    return body


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def run_conversations(
    conversations: list[Conversation], models: list[str] | tuple[str, ...],
    transport: httpx.AsyncBaseTransport, api_key: str, max_tokens: int,
    save_streams: int = 0, name_model: Detector | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
    """Run every conversation x model x (non-streaming, streaming) through the guard."""
    recorder = _RecordingTransport(transport)
    settings = Settings(upstream_base_url=GROQ_BASE_URL, upstream_timeout_s=120.0,
                        ner_enabled=name_model is not None)
    app = create_app(settings, recorder, DEFAULT_POLICY, name_model=name_model)
    detect = build_detector(name_model)  # the guard's detector, to know its placeholders
    log, guard_logger = _GuardLog(), logging.getLogger("leakguard")
    guard_logger.addHandler(log)
    old_level = guard_logger.level
    guard_logger.setLevel(logging.INFO)
    headers = {"authorization": f"Bearer {api_key}"}
    outcomes, fixtures = [], []
    try:
        async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://guard",
                timeout=180.0) as client:
            for n, conversation in enumerate(conversations, 1):
                mapping = mask_messages(conversation.messages, DEFAULT_POLICY, detect).mapping
                for model in models:
                    for stream in (False, True):
                        body = _body(conversation, model, stream, max_tokens)
                        response = await _send(client, body, headers, sleep)
                        raw = bytes(recorder.last)
                        ok = response.status_code == 200
                        forms = _mentions(_reply_texts(raw, stream), mapping) if ok else Counter()
                        record = log.records.get(response.headers.get("x-request-id", ""), {})
                        outcomes.append({
                            "conversation": conversation.id, "model": model, "stream": stream,
                            "tool": conversation.tools is not None,
                            "status": response.status_code, "mentions": dict(forms),
                            "restored": record.get("restored", 0),
                            "unrestored": record.get("unrestored", 0),
                            "mentioned": sum(forms.values()) > 0})
                        if stream and ok and forms and len(fixtures) < save_streams:
                            fixtures.append({"model": model, "placeholders": mapping.placeholders(),
                                             "sse": raw.decode("utf-8")})
                        await _pace(response, sleep)
                if n % 10 == 0:
                    print(f"  {n}/{len(conversations)} conversations", file=sys.stderr)
    finally:
        guard_logger.removeHandler(log)
        guard_logger.setLevel(old_level)
    meta = {"date": datetime.now(UTC).date().isoformat(), "models": list(models),
            "conversations": len(conversations), "max_tokens": max_tokens,
            "temperature": "provider default",
            "reasoning_effort": "low for openai/gpt-oss models, provider default otherwise",
            "ner": name_model is not None, "policy": "default", "guard_commit": _git_commit()}
    result = {"meta": meta, "outcomes": outcomes}
    if save_streams:
        result["fixtures"] = fixtures
    return result


async def _send(client: httpx.AsyncClient, body: dict, headers: dict,
                sleep: Callable[[float], Awaitable[None]]) -> httpx.Response:
    """POST through the guard; on 429 wait as told and retry, at most MAX_RETRIES times."""
    for attempt in range(MAX_RETRIES + 1):
        if body["stream"]:
            async with client.stream("POST", "/v1/chat/completions", json=body,
                                     headers=headers) as response:
                await response.aread()
        else:
            response = await client.post("/v1/chat/completions", json=body, headers=headers)
        if response.status_code != 429 or attempt == MAX_RETRIES:
            return response
        try:
            wait = float(response.headers.get("retry-after", "2"))
        except ValueError:
            wait = 2.0
        await sleep(wait)
    raise AssertionError("unreachable")


async def _pace(response: httpx.Response, sleep: Callable[[float], Awaitable[None]]) -> None:
    """Stay under the per-minute token limit using the rate-limit headers the guard relays."""
    remaining = response.headers.get("x-ratelimit-remaining-tokens")
    if remaining and remaining.isdigit() and int(remaining) < LOW_TOKENS:
        await sleep(parse_reset(response.headers.get("x-ratelimit-reset-tokens", "")) + 0.5)
