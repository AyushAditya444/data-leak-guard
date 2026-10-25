"""Streamed replies (Server-Sent Events): split the byte stream into events, and restore
placeholders inside the chat-chunk events without otherwise changing the stream.

    data: {"choices":[{"index":0,"delta":{"content":"call [PHO"}}]}      ->  content "call "
    data: {"choices":[{"index":0,"delta":{"content":"NE_1] now"}}]}      ->  content "9876543210 now"
    data: [DONE]                                                          ->  unchanged

Each (choice, field) has its own FieldRestorer, so text held back in one event rides along in
a later event of the same field. Events that need no change are sent as their original bytes;
held text is flushed on finish_reason, before [DONE] or any other non-chunk event, and when the
upstream stream ends, so nothing is lost.
"""

import codecs
import json
import re
from dataclasses import dataclass

from leakguard.restore import TEXT_FIELDS, FieldRestorer, Restorer

# An event ends at a blank line. The spec also allows a lone "\r" as a line end; OpenAI-style
# providers use "\n" or "\r\n", which is what is supported here.
_BOUNDARY = re.compile(r"\r?\n\r?\n")
_SYNTHETIC_KEYS = ("id", "object", "created", "model")


@dataclass(frozen=True)
class SSEEvent:
    raw: str  # the event's exact text, including the blank line that ends it
    data: str | None  # its data lines joined with "\n", or None (e.g. a ": ping" comment)


def _event(raw: str) -> SSEEvent:
    data = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    return SSEEvent(raw, "\n".join(data) if data else None)


class SSEParser:
    """Bytes in, complete events out. A read can end mid-event or mid-character."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._scanned = 0  # no event boundary before this index: don't search it again

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buffer += self._decoder.decode(chunk)
        return self._split()

    def close(self) -> list[SSEEvent]:
        """At the end of the stream: whatever is left is the last event."""
        self._buffer += self._decoder.decode(b"", final=True)
        events = self._split()
        if self._buffer:
            events.append(_event(self._buffer))
            self._buffer = ""
        return events

    def _split(self) -> list[SSEEvent]:
        events = []
        while match := _BOUNDARY.search(self._buffer, max(0, self._scanned - 3)):
            events.append(_event(self._buffer[:match.end()]))
            self._buffer = self._buffer[match.end():]
            self._scanned = 0
        self._scanned = len(self._buffer)
        return events


def _chat_chunk(data: str) -> dict | None:
    try:
        obj = json.loads(data)
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
        return obj
    return None


def _serialise(obj: dict, raw: str = "") -> str:
    """A rewritten event: the original's non-data lines (e.g. "id: 7"), then the new data."""
    kept = [line for line in raw.splitlines() if line and not line.startswith("data:")]
    return "".join(f"{line}\n" for line in kept) + \
        f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


class StreamRestorer:
    """Rewrites one streamed reply's events. Call process() per event, finish() at the end."""

    def __init__(self, restorer: Restorer):
        self._restorer = restorer
        # key: (choice index, field name) or (choice index, "tool", tool-call index)
        self._fields: dict[tuple, FieldRestorer] = {}
        self._last_chunk: dict | None = None

    def process(self, event: SSEEvent) -> list[str]:
        if event.data is None:
            return [event.raw]
        obj = _chat_chunk(event.data)
        if obj is None:  # [DONE], an error event, anything that isn't a chat chunk
            return self._flush_all() + [event.raw]
        self._last_chunk = obj
        changed = False
        for choice in obj["choices"]:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            if isinstance(choice.get("delta"), dict):
                changed |= self._restore_delta(index, choice["delta"])
            if choice.get("finish_reason") is not None:
                changed |= self._flush_choice_into(index, choice)
        return [_serialise(obj, event.raw)] if changed else [event.raw]

    def finish(self) -> list[str]:
        """The upstream stream ended: send anything still held, then count leftovers."""
        out = self._flush_all()
        for field in self._fields.values():
            field.close()
        return out

    def _field(self, key: tuple) -> FieldRestorer:
        if key not in self._fields:
            self._fields[key] = self._restorer.field(json_escape=key[1] == "tool")
        return self._fields[key]

    def _restore_delta(self, index: int, delta: dict) -> bool:
        changed = False
        for name in TEXT_FIELDS:
            if isinstance(delta.get(name), str):
                new = self._field((index, name)).feed(delta[name])
                changed |= new != delta[name]
                delta[name] = new
        for call in delta.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                new = self._field((index, "tool", call.get("index", 0))).feed(
                    function["arguments"])
                changed |= new != function["arguments"]
                function["arguments"] = new
        return changed

    def _held_for(self, index: int) -> dict:
        """Flush one choice's fields into a delta: {"content": "...", "tool_calls": [...]}."""
        delta: dict = {}
        for key, field in self._fields.items():
            if key[0] != index:
                continue
            held = field.flush()
            if not held:
                continue
            if key[1] == "tool":
                delta.setdefault("tool_calls", []).append(
                    {"index": key[2], "function": {"arguments": held}})
            else:
                delta[key[1]] = held
        return delta

    def _flush_choice_into(self, index: int, choice: dict) -> bool:
        held = self._held_for(index)
        if not held:
            return False
        if not isinstance(choice.get("delta"), dict):
            choice["delta"] = {}
        delta = choice["delta"]
        for name, text in held.items():
            if name != "tool_calls":
                delta[name] = (delta.get(name) or "") + text
                continue
            calls = delta.setdefault("tool_calls", [])
            for extra in text:
                same = next((c for c in calls if isinstance(c, dict)
                             and c.get("index", 0) == extra["index"]), None)
                if same is None:
                    calls.append(extra)
                else:
                    function = same.setdefault("function", {})
                    function["arguments"] = (function.get("arguments") or "") + \
                        extra["function"]["arguments"]
        return True

    def _flush_all(self) -> list[str]:
        """One extra chunk event carrying every choice's held text, if there is any."""
        choices = []
        for index in sorted({key[0] for key in self._fields}):
            delta = self._held_for(index)
            if delta:
                choices.append({"index": index, "delta": delta, "finish_reason": None})
        if not choices:
            return []
        last = self._last_chunk or {}
        base = {key: last[key] for key in _SYNTHETIC_KEYS if key in last}
        base.setdefault("object", "chat.completion.chunk")
        return [_serialise({**base, "choices": choices})]
