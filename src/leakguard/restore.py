"""Put real values back into the LLM's reply: whole replies, and streamed ones piece by piece.

Streaming is the hard case: a placeholder can arrive split across chunks ("[PHO" + "NE_1]").
A FieldRestorer therefore holds back the shortest tail that could still become a
placeholder, and releases it once it completes (restored) or clearly can't (sent as-is):

    feed("call [PHO")  -> "call "            "[PHO" held back
    feed("NE_1] now")  -> "9876543210 now"   completed and restored

Nothing is guessed: text that isn't an accepted placeholder (placeholders.py) is left as the
LLM wrote it. Such leftovers that refer to one of our placeholders are counted as
`unrestored`, so the logs show how often restoring falls short.
"""

import json
from dataclasses import dataclass

from leakguard.masking import PlaceholderMap, _map_tool_call
from leakguard.placeholders import (
    ACCEPT_BRACKETLESS,
    BRACKETS,
    accepted_mentions,
    could_become_placeholder,
    loose_mentions,
    max_placeholder_len,
)

# Reply fields that carry text the user may see. `reasoning` / `reasoning_content` are the
# "thinking" text some providers add (e.g. gpt-oss on Groq).
TEXT_FIELDS = ("content", "refusal", "reasoning", "reasoning_content")


@dataclass
class RestoreStats:
    restored: int = 0
    unrestored: int = 0


def _json_escaped(value: str) -> str:
    # Tool-call arguments are a JSON document inside a string: a value with a quote or a
    # newline (multi-line addresses) must be escaped, or the arguments stop being valid JSON.
    return json.dumps(value, ensure_ascii=False)[1:-1]


class Restorer:
    """Restores one request's placeholders and counts how that went."""

    def __init__(self, mapping: PlaceholderMap, bracketless: bool = ACCEPT_BRACKETLESS):
        self._mapping = mapping
        self._bracketless = bracketless
        self.entities = frozenset(mapping.entities())
        self.stats = RestoreStats()

    def restore_text(self, text: str, json_escape: bool = False) -> str:
        out, spans = self._replace(text, json_escape)
        self.stats.restored += len(spans)
        self.stats.unrestored += self._count_unrestored(text, spans)
        return out

    def field(self, json_escape: bool = False) -> "FieldRestorer":
        return FieldRestorer(self, json_escape)

    def _replace(
        self, text: str, json_escape: bool, start: int = 0, stop: int | None = None
    ) -> tuple[str, list[tuple[int, int]]]:
        """Restore the accepted placeholders lying inside text[start:stop]. The whole `text`
        is matched, so the characters around the slice still count as context. Returns
        the restored slice and the (start, end) of each placeholder that was swapped."""
        stop = len(text) if stop is None else stop
        pieces, spans, cursor = [], [], start
        for m in accepted_mentions(text, self.entities, self._bracketless):
            if m.start < start or m.end > stop:
                continue
            value = self._mapping.value_for(m.entity, m.number)
            if value is None:  # reserved (the user typed it) or never handed out
                continue
            pieces += [text[cursor:m.start], _json_escaped(value) if json_escape else value]
            spans.append((m.start, m.end))
            cursor = m.end
        pieces.append(text[cursor:stop])
        return "".join(pieces), spans

    def _count_unrestored(self, text: str, restored: list[tuple[int, int]]) -> int:
        """Placeholder-looking leftovers that point at one of our placeholders."""
        return sum(
            1 for m in loose_mentions(text, self.entities)
            if self._mapping.value_for(m.entity, m.number) is not None
            and not any(s < m.end and m.start < e for s, e in restored))


class FieldRestorer:
    """Restores one streamed text field (one choice's content, or one tool call's
    arguments). feed() returns what is safe to send now; flush() returns the rest."""

    def __init__(self, restorer: Restorer, json_escape: bool):
        self._restorer = restorer
        self._json_escape = json_escape
        self._held = ""
        self._before = ""  # last released character: context for word boundaries
        self._raw: list[str] = []  # the field's full text, for counting at the end
        self._bound = max_placeholder_len(restorer.entities)

    def feed(self, text: str) -> str:
        self._raw.append(text)
        work = self._before + self._held + text
        offset = len(self._before)
        cut = self._cut(work, offset)
        return self._release(work, offset, cut)

    def flush(self) -> str:
        work = self._before + self._held
        return self._release(work, len(self._before), len(work))

    def close(self) -> None:
        """Count the unrestored leftovers over the field's full text. Call once, at the end."""
        text = "".join(self._raw)
        _, restored = self._restorer._replace(text, self._json_escape)
        self._restorer.stats.unrestored += self._restorer._count_unrestored(text, restored)

    def _release(self, work: str, offset: int, cut: int) -> str:
        out, spans = self._restorer._replace(work, self._json_escape, offset, cut)
        self._restorer.stats.restored += len(spans)
        self._held = work[cut:]
        if cut > offset:
            self._before = work[cut - 1]
        return out

    def _cut(self, work: str, offset: int) -> int:
        """Where to stop releasing: the earliest position whose tail could still become an
        accepted placeholder. Only the last `_bound` characters can be such a tail."""
        entities, bracketless = self._restorer.entities, self._restorer._bracketless
        for i in range(max(offset, len(work) - self._bound), len(work)):
            ch = work[i]
            starts_word = ch.isupper() and (i == 0 or not (work[i - 1].isalnum()
                                                           or work[i - 1] == "_"))
            if (ch in BRACKETS or (bracketless and starts_word)) \
                    and could_become_placeholder(work[i:], entities, bracketless):
                return i
        return len(work)


def restore_response(body: dict, restorer: Restorer) -> dict:
    """Restore a non-streaming chat completion: every text field and tool-call arguments."""
    choices = body.get("choices")
    if not isinstance(choices, list):
        return body
    restored = []
    for choice in choices:
        if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
            choice = {**choice, "message": _restore_message(choice["message"], restorer)}
        restored.append(choice)
    return {**body, "choices": restored}


def _restore_message(message: dict, restorer: Restorer) -> dict:
    out = dict(message)
    for name in TEXT_FIELDS:
        if isinstance(message.get(name), str):
            out[name] = restorer.restore_text(message[name])
    if isinstance(message.get("tool_calls"), list):
        out["tool_calls"] = [
            _map_tool_call(call, lambda s: restorer.restore_text(s, json_escape=True))
            for call in message["tool_calls"]]
    return out
