"""Reversible masking: swap sensitive values for typed placeholders, and swap them back.

    "call 9876543210"  --mask-->  "call [PHONE_1]"  --LLM-->  "I'll call [PHONE_1]"
                                                    --restore-->  "I'll call 9876543210"

The value <-> placeholder mapping lives only in memory, for one request. Chat requests
resend the whole conversation every time, and numbering is deterministic (order of first
appearance), so the same history always produces the same placeholders: nothing has to
be stored between requests.
"""

import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from leakguard.detect import detect_rules
from leakguard.detect.spans import Span
from leakguard.entities import EntityType
from leakguard.placeholders import accepted_mentions
from leakguard.policy import Action, Policy

PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)_(\d+)\]")

# For these types, formatting is noise: "98765 43210" and "+91-9876543210" are one person.
_DIGIT_TYPES = {EntityType.AADHAAR, EntityType.PHONE, EntityType.CARD, EntityType.BANK_ACCOUNT}

TextFn = Callable[[str], str]


def _normalise(entity: EntityType, value: str) -> str:
    if entity in _DIGIT_TYPES:
        digits = "".join(ch for ch in value if ch.isdigit())
        return digits[-10:] if entity is EntityType.PHONE else digits  # drop +91 / 0 prefix
    return value.strip().casefold()


class PlaceholderMap:
    """Request-scoped two-way mapping between real values and placeholders."""

    def __init__(self, reserved: Mapping[EntityType, set[int]] | None = None):
        # Numbers already present as literal text in the request (e.g. the user typed
        # "[PHONE_1]"). We never hand those out, so a user can't trick the guard into
        # "restoring" a placeholder it didn't create.
        self._reserved = {entity: set(numbers) for entity, numbers in (reserved or {}).items()}
        self._placeholder_by_value: dict[tuple[EntityType, str], str] = {}
        self._value_by_placeholder: dict[str, str] = {}
        self._last_number: dict[EntityType, int] = {}

    def placeholder_for(self, entity: EntityType, value: str) -> str:
        key = (entity, _normalise(entity, value))
        if key in self._placeholder_by_value:
            return self._placeholder_by_value[key]
        number = self._last_number.get(entity, 0) + 1
        while number in self._reserved.get(entity, ()):
            number += 1
        self._last_number[entity] = number
        placeholder = f"[{entity.value}_{number}]"
        self._placeholder_by_value[key] = placeholder
        self._value_by_placeholder[placeholder] = value
        return placeholder

    def value_for(self, entity: EntityType, number: int) -> str | None:
        """The real value behind a placeholder this map created; None for numbers it never
        handed out (including reserved ones the user typed)."""
        return self._value_by_placeholder.get(f"[{entity.value}_{number}]")

    def entities(self) -> set[EntityType]:
        return {entity for entity, _ in self._placeholder_by_value}

    def restore(self, text: str) -> str:
        """Replace placeholders this map created; leave every other bracketed text alone."""
        return PLACEHOLDER_RE.sub(
            lambda m: self._value_by_placeholder.get(m.group(0), m.group(0)), text)

    def __len__(self) -> int:
        return len(self._value_by_placeholder)


@dataclass
class MaskResult:
    messages: list
    mapping: PlaceholderMap
    counts: Counter = field(default_factory=Counter)  # (EntityType, Action) -> how many
    blocked: list[EntityType] = field(default_factory=list)


def _map_tool_call(tool_call: object, fn: TextFn) -> object:
    if not isinstance(tool_call, dict):
        return tool_call
    function = tool_call.get("function")
    if isinstance(function, dict) and isinstance(function.get("arguments"), str):
        return {**tool_call, "function": {**function, "arguments": fn(function["arguments"])}}
    return tool_call


def _map_message(message: dict, fn: TextFn) -> dict:
    """Apply `fn` to every text field of a chat message, returning a new message.
    Text lives in: `content` (a string, or a list of {"type": "text"} parts) and in
    tool-call arguments. Images and other parts pass through untouched."""
    out = dict(message)
    content = message.get("content")
    if isinstance(content, str):
        out["content"] = fn(content)
    elif isinstance(content, list):
        out["content"] = [
            {**part, "text": fn(part["text"])}
            if isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            else part
            for part in content
        ]
    if isinstance(message.get("tool_calls"), list):
        out["tool_calls"] = [_map_tool_call(call, fn) for call in message["tool_calls"]]
    return out


def _reserved_numbers(messages: list[dict]) -> dict[EntityType, set[int]]:
    reserved: dict[EntityType, set[int]] = {}

    def collect(text: str) -> str:
        # Every form the restorer accepts ("[phone 1]", "{PHONE-1}", ...), so no accepted
        # form can be planted by the user.
        for m in accepted_mentions(text, set(EntityType)):
            reserved.setdefault(m.entity, set()).add(m.number)
        return text

    for message in messages:
        _map_message(message, collect)
    return reserved


def mask_messages(
    messages: list[dict], policy: Policy, detect: Callable[[str], list[Span]] = detect_rules,
) -> MaskResult:
    """Detect sensitive data in every message and apply the policy. The input is not
    modified; the returned messages are what gets sent to the LLM."""
    mapping = PlaceholderMap(_reserved_numbers(messages))
    counts: Counter = Counter()
    blocked: set[EntityType] = set()

    def mask_text(text: str) -> str:
        pieces: list[str] = []
        cursor = 0
        for span in detect(text):
            action = policy.action_for(span.type)
            counts[(span.type, action)] += 1
            if action is Action.BLOCK:
                blocked.add(span.type)
            elif action is Action.MASK:
                pieces += [text[cursor:span.start], mapping.placeholder_for(span.type,
                                                                            span.text(text))]
                cursor = span.end
        pieces.append(text[cursor:])
        return "".join(pieces)

    masked = [_map_message(message, mask_text) for message in messages]
    return MaskResult(masked, mapping, counts, sorted(blocked))


def restore_response(body: dict, mapping: PlaceholderMap) -> dict:
    """Restore placeholders in a non-streaming chat completion (content + tool calls)."""
    choices = body.get("choices")
    if not isinstance(choices, list) or len(mapping) == 0:
        return body
    restored = []
    for choice in choices:
        if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
            choice = {**choice, "message": _map_message(choice["message"], mapping.restore)}
        restored.append(choice)
    return {**body, "choices": restored}
