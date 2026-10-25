"""Real Groq streams, recorded on the dev run (masked side, synthetic values only), replayed
through the restorer: real provider chunking and fields, no network, no key."""

import json
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionChunk

from leakguard.entities import EntityType
from leakguard.masking import PlaceholderMap
from leakguard.restore import TEXT_FIELDS, Restorer
from leakguard.sse import SSEParser, StreamRestorer

FIXTURES = Path(__file__).parent / "fixtures" / "groq_streams"


def _restorer_from(placeholders: dict[str, str]) -> Restorer:
    """Rebuild the request's mapping: placeholders were handed out in this order."""
    mapping = PlaceholderMap()
    for placeholder, value in placeholders.items():
        entity = EntityType(placeholder[1:-1].rsplit("_", 1)[0])
        assert mapping.placeholder_for(entity, value) == placeholder
    return Restorer(mapping)


def _run(sse: str, restorer: Restorer, read_size: int) -> str:
    data, parser, stream, out = sse.encode(), SSEParser(), StreamRestorer(restorer), []
    for i in range(0, len(data), read_size):
        for event in parser.feed(data[i:i + read_size]):
            out += stream.process(event)
    for event in parser.close():
        out += stream.process(event)
    return "".join(out + stream.finish())


def _chunks(sse: str) -> list[dict]:
    parser = SSEParser()
    events = parser.feed(sse.encode()) + parser.close()
    return [json.loads(e.data) for e in events if e.data and e.data.startswith("{")]


def _field_texts(sse: str) -> dict[tuple, str]:
    """Per (choice, field) text, tool-call arguments keyed by tool index."""
    texts: dict[tuple, str] = {}
    for obj in _chunks(sse):
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            for name, value in delta.items():
                if isinstance(value, str):
                    key = (choice["index"], name)
                    texts[key] = texts.get(key, "") + value
            for call in delta.get("tool_calls") or []:
                args = (call.get("function") or {}).get("arguments")
                if isinstance(args, str):
                    key = (choice["index"], "tool", call["index"])
                    texts[key] = texts.get(key, "") + args
    return texts


def _fixture_paths() -> list[Path]:
    return sorted(FIXTURES.glob("*.json"))


def test_fixtures_exist():
    assert len(_fixture_paths()) >= 3, "record some with: restore-run --split dev --save-streams 6"


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
@pytest.mark.parametrize("read_size", [1, 7, 4096])
def test_recorded_groq_stream(path, read_size):
    fx = json.loads(path.read_text(encoding="utf-8"))
    out = _run(fx["sse"], _restorer_from(fx["placeholders"]), read_size)
    raw = _field_texts(fx["sse"])
    expected = {}
    for key, text in raw.items():  # other string fields (e.g. "role") pass through unchanged
        is_tool = key[1] == "tool"
        restorable = is_tool or key[1] in TEXT_FIELDS
        whole = _restorer_from(fx["placeholders"])
        expected[key] = whole.restore_text(text, json_escape=is_tool) if restorable else text
    assert _field_texts(out) == expected  # streaming == restoring each whole field at once
    for obj in _chunks(out):  # every event still parses as an OpenAI chunk
        ChatCompletionChunk.model_validate(obj)
    assert out.rstrip().endswith("data: [DONE]")
