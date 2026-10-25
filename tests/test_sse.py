import json
import time

from leakguard.entities import EntityType as T
from leakguard.masking import PlaceholderMap
from leakguard.restore import Restorer
from leakguard.sse import SSEParser, StreamRestorer

DONE = "data: [DONE]\n\n"
PHONE = (T.PHONE, "9876543210")
NAME = (T.NAME, "Ravi")


def _restorer(*pairs):
    m = PlaceholderMap()
    for entity, value in pairs:
        m.placeholder_for(entity, value)
    return Restorer(m, bracketless=False)


def chunk(content=None, index=0, finish=None, tools=None, id_="c1", model="m"):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tools is not None:
        delta["tool_calls"] = tools
    obj = {"id": id_, "object": "chat.completion.chunk", "created": 1, "model": model,
           "choices": [{"index": index, "delta": delta, "finish_reason": finish}]}
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _cut(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def run(sse, restorer, read_size):
    data, parser, stream, out = sse.encode(), SSEParser(), StreamRestorer(restorer), []
    for i in range(0, len(data), read_size):
        for event in parser.feed(data[i:i + read_size]):
            out += stream.process(event)
    for event in parser.close():
        out += stream.process(event)
    out += stream.finish()
    return "".join(out)


def _payloads(out):
    return [line[6:] for line in out.split("\n") if line.startswith("data: ")]


def _chunks(out):
    return [json.loads(p) for p in _payloads(out) if p != "[DONE]"]


def texts(out):
    acc = {}
    for obj in _chunks(out):
        for choice in obj.get("choices", []):
            acc.setdefault(choice["index"], "")
            acc[choice["index"]] += (choice.get("delta") or {}).get("content") or ""
    return acc


def tool_args(out):
    acc = {}
    for obj in _chunks(out):
        for choice in obj.get("choices", []):
            for call in (choice.get("delta") or {}).get("tool_calls") or []:
                key = (choice["index"], call["index"])
                acc[key] = acc.get(key, "") + call.get("function", {}).get("arguments", "")
    return acc


def _parse_in_pieces(data, size):
    parser, events = SSEParser(), []
    for i in range(0, len(data), size):
        events += parser.feed(data[i:i + size])
    return events + parser.close()


def test_parser_handles_splits_crlf_comments_and_multibyte():
    raw = ": ping\r\n\r\n" + 'data: {"x": "नमस्ते"}\r\n\r\n' + "data: [DONE]\n\n"
    for size in (1, 2, 3, 7, len(raw.encode())):
        events = _parse_in_pieces(raw.encode(), size)
        assert [e.data for e in events] == [None, '{"x": "नमस्ते"}', "[DONE]"]
        assert "".join(e.raw for e in events) == raw


def test_parser_close_returns_trailing_event():
    p = SSEParser()
    assert p.feed(b"data: [DONE]") == []
    assert [e.data for e in p.close()] == ["[DONE]"]


def test_split_placeholder_restored_three_char_chunks():
    sse = "".join(chunk(content=c) for c in _cut("I'll call [PHONE_1] today", 3)) + DONE
    out = run(sse, _restorer(PHONE), read_size=5)
    assert texts(out)[0] == "I'll call 9876543210 today"
    assert _payloads(out)[-1] == "[DONE]"


def test_two_choices_interleaved():
    a, b = _cut("hi [PHONE_1]!", 2), _cut("yo [NAME_1]", 2)
    events = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            events.append(chunk(content=a[i], index=0))
        if i < len(b):
            events.append(chunk(content=b[i], index=1))
    out = run("".join(events) + DONE, _restorer(PHONE, NAME), read_size=11)
    assert texts(out) == {0: "hi 9876543210!", 1: "yo Ravi"}


def test_tool_calls_interleaved():
    a, b = _cut('{"p": "[PHONE_1]"}', 3), _cut('{"n": "[NAME_1]"}', 3)
    events = [chunk(tools=[{"index": 0, "id": "t0", "type": "function",
                            "function": {"name": "f", "arguments": ""}}]),
              chunk(tools=[{"index": 1, "id": "t1", "type": "function",
                            "function": {"name": "g", "arguments": ""}}])]
    for i in range(max(len(a), len(b))):
        if i < len(a):
            events.append(chunk(tools=[{"index": 0, "function": {"arguments": a[i]}}]))
        if i < len(b):
            events.append(chunk(tools=[{"index": 1, "function": {"arguments": b[i]}}]))
    out = run("".join(events) + chunk(finish="tool_calls") + DONE, _restorer(PHONE, NAME), 64)
    args = tool_args(out)
    assert json.loads(args[(0, 0)]) == {"p": "9876543210"}
    assert json.loads(args[(0, 1)]) == {"n": "Ravi"}


def test_held_text_flushed_into_finish_event():
    sse = chunk(content="call [PH") + chunk(finish="stop") + DONE
    out = run(sse, _restorer(PHONE), read_size=1000)
    assert texts(out)[0] == "call [PH"
    assert len(_payloads(out)) == 3  # no extra event
    finish = _chunks(out)[1]["choices"][0]
    assert finish["delta"]["content"] == "[PH" and finish["finish_reason"] == "stop"


def test_held_text_flushed_before_done():
    sse = chunk(content="call [PH", id_="abc", model="gpt") + DONE
    out = run(sse, _restorer(PHONE), read_size=1000)
    payloads = _payloads(out)
    assert len(payloads) == 3 and payloads[-1] == "[DONE]"
    synthetic = json.loads(payloads[1])
    assert (synthetic["id"], synthetic["model"], synthetic["object"]) == \
        ("abc", "gpt", "chat.completion.chunk")
    assert synthetic["choices"] == [{"index": 0, "delta": {"content": "[PH"},
                                     "finish_reason": None}]


def test_upstream_ends_without_done():
    out = run(chunk(content="call [PH"), _restorer(PHONE), read_size=1000)
    assert texts(out)[0] == "call [PH" and len(_payloads(out)) == 2


def test_error_event_mid_stream():
    error = 'data: {"error": {"message": "boom"}}\n\n'
    out = run(chunk(content="a [PH") + error, _restorer(PHONE), read_size=1000)
    payloads = _payloads(out)
    assert json.loads(payloads[0])["choices"][0]["delta"]["content"] == "a "
    assert json.loads(payloads[1])["choices"][0]["delta"]["content"] == "[PH"
    assert out.endswith(error)


def test_unchanged_events_are_byte_identical():
    sse = ": keep-alive\n\n" + chunk(content="hello ") + chunk(content="world [x]") + \
        chunk(finish="stop") + DONE
    assert run(sse, _restorer(PHONE), read_size=3) == sse


def test_rewritten_event_keeps_other_lines():
    sse = "id: 7\n" + chunk(content="call [PHONE_1]")
    out = run(sse, _restorer(PHONE), read_size=1000)
    assert out.startswith("id: 7\ndata: ") and texts(out)[0] == "call 9876543210"


def test_contentless_chunks_pass_through():  # Review Focus 4
    sse = ('data: {"choices": [], "usage": {"total_tokens": 5}}\n\n'
           'data: {"choices": [{"index": 0, "delta": {"content": null}}]}\n\n'
           'data: {"choices": [{"index": 0, "delta": {}}]}\n\n'
           'data: {"choices": [{"index": 0, "delta": null, "finish_reason": "stop"}]}\n\n'
           + DONE)
    assert run(sse, _restorer(PHONE), read_size=4) == sse


def test_unrestored_counted_at_finish():
    r = _restorer(PHONE)
    run(chunk(content="call PHO") + chunk(content="NE_1 ok") + DONE, r, read_size=1000)
    assert (r.stats.restored, r.stats.unrestored) == (0, 1)


def test_restored_counted_once():
    r = _restorer(PHONE)
    run(chunk(content="[PHO") + chunk(content="NE_1] and [PHONE_1]") + DONE, r, 1000)
    assert (r.stats.restored, r.stats.unrestored) == (2, 0)


def test_one_char_events_are_fast():  # Review Focus 3
    reply = ("ok [PHONE_1] " * 1600)[:20000]
    sse = "".join(chunk(content=ch) for ch in reply) + DONE
    started = time.perf_counter()
    out = run(sse, _restorer(PHONE), read_size=4096)
    assert time.perf_counter() - started < 1.0
    assert "[PHONE_1]" not in texts(out)[0]
