# Phase 6 — Streaming restore and tolerant matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore placeholders inside streamed replies (and robustly in non-streamed ones), count what could not be restored, and measure the restore rate against real Groq models.

**Architecture:** `placeholders.py` owns the placeholder grammar (accepted forms, loose "looks like ours" forms, the "could this tail still become one?" check). `restore.py` turns a request's `PlaceholderMap` into a `Restorer` that restores whole texts and hands out per-field streaming restorers with a hold-back buffer. `sse.py` parses SSE bytes into events and rewrites chat-chunk events in place through those field restorers. `app.py` wires it in, with byte-for-byte pass-through when nothing was masked and fail-safe pass-through if restoring crashes. The measurement (`evaluation/restore_run.py`, `restore_report.py`) runs the in-process guard against Groq, records raw upstream replies and the guard's counters, and renders a report from a committed results file.

**Tech Stack:** standard library only at runtime (`re`, `json`, `codecs`); httpx, FastAPI as today. No new dependencies (runtime or dev).

**Spec:** [Phase 6 design](../specs/2026-10-25-phase-6-streaming-restore-design.md) (wins for Phase 6) and [main design](../specs/2026-10-24-data-leak-guard-design.md) §6.6, §7.2, §9.

Commands assume the venv at `C:/Users/ayush/.venvs/leakguard` is active (or prefix its `Scripts/python.exe -m`).

## Global Constraints

- Accepted forms (design §3): case-insensitive type; separator `_`, one space, or `-`; matched brackets `[]` `{}` `<>` `()`; `BANK_ACCOUNT`'s inner `_` may also be a space or `-`; 1–6 digits; only types present in the request's mapping.
- Bracketless `NAME_1` is off until the dev run; accepted only if bracketless mentions are **≥ 2%** of all loose mentions on dev across both models. When on: uppercase type, `_` separator, word boundaries both sides. Any other new mutation: same 2% rule, unambiguous, and covered by reservation.
- Reserved numbers come from **every accepted form** (bracketless included only when enabled). A reserved or unknown placeholder is never restored and never counted.
- Never guess: unrecognised text is left as written. Tool-call arguments get the value JSON-escaped (`json.dumps(value, ensure_ascii=False)[1:-1]`).
- Hold-back bound: longest mapped type + 2 brackets + 1 separator + 6 digits.
- Restored text fields: `content`, `refusal`, `reasoning`, `reasoning_content`, and tool-call `function.arguments`, in both modes. Not `logprobs`, not `function_call`.
- Logs: per-request `restored` and `unrestored` counts only; never text, values or keys. New event `restore_failed` (metadata only).
- Measurement: Groq; models `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`; dev 100 / test 300 conversations, each streaming and non-streaming per model; key only from `GROQ_API_KEY`; results in `data/eval/restore-{split}.json` with no reply text; dev decides, test is run once and only reported.
- No new runtime dependencies. Python 3.11, ruff line length 100.

## Review Focus

1. **Ordinary brackets in replies** (markdown links `[docs](http://x)`, `arr[0]`, `(1)`, `<br>`, `{x}`, JSON): pass through unchanged and held only a few characters. Test: `test_ordinary_brackets_pass_through` (Task 2).
2. **Adjacent and glued placeholders** (`[NAME_1][PHONE_1]`, `[PHONE_1]23`, the same placeholder 50 times): each restored, nothing merged or skipped. Test: `test_adjacent_and_repeated_placeholders` (Task 2).
3. **Pathological chunking** (a 20,000-character reply sent one character per event, or in one event): linear, finishes in < 1 s. Test: `test_one_char_events_are_fast` (Task 3).
4. **Chunks without content** (`"choices": []` usage chunk, `delta: {}`, `"content": null`): passed through byte-identical, no crash. Test: `test_contentless_chunks_pass_through` (Task 3).
5. **Non-ASCII values split mid-character across network reads** (a Devanagari name, an address with a newline and quotes inside tool arguments): restored correctly. Test: `test_hindi_value_restored_across_byte_split` (Task 4).

---

## File Structure

```
src/leakguard/placeholders.py          # grammar: accepted / loose mentions, hold-back check     (Task 1)
src/leakguard/masking.py               # reservation via placeholders.py; value_for(); restore code removed (Tasks 1, 2)
src/leakguard/restore.py               # Restorer, FieldRestorer, restore_response               (Task 2)
src/leakguard/sse.py                   # SSEParser, StreamRestorer                                (Task 3)
src/leakguard/app.py, logs.py          # restoring relay, fallbacks, counters                     (Task 4)
src/leakguard/evaluation/restore_run.py      # conversations, Groq runner, results file           (Task 5)
src/leakguard/evaluation/restore_report.py   # summaries, report, overhead benchmark              (Task 6)
src/leakguard/evaluation/__main__.py         # restore-run, restore-report subcommands            (Tasks 5, 6)
data/eval/restore-dev.json, restore-test.json, docs/evaluation/restore-tuning.md, restore-report.md (Tasks 7, 8)
tests/fixtures/groq_streams/*.json     # recorded real streams                                    (Task 7)
README.md, docs/walkthrough/phase-6-streaming-restore.md                                          (Tasks 8, 9)
tests/test_placeholders.py, test_restore.py, test_sse.py, test_streaming.py (extended),
tests/test_recorded_streams.py, tests/evaluation/test_restore_run.py, test_restore_report.py
```

---

### Task 1: Placeholder grammar and reservation

**Files:**
- Create: `src/leakguard/placeholders.py`
- Modify: `src/leakguard/masking.py` (`PLACEHOLDER_RE` → placeholders.py; `_reserved_numbers`; `PlaceholderMap.value_for`, `PlaceholderMap.entities`)
- Test: `tests/test_placeholders.py`, `tests/test_masking.py`

**Interfaces:**
- Produces:
  - `ACCEPT_BRACKETLESS: bool = False` (module constant; flipped only by Task 7's rule)
  - `MAX_DIGITS = 6`; `BRACKETS = {"[": "]", "{": "}", "<": ">", "(": ")"}`
  - `@dataclass(frozen=True) class Mention: start: int; end: int; entity: EntityType; number: int; form: str` with form in `"exact" | "case" | "separator" | "brackets" | "bracketless" | "unclosed"`
  - `accepted_mentions(text: str, entities: Collection[EntityType], bracketless: bool = ACCEPT_BRACKETLESS) -> list[Mention]` (left to right, non-overlapping)
  - `loose_mentions(text: str, entities: Collection[EntityType]) -> list[Mention]`: everything placeholder-looking: any case, any separator, with or without brackets, closed or not; one mention per placeholder (a bracketed match wins over the bare name inside it). Form = most severe trait, in order bracketless/unclosed > brackets > separator > case > exact.
  - `could_become_placeholder(tail: str, entities: Collection[EntityType], bracketless: bool = ACCEPT_BRACKETLESS) -> bool`: True iff `tail` is a proper prefix of some accepted form, or (bracketless) a complete bare form that more digits could extend.
  - `max_placeholder_len(entities: Collection[EntityType]) -> int`
  - `PlaceholderMap.value_for(entity: EntityType, number: int) -> str | None` (None for reserved or never-created numbers); `PlaceholderMap.entities() -> set[EntityType]` (types it created)

- [ ] **Step 1: Write the failing tests**

```python
ALL = set(EntityType)

@pytest.mark.parametrize("text", ["[NAME_1]", "[Name_1]", "[name_1]", "[NAME 1]", "[NAME-1]",
                                  "{NAME_1}", "<NAME_1>", "(NAME_1)", "[BANK ACCOUNT_1]",
                                  "[bank-account 1]"])
def test_accepted_forms(text):
    (m,) = accepted_mentions(text, ALL, bracketless=False)
    assert (m.start, m.end, m.number) == (0, len(text), 1)

@pytest.mark.parametrize("text", ["NAME_1", "[NAME_1}", "[NAME__1]", "[NAMES_1]",
                                  "[NAME_1234567]", "[NAME_]", "[ NAME_1 ]"])
def test_rejected_forms(text):
    assert accepted_mentions(text, ALL, bracketless=False) == []

def test_bracketless_is_strict_when_enabled():
    assert [m.number for m in accepted_mentions("call NAME_1 now", ALL, bracketless=True)] == [1]
    for text in ["call name_1 now", "call NAME 1 now", "XNAME_1", "NAME_1X"]:
        assert accepted_mentions(text, ALL, bracketless=True) == []

def test_only_mapped_entities():
    assert accepted_mentions("[PHONE_1]", {EntityType.NAME}, bracketless=False) == []

def test_loose_forms():
    text = "[NAME_1] [name_1] [NAME 1] {NAME_1} NAME_1 [NAME_1 x"
    assert [m.form for m in loose_mentions(text, {EntityType.NAME})] == [
        "exact", "case", "separator", "brackets", "bracketless", "unclosed"]

@pytest.mark.parametrize("tail", ["[", "[n", "[PHO", "[PHONE_", "[PHONE_1", "{bank acc"])
def test_could_become(tail):
    assert could_become_placeholder(tail, ALL, bracketless=False)

@pytest.mark.parametrize("tail", ["[x", "[PHONE_1]", "[PHONEX", "[PHONE_1234567", "NAM"])
def test_cannot_become(tail):
    assert not could_become_placeholder(tail, ALL, bracketless=False)

def test_bare_prefix_held_only_when_bracketless():
    assert could_become_placeholder("NAM", ALL, bracketless=True)
    assert could_become_placeholder("NAME_1", ALL, bracketless=True)  # NAME_12 still possible
```

In `tests/test_masking.py`:

```python
def test_reserved_numbers_cover_every_accepted_form():
    r = mask_messages(_user("what is [phone 1] and {PHONE-2}? my number is 9876543210"),
                      DEFAULT_POLICY)
    assert r.messages[0]["content"].endswith("my number is [PHONE_3]")
    assert r.mapping.value_for(T.PHONE, 1) is None and r.mapping.value_for(T.PHONE, 3)
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `pytest tests/test_placeholders.py tests/test_masking.py -v`
Expected: FAIL with `ModuleNotFoundError: leakguard.placeholders`

- [ ] **Step 3: Implement `placeholders.py`; switch `_reserved_numbers` to `accepted_mentions(text, set(EntityType))`; add `value_for` / `entities`**

Build the matching regexes per call from the entity set (type alternatives longest first, inner `_` → `[_ -]`, `re.IGNORECASE` for bracketed forms). `could_become_placeholder` is hand-written (no `regex` module): walk opener → type-name prefix (case-insensitive) → separator → ≤ 6 digits → closer, and fail as soon as a character can't continue any type. Keep `PlaceholderMap.restore` for now (Task 2 removes it).

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `pytest -q`
Expected: all pass, including the existing `test_user_typed_placeholder_is_never_restored`.

- [ ] **Step 5: Commit**

```bash
git add src/leakguard/placeholders.py src/leakguard/masking.py tests/test_placeholders.py tests/test_masking.py
git commit -m "feat: tolerant placeholder grammar; reserve numbers in every accepted form"
```

---

### Task 2: Restorer and per-field streaming restorer

**Files:**
- Create: `src/leakguard/restore.py`
- Modify: `src/leakguard/masking.py` (remove `PlaceholderMap.restore`, `restore_response`), `src/leakguard/app.py` (`_restored_content` uses the new `restore_response`), `tests/test_masking.py` (move restore tests to `tests/test_restore.py`)
- Test: `tests/test_restore.py`

**Interfaces:**
- Consumes: Task 1's `accepted_mentions`, `loose_mentions`, `could_become_placeholder`, `max_placeholder_len`, `PlaceholderMap.value_for`, `PlaceholderMap.entities`
- Produces:
  - `TEXT_FIELDS = ("content", "refusal", "reasoning", "reasoning_content")`
  - `@dataclass class RestoreStats: restored: int = 0; unrestored: int = 0`
  - `class Restorer(mapping: PlaceholderMap, bracketless: bool = ACCEPT_BRACKETLESS)` with `.stats: RestoreStats`, `.restore_text(text: str, json_escape: bool = False) -> str` (restores and adds both counts), `.field(json_escape: bool = False) -> FieldRestorer`
  - `class FieldRestorer` with `.feed(text: str) -> str`, `.flush() -> str` (releases everything held; may be called more than once), `.close() -> None` (adds `unrestored` for the field's full raw text to the stats; call once, at stream end)
  - `restore_response(body: dict, restorer: Restorer) -> dict` (every `TEXT_FIELDS` string of `choices[].message` plus tool-call arguments, JSON-escaped)

Counting rule: `unrestored` = loose mentions whose `(entity, number)` has a `value_for` and that don't overlap an accepted mention that was restored. Restored counts accrue in `feed`; `close` must not recount them.

- [ ] **Step 1: Write the failing tests**

```python
def _restorer(*pairs):  # pairs of (EntityType, value), numbered in order
    m = PlaceholderMap()
    for entity, value in pairs:
        m.placeholder_for(entity, value)
    return Restorer(m, bracketless=False)

def test_restore_text_counts_every_form():
    r = _restorer((T.PHONE, "9876543210"))
    assert r.restore_text("call [phone 1] or {PHONE_1}") == "call 9876543210 or 9876543210"
    assert (r.stats.restored, r.stats.unrestored) == (2, 0)

def test_reserved_never_restored_in_any_form():
    res = mask_messages(_user("what is [PHONE_1]? my number is 9876543210"), DEFAULT_POLICY)
    r = Restorer(res.mapping, bracketless=False)
    assert r.restore_text("[phone 1] {PHONE-2}") == "[phone 1] 9876543210"

def test_unknown_placeholder_left_and_not_counted():
    r = _restorer((T.PHONE, "9876543210"))
    assert r.restore_text("[EMAIL_1] [PHONE_9]") == "[EMAIL_1] [PHONE_9]"
    assert (r.stats.restored, r.stats.unrestored) == (0, 0)

def test_unrestored_counts_bare_and_unclosed():
    r = _restorer((T.PHONE, "9876543210"))
    assert r.restore_text("call PHONE_1 or [PHONE_1 now") == "call PHONE_1 or [PHONE_1 now"
    assert r.stats.unrestored == 2

def test_json_escaped_into_tool_arguments():
    address = 'Flat 2,\n"Sai" Apts, Kothrud, Pune 411038'
    r = _restorer((T.ADDRESS, address))
    out = r.restore_text('{"a": "[ADDRESS_1]"}', json_escape=True)
    assert json.loads(out)["a"] == address

def test_restore_response_all_fields():  # content, refusal, reasoning, tool args
    ...  # one choice using all four; assert each restored, tool args still valid JSON

def test_ordinary_brackets_pass_through():          # Review Focus 1
    r = _restorer((T.NAME, "Ravi"))
    text = "See [docs](http://x), arr[0], (1), <br>, {x}, [NOTE_1] and {\"k\": [1]}"
    f = r.field()
    assert f.feed(text) + f.flush() == text
    assert len(text) - len(r.field().feed(text)) <= max_placeholder_len({T.NAME})

def test_adjacent_and_repeated_placeholders():     # Review Focus 2
    r = _restorer((T.NAME, "Ravi"), (T.PHONE, "9876543210"))
    assert r.restore_text("[NAME_1][PHONE_1]23 " + "[NAME_1]" * 50) == \
        "Ravi987654321023 " + "Ravi" * 50

TEXTS = ["call [PHONE_1] now", "[Name 1], hi {NAME_1}!", "x[PH", "[[NAME_1]]", "a [NAME_1",
         "NAME_1 [phone-1]", "tail [PHONE_1]", ""]

@pytest.mark.parametrize("text", TEXTS)
def test_streaming_equals_whole_text_for_every_cut(text):
    expected_restorer = _restorer((T.NAME, "Ravi"), (T.PHONE, "9876543210"))
    expected = expected_restorer.restore_text(text)
    for cuts in _all_two_cut_chunkings(text) + _random_chunkings(text, seed=0, n=50):
        r = _restorer((T.NAME, "Ravi"), (T.PHONE, "9876543210"))
        f = r.field()
        out = "".join(f.feed(c) for c in cuts) + f.flush()
        f.close()
        assert out == expected
        assert r.stats == expected_restorer.stats

def test_nothing_held_without_opener():
    assert _restorer((T.NAME, "Ravi")).field().feed("hello world") == "hello world"

def test_hold_is_bounded():
    f = _restorer((T.PHONE, "9876543210")).field()
    assert f.feed("ok [PHONE_123") == "ok "
    assert f.feed("4567") == "[PHONE_1234567"   # 7 digits: can never be a placeholder
```

(`_all_two_cut_chunkings` and `_random_chunkings` are small helpers in the test file.)

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `pytest tests/test_restore.py -v`
Expected: FAIL with `ModuleNotFoundError: leakguard.restore`

- [ ] **Step 3: Implement `restore.py`, move non-streaming restore into it, update `app.py` and `test_masking.py`**

`FieldRestorer.feed`: append to the held buffer and to the raw text; find the cut = the earliest index within the last `max_placeholder_len` characters where `could_become_placeholder(buffer[i:])` is true (candidate positions: opener characters, plus word starts when bracketless); restore `buffer[:cut]` using the whole buffer for boundary checks but only replacing mentions that end ≤ cut; release that, hold the rest. The property test is the arbiter.

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/leakguard/restore.py src/leakguard/masking.py src/leakguard/app.py tests/test_restore.py tests/test_masking.py
git commit -m "feat: tolerant restorer with streaming hold-back and restore counts"
```

---

### Task 3: SSE parsing and in-place event rewriting

**Files:**
- Create: `src/leakguard/sse.py`
- Test: `tests/test_sse.py`

**Interfaces:**
- Consumes: Task 2's `Restorer`, `FieldRestorer`, `TEXT_FIELDS`
- Produces:
  - `@dataclass(frozen=True) class SSEEvent: raw: str; data: str | None` (`raw` = the event's exact text including its terminating blank line; `data` = joined `data:` lines, or None)
  - `class SSEParser` with `.feed(chunk: bytes) -> list[SSEEvent]` and `.close() -> list[SSEEvent]` (a trailing event without a blank line). Incremental UTF-8 decoding (`codecs.getincrementaldecoder("utf-8")(errors="replace")`); accepts `\n` and `\r\n`.
  - `class StreamRestorer(restorer: Restorer)` with `.process(event: SSEEvent) -> list[str]` and `.finish() -> list[str]` (flush held text as one synthetic event, then `close()` every field restorer)

Rules (design §4.4–4.5): field restorers keyed by `(choice index, field)` and `(choice index, "tool", tool index)` (tool args use `json_escape=True`). An event whose fields all come out unchanged is emitted as its original `raw`. A changed event is re-serialised: its non-`data` lines kept in order, then `data: ` + `json.dumps(obj, ensure_ascii=False)`, then a blank line. On `finish_reason` for a choice, flush that choice's restorers into the same event (append to `delta[field]`, or add a tool-call entry `{"index": i, "function": {"arguments": held}}`). Before `[DONE]` or any non-chunk event, and in `finish()`, if anything is held, emit one synthetic chunk `{"id", "object", "created", "model"}` copied from the last chunk seen, with `choices: [{"index": i, "delta": {...}, "finish_reason": None}, ...]`.

- [ ] **Step 1: Write the failing tests**

Helpers in the test file: `chunk(content=None, index=0, finish=None, tools=None, **delta) -> str` (one `data:` event), `run(sse: str, restorer, read_size: int) -> str` (feeds bytes in `read_size` pieces through parser and restorer), `texts(out: str) -> dict[int, str]` (content per choice).

```python
def test_parser_handles_splits_crlf_comments_and_multibyte():
    raw = ": ping\r\n\r\n" + 'data: {"x": "नमस्ते"}\r\n\r\n' + "data: [DONE]\n\n"
    for size in (1, 2, 3, 7, len(raw.encode())):
        events = _parse_in_pieces(raw.encode(), size)
        assert [e.data for e in events] == [None, '{"x": "नमस्ते"}', "[DONE]"]
        assert "".join(e.raw for e in events) == raw

def test_parser_close_returns_trailing_event():
    p = SSEParser(); p.feed(b"data: [DONE]")
    assert [e.data for e in p.close()] == ["[DONE]"]

def test_split_placeholder_restored_three_char_chunks():
    sse = "".join(chunk(content=c) for c in _cut("I'll call [PHONE_1] today", 3)) + DONE
    assert texts(run(sse, _restorer(PHONE), read_size=5))[0] == "I'll call 9876543210 today"

def test_two_choices_interleaved(): ...      # n=2, each choice its own placeholder split
def test_tool_calls_interleaved(): ...       # tool index 0 and 1, args split mid-placeholder;
                                             # each reassembled args json.loads to restored values
def test_held_text_flushed_into_finish_event():
    sse = chunk(content="call [PH") + chunk(finish="stop") + DONE
    out = run(sse, _restorer(PHONE), read_size=1000)
    assert texts(out)[0] == "call [PH" and out.count("data: ") == 3   # no extra event

def test_held_text_flushed_before_done(): ...           # no finish event; one synthetic event
                                                         # carrying id/model of the last chunk
def test_upstream_ends_without_done(): ...              # finish() emits the synthetic event
def test_error_event_mid_stream(): ...                  # synthetic flush, then the error raw
def test_unchanged_events_are_byte_identical(): ...     # events with no placeholders: out == in
def test_contentless_chunks_pass_through():             # Review Focus 4
    sse = 'data: {"choices": [], "usage": {"total_tokens": 5}}\n\n' + \
          chunk(**{"content": None}) + 'data: {"choices": [{"index": 0, "delta": {}}]}\n\n' + DONE
    assert run(sse, _restorer(PHONE), read_size=4) == sse
def test_unrestored_counted_at_finish(): ...            # "PHONE_1" split across chunks -> 1
def test_one_char_events_are_fast():                    # Review Focus 3
    reply = ("ok [PHONE_1] " * 1500)[:20000]
    sse = "".join(chunk(content=ch) for ch in reply) + DONE
    started = time.perf_counter()
    run(sse, _restorer(PHONE), read_size=4096)
    assert time.perf_counter() - started < 1.0
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `pytest tests/test_sse.py -v`
Expected: FAIL with `ModuleNotFoundError: leakguard.sse`

- [ ] **Step 3: Implement `sse.py`**

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/leakguard/sse.py tests/test_sse.py
git commit -m "feat: SSE parser and in-place chunk rewriting with flush on finish and DONE"
```

---

### Task 4: Wire restoring into the guard

**Files:**
- Modify: `src/leakguard/app.py` (`relay_stream`, `_stream_and_log`, `_proxy_chat`, `RequestContext`), `src/leakguard/logs.py`
- Test: `tests/test_streaming.py`, `tests/test_guard_masking.py`, `tests/test_sdk.py`

**Interfaces:**
- Consumes: Task 2's `Restorer`, `restore_response`; Task 3's `SSEParser`, `StreamRestorer`
- Produces:
  - `relay_stream(resp: httpx.Response, restorer: Restorer | None = None, on_restore_failed: Callable[[], None] | None = None) -> AsyncIterator[bytes]` (still closes upstream in `finally`; `restorer is None` → bytes unchanged)
  - `RequestContext.restored: int = 0`, `RequestContext.unrestored: int = 0`, passed to `log_request(..., restored: int = 0, unrestored: int = 0)`
  - `log_restore_failed(*, request_id: str) -> None` → `{"event": "restore_failed", "request_id": ...}`

Behaviour: `_proxy_chat` builds `Restorer(mapping)` when the mapping is non-empty, else None. Successful stream reply whose content type isn't `text/event-stream` → read fully and relay restored like a non-stream reply. On any exception inside `StreamRestorer.process`/`finish`: call `on_restore_failed`, emit the failing event's `raw`, then emit every later event's `raw` unchanged (held text up to the hold-back bound may be lost in that bug path; documented in the walkthrough). The counts are copied from `restorer.stats` into `ctx` before `ctx.finish`.

- [ ] **Step 1: Write the failing tests**

```python
def test_stream_restores_placeholders_with_mock(guard):        # mock: 3-char chunks
    text = _stream_text(guard, "my phone is 9876543210")
    assert text == "You said: my phone is 9876543210"

def test_openai_sdk_stream_restored(guard): ...                # in test_sdk.py

def test_stream_without_findings_is_byte_identical(make_guard):
    body = b'data: {"choices":[{"delta":{"content":"[PH"}}]}\r\n\r\n: x\n\ndata: [DONE]\n\n'
    ...  # MockTransport returns body; request has no PII; streamed bytes == body

def test_non_sse_reply_to_stream_request_restored(make_guard): ...  # JSON content type

def test_restore_failure_is_fail_safe(make_guard, monkeypatch, caplog):
    monkeypatch.setattr(StreamRestorer, "process", _crash)
    text = _stream_text(make_guard(), "my phone is 9876543210")
    assert "[PHONE_1]" in text and "9876543210" not in caplog.text
    assert '"event": "restore_failed"' in caplog.text

def test_logs_restored_and_unrestored_counts(make_guard, caplog): ...
    # MockTransport replying "[PHONE_1] and PHONE_1": both modes log
    # "restored": 1, "unrestored": 1

def test_hindi_value_restored_across_byte_split(make_guard):   # Review Focus 5
    # fake name model tags "रवि"; upstream streams content "[NAME_1] ji" and tool args
    # '{"address": "[ADDRESS_1]"}' as bytes cut every 2 bytes (mid-character);
    # assert content == "रवि ji" and json.loads(args)["address"] == the multi-line address
```

Extend `test_logs_contain_no_content_or_auth` so a streamed, restored request is also checked.

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `pytest tests/test_streaming.py tests/test_guard_masking.py tests/test_sdk.py -v`
Expected: the new tests FAIL (stream still shows `[PHONE_1]`; no counts in logs).

- [ ] **Step 3: Implement the wiring**

- [ ] **Step 4: Run the full suite and lint**

Run: `pytest -q && ruff check .`
Expected: all pass, no lint errors.

- [ ] **Step 5: Smoke test against Groq through the real server**

Run the guard on port 8081 (8080 is taken by the user's compose stack) with `LEAKGUARD_UPSTREAM_BASE_URL=https://api.groq.com/openai/v1`, then `examples/chat.py` with `OPENAI_API_KEY` from the user-level `GROQ_API_KEY` and `LEAKGUARD_MODEL=openai/gpt-oss-20b`, asking it to repeat the phone number back. Expected: streamed reply shows the real number if the model repeats it; the log line has `restored`/`unrestored`, no text.

- [ ] **Step 6: Commit**

```bash
git add src/leakguard/app.py src/leakguard/logs.py tests/
git commit -m "feat: restore placeholders in streamed replies; log restore counts"
```

---

### Task 5: Real-LLM measurement runner

**Files:**
- Create: `src/leakguard/evaluation/restore_run.py`
- Modify: `src/leakguard/evaluation/__main__.py` (subcommand `restore-run`)
- Test: `tests/evaluation/test_restore_run.py`

**Interfaces:**
- Consumes: `load_split`, `load_detector` from `evaluation/__main__.py` (move them to a small `evaluation/loading.py` if importing `__main__` gets awkward); `create_app`; `loose_mentions`; `SSEParser`; `mask_messages`, `build_detector`
- Produces:
  - `MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b")`; `SIZES = {"dev": 100, "test": 300}`; `GROQ_BASE_URL = "https://api.groq.com/openai/v1"`
  - `@dataclass(frozen=True) class Conversation: id: str; sample_id: str; messages: list[dict]; tools: list[dict] | None; tool_choice: dict | None`
  - `build_conversations(split: str, n: int, seed: int) -> list[Conversation]`: samples from that split whose gold types are masked by the default policy and include no `SECRET`; deterministic; about 20% use the forced `create_ticket` tool (fields name, phone, email, address, notes); the rest end with an instruction that asks the model to repeat the details back
  - `async run_conversations(conversations, models, transport: httpx.AsyncBaseTransport, api_key: str, max_tokens: int, save_streams: int = 0) -> dict` → `{"meta": {...}, "outcomes": [...]}`
  - Outcome keys: `conversation`, `model`, `stream`, `status`, `mentions` (form → count, from the **raw upstream** reply via a recording transport wrapper), `restored`, `unrestored` (from the guard's log line, matched by `x-request-id`), `mentioned` (bool)
  - `meta`: date, split, models, `max_tokens`, temperature (`"provider default"`), guard commit (`git rev-parse --short HEAD`)
  - CLI: `python -m leakguard.evaluation restore-run --split dev|test [--limit N] [--save-streams N]` → writes `data/eval/restore-{split}.json`; exits with a clear message if `GROQ_API_KEY` is unset

Pacing: after each reply, read the relayed `x-ratelimit-remaining-tokens` / `x-ratelimit-reset-tokens`; when remaining < 1,500, sleep until reset. On 429, sleep `retry-after` and retry, at most 5 times; then record `status: 429` and move on. The in-process guard needs its lifespan entered (`async with app.router.lifespan_context(app)`) before calls through `httpx.ASGITransport`.

- [ ] **Step 1: Write the failing tests** (fake Groq via `httpx.MockTransport`, no network)

```python
def test_conversations_are_deterministic_and_safe():
    a, b = build_conversations("dev", 100, seed=1), build_conversations("dev", 100, seed=1)
    assert a == b and len(a) == 100
    assert 15 <= sum(c.tools is not None for c in a) <= 25
    # no SECRET values and no blocked samples: every conversation's sample masks cleanly

def test_dev_and_test_use_their_own_samples(): ...   # sample ids disjoint

async def test_outcomes_count_forms_and_guard_counters():
    # fake upstream echoes "[PHONE_1] / phone 1 / {PHONE_1}" (non-stream) and the same text
    # streamed in 3-char chunks; assert mentions == {"exact": 1, "bracketless": 1, "brackets": 1}
    # and restored == 2, unrestored == 1 for both modes

async def test_429_is_retried(): ...                  # first reply 429 retry-after 0, then 200

async def test_results_contain_no_text_values_or_key(): ...
    # json.dumps(result) contains neither the sample's values, nor reply text, nor the key

def test_missing_key_exits_clearly(monkeypatch): ...
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `pytest tests/evaluation/test_restore_run.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `restore_run.py` and the subcommand**

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `pytest -q && ruff check .`
Expected: all pass.

- [ ] **Step 5: Live mini-run (3 conversations, both models)**

Run: `python -m leakguard.evaluation restore-run --split dev --limit 3` with `GROQ_API_KEY` set from the user variable. Expected: 12 outcomes, `status` 200, file written. Delete the file afterwards (Task 7 writes the real one); note in the build journal how many requests it used.

- [ ] **Step 6: Commit**

```bash
git add src/leakguard/evaluation/ tests/evaluation/test_restore_run.py
git commit -m "feat: Groq restore-rate runner with recorded raw replies and guard counters"
```

---

### Task 6: Restore report and streaming overhead

**Files:**
- Create: `src/leakguard/evaluation/restore_report.py`
- Modify: `src/leakguard/evaluation/__main__.py` (subcommand `restore-report`)
- Test: `tests/evaluation/test_restore_report.py`

**Interfaces:**
- Consumes: the results dict from Task 5
- Produces:
  - `@dataclass class Summary: replies: int; mentioned: int; fully_restored: int; no_mention: int; errors: int; restored: int; unrestored: int; forms: Counter`
  - `summarize(results: dict) -> dict[tuple[str, bool], Summary]` keyed by `(model, stream)`; fully restored = `mentioned and unrestored == 0` among `status == 200`
  - `bracketless_share(results: dict) -> float` = bracketless mentions / all mentions, across both models and modes
  - `measure_stream_overhead(n: int = 200) -> dict[str, dict[str, float]]`: against the in-process mock, `{"pass-through": {...}, "restoring": {...}}` with `ttfb_p50`, `ttfb_p95`, `total_p50`, `total_p95` in ms (pass-through = a message with no PII; restoring = the same length with a phone number)
  - `render_restore_report(results: dict, overhead: dict, notes: list[str]) -> str`
  - `render_restore_tuning(results: dict) -> str`: dev summary, the form table, `bracketless_share`, and the 2% rule's verdict in words
  - CLI: `restore-report --split dev` → `docs/evaluation/restore-tuning.md`; `restore-report --split test` → `docs/evaluation/restore-report.md` (includes the overhead section and NOTES: models change over time, sampling varies, number-as-word limit, same-author templates, bracketless decision made on dev)

- [ ] **Step 1: Write the failing tests**

```python
def test_summarize_rates(): ...   # 4 hand-built outcomes: mentioned+0 unrestored, mentioned+1,
                                  # no mention, status 429 -> fully_restored 1, mentioned 2,
                                  # no_mention 1, errors 1
def test_bracketless_share(): ... # forms {"exact": 98, "bracketless": 2} -> 0.02
def test_tuning_verdict_at_threshold(): ...  # 0.02 -> "accept"; 0.0199 -> "keep off"
def test_report_has_table_and_notes(): ...   # "| Model | Mode | Replies |" header, each note
def test_overhead_keys(): ...                # measure_stream_overhead(n=5) keys and floats > 0
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `pytest tests/evaluation/test_restore_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `pytest -q && ruff check .`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/leakguard/evaluation/ tests/evaluation/test_restore_report.py
git commit -m "feat: restore report, dev tuning page and streaming overhead benchmark"
```

---

### Task 7: Dev run, the bracketless decision, recorded fixtures

Needs `GROQ_API_KEY`. Uses about 200 requests per model.

**Files:**
- Create: `data/eval/restore-dev.json`, `docs/evaluation/restore-tuning.md`, `tests/fixtures/groq_streams/*.json`, `tests/test_recorded_streams.py`
- Modify (only if the rule says so): `src/leakguard/placeholders.py` (`ACCEPT_BRACKETLESS`), tests

- [ ] **Step 1: Run dev**

Run: `python -m leakguard.evaluation restore-run --split dev --save-streams 6` (long job: launch detached and watch, as in Phase 4). Expected: 400 outcomes, almost all `status` 200; six fixture files, each `{"model", "placeholders": {"[PHONE_1]": "<synthetic value>", ...}, "sse": "<raw upstream stream>"}`.

- [ ] **Step 2: Render the tuning page and apply the 2% rule**

Run: `python -m leakguard.evaluation restore-report --split dev`. Read `restore-tuning.md`. If the verdict is "accept": set `ACCEPT_BRACKETLESS = True`, add a test that bracketless restores and that bare `PHONE_1` typed by a user is reserved, and re-run `pytest -q`. If dev shows another mutation ≥ 2% that is unambiguous, add it to the grammar the same way (with a reservation test); otherwise change nothing. The decision and its numbers stay recorded in `restore-tuning.md`.

- [ ] **Step 3: Write the recorded-stream test**

```python
@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")))
def test_recorded_groq_stream(path):
    fx = json.loads(path.read_text(encoding="utf-8"))
    restorer = _restorer_from(fx["placeholders"])      # placeholder_for in number order;
                                                        # asserts the numbers match
    out = _run(fx["sse"], restorer, read_size=7)
    raw_text = _field_texts(fx["sse"])                  # per (choice, field) raw text
    expected = {k: _restorer_from(fx["placeholders"]).restore_text(v)
                for k, v in raw_text.items()}
    assert _field_texts(out) == expected                # streaming == whole-text restore
    _parse_with_openai_sdk(out)                         # every event parses as a ChatCompletionChunk
```

- [ ] **Step 4: Run the suite**

Run: `pytest -q && ruff check .`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add data/eval/restore-dev.json docs/evaluation/restore-tuning.md tests/fixtures tests/test_recorded_streams.py src/leakguard/placeholders.py tests/
git commit -m "docs: Groq dev restore run and bracketless decision; recorded streams as fixtures"
```

---

### Task 8: Test run, report, README

Needs `GROQ_API_KEY`. Uses about 600 requests per model. If Task 7 needed a re-run of dev today, do this on a later day.

- [ ] **Step 1: Run test once**

Run: `python -m leakguard.evaluation restore-run --split test` (detached, watched). Expected: 1,200 outcomes. Do not change any code or setting based on these results.

- [ ] **Step 2: Render the report**

Run: `python -m leakguard.evaluation restore-report --split test`. Expected: `docs/evaluation/restore-report.md` with the model × mode table, forms, no-mention counts, overhead, notes.

- [ ] **Step 3: Update the README**

Status line → "Phase 6 of 7" (streamed replies are restored); remove "still show placeholders until Phase 6"; add a "Restoring replies on real LLMs" table copied from the report (model, mode, replies, fully restored %, mutation rate, unrestored) with a link to the report; mention `GROQ_API_KEY` only for the measurement command; describe `restored`/`unrestored` in the logging section.

- [ ] **Step 4: Run the suite and commit**

Run: `pytest -q && ruff check .`

```bash
git add data/eval/restore-test.json docs/evaluation/restore-report.md README.md
git commit -m "docs: restore-rate report on Groq test run; README results"
```

---

### Task 9: Walkthrough, review, PR

- [ ] **Step 1: Write `docs/walkthrough/phase-6-streaming-restore.md`**: what/why, each new file, the hold-back idea with a worked `[PH` + `ONE_1]` example, flush rules, fail-safe path and its documented text-loss bound, tolerant forms and the injection guard, the measurement and its honesty notes, likely interview questions with answers.
- [ ] **Step 2: Final review** of the whole branch (fresh reviewer on the most capable model), fix findings, re-run `pytest -q && ruff check .`; if a fix touches restoring, re-render nothing from test (record the fix in the report NOTES instead).
- [ ] **Step 3: Push, open the PR, watch CI** (`gh pr create`, `gh pr checks --watch`). The user merges.
- [ ] **Step 4: Private docs**: update `data-leak-guard-private/build-journal.md` (what happened, why, what the user did: Groq account, key) and fill `interview-guide.md` with the measured restore numbers. Remind the user to rotate the Groq key.
