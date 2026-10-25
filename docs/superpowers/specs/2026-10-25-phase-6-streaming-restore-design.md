# Phase 6 design: streaming restore and tolerant matching

Date: 2026-10-25. Extends §6.6, §7.2 and §9 of
[the main design](2026-10-24-data-leak-guard-design.md); where the two differ, this one wins
for Phase 6.

## 1. Intent

Today a streamed reply reaches the user with placeholders still in it (`[PHONE_1]`): nothing
leaks, but the guard is not yet drop-in for streaming apps, which is most chat apps. Phase 6
puts the real values back **while the reply streams**, and makes restoring robust to the small
ways real LLMs rewrite placeholders (`[Name_1]`, `[NAME 1]`).

For the portfolio, the point is a **measured** claim: how often replies from real LLMs come
back fully restored, streaming and non-streaming, reported as a headline table next to the
detection scores, with the misses counted rather than hidden.

## 2. Goals, non-goals, anti-goals

**Goals**
1. Restore placeholders inside SSE streams, including placeholders split across chunks, in
   every choice (`n > 1`) and in every text field (§4.3).
2. Keep the stream faithful: same events in the same order, provider fields untouched,
   `finish_reason` and `[DONE]` preserved, text delayed only while it could still be a
   placeholder.
3. Tolerant matching shared by both modes (§3).
4. Two new counts on every request log line: `restored` and `unrestored` (§5).
5. A real-LLM measurement on Groq, dev and test splits, whose test numbers go in the README
   (§7).

**Non-goals**
- Guessing. A placeholder the matcher does not recognise is left as the LLM wrote it.
- Restoring inside `logprobs` (they hold placeholder tokens, not values, so nothing leaks;
  they just won't match the restored text). Documented.
- The deprecated `function_call` delta, the Responses API, other endpoints.
- Placeholders with the number written as a word ("Name One"): not matched, and not counted
  as `unrestored` either, because they can't be told apart from ordinary words. A known limit
  of the counter, stated in the report.
- Any change to detection.

**Anti-goals** (would make this a failure even if it "works")
- Restoring a **wrong** value. Worse than not restoring.
- Restoring a placeholder the **user typed** (placeholder injection), in any accepted form.
- Losing text: held text that never comes out because the stream ended oddly.
- A stream an OpenAI SDK can't parse, or one that visibly stalls.

## 3. Tolerant matching

**Accepted from the start**, for types present in this request's mapping only:
- case-insensitive type: `[NAME_1]`, `[Name_1]`, `[name_1]`;
- separator `_`, a single space, or `-`: `[NAME 1]`, `[NAME-1]`;
- bracket pairs `[]`, `{}`, `<>`, `()`, matched (`[NAME_1}` is not accepted);
- the type name as the guard writes it, including `BANK_ACCOUNT` with its inner underscore
  also accepted as a space or `-`.

**Decided on dev, by a pre-stated rule:** bare `NAME_1` with no brackets. Accepted only if, on
the dev run across both models, bracketless forms make up **≥ 2%** of all placeholder mentions
the loose detector (§5) finds. If accepted, it is strict: uppercase type, `_` separator,
word boundaries on both sides. The same rule applies to any other mutation dev reveals: it is
added only if it reaches 2%, is unambiguous, and is covered by the injection guard below.

**Placeholder-injection guard, extended:** `_reserved_numbers` today only sees the exact
`[TYPE_N]` form. It must collect numbers from **every form the restorer accepts** (and from
bracketless forms whenever that form is enabled), so no accepted form can be planted by the
user. Side effect, harmless: a user writing "phone 1 and phone 2" in normal text reserves
`PHONE_1` and `PHONE_2` while bracketless matching is on. Unit-tested per form.

**Restoring into tool-call arguments** must JSON-escape the value
(`json.dumps(value, ensure_ascii=False)[1:-1]`), because arguments are a JSON document
inside a string. A multi-line address restored raw would break that JSON. This also fixes the
current non-streaming code, which restores arguments raw.

## 4. Streaming restore (approach A)

### 4.1 Approaches considered

| | Approach | Trade-off |
|---|---|---|
| **A** ✅ | Parse each SSE event, restore text per field with a hold-back buffer, edit the provider's own events in place | Keeps real streaming and every provider field; most code (SSE parser, per-field state) |
| B | Same buffering, but emit new events the guard builds itself | Slightly simpler output code; easy to silently drop provider fields; stream no longer matches the provider's |
| C | Collect the whole reply, restore, send | Trivially correct; defeats streaming. Kept only as the fallback in §4.5 |
| — | Find-and-replace on the raw bytes | Rejected: a split placeholder has JSON between its halves |

**Chosen: A** (decided by the author).

### 4.2 The per-field restorer

A small object per text field: `feed(text) -> str` returns what is safe to send now;
`flush() -> str` returns everything still held, at the end.

- It holds back the **shortest tail that could still become an accepted placeholder** (e.g.
  `[PH`, `{na`, `[NAME_1` waiting for its closing bracket or more digits). Everything before
  that tail is released at once. With no `[{<(` in sight (or, if bracketless is on, no
  uppercase prefix of a mapped type at a word boundary) nothing is held.
- The hold is bounded by the longest possible placeholder (longest mapped type + brackets +
  separator + up to 6 digits). Past that, the text is released as-is.
- A completed placeholder that is in the mapping (and not reserved) is replaced by its value
  and counted `restored`; anything else is released unchanged.
- Restoring a chunk-split placeholder must give exactly the same text as restoring the whole
  string at once. This is the core property test: for random texts and **every** way of
  cutting them into chunks, `"".join(feed(c) for c in chunks) + flush()` equals the
  non-streaming restore of the whole text.
- The "could this still become a placeholder?" check is hand-written, not the third-party
  `regex` module's partial matching: no new dependency. (Builder may revisit if it gets
  hairy.)

### 4.3 Which fields

Per choice index, one restorer for each of:
- `delta.content`;
- `delta.refusal`;
- reasoning text the provider adds as a delta string field (e.g. `reasoning` on Groq's
  gpt-oss models; the builder confirms the exact names from recorded streams);
- each tool call's `function.arguments`, keyed by the tool call's `index` (JSON-escaped
  restore, §3).

The same fields are restored in non-streaming replies (`message.content`, `message.refusal`,
reasoning fields, tool-call arguments), so both modes behave alike.

### 4.4 Reading and rewriting events

- **SSE parser:** events end at a blank line; an event can be split across network reads, or
  several can arrive in one; handle `\n` and `\r\n`; `data:`, `event:`, `id:` and `:` comment
  lines. Bytes are decoded with an incremental UTF-8 decoder so a character split across reads
  isn't broken.
- An event whose data is a JSON chat chunk is rewritten only if a restorer changed one of its
  fields; otherwise its **original bytes** are sent untouched. Rewritten events are
  re-serialised with `ensure_ascii=False`.
- Held text **rides along in a later event** of the same choice and field. An event may end up
  carrying less text than the provider sent, or an empty string. That is valid for SDKs (to be
  confirmed by an OpenAI-SDK test, like `tests/test_sdk.py`).

### 4.5 Flushing and odd endings

Nothing held is ever lost:
- **`finish_reason` on a choice:** flush that choice's restorers into that same event (add
  the held text to its delta) before it is sent.
- **`[DONE]`, a non-chunk event (e.g. a mid-stream `{"error": ...}`), or the upstream ending
  without `[DONE]`:** if anything is still held, send one extra chunk event carrying it first,
  copying `id`, `object`, `created` and `model` from the last chunk seen. Then the original
  event, if any.
- **Client disconnects:** stop, close upstream (as today), log.
- **A stream reply that isn't SSE** (content type not `text/event-stream`, e.g. a provider
  that ignored `stream`): read fully and restore as non-streaming (approach C).
- **Empty mapping** (nothing was masked): stream passes through byte-for-byte, no parsing.
- **Restorer bug mid-stream:** fail safe. Send the rest of the stream unrestored (placeholders
  only, so nothing leaks) and log a `restore_failed` event (metadata only). Never drop the
  stream.

## 5. Counting restores

Each request log line gains:
- `restored`: placeholders swapped back for a real value;
- `unrestored`: placeholder-looking text left in the reply that refers to a **guard-created**
  placeholder (type and number are in this request's mapping, not reserved) but wasn't in an
  accepted form, e.g. bracketless `NAME_1` while that form is off, or `[NAME_1` with no
  closing bracket.

The "loose detector" behind `unrestored` matches a mapped type name (any case, any of the
separators, inner underscore variants for `BANK_ACCOUNT`) followed by the number, with or
without brackets. Placeholders the LLM invents (`[NAME_7]` when there is no NAME_7) are not
ours and are not counted. Only counts are logged, never values, as for `findings`.

Both modes count the same way. The measurement in §7 reads these counters from the guard
rather than re-implementing them.

In streaming, `unrestored` is counted once per field at the end of the stream, over that
field's full text, because a mangled placeholder can also be split across chunks. The guard
therefore keeps each field's text in memory until the stream ends: request-scoped, like the
mapping itself (main spec §6.2), and discarded afterwards. *(Builder's call: counting on the
text before restoring, with restored spans skipped, so a restored value can never be
miscounted.)*

## 6. Code shape (suggested, builder's call)

- `src/leakguard/restore.py`: the tolerant matcher, the loose detector, the per-field
  streaming restorer, and the non-streaming restore (moved from `masking.py`, which keeps
  `PlaceholderMap` and masking).
- `src/leakguard/sse.py`: the SSE parser and the event rewriter (§4.4–4.5).
- `app.py`: `relay_stream` becomes the restoring relay; `RequestContext` carries the two
  counts; `logs.log_request` takes them as named arguments.

## 7. Real-LLM measurement

**Upstream:** Groq free tier, through the guard (real network, real streaming). The key is
read from the `GROQ_API_KEY` environment variable and never logged or written anywhere.
Measured quota on 2026-10-25: 1,000 requests/day and 8,000 tokens/minute per model.

**Models:** `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` (two model families). Provider-default
temperature, recorded in the results.

**Conversations:** built from the existing dev and test samples (so the dev/test pools stay
disjoint), keeping samples whose masked types include anything the default policy masks (not
SECRET, which blocks). Each sample becomes a support-desk conversation whose instruction makes
the model **repeat details back** (the smoke test showed models otherwise often don't): e.g.
"confirm my details back to me", "draft a reply to this customer that includes their phone
number and address". About a fifth use a tool (e.g. `create_ticket` with name, phone, address
fields) with the tool forced, to measure tool-argument restoring. The instruction wording and
tool schema are the builder's.

**Guard settings:** as shipped (default policy, NER on). Conversations are single-turn (a
system prompt plus one user message); the multi-turn concern of main spec §6.2 stays out of
scope here.

**Sizes:** dev 100 conversations, test 300, each run twice (streaming and non-streaming)
against each model: 200 + 600 requests per model, under the daily quota. The runner paces
itself under the token limit and honours `retry-after` on 429. gpt-oss's reasoning tokens count
against the token limit, so the runner sets a `max_tokens` cap (and a low reasoning effort if
the provider supports it), recorded in the run metadata. If dev has to be re-run after a fix,
the test run goes on a later day so it never competes with dev for the daily quota.

**Rule:** dev decides (bracketless forms, any new mutation, any fix); test is run once at the
end and only reported. Same discipline as detection: never tune on test.

**Outcomes per reply:**
- placeholder mentions found by the loose detector, by form (exact, case, separator, other
  brackets, bracketless, unclosed);
- `restored` and `unrestored` counts from the guard;
- **fully restored** = at least one guard placeholder was mentioned and `unrestored` is 0;
- replies that mention no placeholder at all (refusals, generic answers) are counted and
  reported separately, not mixed into the rate.

**Two steps, so the report is reproducible without a key:**
1. `python -m leakguard.evaluation restore-run --split dev|test` calls Groq and writes a
   committed results file, `data/eval/restore-{split}.json`: per-conversation outcomes
   (counts and forms, no reply text) plus run metadata (date, models, temperature,
   `max_tokens`, guard commit).
2. `python -m leakguard.evaluation restore-report` renders
   `docs/evaluation/restore-report.md` from that file. The README's new table (model × mode:
   replies, fully restored %, mutation rate, `unrestored`) is copied from it, as with the
   detection table.

**Recorded streams as test fixtures:** a handful of raw dev SSE streams from Groq (masked
side, synthetic values only) are saved under `tests/` so CI replays real provider chunking and
fields, with no network and no key.

**Streaming overhead** against the mock: time to first chunk and total time, guard with
restore vs plain pass-through, so the report states what the hold-back costs.

**Honesty notes** for the report: Groq models and their behaviour change over time; runs vary
(sampling); the number-as-word limit of §2; the conversation templates are written by the same
author as the guard.

## 8. Testing (beyond §4.2's property test)

- Unit: each accepted form restores; each rejected form doesn't; reserved numbers are never
  restored in any form; JSON-escaped restore inside tool arguments (value with a newline and a
  quote); hold-back bound; empty mapping passthrough.
- SSE: events split across reads, several per read, `\r\n`, comments, multi-byte characters
  split across reads, `n = 2` interleaved choices, interleaved tool calls, finish-event flush,
  `[DONE]` flush, upstream ending without `[DONE]`, mid-stream error event, non-SSE reply to a
  stream request, restorer exception (fail safe).
- End to end with the mock (3-character chunks) and with the recorded Groq fixtures, through
  the OpenAI SDK.
- Logs contain the new counts and still no text (extend
  `test_logs_contain_no_content_or_auth`).

## 9. Deliverables

Code + tests; `docs/evaluation/restore-report.md` and its results files; README (status line
"Phase 6 of 7", streaming no longer shows placeholders, new results table, `GROQ_API_KEY`
for the measurement); `docs/walkthrough/phase-6-streaming-restore.md`; PR. The private build
journal and interview guide get the real measured numbers.

## 10. Left to the builder

- Module layout (§6), the hold-back check's implementation, SSE parser details within §4.4.
- Conversation instruction wording, the tool schema, which samples are picked within §7.
- Exact runner pacing and retry policy within the quota.
- Which and how many recorded streams become fixtures.
