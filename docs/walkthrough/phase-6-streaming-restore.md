# Phase 6 Walkthrough — Restoring streamed replies

> **What Phase 6 builds:** real values put back into **streamed** replies while they stream, robust to the small ways LLMs rewrite placeholders, with every reply's `restored` / `unrestored` counts in the log, and a measured restore rate on two real models (Groq free tier).

Design for this phase: [2026-10-25-phase-6-streaming-restore-design.md](../superpowers/specs/2026-10-25-phase-6-streaming-restore-design.md). Plan: [2026-10-25-phase-6-streaming-restore.md](../superpowers/plans/2026-10-25-phase-6-streaming-restore.md).

---

## 0. Why streaming makes restoring hard

Non-streaming is easy: the whole reply arrives, you find `[PHONE_1]`, you swap in `9876543210`. Done in Phase 2.

Streaming sends the reply as many small **events** (SSE, see Phase 1), each carrying a few characters. A placeholder can be cut anywhere:

```
data: {"choices":[{"index":0,"delta":{"content":"I'll call [PH"}}]}
data: {"choices":[{"index":0,"delta":{"content":"ONE_1] tomo"}}]}
data: {"choices":[{"index":0,"delta":{"content":"rrow"}}]}
data: [DONE]
```

Two things can go wrong:
1. Send `[PH` as it comes, and you can never take it back: neither event holds a whole placeholder, so the user sees `[PHONE_1]` arrive in two halves, unrestored.
2. Wait for the whole reply, and you've thrown away streaming: the user stares at nothing for seconds.

The answer is a **hold-back buffer**: send text immediately, *except* a tail that could still turn out to be a placeholder. Hold only that, only as long as needed.

And a second problem only real LLMs reveal: they don't always copy `[PHONE_1]` exactly. They write `[Phone_1]`, `[PHONE 1]`, `{PHONE_1}`, or drop the brackets. Phase 6 handles the safe variations and **counts** the rest, instead of guessing.

---

## 1. What counts as a placeholder — `placeholders.py`

| Question | Function | Used for |
|---|---|---|
| Which placeholders do we swap back? | `accepted_mentions()` | restoring |
| What merely *looks* like one of ours? | `loose_mentions()` | counting `unrestored`, and how each was mangled |
| Could this unfinished tail still become one? | `could_become_placeholder()` | the streaming hold-back |

**Accepted forms** (decided in the design):

| Form | Example | Accepted |
|---|---|---|
| exact | `[PHONE_1]` | yes |
| case | `[Phone_1]`, `[phone_1]` | yes |
| separator | `[PHONE 1]`, `[PHONE-1]`, `[BANK ACCOUNT_1]` | yes |
| other brackets | `{PHONE_1}`, `<PHONE_1>`, `(PHONE_1)` | yes (pairs must match) |
| bracketless | `PHONE_1` | yes, strictly: uppercase + underscore (decided on dev, see §7) |
| unclosed | `[PHONE_1` | the bare `PHONE_1` inside is restored, leaving a stray `[` |

Only types that appear in **this request's mapping** are matched. If nothing was masked as `NAME`, then `[NAME_1]` in a reply is just text: the guard never invents a value for it.

### Why not accept everything?

Every accepted form is a form a **user could plant**. Phase 2 already blocked the classic attack: a user types `[PHONE_1]` in their message hoping the guard will "restore" someone else's number into the reply. The guard reserves that number (`PHONE_1` is skipped; the real phone becomes `[PHONE_2]`). Phase 6 had to **extend that protection beyond the exact form**: `masking._reserved_numbers` now asks `loose_mentions()`, so typing `[phone 1]`, `{PHONE-2}`, a bare `phone 1` or an unclosed `[phone 1` reserves those numbers too. Loose, not just accepted, because the LLM may "tidy" a near-form the user typed (`phone 1`) into one the restorer accepts (`PHONE_1`); the final review caught that gap (§9). Harmless side effect: "phone 1 and phone 2" in ordinary text skips numbers 1 and 2. Tests: `test_reserved_numbers_cover_every_accepted_form`, `test_near_forms_a_user_types_are_reserved`.

Bare `PHONE_1` is the easiest to plant ("phone 1 and phone 2" in normal prose), which is why it is only accepted if real data says it's common, and then strictly (uppercase, underscore, word boundaries).

### `could_become_placeholder()` — written by hand

Python's `re` can't answer "is this a *prefix* of a match?". The third-party `regex` module can, but adds a dependency. The check is small: walk opener → type name (case-insensitive) → separator → up to 6 digits, fail as soon as a character can't continue any type. `"[PHO"` → yes. `"[x"` → no. `"[PHONE_1]"` → no (it's complete, not "still becoming"). `"[PHONE_1234567"` → no (7 digits can never be ours).

---

## 2. The restorer and the hold-back — `restore.py`

```python
restorer = Restorer(mapping)            # one per request
restorer.restore_text("call [Phone 1]") # -> "call 9876543210", counts restored=1
field = restorer.field()                # one per streamed text field
field.feed("I'll call [PH")             # -> "I'll call "        ("[PH" held)
field.feed("ONE_1] tomo")               # -> "9876543210 tomo"
field.flush()                           # -> ""                   (end of stream)
field.close()                           # count unrestored leftovers once
```

### How `feed()` decides what to hold

1. Join what was held with the new text.
2. Look only at the last *N* characters (N = the longest possible placeholder, e.g. 21 for `[BANK_ACCOUNT_123456]`): find the **earliest** position where an opener starts a tail that `could_become_placeholder()`.
3. Everything before that position is released (with complete placeholders inside it restored); the tail is held.

So with no `[ { < (` in sight, **nothing is held at all**: most chunks pass straight through. The hold is bounded: `test_hold_is_bounded` sends `[PHONE_123` + `4567` and checks the 7-digit text is released, not held forever.

### The property that makes it trustworthy

> Whatever way the text is cut into chunks, streaming restore gives **exactly** the same text (and the same counts) as restoring the whole text at once.

`test_streaming_equals_whole_text_for_every_cut` checks this for every 2-cut and 50 random chunkings of tricky strings (`[[NAME_1]]`, `[PHONE_[PHONE_1]`, `XNAME_1`, a bare `NAME_1` at the very end...), with bracketless on and off. Non-streaming restore is the reference; streaming must agree with it.

One subtle bug this caught by design: a bare `NAME_1` only counts at a **word boundary** (`XNAME_1` is not one). If the previous chunk ended with `X`, the restorer must remember that last released character, or it would wrongly restore `NAME_1` at the start of the next chunk. `FieldRestorer._before` keeps it.

### Tool-call arguments need escaping

Tool-call `arguments` are a JSON document *inside a string*: `{"address": "[ADDRESS_1]"}`. Restoring a multi-line address raw puts a real newline inside a JSON string, and the arguments stop being valid JSON. So values restored into arguments are JSON-escaped (`json.dumps(value)[1:-1]`). This also fixed a latent bug in Phase 2's non-streaming code, which restored arguments raw.

### What gets restored

`content`, `refusal`, `reasoning` / `reasoning_content` (the "thinking" text gpt-oss sends on Groq: it echoes placeholders too) and every tool call's `arguments`, in both modes.

### Counting: `restored` and `unrestored`

- `restored`: placeholders swapped for a value.
- `unrestored`: text `loose_mentions()` finds that points at one of **our** placeholders (its type and number are in the mapping, not reserved) but wasn't in an accepted form: e.g. `PHONE_1` while bracketless is off, or `[PHONE_1` with no closing bracket.

Placeholders the LLM *invents* (`[NAME_7]` when there is no NAME_7) aren't ours and aren't counted. Number-as-word ("Name One") can't be told apart from ordinary words: not counted, a stated limit.

For streams, `unrestored` is counted once at the end over the field's full text (a mangled placeholder can be split across chunks too). That means keeping the field's text in memory until the stream ends: request-scoped, like the mapping, and discarded afterwards.

---

## 3. Reading and rewriting the stream — `sse.py`

**`SSEParser`**: bytes in, complete events out. A network read can end in the middle of an event *or in the middle of a UTF-8 character* (Hindi characters are 3 bytes each). An incremental UTF-8 decoder handles the second; buffering until a blank line handles the first. `\n` and `\r\n` both work. It remembers how far it has already searched, so a huge event arriving one byte at a time stays linear.

**`StreamRestorer`**: per event:
- not JSON / not a chat chunk (`[DONE]`, an error event) → flush anything held first, then pass it on;
- a chat chunk → feed each choice's text fields and each tool call's arguments to their own `FieldRestorer` (keyed by choice index and tool-call index, so `n=2` and parallel tool calls don't mix);
- if nothing changed, send the event's **original bytes**, so streams with nothing to restore are byte-identical; if something changed, re-serialise only that event (keeping lines like `id: 7`).

**Nothing held is ever lost:**

| Situation | What happens to held text |
|---|---|
| a choice's `finish_reason` arrives | appended to that same final event |
| `[DONE]`, an error event, or anything non-chunk | one extra chunk event first, copying `id`/`model`/`created` from the last chunk |
| upstream ends without `[DONE]` | same extra event, from `finish()` |

---

## 4. Wiring — `app.py`, `logs.py`

```
stream request, masked something?  ──no──►  bytes pass through untouched (Phase 1 path)
        │yes
reply is text/event-stream? ──no──►  read fully, restore like non-streaming
        │yes                           (a provider that ignored "stream", or an error)
SSEParser → StreamRestorer → client
        │ exception inside the restorer?
        └──► log restore_failed, send the rest exactly as it came (still masked)
```

The fail-safe matters: a restorer bug must not kill a user's reply. Falling back to "send it masked" can't leak anything (the client only ever gets placeholders it would have got anyway). The one cost: text held at that instant (at most one placeholder's length) is lost. Test: `test_restore_failure_is_fail_safe`.

The log line gains two numbers:

```json
{"event": "request", ..., "findings": {"EMAIL:mask": 1, "PHONE:mask": 1},
 "restored": 8, "unrestored": 0, ...}
```

Still no text, no values. `test_logs_contain_no_content_or_auth` now also streams a restored reply and checks the value never reaches the logs.

---

## 5. Measuring on a real LLM — `evaluation/restore_run.py`

The mock LLM echoes placeholders perfectly, so it can't tell us how real models behave. The runner:

1. Builds conversations from the evaluation set: support messages with data the policy masks (no secrets: they're blocked; no `hard` samples: formats the detector misses, so nothing would be masked), plus an instruction that makes the model **repeat the details back** ("confirm my details back to me"). Without that, our first smoke test showed models often just say "Thanks!", and there'd be nothing to measure. Every 5th conversation forces a `create_ticket` tool call.
2. Runs the **real guard in-process** (`create_app` + `httpx.ASGITransport`), pointed at Groq through a **recording transport** that keeps the raw upstream bytes: what the model wrote *before* restoring.
3. Per reply, records: how the model wrote our placeholders (forms, from the raw reply), and the guard's own `restored`/`unrestored` (from its log line, matched by `x-request-id`). Counts only: no text, no values, no key in the results file.
4. Paces itself with the `x-ratelimit-*` headers the guard relays, and retries 429s after `retry-after`.

Dev 100 conversations, test 300, each × 2 models × 2 modes. Same rule as detection: **dev decides, test is reported once.**

A few dev streams were saved as test fixtures (`tests/fixtures/groq_streams/`), so CI replays *real* Groq chunking and fields through the restorer with no network and no key (`tests/test_recorded_streams.py`).

---

## 6. The report — `evaluation/restore_report.py`

- `restore-report --split dev` → [restore-tuning.md](../evaluation/restore-tuning.md): forms per model and the bracketless verdict.
- `restore-report --split test` → [restore-report.md](../evaluation/restore-report.md): the headline table, forms, and what restoring costs a stream.

Both are rendered from the committed results files, so anyone can regenerate them without a Groq key.

**What restoring costs:** measured on the relay itself (same ~700-character reply in 3-character events, relayed as-is vs restored): about 4 ms per whole reply (~17 µs per event) against about 0.25 ms as-is. It varies a little between runs. Network and the LLM are hundreds of milliseconds, so the hold-back is invisible in practice.

---

## 7. Dev decisions

Full page: [restore-tuning.md](../evaluation/restore-tuning.md). 100 dev conversations × 2 models × 2 modes = 400 replies.

**How the models wrote our placeholders** (745 mentions, raw replies):

| Form | gpt-oss-120b | qwen3.8-27b |
|---|---:|---:|
| exact `[PHONE_1]` | 93.5% | 98.5% |
| bracketless `PHONE_1` | 6.2% | 0.9% |
| other brackets, unclosed | 0.2% | 0.6% |
| case, separator | 0% | 0% |

Two surprises: the variations I expected most (`[Phone_1]`, `[PHONE 1]`) never happened, and the one I was most cautious about, dropping the brackets, was the common one, mostly from gpt-oss, in both plain text and tool-call arguments.

**The bracketless decision.** The rule was written down *before* the run: accept bare `NAME_1` only if it is at least 2% of mentions on dev. It was 3.8% (28 of 745), so `ACCEPT_BRACKETLESS = True`, in its strict form (uppercase, underscore, word boundaries), with reservation extended to it. 28 of the 29 unrestored placeholders on dev were bracketless, so this targets almost the whole gap. Because the decision changes the code, the dev numbers are *before* it; the test run measures the result.

**A side effect, accepted on purpose:** `[PHONE_1` (unclosed) contains a valid bare `PHONE_1`, so it's now restored as `[9876543210`. The user sees the value; the stray bracket is cosmetic. Test: `test_unclosed_placeholder_gets_its_bare_part_restored`.

**The 3 errors** were all gpt-oss, non-streaming, forced tool call: Groq answered HTTP 400 `tool_use_failed` (the provider validates tool calls and rejected the model's malformed one). I re-sent one directly to Groq to check: same error, and the guard had blocked nothing.

---

## 8. Results on the test split

Full report: [restore-report.md](../evaluation/restore-report.md). 300 test conversations × 2 models × 2 modes = 1,200 replies, run once, after the dev decision, with nothing changed afterwards.

| Model | Mode | Mention a placeholder | Fully restored | Mentions rewritten | Unrestored |
|---|---|---:|---:|---:|---:|
| gpt-oss-120b | non-streaming | 280 | 100.0% | 12.9% | 0 |
| gpt-oss-120b | streaming | 273 | 100.0% | 11.8% | 0 |
| qwen3.8-27b | non-streaming | 258 | 99.6% | 6.8% | 1 |
| qwen3.8-27b | streaming | 259 | 99.6% | 3.5% | 1 |

**How to read it.** gpt-oss rewrote about 1 in 8 placeholder mentions (mostly dropping brackets: 10%, plus `{PHONE_1}` 2.3%), yet every one was restored, because those exact forms were accepted. On dev, *before* the bracketless decision, gpt-oss was at 89.5–92.4% fully restored: that jump is what the dev decision bought. Streaming and non-streaming score the same, which is the point: the streaming restorer is held to the non-streaming one by the property test.

**What's left:** two qwen replies each kept one unrestored placeholder: a bare form the strict rule deliberately doesn't accept (lowercase, or with a space instead of the underscore). The results file stores counts, not text, so it doesn't say which of those it was.

**What isn't in the rate, and why:**
- Replies that mention no placeholder at all: 16–39 per model and mode (5–13%). These are models paraphrasing ("your number ending in…"), refusing, or answering generically. Not a restore failure, so not in the rate, but reported.
- 9 requests that failed at Groq: 4 × HTTP 400 (gpt-oss, forced tool call, the same `tool_use_failed` pattern as dev) and 5 × HTTP 503 (qwen, provider temporarily unavailable). The guard never returns 503, and nothing was blocked.

**Honest caveats** (also in the report's notes): real models drift and sampling varies, so a re-run gives similar, not identical, numbers; number-as-word placeholders aren't counted; the conversations were written by the same author as the guard, and they *ask* the model to repeat details, which is the hardest case for restoring but not what every app does; `hard` samples and secrets are excluded.

**Cost of restoring:** about 4 ms per ~700-character streamed reply, against about 0.25 ms for plain pass-through.

---

## 9. The final review: a one-character crash and three gaps

A fresh reviewer read the whole branch and attacked it with its own inputs (80,000 fuzzed chunkings among them). It found:

| Finding | What it was | Fix (each with a test that failed first) |
|---|---|---|
| **Crash from one Unicode character** (critical) | Python's `re.IGNORECASE` is Unicode-aware: the Turkish `İ` matches `i`, the Kelvin sign `K` matches `k`. So `[EMAİL 1]` matched the pattern, then `EntityType("EMAİL")` raised. In a request: detection failed (a 500, or in fail-open mode the whole request sent **unmasked**). In a reply: a bare 500 and a lost reply. | `re.ASCII` on every pattern (placeholders are ASCII), and unknown names are skipped instead of raising |
| **Streaming ≠ whole text next to non-ASCII letters** | The hold-back's "starts a word" check used Unicode `isalnum()`, the bare pattern ASCII `[A-Za-z0-9_]`: `éPHONE_1` restored or not depending on where the chunk was cut | One rule, ASCII, in both places; non-ASCII strings added to the property test |
| **Quadratic overlap checks** | "Does this mention overlap an earlier one?" was checked against *all* earlier ones: 10,000 placeholders took seconds, blocking the event loop | One forward sweep over two sorted lists (`drop_overlapping`): linear |
| **Reservation too narrow** | Only accepted forms were reserved, so a user's `phone 1` could be echoed by the LLM as `PHONE_1` and filled with the real number | Reserve every placeholder-*looking* text (§1) |
| **Non-streaming restore not fail-safe** | A restorer bug turned a good reply into a bare 500 with no log line | Same rule as streaming: log `restore_failed`, relay the masked reply |

Did the fixes change the measured numbers? Re-masking all 400 dev and test conversations with the fixed code gives exactly the same requests, so no; the report's notes say so. Six minor findings (rare Unicode line separators in SSE, bare placeholders right after a JSON `\n` escape in tool arguments, parser copying on multi-megabyte single reads, and others) are listed in the PR for later.

**Lesson worth telling in an interview:** case-insensitive regexes in Python are Unicode-aware by default. For anything that matches a fixed ASCII vocabulary, especially in security code, say `re.ASCII`.

---

## 10. Interview Q&A

**Q: How do you restore placeholders in a streamed response when a placeholder can be split across chunks?**
A: A per-field hold-back buffer. Text is released immediately except the shortest tail that could still become a placeholder: a `[` followed by a prefix of a known type name, a separator and up to 6 digits. That tail is held until it completes (restored) or can't (released as-is). The hold is bounded by the longest possible placeholder, ~20 characters, so the user never notices. And I prove it with a property test: for every way of cutting a text into chunks, the streamed result equals restoring the whole text at once.

**Q: Why not just buffer the whole response and restore at the end?**
A: Then it isn't streaming: the user waits for the full reply. Chat UIs stream precisely so people see text in a second. I kept that as a fallback only, for providers that return JSON to a stream request.

**Q: Why not find-and-replace on the raw bytes?**
A: A split placeholder has JSON between its halves: `[PH"}}]}\n\ndata: {"choices":...,"content":"ONE_1]`. You have to parse events, track each field separately, and rewrite.

**Q: LLMs mangle placeholders. Why not restore anything that looks close?**
A: Two reasons. A wrong guess puts the wrong value in front of the user, worse than leaving a placeholder. And every accepted form is a form a user could type to try to make the guard "restore" a value: so I accept only forms I also reserve, and I added bracketless only if data justified it. Everything else is counted as `unrestored`, so I can see the gap instead of hiding it.

**Q: What happens if your restore code crashes mid-stream?**
A: The relay catches it, logs `restore_failed` (no content), and sends the rest of the stream exactly as received: still masked. The reply never dies, and nothing can leak because the client only gets placeholders. The cost is at most one placeholder's worth of held text.

**Q: How did you measure restore quality, and why can't it run in CI?**
A: 300 test conversations through the real guard to two Groq models, streaming and non-streaming, with prompts that ask the model to repeat details back. A recording transport captures what the model actually wrote, the guard's log gives restored/unrestored. It needs a provider key and real models drift, so CI runs the deterministic part instead: unit and property tests plus recorded real Groq streams replayed offline.

**Q: Why did you need JSON escaping for tool calls?**
A: Tool arguments are JSON inside a string. A multi-line address restored raw puts a literal newline inside a JSON string and breaks `json.loads` in the client. Escaping the value fixes it. It was a latent bug in the non-streaming path too.

**Q: How do you keep the log from leaking data now that it records restore results?**
A: Same design as Phase 1: `log_request()` only takes named numeric fields. There's no parameter for text, so a value can't end up there by accident. A test streams a restored reply and asserts the value isn't in the logs.
