# Phase 2 Walkthrough — Detection, Policy, Masking

> **What Phase 2 builds:** the guard starts protecting data. Rule-based detectors find Aadhaar, PAN, phone, UPI, email, card, IFSC, bank account and secrets; a policy decides mask / block / log; masking swaps values for placeholders before the request leaves, and restores them in the reply.

---

## 1. The building blocks: entity types, spans, merging, checksums

### Files
| File | Purpose |
|---|---|
| `src/leakguard/entities.py` | `EntityType` (the 11 kinds of sensitive data) and `PRIORITY` |
| `src/leakguard/detect/spans.py` | `Span` and `merge_spans()` |
| `src/leakguard/detect/checksums.py` | `verhoeff_valid()` (Aadhaar) and `luhn_valid()` (cards) |

### What a span is
Every detector answers the same question: *where in this text is sensitive data, and what kind?* The answer is a list of **spans**:
```
text:  "call 9876543210 now"
        0123456789...
Span(start=5, end=15, type=PHONE)   →  text[5:15] == "9876543210"
```
`end` is exclusive, exactly like Python slicing, so `text[span.start:span.end]` is the value. Spans are `frozen` dataclasses: once created they can't be changed, which makes them safe to pass around and compare in tests.

### Why merging is needed
Different detectors can claim the same characters:
- `9876543210@ybl` → the **phone** detector sees `9876543210`, the **UPI** detector sees the whole thing.
- `a/c 9876543210` → **phone** *and* **bank account**.

`merge_spans()` keeps one winner per region:
1. **Higher priority wins** (`PRIORITY` in `entities.py`): SECRET > AADHAAR > BANK_ACCOUNT > CARD > PAN = IFSC > UPI > PHONE > EMAIL > ADDRESS > NAME.
2. On a tie, the **longer** span wins (it captured more of the value).
3. Then the **earlier** one.

It's a greedy algorithm: sort the candidates best-first, then accept each one only if it doesn't overlap anything already accepted. Spans that merely *touch* (one ends where the next starts) don't overlap.

**A performance bug found in review.** The first version checked each candidate against *every* span already kept. That's fine for a normal message, but it's **quadratic**: 20,000 phone numbers took 16 s, and a request at the 1 MB size limit (~65,000 numbers) would have taken about 2–3 minutes. That's an easy denial-of-service. The fix uses a property we already have: kept spans never overlap, so they can be kept **sorted by position**, and a new candidate can only collide with its two neighbours. `bisect_right()` finds those neighbours by binary search, giving O(n log n) overall. Now the 1 MB worst case takes about 1.7 s. A test (`test_detect_all_is_fast_with_many_findings`) pins it.

Why is BANK_ACCOUNT above PHONE? Some Indian banks use the customer's mobile number as the account number, so "a/c 9876543210" really is an account. And SECRET is on top because secrets are *blocked*: if any detector thinks something is a key, that judgement must not be overridden.

### Checksums — the cheapest false-positive filter
A **check digit** is computed from the other digits. If someone mistypes a digit, the check fails.
- **Aadhaar → Verhoeff.** Walk the digits from the right, combining each with a running value through two lookup tables (`_D`, a multiplication table of a mathematical group, and `_P`, a permutation table). A valid number ends with running value **0**. Verhoeff catches *every* single-digit error and *every* swap of two neighbouring digits — that's why UIDAI chose it.
- **Cards → Luhn.** From the right, double every second digit (subtract 9 if it goes above 9), add everything up; valid if the total ends in 0.

**Why this matters for detection:** a regex for "12 digits" matches every order ID, invoice number and timestamp. With one check digit, a random 12-digit number passes only about **1 time in 10** — so the checksum throws away ~90% of look-alikes for free, without any ML.

`isascii()` is checked too: Python's `isdigit()` also accepts digits from other scripts (like Devanagari `४`), which `int()` would convert — we only want ASCII 0–9 here.

### Q&A
**Q: Why does a checksum help detection?**
A: Detection has to balance catching real values against false alarms. Many numbers in support chats have the same *shape* as an Aadhaar number — order IDs, tracking numbers. Aadhaar's last digit is a Verhoeff checksum of the other 11, so only ~10% of random 12-digit numbers pass. Checking it removes most look-alikes at near-zero cost, before any expensive model runs.

---

## 2. ID detectors — Aadhaar, PAN, card, IFSC

### File
`src/leakguard/detect/ids.py` — four detector functions, each `text -> list[Span]`.

### Reading the patterns
Regular expressions look scary; here's the Aadhaar one piece by piece:
```
(?<!\d)   [2-9]\d{3}   ([ -]?)   \d{4}   \1   \d{4}   (?!\d)
   │          │           │        │      │      │        │
   │          │           │        │      │      │        └ not followed by a digit
   │          │           │        │      │      └ last 4 digits
   │          │           │        │      └ the SAME separator again (backreference)
   │          │           │        └ middle 4 digits
   │          │           └ optional space or hyphen, remembered as group 1
   │          └ first 4 digits; first digit 2-9 (Aadhaar never starts with 0 or 1)
   └ not preceded by a digit
```
- `(?<!\d)` and `(?!\d)` are **lookarounds**: they check the neighbouring character without consuming it. Together they mean "this number isn't part of a longer number", so `id 12345678901234` (14 digits) is never sliced into a fake Aadhaar.
- `\1` forces a **consistent separator**: `2345 6789 0124` and `2345-6789-0124` match, a messy `2345 6789-0124` doesn't. Real people write it consistently; random digit soup usually isn't.
- After the regex finds a candidate, `verhoeff_valid()` checks the check digit. Both must agree.

| Detector | Shape | Extra check |
|---|---|---|
| **Aadhaar** | 12 digits, first 2-9, groups of 4 | Verhoeff |
| **PAN** | `ABCPS1234K`: 5 letters, 4 digits, 1 letter, case-insensitive | 4th letter must be a real holder type (P, C, H, F, A, T, B, L, J, G) |
| **Card** | 13–19 digits, optional spaces/hyphens | Luhn |
| **IFSC** | `HDFC0001234`: 4 letters, a literal `0`, 6 letters/digits | the `0` |

**Why PAN's 4th letter?** It encodes the holder type (P = person, C = company…). Only 10 letters are valid there, so a random 10-character code like `ABCXS1234K` is rejected — a cheap structural "checksum".

**Why IFSC's `0`?** The 5th character is reserved by RBI and is always zero. It's what separates an IFSC from any other 11-character code.

**The `_find()` helper** removes repetition: run a pattern, optionally validate each match, wrap matches in `Span`s. Each detector is then 1–2 lines, and adding a new one is easy.

### Q&A
**Q: How do you avoid flagging order numbers as Aadhaar numbers?**
A: Three filters in a row. Shape: exactly 12 digits, first digit 2–9, not part of a longer number (lookarounds), consistent grouping. Then the Verhoeff checksum, which ~90% of random numbers fail. And if a number still slips through, the evaluation in Phase 3 measures how often, so the false-positive rate is a known number, not a guess.

---

## 3. Contact and payment detectors — phone, UPI, email, bank account

### Files
| File | Purpose |
|---|---|
| `src/leakguard/detect/contact.py` | `detect_phone`, `detect_upi`, `detect_email`, `detect_bank_account` |
| `src/leakguard/data/upi_handles.txt` | the list of known UPI handles (`okicici`, `ybl`, `paytm`…) |
| `pyproject.toml` | `package-data` so the `.txt` file is installed with the package |

### Phone numbers
```
(?<![\d+])  (?:(?:\+91|91)[ -]?|0)?   [6-9]   (?:[ -]?\d){9}   (?!\d)
            optional +91 / 91 / 0      first   9 more digits,
            prefix                     digit   each optionally after one space/hyphen
```
Indian mobile numbers are 10 digits starting with 6, 7, 8 or 9. Allowing one optional separator between digits covers `98765 43210`, `98765-43210` and `+91 98765 43210` with one pattern. The prefix is part of the span, so the *whole* `+91 98765 43210` gets masked — nothing like `+91 [PHONE_1]` leaks the country code.

**Look-alikes rejected:** a 6-digit **OTP** (`482913`) is too short; `5876543210` starts with 5; `98765432101234` is part of a longer number.

### UPI vs email — why a handle list
`rahul.s@okicici` and `rahul@gmail.com` look alike: `something@something`. The difference is the part after `@`:
- a UPI **handle** is a short bank/app code with no dot (`okicici`, `ybl`, `paytm`);
- an email has a **domain** with a dot (`gmail.com`).

So UPI requires the handle to be in `upi_handles.txt` **and** not followed by a dot. Without the list, every `user@company` mention would be flagged. The list lives in a plain text file so it can be updated without touching code. `importlib.resources` reads it from inside the installed package, so it works the same in Docker, in tests and from a wheel.

### Bank account numbers — context decides
Indian bank account numbers are 9–18 digits with **no checksum**, so on its own a long number could be anything: an order ID, a ticket, a tracking number. The detector only reports a digit run when:
- within **40 characters before** it there's a context word: `a/c`, `ac no`, `acc`, `account`, `acct`, `khata` (Hindi for account) or `bank`; **or**
- an **IFSC code** appears within 50 characters either side (an IFSC next to a number almost always means bank details).

`"Ticket 50100123456789 is open"` → no context → not masked. `"My a/c no 50100123456789"` → masked.
This is a deliberate trade-off: some real account numbers without context will be missed. Phase 3's evaluation measures how many.

### ReDoS — why every pattern starts with a lookbehind
**ReDoS** (Regular-expression Denial of Service) is when a crafted input makes a regex take a very long time. Example: `"a" * 100_000 + "@"`. A naive UPI pattern would try to start a match at *every one* of the 100,000 positions, and each attempt scans forward to the `@`: about 100,000 × 100,000 steps. The lookbehind `(?<![A-Za-z0-9._@-])` allows a match to start only at the *beginning* of a token, so there's one attempt instead of 100,000. Upper bounds like `{2,256}` cap how far one attempt can scan. A test in the next task feeds such inputs and requires each to finish within 2 seconds.

### Q&A
**Q: How do you tell a bank account number from any other long number?**
A: You can't from the digits alone: account numbers have no checksum. So I use context: an account keyword just before it (a/c, account, khata, bank) or an IFSC code nearby. It's a precision/recall trade-off. Without context I'd mask every order number and confuse the chatbot; with context I miss some bare account numbers. The evaluation quantifies that miss rate.

---

## 4. Secret detection and the combined engine

### Files
| File | Purpose |
|---|---|
| `src/leakguard/detect/secrets.py` | `detect_secrets()`, `shannon_entropy()`, `word_fraction()` |
| `src/leakguard/detect/engine.py` | `DETECTORS` (every detector) and `detect_all()` |
| `src/leakguard/detect/__init__.py` | re-exports `detect_all` so callers write `from leakguard.detect import detect_all` |

### Three ways to find a secret
1. **Known prefixes.** Many providers stamp their keys: `sk-` (OpenAI, Anthropic), `AKIA` (AWS), `ghp_` (GitHub), `xoxb-` (Slack), `AIza` (Google), `sk_live_` (Stripe), `eyJ…` (JWT), `-----BEGIN PRIVATE KEY-----`. Precise and cheap.
2. **Key/value context.** `password=hunter22`, `api_key: "zx81"`, `Authorization: Bearer …`. The value after the `:`/`=` is the secret.
3. **Entropy**, for formats nobody has seen before. See below.

All hits are merged inside `detect_secrets()`, because one key can match several rules (a prefix *and* entropy) and should be reported once.

### Shannon entropy, worked example
Entropy measures how unpredictable each character is, in bits:  `H = −Σ p(c)·log₂ p(c)`.
- `"aaaa"`: one symbol, p = 1, so H = 0 bits. Completely predictable.
- `"abcd"`: four symbols, each p = ¼, so H = 4 × (¼ × 2) = **2 bits**.
- A random 28-character key using upper, lower and digits: almost every character is different, so H ≈ 4.8 bits.

A token is flagged when it is **≥ 24 characters**, **mixes lowercase, uppercase and digits**, has **entropy > 4.0**, and is **not mostly words** (next section).

**Why > 4.0 excludes hex:** hex uses only 16 symbols, and the maximum entropy of 16 equally likely symbols is log₂16 = **4.0**. So git commit hashes and UUIDs (very common in developer prompts, and not secret) can never pass.

### The bug the plan didn't foresee: code names
While building this I probed extra strings and found `getUserAccountBalance2024Handler` was flagged as a secret: long, mixed-case, has digits, entropy 4.05. Since secrets are **blocked**, every developer pasting code would be rejected. The fix, which I measured before choosing:
- `word_fraction()` = the share of the token made of word-like chunks (`[A-Z]?[a-z]{3,}` like `Invoice`, or `[A-Z]{4,}` like `HTTP`).
- Code names: **0.83–0.88**. Random keys: median **0.25**, and 99.9% of 9,000 random samples score below **0.79**.
- Threshold **0.75**: tokens above it read like code and are not flagged.
- Result: **1,993 / 2,000** random 32-character keys still caught (99.65%). The few misses are mostly keys that happen to contain no digit.

The first idea (counting how often the character type switches) was tried and **rejected**: `parseInvoiceLineItemsFromCsvV3` switches case 41% of the time, too close to random keys. The lesson: *measure before you pick a threshold.*

### Why "my password is wrong" is not a secret
Because secrets are blocked, false positives are expensive: the user's whole message is rejected. So the key/value rule:
- requires `:` or `=` after the keyword. "password is wrong" has neither.
- strips trailing punctuation and requires the value to be ≥ 4 characters **and contain a digit or symbol**. "reset password: how?" gives `how`, and "password: please help" gives letters only, so neither is flagged.
- The trade-off: a password made only of letters (`password: sunshine`) is missed. That's documented, and the evaluation will measure it.

### The engine
`DETECTORS` is a tuple of functions with the same shape, `str -> list[Span]`. `detect_all()` runs them all and merges the results with `merge_spans()`. Adding the name model (Phase 4) or the address detector (Phase 5) means adding one entry to the tuple and nothing else.

### ReDoS test
`test_detect_all_is_fast_on_adversarial_input` feeds 100,000-character hostile inputs (`"9 " * 50_000`, `"a" * 100_000 + "@"`, …) and requires each to finish in under 2 seconds. The token-start lookbehinds (section 3) are what make this pass.

### Q&A
**Q: How do you detect an API key format you've never seen?**
A: Entropy. Random keys pack about 5 bits per character; normal text and identifiers much less, and hex IDs can't exceed 4. I flag long tokens that mix character classes, exceed 4 bits per character, and aren't mostly English-like word chunks. The last rule came from a real false positive I found: code identifiers. I measured it on 9,000 random samples, so I can state its miss rate (about 0.35% on 32-character keys) instead of guessing.

---

## 5. The policy — what to do with each kind of data

### Files
| File | Purpose |
|---|---|
| `src/leakguard/policy.py` | `Action`, `FailureMode`, `Policy`, `DEFAULT_POLICY`, `load_policy()` |
| `config/policy.yaml` | the documented default policy, ready to copy and edit |
| `src/leakguard/config.py` | `policy_path` setting (`LEAKGUARD_POLICY_PATH`) |

### Three actions
| Action | What happens | Typical use |
|---|---|---|
| `mask` | value replaced by `[TYPE_N]` before sending, restored in the reply | personal data (default) |
| `block` | whole request rejected with HTTP 400, nothing sent | secrets (default): no LLM needs your API key |
| `log_only` | sent unchanged, counted in logs | trying the guard out on live traffic first ("monitor mode") |

A company that's fine sending names writes a two-line file:
```yaml
actions:
  NAME: log_only
```
Every type it doesn't mention keeps its default. That's why `load_policy()` starts from a copy of `DEFAULT_POLICY` and only overrides what the file lists.

### Fail-closed vs fail-open
What if the detector itself crashes (a bug, a bad input)?
- **closed** (default): reject the request. Nothing can leak, but the app gets an error.
- **open**: forward the request **unmasked** and log a warning. The app keeps working, but data may leak.

Security tools default to fail-closed because a silent leak is worse than a visible error. Some companies prefer availability, so it's configurable.

### Fail at startup, not mid-request
A typo like `PHONE: hide` raises `PolicyError` **when the guard starts**, naming the bad value and listing the allowed ones. The alternative, discovering it on the first real request, would mean a broken guard in production. The tests check four kinds of mistakes: an unknown type, an unknown action, an unknown failure mode and an unknown top-level key.

### Why `yaml.safe_load`
Plain `yaml.load` can construct arbitrary Python objects from tags in the file, which is a known code-execution risk if someone can edit the config. `safe_load` only builds plain data (dicts, lists, strings, numbers). For a security tool it's the only acceptable choice.

### Enums instead of strings
`Action` and `EntityType` are `StrEnum`s. Code compares `action is Action.BLOCK` instead of `action == "blcok"`, so a typo in code is an immediate `AttributeError` rather than a silent bug. Because they're *string* enums, they still read and print as plain text in YAML and in logs.

### Q&A
**Q: What does fail-closed mean, and why is it the default?**
A: If the protection mechanism fails, the system denies rather than allows. For the guard, a crash in detection rejects the request instead of forwarding unchecked data. Leaking an Aadhaar number can't be undone, while a rejected request can be retried. So for a data-protection tool the safe failure is denial. It's configurable because some teams value availability more, and then they make that choice explicitly.

---

## 6. Masking and restoring

### File
`src/leakguard/masking.py`: `PlaceholderMap`, `mask_messages()`, `restore_response()`.

### The round trip
```
request:  "I'm Rahul, phone 9876543210, pan ABCPS1234K"
masked:   "I'm Rahul, phone [PHONE_1], pan [PAN_1]"        ← this is all the LLM sees
LLM:      "Updated [PAN_1]; we'll SMS [PHONE_1]."
restored: "Updated ABCPS1234K; we'll SMS 9876543210."      ← what the app receives
```
(Names arrive in Phase 4, so "Rahul" isn't masked yet.)

### Why typed, numbered placeholders, not `****`
- **Typed** (`PHONE`, `PAN`): the LLM still knows *what kind* of thing is there and can answer sensibly ("we'll SMS [PHONE_1]").
- **Numbered**: two different phone numbers become `[PHONE_1]` and `[PHONE_2]`, and the *same* number is always `[PHONE_1]`. The LLM can tell people apart and refer back.
- **Reversible**: the guard can put real values back. With `****` the reply would be useless.

### Deterministic numbering = no database
Numbers are given in **order of first appearance** across all messages. Chat APIs resend the whole conversation on every request, so the same history always produces the same placeholders. That's why the mapping can live in memory for one request and then be thrown away. There's no Redis and nothing persisted, so there's nothing to steal.

### Normalisation: same person, different formatting
`98765 43210` and `+91 9876543210` are the same number. `_normalise()` reduces digit types to digits only (and phones to the last 10 digits, dropping `+91`/`0`), and other types to case-folded text. Both forms map to `[PHONE_1]`. Without this, the LLM would think the customer gave two different numbers.

### Placeholder injection, and the defence
**The attack:** a user types `what is [PHONE_1]?`. If the guard then numbered the real phone `[PHONE_1]` too, its reply would turn the user's literal text into someone's real number. Mixing up "data" and "control" text like this is the same class of bug as SQL injection.

**The defence, in two parts:**
1. Before numbering, `_reserved_numbers()` scans the request for placeholder-looking text and **reserves** those numbers. The real phone becomes `[PHONE_2]`.
2. `restore()` only replaces placeholders **this map created**. `[PHONE_1]` (user-typed) and `[PHONE_9]` (made up by the LLM) are left untouched.

The test `test_user_typed_placeholder_is_never_restored` pins both parts.

### Where text hides in a request
`_map_message()` applies a function to every text field:
- `content` as a plain string;
- `content` as a list of parts. Only `{"type": "text"}` parts are changed; images pass through;
- **tool-call arguments** (`tool_calls[].function.arguments`). When an AI agent calls a function, the arguments can contain personal data too.

The same helper is used three ways: to *collect* placeholders (reservation), to *mask*, and to *restore*. One traversal, three behaviours.

### Inputs are never modified
`mask_messages()` returns **new** dicts and leaves the caller's data untouched. If masking mutated the request in place, a later bug (or the fail-open path) could forward half-masked data, and tests would be harder to reason about.

### Counting for the logs
Every detection is counted as `(type, action)`, e.g. `(PHONE, mask): 3`. Only these counts are logged, never the values.

### Q&A
**Q: What if a user types `[PHONE_1]` themselves?**
A: Before numbering, the guard scans the request for placeholder-shaped text and reserves those numbers, so the real phone becomes `[PHONE_2]`. And restore only replaces placeholders the guard itself created in this request. So user-typed or LLM-invented placeholders are never turned into real values. It's the same principle as SQL injection defences: keep data and control separate.

---

## 7. Wiring it into the guard

### Files
| File | Change |
|---|---|
| `src/leakguard/app.py` | `_apply_policy()`, restore in `_relay()`, policy loaded in `create_app()` |
| `src/leakguard/logs.py` | `findings`, `blocked`, `detection_ms`; `log_detection_failed()` |
| `tests/conftest.py` | `make_guard(policy=...)` |
| `tests/test_guard_masking.py` | end-to-end behaviour |
| `README.md` | what's detected, the policy, the streaming limitation |

### The full request journey now
```
POST /v1/chat/completions
 1. _read_chat_request()   size limit → valid JSON → 'messages' is a list of objects
 2. _apply_policy()        mask_messages(): detect every text field, apply policy
      ├─ detector crashed? → fail-closed: 500 detection_failed
      │                      (fail-open: forward original, log a warning)
      ├─ anything to BLOCK? → 400 sensitive_data_blocked (names the TYPES, never values)
      └─ otherwise         → masked body + PlaceholderMap
 3. _send_upstream()       the LLM only ever sees the masked body
 4a. normal reply   → _restored_content(): parse JSON, restore placeholders, re-serialise
 4b. streamed reply → passed through (masked; restored in Phase 6)
 5. ctx.finish()           log: status, findings counts, timings, never values
```

### Verified in Docker
```
REPLY:  You said: My Aadhaar is 2345 6789 0124, a/c 50100123456789 IFSC HDFC0001234, upi rahul.s@okicici
STREAM: You said: pan [PAN_1]
BLOCKED: 400 sensitive_data_blocked
log: {"findings": {"AADHAAR:mask": 1, "BANK_ACCOUNT:mask": 1, "IFSC:mask": 1, "UPI:mask": 1},
      "detection_ms": 0.13, ...}
```
The fake AI only saw `[AADHAAR_1]`, `[BANK_ACCOUNT_1]`, `[IFSC_1]` and `[UPI_1]`; the app got the real values back. Detection costs about **0.1 ms** per request, against ~20 ms for even the local mock LLM.

### Key design decisions
**1. Block before sending.** When a secret is found, the guard returns 400 **without contacting the LLM at all** (`test_secret_blocked...` asserts the upstream was never called). The error message names the kind (`SECRET`) so the developer knows what to fix, but never echoes the value: error messages end up in logs and screenshots.

**2. Fail-closed in code.** `_apply_policy()` wraps detection in a deliberately broad `try/except Exception`. Normally catching every exception is bad style (the linter flags it), but here it's the whole point: *any* unexpected failure must turn into a refusal, not a silent pass-through. It's marked `# noqa: BLE001` with a comment explaining why.

**3. Restore only what's safe to touch.** `_restored_content()` restores only successful JSON replies. Error bodies and non-JSON content are relayed byte-for-byte. If the reply can't be parsed, the original bytes go back unchanged rather than crashing.

**4. Policy loaded once, at startup.** `create_app()` calls `load_policy()`, so a broken policy file stops the guard from starting instead of failing on the first customer request.

**5. Streaming is safe, just not pretty yet.** Streamed requests *are* masked, so nothing leaks. The reply shows `[PAN_1]` until Phase 6 adds the restorer that handles placeholders split across chunks (`[PA` + `N_1]`). Shipping the safe half first is a deliberate order.

### Q&A
**Q: Walk me through what happens to a request containing an Aadhaar number.**
A: The guard parses and validates the body, then runs every detector over every text field of every message. The Aadhaar regex finds 12 digits in the right shape and the Verhoeff checksum confirms it. The policy says `mask`, so it becomes `[AADHAAR_1]` in a fresh request-scoped mapping. Only the masked request goes to the LLM. When the reply comes back, `[AADHAAR_1]` is swapped back for the real number and the mapping is discarded. The log line records `"AADHAAR:mask": 1` and timings, never the number. Detection takes about a tenth of a millisecond.
