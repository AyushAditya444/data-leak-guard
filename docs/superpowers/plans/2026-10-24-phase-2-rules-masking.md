# Phase 2 — Rule Detectors, Policy, Masking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The guard actually protects data: rule-based detection of Aadhaar, PAN, Indian phone, UPI, email, card, IFSC, bank account and secrets; a YAML policy (mask / block / log_only); reversible masking with typed placeholders; blocking with an OpenAI-shaped 400; restoration in non-streaming replies.

**Architecture:** A `detect` package of small pure functions, each `str -> list[Span]`, combined by `detect_all()`, which merges overlapping spans by a fixed type priority. `masking.mask_messages()` walks every text field in the request's messages, asks the policy what to do per span, and replaces masked spans with placeholders from a request-scoped `PlaceholderMap` (in memory only, never persisted). The app calls it between parsing and forwarding, and calls `PlaceholderMap.restore()` on the reply.

**Tech Stack:** Python 3.11 stdlib `re` (no new detection dependencies), PyYAML for policy, existing FastAPI/httpx stack.

**Spec:** [docs/superpowers/specs/2026-10-24-data-leak-guard-design.md](../specs/2026-10-24-data-leak-guard-design.md) — implements §9 phase 2: §6.2, §6.3 (rules + secrets), §6.4, §6.6 (non-streaming, placeholder injection), §6.7 counts, §6.8 fail-closed.

**Who writes it:** Claude implements; each task appends a section to `docs/walkthrough/phase-2-detection.md`.

## Global Constraints

- Everything from the Phase 1 plan's Global Constraints still holds (Python 3.11, zero cost, never log content/values/auth, OpenAI-shaped errors).
- Entity type names (exact): `AADHAAR, PAN, PHONE, UPI, EMAIL, CARD, IFSC, BANK_ACCOUNT, SECRET, NAME, ADDRESS` (`NAME`/`ADDRESS` exist for the policy now; detectors arrive in Phases 4–5).
- Placeholder format: `[<TYPE>_<n>]`, `n` from 1 per type, in order of first appearance across all messages.
- Default actions: every personal type `mask`; `SECRET` `block`. Default `failure_mode: closed`.
- Merge priority (higher wins on overlap): `SECRET 100 > AADHAAR 90 > BANK_ACCOUNT 80 > CARD 70 > PAN 65 = IFSC 65 > UPI 60 > PHONE 50 > EMAIL 40 > ADDRESS 20 > NAME 10`; ties → longer span, then earlier start.
- Blocked request → HTTP 400, `type: "invalid_request_error"`, `code: "sensitive_data_blocked"`, message names the types, never values.
- Detection failure → fail-closed: HTTP 500, `type: "guard_error"`, `code: "detection_failed"`; `failure_mode: open` forwards unmasked and logs a warning (metadata only).
- Mapping lives only in memory for one request.
- No new runtime dependencies.

## Review Focus

1. **User-typed placeholder text** (`"what is [PHONE_1]?"`) must never be replaced with a real value. Test: `test_user_typed_placeholder_is_never_restored` (Task 6).
2. **The same value in different formats** (`98765 43210` vs `+91 9876543210`) must get the same placeholder, or the LLM thinks they're two people. Test: `test_same_value_different_format_same_placeholder` (Task 6).
3. **Look-alikes must not be masked:** a 12-digit order number failing Verhoeff, a 6-digit OTP, a 10-digit number starting 1–5, a long ticket number without account context. Tests: `test_aadhaar_rejects_bad_checksum_and_lookalikes`, `test_phone_rejects_lookalikes`, `test_bank_account_requires_context` (Tasks 2–3).
4. **Ordinary sentences about passwords** ("my password is wrong", "reset password: how?") must **not** block the request — a false block breaks the chatbot. Test: `test_natural_language_password_mentions_not_flagged` (Task 4).
5. **Adversarial huge inputs** (100k digits, 100k letters with `@`) must not hang the regexes (ReDoS). Test: `test_detect_all_is_fast_on_adversarial_input` (Task 4).

---

## File Structure

```
src/leakguard/
├── entities.py              # EntityType enum + PRIORITY (Task 1)
├── detect/
│   ├── __init__.py          # re-exports detect_all
│   ├── spans.py             # Span, merge_spans (Task 1)
│   ├── checksums.py         # verhoeff_valid, luhn_valid (Task 1)
│   ├── ids.py               # aadhaar, pan, card, ifsc detectors (Task 2)
│   ├── contact.py           # phone, upi, email, bank_account detectors (Task 3)
│   ├── secrets.py           # secret detectors + shannon_entropy (Task 4)
│   └── engine.py            # DETECTORS, detect_all (Task 4)
├── data/upi_handles.txt     # known UPI handles, one per line (Task 3)
├── policy.py                # Action, FailureMode, Policy, load_policy (Task 5)
├── masking.py               # PlaceholderMap, mask_messages, restore_response (Task 6)
├── config.py                # + policy_path (Task 5)
├── logs.py                  # + findings, blocked, detection_ms (Task 7)
└── app.py                   # wire it all in (Task 7)
config/policy.yaml           # documented default policy (Task 5)
tests/detect/test_spans_checksums.py, test_ids.py, test_contact.py, test_secrets.py
tests/test_policy.py, tests/test_masking.py, tests/test_guard_masking.py
```

---

### Task 1: Entity types, spans, merging, checksums

**Files:** Create `src/leakguard/entities.py`, `src/leakguard/detect/__init__.py`, `src/leakguard/detect/spans.py`, `src/leakguard/detect/checksums.py`. Test: `tests/detect/__init__.py`, `tests/detect/test_spans_checksums.py`.

**Interfaces:**
- Produces:
  - `class EntityType(StrEnum)` with the 11 names above (value == name). `PRIORITY: dict[EntityType, int]` per Global Constraints.
  - `@dataclass(frozen=True) class Span: start: int; end: int; type: EntityType; score: float = 1.0; detector: str = ""` with `def text(self, source: str) -> str`.
  - `merge_spans(spans: Iterable[Span]) -> list[Span]` — greedy: sort by (−priority, −length, start); keep a span if it overlaps no kept span; return sorted by `start`.
  - `verhoeff_valid(digits: str) -> bool`, `luhn_valid(digits: str) -> bool` (digits-only strings; anything else → False).

- [ ] **Step 1: Write failing tests**

```python
from leakguard.detect.checksums import luhn_valid, verhoeff_valid
from leakguard.detect.spans import Span, merge_spans
from leakguard.entities import EntityType as T

def test_verhoeff_known_values():
    assert verhoeff_valid("2363")
    assert verhoeff_valid("234567890124") and verhoeff_valid("987654321012")
    assert not verhoeff_valid("234567890125")
    assert not verhoeff_valid("") and not verhoeff_valid("12a4")

def test_luhn_known_values():
    assert luhn_valid("4111111111111111") and luhn_valid("378282246310005")
    assert not luhn_valid("4111111111111112") and not luhn_valid("")

def test_merge_prefers_priority_then_length():
    a = Span(0, 10, T.PHONE); b = Span(0, 10, T.BANK_ACCOUNT); c = Span(20, 30, T.EMAIL)
    assert merge_spans([a, b, c]) == [b, c]
    upi = Span(5, 20, T.UPI); email = Span(5, 24, T.EMAIL)
    assert merge_spans([email, upi]) == [upi]
    short = Span(0, 4, T.PHONE); long = Span(0, 12, T.PHONE)
    assert merge_spans([short, long]) == [long]

def test_merge_keeps_non_overlapping_in_order():
    s = [Span(30, 35, T.PAN), Span(0, 5, T.PHONE), Span(5, 9, T.EMAIL)]  # touching ≠ overlapping
    assert [x.start for x in merge_spans(s)] == [0, 5, 30]
```

- [ ] **Step 2: Run** `pytest tests/detect -q` — Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** per Interfaces (standard Verhoeff `d`/`p` tables; validity = running checksum over reversed digits with `p[i % 8]` ends at 0).
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: entity types, span merging, Verhoeff and Luhn checksums`
- [ ] **Step 6: Walkthrough** — new `docs/walkthrough/phase-2-detection.md`, section 1: what a span is; why merge by priority (UPI beats email, account beats phone); how Verhoeff and Luhn work and why checksums remove ~90% of look-alikes; Q&A: *"Why does a checksum help detection?"*

---

### Task 2: ID detectors — Aadhaar, PAN, card, IFSC

**Files:** Create `src/leakguard/detect/ids.py`. Test: `tests/detect/test_ids.py`.

**Interfaces:**
- Consumes: `Span`, `EntityType`, checksums (Task 1).
- Produces: `detect_aadhaar(text) -> list[Span]`, `detect_pan(text)`, `detect_card(text)`, `detect_ifsc(text)`; `detector` field = function's short name (`"aadhaar"`, …).
- Rules: Aadhaar = 12 digits, first 2–9, optional single space/hyphen between 4-digit groups, not touching other digits, Verhoeff-valid. PAN = 5 letters, 4 digits, 1 letter, 4th letter in `{A,B,C,F,G,H,J,L,P,T}`, case-insensitive, not touching letters/digits. Card = 13–19 digits with optional single space/hyphen between digits, not touching digits, Luhn-valid. IFSC = 4 letters + `0` + 6 alphanumerics, case-insensitive, not touching letters/digits.

- [ ] **Step 1: Write failing tests**

```python
from leakguard.detect.ids import detect_aadhaar, detect_card, detect_ifsc, detect_pan
from leakguard.entities import EntityType as T

def _vals(spans, text):
    return [s.text(text) for s in spans]

def test_aadhaar_formats():
    text = "My Aadhaar is 2345 6789 0124, old one 9876-5432-1012, also 234567890124."
    assert _vals(detect_aadhaar(text), text) == ["2345 6789 0124", "9876-5432-1012", "234567890124"]
    assert all(s.type is T.AADHAAR for s in detect_aadhaar(text))

def test_aadhaar_rejects_bad_checksum_and_lookalikes():
    for text in ["order 123456789012", "2345 6789 0125", "0345 6789 0124",
                 "1234567890124", "id 12345678901234"]:
        assert detect_aadhaar(text) == [], text

def test_pan():
    text = "PAN: ABCPS1234K and abcps1234k"
    assert _vals(detect_pan(text), text) == ["ABCPS1234K", "abcps1234k"]
    assert detect_pan("ABCXS1234K") == []          # X is not a holder type
    assert detect_pan("XABCPS1234K") == []          # touching letters

def test_card_luhn():
    text = "card 4111 1111 1111 1111 or 4111-1111-1111-1111 or 378282246310005"
    assert len(detect_card(text)) == 3
    assert detect_card("4111 1111 1111 1112") == []

def test_ifsc():
    text = "IFSC HDFC0001234, sbin0005943"
    assert _vals(detect_ifsc(text), text) == ["HDFC0001234", "sbin0005943"]
    assert detect_ifsc("HDFC1001234") == [] and detect_ifsc("XHDFC0001234") == []
```

- [ ] **Step 2: Run** `pytest tests/detect/test_ids.py -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** `ids.py` per Interfaces.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: Aadhaar, PAN, card and IFSC detectors`
- [ ] **Step 6: Walkthrough** section 2: each pattern explained piece by piece (lookbehind/lookahead "not touching digits"); why PAN's 4th letter check; why IFSC's literal `0`; Q&A: *"How do you avoid flagging order numbers as Aadhaar?"*

---

### Task 3: Contact and payment detectors — phone, UPI, email, bank account

**Files:** Create `src/leakguard/detect/contact.py`, `src/leakguard/data/upi_handles.txt`; modify `pyproject.toml` (package data `data/*.txt`). Test: `tests/detect/test_contact.py`.

**Interfaces:**
- Produces: `detect_phone(text)`, `detect_upi(text)`, `detect_email(text)`, `detect_bank_account(text)`; `UPI_HANDLES: frozenset[str]` loaded via `importlib.resources`.
- Rules:
  - PHONE: optional `+91` / `91` / `0` prefix (with optional space/hyphen), then 10 digits starting 6–9 with optional single space/hyphen between digits, not touching digits. Span covers prefix + number.
  - UPI: `local@handle`, local = `[A-Za-z0-9._-]{2,}`, handle letters only and in `UPI_HANDLES` (case-insensitive), handle not followed by `.`, letters or digits.
  - EMAIL: standard `local@domain.tld`.
  - BANK_ACCOUNT: 9–18 digits with optional single space/hyphen between digits, not touching digits, **only if** within 40 characters before it there is context matching `a/c`, `ac no`, `acc`, `account`, `acct`, `khata` or `bank` (case-insensitive, word-bounded), **or** an IFSC code appears within 50 characters before or after.
  - `upi_handles.txt`: at least `ybl ibl axl okicici okaxis oksbi okhdfcbank paytm ptyes ptaxis pthdfc ptsbi apl yapl rapl upi sbi icici hdfcbank axisbank kotak idfcbank yesbank pnb barodampay unionbank cnrb freecharge jupiteraxis fbl waaxis wahdfcbank wasbi waicici`.

- [ ] **Step 1: Write failing tests**

```python
from leakguard.detect.contact import (detect_bank_account, detect_email, detect_phone,
                                      detect_upi)

def _vals(spans, text):
    return [s.text(text) for s in spans]

def test_phone_formats():
    text = "call 9876543210, +91 98765 43210, +91-9876543210, 09876543210 or 98765-43210"
    assert _vals(detect_phone(text), text) == [
        "9876543210", "+91 98765 43210", "+91-9876543210", "09876543210", "98765-43210"]

def test_phone_rejects_lookalikes():
    for text in ["OTP 482913", "id 5876543210", "98765432101234", "ref 1234567890"]:
        assert detect_phone(text) == [], text

def test_upi_vs_email():
    text = "pay rahul.s@okicici or 9876543210@ybl, mail rahul@gmail.com, x@unknownbank"
    assert _vals(detect_upi(text), text) == ["rahul.s@okicici", "9876543210@ybl"]
    assert _vals(detect_email(text), text) == ["rahul@gmail.com"]
    assert detect_upi("me@okicici.com") == []          # a domain, not a UPI handle

def test_bank_account_with_context():
    text = "My a/c no 50100123456789 and account number 1234 5678 9012"
    assert _vals(detect_bank_account(text), text) == ["50100123456789", "1234 5678 9012"]
    text2 = "Send to 30012345678 IFSC SBIN0005943"
    assert _vals(detect_bank_account(text2), text2) == ["30012345678"]

def test_bank_account_requires_context():
    for text in ["Ticket 50100123456789 is open", "order no 123456789012 shipped"]:
        assert detect_bank_account(text) == [], text
```

- [ ] **Step 2: Run** `pytest tests/detect/test_contact.py -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** per Interfaces; add `[tool.setuptools.package-data] leakguard = ["data/*.txt"]`, reinstall (`pip install -e ".[dev]"`).
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: phone, UPI, email and bank account detectors`
- [ ] **Step 6: Walkthrough** section 3: phone prefixes; why UPI needs a handle list (vs email); why bank accounts need context (no checksum) and why account beats phone (banks use mobile numbers as account numbers); the OTP trap; Q&A: *"How do you tell a bank account number from any long number?"*

---

### Task 4: Secret detectors and the combined engine

**Files:** Create `src/leakguard/detect/secrets.py`, `src/leakguard/detect/engine.py`; modify `src/leakguard/detect/__init__.py`. Test: `tests/detect/test_secrets.py`.

**Interfaces:**
- Produces:
  - `shannon_entropy(s: str) -> float` (bits per character).
  - `detect_secrets(text) -> list[Span]` (type `SECRET`), combining:
    - known prefixes: OpenAI/Anthropic `sk-…` (≥20 chars after `sk-`), AWS `AKIA|ASIA` + 16 `[0-9A-Z]`, GitHub `gh[pousr]_` + ≥36 alnum and `github_pat_…`, Slack `xox[baprs]-…`, Google `AIza` + 35, Stripe `(sk|rk)_(live|test)_…`, JWT `eyJ….eyJ….…`, PEM header `-----BEGIN … PRIVATE KEY-----`;
    - key/value: `password|passwd|pwd|secret|api_key|api-key|apikey|access_token|auth_token|client_secret` followed by `:` or `=` then a value (span = the value) that — after stripping trailing `.,;:!?` — is ≥4 characters **and contains a digit or symbol** (so "reset password: how?" or "password: please help" are not secrets; pure-letter passwords are a documented miss, because a false *block* breaks the chatbot); `Authorization: Bearer <token ≥8>` (span = token);
    - entropy: tokens of `[A-Za-z0-9+/_=-]` with length ≥ 24, containing lowercase, uppercase **and** a digit, entropy > 4.0 (pure hex like git SHAs / UUIDs max out at 4.0 and are excluded).
  - `detect_secrets` merges its own overlapping hits (a key can match a prefix rule *and* entropy) so each secret is one span.
  - Every pattern that can start anywhere (UPI/email local parts, entropy tokens, digit runs) is anchored with a "not preceded by a token character" lookbehind, so a failed match can't be retried from every position (that is what makes `"a" * 100_000 + "@"` quadratic).
  - `DETECTORS: tuple[Callable[[str], list[Span]], ...]` and `detect_all(text: str) -> list[Span]` = `merge_spans` over all detectors. `leakguard.detect.detect_all` re-exported.

- [ ] **Step 1: Write failing tests**

```python
import time
from leakguard.detect import detect_all
from leakguard.detect.secrets import detect_secrets, shannon_entropy
from leakguard.entities import EntityType as T

def test_entropy():
    assert shannon_entropy("aaaa") == 0.0
    assert abs(shannon_entropy("abcd") - 2.0) < 1e-9

def test_known_key_prefixes():
    for key in ["sk-proj-abcDEF1234567890abcdefXYZ", "AKIAIOSFODNN7EXAMPLE",
                "ghp_" + "a1B2" * 9, "AIza" + "Sy" + "A1b2C3d4E5" * 3 + "xyz"]:
        assert [s.type for s in detect_secrets(f"use {key} please")] == [T.SECRET], key

def test_key_value_and_bearer():
    text = 'password=hunter22 and api_key: "zx81" Authorization: Bearer abc.def.ghi123'
    assert len(detect_secrets(text)) == 3

def test_natural_language_password_mentions_not_flagged():
    for text in ["my password is wrong", "reset password: how?", "I forgot my password",
                 "secret santa list", "the api key expired yesterday"]:
        assert detect_secrets(text) == [], text

def test_entropy_catches_random_tokens_not_hex_ids():
    assert detect_secrets("token Zx8Kq2LmP9wRt4Vb7Nc1Hy6Jd3Fg") != []
    assert detect_secrets("commit 9fceb02d0ae598e95dc970b74767f19372d61af8") == []
    assert detect_secrets("id 550e8400-e29b-41d4-a716-446655440000") == []

def test_detect_all_merges_everything():
    text = "Aadhaar 2345 6789 0124, a/c 9876543210, key sk-abcdefghijklmnopqrstuvwxyz12"
    assert [s.type for s in detect_all(text)] == [T.AADHAAR, T.BANK_ACCOUNT, T.SECRET]

def test_detect_all_is_fast_on_adversarial_input():
    for text in ["9 " * 50_000, "a" * 100_000 + "@", "-" * 100_000, "A1b2" * 25_000]:
        started = time.perf_counter()
        detect_all(text)
        assert time.perf_counter() - started < 2.0
```

- [ ] **Step 2: Run** `pytest tests/detect/test_secrets.py -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: secret detection and combined detection engine`
- [ ] **Step 6: Walkthrough** section 4: three ways to find secrets; Shannon entropy with a worked example; why hex is excluded; why "password is …" isn't matched (false blocks break the chatbot); what ReDoS is and how the test guards it; Q&A: *"How do you detect an API key format you've never seen?"*

---

### Task 5: Policy

**Files:** Create `src/leakguard/policy.py`, `config/policy.yaml`; modify `src/leakguard/config.py`. Test: `tests/test_policy.py`.

**Interfaces:**
- Produces:
  - `class Action(StrEnum)`: `MASK="mask"`, `BLOCK="block"`, `LOG_ONLY="log_only"`; `class FailureMode(StrEnum)`: `CLOSED="closed"`, `OPEN="open"`.
  - `@dataclass(frozen=True) class Policy: actions: Mapping[EntityType, Action]; failure_mode: FailureMode = FailureMode.CLOSED` with `def action_for(self, t: EntityType) -> Action`.
  - `DEFAULT_POLICY: Policy` (personal types `mask`, `SECRET` `block`, closed).
  - `class PolicyError(ValueError)`; `load_policy(path: str | os.PathLike | None) -> Policy` — `None` → `DEFAULT_POLICY`; YAML keys `actions` (partial map; missing types keep defaults) and `failure_mode`; unknown type / action / failure mode / top-level key → `PolicyError` naming the bad value.
  - `Settings.policy_path: str | None = None`, env `LEAKGUARD_POLICY_PATH`.
  - `config/policy.yaml`: every type with its default action and a comment, `failure_mode: closed`.

- [ ] **Step 1: Write failing tests**

```python
import pytest
from leakguard.config import Settings
from leakguard.entities import EntityType as T
from leakguard.policy import (DEFAULT_POLICY, Action, FailureMode, PolicyError,
                              load_policy)

def test_default_policy():
    assert DEFAULT_POLICY.action_for(T.SECRET) is Action.BLOCK
    assert DEFAULT_POLICY.action_for(T.AADHAAR) is Action.MASK
    assert DEFAULT_POLICY.failure_mode is FailureMode.CLOSED
    assert load_policy(None) == DEFAULT_POLICY

def test_partial_yaml_overrides_defaults(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("actions:\n  NAME: log_only\nfailure_mode: open\n")
    pol = load_policy(p)
    assert pol.action_for(T.NAME) is Action.LOG_ONLY
    assert pol.action_for(T.PHONE) is Action.MASK
    assert pol.failure_mode is FailureMode.OPEN

@pytest.mark.parametrize("yaml_text,bad", [
    ("actions:\n  PASSPORT: mask\n", "PASSPORT"),
    ("actions:\n  PHONE: hide\n", "hide"),
    ("failure_mode: maybe\n", "maybe"),
    ("colour: blue\n", "colour"),
])
def test_invalid_policy_rejected(tmp_path, yaml_text, bad):
    p = tmp_path / "p.yaml"
    p.write_text(yaml_text)
    with pytest.raises(PolicyError, match=bad):
        load_policy(p)

def test_shipped_policy_file_matches_defaults():
    assert load_policy("config/policy.yaml") == DEFAULT_POLICY

def test_settings_policy_path_from_env():
    assert Settings.from_env({"LEAKGUARD_POLICY_PATH": "x.yaml"}).policy_path == "x.yaml"
    assert Settings.from_env({}).policy_path is None
```

- [ ] **Step 2: Run** `pytest tests/test_policy.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** per Interfaces (`yaml.safe_load`; empty file → defaults).
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: YAML policy with per-type actions and failure mode`
- [ ] **Step 6: Walkthrough** section 5: the three actions and when each is used; fail-closed vs fail-open; why invalid config must fail at startup, not at request time; why `yaml.safe_load`; Q&A: *"What does fail-closed mean and why is it the default?"*

---

### Task 6: Masking and restoring

**Files:** Create `src/leakguard/masking.py`. Test: `tests/test_masking.py`.

**Interfaces:**
- Consumes: `detect_all`, `Span`, `EntityType`, `Policy`, `Action`.
- Produces:
  - `class PlaceholderMap(reserved: Mapping[EntityType, set[int]] | None = None)`: `placeholder_for(t: EntityType, value: str) -> str` (same **normalised** value → same placeholder; numbers skip `reserved[t]`); `restore(text: str) -> str` (replaces only placeholders it created); `__len__`.
    - Normalisation: digit types (`AADHAAR, PHONE, CARD, BANK_ACCOUNT`) → digits only, and `PHONE` → last 10 digits; others → `casefold()` of stripped text.
  - `PLACEHOLDER_RE` = `\[([A-Z_]+)_(\d+)\]`.
  - `@dataclass class MaskResult: messages: list; mapping: PlaceholderMap; counts: Counter[tuple[EntityType, Action]]; blocked: list[EntityType]` (sorted, unique).
  - `mask_messages(messages: list[dict], policy: Policy, detect: Callable[[str], list[Span]] = detect_all) -> MaskResult` — returns **new** message dicts (input untouched). Text fields: `content` when a string; `text` of each `{"type": "text"}` part; `tool_calls[].function.arguments` strings. Before numbering, every existing placeholder in any text field is reserved.
  - `restore_response(body: dict, mapping: PlaceholderMap) -> dict` — new dict with `choices[].message.content` and `choices[].message.tool_calls[].function.arguments` restored.

- [ ] **Step 1: Write failing tests**

```python
from leakguard.entities import EntityType as T
from leakguard.masking import PlaceholderMap, mask_messages, restore_response
from leakguard.policy import DEFAULT_POLICY, Action, Policy

def _user(content):
    return [{"role": "user", "content": content}]

def test_masks_and_numbers_in_order_of_appearance():
    msgs = [{"role": "system", "content": "Bot"},
            {"role": "user", "content": "I'm at 9876543210 and pan ABCPS1234K"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "also 9123456789, and again 9876543210"}]
    r = mask_messages(msgs, DEFAULT_POLICY)
    assert r.messages[1]["content"] == "I'm at [PHONE_1] and pan [PAN_1]"
    assert r.messages[3]["content"] == "also [PHONE_2], and again [PHONE_1]"
    assert msgs[1]["content"] == "I'm at 9876543210 and pan ABCPS1234K"   # input untouched
    assert r.counts[(T.PHONE, Action.MASK)] == 3 and r.blocked == []

def test_same_value_different_format_same_placeholder():
    r = mask_messages(_user("98765 43210 or +91 9876543210"), DEFAULT_POLICY)
    assert r.messages[0]["content"] == "[PHONE_1] or [PHONE_1]"

def test_restore_round_trip():
    r = mask_messages(_user("upi rahul.s@okicici"), DEFAULT_POLICY)
    assert r.mapping.restore("refund to [UPI_1] done") == "refund to rahul.s@okicici done"

def test_user_typed_placeholder_is_never_restored():
    r = mask_messages(_user("what is [PHONE_1]? my number is 9876543210"), DEFAULT_POLICY)
    assert r.messages[0]["content"] == "what is [PHONE_1]? my number is [PHONE_2]"
    assert r.mapping.restore("[PHONE_1] / [PHONE_2] / [PHONE_9]") == \
        "[PHONE_1] / 9876543210 / [PHONE_9]"

def test_secret_blocks_and_log_only_leaves_text():
    pol = Policy(actions={**DEFAULT_POLICY.actions, T.EMAIL: Action.LOG_ONLY})
    r = mask_messages(_user("mail a@b.com key sk-abcdefghijklmnopqrstuvwxyz12"), pol)
    assert r.blocked == [T.SECRET]
    assert "a@b.com" in r.messages[0]["content"]
    assert r.counts[(T.EMAIL, Action.LOG_ONLY)] == 1

def test_masks_content_parts_and_tool_call_arguments():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "pan ABCPS1234K"},
                                          {"type": "image_url", "image_url": {"url": "x"}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
             "function": {"name": "f", "arguments": '{"phone": "9876543210"}'}}]}]
    r = mask_messages(msgs, DEFAULT_POLICY)
    assert r.messages[0]["content"][0]["text"] == "pan [PAN_1]"
    assert r.messages[0]["content"][1] == msgs[0]["content"][1]
    assert r.messages[1]["tool_calls"][0]["function"]["arguments"] == '{"phone": "[PHONE_1]"}'

def test_restore_response_body():
    m = PlaceholderMap()
    ph = m.placeholder_for(T.PHONE, "9876543210")
    body = {"choices": [{"message": {"role": "assistant", "content": f"call {ph}",
            "tool_calls": [{"function": {"name": "f", "arguments": f'{{"p": "{ph}"}}'}}]}}]}
    out = restore_response(body, m)
    assert out["choices"][0]["message"]["content"] == "call 9876543210"
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == \
        '{"p": "9876543210"}'
```

- [ ] **Step 2: Run** `pytest tests/test_masking.py -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: reversible masking with typed placeholders`
- [ ] **Step 6: Walkthrough** section 6: why typed numbered placeholders (not `****`); deterministic numbering → stateless; normalisation; placeholder-injection attack and the reservation defence; why inputs are never mutated; Q&A: *"What if a user types [PHONE_1] themselves?"*

---

### Task 7: Wire into the guard

**Files:** Modify `src/leakguard/app.py`, `src/leakguard/logs.py`, `tests/test_proxy.py` (log test), `README.md`. Test: `tests/test_guard_masking.py`.

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `create_app(settings, transport=None, policy: Policy | None = None)` — `policy=None` → `load_policy(settings.policy_path)` at app creation (bad config fails startup).
  - Request flow: parse → every `messages` item must be an object (else 400 `invalid_request`) → `mask_messages` (timed) → blocked? 400 `sensitive_data_blocked` naming types → forward masked body → non-streaming 2xx JSON reply: `restore_response`; streaming: forwarded masked, **not restored until Phase 6** (README states it).
  - Detection exception: `failure_mode closed` → 500 `guard_error` / `detection_failed`; `open` → forward original body, `logger.warning` (request id only).
  - `log_request(..., findings: Mapping[str, int] | None = None, blocked: bool = False, detection_ms: float | None = None)` — `findings` keys like `"PHONE:mask"`.

- [ ] **Step 1: Write failing tests** in `tests/test_guard_masking.py`

```python
import json
import httpx
from leakguard.entities import EntityType as T
from leakguard.policy import DEFAULT_POLICY, FailureMode, Policy

CHAT = "/v1/chat/completions"

def _req(content, **extra):
    return {"model": "m", "messages": [{"role": "user", "content": content}], **extra}

def test_end_to_end_mask_and_restore(make_guard):
    seen = {}
    def handler(request):
        body = json.loads(request.content)
        seen["sent"] = body["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": "Refund to " + seen["sent"].split()[-1]}}]})
    guard = make_guard(transport=httpx.MockTransport(handler))
    r = guard.post(CHAT, json=_req("my upi is rahul.s@okicici"))
    assert seen["sent"] == "my upi is [UPI_1]"
    assert r.json()["choices"][0]["message"]["content"] == "Refund to rahul.s@okicici"

def test_with_mock_llm_round_trip(guard):
    r = guard.post(CHAT, json=_req("Aadhaar 2345 6789 0124"))
    assert r.json()["choices"][0]["message"]["content"] == "You said: Aadhaar 2345 6789 0124"

def test_secret_blocked_with_400_naming_type_not_value(make_guard):
    calls = []
    guard = make_guard(transport=httpx.MockTransport(lambda req: calls.append(1)))
    r = guard.post(CHAT, json=_req("debug with sk-abcdefghijklmnopqrstuvwxyz12"))
    assert r.status_code == 400 and calls == []
    err = r.json()["error"]
    assert err["code"] == "sensitive_data_blocked" and "SECRET" in err["message"]
    assert "sk-abc" not in err["message"]

def test_non_object_message_items_rejected(guard):
    r = guard.post(CHAT, json={"model": "m", "messages": ["hi"]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"

def test_detection_failure_fails_closed(make_guard, monkeypatch):
    import leakguard.app as app_module
    def boom(*a, **k):
        raise RuntimeError("detector crashed")
    monkeypatch.setattr(app_module, "mask_messages", boom)
    r = make_guard().post(CHAT, json=_req("hi"))
    assert r.status_code == 500 and r.json()["error"]["code"] == "detection_failed"

def test_detection_failure_open_mode_forwards(make_guard, monkeypatch):
    import leakguard.app as app_module
    monkeypatch.setattr(app_module, "mask_messages",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    pol = Policy(actions=DEFAULT_POLICY.actions, failure_mode=FailureMode.OPEN)
    r = make_guard(policy=pol).post(CHAT, json=_req("hi"))
    assert r.status_code == 200

def test_streaming_request_is_masked(make_guard):
    seen = {}
    def handler(request):
        seen["sent"] = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(200, content=b"data: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})
    with make_guard(transport=httpx.MockTransport(handler)).stream(
            "POST", CHAT, json=_req("pan ABCPS1234K", stream=True)) as r:
        r.read()
    assert seen["sent"] == "pan [PAN_1]"

def test_logs_counts_but_never_values(guard, caplog):
    caplog.set_level("INFO", logger="leakguard")
    guard.post(CHAT, json=_req("phone 9876543210 pan ABCPS1234K"))
    assert "9876543210" not in caplog.text and "ABCPS1234K" not in caplog.text
    assert '"PHONE:mask": 1' in caplog.text and '"PAN:mask": 1' in caplog.text
```

`make_guard` gains a `policy` keyword passed to `create_app`.

- [ ] **Step 2: Run** `pytest tests/test_guard_masking.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** per Interfaces; update `conftest.make_guard` to accept `policy`.
- [ ] **Step 4: Run** `pytest -q` and `ruff check .` — Expected: all pass, clean.
- [ ] **Step 5: Verify with Docker** — `docker compose up -d --build`; `python examples/chat.py` → reply contains the real phone number (restored); `docker compose logs guard` shows `"PHONE:mask": 1` and no phone number; `docker compose down`.
- [ ] **Step 6: Update README** (status → Phase 2; what's detected; policy file + `LEAKGUARD_POLICY_PATH`; streaming restore arrives in Phase 6) and commit `feat: mask, block and restore in the guard`.
- [ ] **Step 7: Walkthrough** section 7: the full request journey now; fail-closed in code; why streaming isn't restored yet (and that it's still *safe*: data is masked, only the display shows placeholders); Q&A: *"Walk me through what happens to a request containing an Aadhaar number."*
