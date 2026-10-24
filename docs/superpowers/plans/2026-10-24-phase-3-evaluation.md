# Phase 3 — Evaluation v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Honest, reproducible numbers: a seeded synthetic dataset (English, Hinglish, pasted records, developer snippets, look-alike negatives, known-hard cases) split dev/test **by template**, a small hand-written set, a scoring harness (per-type precision/recall/F1, coverage, false-positive rate on negatives, latency), and a generated Markdown report — then fix what the dev split reveals.

**Architecture:** A `leakguard.evaluation` subpackage. Datasets are JSONL of `Sample`s whose gold spans come from a single "marked text" format (`[[PHONE:98765 43210]]`) used by both the generator and the hand-written set. Templates contain typed slots (`{PHONE}`, `{SECRET.aws}`) and unlabelled filler slots (`{~ORDER_ID}`); each template belongs to exactly one split. `score()` compares any `detect(text) -> list[Span]` against gold, so Phases 4–7 (NER, addresses, Presidio) reuse the same harness.

**Tech Stack:** stdlib only (`random`, `json`, `statistics`, `time`).

**Spec:** [docs/superpowers/specs/2026-10-24-data-leak-guard-design.md](../specs/2026-10-24-data-leak-guard-design.md) — implements §7.1, §7.2 (detection metrics + latency), §7.4; anti-goals "no invented numbers" and "no tuning on test".

## Global Constraints

- All data synthetic; generated IDs valid where real ones would be (Verhoeff, Luhn, PAN structure, IFSC `0`).
- Deterministic: same seed → byte-identical JSONL.
- **Split by template**: no template text appears in both dev and test.
- Tuning decisions use **dev only**; test + hand-written numbers are only *reported*.
- Matching rule (stated in the report): a prediction is a true positive when it has the **same type** and **overlaps** a gold span (each gold/prediction matched at most once, greedy by overlap size). Also reported: exact-boundary rate and **coverage** = share of gold characters masked by *any* prediction.
- `NAME` and `ADDRESS` are labelled in the data now (so Phases 4–5 measure on the same data) and reported as "detector not built yet".
- Commands (run from repo root): `python -m leakguard.evaluation generate` → `data/eval/dev.jsonl`, `data/eval/test.jsonl`; `python -m leakguard.evaluation report` → `docs/evaluation/report.md`; `python -m leakguard.evaluation errors --split dev` → error analysis to stdout.

## Review Focus

1. **Generator/detector circularity** — if tests validate generated values *with the detectors*, a detector bug becomes invisible. Generator tests must use checksums/structure only. Test: `test_generated_values_are_valid_by_construction` (Task 2).
2. **Spans misaligned with text** (off-by-one, multi-line, Unicode `₹`) silently corrupt every metric. Test: `test_every_span_matches_its_value` (Task 3).
3. **Template leakage between splits** inflates test numbers. Test: `test_splits_share_no_templates` (Task 3).
4. **Double counting** — one prediction matching two golds (or vice versa) inflates recall. Test: `test_one_prediction_matches_one_gold` (Task 1).
5. **A detector crash on odd input** during evaluation should surface, not be swallowed. Test: `test_score_propagates_detector_errors` (Task 1).

---

## File Structure

```
src/leakguard/evaluation/
├── __init__.py
├── samples.py      # Sample, parse_marked(), read_jsonl(), write_jsonl()      (Task 1)
├── metrics.py      # TypeScores, Evaluation, score()                         (Task 1)
├── values.py       # synthetic value generators                              (Task 2)
├── templates.py    # TEMPLATES by category                                    (Task 3)
├── generate.py     # render_template(), build_split()                         (Task 3)
├── report.py       # measure_latency(), render_report(), render_errors()       (Task 5)
└── __main__.py     # CLI: generate | report | errors                          (Task 5)
data/eval/dev.jsonl, test.jsonl          # generated, committed                (Task 3)
data/eval/handwritten.jsonl              # written by hand                      (Task 4)
docs/evaluation/report.md                # generated, committed                (Task 5)
tests/evaluation/test_samples_metrics.py, test_values.py, test_generate.py,
                 test_handwritten.py, test_report.py
```

---

### Task 1: Samples, marked text, scoring

**Files:** Create `src/leakguard/evaluation/__init__.py`, `samples.py`, `metrics.py`. Test: `tests/evaluation/__init__.py`, `tests/evaluation/test_samples_metrics.py`.

**Interfaces:**
- `@dataclass(frozen=True) class Sample: id: str; category: str; template: str; text: str; spans: tuple[Span, ...]` (gold spans use `detector="gold"`).
- `parse_marked(marked: str) -> tuple[str, tuple[Span, ...]]` — `[[TYPE:value]]` → `value` in the clean text, with a gold span; unknown TYPE → `ValueError`.
- `write_jsonl(path, samples)`, `read_jsonl(path) -> list[Sample]`; JSON keys `id, category, template, text, spans:[{start,end,type}]`; UTF-8, one sample per line, `ensure_ascii=False`, keys in that order.
- `@dataclass class TypeScores: tp=0; fp=0; fn=0; exact=0; gold_chars=0; covered_chars=0` with properties `precision`, `recall`, `f1`, `coverage` (0.0 when undefined) and `support` (= tp + fn).
- `@dataclass class Evaluation: by_type: dict[EntityType, TypeScores]; by_category: dict[str, TypeScores]; negatives_total: int; negatives_with_fp: int; errors: list[ErrorCase]` with `@dataclass(frozen=True) class ErrorCase: sample_id: str; kind: str  # "FP" | "FN"; type: EntityType; value: str; text: str`.
- `score(samples: Iterable[Sample], detect: Callable[[str], list[Span]]) -> Evaluation` — matching rule from Global Constraints; `by_category` aggregates all types; a "negative" sample is one with category `"negative"`; detector exceptions propagate.

- [ ] **Step 1: Write failing tests**

```python
import pytest
from leakguard.detect.spans import Span
from leakguard.entities import EntityType as T
from leakguard.evaluation.metrics import score
from leakguard.evaluation.samples import Sample, parse_marked, read_jsonl, write_jsonl

def _sample(marked, category="support", sid="s1"):
    text, spans = parse_marked(marked)
    return Sample(sid, category, "tpl", text, spans)

def test_parse_marked():
    text, spans = parse_marked("call [[PHONE:98765 43210]] or mail [[EMAIL:a@b.in]]")
    assert text == "call 98765 43210 or mail a@b.in"
    assert [(s.type, text[s.start:s.end]) for s in spans] == [
        (T.PHONE, "98765 43210"), (T.EMAIL, "a@b.in")]
    with pytest.raises(ValueError):
        parse_marked("[[PASSPORT:X1]]")

def test_jsonl_round_trip(tmp_path):
    s = _sample("₹500 to [[UPI:ravi@ybl]]\nthanks")
    write_jsonl(tmp_path / "x.jsonl", [s])
    assert read_jsonl(tmp_path / "x.jsonl") == [s]

def test_overlap_same_type_is_tp_and_coverage_counts_chars():
    s = _sample("a/c [[BANK_ACCOUNT:50100123456789]]")
    gold = s.spans[0]
    partial = Span(gold.start, gold.end - 4, T.BANK_ACCOUNT)
    ev = score([s], lambda text: [partial])
    st = ev.by_type[T.BANK_ACCOUNT]
    assert (st.tp, st.fp, st.fn, st.exact) == (1, 0, 0, 0)
    assert st.coverage == pytest.approx(10 / 14)

def test_wrong_type_is_fp_and_fn_but_still_covers():
    s = _sample("a/c [[BANK_ACCOUNT:9876543210]]")
    g = s.spans[0]
    ev = score([s], lambda text: [Span(g.start, g.end, T.PHONE)])
    assert ev.by_type[T.BANK_ACCOUNT].fn == 1 and ev.by_type[T.PHONE].fp == 1
    assert ev.by_type[T.BANK_ACCOUNT].coverage == 1.0

def test_one_prediction_matches_one_gold():
    s = _sample("[[PHONE:9876543210]] [[PHONE:9123456789]]")
    whole = Span(0, len(s.text), T.PHONE)
    st = score([s], lambda text: [whole]).by_type[T.PHONE]
    assert (st.tp, st.fn) == (1, 1)

def test_negatives_fp_rate_and_errors():
    neg = _sample("order 234567890124", category="negative", sid="n1")
    ev = score([neg], lambda text: [Span(6, 18, T.AADHAAR)])
    assert (ev.negatives_total, ev.negatives_with_fp) == (1, 1)
    assert ev.errors[0].kind == "FP" and ev.errors[0].value == "234567890124"

def test_score_propagates_detector_errors():
    def broken(text):
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        score([_sample("hi")], broken)
```

- [ ] **Step 2: Run** `pytest tests/evaluation -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: evaluation samples, marked-text format and span scoring`
- [ ] **Step 6: Walkthrough** — new `docs/walkthrough/phase-3-evaluation.md` §1: precision/recall/F1 with a worked example; why overlap matching + coverage (exact boundaries matter less than "was it hidden?"); why one-to-one matching; Q&A *"Why report coverage as well as recall?"*

---

### Task 2: Synthetic value generators

**Files:** Create `src/leakguard/evaluation/values.py`. Test: `tests/evaluation/test_values.py`.

**Interfaces:**
- `verhoeff_check_digit(digits: str) -> str` (inverse table `0,4,3,2,1,5,6,7,8,9`).
- `VALUE_MAKERS: dict[str, Callable[[random.Random], str]]` keyed by slot name. Slot names = `TYPE` or `TYPE.variant` (gold) and filler names (no `~`). Required keys:
  - gold: `AADHAAR` (random spacing: none/space/hyphen), `AADHAAR.mixed` (mixed separators), `PAN`, `PAN.spaced` (`ABCPS 1234 K`), `PHONE` (plain/`+91 `/`+91-`/`0`/`5-5` grouping), `PHONE.words` (digits spelled in English words), `PHONE.dots` (`98765.43210`), `UPI`, `UPI.unknown_handle` (handle not in `UPI_HANDLES`), `EMAIL`, `CARD` (16-digit, Luhn, grouped or not), `IFSC` (real bank codes, `0`, 6 digits), `BANK_ACCOUNT` (9–18 digits), `NAME` (Indian first + last), `ADDRESS` (formal / landmark / Hinglish styles, with or without PIN), `SECRET.openai`, `SECRET.aws`, `SECRET.github`, `SECRET.slack`, `SECRET.google`, `SECRET.stripe`, `SECRET.jwt`, `SECRET.pem`, `SECRET.password` (mixed chars with digit/symbol), `SECRET.letters` (letters only), `SECRET.token` (random 32 base62);
  - filler: `ORDER_ID` (random 12 digits, **unfiltered**), `OTP` (6 digits), `TICKET` (random 14 digits), `TIMESTAMP` (`YYYYMMDDhhmmss`), `INVOICE`, `DATE`, `AMOUNT` (`₹15,000` / `Rs. 2,499`), `SHA` (40 hex), `UUID`, `TOLLFREE` (`1800-xxx-xxxx`), `CAMEL` (camelCase identifier from words), `TRACKING` (`EM123456789IN`), `COUPON`, `BANK` (bank name).

- [ ] **Step 1: Write failing tests**

```python
import random, re
from leakguard.detect.checksums import luhn_valid, verhoeff_valid
from leakguard.detect.contact import UPI_HANDLES
from leakguard.evaluation.values import VALUE_MAKERS, verhoeff_check_digit

def _many(key, n=200):
    rng = random.Random(7)
    return [VALUE_MAKERS[key](rng) for _ in range(n)]

def _digits(s):
    return re.sub(r"\D", "", s)

def test_verhoeff_check_digit():
    assert verhoeff_check_digit("236") == "3"
    assert all(verhoeff_valid(d + verhoeff_check_digit(d)) for d in ["23456789012", "31234567890"])

def test_generated_values_are_valid_by_construction():
    # Checked with checksums/structure only — never with the detectors themselves.
    assert all(verhoeff_valid(_digits(v)) and _digits(v)[0] in "23456789" and len(_digits(v)) == 12
               for v in _many("AADHAAR"))
    assert all(luhn_valid(_digits(v)) for v in _many("CARD"))
    assert all(re.fullmatch(r"[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]", v.upper()) for v in _many("PAN"))
    assert all(re.fullmatch(r"[A-Z]{4}0\d{6}", v) for v in _many("IFSC"))
    phones = [_digits(v)[-10:] for v in _many("PHONE")]
    assert all(len(p) == 10 and p[0] in "6789" for p in phones)
    assert all(v.split("@")[1] in UPI_HANDLES for v in _many("UPI"))
    assert all(v.split("@")[1] not in UPI_HANDLES for v in _many("UPI.unknown_handle"))
    assert all(9 <= len(v) <= 18 and v.isdigit() for v in _many("BANK_ACCOUNT"))
    assert all(v.isalpha() for v in _many("SECRET.letters"))

def test_every_declared_maker_exists_and_is_deterministic():
    required = {"AADHAAR", "AADHAAR.mixed", "PAN", "PAN.spaced", "PHONE", "PHONE.words",
                "PHONE.dots", "UPI", "UPI.unknown_handle", "EMAIL", "CARD", "IFSC",
                "BANK_ACCOUNT", "NAME", "ADDRESS", "SECRET.openai", "SECRET.aws",
                "SECRET.github", "SECRET.slack", "SECRET.google", "SECRET.stripe",
                "SECRET.jwt", "SECRET.pem", "SECRET.password", "SECRET.letters", "SECRET.token",
                "ORDER_ID", "OTP", "TICKET", "TIMESTAMP", "INVOICE", "DATE", "AMOUNT", "SHA",
                "UUID", "TOLLFREE", "CAMEL", "TRACKING", "COUPON", "BANK"}
    assert required <= set(VALUE_MAKERS)
    assert [VALUE_MAKERS["ADDRESS"](random.Random(1)) for _ in range(2)] == \
        [VALUE_MAKERS["ADDRESS"](random.Random(1)) for _ in range(2)]

def test_values_never_contain_marker_syntax():
    for key in VALUE_MAKERS:
        assert all("[[" not in v and "]]" not in v for v in _many(key, 50)), key
```

- [ ] **Step 2: Run** `pytest tests/evaluation/test_values.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: synthetic value generators for the evaluation set`
- [ ] **Step 6: Walkthrough** §2: why synthetic (never real Aadhaar numbers in a privacy tool); why valid checksums (otherwise detectors reject them trivially); why filler like order IDs is *unfiltered* (12-digit order numbers pass Verhoeff ~10% of the time in real life — the report should show that); why generator tests never use detectors (circularity); Q&A *"How did you get test data without real people's data?"*

---

### Task 3: Templates, dataset builder, committed datasets

**Files:** Create `src/leakguard/evaluation/templates.py`, `generate.py`, `data/eval/dev.jsonl`, `data/eval/test.jsonl`. Test: `tests/evaluation/test_generate.py`.

**Interfaces:**
- Slot grammar in templates: `{NAME}` / `{TYPE.variant}` = gold slot (rendered as `[[TYPE:value]]`, variant stripped from the type); `{~FILLER}` = unlabelled value. Literal braces aren't used in templates.
- `TEMPLATES: dict[str, list[str]]` with categories and **minimum template counts**: `support` ≥ 14, `agent_record` ≥ 6 (multi-line pasted records with ≥ 4 types each), `dev_snippet` ≥ 12 (every `SECRET.*` variant except `letters` appears), `hinglish` ≥ 12, `negative` ≥ 14 (no gold slots; must include `ORDER_ID`, `OTP`, `TICKET`, `TIMESTAMP`, `SHA`, `UUID`, `TOLLFREE`, `CAMEL`, `AMOUNT` near the word "account" or "bank", and natural sentences about passwords/api keys), `hard` ≥ 8 (uses `PHONE.words`, `PHONE.dots`, `AADHAAR.mixed`, `PAN.spaced`, `UPI.unknown_handle`, `SECRET.letters`, a `BANK_ACCOUNT` without context, and `DB_PASSWORD={SECRET.password}`).
- `SPLIT_SIZES: dict[str, int]` per split = `support 450, agent_record 150, dev_snippet 200, hinglish 250, negative 350, hard 100` (1,500 per split).
- `template_split(category: str, index: int) -> str` — even index → `"dev"`, odd → `"test"`.
- `render_template(template: str, rng: random.Random) -> str` (marked text).
- `build_split(split: str, seed: int) -> list[Sample]` — for each category, cycle that split's templates to reach the category size; ids `"{split}-{n:05d}"`; seeds: dev `1`, test `2`.

- [ ] **Step 1: Write failing tests**

```python
import random
from leakguard.entities import EntityType as T
from leakguard.evaluation.generate import SPLIT_SIZES, build_split, render_template
from leakguard.evaluation.samples import parse_marked, read_jsonl
from leakguard.evaluation.templates import TEMPLATES

def test_render_template_marks_gold_and_not_filler():
    marked = render_template("call {PHONE} about order {~ORDER_ID}", random.Random(3))
    text, spans = parse_marked(marked)
    assert [s.type for s in spans] == [T.PHONE] and "[[" not in text

def test_variant_slots_use_base_type():
    _, spans = parse_marked(render_template("key {SECRET.aws}", random.Random(3)))
    assert spans[0].type is T.SECRET

def test_build_split_sizes_and_determinism():
    dev = build_split("dev", seed=1)
    assert len(dev) == sum(SPLIT_SIZES.values())
    assert dev == build_split("dev", seed=1)

def test_every_span_matches_its_value():
    for sample in build_split("test", seed=2):
        for span in sample.spans:
            value = sample.text[span.start:span.end]
            assert value and value == value.strip(), (sample.id, value)

def test_splits_share_no_templates():
    dev = {s.template for s in build_split("dev", seed=1)}
    test = {s.template for s in build_split("test", seed=2)}
    assert dev and test and not dev & test

def test_negatives_have_no_gold_and_minimum_template_counts():
    assert all(not s.spans for s in build_split("dev", 1) if s.category == "negative")
    minimum = {"support": 14, "agent_record": 6, "dev_snippet": 12, "hinglish": 12,
               "negative": 14, "hard": 8}
    assert all(len(TEMPLATES[c]) >= n for c, n in minimum.items())

def test_committed_datasets_match_generator():
    assert read_jsonl("data/eval/dev.jsonl") == build_split("dev", seed=1)
    assert read_jsonl("data/eval/test.jsonl") == build_split("test", seed=2)
```

- [ ] **Step 2: Run** `pytest tests/evaluation/test_generate.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** templates + builder; write the two JSONL files with `write_jsonl` (a temporary inline call, or the `generate` CLI once Task 5 exists — the test above pins equality either way).
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: evaluation templates and seeded dev/test datasets`
- [ ] **Step 6: Walkthrough** §3: the categories and why each exists; slot grammar; splitting by template (and why splitting by sample isn't enough); determinism; Q&A *"How do you make sure your test numbers aren't inflated?"*

---

### Task 4: Hand-written set

**Files:** Create `data/eval/handwritten.jsonl`. Test: `tests/evaluation/test_handwritten.py`.

**Interfaces:** ≥ 40 samples written by hand in marked text, **not** derived from templates (`template: "handwritten"`, category one of the six), covering every gold type except `NAME`/`ADDRESS` at least twice, ≥ 8 negatives, and realistic messiness (typos, emojis, line breaks, mixed case, Hinglish). Stored as normal `Sample` JSONL (marked text parsed when writing).

- [ ] **Step 1: Write failing test**

```python
from collections import Counter
from leakguard.entities import EntityType as T
from leakguard.evaluation.samples import read_jsonl

def test_handwritten_set_shape():
    samples = read_jsonl("data/eval/handwritten.jsonl")
    assert len(samples) >= 40 and all(s.template == "handwritten" for s in samples)
    types = Counter(span.type for s in samples for span in s.spans)
    for t in set(T) - {T.NAME, T.ADDRESS}:
        assert types[t] >= 2, t
    assert sum(s.category == "negative" for s in samples) >= 8
    assert len({s.id for s in samples}) == len(samples)
```

- [ ] **Step 2: Run** — Expected: FAIL (file missing).
- [ ] **Step 3: Write the samples** (source kept as marked text in a script under `scripts/`? — no: write the JSONL directly via a one-off `parse_marked` call; the JSONL is the source of truth).
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat: hand-written evaluation set`
- [ ] **Step 6: Walkthrough** §4: why a hand-written set (catches a generator that's "too easy"); how to read the gap between synthetic and hand-written numbers.

---

### Task 5: Latency, report, CLI

**Files:** Create `src/leakguard/evaluation/report.py`, `__main__.py`, `docs/evaluation/report.md`. Test: `tests/evaluation/test_report.py`.

**Interfaces:**
- `measure_latency(samples, detect, repeats: int = 3) -> dict[str, dict[str, float]]` — per length bucket `"short (<200 chars)"`, `"medium (200-1000)"`, `"long (>1000)"`, `"all"`: `p50_ms`, `p95_ms`, `max_ms`, `n` (best of `repeats` per sample).
- `render_report(results: dict[str, Evaluation], latency, dataset_sizes: dict[str, int]) -> str` — Markdown with: what was measured + matching rule; per-type table per dataset (`Type | Support | Precision | Recall | F1 | Coverage | Exact`), types with zero support omitted, `NAME`/`ADDRESS` rows marked "detector not built yet (Phase 4/5)"; per-category recall; negatives FP rate; latency table; up to 5 FN and 5 FP examples per type from the hand-written set; a "known limitations" list derived from FN categories.
- `render_errors(ev: Evaluation, limit: int = 40) -> str` — plain text, grouped by `(kind, type)`.
- CLI: `python -m leakguard.evaluation generate | report | errors [--split dev|test|handwritten]`.

- [ ] **Step 1: Write failing tests**

```python
from leakguard.detect import detect_all
from leakguard.evaluation.metrics import score
from leakguard.evaluation.report import measure_latency, render_errors, render_report
from leakguard.evaluation.samples import Sample, parse_marked

def _samples():
    out = []
    for i, marked in enumerate(["call [[PHONE:9876543210]]", "order 123456789012",
                                "[[NAME:Ravi Kumar]] here", "a/c [[BANK_ACCOUNT:50100123456789]]"]):
        text, spans = parse_marked(marked)
        out.append(Sample(f"s{i}", "negative" if i == 1 else "support", "t", text, spans))
    return out

def test_report_contains_tables_rule_and_unbuilt_detectors():
    ev = score(_samples(), detect_all)
    md = render_report({"test": ev, "handwritten": ev}, measure_latency(_samples(), detect_all, 1),
                       {"test": 4, "handwritten": 4})
    assert "| Type | Support | Precision | Recall | F1 | Coverage | Exact |" in md
    assert "overlaps" in md and "same type" in md
    assert "detector not built yet" in md
    assert "p95" in md

def test_latency_buckets():
    lat = measure_latency(_samples(), detect_all, repeats=1)
    assert lat["all"]["n"] == 4 and lat["all"]["p95_ms"] >= lat["all"]["p50_ms"] >= 0

def test_render_errors_lists_misses():
    ev = score(_samples(), detect_all)
    assert "FN NAME" in render_errors(ev)
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement**; then run `python -m leakguard.evaluation generate` (must reproduce the committed JSONL — `git diff --exit-code data/eval` → no diff) and `python -m leakguard.evaluation report`.
- [ ] **Step 4: Run** `pytest -q` and `ruff check .` — Expected: pass, clean.
- [ ] **Step 5: Commit** `feat: evaluation report and CLI` (includes `docs/evaluation/report.md`).
- [ ] **Step 6: Walkthrough** §5: reading the report; p50 vs p95; why best-of-3 timing.

---

### Task 6: Error analysis on dev, fixes, publish numbers

**Files:** Modify detectors as findings require; `README.md` (results table linking the report); `docs/evaluation/report.md` (regenerated).

- [ ] **Step 1:** Run `python -m leakguard.evaluation errors --split dev` and read every FN/FP group.
- [ ] **Step 2:** For each **real detector defect** (not a documented trade-off or a `hard` case the design scopes out): write a failing unit test in `tests/detect/` reproducing it, watch it fail, fix, watch it pass. Ledger every decision to fix or to accept, with the dev numbers that justified it. **Never** look at test-split errors to choose fixes.
- [ ] **Step 3:** Regenerate the report; `pytest -q`; `ruff check .`.
- [ ] **Step 4:** README: a "Results" section with the test-split table (from the report, copied verbatim) and a link to the full report.
- [ ] **Step 5: Commit** `fix: detector improvements from dev-set error analysis` and `docs: evaluation results`.
- [ ] **Step 6: Walkthrough** §6: each finding, the decision, the before/after dev numbers; Q&A *"Tell me about a time your metrics surprised you."*
