# Phase 4 — Person-name detection (NER) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect person names (`NAME`) with a pretrained token-classification model, chosen by measurement on the dev split from several candidates, exported to ONNX, int8-quantised, run on CPU, and wired into the guard and the evaluation report.

**Architecture:** A new `leakguard.ner` package. `export.py` (offline, needs PyTorch) downloads a candidate at a pinned revision, exports ONNX, quantises to int8, checks parity, and writes a self-describing model directory (`model.onnx`, `tokenizer.json`, `ner_config.json`, `manifest.json`). At runtime, `NameModel` needs only `onnxruntime` + `tokenizers`: it tokenises with character offsets, runs the text in overlapping 512-token windows, and `decode.py` turns per-token labels into `NAME` spans. `detect.engine.build_detector(name_model)` appends it to the rule detectors; the app loads the model at startup and refuses to start if it's enabled but missing.

**Tech Stack:** runtime `onnxruntime`, `tokenizers`, `numpy`; export-only extra `torch` (CPU), `transformers`, `onnx`, `sentencepiece`, `protobuf`.

**Spec:** [docs/superpowers/specs/2026-10-24-data-leak-guard-design.md](../specs/2026-10-24-data-leak-guard-design.md) — implements §6.3 *NER (NAME)*, §9 phase 4; uses §7.1–7.2 measurement rules.

## Global Constraints

- Detection pipeline runs on **CPU**; no PyTorch in the runtime install or the final Docker image.
- Model choice and every threshold are decided on **dev**; test/handwritten are only reported. Never tune on test.
- Every reported number comes from a command that regenerates it. No invented numbers.
- Model weights are never committed (`models/` is git-ignored). They are rebuilt from a **pinned Hugging Face revision** by one command.
- Only PERSON entities are used (`B-PER`/`I-PER`, or `person` for label sets without BIO).
- Raw text and detected values are never logged (unchanged rule from Phase 2).
- Fail closed: NER enabled + model missing/corrupt → startup error; NER exception mid-request → existing `detection_failed` path.
- Candidates (all ungated). Pinned revisions are the `sha` values from the HF API on 2026-10-25:

| Key | Repo | Revision | Why |
|---|---|---|---|
| `distilbert` | `dslim/distilbert-NER` | `dfa2838a127384aabb82ed7719e16dab84c42a2a` | small/fast English |
| `bert-base` | `dslim/bert-base-NER` | `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc` | standard English baseline |
| `xlmr-hrl` | `Davlan/xlm-roberta-base-ner-hrl` | `253f557bd8249b8515114cfd7f71974fe5fa4d2f` | multilingual |
| `xlmr-indian` | `Venkatesh4342/NER-Indian-xlm-roberta` | `74ca62e07c712a42e3e49480fc6ee3ac69e0cfce` | India-context English (labels `person`, no BIO) |
| `muril-naamapadam` | `vaibhav006/indic-ner-muril-naamapadam` | `e5886d290e6d407a71a83730bc47de20fcd37ee9` | India-aware (MuRIL base, Naamapadam data) |

  `ai4bharat/IndicNER` is deliberately **not** a candidate (decided by the author, 2026-10-25): it is gated on Hugging Face (needs a logged-in account that accepted its terms), which would break "a stranger builds it with one command". The walkthrough and comparison doc state this.

## Review Focus

1. **Long messages** (>512 tokens, e.g. a pasted chat log): a name after token 512 or straddling a window edge must still be found. Test: `test_name_beyond_first_window_is_found`, `test_name_on_window_boundary_found_once` (Task 3).
2. **Subword fragments**: the model labels `Venk` as PER but `##atesh` as O; masking only `Venk` leaks the rest. Spans expand to whole words. Test: `test_span_expands_to_whole_word` (Task 2).
3. **Event-loop blocking**: ONNX inference is ~10–40 ms of CPU; run in the async handler it stalls every other request. Test: `test_slow_detection_does_not_block_other_requests` (Task 6).
4. **Odd input**: empty string, whitespace only, emoji, Devanagari, or text that tokenises to nothing must return `[]` without crashing. Test: `test_odd_inputs_return_no_spans` (Task 3).
5. **Model directory incomplete or from the wrong export** (missing `tokenizer.json`, wrong label set): clear startup error, not a crash on the first request. Test: `test_incomplete_model_dir_fails_at_startup` (Task 6).

---

## File Structure

```
src/leakguard/ner/
├── __init__.py
├── decode.py        # token labels + offsets -> NAME spans               (Task 2)
├── model.py         # NameModel: load dir, windows, run ONNX              (Task 3)
├── candidates.py    # Candidate registry, DEFAULT_CANDIDATE, threshold    (Task 4)
├── export.py        # python -m leakguard.ner.export                      (Task 4)
└── compare.py       # python -m leakguard.ner.compare                     (Task 5)
src/leakguard/detect/engine.py     # detect_rules(), build_detector()      (Task 6)
src/leakguard/config.py, app.py    # ner settings, startup load, threadpool (Task 6)
src/leakguard/evaluation/values.py, templates.py, generate.py              (Task 1)
src/leakguard/evaluation/report.py, __main__.py                            (Task 7)
data/eval/handwritten.jsonl                                                (Task 1)
docs/evaluation/ner-comparison.md  # generated from dev                    (Task 5)
docs/evaluation/report.md          # regenerated                           (Task 7)
Dockerfile, .github/workflows/ci.yml, .gitignore, pyproject.toml           (Tasks 3, 4, 8)
docs/walkthrough/phase-4-names.md                                          (Task 9)
tests/ner/test_decode.py, test_model.py, test_candidates.py, test_real_model.py
```

---

### Task 1: Realistic names in the evaluation data

The current `NAME` data is capitalised "First Last" from one 54-word pool shared by both splits. That's too easy, and test names aren't unseen.

**Files:** Modify `src/leakguard/evaluation/values.py`, `templates.py`, `generate.py`, `data/eval/handwritten.jsonl`. Test: `tests/evaluation/test_values.py`, `test_generate.py`, `test_handwritten.py`.

**Interfaces:**
- `NAME_POOLS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]` — `{"dev": (firsts, lasts), "test": (firsts, lasts)}`, disjoint, ≥50 first and ≥30 last names each, spread across regions/communities (North, South, East, West, North-East, Muslim, Christian, Sikh). The existing `FIRST_NAMES`/`LAST_NAMES` stay for email/UPI local parts.
- `name_makers(split: str) -> dict[str, Maker]` with keys `NAME` (unchanged behaviour: full 80% / first 20%), `NAME.first`, `NAME.lower` (lowercase full or first), `NAME.surname` (used after an `{~HONORIFIC}` filler, so gold excludes "Mr."), `NAME.initial` (`K. Venkatesh`, `R. K. Narayan`), `NAME.three` (`Mohammed Imran Khan`), `NAME.caps` (`RAHUL SHARMA`), `NAME.wordlike` (names that are also English words: Kiran, Sunny, Lucky, Rose, Hira, Pinky…; same list in both splits, it's a hard category).
- `render_template(template, rng, makers=VALUE_MAKERS)`; `build_split` passes `{**VALUE_MAKERS, **name_makers(split)}`.
- New fillers: `HONORIFIC` (Mr./Mrs./Ms./Dr./Shri/Smt.), `FESTIVAL` (Ganesh Chaturthi, Guru Nanak Jayanti, Durga Puja, Raksha Bandhan…), `BRAND` (Paytm, PhonePe, Tata Motors, Maruti Suzuki, Bajaj Finance…), `CITY`, `PRODUCT` (Redmi Note 13, Galaxy M34…).
- Templates are appended **in consecutive pairs** so every form reaches both splits (index parity decides the split): `support` +6 (first/honorific/initial/three/caps/lower), `agent_record` +2, `hinglish` +4 (lowercase names: "mera naam {NAME.lower} hai", "{NAME.lower} bol raha hoon…"), `negative` +6 (festival/brand/city/product/"in May"/"Will the…" sentences, no gold), `hard` +2 (`{NAME.wordlike}`).
- Handwritten: +12 messages bringing `NAME` spans to ≥15 (lowercase Hinglish, single names, honorific, initials, a name next to a phone with no space) and +4 name look-alike negatives.

- [ ] **Step 1: Write failing tests**

```python
# test_values.py
def test_name_pools_are_disjoint_between_splits():
    (dev_f, dev_l), (test_f, test_l) = NAME_POOLS["dev"], NAME_POOLS["test"]
    assert not set(dev_f) & set(test_f) and not set(dev_l) & set(test_l)
    assert min(len(dev_f), len(test_f)) >= 50 and min(len(dev_l), len(test_l)) >= 30

def test_name_forms():
    m = name_makers("dev")
    lower = m["NAME.lower"](random.Random(5)); assert lower == lower.lower()
    caps = m["NAME.caps"](random.Random(5)); assert caps == caps.upper()
    assert re.fullmatch(r"([A-Z]\. )+[A-Z][a-z]+", m["NAME.initial"](random.Random(5)))
    assert len(m["NAME.three"](random.Random(5)).split()) == 3
    assert " " not in m["NAME.surname"](random.Random(5))

# test_generate.py
def test_test_split_names_never_appear_in_dev_pool():
    dev_f, dev_l = NAME_POOLS["dev"]
    for s in build_split("test", seed=2):
        for sp in s.spans:
            if sp.type is T.NAME and "wordlike" not in s.template:
                words = re.findall(r"[A-Za-z]{2,}", sp.text(s.text))
                assert not {w.capitalize() for w in words} & (set(dev_f) | set(dev_l)), s.id

def test_every_name_form_in_both_splits():
    forms = ("first", "lower", "surname", "initial", "three", "caps", "wordlike")
    for split, seed in (("dev", 1), ("test", 2)):
        templates = {s.template for s in build_split(split, seed)}
        for form in forms:
            assert any(f"{{NAME.{form}}}" in t for t in templates), (split, form)

def test_name_negatives_in_both_splits():
    for split, seed in (("dev", 1), ("test", 2)):
        assert any(s.category == "negative" and ("{~FESTIVAL}" in s.template
                   or "{~BRAND}" in s.template) for s in build_split(split, seed))

# test_handwritten.py: extend test_handwritten_set_shape
    assert types[T.NAME] >= 15
```

- [ ] **Step 2: Run** `pytest tests/evaluation -v` → new tests FAIL (`NAME_POOLS`/`name_makers` undefined).
- [ ] **Step 3: Implement** the interfaces above. Write the handwritten messages by hand in marked-text style, then convert them with `parse_marked` + `write_jsonl` appending to the existing file (existing lines unchanged).
- [ ] **Step 4: Run** `pytest -v` → all PASS (existing `test_each_type_well_represented_in_both_splits` still ≥50 per type).
- [ ] **Step 5: Commit** `feat: realistic name forms and split-disjoint name pools in the eval data`

---

### Task 2: Decode token labels into NAME spans

**Files:** Create `src/leakguard/ner/__init__.py`, `src/leakguard/ner/decode.py`. Test: `tests/ner/__init__.py`, `tests/ner/test_decode.py`.

**Interfaces:**
- `person_label_ids(id2label: dict[int, str]) -> tuple[frozenset[int], frozenset[int]]` → `(begin_ids, inside_ids)`. `B-PER`→begin, `I-PER`→inside; a non-BIO `person`/`PER` label counts as both (each token may start or continue). Raises `ValueError` if no person label exists.
- `decode_names(text: str, offsets: Sequence[tuple[int, int]], label_ids: Sequence[int], scores: Sequence[float], begin_ids, inside_ids, threshold: float) -> list[Span]`, where `scores[i]` = the token's probability mass on person labels.
- Rules, in order: (1) group consecutive person tokens; a begin label after a person token *in a different word* starts a new group, inside labels always continue; (2) the group's char range is `[first.start, last.end)`, then **expanded to whole words** (`\w` plus `'` and `.` when followed by a letter, so `R.K.` and `D'Souza` stay whole); (3) strip surrounding whitespace/punctuation; (4) group score = mean token score, dropped if `< threshold`; (5) `Span(start, end, EntityType.NAME, score, detector="ner")`.

- [ ] **Step 1: Write failing tests** — `test_bio_tokens_become_one_span` ("I am Rahul Sharma" → one span `Rahul Sharma`), `test_span_expands_to_whole_word` (`Venk`=B-PER, `##atesh`=O → span covers `Venkatesh`), `test_two_adjacent_names_split_on_begin` ("Rahul Priya" both B-PER → 2 spans), `test_low_score_group_dropped` (score 0.4, threshold 0.5 → `[]`), `test_non_bio_person_label` (`id2label={0:"O",1:"person"}`), `test_no_person_label_raises`, `test_initials_kept_whole` ("R. K. Narayan" labelled on `R`,`K`,`Narayan` → one span incl. dots).
- [ ] **Step 2: Run** `pytest tests/ner/test_decode.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `decode.py`. Pure functions, no numpy needed.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: decode NER token labels into whole-word NAME spans`

---

### Task 3: NameModel runtime (ONNX, windows) + runtime deps

**Files:** Create `src/leakguard/ner/model.py`. Modify `pyproject.toml` (runtime deps `onnxruntime>=1.17`, `tokenizers>=0.15`, `numpy>=1.26`), `src/leakguard/detect/engine.py`, `detect/__init__.py`, callers of `detect_all`. Test: `tests/ner/test_model.py`, `tests/detect/test_engine.py`.

**Interfaces:**
- `detect_all` is renamed `detect_rules(text) -> list[Span]` (same behaviour; update callers and tests).
- `build_detector(name_model: Detector | None = None) -> Detector` in `engine.py` — rules plus names, merged by `merge_spans` (NAME already has the lowest priority, so emails, UPI IDs etc. containing names win). Tasks 5–7 use it.
- Model directory contract (written by Task 4): `model.onnx` (inputs `input_ids`, `attention_mask`, int64 `[batch, seq]`; output logits `[batch, seq, labels]`), `tokenizer.json`, `ner_config.json` = `{"id2label": {...}, "cls_id": int, "sep_id": int, "pad_id": int, "max_length": int, "threshold": float, "candidate": str}`, `manifest.json` (provenance; not read at runtime).
- `class NameModel:`
  - `__init__(self, session, tokenizer, config: dict)` — injectable for tests.
  - `@classmethod load(cls, model_dir: str | os.PathLike, threads: int = 1) -> "NameModel"`: raises `NameModelError` (new, in `model.py`) naming the missing file/invalid key; sets `intra_op_num_threads=threads`, `inter_op_num_threads=1`; runs one warm-up inference.
  - `detect(self, text: str) -> list[Span]` — matches the `Detector` signature.
  - `name: str` (= `config["candidate"]`).
- Windowing: encode with `add_special_tokens=False`; window body length `W = max_length - 2`, stride `W - 64` (64-token overlap); each window wrapped in `cls_id … sep_id`; all windows of one text run as **one batch** (padded with `pad_id`, mask 0). Each token takes its logits from the window where it sits **furthest from an edge**. Softmax → `label_ids = argmax`, `scores = sum of person-label probs`. Then `decode_names`.

- [ ] **Step 1: Write failing tests** using a real `tokenizers` WordLevel tokenizer built in the test (vocab of test words + `[CLS]`,`[SEP]`,`[PAD]`,`[UNK]`, `Whitespace` pre-tokenizer) and a `FakeSession` whose `run(None, feeds)` returns logits marking the ids of `rahul`/`sharma` as `B-PER`/`I-PER`:
  - `test_detects_name_through_onnx_contract` (spans + `detector == "ner"`)
  - `test_name_beyond_first_window_is_found` (`max_length=16`, name at token ~40)
  - `test_name_on_window_boundary_found_once` (name straddling the overlap → exactly one span)
  - `test_all_windows_run_in_one_batch` (`FakeSession.calls == 1` for a 3-window text)
  - `test_odd_inputs_return_no_spans` (`""`, `"   "`, `"🙂🙂"`, `"मेरा नाम"` → `[]`, session not called for empty encodings)
  - `test_load_reports_missing_file` (tmp dir without `tokenizer.json` → `NameModelError` mentioning `tokenizer.json`)
  - `test_build_detector_adds_names_and_rules_win_overlaps` (in `test_engine.py`: fake name detector flags `rahul.sharma` inside `rahul.sharma@gmail.com` and `Rahul` elsewhere → EMAIL kept, separate NAME kept; `build_detector(None)` == `detect_rules`)
- [ ] **Step 2: Run** `pytest tests/ner/test_model.py -v` → FAIL.
- [ ] **Step 3: Implement** `model.py`; `pip install -e ".[dev]"` to pull the new runtime deps.
- [ ] **Step 4: Run** `pytest -v` → all PASS.
- [ ] **Step 5: Commit** `feat: NameModel runs ONNX NER over overlapping windows on CPU`

---

### Task 4: Candidate registry and export command

**Files:** Create `src/leakguard/ner/candidates.py`, `src/leakguard/ner/export.py`. Modify `pyproject.toml` (extra `export = ["torch>=2.2", "transformers>=4.40", "onnx>=1.16", "sentencepiece>=0.2", "protobuf>=4"]`), `.gitignore` (`models/`). Test: `tests/ner/test_candidates.py`.

**Interfaces:**
- `@dataclass(frozen=True) class Candidate: key: str; repo: str; revision: str; note: str; threshold: float = 0.5`
- `CANDIDATES: dict[str, Candidate]` — exactly the table in Global Constraints.
- `DEFAULT_CANDIDATE: str` — set in Task 5 from the dev comparison (start as `"distilbert"`).
- CLI `python -m leakguard.ner.export [--candidate KEY | --all] [--out DIR]`:
  - no args → `DEFAULT_CANDIDATE` into `models/ner/` (int8 only; what the guard loads).
  - `--candidate K` / `--all` → `models/candidates/K/` with both `model.fp32.onnx` and `model.onnx` (int8), for Task 5.
  - Steps: `AutoTokenizer.from_pretrained(repo, revision=..., use_fast=True)` + `AutoModelForTokenClassification`; `torch.onnx.export` (opset 17, legacy TorchScript exporter `dynamo=False`, dynamic axes batch/sequence, inputs `input_ids`/`attention_mask`); `onnxruntime.quantization.quantize_dynamic(weight_type=QuantType.QInt8)`; save `tokenizer.json` via `tokenizer.backend_tokenizer.save`; write `ner_config.json` (ids from the HF tokenizer, `max_length = min(512, model.config.max_position_embeddings - 2)` for XLM-R) and `manifest.json` (`repo`, `revision`, file sha256 + sizes, parity results, export date).
  - **Parity check** (fail the export if broken): on 5 fixed sentences, fp32 ONNX logits vs PyTorch max abs diff `< 1e-3`; int8 vs PyTorch argmax agreement on tokens recorded in the manifest (reported, not enforced).
- [ ] **Step 1: Write failing tests** — `test_revisions_are_pinned_shas` (every revision matches `^[0-9a-f]{40}$`), `test_default_candidate_exists`, `test_thresholds_in_range` (0 < t < 1).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** `pytest tests/ner -v` → PASS.
- [ ] **Step 5: Install and export** (manual, local, not in tests): `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install -e ".[dev,export]"`; `python -m leakguard.ner.export --all`. Expected: one directory per ungated candidate, every manifest showing `fp32_max_abs_diff < 1e-3`. If a candidate fails to export, record why in the comparison notes, don't silently drop it.
- [ ] **Step 6: Commit** `feat: pinned NER candidates and ONNX int8 export with parity check`

---

### Task 5: Compare candidates on dev, tune threshold, pick the model

**Files:** Create `src/leakguard/ner/compare.py`, generated `docs/evaluation/ner-comparison.md`. Modify `candidates.py` (`DEFAULT_CANDIDATE`, winner's `threshold`).

**Interfaces:**
- CLI `python -m leakguard.ner.compare` (dev split only; it has no option for test, by design).
- For each exported candidate in `models/candidates/`, detector = `build_detector(model.detect)` (rules + NER, as deployed) scored with `score()` on dev: NAME P/R/F1, NAME recall **per name form** (grouped by `{NAME.<form>}` in the template; plain `{NAME}` = "mixed"), false-positive NAME count on `negative`, latency p50/p95 of `model.detect` alone (Phase 3's `measure_latency`, 1 thread), `model.onnx` size on disk.
- Threshold sweep for each candidate over `0.30, 0.40, …, 0.90`; table shows F1 at the best threshold (the one used in the comparison row).
- Quantisation table for every candidate: fp32 vs int8 NAME F1, p50 latency and size.
- **Selection rule** (written into the generated doc): highest NAME F1 on dev; when another candidate is within 1.0 F1 point, prefer the one with lower p95 latency.
- [ ] **Step 1: Write failing test** `tests/ner/test_compare.py::test_selection_rule` — `select_winner(rows)` picks the F1 leader, or the faster one within 1 point.
- [ ] **Step 2–4:** implement `select_winner(rows: list[CandidateResult]) -> str` and the CLI; tests PASS.
- [ ] **Step 5: Run** `python -m leakguard.ner.compare` → writes `docs/evaluation/ner-comparison.md`. Set `DEFAULT_CANDIDATE` and its `threshold` from it. Re-export the winner: `python -m leakguard.ner.export`.
- [ ] **Step 6: Decide on the lowercase/Hinglish fallback** with the dev numbers: build Task 5b **only if** the winner's dev NAME recall on the `lower` form is `< 80%`. Record the decision and the number in the comparison doc.
- [ ] **Step 7: Commit** `feat: dev-split NER model comparison and selection`

### Task 5b (conditional): lowercase name fallback

Only if Task 5 Step 6 triggers it.
- `src/leakguard/detect/name_cues.py`: `detect_name_cues(text: str) -> list[Span]` flags a lowercase token in a bundled first-name list **only** when adjacent to a cue (`mera naam … hai`, `naam`, `i am`, `i'm`, `this is`, `name:`, `name is`, `… bol raha/rahi hoon`, `… here`), `detector="name_cues"`, score 0.6.
- Name list: from an openly licensed public source, recorded with its licence in `src/leakguard/data/README.md`, **not** built from `NAME_POOLS`. The comparison doc and report state the overlap between the list and the test pool.
- Tests: cue present → flagged; same name without a cue ("rahul dravid batting") → not flagged; a capitalised name left to the model.
- Commit `feat: cue-based lowercase name fallback`.

---

### Task 6: Wire names into settings and the app

**Files:** Modify `config.py`, `app.py`, `masking.py` (default `detect=detect_rules`), `tests/conftest.py`. Test: `tests/test_ner_app.py`, `tests/test_config_upstream.py`.

**Interfaces:**
- Consumes `build_detector`, `NameModel.load`, `NameModelError` (Task 3).
- `Settings` gains `ner_enabled: bool = True` (env `LEAKGUARD_NER`: `on/off/true/false/1/0`, anything else → `ValueError`), `ner_model_dir: str = "models/ner"` (`LEAKGUARD_NER_MODEL_DIR`), `ner_threads: int = 1` (`LEAKGUARD_NER_THREADS`).
- `create_app(settings, transport=None, policy=None, name_model: Detector | None = None)`: if `name_model` is given it's used; elif `settings.ner_enabled`, `NameModel.load(settings.ner_model_dir, settings.ner_threads)` — `NameModelError` propagates, so **startup fails** with the message `NER model not found or invalid at <dir>: <reason>. Run "python -m leakguard.ner.export" or set LEAKGUARD_NER=off.`; else names are off and one startup warning is logged (`ner_disabled`, no values).
- `_apply_policy(body, policy, ctx, detect)`, called via `await run_in_threadpool(...)` so inference never blocks the event loop.
- `conftest.make_guard` defaults `ner_enabled=False` and accepts `name_model=`.
- [ ] **Step 1: Write failing tests:**
  - `test_name_is_masked_and_restored` (guard with fake name model: request "I am Rahul" → upstream sees `[NAME_1]`, client gets "Rahul" back)
  - `test_name_model_error_fails_closed` (fake model raises → 500 `detection_failed`)
  - `test_incomplete_model_dir_fails_at_startup` (`ner_enabled=True`, empty tmp dir → `NameModelError` from `create_app`, message contains `LEAKGUARD_NER=off`)
  - `test_ner_off_starts_without_model`
  - `test_ner_env_parsing` (`LEAKGUARD_NER=off` → False; `maybe` → `ValueError`)
  - `test_slow_detection_does_not_block_other_requests` (fake detector sleeps 0.5 s; with `httpx.AsyncClient(transport=ASGITransport(app))`, a `/healthz` call issued during a chat request completes in < 0.25 s)
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** `pytest -v` + `ruff check .` → PASS.
- [ ] **Step 5: Real-model smoke tests** `tests/ner/test_real_model.py`, skipped unless `models/ner/model.onnx` exists: "Hi, I'm Priya Nair, call me back" → one NAME covering `Priya Nair`; "Your OTP is 482913" → no NAME; a 3,000-character message completes in < 2 s. (Name look-alikes like "Mumbai Indians" are measured in the eval negatives, not asserted here.)
- [ ] **Step 6: Commit** `feat: load the NER model at startup and detect names in requests`

---

### Task 7: Report with names

**Files:** Modify `src/leakguard/evaluation/report.py`, `__main__.py`, regenerate `docs/evaluation/report.md`. Test: `tests/evaluation/test_report.py`.

**Interfaces:**
- `UNBUILT = {EntityType.ADDRESS: "Phase 5"}`.
- `render_report(results, latencies: dict[str, dict[str, dict[str, float]]], dataset_sizes, hard_cases=None, notes=None, name_forms: dict[str, TypeScores] | None = None, model_info: dict | None = None)` — latency section has one table per layer label (`"rules only"`, `"rules + names"`); a **Names by form (test)** table (Recall, Coverage per form); a model line (candidate, repo@revision short sha, int8 size, threads) from `model_info` (= the model dir's `manifest.json` + `ner_config.json`).
- CLI loads `NameModel` from `LEAKGUARD_NER_MODEL_DIR` or `models/ner`; missing → exits with the export instruction (never reports NAME as unbuilt or zero by accident). `errors` uses the same full detector.
- New `NOTES` entry: dataset extended in Phase 4 (name forms, split-disjoint name pools, name negatives), so non-NAME rows changed slightly because the seeded generator produced new samples; no test-set inspection was used for any Phase 4 decision.
- [ ] **Step 1: Failing tests** — `test_report_has_layer_latency_tables`, `test_report_has_name_forms_table`, `test_name_no_longer_marked_unbuilt`.
- [ ] **Step 2–4:** implement; `pytest -v` PASS.
- [ ] **Step 5: Run** `python -m leakguard.evaluation report`; read the NAME rows and the error examples; **do not change anything based on them** (test is report-only).
- [ ] **Step 6: Commit** `feat: report names, per-layer latency and name forms`

---

### Task 8: Docker and CI with the model

**Files:** Modify `Dockerfile`, `.github/workflows/ci.yml`, `README.md` (run section: export command, `LEAKGUARD_NER`).

- Dockerfile becomes multi-stage: stage `model` (`python:3.11-slim`, CPU torch from `https://download.pytorch.org/whl/cpu`, `pip install ".[export]"`, `python -m leakguard.ner.export --out /models/ner`); final stage unchanged except `COPY --from=model /models/ner /app/models/ner` and `ENV LEAKGUARD_NER_MODEL_DIR=/app/models/ner`. Final image must not contain torch: verify with `docker run --rm <img> python -c "import torch"` → ModuleNotFoundError.
- CI: existing `test` job unchanged (real-model tests skip). New job `ner-model`: cache `~/.cache/huggingface` and `models/ner` keyed on `hashFiles('src/leakguard/ner/candidates.py', 'src/leakguard/ner/export.py')`; install CPU torch + `.[dev,export]`; `python -m leakguard.ner.export`; `pytest -v`.
- [ ] **Step 1:** `docker compose build` then `docker compose up -d`; send "Hi, I'm Priya Nair" through `examples/chat.py` → mock echo shows `[NAME_1]` upstream and the name restored. Record image size (`docker image ls`) for the walkthrough.
- [ ] **Step 2:** torch-absent check above.
- [ ] **Step 3: Commit** `build: export the NER model in a Docker build stage; CI job for real-model tests`

---

### Task 9: Walkthrough, PR, private notes

- [ ] `docs/walkthrough/phase-4-names.md`: what NER is and why names need a model; BIO tags and subword offsets; the windowing; ONNX and int8 quantisation (with the measured fp32-vs-int8 numbers); why PyTorch is only in the build stage; the comparison and selection rule; why IndicNER was left out (gated); Hinglish/lowercase result and the fallback decision; fail-closed startup; event-loop/threadpool; known failures; interview Q&A.
- [ ] Self-review (superpowers:requesting-code-review), push branch `phase-4-names`, open PR, watch CI (both jobs).
- [ ] Private (outside the repo): append Phase 4 to `data-leak-guard-private/build-journal.md` (what/why, glossary: NER, BIO, subword, ONNX, quantisation, gated model, threadpool; "what I did myself"); fill the Names row of `interview-guide.md` with the measured test numbers.
- [ ] The user merges the PR.
