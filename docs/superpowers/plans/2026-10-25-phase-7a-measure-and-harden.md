# Phase 7a — Measure and harden Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare our engine with Microsoft Presidio (two setups) on the same data, measure the guard's speed, running cost and LLM token cost, add the optional guard key and a localhost-only port binding, and write the README threat model.

**Architecture:** Hardening is a small `auth.py` (constant-time key check) wired into `create_app` as a `/v1/` middleware, plus a validated `Settings.api_key`. The Presidio baseline is split into a pure, Presidio-free core (`presidio_baseline.py`: mapping, thresholds, Presidio's default conflict resolution, a `PresidioDetector` adapter, dev tuning) that CI tests with a fake analyser, and a thin runner that imports Presidio lazily. `metrics.score` gains a `cover` hook so coverage can count every Presidio result while typed scores use mapped ones only. Performance is three pieces: client-side measurements (`perf.py`), orchestration of native processes and a CPU-capped container (`perf_run.py`, `docker-compose.perf.yml`), and a renderer (`perf_report.py`). Token cost is `token_cost.py`. All long runs write committed JSON (no message text); reports render from JSON without Presidio, Docker or tiktoken, like `restore-report`.

**Tech Stack:** Python 3.11, FastAPI, httpx (as today). New optional extras only: `[baseline]` (presidio-analyzer, spaCy, `en_core_web_lg`), `[perf]` (tiktoken, psutil). No new runtime dependencies.

**Spec:** [Phase 7 design](../specs/2026-10-25-phase-7-design.md) (wins for Phase 7; this plan covers its 7a part: §3–§7, §9's 7a pieces, §10) and [main design](../specs/2026-10-24-data-leak-guard-design.md) §6.8, §7.

Commands assume the venv at `C:/Users/ayush/.venvs/leakguard` is active. Long runs on this laptop: launch detached (`Start-Process cmd /c "cd . && ..."`; a leading quote breaks cmd) and watch with a Monitor that also checks the runner PID.

## Global Constraints

- No detection change in this phase. A defect found by measurement is written down as a known limit, never fixed against test.
- Test results are looked at only after every setting is fixed; the Presidio test run is made once.
- Presidio "best shot" = default `AnalyzerEngine` + all six Indian recognizers enabled + per-type minimum score tuned on dev. No custom recognizers, patterns, deny lists or models. `language="en"` for every message.
- Mapping (verbatim): `IN_AADHAAR→AADHAAR`, `IN_PAN→PAN`, `PHONE_NUMBER→PHONE`, `EMAIL_ADDRESS→EMAIL`, `CREDIT_CARD→CARD`, `PERSON→NAME`, `LOCATION→ADDRESS` (caveat: place names, not full addresses).
- Not supported: UPI, IFSC, BANK_ACCOUNT, SECRET → precision/recall/F1 cells read "not supported", coverage is shown.
- Unmapped Presidio types: dropped from typed metrics, counted per type; **coverage uses every Presidio result**.
- Conflicts resolved as Presidio's anonymiser does by default (`MERGE_SIMILAR_OR_CONTAINED`), after the thresholds; the number resolved is reported.
- Tuning grid {0.0, 0.05, …, 0.95}; per mapped type maximise dev F1; ties → higher threshold.
- Aggregate "Types both support" = AADHAAR, PAN, PHONE, EMAIL, CARD, NAME, same computation for both engines; ADDRESS shown separately; "All built types" for ours only.
- Every performance number is stored and printed with the hardware: CPU model, logical cores, RAM, OS, Python, Docker server version, Docker's CPUs and memory.
- Overhead and time-to-first-token: native processes over loopback. Throughput, memory, CPU: the real container capped with `--cpus` 1 and 2, client and mock in the same Docker network.
- Token counts: `tiktoken` `o200k_base`.
- Guard key: env `LEAKGUARD_API_KEY`, header `X-Leakguard-Key`, minimum length 16, `hmac.compare_digest` on bytes, 401 before the body is read, never forwarded, never logged, `/healthz` open. Unset or empty = off.
- Compose publishes the guard as `127.0.0.1:8080:8080`.
- Committed JSON results hold scores, counts and timings only, never message text. No real key anywhere. No fake secret committed in a form GitHub push protection flags.
- Python 3.11, ruff line length 100. CI installs neither `[baseline]` nor `[perf]`: code importing Presidio, spaCy, tiktoken or psutil imports them inside functions.

## Review Focus

1. **A guard key header with non-ASCII or odd bytes** (`X-Leakguard-Key: ключ`, trailing space): 401, never a 500. Test: `test_guard_key_non_ascii_header_is_401` (Task 1).
2. **Key on, request to a `/v1/` path the guard doesn't serve** (`/v1/models`): 401 without the key (the key check runs before routing), `/healthz` still 200. Test: `test_guard_key_covers_every_v1_path` (Task 1).
3. **Presidio results that collapse to nothing or point past the text** (zero-length after resolution, `end > len(text)`): dropped, never scored. Test: `test_resolve_drops_empty_and_out_of_range` (Task 3).
4. **Load test with failing requests** (guard returns 500s or times out): errors counted and shown next to the throughput, never silently excluded from it. Test: `test_load_counts_errors` (Task 6).
5. **Overlapping cover ranges and predictions** (a Presidio `URL` over an `EMAIL` it also found): each gold character counted once in coverage. Test: `test_cover_ranges_count_each_char_once` (Task 2).

---

## File Structure

```
src/leakguard/auth.py                          # guard_key_ok()                                   (Task 1)
src/leakguard/config.py, app.py, logs.py       # Settings.api_key, middleware, log_auth_failed       (Task 1)
docker-compose.yml                             # 127.0.0.1 binding                                   (Task 1)
src/leakguard/evaluation/metrics.py            # score(cover=...), evaluation_to_json/from_json      (Task 2)
src/leakguard/evaluation/presidio_baseline.py  # pure core: mapping, Found, thresholds, conflicts,
                                               # PresidioDetector, tune_thresholds                    (Task 3)
src/leakguard/evaluation/presidio_run.py       # make_analyzer, indian_defaults, run_dev, run_test    (Task 4)
src/leakguard/evaluation/presidio_report.py    # render_presidio_tuning, render_presidio_baseline     (Task 4)
src/leakguard/evaluation/token_cost.py         # token_cost()                                       (Task 5)
src/leakguard/evaluation/perf.py               # measure_overhead, load, machine_info, `load` CLI     (Task 6)
src/leakguard/evaluation/perf_run.py           # native processes + capped container orchestration   (Task 7)
src/leakguard/evaluation/perf_report.py        # render_performance, machine_size                     (Task 7)
src/leakguard/evaluation/__main__.py           # presidio-run/-report, perf-run/-report              (Tasks 4, 7)
docker-compose.perf.yml                        # guard (capped) + mock + client, no published ports  (Task 7)
pyproject.toml                                 # [baseline], [perf] extras                            (Tasks 4, 5)
data/eval/presidio-dev.json, presidio-test.json, perf.json                                            (Tasks 8, 9)
docs/evaluation/presidio-tuning.md, presidio-baseline.md, performance.md                              (Tasks 8, 9)
README.md, docs/walkthrough/phase-7a-measure-and-harden.md                                           (Tasks 10, 11)
tests/test_auth.py, tests/test_compose.py, tests/evaluation/test_presidio_baseline.py,
tests/evaluation/test_presidio_report.py, test_token_cost.py, test_perf.py, test_perf_report.py,
tests/evaluation/test_samples_metrics.py (extended)
```

---

### Task 1: Guard key and localhost-only binding

**Files:**
- Create: `src/leakguard/auth.py`, `tests/test_auth.py`, `tests/test_compose.py`
- Modify: `src/leakguard/config.py` (`Settings`), `src/leakguard/app.py` (`create_app`), `src/leakguard/logs.py`, `docker-compose.yml`

**Interfaces:**
- Produces:
  - `config.MIN_API_KEY_LENGTH = 16`; `Settings.api_key: str | None = None`, read from `LEAKGUARD_API_KEY` (empty → `None`); `Settings.__post_init__` raises `ValueError("LEAKGUARD_API_KEY must be at least 16 characters")` when set and shorter.
  - `auth.GUARD_KEY_HEADER = "x-leakguard-key"`; `auth.guard_key_ok(given: str | None, expected: str) -> bool` (`hmac.compare_digest` on UTF-8 bytes; `None` → False).
  - `logs.log_auth_failed(*, request_id: str, path: str) -> None` → `{"event": "auth_failed", "request_id": ..., "path": ...}` (path without query string).
  - When `settings.api_key` is set, `create_app` adds an HTTP middleware: any path starting `/v1/` without a matching key → `openai_error(401, "Missing or wrong X-Leakguard-Key header.", "invalid_request_error", "invalid_guard_key")`, before the route runs.

- [ ] **Step 1: Write the failing tests** (`tests/test_auth.py`, using `make_guard(api_key=KEY)`, `KEY = "k" * 16`)

```python
def test_guard_key_off_by_default(guard): assert guard.post(CHAT, json=_req()).status_code == 200
def test_guard_key_missing_or_wrong_is_401(make_guard):
    g = make_guard(api_key=KEY)
    for headers in ({}, {"X-Leakguard-Key": "k" * 15 + "x"}):
        r = g.post(CHAT, json=_req(), headers=headers)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_guard_key"
def test_guard_key_right_passes(make_guard): ...  # 200 with {"X-Leakguard-Key": KEY}
def test_guard_key_checked_before_body(make_guard):
    # wrong key + oversized body -> 401, not 413; wrong key + invalid JSON -> 401, not 400
def test_guard_key_non_ascii_header_is_401(make_guard):  # Review Focus 1
    # raw header bytes "ключ".encode() and KEY + " " -> 401
def test_guard_key_covers_every_v1_path(make_guard):     # Review Focus 2
    # GET /v1/models without key -> 401; GET /healthz without key -> 200
def test_guard_key_not_forwarded(make_guard):
    # MockTransport records headers; "x-leakguard-key" not in them; authorization still is
def test_guard_key_never_logged(make_guard, caplog):
    # after one right and one wrong request, KEY and the wrong key appear in no log record;
    # an "auth_failed" record exists
def test_short_key_refuses_start(): pytest.raises(ValueError, Settings, api_key="short")
def test_api_key_from_env(): Settings.from_env({"LEAKGUARD_API_KEY": KEY}).api_key == KEY;
    Settings.from_env({"LEAKGUARD_API_KEY": ""}).api_key is None
```

`tests/test_compose.py`:
```python
def test_guard_published_on_localhost_only():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    assert all(str(p).startswith("127.0.0.1:") for p in compose["services"]["guard"]["ports"])
```

- [ ] **Step 2: Run them to see them fail** — `pytest tests/test_auth.py tests/test_compose.py -v` → FAIL (`api_key` unknown, compose binds all interfaces).
- [ ] **Step 3: Implement** `auth.py`, the `Settings` field and validation, `log_auth_failed`, the middleware in `create_app` (registered only when a key is set), and `"127.0.0.1:8080:8080"` with a comment saying how to expose it deliberately (and to set the key).
- [ ] **Step 4: Run** `pytest -q && ruff check .` → all pass.
- [ ] **Step 5: Commit** `feat: optional guard API key (constant-time, 401 before the body) and localhost-only compose binding`

### Task 2: Coverage hook and serialisable scores

**Files:**
- Modify: `src/leakguard/evaluation/metrics.py`
- Test: `tests/evaluation/test_samples_metrics.py`

**Interfaces:**
- Produces:
  - `score(samples, detect, cover: Callable[[str], Iterable[tuple[int, int]]] | None = None) -> Evaluation`: when `cover` is given, coverage counts a gold character as covered if any prediction **or** any cover range contains it; typed TP/FP/FN unchanged. Default behaviour unchanged (existing tests keep passing).
  - `evaluation_to_json(ev: Evaluation) -> dict` = `{"by_type": {type_value: asdict(TypeScores)}, "negatives_total": int, "negatives_with_fp": int}`; `evaluation_from_json(data: dict) -> Evaluation` (inverse; `errors`, `by_category*` left empty).

- [ ] **Step 1: Failing tests**
  - `test_cover_extends_coverage_not_typed_scores`: gold EMAIL `a@b.in`, `detect` → `[]`, `cover` → `[(start, end)]` of it: EMAIL coverage 1.0, recall 0.0, fp 0.
  - `test_cover_ranges_count_each_char_once` (Review Focus 5): a prediction and two cover ranges all over the same gold span → `covered_chars == gold_chars`.
  - `test_evaluation_json_round_trip`: `evaluation_from_json(evaluation_to_json(ev)).by_type == ev.by_type`, and the dict survives `json.dumps`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** (coverage: merge the prediction and cover ranges, then count gold characters inside the union).
- [ ] **Step 4: Run** `pytest tests/evaluation -q` → pass.
- [ ] **Step 5: Commit** `feat: coverage hook for results outside the typed metrics; serialisable scores`

### Task 3: Presidio baseline core (no Presidio needed)

**Files:**
- Create: `src/leakguard/evaluation/presidio_baseline.py`, `tests/evaluation/test_presidio_baseline.py`

**Interfaces:**
- Consumes: `score(..., cover=...)`, `TypeScores` (Task 2).
- Produces:
  - `MAPPING: dict[str, EntityType]` (the Global Constraints mapping); `UNSUPPORTED = (UPI, IFSC, BANK_ACCOUNT, SECRET)`; `BOTH_SUPPORT = (AADHAAR, PAN, PHONE, EMAIL, CARD, NAME)`; `GRID = tuple(i / 20 for i in range(20))`.
  - `@dataclass(frozen=True) class Found: entity: str; start: int; end: int; score: float` (a Presidio result without Presidio's class).
  - `Analyze = Callable[[str], list[Found]]`.
  - `apply_thresholds(found: Iterable[Found], thresholds: Mapping[str, float]) -> list[Found]` (keeps results whose type has no threshold).
  - `resolve_conflicts(found: Iterable[Found], text: str) -> tuple[list[Found], int]` → (kept, number of results removed or merged).
  - `class PresidioDetector(analyze: Analyze, thresholds: Mapping[str, float] = {})`: `detect(text) -> list[Span]` (mapped, after thresholds then conflicts; `detector="presidio"`), `cover(text) -> list[tuple[int, int]]` (all kept results), `unmapped: Counter[str]`, `conflicts: int`. Calls `analyze` once per distinct text (memoised), so `detect` and `cover` share one analysis.
  - `@dataclass(frozen=True) class TuningRow: presidio_type: str; threshold: float; tp: int; fp: int; fn: int; f1: float`.
  - `tune_thresholds(dev: Sequence[Sample], analyze: Analyze) -> tuple[dict[str, float], list[TuningRow]]`: for each Presidio type in `MAPPING`, sweep `GRID` with every other type at 0.0, keep the best F1 (tie → higher threshold).

`resolve_conflicts` mirrors the anonymiser's default strategy in the pinned `presidio-anonymizer` source (`AnonymizerEngine`, `MERGE_SIMILAR_OR_CONTAINED`); the builder checks the order against that source in Task 4 and adjusts if it differs:
1. merge overlapping results of the same type into one span, keeping the max score;
2. remove any result contained in another (any type); identical spans keep the higher score, a tie keeps the later one;
3. merge same-type results separated only by spaces;
then drop zero-length results and results outside `0 <= start < end <= len(text)`.

- [ ] **Step 1: Failing tests** (all with hand-built `Found` lists and a fake `analyze`)
  - `test_mapping_matches_spec`: the seven pairs exactly; `set(UNSUPPORTED) | set(MAPPING.values())` covers every `EntityType`.
  - `test_thresholds_drop_below_and_keep_untuned`.
  - `test_resolve_same_type_overlap_merges`: `PERSON 0-5 (0.6)` + `PERSON 3-9 (0.8)` → one `PERSON 0-9 (0.8)`, count 1.
  - `test_resolve_contained_any_type_removed`: `PHONE_NUMBER 10-20` inside `CREDIT_CARD 5-25` → only the card.
  - `test_resolve_identical_spans_keep_higher_score`: `IN_AADHAAR 0-14 (0.5)` vs `PHONE_NUMBER 0-14 (0.4)` → Aadhaar.
  - `test_resolve_space_separated_same_type_merges`: `PERSON "Rahul"` + `PERSON "Sharma"` with one space → one span.
  - `test_resolve_drops_empty_and_out_of_range` (Review Focus 3).
  - `test_detector_maps_and_counts_unmapped`: `URL` and `DATE_TIME` results are not in `detect`, are in `cover`, and counted in `unmapped`.
  - `test_detector_analyses_each_text_once`.
  - `test_tuning_picks_best_f1_tie_to_higher`: a fake analyser whose PERSON scores make thresholds 0.3 and 0.35 tie on F1 → 0.35.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests/evaluation/test_presidio_baseline.py -v` → pass.
- [ ] **Step 5: Commit** `feat: Presidio baseline core: mapping, thresholds, anonymiser-style conflicts, dev tuning`

### Task 4: Presidio runner, report pages and CLI

**Files:**
- Create: `src/leakguard/evaluation/presidio_run.py`, `src/leakguard/evaluation/presidio_report.py`, `tests/evaluation/test_presidio_report.py`
- Modify: `src/leakguard/evaluation/__main__.py`, `pyproject.toml`

**Interfaces:**
- Consumes: Task 3's names; `load_split`, `load_detector`, `measure_latency`, `evaluation_to_json`, `restore_run._git_commit`.
- Produces:
  - `presidio_run.SETUPS = ("out_of_box", "best_shot")`; `make_analyzer(setup: str) -> Analyze` (lazy imports; `best_shot` adds the six `In*` recognizers to the registry; both call `analyze(text, language="en")` and convert to `Found`).
  - `indian_defaults() -> dict[str, bool]`: for each of `IN_AADHAAR, IN_PAN, IN_PASSPORT, IN_VEHICLE_REGISTRATION, IN_VOTER, IN_GSTIN`, whether a default `AnalyzerEngine()` supports it.
  - `versions() -> dict[str, str]`: presidio-analyzer, spacy, en_core_web_lg.
  - `run_dev() -> dict` → `{"meta": {...}, "thresholds": {...}, "tuning": [asdict(row)...]}`.
  - `run_test(thresholds: Mapping[str, float]) -> dict` → `{"meta": {...}, "thresholds": ..., "engines": {"ours" | "presidio_out_of_box" | "presidio_best_shot": {"test": ev_json, "handwritten": ev_json, "latency": measure_latency(...)["all"], "unmapped": {...}, "conflicts": int}}}` (`unmapped`/`conflicts` for Presidio engines only). Latency uses fresh, un-memoised detectors.
  - `meta` = date, git commit, `versions()`, `indian_defaults()`, thread settings, split sizes.
  - `presidio_report.render_presidio_tuning(dev: dict) -> str` and `render_presidio_baseline(test: dict, notes: Sequence[str]) -> str`.
  - CLI: `presidio-run --split dev` → `data/eval/presidio-dev.json`; `presidio-run --split test` (reads thresholds from `presidio-dev.json`, refuses if missing) → `data/eval/presidio-test.json`; `presidio-report` → `docs/evaluation/presidio-tuning.md` and `presidio-baseline.md`, without Presidio installed.
- `pyproject.toml`: `baseline = ["presidio-analyzer==<latest stable on PyPI>", "spacy==<the version presidio resolves>", "en_core_web_lg @ <exact spacy-models release wheel URL for that spaCy minor>"]`. Before importing spaCy or ONNX Runtime, the CLI sets `OMP_NUM_THREADS=1` (and MKL/OpenBLAS equivalents) with `os.environ.setdefault`; recorded in `meta`.

`render_presidio_baseline` must show:
- a headline table `Type | Ours F1 | Presidio out of the box F1 | Presidio best shot F1 | Ours coverage | … coverage | … coverage` over test, unsupported types with F1 "not supported" and coverage shown, ADDRESS marked with the place-names caveat, then the "Types both support" row for all three engines;
- full P/R/F1/coverage tables per engine for test and hand-written, negatives with false alarms;
- latency p50/p95 per engine; unmapped counts; conflicts resolved; versions; the `indian_defaults()` finding in a sentence; the notes.

- [ ] **Step 1: Failing tests** (fixture dicts shaped like `run_dev`/`run_test` output, no Presidio)
  - `test_baseline_page_marks_unsupported_but_shows_coverage`: the BANK_ACCOUNT row reads "not supported" in Presidio F1 cells and has a percentage in Presidio coverage cells.
  - `test_baseline_page_both_support_row_uses_six_types`: hand-computed micro F1 over the six types for one engine appears.
  - `test_baseline_page_reports_indian_defaults`: `{"IN_AADHAAR": False, ...}` → sentence contains "off by default".
  - `test_tuning_page_lists_chosen_thresholds`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** renderers, runner and CLI; add the extra. **Step 4: Run** `pytest -q && ruff check .` → pass.
- [ ] **Step 5: Install and smoke-run (not committed output):** `pip install -e ".[baseline]"`; check `resolve_conflicts`' order against the installed `presidio_anonymizer/anonymizer_engine.py` (adjust + note if different); `python -c` one Aadhaar message through `make_analyzer("best_shot")` → an `IN_AADHAAR` result.
- [ ] **Step 6: Commit** `feat: Presidio runner (two setups), tuning and baseline pages, [baseline] extra`

### Task 5: Token cost

**Files:**
- Create: `src/leakguard/evaluation/token_cost.py`, `tests/evaluation/test_token_cost.py`
- Modify: `pyproject.toml` (`perf = ["tiktoken==<latest>", "psutil==<latest>"]`)

**Interfaces:**
- Produces: `token_cost(samples: Iterable[Sample], detect: Detector, policy: Policy, count: Callable[[str], int]) -> dict` → `{"messages": int, "blocked": int, "unchanged": int, "tokens_before": int, "tokens_after": int, "mean_change": float, "by_type": {type: {"values": int, "value_tokens": float, "placeholder_tokens": float}}}`. Each sample is masked as one user message with `mask_messages`; blocked ones are excluded and counted. `by_type` compares each masked value with its placeholder tokenised on their own (stated in the report as an isolated comparison). `o200k_counter() -> Callable[[str], int]` imports tiktoken lazily.

- [ ] **Step 1: Failing tests** with `count=len` (characters as "tokens"):
  - `test_token_cost_counts_before_after`: one PHONE message → `tokens_after - tokens_before == len("[PHONE_1]") - len(phone)`.
  - `test_token_cost_excludes_blocked`: a secret message → `blocked == 1`, not in totals.
  - `test_token_cost_unchanged_message`: nothing detected → `unchanged == 1`, change 0.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → pass.
- [ ] **Step 5: Commit** `feat: LLM token cost of masking (before/after, per type)`

### Task 6: Performance measurements (client side)

**Files:**
- Create: `src/leakguard/evaluation/perf.py`, `tests/evaluation/test_perf.py`

**Interfaces:**
- Produces:
  - `forwardable_texts(samples: Iterable[Sample]) -> list[str]` (drops samples with a SECRET gold span: they are blocked, not forwarded).
  - `measure_overhead(direct: httpx.Client, guard: httpx.Client, texts: Sequence[str], stream: bool, repeats: int = 3) -> dict` → `{"n", "p50_ms", "p95_ms"}` of per-message (best-of-`repeats` via guard − best-of-`repeats` direct), plus `"ttft": {"n", "p50_ms", "p95_ms"}` when `stream` (time to first `delta.content` chunk, same pairing). Clients carry their own `base_url`.
  - `async load(client: httpx.AsyncClient, texts: Sequence[str], concurrency: int, seconds: float) -> dict` → `{"concurrency", "seconds", "requests", "errors", "msgs_per_s", "p50_ms", "p95_ms"}`; non-streaming; any non-200 or exception is an error; `msgs_per_s` counts successful requests only.
  - `machine_info() -> dict`: cpu model (Windows `winreg` `ProcessorNameString`, else `/proc/cpuinfo`), logical cores, RAM GB (psutil, lazy), `platform.platform()`, Python version, and `docker version`/`docker info` server version, NCPU, MemTotal (None if Docker is absent).
  - `python -m leakguard.evaluation.perf load --url URL --concurrency N --seconds S` prints `load(...)` as JSON (texts = `forwardable_texts(load_split("test"))`); used inside the client container.

- [ ] **Step 1: Failing tests** (in-process: `httpx.Client(transport=httpx.ASGITransport(...))` is async-only, so use the `make_guard` TestClient and the mock's TestClient as the two sync clients; `load` uses `httpx.AsyncClient(transport=ASGITransport(app=...))`)
  - `test_forwardable_texts_drops_secrets`.
  - `test_measure_overhead_shapes_and_ttft`: stream=True returns `ttft` with `n == len(texts)`.
  - `test_load_counts_requests` (concurrency 2, 0.3 s → `requests > 0`, `errors == 0`).
  - `test_load_counts_errors` (Review Focus 4): an app answering 500 → `errors == requests`, `msgs_per_s == 0`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → pass.
- [ ] **Step 5: Commit** `feat: client-side performance measurements and machine info`

### Task 7: Performance runner, compose file and report

**Files:**
- Create: `src/leakguard/evaluation/perf_run.py`, `src/leakguard/evaluation/perf_report.py`, `docker-compose.perf.yml`, `tests/evaluation/test_perf_report.py`
- Modify: `src/leakguard/evaluation/__main__.py`

**Interfaces:**
- Consumes: Task 5 `token_cost`, `o200k_counter`; Task 6 everything.
- Produces:
  - `docker-compose.perf.yml`, project name `leakguard-perf`, **no published ports**: `mock` (as in `docker-compose.yml`), `guard` (same image, `LEAKGUARD_UPSTREAM_BASE_URL=http://mock:9000/v1`, `cpus: ${PERF_CPUS:-1}`), `client` (same image, `profiles: ["client"]`, runs `python -m leakguard.evaluation.perf load ...`).
  - `perf_run.native_overhead(ner: bool) -> dict`: starts mock and guard as uvicorn subprocesses on free loopback ports (guard env: upstream = the mock, `LEAKGUARD_NER=on|off`, `LEAKGUARD_NER_MODEL_DIR=models/ner`; output to DEVNULL), waits for `/healthz`, runs `measure_overhead` non-streaming and streaming over `forwardable_texts(load_split("test"))`, always stops both.
  - `perf_run.container_runs(cpus: Sequence[int] = (1, 2), concurrency: Sequence[int] = (1, 8), seconds: float = 30) -> dict`: builds and starts the perf stack per CPU cap; records idle `memory.current` (10 s after healthy) from `/sys/fs/cgroup` via `docker exec`; runs the client per concurrency; reads `cpu.stat usage_usec` before/after and `memory.peak` (falls back to polling `docker stats` if absent, noted in the result); image size via `docker image inspect`; model size via `du -sb /app/models/ner`; `docker compose down` always.
  - `perf_run.run() -> dict` → `{"meta": {date, git commit, machine_info()}, "overhead": {"ner_on": ..., "ner_off": ...}, "containers": {...}, "tokens": token_cost(load_split("test"), full detector, default policy, o200k_counter())}`.
  - `perf_report.machine_size(peak_mb: float, cpus: int) -> str`: smallest of 0.5/1/2/4 GB that is ≥ 1.25 × peak → `"1 vCPU / 1 GB"`.
  - `perf_report.render_performance(results: dict) -> str`: hardware block first; overhead (p50/p95, non-streaming and streaming, NER on/off) and time to first token; throughput per CPU cap × concurrency with p95 and errors; memory idle/peak, image and model size, CPU ms per message (= Δusage_usec / 1000 / successful requests); the machine-size line per cap; token cost table; notes (laptop not server; Docker Desktop on Windows runs a VM; loopback vs container networking).
  - CLI: `perf-run` → `data/eval/perf.json` (needs Docker, the model, `[perf]`); `perf-report` → `docs/evaluation/performance.md` from the JSON alone.

- [ ] **Step 1: Failing tests** (`test_perf_report.py`, fixture JSON)
  - `test_machine_size_rule`: `(380, 1) → "1 vCPU / 0.5 GB"`; `(420, 1) → "1 vCPU / 1 GB"`; `(900, 2) → "2 vCPU / 2 GB"`.
  - `test_performance_page_shows_hardware_first`: the CPU model string appears before the first table.
  - `test_performance_page_shows_errors_next_to_throughput`: a run with `errors: 3` shows `3` in its row.
  - `test_cpu_ms_per_message`: Δusage 2,000,000 µs over 1,000 requests → `2.0 ms`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `pytest -q && ruff check .` → pass.
- [ ] **Step 5: Smoke-run** `native_overhead(ner=False)` on 20 texts and one `container_runs(cpus=(1,), concurrency=(1,), seconds=5)`; check the stack is gone afterwards (`docker compose -p leakguard-perf ps` empty) and the author's own stack on port 8080 is untouched.
- [ ] **Step 6: Commit** `feat: performance runner (loopback overhead, CPU-capped container) and report`

### Task 8: Presidio runs (dev, then test once)

- [ ] **Step 1:** `python -m leakguard.evaluation presidio-run --split dev` (detached; minutes) → `data/eval/presidio-dev.json`; `presidio-report` → `docs/evaluation/presidio-tuning.md`. Read only the dev page; thresholds are now fixed.
- [ ] **Step 2: Commit** `docs: Presidio dev tuning`.
- [ ] **Step 3:** `presidio-run --split test`, once → `presidio-test.json`; `presidio-report` → `presidio-baseline.md`. Write the notes (one line each): what `indian_defaults()` found in the pinned version; LOCATION caveat; unmapped types dropped but counted in coverage; conflicts resolved as the anonymiser does; same matching rule and data as our report; Hinglish run as `en`; dev-only tuning; the test run was made once after the thresholds were fixed. Re-render.
- [ ] **Step 4: Verify** no message text in either JSON (`grep` a few known test values → no hits). **Commit** `docs: Presidio baseline on test and hand-written sets`.

### Task 9: Performance run

- [ ] **Step 1:** Docker Desktop running; `python -m leakguard.evaluation perf-run` (detached; ~10 min) → `data/eval/perf.json`; `perf-report` → `docs/evaluation/performance.md`.
- [ ] **Step 2: Verify** every number on the page has the hardware block above it, error counts are 0 or explained, and the perf stack is down. **Commit** `docs: performance, running cost and token cost`.

### Task 10: README (threat model, settings, results)

**Files:** Modify `README.md`.

- [ ] **Step 1:** Status → "Phase 7a of 7" (7b finishes v1).
- [ ] **Step 2:** Results gain a **"Compared with Presidio"** subsection: the headline F1 + coverage table from `presidio-baseline.md` (copied from the committed page, with its link), one sentence on the two setups and on the `indian_defaults()` finding.
- [ ] **Step 3:** **"Speed and cost"** subsection from `performance.md`: overhead p50/p95, time to first token, throughput, memory, machine size, token change, the hardware line, link.
- [ ] **Step 4:** **"Security of the guard itself"** section: the threat table (threat · who · what the guard does · proof) for: stolen mappings; content or keys in logs; placeholder injection; ReDoS; open relay (localhost binding + guard key); detection failure (fail-closed); oversized requests (413); the guard key itself. Each proof is a test name, file or number that exists. Then **"What it does not protect against"**: images and files; unmeasured non-Latin scripts; identity inferred from context; the report's known-hard formats (with numbers); traffic that bypasses the guard; an upstream that sees placeholders and context.
- [ ] **Step 5:** Settings table: `LEAKGUARD_API_KEY` row; a short "Exposing the guard beyond localhost" note with `default_headers={"X-Leakguard-Key": ...}`.
- [ ] **Step 6:** Docs list: add the missing Phase 6 walkthrough, the Phase 7a walkthrough, and the three new report pages.
- [ ] **Step 7: Verify** every number in the README matches a committed page (check each by eye against the source table). **Commit** `docs: README threat model, Presidio comparison, speed and cost`.

### Task 11: Walkthrough, review, PR, private docs

- [ ] **Step 1:** `docs/walkthrough/phase-7a-measure-and-harden.md`: what and why, each new file, design decisions (two Presidio setups, conflict resolution, coverage hook, loopback vs capped container, machine-size rule, token cost, constant-time compare, 401 before the body, localhost binding), likely interview questions with answers using the measured numbers. Commit.
- [ ] **Step 2:** `pytest -q && ruff check .` → pass. Final review of the whole branch by a fresh reviewer (most capable model); fix findings; re-run.
- [ ] **Step 3:** Push, open the PR (body: summary, numbers, deferred minors, attribution line), watch CI to green.
- [ ] **Step 4:** Private, outside the repo: `data-leak-guard-private/interview-guide.md` (fill TBDs from committed pages; correct the Presidio line: Indian recognizers exist but are **off by default**, as found in the pinned version) and `build-journal.md` (Phase 7a entry: what/why, what the author did and why). The author merges.
