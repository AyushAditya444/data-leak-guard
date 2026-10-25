# Phase 5 — Indian address detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect full Indian postal addresses as one `ADDRESS` span (detail + place, close together), measured per address style on rebuilt evaluation data, and wired into the guard and the report.

**Architecture:** A bundled gazetteer built once from India Post's All India Pincode Directory (`detect/gazetteer.py`, compact gzipped file in `src/leakguard/data/`). `detect/address_signals.py` finds the pieces (PIN, unit, building, street, landmark, place); `detect/address.py` chains nearby pieces into groups and emits `ADDRESS` spans for groups with a detail and a place. The engine runs the address detector *after* the other rules so an address never swallows a phone or email. Optionally the name model's LOC tags (same inference call) become extra place signals — kept only if dev says so.

**Tech Stack:** standard library only at runtime (`re`, `gzip`, `importlib.resources`); existing `onnxruntime` model for LOC.

**Spec:** [Phase 5 design](../specs/2026-10-25-phase-5-addresses-design.md) (wins for Phase 5) and [main design](../specs/2026-10-24-data-leak-guard-design.md) §5 Addresses, §7, §9 phase 5.

## Global Constraints

- Address = **at least one detail and at least one place**, close together (design §2). Not masked: a place alone, places only ("Kothrud, Pune"), cue + place ("near Kothrud"), landmark with no place, invalid PIN, valid PIN with no context.
- PIN: `[1-9]\d{5}` with word boundaries, present in the bundled directory, with context (next to another address piece, or after `pin`, `pincode`, `pin code`, `postal code`). Preceded within 3 words by `otp`, `code`, `order`, `ticket`, `invoice`, `ref`, `txn`, `id`, `rs`, `₹` → never a PIN.
- Span: first piece to last piece of the group; edge punctuation trimmed; whole words; may cross line breaks; surrounding words excluded.
- Every knob (gap **G**, LOC on/off, stop list) is chosen on **dev**; test and handwritten are report-only. Never tune on test.
- Raw directory CSV is **never committed**; the compact derived file is, with source URL, download date and licence recorded in `src/leakguard/data/README.md`.
- The address detector works with NER **off**; no torch at runtime; no new runtime dependencies.
- Every reported number comes from a command that regenerates it. No invented numbers, no targets set in advance.
- Raw text and detected values are never logged (unchanged).

## Review Focus

1. **Address next to a higher-priority value with no clean separator** ("Pune 411038 9876543210", "Baner,Pune-411045,ph:9820031476"): the phone must stay `PHONE`, the address must still be masked, and nothing may be dropped by merging. Test: `test_address_stops_at_phone` (Task 4).
2. **Pasted multi-line records with Windows line endings** (`"Flat 302,\r\nSai Krupa Apts,\r\nKothrud, Pune 411038"`): one span, not three. Test: `test_crlf_multiline_is_one_span` (Task 4).
3. **Common words that are also post office names** ("I'll meet you at the station near the market", "Camp", "Ram Nagar" vs a person called Ram): no ADDRESS. Test: `test_common_words_are_not_places` (Task 1) and `test_small_talk_not_masked` (Task 4).
4. **Many numbers in one message** (a pasted table of 20,000 six-digit codes or order IDs): stays linear, completes in < 1 s, no ADDRESS without context. Test: `test_many_six_digit_numbers_is_fast` (Task 4).
5. **Odd input** (empty, whitespace, emoji, Devanagari address, full-width digits): returns a list without crashing. Test: `test_odd_inputs_return_lists` (Task 4).

---

## File Structure

```
src/leakguard/detect/gazetteer.py        # load compact data; find_places(); build CLI   (Task 1)
src/leakguard/data/pin_directory.tsv.gz  # derived: pin, office, district, state          (Task 1)
src/leakguard/data/place_stopwords.txt   # office names that are common words             (Task 1)
src/leakguard/data/README.md             # sources + licences (UPI handles + directory)    (Task 1)
src/leakguard/evaluation/values.py, templates.py, generate.py                             (Task 2)
data/eval/handwritten.jsonl                                                               (Task 2)
src/leakguard/detect/address_signals.py  # Signal + one finder per piece kind            (Task 3)
src/leakguard/detect/address.py          # assembly -> ADDRESS spans                      (Task 4)
src/leakguard/detect/engine.py           # address after other rules; ner= wiring         (Tasks 4, 5)
src/leakguard/ner/decode.py, model.py    # LOC ranges from the same inference             (Task 5)
src/leakguard/evaluation/address_tuning.py, docs/evaluation/address-tuning.md            (Task 6)
src/leakguard/app.py                     # wire LOC if kept                               (Task 6)
src/leakguard/evaluation/report.py, __main__.py, docs/evaluation/report.md               (Task 7)
pyproject.toml (package-data), README.md, Dockerfile check                               (Task 8)
docs/walkthrough/phase-5-addresses.md                                                     (Task 9)
tests/detect/test_gazetteer.py, test_address_signals.py, test_address.py, test_engine.py
tests/evaluation/test_values.py, test_generate.py, test_handwritten.py, test_report.py
tests/ner/test_decode.py, test_model.py
```

---

### Task 1: PIN directory → bundled gazetteer

**Files:** Create `src/leakguard/detect/gazetteer.py`, `src/leakguard/data/pin_directory.tsv.gz`, `place_stopwords.txt`, `README.md`. Modify `pyproject.toml` (`leakguard = ["data/*.txt", "data/*.gz"]`), `.gitignore` (`data/raw/`). Test: `tests/detect/test_gazetteer.py`.

**Interfaces:**
- `PlaceKind = Literal["locality", "city", "district", "state"]`.
- `@dataclass(frozen=True) class PostOffice: pin: str; office: str; district: str; state: str`.
- `@dataclass(frozen=True) class Gazetteer: pins: frozenset[str]; places: dict[str, PlaceKind]; offices: tuple[PostOffice, ...]; max_words: int` — `places` keys are lowercase, single-spaced; strongest kind wins when a name is several (state > city > district > locality).
- `Gazetteer.find_places(text: str) -> list[tuple[int, int, PlaceKind]]` — whole-word, case-insensitive, longest match first, non-overlapping; tokenises once, checks n-grams up to `max_words`.
- `load_gazetteer() -> Gazetteer` (cached with `functools.cache`), reading `data/pin_directory.tsv.gz` and `data/place_stopwords.txt` via `importlib.resources.files("leakguard")`.
- `normalise_office(name: str) -> str` strips ` S.O`, ` B.O`, ` H.O`, ` G.P.O`, ` SO`, ` BO`, ` HO`, parenthesised qualifiers ("Kothrud (Pune)" → "Kothrud"), title-cases.
- CLI `python -m leakguard.detect.gazetteer build --csv data/raw/<file>.csv` writes the `.tsv.gz` (deduplicated, sorted by pin), prints row/PIN/place counts.
- A small hand-curated alias list inside `gazetteer.py`: `CITY_ALIASES` — major cities whose common name is not a district name or is an old name (Bengaluru/Bangalore, Mumbai/Bombay, Gurugram/Gurgaon, Noida, Greater Noida, Navi Mumbai, Thane, Kolkata/Calcutta, Chennai/Madras, Kochi/Cochin, Mysuru/Mysore, Puducherry/Pondicherry, Prayagraj/Allahabad, Vadodara/Baroda, Thiruvananthapuram/Trivandrum, Varanasi/Banaras, Delhi/New Delhi), kind `city`.

- [ ] **Step 1: Get the raw CSV.** Try data.gov.in "All India Pincode Directory" (needs a free API key or the download form). If no unauthenticated download works, **ask the user** to download the CSV (or create a free data.gov.in API key) and place it under `data/raw/`; record in the build journal why. Record source URL, resource id, download date and the licence text seen on the page in `src/leakguard/data/README.md`; move the existing UPI handles source note there too if it lives elsewhere.
- [ ] **Step 2: Write failing tests** in `tests/detect/test_gazetteer.py`:
  - `test_normalise_office` — `"Kothrud S.O" → "Kothrud"`, `"Baner B.O" → "Baner"`, `"Pune H.O" → "Pune"`, `"Andheri (West) S.O" → "Andheri"` .
  - `test_known_pins_and_places` — `"411038" in g.pins`, `"110092" in g.pins`, `"999999" not in g.pins`; `find_places("Flat 2, Kothrud, Pune")` has `Kothrud` (locality) and `Pune` with kind `city` or `district`.
  - `test_longest_match_first` — "Navi Mumbai" is one match, not "Mumbai".
  - `test_whole_words_only` — "Punekar" and "pune123" match nothing.
  - `test_common_words_are_not_places` — `find_places("meet me at the station near the market camp")` is `[]`.
  - `test_bundle_is_small` — the `.tsv.gz` file is < 5 MB.
- [ ] **Step 3: Run** `pytest tests/detect/test_gazetteer.py -v` → FAIL (module missing).
- [ ] **Step 4: Implement**, run the build CLI on the raw CSV, seed `place_stopwords.txt` with office names that are English/Hinglish words (station, market, camp, bazar, fort, colony, college, university, hospital, court, mandi, chowk, road, nagar, gate, bridge, mill, factory, airport, cantonment, ...) — dev will extend it in Task 6.
- [ ] **Step 5: Run** tests → PASS; `ruff check .` clean.
- [ ] **Step 6: Commit** `feat: bundle India Post PIN directory as a gazetteer`

### Task 2: Address evaluation data (before any detector rules)

**Files:** Modify `src/leakguard/evaluation/values.py`, `templates.py`, `generate.py`, `data/eval/handwritten.jsonl`. Test: `tests/evaluation/test_values.py`, `test_generate.py`, `test_handwritten.py`.

**Interfaces:**
- Consumes `load_gazetteer().offices` (Task 1).
- `place_pools(split: str) -> list[PostOffice]` — split by district: `"dev"` if `crc32(district) % 2 == 0` else `"test"`, so the splits share no district. Plus `INVENTED_LOCALITIES: dict[str, tuple[str, ...]]` (dev/test disjoint, ≥20 each, not in the gazetteer: e.g. "Hiranandani Meadows", "Sector 62", "Prestige Shantiniketan", "DLF Phase 3", layouts and new societies); ~25% of addresses use one.
- `address_makers(split: str) -> dict[str, Maker]` with keys and gold boundaries per design §2:
  - `ADDRESS` (random style, as today), `ADDRESS.formal_pin` ("Flat 302, Sai Krupa Apts, Kothrud, Pune - 411038"), `ADDRESS.formal` (no PIN), `ADDRESS.landmark` ("near Hanuman Mandir, Kothrud, Pune"; also "opp. SBI ATM, Baner"), `ADDRESS.hinglish` ("H.No. 45, gali no 4, Laxmi Nagar", "sector 15 mein", "Hanuman Mandir ke paas, Kothrud"), `ADDRESS.multiline` (one part per line; `\n` or `\r\n` at random), `ADDRESS.partial` (unit + locality only: "B-204, Baner"), `ADDRESS.nocomma` ("flat 302 sai krupa apts kothrud pune 411038"), `ADDRESS.abbrev` ("nr.", "opp", "apts", "rd", "H.No.", "Pin-").
  - `ADDRESS.landmark_only` ("near the metro station") — gold ADDRESS, used only in `hard` (known miss by design).
  - Mixed case at random (capitalised / lowercase) per address.
- Building, street, landmark word pools are **split-disjoint** (two tuples each); building suffixes and street words include forms not used in the other split (e.g. dev "Apartments", test "Apts"; dev "Marg", test "Path") — the detector's keyword list is written from the design, not from these pools.
- New fillers: `PLACE` (a city or locality, unlabelled), `SIXDIGIT_PIN_LIKE` (a valid PIN used as an order/OTP number).
- `address_style(template: str) -> str | None` in `generate.py` (like `name_form`): `"mixed"` for plain `{ADDRESS}`.
- Templates appended **in consecutive pairs** (index parity = split): `support` +8 (formal_pin, formal, landmark, partial), `agent_record` +4 (multiline, abbrev), `hinglish` +4 (hinglish, nocomma), `negative` +10 (no gold: "OTP is {~SIXDIGIT_PIN_LIKE}", "order {~SIXDIGIT_PIN_LIKE} not delivered", "I live in {~PLACE}", "shifting to {~PLACE}, {~PLACE} next month", "stuck in traffic near {~PLACE}", "the society meeting is on Sunday", "score was 12/3 at lunch", "meet me near the station", "is delivery available in {~PLACE}?", "pin your message please"), `hard` +2 (`landmark_only`).
- `build_split` passes `{**VALUE_MAKERS, **name_makers(split), **address_makers(split)}`; `SPLIT_SIZES` unchanged unless a category's template count makes a style appear < 20 times per split — then raise that category's size and say so in the report notes (Task 7).
- Handwritten: +25 address messages written **now, before Tasks 3–4**, covering every style plus things the generator lacks (typos "pincode-411 038" is a known-hard case, "Opp:", "landmark:", "c/o", floor numbers, two addresses in one message, an address with a phone right after it), and +8 address look-alike negatives.

- [ ] **Step 1: Write failing tests:**
  - `test_place_pools_disjoint` — no district appears in both `place_pools("dev")` and `place_pools("test")`; each pool ≥ 1,000 offices.
  - `test_invented_localities_not_in_gazetteer` — none of `INVENTED_LOCALITIES[*]` is a key in `load_gazetteer().places`.
  - `test_address_gold_boundaries` — for each style, the maker's value starts with a detail or place and ends with a PIN/place (no leading "Deliver to", no trailing punctuation).
  - `test_every_address_style_in_both_splits` — every style returned by `address_style` appears in dev and test, ≥ 20 samples each.
  - `test_address_negatives_have_no_gold` — negative-category samples have no spans.
  - `test_handwritten_set_shape` — extend: `types[T.ADDRESS] >= 25`, negatives ≥ 20.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests/evaluation -v` → PASS.
- [ ] **Step 5: Commit** `feat: realistic, split-disjoint address evaluation data` — the commit must come before any `address_signals.py` exists (the blind-check ordering the design requires).

### Task 3: Address signals

**Files:** Create `src/leakguard/detect/address_signals.py`. Test: `tests/detect/test_address_signals.py`.

**Interfaces:**
- `SignalKind = Literal["pin", "unit", "building", "street", "landmark", "place"]`; `DETAIL_KINDS = {"pin", "unit", "building", "street", "landmark"}`, `PLACE_KINDS = {"pin", "place"}`.
- `@dataclass(frozen=True) class Signal: start: int; end: int; kind: SignalKind; bare: bool = False` — `bare=True` for unit forms without a keyword ("12/3", "B-204", "#45") and PINs without a "pin" cue: they only count when chained with another, non-bare signal.
- `find_signals(text: str, gazetteer: Gazetteer, locations: Sequence[tuple[int, int]] = ()) -> list[Signal]` — sorted by start; `locations` (NER LOC ranges) become `place` signals.
- Keyword lists live as module constants (`UNIT_WORDS`, `BUILDING_WORDS`, `STREET_WORDS`, `LANDMARK_CUES`, `HINGLISH_POSTPOSITIONS`, `PIN_CUES`, `PIN_BLOCKERS`) with exactly the words in design §4.1 and Global Constraints, plus abbreviations with optional dots (`nr.?`, `opp.?`, `apts?.?`, `rd.?`, `h\.?\s?no\.?`).
- Landmark = cue + 1–4 capitalised words or a known landmark noun (mandir, masjid, church, gurudwara, temple, school, hospital, mall, atm, bank, petrol pump, metro station, bus stand, chowk, park, market). **A cue followed only by a gazetteer place is not a landmark** (that place is still a `place` signal).

- [ ] **Step 1: Write failing tests** (one per kind, positive and negative):
  - `test_pin_with_cue` — "pincode 411038" → one `pin`, not bare; "411038" alone → bare `pin`; "999999" → none; "OTP 411038" and "order no 560034" → none.
  - `test_unit_forms` — "Flat 302", "H.No. 45", "house no 12", "Plot 7", "Door No 4-5", "Shop No. 3", "3rd floor" → `unit`; "12/3" and "B-204" → bare `unit`.
  - `test_building` — "Sai Krupa Apts", "Green Valley Society", "Shanti Niwas", "Krishna Kunj" → `building`; "the society meeting" → none.
  - `test_street` — "MG Road", "Nehru Marg", "gali no 4", "2nd Cross", "Sector 15", "Phase 2", "Block C" → `street`.
  - `test_landmark` — "near Hanuman Mandir", "opp. SBI ATM", "behind City Mall", "Hanuman Mandir ke paas" → `landmark`; "near Kothrud" → only a `place`.
  - `test_places_and_locations` — gazetteer places become `place`; a `locations=[(0, 5)]` range becomes a `place`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: address signal finders`

### Task 4: Assembly, engine wiring, merging

**Files:** Create `src/leakguard/detect/address.py`. Modify `src/leakguard/detect/engine.py`. Test: `tests/detect/test_address.py`, `tests/detect/test_engine.py`.

**Interfaces:**
- Consumes `find_signals`, `Signal`, `DETAIL_KINDS`, `PLACE_KINDS` (Task 3), `load_gazetteer` (Task 1).
- `DEFAULT_GAP = 12` (placeholder until Task 6 tunes it).
- `detect_address(text: str, stops: Sequence[Span] = (), locations: Sequence[tuple[int, int]] = (), gap: int = DEFAULT_GAP) -> list[Span]` — `Span(type=ADDRESS, detector="address", score=1.0)`.
  - Signals overlapping a `stop` span are discarded; a group never chains across a stop span.
  - Two neighbouring signals chain when the text between them, with separators (`,` `;` `-` `/` `.` `:` whitespace incl. `\r\n`, "and", "&") removed, has ≤ `gap` characters.
  - A group is emitted when it has ≥ 1 signal in `DETAIL_KINDS`, ≥ 1 in `PLACE_KINDS`, and ≥ 1 non-bare signal; a group whose only detail is a PIN and only place is that PIN is emitted only if the PIN is cued ("pincode 411038").
  - Span = first signal start → last signal end, trimmed of edge separators, widened to whole words.
- `engine.detect_rules(text)` becomes: `others = [s for d in RULE_DETECTORS for s in d(text)]`, then `merge_spans([*others, *detect_address(text, stops=others)])`. `RULE_DETECTORS` keeps the existing tuple (update the "Phase 5 adds..." comment to say why the address detector runs after the others).

- [ ] **Step 1: Write failing tests** in `tests/detect/test_address.py`:
  - `test_full_address_is_one_span` — "Deliver to Flat 302, Sai Krupa Apts, Kothrud, Pune 411038. Call me" → one span exactly `"Flat 302, Sai Krupa Apts, Kothrud, Pune 411038"`.
  - `test_landmark_address` — "near Hanuman Mandir, Kothrud" → one span covering both.
  - `test_not_addresses` — each of "I live in Pune", "I live in Kothrud, Pune", "stuck in traffic near Kothrud", "near the metro station", "OTP 411038", "score was 12/3" → `[]`.
  - `test_cued_pin_alone` — "my pincode is 411038" → span `"411038"`.
  - `test_crlf_multiline_is_one_span` — `"Flat 302,\r\nSai Krupa Apts,\r\nKothrud, Pune 411038"` → one span over all of it.
  - `test_small_talk_not_masked` — "I'll meet you at the station near the market" → `[]`.
  - `test_two_addresses_two_spans` — "from Flat 2, Baner, Pune to H.No. 45, Laxmi Nagar, Delhi" → two spans.
  - `test_many_six_digit_numbers_is_fast` — 20,000 random valid PINs joined by spaces: completes < 1 s, returns `[]` (bare PINs never chain to a non-bare signal).
  - `test_odd_inputs_return_lists` — `""`, `"   "`, `"🙏🙏"`, `"फ्लैट 302, कोथरूड, पुणे ४११०३८"`, full-width `"４１１０３８"` → each returns a list, no exception.
- [ ] **Step 2: Write failing tests** in `tests/detect/test_engine.py`:
  - `test_address_stops_at_phone` — "Pune 411038 9876543210" and "Baner,Pune-411045,ph:9820031476" → a `PHONE` span on the number and an `ADDRESS` span that ends before it.
  - `test_name_inside_address_absorbed` — `build_detector(name_model=stub returning NAME on "Sai Krupa")` on "Flat 302, Sai Krupa Apts, Kothrud, Pune" → one ADDRESS, no NAME.
  - `test_address_next_to_email_kept_separate` — "B-204, Baner, Pune, email a@b.com" → ADDRESS and EMAIL.
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement.** **Step 5:** `pytest -v` (whole suite; existing guard/masking tests must still pass) + `ruff check .` → PASS.
- [ ] **Step 6: Commit** `feat: assemble address signals into ADDRESS spans`

### Task 5: LOC tags from the name model (same inference)

**Files:** Modify `src/leakguard/ner/decode.py`, `src/leakguard/ner/model.py`, `src/leakguard/detect/engine.py`. Test: `tests/ner/test_decode.py`, `tests/ner/test_model.py`, `tests/detect/test_engine.py`.

**Interfaces:**
- `location_label_ids(id2label) -> tuple[frozenset[int], frozenset[int]]` — `B-LOC/I-LOC`, `B-LOCATION/I-LOCATION`, bare `LOC`/`location`; returns empty sets (no error) for a model without location labels.
- `decode_locations(text, offsets, label_ids, scores, begin, inside, threshold) -> list[tuple[int, int]]` — same grouping and whole-word widening as names (reuse the private helpers; no duplicated logic).
- `@dataclass(frozen=True) class Entities: names: list[Span]; locations: list[tuple[int, int]]` in `ner/model.py`.
- `NameModel.analyse(text: str) -> Entities` — one ONNX pass; `NameModel.detect(text)` returns `analyse(text).names` (compare.py and stubs unchanged).
- `build_detector(name_model: Detector | None = None, ner: Callable[[str], Entities] | None = None) -> Detector` — `ner` takes precedence; when given, names come from it and its `locations` are passed to `detect_address`. Requires a `detect_rules(text, locations=())` parameter.

- [ ] **Step 1: Failing tests:** `test_location_labels_found` (hrl-style labels), `test_no_location_labels_is_empty`, `test_decode_locations_whole_words`, `test_analyse_runs_model_once` (fake session counts calls), `test_ner_locations_reach_address` (stub `ner` returning a LOC range over an invented locality makes "Flat 2, Hiranandani Meadows" an ADDRESS; without it, `[]`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** `pytest -v` → PASS (real-model tests run only if `models/ner` exists).
- [ ] **Step 5: Commit** `feat: expose NER location tags as address signals`

### Task 6: Tune on dev, decide LOC, wire the app

**Files:** Create `src/leakguard/evaluation/address_tuning.py`, generated `docs/evaluation/address-tuning.md`. Modify `src/leakguard/detect/address.py` (`DEFAULT_GAP`), `place_stopwords.txt`, `src/leakguard/app.py`, `src/leakguard/evaluation/__main__.py` (`load_detector`). Test: `tests/evaluation/test_address_tuning.py`, `tests/test_ner_app.py`.

**Interfaces:**
- CLI `python -m leakguard.evaluation.address_tuning` (dev only; no test option, by design). Grid: `gap ∈ {0, 6, 12, 18, 24, 32}` × `LOC ∈ {off, on}` (on only if `models/ner` exists). Per cell: ADDRESS P / R / F1 / coverage on dev, FP count on address negatives, p95 latency of `detect_address`.
- `select_setting(rows) -> tuple[int, bool]` — highest ADDRESS F1; ties within 1.0 point → higher coverage; LOC is chosen only if it beats LOC-off at its best gap on F1 or coverage **and** its precision is not lower. Rule written into the generated doc.
- Stop-list pass: list the dev false positives caused by `place` signals; add an office name to `place_stopwords.txt` only if it is a common word (not because it fixes one sample); the doc lists every word added.
- If LOC is kept: `app._load_name_model` returns the `NameModel`; `create_app` calls `build_detector(ner=model.analyse)`; `load_detector` in `__main__` likewise. If not: no app change, and the doc says why.
- [ ] **Step 1: Failing test** `test_select_setting_rule` (F1 leader; coverage tie-break; LOC rejected when precision drops).
- [ ] **Step 2–3: Implement**; tests PASS.
- [ ] **Step 4: Run the CLI**, apply the stop-list pass, re-run, set `DEFAULT_GAP` (and LOC wiring) from the doc. Inspect dev errors with `python -m leakguard.evaluation errors --split dev` and fix detector **defects** found there (rules, not per-sample hacks); re-run tuning after each fix.
- [ ] **Step 5:** if LOC kept, add `test_app_uses_ner_locations` (stub `ner` through `create_app`); `pytest -v` PASS.
- [ ] **Step 6: Commit** `feat: tune address gap and LOC signal on dev`

### Task 7: Report with addresses

**Files:** Modify `src/leakguard/evaluation/report.py`, `__main__.py`; regenerate `docs/evaluation/report.md`. Test: `tests/evaluation/test_report.py`.

**Interfaces:**
- `UNBUILT = {}` (keep the mechanism for future types).
- `render_report(..., address_styles: dict[str, TypeScores] | None = None)` → **Addresses by style (test)** table: Recall, Coverage, Exact per style, built with `address_style()` like `name_forms`.
- Latency: add `"address detector alone"` row set (`measure_latency(test, lambda t: detect_address(t))`).
- NOTES additions: dataset extended in Phase 5 so non-ADDRESS rows moved; address keyword lists and templates written by the same author (handwritten set and invented localities are the more independent checks); the PIN directory snapshot date; any `SPLIT_SIZES` change.
- [ ] **Step 1: Failing tests** `test_address_not_marked_unbuilt`, `test_report_has_address_styles_table`.
- [ ] **Step 2–4:** implement; PASS.
- [ ] **Step 5: Run** `python -m leakguard.evaluation report` once. Read the ADDRESS rows and examples; **change nothing based on them**.
- [ ] **Step 6: Commit** `feat: report addresses by style`

### Task 8: Packaging, Docker, README

**Files:** Modify `README.md` (types table: ADDRESS built, the address definition in one paragraph, data attribution), verify `pyproject.toml` package-data and `Dockerfile`.
- [ ] **Step 1:** `pip install . --target <scratch dir>` and confirm `leakguard/data/pin_directory.tsv.gz` is inside.
- [ ] **Step 2:** `docker compose build && docker compose up -d`; send "Deliver to Flat 302, Sai Krupa Apts, Kothrud, Pune 411038, call 9820031476" via `examples/chat.py` → upstream sees `[ADDRESS_1]` and `[PHONE_1]`; client gets both restored. Record image size change.
- [ ] **Step 3: Commit** `docs: README for address detection` (plus any packaging fix).

### Task 9: Walkthrough, PR, private notes

- [ ] `docs/walkthrough/phase-5-addresses.md`: why addresses are hard; the definition and what is deliberately not masked; the PIN directory (source, licence, why only a derived file is committed); signals → assembly; why the address detector runs after the other rules (merge would otherwise drop a whole address overlapping a phone); OTP defence; split-disjoint places and invented localities; the tuning result and LOC decision; per-style results; known failures; interview Q&A.
- [ ] Self-review (superpowers:requesting-code-review), push `phase-5-addresses`, open PR, watch CI.
- [ ] Private (outside the repo): append Phase 5 to `data-leak-guard-private/build-journal.md` (glossary: gazetteer, PIN directory, span assembly, coverage; "what I did myself" if the user downloaded the CSV); fill the Addresses row in `interview-guide.md` with measured test numbers.
- [ ] The user merges the PR.
