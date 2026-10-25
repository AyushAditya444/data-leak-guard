# Phase 5 Walkthrough — Indian postal addresses

> **What Phase 5 builds:** detection of full Indian postal addresses as **one** `[ADDRESS_n]` placeholder, from small "piece" detectors plus India Post's PIN code directory, grouped by an assembler whose one setting was tuned on the dev split. Measured per address style on rebuilt evaluation data.

Design for this phase: [2026-10-25-phase-5-addresses-design.md](../superpowers/specs/2026-10-25-phase-5-addresses-design.md). Plan: [2026-10-25-phase-5-addresses.md](../superpowers/plans/2026-10-25-phase-5-addresses.md).

---

## 0. Why addresses are the hardest type

Every earlier type has a shape: Aadhaar is 12 digits with a checksum, a PAN is `ABCPS1234K`, even a name is a few capitalised words the model recognises. An Indian address has no fixed shape at all:

- `Flat 302, Sai Krupa Apts, Kothrud, Pune 411038` (formal)
- `near Hanuman Mandir, Kothrud` (landmark instead of a street number)
- `h.no. 45, gali no 4, laxmi nagar` (Hinglish, lowercase)
- `B-204, Baner` (partial: no city, no PIN)
- one part per line, pasted from a CRM

…and its edges are fuzzy: where does "Deliver to Flat 302, …, Pune 411038. Call me" start and end?

So the first decision was **not technical: what counts as an address?** Without a precise definition you can't label test data, can't measure, and can't defend the numbers in an interview.

---

## 1. The definition

> An address is masked when it has **at least one specific detail** and **at least one place**, close together.

| Detail | Place |
|---|---|
| flat/house/plot no. (`Flat 302`, `H.No. 45`, `12/3`, `B-204`), building (`Sai Krupa Apts`), street (`MG Road`, `gali no 4`, `Sector 15`), landmark (`near Hanuman Mandir`, `Shiv Mandir ke paas`), PIN code | locality, city, district, state (from the PIN directory), a name with a locality ending (`Palm Grove Layout`), an NER location tag, PIN code |

**Deliberately not masked:**

| Text | Why not |
|---|---|
| `I live in Pune` | a place alone is not an address (the main spec's non-goal) |
| `I live in Kothrud, Pune` | places only; with a PIN it *becomes* an address |
| `stuck in traffic near Kothrud` | "near" + a place is someone's rough location, not their address |
| `near the metro station` | no place: could be anywhere. Labelled as an address in the data, so it shows up as a **known miss** |
| `OTP 411038`, `order no 560034` | a real PIN after "OTP"/"order" is not used as a PIN |

**The span** runs from the first piece to the last, as one span. That rule matters for leaks: masking "Kothrud, Pune 411038" while leaving "Flat 302, Sai Krupa Apts" visible would still let the LLM (and its logs) know which flat.

### Q&A
- *Why not mask "near Kothrud" to be safe?* Over-masking has a cost too: the LLM loses meaning ("is delivery available near Kothrud?" becomes useless). The line is drawn where the text identifies *a dwelling*, not *an area*.

---

## 2. The PIN directory — `detect/gazetteer.py`

### Files
| File | What |
|---|---|
| `src/leakguard/detect/gazetteer.py` | `load_gazetteer()`, `find_places()`, `normalise_office()`, the build CLI |
| `src/leakguard/data/pin_directory.tsv.gz` | derived file: PIN, office, district, state (1.06 MB) |
| `src/leakguard/data/place_stopwords.txt` | office names that are everyday words |
| `src/leakguard/data/README.md` | source, download date, licence, how to rebuild |

The source is India Post's **All India Pincode Directory** on data.gov.in: 165,627 post offices, 19,586 PINs, 750 districts, under the Government Open Data License – India. It answers two questions:

1. **Is this 6-digit number a real PIN?** `999999` isn't; random OTPs often aren't either.
2. **Where are the place names in this text?** All ~133,000 office, district and state names (plus common city names like "Bangalore", "Gurgaon").

**Why only a derived file is committed.** The raw CSV is 23 MB, with fields we don't need (lat/long, circle, delivery status). The build keeps four columns, strips the office type ("Kothrud S.O" → "Kothrud", "Laxmi Nagar SO East Delhi" → "Laxmi Nagar"), deduplicates and gzips: **1.06 MB**. The proxy and the Docker image then work offline.

**`find_places`** splits the text into words once and, at each word, tries the longest name first (up to 5 words), so "Navi Mumbai" is one place, not "Mumbai". Words of one place may only be separated by spaces, so "Kothrud, Pune" is two places. A direction word right after a place is part of it ("Andheri **West**"), otherwise the mask would leak "West".

**The noise problem.** 133,000 names include ordinary words: villages called *Late*, *Main*, *Bank*, *Mango*, and surnames like *Sharma*. Two defences:
- a place alone is never masked (§1), so noise only matters next to a detail;
- a stop list of everyday words, extended on dev (§6).

### Q&A
- *Why a gazetteer and not a model for places?* The model's location tags help (§6) but miss lowercase Hinglish entirely; a lookup table is exact, offline and free.
- *How did you get the data?* data.gov.in needs a form or an API key; the author downloaded the CSV once. The build is one command from that file.

---

## 3. The pieces — `detect/address_signals.py`

Each finder marks one kind of piece as a `Signal(start, end, kind, bare)`:

| Finder | Matches | Notes |
|---|---|---|
| `_pins` | `[1-9]\d{5}` in the directory | after "pin/pincode/postal code (is)" → cued; after OTP, order, ticket, invoice, ref, txn, id, Rs, ₹… → ignored |
| `_units` | `Flat 302`, `H.No. 45`, `Plot 7`, `3rd floor`, `QTR NO 45` | bare forms `12/3`, `B-204`, `#45` are marked **bare** |
| buildings | 1–3 name words + Apts/Society/Residency/Niwas/CHS/… | "the society meeting" is rejected (leading "the", no name left) |
| `_streets` | `MG Road`, `Nehru Marg`, `45 Anna Salai`, `2nd Cross`, `gali no 4`, `Sector 15` | |
| `_landmarks` | `near/opp./behind/next to` + a named thing; `X ke paas/saamne/peeche` | `near Kothrud` is **not** a landmark (the thing is just a place) |
| places | gazetteer, locality endings (`Palm Grove Layout`), NER location tags | a place *inside* a landmark is dropped |

**Bare pieces.** A bare unit (`12/3`) or an uncued PIN (`411038`) is too common in other text (scores, dates, OTPs) to count on its own. A group needs at least one non-bare piece to become an address. That is what makes a pasted table of 20,000 six-digit codes produce **no** addresses (test `test_many_six_digit_numbers_is_fast`).

**"Bus Stand" is a post office.** So is "Hanuman Mandir". Two consequences, both found by tests:
- a landmark whose name contains a landmark noun (mandir, stand, station…) stays a landmark even if it's also an office name;
- the place *inside* that landmark is dropped, or "I'm near the bus stand" would be landmark + place = an address.

### A performance bug the old tests caught
The building and street patterns start with "up to 3 words, then a suffix". On `"a" * 100_000 + "@"` (one of Phase 2's adversarial inputs), Python's regex engine retried that from **every letter** of the long word: quadratic, **108 s**. The fix is one lookbehind, `(?<![\w.'&])`: a match may only start at the start of a word. Now it's under 0.4 s. This is the same ReDoS lesson as Phase 2's email pattern, and the reason the adversarial test exists.

---

## 4. Assembly — `detect/address.py`

`detect_address(text, stops, locations, gap)`:

1. find all signals; drop those overlapping a **stop** (spans other detectors already found);
2. walk them left to right and chain neighbours into groups while the text between them, ignoring separators (`, ; : / . - & ( )` whitespace, "and"), has at most **gap** characters;
3. never chain across a stop or a blank line; a house/flat number that follows a place with words in between starts a **new** address ("from Flat 2, Baner, Pune **to** H.No. 45, …" is two);
4. drop a leading place joined to the rest only by ordinary words ("**Reddy** here, my new address is behind Axis Bank, …": Reddy is also a village);
5. keep groups with a detail, a place and a non-bare piece; the span runs from the first piece to the last, trimmed and widened to whole words.

### Why the address detector runs *after* the other rules (`detect/engine.py`)
Merging keeps the higher-priority span when two overlap. ADDRESS has low priority, so an address span that swallowed a phone number ("Pune 411038 9876543210") would lose to the PHONE and be **dropped whole**, leaking the address. So `detect_rules` runs every other detector first and passes their spans as `stops`: the address stops before the phone. Test: `test_address_stops_at_phone`.

### Names inside addresses
The spec says a name inside an address ("Sai Krupa Apts" tagged as a person) is absorbed. Dev showed a harder case: the surname *Sharma* is also a village, so in "customer Kavya Sharma, Flat 302, Kothrud" the address started **inside** the name; merging then dropped the NAME and "Kavya" leaked. `engine._merge` now widens an address over any name it overlaps (unless that name also overlaps something else, like an email). Test: `test_name_partly_inside_address_is_absorbed`.

---

## 5. Test data that doesn't grade itself — `evaluation/values.py`, `templates.py`

The Phase 3 address generator had 12 localities in 4 fixed patterns, all ending with the city, shared by dev and test. Phase 4 taught the lesson: if the generator and the detector share their vocabulary, the scores measure the generator. So:

- **Places split by district.** Dev and test draw post offices from different districts (by a hash of the district name), so test addresses use localities tuning never saw. Half are drawn from big-city districts, since 85% of post offices are rural branches and support chats come mostly from cities.
- **Invented localities** (~25% of addresses): real-sounding layouts and societies that are *not* post offices ("Whispering Woods Layout", "Rohan Abhilasha"), different in dev and test, so finding places isn't 100% by construction.
- **Split vocabularies:** building names and suffixes, streets, landmark cues and things differ between dev and test (`Apartments` vs `Apts`, `near` vs `nr.`).
- **Nine styles**, each its own template slot so the report can break results down: `formal_pin`, `formal`, `landmark`, `hinglish`, `multiline` (`\n` or `\r\n`), `partial`, `nocomma`, `abbrev`, and `landmark_only` (the known miss).
- **Look-alikes:** a real PIN used as an OTP, order number or coupon code; "I live in {city}"; "stuck in traffic near {place}"; "the society meeting"; "score was 12/3"; "meet me near the station".
- **Hand-written:** 34 new messages (27 addresses, 8 negatives), written and committed **before any detector code existed**, so they're a blind check on both the generator and the rules. Every PIN in them was checked against the directory.

---

## 6. Tuning on dev — `evaluation/address_tuning.py`

One command, dev only, writes [address-tuning.md](../evaluation/address-tuning.md). Grid: gap ∈ {0, 6, 12, 18, 24, 32} × location tags {off, on}, each scored with the full pipeline as deployed.

**Rule (written before running):** highest ADDRESS F1; within 1 F1 point, highest coverage, then the smaller gap. Location tags are kept only if they beat "off" on F1 or coverage *without* lowering precision.

**Result:** gap **24**, location tags **on**: 100.0% precision, 91.1% recall, 95.3 F1, 95.1% coverage on dev, and **no** negative message flagged. Recall can't go much higher on dev: the 24 `landmark_only` addresses (8% of the dev addresses) are misses by definition.

Why a gap as wide as 24? Invented localities ("Green Glen Layout", 16 characters) sit *between* the known pieces; with a small gap the address split in two and part of it leaked.

**Fixes found in dev errors** (each with a test that failed first):
- locality endings (Nagar, Colony, Layout, Vihar, Puram, Bagh, Ganj, Peth, Extn, …) as place signals;
- names overlapping an address are absorbed (§4);
- leading places joined by ordinary words are dropped (§4);
- the stop list gained 15 everyday/festival words seen matching outside addresses: `late`, `main`, `dena`, `tak`, `khata`, `naya`, `both`, `same`, `mango`, `bank`, `royal`, `diwali`, `raksha`, `bandhan`, `guru`. Surnames were deliberately **not** added (a list of names would be both endless and circular); the merge fix handles them.

**Location tags.** `NameModel.analyse()` returns names *and* location ranges from the **same** model run, so they cost nothing extra. The model finds "Powai, Mumbai" but not lowercase Hinglish, which is why they're an extra signal, never the only one. The address detector works with NER off.

---

## 7. Results on the test split

From [report.md](../evaluation/report.md) (test split, 1,600 messages, run once):

| | Precision | Recall | F1 | Coverage | Exact |
|---|---:|---:|---:|---:|---:|
| ADDRESS (271 addresses) | 99.6% | 89.7% | 94.4% | 94.3% | 81.9% |

By style: every style except `landmark_only` has **93–100% recall**; `landmark_only` is 0%, as defined. Hand-written set: 96.4% precision and recall on 28 addresses. The address detector alone takes **0.11 ms** per message (p50); the whole pipeline is dominated by the name model (~17 ms).

### What it gets wrong (and why)
- **Landmark-only addresses** ("near the old post office"): by definition. A follow-up could mask them when the message says it's an address ("my address is near…").
- **Partial addresses with an unknown locality** ("H.No. 41, Sunshine Enclave"): no place piece, so no address. "Enclave" is a building word, and the locality isn't in the directory.
- **Spaced PINs** ("pincode-411 038"): known-hard, in the hand-written set.
- **Boundaries in lowercase Hinglish:** "**bhai** shivaji park ke paas, …" also masks "bhai", and "the Gurudwara ke paas" loses "the". That's one filler word too many or too few: harmless for leaks, but it's why Exact is 0% for the `hinglish` and `abbrev` styles. Noticed while reading the test report, so it was **not** fixed in this phase (fixing it from test would make the test numbers optimistic).

### Honest caveats (also in the report's notes)
- The same author wrote the keyword lists, the generator's vocabulary and the hand-written addresses.
- The generator's invented localities in *both* splits often end in Layout/Colony/Extn, which the detector recognises, so recall on those may be optimistic.
- Directory places are found by construction, because generator and detector use the same directory. Real users' localities that aren't post offices depend on the endings and the NER tags.
- Other types' rows moved from the Phase 4 report because the `hard` category grew (170 messages per split, so every address style reaches 20 samples). On dev, adding the address detector changed no other type's recall.

---

## 8. Wiring and Docker

- `app.create_app` builds the detector with `build_detector(ner=model.analyse)`: names and location tags from one model run. With `LEAKGUARD_NER=off` it's rules only, addresses included.
- It also calls `load_gazetteer()` **at startup**. Found while testing the Docker image: the first request after a restart spent **754 ms** in detection loading the 1 MB directory lazily; now it's 42 ms. It also means missing data stops the guard at boot, not mid-request.
- The bundled files ship inside the package (`pyproject.toml` package-data: `data/*.gz`). The image grew by about 10 MB (1.02 → 1.03 GB) and still has no PyTorch.
- In Docker, "Deliver to Flat 302, Sai Krupa Apts, Kothrud, Pune 411038, call 9820031476" logs `{"ADDRESS:mask": 1, "PHONE:mask": 1}` and the reply comes back with both restored. "Your OTP is 411038, I live in Pune" logs no findings.

---

## 9. Interview Q&A

- **How do you detect addresses?** Small rules for each piece (units, buildings, streets, landmarks, PINs), places from India Post's directory and the NER model, then an assembler that groups nearby pieces and keeps groups with a detail and a place. One knob, the gap, tuned on dev.
- **Why not train a model?** I'd have to train it on my own generated data, so it would learn my generator and the test would grade itself. Rules make every miss explainable, run in 0.1 ms, and need no GPU.
- **How do you avoid flagging OTPs?** A 6-digit number counts only if it's a real PIN, isn't after OTP/order/ticket…, and sits next to another address piece (or after "pincode"). The negatives include real PINs used as OTPs and order numbers: 0 of them flagged on dev.
- **What's coverage and why report it?** The share of the address's characters actually hidden. A detector that masks "Kothrud, Pune 411038" but leaves "Flat 302, Sai Krupa Apts" visible "overlaps" the gold span, and still leaked the flat.
- **What was the nastiest bug?** Two. A regex that was quadratic on one long word (108 s → <0.4 s, caught by an old adversarial test). And a privacy bug only real data showed: surnames that are also village names made an address start inside a name, so merging dropped the name and leaked the first name.
- **Why does the address detector run after the other rules?** Because merging keeps the higher-priority span: an address that overlapped a phone would be dropped whole. Running last, with the other spans as stops, means it ends before the phone.
- **What would you do next?** Mask landmark-only addresses when the text says "my address is…"; tighten the Hinglish landmark boundaries; add a small city/locality alias list for spellings people type (Bombay → Mumbai is done; Gurgaon sectors, "Blr").
