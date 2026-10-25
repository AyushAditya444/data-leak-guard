# Phase 5 design: Indian address detection

Date: 2026-10-25. Extends §5 (Addresses), §7 and §9 of
[the main design](2026-10-24-data-leak-guard-design.md); where the two differ, this one wins
for Phase 5.

## 1. Intent

Mask a full Indian postal address in a chat message as **one** `[ADDRESS_n]` placeholder, so
no part of it reaches the LLM — not the PIN, not the flat number, not the society name.
Addresses are the hardest type in the project: no fixed format, landmarks instead of street
numbers, Hinglish, missing parts, fuzzy edges.

The goal for the portfolio is not "catches every address". It is a **precise, defensible
definition** of what the system calls an address, measured honestly (recall, precision and
coverage per address style), with known misses listed — deliberate ones and accidental ones.

## 2. What counts as an address

An address is masked when it has **at least one specific detail and at least one place**,
close together.

| Kind | Examples |
|---|---|
| **Detail** | flat / house / plot number ("Flat 302", "H.No. 45", "Plot 7", "12/3", "B-204"); a named building or society ("Sai Krupa Apts", "Green Valley Society"); a street ("MG Road", "Nehru Marg", "gali no 4", "2nd Cross"); a landmark phrase ("near Hanuman Mandir", "opp. SBI ATM", "behind City Mall", "Hanuman Mandir ke paas"); a PIN code |
| **Place** | a locality, city, district or state; a "sector N" / "phase N" with a city; a PIN code |

A PIN code counts as both, so "Kothrud, Pune 411038" is an address.

**Not masked (by design):**
- a place alone: "I live in Pune", "shifting to Kothrud next month";
- places only, with no detail: "I live in Kothrud, Pune" (with a PIN it becomes an address);
- a landmark cue plus a place with no landmark name: "stuck in traffic near Kothrud";
- a landmark with no place: "near the metro station" — reported as a known miss;
- a 6-digit number that is not a valid PIN, or a valid PIN with no address context ("OTP
  411038", "order 560034").

**PIN codes:** 6 digits, first digit 1–9, present in the bundled India Post directory. A PIN
needs context — next to an address piece, or after "pin", "pincode", "PIN code", "postal
code". A standalone PIN with that context is masked as `ADDRESS` on its own (main spec §5).

**Span:** from the first detail or place that belongs to the address to the last one (usually
the PIN or city), as one span, so masking "Kothrud, Pune 411038" while leaving "Flat 302, Sai
Krupa Apts" visible cannot happen. Surrounding words ("Deliver to", "pe chhod dena") are
outside the span. Addresses may run over line breaks (agents paste them one part per line),
unlike names.

## 3. Approach

**Considered:**
- **A. Signals + assembly rules (chosen).** Small rules each find one kind of piece; an
  assembler groups nearby pieces and applies the §2 rule. Fast, explainable, every miss has a
  reason. Weakness: phrasing outside its vocabulary is missed — reported, not hidden.
- **B. Train an address-tagging model** on generated data. Handles odd phrasing better, but it
  would learn the generator's habits, so test scores would mostly measure how well it copies
  our own data — the circularity Phase 4 avoided. Also brings back torch training and model
  weight on a laptop that struggled with Phase 4's model jobs. Rejected.
- **Off-the-shelf parsers (libpostal etc.).** They parse text already known to be an address;
  they do not find addresses inside chat. libpostal also needs ~2 GB of data. Rejected.

The user chose A.

## 4. Design

### 4.1 Signals

Each signal is a span with a kind (`detail` or `place`, PIN = both) and a sub-kind for
debugging and the report (`unit`, `building`, `street`, `landmark`, `locality`, `city`,
`pin`, `ner_loc`).

1. **PIN codes** — validated against the directory set (§4.4).
2. **Unit numbers** — keyword + number: flat, flat no, H.No., house no, plot, door no, room,
   floor, shop no, quarter no, and bare forms like "12/3", "B-204", "#45" when next to
   another address piece (bare forms alone are too common: dates, scores, ratios).
3. **Buildings** — capitalised words ending in a building word: apartments/apts, society,
   residency, towers, heights, enclave, niwas, sadan, bhavan, complex, chs, kunj, villa(s).
4. **Streets** — a name ending in road/rd, marg, street/st, lane, cross, main, path, chowk,
   bazaar, or "gali (no) N", "lane N", "sector N", "phase N", "block X".
5. **Landmarks** — a cue plus a named thing: near, nr., opp., opposite, behind, beside, next
   to, in front of, and the Hinglish postpositions "X ke paas / ke saamne / ke peeche". A cue
   followed directly by a place name only ("near Kothrud") is **not** a landmark.
6. **Places (gazetteer)** — localities, cities, districts, states from the bundled directory
   (§4.4), matched case-insensitively on whole words, longest match first.
7. **NER location tags** — when the name model is enabled, its LOC predictions become extra
   `place` signals. They are signals only, never masked on their own. Whether they are kept
   is decided **on dev**: compare rules-only vs rules + LOC; keep LOC only if it improves
   ADDRESS F1 or coverage without lowering precision. Either way, the address detector must
   work with NER off.

### 4.2 Assembly

Signals are sorted by position and chained into a group while the gap between neighbours is
small: only separators (commas, spaces, newlines, hyphens, "and"), at most **G** characters
of other text. A group becomes an `ADDRESS` span when it contains at least one detail and at
least one place, and is not one of the §2 "not masked" shapes. **G** (and any other scoring
knob) is tuned on dev, like the name threshold in Phase 4. The span covers the group,
trimmed of edge punctuation and widened to whole words.

### 4.3 Merging

Uses the existing fixed precedence (secrets > checksum IDs > other rules > ADDRESS > NAME). A
name inside an address ("Sai Krupa Apts", "Nehru Marg") is absorbed by the ADDRESS span. A
phone, email or other higher-priority value next to an address stays separate; the address
span must not swallow it (the assembler stops at it). Unit tests cover these cases.

### 4.4 Bundled data

Source: India Post's **All India Pincode Directory** on data.gov.in (fields include office
name, district, state, PIN). At implementation time the builder confirms the current
licence (expected: Government Open Data License – India), records source URL, download date
and licence in `src/leakguard/data/README` with the required attribution, and falls back to a
named mirror only if the official download is unavailable.

- A build script turns the raw CSV into a compact runtime file (PIN set + normalised place
  names; office suffixes like "S.O", "B.O", "H.O" stripped). The **raw CSV is not committed**;
  the compact file is, so the proxy and Docker image work offline.
- Place names that are common English or Hinglish words (e.g. "Market", "Camp", "Station")
  are dropped from the gazetteer by a stop list, measured by the false positives they cause
  on dev.

### 4.5 Integration

- New `detect/address.py`; added to `RULE_DETECTORS` (engine.py already marks the slot).
- NER LOC: the name model currently returns NAME spans only. It must also expose LOC spans
  from the **same** inference call (no second model run), which the address detector
  consumes. Shape of that API is the builder's call.
- Remove `ADDRESS` from the report's `UNBUILT` list. Policy already has `ADDRESS: mask`.

## 5. Evaluation data (most of the work)

Lessons from Phase 4 apply: the detector and the test data must not share their vocabulary,
or the scores measure the generator.

1. **Split-disjoint places.** Dev and test draw localities and cities from disjoint parts of
   the directory (e.g. split by district), so test addresses use places the tuning never saw.
   The detector's gazetteer still holds the whole directory, as in production.
2. **Places outside the directory.** Some generated addresses use real-sounding localities
   that are not post office names (new societies, layouts, "Sector 62"), so place recall is
   not 100% by construction.
3. **Many more styles**, each tagged so the report can break results down by style: formal
   with PIN, formal without PIN, landmark-led, Hinglish ("sector 15 pe", "ke paas"),
   multi-line pasted, partial (unit + locality only), comma-less, lowercase/typed fast,
   abbreviations ("nr.", "opp", "apts", "rd").
4. **Negatives:** 6-digit OTPs, order and ticket numbers that happen to be valid PINs, a city
   or locality outside an address, "near X" small talk, ratios and dates like "12/3",
   building-like words in non-address text ("the society meeting").
5. **Hand-written addresses**, written **before the detector rules are finalised**, so they
   are a blind check on the generator and the rules. Who writes them is open (§8).
6. **Gold boundaries** follow §2: first piece to last piece, PIN included, surrounding words
   excluded.

## 6. Metrics and report

- ADDRESS precision / recall / F1 with the existing overlap matching, plus **coverage**
  (already implemented in `metrics.py`), overall and **per address style**.
- False positives on the address negatives, especially OTPs.
- Detection latency with addresses added (p50/p95); the address detector's own cost.
- Dev comparison table: rules-only vs rules + NER LOC, and the chosen G.
- Test is run once for reported numbers; never tuned on (as in Phases 3–4).
- No target numbers are set in advance; whatever is measured is reported.

## 7. Non-goals for Phase 5

- Validating that a PIN matches its city, geocoding, or normalising addresses.
- Non-Indian addresses; Devanagari or other scripts; numbers written as words.
- Masking places outside addresses (main spec non-goal).
- Landmark-only mentions with no place.

## 8. Open / left to the builder

- **Open (user):** whether the user writes some of the hand-written addresses themselves —
  the strongest independent check, since the builder also writes the rules.
- Builder's call: exact keyword lists, regexes, gazetteer data structure, NER LOC API shape,
  stop-list contents, generator style weights, split method for places, file layout of the
  compact data file — all to be stated in the plan and the walkthrough.
