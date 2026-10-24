# Phase 3 Walkthrough — Evaluation

> **What Phase 3 builds:** the measuring stick. Synthetic labelled data, a scoring harness, and a generated report, so every claim in the README is a number that can be reproduced with one command.

---

## 1. Samples, marked text, and scoring

### Files
| File | Purpose |
|---|---|
| `src/leakguard/evaluation/samples.py` | `Sample`, `parse_marked()`, JSONL read/write |
| `src/leakguard/evaluation/metrics.py` | `TypeScores`, `Evaluation`, `ErrorCase`, `score()` |

### Marked text: labels without counting characters
A labelled sample needs the exact position of every sensitive value. Counting offsets by hand is error-prone, so samples are written like this:
```
call [[PHONE:98765 43210]] or mail [[EMAIL:a@b.in]]
```
`parse_marked()` removes the markers and records where each value landed:
```
text  = "call 98765 43210 or mail a@b.in"
spans = [PHONE 5..16, EMAIL 25..31]
```
The same format is used by the generator (Task 3) and the hand-written set (Task 4), so there is one code path for labels.

### Precision, recall, F1: a worked example
Say the test data contains **10** phone numbers. The detector flags **9** things as phones: **8** are real, **1** is an order ID.
- **True positives (TP)** = 8, **false positives (FP)** = 1, **false negatives (FN)** = 10 − 8 = 2
- **Precision** = TP / (TP + FP) = 8/9 = **89%**: "when it says phone, how often is it right?"
- **Recall** = TP / (TP + FN) = 8/10 = **80%**: "of all real phones, how many did it catch?"
- **F1** = 2·P·R / (P + R) = **84%**: one number balancing both.

For a data-leak guard, **recall is the safety number** (a miss is a leak) and **precision is the usability number** (false alarms garble the chatbot's answers, or block requests for secrets).

### The matching rule
A prediction counts as correct when it has the **same type** and **overlaps** a gold span.
- **Why overlap, not exact boundaries?** If the gold is `+91 98765 43210` and the detector masks `98765 43210`, the number was still hidden. Exact-boundary matches are reported separately (`exact`), so boundary sloppiness is visible but doesn't count as a miss.
- **One-to-one.** Each gold and each prediction can be matched only once, and the biggest overlaps are paired first. Otherwise one huge prediction covering two phone numbers would be counted as two hits (`test_one_prediction_matches_one_gold`).

### Coverage: "was it hidden?"
Recall asks "did we find it *with the right label*?". But if a 10-digit bank account is labelled PHONE by mistake, it is **still masked**, so nothing leaked. **Coverage** is the share of gold characters covered by *any* prediction:
```
gold:  a/c 50100123456789        (14 characters)
pred:      5010012345            (10 characters)
coverage = 10/14 = 71%  → 4 digits leaked
```
For leaks, coverage is the honest metric; recall is the precise one. The report shows both.

### Errors are kept, not just counted
Every FP and FN is stored as an `ErrorCase` (sample id, kind, type, value, text). That's what makes error analysis possible: reading *which* things were missed, not just how many.

### Detector crashes surface
`score()` does not catch exceptions. If a detector crashes on some input during evaluation, that's a bug to see and fix, not a silently skipped sample (`test_score_propagates_detector_errors`).

### Q&A
**Q: Why report coverage as well as recall?**
A: They answer different questions. Recall asks whether the detector found the value with the right type. Coverage asks whether the characters were actually hidden from the LLM. A bank account mislabelled as a phone number hurts recall but is still fully masked; a phone number masked without its last digits counts for recall (it overlaps) but leaks a digit, which coverage shows. For a privacy tool, the leak question is the one users care about.

---

## 2. Synthetic values: realistic, valid, and fake

### File
`src/leakguard/evaluation/values.py`: `VALUE_MAKERS`, one function per kind of value.

### Why synthetic?
Testing a privacy tool with real people's Aadhaar numbers would itself be a privacy violation. Every value here is generated: no real person owns `6580 7302 1578`.

### Why *valid* synthetic values?
If test Aadhaar numbers had random check digits, the detector would correctly reject ~90% of them, and the evaluation would report terrible recall for the wrong reason. So:
- Aadhaar: 11 random digits (first 2–9), then the **Verhoeff check digit computed** with `verhoeff_check_digit()`. Computing a check digit needs an extra *inverse* table, compared with only validating one.
- Card: 15 digits, then the one final digit that makes **Luhn** pass.
- PAN: real structure, with a valid holder-type letter (mostly `P` for person).
- IFSC: real bank codes (`HDFC`, `SBIN`…) and the reserved `0`.

Example output:
```
AADHAAR          6580 7302 1578
PHONE            +91 98193 03642
UPI              kavyap@kotak
ADDRESS          near Big Bazaar, Gomti Nagar, Lucknow
SECRET.password  Lotus#154
CAMEL            renderInvoiceLedgerFactory      ← a filler, must NOT be flagged
```

### Variants: the formats people really use
Slot names like `PHONE.words` (`nine eight seven…`), `PHONE.dots` (`98765.43210`), `AADHAAR.mixed` (`2345 6789-0124`), `PAN.spaced` and `UPI.unknown_handle` produce **known-hard** formats. Some are expected misses. They exist so the report can say *exactly* what the guard doesn't handle, instead of hiding it.

### Filler values are deliberately *not* filtered
`ORDER_ID` is a random 12-digit number. About 10% of random 12-digit numbers (starting 2–9) happen to pass the Verhoeff check, so they *look* exactly like Aadhaar numbers, and the detector will flag them. That's not a bug in the data: real order numbers do the same thing. Leaving it in means the report shows the **real-world false-positive rate**, not an artificially perfect one.

### Avoiding circular tests
The generator's tests check values with **checksums and regex structure only**, never with the detectors. If the tests used the detectors ("the Aadhaar detector finds every generated Aadhaar"), a detector bug and a matching generator bug could hide each other.

### Determinism, and a subtle trap
Every maker takes a `random.Random`, so a fixed seed produces the same dataset every time. A trap found while writing it: the UPI handle list is a `frozenset`, and **Python randomises string hashing per process**, so iterating a set of strings gives a different order on every run. `rng.choice(list(the_set))` would therefore break determinism. The generator uses `sorted(...)` first.

### Q&A
**Q: How did you get test data without using real people's data?**
A: A seeded generator. It builds identifiers that are structurally valid (correct Verhoeff and Luhn check digits, real PAN and IFSC structure) but belong to nobody, and fills them into message templates. Look-alikes such as order numbers are left unfiltered, so the false-positive rate reflects reality. A fixed seed makes the whole dataset reproducible byte for byte.

---

## 3. Templates and the dev/test datasets

### Files
| File | Purpose |
|---|---|
| `src/leakguard/evaluation/templates.py` | ~95 message templates in 6 categories |
| `src/leakguard/evaluation/generate.py` | `render_template()`, `build_split()`, `SPLIT_SIZES` |
| `data/eval/dev.jsonl`, `data/eval/test.jsonl` | 1,500 generated samples each, committed |

### The six categories, and why each exists
| Category | Per split | What it represents |
|---|---|---|
| `support` | 450 | customers typing into a chatbot |
| `agent_record` | 150 | support agents pasting **whole customer records** (case 2 from the design) |
| `dev_snippet` | 200 | developers pasting code, `.env` files, curl commands **with keys** (case 3) |
| `hinglish` | 250 | romanised Hindi-English, the way Indians actually type |
| `negative` | 350 | **nothing sensitive**: order IDs, OTPs, tickets, git hashes, UUIDs, "my password is wrong"… Every flag here is a false positive |
| `hard` | 100 | formats the rules are *known* not to handle (spelled-out numbers, dots, unknown UPI handles): measured, not hidden |

### Slot grammar
```
"Please send the refund to a/c no {BANK_ACCOUNT}, IFSC {IFSC}."      gold slots
"My order {~ORDER_ID} was delivered late."                          filler (~): must NOT be flagged
"export AWS_ACCESS_KEY_ID={SECRET.aws}"                              .variant picks a format
```
`render_template()` turns gold slots into marked text (`[[IFSC:HDFC0123456]]`) and fillers into plain values; `parse_marked()` then produces exact spans.

### Splitting by template, not just by sample
A naive split takes 3,000 generated messages and puts half in each set. But then the *same sentences* (only with different numbers) appear in both, so any rule tuned on dev phrasing scores artificially well on test. Here, **each template belongs to one split only** (even index → dev, odd → test). The test set contains phrasings the tuning process never saw, which is closer to meeting real users. `test_splits_share_no_templates` enforces it.

### A balance problem caught by a test
The first version gave the test set only **38** Aadhaar numbers, because most Aadhaar templates happened to sit at even positions. Results built on 38 examples are noisy: one miss moves recall by almost 3 points. A new test (`test_each_type_well_represented_in_both_splits`) requires ≥ 50 examples of every type in each split. Swapping three templates' positions fixed it; now every type has ≥ 88 per split.

### Determinism and committed data
`build_split("dev", seed=1)` always produces byte-identical output. The JSONL files are committed so anyone can inspect exactly what was measured, and `test_committed_datasets_match_generator` fails if the files and the generator ever drift apart.

### Q&A
**Q: How do you make sure your test numbers aren't inflated?**
A: Four ways. Thresholds are tuned only on the dev split. Dev and test share no templates, so test sentences are phrased in ways tuning never saw. Negatives and look-alikes are a fifth of the data, so false positives show up. And a separate hand-written set, not produced by the generator at all, checks the generator isn't making things too easy.
