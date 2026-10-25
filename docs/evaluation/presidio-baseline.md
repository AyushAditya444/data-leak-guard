# Presidio baseline

Generated 2026-10-25 at commit `b4cda99` with `python -m leakguard.evaluation presidio-report`, from `data/eval/presidio-test.json`.

Microsoft Presidio is the best-known open-source PII detector. This page runs it on the same **test** split (1600 messages) and **hand-written** set (93 messages) as [our report](report.md), scored the same way (same type and overlapping; coverage = share of sensitive characters hidden).

- **Out of the box:** `AnalyzerEngine()` exactly as installed, `language="en"`.
- **Best shot:** the same, plus all of Presidio's Indian recognizers turned on and a minimum score per type tuned on dev ([presidio-tuning.md](presidio-tuning.md)). Only Presidio's own settings: no custom recognizers, patterns or models.

In presidio-analyzer 2.2.364, all of Presidio's Indian recognizers (IN_AADHAAR, IN_GSTIN, IN_PAN, IN_PASSPORT, IN_VEHICLE_REGISTRATION, IN_VOTER) are **off by default**, so the out-of-the-box setup cannot find Aadhaar or PAN numbers at all. The best-shot setup turns them on.

Versions: presidio-analyzer 2.2.364, spacy 3.8.16, en_core_web_lg 3.8.0.

## Test split: F1 and coverage

| Type | Ours: F1 | Presidio, out of the box: F1 | Presidio, best shot: F1 | Ours: coverage | Presidio, out of the box: coverage | Presidio, best shot: coverage |
|---|---:|---:|---:|---:|---:|---:|
| AADHAAR | 99.0% | 0.0% | 92.6% | 100.0% | 30.9% | 100.0% |
| PAN | 93.2% | 0.0% | 91.8% | 85.1% | 3.4% | 85.1% |
| PHONE | 95.3% | 77.8% | 77.9% | 94.2% | 100.0% | 100.0% |
| UPI | 64.7% | not supported | not supported | 56.6% | 32.9% | 32.9% |
| EMAIL | 100.0% | 93.9% | 93.9% | 100.0% | 100.0% | 100.0% |
| CARD | 100.0% | 94.7% | 94.7% | 100.0% | 100.0% | 100.0% |
| IFSC | 100.0% | not supported | not supported | 100.0% | 4.8% | 4.8% |
| BANK_ACCOUNT | 98.9% | not supported | not supported | 100.0% | 87.1% | 87.1% |
| SECRET | 94.9% | not supported | not supported | 96.7% | 24.6% | 24.6% |
| NAME | 92.3% | 54.2% | 54.2% | 88.2% | 83.0% | 83.0% |
| ADDRESS † | 94.0% | 48.5% | 48.5% | 93.7% | 39.9% | 39.9% |
| **Types both support** | **95.3%** | **62.6%** | **74.5%** | **93.9%** | **79.6%** | **94.3%** |

*not supported*: Presidio has no detector for this type; its coverage column still shows how much of these values Presidio hid anyway through other detectors. † Presidio's `LOCATION` finds place names, not full postal addresses, so its ADDRESS precision and recall are not like for like; coverage is the fairer measure. *Types both support* adds up AADHAAR, PAN, PHONE, EMAIL, CARD and NAME the same way for every engine.

## False alarms on messages with nothing sensitive (test split)

| Engine | Any type | Types both support |
|---|---:|---:|
| Ours | 5 of 350 | 4 of 350 |
| Presidio, out of the box | 115 of 350 | 67 of 350 |
| Presidio, best shot | 116 of 350 | 68 of 350 |

## Test split: Ours

| Type | Support | Precision | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|
| AADHAAR | 95 | 97.9% | 100.0% | 99.0% | 100.0% |
| PAN | 197 | 100.0% | 87.3% | 93.2% | 85.1% |
| PHONE | 397 | 96.9% | 93.7% | 95.3% | 94.2% |
| UPI | 92 | 100.0% | 47.8% | 64.7% | 56.6% |
| EMAIL | 168 | 100.0% | 100.0% | 100.0% | 100.0% |
| CARD | 72 | 100.0% | 100.0% | 100.0% | 100.0% |
| IFSC | 72 | 100.0% | 100.0% | 100.0% | 100.0% |
| BANK_ACCOUNT | 94 | 100.0% | 97.9% | 98.9% | 100.0% |
| SECRET | 248 | 100.0% | 90.3% | 94.9% | 96.7% |
| NAME | 364 | 97.8% | 87.4% | 92.3% | 88.2% |
| ADDRESS † | 271 | 99.6% | 88.9% | 94.0% | 93.7% |

Negative messages with at least one false alarm: 5 of 350.

## Test split: Presidio, out of the box

| Type | Support | Precision | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|
| AADHAAR | 95 | 0.0% | 0.0% | 0.0% | 30.9% |
| PAN | 197 | 0.0% | 0.0% | 0.0% | 3.4% |
| PHONE | 397 | 80.4% | 75.3% | 77.8% | 100.0% |
| UPI | 92 | not supported | not supported | not supported | 32.9% |
| EMAIL | 168 | 88.4% | 100.0% | 93.9% | 100.0% |
| CARD | 72 | 91.0% | 98.6% | 94.7% | 100.0% |
| IFSC | 72 | not supported | not supported | not supported | 4.8% |
| BANK_ACCOUNT | 94 | not supported | not supported | not supported | 87.1% |
| SECRET | 248 | not supported | not supported | not supported | 24.6% |
| NAME | 364 | 40.2% | 83.0% | 54.2% | 83.0% |
| ADDRESS † | 271 | 43.4% | 55.0% | 48.5% | 39.9% |

Negative messages with at least one false alarm: 115 of 350.

## Test split: Presidio, best shot

| Type | Support | Precision | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|
| AADHAAR | 95 | 87.0% | 98.9% | 92.6% | 100.0% |
| PAN | 197 | 100.0% | 84.8% | 91.8% | 85.1% |
| PHONE | 397 | 80.6% | 75.3% | 77.9% | 100.0% |
| UPI | 92 | not supported | not supported | not supported | 32.9% |
| EMAIL | 168 | 88.4% | 100.0% | 93.9% | 100.0% |
| CARD | 72 | 91.0% | 98.6% | 94.7% | 100.0% |
| IFSC | 72 | not supported | not supported | not supported | 4.8% |
| BANK_ACCOUNT | 94 | not supported | not supported | not supported | 87.1% |
| SECRET | 248 | not supported | not supported | not supported | 24.6% |
| NAME | 364 | 40.3% | 83.0% | 54.2% | 83.0% |
| ADDRESS † | 271 | 43.4% | 55.0% | 48.5% | 39.9% |

Negative messages with at least one false alarm: 116 of 350.

## Hand-written set: Ours

| Type | Support | Precision | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|
| AADHAAR | 5 | 100.0% | 100.0% | 100.0% | 100.0% |
| PAN | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| PHONE | 13 | 100.0% | 92.3% | 96.0% | 72.2% |
| UPI | 5 | 100.0% | 80.0% | 88.9% | 78.6% |
| EMAIL | 3 | 100.0% | 100.0% | 100.0% | 100.0% |
| CARD | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| IFSC | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| BANK_ACCOUNT | 5 | 100.0% | 80.0% | 88.9% | 77.3% |
| SECRET | 7 | 100.0% | 100.0% | 100.0% | 100.0% |
| NAME | 22 | 95.0% | 86.4% | 90.5% | 90.5% |
| ADDRESS † | 28 | 100.0% | 92.9% | 96.3% | 94.8% |

Negative messages with at least one false alarm: 0 of 23.

## Hand-written set: Presidio, out of the box

| Type | Support | Precision | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|
| AADHAAR | 5 | 0.0% | 0.0% | 0.0% | 38.2% |
| PAN | 4 | 0.0% | 0.0% | 0.0% | 0.0% |
| PHONE | 13 | 83.3% | 76.9% | 80.0% | 85.8% |
| UPI | 5 | not supported | not supported | not supported | 25.7% |
| EMAIL | 3 | 75.0% | 100.0% | 85.7% | 100.0% |
| CARD | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| IFSC | 4 | not supported | not supported | not supported | 25.0% |
| BANK_ACCOUNT | 5 | not supported | not supported | not supported | 100.0% |
| SECRET | 7 | not supported | not supported | not supported | 21.2% |
| NAME | 22 | 43.2% | 72.7% | 54.2% | 67.6% |
| ADDRESS † | 28 | 47.4% | 64.3% | 54.5% | 41.1% |

Negative messages with at least one false alarm: 8 of 23.

## Hand-written set: Presidio, best shot

| Type | Support | Precision | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|
| AADHAAR | 5 | 100.0% | 100.0% | 100.0% | 100.0% |
| PAN | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| PHONE | 13 | 83.3% | 76.9% | 80.0% | 85.8% |
| UPI | 5 | not supported | not supported | not supported | 25.7% |
| EMAIL | 3 | 75.0% | 100.0% | 85.7% | 100.0% |
| CARD | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| IFSC | 4 | not supported | not supported | not supported | 25.0% |
| BANK_ACCOUNT | 5 | not supported | not supported | not supported | 100.0% |
| SECRET | 7 | not supported | not supported | not supported | 21.2% |
| NAME | 22 | 43.2% | 72.7% | 54.2% | 67.6% |
| ADDRESS † | 28 | 47.4% | 64.3% | 54.5% | 41.1% |

Negative messages with at least one false alarm: 8 of 23.

## Latency (detection only, test split)

Best of 3 runs per message, one CPU thread.

| Engine | p50 | p95 |
|---|---:|---:|
| Ours | 16.977 ms | 40.209 ms |
| Presidio, out of the box | 6.470 ms | 10.018 ms |
| Presidio, best shot | 6.624 ms | 9.973 ms |

## Presidio results we don't score (test split)

Presidio types outside the mapping are not counted as false alarms, but they do count for coverage. How often each fired, after conflicts were resolved:

| Presidio type | Presidio, out of the box | Presidio, best shot |
|---|---:|---:|
| `DATE_TIME` | 423 | 404 |
| `IP_ADDRESS` | 25 | 25 |
| `MEDICAL_LICENSE` | 4 | 4 |
| `NRP` | 15 | 15 |
| `UK_NHS` | 11 | 11 |
| `URL` | 58 | 58 |
| `US_BANK_NUMBER` | 201 | 171 |
| `US_DRIVER_LICENSE` | 146 | 146 |
| `US_SSN` | 2 | 2 |

Overlapping results resolved the way Presidio's anonymiser does by default (removed or merged): 1588 and 1645 (out of the box, best shot).

Mapping: `IN_AADHAAR`→AADHAAR, `IN_PAN`→PAN, `PHONE_NUMBER`→PHONE, `EMAIL_ADDRESS`→EMAIL, `CREDIT_CARD`→CARD, `PERSON`→NAME, `LOCATION`→ADDRESS.

Best-shot thresholds: `CREDIT_CARD` 0.95, `EMAIL_ADDRESS` 0.95, `IN_AADHAAR` 0.95, `IN_PAN` 0.85, `LOCATION` 0.85, `PERSON` 0.85, `PHONE_NUMBER` 0.40.

## Notes

- Presidio was run as installed (`AnalyzerEngine()`, its default spaCy model `en_core_web_lg`), with `language="en"` for every message, Hinglish included: Presidio has no Hinglish support to choose instead.
- Presidio types outside the mapping (dates, URLs, US and UK identifiers, ...) are not counted as false alarms, but they do count for coverage: an anonymiser built on Presidio would hide them too. Where one of them wins a conflict (e.g. a 10-digit mobile number labelled `UK_NHS` with a higher score than `PHONE_NUMBER`), the value is hidden but counts as a missed PHONE.
- Overlapping results are resolved before scoring the way Presidio's anonymiser does by default (`MERGE_SIMILAR_OR_CONTAINED`, then joining same-type results separated only by spaces), as read from the pinned presidio-anonymizer source. Scoring raw overlaps would count each duplicate as a false alarm.
- The best-shot thresholds were chosen on dev only. The test comparison was run after they were fixed, and changed nothing.
- The test comparison was run a second time after the final code review, with no setting changed, for two reasons: to record which types each false alarm had (so false alarms can be counted on the types both engines support, above), and with Python's hash seed fixed (`PYTHONHASHSEED=0`). Presidio returns some equal-score results in an order that depends on that seed, so the tie-break between them, and a few counts, moved between runs. `presidio-run` now fixes the seed itself. The dev tuning reproduced exactly and was not re-run.
- The data, its labels and our detectors were written by the same author, so our engine has a home advantage: the message styles are the ones it was built for. Presidio was not built for Indian identifiers, UPI or Hinglish; this page measures how much that matters for this use case, not Presidio's quality in general.
