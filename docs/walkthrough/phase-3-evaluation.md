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
