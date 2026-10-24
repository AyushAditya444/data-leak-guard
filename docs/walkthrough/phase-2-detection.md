# Phase 2 Walkthrough — Detection, Policy, Masking

> **What Phase 2 builds:** the guard starts protecting data. Rule-based detectors find Aadhaar, PAN, phone, UPI, email, card, IFSC, bank account and secrets; a policy decides mask / block / log; masking swaps values for placeholders before the request leaves, and restores them in the reply.

---

## 1. The building blocks: entity types, spans, merging, checksums

### Files
| File | Purpose |
|---|---|
| `src/leakguard/entities.py` | `EntityType` (the 11 kinds of sensitive data) and `PRIORITY` |
| `src/leakguard/detect/spans.py` | `Span` and `merge_spans()` |
| `src/leakguard/detect/checksums.py` | `verhoeff_valid()` (Aadhaar) and `luhn_valid()` (cards) |

### What a span is
Every detector answers the same question: *where in this text is sensitive data, and what kind?* The answer is a list of **spans**:
```
text:  "call 9876543210 now"
        0123456789...
Span(start=5, end=15, type=PHONE)   →  text[5:15] == "9876543210"
```
`end` is exclusive, exactly like Python slicing, so `text[span.start:span.end]` is the value. Spans are `frozen` dataclasses: once created they can't be changed, which makes them safe to pass around and compare in tests.

### Why merging is needed
Different detectors can claim the same characters:
- `9876543210@ybl` → the **phone** detector sees `9876543210`, the **UPI** detector sees the whole thing.
- `a/c 9876543210` → **phone** *and* **bank account**.

`merge_spans()` keeps one winner per region:
1. **Higher priority wins** (`PRIORITY` in `entities.py`): SECRET > AADHAAR > BANK_ACCOUNT > CARD > PAN = IFSC > UPI > PHONE > EMAIL > ADDRESS > NAME.
2. On a tie, the **longer** span wins (it captured more of the value).
3. Then the **earlier** one.

It's a greedy algorithm: sort the candidates best-first, then accept each one only if it doesn't overlap anything already accepted. Spans that merely *touch* (one ends where the next starts) don't overlap.

Why is BANK_ACCOUNT above PHONE? Some Indian banks use the customer's mobile number as the account number, so "a/c 9876543210" really is an account. And SECRET is on top because secrets are *blocked*: if any detector thinks something is a key, that judgement must not be overridden.

### Checksums — the cheapest false-positive filter
A **check digit** is computed from the other digits. If someone mistypes a digit, the check fails.
- **Aadhaar → Verhoeff.** Walk the digits from the right, combining each with a running value through two lookup tables (`_D`, a multiplication table of a mathematical group, and `_P`, a permutation table). A valid number ends with running value **0**. Verhoeff catches *every* single-digit error and *every* swap of two neighbouring digits — that's why UIDAI chose it.
- **Cards → Luhn.** From the right, double every second digit (subtract 9 if it goes above 9), add everything up; valid if the total ends in 0.

**Why this matters for detection:** a regex for "12 digits" matches every order ID, invoice number and timestamp. With one check digit, a random 12-digit number passes only about **1 time in 10** — so the checksum throws away ~90% of look-alikes for free, without any ML.

`isascii()` is checked too: Python's `isdigit()` also accepts digits from other scripts (like Devanagari `४`), which `int()` would convert — we only want ASCII 0–9 here.

### Q&A
**Q: Why does a checksum help detection?**
A: Detection has to balance catching real values against false alarms. Many numbers in support chats have the same *shape* as an Aadhaar number — order IDs, tracking numbers. Aadhaar's last digit is a Verhoeff checksum of the other 11, so only ~10% of random 12-digit numbers pass. Checking it removes most look-alikes at near-zero cost, before any expensive model runs.
