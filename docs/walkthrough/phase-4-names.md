# Phase 4 Walkthrough — Person names (NER)

> **What Phase 4 builds:** detection of people's names with a pretrained machine-learning model, chosen by measurement from five candidates, converted to ONNX, compressed to int8, run on a plain CPU, and wired into the guard and the evaluation report.

---

## 0. Why names need a model

Aadhaar numbers, PANs and phone numbers have fixed shapes, so rules plus checksums catch them in microseconds. Names don't: "Rahul Sharma", "K. Venkatesh", "Mr. Iyer" and "mera naam rahul hai" share no pattern, and a capitalised word may just as well be a festival ("Ganesh Chaturthi") or a brand ("Paytm"). Deciding which words are a person's name means reading the context. That's what **NER** (Named Entity Recognition) models are trained to do: label each word of a sentence as a person, a place, an organisation, or nothing.

The rule used throughout the project: **use the cheapest tool that works.** Rules handle everything with a pattern; the model handles only names, the one type without one.

---

## 1. Better name data first

### Files
| File | Change |
|---|---|
| `src/leakguard/evaluation/values.py` | `NAME_POOLS` (separate dev/test names), `name_makers(split)`, new fillers |
| `src/leakguard/evaluation/templates.py` | 22 new templates: name forms + non-name negatives |
| `src/leakguard/evaluation/generate.py` | names come from the split's own pool; `name_form()` |
| `data/eval/handwritten.jsonl` | 16 new hand-written messages |

### What was wrong
Phase 3's names were all capitalised "First Last" from one 54-word list shared by dev and test. A model would look great on that and then fail on real chats. Two fixes:

1. **Name forms** that real messages contain, each a template slot:

| Slot | Example | Why it's hard |
|---|---|---|
| `{NAME}` | Rahul Sharma / Rahul | the easy case |
| `{NAME.first}` | Priya | one word, less context |
| `{NAME.surname}` after `Mr.`/`Shri`… | Mr. Iyer | gold covers only "Iyer" |
| `{NAME.initial}` | K. Venkatesh, R. K. Narayan | South Indian style; dots split tokens |
| `{NAME.three}` | Mohammed Imran Khan | longer span |
| `{NAME.caps}` | RAHUL SHARMA | forms in capitals |
| `{NAME.lower}` | mera naam rahul hai | no capital letter to lean on |
| `{NAME.wordlike}` | Kiran, Sunny, Rose | names that are also English words |

2. **Separate name pools.** Dev and test draw names from different lists (no name, first or last, appears in both). The test numbers then measure names nobody tuned against. `test_test_split_names_never_appear_in_dev_pool` checks it on every generated test sample.

And new **negatives**: festivals, brands, cities, products and "in May": capitalised words that must *not* be masked.

### Q&A
- **Why not let the test data reuse dev's names? The model wasn't trained on either.** The model wasn't, but the *threshold* was tuned on dev. With shared names, a threshold could quietly fit those particular names.
- **Why did the other types' numbers move slightly?** The generator is seeded: drawing names differently changes the random sequence, so the whole dataset is regenerated. The report states this in its methodology notes.

---

## 2. From model output to names — `ner/decode.py`

### Tokens, offsets and BIO tags
A model doesn't see words. It sees **tokens**, often pieces of words:
```
"I am Venkatesh"  ->  ["I", "am", "Venk", "##atesh"]
```
and for each token the tokenizer reports its **character offsets** in the original text, e.g. `Venk` = 5..9. The model gives every token a label in **BIO** form:
- `B-PER`: begins a person's name
- `I-PER`: inside (continues) a name
- `O`: outside, not a name

`decode_names()` turns these into `Span(start, end, NAME, score)`:
1. **Group** consecutive person tokens. Two tokens of the same word always group. Across a gap, an `I-PER` continues the name only if the gap is a space, a dot ("R. K."), an apostrophe or a hyphen, never a newline or a real word. A `B-PER` after a name starts a new name ("Rahul Priya" is two names).
2. **Widen to whole words.** If the model tags `Venk` but not `##atesh`, masking only `Venk` would send "atesh" to the LLM. Every span is widened to word boundaries (`test_span_expands_to_whole_word`).
3. **Score** a name by the mean of its tokens' person probabilities; drop it below the threshold.

Some models don't use BIO (one candidate just says `person`). `person_label_ids()` handles both.

---

## 3. Running the model — `ner/model.py`

### Why ONNX, and what's in the model directory
The models are published as PyTorch weights. PyTorch is a ~200 MB+ install built for training. **ONNX** is a portable format for a trained network, and **ONNX Runtime** is a small, fast engine that runs it on CPU. So PyTorch is used once, offline, to export the model, and the guard needs only `onnxruntime` + `tokenizers`.

A model directory holds `model.onnx`, `tokenizer.json`, `ner_config.json` (labels, special-token ids, max length, threshold) and `manifest.json` (where it came from, file hashes, parity results). `NameModel.load()` checks all of this at startup.

### Long messages: overlapping windows
These models read at most **512 tokens** at once. A pasted chat log can be longer, and a name at token 900 must still be found. So:
- the text is tokenised **without** truncation (many published `tokenizer.json` files switch truncation on at 512, which would silently stop reading; `NameModel` switches it off),
- tokens are cut into windows of 510 (+ the two special tokens), **overlapping by 64**, so a name on an edge appears whole in at least one window,
- each token takes its prediction from the window where it sits **furthest from an edge**, where the model sees the most context around it,
- all windows of one message run as **one batch**: one call into ONNX Runtime, not one per window.

Short messages are padded only to their own length, not to 512, which is most of why inference takes milliseconds.

Tests use a real `tokenizers` tokenizer with a tiny vocabulary and a fake session that marks "rahul" as `B-PER`, so windowing is tested exactly without a 100 MB model (`test_name_beyond_first_window_is_found`, `test_name_on_window_boundary_found_once`, `test_all_windows_run_in_one_batch`).

---

## 4. Exporting — `ner/export.py`, `ner/candidates.py`

### Pinned candidates
`candidates.py` lists five models, each pinned to a Hugging Face **commit sha**, not "main". The author could update a model tomorrow; a sha always means the same weights, so the measured model is exactly the shipped one.

`ai4bharat/IndicNER` was left out: it's **gated** (download needs a logged-in account that accepted its terms), which breaks "a stranger builds it with one command".

### What one export does
```
download pinned revision -> PyTorch model
      -> torch.onnx.export (dynamo)            model.fp32.onnx
      -> int8 dynamic quantisation             model.onnx
      -> parity checks against PyTorch         manifest.json
```
**Quantisation:** weights are stored as 8-bit integers instead of 32-bit floats: about **4× smaller** and faster on CPU, at a possible small accuracy cost, which §5 measures.

### Two bugs the parity check caught
The export refuses to write a model unless (a) the fp32 ONNX logits match PyTorch within 0.001 and (b) our runtime tokenisation gives exactly Hugging Face's token ids. Both checks paid off on day one:
1. **The old exporter mis-traced the model.** `torch.onnx.export(dynamo=False)` records one example run. With transformers 5.x it froze shape-dependent attention code, and the logits on real sentences were off by up to **3.4**. The model would have loaded fine and quietly given wrong answers. The newer `dynamo=True` exporter matches PyTorch to ~0.00001.
2. **My own bug:** the example input passed *the same tensor object* as both `input_ids` and `attention_mask`, so the exporter treated them as one input. The mask became the token ids. Logits were off by up to 9. Fixed with two separate tensors (commented in the code).

> Interview point: "the model loads and returns labels" proves nothing. Compare the converted model's outputs against the original on fixed inputs, and fail the build if they differ.

---

## 5. Choosing the model — `ner/compare.py`

Full results: [docs/evaluation/ner-comparison.md](../evaluation/ner-comparison.md). **Dev split only**; the command has no option to run on test.

### The rule, fixed before running
Highest NAME F1 on dev **as deployed** (rules + model, merged); if another candidate is within 1.0 F1 point, the one with the lower p95 latency. Writing the rule first stops "pick whichever looks best afterwards".

### What was tuned per model on dev
- **Threshold** (0.3 … 0.9): each model's scores mean different things.
- **int8 mode** (one scale per tensor, or per channel): see below.

Each model runs once per message at threshold 0, and the sweep filters the cached spans. Grouping never depends on the score, so this equals re-running seven times, at one seventh of the cost.

### The results

| Candidate | F1 as deployed | F1 model alone | p95 | Size |
|---|---:|---:|---:|---:|
| **xlmr-hrl** (multilingual XLM-R) | **98.6%** | **89.3%** | 40 ms | 280 MB |
| bert-base | 94.0% | 80.7% | 47 ms | 110 MB |
| muril-naamapadam (India-aware) | 93.9% | 85.2% | 42 ms | 239 MB |
| distilbert | 93.6% | 81.1% | 23 ms | 66 MB |
| xlmr-indian ("Indian context") | 78.7% | 72.3% | 39 ms | 280 MB |

**Selected: `xlmr-hrl`**, int8 per-channel, threshold 0.5: clearly ahead, with the fewest false alarms on the clean messages (3). Two lessons:
- The model described as trained "for Indian context" came **last** (precision 65%: it tags too much). The model card's description is a claim; the dev split is the evidence.
- The multilingual model wins on Indian names, likely because its pretraining saw far more varied names than English news text.

### Quantisation: a story worth telling in an interview
The first comparison quantised every model with **one scale per weight matrix**. MuRIL scored **93% fp32 and 0% int8**: a few large weights set the scale, and everything else rounded to zero. A probe of the alternatives on dev:

| int8 setting | MuRIL F1 | xlmr-hrl F1 |
|---|---:|---:|
| one scale per tensor | 0.0% | 89.0% |
| **one scale per channel** | **85.2%** | **89.3%** |

(model alone, no cue fallback)

So I switched everything to per-channel. That was **wrong for distilbert** (81.1% → 78.9%). A 2×2 check (quantisation × cues) proved it was the quantisation, not something else I'd changed. The fix: treat the int8 mode as a **hyperparameter chosen per model on dev**, like the threshold. Each candidate export now writes both int8 files; the comparison keeps the better one.

What int8 bought the winner: **1,112 MB → 280 MB, p50 43 ms → 17 ms**, with F1 98.1% → 98.6% as deployed (within noise).

### Lowercase and ALL-CAPS names — `detect/name_cues.py`
The models lean on capital letters. Alone, xlmr-hrl finds **14%** of lowercase names ("mera naam rahul hai") and **44%** of all-caps ones ("NAME: RAHUL SHARMA"). The plan set a trigger: below 80% on lowercase, build a fallback.

The fallback is a **rule, not a model, and uses no name list**. It looks for phrases that announce a name:
- `mera naam ___ hai`, `my name is ___`, `main ___ hoon`
- `___ bol raha hoon`
- record fields: `Name: ___`, `Cust name - ___`, `APPLICANT NAME - ___`

The slot must be 1–3 plain words, none of them a common word, so "main theek hoon" ("I'm fine") and "my name is wrong on the invoice" are not names (both tested).

Why no name list: I'd have had to write it myself, and I also wrote the test names, so the list would be contaminated by what I know. A licensed public list wasn't at hand. On dev: **141 names found, 0 false alarms**. The report discloses that the cue phrases and the test templates have the same author.

---

## 6. Wiring it into the guard

### Files
| File | Change |
|---|---|
| `detect/engine.py` | `detect_all` → `detect_rules`; `build_detector(name_model)` adds names |
| `config.py` | `LEAKGUARD_NER` (on/off), `LEAKGUARD_NER_MODEL_DIR`, `LEAKGUARD_NER_THREADS` |
| `app.py` | loads the model at startup; detection runs in a worker thread |

### Merging
Names are merged with the rule detections by the existing priority list, where `NAME` is lowest. If the model tags "rahul" inside `rahul.sharma@gmail.com`, the EMAIL wins and the address stays whole.

### Fail closed, at startup
If name detection is on and the model is missing or broken, `create_app` raises and the guard **refuses to start**, with the fix in the message ("run the export, or set `LEAKGUARD_NER=off`"). The alternative, starting without names, would leak every name while looking healthy. `LEAKGUARD_NER=off` is allowed, but logs a `ner_disabled` warning. A typo like `LEAKGUARD_NER=of` is an error, not "off": a typo must never switch a protection off.

### Not blocking the event loop
FastAPI serves many requests on one event loop. Inference is CPU work taking milliseconds, and run directly in the async handler it would freeze *every other* request meanwhile. Detection now runs via `run_in_threadpool`. `test_slow_detection_does_not_block_other_requests` makes detection take 0.5 s and checks `/healthz` still answers fast. Its first version passed even with the blocking code: the block happened during the test's own `sleep`, before the timer started. The fixed test was seen failing (0.51 s) without the thread pool and passing with it.

---
## 7. Results on the test split

From [docs/evaluation/report.md](../evaluation/report.md): names nobody tuned against, phrased in ways dev never showed.

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| **NAME (test)** | **96.0%** | **89.0%** | **92.4%** |
| NAME (hand-written) | 94.4% | 85.0% | 89.5% |
| All built types (test) | 98.6% | 93.5% | 96.0% |

Names by form (test recall): caps 100% · initials 100% · three-part 100% · surname after Mr./Shri 97% · mixed 97% · **first name alone 78%** · **wordlike 63%** · **lowercase 50%**.

**Latency:** rules alone 0.05 ms per message; with the name model **p50 17 ms, p95 42 ms** on one CPU thread. The model is now ~99.7% of detection time, and still small next to an LLM call (typically 500 ms or more).

### What it gets wrong (and why)
- **Lowercase without a cue: 50%.** One test template has a cue ("___ bol raha hoon": found), the other doesn't ("bhaiya ___ ko parcel de dena": missed). The fallback only knows announced names. Expected, and stated.
- **First name alone: 78%.** "Riya here, can you check…" gives the model little context.
- **Names that are words: 63%.** "Sunny here" vs "it's sunny here": genuinely ambiguous.
- **Clean messages flagged: 9 of 350 (2.6%)**, up from 1 in Phase 3. Some capitalised non-names (festivals, products) are tagged as people. A masked "Diwali" costs the chatbot a little meaning but leaks nothing. The trade-off falls on the safe side.
- **"SHAIKH MOHAMMED ASLAM" → only "ASLAM" flagged** (hand-written): an all-caps name without a cue; the model is unsure about capitals.

### Known consequence from the spec, now real
§6.2 warned that context-sensitive detection might catch a name in one turn and miss it in the next (when the restored reply comes back in the history). Multi-turn consistency gets measured in Phase 6, against a real LLM.

---

## 8. Docker and CI

- **Multi-stage Dockerfile.** Stage 1 installs CPU-only PyTorch and runs the export. Stage 2, the image that ships, copies only the exported files. PyTorch never enters the final image (checked: `import torch` fails inside it).
- **Same model on Linux?** The image exports the model inside Linux, and the file came out **not byte-identical** to the Windows export that was measured (tiny floating-point differences between the platforms' PyTorch builds). Rather than assume, I ran both over every dev, test and hand-written message: **0 of 3,059 messages** got different name spans, so the reported numbers hold for the image.
- **End to end in Docker:** `docker compose up`, then "Hi, I am Priya Nair, my number is 9876543210": the mock LLM received `Hi, I am [NAME_1], my number is [PHONE_1]`, the client got the original back, and the guard's log line held only `{"NAME:mask": 1, "PHONE:mask": 1}` and timings (detection 14 ms).
- **CI has two jobs.** `test` (fast; the model-backed tests skip) and `ner-model`. The second exports the shipped model once, caches it until `candidates.py` or `export.py` change, then runs the full suite including the real-model tests.

---

## 9. Interview Q&A

**Why not just use Presidio's or spaCy's name detection?**
Phase 7 runs Presidio on the same test set, so that answer will be a number. The design reason: those are general English pipelines; I needed Hinglish, lowercase chat text, and a model I could measure, choose, quantise and ship in a small CPU image.

**How did you choose the model?**
Five pinned candidates, one rule written down before running (best dev F1 as deployed; within 1 point, faster wins), threshold and quantisation mode tuned per model on dev. The test set was used only for the final report.

**What is quantisation, and what did it cost?**
Storing weights as 8-bit integers instead of 32-bit floats. For the winner: 4× smaller (1.1 GB → 280 MB) and 2.6× faster (p50 43 → 17 ms), with no measurable accuracy loss. But it isn't free in general: with one scale per tensor, MuRIL went from 93% to 0%. Always measure after quantising.

**How do you handle text longer than the model's 512-token limit?**
Overlapping windows (64 tokens of overlap), all in one batch. Each token takes the prediction from the window where it has the most context. Tested with a name beyond the first window and one straddling a window edge.

**What if the model file is missing in production?**
The guard refuses to start, with the fix in the error message. Starting without names would leak every name while looking healthy: fail closed.

**Doesn't a 20–40 ms model slow down every request?**
It runs in a worker thread, so other requests aren't blocked. Next to an LLM call of hundreds of milliseconds it's small. If needed: batch all messages of a request into one inference, or run more threads.

**Your lowercase fallback is just regexes. Isn't that hacky?**
It's the cheapest tool that works for a specific, measured gap: 14% → 100% on dev lowercase names, 0 false alarms on dev. And its limit is honest and measured: names with no announcing phrase are still missed (50% on test lowercase).

**What did you learn that surprised you?**
The model described as "trained for Indian context" came last. And the export step silently produced wrong logits twice: once from the old exporter, once from my own bug. Only the parity check caught it.
