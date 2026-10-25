# Phase 7a Walkthrough — Presidio comparison, speed and cost, hardening

> **What Phase 7a builds:** a measured answer to the four questions an interviewer asks about a tool like this. *Why not just use Presidio?* (a side-by-side on the same data, Presidio set up two ways). *Will it slow my app down, and what does it cost to run?* (overhead, throughput, memory, LLM tokens). *Is the guard itself secure?* (a threat model where every line points to a test). Plus two hardening fixes: an optional guard key and a localhost-only port.

Design for this phase: [2026-10-25-phase-7-design.md](../superpowers/specs/2026-10-25-phase-7-design.md) (7a is §4–§7). Plan: [2026-10-25-phase-7a-measure-and-harden.md](../superpowers/plans/2026-10-25-phase-7a-measure-and-harden.md). Phase 7b adds the demo video and the final README pass.

---

## 0. Why this phase exists

Phases 1–6 built a guard and measured how well it detects and restores. But numbers about *only your own tool* invite two easy dismissals: "a free library already does this" and "fine in a notebook, too slow for production". This phase answers both with measurements, and fixes the two security gaps a reviewer would spot first.

The rule throughout: **every number in the README comes from a committed results file**, and every results file records how it was made (versions, commit, hardware).

---

## 1. The guard key — `auth.py`, `config.py`, `app.py`

**The threat.** The guard passes the caller's own provider key through, so it holds no keys. That sounds safe, until someone points it at a *keyless* upstream (a local Ollama) and exposes it on a network: then anyone who can reach the guard gets free use of that model. That's an **open relay**.

**The fix** is a shared secret, off by default:

```python
# auth.py
def guard_key_ok(given: str | None, expected: str) -> bool:
    if given is None:
        return False
    return hmac.compare_digest(given.encode("utf-8", "surrogateescape"),
                               expected.encode("utf-8"))
```

Four decisions, each a likely interview question:

1. **Constant-time comparison (`hmac.compare_digest`).** A normal `==` stops at the first wrong character, so a wrong key with a correct first character takes slightly longer to reject. Measure that over many tries and you can guess the key one character at a time (a *timing attack*). `compare_digest` takes the same time however many characters match.
2. **Bytes, not str.** `compare_digest` raises `TypeError` on non-ASCII `str`. A header can hold any byte, so `X-Leakguard-Key: ключ` would have crashed the guard with a 500. Comparing bytes makes it a plain 401 (`test_guard_key_non_ascii_header_is_401`).
3. **A middleware, not a check inside the route.** It runs *before routing and before the body is read*, for every `/v1/` path, including ones that don't exist and ones added later. So without the key you can't even make the guard parse your JSON or count your bytes: an oversized body gets 401, not 413 (`test_guard_key_checked_before_body`).
4. **A short key stops the guard at startup** (`Settings.__post_init__`, minimum 16 characters). Same philosophy as `LEAKGUARD_NER`: a misconfiguration must fail loudly, not quietly weaken protection.

The key is never forwarded upstream (only `Authorization` ever is) and never logged: a failure logs `{"event": "auth_failed", "request_id": ..., "path": ...}` and nothing else. `/healthz` stays open so health checks work.

---

## 2. Localhost-only by default — `docker-compose.yml`

`ports: ["8080:8080"]` looks harmless, but Docker binds it on **every network interface**: on café Wi-Fi, other machines could reach the guard. Now it's `"127.0.0.1:8080:8080"`, pinned by `tests/test_compose.py`. The README explains how to expose it on purpose, and to set the key when you do.

(The Dockerfile still says `--host 0.0.0.0`: that's *inside* the container, where it has to listen on the container's network interface for the port mapping to reach it.)

---

## 3. The Presidio baseline

### 3.1 Two setups, so the comparison can't be called rigged

| Setup | What it is |
|---|---|
| **Out of the box** | `AnalyzerEngine()` exactly as installed. What a developer gets after `pip install`. |
| **Best shot** | Plus all six Indian recognizers turned on, and a minimum score per type **tuned on dev**, the same way our thresholds were tuned. Only Presidio's own settings: no custom recognizers (that would be our engine wearing Presidio's clothes). |

**Finding:** in presidio-analyzer 2.2.364, Presidio's Indian recognizers (Aadhaar, PAN, passport, voter ID, vehicle registration, GSTIN) are **all off by default**. Out of the box, Presidio can't find a single Aadhaar or PAN. `indian_defaults()` checks this against the installed version and writes it into the results, so it's a measured fact, not a claim.

### 3.2 The pure core and the thin runner — `presidio_baseline.py`, `presidio_run.py`

Presidio plus its spaCy model is ~600 MB, too heavy for CI. So the code is split:

- `presidio_baseline.py` has **no Presidio import**. It holds the mapping, thresholds, conflict resolution, the adapter to our `Span`s, and the tuning. A Presidio result is our own little `Found(entity, start, end, score)` dataclass, so CI tests everything here with a fake analyser.
- `presidio_run.py` imports Presidio *inside functions* and converts its results to `Found`. It's the only file that needs the `[baseline]` extra.

### 3.3 Fairness rules, in code

- **Mapping** (`MAPPING`): `IN_AADHAAR→AADHAAR`, `PERSON→NAME`, `LOCATION→ADDRESS`, and so on. Types Presidio lacks (UPI, IFSC, bank account, secrets) say *not supported* in the tables, never 0%.
- **Presidio's other types** (dates, URLs, US identifiers) aren't counted as false alarms. **But they do count for coverage**, because an anonymiser built on Presidio would hide them too. That needed one change to our scorer: `score(samples, detect, cover=...)` now accepts extra ranges that count for coverage only (`metrics.py`). This is also why Presidio's bank-account coverage is 87%: its *US* bank-number detector happens to fire on Indian account numbers that have "account" nearby.
- **Overlaps are resolved the way Presidio's anonymiser does** (`resolve_conflicts`). Presidio's analyser often returns the same digits twice: `PHONE_NUMBER` *and* `UK_NHS` on `9876543210`. Scoring both would count one as a false alarm and unfairly sink Presidio's precision. So we apply its anonymiser's default strategy, read from the pinned source: merge overlapping results of the same type, drop results contained in another (identical spans keep the higher score), then join same-type results separated only by spaces. One detail found by reading the source: that last join keeps the *second* result's score, not the higher one, and our code matches.
- **Tuning rule, fixed before the run** (`tune_thresholds`): for each type, the threshold from 0.00 to 0.95 with the best dev F1; ties go to the higher threshold. The test split was then run **once**.

### 3.4 Two practical snags

- **spaCy crashed on Windows** with `OverflowError: Python int too large to convert to C ssize_t` while loading its model. A blank spaCy pipeline worked, so the model's tokenizer data was the trigger, inside `preshed` (spaCy's hash-table library). Hypothesis: the newest `preshed` (3.0.14) broke it. Installing 3.0.13 fixed it, so it's pinned in `[baseline]` with a comment.
- **"One CPU thread" had to be enforced.** spaCy's maths goes through numpy, which uses every core unless told otherwise, and it reads that setting only when numpy is first imported. So `leakguard/evaluation/__init__.py` sets `OMP_NUM_THREADS=1` (and the BLAS equivalents) before anything imports numpy.

### 3.5 What it found (test split)

| | Ours | Presidio, out of the box | Presidio, best shot |
|---|---:|---:|---:|
| F1, types both support | 95.3% | 62.6% | 74.5% |
| Coverage, types both support | 93.9% | 79.6% | 94.3% |
| Clean messages with a false alarm (any type) | 5 / 350 | 115 / 350 | 116 / 350 |
| ... on the six types both support | 4 / 350 | 67 / 350 | 68 / 350 |
| Detection p50 | 17.0 ms | 6.5 ms | 6.6 ms |

How to read it honestly:
- **Presidio's best shot hides as much as ours does** on the types both support (94.3% vs 93.9% coverage). It gets there by **over-masking**: on those six types, 68 of 350 clean messages get a false alarm (ours: 4); counting its place names, 116 (ours: 5), and only 40% of what it calls a name is a name. Over-masking hides words the LLM needs, so answers get worse, and every false alarm is a placeholder the reply has to restore.
- **Presidio is faster** (6.6 ms vs 17.0 ms), because our name model costs ~17 ms. Say so; don't hide it.
- **The home advantage:** the test data was written by the same author as our detectors, so the message styles are the ones ours was built for. The page says this too. What it measures is how well each tool fits *this* use case (Indian identifiers, UPI, Hinglish), not Presidio's quality in general.

---

## 4. Speed and running cost — `perf.py`, `perf_run.py`, `perf_report.py`

### 4.1 Why two different setups

- **Overhead and time to first token: plain processes on loopback.** On Windows, Docker Desktop forwards ports through a VM, which can cost more than a millisecond per hop. The rules take 0.175 ms. Measured through Docker, we'd be measuring Docker. So client, guard and mock run as normal processes, and overhead = (time through the guard) − (time straight to the mock), per message, best of 3 each.
- **Throughput, memory, CPU: the real container, capped with `--cpus`.** The guard runs detection in a thread pool, and ONNX Runtime releases Python's GIL, so one "single-threaded" worker can quietly use several cores. Capping the container to 1 and then 2 CPUs makes "fits a 1 vCPU machine" a measurement.

### 4.2 How the numbers are read

- **Memory** comes from the container's cgroup files (`/sys/fs/cgroup/memory.current`, `memory.peak`), read with `docker exec`. This is the kernel's own accounting, the same number that triggers an out-of-memory kill.
- **CPU per message** = change in `cpu.stat`'s `usage_usec` during a load run ÷ successful requests.
- **Errors are never hidden:** `load()` counts every non-200 or timeout, shows it next to the throughput, and leaves it out of messages per second (`test_load_counts_errors`).
- **Machine size** (`machine_size`) is a stated rule: the smallest of 0.5 / 1 / 2 / 4 GB with 25% headroom over the measured peak.
- **Image size has two answers.** With Docker's containerd image store, `docker image inspect` reports the *compressed* size (what a pull downloads) and `docker image ls` the *unpacked* size on disk. The smoke run caught the difference (353 MB vs 1.03 GB), so the report shows both.

The perf stack (`docker-compose.perf.yml`) publishes **no ports**: the load client reaches the guard over the stack's own Docker network, so it can't clash with a guard already running on 8080.

### 4.3 Results

On the author's laptop (i7-1165G7), from [performance.md](../evaluation/performance.md):

| | Name model on | Name model off |
|---|---:|---:|
| Added per request, p50 / p95 | 19.2 / 30.4 ms | 1.7 / 3.1 ms |
| Added before the first streamed token, p50 / p95 | 19.7 / 31.2 ms | 1.5 / 2.2 ms |

| CPUs | 1 client | 8 clients |
|---:|---:|---:|
| 1 | 42.9 msg/s | 24.5 msg/s |
| 2 | 41.8 msg/s | 56.5 msg/s |

Memory: 694 MB idle, 734 MB peak, so it **fits a 1 vCPU / 1 GB machine**. Zero errors in every run.

The surprise is the 1-CPU, 8-client cell: throughput *falls*. Eight requests' detection threads share one CPU's quota, so each one waits, gets throttled and switches, and CPU time per message nearly doubles (21.8 → 41.0 ms). With 2 CPUs the same load reaches 56.5 msg/s. The lesson for deployment: size CPUs or replicas to the concurrency you expect, rather than piling requests onto one core.

The name model is ~90% of the overhead (19 ms on, 1.7 ms off). Next to a real LLM, which takes hundreds of milliseconds to seconds, it's small.

---

## 5. LLM token cost — `token_cost.py`

LLM providers bill per token. Swapping `2345 6789 0124` for `[AADHAAR_1]` changes the token count, in either direction. `token_cost()` masks each test message exactly as the guard would, then counts tokens before and after with tiktoken's `o200k_base` (the encoding OpenAI's current models and gpt-oss use). Blocked messages (secrets) are never sent, so they're left out and counted. The per-type table compares each value with its placeholder on their own, which the page labels as an isolated comparison, since in a sentence tokens can merge across the boundary.

**Result: masking saves tokens.** Over 1,376 test messages the guard would forward, prompts went from 31,139 to 26,119 tokens (**−16.1%**). Most of the saving is addresses (17.7 tokens on average become a 5-token `[ADDRESS_1]`). A few types cost slightly more (a bank account is 4.7 tokens and its placeholder 6.0), but they're outweighed.

---

## 6. The threat model — README

The README's *Security of the guard itself* table has four columns: **threat · who · what the guard does · proof**, and every proof is a test name or a file that exists. It covers stolen mappings, logs, placeholder injection, ReDoS, open relay, detection failure, oversized requests and the guard key itself. Then comes *What it does not protect against*: images and files, unmeasured scripts, identity inferred from context, the known-hard formats (with their numbers), traffic that bypasses the guard, and what the provider still sees.

That second list is what makes a threat model credible. A tool that claims to stop everything stops nothing you can check.

---

## 7. The final review: a key that could silently be off

A fresh reviewer read the whole branch and probed it with real Presidio and a real uvicorn server. It confirmed the measurements: our conflict resolution matched Presidio's anonymiser on all 4,586 real outputs, the tuning reproduced, and every README number matched the JSON. It also found four real problems, all fixed with a test that failed first:

1. **The guard key never reached the container (critical).** `docker-compose.yml` only passes environment variables that the service lists, and it didn't list `LEAKGUARD_API_KEY`. So `LEAKGUARD_API_KEY=... docker compose up` started a guard **with no key**, and clients sending the header still worked, so nobody would notice. The one setup the key exists for was the one where it was off. Fixed, and `test_guard_key_reaches_the_container` pins it.
2. **A path prefix switched the key off.** Behind a reverse proxy that serves the guard under `/guard`, uvicorn's `--root-path` puts the prefix *in* the path (`/guard/v1/...`). Routing strips it, but the middleware checked `startswith("/v1/")`, so it waved the request through. Now the middleware strips the prefix too. (The first version of the test passed without the fix, because TestClient keeps the prefix out of the path; it was rewritten to send the path the way uvicorn does.)
3. **"Huge requests are rejected" wasn't fully true.** A body sent in chunks has no `Content-Length`, and it was read whole before the size check. So one request could fill memory first. Now the guard counts while reading and stops at the limit. TestClient also reads whole bodies, so this test drives the raw ASGI app and counts how many chunks the guard pulled: 100 before, at most 11 after.
4. **A Presidio sentence was scoped wrong.** "116 of 350 clean messages" counted Presidio's place names, but the sentence said "on the six shared types". The page now records which types each false alarm had and shows both counts: **68** on the shared types, **116** in total. Getting that needed one more test run. It was made with Python's hash seed fixed, because the reviewer found that Presidio's order for equal-score results depends on it, which moved a few counts between runs. `presidio-run` now fixes the seed itself, and the page says the test was run twice and why.

The lesson from 1 and 2: a security setting that can be **silently** off is worse than one that's visibly off. Both now have tests that check the setting from the outside, the way a user would actually set it.

---

## 8. Interview Q&A

**Q: Why not just use Presidio?**
A: I measured it. On the same Indian test set, Presidio's Indian recognizers are off by default, so out of the box it finds no Aadhaar or PAN at all. With them turned on and its thresholds tuned on dev, it hides about as much as mine (94% coverage) but on those types it flags 68 of 350 clean messages; mine flags 4. It has no UPI, IFSC or bank-account detection, and it's a library, not a drop-in proxy with streaming restore. It's faster than mine, though: about 7 ms against 17 ms, because of my name model.

**Q: How do I know you didn't set Presidio up to lose?**
A: Two setups: exactly as installed, and its best shot with its own settings tuned on dev by a rule fixed before the run. Duplicates were resolved the way Presidio's own anonymiser does, so I didn't count its duplicates against it, and its detections of types I don't track count towards its coverage. Everything's in the committed JSON and regenerates with one command.

**Q: Why constant-time comparison for an API key?**
A: A normal string compare returns as soon as a character differs, so response time leaks how much of the key was right. With enough requests, you can recover it character by character. `hmac.compare_digest` takes the same time either way. And I compare bytes, because it rejects non-ASCII strings, which would otherwise have been a 500 on a weird header.

**Q: Why is the key check a middleware?**
A: It runs before routing and before the body is read, so an unauthenticated caller can't make the guard parse anything, and it covers `/v1/` paths that don't exist yet.

**Q: How did you measure overhead without Docker skewing it?**
A: Docker Desktop's port forwarding on Windows costs more than my rule detection does, so overhead is measured with plain processes on loopback: time through the guard minus time straight to the same mock, per message, best of three. Throughput and memory use the real container, CPU-capped, because that's what you'd deploy.

**Q: Does masking make my LLM bill bigger?**
A: No, it makes it slightly smaller. I counted it with the tokenizer OpenAI's models use: masking cut prompt tokens by 16% on my test set, mostly because a long address becomes `[ADDRESS_1]`. Bank-account placeholders are a token longer than the number, but that's outweighed.

**Q: What's something you found while measuring that you didn't expect?**
A: Three things. Presidio's Indian recognizers being off by default. Presidio labelling some Indian mobile numbers (11 in the test set) as UK NHS numbers with a higher score than "phone". And `docker image inspect` reporting the compressed image size under the containerd store, about a third of what's on disk.
