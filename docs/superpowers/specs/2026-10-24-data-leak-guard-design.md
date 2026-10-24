# Data Leak Guard — Design

**Date:** 2026-10-24
**Status:** Approved by the author, 2026-10-24

---

## 1. Intent

### What it is
An OpenAI-compatible reverse proxy that sits between an application and an LLM provider. Before a request leaves, it detects sensitive data — Indian personal identifiers, contact details, secrets, and person names — and either masks it with typed placeholders, blocks the request, or logs it, according to a per-type policy. When the LLM replies, it restores the original values so the application works normally.

### Who it's for
Developers at small Indian startups building LLM features (e.g. a customer-support chatbot) who can't afford enterprise DLP and need to reduce what personal data they send to third-party AI providers, with India's DPDP Act in mind. They adopt it by changing their SDK's `base_url` — no other code changes.

### Why it's being built
It's a portfolio project demonstrating LLM engineering with a security focus. It must:
- show production-style LLM engineering (proxying, streaming, evaluation);
- apply information-security practice (threat modelling, secure logging, fail-closed design);
- produce **measured, honest results** that can be reproduced and defended;
- be something the author understands line by line (see §8, who writes what).

### What success looks like
- Public GitHub repo with a clear README, a results table of real measured numbers, a ~2-minute demo video and a GIF, and a one-command local run (`docker compose up`).
- An evaluation that compares this engine against Microsoft Presidio on the same test set.
- The author can explain every design decision and every known failure.
- First version shipped in **~4–5 weeks** (extended from 3–4 by the author's choice to include full Indian address detection).

---

## 2. Goals, non-goals, anti-goals

### Goals (v1)
1. Drop-in OpenAI-compatible endpoint: `POST /v1/chat/completions`, streaming and non-streaming.
2. Detect: Aadhaar, PAN, Indian mobile numbers, UPI IDs, email addresses, payment card numbers, **bank account numbers, IFSC codes**, secrets (API keys, tokens, passwords, private keys), person names, and **full Indian postal addresses** (including PIN codes).
3. Per-type policy: `mask` (default for personal data), `block` (default for secrets), `log_only`.
4. Reversible masking with typed, numbered placeholders, restored in both non-streaming and streaming responses.
5. Logs that record counts and types only — never raw values.
6. A labelled synthetic evaluation set (English + Hinglish) and an evaluation harness reporting per-type precision, recall, F1, latency p50/p95, a list of failure cases, and a side-by-side comparison with Presidio.
7. Runs entirely at zero cost: a bundled mock LLM for tests, plus configurable free upstreams (Ollama locally, Gemini / Groq free tiers via their OpenAI-compatible endpoints).
8. One-command local run via Docker Compose; CI running tests on every push.

### Non-goals (v1)
- Prompt-injection detection (belongs to the author's separate red-teaming project — keep the two projects separate).
- Browser extension or anything aimed at non-developer employees.
- Web dashboard.
- Multi-tenant SaaS (accounts, per-customer policies, billing). v1 is **single-tenant, self-hosted**: one deployment, one policy file.
- Endpoints other than chat completions (embeddings, Responses API, Assistants, images, audio).
- Organisations, dates of birth, and place names on their own (a city mentioned outside an address, e.g. "I live in Pune", is not masked in v1).
- Indian languages beyond English and romanised Hinglish; numbers written as words.
- Claiming or implying "DPDP compliance".
- A publicly hosted live demo. Deferred to after v1 as an optional stretch (see §11); v1 must not make it harder.

### Anti-goals (failure even if it technically works)
- **Inventing or inflating numbers.** Every number in the README must come from the evaluation harness, reproducibly.
- **The guard becoming the leak.** Raw sensitive values must never be written to logs, disk, or any persistent store.
- **Tuning on the test set.** Thresholds and model choices are tuned on a dev split; the test split is only used for reported numbers.
- **A codebase the author can't explain**, or one built mostly by copying a library. (Mitigated by per-phase walkthrough docs, §8.)
- **Requiring paid services** to build, test, or demo.
- **Scope creep** that delays v1 past ~5 weeks.

---

## 3. Constraints

- **Budget:** ₹0. Paid API credit (~$5) is an optional later choice by the author for a live demo only.
- **Hardware:** author's laptop — Windows 11, i7-1165G7 (4 cores / 8 threads), 32 GB RAM, NVIDIA T500 (4 GB). The detection pipeline must run on **CPU**; GPU is not assumed in deployment.
- **Language/stack:** Python 3.11, FastAPI, Docker — the author's existing strengths.
- **Time:** ~4–5 weeks to v1.
- **Data:** no real personal data anywhere — all test data is synthetic.
- **Keep the door open for free hosting** later (CPU-only, limited memory), so the Docker image should stay reasonably small.

---

## 4. Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | Build on Presidio as the detection engine, add UPI/secrets/proxy/streaming | Fastest and battle-tested, but the core is someone else's code → a thinner engineering contribution; heavy dependency, harder to optimise latency |
| B | Own detection engine from scratch (rules + checksums + entropy + pretrained NER via ONNX) | Full ownership and control of latency; more work; no reference point for whether results are good |
| **C** ✅ | **B, plus Presidio run as a baseline on the same evaluation set** | Everything in B, plus a measured comparison that turns "why not Presidio?" into a strength. Costs ~2–3 extra days |
| — | Ask an LLM to find the PII | Rejected: slow, and if the LLM is external the data has already leaked |

**Chosen: C** (decided by the author).

---

## 5. Decided behaviour

These were decided in conversation with the author:

- **Users are developers**, integrating via `base_url`. Not end-user employees.
- **Default actions:** personal data → `mask` (hide and restore); secrets (API keys, passwords, tokens, private keys) → always `block`; any type can be set to `log_only` by the deploying company.
- **Placeholders are typed and numbered**, e.g. `[NAME_1]`, `[UPI_1]`, `[PHONE_2]`, so the LLM keeps the meaning ("a name here, the same name as before") without the value.
- **Blocked requests** return an OpenAI-shaped error response (HTTP 400) naming the detected types (never the values), so SDKs surface it cleanly.
- **Zero-cost development** with a mock LLM and free upstreams.
- **Full Indian address detection is in v1**, accepting ~1 extra week, rather than PIN codes only.
- **Demo for v1:** runs locally with one command, plus a ~2-minute demo video and a GIF in the README. No public hosted demo in v1 (see §11).
- **Work split:** see §8.

---

## 6. Design

### 6.1 Request flow

```
client ──► POST /v1/chat/completions
             │
             ├─ 1. Validate & size-limit request
             ├─ 2. Extract text from every message (system, user, assistant, tool)
             ├─ 3. Detect spans (rules → secrets → NER), merge overlaps
             ├─ 4. Apply policy per span type
             │      any `block` hit  → return 400 error, stop
             │      `log_only`       → leave text, record
             │      `mask`           → replace with placeholder
             ├─ 5. Forward masked request to configured upstream
             ├─ 6. Restore placeholders in response
             │      non-streaming: on the full message
             │      streaming:     on SSE chunks, with a hold-back buffer
             ├─ 7. Emit log record (counts/types/latency only)
             ▼
client ◄── restored response (same shape as upstream's)
```

### 6.2 Masking is request-scoped and stateless

Chat-completion requests carry the **whole conversation history** on every call. So the guard re-detects and re-masks the full history on each request, assigning placeholder numbers **deterministically in order of first appearance**. The same history produces the same placeholders every time, so numbering stays consistent across turns without the guard storing anything.

The value↔placeholder mapping lives **in memory for the lifetime of a single request** (including the full duration of a streamed response) and is discarded afterwards.

> **Change from the earlier plan:** the earlier explanation proposed storing mappings in Redis with a TTL. Request-scoped memory is simpler **and** more secure — real values are never persisted anywhere, not even briefly. Redis is therefore **not** in v1.

Known consequence to measure, not solve in v1: values the guard restored into an assistant reply come back in the next request's history and must be detected again. If detection is context-sensitive (NER), a value caught in one turn might be missed in another. The evaluation should include multi-turn cases to quantify this.

### 6.3 Detection engine

A pipeline of detectors, each returning spans `(start, end, type, score, detector)`:

**Rules (deterministic):**

| Type | Pattern essentials | Validation |
|---|---|---|
| `AADHAAR` | 12 digits, first digit 2–9, optional space/hyphen grouping 4-4-4 | Verhoeff checksum |
| `PAN` | `[A-Z]{5}[0-9]{4}[A-Z]`, 4th char in valid holder-type set | holder-type letter |
| `PHONE` | Indian mobile: 10 digits starting 6–9, optional `+91`/`91`/`0` prefix, tolerant of spaces/hyphens | context words adjust score |
| `UPI` | `local@handle` where handle is in a maintained list of UPI handles | handle list (distinguishes from email) |
| `EMAIL` | standard email pattern | UPI takes precedence on overlap |
| `CARD` | 13–19 digits with optional separators | Luhn checksum |
| `IFSC` | `[A-Z]{4}0[A-Z0-9]{6}` (bank code, a literal `0`, branch code), case-insensitive input | 5th character must be `0`; first 4 letters optionally checked against a bundled bank-code list |
| `BANK_ACCOUNT` | 9–18 digits, optional separators | **Only with context** — words like "account", "a/c", "acc no", "account number", "khata", "bank", or an IFSC code nearby. No checksum exists, so without context a long number is not flagged |

**Secrets:**
- Known-prefix patterns (OpenAI, Anthropic, AWS, GitHub, Slack, Google, Stripe-style keys, JWTs, PEM private-key blocks).
- Key/value context (`password=`, `api_key:`, `Authorization: Bearer …`).
- Shannon-entropy check for unknown high-randomness tokens; threshold tuned on the dev split.

**NER (`NAME`):**
- A pretrained token-classification model, exported to ONNX and int8-quantised, run on CPU with ONNX Runtime. Only PERSON entities are used in v1.
- Model choice is made **by measurement** on the dev split, comparing at least a general English NER model and an India-aware one (e.g. AI4Bharat IndicNER), on recall, precision, latency and model size.

**Addresses (`ADDRESS`):**

Indian addresses have no fixed format — landmarks ("near", "opp."), Hinglish ("sector 15 pe"), missing parts, fuzzy start and end — so detection combines several signals:
- **PIN code:** 6 digits, first digit 1–9, validated against India Post's public PIN code directory (bundled as a static list). A standalone 6-digit number is only treated as an address signal with nearby context ("pin", "pincode", a city name) — 6-digit **OTPs** are a major false-positive risk and must be in the negatives of the evaluation set.
- **Address keywords:** e.g. flat, H.No., house no, plot, floor, gali, lane, road, marg, sector, block, nagar, colony, apartments/apts, society, near, opp., behind.
- **Gazetteer:** Indian states, cities, and common locality names (from a free public dataset; source is the builder's call).
- **NER location entities** as an additional weak signal.
- **Span assembly:** where signals cluster within a short window, the detector emits one `ADDRESS` span covering the whole address (from the first address-like token to the PIN code/city), not separate fragments. Window size and scoring are tuned on the dev split.

A PIN code inside a detected address is masked as part of that `ADDRESS` span; a standalone PIN code with context is masked as `ADDRESS` on its own.

**Merging:** overlapping spans are resolved by a fixed precedence (secrets > checksum-validated IDs > other rules > ADDRESS > NAME), then by score. Among number types: `AADHAAR` > `BANK_ACCOUNT` (with account context) > `CARD` > `PHONE` — e.g. some Indian banks use the customer's mobile number as the account number, so "a/c 9876543210" is a `BANK_ACCOUNT`. A name inside an address (e.g. "Sai Krupa Apts") is absorbed into the `ADDRESS` span. Merging rules must be unit-tested.

**Text normalisation:** detectors must handle common separators (spaces, hyphens, dots) inside numbers. Unicode look-alikes and numbers-as-words are out of scope for v1 but should appear in the evaluation as documented failures.

### 6.4 Policy configuration

A YAML file loaded at startup:

```yaml
actions:
  AADHAAR: mask
  PAN: mask
  PHONE: mask
  UPI: mask
  EMAIL: mask
  CARD: mask
  BANK_ACCOUNT: mask
  IFSC: mask
  NAME: mask
  ADDRESS: mask
  SECRET: block
failure_mode: closed        # closed | open
max_request_bytes: 1048576
```

Unknown types or actions fail startup with a clear error.

### 6.5 Upstream configuration

The guard forwards to one configured upstream, any OpenAI-compatible base URL: the bundled mock LLM, Ollama, Gemini's or Groq's OpenAI-compatible endpoints, or OpenAI itself. **The client's `Authorization` header is passed through** to the upstream; the guard holds no provider keys itself and never logs auth headers. Unknown request fields are forwarded unchanged so newer SDK features don't break.

### 6.6 Restoration

- **Non-streaming:** replace placeholders in `choices[].message.content` and in tool-call argument strings.
- **Streaming (SSE):** a placeholder may be split across chunks (`[PHO` + `NE_1]`). The restorer holds back text from a possible placeholder start until it either completes or provably can't be a placeholder (bounded by the maximum placeholder length), then releases it. It must preserve chunk ordering, the final `[DONE]`, and finish reasons.
- **Tolerant matching:** handle common LLM mutations (`[Name_1]`, `[NAME 1]`, missing brackets) where unambiguous; measure how often mutations occur against the free upstreams.
- **Only guard-created placeholders are restored.** Placeholder-looking text that the user typed must never be replaced with a real value. The builder chooses the mechanism (e.g. skipping numbers already present in the input, or a per-request marker), and it must be unit-tested.

### 6.7 Logging and observability

- Structured JSON log line per request: request id, timestamp, per-type counts by action, blocked yes/no, detection latency, upstream latency, total latency.
- **Never**: raw values, message text, or auth headers.
- `GET /healthz` for liveness. A simple counters endpoint (`GET /stats`) is optional.

### 6.8 Failure handling and security of the guard itself

- **Fail-closed by default:** if detection throws, the request is rejected (5xx, OpenAI-shaped error). `failure_mode: open` forwards unmasked and logs loudly — documented as unsafe.
- Request size limit; regexes written to avoid catastrophic backtracking, with a test using adversarial inputs (ReDoS).
- Upstream errors are passed back in OpenAI shape; timeouts configurable.
- Threat model (written as a section of the README): stolen mappings (mitigated: none persisted), log leakage, placeholder injection, ReDoS, the guard as an open relay (mitigated in the self-hosted model by network placement; optional static guard API key header).

---

## 7. Evaluation

### 7.1 Dataset
- A **synthetic data generator** (seeded, reproducible) producing labelled messages with exact character spans and types: customer-support style chats, agent notes, developer snippets containing keys, Hinglish phrasing, and **negatives** (look-alike numbers like order IDs, invoice numbers, timestamps; emails that aren't UPI; random strings that aren't secrets).
- Generated identifiers must be **valid where real ones would be** (correct Verhoeff/Luhn, valid PAN structure) so checksums can't trivially reject them.
- Includes customer records as a support agent would paste them (many types in one message: name, phone, Aadhaar, address, bank account + IFSC), and bank-detail look-alikes (long order/ticket/invoice numbers without account context).
- Includes realistic Indian addresses in varied styles (formal, landmark-based, Hinglish, partial, with and without PIN code) and address look-alikes (6-digit OTPs, order numbers, a city mentioned outside an address).
- Includes multi-turn conversations (see §6.2) and known-hard cases (spaced digits, mixed formats, numbers-as-words) that are expected to fail.
- Target: a few thousand messages, split into **dev** (for tuning) and **test** (for reported numbers only), plus a small **hand-written** set the author writes to sanity-check the generator isn't too easy.

### 7.2 Metrics
- Span-level, per type: precision, recall, F1. Matching rule (exact vs. overlap) is the builder's call but must be stated in the report. For `ADDRESS`, whose boundaries are fuzzy, also report **coverage**: the fraction of the true address's characters that were masked — a detector that masks "Andheri West, Mumbai 400058" but leaves "Flat 302, Sai Krupa Apts" visible has leaked, even if it "overlaps".
- Latency: detection-only p50/p95 across message sizes; end-to-end overhead against the mock LLM.
- Restoration: fraction of responses fully restored, broken down by streaming vs. non-streaming, against at least one real free upstream.
- Model size on disk and memory use of the deployed container.

### 7.3 Presidio baseline
Run Presidio on the same test split with a documented entity mapping (e.g. `IN_AADHAAR→AADHAAR`, `IN_PAN→PAN`, `PHONE_NUMBER→PHONE`, `EMAIL_ADDRESS→EMAIL`, `CREDIT_CARD→CARD`, `PERSON→NAME`, `LOCATION→ADDRESS` — noting that Presidio's `LOCATION` finds place names, not full addresses, so its address coverage is reported with that caveat). Types Presidio lacks (e.g. UPI, IFSC, Indian bank account numbers) are reported as "not supported", not as zeros hidden in an average. Report both engines side by side, with latency.

### 7.4 Report
A generated Markdown report (tables + failure examples with values synthetic) committed to the repo and linked from the README. Re-running one command regenerates it.

---

## 8. Who writes what

**Changed 2026-10-24:** Claude (an AI coding assistant) implements the whole project; the author directs the design, reviews the work, and studies every phase top to bottom.

- After each phase, Claude writes a plain-language walkthrough at `docs/walkthrough/phase-N-<name>.md`: what the phase does, why, how each file works, the key design decisions, and Q&A about that code.
- Code should be written to be **readable and teachable** — clear names, small functions, comments explaining *why* where it isn't obvious — because the author will study it.
- After implementation, Claude walks the author through each phase on request.

---

## 9. Phasing (target ~4–5 weeks)

1. **Skeleton:** scaffold, mock LLM, pass-through proxy (no detection), non-streaming + streaming pass-through, Docker, CI.
2. **Rules + secrets:** all rule detectors with unit tests, policy loading, masking/blocking, non-streaming restore.
3. **Evaluation v0:** generator, dev/test split, harness for rules/secrets, first numbers.
4. **NER:** model comparison on dev, ONNX + quantisation, integrated.
5. **Addresses:** PIN code directory, keywords, gazetteer, span assembly; address cases added to the generator; coverage metric.
6. **Streaming restore** and tolerant matching, tested against a real free upstream.
7. **Presidio baseline, final report, README, threat model, demo video + GIF.**

v1 is "done" when all seven are complete, the README's numbers come from the committed report, the demo video is linked (recording it is the author's step), the app is runnable by a stranger with one command, and every phase has its walkthrough document.

---

## 10. Left to the builder

- Exact project layout, module names, and library choices within the stack (e.g. httpx, pydantic, pytest, ONNX Runtime / Optimum).
- Span matching rule for evaluation (exact vs. overlap), as long as it's stated.
- Placeholder-injection mechanism (§6.6), as long as it's tested.
- Exact entropy threshold and context-word lists (tuned on dev).
- The UPI handle list's source and maintenance approach.
- Whether to include the optional `/stats` endpoint and optional guard API key.
- How to record the demo video and GIF (tooling, script).

---

## 11. Kept open for later: public live demo

The author may decide after v1 to add a public demo link. It is a separate add-on of ~2–3 days, not part of v1:
- a small demo web page (chat box + a view showing what was masked);
- the author's own free-tier key (or ~$5 paid credit, author's choice) held server-side for the demo only;
- strict per-visitor rate limits and input-size limits so strangers can't exhaust the quota;
- free CPU-only hosting.

v1 must not block this: keep the guard a self-contained Docker image with config via environment/YAML, and keep the image small.
