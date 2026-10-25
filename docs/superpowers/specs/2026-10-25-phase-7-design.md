# Phase 7 design: Presidio baseline, performance and cost, hardening, demo

Date: 2026-10-25. Extends §6.8, §7, §9 and §10 of
[the main design](2026-10-24-data-leak-guard-design.md); where the two differ, this one wins
for Phase 7.

## 1. Intent

Phases 1–6 built a guard that works and measured how well it detects and restores. Phase 7
turns it into a v1 that a stranger can run and an interviewer can check:

- **"Why not just use Presidio?"** answered with numbers on the same data, with Presidio
  set up two ways so the comparison can't be dismissed as rigged.
- **"Will it slow my app down, and what does it cost to run?"** answered with measured
  speed, resource use and LLM token cost.
- **"Is the guard itself secure?"** answered with a threat model in the README where every
  threat points to a mitigation in the code, a test, or a stated acceptance, plus two small
  hardening fixes.
- **"Show me it working."** answered with a ~2-minute narrated, subtitled demo video and a
  GIF, generated from scripts in the repo.

The audience for the video and the README is an **interviewer who has the repo open** and
wants to see the claims proven, not pitched.

## 2. Goals, non-goals, anti-goals

**Goals**
1. Presidio on the test and hand-written sets in two setups, side by side with our engine,
   with latency (§4).
2. Speed, running-cost and token-cost numbers in a committed report (§5).
3. Guard API key (optional) and a localhost-only default port binding (§6).
4. Threat model and "what it does not protect against" in the README (§7).
5. Demo video (synthetic voice, burned-in subtitles), GIF, and the scripts that regenerate
   them (§8).
6. Final README, report summary, walkthroughs, and the main design's "done" criteria met.

**Non-goals**
- Improving our detectors. No detection change in this phase; if measurement exposes a
  defect, it is written down as a known limit (or becomes a post-v1 task), not fixed against
  the test set.
- Writing custom Presidio recognizers (that would be our engine inside Presidio).
- Plugging our name model into Presidio.
- A public hosted demo (still §11 of the main design, post-v1).
- Rupee cost estimates. Running cost is stated as measured resources and the smallest machine
  size they fit. One cloud provider's published price (with link and date) may be added only
  if the author asks.
- Rate limiting, TLS termination, multi-key auth, key rotation for the guard key.

**Anti-goals** (failure even if it "works")
- A Presidio setup an informed reviewer could fairly call sabotaged (wrong language, wrong
  mapping, a disabled feature we didn't disclose).
- Any decision made by looking at test results (ours or Presidio's).
- A number in the README or the video that is not in a committed report.
- A real API key anywhere on screen, in a file, in logs, or in git history.
- Fake secrets committed in a form GitHub push protection flags (it blocked this before;
  see the `.gitignore` note on dev/test splits).
- A performance number without the hardware it was measured on.

## 3. Delivery: two PRs (approach B)

| | Approaches considered | Trade-off |
|---|---|---|
| A | One phase, one PR | Simplest; the biggest PR yet, and a demo-tooling problem would hold up the measurements |
| **B** ✅ | **7a "measure and harden", then 7b "demo and final README"** | One extra merge; numbers are settled before the video shows them; risky video tooling is isolated; each PR is reviewable |
| C | Throwaway spike of the demo tools first | Removes the main unknown early but delays the certain work; folded into the start of 7b instead |

The author chose B. Each PR follows the usual workflow (plan in `docs/superpowers/plans/`,
branch, TDD, walkthrough, self-review, PR, CI green, private build journal and interview
guide updated, the author merges).

- **7a** (branch `phase-7a-measure-and-harden`): §4, §5, §6, §7, and their walkthrough.
- **7b** (branch `phase-7b-demo`): §8, the final README pass (§9), and its walkthrough. Starts
  with a feasibility check of VHS, Kokoro and ffmpeg on the author's laptop (§8.6).

## 4. Presidio baseline (7a)

### 4.1 Same yardstick
Both engines are scored by the existing `evaluation.metrics.score` on the **test** split and
the **hand-written** set, with the existing matching rule (same type and overlapping) and
coverage metric. Latency uses the existing `measure_latency` (best of 3, one CPU thread).

### 4.2 Two setups
| Setup | What it is |
|---|---|
| **Out of the box** | `AnalyzerEngine()` with no arguments, `language="en"`, Presidio's default score threshold, its default spaCy model (`en_core_web_lg`). Exactly what a developer gets after `pip install`. |
| **Best shot** | The same, plus every Indian recognizer Presidio ships (`IN_AADHAAR`, `IN_PAN`, and the others) enabled, and a **minimum score per mapped type tuned on dev** (§4.4). Only Presidio's own configuration surface: no custom recognizers, patterns, deny lists or models. |

Recon (Presidio `main`, 2026-10-25): in `conf/default_recognizers.yaml` all six Indian
recognizers are `enabled: false`; `PhoneRecognizer`'s default regions already include `IN`.
The builder **re-checks this in the pinned version** and records what was found in the
baseline report. If the Indian recognizers turn out to be on by default in that version, the
out-of-the-box setup includes them and the report says so.

### 4.3 Entity mapping
| Presidio | Ours |
|---|---|
| `IN_AADHAAR` | AADHAAR |
| `IN_PAN` | PAN |
| `PHONE_NUMBER` | PHONE |
| `EMAIL_ADDRESS` | EMAIL |
| `CREDIT_CARD` | CARD |
| `PERSON` | NAME |
| `LOCATION` | ADDRESS (caveat: Presidio finds place names, not full addresses; its address numbers are shown with this caveat and with coverage, which is the fairer measure here) |

- **Not supported** by Presidio: UPI, IFSC, BANK_ACCOUNT, SECRET. Their precision, recall and
  F1 cells say "not supported", never 0%. Their **coverage is still shown**: the share of
  those values Presidio hid anyway through other recognizers (e.g. `US_BANK_NUMBER` firing
  on account numbers with "account" nearby). *(The author's decision: hiding a point in
  Presidio's favour would be the "rigged" moment the anti-goals rule out.)*
- **Unmapped Presidio types** (DATE_TIME, URL, IP_ADDRESS, NRP, US_*, IN_VEHICLE_REGISTRATION,
  …) are **dropped from the typed metrics** (not counted as false positives), and reported as a
  count of how often each fired.
- **Coverage uses every Presidio prediction, mapped or not**, because an anonymiser built on
  Presidio would hide all of them. Typed precision/recall use mapped predictions only.
  (Builder's call how to wire this; `score` currently uses one prediction list for both.)
- **Overlapping Presidio results** (e.g. `IN_AADHAAR` and `PHONE_NUMBER` on the same digits)
  are resolved **the way Presidio's own anonymiser resolves them** before scoring (a larger
  span wins over one it contains, otherwise the higher score), because that is what a
  developer using Presidio actually gets masked. The per-type minimum score is applied
  first, then conflicts are resolved, matching the analyser → anonymiser order. The report
  states this and gives the number of conflicts resolved. *(Builder's-check answer, Claude's
  call: scoring raw duplicates would count each as a false positive and unfairly lower
  Presidio's precision.)*
- **Aggregate rows:** an "all types both support" row (AADHAAR, PAN, PHONE, EMAIL, CARD, NAME)
  computed the same way for both engines, so the averages compare like with like; ADDRESS
  shown separately because of its caveat; our "all built types" row shown for our engine only.

### 4.4 Best-shot tuning rule (stated before running)
For each mapped type, choose the minimum score from {0.0, 0.05, …, 0.95} that maximises that
type's **F1 on dev**; ties go to the higher threshold. Types whose scores don't vary (e.g. spaCy
`PERSON`) will show a flat curve; that is reported, not worked around. The dev tuning table is
committed (`docs/evaluation/presidio-tuning.md`), as earlier tuning pages were.

### 4.5 Packaging
- New optional extra `[baseline]`: `presidio-analyzer` pinned to an exact version, spaCy and
  `en_core_web_lg` pinned to a matching version (via its wheel URL). Never in the Docker image
  or the guard's dependencies.
- CI does not install it (the spaCy model is ~600 MB). Baseline results are written to a
  committed JSON (`data/eval/presidio-{dev,test}.json` or similar, holding scores and counts,
  no message text) and the report renders from it, as `restore-report` does.
- Command: `python -m leakguard.evaluation presidio-run --split {dev,test}` (needs the extra),
  then the report renders without it.

## 5. Performance and cost (7a)

All measured on the author's laptop; the report records CPU model, cores, RAM, OS, Python and
Docker versions next to every number. Results go to committed JSON; the report renders from it.

**Where things run** *(builder's-check answer, Claude's call)*:
- **Overhead and time to first token:** client, guard and mock all run as native processes on
  the laptop over loopback, so Docker Desktop's port forwarding (which can cost more than the
  rules' 0.175 ms) is not measured as guard overhead.
- **Throughput, memory and CPU:** the guard runs as the real container, **capped with
  `--cpus`** (1 and 2), with the client and mock in the same Docker network. The cap matters
  because detection runs in a thread pool and ONNX Runtime releases the GIL, so an uncapped
  guard would quietly use several cores. The machine-size conclusion is then measured, not
  estimated.

### 5.1 Speed
- **Overhead per request:** client → guard → mock vs. client → mock directly, both as real
  processes (uvicorn), over the test messages; p50/p95 of (through guard − direct). Non-streaming
  and streaming; name model on and off.
- **Time to first streamed token:** the same comparison for the first content chunk.
- **Throughput:** messages per second on one uvicorn worker with `LEAKGUARD_NER_THREADS=1`,
  at a low and a moderate number of concurrent clients, with p95 latency at each. Builder
  picks the concurrency levels and duration and states them.

### 5.2 Running cost
- Guard container memory (from `docker stats`): idle after startup, and peak during the
  throughput test.
- Image size on disk (`docker image ls`), and the exported model's size.
- CPU time per message (process CPU time ÷ messages during the throughput test).
- A one-line conclusion of the form "fits a N vCPU / M GB machine at X msg/s", derived only
  from the numbers above.

### 5.3 LLM token cost
For each test message the guard would forward (secrets are blocked, so blocked messages are
excluded and counted), count tokens before and after masking with `tiktoken`'s `o200k_base`
(the encoding used by OpenAI's current models and `gpt-oss`; stated in the report). Report total
and mean change, and change per type. Whatever the sign, it is reported as measured.

### 5.4 Where it goes
`docs/evaluation/performance.md` (speed, running cost, token cost), generated by
`python -m leakguard.evaluation perf-run` (needs Docker; builder's call on exact shape) and
`perf-report`.

## 6. Hardening (7a)

### 6.1 Localhost-only by default
`docker-compose.yml` publishes the guard as `127.0.0.1:8080:8080` (today `8080:8080`, which
Docker binds on every interface, so the guard is reachable from the local network). The
README says how to expose it deliberately, and to set the guard key when doing so. The
Dockerfile's in-container `--host 0.0.0.0` stays (needed inside the container).

### 6.2 Optional guard key
- `LEAKGUARD_API_KEY` env var. Unset or empty → off (today's behaviour). Set → every `/v1/*`
  request must carry header `X-Leakguard-Key` with that value, otherwise **401**, OpenAI-shaped
  error, returned **before the body is read**.
- Compared with `hmac.compare_digest` (constant time).
- A set key shorter than 16 characters stops the guard at startup (a weak key is a
  misconfiguration, consistent with how `LEAKGUARD_NER` typos are handled).
- The header is never forwarded upstream (today only `authorization` is forwarded; a test
  pins this), never logged; the log line records only that auth failed.
- `/healthz` stays open.
- README shows the OpenAI SDK usage: `OpenAI(..., default_headers={"X-Leakguard-Key": ...})`.

## 7. Threat model (7a, README section)

A table: **threat · who · what the guard does · proof** (a test, file or measured number). At
least: stolen placeholder mappings; content or keys in logs; placeholder injection; ReDoS;
the guard as an open relay (§6); detection failure (fail-closed); oversized requests (413);
the guard key itself (timing, logging, forwarding).

Then **"What it does not protect against"**, each with the measured number where one exists:
data in images or attached files; non-Latin scripts (e.g. Devanagari) that were not measured;
identity inferred from unmasked context; the known-hard formats from the report (spaced PANs,
numbers as words, unknown UPI handles); anything sent around the guard; a malicious or
compromised upstream seeing placeholders and surrounding context.

## 8. Demo (7b)

### 8.1 What the viewer sees
One terminal pane with **labelled boxes** (layout A, chosen by the author): YOUR APP SENT →
WHAT LEFT YOUR MACHINE → YOUR APP GOT BACK, with burned-in subtitles. Draft scenes (~2 min;
wording tightened in the narration script):

| # | Scene |
|---|---|
| 1 | Title card |
| 2 | The one changed line (`base_url`) |
| 3 | Core proof: a support message with Aadhaar, UPI, phone, name, address; what left the machine; the restored reply (real model) |
| 4 | Streaming, with a placeholder split across chunks visible in the wire tap |
| 5 | A pasted secret → 400, wire tap shows nothing sent |
| 6 | Look-alikes left alone (an OTP that is a real PIN code, an order number) |
| 7 | The guard's log line: counts only |
| 8 | Results: our engine vs Presidio (both setups), read from the committed 7a results, not typed in |
| 9 | `docker compose up`, repo link, one honest limitation |

Narration and subtitles describe what happens ("the reply comes back with the real phone
number restored"), never quote the model's words, so they stay true for any reply.

### 8.2 Upstream
The recording uses **real Groq** (`openai/gpt-oss-120b`). The same demo runs against the
**mock** for anyone without a key, and CI runs it against the mock (without VHS) as a smoke
test so the demo cannot silently break.

### 8.3 Wire tap
A small demo-only pass-through in `examples/demo/` between the guard and the upstream. It
records exactly what the guard sent (request messages, and streamed chunks) for the demo
script to display. It never records or prints `Authorization`. It is not part of the guard
package or image.

### 8.4 Keys and fake secrets
- The Groq key comes from the author's Windows user environment at record time; it never
  appears on screen, in a committed file, in the tap's output, or in a rendered frame.
  Checked to still work before recording (the author planned to rotate it after Phase 6).
- The fake secret for scene 5 must not be a string GitHub push protection flags when
  committed (e.g. assembled at runtime, or a `password=` form the detector blocks).

### 8.5 Pipeline (all free; offline except the Groq call)
- **VHS** (in Docker) renders the terminal from a committed `.tape` script → silent MP4.
- **Kokoro** (open-source TTS, offline) reads the committed narration script → audio.
- Subtitles (`.srt`) generated from the same narration script and the audio timings, so they
  always match; **burned into** the final MP4 and also shipped as `.srt`.
- **ffmpeg** joins video + audio + subtitles, and cuts a short GIF of scene 3 for the top of
  the README.
- Visuals and narration stay **separate files** until the final join, so the author can record
  their own voiceover against the same script and swap it in with one documented command
  (subtitles stay as written, so the voiceover should follow the script).
- One command regenerates everything. Scene timing must not let one scene's narration run into
  the next (builder's call how: e.g. audio first, then scene durations from audio length).
- Kokoro's model licence is recorded next to the demo files.

### 8.6 Feasibility check first
7b starts by checking that VHS (Docker), Kokoro and ffmpeg run on the author's Windows laptop.
If one doesn't, report back with the alternative before building on it.

### 8.7 Where the files live
- Scripts, tape, narration, `.srt`: `examples/demo/` (committed).
- GIF: committed (kept small, target a few MB).
- MP4: **not committed** (binaries bloat history). The author attaches it in the README through
  GitHub's web editor (inline player); this upload is the author's step.

## 9. Final README and docs (7b, with 7a pieces)
- Status → v1; headline table gains Presidio columns; links to the new reports.
- GIF at the top; video link; threat model (from 7a); guard key and port binding settings.
- Docs list gains the missing Phase 6 walkthrough and the Phase 7 walkthroughs
  (`docs/walkthrough/phase-7a-measure-and-harden.md`, `phase-7b-demo.md`).
- `docs/evaluation/report.md` gains a short summary at the top linking every report page.
- Main design §9 "done" criterion updated: the demo video is generated by the build; the
  author's steps are the upload and an optional voiceover.
- Private (outside the repo): interview guide numbers filled from the committed reports,
  including the correction that Presidio's Indian recognizers are off by default; build
  journal updated.

## 10. Testing
- Guard key: missing, wrong, right; short key refuses startup; header not forwarded; not
  logged; `/healthz` open; 401 before the body is read.
- Presidio mapping and "unsupported"/"unmapped" handling, with a fake analyzer (no spaCy in
  CI). Tuning rule (tie-break) as a unit test.
- Token counting on a few fixed examples.
- Perf and Presidio report renderers from fixture JSON.
- Demo script against the mock in CI.

## 11. Left to the builder
- Module names and file layout within the above.
- Exact Presidio and spaCy versions (latest stable at build time, pinned).
- Concurrency levels and durations for the throughput test.
- How coverage is computed from all Presidio predictions.
- VHS tape structure, terminal size and font, Kokoro voice choice, GIF length.
- Whether the perf runner drives Docker itself or documents a two-step run.
