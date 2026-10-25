"""The performance page: speed, running cost and token cost, rendered from data/eval/perf.json.

Every number here was measured on one laptop, so the hardware comes first.
"""

SIZES_GB = (0.5, 1, 2, 4)
HEADROOM = 1.25  # a machine should have 25% more memory than the measured peak


def machine_size(peak_mb: float, cpus: int) -> str:
    """The smallest common machine size with HEADROOM over the measured peak memory."""
    for gb in SIZES_GB:
        if gb * 1024 >= HEADROOM * peak_mb:
            return f"{cpus} vCPU / {gb:g} GB"
    return f"{cpus} vCPU / more than {SIZES_GB[-1]:g} GB"


def cpu_ms_per_message(load: dict) -> float | None:
    ok = load["requests"] - load["errors"]
    return load["cpu_usec"] / 1000 / ok if ok else None


def _ms(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f} ms"


def _hardware(machine: dict) -> list[str]:
    docker = machine.get("docker")
    lines = [f"- CPU: {machine['cpu']}, {machine['logical_cores']} logical cores",
             f"- RAM: {machine['ram_gb']} GB" if machine.get("ram_gb") else "- RAM: unknown",
             f"- OS: {machine['os']}, Python {machine['python']}"]
    if docker:
        lines.append(f"- Docker {docker['server']} (its VM: {docker['cpus']} CPUs, "
                     f"{docker['memory_gb']} GB)")
    return lines


def _overhead(overhead: dict) -> list[str]:
    lines = [("| Name model | Mode | Overhead p50 | Overhead p95 | First token p50 "
             "| First token p95 |"), "|---|---|---:|---:|---:|---:|"]
    for key, label in (("ner_on", "on"), ("ner_off", "off")):
        for mode in ("non_streaming", "streaming"):
            row = overhead[key][mode]
            ttft = row.get("ttft") or {}
            lines.append(f"| {label} | {mode.replace('_', '-')} | {_ms(row['p50_ms'])} "
                         f"| {_ms(row['p95_ms'])} | {_ms(ttft.get('p50_ms'))} "
                         f"| {_ms(ttft.get('p95_ms'))} |")
    return lines


def _throughput(containers: dict) -> list[str]:
    lines = [("| CPUs | Concurrent clients | Messages/s | p50 | p95 | Requests | Errors "
             "| CPU per message |"), "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in containers["runs"]:
        for load in run["loads"]:
            lines.append(f"| {run['cpus']} | {load['concurrency']} | {load['msgs_per_s']:.1f} "
                         f"| {_ms(load['p50_ms'])} | {_ms(load['p95_ms'])} "
                         f"| {load['requests']} | {load['errors']} "
                         f"| {_ms(cpu_ms_per_message(load))} |")
    return lines


def _memory(containers: dict) -> list[str]:
    lines = ["| CPUs | Memory idle | Memory peak | Fits |", "|---:|---:|---:|---|"]
    for run in containers["runs"]:
        lines.append(f"| {run['cpus']} | {run['idle_mb']:.0f} MB | {run['peak_mb']:.0f} MB "
                     f"| {machine_size(run['peak_mb'], run['cpus'])} |")
    return lines


def _tokens(tokens: dict, encoding: str) -> list[str]:
    change = tokens["tokens_after"] - tokens["tokens_before"]
    share = change / tokens["tokens_before"] if tokens["tokens_before"] else 0.0
    lines = [(f"Counted with tiktoken `{encoding}` (OpenAI's current models and gpt-oss) over "
             f"{tokens['messages']} test messages the guard would forward ({tokens['blocked']} "
             f"blocked messages left out; {tokens['unchanged']} had nothing to mask)."), "",
             (f"- Tokens before masking: {tokens['tokens_before']:,}; after: "
             f"{tokens['tokens_after']:,} ({change:+,}, {100 * share:+.1f}%)."),
             f"- Per message: {tokens['mean_change']:+.2f} tokens on average.", "",
             "Each value against its placeholder, tokenised on their own:", "",
             "| Type | Values | Value tokens (mean) | Placeholder tokens (mean) |",
             "|---|---:|---:|---:|"]
    for entity, row in tokens["by_type"].items():
        lines.append(f"| {entity} | {row['values']} | {row['value_tokens']:.1f} "
                     f"| {row['placeholder_tokens']:.1f} |")
    return lines


def render_performance(results: dict) -> str:
    meta, containers = results["meta"], results["containers"]
    lines = [
        "# Performance, running cost and token cost",
        "",
        (f"Generated {meta['date']} at commit `{meta['guard_commit']}` with "
        "`python -m leakguard.evaluation perf-report`, from `data/eval/perf.json` "
        "(measured with `perf-run`)."),
        "",
        "## Measured on",
        "",
        *_hardware(meta["machine"]),
        "",
        ("A laptop, not a server: absolute numbers will differ elsewhere; the comparisons "
        "between rows are what carries over."),
        "",
        "## What the guard adds to each request",
        "",
        ("Client, guard and the mock LLM run as plain processes on the same machine, talking "
        "over loopback. Overhead = time through the guard minus time straight to the mock, per "
        "message (best of 3 each), over test messages the guard would forward. *First token* "
        "is the same difference for the first piece of reply text in a streamed reply."),
        "",
        *_overhead(results["overhead"]),
        "",
        "## Throughput",
        "",
        ("The real guard container, capped with Docker's `--cpus`, one uvicorn worker, the name "
        "model on one thread; the load client and the mock run in their own containers on the "
        "same Docker network. Errors are counted and left out of messages/s. *CPU per "
        "message* is the guard container's CPU time divided by successful requests."),
        "",
        *_throughput(containers),
        "",
        "## Running cost",
        "",
        (f"- Image size: {containers['image_disk']} on disk, {containers['image_mb']:.0f} MB "
         "compressed (what a pull downloads); exported name model: "
         f"{containers['model_mb']:.0f} MB."),
        (f"- Memory read from the container's {containers['memory_source']}. *Fits* = the "
        f"smallest of {', '.join(f'{g:g}' for g in SIZES_GB)} GB with "
        f"{100 * (HEADROOM - 1):.0f}% headroom over the peak."),
        "",
        *_memory(containers),
        "",
        "## LLM token cost of masking",
        "",
        *_tokens(results["tokens"], results["encoding"]),
        "",
        "## Notes",
        "",
        ("- Docker Desktop on Windows runs containers in a VM, so container numbers include its "
        "overhead; that is why overhead is measured on plain processes instead."),
        ("- The mock LLM answers instantly, so these numbers are the guard's own cost; a real "
        "LLM takes hundreds of milliseconds to seconds per reply."),
    ]
    return "\n".join(lines)
