from leakguard.evaluation.perf_report import cpu_ms_per_message, machine_size, render_performance

STATS = {"n": 300, "p50_ms": 1.2, "p95_ms": 3.4}


def _load(concurrency, requests=1000, errors=0, cpu_usec=2_000_000):
    return {"concurrency": concurrency, "seconds": 30, "requests": requests, "errors": errors,
            "msgs_per_s": 33.3, "p50_ms": 20.0, "p95_ms": 45.0, "cpu_usec": cpu_usec}


def _results(errors=0):
    overhead = {"non_streaming": STATS, "streaming": {**STATS, "ttft": STATS}}
    return {
        "meta": {"date": "2026-10-25", "guard_commit": "abc1234",
                 "machine": {"cpu": "Intel(R) Core(TM) i5-TEST", "logical_cores": 8,
                             "ram_gb": 16.0, "os": "Windows-11", "python": "3.11.0",
                             "docker": {"server": "28.0", "cpus": 8, "memory_gb": 7.6}}},
        "overhead": {"ner_on": overhead, "ner_off": overhead},
        "containers": {"image_mb": 900.0, "image_disk": "1.03GB", "model_mb": 280.0,
                       "memory_source": "cgroup memory.peak",
                       "runs": [{"cpus": 1, "idle_mb": 350.0, "peak_mb": 420.0,
                                 "loads": [_load(1), _load(8, errors=errors)]}]},
        "tokens": {"messages": 10, "blocked": 1, "unchanged": 2, "tokens_before": 100,
                   "tokens_after": 104, "mean_change": 0.4,
                   "by_type": {"PHONE": {"values": 3, "value_tokens": 4.0,
                                         "placeholder_tokens": 4.0}}},
        "encoding": "o200k_base",
    }


def test_machine_size_rule():
    assert machine_size(380, 1) == "1 vCPU / 0.5 GB"
    assert machine_size(420, 1) == "1 vCPU / 1 GB"
    assert machine_size(900, 2) == "2 vCPU / 2 GB"
    assert machine_size(4000, 2) == "2 vCPU / more than 4 GB"


def test_performance_page_shows_hardware_first():
    md = render_performance(_results())
    assert md.index("Intel(R) Core(TM) i5-TEST") < md.index("|")


def test_performance_page_shows_errors_next_to_throughput():
    md = render_performance(_results(errors=3))
    row = next(line for line in md.splitlines() if line.startswith("| 1 | 8 |"))
    assert "| 3 |" in row


def test_cpu_ms_per_message():
    assert cpu_ms_per_message(_load(1, requests=1000, cpu_usec=2_000_000)) == 2.0
    assert cpu_ms_per_message(_load(1, requests=10, errors=10)) is None


def test_performance_page_shows_compressed_and_disk_image_size():
    md = render_performance(_results())
    assert "1.03GB on disk" in md and "900 MB compressed" in md
