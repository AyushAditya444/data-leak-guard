"""Evaluation: synthetic labelled data, scoring, and the results report.

Run from the repo root:
    python -m leakguard.evaluation generate     # write data/eval/{dev,test}.jsonl to browse
    python -m leakguard.evaluation report       # write docs/evaluation/report.md
    python -m leakguard.evaluation errors --split dev
"""

import os

# Latency is reported per message on ONE CPU thread. Math libraries (numpy's BLAS, used by
# spaCy in the Presidio baseline) read these when first imported and would otherwise use
# every core, so set them here, before anything imports numpy. setdefault: an explicit
# setting from the caller wins.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _name in THREAD_VARS:
    os.environ.setdefault(_name, "1")
