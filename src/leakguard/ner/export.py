"""Export a candidate NER model to ONNX, quantise it to int8, and check it still agrees.

    python -m leakguard.ner.export                    # shipped model -> models/ner/
    python -m leakguard.ner.export --candidate KEY    # one candidate -> models/candidates/KEY/
    python -m leakguard.ner.export --all              # every candidate (for the comparison)

Needs the export extra (PyTorch etc.): pip install -e ".[export]". The guard itself never
imports this module; it only loads the files written here (see model.py).

Steps: download the pinned revision -> trace the PyTorch model to ONNX (fp32) -> int8
dynamic quantisation (weights stored as 8-bit integers: ~4x smaller, faster on CPU) ->
parity checks against PyTorch -> write ner_config.json and manifest.json.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

from leakguard.ner.candidates import CANDIDATES, DEFAULT_CANDIDATE, Candidate

DEPLOY_DIR = Path("models/ner")
CANDIDATES_DIR = Path("models/candidates")
MAX_FP32_DIFF = 1e-3  # fp32 ONNX must reproduce PyTorch's logits to this precision
# Candidate exports keep both int8 modes so the comparison can pick per model.
QUANT_FILES = {"per-channel": "model.int8-channel.onnx", "per-tensor": "model.int8-tensor.onnx"}

# Fixed sentences for the parity checks (synthetic; nobody real).
PARITY_SENTENCES = (
    "Hi, I'm Priya Nair and my order hasn't arrived.",
    "mera naam rahul hai, refund kab aayega?",
    "Customer: K. Venkatesh | Mob: 98765 43210",
    "Please ask Mr. Iyer to call Fatima Shaikh in Pune.",
    "Your OTP is 482913. Do not share it with anyone.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_onnx(model, path: Path) -> None:
    import torch

    class LogitsOnly(torch.nn.Module):
        """The exported graph takes input_ids + attention_mask and returns only logits."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    # The dynamo exporter captures the real computation graph. The older TorchScript
    # tracer (dynamo=False) records one example run, and with transformers 5.x it froze
    # shape-dependent attention code: logits were off by up to 3.4 on other inputs.
    # Two separate example tensors: passing one tensor object for both inputs makes the
    # exporter treat them as a single graph input (the ids would be used as the mask).
    input_ids = torch.ones((2, 8), dtype=torch.long)
    attention_mask = torch.ones((2, 8), dtype=torch.long)
    dims = {0: torch.export.Dim("batch"), 1: torch.export.Dim("sequence")}
    torch.onnx.export(
        LogitsOnly(model), (input_ids, attention_mask), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_shapes=(dims, dims), opset_version=18, dynamo=True,
        external_data=False,  # one self-contained file (all candidates are < 2 GB)
    )


def _parity(model, hf_tokenizer, fp32: Path, int8: Path, tokenizer_json: Path,
            config: dict) -> dict:
    """Compare the ONNX files with PyTorch on fixed sentences. Also checks that our runtime
    tokenisation (no special tokens, then [CLS]/[SEP] added by NameModel) gives exactly the
    ids Hugging Face would, since the model only works on the ids it was trained with."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from tokenizers import Tokenizer

    runtime_tok = Tokenizer.from_file(str(tokenizer_json))
    runtime_tok.no_truncation()
    sessions = {p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for p in (fp32, int8)}
    max_diff, agree, total, tokenizer_match = 0.0, 0, 0, True
    for sentence in PARITY_SENTENCES:
        enc = hf_tokenizer(sentence, return_tensors="pt")
        ours = [config["cls_id"], *runtime_tok.encode(sentence, add_special_tokens=False).ids,
                config["sep_id"]]
        tokenizer_match &= ours == enc["input_ids"][0].tolist()
        with torch.no_grad():
            expected = model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"]).logits.numpy()
        feeds = {"input_ids": enc["input_ids"].numpy(),
                 "attention_mask": enc["attention_mask"].numpy()}
        got_fp32 = sessions[fp32].run(None, feeds)[0]
        got_int8 = sessions[int8].run(None, feeds)[0]
        max_diff = max(max_diff, float(np.abs(got_fp32 - expected).max()))
        agree += int((got_int8.argmax(-1) == expected.argmax(-1)).sum())
        total += expected.shape[1]
    return {"fp32_max_abs_diff": max_diff, "int8_label_agreement": agree / total,
            "tokenizer_match": bool(tokenizer_match)}


def export(candidate: Candidate, out_dir: Path, keep_fp32: bool) -> dict:
    import onnxruntime
    import torch
    import transformers
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(candidate.repo, revision=candidate.revision)
    model = AutoModelForTokenClassification.from_pretrained(
        candidate.repo, revision=candidate.revision)
    model.eval()

    fp32, int8 = out_dir / "model.fp32.onnx", out_dir / "model.onnx"
    _export_onnx(model, fp32)
    # Per-channel: each output channel of a weight matrix gets its own int8 scale; per-tensor:
    # one scale for the whole matrix, so a few large weights can squash the rest to zero (on
    # dev, MuRIL's NAME F1 fell from 93% to 0% that way). Neither is best for every model,
    # so the mode is chosen per candidate on dev, like the threshold.
    modes = {"per-channel": True, "per-tensor": False}
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8,
                     per_channel=candidate.per_channel)
    if keep_fp32:
        for mode, filename in QUANT_FILES.items():
            quantize_dynamic(str(fp32), str(out_dir / filename), weight_type=QuantType.QInt8,
                             per_channel=modes[mode])

    tokenizer_json = out_dir / "tokenizer.json"
    hf_tokenizer.backend_tokenizer.save(str(tokenizer_json))
    config = {
        "candidate": candidate.key,
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "cls_id": hf_tokenizer.cls_token_id,
        "sep_id": hf_tokenizer.sep_token_id,
        "pad_id": hf_tokenizer.pad_token_id,
        "max_length": min(512, model.config.max_position_embeddings),
        "threshold": candidate.threshold,
        "per_channel": candidate.per_channel,
    }
    (out_dir / "ner_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    parity = _parity(model, hf_tokenizer, fp32, int8, tokenizer_json, config)
    if parity["fp32_max_abs_diff"] >= MAX_FP32_DIFF or not parity["tokenizer_match"]:
        raise RuntimeError(f"{candidate.key}: export does not match PyTorch: {parity}")
    if not keep_fp32:
        fp32.unlink()

    files = sorted(p for p in out_dir.iterdir() if p.name != "manifest.json")
    manifest = {
        "candidate": candidate.key, "repo": candidate.repo, "revision": candidate.revision,
        "exported_on": date.today().isoformat(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "onnxruntime": onnxruntime.__version__},
        "parity": parity,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    # torch.onnx prints emoji progress marks, which crash a legacy (cp1252) Windows console.
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="python -m leakguard.ner.export")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--candidate", choices=sorted(CANDIDATES))
    which.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, help="output directory (single model only)")
    args = parser.parse_args()
    if args.all and args.out:
        parser.error("--out works with a single model, not --all")

    if args.all or args.candidate:
        # Candidate exports keep the fp32 file too, so the comparison can measure what
        # quantisation costs in accuracy and gains in size and speed.
        keys = sorted(CANDIDATES) if args.all else [args.candidate]
        jobs = [(CANDIDATES[k], args.out or CANDIDATES_DIR / k, True) for k in keys]
    else:
        jobs = [(CANDIDATES[DEFAULT_CANDIDATE], args.out or DEPLOY_DIR, False)]

    failed = []
    for candidate, out_dir, keep_fp32 in jobs:
        print(f"exporting {candidate.key} ({candidate.repo}@{candidate.revision[:7]}) "
              f"-> {out_dir}", flush=True)
        try:
            manifest = export(candidate, out_dir, keep_fp32)
        except Exception as exc:  # noqa: BLE001 - report every candidate, then fail
            print(f"  FAILED: {exc}", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)  # never leave a half-written model
            failed.append(candidate.key)
            continue
        size_mb = manifest["files"]["model.onnx"]["bytes"] / 1e6
        print(f"  ok: int8 {size_mb:.0f} MB, parity {manifest['parity']}", flush=True)
    if failed:
        sys.exit(f"export failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
