"""NameModel: runs an exported NER model (ONNX) over a text and returns NAME spans.

Only `onnxruntime` and `tokenizers` are needed here, both small: PyTorch is used once,
offline, to export the model (see export.py), and is never installed in the guard.

A model directory (written by `python -m leakguard.ner.export`) holds:
    model.onnx         int8-quantised network; inputs input_ids + attention_mask
    tokenizer.json     the model's own tokenizer, with character offsets
    ner_config.json    labels, special token ids, max length, score threshold
    manifest.json      where the model came from (not read at runtime)
"""

import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from leakguard.detect.spans import Span
from leakguard.ner.decode import decode_names, person_label_ids

REQUIRED_KEYS = ("id2label", "cls_id", "sep_id", "pad_id", "max_length", "threshold",
                 "candidate")
_OVERLAP = 64  # tokens shared by neighbouring windows, so no name is cut at an edge
# Windows per inference call. Memory grows with the batch (measured: ~52 MB per KB of text
# in a single batch), so a long message in one batch could exhaust RAM. With 4, peak memory
# stays bounded however long the message is; most messages need just one window anyway.
MAX_BATCH = 4


class NameModelError(RuntimeError):
    """The model directory is missing or invalid. Raised at startup, never mid-request."""


def _windows(n_tokens: int, body: int) -> list[tuple[int, int]]:
    """[start, end) token ranges of at most `body` tokens covering 0..n_tokens, overlapping."""
    stride = body - min(_OVERLAP, body // 2)
    windows, start = [], 0
    while True:
        windows.append((start, min(start + body, n_tokens)))
        if start + body >= n_tokens:
            return windows
        start += stride


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


class NameModel:
    def __init__(self, session, tokenizer: Tokenizer, config: dict):
        self._session = session
        self._tokenizer = tokenizer
        # Published tokenizer.json files often switch truncation on at 512 tokens, which
        # would silently stop looking at the rest of a long message. Windows handle length.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        id2label = {int(k): v for k, v in config["id2label"].items()}
        self._begin, self._inside = person_label_ids(id2label)
        self._person_cols = sorted(self._begin | self._inside)
        self._cls, self._sep, self._pad = config["cls_id"], config["sep_id"], config["pad_id"]
        self._body = int(config["max_length"]) - 2  # room for [CLS] and [SEP]
        self.threshold = float(config["threshold"])
        self.name = str(config["candidate"])

    @classmethod
    def load(cls, model_dir: str | os.PathLike, threads: int = 1,
             model_file: str = "model.onnx") -> "NameModel":
        """`model_file` lets the comparison load the unquantised model.fp32.onnx."""
        model_dir = Path(model_dir)
        required = (model_file, "tokenizer.json", "ner_config.json")
        missing = [f for f in required if not (model_dir / f).is_file()]
        if missing:
            raise NameModelError(f"missing {', '.join(missing)} in {model_dir}")
        try:
            config = json.loads((model_dir / "ner_config.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise NameModelError(f"ner_config.json is not valid JSON: {exc}") from None
        absent = [k for k in REQUIRED_KEYS if k not in config]
        if absent:
            raise NameModelError(f"ner_config.json lacks {', '.join(absent)}")

        import onnxruntime as ort  # imported here so tests of the rest never need it

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(str(model_dir / model_file), options,
                                           providers=["CPUExecutionProvider"])
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
            model = cls(session, tokenizer, config)
            model.detect("Warm-up call so the first real request isn't slow.")
        except NameModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - any load failure is a startup error
            raise NameModelError(f"could not load the model in {model_dir}: {exc}") from None
        return model

    def _run(self, ids: list[int], windows: list[tuple[int, int]]) -> np.ndarray:
        """Label probabilities for a batch of windows: [CLS] tokens [SEP], padded to the
        longest window in the batch."""
        width = max(end - start for start, end in windows) + 2
        input_ids = np.full((len(windows), width), self._pad, dtype=np.int64)
        mask = np.zeros((len(windows), width), dtype=np.int64)
        for row, (start, end) in enumerate(windows):
            n = end - start
            input_ids[row, 0], input_ids[row, n + 1] = self._cls, self._sep
            input_ids[row, 1:n + 1] = ids[start:end]
            mask[row, :n + 2] = 1
        return _softmax(self._session.run(None, {"input_ids": input_ids,
                                                 "attention_mask": mask})[0])

    def detect(self, text: str) -> list[Span]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids, offsets = encoding.ids, encoding.offsets
        if not ids:
            return []
        windows = _windows(len(ids), self._body)
        # one array per window (batches are padded to different widths)
        probs = [row for i in range(0, len(windows), MAX_BATCH)
                 for row in self._run(ids, windows[i:i + MAX_BATCH])]

        # A token near a window's edge has little context on one side, so each token
        # takes its prediction from the window where it sits furthest from an edge.
        token_probs = np.zeros((len(ids), probs[0].shape[-1]), dtype=np.float32)
        best_distance = np.full(len(ids), -1)
        for row, (start, end) in enumerate(windows):
            for t in range(start, end):
                distance = min(t - start, end - 1 - t)
                if distance > best_distance[t]:
                    best_distance[t] = distance
                    token_probs[t] = probs[row][t - start + 1]  # +1 skips [CLS]

        label_ids = token_probs.argmax(axis=-1).tolist()
        scores = token_probs[:, self._person_cols].sum(axis=-1).tolist()
        return decode_names(text, offsets, label_ids, scores, self._begin, self._inside,
                            self.threshold)
