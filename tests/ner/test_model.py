import numpy as np
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from leakguard.ner.model import NameModel, NameModelError

ID2LABEL = {"0": "O", "1": "B-MISC", "2": "I-MISC", "3": "B-PER", "4": "I-PER", "5": "B-ORG",
            "6": "I-ORG", "7": "B-LOC", "8": "I-LOC"}
VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "rahul": 4, "sharma": 5, "word": 6,
         "hi": 7, "i": 8, "am": 9}
RAHUL, SHARMA = VOCAB["rahul"], VOCAB["sharma"]


def _tokenizer(truncate_at: int | None = None) -> Tokenizer:
    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if truncate_at:
        tok.enable_truncation(truncate_at)  # like many published tokenizer.json files
    return tok


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: "rahul" -> B-PER, "sharma" -> I-PER."""

    def __init__(self):
        self.calls = []

    def run(self, output_names, feeds):
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        assert ids.dtype == np.int64 and ids.shape == mask.shape
        self.calls.append(ids.shape)
        logits = np.full((*ids.shape, len(ID2LABEL)), -5.0, dtype=np.float32)
        logits[..., 0] = 5.0
        for label, token in ((3, RAHUL), (4, SHARMA)):
            hit = ids == token
            logits[hit, 0], logits[hit, label] = -5.0, 5.0
        return [logits]


def _model(max_length=512, truncate_at=None, session=None):
    config = {"id2label": ID2LABEL, "cls_id": 2, "sep_id": 3, "pad_id": 0,
              "max_length": max_length, "threshold": 0.5, "candidate": "fake"}
    return NameModel(session or FakeSession(), _tokenizer(truncate_at), config)


def _names(model, text):
    return [s.text(text) for s in model.detect(text)]


def test_detects_name_through_onnx_contract():
    model = _model()
    spans = model.detect("hi I am Rahul Sharma")
    assert [s.text("hi I am Rahul Sharma") for s in spans] == ["Rahul Sharma"]
    assert spans[0].detector == "ner" and model.name == "fake"


def test_name_beyond_first_window_is_found():
    # truncation set in tokenizer.json must not silently cut long messages
    text = "word " * 40 + "Rahul Sharma"
    assert _names(_model(max_length=16, truncate_at=10), text) == ["Rahul Sharma"]


def test_name_on_window_boundary_found_once():
    # max_length 16 -> 14 tokens per window, 7 overlap: windows start at 0, 7, 14...
    text = "word " * 6 + "Rahul Sharma " + "word " * 12
    assert _names(_model(max_length=16), text) == ["Rahul Sharma"]


def test_all_windows_run_in_one_batch():
    session = FakeSession()
    _model(max_length=16, session=session).detect("word " * 30)
    assert len(session.calls) == 1 and session.calls[0][0] >= 3


def test_short_text_is_not_padded_to_max_length():
    session = FakeSession()
    _model(session=session).detect("hi Rahul")
    assert session.calls == [(1, 4)]  # [CLS] hi rahul [SEP]


@pytest.mark.parametrize("text", ["", "   ", "🙂🙂", "मेरा नाम"])
def test_odd_inputs_return_no_spans(text):
    session = FakeSession()
    assert _model(session=session).detect(text) == []
    if not text.strip():
        assert session.calls == []


def test_load_reports_missing_file(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "ner_config.json").write_text("{}")
    with pytest.raises(NameModelError, match="tokenizer.json"):
        NameModel.load(tmp_path)


def test_load_reports_missing_config_key(tmp_path):
    for name in ("model.onnx", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "ner_config.json").write_text('{"cls_id": 1}')
    with pytest.raises(NameModelError, match="id2label"):
        NameModel.load(tmp_path)
