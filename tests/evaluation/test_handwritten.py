from collections import Counter

from leakguard.entities import EntityType as T
from leakguard.evaluation.samples import read_jsonl


def test_handwritten_set_shape():
    samples = read_jsonl("data/eval/handwritten.jsonl")
    assert len(samples) >= 40 and all(s.template == "handwritten" for s in samples)
    types = Counter(span.type for s in samples for span in s.spans)
    for t in set(T) - {T.NAME, T.ADDRESS}:
        assert types[t] >= 2, t
    assert types[T.NAME] >= 15
    assert sum(s.category == "negative" for s in samples) >= 12
    assert len({s.id for s in samples}) == len(samples)
