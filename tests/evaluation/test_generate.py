import random
import re

from leakguard.entities import EntityType as T
from leakguard.evaluation.generate import SPLIT_SIZES, build_split, render_template
from leakguard.evaluation.samples import parse_marked, read_jsonl
from leakguard.evaluation.templates import TEMPLATES
from leakguard.evaluation.values import NAME_POOLS


def test_render_template_marks_gold_and_not_filler():
    marked = render_template("call {PHONE} about order {~ORDER_ID}", random.Random(3))
    text, spans = parse_marked(marked)
    assert [s.type for s in spans] == [T.PHONE] and "[[" not in text


def test_variant_slots_use_base_type():
    _, spans = parse_marked(render_template("key {SECRET.aws}", random.Random(3)))
    assert spans[0].type is T.SECRET


def test_build_split_sizes_and_determinism():
    dev = build_split("dev", seed=1)
    assert len(dev) == sum(SPLIT_SIZES.values())
    assert dev == build_split("dev", seed=1)


def test_every_span_matches_its_value():
    for sample in build_split("test", seed=2):
        for span in sample.spans:
            value = sample.text[span.start:span.end]
            assert value and value == value.strip(), (sample.id, value)


def test_splits_share_no_templates():
    dev = {s.template for s in build_split("dev", seed=1)}
    test = {s.template for s in build_split("test", seed=2)}
    assert dev and test and not dev & test


def test_negatives_have_no_gold_and_minimum_template_counts():
    assert all(not s.spans for s in build_split("dev", 1) if s.category == "negative")
    minimum = {"support": 14, "agent_record": 6, "dev_snippet": 12, "hinglish": 12,
               "negative": 14, "hard": 8}
    assert all(len(TEMPLATES[c]) >= n for c, n in minimum.items())


def test_each_type_well_represented_in_both_splits():
    # Otherwise a split's numbers for a type rest on a handful of examples.
    from collections import Counter
    for split, seed in (("dev", 1), ("test", 2)):
        counts = Counter(sp.type for s in build_split(split, seed) for sp in s.spans)
        for t in T:
            assert counts[t] >= 50, (split, t, counts[t])


def test_generated_splits_are_built_in_memory_not_read_from_disk():
    # dev/test are not committed (their fake API keys trip GitHub's secret scanning);
    # the CLI rebuilds them deterministically instead.
    from leakguard.evaluation.__main__ import load_split
    assert load_split("dev") == build_split("dev", seed=1)
    assert load_split("test") == build_split("test", seed=2)
    assert load_split("handwritten") == read_jsonl("data/eval/handwritten.jsonl")


def test_test_split_names_never_appear_in_dev_pool():
    dev_f, dev_l = NAME_POOLS["dev"]
    for s in build_split("test", seed=2):
        for sp in s.spans:
            if sp.type is T.NAME and "wordlike" not in s.template:
                words = re.findall(r"[A-Za-z]{2,}", sp.text(s.text))
                assert not {w.capitalize() for w in words} & (set(dev_f) | set(dev_l)), s.id


def test_every_name_form_in_both_splits():
    forms = ("first", "lower", "surname", "initial", "three", "caps", "wordlike")
    for split, seed in (("dev", 1), ("test", 2)):
        templates = {s.template for s in build_split(split, seed)}
        for form in forms:
            assert any(f"{{NAME.{form}}}" in t for t in templates), (split, form)


def test_name_negatives_in_both_splits():
    for split, seed in (("dev", 1), ("test", 2)):
        assert any(s.category == "negative" and ("{~FESTIVAL}" in s.template
                   or "{~BRAND}" in s.template) for s in build_split(split, seed))
