import pytest

from leakguard.entities import EntityType
from leakguard.placeholders import (
    accepted_mentions,
    could_become_placeholder,
    loose_mentions,
    max_placeholder_len,
)

ALL = set(EntityType)


@pytest.mark.parametrize("text", ["[NAME_1]", "[Name_1]", "[name_1]", "[NAME 1]", "[NAME-1]",
                                  "{NAME_1}", "<NAME_1>", "(NAME_1)", "[BANK ACCOUNT_1]",
                                  "[bank-account 1]"])
def test_accepted_forms(text):
    (m,) = accepted_mentions(text, ALL, bracketless=False)
    assert (m.start, m.end, m.number) == (0, len(text), 1)


@pytest.mark.parametrize("text", ["NAME_1", "[NAME_1}", "[NAME__1]", "[NAMES_1]",
                                  "[NAME_1234567]", "[NAME_]", "[ NAME_1 ]"])
def test_rejected_forms(text):
    assert accepted_mentions(text, ALL, bracketless=False) == []


def test_accepted_mention_reports_entity_and_position():
    (m,) = accepted_mentions("call [bank account_12] now", ALL, bracketless=False)
    assert (m.entity, m.number, m.start, m.end) == (EntityType.BANK_ACCOUNT, 12, 5, 22)


def test_bracketless_is_strict_when_enabled():
    assert [m.number for m in accepted_mentions("call NAME_1 now", ALL, bracketless=True)] == [1]
    for text in ["call name_1 now", "call NAME 1 now", "XNAME_1", "NAME_1X"]:
        assert accepted_mentions(text, ALL, bracketless=True) == []


def test_bracketed_wins_over_bare_name_inside_it():
    mentions = accepted_mentions("[NAME_1]", ALL, bracketless=True)
    assert [(m.start, m.end) for m in mentions] == [(0, 8)]


def test_only_mapped_entities():
    assert accepted_mentions("[PHONE_1]", {EntityType.NAME}, bracketless=False) == []


def test_loose_forms():
    text = "[NAME_1] [name_1] [NAME 1] {NAME_1} NAME_1 [NAME_1 x"
    assert [m.form for m in loose_mentions(text, {EntityType.NAME})] == [
        "exact", "case", "separator", "brackets", "bracketless", "unclosed"]


def test_loose_ignores_unmapped_types_and_plain_words():
    assert loose_mentions("[PHONE_1] my name is Ravi", {EntityType.NAME}) == []


@pytest.mark.parametrize("tail", ["[", "[n", "[PHO", "[PHONE_", "[PHONE_1", "{bank acc",
                                  "[BANK_ACCOUNT 12"])
def test_could_become(tail):
    assert could_become_placeholder(tail, ALL, bracketless=False)


@pytest.mark.parametrize("tail", ["[x", "[PHONE_1]", "[PHONEX", "[PHONE_1234567", "NAM", "",
                                  "[PHONE_1}"])
def test_cannot_become(tail):
    assert not could_become_placeholder(tail, ALL, bracketless=False)


def test_bare_prefix_held_only_when_bracketless():
    assert could_become_placeholder("NAM", ALL, bracketless=True)
    assert could_become_placeholder("NAME_1", ALL, bracketless=True)  # NAME_12 still possible
    assert not could_become_placeholder("nam", ALL, bracketless=True)


def test_max_placeholder_len():
    # "[" + "BANK_ACCOUNT" + "_" + 6 digits + "]"
    assert max_placeholder_len({EntityType.NAME, EntityType.BANK_ACCOUNT}) == 1 + 12 + 1 + 6 + 1


@pytest.mark.parametrize("text", ["[EMA\u0130L 1]", "<UP\u0130 1>", "[BAN\u212a_ACCOUNT_1]",
                                  "EMA\u0130L_1", "[PHONE_\u0661]"])
def test_unicode_lookalikes_never_match_or_crash(text):  # final review C1
    assert accepted_mentions(text, ALL, bracketless=True) == []
    assert loose_mentions(text, ALL) == []
    assert not could_become_placeholder(text[:-1], ALL, bracketless=True)
