"""What counts as a placeholder in an LLM's reply.

The guard writes placeholders as `[PHONE_1]`, but LLMs rewrite them a little:
`[Phone_1]`, `[PHONE 1]`, `{PHONE_1}`. Three questions are answered here:

    accepted_mentions()         which placeholders the restorer swaps back
    loose_mentions()            which text merely *looks* like a placeholder (for counting
                                the ones that could not be restored, and how they were mangled)
    could_become_placeholder()  while streaming: could this unfinished tail still turn into an
                                accepted placeholder once more text arrives?

Only types in the current request's mapping are matched, so ordinary text such as
"[see docs]" is never mistaken for one.
"""

import re
from collections.abc import Collection
from dataclasses import dataclass
from functools import lru_cache

from leakguard.entities import EntityType

# Bare `NAME_1` with no brackets (strict: uppercase, underscore, word boundaries). The rule
# set before measuring: accept only if >= 2% of placeholder mentions on the dev run. It was
# 3.8% (28 of 745, mostly gpt-oss), so it is on: docs/evaluation/restore-tuning.md.
ACCEPT_BRACKETLESS = True

MAX_DIGITS = 6
BRACKETS = {"[": "]", "{": "}", "<": ">", "(": ")"}
_SEPARATORS = "_ -"


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    entity: EntityType
    number: int
    form: str  # exact | case | separator | brackets | bracketless | unclosed


def _type_alternatives(entities: Collection[EntityType], strict: bool) -> str:
    """Regex alternatives for the type names, longest first. Outside strict mode the
    underscore inside a name (BANK_ACCOUNT) may also be a space or a hyphen."""
    names = sorted((e.value for e in entities), key=len, reverse=True)
    inner = "_" if strict else "[_ -]"
    return "|".join(inner.join(map(re.escape, name.split("_"))) for name in names)


@lru_cache(maxsize=256)
def _patterns(entities: frozenset[EntityType]) -> tuple[re.Pattern, re.Pattern, re.Pattern]:
    loose_types = _type_alternatives(entities, strict=False)
    bracketed = re.compile(
        rf"(?P<open>[\[{{<(])(?P<type>{loose_types})(?P<sep>[_ -])(?P<num>\d{{1,{MAX_DIGITS}}})"
        r"(?P<close>[\]}>)])", re.IGNORECASE)
    bare = re.compile(
        rf"(?<![A-Za-z0-9_])(?P<type>{_type_alternatives(entities, strict=True)})_"
        rf"(?P<num>\d{{1,{MAX_DIGITS}}})(?![A-Za-z0-9_])")
    loose = re.compile(
        rf"(?P<open>[\[{{<(])?(?<![A-Za-z0-9])(?P<type>{loose_types})(?P<sep>[_ -])"
        rf"(?P<num>\d{{1,{MAX_DIGITS}}})(?!\d)(?P<close>[\]}}>)])?", re.IGNORECASE)
    return bracketed, bare, loose


def _entity(type_text: str) -> EntityType:
    return EntityType(re.sub("[ -]", "_", type_text.upper()))


def _form(m: re.Match) -> str:
    """The most severe way this mention differs from the guard's own `[TYPE_N]`."""
    opener, closer = m.group("open"), m.group("close")
    if opener is None:
        return "bracketless"
    if closer != BRACKETS[opener]:
        return "unclosed"
    if opener != "[":
        return "brackets"
    if m.group("sep") != "_" or m.group("type").upper() != _entity(m.group("type")).value:
        return "separator"
    if m.group("type") != m.group("type").upper():
        return "case"
    return "exact"


def _mention(m: re.Match, form: str) -> Mention:
    return Mention(m.start(), m.end(), _entity(m.group("type")), int(m.group("num")), form)


def accepted_mentions(
    text: str, entities: Collection[EntityType], bracketless: bool = ACCEPT_BRACKETLESS
) -> list[Mention]:
    """Placeholders the restorer swaps back, left to right, never overlapping."""
    if not entities:
        return []
    bracketed, bare, _ = _patterns(frozenset(entities))
    found = [_mention(m, _form(m)) for m in bracketed.finditer(text)
             if m.group("close") == BRACKETS[m.group("open")]]
    if bracketless:
        # A bracketed match wins over the bare name inside it: "[NAME_1]" is one mention.
        taken = [(m.start, m.end) for m in found]
        found += [_mention(m, "bracketless") for m in bare.finditer(text)
                  if not any(s < m.end() and m.start() < e for s, e in taken)]
    return sorted(found, key=lambda m: m.start)


def loose_mentions(text: str, entities: Collection[EntityType]) -> list[Mention]:
    """Everything that looks like a placeholder of these types: any case, any separator,
    with or without brackets, closed or not."""
    if not entities:
        return []
    _, _, loose = _patterns(frozenset(entities))
    return [_mention(m, _form(m)) for m in loose.finditer(text)]


def _continues_type(body: str, name: str, case_sensitive: bool) -> int | None:
    """How many characters of `body` the type `name` accounts for, or None if `body`
    breaks away from it. Returns len(body) when `body` ends inside the name."""
    for i, expected in enumerate(name):
        if i == len(body):
            return i
        got = body[i] if case_sensitive else body[i].upper()
        if expected == "_" and not case_sensitive:
            if got not in _SEPARATORS:
                return None
        elif got != expected:
            return None
    return len(name)


def _could_continue(body: str, names: list[str], case_sensitive: bool) -> bool:
    """Is `body` a prefix of TYPE SEP DIGITS (at most MAX_DIGITS digits)?"""
    separators = "_" if case_sensitive else _SEPARATORS
    for name in names:
        used = _continues_type(body, name, case_sensitive)
        if used is None:
            continue
        rest = body[used:]
        if not rest:
            return True
        digits = rest[1:]
        # isdecimal() is what the regexes' \d matches
        if rest[0] in separators and (digits == "" or digits.isdecimal()) \
                and len(digits) <= MAX_DIGITS:
            return True
    return False


def could_become_placeholder(
    tail: str, entities: Collection[EntityType], bracketless: bool = ACCEPT_BRACKETLESS
) -> bool:
    """True if more text could still turn `tail` into an accepted placeholder. A complete
    bracketed placeholder is not "still becoming" one. A complete bare one is, because the
    next character decides whether it ends at a word boundary (NAME_1 vs NAME_12 vs NAME_1x)."""
    if not tail or not entities:
        return False
    names = [e.value for e in entities]
    if tail[0] in BRACKETS:
        return _could_continue(tail[1:], names, case_sensitive=False)
    if bracketless and tail[0].isupper():
        return _could_continue(tail, names, case_sensitive=True)
    return False


def max_placeholder_len(entities: Collection[EntityType]) -> int:
    """The longest accepted placeholder: brackets + longest type + separator + digits."""
    if not entities:
        return 0
    return 1 + max(len(e.value) for e in entities) + 1 + MAX_DIGITS + 1
