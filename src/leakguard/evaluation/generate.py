"""Builds the seeded dev/test datasets from templates."""

import random
import re

from leakguard.evaluation.samples import Sample, parse_marked
from leakguard.evaluation.templates import TEMPLATES
from leakguard.evaluation.values import VALUE_MAKERS, Maker, address_makers, name_makers

_SLOT = re.compile(r"\{(~?)([A-Z_]+)(?:\.([a-z_]+))?\}")

# Samples per category, per split (1,600 per split). Phase 5 raised agent_record and hard
# so that every address style has at least 20 samples per split.
SPLIT_SIZES: dict[str, int] = {
    "support": 450,
    "agent_record": 180,
    "dev_snippet": 200,
    "hinglish": 250,
    "negative": 350,
    "hard": 170,
}
SPLIT_SEEDS = {"dev": 1, "test": 2}


_NAME_SLOT = re.compile(r"\{NAME(?:\.([a-z]+))?\}")
_ADDRESS_SLOT = re.compile(r"\{ADDRESS(?:\.([a-z_]+))?\}")


def name_form(template: str) -> str | None:
    """Which name form a template uses ("lower", "initial"...; plain {NAME} is "mixed"),
    or None if it has no name. Each template uses at most one form."""
    m = _NAME_SLOT.search(template)
    return None if m is None else (m.group(1) or "mixed")


def address_style(template: str) -> str | None:
    """Which address style a template uses ("multiline", "hinglish"...; plain {ADDRESS} is
    "mixed"), or None if it has no address. Each template uses at most one style."""
    m = _ADDRESS_SLOT.search(template)
    return None if m is None else (m.group(1) or "mixed")


def template_split(category: str, index: int) -> str:
    return "dev" if index % 2 == 0 else "test"


def render_template(
    template: str, rng: random.Random, makers: dict[str, Maker] = VALUE_MAKERS,
) -> str:
    """Fill every slot; gold slots become [[TYPE:value]] marks, fillers plain values."""
    def fill(m: re.Match) -> str:
        is_filler, name, variant = m.group(1), m.group(2), m.group(3)
        value = makers[f"{name}.{variant}" if variant else name](rng)
        return value if is_filler else f"[[{name}:{value}]]"

    return _SLOT.sub(fill, template)


def build_split(split: str, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    # names and places come from this split's own pools
    makers = {**VALUE_MAKERS, **name_makers(split), **address_makers(split)}
    samples: list[Sample] = []
    for category, size in SPLIT_SIZES.items():
        templates = [t for i, t in enumerate(TEMPLATES[category])
                     if template_split(category, i) == split]
        for k in range(size):
            template = templates[k % len(templates)]
            text, spans = parse_marked(render_template(template, rng, makers))
            samples.append(Sample(f"{split}-{len(samples) + 1:05d}", category, template,
                                  text, spans))
    return samples
