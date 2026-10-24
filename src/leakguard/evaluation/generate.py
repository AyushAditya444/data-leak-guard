"""Builds the seeded dev/test datasets from templates."""

import random
import re

from leakguard.evaluation.samples import Sample, parse_marked
from leakguard.evaluation.templates import TEMPLATES
from leakguard.evaluation.values import VALUE_MAKERS

_SLOT = re.compile(r"\{(~?)([A-Z_]+)(?:\.([a-z_]+))?\}")

# Samples per category, per split (1,500 per split).
SPLIT_SIZES: dict[str, int] = {
    "support": 450,
    "agent_record": 150,
    "dev_snippet": 200,
    "hinglish": 250,
    "negative": 350,
    "hard": 100,
}
SPLIT_SEEDS = {"dev": 1, "test": 2}


def template_split(category: str, index: int) -> str:
    return "dev" if index % 2 == 0 else "test"


def render_template(template: str, rng: random.Random) -> str:
    """Fill every slot; gold slots become [[TYPE:value]] marks, fillers plain values."""
    def fill(m: re.Match) -> str:
        is_filler, name, variant = m.group(1), m.group(2), m.group(3)
        value = VALUE_MAKERS[f"{name}.{variant}" if variant else name](rng)
        return value if is_filler else f"[[{name}:{value}]]"

    return _SLOT.sub(fill, template)


def build_split(split: str, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []
    for category, size in SPLIT_SIZES.items():
        templates = [t for i, t in enumerate(TEMPLATES[category])
                     if template_split(category, i) == split]
        for k in range(size):
            template = templates[k % len(templates)]
            text, spans = parse_marked(render_template(template, rng))
            samples.append(Sample(f"{split}-{len(samples) + 1:05d}", category, template,
                                  text, spans))
    return samples
