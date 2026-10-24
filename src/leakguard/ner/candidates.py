"""The pretrained NER models compared for name detection, pinned to exact versions.

Each revision is a Hugging Face commit sha, so `python -m leakguard.ner.export` downloads
exactly the weights that were measured, even if the author later updates the model.

Left out on purpose: ai4bharat/IndicNER is gated (downloading needs a logged-in account
that accepted its terms), so a stranger couldn't build the guard with one command.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    key: str
    repo: str
    revision: str
    note: str
    threshold: float = 0.5  # minimum mean person-score for a name; tuned on dev


CANDIDATES: dict[str, Candidate] = {c.key: c for c in (
    Candidate("distilbert", "dslim/distilbert-NER",
              "dfa2838a127384aabb82ed7719e16dab84c42a2a", "small, fast English model"),
    Candidate("bert-base", "dslim/bert-base-NER",
              "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc", "standard English baseline"),
    Candidate("xlmr-hrl", "Davlan/xlm-roberta-base-ner-hrl",
              "253f557bd8249b8515114cfd7f71974fe5fa4d2f", "multilingual (10 languages)"),
    Candidate("xlmr-indian", "Venkatesh4342/NER-Indian-xlm-roberta",
              "74ca62e07c712a42e3e49480fc6ee3ac69e0cfce", "trained on Indian-context text"),
    Candidate("muril-naamapadam", "vaibhav006/indic-ner-muril-naamapadam",
              "e5886d290e6d407a71a83730bc47de20fcd37ee9",
              "India-aware: MuRIL base, Naamapadam data"),
)}

# The model the guard ships with: chosen by `python -m leakguard.ner.compare` on dev.
DEFAULT_CANDIDATE = "distilbert"
