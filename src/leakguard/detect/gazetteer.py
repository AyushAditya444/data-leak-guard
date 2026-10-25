"""Indian places and PIN codes, from India Post's All India Pincode Directory.

The raw directory (~165,000 post offices, ~23 MB CSV) is turned once into a compact,
gzipped file bundled with the package:

    python -m leakguard.detect.gazetteer build --csv data/raw/<file>.csv

At runtime `load_gazetteer()` reads that file and answers two questions for the address
detector: "is this 6-digit number a real PIN?" and "where are the place names in this
text?". A place name on its own is never masked; it is one signal among several.
"""

import argparse
import csv
import gzip
import io
import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import Literal

PlaceKind = Literal["locality", "city", "district", "state"]

BUNDLE = "data/pin_directory.tsv.gz"
STOPWORDS = "data/place_stopwords.txt"

# When one name is several kinds of place ("Pune" is a city and a district), keep the
# broadest. Only used for debugging and the report; any kind counts as a place.
_KIND_RANK: dict[str, int] = {"locality": 0, "district": 1, "city": 2, "state": 3}

# Common names of big cities that are not (or not exactly) district names, and old names
# people still type. Both spellings are added as cities.
CITY_ALIASES: tuple[str, ...] = (
    "Bengaluru", "Bangalore", "Mumbai", "Bombay", "Navi Mumbai", "Thane", "Gurugram",
    "Gurgaon", "Noida", "Greater Noida", "Ghaziabad", "Kolkata", "Calcutta", "Chennai",
    "Madras", "Kochi", "Cochin", "Mysuru", "Mysore", "Puducherry", "Pondicherry",
    "Prayagraj", "Allahabad", "Vadodara", "Baroda", "Thiruvananthapuram", "Trivandrum",
    "Varanasi", "Banaras", "Delhi", "New Delhi", "Hyderabad", "Secunderabad", "Pune",
    "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore", "Bhopal", "Nagpur", "Patna",
)

# Post office names end in an office type ("Kothrud S.O", "Baner B.O", "Andheri H.O"),
# sometimes followed by the district ("Laxmi Nagar SO East Delhi"). Everything from that
# word on is dropped.
_OFFICE_TYPE = re.compile(
    r"\s+(?:S\.?\s?O|B\.?\s?O|H\.?\s?O|G\.?\s?P\.?\s?O|D\.?\s?S\.?\s?O|H\.?\s?Q)\.?(?:\s.*)?$",
    re.IGNORECASE)
_BRACKETS = re.compile(r"\([^)]*\)?")
_WORD = re.compile(r"[A-Za-z0-9]+")

# A direction or "Extension" right after a place is part of it: "Andheri West".
_SUFFIX = re.compile(r"[ -]+(?:east|west|north|south|extension|extn|ext)\b\.?", re.IGNORECASE)

MAX_WORDS = 5  # longest place name looked for, in words


@dataclass(frozen=True)
class PostOffice:
    pin: str
    office: str
    district: str
    state: str


def normalise_office(name: str) -> str:
    """"Kothrud S.O" -> "Kothrud", "Andheri (West) S.O" -> "Andheri"."""
    name = _BRACKETS.sub(" ", name)
    name = _OFFICE_TYPE.sub("", " ".join(name.split()))
    return " ".join(name.split()).title()


def _key(name: str) -> str:
    """Lookup key: lowercase words joined by single spaces."""
    return " ".join(w.lower() for w in _WORD.findall(name))


@dataclass(frozen=True)
class Gazetteer:
    pins: frozenset[str]
    places: dict[str, PlaceKind]
    offices: tuple[PostOffice, ...]
    max_words: int = MAX_WORDS

    def find_places(self, text: str) -> list[tuple[int, int, PlaceKind]]:
        """Place names in `text`: whole words, any case, longest match first, no overlaps.
        Words of one place may only be separated by spaces ("Kothrud, Pune" is two)."""
        words = list(_WORD.finditer(text))
        found: list[tuple[int, int, PlaceKind]] = []
        i = 0
        while i < len(words):
            match = None
            for n in range(min(self.max_words, len(words) - i), 0, -1):
                group = words[i:i + n]
                if any(text[a.end():b.start()].strip(" ") for a, b in pairwise(group)):
                    continue  # something other than spaces between the words
                kind = self.places.get(" ".join(w.group().lower() for w in group))
                if kind is not None:
                    match = (n, kind)
                    break
            if match is None:
                i += 1
                continue
            n, kind = match
            start, end = words[i].start(), words[i + n - 1].end()
            suffix = _SUFFIX.match(text, end)
            if suffix:
                end = suffix.end()
            found.append((start, end, kind))
            while i < len(words) and words[i].start() < end:
                i += 1
        return found


def _read_stopwords() -> frozenset[str]:
    text = files("leakguard").joinpath(STOPWORDS).read_text(encoding="utf-8")
    return frozenset(_key(line) for line in text.splitlines()
                     if line.strip() and not line.startswith("#"))


def _read_bundle() -> list[PostOffice]:
    raw = gzip.decompress(files("leakguard").joinpath(BUNDLE).read_bytes()).decode("utf-8")
    return [PostOffice(*line.split("\t")) for line in raw.splitlines()[1:] if line]


def _usable(key: str, stopwords: frozenset[str]) -> bool:
    # Very short names ("Ra", "Mo") and numbers match far too much ordinary text.
    return len(key) >= 3 and not key.isdigit() and key not in stopwords


@cache
def load_gazetteer() -> Gazetteer:
    offices = tuple(_read_bundle())
    stopwords = _read_stopwords()
    places: dict[str, PlaceKind] = {}

    def add(name: str, kind: PlaceKind) -> None:
        key = _key(name)
        if not _usable(key, stopwords) or len(key.split()) > MAX_WORDS:
            return
        if key not in places or _KIND_RANK[kind] > _KIND_RANK[places[key]]:
            places[key] = kind

    for office in offices:
        add(office.office, "locality")
        add(office.district, "district")
        add(office.state, "state")
    for city in CITY_ALIASES:
        add(city, "city")
    return Gazetteer(frozenset(o.pin for o in offices), places, offices)


# --- building the bundle (offline, once) -----------------------------------------------

def build_bundle(csv_path: Path, out_path: Path) -> list[PostOffice]:
    """Raw directory CSV -> deduplicated, sorted TSV, gzipped. Rows without a valid PIN
    are dropped; a missing district or state ("NA") is kept as "", so the PIN still
    counts."""
    seen: set[PostOffice] = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pin = row["pincode"].strip()
            if not re.fullmatch(r"[1-9]\d{5}", pin):
                continue
            district, state = (("" if v.strip() == "NA" else v.strip().title())
                               for v in (row["district"], row["statename"]))
            office = normalise_office(row["officename"])
            if office:
                seen.add(PostOffice(pin, office, district, state))
    rows = sorted(seen, key=lambda o: (o.pin, o.office))
    buffer = io.StringIO()
    buffer.write("pin\toffice\tdistrict\tstate\n")
    for o in rows:
        buffer.write(f"{o.pin}\t{o.office}\t{o.district}\t{o.state}\n")
    # mtime=0 keeps the file byte-identical across rebuilds of the same CSV
    out_path.write_bytes(gzip.compress(buffer.getvalue().encode("utf-8"), mtime=0))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m leakguard.detect.gazetteer")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="turn the raw directory CSV into the bundled file")
    build.add_argument("--csv", required=True, type=Path)
    build.add_argument("--out", type=Path, default=Path("src/leakguard") / BUNDLE)
    args = parser.parse_args()
    rows = build_bundle(args.csv, args.out)
    print(f"wrote {args.out}: {len(rows)} offices, {len({o.pin for o in rows})} PINs, "
          f"{len({o.district for o in rows})} districts, "
          f"{args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
