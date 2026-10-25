"""Synthetic values for the evaluation set. None of these belong to real people.

Identifiers are generated *valid* (Verhoeff for Aadhaar, Luhn for cards, real PAN
structure, IFSC with its reserved 0) so detectors can't reject them for being obviously
fake. Filler values (order IDs, tickets...) are deliberately NOT filtered: a random
12-digit order number passes the Aadhaar checksum ~10% of the time in real life too,
and the evaluation should show that.

Every maker takes a random.Random, so a fixed seed gives the same dataset every time.
"""

import random
import re
import string
import uuid
import zlib
from collections.abc import Callable
from functools import cache

from leakguard.detect.checksums import luhn_valid
from leakguard.detect.contact import UPI_HANDLES
from leakguard.detect.gazetteer import CITY_ALIASES, PostOffice, load_gazetteer

Maker = Callable[[random.Random], str]

# Verhoeff tables (same as leakguard.detect.checksums) plus the inverse table needed to
# *compute* a check digit rather than only validate one.
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_check_digit(digits: str) -> str:
    check = 0
    for position, char in enumerate(reversed(digits)):
        check = _D[check][_P[(position + 1) % 8][int(char)]]
    return str(_INV[check])


# --- small vocabularies ---------------------------------------------------------------

FIRST_NAMES = ("Rahul", "Priya", "Amit", "Sneha", "Arjun", "Kavya", "Rohan", "Ananya",
               "Vikram", "Pooja", "Suresh", "Lakshmi", "Imran", "Fatima", "Harpreet",
               "Deepak", "Neha", "Karthik", "Meera", "Aditya", "Divya", "Sanjay", "Ritu",
               "Abdul", "Ayesha", "Joseph", "Mary", "Venkatesh", "Swati", "Gurpreet")
LAST_NAMES = ("Sharma", "Verma", "Patel", "Reddy", "Iyer", "Nair", "Singh", "Khan", "Gupta",
              "Mehta", "Das", "Banerjee", "Joshi", "Kulkarni", "Pillai", "Menon", "Rao",
              "Shaikh", "Fernandes", "Mishra", "Yadav", "Choudhary", "Agarwal", "Bose")
EMAIL_DOMAINS = ("gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com")
IFSC_BANKS = ("HDFC", "SBIN", "ICIC", "UTIB", "KKBK", "PUNB", "BARB", "CNRB", "IDIB", "YESB",
              "UBIN", "IOBA")
BANK_NAMES = ("HDFC", "SBI", "ICICI", "Axis", "Kotak", "PNB", "Canara", "Bank of Baroda")
UNKNOWN_UPI_HANDLES = ("okbizaxis", "axisb", "hsbc", "dbs", "rbl", "aubank", "kmbl", "idbi")
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
WORDS = ("monsoon", "tiger", "sunshine", "river", "mango", "cricket", "lotus", "falcon",
         "garden", "silver", "thunder", "rocket")
CODE_VERBS = ("get", "parse", "load", "update", "build", "validate", "fetch", "render")
CODE_NOUNS = ("Customer", "Invoice", "Account", "Order", "Payment", "Ledger", "Session", "User")
CODE_SUFFIXES = ("Handler", "Service", "Factory", "Controller", "Repository", "Mapper")

# Person names for NAME gold values, split by dataset: test names never appear in dev,
# so the reported numbers come from names no tuning decision ever saw. Spread across
# regions and communities. (FIRST_NAMES/LAST_NAMES above only build email/UPI local parts.)
NAME_POOLS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "dev": (
        ("Rahul", "Priya", "Amit", "Sneha", "Arjun", "Kavya", "Rohan", "Ananya", "Vikram",
         "Pooja", "Suresh", "Lakshmi", "Imran", "Fatima", "Harpreet", "Deepak", "Neha",
         "Karthik", "Meera", "Aditya", "Divya", "Sanjay", "Ritu", "Abdul", "Ayesha", "Joseph",
         "Mary", "Venkatesh", "Swati", "Gurpreet", "Ankit", "Shreya", "Manish", "Nikita",
         "Tanvi", "Rakesh", "Bhavna", "Sachin", "Farhan", "Zoya", "Thomas", "Anita",
         "Balwinder", "Lalremruati", "Tenzin", "Bikash", "Sourav", "Moumita", "Hemant", "Jyoti",
         "Prakash", "Shalini", "Naveen", "Sunita", "Yusuf"),
        ("Sharma", "Verma", "Patel", "Reddy", "Iyer", "Nair", "Singh", "Khan", "Gupta", "Mehta",
         "Das", "Banerjee", "Joshi", "Kulkarni", "Pillai", "Menon", "Rao", "Shaikh",
         "Fernandes", "Mishra", "Yadav", "Choudhary", "Agarwal", "Bose", "Saxena", "Thakur",
         "Gill", "Hazarika", "Sangma", "Pandey", "Chatterjee", "Dsouza"),
    ),
    "test": (
        ("Aarav", "Ishita", "Varun", "Riya", "Siddharth", "Nandini", "Kunal", "Sakshi",
         "Rajesh", "Geeta", "Arvind", "Kamala", "Salman", "Nazia", "Jaspreet", "Mohit",
         "Komal", "Srinivas", "Revathi", "Abhishek", "Anjali", "Vivek", "Preeti", "Rizwan",
         "Sana", "Peter", "Grace", "Murugan", "Ashwini", "Manpreet", "Gaurav", "Isha", "Pankaj",
         "Rupali", "Akash", "Madhuri", "Dinesh", "Kiranjit", "Arif", "Shabana", "Francis",
         "Leela", "Sukhwinder", "Zothanpuii", "Pema", "Debashish", "Arnab", "Sudipta", "Girish",
         "Pallavi", "Ramesh", "Usha", "Harish", "Nasreen", "Tejas"),
        ("Tiwari", "Srivastava", "Desai", "Naidu", "Krishnan", "Kurup", "Sandhu", "Ansari",
         "Jain", "Shah", "Ghosh", "Mukherjee", "Pawar", "Deshpande", "Nambiar", "Warrier",
         "Hegde", "Qureshi", "Pinto", "Dubey", "Chauhan", "Bhatt", "Goel", "Sen", "Kapoor",
         "Rathore", "Grewal", "Bora", "Marak", "Lyngdoh", "Trivedi", "Chakraborty"),
    ),
}
# Names that are also ordinary English words: genuinely hard for a model, used in "hard".
WORDLIKE_NAMES = ("Kiran", "Sunny", "Lucky", "Rose", "Hira", "Pinky", "Honey", "Bunty",
                  "Prince", "Lovely", "Golden", "Daisy")
HONORIFICS = ("Mr.", "Mrs.", "Ms.", "Dr.", "Shri", "Smt.")
FESTIVALS = ("Ganesh Chaturthi", "Guru Nanak Jayanti", "Durga Puja", "Raksha Bandhan",
             "Diwali", "Eid", "Onam", "Makar Sankranti", "Christmas")
BRANDS = ("Paytm", "PhonePe", "Tata Motors", "Maruti Suzuki", "Bajaj Finance", "Flipkart",
          "Amazon Pay", "Reliance Jio")
CITIES = ("Pune", "Chennai", "Lucknow", "Indore", "Guwahati", "Kochi", "Jaipur", "Bhopal")
PRODUCTS = ("Redmi Note 13", "Galaxy M34", "iPhone 15", "OnePlus Nord", "Boat Airdopes",
            "Prestige cooker")

_BASE62 = string.ascii_letters + string.digits
_UPI_HANDLES_SORTED = tuple(sorted(UPI_HANDLES))  # sets of strings iterate in random order


def _digits(rng: random.Random, n: int, first: str = "0123456789") -> str:
    return rng.choice(first) + "".join(rng.choice(string.digits) for _ in range(n - 1))


def _chars(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


# --- gold values -----------------------------------------------------------------------

def aadhaar(rng: random.Random) -> str:
    base = _digits(rng, 11, first="23456789")
    full = base + verhoeff_check_digit(base)
    sep = rng.choice(("", " ", " ", "-"))
    return sep.join((full[:4], full[4:8], full[8:]))


def aadhaar_mixed(rng: random.Random) -> str:
    full = aadhaar(rng).replace(" ", "").replace("-", "")
    return f"{full[:4]} {full[4:8]}-{full[8:]}"


def pan(rng: random.Random) -> str:
    holder = rng.choice("PPPPPPPPCFHT")
    value = (_chars(rng, string.ascii_uppercase, 3) + holder
             + rng.choice(string.ascii_uppercase) + _digits(rng, 4)
             + rng.choice(string.ascii_uppercase))
    return value.lower() if rng.random() < 0.15 else value


def pan_spaced(rng: random.Random) -> str:
    p = pan(rng).upper()
    return f"{p[:5]} {p[5:9]} {p[9]}"


def _mobile(rng: random.Random) -> str:
    return _digits(rng, 10, first="6789")


def phone(rng: random.Random) -> str:
    d = _mobile(rng)
    return rng.choice((d, d, f"+91 {d}", f"+91-{d}", f"0{d}", f"{d[:5]} {d[5:]}",
                       f"+91 {d[:5]} {d[5:]}", f"{d[:5]}-{d[5:]}"))


def phone_words(rng: random.Random) -> str:
    return " ".join(NUMBER_WORDS[int(c)] for c in _mobile(rng))


def phone_dots(rng: random.Random) -> str:
    d = _mobile(rng)
    return f"{d[:5]}.{d[5:]}"


def _local_part(rng: random.Random) -> str:
    first, last = rng.choice(FIRST_NAMES).lower(), rng.choice(LAST_NAMES).lower()
    return rng.choice((f"{first}.{last}", f"{first}{rng.randint(1, 99)}", f"{first}{last[0]}",
                       _mobile(rng)))


def upi(rng: random.Random) -> str:
    return f"{_local_part(rng)}@{rng.choice(_UPI_HANDLES_SORTED)}"


def upi_unknown_handle(rng: random.Random) -> str:
    return f"{_local_part(rng)}@{rng.choice(UNKNOWN_UPI_HANDLES)}"


def email(rng: random.Random) -> str:
    first, last = rng.choice(FIRST_NAMES).lower(), rng.choice(LAST_NAMES).lower()
    local = rng.choice((f"{first}.{last}", f"{first}{rng.randint(1, 999)}", f"{first}_{last}"))
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def card(rng: random.Random) -> str:
    base = "4" + _digits(rng, 14)
    full = next(base + d for d in string.digits if luhn_valid(base + d))
    sep = rng.choice(("", " ", "-"))
    return sep.join(full[i:i + 4] for i in range(0, 16, 4))


def ifsc(rng: random.Random) -> str:
    return rng.choice(IFSC_BANKS) + "0" + _digits(rng, 6)


def bank_account(rng: random.Random) -> str:
    return _digits(rng, rng.randint(9, 18), first="123456789")


def name_makers(split: str) -> dict[str, Maker]:
    """NAME value makers drawing on one split's name pool, one per name form."""
    firsts, lasts = NAME_POOLS[split]

    def full(rng: random.Random) -> str:
        return f"{rng.choice(firsts)} {rng.choice(lasts)}"

    def mixed(rng: random.Random) -> str:  # the original {NAME}: usually full, sometimes first
        return rng.choice(firsts) if rng.random() < 0.2 else full(rng)

    def initial(rng: random.Random) -> str:  # South Indian style: "K. Venkatesh"
        initials = rng.sample(string.ascii_uppercase, rng.choice((1, 1, 2)))
        return " ".join(f"{c}." for c in initials) + " " + rng.choice(firsts)

    def three(rng: random.Random) -> str:
        first, middle = rng.sample(firsts, 2)
        return f"{first} {middle} {rng.choice(lasts)}"

    def wordlike(rng: random.Random) -> str:
        first = rng.choice(WORDLIKE_NAMES)
        return first if rng.random() < 0.5 else f"{first} {rng.choice(lasts)}"

    return {
        "NAME": mixed,
        "NAME.first": lambda rng: rng.choice(firsts),
        "NAME.lower": lambda rng: mixed(rng).lower(),
        "NAME.surname": lambda rng: rng.choice(lasts),
        "NAME.initial": initial,
        "NAME.three": three,
        "NAME.caps": lambda rng: full(rng).upper(),
        "NAME.wordlike": wordlike,
    }

# --- addresses -------------------------------------------------------------------------
#
# Places come from the real India Post directory, split by district so the test split
# only has places the dev split (used for tuning) never showed. The other vocabularies
# are split too, and were written for variety, not copied from the detector's keyword
# lists: test uses spellings dev never shows ("Apts" vs "Apartments", "nr." vs "near").

# Real-sounding localities that are NOT post office names (new layouts and societies),
# so finding places can't be 100% just because the generator and detector share a list.
INVENTED_LOCALITIES: dict[str, tuple[str, ...]] = {
    "dev": ("Hiranandani Meadows", "Palm Grove Layout", "Green Glen Layout",
            "Sunshine Enclave", "Lakeshore Gardens", "Silver Springs Colony",
            "Royal Palms Estate", "Orchid Valley", "Maple Heights Layout", "Rose Garden Extn",
            "Emerald Isle Township", "Prestige Shantiniketan", "Sobha Dream Acres",
            "Brigade Meadows", "Lodha Palava", "DLF Phase 3", "Amrapali Zodiac",
            "Mahagun Moderne", "Jaypee Greens", "Godrej Garden City", "Kumar Pebble Park",
            "Nyati Elysia"),
    "test": ("Whispering Woods Layout", "Coconut Grove Colony", "Golden Meadows Township",
             "Blue Ridge Estate", "Cedar Park Layout", "Ivory Towers Enclave",
             "Pearl Residency Layout", "Bluebell Gardens", "Willow Creek Colony",
             "Tulip Valley Extn", "Rohan Abhilasha", "Purva Riviera", "Mantri Espana",
             "Supertech Capetown", "Gaur City 2", "ATS Pristine", "Kolte Patil Life Republic",
             "Paranjape Blue Ridge", "Raheja Vistas", "Hiranandani Estate Thane West",
             "Salarpuria Greenage", "Tata Primanti"),
}

_BUILDING_NAMES = {
    "dev": ("Sai Krupa", "Green Valley", "Shanti", "Lakeview", "Sunrise", "Royal", "Ganesh",
            "Laxmi", "Sagar", "Kailash", "Panchsheel", "Ashirwad"),
    "test": ("Om Sai", "Silver Oak", "Gokul", "Mayur", "Vrindavan", "Shivam", "Navkar",
             "Saraswati", "Parijat", "Devashree", "Swapnapurti", "Mangalmurti"),
}
_BUILDING_SUFFIXES = {
    "dev": ("Apartments", "Society", "Residency", "Towers", "Niwas", "Enclave", "Bhavan"),
    "test": ("Apts", "CHS", "Heights", "Sadan", "Complex", "Kunj", "Villas"),
}
_STREETS = {
    "dev": ("MG Road", "Station Road", "Nehru Marg", "2nd Cross Road", "Link Road",
            "Tilak Road", "gali no {n}", "Sector {n}", "Block {L}"),
    "test": ("FC Road", "Gandhi Path", "Ring Rd", "5th Main", "Church Street",
             "Netaji Subhash Marg", "lane no {n}", "Phase {n}", "{n}th Cross"),
}
_LANDMARK_CUES = {"dev": ("near", "opp.", "behind"),
                  "test": ("nr.", "opposite", "next to", "beside")}
_LANDMARK_THINGS = {
    "dev": ("Hanuman Mandir", "SBI ATM", "City Mall", "the Gurudwara", "Govt School",
            "Axis Bank", "Shivaji Park"),
    "test": ("Shiv Temple", "HDFC Bank", "Big Bazaar", "Jama Masjid", "the petrol pump",
             "the bus stand", "St. Mary's Church"),
}
_POSTPOSITIONS = ("ke paas", "ke saamne", "ke peeche")
_UNITS = ("Flat {n}", "Flat No. {n}", "H.No. {n}", "House No {n}", "Plot {n}", "{a}/{b}",
          "{L}-{n}", "#{n}", "Door No {a}-{b}", "Room {n}", "Shop No. {a}")
_ABBREV_UNITS = ("Flat No. {n}", "H.No. {n}", "Plot No. {a}")

# Every style the generator can produce. "landmark_only" (no place) is not an address by
# the detector's definition: it appears only in the "hard" category, as a known miss.
ADDRESS_STYLES = ("formal_pin", "formal", "landmark", "hinglish", "multiline", "partial",
                  "nocomma", "abbrev", "landmark_only")

_CITY_SUFFIX = re.compile(r"\s+(?:Urban|Rural|Suburban|City)$")


def city_name(district: str) -> str:
    """How people write a district as their city: "Bengaluru Urban" -> "Bengaluru".
    A trailing dot ("Y.S.R.") is dropped: gold spans never end in punctuation."""
    return _CITY_SUFFIX.sub("", district).rstrip(".")


@cache
def place_pools(split: str) -> list[PostOffice]:
    """Post offices whose district belongs to `split` (districts never shared)."""
    want = 0 if split == "dev" else 1
    return [o for o in load_gazetteer().offices
            if o.district and zlib.crc32(o.district.lower().encode()) % 2 == want]


def address_makers(split: str) -> dict[str, Maker]:
    """ADDRESS makers (one per style) and the address look-alike fillers, for one split."""
    pool = place_pools(split)
    # 85% of post offices are rural branch offices, but support chats come mostly from
    # cities, so half the addresses are drawn from big-city districts.
    cities = {c.lower() for c in CITY_ALIASES}
    urban = [o for o in pool if city_name(o.district).lower() in cities]
    invented = INVENTED_LOCALITIES[split]

    def fill(rng: random.Random, pattern: str) -> str:
        return pattern.format(n=rng.randint(1, 120), a=rng.randint(1, 60), b=rng.randint(1, 9),
                              L=rng.choice("ABCDEFGH"))

    def place(rng: random.Random) -> tuple[str, str, str]:
        """(locality, city, PIN): a real office, or now and then an invented locality."""
        office = rng.choice(urban if urban and rng.random() < 0.5 else pool)
        locality = rng.choice(invented) if rng.random() < 0.25 else office.office.rstrip(".")
        return locality, city_name(office.district), office.pin

    def unit(rng):
        return fill(rng, rng.choice(_UNITS))

    def building(rng):
        return f"{rng.choice(_BUILDING_NAMES[split])} {rng.choice(_BUILDING_SUFFIXES[split])}"

    def street(rng):
        return fill(rng, rng.choice(_STREETS[split]))

    def landmark(rng):
        return f"{rng.choice(_LANDMARK_CUES[split])} {rng.choice(_LANDMARK_THINGS[split])}"

    def pin_text(rng, pin: str) -> str:
        return rng.choice((f" {pin}", f" - {pin}", f", {pin}", f", Pin {pin}", f" PIN: {pin}"))

    def case(rng, value: str) -> str:  # people often type in lowercase
        return value.lower() if rng.random() < 0.2 else value

    def formal_parts(rng) -> tuple[list[str], str]:
        locality, city, pin = place(rng)
        middle = rng.choice(([building(rng)], [street(rng)], [building(rng), street(rng)]))
        return [unit(rng), *middle, locality, city], pin

    def formal_pin(rng):
        parts, pin = formal_parts(rng)
        return case(rng, ", ".join(parts) + pin_text(rng, pin))

    def formal(rng):
        return case(rng, ", ".join(formal_parts(rng)[0]))

    def landmark_style(rng):
        locality, city, _ = place(rng)
        lead = [landmark(rng)] if rng.random() < 0.6 else [unit(rng), landmark(rng)]
        return case(rng, ", ".join([*lead, locality, city]))

    def hinglish(rng):
        locality, city, _ = place(rng)
        thing = rng.choice(_LANDMARK_THINGS[split])
        return rng.choice((
            f"H.No. {rng.randint(1, 300)}, gali no {rng.randint(1, 20)}, {locality}",
            f"{thing} {rng.choice(_POSTPOSITIONS)}, {locality}, {city}",
            f"sector {rng.randint(1, 120)}, {city}",
            f"makaan no {rng.randint(1, 300)}, {locality}, {city}",
        )).lower()

    def multiline(rng):
        parts, pin = formal_parts(rng)
        newline = rng.choice(("\n", "\r\n"))
        comma = rng.choice((",", ""))
        return (comma + newline).join(parts) + pin_text(rng, pin).replace(", ", newline)

    def partial(rng):
        locality, _, _ = place(rng)
        return case(rng, f"{unit(rng)}, {locality}")

    def nocomma(rng):
        parts, pin = formal_parts(rng)
        return f"{' '.join(parts)} {pin}".lower()

    def abbrev(rng):
        locality, city, pin = place(rng)
        name = rng.choice(_BUILDING_NAMES[split])
        return (f"{fill(rng, rng.choice(_ABBREV_UNITS))}, {name} Apts, nr. "
                f"{rng.choice(_LANDMARK_THINGS[split])}, {locality}, {city}, Pin-{pin}")

    def landmark_only(rng):
        thing = rng.choice(("metro station", "water tank", "big banyan tree", "old post office"))
        return f"{rng.choice(_LANDMARK_CUES[split])} the {thing}"

    styles: dict[str, Maker] = {
        "formal_pin": formal_pin, "formal": formal, "landmark": landmark_style,
        "hinglish": hinglish, "multiline": multiline, "partial": partial,
        "nocomma": nocomma, "abbrev": abbrev, "landmark_only": landmark_only,
    }
    mixed_styles = ("formal_pin", "formal", "landmark", "hinglish", "partial")

    def mixed(rng):  # plain {ADDRESS}: any common style
        return styles[rng.choice(mixed_styles)](rng)

    def place_filler(rng):  # a place outside any address: "I live in Pune"
        locality, city, _ = place(rng)
        return rng.choice((locality, city))

    return {
        "ADDRESS": mixed,
        **{f"ADDRESS.{style}": maker for style, maker in styles.items()},
        "PLACE": place_filler,
        "PIN_LIKE": lambda rng: rng.choice(pool).pin,  # a real PIN used as OTP / order no
    }


def _random_word_case(rng: random.Random, word: str) -> str:
    return word.capitalize() if rng.random() < 0.7 else word


SECRET_MAKERS: dict[str, Maker] = {
    "openai": lambda rng: "sk-proj-" + _chars(rng, _BASE62 + "-_", 40),
    "aws": lambda rng: "AKIA" + _chars(rng, string.ascii_uppercase + string.digits, 16),
    "github": lambda rng: "ghp_" + _chars(rng, _BASE62, 36),
    "slack": lambda rng: f"xoxb-{_digits(rng, 11)}-{_digits(rng, 12)}-{_chars(rng, _BASE62, 24)}",
    "google": lambda rng: "AIza" + _chars(rng, _BASE62 + "-_", 35),
    "stripe": lambda rng: "sk_live_" + _chars(rng, _BASE62, 24),
    "jwt": lambda rng: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ" + _chars(rng, _BASE62, 30)
                       + "." + _chars(rng, _BASE62 + "-_", 43),
    "pem": lambda rng: rng.choice(("-----BEGIN RSA PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----",
                                   "-----BEGIN OPENSSH PRIVATE KEY-----")),
    "password": lambda rng: _random_word_case(rng, rng.choice(WORDS)) + rng.choice("@#$!_")
                            + _digits(rng, rng.randint(2, 4)),
    "letters": lambda rng: rng.choice(WORDS) + rng.choice(WORDS),
    "token": lambda rng: _chars(rng, _BASE62, 32),
}


# --- filler (never labelled) -----------------------------------------------------------

def _camel(rng: random.Random) -> str:
    return (rng.choice(CODE_VERBS) + rng.choice(CODE_NOUNS) + rng.choice(CODE_NOUNS)
            + rng.choice(CODE_SUFFIXES) + rng.choice(("", "V2", "2024", "")))


FILLER_MAKERS: dict[str, Maker] = {
    "ORDER_ID": lambda rng: _digits(rng, 12, first="123456789"),
    "OTP": lambda rng: _digits(rng, 6),
    "TICKET": lambda rng: _digits(rng, 14, first="123456789"),
    "TIMESTAMP": lambda rng: f"2024{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
                             f"{rng.randint(0, 23):02d}{rng.randint(0, 59):02d}"
                             f"{rng.randint(0, 59):02d}",
    "INVOICE": lambda rng: f"INV-2024-{_digits(rng, 6)}",
    "DATE": lambda rng: f"{rng.randint(1, 28):02d}{rng.choice('-/')}{rng.randint(1, 12):02d}"
                        f"{rng.choice('-/')}2024",
    "AMOUNT": lambda rng: rng.choice(("₹", "Rs. ", "INR ")) + f"{rng.randint(100, 99_999):,}",
    "SHA": lambda rng: _chars(rng, "0123456789abcdef", 40),
    "UUID": lambda rng: str(uuid.UUID(int=rng.getrandbits(128), version=4)),
    "TOLLFREE": lambda rng: f"1800-{_digits(rng, 3)}-{_digits(rng, 4)}",
    "CAMEL": _camel,
    "TRACKING": lambda rng: f"EM{_digits(rng, 9)}IN",
    "COUPON": lambda rng: rng.choice(("SAVE", "DIWALI", "FEST", "WELCOME")) + str(
        rng.choice((10, 20, 50, 100))),
    "BANK": lambda rng: rng.choice(BANK_NAMES),
    "HONORIFIC": lambda rng: rng.choice(HONORIFICS),
    "FESTIVAL": lambda rng: rng.choice(FESTIVALS),
    "BRAND": lambda rng: rng.choice(BRANDS),
    "CITY": lambda rng: rng.choice(CITIES),
    "PRODUCT": lambda rng: rng.choice(PRODUCTS),
}

VALUE_MAKERS: dict[str, Maker] = {
    "AADHAAR": aadhaar,
    "AADHAAR.mixed": aadhaar_mixed,
    "PAN": pan,
    "PAN.spaced": pan_spaced,
    "PHONE": phone,
    "PHONE.words": phone_words,
    "PHONE.dots": phone_dots,
    "UPI": upi,
    "UPI.unknown_handle": upi_unknown_handle,
    "EMAIL": email,
    "CARD": card,
    "IFSC": ifsc,
    "BANK_ACCOUNT": bank_account,
    # Default for ad-hoc rendering; build_split always uses the split's own pool.
    **name_makers("dev"),
    **address_makers("dev"),
    **{f"SECRET.{kind}": maker for kind, maker in SECRET_MAKERS.items()},
    **FILLER_MAKERS,
}
