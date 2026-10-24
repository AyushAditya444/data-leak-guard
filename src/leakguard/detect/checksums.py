"""Check digits. A real Aadhaar or card number must pass its checksum, so a random
number that merely *looks* right is rejected about 9 times out of 10."""

# Verhoeff algorithm (used by Aadhaar): multiplication table of the dihedral group D5
# and a permutation table. Catches every single-digit typo and every swap of two
# neighbouring digits.
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    if not digits.isdigit() or not digits.isascii():
        return False
    check = 0
    for position, char in enumerate(reversed(digits)):
        check = _D[check][_P[position % 8][int(char)]]
    return check == 0


def luhn_valid(digits: str) -> bool:
    """Luhn (used by payment cards): double every second digit from the right,
    add up all the digits, and the total must be divisible by 10."""
    if not digits.isdigit() or not digits.isascii():
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        n = int(char)
        if position % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0
