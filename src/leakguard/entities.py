"""The kinds of sensitive data the guard knows about."""

from enum import StrEnum


class EntityType(StrEnum):
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    PHONE = "PHONE"
    UPI = "UPI"
    EMAIL = "EMAIL"
    CARD = "CARD"
    IFSC = "IFSC"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    SECRET = "SECRET"
    NAME = "NAME"  # detector arrives in Phase 4
    ADDRESS = "ADDRESS"  # detector arrives in Phase 5


# When two detections overlap, the higher priority wins. Examples:
# - a 10-digit number after "a/c" is a BANK_ACCOUNT, not a PHONE (some banks use
#   the customer's mobile number as the account number);
# - "name@okicici" is a UPI ID, not an EMAIL.
PRIORITY: dict[EntityType, int] = {
    EntityType.SECRET: 100,
    EntityType.AADHAAR: 90,
    EntityType.BANK_ACCOUNT: 80,
    EntityType.CARD: 70,
    EntityType.PAN: 65,
    EntityType.IFSC: 65,
    EntityType.UPI: 60,
    EntityType.PHONE: 50,
    EntityType.EMAIL: 40,
    EntityType.ADDRESS: 20,
    EntityType.NAME: 10,
}
