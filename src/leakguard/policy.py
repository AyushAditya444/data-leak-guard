"""What the guard does with each kind of sensitive data, loaded from a YAML file."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from leakguard.entities import EntityType


class Action(StrEnum):
    MASK = "mask"  # hide before sending, restore in the reply
    BLOCK = "block"  # reject the whole request
    LOG_ONLY = "log_only"  # let it through, but count it in the logs


class FailureMode(StrEnum):
    CLOSED = "closed"  # if detection breaks, reject the request (safe default)
    OPEN = "open"  # if detection breaks, forward unmasked (keeps the app up, risks a leak)


class PolicyError(ValueError):
    """The policy file is invalid. Raised at startup, never mid-request."""


@dataclass(frozen=True)
class Policy:
    actions: Mapping[EntityType, Action]
    failure_mode: FailureMode = FailureMode.CLOSED

    def action_for(self, entity: EntityType) -> Action:
        return self.actions[entity]


DEFAULT_POLICY = Policy(
    actions={entity: Action.BLOCK if entity is EntityType.SECRET else Action.MASK
             for entity in EntityType},
)

_ALLOWED_KEYS = {"actions", "failure_mode"}


def _parse_enum(enum_cls: type[StrEnum], value: object, what: str) -> StrEnum:
    try:
        return enum_cls(str(value))
    except ValueError:
        allowed = ", ".join(member.value for member in enum_cls)
        raise PolicyError(f"Unknown {what} '{value}'. Allowed: {allowed}.") from None


def load_policy(path: str | os.PathLike | None) -> Policy:
    """Load a policy file; types it doesn't mention keep their default action."""
    if path is None:
        return DEFAULT_POLICY
    try:
        # safe_load only builds plain data (dicts, lists, strings): a malicious YAML file
        # can't make Python construct arbitrary objects.
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PolicyError(f"Policy file {path} is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise PolicyError(f"Policy file {path} must be a mapping at the top level.")

    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise PolicyError(f"Unknown policy key(s): {', '.join(sorted(map(str, unknown)))}.")

    actions = dict(DEFAULT_POLICY.actions)
    raw_actions = raw.get("actions") or {}
    if not isinstance(raw_actions, dict):
        raise PolicyError("'actions' must map entity types to actions.")
    for entity_name, action_name in raw_actions.items():
        entity = _parse_enum(EntityType, entity_name, "entity type")
        actions[entity] = _parse_enum(Action, action_name, "action")

    failure_mode = _parse_enum(FailureMode, raw.get("failure_mode", "closed"), "failure_mode")
    return Policy(actions=actions, failure_mode=failure_mode)
