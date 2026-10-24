import pytest

from leakguard.config import Settings
from leakguard.entities import EntityType as T
from leakguard.policy import DEFAULT_POLICY, Action, FailureMode, PolicyError, load_policy


def test_default_policy():
    assert DEFAULT_POLICY.action_for(T.SECRET) is Action.BLOCK
    assert DEFAULT_POLICY.action_for(T.AADHAAR) is Action.MASK
    assert DEFAULT_POLICY.failure_mode is FailureMode.CLOSED
    assert load_policy(None) == DEFAULT_POLICY


def test_partial_yaml_overrides_defaults(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("actions:\n  NAME: log_only\nfailure_mode: open\n")
    pol = load_policy(p)
    assert pol.action_for(T.NAME) is Action.LOG_ONLY
    assert pol.action_for(T.PHONE) is Action.MASK
    assert pol.failure_mode is FailureMode.OPEN


@pytest.mark.parametrize("yaml_text,bad", [
    ("actions:\n  PASSPORT: mask\n", "PASSPORT"),
    ("actions:\n  PHONE: hide\n", "hide"),
    ("failure_mode: maybe\n", "maybe"),
    ("colour: blue\n", "colour"),
])
def test_invalid_policy_rejected(tmp_path, yaml_text, bad):
    p = tmp_path / "p.yaml"
    p.write_text(yaml_text)
    with pytest.raises(PolicyError, match=bad):
        load_policy(p)


def test_shipped_policy_file_matches_defaults():
    assert load_policy("config/policy.yaml") == DEFAULT_POLICY


def test_settings_policy_path_from_env():
    assert Settings.from_env({"LEAKGUARD_POLICY_PATH": "x.yaml"}).policy_path == "x.yaml"
    assert Settings.from_env({}).policy_path is None
