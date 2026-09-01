import pytest
from nag.sandbox import AssistiveSandbox, TIER3_ACTIONS
from nag.taxonomy import ACTIONS, TIERS


def test_every_taxonomy_action_is_executable_and_changes_state():
    """A missing handler must fail loudly, not silently no-op."""
    for name in ACTIONS:
        sb = AssistiveSandbox()
        assert not sb.changed_from_initial()
        sb.execute(name)
        assert sb.changed_from_initial(), f"{name} executed but changed nothing"


def test_unknown_action_is_refused():
    sb = AssistiveSandbox()
    with pytest.raises(KeyError):
        sb.execute("launch_missiles")


def test_consent_and_refusal_overwrite_the_same_field():
    """The clearest irreversible unintended change: one silently replaces the other."""
    sb = AssistiveSandbox()
    sb.execute("record_consent")
    assert sb.state["preference_record"] == "consent"
    sb.execute("record_refusal")
    assert sb.state["preference_record"] == "refusal"


def test_tier3_set_matches_the_frozen_taxonomy():
    assert TIER3_ACTIONS == frozenset(a for a in ACTIONS if TIERS[a] == 3)
    assert len(TIER3_ACTIONS) == 3


def test_snapshot_does_not_alias_state():
    sb = AssistiveSandbox()
    snap = sb.snapshot()
    sb.execute("save_note")
    assert snap != sb.snapshot(), "snapshot aliased live state"
