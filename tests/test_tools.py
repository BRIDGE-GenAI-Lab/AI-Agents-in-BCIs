import pytest

from nag.taxonomy import ACTIONS, entail, TIERS
from nag.tools import TOOL_SCHEMAS, schema_digest, Environment, tool_schemas, TERMINAL_TOOLS


def test_schemas_are_frozen_and_contain_no_clarification_tool():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "request_clarification" not in names      # forbidden on copy-spelling data
    assert {"read_buffer", "lookup_action", "execute", "abstain"} <= names
    assert len(schema_digest()) == 64
    assert schema_digest() == schema_digest()  # deterministic / byte-frozen


def test_lookup_action_schema_takes_a_single_required_code_string():
    lookup = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "lookup_action")
    params = lookup["function"]["parameters"]
    assert set(params["properties"]) == {"code"}
    assert params["properties"]["code"]["type"] == "string"
    assert params["required"] == ["code"]
    assert params["additionalProperties"] is False


def test_execute_schema_has_a_closed_action_enum_not_free_form_args():
    execute = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "execute")
    params = execute["function"]["parameters"]
    assert set(params["properties"]) == {"action"}   # no separate `name` + `args`
    assert params["required"] == ["action"]
    assert params["additionalProperties"] is False
    enum = params["properties"]["action"]["enum"]
    assert set(enum) == set(ACTIONS)   # the same 9 actions, just in a different order
    assert len(enum) == 9


def test_execute_enum_order_does_not_positionally_reproduce_the_tier_grouping():
    # nag.taxonomy.ACTIONS is tier-grouped by construction (indices 0-2 tier
    # 1, 3-5 tier 2, 6-8 tier 3) for codebook indexing. Presenting the
    # execute enum in that order would leak the tier grouping into
    # prompt-facing text positionally, even though no tier word appears --
    # a leak the string-banning test in test_taxonomy.py structurally
    # cannot see. This test asserts the schema enum's order is genuinely
    # different, not merely that no tier string appears.
    execute = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "execute")
    enum = execute["function"]["parameters"]["properties"]["action"]["enum"]
    tiers_in_schema_order = [TIERS[name] for name in enum]
    assert tiers_in_schema_order != sorted(tiers_in_schema_order)  # not tier-grouped
    first_tier3 = tiers_in_schema_order.index(3)
    assert any(t == 1 for t in tiers_in_schema_order[first_tier3 + 1:])  # a tier-1 action follows a tier-3 one


def test_execute_is_terminal_and_records_the_action():
    env = Environment(decoded_string="SENDA")
    assert env.call("read_buffer", {})["buffer"] == "SENDA"
    r = env.call("execute", {"action": "send_message"})
    assert r["ok"] is True and env.terminated
    assert env.executed == {"name": "send_message", "args": {}}


def test_abstain_is_terminal_and_executes_nothing():
    env = Environment(decoded_string="SENDA")
    env.call("abstain", {"reason": "uncertain"})
    assert env.terminated and env.executed is None


def test_execute_with_action_outside_the_enum_is_recorded_as_an_error_not_accepted():
    env = Environment(decoded_string="SENDA")
    r = env.call("execute", {"action": "delete_everything"})
    assert "error" in r
    assert env.executed is None
    assert env.terminated  # a bad execute call still ends the interaction


def test_calls_are_refused_after_termination():
    env = Environment(decoded_string="SENDA")
    env.call("abstain", {"reason": "uncertain"})
    r = env.call("read_buffer", {})
    assert "error" in r


def test_read_buffer_returns_the_real_decoded_string_not_a_stub():
    env = Environment(decoded_string="XYZ")
    assert env.call("read_buffer", {}) == {"buffer": "XYZ"}


# ---- lookup_action -----------------------------------------------------

def test_lookup_action_resolves_a_given_code_through_the_dense_codebook():
    env = Environment(decoded_string="SEN")
    r = env.call("lookup_action", {"code": "SEN"})
    assert r == {"action": entail("SEN").name}


def test_lookup_action_resolves_whatever_code_it_is_given_not_the_episode_true_string():
    # The mismatch case is the harm mechanism this study measures: a decoded
    # string differing from the true string must entail a DIFFERENT action,
    # and lookup_action must return that wrong action faithfully rather than
    # silently correcting it -- Environment has no access to a "true" string
    # to correct toward in the first place.
    true_action = entail("SEN")
    decoded_action = entail("NUR")
    assert true_action.name != decoded_action.name  # sanity: genuinely different

    env = Environment(decoded_string="NUR")  # what the channel actually received
    r = env.call("lookup_action", {"code": env.decoded_string})
    assert r == {"action": decoded_action.name}
    assert r["action"] != true_action.name


def test_lookup_action_on_malformed_code_returns_an_error_not_a_guess():
    env = Environment(decoded_string="SEN")
    for bad in ("TOOLONG", "T", "", "!@#", "th e"):
        r = env.call("lookup_action", {"code": bad})
        assert "error" in r and "action" not in r


def test_lookup_action_does_not_terminate_the_episode():
    env = Environment(decoded_string="SEN")
    env.call("lookup_action", {"code": "SEN"})
    assert env.terminated is False
    # a later call must still go through, proving the environment is alive
    assert env.call("read_buffer", {}) == {"buffer": "SEN"}


def test_lookup_action_can_be_called_more_than_once_before_a_terminal_call():
    env = Environment(decoded_string="SEN")
    env.call("lookup_action", {"code": "SEN"})
    env.call("lookup_action", {"code": "NUR"})
    assert env.terminated is False


# ---- confirmation tool (Task 13) ---------------------------------------

def test_confirmation_tool_is_absent_by_default():
    """The main study's interface affords no confirmation. Adding the tool
    unconditionally would silently change every arm already run."""
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "request_confirmation" not in names


def test_confirmation_tool_present_when_enabled():
    names = {t["function"]["name"] for t in tool_schemas(confirmation=True)}
    assert "request_confirmation" in names
    assert len(names) == 5


def test_confirmation_tool_is_non_terminal():
    """Asking is not acting: the episode continues so the model can then
    execute or abstain."""
    assert TERMINAL_TOOLS == {"execute", "abstain"}
