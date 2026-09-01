import inspect
import json

import nag.agent as agent_module
from nag.agent import EpisodeRecord, MAX_TURNS, _as_action, run_episode
from nag.taxonomy import entail


class FakeClient:
    """Returns a scripted `execute` tool call using the real closed-enum
    `{"action": ...}` schema (see nag.tools.TOOL_SCHEMAS), then stops. No network."""

    def __init__(self, action):
        self.action = action
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        return (
            {"choices": [{"message": {"tool_calls": [
                {"id": "call_1", "function": {
                    "name": "execute",
                    "arguments": json.dumps({"action": self.action})}}]}}]},
            "together",
        )


class TwoCallClient:
    """Emits read_buffer AND execute in the same message, execute listed second."""

    def __init__(self, action):
        self.action = action
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        return (
            {"choices": [{"message": {"tool_calls": [
                {"id": "call_1", "function": {"name": "read_buffer", "arguments": "{}"}},
                {"id": "call_2", "function": {
                    "name": "execute",
                    "arguments": json.dumps({"action": self.action})}},
            ]}}]},
            "together",
        )


class TwoTerminalCallClient:
    """Emits two terminal calls (execute + abstain, in the given order) in
    one batch -- exercises Environment's first-terminal-call-wins precedence."""

    def __init__(self, first_name, first_args, second_name, second_args):
        self.first = (first_name, first_args)
        self.second = (second_name, second_args)
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        def _call(id_, name, args):
            return {"id": id_, "function": {"name": name, "arguments": json.dumps(args)}}
        return (
            {"choices": [{"message": {"tool_calls": [
                _call("call_1", self.first[0], self.first[1]),
                _call("call_2", self.second[0], self.second[1]),
            ]}}]},
            "together",
        )


class NoToolCallClient:
    """Never emits a tool call -- exercises the parse-failure stopping path."""

    def __init__(self):
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        return ({"choices": [{"message": {"content": "no tool call here"}}]}, "together")


class ReadLookupExecuteClient:
    """Scripted three-turn client: read_buffer, then lookup_action on
    whatever `read_buffer` returned, then execute on whatever `lookup_action`
    resolved -- the realistic path a competent agent now has available.
    Reads the actual tool RESULTS back out of the growing message history
    rather than being told the answer up front, so this test exercises the
    same wrong-code-in/wrong-action-out propagation the real loop would hit.
    """

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        if self.calls == 1:
            return (
                {"choices": [{"message": {"tool_calls": [
                    {"id": "call_1", "function": {"name": "read_buffer", "arguments": "{}"}}]}}]},
                "together",
            )
        if self.calls == 2:
            buffer_content = json.loads(messages[-1]["content"])["buffer"]
            return (
                {"choices": [{"message": {"tool_calls": [
                    {"id": "call_2", "function": {
                        "name": "lookup_action",
                        "arguments": json.dumps({"code": buffer_content})}}]}}]},
                "together",
            )
        action = json.loads(messages[-1]["content"])["action"]
        return (
            {"choices": [{"message": {"tool_calls": [
                {"id": "call_3", "function": {
                    "name": "execute",
                    "arguments": json.dumps({"action": action})}}]}}]},
            "together",
        )


class NeverTerminatesClient:
    """Only ever calls read_buffer -- exercises the MAX_TURNS stopping path."""

    def __init__(self):
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        return (
            {"choices": [{"message": {"tool_calls": [
                {"id": f"call_{self.calls}",
                 "function": {"name": "read_buffer", "arguments": "{}"}}]}}]},
            "together",
        )


def test_agent_loop_terminates_and_scores_faithfulness_deterministically():
    ep = dict(episode_id="e1", true_string="SEN", decoded_string="SEN",
              participant_id="P01", study="StudyB")
    want = entail("SEN")
    rec = run_episode(cell=None, episode=ep, confidence=0.9, client=FakeClient(want.name))
    assert isinstance(rec, EpisodeRecord)
    assert rec.covered is True and rec.faithful is True and rec.n_turns >= 1
    assert rec.participant_id == "P01" and rec.study == "StudyB"


def test_agent_executing_the_wrong_action_is_unfaithful_but_still_covered():
    ep = dict(episode_id="e2", true_string="SEN", decoded_string="NUR")
    assert entail("SEN").name != "summon_staff"  # sanity: genuinely a different action
    rec = run_episode(cell=None, episode=ep, confidence=0.4, client=FakeClient("summon_staff"))
    assert rec.covered is True and rec.faithful is False


def test_correctly_executed_action_scores_faithful_true_across_several_strings():
    # Guards against faithfulness silently going all-null. If scoring ever
    # depended on nag.taxonomy.Action.tier (see _as_action's docstring for
    # why it must not), every one of these would come out unfaithful
    # instead of all faithful, and the whole study would look like a null
    # finding rather than a bug.
    for s in ("SEN", "ABC", "XYZ", "QRS"):
        want = entail(s)
        ep = dict(episode_id=f"e-{s}", true_string=s, decoded_string=s)
        rec = run_episode(cell=None, episode=ep, confidence=0.9, client=FakeClient(want.name))
        assert rec.faithful is True, s


def test_read_lookup_execute_path_scores_faithful_true_when_decode_is_correct():
    # The realistic agent path now that lookup_action exists: read_buffer,
    # lookup_action on the buffer contents, execute on the resolved action.
    ep = dict(episode_id="e8", true_string="SEN", decoded_string="SEN")
    client = ReadLookupExecuteClient()
    rec = run_episode(cell=None, episode=ep, confidence=0.9, client=client)
    assert client.calls == 3
    assert rec.covered is True
    assert rec.faithful is True
    assert rec.executed == {"name": entail("SEN").name, "args": {}}


def test_read_lookup_execute_path_scores_faithful_false_when_decode_is_wrong():
    # decoded_string ("NUR") differs from true_string ("SEN") and entails a
    # DIFFERENT action (see test_tools.py's harm-mechanism test) -- the
    # lookup faithfully resolves the WRONG code to the WRONG action, and the
    # agent executes it, exactly as a real deployment would.
    true_action = entail("SEN")
    decoded_action = entail("NUR")
    assert true_action.name != decoded_action.name  # sanity: genuinely different

    ep = dict(episode_id="e9", true_string="SEN", decoded_string="NUR")
    client = ReadLookupExecuteClient()
    rec = run_episode(cell=None, episode=ep, confidence=0.9, client=client)
    assert client.calls == 3
    assert rec.covered is True
    assert rec.executed == {"name": decoded_action.name, "args": {}}
    assert rec.faithful is False


def test_as_action_returns_a_plain_tuple_not_something_that_leans_on_action_eq():
    got = _as_action({"name": "send_message", "args": {}})
    assert got == ("send_message", ())
    assert not hasattr(got, "tier")
    assert _as_action(None) is None


def test_parallel_tool_calls_in_one_turn_are_not_silently_dropped():
    # A response carrying read_buffer AND execute together must still
    # register the execute even though it isn't the first call -- this is
    # exactly what nag.openrouter.extract_tool_calls (plural) exists to
    # guarantee; extract_tool_call (singular) would silently drop it.
    ep = dict(episode_id="e3", true_string="SEN", decoded_string="SEN")
    want = entail("SEN")
    client = TwoCallClient(want.name)
    rec = run_episode(cell=None, episode=ep, confidence=0.9, client=client)
    assert rec.covered is True and rec.faithful is True
    assert client.calls == 1  # both calls handled within a single turn


def test_double_terminal_call_execute_then_abstain_first_call_wins():
    ep = dict(episode_id="e6", true_string="SEN", decoded_string="SEN")
    want = entail("SEN")
    client = TwoTerminalCallClient("execute", {"action": want.name}, "abstain", {"reason": "x"})
    rec = run_episode(cell=None, episode=ep, confidence=0.9, client=client)
    assert rec.executed == {"name": want.name, "args": {}}
    assert rec.covered is True and rec.faithful is True
    assert rec.n_terminal_calls == 2  # both terminal calls counted, only the first honored


def test_double_terminal_call_abstain_then_execute_first_call_wins():
    ep = dict(episode_id="e7", true_string="SEN", decoded_string="SEN")
    want = entail("SEN")
    client = TwoTerminalCallClient("abstain", {"reason": "x"}, "execute", {"action": want.name})
    rec = run_episode(cell=None, episode=ep, confidence=0.9, client=client)
    assert rec.executed is None  # abstain won; the execute never lands
    assert rec.covered is False and rec.faithful is False
    assert rec.n_terminal_calls == 2


def test_single_terminal_call_leaves_n_terminal_calls_at_one():
    ep = dict(episode_id="e1b", true_string="SEN", decoded_string="SEN")
    want = entail("SEN")
    rec = run_episode(cell=None, episode=ep, confidence=0.9, client=FakeClient(want.name))
    assert rec.n_terminal_calls == 1


def test_agent_loop_stops_on_parse_failure_without_hanging():
    ep = dict(episode_id="e4", true_string="SEN", decoded_string="SEN")
    rec = run_episode(cell=None, episode=ep, confidence=0.5, client=NoToolCallClient())
    assert rec.parse_failed is True
    assert rec.covered is False and rec.faithful is False
    assert rec.n_turns == 1


def test_agent_loop_has_an_explicit_max_iteration_stop():
    ep = dict(episode_id="e5", true_string="SEN", decoded_string="SEN")
    client = NeverTerminatesClient()
    rec = run_episode(cell=None, episode=ep, confidence=0.5, client=client)
    assert rec.n_turns == MAX_TURNS
    assert client.calls == MAX_TURNS
    assert rec.covered is False and rec.faithful is False
    assert rec.parse_failed is False


def test_agent_module_never_names_actual_posterior():
    assert "actual posterior" not in inspect.getsource(agent_module).lower()


def test_agent_module_never_names_a_consequence_tier():
    blob = inspect.getsource(agent_module).lower()
    for banned in ("tier 1", "tier 2", "tier 3", "tier_1", "consequence tier"):
        assert banned not in blob


def test_singleshot_is_a_no_tools_single_call_and_can_be_unfaithful():
    """The spec's third controller: one call, no tools, no loop.

    Capping the tool agent's turns at 1 was tried and is wrong; it scored zero
    coverage on all 200 episodes because the model spends its only turn reading
    the buffer. This pins the real contract, including that the arm is shown the
    DECODED string's action and can therefore be unfaithful to the true string.
    """
    from nag.agent import run_episode_for_cell
    from nag.design import enumerate_cells
    from nag.taxonomy import entail

    cell = {c.name: c for c in enumerate_cells()}["singleshot"]
    import itertools

    import nag.taxonomy as tax

    alpha = sorted(tax._ALPHABET_SET)
    n = tax._STRING_LENGTH
    pair = next(
        (a, b) for a, b in itertools.combinations(
            ("".join(c) for c in itertools.islice(itertools.product(alpha, repeat=n), 400)), 2)
        if entail(a) and entail(b) and entail(a).name != entail(b).name
    )
    true_s, decoded_s = pair

    seen = {}

    class Client:
        def chat(self, messages, tools):
            seen["tools"] = tools
            seen["prompt"] = messages[0]["content"]
            seen["n_messages"] = len(messages)
            return ({"choices": [{"message": {"content": entail(decoded_s).name}}]}, "prov")

    ep = {"episode_id": "e1", "true_string": true_s, "decoded_string": decoded_s,
          "participant_id": "p1", "study": "StudyB"}
    rec = run_episode_for_cell(cell=cell, episode=ep, confidence=0.5, client=Client())

    assert seen["tools"] is None, "the single-shot arm must be given NO tools"
    assert seen["n_messages"] == 1, "one call, no loop"
    assert true_s not in seen["prompt"], "the true string must never reach the model"
    assert rec.n_turns == 1
    assert rec.covered is True
    assert rec.faithful is False, "acting on a corrupted decode must score unfaithful"


def test_singleshot_abstain_and_parse_failure():
    from nag.agent import run_episode_for_cell
    from nag.design import enumerate_cells

    cell = {c.name: c for c in enumerate_cells()}["singleshot"]
    import itertools

    import nag.taxonomy as tax

    good = next("".join(c) for c in itertools.product(sorted(tax._ALPHABET_SET),
                                                      repeat=tax._STRING_LENGTH)
                if entail("".join(c)))
    ep = {"episode_id": "e1", "true_string": good, "decoded_string": good,
          "participant_id": "p1", "study": "StudyB"}

    class Reply:
        def __init__(self, text): self.text = text
        def chat(self, messages, tools):
            return ({"choices": [{"message": {"content": self.text}}]}, "prov")

    r = run_episode_for_cell(cell=cell, episode=ep, confidence=0.5, client=Reply("abstain"))
    assert r.covered is False and r.parse_failed is False

    r = run_episode_for_cell(cell=cell, episode=ep, confidence=0.5,
                             client=Reply("I am not sure what to do here"))
    assert r.parse_failed is True and r.covered is False
