"""Differential tests (Ruling 28): same episode, same seed, contrasting cells.

A live probe found that `run_episode`, called directly for every cell,
produced identical behaviour for `factorial:decoder_confidence:advisory:s0`,
`factorial:decoder_confidence:enforced:s0`, and `nonllm_gate` -- Factor B
(control_mechanism) did not exist in the build. Every pre-existing test
either unit-tests a pure function in isolation, or drives `run_episode` with
a scripted fake that returns the SAME tool call regardless of cell -- so
nothing asserted that two cells produce DIFFERENT outcomes on the same
input. This file is that missing category: it exists specifically to prove
`run_episode_for_cell` / `apply_enforced_gate` make control_mechanism do
something, not just that each piece works alone.
"""
import numpy as np
import pytest

from nag.agent import EpisodeRecord, apply_enforced_gate, run_episode_for_cell
from nag.design import Cell, enumerate_cells
from nag.taxonomy import entail

_CELLS = {c.name: c for c in enumerate_cells()}
NONLLM_GATE_CELL = _CELLS["nonllm_gate"]           # knob=confidence, uses_llm=False
RANDOM_GATE_CELL = _CELLS["random_gate"]           # knob=coverage,   uses_llm=False
ADVISORY_TWIN = _CELLS["factorial:decoder_confidence:advisory:s0"]
ENFORCED_TWIN = _CELLS["factorial:decoder_confidence:enforced:s0"]


class PoisonClient:
    """Raises if `.chat()` is ever called. Proves a code path makes ZERO
    API calls by making any call fail loudly, rather than by counting."""

    def chat(self, **kw):
        raise AssertionError("PoisonClient.chat() was called -- a non-LLM cell made an API call")


class ScriptedExecuteClient:
    """Always proposes the same fixed action, regardless of what the system
    prompt says -- a stand-in for "whatever the model would have proposed."
    Records every `messages` list it is given, so a test can inspect what
    system prompt each cell actually produced."""

    def __init__(self, action: str):
        self.action = action
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages, **kw):
        self.seen_messages.append(messages)
        import json
        return (
            {"choices": [{"message": {"tool_calls": [
                {"id": "call_1", "function": {
                    "name": "execute", "arguments": json.dumps({"action": self.action})}}]}}]},
            "together",
        )


def _episode(true_string="SEN", decoded_string="SEN", **kw):
    return dict(episode_id="diff-1", true_string=true_string, decoded_string=decoded_string,
               participant_id="P01", study="StudyB", **kw)


# --- non-LLM cells make zero API calls --------------------------------------

@pytest.mark.parametrize("cell", [NONLLM_GATE_CELL, RANDOM_GATE_CELL])
def test_nonllm_cells_make_zero_api_calls(cell):
    ep = _episode()
    rec = run_episode_for_cell(cell, ep, confidence=0.9, client=PoisonClient())
    assert isinstance(rec, EpisodeRecord)
    assert rec.n_turns == 0  # no LLM turns happened at all


def test_nonllm_gate_via_dispatcher_uses_decoded_string_not_true_string():
    """Mirrors nag.controllers' own pinned warning: fed the TRUE string's
    action the gate can never lose. The dispatcher must feed it the DECODED
    string's entailed action, so a corrupted decode can score unfaithful."""
    true_string, decoded_string = "SEN", "NUR"
    true_entailed = entail(true_string)
    decoded_entailed = entail(decoded_string)
    assert decoded_entailed.name != true_entailed.name  # sanity: a real substitution

    ep = _episode(true_string=true_string, decoded_string=decoded_string)
    rec = run_episode_for_cell(NONLLM_GATE_CELL, ep, confidence=0.99, client=PoisonClient())
    assert rec.covered is True
    assert rec.executed == {"name": decoded_entailed.name, "args": {}}
    assert rec.faithful is False


# --- random_gate hits its target coverage within tolerance ------------------

def test_random_gate_via_dispatcher_hits_target_coverage():
    ep = _episode()
    rng = np.random.default_rng(0)
    covered = [
        run_episode_for_cell(RANDOM_GATE_CELL, ep, confidence=0.5, client=PoisonClient(),
                             rng=rng, coverage=0.3).covered
        for _ in range(4000)
    ]
    assert abs(np.mean(covered) - 0.3) < 0.03


def test_nonllm_default_mode_admits_everything_for_offline_sweeping():
    """Left at threshold=None, the non-LLM path always proposes (when the
    decode resolves) -- the same 'record everything, decide later' point as
    the LLM-mediated enforced cells, so ONE run produces the whole curve."""
    ep = _episode()
    rec = run_episode_for_cell(NONLLM_GATE_CELL, ep, confidence=1e-6, client=PoisonClient())
    assert rec.covered is True


# --- enforced vs advisory: different system prompts -------------------------

def test_enforced_and_advisory_twin_receive_different_system_prompts():
    ep = _episode()
    advisory_client = ScriptedExecuteClient(entail("SEN").name)
    enforced_client = ScriptedExecuteClient(entail("SEN").name)

    run_episode_for_cell(ADVISORY_TWIN, ep, confidence=0.02, client=advisory_client)
    run_episode_for_cell(ENFORCED_TWIN, ep, confidence=0.02, client=enforced_client)

    advisory_system = advisory_client.seen_messages[0][0]["content"]
    enforced_system = enforced_client.seen_messages[0][0]["content"]
    assert "0.02" in advisory_system
    assert "0.02" not in enforced_system
    assert advisory_system != enforced_system


# --- enforced post-hoc gate suppresses where advisory acts ------------------

def test_enforced_post_hoc_gate_suppresses_low_confidence_action_where_advisory_twin_acts():
    """The key differential behaviour Factor B is supposed to produce: at a
    low confidence, an enforced cell's harness suppresses the proposed
    action while its advisory twin's identical proposal stands unmodified.
    """
    ep = _episode()
    action = entail("SEN").name
    low_confidence = 0.02
    threshold = 0.5

    advisory_rec = run_episode_for_cell(
        ADVISORY_TWIN, ep, confidence=low_confidence, client=ScriptedExecuteClient(action))
    enforced_rec = run_episode_for_cell(
        ENFORCED_TWIN, ep, confidence=low_confidence, client=ScriptedExecuteClient(action))

    # Raw proposals are identical -- the scripted client ignores the prompt,
    # and run_episode_for_cell never gates an enforced cell at run time.
    assert advisory_rec.covered is True and advisory_rec.faithful is True
    assert enforced_rec.covered is True and enforced_rec.faithful is True

    # Advisory: the model's own decision IS the final outcome. No gate is
    # ever applied to it.
    final_advisory = advisory_rec
    assert final_advisory.covered is True

    # Enforced: the harness's post-hoc gate, applied at the real threshold,
    # suppresses the low-confidence proposal into an abstention.
    final_enforced = apply_enforced_gate(enforced_rec, threshold=threshold)
    assert final_enforced.covered is False
    assert final_enforced.faithful is False
    assert final_enforced.executed is None

    # The suppressed outcome now diverges from the advisory twin, even
    # though both models proposed the identical action.
    assert final_advisory.covered != final_enforced.covered


def test_enforced_post_hoc_gate_admits_high_confidence_proposal_unchanged():
    ep = _episode()
    action = entail("SEN").name
    rec = run_episode_for_cell(ENFORCED_TWIN, ep, confidence=0.95, client=ScriptedExecuteClient(action))
    gated = apply_enforced_gate(rec, threshold=0.5)
    assert gated == rec  # untouched: 0.95 clears the threshold


def test_apply_enforced_gate_refuses_an_advisory_record():
    ep = _episode()
    rec = run_episode_for_cell(ADVISORY_TWIN, ep, confidence=0.02,
                               client=ScriptedExecuteClient(entail("SEN").name))
    with pytest.raises(ValueError):
        apply_enforced_gate(rec, threshold=0.5)


def test_apply_enforced_gate_refuses_a_record_with_no_cell():
    rec = EpisodeRecord(episode_id="x", cell=None, executed=None, faithful=False,
                        covered=False, n_turns=0, parse_failed=False, served_provider=None,
                        confidence=0.5)
    with pytest.raises(ValueError):
        apply_enforced_gate(rec, threshold=0.5)


def test_apply_enforced_gate_requires_confidence_on_the_record():
    rec = EpisodeRecord(episode_id="x", cell=ENFORCED_TWIN, executed=None, faithful=False,
                        covered=False, n_turns=0, parse_failed=False, served_provider=None,
                        confidence=None)
    with pytest.raises(ValueError):
        apply_enforced_gate(rec, threshold=0.5)


def test_run_episode_for_cell_records_confidence_on_every_path():
    ep = _episode()
    nonllm_rec = run_episode_for_cell(NONLLM_GATE_CELL, ep, confidence=0.42, client=PoisonClient())
    advisory_rec = run_episode_for_cell(ADVISORY_TWIN, ep, confidence=0.42,
                                        client=ScriptedExecuteClient(entail("SEN").name))
    enforced_rec = run_episode_for_cell(ENFORCED_TWIN, ep, confidence=0.42,
                                        client=ScriptedExecuteClient(entail("SEN").name))
    assert nonllm_rec.confidence == 0.42
    assert advisory_rec.confidence == 0.42
    assert enforced_rec.confidence == 0.42


def test_run_episode_for_cell_rejects_an_llm_cell_with_unknown_control_mechanism():
    bad_cell = Cell("bogus", "none", "sideways")
    with pytest.raises(ValueError):
        run_episode_for_cell(bad_cell, _episode(), confidence=0.5, client=PoisonClient())
