import inspect

import numpy as np

import nag.controllers as controllers_module
from nag.controllers import nonllm_gate, random_gate
from nag.taxonomy import entail


def test_nonllm_gate_abstains_below_threshold_and_acts_above():
    e = entail("SEN")
    assert nonllm_gate(0.2, 0.5, e)["covered"] is False
    assert nonllm_gate(0.9, 0.5, e)["covered"] is True


def test_nonllm_gate_abstains_when_entailed_is_none_even_above_threshold():
    assert nonllm_gate(0.99, 0.5, None)["covered"] is False


def test_random_gate_hits_requested_coverage():
    e = entail("SEN")
    rng = np.random.default_rng(0)
    cov = np.mean([random_gate(rng, 0.3, e)["covered"] for _ in range(4000)])
    assert abs(cov - 0.3) < 0.03


def test_random_gate_abstains_when_entailed_is_none():
    rng = np.random.default_rng(0)
    assert random_gate(rng, 1.0, None)["covered"] is False


def test_nonllm_gate_fed_the_decoded_entailed_action_can_be_unfaithful_to_the_true_string():
    # Fed the TRUE string's entailed action, these gates can never lose --
    # `executed` equals the answer key by construction. The real driver
    # (Task 9) must instead feed them the action entailed by the DECODED
    # string -- what the channel actually delivered, the same thing the LLM
    # agent sees via read_buffer -- so a gate that fires on a corrupted
    # decode can come out unfaithful. This test simulates that wiring
    # directly and confirms the baseline really can lose; without this, the
    # non-LLM baselines would be useless for the comparison.
    true_string, decoded_string = "SEN", "NUR"
    true_entailed = entail(true_string)
    decoded_entailed = entail(decoded_string)
    assert decoded_entailed.name != true_entailed.name  # sanity: a real substitution

    out = nonllm_gate(0.9, 0.5, decoded_entailed)  # confidence clears threshold: fires
    assert out["covered"] is True
    faithful = out["executed"] is not None and out["executed"]["name"] == true_entailed.name
    assert faithful is False


def test_random_gate_fed_the_decoded_entailed_action_can_be_unfaithful_to_the_true_string():
    true_string, decoded_string = "SEN", "NUR"
    true_entailed = entail(true_string)
    decoded_entailed = entail(decoded_string)
    rng = np.random.default_rng(0)

    out = random_gate(rng, 1.0, decoded_entailed)  # coverage=1.0: always fires
    assert out["covered"] is True
    faithful = out["executed"] is not None and out["executed"]["name"] == true_entailed.name
    assert faithful is False


def test_controllers_module_never_names_actual_posterior():
    assert "actual posterior" not in inspect.getsource(controllers_module).lower()


def test_controllers_module_never_names_a_consequence_tier():
    blob = inspect.getsource(controllers_module).lower()
    for banned in ("tier 1", "tier 2", "tier 3", "tier_1", "consequence tier"):
        assert banned not in blob
