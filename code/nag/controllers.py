"""Non-LLM reference controllers. These exist to be able to beat the agent.

Both gates below act on whatever `Action` the caller passes as `entailed` --
they don't look at any string themselves. That is deliberate: it keeps the
gating logic (threshold comparison, coverage draw) independent of which
string produced the action, so it can be unit-tested in isolation here.

It also means the caller controls whether these baselines CAN lose. Wiring
them to `nag.taxonomy.entail(episode["true_string"])` would make them act on
the answer key and score faithful=1.0 by construction -- a baseline that
cannot lose is not a baseline. The real driver (Task 9) must instead pass
`nag.taxonomy.entail(episode["decoded_string"])` -- the action entailed by
what the channel actually delivered, exactly what the LLM agent sees via
`read_buffer` -- so a gate that fires on a corrupted decode is scored
against the true string and can come out unfaithful. See
`tests/test_controllers.py::test_nonllm_gate_fed_the_decoded_entailed_action_can_be_unfaithful_to_the_true_string`
for that failure mode pinned directly.
"""
from __future__ import annotations
import numpy as np
from nag.taxonomy import Action


def nonllm_gate(conf: float, threshold: float, entailed: Action | None) -> dict:
    """Plain threshold gate on reconstructed CALIBRATED confidence. No model in the loop.

    `conf` must be calibrated confidence (see `nag.confidence`), not a raw
    decoder score -- raw scores are not on a probability scale comparable to
    the thresholds swept across cells, and thresholding on them would bunch
    operating points into a degenerate risk-coverage curve.
    """
    if entailed is None or conf < threshold:
        return dict(covered=False, faithful=False, executed=None)
    return dict(covered=True, faithful=True,
                executed={"name": entailed.name, "args": dict(entailed.args)})


def random_gate(rng: np.random.Generator, coverage: float, entailed: Action | None) -> dict:
    """Acts at the requested coverage regardless of confidence.

    Isolates how much of a gate's apparent benefit is coverage reduction alone.
    """
    if entailed is None or rng.random() >= coverage:
        return dict(covered=False, faithful=False, executed=None)
    return dict(covered=True, faithful=True,
                executed={"name": entailed.name, "args": dict(entailed.args)})
