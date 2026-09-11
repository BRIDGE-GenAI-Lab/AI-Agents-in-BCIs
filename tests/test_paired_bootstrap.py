import numpy as np
import pandas as pd
from nag.paired_bootstrap import paired_risk_difference, _joint_cluster_draw

def _arm(vals, participants):
    return pd.DataFrame({"episode_id": [f"e{i}" for i in range(len(vals))],
                         "unsafe": vals, "participant_id": participants})

def test_same_participants_drawn_for_both_arms():
    """The whole point: one draw, applied to both arms."""
    rng = np.random.default_rng(0)
    clusters = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
    idx_a, idx_b = _joint_cluster_draw(clusters, clusters, rng)
    assert list(clusters[idx_a]) == list(clusters[idx_b])

def test_perfectly_correlated_arms_give_a_zero_width_interval():
    """Identical arms differ by exactly zero in every replicate. An
    independent-resampling bootstrap would produce a wide interval here, which
    is the bug this module exists to fix."""
    v = [1, 0, 1, 0, 1, 0]
    p = ["p1", "p1", "p2", "p2", "p3", "p3"]
    out = paired_risk_difference(_arm(v, p), _arm(v, p), cluster="participant_id",
                                 stat="unsafe", n_boot=200, seed=1)
    assert out["estimate"] == 0.0
    assert out["lo"] == 0.0 and out["hi"] == 0.0

def test_constant_offset_interval_excludes_zero():
    p = ["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4
    a = _arm([0] * 12, p)
    b = _arm([1] * 12, p)
    out = paired_risk_difference(a, b, cluster="participant_id", stat="unsafe",
                                 n_boot=500, seed=2)
    assert out["estimate"] == -1.0
    assert out["hi"] < 0.0

def test_raises_when_arms_do_not_share_episodes():
    a = _arm([1, 0], ["p1", "p1"])
    b = _arm([1, 0], ["p1", "p1"]); b["episode_id"] = ["x", "y"]
    try:
        paired_risk_difference(a, b, cluster="participant_id", stat="unsafe")
    except ValueError as e:
        assert "same episode set" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_raises_when_arms_span_different_participants():
    """A participant present in only one arm would be silently dropped from
    every replicate, narrowing the interval with no error. Caught, not averaged
    over. Raised by the task-1-3 reviewer as a latent trap."""
    a = _arm([1, 0, 1, 0], ["p1", "p1", "p2", "p2"])
    b = _arm([1, 0, 1, 0], ["p1", "p1", "p3", "p3"])
    try:
        paired_risk_difference(a, b, cluster="participant_id", stat="unsafe", n_boot=10)
    except ValueError as e:
        assert "same participants" in str(e)
    else:
        raise AssertionError("expected ValueError")
