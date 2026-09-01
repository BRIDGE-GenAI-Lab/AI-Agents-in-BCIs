import pandas as pd
from nag.analysis_population import population, outcome_triple

def _frame():
    # 4 episodes: 1 clean+faithful, 1 error+unfaithful, 1 error+abstain, 1 parse failure
    return pd.DataFrame([
        dict(episode_id="a", err=False, covered=True,  faithful=True,  parse_failed=False, participant_id="p1"),
        dict(episode_id="b", err=True,  covered=True,  faithful=False, parse_failed=False, participant_id="p1"),
        dict(episode_id="c", err=True,  covered=False, faithful=False, parse_failed=False, participant_id="p2"),
        dict(episode_id="d", err=True,  covered=False, faithful=False, parse_failed=True,  participant_id="p2"),
    ])

def test_end_to_end_keeps_every_episode():
    assert len(population(_frame(), "end_to_end")) == 4

def test_error_conditional_keeps_only_error_bearing():
    out = population(_frame(), "error_conditional")
    assert len(out) == 3 and out["err"].all()

def test_intention_to_deploy_is_end_to_end():
    """Parse failures are operational behaviour, not missing data, so the
    intention-to-deploy population excludes nothing."""
    assert len(population(_frame(), "intention_to_deploy")) == 4

def test_outcome_triple_separates_the_three_dimensions():
    t = outcome_triple(_frame())
    assert t["coverage"] == 0.5                 # 2 of 4 executed
    assert t["conditional_fidelity"] == 0.5     # 1 of 2 executed actions faithful
    assert t["parse_failure"] == 0.25           # 1 of 4
    assert t["unfaithful_of_all"] == 0.25       # 1 of 4 episodes ended in a wrong action
    assert t["n_episodes"] == 4 and t["n_executed"] == 2 and t["n_unfaithful"] == 1

def test_outcome_triple_reports_nan_fidelity_when_nothing_executed():
    import math
    df = _frame(); df["covered"] = False
    t = outcome_triple(df)
    assert t["coverage"] == 0.0
    assert math.isnan(t["conditional_fidelity"])
    assert t["n_executed"] == 0
