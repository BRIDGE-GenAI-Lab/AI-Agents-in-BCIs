import numpy as np
import pandas as pd
import pytest
from nag.replay import OutcomeCounts, classify, outcome_distribution, replay_panel, simulate


def test_distribution_is_a_probability_distribution():
    c = OutcomeCounts(success=5, wrong_tier3=2, wrong_other=3, decline_abstain=7, decline_parse=5)
    d = outcome_distribution(c)
    total = sum(d["p_success_by_attempt"]) + sum(d["p_wrong_by_attempt"]) + d["p_unresolved"]
    assert total == pytest.approx(1.0)


def test_single_attempt_is_the_bare_rate():
    c = OutcomeCounts(success=5, wrong_tier3=0, wrong_other=5, decline_abstain=10, decline_parse=0)
    d = outcome_distribution(c, max_attempts=1)
    assert d["p_success_by_attempt"][0] == pytest.approx(0.25)
    assert d["p_unresolved"] == pytest.approx(0.5)


def test_a_policy_that_never_declines_cannot_reach_attempt_two():
    c = OutcomeCounts(success=8, wrong_tier3=1, wrong_other=1, decline_abstain=0, decline_parse=0)
    d = outcome_distribution(c)
    assert d["p_success_by_attempt"][1] == 0.0
    assert d["p_success_by_attempt"][2] == 0.0
    assert d["p_unresolved"] == 0.0


def test_all_declines_is_always_unresolved():
    c = OutcomeCounts(success=0, wrong_tier3=0, wrong_other=0, decline_abstain=4, decline_parse=4)
    assert outcome_distribution(c)["p_unresolved"] == pytest.approx(1.0)


def test_retries_are_attributed_to_the_right_decline_type():
    """A pool that only ever declines by parse failure must attribute no retry
    to explicit abstention, and vice versa."""
    c = OutcomeCounts(success=2, wrong_tier3=0, wrong_other=0, decline_abstain=0, decline_parse=8)
    d = outcome_distribution(c)
    assert d["e_retries_abstain"] == pytest.approx(0.0)
    assert d["e_retries_parse"] > 0.0


def test_without_replacement_exhausts_a_tiny_pool():
    """Two donors, both declines: three attempts cannot be drawn, and the task
    is unresolved rather than raising or silently reusing a donor."""
    c = OutcomeCounts(success=0, wrong_tier3=0, wrong_other=0, decline_abstain=2, decline_parse=0)
    d = outcome_distribution(c, with_replacement=False)
    assert d["p_unresolved"] == pytest.approx(1.0)
    assert d["e_attempts"] == pytest.approx(2.0)


def test_with_replacement_differs_from_without_on_the_same_counts():
    c = OutcomeCounts(success=1, wrong_tier3=0, wrong_other=0, decline_abstain=3, decline_parse=0)
    assert (outcome_distribution(c, with_replacement=True)["p_unresolved"]
            != pytest.approx(outcome_distribution(c, with_replacement=False)["p_unresolved"]))


def test_classify_reads_the_task20_schema():
    df = pd.DataFrame({
        "covered":       [True,  True,        True,       False, False],
        "faithful":      [True,  False,       False,      False, False],
        "executed_name": ["save_note", "summon_staff", "play_media", None, None],
        "parse_failed":  [False, False,       False,      False, True],
    })
    c = classify(df)
    assert (c.success, c.wrong_tier3, c.wrong_other) == (1, 1, 1)
    assert (c.decline_abstain, c.decline_parse) == (1, 1)
    assert c.total == 5


def test_classify_treats_missing_parse_failed_as_abstention():
    """The three deterministic comparator files have no `parse_failed` column;
    they cannot parse-fail by construction."""
    df = pd.DataFrame({"covered": [False], "faithful": [False], "executed_name": [None]})
    c = classify(df)
    assert (c.decline_abstain, c.decline_parse) == (1, 0)


@pytest.mark.parametrize("counts", [
    OutcomeCounts(5, 2, 3, 7, 5),
    OutcomeCounts(1, 1, 0, 15, 4),
    OutcomeCounts(12, 0, 1, 6, 3),
    OutcomeCounts(0, 3, 3, 8, 8),
])
def test_monte_carlo_through_the_sandbox_reproduces_the_closed_form(counts):
    """Two independent implementations of one quantity.

    The closed form enumerates the draw tree; `simulate` draws donors and
    applies real state changes to an AssistiveSandbox. Agreement is the only
    evidence that the arithmetic is right.
    """
    exact = outcome_distribution(counts)
    got = simulate(counts, rng=np.random.default_rng(20260901), n_traj=200_000)
    for key in ("p_success", "p_wrong", "p_wrong_tier3", "p_unresolved", "e_attempts"):
        assert got[key] == pytest.approx(exact[key], abs=0.005), key


def test_simulation_actually_changes_sandbox_state():
    """Guards against a `simulate` that counts outcomes without executing."""
    counts = OutcomeCounts(success=10, wrong_tier3=0, wrong_other=0,
                           decline_abstain=0, decline_parse=0)
    got = simulate(counts, rng=np.random.default_rng(1), n_traj=100)
    assert got["n_state_changes"] == 100


def _frame(n_per_cmd, commands, participants, covered, faithful, seed=0):
    rows = []
    for c in commands:
        for i in range(n_per_cmd):
            rows.append({"episode_id": f"{c}-{i}", "participant_id": participants[i % len(participants)],
                         "assigned_command": c, "covered": covered, "faithful": faithful,
                         "executed_name": "save_note" if covered else None, "parse_failed": False})
    return pd.DataFrame(rows)


def test_panel_uses_one_joint_participant_draw_across_policies():
    """Two IDENTICAL policies must show a difference of exactly zero with a
    zero-width interval. Independent resampling would produce a non-zero
    interval, which is the error the supplement warns about."""
    cmds = ["a", "b"]; parts = ["p1", "p2", "p3"]
    f = _frame(6, cmds, parts, covered=True, faithful=True)
    out = replay_panel({"x": f, "y": f.copy()}, commands=cmds, n_boot=200, seed=20260901)
    d = out.set_index("policy")
    assert d.loc["x", "p_success"] == pytest.approx(d.loc["y", "p_success"])
    assert d.loc["x", "p_success_lo"] == pytest.approx(d.loc["y", "p_success_lo"])


def test_panel_reports_every_required_endpoint():
    cmds = ["a"]; f = _frame(6, cmds, ["p1", "p2"], covered=True, faithful=True)
    out = replay_panel({"x": f}, commands=cmds, n_boot=50, seed=1)
    for col in ("p_success", "p_wrong", "p_wrong_tier3", "p_unresolved",
                "e_attempts", "success_per_100_attempts",
                "e_retries_abstain", "e_retries_parse"):
        assert col in out.columns, col
        assert f"{col}_lo" in out.columns and f"{col}_hi" in out.columns


def test_commands_are_weighted_uniformly_not_by_donor_count():
    """A command with twice the donors must not get twice the weight."""
    cmds = ["a", "b"]
    good = _frame(4, ["a"], ["p1"], covered=True, faithful=True)
    bad = _frame(8, ["b"], ["p1"], covered=True, faithful=False)
    f = pd.concat([good, bad], ignore_index=True)
    out = replay_panel({"x": f}, commands=cmds, n_boot=20, seed=1)
    assert out.loc[0, "p_success"] == pytest.approx(0.5, abs=1e-9)


from nag.replay import paired_difference


def _mixed_frame(commands, participants, seed=0):
    """Per-participant success varies, unlike `_frame`'s uniform outcome, so a
    joint-vs-independent resampling difference is actually detectable."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in commands:
        for p in participants:
            covered = bool(rng.integers(0, 2))
            rows.append({"episode_id": f"{c}-{p}", "participant_id": p,
                         "assigned_command": c, "covered": covered, "faithful": covered,
                         "executed_name": "save_note" if covered else None,
                         "parse_failed": False})
    return pd.DataFrame(rows)


def test_paired_difference_is_exactly_zero_for_identical_policies_with_participant_variability():
    """Two IDENTICAL policies, with real between-participant variance in the
    endpoint, must show a difference of exactly zero AND a zero-width CI in
    every one of many replicates. Reconstructing the interval from two
    independently-resampled marginals would not give exactly zero here,
    because the two draws would occasionally disagree."""
    cmds = ["a", "b", "c"]
    parts = [f"p{i}" for i in range(8)]
    f = _mixed_frame(cmds, parts, seed=3)
    out = paired_difference({"x": f, "y": f.copy()}, "x", "y", "p_success",
                            n_boot=300, seed=20260901)
    assert out["difference"] == pytest.approx(0.0)
    assert out["lo"] == pytest.approx(0.0)
    assert out["hi"] == pytest.approx(0.0)
    assert out["n_boot_dropped"] == 0


def test_paired_difference_point_estimate_matches_replay_panel():
    cmds = ["a", "b"]; parts = ["p1", "p2", "p3"]
    good = _frame(6, cmds, parts, covered=True, faithful=True)
    bad = _frame(6, cmds, parts, covered=True, faithful=False)
    panel = replay_panel({"good": good, "bad": bad}, commands=cmds, n_boot=50, seed=1)
    p = panel.set_index("policy")
    out = paired_difference({"good": good, "bad": bad}, "good", "bad", "p_success",
                            n_boot=50, seed=1)
    assert out["difference"] == pytest.approx(p.loc["good", "p_success"] - p.loc["bad", "p_success"])
    assert out["a"] == pytest.approx(p.loc["good", "p_success"])
    assert out["b"] == pytest.approx(p.loc["bad", "p_success"])


def test_paired_difference_reports_the_required_fields():
    cmds = ["a"]; f = _frame(6, cmds, ["p1", "p2"], covered=True, faithful=True)
    out = paired_difference({"x": f, "y": f.copy()}, "x", "y", "e_attempts", n_boot=20, seed=1)
    for key in ("policy_a", "policy_b", "endpoint", "a", "b", "difference",
               "lo", "hi", "n_boot", "n_boot_dropped"):
        assert key in out, key


from nag.replay import paired_differences


def test_paired_differences_matches_paired_difference_per_endpoint():
    cmds = ["a", "b"]; parts = ["p1", "p2", "p3"]
    good = _frame(6, cmds, parts, covered=True, faithful=True)
    bad = _frame(6, cmds, parts, covered=True, faithful=False)
    endpoints = ("p_success", "p_wrong", "e_attempts")
    multi = paired_differences({"good": good, "bad": bad}, "good", "bad", endpoints,
                               commands=cmds, n_boot=50, seed=7)
    for ep in endpoints:
        single = paired_difference({"good": good, "bad": bad}, "good", "bad", ep,
                                   commands=cmds, n_boot=50, seed=7)
        assert multi[ep] == single, ep


def test_paired_differences_one_pass_beats_looping_paired_difference_in_time():
    """The whole point of the plural function: one bootstrap pass computing
    several endpoints must be faster than calling the singular version once
    per endpoint, because both do the same number of participant draws but
    the loop repeats every draw's dataframe work once per endpoint."""
    import time
    cmds = ["a", "b", "c"]
    parts = [f"p{i}" for i in range(8)]
    f = _mixed_frame(cmds, parts, seed=5)
    endpoints = ("p_success", "p_wrong", "p_unresolved", "e_attempts")

    t0 = time.perf_counter()
    paired_differences({"x": f, "y": f.copy()}, "x", "y", endpoints,
                       commands=cmds, n_boot=300, seed=1)
    t_multi = time.perf_counter() - t0

    t0 = time.perf_counter()
    for ep in endpoints:
        paired_difference({"x": f, "y": f.copy()}, "x", "y", ep,
                          commands=cmds, n_boot=300, seed=1)
    t_loop = time.perf_counter() - t0

    assert t_multi < t_loop
