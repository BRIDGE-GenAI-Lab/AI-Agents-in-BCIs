import inspect

import numpy as np
import pandas as pd
import pytest

from nag.design import (
    Cell,
    EARLIER_SESSION,
    cost_estimate,
    enumerate_cells,
    factorial_base_cells,
    natural_error_rate,
    power_curve,
    run_allocation,
    sample_episodes,
)


# --- a synthetic pool shaped like build_episode_pool()'s real output -------

def _synthetic_pool(seed=0, n_participants=10, per_participant=40):
    """Mimics build_episode_pool()'s schema: episode_id, participant_id, err,
    tier, fit_match, confidence -- with the real anti-predictive earlier_session
    stratum represented at low (but nonzero) prevalence, like the real data."""
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for p in range(n_participants):
        pid = f"P_{p:02d}"
        for _ in range(per_participant):
            err = bool(rng.random() < 0.34)  # natural ~34% prevalence
            tier = int(rng.choice([1, 2, 3], p=[0.27, 0.33, 0.40]))  # uneven, like the real codebook
            fit_match = str(rng.choice(
                ["own_session_own_condition", "own_session_other_condition", EARLIER_SESSION],
                p=[0.65, 0.33, 0.02],
            ))
            rows.append(dict(
                episode_id=f"ep_{i:05d}", participant_id=pid, err=err, tier=tier, fit_match=fit_match,
                n_errors=1 if err else 0, confidence=float(rng.uniform(0.2, 0.9)),
            ))
            i += 1
    return pd.DataFrame(rows)


# --- cell enumeration -------------------------------------------------------

def test_cell_enumeration_covers_the_factorial_and_reference_arms():
    cells = enumerate_cells()
    srcs = {c.uncertainty_source for c in cells}
    mechs = {c.control_mechanism for c in cells}
    assert srcs >= {"none", "self_confidence", "decoder_confidence"}
    assert mechs >= {"advisory", "enforced"}
    assert any(c.name == "nonllm_gate" for c in cells)
    assert any(c.name == "random_gate" for c in cells)


def test_reference_arms_use_no_llm_and_are_excluded_from_the_llm_count():
    cells = enumerate_cells()
    non_llm = {c.name for c in cells if not c.uses_llm}
    assert non_llm == {"nonllm_gate", "random_gate"}
    assert all(c.uses_llm for c in cells if c.name not in non_llm)


def test_factorial_base_cells_are_exactly_the_six_uncertainty_x_mechanism_combos():
    base = factorial_base_cells()
    assert len(base) == 6
    combos = {(c.uncertainty_source, c.control_mechanism) for c in base}
    assert len(combos) == 6
    assert all(c.scaffold == 0 for c in base)


def test_cell_names_are_unique():
    cells = enumerate_cells()
    names = [c.name for c in cells]
    assert len(names) == len(set(names))


# --- sample_episodes: paired design, stratification, exclusion ------------

def test_sample_episodes_has_no_per_cell_argument():
    # Ruling 6: the design is fully paired -- every cell sees the SAME shared
    # set, so this function must not branch on `cell` at all.
    params = set(inspect.signature(sample_episodes).parameters)
    assert "cell" not in params and "per_cell" not in params


def test_sample_episodes_is_deterministic_given_the_same_seed():
    pool = _synthetic_pool()
    a = sample_episodes(pool, n_total=120, seed=7)
    b = sample_episodes(pool, n_total=120, seed=7)
    assert set(a["episode_id"]) == set(b["episode_id"])


def test_sample_episodes_enriches_error_bearing_toward_fifty_fifty():
    pool = _synthetic_pool()
    assert pool["err"].mean() < 0.40  # natural prevalence well under 50%
    sampled = sample_episodes(pool, n_total=120, seed=1)
    frac = sampled["err"].mean()
    assert 0.40 <= frac <= 0.60


def test_sample_episodes_balances_tier():
    pool = _synthetic_pool()
    sampled = sample_episodes(pool, n_total=300, seed=2)
    counts = sampled["tier"].value_counts(normalize=True)
    # not exactly 1/3 each (integer rounding + availability caps), but no
    # tier should be left anywhere near its skewed natural share
    assert counts.min() > 0.20
    assert counts.max() < 0.45


def test_sample_episodes_represents_but_does_not_over_or_under_sample_earlier_session():
    pool = _synthetic_pool()
    sampled = sample_episodes(pool, n_total=300, seed=3)
    assert (sampled["fit_match"] == EARLIER_SESSION).sum() > 0  # represented for sensitivity analysis
    assert (sampled["fit_match"] == EARLIER_SESSION).mean() < 0.15  # not force-balanced against 3 strata


def test_sample_episodes_marks_earlier_session_as_not_primary_eligible():
    pool = _synthetic_pool()
    sampled = sample_episodes(pool, n_total=300, seed=4)
    assert "primary_eligible" in sampled.columns
    is_earlier = sampled["fit_match"] == EARLIER_SESSION
    assert (~sampled.loc[is_earlier, "primary_eligible"]).all()
    assert sampled.loc[~is_earlier, "primary_eligible"].all()
    # the exclusion is pre-specified, not a physical drop: earlier_session
    # rows stay IN the returned set for the sensitivity analysis
    assert is_earlier.any()


def test_sample_episodes_no_single_participant_dominates():
    pool = _synthetic_pool(n_participants=10, per_participant=40)
    sampled = sample_episodes(pool, n_total=200, seed=5)
    share = sampled["participant_id"].value_counts(normalize=True)
    assert share.max() < 0.30  # well under what 1/10 participants dominating would look like


def test_natural_error_rate_reports_the_unenriched_prevalence():
    pool = _synthetic_pool()
    rate = natural_error_rate(pool)
    assert 0.20 < rate < 0.45
    sampled = sample_episodes(pool, n_total=120, seed=6)
    assert sampled["err"].mean() > rate  # enrichment moved it up, as intended


# --- cost model -------------------------------------------------------------

def test_cost_scales_linearly_with_episodes():
    m = dict(total_episodes=1000, calls_per_episode=4, in_tok=2000, out_tok=300)
    a = cost_estimate(m, 2.0, 10.0, 0.0)["usd_total"]
    m2 = dict(m, total_episodes=2000)
    assert abs(cost_estimate(m2, 2.0, 10.0, 0.0)["usd_total"] - 2 * a) < 1e-6


def test_cost_estimate_caching_reduces_cost():
    m = dict(total_episodes=1000, calls_per_episode=4, in_tok=2000, out_tok=300)
    no_cache = cost_estimate(m, 2.0, 10.0, 0.0)["usd_total"]
    cached = cost_estimate(m, 2.0, 10.0, 0.75)["usd_total"]
    assert cached < no_cache


def test_run_allocation_excludes_non_llm_reference_arms_from_the_llm_count():
    alloc = run_allocation(n_total=100)
    cells = enumerate_cells()
    assert alloc["llm_cells"] == len([c for c in cells if c.uses_llm])
    assert set(alloc["non_llm_cells"]) == {"nonllm_gate", "random_gate"}
    assert alloc["total_episode_runs"] > 0


def test_run_allocation_scales_with_n_total():
    small = run_allocation(n_total=100)
    big = run_allocation(n_total=200)
    assert big["core_episode_runs"] == 2 * small["core_episode_runs"]
    assert big["panel_episode_runs"] == 2 * small["panel_episode_runs"]
    assert big["variance_episode_runs"] == small["variance_episode_runs"]  # fixed sub-study


# --- power curve --------------------------------------------------------

def test_power_curve_narrows_with_n():
    pc = power_curve(rate=0.30, n_grid=[100, 400, 1600])
    w = pc.sort_values("n")["ci_halfwidth"].to_numpy()
    assert w[0] > w[1] > w[2]


def test_power_curve_clustered_is_wider_than_naive():
    pc = power_curve(rate=0.30, n_grid=[300], n_participants=42, iccs=(0.05, 0.15))
    row = pc.iloc[0]
    assert row["ci_halfwidth_icc0.05"] > row["ci_halfwidth"]
    assert row["ci_halfwidth_icc0.15"] > row["ci_halfwidth_icc0.05"]


def test_power_curve_clustered_narrows_much_less_than_naive_between_large_n():
    # the whole point of Ruling-style clustering: doubling episodes buys far
    # less than doubling the naive (unclustered) formula would suggest,
    # because participant count is the real bottleneck.
    pc = power_curve(rate=0.30, n_grid=[300, 800], n_participants=42, iccs=(0.15,))
    naive_ratio = pc.iloc[0]["ci_halfwidth"] / pc.iloc[1]["ci_halfwidth"]
    clustered_ratio = pc.iloc[0]["ci_halfwidth_icc0.15"] / pc.iloc[1]["ci_halfwidth_icc0.15"]
    assert clustered_ratio < naive_ratio
