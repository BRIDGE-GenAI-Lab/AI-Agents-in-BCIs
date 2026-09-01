import numpy as np
import pandas as pd
import pytest

from nag.stats import bh_adjust, cluster_bootstrap, risk_difference, risk_ratio


# --- Correction 9.1: the hierarchy is the whole point -----------------------

def test_cluster_bootstrap_resamples_participants_not_rows():
    """Same row count, fewer participants -> wider CI. This is the brief's
    own test (kept in spirit, per the team lead's instruction), rewritten
    two ways:

    1. named dict keys instead of `list(...values())[1:]` (Correction 9.4:
       that indexing depends on dict insertion order).
    2. the DGP now gives each participant their OWN true rate (real
       between-participant heterogeneity), not one shared p=0.3 for every
       row regardless of label. The brief's literal version draws `y` i.i.d.
       independent of `participant_id`, so there is no actual clustering
       structure for the bootstrap to detect -- "fewer clusters -> wider CI"
       becomes a coin flip on that data (verified: the brief's own
       reference implementation fails this exact assertion under seed=0,
       n_boot=500 -- 0.056 vs 0.0665). With genuine per-participant rates
       the effect is robust across seeds (checked 0-9).
    """
    def make(n_clusters: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        per = 1000 // n_clusters
        rate = np.repeat(rng.uniform(0.1, 0.6, n_clusters), per)
        pid = np.repeat(np.arange(n_clusters), per)
        return pd.DataFrame({"participant_id": pid, "y": rng.binomial(1, rate)})

    f = lambda d: d["y"].mean()
    few = make(5, seed=0)
    many = make(50, seed=0)
    r_few = cluster_bootstrap(few, f, n_boot=500, seed=0)
    r_many = cluster_bootstrap(many, f, n_boot=500, seed=0)
    w_few = r_few["hi"] - r_few["lo"]
    w_many = r_many["hi"] - r_many["lo"]
    assert w_few > w_many          # same row count, fewer clusters -> wider CI


def test_cluster_bootstrap_returns_named_keys():
    df = pd.DataFrame({"participant_id": np.repeat(np.arange(10), 5),
                       "y": np.random.default_rng(1).binomial(1, 0.4, 50)})
    result = cluster_bootstrap(df, lambda d: d["y"].mean(), n_boot=200)
    assert set(result.keys()) == {"estimate", "lo", "hi"}
    assert result["lo"] <= result["estimate"] <= result["hi"]


def test_repetition_level_resample_does_not_narrow_like_an_independent_row_resample():
    """The pseudoreplication guard. Repetitions within a participant estimate
    model stochasticity only -- they are NOT independent neural observations
    (Correction 9.1). Build data with an extreme within-participant
    correlation (each participant's outcome is ~fixed across all of their
    repetitions) and show that clustering on `participant_id` gives a wider,
    honest interval than clustering on a per-row id, which is mathematically
    identical to a naive i.i.d. bootstrap over rows and would treat every
    repetition as its own independent piece of evidence.
    """
    rng = np.random.default_rng(7)
    n_participants, n_reps = 10, 40
    participant_y = rng.integers(0, 2, n_participants)  # each participant is ~all-1 or ~all-0
    participant_id = np.repeat(np.arange(n_participants), n_reps)
    # small per-repetition noise so the column isn't degenerate, but the
    # participant-level value dominates
    noise = rng.random(n_participants * n_reps) < 0.05
    y = np.repeat(participant_y, n_reps).astype(bool)
    y = np.where(noise, ~y, y).astype(int)
    df = pd.DataFrame({
        "participant_id": participant_id,
        "row_id": np.arange(n_participants * n_reps),  # unique per row
        "y": y,
    })
    f = lambda d: d["y"].mean()

    naive = cluster_bootstrap(df, f, cluster="row_id", n_boot=500, seed=0)   # == plain row bootstrap
    correct = cluster_bootstrap(df, f, cluster="participant_id", n_boot=500, seed=0)

    w_naive = naive["hi"] - naive["lo"]
    w_correct = correct["hi"] - correct["lo"]
    assert w_correct > w_naive, (
        "clustering on participant_id must not produce a narrower interval "
        "than treating each repetition as independent -- that would mean the "
        "pseudoreplication guard isn't doing anything"
    )


# --- Correction 9.2: risk difference / risk ratio lead, with CIs -----------

def test_risk_difference_and_ratio_lead_over_odds():
    rd = risk_difference(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert abs(rd["estimate"] - 0.25) < 1e-9
    assert rd["lo"] <= rd["estimate"] <= rd["hi"]

    rr = risk_ratio(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert abs(rr["estimate"] - 2.0) < 1e-9
    assert rr["lo"] <= rr["estimate"] <= rr["hi"]


def test_risk_difference_recovers_known_value_with_clustered_ci():
    rng = np.random.default_rng(2)
    n_participants = 20
    a = rng.binomial(1, 0.6, n_participants * 10)
    b = rng.binomial(1, 0.4, n_participants * 10)
    cluster = np.repeat(np.arange(n_participants), 10)
    rd = risk_difference(a, b, cluster_a=cluster, cluster_b=cluster, n_boot=300, seed=3)
    assert 0.0 < rd["estimate"] < 0.4
    assert rd["lo"] < rd["estimate"] < rd["hi"]
    assert np.isfinite(rd["lo"]) and np.isfinite(rd["hi"])


def test_risk_ratio_recovers_known_value_with_clustered_ci():
    rng = np.random.default_rng(4)
    n_participants = 20
    a = rng.binomial(1, 0.6, n_participants * 10)
    b = rng.binomial(1, 0.3, n_participants * 10)
    cluster = np.repeat(np.arange(n_participants), 10)
    rr = risk_ratio(a, b, cluster_a=cluster, cluster_b=cluster, n_boot=300, seed=5)
    assert 1.0 < rr["estimate"] < 3.0
    assert rr["lo"] < rr["estimate"] < rr["hi"]
    assert np.isfinite(rr["lo"]) and np.isfinite(rr["hi"])


# --- Correction 9.2: zero-event cells are EXPECTED, not an edge case -------

def test_risk_ratio_zero_event_arm_is_finite_not_dropped_or_inf():
    """A high-threshold gate that admits nothing unsafe is the expected,
    good outcome -- it must not crash, return inf/nan, or be silently
    excluded from the reported comparison."""
    a = np.array([1, 0, 1, 0, 1] * 4)          # some events
    b = np.zeros(20)                            # zero-event cell: the gate worked
    rr = risk_ratio(a, b)
    assert np.isfinite(rr["estimate"])
    assert np.isfinite(rr["lo"]) and np.isfinite(rr["hi"])
    assert rr["estimate"] > 0


def test_risk_ratio_both_arms_zero_events_is_finite():
    a = np.zeros(15)
    b = np.zeros(15)
    rr = risk_ratio(a, b)
    assert np.isfinite(rr["estimate"])
    assert abs(rr["estimate"] - 1.0) < 1e-9  # no evidence of a difference
    assert np.isfinite(rr["lo"]) and np.isfinite(rr["hi"])


def test_risk_ratio_zero_event_arm_finite_under_clustered_ci_too():
    rng = np.random.default_rng(6)
    n_participants = 12
    a = rng.binomial(1, 0.3, n_participants * 5)
    b = np.zeros(n_participants * 5)
    cluster = np.repeat(np.arange(n_participants), 5)
    rr = risk_ratio(a, b, cluster_a=cluster, cluster_b=cluster, n_boot=200, seed=8)
    assert np.isfinite(rr["estimate"])
    assert np.isfinite(rr["lo"]) and np.isfinite(rr["hi"])


def test_risk_difference_zero_event_arm_is_finite():
    a = np.array([1, 0, 1, 0])
    b = np.zeros(10)
    rd = risk_difference(a, b)
    assert np.isfinite(rd["estimate"])
    assert np.isfinite(rd["lo"]) and np.isfinite(rd["hi"])


# --- bh_adjust ---------------------------------------------------------

def test_bh_adjust_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.04, 0.2])
    q = bh_adjust(p)
    assert np.all(q >= p) and np.all(q <= 1.0)


def test_bh_adjust_matches_hand_computed_example():
    p = np.array([0.001, 0.01, 0.04, 0.2])
    # BH step-up by hand (n=4): q4=0.2*4/4=0.2; q3=min(0.2,0.04*4/3)=0.05333..;
    # q2=min(q3,0.01*4/2=0.02)=0.02; q1=min(q2,0.001*4/1=0.004)=0.004
    expected = np.array([0.004, 0.02, 0.04 * 4 / 3, 0.2])
    q = bh_adjust(p)
    assert np.allclose(q, expected)


def test_bh_adjust_step_up_ordering_is_respected_for_unsorted_input():
    p = np.array([0.2, 0.001, 0.2, 0.04, 0.01])
    q = bh_adjust(p)
    # sort both by the underlying p-value and check the step-up monotone
    # property holds regardless of input order
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
