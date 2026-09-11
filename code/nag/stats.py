"""Hierarchical inference: participant-clustered CIs, risk differences and
ratios with explicit zero-event handling, and multiplicity correction.

The data hierarchy is participant -> session -> episode -> repetition
(`nag.agent.EpisodeRecord` carries `participant_id` and `study` for exactly
this reason). Repetitions estimate MODEL STOCHASTICITY only -- rerunning the
same episode against the same decode is not a second independent neural
observation. Precision in this study is bounded by the ~47 ALS participants
(`nag.design.power_curve`'s docstring), not by episode or repetition count.
Every interval built here therefore resamples PARTICIPANTS, never rows: a
plain i.i.d. bootstrap over rows would treat correlated repetitions (or
correlated episodes from a single prolific participant) as independent
evidence and report a confidently wrong, too-narrow interval.

Effect measures: risk differences and risk ratios LEAD; odds ratios are
deliberately not implemented here. Event rates in this study are high
(faithful-execution rates, not rare-disease incidence), and odds ratios
exaggerate effects at high base rates in a way that misleads a clinical
reader -- see Correction 9.2. `risk_difference` and `risk_ratio` are the only
effect measures this module offers.

Zero-event cells are EXPECTED, not an edge case: a good gate at a high
threshold is supposed to admit nothing unsafe, i.e. zero unfaithful
executions among the covered episodes. `risk_ratio` never divides by zero and
never returns a bare `inf` -- see its docstring for the small-sample
correction applied only when a cell is empty. Silently dropping a zero-event
cell would bias every headline toward "gates don't help," because it drops
exactly the cells where the gate worked.

The panel of models evaluated is a PRESPECIFIED SELECTION, not a random draw
from the population of all LLMs (Correction 9.3). This module deliberately
does not offer a single pooled "does the effect hold across models" test
function, because a bundled p-value invites exactly the generalization claim
("this holds for LLMs in general") that a fixed, non-random panel cannot
license. The intended usage is: call `cluster_bootstrap` / `risk_difference`
/ `risk_ratio` once per model (`df.groupby("model")`), report each model's
own estimate and CI, and use `bh_adjust` across that panel of per-model
p-values to control the multiple-comparisons rate across the SAME
prespecified panel -- consistency (or its absence) across the reported
per-model estimates is itself the finding, not a single averaged number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

_Z975 = sps.norm.ppf(0.975)


# --- the clustered bootstrap: the one interval-building primitive ---------

def cluster_bootstrap(df: pd.DataFrame, stat_fn, cluster: str = "participant_id",
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile bootstrap CI for `stat_fn(df)`, resampling `cluster` values
    (default `participant_id`) WITH replacement rather than rows.

    Each bootstrap draw picks `len(unique ids)` cluster ids with replacement
    and reassembles the rows belonging to those ids (a participant picked
    twice contributes all of their rows twice) before calling `stat_fn` on
    the reassembled frame. `stat_fn` must accept a `pd.DataFrame` with the
    same columns as `df` and return a single float.

    Returns `{"estimate": stat_fn(df), "lo": ..., "hi": ...}` (95% percentile
    interval) -- named keys, not a positional tuple, so a caller cannot
    silently depend on dict insertion order (Correction 9.4).

    This is deliberately the ONLY interval machinery in this module:
    `risk_difference`/`risk_ratio` fall back to it (via `cluster_a`/
    `cluster_b`) whenever real participant ids are available, rather than
    duplicating a second resampling scheme.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    groups_idx = df.groupby(cluster, sort=False).indices  # {cluster_id: int positions}
    ids = np.array(list(groups_idx.keys()), dtype=object)

    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([groups_idx[c] for c in pick])
        boots[b] = stat_fn(df.iloc[rows])

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(estimate=float(stat_fn(df)), lo=float(lo), hi=float(hi))


def _cluster_resample_array(values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One clustered bootstrap draw of a flat array, resampling `clusters`'
    unique values with replacement. The array-valued sibling of
    `cluster_bootstrap`'s per-draw logic, used by `risk_difference` /
    `risk_ratio` when given `cluster_a`/`cluster_b` directly rather than a
    shared `pd.DataFrame`."""
    idx_by_id: dict = {}
    for pos, cid in enumerate(clusters):
        idx_by_id.setdefault(cid, []).append(pos)
    idx_by_id = {k: np.array(v) for k, v in idx_by_id.items()}
    ids = np.array(list(idx_by_id.keys()), dtype=object)
    pick = rng.choice(ids, size=len(ids), replace=True)
    rows = np.concatenate([idx_by_id[c] for c in pick])
    return values[rows]


# --- risk difference / risk ratio ------------------------------------------

def _rd_point(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.mean() - b.mean())


def _rr_point(a: np.ndarray, b: np.ndarray) -> float:
    """Risk ratio a.mean()/b.mean(), with the Haldane-Anscombe 0.5 continuity
    correction (Haldane 1940; Anscombe 1956) applied to BOTH arms whenever
    either has zero events. This is the standard small-sample fix for the
    zero-cell case in epidemiology: it adds 0.5 to every one of the four
    counts (events and non-events in each arm), which keeps every log-ratio
    finite and shrinks the estimate toward the null exactly when the raw
    counts alone cannot support one -- the same shrink-toward-null spirit as
    Firth penalisation, in closed form rather than by penalised likelihood.
    Applied only when a zero cell is actually present, so a well-behaved
    comparison is never perturbed by it.
    """
    x1, n1 = float(a.sum()), len(a)
    x2, n2 = float(b.sum()), len(b)
    if x1 == 0 or x2 == 0:
        x1, n1, x2, n2 = x1 + 0.5, n1 + 1, x2 + 0.5, n2 + 1
    return (x1 / n1) / (x2 / n2)


def _wald_rd_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    p1, p2 = a.mean(), b.mean()
    se = np.sqrt(p1 * (1 - p1) / len(a) + p2 * (1 - p2) / len(b))
    est = p1 - p2
    return float(max(est - _Z975 * se, -1.0)), float(min(est + _Z975 * se, 1.0))


def _katz_rr_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Katz log-method CI (Katz et al. 1978) for a risk ratio of two
    independent proportions, with the same Haldane-Anscombe correction as
    `_rr_point` applied whenever a cell is empty -- this is what keeps
    `log(0)` and `1/0` out of the formula for a zero-event arm, rather than
    letting the interval blow up to +-inf or raise.
    """
    x1, n1 = float(a.sum()), len(a)
    x2, n2 = float(b.sum()), len(b)
    if x1 == 0 or x2 == 0:
        x1, n1, x2, n2 = x1 + 0.5, n1 + 1, x2 + 0.5, n2 + 1
    p1, p2 = x1 / n1, x2 / n2
    log_rr = np.log(p1 / p2)
    se = np.sqrt((1 - p1) / x1 + (1 - p2) / x2)
    return float(np.exp(log_rr - _Z975 * se)), float(np.exp(log_rr + _Z975 * se))


def _two_sample_cluster_ci(a, b, cluster_a, cluster_b, point_fn, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    cluster_a, cluster_b = np.asarray(cluster_a, dtype=object), np.asarray(cluster_b, dtype=object)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ra = _cluster_resample_array(a, cluster_a, rng)
        rb = _cluster_resample_array(b, cluster_b, rng)
        boots[i] = point_fn(ra, rb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def risk_difference(a, b, cluster_a=None, cluster_b=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Risk difference `mean(a) - mean(b)` with a 95% CI.

    `a`/`b` are flat 0/1 (or bool) outcome arrays. If `cluster_a`/`cluster_b`
    (participant ids, one per row of `a`/`b` respectively) are given, the CI
    is a participant-clustered bootstrap (resampling participants
    independently within each arm, sharing `cluster_bootstrap`'s per-draw
    resampling logic via `_cluster_resample_array`) -- this is the correct
    choice for real episode/repetition-level data (Correction 9.1). Without
    cluster ids, the CI falls back to a closed-form Wald interval that treats
    every row as an independent draw; that is only valid when `a`/`b` truly
    are i.i.d. (e.g. already one row per participant, or synthetic/unit-test
    data with no participant structure) -- passing raw episode-level rows
    without `cluster_a`/`cluster_b` will understate uncertainty.

    Never divides by a zero count, so a zero-event arm on either side never
    crashes; the CI is simply as wide (or as degenerate, if BOTH arms are
    all-zero or all-one) as the data support.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    estimate = _rd_point(a, b)
    if cluster_a is not None and cluster_b is not None:
        lo, hi = _two_sample_cluster_ci(a, b, cluster_a, cluster_b, _rd_point, n_boot, seed)
    else:
        lo, hi = _wald_rd_ci(a, b)
    return dict(estimate=estimate, lo=lo, hi=hi)


def risk_ratio(a, b, cluster_a=None, cluster_b=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Risk ratio `mean(a) / mean(b)` with a 95% CI.

    Same `cluster_a`/`cluster_b` contract as `risk_difference`. Zero-event
    cells are handled explicitly (see `_rr_point`/`_katz_rr_ci`'s
    Haldane-Anscombe correction, Correction 9.2): a zero-event arm on either
    side produces a finite, reported estimate and a finite CI -- never a bare
    `inf`, a `nan`, a crash, or a silently dropped cell. Applied to the
    clustered-bootstrap path too, so a bootstrap replicate that happens to
    resample zero events is handled the same way as the point estimate.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    estimate = _rr_point(a, b)
    if cluster_a is not None and cluster_b is not None:
        lo, hi = _two_sample_cluster_ci(a, b, cluster_a, cluster_b, _rr_point, n_boot, seed)
    else:
        lo, hi = _katz_rr_ci(a, b)
    return dict(estimate=estimate, lo=lo, hi=hi)


# --- multiplicity -----------------------------------------------------------

def bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR adjustment. Returns q-values in the
    SAME order as `pvals`, each in `[p_i, 1.0]` and monotone under the
    procedure's own step-up ordering (a larger raw p-value never gets a
    smaller q-value than a smaller one below it)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    running_min = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        running_min = min(running_min, p[i] * n / k)
        q[i] = running_min
    return np.clip(q, 0.0, 1.0)
