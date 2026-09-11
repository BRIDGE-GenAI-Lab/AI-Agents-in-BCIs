"""Calibration of the EPISODE score, not just of its ingredients.

The gate thresholds the product of three calibrated per-selection
probabilities. That product equals the probability that all three characters
are correct only if the three correctness events are conditionally independent
given their scores, which is not established: the three selections share a
participant, a session, an electrode montage, and a fatigue state, so positive
dependence is the expected direction and the product is expected to be
under-confident.

Selection-level calibration does not transfer to the product. Since the
product is the quantity actually gated on, it has to be validated directly,
and against alternative combination rules that make different independence
assumptions.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

RULES = ("product", "min", "mean", "logsum")


def combine(per_selection: np.ndarray, rule: str) -> np.ndarray:
    """Combine per-selection calibrated probabilities into one episode score.

    `logsum` returns the summed log probability, which is monotone with
    `product` and therefore induces an identical threshold sweep; it is kept
    because it is numerically stable for long episodes and because reporting
    both makes the monotonicity explicit rather than assumed.
    """
    x = np.asarray(per_selection, dtype=float)
    if rule == "product":
        return x.prod(axis=1)
    if rule == "min":
        return x.min(axis=1)
    if rule == "mean":
        return x.mean(axis=1)
    if rule == "logsum":
        return np.log(np.clip(x, 1e-12, 1.0)).sum(axis=1)
    raise ValueError(f"unknown combination rule {rule!r}; expected one of {RULES}")


def episode_reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability of an episode-level score against episode-level correctness.

    `correct` is 1 when EVERY selection in the episode was decoded correctly,
    which is the event the product rule claims to estimate.
    """
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            bins.append({"bin": b, "n": 0, "mean_conf": None, "observed": None})
            continue
        mc, ob = float(conf[m].mean()), float(correct[m].mean())
        bins.append({"bin": b, "n": int(m.sum()), "mean_conf": mc, "observed": ob})
        ece += (m.sum() / len(conf)) * abs(mc - ob)
    single_class = len(np.unique(correct)) < 2
    return {
        "ece": float(ece),
        "brier": float(brier_score_loss(correct, np.clip(conf, 0, 1))),
        "auroc": float(roc_auc_score(correct, conf)) if not single_class else float("nan"),
        "auprc": float(average_precision_score(correct, conf)) if not single_class else float("nan"),
        "bins": bins,
    }
