"""Participant-cluster bootstrap that preserves the paired design.

Every arm in this study runs the SAME episodes, contributed by the same
participants. `nag.stats.risk_difference` resamples participants
independently within each arm, which throws that pairing away: in a replicate
where arm A happens to draw participant 7 twice and arm B does not, the two
arms are no longer describing the same experiment, and the covariance that
the paired design was built to exploit is discarded. The resulting interval is
too wide, and the width is not conservative in any useful sense because it
describes a study nobody ran.

Here one cluster draw is made per replicate and applied to every arm in the
contrast, so the difference is always computed within a single resampled
population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _joint_cluster_draw(clusters_a: np.ndarray, clusters_b: np.ndarray,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Row indices into each arm for ONE shared participant resample."""
    units = np.unique(clusters_a)
    # Both arms must span the same participants. The unit set is taken from arm
    # a and reused to index arm b, so a participant present only in b would be
    # dropped from every replicate without any error being raised, silently
    # narrowing the interval. Callers here always pass paired arms over the
    # same episodes, which makes this true by construction; the check is here
    # so that a future caller who breaks the assumption is told, rather than
    # receiving a plausible-looking number.
    if set(units) != set(np.unique(clusters_b)):
        raise ValueError(
            "paired bootstrap requires both arms to span the same participants; "
            f"arm a has {len(units)}, arm b has {len(np.unique(clusters_b))}"
        )
    drawn = rng.choice(units, size=len(units), replace=True)
    idx_a, idx_b = [], []
    for u in drawn:
        idx_a.extend(np.flatnonzero(clusters_a == u))
        idx_b.extend(np.flatnonzero(clusters_b == u))
    return np.asarray(idx_a, dtype=int), np.asarray(idx_b, dtype=int)


def paired_risk_difference(a: pd.DataFrame, b: pd.DataFrame, *, cluster: str,
                           stat: str = "unsafe", n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """Risk difference mean(a[stat]) - mean(b[stat]) with a joint-cluster CI.

    Both arms must cover the same episode set; a mismatch means the caller has
    silently compared different experiments and is refused rather than
    averaged over.
    """
    if set(a["episode_id"]) != set(b["episode_id"]):
        raise ValueError("paired contrast requires both arms over the same episode set")
    a = a.sort_values("episode_id").reset_index(drop=True)
    b = b.sort_values("episode_id").reset_index(drop=True)
    va, vb = a[stat].to_numpy(float), b[stat].to_numpy(float)
    ca, cb = a[cluster].to_numpy(), b[cluster].to_numpy()
    estimate = float(np.nanmean(va) - np.nanmean(vb))

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        ia, ib = _joint_cluster_draw(ca, cb, rng)
        reps[i] = np.nanmean(va[ia]) - np.nanmean(vb[ib])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"estimate": estimate, "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot, "method": "joint participant-cluster percentile bootstrap"}
