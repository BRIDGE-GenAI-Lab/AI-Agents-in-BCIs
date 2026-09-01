"""Reconstructed decoder confidence and its calibration.

There is NO online decoder score in bigP3BCI: the 36 grid-cell EDF channels are
binary stimulus-flash indicators. Everything here is reconstructed from
calibration EEG and must be reported as reconstructed calibrated confidence.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold


@dataclass
class Calibrator:
    iso: IsotonicRegression

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(self.iso.predict(np.asarray(scores, dtype=float)), 1e-6, 1 - 1e-6)


class CalibrationFit(NamedTuple):
    """Result of `fit_calibrator`. Both fields must be considered.

    calibrator: an IsotonicRegression fit on ALL the input data. This is the
        object to deploy -- call `.transform()` on NEW scores with it. It has
        seen every participant, so it must NEVER be used to report a
        reliability number: doing so is optimistically biased.
    oof: out-of-fold predictions from GroupKFold cross-validation over
        participant `groups`, so no participant appears on both the fit and
        the evaluation side of any fold. This is what `reliability()` must be
        computed against for ANY reported ece/brier. `None` when there were
        fewer than 2 unique participant groups, i.e. cross-validated
        evaluation was not possible -- callers must not treat `None` as
        though it were valid out-of-fold data (for example by silently
        falling back to in-sample predictions).
    """
    calibrator: Calibrator
    oof: Optional[np.ndarray]


def fit_calibrator(scores, correct, groups, n_splits: int = 5) -> CalibrationFit:
    """Isotonic calibration fitted with participant-level separation.

    groups must be participant ids so no participant appears in both the fit and
    the evaluation of any fold. Returns a `CalibrationFit(calibrator, oof)`: see
    that class for which field is safe to use for reporting.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    groups = np.asarray(groups)
    n = min(n_splits, len(np.unique(groups)))
    oof = None
    if n >= 2:
        oof = np.zeros_like(scores)
        for tr, te in GroupKFold(n_splits=n).split(scores, correct, groups):
            f = IsotonicRegression(out_of_bounds="clip").fit(scores[tr], correct[tr])
            oof[te] = np.clip(f.predict(scores[te]), 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores, correct)
    return CalibrationFit(calibrator=Calibrator(iso=iso), oof=oof)


def reliability(p, y, n_bins: int = 10) -> dict:
    """Expected calibration error, Brier score, and per-bin detail."""
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            bins.append(dict(bin=b, n=0, mean_p=np.nan, frac_pos=np.nan)); continue
        mp, fp = float(p[m].mean()), float(y[m].mean())
        ece += (m.sum() / len(p)) * abs(mp - fp)
        bins.append(dict(bin=b, n=int(m.sum()), mean_p=mp, frac_pos=fp))
    return dict(ece=float(ece), brier=float(np.mean((p - y) ** 2)), bins=bins)


def episode_confidence(selection_confidences, expected_len: Optional[int] = None) -> float:
    """Joint probability that every selection in the episode transmitted correctly.

    An episode is faithful only if the whole string arrived correctly, so the
    product of per-selection calibrated confidences is the natural episode-level
    quantity. Returns 1.0 for an empty episode with no `expected_len` given
    (vacuously certain: there is nothing that could have transmitted
    incorrectly) -- this is the existing, still-supported default.

    `expected_len`: pass this when wiring real per-episode selection
    confidences (e.g. from selection_scores.parquet) so a short or empty
    episode RAISES instead of silently returning 1.0. A pipeline bug that
    drops or truncates selections would otherwise present as vacuous
    certainty -- exactly backwards, since high confidence is what licenses
    the agent to act. Not applied when `expected_len` is omitted, so
    existing callers of the un-parameterized function are unaffected.
    """
    conf = np.asarray(list(selection_confidences), dtype=float)
    if expected_len is not None and conf.size != expected_len:
        raise ValueError(
            f"episode_confidence: expected {expected_len} selections, got {conf.size}"
        )
    if conf.size == 0:
        return 1.0
    return float(np.prod(conf))
