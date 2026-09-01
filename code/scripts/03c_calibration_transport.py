"""Calibrate reconstructed per-selection decoder confidence and report its
reliability, in-distribution and under cross-study transport.

Reads output/intermediate/selection_scores.parquet (Task 3b). Confidence for
calibration purposes is `score_top`: the sum-normalised score the
reconstructed classifier assigns to its own top-ranked grid cell for a
selection -- the natural stand-in for "the decoder's confidence in the
decision it made," since P300-speller online decisions are themselves the
argmax over accumulated per-cell evidence (the same accumulation rule used
here). The ground-truth label is `correct` (target == selected), i.e.
whether the transmission the agent downstream would act on was faithful --
NOT whether this reconstruction's own argmax matches the target, which is a
different (reconstruction-fidelity) question reported separately as a
diagnostic.

Two calibration reports, each answering a different generalisation question:

1. output/tables/calibration_reliability.csv -- in-distribution reliability.
   ONE `fit_calibrator` call over all valid ALS rows, participant-grouped.
   Every reported ece/brier here comes from `fit.oof` (true GroupKFold
   held-out predictions), never from `fit.calibrator.transform()` on data it
   was fit on -- that would be optimistically biased (nag.confidence
   docstring; see also Ruling 12). Per study and overall.

2. output/tables/calibration_transport.csv -- cross-study transport.
   For every ordered (train_study, test_study) pair among StudyB/F/L/N,
   fit_calibrator is called on ONLY train_study's rows, then
   `.calibrator.transform()` is applied to test_study's raw scores.
   test_study's participants never appear in that fit at all -- a stronger
   separation than participant-grouped OOF within one study -- so using
   `.calibrator` here is correct, not a violation of the fit.oof rule: the
   rule exists to prevent evaluating a calibrator on data it has seen, and
   the transport calibrator has seen neither test_study's scores nor its
   participants. `.oof` cannot answer this question at all, since it is by
   construction restricted to held-out folds *within* the training fit's own
   groups.

Run: PYTHONPATH=code uv run python3 code/scripts/03c_calibration_transport.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.confidence import fit_calibrator, reliability  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = REPO_ROOT / "output" / "intermediate" / "selection_scores.parquet"
RELIABILITY_OUT = REPO_ROOT / "output" / "tables" / "calibration_reliability.csv"
TRANSPORT_OUT = REPO_ROOT / "output" / "tables" / "calibration_transport.csv"
ALS_STUDIES = ("StudyB", "StudyF", "StudyL", "StudyN")

UNEVALUABLE = dict(ece=np.nan, brier=np.nan, n=0)


def _load_valid_rows() -> pd.DataFrame:
    df = pd.read_parquet(SCORES_PATH)
    df = df[df["study"].isin(ALS_STUDIES)].copy()
    valid = df["score_top"].notna() & df["correct"].notna()
    dropped = int((~valid).sum())
    if dropped:
        print(
            f"dropping {dropped}/{len(df)} rows with no computable score_top or "
            "unknown correctness (ineligible trial, or zero illuminated-cell "
            "evidence) before calibration"
        )
    out = df[valid].copy()
    out["correct_int"] = out["correct"].astype(bool).astype(int)
    return out


def build_reliability_table(df: pd.DataFrame) -> pd.DataFrame:
    """In-distribution reliability, OOF-only, per study and overall."""
    fit = fit_calibrator(df["score_top"].to_numpy(), df["correct_int"].to_numpy(), df["participant_id"].to_numpy())
    rows = []
    if fit.oof is None:
        print("WARNING: fit.oof is None (fewer than 2 participant groups) -- every cell unevaluable")
        for study in (*ALS_STUDIES, "overall"):
            rows.append(dict(study=study, **UNEVALUABLE))
        return pd.DataFrame(rows)

    oof = fit.oof
    correct = df["correct_int"].to_numpy()
    for study in ALS_STUDIES:
        mask = (df["study"] == study).to_numpy()
        if mask.sum() == 0:
            rows.append(dict(study=study, **UNEVALUABLE))
            continue
        r = reliability(oof[mask], correct[mask])
        rows.append(dict(study=study, ece=r["ece"], brier=r["brier"], n=int(mask.sum())))
    r_all = reliability(oof, correct)
    rows.append(dict(study="overall", ece=r_all["ece"], brier=r_all["brier"], n=len(df)))
    return pd.DataFrame(rows)


def build_transport_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise train-one-study/test-another cross-study transport."""
    rows = []
    for train_study in ALS_STUDIES:
        train = df[df["study"] == train_study]
        if train["participant_id"].nunique() < 2:
            for test_study in ALS_STUDIES:
                if test_study == train_study:
                    continue
                rows.append(dict(train_study=train_study, test_study=test_study, **UNEVALUABLE))
            continue
        fit = fit_calibrator(
            train["score_top"].to_numpy(), train["correct_int"].to_numpy(), train["participant_id"].to_numpy()
        )
        for test_study in ALS_STUDIES:
            if test_study == train_study:
                continue
            test = df[df["study"] == test_study]
            if len(test) == 0:
                rows.append(dict(train_study=train_study, test_study=test_study, **UNEVALUABLE))
                continue
            calibrated = fit.calibrator.transform(test["score_top"].to_numpy())
            r = reliability(calibrated, test["correct_int"].to_numpy())
            rows.append(dict(train_study=train_study, test_study=test_study, ece=r["ece"], brier=r["brier"], n=len(test)))
    return pd.DataFrame(rows)


def main() -> None:
    if not SCORES_PATH.exists():
        raise FileNotFoundError(f"{SCORES_PATH} not found -- run 03b_selection_scores.py first")
    df = _load_valid_rows()
    print(f"{len(df)} valid ALS selections across {df['participant_id'].nunique()} participants")

    reliability_table = build_reliability_table(df)
    RELIABILITY_OUT.parent.mkdir(parents=True, exist_ok=True)
    reliability_table.to_csv(RELIABILITY_OUT, index=False)
    print(f"\nwrote {RELIABILITY_OUT}")
    print(reliability_table.to_string(index=False))

    transport_table = build_transport_table(df)
    transport_table.to_csv(TRANSPORT_OUT, index=False)
    print(f"\nwrote {TRANSPORT_OUT}")
    print(transport_table.to_string(index=False))


if __name__ == "__main__":
    main()
