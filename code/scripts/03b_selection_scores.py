"""Derive per-selection reconstructed calibrated decoder confidence for the
ALS bigP3BCI cohort (StudyB, StudyF, StudyL, StudyN) from raw EEG.

Checkpointed per Test-phase (study, participant_id, session_id, condition)
unit: each unit's rows are written to their own part file under
output/intermediate/selection_scores_parts/ as soon as that unit finishes.
On restart, any unit whose part file already exists is skipped, so an
interrupted run resumes rather than restarting (Ruling 16 -- an earlier
attempt at this exact task lost ~35 minutes of an inline, non-resumable
`python3 -c` run to a machine-sleep interruption; see progress.md).

After the per-unit loop, all parts are assembled into
output/intermediate/selection_scores.parquet and joined against the
sibling project's online_trials_all20.csv to report the match rate.

Run: PYTHONPATH=code uv run python3 code/scripts/03b_selection_scores.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.eeg_scoring import (  # noqa: E402
    SELECTION_SCORE_COLUMNS,
    epoch_train_file,
    extract_test_stimuli,
    fit_files_for,
    fit_stimulus_classifier,
    index_source_files,
    resolve_fit_unit,
    score_test_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT.parent / "study_bigp3_als_calibration" / "data" / "source_cache"
INTERMEDIATE_DIR = REPO_ROOT / "output" / "intermediate"
PARTS_DIR = INTERMEDIATE_DIR / "selection_scores_parts"
OUTPUT_PATH = INTERMEDIATE_DIR / "selection_scores.parquet"
MANIFEST_PATH = INTERMEDIATE_DIR / "selection_scores_unit_manifest.csv"
ONLINE_TRIALS_CSV = (
    REPO_ROOT.parent / "study_bigp3_als_calibration" / "output" / "intermediate" / "online_trials_all20.csv"
)
MAX_WORKERS = 8
MANIFEST_COLUMNS = [
    "study", "participant_id", "session_id", "condition",
    "fit_session_id", "fit_condition", "status", "n_rows", "detail", "seconds",
]


def _part_path(unit: tuple[str, str, str, str]) -> Path:
    study, participant_id, session_id, condition = unit
    return PARTS_DIR / f"{study}__{participant_id}__{session_id}__{condition}.parquet"


def _append_manifest_row(row: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not MANIFEST_PATH.exists()
    pd.DataFrame([row], columns=MANIFEST_COLUMNS).to_csv(MANIFEST_PATH, mode="a", header=header, index=False)


def run(cache_path: Path = CACHE_PATH, max_workers: int = MAX_WORKERS) -> None:
    """Process every not-yet-checkpointed Test unit, writing one part file each."""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    files = index_source_files(cache_path)
    test_files = [f for f in files if f.phase == "Test"]

    by_unit: dict[tuple[str, str, str, str], list] = {}
    for f in test_files:
        by_unit.setdefault((f.study, f.participant_id, f.session_id, f.condition), []).append(f)

    units = sorted(by_unit)
    todo = [u for u in units if not _part_path(u).exists()]
    print(f"{len(units)} Test units total; {len(units) - len(todo)} already checkpointed; {len(todo)} to run")
    if not todo:
        return

    model_cache: dict[tuple, object] = {}
    train_epoch_cache: dict[Path, tuple] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for i, unit in enumerate(todo, start=1):
            study, participant_id, session_id, condition = unit
            t0 = time.time()
            unit_files = by_unit[unit]
            try:
                fit_session, fit_condition = resolve_fit_unit(
                    files, study, participant_id, session_id, condition
                )
                fit_key = (study, participant_id, fit_session, fit_condition)
                if fit_key not in model_cache:
                    ffiles = fit_files_for(files, study, participant_id, fit_session, fit_condition)
                    assert all(f.phase == "Train" for f in ffiles)  # no-leakage
                    missing = [f for f in ffiles if f.path not in train_epoch_cache]
                    if missing:
                        results = list(pool.map(epoch_train_file, [f.path for f in missing]))
                        for f, r in zip(missing, results):
                            train_epoch_cache[f.path] = r
                    model, _n_target, _n_nontarget = fit_stimulus_classifier(ffiles, train_epoch_cache)
                    model_cache[fit_key] = model
                model = model_cache[fit_key]

                stimuli_list = list(pool.map(extract_test_stimuli, [f.path for f in unit_files]))
                frames = [
                    score_test_file(sf, model, stimuli, fit_session, fit_condition)
                    for sf, stimuli in zip(unit_files, stimuli_list)
                ]
                unit_df = (
                    pd.concat(frames, ignore_index=True)
                    if frames
                    else pd.DataFrame(columns=SELECTION_SCORE_COLUMNS)
                )
                unit_df.to_parquet(_part_path(unit), index=False)
                _append_manifest_row(dict(
                    study=study, participant_id=participant_id, session_id=session_id, condition=condition,
                    fit_session_id=fit_session,
                    fit_condition=fit_condition if fit_condition is not None else "(pooled)",
                    status="ok", n_rows=len(unit_df), detail="", seconds=round(time.time() - t0, 2),
                ))
                print(f"  [{i}/{len(todo)}] {unit} -> {len(unit_df)} rows ({time.time() - t0:.1f}s)")
            except Exception as exc:  # noqa: BLE001 - checkpoint the failure and keep going
                pd.DataFrame(columns=SELECTION_SCORE_COLUMNS).to_parquet(_part_path(unit), index=False)
                _append_manifest_row(dict(
                    study=study, participant_id=participant_id, session_id=session_id, condition=condition,
                    fit_session_id="", fit_condition="", status="failed",
                    n_rows=0, detail=repr(exc), seconds=round(time.time() - t0, 2),
                ))
                print(f"  [{i}/{len(todo)}] FAILED {unit}: {exc!r}")


def assemble() -> pd.DataFrame:
    """Concatenate every checkpointed part into the final selection_scores.parquet."""
    # Exclude macOS AppleDouble sidecars (._name.parquet): this exFAT/APFS
    # volume writes one next to every real file, and pathlib's glob() -- unlike
    # a shell glob -- does match a leading dot, so an unfiltered glob would try
    # to parse those 4KB sidecars as parquet and fail (seen on the first run).
    parts = sorted(p for p in PARTS_DIR.glob("*.parquet") if not p.name.startswith("._"))
    frames = [pd.read_parquet(p) for p in parts]
    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=SELECTION_SCORE_COLUMNS)
    )
    df = df.sort_values(["relative_path", "trial_number"], ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    return df


def report(df: pd.DataFrame) -> None:
    print(f"\nwrote {len(df)} selection rows to {OUTPUT_PATH}")
    if ONLINE_TRIALS_CSV.exists():
        online = pd.read_csv(ONLINE_TRIALS_CSV)
        als_online = online[online["study"].isin(["StudyB", "StudyF", "StudyL", "StudyN"])]
        key_cols = ["relative_path", "trial_number"]
        merged = df.merge(als_online[key_cols], on=key_cols, how="inner")
        match_rate = len(merged) / len(als_online) if len(als_online) else float("nan")
        print(
            f"join vs online_trials_all20.csv (ALS rows={len(als_online)}): "
            f"{len(merged)} matched ({match_rate:.4%})"
        )
    else:
        print(f"WARNING: {ONLINE_TRIALS_CSV} not found, skipping join-rate check")


def main() -> None:
    t0 = time.time()
    run()
    df = assemble()
    report(df)
    print(f"\ntotal wall clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
