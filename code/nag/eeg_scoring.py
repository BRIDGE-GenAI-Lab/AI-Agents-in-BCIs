"""Per-selection reconstructed calibrated decoder confidence from raw ALS EEG.

bigP3BCI carries NO online decoder score: the 36 grid-cell EDF channels
(`A_1_1`..`9_6_6`) are binary stimulus-flash indicators, not classifier
output. Everything in this module is reconstructed from calibration EEG and
must be reported as reconstructed calibrated decoder confidence, never
described as the decoder's own true output (see nag.confidence and its
banned-string test).

Method, per fitted unit:
  1. Fit a stimulus-level target/non-target classifier on Train-phase
     calibration EEG only.
  2. Score Test-phase stimuli with that classifier -- Test data never
     touches the fit (see `resolve_fit_unit` / `fit_files_for`).
  3. For each Test selection, accumulate the per-stimulus scores onto the
     36 grid cells that stimulus illuminated across the whole flash
     sequence for that selection, then sum-normalise across the 36 cells.

This measures signal TRANSMISSION only. It never claims participant intent.

Reuses the validated calibration machinery in the sibling project
`study_bigp3_als_calibration/src/bigp3_als` (bandpass filter, epoch window,
artifact threshold, calibration-event detection, feature decimation, and
trial reconstruction) rather than re-deriving any of those numeric choices.
`extract_test_stimuli` below duplicates the *control flow* of
`bigp3_als.features._extract_file_epochs` (not its numeric choices) because
that function discards each retained stimulus's sample index and on/off
grid state once it has produced epoch/label arrays -- exactly the
information per-selection grid-cell attribution needs.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

_SIBLING_SRC = Path(__file__).resolve().parents[3] / "study_bigp3_als_calibration" / "src"
if str(_SIBLING_SRC) not in sys.path:
    sys.path.insert(0, str(_SIBLING_SRC))

from bigp3_als.edf import SHARED_EEG_CHANNELS, parse_source_path, select_edf_paths  # noqa: E402
from bigp3_als.features import (  # noqa: E402
    ARTIFACT_THRESHOLD_UV,
    EPOCH_END_SECONDS,
    EPOCH_START_SECONDS,
    MIN_NONTARGET_EPOCHS,
    MIN_TARGET_EPOCHS,
    _bandpass,
    _downsampled_epoch_features,
    _extract_file_epochs,
    minimum_sampling_frequency,
    select_calibration_events,
)
from bigp3_als.trials import _contiguous_start, reconstruct_edf_trials  # noqa: E402

from nag.episodes import ALS_STUDIES  # noqa: E402
from nag.taxonomy import _ALPHABET  # noqa: E402

FIT_RANDOM_STATE = 20260718  # same seed features.py uses for its own LR/CV fits


def _grid_channel_name(code: int) -> str:
    """36-cell grid channel name for a 1-indexed code, e.g. 1 -> 'A_1_1'.

    Verified against real EDF channel names across StudyB/F/L/N: channels
    are ordered A_1_1 .. Z_5_2, Sp_5_3, 1_5_4 .. 9_6_6 -- a 6x6 layout in
    row-major order, with '_' (space) rendered as 'Sp'. This is the SAME
    alphabet nag.taxonomy/nag.episodes use for grid codes, so a code here
    means the same cell it means there.
    """
    letter = _ALPHABET[code - 1]
    prefix = "Sp" if letter == "_" else letter
    row, col = divmod(code - 1, 6)
    return f"{prefix}_{row + 1}_{col + 1}"


GRID_CHANNELS: tuple[str, ...] = tuple(_grid_channel_name(c) for c in range(1, 37))
assert len(GRID_CHANNELS) == 36

_GRID_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_(?P<row>[0-9]+)_(?P<col>[0-9]+)$")


def resolve_grid_channels(ch_names: list[str]) -> list[str]:
    """The 36 grid-cell channel names for THIS file, in code order 1..36.

    Cell identity comes from the `<label>_<row>_<col>` suffix every grid
    channel carries, restricted to the 6x6 layout (row, col in 1..6) --
    never from the label text before it, and never from raw exclusion of
    known non-grid names. Two real deviations required this:
      - StudyB/B_03/SE001/Test/CB/..._Test12.edf labels its last cell
        `Bs_6_6` (Backspace) instead of the standard `9_6_6` -- the row/col
        suffix still identifies it as cell 36 regardless of prefix text.
      - StudyB/B_01/SE002/Test/CB/..._Test10.edf has SIX EXTRA channels
        beyond the standard 36 -- `Prd_7_1`, `Esc_7_2`, `BS_7_3`,
        `Pause_7_4`, `Sleep_7_5`, `Enter_7_6` -- a genuine 7th keyboard row
        of function keys on this one file, verified on disk (an
        exclude-known-non-grid-names approach found 42 channels here and
        raised; the codebase's frozen 36-character taxonomy alphabet
        (nag.taxonomy) cannot represent codes beyond 36 regardless, so a
        stimulus that only illuminates a row-7 cell correctly contributes
        no evidence to any of the 36 interpretable cells here).
    Raises if the row/col-restricted set is not exactly the 36 cells of a
    complete 6x6 grid (every (row, col) in 1..6 x 1..6 exactly once) --
    never silently accepting a partial or duplicated grid.
    """
    by_position: dict[tuple[int, int], str] = {}
    for ch in ch_names:
        match = _GRID_SUFFIX_RE.match(ch)
        if not match:
            continue
        row, col = int(match.group("row")), int(match.group("col"))
        if not (1 <= row <= 6 and 1 <= col <= 6):
            continue
        if (row, col) in by_position:
            raise ValueError(f"duplicate grid position ({row},{col}): {by_position[(row, col)]!r} and {ch!r}")
        by_position[(row, col)] = ch
    if len(by_position) != 36:
        raise ValueError(
            f"expected exactly 36 grid-cell channels (6x6), found {len(by_position)}: {sorted(by_position)}"
        )
    return [by_position[(row, col)] for row in range(1, 7) for col in range(1, 7)]


# ---------------------------------------------------------------------------
# File indexing and no-leakage-assertable fit-unit resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    study: str
    participant_id: str
    session_id: str
    phase: str
    condition: str


def index_source_files(cache_path: Path, studies: tuple[str, ...] = ALS_STUDIES) -> list[SourceFile]:
    """List every ALS-study EDF once, as plain metadata (no EEG touched)."""
    out = []
    for path in select_edf_paths(cache_path):
        relative_path = path.relative_to(cache_path).as_posix()
        source = parse_source_path(relative_path)
        if source.study in studies:
            out.append(
                SourceFile(
                    path=path,
                    relative_path=relative_path,
                    study=source.study,
                    participant_id=source.participant_id,
                    session_id=source.session_id,
                    phase=source.phase,
                    condition=source.condition,
                )
            )
    return out


def _session_number(session_id: str) -> int:
    return int(session_id.removeprefix("SE"))


def resolve_fit_unit(
    files: list[SourceFile], study: str, participant_id: str, test_session_id: str, condition: str
) -> tuple[str, Optional[str]]:
    """Resolve which Train data calibrates a given Test (session, condition).

    Prefers an EXACT match: Train data from the same session and condition
    as the Test unit (`condition` returned unchanged). bigP3BCI participants
    do not always have this though -- some sessions test a condition with no
    Train counterpart at all, and some studies (StudyF) calibrate once per
    session under one condition label and test several other condition
    labels against it. When no exact match exists, falls back to the
    NEAREST Train-bearing session at or before `test_session_id` for this
    participant (never a later one -- that would let calibration leak
    forward in time), pooling every condition in that session (returned
    condition is None, meaning "pooled"). Raises if no such session exists
    at all, rather than silently guessing.
    """
    train = [f for f in files if f.phase == "Train" and f.study == study and f.participant_id == participant_id]
    own = [f for f in train if f.session_id == test_session_id and f.condition == condition]
    if own:
        return test_session_id, condition
    sessions_with_train = {f.session_id for f in train}
    candidates = [s for s in sessions_with_train if _session_number(s) <= _session_number(test_session_id)]
    if not candidates:
        raise ValueError(
            f"no Train session at or before {test_session_id} for {study}:{participant_id}"
        )
    return max(candidates, key=_session_number), None


def classify_fit_match(
    session_id: str, condition: str, fit_session: str, fit_condition: Optional[str]
) -> str:
    """Categorise how a unit's fit relates to what it was tested on.

    Exactly one of three values, so a reviewer's "were all classifiers fit
    on matched data?" has a number, not a caveat:
      - "own_session_own_condition": the exact-match branch of
        `resolve_fit_unit` -- Train data from this unit's own session AND
        condition.
      - "own_session_other_condition": the pooled-fallback branch resolved
        to THIS SAME session (a different condition's Train data borrowed
        within-session -- fit_condition is None, meaning pooled).
      - "earlier_session": the pooled-fallback branch resolved to a
        strictly earlier session for this participant (never a later one --
        `resolve_fit_unit` guarantees that).
    """
    if fit_session == session_id and fit_condition == condition:
        return "own_session_own_condition"
    if fit_session == session_id:
        return "own_session_other_condition"
    return "earlier_session"


def fit_files_for(
    files: list[SourceFile], study: str, participant_id: str, fit_session_id: str, fit_condition: Optional[str]
) -> list[SourceFile]:
    """The exact Train files a fit unit is built from -- directly assertable
    (every returned file has phase == "Train") to verify no Test data ever
    reaches a fit."""
    return sorted(
        (
            f
            for f in files
            if f.phase == "Train"
            and f.study == study
            and f.participant_id == participant_id
            and f.session_id == fit_session_id
            and (fit_condition is None or f.condition == fit_condition)
        ),
        key=lambda f: f.relative_path,
    )


# ---------------------------------------------------------------------------
# Train-phase fitting
# ---------------------------------------------------------------------------


def epoch_train_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(epochs, labels) for one Train EDF, via the validated shared epoching."""
    epochs, labels, _sfreq, _n_pre_artifact = _extract_file_epochs(path)
    return epochs, labels


def fit_stimulus_classifier(
    fit_files: list[SourceFile], epoch_cache: dict[Path, tuple[np.ndarray, np.ndarray]]
) -> tuple[Pipeline, int, int]:
    """Fit standardised logistic regression on pooled Train epochs.

    Unscaled logistic regression overflows at this feature scale, hence
    StandardScaler in the pipeline. Returns (model, n_target, n_nontarget)
    so callers can enforce the same MIN_TARGET_EPOCHS/MIN_NONTARGET_EPOCHS
    viability gate features.py uses for its own calibration fits.
    """
    if not fit_files:
        raise ValueError("fit_stimulus_classifier: no Train files given")
    epochs_list, labels_list = [], []
    for f in fit_files:
        epochs, labels = epoch_cache[f.path]
        epochs_list.append(epochs)
        labels_list.append(labels)
    epochs = np.concatenate(epochs_list)
    labels = np.concatenate(labels_list)
    n_target = int((labels == 1).sum())
    n_nontarget = int((labels == 0).sum())
    if n_target < MIN_TARGET_EPOCHS or n_nontarget < MIN_NONTARGET_EPOCHS:
        raise ValueError(
            f"insufficient calibration epochs: target={n_target} nontarget={n_nontarget}"
        )
    features = _downsampled_epoch_features(epochs)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=FIT_RANDOM_STATE),
    )
    model.fit(features, labels)
    return model, n_target, n_nontarget


# ---------------------------------------------------------------------------
# Test-phase stimulus extraction with grid-cell / trial attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestFileStimuli:
    features: np.ndarray  # (n, n_features) decimated epoch features, ready for predict_proba
    grid_on: np.ndarray  # (n, 36) bool -- which cells this stimulus illuminated
    trial_number: np.ndarray  # (n,) int -- 1-indexed selection this stimulus belongs to
    n_stimuli_total: int  # retained (artifact-free) stimuli before trial-window attribution
    n_unattributed: int  # retained stimuli whose sample fell in no selection window


def _selection_windows(phase: np.ndarray) -> list[tuple[int, int, int]]:
    """(trial_number, phase2_window_start, phase3_start) per selection.

    Mirrors bigp3_als.trials.reconstruct_online_trials's own rule for the
    flash sequence behind a trial: the immediately preceding contiguous
    phase-2 block. A phase-3 block not immediately preceded by phase 2 gets
    no window here, exactly as reconstruct_online_trials leaves that
    trial's `target` as None.
    """
    phase3_starts = np.flatnonzero((phase == 3) & np.r_[True, phase[:-1] != 3])
    windows = []
    for trial_number, phase3_start in enumerate(phase3_starts, start=1):
        preceding = phase3_start - 1
        if preceding < 0 or phase[preceding] != 2:
            continue
        phase2_start = _contiguous_start(phase, int(preceding), 2)
        windows.append((trial_number, int(phase2_start), int(phase3_start)))
    return windows


def _assign_trial_numbers(samples: np.ndarray, windows: list[tuple[int, int, int]]) -> np.ndarray:
    trial_number = np.full(len(samples), -1, dtype=int)
    for tn, start, end in windows:
        trial_number[(samples >= start) & (samples < end)] = tn
    return trial_number


_EMPTY_STIMULI = TestFileStimuli(
    features=np.zeros((0, 1)),
    grid_on=np.zeros((0, 36), dtype=bool),
    trial_number=np.zeros((0,), dtype=int),
    n_stimuli_total=0,
    n_unattributed=0,
)


def extract_test_stimuli(edf_path: Path) -> TestFileStimuli:
    """Per-stimulus classifier-ready features and grid/trial attribution for
    one Test EDF. See module docstring for why this duplicates
    _extract_file_epochs's control flow instead of calling it directly."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    required = set(SHARED_EEG_CHANNELS) | {"StimulusBegin", "StimulusType", "PhaseInSequence"}
    missing = sorted(required - set(raw.ch_names))
    if missing:
        raise ValueError(f"missing channels in {edf_path}: {', '.join(missing)}")
    grid_channels = resolve_grid_channels(raw.ch_names)  # per-file: labels vary (e.g. Bs_6_6 not 9_6_6)
    sampling_frequency = float(raw.info["sfreq"])
    required_rate = minimum_sampling_frequency()
    if sampling_frequency < required_rate:
        raise ValueError(
            f"sampling frequency {sampling_frequency:g} Hz in {edf_path} is below "
            f"the {required_rate:g} Hz required"
        )

    eeg = _bandpass(raw.get_data(picks=list(SHARED_EEG_CHANNELS)), sampling_frequency)
    stimulus_begin, stimulus_type, phase = raw.get_data(
        picks=["StimulusBegin", "StimulusType", "PhaseInSequence"]
    )
    samples, _labels = select_calibration_events(stimulus_begin, phase, stimulus_type)

    pre_samples = round(-EPOCH_START_SECONDS * sampling_frequency)
    post_samples = round(EPOCH_END_SECONDS * sampling_frequency)
    valid = (samples >= pre_samples) & (samples + post_samples <= eeg.shape[1])
    samples = samples[valid]
    if len(samples) == 0:
        return _EMPTY_STIMULI

    epochs = np.stack([eeg[:, s - pre_samples : s + post_samples] for s in samples])
    epochs = epochs - epochs[:, :, :pre_samples].mean(axis=2, keepdims=True)
    artifact_free = np.max(np.abs(epochs), axis=(1, 2)) * 1e6 <= ARTIFACT_THRESHOLD_UV
    samples, epochs = samples[artifact_free], epochs[artifact_free]
    if len(samples) == 0:
        return _EMPTY_STIMULI

    grid_raw = raw.get_data(picks=grid_channels)
    grid_on = (np.rint(grid_raw[:, samples]) != 0).T  # (n, 36)

    phase_i = np.rint(phase).astype(int)
    windows = _selection_windows(phase_i)
    trial_number = _assign_trial_numbers(samples, windows)
    keep = trial_number >= 0

    features = _downsampled_epoch_features(epochs[keep]) if keep.any() else np.zeros((0, 1))
    return TestFileStimuli(
        features=features,
        grid_on=grid_on[keep],
        trial_number=trial_number[keep],
        n_stimuli_total=int(len(samples)),
        n_unattributed=int((~keep).sum()),
    )


# ---------------------------------------------------------------------------
# Score assembly: per-stimulus classifier output -> per-selection grid distribution
# ---------------------------------------------------------------------------

SELECTION_SCORE_COLUMNS = [
    "relative_path",
    "trial_number",
    "study",
    "participant_id",
    "study_participant_id",
    "session_id",
    "condition",
    "target",
    "selected",
    "correct",
    "score_top",
    "score_target",
    "margin",
    "n_stimuli",
    "fit_session",
    "fit_condition",
    "fit_match",
]


def normalize_grid_scores(grid_on: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Sum-normalised distribution over the 36 grid cells for one selection.

    Sum-normalise (not softmax): the per-stimulus scores are already
    calibrated-scale probabilities in [0, 1], and summing them per cell
    across repeated flashes is the standard P300-speller evidence
    accumulation (more hits at a higher score -> more accumulated evidence).
    Softmax would apply an arbitrary exponential/temperature distortion to
    values that are already on a meaningful probability scale, amplifying
    differences the raw sums do not warrant. NaN (all 36 cells all-zero,
    i.e. no stimuli or all-zero scores) when there is nothing to normalise.
    """
    accum = (grid_on.astype(float) * scores[:, None]).sum(axis=0)
    total = accum.sum()
    if total <= 0:
        return np.full(36, np.nan)
    return accum / total


def score_test_file(
    source_file: SourceFile,
    model: Pipeline,
    stimuli: TestFileStimuli,
    fit_session: str,
    fit_condition: Optional[str],
) -> pd.DataFrame:
    """One row per selection in this Test EDF, joined against its own
    reconstructed trial table (same code path online_trials_all20.csv was
    built from -- reconstruct_edf_trials -- so the join back to that CSV on
    (relative_path, trial_number) is exact by construction).

    `fit_session`/`fit_condition` are the resolved fit unit this file's
    classifier actually came from (`resolve_fit_unit`'s return value) --
    stamped onto every row (with the derived `fit_match` category) so a
    borrowed fit is distinguishable from a matched one downstream, per
    Ruling 20.
    """
    fit_match = classify_fit_match(source_file.session_id, source_file.condition, fit_session, fit_condition)
    trial_table = reconstruct_edf_trials(source_file.path, source_file.relative_path)
    if len(stimuli.features) == 0:
        scores = np.zeros(0)
    else:
        scores = model.predict_proba(stimuli.features)[:, 1]

    by_trial: dict[int, tuple[np.ndarray, int]] = {}
    for trial_number in np.unique(stimuli.trial_number):
        mask = stimuli.trial_number == trial_number
        normalized = normalize_grid_scores(stimuli.grid_on[mask], scores[mask])
        by_trial[int(trial_number)] = (normalized, int(mask.sum()))

    rows = []
    for row in trial_table.itertuples(index=False):
        trial_number = int(row.trial_number)
        if trial_number not in by_trial:
            continue  # no attributable stimuli survived artifact rejection for this selection
        normalized, n_stimuli = by_trial[trial_number]
        if np.all(np.isnan(normalized)):
            score_top = margin = np.nan
        else:
            top_two = np.sort(normalized)[::-1][:2]
            score_top = float(top_two[0])
            margin = float(top_two[0] - top_two[1]) if len(top_two) > 1 else np.nan
        target = row.target
        if target is not None and not (isinstance(target, float) and np.isnan(target)) and 1 <= int(target) <= 36:
            score_target = float(normalized[int(target) - 1])
        else:
            score_target = np.nan
        rows.append(
            dict(
                relative_path=source_file.relative_path,
                trial_number=trial_number,
                study=source_file.study,
                participant_id=source_file.participant_id,
                study_participant_id=f"{source_file.study}:{source_file.participant_id}",
                session_id=source_file.session_id,
                condition=source_file.condition,
                target=row.target,
                selected=row.selected,
                correct=row.correct,
                score_top=score_top,
                score_target=score_target,
                margin=margin,
                n_stimuli=n_stimuli,
                fit_session=fit_session,
                fit_condition=fit_condition if fit_condition is not None else "(pooled)",
                fit_match=fit_match,
            )
        )
    return pd.DataFrame(rows, columns=SELECTION_SCORE_COLUMNS)
