import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nag.eeg_scoring as eeg_scoring
from nag.confidence import episode_confidence
from nag.eeg_scoring import (
    GRID_CHANNELS,
    SELECTION_SCORE_COLUMNS,
    SHARED_EEG_CHANNELS,
    SourceFile,
    classify_fit_match,
    fit_files_for,
    normalize_grid_scores,
    resolve_fit_unit,
    resolve_grid_channels,
)

_SELECTION_SCORES = "output/intermediate/selection_scores.parquet"
_ONLINE_TRIALS_CSV = "../study_bigp3_als_calibration/output/intermediate/online_trials_all20.csv"


def _sf(phase, session, condition, participant="P_01", study="StudyX"):
    return SourceFile(
        path=Path(f"/fake/{study}/{participant}/{session}/{phase}/{condition}/f.edf"),
        relative_path=f"bigP3BCI-data/{study}/{participant}/{session}/{phase}/{condition}/f.edf",
        study=study,
        participant_id=participant,
        session_id=session,
        phase=phase,
        condition=condition,
    )


# --- no-leakage: the fit file list is directly assertable -----------------


def test_fit_files_for_never_returns_a_test_phase_file():
    files = [_sf("Train", "SE001", "CB"), _sf("Test", "SE001", "CB"), _sf("Train", "SE002", "CB")]
    fit = fit_files_for(files, "StudyX", "P_01", "SE001", "CB")
    assert len(fit) == 1
    assert all(f.phase == "Train" for f in fit)


def test_resolve_fit_unit_prefers_exact_session_condition_match():
    files = [_sf("Train", "SE001", "CB"), _sf("Train", "SE002", "CB"), _sf("Test", "SE002", "CB")]
    fit_session, fit_condition = resolve_fit_unit(files, "StudyX", "P_01", "SE002", "CB")
    assert (fit_session, fit_condition) == ("SE002", "CB")


def test_resolve_fit_unit_falls_back_to_nearest_earlier_session_pooled_across_conditions():
    # SE003 tests condition RC, which has no Train anywhere for this participant;
    # only SE001's Train/CB exists. Mirrors the real StudyF Train-CBCol /
    # Test-Dyn,DynBigram,Static pattern and the real StudyB test-only-session gap.
    files = [_sf("Train", "SE001", "CB"), _sf("Test", "SE001", "CB"), _sf("Test", "SE003", "RC")]
    fit_session, fit_condition = resolve_fit_unit(files, "StudyX", "P_01", "SE003", "RC")
    assert fit_session == "SE001"
    assert fit_condition is None  # pooled fallback, never a fabricated exact match


def test_resolve_fit_unit_never_uses_a_later_session():
    files = [_sf("Train", "SE001", "CB"), _sf("Train", "SE005", "CB"), _sf("Test", "SE002", "CB")]
    fit_session, _ = resolve_fit_unit(files, "StudyX", "P_01", "SE002", "CB")
    assert fit_session == "SE001"  # not SE005, which would leak calibration backward in time


def test_resolve_fit_unit_raises_when_no_train_session_exists_at_all():
    files = [_sf("Test", "SE001", "CB")]
    with pytest.raises(ValueError):
        resolve_fit_unit(files, "StudyX", "P_01", "SE001", "CB")


def test_fit_files_for_only_matches_the_resolved_participant_and_session():
    files = [
        _sf("Train", "SE001", "CB", participant="P_01"),
        _sf("Train", "SE001", "CB", participant="P_02"),
        _sf("Train", "SE002", "CB", participant="P_01"),
    ]
    fit = fit_files_for(files, "StudyX", "P_01", "SE001", "CB")
    assert [f.relative_path for f in fit] == [
        "bigP3BCI-data/StudyX/P_01/SE001/Train/CB/f.edf"
    ]


def test_eeg_scoring_never_uses_the_banned_actual_posterior_phrase():
    assert "actual posterior" not in inspect.getsource(eeg_scoring).lower()


# --- Ruling 20: fit provenance must be recoverable from the data itself ---


def test_classify_fit_match_own_session_own_condition():
    assert classify_fit_match("SE002", "CB", "SE002", "CB") == "own_session_own_condition"


def test_classify_fit_match_own_session_other_condition():
    # pooled fallback (fit_condition is None) that still resolved to THIS session
    assert classify_fit_match("SE002", "RC", "SE002", None) == "own_session_other_condition"


def test_classify_fit_match_earlier_session():
    assert classify_fit_match("SE003", "RC", "SE001", None) == "earlier_session"


# --- grid-channel resolution: label text varies (Bs_6_6, not just 9_6_6) ---


def test_resolve_grid_channels_handles_the_real_backspace_label_variant():
    # StudyB/B_03/SE001/Test/CB/..._Test12.edf labels its last grid cell
    # 'Bs_6_6' (Backspace) instead of the standard '9_6_6' -- verified on
    # disk, and the reason extract_test_stimuli used to raise "missing
    # channels ... 9_6_6" on exactly this file. Cell identity must come from
    # position/exclusion, not the literal label text.
    standard_last_five = list(GRID_CHANNELS[-5:-1]) + ["Bs_6_6"]
    ch_names = list(SHARED_EEG_CHANNELS) + list(GRID_CHANNELS[:-1]) + ["Bs_6_6"] + [
        "StimulusType", "SelectedTarget", "SelectedRow", "SelectedColumn",
        "PhaseInSequence", "StimulusBegin", "StimulusCode", "CurrentTarget",
        "FakeFeedback", "DisplayResults",
    ]
    resolved = resolve_grid_channels(ch_names)
    assert len(resolved) == 36
    assert resolved[-1] == "Bs_6_6"
    assert resolved[:-1] == list(GRID_CHANNELS[:-1])
    del standard_last_five  # only used to spell out intent above


def test_resolve_grid_channels_raises_when_not_exactly_36():
    ch_names = list(SHARED_EEG_CHANNELS) + list(GRID_CHANNELS[:-1]) + [  # one short
        "StimulusType", "SelectedTarget", "SelectedRow", "SelectedColumn",
        "PhaseInSequence", "StimulusBegin", "StimulusCode", "CurrentTarget",
        "FakeFeedback", "DisplayResults",
    ]
    with pytest.raises(ValueError):
        resolve_grid_channels(ch_names)


def test_resolve_grid_channels_ignores_a_real_extra_seventh_row_of_function_keys():
    # StudyB/B_01/SE002/Test/CB/..._Test10.edf has SIX extra channels beyond
    # the standard 36 -- Prd_7_1, Esc_7_2, BS_7_3, Pause_7_4, Sleep_7_5,
    # Enter_7_6, a genuine 7th keyboard row -- verified on disk. The 36
    # interpretable cells (rows 1-6) must still resolve correctly; the extra
    # row-7 channels are outside the 6x6 layout nag.taxonomy's frozen
    # alphabet can represent, so they must be excluded, not counted as
    # "extra" grid cells that make the file invalid.
    extra_row = ["Prd_7_1", "Esc_7_2", "BS_7_3", "Pause_7_4", "Sleep_7_5", "Enter_7_6"]
    ch_names = list(SHARED_EEG_CHANNELS) + list(GRID_CHANNELS) + extra_row + [
        "StimulusType", "SelectedTarget", "SelectedRow", "SelectedColumn",
        "PhaseInSequence", "StimulusBegin", "StimulusCode", "CurrentTarget",
        "FakeFeedback", "DisplayResults",
    ]
    resolved = resolve_grid_channels(ch_names)
    assert len(resolved) == 36
    assert resolved == list(GRID_CHANNELS)
    assert not any(ch in resolved for ch in extra_row)


def test_resolve_grid_channels_raises_on_a_duplicate_grid_position():
    ch_names = list(SHARED_EEG_CHANNELS) + list(GRID_CHANNELS) + ["Dup_1_1"] + [
        "StimulusType", "SelectedTarget", "SelectedRow", "SelectedColumn",
        "PhaseInSequence", "StimulusBegin", "StimulusCode", "CurrentTarget",
        "FakeFeedback", "DisplayResults",
    ]
    with pytest.raises(ValueError):
        resolve_grid_channels(ch_names)


# --- score normalisation ----------------------------------------------------


def test_normalize_grid_scores_sums_to_one_across_36_cells():
    rng = np.random.default_rng(0)
    grid_on = rng.integers(0, 2, size=(50, 36)).astype(bool)
    grid_on[:, 0] = True  # guarantee at least one illuminated cell somewhere
    scores = rng.uniform(0.1, 0.9, size=50)
    normalized = normalize_grid_scores(grid_on, scores)
    assert normalized.shape == (36,)
    assert np.isclose(normalized.sum(), 1.0)
    assert np.all(normalized >= 0)


def test_normalize_grid_scores_is_all_nan_when_no_cell_is_ever_illuminated():
    grid_on = np.zeros((5, 36), dtype=bool)
    scores = np.array([0.5] * 5)
    normalized = normalize_grid_scores(grid_on, scores)
    assert np.all(np.isnan(normalized))


def test_normalize_grid_scores_weights_repeated_hits_more_than_single_hits():
    # cell 0 illuminated twice at high score, cell 1 illuminated once at the
    # same score -- sum-normalisation should favor the repeatedly-hit cell.
    grid_on = np.zeros((3, 36), dtype=bool)
    grid_on[0, 0] = grid_on[1, 0] = True
    grid_on[2, 1] = True
    scores = np.array([0.8, 0.8, 0.8])
    normalized = normalize_grid_scores(grid_on, scores)
    assert normalized[0] > normalized[1]


# --- Ruling 13: episode_confidence must not be vacuously certain on real wiring ---


def test_episode_confidence_raises_on_empty_episode_when_length_is_asserted():
    with pytest.raises(ValueError):
        episode_confidence([], expected_len=5)


def test_episode_confidence_raises_on_short_episode_when_length_is_asserted():
    with pytest.raises(ValueError):
        episode_confidence([0.9, 0.8], expected_len=5)


def test_episode_confidence_default_behavior_is_unchanged_without_expected_len():
    # Existing callers that never pass expected_len keep the old, documented
    # vacuous-1.0-for-empty behavior -- this guard is opt-in.
    assert episode_confidence([]) == 1.0
    assert episode_confidence([0.9, 0.8, 0.5]) == pytest.approx(0.9 * 0.8 * 0.5)


# --- real derived output: schema and join integrity -------------------------

_HAVE_SCORES = Path(_SELECTION_SCORES).exists()


@pytest.mark.skipif(not _HAVE_SCORES, reason="run code/scripts/03b_selection_scores.py first")
def test_selection_scores_have_the_exact_required_schema():
    df = pd.read_parquet(_SELECTION_SCORES)
    assert list(df.columns) == SELECTION_SCORE_COLUMNS
    assert len(df) > 0
    assert set(df["study"].unique()) <= {"StudyB", "StudyF", "StudyL", "StudyN"}


@pytest.mark.skipif(not _HAVE_SCORES, reason="run code/scripts/03b_selection_scores.py first")
def test_selection_scores_are_bounded_probabilities():
    df = pd.read_parquet(_SELECTION_SCORES)
    for col in ("score_top", "score_target"):
        finite = df[col].dropna()
        assert len(finite) > 0
        assert (finite >= 0).all() and (finite <= 1.0 + 1e-9).all()
    assert (df["n_stimuli"] >= 0).all()


@pytest.mark.skipif(not _HAVE_SCORES, reason="run code/scripts/03b_selection_scores.py first")
def test_selection_scores_fit_match_is_populated_and_categorical():
    df = pd.read_parquet(_SELECTION_SCORES)
    assert df["fit_match"].notna().all()
    assert set(df["fit_match"].unique()) <= {
        "own_session_own_condition",
        "own_session_other_condition",
        "earlier_session",
    }
    assert df["fit_session"].notna().all()
    assert df["fit_condition"].notna().all()


@pytest.mark.skipif(not _HAVE_SCORES, reason="run code/scripts/03b_selection_scores.py first")
def test_selection_scores_fit_session_is_never_later_than_its_own_session():
    # No-temporal-leakage invariant, now assertable directly from the data
    # (not only from resolve_fit_unit's own code path): a classifier must
    # never be fit on a session that comes after the one it is applied to.
    df = pd.read_parquet(_SELECTION_SCORES)

    def _n(session_id: str) -> int:
        return int(session_id.removeprefix("SE"))

    fit_n = df["fit_session"].map(_n)
    own_n = df["session_id"].map(_n)
    assert (fit_n <= own_n).all()


@pytest.mark.skipif(
    not (_HAVE_SCORES and Path(_ONLINE_TRIALS_CSV).exists()),
    reason="requires selection_scores.parquet and the sibling online_trials_all20.csv",
)
def test_selection_scores_join_against_online_trials_all20_matches_almost_every_row():
    scores = pd.read_parquet(_SELECTION_SCORES)
    online = pd.read_csv(_ONLINE_TRIALS_CSV)
    als_online = online[online["study"].isin(["StudyB", "StudyF", "StudyL", "StudyN"])]

    key_cols = ["relative_path", "trial_number"]
    assert not scores.duplicated(key_cols).any()  # 1:1, not 1:many

    merged = scores.merge(als_online[key_cols], on=key_cols, how="inner")
    match_rate = len(merged) / len(als_online)
    # Reported in task-3b-report.md; a low rate would mean the trial
    # reconstruction in this module disagrees with the sibling project's.
    assert match_rate > 0.95, f"join match rate too low: {match_rate:.4f}"
