import pandas as pd
from nag.episodes import build_episodes


def _row(pid, study, trial, target, selected, t):
    return dict(study_participant_id=pid, study=study, participant_id=pid,
                session_id="SE001", condition="CB", relative_path=f"{pid}.edf",
                trial_number=trial, target=target, selected=selected,
                correct=(target == selected), eligible=True, phase3_time_seconds=t)


def test_episode_preserves_real_errors_and_does_not_cross_files():
    rows = [_row("StudyB:B_01", "StudyB", i, t, s, i * 30.0)
            for i, (t, s) in enumerate([(1, 1), (2, 9), (3, 3), (4, 4), (5, 5)], start=1)]
    rows += [_row("StudyB:B_02", "StudyB", i, t, t, i * 30.0)
             for i, t in enumerate([1, 2, 3, 4, 5], start=1)]
    rows[5]["relative_path"] = "B_02.edf"
    for r in rows[5:]:
        r["relative_path"] = "B_02.edf"
    df = pd.DataFrame(rows)

    eps = build_episodes(df, length=5, als_only=True)
    assert len(eps) == 2
    e0 = eps[eps.participant_id == "StudyB:B_01"].iloc[0]
    assert e0.n_selections == 5
    assert e0.n_errors == 1                      # the real 2->9 error is preserved
    assert e0.true_string != e0.decoded_string
    assert set(eps.stratum) == {"ALS"}

def test_able_bodied_excluded_when_als_only():
    rows = [_row("StudyA:A_01", "StudyA", i, i, i, i * 10.0) for i in range(1, 6)]
    assert len(build_episodes(pd.DataFrame(rows), length=5, als_only=True)) == 0


def test_out_of_range_code_drops_episode_and_is_counted(caplog):
    # codes run 1..36 over the 6x6 grid; 37 is out of range and must never
    # become "?" -- the whole episode is dropped instead.
    rows = [_row("StudyB:B_03", "StudyB", i, t, s, i * 30.0)
            for i, (t, s) in enumerate([(1, 1), (2, 2), (3, 3), (4, 4), (5, 37)], start=1)]
    df = pd.DataFrame(rows)

    with caplog.at_level("WARNING", logger="nag.episodes"):
        eps = build_episodes(df, length=5, als_only=True)

    assert len(eps) == 0
    assert not any("?" in s for s in eps["true_string"].tolist() + eps["decoded_string"].tolist())
    assert any("dropped 1 episode" in rec.message for rec in caplog.records)
