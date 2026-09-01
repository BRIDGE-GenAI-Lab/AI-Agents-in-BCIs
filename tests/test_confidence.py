import numpy as np, pytest
from nag.confidence import fit_calibrator, reliability, episode_confidence


def test_calibrator_improves_ece_on_miscalibrated_input():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.7, 4000)
    raw = np.clip(y * 0.55 + 0.20 + rng.normal(0, 0.08, 4000), 1e-4, 1 - 1e-4)
    groups = rng.integers(0, 20, 4000)
    fit = fit_calibrator(raw, y, groups)
    assert reliability(fit.calibrator.transform(raw), y)["ece"] < reliability(raw, y)["ece"]

def test_reliability_outputs_are_bounded_and_named():
    p = np.linspace(0.01, 0.99, 500); y = (p > 0.5).astype(int)
    r = reliability(p, y)
    assert 0.0 <= r["ece"] <= 1.0 and 0.0 <= r["brier"] <= 1.0
    assert len(r["bins"]) == 10

def test_transform_never_returns_the_string_actual_posterior():
    import inspect, nag.confidence as m
    assert "actual posterior" not in inspect.getsource(m).lower()


def test_in_sample_and_oof_reliability_differ_with_participant_structure():
    # Each participant has an idiosyncratic score->outcome relationship (a
    # random per-participant shift), so a mapping fit on OTHER participants
    # does not transfer perfectly to a held-out one. An all-data fit that has
    # seen every participant's quirk should look strictly better (lower ECE)
    # than honest out-of-fold evaluation -- if it doesn't, the participant
    # separation isn't doing anything.
    rng = np.random.default_rng(1)
    n_participants, per_participant = 15, 300
    scores, correct, groups = [], [], []
    for g in range(n_participants):
        shift = rng.uniform(-0.25, 0.25)
        raw = rng.uniform(0, 1, per_participant)
        p_true = np.clip(raw + shift, 0.02, 0.98)
        scores.append(raw)
        correct.append(rng.binomial(1, p_true))
        groups.append(np.full(per_participant, g))
    scores = np.concatenate(scores)
    correct = np.concatenate(correct)
    groups = np.concatenate(groups)

    fit = fit_calibrator(scores, correct, groups)
    assert fit.oof is not None

    in_sample_ece = reliability(fit.calibrator.transform(scores), correct)["ece"]
    oof_ece = reliability(fit.oof, correct)["ece"]

    assert in_sample_ece <= oof_ece
    assert not np.isclose(in_sample_ece, oof_ece)


def test_oof_is_none_when_too_few_participant_groups():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, 50)
    correct = rng.binomial(1, scores)
    groups = np.zeros(50, dtype=int)  # a single participant: no valid fold split
    fit = fit_calibrator(scores, correct, groups)
    assert fit.oof is None
    # the deployable calibrator itself must still work
    assert fit.calibrator.transform(scores).shape == scores.shape


def test_episode_confidence_known_product():
    assert episode_confidence([0.9, 0.8, 0.5]) == pytest.approx(0.9 * 0.8 * 0.5)


def test_episode_confidence_monotone_decreasing_with_length():
    confs = [0.95, 0.9, 0.85, 0.8, 0.75]
    running = [episode_confidence(confs[:k]) for k in range(1, len(confs) + 1)]
    assert all(running[i + 1] < running[i] for i in range(len(running) - 1))


def test_episode_confidence_single_zero_drives_to_zero():
    assert episode_confidence([0.9, 0.0, 0.8]) == 0.0


def test_episode_confidence_empty_input():
    assert episode_confidence([]) == 1.0
