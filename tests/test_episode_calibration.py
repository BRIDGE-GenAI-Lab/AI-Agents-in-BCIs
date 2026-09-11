import numpy as np
from nag.episode_calibration import combine, episode_reliability

def test_product_rule_multiplies():
    assert abs(combine(np.array([[0.5, 0.5, 0.5]]), "product")[0] - 0.125) < 1e-12

def test_min_rule_takes_the_weakest_selection():
    assert combine(np.array([[0.9, 0.2, 0.8]]), "min")[0] == 0.2

def test_logsum_is_monotone_with_product():
    x = np.array([[0.9, 0.9, 0.9], [0.5, 0.5, 0.5]])
    p, l = combine(x, "product"), combine(x, "logsum")
    assert (np.argsort(p) == np.argsort(l)).all()

def test_unknown_rule_raises():
    try:
        combine(np.array([[0.5]]), "geometric")
    except ValueError as e:
        assert "geometric" in str(e)
    else:
        raise AssertionError("expected ValueError")

def test_perfect_separation_gives_auroc_one():
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    out = episode_reliability(conf, correct)
    assert out["auroc"] == 1.0

def test_reliability_reports_ece_brier_and_bins():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=400)
    correct = (rng.uniform(size=400) < conf).astype(int)
    out = episode_reliability(conf, correct, n_bins=10)
    assert 0.0 <= out["ece"] < 0.2
    assert 0.0 < out["brier"] < 0.5
    assert len(out["bins"]) == 10
