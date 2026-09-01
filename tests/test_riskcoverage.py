import numpy as np
import pandas as pd
import pytest

from nag.design import EARLIER_SESSION, enumerate_cells
from nag.riskcoverage import (
    oracle_confidence,
    aurc,
    coverage_sweep_curve,
    curve_knob,
    dominates,
    primary_and_sensitivity_curves,
    rc_curve,
    rc_curves_by_stratum,
    risk_at_coverage,
)

# Real cells from the actual registry, not hand-built stand-ins -- so these
# tests are grounded in what nag.design.enumerate_cells() actually produces.
_CELLS = {c.name: c for c in enumerate_cells()}
GATE_CELL = _CELLS["nonllm_gate"]                                     # knob=confidence
DECODER_ENFORCED_CELL = _CELLS["factorial:decoder_confidence:enforced:s0"]  # knob=confidence (headline arm)
SELF_ENFORCED_CELL = _CELLS["factorial:self_confidence:enforced:s0"]  # knob=confidence
ORACLE_CELL = _CELLS["oracle"]                                        # knob=confidence
RANDOM_GATE_CELL = _CELLS["random_gate"]                              # knob=coverage
NONE_ENFORCED_CELL = _CELLS["factorial:none:enforced:s0"]             # knob=coverage
ADVISORY_CELL = _CELLS["factorial:none:advisory:s0"]                  # knob=None
CAUTION_CELL = _CELLS["caution:w0"]                                   # knob=None
SINGLESHOT_CELL = _CELLS["singleshot"]                                # knob=None


# --- core curve behavior -----------------------------------------------

def test_perfect_confidence_gives_monotone_curve_and_low_aurc():
    conf = np.linspace(0, 1, 200)
    faithful = conf > 0.5  # confidence perfectly predicts faithfulness
    c = rc_curve(conf, faithful, np.ones_like(conf, dtype=bool), is_gate=True)
    assert c["coverage"].is_monotonic_increasing
    assert aurc(c) < 0.25


def test_always_abstain_is_not_rewarded():
    """A system that never acts has zero coverage and cannot win on risk alone."""
    conf = np.linspace(0, 1, 100)
    c = rc_curve(conf, np.zeros(100, dtype=bool), np.zeros(100, dtype=bool), is_gate=True)
    assert c["coverage"].max() == 0.0
    # the coverage==0 anchor is not a "safe" win -- an abstain point sitting
    # exactly on it does not dominate (dominance requires strictly less risk)
    assert dominates(0.0, 0.0, c) is False


def test_risk_at_coverage_interpolates_at_the_matched_point():
    c = pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "risk": [0.0, 0.1, 0.4],
                      "threshold": [1.0, 0.5, 0.0]})
    assert abs(risk_at_coverage(c, 0.75) - 0.25) < 1e-9


def test_dominance_flags_a_point_above_and_below_the_frontier():
    c = pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "risk": [0.0, 0.1, 0.4],
                      "threshold": [1.0, 0.5, 0.0]})
    assert dominates(0.5, 0.30, c) is False    # prompt arm worse than the gate at same coverage
    assert dominates(0.5, 0.05, c) is True     # prompt arm better than the gate at same coverage


# --- Correction 8.1 / Ruling 27: rc_curve must refuse non-confidence-knob arms

def test_rc_curve_raises_for_an_advisory_cell():
    conf = np.linspace(0, 1, 50)
    faithful = conf > 0.5
    covered = np.ones(50, dtype=bool)
    with pytest.raises(ValueError):
        rc_curve(conf, faithful, covered, cell=ADVISORY_CELL)


def test_rc_curve_raises_for_random_gate_despite_enforced_label():
    """control_mechanism == 'enforced' is not sufficient: random_gate never
    consults confidence at all (nag.controllers.random_gate takes a
    coverage probability, not conf), so its knob is "coverage", not
    "confidence" -- it belongs in coverage_sweep_curve, not rc_curve."""
    conf = np.linspace(0, 1, 50)
    faithful = conf > 0.5
    covered = np.ones(50, dtype=bool)
    with pytest.raises(ValueError):
        rc_curve(conf, faithful, covered, cell=RANDOM_GATE_CELL)


def test_rc_curve_raises_for_none_enforced_despite_enforced_label():
    """Ruling 27's key clarification: 'enforced' with uncertainty_source ==
    'none' has nothing to threshold on -- it is definitionally the random
    gate, a coverage-knob arm, not a confidence-knob one."""
    conf = np.linspace(0, 1, 50)
    faithful = conf > 0.5
    covered = np.ones(50, dtype=bool)
    with pytest.raises(ValueError):
        rc_curve(conf, faithful, covered, cell=NONE_ENFORCED_CELL)


def test_rc_curve_raises_when_is_gate_explicitly_false():
    conf = np.linspace(0, 1, 50)
    faithful = conf > 0.5
    covered = np.ones(50, dtype=bool)
    with pytest.raises(ValueError):
        rc_curve(conf, faithful, covered, is_gate=False)


def test_rc_curve_raises_when_neither_cell_nor_is_gate_given():
    conf = np.linspace(0, 1, 50)
    faithful = conf > 0.5
    covered = np.ones(50, dtype=bool)
    with pytest.raises(ValueError):
        rc_curve(conf, faithful, covered)


@pytest.mark.parametrize("cell", [GATE_CELL, DECODER_ENFORCED_CELL, SELF_ENFORCED_CELL])
def test_rc_curve_accepts_every_confidence_knob_cell(cell):
    conf = np.linspace(0, 1, 50)
    faithful = conf > 0.5
    covered = np.ones(50, dtype=bool)
    c = rc_curve(conf, faithful, covered, cell=cell)
    assert c["coverage"].is_monotonic_increasing


def test_oracle_arm_accepts_only_oracle_confidence():
    """The error-indicator arm is defined by its sweep variable and nothing else.

    Its prompt is byte-identical to five other cells, so sweeping it on the
    reconstructed decoder confidence silently turns it into a copy of
    factorial:decoder_confidence:enforced. That is what had happened, and the
    resulting curve was perfectly well formed, so nothing downstream could
    notice. This test previously asserted the WRONG behaviour: it required
    rc_curve to accept the oracle arm on an arbitrary confidence vector.
    """
    err = np.array([False, True] * 25)
    faithful = ~err
    covered = np.ones(50, dtype=bool)

    c = rc_curve(oracle_confidence(err), faithful, covered, cell=ORACLE_CELL)
    assert c["coverage"].is_monotonic_increasing

    with pytest.raises(ValueError, match="oracle"):
        rc_curve(np.linspace(0, 1, 50), faithful, covered, cell=ORACLE_CELL)


def test_oracle_confidence_is_perfect_knowledge_of_correctness():
    err = np.array([False, True, False])
    assert list(oracle_confidence(err)) == [1.0, 0.0, 1.0]


def test_decoder_confidence_enforced_is_curve_eligible():
    """The headline arm: factorial:decoder_confidence:enforced:* was wrongly
    refused by the pre-Ruling-27 name allowlist. It must not raise now."""
    conf = np.linspace(0, 1, 50)
    faithful = conf > 0.5
    covered = np.ones(50, dtype=bool)
    rc_curve(conf, faithful, covered, cell=DECODER_ENFORCED_CELL)  # must not raise


# --- Ruling 27: curve_knob classifies all three knob groups correctly --

@pytest.mark.parametrize("cell", [GATE_CELL, DECODER_ENFORCED_CELL, SELF_ENFORCED_CELL, ORACLE_CELL])
def test_curve_knob_is_confidence_for_the_confidence_group(cell):
    assert curve_knob(cell) == "confidence"


@pytest.mark.parametrize("cell", [RANDOM_GATE_CELL, NONE_ENFORCED_CELL])
def test_curve_knob_is_coverage_for_the_coverage_group(cell):
    assert curve_knob(cell) == "coverage"


@pytest.mark.parametrize("cell", [ADVISORY_CELL, CAUTION_CELL, SINGLESHOT_CELL])
def test_curve_knob_is_none_for_the_no_knob_group(cell):
    assert curve_knob(cell) is None


def test_curve_knob_classifies_every_registered_cell_into_a_known_tier():
    for cell in enumerate_cells():
        assert curve_knob(cell) in ("confidence", "coverage", None)


# --- coverage_sweep_curve: the coverage-knob path -----------------------

def test_coverage_sweep_curve_accepts_a_coverage_knob_cell_and_is_flat():
    rng = np.random.default_rng(2)
    wbf = rng.random(500) > 0.35
    c = coverage_sweep_curve(wbf, cell=RANDOM_GATE_CELL)
    assert c["risk"].nunique() == 1  # flat: coverage buys nothing for this knob
    assert abs(c["risk"].iloc[0] - float((~wbf).mean())) < 1e-9
    assert c["coverage"].is_monotonic_increasing


def test_coverage_sweep_curve_raises_for_a_confidence_knob_cell():
    wbf = np.random.default_rng(3).random(100) > 0.3
    with pytest.raises(ValueError):
        coverage_sweep_curve(wbf, cell=GATE_CELL)


def test_coverage_sweep_curve_raises_for_an_advisory_cell():
    wbf = np.random.default_rng(4).random(100) > 0.3
    with pytest.raises(ValueError):
        coverage_sweep_curve(wbf, cell=ADVISORY_CELL)


def test_coverage_sweep_curve_raises_when_neither_cell_nor_flag_given():
    wbf = np.random.default_rng(5).random(100) > 0.3
    with pytest.raises(ValueError):
        coverage_sweep_curve(wbf)


# --- Correction 8.2: undefined risk at zero coverage / min_coverage floor

def test_aurc_min_coverage_floor_excludes_the_near_zero_region():
    # a curve with an artificially noisy near-zero-coverage spike
    c = pd.DataFrame({
        "coverage": [0.0, 0.01, 0.5, 1.0],
        "risk": [0.0, 0.9, 0.1, 0.4],
        "threshold": [np.inf, 0.99, 0.5, 0.0],
    })
    full = aurc(c, min_coverage=0.0)
    floored = aurc(c, min_coverage=0.05)
    assert floored < full  # the noisy spike no longer contributes


def test_aurc_min_coverage_interpolates_the_floor_point():
    c = pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "risk": [0.0, 0.2, 0.4],
                      "threshold": [1.0, 0.5, 0.0]})
    # risk is linear here, so area from 0.25 to 1.0 is a trapezoid we can
    # compute by hand: interpolated risk at 0.25 is 0.1
    expected = np.trapezoid([0.1, 0.2, 0.4], [0.25, 0.5, 1.0])
    assert abs(aurc(c, min_coverage=0.25) - expected) < 1e-9


# --- Correction 8.4: stratify by fit_match, earlier_session excluded ----

def test_rc_curves_by_stratum_produces_one_curve_per_level():
    rng = np.random.default_rng(0)
    n = 300
    conf = rng.uniform(0, 1, n)
    faithful = rng.random(n) > 0.3
    covered = np.ones(n, dtype=bool)
    fit_match = rng.choice(
        ["own_session_own_condition", "own_session_other_condition", EARLIER_SESSION], size=n
    )
    curves = rc_curves_by_stratum(conf, faithful, covered, fit_match, is_gate=True)
    assert set(curves.keys()) == set(np.unique(fit_match))
    for level, curve in curves.items():
        assert curve["coverage"].is_monotonic_increasing


def test_primary_curve_excludes_earlier_session_but_sensitivity_includes_it():
    rng = np.random.default_rng(1)
    n_primary, n_excluded = 200, 40
    conf = np.concatenate([rng.uniform(0, 1, n_primary), rng.uniform(0, 1, n_excluded)])
    # anti-predictive within earlier_session: higher confidence, LOWER accuracy
    faithful_primary = rng.random(n_primary) > 0.3
    faithful_excluded = conf[n_primary:] < 0.5
    faithful = np.concatenate([faithful_primary, faithful_excluded])
    covered = np.ones(n_primary + n_excluded, dtype=bool)
    fit_match = np.array(["own_session_own_condition"] * n_primary + [EARLIER_SESSION] * n_excluded)

    result = primary_and_sensitivity_curves(conf, faithful, covered, fit_match, is_gate=True)
    assert set(result.keys()) == {"primary", "sensitivity_all", "by_stratum"}

    # the primary curve's finest-grained coverage step must correspond to
    # 1/n_primary, not 1/(n_primary + n_excluded) -- proof earlier_session
    # rows never entered the primary sweep's denominator
    primary_steps = np.diff(sorted(result["primary"]["coverage"].unique()))
    assert abs(min(s for s in primary_steps if s > 0) - 1 / n_primary) < 1e-9

    assert EARLIER_SESSION in result["by_stratum"]
    assert result["sensitivity_all"]["coverage"].max() == 1.0


# --- Common-support AURC ------------------------------------------------

from nag.riskcoverage import common_support, aurc_common


def _curve(covs, risks):
    return pd.DataFrame({"coverage": covs, "risk": risks})


def test_common_support_is_the_intersection_of_coverage_ranges():
    curves = {"gate": _curve([0.0, 0.5, 1.0], [0.0, 0.5, 0.8]),
              "arm":  _curve([0.0, 0.5, 0.97], [0.0, 0.5, 0.78])}
    lo, hi = common_support(curves)
    assert lo == 0.0 and hi == 0.97


def test_truncated_curve_no_longer_wins_on_area_alone():
    """The bug this fixes: an arm whose frontier stops at 0.973 coverage
    integrates over a shorter, lower-risk range and looks better than a gate
    that runs to 1.0. On common support the artefact disappears."""
    gate = _curve([0.1, 0.5, 1.0], [0.5, 0.6, 0.9])
    arm  = _curve([0.1, 0.5, 0.9], [0.5, 0.6, 0.85])
    lo, hi = common_support({"gate": gate, "arm": arm})
    assert hi == 0.9
    assert abs(aurc_common(gate, lo, hi) - aurc_common(arm, lo, hi)) < 0.02


def test_aurc_common_rejects_a_curve_that_does_not_reach_the_window():
    short = _curve([0.1, 0.4], [0.5, 0.6])
    try:
        aurc_common(short, 0.1, 0.9)
    except ValueError as e:
        assert "does not span" in str(e)
    else:
        raise AssertionError("expected ValueError")
