"""Task 7: the primary semantic fair-comparison table, and the recorded-proposal
sweep it rests on.

The load-bearing risk this file guards against is NOT that the script crashes.
It is that a proposal arm's raw `covered` column -- which means "the model or
resolver produced a candidate command", not "this episode was admitted at an
operating point" -- reaches the table as though it were an operating point.
Several models sit at 0.99-1.00 on that column precisely because nothing has
been gated by confidence yet, so a silent pass-through would overstate them
against arms whose rows ARE their final outcome.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "code" / "scripts" / "26_semantic_primary_table.py"
TABLE = REPO / "output" / "tables" / "semantic_primary_comparison.csv"
FAIR_RUNS = REPO / "output" / "intermediate" / "runs_semantic_fair"
RESOLVER_CURVES = REPO / "output" / "tables" / "semantic_fair_resolver_curves.csv"

FAIR_ARMS = [
    "fair:llm_vocab:advisory",
    "fair:llm_vocab:enforced",
    "fair:hybrid_semantic_gate",
    "fair:exact_resolver_gate",
    "fair:lexical_resolver_gate",
]


def _read_fair(stem: str) -> pd.DataFrame:
    return pd.read_parquet(FAIR_RUNS / f"{stem}.parquet")


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    """Run the real script once and read what it wrote.

    `sys.executable`, not a bare `python3`: the suite runs under the pinned
    `/private/tmp/nag_venv` interpreter and a PATH `python3` is a different
    environment that cannot even import pandas.
    """
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=str(REPO))
    return pd.read_csv(TABLE)


def test_semantic_primary_table_includes_every_new_fair_comparison_arm(table):
    """The whole point of Task 6-7 is that these five arm names appear in the
    final table -- a silent drop of any one of them (e.g. a merge key
    mismatch) would make the fair-comparison headline claim rest on fewer
    arms than the paper reports."""
    for arm in FAIR_ARMS:
        assert arm in table["arm_name"].values, f"{arm} missing from the primary table"


def test_every_paid_model_survives_into_the_table_for_every_llm_fair_arm(table):
    """Ten models x three paid arms were actually purchased. A groupby that
    dropped a model (a NaN key, a stale filename filter) would quietly shrink
    the panel the headline claim averages over."""
    for arm in ["fair:llm_vocab:advisory", "fair:llm_vocab:enforced", "fair:hybrid_semantic_gate"]:
        rows = table[table["arm_name"] == arm]
        assert len(rows) == 10, f"{arm} has {len(rows)} model rows, expected 10"
        assert rows["model"].nunique() == 10
        assert (rows["n_episodes"] == 200).all()


def test_recorded_proposal_gate_curve_reproduces_resolver_gate_curve_on_real_rows():
    """The decisive correctness check for the new sweep: fed the exact-resolver
    arm's OWN recorded rows, the recorded-proposal sweep must reproduce, value
    for value, the curve Task 2's already-reviewed `resolver_gate_curve`
    computed for that same arm by replaying the resolver. If the two disagree
    anywhere, the hybrid and enforced arms are being swept by a different rule
    than the resolver arms they are compared against."""
    from nag.naturalistic import recorded_proposal_gate_curve

    rows = _read_fair("fair_exact_resolver_gate")
    grid = np.linspace(0.0, 1.0, 101)
    mine = recorded_proposal_gate_curve(rows["covered"], rows["faithful"], rows["confidence"],
                                        threshold_grid=grid)

    published = pd.read_csv(RESOLVER_CURVES)
    theirs = published[published["arm_name"] == "fair:exact_resolver_gate"].reset_index(drop=True)

    assert len(mine) == len(theirs) == 101
    for col in ("threshold", "coverage", "risk", "n_covered"):
        np.testing.assert_allclose(mine[col].to_numpy(float), theirs[col].to_numpy(float),
                                   rtol=0, atol=1e-12, err_msg=f"{col} diverges")


def test_recorded_proposal_gate_curve_matches_riskcoverage_rc_curve_on_shared_thresholds():
    """A second, independent cross-check against `nag.riskcoverage.rc_curve`,
    which sweeps the same `covered & (confidence >= t)` rule on the data's own
    unique confidences. Swept on that grid the two must agree exactly; the new
    function differs from `rc_curve` only in taking the frozen manifest's
    101-point grid instead of the observed confidences."""
    from nag.naturalistic import recorded_proposal_gate_curve
    from nag.riskcoverage import rc_curve

    rows = _read_fair("anthropic_claude_sonnet_5__fair_hybrid_semantic_gate")
    conf = rows["confidence"].to_numpy(float)
    grid = np.unique(conf)
    mine = recorded_proposal_gate_curve(rows["covered"], rows["faithful"], conf, threshold_grid=grid)
    theirs = rc_curve(conf, rows["faithful"], rows["covered"], is_gate=True)

    merged = mine.merge(theirs, on="threshold", suffixes=("_mine", "_rc"))
    assert len(merged) == len(grid)
    np.testing.assert_allclose(merged["coverage_mine"], merged["coverage_rc"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(merged["risk_mine"], merged["risk_rc"], rtol=0, atol=1e-12)


def test_recorded_proposal_gate_curve_anchors_an_empty_threshold_at_zero_coverage():
    """A threshold above every confidence covers nothing. Risk is undefined
    there and must be written as the 0.0 integration anchor both
    `resolver_gate_curve` and `rc_curve` already use -- a NaN would poison
    `aurc`'s trapezoidal integration for the whole arm."""
    from nag.naturalistic import recorded_proposal_gate_curve

    curve = recorded_proposal_gate_curve(
        covered=[True, True, False], faithful=[True, False, False],
        confidence=[0.4, 0.6, 0.9], threshold_grid=[0.0, 0.5, 1.0])
    assert curve.loc[2, "coverage"] == 0.0
    assert curve.loc[2, "risk"] == 0.0
    assert curve.loc[2, "n_covered"] == 0
    # threshold 0.5 admits only the covered episode at 0.6, whose proposal was wrong
    assert curve.loc[1, "n_covered"] == 1
    assert curve.loc[1, "risk"] == 1.0


def test_recorded_proposal_gate_curve_never_admits_an_episode_with_no_proposal():
    """`covered=False` means no candidate command exists. No threshold, however
    low, may admit such an episode -- that would score an action the arm never
    proposed."""
    from nag.naturalistic import recorded_proposal_gate_curve

    curve = recorded_proposal_gate_curve(covered=[False, False], faithful=[False, False],
                                         confidence=[1.0, 1.0], threshold_grid=[0.0])
    assert curve.loc[0, "n_covered"] == 0


def test_hybrid_arm_swept_coverage_falls_below_its_raw_proposal_rate(table):
    """The specific failure this task exists to prevent. The hybrid arm's raw
    `covered` column is a PROPOSAL rate (0.915-1.000 across the panel) because
    it was run once at threshold=-inf. Once the confidence gate is actually
    applied, coverage must fall; if it did not, the sweep is not reaching the
    data."""
    from nag.naturalistic import recorded_proposal_gate_curve

    rows = _read_fair("google_gemini_3_7_flash__fair_hybrid_semantic_gate")
    raw = float(rows["covered"].mean())
    curve = recorded_proposal_gate_curve(rows["covered"], rows["faithful"], rows["confidence"],
                                         threshold_grid=np.linspace(0.0, 1.0, 101))
    gated = float(curve.loc[curve["threshold"] == 0.5, "coverage"].iloc[0])
    assert gated < raw, (
        f"gating at 0.5 left coverage at {gated}, unchanged from the raw proposal rate {raw} -- "
        "the sweep is not being applied"
    )

    # and the table must carry that whole curve, not just the raw rate
    row = table[(table["arm_name"] == "fair:hybrid_semantic_gate")
                & (table["model"] == "google/gemini-3.7-flash")].iloc[0]
    assert row["operating_point_kind"] == "swept_curve"
    assert np.isfinite(row["aurc"])


def test_proposal_arms_are_swept_and_advisory_arms_are_fixed_points(table):
    """Classification is the whole correctness question. An advisory arm's row
    IS its final outcome and must never be swept post hoc; every enforced arm
    on this benchmark recorded a proposal and must be."""
    swept = table[table["operating_point_kind"] == "swept_curve"]["arm_name"].unique()
    fixed = table[table["operating_point_kind"] == "fixed_point"]["arm_name"].unique()

    for arm in ["fair:llm_vocab:enforced", "fair:hybrid_semantic_gate",
                "fair:exact_resolver_gate", "fair:lexical_resolver_gate"]:
        assert arm in swept, f"{arm} records a proposal and must be threshold-swept"
    assert "fair:llm_vocab:advisory" in fixed
    assert set(swept).isdisjoint(fixed), "an arm cannot be both swept and a fixed point"


def test_only_swept_arms_carry_an_aurc(table):
    """A fixed point has no curve. An AURC on an advisory row would be a number
    describing operating points that arm was never run at."""
    swept = table[table["operating_point_kind"] == "swept_curve"]
    fixed = table[table["operating_point_kind"] == "fixed_point"]
    assert swept["aurc"].notna().all()
    assert fixed["aurc"].isna().all()


def test_reported_risk_matches_a_recomputation_from_the_raw_rows(table):
    """Guards the arithmetic itself: risk is the unfaithful fraction AMONG
    covered episodes, not among all episodes. The two differ by a factor of the
    coverage and the confusion is easy to ship."""
    rows = _read_fair("qwen_qwen3_8_max_0902__fair_llm_vocab_advisory")
    covered = rows["covered"].to_numpy(bool)
    expected_cov = covered.mean()
    expected_risk = 1.0 - rows["faithful"].to_numpy(bool)[covered].mean()

    row = table[(table["arm_name"] == "fair:llm_vocab:advisory")
                & (table["model"] == "qwen/qwen3.8-max-0902")].iloc[0]
    assert row["coverage"] == pytest.approx(expected_cov)
    assert row["risk"] == pytest.approx(expected_risk)
    assert row["n_covered"] == int(covered.sum())


def _load_script():
    """Import the table script as a module so its internal guards can be
    exercised directly."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_t26", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_equivalence_check_rejects_a_pass_through_of_the_raw_proposal_rate():
    """A guard that only ever runs against correct data passes on the failure it
    was written to catch. This runs the equivalence check against the specific
    bug it exists to prevent -- reporting the raw proposal rate at every
    threshold, i.e. not sweeping at all -- and requires it to disagree."""
    from nag.naturalistic import recorded_proposal_gate_curve

    rows = _read_fair("fair_exact_resolver_gate")
    grid = np.linspace(0.0, 1.0, 101)
    real = recorded_proposal_gate_curve(rows["covered"], rows["faithful"], rows["confidence"], grid)
    not_swept = pd.DataFrame(dict(threshold=grid, coverage=float(rows["covered"].mean()),
                                  risk=0.0, n_covered=int(rows["covered"].sum())))
    assert not np.allclose(real["coverage"], not_swept["coverage"])
    assert real["coverage"].iloc[-1] < not_swept["coverage"].iloc[-1]


def test_the_sweep_rejects_admitting_episodes_that_carry_no_proposal():
    """The other way to get this wrong: dropping the `& covered` term so a low
    threshold admits episodes on which nothing was ever proposed. That inflates
    coverage and scores actions the arm never took."""
    from nag.naturalistic import recorded_proposal_gate_curve

    rows = _read_fair("fair_exact_resolver_gate")
    conf = rows["confidence"].to_numpy(float)
    real = recorded_proposal_gate_curve(rows["covered"], rows["faithful"], conf, [0.0])
    ignoring_proposals = float((conf >= 0.0).mean())
    assert real.loc[0, "coverage"] < ignoring_proposals


def test_the_script_refuses_a_curve_that_disagrees_with_the_rows_it_summarizes():
    """The published resolver curves are read from a CSV rather than recomputed,
    so a stale or edited CSV would silently misreport those two arms. The guard
    must fire on a curve whose threshold-0 admission count does not match the
    parquet it claims to describe.

    Sabotage is scoped to `fair:exact_resolver_gate` because the comparator arm
    now hits an EARLIER guard on the same break -- the comparator's episode set
    is checked against its published curve before the per-arm loop begins (see
    `test_the_script_refuses_a_comparator_episode_set_that_disagrees_with_its_published_curve`).
    Both arms reach this check by the identical code path, so scoping it here
    keeps this guard genuinely exercised rather than shadowed by the other."""
    mod = _load_script()
    original = mod.published_resolver_curve

    def sabotaged(arm_name):
        curve = original(arm_name).copy()
        if arm_name == "fair:exact_resolver_gate":
            curve.loc[0, "n_covered"] = int(curve.loc[0, "n_covered"]) + 7
        return curve

    mod.published_resolver_curve = sabotaged
    with pytest.raises(ValueError, match="describe different episode sets"):
        mod.build_rows()
    mod.published_resolver_curve = original
    assert mod.build_rows() is not None  # and the unsabotaged build still works


def test_risk_is_measured_among_covered_episodes_not_all_episodes():
    """The denominator confusion, run against the real exact-resolver arm where
    the two differ enormously (0.34 against 0.00). A test that only checked the
    correct denominator on an arm with full coverage would pass either way."""
    rows = _read_fair("fair_exact_resolver_gate")
    covered = rows["covered"].to_numpy(bool)
    faithful = rows["faithful"].to_numpy(bool)
    among_all = 1.0 - faithful.mean()
    among_covered = 1.0 - faithful[covered].mean()
    assert among_all == pytest.approx(0.34)
    assert among_covered == pytest.approx(0.0)


def test_the_same_arm_name_on_two_benchmarks_stays_distinguishable(table):
    """`factorial:decoder_confidence:advisory:s0` appears twice per model: once
    on the naturalistic 200 and once on the recalibrated hashed-codebook run.
    A reader keying on (arm, model) alone would silently average two different
    benchmarks, so the composite key must be unique and the disambiguating
    columns must be present."""
    key = ["arm_name", "model", "source_run"]
    assert not table.duplicated(key).any()
    dupes = table.duplicated(["arm_name", "model"], keep=False)
    assert dupes.any(), "expected at least one arm measured on both benchmarks"
    assert table.loc[dupes, "episode_set"].nunique() == 2


def test_an_arm_above_the_comparators_ceiling_reports_no_matched_verdict(table):
    """The comparator resolves only 94% of episodes. An arm operating above
    that has no matched point to be compared at, and the table must say so in
    words -- a bare NaN in `beats_comparator` would read as 'not evaluated'
    and invite a reader to drop exactly the rows carrying the finding."""
    above = table[table["coverage"] > table["comparator_max_coverage"] + 1e-12]
    assert len(above) > 0
    assert above["beats_comparator"].isna().all()
    assert above["frontier_verdict"].str.startswith(
        ("exceeds the comparator's frontier", "higher coverage than the comparator")).all()

    below = table[(table["episode_set"] == "naturalistic_200")
                  & (table["coverage"] <= table["comparator_max_coverage"] + 1e-12)
                  & (table["arm_name"] != "fair:lexical_resolver_gate")]
    assert below["comparator_risk_at_own_coverage"].notna().all()
    assert below["beats_comparator"].notna().all()


def test_a_zero_risk_arm_below_the_comparators_ceiling_is_dominated_not_tied(table):
    """The degenerate-label bug this verdict exists to prevent. The comparator's
    risk is 0.0 at every one of its 101 thresholds, so a verdict built on
    matched-coverage risk alone calls EVERY zero-risk arm a 'tie' at ANY
    coverage -- and 'most arms tied the deterministic gate' would then be
    literally true off this column and substantively false, because an arm
    covering 0.62 at zero risk is beaten outright by a comparator covering
    0.94 at zero risk."""
    low = table[(table["episode_set"] == "naturalistic_200")
                & (table["risk"] == 0.0)
                & (table["coverage"] < table["comparator_max_coverage"] - 1e-12)]
    assert len(low) > 0
    assert low["frontier_verdict"].eq(
        "dominated by the comparator (it reaches higher coverage at no more risk)").all()
    assert "ties" not in " ".join(table["frontier_verdict"].unique())

    # only a point the comparator cannot improve on either axis may be "matched"
    matched = table[table["frontier_verdict"].str.startswith("matches the comparator")]
    assert len(matched) > 0
    np.testing.assert_allclose(matched["coverage"], matched["comparator_max_coverage"],
                               rtol=0, atol=1e-12)
    assert matched["risk"].eq(0.0).all()


def test_extra_coverage_bought_with_extra_risk_is_labelled_apart(table):
    """Above the comparator's ceiling there is no matched point, but the two
    ways of getting there are not the same result: zero-risk coverage the
    comparator cannot reach is a win, and coverage bought with a 5.6%
    unfaithful rate is not. One shared label would let the second be read as
    the first."""
    above = table[table["coverage"] > table["comparator_max_coverage"] + 1e-12]
    wins = above[above["frontier_verdict"].str.startswith("exceeds")]
    costly = above[above["frontier_verdict"].str.startswith("higher coverage than")]
    assert len(wins) > 0 and len(costly) > 0
    assert wins["risk"].eq(0.0).all()
    assert (costly["risk"] > 0.0).all()


def test_every_observed_zero_risk_carries_its_upper_confidence_bound(table):
    """A risk of 0.0 on at most 200 episodes is an OBSERVED zero. Without this
    column beside it, the CSV alone reads as a claim that an arm is never
    unfaithful."""
    zeros = table[(table["risk"] == 0.0) & (table["n_covered"] > 0)]
    assert len(zeros) > 0
    assert (zeros["risk_upper95_one_sided"] > 0.0).all()
    # at n_unfaithful == 0 the Clopper-Pearson bound is exactly the rule of three
    expected = 1.0 - 0.05 ** (1.0 / zeros["n_covered"].to_numpy(float))
    np.testing.assert_allclose(zeros["risk_upper95_one_sided"], expected, rtol=0, atol=1e-12)
    assert table["risk_upper95_one_sided"].ge(table["risk"]).all()


def test_the_aurc_difference_is_flagged_as_unusable_for_ranking(table):
    """Both resolvers have exactly 0.0 risk everywhere, so the comparator's area
    is 0.0 over every window and `aurc_common_minus_comparator` equals
    `aurc_common` identically -- it can never go negative. It looks like a
    ranking statistic and is not one, so the table has to say so before
    somebody sorts a figure by it."""
    swept = table[table["operating_point_kind"] == "swept_curve"]
    assert swept["aurc_ranking_valid"].eq(False).all()
    assert swept["comparator_aurc_common"].eq(0.0).all()
    np.testing.assert_allclose(swept["aurc_common_minus_comparator"], swept["aurc_common"],
                               rtol=0, atol=1e-12)
    assert table[table["operating_point_kind"] == "fixed_point"]["aurc_ranking_valid"].isna().all()


def test_the_table_records_that_the_arms_do_not_share_a_model_panel(table):
    """The fair arms ran 10 models and the Task 20 controls ran 5, so a bare
    `groupby("arm_name").coverage.mean()` contrasts a 10-model average against
    a 5-model one. The column that makes that visible has to be in the CSV."""
    sizes = table.groupby("arm_name")["model_panel_size"].max()
    assert sizes["fair:llm_vocab:enforced"] == 10
    assert sizes["factorial:decoder_confidence:enforced:s0"] == 5
    for arm, size in sizes.items():
        assert (table["arm_name"] == arm).sum() >= size


def test_risk_at_the_comparators_ceiling_is_never_extrapolated(table):
    """`np.interp` clamps rather than refusing, so an arm whose frontier stops
    below the comparator's ceiling would otherwise be handed the risk at its
    own maximum and reported as if it operated at 0.94."""
    swept = table[table["operating_point_kind"] == "swept_curve"]
    short = swept[swept["coverage"] < swept["comparator_max_coverage"] - 1e-12]
    assert len(short) > 0
    assert short["risk_at_comparator_max_coverage"].isna().all()


def test_common_support_window_is_pairwise_with_the_comparator(table):
    """Panel-wide common support would shrink every arm's reported area to the
    exact resolver's 0.66 ceiling. The window must be this arm against the
    comparator, so the two areas in `aurc_common_minus_comparator` describe the
    same interval and nothing else's limits leak in."""
    swept = table[table["operating_point_kind"] == "swept_curve"]
    expected_hi = np.minimum(swept["coverage"], swept["comparator_max_coverage"])
    np.testing.assert_allclose(swept["support_hi"], expected_hi, rtol=0, atol=1e-12)
    assert swept["support_hi"].nunique() > 1, "a single window for every arm means it is not pairwise"
    np.testing.assert_allclose(
        swept["aurc_common_minus_comparator"],
        swept["aurc_common"] - swept["comparator_aurc_common"], rtol=0, atol=1e-12)


def test_the_original_no_vocabulary_arms_are_kept_as_the_control_condition(table):
    """Route B: the original hashed and no-vocabulary experiments stay in the
    table as the control condition rather than being replaced by the fair
    arms. Losing them would leave the fair-comparison result with nothing to
    be fair RELATIVE TO."""
    assert "factorial:decoder_confidence:enforced:s0" in table["arm_name"].values
    assert "natural_confidence_gate_lexical" in table["arm_name"].values
    assert set(table["episode_set"]) >= {"naturalistic_200", "main_study_codebook"}
    saw = table.set_index("arm_name")["model_saw_vocabulary"]
    assert saw.loc["fair:llm_vocab:enforced"].eq("yes").all()
    assert saw.loc["factorial:decoder_confidence:enforced:s0"].eq("no").all()


def _all_fair_rows() -> pd.DataFrame:
    """Every semantic-fair checkpoint concatenated, so a test can regroup the
    raw episodes itself instead of trusting the script's own grouping."""
    return pd.concat([pd.read_parquet(p) for p in sorted(FAIR_RUNS.glob("*.parquet"))
                      if not p.name.startswith("._")], ignore_index=True)


def _comparator_covered_episode_ids() -> set:
    """The 188 episodes `fair:lexical_resolver_gate` acts on at its own
    maximum-coverage operating point, read straight from its parquet."""
    rows = _read_fair("fair_lexical_resolver_gate")
    return set(rows.loc[rows["covered"].to_numpy(bool), "episode_id"])


def test_the_risk_bound_for_the_incremental_claim_is_computed_on_the_incremental_episodes(table):
    """The positive finding is about the episodes an arm reaches that the
    comparator cannot -- 8-12 of them, not the ~200 the arm covers in total.
    `risk_upper95_one_sided` bounds the whole arm at about 0.015, so quoting it
    beside the incremental claim understates that claim's uncertainty by more
    than an order of magnitude. The bound that belongs to it is computed on the
    increment alone, and this test recomputes the increment from the raw
    parquets rather than reading the script's own arithmetic back."""
    comparator_ids = _comparator_covered_episode_ids()
    raw = _all_fair_rows()
    above = table[table["frontier_verdict"].str.startswith("exceeds the comparator's frontier")]
    assert len(above) == 10

    for _, row in above.iterrows():
        sub = raw[(raw["cell"] == row["arm_name"]) & (raw["model"] == row["model"])]
        covered = sub["covered"].to_numpy(bool)
        beyond = set(sub.loc[covered, "episode_id"]) - comparator_ids
        in_beyond = sub["episode_id"].isin(beyond).to_numpy(bool)
        n_unfaithful = int((in_beyond & ~sub["faithful"].to_numpy(bool)).sum())

        assert row["n_beyond_comparator"] == len(beyond)
        assert row["n_unfaithful_beyond_comparator"] == n_unfaithful == 0
        assert 8 <= len(beyond) <= 12
        # at zero observed failures Clopper-Pearson is exactly the rule of three
        assert row["risk_upper95_beyond_comparator"] == pytest.approx(
            1.0 - 0.05 ** (1.0 / len(beyond)), abs=1e-12)
        assert 0.22 <= row["risk_upper95_beyond_comparator"] <= 0.32
        ratio = row["risk_upper95_beyond_comparator"] / row["risk_upper95_one_sided"]
        assert 14.0 <= ratio <= 21.0, (
            f"{row['arm_name']}/{row['model']}: the whole-arm bound is only {ratio:.1f}x tighter "
            "than the incremental one -- one of the two is not being computed on the set it names"
        )


def test_covering_more_episodes_than_the_comparator_is_not_covering_everything_it_covers(table):
    """"Higher coverage at no more risk" is a count, and a count hides a trade:
    an arm can act on more episodes than the comparator while abstaining on
    episodes the comparator resolves. Five of the ten frontier-exceeding cells
    do exactly that, so "does everything the resolver does, and more" is false
    for half of them and the table has to carry the distinction rather than
    leave it to prose."""
    comparator_ids = _comparator_covered_episode_ids()
    raw = _all_fair_rows()
    above = table[table["frontier_verdict"].str.startswith("exceeds the comparator's frontier")]
    supersets = above[above["is_strict_superset_of_comparator_coverage"]]
    traders = above[~above["is_strict_superset_of_comparator_coverage"]]

    assert len(supersets) == 5 and len(traders) == 5
    assert supersets["n_comparator_episodes_missed"].eq(0).all()
    assert supersets["n_beyond_comparator"].gt(0).all()
    assert traders["n_comparator_episodes_missed"].gt(0).all()

    for model, expected_missed in [("qwen/qwen3.8-max-0902", 6), ("moonshotai/kimi-k3", 2),
                                   ("deepseek/deepseek-v4-flash", 1), ("x-ai/grok-4.6", 1)]:
        sub = raw[(raw["cell"] == "fair:llm_vocab:enforced") & (raw["model"] == model)]
        acted = set(sub.loc[sub["covered"].to_numpy(bool), "episode_id"])
        assert len(comparator_ids - acted) == expected_missed
        row = table[(table["arm_name"] == "fair:llm_vocab:enforced")
                    & (table["model"] == model)].iloc[0]
        assert row["n_comparator_episodes_missed"] == expected_missed
        assert not row["is_strict_superset_of_comparator_coverage"]


def test_an_arm_that_only_reproduces_the_comparators_coverage_is_not_flagged_as_extending_it(table):
    """`natural_confidence_gate_lexical` IS the comparator's algorithm as Task 20
    ran it: identical 188 episodes, nothing added. The flag must be False there,
    or "extends the comparator" would collect a row that extends nothing --
    which is the same overclaim as the three-of-ten superset error, arriving
    from the opposite direction."""
    row = table[table["arm_name"] == "natural_confidence_gate_lexical"].iloc[0]
    assert row["n_comparator_episodes_missed"] == 0
    assert row["n_beyond_comparator"] == 0
    assert not row["is_strict_superset_of_comparator_coverage"]


def test_the_episode_sets_reconcile_with_the_coverage_counts_on_every_shared_benchmark_row(table):
    """The identity that ties the new set arithmetic to the old rate arithmetic:
    an arm's covered episodes are exactly the comparator's minus the ones it
    misses, plus the ones it reaches beyond. If any row breaks it, the episode
    sets and the coverage column are describing different runs."""
    shared = table[table["episode_set"] == "naturalistic_200"]
    n_comparator = len(_comparator_covered_episode_ids())
    assert n_comparator == 188
    reconstructed = (shared["n_beyond_comparator"]
                     + n_comparator - shared["n_comparator_episodes_missed"])
    np.testing.assert_array_equal(reconstructed.to_numpy(), shared["n_covered"].to_numpy())

    other = table[table["episode_set"] != "naturalistic_200"]
    assert other["n_beyond_comparator"].isna().all()
    assert other["risk_upper95_beyond_comparator"].isna().all()
    assert other["is_strict_superset_of_comparator_coverage"].eq(False).all()


def test_the_script_refuses_an_episode_reconstruction_that_disagrees_with_its_coverage_rate():
    """The set arithmetic is only trustworthy if the episodes it admits are the
    same episodes the coverage rate counts. Run against the break it exists to
    catch -- one episode silently dropped from every non-comparator arm's
    admitted set -- the guard must fire."""
    mod = _load_script()
    original = mod.admitted_episode_ids

    def sabotaged(rows, threshold):
        ids = original(rows, threshold)
        arm = rows["cell"].iloc[0] if "cell" in rows.columns else ""
        if arm == "fair:lexical_resolver_gate" or not ids:
            return ids
        return set(sorted(ids)[:-1])

    mod.admitted_episode_ids = sabotaged
    with pytest.raises(ValueError, match="disagrees with the rate"):
        mod.build_rows()
    mod.admitted_episode_ids = original
    assert mod.build_rows() is not None


def test_the_script_refuses_a_comparator_episode_set_that_disagrees_with_its_published_curve():
    """Every set difference in this table is taken against the comparator's
    admitted episodes, so a wrong comparator set would corrupt all of them at
    once and none of the per-row checks would notice. The comparator's own set
    is therefore checked against the count its published curve reports, and
    that check is run here against a deliberately shortened set."""
    mod = _load_script()
    original = mod.admitted_episode_ids

    def sabotaged(rows, threshold):
        ids = original(rows, threshold)
        arm = rows["cell"].iloc[0] if "cell" in rows.columns else ""
        return set(sorted(ids)[:-1]) if arm == "fair:lexical_resolver_gate" else ids

    mod.admitted_episode_ids = sabotaged
    with pytest.raises(ValueError, match="episode-level reconstruction and"):
        mod.build_rows()
    mod.admitted_episode_ids = original
    assert mod.build_rows() is not None
