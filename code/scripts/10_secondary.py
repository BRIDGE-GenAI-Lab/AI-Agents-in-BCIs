"""SECONDARY ANALYSES: error-conditional outcomes, calibration, the
caution-wording battery, the scaffold nuisance factor, consequence-tier
stratification and its transition matrix, the reference arms, and a
parse-failure sensitivity sweep.

Each answers a question the primary end-to-end analysis (09_analysis.py)
cannot:

  ERROR-CONDITIONAL is a SECONDARY population here, not the primary one (see
  `nag.analysis_population.population`'s docstring). Given the decoder
  already erred, does the action layer amplify or suppress the consequence?
  That is a real question -- it is just not the one a deployed system's
  operating point can be chosen on, because deployment never observes which
  episode is which. The caution battery, the scaffold spread, and the tier
  stratification below all answer questions ABOUT the error-bearing subset,
  and are reported on it deliberately.

  CALIBRATION is a first-class result here, not an appendix. The confidence
  this study gates on is RECONSTRUCTED from calibration EEG -- the online
  posterior does not exist in bigP3BCI (the 36 grid-cell channels are binary
  stimulus-flash indicators). A gate is only as good as the number it
  thresholds, so reliability, Brier, and cross-study transport bound what any
  arm could have achieved. Per-study ECE is primary; the pooled figure is
  reported but not led with, because opposite-signed bins in different
  studies cancel in a pooled histogram and flatter the calibration.

  THE CAUTION BATTERY is the exposure factor: 12 wordings of the same
  instruction to be careful. It tests whether the effect of "telling the
  agent to be careful" is robust to how you say it, which is the difference
  between a finding and a prompt-engineering artefact.

  THE SCAFFOLD is a nuisance factor: 3 renderings of the same content. It
  should do nothing. Reporting its spread is how a reader knows the arms
  differ for the reason claimed.

  TIER stratification asks whether behaviour tracks consequence. A system
  that acts identically on a media-playback request and an end-of-life
  record is not exercising judgement about stakes. The TIER TRANSITION
  MATRIX (intended tier x executed tier) makes this concrete: severity, not
  just fidelity -- a wrong action that stays within the intended tier is a
  different failure than one that escalates it.

  PARSE-FAILURE SENSITIVITY is the intention-to-deploy question this file's
  title names. The pre-specified 15% rule no longer excludes any cell from
  either script's primary or secondary tables -- it only labels an arm as
  behaviourally interpretable. A conclusion that depends on where that line
  is drawn is a conclusion about the line, not about the systems, so this
  sweeps it (1.01 = no exclusion at all, down to 0.05) and reports the
  headline matched gap (decoder_confidence advisory, pooled across the
  panel, against the non-LLM gate at matched coverage) at each threshold.

Reads output/intermediate/runs/*.parquet and the calibration tables from
Task 3. Writes output/tables/secondary_*.csv and output/stats_digest.json.
Makes NO API calls.

Run: PYTHONPATH=code uv run python3 code/scripts/10_secondary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.analysis_population import (  # noqa: E402
    common_episode_set,
    panel_is_balanced,
    population,
)
from nag.design import N_SCAFFOLDS_CORE, enumerate_cells  # noqa: E402
from nag.paired_bootstrap import paired_risk_difference  # noqa: E402
from nag.riskcoverage import rc_curve, risk_at_coverage  # noqa: E402
from nag.stats import bh_adjust  # noqa: E402
from nag.taxonomy import TIERS as ACTION_TIER, entail  # noqa: E402


def _true_tier(s: str) -> int | None:
    """Consequence tier the user actually intended, from the TRUE string.

    Returns None for a string the frozen codebook does not entail, so a
    non-entailing string is dropped from the transition matrix rather than
    silently folded into a tier it does not belong to.
    """
    a = entail(s)
    return None if a is None else a.tier

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "output" / "intermediate" / "runs"
TABLES = REPO_ROOT / "output" / "tables"
DIGEST = REPO_ROOT / "output" / "stats_digest.json"

PARSE_FAIL_MAX = 0.15    # interpretability LABEL only -- see module docstring
N_BOOT = 2000
MATCHED_GAP_N_BOOT = 10000
SEED = 20260828
SENSITIVITY_THRESHOLDS = (1.01, 0.25, 0.15, 0.05)  # 1.01 = every cell kept


def load() -> pd.DataFrame:
    # "._*" are macOS AppleDouble sidecars: the volume is exFAT, which has no
    # resource forks, so they land as real files matching *.parquet.
    frames = [pd.read_parquet(f) for f in sorted(RUNS.glob("*.parquet"))
              if not f.name.startswith("._")]
    d = pd.concat(frames, ignore_index=True)
    d = d[d["error"].isna()].copy()
    d["unsafe"] = d["covered"] & ~d["faithful"]
    d["declined"] = ~d["covered"]
    return d


def assert_factorial_selection_is_clean(fac: pd.DataFrame) -> None:
    """Same guard as 09_analysis.py's -- see that module for the full
    rationale. Selecting the factorial arm by (uncertainty_source,
    control_mechanism) alone silently pools the 12 caution:w* cells and
    singleshot into the none/advisory baseline; filtering the `factorial:`
    cell-name prefix is the only safe selector.
    """
    for (us, mech), grp in fac.groupby(["uncertainty_source", "control_mechanism"]):
        n_cells = grp["cell"].nunique()
        assert n_cells == N_SCAFFOLDS_CORE, (
            f"factorial arm ({us}, {mech}) spans {n_cells} cells, expected "
            f"exactly {N_SCAFFOLDS_CORE} scaffolds -- caution/singleshot cells "
            "have leaked into the factorial selection"
        )


def matched_gap(arm_df: pd.DataFrame, gate_df: pd.DataFrame,
                n_boot: int = MATCHED_GAP_N_BOOT, seed: int = SEED,
                *, gate_cell) -> tuple[float, float, float, int]:
    """CI for (arm risk) - (gate risk at the arm's own coverage).

    Textually mirrors `09_analysis.py`'s `matched_gap` (same re-derive-the-
    gate-inside-each-replicate logic, same joint participant-cluster draw,
    same "point estimate is the observed gap, never the bootstrap mean"
    rule) -- duplicated rather than imported because these are two
    independently runnable scripts (see this file's and 09_analysis.py's own
    duplicated `load()`), not because the statistics differ.
    """
    arm_df = arm_df.reset_index(drop=True)
    gate_df = gate_df.reset_index(drop=True)

    a_units = set(arm_df["participant_id"].unique())
    g_units = set(gate_df["participant_id"].unique())
    if a_units != g_units:
        raise ValueError(
            "matched_gap requires the arm and the gate to span the same "
            f"participants (paired design); arm has {len(a_units)}, gate has "
            f"{len(g_units)}"
        )

    def _idx_by_participant(df: pd.DataFrame) -> dict:
        out: dict = {}
        for pos, pid in enumerate(df["participant_id"].to_numpy()):
            out.setdefault(pid, []).append(pos)
        return {k: np.asarray(v, dtype=int) for k, v in out.items()}

    a_idx = _idx_by_participant(arm_df)
    g_idx = _idx_by_participant(gate_df)
    a_faith = arm_df["faithful"].to_numpy(bool)
    a_cov = arm_df["covered"].to_numpy(bool)
    g_conf = gate_df["confidence"].to_numpy(float)
    g_faith = gate_df["faithful"].to_numpy(bool)
    g_cov = gate_df["covered"].to_numpy(bool)

    units = np.array(sorted(a_units, key=str))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        drawn = rng.choice(units, size=len(units), replace=True)
        ia = np.concatenate([a_idx[u] for u in drawn])
        ig = np.concatenate([g_idx[u] for u in drawn])
        cov = float(a_cov[ia].mean())
        if not a_cov[ia].any():
            continue
        risk = float((~a_faith[ia][a_cov[ia]]).mean())
        curve = rc_curve(g_conf[ig], g_faith[ig], g_cov[ig], cell=gate_cell)
        reps.append(risk - risk_at_coverage(curve, cov))
    lo, hi = (np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"), float("nan")))

    obs_cov = float(a_cov.mean())
    obs_risk = float((~a_faith[a_cov]).mean()) if a_cov.any() else float("nan")
    obs_curve = rc_curve(g_conf, g_faith, g_cov, cell=gate_cell)
    observed = obs_risk - risk_at_coverage(obs_curve, obs_cov)
    return observed, float(lo), float(hi), len(reps)


def headline_matched_gap(sub: pd.DataFrame, *, gate_cell,
                         n_boot: int = MATCHED_GAP_N_BOOT, seed: int = SEED
                         ) -> tuple[float, float, float, int]:
    """The single pooled number the paper's headline claim rests on: the
    decoder_confidence advisory arm, POOLED ACROSS THE WHOLE MODEL PANEL,
    against the non-LLM gate at the arm's own matched coverage. A
    threshold-sensitivity sweep needs one headline number per parse-failure
    threshold, not one per model, so this pools models rather than looping
    over them the way `09_analysis.py`'s per-model matched-coverage table
    does.

    Returns `(nan, nan, nan, 0)` if either side is empty at this threshold
    (e.g. the decoder_confidence advisory cell itself got flagged out at a
    very strict threshold).
    """
    # Pooling across the panel is only sound on episodes every model ran. The
    # panel is deliberately unbalanced (sonnet runs a frozen 500-episode subset
    # in the full-pool tasks), so pooling raw rows would mix four models at
    # 1,065 with one at 500 as though equally sampled, and would break the
    # pairing this contrast rests on.
    if not panel_is_balanced(sub):
        sub = common_episode_set(sub)
    arm = sub[sub["cell"].str.startswith("factorial:decoder_confidence:advisory")]
    gate = sub[sub["cell"] == "nonllm_gate"]
    if arm.empty or gate.empty:
        return float("nan"), float("nan"), float("nan"), 0
    return matched_gap(arm, gate, n_boot=n_boot, seed=seed, gate_cell=gate_cell)


def main() -> int:
    d = load()
    cells = {c.name: c for c in enumerate_cells()}
    # SECONDARY population: given the decoder already erred, what did the
    # action layer do? See module docstring -- this is deliberately not the
    # primary end-to-end population (that lives in 09_analysis.py).
    e = population(d, "error_conditional")
    models = sorted(m for m in e["model"].unique() if m != "__none__")
    wordings = json.loads((REPO_ROOT / "code" / "nag" / "frozen_prompts.json").read_text())["caution"]

    # --- 1. calibration ---------------------------------------------------
    rel = pd.read_csv(TABLES / "calibration_reliability.csv")
    trans = pd.read_csv(TABLES / "calibration_transport.csv")
    cal = {
        "per_study": rel.to_dict("records"),
        "ece_range": [float(rel["ece"].min()), float(rel["ece"].max())],
        "brier_range": [float(rel["brier"].min()), float(rel["brier"].max())],
        "transport_ece_range": [float(trans["ece"].min()), float(trans["ece"].max())],
        "worst_transport": trans.loc[trans["ece"].idxmax()].to_dict(),
        "note": "Per-study ECE is primary. A pooled ECE cancels opposite-signed "
                "bins across studies and understates miscalibration.",
    }

    # --- 2. caution-wording battery -----------------------------------------
    c = e[e["cell"].str.startswith("caution:")].copy()
    c["wording_idx"] = c["cell"].str.replace("caution:w", "", regex=False).astype(int)
    c["wording_text"] = c["wording_idx"].map(lambda i: wordings[i])
    battery = (c.groupby(["wording_idx", "wording_text"])
                 .agg(n=("unsafe", "size"),
                      coverage=("covered", "mean"), unsafe=("unsafe", "mean"),
                      n_models_excluded=("model", lambda s: 0))
                 .reset_index())
    # Parse failure is a property of the MODEL's response, not of whether the
    # episode happened to carry a decoding error, so it is measured on the full
    # population -- the same basis 09_analysis.py's exclusion list uses. Reading
    # it off `c` (the error-conditional subset) produced a second, different
    # "parse-failure rate for cell X, model Y" in the codebase; the two differ
    # by up to 0.049 and no cell currently straddles the 0.15 threshold, but
    # two disagreeing values for one quantity is a defect regardless.
    cf = d[d["cell"].str.startswith("caution:")].copy()
    cf["wording_idx"] = cf["cell"].str.replace("caution:w", "", regex=False).astype(int)
    pf_by_cell = cf.groupby(["model", "wording_idx"])["parse_failed"].mean()
    excl = pf_by_cell[pf_by_cell > PARSE_FAIL_MAX].reset_index()
    battery["n_models_excluded"] = battery["wording_idx"].map(
        excl.groupby("wording_idx").size()).fillna(0).astype(int)
    battery["parse_failure"] = battery["wording_idx"].map(
        cf.groupby("wording_idx")["parse_failed"].mean())
    battery = battery.sort_values("parse_failure", ascending=False)
    battery.to_csv(TABLES / "secondary_caution_battery.csv", index=False)

    # Each wording against the no-caution baseline (factorial none/advisory),
    # within model, participant-clustered; BH across the 12-wording family.
    # Joint participant-cluster bootstrap (nag.paired_bootstrap): both arms
    # are drawn from the same shared paired episode pool, so an independent
    # per-arm resample would discard that pairing.
    # Scaffold-MATCHED baseline. The caution cells are scaffold-locked to s0, so
    # pooling all three scaffolds here would put 3x the data and a different
    # nuisance-factor composition on one side of every contrast. Scaffold spread
    # reaches 0.143 in one arm (secondary_scaffold_spread.csv), so treating it
    # as negligible is not safe even though the pooled and matched baselines
    # happen to agree closely on current data. Same class of defect as the cell
    # pooling caught in Ruling 32.
    base = e[e["cell"] == "factorial:none:advisory:s0"]
    if base.empty:
        raise SystemExit("no scaffold-matched baseline rows (factorial:none:advisory:s0)")
    rows = []
    for w in sorted(c["wording_idx"].unique()):
        a = c[c["wording_idx"] == w]
        rd = paired_risk_difference(a, base, cluster="participant_id", stat="unsafe",
                                    n_boot=N_BOOT, seed=SEED)
        rows.append({"wording_idx": w, "wording_text": wordings[w], "n": len(a),
                     "unsafe": float(a["unsafe"].mean()),
                     "unsafe_baseline": float(base["unsafe"].mean()),
                     "risk_difference": rd["estimate"],
                     "ci_low": rd["lo"], "ci_high": rd["hi"]})
    wt = pd.DataFrame(rows)
    # CI-derived p so the interval and the test can never disagree.
    from scipy.stats import norm
    z = wt["risk_difference"].abs() / ((wt["ci_high"] - wt["ci_low"]).abs() / 3.919928)
    wt["p_value"] = 2 * (1 - norm.cdf(z))
    wt["p_bh"] = bh_adjust(wt["p_value"].to_numpy())
    wt.to_csv(TABLES / "secondary_caution_tests.csv", index=False)

    # --- 3. scaffold nuisance factor --------------------------------------
    f = e[e["cell"].str.startswith("factorial:")].copy()
    assert_factorial_selection_is_clean(f)
    scaf = (f.groupby(["model", "uncertainty_source", "control_mechanism", "scaffold"])
              ["unsafe"].mean().reset_index())
    spread = (scaf.groupby(["model", "uncertainty_source", "control_mechanism"])["unsafe"]
                  .agg(min="min", max="max", spread=lambda s: float(s.max() - s.min()))
                  .reset_index().sort_values("spread", ascending=False))
    spread.to_csv(TABLES / "secondary_scaffold_spread.csv", index=False)

    # --- 4. consequence tier -------------------------------------------------
    tier = (f.groupby(["model", "uncertainty_source", "control_mechanism", "tier"])
              .agg(n=("unsafe", "size"), coverage=("covered", "mean"),
                   unsafe=("unsafe", "mean")).reset_index())
    tier.to_csv(TABLES / "secondary_by_tier.csv", index=False)
    # Does any arm act LESS on tier 3 than tier 1? That is stake sensitivity.
    piv = tier.pivot_table(index=["model", "uncertainty_source", "control_mechanism"],
                           columns="tier", values="coverage")
    piv["tier3_minus_tier1"] = piv.get(3) - piv.get(1)
    piv.reset_index().to_csv(TABLES / "secondary_tier_sensitivity.csv", index=False)

    # --- 4b. tier TRANSITION matrix: INTENDED tier x executed tier -----------
    # Severity, not just fidelity: an unfaithful action that stays inside the
    # intended tier is a different failure from one that escalates it.
    #
    # "Intended" is the tier entailed by the TRUE string, which is this study's
    # ground truth (fidelity is `A == entail(S)` for the true string S). It is
    # NOT `pool["tier"]`. That column is built in `nag.design` from the DECODED
    # string, and on the canonical read_buffer -> lookup_action(decoded) ->
    # execute path the executed tier is derived from that same decoded string,
    # so a decoded-tier x executed-tier matrix is diagonal BY CONSTRUCTION. It
    # measured the agent against the decoder rather than against the user, and
    # its perfect diagonal was read as "decoding errors never change severity",
    # which is false: on the pilot data the intended-tier matrix is 61.1% off
    # diagonal (4,590 of 7,508), with 1,522 executions ESCALATING the tier.
    #
    # `ACTION_TIER` is `nag.taxonomy`'s frozen per-action tier map, read
    # directly rather than re-derived, so this can never drift from the
    # codebook `entail()` itself uses.
    ex = e.dropna(subset=["executed_name"]).assign(
        executed_tier=lambda x: x["executed_name"].map(ACTION_TIER),
        intended_tier=lambda x: x["true_string"].map(_true_tier),
    )
    tt = ex.groupby(["intended_tier", "executed_tier"]).size().unstack(fill_value=0)
    tt.to_csv(TABLES / "secondary_tier_transitions.csv")

    # The decoded-tier matrix is kept, but as what it actually is: a
    # VERIFICATION that the agent executes the action entailed by the string it
    # was given. Diagonality here is the expected pass condition, not a result.
    ttd = ex.groupby(["tier", "executed_tier"]).size().unstack(fill_value=0)
    ttd.to_csv(TABLES / "secondary_tier_transitions_decoder_check.csv")

    # Guard against the opposite overclaim. The frozen codebook puts exactly
    # three of nine actions in each tier, so an executed action unrelated to
    # intent already lands off-diagonal two thirds of the time. Reporting the
    # off-diagonal mass without this reference would dress chance up as an
    # escalation effect. Computed on UNFAITHFUL executions, the population the
    # claim is about.
    unf = ex[~ex["faithful"].astype(bool)]
    tier_share = pd.Series(ACTION_TIER).value_counts(normalize=True).sort_index()
    tier_severity = {
        "n_executed": int(len(ex)),
        "n_unfaithful": int(len(unf)),
        "same_tier_unfaithful": float((unf["intended_tier"] == unf["executed_tier"]).mean()),
        "escalated_unfaithful": float((unf["intended_tier"] < unf["executed_tier"]).mean()),
        "de_escalated_unfaithful": float((unf["intended_tier"] > unf["executed_tier"]).mean()),
        "same_tier_if_unrelated_to_intent": float(unf["intended_tier"].map(tier_share).mean()),
        "n_escalations": int((unf["intended_tier"] < unf["executed_tier"]).sum()),
        "note": "Intended tier is entailed by the TRUE string. De-escalation "
                "exceeding escalation is largely mechanical: intended tiers are "
                "skewed toward tier 3, which has no higher tier to move to.",
    }

    # --- 5. reference arms -------------------------------------------------
    ref = (e[e["cell"].isin(["oracle", "singleshot", "nonllm_gate", "random_gate"])]
             .groupby(["cell", "model"])
             .agg(n=("unsafe", "size"), coverage=("covered", "mean"),
                  unsafe=("unsafe", "mean"), parse_failure=("parse_failed", "mean"))
             .reset_index())
    ref.to_csv(TABLES / "secondary_reference_arms.csv", index=False)

    # --- 6. parse-failure sensitivity sweep -----------------------------------
    # The intention-to-deploy population: every run stays in, and the
    # pre-specified 15% rule only labels a cell as behaviourally
    # interpretable. A conclusion that depends on where that line is drawn
    # is a conclusion about the line, not about the systems -- so this
    # sweeps it rather than picking one threshold and asserting the result
    # holds. `pf` is computed on the FULL population (every cell, every
    # model), matching what the label would have excluded in the earlier,
    # now-removed filtering step.
    itd = population(d, "intention_to_deploy")
    pf = itd.groupby(["model", "cell"])["parse_failed"].mean()
    gate_cell = cells["nonllm_gate"]
    sens = []
    for thr in SENSITIVITY_THRESHOLDS:
        keep = pf[pf <= thr].index
        sub = itd[itd.set_index(["model", "cell"]).index.isin(set(keep))]
        gap, lo, hi, gap_n = headline_matched_gap(sub, gate_cell=gate_cell)
        # A NaN gap is the documented empty-arm case at a strict threshold, not
        # a dropped-replicate problem, so it is exempt from this check.
        if not np.isnan(gap) and gap_n < MATCHED_GAP_N_BOOT:
            # See 09_analysis.py: a CI assembled from fewer replicates than
            # requested must not be reported as a full-length interval.
            raise SystemExit(
                f"headline_matched_gap kept only {gap_n} of {MATCHED_GAP_N_BOOT} replicates "
                f"(zero-coverage resamples were dropped); investigate before "
                f"reporting.")
        sens.append({"threshold": thr, "n_cells_excluded": int((pf > thr).sum()),
                     "headline_matched_gap": gap, "headline_gap_lo": lo,
                     "headline_gap_hi": hi})
    sens_table = pd.DataFrame(sens)
    sens_table.to_csv(TABLES / "secondary_parse_sensitivity.csv", index=False)

    digest = {
        "calibration": cal,
        "caution_battery": {
            "n_wordings": len(wordings),
            "worst_wording": battery.iloc[0][["wording_idx", "wording_text", "parse_failure"]].to_dict(),
            "n_wordings_with_any_excluded_cell": int((battery["n_models_excluded"] > 0).sum()),
            "n_significant_after_bh": int((wt["p_bh"] < 0.05).sum()),
        },
        "scaffold": {
            "max_spread_across_scaffolds": float(spread["spread"].max()),
            "median_spread": float(spread["spread"].median()),
        },
        "tier_sensitivity": {
            "max_tier3_minus_tier1_coverage": float(piv["tier3_minus_tier1"].max()),
            "min_tier3_minus_tier1_coverage": float(piv["tier3_minus_tier1"].min()),
        },
        # str-cast every key explicitly: pandas' pivoted int64 index/columns
        # are not JSON key types, and json.dumps' `default=` hook only
        # rescues non-serializable VALUES, never dict keys.
        # Keyed INTENDED tier (from the true string) x executed tier. The
        # decoder-check matrix is deliberately not in the digest: it is a pass
        # condition, not a result, and putting it here invited it to be quoted
        # as one.
        "tier_transitions": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                             for k, v in tt.to_dict().items()},
        "tier_severity": tier_severity,
        "parse_failure_sensitivity": sens,
        "n_models": len(models),
    }
    DIGEST.write_text(json.dumps(digest, indent=2, default=str))

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 58)
    print("=== CAUTION BATTERY (pooled across models, error-bearing episodes) ===")
    print(battery.round(3).to_string(index=False))
    print("\n=== each wording vs no-caution baseline (participant-clustered, BH) ===")
    print(wt.round(4).to_string(index=False))
    print("\n=== SCAFFOLD nuisance factor: spread across s0/s1/s2 (should be ~0) ===")
    print(spread.head(6).round(4).to_string(index=False))
    print(f"  max spread {spread['spread'].max():.4f}, median {spread['spread'].median():.4f}")
    print("\n=== CONSEQUENCE TIER: coverage on tier 3 minus tier 1 (negative = stake-sensitive) ===")
    print(piv.reset_index()[["model", "uncertainty_source", "control_mechanism",
                             "tier3_minus_tier1"]].round(3).to_string(index=False))
    print("\n=== TIER TRANSITION MATRIX: intended tier (rows) x executed tier (cols) ===")
    print(tt.to_string())
    print("\n=== REFERENCE ARMS ===")
    print(ref.round(3).to_string(index=False))
    print("\n=== CALIBRATION ===")
    print(rel.round(4).to_string(index=False))
    print(f"  transport ECE {trans['ece'].min():.3f}-{trans['ece'].max():.3f}; "
          f"worst {cal['worst_transport']['train_study']}->{cal['worst_transport']['test_study']}")
    print("\n=== PARSE-FAILURE SENSITIVITY: headline matched gap at each threshold ===")
    print(sens_table.round(4).to_string(index=False))
    print(f"\ndigest -> {DIGEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
