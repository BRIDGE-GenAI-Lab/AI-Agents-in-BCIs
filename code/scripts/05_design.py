"""Design enumeration, the one shared stratified episode sample, and the cost
envelope. Ends at a HUMAN BUDGET GATE: this script makes NO API calls of any
kind, and writes `budget_usd: null` into the manifest -- only a human sets it.

Reads output/intermediate/selection_scores.parquet (Task 3b) and the sibling
project's online_trials_all20.csv. Writes:
  output/tables/run_manifest.json  -- cells, the chosen episode ids, per-model
    allocation, seeds, the pre-specified earlier_session exclusion, and the
    mapping/schema/prompts digests.
  output/tables/cost_envelope.csv  -- across price blends x cached_frac.
  output/tables/power_curve.csv    -- CI half-width vs episode count,
    clustered on participant.

Run: PYTHONPATH=code uv run python3 code/scripts/05_design.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.design import (  # noqa: E402
    CALLS_PER_EPISODE_CEILING,
    CALLS_PER_EPISODE_LIKELY,
    DEFAULT_SAMPLE_SIZE,
    EARLIER_SESSION,
    REALISTIC_BLEND,
    build_episode_pool,
    cost_envelope,
    enumerate_cells,
    episode_set_digest,
    natural_error_rate,
    power_curve,
    run_allocation,
    sample_episodes,
)
from nag.prompts import prompts_digest  # noqa: E402
from nag.taxonomy import mapping_digest  # noqa: E402
from nag.tools import schema_digest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "output" / "tables"
SEED = 20260828  # date-derived, fixed and recorded

N_TOTAL = DEFAULT_SAMPLE_SIZE
POWER_N_GRID = [100, 150, 200, 300, 400, 600, 800, 1200]


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("building the fixed ALS episode pool (build_episodes + selection_scores join + calibration)...")
    pool = build_episode_pool()
    n_pool = len(pool)
    n_participants_pool = pool["participant_id"].nunique()
    n_err_pool = int(pool["err"].sum())
    n_primary_pool = int((pool["fit_match"] != EARLIER_SESSION).sum())
    n_participants_error_bearing = pool.loc[pool["err"], "participant_id"].nunique()
    print(
        f"  pool: {n_pool} usable episodes, {n_participants_pool} participants, "
        f"{n_err_pool} error-bearing ({n_err_pool / n_pool:.1%}), "
        f"{n_primary_pool} primary-eligible (excl. {EARLIER_SESSION}), "
        f"{n_participants_error_bearing} participants with >=1 error-bearing episode"
    )

    print(f"drawing the ONE shared stratified sample (n_total={N_TOTAL}, seed={SEED})...")
    sampled = sample_episodes(pool, n_total=N_TOTAL, seed=SEED)
    n_sampled = len(sampled)
    n_sampled_participants = sampled["participant_id"].nunique()
    err_frac = float(sampled["err"].mean())
    tier_counts = sampled["tier"].value_counts().sort_index().to_dict()
    fit_match_counts = sampled["fit_match"].value_counts().to_dict()
    n_primary = int(sampled["primary_eligible"].sum())
    n_sensitivity_only = n_sampled - n_primary
    print(
        f"  sampled: {n_sampled} episodes, {n_sampled_participants} participants, "
        f"error-bearing {err_frac:.1%}, tiers {tier_counts}, fit_match {fit_match_counts}"
    )
    print(f"  primary-eligible: {n_primary} ({EARLIER_SESSION} sensitivity-only: {n_sensitivity_only})")

    cells = enumerate_cells()
    alloc = run_allocation(n_total=n_sampled, cells=cells)

    manifest = dict(
        seed=SEED,
        design=dict(
            fully_paired=True,
            note=(
                "Every cell below shares the SAME episode set (Ruling 6) -- the "
                "matched-coverage contrast is within-episode, not a cross-sample comparison."
            ),
            n_total_shared_episodes=n_sampled,
        ),
        cells=[
            dict(
                name=c.name, uncertainty_source=c.uncertainty_source, control_mechanism=c.control_mechanism,
                scaffold=c.scaffold, wording=c.wording, uses_llm=c.uses_llm,
            )
            for c in cells
        ],
        episode_pool=dict(
            n_usable_episodes=n_pool, n_participants=n_participants_pool, n_error_bearing=n_err_pool,
            natural_error_rate=natural_error_rate(pool), n_primary_eligible=n_primary_pool,
            n_participants_with_error_bearing_episode=n_participants_error_bearing,
        ),
        shared_sample=dict(
            n_total=n_sampled,
            n_participants=n_sampled_participants,
            error_bearing_frac=err_frac,
            tier_counts=tier_counts,
            fit_match_counts=fit_match_counts,
            n_primary_eligible=n_primary,
            n_sensitivity_only=n_sensitivity_only,
            episode_ids=sorted(sampled["episode_id"].tolist()),
            episode_set_digest=episode_set_digest(sampled["episode_id"]),
        ),
        preregistered_exclusions=dict(
            primary_analysis_excludes=EARLIER_SESSION,
            rationale=(
                "fit_match == 'earlier_session' (n=19 episodes / 77 selections in the full pool) is "
                "ANTI-predictive: mean score_top gap for correct vs wrong selections is -0.0473, versus "
                "+0.0275 (own_session_own_condition) and +0.0565 (own_session_other_condition). Pooling "
                "it into calibration or risk-coverage reporting degrades the apparent gate quality rather "
                "than flattering it. Excluded from the PRIMARY analysis; included in a required "
                "sensitivity analysis. Decided before any model call, per Ruling 23."
            ),
        ),
        model_allocation=alloc,
        calls_per_episode=dict(likely=CALLS_PER_EPISODE_LIKELY, ceiling=CALLS_PER_EPISODE_CEILING),
        variance_substudy=dict(
            cells=5, repetitions=30, episodes=50,
            note="Repetitions estimate model stochasticity only, not independent neural observations.",
        ),
        digests=dict(
            mapping_digest=mapping_digest(),
            schema_digest=schema_digest(),
            prompts_digest=prompts_digest(),
        ),
        budget_usd=None,  # HUMAN BUDGET GATE: only a human sets this
    )

    manifest_path = TABLES_DIR / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nwrote {manifest_path}")

    print("computing the cost envelope...")
    cost_df = cost_envelope(n_total=n_sampled, cells=cells)
    cost_path = TABLES_DIR / "cost_envelope.csv"
    cost_df.to_csv(cost_path, index=False)
    print(f"wrote {cost_path}")
    realistic = cost_df[cost_df["blend"] == "realistic_15model"]
    print("\nrealistic 15-model blend ($1.00/M in, $4.57/M out):")
    print(realistic.to_string(index=False))

    print("\nfull cost table:")
    print(cost_df.to_string(index=False))

    print("\ncomputing power curves (clustered on participant)...")
    pc = power_curve(rate=err_frac, n_grid=POWER_N_GRID, n_participants=n_sampled_participants)
    power_path = TABLES_DIR / "power_curve.csv"
    pc.to_csv(power_path, index=False)
    print(f"wrote {power_path}")
    print(pc.to_string(index=False))
    print(
        f"\nPrecision is bounded by participant count ({n_sampled_participants} in this sample; "
        f"{n_participants_pool} in the full pool; {n_participants_error_bearing} have >=1 "
        "error-bearing episode), NOT by episode count -- see how little the clustered columns "
        "narrow between n=300 and n=1200 above, versus the naive column."
    )

    print("\n" + "=" * 70)
    print("HUMAN BUDGET GATE -- STOPPING HERE.")
    print("budget_usd is null in run_manifest.json. No API calls were made.")
    print("Do not proceed to Phase B execution without a human-approved spend ceiling.")
    print("=" * 70)


if __name__ == "__main__":
    main()
