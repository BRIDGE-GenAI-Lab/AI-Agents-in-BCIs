"""Write the FINAL model panel, the main-run episode sample, and the HUMAN-SET
budget ceiling into output/tables/run_manifest.json.

Separate from 05_design.py because 05 ends at the budget gate by design: it
emits `budget_usd: null` and only a human may set it. This script carries the
human's decision (panel composition, episodes/cell, and the ceiling -- $100 at
first, twice raised since, see BUDGET_USD) into the manifest. It makes NO API
calls.

The `main_run` block it writes is a PROJECTION of a full-factorial design that
was never executed, and its figures scale with whatever the panel currently
holds. The published run is `principal_run`, on the five models recorded in
`model_panel_at_principal_run`; see the `status` string written below.

Panel rationale (all five are current-generation, one per vendor; prices pulled
live from the OpenRouter model list on the date below, never recalled):
  openai/gpt-5.6-luna          capable end, cheapest frontier
  anthropic/claude-sonnet-5    capable end, most expensive -- the ceiling arm
  google/gemini-3.7-flash      mid
  z-ai/glm-5.3-flash           open-weight, cheapest on the list
  deepseek/deepseek-v4-flash   open-weight alternate

2026-09-09: expanded with 5 more current-generation models for the semantic
fair-comparison experiment (Task 5 of the 2026-09-09 plan), leaning toward
reasoning/agentic-tool-use strength per the reviewer's "8-10 strategically
chosen models" critique. Prices again pulled live and human-confirmed:
  x-ai/grok-4.6                    reasoning-leaning frontier
  qwen/qwen3.8-max-0902            reasoning-leaning frontier
  moonshotai/kimi-k3               agentic tool-use focus
  nvidia/nemotron-3.5-lightning    cheapest on the expanded list
  mistralai/mistral-medium-3-5     included despite being older (2026-04-30)
                                    than the other four -- human confirmed
                                    knowingly

Run: PYTHONPATH=code uv run python3 code/scripts/05b_panel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.design import (  # noqa: E402
    build_episode_pool,
    enumerate_cells,
    episode_set_digest,
    sample_episodes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"

SEED = 20260828
N_PER_CELL = 100
BUDGET_USD = 120.0          # HUMAN-SET. Raised 100.0 -> 115.0 on 2026-09-09 (commit 032e96d)
                            # for the semantic fair-comparison experiment, then 115.0 -> 120.0
                            # on 2026-09-10 (commit f2e01d2) once Task 6's real per-episode
                            # costs came in 1.16x-1.78x above this file's projections. Do not
                            # change again without the human saying so. Both raises were made
                            # in the manifest first, and this constant was left at the old
                            # value for a day each time -- so `main()` now refuses to write a
                            # ceiling BELOW the one the manifest already carries rather than
                            # silently undoing a raise it has not been told about.
PRICES_PULLED = "2026-09-09"

# ($/input token, $/output token) exactly as the live OpenRouter list reported.
PANEL = {
    "openai/gpt-5.6-luna":        (0.0000002,   0.0000012),
    "anthropic/claude-sonnet-5":  (0.000002,    0.00001),
    "google/gemini-3.7-flash":    (0.00000075,  0.00000375),
    "z-ai/glm-5.3-flash":         (0.000000075, 0.00000025),
    "deepseek/deepseek-v4-flash": (0.0000000868, 0.0000001736),
    "x-ai/grok-4.6":                 (0.000002,    0.000006),
    "qwen/qwen3.8-max-0902":         (0.000002,    0.000006),
    "moonshotai/kimi-k3":            (0.000003,    0.000015),
    "nvidia/nemotron-3.5-lightning": (0.00000008,  0.0000002),
    "mistralai/mistral-medium-3-5":  (0.0000015,   0.0000075),
}
# The 12 caution-wording cells (`caution:w0`..`w11`) are ALREADY among the 32
# LLM cells `enumerate_cells()` returns -- they are not a separate battery
# bolted on top. An earlier version of this script priced them twice (a
# "caution battery on 3 of 5 models" term added to a main term that already
# contained them), overstating the projection by $7.52. That made sense only
# under the ORIGINAL allocation, where 6 core models ran all 32 cells and 15
# panel models ran just the 6 factorial base cells. The human replaced that
# with "all 5 models run all 32 cells", which subsumes the battery entirely
# and makes every wording contrast within-model as well as within-episode.

# Measured, not modelled: means over the 3 successful qwen-2.5-72b episodes in
# output/tables/cost_probe.json (the 3-call, fuller-prompt end of the observed
# distribution, so the projection errs high).
OBSERVED_PROMPT_TOKENS = 2230
OBSERVED_COMPLETION_TOKENS = 104


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    # A budget ceiling only ever moves by a human decision, and every one of
    # those decisions has landed in the manifest before it landed here. A
    # BUDGET_USD below the manifest's own value therefore means this constant is
    # stale, not that the ceiling has been lowered -- and writing it would undo a
    # raise the runners are already spending against.
    already_set = manifest.get("budget_usd")
    if already_set is not None and already_set > BUDGET_USD:
        raise SystemExit(
            f"REFUSING TO WRITE. {MANIFEST.name} already carries a HUMAN-SET ceiling of "
            f"${already_set:.2f} and this script would lower it to ${BUDGET_USD:.2f}, which no "
            "human asked for. BUDGET_USD in this file is stale: set it to the human's current "
            "ceiling and re-run."
        )
    # A prior run can have hand-corrected a model's projected_usd_per_episode
    # with real MEASURED pilot cost (commit a66a649 did this for all 5
    # original models -- the naive OBSERVED_PROMPT/COMPLETION_TOKENS formula
    # below was wrong for every one of them, 1.59x low for
    # anthropic/claude-sonnet-5). That correction is recognisable by the
    # presence of `projected_usd_per_episode_original` (this script itself
    # never writes that key). Without this check, re-running this script --
    # its own documented `Run:` command above -- would silently overwrite
    # the correction with the naive number again on every future run.
    old_models = {m["slug"]: m for m in manifest.get("model_panel", {}).get("models", [])}

    cells = enumerate_cells()
    llm_cells = [c for c in cells if c.uses_llm]
    non_llm_cells = [c.name for c in cells if not c.uses_llm]

    pool = build_episode_pool()
    sampled = sample_episodes(pool, n_total=N_PER_CELL, seed=SEED)
    episode_ids = sampled["episode_id"].tolist()

    def naive_per_episode(model: str) -> float:
        pin, pout = PANEL[model]
        return OBSERVED_PROMPT_TOKENS * pin + OBSERVED_COMPLETION_TOKENS * pout

    def per_episode(model: str) -> float:
        old = old_models.get(model)
        if old is not None and "projected_usd_per_episode_original" in old:
            return old["projected_usd_per_episode"]
        return naive_per_episode(model)

    n_caution_cells = sum(1 for c in llm_cells if c.name.startswith("caution:"))
    total_runs = len(PANEL) * len(llm_cells) * N_PER_CELL
    projected = sum(per_episode(m) for m in PANEL) * len(llm_cells) * N_PER_CELL

    preserved_measured = []
    model_entries = []
    for m in PANEL:
        old = old_models.get(m)
        entry = {"slug": m, "price_in_per_token": PANEL[m][0], "price_out_per_token": PANEL[m][1]}
        if old is not None and "projected_usd_per_episode_original" in old:
            entry["projected_usd_per_episode"] = old["projected_usd_per_episode"]
            entry["projected_usd_per_episode_original"] = old["projected_usd_per_episode_original"]
            preserved_measured.append(m)
        else:
            entry["projected_usd_per_episode"] = round(naive_per_episode(m), 6)
        model_entries.append(entry)

    manifest["model_panel"] = {
        "models": model_entries,
        "n_models": len(PANEL),
        "prices_pulled_from_live_openrouter_list": PRICES_PULLED,
        "note": "One vendor each, all current-generation. Every model runs every "
                "LLM cell -- there is no core/panel split, so every between-arm "
                "contrast is within-model as well as within-episode.",
    }
    if preserved_measured:
        manifest["model_panel"]["projection_basis"] = (
            f"{', '.join(preserved_measured)}: projected_usd_per_episode is MEASURED "
            "cost from a completed pilot (see commit a66a649), carried forward "
            "automatically from the prior manifest rather than recomputed from "
            "OBSERVED_PROMPT/COMPLETION_TOKENS below -- that naive formula was "
            "wrong for every one of these models. projected_usd_per_episode_original "
            "keeps the naive value for reference. Every other model below has no "
            "pilot data yet and uses the naive formula."
        )
    manifest["main_run"] = {
        "status": (
            "HISTORICAL / SUPERSEDED -- NOT A RECORD OF ANYTHING THAT RAN. This block is "
            "05b_panel.py's own projection of a full-factorial design (every panel model x "
            "every LLM cell x n_episodes_per_cell episodes) that was never executed. Its "
            "figures are recomputed from the CURRENT panel size on every run of that script, "
            f"so `episode_runs.total` reads {total_runs} under today's {len(PANEL)}-model panel "
            "and read 16000 under the five-model panel that produced the manuscript -- neither "
            "number counts an episode that was run, and neither appears in the paper. What "
            "actually ran, and what the manuscript reports, is in `principal_run`: 6 cells over "
            "1065 episodes on the five models recorded in `model_panel_at_principal_run`, with "
            "anthropic/claude-sonnet-5 on a frozen 500-episode subset. `projected_cost_usd` "
            "here is stale for the same reason -- see `budget_note`."
        ),
        "n_episodes_per_cell": N_PER_CELL,
        "episode_set_digest": episode_set_digest(episode_ids),
        "episode_ids": episode_ids,
        "n_participants": int(sampled["participant_id"].nunique()),
        "error_bearing_frac": round(float(sampled["err"].mean()), 4),
        "tier_counts": {str(k): int(v) for k, v in sampled["tier"].value_counts().sort_index().items()},
        "fit_match_counts": {str(k): int(v) for k, v in sampled["fit_match"].value_counts().items()},
        "n_primary_eligible": int(sampled["primary_eligible"].sum()),
        "llm_cells": len(llm_cells),
        "non_llm_cells": non_llm_cells,
        "caution_cells_included_in_llm_cells": n_caution_cells,
        "episode_runs": {"total": total_runs,
                         "note": "every model runs every LLM cell, the 12 caution "
                                 "wordings included -- there is no separate battery"},
        "projected_cost_usd": {
            "total": round(projected, 2),
            "basis": f"measured {OBSERVED_PROMPT_TOKENS} prompt / "
                     f"{OBSERVED_COMPLETION_TOKENS} completion tokens per episode "
                     "(output/tables/cost_probe.json), no cache discount assumed",
        },
    }
    manifest["budget_usd"] = BUDGET_USD
    # Written only when absent. The note in the manifest is the hand-written
    # audit trail of every raise -- what was spent, what prompted it, which
    # projections had gone stale -- and regenerating it here would replace that
    # history with the one-line summary below. The projection it would have
    # carried is not lost: it is written to main_run.projected_cost_usd and
    # printed at the end of this run.
    manifest.setdefault("budget_note", (
        f"HUMAN-SET hard ceiling ${BUDGET_USD:.2f}. Projected spend "
        f"${projected:.2f} leaves ${BUDGET_USD - projected:.2f} of headroom -- "
        "enough for one full re-run. The runner must abort when cumulative "
        "observed `usage.cost` reaches this ceiling."
    ))

    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"panel: {len(PANEL)} models x {len(llm_cells)} LLM cells x {N_PER_CELL} episodes")
    if preserved_measured:
        print(f"  measured (not naive) cost preserved for: {', '.join(preserved_measured)}")
    print(f"  ({n_caution_cells} of those cells ARE the caution battery -- not priced separately)")
    print(f"  non-LLM cells (zero API cost, model-independent): {non_llm_cells}")
    print(f"  TOTAL {total_runs:,} runs  ${projected:.2f}  of ${BUDGET_USD:.2f} ceiling")
    print(f"  headroom ${BUDGET_USD - projected:.2f} ({(BUDGET_USD - projected) / projected:.1f}x re-runs)")
    print(f"  sample: {len(episode_ids)} episodes, "
          f"{sampled['participant_id'].nunique()} participants, "
          f"error-bearing {sampled['err'].mean():.1%}")


if __name__ == "__main__":
    main()
