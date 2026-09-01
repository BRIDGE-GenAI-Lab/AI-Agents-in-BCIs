"""Write the FINAL model panel, the main-run episode sample, and the HUMAN-SET
budget ceiling into output/tables/run_manifest.json.

Separate from 05_design.py because 05 ends at the budget gate by design: it
emits `budget_usd: null` and only a human may set it. This script carries the
human's decision (panel composition, episodes/cell, and the $100 ceiling) into
the manifest. It makes NO API calls.

Panel rationale (all five are current-generation, one per vendor; prices pulled
live from the OpenRouter model list on the date below, never recalled):
  openai/gpt-5.6-luna          capable end, cheapest frontier
  anthropic/claude-sonnet-5    capable end, most expensive -- the ceiling arm
  google/gemini-3.7-flash      mid
  z-ai/glm-5.3-flash           open-weight, cheapest on the list
  deepseek/deepseek-v4-flash   open-weight alternate

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
BUDGET_USD = 100.0          # HUMAN-SET. Do not change without the human saying so.
PRICES_PULLED = "2026-08-28"

# ($/input token, $/output token) exactly as the live OpenRouter list reported.
PANEL = {
    "openai/gpt-5.6-luna":        (0.0000002,   0.0000012),
    "anthropic/claude-sonnet-5":  (0.000002,    0.00001),
    "google/gemini-3.7-flash":    (0.00000075,  0.00000375),
    "z-ai/glm-5.3-flash":         (0.000000075, 0.00000025),
    "deepseek/deepseek-v4-flash": (0.0000000868, 0.0000001736),
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

    cells = enumerate_cells()
    llm_cells = [c for c in cells if c.uses_llm]
    non_llm_cells = [c.name for c in cells if not c.uses_llm]

    pool = build_episode_pool()
    sampled = sample_episodes(pool, n_total=N_PER_CELL, seed=SEED)
    episode_ids = sampled["episode_id"].tolist()

    def per_episode(model: str) -> float:
        pin, pout = PANEL[model]
        return OBSERVED_PROMPT_TOKENS * pin + OBSERVED_COMPLETION_TOKENS * pout

    n_caution_cells = sum(1 for c in llm_cells if c.name.startswith("caution:"))
    total_runs = len(PANEL) * len(llm_cells) * N_PER_CELL
    projected = sum(per_episode(m) for m in PANEL) * len(llm_cells) * N_PER_CELL

    manifest["model_panel"] = {
        "models": [
            {"slug": m, "price_in_per_token": PANEL[m][0], "price_out_per_token": PANEL[m][1],
             "projected_usd_per_episode": round(per_episode(m), 6)}
            for m in PANEL
        ],
        "n_models": len(PANEL),
        "prices_pulled_from_live_openrouter_list": PRICES_PULLED,
        "note": "One vendor each, all current-generation. Every model runs every "
                "LLM cell -- there is no core/panel split, so every between-arm "
                "contrast is within-model as well as within-episode.",
    }
    manifest["main_run"] = {
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
    manifest["budget_note"] = (
        f"HUMAN-SET hard ceiling ${BUDGET_USD:.2f}. Projected spend "
        f"${projected:.2f} leaves ${BUDGET_USD - projected:.2f} of headroom -- "
        "enough for one full re-run. The runner must abort when cumulative "
        "observed `usage.cost` reaches this ceiling."
    )

    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"panel: {len(PANEL)} models x {len(llm_cells)} LLM cells x {N_PER_CELL} episodes")
    print(f"  ({n_caution_cells} of those cells ARE the caution battery -- not priced separately)")
    print(f"  non-LLM cells (zero API cost, model-independent): {non_llm_cells}")
    print(f"  TOTAL {total_runs:,} runs  ${projected:.2f}  of ${BUDGET_USD:.2f} ceiling")
    print(f"  headroom ${BUDGET_USD - projected:.2f} ({(BUDGET_USD - projected) / projected:.1f}x re-runs)")
    print(f"  sample: {len(episode_ids)} episodes, "
          f"{sampled['participant_id'].nunique()} participants, "
          f"error-bearing {sampled['err'].mean():.1%}")


if __name__ == "__main__":
    main()
