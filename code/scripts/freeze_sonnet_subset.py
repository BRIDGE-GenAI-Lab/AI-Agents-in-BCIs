"""Freeze the 500-episode subset claude-sonnet-5 runs in Tasks 12 and 19.

Measured API cost for `anthropic/claude-sonnet-5` is $0.00876 per episode-cell
against the $0.00550 the manifest projected, 1.59x. At the full 1,065-episode
pool that one model would consume $37.32 of a $100 ceiling with $63.43 left, so
the approved plan did not fit. The human's decision was to buy the shortfall
with PRECISION on the expensive model rather than by deleting an experiment or
dropping the model from the panel: sonnet runs a frozen 500-episode subset in
the two full-pool tasks, every other model runs the complete pool, and Tasks 13,
14 and 20 are untouched on all five models.

Two properties this file exists to guarantee.

FIRST, the subset is drawn, not taken off the top. `--episodes N` in the runners
truncates to the first N of the frozen order, which is only sound if that order
is random with respect to error and participant. It is not guaranteed to be, so
truncation could quietly shift decoder-error prevalence or over-weight whichever
participants sort early. This samples without replacement under seed 20260830,
stratified on error-bearing status and, within that, allocated across
participants in proportion to their share of the source pool by largest
remainder.

SECOND, one subset, used everywhere. Every sonnet condition and comparator in
Tasks 12 and 19 reads these same 500 ids, so all within-sonnet contrasts stay
exactly paired. A per-task or per-cell draw would silently unpair them, which is
the property the primary endpoint depends on.

Writes `principal_run.model_episode_subsets` into run_manifest.json. Makes no
API calls and spends nothing. Commit the manifest before running sonnet.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"

MODEL = "anthropic/claude-sonnet-5"
SEED = 20260830
N_ERROR_BEARING = 170
N_CLEAN = 330

# Measured per episode-cell over the 400 completed pilot rows per model across
# the four principal cells; retries included. These replace the projections the
# manifest shipped with, every one of which was wrong.
MEASURED_USD_PER_EPISODE = {
    "anthropic/claude-sonnet-5": 0.00876,
    "google/gemini-3.7-flash": 0.00158,
    "openai/gpt-5.6-luna": 0.00024,
    "z-ai/glm-5.3-flash": 0.00016,
    "deepseek/deepseek-v4-flash": 0.00010,
}


def _largest_remainder(shares: dict, total: int) -> dict:
    """Allocate `total` across `shares` proportionally, summing EXACTLY to total.

    Plain rounding does not sum to the target, and topping the difference up
    from whichever key happens to be first would bias the composition toward
    it. Largest remainder puts each leftover unit where the rounding loss was
    greatest.
    """
    denom = sum(shares.values())
    exact = {k: total * v / denom for k, v in shares.items()}
    floors = {k: int(np.floor(v)) for k, v in exact.items()}
    short = total - sum(floors.values())
    order = sorted(shares, key=lambda k: (-(exact[k] - floors[k]), str(k)))
    for k in order[:short]:
        floors[k] += 1
    return floors


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "code"))
    from nag.design import build_episode_pool

    manifest = json.loads(MANIFEST.read_text())
    frozen = manifest["principal_run"]["episode_set"]["episode_ids"]

    pool = build_episode_pool().set_index("episode_id")
    missing = [e for e in frozen if e not in pool.index]
    if missing:
        print(f"{len(missing)} frozen episode(s) absent from the pool; refusing",
              file=sys.stderr)
        return 1
    sub = pool.loc[frozen]

    err = sub["err"].astype(bool)
    strata = {True: N_ERROR_BEARING, False: N_CLEAN}
    src_prev = float(err.mean())

    rng = np.random.default_rng(SEED)
    chosen: list[str] = []
    for is_err, want in strata.items():
        block = sub[err == is_err]
        if len(block) < want:
            print(f"stratum err={is_err} has {len(block)} < {want} requested",
                  file=sys.stderr)
            return 1
        by_participant = Counter(block["participant_id"])
        alloc = _largest_remainder(dict(by_participant), want)
        for pid, k in sorted(alloc.items(), key=lambda kv: str(kv[0])):
            if k <= 0:
                continue
            ids = sorted(block.index[block["participant_id"] == pid])
            chosen.extend(rng.choice(ids, size=k, replace=False).tolist())

    chosen = sorted(set(chosen))
    if len(chosen) != N_ERROR_BEARING + N_CLEAN:
        print(f"drew {len(chosen)}, expected {N_ERROR_BEARING + N_CLEAN}", file=sys.stderr)
        return 1

    picked = sub.loc[chosen]
    got_prev = float(picked["err"].astype(bool).mean())

    manifest["principal_run"].setdefault("model_episode_subsets", {})[MODEL] = {
        "n": len(chosen),
        "seed": SEED,
        "n_error_bearing": int(picked["err"].astype(bool).sum()),
        "n_clean": int((~picked["err"].astype(bool)).sum()),
        "source_pool_n": len(frozen),
        "source_error_prevalence": round(src_prev, 6),
        "subset_error_prevalence": round(got_prev, 6),
        "n_participants": int(picked["participant_id"].nunique()),
        "source_n_participants": int(sub["participant_id"].nunique()),
        "applies_to_tasks": ["12", "19"],
        "reason": "Measured cost for this model was 1.59x the manifest projection, "
                  "so the approved plan did not fit the $100 ceiling. The shortfall "
                  "is paid in precision on this model rather than by dropping an "
                  "experiment or a model.",
        "episode_ids": chosen,
    }

    for m in manifest["model_panel"]["models"]:
        if m["slug"] in MEASURED_USD_PER_EPISODE:
            m["projected_usd_per_episode_original"] = m["projected_usd_per_episode"]
            m["projected_usd_per_episode"] = MEASURED_USD_PER_EPISODE[m["slug"]]
    manifest["model_panel"]["projection_basis"] = (
        "Measured per episode-cell from the completed pilot (400 rows per model "
        "across the four principal cells), retries included. The original shipped "
        "projections were wrong for every model and 1.59x low for "
        "anthropic/claude-sonnet-5, which is 81% of the remaining bill; they are "
        "retained per model as projected_usd_per_episode_original."
    )

    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"model            {MODEL}")
    print(f"seed             {SEED}")
    print(f"subset           {len(chosen)} of {len(frozen)}")
    print(f"error-bearing    {int(picked['err'].astype(bool).sum())} "
          f"(prevalence {got_prev:.4f} vs source {src_prev:.4f})")
    print(f"participants     {picked['participant_id'].nunique()} of "
          f"{sub['participant_id'].nunique()}")
    print(f"applies to       Tasks 12 and 19; every sonnet condition uses these ids")
    print(f"\nwrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
