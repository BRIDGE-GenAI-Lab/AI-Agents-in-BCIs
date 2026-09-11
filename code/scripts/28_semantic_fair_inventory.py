"""Dataset inventory for the semantic fair-information comparison.

The six pre-specified datasets are inventoried by `17_build_supplement.py` into
`followup_dataset_inventory.csv`, and every run total the manuscript states is
reconciled against that file by `16_number_audit.py`. The fair-information
comparison is a seventh dataset, run later and outside that script's scope, so
its episode-run, request and cost totals had no tabulated source: stating them
in Methods made them UNMATCHED in the number audit, which is how a mistyped
total would look. This writes them from the run rows themselves.

    python3 code/scripts/28_semantic_fair_inventory.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "output" / "intermediate" / "runs_semantic_fair"
OUT = REPO_ROOT / "output" / "tables" / "semantic_fair_dataset_inventory.csv"


def main() -> int:
    files = sorted(f for f in RUNS.glob("*.parquet") if not f.name.startswith("._"))
    if not files:
        print(f"no run files under {RUNS}")
        return 1
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # The sentinel model identifier carries the two non-LLM comparator arms,
    # which issue no request; it is not a model and is excluded from the count.
    models = sorted(m for m in d["model"].unique() if d.loc[d["model"] == m, "uses_llm"].any())

    row = {
        "run directory": "`runs_semantic_fair/`",
        "episode runs": len(d),
        "cells": d["cell"].nunique(),
        "episodes": d["episode_id"].nunique(),
        "models": len(models),
        "API requests": int(d["n_api_calls"].sum()),
        "retry attempts": int(d["n_retries"].sum()),
        "measured cost (US $)": round(float(d["cost_usd"].sum()), 2),
        "episode runs with at least one retry": int((d["n_retries"] > 0).sum()),
        "episode runs recording an error": int(d["error"].notna().sum()),
    }
    pd.DataFrame([row]).to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    for k, v in row.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
