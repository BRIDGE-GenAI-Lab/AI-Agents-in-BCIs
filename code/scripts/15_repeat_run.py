"""TASK 14: stochastic-repetition check (review item R1-7) -- 3 independent
generations of the two advisory arms, five models, the SAME frozen
100-episode set as the main study, all at the SAME temperature 0.7 used
throughout.

WHY ONLY THESE TWO ARMS, AND WHY THREE REPETITIONS, NOT FIVE. Scope was cut
twice, deliberately, and this is stated here (and must be stated in the
Methods) rather than left implicit: five repetitions became three to help
fund Task 20's naturalistic benchmark; four arms became two (`factorial:
none:advisory:s0` and `factorial:decoder_confidence:advisory:s0`) to fund a
FULL-POOL Task 19 while keeping a real budget buffer. The two arms kept are
the ones carrying the study's central fixed-point comparison, and the ones
that showed the strongest variation in abstention and parse behaviour in the
exploratory run -- they are where stochasticity is most likely to matter.

WHAT THIS DOES AND DOES NOT ESTABLISH. This restriction to two ADVISORY arms
is a SCOPE LIMIT, not a claim that enforced arms are deterministic. An
enforced cell's model call is identical in every respect except what
`nag.prompts.build_system` renders (see `nag.agent.run_episode_for_cell`'s
docstring) -- model stochasticity still affects an enforced arm's PROPOSED
action, its parse behaviour, and its voluntary abstention; only the
THRESHOLD is withheld from the model, and the threshold is not the only
source of variability an enforced cell has. The Methods must report this
restriction as a scope limit on what was measured, never imply enforced arms
were checked and found stable.

REUSE, NOT REIMPLEMENTATION. Both arms here are existing, unmodified cells
(`nag.design.enumerate_cells()`), run through the completely unmodified
`nag.agent.run_episode_for_cell` / `nag.tools.Environment` via `08_run.py`'s
`run_cell` -- exactly as `13_principal_run.py` and `18_recalibrated_run.py`
reuse it. Nothing about a "repetition" needs new agent-loop logic: at
temperature 0.7 (`run08.TEMPERATURE`, unchanged), the OpenRouter completion
itself is server-side stochastic, so calling `run_cell` again for the SAME
(model, cell, episode) triple is already an independent draw -- no seed
manipulation is needed or attempted here to force variation. `repetition`
(0, 1, 2) is attached to each row purely as a label, never used to alter the
call. Checkpoints are one parquet per (model, cell, repetition) under
`output/intermediate/runs_repeat/`, so any one repetition is independently
resumable exactly like a `08_run.py`/`13_principal_run.py` (model, cell)
checkpoint.

`repetitions estimate model stochasticity only`: they are three draws of the
SAME (model, cell, episode) triple, never treated as three additional
independent neural observations -- an episode's decoded/true string and
calibrated confidence are identical across its three repetitions, only the
model's response differs. Downstream reporting must summarise repetitions
SEPARATELY (mean and range across the 3 runs, per arm and model, of
coverage / unfaithful execution / explicit abstention / parse failure --
`output/tables/secondary_repeat_variability.csv`) rather than pooling them
into the primary analysis's episode count.

BUDGET. `spent_so_far()` is textually identical to the other three review-
response runner scripts' -- see `18_recalibrated_run.py`'s docstring.

Run:
  UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/15_repeat_run.py --dry-run

  OPENROUTER_API_KEY=... UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/15_repeat_run.py

  # after it finishes (the secondary_repeat_variability.csv report is built
  # by the downstream analysis step, not this runner):
  git add code/scripts/15_repeat_run.py output/intermediate/runs_repeat/ output/tables/secondary_repeat_variability.csv
  git commit -m "run: 3-repetition stochasticity check on the two advisory arms (review item R1-7)"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

from nag.design import build_episode_pool, enumerate_cells  # noqa: E402
from nag.openrouter import resolve_endpoint  # noqa: E402


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run08 = _load_sibling("_nag_run08", "08_run.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
RUNS_DIR = REPO_ROOT / "output" / "intermediate" / "runs_repeat"

# Identical across all four review-response runner scripts -- see
# 18_recalibrated_run.py's spent_so_far() docstring for why.
# Discovered, not listed. A hardcoded list silently under-counts the moment a
# new run directory appears, and two already have: runs_superseded_singleshot
# and runs_superseded_singleshot_turncap hold arms that were paid for and then
# replaced. That money was spent and must stay on the meter, so the ceiling is
# computed from every directory rather than from a list someone has to remember
# to update. Analysis still reads only `runs/`, so superseded data cannot leak
# into a result.
ALL_RUN_DIRS = sorted(
    d for d in (REPO_ROOT / "output" / "intermediate").glob("runs*") if d.is_dir()
)
SUNK_PATH = REPO_ROOT / "output" / "intermediate" / "runs_sunk.json"

SEED = run08.SEED
REPEAT_CELL_NAMES = ["factorial:none:advisory:s0", "factorial:decoder_confidence:advisory:s0"]
N_REPETITIONS = 3


def spent_so_far() -> float:
    """See 18_recalibrated_run.py's spent_so_far() -- identical by design."""
    total = json.loads(SUNK_PATH.read_text())["sunk_usd"] if SUNK_PATH.exists() else 0.0
    for runs_dir in ALL_RUN_DIRS:
        if not runs_dir.exists():
            continue
        for f in runs_dir.glob("*.parquet"):
            if f.name.startswith("._"):
                continue
            try:
                total += float(pd.read_parquet(f, columns=["cost_usd"])["cost_usd"].sum())
            except Exception as exc:
                # Never skip a checkpoint silently. This function's only job is
                # enforcing a human-set $100 ceiling, so an unreadable file must
                # stop the run, not quietly lower the meter and let a launch
                # proceed that would breach the ceiling once the file is
                # readable again. Checkpoints are written write-then-rename with
                # a `.parquet.tmp` name that this glob does not match, so a
                # partially written file is not a state a reader can observe;
                # anything unreadable here is a real defect.
                raise SystemExit(
                    f"cannot read checkpoint {f} while computing spend: "
                    f"{type(exc).__name__}: {exc}. Refusing to run on an "
                    f"undercounted budget. Inspect or remove the file."
                ) from exc
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated subset of the manifest panel")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    budget = manifest.get("budget_usd")
    if budget is None:
        print("budget_usd is null in the manifest. Refusing to run.", file=sys.stderr)
        return 1
    budget = float(budget)

    frozen = manifest["main_run"]["episode_ids"]  # the same frozen 100-episode set as the main study
    pool = build_episode_pool().set_index("episode_id")
    missing = [e for e in frozen if e not in pool.index]
    if missing:
        print(f"{len(missing)} manifest episode(s) absent from the pool; refusing to run", file=sys.stderr)
        return 1
    episodes = [dict(pool.loc[e], episode_id=e) for e in frozen]

    all_cells = {c.name: c for c in enumerate_cells()}
    missing_cells = [n for n in REPEAT_CELL_NAMES if n not in all_cells]
    if missing_cells:
        print(f"cell(s) not found in enumerate_cells(): {missing_cells}", file=sys.stderr)
        return 1
    repeat_cells = [all_cells[n] for n in REPEAT_CELL_NAMES]

    panel = [m["slug"] for m in manifest["model_panel"]["models"]]
    per_ep_projection = {m["slug"]: m["projected_usd_per_episode"] for m in manifest["model_panel"]["models"]}
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in panel]
        if unknown:
            print(f"not in the manifest panel: {unknown}\npanel is {panel}", file=sys.stderr)
            return 1
        panel = wanted

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    already = spent_so_far()

    # (model, cell, repetition) units. Each repetition of a (model, cell) is
    # its OWN checkpoint file -- an interrupted repetition 1 never touches
    # repetition 0's already-paid-for data, and a repair pass only re-runs
    # the episodes that actually failed within one repetition.
    units = [(m, c, k) for m in panel for c in repeat_cells for k in range(N_REPETITIONS)]
    pending = []
    for m, c, k in units:
        f = RUNS_DIR / f"{run08._safe(m)}__{run08._safe(c.name)}__rep{k}.parquet"
        if not f.exists():
            pending.append((m, c, k, None))
            continue
        prev = pd.read_parquet(f)
        done = set(prev.loc[prev["error"].isna(), "episode_id"])
        owed = [e["episode_id"] for e in episodes if e["episode_id"] not in done]
        if owed:
            pending.append((m, c, k, set(owed)))

    n_repair = sum(1 for _, _, _, o in pending if o is not None)
    projected = sum(per_ep_projection.get(m, 0.0) * (len(o) if o else len(episodes))
                    for m, _, _, o in pending)
    print(f"arms:       {REPEAT_CELL_NAMES}")
    print(f"repetitions: {N_REPETITIONS} (labels only, never used to alter the call -- "
          f"temperature {run08.TEMPERATURE} sampling itself supplies the independent draws)")
    print(f"panel:      {panel}")
    print(f"episodes:   {len(episodes)} (frozen 100-episode set, main_run.episode_ids)")
    print(f"units:      {len(pending)} pending of {len(units)} "
          f"({len(units) - len(pending)} complete, {n_repair} needing repair)")
    print(f"budget:     ${budget:.2f} GLOBAL ceiling, ${already:.4f} already spent (all "
          f"checkpoints, every run directory), ~${projected:.2f} projected for this run")
    if already + projected > budget:
        print("PROJECTED SPEND EXCEEDS THE GLOBAL CEILING. Refusing to start.", file=sys.stderr)
        return 1
    if args.dry_run:
        print("(dry run -- no calls made)")
        return 0

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    endpoints = {}
    for m in panel:
        ep = resolve_endpoint(m, key)
        override = run08.MODEL_TAG_OVERRIDE.get(m)
        if override and override != ep["tag"]:
            match = next((c for c in ep["candidates"] if c.get("tag") == override), None)
            if match is None:
                print(f"override tag {override!r} not offered for {m}", file=sys.stderr)
                return 1
            ep = {"provider_name": match["provider_name"], "tag": match["tag"],
                  "quantization": match.get("quantization"), "context_length": match.get("context_length"),
                  "supports_tools": True, "candidates": ep["candidates"]}
            print(f"  pinned {m:<30} tag={ep['tag']!r}  (OVERRIDE, resolver chose another)")
        else:
            print(f"  pinned {m:<30} tag={ep['tag']!r}")
        endpoints[m] = ep

    limiters = {m: run08.RateLimiter(run08.MODEL_RPM.get(m)) for m in panel}
    for m, rpm in ((m, run08.MODEL_RPM.get(m)) for m in panel):
        if rpm:
            print(f"  paced  {m:<30} {rpm} requests/min (account cap)")

    t0 = time.time()
    spend = already
    for n, (model, cell, rep, owed) in enumerate(pending, 1):
        out = RUNS_DIR / f"{run08._safe(model)}__{run08._safe(cell.name)}__rep{rep}.parquet"
        if spend >= budget:
            print(f"\nBUDGET CEILING ${budget:.2f} REACHED at ${spend:.4f}. Stopping cleanly; "
                  f"{len(pending) - n + 1} unit(s) unrun.", file=sys.stderr)
            return 2
        conc = run08.MODEL_CONCURRENCY.get(model, args.concurrency)
        rows, cost = run08.run_cell(cell, model, endpoints.get(model), episodes, key, SEED,
                                    concurrency=conc, only=owed, limiter=limiters.get(model))
        for r in rows:
            r["repetition"] = rep
        df = pd.DataFrame(rows)
        if owed is not None and out.exists():
            prev = pd.read_parquet(out)
            df = pd.concat([prev[prev["error"].isna()], df], ignore_index=True)
            order = {e["episode_id"]: i for i, e in enumerate(episodes)}
            df = df.sort_values("episode_id", key=lambda s: s.map(order)).reset_index(drop=True)
        tmp = out.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        spend += cost
        ok = df["error"].isna()
        cov = df.loc[ok, "covered"]
        print(f"[{n:>3}/{len(pending)}] {model:<28} {cell.name:<40} rep{rep} "
              f"cov={cov.mean() if len(cov) else float('nan'):.2f} "
              f"faith={df.loc[ok, 'faithful'].mean() if ok.any() else float('nan'):.2f} "
              f"FAILED={int((~ok).sum())} ${cost:.4f} (cum ${spend:.4f})")

    print(f"\nDONE. {len(pending)} unit(s) in {(time.time() - t0) / 60:.1f} min. "
          f"Total measured GLOBAL spend ${spend:.4f} of ${budget:.2f}.")
    print(f"Checkpoints: {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
