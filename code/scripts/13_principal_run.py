"""TASK 12: the principal confirmatory run, on the FULL 1,065-episode
primary-eligible pool, restricted to the six PRINCIPAL arms (Ruling 22's
GO after the human authorized the paid runs).

WHY THE FULL POOL, NOT A BALANCED SAMPLE. An earlier version of this task
built a 50:50 sample of 363 error-bearing and 363 clean episodes. That
reinstates exactly the enrichment the move to an end-to-end population was
meant to remove: a deployed system experiences coverage across every
attempt at the pool's NATURAL error prevalence (363/1,065, about a third),
and only running the full pool lets the absolute risks be reported as
deployment estimates rather than sample-composition artifacts.

WHY ONE SCAFFOLD, DECLARED BEFORE ANY CALL. Scaffold is a nuisance factor;
running only s0 cuts the confirmatory run's cost by two thirds. Selecting
s0 after seeing the exploratory 100-episode factorial would be post-hoc
selection, and a reviewer is entitled to say so. `--declare` writes the
choice -- and the frozen episode set -- into run_manifest.json and exits
WITHOUT making any request, so the human/operator can commit it. A commit
cannot record its own hash, so `declared_at_commit` is filled by a second,
immediately-following commit via `--fill-commit-sha`. Both commits land in
git history before any request is issued, which is what makes the
declaration checkable rather than merely asserted. The actual run REFUSES
to start unless `principal_run.declared_at_commit` already resolves to a
real commit -- this is enforced in code, not just in the operator's
discipline.

THE SIX ARMS. `factorial:none:advisory:s0`, `factorial:decoder_confidence:
advisory:s0`, `factorial:decoder_confidence:enforced:s0`, `nonllm_gate`,
`random_gate`, `oracle`. Five models run every PAID arm (the three
factorial cells + `oracle` -- FOUR, not six: `nonllm_gate`/`random_gate`
issue zero API requests by construction, per `nag.agent.
run_episode_for_cell`'s non-LLM path, and must never be counted in the
cost projection).

REUSE, NOT REIMPLEMENTATION. `08_run.py` already contains every hard part
(PinnedClient pinned by routable tag, the per-model RateLimiter/MODEL_RPM
18 rpm pacing, MODEL_TAG_OVERRIDE, `run_cell`'s per-episode concurrency and
row shape, and the resumable self-repairing checkpoint pattern) and is
battle-tested against this exact API. It is loaded here as a sibling
module (its filename starts with a digit, so `import 08_run` is not valid
Python) rather than copied, so a fix to `08_run.py` benefits both runners.
`main()` below mirrors `08_run.py`'s own main() rather than calling it,
because that function is hardwired to its own manifest key (`main_run`)
and checkpoint directory (`runs/`); this script targets `principal_run`
and a SEPARATE checkpoint directory, `runs_principal/`, so the original
100-episode exploratory run's checkpoints are never touched. The one
genuinely new piece is `spent_so_far()` below, which sums BOTH checkpoint
directories (plus the sunk-cost ledger): the $100 ceiling in
run_manifest.json is one global number, not one per script, so a budget
check that only saw this script's own checkpoints could be fooled by
spend the exploratory run already committed.

Run (two-step, per the module docstring above):

  # 1) declare -- no cost, no API calls
  PYTHONPATH=code uv run python3 code/scripts/13_principal_run.py --declare
  git add output/tables/run_manifest.json
  git commit -m "manifest: declare Task 12 scaffold + frozen episode set"

  # 2) record that commit's own SHA -- still no cost, no API calls
  PYTHONPATH=code uv run python3 code/scripts/13_principal_run.py \\
      --fill-commit-sha $(git rev-parse HEAD)
  git add output/tables/run_manifest.json
  git commit -m "manifest: record Task 12 declaration commit SHA"

  # 3) the actual paid run -- one process per rate-capped model is fine,
  #    the caps are per-model (see MODEL_RPM in 08_run.py)
  OPENROUTER_API_KEY=... UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/13_principal_run.py \\
      --models openai/gpt-5.6-luna
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

from nag.design import build_episode_pool, enumerate_cells, episode_set_digest  # noqa: E402
from nag.openrouter import resolve_endpoint  # noqa: E402


def _load_sibling(name: str, filename: str):
    """Import a same-directory script whose filename is not a valid module
    identifier (`08_run.py` cannot be `import`ed directly)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run08 = _load_sibling("_nag_run08", "08_run.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
RUNS_DIR = REPO_ROOT / "output" / "intermediate" / "runs_principal"
EXPLORATORY_RUNS_DIR = REPO_ROOT / "output" / "intermediate" / "runs"
SUNK_PATH = REPO_ROOT / "output" / "intermediate" / "runs_sunk.json"

PRINCIPAL_CELL_NAMES = [
    "factorial:none:advisory:s0",
    "factorial:decoder_confidence:advisory:s0",
    "factorial:decoder_confidence:enforced:s0",
    "nonllm_gate",
    "random_gate",
    "oracle",
]
SCAFFOLD = "s0"
EARLIER_SESSION = "earlier_session"
SEED = run08.SEED  # same constant as the exploratory run; determinism only requires
                    # this be FIXED and recorded, not distinct across independent studies


# --- spend, spanning BOTH checkpoint directories -------------------------

def spent_so_far() -> float:
    """Cumulative measured spend across the ENTIRE study, not just this
    script's own checkpoints: sunk cost (rate-limit storms, discarded) plus
    every checkpoint in the exploratory 100-episode run AND this principal
    run. The $100 ceiling in run_manifest.json is one global number; a
    meter that only saw `runs_principal/` could resume this script and
    think it had the whole ceiling to itself even though the exploratory
    run already spent ~$35 of it.

    Excludes macOS AppleDouble sidecar files (`._*.parquet`), which are
    real files matching the glob but are not parquet.
    """
    total = json.loads(SUNK_PATH.read_text())["sunk_usd"] if SUNK_PATH.exists() else 0.0
    # EVERY run directory, not just this script's two. The $100 ceiling is one
    # global number, and the four review-response runners write to four more
    # directories. Scanning only these two made the meter blind to them, so a
    # resume of this script after those tasks had run would have measured a
    # ceiling that was already largely spent. Same defect, same fix, as
    # 08_run.py and the four downstream runners.
    for runs_dir in sorted(EXPLORATORY_RUNS_DIR.parent.glob("runs*")):
        if not runs_dir.is_dir():
            continue
        for f in runs_dir.glob("*.parquet"):
            if f.name.startswith("._"):
                continue
            try:
                total += float(pd.read_parquet(f, columns=["cost_usd"])["cost_usd"].sum())
            except Exception as exc:
                raise SystemExit(
                    f"cannot read checkpoint {f} while computing spend: "
                    f"{type(exc).__name__}: {exc}. Refusing to run on an "
                    f"undercounted budget."
                ) from exc
    return total


# --- the frozen episode set + scaffold declaration ------------------------

def build_principal_episode_set() -> tuple[pd.DataFrame, dict]:
    """The full primary-eligible pool: `build_episode_pool()` minus the 19
    episodes whose `fit_match == 'earlier_session'` (Ruling 23's
    pre-specified primary-analysis exclusion). Sorted by episode_id for a
    deterministic, order-independent frozen list.
    """
    pool = build_episode_pool()
    elig = pool[pool["fit_match"] != EARLIER_SESSION].copy()
    elig = elig.sort_values("episode_id").reset_index(drop=True)
    tiers = elig["tier"].value_counts().sort_index()
    summary = dict(
        n_total=int(len(elig)),
        n_error_bearing=int(elig["err"].sum()),
        n_clean=int((~elig["err"]).sum()),
        natural_error_prevalence=float(elig["err"].mean()),
        n_participants=int(elig["participant_id"].nunique()),
        tier_counts={str(k): int(v) for k, v in tiers.items()},
        n_pool_total=int(len(pool)),
        excluded_earlier_session=int((pool["fit_match"] == EARLIER_SESSION).sum()),
        fit_match_counts={str(k): int(v) for k, v in elig["fit_match"].value_counts().items()},
        selection_rule=(
            "nag.design.build_episode_pool() minus episodes whose fit_match == "
            "'earlier_session' (Ruling 23's pre-specified primary-analysis exclusion)"
        ),
        episode_set_digest=episode_set_digest(elig["episode_id"]),
        episode_ids=elig["episode_id"].tolist(),
    )
    return elig, summary


def declare() -> dict:
    """Write `principal_run` into run_manifest.json. Makes NO API calls and
    spends NOTHING. Refuses to overwrite an already-declared block (i.e. one
    whose `declared_at_commit` is already set) -- redeclaring after the
    scaffold choice is committed would let it be revisited after seeing
    results, exactly what this two-step process exists to prevent.
    """
    manifest = json.loads(MANIFEST.read_text())
    existing = manifest.get("principal_run")
    if existing and existing.get("declared_at_commit"):
        raise SystemExit(
            "principal_run.declared_at_commit is already set "
            f"({existing['declared_at_commit']!r}). Refusing to re-declare. If this is "
            "genuinely a redo, remove the block from run_manifest.json by hand first."
        )

    _, summary = build_principal_episode_set()
    all_cells = {c.name: c for c in enumerate_cells()}
    missing = [n for n in PRINCIPAL_CELL_NAMES if n not in all_cells]
    if missing:
        raise SystemExit(f"principal cell(s) not found in enumerate_cells(): {missing}")
    paid = [n for n in PRINCIPAL_CELL_NAMES if all_cells[n].uses_llm]
    free = [n for n in PRINCIPAL_CELL_NAMES if not all_cells[n].uses_llm]

    panel_models = manifest["model_panel"]["models"]
    per_ep_total = sum(m["projected_usd_per_episode"] for m in panel_models)
    projected = per_ep_total * len(paid) * summary["n_total"]
    spent_before = spent_so_far()

    manifest["principal_run"] = {
        "scaffold": SCAFFOLD,
        "rationale": (
            "s0 is the canonical frozen scaffold, declared before this run and chosen "
            "because it is the first-listed rendering in the frozen prompt bank, not "
            "because of its performance in the 100-episode exploratory factorial."
        ),
        "declared_at_commit": None,
        "declared_at_commit_note": (
            "A commit cannot record its own hash. This field is null in the commit that "
            "adds this block and is set to that commit's SHA by the very next commit "
            "(--fill-commit-sha) -- both land in git history before any request is "
            "issued, which is what makes the declaration checkable rather than merely "
            "asserted."
        ),
        "cells": list(PRINCIPAL_CELL_NAMES),
        "paid_cells": paid,
        "free_cells": free,
        "episode_set": summary,
        "budget": {
            "manifest_budget_usd": manifest.get("budget_usd"),
            "spent_before_this_task_usd": round(spent_before, 4),
            "projected_usd_this_task": round(projected, 2),
            "projected_usd_per_episode_all_five_models": round(per_ep_total, 6),
            "basis": (
                "sum of model_panel.models[].projected_usd_per_episode over the 5-model "
                "panel, times 4 paid cells (oracle + 3 factorial-base cells; nonllm_gate "
                "and random_gate issue zero requests by construction and are excluded), "
                f"times {summary['n_total']} primary-eligible episodes"
            ),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["principal_run"]


def fill_commit_sha(sha: str) -> None:
    """Record the declaring commit's own SHA. Called AFTER `git commit` on
    the file `declare()` just wrote, so this is itself a second, separate
    commit -- still strictly before any request is issued."""
    manifest = json.loads(MANIFEST.read_text())
    if "principal_run" not in manifest:
        raise SystemExit("no principal_run block in the manifest -- run --declare first")
    manifest["principal_run"]["declared_at_commit"] = sha
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


# --- the actual run ---------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--declare", action="store_true",
                    help="write principal_run into run_manifest.json and exit (no cost, no calls)")
    ap.add_argument("--fill-commit-sha", metavar="SHA", default=None,
                    help="record the declaring commit's SHA and exit (no cost, no calls)")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of the manifest panel (default: all 5)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="episodes in flight per cell, per model (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would run and what it would cost; make no calls")
    args = ap.parse_args()

    if args.declare:
        block = declare()
        print(json.dumps(block, indent=2))
        print("\n--- wrote run_manifest.json['principal_run']. COMMIT NOW, before any request: ---")
        print("  git add output/tables/run_manifest.json")
        print('  git commit -m "manifest: declare Task 12 scaffold + frozen episode set"')
        print("\nThen record that commit's own SHA with a second, tiny commit:")
        print("  python3 code/scripts/13_principal_run.py --fill-commit-sha $(git rev-parse HEAD)")
        print("  git add output/tables/run_manifest.json")
        print('  git commit -m "manifest: record Task 12 declaration commit SHA"')
        return 0

    if args.fill_commit_sha:
        fill_commit_sha(args.fill_commit_sha)
        print(f"principal_run.declared_at_commit = {args.fill_commit_sha}")
        print("COMMIT NOW (second, tiny commit), still before any request:")
        print("  git add output/tables/run_manifest.json")
        print('  git commit -m "manifest: record Task 12 declaration commit SHA"')
        return 0

    manifest = json.loads(MANIFEST.read_text())
    budget = manifest.get("budget_usd")
    if budget is None:
        print("budget_usd is null in the manifest. Only a human sets it. Refusing to run.",
              file=sys.stderr)
        return 1
    budget = float(budget)

    pr = manifest.get("principal_run")
    if not pr or not pr.get("declared_at_commit"):
        print("principal_run.declared_at_commit is missing or null. The scaffold + episode-"
              "set declaration must be committed to git BEFORE any request is issued. Run "
              "--declare, commit, then --fill-commit-sha, commit again. Refusing to run.",
              file=sys.stderr)
        return 1
    sha = pr["declared_at_commit"]
    check = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                            cwd=REPO_ROOT, capture_output=True)
    if check.returncode != 0:
        print(f"declared_at_commit {sha!r} does not resolve to a real commit in this repo. "
              f"Refusing to run.", file=sys.stderr)
        return 1

    frozen = pr["episode_set"]["episode_ids"]
    if episode_set_digest(frozen) != pr["episode_set"]["episode_set_digest"]:
        print("principal_run.episode_set.episode_ids does not match its own recorded digest "
              "-- the manifest was edited after declaration. Refusing to run.", file=sys.stderr)
        return 1

    panel = [m["slug"] for m in manifest["model_panel"]["models"]]
    per_ep_projection = {m["slug"]: m["projected_usd_per_episode"]
                         for m in manifest["model_panel"]["models"]}
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in panel]
        if unknown:
            print(f"not in the manifest panel: {unknown}\npanel is {panel}", file=sys.stderr)
            return 1
        panel = wanted

    all_cells = {c.name: c for c in enumerate_cells()}
    missing_cells = [n for n in PRINCIPAL_CELL_NAMES if n not in all_cells]
    if missing_cells:
        print(f"principal cell(s) not found in enumerate_cells(): {missing_cells}", file=sys.stderr)
        return 1
    principal_cells = [all_cells[n] for n in PRINCIPAL_CELL_NAMES]
    llm_cells = [c for c in principal_cells if c.uses_llm]          # 4: oracle + 3 factorial
    non_llm_cells = [c for c in principal_cells if not c.uses_llm]  # 2: nonllm_gate, random_gate

    pool = build_episode_pool().set_index("episode_id")
    missing_eps = [e for e in frozen if e not in pool.index]
    if missing_eps:
        print(f"{len(missing_eps)} manifest episode(s) absent from the pool; refusing to run",
              file=sys.stderr)
        return 1
    episodes = [dict(pool.loc[e], episode_id=e) for e in frozen]

    # Per-model episode subsets. `anthropic/claude-sonnet-5` measured 1.59x its
    # projected cost and is 81% of the remaining bill, so at the full pool the
    # approved plan did not fit the $100 ceiling. The human's decision was to
    # pay the shortfall in PRECISION on that model rather than by deleting an
    # experiment or dropping it from the panel. `freeze_sonnet_subset.py` draws
    # and freezes its 500 ids; they are read here, never re-derived, and never
    # taken as the first N of the frozen order, which would not preserve error
    # prevalence or participant composition.
    subsets = pr.get("model_episode_subsets", {})
    episodes_by_model = {}
    for m in panel:
        spec = subsets.get(m)
        if spec is None:
            episodes_by_model[m] = episodes
            continue
        ids = spec["episode_ids"]
        unknown = [e for e in ids if e not in pool.index]
        if unknown:
            print(f"{len(unknown)} subset episode(s) for {m} absent from the pool; refusing",
                  file=sys.stderr)
            return 1
        if not set(ids) <= set(frozen):
            print(f"the {m} subset is not contained in the frozen episode set; refusing",
                  file=sys.stderr)
            return 1
        episodes_by_model[m] = [dict(pool.loc[e], episode_id=e) for e in ids]
        print(f"subset:     {m} runs {len(ids)} of {len(frozen)} episodes "
              f"(seed {spec['seed']}, error prevalence {spec['subset_error_prevalence']:.4f} "
              f"vs pool {spec['source_error_prevalence']:.4f})")
    episodes_by_model[run08.NO_MODEL] = episodes

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    already = spent_so_far()

    # non-LLM units first (free, model-independent): a budget abort never
    # leaves the deterministic baselines missing from an otherwise-usable
    # partial run. Mirrors 08_run.py's `units` construction exactly.
    units = [(run08.NO_MODEL, c) for c in non_llm_cells] + \
            [(m, c) for m in panel for c in llm_cells]

    pending = []
    for m, c in units:
        f = RUNS_DIR / f"{run08._safe(m)}__{run08._safe(c.name)}.parquet"
        if not f.exists():
            pending.append((m, c, None))
            continue
        prev = pd.read_parquet(f)
        done = set(prev.loc[prev["error"].isna(), "episode_id"])
        owed = [e["episode_id"] for e in episodes_by_model.get(m, episodes)
                if e["episode_id"] not in done]
        if owed:
            pending.append((m, c, set(owed)))

    n_repair = sum(1 for _, _, o in pending if o is not None)
    projected = sum(per_ep_projection.get(m, 0.0)
                    * (len(o) if o else len(episodes_by_model.get(m, episodes)))
                    for m, _, o in pending)
    print(f"scaffold:   {pr['scaffold']!r}  (declared at commit {sha})")
    print(f"panel:      {panel}")
    print(f"cells:      {[c.name for c in principal_cells]}")
    print(f"paid cells: {[c.name for c in llm_cells]}")
    print(f"episodes:   {len(episodes)} (frozen primary-eligible pool, principal_run.episode_set)")
    print(f"units:      {len(pending)} pending of {len(units)} "
          f"({len(units) - len(pending)} complete, {n_repair} needing repair)")
    print(f"budget:     ${budget:.2f} GLOBAL ceiling, ${already:.4f} already spent (all "
          f"checkpoints, both directories), ~${projected:.2f} projected for this run")
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
                  "quantization": match.get("quantization"),
                  "context_length": match.get("context_length"),
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
    for n, (model, cell, owed) in enumerate(pending, 1):
        out = RUNS_DIR / f"{run08._safe(model)}__{run08._safe(cell.name)}.parquet"
        if spend >= budget:
            print(f"\nBUDGET CEILING ${budget:.2f} REACHED at ${spend:.4f}. Stopping cleanly; "
                  f"{len(pending) - n + 1} unit(s) unrun. Re-run to resume after raising it.",
                  file=sys.stderr)
            return 2
        conc = run08.MODEL_CONCURRENCY.get(model, args.concurrency)
        model_episodes = episodes_by_model.get(model, episodes)
        rows, cost = run08.run_cell(cell, model, endpoints.get(model), model_episodes, key, SEED,
                                    concurrency=conc, only=owed, limiter=limiters.get(model))
        df = pd.DataFrame(rows)
        if owed is not None and out.exists():
            prev = pd.read_parquet(out)
            df = pd.concat([prev[prev["error"].isna()], df], ignore_index=True)
            order = {e["episode_id"]: i for i, e in enumerate(model_episodes)}
            df = df.sort_values("episode_id", key=lambda s: s.map(order)).reset_index(drop=True)
        tmp = out.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        spend += cost
        ok = df["error"].isna()
        cov = df.loc[ok, "covered"]
        print(f"[{n:>4}/{len(pending)}] {model:<28} {cell.name:<40} "
              f"cov={cov.mean() if len(cov) else float('nan'):.2f} "
              f"faith={df.loc[ok, 'faithful'].mean() if ok.any() else float('nan'):.2f} "
              f"FAILED={int((~ok).sum())} ${cost:.4f} (cum ${spend:.4f})")

    print(f"\nDONE. {len(pending)} unit(s) in {(time.time() - t0) / 60:.1f} min. "
          f"Total measured GLOBAL spend ${spend:.4f} of ${budget:.2f}.")
    print(f"Checkpoints: {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
