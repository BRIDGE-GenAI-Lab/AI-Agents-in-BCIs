"""TASK 19: the recalibrated-confidence advisory arm, and the matched gate on
the SAME recalibrated score, on the FULL 1,065-episode primary-eligible pool.

STATUS: reviewer-motivated follow-up robustness experiment. This task was
designed after inspection of the initial 100-episode benchmark, in response
to an identified calibration asymmetry (Task 4: the product confidence score
has episode-level ECE 0.080, while a directly fitted episode-level isotonic
model reaches 0.049). It is frozen before any follow-up model execution here,
which is good prospective discipline, but it is NOT part of the original
pre-specified experiment, and the manuscript must say so in those words.
Freezing a design after seeing earlier results does not retroactively make it
prespecified.

WHY THIS EXISTS. The deterministic gate thresholds on ORDERING alone, so its
AURC is identical (0.6124) whether it thresholds the product score or any
monotone transform of it (rank, log, square) -- recalibration cannot help or
hurt a pure threshold. The advisory arms are NOT rank-invariant, because the
NUMERIC confidence value is rendered into their prompt (see
`nag.prompts.build_system`): a model reading "0.81" versus a model reading
the same episode's true recalibrated probability sees materially different
numbers whenever the product score is miscalibrated. The two sides of the
paper's headline comparison therefore did not receive equally good signal,
and the asymmetry favours the gate -- a fairness objection to the paper's
central claim that has to be closed, not merely disclosed. This script closes
it: it runs the SAME advisory arm the principal run already ran, with the
episode-level isotonic out-of-fold score substituted for the product score in
the prompt, on the SAME 1,065-episode pool (not a 100-episode sensitivity
slice -- an earlier version of this task did exactly that, and it would have
left the paper open to the objection that the definitive full-cohort
comparison used the miscalibrated value while the fair comparison was a small
side analysis). Recomputing the matched gate on this same recalibrated score
costs nothing further (it is a threshold sweep over a column already on
disk) and is not this script's job; it belongs to the analysis step that
consumes these checkpoints, not to the runner. Note the recalibrated score is
expected to RANK worse than the product score (AUROC 0.761 vs 0.785), so the
gate's own frontier is expected to degrade under it -- that degradation is
exactly the trade this experiment measures, not a bug to chase.

THE ONE ARM, ONE POOL. `factorial:decoder_confidence:advisory:s0`, five
models, the SAME frozen 1,065-episode set `13_principal_run.py` already
declared and committed (`run_manifest.json["principal_run"]`) -- this script
does not redeclare an episode set of its own; reusing the already-committed
one is what pins it to the definitive cohort rather than a fresh, potentially
post-hoc-tunable sample. This script refuses to run unless that declaration's
`declared_at_commit` still resolves to a real commit and its recorded episode
ids still match their own digest, exactly the guard `13_principal_run.py`
enforces on itself.

THE RECALIBRATED SCORE. `output/tables/episode_confidence_per_episode.csv`,
column `isotonic_episode` -- the PARTICIPANT-GROUPED OUT-OF-FOLD prediction
from `nag.episode_calibration` (never the in-sample fit, which would be
optimistically biased and is not what "recalibrated" is allowed to mean
here). This is the ONLY column this script may use for that purpose; it does
not refit anything itself. The pool's own `confidence` column (the product
rule) is kept on every output row as `confidence_product`, purely for
downstream traceability -- it is NEVER what is rendered into the prompt or
what the model sees; `confidence` on the output row is the value actually
shown, i.e. the recalibrated one.

REUSE, NOT REIMPLEMENTATION. Loads `08_run.py` as a sibling module exactly as
`13_principal_run.py` does, for `PinnedClient`, the per-model `RateLimiter`/
`MODEL_RPM` pacing, `MODEL_TAG_OVERRIDE`, `run_cell`'s per-episode
concurrency and resumable checkpoint pattern, and `_safe`/`NO_MODEL`/`SEED`.
The only genuinely new logic here is substituting the recalibrated confidence
into each episode dict before it reaches `run_cell` -- everything downstream
of that substitution (the agent loop, the tool surface, the checkpoint
format) is completely unaware anything changed, which is the point: this arm
differs from the principal run's `factorial:decoder_confidence:advisory:s0`
in exactly one number per episode, nothing else.

BUDGET. `spent_so_far()` below sums EVERY checkpoint directory this review-
response round has introduced (`runs/`, `runs_principal/`, `runs_recal/`,
`runs_confirmation/`, `runs_repeat/`, `runs_natural/`) plus
`runs_sunk.json`, because the $100 ceiling in `run_manifest.json` is one
GLOBAL number shared by every script that spends against it, not a private
allowance per script. This is the identical function, by construction, in
every one of the four review-response runner scripts (`14_confirmation_run.
py`, `15_repeat_run.py`, `18_recalibrated_run.py`, `19_naturalistic_run.py`)
-- keep any change to the directory list in sync across all four.

Run:
  UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/18_recalibrated_run.py --dry-run

  OPENROUTER_API_KEY=... UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/18_recalibrated_run.py

  # after it finishes:
  git add code/scripts/18_recalibrated_run.py output/intermediate/runs_recal/ output/tables/
  git commit -m "run: recalibrated advisory arm and matched recalibrated gate"
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
    identifier (see `13_principal_run.py`'s identical helper)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run08 = _load_sibling("_nag_run08", "08_run.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
RECAL_CONF_CSV = REPO_ROOT / "output" / "tables" / "episode_confidence_per_episode.csv"
RECAL_CONF_COLUMN = "isotonic_episode"

RUNS_DIR = REPO_ROOT / "output" / "intermediate" / "runs_recal"

# Every checkpoint directory any of the four review-response runner scripts
# can write to, plus the two that predate this round -- see the module
# docstring's BUDGET section. Kept textually identical across all four
# scripts on purpose.
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

RECAL_CELL_NAME = "factorial:decoder_confidence:advisory:s0"
SEED = run08.SEED


def spent_so_far() -> float:
    """Cumulative measured spend across the ENTIRE study: `runs_sunk.json`
    plus every `.parquet` checkpoint in every directory ANY review-response
    runner script (or the two pre-existing runs) can write to. The $100
    ceiling is one global number; a meter that only saw this script's own
    `runs_recal/` could resume it while blind to what the other three
    scripts -- or the principal run still writing to `runs_principal/` --
    have already spent. Excludes macOS AppleDouble sidecars (`._*.parquet`),
    which are real files matching the glob but are not parquet.
    """
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


def _verify_principal_declaration(manifest: dict) -> list[str]:
    """Refuse to proceed unless the principal run's episode-set declaration
    is real and unmodified: this script deliberately does not declare an
    episode set of its own, so its only anchor to "the same 1,065-episode
    pool Task 12 uses" is that declaration still being trustworthy.
    """
    pr = manifest.get("principal_run")
    if not pr or not pr.get("declared_at_commit"):
        print("principal_run.declared_at_commit is missing or null in run_manifest.json. "
              "This script reuses that declaration's frozen episode set rather than declaring "
              "its own; run 13_principal_run.py --declare (and commit) first. Refusing to run.",
              file=sys.stderr)
        raise SystemExit(1)
    sha = pr["declared_at_commit"]
    check = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                           cwd=REPO_ROOT, capture_output=True)
    if check.returncode != 0:
        print(f"principal_run.declared_at_commit {sha!r} does not resolve to a real commit "
              "in this repo. Refusing to run.", file=sys.stderr)
        raise SystemExit(1)
    frozen = pr["episode_set"]["episode_ids"]
    if episode_set_digest(frozen) != pr["episode_set"]["episode_set_digest"]:
        print("principal_run.episode_set.episode_ids does not match its own recorded digest -- "
              "the manifest was edited after declaration. Refusing to run.", file=sys.stderr)
        raise SystemExit(1)
    return frozen


def build_recalibrated_episodes(frozen_ids: list[str]) -> list[dict]:
    """The frozen 1,065-episode pool, with `confidence` OVERWRITTEN by the
    recalibrated out-of-fold isotonic episode score and the original product
    score kept alongside as `confidence_product` for traceability only.

    Refuses (rather than silently dropping or falling back) if any frozen
    episode id is missing from the recalibration csv, or if its recalibrated
    value is null: a silent gap here would mean some episodes quietly kept
    running on the very product score this whole task exists to replace.
    """
    if not RECAL_CONF_CSV.exists():
        print(f"{RECAL_CONF_CSV} does not exist -- run the episode-calibration step "
              "(12_episode_calibration.py) first. Refusing to run.", file=sys.stderr)
        raise SystemExit(1)
    recal = pd.read_csv(RECAL_CONF_CSV).set_index("episode_id")
    if RECAL_CONF_COLUMN not in recal.columns:
        print(f"{RECAL_CONF_CSV} has no column {RECAL_CONF_COLUMN!r}. Refusing to run.",
              file=sys.stderr)
        raise SystemExit(1)

    missing = [e for e in frozen_ids if e not in recal.index]
    if missing:
        print(f"{len(missing)} frozen episode(s) missing from {RECAL_CONF_CSV.name}, e.g. "
              f"{missing[:5]}. Refusing to run.", file=sys.stderr)
        raise SystemExit(1)
    null_conf = recal.loc[frozen_ids, RECAL_CONF_COLUMN].isna()
    if null_conf.any():
        print(f"{int(null_conf.sum())} frozen episode(s) have a null {RECAL_CONF_COLUMN} value. "
              "Refusing to run.", file=sys.stderr)
        raise SystemExit(1)

    pool = build_episode_pool().set_index("episode_id")
    missing_pool = [e for e in frozen_ids if e not in pool.index]
    if missing_pool:
        print(f"{len(missing_pool)} frozen episode(s) absent from build_episode_pool(); "
              "refusing to run", file=sys.stderr)
        raise SystemExit(1)

    episodes = []
    for eid in frozen_ids:
        row = dict(pool.loc[eid], episode_id=eid)
        row["confidence_product"] = float(row["confidence"])
        row["confidence"] = float(recal.loc[eid, RECAL_CONF_COLUMN])
        episodes.append(row)
    return episodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of the manifest panel (default: all 5)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="episodes in flight per model (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would run and what it would cost; make no calls")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    budget = manifest.get("budget_usd")
    if budget is None:
        print("budget_usd is null in the manifest. Only a human sets it. Refusing to run.",
              file=sys.stderr)
        return 1
    budget = float(budget)

    frozen = _verify_principal_declaration(manifest)
    episodes = build_recalibrated_episodes(frozen)

    all_cells = {c.name: c for c in enumerate_cells()}
    if RECAL_CELL_NAME not in all_cells:
        print(f"cell {RECAL_CELL_NAME!r} not found in enumerate_cells()", file=sys.stderr)
        return 1
    cell = all_cells[RECAL_CELL_NAME]

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

    # Per-model subsets, read from the same manifest declaration Task 12 uses so
    # a model cannot run one episode set in Task 12 and a different one here.
    # `anthropic/claude-sonnet-5` measured 1.59x its projected cost, so it runs
    # a frozen 500-episode draw (freeze_sonnet_subset.py, seed 20260830) in BOTH
    # full-pool tasks. Identical ids across both is what keeps every sonnet
    # contrast paired; a per-task draw would silently unpair them.
    subsets = manifest.get("principal_run", {}).get("model_episode_subsets", {})
    by_id = {e["episode_id"]: e for e in episodes}
    episodes_by_model = {}
    for m in panel:
        spec = subsets.get(m)
        if spec is None:
            episodes_by_model[m] = episodes
            continue
        if "19" not in [str(t) for t in spec.get("applies_to_tasks", [])]:
            episodes_by_model[m] = episodes
            continue
        ids = spec["episode_ids"]
        unknown = [e for e in ids if e not in by_id]
        if unknown:
            print(f"{len(unknown)} subset episode(s) for {m} are not in the recalibrated "
                  f"pool; refusing", file=sys.stderr)
            return 1
        episodes_by_model[m] = [by_id[e] for e in ids]
        print(f"subset:     {m} runs {len(ids)} of {len(episodes)} episodes "
              f"(seed {spec['seed']}, the SAME ids it runs in Task 12)")


    conf_product_map = {e["episode_id"]: e["confidence_product"] for e in episodes}

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    already = spent_so_far()

    units = [(m, cell) for m in panel]
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
    print(f"arm:        {RECAL_CELL_NAME!r}  (Task 19 -- recalibrated {RECAL_CONF_COLUMN} in the "
          f"prompt, replacing the product score)")
    print(f"panel:      {panel}")
    print(f"episodes:   {len(episodes)} (frozen primary-eligible pool, principal_run.episode_set)")
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
    for n, (model, c, owed) in enumerate(pending, 1):
        out = RUNS_DIR / f"{run08._safe(model)}__{run08._safe(c.name)}.parquet"
        if spend >= budget:
            print(f"\nBUDGET CEILING ${budget:.2f} REACHED at ${spend:.4f}. Stopping cleanly; "
                  f"{len(pending) - n + 1} unit(s) unrun. Re-run to resume after raising it.",
                  file=sys.stderr)
            return 2
        conc = run08.MODEL_CONCURRENCY.get(model, args.concurrency)
        model_episodes = episodes_by_model.get(model, episodes)
        rows, cost = run08.run_cell(c, model, endpoints.get(model), model_episodes, key, SEED,
                                    concurrency=conc, only=owed, limiter=limiters.get(model))
        df = pd.DataFrame(rows)
        # run08.run_cell's row shape is fixed and does not carry arbitrary
        # extra episode-dict keys through, so the original product-rule
        # confidence -- kept only for downstream traceability, NEVER shown
        # to the model -- is attached here rather than lost.
        df["confidence_product"] = df["episode_id"].map(conf_product_map)
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
        print(f"[{n:>4}/{len(pending)}] {model:<28} {c.name:<40} "
              f"cov={cov.mean() if len(cov) else float('nan'):.2f} "
              f"faith={df.loc[ok, 'faithful'].mean() if ok.any() else float('nan'):.2f} "
              f"FAILED={int((~ok).sum())} ${cost:.4f} (cum ${spend:.4f})")

    print(f"\nDONE. {len(pending)} unit(s) in {(time.time() - t0) / 60:.1f} min. "
          f"Total measured GLOBAL spend ${spend:.4f} of ${budget:.2f}.")
    print(f"Checkpoints: {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
