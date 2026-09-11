"""THE MAIN RUN: every (model, cell, episode) triple, checkpointed and resumable.

Reads output/tables/run_manifest.json for the panel, the frozen episode set,
and the HUMAN-SET `budget_usd` ceiling. Refuses to start if that ceiling is
null -- 05_design.py leaves it null by design and only a human may set it.

Three properties this script exists to guarantee:

  RESUMABLE. One parquet checkpoint per (model, cell) under
  output/intermediate/runs/. A complete checkpoint is skipped, so an
  interrupted run costs nothing to restart and a partial panel can be
  extended later without re-running what is already paid for. The episode
  set is frozen in the manifest, so a model added next week runs the SAME
  episodes as one run today -- the design stays fully paired across
  interruptions.

  BUDGET-SAFE. Spend is measured, never modelled: every response carries
  `usage.cost` in exact USD. Cumulative spend is recomputed from ALL
  existing checkpoints at startup, so resuming does not reset the meter,
  and the run aborts the moment the next cell could carry it past the
  ceiling.

  RATE-LIMITED PER MODEL. Concurrency alone cannot respect a requests-per-
  minute cap: 8 workers each waiting on a slow response is fine, 8 workers
  each finishing fast is a burst. OpenRouter enforces a per-model RPM cap on
  new accounts -- probing this key returned
  "new-account-rpm/openai/gpt-5.6-luna: new accounts are limited to 20
  requests per minute for this model", with `limit_source:
  openrouter_new_account`. That is an ACCOUNT property, not a provider one,
  so no amount of endpoint switching or concurrency tuning evades it; the
  same probe found claude-sonnet-5 and gemini-3.7-flash unthrottled at burst
  12, and glm-5.3-flash / deepseek-v4-flash throttled only provider-side
  (retries absorb those). A token bucket paces each model just under its cap,
  which turns a 40%-failure storm into zero failures at the same throughput.

  CONCURRENT WITHIN A CELL. One episode is ~3 sequential API round trips;
  100 of them in series is ~10 minutes, and 64 units of that is 11 hours.
  Episodes inside a cell are independent, so they run on a thread pool.
  Each worker gets its OWN client instance -- a shared one would interleave
  provenance records from concurrent episodes and mis-attribute cost per
  episode, which is the number the budget meter is built on. Concurrency is
  modest by default because the burst limit is real (the live smoke tripped
  429s at concurrency 1); `chat`'s jittered backoff absorbs what is left.

  MODEL-INDEPENDENT WORK RUNS ONCE. `nonllm_gate` and `random_gate` make
  zero API calls and do not consult a model, so they are executed once
  under the sentinel model `__none__` rather than five times.

Partial runs are first-class (`--models`), because the human may want to
start with two cheap models and extend. Nothing about a partial run
differs from the corresponding slice of a full one.

Run:
  OPENROUTER_API_KEY=... UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv \\
  PYTHONPATH=code uv run python3 code/scripts/08_run.py --models deepseek/deepseek-v4-flash,openai/gpt-5.6-luna
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.agent import run_episode_for_cell  # noqa: E402
from nag.design import build_episode_pool, enumerate_cells  # noqa: E402
from nag.openrouter import chat, resolve_endpoint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
RUNS_DIR = REPO_ROOT / "output" / "intermediate" / "runs"

MAX_TOKENS, TEMPERATURE, TOP_P = 512, 0.7, 1.0
NO_MODEL = "__none__"   # sentinel for the model-independent non-LLM cells
SEED = 20260828


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# Endpoints differ by orders of magnitude in what concurrency they tolerate.
# Measured on the first long run at --concurrency 24: deepseek-v4-flash via
# DigitalOcean returned 35 errors in 3,200 episodes (1.1%), while
# gpt-5.6-luna via Azure returned 1,772 in 1,900 (93%) -- the same setting,
# a completely different endpoint. One global number cannot serve both, so
# slow endpoints get an explicit override rather than a lower global default
# that would trebles the run time of the endpoints that are fine.
# Requests per minute per model. `None` means unthrottled (provider-side 429s
# are absorbed by `chat`'s backoff). Values are measured, not guessed -- see
# the module docstring. The cap is on REQUESTS, and one episode is ~3 requests.
# Measured with a SUSTAINED 30-request probe, not a burst: a burst of 12
# cannot reveal a 20/min cap, and an earlier burst probe wrongly cleared
# claude-sonnet-5, which then failed 121 of 200 episodes. gemini-3.7-flash
# returned exactly 20/30 -- the cap, precisely. glm-5.3-flash and
# deepseek-v4-flash show no account cap at all (their 429s are provider-side
# and absorbed by `chat`'s backoff). The caps are PER MODEL, so capped models
# can run as parallel processes without stealing each other's budget.
MODEL_RPM = {
    "openai/gpt-5.6-luna": 18,          # account cap 20; leave headroom
    "anthropic/claude-sonnet-5": 18,    # account cap 20
    "google/gemini-3.7-flash": 18,      # account cap 20
}

MODEL_CONCURRENCY = {
    "openai/gpt-5.6-luna": 6,
    "anthropic/claude-sonnet-5": 6,
    "google/gemini-3.7-flash": 6,
}

# `resolve_endpoint` picks deterministically (highest context, then plain tag,
# then alphabetical by provider_name), which is the right default but has no
# way to know an endpoint's RATE limit. For gpt-5.6-luna it chose 'azure' over
# 'openai' purely on the alphabet, at identical context length -- and Azure
# rate-limited this key so hard that 1,772 of 1,900 episodes failed on 429
# after exhausting retries. An override records that operational fact
# explicitly instead of hiding it in a tie-break.
#
# Switching a model's endpoint INVALIDATES its existing rows: different
# providers serve different quantizations, and mixing them inside one model's
# data is precisely the corruption `provider` pinning exists to prevent. Every
# checkpoint for an overridden model must be deleted, never repaired.
MODEL_TAG_OVERRIDE = {
    "openai/gpt-5.6-luna": "openai",
}


class RateLimiter:
    """Token bucket over REQUESTS, shared by every worker on one model.

    Threads block here rather than discovering the cap as a 429, so the run
    spends its time waiting instead of burning retries and losing episodes.
    """

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            self._next = max(now, self._next) + self.interval
            wait = self._next - self.interval - now
        if wait > 0:
            time.sleep(wait)


def _cell_seed(name: str) -> int:
    """A STABLE per-cell seed component.

    Not `hash(name)`: Python salts str hashing per process, so `random_gate`
    would draw differently on every run and the deterministic baseline would
    not be reproducible -- the one property that baseline exists to have.
    """
    import hashlib
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


class PinnedClient:
    """Adapts nag.openrouter.chat to the run_episode client contract, pinning
    by the endpoint's ROUTABLE TAG (Ruling 29), never its display name."""

    def __init__(self, model, endpoint, api_key, limiter=None):
        self.model, self._key = model, api_key
        self._limiter = limiter
        self.tag = endpoint["tag"]
        self.provider_name = endpoint["provider_name"]
        self.quantization = endpoint["quantization"]
        self.context_length = endpoint["context_length"]
        self.supports_tools = endpoint["supports_tools"]
        self.records: list[dict] = []

    def chat(self, messages, tools):
        if self._limiter is not None:
            self._limiter.acquire()
        os.environ["OPENROUTER_API_KEY"] = self._key
        return chat(model=self.model, messages=messages, tools=tools, provider=self.tag,
                    temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
                    quantization=self.quantization, context_length=self.context_length,
                    supports_tools=self.supports_tools, provider_name=self.provider_name,
                    record_hook=self.records.append)


def spent_so_far() -> float:
    """Cumulative measured spend across every checkpoint already on disk.

    Recomputed from the files rather than tracked in memory, so an
    interrupted-and-resumed run does not silently reset the budget meter --
    the single easiest way to blow through a ceiling. `runs_sunk.json` adds
    money that WAS spent on checkpoints later discarded (rate-limit storms),
    which the files themselves can no longer account for.
    """
    sunk = RUNS_DIR.parent / "runs_sunk.json"
    total = json.loads(sunk.read_text())["sunk_usd"] if sunk.exists() else 0.0
    # EVERY run directory, not just this script's own. The $100 ceiling is one
    # global number. Scanning only RUNS_DIR made this meter blind to the
    # principal run and the four review-response runners, so a re-run launched
    # from here would have measured ~$35 of a ceiling that was already most
    # spent. AppleDouble sidecars (`._*.parquet`) match the glob on this exFAT
    # volume and are not parquet.
    for runs_dir in sorted(RUNS_DIR.parent.glob("runs*")):
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


def run_cell(cell, model, endpoint, episodes, key, seed, concurrency=1, only=None,
             limiter=None):
    """Every episode for one (model, cell). Returns (rows, spend).

    Episodes run concurrently, each on its OWN `PinnedClient`, so the
    provenance records a row reads back are exactly its own. A shared client
    would interleave records from concurrent episodes and mis-attribute
    per-episode cost -- the number the budget meter is built on. Results are
    reassembled in input order, so a checkpoint's row order never depends on
    completion order and reruns stay comparable.

    `only` restricts the run to a set of episode_ids -- how a repair pass
    re-runs just the episodes that failed, at full episode indices so
    `random_gate`'s per-episode seed is unchanged by the repair.
    """
    def one(indexed_episode):
        i, episode = indexed_episode
        # Seeded per (cell, episode) so random_gate is reproducible and depends
        # neither on how many episodes ran before it nor on the order the pool
        # happened to finish them in.
        rng = np.random.default_rng([seed, _cell_seed(cell.name), i])
        client = PinnedClient(model, endpoint, key, limiter) if endpoint is not None else None
        conf = float(episode["confidence"])
        row = {
            "model": model, "cell": cell.name,
            "uncertainty_source": cell.uncertainty_source,
            "control_mechanism": cell.control_mechanism,
            "scaffold": cell.scaffold, "wording": cell.wording,
            "uses_llm": cell.uses_llm,
            "episode_id": episode["episode_id"],
            "participant_id": episode["participant_id"], "study": episode["study"],
            "session_id": episode["session_id"], "condition": episode["condition"],
            "true_string": episode["true_string"], "decoded_string": episode["decoded_string"],
            "n_errors": int(episode["n_errors"]), "err": bool(episode["err"]),
            "tier": int(episode["tier"]), "fit_match": episode["fit_match"],
            "confidence": conf,
            "provider_tag": endpoint["tag"] if endpoint else None,
            "quantization": endpoint["quantization"] if endpoint else None,
        }
        try:
            rec = run_episode_for_cell(cell=cell, episode=episode, confidence=conf,
                                       client=client, rng=rng)
            calls = client.records if client is not None else []
            row.update({
                "covered": bool(rec.covered), "faithful": bool(rec.faithful),
                "parse_failed": bool(rec.parse_failed), "n_turns": int(rec.n_turns),
                "executed_name": (rec.executed or {}).get("name"),
                "executed_args": json.dumps((rec.executed or {}).get("args"), sort_keys=True),
                "served_provider": rec.served_provider,
                "n_api_calls": len(calls),
                "n_retries": sum(int(c.get("n_retries") or 0) for c in calls),
                "prompt_tokens": sum(int(c.get("prompt_tokens") or 0) for c in calls),
                "completion_tokens": sum(int(c.get("completion_tokens") or 0) for c in calls),
                "cost_usd": sum(float(c.get("cost") or 0.0) for c in calls),
                "error": None,
            })
        except Exception as e:
            calls = client.records if client is not None else []
            row.update({
                "covered": None, "faithful": None, "parse_failed": None, "n_turns": None,
                "executed_name": None, "executed_args": None, "served_provider": None,
                "n_api_calls": len(calls), "n_retries": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "cost_usd": sum(float(c.get("cost") or 0.0) for c in calls),
                "error": f"{type(e).__name__}: {str(e)[:300]}",
            })
        return row

    items = [(i, e) for i, e in enumerate(episodes)
             if only is None or e["episode_id"] in only]
    if concurrency <= 1 or endpoint is None:
        rows = [one(it) for it in items]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, items))   # .map preserves input order
    return rows, sum(r["cost_usd"] for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of the manifest panel (default: all)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="cap episodes per cell (default: the manifest's frozen set)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="episodes in flight per cell (default 8)")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell names to run (default: all). Used to "
                         "re-run a single arm after a defect fix without paying for "
                         "the other 33.")
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

    cells = enumerate_cells()
    llm_cells = [c for c in cells if c.uses_llm]
    non_llm_cells = [c for c in cells if not c.uses_llm]

    frozen = manifest["main_run"]["episode_ids"]
    pool = build_episode_pool().set_index("episode_id")
    missing = [e for e in frozen if e not in pool.index]
    if missing:
        print(f"{len(missing)} manifest episode(s) absent from the pool; refusing to run",
              file=sys.stderr)
        return 1
    if args.episodes:
        frozen = frozen[:args.episodes]
    episodes = [dict(pool.loc[e], episode_id=e) for e in frozen]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    already = spent_so_far()

    # (model, cell) units, non-LLM first: they are free, model-independent,
    # and finishing them up front means a budget abort never leaves the
    # deterministic baselines missing from an otherwise analysable partial run.
    units = [(NO_MODEL, c) for c in non_llm_cells] + \
            [(m, c) for m in panel for c in llm_cells]

    if args.cells:
        wanted = {n.strip() for n in args.cells.split(",") if n.strip()}
        known = {c.name for _, c in units}
        unknown = wanted - known
        if unknown:
            print(f"unknown cell name(s): {sorted(unknown)}. Known: {sorted(known)}",
                  file=sys.stderr)
            return 1
        units = [(m, c) for m, c in units if c.name in wanted]
        print(f"cell filter: {sorted(wanted)} -> {len(units)} unit(s)")

    # A unit is pending if it has no checkpoint OR its checkpoint carries
    # failed rows. `todo` holds the episode_ids still owed, so a repair pass
    # pays only for what actually failed.
    pending = []
    for m, c in units:
        f = RUNS_DIR / f"{_safe(m)}__{_safe(c.name)}.parquet"
        if not f.exists():
            pending.append((m, c, None))
            continue
        prev = pd.read_parquet(f)
        done = set(prev.loc[prev["error"].isna(), "episode_id"])
        owed = [e["episode_id"] for e in episodes if e["episode_id"] not in done]
        if owed:
            pending.append((m, c, set(owed)))

    n_repair = sum(1 for _, _, o in pending if o is not None)
    projected = sum(per_ep_projection.get(m, 0.0) * (len(o) if o else len(episodes))
                    for m, _, o in pending)
    print(f"panel:     {panel}")
    print(f"episodes:  {len(episodes)} (frozen, shared by every cell and model)")
    print(f"units:     {len(pending)} pending of {len(units)} "
          f"({len(units) - len(pending)} complete, {n_repair} needing repair)")
    print(f"budget:    ${budget:.2f} ceiling, ${already:.4f} already spent, "
          f"~${projected:.2f} projected for this run")
    if already + projected > budget:
        print("PROJECTED SPEND EXCEEDS THE CEILING. Refusing to start.", file=sys.stderr)
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
        override = MODEL_TAG_OVERRIDE.get(m)
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

    limiters = {m: RateLimiter(MODEL_RPM.get(m)) for m in panel}
    for m, rpm in ((m, MODEL_RPM.get(m)) for m in panel):
        if rpm:
            print(f"  paced  {m:<30} {rpm} requests/min (account cap)")

    t0 = time.time()
    spend = already
    for n, (model, cell, owed) in enumerate(pending, 1):
        out = RUNS_DIR / f"{_safe(model)}__{_safe(cell.name)}.parquet"
        if spend >= budget:
            print(f"\nBUDGET CEILING ${budget:.2f} REACHED at ${spend:.4f}. Stopping cleanly; "
                  f"{len(pending) - n + 1} unit(s) unrun. Re-run to resume after raising it.",
                  file=sys.stderr)
            return 2
        conc = MODEL_CONCURRENCY.get(model, args.concurrency)
        rows, cost = run_cell(cell, model, endpoints.get(model), episodes, key, SEED,
                              concurrency=conc, only=owed, limiter=limiters.get(model))
        df = pd.DataFrame(rows)
        if owed is not None and out.exists():
            # Keep the previously SUCCESSFUL rows, replace the failed ones.
            prev = pd.read_parquet(out)
            df = pd.concat([prev[prev["error"].isna()], df], ignore_index=True)
            order = {e["episode_id"]: i for i, e in enumerate(episodes)}
            df = df.sort_values("episode_id", key=lambda s: s.map(order)).reset_index(drop=True)
        # Atomic: write beside the target, then rename. A direct write leaves a
        # truncated .parquet visible to any reader (and permanently, if the run
        # is killed mid-write) -- pyarrow reports it only as "Parquet magic
        # bytes not found in footer", with no hint which file is at fault.
        tmp = out.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        spend += cost
        ok = df["error"].isna()
        cov = df.loc[ok, "covered"]
        print(f"[{n:>3}/{len(pending)}] {model:<28} {cell.name:<40} "
              f"cov={cov.mean() if len(cov) else float('nan'):.2f} "
              f"faith={df.loc[ok, 'faithful'].mean() if ok.any() else float('nan'):.2f} "
              f"FAILED={int((~ok).sum())} ${cost:.4f} (cum ${spend:.4f})")

    print(f"\nDONE. {len(pending)} unit(s) in {(time.time() - t0) / 60:.1f} min. "
          f"Total measured spend ${spend:.4f} of ${budget:.2f}.")
    print(f"Checkpoints: {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
