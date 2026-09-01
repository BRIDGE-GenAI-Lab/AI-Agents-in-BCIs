"""TASK 13: the confirmation-tool experiment (review item R2-3), on the SAME
frozen 100-episode set the main study used, five models.

WHY THIS EXISTS. The main study found that instructing a model to be careful
(`caution:w1`) with no way to actually confirm anything produces a 42.9%
parse-failure rate: the model tries to ask, has no tool for it, and the
attempt fails to parse as a valid call. That result is a PROTOCOL MISMATCH,
not evidence that asking-to-confirm is useless -- an instruction to confirm,
given an interface that affords no confirmation, is not a fair test of
confirmation. This script gives the model an actual confirmation channel and
asks whether the instruction becomes operationally effective once the
affordance exists.

THREE CONDITIONS, same episodes, same models, same MAX_TURNS=6:

  c1  no caution wording, no confirmation tool     (`factorial:none:advisory:s0`,
                                                      run via the SAME
                                                      `nag.agent.run_episode_for_cell`
                                                      path every principal-run cell uses)
  c2  `caution:w1` wording, no confirmation tool    (`caution:w1`, same path --
                                                      reproduces the 42.9%
                                                      parse-failure finding on
                                                      THIS script's own fresh
                                                      pinned endpoints, so it is
                                                      directly comparable to c3
                                                      rather than borrowed from a
                                                      different run)
  c3  `caution:w1` wording + the confirmation tool  (a NEW synthetic cell,
                                                      `caution:w1+confirmation`;
                                                      needs its own loop -- see below)

c1 and c2 are exactly the cells the main study already defines
(`nag.design.enumerate_cells()`), run through the completely unmodified
`nag.agent.run_episode_for_cell` / `nag.tools.Environment` / `nag.tools.
TOOL_SCHEMAS` -- reused via `08_run.py`'s `run_cell`, precisely as
`13_principal_run.py` reuses it. They are re-run here (not read back from an
existing checkpoint) so all three conditions in this comparison share the
same endpoint pin, the same temperature, and the same moment in time; mixing
a fresh condition against an old checkpoint's provenance would risk a
confound the design does not need to accept.

c3 is genuinely new: it needs `nag.tools.tool_schemas(confirmation=True)` as
the tool surface AND an Environment that can answer `request_confirmation`,
neither of which `nag.agent.run_episode` supports (it hardcodes the 4-tool
schema and `nag.tools.Environment`, which has no branch for that tool name at
all -- see its own comment: the answering logic "belongs to Task 13's
runner, not to this schema module"). `ConfirmationEnvironment` below
subclasses `nag.tools.Environment`, adding exactly that one branch;
`run_confirmation_episode` mirrors `nag.agent.run_episode`'s loop line for
line, swapped to use it. Nothing about the termination rule, the tool-call
extraction, or the faithfulness definition differs from the rest of the
study.

THE ORACLE, AND WHY IT MUST NEVER BE READ AS A DEPLOYMENT NUMBER.
`request_confirmation` is answered by comparing the agent's candidate
`action` argument against the action entailed by the SOURCE (true) string --
`nag.taxonomy.entail(episode["true_string"]).name` -- never the decoded
string the model itself already saw, and never anything a real interface
could compute, because no real user-confirmation channel has access to
ground truth with zero noise. It therefore bounds the benefit of a
confirmation affordance from ABOVE. c3 changes two things relative to c2 at
once -- an executable confirmation pathway, AND perfect ground-truth feedback
through it -- and this design cannot separate them. A positive result
supports only: under IDEALIZED confirmation, providing an executable
confirmation pathway resolves the protocol mismatch produced when a model is
told to confirm and given no means of doing so. It does NOT establish that
the instruction was never the problem. Every place c3's results are reported
must carry this label; it is not a caveat to bury in a footnote.

REPORTING (left to the downstream analysis step, not this runner): parse
failure, coverage, unfaithful execution, and `n_confirmation_calls` per
condition per model. The checkpoints below carry every column needed for
that; this script's job stops at producing them.

BUDGET. `spent_so_far()` is textually identical to the other three review-
response runner scripts' (`15_repeat_run.py`, `18_recalibrated_run.py`,
`19_naturalistic_run.py`) -- see that function's docstring.

Run:
  UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/14_confirmation_run.py --dry-run

  OPENROUTER_API_KEY=... UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/14_confirmation_run.py

  # after it finishes:
  git add code/nag/tools.py code/scripts/14_confirmation_run.py tests/test_tools.py output/intermediate/runs_confirmation/
  git commit -m "feat: confirmation-tool experiment, instruction with and without affordance (review item R2-3)"
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

from nag.agent import MAX_TURNS, EpisodeRecord, _as_action  # noqa: E402
from nag.design import Cell, build_episode_pool, enumerate_cells  # noqa: E402
from nag.openrouter import ParseFailure, extract_tool_calls, resolve_endpoint  # noqa: E402
from nag.taxonomy import entail  # noqa: E402
from nag.tools import TERMINAL_TOOLS, Environment, tool_schemas  # noqa: E402


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run08 = _load_sibling("_nag_run08", "08_run.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
RUNS_DIR = REPO_ROOT / "output" / "intermediate" / "runs_confirmation"

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
CAUTION_WORDING_INDEX = 1  # caution:w1 -- the wording that produced the 42.9% parse-failure finding
C3_CELL_NAME = "caution:w1+confirmation"

CONDITIONS = [
    ("c1_none_no_tool", "factorial:none:advisory:s0"),
    ("c2_caution_w1_no_tool", "caution:w1"),
    ("c3_caution_w1_with_tool", C3_CELL_NAME),
]


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


# --- condition 3: the confirmation-tool environment and its agent loop -----

class ConfirmationEnvironment(Environment):
    """`nag.tools.Environment` plus exactly one branch: `request_confirmation`,
    answered by the SIMULATED-USER ORACLE described in this script's module
    docstring. Every other tool (`read_buffer`, `lookup_action`, `execute`,
    `abstain`) behaves identically to the unmodified Environment, via
    `super().call()` -- this class changes nothing about the codebook, the
    termination rule, or the first-terminal-call-wins precedence.
    """

    def __init__(self, decoded_string: str, true_action_name: str):
        super().__init__(decoded_string=decoded_string)
        self._true_action_name = true_action_name
        self.n_confirmation_calls = 0

    def call(self, name: str, args: dict) -> dict:
        if self.terminated:
            return {"error": "interaction already ended"}
        if name == "request_confirmation":
            self.trace.append({"tool": name, "args": args})
            self.n_confirmation_calls += 1
            confirmed = args.get("action") == self._true_action_name
            # Non-terminal (see TERMINAL_TOOLS): self.terminated is left
            # untouched, so the model may still call execute or abstain.
            return {"confirmed": bool(confirmed)}
        return super().call(name, args)


def run_confirmation_episode(cell, episode: dict, confidence: float, client, system: str) -> EpisodeRecord:
    """`nag.agent.run_episode`'s loop, line for line, with
    `nag.tools.Environment` swapped for `ConfirmationEnvironment` and the
    tool surface widened to `tool_schemas(confirmation=True)`. Everything
    else -- MAX_TURNS, tool-call extraction, first-terminal-call-wins,
    faithfulness scored against the TRUE string's entailed action -- is
    identical to the rest of the study.
    """
    true_action = entail(episode["true_string"])
    if true_action is None:
        raise ValueError(f"episode {episode['episode_id']!r}: true_string does not entail an action")
    env = ConfirmationEnvironment(decoded_string=episode["decoded_string"], true_action_name=true_action.name)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Act on the user's input channel."}]
    tools = tool_schemas(confirmation=True)
    served, turns, parse_failed, n_terminal_calls = None, 0, False, 0

    while not env.terminated and turns < MAX_TURNS:
        turns += 1
        resp, served = client.chat(messages=messages, tools=tools)
        try:
            calls = extract_tool_calls(resp)
        except ParseFailure:
            parse_failed = True
            break
        messages.append(resp["choices"][0]["message"])
        for c in calls:
            if c["name"] in TERMINAL_TOOLS:
                n_terminal_calls += 1
            result = env.call(c["name"], c["arguments"])
            messages.append({"role": "tool", "tool_call_id": c.get("id"),
                              "name": c["name"], "content": json.dumps(result)})

    got = _as_action(env.executed)
    covered = env.executed is not None
    faithful = bool(covered and got == (true_action.name, true_action.args))
    rec = EpisodeRecord(
        episode_id=episode["episode_id"], cell=cell, executed=env.executed,
        faithful=faithful, covered=covered, n_turns=turns, parse_failed=parse_failed,
        served_provider=served, participant_id=episode.get("participant_id"),
        study=episode.get("study"), n_terminal_calls=n_terminal_calls, confidence=confidence,
    )
    rec.n_confirmation_calls = env.n_confirmation_calls  # dynamic attr; read back by run_confirmation_cell
    return rec


def run_confirmation_cell(cell, model, endpoint, episodes, key, seed, concurrency=1, only=None, limiter=None):
    """`08_run.py`'s `run_cell`, adapted to call `run_confirmation_episode`
    instead of `nag.agent.run_episode_for_cell` and to carry
    `n_confirmation_calls` through to the row. Same per-episode-own-client
    concurrency model, same input-order-preserving `.map`, same cost
    accounting -- see `run08.run_cell`'s docstring for why each of those
    matters.
    """
    def one(indexed_episode):
        i, episode = indexed_episode
        client = run08.PinnedClient(model, endpoint, key, limiter) if endpoint is not None else None
        conf = float(episode["confidence"])
        row = {
            # NOTE: "cell" alone disambiguates the three Task-13 conditions
            # (each maps to a distinct cell name: factorial:none:advisory:s0
            # / caution:w1 / caution:w1+confirmation). "condition" keeps ITS
            # existing meaning throughout this study -- the underlying
            # bigP3BCI recording condition (e.g. "CB"), episode["condition"]
            # -- so this checkpoint's schema matches run08.run_cell's row
            # shape exactly and c1/c2 (written by run08.run_cell) and c3
            # (written here) stay comparable under the same column names.
            "model": model, "cell": cell.name,
            "uncertainty_source": cell.uncertainty_source, "control_mechanism": cell.control_mechanism,
            "scaffold": cell.scaffold, "wording": cell.wording, "uses_llm": cell.uses_llm,
            "episode_id": episode["episode_id"], "participant_id": episode["participant_id"],
            "study": episode["study"], "session_id": episode["session_id"], "condition": episode["condition"],
            "true_string": episode["true_string"], "decoded_string": episode["decoded_string"],
            "n_errors": int(episode["n_errors"]), "err": bool(episode["err"]),
            "tier": int(episode["tier"]), "fit_match": episode["fit_match"],
            "confidence": conf,
            "provider_tag": endpoint["tag"] if endpoint else None,
            "quantization": endpoint["quantization"] if endpoint else None,
        }
        try:
            rec = run_confirmation_episode(cell=cell, episode=episode, confidence=conf,
                                           client=client, system=system_for(cell, conf))
            calls = client.records if client is not None else []
            row.update({
                "covered": bool(rec.covered), "faithful": bool(rec.faithful),
                "parse_failed": bool(rec.parse_failed), "n_turns": int(rec.n_turns),
                "executed_name": (rec.executed or {}).get("name"),
                "executed_args": json.dumps((rec.executed or {}).get("args"), sort_keys=True),
                "served_provider": rec.served_provider,
                "n_confirmation_calls": int(getattr(rec, "n_confirmation_calls", 0)),
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
                "n_confirmation_calls": None,
                "n_api_calls": len(calls), "n_retries": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "cost_usd": sum(float(c.get("cost") or 0.0) for c in calls),
                "error": f"{type(e).__name__}: {str(e)[:300]}",
            })
        return row

    items = [(i, e) for i, e in enumerate(episodes) if only is None or e["episode_id"] in only]
    if concurrency <= 1 or endpoint is None:
        rows = [one(it) for it in items]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, items))
    return rows, sum(r["cost_usd"] for r in rows)


def system_for(cell, confidence) -> str:
    from nag.prompts import build_system
    return build_system(cell, confidence=confidence)


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
    for _, name in CONDITIONS[:2]:
        if name not in all_cells:
            print(f"cell {name!r} not found in enumerate_cells()", file=sys.stderr)
            return 1
    c1_cell = all_cells[CONDITIONS[0][1]]
    c2_cell = all_cells[CONDITIONS[1][1]]
    # c3: same wording as c2 (caution:w1), synthetic name so build_system's
    # "caution:" name-prefix check still fires and appends CAUTION_WORDINGS[1].
    c3_cell = Cell(name=C3_CELL_NAME, uncertainty_source="none", control_mechanism="advisory",
                   scaffold=0, wording=CAUTION_WORDING_INDEX)
    assert c2_cell.wording == CAUTION_WORDING_INDEX

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

    units = [(m, cell, is_c3) for m in panel
             for cell, is_c3 in ((c1_cell, False), (c2_cell, False), (c3_cell, True))]
    pending = []
    for m, c, is_c3 in units:
        f = RUNS_DIR / f"{run08._safe(m)}__{run08._safe(c.name)}.parquet"
        if not f.exists():
            pending.append((m, c, is_c3, None))
            continue
        prev = pd.read_parquet(f)
        done = set(prev.loc[prev["error"].isna(), "episode_id"])
        owed = [e["episode_id"] for e in episodes if e["episode_id"] not in done]
        if owed:
            pending.append((m, c, is_c3, set(owed)))

    n_repair = sum(1 for _, _, _, o in pending if o is not None)
    projected = sum(per_ep_projection.get(m, 0.0) * (len(o) if o else len(episodes))
                    for m, _, _, o in pending)
    print(f"conditions: {[label for label, _ in CONDITIONS]}")
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
    for n, (model, cell, is_c3, owed) in enumerate(pending, 1):
        out = RUNS_DIR / f"{run08._safe(model)}__{run08._safe(cell.name)}.parquet"
        if spend >= budget:
            print(f"\nBUDGET CEILING ${budget:.2f} REACHED at ${spend:.4f}. Stopping cleanly; "
                  f"{len(pending) - n + 1} unit(s) unrun.", file=sys.stderr)
            return 2
        conc = run08.MODEL_CONCURRENCY.get(model, args.concurrency)
        if is_c3:
            rows, cost = run_confirmation_cell(cell, model, endpoints.get(model), episodes, key, SEED,
                                               concurrency=conc, only=owed, limiter=limiters.get(model))
        else:
            rows, cost = run08.run_cell(cell, model, endpoints.get(model), episodes, key, SEED,
                                        concurrency=conc, only=owed, limiter=limiters.get(model))
            for r in rows:
                r["n_confirmation_calls"] = 0  # tool not offered in c1/c2 -- explicit zero, not absent
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
        print(f"[{n:>3}/{len(pending)}] {model:<28} {cell.name:<28} "
              f"cov={cov.mean() if len(cov) else float('nan'):.2f} "
              f"faith={df.loc[ok, 'faithful'].mean() if ok.any() else float('nan'):.2f} "
              f"parse_fail={df.loc[ok, 'parse_failed'].mean() if ok.any() else float('nan'):.2f} "
              f"FAILED={int((~ok).sum())} ${cost:.4f} (cum ${spend:.4f})")

    print(f"\nDONE. {len(pending)} unit(s) in {(time.time() - t0) / 60:.1f} min. "
          f"Total measured GLOBAL spend ${spend:.4f} of ${budget:.2f}.")
    print(f"Checkpoints: {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
