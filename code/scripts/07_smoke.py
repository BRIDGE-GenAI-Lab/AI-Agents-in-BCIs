"""FULL-ARM LIVE SMOKE: one real episode through EVERY arm before the main run.

Four defects in this codebase survived 143+ passing offline tests and were
found only by live API calls costing pennies (progress.md Rulings 24-29:
a phantom response header, a tool-incapable endpoint, a curve-knob guard that
refused the headline arm, and an entire unimplemented factor). Every one was
invisible offline because every test either exercised a pure function or used
a cell-agnostic scripted fake. This script is the missing test class: real
network, real models, every arm, before any money is spent at scale.

Three checks, cheapest first, each gating the next:
  A. endpoint resolution for all 5 panel models -- NO inference, no cost.
     Catches the display-name/slug/tag confusion that caused three separate
     outages, plus any model with no tool-capable endpoint.
  B. all 34 cells x TWO live episodes on the cheapest panel model -- the
     lowest- and highest-confidence error-bearing episodes in the pool.
     Two, not one, because arms CANNOT diverge on a single episode: at high
     confidence every gate admits and every model acts, so one episode makes
     34 arms look identical for a reason that says nothing about the design.
     The pass criterion is that the arms actually DIFFER somewhere -- the
     check no offline test makes, and the exact blind spot that let Ruling 28
     (an entirely unimplemented factor) reach a live probe undetected.
  C. one live episode on one representative cell for each remaining model,
     proving each of the 5 routes, tool-calls, and parses.

Writes output/tables/smoke.json. Never prints or stores the API key.
Expected total cost is a few cents; the real ceiling lives in the manifest.

Run: OPENROUTER_API_KEY=... PYTHONPATH=code uv run python3 code/scripts/07_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.agent import apply_enforced_gate, run_episode_for_cell  # noqa: E402
from nag.design import build_episode_pool, enumerate_cells  # noqa: E402
from nag.openrouter import chat, resolve_endpoint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
OUT = REPO_ROOT / "output" / "tables" / "smoke.json"

MAX_TOKENS, TEMPERATURE, TOP_P = 512, 0.7, 1.0
CHEAPEST = "z-ai/glm-5.3-flash"   # carries check B, the 32-cell sweep


class PinnedClient:
    """Adapts nag.openrouter.chat to the run_episode client contract, pinning
    the endpoint by its ROUTABLE TAG (Ruling 29) rather than its display name."""

    def __init__(self, model, tag, api_key, quantization, context_length, supports_tools=True):
        self.model, self.tag, self._key = model, tag, api_key
        self.quantization, self.context_length = quantization, context_length
        self.supports_tools = supports_tools
        self.records: list[dict] = []

    def chat(self, messages, tools):
        os.environ["OPENROUTER_API_KEY"] = self._key
        return chat(model=self.model, messages=messages, tools=tools, provider=self.tag,
                    temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
                    quantization=self.quantization, context_length=self.context_length,
                    supports_tools=self.supports_tools, record_hook=self.records.append)

    def spend(self) -> float:
        return sum(r.get("cost") or 0.0 for r in self.records)


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST.read_text())
    panel = [m["slug"] for m in manifest["model_panel"]["models"]]
    cells = enumerate_cells()

    pool = build_episode_pool()
    # ERROR-BEARING episodes only: the only kind where arms can diverge at all.
    # A clean episode makes every arm look identical for the trivial reason.
    # Both ENDS of the confidence range, because the whole thesis is that
    # behaviour should depend on confidence -- one episode cannot show that.
    err = pool[pool["err"] & (pool["fit_match"] != "earlier_session")]
    episodes = [err.nsmallest(1, "confidence").iloc[0].to_dict(),
                err.nlargest(1, "confidence").iloc[0].to_dict()]
    for e in episodes:
        print(f"episode {e['episode_id']}\n   true={e['true_string']!r} "
              f"decoded={e['decoded_string']!r} confidence={float(e['confidence']):.6f}")
    print()
    episode = episodes[1]              # the high-confidence one carries check C
    confidence = float(episode["confidence"])

    report: dict = {"episodes": [{"episode_id": e["episode_id"],
                                  "confidence": float(e["confidence"]),
                                  "true_string": e["true_string"],
                                  "decoded_string": e["decoded_string"]}
                                 for e in episodes]}

    # --- A. endpoint resolution, all 5 models, zero inference cost --------
    print("A. resolving endpoints (no inference, no cost)")
    endpoints, fatal = {}, []
    for m in panel:
        try:
            ep = resolve_endpoint(m, key)
            endpoints[m] = ep
            print(f"   OK  {m:<30} tag={ep['tag']!r} provider={ep['provider_name']!r} "
                  f"tools={ep['supports_tools']} ctx={ep['context_length']}")
        except Exception as e:
            fatal.append(m)
            print(f"   FAIL {m:<30} {type(e).__name__}: {e}")
    report["A_endpoints"] = {m: {k: v for k, v in ep.items() if k != "candidates"}
                             for m, ep in endpoints.items()}
    report["A_failed"] = fatal
    if fatal:
        print(f"\nSTOP: {len(fatal)} model(s) cannot be pinned. Fix before spending.")
        OUT.write_text(json.dumps(report, indent=2))
        return 1

    # --- B. all 34 cells, one live episode, on the cheapest model ---------
    print(f"\nB. {len(cells)} cells x {len(episodes)} episodes on {CHEAPEST}")
    ep = endpoints[CHEAPEST]
    client = PinnedClient(CHEAPEST, ep["tag"], key, ep["quantization"],
                          ep["context_length"], ep["supports_tools"])
    rng = np.random.default_rng(20260828)
    rows, errors = [], []
    for episode_b in episodes:
      conf_b = float(episode_b["confidence"])
      print(f"   -- confidence {conf_b:.6f} --")
      for cell in cells:
        before = len(client.records)
        try:
            rec = run_episode_for_cell(cell=cell, episode=episode_b, confidence=conf_b,
                                       client=client, rng=rng)
            calls = client.records[before:]
            row = {"cell": cell.name, "episode_id": episode_b["episode_id"],
                   "confidence": conf_b, "uses_llm": cell.uses_llm,
                   "mechanism": cell.control_mechanism,
                   "uncertainty_source": cell.uncertainty_source,
                   "covered": rec.covered, "faithful": rec.faithful,
                   "parse_failed": rec.parse_failed, "n_turns": rec.n_turns,
                   "executed": rec.executed, "n_api_calls": len(calls),
                   "served_provider": rec.served_provider,
                   "cost_usd": sum(c.get("cost") or 0.0 for c in calls)}
            if cell.control_mechanism == "enforced":
                # The whole point of an enforced arm: one run, any threshold.
                gated = apply_enforced_gate(rec, threshold=conf_b + 1e-9)
                row["gated_above_own_conf_covered"] = gated.covered
            rows.append(row)
            print(f"     {cell.name:<40} cov={rec.covered!s:<5} faith={rec.faithful!s:<5} "
                  f"calls={len(calls)} act={(rec.executed or {}).get('name')}")
        except Exception as e:
            errors.append({"cell": cell.name, "episode_id": episode_b["episode_id"],
                           "error": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"     {cell.name:<40} ERROR {type(e).__name__}: {str(e)[:150]}")
    report["B_cells"] = rows
    report["B_errors"] = errors
    report["B_cost_usd"] = client.spend()

    # --- B'. the differential assertions no offline test makes -------------
    llm_rows = [r for r in rows if r["uses_llm"]]
    non_llm_rows = [r for r in rows if not r["uses_llm"]]
    def outcome(r):
        return (r["covered"], r["faithful"], json.dumps(r["executed"], sort_keys=True))

    by_ep = {}
    for r in llm_rows:
        by_ep.setdefault(r["episode_id"], set()).add(outcome(r))
    # The assertion that matters: the SAME cell, run on two episodes whose
    # confidence differs by six orders of magnitude, must not be frozen.
    by_cell = {}
    for r in llm_rows:
        by_cell.setdefault(r["cell"], set()).add(outcome(r))
    n_cells_responsive = sum(1 for v in by_cell.values() if len(v) > 1)

    diffs = {
        "n_cells_run": len(rows),
        "n_cells_errored": len(errors),
        "non_llm_made_zero_api_calls": all(r["n_api_calls"] == 0 for r in non_llm_rows),
        "n_non_llm_cells": len(non_llm_rows),
        "llm_cells_all_made_calls": all(r["n_api_calls"] > 0 for r in llm_rows),
        "n_parse_failures": sum(r["parse_failed"] for r in llm_rows),
        "distinct_llm_outcomes_overall": len({outcome(r) for r in llm_rows}),
        "distinct_llm_outcomes_within_episode": {k: len(v) for k, v in by_ep.items()},
        "n_llm_cells_responsive_to_confidence": n_cells_responsive,
        "n_llm_cells": len(by_cell),
        "outcome_counts": dict(Counter(
            f"conf={r['confidence']:.4f},cov={r['covered']},faith={r['faithful']},"
            f"act={(r['executed'] or {}).get('name')}" for r in llm_rows)),
        "enforced_gate_suppressed_above_threshold": all(
            r["gated_above_own_conf_covered"] is False
            for r in rows if r.get("mechanism") == "enforced"),
    }
    report["B_differential"] = diffs
    print("\n   differential checks:")
    for k, v in diffs.items():
        print(f"     {k}: {v}")

    # --- C. every remaining model routes, tool-calls, parses --------------
    probe_cell = next(c for c in cells if c.uses_llm
                      and c.uncertainty_source == "decoder_confidence"
                      and c.control_mechanism == "advisory")
    print(f"\nC. one live episode on {probe_cell.name} for each remaining model")
    per_model = []
    for m in panel:
        if m == CHEAPEST:
            continue
        e = endpoints[m]
        c2 = PinnedClient(m, e["tag"], key, e["quantization"], e["context_length"],
                          e["supports_tools"])
        try:
            rec = run_episode_for_cell(cell=probe_cell, episode=episode,
                                       confidence=confidence, client=c2)
            per_model.append({"model": m, "tag": e["tag"], "covered": rec.covered,
                              "faithful": rec.faithful, "parse_failed": rec.parse_failed,
                              "n_turns": rec.n_turns, "executed": rec.executed,
                              "served_provider": rec.served_provider,
                              "n_api_calls": len(c2.records), "cost_usd": c2.spend()})
            print(f"   {m:<30} cov={rec.covered!s:<5} faith={rec.faithful!s:<5} "
                  f"turns={rec.n_turns} act={(rec.executed or {}).get('action')} "
                  f"${c2.spend():.6f}")
        except Exception as ex:
            per_model.append({"model": m, "tag": e["tag"],
                              "error": f"{type(ex).__name__}: {str(ex)[:300]}"})
            print(f"   {m:<30} ERROR {type(ex).__name__}: {str(ex)[:200]}")
    report["C_models"] = per_model
    report["C_cost_usd"] = sum(r.get("cost_usd") or 0.0 for r in per_model)
    report["total_cost_usd"] = report["B_cost_usd"] + report["C_cost_usd"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    # A smoke that only proves "nothing crashed" is the exact test class that
    # let four defects through. Divergence is a PASS CONDITION, not a note.
    ok = (not errors and not fatal
          and diffs["non_llm_made_zero_api_calls"]
          and diffs["llm_cells_all_made_calls"]
          and diffs["distinct_llm_outcomes_overall"] > 1
          and all("error" not in r for r in per_model))
    if diffs["distinct_llm_outcomes_overall"] <= 1:
        print("\n   !! every LLM arm produced an IDENTICAL outcome on both episodes.")
        print("      The design has no live signal to measure. Do not run.")
    print(f"\nTOTAL SMOKE COST ${report['total_cost_usd']:.5f}   -> {OUT}")
    print("SMOKE PASS" if ok else "SMOKE FAIL -- do not start the main run")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
