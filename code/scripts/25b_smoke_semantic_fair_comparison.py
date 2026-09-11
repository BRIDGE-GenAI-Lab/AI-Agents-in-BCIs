"""FULL-ARM LIVE SMOKE for the semantic fair-comparison experiment: every new
`fair:*` arm, on two real episodes, before any money is spent at scale.

The house ruling this implements: four defects in this codebase survived 143+
passing offline tests and were found only by live API calls costing pennies
(see `07_smoke.py`, the template for this file). Every one was invisible
offline because every test either exercised a pure function or used a
cell-agnostic scripted fake. A smoke that only proves "nothing crashed" is
the exact test class that let them through, so DIVERGENCE IS A PASS
CONDITION here, not a note.

Three checks, cheapest first:

  A. Endpoint resolution for all 10 panel models -- NO inference, no cost.
     Applies `08_run.py`'s `MODEL_TAG_OVERRIDE` exactly as the paid run does,
     so what this check clears is what the paid run will actually dial. The
     panel grew from 5 to 10 models (Task 5) and the five additions have
     never been pinned by anything; a model with no tool-capable endpoint
     must surface here, not four hours into a paid run.

  B. All five `fair:*` arms x THREE frozen episodes on the cheapest panel
     model. Never one, because arms CANNOT diverge on a single episode. The
     episodes span TWO axes, because a first version of this gate spanned
     only one and could not observe the divergence it demanded:

       confidence_min / confidence_max -- the lowest- and highest-confidence
         ERROR-BEARING episodes. Error-bearing because a clean command is an
         exact match every arm resolves identically for a reason that says
         nothing about the design; both ends of the range because the whole
         thesis is that behaviour should depend on confidence.

       resolver_unresolvable -- the lowest-confidence episode among those
         error-bearing episodes `lexical_resolve` ABSTAINS on (12 of the 68;
         all 3-substitution corruptions). Confidence and corruption
         DIFFICULTY are separate axes, and selecting only on the first
         picked two easy 2-substitution episodes that every arm solved
         identically: with all arms correct there is no room for two of them
         to differ, so the check below could not fire whether or not the
         code was sound. This is where the deterministic resolver cannot
         repair and a language model's semantic reasoning is the only thing
         that could, i.e. the one regime where the two LLM architectures can
         actually part company.

         The rule is stated on episode PROPERTIES (error-bearing, resolver
         abstains, minimum confidence) and never on any observed outcome.
         Picking the episode that had already been seen to diverge would
         make this gate select on its own pass condition, which is how a
         gate stops catching what it was written to catch.

         DISCLOSURE, because that independence is partial and the difference
         matters. The rule is computed at runtime and hardcodes no episode
         id, so the EPISODE was not cherry-picked. But the choice between an
         argmin- and an argmax-confidence rule was made after a diagnostic
         probe had already seen BOTH ends of this subset diverge (the
         recommendation that came out of that probe named the argmax
         episode; the argmin rule shipped); only the untested middle of the
         subset was blind. So the choice of WHICH rule was not itself
         independent of observed outcomes, even though the rule it produced
         is deterministic and outcome-free.

     The hybrid arm additionally gets a GATE PROBE: one live reply per
     episode, scored twice through the real `run_hybrid_semantic_episode` --
     at threshold=-inf (what the paid run records) and at that episode's own
     confidence + 1e-9. One reply scored twice, not two replies: at
     temperature 0.7 two calls are two different answers, and a difference in
     admission would then be confounded with a difference in what the model
     said. Scoring the same reply at both thresholds isolates the gate, which
     is the live proof that admission is decided outside the model -- the
     same trick `07_smoke.py` plays with `apply_enforced_gate` on the main
     study's enforced cells.

     Both halves are asserted, because `covered=False` on its own proves
     nothing: a reply of "abstain" leaves no proposal to admit and lands on
     covered=False for a reason that has nothing to do with the gate. The -inf
     scoring says whether a proposal existed at all, so the probe can require
     that at least one refusal was a refusal OF SOMETHING.

  C. The required divergence assertions, unweakened. `fair:hybrid_semantic
     _gate` and `fair:llm_vocab:advisory` must produce different outcomes on
     the `resolver_unresolvable` episode specifically: if they do not, either
     the vocabulary disclosure or the hybrid gate is not actually doing
     anything different, which is a defect to fix before spending real
     money, not a result to report. The two confidence-extreme episodes keep
     every assertion they already carried -- the exact and lexical resolver
     arms must still part company on both of them, and each must still show
     more than one distinct outcome across the five arms.

Writes output/tables/smoke_semantic_fair.json. Never prints or stores the
API key. Expected total cost is a fraction of a cent.

Run: OPENROUTER_API_KEY=... PYTHONPATH=code /private/tmp/nag_venv/bin/python3 \\
     code/scripts/25b_smoke_semantic_fair_comparison.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

from nag.naturalistic import lexical_resolve, run_hybrid_semantic_episode  # noqa: E402
from nag.openrouter import resolve_endpoint  # noqa: E402


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fair = _load_sibling("_nag_fair25", "25_semantic_fair_comparison.py")
run08 = fair.run08

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
OUT = REPO_ROOT / "output" / "tables" / "smoke_semantic_fair.json"


def _outcome(row) -> tuple:
    return (row["covered"], row["faithful"], row["executed_name"])


class _RecordingClient:
    """A `PinnedClient` that also keeps the raw response, so ONE live reply can
    be scored at two thresholds. Observes only; it parses nothing."""

    def __init__(self, inner):
        self.inner = inner
        self.records = inner.records
        self.last: tuple | None = None

    def chat(self, messages, tools):
        self.last = self.inner.chat(messages, tools)
        return self.last


class _ReplayClient:
    """Hands back an already-received response without calling anything.

    Lets the gate probe below evaluate the SAME model reply at two different
    thresholds through the real `run_hybrid_semantic_episode`, so the only
    thing that differs between the two evaluations is the threshold. Two live
    calls could not support that claim -- at temperature 0.7 they are two
    different replies, and a difference in admission would be confounded with
    a difference in what the model said. This also keeps every parse in
    production code: the smoke never reimplements the vocabulary match.
    """

    def __init__(self, response):
        self.response = response

    def chat(self, messages, tools):
        return self.response


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="override the auto-selected cheapest panel model")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST.read_text())
    models = manifest["model_panel"]["models"]
    panel = [m["slug"] for m in models]

    # The cheapest model is CHOSEN, not remembered: the panel doubled in Task 5
    # and the previous cheapest (07_smoke.py's hardcoded z-ai/glm-5.3-flash) is
    # no longer it. Sorted output makes the choice auditable rather than
    # implicit.
    by_price = sorted(models, key=lambda m: m["projected_usd_per_episode"])
    cheapest = args.model or by_price[0]["slug"]
    if cheapest not in panel:
        print(f"{cheapest!r} is not in the panel {panel}", file=sys.stderr)
        return 1

    episodes, _ = fair._load_source_episodes()
    err = [e for e in episodes if e["is_error_bearing"]]
    # Two axes, three corners -- see the module docstring. The third rule is
    # stated purely on episode properties, never on an observed outcome.
    unresolvable = [e for e in err
                    if lexical_resolve(e["corrupted_string"],
                                       max_distance=fair.LEXICAL_MAX_DISTANCE) is None]
    if not unresolvable:
        print("no error-bearing episode is unresolvable by the lexical resolver -- the "
              "difficulty axis this gate needs does not exist in the frozen set", file=sys.stderr)
        return 1
    roles = [("confidence_min", min(err, key=lambda e: e["confidence"])),
             ("confidence_max", max(err, key=lambda e: e["confidence"])),
             ("resolver_unresolvable", min(unresolvable, key=lambda e: e["confidence"]))]
    smoke_eps = [e for _, e in roles]
    role_of = {e["episode_id"]: role for role, e in roles}
    # The divergence check below names this episode specifically, so it must be
    # a genuinely distinct third episode and not a duplicate of either extreme.
    if len({e["episode_id"] for e in smoke_eps}) != 3:
        print("the three selection rules did not yield three distinct episodes", file=sys.stderr)
        return 1
    hard_ep = roles[2][1]

    print("panel prices (projected $/episode):")
    for m in by_price:
        mark = "  <- carries check B" if m["slug"] == cheapest else ""
        print(f"   {m['slug']:<32} {m['projected_usd_per_episode']:.6f}{mark}")
    print(f"\nsmoke episodes ({len(unresolvable)} of {len(err)} error-bearing episodes are "
          "unresolvable by the lexical resolver):")
    for role, e in roles:
        print(f"   [{role}] {e['episode_id']}\n      command={e['assigned_command']!r} "
              f"corrupted={e['corrupted_string']!r} true_action={e['true_action']!r} "
              f"confidence={e['confidence']:.6f} n_substitutions={e['n_substitutions']}")

    report = {
        "model": cheapest,
        "n_error_bearing": len(err),
        "n_lexically_unresolvable": len(unresolvable),
        "episodes": [{"role": role,
                      **{k: e[k] for k in ("episode_id", "assigned_command", "corrupted_string",
                                           "true_action", "confidence", "n_substitutions")}}
                     for role, e in roles],
    }

    # --- A. endpoint resolution, all 10 models, zero inference cost ---------
    print(f"\nA. resolving endpoints for {len(panel)} panel models (no inference, no cost)")
    endpoints, fatal = {}, []
    for m in panel:
        try:
            ep = resolve_endpoint(m, key)
            override = run08.MODEL_TAG_OVERRIDE.get(m)
            if override and override != ep["tag"]:
                match = next((c for c in ep["candidates"] if c.get("tag") == override), None)
                if match is None:
                    raise RuntimeError(f"override tag {override!r} not offered")
                ep = {"provider_name": match["provider_name"], "tag": match["tag"],
                      "quantization": match.get("quantization"),
                      "context_length": match.get("context_length"),
                      "supports_tools": True, "candidates": ep["candidates"]}
                note = "  (OVERRIDE)"
            else:
                note = ""
            endpoints[m] = ep
            print(f"   OK   {m:<32} tag={ep['tag']!r} tools={ep['supports_tools']} "
                  f"ctx={ep['context_length']}{note}")
        except Exception as e:
            fatal.append(m)
            print(f"   FAIL {m:<32} {type(e).__name__}: {e}")
    report["A_endpoints"] = {m: {k: v for k, v in ep.items() if k != "candidates"}
                             for m, ep in endpoints.items()}
    report["A_failed"] = fatal
    if fatal:
        print(f"\nSTOP: {len(fatal)} model(s) cannot be pinned. Fix before spending.")
        OUT.write_text(json.dumps(report, indent=2, default=str))
        return 1

    # --- B. every fair:* arm x both episodes -------------------------------
    endpoint = endpoints[cheapest]
    print(f"\nB. {len(fair.LLM_CELLS) + len(fair.FREE_CELLS)} arms x {len(smoke_eps)} episodes "
          f"on {cheapest}")
    rows, errors = [], []
    for ep_row in smoke_eps:
        conf = float(ep_row["confidence"])
        print(f"   -- confidence {conf:.6f}  corrupted={ep_row['corrupted_string']!r} --")

        for cell in fair.FREE_CELLS:
            row = fair.build_free_resolver_rows(cell, [ep_row])[0]
            rows.append(row)
            print(f"     {cell.name:<28} cov={row['covered']!s:<5} faith={row['faithful']!s:<5} "
                  f"calls=0 act={row['executed_name']}")

        for cell in fair.LLM_CELLS:
            client = run08.PinnedClient(cheapest, endpoint, key)
            try:
                rec = fair.run_fair_episode(cell, ep_row, conf, client)
                row = fair._base_row(cell, cheapest, ep_row, endpoint)
                row.update({
                    "covered": bool(rec.covered), "faithful": bool(rec.faithful),
                    "parse_failed": bool(rec.parse_failed), "n_turns": int(rec.n_turns),
                    "executed_name": (rec.executed or {}).get("name"),
                    "served_provider": rec.served_provider,
                    "n_api_calls": len(client.records),
                    "cost_usd": sum(float(c.get("cost") or 0.0) for c in client.records),
                    "error": None,
                })
                rows.append(row)
                print(f"     {cell.name:<28} cov={rec.covered!s:<5} faith={rec.faithful!s:<5} "
                      f"calls={len(client.records)} act={(rec.executed or {}).get('name')}")
            except Exception as e:
                errors.append({"cell": cell.name, "episode_id": ep_row["episode_id"],
                               "error": f"{type(e).__name__}: {str(e)[:300]}"})
                print(f"     {cell.name:<28} ERROR {type(e).__name__}: {str(e)[:150]}")

    # The hybrid arm's whole architecture is "the gate lives outside the
    # model". Re-running it at a threshold just above the episode's own
    # confidence is the live proof that the gate is real: whatever the model
    # proposes, admission must be refused.
    # ONE live reply per episode, scored at two thresholds. `covered=False`
    # alone would not prove the GATE refused anything: a reply of "abstain"
    # (or an off-vocabulary one) yields no proposal and so also lands on
    # covered=False, passing this check for a reason that says nothing about
    # the deterministic gate. Scoring the SAME reply at -inf shows whether
    # there was a proposal to refuse in the first place, and the pair
    # (admitted at -inf, refused just above its own confidence) is the only
    # version of this claim that means anything.
    print("\n   hybrid gate probe (one reply, scored at -inf and at conf + 1e-9)")
    hybrid_cell = next(c for c in fair.LLM_CELLS if c.name == fair.HYBRID_CELL)
    gate_probe = []
    for ep_row in smoke_eps:
        conf = float(ep_row["confidence"])
        client = _RecordingClient(run08.PinnedClient(cheapest, endpoint, key))
        try:
            gated = run_hybrid_semantic_episode(cell=hybrid_cell, episode=ep_row, confidence=conf,
                                                threshold=conf + 1e-9, client=client, system="")
            replay = _ReplayClient(client.last)
            ungated = run_hybrid_semantic_episode(cell=hybrid_cell, episode=ep_row,
                                                  confidence=conf, threshold=float("-inf"),
                                                  client=replay, system="")
            gate_probe.append({
                "episode_id": ep_row["episode_id"], "threshold": conf + 1e-9,
                "covered": bool(gated.covered), "parse_failed": bool(gated.parse_failed),
                "proposal_existed": bool(ungated.covered),
                "proposed_action": (ungated.executed or {}).get("name"),
                "cost_usd": sum(float(c.get("cost") or 0.0) for c in client.records),
                "error": None})
            print(f"     {ep_row['episode_id'].split('#')[-1]:<6} proposal="
                  f"{(ungated.executed or {}).get('name')!s:<15} admitted_at_-inf={ungated.covered!s:<5} "
                  f"covered_at_gate={gated.covered} (must be False)")
        except Exception as e:
            gate_probe.append({"episode_id": ep_row["episode_id"], "threshold": conf + 1e-9,
                               "covered": None, "proposal_existed": None,
                               "error": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"     ERROR {type(e).__name__}: {str(e)[:150]}")
    report["B_gate_probe"] = gate_probe
    report["B_rows"] = rows
    report["B_errors"] = errors

    # --- C. the differential assertions no offline test makes --------------
    by_arm_ep = {(r["cell"], r["episode_id"]): r for r in rows}
    llm_rows = [r for r in rows if r["uses_llm"]]
    free_rows = [r for r in rows if not r["uses_llm"]]

    def _pairwise(cell_a, cell_b, key_a, key_b):
        out = []
        for role, e in roles:
            a = by_arm_ep.get((cell_a, e["episode_id"]))
            b = by_arm_ep.get((cell_b, e["episode_id"]))
            out.append({"role": role, "episode_id": e["episode_id"],
                        key_a: _outcome(a) if a else None,
                        key_b: _outcome(b) if b else None,
                        "differ": bool(a and b and _outcome(a) != _outcome(b))})
        return out

    hybrid_vs_advisory = _pairwise(fair.HYBRID_CELL, fair.ADVISORY_CELL, "hybrid", "advisory")
    exact_vs_lexical = _pairwise(fair.EXACT_RESOLVER_CELL, fair.LEXICAL_RESOLVER_CELL,
                                 "exact", "lexical")
    per_episode_distinct = {
        e["episode_id"]: len({_outcome(r) for r in rows if r["episode_id"] == e["episode_id"]})
        for e in smoke_eps}
    extremes = [e["episode_id"] for role, e in roles if role != "resolver_unresolvable"]

    diffs = {
        "n_rows": len(rows),
        "n_errored": len(errors),
        "free_arms_made_zero_api_calls": all(r["n_api_calls"] == 0 for r in free_rows),
        "llm_arms_all_made_calls": all(r["n_api_calls"] > 0 for r in llm_rows),
        "n_parse_failures": sum(bool(r["parse_failed"]) for r in llm_rows),
        "distinct_outcomes_overall": len({_outcome(r) for r in rows}),
        "distinct_outcomes_within_episode": per_episode_distinct,
        "hybrid_vs_advisory": hybrid_vs_advisory,
        # THE check. Required on the resolver_unresolvable episode specifically:
        # that is the only one of the three where the two LLM architectures can
        # part company at all, so demanding it there is what gives the assertion
        # its teeth. Recorded across all three for the record.
        "hybrid_differs_from_advisory_on_hard_episode": any(
            d["differ"] for d in hybrid_vs_advisory if d["role"] == "resolver_unresolvable"),
        "hybrid_differs_from_advisory_somewhere": any(d["differ"] for d in hybrid_vs_advisory),
        "exact_vs_lexical": exact_vs_lexical,
        # Unchanged from the original two-episode gate: the resolver arms must
        # still part company on BOTH confidence extremes, and each extreme must
        # still show more than one distinct outcome across the five arms.
        "exact_differs_from_lexical_on_both_extremes": all(
            d["differ"] for d in exact_vs_lexical if d["role"] != "resolver_unresolvable"),
        "extremes_each_show_multiple_outcomes": all(
            per_episode_distinct[eid] > 1 for eid in extremes),
        # Nothing may be admitted above its own confidence...
        "hybrid_gate_suppressed_above_own_confidence": all(
            p.get("parse_failed") is False and p.get("covered") is False for p in gate_probe),
        # ...and at least one of those refusals must have had a real proposal
        # to refuse, established by scoring the SAME reply at -inf. Without
        # this second half the check passes on a run where the model simply
        # abstained everywhere, which proves nothing about the gate. "At least
        # one" rather than "all" because an occasional abstain is ordinary
        # model behaviour, not a defect.
        "hybrid_gate_refused_a_real_proposal": any(
            p.get("proposal_existed") is True and p.get("covered") is False for p in gate_probe),
    }
    report["C_differential"] = diffs
    report["total_cost_usd"] = (sum(float(r.get("cost_usd") or 0.0) for r in rows)
                                + sum(float(p.get("cost_usd") or 0.0) for p in gate_probe))

    print("\nC. differential checks:")
    for k, v in diffs.items():
        if k in ("hybrid_vs_advisory", "exact_vs_lexical"):
            print(f"   {k}:")
            for d in v:
                print(f"      [{d['role']:<21}] differ={d['differ']!s:<5} {d}")
        else:
            print(f"   {k}: {v}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))

    ok = (not errors and not fatal
          and diffs["free_arms_made_zero_api_calls"]
          and diffs["llm_arms_all_made_calls"]
          # Counted since the first version of this gate, but never gated on --
          # a parse failure is a real defect on an arm whose whole output is a
          # parsed line, and it must fail the run, not merely appear in a report.
          and diffs["n_parse_failures"] == 0
          and diffs["distinct_outcomes_overall"] > 1
          and diffs["hybrid_differs_from_advisory_on_hard_episode"]
          and diffs["exact_differs_from_lexical_on_both_extremes"]
          and diffs["extremes_each_show_multiple_outcomes"]
          and diffs["hybrid_gate_suppressed_above_own_confidence"]
          and diffs["hybrid_gate_refused_a_real_proposal"])
    if not diffs["hybrid_differs_from_advisory_on_hard_episode"]:
        print("\n   !! fair:hybrid_semantic_gate and fair:llm_vocab:advisory produced IDENTICAL")
        print(f"      outcomes on {hard_ep['corrupted_string']!r}, the episode the lexical")
        print("      resolver cannot repair. Either the vocabulary disclosure or the hybrid")
        print("      gate is not doing anything different. Do not run.")
    print(f"\nTOTAL SMOKE COST ${report['total_cost_usd']:.6f}   -> {OUT}")
    print("SMOKE PASS" if ok else "SMOKE FAIL -- do not start the paid run")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
