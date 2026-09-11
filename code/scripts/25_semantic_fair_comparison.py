"""TASK 6: the SEMANTIC FAIR-COMPARISON experiment -- five new arms over the
SAME 200 frozen naturalistic episodes, under symmetric information.

STATUS: reviewer-motivated follow-up experiment (fair-comparison plan,
2026-09-09), designed after the Task 20 naturalistic benchmark was already
run and reported, and frozen (`--build-manifest`, committed) before any
model call. It is NOT part of the original pre-specified experiment and the
manuscript must label it that way, exactly as Task 20 already is.

WHY THIS EXISTS. Task 20 (`19_naturalistic_run.py`) answered the ecological-
validity objection but left two asymmetries the editor can still name:

  INFORMATION. `lexical_resolve` is handed `NATURAL_COMMANDS` as a literal.
  The model, reading only `nag.prompts.build_system`'s output, never sees
  that list. "Language models cannot beat a lexical resolver" is not a claim
  a comparison can support when only one side holds the answer key. The two
  `fair:llm_vocab:*` cells close this by prompting through
  `nag.naturalistic.build_naturalistic_system`, which discloses all nine
  commands verbatim.

  OPERATING POINT. Task 20's two deterministic comparators are FIXED POINTS,
  not curves: they call `nonllm_gate(conf, -inf, action)`, so confidence
  never enters admission and there is no frontier to compare against the
  models' own. The two `fair:*_resolver_gate` arms replace that single point
  with a genuine sweep over `threshold_grid`, via
  `nag.naturalistic.resolver_gate_curve`.

A third arm, `fair:hybrid_semantic_gate`, is the architecture the reviewer
actually proposed rather than either side of the original contrast:
"language models are useful for semantic error correction, but decoder
uncertainty should remain outside the model as an enforced control signal."
The model's whole job is one text proposal of which canonical command was
meant; a deterministic threshold outside the model decides admission.

NOTHING FROZEN MOVES. `nag/taxonomy.py`, `nag/naturalistic.py`'s
`canonical_action` / `NATURAL_COMMANDS` / `COMMAND_TO_ACTION`, and every
script numbered 00-24 are untouched. This script adds arms; the already-
reported Task 20 numbers are produced by code this file never calls.

THE EPISODES ARE NOT REDRAWN. This experiment reuses
`output/tables/naturalistic_manifest.json`'s 200 episodes VERBATIM -- same
`corrupted_string`, `true_action`, and recalibrated `confidence` per
episode. They are referenced, not copied: duplicating 200 episode records
into a second manifest would create two sources of truth that can silently
diverge. Instead this manifest records the source file, the commit its
declaration is pinned to, the episode count, and the source's own
`natural_manifest_digest`, and BOTH `--build-manifest` and the run step
recompute that digest from the source file's episodes before proceeding.
A new draw here would be exactly the post-hoc selection the two-step freeze
exists to prevent.

ONE CALL PER EPISODE, SWEPT AFTERWARDS. `run_hybrid_semantic_episode` takes
a threshold, but its LLM call does not depend on one: the threshold reaches
neither `build_hybrid_user` nor the system prompt, so the model cannot
condition on it. The hybrid arm is therefore called ONCE per (model,
episode) at `PROPOSAL_THRESHOLD = -inf`, which records the raw proposal, and
the whole 101-point sweep is reconstructed downstream from that proposal
plus the episode's confidence. Running the grid live would have multiplied
this arm's cost by 101 for identical rows. The same logic is why
`fair:llm_vocab:enforced` is run once (see `nag.agent.run_episode_for_cell`,
which documents it for the main study's enforced cells) and why the two
resolver arms are computed once and not per model.

READ `covered` ON THE THREE PROPOSAL ARMS CORRECTLY. For
`fair:hybrid_semantic_gate`, `fair:exact_resolver_gate` and
`fair:lexical_resolver_gate`, a row's `covered` means "a proposal exists",
NOT "this episode is covered at the reported operating point". Coverage at
threshold t is `covered & (confidence >= t)`; risk is computed among those.
`fair:llm_vocab:advisory` is the one arm whose row IS its final outcome --
an advisory arm's own decision is what it reports (see
`nag.agent.run_episode_for_cell`, same convention).

BUDGET. `spent_so_far` is `08_run.py`'s, imported rather than reimplemented,
because it globs every `output/intermediate/runs*` directory AT CALL TIME
and so counts this script's own checkpoints on a resumed run. The three LLM
arms x the panel x 200 episodes are the only paid work; the two resolver
arms make zero API calls by construction.

Run:
  # step 1 -- build and commit the manifest (no cost, no calls)
  PYTHONPATH=code /private/tmp/nag_venv/bin/python3 code/scripts/25_semantic_fair_comparison.py --build-manifest
  git add output/tables/semantic_fair_comparison_manifest.json
  git commit -m "manifest: freeze the semantic fair-comparison experiment's cells and threshold grid"

  # step 2 -- the live smoke gate (cents), which must pass before step 3
  OPENROUTER_API_KEY=... PYTHONPATH=code /private/tmp/nag_venv/bin/python3 code/scripts/25b_smoke_semantic_fair_comparison.py

  # step 3 -- dry run, then the actual paid run
  PYTHONPATH=code /private/tmp/nag_venv/bin/python3 code/scripts/25_semantic_fair_comparison.py --dry-run
  OPENROUTER_API_KEY=... PYTHONPATH=code /private/tmp/nag_venv/bin/python3 code/scripts/25_semantic_fair_comparison.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

from nag.design import Cell  # noqa: E402
from nag.naturalistic import (  # noqa: E402
    NATURAL_COMMANDS,
    build_naturalistic_system,
    canonical_action,
    lexical_resolve,
    resolver_gate_curve,
    run_hybrid_semantic_episode,
    run_naturalistic_episode,
)
from nag.openrouter import resolve_endpoint  # noqa: E402


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run08 = _load_sibling("_nag_run08", "08_run.py")
run19 = _load_sibling("_nag_run19", "19_naturalistic_run.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
NATURAL_MANIFEST = REPO_ROOT / "output" / "tables" / "naturalistic_manifest.json"
FAIR_MANIFEST = REPO_ROOT / "output" / "tables" / "semantic_fair_comparison_manifest.json"
RESOLVER_CURVES = REPO_ROOT / "output" / "tables" / "semantic_fair_resolver_curves.csv"

RUNS_DIR = REPO_ROOT / "output" / "intermediate" / "runs_semantic_fair"

# Matches `resolver_gate_curve`'s own default (nag/naturalistic.py) so the
# hybrid arm and both resolver arms are swept over an identical grid and their
# AURCs are comparable without interpolation.
THRESHOLD_GRID = np.linspace(0.0, 1.0, 101)

# `lexical_resolve`'s default, and the value Task 20's frozen manifest already
# recorded (`max_distance`). Changing it here would make this experiment's
# lexical comparator a different object from the one already reported.
LEXICAL_MAX_DISTANCE = 2

# The hybrid and enforced arms are called ONCE, at a threshold that admits
# every parseable proposal, because the model never sees a threshold and so
# cannot condition on one -- see the module docstring.
PROPOSAL_THRESHOLD = float("-inf")

# Episodes per checkpoint flush -- see `run_fair_cell`'s `on_batch`.
BATCH_SIZE = 25

# Requests/min overriding `08_run.py`'s MODEL_RPM for this experiment only.
#
# That table paces three models at 18/min against an OpenRouter
# `openrouter_new_account` cap of 20/min, measured when the account was new.
# Re-probed 2026-09-10 after the account was upgraded: 40 requests each,
# sustained, against the same pinned endpoint tags the run uses --
#
#   openai/gpt-5.6-luna        40/40 OK at 51.9/min
#   anthropic/claude-sonnet-5  40/40 OK at 60.2/min
#   google/gemini-3.7-flash    40/40 OK at 60.3/min
#
# -- zero 429s anywhere. SUSTAINED, not a burst: a 20/min cap would have
# fired after the 20th request in the first minute, which is exactly the
# mistake a burst of 12 made once before (it wrongly cleared claude-sonnet-5,
# which then failed 121 of 200 episodes and cost $0.849 of discarded rows).
#
# Set to 45, not the measured 60: the probe used a two-token prompt and
# max_tokens=1, while a real episode carries the tool schemas and several
# turns, so a token-rate limit this probe cannot see would bite a real run
# first. Same "leave headroom" posture as the 18-against-20 it replaces.
#
# Lives HERE, not in `08_run.py`, because scripts 00-24 are frozen (see this
# plan's Global Constraints). `chat`'s jittered backoff still absorbs
# provider-side 429s, and with BATCH_SIZE checkpointing a bad guess costs at
# most 25 episodes rather than a cell.
RPM_OVERRIDE = {
    "openai/gpt-5.6-luna": 45,
    "anthropic/claude-sonnet-5": 45,
    "google/gemini-3.7-flash": 45,
}


def _rpm_for(model: str):
    """This experiment's pacing for `model`: the re-measured override where
    one exists, otherwise `08_run.py`'s frozen table (`None` = unthrottled)."""
    if model in RPM_OVERRIDE:
        return RPM_OVERRIDE[model]
    return run08.MODEL_RPM.get(model)

ADVISORY_CELL = "fair:llm_vocab:advisory"
ENFORCED_CELL = "fair:llm_vocab:enforced"
HYBRID_CELL = "fair:hybrid_semantic_gate"
EXACT_RESOLVER_CELL = "fair:exact_resolver_gate"
LEXICAL_RESOLVER_CELL = "fair:lexical_resolver_gate"

LLM_CELLS = [
    Cell(ADVISORY_CELL, "decoder_confidence", "advisory", scaffold=0),
    Cell(ENFORCED_CELL, "decoder_confidence", "enforced", scaffold=0),
    Cell(HYBRID_CELL, "decoder_confidence", "enforced", scaffold=0),
]
FREE_CELLS = [
    Cell(EXACT_RESOLVER_CELL, "decoder_confidence", "enforced", scaffold=0, uses_llm=False),
    Cell(LEXICAL_RESOLVER_CELL, "decoder_confidence", "enforced", scaffold=0, uses_llm=False),
]
RESOLVERS = {
    EXACT_RESOLVER_CELL: canonical_action,
    LEXICAL_RESOLVER_CELL: (lambda s: lexical_resolve(s, max_distance=LEXICAL_MAX_DISTANCE)),
}

CELL_SPEC = {
    ADVISORY_CELL: {
        "uses_llm": True,
        "harness": "nag.naturalistic.run_naturalistic_episode (4-tool agent loop, unchanged)",
        "system_prompt": "nag.naturalistic.build_naturalistic_system(cell, confidence) -- nine commands disclosed verbatim",
        "uncertainty_source": "decoder_confidence",
        "control_mechanism": "advisory",
        "renders_confidence_to_the_model": True,
        "reported_as": "FIXED POINT -- an advisory arm's own decision is its final outcome, never swept",
        "why": (
            "the information-symmetric counterpart of Task 20's "
            "factorial:decoder_confidence:advisory:s0 cell: identical harness, identical "
            "recalibrated confidence, the nine-command vocabulary added to the prompt"
        ),
    },
    ENFORCED_CELL: {
        "uses_llm": True,
        "harness": "nag.naturalistic.run_naturalistic_episode (4-tool agent loop, unchanged)",
        "system_prompt": "nag.naturalistic.build_naturalistic_system(cell, confidence) -- vocabulary disclosed; build_system withholds the numeric confidence for an enforced cell",
        "uncertainty_source": "decoder_confidence",
        "control_mechanism": "enforced",
        "renders_confidence_to_the_model": False,
        "reported_as": "PROPOSAL -- run once, swept over threshold_grid downstream (covered & confidence >= t)",
        "why": (
            "the information-symmetric counterpart of Task 20's "
            "factorial:decoder_confidence:enforced:s0 cell; one run serves every threshold "
            "because the model never sees one (nag.agent.run_episode_for_cell, same argument)"
        ),
    },
    HYBRID_CELL: {
        "uses_llm": True,
        "harness": "nag.naturalistic.run_hybrid_semantic_episode -- ONE call, no tools, no loop",
        "system_prompt": "\"\" (empty) -- the whole prompt is build_hybrid_user, which already carries SINGLESHOT_SCAFFOLD and the nine-command vocabulary; mirrors nag.agent._run_singleshot, which likewise sends no system message",
        "uncertainty_source": "decoder_confidence",
        "control_mechanism": "enforced",
        "renders_confidence_to_the_model": False,
        "reported_as": "PROPOSAL -- called once at threshold=-inf, swept over threshold_grid downstream",
        "why": (
            "the reviewer's own proposed architecture: semantic correction from the model, "
            "admission from a deterministic threshold outside it. Confidence is deliberately "
            "withheld from this arm's prompt -- an architecture defined by keeping decoder "
            "uncertainty outside the model must not leak it back in through the prompt"
        ),
    },
    EXACT_RESOLVER_CELL: {
        "uses_llm": False,
        "harness": "nag.naturalistic.canonical_action + a deterministic confidence gate",
        "uncertainty_source": "decoder_confidence",
        "control_mechanism": "enforced",
        "reported_as": "PROPOSAL + swept curve via nag.naturalistic.resolver_gate_curve",
        "cost": "zero API calls by construction; computed ONCE, not per model",
        "why": (
            "the swept version of Task 20's frozen natural_confidence_gate_canonical fixed "
            "point; exact match only, never repairs (canonical_action's own docstring)"
        ),
    },
    LEXICAL_RESOLVER_CELL: {
        "uses_llm": False,
        "harness": f"nag.naturalistic.lexical_resolve(max_distance={LEXICAL_MAX_DISTANCE}) + a deterministic confidence gate",
        "uncertainty_source": "decoder_confidence",
        "control_mechanism": "enforced",
        "reported_as": "PROPOSAL + swept curve via nag.naturalistic.resolver_gate_curve",
        "cost": "zero API calls by construction; computed ONCE, not per model",
        "why": (
            "the PRIMARY COMPARATOR, now swept: the system the language models have to beat, "
            "and the one that motivated the information-symmetry fix (it is handed "
            "NATURAL_COMMANDS as a literal, so the models must be too)"
        ),
    },
}


def _git_state(path: Path) -> tuple[bool, bool, bool]:
    """(exists_on_disk, tracked_by_git, no_uncommitted_diff) for one file.

    `19_naturalistic_run.py::_manifest_git_state` with the path lifted out of
    the closure -- this script has two manifests to check, its own and the
    source it reuses, and both must be frozen before a request is issued.
    """
    if not path.exists():
        return False, False, False
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                             cwd=REPO_ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)],
                           cwd=REPO_ROOT, capture_output=True).returncode == 0
    return True, tracked, clean


def _head_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def _declaration_digest(d: dict) -> str:
    """Digest over everything this manifest DECLARES -- the arms, the sweep,
    the panel, and which episode set it is pinned to. Recomputed by the run
    step, so an edit to any declared parameter after freezing stops the run
    instead of silently changing what was run.
    """
    blob = json.dumps({k: d[k] for k in ("episode_source", "threshold_grid", "cells",
                                         "arms_to_run", "free_arms",
                                         "model_panel_at_declaration")},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_source_episodes() -> tuple[list[dict], dict]:
    """The 200 Task 20 episodes, verbatim, with every freeze guard applied.

    Refuses on anything that would mean the episodes are not the ones already
    committed and already used for the reported Task 20 numbers: the source
    manifest missing, untracked, dirty against HEAD, or carrying episodes that
    no longer hash to its own recorded `natural_manifest_digest`. The digest is
    recomputed with `19_naturalistic_run.py`'s OWN `_episodes_digest`, imported
    rather than reimplemented, so a reimplementation here could never disagree
    with the function that wrote the number in the first place.
    """
    exists, tracked, clean = _git_state(NATURAL_MANIFEST)
    if not exists:
        raise SystemExit(f"{NATURAL_MANIFEST} does not exist -- this experiment reuses Task 20's "
                         "frozen episodes and never draws its own.")
    if not tracked:
        raise SystemExit(f"{NATURAL_MANIFEST} is not tracked by git; it must be committed.")
    if not clean:
        raise SystemExit(f"{NATURAL_MANIFEST} has uncommitted changes relative to HEAD. The "
                         "episodes this experiment reuses must be the frozen, committed ones.")
    nm = json.loads(NATURAL_MANIFEST.read_text())
    episodes = nm["natural_episodes"]
    recomputed = run19._episodes_digest(episodes)
    if recomputed != nm["natural_manifest_digest"]:
        raise SystemExit("naturalistic_manifest.json's episodes do not match its own recorded "
                         "natural_manifest_digest -- the file was edited after being written.")
    if len(episodes) != run19.N_TOTAL_DONORS:
        raise SystemExit(f"expected {run19.N_TOTAL_DONORS} frozen episodes, found {len(episodes)}")
    return episodes, nm


def build_manifest() -> dict:
    episodes, nm = _load_source_episodes()
    panel = json.loads(MANIFEST.read_text())["model_panel"]["models"]
    realized = nm["draw"]["realized_prevalence"]

    d = {
        "task": "Task 6: semantic fair-comparison experiment",
        "status": (
            "reviewer-motivated follow-up experiment (fair-comparison plan, 2026-09-09); NOT part "
            "of the original pre-specified experiment, and not part of the already-reported Task "
            "20 naturalistic benchmark either -- see this script's module docstring"
        ),
        "frozen_before_run_note": (
            "this file must be committed to git BEFORE any model request is issued -- the run "
            "step refuses to start unless it is tracked, has no uncommitted diff against HEAD, "
            "and its declaration hashes to fair_manifest_digest below"
        ),
        "episode_source": {
            "file": "output/tables/naturalistic_manifest.json",
            "n_episodes": len(episodes),
            "natural_manifest_digest": nm["natural_manifest_digest"],
            "realized_error_prevalence": realized,
            "reuse_note": (
                "the SAME 200 episodes as Task 20, reused verbatim (identical corrupted_string, "
                "true_action, and recalibrated confidence per episode) and referenced rather than "
                "copied, so there is exactly one source of truth for them. This experiment is a "
                "new set of ARMS over a frozen episode set, never a new draw"
            ),
            "confidence_used": nm["confidence_used"],
        },
        "threshold_grid": {
            "method": "np.linspace(0.0, 1.0, 101)",
            "n": int(THRESHOLD_GRID.size),
            "min": float(THRESHOLD_GRID.min()),
            "max": float(THRESHOLD_GRID.max()),
            "note": (
                "matches nag.naturalistic.resolver_gate_curve's own default, so the hybrid arm "
                "and both resolver arms are swept over an identical grid and their AURCs are "
                "comparable without interpolation"
            ),
            "values": [float(t) for t in THRESHOLD_GRID],
        },
        "cells": CELL_SPEC,
        "arms_to_run": [c.name for c in LLM_CELLS],
        "free_arms": [c.name for c in FREE_CELLS],
        "lexical_max_distance": LEXICAL_MAX_DISTANCE,
        "commands": {c: True for c in NATURAL_COMMANDS},
        "proposal_threshold": (
            "-inf: the enforced and hybrid arms are called once per (model, episode) at a "
            "threshold that admits every parseable proposal, because the model never sees a "
            "threshold and cannot condition on one. Their rows record the PROPOSAL; coverage at "
            "threshold t is (covered & confidence >= t), swept downstream over threshold_grid"
        ),
        "model_panel_at_declaration": [
            {"slug": m["slug"], "projected_usd_per_episode": m["projected_usd_per_episode"]}
            for m in panel
        ],
        "analysis_endpoints": [
            "the vocabulary-disclosed LLM arms against the SWEPT lexical resolver+gate curve, "
            "under symmetric information -- the comparison Task 20 could not make",
            "the hybrid semantic-correction + deterministic-admission architecture against both "
            "the pure-LLM arms and the pure-resolver arms, on one risk-coverage frontier",
            "whether the study's headline (no LLM arm beats a deterministic gate at matched "
            "coverage) survives once the models hold the same vocabulary the resolver does",
        ],
        "reporting_note": (
            "absolute rates here are BENCHMARK ABSOLUTE RISKS AT THE SOURCE POOL'S OBSERVED "
            "decoder-error prevalence, never deployment estimates -- inherited unchanged from "
            "Task 20, whose episodes these are"
        ),
    }
    d["fair_manifest_digest"] = _declaration_digest(d)
    return d


def cmd_build_manifest() -> int:
    exists, tracked, clean = _git_state(FAIR_MANIFEST)
    if exists and tracked and clean:
        print(f"{FAIR_MANIFEST} already exists, is tracked, and matches HEAD -- already frozen. "
              "Refusing to rebuild (redeclaring arms after freezing would defeat the point of "
              "freezing them). Remove it by hand first if this is genuinely a redo.",
              file=sys.stderr)
        return 1
    d = build_manifest()
    FAIR_MANIFEST.write_text(json.dumps(d, indent=2) + "\n")
    print(f"wrote {FAIR_MANIFEST}")
    print(f"episodes:   {d['episode_source']['n_episodes']} reused verbatim from "
          f"{d['episode_source']['file']} (digest {d['episode_source']['natural_manifest_digest'][:16]}...)")
    print(f"paid arms:  {d['arms_to_run']}")
    print(f"free arms:  {d['free_arms']}")
    print(f"grid:       {d['threshold_grid']['n']} thresholds over "
          f"[{d['threshold_grid']['min']}, {d['threshold_grid']['max']}]")
    print(f"panel:      {len(d['model_panel_at_declaration'])} models")
    print(f"fair_manifest_digest: {d['fair_manifest_digest']}")
    print("\n--- COMMIT NOW, before any request: ---")
    print("  git add output/tables/semantic_fair_comparison_manifest.json")
    print('  git commit -m "manifest: freeze the semantic fair-comparison experiment\'s cells and threshold grid"')
    return 0


def _load_committed_manifest() -> dict:
    exists, tracked, clean = _git_state(FAIR_MANIFEST)
    if not exists:
        raise SystemExit(f"{FAIR_MANIFEST} does not exist. Run --build-manifest first, then commit it.")
    if not tracked:
        raise SystemExit(f"{FAIR_MANIFEST} exists but is not tracked by git. Commit it before running.")
    if not clean:
        raise SystemExit(f"{FAIR_MANIFEST} has uncommitted changes relative to HEAD. The manifest "
                         "must be committed BEFORE any request is issued.")
    d = json.loads(FAIR_MANIFEST.read_text())
    if _declaration_digest(d) != d["fair_manifest_digest"]:
        raise SystemExit("semantic_fair_comparison_manifest.json's declaration does not match its "
                         "own recorded fair_manifest_digest -- it was edited after being written.")
    return d


# --- the run ----------------------------------------------------------------

def _base_row(cell, model, ep, endpoint) -> dict:
    """The per-episode row prefix, identical in shape to
    `19_naturalistic_run.py::run_natural_cell`'s, so `runs_semantic_fair/`
    and `runs_natural/` can be concatenated and analysed by one code path.
    Every arm -- paid or free -- goes through this one builder, so the two
    schemas cannot drift apart.
    """
    return {
        "model": model, "cell": cell.name,
        "uncertainty_source": cell.uncertainty_source, "control_mechanism": cell.control_mechanism,
        "scaffold": cell.scaffold, "wording": cell.wording, "uses_llm": cell.uses_llm,
        "episode_id": ep["episode_id"], "donor_episode_id": ep["donor_episode_id"],
        "participant_id": ep["participant_id"], "study": ep["study"],
        "assigned_command": ep["assigned_command"], "corrupted_string": ep["corrupted_string"],
        "true_action": ep["true_action"], "n_substitutions": int(ep["n_substitutions"]),
        "is_error_bearing": bool(ep["is_error_bearing"]), "tier": int(ep["tier"]),
        "confidence": float(ep["confidence"]), "confidence_product": float(ep["confidence_product"]),
        "provider_tag": endpoint["tag"] if endpoint else None,
        "quantization": endpoint["quantization"] if endpoint else None,
    }


def run_fair_episode(cell, ep, conf, client):
    """Dispatch one episode to its arm's harness.

    Two harnesses, chosen on the cell name: the hybrid arm is a single
    toolless call (`run_hybrid_semantic_episode`), the two `llm_vocab` arms
    are the ordinary naturalistic agent loop with the vocabulary-disclosed
    system prompt. The hybrid arm sends NO system message (`system=""`),
    matching `nag.agent._run_singleshot`; its user prompt already carries the
    scaffold and the vocabulary, and adding an advisory system prompt would
    leak the decoder confidence into the one architecture defined by keeping
    it outside the model.
    """
    if cell.name == HYBRID_CELL:
        return run_hybrid_semantic_episode(cell=cell, episode=ep, confidence=conf,
                                           threshold=PROPOSAL_THRESHOLD, client=client, system="")
    system = build_naturalistic_system(cell, confidence=conf)
    return run_naturalistic_episode(cell=cell, episode=ep, confidence=conf, client=client,
                                    system=system)


def run_fair_cell(cell, model, endpoint, episodes, key, concurrency=1, only=None, limiter=None,
                  on_batch=None):
    """Every episode for one (model, cell). `19_naturalistic_run.py::
    run_natural_cell` with this experiment's harness dispatch: same
    per-episode-own-client concurrency (a shared client interleaves
    provenance records and mis-attributes the per-episode cost the budget
    meter is built on), same input-order-preserving `.map`, same accounting.

    `on_batch`, if given, is called with every row completed so far after
    each batch of `BATCH_SIZE` episodes, so the caller can persist partial
    progress. A cell is 200 episodes and can take a quarter of an hour; the
    original write-once-at-the-end design meant a process killed at minute
    fourteen had paid for 190 episodes and kept none of them, which is
    exactly what happened three times on 2026-09-09 (twice to the harness's
    low-memory reaper, once to a tool timeout). Batching costs a little
    parallelism at each boundary and buys a bounded loss of at most
    BATCH_SIZE episodes. The resume path needs no changes to understand
    these files -- it already selects the episodes whose rows are missing or
    errored, which is precisely what a partial checkpoint is.
    """
    def one(indexed_ep):
        _, ep = indexed_ep
        client = run08.PinnedClient(model, endpoint, key, limiter) if endpoint is not None else None
        conf = float(ep["confidence"])
        row = _base_row(cell, model, ep, endpoint)
        try:
            rec = run_fair_episode(cell, ep, conf, client)
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

    items = [(i, e) for i, e in enumerate(episodes) if only is None or e["episode_id"] in only]
    rows = []
    if concurrency <= 1 or endpoint is None:
        for it in items:
            rows.append(one(it))
            if on_batch is not None and len(rows) % BATCH_SIZE == 0:
                on_batch(rows)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for start in range(0, len(items), BATCH_SIZE):
                rows.extend(pool.map(one, items[start:start + BATCH_SIZE]))  # .map preserves order
                if on_batch is not None:
                    on_batch(rows)
    return rows, sum(r["cost_usd"] for r in rows)


def build_free_resolver_rows(cell, episodes) -> list[dict]:
    """One row per episode for a deterministic resolver arm, in the SAME
    schema the paid arms use. `covered` is "the resolver returned a
    proposal", `faithful` is "that proposal is the true action" -- the
    threshold=-inf convention the hybrid arm also uses, so all three
    propose-then-gate architectures are swept downstream by one rule.
    """
    resolver = RESOLVERS[cell.name]
    rows = []
    for ep in episodes:
        proposal = resolver(ep["corrupted_string"])
        row = _base_row(cell, run08.NO_MODEL, ep, None)
        row.update({
            "covered": proposal is not None,
            "faithful": bool(proposal is not None and proposal == ep["true_action"]),
            "parse_failed": False, "n_turns": 0,
            "executed_name": proposal,
            "executed_args": json.dumps({}, sort_keys=True),
            "served_provider": None, "n_api_calls": 0, "n_retries": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "error": None,
        })
        rows.append(row)
    return rows


def write_resolver_curves(episodes) -> pd.DataFrame:
    """The two swept resolver+gate curves, via Task 2's `resolver_gate_curve`,
    written to a table rather than to the run directory: a curve is one row
    per THRESHOLD, not per episode, and mixing the two schemas inside
    `runs_semantic_fair/` would break every reader that treats that directory
    as episode rows. The per-episode rows for the same two arms are written
    separately by `build_free_resolver_rows`, from the same resolvers.
    """
    corrupted = pd.Series([e["corrupted_string"] for e in episodes])
    true_action = pd.Series([e["true_action"] for e in episodes])
    confidence = pd.Series([float(e["confidence"]) for e in episodes])
    frames = []
    for cell in FREE_CELLS:
        curve = resolver_gate_curve(corrupted, true_action, confidence, RESOLVERS[cell.name],
                                    threshold_grid=THRESHOLD_GRID)
        curve.insert(0, "arm_name", cell.name)
        frames.append(curve)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESOLVER_CURVES, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-manifest", action="store_true",
                    help="build and write semantic_fair_comparison_manifest.json; no cost, then exit")
    ap.add_argument("--models", default=None, help="comma-separated subset of the manifest panel")
    ap.add_argument("--cells", default=None, help="comma-separated subset of the paid arms")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.build_manifest:
        return cmd_build_manifest()

    manifest = json.loads(MANIFEST.read_text())
    budget = manifest.get("budget_usd")
    if budget is None:
        print("budget_usd is null in the manifest. Only a human sets it. Refusing to run.",
              file=sys.stderr)
        return 1
    budget = float(budget)

    fm = _load_committed_manifest()

    # `_load_committed_manifest` proves the manifest matches ITSELF. It cannot
    # prove the manifest still describes the code about to run: editing
    # LEXICAL_MAX_DISTANCE from 2 to 3, or dropping an arm from LLM_CELLS, runs
    # a different experiment against an unchanged and still perfectly valid
    # manifest. Every file-side tamper is caught earlier by the git-clean check;
    # this is the code side, which nothing else looks at.
    declared = {
        "threshold_grid_n": fm["threshold_grid"]["n"],
        "arms_to_run": fm["arms_to_run"],
        "free_arms": fm["free_arms"],
        "lexical_max_distance": fm["lexical_max_distance"],
    }
    live = {
        "threshold_grid_n": int(THRESHOLD_GRID.size),
        "arms_to_run": [c.name for c in LLM_CELLS],
        "free_arms": [c.name for c in FREE_CELLS],
        "lexical_max_distance": LEXICAL_MAX_DISTANCE,
    }
    if declared != live:
        print(f"the code has drifted from the frozen declaration.\n"
              f"  frozen: {declared}\n  live:   {live}\n"
              "Re-freeze the manifest (delete it, --build-manifest, commit) before running.",
              file=sys.stderr)
        return 1
    # Count, min and max agreeing does not pin the SPACING -- linspace(0,1,101)
    # and a 101-point geometric grid share all three -- so the grid is compared
    # elementwise, not by its summary.
    if not np.allclose(np.asarray(fm["threshold_grid"]["values"], dtype=float), THRESHOLD_GRID):
        print("the live THRESHOLD_GRID does not match the frozen grid elementwise. "
              "Re-freeze the manifest before running.", file=sys.stderr)
        return 1

    episodes, _ = _load_source_episodes()
    if len(episodes) != fm["episode_source"]["n_episodes"]:
        print("the source manifest's episode count no longer matches what this experiment froze",
              file=sys.stderr)
        return 1

    panel = [m["slug"] for m in manifest["model_panel"]["models"]]
    per_ep_projection = {m["slug"]: m["projected_usd_per_episode"]
                         for m in manifest["model_panel"]["models"]}
    declared_panel = [m["slug"] for m in fm["model_panel_at_declaration"]]
    if panel != declared_panel:
        # The panel is part of the frozen declaration. A model added or removed
        # after freezing changes what the experiment is, so it must be
        # re-declared and re-committed rather than silently absorbed.
        print(f"the live panel {panel} differs from the frozen declaration {declared_panel}. "
              "Re-freeze the manifest (delete it, --build-manifest, commit) before running.",
              file=sys.stderr)
        return 1
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in panel]
        if unknown:
            print(f"not in the manifest panel: {unknown}\npanel is {panel}", file=sys.stderr)
            return 1
        panel = wanted

    cells = list(LLM_CELLS)
    if args.cells:
        wanted = {n.strip() for n in args.cells.split(",") if n.strip()}
        unknown = wanted - {c.name for c in cells}
        if unknown:
            print(f"unknown paid arm(s): {sorted(unknown)}. Known: {[c.name for c in cells]}",
                  file=sys.stderr)
            return 1
        cells = [c for c in cells if c.name in wanted]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    already = run08.spent_so_far()

    units = [(m, c) for m in panel for c in cells]
    pending = []
    for m, c in units:
        f = RUNS_DIR / f"{run08._safe(m)}__{run08._safe(c.name)}.parquet"
        if not f.exists():
            pending.append((m, c, None))
            continue
        prev = pd.read_parquet(f)
        done = set(prev.loc[prev["error"].isna(), "episode_id"])
        owed = [e["episode_id"] for e in episodes if e["episode_id"] not in done]
        if owed:
            pending.append((m, c, set(owed)))

    n_repair = sum(1 for _, _, o in pending if o is not None)
    # Indexed, not `.get(m, 0.0)`. A slug missing from the projection table is a
    # model whose cost this loop cannot price, and defaulting it to zero in the
    # one place that decides whether a run may start would silently understate
    # the projection by exactly the models it knows least about.
    projected = sum(per_ep_projection[m] * (len(o) if o else len(episodes))
                    for m, _, o in pending)
    print(f"paid arms:  {[c.name for c in cells]}")
    print(f"free arms:  {[c.name for c in FREE_CELLS]} (zero API calls, computed once)")
    print(f"panel:      {panel}")
    print(f"episodes:   {len(episodes)} reused verbatim from naturalistic_manifest.json "
          f"(digest {fm['episode_source']['natural_manifest_digest'][:16]}...)")
    print(f"units:      {len(pending)} pending of {len(units)} "
          f"({len(units) - len(pending)} complete, {n_repair} needing repair)")
    print(f"budget:     ${budget:.2f} GLOBAL ceiling, ${already:.4f} already spent (every run "
          f"directory), ~${projected:.2f} projected for this run")
    if already + projected > budget:
        print("PROJECTED SPEND EXCEEDS THE GLOBAL CEILING. Refusing to start.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("(dry run -- no calls made)")
        return 0

    # Checked BEFORE the free arms are written: a missing key is going to abort
    # this run either way, and it should do so without having first mutated
    # output/ with half an experiment's worth of artefacts.
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    free_paths = {c.name: RUNS_DIR / f"{run08._safe(c.name)}.parquet" for c in FREE_CELLS}
    for cell in FREE_CELLS:
        path = free_paths[cell.name]
        if path.exists():
            continue
        df = pd.DataFrame(build_free_resolver_rows(cell, episodes))
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        print(f"wrote free arm {cell.name}: coverage={df['covered'].mean():.3f} "
              f"faithful={df['faithful'].mean():.3f} -> {path.name}")
    curves = write_resolver_curves(episodes)
    print(f"wrote {len(curves)} swept curve rows -> {RESOLVER_CURVES.name}")

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
            print(f"  pinned {m:<32} tag={ep['tag']!r}  (OVERRIDE, resolver chose another)")
        else:
            print(f"  pinned {m:<32} tag={ep['tag']!r}")
        endpoints[m] = ep

    limiters = {m: run08.RateLimiter(_rpm_for(m)) for m in panel}
    for m in panel:
        rpm = _rpm_for(m)
        if rpm:
            src = "re-measured 2026-09-10" if m in RPM_OVERRIDE else "account cap"
            print(f"  paced  {m:<32} {rpm} requests/min ({src})")

    t0 = time.time()
    spend = already
    for n, (model, cell, owed) in enumerate(pending, 1):
        out = RUNS_DIR / f"{run08._safe(model)}__{run08._safe(cell.name)}.parquet"
        if spend >= budget:
            print(f"\nBUDGET CEILING ${budget:.2f} REACHED at ${spend:.4f}. Stopping cleanly; "
                  f"{len(pending) - n + 1} unit(s) unrun.", file=sys.stderr)
            return 2
        conc = run08.MODEL_CONCURRENCY.get(model, args.concurrency)

        # The rows this cell already has and is keeping -- read ONCE, before
        # the run, because every incremental flush below has to merge against
        # them and re-reading a file we are ourselves rewriting would be both
        # slower and a way to read back a half-written state.
        prev_ok = None
        if owed is not None and out.exists():
            prev = pd.read_parquet(out)
            prev_ok = prev[prev["error"].isna()]
        order = {e["episode_id"]: i for i, e in enumerate(episodes)}

        def _write(rows_so_far, _out=out, _prev_ok=prev_ok, _order=order):
            df = pd.DataFrame(rows_so_far)
            if _prev_ok is not None and len(_prev_ok):
                df = pd.concat([_prev_ok, df], ignore_index=True)
            df = df.sort_values("episode_id", key=lambda s: s.map(_order)).reset_index(drop=True)
            # Atomic: write beside the target, then rename, so a reader (or a
            # kill) never observes a truncated parquet. Matters far more now
            # that this runs every BATCH_SIZE episodes rather than once.
            tmp = _out.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, _out)
            return df

        rows, cost = run_fair_cell(cell, model, endpoints.get(model), episodes, key,
                                   concurrency=conc, only=owed, limiter=limiters.get(model),
                                   on_batch=_write)
        df = _write(rows)
        spend += cost
        ok = df["error"].isna()
        cov = df.loc[ok, "covered"]
        print(f"[{n:>3}/{len(pending)}] {model:<30} {cell.name:<28} "
              f"cov={cov.mean() if len(cov) else float('nan'):.2f} "
              f"faith={df.loc[ok, 'faithful'].mean() if ok.any() else float('nan'):.2f} "
              f"FAILED={int((~ok).sum())} ${cost:.4f} (cum ${spend:.4f})")

    print(f"\nDONE. {len(pending)} unit(s) in {(time.time() - t0) / 60:.1f} min. "
          f"Total measured GLOBAL spend ${spend:.4f} of ${budget:.2f}.")
    print(f"Checkpoints: {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
