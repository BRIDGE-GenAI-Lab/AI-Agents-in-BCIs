"""TASK 20: the naturalistic semantic-action benchmark -- 200 episodes built
from real donor episodes, three arms, five models, plus three free
deterministic comparators.

STATUS: reviewer-motivated follow-up robustness experiment, designed after
inspection of the initial benchmark in response to an identified
ecological-validity concern, and frozen (`--build-manifest`, committed)
before any follow-up model execution. It is NOT part of the original
pre-specified experiment and the manuscript must label it that way.

WHY THIS EXISTS. The frozen dense codebook (`nag.taxonomy`) maps a string to
an action by SHA-256 modulo nine: a one-character decoding error sends the
entailed action essentially anywhere (88.5% of error-bearing episodes change
their entailed action, close to the 8/9 expected under independence). That
is excellent for isolating uncertainty gating, but it deliberately destroys
the lexical structure a language model might otherwise exploit -- the
strongest standing objection to the rest of this study. This benchmark
answers it with nine ordinary natural-language commands
(`nag.naturalistic.NATURAL_COMMANDS`) mapped one-to-one onto the SAME nine
actions, so the action space and tier structure are unchanged and only the
string-to-action mapping differs.

THE DESIGN TRAP THIS SCRIPT MUST NOT FALL INTO: repair belongs to the MODEL,
never the tool. `nag.naturalistic.canonical_action` (what the agent's
`lookup_action` resolves through -- see `nag.naturalistic.
NaturalisticEnvironment`) recognises EXACT commands only; it is `lexical_
resolve` -- deliberately kept OUT of the tool surface -- that is the
DETERMINISTIC COMPARATOR the model has to beat. Confusing which of these two
belongs behind the agent's tool call is exactly the mistake this design was
reviewed to prevent; see `nag.naturalistic`'s module docstring.

TWO-STEP PROCESS, MIRRORING `13_principal_run.py`'s declare/run split:

  1) `--build-manifest` -- constructs the 200 episodes and every corruption
     parameter, writes `output/tables/naturalistic_manifest.json`, and exits.
     Makes NO api call and spends NOTHING. Refuses to rebuild a manifest
     that is already committed unchanged at HEAD (redoing the draw after
     seeing anything would be exactly the post-hoc selection this two-step
     process exists to prevent).

       git add output/tables/naturalistic_manifest.json
       git commit -m "manifest: freeze Task 20's 200-episode naturalistic benchmark"

  2) the actual run (no `--build-manifest`) -- REFUSES to start unless that
     manifest file (a) exists, (b) is tracked by git, (c) has no uncommitted
     diff against HEAD, and (d) its embedded episodes still hash to its own
     recorded `natural_manifest_digest`. All four are enforced in code, not
     merely operator discipline -- the same posture `13_principal_run.py`
     takes toward `principal_run.declared_at_commit`.

DONOR SELECTION, FROZEN EXACTLY (see the Task 20 plan for the full
rationale). 68 error-bearing + 132 clean donor episodes, drawn WITHOUT
replacement within those two strata from the SAME frozen 1,065-episode
primary-eligible pool `13_principal_run.py` already declared and committed
(`run_manifest.json["principal_run"]`) -- this script does not draw from a
fresh sample, it reuses that committed declaration, under seed 20260830.
From each donor: `participant_id`, the RECALIBRATED (Task 19)
`isotonic_episode` out-of-fold confidence, and the donor's own observed
character-level error pattern (position/true-char/decoded-char) are carried
into the manifest for provenance -- the actual corruption applied to the
assigned command uses the donor's error COUNT (0-3) but draws fresh
positions and empirically-parameterized substitution characters (see below),
never the donor's own specific 3-character substitution replayed verbatim.

COMMAND ASSIGNMENT is independent of donor error status: the 200 donor ids
(both strata already drawn) are shuffled together with the SAME seeded
generator BEFORE being sliced into the nine commands' near-balanced
schedule (23/23/22/22/22/22/22/22/22 in `NATURAL_COMMANDS` order), so error
status and command identity are not confounded.

CORRUPTION is empirically parameterized, never fabricated: for a donor with
k errors, k of the assigned command's alphabetic positions are drawn
uniformly (seeded, reproducible per donor via a sha256-derived integer seed
-- never Python's salted `hash()`, matching `08_run.py`'s `_cell_seed`
convention), and the substituted character at each position is drawn from
the EMPIRICAL confusion distribution estimated from real BigP3BCI ALS
decoder errors on disk (`online_trials_all20.csv`, eligible ALS rows,
target != selected, both characters alphabetic -- 464 such pairs, covering
23 of 26 letters as a true character, including all 19 letters that actually
appear in `NATURAL_COMMANDS`). The COLLISION RULE: if a candidate corruption
exactly equals a DIFFERENT valid command, redraw the substitution (fresh
positions AND characters) up to 10 attempts, then fall back to a fixed
different-position corruption with no further collision-checking. This never
actually triggers for this 9-command vocabulary (see
`corruption.n_collision_fallbacks` in the frozen manifest, expected 0) but is
implemented per the plan regardless.

BOTH the model's prompt (for the two advisory arms and, per `nag.prompts.
build_system`, withheld for the enforced arm) AND the deterministic gate use
the RECALIBRATED confidence -- reusing the product score here would rebuild
inside this benchmark exactly the calibration asymmetry Task 19 exists to
remove from the main comparison.

THREE FREE DETERMINISTIC COMPARATORS run alongside the three paid LLM arms,
built by reusing `nag.controllers.nonllm_gate` / `random_gate` UNCHANGED
(they act on whatever `Action` they are handed and never inspect a string
themselves -- see that module's docstring) with two different resolvers
supplying the `Action`:

  natural_confidence_gate_canonical  -- gate + `canonical_action` (no
    repair): the naturalistic-domain analogue of the main study's
    `nonllm_gate`. Corrupted commands are simply unresolvable through it.
  natural_random_gate_canonical      -- gate replaced by a coverage draw
    (coverage=1.0, the always-propose reference point -- see `nag.agent.
    run_episode_for_cell`'s docstring for why one run at this point is
    sufficient) + `canonical_action`: the analogue of `random_gate`.
  natural_confidence_gate_lexical    -- gate + `lexical_resolve` (DOES
    repair): the PRIMARY COMPARATOR -- "does semantic reasoning by a
    language model improve on a simple lexical resolver plus an uncertainty
    gate?" This is the system the language model has to beat; it is a
    stronger reference than the canonical-only gate precisely because it
    performs the same kind of repair a competent model could, deterministically.

All non-LLM, all excluded from the cost projection by construction (nonllm_
gate/random_gate never call an API -- see `13_principal_run.py`'s identical
exclusion).

REPORTING (matched-coverage comparison, the language model against the
lexical-resolver-plus-gate primary comparator, and whether the study's
headline survives) is the downstream analysis step's job, not this runner's.

Report the resulting absolute rates as BENCHMARK ABSOLUTE RISKS AT THE
SOURCE POOL'S OBSERVED DECODER-ERROR PREVALENCE, never as deployment
estimates -- the error distribution is empirical, but the nine commands
themselves are constructed and no participant ever sent them.

BUDGET. `spent_so_far()` is textually identical to the other three review-
response runner scripts' -- see `18_recalibrated_run.py`'s docstring. Only
the three paid LLM arms x 5 models x 200 episodes count toward the
projection; the three free comparators above are excluded by construction.

Run:
  # step 1 -- build and commit the manifest (no cost, no calls)
  UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/19_naturalistic_run.py --build-manifest
  git add output/tables/naturalistic_manifest.json
  git commit -m "manifest: freeze Task 20's 200-episode naturalistic benchmark"

  # step 2 -- dry run (no cost, no calls)
  UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/19_naturalistic_run.py --dry-run

  # step 3 -- the actual paid run
  OPENROUTER_API_KEY=... UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv UV_LINK_MODE=copy \\
  PYTHONPATH=code uv run python3 code/scripts/19_naturalistic_run.py

  # after it finishes:
  git add code/nag/naturalistic.py code/scripts/19_naturalistic_run.py tests/test_naturalistic.py output/tables/naturalistic_manifest.json output/intermediate/runs_natural/
  git commit -m "feat+run: naturalistic benchmark; model performs the repair, edit-distance resolver is the comparator"
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

from nag.agent import _as_action  # noqa: E402
from nag.controllers import nonllm_gate, random_gate  # noqa: E402
from nag.design import DEFAULT_ONLINE_CSV, build_episode_pool, episode_set_digest  # noqa: E402
from nag.episodes import ALS_STUDIES, _to_char  # noqa: E402
from nag.naturalistic import (  # noqa: E402
    COMMAND_TO_ACTION,
    NATURAL_COMMANDS,
    canonical_action,
    lexical_resolve,
    pairwise_edit_distances,
    run_naturalistic_episode,
)
from nag.openrouter import resolve_endpoint  # noqa: E402
from nag.prompts import build_system  # noqa: E402
from nag.taxonomy import TIERS, Action  # noqa: E402


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run08 = _load_sibling("_nag_run08", "08_run.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "tables" / "run_manifest.json"
NATURAL_MANIFEST = REPO_ROOT / "output" / "tables" / "naturalistic_manifest.json"
RECAL_CONF_CSV = REPO_ROOT / "output" / "tables" / "episode_confidence_per_episode.csv"
RECAL_CONF_COLUMN = "isotonic_episode"

RUNS_DIR = REPO_ROOT / "output" / "intermediate" / "runs_natural"

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
CORRUPTION_SEED = 20260830
N_ERR_DRAW, N_CLEAN_DRAW = 68, 132
N_TOTAL_DONORS = N_ERR_DRAW + N_CLEAN_DRAW
COMMAND_SCHEDULE_COUNTS = [23, 23] + [22] * 7  # NATURAL_COMMANDS order; sums to 200
LEXICAL_MAX_DISTANCE = 2
MAX_COLLISION_ATTEMPTS = 10

NATURAL_CELL_NAMES = [
    "factorial:none:advisory:s0",
    "factorial:decoder_confidence:advisory:s0",
    "factorial:decoder_confidence:enforced:s0",
]
FREE_COMPARATOR_NAMES = [
    "natural_confidence_gate_canonical",
    "natural_random_gate_canonical",
    "natural_confidence_gate_lexical",
]

assert len(COMMAND_SCHEDULE_COUNTS) == len(NATURAL_COMMANDS)
assert sum(COMMAND_SCHEDULE_COUNTS) == N_TOTAL_DONORS


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


def _stable_seed(*parts: object) -> int:
    """A sha256-derived integer seed from arbitrary parts -- never Python's
    salted `hash()` (see `08_run.py`'s `_cell_seed`, the same rationale)."""
    blob = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(blob.encode()).hexdigest()[:16], 16)


def _load_principal_frozen_ids(manifest: dict) -> tuple[list[str], str]:
    """The same guard `18_recalibrated_run.py` applies: this script never
    draws its own episode sample, it reuses `13_principal_run.py`'s already-
    committed 1,065-episode declaration, and refuses to proceed if that
    declaration is missing, unresolvable, or tampered with.
    """
    pr = manifest.get("principal_run")
    if not pr or not pr.get("declared_at_commit"):
        raise SystemExit(
            "principal_run.declared_at_commit is missing or null in run_manifest.json. "
            "This script draws donors from that already-declared 1,065-episode pool "
            "rather than declaring one of its own; run 13_principal_run.py --declare "
            "(and commit) first."
        )
    sha = pr["declared_at_commit"]
    check = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                           cwd=REPO_ROOT, capture_output=True)
    if check.returncode != 0:
        raise SystemExit(f"principal_run.declared_at_commit {sha!r} does not resolve to a "
                         "real commit in this repo.")
    frozen = pr["episode_set"]["episode_ids"]
    if episode_set_digest(frozen) != pr["episode_set"]["episode_set_digest"]:
        raise SystemExit("principal_run.episode_set.episode_ids does not match its own "
                         "recorded digest -- the manifest was edited after declaration.")
    return frozen, sha


# --- empirical confusion distribution --------------------------------------

def _build_confusion_counts() -> dict[str, dict[str, int]]:
    """Empirical character-confusion counts from real BigP3BCI ALS decoder
    errors on disk: every eligible ALS row where the decoded character
    differs from the target and BOTH are alphabetic. Never conditioned on
    the confidence-model pipeline (`selection_scores.parquet`) -- this is
    about raw channel-level substitution behaviour, the same universe
    `nag.design.build_episode_pool` starts from, filtered no further.
    """
    df = pd.read_csv(DEFAULT_ONLINE_CSV)
    d = df[df["eligible"] == True].dropna(subset=["target", "selected"])  # noqa: E712
    d = d[d["study"].isin(ALS_STUDIES)]
    counts: dict[str, dict[str, int]] = {}
    for t, s in zip(d["target"], d["selected"]):
        tc, sc = _to_char(t), _to_char(s)
        if tc is None or sc is None or tc == sc:
            continue
        if not (tc.isalpha() and sc.isalpha()):
            continue
        tl, sl = tc.lower(), sc.lower()
        counts.setdefault(tl, {}).setdefault(sl, 0)
        counts[tl][sl] += 1
    return counts


def _fallback_letters(confusion: dict, vocabulary_letters: set[str]) -> list[str]:
    """Letters actually used in NATURAL_COMMANDS with zero observed empirical
    substitutions -- recorded in the manifest so the (expected empty) uniform-
    fallback path is auditable rather than silently invoked."""
    return sorted(l for l in vocabulary_letters if not confusion.get(l))


def _sample_substitution_char(true_char: str, confusion: dict, rng: random.Random) -> str:
    row = confusion.get(true_char)
    if not row:
        # Defensive only -- see corruption.fallback_letters in the frozen
        # manifest, expected empty for this 19-letter command vocabulary.
        options = [c for c in string.ascii_lowercase if c != true_char]
        return rng.choice(options)
    options, weights = zip(*row.items())
    return rng.choices(options, weights=weights, k=1)[0]


def _confusion_corrupt(command: str, n_subs: int, donor_episode_id: str,
                       confusion: dict) -> tuple[str, list[int], int, bool]:
    """Empirically parameterized corruption: `n_subs` of `command`'s
    alphabetic positions, drawn uniformly under a seed derived from
    `CORRUPTION_SEED` and `donor_episode_id`; the replacement character at
    each position drawn from `confusion`. Returns (corrupted_string,
    positions_used, n_attempts_before_success, collision_fallback_used).

    Collision rule: if a candidate exactly equals a DIFFERENT valid command,
    redraw fresh positions AND characters, up to `MAX_COLLISION_ATTEMPTS`;
    after that, fall back to a fixed different-position corruption with no
    further collision-checking. `candidate == command` is structurally
    impossible whenever n_subs >= 1: `confusion` is built only from true !=
    decoded observations (see `_build_confusion_counts`), so every
    substituted position differs from its original character by
    construction.
    """
    if n_subs == 0:
        return command, [], 0, False
    alpha_positions = [i for i, ch in enumerate(command) if ch.isalpha()]
    if n_subs > len(alpha_positions):
        raise ValueError(f"{command!r} has only {len(alpha_positions)} alphabetic characters, "
                         f"cannot apply {n_subs} substitutions (donor {donor_episode_id!r})")
    for attempt in range(MAX_COLLISION_ATTEMPTS):
        rng = random.Random(_stable_seed(CORRUPTION_SEED, donor_episode_id, "attempt", attempt))
        positions = sorted(rng.sample(alpha_positions, n_subs))
        chars = list(command)
        for pos in positions:
            chars[pos] = _sample_substitution_char(command[pos], confusion, rng)
        candidate = "".join(chars)
        if candidate not in COMMAND_TO_ACTION:
            return candidate, positions, attempt, False
        # else: candidate collided with a DIFFERENT valid command -- redraw.
    fallback_positions = sorted(alpha_positions[:n_subs])
    rng = random.Random(_stable_seed(CORRUPTION_SEED, donor_episode_id, "fallback"))
    chars = list(command)
    for pos in fallback_positions:
        chars[pos] = _sample_substitution_char(command[pos], confusion, rng)
    return "".join(chars), fallback_positions, MAX_COLLISION_ATTEMPTS, True


# --- manifest construction (Task 20 step 4-5) -------------------------------

def _episodes_digest(natural_episodes: list[dict]) -> str:
    blob = json.dumps(
        [[e["episode_id"], e["assigned_command"], e["corrupted_string"]] for e in natural_episodes],
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def build_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    frozen_ids, declared_sha = _load_principal_frozen_ids(manifest)

    pool = build_episode_pool().set_index("episode_id")
    missing_pool = [e for e in frozen_ids if e not in pool.index]
    if missing_pool:
        raise SystemExit(f"{len(missing_pool)} frozen episode(s) absent from build_episode_pool()")

    if not RECAL_CONF_CSV.exists():
        raise SystemExit(f"{RECAL_CONF_CSV} does not exist -- run 12_episode_calibration.py first")
    recal = pd.read_csv(RECAL_CONF_CSV).set_index("episode_id")
    missing_recal = [e for e in frozen_ids if e not in recal.index or pd.isna(recal.loc[e, RECAL_CONF_COLUMN])]
    if missing_recal:
        raise SystemExit(f"{len(missing_recal)} frozen episode(s) missing/null {RECAL_CONF_COLUMN} "
                         f"in {RECAL_CONF_CSV.name}")

    sub = pool.loc[frozen_ids]
    err_ids = sorted(sub.index[sub["err"]].tolist())
    clean_ids = sorted(sub.index[~sub["err"]].tolist())
    if len(err_ids) < N_ERR_DRAW or len(clean_ids) < N_CLEAN_DRAW:
        raise SystemExit(f"source pool too small for the frozen {N_ERR_DRAW}/{N_CLEAN_DRAW} "
                         f"stratified draw (have {len(err_ids)} error-bearing, {len(clean_ids)} clean)")

    rng = np.random.default_rng(CORRUPTION_SEED)
    err_draw = rng.choice(np.array(err_ids, dtype=object), size=N_ERR_DRAW, replace=False).tolist()
    clean_draw = rng.choice(np.array(clean_ids, dtype=object), size=N_CLEAN_DRAW, replace=False).tolist()
    donor_ids = list(err_draw) + list(clean_draw)
    rng.shuffle(donor_ids)  # mixes error/clean strata BEFORE command assignment

    assignment: dict[str, str] = {}
    idx = 0
    for command, n in zip(NATURAL_COMMANDS, COMMAND_SCHEDULE_COUNTS):
        for did in donor_ids[idx: idx + n]:
            assignment[did] = command
        idx += n
    assert idx == len(donor_ids) == N_TOTAL_DONORS

    confusion = _build_confusion_counts()
    vocab_letters = {ch for cmd in NATURAL_COMMANDS for ch in cmd if ch.isalpha()}
    fallback_letters = _fallback_letters(confusion, vocab_letters)

    natural_episodes = []
    n_collision_fallbacks = 0
    total_attempts = 0
    for i, did in enumerate(donor_ids):
        drow = sub.loc[did]
        command = assignment[did]
        n_subs = int(drow["n_errors"])
        corrupted, positions, attempts, fallback = _confusion_corrupt(command, n_subs, did, confusion)
        total_attempts += attempts
        if fallback:
            n_collision_fallbacks += 1
        true_s, dec_s = str(drow["true_string"]), str(drow["decoded_string"])
        donor_error_positions = [
            {"position": p, "true_char": true_s[p], "decoded_char": dec_s[p]}
            for p in range(len(true_s)) if true_s[p] != dec_s[p]
        ]
        true_action = COMMAND_TO_ACTION[command]
        is_error_bearing = corrupted != command
        assert is_error_bearing == (n_subs > 0)  # corruption always changes the string when n_subs >= 1
        natural_episodes.append(dict(
            index=i,
            episode_id=did,               # the natural episode's key IS the donor's id -- 1:1, never reused
            donor_episode_id=did,
            participant_id=str(drow["participant_id"]),
            study=str(drow["study"]),
            donor_n_errors=n_subs,
            donor_err=bool(drow["err"]),
            donor_error_positions=donor_error_positions,
            confidence=float(recal.loc[did, RECAL_CONF_COLUMN]),   # RECALIBRATED -- shown to the model AND the gate
            confidence_product=float(drow["confidence"]),           # traceability only, never used here
            assigned_command=command,
            true_action=true_action,
            tier=int(TIERS[true_action]),
            corrupted_string=corrupted,
            n_substitutions=n_subs,
            substitution_positions=positions,
            collision_attempts=attempts,
            collision_fallback_used=fallback,
            is_error_bearing=bool(is_error_bearing),
        ))

    realized_err = sum(1 for e in natural_episodes if e["is_error_bearing"])
    assert realized_err == N_ERR_DRAW

    edit_distances = [
        {"a": a, "b": b, "distance": d} for (a, b), d in pairwise_edit_distances().items()
    ]

    out = {
        "task": "Task 20: naturalistic semantic-action benchmark",
        "status": (
            "reviewer-motivated follow-up robustness experiment; NOT part of the original "
            "pre-specified experiment (see nag.naturalistic module docstring and the Task 20 plan)"
        ),
        "frozen_before_run_note": (
            "this file must be committed to git BEFORE any model request is issued -- the run "
            "step refuses to start unless it is tracked, has no uncommitted diff against HEAD, "
            "and its embedded episodes hash to natural_manifest_digest below"
        ),
        "donor_pool": {
            "source": "run_manifest.json['principal_run']['episode_set']['episode_ids']",
            "source_declared_at_commit": declared_sha,
            "n_total": len(frozen_ids),
            "n_error_bearing": len(err_ids),
            "n_clean": len(clean_ids),
            "source_error_prevalence": len(err_ids) / len(frozen_ids),
        },
        "draw": {
            "seed": CORRUPTION_SEED,
            "n_error_bearing_drawn": N_ERR_DRAW,
            "n_clean_drawn": N_CLEAN_DRAW,
            "method": (
                "np.random.default_rng(20260830); error-bearing donor ids drawn first via "
                "rng.choice(sorted(err_ids), 68, replace=False), then clean donor ids via "
                "rng.choice(sorted(clean_ids), 132, replace=False) from the SAME generator, both "
                "without replacement; donor_ids = err_draw + clean_draw, then rng.shuffle(donor_ids) "
                "in place (still the same generator) to mix strata before command assignment"
            ),
            "realized_prevalence": {"n_error_bearing": realized_err, "n_total": len(natural_episodes),
                                    "frac": realized_err / len(natural_episodes)},
        },
        "commands": dict(COMMAND_TO_ACTION),
        "command_assignment": {
            "schedule": dict(zip(NATURAL_COMMANDS, COMMAND_SCHEDULE_COUNTS)),
            "order_note": "NATURAL_COMMANDS order; first 2 commands get 23 donors, remaining 7 get 22",
            "independence_note": (
                "donor_ids shuffled (mixing error/clean strata) BEFORE being sliced by command, so "
                "command assignment is independent of donor error status by construction"
            ),
        },
        "pairwise_edit_distances": edit_distances,
        "corruption": {
            "algorithm": (
                "for a donor with k errors (k = the donor's OWN n_errors, 0-3, from the frozen "
                "codebook episode it was drawn from), k of the assigned command's alphabetic "
                "positions are drawn uniformly under a sha256-derived seed "
                "(CORRUPTION_SEED, donor_episode_id, attempt), and the replacement character at "
                "each position is drawn from the empirical confusion distribution below, weighted "
                "by observed count"
            ),
            "confusion_source": (
                "online_trials_all20.csv, ALS_STUDIES only, eligible rows, target != selected, "
                "both characters alphabetic -- the same universe nag.design.build_episode_pool "
                "starts from, filtered no further"
            ),
            "confusion_n_pairs": sum(len(v) for v in confusion.values()),
            "confusion_n_observations": sum(sum(v.values()) for v in confusion.values()),
            "confusion_counts": confusion,
            "fallback_rule": "uniform over the 25 other lowercase letters if a true character has no observed substitution",
            "fallback_letters_used_by_natural_commands": fallback_letters,
            "collision_rule": (
                "if a candidate corruption exactly equals a DIFFERENT valid command, redraw fresh "
                f"positions and characters, up to {MAX_COLLISION_ATTEMPTS} attempts; after that, "
                "fall back to a fixed different-position corruption with no further "
                "collision-checking"
            ),
            "n_collision_fallbacks": n_collision_fallbacks,
            "mean_attempts_to_success": total_attempts / len(natural_episodes),
        },
        "confidence_used": (
            "the RECALIBRATED episode-level out-of-fold isotonic score from Task 19 "
            f"({RECAL_CONF_CSV.name}, column {RECAL_CONF_COLUMN!r}) is used by BOTH the model's "
            "prompt (advisory arms) and the deterministic gate; the product score is kept only as "
            "confidence_product, for traceability, and is never rendered or gated on here"
        ),
        "max_distance": LEXICAL_MAX_DISTANCE,
        "ambiguity_rule": (
            "lexical_resolve returns None (abstain) when the nearest command exceeds max_distance, "
            "OR when two or more commands tie at the minimum distance -- never a silent tie-break"
        ),
        "arms_to_run": list(NATURAL_CELL_NAMES),
        "free_comparators": {
            "natural_confidence_gate_canonical": "nonllm_gate + canonical_action (no repair) -- the naturalistic analogue of the main study's nonllm_gate",
            "natural_random_gate_canonical": "random_gate (coverage=1.0 reference point) + canonical_action -- the analogue of random_gate",
            "natural_confidence_gate_lexical": "nonllm_gate + lexical_resolve (DOES repair) -- the PRIMARY COMPARATOR the language model has to beat",
        },
        "primary_comparator": "natural_confidence_gate_lexical",
        "analysis_endpoints": [
            "matched-coverage unfaithful-execution comparison, same definition as the main study's primary endpoint",
            "language model vs. natural_confidence_gate_lexical (the primary comparator) at matched coverage",
            "whether the study's headline (no arm beats the deterministic gate at matched coverage) survives in an environment where language context IS available",
        ],
        "reporting_note": (
            "absolute rates here are BENCHMARK ABSOLUTE RISKS AT THE SOURCE POOL'S OBSERVED "
            "decoder-error prevalence, never deployment estimates -- the error distribution is "
            "empirical, the nine commands are constructed and no participant ever sent them"
        ),
        "n_models": 5,
        "natural_episodes": natural_episodes,
        "natural_manifest_digest": _episodes_digest(natural_episodes),
    }
    return out


def _manifest_git_state() -> tuple[bool, bool, bool]:
    """(exists_on_disk, tracked_by_git, no_uncommitted_diff)."""
    if not NATURAL_MANIFEST.exists():
        return False, False, False
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(NATURAL_MANIFEST)],
                             cwd=REPO_ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(NATURAL_MANIFEST)],
                           cwd=REPO_ROOT, capture_output=True).returncode == 0
    return True, tracked, clean


def cmd_build_manifest() -> int:
    exists, tracked, clean = _manifest_git_state()
    if exists and tracked and clean:
        print(f"{NATURAL_MANIFEST} already exists, is tracked, and matches HEAD -- already frozen. "
              "Refusing to rebuild (redoing the draw after freezing would defeat the point of "
              "freezing it). Remove it by hand first if this is genuinely a redo.", file=sys.stderr)
        return 1
    m = build_manifest()
    NATURAL_MANIFEST.write_text(json.dumps(m, indent=2) + "\n")
    print(f"wrote {NATURAL_MANIFEST}")
    print(f"donors: {N_ERR_DRAW} error-bearing + {N_CLEAN_DRAW} clean = {N_TOTAL_DONORS}")
    print(f"realized error prevalence: {m['draw']['realized_prevalence']['frac']:.4f} "
          f"({m['draw']['realized_prevalence']['n_error_bearing']}/{m['draw']['realized_prevalence']['n_total']})")
    print(f"collision fallbacks: {m['corruption']['n_collision_fallbacks']} "
          f"(expected 0 for this 9-command vocabulary)")
    print(f"natural_manifest_digest: {m['natural_manifest_digest']}")
    print("\n--- COMMIT NOW, before any request: ---")
    print("  git add output/tables/naturalistic_manifest.json")
    print('  git commit -m "manifest: freeze Task 20\'s 200-episode naturalistic benchmark"')
    return 0


def _load_committed_manifest() -> dict:
    exists, tracked, clean = _manifest_git_state()
    if not exists:
        raise SystemExit(f"{NATURAL_MANIFEST} does not exist. Run --build-manifest first, then commit it.")
    if not tracked:
        raise SystemExit(f"{NATURAL_MANIFEST} exists but is not tracked by git. Commit it before running.")
    if not clean:
        raise SystemExit(f"{NATURAL_MANIFEST} has uncommitted changes relative to HEAD. The manifest "
                         "must be committed BEFORE any request is issued. Commit it (or revert the "
                         "working-tree change) first.")
    m = json.loads(NATURAL_MANIFEST.read_text())
    recomputed = _episodes_digest(m["natural_episodes"])
    if recomputed != m["natural_manifest_digest"]:
        raise SystemExit("naturalistic_manifest.json's natural_episodes do not match its own recorded "
                         "natural_manifest_digest -- the file was edited after being written.")
    return m


# --- the actual run (Task 20 step 6) ----------------------------------------

def run_natural_cell(cell, model, endpoint, natural_episodes, key, seed, concurrency=1, only=None, limiter=None):
    """`08_run.py`'s `run_cell`, adapted to the naturalistic domain: calls
    `nag.naturalistic.run_naturalistic_episode` instead of `nag.agent.
    run_episode_for_cell` (the codebook dispatcher has no notion of a
    corrupted natural-language command), and the episode dict shape is this
    benchmark's own (see `build_manifest`'s `natural_episodes` entries)
    rather than `nag.design.build_episode_pool`'s. Same per-episode-own-
    client concurrency, same input-order-preserving `.map`, same cost
    accounting as `run08.run_cell`.
    """
    def one(indexed_ep):
        i, ep = indexed_ep
        client = run08.PinnedClient(model, endpoint, key, limiter) if endpoint is not None else None
        conf = float(ep["confidence"])
        row = {
            "model": model, "cell": cell.name,
            "uncertainty_source": cell.uncertainty_source, "control_mechanism": cell.control_mechanism,
            "scaffold": cell.scaffold, "wording": cell.wording, "uses_llm": cell.uses_llm,
            "episode_id": ep["episode_id"], "donor_episode_id": ep["donor_episode_id"],
            "participant_id": ep["participant_id"], "study": ep["study"],
            "assigned_command": ep["assigned_command"], "corrupted_string": ep["corrupted_string"],
            "true_action": ep["true_action"], "n_substitutions": int(ep["n_substitutions"]),
            "is_error_bearing": bool(ep["is_error_bearing"]), "tier": int(ep["tier"]),
            "confidence": conf, "confidence_product": float(ep["confidence_product"]),
            "provider_tag": endpoint["tag"] if endpoint else None,
            "quantization": endpoint["quantization"] if endpoint else None,
        }
        try:
            system = build_system(cell, confidence=conf)
            rec = run_naturalistic_episode(cell=cell, episode=ep, confidence=conf, client=client, system=system)
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

    items = [(i, e) for i, e in enumerate(natural_episodes) if only is None or e["episode_id"] in only]
    if concurrency <= 1 or endpoint is None:
        rows = [one(it) for it in items]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, items))
    return rows, sum(r["cost_usd"] for r in rows)


def build_free_comparators(natural_episodes: list[dict]) -> dict[str, pd.DataFrame]:
    """The three free deterministic comparators -- see the module docstring.
    Pure reuse of `nag.controllers.nonllm_gate` / `random_gate`; the only
    thing that varies between them is which resolver supplies the `Action`.
    """
    rows = {name: [] for name in FREE_COMPARATOR_NAMES}
    for i, ep in enumerate(natural_episodes):
        conf = float(ep["confidence"])
        want = (ep["true_action"], ())
        base = dict(episode_id=ep["episode_id"], donor_episode_id=ep["donor_episode_id"],
                    participant_id=ep["participant_id"], assigned_command=ep["assigned_command"],
                    corrupted_string=ep["corrupted_string"], true_action=ep["true_action"],
                    is_error_bearing=ep["is_error_bearing"], confidence=conf)

        canon_name = canonical_action(ep["corrupted_string"])
        canon_action = Action(name=canon_name, args=(), tier=TIERS[canon_name]) if canon_name else None
        out = nonllm_gate(conf, float("-inf"), canon_action)
        got = _as_action(out["executed"])
        rows["natural_confidence_gate_canonical"].append({
            **base, "covered": bool(out["covered"]), "faithful": bool(out["covered"] and got == want),
            "executed_name": (out["executed"] or {}).get("name"), "cost_usd": 0.0, "error": None,
        })

        rgen = np.random.default_rng([SEED, i])
        out_r = random_gate(rgen, 1.0, canon_action)
        got_r = _as_action(out_r["executed"])
        rows["natural_random_gate_canonical"].append({
            **base, "covered": bool(out_r["covered"]), "faithful": bool(out_r["covered"] and got_r == want),
            "executed_name": (out_r["executed"] or {}).get("name"), "cost_usd": 0.0, "error": None,
        })

        lex_name = lexical_resolve(ep["corrupted_string"], max_distance=LEXICAL_MAX_DISTANCE)
        lex_action = Action(name=lex_name, args=(), tier=TIERS[lex_name]) if lex_name else None
        out_l = nonllm_gate(conf, float("-inf"), lex_action)
        got_l = _as_action(out_l["executed"])
        rows["natural_confidence_gate_lexical"].append({
            **base, "covered": bool(out_l["covered"]), "faithful": bool(out_l["covered"] and got_l == want),
            "executed_name": (out_l["executed"] or {}).get("name"), "cost_usd": 0.0, "error": None,
        })
    return {name: pd.DataFrame(r) for name, r in rows.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-manifest", action="store_true",
                    help="build and write naturalistic_manifest.json; no cost, no calls, then exit")
    ap.add_argument("--models", default=None, help="comma-separated subset of the manifest panel")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.build_manifest:
        return cmd_build_manifest()

    manifest = json.loads(MANIFEST.read_text())
    budget = manifest.get("budget_usd")
    if budget is None:
        print("budget_usd is null in the manifest. Refusing to run.", file=sys.stderr)
        return 1
    budget = float(budget)

    nm = _load_committed_manifest()
    natural_episodes = nm["natural_episodes"]

    from nag.design import enumerate_cells
    all_cells = {c.name: c for c in enumerate_cells()}
    missing_cells = [n for n in NATURAL_CELL_NAMES if n not in all_cells]
    if missing_cells:
        print(f"cell(s) not found in enumerate_cells(): {missing_cells}", file=sys.stderr)
        return 1
    cells = [all_cells[n] for n in NATURAL_CELL_NAMES]

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

    units = [(m, c) for m in panel for c in cells]
    pending = []
    for m, c in units:
        f = RUNS_DIR / f"{run08._safe(m)}__{run08._safe(c.name)}.parquet"
        if not f.exists():
            pending.append((m, c, None))
            continue
        prev = pd.read_parquet(f)
        done = set(prev.loc[prev["error"].isna(), "episode_id"])
        owed = [e["episode_id"] for e in natural_episodes if e["episode_id"] not in done]
        if owed:
            pending.append((m, c, set(owed)))

    n_repair = sum(1 for _, _, o in pending if o is not None)
    projected = sum(per_ep_projection.get(m, 0.0) * (len(o) if o else len(natural_episodes))
                    for m, _, o in pending)
    err_frac = nm["draw"]["realized_prevalence"]["frac"]
    print(f"arms:       {NATURAL_CELL_NAMES}")
    print(f"free:       {FREE_COMPARATOR_NAMES} (no cost, computed once)")
    print(f"panel:      {panel}")
    print(f"episodes:   {len(natural_episodes)} (frozen naturalistic benchmark, "
          f"naturalistic_manifest.json) -- realized error prevalence {err_frac:.4f} "
          f"({nm['draw']['realized_prevalence']['n_error_bearing']}/"
          f"{nm['draw']['realized_prevalence']['n_total']})")
    print(f"units:      {len(pending)} pending of {len(units)} "
          f"({len(units) - len(pending)} complete, {n_repair} needing repair)")
    print(f"budget:     ${budget:.2f} GLOBAL ceiling, ${already:.4f} already spent (all "
          f"checkpoints, every run directory), ~${projected:.2f} projected for this run")
    if already + projected > budget:
        print("PROJECTED SPEND EXCEEDS THE GLOBAL CEILING. Refusing to start.", file=sys.stderr)
        return 1

    donor_counts = {}
    for e in natural_episodes:
        donor_counts[e["donor_episode_id"]] = donor_counts.get(e["donor_episode_id"], 0) + 1
    dup_donors = [d for d, n in donor_counts.items() if n > 1]
    if dup_donors:
        print(f"REFUSING: {len(dup_donors)} donor(s) drawn more than once: {dup_donors[:5]}", file=sys.stderr)
        return 1
    print(f"donor uniqueness: {len(donor_counts)} distinct donors for {len(natural_episodes)} episodes -- OK")

    if args.dry_run:
        print("(dry run -- no calls made)")
        return 0

    free_paths = {name: RUNS_DIR / f"{run08._safe(name)}.parquet" for name in FREE_COMPARATOR_NAMES}
    free_missing = [name for name, p in free_paths.items() if not p.exists()]
    if free_missing:
        # Computed once for all three (no cost, deterministic) rather than
        # recomputing per missing name -- see build_free_comparators.
        dfs = build_free_comparators(natural_episodes)
        for name in free_missing:
            tmp = free_paths[name].with_suffix(".parquet.tmp")
            dfs[name].to_parquet(tmp, index=False)
            os.replace(tmp, free_paths[name])
        print(f"wrote free comparators: {free_missing}")

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
    for n, (model, cell, owed) in enumerate(pending, 1):
        out = RUNS_DIR / f"{run08._safe(model)}__{run08._safe(cell.name)}.parquet"
        if spend >= budget:
            print(f"\nBUDGET CEILING ${budget:.2f} REACHED at ${spend:.4f}. Stopping cleanly; "
                  f"{len(pending) - n + 1} unit(s) unrun.", file=sys.stderr)
            return 2
        conc = run08.MODEL_CONCURRENCY.get(model, args.concurrency)
        rows, cost = run_natural_cell(cell, model, endpoints.get(model), natural_episodes, key, SEED,
                                      concurrency=conc, only=owed, limiter=limiters.get(model))
        df = pd.DataFrame(rows)
        if owed is not None and out.exists():
            prev = pd.read_parquet(out)
            df = pd.concat([prev[prev["error"].isna()], df], ignore_index=True)
            order = {e["episode_id"]: i for i, e in enumerate(natural_episodes)}
            df = df.sort_values("episode_id", key=lambda s: s.map(order)).reset_index(drop=True)
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
          f"Total measured GLOBAL spend ${spend:.4f} of ${budget:.2f}.")
    print(f"Checkpoints: {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
