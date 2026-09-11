"""TASK 7: the ONE table that makes the semantic fair comparison real.

Builds `output/tables/semantic_primary_comparison.csv`: one row per (arm,
model) across the five new information-symmetric arms (Task 6,
`runs_semantic_fair/`) AND the two control conditions they have to be fair
RELATIVE TO -- the original naturalistic benchmark that withheld the command
vocabulary from the model (`runs_natural/`, Task 20) and the recalibrated
hashed-codebook arms (`runs_recal/`). The controls are kept, not replaced:
"language models do no better than a threshold gate" is only interesting
alongside the condition in which they were denied the information the gate
had.

THE ONE THING THIS SCRIPT EXISTS TO GET RIGHT. Most arms on this benchmark
did not record an operating point. They recorded a PROPOSAL. Every ENFORCED
arm -- `fair:llm_vocab:enforced`, `fair:hybrid_semantic_gate`, both
`fair:*_resolver_gate` arms, and Task 20's
`factorial:decoder_confidence:enforced:s0` -- was run once at
threshold=-inf, because the model never sees a threshold and so cannot
condition on one (`25_semantic_fair_comparison.py`'s docstring, and
`nag.agent.apply_enforced_gate`'s, both say so). On those rows `covered`
means "a candidate command exists", not "this episode was admitted". The
hybrid arm sits at 0.915-1.000 there for exactly that reason. Copying that
column into a table beside an ADVISORY arm, whose row genuinely IS its final
outcome, would compare a proposal rate against an operating point and
overstate the proposal arms across the board.

So every enforced arm is threshold-swept here before it is reported, by the
same `covered & (confidence >= t)` rule for all of them:
`nag.naturalistic.resolver_gate_curve` for the two resolver arms (already
computed by Task 6 into `semantic_fair_resolver_curves.csv`, read rather than
recomputed so this table cannot drift from the run's own output) and
`nag.naturalistic.recorded_proposal_gate_curve` for the LLM arms, whose
proposals came from a paid call and cannot be replayed offline. Which arm
gets which treatment is decided by `nag.riskcoverage.curve_knob` -- the
repo's own classifier, a property of `control_mechanism` x
`uncertainty_source` (Ruling 27) -- never by a hardcoded name list.

WHAT `coverage` AND `risk` MEAN IN THIS TABLE. For a fixed-point arm they are
the arm's own outcome. For a swept arm they are its curve at
`reference_threshold = 0.0`, the maximum-coverage end of its frontier, where
the gate admits every proposal; the rest of the frontier is summarized by
`aurc` and `aurc_common`. `proposal_rate` is the raw recorded `covered` mean,
carried so a reader can see it is the same number as the swept arm's coverage
AT THRESHOLD ZERO and nothing more.

WHY `aurc_common` AND NOT JUST `aurc`. Arms reach different maximum
coverages, and comparing raw areas across unequal support is not a
comparison -- this manuscript once reported two models as beating the gate
purely because their integrals omitted the highest-risk region
(`nag.riskcoverage.common_support`'s docstring). `aurc_common` therefore
integrates this arm over the coverage window it SHARES WITH THE COMPARATOR,
and `comparator_aurc_common` integrates the comparator over that same
window, so the two areas in `aurc_common_minus_comparator` describe the same
interval. The window is pairwise, as in the existing `primary_aurc.csv`, not
panel-wide: one arm that stops early (the exact resolver tops out at 0.66)
would otherwise shrink every other arm's reported area to its own ceiling.

READ `beats_comparator` WITH `frontier_verdict`. Several arms operate at
coverage the deterministic comparator cannot reach at all, so no matched
point exists and `beats_comparator` is NaN there -- which is a finding, not a
missing value. `frontier_verdict` says in words which case each row is, so
those rows cannot be mistaken for rows that failed to evaluate.

`frontier_verdict` IS A PARETO VERDICT, NOT A RISK COMPARISON, and it has to
be. The comparator's risk is 0.0 at every one of its 101 thresholds, so a
verdict built on matched-coverage risk alone would call EVERY zero-risk arm
a "tie" at ANY coverage -- including an arm covering 0.62 while the
comparator covers 0.94 at the same zero risk, which is dominated, not tied.
So a row is only "matched" when the comparator cannot do strictly better on
either axis, and the two ways of being outside the comparator's reach --
more coverage at no more risk, versus more coverage bought with more risk --
carry different labels rather than one shared "operates above" one.

HOW THIS TABLE'S HEADLINE MUST BE STATED. Not as "0 of 33 rows beat the
deterministic resolver". That sentence is wrong twice over. It reports a
property of the STATISTIC as though it were a measurement: the comparator
records 0 unfaithful executions in all 188 episodes it admits, and no arm
can show a rate below zero observed failures, so `beats_comparator` is False
wherever it can be evaluated at all and no run could have made it otherwise.
And its denominator mixes three different things -- of the 33 rows carrying
a matched comparison, 4 are not language-model arms
(`natural_confidence_gate_lexical`, which is the comparator's own algorithm
as Task 20 ran it, `natural_confidence_gate_canonical`,
`natural_random_gate_canonical`, and `fair:exact_resolver_gate`) and 15 are
the vocabulary-WITHHELD Task 20 `factorial:*` controls, which are the
CONTROL condition rather than the treatment. The vocabulary-disclosed
fair-comparison set is the remaining 14 of those 33, out of 30 (arm, model)
cells once the rows operating above the comparator's ceiling are counted
too. The finding is on the coverage axis, and this is the vetted sentence
for it:

    "With the command vocabulary disclosed, no language-model arm reduced
    unfaithful-execution risk below the deterministic lexical resolver's
    observed risk, which was 0 in all 188 admissions (one-sided 95% upper
    bound 0.0158) -- an arm cannot show lower risk than zero observed
    failures, so this comparison cannot demonstrate an LLM arm ahead on
    risk, though it does not establish the resolver is infallible -- but 16
    of 30 (arm, model) cells acted on episodes the resolver structurally
    cannot reach. 10 of those 16 did so with no unfaithful execution
    observed anywhere (one-sided 95% upper bound 0.22-0.31 on the 8-12
    incremental episodes each covered); the other 6 bought that reach at
    the cost of some unfaithful executions, including 3 with unfaithful
    executions inside the incremental episodes themselves (hybrid /
    deepseek-v4-flash 4 of 11, hybrid / nemotron-3.5-lightning 4 of 11,
    advisory / mistral-medium-3-5 1 of 9)."

    The 10-of-16 split is not incidental: `frontier_verdict` labels a cell
    "exceeds the comparator's frontier" only when its risk is within TOL of
    the comparator's own (zero) risk at its ceiling -- so "no unfaithful
    execution observed" is guaranteed by that selection, not an independent
    finding, for those 10. The 6 EXCLUDED by that same selection are where
    the actual risk-tradeoff data lives, and the sentence above states them
    rather than omitting them.

That sentence says "observed risk" rather than "risk 0.0 by construction"
because the two are not the same comparator. `canonical_action` demands an
exact match and is faithful by construction; `lexical_resolve`, which IS the
comparator here, proposes the unique nearest command within edit distance 2
and could in principle propose a wrong one. It never did on these 188
episodes. What the design guarantees is narrower and sufficient: the
comparison cannot show a language model ahead on risk.

EVERY ZERO IN `risk` IS AN OBSERVED ZERO. 37 of 55 rows report risk 0.0 on at
most 200 episodes. `risk_upper95_one_sided` carries the one-sided 95% upper
bound beside it (about 0.015 at n=200) so the CSV alone cannot be read as a
claim that an arm is never unfaithful.

`risk_upper95_one_sided` IS THE WRONG BOUND FOR THE POSITIVE CLAIM. It is
computed over all ~200 covered episodes, but the claim the ten
frontier-exceeding cells support is about the INCREMENT -- the 8-12 episodes
each of them acts on that the comparator does not reach. On that many
episodes a zero-failure bound is 0.22-0.31, not 0.015, so quoting the
whole-arm column beside the incremental claim understates its uncertainty
by 15-20x. `n_beyond_comparator` and `risk_upper95_beyond_comparator` carry
the incremental count and its bound; use those two whenever the sentence is
about episodes the comparator cannot reach.

"HIGHER COVERAGE" IS NOT "DOES EVERYTHING THE RESOLVER DOES, AND MORE".
Coverage is a count, and an arm can outnumber the comparator while missing
episodes the comparator resolves: `fair:llm_vocab:enforced` does exactly
that on qwen3.8-max-0902 (misses 6 of the comparator's 188), on kimi-k3 (2),
on deepseek-v4-flash (1) and on grok-4.6 (1), as does
`fair:hybrid_semantic_gate` on glm-5.3-flash (1). Those trades are invisible
in `coverage` and visible in `is_strict_superset_of_comparator_coverage` and
`n_comparator_episodes_missed`, which is why both columns exist.

NOTHING FROZEN MOVES. Scripts 00-24 are untouched, Task 20's three
deterministic comparators enter as the published FIXED POINTS they were run
as (re-sweeping them would move numbers already in the manuscript; their
swept counterparts are the two new `fair:*_resolver_gate` arms), and this
script only reads.

Run:
  PYTHONPATH=code /private/tmp/nag_venv/bin/python3 code/scripts/26_semantic_primary_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

from nag.design import Cell  # noqa: E402
from nag.naturalistic import recorded_proposal_gate_curve  # noqa: E402
from nag.riskcoverage import aurc, aurc_common, common_support, curve_knob, risk_at_coverage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = REPO_ROOT / "output" / "intermediate"
RESOLVER_CURVES = REPO_ROOT / "output" / "tables" / "semantic_fair_resolver_curves.csv"
OUT = REPO_ROOT / "output" / "tables" / "semantic_primary_comparison.csv"

# Matches the frozen `threshold_grid` in
# `output/tables/semantic_fair_comparison_manifest.json`, which is also
# `resolver_gate_curve`'s default. The published resolver curves were swept on
# it, so the LLM arms must be too or their AURCs are not comparable.
THRESHOLD_GRID = np.linspace(0.0, 1.0, 101)

# The PRIMARY COMPARATOR: the deterministic system the language models have to
# beat, and the one whose privileged access to `NATURAL_COMMANDS` motivated the
# whole information-symmetry fix (see `25_semantic_fair_comparison.py`'s
# CELL_SPEC entry for this arm).
COMPARATOR_ARM = "fair:lexical_resolver_gate"

# Coverage and risk here are ratios of small integer counts, so exact equality
# would mostly work; `risk_at_coverage` interpolates, so mostly is not enough.
TOL = 1e-12

NATURALISTIC_200 = "naturalistic_200"
MAIN_STUDY = "main_study_codebook"

SOURCES = {
    "runs_semantic_fair": NATURALISTIC_200,
    "runs_natural": NATURALISTIC_200,
    "runs_recal": MAIN_STUDY,
}

# Task 20's three deterministic comparators. Their parquets carry no `cell` or
# `model` column -- the filename is the arm -- and they were run as FIXED
# POINTS at threshold=-inf, which is how the manuscript already reports them.
# Sweeping them here would both move published numbers and duplicate the two
# new `fair:*_resolver_gate` arms, which are precisely their swept versions.
FROZEN_POINT_FILES = {
    "natural_confidence_gate_canonical",
    "natural_confidence_gate_lexical",
    "natural_random_gate_canonical",
}

# Whether the arm's prompt disclosed the nine canonical commands. This is the
# asymmetry the fair-comparison experiment closes, so it belongs in the table
# rather than in a reader's memory of which arm was which.
VOCAB_YES = "yes"
VOCAB_NO = "no"
VOCAB_NOT_LLM = "n/a: not an LLM arm"
VOCAB_NO_VOCAB_EXISTS = "n/a: hashed codebook has no command vocabulary"

FAIR_VOCAB_ARMS = {"fair:llm_vocab:advisory", "fair:llm_vocab:enforced", "fair:hybrid_semantic_gate"}


def risk_upper95_one_sided(n_unfaithful: int, n_covered: int) -> float:
    """One-sided 95% upper confidence bound on this arm's risk (Clopper-Pearson).

    Exists because 37 of this table's 55 rows report risk 0.0 on at most 200
    episodes. That is an OBSERVED zero, not a demonstrated one, and a CSV that
    carries only the point estimate lets a reader state it as "never
    unfaithful". At `n_unfaithful == 0` this reduces exactly to the rule-of-
    three form `1 - 0.05 ** (1 / n_covered)` -- about 0.015 at n=200 -- and
    generalizes it to the rows that did observe failures rather than leaving
    those blank.

    NaN at zero coverage: an arm that acted on nothing has no error rate to
    bound.

    Applied to a SUBSET of an arm's episodes it bounds that subset's risk and
    nothing wider, which is what `risk_upper95_beyond_comparator` uses it for.
    """
    if n_covered <= 0:
        return np.nan
    return float(beta.ppf(0.95, n_unfaithful + 1, n_covered - n_unfaithful))


def admitted_episode_ids(rows: pd.DataFrame, threshold: float) -> set[str]:
    """WHICH episodes this arm acted on at `threshold`, not how many.

    The same admission rule every curve here is built from -- a proposal
    exists AND its confidence clears the threshold -- resolved back to the
    episode identities that the coverage RATE discards. Two arms at coverage
    0.94 can act on two different sets of 188 episodes, and no rate can tell
    them apart; `n_beyond_comparator` and
    `is_strict_superset_of_comparator_coverage` are exactly the questions that
    need the identities.

    A fixed-point row has no threshold (`NaN`) because its recorded `covered`
    already IS its decision.
    """
    admitted = rows["covered"].to_numpy(bool)
    if not np.isnan(threshold):
        admitted = admitted & (rows["confidence"].to_numpy(float) >= threshold)
    return set(rows.loc[admitted, "episode_id"])


def _parquets(dirname: str) -> list[Path]:
    """Every checkpoint in one run directory, excluding the `._` AppleDouble
    sidecars exFAT leaves beside each file -- pyarrow reads one as a corrupt
    parquet, not as a file to skip."""
    d = INTERMEDIATE / dirname
    return sorted(p for p in d.glob("*.parquet") if not p.name.startswith("._"))


def load_arms() -> list[dict]:
    """Every (arm, model) group across the three run directories, as dicts of
    the raw per-episode arrays the summary is computed from.

    A frozen-comparator file (no `cell`/`model` column) is keyed by its
    filename stem; everything else is grouped by its own `cell` and `model`
    columns, so an arm that gained or lost a model between runs is visible as
    a row count rather than silently averaged away.
    """
    groups = []
    for dirname, episode_set in SOURCES.items():
        for path in _parquets(dirname):
            rows = pd.read_parquet(path)
            if "cell" not in rows.columns:
                if path.stem not in FROZEN_POINT_FILES:
                    raise ValueError(
                        f"{path} has no `cell` column and is not one of Task 20's three known "
                        f"frozen comparators {sorted(FROZEN_POINT_FILES)} -- refusing to guess "
                        "what arm it is"
                    )
                groups.append(dict(arm_name=path.stem, model=None, source_run=dirname,
                                   episode_set=episode_set, cell=None, rows=rows))
                continue
            for (arm, model), sub in rows.groupby(["cell", "model"], dropna=False):
                cell = Cell(arm, sub["uncertainty_source"].iloc[0], sub["control_mechanism"].iloc[0],
                            scaffold=int(sub["scaffold"].iloc[0]),
                            uses_llm=bool(sub["uses_llm"].iloc[0]))
                groups.append(dict(arm_name=arm, model=(None if not bool(sub["uses_llm"].iloc[0]) else model),
                                   source_run=dirname, episode_set=episode_set, cell=cell, rows=sub))
    return groups


def vocabulary_disclosure(arm_name: str, model: str | None, episode_set: str) -> str:
    if model is None:
        return VOCAB_NOT_LLM
    if episode_set == MAIN_STUDY:
        return VOCAB_NO_VOCAB_EXISTS
    return VOCAB_YES if arm_name in FAIR_VOCAB_ARMS else VOCAB_NO


def is_swept(group: dict) -> bool:
    """Does this arm's row record a PROPOSAL that must be threshold-swept, or
    its own final outcome?

    `curve_knob` (Ruling 27) answers it from the arm's mechanism rather than
    its name: an enforced arm with a real uncertainty signal has a genuine
    confidence threshold and recorded a proposal; an advisory arm decided for
    itself and is a fixed point; a `"coverage"`-knob arm (the random gate) has
    no confidence to threshold on and is likewise reported as the point it was
    drawn at.
    """
    if group["cell"] is None:
        return False  # Task 20's published fixed points -- see FROZEN_POINT_FILES
    return curve_knob(group["cell"]) == "confidence"


def published_resolver_curve(arm_name: str) -> pd.DataFrame:
    curves = pd.read_csv(RESOLVER_CURVES)
    curve = curves[curves["arm_name"] == arm_name]
    if curve.empty:
        raise ValueError(f"{RESOLVER_CURVES} carries no curve for {arm_name}")
    return curve.reset_index(drop=True)


def build_rows() -> pd.DataFrame:
    groups = load_arms()

    # Curves first: every row is scored against the comparator's frontier, so
    # no row can be finalized until that arm's curve exists.
    for g in groups:
        rows = g["rows"]
        if not is_swept(g):
            g["curve"] = None
            continue
        if g["arm_name"] in ("fair:exact_resolver_gate", "fair:lexical_resolver_gate"):
            # Task 6's own output, read rather than recomputed so this table and
            # the run it summarizes cannot drift apart.
            g["curve"] = published_resolver_curve(g["arm_name"])
        else:
            g["curve"] = recorded_proposal_gate_curve(rows["covered"], rows["faithful"],
                                                      rows["confidence"], threshold_grid=THRESHOLD_GRID)

    comparator_group = next(g for g in groups if g["arm_name"] == COMPARATOR_ARM)
    comparator = comparator_group["curve"]
    comparator_max = float(comparator["coverage"].max())
    comparator_risk_at_max = risk_at_coverage(comparator, comparator_max)

    # The comparator's own maximum-coverage operating point, as the set of
    # episodes it acts on there. Threshold 0 admits every proposal, so this is
    # the widest the deterministic resolver ever reaches: 188 of the 200.
    comparator_at_max = comparator.loc[comparator["threshold"].astype(float).idxmin()]
    if abs(float(comparator_at_max["coverage"]) - comparator_max) > TOL:
        raise ValueError(
            f"{COMPARATOR_ARM}: coverage at its lowest threshold is "
            f"{float(comparator_at_max['coverage'])} but its frontier peaks at {comparator_max} -- "
            "every set difference below is taken against the episodes it admits at its MAXIMUM "
            "coverage, and this row is not that point"
        )
    comparator_covered_ids = admitted_episode_ids(comparator_group["rows"],
                                                  float(comparator_at_max["threshold"]))
    if len(comparator_covered_ids) != int(comparator_at_max["n_covered"]):
        raise ValueError(
            f"{COMPARATOR_ARM}: {len(comparator_covered_ids)} distinct episodes are admitted at "
            f"threshold {float(comparator_at_max['threshold'])} but its published curve records "
            f"{int(comparator_at_max['n_covered'])} there -- the episode-level reconstruction and "
            "the curve disagree, so no set difference computed against it can be trusted"
        )

    # How many (model) rows each arm contributes. Carried into the table because
    # the arms do not share a panel -- the three fair:* LLM arms ran 10 models,
    # the Task 20 LLM controls 5, and the two fair:*_resolver_gate arms are
    # deterministic and model-independent, so they contribute one row each and
    # ran no model at all -- so a bare `groupby("arm_name").coverage.mean()`
    # compares a 10-model average against a 5-model one. Restrict to the shared
    # models before contrasting two arms, or read this column and know not to.
    panel_size = {}
    for g in groups:
        panel_size[(g["arm_name"], g["source_run"])] = panel_size.get((g["arm_name"], g["source_run"]), 0) + 1

    out = []
    for g in groups:
        rows, curve = g["rows"], g["curve"]
        covered = rows["covered"].to_numpy(bool)
        faithful = rows["faithful"].to_numpy(bool)
        n_episodes = int(len(rows))
        proposal_rate = float(covered.mean())
        # Faithful is a strict subset of covered by construction in every
        # harness here, so this counts unfaithful EXECUTIONS, never abstentions.
        n_covered = int(covered.sum())
        n_unfaithful = n_covered - int((covered & faithful).sum())

        if curve is None:
            # A fixed point: the arm's own recorded decision, exactly as the
            # existing Results section defines it -- coverage is the fraction of
            # episodes acted on, risk the unfaithful fraction AMONG those.
            coverage = proposal_rate
            risk = (n_unfaithful / n_covered) if n_covered else np.nan
            reference_threshold = np.nan
            arm_aurc = arm_aurc_common = comparator_area = np.nan
            lo = hi = np.nan
        else:
            # Threshold 0.0 admits every proposal, so this row is the
            # maximum-coverage end of the arm's own frontier -- and is why
            # `coverage` equals `proposal_rate` for a swept arm.
            at_zero = curve.loc[curve["threshold"].astype(float).idxmin()]
            coverage = float(at_zero["coverage"])
            risk = float(at_zero["risk"])
            reference_threshold = float(at_zero["threshold"])
            if int(at_zero["n_covered"]) != n_covered:
                raise ValueError(
                    f"{g['arm_name']}/{g['model']}: the swept curve admits {int(at_zero['n_covered'])} "
                    f"episodes at threshold 0 but {n_covered} rows record a proposal -- the curve "
                    "and the rows it was built from describe different episode sets"
                )
            arm_aurc = aurc(curve)
            lo, hi = common_support({"arm": curve, "comparator": comparator})
            arm_aurc_common = aurc_common(curve, lo, hi)
            comparator_area = aurc_common(comparator, lo, hi)

        # Risk at the comparator's own ceiling, for every swept arm that
        # actually reaches it. NaN rather than an extrapolation for an arm
        # whose frontier stops short -- the exact resolver tops out at 0.66 and
        # has no operating point at 0.94 to report.
        matched_risk = np.nan
        if curve is not None and float(curve["coverage"].max()) >= comparator_max - 1e-12:
            matched_risk = risk_at_coverage(curve, comparator_max)

        # The comparator's risk at THIS arm's coverage -- `nag.riskcoverage.
        # dominates`'s test, applied uniformly to swept and fixed-point arms
        # alike. NaN where the arm operates above the comparator's ceiling:
        # the deterministic resolver has no operating point there at all, which
        # is itself a result and not a missing value to be filled by clamping.
        comparator_risk = beats = np.nan
        if g["episode_set"] != NATURALISTIC_200:
            verdict = "n/a: different episode set from the comparator"
        elif g["arm_name"] == COMPARATOR_ARM:
            verdict = "n/a: this row is the comparator"
        elif coverage > comparator_max + TOL:
            # The comparator has no operating point here at all. Whether that
            # favours the arm depends entirely on what the extra coverage cost
            # in risk, so these two cases must not share a label.
            verdict = ("exceeds the comparator's frontier (higher coverage at no more risk)"
                       if risk <= comparator_risk_at_max + TOL
                       else "higher coverage than the comparator reaches, but at higher risk")
        else:
            comparator_risk = risk_at_coverage(comparator, coverage)
            beats = bool(risk < comparator_risk - TOL)
            # The two branches below read `comparator_risk_at_max <= risk` as
            # "the comparator gets no worse by going further", which is only
            # the right reading while that risk is zero. A comparator with
            # nonzero risk at its ceiling would make a row with slightly worse
            # risk look like a tie, so fail loud rather than mislabel it.
            assert comparator_risk_at_max <= TOL, (
                f"{COMPARATOR_ARM} carries risk {comparator_risk_at_max} at its maximum coverage "
                f"{comparator_max}; the Pareto verdicts below assume it is 0.0 and would silently "
                "mislabel dominated rows as matches"
            )
            if beats:
                verdict = "beats the comparator at its own coverage"
            elif risk > comparator_risk + TOL:
                verdict = "dominated by the comparator (worse risk at the same coverage)"
            elif coverage < comparator_max - TOL and comparator_risk_at_max <= risk + TOL:
                # Equal risk AT THIS COVERAGE, but the comparator keeps going and
                # gets no worse. Calling that a tie is the degenerate reading this
                # branch exists to prevent: with a comparator whose risk is 0.0 at
                # every threshold, every zero-risk arm would "tie" it at any
                # coverage at all, including coverages it strictly dominates.
                verdict = "dominated by the comparator (it reaches higher coverage at no more risk)"
            else:
                verdict = "matches the comparator exactly (same coverage, same risk)"

        # WHICH episodes, not how many. The positive claim this table supports
        # is about the episodes an arm reaches that the comparator does not, so
        # it needs the set difference and the bound computed ON that difference
        # -- 8-12 episodes, where a zero-failure bound is 0.22-0.31, and not the
        # ~200 covered episodes `risk_upper95_one_sided` is computed over. It
        # also needs the difference in the OTHER direction: an arm can cover
        # more episodes than the comparator while missing some the comparator
        # resolves, which "higher coverage" hides entirely.
        n_beyond = n_missed = n_unfaithful_beyond = beyond_bound = np.nan
        strict_superset = False
        if g["episode_set"] == NATURALISTIC_200:
            admitted = admitted_episode_ids(rows, reference_threshold)
            if len(admitted) != n_covered:
                raise ValueError(
                    f"{g['arm_name']}/{g['model']}: {len(admitted)} distinct episodes are admitted "
                    f"at threshold {reference_threshold} but the row reports {n_covered} covered -- "
                    "the episode-level reconstruction disagrees with the rate it must be consistent with"
                )
            beyond = admitted - comparator_covered_ids
            missed = comparator_covered_ids - admitted
            in_beyond = rows["episode_id"].isin(beyond).to_numpy(bool)
            n_beyond = len(beyond)
            n_missed = len(missed)
            n_unfaithful_beyond = int((in_beyond & ~faithful).sum())
            beyond_bound = risk_upper95_one_sided(n_unfaithful_beyond, n_beyond)
            # PROPER superset: covers every episode the comparator covers AND at
            # least one it does not. An arm that reproduces the comparator's set
            # exactly (`natural_confidence_gate_lexical` is the same algorithm)
            # adds nothing and is False here; read `n_comparator_episodes_missed`
            # == 0 for the weaker "does everything the comparator does".
            strict_superset = bool(n_missed == 0 and n_beyond > 0)

        out.append(dict(
            arm_name=g["arm_name"],
            model=g["model"],
            source_run=g["source_run"],
            episode_set=g["episode_set"],
            model_saw_vocabulary=vocabulary_disclosure(g["arm_name"], g["model"], g["episode_set"]),
            operating_point_kind="swept_curve" if curve is not None else "fixed_point",
            sweep_source=(
                "none: the arm's own recorded decision" if curve is None
                else "semantic_fair_resolver_curves.csv (resolver_gate_curve, Task 2)"
                if g["arm_name"] in ("fair:exact_resolver_gate", "fair:lexical_resolver_gate")
                else "recorded_proposal_gate_curve over the frozen 101-point threshold_grid"
            ),
            reference_threshold=reference_threshold,
            n_episodes=n_episodes,
            proposal_rate=proposal_rate,
            n_covered=n_covered,
            n_unfaithful=n_unfaithful,
            model_panel_size=panel_size[(g["arm_name"], g["source_run"])],
            coverage=coverage,
            risk=risk,
            risk_upper95_one_sided=risk_upper95_one_sided(n_unfaithful, n_covered),
            # The incremental claim's own numbers. `risk_upper95_one_sided`
            # bounds the whole arm and is 15-20x tighter than the increment
            # deserves, so quote the first three of these whenever the sentence
            # is about episodes the comparator cannot reach -- and the last two
            # whenever it is about doing everything the comparator does.
            n_beyond_comparator=n_beyond,
            n_unfaithful_beyond_comparator=n_unfaithful_beyond,
            risk_upper95_beyond_comparator=beyond_bound,
            n_comparator_episodes_missed=n_missed,
            is_strict_superset_of_comparator_coverage=strict_superset,
            aurc=arm_aurc,
            aurc_common=arm_aurc_common,
            comparator_aurc_common=comparator_area,
            aurc_common_minus_comparator=arm_aurc_common - comparator_area,
            # False whenever the comparator's own area over the shared window is
            # exactly zero: the difference column is then bounded below by zero
            # and equals `aurc_common` identically, so it can order arms against
            # each other but never against the comparator. It looks like a
            # ranking statistic and is not one.
            aurc_ranking_valid=(np.nan if curve is None else bool(comparator_area > 0.0)),
            support_lo=lo,
            support_hi=hi,
            risk_at_comparator_max_coverage=matched_risk,
            comparator_arm=COMPARATOR_ARM if g["episode_set"] == NATURALISTIC_200 else None,
            comparator_max_coverage=comparator_max if g["episode_set"] == NATURALISTIC_200 else np.nan,
            comparator_risk_at_own_coverage=comparator_risk,
            beats_comparator=beats,
            frontier_verdict=verdict,
        ))

    table = pd.DataFrame(out).sort_values(["episode_set", "arm_name", "model"],
                                          na_position="first").reset_index(drop=True)
    if table.duplicated(["arm_name", "model", "source_run"]).any():
        raise ValueError("two groups collapsed onto the same (arm, model, source_run) key")
    return table


def main() -> int:
    table = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(table)} arm rows "
          f"({int((table['operating_point_kind'] == 'swept_curve').sum())} swept, "
          f"{int((table['operating_point_kind'] == 'fixed_point').sum())} fixed points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
