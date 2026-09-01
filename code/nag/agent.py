"""Multi-turn tool-calling agent loop with an explicit stopping condition.

Scores transmission fidelity only: an episode is faithful iff the executed
action equals the action entailed (see `nag.taxonomy.entail`) by the TRUE
string, never a claim about what the participant meant to communicate. No
cross-session memory is carried between episodes -- each `run_episode` call
builds a fresh message history and a fresh `Environment`.

`run_episode_for_cell` (Ruling 28) is the study's actual entry point: it
dispatches a `nag.design.Cell` to the right execution path (non-LLM
reference gate, LLM-mediated advisory, or LLM-mediated enforced) and is
what makes `control_mechanism` do anything at all. `run_episode` alone,
called directly, never applies enforcement -- see its own docstring.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import numpy as np

from nag.controllers import nonllm_gate, random_gate
from nag.openrouter import ParseFailure, extract_tool_calls
from nag.prompts import build_system
from nag.riskcoverage import curve_knob
from nag.taxonomy import entail
from nag.tools import TOOL_SCHEMAS, Environment

MAX_TURNS = 6

SINGLESHOT_CELL_NAME = "singleshot"

# Per-cell turn limits. Empty today and kept because the mechanism is sound;
# the single-shot arm is NOT built with it. Capping that arm's turns at 1 was
# tried and is wrong: the spec defines it as "Single-shot LLM (no tools), the
# intent-drift regime", and given tools plus one turn every model spends the
# turn on `read_buffer` and never reaches `execute`. That attempt scored zero
# coverage on all 200 episodes it ran, across two models. A no-tools arm is a
# different object, not a throttled agent, so it gets its own path below.
CELL_MAX_TURNS: dict[str, int] = {}


def max_turns_for(cell) -> int:
    """Turn limit for `cell`. `MAX_TURNS` unless the cell overrides it."""
    name = getattr(cell, "name", None)
    return CELL_MAX_TURNS.get(name, MAX_TURNS)


_TERMINAL_TOOLS = ("execute", "abstain")


@dataclass
class EpisodeRecord:
    episode_id: str
    cell: object
    executed: dict | None
    faithful: bool
    covered: bool
    n_turns: int
    parse_failed: bool
    served_provider: str | None
    participant_id: object = None
    study: object = None
    n_terminal_calls: int = 0
    confidence: float | None = None


def _as_action(executed: dict | None) -> tuple | None:
    """Reduce an executed tool call to a plain ``(name, args)`` key.

    Deliberately returns a plain tuple, not a `nag.taxonomy.Action`. Building
    a fake `Action(name, args, tier=0)` here would make faithfulness scoring
    depend on `Action.tier` being `compare=False` -- an implementation
    detail of `nag.taxonomy` this module has no business relying on. If that
    ever changed, every correctly-executed action would silently score
    unfaithful and the whole study would go quietly null. Comparing plain
    `(name, args)` tuples makes that failure mode structurally impossible.
    """
    if executed is None:
        return None
    return (executed["name"], tuple(sorted((executed.get("args") or {}).items())))


def run_episode(cell, episode, confidence, client, system: str = "",
                max_turns: int | None = None) -> EpisodeRecord:
    """Run one episode through the agent loop and score it.

    `confidence`: CALIBRATED confidence for this episode (the output of
    `nag.confidence.Calibrator.transform` / `episode_confidence`), never a
    raw decoder score -- raw `score_top` spans only ~0.037-0.317 against a
    1/36 chance floor and is not on a probability scale comparable across
    gates or cells. This loop itself never branches on it -- whether it
    reaches the model at all is entirely `system`'s call (see
    `nag.prompts.build_system`, which renders it for an advisory cell and
    withholds it for an enforced one). It is recorded on the returned
    `EpisodeRecord` regardless, because an ENFORCED cell's post-hoc gate
    (`apply_enforced_gate`, `nag.riskcoverage.rc_curve`) needs it after the
    fact. Callers driving a real `nag.design.Cell` through the study should
    use `run_episode_for_cell` below, which builds `system` correctly and
    routes non-LLM reference cells away from this function entirely, rather
    than calling `run_episode` directly.

    Terminates when the environment reaches a terminal tool call
    (`execute` or `abstain`), when the model emits no parseable tool call
    (`parse_failed`), or after `MAX_TURNS` turns -- whichever comes first.
    An episode that hits `MAX_TURNS` without a terminal call is scored as
    uncovered, not silently dropped. `EpisodeRecord.n_terminal_calls` counts
    every `execute`/`abstain` call that arrived across the episode -- almost
    always 0 or 1, but 2 when a model emits both in one parallel batch (see
    `Environment.call`'s first-terminal-call-wins precedence).
    """
    env = Environment(decoded_string=episode["decoded_string"])
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Act on the user's input channel."}]
    served, turns, parse_failed, n_terminal_calls = None, 0, False, 0
    limit = max_turns_for(cell) if max_turns is None else max_turns

    while not env.terminated and turns < limit:
        turns += 1
        resp, served = client.chat(messages=messages, tools=TOOL_SCHEMAS)
        try:
            calls = extract_tool_calls(resp)
        except ParseFailure:
            parse_failed = True
            break
        messages.append(resp["choices"][0]["message"])
        for c in calls:
            # Every call in this turn is executed, in order -- a response
            # carrying `read_buffer` *and* `execute` must not lose the
            # `execute` just because it wasn't first (see nag.openrouter's
            # extract_tool_calls docstring). Environment.call already
            # refuses further calls once terminated, so processing calls
            # after termination is a harmless no-op, not a bug. If a batch
            # carries more than one terminal call (execute+abstain, or
            # execute+execute), Environment's first-terminal-call-wins
            # guard (see its docstring) means the FIRST one recorded is
            # what determines the episode's outcome -- later terminal
            # calls in the same batch are refused, not honored. That
            # precedence is deliberate and safety-conservative (a model
            # that emitted `execute` at all would have fired it in a real
            # deployment), but the rate at which models emit more than one
            # terminal call is a quantity worth keeping, not discarding --
            # it may vary by experimental arm -- so it is counted here.
            if c["name"] in _TERMINAL_TOOLS:
                n_terminal_calls += 1
            result = env.call(c["name"], c["arguments"])
            messages.append({"role": "tool", "tool_call_id": c.get("id"),
                              "name": c["name"], "content": json.dumps(result)})

    want = entail(episode["true_string"])
    got = _as_action(env.executed)
    covered = env.executed is not None
    faithful = bool(covered and want is not None and got == (want.name, want.args))
    return EpisodeRecord(
        episode_id=episode["episode_id"], cell=cell, executed=env.executed,
        faithful=faithful, covered=covered, n_turns=turns, parse_failed=parse_failed,
        served_provider=served,
        participant_id=episode.get("participant_id"), study=episode.get("study"),
        n_terminal_calls=n_terminal_calls, confidence=confidence,
    )


# --- Ruling 28: the actual dispatcher, and the enforced cell's post-hoc gate

def run_episode_for_cell(cell, episode, confidence: float, client, *,
                         rng: np.random.Generator | None = None,
                         threshold: float | None = None,
                         coverage: float | None = None) -> EpisodeRecord:
    """Dispatch one episode to the right execution path for `cell` and
    return a raw, pre-harness-decision `EpisodeRecord`. This is the
    factorial's control_mechanism arm actually happening (Ruling 28): a
    live probe found that calling `run_episode` directly, for every cell,
    made all 34 arms behave identically -- 12 "enforced" cells ran exactly
    like their advisory twins, and `nonllm_gate`/`random_gate` were never
    reached at all. This function is what fixes that.

    Three paths, chosen on `cell.uses_llm` first and `cell.control_mechanism`
    second:

    - NON-LLM (`cell.uses_llm is False` -- today only `nonllm_gate` and
      `random_gate`, `nag.design.enumerate_cells`): routed to
      `nag.controllers.nonllm_gate` / `random_gate`, fed the action entailed
      by the DECODED string (never the true string -- see
      `nag.controllers`'s module docstring for why that is what lets these
      baselines lose). `client` is never referenced on this path: it makes
      ZERO API calls by construction, not by discipline.

    - ADVISORY (`cell.control_mechanism == "advisory"`): unchanged
      `run_episode`, with `system` built by
      `nag.prompts.build_system(cell, confidence)`, which renders the
      confidence value into the prompt for this mechanism. The model's own
      decision IS the final, reported outcome for this arm -- nothing here
      or downstream overrides it (advisory arms are scored as fixed points,
      `nag.riskcoverage.dominates`, never swept).

    - ENFORCED (`cell.control_mechanism == "enforced"`): ALSO an unchanged
      `run_episode` call, but `build_system` withholds the confidence value
      for this mechanism (enforcement lives in the harness, never the
      prompt). The returned record's `covered`/`faithful`/`executed`
      describe what the model PROPOSED, not what the harness admits.
      Deliberately: the model cannot condition its behaviour on a threshold
      it never sees, so ONE run is sufficient for every possible threshold.
      `apply_enforced_gate` turns one proposal into a final decision at a
      given threshold; `nag.riskcoverage.rc_curve` / `coverage_sweep_curve`
      sweep a whole frontier from a single batch of these records. Never
      call this function once per threshold for an enforced cell -- that
      would multiply LLM spend for zero additional information, since the
      model's behaviour does not depend on the threshold at all.

    `threshold`/`coverage` are consulted ONLY on the non-LLM path (an
    LLM-mediated cell has no live gate to parametrize -- it always runs at
    the "record everything, decide later" point). Left at their default
    `None`, the non-LLM path ALSO runs at that same point (`threshold=-inf`
    for a confidence-knob gate so it always proposes when the decode
    resolves; `coverage=1.0` for a coverage-knob gate so it always
    proposes), so its curve comes from one run per episode too, swept
    exactly like the enforced LLM arms via `rc_curve`/`coverage_sweep_curve`.
    Pass an explicit value only to realize one concrete reference operating
    point directly (e.g. to check `random_gate` actually hits a target
    coverage, or to report a single number alongside the swept curve).
    """
    true_entailed = entail(episode["true_string"])

    if not getattr(cell, "uses_llm", True):
        decoded_entailed = entail(episode["decoded_string"])
        knob = curve_knob(cell)
        if knob == "confidence":
            out = nonllm_gate(confidence, threshold if threshold is not None else float("-inf"),
                              decoded_entailed)
        elif knob == "coverage":
            gen = rng if rng is not None else np.random.default_rng()
            out = random_gate(gen, coverage if coverage is not None else 1.0, decoded_entailed)
        else:
            raise ValueError(
                f"non-LLM cell {getattr(cell, 'name', cell)!r} has no recognized gate knob "
                f"(curve_knob={knob!r}) -- nonllm_gate and random_gate are the only non-LLM "
                "cells this dispatcher knows how to route"
            )
        covered = bool(out["covered"])
        got = _as_action(out["executed"])
        want = (true_entailed.name, true_entailed.args) if true_entailed is not None else None
        faithful = bool(covered and want is not None and got == want)
        return EpisodeRecord(
            episode_id=episode["episode_id"], cell=cell, executed=out["executed"],
            faithful=faithful, covered=covered, n_turns=0, parse_failed=False,
            served_provider=None, participant_id=episode.get("participant_id"),
            study=episode.get("study"), n_terminal_calls=0, confidence=confidence,
        )

    if getattr(cell, "name", None) == SINGLESHOT_CELL_NAME:
        return _run_singleshot(cell, episode, confidence, client, true_entailed)

    if cell.control_mechanism not in ("advisory", "enforced"):
        raise ValueError(f"unknown control_mechanism {cell.control_mechanism!r} on LLM cell "
                         f"{getattr(cell, 'name', cell)!r}")

    # advisory and enforced both run the SAME loop -- the only difference is
    # what build_system puts in the prompt. Enforcement is applied later, by
    # apply_enforced_gate, never by branching this call.
    system = build_system(cell, confidence=confidence)
    return run_episode(cell=cell, episode=episode, confidence=confidence, client=client, system=system)


def _run_singleshot(cell, episode, confidence, client, true_entailed) -> EpisodeRecord:
    """The single-shot arm: ONE model call, NO tools, no loop.

    The spec's third controller class, "Single-shot LLM (no tools), the
    intent-drift regime". It is not the tool-using agent with its turns capped:
    that variant was built first and scored zero coverage on every one of 200
    episodes, because a model given tools and one turn spends it reading the
    buffer and never reaches `execute`.

    The model is handed the decoded code AND the action it resolves to, because
    the codebook is a salted hash and no prompt could let a model derive it. The
    tool arm gets that same resolution free from `lookup_action`, which is
    non-terminal and never refuses, so the two arms carry identical information
    and the contrast is the loop alone.

    Scoring mirrors the tool arm exactly. Naming an action is `execute`; the
    word "abstain" is `abstain`; anything else is a parse failure, counted and
    reported rather than repaired, and it is a DIFFERENT parse-failure mode from
    the tool arm's (free text, not a malformed tool call), so the two must not
    be pooled without saying so.
    """
    from nag.prompts import build_singleshot_user
    from nag.taxonomy import ACTIONS

    # The resolution shown is the DECODED string's action, never the true
    # string's. Showing the true action would hand this arm the answer key and
    # make it unable to lose, which is the same mistake `nag.controllers`
    # documents for the non-LLM gates.
    decoded_entailed = entail(episode["decoded_string"])
    prompt = build_singleshot_user(
        cell, episode["decoded_string"],
        decoded_entailed.name if decoded_entailed is not None else "",
    )

    resp, served = client.chat(
        messages=[{"role": "user", "content": prompt}], tools=None)
    text = (resp["choices"][0]["message"].get("content") or "").strip().lower()

    executed, covered, parse_failed = None, False, False
    first = text.splitlines()[0].strip().strip(".'\"` ") if text else ""
    if first == "abstain":
        pass
    elif first in ACTIONS:
        executed = {"name": first, "args": {}}
        covered = True
    else:
        parse_failed = True

    got = _as_action(executed)
    want = (true_entailed.name, true_entailed.args) if true_entailed is not None else None
    return EpisodeRecord(
        episode_id=episode["episode_id"], cell=cell, executed=executed,
        faithful=bool(covered and want is not None and got == want),
        covered=covered, n_turns=1, parse_failed=parse_failed,
        served_provider=served, participant_id=episode.get("participant_id"),
        study=episode.get("study"), n_terminal_calls=0 if parse_failed else 1,
        confidence=confidence,
    )


def apply_enforced_gate(record: EpisodeRecord, threshold: float) -> EpisodeRecord:
    """The harness's post-hoc admission decision for an ENFORCED cell's
    recorded proposal (Ruling 28).

    `run_episode_for_cell`'s enforced-cell records describe what the model
    PROPOSED -- the model never saw the threshold, or even that its cell is
    enforced (`build_system` withholds confidence identically for every
    enforced cell), so its behaviour cannot depend on it. Enforcement
    happens entirely here, offline, against the real recorded
    `record.confidence`: a proposal below `threshold` is suppressed into an
    abstention (`executed=None`, `covered=False`, `faithful=False`) rather
    than scored as an unfaithful execution -- an action the harness refused
    to admit never happened.

    Refuses to gate a record whose `cell` is not an enforced cell: advisory
    arms are fixed points (`nag.riskcoverage.dominates`), never swept post
    hoc, and silently gating one here would misreport the model's own
    decision as a harness decision.

    Sweeping this function (or the array-level equivalent inside
    `nag.riskcoverage.rc_curve`) over a grid of thresholds against ONE
    batch of `run_episode_for_cell` records is how the full risk-coverage
    frontier for an enforced cell is built -- never by calling
    `run_episode_for_cell` again per threshold.
    """
    if record.cell is None or getattr(record.cell, "control_mechanism", None) != "enforced":
        raise ValueError(
            "apply_enforced_gate refuses to gate a record whose cell is not an enforced cell "
            f"({getattr(record.cell, 'name', record.cell)!r}): advisory arms are fixed points, "
            "never swept post hoc -- see nag.riskcoverage.dominates"
        )
    if record.confidence is None:
        raise ValueError(
            "apply_enforced_gate needs record.confidence -- run this episode through "
            "run_episode_for_cell (or pass confidence into run_episode directly) first"
        )
    if record.confidence >= threshold:
        return record
    return dataclasses.replace(record, executed=None, covered=False, faithful=False)
