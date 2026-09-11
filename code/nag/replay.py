"""Repeated-attempt empirical replay: exact outcome distributions over three attempts.

Every policy's response to every donor episode is ALREADY RECORDED, so a
trajectory is fully determined by the sequence of donor episodes drawn. The
outcome distribution is therefore computed exactly, by enumerating the draw
tree to depth three, rather than by Monte Carlo.

That is not a performance choice. Simulating would introduce noise into a
quantity that has a closed form, and it would invite the error of reporting the
number of simulated trajectories as though it were a sample size. There is no
simulation n here. The sample size is 200 episodes from 46 participants, and it
does not change no matter how the trajectories are counted.

`nag.sandbox` provides an independent Monte Carlo path that executes real state
changes; `tests/test_replay.py` requires the two to agree. Two implementations
of one quantity is the only test class that has ever caught a defect in this
project's analysis code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nag.sandbox import TIER3_ACTIONS

MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class OutcomeCounts:
    """How a policy's donor pool for one command partitions.

    Wrong executions are split by tier because a wrong tier-3 action is the
    endpoint that matters; declines are split because a malformed tool call and
    a deliberate refusal are not the same behaviour, and only one of them is an
    oversight mechanism a deployment could rely on.
    """

    success: int
    wrong_tier3: int
    wrong_other: int
    decline_abstain: int
    decline_parse: int

    @property
    def total(self) -> int:
        return (self.success + self.wrong_tier3 + self.wrong_other
                + self.decline_abstain + self.decline_parse)


def classify(df: pd.DataFrame) -> OutcomeCounts:
    """Partition one policy's episodes for one command."""
    covered = df["covered"].fillna(False).astype(bool)
    faithful = df["faithful"].fillna(False).astype(bool)
    # The three deterministic comparators ship without this column; they resolve
    # through an exact lookup and cannot emit a malformed tool call.
    parse = (df["parse_failed"].fillna(False).astype(bool) if "parse_failed" in df
             else pd.Series(False, index=df.index))
    executed = df["executed_name"]

    wrong = covered & ~faithful
    is_t3 = executed.isin(TIER3_ACTIONS)
    return OutcomeCounts(
        success=int((covered & faithful).sum()),
        wrong_tier3=int((wrong & is_t3).sum()),
        wrong_other=int((wrong & ~is_t3).sum()),
        decline_abstain=int((~covered & ~parse).sum()),
        decline_parse=int((~covered & parse).sum()),
    )


def outcome_distribution(counts: OutcomeCounts, max_attempts: int = MAX_ATTEMPTS,
                         with_replacement: bool = False) -> dict:
    """Exact distribution over outcomes of a task allowed `max_attempts` tries.

    Success and wrong execution are terminal. A decline consumes the attempt and
    the next attempt draws a DIFFERENT donor for the same command (or the same
    pool, under `with_replacement`, reported only as a sensitivity analysis:
    a real retry never reproduces a byte-identical corrupted string).
    """
    p_success = [0.0] * max_attempts
    p_wrong = [0.0] * max_attempts
    p_wrong_t3 = [0.0] * max_attempts
    e_retry = {"abstain": 0.0, "parse": 0.0}
    e_attempts = 0.0
    unresolved = 0.0

    def walk(s: int, w3: int, wo: int, da: int, dp: int, attempt: int, weight: float) -> None:
        nonlocal e_attempts, unresolved
        k = s + w3 + wo + da + dp
        # An exhausted pool ends the task where it stands. Without replacement a
        # command with fewer donors than `max_attempts` simply cannot be retried
        # further, which is a property of the data, not an error. `attempt` is
        # the number of attempts already consumed, so it is the right cost here.
        if attempt >= max_attempts or k == 0:
            unresolved += weight
            e_attempts += weight * attempt
            return

        i = attempt  # 0-based index into the per-attempt arrays
        p_s, p_w3, p_wo = s / k, w3 / k, wo / k
        p_da, p_dp = da / k, dp / k

        p_success[i] += weight * p_s
        p_wrong[i] += weight * (p_w3 + p_wo)
        p_wrong_t3[i] += weight * p_w3
        e_attempts += weight * (p_s + p_w3 + p_wo) * (i + 1)
        e_retry["abstain"] += weight * p_da
        e_retry["parse"] += weight * p_dp

        if attempt + 1 >= max_attempts:
            unresolved += weight * (p_da + p_dp)
            e_attempts += weight * (p_da + p_dp) * max_attempts
            return
        if with_replacement:
            if p_da:
                walk(s, w3, wo, da, dp, attempt + 1, weight * p_da)
            if p_dp:
                walk(s, w3, wo, da, dp, attempt + 1, weight * p_dp)
        else:
            if p_da:
                walk(s, w3, wo, da - 1, dp, attempt + 1, weight * p_da)
            if p_dp:
                walk(s, w3, wo, da, dp - 1, attempt + 1, weight * p_dp)

    walk(counts.success, counts.wrong_tier3, counts.wrong_other,
         counts.decline_abstain, counts.decline_parse, 0, 1.0)

    p_succ_total = sum(p_success)
    return {
        "p_success_by_attempt": p_success,
        "p_wrong_by_attempt": p_wrong,
        "p_success": p_succ_total,
        "p_wrong": sum(p_wrong),
        "p_wrong_tier3": sum(p_wrong_t3),
        "p_unresolved": unresolved,
        "e_attempts": e_attempts,
        "e_retries_abstain": e_retry["abstain"],
        "e_retries_parse": e_retry["parse"],
        # Conditional on completing, how many BCI attempts it cost.
        "e_attempts_given_success": (
            sum((i + 1) * p for i, p in enumerate(p_success)) / p_succ_total
            if p_succ_total > 0 else float("nan")),
        "success_per_100_attempts": (
            100.0 * p_succ_total / e_attempts if e_attempts > 0 else float("nan")),
    }


def simulate(counts: OutcomeCounts, rng, n_traj: int = 200_000,
             max_attempts: int = MAX_ATTEMPTS, with_replacement: bool = False) -> dict:
    """Monte Carlo replay that EXECUTES against a real AssistiveSandbox.

    This exists only to check `outcome_distribution`. It is deliberately written
    from the trajectory downwards rather than sharing any code with the closed
    form, because two implementations that share their arithmetic check nothing.
    Reported numbers always come from the closed form.
    """
    from nag.sandbox import AssistiveSandbox

    # One representative action per category. Which tier-1 action stands in for
    # "wrong, not tier 3" does not matter; that it writes to the sandbox does.
    pool = (["S"] * counts.success + ["W3"] * counts.wrong_tier3
            + ["WO"] * counts.wrong_other + ["DA"] * counts.decline_abstain
            + ["DP"] * counts.decline_parse)
    ACTION = {"S": "save_note", "W3": "record_refusal", "WO": "play_media"}

    n_succ = n_wrong = n_wrong3 = n_unres = n_changes = 0
    total_attempts = 0
    for _ in range(n_traj):
        sb = AssistiveSandbox()
        remaining = list(pool)
        resolved = False
        used = 0
        for _attempt in range(max_attempts):
            if not remaining:
                break
            idx = rng.integers(len(remaining))
            kind = remaining[idx] if with_replacement else remaining.pop(idx)
            used += 1
            if kind in ("S", "W3", "WO"):
                sb.execute(ACTION[kind])
                n_changes += int(sb.changed_from_initial())
                if kind == "S":
                    n_succ += 1
                else:
                    n_wrong += 1
                    n_wrong3 += int(kind == "W3")
                resolved = True
                break
        total_attempts += used
        n_unres += int(not resolved)

    return {
        "p_success": n_succ / n_traj,
        "p_wrong": n_wrong / n_traj,
        "p_wrong_tier3": n_wrong3 / n_traj,
        "p_unresolved": n_unres / n_traj,
        "e_attempts": total_attempts / n_traj,
        "n_state_changes": n_changes,
    }


def _endpoints_for(frame: pd.DataFrame, commands, with_replacement: bool = False) -> dict:
    """Uniform-weighted average of the per-command exact distributions.

    Uniform over commands, NOT over episodes: the estimand is the outcome of a
    task drawn uniformly from the nine assistive commands, so a command that
    happens to carry one more donor episode must not carry more weight.
    """
    keys = ("p_success", "p_wrong", "p_wrong_tier3", "p_unresolved", "e_attempts",
            "e_retries_abstain", "e_retries_parse")
    acc = {k: [] for k in keys}
    for c in commands:
        sub = frame[frame["assigned_command"] == c]
        if sub.empty:
            continue                      # dropped command, counted by the caller
        d = outcome_distribution(classify(sub), with_replacement=with_replacement)
        for k in keys:
            acc[k].append(d[k])
    if not acc["p_success"]:
        return {k: float("nan") for k in (*keys, "success_per_100_attempts")}
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["success_per_100_attempts"] = (100.0 * out["p_success"] / out["e_attempts"]
                                       if out["e_attempts"] > 0 else float("nan"))
    return out


def replay_panel(frames: dict, commands, n_boot: int = 2000, seed: int = 20260901,
                 with_replacement: bool = False) -> pd.DataFrame:
    """Endpoints with joint participant-cluster bootstrap intervals.

    ONE participant draw per replicate is applied to EVERY policy. The policies
    ran the same 200 episodes, so resampling them independently would discard
    that pairing and widen every contrast to describe a study nobody ran. This
    is the same posture as `nag.paired_bootstrap.paired_risk_difference`.
    """
    names = list(frames)
    participants = sorted(set().union(*(set(f["participant_id"]) for f in frames.values())))
    point = {n: _endpoints_for(frames[n], commands, with_replacement) for n in names}

    rng = np.random.default_rng(seed)
    boot = {n: [] for n in names}
    dropped = 0
    by_part = {n: {p: g for p, g in frames[n].groupby("participant_id")} for n in names}
    for _ in range(n_boot):
        draw = rng.choice(participants, size=len(participants), replace=True)
        for n in names:
            parts = [by_part[n][p] for p in draw if p in by_part[n]]
            if not parts:
                dropped += 1
                continue
            boot[n].append(_endpoints_for(pd.concat(parts, ignore_index=True),
                                          commands, with_replacement))

    rows = []
    for n in names:
        row = {"policy": n, **point[n]}
        reps = boot[n]
        for k in point[n]:
            vals = np.array([r[k] for r in reps], dtype=float)
            vals = vals[~np.isnan(vals)]
            row[f"{k}_lo"] = float(np.percentile(vals, 2.5)) if vals.size else float("nan")
            row[f"{k}_hi"] = float(np.percentile(vals, 97.5)) if vals.size else float("nan")
        row["n_boot_dropped"] = dropped
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(frames: dict, policy_a: str, policy_b: str, endpoints,
                       commands=None, n_boot: int = 2000, seed: int = 20260901,
                       with_replacement: bool = False) -> dict:
    """Bootstrap CIs for policy_a minus policy_b on SEVERAL endpoints at once.

    This is `replay_panel`'s pairing discipline applied to a difference
    instead of two marginals: one joint participant draw per replicate,
    applied to BOTH policies, and every endpoint's difference is taken WITHIN
    that single replicate. Differences are never built by combining two
    independently-computed marginal intervals after the fact -- that
    reconstruction throws away the covariance the paired design exists to
    exploit and widens the interval to describe a study nobody ran, which is
    exactly the error `nag.paired_bootstrap` was written to avoid for the
    arm-level contrasts.

    Computing several endpoints from ONE bootstrap pass, rather than one pass
    per endpoint, matters here beyond tidiness: `_endpoints_for` already
    returns every endpoint on every replicate, so re-running the pass once per
    endpoint recomputes identical draws for no new information. A driver
    reporting the full endpoint set for a single contrast should call this,
    not loop `paired_difference`.

    Returns ``{endpoint: {...}}``, one dict per endpoint with the same shape
    `paired_difference` returns.
    """
    fa, fb = frames[policy_a], frames[policy_b]
    if commands is None:
        commands = sorted(set(fa["assigned_command"]) | set(fb["assigned_command"]))
    participants = sorted(set(fa["participant_id"]) | set(fb["participant_id"]))

    point_a = _endpoints_for(fa, commands, with_replacement)
    point_b = _endpoints_for(fb, commands, with_replacement)

    rng = np.random.default_rng(seed)
    by_a = {p: g for p, g in fa.groupby("participant_id")}
    by_b = {p: g for p, g in fb.groupby("participant_id")}
    diffs = {ep: [] for ep in endpoints}
    dropped = {ep: 0 for ep in endpoints}
    for _ in range(n_boot):
        draw = rng.choice(participants, size=len(participants), replace=True)
        parts_a = [by_a[p] for p in draw if p in by_a]
        parts_b = [by_b[p] for p in draw if p in by_b]
        if not parts_a or not parts_b:
            for ep in endpoints:
                dropped[ep] += 1
            continue
        ra = _endpoints_for(pd.concat(parts_a, ignore_index=True), commands, with_replacement)
        rb = _endpoints_for(pd.concat(parts_b, ignore_index=True), commands, with_replacement)
        for ep in endpoints:
            va, vb = ra[ep], rb[ep]
            if np.isnan(va) or np.isnan(vb):
                dropped[ep] += 1
                continue
            diffs[ep].append(va - vb)

    out = {}
    for ep in endpoints:
        vals = np.array(diffs[ep], dtype=float)
        if vals.size:
            lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        else:
            lo = hi = float("nan")
        out[ep] = {
            "policy_a": policy_a, "policy_b": policy_b, "endpoint": ep,
            "a": point_a[ep], "b": point_b[ep], "difference": point_a[ep] - point_b[ep],
            "lo": lo, "hi": hi, "n_boot": n_boot, "n_boot_dropped": dropped[ep],
            "method": "joint participant-cluster percentile bootstrap on the difference",
        }
    return out


def paired_difference(frames: dict, policy_a: str, policy_b: str, endpoint: str,
                      commands=None, n_boot: int = 2000, seed: int = 20260901,
                      with_replacement: bool = False) -> dict:
    """Bootstrap CI for policy_a minus policy_b on ONE endpoint.

    Thin single-endpoint wrapper around `paired_differences`; see there for
    the pairing rationale. Calling this in a loop over several endpoints for
    the same policy pair repeats the same 2000-replicate draw once per
    endpoint -- correct, but wasteful -- so a caller that wants the whole
    endpoint set for one contrast should call `paired_differences` directly.
    """
    return paired_differences(frames, policy_a, policy_b, [endpoint], commands,
                              n_boot, seed, with_replacement)[endpoint]
