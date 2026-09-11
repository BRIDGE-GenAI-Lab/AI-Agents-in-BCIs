"""TASK 22: Figure 5, the repeated-attempt replay (renumbered from Figure 6).

Reads output/tables/repeated_attempt_replay.csv (written by 21_repeated_attempt_replay.py)
for panels b, c and d, which need only the already-reported table values and their
joint-bootstrap intervals, plus output/tables/repeated_attempt_contrasts.csv for the
enforced-minus-advisory contrast panel b annotates. Panel c needs a per-attempt
breakdown the table does not carry, so it recomputes that breakdown directly from
the same public `nag.replay.classify` / `outcome_distribution` closed form, using
the same uniform-over-commands averaging the table build uses -- no new statistic,
just the existing estimator read out at finer grain. Panel a executes real donor
episodes through `nag.sandbox.AssistiveSandbox` to produce two genuine
trajectories; nothing in panel a is hand-written.

This is a presentation-only rewrite of the former 22_figure6.py: same data, same
estimators, new rendering. Panel a used to print the trajectories as monospaced
console text; it now draws them as a flow diagram. Panel b used to bar-chart all
seventeen policies; it now shows the five advisory-to-enforced transitions the
manuscript's text actually discusses. Panel c used to plot all seventeen
cumulative-completion curves; it now plots the envelope across the fifteen
model-arm curves against the two deterministic references. Panel d is new: the
three reference values the text calls out, with no seventeen-row table beside
them (that table is eTable 10).

MAKES NO API CALL AND SPENDS NOTHING.

Run: UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv PYTHONPATH=code python3 \
     code/scripts/22_figure5_replay.py
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from nag.figstyle import (BLUE, GREY, LABEL, MARKER, MUTED, RULE, SALMON,  # noqa: E402
                          WIDTH_DOUBLE, apply_style, panel_label, save)
from nag.naturalistic import NATURAL_COMMANDS                              # noqa: E402
from nag.replay import classify, outcome_distribution                      # noqa: E402
from nag.sandbox import AssistiveSandbox                                   # noqa: E402
import nag.openrouter as _openrouter                                       # noqa: E402

RUNS = REPO_ROOT / "output" / "intermediate" / "runs_natural"
TABLES = REPO_ROOT / "output" / "tables"
FIGS = REPO_ROOT / "output" / "figures"


def _forbid_network_calls() -> None:
    """See 21_repeated_attempt_replay.py for why this patches functions rather
    than checking `sys.modules`: importing NATURAL_COMMANDS-adjacent names
    from `nag.naturalistic` always pulls in `nag.openrouter`, so a
    module-presence check cannot distinguish "imported" from "called"."""
    def _refuse(name):
        def _raise(*args, **kwargs):
            raise RuntimeError(f"nag.openrouter.{name} called: this script must "
                               "make no API call and spends $0.00 by construction")
        return _raise
    _openrouter.chat = _refuse("chat")
    _openrouter.resolve_endpoint = _refuse("resolve_endpoint")


MODELS = ("claude-sonnet-5", "gemini-3.7-flash", "gpt-5.6-luna",
          "glm-5.3-flash", "deepseek-v4-flash")
SHORT = {
    "claude-sonnet-5": "Claude Sonnet 5",
    "deepseek-v4-flash": "DeepSeek v4 Flash",
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "glm-5.3-flash": "GLM 5.3 Flash",
}
ARM_LABEL = {"none": "no uncertainty", "advisory": "confidence, advisory",
            "enforced": "confidence, enforced"}
# Policy colour follows the manuscript-wide rule: salmon is the agent deciding
# (with or without a rendered confidence value), grey is the agent under
# enforced control, blue is reserved for the deterministic references. Shape
# carries the "none" vs "advisory" subtype within salmon.
ARM_COLOR = {"none": SALMON, "advisory": SALMON, "enforced": GREY}
ARM_MARKER = {"none": MARKER["advisory_none"], "advisory": MARKER["advisory_conf"],
             "enforced": MARKER["enforced"]}

POLICY_FILES = {
    "gate:confidence": "natural_confidence_gate_canonical.parquet",
    "gate:lexical": "natural_confidence_gate_lexical.parquet",
}
for slug, model in [("anthropic_claude_sonnet_5", "claude-sonnet-5"),
                    ("google_gemini_3_7_flash", "gemini-3.7-flash"),
                    ("openai_gpt_5_6_luna", "gpt-5.6-luna"),
                    ("z_ai_glm_5_3_flash", "glm-5.3-flash"),
                    ("deepseek_deepseek_v4_flash", "deepseek-v4-flash")]:
    for arm, cell in [("none", "factorial_none_advisory_s0"),
                      ("advisory", "factorial_decoder_confidence_advisory_s0"),
                      ("enforced", "factorial_decoder_confidence_enforced_s0")]:
        POLICY_FILES[f"{model}:{arm}"] = f"{slug}__{cell}.parquet"

# Deterministic, human-readable order: the two comparators, then each model's
# none/advisory/enforced triple, matching the order the manuscript's other
# naturalistic-benchmark tables already use.
POLICY_ORDER = ["gate:lexical", "gate:confidence"]
for model in MODELS:
    for arm in ("none", "advisory", "enforced"):
        POLICY_ORDER.append(f"{model}:{arm}")

POLICY_LABEL = {
    "gate:lexical": "Lexical resolver + gate",
    "gate:confidence": "Exact matcher + gate",
}
for model in MODELS:
    for arm in ("none", "advisory", "enforced"):
        POLICY_LABEL[f"{model}:{arm}"] = f"{SHORT[model]} · {ARM_LABEL[arm]}"


def _policy_parts(policy: str) -> tuple[str, str]:
    model, arm = policy.split(":")
    return model, arm


def load_frames() -> dict:
    frames = {}
    for name, fname in POLICY_FILES.items():
        f = RUNS / fname
        if f.name.startswith("._") or not f.exists():
            raise SystemExit(f"missing policy file {fname}")
        d = pd.read_parquet(f)
        if set(d["assigned_command"]) != set(NATURAL_COMMANDS):
            raise SystemExit(f"{fname}: command set does not match the frozen nine")
        frames[name] = d
    return frames


def _load_primary_table() -> pd.DataFrame:
    t = pd.read_csv(TABLES / "repeated_attempt_replay.csv")
    t = t[t["draw"] == "without_replacement"].set_index("policy")
    return t.loc[POLICY_ORDER]


def _load_contrasts() -> pd.DataFrame:
    c = pd.read_csv(TABLES / "repeated_attempt_contrasts.csv")
    return c.set_index(["model", "contrast", "endpoint"])


# --------------------------------------------------------------------------
# Panel a -- the retry mechanism, drawn as a diagram
# --------------------------------------------------------------------------
def _draw_trajectory(pool: pd.DataFrame, order: np.ndarray, max_attempts: int = 3):
    """Draw the command's donor pool WITHOUT REPLACEMENT in `order`, executing
    every non-decline outcome against a fresh `AssistiveSandbox`. Mirrors
    `outcome_distribution`'s rule exactly: decline consumes the attempt and
    moves to the next donor; success and wrong execution are terminal.
    """
    sb = AssistiveSandbox()
    rows = pool.iloc[order]
    steps = []
    outcome = "unresolved"
    for attempt, (_, row) in enumerate(rows.iterrows(), start=1):
        if attempt > max_attempts:
            break
        covered = bool(row["covered"]) if pd.notna(row["covered"]) else False
        if not covered:
            parse_failed = bool(row["parse_failed"]) if "parse_failed" in row and pd.notna(row["parse_failed"]) else False
            steps.append({"attempt": attempt, "kind": "decline",
                         "reason": "parse failure" if parse_failed else "abstained"})
            continue
        faithful = bool(row["faithful"]) if pd.notna(row["faithful"]) else False
        action = row["executed_name"]
        before = sb.snapshot()
        sb.execute(action)
        after = sb.snapshot()
        steps.append({"attempt": attempt, "kind": "success" if faithful else "wrong",
                     "action": action, "before": before, "after": after})
        outcome = "completed" if faithful else "wrong"
        break
    # Falling out of the loop without a break means every attempt drawn (up
    # to max_attempts, or fewer if the pool ran out) was a decline; `outcome`
    # is already "unresolved" by default, matching that.
    return steps, outcome, sb


def _state_diff_text(before: dict, after: dict) -> str:
    for key in before:
        if before[key] != after[key]:
            return f"{key} = {after[key]!r}"
    return "no visible change"  # should not happen for a real action


N_SHUFFLES = 40


def _find_examples(frames: dict):
    """Walk a fixed, deterministic order of (policy, command) pairs, each with
    ONE fixed without-replacement shuffle (rng seeded 20260901), and take the
    first pair whose real trajectory completes on attempt 3 and the first
    whose real trajectory ends in an unintended state change on attempt 1.
    Nothing here is hand-written; every field drawn comes from `sb.execute`.
    """
    completed_3 = None
    wrong_1 = None
    for policy in POLICY_ORDER:
        if policy.startswith("gate:"):
            continue  # the gates never decline by parse failure; least illustrative
        frame = frames[policy]
        for command in NATURAL_COMMANDS:
            pool = frame[frame["assigned_command"] == command].reset_index(drop=True)
            if len(pool) < 3:
                continue
            # zlib.crc32, NOT hash(): Python randomises string hashing per
            # process, so hash() here made the "deterministic search order"
            # differ on every run. The figure built once and then failed to
            # rebuild, which is the visible symptom of an unreproducible seed.
            key = f"{policy}|{command}".encode()
            rng = np.random.default_rng(20260901 + zlib.crc32(key) % 10_000)
            # One shuffle per pair is too narrow: whether a pair yields a
            # three-attempt completion depends on the draw, not only on the
            # pair. Deterministically try N_SHUFFLES draws from the same seeded
            # generator and take the first match, so the search covers the space
            # of real trajectories rather than one arbitrary sample of it.
            for _ in range(N_SHUFFLES):
                order = rng.permutation(len(pool))
                steps, outcome, sb = _draw_trajectory(pool, order)
                if (completed_3 is None and outcome == "completed"
                        and len(steps) == 3 and steps[0]["kind"] == "decline"
                        and steps[1]["kind"] == "decline"):
                    completed_3 = (policy, command, steps, outcome)
                if (wrong_1 is None and outcome == "wrong" and len(steps) == 1):
                    wrong_1 = (policy, command, steps, outcome)
                if completed_3 is not None and wrong_1 is not None:
                    return completed_3, wrong_1
    return completed_3, wrong_1


def box(ax, x, y, w, h, text, edge=RULE, fill="white", fontsize=6.0,
       textcolor=LABEL, family=None, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                linewidth=lw, edgecolor=edge, facecolor=fill,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
           fontsize=fontsize, color=textcolor, zorder=4, family=family)
    return x + w


def arrow(ax, x0, y0, x1, y1, colour=RULE, style="-|>", lw=0.7, mutation=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=mutation, linewidth=lw,
                                 color=colour, zorder=2))


# The former per-row "decode" box is gone: every attempt begins with a decode
# by construction, so stating it five times added a repeated box and a repeated
# arrow without adding information, and was the main reason the row felt dense
# rather than legible at print size. One box per attempt now carries the agent's
# output (decline, or the executed action), coloured by ARM_COLOR as before --
# blue survives in the panel only via the connecting arrows, which are RULE-
# coloured reference lines rather than a policy statement, so the colour
# grammar (salmon = advisory, grey = enforced) is unambiguous with one box.
ACTION_W, ROW_H, BOX_H = 3.35, 1.55, 0.98


def _draw_attempt_row(ax, y: float, attempt: int, kind: str, arm: str, *,
                      action: str = "", consequence: str = ""):
    """One attempt: label -> agent output (policy colour) -> plain-text
    consequence, compressed to what the panel needs at this grain. Whether a
    decline was a parse failure or an abstention is Figure 3's distinction,
    not this panel's; here every decline reads simply as "decline"."""
    x0 = 0.9
    ax.text(x0 - 0.15, y + BOX_H / 2, f"attempt {attempt}", ha="right", va="center",
           fontsize=6.8, color=MUTED)
    x1 = x0 + 0.4
    arrow(ax, x0 + 0.05, y + BOX_H / 2, x1, y + BOX_H / 2)
    if kind == "decline":
        x2 = box(ax, x1, y, ACTION_W, BOX_H, "decline", edge=ARM_COLOR[arm], fontsize=7.0)
    else:
        x2 = box(ax, x1, y, ACTION_W, BOX_H, action, edge=ARM_COLOR[arm], fontsize=7.0,
                 family="monospace")
    arrow(ax, x2, y + BOX_H / 2, x2 + 0.35, y + BOX_H / 2)
    ax.text(x2 + 0.52, y + BOX_H / 2, consequence, ha="left", va="center",
           fontsize=7.0, color=LABEL)


def _draw_trajectory_block(ax, y_top: float, policy: str, command: str,
                          steps: list, outcome: str) -> float:
    model, arm = _policy_parts(policy)
    header = f"Intended: {command.upper()}   ·   {SHORT[model]}, {ARM_LABEL[arm]}"
    ax.text(0.9, y_top, header, ha="left", va="top", fontsize=7.0,
           fontweight="bold", color=LABEL)
    y = y_top - 0.85
    for s in steps:
        if s["kind"] == "decline":
            _draw_attempt_row(ax, y - ROW_H, s["attempt"], "decline", arm,
                             consequence="retry")
        else:
            consequence = "completed" if s["kind"] == "success" else "unintended state change"
            _draw_attempt_row(ax, y - ROW_H, s["attempt"], s["kind"], arm,
                             action=s["action"], consequence=consequence)
        y -= ROW_H
    return y  # bottom of this block, in the same coordinate system


def panel_a(ax, frames: dict) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 9.6)
    completed_3, wrong_1 = _find_examples(frames)
    if completed_3 is None or wrong_1 is None:
        raise SystemExit("could not find both illustrative trajectory patterns "
                         "in the deterministic search order; widen the search, "
                         "do not hand-write a substitute")
    y = 7.05
    y = _draw_trajectory_block(ax, y, *completed_3[:2], completed_3[2], completed_3[3])
    y -= 0.65
    y = _draw_trajectory_block(ax, y, *wrong_1[:2], wrong_1[2], wrong_1[3])
    ax.set_ylim(y - 0.2, 7.4)
    panel_label(ax, "a", x=-0.015, y=1.02)


# --------------------------------------------------------------------------
# Panel b -- advisory to enforced, by model
# --------------------------------------------------------------------------
# Half-row offset applied to each arm within a model row.
OFFSET = 0.15
# Row-to-row spacing (data units); widened from the default 1.0 so the
# difference labels below have somewhere to grow into without reaching the
# row underneath. Label clearance and font size are read off this panel at
# render time -- see the note at the difference-label draw call.
ROW_SPACING = 1.30
# Gap between the enforced-arm marker and the start of its difference label.
LABEL_CLEARANCE = 0.24


def panel_b(ax, t: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    y = np.arange(len(MODELS))[::-1] * ROW_SPACING
    adv_rows = [f"{m}:advisory" for m in MODELS]
    enf_rows = [f"{m}:enforced" for m in MODELS]
    adv = t.loc[adv_rows, "success_per_100_attempts"].to_numpy()
    enf = t.loc[enf_rows, "success_per_100_attempts"].to_numpy()
    adv_lo = t.loc[adv_rows, "success_per_100_attempts_lo"].to_numpy()
    adv_hi = t.loc[adv_rows, "success_per_100_attempts_hi"].to_numpy()
    enf_lo = t.loc[enf_rows, "success_per_100_attempts_lo"].to_numpy()
    enf_hi = t.loc[enf_rows, "success_per_100_attempts_hi"].to_numpy()

    for i, model in enumerate(MODELS):
        yi = y[i]
        # Each interval takes its ARM's colour. Drawn in one shade of MUTED
        # they merged into a single line whose interior caps belonged to
        # whichever arm happened to be narrower, and the reader had no way to
        # tell which interval was which. The arrow is LABEL rather than RULE
        # for the same reason: at this size a rule-grey arrow between two
        # rule-grey intervals was indistinguishable from them.
        # The two intervals overlap on almost every row, so drawn on one line
        # the upper one occludes the middle of the lower and only its caps
        # survive. Offsetting each arm onto its own half-row gives each marker
        # its own interval to sit on; the arrow between them is then diagonal,
        # which is what makes the transition legible rather than the colour.
        ya, ye = yi + OFFSET, yi - OFFSET
        ax.errorbar(adv[i], ya, xerr=[[adv[i] - adv_lo[i]], [adv_hi[i] - adv[i]]],
                   fmt="none", ecolor=SALMON, elinewidth=1.0, capsize=1.4, zorder=3)
        ax.errorbar(enf[i], ye, xerr=[[enf[i] - enf_lo[i]], [enf_hi[i] - enf[i]]],
                   fmt="none", ecolor=GREY, elinewidth=1.0, capsize=1.4, zorder=3)
        arrow(ax, adv[i], ya, enf[i], ye, colour=LABEL, lw=0.7, mutation=5)
        ax.scatter(adv[i], ya, marker=ARM_MARKER["advisory"], s=24, color=SALMON,
                  edgecolor=LABEL, linewidth=0.35, zorder=5)
        ax.scatter(enf[i], ye, marker=ARM_MARKER["enforced"], s=24, color=GREY,
                  edgecolor=LABEL, linewidth=0.35, zorder=5)
        delta = contrasts.loc[(model, "enforced_minus_advisory",
                              "success_per_100_attempts"), "difference"]
        ax.text((adv[i] + enf[i]) / 2, ye - LABEL_CLEARANCE, f"{delta:+.1f}",
               ha="center", va="top", fontsize=6.2, color=LABEL)

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in MODELS])
    ax.set_xlabel("Successful tasks per 100 BCI attempts")
    ax.set_xlim(40, 100)
    # Bottom padding is larger than the top's: the difference label hangs
    # BELOW its row, so the last row's label had nowhere to go and sat on
    # the axis spine. The top row has no label under it to clear.
    ax.set_ylim(y.min() - 0.95, y.max() + 0.6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    handles = [mlines.Line2D([], [], color=SALMON, marker=ARM_MARKER["advisory"],
                            linestyle="none", markersize=4.2,
                            markeredgecolor=LABEL, markeredgewidth=0.35,
                            label="Advisory"),
              mlines.Line2D([], [], color=GREY, marker=ARM_MARKER["enforced"],
                            linestyle="none", markersize=4.2,
                            markeredgecolor=LABEL, markeredgewidth=0.35,
                            label="Enforced")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.00),
              ncol=2, frameon=False, fontsize=5.6,
             handletextpad=0.4, borderaxespad=0.1)
    panel_label(ax, "b", x=-0.03, y=1.04)


# --------------------------------------------------------------------------
# Panel c -- three-attempt completion, envelope not spaghetti
# --------------------------------------------------------------------------
def _cumulative_completion(frame: pd.DataFrame) -> np.ndarray:
    """Uniform-over-commands average of the closed-form per-attempt success
    probability, cumulated over the three attempts. Same weighting rule as
    `nag.replay._endpoints_for`: a command is not weighted by its donor count.
    """
    per_command = []
    for c in NATURAL_COMMANDS:
        sub = frame[frame["assigned_command"] == c]
        if sub.empty:
            continue
        per_command.append(outcome_distribution(classify(sub))["p_success_by_attempt"])
    return np.cumsum(np.mean(np.array(per_command), axis=0))


def panel_c(ax, frames: dict) -> None:
    curves = {p: _cumulative_completion(frames[p]) for p in POLICY_ORDER}
    model_arm = np.array([curves[p] for p in POLICY_ORDER if not p.startswith("gate:")])
    lo, hi = model_arm.min(axis=0), model_arm.max(axis=0)
    x = np.array([1, 2, 3])

    ax.fill_between(x, lo, hi, color=GREY, alpha=0.30, linewidth=0, zorder=1)
    ax.plot(x, curves["gate:lexical"], color=BLUE, lw=1.2, ls=(0, (5, 2)),
           marker="o", markersize=3.2, zorder=4)
    ax.plot(x, curves["gate:confidence"], color=BLUE, lw=1.0, ls=(0, (1, 1.6)),
           marker="o", markersize=3.0, markerfacecolor="white", zorder=4)

    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("BCI attempt")
    ax.set_ylabel("Cumulative task completion")
    ax.set_xlim(0.85, 3.15)
    ax.set_ylim(0.58, 1.01)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [mpatches.Patch(color=GREY, alpha=0.30,
                             label="Range, 15 model-arm policies"),
              mlines.Line2D([], [], color=BLUE, lw=1.2, ls=(0, (5, 2)),
                            label="Lexical reference + gate"),
              mlines.Line2D([], [], color=BLUE, lw=1.0, ls=(0, (1, 1.6)),
                            marker="o", markersize=3.0, markerfacecolor="white",
                            label="Exact matcher + gate")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=5.4,
             handletextpad=0.5, borderaxespad=0.15)
    panel_label(ax, "c", x=-0.20, y=1.04)


# --------------------------------------------------------------------------
# Panel d -- the deterministic references
# --------------------------------------------------------------------------
def panel_d(ax, t: pd.DataFrame) -> None:
    model_arm_rows = [f"{m}:{a}" for m in MODELS for a in ("none", "advisory", "enforced")]
    best_policy = t.loc[model_arm_rows, "success_per_100_attempts"].idxmax()
    best_model, best_arm = _policy_parts(best_policy)
    best_val = t.loc[best_policy, "success_per_100_attempts"]
    # The row label below names this only as "Best agent operating point" --
    # the identity is derived here (not hard-coded) so the label is always
    # correct, but it no longer prints in the artwork because the full
    # identity string didn't fit the panel. Printed here instead, so a
    # rebuild surfaces it if the derived best model ever changes; the legend
    # (edited separately) is where the identity now lives for the reader.
    print(f"  panel d: best agent operating point = {SHORT[best_model]} "
         f"({ARM_LABEL[best_arm]}), {best_val:.1f} successes/100 attempts")

    rows = [
        ("Lexical reference + gate", t.loc["gate:lexical", "success_per_100_attempts"],
        BLUE, "o", True),
        ("Best agent operating point",
        best_val, ARM_COLOR[best_arm], ARM_MARKER[best_arm], True),
        ("Exact matcher + gate", t.loc["gate:confidence", "success_per_100_attempts"],
        BLUE, "o", False),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (label, val, color, marker, filled) in zip(y, rows):
        ax.hlines(yi, 0, val, color=RULE, lw=0.8, zorder=2)
        face = color if filled else "white"
        ax.scatter(val, yi, marker=marker, s=34, color=face, edgecolor=color,
                  linewidth=1.0, zorder=4)
        ax.text(val + 2.5, yi, f"{val:.1f}", va="center", fontsize=6.2, color=LABEL)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.0)
    ax.set_xlabel("Successful tasks per 100 BCI attempts")
    ax.set_xlim(0, 100)
    # Bottom padding is larger than the top's: the difference label hangs
    # BELOW its row, so the last row's label had nowhere to go and sat on
    # the axis spine. The top row has no label under it to clear.
    ax.set_ylim(y.min() - 0.95, y.max() + 0.6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    panel_label(ax, "d", x=-0.18, y=1.06)


def main() -> int:
    _forbid_network_calls()
    apply_style()
    frames = load_frames()
    t = _load_primary_table()
    contrasts = _load_contrasts()

    # Panel a's share (first height_ratio) went from 0.34 to 0.75 and the
    # figure height from 5.8in to 6.70in together, so panels b/c/d keep
    # their previous absolute size in inches -- only panel a grows. hspace
    # stays 0.42: hspace=0.34 was tried here once and reverted, because it
    # pulled panel d's label onto panel b's x-axis title, and that fix is
    # orthogonal to how tall panel a's own row is.
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 6.70))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.75, 1.14, 1.16],
                         hspace=0.42, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    panel_a(ax_a, frames)
    panel_b(ax_b, t, contrasts)
    panel_c(ax_c, frames)
    panel_d(ax_d, t)

    fig.subplots_adjust(left=0.145, right=0.98, top=0.97, bottom=0.07)
    save(fig, "Figure5", FIGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
