"""TASK 22: Figure 6, the repeated-attempt replay.

Reads output/tables/repeated_attempt_replay.csv (written by 21_repeated_attempt_replay.py)
for panels b and c, which need only the already-reported table values and their
joint-bootstrap intervals. Panel a needs a per-attempt breakdown that the table
does not carry, so it recomputes that breakdown directly from the same public
`nag.replay.classify` / `outcome_distribution` closed form, using the same
uniform-over-commands averaging the table build uses -- no new statistic, just
the existing estimator read out at finer grain. Panel d executes real donor
episodes through `nag.sandbox.AssistiveSandbox` to produce two genuine
trajectories; nothing in panel d is hand-written.

MAKES NO API CALL AND SPENDS NOTHING.

Run: UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv PYTHONPATH=code python3 \
     code/scripts/22_figure6.py
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from nag.naturalistic import COMMAND_TO_ACTION, NATURAL_COMMANDS  # noqa: E402
from nag.replay import classify, outcome_distribution              # noqa: E402
from nag.sandbox import AssistiveSandbox, TIER3_ACTIONS             # noqa: E402
import nag.openrouter as _openrouter                                # noqa: E402

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
ARM_COLOR = {"none": None, "advisory": None, "enforced": None}  # set after palette

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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE, GREY, SALMON = "#00468B", "#ADB6B6", "#FDAF91"
LABEL, MUTED = "#2B2B2B", "#6B7280"
GRID, ZERO = "#F3F4F6", "#9CA3AF"
DARK = "#1F2A44"
ARM_COLOR = {"none": BLUE, "advisory": SALMON, "enforced": GREY}


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(ZERO)
        ax.spines[s].set_linewidth(0.8)


def panel_label(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=LABEL)


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


# --------------------------------------------------------------------------
# Panel a -- cumulative completion by attempt
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


def panel_a(ax, frames: dict) -> None:
    style(ax)
    for policy in POLICY_ORDER:
        cum = _cumulative_completion(frames[policy])
        if policy.startswith("gate:"):
            color = BLUE if policy == "gate:lexical" else DARK
            ls = (0, (5, 2)) if policy == "gate:lexical" else (0, (1, 2.5))
            ax.plot([1, 2, 3], cum, color=color, lw=1.6, ls=ls, marker="o",
                    markersize=4, zorder=6)
        else:
            arm = policy.split(":")[-1]
            ax.plot([1, 2, 3], cum, color=ARM_COLOR[arm], lw=1.0, alpha=0.75,
                    marker="o", markersize=3, zorder=4)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("BCI attempt", fontsize=10.5, labelpad=6)
    ax.set_ylabel("Cumulative task completion", fontsize=10.5)
    ax.set_xlim(0.85, 3.15)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=9.5)
    panel_label(ax, "a", x=-0.20, y=1.05)


# --------------------------------------------------------------------------
# Panels b, c -- read straight from the committed table
# --------------------------------------------------------------------------
def _load_primary_table() -> pd.DataFrame:
    t = pd.read_csv(TABLES / "repeated_attempt_replay.csv")
    t = t[t["draw"] == "without_replacement"].set_index("policy")
    return t.loc[POLICY_ORDER]


def panel_b(ax, t: pd.DataFrame) -> None:
    """Successes per 100 BCI attempts.

    This is the panel that carries the signal. Task completion saturates
    between 0.945 and 1.000 and unintended tier-3 changes are identically zero,
    so a figure built only on those two endpoints would be three near-empty
    panels reporting a real but invisible result. Efficiency spans 62.0 to 94.1
    and is what an assistive-technology engineer actually trades against.
    """
    style(ax)
    y = np.arange(len(POLICY_ORDER))[::-1]
    val = t["success_per_100_attempts"]
    lo = t["success_per_100_attempts_lo"]
    hi = t["success_per_100_attempts_hi"]
    colors = [BLUE if p.startswith("gate:") else ARM_COLOR[p.split(":")[1]]
              for p in POLICY_ORDER]
    ax.barh(y, val, color=colors, zorder=3, height=0.62)
    ax.errorbar(val, y, xerr=[val - lo, hi - val], fmt="none",
                ecolor=LABEL, elinewidth=0.9, capsize=2, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([POLICY_LABEL[p] for p in POLICY_ORDER], fontsize=8.2)
    ax.set_xlabel("Successful tasks per 100 BCI attempts", fontsize=10, labelpad=6)
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=8.5)
    panel_label(ax, "b", x=-0.46, y=1.03)


def panel_c(ax, t: pd.DataFrame) -> None:
    """Unintended state changes per 100 attempted tasks.

    Tier 3 is drawn even though it is identically zero for every policy. An
    endpoint that came out at zero is a result, and a panel that silently
    omitted it would let a reader assume it was never measured.
    """
    style(ax)
    y = np.arange(len(POLICY_ORDER))[::-1]
    total = t["p_wrong"] * 100
    tier3 = t["p_wrong_tier3"] * 100
    lo, hi = t["p_wrong_lo"] * 100, t["p_wrong_hi"] * 100
    ax.barh(y, total - tier3, color=GREY, zorder=3, height=0.62)
    ax.barh(y, tier3, left=total - tier3, color=SALMON, zorder=3, height=0.62)
    ax.errorbar(total, y, xerr=[total - lo, hi - total], fmt="none",
                ecolor=LABEL, elinewidth=0.9, capsize=2, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([])                      # shares panel b's row labels
    ax.tick_params(left=False, labelleft=False)
    ax.set_xlabel("Unintended state changes\nper 100 attempted tasks", fontsize=10, labelpad=6)
    ax.tick_params(labelsize=8.5)
    ax.text(0.97, 0.02, "tier 3 = 0.00 in every policy", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")
    panel_label(ax, "c", x=-0.06, y=1.03)


# --------------------------------------------------------------------------
# Panel d -- two real trajectories, executed through the sandbox
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
    Nothing here is hand-written; every field printed comes from `sb.execute`.
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


def _format_trajectory(policy: str, command: str, steps: list, outcome: str) -> str:
    lines = [f"Policy: {POLICY_LABEL[policy]}", f"Intended: {command.upper()}"]
    for s in steps:
        if s["kind"] == "decline":
            lines.append(f"  Attempt {s['attempt']}  {s['reason']:<15} no state change")
        else:
            diff = _state_diff_text(s["before"], s["after"])
            tag = "correct action" if s["kind"] == "success" else "WRONG action"
            lines.append(f"  Attempt {s['attempt']}  {tag}: {s['action']:<16} {diff}")
    if outcome == "completed":
        lines.append(f"  Outcome: completed after {len(steps)} BCI attempt(s)")
    elif outcome == "wrong":
        lines.append("  Outcome: unintended state change, terminal")
    else:
        lines.append("  Outcome: unresolved after three BCI attempts")
    return "\n".join(lines)


def panel_d(ax, frames: dict) -> None:
    ax.axis("off")
    completed_3, wrong_1 = _find_examples(frames)
    if completed_3 is None or wrong_1 is None:
        raise SystemExit("could not find both illustrative trajectory patterns "
                         "in the deterministic search order; widen the search, "
                         "do not hand-write a substitute")
    text = (_format_trajectory(*completed_3) + "\n\n"
           + _format_trajectory(*wrong_1))
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=8.0,
           family="monospace", va="top", ha="left", color=LABEL,
           linespacing=1.55)
    panel_label(ax, "d", x=-0.04, y=1.05)


def main() -> int:
    _forbid_network_calls()
    frames = load_frames()
    t = _load_primary_table()

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.06,
                                          "height_ratios": [0.82, 1.0],
                                          "width_ratios": [1.0, 0.88]})
    panel_a(axes[0, 0], frames)
    panel_d(axes[0, 1], frames)
    panel_b(axes[1, 0], t)
    panel_c(axes[1, 1], t)
    # NOT sharey(): it re-enables the tick labels panel_c blanked, drawing
    # panel c's row labels straight over panel b's bars. Match limits instead.
    axes[1, 1].set_ylim(axes[1, 0].get_ylim())

    handles = [mlines.Line2D([], [], color=ARM_COLOR["none"], lw=1.4, marker="o",
                            markersize=4, label="LLM, no uncertainty"),
              mlines.Line2D([], [], color=ARM_COLOR["advisory"], lw=1.4, marker="o",
                            markersize=4, label="LLM, decoder confidence advisory"),
              mlines.Line2D([], [], color=ARM_COLOR["enforced"], lw=1.4, marker="o",
                            markersize=4, label="LLM, decoder confidence enforced"),
              mlines.Line2D([], [], color=BLUE, lw=1.6, ls=(0, (5, 2)),
                            label="Lexical resolver + gate, no model"),
              mlines.Line2D([], [], color=DARK, lw=1.6, ls=(0, (1, 2.5)),
                            label="Exact matcher + gate, no model"),
              mpatches.Patch(color=GREY, label="Unintended change, not tier 3"),
              mpatches.Patch(color=SALMON, label="Unintended change, tier 3")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.075),
              ncol=3, fontsize=8.6, frameon=False, handletextpad=0.5, columnspacing=1.8)
    fig.text(0.5, -0.155,
             "Repeated-attempt empirical replay: three BCI attempts maximum, without "
             "replacement, wrong execution terminal, no simulated human correction. "
             "The agent is memoryless across attempts and never perceives the sandbox. "
             "The number of replayed trajectories does not increase the neural sample "
             "size, which remains 200 episodes from 46 participants.",
             ha="center", fontsize=8, color=MUTED, style="italic", wrap=True)

    save(fig, "figure6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
