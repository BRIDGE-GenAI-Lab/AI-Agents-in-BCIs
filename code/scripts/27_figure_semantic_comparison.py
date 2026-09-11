"""TASK 2 (Plan B): the semantic fair-comparison figure.

Draws the two panels the new Results subsection ("A Fair-Information
Comparison Gave the Agent a Legitimate Semantic Channel") needs.

  Panel a: the two resolver-plus-gate frontiers (exact and lexical), swept
           and read directly from `output/tables/semantic_fair_resolver_curves.csv`
           (Task 6's own output, not recomputed here, so this figure cannot
           drift from the curve the run actually produced), overlaid with
           each model's `fair:llm_vocab:advisory` fixed operating point and
           each model's `fair:hybrid_semantic_gate` swept curve. The hybrid
           curve IS recomputed here, via
           `nag.naturalistic.recorded_proposal_gate_curve` on the same
           101-point threshold grid `26_semantic_primary_table.py` uses,
           because hybrid's raw `covered` column is a proposal rate, not an
           operating point (see that script's docstring); recomputing it the
           same way is required to draw a genuine frontier rather than the
           single already-tabulated point at threshold zero.
  Panel b: matched-coverage forest across the 10-model panel, one row per
           (arm, model), for all three vocabulary-disclosed arms
           (advisory, enforced, hybrid). Position is the arm's own coverage;
           a dashed blue rule marks the lexical resolver's own ceiling
           (0.940) so a reader can see at a glance which cells sit beyond it.
           Marker fill encodes risk: filled when risk is exactly 0, open
           when it is not, with the risk value printed in the gutter -- the
           same "position plus printed value" grammar `11_figures.py::fig2()`
           uses for its matched-coverage difference column, applied here to
           coverage and risk instead of a single differenced statistic
           because these 30 cells do not share one matched baseline (only 14
           of 30 have a comparator-restricted matched point at all -- see
           `26_semantic_primary_table.py`'s docstring on `frontier_verdict`).

Every number drawn from `semantic_primary_comparison.csv` is read from that
table's own `coverage` and `risk` columns, never recomputed, so the figure
cannot disagree with the Results text it illustrates.

Run:
  PYTHONPATH=code /private/tmp/nag_venv/bin/python3 code/scripts/27_figure_semantic_comparison.py
  PYTHONPATH=code /private/tmp/nag_venv/bin/python3 code/scripts/check_figure_specs.py output/figures/fig_semantic_comparison.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

from nag.naturalistic import recorded_proposal_gate_curve  # noqa: E402
from nag.figstyle import (  # noqa: E402
    WIDTH_DOUBLE, BLUE, GREY, SALMON, LABEL, MUTED, RULE,
    apply_style, panel_label, save,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = REPO_ROOT / "output" / "intermediate" / "runs_semantic_fair"
TABLES = REPO_ROOT / "output" / "tables"
FIGS = REPO_ROOT / "output" / "figures"

THRESHOLD_GRID = np.linspace(0.0, 1.0, 101)  # matches 26_semantic_primary_table.py

# The 10-model expanded panel this experiment ran, beyond the 5-model primary
# benchmark panel used elsewhere in the manuscript.
SHORT = {
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "deepseek/deepseek-v4-flash": "DeepSeek v4 Flash",
    "google/gemini-3.7-flash": "Gemini 3.7 Flash",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "z-ai/glm-5.3-flash": "GLM 5.3 Flash",
    "mistralai/mistral-medium-3-5": "Mistral Medium 3.5",
    "moonshotai/kimi-k3": "Kimi K3",
    "nvidia/nemotron-3.5-lightning": "Nemotron 3.5 Lightning",
    "qwen/qwen3.8-max-0902": "Qwen3.8 Max",
    "x-ai/grok-4.6": "Grok 4.6",
}
MARKERS = {
    "anthropic/claude-sonnet-5": "o",
    "deepseek/deepseek-v4-flash": "s",
    "google/gemini-3.7-flash": "D",
    "openai/gpt-5.6-luna": "^",
    "z-ai/glm-5.3-flash": "v",
    "mistralai/mistral-medium-3-5": "P",
    "moonshotai/kimi-k3": "X",
    "nvidia/nemotron-3.5-lightning": "*",
    "qwen/qwen3.8-max-0902": "p",
    "x-ai/grok-4.6": "h",
}
MODEL_ORDER = list(SHORT)  # fixed row order, shared by both panels

ADVISORY = "fair:llm_vocab:advisory"
ENFORCED = "fair:llm_vocab:enforced"
HYBRID = "fair:hybrid_semantic_gate"

apply_style()


def _parquets(dirname: Path) -> list[Path]:
    return sorted(p for p in dirname.glob("*.parquet") if not p.name.startswith("._"))


def load_hybrid_curves() -> dict[str, pd.DataFrame]:
    """Per-model swept `fair:hybrid_semantic_gate` frontier.

    Recomputed from the raw recorded proposals rather than read from
    `semantic_primary_comparison.csv`, which carries only the summary point
    at coverage/risk (threshold 0), not the whole curve panel a needs.
    """
    curves = {}
    for path in _parquets(INTERMEDIATE):
        if "fair_hybrid_semantic_gate" not in path.name:
            continue
        rows = pd.read_parquet(path)
        model = rows["model"].iloc[0]
        curves[model] = recorded_proposal_gate_curve(
            rows["covered"], rows["faithful"], rows["confidence"],
            threshold_grid=THRESHOLD_GRID,
        )
    return curves


def panel_a(ax, resolver_curves: pd.DataFrame, hybrid_curves: dict, advisory: pd.DataFrame) -> None:
    lex = resolver_curves[resolver_curves.arm_name == "fair:lexical_resolver_gate"].sort_values("coverage")
    exact = resolver_curves[resolver_curves.arm_name == "fair:exact_resolver_gate"].sort_values("coverage")

    ax.fill_between(lex["coverage"], 0, lex["risk"], color=BLUE, alpha=0.055, zorder=1)
    ax.plot(exact["coverage"], exact["risk"], color=BLUE, lw=1.2, ls=(0, (4, 2)), zorder=3)
    ax.plot(lex["coverage"], lex["risk"], color=BLUE, lw=2.6, zorder=4)

    for m in MODEL_ORDER:
        if m not in hybrid_curves:
            continue
        c = hybrid_curves[m]
        ax.plot(c["coverage"], c["risk"], color=GREY, lw=0.9, alpha=0.55, zorder=2)

    for _, r in advisory.iterrows():
        ax.scatter(r["coverage"], r["risk"], s=42,
                   marker=MARKERS.get(r["model"], "o"), color=SALMON,
                   edgecolors="white", linewidths=0.7, zorder=5)

    ax.set_xlabel("Coverage (fraction of episodes acted on)", fontsize=7)
    ax.set_ylabel("Unfaithful execution among actions admitted", fontsize=7)
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, 0.095)
    ax.tick_params(labelsize=6.5)
    panel_label(ax, "a")

    handles = [
        mlines.Line2D([], [], color=BLUE, lw=2.6, label="Lexical resolver + gate (swept)"),
        mlines.Line2D([], [], color=BLUE, lw=1.2, ls=(0, (4, 2)), label="Exact resolver + gate (swept)"),
        mlines.Line2D([], [], color=GREY, lw=0.9, label="Hybrid arm (swept), one line per model"),
        mlines.Line2D([], [], color=SALMON, marker="o", lw=0, markersize=6,
                      markeredgecolor="white", label="Vocabulary-disclosed advisory arm (fixed point)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6, frameon=False,
              handletextpad=0.5, labelspacing=0.4)


def panel_b(fig, ax, primary: pd.DataFrame) -> None:
    from matplotlib.transforms import blended_transform_factory

    groups = [(HYBRID, "Hybrid architecture\n(model proposes, gate decides)", GREY, "o"),
              (ENFORCED, "Vocabulary-disclosed\nenforced arm", GREY, "s"),
              (ADVISORY, "Vocabulary-disclosed\nadvisory arm", SALMON, "D")]

    ypos, rows, headers, colors, markers = [], [], [], [], []
    y = 0.0
    for arm, header, color, marker in groups:
        sub = primary[primary.arm_name == arm].set_index("model").loc[
            [m for m in MODEL_ORDER if m in primary[primary.arm_name == arm]["model"].to_numpy()]
        ].reset_index()
        headers.append((y + 0.85, header))
        for _, r in sub.iterrows():
            rows.append(r)
            ypos.append(y)
            colors.append(color)
            markers.append(marker)
            y -= 1.0
        y -= 1.3
    rows = pd.DataFrame(rows).reset_index(drop=True)
    ypos = np.array(ypos)

    resolver_ceiling = 0.940
    ax.axvline(resolver_ceiling, color=BLUE, lw=1.2, ls=(0, (4, 2)), zorder=1)

    for yi, color, marker, (_, r) in zip(ypos, colors, markers, rows.iterrows()):
        filled = bool(r["risk"] <= 0.0)
        ax.scatter([r["coverage"]], [yi], s=46, marker=marker,
                   facecolors=color if filled else "white",
                   edgecolors=color, linewidths=1.1, zorder=4)

    ax.set_xlim(0.60, 1.04)
    ax.set_xlabel("Coverage (arm's own operating point)", fontsize=7)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{SHORT.get(r['model'], r['model'])}" for _, r in rows.iterrows()],
                       fontsize=6)
    ax.set_ylim(ypos.min() - 0.9, ypos.max() + 1.6)
    ax.tick_params(labelsize=6)
    for yh, header in headers:
        ax.text(0.02, yh, header, transform=blended_transform_factory(fig.transFigure, ax.transData),
                ha="left", va="center", fontsize=6.3, fontweight="bold", color=LABEL, linespacing=1.2)
    panel_label(ax, "b", x=-0.12, y=1.02)

    trans = blended_transform_factory(fig.transFigure, ax.transData)
    ax.text(0.975, ypos.max() + 1.2, "Risk", transform=trans, ha="center", va="center",
            fontsize=6.3, fontweight="bold", color=LABEL)
    for yi, (_, r) in zip(ypos, rows.iterrows()):
        ax.text(0.975, yi, f"{r['risk']:.3f}", transform=trans, ha="center", va="center",
                fontsize=6, color=MUTED)

    ax.text(resolver_ceiling, ypos.max() + 1.55, "Lexical resolver\nceiling (0.940)",
            ha="center", va="bottom", fontsize=6, color=BLUE, linespacing=1.1)

    handles = [
        mlines.Line2D([], [], color=GREY, marker="o", lw=0, markersize=6, markeredgecolor=GREY,
                      markerfacecolor=GREY, label="Zero risk"),
        mlines.Line2D([], [], color=GREY, marker="o", lw=0, markersize=6, markeredgecolor=GREY,
                      markerfacecolor="white", label="Nonzero risk"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.78, 0.0),
               ncol=2, fontsize=6, frameon=False, handletextpad=0.5, columnspacing=1.2)


def main() -> None:
    resolver_curves = pd.read_csv(TABLES / "semantic_fair_resolver_curves.csv")
    primary = pd.read_csv(TABLES / "semantic_primary_comparison.csv")
    advisory = primary[primary.arm_name == ADVISORY]
    hybrid_curves = load_hybrid_curves()

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.40)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.40, right=0.90, top=0.965, bottom=0.055)

    panel_a(ax_a, resolver_curves, hybrid_curves, advisory)
    panel_b(fig, ax_b, primary)

    save(fig, "fig_semantic_comparison", FIGS)


if __name__ == "__main__":
    main()
