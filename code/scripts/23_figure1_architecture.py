"""TASK 23: Figure 1, the agentic BCI architecture (panel a) composed with the
risk-coverage frontier (panel b).

Panel a is new: a matplotlib-primitives schematic of the pipeline this paper is
about, drawn once so the reader has the architecture before the first data
panel. It has no data in it and computes nothing -- boxes, arrows and a single
vertical rule marking the decoder-to-agent action boundary the manuscript's
title names ("preserving decoder uncertainty at the action boundary").

Panel b is NOT redrawn here. It is the paper's central claim, and
`code/scripts/11_figures.py:draw_risk_coverage()` already owns it; this script
imports that function and calls it on a caller-supplied axis. A second copy of
that plot would diverge from the first the first time either one is touched,
so none is written. `11_figures.py` also still builds panel b standalone as
`figure1_risk_coverage_frontier.pdf/.png`; that build is untouched by this
file except for one additive, keyword-only, default-preserving parameter on
`draw_risk_coverage` (`legend_bbox_to_anchor`), needed because panel b's
axes-fraction height differs inside this two-panel composition.

MAKES NO API CALL AND SPENDS NOTHING. Presentation only: no new experiment, no
new model call, no re-analysis. Every number panel b draws is unchanged from
the standalone figure -- see 11_figures.py:draw_risk_coverage for where each
one comes from.

Run: UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv PYTHONPATH=code python3 \
     code/scripts/23_figure1_architecture.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from nag.figstyle import (BLUE, LABEL, MUTED, RULE, SALMON,  # noqa: E402
                          WIDTH_DOUBLE, apply_style, panel_label, save)

FIGS = REPO_ROOT / "output" / "figures"


def _load_sibling(name: str, filename: str):
    """`11_figures.py` is not an importable module name (leading digit), and
    this repo's other scripts load numbered siblings this way -- see
    15_repeat_run.py's `_load_sibling` for the same pattern."""
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig11 = _load_sibling("_nag_fig11", "11_figures.py")


# --------------------------------------------------------------------------
# Panel a -- the agentic BCI architecture, drawn with matplotlib primitives
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, text, *, edge=RULE, fill="white", fontsize=6.5,
       textcolor=LABEL, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=lw, edgecolor=edge, facecolor=fill,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, zorder=4)
    return x + w


def arrow(ax, x0, y0, x1, y1, *, colour=RULE, style="-|>", lw=0.8, mutation=6.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=mutation, linewidth=lw,
                                 color=colour, zorder=2))


def draw_architecture(ax) -> None:
    """Where the decoder's two outputs go, which is the question the study asks.

    The first version of this panel was a chain of six identically-weighted
    hollow rectangles -- neural signal, decoder, decoded string + confidence,
    agent, tool call, action -- with the two policies as a second chain
    underneath. Peer review's verdict was that it read as a generic
    auto-generated flowchart rather than as scientific visualisation, and that
    the top row implied the agent always receives the confidence value. Under
    enforced control it specifically does not, and that is the whole design.

    So this panel is built around ROUTING rather than around components. One
    decoder emits two signals. They cross a boundary. On the far side, two
    architectures differ only in where the confidence goes: into the agent's
    prompt, or into a gate the agent never sees. A reader can follow the
    confidence line with a finger and watch it arrive somewhere different.

    Boxes are computational components and there are only four, one of them
    repeated per lane. Everything else is a label on an arrow or a terminal
    outcome, which is what keeps the panel from reading as an organisation
    chart. Colour is the paper's: BLUE for the deterministic side, the decoder
    that produces the confidence and the gate that enforces it; SALMON for the
    one segment where that value becomes prompt text and the agent's own
    judgement takes over; neutral RULE for the agent, which is neither.

    No data, no computation, no clip-art. Drawn in fixed data coordinates with
    `set_aspect("equal")` so the geometry cannot be stretched by whatever axes
    box the caller supplies.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 90)
    ax.set_ylim(1.2, 34.6)
    ax.set_aspect("equal")

    # Type sizes, not axes size, are what made the first draft of this panel
    # hard to read: a point is a point whatever box you draw it in, so
    # enlarging the axes changed nothing. These sit at the 7 pt spec ceiling.
    #
    # The drawing is enlarged instead by narrowing the data range, 102 -> 96 ->
    # 90 across two passes, and by giving panel a its own wider axes box in
    # main(). Scale is axes width divided by the x range, so both act on it.
    # The left block was compressed to pay for the narrower range: it held 31
    # units for three elements while the two lanes, which carry the argument,
    # had 56.
    TINY, SMALL, BOXED = 6.0, 6.5, 7.0
    ADV_Y, ENF_Y = 27.0, 11.0          # lane centre lines
    LANE_X, AGENT_X, AGENT_W = 32.5, 46.0, 13.0
    BOUND_X, BOUND_W = 28.2, 2.0

    # -- the boundary, a band rather than a dashed rule ---------------------
    # The rotated micro-label on the old dashed line was the single strongest
    # "draft schematic" signal in the panel. A pale band with a horizontal
    # caption beside it says the same thing and stays behind the content. The
    # caption is right-aligned to the band rather than centred on it, so it
    # cannot run into the ADVISORY lane title.
    ax.add_patch(plt.Rectangle((BOUND_X, 2.0), BOUND_W, 29.5, facecolor="#EDEFF1",
                               edgecolor="none", zorder=0))
    ax.text(BOUND_X - 0.5, 32.4, "decoder-to-agent boundary",
            ha="right", va="bottom", fontsize=SMALL, color=LABEL)

    # -- left of the boundary: one decoder, two outputs ---------------------
    # A stylised P300 trace rather than a box labelled "Neural signal", which
    # is the most generic object a schematic can contain. The decoder sits
    # between the two lanes rather than above them, which is what closes the
    # dead quadrant the first layout left in the lower left.
    xs = [2.0 + 6.5 * i / 60 for i in range(61)]
    ys = [19.0 + 0.8 * math.sin(i / 60 * 9.4)
          - 1.7 * math.exp(-((i / 60 - 0.62) ** 2) / 0.004) for i in range(61)]
    ax.plot(xs, ys, color=LABEL, lw=0.8, solid_capstyle="round", zorder=3)
    ax.text(5.25, 15.6, "P300 EEG", ha="center", va="top", fontsize=TINY, color=LABEL)
    arrow(ax, 9.1, 19.0, 10.8, 19.0)

    box(ax, 11.0, 15.0, 14.0, 8.0, "P300\ndecoder", edge=BLUE, fontsize=BOXED)

    # Right-aligned to the band, and clear of the decoder box above and
    # below it. Centred on their arrows they overran the arrow head onto the
    # boundary band; placed level with them they sat on the box border.
    arrow(ax, 25.2, 21.2, 27.9, 21.2, colour=LABEL)
    ax.text(27.7, 23.4, "decoded text", ha="right", va="bottom",
            fontsize=TINY, color=LABEL)
    arrow(ax, 25.2, 16.8, 27.9, 16.8, colour=BLUE)
    ax.text(27.7, 14.6, "confidence", ha="right", va="top",
            fontsize=TINY, color=LABEL)

    # -- advisory lane: the confidence reaches the agent --------------------
    ax.text(LANE_X, 31.7, "ADVISORY", ha="left", va="bottom", fontsize=BOXED,
            fontweight="bold", color=LABEL)
    ax.plot([LANE_X, 43], [29.0, 29.0], color=LABEL, lw=0.9, zorder=2)
    ax.text(LANE_X + 0.5, 29.7, "decoded text", ha="left", va="bottom",
            fontsize=TINY, color=LABEL)
    ax.plot([LANE_X, 43], [25.0, 25.0], color=SALMON, lw=1.3, zorder=2)
    # ONE label on the salmon line, not "prompt" above and "confidence" below.
    # Split across the line, a reader could not tell at a glance whether the
    # signal was the confidence or the prompt; it is the confidence, carried as
    # prompt text, and that is what the experiment does.
    ax.text(LANE_X + 0.5, 24.3, "confidence in prompt", ha="left", va="top",
            fontsize=TINY, color=LABEL)
    ax.plot([43, 43], [25.0, 29.0], color=RULE, lw=0.9, zorder=2)   # the merge
    arrow(ax, 43, ADV_Y, AGENT_X, ADV_Y)

    box(ax, AGENT_X, ADV_Y - 3.5, AGENT_W, 7.0, "AI agent", fontsize=BOXED)
    arrow(ax, AGENT_X + AGENT_W, ADV_Y, 66.5, ADV_Y)
    ax.text(62.7, ADV_Y + 0.8, "tool call", ha="center", va="bottom",
            fontsize=TINY, color=LABEL)
    ax.text(67.5, ADV_Y + 0.9, "external state change", ha="left", va="center",
            fontsize=BOXED, color=LABEL)
    # One line, not a stacked list of three, which read as diagram software.
    ax.text(67.5, ADV_Y - 1.9, "message  ·  nurse call  ·  preference record",
            ha="left", va="center", fontsize=TINY, color=MUTED)

    # -- enforced lane: the confidence bypasses the agent -------------------
    ax.text(LANE_X, 14.6, "ENFORCED", ha="left", va="bottom", fontsize=BOXED,
            fontweight="bold", color=LABEL)
    arrow(ax, LANE_X, ENF_Y, AGENT_X, ENF_Y, colour=LABEL)
    ax.text(LANE_X + 0.5, ENF_Y + 0.7, "decoded text", ha="left", va="bottom",
            fontsize=TINY, color=LABEL)

    box(ax, AGENT_X, ENF_Y - 3.5, AGENT_W, 7.0, "AI agent", fontsize=BOXED)
    arrow(ax, AGENT_X + AGENT_W, ENF_Y, 67.0, ENF_Y)
    ax.text(63.0, ENF_Y + 0.8, "proposal", ha="center", va="bottom",
            fontsize=TINY, color=LABEL)

    # The confidence runs along the floor of the lane, under the agent, and up
    # into the gate. That detour IS the enforced architecture: the value never
    # reaches the model, and a reader can follow the blue line and see it.
    ax.plot([LANE_X, 72.0], [3.5, 3.5], color=BLUE, lw=1.3, zorder=2)
    ax.text(LANE_X + 0.5, 4.2, "confidence", ha="left", va="bottom",
            fontsize=TINY, color=LABEL)
    arrow(ax, 72.0, 3.5, 72.0, ENF_Y - 3.5, colour=BLUE)

    box(ax, 67.0, ENF_Y - 3.5, 10.0, 7.0, "gate", edge=BLUE, fontsize=BOXED)
    arrow(ax, 77.0, ENF_Y + 2.0, 79.5, ENF_Y + 2.0)
    ax.text(80.1, ENF_Y + 2.0, "execute", ha="left", va="center",
            fontsize=SMALL, color=LABEL)
    arrow(ax, 77.0, ENF_Y - 2.0, 79.5, ENF_Y - 2.0)
    ax.text(80.1, ENF_Y - 2.0, "block", ha="left", va="center",
            fontsize=SMALL, color=LABEL)


# --------------------------------------------------------------------------
# Panel b -- make the coverage truncation unmistakable
# --------------------------------------------------------------------------
def _mark_truncated_axis(ax) -> None:
    """The coverage axis starts at 0.32, not 0. Two redundant cues rather
    than relying on the reader to notice the left tick is not zero: an
    explicit 0.32 tick label (so the number itself states the truncation),
    and a break mark at the bottom-left corner of the axes -- the standard
    matplotlib idiom for "this axis does not start at its natural origin".
    """
    xt = sorted(set([0.32, 0.4, 0.6, 0.8, 1.0]))
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:.2f}" if v == 0.32 else f"{v:.1f}" for v in xt])

    # Two short diagonals straddling the bottom-left corner of the axes, not
    # dropped below it: the first attempt placed them under the spine, in the
    # same band as the "0.32" tick label, and they read as a strikethrough
    # over the number rather than a break mark next to it.
    d = 0.010
    kwargs = dict(transform=ax.transAxes, color=LABEL, clip_on=False, lw=0.9)
    ax.plot([-d, d], [-d, d], **kwargs)
    ax.plot([-d + 0.02, d + 0.02], [-d, d], **kwargs)


def main() -> int:
    apply_style()
    d = fig11.load_principal()
    cells = {c.name: c for c in fig11.enumerate_cells()}

    # Panel b is `draw_risk_coverage()` at close to its standalone size: the
    # standalone build (11_figures.py:fig1) gives that axis 4.85 in of height
    # in a 6.3 in figure and leaves 1.26 in below it for the legend, and every
    # annotation inside the function is placed assuming that room. Shrinking
    # the axis is what caused the legend, the "gate/advisory" callout and the
    # GLM label to collide during development; explicit axes positions (in
    # figure-fraction, computed from inches below) reproduce comparable room
    # for panel b while giving panel a its own fixed band above it, which
    # GridSpec's height_ratios/hspace do not make easy to reason about exactly.
    # 8.6 in is 218 mm, near a full journal page, and after panel b's type was
    # reset from 8.5-10.5 pt to the 5-7 pt the spec allows, most of that height
    # was empty field around a smooth curve. 7.55 in is 192 mm. bottom_margin
    # came down with it: 1.35 in reserved for a two-row 6.5 pt legend left a
    # visible band of nothing under the axis title.
    top_margin, gap, bottom_margin = 0.15, 0.42, 1.00
    axis_b_h = 4.1
    # Panel a is drawn at aspect 42:102 with set_aspect('equal'), so its axes
    # box is sized to that ratio rather than to whatever height is left over.
    # A box of the wrong shape would letterbox the schematic and shrink it,
    # which is the opposite of what this redraw is for.
    axis_a_h = round((0.995 - 0.025) * WIDTH_DOUBLE * 33.4 / 90, 3)
    left, right = 0.09, 0.985
    width = right - left
    # Panel a gets its own, wider box. Panel b spends the left margin on
    # its y-axis title; panel a has no axis and does not need it, and the
    # schematic's scale is its axes width divided by its x range, so the
    # margin was costing legibility for nothing.
    a_left, a_right = 0.025, 0.995
    a_width = a_right - a_left

    fig_h = top_margin + axis_a_h + gap + axis_b_h + bottom_margin
    fig = plt.figure(figsize=(WIDTH_DOUBLE, fig_h))
    ax_b = fig.add_axes([left, bottom_margin / fig_h, width, axis_b_h / fig_h])
    ax_a = fig.add_axes([a_left, (bottom_margin + axis_b_h + gap) / fig_h,
                         a_width, axis_a_h / fig_h])

    draw_architecture(ax_a)
    panel_label(ax_a, "a", x=-0.01, y=1.00)

    # The legend anchor is in AXES fraction, so shrinking the axis moves the
    # legend down in inches. -0.24 of 4.6 in cleared the shorter bottom
    # margin; -0.155 of 4.1 in does the same for this one.
    result = fig11.draw_risk_coverage(ax_b, d, cells,
                                      legend_bbox_to_anchor=(0.5, -0.215))
    _mark_truncated_axis(ax_b)
    panel_label(ax_b, "b", x=-0.09, y=1.02)

    save(fig, "Figure1", FIGS)
    print(f"panel b checks: worst_excursion={result['worst_excursion']:.4f}, "
         f"random_gate_risk={result['random_gate_risk']:.4f}")
    print(f"-> {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
