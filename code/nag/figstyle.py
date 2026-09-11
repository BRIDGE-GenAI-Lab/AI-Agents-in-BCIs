"""One visual language for every figure in this manuscript.

Two defects made this necessary and both were invisible inside any single
script. First, every figure was authored wider than the journal can print, from
188 mm to 483 mm against a 183 mm maximum, so the journal's own downscaling was
shrinking 8 pt labels to 4.5 pt and, for eFigure 1, to 3 pt. Authoring at final
width makes font sizes literal. Second, blue meant the deterministic gate in one
figure, the no-uncertainty arm in another, expected calibration error in a third
and a significant interval in a fourth. Each figure was internally sensible and
the manuscript had no shared language.

Colour carries policy and nothing else. Subtypes are distinguished by SHAPE,
because adding a colour is how the language was lost the first time.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Journal widths, in inches, at final printed size.
WIDTH_DOUBLE = 183 / 25.4
WIDTH_SINGLE = 89 / 25.4

BLUE = "#00468B"      # deterministic / interface control
SALMON = "#FDAF91"    # the agent decided (advisory)
GREY = "#ADB6B6"      # the agent under enforced control
LABEL = "#2B2B2B"     # all text
MUTED = "#6B7280"     # secondary text
RULE = "#9CA3AF"      # reference lines only, never decoration

POLICY_COLOURS = {"deterministic": BLUE, "advisory": SALMON, "enforced": GREY}
MARKER = {"advisory_none": "o", "advisory_conf": "D",
          "enforced": "s", "deterministic": "o"}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 7,                 # literal, because width is final
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.grid": False,             # Nature: no background gridlines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.6,
        "text.color": LABEL,
        "axes.labelcolor": LABEL,
        "xtick.color": LABEL,
        "ytick.color": LABEL,
        "pdf.fonttype": 42,             # editable vector text
        "ps.fonttype": 42,
    })


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04):
    """Bold lowercase, 8 pt, per Nature's panel convention."""
    if letter != letter.lower():
        raise ValueError(f"panel letters are lowercase in Nature figures: {letter!r}")
    return ax.text(x, y, letter, transform=ax.transAxes, fontsize=8,
                   fontweight="bold", color=LABEL, va="bottom")


def save(fig, name: str, figdir) -> None:
    """PDF is the master and keeps live text; PNG is a preview only.

    No bbox_inches='tight': it silently changes the figure's width, which is the
    one property the journal constrains and this module exists to control.
    """
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.pdf", dpi=600)
    fig.savefig(figdir / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")
