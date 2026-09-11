"""Figure 3: the mechanism of non-action, abstention versus parse failure.

Reported until now only as running text and a supplement table. It carries one
of the paper's three headline claims: apparent agent safety is sometimes
malformed output rather than judgement. A model that declines by emitting no
valid tool call is not exercising oversight; a model that calls `abstain` is.
Without this distinction the model that broke most often would rank safest.

Presentation only: no new analysis, no model calls. Every number is read from
the committed `primary_abstention_mechanism.csv`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nag.figstyle import GREY, LABEL, MUTED, WIDTH_DOUBLE, apply_style, save

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGS = REPO_ROOT / "output" / "figures"

SHORT = {"anthropic/claude-sonnet-5": "Claude Sonnet 5",
         "google/gemini-3.7-flash": "Gemini 3.7 Flash",
         "openai/gpt-5.6-luna": "GPT-5.6 Luna",
         "z-ai/glm-5.3-flash": "GLM 5.3 Flash",
         "deepseek/deepseek-v4-flash": "DeepSeek v4 Flash"}


def main() -> int:
    apply_style()
    d = pd.read_csv(REPO_ROOT / "output" / "tables" / "primary_abstention_mechanism.csv")
    d = d.sort_values("declined_total")            # largest at the top after invert
    y = np.arange(len(d))

    # At double width the five rows would read as a thin strip stranded in a
    # tall canvas if height simply followed the old width:height ratio, so
    # height grows enough to keep bars looking deliberate rather than the full
    # 2.06x the width did.
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 3.15))
    ab = d["declined_by_calling_abstain"] * 100
    pf = d["declined_by_parse_failure"] * 100
    ax.barh(y, ab, color="white", edgecolor=GREY, linewidth=0.9, height=0.52,
            label="Declined by calling abstain", zorder=3)
    ax.barh(y, pf, left=ab, color=GREY, height=0.52,
            label="Declined by emitting no valid tool call", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in d["model"]])
    ax.set_xlabel("Episodes declined (%)")
    ax.set_xlim(0, 100)
    # Two entries fit one row at this width; stacking them the way the single
    # column figure had to only reproduces the cramped-for-space look on a
    # canvas that no longer needs it.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              frameon=False, fontsize=6.5, handlelength=1.4, columnspacing=1.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)

    # Annotate the inversion directly on the bar for gemini-3.7-flash: most of
    # its 70.0% decline rate is parse failure (54.3 pts), not abstention (15.7
    # pts). Stated in text and a supplement table elsewhere; this is the row
    # the claim is about, so it is labelled rather than left to the caption.
    # Label every segment wide enough to hold a number. Labelling only the two
    # the callout names would imply those two are special rather than merely
    # the largest. At double width a segment need only clear ~3 points before
    # a number reads cleanly, against 6 at single column.
    # The printed row total is `declined_total`, NOT the sum of the two plotted
    # segments. They disagree for deepseek-v4-flash: the CSV stores each field
    # rounded to three decimals, and 0.003 + 0.006 = 0.009 while declined_total
    # is 0.008. The manuscript quotes declined_total ("declined on 0.3% and 0.8%
    # of episodes"), so printing the segment sum put 0.9 in the figure against
    # 0.8 in the text. The 0.1-point difference in the drawn bar length is
    # invisible at this scale; a number that contradicts the manuscript is not.
    total = d["declined_total"].to_numpy() * 100
    for i, (a, p_) in enumerate(zip(ab, pf)):
        if a >= 3:
            ax.text(a / 2, i, f"{a:.1f}", ha="center", va="center",
                    fontsize=6.5, color=LABEL, zorder=5)
        if p_ >= 3:
            ax.text(a + p_ / 2, i, f"{p_:.1f}", ha="center", va="center",
                    fontsize=6.5, color="white", zorder=5)
        # A bar too short to hold either inline segment label (deepseek-v4-flash
        # at 0.8%, gpt-5.6-luna at 0.3%) reads as an absent row rather than a
        # near-zero measurement. Print the row total just past the bar end so
        # it is unambiguously data. Rows with an inline label already carry a
        # legible number, so they are left alone rather than doubling it.
        if a < 3 and p_ < 3:
            ax.text(a + p_ + 1.5, i, f"{total[i]:.1f}", ha="left", va="center",
                    fontsize=6.5, color=LABEL, zorder=5)

    # The callout must sit beside the row it describes. Anchored to gemini's own
    # y index: positioned by eye it landed next to GLM and silently attributed
    # gemini's finding to a different model.
    gi = int(np.where(d["model"].values == "google/gemini-3.7-flash")[0][0])
    grow = d.iloc[gi]
    # `declined_total`, not the sum of the two segments, for the same reason
    # the row totals use it: the CSV rounds each field to three decimals, so
    # for deepseek-v4-flash the parts sum to 0.9 while the total is 0.8, and
    # the manuscript quotes the total. Gemini's two happen to agree at 70.0,
    # so this changes nothing today and stops the call-out drifting from the
    # text if the underlying numbers ever move.
    gtot = grow["declined_total"] * 100
    gbar = (grow["declined_by_calling_abstain"] + grow["declined_by_parse_failure"]) * 100
    gpf = grow["declined_by_parse_failure"] * 100
    # Review: the prior wording ("most of this is parse failure, not judgement")
    # was rhetorical and the bar already shows the 15.7 vs 54.3 split. State the
    # two numbers instead, read from the same CSV and formatted with the same
    # one-decimal rule as the bar labels, so the callout cannot disagree with
    # the bars it sits beside. It reads as one line at this width; it would
    # only wrap if the figure were narrowed back toward single column.
    ax.annotate(f"{gpf:.1f} of {gtot:.1f} percentage points were parse failures",
                xy=(gbar, gi), xytext=(gbar + 3, gi),
                fontsize=6.5, color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5,
                                shrinkA=1, shrinkB=1))

    fig.tight_layout()
    save(fig, "Figure3", FIGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
