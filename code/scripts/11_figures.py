"""Publication figures (Nature Machine Intelligence house style).

Helvetica, colorblind-safe Lancet palette, 600 DPI PDF + PNG, no in-figure
titles, lowercase bold panel labels, top/right spines removed.

DATA ROUTING IS THE POINT OF THIS FILE, so it is stated first.

  PRINCIPAL (`output/intermediate/runs_principal/`, 6 cells, 1,065
  primary-eligible episodes, natural decoding-error prevalence 0.3408) is the
  source for every PRIMARY figure. `09_analysis.py` reads the same directory,
  and wherever a number already exists in `output/tables/primary_*.csv` this
  file reads the table rather than recomputing it, so a figure cannot disagree
  with the text.

  PILOT (`output/intermediate/runs/`, 34 cells, 100 episodes stratified 50:50
  on decoding error) is the EXPLORATORY design. It is the only source for the
  12-wording caution battery, the 3 scaffolds and the self-confidence arms,
  none of which were ever run at full pool. Its absolute rates are enriched by
  construction and are not deployment estimates, so Figure 3 says so on its
  face. The two datasets are never mixed inside one panel.

  NATURALISTIC (`output/intermediate/runs_natural/`, 200 constructed episodes
  drawn from real donors at the source pool's 0.340 error prevalence) is the
  source for Figure 5 only. Its absolute rates are benchmark risks at that
  prevalence, never deployment estimates: the error distribution is empirical
  but the nine commands are constructed.

  Figure 1  The risk-coverage frontier on the PRINCIPAL data. The
            deterministic gate's swept curve, the five enforced LLM arms
            swept on the same confidence, the random gate as a flat
            coverage-knob reference, and every advisory arm as the fixed
            POINT it is (no knob to sweep).
  Figure 2  Matched-coverage comparison across the 5-model panel, PRINCIPAL:
            risk against the gate at the arm's own coverage, and the matched
            gap with its participant-clustered 95% CI.
  Figure 3  The caution-wording battery, PILOT: unfaithful execution against
            parse failure, which separates "the wording worked" from "the
            wording broke the agent".
  Figure 4  Calibration of the reconstructed decoder confidence: per-study
            reliability, and cross-study transport. Neither depends on any
            model run.
  Figure 5  The naturalistic benchmark: five models x three arms against the
            deterministic lexical resolver paired with the same gate.

`anthropic/claude-sonnet-5` ran a frozen 500-episode subset of the principal
pool (Ruling 31), so every comparison involving it is drawn against the gate
RESTRICTED to those 500 episodes. Plotting its curve against the 1,065-episode
gate would compare two different episode sets and make the arm look better than
it is at low coverage; `primary_aurc.csv` does exactly that and is not used
here (see the note in `fig1`).

Run: PYTHONPATH=code uv run python3 code/scripts/11_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.design import enumerate_cells  # noqa: E402
from nag.riskcoverage import coverage_sweep_curve, rc_curve, risk_at_coverage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = REPO_ROOT / "output" / "intermediate"
PRINCIPAL_RUNS = INTERMEDIATE / "runs_principal"   # PRIMARY figures
PILOT_RUNS = INTERMEDIATE / "runs"                 # exploratory battery only
NATURAL_RUNS = INTERMEDIATE / "runs_natural"       # Figure 5 only
TABLES = REPO_ROOT / "output" / "tables"
FIGS = REPO_ROOT / "output" / "figures"

# The single enforced/advisory cells the principal run carries.
ADVISORY_DC = "factorial:decoder_confidence:advisory:s0"
ADVISORY_NONE = "factorial:none:advisory:s0"
ENFORCED_DC = "factorial:decoder_confidence:enforced:s0"

MIN_COVERAGE_PLOTTED = 0.10   # below this the risk is a fraction of a handful
PARSE_FAIL_LIMIT = 0.15       # pre-specified interpretability LABEL threshold

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

SHORT = {
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "deepseek/deepseek-v4-flash": "DeepSeek v4 Flash",
    "google/gemini-3.7-flash": "Gemini 3.7 Flash",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "z-ai/glm-5.3-flash": "GLM 5.3 Flash",
}
SUBSET_MODEL = "anthropic/claude-sonnet-5"   # frozen 500-episode subset
MARKERS = {
    "anthropic/claude-sonnet-5": "o",
    "deepseek/deepseek-v4-flash": "s",
    "google/gemini-3.7-flash": "D",
    "openai/gpt-5.6-luna": "^",
    "z-ai/glm-5.3-flash": "v",
}


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _read_runs(path: Path) -> pd.DataFrame:
    # "._*" are macOS AppleDouble sidecars: the volume is exFAT, which has no
    # resource forks, so they land as real files matching *.parquet.
    files = [f for f in sorted(path.glob("*.parquet")) if not f.name.startswith("._")]
    if not files:
        raise SystemExit(f"no parquet checkpoints in {path}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return d[d["error"].isna()].copy()


def load_principal() -> pd.DataFrame:
    d = _read_runs(PRINCIPAL_RUNS)
    # Same guard as 09_analysis.py's. If this ever silently reads the pilot,
    # every absolute risk in the figures becomes an enriched-sample number
    # drawn as a deployment estimate, and the panels would look perfectly fine.
    n_eps = d["episode_id"].nunique()
    if n_eps < 1000:
        raise SystemExit(
            f"primary figures loaded {n_eps} distinct episodes from "
            f"{PRINCIPAL_RUNS.name}; the principal pool is 1,065. Refusing to "
            "draw enriched pilot rates on a primary figure."
        )
    return d


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(ZERO)
        ax.spines[s].set_linewidth(0.8)


def panel_label(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=LABEL)


# --------------------------------------------------------------------------
# Figure 1 -- risk-coverage frontier, PRINCIPAL data
# --------------------------------------------------------------------------
def fig1(d, cells):
    """The paper's central claim in one panel, on the full principal pool.

    Advisory arms have no knob to sweep, so they enter as fixed POINTS; the
    gate and the enforced arms are swept curves over the same calibrated
    confidence.

    `anthropic/claude-sonnet-5` is compared against the gate RESTRICTED to its
    own 500 episodes. `primary_aurc.csv` does not do this: it scores that arm's
    500-episode area (0.146) against the 1,065-episode gate area (0.154), which
    makes the arm look better. On the shared 500 episodes the gate is ahead,
    0.142 to 0.146. That table is a known defect and no number from it is drawn
    here.
    """
    ng = d[d.cell == "nonllm_gate"]
    gate_full = rc_curve(ng["confidence"], ng["faithful"], ng["covered"],
                         cell=cells["nonllm_gate"])

    # The gate restricted to the subset model's episodes: the curve its arm is
    # actually matched against.
    subset_ids = set(d.loc[(d.model == SUBSET_MODEL) & (d.cell == ENFORCED_DC),
                           "episode_id"])
    ng_sub = ng[ng["episode_id"].isin(subset_ids)]
    gate_sub = rc_curve(ng_sub["confidence"], ng_sub["faithful"], ng_sub["covered"],
                        cell=cells["nonllm_gate"])

    rg = d[d.cell == "random_gate"]
    random_risk = float(coverage_sweep_curve(rg["faithful"],
                                             cell=cells["random_gate"])["risk"].iloc[0])

    # Enforced arms, each with the gate curve it is legitimately comparable to.
    arms, excursions = {}, {}
    for m in sorted(SHORT):
        a = d[(d.model == m) & (d.cell == ENFORCED_DC)]
        if a.empty:
            continue
        curve = rc_curve(a["confidence"], a["faithful"], a["covered"], cell=cells[ENFORCED_DC])
        ref = gate_sub if m == SUBSET_MODEL else gate_full
        arms[m] = curve
        # Evaluate at the REAL operating points of both curves, never on a
        # linspace. The gate's confidence values are heavily tied: it has an
        # operating point at coverage 0.015 and its next one is 0.327, so a
        # regular grid would score the arms across a 31-point gap where neither
        # curve was ever run and both are pure interpolation.
        top = min(curve["coverage"].max(), ref["coverage"].max())
        pts = np.unique(np.concatenate([curve["coverage"].to_numpy(),
                                        ref["coverage"].to_numpy()]))
        pts = pts[(pts >= MIN_COVERAGE_PLOTTED) & (pts <= top)]
        excursions[m] = float(min(risk_at_coverage(curve, x) - risk_at_coverage(ref, x)
                                  for x in pts))
    worst = min(excursions.values())

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    style(ax)

    g = gate_full[gate_full["coverage"] >= MIN_COVERAGE_PLOTTED]
    ax.fill_between(g["coverage"], 0, g["risk"], color=BLUE, alpha=0.055, zorder=1)
    for m, curve in arms.items():
        c = curve[curve["coverage"] >= MIN_COVERAGE_PLOTTED]
        ax.plot(c["coverage"], c["risk"], color=GREY, lw=1.1, alpha=0.9, zorder=3)
    gs = gate_sub[gate_sub["coverage"] >= MIN_COVERAGE_PLOTTED]
    ax.plot(gs["coverage"], gs["risk"], color=BLUE, lw=1.2, ls=(0, (4, 2)), zorder=4)
    ax.plot(g["coverage"], g["risk"], color=BLUE, lw=2.6, zorder=5)
    ax.axhline(random_risk, color=DARK, lw=0.9, ls=(0, (1, 2.5)), zorder=2)
    ax.text(0.335, random_risk + 0.006, f"Random gate, any coverage ({random_risk:.3f})",
            fontsize=8.5, color=DARK)

    ax.text(0.62, 0.050, "Below the deterministic gate", fontsize=9.5, color=BLUE,
            ha="center", style="italic", zorder=6)
    ax.text(0.62, 0.030, f"largest excursion into it by any arm: {abs(worst):.3f}",
            fontsize=8.5, color=MUTED, ha="center", zorder=6)

    # Advisory arms as fixed points, read from the analysis table so the figure
    # cannot disagree with Table 2.
    mc = pd.read_csv(TABLES / "primary_matched_coverage.csv")
    mc = mc[mc.uncertainty_source == "decoder_confidence"].sort_values("coverage")
    # Label placement is explicit rather than automatic: three of the five
    # models sit on top of each other near coverage 1.0, where an offset that
    # suits one label makes the other two unreadable.
    # DeepSeek and GPT-5.6 Luna both sit at coverage ~0.997 and risk ~0.301,
    # about 0.002 apart on each axis. Two separate labels there would collide
    # whatever the offsets, so they share one callout with one leader line.
    CLUSTERED = ("deepseek/deepseek-v4-flash", "openai/gpt-5.6-luna")
    offsets = {"anthropic/claude-sonnet-5": (-10, 10),
               "z-ai/glm-5.3-flash": (0, 13),
               "google/gemini-3.7-flash": (13, -15)}
    aligns = {"google/gemini-3.7-flash": "left", "z-ai/glm-5.3-flash": "center"}
    for _, r in mc.iterrows():
        ax.scatter(r["coverage"], r["risk_among_acted"], s=78,
                   marker=MARKERS.get(r["model"], "o"), color=SALMON,
                   edgecolors="white", linewidths=1.0, zorder=7)
        if r["model"] in CLUSTERED:
            continue
        dx, dy = offsets.get(r["model"], (-10, 10))
        ax.annotate(SHORT.get(r["model"], r["model"]),
                    (r["coverage"], r["risk_among_acted"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=aligns.get(r["model"], "right"), fontsize=8.5,
                    color=LABEL, zorder=8)
    cl = mc[mc.model.isin(CLUSTERED)]
    if len(cl) == 2:
        ax.annotate("DeepSeek v4 Flash\nGPT-5.6 Luna",
                    (float(cl["coverage"].mean()), float(cl["risk_among_acted"].mean())),
                    textcoords="offset points", xytext=(-24, 18), ha="right",
                    fontsize=8.5, color=LABEL, zorder=8,
                    arrowprops=dict(arrowstyle="-", color=ZERO, lw=0.7))

    # The decisive comparison, drawn: the subset model's advisory point against
    # the gate at the SAME coverage, on the SAME 500 episodes.
    son = mc[mc.model == SUBSET_MODEL]
    if not son.empty:
        r = son.iloc[0]
        ax.plot([r["coverage"], r["coverage"]],
                [r["nonllm_risk_at_matched_coverage"], r["risk_among_acted"]],
                color=LABEL, lw=1.0, ls=":", zorder=6)
        ax.annotate(f"gate {r['nonllm_risk_at_matched_coverage']:.3f}\n"
                    f"advisory arm {r['risk_among_acted']:.3f}",
                    (r["coverage"],
                     (r["risk_among_acted"] + r["nonllm_risk_at_matched_coverage"]) / 2),
                    textcoords="offset points", xytext=(-22, -34), ha="right",
                    fontsize=8.5, color=LABEL,
                    arrowprops=dict(arrowstyle="-", color=ZERO, lw=0.7))

    ax.set_xlabel("Coverage (fraction of episodes acted on)", fontsize=10.5)
    ax.set_ylabel("Unfaithful execution among actions taken", fontsize=10.5)
    # The gate has no operating point between coverage 0.015 and 0.327 (its
    # confidence values are tied), so an axis starting near zero would be
    # mostly empty and would imply operating points that do not exist.
    ax.set_xlim(0.32, 1.03)
    ax.set_ylim(0.0, 0.345)
    ax.tick_params(labelsize=9.5)

    handles = [
        mlines.Line2D([], [], color=BLUE, lw=2.6,
                      label="Deterministic gate, 1,065 episodes"),
        mlines.Line2D([], [], color=BLUE, lw=1.2, ls=(0, (4, 2)),
                      label="Deterministic gate, Claude Sonnet 5 subset (500)"),
        mlines.Line2D([], [], color=GREY, lw=1.1, label="Enforced LLM arms (swept)"),
        mlines.Line2D([], [], color=SALMON, marker="o", lw=0, markersize=7,
                      markeredgecolor="white", label="Advisory LLM arms (fixed points)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, fontsize=9, frameon=False, handletextpad=0.5, columnspacing=1.6)
    fig.text(0.5, -0.155,
             "Principal dataset: 1,065 episodes, 47 participants, decoding-error "
             "prevalence 0.341. Advisory arms carry reconstructed decoder confidence "
             "in the prompt.",
             ha="center", fontsize=8, color=MUTED, style="italic")
    save(fig, "figure1_risk_coverage_frontier")
    return {"worst_excursion": worst, "excursions": excursions,
            "random_gate_risk": random_risk}


# --------------------------------------------------------------------------
# Figure 2 -- matched-coverage comparison, PRINCIPAL data
# --------------------------------------------------------------------------
def fig2():
    """Every number here comes from `primary_matched_coverage.csv`."""
    mc = pd.read_csv(TABLES / "primary_matched_coverage.csv")
    groups = [("decoder_confidence", "Reconstructed decoder confidence in the prompt"),
              ("none", "No uncertainty in the prompt")]

    rows, ypos, headers = [], [], []
    y = 0.0
    for src, header in groups:
        sub = mc[mc.uncertainty_source == src].sort_values("coverage")
        headers.append((y + 0.85, header))
        for _, r in sub.iterrows():
            rows.append(r)
            ypos.append(y)
            y -= 1.0
        y -= 1.1
    rows = pd.DataFrame(rows).reset_index(drop=True)
    ypos = np.array(ypos)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.50})
    fig.subplots_adjust(left=0.30)

    # --- a: risk at the arm's own coverage, arm against gate ---------------
    ax = axes[0]
    style(ax)
    for yi, (_, r) in zip(ypos, rows.iterrows()):
        ax.plot([r["nonllm_risk_at_matched_coverage"], r["risk_among_acted"]],
                [yi, yi], color=GREY, lw=1.3, zorder=2)
    # Gate drawn larger and behind: in the five no-uncertainty rows the two
    # points differ by under 0.005 and a same-size marker would hide one of
    # them entirely, which would read as a single measurement.
    ax.scatter(rows["nonllm_risk_at_matched_coverage"], ypos, s=86, color=BLUE,
               edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter(rows["risk_among_acted"], ypos, s=46, color=SALMON, marker="D",
               edgecolors="white", linewidths=0.9, zorder=6)
    ax.set_xlim(0.08, 0.34)
    ax.set_xlabel("Unfaithful execution among actions taken", fontsize=10.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT.get(r["model"], r["model"]) for _, r in rows.iterrows()],
                       fontsize=9.5)
    ax.set_ylim(ypos.min() - 0.9, ypos.max() + 1.45)
    ax.tick_params(labelsize=9.5)
    for yh, header in headers:
        ax.text(-0.62, yh, header, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=10.5, fontweight="bold", color=BLUE)
    panel_label(ax, "a", x=-0.62, y=1.03)

    # Coverage column, in the gutter between the two panels.
    ax.text(1.16, ypos.max() + 0.75, "Coverage", transform=ax.get_yaxis_transform(),
            ha="center", va="center", fontsize=8.5, color=LABEL)
    for yi, (_, r) in zip(ypos, rows.iterrows()):
        ax.text(1.16, yi, f"{r['coverage']:.3f}", transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=8.5, color=MUTED)

    # --- b: matched gap with its participant-clustered 95% CI --------------
    ax = axes[1]
    style(ax)
    ax.axvline(0.0, color=ZERO, ls="--", lw=0.7, zorder=1)
    for yi, (_, r) in zip(ypos, rows.iterrows()):
        excl = r["matched_gap_lo"] > 0 or r["matched_gap_hi"] < 0
        col = BLUE if excl else GREY
        ax.plot([r["matched_gap_lo"], r["matched_gap_hi"]], [yi, yi],
                color=col, lw=1.6, zorder=4, solid_capstyle="butt")
        ax.scatter([r["matched_gap"]], [yi], s=52, color=col, marker="D",
                   edgecolors="white", linewidths=0.9, zorder=6)
    ax.set_xlim(-0.028, 0.20)
    ax.set_xlabel("Matched-coverage difference in unfaithful execution\n"
                  "(arm minus gate; positive favours the gate)", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.text(1.04, ypos.max() + 0.75, "Difference (95% CI)",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8.5, color=LABEL)
    for yi, (_, r) in zip(ypos, rows.iterrows()):
        ax.text(1.04, yi, f"{r['matched_gap']:.3f} ({r['matched_gap_lo']:.3f} to "
                          f"{r['matched_gap_hi']:.3f})",
                transform=ax.get_yaxis_transform(), va="center",
                fontsize=8.5, color=MUTED)
    panel_label(ax, "b", x=-0.12, y=1.03)

    handles = [
        mlines.Line2D([], [], color=BLUE, marker="o", lw=0, markersize=7,
                      markeredgecolor="white", label="Deterministic gate at matched coverage"),
        mlines.Line2D([], [], color=SALMON, marker="D", lw=0, markersize=7,
                      markeredgecolor="white", label="LLM advisory arm"),
        mlines.Line2D([], [], color=BLUE, lw=1.6,
                      label="95% CI excludes zero"),
        mlines.Line2D([], [], color=GREY, lw=1.6, label="95% CI includes zero"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, -0.13),
               ncol=2, fontsize=9, frameon=False, handletextpad=0.5, columnspacing=2.0)
    fig.text(0.55, -0.205,
             "Principal dataset: 1,065 episodes (500 for Claude Sonnet 5, its frozen "
             "subset). Intervals are 10,000 bootstrap replicates clustered on participant.",
             ha="center", fontsize=8, color=MUTED, style="italic")
    save(fig, "figure2_matched_coverage")
    return rows


# --------------------------------------------------------------------------
# Figure 3 -- caution-wording battery, PILOT (exploratory) data
# --------------------------------------------------------------------------
def fig3():
    """The 12-wording battery. PILOT data, and the panel says so.

    The battery was never run at full pool. All three panels are the
    error-conditional subset of the 100-episode exploratory dataset, whose 50:50
    stratification enriches decoding errors, except parse failure, which is
    measured on the full exploratory population because parse failure is a
    property of the model and not of the error subset.

    WHAT PANEL a PLOTS. `secondary_caution_battery.csv`'s column is named
    `unsafe`, but 10_secondary.py defines it as `covered & ~faithful` and
    averages it over EVERY row, so it is unfaithful execution as a fraction of
    all error-bearing episodes, not the risk among actions taken. The two are
    far apart here: wording 1 is 0.478 of all episodes and 0.842 among the
    actions it took. Panel a is labelled for the denominator it has, and panel
    b carries the coverage that produced it, because reading panel a alone
    would reproduce exactly the bare-rate error this paper exists to document.

    Drawn as aligned dot columns rather than as a scatter. Six of the 12
    wordings share the same value to the digit in panel a and four share theirs
    in panel c, so in a scatter they land on top of one another and half the
    battery becomes invisible.
    """
    b = pd.read_csv(TABLES / "secondary_caution_battery.csv")
    tests = pd.read_csv(TABLES / "secondary_caution_tests.csv")
    baselines = tests["unsafe_baseline"].unique()
    assert len(baselines) == 1, f"more than one no-caution baseline: {baselines}"
    base = float(baselines[0])
    # Sorted so the one wording that moved the endpoint is at the bottom, and
    # the coverage that bought the move is on the same row.
    b = b.sort_values("unsafe", ascending=False).reset_index(drop=True)
    # Which wordings actually differ from the no-caution baseline after BH
    # correction across the 12-wording family. Marked on the row label because
    # without it the panel reads as "nothing differed", and two of the twelve
    # did: `secondary_caution_tests.csv` puts wording 1 and wording 10 below
    # p_bh = .05, which `stats_digest.json` records as n_significant_after_bh.
    sig = set(tests.loc[tests["p_bh"] < 0.05, "wording_idx"].astype(int))
    y = -np.arange(len(b), dtype=float)
    over = (b["parse_failure"] > PARSE_FAIL_LIMIT).to_numpy()
    colors = np.where(over, SALMON, BLUE)

    w1 = b[b.wording_idx == 1].iloc[0]
    n_excluded_models = int(w1["n_models_excluded"])
    # Panel size comes from the digest the same tables were written with, so a
    # change to the panel cannot leave a stale "of 5" in the legend.
    n_models = int(json.loads((REPO_ROOT / "output" / "stats_digest.json")
                              .read_text())["n_models"])

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0],
                                          "wspace": 0.26})
    fig.subplots_adjust(left=0.36)

    # `val_side` and `ref_side` are +1 to put the label LEFT of the point or
    # line and -1 to put it right. They are set per panel rather than shared:
    # in panel c the values are all near zero, and a label to their left is
    # pushed off the axis and lands on the spine and the tick marks.
    specs = [
        ("unsafe", "Unfaithful execution\n(fraction of error-bearing episodes)",
         (0.40, 0.96), base, f"No-caution\nbaseline {base:.3f}", -1, +1),
        ("coverage", "Coverage\n(fraction of episodes acted on)",
         (0.42, 1.05), None, None, +1, +1),
        ("parse_failure", "Parse failure\n(no valid tool call emitted)",
         (-0.03, 0.56), PARSE_FAIL_LIMIT,
         f"Pre-specified\n{PARSE_FAIL_LIMIT:.0%} limit", -1, -1),
    ]
    for k, (col, xlabel, xlim, ref, ref_label, val_side, ref_side) in enumerate(specs):
        ax = axes[k]
        style(ax)
        span = xlim[1] - xlim[0]
        if ref is not None:
            ax.axvline(ref, color=ZERO, ls="--" if col == "unsafe" else ":",
                       lw=0.8, zorder=1)
            # Offset off the line: a centred label sits astride it and the rule
            # runs through the text.
            ax.text(ref - ref_side * 0.020 * span, 0.72, ref_label,
                    ha="right" if ref_side > 0 else "left", va="bottom",
                    fontsize=8.5, color=MUTED)
        ax.scatter(b[col], y, s=62, c=colors, edgecolors="white", linewidths=0.9,
                   zorder=5)
        for yi, v in zip(y, b[col]):
            # 0.055 of the span clears the 62-point marker; 0.026 put the
            # digits under it and 0.040 left them touching its edge.
            # White bbox because a reference line can fall inside a label:
            # wording 10 sits at 0.784 and its label runs across the 0.837
            # baseline rule.
            ax.text(v - val_side * 0.055 * span, yi, f"{v:.3f}",
                    ha="right" if val_side > 0 else "left", va="center", fontsize=8,
                    color=MUTED, zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
        ax.tick_params(labelsize=9)
        panel_label(ax, "abc"[k], x=-0.72 if k == 0 else -0.10, y=1.02)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [f"w{int(r.wording_idx)}  “{r.wording_text}”"
         + ("  *" if int(r.wording_idx) in sig else "")
         for r in b.itertuples()], fontsize=8.5)
    axes[0].set_ylim(y.min() - 0.8, y.max() + 1.9)

    handles = [
        mlines.Line2D([], [], color=BLUE, marker="o", lw=0, markersize=7,
                      markeredgecolor="white",
                      label=f"Within the pre-specified {PARSE_FAIL_LIMIT:.0%} "
                            "parse-failure limit"),
        mlines.Line2D([], [], color=SALMON, marker="o", lw=0, markersize=7,
                      markeredgecolor="white",
                      label=f"Exceeds the limit; labelled in {n_excluded_models} of "
                            f"{n_models} models"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.66, -0.10),
               ncol=2, fontsize=9, frameon=False, handletextpad=0.5, columnspacing=2.4)
    # Wrapped by hand: bbox_inches="tight" grows the whole figure to fit a
    # single long line, and a 14-inch figure is not a journal figure.
    fig.text(0.66, -0.175,
             f"* Differs from the no-caution baseline after Benjamini-Hochberg "
             f"correction across the 12-wording family ({len(sig)} of {len(b)} "
             "wordings).\nEXPLORATORY 100-episode dataset, stratified 50:50 on "
             "decoding error; rates are enriched by construction and are not "
             "deployment estimates.",
             ha="center", va="top", fontsize=8, color=MUTED, style="italic",
             linespacing=1.5)
    save(fig, "figure3_caution_battery")
    return {"baseline": base, "w1_parse_failure": float(w1["parse_failure"]),
            "n_models_excluded": n_excluded_models}


# --------------------------------------------------------------------------
# Figure 4 -- calibration of the reconstructed confidence
# --------------------------------------------------------------------------
def fig4():
    """Neither panel depends on a model run: both describe the EEG-derived
    confidence every arm and every gate in this study thresholds."""
    rel = pd.read_csv(TABLES / "calibration_reliability.csv")
    pooled = rel[rel["study"] == "overall"]
    rel = rel[rel["study"] != "overall"].reset_index(drop=True)
    trans = pd.read_csv(TABLES / "calibration_transport.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9),
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    style(ax)
    x = np.arange(len(rel))
    ax.bar(x - 0.19, rel["ece"], width=0.36, color=BLUE, label="Expected calibration error",
           zorder=3)
    ax.bar(x + 0.19, rel["brier"], width=0.36, color=SALMON, label="Brier score", zorder=3)
    for xi, (ece, br, n) in enumerate(zip(rel["ece"], rel["brier"], rel["n"])):
        ax.text(xi, max(ece, br) + 0.010, f"n = {int(n):,}", ha="center",
                fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(rel["study"], fontsize=9.5)
    ax.set_ylabel("Calibration error", fontsize=10.5)
    ax.set_ylim(0, max(rel["brier"].max(), rel["ece"].max()) * 1.34)
    ax.tick_params(labelsize=9.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left", handlelength=1.2)
    panel_label(ax, "a", x=-0.16, y=1.05)

    ax = axes[1]
    studies = sorted(set(trans["train_study"]) | set(trans["test_study"]))
    M = np.full((len(studies), len(studies)), np.nan)
    for _, r in trans.iterrows():
        M[studies.index(r["train_study"]), studies.index(r["test_study"])] = r["ece"]
    vmax = float(np.nanmax(M))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(studies)))
    ax.set_xticklabels(studies, fontsize=9)
    ax.set_yticks(range(len(studies)))
    ax.set_yticklabels(studies, fontsize=9)
    ax.set_xlabel("Applied to", fontsize=10.5)
    ax.set_ylabel("Calibrated on", fontsize=10.5)
    for i in range(len(studies)):
        for j in range(len(studies)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                        color="white" if M[i, j] > vmax * 0.55 else LABEL)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Expected calibration error", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    panel_label(ax, "b", x=-0.24, y=1.05)

    pooled_note = (f"Pooled across the four studies, ECE is "
                   f"{float(pooled['ece'].iloc[0]):.3f}; it is not the primary "
                   "statistic, because bins with opposite-signed deviations in "
                   "different studies cancel when pooled. "
                   if not pooled.empty else "")
    fig.text(0.5, -0.10,
             "Reconstructed calibrated confidence from held-out calibration EEG, "
             f"{int(rel['n'].sum()):,} selections across four studies. {pooled_note}"
             "The diagonal of b is empty: a calibrator is never transported to the "
             "study it was fitted on.",
             ha="center", fontsize=8, color=MUTED, style="italic", wrap=True)
    save(fig, "figure4_calibration")
    return {"n_studies": len(studies), "worst_transport": vmax}


# --------------------------------------------------------------------------
# Figure 5 -- the naturalistic benchmark
# --------------------------------------------------------------------------
NATURAL_ARMS = [
    (ADVISORY_NONE, "No uncertainty, advisory", BLUE, "o", 0.24),
    (ADVISORY_DC, "Decoder confidence, advisory", SALMON, "D", 0.0),
    (ENFORCED_DC, "Decoder confidence, enforced", GREY, "s", -0.24),
]
COMPARATORS = {
    "natural_confidence_gate_lexical": "Lexical resolver + gate",
    "natural_confidence_gate_canonical": "Exact matcher + gate",
}


def fig5():
    """Five models x three arms against the deterministic comparators.

    Absolute rates here are BENCHMARK risks at the source pool's observed
    decoder-error prevalence (0.340), never deployment estimates: the error
    distribution is empirical but the nine commands are constructed and no
    participant ever sent them.
    """
    files = [f for f in sorted(NATURAL_RUNS.glob("*.parquet"))
             if not f.name.startswith("._")]
    llm, comp, other = [], {}, {}
    for f in files:
        t = pd.read_parquet(f)
        t = t[t["error"].isna()]
        cov = t["covered"].fillna(False).astype(bool)
        faith = t["faithful"].fillna(False).astype(bool)
        rec = {
            "n": len(t),
            "coverage": float(cov.mean()),
            "risk": float((~faith[cov]).mean()) if cov.any() else np.nan,
            "n_unfaithful": int((cov & ~faith).sum()),
            "parse_failure": float(t["parse_failed"].fillna(False).astype(bool).mean())
            if "parse_failed" in t.columns else np.nan,
            "err_prevalence": float(t["is_error_bearing"].astype(bool).mean()),
        }
        if "model" in t.columns and t["model"].notna().any():
            rec.update(model=t["model"].iloc[0], cell=t["cell"].iloc[0])
            llm.append(rec)
        else:
            # Every model-free unit is recorded, not only the two the panel
            # draws a reference line for. `natural_random_gate_canonical` is a
            # third free comparator that this figure does not plot, and a
            # source table that silently omits a unit present in the run
            # directory is the kind of gap nothing downstream can detect.
            other[f.stem] = rec
            if f.stem in COMPARATORS:
                comp[f.stem] = rec
    llm = pd.DataFrame(llm)

    src = llm.copy()
    src["arm"] = src["cell"]
    for name, rec in other.items():
        src = pd.concat([src, pd.DataFrame([{**rec, "model": "__none__", "cell": name,
                                             "arm": name}])], ignore_index=True)
    src.to_csv(TABLES / "figure5_naturalistic.csv", index=False)
    assert len(src) == len(files), (
        f"figure5_naturalistic.csv has {len(src)} rows for {len(files)} run units"
    )

    models = sorted(SHORT)
    ybase = {m: -i for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.2), sharey=True,
                             gridspec_kw={"wspace": 0.16})
    fig.subplots_adjust(left=0.20)

    specs = [
        ("coverage", "Coverage\n(fraction of episodes acted on)", (0.55, 1.02), True),
        ("risk", "Unfaithful execution\namong actions taken", (-0.0018, 0.028), True),
        ("parse_failure", "Parse failure\n(no valid tool call emitted)", (-0.012, 0.20),
         False),
    ]
    for k, (col, xlabel, xlim, show_ref) in enumerate(specs):
        ax = axes[k]
        style(ax)
        if show_ref:
            for name, label in COMPARATORS.items():
                v = comp[name][col]
                ls = (0, (5, 2)) if "lexical" in name else (0, (1, 2.5))
                ax.axvline(v, color=BLUE if "lexical" in name else DARK,
                           lw=1.1, ls=ls, zorder=2)
        for cell, _, colr, mk, dy in NATURAL_ARMS:
            sub = llm[llm.cell == cell]
            ax.scatter(sub[col], [ybase[m] + dy for m in sub["model"]], s=54,
                       color=colr, marker=mk,
                       edgecolors="white" if colr != GREY else MUTED,
                       linewidths=0.9 if colr != GREY else 0.7, zorder=6)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
        ax.tick_params(labelsize=9)
        panel_label(ax, "abc"[k], x=-0.30 if k == 0 else -0.10, y=1.05)
    axes[0].set_yticks([ybase[m] for m in models])
    axes[0].set_yticklabels([SHORT[m] for m in models], fontsize=9.5)
    axes[0].set_ylim(min(ybase.values()) - 0.7, max(ybase.values()) + 0.7)

    # The comparator values live in the legend, not next to their lines: the
    # two lines in panel a are 0.28 apart on an axis 0.47 wide, and two labels
    # anchored to them overlapped in the middle whatever the alignment.
    axes[1].text(0.0006, max(ybase.values()) + 0.80,
                 "Both comparators 0.000", ha="left", va="bottom",
                 fontsize=8, color=BLUE)
    # The comparators never emit a tool call, so parse failure is not a
    # quantity they have. Saying so is better than an absent reference line a
    # reader could read as zero.
    axes[2].text(0.0, max(ybase.values()) + 0.80,
                 "Not defined for the comparators", ha="left", va="bottom",
                 fontsize=8, color=MUTED)

    handles = [mlines.Line2D([], [], color=c, marker=mk, lw=0, markersize=7,
                             markeredgecolor="white" if c != GREY else MUTED,
                             label=lab)
               for _, lab, c, mk, _ in NATURAL_ARMS]
    handles += [
        mlines.Line2D([], [], color=BLUE, lw=1.1, ls=(0, (5, 2)),
                      label=f"Lexical resolver + gate, no model "
                            f"(coverage {comp['natural_confidence_gate_lexical']['coverage']:.3f})"),
        mlines.Line2D([], [], color=DARK, lw=1.1, ls=(0, (1, 2.5)),
                      label=f"Exact matcher + gate, no model "
                            f"(coverage {comp['natural_confidence_gate_canonical']['coverage']:.3f})"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, -0.24),
               ncol=2, fontsize=9, frameon=False, handletextpad=0.5, columnspacing=2.2)
    fig.text(0.56, -0.335,
             "Naturalistic benchmark: 200 constructed episodes, real donor confidences "
             "and error patterns, decoder-error prevalence 0.340. Absolute risks are "
             "benchmark rates at that prevalence, not deployment estimates.",
             ha="center", fontsize=8, color=MUTED, style="italic")
    save(fig, "figure5_naturalistic_benchmark")
    return {"llm": llm, "comparators": comp}


def main() -> int:
    d = load_principal()
    cells = {c.name: c for c in enumerate_cells()}
    print(f"principal: {d['episode_id'].nunique()} episodes, "
          f"prevalence {d.drop_duplicates('episode_id')['err'].astype(bool).mean():.4f}")
    print("building figures...")
    f1 = fig1(d, cells)
    fig2()
    f3 = fig3()
    fig4()
    f5 = fig5()

    checks = {
        "figure1_worst_excursion_below_gate": f1["worst_excursion"],
        "figure1_random_gate_risk": f1["random_gate_risk"],
        "figure3_no_caution_baseline": f3["baseline"],
        "figure5_lexical_coverage": f5["comparators"]["natural_confidence_gate_lexical"]["coverage"],
        "figure5_lexical_risk": f5["comparators"]["natural_confidence_gate_lexical"]["risk"],
        "figure5_llm_mean_coverage": float(f5["llm"]["coverage"].mean()),
        "figure5_llm_total_unfaithful": int(f5["llm"]["n_unfaithful"].sum()),
    }
    (FIGS / "figure_checks.json").write_text(json.dumps(checks, indent=2))
    print("\nchecks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\n-> {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
