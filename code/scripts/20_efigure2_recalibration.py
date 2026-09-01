"""eFigure 2 -- recalibration sensitivity (Task 19).

WHAT THIS ASKS. The confidence rendered into the advisory arm's prompt was a
product of three per-selection calibrated probabilities, and that product is
empirically miscalibrated at episode level (eTable S4c). A threshold gate is
invariant to any strictly monotone recalibration; an advisory arm is not,
because the numeric value reaches the model. Task 19 re-ran the identical arm
with the participant-grouped out-of-fold isotonic score substituted for the
product, so the two sides of the headline comparison receive equally
well-calibrated signal.

THE PAIRING RULE, which is the whole reason this file exists rather than a
two-line plot of two tables. `anthropic/claude-sonnet-5` ran a frozen
500-episode subset and the other four models ran all 1,065, so a statistic
pooled across the panel would combine unequal samples. Every quantity below is
computed PER MODEL on the INTERSECTION of the episode identifiers the two runs
share, and the intersection size is asserted and printed. Comparing across
different episode sets is the exact defect that reversed a conclusion in the
AURC table (`primary_aurc.csv`), so it is checked here rather than assumed.

  runs_principal/ cell factorial:decoder_confidence:advisory:s0  -> product score
  runs_recal/     same cell                                      -> recalibrated

NO NUMBER IS RECOMPUTED FOR PUBLICATION. Panels a and b are computed from the
episode runs and then asserted equal to the published
`followup_task19_recalibration.csv` to 4 decimal places; the script aborts on
any disagreement rather than drawing a figure that disagrees with the
supplement. Panel c is READ from `followup_task19_matched_coverage.csv` and
never recomputed, because its intervals are 10,000 participant-clustered
bootstrap replicates that are already frozen in eTable S7b.

Writes: output/figures/efigure2_recalibration_sensitivity.pdf / .png
        output/tables/efigure2_recalibration_paired.csv

Run: PYTHONPATH=code python3 code/scripts/20_efigure2_recalibration.py
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

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = REPO_ROOT / "output" / "intermediate"
PRINCIPAL_RUNS = INTERMEDIATE / "runs_principal"
RECAL_RUNS = INTERMEDIATE / "runs_recal"
TABLES = REPO_ROOT / "output" / "tables"
FIGS = REPO_ROOT / "output" / "figures"

CELL = "factorial:decoder_confidence:advisory:s0"

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

SHORT = {
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "deepseek/deepseek-v4-flash": "DeepSeek v4 Flash",
    "google/gemini-3.7-flash": "Gemini 3.7 Flash",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "z-ai/glm-5.3-flash": "GLM 5.3 Flash",
}


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(ZERO)
        ax.spines[s].set_linewidth(0.8)


def panel_label(ax, letter, x=-0.14, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=LABEL)


def _read_runs(path: Path, cell: str) -> pd.DataFrame:
    # "._*" are macOS AppleDouble sidecars: the volume is exFAT, so they land
    # as real files matching *.parquet.
    files = [f for f in sorted(path.glob("*.parquet")) if not f.name.startswith("._")]
    if not files:
        raise SystemExit(f"no parquet checkpoints in {path}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[d["error"].isna()]
    return d[d["cell"] == cell].copy()


def paired_table() -> pd.DataFrame:
    """Per model, on the episodes the two runs share and on no others."""
    product = _read_runs(PRINCIPAL_RUNS, CELL)
    recal = _read_runs(RECAL_RUNS, CELL)

    rows = []
    for model in sorted(SHORT):
        a = product[product["model"] == model]
        b = recal[recal["model"] == model]
        if a.empty or b.empty:
            raise SystemExit(f"{model} missing from one of the two runs")
        shared = set(a["episode_id"]) & set(b["episode_id"])
        # An unpaired episode on either side means the two scores were measured
        # on different samples for this model, and every difference below would
        # then confound the score with the sample.
        only_a, only_b = len(set(a["episode_id"]) - shared), len(set(b["episode_id"]) - shared)
        print(f"  {SHORT[model]:<18s} product {len(a):>5d}  recalibrated {len(b):>5d}  "
              f"shared {len(shared):>5d}  unpaired {only_a}/{only_b}")
        if not shared:
            raise SystemExit(f"{model}: the two runs share no episodes")
        for score, t in (("product", a), ("recalibrated", b)):
            s = t[t["episode_id"].isin(shared)]
            cov = s["covered"].fillna(False).astype(bool)
            faith = s["faithful"].fillna(False).astype(bool)
            pf = s["parse_failed"].fillna(False).astype(bool)
            rows.append({
                "score": score,
                "model": model,
                "n_shared_episodes": len(shared),
                "n_participants": int(s["participant_id"].nunique()),
                "coverage": float(cov.mean()),
                "unfaithful_of_all": float((cov & ~faith).mean()),
                "risk_among_acted": float((~faith[cov]).mean()) if cov.any() else np.nan,
                "parse_failure": float(pf.mean()),
                "n_executed": int(cov.sum()),
                "n_unfaithful": int((cov & ~faith).sum()),
            })
    return pd.DataFrame(rows)


def assert_matches_published(paired: pd.DataFrame) -> None:
    """The figure may not disagree with eTable S7a. Abort if it does."""
    pub = pd.read_csv(TABLES / "followup_task19_recalibration.csv")
    checks = [("coverage", "coverage"),
              ("unfaithful_of_all", "unfaithful of all"),
              ("parse_failure", "parse failure"),
              ("n_executed", "executed"),
              ("n_unfaithful", "unfaithful executions")]
    bad = []
    for _, r in paired.iterrows():
        p = pub[(pub["score"] == r["score"]) & (pub["model"] == r["model"])]
        if len(p) != 1:
            bad.append(f"{r['score']}/{r['model']}: {len(p)} published rows")
            continue
        p = p.iloc[0]
        for mine, theirs in checks:
            if abs(float(r[mine]) - float(p[theirs])) > 5e-5:
                bad.append(f"{r['score']}/{r['model']} {mine}: "
                           f"recomputed {r[mine]} vs published {p[theirs]}")
    if bad:
        raise SystemExit("recomputed values disagree with "
                         "followup_task19_recalibration.csv:\n  " + "\n  ".join(bad))
    print(f"  every value matches followup_task19_recalibration.csv "
          f"({len(paired)} rows x {len(checks)} columns)")


def efig4(paired: pd.DataFrame, mc: pd.DataFrame):
    models = sorted(SHORT)
    ybase = {m: -i for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.3), sharey=True,
                             gridspec_kw={"wspace": 0.20})
    fig.subplots_adjust(left=0.20)

    specs = [
        ("coverage", "Coverage\n(fraction of episodes acted on)", (0.50, 1.06)),
        ("unfaithful_of_all", "Unfaithful execution\n(fraction of all episodes)",
         (0.06, 0.34)),
    ]
    for k, (col, xlabel, xlim) in enumerate(specs):
        ax = axes[k]
        style(ax)
        for m in models:
            y = ybase[m]
            v0 = float(paired.loc[(paired.model == m) & (paired.score == "product"), col].iloc[0])
            v1 = float(paired.loc[(paired.model == m) & (paired.score == "recalibrated"), col].iloc[0])
            # An arrow rather than a plain segment: the direction IS the
            # finding, and both panels point the same way in the three models
            # that respond at all.
            if abs(v1 - v0) > 0.004:
                ax.annotate("", xy=(v1, y), xytext=(v0, y),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1,
                                            shrinkA=5.5, shrinkB=5.5,
                                            mutation_scale=11), zorder=3)
            else:
                ax.plot([v0, v1], [y, y], color=GREY, lw=1.2, zorder=3)
            # Product drawn larger and behind. In DeepSeek and GPT-5.6 Luna the
            # two values differ by under 0.002, and equal-sized markers hide one
            # of them completely, which reads as a single measurement rather
            # than as two that did not move.
            ax.scatter([v0], [y], s=104, color=GREY, marker="o",
                       edgecolors="white", linewidths=0.9, zorder=5)
            ax.scatter([v1], [y], s=46, color=SALMON, marker="D",
                       edgecolors="white", linewidths=0.9, zorder=6)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
        ax.tick_params(labelsize=9)
        panel_label(ax, "abc"[k], x=-0.32 if k == 0 else -0.10)

    # --- c: the matched gap against the gate, under each score --------------
    # Read, never recomputed: these intervals are 10,000 participant-clustered
    # bootstrap replicates already frozen in eTable S7b.
    ax = axes[2]
    style(ax)
    ax.axvline(0.0, color=ZERO, ls="--", lw=0.7, zorder=1)
    # Colour means SCORE here, exactly as in a and b. It deliberately does not
    # encode whether the interval excludes zero, for two reasons: a reader who
    # has just learned grey/salmon in the first two panels would have to unlearn
    # it, and three of the five recalibrated intervals have a lower bound of
    # 0.000 or 0.001, so a significance colour would draw a hard line through a
    # set of knife-edge cases. The dashed rule at zero is in the panel and the
    # interval bounds are tabulated in eTable S7b.
    for m in models:
        for score, dy, mk, col in (("product", 0.19, "o", GREY),
                                   ("recalibrated", -0.19, "D", SALMON)):
            r = mc[(mc["confidence score"] == score) & (mc["model"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            lo, hi, est = float(r["gap 95% CI low"]), float(r["gap 95% CI high"]), float(r["matched gap"])
            y = ybase[m] + dy
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, zorder=4, solid_capstyle="butt")
            ax.scatter([est], [y], s=50, color=col, marker=mk,
                       edgecolors="white", linewidths=0.8, zorder=6)
    ax.set_xlim(-0.012, 0.165)
    ax.set_xlabel("Matched-coverage difference\n(arm minus gate; positive favours the gate)",
                  fontsize=10, labelpad=6)
    ax.tick_params(labelsize=9)
    panel_label(ax, "c", x=-0.10)

    axes[0].set_yticks([ybase[m] for m in models])
    axes[0].set_yticklabels([SHORT[m] for m in models], fontsize=9.5)
    axes[0].set_ylim(min(ybase.values()) - 0.7, max(ybase.values()) + 0.7)

    handles = [
        mlines.Line2D([], [], color=GREY, marker="o", lw=0, markersize=7,
                      markeredgecolor="white",
                      label="Product confidence (the main study's signal)"),
        mlines.Line2D([], [], color=SALMON, marker="D", lw=0, markersize=7,
                      markeredgecolor="white",
                      label="Recalibrated confidence (out-of-fold isotonic)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.58, -0.17),
               ncol=2, fontsize=9, frameon=False, handletextpad=0.5, columnspacing=2.0)
    n_sonnet = int(paired.loc[paired.model == "anthropic/claude-sonnet-5",
                              "n_shared_episodes"].iloc[0])
    n_other = int(paired.loc[paired.model == "z-ai/glm-5.3-flash",
                             "n_shared_episodes"].iloc[0])
    fig.text(0.58, -0.245,
             f"Both scores were run on identical episode sets, and every comparison is "
             f"restricted to the episodes the two runs share: {n_sonnet:,} for Claude "
             f"Sonnet 5, its frozen subset, and {n_other:,} for the other four models.\n"
             "Bars in c are 95% CIs from 10,000 participant-clustered bootstrap "
             "replicates, and the gate is recomputed on the same score the arm was shown.",
             ha="center", va="top", fontsize=8, color=MUTED, style="italic",
             linespacing=1.5)

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"efigure2_recalibration_sensitivity.{ext}", dpi=600,
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  wrote efigure2_recalibration_sensitivity.pdf / .png")


def main() -> int:
    print("pairing the two runs on shared episodes:")
    paired = paired_table()
    print("checking against the published table:")
    assert_matches_published(paired)
    mc = pd.read_csv(TABLES / "followup_task19_matched_coverage.csv")
    print("drawing:")
    efig4(paired, mc)
    paired.to_csv(TABLES / "efigure2_recalibration_paired.csv", index=False)
    print(f"  wrote {TABLES / 'efigure2_recalibration_paired.csv'}")

    print("\nwhat the panels show, per model (product -> recalibrated):")
    for m in sorted(SHORT):
        p = paired[(paired.model == m) & (paired.score == "product")].iloc[0]
        r = paired[(paired.model == m) & (paired.score == "recalibrated")].iloc[0]
        print(f"  {SHORT[m]:<18s} coverage {p['coverage']:.3f} -> {r['coverage']:.3f}   "
              f"unfaithful {p['unfaithful_of_all']:.3f} -> {r['unfaithful_of_all']:.3f}   "
              f"(n={p['n_shared_episodes']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
