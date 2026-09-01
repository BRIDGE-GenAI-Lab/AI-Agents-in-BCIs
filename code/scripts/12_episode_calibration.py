"""Episode-level calibration validation (review item C).

The gate thresholds `episode_confidence` -- the PRODUCT of three calibrated
per-selection probabilities. That product equals the probability that all
three characters transmit correctly only under conditional independence,
which is not established here (see `nag.episode_calibration` docstring).
This script validates the product directly against episode-level
correctness, and against three alternative combination rules that make
different independence assumptions, plus a fifth model fit directly at the
episode level rather than assembled from selection-level pieces.

The three per-selection probabilities that feed every rule here are
OUT-OF-FOLD (participant-separated): `build_episode_pool` embeds the
IN-SAMPLE calibrator into its `confidence` column, which is correct for the
number the gate actually deploys but optimistic for reporting reliability
(see `per_selection_oof` below). This script reproduces `build_episode_pool`'s
join to recover the three per-selection scores per episode, but threads
`fit.oof` through it instead.

Writes output/tables/episode_calibration.csv (one row per rule: product,
min, mean, logsum, isotonic_episode) and
output/figures/efigure1_episode_reliability.pdf/.png (one reliability panel
per rule).

Run: PYTHONPATH=code uv run python3 code/scripts/12_episode_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from nag.confidence import fit_calibrator  # noqa: E402
from nag.design import DEFAULT_ONLINE_CSV, DEFAULT_SCORES_PATH, build_episode_pool  # noqa: E402
from nag.episode_calibration import RULES, combine, episode_reliability  # noqa: E402
from nag.episodes import ALS_STUDIES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES = REPO_ROOT / "output" / "tables"
FIGS = REPO_ROOT / "output" / "figures"

EPISODE_LENGTH = 3  # matches nag.design.build_episode_pool's default

BLUE, GREY, SALMON = "#00468B", "#ADB6B6", "#FDAF91"
LABEL, MUTED = "#2B2B2B", "#6B7280"
GRID, ZERO = "#F3F4F6", "#9CA3AF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

LABELS = {
    "product": "Product",
    "min": "Min",
    "mean": "Mean",
    "logsum": "Log-sum",
    "isotonic_episode": "Direct isotonic (episode-level)",
}
ORDER = ["product", "min", "mean", "logsum", "isotonic_episode"]


def per_selection_oof(
    online_csv: Path = DEFAULT_ONLINE_CSV,
    scores_path: Path = DEFAULT_SCORES_PATH,
    length: int = EPISODE_LENGTH,
) -> pd.DataFrame:
    """Reproduce `nag.design.build_episode_pool`'s episode/selection join,
    but keep the three per-selection OUT-OF-FOLD calibrated probabilities
    per episode rather than collapsing them into a single product column.

    `build_episode_pool` calibrates every selection with `fit.calibrator`,
    the isotonic model fitted on ALL valid ALS selections -- the right
    object to deploy, but optimistic for reporting reliability, since it has
    seen every participant. This function threads `fit.oof` -- the
    participant-separated out-of-fold prediction -- through the same
    filtering, sorting, and chunking logic instead, so no episode's
    reliability number is evaluated against a calibrator that trained on
    that episode's own participant.

    Returns one row per usable episode: `episode_id` and `per_selection`, a
    length-`length` list of out-of-fold calibrated probabilities in trial
    order.
    """
    online = pd.read_csv(online_csv)
    d = online[online["eligible"] == True].copy()  # noqa: E712 - explicit for object dtype
    d = d.dropna(subset=["target", "selected"])
    d = d[d["study"].isin(ALS_STUDIES)]
    d = d.sort_values(["relative_path", "trial_number"])

    ep_trial_keys: dict[str, list[tuple]] = {}
    for path, g in d.groupby("relative_path", sort=False):
        n_full = len(g) // length
        for k in range(n_full):
            chunk = g.iloc[k * length:(k + 1) * length]
            eid = f"{path}#{k:04d}"
            ep_trial_keys[eid] = list(zip(chunk["relative_path"], chunk["trial_number"]))

    scores = pd.read_parquet(scores_path)
    valid = scores[scores["score_top"].notna() & scores["correct"].notna()].copy()
    valid["correct_int"] = valid["correct"].astype(bool).astype(int)
    valid = valid.reset_index(drop=True)

    fit = fit_calibrator(
        valid["score_top"].to_numpy(), valid["correct_int"].to_numpy(), valid["participant_id"].to_numpy()
    )
    if fit.oof is None:
        raise RuntimeError("per_selection_oof: fewer than 2 participant groups; no out-of-fold predictions")
    valid["oof"] = fit.oof
    oof_lookup = valid.set_index(["relative_path", "trial_number"])["oof"]

    rows = []
    for eid, keys in ep_trial_keys.items():
        if len(keys) != length or any(k not in oof_lookup.index for k in keys):
            continue
        rows.append({"episode_id": eid, "per_selection": [float(oof_lookup.loc[k]) for k in keys]})
    return pd.DataFrame(rows, columns=["episode_id", "per_selection"])


def isotonic_episode_oof(product_score: np.ndarray, correct: np.ndarray, groups: np.ndarray,
                          n_splits: int = 5) -> np.ndarray:
    """A directly-fitted episode-level isotonic model: product score ->
    episode correctness, evaluated out-of-fold over GroupKFold(participant).

    This is the fifth row: rather than assembling an episode score from
    selection-level pieces under an independence assumption, it fits the
    mapping from the product score straight to the episode-level event, so
    it can absorb whatever positive dependence the product rule misses.
    """
    x = np.asarray(product_score, dtype=float)
    y = np.asarray(correct, dtype=int)
    g = np.asarray(groups)
    n = min(n_splits, len(np.unique(g)))
    if n < 2:
        raise RuntimeError("isotonic_episode_oof: fewer than 2 participant groups")
    oof = np.zeros_like(x)
    for tr, te in GroupKFold(n_splits=n).split(x, y, g):
        f = IsotonicRegression(out_of_bounds="clip").fit(x[tr], y[tr])
        oof[te] = np.clip(f.predict(x[te]), 1e-6, 1 - 1e-6)
    return oof


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(ZERO)
        ax.spines[s].set_linewidth(0.8)


def panel(ax, out: dict, rule: str, letter: str, is_probability: bool):
    """One reliability panel. `logsum` is not on a probability scale (it is
    a summed log-probability), so its panel reports the valid rank
    statistics (AUROC/AUPRC) without drawing a diagonal calibration curve
    that a clipped log score would make meaningless."""
    if is_probability:
        ax.plot([0, 1], [0, 1], color=ZERO, ls="--", lw=0.8, zorder=1)
        xs = [b["mean_conf"] for b in out["bins"] if b["n"] > 0]
        ys = [b["observed"] for b in out["bins"] if b["n"] > 0]
        ns = [b["n"] for b in out["bins"] if b["n"] > 0]
        ax.scatter(xs, ys, s=[20 + 3 * n for n in ns], color=BLUE, edgecolors="white",
                   linewidths=0.7, zorder=5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean episode score", fontsize=9)
        ax.text(0.03, 0.90, f"ECE {out['ece']:.3f}\nBrier {out['brier']:.3f}", fontsize=7.5,
                color=MUTED, transform=ax.transAxes, va="top")
    else:
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.6, zorder=1)  # keep axes comparable, no diagonal claim
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "not a probability scale\n(rank statistics only)", fontsize=8.5,
                color=MUTED, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.03, 1.14, f"{LABELS[rule]}", fontsize=9.5, color=LABEL, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.03, 1.03, f"AUROC {out['auroc']:.3f}   AUPRC {out['auprc']:.3f}   n={out['n']}",
            fontsize=7.5, color=MUTED, transform=ax.transAxes)
    ax.text(-0.18, 1.24, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=LABEL)


def main() -> int:
    pool = build_episode_pool()
    sel = per_selection_oof()
    d = pool.merge(sel, on="episode_id", how="inner")
    n_missing = len(pool) - len(d)
    if n_missing:
        print(f"warning: {n_missing} pool episode(s) had no out-of-fold per-selection score; dropped")

    per_selection = np.array(d["per_selection"].tolist())
    correct = (d["n_errors"] == 0).astype(int).to_numpy()
    participant = d["participant_id"].to_numpy()

    scores = {rule: combine(per_selection, rule) for rule in RULES}
    scores["isotonic_episode"] = isotonic_episode_oof(scores["product"], correct, participant)

    reliab, rows = {}, []
    for rule in ORDER:
        out = episode_reliability(scores[rule], correct)
        out["n"] = int(len(scores[rule]))
        note = ""
        if rule == "logsum":
            n_unique_product = int(np.unique(scores["product"]).size)
            note = ("logsum is a summed log-probability, not a probability on [0, 1]; ece and brier "
                     "are not meaningful on that scale and are reported as NaN rather than computed on "
                     "a clipped log score. auroc and auprc are rank-based and mathematically monotonic "
                     "with the product rule (see test_logsum_is_monotone_with_product), so they match to "
                     f"within floating-point tie-breaking: many episodes share identical per-selection "
                     f"score triples ({n_unique_product} unique product values over {len(scores['product'])} "
                     "episodes here), and log(a)+log(b)+log(c) vs a*b*c can break an exact tie in either "
                     "direction without any real change in ranking.")
            out["ece"] = float("nan")
            out["brier"] = float("nan")
        reliab[rule] = out
        rows.append({"rule": rule, "ece": out["ece"], "brier": out["brier"], "auroc": out["auroc"],
                      "auprc": out["auprc"], "n": out["n"], "note": note})

    TABLES.mkdir(parents=True, exist_ok=True)

    # PERSIST THE PER-EPISODE SCORES, not only the summary.
    #
    # Tasks 19 and 20 both gate on the recalibrated episode confidence, and an
    # earlier version of this script computed it, reported five summary rows,
    # and threw the per-episode values away. A downstream task would then have
    # had to refit the calibrator itself, and any difference in how it split
    # folds would have silently changed the number the gate thresholds. This
    # project has already been bitten once by an out-of-fold prediction that
    # was computed and then discarded, so the vector is written to disk here
    # and every consumer reads the same one.
    #
    # `isotonic_episode` is the participant-grouped OUT-OF-FOLD prediction. It
    # is the only column that may be used to report a reliability figure or to
    # gate an arm; an in-fold fit would be optimistic in a way no downstream
    # check would catch.
    per_ep = pd.DataFrame({
        "episode_id": d["episode_id"].to_numpy(),
        "participant_id": participant,
        "n_errors": d["n_errors"].to_numpy(),
        "episode_correct": correct,
        **{rule: scores[rule] for rule in ORDER},
    })
    per_ep.to_csv(TABLES / "episode_confidence_per_episode.csv", index=False)
    print(f"wrote {TABLES / 'episode_confidence_per_episode.csv'} "
          f"({len(per_ep)} episodes x {len(ORDER)} scoring rules)")

    out_df = pd.DataFrame(rows, columns=["rule", "ece", "brier", "auroc", "auprc", "n", "note"])
    out_df.to_csv(TABLES / "episode_calibration.csv", index=False)
    print(f"wrote {TABLES / 'episode_calibration.csv'} ({len(out_df)} rows)")
    print(out_df[["rule", "ece", "brier", "auroc", "auprc", "n"]].to_string(index=False))

    fig, axes = plt.subplots(1, len(ORDER), figsize=(19.0, 4.0))
    for ax, rule, letter in zip(axes, ORDER, "abcde"):
        style(ax)
        panel(ax, reliab[rule], rule, letter, is_probability=(rule != "logsum"))
    axes[0].set_ylabel("Observed episode correctness", fontsize=9.5)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"efigure1_episode_reliability.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {FIGS / 'efigure1_episode_reliability.pdf'} / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
