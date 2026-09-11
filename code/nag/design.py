"""Experimental design: cells, one shared stratified episode set, cost, and power.

STALE-BRIEF NOTICE: the original Task 5 brief predates several rulings and is
superseded wherever it disagrees with this module. In particular:

  Ruling 5/Ruling-21: episodes are 3 characters (not 5), and the ALS pool is
    FIXED at ~1,084-1,088 episodes -- there is no room for the brief's "300
    episodes per cell, sampled independently."

  Ruling 6: the design is FULLY PAIRED. Every experimental cell sees the SAME
    shared episode set -- `sample_episodes` therefore takes no `per_cell` or
    `cell` argument at all, unlike the brief's `sample_episodes(eps, per_cell,
    seed)` signature. This turns the matched-coverage contrast within-episode
    and caps cost, since one physical episode pool serves every cell.

  Ruling 8: the dense codebook's tier distribution is uneven over real
    strings. Balance is achieved in SAMPLING here, never by retuning
    `nag.taxonomy`'s frozen salt.

  Ruling 23: `fit_match` has three values and one of them
    (`earlier_session`, n=19 episodes / 77 selections) is ANTI-predictive --
    confidence there is *negatively* correlated with correctness. That
    stratum is pre-specified for exclusion from the primary analysis (with
    a sensitivity analysis including it), decided here, before any model
    call, so it can never look post-hoc.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from nag.confidence import episode_confidence, fit_calibrator
from nag.episodes import ALS_STUDIES, build_episodes
from nag.taxonomy import entail

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONLINE_CSV = (
    REPO_ROOT.parent / "study_bigp3_als_calibration" / "output" / "intermediate" / "online_trials_all20.csv"
)
DEFAULT_SCORES_PATH = REPO_ROOT / "output" / "intermediate" / "selection_scores.parquet"

# --- cells -------------------------------------------------------------

UNCERTAINTY_SOURCES = ("none", "self_confidence", "decoder_confidence")
CONTROL_MECHANISMS = ("advisory", "enforced")
N_CAUTION_WORDINGS = 12
N_SCAFFOLDS_CORE = 3
EARLIER_SESSION = "earlier_session"


@dataclass(frozen=True)
class Cell:
    name: str
    uncertainty_source: str
    control_mechanism: str
    scaffold: int = 0
    wording: int = 0
    uses_llm: bool = True


def enumerate_cells() -> list[Cell]:
    """The full cell set: the 3x2 factorial (x3 scaffold paraphrases), the
    12-wording caution family, an oracle-confidence arm, a single-shot LLM
    controller, and two NO-LLM reference gates (`nonllm_gate`, `random_gate`
    -- see `cost_estimate`, which must exclude these from the LLM cost model).
    """
    cells: list[Cell] = []
    for s in UNCERTAINTY_SOURCES:
        for m in CONTROL_MECHANISMS:
            for k in range(N_SCAFFOLDS_CORE):
                cells.append(Cell(f"factorial:{s}:{m}:s{k}", s, m, scaffold=k))
    for w in range(N_CAUTION_WORDINGS):
        cells.append(Cell(f"caution:w{w}", "none", "advisory", wording=w))
    cells.append(Cell("oracle", "oracle", "enforced"))
    cells.append(Cell("singleshot", "none", "advisory"))
    cells.append(Cell("nonllm_gate", "decoder_confidence", "enforced", uses_llm=False))
    cells.append(Cell("random_gate", "none", "enforced", uses_llm=False))
    return cells


def factorial_base_cells(cells: list[Cell] | None = None) -> list[Cell]:
    """The 6 primary uncertainty x mechanism combos at the reference scaffold
    (scaffold 0) -- what the broader ~15-model panel runs (Task 5 dispatch:
    "broader ~15-model panel on the 6 factorial cells only")."""
    cells = enumerate_cells() if cells is None else cells
    return [c for c in cells if c.name.startswith("factorial:") and c.scaffold == 0]


# --- the fixed, fully-scored episode pool -------------------------------

def build_episode_pool(
    online_csv: Path = DEFAULT_ONLINE_CSV,
    scores_path: Path = DEFAULT_SCORES_PATH,
    length: int = 3,
) -> pd.DataFrame:
    """The fixed ALS episode pool: `nag.episodes.build_episodes(length=3,
    als_only=True)` intersected with fully-scored selections in
    `selection_scores.parquet`, with entailed tier and calibrated episode
    confidence attached.

    An episode is USABLE only when every one of its `length` selections has a
    non-null `score_top` and `correct` in selection_scores.parquet -- never
    imputed. Confidence is CALIBRATED (via `nag.confidence.fit_calibrator` on
    all valid ALS selections, participant-grouped) and combined per episode
    via `nag.confidence.episode_confidence(..., expected_len=length)`, which
    raises rather than returning a vacuous 1.0 for a short/empty episode
    (Ruling 13). Tier is computed from the DECODED string, not the true
    string -- `nag.taxonomy.entail(decoded_string)` is what an LLM agent
    reading `read_buffer` actually sees (see `nag.controllers` docstring for
    why this distinction matters for baselines that could otherwise cheat).

    Measured on the real data (2026-08-28, post Ruling-20 fix,
    3,400/3,400 selection rows): 1,088 episodes from `build_episodes`, 1,084
    fully scored (the shortfall is 4 episodes touching an EEG-artifact-
    rejected stimulus), 47 participants, 364 error-bearing (33.6%, close to
    the natural ALS-episode prevalence of 33.7%). `fit_match` splits
    own_session_own_condition=710 / own_session_other_condition=355 /
    earlier_session=19. This differs slightly from the ~1,065/~366 figures in
    the Task 5 dispatch message, which were measured mid-regeneration before
    the Ruling-20 fix's final parquet (3,400 rows, up from 3,323) landed --
    see progress.md's "INTEGRATION VERIFIED" entry. The 1,065 figure
    corresponds almost exactly to 1,084 minus the 19 `earlier_session`
    episodes (this module's PRIMARY analysis pool), which is presumably what
    the dispatcher had in mind.
    """
    online = pd.read_csv(online_csv)
    eps = build_episodes(online, length=length, als_only=True)

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
    scores_idx = scores.set_index(["relative_path", "trial_number"])

    rows = []
    for eid, keys in ep_trial_keys.items():
        sub = []
        ok = True
        for k in keys:
            if k not in scores_idx.index:
                ok = False
                break
            r = scores_idx.loc[k]
            if pd.isna(r["score_top"]) or pd.isna(r["correct"]):
                ok = False
                break
            sub.append(r)
        if not ok:
            continue
        subdf = pd.DataFrame(sub)
        fit_matches = subdf["fit_match"].unique().tolist()
        if len(fit_matches) != 1:
            # Episodes never span files (nag.episodes docstring), and
            # fit_match is resolved once per (study, participant, session,
            # condition) Test unit (03b_selection_scores.py), so every
            # selection in one episode must share it. A split here would
            # signal a join-key bug, not real heterogeneity.
            raise ValueError(f"episode {eid}: selections span multiple fit_match units {fit_matches}")
        rows.append(dict(episode_id=eid, fit_match=fit_matches[0], score_tops=subdf["score_top"].tolist()))
    joined = pd.DataFrame(rows, columns=["episode_id", "fit_match", "score_tops"])

    pool = eps.merge(joined, on="episode_id", how="inner")
    pool["err"] = pool["n_errors"] > 0
    pool["tier"] = pool["decoded_string"].map(lambda s: entail(s).tier)

    valid = scores[scores["score_top"].notna() & scores["correct"].notna()].copy()
    valid["correct_int"] = valid["correct"].astype(bool).astype(int)
    fit = fit_calibrator(
        valid["score_top"].to_numpy(), valid["correct_int"].to_numpy(), valid["participant_id"].to_numpy()
    )
    pool["confidence"] = pool["score_tops"].map(
        lambda s: episode_confidence(fit.calibrator.transform(np.asarray(s)), expected_len=length)
    )
    return pool.drop(columns=["score_tops"]).reset_index(drop=True)


# --- one shared, stratified sample --------------------------------------

DEFAULT_SAMPLE_SIZE = 400  # Ruling 6: 200 error-bearing + 200 clean, shared by every cell


def sample_episodes(pool: pd.DataFrame, n_total: int = DEFAULT_SAMPLE_SIZE, seed: int = 0) -> pd.DataFrame:
    """The ONE shared, stratified episode set every cell and every model runs.

    There are only ~1,084 usable ALS episodes in total -- nowhere near enough
    for independent per-cell samples -- so every experimental cell sees this
    SAME set (Ruling 6). This function therefore takes no `cell` or
    `per_cell` argument at all: the between-arm contrast is within-episode by
    construction, not a comparison across different draws.

    Stratified on three axes:
      - error-bearing vs clean, enriched from the natural ~34% prevalence to
        50/50. Arm CONTRASTS are unbiased on the enriched set because every
        arm sees identical episodes; ABSOLUTE rates must be reweighted back
        toward natural prevalence for reporting (see `natural_error_rate`).
      - entailed consequence tier (1/2/3, from the DECODED string), balanced
        by sampling -- never by retuning the frozen codebook salt (Ruling 8).
      - `fit_match`: sampled at each (err, tier) cell's own natural mix, so
        the rare, anti-predictive `earlier_session` stratum (Ruling 23) is
        still represented for the pre-specified sensitivity analysis,
        without being force-balanced against strata that discriminate the
        normal direction.

    Within each (err, tier) stratum, episodes are drawn round-robin across
    participants (seeded shuffle) before any plain top-up fill, so no single
    prolific participant dominates the shared set.

    Returns the sampled subset of `pool` plus `primary_eligible` (False iff
    `fit_match == "earlier_session"`) -- the pre-specified exclusion.
    Excluded episodes stay IN the returned set, because the sensitivity
    analysis needs them; callers computing the PRIMARY analysis must filter
    on this column rather than expect it pre-dropped.
    """
    rng = np.random.default_rng(seed)
    pool = pool.reset_index(drop=True).copy()
    pool["primary_eligible"] = pool["fit_match"] != EARLIER_SESSION

    err_buckets = [True, False]
    tiers = sorted(pool["tier"].unique())
    cell_keys = [(e, t) for e in err_buckets for t in tiers]
    n_cells = len(cell_keys)
    base, remainder = divmod(n_total, n_cells)
    targets = {k: base for k in cell_keys}
    # give the remainder to the largest cells first, so a small stratum is
    # never asked to exceed its own availability just to round the total up
    sizes = {k: int(((pool.err == k[0]) & (pool.tier == k[1])).sum()) for k in cell_keys}
    for k in sorted(cell_keys, key=lambda k: -sizes[k])[:remainder]:
        targets[k] += 1

    chosen_ids: list[str] = []
    for err, tier in cell_keys:
        cell = pool[(pool.err == err) & (pool.tier == tier)]
        target = min(targets[(err, tier)], len(cell))
        chosen_ids.extend(_round_robin_by_participant(cell, target, rng))

    sampled = pool[pool["episode_id"].isin(chosen_ids)].copy()
    if len(sampled) < n_total:
        # a stratum ran dry; top up from whatever's left so the shared set
        # still hits n_total where the pool as a whole can support it
        remaining = pool[~pool["episode_id"].isin(chosen_ids)]
        n_top_up = min(n_total - len(sampled), len(remaining))
        if n_top_up > 0:
            top_up = remaining.sample(n=n_top_up, random_state=int(rng.integers(1 << 31)))
            sampled = pd.concat([sampled, top_up], ignore_index=True)
    return sampled.reset_index(drop=True)


def _round_robin_by_participant(cell: pd.DataFrame, target: int, rng: np.random.Generator) -> list[str]:
    if target <= 0 or len(cell) == 0:
        return []
    groups = {
        pid: g["episode_id"].sample(frac=1, random_state=int(rng.integers(1 << 31))).tolist()
        for pid, g in cell.groupby("participant_id")
    }
    order = list(groups.keys())
    rng.shuffle(order)
    chosen: list[str] = []
    round_idx = 0
    while len(chosen) < target:
        progressed = False
        for pid in order:
            if round_idx < len(groups[pid]):
                chosen.append(groups[pid][round_idx])
                progressed = True
                if len(chosen) == target:
                    return chosen
        if not progressed:
            break
        round_idx += 1
    return chosen


def natural_error_rate(pool: pd.DataFrame) -> float:
    """Natural (unenriched) error-bearing prevalence in the full pool -- the
    reweighting target for absolute-rate reporting off the enriched sample."""
    return float(pool["err"].mean())


# --- cost model ----------------------------------------------------------

CORE_MODELS = 6
PANEL_MODELS = 15
VARIANCE_CELLS, VARIANCE_REPS, VARIANCE_EPISODES = 5, 30, 50  # brief's allocation principle, unchanged
CALLS_PER_EPISODE_LIKELY = 2.5  # most episodes terminate in read_buffer -> execute
CALLS_PER_EPISODE_CEILING = 4.0
IN_TOK, OUT_TOK = 2000, 300
PRICE_BLENDS = ((0.5, 1.5), (1.0, 4.0), (2.0, 10.0), (3.0, 15.0))
REALISTIC_BLEND = (1.00, 4.57)  # 3 frontier / 5 mid / 7 open-weight = 15 models
CACHED_FRACS = (0.0, 0.75)


def run_allocation(n_total: int = DEFAULT_SAMPLE_SIZE, cells: list[Cell] | None = None) -> dict:
    """Episode-run counts for the main run (core panel + broad panel) plus
    the fixed variance sub-study, on the shared `n_total`-episode set.

    "Episode-runs" (episode x cell x model combinations) is the unit that
    drives LLM spend under the paired design: the SAME `n_total` physical
    episodes are reused across every cell and model (Ruling 6), so cost
    scales with how many (cell, model) combinations run over them, not with
    the physical pool size alone.
    """
    cells = enumerate_cells() if cells is None else cells
    llm_cells = [c for c in cells if c.uses_llm]
    non_llm_cells = [c for c in cells if not c.uses_llm]
    n_factorial_base = len(factorial_base_cells(cells))

    core_eps = CORE_MODELS * len(llm_cells) * n_total
    panel_eps = PANEL_MODELS * n_factorial_base * n_total
    variance_eps = VARIANCE_CELLS * VARIANCE_REPS * VARIANCE_EPISODES
    return dict(
        n_total=n_total,
        core_models=CORE_MODELS,
        panel_models=PANEL_MODELS,
        llm_cells=len(llm_cells),
        non_llm_cells=[c.name for c in non_llm_cells],
        factorial_base_cells=n_factorial_base,
        core_episode_runs=core_eps,
        panel_episode_runs=panel_eps,
        variance_episode_runs=variance_eps,
        total_episode_runs=core_eps + panel_eps + variance_eps,
    )


def cost_estimate(manifest: dict, price_in: float, price_out: float, cached_frac: float) -> dict:
    """USD cost for the token volume implied by `manifest`.

    `manifest["total_episodes"]` is total EPISODE-RUNS (episode x cell x
    model combinations) -- see `run_allocation`, not the ~400-episode
    physical pool size, since the paired design reuses that same physical
    pool across every cell and model. `cached_frac` of input tokens (the
    frozen system+tool-schema prefix; measured at ~60% of a call's ~2,000
    input tokens) are billed at 10% of `price_in`, a standard prompt-cache
    read discount -- never free, and never the full price either.
    """
    calls = manifest["total_episodes"] * manifest["calls_per_episode"]
    tin, tout = calls * manifest["in_tok"], calls * manifest["out_tok"]
    usd_in = (tin * (1 - cached_frac) * price_in + tin * cached_frac * price_in * 0.1) / 1e6
    usd_out = tout * price_out / 1e6
    return dict(
        calls=calls, tokens_in=tin, tokens_out=tout, usd_in=usd_in, usd_out=usd_out, usd_total=usd_in + usd_out
    )


def cost_envelope(
    n_total: int = DEFAULT_SAMPLE_SIZE,
    calls_per_episode_values: Sequence[float] = (CALLS_PER_EPISODE_LIKELY, CALLS_PER_EPISODE_CEILING),
    price_blends: Sequence[tuple[float, float]] = PRICE_BLENDS,
    cached_fracs: Sequence[float] = CACHED_FRACS,
    cells: list[Cell] | None = None,
) -> pd.DataFrame:
    """The full cost table: every (price blend x cached_frac x calls/episode)
    combination, plus the REALISTIC_BLEND called out explicitly since it is
    not one of the four requested sweep blends (which include a deliberate
    pessimistic ceiling, 3/15, and a deliberate floor, 0.5/1.5)."""
    alloc = run_allocation(n_total=n_total, cells=cells)
    rows = []
    all_blends = list(price_blends) + [REALISTIC_BLEND]
    for price_in, price_out in all_blends:
        label = "realistic_15model" if (price_in, price_out) == REALISTIC_BLEND else f"{price_in:g}/{price_out:g}"
        for cached_frac in cached_fracs:
            for cpe in calls_per_episode_values:
                manifest = dict(
                    total_episodes=alloc["total_episode_runs"], calls_per_episode=cpe, in_tok=IN_TOK, out_tok=OUT_TOK
                )
                est = cost_estimate(manifest, price_in, price_out, cached_frac)
                rows.append(
                    dict(
                        blend=label, price_in=price_in, price_out=price_out, cached_frac=cached_frac,
                        calls_per_episode=cpe, total_episode_runs=alloc["total_episode_runs"],
                        total_calls=est["calls"], usd_in=round(est["usd_in"], 2), usd_out=round(est["usd_out"], 2),
                        usd_total=round(est["usd_total"], 2),
                    )
                )
    return pd.DataFrame(rows)


# --- power ----------------------------------------------------------------

DEFAULT_ICCS = (0.05, 0.15)


def power_curve(
    rate: float, n_grid: Sequence[int], n_participants: int = 42, iccs: Sequence[float] = DEFAULT_ICCS
) -> pd.DataFrame:
    """Half-width of a 95% CI on a proportion, naive (`ci_halfwidth`) and
    design-effect-adjusted for clustering on participant (one
    `ci_halfwidth_iccNN` column per entry of `iccs`).

    Precision here is bounded by the number of ALS participants -- 47 in the
    full episode pool, but only 39 have at least one error-bearing episode,
    hence "42-47" as the operative range for this cohort -- not by episode
    count. Adding episodes from participants already in the sample buys much
    less than adding new participants, because within-participant episodes
    are correlated: design effect DEFF = 1 + (avg_cluster_size - 1) * rho
    under an equal-cluster-size approximation, avg_cluster_size = n /
    n_participants (Killip et al. 2004; standard cluster-CI adjustment).
    `n_participants` defaults to 42, the conservative (lower) end of the
    observed range; pass the actual sampled-set participant count for an
    exact figure for a specific run.
    """
    rows = []
    for n in n_grid:
        naive = 1.96 * np.sqrt(rate * (1 - rate) / n)
        avg_cluster = n / n_participants
        row = dict(n=n, rate=rate, n_participants=n_participants, ci_halfwidth=naive)
        for rho in iccs:
            deff = 1 + max(avg_cluster - 1, 0.0) * rho
            row[f"ci_halfwidth_icc{rho:g}"] = naive * np.sqrt(deff)
        rows.append(row)
    return pd.DataFrame(rows)


# --- digests for the run manifest -----------------------------------------

def episode_set_digest(episode_ids: Sequence[str]) -> str:
    """sha256 over the sorted episode-id list, so the exact shared sample used
    for a run is pinned and re-verifiable independent of row order."""
    blob = json.dumps(sorted(str(e) for e in episode_ids), separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
