# Design spec v2 - Preserving calibrated neural uncertainty across the decoder–agent boundary

Date: 2026-08-28
Status: draft for review. **Supersedes v1** (`2026-08-28-bci-agent-oversight-design.md`, commit `1136a18`).
Project dir: `/Volumes/Extreme SSD/Mimic-IV/study_bci_agent_oversight/`

---

## 0. What changed from v1, and why

v1 was reviewed and three defects were fatal. All three were design errors, not presentation problems.

| v1 defect | Why fatal | v2 response |
|---|---|---|
| Primary endpoint (UAR) rewarded abstention - an always-abstain agent scores 0, and throughput was only secondary | The predicted result was obtainable trivially; a reviewer catches it immediately | Primary endpoint is now **unsafe execution at matched action coverage / the risk–coverage frontier** (§6) |
| `request_clarification` returned the participant's next historical selection | bigP3BCI is copy-spelling: the next selection answers the next experimenter cue, not the agent's question. It is not noisy clarification, it is *not clarification* | Removed from the retrospective experiment entirely. Clarification requires prospective data (§7) |
| "Words vs structure" ladder | E/F supplied privileged information **and** external enforcement; B/C supplied neither. Summarisable as "hard-coded constraints beat asking an LLM nicely," which is not a finding | Replaced by a **factorial separating uncertainty source from control mechanism**, plus a non-LLM deterministic baseline (§5) |

Also corrected: **bigP3BCI contains no online decoder posterior.** Verified on disk - the 36 grid-cell
channels (`A_1_1`…`9_6_6`) are binary stimulus-flash indicators (two distinct values across a full
recording), not score accumulators. v1's repeated "the decoder's *actual* posterior" was an overclaim.
Confidence must be reconstructed and calibrated, and named as such (§4).

Further corrections carried in: harness dependence does not literally "cancel" in a contrast; 30 API
repetitions are not 30 independent neural observations; risk differences lead over odds ratios; the
human panel loses its physician fallback; the 35 s/selection figure is decentred.

Retained from v1 unchanged: deterministic transmission-fidelity scoring, ALS-first stratification, the
OpenRouter provider contract, frozen tool schemas, anti-overclaim language discipline.

## 1. Thesis

An uncertainty-blind AI action layer converts neural decoding errors into unauthorised actions.
Preserving **calibrated neural uncertainty across the decoder–agent interface** enables
consequence-sensitive selective autonomy, reducing unsafe execution **at matched task coverage**,
where natural-language caution and model self-confidence do not.

The claim is not "prompts bad, gates good." It is that **uncertainty is being lost at an interface
boundary, and preserving it buys measurable safety at usable levels of autonomy.**

**Engineering contribution - the Neural Action Gateway.** A model-agnostic interface contract that
carries calibrated channel uncertainty across the decoder→agent boundary, binds it to action-specific
risk constraints, and exposes an explicit safety–utility curve. The paper defines it, validates it
retrospectively, and (scope permitting, §7) tests it prospectively.

## 2. Materials

### 2.1 Decoder stream

Source: `study_bigp3_als_calibration/output/intermediate/online_trials_all20.csv`, from verified EDF
archives via `src/bigp3_als/trials.py`. Per row: intended `target`, actual `selected`, `correct`,
participant, session, condition, `phase3_time_seconds`.

Only Studies **B, F, L, N** enrolled people with ALS; the rest are able-bodied.

| Stratum | Selections | Participants (≥20) | Accuracy | Per-participant range | <0.70 / <0.50 | Median s/selection |
|---|---|---|---|---|---|---|
| **ALS (B, F, L, N)** - primary | 3,395 | 47 (42) | 0.816 | 0.219–1.000 | 8 / 4 | 35.00 (IQR 20.75–35.25) |
| Able-bodied - replication | 16,293 | 224 (224) | 0.841 | 0.214–1.000 | 33 / 9 | - |

ALS difficulty ladder: StudyN 0.694 < StudyF 0.779 < StudyL 0.839 < StudyB 0.906. ALS sessions n=9.

**Timing is decentred.** The 35 s/selection figure (→ ~140 s to spell a 4-character retraction, best
case, before accounting for 81.6% accuracy) is reported as a **channel-specific correction-cost
analysis**, not a general claim about BCIs. Contemporary intracortical speech BCIs operate far faster,
and a reviewer will say so. It is also a *transmission floor*: already-attending, already-decided user,
excluding perception and decision latency.

### 2.2 Model panel

Multi-vendor frontier panel via OpenRouter, prespecified. Provider contract in §9.2.

## 3. Ground truth - what it can and cannot establish

Channel carried true string `S`; decoder emitted `S'`; agent executed `A`. `A` is **faithful** iff
`A == entail(S)` under a pre-specified frozen string→action mapping. No claim is made about any
participant's mental state.

**Stated honestly, this establishes:** whether an agent preserves information transmitted through an
empirically noisy neural-control channel.

**It does not establish:** whether an autonomous agent preserves a disabled user's intended real-world
action. Participants were attending copy-spelling cues; they did not intend to decline care, message
anyone, or summon a nurse. These are close but not identical, and the manuscript must say so in
Limitations rather than blur them.

**Harness dependence is mitigated, not eliminated.** A within-harness contrast removes the *level*
artifact but treatment effects can still interact with tool descriptions, affordances, and scenario
construction. The scaffold-robustness arm (§5.4) bounds this; it does not abolish it. v1's "defeats all
three at once" is withdrawn.

## 4. The uncertainty layer (now a first-class component)

There is no online posterior in the data (§0). We construct one and must earn the right to call it
calibrated.

1. **Reconstruction.** Per-stimulus classifier scores from calibration EEG via grouped-CV
   `predict_proba` (existing `features.py::_grouped_cv_predictions`), accumulated across the stimulus
   sequence to a per-cell score, normalised across the 36 grid cells to a per-selection distribution.
2. **Calibration.** Platt / isotonic, fit and evaluated with strict participant-level separation.
3. **Reporting - a results component, not a footnote.** Reliability diagrams, Brier score, ECE, per
   participant and per study; calibration under **distribution shift** (train one study, test another).
   **Per-study ECE is the primary figure; the pooled value must never be the headline.** Measured
   out-of-fold: pooled ECE 0.012 against per-study 0.069 / 0.080 / 0.091 / 0.129 - the pooled number is
   6-11x better than any individual study because studies miscalibrated in opposite directions cancel.
   Quoting it alone would assert a calibration quality that holds in no study.
   Shift is known to be substantial in this dataset - the companion transportability analysis reports
   τ 0.87, I² 86.1 - so transported calibration is an empirical question, not an assumption.
4. **Naming discipline.** "Reconstructed calibrated decoder confidence" everywhere. Never "the actual
   posterior."

An **oracle confidence** arm (constructed post hoc from known correctness) provides the upper bound on
what any confidence signal could buy.

## 5. Experiment 2 - mechanism (the core factorial)

Two factors, independently manipulated. This is the identification strategy v1 lacked.

**Factor A - uncertainty source:** `none` · `model self-confidence` (verbalized or logprob) ·
`reconstructed decoder confidence`

**Factor B - control mechanism:** `advisory` (uncertainty is given to the agent; the agent decides) ·
`enforced` (external gate; the harness decides)

This separates four questions v1 could not distinguish:

- Does knowing decoder confidence change agent behaviour at all?
- Does a hard gate work **without** the LLM?
- Can the LLM use neural uncertainty appropriately **without** enforcement?
- Does neural uncertainty beat the model's own confidence **at equal coverage**?

### 5.1 Reference arms (non-negotiable)

- **Non-LLM deterministic gate.** Plain threshold on reconstructed confidence, no LLM in the loop.
  **If this matches the agent architecture, that is the paper's most important result** - the
  contribution is then the interface, not the model. Designed to be able to lose.
- **Random gate at matched coverage.** Isolates how much of any gate's benefit is coverage reduction
  alone.
- **Oracle confidence.** Upper bound.
- **Caution-prompt family.** The v1 rescue battery survives, demoted from headline to one cell of
  Factor A × B (`none` × `advisory`), which is what it always was.

### 5.2 Primary endpoint

**Unsafe execution rate at matched action coverage**, plus the full **risk–coverage frontier** and its
area (AURC). An always-abstain system lands where it belongs: perfectly safe, entirely useless.

Secondary: successful intended-action throughput under a prespecified safety constraint;
tier-weighted unsafe execution; selections and seconds per completed intent.

### 5.3 The coverage-knob problem - resolved explicitly

Matched-coverage comparison needs a continuous operating knob per arm. **Gate arms have one** (the
threshold) and sweep a genuine curve. **Advisory and prompt arms do not** - "caution wording" cannot be
dialled to 80% coverage. Resolution, pre-specified:

- Gate arms produce continuous risk–coverage curves via threshold sweep.
- Advisory/prompt arms are plotted as **points** in the risk–coverage plane, and the pre-specified test
  is **dominance**: does the point lie on or above the gate frontier?
- An ordered wording-strictness family gives advisory arms a coarse **ordinal** pseudo-curve, labelled
  ordinal, never interpolated as continuous.

This is stated up front because it determines what the primary comparison can mean.

### 5.4 Robustness arms

- **Scaffold robustness.** 2–3 semantically neutral paraphrases of tool descriptions and task framing
  within each cell. Nuisance factor; enters the variance structure (§8). Distinct from the caution-
  content arm, which is an exposure.
- **Protocol sensitivity.** Model subset run twice: native tool-calling and uniform text/JSON protocol.
- **Cohort.** ALS primary, able-bodied replication.

## 6. Experiment 1 - retrospective discovery

Does *agency* itself change how decoder errors propagate? Three controllers over the same decode stream:

1. **Deterministic router** (no LLM) - the null.
2. **Single-shot LLM** (no tools) - the intent-drift regime.
3. **True agent** (multi-turn, tool-using, real environmental feedback).

Endpoint: unsafe execution as a function of reconstructed confidence, by controller class and
consequence tier. Establishes whether the agentic layer amplifies, attenuates, or merely passes through
decoder error - which v1 assumed rather than measured.

## 7. Experiment 3 - prospective closed-loop (**explicit scope decision, not assumed**)

Retrospective data cannot support clarification (§0) and cannot establish that agents preserve
*intended real-world action* (§3). Only prospective work closes those gaps. Three tiers, costed
differently; **the choice is the user's and is not folded silently into the redesign**:

**DECIDED 2026-08-28: TIER 0.** No prospective human participants. The study uses open datasets only.
This is a scope decision by the principal investigator, not a limitation discovered late, and the
manuscript must present it as such.

| Tier | Content | Cost | Ceiling |
|---|---|---|---|
| **0 - CHOSEN** | None. Computational only, gaps stated as limitations | Compute only | NMI / npj Digital Medicine / NEJM AI |
| 1 (not taken) | Prospective non-ALS BCI participants | IRB + hardware + recruitment | NBE plausible |
| 2 (not taken) | Prospective ALS/AAC participants | Years | NBE / Nature Medicine credible |

**What Tier 0 costs us, stated plainly so it is never quietly forgotten.** Two claims become
permanently unavailable and must not appear in any draft:
1. **Clarification.** `request_clarification` cannot be studied at all. Copy-spelling data cannot
   answer an agent's question (§0), and no prospective arm exists to supply one.
2. **Intended real-world action.** The study establishes that an agent preserves or corrupts
   information transmitted through an empirically noisy neural channel (§3). It cannot establish that
   an agent preserves a disabled user's *intended* action. Limitations must say this in those words.

The end-user panel (§10) remains valuable and is unaffected by this decision - rating which action
tiers require confirmation is offline work requiring no BCI session.

## 8. Statistics

**Preserve the hierarchy.** participant → session → neural episode → repeated generation. Repetitions
estimate *model stochasticity only* and enter as the lowest level; they are not independent biological
observations. Nested random effects, or cluster bootstrap at participant level. Scaffold wording and
scenario enter as variance components.

**The model panel is a prespecified selection, not a random sample.** Report consistency across the
panel and per-model estimates with CIs. Do **not** present a paired test as generalising to "all
models."

**Effect measures.** Absolute **risk differences and risk ratios lead**; odds ratios secondary. Handle
zero-event cells explicitly (Firth penalisation or exact methods) - they are expected at high-threshold
gate settings.

Benjamini-Hochberg across the primary family. Power analysis **before** launch. ≥30 repetitions per
cell for stochasticity, run-variance reported, **no temp-0 determinism claim**. Single
`output/results_digest.json` + `output/stats_digest.json` as source of truth. Descriptive language only.

## 9. Agent architecture and harness

### 9.1 It must actually be an agent

An agent is an LLM in a loop that plans, calls tools, and **uses environmental feedback at each step**;
workflows follow predefined code paths. `study_bci_llm_intent_drift/idrift/models/o2_infer.py` is a
single-shot Ollama `/api/generate` call (`num_predict:128`, regex confidence scrape) and **cannot be
reused**. Requirements: multi-turn loop, real tool results, dedicated typed tools (never a shell),
explicit max-iteration stop, no cross-session memory in any arm.

**Frozen-schema constraint.** Tool *descriptions* vary only where designated as an exposure; tool
*schemas* are byte-identical across all cells. Consequence tiers are conveyed only through naturalistic
affordances and scenario text - never an explicit tier label.

**What this does and does not claim.** The study does NOT claim the agent lacks access to stakes
information, and must never be written as if it did. Action names such as `summon_staff` versus
`set_light` plainly carry stakes semantics, and that is intended: consequence-sensitive selective
autonomy is only possible if the agent can infer stakes from the action itself. If it could not,
tier-sensitivity would be impossible by construction and the consequence axis would be dead on
arrival. A deployed assistive system would likewise have meaningful action names. The discipline is
narrower and precise: no *explicit tier vocabulary* ("tier 3", "irreversible", "consequence tier")
appears in prompt-facing text, and no tier structure is encoded incidentally - including
**positionally**, in the ordering of the `execute` enum, which must therefore be a frozen shuffle
rather than the tier-grouped order of `ACTIONS`.

`request_clarification` is **absent** from the retrospective experiments and appears only in
Experiment 3.

### 9.2 OpenRouter contract

Native tool-calling primary (OpenRouter normalises `tools` to OpenAI shape across providers); uniform
text protocol as the robustness arm.

1. **Pin the provider.** Default routing is price-based across providers serving different
   quantizations, so an unpinned slug is not a fixed object across the run. Every request:
   `"provider": {"order": ["<pinned>"], "allow_fallbacks": false, "quantizations": ["bf16"]}`.
   Log `openrouter-selected-provider` per request, assert it matches, ship the log as an eTable.
2. **Set every sampling parameter explicitly.** OpenRouter omits absent parameters upstream, letting
   providers apply their own defaults. `temperature`, `top_p`, `max_tokens` on every call.
3. `seed` exists; determinism is not guaranteed for all models.

**Pre-specified parse-failure rule.** Malformed-output rate per cell; exclude cells >15%; sensitivity
analysis excluding any-failure episodes. Lenient parsing.

## 10. End-user panel - integral, no physician fallback

ALS/AAC end users rate which action tiers require confirmation before execution, anchoring the
consequence taxonomy. **For a paper about autonomy, clinicians cannot be the fallback arbiters of what
risks users are willing to tolerate.** v1's two-physician fallback is withdrawn; physician adjudication
may supplement but not substitute.

**Status: recruitment gate, user-owned.** Everything else runs offline.

## 11. Prior work

- **Tai, arXiv:2606.09315** - nearest neighbour: BCI→agent route safety, confirmation, formal contract,
  conformal calibration on EEGMMI. Adversarial perturbation framing. **"They prove X, we show what
  repairs X" is insufficient on its own** if the repair is foreseeable - hence the Gateway contract,
  the matched-coverage frontier, and the non-LLM baseline. *(Sole author Jianwei Tai; a scout verifier
  falsely attributed it to this lab. Verified by direct fetch.)*
- **arXiv:2604.23283** reversibility taxonomy · **arXiv:2606.08919** bounded oversight capacity ·
  **arXiv:2602.16943** text-safety ≠ tool-call safety · **AgentHarm 2410.09024** · **BRACE / shared-
  autonomy** confidence-gated arbitration (establishes the principle in non-LLM control; we test whether
  LLM agents honour it).
- **Own work, cite-and-differentiate:** LLM-in-BCI systematic review (BPEX 2026); intent-drift (text
  faithfulness); authorship attribution; trial-adaptive fusion; the ward-agent companion.
- **Asimov framing excluded from the spine.** One Discussion sentence at most.

## 12. Venue

Given the Tier 0 decision (§7), the realistic target set narrows and the spec records that honestly:

- **Nature Machine Intelligence** - the strongest available fit, since the contribution is agent
  behaviour and safety architecture rather than a new BCI. Still a high bar: NMI has published BCI
  work with live tasks and a participant with paralysis, so a purely offline study needs an unusually
  strong mechanistic contribution to compete. The Neural Action Gateway contract plus the
  matched-coverage frontier and the non-LLM baseline are that contribution or nothing is.
- **npj Digital Medicine / NEJM AI** - the credible, realistic home. NEJM AI has accepted
  simulation-only work before.
- **Nature Biomedical Engineering** - a reach, not a plan. NBE's recent BCI work carries online human
  validation, which Tier 0 forecloses. Submit only if Experiments 1-2 produce an unusually clean
  mechanistic result.
- **Nature Medicine** - OUT. Not a clinical medicine study, and Tier 0 removes the only route in.

Final selection at Phase 8, but the framing should be written toward NMI / npj from the start rather
than written toward NBE and then downgraded.

## 13. Scope guardrails

One paper. No fabricated data, citations, IRB numbers, or estimates. Local git only; no push or
submission without instruction. Authors/affiliations/repo URL stay placeholders. **Costed run plan and
power analysis required before any model call.**

## 14. Open items

1. **Experiment 3 tier - user decision** (§7). Gates the venue ceiling.
2. End-user panel recruitment - user-gated (§10).
3. Costed run plan + power analysis - implementation plan.
4. Consequence taxonomy contents - draft, then end-user reviewed.
5. Frozen string→action mapping - pre-specified before any model call.
6. Calibration transport design - which studies train, which test (§4.3).
