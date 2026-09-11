# AI agents at the brain-computer interface: separating inference from control

Analysis code, the frozen experimental instruments, the run manifest, and the derived
result tables for a computational benchmark study of what happens when a language-model
agent is placed downstream of a P300 speller decoder.

**Status: not yet submitted.** Repository: <https://github.com/BRIDGE-GenAI-Lab/AI-Agents-in-BCIs>. The archive DOI is
filled in below once it exists.

Target journal: *Nature Machine Intelligence*.

## The question

A P300 speller turns brain signals into characters. The classifier that does this computes
a confidence in every selection it makes, and the interface that hands the decoded string
to whatever comes next almost always discards that confidence. Historically that did not
matter much, because what came next was a text box and a person who could read the
character and delete it.

It matters now, because what comes next is increasingly a language-model agent that can
send a message, place a call, or record a care preference. A decoding error stops being a
mistyped character and becomes an executed action, and the person who produced the neural
signal may have no channel through which to intervene.

This study measures whether restoring the discarded confidence signal at that interface
changes what the agent does, and whether an agent given the number in its prompt uses it
better than a plain threshold on the same number does.

## What the outcome is, precisely

The ground truth here is **transmission fidelity**, not intent. For every episode the study
knows the source string the participant was copy-spelling and the string the decoder
actually emitted. An executed action is **faithful** when it is the action entailed by the
decoded string that the agent was given, and **unfaithful** when it is not.

The study never claims to know what a participant wanted. It cannot: BigP3BCI is a
copy-spelling corpus, so the source string is a task instruction rather than a freely
formed intention. Every statement in this codebase is about whether a signal crossed an
interface intact, and any wording that suggests otherwise is a bug in the wording.

The confidence the study gates on is likewise **reconstructed**, not recovered. BigP3BCI
records no online classifier score: its 36 grid-cell EDF channels are binary
stimulus-flash indicators, not score accumulators. Confidence is refitted from each unit's
own calibration block, scored on held-out test data, and calibrated participant-grouped out
of fold. It is referred to throughout as reconstructed calibrated decoder confidence and
must never be described as the decoder's own output.


## The repeated-attempt replay

The primary analyses score one decision per episode, which charges an abstention as coverage
lost and nothing more. A person whose system declines has not finished their task; they must
make another BCI attempt. `code/scripts/21_repeated_attempt_replay.py` replays the 200 naturalistic
episodes as tasks allowing up to three attempts, in a persistent assistive environment
(`code/nag/sandbox.py`) where an executed action changes state rather than merely being scored.

Two properties of it matter and are easy to overstate. **It is not closed loop in the sense a
BCI paper usually means, which is why it is not called that.** The agent replays decisions that were already recorded, so it never
perceives the sandbox and never learns that a previous attempt failed; the loop closed here is
the retry, not the agent's own feedback. And it is **population-level, not participant-specific**:
only 6 of 155 participant-by-command pairs hold the three episodes a same-participant retry would
need, so a retry may come from a participant whose decoding was better than the one who failed.

Because every policy's decision for every donor episode was already recorded, a trajectory is
fully determined by the sequence of donors drawn, so `code/nag/replay.py` computes the outcome
distribution **exactly**, by enumerating the draw tree, rather than by Monte Carlo. There is
therefore no simulation sample size to mistake for a neural one: it remains 200 episodes from 46
participants. `simulate()` exists only as an independent check, executing real state changes
through the sandbox, and `tests/test_replay.py` requires the two to agree.

The frozen protocol is `docs/specs/2026-09-01-repeated-attempt-replay-protocol.md`, committed before
any replay was run.

## The design in one page

**Episodes.** Each test file is partitioned into consecutive non-overlapping blocks of
three selections. Three decoded characters make one episode. An episode is kept only when
every constituent selection carries both a non-null decoder score and a non-null
correctness label. Nothing is imputed.

**Actions.** A frozen codebook (`code/nag/taxonomy.py`) maps a three-character string to
one of nine actions by SHA-256 modulo nine, so the mapping is dense and deterministic and a
one-character decoding error sends the entailed action essentially anywhere. That is
deliberate: it isolates uncertainty gating from any lexical structure a language model
might exploit. The nine actions carry three consequence tiers, three actions each:

| Tier | Actions |
|---|---|
| 1 | `save_note`, `play_media`, `set_light` |
| 2 | `send_message`, `place_call`, `post_update` |
| 3 | `summon_staff`, `record_refusal`, `record_consent` |

No tier label ever appears in prompt-facing text. The agent sees an action name and can
look up its description; it is never told which actions are consequential.

**The agent.** A multi-turn tool-calling loop over four frozen tools: `read_buffer`,
`lookup_action`, `execute`, `abstain`. The last two are terminal. The tool schemas are
byte-identical across every cell of the main study, and their SHA-256 digest is recorded in
the manifest so that claim is checkable rather than assertable.

**The factorial.** Uncertainty source (none, decoder confidence, or the model's own
elicited confidence) crossed with control mechanism (advisory, meaning the number is
rendered into the prompt and the model decides; or enforced, meaning a threshold decides
and the model is not shown the number). Three prompt scaffolds are a nuisance factor.
Twelve caution wordings are a separate exposure that tests whether telling the agent to be
careful survives a change of phrasing, which is the difference between a finding and a
prompt-engineering artefact. Thirty-four cells in total, enumerated by
`nag.design.enumerate_cells()`.

**The reference arms.** A deterministic threshold gate on the same calibrated confidence,
a random gate that acts at a matched rate without using the score, and an error-indicator
arm that is handed the correctness flag but not the source string, which bounds error
detection rather than error correction. The two gates issue zero API requests by
construction.

**The primary comparison.** Unfaithful execution at **matched action coverage**. A bare
unfaithful-execution rate is not comparable across arms, because an arm can always lower it
by acting less often. Every advisory arm is a fixed operating point with no threshold to
sweep, so it is compared against the gate's frontier read at that arm's own coverage, with
a joint participant-cluster bootstrap that resamples once and applies the same draw to both
sides of the contrast.

**The population.** Primary analyses are **end to end**, over every episode. A deployed
system cannot know which episodes contain a decoding error, so a coverage figure
conditioned on that unobservable fact is not an operating point anyone can choose. The
error-conditional analysis, which asks whether the action layer amplifies or suppresses a
decoding error that has already happened, is reported as a secondary population.

## Design vocabulary

These four words are used precisely throughout the code and the manuscript.

- **cell**: one row of the design, for example `factorial:decoder_confidence:advisory:s0`.
  Fully specified by uncertainty source, control mechanism, scaffold, and wording.
- **scaffold**: one of three paraphrases of the same system prompt. A nuisance factor.
- **arm**: a set of cells pooled over scaffold. "The decoder-confidence advisory arm" spans
  three cells.
- **episode run**: one model, one cell, one episode. The unit of the run records.

Analyses that use the factorial must filter cell names on the `factorial:` prefix.
Selecting on `(uncertainty_source, control_mechanism)` alone silently pools the twelve
`caution:w*` cells and the single-shot cell into the none-advisory baseline.

## Data

**The neural recordings are not redistributed here.** BigP3BCI is a public P300 speller
corpus available from PhysioNet:

> https://doi.org/10.13026/0byy-ry86

Download it from PhysioNet under its own licence and terms. This study analyses only the
four source studies recorded in participants with amyotrophic lateral sclerosis: StudyB,
StudyF, StudyL, and StudyN. The remaining studies were recorded in able-bodied
participants and are not analysed.

**A sibling repository is also required for the EEG stage.** `code/nag/eeg_scoring.py`
reuses the validated calibration machinery (bandpass filter, epoch window, artifact
threshold, calibration-event detection, feature decimation, trial reconstruction) from the
authors' P300 calibration study rather than re-deriving those numeric choices. It expects
that repository to sit **next to this one in the same parent directory**, under the exact
name `study_bigp3_als_calibration`:

```
<parent>/
  study_bigp3_als_calibration/     # the sibling calibration repository
  <this repository>/
```

That layout is not optional. Three files resolve the sibling by a path relative to this
repository or to the current working directory, so a checkout without it will skip or fail
the tests that touch real decodes, and step `03b` will not run at all.

The sibling `study_bigp3_als_calibration` repository is not public at the time of
writing. The calibration artefacts this study consumes are reproduced under
`output/tables/` so that nothing here depends on an unavailable repository.

**Per-episode run records are not shipped.** The parquet checkpoints under
`output/intermediate/` hold one row per episode run and are too large for a code
repository. They are available from the corresponding author on request. Everything the
manuscript reports is derived from them into `output/tables/`, `output/results_digest.json`
and `output/stats_digest.json`, all of which are shipped.

## Layout

```
code/nag/          the analysis package
code/scripts/      the numbered pipeline scripts
tests/             the pytest suite
docs/              the binding design spec, its superseded first version,
                   and the script inventory
output/tables/     every derived table, the run manifest, and the frozen
                   naturalistic-benchmark manifest
output/figures/    figures, as PDF and PNG
output/*.json      results_digest.json and stats_digest.json
```

`docs/SCRIPT_INVENTORY.md` maps every script to what it reads and what it writes, and
gives the runnable order. Read it before running anything: the numeric prefixes are
design-plan task ordinals, not pipeline positions, and two of the scripts run before
lower-numbered ones.

## Environment

Python **3.11.14**. The pinned versions in `requirements.txt` were read out of the
interpreter the pipeline is run in, by importing each package and printing its
`__version__`, rather than taken from a lockfile. `uv.lock` resolves several different
version sets across the Python range the project declares, and only one of them is what
actually ran.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # direct dependencies
# or, for a byte-exact environment including every transitive package:
pip install -r requirements-lock.txt
```

`statsmodels` appears in `requirements.txt` although this package never imports it. It is a
hard requirement of the sibling calibration package, so step `03b` fails without it.

If you work on an exFAT volume, note that a virtual environment cannot be built there;
place it elsewhere. The authors set `UV_PROJECT_ENVIRONMENT` to a path outside the data
volume for this reason.

## Running it

**Everything except the EEG stage and the live model calls runs from what is in this
repository.** Start with the tests, from the repository root:

```bash
python -m pytest tests -q
```

Expect a small number of skips: one live-API test that runs only when
`NAG_LIVE_TESTS=1` and an API key are both set, and five tests that need
`output/intermediate/selection_scores.parquet`, which is produced by step `03b` from the
EEG archive and is not shipped.

The pipeline order, the inputs, and the outputs of each of the eighteen pipeline scripts
are in `docs/SCRIPT_INVENTORY.md`. In summary:

1. **EEG to confidence** (`03b`, `03c`). Needs the PhysioNet archive and the sibling
   repository. Produces the per-selection reconstructed confidence and its calibration
   reports.
2. **Design freeze** (`05`, `05b`, `12`). No network. Enumerates the cells, draws the
   shared stratified episode sample with a fixed seed, records every digest, and stops at a
   budget gate that only a human may open. `05_design.py` writes `budget_usd: null`
   deliberately.
3. **Live runs** (`06`, `07`, `08`, `13`, `18`, `14`, `15`, `19`). Every one of these
   issues paid API requests. Each is checkpointed per model and cell, resumable, and aborts
   rather than exceed the ceiling recorded in the manifest. Spend is measured from each
   response's reported cost, never modelled.
4. **Analysis** (`09`, `10`). No network. Reads the run checkpoints and writes the tables
   and the two digests.
5. **Reporting** (`11`, `17`, `16`). Figures, the supplement built from disk with every
   table inlined, and a number audit that extracts every numeric literal from the
   manuscript and the supplement and tries to reconcile it against a value that exists in
   the result tables.

### Model access

Live calls go through OpenRouter. The API key is read from the `OPENROUTER_API_KEY`
environment variable and from nowhere else. It is never written to a file, never printed,
and never recorded in a checkpoint or a manifest. There is no key in this repository; if
you find one, treat it as a defect and report it.

```bash
export OPENROUTER_API_KEY=...
```

Provider-endpoint pinning is mandatory in `nag/openrouter.py`. OpenRouter routes the same
model slug to different providers with different tokenizers and different tool-calling
behaviour, so an unpinned multi-model study silently compares provider differences as if
they were model differences.

### What the number audit can and cannot do

`16_number_audit.py` reconciles numeric literals in the prose against values that exist in
the result tables. Two classes of error are structurally out of its reach, and this is
stated here because a clean audit report is not proof of correctness:

- A claim like "in all 5 models" matches, because 5 is a real value in the study, and the
  claim can still be false.
- A claim like "the two arms below the gate were A, B and C" is never caught, because
  "two" is a word and is never extracted as a digit.

The script therefore also prints, last, every sentence that states a count and then lists
things, and writes them to `output/tables/number_audit_enumeration_claims.csv`. That list
verifies nothing. It is a read-this list.

## The follow-up experiments

Two experiments in this repository were designed **after** inspection of the initial
benchmark, in response to specific reviewer concerns, and were frozen in version control
before any further model execution. That is good prospective discipline, but it does not
make them pre-specified, and the manuscript labels them as reviewer-motivated follow-up
experiments rather than as part of the original design.

- **`18_recalibrated_run.py`** closes a fairness objection. The gate thresholds on ordering
  alone, so recalibrating its score cannot change its frontier. The advisory arms are not
  ordering-invariant, because the numeric confidence value is rendered into their prompt.
  The two sides of the central comparison therefore did not receive equally well calibrated
  signal. This script re-runs the advisory arm on an episode-level isotonic out-of-fold
  score and recomputes the gate on that same score.
- **`19_naturalistic_run.py`** answers an ecological-validity objection. The dense SHA-256
  codebook deliberately destroys lexical structure, so it cannot show whether a language
  model would do better on ordinary commands. This benchmark maps nine natural-language
  commands one to one onto the same nine actions and corrupts them from real donor
  episodes' own error patterns. The environment does **not** repair the corrupted string:
  `nag.naturalistic.canonical_action` recognises exact commands only and returns nothing
  for `cala nurse`, so inferring the intended command is the model's job. The edit-distance
  resolver, `lexical_resolve`, is deliberately kept out of the tool surface and serves as
  the deterministic comparator the model has to beat.

`14_confirmation_run.py` gives the agent a `request_confirmation` tool that answers against
ground truth. It is a **simulated-user oracle**. It is more reliable than any real user
would be, so it bounds the benefit of a confirmation channel from above and can never be
reported as an achievable deployment result. It is absent from the default tool surface and
reachable only through `tool_schemas(confirmation=True)`, so the main study's frozen-schema
digest is unaffected.

- **`25_semantic_fair_comparison.py`** closes a third objection, raised only after the first
  two follow-ups and the naturalistic benchmark were already reported: the naturalistic
  benchmark's own deterministic comparator, an edit-distance lexical resolver, is handed the
  nine-command vocabulary as a literal, while the model reading only the system prompt is
  not. "Language models cannot beat a lexical resolver" is not a claim that comparison can
  support when only one side holds the answer key. This script reuses the same 200 frozen
  naturalistic episodes and disclosed vocabulary to run five new arms, information-symmetric
  with the resolver, across a panel widened to ten models: two direct arms (the model given
  the vocabulary, advisory and enforced), two swept resolver-plus-gate frontiers (exact-match
  and lexical), and a **hybrid architecture**, in which the model proposes a semantic
  correction by text while a deterministic gate alone retains admission authority: the
  architecture the objection actually named, rather than either side of the original
  contrast. `26_semantic_primary_table.py` builds the one table that makes the comparison
  fair (every enforced arm is threshold-swept before being reported, so a proposal rate is
  never compared against an operating point), `27_figure_semantic_comparison.py` draws the
  frontier and matched-coverage figure, and `28_semantic_fair_inventory.py` tabulates this
  run's own totals, since it falls outside the six pre-specified datasets'
  inventory. `25b_smoke_semantic_fair_comparison.py` is the live-arm-divergence smoke gate
  this study runs before any paid execution, in the same spirit as `07_smoke.py`.

## Scale of the study

Read from `output/tables/run_manifest.json`, which is frozen and committed:

| Quantity | Manifest key | Value |
|---|---|---|
| Usable episodes in the pool | `episode_pool.n_usable_episodes` | 1,084 |
| Participants | `episode_pool.n_participants` | 47 |
| Primary-eligible episodes | `principal_run.episode_set.n_total` | 1,065 |
| Error-bearing, of those | `principal_run.episode_set.n_error_bearing` | 363 |
| Excluded, calibration fitted from an earlier session | `principal_run.episode_set.excluded_earlier_session` | 19 |
| Design cells | `cells` | 34 |
| Models | `model_panel.n_models` | 5 |

The nineteen excluded episodes are removed by a rule fixed before the runs: their
calibration was fitted on a different session, and their score is anti-predictive of
correctness. They are retained for a sensitivity analysis and excluded from the primary
one.

The five-model panel above is the **primary study's** panel. The post-hoc fair-information
comparison (`25_semantic_fair_comparison.py`, below) widened the panel to ten models over
the same 200 naturalistic episodes: 6,400 episode runs, 14,336 tool-calling requests (400 of
them model-free, from the two resolver-plus-gate arms), measured cost US $19.46, tabulated
separately in `output/tables/semantic_fair_dataset_inventory.csv` because it falls outside
the six pre-specified datasets' inventory.

## Results

**No arm of the agent reached lower unfaithful execution than a deterministic threshold on
the same confidence, evaluated at the arm's own coverage.** Dominance was not achieved in
any of the 10 model-by-arm combinations. Two were significantly worse than the gate:
`anthropic/claude-sonnet-5` under advisory control by 0.087 (95% CI, 0.042 to 0.149) and
`z-ai/glm-5.3-flash` by 0.040 (95% CI, 0.017 to 0.063).

Supplying the confidence as prompt text appeared to cut unfaithful execution by up to 22
percentage points, but that came from acting less often and reversed once coverage was
matched. Two of the three models that declined often did so by emitting no valid tool call
rather than by calling `abstain`, which is not an oversight mechanism a deployment can rely
on.

In the repeated-attempt replay, where a refusal costs the user another BCI attempt, the
tool-free lexical resolver paired with the gate completed every task in 1.06 attempts at
94.1 successful tasks per 100 attempts, against 88.8 for the best language-model operating
point. Enforcement produced no detectable reduction in three-attempt completion in any
model, improved efficiency in two, and worsened it in one.

**A later, post-hoc fair-information comparison (`25_semantic_fair_comparison.py`) gave a
ten-model panel the same nine-command vocabulary the naturalistic benchmark's own
deterministic comparator already uses.** Disclosing it did not lower any arm's risk at
matched coverage relative to a resolver-plus-gate architecture: of the 20 (arm, model)
cells the two direct (non-hybrid) vocabulary-disclosed arms contribute, the 12 with an
evaluable matched comparison improved on none. But the **hybrid architecture**, in which the
model proposes a semantic correction by text while a deterministic gate alone retains
admission authority, extended coverage past the lexical resolver's 0.940 ceiling at zero
observed risk in 5 of the 10 models, and past it at some risk cost in 3 more. Across all
three vocabulary-disclosed arms, 16 of the 30 (arm, model) cells reached episodes the
resolver structurally cannot reach; 10 of those 16 carried no unfaithful execution anywhere,
including the episodes gained. **Restoring the discarded confidence signal did widen what
the agent could safely do, but only when a deterministic gate, not the model, kept
admission authority.**

Figures and tables under `output/` are derived from the principal full-pool run except
where a file or legend states that it comes from the exploratory hundred-episode set,
which is enriched 50:50 on decoding error and whose absolute rates are therefore not
benchmark risks at the observed prevalence.

Six pre-specified datasets, 50,230 episode runs, 141,879 tool-calling requests, no failed
rows, measured API spend US $86.62. The seventh, the fair-information comparison above,
added 6,400 runs, 14,336 requests and US $19.46.

## Reporting and ethics

The study analysed an open, fully de-identified archive and enrolled no participants. No
new human-subjects data were collected. Institutional review board review was therefore not
required. The source studies' own approvals and consent statements are documented in the
archive.

Large language models are the object of study here. The authors also used a generative-AI
coding assistant to help implement and test this pipeline and to help draft manuscript
text. All code was reviewed and tested by the authors and every reported number is checked
against the archived digests.

## Known gaps in this repository

Stated plainly, because a reviewer will find them anyway.

- **Three files resolve the sibling repository by a path relative to the current working
  directory** rather than to the repository root: `tests/test_taxonomy.py`,
  `tests/test_selection_scores.py`, and `code/scripts/06_cost_probe.py`. Run everything
  from the repository root, with the sibling checked out beside it, and they resolve. Run
  pytest from anywhere else and one test fails outright rather than skipping.
- **The intermediate run records are not shipped**, so `09_analysis.py`,
  `10_secondary.py` and `11_figures.py` cannot be re-run from a fresh clone. Their outputs
  are shipped instead. Request the checkpoints from the corresponding author to re-run
  them. `tests/test_semantic_primary_table.py` reads those same checkpoints (the
  `runs_semantic_fair/`, `runs_natural/` and `runs_recal/` directories) to verify
  `26_semantic_primary_table.py` against real rows, so a fresh clone's `pytest tests -q`
  reports failures and collection errors in that one file for the same reason, not a code
  defect: every test in it passes once the checkpoints are present.
- **The numeric prefixes on the scripts are not a run order.** See the inventory.

## Citation

Repository: <https://github.com/BRIDGE-GenAI-Lab/AI-Agents-in-BCIs>

> The archive DOI is inserted here once the manuscript is accepted and the archive is
> deposited. Until then, cite the repository URL and the commit.

Gorenshtein A, Omar M, Jia E, Adiniaev Y, Daniel O, Kruskal J, Ahmed M, Brook O, Klang E,
Barash Y. *AI agents at the brain-computer interface: separating inference from control.* Manuscript under review.

## Licence

MIT. See `LICENSE`.
