# Script inventory

Every file in `code/scripts/`, what it reads, and what it writes.

## Read this first: the numeric prefixes are not a run order

A script's number is the ordinal of the design-plan task that created it, not its
position in the pipeline. The sequence has gaps (there is no `01`, `02`, `04`), and
two of the numbered scripts run before lower-numbered ones. The runnable order is
the table in the next section. The prefix-to-task mapping, which a reader of the
design spec will need, is:

| Script | Design-plan task |
|---|---|
| `13_principal_run.py` | Task 12, the principal run |
| `14_confirmation_run.py` | Task 13, the confirmation-tool experiment |
| `15_repeat_run.py` | Task 14, the stochastic-repetition check |
| `18_recalibrated_run.py` | Task 19, the recalibrated-confidence run |
| `19_naturalistic_run.py` | Task 20, the naturalistic semantic benchmark |
| `20_efigure2_recalibration.py` | Task 18, eFigure 2, the recalibration sensitivity display item |

## Runnable order

| # | Script | Needs network | Reads | Writes |
|---|---|---|---|---|
| 1 | `03b_selection_scores.py` | no | BigP3BCI EDF recordings; the sibling calibration package | `output/intermediate/selection_scores_parts/*.parquet`, then `output/intermediate/selection_scores.parquet` |
| 2 | `03c_calibration_transport.py` | no | `selection_scores.parquet` | `output/tables/calibration_reliability.csv`, `output/tables/calibration_transport.csv` |
| 3 | `05_design.py` | no | `selection_scores.parquet`, the sibling's `online_trials_all20.csv` | `output/tables/run_manifest.json` with `budget_usd: null`, `output/tables/cost_envelope.csv`, `output/tables/power_curve.csv` |
| 4 | `05b_panel.py` | no | `run_manifest.json`, `output/tables/cost_probe.json` | `run_manifest.json` with the model panel, the episode sample and the human-set budget ceiling |
| 5 | `12_episode_calibration.py` | no | `selection_scores.parquet`, `run_manifest.json` | `output/tables/episode_calibration.csv`, `output/tables/episode_confidence_per_episode.csv`, `output/figures/efigure1_episode_reliability.pdf` and `.png` |
| 6 | `06_cost_probe.py` | yes, small | `selection_scores.parquet` | `output/tables/cost_probe.json` |
| 7 | `07_smoke.py` | yes, small | `run_manifest.json` | `output/tables/smoke.json` |
| 8 | `08_run.py` | yes | `run_manifest.json` | one parquet checkpoint per model and cell under `output/intermediate/runs/` |
| 9 | `13_principal_run.py` | yes | `run_manifest.json` | `run_manifest.json['principal_run']` when called with `--declare`; checkpoints under `output/intermediate/runs_principal/` when called to run |
| 10 | `18_recalibrated_run.py` | yes | `run_manifest.json`, `episode_confidence_per_episode.csv` | checkpoints under `output/intermediate/runs_recal/` |
| 11 | `14_confirmation_run.py` | yes | `run_manifest.json` | checkpoints under `output/intermediate/runs_confirmation/` |
| 12 | `15_repeat_run.py` | yes | `run_manifest.json` | checkpoints under `output/intermediate/runs_repeat/`, `output/tables/secondary_repeat_variability.csv` |
| 13 | `19_naturalistic_run.py` | yes | `run_manifest.json`, `episode_confidence_per_episode.csv` | `output/tables/naturalistic_manifest.json` when called with `--build-manifest`; checkpoints under `output/intermediate/runs_natural/` when called to run |
| 14 | `09_analysis.py` | no | run checkpoints | `output/tables/primary_end_to_end.csv`, `primary_aurc.csv`, `primary_matched_coverage.csv`, `primary_contrast.csv`, `primary_abstention_mechanism.csv`, `output/results_digest.json` |
| 15 | `10_secondary.py` | no | run checkpoints | `output/tables/secondary_caution_battery.csv`, `secondary_caution_tests.csv`, `secondary_scaffold_spread.csv`, `secondary_by_tier.csv`, `secondary_tier_sensitivity.csv`, `secondary_tier_transitions.csv`, `secondary_reference_arms.csv`, `secondary_parse_sensitivity.csv`, `output/stats_digest.json` |
| 16 | `11_figures.py` | no | run checkpoints, the primary and secondary tables | `output/figures/figure1_risk_coverage_frontier` (the standalone build of Figure 1's panel b), `Figure2`, `Figure4`, `eFigure3`, `eFigure4`, each as `.pdf` and `.png`, plus `output/tables/figure5_naturalistic.csv` and `output/figures/figure_checks.json`. It also exposes `draw_risk_coverage(ax, d, cells)`, which `23_figure1_architecture.py` calls to compose the same panel rather than duplicating the plot |
| 17 | `17_build_supplement.py` | no | `output/tables/`, `run_manifest.json`, `naturalistic_manifest.json`, run checkpoints | `manuscript/supplement.md`, with every advertised table inlined |
| 18 | `16_number_audit.py` | no | `manuscript/manuscript.md`, `manuscript/supplement.md`, `output/tables/*`, both digests, the two frozen instrument files | `output/tables/number_audit.csv`, `output/tables/number_audit_enumeration_claims.csv` |
| 19 | `20_efigure2_recalibration.py` | no | `runs_principal/` and `runs_recal/` checkpoints, `followup_task19_recalibration.csv`, `followup_task19_matched_coverage.csv` | `output/figures/efigure2_recalibration_sensitivity.pdf` and `.png`, `output/tables/efigure2_recalibration_paired.csv` |
| 20 | `21_repeated_attempt_replay.py` | no | `runs_natural/` checkpoints, `naturalistic_manifest.json` | `output/tables/repeated_attempt_replay.csv`, `repeated_attempt_contrasts.csv`. Enumerates the three-attempt draw tree in closed form from decisions already recorded; it makes no model call and adds no neural data |
| 21 | `22_figure5_replay.py` | no | `runs_natural/` checkpoints, the two replay tables | `output/figures/Figure5.pdf` and `.png` |
| 22 | `23_figure1_architecture.py` | no | run checkpoints, the primary tables | `output/figures/Figure1.pdf` and `.png`, both panels |
| 23 | `24_figure3_abstention.py` | no | `primary_abstention_mechanism.csv` | `output/figures/Figure3.pdf` and `.png` |

`freeze_sonnet_subset.py` is not in the runnable order either, because it runs once, before the principal run, and its output is a list of episode identifiers recorded in `run_manifest.json` rather than a table. It draws the frozen 500-episode subset that `anthropic/claude-sonnet-5` ran in the full-pool tasks: without replacement under a fixed seed, stratified on decoder-error status and allocated across participants by largest remainder. It is shipped so that the subset-draw procedure the Methods describe can be checked rather than taken on trust.

`00_cost_envelope.py` is not in the pipeline. It is a standalone arithmetic model of run
size and price that reads nothing and writes nothing, kept because it is the record of how
the study was sized before any money was committed.

Four scripts are checks rather than pipeline steps, and are shipped so that what they enforce can be read rather than taken on trust. `check_figure_specs.py` fails the build on any figure that is not a journal column width or whose text has been converted to outlines; it deliberately does not check font size, which is a judgement a measurement cannot make. `wordcount_nmi.py` counts the abstract and main text against the journal's limits and fails if any prose in the submission packet states a count that is neither current nor a limit. `renumber_figures.py` renumbers figure cross-references in one atomic placeholder pass, because applying such a map one rule at a time silently merges two figures onto one number and leaves valid English behind. `check_packet_docx.py` verifies that every shipped `.docx` is at least as new as the `.md` it was built from, carries no superseded string, and contains every substantial paragraph of its source.

Four scripts are not shipped here. They are the maintenance scripts in the private study
repository, and they write into directories that do not exist in this tree:
`build_release.py` syncs this one, `build_packet.py` assembles the submission packet from
the manuscript sources and the figures, `build_manuscript_with_figures.py` composes the
reading copy that packet ships as item 7, and `sync_review_repo.py` is the earlier syncer
`build_release.py` superseded.

## The analysis package

| Module | Responsibility |
|---|---|
| `nag/taxonomy.py` | the frozen dense codebook mapping a decoded string to an entailed action, plus the consequence tiers. A channel code, not a claim about anyone's intent. |
| `nag/episodes.py` | deterministic episode construction from real BigP3BCI online decodes |
| `nag/eeg_scoring.py` | per-selection reconstructed calibrated decoder confidence, derived from calibration EEG. Imports the sibling calibration package. |
| `nag/confidence.py` | calibration of the reconstructed score, participant-grouped out of fold |
| `nag/episode_calibration.py` | calibration of the episode-level score, and the four alternative combination rules it is compared against |
| `nag/design.py` | cell enumeration, the shared stratified episode pool, cost and power |
| `nag/prompts.py` | the frozen prompt instrument: twelve caution wordings (the exposure) and three scaffolds (the nuisance factor) |
| `nag/tools.py` | the frozen tool surface. Schemas are byte-identical across every cell of the main study; the confirmation tool is reachable only through `tool_schemas(confirmation=True)`. |
| `nag/agent.py` | the multi-turn tool-calling loop with an explicit stopping condition |
| `nag/controllers.py` | the non-LLM reference controllers, including the deterministic confidence gate and the random gate |
| `nag/openrouter.py` | the model client, with provider-endpoint pinning made mandatory |
| `nag/naturalistic.py` | the naturalistic semantic benchmark: nine ordinary commands, the corruption model, and the lexical resolver that serves as the deterministic comparator |
| `nag/analysis_population.py` | which episodes count, and the three-dimensional outcome |
| `nag/paired_bootstrap.py` | the joint participant-cluster bootstrap that resamples once per contrast |
| `nag/riskcoverage.py` | the risk-coverage frontier, common-support AURC, matched-coverage interpolation, and the fixed-point dominance test |
| `nag/stats.py` | participant-clustered intervals, risk differences and ratios, multiplicity correction |

Two frozen instrument files travel with the package and are read by the audit: 
`nag/frozen_mapping.json` (the action codebook) and `nag/frozen_prompts.json` (the prompt
bank). Their SHA-256 digests are recorded in `output/tables/run_manifest.json` under
`digests`, alongside the tool-schema digest.
