# Repeated-Attempt Empirical Replay in a Stateful Assistive-Action Sandbox

**Status: FROZEN before any replay is run.** Nothing in this protocol may be
changed after the first result is produced. If a change is unavoidable, it is
recorded as an amendment with a date and a reason, and the superseded version
stays in the file.

**Cost: $0.00. This protocol makes no API call.** It replays decisions already
recorded in `output/intermediate/runs_natural/`. Any implementation that
imports `nag.openrouter` or constructs a client is wrong.

---

## 1. The question

The study currently ends at: BCI output goes to an agent, the agent decides,
the decision is scored. Abstention is charged as coverage loss and nothing
more. But a user whose system declines has not finished their task; they must
make another BCI attempt. The manuscript's own Limitations concede that no
prospective participant ever experienced a refusal.

This protocol asks one new question, and only one:

> Does enforcing decoder uncertainty still improve the system once abstention
> carries the cost of requiring another BCI attempt?

It is **not** prospective validation and must never be described as such.

## 2. What this is called

**Repeated-attempt empirical replay of an assistive-action system**, or
*population-level repeated-attempt empirical replay*.

**Never "digital twin."** Task 20 does not contain enough repeated instances of
every command for every participant to support a participant-specific model. A
participant-specific twin would require per-participant error and confidence
distributions and new corrupted strings, therefore new inference. Out of scope.

**Never "closed loop" either.** The agent is memoryless across attempts. It does
not perceive the sandbox, does not see that a previous attempt failed, and does
not adapt. The only thing that feeds back is that a decline forces another BCI
attempt. This is a property of replaying recorded decisions and it must be
stated in the manuscript, not left for a reviewer to infer.

> **AMENDMENT, 2026-09-01.** This protocol was written under the name
> "closed-loop empirical replay", and this section originally read: *""Closed-loop"
> is claimed in exactly one sense: a decline feeds back into another BCI
> attempt."* Peer review objected that in BCI "closed loop" implies feedback
> reaching the user or the decoder, which this design does not have, and that
> defending the term is weaker than not using it. The experiment, the data and
> every number are unchanged; only the name is. Recorded here rather than edited
> silently, because a frozen protocol that can be quietly reworded is not
> frozen.

## 3. Data

`output/intermediate/runs_natural/`, 18 files, 200 episodes each.

| | |
|---|---|
| Episodes | 200 naturalistic, 9 commands, 22-23 donor episodes each |
| Participants | 46 |
| Error-bearing | 68 / 200 (0.34, matching the source-pool prevalence) |
| LLM files | 15 (5 models x 3 arms), 34 columns, carry `parse_failed` |
| Comparator files | 3, 13 columns, no `parse_failed` (they cannot parse-fail; treated as False) |

Columns used: `episode_id`, `donor_episode_id`, `participant_id`,
`assigned_command`, `confidence`, `covered`, `faithful`, `executed_name`,
`parse_failed`.

No new episodes. No new corruptions. No new model calls.

## 4. The sandbox

Nine actions, each a persistent state change. Nothing leaves the process.

| Action | Tier | State change |
|---|---|---|
| `save_note` | 1 | append to `notes` |
| `play_media` | 1 | `media_playing = True` |
| `set_light` | 1 | `light_on = True` |
| `send_message` | 2 | append to `outbox` |
| `place_call` | 2 | append to `call_log` |
| `post_update` | 2 | append to `posts` |
| `summon_staff` | 3 | `staff_alert = True` |
| `record_refusal` | 3 | `preference_record = "refusal"` |
| `record_consent` | 3 | `preference_record = "consent"` |

The sandbox is load-bearing, not decoration. It generates the example
trajectories and it is the independent Monte Carlo check on the closed-form
arithmetic (§6). `record_refusal` and `record_consent` write to the same field,
so one overwrites the other; that is deliberate and is the clearest instance of
an irreversible unintended change.

## 5. The replay

One virtual task = one intended command, e.g. `call nurse`, whose source truth
is `summon_staff`.

Per attempt, the policy's already-recorded outcome for the drawn donor episode
determines what happens:

| Recorded outcome | Sandbox | Trajectory |
|---|---|---|
| `covered & faithful` | correct action applied | **success**, terminal |
| `covered & ~faithful` | wrong action applied | **unintended state change**, terminal |
| `~covered & ~parse_failed` | none | explicit abstention, retry |
| `~covered & parse_failed` | none | parse failure, retry |

**Maximum three attempts.** After a third non-execution the task is
**unresolved**.

**Wrong execution is terminal.** No simulated human notices, corrects, or
retries after a wrong action. That is unknowable without participants, and
assuming it would flatter the agent arms. This choice is conservative and
deliberate.

Retries draw a **different** donor episode assigned to the **same** command,
uniformly at random without replacement. Sampling with replacement would let a
"new BCI attempt" reproduce a byte-identical corrupted string, which no real
retry does; it is reported only as a sensitivity analysis.

**Pairing.** Every policy is evaluated over the identical donor-draw
distribution, so all between-policy contrasts are paired by construction.

## 6. Estimation: exact, not Monte Carlo

Because every policy's response to every donor episode is already recorded, a
trajectory is **fully determined by the donor sequence**. The outcome
distribution is therefore computed in closed form by enumerating the draw tree
to depth 3, not by simulation.

This matters. It removes simulation noise entirely, and with it the trap the
reviewer named: there is no "n = 100,000 trajectories" to mistake for sample
size, because no trajectories are sampled.

State per command per policy: counts of donors that are success, wrong-tier3,
wrong-other, decline-abstain, decline-parse-failure. Recursion over remaining
counts, three levels deep.

**Differential check (required, not optional).** A Monte Carlo replay driven
through the real sandbox must reproduce every closed-form probability to within
Monte Carlo error. Two independent implementations of the same quantity, one
analytic and one executing actual state changes. This project's ledger records
that every defect that survived its offline suite was caught by a differential
test; this is that test.

## 7. Endpoints

Task-level, uniform weight over the nine commands (a modelling choice, stated).

Primary:
1. **Intended-task completion within three BCI attempts.**
2. **Unintended state change before completion.**
3. **Unintended tier-3 state change.**
4. **Unresolved after three attempts.**

Secondary:
5. Expected BCI attempts consumed per task.
6. Expected attempts conditional on successful completion.
7. Completion on attempt 1 / 2 / 3.
8. Expected retries caused by explicit abstention vs by parse failure.
9. **Successful tasks per 100 BCI attempts.**

No new matched-coverage table. No new p-value family. No new hypothesis zoo.

## 8. Policies

Per model (5), three arms already run on these episodes:

- `factorial:none:advisory:s0` — LLM, no decoder confidence
- `factorial:decoder_confidence:advisory:s0` — LLM, confidence in the prompt
- `factorial:decoder_confidence:enforced:s0` — interface enforces the gate

Plus two deterministic references:

- `natural_confidence_gate_canonical` — threshold gate
- `natural_confidence_gate_lexical` — lexical resolver + gate

**The headline contrast is agent judgement versus interface enforcement**, which
is the paper's thesis. The lexical resolver is a secondary reference only: the
supplement already concedes it is given the nine canonical command strings while
the LLM is not, and no translational claim may rest on an information asymmetry.

## 9. Inference

- Participant-cluster bootstrap over the 46 participants, 2,000 resamples,
  seed 20260901.
- **One joint participant draw per replicate, applied to every policy**, so
  contrasts stay paired. Resampling policies independently is the error the
  supplement already warns about at `nag.paired_bootstrap`.
- Within a replicate, recompute the closed-form distribution on the resampled
  donor pool.
- A resampled command pool smaller than 1 donor is skipped for that replicate
  and the frequency is reported.
- 95% percentile intervals. Report differences with intervals; do not open a new
  multiplicity family.

**Required sentence in the manuscript:** the number of replayed trajectories
does not increase the neural sample size, which remains 200 episodes from 46
participants.

## 10. Limitations that must be stated

The replay models the mechanical consequence of repeated BCI attempts. It does
not model user frustration, adaptation, learning, fatigue, or willingness to
retry. The agent does not perceive the sandbox or its own prior failures. It is
not prospective validation.

## 11. Stopping rule

This is the final analysis. Nothing is added after it. Anything further is
result-chasing, not validation.
