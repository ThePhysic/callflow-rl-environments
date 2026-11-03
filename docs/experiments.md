# Reproducible learning protocol

CallFlow trains a compact actor/critic with clipped PPO and generalized advantage
estimation on 64 public numeric features and 20 bounded decisions. Two 64-unit
tanh layers feed categorical action logits and a scalar value head. This trains
network weights; it does not train unrestricted English or a language model.
The algorithm follows the clipped surrogate objective in
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) and
the advantage estimator in [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
This repository implements its own small trainer; those algorithmic ideas are
credited to their original authors.

The random and PPO policies share `policies/decisions.py` public workflow
scaffolding. Missing-argument writes are unavailable, END requires an observed
report/refusal, and HANDOFF requires public policy or availability evidence.
The mask is a pure function of the observation, tested against hidden goal
changes. Clarification, repeated queries, slot choices, recovery and efficiency
remain policy decisions. Native agents can still submit every documented action,
including premature termination and unauthorized writes; adversarial tests use
that unrestricted interface. Results for this constrained policy class must not
be presented as free-form conversational or speech performance.

`configs/ppo.yaml` registers seeds 11, 29 and 47; 32,768 interactions per baseline
seed; and development evaluations at 0, 8,192, 16,384 and 32,768 interactions.
The final fixed-budget checkpoint is used regardless of its development score.
Neither test nor transfer scenarios supply gradient updates, early stopping or
checkpoint selection. Fresh training episodes sample only training-owned
archetypes with independent generated parameter seeds. Test definitions are
visible to environment developers and were used for simulator/heuristic contract
testing. This is an authored held-out-family evaluation, not an externally blind
benchmark; the public workflow scaffold is part of the designed task interface.

The sparse-reward comparison and no-fault-training ablation use seed 11 with the
same budget and architecture. These one-seed comparisons are exploratory. The
ablation removes all archetypes tagged `fault` from the training generator while
retaining the same fixed evaluations. Transfer evaluates previously unseen
interleaved calendars and shared capacity. The held-out test set reserves user
corrections and selected fault/time-zone behavior classes.

The trainer bootstraps the value of the final observation on truncation and
cuts advantage recurrence at both termination and truncation. It never
bootstraps through a genuine terminal outcome or across an episode reset.
Checkpoint files contain model dimensions, weights, seed, interaction count,
variant and exact configuration. JSON reports record dependency versions,
manifest hashes, selected training archetypes and evaluation seeds. The full
experiment plan additionally records the Git revision and content fingerprint
of all Python source, covering uncommitted source as well as committed code.

Reproduce on CPU after installing locked dependencies:

```powershell
uv run python -c "from callflow_rl.evaluation.experiments import run_experiments; run_experiments(output_dir='artifacts/local/retrained')"
```

This writes a new `registered_plan.json` in the selected output directory before training and saves
per-run checkpoints, development learning curves and final train/dev/test/transfer
evaluations. `summary.json`, `RESULTS.md`, `learning_curve.png` and
`baseline_comparison.png` are generated from actual run outputs.
This command trains the current environment and can produce different curves
after simulator/verifier corrections. To reproduce the historical registered
experiment exactly, use an isolated checkout of commit `837af56`, its committed
`uv.lock`, the recorded Python/dependency versions, and the original
`configs/ppo.yaml`. The source-integrity artifact verifies that commit's source
content against the registered training hash. Keep new runs in a separate output
directory rather than overwriting the historical evidence.

Every rate retains numerator and denominator. Wilson intervals describe scenario
variation within the fixed benchmark. PPO aggregates bootstrap the three
independent training-seed success rates; three seeds provide coarse descriptive
uncertainty. Do not pool their episodes and treat those as independent training
runs. The clean/fault chart compares distinct authored archetypes, so its gap is
not a causal estimate of injecting a fault into an otherwise identical episode.

Task success requires verified appointment completion. Appropriate refusal and
handoff are separate metrics, combined only in the explicitly named appropriate
resolution metric. Timeout recovery uses episodes actually exposed to a
post-commit timeout; scheduled and exposed counts are both reported to reveal
policies that never reach the write. Unsafe effects, false claims, duplicate
writes, truncations, simulator truth-model consistency checks, tool use and
latency appear in each evaluation JSON. Failure taxonomy retains the verifier's
violations and unresolved termination reasons. Replay and four-worker throughput
are measured separately by the rollout release gate.

Measured results and limitations are in
[the generated report](../artifacts/experiments/RESULTS.md). Claims of learning
must follow those measurements, including seeds that fail to generalize or learn.

## Observed learning and error analysis

The following numbers and diagnostics describe the original training build
`837af56` and its historical evaluations. They remain preserved even when later
simulator/verifier corrections change the same checkpoints' current outcomes.
The latest corrected evaluation is identified by environment and verifier
versions in `artifacts/experiments/evaluation_release.json`; its current results
are in `artifacts/experiments/RESULTS.md`.

The registered shaped-reward runs are unstable. Seed 11 improves development task
success from 0/80 initially to 20/80 at 32,768 interactions; seeds 29 and 47 finish
at 0/80 task success. Seed 29 learns an appropriate handoff on 20/80 development
episodes, which must not be reported as appointment completion. Seed 11 completes
50/200 held-out tasks and 40/40 transfer tasks; the other two seeds complete none.
The three-seed held-out mean is 8.33%, with a coarse seed-bootstrap 95% interval of
0–25%, compared with the random policy's 7/200 (3.5%). This does not establish a
reliable improvement across training seeds, and transfer success is not robust.

[Recorded failure diagnostics](../artifacts/experiments/failure_diagnostics.json)
retain checkpoint hashes, scenario IDs, evaluation seeds, actions and final
verdicts. On a development request with a missing exact time window, seed 11 makes
19 identity lookups and one availability query before exhausting the tool budget.
Seed 29 repeatedly offers slots during a cancellation task, and seed 47 alternates
availability queries and offers without completing the booking. These failures
show incomplete use of task/observation features and repeated-action loops, even
though the backend rejects unsafe changes. Zero unsafe side effects in these
runs is evidence for simulator guards and constrained actions; it does not prove
that the learned policies independently acquired safe behavior.

The heuristic resolves all authored evaluation cases, while compact PPO learns a
narrow subset under this budget. Further work should use development-only
diagnostics to study exploration, shaping magnitude and redundant public choices,
then register a new plan before running it. These initial negative and partial
results remain part of the project evidence.

The one-seed sparse comparison finishes at 40/80 development tasks and 50/200
held-out tasks, compared with 20/80 and 50/200 for the shaped run with the same
seed. This does not establish a shaping benefit. The no-fault-training ablation
reaches 40/80 development tasks at 16,384 interactions but collapses to 0/80 at the
registered final budget, also finishing at 0/200 held-out tasks. The intermediate
checkpoint was not substituted for the final checkpoint. These observations
demonstrate non-monotonic learning and instability, not a reliable causal effect
from removing tool-fault training.

The sparse variant overrides the common YAML `reward_mode` with `sparse` in the
registered trainer; its `variant` field identifies that override even though the
shared configuration dictionary remains unchanged. Release provenance records
the actual training reward mode explicitly. Guard/metadata fixes after training
retain the original training build and all original evaluation JSON files, then
refresh evaluations with the same checkpoints through
`examples/refresh_evaluations.py`. The refresh compares per-scenario outcomes and
records both source versions and checkpoint hashes in `evaluation_release.json`.
Archive/evaluation text checksums normalize CRLF to LF so Git's text-line-ending
conversion does not invalidate them on a fresh clone. Binary checkpoint checksums
use the original bytes; the archive manifest and release provenance declare that
hashing scheme.

## Corrected release evaluation

The first Astra review identified missing reservation-to-consent bindings,
incomplete verification of published policy, and cancellation users accepting
irrelevant slot offers. Environment 0.1.1 and verifier 1.1 correct those behaviors.
The original trainer, public action mask, saved weights, training plan and curves
remain unchanged. Corrected simulator responses can change numeric observations,
subsequent actions, resource use and verdicts for an unchanged checkpoint. Such
changes are evaluation corrections and must not be described as additional RL
learning.
Subsequent adversarial checks also required historical consent revocation,
immutable business-data anchors, capacity validation after every reconstructed
write, and a final report that matches the final reservation row and follows its
last write. Those fixes strengthen verification of intermediate effects and the
final acknowledgement; their exact release source is recorded with evaluations.

`examples/refresh_evaluations.py` archives the preceding release, stages fresh
evaluations and reports per-scenario differences, including episode returns.
A `--workers 3` refresh assigns each policy/split pair to an isolated process,
with a fresh policy and one Torch thread. Suite order and the per-suite random
policy seed are preserved; workers write distinct staged reports and the parent
alone archives, compares and publishes. `--parallel-smoke --workers 3` checks
serial/process parity on booking, cancellation, rescheduling and timeout cases,
including final state hashes and all episode fields except wall-clock latency.
Report latency and throughput depend on the recorded concurrent worker count and
are not interpreted as a speedup against the historical serial evaluations.
A strict run leaves published reports and their summary unchanged when differences
exist. After inspecting those differences, the explicit command
`uv run python examples/refresh_evaluations.py --reuse-candidate --accept-reviewed-outcome-changes`
publishes the checksummed candidate that was reviewed. The original 28 reports
remain under `original_evaluations/`; prior release reports and presentation are
retained under `evaluation_history/`. Checkpoint and source hashes, verifier
versions, every compared field and both original/current metrics are recorded in
the release provenance. The earlier return-parity attestation refers to release
`6b69d86`; it does not assert parity for subsequent corrected releases.

The registered random baseline uses one local policy RNG stream for each fixed
suite, while environments have isolated per-episode RNGs. Changed cancellation
responses can change that policy's action choices and advance its random stream
differently, affecting later episodes in the same suite. Reproducing that baseline
therefore uses the recorded policy seed and frozen suite order. Differences in
random-policy scores across simulator versions are not evidence of learning or
a causal benefit from the verifier correction.

The current `c80e598` evaluation records 1,092 compared-field differences over
318 episodes, including changed returns for all 318. Random held-out success is
5/200 (2.5%, Wilson 95% interval 1.07–5.72%), compared with the historical 7/200.
Its familiar/transfer counts are now 3/160 and 1/40. PPO task-success counts remain
unchanged for all five runs; the shaped three-seed held-out mean remains 8.33%
with a coarse 0–25% seed-bootstrap interval. The heuristic still resolves all
200 held-out requests appropriately (175 completions and 25 refusals).

The PPO differences affect seeds 11 and 47 on the 61 cancellation instances in
each run. Correct cancellation responses change their loop behavior and budget
use, while task success remains false. For example, seed 11 on
`cancel_direct:101700` changes from a 21-step/20-tool-call tool-limit failure to a
40-step/19-tool-call turn-limit failure. Across all compared episodes, compliance,
safety, false claims, duplicate effects and violation labels do not change.
These are explicitly accepted evaluation corrections, with outcome/return parity
marked false; the historical curves and original training evidence are retained.
