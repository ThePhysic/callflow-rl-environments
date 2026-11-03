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
uv run python -m callflow_rl.evaluation.experiments
```

This writes `artifacts/experiments/registered_plan.json` before training and saves
per-run checkpoints, development learning curves and final train/dev/test/transfer
evaluations. `summary.json`, `RESULTS.md`, `learning_curve.png` and
`baseline_comparison.png` are generated from actual run outputs.

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
