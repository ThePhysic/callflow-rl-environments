# Measured experiment results

Every row uses the final fixed-budget checkpoint; no test-based selection. Success means the independent verifier accepted the completed database task. Appropriate handoffs/refusals are reported separately in JSON.

| Policy | Familiar task success | Held-out task success | Transfer task success | Held-out appropriate resolution |
| --- | ---: | ---: | ---: | ---: |
| random | 3/160 (1.9%) | 5/200 (2.5%) | 1/40 (2.5%) | 27/200 (13.5%) |
| heuristic | 136/160 (85.0%) | 175/200 (87.5%) | 40/40 (100.0%) | 200/200 (100.0%) |
| ppo_shaped_seed_11 | 64/160 (40.0%) | 50/200 (25.0%) | 40/40 (100.0%) | 50/200 (25.0%) |
| ppo_shaped_seed_29 | 0/160 (0.0%) | 0/200 (0.0%) | 0/40 (0.0%) | 0/200 (0.0%) |
| ppo_shaped_seed_47 | 0/160 (0.0%) | 0/200 (0.0%) | 0/40 (0.0%) | 0/200 (0.0%) |
| ppo_sparse_seed_11 | 72/160 (45.0%) | 50/200 (25.0%) | 40/40 (100.0%) | 50/200 (25.0%) |
| ppo_no_fault_seed_11 | 0/160 (0.0%) | 0/200 (0.0%) | 0/40 (0.0%) | 0/200 (0.0%) |

The three-seed PPO aggregate, confidence intervals, per-scenario outcomes, failures, safety and recovery denominators are in summary.json and each evaluation JSON. Sparse reward and no-fault-training comparisons each use one seed, so their differences are exploratory. The clean/fault chart compares distinct authored archetypes and does not isolate a causal fault effect.

Intervals over scenarios are Wilson intervals; intervals over PPO runs bootstrap three independent training-seed means and are coarse. These fixed benchmark results do not establish performance on real conversations or unrestricted natural language.

The learned policy chooses bounded structured decisions from public numeric features. It does not train a language model. All fixtures and utterances are synthetic.

Release evaluations use the original checkpoints after adversarial review corrected verifier bindings, policy enforcement and cancellation dialogue. Training remains on build `837af56`; no gradients were rerun. Original and preceding evaluations are archived. There are 1092 per-scenario field differences against the original evaluations; differences include returns and resource use as well as verdicts. [evaluation_release.json](evaluation_release.json) records every difference, source version, verifier version and checkpoint hash.

The current environment 0.1.2/verifier 1.2.0 evaluation uses source `4fd3e16` and adds zero new compared outcome, resource or return differences against environment 0.1.1's `c80e598` evaluation. The 1,092 differences against the original `837af56` evaluations remain retained historical corrections. All five checkpoint weights are unchanged. Reports contain fresh version and state metadata, and the five example traces have been regenerated and exactly replayed against the current source.
