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

Release evaluations use the original checkpoints after adversarial review corrected verifier bindings, policy enforcement and cancellation dialogue. Training remains on build 837af56; no gradients were rerun. Original and preceding evaluations are archived. There are 1092 per-scenario field differences against the original evaluations; differences include returns and resource use as well as verdicts. [evaluation_release.json](evaluation_release.json) records every difference, source version, verifier version and checkpoint hash.

Current corrected evaluation (environment 0.1.1, verifier 1.1.0, source c80e598) uses the original weights. Random held-out success changes from 7/200 (3.5%) to 5/200 (2.5%; Wilson 95% interval 1.07 to 5.72%). Its familiar and transfer counts change from 6/160 and 2/40 to 3/160 and 1/40. The final-report protocol and cancellation responses change trajectories; the random policy also carries a deterministic RNG stream across each suite, so later draws can shift. These changes are not additional learning.

All five PPO runs and the heuristic retain their task-success counts. The three shaped PPO held-out rates remain 25%, 0%, 0% (mean 8.33%, coarse seed-bootstrap interval 0 to 25%). The heuristic still completes 175/200 held-out appointments and correctly refuses the remaining 25. Cross-seed PPO improvement remains unestablished.

The corrected release records 1,092 compared-field differences across 318 episodes, including 318 changed episode returns. PPO seeds 11 and 47 change resource use only on cancellation families: for example, seed 11 on cancel_direct:101700 changes from 21 steps/20 tool calls (tool limit) to 40 steps/19 tool calls (turn limit). Their task outcomes stay unsuccessful. No comparison changes policy compliance, safety, false-claim, duplicate-effect or violation-label outcomes. Original training curves and the original/prior evaluation archives are unchanged; the published provenance explicitly marks outcome and return parity false.
