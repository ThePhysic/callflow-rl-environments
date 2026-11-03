# Release gates and evidence

The original brief is a design source. This table maps its core gates to actual
implementation and executable evidence; planned features are not implied to exist.

| Gate | Implementation and evidence |
| --- | --- |
| Native and Gymnasium contract | Typed dialogue/tool validation; `CallFlowGymEnv`; environment checker; native/Gym transition equality over multiple seeds in `tests/test_core.py` |
| Booking, rescheduling, cancellation | Session authentication, explicit user approval, optimistic slot/reservation versions, atomic reschedule, idempotent writes; positive and negative core tests |
| Core invariants | Capacity, cross-customer isolation, duplicate retry prevention, consent revocation and finite timestamp behavior in core/adversarial/stateful tests |
| Determinism | Isolated RNGs, episode worlds, full snapshots, all-state fingerprints, seeded action-sequence replay and worker/serial comparison tests |
| Independent verifier | Reconstructed database/audit chain; wrong customer/slot, missing consent, duplicate writes, false claims and partial completion negatives |
| Reward robustness | Nine executable shortcut probes; bounded one-time milestone shaping; naive comparator isolated from production reward modes |
| Scenario integrity | 34 authored archetypes; versioned frozen manifests for 160/80/200/40 instances; split and manifest hash checks |
| Rollouts | 200 fixed-test episodes, four worker processes, bounded infrastructure retries, JSONL/Parquet traces and trace viewer; recorded benchmark |
| Replay and branches | Five saved traces exactly re-executed; checksum/transition corruption tests; saved-snapshot branch changes the outcome without mutating the original |
| Actual learning | Three independently seeded PPO runs; fixed-budget checkpoints; dev learning curves; random/heuristic comparisons and full evaluations |
| Experimental comparisons | Sparse/shaped comparison, clean/fault slices, train/held-out gap, no-fault ablation, unseen-calendar transfer and saved failure taxonomy |
| Measurement | Denominators, Wilson scenario intervals, descriptive training-seed bootstrap intervals, throughput and p50/p95 latency; explicit single-seed comparison limitations |
| Reproducibility and quality | Locked uv dependencies, type/lint checks, CI configuration, unit/integration/property tests, packaged fixture smoke check, no paid services/GPU needed |
| Presentation | README diagram and recorded-trace figure, five representative traces, reward audit, learning plots, architecture/scenario/reward/experiment documentation |

Reproduction commands and their captured outputs are in
[`artifacts/validation/quality.json`](../artifacts/validation/quality.json).
Packaging checks are in
[`artifacts/validation/package_check.json`](../artifacts/validation/package_check.json).
The measured four-worker run is in
[`artifacts/validation/parallel/benchmark.json`](../artifacts/validation/parallel/benchmark.json).
Actual experiment outputs and their source/configuration provenance are under
[`artifacts/experiments/`](../artifacts/experiments/).

The scenario catalog was authored and reviewed by implementation agents and
checked automatically. It has not been independently reviewed by human domain
experts. Evaluation families were excluded from PPO updates and checkpoint
selection, but were visible to developers during environment contract testing.
Neither the numeric policy nor the simulator establishes unrestricted language,
speech, real customer behavior or real-calendar production readiness.

Actual audio, external LLM user simulation, distributed training and additional
business domains remain stretch work. Recorded-output playback is implemented;
this release does not contain an external stochastic user service.
