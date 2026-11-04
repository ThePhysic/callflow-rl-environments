# GitHub presentation suggestions

These are prepared suggestions, not applied repository settings or a published
release. At the November 3, 2026 inspection, GitHub reported public visibility,
no topics, an empty homepage field and no published release. Access settings may
change independently of this document. Package **0.1.2** and verifier **1.2.0**
match the tracked [evaluation provenance](../artifacts/experiments/evaluation_release.json).

## Suggested manual actions

Each action below is optional and requires explicit user approval before an
agent applies it. Documentation revisions do not authorize these actions.

1. In the repository's **About** settings, use this description:
   "Headless Gymnasium appointment simulator with independent verification,
   deterministic replay, fault injection and measured structured PPO baselines."
2. Add these topics: `reinforcement-learning`, `gymnasium`, `ppo`,
   `agent-evaluation`, `simulators`, `reproducibility`, `python`.
3. Leave the homepage field empty while there is no hosted project site. The
   README already provides the local demo and evidence links; it should not be
   presented as a hosted interactive application.
4. If a release is wanted, first choose the reviewed `main` commit containing
   this documentation pass, confirm its checks pass and confirm package/evidence
   versions still match. Create a **new `v0.1.2` tag** at that exact commit and
   publish a GitHub release titled **CallFlow RL v0.1.2** using the draft below.
   Confirm the tag does not already exist; do not move or replace an existing tag.
5. Keep current visibility unchanged. A visibility change or hosted deployment
   would be a separate decision.

No tag creation/push, release publication, settings/topics edit, visibility
change, deployment or history rewrite is included in the documentation pass.

## Draft release notes: CallFlow RL v0.1.2

CallFlow RL is a headless Python/Gymnasium research environment for synthetic
appointment booking, rescheduling and cancellation. Policies act through typed
dialogue/tool decisions; an independent verifier checks the final database and
full audited write history. Python APIs and a CLI support episodes, training,
evaluation, trace inspection, exact replay and reward audits.

This version includes corrections for booking-ID allocation, duplicate
rescheduling, intermediate write validation, contradictory status reports,
seeded resets, time budgets, identity corrections, worker-process crash recovery
and trace integrity. The [follow-up audit](https://github.com/ThePhysic/callflow-rl-environments/blob/main/docs/postrelease-audit.md) describes the
counterexamples and regression coverage. Recorded release validation passed
**372 tests**, lint/type checks, five exact example replays and a 200-episode
four-worker run; these are measurements of the recorded build, not guarantees
about future changes. [Quality evidence](https://github.com/ThePhysic/callflow-rl-environments/blob/main/artifacts/validation/quality.json)
and [replay verdicts](https://github.com/ThePhysic/callflow-rl-environments/blob/main/examples/replay_verdicts.json) are retained in the repo.

The repository includes runnable synthetic examples, nine executable reward
shortcut probes, frozen train/development/test/transfer manifests, and actual
small PyTorch PPO checkpoints. PPO uses bounded structured decisions and numeric
public observations; it is not a trained language model. No hosted dashboard,
microphone pipeline, ASR/TTS, live scheduling integration or unconstrained LLM
policy ships with this version. Demos need no paid keys, GPU or real customer
data after dependency installation.

Held-out task success in the saved evaluation is **5/200 (2.5%)** for random,
**175/200 (87.5%)** for the heuristic, and **50/200, 0/200, 0/200** for shaped PPO
training seeds 11, 29 and 47. The heuristic's other 25 cases are justified
refusals, yielding **200/200 appropriate resolutions**. PPO's three-seed mean is
about **8.3%**; its variability does **not** establish stable learning, reliable
improvement over random or robust generalization. The authored evaluation suite
is held out from model training but visible to developers, not externally blind.
See the [measured results](https://github.com/ThePhysic/callflow-rl-environments/blob/main/artifacts/experiments/RESULTS.md).

Original PPO checkpoints and learning curves came from training build
`837af56`. Environment/verifier corrections reevaluated **the same saved
weights**, without further gradient training. The current source `4fd3e16`
evaluation adds zero new compared outcome/resource/return differences against
the preceding `0.1.1` evaluation; the earlier 1,092 historical field differences
remain disclosed. Prior reports, scenario manifests, checkpoints and hashes are
preserved in [evaluation provenance](https://github.com/ThePhysic/callflow-rl-environments/blob/main/artifacts/experiments/evaluation_release.json).

AI-assisted internal adversarial review is documented alongside reproducible
tests; it is not external certification. The project is MIT licensed.
