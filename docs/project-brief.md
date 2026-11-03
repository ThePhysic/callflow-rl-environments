A standalone, reproducible reinforcement-learning environment for conversational tool-using agents. Its central task is to resolve a customer's request for an appointment, interact with a simulated scheduling service, recover from tool and conversation failures, and reach a **verifiably correct final database state**. It is designed as a portfolio project for Besimple AI's RL Environment Engineer role, with environment engineering as the primary deliverable and model training as evidence that the environment supplies a useful learning signal.

**Core decision:** Build one scheduling domain deeply, as a separate repository. Reuse engineering ideas from the existing computer-use automation and adaptive training agent projects, but do not require either repository to run. Start text-first with a voice-like event interface; add actual audio only after the environment and reward verification are strong.

**Working name:** CallFlow RL. **Suggested repository:** `callflow-rl-environments`. **Status:** design specification, not an implementation or claimed experiment result.

# Product and research questions

The environment should answer:

1. Can a policy ask for information it lacks instead of assuming it?

2. Can it use APIs to find available slots and commit a correct booking?

3. Can it recover from a timeout, stale availability, contradictory user updates, or an interrupted turn without duplicating the booking?

4. Do reward and verifier rules reflect the actual user goal, rather than just a polite response or a successful HTTP code?

5. Does learning improve across unseen task variants, or merely exploit the simulator's regularities?

**Core deliverables:** runnable environment; task schema and seeded scenario generator; user and tool simulators; independent goal-state verifier and reward audit; rollout, trace, and replay system; random, scripted, and learned baselines; experimental results and failure analysis.

**Non-goals for the initial version:** production voice recognition or synthesis, full-duplex real-time audio infrastructure, arbitrary open-domain conversation, a large fine-tuned language model, a multi-agent research platform, or deploying to a real calendar/CRM.

# Domain and task families

**Primary domain:** synthetic service-appointment scheduling. A simulated customer communicates with an agent representing a fictional business with locations, services, calendars, policies, and booking APIs. No real personal data or real-world bookings are required.

**Initial task family — book an appointment:**

- Resolve customer identity using permitted synthetic account fields.

- Clarify service type, location, preferred date/time window, constraints, and whether flexibility is acceptable.

- Query availability; offer admissible alternatives; obtain explicit confirmation.

- Commit an appointment and report only the booking that actually exists.

**Second task family — reschedule:** identify an existing appointment, determine the new acceptable slot, and atomically change it without losing or duplicating the original.

**Third task family — cancel:** identify the correct reservation, verify authorization and intent, cancel exactly once, and communicate the true resulting status.

Implement booking first and add rescheduling/cancellation when booking is reliable. All three should reuse the same calendar state, event log, action schema, and verifier interface.

**Scenario classes:** straightforward request; missing information; ambiguous date/time; multiple valid slots; no availability; customer changes mind; customer supplies conflicting constraints; calendar entry goes stale; API timeout before commit; API timeout after a successful commit; duplicate request/retry; interrupted utterance; user declines alternative; out-of-policy request; handoff required; daylight-saving/time-zone boundary as an advanced case.

An illustrative episode: user asks for a tire-rotation appointment on Tuesday afternoon; agent clarifies the location and exact window; the preferred slot is unavailable; user agrees to another slot; the commit succeeds but its response is lost; agent checks booking state and avoids a duplicate before confirming. Successful text alone earns no completion credit.

# Environment contract and task interface

Represent the problem as a partially observable sequential decision process. The policy receives only allowed observations; an evaluator retains the true task goal and backend state.

- **Hidden task goal:** target customer, authorized service, location, slot constraints, required approvals, disallowed side effects, goal-state predicate.

- **World state:** synthetic customer and account records, business policy, available slots, reservations, slot versions, tool fault state, event queue, clock.

- **Dialogue/user state:** what the user has already said, what they are willing to accept, pending questions, interruptions, consent/confirmation status, simulator persona, hidden preferences.

- **Policy-visible observation:** latest user message/event; selected conversation history; published policy/tool schemas; tool responses; current step/time budget; no answer key or private evaluator labels.

- **Action:** `ASK_CLARIFICATION`, `OFFER_SLOT`, `CONFIRM_INTENT`, `CALL_TOOL` (tool name and typed arguments), `MESSAGE_USER`, `HANDOFF`, `END`. Reject malformed or unauthorized actions explicitly.

- **Transition:** validate action, apply a business-tool state change if permitted, advance simulated time, generate user/tool events, update observable history, then evaluate reward and stop conditions.

- **Termination:** goal met and task appropriately ended; cancellation/refusal completed; irrecoverable violation; explicit handoff/abandonment under policy.

- **Truncation:** max policy turns, elapsed virtual time, tool-call limit, or resource quota. Report `terminated` separately from `truncated`.

Expose a native structured-event interface for LLM/tool-call policies and a Gymnasium-compatible `reset(seed, options)` / `step(action)` interface. Return `(observation, reward, terminated, truncated, info)` from each step. `info` may contain debug metadata in training or test harnesses; a policy adapter must never consume hidden goal labels or future outcomes.

**Important action-space choice:** The production-facing interface uses typed variable-argument tool calls. A separate constrained Gymnasium training wrapper maps a bounded list of valid dialogue/tool decisions and slot candidates to a finite action space, with masked or explicit invalid-action handling. This lets a compact PPO policy train without pretending that vanilla PPO directly operates over unrestricted English or arbitrary JSON. Keep the two interfaces behaviorally consistent with contract tests.

# Synthetic user and tool simulators

**User simulator v1:** deterministic, scenario-conditioned state machine with utterance paraphrases and controlled variants. Inputs include the hidden goal, persona, clarification tolerance, preferences, and scripted event triggers. It must remember accepted alternatives and corrections, not contradict its own facts unless the scenario explicitly contains a change of mind. It may refuse, interrupt, request repetition, or end a call. Seed every random choice through isolated environment RNGs.

**User simulator v2:** optional LLM-based paraphrase/behavior generator. Validate generated turns against a structured truth model, record actual outputs for replay, and report nondeterminism honestly. Do not make an LLM judge the ground-truth success state.

**Tool simulator:** an in-memory transactionally consistent service API backed by synthetic data. Required methods can include `lookup_customer`, `list_services`, `get_availability`, `get_booking`, `create_booking`, `reschedule_booking`, `cancel_booking`. Every write has version/availability checks, authorization/consent checks where applicable, and idempotency semantics.

**Fault injector:** configurable latency, rejected arguments, stale reads, booking conflicts, network timeouts before commit, and ambiguous timeouts after commit. A post-commit timeout is particularly important: a blind retry must not create an extra reservation.

**Isolation:** each episode gets an independent logical datastore and fault schedule. Parallel episodes must never share mutable customer, calendar, reward, or RNG state.

# Rewards and independent verification

**Ground truth:** an independent verifier compares the final database state and audit log with the episode's private goal specification. It should not infer success from agent text, a tool's reported success code, or a mutable policy-visible flag.

**Success predicate for booking:** there is exactly one active booking with the correct customer, service, allowed location, accepted time, and required confirmation; the history contains no forbidden side effect. If the correct booking exists but an unauthorized second booking was created, the episode must fail.

**Separate outcomes:** `task_success`, `policy_compliance`, `safety_violation`, `duplicate_side_effect`, `termination_reason`, `invalid_action_count`, and `verifier_version`. Do not collapse these into a single opaque model-generated score.

**Reward design, proposed rather than measured:**

- Highest positive terminal signal for independently verified completion with policy compliance.

- Strong negative terminal signals for unauthorized bookings, wrong customer, wrong time/location, duplicate side effects, or falsely claiming a booking was made.

- Small step/time/tool-call cost to encourage efficiency.

- Optional bounded progress shaping based on newly satisfied verified constraints. Prefer potential-based shaping or one-time monotone milestones so repeated clarifications and repeated queries cannot farm rewards.

- A justified handoff/refusal should score appropriately on scenarios where booking is impossible or disallowed; it should not receive the same reward as a completed booking.

**Reward-hacking tests:** ending the episode with an unsupported success claim; booking an unconfirmed slot; repeatedly asking a question for intermediate credit; querying until a simulator reveals hidden hints; creating an extra reservation after a lost response; tampering with goal metadata; exploiting date/time normalization; deliberately exhausting the action budget to avoid failure accounting; or calling an admin/debug endpoint that the policy should not have.

Implement a deliberately vulnerable reward variant (e.g. success awarded from agent assertion or tool success response), demonstrate a shortcut policy exploiting it, then show the independent verifier refusing that shortcut. Keep the vulnerable variant in a clearly marked educational audit module, never as the default environment.

# Scenario data and evaluation integrity

Define a versioned `ScenarioSpec` with scenario ID, template family, synthetic starting world, private goal, public task instruction, user persona, constraints, fault schedule, termination limits, and split. Store only synthetic or licensed/consented, de-identified demonstrations; never include raw personal calls, credentials, actual patient/customer data, or copyrighted recordings without permissions.

**Initial data target:** about 25–40 hand-reviewed scenario archetypes with seeded parameterizations, plus a fixed evaluation suite of roughly 150–250 scenario instances. These are design targets, not current counts.

Use independent train/development/test partitions by **scenario archetype and behavior class**, not just by different RNG seeds. Keep some failure families and paraphrase patterns fully held out. Freeze test definitions and hash the manifests. Maintain explicit challenge slices for tool faults, ambiguous requests, policy constraints, and interruptions.

For human-demonstration conversion (later milestone): define a manual annotation schema for intent, known facts, decision points, valid alternatives, tool effects, success predicates, and counterfactual continuations. Begin with fully synthetic authored dialogues; add real interactions only with lawful rights, consent, and redaction.

# Rollout infrastructure and replay

Build an episode runner that accepts any `Policy` adapter and yields structured `EpisodeTrace` records. A trace should contain: environment/build version, scenario manifest hash, policy/checkpoint version, seed and RNG sub-seeds, initial-state fingerprint, timestamped observations/actions/tool calls/results, state change hashes, rewards and verifier decisions, termination reason, and resource consumption.

**Replay modes:**

1. **Exact deterministic replay:** re-execute from seed and saved event schedule; assert matching state hashes, actions supplied, rewards, and final verdict.

2. **Recorded external replay:** use recorded user/LLM/tool outputs when underlying services are stochastic.

3. **Counterfactual branch:** fork a saved pre-action snapshot, alter one policy action or injected fault, and compare the final verified state.

Run batches locally in parallel with per-episode isolation. Include bounded retries for worker failures without conflating infrastructure retries and policy actions. Export JSONL/Parquet traces plus a lightweight HTML or command-line trace viewer that explains why each reward or failure occurred. Full Ray/RLlib distribution should be an optional extension, not a core dependency.

# Training policies and experiments

**Baselines in order:**

1. Random valid-action policy (sanity lower bound).

2. Scripted/heuristic agent that clarifies known required slots, calls APIs, and confirms the result.

3. Small PyTorch policy with a bounded structured observation/action wrapper, trained with PPO. DQN may be an additional comparison for the purely discrete wrapper, not an obligation.

4. Optional LLM tool-calling agent evaluated on the same hidden-goal verifier; distinguish prompt/model changes from RL weight updates.

**Experiments:**

- A learning curve over environment interactions, with evaluation at pre-registered checkpoints.

- At least 3 independent training seeds for the compact PPO baseline, with fixed evaluation suites and uncertainty intervals; add more seeds if feasible.

- Comparisons: random vs heuristic vs learned; sparse reward vs shaped reward; clean vs tool-fault scenarios; training archetypes vs held-out archetypes.

- One ablation removing a useful observation, reward component, or fault-recovery training family.

- One reward-hacking demonstration using the intentionally naive scorer and the corrected independent verifier.

- One transfer test on a genuinely unseen scenario family, such as a changed booking policy or revised availability pattern.

**Metrics:** true task-success rate, policy-compliant success, false-success-claim rate, unsafe side-effect rate, duplicate-write rate, recovery rate after ambiguous timeouts, average steps/tool calls, truncation rate, held-out success gap, simulator inconsistency rate, replay-match rate, and episode throughput/latency. Report denominators, run seeds, test-set version, and confidence intervals. Never claim improvements until the experiments have actually produced them.

A trained PPO policy does not need to outperform a strong hand-written oracle. The evidence should show credible learning relative to a weak baseline, accurate failure measurement, and honest generalization analysis.

# Engineering architecture and repository

Recommended stack: Python 3.12+, `uv` for dependency and environment management, typed dataclasses or Pydantic for contracts, Gymnasium for the RL-facing adapter, NumPy and PyTorch for training, pytest and Hypothesis for stateful/property tests, JSONL/Parquet for traces, and a lightweight plotting/report tool. Stable-Baselines3 is reasonable only for the supported constrained action wrapper; evaluate compatibility with its checker before choosing it. Optional MLflow or TensorBoard for experiment tracking; optional Ray later.

Suggested modules:

```text
src/callflow_rl/
  contracts/       scenario, action, observation, event, trace schemas
  core/            episode state, transition engine, clocks, RNG
  simulators/      user, scheduling tools, fault injection
  envs/            native event environment, Gymnasium adapter
  verifiers/       state-goal verifier, policy checks, reward rules
  scenarios/       generators, immutable fixtures, split manifests
  policies/        random, heuristic, PPO adapter, optional LLM adapter
  rollouts/        episode runner, parallel worker, replay, branching
  evaluation/      metrics, statistics, reports, failure taxonomy
configs/           env, policy, train/eval configuration
tests/             unit, stateful, contract, adversarial, replay
examples/          five representative traces, demo scripts
docs/              architecture, scenario semantics, reward spec, experiments
```

CLI entry points (design targets):

- `callflow validate --suite core`

- `callflow rollout --policy heuristic --split dev --episodes 200 --seed 42`

- `callflow replay --trace examples/timeout_after_commit.jsonl`

- `callflow audit-rewards --suite adversarial`

- `callflow train --algo ppo --config configs/ppo.yaml`

- `callflow evaluate --checkpoint artifacts/ppo_run_01 --split test`

Provide a reproducible `uv sync`, `pytest`, quickstart rollout, and a short local demo that does not need paid API keys, a GPU, or network connectivity.

# Reliability and acceptance criteria

Treat these as **release gates to verify**, not pre-existing achievements.

- **Contract:** Gymnasium `reset/step` passes the environment checker; valid structured actions have documented outcomes; invalid actions fail deterministically and safely; correct `terminated` / `truncated` semantics.

- **Core invariants:** booking capacity never exceeded; no cross-customer state modification; no duplicate active booking on an idempotent retry; no booking without required approval; time-zone and date behavior is explicit.

- **Determinism:** identical build, scenario, policy actions, and seed reproduce the same final state/reward; any LLM-involved path supports recorded-output replay. Test many seeds and event sequences, not just one example.

- **Verifier:** tests include positive tasks and negative examples covering wrong customer, wrong slot, consent absence, double write, false claim, and partial completion. Verifier logic cannot depend on the agent's declared success.

- **Reward robustness:** every enumerated adversarial shortcut is tested; the intentionally vulnerable comparator is separate from the production score.

- **Rollouts:** 200+ fixed-suite episodes can run unattended, including at least 4 isolated concurrent workers; traces are readable and failed episodes can be replayed.

- **Experiments:** random, heuristic, and a trained compact policy have recorded reproducible evaluations; at least one learning curve, one OOD test, one ablation, and one error analysis.

- **Quality:** type checks, lint, CI, unit/integration/property tests, dependency pinning, privacy-safe fixtures, stable public API examples, and no secrets in repository or logs.

A useful performance target for the deterministic, no-LLM simulator is high enough throughput to run hundreds of episodes conveniently on a laptop. Measure and report actual episodes/sec and p50/p95 episode latency rather than claiming a numerical speedup without a benchmark.

# Milestones and scope control

| Phase | What to build | Exit evidence | Approximate effort |
| - | - | - | - |
| R0 Specification | Formal scenario schema, policy-visible vs hidden fields, example traces, 10 golden cases, reward invariants | Reviewed design and fixtures | 8–12 h |
| R1 Working environment | Booking domain, deterministic user/tool simulations, Gymnasium adapter, reset/step, seeded scenario generator | End-to-end traces, interface checks | 20–25 h |
| R2 Rewards and hard cases | Independent verifier, fault injection, policy rules, reward-hacking audit, property tests | Correct negative and adversarial results | 20–25 h |
| R3 Rollouts and reproducibility | Runner, parallel isolation, trace persistence, deterministic replay, scenario splits | Replayable batch evaluation | 15–20 h |
| R4 Learning and evidence | Random/heuristic/PPO, training runs, held-out evaluations, ablation, failure analysis | Plots and reproducible experiments | 20–30 h |
| R5 Presentation | README, architecture docs, example episodes, compact demo, CI, results table | Public portfolio-ready repository | 8–12 h |
| R6 Optional voice extension | Event-level ASR ambiguity, barge-in, interruptions; then real audio if justified | Measured behavior on voice-like tasks | Separate stretch phase |

**Core estimate:** approximately 90–125 hours, varying with test depth and experiment stability, consistent with an approximately 4–8 week effort at 15–20 hours per week. These are planning estimates. Shipping R1–R3 with verifiable rewards and replay is already meaningful environment-engineering evidence; the strongest application story includes at least one actual learned-policy experiment from R4.

**Defer:** distributed training clusters, real-time duplex audio, multiple business domains, generative user simulation as the default oracle, large-scale fine-tuning, and elaborate frontend dashboards until the core gates pass.

# Voice focused extension

For voice-agent relevance, retain the same hidden task goal and database verifier while changing the observation/action timing layer.

**First add event-level voice phenomena without raw audio:** partial ASR hypotheses that later get corrected; interruptions/barge-in; overlapping speech timestamps; silence/timeout; user self-correction; agent response delay; a `WAIT` or `YIELD_TURN` action. Evaluate whether the policy commits too early, speaks over the user, loses corrections, or fails to recover. Implement an explicit simulated clock and event queue so event order is replayable.

**Later, if worthwhile:** plug in ASR/TTS or full-duplex streams, measure transcription error rates, latency, turn-taking, and interruptions, and persist original media references only where rights and privacy protections permit. Treat waveform/audio quality as a new modality; do not label text-event simulations as actual speech training.

The role states speech as a plus, but hands-on RL environments as the core requirement, so this is a secondary phase.

# Public presentation and interview evidence

The repository home page should answer, within a minute: what task is simulated, what a policy sees and does, why naive scoring fails, what the verifier checks, how to reproduce episodes, and what a learned policy improved or failed to improve.

**README sections:** overview and short GIF/trace screenshot; example hard scenario; environment interface; state/observation/action diagram; verifier and reward logic; reproducibility commands; baseline metrics with seed counts and uncertainty; failure examples; scope limits; architecture; CI/test badges and setup.

**Showcase artifacts:**

- One successful multi-turn episode, including questions, API calls, confirmation, state changes, and final check.

- One ambiguous post-commit timeout recovered safely via state inspection/idempotency.

- One reward exploit against a naive scorer and a blocked exploit against the independent verifier.

- One replay showing identical final state and score.

- One plot comparing random/heuristic/learned policies across clean vs held-out fault scenarios.

**Proposed resume framing after verified completion:** “Built a Gymnasium-compatible conversational RL environment with seeded customer and scheduling simulators, state-based reward verification, fault injection, deterministic replay, and parallel rollouts; evaluated scripted and PPO policies on held-out task scenarios.” Insert actual measured results only once experiments exist.

**Strong interview explanation:** “I defined task state and partial observations; the agent could ask, call tools, confirm or hand off; the verifier checked database state independently of conversation claims. I then injected ambiguous timeouts, replayed failures, tested reward-hacking attempts, and measured what trained policies generalized to.”

# Related projects and positioning

- **Computer-Use Automation:** reuse experience designing guarded tool actions, recoverability, bounded execution, and trace replay. CallFlow RL's distinct contribution is a resettable training environment with hidden goals, reward signals, randomized scenarios, and learned-policy evaluations.

- **Adaptive Training Agent:** reuse experience with deterministic contracts, simulation, durability, experiments, and evidence-backed evaluation. Do not fold CallFlow into that project's product/domain logic.

- **Visual Inspection and Deep Learning:** remain separate; RL does not need to be added to the vision system.

- **LLM fine-tuning and serving:** complementary serving/experiment knowledge; not a substitute for an actual stateful RL task environment.

This should remain a standalone and open-source-friendly portfolio artifact. One deep, well-tested domain is a stronger match for the listed role than several shallow toy environments.

# Sources and technical precedents

- Gymnasium custom environment and `reset/step` contract: [https://gymnasium.farama.org/main/api/env/](https://gymnasium.farama.org/main/api/env/) and [https://gymnasium.farama.org/main/tutorials/environment_creation/](https://gymnasium.farama.org/main/tutorials/environment_creation/)

- Stable-Baselines3 custom environment compatibility: [https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)

- Ray RLlib environment runners (optional scaling): [https://docs.ray.io/en/latest/rllib/rllib-env.html](https://docs.ray.io/en/latest/rllib/rllib-env.html)

- Hypothesis stateful testing: [https://hypothesis.readthedocs.io/en/latest/stateful.html](https://hypothesis.readthedocs.io/en/latest/stateful.html)

- τ-bench, state-based tool-agent-user verification: [https://openreview.net/pdf?id=roNSXZpUDN](https://openreview.net/pdf?id=roNSXZpUDN)

- Sierra's τ³-bench voice extension: [https://sierra.ai/resources/research/tau-3-bench](https://sierra.ai/resources/research/tau-3-bench)

- AgentGym's interactive training and evaluation precedent: [https://aclanthology.org/2025.acl-long.1355/](https://aclanthology.org/2025.acl-long.1355/)

Use these as conceptual references, not as code to claim as original. The environment, scenario semantics, reward audit, and experimental results should be original and correctly attributed.
