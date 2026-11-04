# Environment and contributor contracts

This guide describes the implementation boundaries and how to extend them.
Use [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the routine check commands,
[architecture](docs/architecture.md) for replay and trust boundaries, and
[release gates](docs/acceptance.md) for executable evidence. The
[original project brief](docs/project-brief.md) is a design reference; it also
contains ideas beyond the shipped headless Python environment.

## Public APIs and evaluator state

`callflow_rl.contracts.ScenarioSpec` is the evaluator's JSON-round-trippable
scenario input. `to_dict()` and `from_dict()` preserve its representation;
validation rejects unsupported schemas and invalid limits or fault schedules.
All fixtures and demonstrations must remain synthetic.

The native environment exposes these contracts:

- `CallFlowEnv(scenario=None, reward_mode="shaped")` in `envs/native.py` accepts
  `shaped` or `sparse` rewards.
- `reset(seed=None, options=None)` returns `(public_observation, info)`.
  `options["scenario"]` accepts a `ScenarioSpec` or its dictionary representation.
  Automatic scenarios regenerate from the requested reset seed; explicitly
  supplied scenarios remain fixed.
- `step(action)` returns `(observation, reward, terminated, truncated, info)`.
  Native actions are dictionaries validated against `contracts.ACTION_FIELDS`.
  For example, a tool call is
  `{"type": "CALL_TOOL", "tool": "get_booking", "args": {...}}`.
  Invalid decisions produce explicit safe errors and still consume resources.
- `snapshot()` returns full JSON-serializable evaluator state; `restore(snapshot)`
  restores it. `state_hash()` covers transition-relevant state, including RNGs,
  dialogue memory, queued events, faults and scenario-generation mode.
- `.scenario`, snapshots and `verdict()` are evaluator-only. `info` contains
  evaluator metadata, including rewards and verification details.

Policies implement `act(observation: dict) -> dict`. The runner supplies a deep
copy of the public observation, without the environment, private goal, future
events, backend rows or evaluator `info`. Debug traces contain private snapshots
and must never become policy observations or training features. This is an API
boundary for trusted Python adapters, not a sandbox for hostile Python code.

`envs/gym_env.py` provides `CallFlowGymEnv`, with the Gymnasium reset/step
contract, `.native`, and `.action_for(index)`. Its action space is `Discrete(20)`;
its observation is a 64-element `float32` encoding of public facts, including
features for up to three returned slot candidates.
The native interface supports more expressive variable-argument actions.
Encoding and action expansion use only public observations. Random and PPO
policies additionally share the public workflow mask in `policies/decisions.py`;
the native environment remains available for unrestricted adversarial tests.
The numeric PPO adapter does not learn unrestricted natural-language dialogue.

### Native API example

This loop keeps evaluator `info` separate from the policy's observation:

```python
from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import generate

env = CallFlowEnv(generate("book_timeout_after", seed=42))
observation, info = env.reset(seed=42)
policy = HeuristicPolicy()
while True:
    action = policy.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(info["verdict"])
```

## Responsibilities and extension points

Source paths below are relative to `src/callflow_rl/`.

| Component | Responsibility and where to extend it |
| --- | --- |
| `contracts/` and `core/` | Versioned scenario/action validation, canonical state fingerprints and deterministic RNG helpers. Update contracts when adding fields or fault kinds. |
| `scenarios/catalog.py` and `generator.py` | Archetype split/behavior definitions and seeded synthetic worlds, private goals, user facts and schedules. Add an archetype here and document its obligations in `scenarios/fixtures/golden_cases.json` and `docs/scenarios.md`. |
| `simulators/user.py` | User truth model, offers, consent, corrections and interruptions. Preserve the distinction between visible facts and final private task constraints. |
| `simulators/tools.py` | Tool schemas and transactional scheduling effects: identity, consent, versions, capacity and idempotency. A successful tool response alone cannot establish task success. |
| `envs/native.py` | Dialogue/tool transitions, scheduled fault application, virtual budgets, termination and reward assembly. New faults also need `contracts.validate_fault` validation and scenario coverage. |
| `verifiers/state.py` | Independent reconstruction of initial rows and audited writes, final task correctness, policy compliance, truthful reports and forbidden side effects. Add correctness rules here, with backend-regression counterexamples that do not rely on tool enforcement. |
| `verifiers/audit.py` | Executable shortcut probes and the isolated educational naive scorer. The naive comparator must never become a production reward mode. |
| `rollouts/runner.py` and `trace.py` | Episode isolation, observation-only policy calls, bounded infrastructure retries, process recovery, trace persistence, exact replay and counterfactual branches. Resource accounting must distinguish policy tool attempts from dispatched calls and infrastructure retries. |
| `policies/` and `evaluation/` | Public-observation baselines, PPO training, registered experiment plans and separately named outcome metrics. Keep training and evaluation provenance explicit. |

For a new reward rule, update the independent verdict if correctness changes,
then the reward assembly in `envs/native.py` and [reward specification](docs/rewards.md).
Add an executable shortcut probe and regression where applicable. Shaping must
remain bounded against repeated questions, queries and correction cycles.
Task success, justified refusal/handoff and appropriate resolution remain
distinct metrics; a refusal must not count as a completed appointment.

## Checks for behavioral changes

Start with a regression that exposes the changed invariant. Cover both the
intended successful path and a counterexample, rather than testing only that a
tool returned `ok`.

1. Run the routine lint, type, test, contract and reward-audit checks in
   [CONTRIBUTING.md](CONTRIBUTING.md). Include stateful/adversarial coverage for
   authorization, consent, capacity, duplicates, versions and reporting when
   changing scheduling or verification behavior.
2. Check native/Gym transition agreement when changing action mapping or public
   observations. Hidden-goal changes must not alter policy encoding or masks.
3. Check seeded reset determinism, snapshot/restore, exact replay, and
   serial/process parity for transition, RNG, fault, trace or runner changes.
   Run representative saved traces with `callflow replay --trace ...`.
   Exact replay rejects a changed source fingerprint: retain the historical
   trace, and capture a separate candidate trace when a new build is needed.
   Allowing a build mismatch is investigative and does not establish parity.
4. Identify which published measurements a behavior change invalidates before
   changing evidence. Document the required reevaluation and preserve historical
   reports/checkpoints. Put candidate outputs in a separate directory, compare
   outcomes and provenance, and review differences before replacing any current
   publication. Documentation-only edits do not require new experiments.

Use [the scenario guide](docs/scenarios.md), [reward contract](docs/rewards.md),
and [experiment protocol](docs/experiments.md) for the detailed checks.

## Frozen suites and research integrity

`callflow_rl.scenarios` exposes
`generate(archetype, seed, split="train") -> ScenarioSpec`,
`load_suite(split) -> list[ScenarioSpec]`, and `manifest_hash(split) -> str`.
Generation uses local seeded randomness and rejects cross-split archetypes.
Loading checks frozen manifest hashes and returns fresh scenario objects.

Use `uv run --no-sync python -m callflow_rl.scenarios.freeze --check` to check
the generator against tracked definitions. `--write` explicitly changes those
definitions; it is a reviewed suite-maintenance operation, not routine setup or
evaluation. Decide the suite version and replacement evidence before writing.
Never silently regenerate a test manifest, move an archetype between splits or
replace a failing held-out case to improve a score.

Choose architectures, hyperparameters, training budgets and checkpoints using
train/development data only. Test and transfer archetypes are held out from PPO
updates and selection, but visible to environment developers; this is not an
externally blind benchmark. Do not tune a policy against test failures. Register
new research plans, seeds and selection rules before running experiments.

The original PPO weights and learning curves were trained at `837af56`. Later
environment/verifier evaluations reuse those saved weights; rescoring is not
additional gradient training. Record environment/verifier versions, source and
manifest hashes, checkpoint hashes and evaluation seeds. Archive superseded
reports with checksums, keep new training runs separate, and retain negative
results and across-seed variability. See
[evaluation provenance](artifacts/experiments/evaluation_release.json) and
[measured results](artifacts/experiments/RESULTS.md).
