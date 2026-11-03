# Shared implementation contract

The project brief in docs/project-brief.md is the acceptance source. All data are synthetic.

Core API (coordinate changes with other owners):

- `callflow_rl.contracts`: dataclasses (or equivalent) for `ScenarioSpec`, private goals, actions, traces, verifier outcomes. Scenario specs must be JSON round-trippable.
- `callflow_rl.envs.native.CallFlowEnv(scenario=None, reward_mode="shaped")`.
- `reset(seed=None, options=None)` returns `(public_observation: dict, info: dict)`. `options["scenario"]` may supply a ScenarioSpec.
- `step(action: dict)` returns `(observation, reward, terminated, truncated, info)`.
- Native actions use `{"type": "CALL_TOOL", "tool": "...", "args": {...}}` or the other published action names. Invalid inputs yield explicit safe errors.
- `snapshot()` returns a JSON-serializable full evaluator snapshot; `restore(snapshot)` restores it. `state_hash()` covers all transition-relevant state including RNGs and dialogue/fault queues. Evaluator snapshots are never exposed to policies.
- `.scenario` is evaluator-only; `.verdict()` yields a JSON-serializable independent verification result. `info` may hold evaluator metadata but policies receive observations only.
- `callflow_rl.envs.gym_env.CallFlowGymEnv` provides a bounded numeric wrapper, `.native`, `.action_for(index)` for finite actions and public observation encoding. No hidden goal information used by encoding/action mapping.
- `callflow_rl.scenarios`: `generate(archetype, seed, split="train") -> ScenarioSpec`, `load_suite(split) -> list[ScenarioSpec]`, `manifest_hash(split) -> str`.
- `callflow_rl.policies`: `RandomPolicy(seed=0)` / `HeuristicPolicy()` / learned adapter each have `act(observation: dict) -> dict`. No environment, info or hidden metadata argument.

Ownership: core agent owns contracts/core/simulators/envs/verifiers; scenarios agent owns scenarios and core/adversarial tests plus scenario/reward docs; learning agent owns policies/evaluation, configs/ppo.yaml, learning tests and experiment outputs. Root owns rollouts, CLI, integration/quality/release docs and packaging. Agents may read everything; do not commit other owners' files. Root coordinates commits and formatting after integration.

Keep updates explicit, tests substantive, no fabricated metrics, and no external services needed by the demo after dependencies install.
