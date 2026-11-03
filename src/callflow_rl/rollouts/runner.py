"""Only observations cross the policy boundary; evaluator state stays in traces."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from callflow_rl import __version__
from callflow_rl.contracts import ScenarioSpec
from callflow_rl.envs.native import CallFlowEnv


class Policy(Protocol):
    def act(self, observation: dict[str, Any]) -> dict[str, Any]: ...


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def build_fingerprint() -> str:
    """Hash installed source, independent of checkout location or git status."""
    root = Path(__file__).resolve().parents[1]
    entries = {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    return digest(entries)


@dataclass
class EpisodeTrace:
    metadata: dict[str, Any]
    initial_snapshot: dict[str, Any]
    initial_observation: dict[str, Any]
    initial_state_hash: str
    steps: list[dict[str, Any]]
    verdict: dict[str, Any]
    resources: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeTrace:
        return cls(**value)


def run_episode(
    policy: Policy,
    scenario: ScenarioSpec,
    seed: int = 0,
    reward_mode: str = "shaped",
    capture_snapshots: bool = True,
    policy_version: str | None = None,
    manifest: str | None = None,
) -> EpisodeTrace:
    start = time.perf_counter()
    env = CallFlowEnv(scenario=scenario, reward_mode=reward_mode)
    obs, _ = env.reset(seed=seed)
    initial_snapshot = copy.deepcopy(env.snapshot())
    initial_hash = env.state_hash()
    first_obs = copy.deepcopy(obs)
    steps: list[dict[str, Any]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        before_hash = env.state_hash()
        before_snapshot = copy.deepcopy(env.snapshot()) if capture_snapshots else None
        action = copy.deepcopy(policy.act(copy.deepcopy(obs)))
        obs, reward, terminated, truncated, info = env.step(action)
        record = {
            "index": len(steps),
            "action": action,
            "observation": copy.deepcopy(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": copy.deepcopy(info),
            "pre_state_hash": before_hash,
            "post_state_hash": env.state_hash(),
        }
        if capture_snapshots:
            record["pre_snapshot"] = before_snapshot
        steps.append(record)
    return EpisodeTrace(
        metadata={
            "trace_version": "1",
            "environment_version": __version__,
            "build_fingerprint": build_fingerprint(),
            "scenario": scenario.to_dict(),
            "scenario_hash": digest(scenario.to_dict()),
            "scenario_manifest_hash": manifest or digest(scenario.to_dict()),
            "policy_version": policy_version or getattr(policy, "version", type(policy).__name__),
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "seed": seed,
            "runtime": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "gymnasium", "torch")},
            },
            "rng_subseeds": initial_snapshot["state"].get("rng_subseeds", {"episode": seed}),
            "reward_mode": reward_mode,
            "initial_state_fingerprint": initial_hash,
        },
        initial_snapshot=initial_snapshot,
        initial_observation=first_obs,
        initial_state_hash=initial_hash,
        steps=steps,
        verdict=copy.deepcopy(env.verdict()),
        resources={
            "wall_seconds": time.perf_counter() - start,
            "steps": len(steps),
            "tool_calls": sum(
                isinstance(s["action"], dict) and s["action"].get("type") == "CALL_TOOL"
                for s in steps
            ),
            "total_reward": sum(s["reward"] for s in steps),
            "final_state_hash": env.state_hash(),
            "final_snapshot_hash": digest(env.snapshot()),
        },
    )


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    from callflow_rl.policies import HeuristicPolicy, RandomPolicy

    name = task["policy"]
    if name == "heuristic":
        policy: Policy = HeuristicPolicy()
    elif name == "random":
        policy = RandomPolicy(seed=task["seed"])
    elif name == "ppo":
        from callflow_rl.policies import PPOPolicy

        policy = PPOPolicy(task["checkpoint"])
    else:
        raise ValueError(f"Unknown policy {name!r}")
    trace = run_episode(
        policy,
        ScenarioSpec.from_dict(task["scenario"]),
        seed=task["seed"],
        reward_mode=task["reward_mode"],
        capture_snapshots=False,
        manifest=task["manifest"],
        policy_version=getattr(policy, "version", name),
    )
    result = trace.to_dict()
    result["metadata"]["infrastructure_attempt"] = task["attempt"]
    return result


def run_batch(
    scenarios: list[ScenarioSpec],
    policy: str = "heuristic",
    seed: int = 42,
    workers: int = 4,
    checkpoint: str | None = None,
    reward_mode: str = "shaped",
    manifest: str | None = None,
    infrastructure_retries: int = 1,
) -> dict[str, Any]:
    """Spawned processes own separate envs/RNGs; retries are evaluator metadata."""
    if not scenarios or workers < 1 or infrastructure_retries < 0:
        raise ValueError("Require scenarios, positive workers and nonnegative retries")
    if policy not in {"heuristic", "random", "ppo"}:
        raise ValueError("Unknown policy")
    if policy == "ppo" and not checkpoint:
        raise ValueError("PPO requires a checkpoint")
    tasks: list[dict[str, Any]] = [
        {
            "scenario": s.to_dict(),
            "policy": policy,
            "seed": seed + i,
            "checkpoint": checkpoint,
            "reward_mode": reward_mode,
            "attempt": 0,
            "manifest": manifest,
        }
        for i, s in enumerate(scenarios)
    ]
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    retries = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, task): i for i, task in enumerate(tasks)}
        while futures:
            for future in as_completed(list(futures)):
                i = futures.pop(future)
                try:
                    results[i] = future.result()
                except Exception:
                    if tasks[i]["attempt"] >= infrastructure_retries:
                        raise
                    tasks[i]["attempt"] += 1
                    retries += 1
                    futures[pool.submit(_worker, tasks[i])] = i
    elapsed = time.perf_counter() - started
    complete = [r for r in results if r is not None]
    latencies = [r["resources"]["wall_seconds"] for r in complete]
    return {
        "episodes": complete,
        "benchmark": {
            "episode_count": len(complete),
            "workers": workers,
            "wall_seconds_including_worker_startup": elapsed,
            "episodes_per_second": len(complete) / elapsed,
            "p50_episode_seconds": float(np.percentile(latencies, 50)),
            "p95_episode_seconds": float(np.percentile(latencies, 95)),
            "infrastructure_retries": retries,
            "task_success_count": sum(bool(r["verdict"].get("task_success")) for r in complete),
            "appropriate_resolution_count": sum(
                bool(
                    r["verdict"].get("task_success")
                    or r["verdict"].get("justified_handoff")
                    or r["verdict"].get("justified_refusal")
                )
                for r in complete
            ),
            "justified_handoff_count": sum(
                bool(r["verdict"].get("justified_handoff")) for r in complete
            ),
            "justified_refusal_count": sum(
                bool(r["verdict"].get("justified_refusal")) for r in complete
            ),
        },
    }
