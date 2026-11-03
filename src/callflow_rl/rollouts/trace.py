"""Lossless traces and replay; trace files contain private evaluator data."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from callflow_rl.contracts import ScenarioSpec
from callflow_rl.envs.native import CallFlowEnv

from .runner import EpisodeTrace, Policy, build_fingerprint, canonical, digest


def save_trace(trace: EpisodeTrace, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = trace.to_dict()
    rows = [
        {"record": "header", **{k: v for k, v in payload.items() if k != "steps"}},
        *({"record": "step", **step} for step in trace.steps),
        {"record": "checksum", "sha256": digest(payload)},
    ]
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"record_json": [canonical(row) for row in rows]}), path)
    else:
        path.write_text("\n".join(canonical(row) for row in rows) + "\n", encoding="utf-8")


def load_trace(path: str | Path) -> EpisodeTrace:
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = [json.loads(row) for row in pq.read_table(path)["record_json"].to_pylist()]
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 2 or rows[0].get("record") != "header":
        raise ValueError("Missing trace header")
    if rows[-1].get("record") != "checksum":
        raise ValueError("Incomplete trace: missing checksum")
    data = {k: v for k, v in rows[0].items() if k != "record"}
    if data.get("metadata", {}).get("trace_version") != "1":
        raise ValueError("Unsupported trace version")
    if any(r.get("record") != "step" for r in rows[1:-1]):
        raise ValueError("Unexpected trace record")
    data["steps"] = [{k: v for k, v in row.items() if k != "record"} for row in rows[1:-1]]
    if digest(data) != rows[-1]["sha256"]:
        raise ValueError("Trace checksum mismatch")
    return EpisodeTrace.from_dict(data)


def replay(trace: EpisodeTrace, allow_build_mismatch: bool = False) -> dict[str, Any]:
    if not allow_build_mismatch and trace.metadata["build_fingerprint"] != build_fingerprint():
        raise ValueError("Build differs from recorded source; explicitly allow to investigate")
    spec = ScenarioSpec.from_dict(trace.metadata["scenario"])
    if digest(spec.to_dict()) != trace.metadata["scenario_hash"]:
        raise ValueError("Scenario differs from recorded manifest")
    env = CallFlowEnv(scenario=spec, reward_mode=trace.metadata["reward_mode"])
    obs, _ = env.reset(seed=trace.metadata["seed"])
    if obs != trace.initial_observation or env.state_hash() != trace.initial_state_hash:
        raise AssertionError("Initial observation/state does not match")
    for i, expected in enumerate(trace.steps):
        if expected["index"] != i or env.state_hash() != expected["pre_state_hash"]:
            raise AssertionError(f"Pre-action state mismatch at step {i}")
        obs, reward, terminated, truncated, info = env.step(copy.deepcopy(expected["action"]))
        actual = {
            "observation": obs,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "post_state_hash": env.state_hash(),
        }
        for key, value in actual.items():
            if canonical(value) != canonical(expected[key]):
                raise AssertionError(f"Replay mismatch at step {i}: {key}")
    if env.verdict() != trace.verdict:
        raise AssertionError("Final verifier outcome mismatch")
    if env.state_hash() != trace.resources["final_state_hash"]:
        raise AssertionError("Final state mismatch")
    return {"matched": True, "steps": len(trace.steps), "verdict": env.verdict()}


def recorded_replay(trace: EpisodeTrace) -> list[dict[str, Any]]:
    """Play saved outputs without external calls; this is not causal verification."""
    previous = trace.initial_state_hash
    for i, step in enumerate(trace.steps):
        if step["index"] != i or step["pre_state_hash"] != previous:
            raise ValueError("Broken recorded state chain")
        previous = step["post_state_hash"]
    if previous != trace.resources["final_state_hash"]:
        raise ValueError("Recorded final state mismatch")
    return copy.deepcopy(trace.steps)


def branch(
    trace: EpisodeTrace,
    step_index: int,
    action: dict[str, Any],
    continuation: Policy,
    fault: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fork a pre-action snapshot, apply a different action/fault, then continue."""
    if not 0 <= step_index < len(trace.steps):
        raise ValueError("Branch index outside trace")
    env = CallFlowEnv(
        ScenarioSpec.from_dict(trace.metadata["scenario"]),
        reward_mode=trace.metadata["reward_mode"],
    )
    env.reset(seed=trace.metadata["seed"])
    record = trace.steps[step_index]
    if "pre_snapshot" in record:
        env.restore(copy.deepcopy(record["pre_snapshot"]))
    else:
        for prior in trace.steps[:step_index]:
            env.step(copy.deepcopy(prior["action"]))
    if env.state_hash() != record["pre_state_hash"]:
        raise AssertionError("Cannot reproduce branch state")
    if fault is not None:
        env.inject_fault(copy.deepcopy(fault))
    obs, reward, terminated, truncated, _ = env.step(copy.deepcopy(action))
    rewards = [float(reward)]
    actions = [copy.deepcopy(action)]
    while not (terminated or truncated):
        next_action = copy.deepcopy(continuation.act(copy.deepcopy(obs)))
        obs, reward, terminated, truncated, _ = env.step(next_action)
        rewards.append(float(reward))
        actions.append(next_action)
    return {
        "fork_step": step_index,
        "original_verdict": trace.verdict,
        "branch_verdict": env.verdict(),
        "actions": actions,
        "branch_reward": sum(rewards),
        "final_state_hash": env.state_hash(),
        "state_changed": env.state_hash() != trace.resources["final_state_hash"],
    }


def view(trace: EpisodeTrace) -> str:
    lines = [
        f"Scenario: {trace.metadata['scenario'].get('scenario_id')}",
        f"Policy: {trace.metadata['policy_version']} | seed={trace.metadata['seed']}",
    ]
    for step in trace.steps:
        obs = step["observation"]
        lines.append(
            f"{step['index']:02d} {canonical(step['action'])} -> reward={step['reward']:.3f}"
        )
        if obs.get("last_tool"):
            lines.append(f"   tool: {canonical(obs['last_tool'])}")
        if step["info"]:
            lines.append(f"   evaluator: {canonical(step['info'])}")
    lines.append(f"Final verifier: {canonical(trace.verdict)}")
    return "\n".join(lines)
