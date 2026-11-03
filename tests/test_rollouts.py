"""Replay verifies transitions, not just a saved success bit."""

import copy
import json

import pytest

from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.rollouts import (
    branch,
    load_trace,
    recorded_replay,
    replay,
    run_batch,
    run_episode,
    save_trace,
)
from callflow_rl.scenarios import generate


def successful_trace():
    return run_episode(HeuristicPolicy(), generate("book_direct", 42), seed=42)


@pytest.mark.parametrize("suffix", ["jsonl", "parquet"])
def test_trace_round_trip_reexecutes_every_transition(tmp_path, suffix):
    trace = successful_trace()
    assert trace.verdict["task_success"]
    path = tmp_path / f"episode.{suffix}"
    save_trace(trace, path)
    loaded = load_trace(path)
    assert loaded.to_dict() == trace.to_dict()
    assert replay(loaded)["matched"]
    assert recorded_replay(loaded) == trace.steps


def test_corrupt_and_truncated_trace_fail_closed(tmp_path):
    path = tmp_path / "episode.jsonl"
    save_trace(successful_trace(), path)
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1]))
    with pytest.raises(ValueError, match="Incomplete"):
        load_trace(path)
    step = json.loads(rows[1])
    step["reward"] += 1
    rows[1] = json.dumps(step)
    path.write_text("\n".join(rows))
    with pytest.raises(ValueError, match="checksum"):
        load_trace(path)


def test_modified_reward_and_wrong_build_rejected():
    trace = successful_trace()
    trace.steps[0]["reward"] += 1
    with pytest.raises(AssertionError, match="reward"):
        replay(trace)
    trace = successful_trace()
    trace.metadata["build_fingerprint"] = "different"
    with pytest.raises(ValueError, match="Build differs"):
        replay(trace)


def test_branch_from_snapshot_does_not_mutate_original():
    trace = successful_trace()
    saved = copy.deepcopy(trace.to_dict())
    fork = branch(trace, 0, {"type": "END"}, HeuristicPolicy())
    assert not fork["branch_verdict"]["task_success"]
    assert trace.to_dict() == saved
    assert replay(trace)["matched"]


def test_midpoint_snapshot_round_trip():
    trace = successful_trace()
    step = trace.steps[len(trace.steps) // 2]
    env = CallFlowEnv(generate("book_direct", 42))
    env.reset(seed=42)
    env.restore(copy.deepcopy(step["pre_snapshot"]))
    assert env.state_hash() == step["pre_state_hash"]
    obs, reward, terminated, truncated, info = env.step(step["action"])
    assert (obs, reward, terminated, truncated, info) == (
        step["observation"],
        step["reward"],
        step["terminated"],
        step["truncated"],
        step["info"],
    )


def test_parallel_workers_match_serial_actions_and_final_hashes():
    scenarios = [generate("book_timeout_after", 90 + i) for i in range(8)]
    batch = run_batch(scenarios, seed=100, workers=4)
    assert batch["benchmark"]["episode_count"] == 8
    assert batch["benchmark"]["workers"] == 4
    assert batch["benchmark"]["infrastructure_retries"] == 0
    for i, (spec, result) in enumerate(zip(scenarios, batch["episodes"], strict=True)):
        serial = run_episode(HeuristicPolicy(), spec, seed=100 + i)
        assert result["resources"]["final_state_hash"] == serial.resources["final_state_hash"]
        assert result["verdict"] == serial.verdict
        assert result["verdict"]["task_success"]


def test_policy_cannot_mutate_environment_through_observation():
    class MutatingPolicy:
        def act(self, obs):
            obs.clear()
            obs["private_goal"] = {"outcome": "completed"}
            return {"type": "END"}

    trace = run_episode(MutatingPolicy(), generate("book_direct", 0))
    assert not trace.verdict["task_success"]
    assert "private_goal" not in trace.initial_observation
