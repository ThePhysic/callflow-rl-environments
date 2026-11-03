"""Real process-failure recovery and semantic checks beyond trace checksums."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from callflow_rl.policies import HeuristicPolicy
from callflow_rl.rollouts import load_trace, replay, run_episode, save_trace
from callflow_rl.scenarios import generate


class OnceInvalidTool:
    def __init__(self):
        self.first = True
        self.heuristic = HeuristicPolicy()

    def act(self, observation):
        if self.first:
            self.first = False
            return {"type": "CALL_TOOL", "tool": "list_services", "args": {"unexpected": 1}}
        return self.heuristic.act(observation)


@pytest.fixture
def trace():
    return run_episode(OnceInvalidTool(), generate("book_direct", 7), seed=7)


def test_v2_records_dispatched_calls_separately_from_tool_decisions(trace, tmp_path):
    assert trace.metadata["trace_version"] == "2"
    assert trace.resources["tool_calls"] == trace.verdict["tool_call_count"] == 3
    assert trace.resources["tool_decisions"] == 4
    path = tmp_path / "v2.jsonl"
    save_trace(trace, path)
    assert replay(load_trace(path))["matched"]


def test_v1_legacy_tool_call_accounting_still_loads_and_replays(trace, tmp_path):
    trace.metadata["trace_version"] = "1"
    trace.resources["tool_calls"] = trace.resources.pop("tool_decisions")
    path = tmp_path / "v1.jsonl"
    save_trace(trace, path)
    assert replay(load_trace(path))["matched"]


@pytest.mark.parametrize(
    "field",
    [
        "steps",
        "total_reward",
        "tool_calls",
        "tool_decisions",
        "final_snapshot_hash",
        "wall_seconds",
    ],
)
def test_checksummed_but_inconsistent_resources_fail_replay(trace, tmp_path, field):
    trace.resources[field] = "bad" if field == "final_snapshot_hash" else -1
    path = tmp_path / "damaged.jsonl"
    save_trace(trace, path)
    loaded = load_trace(path)
    with pytest.raises(AssertionError, match="Resource"):
        replay(loaded)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, "1"])
def test_wall_clock_requires_finite_nonnegative_number(trace, value):
    trace.resources["wall_seconds"] = value
    with pytest.raises(AssertionError, match="wall_seconds"):
        replay(trace)


def test_wall_clock_does_not_have_to_match_replay_runtime(trace):
    trace.resources["wall_seconds"] = 0.0
    assert replay(trace)["matched"]


@pytest.mark.parametrize("field", ["initial_snapshot", "pre_snapshot", "initial_fingerprint"])
def test_checksummed_but_inconsistent_snapshots_fail_replay(trace, tmp_path, field):
    if field == "initial_snapshot":
        trace.initial_snapshot["state"]["turn"] += 1
    elif field == "pre_snapshot":
        trace.steps[2]["pre_snapshot"]["state"]["turn"] += 1
    else:
        trace.metadata["initial_state_fingerprint"] = "bad"
    path = tmp_path / "damaged.jsonl"
    save_trace(trace, path)
    with pytest.raises(AssertionError, match="snapshot|fingerprint"):
        replay(load_trace(path))


def test_replay_accepts_trace_without_optional_pre_action_snapshots():
    trace = run_episode(
        HeuristicPolicy(), generate("book_direct", 7), seed=7, capture_snapshots=False
    )
    assert replay(trace)["matched"]


WORKER_DRIVER = """\
import json
import os
import sys
import time
from pathlib import Path

import callflow_rl.rollouts.runner as runner
from callflow_rl.scenarios import generate

ORIGINAL = runner._worker
ROOT = Path(sys.argv[1])
MODE = sys.argv[2]


def controlled_worker(task):
    name = task["scenario"]["scenario_id"]
    with (ROOT / (name + ".attempts")).open("a", encoding="utf-8") as handle:
        handle.write(str(task["attempt"]) + "\\n")
    if name == "bad" and (MODE.endswith("permanent") or task["attempt"] == 0):
        # Give the parent time to receive the completed good task before death.
        time.sleep(0.15)
        if MODE.startswith("death"):
            os._exit(19)
        raise RuntimeError("controlled ordinary worker failure")
    return ORIGINAL(task)


if __name__ == "__main__":
    runner._worker = controlled_worker
    good = generate("book_direct", 7)
    bad = generate("book_direct", 8)
    good.scenario_id = "good"
    bad.scenario_id = "bad"
    try:
        result = runner.run_batch([good, bad], workers=1, infrastructure_retries=1)
        outcome = {
            "benchmark": result["benchmark"],
            "attempts": [row["metadata"]["infrastructure_attempt"] for row in result["episodes"]],
            "ids": [row["metadata"]["scenario"]["scenario_id"] for row in result["episodes"]],
        }
    except Exception as error:
        outcome = {"error": type(error).__name__, "message": str(error)}
    outcome["good_executions"] = (ROOT / "good.attempts").read_text().splitlines()
    outcome["bad_executions"] = (ROOT / "bad.attempts").read_text().splitlines()
    print(json.dumps(outcome))
"""


@pytest.mark.parametrize("mode", ["death_transient", "exception_transient"])
def test_transient_worker_failure_recovers_without_rerunning_completed_task(tmp_path, mode):
    outcome = run_worker_driver(tmp_path, mode)
    assert "error" not in outcome, outcome
    assert outcome["ids"] == ["good", "bad"]
    assert outcome["attempts"] == [0, 1]
    assert outcome["good_executions"] == ["0"]
    assert outcome["bad_executions"] == ["0", "1"]
    assert outcome["benchmark"]["infrastructure_retries"] == 1
    assert outcome["benchmark"]["worker_pool_restarts"] == int(mode.startswith("death"))
    assert outcome["benchmark"]["task_success_count"] == 2


@pytest.mark.parametrize(
    ("mode", "error"),
    [("death_permanent", "BrokenProcessPool"), ("exception_permanent", "RuntimeError")],
)
def test_permanent_worker_failure_stops_at_per_task_retry_budget(tmp_path, mode, error):
    outcome = run_worker_driver(tmp_path, mode)
    assert outcome["error"] == error
    assert outcome["good_executions"] == ["0"]
    assert outcome["bad_executions"] == ["0", "1"]


def run_worker_driver(tmp_path: Path, mode: str):
    driver = tmp_path / "worker_driver.py"
    driver.write_text(WORKER_DRIVER, encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(driver), str(tmp_path), mode],
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    return json.loads(process.stdout.strip())
