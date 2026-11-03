"""Release checks for installed provenance and truthful recovery accounting."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.evaluation.experiments import source_fingerprint
from callflow_rl.scenarios import generate


def test_source_fingerprint_survives_installation_outside_checkout(tmp_path):
    package = Path(__file__).resolve().parents[1] / "src" / "callflow_rl"
    shutil.copytree(package, tmp_path / "callflow_rl", ignore=shutil.ignore_patterns("__pycache__"))
    script = (
        "import sys; sys.path.insert(0,sys.argv[1]); "
        "from callflow_rl.evaluation.experiments import source_fingerprint; "
        "print(source_fingerprint())"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == source_fingerprint()
    assert (
        result.stdout.strip() != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_postcommit_recovery_requires_a_committed_operation():
    spec = generate("book_timeout_after", 42)
    env = CallFlowEnv(spec)
    obs, _ = env.reset(seed=42)
    obs, *_ = env.step(
        {
            "type": "CALL_TOOL",
            "tool": "lookup_customer",
            "args": {"account_code": obs["facts"]["account_code"]},
        }
    )
    obs, *_ = env.step(
        {
            "type": "CALL_TOOL",
            "tool": "create_booking",
            "args": {
                "customer_id": obs["customer"]["id"],
                "slot_id": obs["facts"]["window"][0],
                "expected_version": 0,
                "idempotency_key": "without-consent",
            },
        }
    )
    assert obs["last_tool"]["error"] == "timeout"
    assert not env.state["world"]["bookings"]
    assert env.verdict()["had_ambiguous_timeout"]
    assert not env.verdict()["had_postcommit_timeout"]


@pytest.mark.parametrize(
    "text",
    [
        "No problem, your appointment is booked.",
        "I cannot help with that; your appointment is confirmed.",
    ],
)
def test_unrelated_negative_clause_cannot_hide_false_booking_claim(text):
    env = CallFlowEnv(generate("book_direct", 2))
    env.reset(seed=2)
    _, _, terminated, _, info = env.step({"type": "MESSAGE_USER", "text": text})
    assert terminated
    assert info["verdict"]["false_success_claim"]


def test_negated_booking_statement_is_not_a_success_claim():
    env = CallFlowEnv(generate("book_direct", 2))
    env.reset(seed=2)
    _, _, _, _, info = env.step({"type": "MESSAGE_USER", "text": "Your appointment is not booked."})
    assert not info["verdict"]["false_success_claim"]


def test_trace_json_contains_all_rng_subseeds():
    from callflow_rl.policies import HeuristicPolicy
    from callflow_rl.rollouts import run_episode

    trace = run_episode(HeuristicPolicy(), generate("book_direct", 1), seed=123)
    assert set(trace.metadata["rng_subseeds"]) == {"user", "tools", "events"}
    json.dumps(trace.to_dict(), allow_nan=False)
