"""Commit-outcome accounting and clause-scoped report assertions."""

from copy import deepcopy

import pytest

from callflow_rl.envs.gym_env import action_from_observation
from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.scenarios import generate


def _ready_for_booking(archetype="book_direct"):
    env = CallFlowEnv(generate(archetype, 42))
    obs, _ = env.reset(seed=42)
    for index in [4, 5, 6, 9]:
        obs, *_ = env.step(action_from_observation(index, obs))
    return env, action_from_observation(10, obs)


def test_successful_postcommit_timeout_has_private_commit_evidence():
    env, action = _ready_for_booking("book_timeout_after")
    obs, *_ = env.step(action)
    assert obs["last_tool"] == {
        "tool": "create_booking",
        "ok": False,
        "error": "timeout",
        "ambiguous": True,
    }
    assert env.verdict()["had_postcommit_timeout"]
    faults = [e for e in env.state["audit"] if e["kind"] == "fault"]
    assert faults[-1]["operation_committed"] is True
    assert "operation_committed" not in str(obs)


def test_lost_idempotent_response_still_refers_to_committed_operation():
    env, action = _ready_for_booking()
    env.step(action)
    env.inject_fault({"tool": "create_booking", "kind": "timeout_after", "occurrence": 2})
    obs, *_ = env.step(deepcopy(action))
    assert obs["last_tool"]["error"] == "timeout"
    assert env.verdict()["had_postcommit_timeout"]
    assert len([e for e in env.state["audit"] if e["kind"] == "write"]) == 1
    assert len(env.state["world"]["bookings"]) == 1


def test_precommit_and_postcommit_timeout_have_same_public_response():
    before, before_action = _ready_for_booking("book_timeout_before")
    after, after_action = _ready_for_booking("book_timeout_after")
    before_obs, *_ = before.step(before_action)
    after_obs, *_ = after.step(after_action)
    assert before_obs["last_tool"] == after_obs["last_tool"]
    assert not before.verdict()["had_postcommit_timeout"]
    assert after.verdict()["had_postcommit_timeout"]


@pytest.mark.parametrize(
    "text",
    [
        "I cannot help with that, but your appointment is booked.",
        "No problem — your appointment is reserved.",
        "Your appointment is not cancelled; your appointment is booked.",
    ],
)
def test_unrelated_negation_cannot_conceal_positive_claim(text):
    env = CallFlowEnv(generate("book_direct", 42))
    env.reset(seed=42)
    _, _, terminated, _, info = env.step({"type": "MESSAGE_USER", "text": text})
    assert terminated
    assert info["verdict"]["false_success_claim"]


@pytest.mark.parametrize(
    "text",
    [
        "Your appointment is not booked.",
        "No appointment was confirmed.",
        "I cannot say your appointment is reserved.",
    ],
)
def test_negated_claims_remain_nonassertions(text):
    env = CallFlowEnv(generate("book_direct", 42))
    env.reset(seed=42)
    _, _, terminated, _, info = env.step({"type": "MESSAGE_USER", "text": text})
    assert not terminated
    assert not info["verdict"]["false_success_claim"]
