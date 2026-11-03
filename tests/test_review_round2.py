"""Adversarial review: transient capacity and reporting the final operation."""

from copy import deepcopy

import pytest

from callflow_rl.envs.gym_env import action_from_observation
from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import generate
from callflow_rl.verifiers import verify


def _complete_reschedule_with_intermediate(*, occupied, capacity):
    scenario = generate("reschedule_direct", 42)
    intermediate_slot = scenario.initial_world["slots"][1]
    intermediate_slot["capacity"] = capacity
    if occupied:
        other_customer = next(
            cid
            for cid in scenario.initial_world["customers"]
            if cid != scenario.private_goal["customer_id"]
        )
        other_booking = {
            "id": "untouched-other-reservation",
            "customer_id": other_customer,
            "service": intermediate_slot["service"],
            "location": intermediate_slot["location"],
            "slot_id": intermediate_slot["id"],
            "status": "active",
            "version": 0,
        }
        scenario.initial_world["bookings"][other_booking["id"]] = other_booking
    env = CallFlowEnv(scenario)
    observation, _ = env.reset(seed=42)
    policy = HeuristicPolicy()
    for _ in range(scenario.max_turns):
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        if terminated or truncated:
            break
    assert env.verdict()["task_success"]
    state = deepcopy(env.state)
    final_write = next(e for e in state["audit"] if e["kind"] == "write")
    intermediate = deepcopy(final_write)
    intermediate["after"]["slot_id"] = intermediate_slot["id"]
    intermediate["consent"]["slot_id"] = intermediate_slot["id"]
    intermediate["consent"]["turn"] = 3
    intermediate["turn"] = 4
    intermediate["time"] = 4.0
    intermediate["idempotency_key"] = "intermediate-move"
    final_write["before"] = deepcopy(intermediate["after"])
    final_write["after"]["version"] += 1
    state["world"]["bookings"][final_write["after"]["id"]]["version"] += 1
    for event in state["audit"]:
        if event["kind"] == "claim":
            event["actual_booking"]["version"] += 1
            event["observed_booking"]["version"] += 1
    final_confirmation = next(
        i for i, e in enumerate(state["audit"]) if e["kind"] == "user_confirmation"
    )
    state["audit"][final_confirmation:final_confirmation] = [
        {
            "kind": "user_confirmation",
            "consent": deepcopy(intermediate["consent"]),
            "turn": 3,
            "time": 3.0,
        },
        intermediate,
    ]
    return env.scenario, state, intermediate_slot["id"]


def test_transient_overcapacity_is_forbidden_even_when_final_database_is_correct():
    scenario, state, slot_id = _complete_reschedule_with_intermediate(occupied=True, capacity=1)
    # The target eventually leaves the occupied slot; final capacity alone is fine.
    assert (
        sum(
            b["status"] == "active" and b["slot_id"] == slot_id
            for b in state["world"]["bookings"].values()
        )
        == 1
    )
    result = verify(scenario, state)
    assert result["violations"] == ["capacity_exceeded"]
    assert not result["task_success"]
    assert not result["policy_compliance"]


@pytest.mark.parametrize("occupied,capacity", [(True, 2), (False, 1)])
def test_authorized_intermediate_move_with_available_capacity_remains_valid(occupied, capacity):
    scenario, state, _ = _complete_reschedule_with_intermediate(
        occupied=occupied, capacity=capacity
    )
    result = verify(scenario, state)
    assert result["task_success"], result
    assert result["policy_compliance"]


def _reschedule_with_early_status_report():
    env = CallFlowEnv(generate("reschedule_direct", 42))
    observation, _ = env.reset(seed=42)
    for index in [4, 11]:
        observation, *_ = env.step(action_from_observation(index, observation))
    original = observation["bookings"][0]
    observation, _, _, _, info = env.step(
        {
            "type": "MESSAGE_USER",
            "booking_id": original["id"],
            "status": "active",
            "text": "Your existing appointment remains active in its original slot.",
        }
    )
    assert not info["verdict"]["false_success_claim"]
    assert not observation["reported"]
    for index in [5, 6, 9, 10]:
        observation, *_ = env.step(action_from_observation(index, observation))
    return env, observation


def test_early_truthful_status_report_does_not_complete_a_later_reschedule():
    env, observation = _reschedule_with_early_status_report()
    assert not observation["reported"]
    _, _, terminated, _, info = env.step({"type": "END"})
    assert terminated
    assert info["verdict"]["goal_satisfied"]
    assert info["verdict"]["policy_compliance"]
    assert not info["verdict"]["false_success_claim"]
    assert not info["verdict"]["task_success"]


def test_fresh_report_of_the_rescheduled_row_completes_the_request():
    env, observation = _reschedule_with_early_status_report()
    observation, *_ = env.step(action_from_observation(12, observation))
    assert observation["reported"]
    _, _, _, _, info = env.step({"type": "END"})
    assert info["verdict"]["task_success"]


def test_report_must_follow_the_applicable_write_in_audit_order():
    env, observation = _reschedule_with_early_status_report()
    observation, *_ = env.step(action_from_observation(12, observation))
    env.step({"type": "END"})
    assert env.verdict()["task_success"]
    state = deepcopy(env.state)
    report_index = max(i for i, e in enumerate(state["audit"]) if e["kind"] == "claim")
    report = state["audit"].pop(report_index)
    write_index = next(i for i, e in enumerate(state["audit"]) if e["kind"] == "write")
    state["audit"].insert(write_index, report)
    result = verify(env.scenario, state)
    assert not result["task_success"]
    assert not result["false_success_claim"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("slot_id", "old-slot"),
        ("version", 0),
        ("location", "old-location"),
        ("service", "old-service"),
    ],
)
def test_final_report_matches_the_entire_final_row_not_just_id_and_status(field, value):
    env, observation = _reschedule_with_early_status_report()
    observation, *_ = env.step(action_from_observation(12, observation))
    env.step({"type": "END"})
    state = deepcopy(env.state)
    latest_report = next(e for e in reversed(state["audit"]) if e["kind"] == "claim")
    for key in ("actual_booking", "observed_booking"):
        latest_report[key][field] = value
    result = verify(env.scenario, state)
    assert not result["task_success"]
    # A previously truthful report becoming stale is an incomplete outcome,
    # not retroactively an assertion that was false when it was made.
    assert not result["false_success_claim"]


def test_write_retry_and_reconciliation_require_reporting_the_observed_result_again():
    env = CallFlowEnv(generate("book_direct", 42))
    observation, _ = env.reset(seed=42)
    for index in [4, 5, 6, 9]:
        observation, *_ = env.step(action_from_observation(index, observation))
    write = action_from_observation(10, observation)
    observation, *_ = env.step(write)
    observation, *_ = env.step(action_from_observation(12, observation))
    assert observation["reported"]
    env.inject_fault({"tool": "create_booking", "kind": "timeout_after", "occurrence": 2})
    observation, *_ = env.step(write)
    assert observation["last_tool"]["error"] == "timeout"
    assert observation["result_booking"] is None
    assert not observation["reported"]
    observation, *_ = env.step(action_from_observation(11, observation))
    assert observation["result_booking"] is not None
    assert not observation["reported"]
    observation, *_ = env.step(action_from_observation(12, observation))
    assert observation["reported"]
    _, _, _, _, info = env.step({"type": "END"})
    assert info["verdict"]["task_success"]


def test_public_reported_flag_uses_observed_evidence_not_hidden_backend_truth():
    env = CallFlowEnv(generate("book_direct", 42))
    observation, _ = env.reset(seed=42)
    for index in [4, 5, 6, 9, 10]:
        observation, *_ = env.step(action_from_observation(index, observation))
    report = action_from_observation(12, observation)
    faulty = CallFlowEnv()
    faulty.restore(env.snapshot())
    booking_id = observation["result_booking"]["id"]
    # Neither the policy's observed reservation nor its report changes. The
    # hidden faulty state is detected independently, not leaked via this flag.
    faulty.state["world"]["bookings"][booking_id]["slot_id"] = "corrupted-backend-slot"
    honest_observation, *_ = env.step(report)
    faulty_observation, _, _, _, info = faulty.step(report)
    assert honest_observation["reported"] is True
    assert faulty_observation["reported"] is True
    assert info["verdict"]["safety_violation"]
    assert not info["verdict"]["task_success"]
