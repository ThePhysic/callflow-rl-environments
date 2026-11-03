"""New counterexamples from the independent audit of the first release."""

from copy import deepcopy

import pytest
from gymnasium.utils.env_checker import check_env

from callflow_rl.contracts import MAX_VIRTUAL_DURATION, ScenarioSpec
from callflow_rl.envs.gym_env import CallFlowGymEnv
from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import generate
from callflow_rl.verifiers import verify


def finish(env, observation):
    policy = HeuristicPolicy()
    while True:
        observation, reward, terminated, truncated, info = env.step(policy.act(observation))
        if terminated or truncated:
            return observation, reward, terminated, truncated, info


def before_report(archetype="book_direct"):
    env = CallFlowEnv(generate(archetype, 42))
    observation, _ = env.reset(seed=42)
    policy = HeuristicPolicy()
    while not observation["result_booking"]:
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        assert not (terminated or truncated)
    return env, observation["result_booking"]


@pytest.mark.parametrize("gym", [False, True])
def test_default_seed_reset_does_not_depend_on_previous_reset(gym):
    first = CallFlowGymEnv() if gym else CallFlowEnv()
    first.reset(seed=1)
    first.reset(seed=2)
    second = CallFlowGymEnv() if gym else CallFlowEnv()
    second.reset(seed=2)
    native_first = first.native if gym else first
    native_second = second.native if gym else second
    assert native_first.observation() == native_second.observation()
    assert native_first.state_hash() == native_second.state_hash()


def test_explicit_scenario_is_preserved_and_automatic_mode_survives_snapshot():
    fixed = generate("book_direct", 7)
    env = CallFlowEnv(fixed)
    env.reset(seed=1)
    env.reset(seed=2)
    assert env.scenario.to_dict() == fixed.to_dict()
    automatic = CallFlowEnv()
    automatic.reset(seed=2)
    restored = CallFlowEnv(fixed)
    restored.restore(automatic.snapshot())
    restored.reset(seed=3)
    fresh = CallFlowEnv()
    fresh.reset(seed=3)
    assert restored.state_hash() == fresh.state_hash()
    automatic.reset(seed=2, options={"scenario": fixed})
    automatic.reset(seed=3)
    assert automatic.scenario.to_dict() == fixed.to_dict()


def test_default_gym_checker_after_seed_reset_fix():
    check_env(CallFlowGymEnv(), skip_render_check=True)


@pytest.mark.parametrize("deadline,success", [(6.5, False), (7.0, True)])
def test_end_cannot_bypass_virtual_deadline(deadline, success):
    spec = generate("book_direct", 42)
    spec.max_time = deadline
    env = CallFlowEnv(spec)
    observation, _ = env.reset(seed=42)
    _, reward, terminated, truncated, info = finish(env, observation)
    assert env.state["clock"] == 7
    assert info["verdict"]["task_success"] is success
    assert (terminated, truncated) == (success, not success)
    assert (reward > 0) is success
    assert len(env.state["world"]["bookings"]) == 1


def test_late_handoff_gets_no_justified_handoff_bonus():
    spec = generate("policy_handoff", 42)
    spec.max_time = 0.5
    env = CallFlowEnv(spec)
    env.reset(seed=42)
    _, reward, terminated, truncated, info = env.step({"type": "HANDOFF"})
    assert not terminated and truncated and reward < 0
    assert not info["verdict"]["justified_handoff"]


def test_inflight_write_is_retained_when_latency_exceeds_deadline():
    spec = generate("book_direct", 42)
    spec.max_time = 10
    spec.faults = [{"tool": "create_booking", "kind": "latency", "latency": 100}]
    env = CallFlowEnv(spec)
    observation, _ = env.reset(seed=42)
    _, reward, terminated, truncated, info = finish(env, observation)
    assert not terminated and truncated and reward < 0
    assert len(env.state["world"]["bookings"]) == 1
    assert len([e for e in env.state["audit"] if e["kind"] == "write"]) == 1
    assert info["verdict"]["termination_reason"] == "time_limit"


def test_safety_violation_takes_precedence_over_expired_time():
    spec = generate("book_direct", 42)
    spec.max_time = 0.5
    env = CallFlowEnv(spec)
    env.reset(seed=42)
    _, _, terminated, truncated, info = env.step(
        {"type": "MESSAGE_USER", "text": "Your appointment is booked."}
    )
    assert terminated and not truncated
    assert info["verdict"]["termination_reason"] == "safety_violation"


def test_verifier_checks_budget_without_trusting_truncation_flag():
    env = CallFlowEnv(generate("book_direct", 42))
    observation, _ = env.reset(seed=42)
    assert finish(env, observation)[-1]["verdict"]["task_success"]
    limited = deepcopy(env.scenario)
    limited.max_time = 6.5
    assert not verify(limited, env.state)["task_success"]


@pytest.mark.parametrize(
    "archetype,status,text",
    [
        ("book_direct", "active", "Your appointment is cancelled."),
        ("cancel_direct", "cancelled", "Your appointment is booked."),
        ("cancel_direct", "cancelled", "Your appointment is active."),
        ("book_direct", None, "Your appointment is booked; your appointment is cancelled."),
        ("book_direct", "active", "Your cancellation is confirmed."),
    ],
)
def test_every_supported_assertion_is_audited(archetype, status, text):
    env, booking = before_report(archetype)
    action = {"type": "MESSAGE_USER", "booking_id": booking["id"], "text": text}
    if status:
        action["status"] = status
    observation, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated and reward < 0
    assert info["verdict"]["false_success_claim"]
    assert not info["verdict"]["task_success"]
    assert not observation["reported"]


@pytest.mark.parametrize(
    "archetype,status,text",
    [
        ("cancel_direct", "cancelled", "Your cancellation is confirmed."),
        ("book_direct", "active", "Your appointment is booked but not cancelled."),
        ("cancel_direct", "cancelled", "Your appointment is cancelled and not booked."),
    ],
)
def test_matching_and_negated_reports_remain_valid(archetype, status, text):
    env, booking = before_report(archetype)
    _, _, terminated, truncated, _ = env.step(
        {"type": "MESSAGE_USER", "booking_id": booking["id"], "status": status, "text": text}
    )
    assert not (terminated or truncated)
    assert env.step({"type": "END"})[-1]["verdict"]["task_success"]


def test_identity_typo_recovery_uses_explicit_canonical_actor():
    spec = generate("book_direct", 42)
    principal = spec.private_goal["customer_id"]
    correct_code = spec.user["facts"]["account_code"]
    spec.user["principal_customer_id"] = principal
    spec.user["facts"]["account_code"] = "mistyped-code"
    spec.events = [
        {"turn": 1, "type": "correction", "field": "account_code", "value": correct_code}
    ]
    env = CallFlowEnv(spec)
    env.reset(seed=42)
    observation, *_ = env.step({"type": "WAIT"})
    assert "principal_customer_id" not in observation
    assert finish(env, observation)[-1]["verdict"]["task_success"]
    assert env.state["session_customer_id"] == principal


def test_correction_cannot_change_an_established_customer_principal():
    spec = generate("book_direct", 42)
    principal = spec.private_goal["customer_id"]
    other = next(c for cid, c in spec.initial_world["customers"].items() if cid != principal)
    spec.events = [
        {"turn": 1, "type": "correction", "field": "account_code", "value": other["account_code"]}
    ]
    env = CallFlowEnv(spec)
    env.reset(seed=42)
    env.step({"type": "WAIT"})
    observation, *_ = env.step(
        {
            "type": "CALL_TOOL",
            "tool": "lookup_customer",
            "args": {"account_code": other["account_code"]},
        }
    )
    assert observation["last_tool"]["error"] == "unauthorized_customer"
    assert env.state["session_customer_id"] == principal


def test_unresolvable_identity_correction_is_a_clear_configuration_error():
    spec = generate("book_direct", 42)
    correct = spec.user["facts"]["account_code"]
    spec.user["facts"]["account_code"] = "mistyped-code"
    spec.events = [{"turn": 1, "type": "correction", "field": "account_code", "value": correct}]
    with pytest.raises(ValueError, match="principal_customer_id"):
        CallFlowEnv(spec).reset(seed=42)


def test_postcommit_correction_invalidates_stale_operation_result():
    spec = generate("reschedule_direct", 42)
    revised_slot = spec.initial_world["slots"][1]["id"]
    spec.private_goal["allowed_slots"] = [revised_slot]
    spec.events = [{"turn": 6, "type": "correction", "field": "window", "value": [revised_slot]}]
    env = CallFlowEnv(spec)
    observation, _ = env.reset(seed=42)
    result = finish(env, observation)
    assert result[-1]["verdict"]["task_success"]
    writes = [e for e in env.state["audit"] if e["kind"] == "write"]
    assert len(writes) == 2 and writes[-1]["after"]["slot_id"] == revised_slot


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_time", float("nan")),
        ("max_time", float("inf")),
        ("max_time", True),
        ("max_time", 10**1000),
        ("max_time", 1.7e308),
        ("max_time", 0),
        ("max_turns", True),
        ("max_turns", 1.5),
        ("max_tool_calls", True),
        ("max_tool_calls", -1),
        ("seed", -1),
        ("seed", True),
    ],
)
def test_invalid_resource_configuration_is_rejected(field, value):
    data = generate("book_direct", 42).to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        ScenarioSpec.from_dict(data)


@pytest.mark.parametrize("latency", [-1, float("nan"), float("inf"), True, "slow", 10**1000, 1e308])
def test_bad_fault_latency_is_rejected_before_state_mutation(latency):
    env = CallFlowEnv(generate("book_direct", 42))
    env.reset(seed=42)
    before = env.state_hash()
    bad = {"tool": "create_booking", "kind": "timeout_after", "latency": latency}
    with pytest.raises(ValueError):
        env.inject_fault(bad)
    assert env.state_hash() == before
    spec = generate("book_direct", 42).to_dict()
    spec["faults"] = [bad]
    with pytest.raises(ValueError):
        ScenarioSpec.from_dict(spec)


def test_fault_metadata_cannot_poison_snapshot_json():
    env = CallFlowEnv(generate("book_direct", 42))
    env.reset(seed=42)
    original = env.state_hash()
    with pytest.raises(ValueError, match="JSON"):
        env.inject_fault({"tool": "get_booking", "kind": "latency", "metadata": float("nan")})
    assert env.state_hash() == original


def test_supported_horizon_keeps_inflight_clock_finite():
    spec = generate("book_direct", 42)
    spec.max_time = MAX_VIRTUAL_DURATION
    spec.faults = [{"tool": "list_services", "kind": "latency", "latency": MAX_VIRTUAL_DURATION}]
    env = CallFlowEnv(spec)
    env.reset(seed=42)
    _, _, terminated, truncated, _ = env.step(
        {"type": "CALL_TOOL", "tool": "list_services", "args": {}}
    )
    assert not terminated and truncated
    assert env.state["clock"] == MAX_VIRTUAL_DURATION + 1
    env.state_hash()
