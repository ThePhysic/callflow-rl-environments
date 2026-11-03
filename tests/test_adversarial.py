"""Reward shortcuts and malformed/unauthorized policy behavior."""

from copy import deepcopy

import pytest

from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import generate
from callflow_rl.simulators.tools import SchedulingService
from callflow_rl.verifiers.audit import naive_score, run_audit, shortcut_action


def test_all_documented_reward_shortcuts_are_audited():
    report = run_audit()
    assert report["case_count"] == 9
    assert report["all_passed"], report


@pytest.mark.parametrize(
    "action",
    [
        None,
        [],
        "END",
        True,
        {},
        {"type": []},
        {"type": "UNKNOWN"},
        {"type": "END", "private_goal": {}},
        {"type": "ASK_CLARIFICATION", "field": []},
        {"type": "ASK_CLARIFICATION", "field": "private_goal"},
        {"type": "OFFER_SLOT", "slot_id": 42},
        {"type": "CALL_TOOL", "tool": "create_booking", "args": []},
        {"type": "CALL_TOOL", "tool": "delete_database", "args": {}},
        {"type": "CALL_TOOL", "tool": "get_booking", "args": {"customer_id": True}},
        {
            "type": "CALL_TOOL",
            "tool": "create_booking",
            "args": {
                "customer_id": "a",
                "slot_id": "s",
                "expected_version": True,
                "idempotency_key": "x",
            },
        },
        {"type": "MESSAGE_USER", "text": 13},
    ],
)
def test_invalid_actions_fail_deterministically_without_world_changes(action):
    first = CallFlowEnv(generate("book_direct", 461))
    second = CallFlowEnv(generate("book_direct", 461))
    first.reset(seed=461)
    second.reset(seed=461)
    before = deepcopy(first.state["world"])
    actual = first.step(action)
    assert actual == second.step(action)
    assert first.state["world"] == before
    assert first.state["invalid_action_count"] == 1
    assert not first.verdict()["task_success"]


def test_success_text_alone_cannot_complete_or_earn_completion_reward():
    env = CallFlowEnv(generate("book_direct", 98))
    env.reset(seed=98)
    action = shortcut_action()
    assert naive_score(action) == 10.0
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    assert reward < 0
    assert not info["verdict"]["task_success"]
    assert info["verdict"]["false_success_claim"]


@pytest.mark.parametrize(
    "text",
    [
        "Your appointment is booked.",
        "Confirmed!",
        "I reserved your appointment.",
        "Your booking is cancelled.",
    ],
)
def test_omitting_structured_claim_fields_does_not_hide_a_false_claim(text):
    env = CallFlowEnv(generate("book_direct", 98))
    env.reset(seed=98)
    env.step({"type": "MESSAGE_USER", "text": text})
    assert env.verdict()["false_success_claim"]


def test_looking_up_another_account_does_not_authenticate_or_mutate_it():
    scenario = generate("book_direct", 571)
    env = CallFlowEnv(scenario)
    env.reset(seed=571)
    other = next(
        c
        for cid, c in scenario.initial_world["customers"].items()
        if cid != scenario.private_goal["customer_id"]
    )
    observation, *_ = env.step(
        {
            "type": "CALL_TOOL",
            "tool": "lookup_customer",
            "args": {"account_code": other["account_code"]},
        }
    )
    assert observation["last_tool"]["error"] == "unauthorized_customer"
    assert env.state["authenticated_customer_id"] is None
    assert env.state["world"] == scenario.initial_world


def test_changed_payload_cannot_reuse_an_idempotency_key():
    scenario = generate("book_flexible", 307)
    env = CallFlowEnv(scenario)
    observation, _ = env.reset(seed=307)
    policy = HeuristicPolicy()
    write = None
    while not env.state["terminated"] and not env.state["truncated"]:
        action = policy.act(observation)
        observation, *_ = env.step(action)
        if action.get("tool") == "create_booking" and observation["last_tool"].get("ok"):
            write = deepcopy(action)
            break
    assert write is not None
    write["args"]["slot_id"] = scenario.initial_world["slots"][1]["id"]
    before = deepcopy(env.state["world"])
    response = SchedulingService(env.state).call(write["tool"], write["args"])
    assert response["error"] == "idempotency_key_reused"
    assert env.state["world"] == before


def test_stale_consent_is_rejected_after_correction():
    scenario = generate("book_flexible", 558)
    scenario.events = [
        {
            "turn": 4,
            "type": "correction",
            "field": "window",
            "value": [scenario.initial_world["slots"][1]["id"]],
        }
    ]
    env = CallFlowEnv(scenario)
    observation, _ = env.reset(seed=558)
    policy = HeuristicPolicy()
    for _ in range(4):
        observation, *_ = env.step(policy.act(observation))
    assert env.state["consent"] is None
    assert env.state["user_revision"] == 1
    slot = scenario.initial_world["slots"][0]
    response = SchedulingService(env.state).call(
        "create_booking",
        {
            "customer_id": scenario.private_goal["customer_id"],
            "slot_id": slot["id"],
            "expected_version": slot["version"],
            "idempotency_key": "stale-consent",
        },
    )
    assert response["error"] == "consent_required"
    assert not env.state["world"]["bookings"]


def test_repeated_queries_and_clarifications_cannot_farm_shaping_reward():
    env = CallFlowEnv(generate("book_all_missing", 609))
    env.reset(seed=609)
    rewards = [env.step({"type": "ASK_CLARIFICATION", "field": "service"})[1] for _ in range(16)]
    assert all(r < 0 for r in rewards[1:])
    assert sum(rewards) < 0


@pytest.mark.parametrize("limit", ["turn", "time", "tool"])
def test_budget_exhaustion_remains_a_measured_failure(limit):
    scenario = generate("book_direct", 112)
    if limit == "turn":
        scenario.max_turns = 2
    elif limit == "time":
        scenario.max_time = 2
    else:
        scenario.max_tool_calls = 2
    env = CallFlowEnv(scenario)
    env.reset(seed=112)
    action = (
        {"type": "CALL_TOOL", "tool": "list_services", "args": {}}
        if limit == "tool"
        else {"type": "WAIT"}
    )
    env.step(action)
    _, reward, terminated, truncated, info = env.step(action)
    assert truncated and not terminated
    assert reward < 0
    assert info["verdict"]["termination_reason"] == f"{limit}_limit"
    assert not info["verdict"]["task_success"]


def test_timestamp_alias_cannot_substitute_for_a_confirmed_slot_id():
    env = CallFlowEnv(generate("book_dst_fold", 111, "test"))
    observation, _ = env.reset(seed=111)
    env.step(
        {
            "type": "CALL_TOOL",
            "tool": "get_availability",
            "args": {
                "service": observation["facts"]["service"],
                "location": observation["facts"]["location"],
            },
        }
    )
    observation, *_ = env.step({"type": "OFFER_SLOT", "slot_id": "2026-11-01T01:30:00"})
    assert not observation["offer_accepted"]
    assert observation["consent"] is None
