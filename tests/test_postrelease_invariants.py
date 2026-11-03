"""Native write isolation and independent verification of historical effects."""

from copy import deepcopy

import pytest

from callflow_rl.envs.gym_env import action_from_observation
from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import generate
from callflow_rl.verifiers import verify


def _complete(scenario):
    env = CallFlowEnv(scenario)
    observation, _ = env.reset()
    policy = HeuristicPolicy()
    while not env.state["terminated"] and not env.state["truncated"]:
        observation, *_ = env.step(policy.act(observation))
    assert env.verdict()["task_success"], env.verdict()
    return env.scenario, deepcopy(env.state)


def _existing_booking(scenario, booking_id, slot_index, *, same_customer):
    slot = scenario.initial_world["slots"][slot_index]
    customer_id = scenario.private_goal["customer_id"]
    if not same_customer:
        customer_id = next(cid for cid in scenario.initial_world["customers"] if cid != customer_id)
    booking = {
        "id": booking_id,
        "customer_id": customer_id,
        "service": slot["service"],
        "location": slot["location"],
        "slot_id": slot["id"],
        "status": "active",
        "version": 0,
    }
    scenario.initial_world["bookings"][booking_id] = booking
    return booking


@pytest.mark.parametrize("reserved", [[2], [2, 3]])
def test_each_create_skips_existing_sparse_booking_ids_without_overwriting_rows(reserved):
    scenario = generate("book_flexible", 42)
    for index, number in enumerate(reserved):
        _existing_booking(scenario, f"B{number:06d}", 2 + index, same_customer=False)
    original = deepcopy(scenario.initial_world["bookings"])
    env = CallFlowEnv(scenario)
    observation, _ = env.reset()
    for index in [4, 5, 6, 9, 10]:
        observation, *_ = env.step(action_from_observation(index, observation))
    assert observation["result_booking"]["id"] == "B000001"
    slot_id = scenario.initial_world["slots"][1]["id"]
    observation, *_ = env.step({"type": "OFFER_SLOT", "slot_id": slot_id})
    observation, *_ = env.step({"type": "CONFIRM_INTENT", "slot_id": slot_id})
    observation, *_ = env.step(
        {
            "type": "CALL_TOOL",
            "tool": "create_booking",
            "args": {
                "customer_id": observation["customer"]["id"],
                "slot_id": slot_id,
                "expected_version": 0,
                "idempotency_key": "second-approved-create",
            },
        }
    )
    assert observation["last_tool"]["ok"]
    assert observation["last_tool"]["booking"]["id"] == f"B{max(reserved) + 1:06d}"
    for booking_id, booking in original.items():
        assert env.state["world"]["bookings"][booking_id] == booking
    # Extra task effects still fail independently, without corrupting existing rows.
    assert env.verdict()["duplicate_side_effect"]
    assert "cross_customer_change" not in env.verdict()["violations"]


@pytest.mark.parametrize("task", ["book", "reschedule"])
def test_native_write_rejects_an_existing_same_customer_same_slot_booking(task):
    scenario = generate(f"{task}_direct", 42)
    _existing_booking(scenario, "already-at-destination", 0, same_customer=True)
    scenario.initial_world["slots"][0]["capacity"] = 2
    env = CallFlowEnv(scenario)
    observation, _ = env.reset()
    indices = [4, 11, 5, 6, 9] if task == "reschedule" else [4, 5, 6, 9]
    for index in indices:
        observation, *_ = env.step(action_from_observation(index, observation))
    original = deepcopy(env.state["world"])
    observation, _, terminated, truncated, _ = env.step(action_from_observation(10, observation))
    assert observation["last_tool"]["error"] == "duplicate_booking"
    assert env.state["world"] == original
    assert not any(event["kind"] == "write" for event in env.state["audit"])
    assert not terminated and not truncated


@pytest.mark.parametrize("task", ["book", "reschedule"])
@pytest.mark.parametrize("same_customer,slot_index", [(True, 2), (False, 0)])
def test_distinct_slots_and_shared_capacity_with_other_customers_remain_valid(
    task, same_customer, slot_index
):
    scenario = generate(f"{task}_direct", 42)
    _existing_booking(scenario, "existing-unrelated", slot_index, same_customer=same_customer)
    scenario.initial_world["slots"][slot_index]["capacity"] = 2
    _, state = _complete(scenario)
    assert (
        state["world"]["bookings"]["existing-unrelated"]
        == scenario.initial_world["bookings"]["existing-unrelated"]
    )


def test_reschedule_excludes_the_moved_reservation_from_its_duplicate_check():
    scenario = generate("reschedule_direct", 42)
    slot = scenario.initial_world["slots"][3]
    slot["capacity"] = 2
    scenario.user["facts"]["window"] = [slot["id"]]
    scenario.private_goal["allowed_slots"] = [slot["id"]]
    env = CallFlowEnv(scenario)
    observation, _ = env.reset()
    for index in [4, 11, 5]:
        observation, *_ = env.step(action_from_observation(index, observation))
    observation, *_ = env.step({"type": "OFFER_SLOT", "slot_id": slot["id"]})
    for index in [9, 10, 12, 13]:
        observation, *_ = env.step(action_from_observation(index, observation))
    assert env.verdict()["task_success"], env.verdict()
    assert len(env.state["world"]["bookings"]) == 1


@pytest.mark.parametrize("task", ["book", "reschedule"])
def test_verifier_rejects_duplicate_with_a_preexisting_reservation_independently(task):
    scenario, state = _complete(generate(f"{task}_direct", 42))
    existing = _existing_booking(scenario, "already-at-destination", 0, same_customer=True)
    state["world"]["bookings"][existing["id"]] = deepcopy(existing)
    scenario.initial_world["slots"][0]["capacity"] = 2
    state["world"]["slots"][0]["capacity"] = 2
    result = verify(scenario, state)
    assert result["duplicate_side_effect"]
    assert result["violations"] == ["duplicate_side_effect"]
    assert not result["task_success"]


def _intermediate_reschedule():
    scenario, state = _complete(generate("reschedule_direct", 42))
    write_index = next(i for i, event in enumerate(state["audit"]) if event["kind"] == "write")
    final_write = state["audit"][write_index]
    intermediate = deepcopy(final_write)
    slot_id = scenario.initial_world["slots"][1]["id"]
    intermediate["after"]["slot_id"] = slot_id
    intermediate["consent"]["slot_id"] = slot_id
    intermediate["consent"]["turn"] = 1
    intermediate["turn"] = 2
    intermediate["time"] = 2.0
    intermediate["idempotency_key"] = "intermediate-move"
    final_write["before"] = deepcopy(intermediate["after"])
    final_write["after"]["version"] += 1
    state["world"]["bookings"][final_write["after"]["id"]]["version"] += 1
    for event in state["audit"]:
        if event["kind"] == "claim":
            event["actual_booking"]["version"] += 1
            event["observed_booking"]["version"] += 1
    state["audit"][write_index:write_index] = [
        {
            "kind": "user_confirmation",
            "consent": deepcopy(intermediate["consent"]),
            "turn": 1,
            "time": 1.0,
        },
        intermediate,
    ]
    return scenario, state


def test_verifier_remembers_a_transient_same_customer_duplicate_after_the_final_move():
    scenario, state = _intermediate_reschedule()
    existing = _existing_booking(scenario, "already-at-intermediate", 1, same_customer=True)
    state["world"]["bookings"][existing["id"]] = deepcopy(existing)
    for slots in [scenario.initial_world["slots"], state["world"]["slots"]]:
        slots[1]["capacity"] = 2
    result = verify(scenario, state)
    assert result["violations"] == ["duplicate_side_effect"]
    assert result["duplicate_side_effect"]
    assert not result["task_success"]


@pytest.mark.parametrize("invalid_time", ["zero_length", "backwards", "no_offset", "malformed"])
def test_verifier_remembers_invalid_intermediate_time_after_the_final_move(invalid_time):
    scenario, state = _intermediate_reschedule()
    for slots in [scenario.initial_world["slots"], state["world"]["slots"]]:
        slot = slots[1]
        if invalid_time == "zero_length":
            slot["end"] = slot["start"]
        elif invalid_time == "backwards":
            slot["start"], slot["end"] = slot["end"], slot["start"]
        elif invalid_time == "no_offset":
            slot["start"] = "2026-10-06T13:00:00"
            slot["end"] = "2026-10-06T13:30:00"
        else:
            slot["start"] = "not-a-timestamp"
    result = verify(scenario, state)
    assert result["violations"] == ["invalid_booking_time"]
    assert not result["task_success"]


def test_valid_intermediate_move_still_passes():
    scenario, state = _intermediate_reschedule()
    assert verify(scenario, state)["task_success"]


@pytest.mark.parametrize("invalid_time", ["zero_length", "no_offset"])
def test_verifier_allows_cancelling_an_initial_invalid_slot_without_new_invalid_effects(
    invalid_time,
):
    scenario, state = _complete(generate("cancel_direct", 42))
    for slots in [scenario.initial_world["slots"], state["world"]["slots"]]:
        slot = slots[3]
        if invalid_time == "zero_length":
            slot["end"] = slot["start"]
        else:
            slot["start"] = "2026-10-06T13:00:00"
            slot["end"] = "2026-10-06T13:30:00"
    assert verify(scenario, state)["task_success"]


@pytest.mark.parametrize("task", ["book", "reschedule", "cancel"])
@pytest.mark.parametrize("noninteger", [False, True, 0.0, 1.0])
def test_verifier_requires_canonical_integer_versions_after_every_write(task, noninteger):
    scenario, state = _complete(generate(f"{task}_direct", 42))
    write = next(event for event in state["audit"] if event["kind"] == "write")
    write["after"]["version"] = noninteger
    state["world"]["bookings"][write["after"]["id"]]["version"] = noninteger
    for event in state["audit"]:
        if event["kind"] == "claim":
            event["actual_booking"]["version"] = noninteger
            event["observed_booking"]["version"] = noninteger
    result = verify(scenario, state)
    assert "invalid_write_transition" in result["violations"]
    assert not result["task_success"]


@pytest.mark.parametrize("noninteger", [True, 1.0])
def test_equal_numeric_capacity_values_cannot_change_the_published_capacity_type(noninteger):
    scenario, state = _complete(generate("book_direct", 42))
    state["world"]["slots"][0]["capacity"] = noninteger
    result = verify(scenario, state)
    assert "slot_definition_changed" in result["violations"]
    assert not result["task_success"]
