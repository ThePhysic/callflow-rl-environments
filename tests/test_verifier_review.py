"""Independent verification of reservation authorization and published policy.

These mutate otherwise valid database/audit fixtures to model a buggy scheduling
service. The verifier must reject an unauthorized committed effect even when the
tool itself did not reject it and the desired final reservation is correct.
"""

from copy import deepcopy

import pytest

from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import generate
from callflow_rl.verifiers import verify


def _completed(task, *, extra_reservation=False):
    scenario = generate(f"{task}_direct", 381)
    if extra_reservation:
        original = next(iter(scenario.initial_world["bookings"].values()))
        extra = deepcopy(original)
        extra.update(
            id="unapproved-existing-reservation", slot_id=scenario.initial_world["slots"][2]["id"]
        )
        scenario.initial_world["bookings"][extra["id"]] = extra
        if task == "reschedule":
            scenario.initial_world["slots"][0]["capacity"] = 2
    env = CallFlowEnv(scenario)
    obs, _ = env.reset(seed=381)
    policy = HeuristicPolicy()
    for _ in range(scenario.max_turns):
        obs, _, terminated, truncated, _ = env.step(policy.act(obs))
        if terminated or truncated:
            break
    assert env.verdict()["task_success"], env.verdict()
    return env.scenario, deepcopy(env.state)


@pytest.mark.parametrize("task", ["cancel", "reschedule"])
def test_real_target_consent_cannot_authorize_an_extra_same_customer_reservation(task):
    scenario, state = _completed(task, extra_reservation=True)
    extra = state["world"]["bookings"]["unapproved-existing-reservation"]
    authorized_write = next(e for e in state["audit"] if e["kind"] == "write")
    unapproved_write = deepcopy(authorized_write)
    unapproved_write["before"] = deepcopy(extra)
    unapproved_write["after"]["id"] = extra["id"]
    unapproved_write["turn"] += 1
    unapproved_write["idempotency_key"] = "rogue-second-write"
    state["world"]["bookings"][extra["id"]] = deepcopy(unapproved_write["after"])
    state["audit"].append(unapproved_write)
    result = verify(scenario, state)
    assert not result["task_success"]
    assert not result["policy_compliance"]
    assert result["safety_violation"]
    assert {"consent_booking_mismatch", "unauthorized_booking_target"} <= set(result["violations"])
    assert "audit_chain_mismatch" not in result["violations"]


@pytest.mark.parametrize("task", ["cancel", "reschedule"])
def test_initial_target_and_consent_identity_are_independently_bound(task):
    scenario, state = _completed(task, extra_reservation=True)
    goal_target = scenario.private_goal["booking_id"]
    # Even a self-consistent consent for the wrong reservation cannot authorize
    # changing the episode's private target to that other reservation.
    wrong = "unapproved-existing-reservation"
    for event in state["audit"]:
        if event["kind"] in {"write", "user_confirmation"}:
            event["consent"]["booking_id"] = wrong
    result = verify(scenario, state)
    assert scenario.private_goal["booking_id"] == goal_target
    assert not result["task_success"]
    assert "consent_booking_mismatch" in result["violations"]


@pytest.mark.parametrize("task", ["cancel", "reschedule"])
@pytest.mark.parametrize("mutation", ["identity", "customer", "service", "version", "status"])
def test_write_record_transitions_do_not_inherit_authority_from_valid_consent(task, mutation):
    scenario, state = _completed(task)
    write = next(e for e in state["audit"] if e["kind"] == "write")
    if mutation == "identity":
        write["before"]["id"] = "different-reservation"
        expected = "write_identity_changed"
    elif mutation == "customer":
        write["after"]["customer_id"] = "different-customer"
        expected = "write_customer_mismatch"
    elif mutation == "service":
        for event in state["audit"]:
            if event["kind"] in {"write", "user_confirmation"}:
                event["consent"]["service"] = "unapproved-service"
        expected = "consent_service_mismatch"
    elif mutation == "version":
        write["after"]["version"] += 1
        expected = "invalid_write_transition"
    else:
        write["after"]["status"] = "active" if task == "cancel" else "cancelled"
        expected = "invalid_write_transition"
    result = verify(scenario, state)
    assert expected in result["violations"]
    assert not result["policy_compliance"]


@pytest.mark.parametrize("task", ["book", "reschedule", "cancel"])
def test_published_handoff_policy_is_independent_of_backend_policy(task):
    scenario, state = _completed(task)
    scenario.initial_world["policy"]["handoff_required"] = True
    assert not state["world"]["policy"]["handoff_required"]
    result = verify(scenario, state)
    assert not result["task_success"]
    assert "policy_handoff_required" in result["violations"]


@pytest.mark.parametrize("task", ["book", "reschedule"])
@pytest.mark.parametrize(
    "policy_field, violation",
    [
        ("permitted_services", "policy_service_forbidden"),
        ("permitted_locations", "policy_location_forbidden"),
    ],
)
def test_published_booking_policy_is_checked_even_with_real_approval_and_valid_goal_state(
    task, policy_field, violation
):
    scenario, state = _completed(task)
    scenario.initial_world["policy"][policy_field] = []
    # The simulated faulty service's mutable policy still permits the write.
    assert state["world"]["policy"].get(policy_field) != []
    result = verify(scenario, state)
    assert not result["task_success"]
    assert not result["policy_compliance"]
    assert violation in result["violations"]


def test_cancellation_remains_allowed_when_service_or_location_is_no_longer_bookable():
    scenario, state = _completed("cancel")
    scenario.initial_world["policy"].update(permitted_services=[], permitted_locations=[])
    result = verify(scenario, state)
    # This matches SchedulingService: booking restrictions do not prohibit
    # cancelling an existing reservation; an explicit handoff rule still does.
    assert result["task_success"]
    assert result["policy_compliance"]


def test_cancellation_cannot_change_appointment_details_as_a_hidden_side_effect():
    scenario, state = _completed("cancel")
    write = next(e for e in state["audit"] if e["kind"] == "write")
    after = write["after"]
    after["slot_id"] = scenario.initial_world["slots"][1]["id"]
    state["world"]["bookings"][after["id"]] = deepcopy(after)
    result = verify(scenario, state)
    assert "invalid_write_transition" in result["violations"]
    assert "audit_chain_mismatch" not in result["violations"]
    assert not result["task_success"]


def test_forbidden_intermediate_write_is_rejected_after_a_correct_final_move():
    scenario, state = _completed("reschedule")
    write_index = next(i for i, e in enumerate(state["audit"]) if e["kind"] == "write")
    final_write = state["audit"][write_index]
    intermediate = deepcopy(final_write)
    alternate = scenario.initial_world["slots"][1]
    alternate["location"] = "restricted-location"
    scenario.initial_world["policy"]["permitted_locations"] = [final_write["after"]["location"]]
    intermediate["after"].update(slot_id=alternate["id"], location=alternate["location"])
    intermediate["consent"]["slot_id"] = alternate["id"]
    intermediate["consent"]["turn"] = 1
    intermediate["turn"] = 2
    intermediate["idempotency_key"] = "forbidden-intermediate"
    final_write["before"] = deepcopy(intermediate["after"])
    final_write["after"]["version"] += 1
    state["world"]["bookings"][final_write["after"]["id"]]["version"] += 1
    # Keep the report's copied final row consistent with the final database.
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
    result = verify(scenario, state)
    assert "policy_location_forbidden" in result["violations"]
    assert "audit_chain_mismatch" not in result["violations"]
    assert not result["task_success"]


def test_confirmed_intermediate_move_is_not_constrained_by_final_goal_slot():
    scenario, state = _completed("reschedule")
    write_index = next(i for i, e in enumerate(state["audit"]) if e["kind"] == "write")
    final_write = state["audit"][write_index]
    intermediate = deepcopy(final_write)
    intermediate["after"]["slot_id"] = scenario.initial_world["slots"][1]["id"]
    intermediate["consent"]["slot_id"] = intermediate["after"]["slot_id"]
    intermediate["consent"]["turn"] = 1
    intermediate["turn"] = 2
    intermediate["idempotency_key"] = "authorized-intermediate"
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
    result = verify(scenario, state)
    assert intermediate["after"]["slot_id"] not in scenario.private_goal["allowed_slots"]
    assert result["task_success"], result
