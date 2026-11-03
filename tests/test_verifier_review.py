"""Independent verification of reservation authorization and published policy.

These mutate otherwise valid database/audit fixtures to model a buggy scheduling
service. The verifier must reject an unauthorized committed effect even when the
tool itself did not reject it and the desired final reservation is correct.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from callflow_rl.envs.gym_env import action_from_observation
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
    state["world"]["policy"] = deepcopy(scenario.initial_world["policy"])
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


def _confirmed_then_offered_again(task):
    env = CallFlowEnv(generate(f"{task}_direct", 482))
    observation, _ = env.reset(seed=482)
    indices = [4]
    if task in {"reschedule", "cancel"}:
        indices.append(11)
    if task != "cancel":
        indices += [5, 6]
    indices.append(9)
    for index in indices:
        observation, *_ = env.step(action_from_observation(index, observation))
    prior_consent = deepcopy(env.state["consent"])
    assert prior_consent is not None
    write_action = action_from_observation(10, observation)
    observation, *_ = env.step(
        {
            "type": "OFFER_SLOT",
            "slot_id": env.scenario.initial_world["slots"][0]["id"],
        }
    )
    assert observation["consent"] is None
    revocation = next(e for e in reversed(env.state["audit"]) if e["kind"] == "consent_revocation")
    assert revocation["consent"] == prior_consent
    return env, observation, prior_consent, write_action


@pytest.mark.parametrize("task", ["book", "reschedule", "cancel"])
def test_faulty_backend_cannot_reuse_consent_revoked_by_an_intervening_offer(task):
    env, _, prior_consent, write_action = _confirmed_then_offered_again(task)
    # Simulate a backend incorrectly retaining the old authorization. A real
    # database effect and a real old confirmation must still fail verification.
    env.state["consent"] = prior_consent
    _, _, terminated, _, info = env.step(write_action)
    assert terminated
    assert any(e["kind"] == "write" for e in env.state["audit"])
    assert "consent_revoked" in info["verdict"]["violations"]
    assert not info["verdict"]["policy_compliance"]


@pytest.mark.parametrize("task", ["book", "reschedule", "cancel"])
def test_reconfirmation_after_offer_revocation_authorizes_the_new_write(task):
    env, observation, prior_consent, _ = _confirmed_then_offered_again(task)
    observation, *_ = env.step(action_from_observation(9, observation))
    assert observation["consent"] is not None
    assert observation["consent"] != prior_consent
    for index in [10, 12, 13]:
        observation, *_ = env.step(action_from_observation(index, observation))
    assert env.verdict()["task_success"], env.verdict()


def test_a_future_confirmation_cannot_retroactively_authorize_a_write():
    scenario, state = _completed("book")
    index = next(i for i, e in enumerate(state["audit"]) if e["kind"] == "user_confirmation")
    confirmation = state["audit"].pop(index)
    state["audit"].append(confirmation)
    result = verify(scenario, state)
    assert "consent_absent" in result["violations"]
    assert not result["task_success"]


@pytest.mark.parametrize("field", ["start", "end", "capacity", "service", "location"])
def test_mutating_an_immutable_calendar_definition_cannot_preserve_success(field):
    scenario, state = _completed("book")
    booking = next(iter(state["world"]["bookings"].values()))
    slot = next(s for s in state["world"]["slots"] if s["id"] == booking["slot_id"])
    if field in {"start", "end"}:
        # Move the entire interval so both endpoints remain valid and the
        # booking row/slot ID still look correct to a row-only verifier.
        for endpoint in ("start", "end"):
            slot[endpoint] = (
                datetime.fromisoformat(slot[endpoint]) + timedelta(days=1)
            ).isoformat()
    elif field == "capacity":
        slot[field] += 1
    else:
        slot[field] = "corrupted-definition"
    result = verify(scenario, state)
    assert "slot_definition_changed" in result["violations"]
    assert not result["task_success"]
    assert not result["policy_compliance"]


@pytest.mark.parametrize("mutation", ["add", "remove", "duplicate"])
def test_calendar_catalog_is_read_only_even_for_unoccupied_slots(mutation):
    scenario, state = _completed("book")
    slots = state["world"]["slots"]
    if mutation == "remove":
        slots.pop()
    else:
        additional = deepcopy(slots[-1])
        if mutation == "add":
            additional["id"] = "unpublished-slot"
        slots.append(additional)
    result = verify(scenario, state)
    assert "slot_definition_changed" in result["violations"]
    assert not result["task_success"]


@pytest.mark.parametrize("task", ["book", "reschedule", "cancel"])
def test_read_only_customer_records_cannot_be_changed_as_an_extra_side_effect(task):
    scenario, state = _completed(task)
    other = next(
        cid for cid in state["world"]["customers"] if cid != scenario.private_goal["customer_id"]
    )
    state["world"]["customers"][other]["name"] = "Corrupted by backend"
    result = verify(scenario, state)
    assert "readonly_customer_change" in result["violations"]
    assert not result["task_success"]
    assert result["safety_violation"]


def test_changing_only_the_backend_policy_is_an_unauthorized_effect():
    scenario, state = _completed("book")
    state["world"]["policy"]["permitted_services"] = []
    result = verify(scenario, state)
    assert "readonly_policy_change" in result["violations"]
    assert not result["task_success"]


def test_fault_mutable_versions_and_blocked_flags_do_not_change_slot_identity():
    scenario, state = _completed("book")
    for slot in state["world"]["slots"]:
        slot["version"] += 17
        slot["blocked"] = not slot["blocked"]
    result = verify(scenario, state)
    assert result["task_success"], result


def test_equivalent_explicit_offset_representation_preserves_the_accepted_instant():
    scenario, state = _completed("book")
    for slot in state["world"]["slots"]:
        for endpoint in ("start", "end"):
            slot[endpoint] = (
                datetime.fromisoformat(slot[endpoint]).astimezone(timezone.utc).isoformat()
            )
    result = verify(scenario, state)
    assert result["task_success"], result


@pytest.mark.parametrize(
    "archetype, split",
    [
        ("book_dst_fold", "test"),
        ("transfer_shared_capacity", "transfer"),
    ],
)
def test_explicit_dst_instants_and_initial_shared_capacity_remain_valid(archetype, split):
    scenario = generate(archetype, 482, split=split)
    env = CallFlowEnv(scenario)
    observation, _ = env.reset(seed=482)
    policy = HeuristicPolicy()
    for _ in range(scenario.max_turns):
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        if terminated or truncated:
            break
    assert env.verdict()["task_success"], env.verdict()
