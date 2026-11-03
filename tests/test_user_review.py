"""Adversarial regressions for task-conditioned simulated user acceptance."""

from copy import deepcopy

import pytest

from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import generate


def available(env):
    facts = env.observation()["facts"]
    observation, *_ = env.step(
        {
            "type": "CALL_TOOL",
            "tool": "get_availability",
            "args": {"service": facts["service"], "location": facts["location"]},
        }
    )
    return observation["slots"][0]["id"]


def latest_check(env):
    return next(e for e in reversed(env.state["audit"]) if e["kind"] == "simulator_check")


def complete(env):
    policy = HeuristicPolicy()
    while not env.state["terminated"] and not env.state["truncated"]:
        env.step(policy.act(env.observation()))
    return env.verdict()


@pytest.mark.parametrize("candidate", ["available", "unknown"])
def test_cancellation_rejects_new_slot_without_changing_the_reservation(candidate):
    env = CallFlowEnv(generate("cancel_direct", 409))
    env.reset(seed=409)
    original = deepcopy(env.state["world"]["bookings"])
    slot_id = available(env) if candidate == "available" else "not-an-observed-slot"
    observation, _, terminated, truncated, _ = env.step({"type": "OFFER_SLOT", "slot_id": slot_id})
    assert not observation["offer_accepted"]
    assert observation["offered_slot"] is None
    assert observation["consent"] is None
    assert not observation["user_declined"]
    assert "cancel my existing reservation" in observation["user_message"]
    assert env.state["world"]["bookings"] == original
    assert env.state["invalid_action_count"] == 0
    assert latest_check(env)["consistent"]
    assert not terminated and not truncated
    verdict = complete(env)
    assert verdict["task_success"]
    assert verdict["simulator_inconsistency_count"] == 0


def test_irrelevant_offer_revokes_existing_cancellation_consent_and_allows_reconfirmation():
    env = CallFlowEnv(generate("cancel_direct", 410))
    env.reset(seed=410)
    policy = HeuristicPolicy()
    while not env.state["consent"]:
        env.step(policy.act(env.observation()))
    original = deepcopy(env.state["world"]["bookings"])
    old_consent = deepcopy(env.state["consent"])
    assert old_consent["task"] == "cancel"
    slot_id = available(env)
    env.step({"type": "OFFER_SLOT", "slot_id": slot_id})
    assert env.state["consent"] is None
    assert env.state["world"]["bookings"] == original
    assert latest_check(env)["consistent"]
    verdict = complete(env)
    assert verdict["task_success"]
    confirmations = [e for e in env.state["audit"] if e["kind"] == "user_confirmation"]
    assert len(confirmations) == 2
    assert confirmations[-1]["consent"]["turn"] > old_consent["turn"]


def test_independent_simulator_check_detects_forged_accepted_cancellation_offer():
    env = CallFlowEnv(generate("cancel_direct", 411))
    env.reset(seed=411)
    slot_id = available(env)
    env.state["offered_slot"] = slot_id
    env.state["offer_accepted"] = True
    assert env.state["consent"] is None
    env._check_user_consistency()
    assert not latest_check(env)["consistent"]
    assert env.verdict()["simulator_inconsistency_count"] == 1


@pytest.mark.parametrize("archetype", ["book_direct", "reschedule_direct"])
def test_booking_and_rescheduling_still_accept_and_confirm_a_permitted_slot(archetype):
    env = CallFlowEnv(generate(archetype, 412))
    env.reset(seed=412)
    slot_id = available(env)
    observation, *_ = env.step({"type": "OFFER_SLOT", "slot_id": slot_id})
    assert observation["offer_accepted"]
    assert observation["offered_slot"] == slot_id
    assert observation["consent"] is None
    assert latest_check(env)["consistent"]
    verdict = complete(env)
    assert verdict["task_success"]
    assert verdict["simulator_inconsistency_count"] == 0


@pytest.mark.parametrize(
    "mutation", ["missing_candidate", "wrong_window", "wrong_location", "decline"]
)
def test_independent_simulator_check_rejects_acceptance_inconsistent_with_user_truth(mutation):
    env = CallFlowEnv(generate("book_direct", 413))
    env.reset(seed=413)
    slot_id = available(env)
    env.step({"type": "OFFER_SLOT", "slot_id": slot_id})
    assert latest_check(env)["consistent"]
    if mutation == "missing_candidate":
        env.state["world"]["slots"] = []
    elif mutation == "wrong_window":
        env.state["user_truth"]["window"] = []
        env.state["facts"]["window"] = []
    elif mutation == "wrong_location":
        env.state["user_truth"]["location"] = "another-location"
        env.state["facts"]["location"] = "another-location"
    else:
        env.state["user_config"]["decline_all"] = True
    env._check_user_consistency()
    assert not latest_check(env)["consistent"]


@pytest.mark.parametrize("archetype", ["book_direct", "reschedule_direct"])
def test_acceptance_stays_consistent_when_booked_slot_leaves_availability(archetype):
    env = CallFlowEnv(generate(archetype, 416))
    env.reset(seed=416)
    policy = HeuristicPolicy()
    while not env.state["result_booking"]:
        env.step(policy.act(env.observation()))
    accepted_slot = env.state["offered_slot"]
    available(env)
    assert accepted_slot not in {s["id"] for s in env.state["slots"]}
    assert env.state["offer_accepted"]
    assert latest_check(env)["consistent"]
    assert complete(env)["task_success"]


def test_declined_booking_offer_and_corrected_reschedule_remain_consistent():
    declined = CallFlowEnv(generate("book_decline", 414))
    declined.reset(seed=414)
    declined.step({"type": "OFFER_SLOT", "slot_id": available(declined)})
    assert not declined.state["offer_accepted"]
    assert declined.state["user_declined"]
    assert latest_check(declined)["consistent"]
    corrected = CallFlowEnv(generate("reschedule_correction", 415, "test"))
    corrected.reset(seed=415)
    assert complete(corrected)["task_success"]
    assert corrected.verdict()["simulator_inconsistency_count"] == 0
