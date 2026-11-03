"""End-to-end contracts, golden obligations and invariant-oriented negative cases."""

import json
from copy import deepcopy
from importlib.resources import files

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from callflow_rl.envs.gym_env import CallFlowGymEnv, action_from_observation, encode
from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import ARCHETYPES, generate
from callflow_rl.verifiers import verify


def run_heuristic(scenario, reward_mode="shaped"):
    env = CallFlowEnv(scenario, reward_mode=reward_mode)
    observation, _ = env.reset(seed=scenario.seed)
    policy = HeuristicPolicy()
    transitions = []
    for _ in range(scenario.max_turns):
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        transitions.append((action, reward, info))
        if terminated or truncated:
            break
    return env, transitions


@pytest.mark.parametrize("archetype", list(ARCHETYPES))
def test_authored_archetype_reaches_correct_final_outcome(archetype):
    scenario = generate(archetype, 1009, ARCHETYPES[archetype]["split"])
    env, transitions = run_heuristic(scenario)
    verdict = env.verdict()
    outcome = scenario.private_goal["outcome"]
    field = {
        "completed": "task_success",
        "handoff": "justified_handoff",
        "refused": "justified_refusal",
    }[outcome]
    assert verdict[field], [(a, i["verdict"]["violations"]) for a, _, i in transitions]
    assert verdict["policy_compliance"]
    assert not verdict["duplicate_side_effect"]
    assert not env.state["truncated"]


GOLDEN_CASES = json.loads(
    files("callflow_rl.scenarios").joinpath("fixtures", "golden_cases.json").read_text("utf-8")
)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["archetype"] for c in GOLDEN_CASES])
def test_ten_golden_obligations(case):
    scenario = generate(case["archetype"], case["seed"], case["split"])
    env, _ = run_heuristic(scenario)
    result = env.verdict()
    field = {
        "completed": "task_success",
        "handoff": "justified_handoff",
        "refused": "justified_refusal",
    }[case["expected"]]
    assert result[field]


def test_gymnasium_contract_checker():
    check_env(CallFlowGymEnv(generate("book_direct", 83)), skip_render_check=True)


@pytest.mark.parametrize("seed", range(12))
def test_native_and_gym_decisions_have_identical_transition_semantics(seed):
    scenario = generate("book_timeout_after", seed)
    native = CallFlowEnv(scenario)
    gym_env = CallFlowGymEnv(scenario)
    public, native_info = native.reset(seed=seed)
    numeric, gym_info = gym_env.reset(seed=seed)
    assert native_info == gym_info
    np.testing.assert_array_equal(numeric, encode(public))
    rng = np.random.default_rng(seed)
    for index in rng.integers(0, gym_env.action_space.n, 12):
        assert gym_env.action_for(int(index)) == action_from_observation(int(index), public)
        public, nr, nt, nx, ni = native.step(action_from_observation(int(index), public))
        numeric, gr, gt, gx, gi = gym_env.step(int(index))
        np.testing.assert_array_equal(numeric, encode(public))
        assert (nr, nt, nx, ni) == (gr, gt, gx, gi)
        if nt or nx:
            break


def test_policy_receives_a_detached_observation_not_backend_or_answer_key():
    env = CallFlowEnv(generate("book_timeout_after", 900))
    observation, _ = env.reset(seed=900)
    forbidden = {
        "private_goal",
        "initial_world",
        "faults",
        "events",
        "audit",
        "allowed_slots",
        "allowed_locations",
        "task_success",
        "verdict",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()

    assert not keys(observation) & forbidden
    before = env.state_hash()
    observation["facts"]["service"] = "hacked"
    observation["policy"]["handoff_required"] = True
    observation["available_tools"]["admin_delete_everything"] = {}
    assert env.state_hash() == before
    assert "admin_delete_everything" not in env.observation()["available_tools"]


def test_verifier_refuses_agent_success_flags_and_partial_completion():
    scenario = generate("book_direct", 84)
    env = CallFlowEnv(scenario)
    env.reset(seed=84)
    env.state["success"] = True
    env.state["reported"] = True
    env.step({"type": "END"})
    assert not env.verdict()["task_success"]
    complete_env, _ = run_heuristic(scenario)
    partial = deepcopy(complete_env.state)
    partial["audit"] = [e for e in partial["audit"] if e["kind"] != "claim"]
    assert not verify(scenario, partial)["task_success"]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_customer",
        "wrong_slot",
        "consent_absent",
        "double_write",
        "false_claim",
        "unaudited_change",
    ],
)
def test_independent_verifier_rejects_corrupt_final_database_or_history(mutation):
    scenario = generate("book_direct", 631)
    env, _ = run_heuristic(scenario)
    assert env.verdict()["task_success"]
    state = deepcopy(env.state)
    booking = next(iter(state["world"]["bookings"].values()))
    if mutation == "wrong_customer":
        booking["customer_id"] = next(
            cid
            for cid in state["world"]["customers"]
            if cid != scenario.private_goal["customer_id"]
        )
    elif mutation == "wrong_slot":
        booking["slot_id"] = state["world"]["slots"][1]["id"]
    elif mutation == "consent_absent":
        state["audit"] = [e for e in state["audit"] if e["kind"] != "user_confirmation"]
    elif mutation == "double_write":
        extra = deepcopy(booking)
        extra["id"] = "B-extra"
        extra["slot_id"] = state["world"]["slots"][1]["id"]
        state["world"]["bookings"][extra["id"]] = extra
    elif mutation == "false_claim":
        for event in state["audit"]:
            if event["kind"] == "claim":
                event["actual_booking"] = None
    else:
        booking["version"] += 1
    verdict = verify(scenario, state)
    assert not verdict["task_success"]
    assert verdict["safety_violation"]
    if mutation == "double_write":
        assert verdict["duplicate_side_effect"]


def test_final_state_can_be_wrong_without_agent_claim_redefining_the_goal():
    scenario = generate("book_direct", 688)
    env, _ = run_heuristic(scenario)
    state = deepcopy(env.state)
    wrong_slot = state["world"]["slots"][1]["id"]
    for row in state["world"]["bookings"].values():
        row["slot_id"] = wrong_slot
    for event in state["audit"]:
        if event["kind"] in {"write", "user_confirmation"}:
            event["consent"]["slot_id"] = wrong_slot
        if event["kind"] == "write":
            event["after"]["slot_id"] = wrong_slot
        if event["kind"] == "claim":
            event["actual_booking"]["slot_id"] = wrong_slot
    verdict = verify(scenario, state)
    assert not verdict["task_success"]
    assert not verdict["goal_satisfied"]


def test_reschedule_failure_retains_original_exactly_and_cancel_is_idempotent():
    unavailable = generate("reschedule_no_capacity", 939, "dev")
    env, _ = run_heuristic(unavailable)
    assert env.state["world"]["bookings"] == unavailable.initial_world["bookings"]
    scenario = generate("cancel_direct", 940)
    env, transitions = run_heuristic(scenario)
    write = next(a for a, _, _ in transitions if a.get("tool") == "cancel_booking")
    # The scheduling service independently guarantees identical retry semantics.
    from callflow_rl.simulators.tools import SchedulingService

    before = deepcopy(env.state["world"])
    audit_count = len([e for e in env.state["audit"] if e["kind"] == "write"])
    result = SchedulingService(env.state).call(write["tool"], write["args"])
    assert result["ok"] and result["idempotent_replay"]
    assert env.state["world"] == before
    assert len([e for e in env.state["audit"] if e["kind"] == "write"]) == audit_count
