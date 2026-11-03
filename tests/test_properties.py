"""Random action sequences exercise state invariants rather than implementation copies."""

import json
from copy import deepcopy

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from callflow_rl.envs.gym_env import N_ACTIONS, action_from_observation
from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.policies import HeuristicPolicy
from callflow_rl.scenarios import ARCHETYPES, generate
from callflow_rl.simulators.tools import SchedulingService


def assert_world_invariants(env):
    world = env.state["world"]
    goal = env.scenario.private_goal
    active = [b for b in world["bookings"].values() if b["status"] == "active"]
    for slot in world["slots"]:
        assert sum(b["slot_id"] == slot["id"] for b in active) <= slot["capacity"]
    for booking_id, original in env.scenario.initial_world["bookings"].items():
        if original["customer_id"] != goal["customer_id"]:
            assert world["bookings"][booking_id] == original
    for event in env.state["audit"]:
        if event["kind"] == "write":
            assert event["customer_id"] == goal["customer_id"]
            assert event["consent"]
            assert any(
                e["kind"] == "user_confirmation" and e["consent"] == event["consent"]
                for e in env.state["audit"]
            )


class AppointmentMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.env = CallFlowEnv(generate("book_flexible", 0))
        self.env.reset(seed=0)
        self.policy = HeuristicPolicy()

    @rule(
        archetype=st.sampled_from(list(ARCHETYPES)), seed=st.integers(min_value=0, max_value=10_000)
    )
    def new_episode(self, archetype, seed):
        scenario = generate(archetype, seed, ARCHETYPES[archetype]["split"])
        self.env = CallFlowEnv(scenario)
        self.env.reset(seed=seed)

    @rule(index=st.integers(min_value=0, max_value=N_ACTIONS - 1))
    def arbitrary_public_decision(self, index):
        if not self.env.state["terminated"] and not self.env.state["truncated"]:
            observation = self.env.observation()
            self.env.step(action_from_observation(index, observation))

    @rule()
    def follow_public_evidence(self):
        if not self.env.state["terminated"] and not self.env.state["truncated"]:
            self.env.step(self.policy.act(self.env.observation()))

    @rule()
    def retry_a_committed_request_without_duplicate_effect(self):
        prior = next(iter(self.env.state["idempotency"].values()), None)
        if prior:
            request = deepcopy(prior["request"])
            key = next(k for k, v in self.env.state["idempotency"].items() if v == prior)
            request["args"]["idempotency_key"] = key
            before = deepcopy(self.env.state["world"])
            writes = len([e for e in self.env.state["audit"] if e["kind"] == "write"])
            result = SchedulingService(self.env.state).call(request["tool"], request["args"])
            assert result["idempotent_replay"]
            assert self.env.state["world"] == before
            assert len([e for e in self.env.state["audit"] if e["kind"] == "write"]) == writes

    @rule()
    def snapshot_roundtrip_preserves_future(self):
        snapshot = json.loads(json.dumps(self.env.snapshot()))
        restored = CallFlowEnv()
        restored.restore(snapshot)
        assert restored.state_hash() == self.env.state_hash()
        if not self.env.state["terminated"] and not self.env.state["truncated"]:
            action = self.policy.act(self.env.observation())
            assert restored.step(action) == self.env.step(action)

    @invariant()
    def capacities_permissions_and_confirmation_are_preserved(self):
        assert_world_invariants(self.env)


TestAppointmentStateMachine = AppointmentMachine.TestCase
TestAppointmentStateMachine.settings = settings(
    max_examples=35, stateful_step_count=35, deadline=None
)


@given(
    seed=st.integers(min_value=0, max_value=5000),
    indices=st.lists(st.integers(min_value=0, max_value=N_ACTIONS - 1), min_size=1, max_size=25),
)
@settings(max_examples=40, deadline=None)
def test_many_seeded_action_sequences_replay_exactly(seed, indices):
    scenario = generate("book_timeout_after", seed)
    first, second = CallFlowEnv(scenario), CallFlowEnv(scenario)
    observation, first_info = first.reset(seed=seed)
    assert second.reset(seed=seed) == (observation, first_info)
    for index in indices:
        action = action_from_observation(index, observation)
        result = first.step(action)
        assert result == second.step(action)
        observation = result[0]
        assert first.state_hash() == second.state_hash()
        if result[2] or result[3]:
            break


@given(seed=st.integers(min_value=0, max_value=5000))
@settings(max_examples=20, deadline=None)
def test_episode_mutation_and_rng_consumption_are_isolated(seed):
    scenario = generate("book_all_missing", seed)
    first, second, control = (CallFlowEnv(scenario) for _ in range(3))
    first.reset(seed=seed)
    second.reset(seed=seed)
    control.reset(seed=seed)
    for _ in range(5):
        first.step({"type": "ASK_CLARIFICATION", "field": "account_code"})
    first.state["world"]["slots"][0]["capacity"] = 99
    assert second.state_hash() == control.state_hash()
    action = {"type": "ASK_CLARIFICATION", "field": "location"}
    assert second.step(action) == control.step(action)
    assert scenario.initial_world["slots"][0]["capacity"] == 1


def test_simulated_acceptance_agrees_with_final_goal_after_scheduled_corrections():
    for archetype, metadata in ARCHETYPES.items():
        for seed in (6, 91, 903):
            scenario = generate(archetype, seed, metadata["split"])
            env = CallFlowEnv(scenario)
            env.reset(seed=seed)
            for _ in range(max((e["turn"] for e in scenario.events), default=0)):
                env.step({"type": "WAIT"})
            env.state["slots"] = deepcopy(env.state["world"]["slots"])
            for slot in env.state["slots"]:
                env._user().offer(slot["id"])
                if env.state["offer_accepted"]:
                    assert slot["id"] in scenario.private_goal["allowed_slots"]
                    assert slot["location"] in scenario.private_goal["allowed_locations"]
                    assert slot["service"] == scenario.private_goal["service"]


def test_clarification_after_correction_remembers_the_correction():
    scenario = generate("book_correction_time", 445, "test")
    env = CallFlowEnv(scenario)
    env.reset(seed=445)
    for _ in range(3):
        env.step({"type": "WAIT"})
    for _ in range(3):
        observation, *_ = env.step({"type": "ASK_CLARIFICATION", "field": "window"})
        assert observation["facts"]["window"] == scenario.private_goal["allowed_slots"]
    assert env.state["user_revision"] == 1
