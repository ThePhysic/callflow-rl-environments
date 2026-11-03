"""Learning-contract tests: boundary math, real weight updates, and public masks."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from callflow_rl.envs.gym_env import encode
from callflow_rl.envs.native import CallFlowEnv
from callflow_rl.evaluation.statistics import bootstrap_seed_means, wilson
from callflow_rl.policies import HeuristicPolicy, RandomPolicy
from callflow_rl.policies.decisions import decision_mask
from callflow_rl.policies.ppo import PPOPolicy, generalized_advantages, train_ppo
from callflow_rl.scenarios import generate


def test_gae_bootstraps_truncation_without_cross_episode_leakage():
    advantages, returns = generalized_advantages(
        np.array([1, 10], dtype=np.float32),
        np.array([2, 3], dtype=np.float32),
        np.array([5, 999], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
        np.array([1, 1], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages, [3.5, 7])
    np.testing.assert_allclose(returns, [5.5, 10])


def test_policy_inputs_and_mask_do_not_depend_on_hidden_goal():
    spec = generate("book_direct", 67)
    altered = deepcopy(spec)
    altered.private_goal["allowed_slots"] = ["secret-wrong-slot"]
    original_env, changed_env = CallFlowEnv(spec), CallFlowEnv(altered)
    first, _ = original_env.reset(seed=67)
    second, _ = changed_env.reset(seed=67)
    assert first == second
    np.testing.assert_array_equal(encode(first), encode(second))
    np.testing.assert_array_equal(decision_mask(first), decision_mask(second))
    assert HeuristicPolicy().act(first) == HeuristicPolicy().act(second)
    assert RandomPolicy(13).act(first) == RandomPolicy(13).act(second)
    # Even spurious evaluator keys must not influence pure adapters.
    polluted = dict(first, private_goal={"success_action": 10}, task_success=True)
    np.testing.assert_array_equal(encode(first), encode(polluted))
    np.testing.assert_array_equal(decision_mask(first), decision_mask(polluted))


def test_interrupted_mask_has_only_yield_and_terminal_scaffolding_is_public():
    observation, _ = CallFlowEnv(generate("book_direct", 8)).reset(seed=8)
    assert not decision_mask(observation)[13]
    assert not decision_mask(observation)[14]
    assert decision_mask(dict(observation, reported=True))[13]
    interrupted = decision_mask(dict(observation, interrupted=True))
    assert list(np.flatnonzero(interrupted)) == [18]


def test_random_rng_is_local_and_repeatable():
    observation, _ = CallFlowEnv(generate("book_direct", 8)).reset(seed=8)
    one, two = RandomPolicy(91), RandomPolicy(91)
    assert [one.act(observation) for _ in range(50)] == [two.act(observation) for _ in range(50)]


def test_confidence_intervals_handle_boundary_and_seed_values():
    assert wilson(0, 10)[0] == 0
    assert 0.2 < wilson(0, 10)[1] < 0.4
    assert wilson(10, 10)[1] == pytest.approx(1)
    assert bootstrap_seed_means([0.25, 0.25, 0.25]) == [0.25, 0.25]


def test_actual_ppo_updates_weights_and_reloads_public_adapter(tmp_path):
    config = {
        "total_steps": 64,
        "rollout_steps": 32,
        "checkpoint_steps": [0, 64],
        "update_epochs": 2,
        "minibatch_size": 16,
        "hidden_size": 16,
        "evaluation_split": "dev",
        "torch_threads": 1,
    }
    result = train_ppo(config, tmp_path, seed=7)
    initial = torch.load(tmp_path / "checkpoint_000000.pt", weights_only=True)
    final = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert result["environment_interactions"] == 64
    assert any(
        not torch.equal(initial["model"][key], final["model"][key]) for key in initial["model"]
    )
    assert all(torch.isfinite(value).all() for value in final["model"].values())
    assert result["test_used_for_training_or_selection"] is False
    policy = PPOPolicy(tmp_path)
    observation, _ = CallFlowEnv(generate("book_direct", 8)).reset(seed=8)
    action = policy.act(observation)
    assert isinstance(action, dict) and "type" in action
    assert policy.act(observation) == action
