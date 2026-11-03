"""Policies consume public observations only; evaluator metadata is never an input."""

from typing import Any

from callflow_rl.policies.baselines import HeuristicPolicy, RandomPolicy

__all__ = ["HeuristicPolicy", "RandomPolicy", "PPOPolicy"]


def __getattr__(name: str) -> Any:
    if name == "PPOPolicy":
        from callflow_rl.policies.ppo import PPOPolicy

        return PPOPolicy
    raise AttributeError(name)
