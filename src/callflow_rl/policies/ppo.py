"""Compact CPU PPO over public numeric observations and bounded native actions.

This is weight-updating clipped PPO, not imitation learning or prompt tuning.
Only terminal flags and scalar rewards cross the evaluator/trainer boundary.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from callflow_rl.envs.gym_env import action_from_observation, encode
from callflow_rl.policies.decisions import MASK_VERSION, decision_mask


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int = 64, action_size: int = 20):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(observation)
        return self.actor(features), self.critic(features).squeeze(-1)


class PPOPolicy:
    """Frozen policy adapter. `act` accepts the same observation as other baselines."""

    def __init__(self, checkpoint: str | Path | None = None, *, model: ActorCritic | None = None):
        if checkpoint is not None:
            path = Path(checkpoint)
            if path.is_dir():
                path = path / "checkpoint.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = ActorCritic(saved["observation_size"], saved["hidden_size"])
            model.load_state_dict(saved["model"])
            self.version = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            self.version = "ppo-live"
        if model is None:
            raise ValueError("checkpoint or model is required")
        self.model = model.eval()

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        return action_from_observation(self.select_index(observation), observation)

    def select_index(self, observation: dict[str, Any]) -> int:
        with torch.no_grad():
            logits, _ = self.model(torch.as_tensor(encode(observation), dtype=torch.float32))
            logits = logits.masked_fill(~torch.as_tensor(decision_mask(observation)), -1e9)
            return int(logits.argmax().item())


def generalized_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    ended: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap truncations, but stop recurrence across reset boundaries.

    `next_values[t]` is from the final observation before an episode reset.
    `ended` is terminated OR truncated. Treating truncation as zero value biases
    the critic; carrying recurrence over the reset mixes unrelated episodes.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        last = float(delta + gamma * gae_lambda * (1 - ended[t]) * last)
        advantages[t] = last
    return advantages, advantages + values


def train_ppo(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    seed: int,
    variant: str = "shaped",
) -> dict[str, Any]:
    """Train on generated training archetypes only; fixed dev checkpoints are logged.

    The final checkpoint is always the pre-registered final interaction budget.
    Test and transfer suites cannot influence checkpoint selection.
    """
    from callflow_rl.envs.native import CallFlowEnv
    from callflow_rl.evaluation.metrics import evaluate_policy
    from callflow_rl.scenarios import generate, manifest_hash
    from callflow_rl.scenarios.catalog import ARCHETYPES, archetypes_for

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    chooser = random.Random(seed)
    archetypes = archetypes_for("train")
    if variant == "no_fault_training":
        archetypes = [a for a in archetypes if "fault" not in ARCHETYPES[a]["slices"]]
    if not archetypes:
        raise ValueError("no training archetypes selected")
    reward_mode = "sparse" if variant == "sparse" else config.get("reward_mode", "shaped")
    total_steps = int(config.get("total_steps", 32768))
    rollout_steps = int(config.get("rollout_steps", 512))
    checkpoints = sorted(set(int(s) for s in config.get("checkpoint_steps", [0, total_steps])))
    if checkpoints[-1] != total_steps or any(s % rollout_steps for s in checkpoints):
        raise ValueError("checkpoint steps must include total_steps and align with rollout_steps")
    if total_steps < rollout_steps or total_steps % rollout_steps:
        raise ValueError("total_steps must be a positive multiple of rollout_steps")
    eval_split = config.get("evaluation_split", "dev")
    if eval_split != "dev":
        raise ValueError("training checkpoint evaluation is restricted to development split")

    def new_episode() -> tuple[Any, dict[str, Any]]:
        episode_seed = chooser.randrange(2**31)
        spec = generate(chooser.choice(archetypes), episode_seed, split="train")
        environment = CallFlowEnv(spec, reward_mode=reward_mode)
        observation, _ = environment.reset(seed=episode_seed)
        return environment, observation

    env, observation = new_episode()
    model = ActorCritic(len(encode(observation)), int(config.get("hidden_size", 64)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 5e-4)))
    metadata: dict[str, Any] = {
        "seed": seed,
        "variant": variant,
        "config": config,
        "training_archetypes": archetypes,
        "training_manifest_hash": manifest_hash("train"),
        "development_manifest_hash": manifest_hash("dev"),
        "test_used_for_training_or_selection": False,
        "checkpoint_selection": "fixed_final_budget",
        "algorithm": "clipped_ppo_gae",
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "device": "cpu",
        "public_action_mask": MASK_VERSION,
        "deterministic_algorithms": True,
    }
    torch.use_deterministic_algorithms(True)
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    curves: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    completed_episodes = 0
    total_reward = 0.0
    start = time.perf_counter()

    def save_checkpoint(step: int) -> None:
        saved = {
            "model": model.state_dict(),
            "observation_size": len(encode(observation)),
            "hidden_size": int(config.get("hidden_size", 64)),
            "seed": seed,
            "steps": step,
            "variant": variant,
            "config": config,
        }
        torch.save(saved, output / f"checkpoint_{step:06d}.pt")
        result = evaluate_policy(
            PPOPolicy(model=model),
            split="dev",
            seed=int(config.get("evaluation_seed", 2026)),
        )
        curves.append({"interactions": step, "metrics": result["metrics"]})
        (output / "learning_curve.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")

    if 0 in checkpoints:
        save_checkpoint(0)
    for collected in range(0, total_steps, rollout_steps):
        obs_buffer: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        values: list[float] = []
        next_values: list[float] = []
        terminal_flags: list[float] = []
        ended_flags: list[float] = []
        for _ in range(rollout_steps):
            numeric_obs = encode(observation)
            mask = decision_mask(observation)
            with torch.no_grad():
                logits, value = model(torch.as_tensor(numeric_obs, dtype=torch.float32))
                logits = logits.masked_fill(~torch.as_tensor(mask), -1e9)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
            action_index = int(action.item())
            following, reward, terminated, truncated, _info = env.step(
                action_from_observation(action_index, observation)
            )
            with torch.no_grad():
                _, following_value = model(torch.as_tensor(encode(following), dtype=torch.float32))
            obs_buffer.append(numeric_obs)
            masks.append(mask)
            actions.append(action_index)
            log_probs.append(float(log_prob.item()))
            rewards.append(float(reward))
            values.append(float(value.item()))
            next_values.append(float(following_value.item()))
            terminal_flags.append(float(terminated))
            ended_flags.append(float(terminated or truncated))
            total_reward += float(reward)
            observation = following
            if terminated or truncated:
                completed_episodes += 1
                env, observation = new_episode()
        advantages, returns = generalized_advantages(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            np.asarray(next_values, dtype=np.float32),
            np.asarray(terminal_flags, dtype=np.float32),
            np.asarray(ended_flags, dtype=np.float32),
            float(config.get("gamma", 0.99)),
            float(config.get("gae_lambda", 0.95)),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_obs = torch.as_tensor(np.stack(obs_buffer), dtype=torch.float32)
        batch_masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)
        batch_actions = torch.as_tensor(actions, dtype=torch.int64)
        batch_old_logs = torch.as_tensor(log_probs, dtype=torch.float32)
        batch_adv = torch.as_tensor(advantages, dtype=torch.float32)
        batch_returns = torch.as_tensor(returns, dtype=torch.float32)
        losses: list[float] = []
        for _ in range(int(config.get("update_epochs", 6))):
            permutation = np_rng.permutation(rollout_steps)
            minibatch = int(config.get("minibatch_size", 128))
            for first in range(0, rollout_steps, minibatch):
                indices = permutation[first : first + minibatch]
                logits, value = model(batch_obs[indices])
                logits = logits.masked_fill(~batch_masks[indices], -1e9)
                distribution = Categorical(logits=logits)
                ratio = (
                    distribution.log_prob(batch_actions[indices]) - batch_old_logs[indices]
                ).exp()
                unclipped = ratio * batch_adv[indices]
                clipped = (
                    ratio.clamp(
                        1 - float(config.get("clip_ratio", 0.2)),
                        1 + float(config.get("clip_ratio", 0.2)),
                    )
                    * batch_adv[indices]
                )
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_loss = (value - batch_returns[indices]).square().mean()
                loss = (
                    actor_loss
                    + float(config.get("value_coefficient", 0.5)) * critic_loss
                    - float(config.get("entropy_coefficient", 0.015))
                    * distribution.entropy().mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.get("max_grad_norm", 0.5))
                )
                optimizer.step()
                losses.append(float(loss.item()))
        step = collected + rollout_steps
        updates.append({"interactions": step, "loss": float(np.mean(losses))})
        if step in checkpoints:
            save_checkpoint(step)
    # Convenience artifact always aliases the fixed final checkpoint.
    (output / "checkpoint.pt").write_bytes(
        (output / f"checkpoint_{total_steps:06d}.pt").read_bytes()
    )
    metadata.update(
        {
            "completed_episodes": completed_episodes,
            "environment_interactions": total_steps,
            "training_reward_sum": total_reward,
            "elapsed_seconds": time.perf_counter() - start,
            "learning_curve": curves,
            "updates": updates,
        }
    )
    (output / "training_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
