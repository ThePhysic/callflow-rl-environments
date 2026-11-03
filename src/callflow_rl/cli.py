"""Small offline command line interface. Training is opt-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="callflow", description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Check schemas, split integrity and Gym contract"
    )
    validate.add_argument("--suite", choices=["core"], default="core")
    rollout = commands.add_parser("rollout", help="Run isolated episodes and export traces")
    rollout.add_argument("--policy", choices=["random", "heuristic", "ppo"], default="heuristic")
    rollout.add_argument("--checkpoint")
    rollout.add_argument("--split", choices=["train", "dev", "test", "transfer"], default="dev")
    rollout.add_argument("--episodes", type=int, default=200)
    rollout.add_argument("--seed", type=int, default=42)
    rollout.add_argument("--workers", type=int, default=4)
    rollout.add_argument("--output", type=Path, default=Path("artifacts/local/rollout"))
    rollout.add_argument("--save-traces", action="store_true")
    rollout.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    replay_cmd = commands.add_parser("replay", help="Re-execute a saved episode")
    replay_cmd.add_argument("--trace", type=Path, required=True)
    replay_cmd.add_argument("--allow-build-mismatch", action="store_true")
    replay_cmd.add_argument(
        "--recorded", action="store_true", help="Play outputs without re-execution"
    )
    viewer = commands.add_parser("view", help="Explain actions, tool results and rewards")
    viewer.add_argument("--trace", type=Path, required=True)
    branch_cmd = commands.add_parser("branch", help="Fork a pre-action state")
    branch_cmd.add_argument("--trace", type=Path, required=True)
    branch_cmd.add_argument("--step", type=int, required=True)
    branch_cmd.add_argument("--action", required=True, help="JSON structured action")
    branch_cmd.add_argument("--fault", help="Optional JSON evaluator fault")
    audit = commands.add_parser("audit-rewards", help="Demonstrate naive scoring exploit")
    audit.add_argument("--suite", choices=["adversarial"], default="adversarial")
    train = commands.add_parser("train", help="Train compact PPO on the finite action wrapper")
    train.add_argument("--algo", choices=["ppo"], default="ppo")
    train.add_argument("--config", type=Path, default=Path("configs/ppo.yaml"))
    train.add_argument("--output", type=Path, default=Path("artifacts/local/ppo"))
    evaluate = commands.add_parser("evaluate", help="Evaluate a checkpoint against a fixed suite")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=["train", "dev", "test", "transfer"], default="test")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/local/evaluation.json"))
    experiments = commands.add_parser(
        "experiments", help="Reproduce the registered experiment matrix"
    )
    experiments.add_argument("--config", type=Path, default=Path("configs/ppo.yaml"))
    experiments.add_argument("--output", type=Path, default=Path("artifacts/local/experiments"))
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    from callflow_rl.scenarios import load_suite, manifest_hash

    if args.command == "validate":
        from gymnasium.utils.env_checker import check_env

        from callflow_rl.contracts import ScenarioSpec
        from callflow_rl.envs.gym_env import CallFlowGymEnv

        check_env(CallFlowGymEnv(), skip_render_check=True)
        suites = {s: load_suite(s) for s in ("train", "dev", "test", "transfer")}
        for specs in suites.values():
            for spec in specs:
                assert ScenarioSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
        family_sets = {s: {x.archetype for x in specs} for s, specs in suites.items()}
        for split, families in family_sets.items():
            for other, other_families in family_sets.items():
                if other != split and families & other_families:
                    raise AssertionError(f"Archetype leakage: {split} vs {other}")
        print(
            json.dumps(
                {
                    "gymnasium_contract": "passed",
                    "suites": {
                        s: {"instances": len(specs), "manifest_hash": manifest_hash(s)}
                        for s, specs in suites.items()
                    },
                },
                indent=2,
            )
        )
    elif args.command == "rollout":
        from callflow_rl.rollouts import EpisodeTrace, run_batch, save_trace

        if args.episodes < 1:
            raise ValueError("episodes must be positive")
        suite = load_suite(args.split)
        specs = [suite[i % len(suite)] for i in range(args.episodes)]
        result = run_batch(
            specs,
            policy=args.policy,
            seed=args.seed,
            workers=args.workers,
            checkpoint=args.checkpoint,
            manifest=manifest_hash(args.split),
        )
        _write_json(args.output / "benchmark.json", result["benchmark"])
        records = []
        for i, episode in enumerate(result["episodes"]):
            records.append(
                {
                    "metadata": episode["metadata"],
                    "verdict": episode["verdict"],
                    "resources": episode["resources"],
                }
            )
            if args.save_traces:
                save_trace(
                    EpisodeTrace.from_dict(episode), args.output / f"episode_{i:04d}.{args.format}"
                )
        _write_json(args.output / "episodes.json", records)
        print(json.dumps(result["benchmark"], indent=2))
    elif args.command in {"view", "replay", "branch"}:
        from callflow_rl.rollouts import branch, load_trace, recorded_replay, replay
        from callflow_rl.rollouts.trace import view

        trace = load_trace(args.trace)
        if args.command == "view":
            print(view(trace))
        elif args.command == "branch":
            from callflow_rl.policies import HeuristicPolicy

            print(
                json.dumps(
                    branch(
                        trace,
                        args.step,
                        json.loads(args.action),
                        HeuristicPolicy(),
                        json.loads(args.fault) if args.fault else None,
                    ),
                    indent=2,
                )
            )
        elif args.recorded:
            print(
                json.dumps(
                    {"mode": "recorded_output_playback", "steps": len(recorded_replay(trace))}
                )
            )
        else:
            print(json.dumps(replay(trace, args.allow_build_mismatch), indent=2))
    elif args.command == "audit-rewards":
        from callflow_rl.verifiers.audit import run_audit

        print(json.dumps(run_audit(), indent=2))
    elif args.command == "train":
        from callflow_rl.evaluation.experiments import train

        print(json.dumps(train(args.config, args.output), indent=2, default=str))
    elif args.command == "evaluate":
        from callflow_rl.evaluation.metrics import evaluate_policy
        from callflow_rl.policies import PPOPolicy

        result = evaluate_policy(
            PPOPolicy(args.checkpoint), load_suite(args.split), split=args.split, seed=args.seed
        )
        _write_json(args.output, result)
        print(json.dumps(result["metrics"], indent=2, default=str))
    elif args.command == "experiments":
        from callflow_rl.evaluation.experiments import run_experiments

        result = run_experiments(args.output, args.config)
        print(json.dumps(result, indent=2, default=str))
