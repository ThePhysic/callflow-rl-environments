"""Frozen CPU experiment plan, saved checkpoints, learning curves and diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from callflow_rl.evaluation.metrics import build_revision, evaluate_policy
from callflow_rl.evaluation.statistics import bootstrap_seed_means
from callflow_rl.policies import HeuristicPolicy, RandomPolicy
from callflow_rl.policies.ppo import PPOPolicy, train_ppo


def _read_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    return dict(yaml.safe_load(Path(config).read_text(encoding="utf-8")))


def train(
    config: dict[str, Any] | str | Path = "configs/ppo.yaml",
    output_dir: str | Path = "artifacts/local/training",
) -> dict[str, Any]:
    """CLI entry: train the registered baseline seeds with dev-only checkpoints."""
    settings = _read_config(config)
    output = Path(output_dir)
    results = []
    for seed in settings.get("training_seeds", [11, 29, 47]):
        results.append(train_ppo(settings, output / f"seed_{seed}", seed=int(seed)))
    return {"runs": results, "output_dir": str(output)}


def run_experiments(
    output_dir: str | Path = "artifacts/experiments",
    config_path: str | Path = "configs/ppo.yaml",
) -> dict[str, Any]:
    """Execute the pre-registered final-budget comparison without test selection."""
    from callflow_rl.scenarios import manifest_hash

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = _read_config(config_path)
    plan = {
        "config": settings,
        "build_revision": build_revision(),
        "source_hash": source_fingerprint(),
        "manifests": {
            split: manifest_hash(split) for split in ("train", "dev", "test", "transfer")
        },
        "selection_rule": "Use final pre-registered interaction budget for every run; no test-based selection.",
        "baseline_seeds": settings["training_seeds"],
        "sparse_comparison_seeds": [settings["training_seeds"][0]],
        "no_fault_ablation_seeds": [settings["training_seeds"][0]],
        "exploratory_caveat": "Sparse and no-fault comparisons use one seed and are descriptive only.",
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    (output / "registered_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    results: dict[str, Any] = {}
    for name, constructor in (
        ("random", lambda: RandomPolicy(2026)),
        ("heuristic", HeuristicPolicy),
    ):
        results[name] = {}
        for split in ("train", "dev", "test", "transfer"):
            results[name][split] = evaluate_policy(
                constructor(), split=split, seed=2026, output_path=output / f"{name}_{split}.json"
            )
    run_paths = []
    for seed in settings["training_seeds"]:
        run_paths.append((f"ppo_shaped_seed_{seed}", int(seed), "shaped"))
    first_seed = int(settings["training_seeds"][0])
    run_paths.extend(
        [
            (f"ppo_sparse_seed_{first_seed}", first_seed, "sparse"),
            (f"ppo_no_fault_seed_{first_seed}", first_seed, "no_fault_training"),
        ]
    )
    for name, seed, variant in run_paths:
        run_dir = output / name
        train_ppo(settings, run_dir, seed=seed, variant=variant)
        policy = PPOPolicy(run_dir)
        results[name] = {}
        for split in ("train", "dev", "test", "transfer"):
            results[name][split] = evaluate_policy(
                policy,
                split=split,
                seed=2026,
                output_path=run_dir / f"evaluation_{split}.json",
            )
    summary: dict[str, Any] = {"plan": plan, "policies": {}, "training_seed_aggregate": {}}
    for name, splits in results.items():
        summary["policies"][name] = {
            split: {"metrics": result["metrics"], "slices": result["slices"]}
            for split, result in splits.items()
        }
        summary["policies"][name]["heldout_success_gap"] = (
            splits["train"]["metrics"]["task_success"]["rate"]
            - splits["test"]["metrics"]["task_success"]["rate"]
        )
    learned_names = [f"ppo_shaped_seed_{s}" for s in settings["training_seeds"]]
    for split in ("train", "dev", "test", "transfer"):
        seed_rates = [
            results[name][split]["metrics"]["task_success"]["rate"] for name in learned_names
        ]
        summary["training_seed_aggregate"][split] = {
            "seeds": settings["training_seeds"],
            "task_success_rates": seed_rates,
            "mean": float(np.mean(seed_rates)),
            "ci95_seed_bootstrap": bootstrap_seed_means(seed_rates),
            "caveat": "Only three independent seeds; bootstrap interval is descriptive and coarse.",
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(output, settings, results)
    write_results_markdown(output, summary)
    return summary


def source_fingerprint() -> str:
    """Content fingerprint includes uncommitted source, unlike Git HEAD alone."""
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "callflow_rl").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_plots(output: Path, settings: dict[str, Any], results: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for seed in settings["training_seeds"]:
        curve = json.loads((output / f"ppo_shaped_seed_{seed}" / "learning_curve.json").read_text())
        ax.plot(
            [r["interactions"] for r in curve],
            [100 * r["metrics"]["task_success"]["rate"] for r in curve],
            marker="o",
            label=f"PPO seed {seed}",
        )
    ax.set(
        xlabel="Training environment interactions",
        ylabel="Dev verified task success (%)",
        ylim=(0, 102),
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(3)
    width = 0.25
    for offset, (label, names) in enumerate(
        (
            ("Random", ["random"]),
            ("Heuristic", ["heuristic"]),
            ("PPO (3-seed mean)", [f"ppo_shaped_seed_{s}" for s in settings["training_seeds"]]),
        )
    ):
        values = []
        for split, tag in (("train", "clean"), ("train", "fault"), ("test", "fault")):
            values.append(
                100
                * float(
                    np.mean(
                        [
                            results[name][split]["slices"].get(
                                tag, results[name][split]["metrics"]
                            )["task_success"]["rate"]
                            for name in names
                        ]
                    )
                )
            )
        ax.bar(positions + (offset - 1) * width, values, width, label=label)
    ax.set(
        xticks=positions,
        xticklabels=["Familiar clean", "Familiar tool faults", "Held-out tool faults"],
        ylabel="Verified task success (%)",
        ylim=(0, 105),
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "baseline_comparison.png", dpi=160)
    plt.close(fig)


def write_results_markdown(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Measured experiment results",
        "",
        "Every row uses the final fixed-budget checkpoint; no test-based selection. "
        "Success means the independent verifier accepted the completed database task. "
        "Appropriate handoffs/refusals are reported separately in JSON.",
        "",
        "| Policy | Familiar task success | Held-out task success | Transfer task success | Held-out appropriate resolution |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["policies"].items():
        values = []
        for split, key in (
            ("train", "task_success"),
            ("test", "task_success"),
            ("transfer", "task_success"),
            ("test", "appropriate_resolution"),
        ):
            metric = result[split]["metrics"][key]
            values.append(
                f"{metric['count']}/{metric['denominator']} ({100 * metric['rate']:.1f}%)"
            )
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "The three-seed PPO aggregate, confidence intervals, per-scenario outcomes, failures, "
            "safety and recovery denominators are in summary.json and each evaluation JSON. "
            "Sparse reward and no-fault-training comparisons each use one seed, so their differences "
            "are exploratory. The clean/fault chart compares distinct authored archetypes and "
            "does not isolate a causal fault effect.",
            "",
            "Intervals over scenarios are Wilson intervals; intervals over PPO runs bootstrap "
            "three independent training-seed means and are coarse. These fixed benchmark results "
            "do not establish performance on real conversations or unrestricted natural language.",
            "",
            "The learned policy chooses bounded structured decisions from public numeric features. "
            "It does not train a language model. All fixtures and utterances are synthetic.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_experiments()
