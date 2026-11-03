"""Refresh release evaluations without replacing the registered training evidence.

Run from the repository root after committing release source fixes:
    uv run python examples/refresh_evaluations.py

The first invocation archives all 28 original fixed-suite evaluations, then
evaluates the same checkpoints. It refuses to hide changed per-scenario outcomes.
Training reports, development learning curves and the original plan are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from callflow_rl.evaluation.experiments import (
    make_plots,
    source_fingerprint,
    write_results_markdown,
)
from callflow_rl.evaluation.metrics import build_revision, evaluate_policy
from callflow_rl.evaluation.statistics import bootstrap_seed_means
from callflow_rl.policies import HeuristicPolicy, PPOPolicy, RandomPolicy

SPLITS = ("train", "dev", "test", "transfer")
HASH_NORMALIZATION = "CRLF-to-LF for .json/.jsonl/.py/.md UTF-8 text; raw bytes for binary files"
TEXT_SUFFIXES = {".json", ".jsonl", ".py", ".md"}
OUTCOME_KEYS = (
    "task_success",
    "policy_compliance",
    "false_success_claim",
    "safety_violation",
    "duplicate_side_effect",
    "justified_handoff",
    "justified_refusal",
    "appropriate_resolution",
    "termination_reason",
    "steps",
    "tool_calls",
    "truncated",
    "invalid_action_count",
    "simulator_inconsistency_count",
)


def sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def policy_paths(output: Path, plan: dict[str, Any]) -> dict[str, dict[str, Path]]:
    settings = plan["config"]
    names = [f"ppo_shaped_seed_{s}" for s in settings["training_seeds"]]
    first = settings["training_seeds"][0]
    names += [f"ppo_sparse_seed_{first}", f"ppo_no_fault_seed_{first}"]
    result = {
        name: {split: output / f"{name}_{split}.json" for split in SPLITS}
        for name in ("random", "heuristic")
    }
    result.update(
        {
            name: {split: output / name / f"evaluation_{split}.json" for split in SPLITS}
            for name in names
        }
    )
    return result


def archive_originals(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Never overwrite the original evidence on subsequent release refreshes."""
    archive = output / "original_evaluations"
    manifest_path = archive / "archive_manifest.json"
    paths = policy_paths(output, plan)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("hash_normalization") != HASH_NORMALIZATION:
            raise RuntimeError("Archive manifest must declare the canonical text hashing scheme")
        for relative, expected in manifest["files"].items():
            if sha256(archive / relative) != expected:
                raise RuntimeError(f"Archived original was changed: {relative}")
        return manifest
    archive.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for split_paths in paths.values():
        for path in split_paths.values():
            if not path.exists():
                raise RuntimeError(f"Original experiment is incomplete: {path}")
            relative = path.relative_to(output)
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            files[relative.as_posix()] = sha256(destination)
    manifest = {
        "hash_normalization": HASH_NORMALIZATION,
        "training_build_revision": plan["build_revision"],
        "training_source_hash": plan["source_hash"],
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def refresh(output: Path, *, archive_only: bool = False) -> None:
    plan = json.loads((output / "registered_plan.json").read_text(encoding="utf-8"))
    settings = plan["config"]
    archive_manifest = archive_originals(output, plan)
    if archive_only:
        print(f"Archived and checksummed {len(archive_manifest['files'])} original evaluations.")
        return
    torch.set_num_threads(int(settings.get("torch_threads", 1)))
    paths = policy_paths(output, plan)
    results: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    checkpoint_hashes = {}
    for name, split_paths in paths.items():
        results[name] = {}
        for split, path in split_paths.items():
            original = json.loads(
                (output / "original_evaluations" / path.relative_to(output)).read_text(
                    encoding="utf-8"
                )
            )
            if name == "random":
                policy: Any = RandomPolicy(2026)
            elif name == "heuristic":
                policy = HeuristicPolicy()
            else:
                policy = PPOPolicy(output / name)
                checkpoint_hashes[name] = policy.version
            fresh = evaluate_policy(
                policy,
                split=split,
                seed=original["seed"],
                reward_mode=original["reward_mode"],
                output_path=path,
            )
            results[name][split] = fresh
            if original["evaluated_scenarios_hash"] != fresh["evaluated_scenarios_hash"]:
                raise RuntimeError(f"Fixed evaluation suite changed for {name}/{split}")
            if fresh["manifest_hash"] != plan["manifests"][split]:
                raise RuntimeError(f"Registered manifest changed for {split}")
            previous = {r["scenario_id"]: r for r in original["records"]}
            if set(previous) != {r["scenario_id"] for r in fresh["records"]}:
                raise RuntimeError(f"Scenario membership changed for {name}/{split}")
            for row in fresh["records"]:
                before = previous[row["scenario_id"]]
                for key in OUTCOME_KEYS:
                    if row.get(key) != before.get(key):
                        mismatches.append(
                            {
                                "policy": name,
                                "scenario_id": row["scenario_id"],
                                "key": key,
                                "original": before.get(key),
                                "release": row.get(key),
                            }
                        )
            comparison.append(
                {
                    "policy": name,
                    "split": split,
                    "original_task_success": original["metrics"]["task_success"],
                    "release_task_success": fresh["metrics"]["task_success"],
                    "original_postcommit_timeout_recovery": original["metrics"][
                        "postcommit_timeout_recovery"
                    ],
                    "release_postcommit_timeout_recovery": fresh["metrics"][
                        "postcommit_timeout_recovery"
                    ],
                }
            )
            print(name, split, fresh["metrics"]["task_success"]["count"], flush=True)
    provenance = {
        "hash_normalization": HASH_NORMALIZATION,
        "registered_training_build_revision": plan["build_revision"],
        "registered_training_source_hash": plan["source_hash"],
        "evaluation_release_revision": build_revision(),
        "evaluation_release_source_hash": source_fingerprint(),
        "refresh_helper_sha256": sha256(Path(__file__)),
        "gradient_training_repeated": False,
        "original_evaluation_archive_file_count": len(archive_manifest["files"]),
        "checkpoint_sha256": checkpoint_hashes,
        "compared_per_scenario_keys": list(OUTCOME_KEYS),
        "per_scenario_outcomes_unchanged": not mismatches,
        "mismatches": mismatches,
        "comparison": comparison,
        "training_reward_modes": {
            name: "sparse" if "_sparse_" in name else "shaped" for name in checkpoint_hashes
        },
        "release_evaluation_file_sha256": {
            path.relative_to(output).as_posix(): sha256(path)
            for split_paths in paths.values()
            for path in split_paths.values()
        },
    }
    (output / "evaluation_release.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    if mismatches:
        raise RuntimeError(
            "Release outcomes differ; investigate evaluation_release.json before publishing."
        )
    summary: dict[str, Any] = {
        "plan": plan,
        "evaluation_release": provenance,
        "policies": {},
        "training_seed_aggregate": {},
    }
    for name, splits in results.items():
        summary["policies"][name] = {
            split: {"metrics": value["metrics"], "slices": value["slices"]}
            for split, value in splits.items()
        }
        summary["policies"][name]["heldout_success_gap"] = (
            splits["train"]["metrics"]["task_success"]["rate"]
            - splits["test"]["metrics"]["task_success"]["rate"]
        )
    for split in SPLITS:
        rates = [
            results[f"ppo_shaped_seed_{s}"][split]["metrics"]["task_success"]["rate"]
            for s in settings["training_seeds"]
        ]
        summary["training_seed_aggregate"][split] = {
            "seeds": settings["training_seeds"],
            "task_success_rates": rates,
            "mean": float(np.mean(rates)),
            "ci95_seed_bootstrap": bootstrap_seed_means(rates),
            "caveat": "Only three independent seeds; bootstrap interval is descriptive and coarse.",
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(output, settings, results)
    write_results_markdown(output, summary)
    with (output / "RESULTS.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\nRelease evaluations use the same checkpoints after guard/metadata corrections. "
            "Original training and 28 evaluation files are retained. "
            "Per-scenario outcomes are unchanged; "
            "[evaluation_release.json](evaluation_release.json) records both source versions, "
            "checkpoint hashes and fault-exposure denominator comparisons.\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--archive-only", action="store_true")
    options = parser.parse_args()
    refresh(options.output, archive_only=options.archive_only)
