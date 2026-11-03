"""Refresh release evaluations without replacing the registered training evidence.

Run from the repository root after committing release source fixes:
    uv run python examples/refresh_evaluations.py

Fresh reports are staged and all per-scenario differences are recorded. A changed
summary is published only with --accept-reviewed-outcome-changes after inspection.
Use --reuse-candidate with that flag to publish the exact checksummed candidate
already reviewed. Training reports, curves and original evaluations are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "unsafe_side_effect",
    "goal_satisfied",
    "violations",
    "had_ambiguous_timeout",
    "had_postcommit_timeout",
    "fault_kinds",
    "return",
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def evaluate_job(job: dict[str, Any]) -> dict[str, Any]:
    """One isolated policy/split; only this worker writes its unique report."""
    torch.set_num_threads(1)
    if source_fingerprint() != job["source_hash"]:
        raise RuntimeError("Source changed before evaluation worker started")
    if job["policy"] == "random":
        policy: Any = RandomPolicy(2026)
    elif job["policy"] == "heuristic":
        policy = HeuristicPolicy()
    else:
        policy = PPOPolicy(job["checkpoint"])
    scenarios = None
    if job.get("scenarios") is not None:
        from callflow_rl.contracts import ScenarioSpec

        scenarios = [ScenarioSpec.from_dict(item) for item in job["scenarios"]]
    report = evaluate_policy(
        policy,
        scenarios=scenarios,
        split=job["split"],
        seed=job["seed"],
        reward_mode=job["reward_mode"],
    )
    if source_fingerprint() != job["source_hash"]:
        raise RuntimeError("Source changed during evaluation worker execution")
    report["evaluation_execution"] = {
        "worker_pid": os.getpid(),
        "torch_threads": 1,
        "random_policy_seed": 2026 if job["policy"] == "random" else None,
        "policy_reset_scope": "one new policy per frozen suite",
    }
    write_json(Path(job["destination"]), report)
    return {"policy": job["policy"], "split": job["split"], "report": report}


def evaluate_jobs(jobs: list[dict[str, Any]], workers: int) -> dict[tuple[str, str], Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    results = {}
    if workers == 1:
        for job in jobs:
            result = evaluate_job(job)
            results[(result["policy"], result["split"])] = result["report"]
        return results
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = [executor.submit(evaluate_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results[(result["policy"], result["split"])] = result["report"]
            print(
                "worker",
                result["report"]["evaluation_execution"]["worker_pid"],
                result["policy"],
                result["split"],
                result["report"]["metrics"]["task_success"]["count"],
                flush=True,
            )
    except BaseException:
        # Python 3.14 has a public termination method. The supported 3.12/3.13
        # fallback stops only this executor's processes before source may change.
        if hasattr(executor, "terminate_workers"):
            executor.terminate_workers()
        else:
            for process in list((getattr(executor, "_processes", None) or {}).values()):
                process.terminate()
            executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def archive_current_release(output: Path, plan: dict[str, Any]) -> Path:
    """Keep the preceding release reports and derived presentation unchanged."""
    provenance = read_json(output / "evaluation_release.json")
    key = provenance["evaluation_release_source_hash"][:16]
    archive = output / "evaluation_history" / key
    manifest_path = archive / "archive_manifest.json"
    if manifest_path.exists():
        for relative, expected in read_json(manifest_path)["files"].items():
            if sha256(archive / relative) != expected:
                raise RuntimeError(f"Previous release archive changed: {relative}")
        return archive
    files = [path for group in policy_paths(output, plan).values() for path in group.values()]
    files += [
        output / name
        for name in (
            "evaluation_release.json",
            "summary.json",
            "RESULTS.md",
            "learning_curve.png",
            "baseline_comparison.png",
        )
    ]
    hashes = {}
    for path in files:
        relative = path.relative_to(output)
        previous_hash = provenance["release_evaluation_file_sha256"].get(relative.as_posix())
        if previous_hash is not None and sha256(path) != previous_hash:
            raise RuntimeError(f"Current release report changed before archival: {relative}")
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[relative.as_posix()] = sha256(destination)
    write_json(
        manifest_path,
        {
            "hash_normalization": HASH_NORMALIZATION,
            "evaluation_release_revision": provenance["evaluation_release_revision"],
            "evaluation_release_source_hash": provenance["evaluation_release_source_hash"],
            "files": hashes,
        },
    )
    return archive


def differences(
    name: str, split: str, original: dict[str, Any], fresh: dict[str, Any]
) -> list[dict[str, Any]]:
    before = {r["scenario_id"]: r for r in original["records"]}
    if set(before) != {r["scenario_id"] for r in fresh["records"]}:
        raise RuntimeError(f"Scenario membership changed for {name}/{split}")
    changed = []
    for row in fresh["records"]:
        previous = before[row["scenario_id"]]
        if row["seed"] != previous["seed"]:
            raise RuntimeError(f"Evaluation seed changed for {name}/{row['scenario_id']}")
        for key in OUTCOME_KEYS:
            if row.get(key) != previous.get(key):
                changed.append(
                    {
                        "policy": name,
                        "split": split,
                        "scenario_id": row["scenario_id"],
                        "key": key,
                        "original": previous.get(key),
                        "release": row.get(key),
                    }
                )
    return changed


def make_summary(
    plan: dict[str, Any], provenance: dict[str, Any], results: dict[str, Any]
) -> dict[str, Any]:
    settings = plan["config"]
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
    return summary


def refresh(
    output: Path,
    *,
    archive_only: bool = False,
    accept_reviewed_outcome_changes: bool = False,
    reuse_candidate: bool = False,
    workers: int = 1,
) -> None:
    plan = read_json(output / "registered_plan.json")
    settings = plan["config"]
    archive_manifest = archive_originals(output, plan)
    if archive_only:
        print(f"Archived and checksummed {len(archive_manifest['files'])} original evaluations.")
        return
    preceding = archive_current_release(output, plan)
    source_hash = source_fingerprint()
    stage = output / "release_candidates" / source_hash[:16]
    paths = policy_paths(output, plan)
    staged_paths = policy_paths(stage, plan)
    checkpoint_hashes = {
        name: sha256(output / name / "checkpoint.pt") for name in paths if name.startswith("ppo_")
    }
    if reuse_candidate:
        cached = read_json(stage / "candidate_manifest.json")
        expected = {
            "evaluation_release_source_hash": source_hash,
            "checkpoint_sha256": checkpoint_hashes,
            "refresh_helper_sha256": sha256(Path(__file__)),
            "registered_manifests": plan["manifests"],
        }
        for key, value in expected.items():
            if cached[key] != value:
                raise RuntimeError(f"Reviewed candidate no longer matches {key}")
        for relative, expected_hash in cached["files"].items():
            if sha256(stage / relative) != expected_hash:
                raise RuntimeError(f"Candidate changed since inspection: {relative}")
        evaluation_revision = cached["evaluation_release_revision"]
        evaluation_workers = cached["evaluation_workers_requested"]
    else:
        evaluation_revision = build_revision()
        evaluation_workers = workers
    fresh_reports = {}
    if not reuse_candidate:
        jobs = []
        for name, group in paths.items():
            for split, path in group.items():
                original = read_json(output / "original_evaluations" / path.relative_to(output))
                jobs.append(
                    {
                        "policy": name,
                        "split": split,
                        "seed": original["seed"],
                        "reward_mode": original["reward_mode"],
                        "source_hash": source_hash,
                        "checkpoint": str((output / name).resolve())
                        if name.startswith("ppo_")
                        else None,
                        "destination": str(staged_paths[name][split].resolve()),
                    }
                )
        fresh_reports = evaluate_jobs(jobs, workers)
    torch.set_num_threads(int(settings.get("torch_threads", 1)))
    results: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    previous_mismatches: list[dict[str, Any]] = []
    for name, split_paths in paths.items():
        results[name] = {}
        for split, path in split_paths.items():
            relative = path.relative_to(output)
            original = read_json(output / "original_evaluations" / relative)
            previous = read_json(preceding / relative)
            if reuse_candidate:
                fresh = read_json(staged_paths[name][split])
            else:
                fresh = fresh_reports[(name, split)]
            if original["evaluated_scenarios_hash"] != fresh["evaluated_scenarios_hash"]:
                raise RuntimeError(f"Fixed evaluation suite changed for {name}/{split}")
            if fresh["manifest_hash"] != plan["manifests"][split]:
                raise RuntimeError(f"Registered manifest changed for {split}")
            results[name][split] = fresh
            mismatches.extend(differences(name, split, original, fresh))
            previous_mismatches.extend(differences(name, split, previous, fresh))
            comparison.append(
                {
                    "policy": name,
                    "split": split,
                    "original_task_success": original["metrics"]["task_success"],
                    "previous_release_task_success": previous["metrics"]["task_success"],
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
    file_hashes = {
        path.relative_to(stage).as_posix(): sha256(path)
        for group in staged_paths.values()
        for path in group.values()
    }
    write_json(
        stage / "candidate_manifest.json",
        {
            "hash_normalization": HASH_NORMALIZATION,
            "evaluation_release_source_hash": source_hash,
            "evaluation_release_revision": evaluation_revision,
            "evaluation_workers_requested": evaluation_workers,
            "checkpoint_sha256": checkpoint_hashes,
            "refresh_helper_sha256": sha256(Path(__file__)),
            "registered_manifests": plan["manifests"],
            "files": file_hashes,
        },
    )
    changes = bool(mismatches or previous_mismatches)
    provenance = {
        "hash_normalization": HASH_NORMALIZATION,
        "registered_training_build_revision": plan["build_revision"],
        "registered_training_source_hash": plan["source_hash"],
        "evaluation_release_revision": evaluation_revision,
        "publication_revision": build_revision()
        if accept_reviewed_outcome_changes or not changes
        else None,
        "evaluation_release_source_hash": source_hash,
        "evaluation_workers_requested": evaluation_workers,
        "evaluation_worker_pids": sorted(
            {
                report.get("evaluation_execution", {}).get("worker_pid")
                for group in results.values()
                for report in group.values()
            }
        ),
        "latency_caveat": "Episode latency/throughput were measured with the recorded concurrent worker count; no speedup is inferred against historical serial evaluations.",
        "environment_versions": sorted(
            {r["environment_version"] for group in results.values() for r in group.values()}
        ),
        "verifier_versions": sorted(
            {
                row["verifier_version"]
                for group in results.values()
                for report in group.values()
                for row in report["records"]
            }
        ),
        "refresh_helper_sha256": sha256(Path(__file__)),
        "gradient_training_repeated": False,
        "original_evaluation_archive_file_count": len(archive_manifest["files"]),
        "previous_release_archive": preceding.relative_to(output).as_posix(),
        "checkpoint_sha256": checkpoint_hashes,
        "compared_per_scenario_keys": list(OUTCOME_KEYS),
        "per_scenario_outcomes_unchanged": not mismatches,
        "episode_returns_unchanged": not any(m["key"] == "return" for m in mismatches),
        "mismatches": mismatches,
        "previous_release_mismatches": previous_mismatches,
        "reviewed_outcome_changes_accepted": bool(changes and accept_reviewed_outcome_changes),
        "comparison": comparison,
        "training_reward_modes": {
            name: "sparse" if "_sparse_" in name else "shaped" for name in checkpoint_hashes
        },
        "release_evaluation_file_sha256": file_hashes,
    }
    write_json(stage / "evaluation_release.json", provenance)
    summary = make_summary(plan, provenance, results)
    write_json(stage / "summary.json", summary)
    # The original curves describe training on build 837af56, even when final
    # checkpoint evaluation uses corrected release dynamics and verification.
    for name in checkpoint_hashes:
        target = stage / name / "learning_curve.json"
        shutil.copy2(output / name / "learning_curve.json", target)
    make_plots(stage, settings, results)
    write_results_markdown(stage, summary)
    with (stage / "RESULTS.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\nRelease evaluations use the original checkpoints after adversarial review "
            "corrected verifier bindings, policy enforcement and cancellation dialogue. "
            "Training remains on build837af56; no gradients were rerun. Original and "
            "preceding evaluations are archived. "
            f"There are {len(mismatches)} per-scenario field differences against the original "
            "evaluations; differences include returns and resource use as well as verdicts. "
            "[evaluation_release.json](evaluation_release.json) records every difference, "
            "source version, verifier version and checkpoint hash.\n"
        )
    if changes and not accept_reviewed_outcome_changes:
        raise SystemExit(
            f"Changed outcomes are staged at {stage}; review {len(mismatches)} original "
            f"and {len(previous_mismatches)} preceding-release field differences. "
            "Published reports/summary are unchanged. After review, rerun with "
            "--reuse-candidate --accept-reviewed-outcome-changes."
        )
    for name, group in staged_paths.items():
        for split, path in group.items():
            shutil.copy2(path, paths[name][split])
    for filename in (
        "evaluation_release.json",
        "summary.json",
        "RESULTS.md",
        "learning_curve.png",
        "baseline_comparison.png",
    ):
        shutil.copy2(stage / filename, output / filename)
    print(f"Published evaluated candidate {source_hash[:16]}; reviewed changes accepted={changes}.")


def parallel_smoke(output: Path, workers: int) -> None:
    """Compare full deterministic episode records with serial and process execution."""
    from callflow_rl.scenarios import load_suite

    families = {
        "book_direct",
        "book_identity_missing",
        "book_timeout_before",
        "book_timeout_after",
        "reschedule_direct",
        "reschedule_timeout_after",
        "cancel_direct",
        "cancel_timeout_before",
    }
    selected = {}
    for scenario in load_suite("train"):
        if scenario.archetype in families and scenario.archetype not in selected:
            selected[scenario.archetype] = scenario.to_dict()
    if len(selected) != len(families):
        raise RuntimeError("Parallel smoke fixture families are incomplete")
    source_hash = source_fingerprint()
    base = Path("artifacts/local/evaluation_parallel_smoke")
    modes = {}
    for mode, count in (("serial", 1), ("parallel", workers)):
        jobs = []
        for name in ("random", "heuristic", "ppo_shaped_seed_11", "ppo_shaped_seed_29"):
            jobs.append(
                {
                    "policy": name,
                    "split": "train",
                    "seed": 2026,
                    "reward_mode": "shaped",
                    "source_hash": source_hash,
                    "checkpoint": str((output / name).resolve())
                    if name.startswith("ppo_")
                    else None,
                    "destination": str((base / mode / f"{name}.json").resolve()),
                    "scenarios": list(selected.values()),
                }
            )
        modes[mode] = evaluate_jobs(jobs, count)
    checks = []
    for key, original in modes["serial"].items():
        fresh = modes["parallel"][key]
        before = [
            {k: v for k, v in row.items() if k != "latency_seconds"} for row in original["records"]
        ]
        after = [
            {k: v for k, v in row.items() if k != "latency_seconds"} for row in fresh["records"]
        ]
        if before != after:
            raise RuntimeError(f"Serial/parallel episode mismatch: {key}")
        checks.append({"policy": key[0], "episodes": len(before), "episode_records_match": True})
    report = {
        "build_revision": build_revision(),
        "source_hash": source_hash,
        "workers_requested": workers,
        "policies": checks,
        "episodes_compared": sum(c["episodes"] for c in checks),
        "serial_parallel_parity": True,
        "compared_records": "All fields including verdicts, rewards and state hashes, except wall-clock latency.",
        "parallel_worker_pids": sorted(
            {report["evaluation_execution"]["worker_pid"] for report in modes["parallel"].values()}
        ),
    }
    write_json(Path("artifacts/validation/evaluation_parallel_parity.json"), report)
    print("Serial/parallel parity passed", report["episodes_compared"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--accept-reviewed-outcome-changes", action="store_true")
    parser.add_argument("--reuse-candidate", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-smoke", action="store_true")
    options = parser.parse_args()
    if options.parallel_smoke:
        parallel_smoke(options.output, options.workers)
    else:
        refresh(
            options.output,
            archive_only=options.archive_only,
            accept_reviewed_outcome_changes=options.accept_reviewed_outcome_changes,
            reuse_candidate=options.reuse_candidate,
            workers=options.workers,
        )
