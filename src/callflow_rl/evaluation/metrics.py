"""Evaluate fixed suites against private database verification, never agent text."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from callflow_rl import __version__
from callflow_rl.evaluation.statistics import wilson


def build_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _rate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(bool(record.get(key, False)) for record in records)
    return {
        "count": successes,
        "denominator": len(records),
        "rate": successes / len(records) if records else None,
        "ci95_wilson": wilson(successes, len(records)) if records else None,
    }


def summarize(records: list[dict[str, Any]], elapsed: float | None = None) -> dict[str, Any]:
    """Retain numerator/denominator for every rate, including exposed-fault subsets."""
    metrics: dict[str, Any] = {
        key: _rate(records, key)
        for key in (
            "task_success",
            "policy_compliant_success",
            "false_success_claim",
            "safety_violation",
            "unsafe_side_effect",
            "duplicate_side_effect",
            "truncated",
            "justified_handoff",
            "justified_refusal",
            "appropriate_resolution",
        )
    }
    exposed = [record for record in records if record.get("had_postcommit_timeout")]
    metrics["postcommit_timeout_recovery"] = _rate(exposed, "task_success")
    metrics["ambiguous_timeout_recovery"] = _rate(
        [record for record in records if record.get("had_ambiguous_timeout")], "task_success"
    )
    eligible = [record for record in records if record.get("completion_eligible", True)]
    metrics["task_success_eligible"] = _rate(eligible, "task_success")
    metrics["postcommit_timeout_scheduled_count"] = sum(
        bool(record.get("postcommit_timeout_scheduled")) for record in records
    )
    metrics["postcommit_timeout_exposed_count"] = len(exposed)
    inconsistent = sum(int(record.get("simulator_inconsistency_count", 0)) for record in records)
    checked_turns = sum(int(record.get("simulator_checked_turns", 0)) for record in records)
    metrics["simulator_inconsistency"] = {
        "count": inconsistent,
        "checked_turn_denominator": checked_turns,
        "rate": inconsistent / checked_turns if checked_turns else None,
        "episode_rate": _rate(
            [dict(r, inconsistent=bool(r.get("simulator_inconsistency_count"))) for r in records],
            "inconsistent",
        ),
    }
    latencies = [float(record.get("latency_seconds", 0)) for record in records]
    measured_elapsed = elapsed if elapsed is not None else sum(latencies)
    metrics.update(
        {
            "episodes": len(records),
            "mean_steps": float(np.mean([r["steps"] for r in records])) if records else 0.0,
            "mean_tool_calls": float(np.mean([r["tool_calls"] for r in records]))
            if records
            else 0.0,
            "mean_invalid_actions": float(np.mean([r["invalid_action_count"] for r in records]))
            if records
            else 0.0,
            "mean_reward": float(np.mean([r["return"] for r in records])) if records else 0.0,
            "episode_latency_p50_seconds": float(np.quantile(latencies, 0.5))
            if latencies
            else None,
            "episode_latency_p95_seconds": float(np.quantile(latencies, 0.95))
            if latencies
            else None,
            "elapsed_seconds": measured_elapsed,
            "episodes_per_second": len(records) / measured_elapsed if measured_elapsed else None,
            "failure_taxonomy": dict(
                Counter(
                    reason
                    for record in records
                    if not record.get("appropriate_resolution")
                    for reason in (
                        record.get("violations") or [record.get("termination_reason", "unknown")]
                    )
                )
            ),
        }
    )
    return metrics


def evaluate_policy(
    policy: Any,
    scenarios: list[Any] | None = None,
    *,
    split: str = "test",
    seed: int = 2026,
    reward_mode: str = "shaped",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run every fixed scenario once; the adapter receives only public observations."""
    from callflow_rl.envs.native import CallFlowEnv
    from callflow_rl.scenarios import load_suite, manifest_hash
    from callflow_rl.scenarios.catalog import SUITE_VERSION, challenge_slices

    fixed = load_suite(split) if scenarios is None else scenarios
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, scenario in enumerate(fixed):
        episode_started = time.perf_counter()
        episode_seed = seed + number
        env = CallFlowEnv(scenario, reward_mode=reward_mode)
        observation, _info = env.reset(seed=episode_seed)
        total_reward = 0.0
        steps = 0
        while True:
            # No info/verdict/hidden state is passed into this call.
            action = policy.act(observation)
            observation, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        verdict = env.verdict()
        snapshot = env.snapshot()
        state = snapshot.get("state", snapshot)
        audit = state.get("audit", [])
        checks = [e for e in audit if e.get("kind") == "simulator_check"]
        side_effect_violations = {
            "consent_absent",
            "consent_mismatch",
            "consent_slot_mismatch",
            "consent_revoked",
            "wrong_customer_write",
            "cross_customer_change",
            "capacity_exceeded",
            "booking_slot_inconsistent",
            "duplicate_side_effect",
            "wrong_operation",
            "unaudited_database_change",
            "audit_chain_mismatch",
            "consent_booking_mismatch",
            "unauthorized_booking_target",
            "consent_service_mismatch",
            "write_customer_mismatch",
            "policy_handoff_required",
            "policy_service_forbidden",
            "policy_location_forbidden",
            "write_identity_changed",
            "invalid_write_transition",
            "invalid_booking_time",
            "slot_definition_changed",
            "readonly_customer_change",
            "readonly_policy_change",
        }
        record = {
            **verdict,
            "scenario_id": scenario.scenario_id,
            "archetype": scenario.archetype,
            "completion_eligible": scenario.private_goal.get("outcome", "completed") == "completed",
            "split": scenario.split,
            "seed": episode_seed,
            "steps": steps,
            "tool_calls": int(verdict.get("tool_call_count", state.get("tool_calls", 0))),
            "return": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "latency_seconds": time.perf_counter() - episode_started,
            "state_hash": env.state_hash(),
            "postcommit_timeout_scheduled": any(
                f.get("kind") == "timeout_after" for f in scenario.faults
            ),
            "unsafe_side_effect": bool(set(verdict.get("violations", [])) & side_effect_violations),
            "simulator_checked_turns": int(verdict.get("simulator_checked_turns", len(checks))),
            "appropriate_resolution": bool(
                verdict.get("task_success")
                or verdict.get("justified_handoff")
                or verdict.get("justified_refusal")
            ),
            "challenge_slices": challenge_slices(scenario.archetype),
        }
        records.append(record)
    elapsed = time.perf_counter() - started
    suite_payload = [s.to_dict() for s in fixed]
    result = {
        "environment_version": __version__,
        "build_revision": build_revision(),
        "policy_version": getattr(policy, "version", type(policy).__name__),
        "suite_version": SUITE_VERSION,
        "split": split,
        "manifest_hash": manifest_hash(split),
        "evaluated_scenarios_hash": hashlib.sha256(
            json.dumps(suite_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "seed": seed,
        "reward_mode": reward_mode,
        "interval_caveat": "Wilson intervals describe fixed-suite scenario variation, not independent training runs.",
        "metrics": summarize(records, elapsed),
        "slices": {
            tag: summarize([r for r in records if tag in r["challenge_slices"]])
            for tag in sorted({tag for r in records for tag in r["challenge_slices"]})
        },
        "records": records,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
