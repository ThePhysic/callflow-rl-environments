"""Record local release checks; run with the locked development environment."""

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from callflow_rl.evaluation.experiments import source_fingerprint


def main():
    commands = [
        ["-m", "ruff", "check", "."],
        ["-m", "ruff", "format", "--check", "."],
        ["-m", "mypy"],
        ["-m", "pytest", "-q"],
        ["-m", "callflow_rl", "validate", "--suite", "core"],
        ["-m", "callflow_rl", "audit-rewards", "--suite", "adversarial"],
        ["examples/build_examples.py"],
        ["examples/build_trace_preview.py"],
        ["-m", "callflow_rl", "replay", "--trace", "examples/timeout_after_commit.jsonl"],
        ["examples/demo.py"],
        [
            "-m",
            "callflow_rl",
            "train",
            "--config",
            "configs/smoke.yaml",
            "--output",
            "artifacts/local/cli_smoke",
        ],
        [
            "-m",
            "callflow_rl",
            "evaluate",
            "--checkpoint",
            "artifacts/experiments/ppo_shaped_seed_11",
            "--split",
            "test",
            "--output",
            "artifacts/validation/cli_evaluation.json",
        ],
        [
            "-m",
            "callflow_rl",
            "rollout",
            "--policy",
            "heuristic",
            "--split",
            "test",
            "--episodes",
            "200",
            "--workers",
            "4",
            "--seed",
            "42",
            "--output",
            "artifacts/validation/parallel",
        ],
    ]
    records = []
    for command in commands:
        result = subprocess.run([sys.executable, *command], capture_output=True, text=True)
        records.append(
            {
                "command": ["python", *command],
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        print("PASS" if result.returncode == 0 else "FAIL", " ".join(command), flush=True)
        if result.returncode:
            print(result.stdout, result.stderr, flush=True)
    report = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_fingerprint": source_fingerprint(),
        "checks": records,
        "all_passed": all(r["exit_code"] == 0 for r in records),
    }
    destination = Path("artifacts/validation/quality.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
