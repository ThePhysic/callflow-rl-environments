"""Historical evidence checksums must survive a fresh LF-normalized checkout."""

import hashlib
import json
from pathlib import Path


def test_archived_evaluations_have_portable_content_checksums():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "experiments"
    archive = root / "original_evaluations"
    manifest = json.loads((archive / "archive_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 28
    for name, expected in manifest["files"].items():
        data = (archive / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == expected


def test_all_main_training_runs_reached_registered_budget():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "experiments"
    plan = json.loads((root / "registered_plan.json").read_text(encoding="utf-8"))
    for seed in plan["config"]["training_seeds"]:
        run = root / f"ppo_shaped_seed_{seed}"
        report = json.loads((run / "training_report.json").read_text(encoding="utf-8"))
        assert report["environment_interactions"] == plan["config"]["total_steps"]
        assert report["seed"] == seed
        assert report["test_used_for_training_or_selection"] is False
