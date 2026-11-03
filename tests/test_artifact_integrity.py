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


def test_published_evaluations_and_weights_match_recorded_provenance():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "experiments"
    release = json.loads((root / "evaluation_release.json").read_text(encoding="utf-8"))
    assert len(release["release_evaluation_file_sha256"]) == 28
    for name, expected in release["release_evaluation_file_sha256"].items():
        contents = (root / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(contents).hexdigest() == expected
    assert len(release["checkpoint_sha256"]) == 5
    for name, expected in release["checkpoint_sha256"].items():
        assert hashlib.sha256((root / name / "checkpoint.pt").read_bytes()).hexdigest() == expected
    if release["mismatches"]:
        assert release["reviewed_outcome_changes_accepted"] is True
