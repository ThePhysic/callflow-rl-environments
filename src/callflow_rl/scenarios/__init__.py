"""Synthetic authored scenarios and immutable, integrity-checked evaluation suites."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any

from callflow_rl.contracts import ScenarioSpec

from .catalog import ARCHETYPES, SPLITS, SUITE_VERSION, archetypes_for, challenge_slices
from .generator import generate

__all__ = [
    "ARCHETYPES", "SPLITS", "SUITE_VERSION", "archetypes_for", "challenge_slices",
    "generate", "load_suite", "manifest_hash", "suite_metadata",
]


def suite_metadata() -> dict[str, Any]:
    return json.loads(files(__package__).joinpath("manifests", "index.json").read_text("utf-8"))


def _manifest_bytes(split: str) -> bytes:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    payload = files(__package__).joinpath("manifests", f"{split}.jsonl").read_bytes()
    expected = suite_metadata()["splits"][split]["sha256"]
    actual = hashlib.sha256(payload).hexdigest()
    if expected != actual:
        raise ValueError(f"{split} manifest integrity failure: expected {expected}, got {actual}")
    return payload


def load_suite(split: str) -> list[ScenarioSpec]:
    """Return fresh instances from the frozen, hashed manifest; no shared state."""
    result = [ScenarioSpec.from_dict(json.loads(line)) for line in _manifest_bytes(split).splitlines()]
    metadata = suite_metadata()["splits"][split]
    if len(result) != metadata["count"]:
        raise ValueError(f"{split} manifest count mismatch")
    if any(s.split != split or s.archetype not in archetypes_for(split) for s in result):
        raise ValueError(f"{split} manifest contains an archetype from another split")
    return result


def manifest_hash(split: str) -> str:
    return hashlib.sha256(_manifest_bytes(split)).hexdigest()
