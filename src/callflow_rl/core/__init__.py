"""Canonical hashes and isolated pseudo-random streams."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def rng_seed(seed: int, stream: str) -> int:
    return int(hashlib.sha256(f"{seed}:{stream}".encode()).hexdigest()[:16], 16)


def tuple_tree(value: Any) -> Any:
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value
