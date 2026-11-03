"""Explicit manifest maintenance: ``python -m callflow_rl.scenarios.freeze --check``."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .catalog import ARCHETYPES, SPLITS, SUITE_VERSION, archetypes_for
from .generator import generate

INSTANCES_PER_ARCHETYPE = {"train": 8, "dev": 20, "test": 25, "transfer": 20}
SEED_BASE = {"train": 100_000, "dev": 200_000, "test": 300_000, "transfer": 400_000}


def build_payloads() -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    index = {"version": SUITE_VERSION, "synthetic_only": True, "splits": {}}
    for split in SPLITS:
        records = []
        for family_index, archetype in enumerate(archetypes_for(split)):
            for instance in range(INSTANCES_PER_ARCHETYPE[split]):
                seed = SEED_BASE[split] + 100 * family_index + instance
                records.append(generate(archetype, seed, split).to_dict())
        payload = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
        ).encode("utf-8")
        payloads[f"{split}.jsonl"] = payload
        index["splits"][split] = {  # type: ignore[index]
            "count": len(records),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "archetypes": archetypes_for(split),
            "behaviors": [ARCHETYPES[a]["behavior"] for a in archetypes_for(split)],
        }
    payloads["index.json"] = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify generator matches frozen files")
    mode.add_argument(
        "--write", action="store_true", help="explicitly rewrite all suite definitions"
    )
    args = parser.parse_args()
    target = Path(__file__).parent / "manifests"
    for name, payload in build_payloads().items():
        path = target / name
        if args.write:
            path.write_bytes(payload)
        elif not path.exists() or path.read_bytes() != payload:
            raise SystemExit(f"Frozen manifest differs from generator: {path}")
    print("Frozen scenario manifests match." if args.check else "Wrote frozen scenario manifests.")


if __name__ == "__main__":
    main()
