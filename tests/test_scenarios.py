"""Frozen scenario integrity, partitioning and independently seeded generation."""

import json
import random
from importlib.resources import files

import pytest

from callflow_rl.contracts import ScenarioSpec
from callflow_rl.scenarios import (
    ARCHETYPES,
    SPLITS,
    archetypes_for,
    generate,
    load_suite,
    manifest_hash,
    suite_metadata,
)
from callflow_rl.scenarios.freeze import build_payloads


def test_frozen_counts_and_partition_by_archetype_and_behavior():
    assert 25 <= len(ARCHETYPES) <= 40
    expected_counts = {"train": 160, "dev": 80, "test": 200, "transfer": 40}
    ids = set()
    archetypes = set()
    behaviors = set()
    for split in SPLITS:
        suite = load_suite(split)
        assert len(suite) == expected_counts[split]
        current_ids = {s.scenario_id for s in suite}
        current_archetypes = {s.archetype for s in suite}
        current_behaviors = {ARCHETYPES[a]["behavior"] for a in current_archetypes}
        assert len(current_ids) == len(suite)
        assert not ids & current_ids
        assert not archetypes & current_archetypes
        assert not behaviors & current_behaviors
        assert current_archetypes == set(archetypes_for(split))
        ids.update(current_ids)
        archetypes.update(current_archetypes)
        behaviors.update(current_behaviors)
        assert suite_metadata()["splits"][split]["sha256"] == manifest_hash(split)


def test_manifest_generation_matches_frozen_data():
    for name, payload in build_payloads().items():
        assert files("callflow_rl.scenarios").joinpath("manifests", name).read_bytes() == payload


@pytest.mark.parametrize("split", SPLITS)
def test_loaded_instances_do_not_share_mutable_world_goal_or_user(split):
    first = load_suite(split)
    second = load_suite(split)
    first[0].private_goal["customer_id"] = "tampered"
    first[0].initial_world["slots"][0]["blocked"] = True
    first[0].user["facts"]["account_code"] = "tampered"
    assert second[0].private_goal["customer_id"] != "tampered"
    assert second[0].user["facts"]["account_code"] != "tampered"
    assert first[0].to_dict() != second[0].to_dict()


@pytest.mark.parametrize("archetype", list(ARCHETYPES))
def test_seeded_generation_roundtrip_and_synthetic_world_invariants(archetype):
    split = ARCHETYPES[archetype]["split"]
    global_rng_state = random.getstate()
    first = generate(archetype, 1472, split)
    second = generate(archetype, 1472, split)
    assert random.getstate() == global_rng_state
    assert first.to_dict() == second.to_dict()
    assert first.to_dict() != generate(archetype, 1473, split).to_dict()
    assert (
        ScenarioSpec.from_dict(json.loads(json.dumps(first.to_dict()))).to_dict() == first.to_dict()
    )
    world = first.initial_world
    customer_ids = set(world["customers"])
    assert first.private_goal["customer_id"] in customer_ids
    assert all(c["name"].startswith("Synthetic ") for c in world["customers"].values())
    slot_ids = {s["id"] for s in world["slots"]}
    assert len(slot_ids) == len(world["slots"])
    assert set(first.private_goal["allowed_slots"]) <= slot_ids
    for booking in world["bookings"].values():
        assert booking["customer_id"] in customer_ids
        assert booking["slot_id"] in slot_ids
    for slot in world["slots"]:
        occupied = sum(
            b["status"] == "active" and b["slot_id"] == slot["id"]
            for b in world["bookings"].values()
        )
        assert occupied <= slot["capacity"]


def test_evaluation_only_behaviors_never_generated_in_training():
    train = load_suite("train")
    assert not any(s.events and any(e["type"] == "correction" for e in s.events) for s in train)
    assert not any(f["kind"] in {"stale", "reject"} for s in train for f in s.faults)
    assert all(slot["capacity"] == 1 for s in train for slot in s.initial_world["slots"])
    assert any(
        slot["capacity"] == 2 for s in load_suite("transfer") for slot in s.initial_world["slots"]
    )
    assert any(any(e["type"] == "correction" for e in s.events) for s in load_suite("test"))


@pytest.mark.parametrize("bad_seed", [-1, 1.5, True, "1"])
def test_reject_bad_seed(bad_seed):
    with pytest.raises(ValueError):
        generate("book_direct", bad_seed)


def test_reject_unknown_archetypes_splits_and_cross_split_generation():
    with pytest.raises(ValueError):
        generate("invented", 1)
    with pytest.raises(ValueError):
        generate("book_correction_time", 1, "train")
    with pytest.raises(ValueError):
        load_suite("invented")
