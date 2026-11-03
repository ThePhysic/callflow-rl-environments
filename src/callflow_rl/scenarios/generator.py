"""Seeded synthetic worlds with explicit, authored behavioral differences."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from callflow_rl.contracts import ScenarioSpec

from .catalog import ARCHETYPES


def generate(archetype: str, seed: int, split: str = "train") -> ScenarioSpec:
    """Generate one episode without global RNGs or crossing split ownership.

    Evaluation uses committed manifests, not regeneration. Regeneration is an
    explicit maintenance operation so generator edits cannot silently alter a test.
    """
    if archetype not in ARCHETYPES:
        raise ValueError(f"Unknown archetype {archetype!r}")
    metadata = ARCHETYPES[archetype]
    if metadata["split"] != split:
        raise ValueError(f"{archetype} belongs to {metadata['split']}, not {split}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(seed)
    task = metadata["task"]
    customer_id = f"customer-{rng.randrange(10000, 99999)}"
    other_id = f"other-{rng.randrange(10000, 99999)}"
    account_code = f"SYN-{rng.randrange(100000, 999999)}"
    location = rng.choice(["north", "central", "south"])
    other_location = next(x for x in ["north", "central", "south"] if x != location)
    service = rng.choice(["tire_rotation", "oil_change"])
    base = datetime(2026, 10, 6, 13, tzinfo=timezone(timedelta(hours=-7)))
    base += timedelta(days=rng.randrange(20), minutes=30 * rng.randrange(4))
    slots: list[dict[str, Any]] = []
    for index in range(4):
        start = base + timedelta(hours=index)
        slots.append(
            {
                "id": f"slot-{seed}-{index}",
                "service": service,
                "location": location,
                "start": start.isoformat(),
                "end": (start + timedelta(minutes=30)).isoformat(),
                "capacity": 1,
                "version": 0,
                "blocked": False,
            }
        )
    ids = [slot["id"] for slot in slots]
    allowed = [ids[0]]
    facts: dict[str, Any] = {
        "task": task,
        "account_code": account_code,
        "service": service,
        "location": location,
        "window": list(allowed),
    }
    initial_known = list(facts)
    user: dict[str, Any] = {
        "facts": facts,
        "initial_known": initial_known,
        "accept_alternatives": False,
        "decline_all": False,
        "tolerance": 40,
        "persona": rng.choice(["brief", "patient", "specific"]),
    }
    faults: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    outcome = "completed"
    bookings: dict[str, dict[str, Any]] = {}
    booking_id = f"booking-original-{seed}"
    if task in ("reschedule", "cancel"):
        original = slots[3]
        bookings[booking_id] = {
            "id": booking_id,
            "customer_id": customer_id,
            "service": service,
            "location": location,
            "slot_id": original["id"],
            "status": "active",
            "version": 0,
        }
        facts["booking_id"] = booking_id
        initial_known.append("booking_id")

    missing = {
        "book_identity_missing": "account_code",
        "book_service_missing": "service",
        "book_location_missing": "location",
        "book_window_missing": "window",
        "book_ambiguous_date": "window",
    }
    if archetype in missing:
        initial_known.remove(missing[archetype])
    if archetype == "book_all_missing":
        initial_known[:] = ["task"]
    if archetype in ("book_flexible", "book_conflict", "reschedule_conflict"):
        allowed = ids[:3]
        facts["window"] = list(allowed)
    write_tool = {
        "book": "create_booking",
        "reschedule": "reschedule_booking",
        "cancel": "cancel_booking",
    }[task]
    if archetype in ("book_timeout_before", "cancel_timeout_before"):
        faults.append({"tool": write_tool, "kind": "timeout_before", "occurrence": 1})
    if archetype in (
        "book_timeout_after",
        "reschedule_timeout_after",
        "cancel_timeout_after",
        "book_duplicate_request",
    ):
        faults.append({"tool": write_tool, "kind": "timeout_after", "occurrence": 1})
    if archetype in ("book_conflict", "reschedule_conflict"):
        faults.append({"tool": write_tool, "kind": "conflict", "occurrence": 1})
    if archetype == "book_stale_read":
        slots[0]["version"] = 1
        faults.append({"tool": "get_availability", "kind": "stale", "occurrence": 1})
    if archetype == "book_rejected_arguments":
        faults.append({"tool": "get_availability", "kind": "reject", "occurrence": 1})
    if archetype == "book_slow_tools":
        faults.extend(
            {"tool": tool, "kind": "latency", "occurrence": 1, "latency": 4}
            for tool in ["lookup_customer", "get_availability", write_tool]
        )
    if archetype in ("book_no_availability", "reschedule_no_capacity"):
        for slot in slots[:3]:
            slot["blocked"] = True
        outcome = "handoff"
    if archetype == "book_decline":
        user["decline_all"] = True
        user["accept_alternatives"] = False
        outcome = "refused"
    if archetype == "book_interruption":
        events.append({"turn": 2, "type": "interrupt"})
    if archetype == "cancel_interrupted":
        events.append({"turn": 4, "type": "interrupt"})
    if archetype in ("book_correction_time", "reschedule_correction"):
        allowed = [ids[1]]
        events.append({"turn": 3, "type": "correction", "field": "window", "value": allowed})
    if archetype == "book_correction_location":
        slots[1]["location"] = other_location
        allowed = [ids[1]]
        events.extend(
            [
                {"turn": 3, "type": "correction", "field": "location", "value": other_location},
                {"turn": 3, "type": "correction", "field": "window", "value": allowed},
            ]
        )
    if archetype == "book_duplicate_request":
        events.append({"turn": 6, "type": "repeat"})
    if archetype == "book_dst_fold":
        slots[0]["start"] = "2026-11-01T01:30:00-07:00"
        slots[0]["end"] = "2026-11-01T01:50:00-07:00"
        slots[1]["start"] = "2026-11-01T01:30:00-08:00"
        slots[1]["end"] = "2026-11-01T01:50:00-08:00"
        allowed = [ids[1]]
        facts["window"] = list(allowed)
        initial_known.remove("window")

    policy: dict[str, Any] = {"handoff_required": False, "permitted_services": [service]}
    if archetype == "policy_handoff":
        policy["handoff_required"] = True
        outcome = "handoff"
    if archetype == "book_policy_refusal":
        policy["permitted_services"] = []
        outcome = "refused"

    if archetype == "transfer_mixed_calendar":
        # Training calendars have one service/location. This transfer family
        # introduces interleaved distractors and a later acceptable candidate.
        slots[0]["service"] = "inspection"
        slots[1]["location"] = other_location
        allowed = [ids[2]]
        facts["window"] = list(allowed)
    if archetype == "transfer_shared_capacity":
        # A capacity-two slot with one other customer is still admissible.
        slots[0]["capacity"] = 2
        bookings[f"booking-other-{seed}"] = {
            "id": f"booking-other-{seed}",
            "customer_id": other_id,
            "service": service,
            "location": location,
            "slot_id": ids[0],
            "status": "active",
            "version": 0,
        }

    customers = {
        customer_id: {"name": f"Synthetic Customer {seed}", "account_code": account_code},
        other_id: {"name": "Synthetic Other Customer", "account_code": f"OTHER-{seed}"},
    }
    if rng.choice([False, True]):
        customers = dict(reversed(list(customers.items())))
    goal: dict[str, Any] = {
        "task": task,
        "customer_id": customer_id,
        "service": service,
        "allowed_locations": [
            other_location if archetype == "book_correction_location" else location
        ],
        "allowed_slots": list(allowed),
        "outcome": outcome,
    }
    if task in ("reschedule", "cancel"):
        goal["booking_id"] = booking_id
    public_task = {
        "book": "Please help me book a service appointment.",
        "reschedule": "Please move my existing appointment to another acceptable slot.",
        "cancel": "Please cancel my existing appointment.",
    }[task]
    if split == "dev":
        public_task = "Could you assist with this request? " + public_task
    if split == "test":
        public_task = "I am calling about my reservation. " + public_task
    if archetype == "book_ambiguous_date":
        public_task += " I said Tuesday afternoon; please check the exact slot with me."
    if archetype == "book_dst_fold":
        public_task += " I mean the second 1:30 AM after the clocks change, with UTC offset -08:00."
    return ScenarioSpec(
        scenario_id=f"{archetype}:{seed}",
        archetype=archetype,
        split=split,
        seed=seed,
        public_task=public_task,
        private_goal=goal,
        initial_world={
            "customers": customers,
            "slots": slots,
            "bookings": bookings,
            "policy": policy,
        },
        user=user,
        faults=faults,
        events=events,
        max_turns=40,
        max_time=120,
        max_tool_calls=20,
        version="1",
    )
