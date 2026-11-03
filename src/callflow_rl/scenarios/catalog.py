"""Authored scenario archetypes and split ownership, independent of parameter seeds."""

from typing import Any


def _entry(split: str, task: str, behavior: str, *slices: str) -> dict[str, Any]:
    return {"split": split, "task": task, "behavior": behavior, "slices": list(slices)}


ARCHETYPES: dict[str, dict[str, Any]] = {
    "book_direct": _entry("train", "book", "complete_request", "clean"),
    "book_identity_missing": _entry("train", "book", "missing_identity", "ambiguous"),
    "book_service_missing": _entry("train", "book", "missing_service", "ambiguous"),
    "book_location_missing": _entry("train", "book", "missing_location", "ambiguous"),
    "book_window_missing": _entry("train", "book", "missing_window", "ambiguous"),
    "book_all_missing": _entry("train", "book", "incremental_clarification", "ambiguous"),
    "book_flexible": _entry("train", "book", "multiple_acceptable_slots", "clean"),
    "book_timeout_before": _entry("train", "book", "precommit_timeout", "fault"),
    "book_timeout_after": _entry("train", "book", "ambiguous_commit", "fault", "timeout_after"),
    "book_conflict": _entry("train", "book", "optimistic_conflict", "fault"),
    "book_no_availability": _entry("train", "book", "unavailable_calendar", "policy"),
    "book_decline": _entry("train", "book", "all_alternatives_declined", "policy"),
    "book_interruption": _entry("train", "book", "early_interruption", "interruption"),
    "book_slow_tools": _entry("train", "book", "tool_latency", "fault"),
    "reschedule_direct": _entry("train", "reschedule", "atomic_move", "clean"),
    "reschedule_timeout_after": _entry(
        "train", "reschedule", "ambiguous_move", "fault", "timeout_after"
    ),
    "reschedule_conflict": _entry("train", "reschedule", "conflicting_move", "fault"),
    "cancel_direct": _entry("train", "cancel", "authorized_cancellation", "clean"),
    "cancel_timeout_before": _entry("train", "cancel", "precommit_cancel_timeout", "fault"),
    "policy_handoff": _entry("train", "book", "explicit_handoff", "policy"),
    "book_ambiguous_date": _entry("dev", "book", "relative_date_clarification", "ambiguous"),
    "book_rejected_arguments": _entry("dev", "book", "transient_argument_rejection", "fault"),
    "reschedule_no_capacity": _entry("dev", "reschedule", "retain_original_if_full", "policy"),
    "cancel_timeout_after": _entry(
        "dev", "cancel", "ambiguous_cancellation", "fault", "timeout_after"
    ),
    "book_correction_time": _entry("test", "book", "accepted_window_correction", "ambiguous"),
    "book_correction_location": _entry("test", "book", "accepted_location_correction", "ambiguous"),
    "book_stale_read": _entry("test", "book", "stale_version_read", "fault"),
    "book_duplicate_request": _entry(
        "test", "book", "duplicate_request_after_lost_reply", "fault", "timeout_after"
    ),
    "reschedule_correction": _entry("test", "reschedule", "move_window_correction", "ambiguous"),
    "cancel_interrupted": _entry(
        "test", "cancel", "late_cancellation_interruption", "interruption"
    ),
    "book_dst_fold": _entry("test", "book", "dst_offset_disambiguation", "ambiguous", "timezone"),
    "book_policy_refusal": _entry("test", "book", "restricted_service_refusal", "policy"),
    "transfer_mixed_calendar": _entry(
        "transfer", "book", "interleaved_calendar_pattern", "transfer", "clean"
    ),
    "transfer_shared_capacity": _entry(
        "transfer", "book", "shared_capacity_pattern", "transfer", "clean"
    ),
}

SPLITS = ("train", "dev", "test", "transfer")
SUITE_VERSION = "1.0.0"


def archetypes_for(split: str) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    return [name for name, entry in ARCHETYPES.items() if entry["split"] == split]


def challenge_slices(archetype: str) -> list[str]:
    return list(ARCHETYPES[archetype]["slices"])
