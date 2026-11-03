"""Versioned JSON contracts. ScenarioSpec is evaluator-only private data."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1"
VERIFIER_VERSION = "1.2.0"
MAX_VIRTUAL_DURATION = 2**40


def _finite_number(value: Any) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass
class ScenarioSpec:
    scenario_id: str
    archetype: str
    split: str = "train"
    seed: int = 0
    public_task: str = "Please help with my appointment."
    private_goal: dict[str, Any] = field(default_factory=dict)
    initial_world: dict[str, Any] = field(default_factory=dict)
    user: dict[str, Any] = field(default_factory=dict)
    faults: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    max_turns: int = 40
    max_time: float = 120.0
    max_tool_calls: int = 20
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"unsupported scenario schema {self.version}")
        if (
            not isinstance(self.scenario_id, str)
            or not self.scenario_id
            or not isinstance(self.archetype, str)
            or not self.archetype
        ):
            raise ValueError("scenario_id and archetype must be nonempty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            type(self.max_turns) is not int
            or self.max_turns < 1
            or type(self.max_tool_calls) is not int
            or self.max_tool_calls < 1
            or not _finite_number(self.max_time)
            or self.max_time <= 0
            or self.max_time > MAX_VIRTUAL_DURATION
        ):
            raise ValueError(
                "turn/tool limits must be positive integers; max_time must be finite "
                f"and between 0 (exclusive) and {MAX_VIRTUAL_DURATION}"
            )
        if not all(
            isinstance(value, dict) for value in (self.private_goal, self.initial_world, self.user)
        ):
            raise ValueError("goal, initial_world and user must be objects")
        if not isinstance(self.private_goal.get("task", "book"), str) or self.private_goal.get(
            "task", "book"
        ) not in {"book", "reschedule", "cancel"}:
            raise ValueError("unsupported appointment task")
        if not isinstance(self.faults, list):
            raise ValueError("faults must be a list")
        for fault in self.faults:
            error = validate_fault(fault)
            if error:
                raise ValueError(error)
        if not isinstance(self.events, list) or any(
            not isinstance(event, dict)
            or type(event.get("turn", 0)) is not int
            or event.get("turn", 0) < 0
            or not isinstance(event.get("type"), str)
            or event.get("type") not in {"correction", "interrupt", "hangup", "repeat"}
            or (
                event.get("type") == "correction"
                and (
                    not isinstance(event.get("field"), str)
                    or event.get("field") not in FACT_FIELDS
                    or "value" not in event
                )
            )
            for event in self.events
        ):
            raise ValueError("invalid user event schedule")
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scenario must contain finite JSON-compatible values") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioSpec:
        return cls(**deepcopy(data))


ACTION_FIELDS: dict[str, set[str]] = {
    "ASK_CLARIFICATION": {"type", "field"},
    "OFFER_SLOT": {"type", "slot_id"},
    "CONFIRM_INTENT": {"type", "slot_id", "booking_id"},
    "CALL_TOOL": {"type", "tool", "args"},
    "MESSAGE_USER": {"type", "text", "booking_id", "status"},
    "HANDOFF": {"type", "reason"},
    "END": {"type"},
    "WAIT": {"type"},
    "YIELD_TURN": {"type"},
}
FACT_FIELDS = {"account_code", "service", "location", "window", "task", "booking_id"}


def validate_fault(fault: Any) -> str | None:
    """Faults advance virtual time; malformed schedules must not corrupt it."""
    from callflow_rl.simulators.tools import TOOL_SCHEMAS

    if not isinstance(fault, dict):
        return "fault must be an object"
    if not isinstance(fault.get("tool"), str) or fault["tool"] not in TOOL_SCHEMAS:
        return "invalid fault tool"
    if not isinstance(fault.get("kind"), str) or fault["kind"] not in {
        "timeout_before",
        "timeout_after",
        "conflict",
        "stale",
        "reject",
        "latency",
    }:
        return "invalid fault kind"
    occurrence = fault.get("occurrence", 1)
    latency = fault.get("latency", 1.0)
    if type(occurrence) is not int or occurrence < 1:
        return "fault occurrence must be a positive integer"
    if not _finite_number(latency) or latency < 0 or latency > MAX_VIRTUAL_DURATION:
        return f"fault latency must be finite and between 0 and {MAX_VIRTUAL_DURATION}"
    try:
        json.dumps(fault, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return "fault must contain finite JSON-compatible values"
    return None


def validate_action(action: Any) -> str | None:
    if not isinstance(action, dict):
        return "action must be an object"
    try:
        json.dumps(action, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return "action must contain finite JSON-compatible values"
    kind = action.get("type")
    if not isinstance(kind, str) or kind not in ACTION_FIELDS:
        return "unknown action type"
    if set(action) - ACTION_FIELDS[kind]:
        return "unexpected action fields"
    if kind == "ASK_CLARIFICATION" and (
        not isinstance(action.get("field"), str) or action.get("field") not in FACT_FIELDS
    ):
        return "unknown clarification field"
    if kind == "OFFER_SLOT" and not isinstance(action.get("slot_id"), str):
        return "slot_id must be a string"
    if kind == "CONFIRM_INTENT" and any(
        not isinstance(v, str) for k, v in action.items() if k != "type"
    ):
        return "confirmation identifiers must be strings"
    if kind == "CALL_TOOL" and (
        not isinstance(action.get("tool"), str) or not isinstance(action.get("args"), dict)
    ):
        return "tool and args must be a string and object"
    if kind == "MESSAGE_USER":
        if not isinstance(action.get("text"), str):
            return "message text must be a string"
        if "booking_id" in action and not isinstance(action["booking_id"], str):
            return "booking_id must be a string"
        if "status" in action and (
            not isinstance(action["status"], str)
            or action["status"] not in {"active", "cancelled", "unknown"}
        ):
            return "unsupported claimed status"
    if kind == "HANDOFF" and not isinstance(action.get("reason", ""), str):
        return "handoff reason must be a string"
    return None
