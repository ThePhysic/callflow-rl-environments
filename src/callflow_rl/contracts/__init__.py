"""Versioned JSON contracts. ScenarioSpec is evaluator-only private data."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1"
VERIFIER_VERSION = "1.1.0"


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
        if not self.scenario_id or not self.archetype:
            raise ValueError("scenario_id and archetype must be nonempty")
        if self.max_turns < 1 or self.max_time <= 0 or self.max_tool_calls < 1:
            raise ValueError("scenario limits must be positive")
        if self.private_goal.get("task", "book") not in {"book", "reschedule", "cancel"}:
            raise ValueError("unsupported appointment task")

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
