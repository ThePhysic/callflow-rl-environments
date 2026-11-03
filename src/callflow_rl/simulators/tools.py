"""Scheduling service. Authorization uses a session, never the private goal."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

TOOL_SCHEMAS: dict[str, dict[str, str]] = {
    "lookup_customer": {"account_code": "str"},
    "list_services": {},
    "get_availability": {"service": "str", "location": "str", "window?": "list[str]"},
    "get_booking": {"customer_id": "str", "booking_id?": "str"},
    "create_booking": {
        "customer_id": "str",
        "slot_id": "str",
        "expected_version": "int",
        "idempotency_key": "str",
    },
    "reschedule_booking": {
        "customer_id": "str",
        "booking_id": "str",
        "slot_id": "str",
        "expected_version": "int",
        "booking_version": "int",
        "idempotency_key": "str",
    },
    "cancel_booking": {
        "customer_id": "str",
        "booking_id": "str",
        "booking_version": "int",
        "idempotency_key": "str",
    },
}
WRITE_TOOLS = {"create_booking", "reschedule_booking", "cancel_booking"}


def validate_tool(name: str, args: dict[str, Any]) -> str | None:
    if name not in TOOL_SCHEMAS:
        return "tool_not_allowed"
    schema = TOOL_SCHEMAS[name]
    allowed = {k.rstrip("?") for k in schema}
    required = {k for k in schema if not k.endswith("?")}
    if set(args) - allowed or required - set(args):
        return "argument_schema"
    for key, expected in schema.items():
        key = key.rstrip("?")
        if key not in args:
            continue
        value = args[key]
        if expected == "str" and (not isinstance(value, str) or not value):
            return "argument_type"
        if expected == "int" and (type(value) is not int or value < 0):
            return "argument_type"
        if expected == "list[str]" and (
            not isinstance(value, list) or any(not isinstance(x, str) for x in value)
        ):
            return "argument_type"
    return None


def slot_time_valid(slot: dict[str, Any]) -> bool:
    try:
        start = datetime.fromisoformat(slot["start"])
        end = datetime.fromisoformat(slot["end"])
        return start.tzinfo is not None and end.tzinfo is not None and end > start
    except (ValueError, TypeError, KeyError):
        return False


class SchedulingService:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    @property
    def world(self) -> dict[str, Any]:
        return self.state["world"]

    def _slot(self, slot_id: str) -> dict[str, Any] | None:
        return next((s for s in self.world["slots"] if s["id"] == slot_id), None)

    def _active(self, slot_id: str, exclude: str | None = None) -> int:
        return sum(
            b["status"] == "active" and b["slot_id"] == slot_id and b["id"] != exclude
            for b in self.world["bookings"].values()
        )

    def _authorized(self, customer_id: str) -> bool:
        return (
            customer_id == self.state["session_customer_id"]
            and customer_id == self.state["authenticated_customer_id"]
        )

    def _consent(self, task: str, args: dict[str, Any]) -> bool:
        c = self.state["consent"]
        return bool(
            c
            and c["task"] == task
            and c["customer_id"] == args["customer_id"]
            and c.get("slot_id") == args.get("slot_id")
            and c.get("booking_id") == args.get("booking_id")
            and c["revision"] == self.state["user_revision"]
        )

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        error = validate_tool(name, args)
        if error:
            return {"ok": False, "error": error}
        if name == "lookup_customer":
            customer = next(
                (
                    dict(c, id=cid)
                    for cid, c in self.world["customers"].items()
                    if c["account_code"] == args["account_code"]
                ),
                None,
            )
            if customer is None:
                return {"ok": False, "error": "customer_not_found"}
            if customer["id"] != self.state["session_customer_id"]:
                return {"ok": False, "error": "unauthorized_customer"}
            self.state["authenticated_customer_id"] = customer["id"]
            return {"ok": True, "customer": customer}
        if name == "list_services":
            return {"ok": True, "services": sorted({s["service"] for s in self.world["slots"]})}
        if name == "get_availability":
            slots = [
                deepcopy(s)
                for s in self.world["slots"]
                if s["service"] == args["service"]
                and s["location"] == args["location"]
                and not s.get("blocked", False)
                and self._active(s["id"]) < s.get("capacity", 1)
                and ("window" not in args or s["id"] in args["window"])
                and slot_time_valid(s)
            ]
            return {"ok": True, "slots": slots}
        if not self._authorized(args["customer_id"]):
            return {"ok": False, "error": "unauthorized_customer"}
        if name == "get_booking":
            return {
                "ok": True,
                "bookings": [
                    deepcopy(b)
                    for b in self.world["bookings"].values()
                    if b["customer_id"] == args["customer_id"]
                    and ("booking_id" not in args or b["id"] == args["booking_id"])
                ],
            }
        key = args["idempotency_key"]
        request = {"tool": name, "args": {k: v for k, v in args.items() if k != "idempotency_key"}}
        prior = self.state["idempotency"].get(key)
        if prior:
            if prior["request"] != request:
                return {"ok": False, "error": "idempotency_key_reused"}
            return dict(deepcopy(prior["response"]), idempotent_replay=True)
        task = {
            "create_booking": "book",
            "reschedule_booking": "reschedule",
            "cancel_booking": "cancel",
        }[name]
        if not self._consent(task, args):
            return {"ok": False, "error": "consent_required"}
        policy = self.world.get("policy", {})
        if policy.get("handoff_required", False):
            return {"ok": False, "error": "handoff_required"}
        old = None
        if name != "create_booking":
            old = self.world["bookings"].get(args["booking_id"])
            if not old or old["customer_id"] != args["customer_id"]:
                return {"ok": False, "error": "booking_not_found"}
            if old["status"] != "active":
                return {"ok": False, "error": "booking_not_active"}
            if old["version"] != args["booking_version"]:
                return {"ok": False, "error": "booking_version_conflict"}
        slot = None
        if name != "cancel_booking":
            slot = self._slot(args["slot_id"])
            if not slot or not slot_time_valid(slot):
                return {"ok": False, "error": "invalid_slot"}
            if slot["service"] not in policy.get("permitted_services", [slot["service"]]):
                return {"ok": False, "error": "service_not_permitted"}
            if slot["location"] not in policy.get("permitted_locations", [slot["location"]]):
                return {"ok": False, "error": "location_not_permitted"}
            if self.state["consent"]["service"] != slot["service"]:
                return {"ok": False, "error": "consent_service_mismatch"}
            if slot["version"] != args["expected_version"]:
                return {"ok": False, "error": "slot_version_conflict"}
            if slot.get("blocked", False) or self._active(
                slot["id"], args.get("booking_id")
            ) >= slot.get("capacity", 1):
                return {"ok": False, "error": "slot_unavailable"}
            if name == "create_booking" and any(
                b["customer_id"] == args["customer_id"]
                and b["status"] == "active"
                and b["slot_id"] == slot["id"]
                for b in self.world["bookings"].values()
            ):
                return {"ok": False, "error": "duplicate_booking"}
        before = deepcopy(old)
        if name == "create_booking":
            assert slot is not None
            booking_id = f"B{self.state['next_booking']:06d}"
            self.state["next_booking"] += 1
            booking = {
                "id": booking_id,
                "customer_id": args["customer_id"],
                "service": slot["service"],
                "location": slot["location"],
                "slot_id": slot["id"],
                "status": "active",
                "version": 0,
            }
            self.world["bookings"][booking_id] = booking
        elif name == "reschedule_booking":
            assert old is not None and slot is not None
            old_slot = self._slot(old["slot_id"])
            if old_slot and old_slot["id"] != slot["id"]:
                old_slot["version"] += 1
            booking = old
            booking.update(slot_id=slot["id"], service=slot["service"], location=slot["location"])
            booking["version"] += 1
        else:
            assert old is not None
            booking = old
            booking["status"] = "cancelled"
            booking["version"] += 1
            old_slot = self._slot(old["slot_id"])
            if old_slot:
                old_slot["version"] += 1
        if slot:
            slot["version"] += 1
        self.state["audit"].append(
            {
                "kind": "write",
                "tool": name,
                "task": task,
                "customer_id": args["customer_id"],
                "before": before,
                "after": deepcopy(booking),
                "consent": deepcopy(self.state["consent"]),
                "turn": self.state["turn"],
                "time": self.state["clock"],
                "idempotency_key": key,
            }
        )
        response = {"ok": True, "booking": deepcopy(booking)}
        self.state["idempotency"][key] = {"request": request, "response": deepcopy(response)}
        return response
