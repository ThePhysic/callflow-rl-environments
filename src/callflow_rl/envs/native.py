"""Partially observed appointment environment with evaluator-only snapshots.

Every mutable field is per episode. Tool faults deliberately expose identical
timeout observations before and after commit; only state inspection or an
idempotent retry can resolve the ambiguity.
"""

from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from typing import Any

from callflow_rl.contracts import ScenarioSpec, validate_action
from callflow_rl.core import fingerprint, rng_seed, tuple_tree
from callflow_rl.simulators.tools import TOOL_SCHEMAS, WRITE_TOOLS, SchedulingService, validate_tool
from callflow_rl.simulators.user import UserSimulator
from callflow_rl.verifiers import verify


class CallFlowEnv:
    def __init__(self, scenario: ScenarioSpec | None = None, reward_mode: str = "shaped") -> None:
        if reward_mode not in {"sparse", "shaped"}:
            raise ValueError("reward_mode must be sparse or shaped")
        self.scenario = deepcopy(scenario)
        self.reward_mode = reward_mode
        self.state: dict[str, Any] = {}
        self._rng = random.Random(0)
        self._seed = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if options and "scenario" in options:
            item = options["scenario"]
            self.scenario = (
                ScenarioSpec.from_dict(item) if isinstance(item, dict) else deepcopy(item)
            )
        if self.scenario is None:
            from callflow_rl.scenarios import generate

            self.scenario = generate("book_direct", seed or 0)
        assert self.scenario is not None
        self._seed = self.scenario.seed if seed is None else seed
        self._rng = random.Random(rng_seed(self._seed, "user"))
        world = deepcopy(self.scenario.initial_world)
        world.setdefault("customers", {})
        world.setdefault("slots", [])
        world.setdefault("bookings", {})
        world.setdefault("policy", {})
        for slot in world["slots"]:
            slot.setdefault("version", 0)
            slot.setdefault("capacity", 1)
            slot.setdefault("blocked", False)
        for booking in world["bookings"].values():
            booking.setdefault("version", 0)
            booking.setdefault("status", "active")
        config = deepcopy(self.scenario.user)
        truth = deepcopy(config.get("facts", {}))
        truth.setdefault("task", "book")
        known = config.get("initial_known", ["task"])
        facts = {k: deepcopy(v) for k, v in truth.items() if k in known or k == "task"}
        principal = next(
            (
                cid
                for cid, c in world["customers"].items()
                if c.get("account_code") == truth.get("account_code")
            ),
            None,
        )
        next_booking = 1
        while f"B{next_booking:06d}" in world["bookings"]:
            next_booking += 1
        self.state = {
            "world": world,
            "user_truth": truth,
            "user_config": config,
            "facts": facts,
            "session_customer_id": principal,
            "authenticated_customer_id": None,
            "customer": None,
            "slots": [],
            "bookings": [],
            "consent": None,
            "offered_slot": None,
            "offer_accepted": False,
            "user_declined": False,
            "declined_slots": [],
            "user_revision": 0,
            "interrupted": False,
            "user_ended": False,
            "clarifications": 0,
            "user_message": self.scenario.public_task,
            "last_tool": None,
            "last_error": None,
            "result_booking": None,
            "reported": False,
            "refusal_sent": False,
            "availability_checked": False,
            "pending_write": None,
            "history": [
                {"speaker": "user", "turn": 0, "time": 0.0, "text": self.scenario.public_task}
            ],
            "audit": [],
            "clock": 0.0,
            "turn": 0,
            "tool_calls": 0,
            "invalid_action_count": 0,
            "terminated": False,
            "truncated": False,
            "termination_reason": "running",
            "next_booking": next_booking,
            "idempotency": {},
            "faults": deepcopy(self.scenario.faults),
            "faults_applied": [],
            "tool_occurrences": {},
            "events_applied": [],
            "milestones": [],
            "reward_total": 0.0,
            "rng_subseeds": {
                "user": rng_seed(self._seed, "user"),
                "tools": rng_seed(self._seed, "tools"),
                "events": rng_seed(self._seed, "events"),
            },
        }
        self._check_user_consistency()
        return self.observation(), self._info()

    def _user(self) -> UserSimulator:
        return UserSimulator(self.state, self._rng)

    def _check_user_consistency(self) -> None:
        s = self.state
        valid = all(s["user_truth"].get(k) == v for k, v in s["facts"].items())
        c = s["consent"]
        if c:
            valid = (
                valid
                and c["customer_id"] == s["session_customer_id"]
                and c["task"] == s["user_truth"].get("task")
                and c["service"] == s["user_truth"].get("service")
                and c["revision"] == s["user_revision"]
            )
        s["audit"].append(
            {
                "kind": "simulator_check",
                "consistent": bool(valid),
                "turn": s["turn"],
                "time": s["clock"],
            }
        )

    def observation(self) -> dict[str, Any]:
        if not self.state:
            raise RuntimeError("reset must be called first")
        s = self.state
        assert self.scenario is not None
        public = {
            "schema_version": "1",
            "user_message": s["user_message"],
            "facts": s["facts"],
            "customer": s["customer"],
            "slots": s["slots"],
            "bookings": s["bookings"],
            "consent": s["consent"],
            "offered_slot": s["offered_slot"],
            "offer_accepted": s["offer_accepted"],
            "rejected_slots": s["declined_slots"],
            "user_declined": s["user_declined"],
            "interrupted": s["interrupted"],
            "user_ended": s["user_ended"],
            "last_tool": s["last_tool"],
            "last_error": s["last_error"],
            "result_booking": s["result_booking"],
            "reported": s["reported"],
            "refusal_sent": s["refusal_sent"],
            "availability_checked": s["availability_checked"],
            "pending_write": s["pending_write"],
            "policy": s["world"].get("policy", {}),
            "available_tools": TOOL_SCHEMAS,
            "history": s["history"],
            "clock": s["clock"],
            "request_key": f"request-{s['user_revision']}",
            "remaining": {
                "turns": max(0, self.scenario.max_turns - s["turn"]),
                "time": max(0.0, self.scenario.max_time - s["clock"]),
                "tools": max(0, self.scenario.max_tool_calls - s["tool_calls"]),
            },
        }
        # No private goal, future event/fault schedule, backend rows or audit labels.
        return deepcopy(public)

    def _info(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict(),
            "state_hash": self.state_hash(),
            "step_count": self.state["turn"],
            "tool_call_count": self.state["tool_calls"],
            "virtual_time": self.state["clock"],
            "reward_explanation": deepcopy(self.state.get("reward_explanation", {})),
        }

    def verdict(self) -> dict[str, Any]:
        if self.scenario is None or not self.state:
            raise RuntimeError("reset must be called first")
        return verify(self.scenario, self.state)

    def snapshot(self) -> dict[str, Any]:
        if self.scenario is None or not self.state:
            raise RuntimeError("reset must be called first")
        # JSON conversion makes the external snapshot independent of all live mutable data.
        return json.loads(
            json.dumps(
                {
                    "snapshot_version": "1",
                    "scenario": self.scenario.to_dict(),
                    "state": self.state,
                    "seed": self._seed,
                    "reward_mode": self.reward_mode,
                    "rng_state": self._rng.getstate(),
                }
            )
        )

    def restore(self, snapshot: dict[str, Any]) -> None:
        snap = deepcopy(snapshot)
        if snap["snapshot_version"] != "1":
            raise ValueError("unsupported snapshot schema")
        self.scenario = ScenarioSpec.from_dict(snap["scenario"])
        self.state = snap["state"]
        self._seed = snap["seed"]
        self.reward_mode = snap["reward_mode"]
        self._rng = random.Random()
        self._rng.setstate(tuple_tree(snap["rng_state"]))

    def state_hash(self) -> str:
        return fingerprint(self.snapshot())

    def inject_fault(self, fault: dict[str, Any]) -> None:
        if (
            not isinstance(fault, dict)
            or fault.get("kind")
            not in {"timeout_before", "timeout_after", "conflict", "stale", "reject", "latency"}
            or fault.get("tool") not in TOOL_SCHEMAS
        ):
            raise ValueError("invalid fault schedule entry")
        self.state["faults"].append(deepcopy(fault))

    def _invalid(self, error: str) -> None:
        self.state["invalid_action_count"] += 1
        self.state["last_error"] = error
        self.state["history"].append(
            {
                "speaker": "environment",
                "turn": self.state["turn"],
                "time": self.state["clock"],
                "error": error,
            }
        )

    def _apply_tool_response(self, name: str, result: dict[str, Any]) -> None:
        s = self.state
        s["last_tool"] = dict(deepcopy(result), tool=name)
        if not result["ok"]:
            s["last_error"] = result["error"]
            if result["error"] not in {"timeout"}:
                s["pending_write"] = None
            return
        if name == "lookup_customer":
            s["customer"] = deepcopy(result["customer"])
        elif name == "get_availability":
            s["slots"] = deepcopy(result["slots"])
            s["availability_checked"] = True
        elif name == "get_booking":
            s["bookings"] = deepcopy(result["bookings"])
            # Reconciliation requires observable evidence matching the current operation.
            facts = s["facts"]
            task = facts.get("task")
            candidates = [
                b
                for b in result["bookings"]
                if (
                    task == "cancel"
                    and b["status"] == "cancelled"
                    and b["id"] == facts.get("booking_id")
                )
                or (
                    task in {"book", "reschedule"}
                    and b["status"] == "active"
                    and b["slot_id"] == (s["consent"] or {}).get("slot_id")
                    and (task != "reschedule" or b["id"] == facts.get("booking_id"))
                )
            ]
            if candidates:
                s["result_booking"] = deepcopy(candidates[0])
                s["pending_write"] = None
        elif name in WRITE_TOOLS:
            s["result_booking"] = deepcopy(result["booking"])
            s["bookings"] = [b for b in s["bookings"] if b["id"] != result["booking"]["id"]] + [
                deepcopy(result["booking"])
            ]
            s["pending_write"] = None

    def _call_tool(self, action: dict[str, Any]) -> None:
        s = self.state
        name, args = action["tool"], action["args"]
        error = validate_tool(name, args)
        if error:
            self._invalid(error)
            self._apply_tool_response(name, {"ok": False, "error": error})
            return
        s["tool_calls"] += 1
        s["tool_occurrences"][name] = s["tool_occurrences"].get(name, 0) + 1
        fault = None
        for index, entry in enumerate(s["faults"]):
            if (
                index not in s["faults_applied"]
                and entry["tool"] == name
                and entry.get("occurrence", 1) == s["tool_occurrences"][name]
            ):
                fault = entry
                s["faults_applied"].append(index)
                break
        if name in WRITE_TOOLS:
            s["pending_write"] = deepcopy(action)
        if fault:
            s["clock"] += float(fault.get("latency", 1.0))
            s["audit"].append(
                {"kind": "fault", "fault": deepcopy(fault), "turn": s["turn"], "time": s["clock"]}
            )
        kind = fault["kind"] if fault else None
        service = SchedulingService(s)
        result: dict[str, Any]
        if kind == "timeout_before":
            result = {"ok": False, "error": "timeout", "ambiguous": True}
        elif kind == "reject":
            result = {"ok": False, "error": "arguments_rejected"}
        else:
            if kind == "conflict" and "slot_id" in args:
                slot = next((x for x in s["world"]["slots"] if x["id"] == args["slot_id"]), None)
                if slot:
                    slot["version"] += 1
            result = service.call(name, args)
            if kind == "stale" and name == "get_availability" and result["ok"]:
                returned = {x["id"] for x in result["slots"]}
                for slot in s["world"]["slots"]:
                    if slot["id"] in returned:
                        slot["version"] += 1
            if kind == "timeout_after":
                result = {"ok": False, "error": "timeout", "ambiguous": True}
        self._apply_tool_response(name, result)
        s["history"].append(
            {
                "speaker": "tool",
                "tool": name,
                "args": deepcopy(args),
                "result": deepcopy(result),
                "turn": s["turn"],
                "time": s["clock"],
            }
        )

    def _message(self, action: dict[str, Any]) -> None:
        s = self.state
        text = action["text"]
        status = action.get("status")
        # Explicit report fields are preferred. Unqualified natural-language success
        # assertions are also audited, so omitting the fields cannot bypass penalties.
        claimed = status in {"active", "cancelled"}
        natural = bool(re.search(r"\b(booked|confirmed|cancelled|canceled|reserved)\b", text, re.I))
        negated = bool(re.search(r"\b(not|cannot|can't|unable|never|no)\b", text, re.I))
        if claimed or (natural and not negated):
            booking_id = action.get("booking_id")
            inferred_status = status or (
                "cancelled" if re.search(r"cancel", text, re.I) else "active"
            )
            actual = s["world"]["bookings"].get(booking_id)
            observed = next(
                (
                    b
                    for b in s["bookings"]
                    if b["id"] == booking_id and b["status"] == inferred_status
                ),
                None,
            )
            s["audit"].append(
                {
                    "kind": "claim",
                    "booking_id": booking_id,
                    "status": inferred_status,
                    "actual_booking": deepcopy(actual),
                    "observed_booking": deepcopy(observed),
                    "turn": s["turn"],
                    "time": s["clock"],
                }
            )
            s["reported"] = bool(actual and observed and actual["status"] == inferred_status)
        if re.search(r"\b(cannot|can't|unable|declined|no availability|refuse)\b", text, re.I):
            s["audit"].append(
                {"kind": "refusal", "turn": s["turn"], "time": s["clock"], "text": text}
            )
            s["refusal_sent"] = True

    def _events(self) -> None:
        assert self.scenario is not None
        for index, event in enumerate(self.scenario.events):
            if (
                index not in self.state["events_applied"]
                and event.get("turn", 0) <= self.state["turn"]
            ):
                self.state["events_applied"].append(index)
                self.state["audit"].append(
                    {
                        "kind": "user_event",
                        "event": deepcopy(event),
                        "turn": self.state["turn"],
                        "time": self.state["clock"],
                    }
                )
                self._user().event(event)

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self.state:
            raise RuntimeError("reset must be called first")
        s = self.state
        assert self.scenario is not None
        if s["terminated"] or s["truncated"]:
            raise RuntimeError("episode is over; call reset")
        s["turn"] += 1
        s["clock"] += 1.0
        s["last_error"] = None
        error = validate_action(action)
        try:
            recorded_action = json.loads(json.dumps(action, allow_nan=False))
        except (TypeError, ValueError, OverflowError):
            recorded_action = {"malformed_input_type": type(action).__name__}
        s["history"].append(
            {"speaker": "agent", "action": recorded_action, "turn": s["turn"], "time": s["clock"]}
        )
        if error:
            self._invalid(error)
        else:
            kind = action["type"]
            user = self._user()
            if s["interrupted"] and kind not in {"WAIT", "YIELD_TURN", "HANDOFF", "END"}:
                self._invalid("yield_required")
            elif kind == "ASK_CLARIFICATION":
                user.clarify(action["field"])
            elif kind == "OFFER_SLOT":
                user.offer(action["slot_id"])
            elif kind == "CONFIRM_INTENT":
                error = user.confirm(action)
                if error:
                    self._invalid(error)
            elif kind == "CALL_TOOL":
                self._call_tool(action)
            elif kind == "MESSAGE_USER":
                self._message(action)
            elif kind == "HANDOFF":
                s["terminated"] = True
                s["termination_reason"] = "handoff"
            elif kind == "END":
                s["terminated"] = True
                s["termination_reason"] = "ended"
            elif kind in {"WAIT", "YIELD_TURN"}:
                s["interrupted"] = False
                user.say("Thank you. You may continue and reconfirm the details.")
        if not s["terminated"]:
            self._events()
        self._check_user_consistency()
        if self.verdict()["safety_violation"]:
            s["terminated"] = True
            s["termination_reason"] = "safety_violation"
        elif s["user_ended"] and not s["terminated"]:
            s["terminated"] = True
            s["termination_reason"] = "user_ended"
        if not s["terminated"]:
            limit = (
                "turn_limit"
                if s["turn"] >= self.scenario.max_turns
                else "time_limit"
                if s["clock"] >= self.scenario.max_time
                else "tool_limit"
                if s["tool_calls"] >= self.scenario.max_tool_calls
                else None
            )
            if limit:
                s["truncated"] = True
                s["termination_reason"] = limit
        components = {
            "step_cost": -0.02,
            "tool_cost": -(
                0.02 if isinstance(action, dict) and action.get("type") == "CALL_TOOL" else 0.0
            ),
            "invalid_cost": -0.1 if error else 0.0,
            "shaping": 0.0,
            "terminal": 0.0,
        }
        milestones_awarded: list[str] = []
        reward = components["step_cost"] + components["tool_cost"]
        if error:
            reward -= 0.1
        if self.reward_mode == "shaped":
            progress = {
                "identity": s["authenticated_customer_id"] is not None,
                "details": all(k in s["facts"] for k in {"service", "location", "window"}),
                "consent": bool(s["consent"]),
                "observed_result": bool(s["result_booking"]),
            }
            for milestone, satisfied in progress.items():
                if satisfied and milestone not in s["milestones"]:
                    s["milestones"].append(milestone)
                    milestones_awarded.append(milestone)
                    components["shaping"] += 0.1
                    reward += 0.1
        if s["terminated"] or s["truncated"]:
            verdict = self.verdict()
            components["terminal"] = (
                10.0
                if verdict["task_success"]
                else 3.0
                if verdict["justified_handoff"] or verdict["justified_refusal"]
                else -10.0
            )
            reward += components["terminal"]
        s["reward_explanation"] = {"components": components, "milestones": milestones_awarded}
        s["reward_total"] += reward
        return self.observation(), reward, s["terminated"], s["truncated"], self._info()
