"""Seeded truth-conditioned customer; no generated text acts as a success oracle."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any


class UserSimulator:
    def __init__(self, state: dict[str, Any], rng: random.Random) -> None:
        self.state = state
        self.rng = rng

    def say(self, text: str) -> None:
        self.state["user_message"] = text
        self.state["history"].append(
            {
                "speaker": "user",
                "time": self.state["clock"],
                "turn": self.state["turn"],
                "text": text,
            }
        )

    def clarify(self, field: str) -> None:
        self.state["clarifications"] += 1
        if self.state["clarifications"] > self.state["user_config"].get("tolerance", 100):
            self.state["user_ended"] = True
            self.say("I cannot keep answering. Please end this conversation.")
            return
        truth = self.state["user_truth"]
        if field not in truth:
            self.say(f"I do not have a {field} to provide.")
            return
        self.state["facts"][field] = deepcopy(truth[field])
        prefix = self.rng.choice(["For clarity", "To be precise", "Here is that detail"])
        self.say(f"{prefix}, {field} is {truth[field]}.")

    def offer(self, slot_id: str) -> None:
        truth = self.state["user_truth"]
        if truth.get("task", "book") == "cancel":
            # A cancellation request has no new appointment to accept. Treat an
            # irrelevant offer as a clarification of intent, not as a refusal
            # of the cancellation itself, and require renewed authorization.
            self.state["offered_slot"] = None
            self.state["offer_accepted"] = False
            self.state["consent"] = None
            self.say("My request is to cancel my existing reservation. I do not want a new slot.")
            return
        slot = next((s for s in self.state["slots"] if s["id"] == slot_id), None)
        if slot is None:
            self.say("Please check availability before offering that slot.")
            return
        config = self.state["user_config"]
        acceptable = list(truth.get("window", []))
        if config.get("accept_alternatives", False):
            acceptable += config.get("alternative_slots", [])
        admissible = (
            slot["service"] == truth.get("service")
            and slot["location"] == truth.get("location")
            and slot_id in acceptable
            and not config.get("decline_all", False)
        )
        self.state["offered_slot"] = slot_id
        self.state["offer_accepted"] = admissible
        self.state["consent"] = None
        if admissible:
            self.say(f"That option, {slot_id}, works. Please confirm the details with me.")
        else:
            self.state["user_declined"] = True
            self.state["declined_slots"].append(slot_id)
            self.say(f"I decline {slot_id}; it does not meet my current preferences.")

    def confirm(self, action: dict[str, Any]) -> str | None:
        truth = self.state["user_truth"]
        task = truth.get("task", "book")
        customer_id = self.state["authenticated_customer_id"]
        if not customer_id or customer_id != self.state["session_customer_id"]:
            return "identity_required"
        booking_id = action.get("booking_id")
        slot_id = action.get("slot_id")
        if task == "cancel":
            if slot_id is not None or booking_id != truth.get("booking_id"):
                return "intent_mismatch"
            observed = next(
                (
                    b
                    for b in self.state["bookings"]
                    if b["id"] == booking_id and b["status"] == "active"
                ),
                None,
            )
            if observed is None:
                return "booking_evidence_required"
        else:
            if slot_id != self.state["offered_slot"] or not self.state["offer_accepted"]:
                return "accepted_offer_required"
            if task == "reschedule" and booking_id != truth.get("booking_id"):
                return "intent_mismatch"
            if task == "book" and booking_id is not None:
                return "intent_mismatch"
        if self.state["user_config"].get("decline_all", False):
            return "user_declined"
        consent = {
            "task": task,
            "customer_id": customer_id,
            "service": truth.get("service"),
            "slot_id": slot_id,
            "booking_id": booking_id,
            "revision": self.state["user_revision"],
            "turn": self.state["turn"],
        }
        self.state["consent"] = consent
        self.state["audit"].append(
            {
                "kind": "user_confirmation",
                "consent": deepcopy(consent),
                "turn": self.state["turn"],
                "time": self.state["clock"],
            }
        )
        self.say("Yes, I explicitly authorize those appointment details.")
        return None

    def event(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "correction":
            field = event["field"]
            self.state["user_truth"][field] = deepcopy(event["value"])
            self.state["facts"][field] = deepcopy(event["value"])
            self.state["user_revision"] += 1
            self.state["consent"] = None
            self.state["offered_slot"] = None
            self.state["offer_accepted"] = False
            self.state["slots"] = []
            self.state["availability_checked"] = False
            self.state["declined_slots"] = []
            self.state["user_declined"] = False
            self.state["pending_write"] = None
            if field == "account_code":
                self.state["authenticated_customer_id"] = None
                self.state["customer"] = None
            self.say(f"Correction: my {field} is now {event['value']}.")
        elif kind == "interrupt":
            self.state["consent"] = None
            self.state["interrupted"] = True
            self.say(
                "Please pause; I have not finished speaking. Yield the turn, then confirm again."
            )
        elif kind == "hangup":
            self.state["user_ended"] = True
            self.say("I am ending the conversation now.")
        elif kind == "repeat":
            self.say(f"To repeat my request: {self.state['user_truth']}.")
