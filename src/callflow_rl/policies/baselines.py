"""Deterministic and random policies for the native structured-event interface."""

from __future__ import annotations

import random
from typing import Any

from callflow_rl.envs.gym_env import action_from_observation
from callflow_rl.policies.decisions import decision_mask


class RandomPolicy:
    """Random bounded decisions, with deterministic local RNG and no evaluator access.

    Uses the same public workflow mask as PPO. Business preconditions may still
    fail, which is measured rather than supplied by an oracle validity signal.
    """

    version = "random-v1"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        valid = [i for i, permitted in enumerate(decision_mask(observation)) if permitted]
        return action_from_observation(self.rng.choice(valid), observation)


class HeuristicPolicy:
    """Public-evidence scheduler, including inspection after ambiguous writes.

    This policy deliberately implements a strong baseline, not an oracle: it
    learns preferences through user responses and relies on returned tools only.
    """

    version = "heuristic-v1"

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        if observation.get("interrupted"):
            return action_from_observation(18, observation)
        if self._refusal_needed(observation):
            return {
                "type": "MESSAGE_USER",
                "text": "I cannot complete this request under your preferences and the published policy.",
            }
        return action_from_observation(self.select_index(observation), observation)

    @staticmethod
    def _refusal_needed(observation: dict[str, Any]) -> bool:
        if observation.get("refusal_sent"):
            return False
        facts = observation.get("facts", {})
        policy = observation.get("policy", {})
        restricted = bool(
            facts.get("service")
            and facts["service"] not in policy.get("permitted_services", [facts["service"]])
        )
        rejected = observation.get("rejected_slots", [])
        known = [
            s["id"] for s in observation.get("slots", []) if s["id"] in facts.get("window", [])
        ]
        declined = bool(
            observation.get("user_declined") and known and all(s in rejected for s in known)
        )
        return restricted or declined

    def select_index(self, observation: dict[str, Any]) -> int:
        if observation.get("interrupted"):
            return 18
        if observation.get("reported") or observation.get("refusal_sent"):
            return 13
        if observation.get("result_booking"):
            return 12
        facts = observation.get("facts", {})
        if not facts.get("task"):
            return 15
        if not facts.get("account_code"):
            return 0
        if not observation.get("customer"):
            return 4
        task = facts.get("task", "book")
        if task in {"reschedule", "cancel"} and not facts.get("booking_id"):
            return 16
        if task != "cancel":
            for field, index in (("service", 1), ("location", 2), ("window", 3)):
                if not facts.get(field):
                    return index
        last = observation.get("last_tool") or {}
        if last.get("error") in {"timeout", "ambiguous_timeout"}:
            return 11
        if observation.get("policy", {}).get("handoff_required"):
            return 14
        if last.get("error") in {"slot_version_conflict", "slot_unavailable", "arguments_rejected"}:
            return 5
        bookings = observation.get("bookings", [])
        if task == "cancel":
            if not bookings:
                return 11
            if all(item.get("status") == "cancelled" for item in bookings):
                return 12
            if not observation.get("consent"):
                return 9
            return 10
        if task == "reschedule":
            if not bookings:
                return 11
            if observation.get("consent") and any(
                item.get("slot_id") == observation["consent"].get("slot_id") for item in bookings
            ):
                return 12
        compatible = [
            (i, s)
            for i, s in enumerate(observation.get("slots", [])[:3])
            if s["id"] in facts.get("window", [])
            and s["id"] not in observation.get("rejected_slots", [])
        ]
        if not compatible:
            if observation.get("availability_checked"):
                return 14
            return 5
        if not observation.get("offered_slot") or not observation.get("offer_accepted"):
            return 6 + compatible[0][0]
        if not observation.get("consent"):
            return 9
        return 10
