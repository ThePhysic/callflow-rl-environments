"""Public workflow scaffolding shared by the random and PPO policies.

The native API permits premature END/HANDOFF and invalid tool arguments. This
constrained policy class removes missing-argument calls, permits END only after
an observed report/refusal, and permits HANDOFF only with public policy or
availability evidence. It still chooses clarification, queries, offers, consent,
write/recovery and efficiency decisions. No evaluator labels are used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MASK_VERSION = "public-workflow-v1"


def decision_mask(observation: dict[str, Any]) -> np.ndarray:
    o = observation
    facts = o.get("facts", {})
    task = facts.get("task", "book")
    customer = bool(o.get("customer"))
    slots = o.get("slots", [])
    rejected = o.get("rejected_slots", [])
    window = facts.get("window", [])
    policy = o.get("policy", {})
    allowed: np.ndarray = np.zeros(20, dtype=np.bool_)
    if o.get("interrupted"):
        allowed[18] = True
        return allowed
    # Repetition remains possible and incurs cost; neither known facts nor
    # completed milestones can be farmed for repeated shaping credit.
    allowed[[0, 1, 2, 3, 15, 17, 19]] = True
    allowed[4] = bool(facts.get("account_code"))
    allowed[5] = bool(facts.get("service") and facts.get("location"))
    for index, slot in enumerate(slots[:3]):
        allowed[6 + index] = slot["id"] not in rejected
    if task == "cancel":
        allowed[9] = customer and any(
            b["id"] == facts.get("booking_id") and b["status"] == "active"
            for b in o.get("bookings", [])
        )
    else:
        allowed[9] = customer and bool(o.get("offered_slot") and o.get("offer_accepted"))
    allowed[10] = customer and bool(o.get("consent") or o.get("pending_write"))
    allowed[11] = customer
    allowed[12] = bool(o.get("result_booking"))
    allowed[13] = bool(o.get("reported") or o.get("refusal_sent"))
    permitted = policy.get("permitted_services", [facts.get("service")])
    restricted = bool(facts.get("service") and facts["service"] not in permitted)
    no_requested_slots = bool(
        o.get("availability_checked") and window and not any(s["id"] in window for s in slots)
    )
    allowed[14] = bool(policy.get("handoff_required") or no_requested_slots)
    # Refusal itself is index19; handoff has a different evaluator outcome.
    if restricted:
        allowed[19] = True
    allowed[16] = task in {"cancel", "reschedule"}
    return allowed
