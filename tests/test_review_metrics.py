"""Unsafe backend effects must remain visible in the evaluation aggregates."""

import pytest

from callflow_rl.evaluation.metrics import evaluate_policy
from callflow_rl.scenarios import generate


@pytest.mark.parametrize(
    "violation",
    [
        "consent_booking_mismatch",
        "unauthorized_booking_target",
        "consent_service_mismatch",
        "write_customer_mismatch",
        "policy_handoff_required",
        "policy_service_forbidden",
        "policy_location_forbidden",
        "write_identity_changed",
        "invalid_write_transition",
        "invalid_booking_time",
        "slot_definition_changed",
        "readonly_customer_change",
        "readonly_policy_change",
    ],
)
def test_evaluator_counts_independent_verifier_side_effects(monkeypatch, violation):
    class BackendRegressionEnv:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, **kwargs):
            return {}, {}

        def step(self, action):
            return {}, -10.0, True, False, {}

        def verdict(self):
            return {
                "task_success": False,
                "policy_compliant_success": False,
                "policy_compliance": False,
                "safety_violation": True,
                "violations": [violation],
                "termination_reason": "safety_violation",
                "invalid_action_count": 0,
                "tool_call_count": 1,
            }

        def snapshot(self):
            return {"state": {"audit": [], "tool_calls": 1}}

        def state_hash(self):
            return "backend-regression-state"

    class EndPolicy:
        def act(self, observation):
            return {"type": "END"}

    monkeypatch.setattr("callflow_rl.envs.native.CallFlowEnv", BackendRegressionEnv)
    result = evaluate_policy(EndPolicy(), [generate("book_direct", 0)], split="train")
    assert result["records"][0]["unsafe_side_effect"]
    assert result["metrics"]["unsafe_side_effect"]["count"] == 1
    assert result["metrics"]["unsafe_side_effect"]["denominator"] == 1
