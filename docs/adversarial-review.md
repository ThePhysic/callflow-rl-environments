# Independent adversarial review

Reviewer: **gpt-6-astra**, reasoning effort **medium**. Review covers the core
project brief, implementation, regression tests, measured experiments and claims.
Private GitHub publication waits on an explicit clean verdict.

## Round 1 — changes required

Candidate: `85c8d6643e904330c3012086a49f9dbe4096cab0`.
The reviewer independently reran all 180 tests and identified three material gaps:

| Severity | Finding | Required correction |
| --- | --- | --- |
| P1 | An audited cancellation/reschedule could modify another same-customer reservation using the target's consent and still pass verification. | Bind affected row identity and ownership to consent and task authorization; add backend-regression negatives. |
| P1 | The verifier could accept audited writes forbidden by the published service/location/handoff policy. | Check authoritative policy independently of backend guards and include violations in unsafe-side-effect metrics. |
| P2 | Cancellation dialogue accepted proposed slots and truth-model consistency checks missed it. | Reject irrelevant offers while preserving cancellation intent; check task-conditioned offer state and refresh evaluations honestly. |

The reviewer found no additional material issue in PPO clipping/GAE, observation
boundaries, replay implementation or the disclosed negative learning conclusions.
These findings are being resolved in release `0.1.1` with verifier `1.1.0`.
Historical training and evaluation evidence remains versioned and preserved;
corrected simulator observations may change checkpoint trajectories.

Final review outcome will be recorded here after independent rereview.
