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
Implementation and regression-test corrections are committed at `62a1aaf` in
release `0.1.1` with verifier `1.1.0`; independent rereview is still required.
Historical training and evaluation evidence remains versioned and preserved;
corrected simulator observations may change checkpoint trajectories.

The corrected verifier checks affected reservation IDs, ownership, consent
service, and valid identity/status/version transitions. It reads business policy
from the initial scenario, with cancellation service/location rules matching the
documented API. Twenty-five new verifier tests include backend-regression
negatives and an authorized intermediate reschedule outside the final slot set.
The user correction rejects new-slot offers during cancellation, clears approval,
and adds task/preference consistency checks. Thirteen new dialogue tests include
postcommit availability refreshes. Nine evaluation tests ensure new violations
are counted as unsafe side effects. The complete suite passes 227 tests.

Follow-up checks expanded the same authorization guarantees: offer-driven consent
revocations are explicit audit events, and confirmation order is verified. They
also demonstrated that changing a confirmed slot's timestamps could evade the
original ID-based goal check. Slot definitions, customer records and published
policy are now independently anchored to the scenario, with equivalent timestamp
representations and legitimate availability/version changes covered positively.
The expanded suite passes **262 tests**, with type and lint checks also passing.

Final review outcome will be recorded here after independent rereview.
