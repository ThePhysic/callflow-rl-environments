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

## Round 2 — changes required

Source reviewed: `10c33460f9d708269a84ebcc4bc36dc54078f0e0`.
The reviewer independently passed all 82 review-specific regressions and confirmed
the original three findings were addressed. Two additional P1 gaps remained:

- An observed report of the original active reservation could satisfy completion
  after rescheduling because completion matched only reservation ID and status.
  Completion must require a report of the final row after its final write;
  earlier truthful reports remain harmless and insufficient.
- Capacity was checked only in final rows. An intermediate move into an occupied
  slot could be hidden by a later correct move. Capacity must hold after every
  reconstructed write, with positive shared-capacity and intermediate-move cases.

The interrupted checkpoint refreshes for intermediate candidates were never
published. Implementation fixes now reconstruct capacity after every write and
require full-row reporting after the final mutation. Twelve additional regressions
cover both native report ordering and transient capacity, with positive retry,
reconciliation and shared-capacity cases. Public report flags depend only on
observed operation results; private database truth remains in the verifier.

Final review outcome will be recorded here after independent rereview.

## Round 3 — code clean; final evidence review pending

Source reviewed: `c80e5984653e7d7495982751aff2bdd481e10689`.
The reviewer independently passed all 94 review-specific regressions and found
no remaining code-level complaints. Local release verification passed 274 tests,
type/lint checks, packaging, replay, real CLI training/evaluation and a 200-episode
batch using four observed worker processes. An additional artifact-integrity
regression verifies the published evaluation and checkpoint hashes.

The corrected 28-report evaluation was staged, inspected and explicitly accepted.
It preserves the historical weights and data, reports 1,092 changed fields across
318 episodes, and marks outcome/return parity false. The random test score is
5/200; PPO task scores remain 25%, 0%, 0%. Independent review of these final
artifacts and disclosures is pending before publication.
