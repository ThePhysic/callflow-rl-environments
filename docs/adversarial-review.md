# Internal adversarial review history

The shipped implementation is environment **0.1.2** / verifier **1.2.0**, with
headless Python/Gymnasium APIs, CLI demos and saved replay/evaluation evidence.
The latest implementation review is summarized in [the follow-up audit](postrelease-audit.md);
the rounds below preserve the earlier **0.1.1** release chronology. GitHub
availability is controlled by the repository's current access settings. Earlier
private-publication gates below applied to those historical release candidates.

These were AI-assisted **internal** reviews using **gpt-6-astra**, reasoning
effort **medium**, separately from the implementation agents. "Independent"
describes that separation within the project workflow, not third-party external
certification or a human domain-expert audit. The strongest evidence is the
reproduced counterexamples, regression tests and linked measurements. Review
covered the project brief, implementation, tests, experiments and public claims.

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
release `0.1.1` with verifier `1.1.0`; rereview was required at that point.
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
are counted as unsafe side effects. The complete suite then passed 227 tests.

Follow-up checks expanded the same authorization guarantees: offer-driven consent
revocations are explicit audit events, and confirmation order is verified. They
also demonstrated that changing a confirmed slot's timestamps could evade the
original ID-based goal check. Slot definitions, customer records and published
policy are now independently anchored to the scenario, with equivalent timestamp
representations and legitimate availability/version changes covered positively.
The expanded suite then passed **262 tests**, with type and lint checks also passing.

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
published. The fixes reconstructed capacity after every write and required
full-row reporting after the final mutation. Twelve additional regressions
cover both native report ordering and transient capacity, with positive retry,
reconciliation and shared-capacity cases. Public report flags depend only on
observed operation results; private database truth remains in the verifier.

The subsequent rereview outcome is recorded in Round 3 below.

## Round 3 — overall clean: no remaining complaints

Source reviewed: `c80e5984653e7d7495982751aff2bdd481e10689`.
The reviewer independently passed all 94 review-specific regressions and found
no remaining code-level complaints. Local release verification passed 274 tests,
type/lint checks, packaging, replay, real CLI training/evaluation and a 200-episode
batch using four observed worker processes. An additional artifact-integrity
regression verifies the published evaluation and checkpoint hashes.

The corrected 28-report evaluation was staged, inspected and explicitly accepted.
It preserves the historical weights and data, reports 1,092 changed fields across
318 episodes, and marks outcome/return parity false. The random test score is
5/200; PPO task scores remain 25%, 0%, 0%.

The reviewer issued the explicit final verdict **“Overall review clean: no
remaining complaints”** for candidate
`0bab1aecc4b426e25d71a65e1618d9c62a673324`, source `c80e598`. All five
substantive findings from the first two rounds are resolved. Independent final
verification passed 94 review regressions and three artifact-integrity tests;
reproduced every saved state hash, return, outcome and resource count for the
200-episode random test suite; checked all 28 summaries against their reports;
and verified source/checkpoint provenance and preserved historical evidence.
Quality, packaging, replay and parallel-rollout evidence were reviewed. The
reviewer approved the candidate for private GitHub publication under the access
settings requested for that release. This is a historical approval, not a claim
about current repository visibility.

The original clean verdict was followed only by the review-record commit
`f3c4b8e`. The subsequent follow-up audit and implementation changes are recorded
below.

## Follow-up requested audit

The preceding verdict and commit identifiers describe release `0.1.1`. The user's
subsequent request for another audit found additional custom-scenario and worker
failure cases. Corrections and new regression evidence for release `0.1.2` are
recorded in [the follow-up audit](postrelease-audit.md); they do not retroactively
claim those cases were covered by the original passing tests.

Fresh Astra medium approved candidate `0e691c4`, source `4fd3e16`, with no remaining
material complaints established, then approved the review-record commit
`298546759cf4fe5ce26e0af68904ee3b81a5ba8b`. The follow-up record contains its
372-test and refreshed-evidence scope. The documentation polish that followed
does not change implementation, training weights or evaluation data, and is not
an additional implementation-certification claim.
