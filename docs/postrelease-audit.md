# Follow-up audit of the first release

Environment **0.1.2** / verifier **1.2.0** contains the corrections below. It ships
as headless Python research infrastructure: native/Gymnasium APIs, a CLI, saved
synthetic traces and measured experiments. See [the quick demo](../README.md#quick-demo)
and [current experiment provenance](../artifacts/experiments/evaluation_release.json).
The audit concluded with approval of `2985467`; the historical private-publication
approval at the end of this record does not describe current access settings.

This was AI-assisted **internal** review, with a fresh **gpt-6-astra** reviewer at
**medium** reasoning effort separate from the implementation agents. It was not
third-party external certification or human domain-expert review. The executable
failure cases, regression coverage and preserved measurements define its scope.

## Audit starting point and reproduced failures

The user requested a fresh check after release `0.1.1`. The audit started from
`f3c4b8e`, with 275 existing tests passing. Two implementation auditors and a fresh
independent Astra medium reviewer examined code, custom scenarios and saved
evidence. Passing tests did not cover several executable counterexamples.

Corrections in `0.1.2` / verifier `1.2.0`:

| Area | Confirmed failure | Correction and evidence |
| --- | --- | --- |
| Reservation allocation | A second create could reuse a sparse existing `B000002` ID and overwrite another customer. | Find an unused ID at every allocation; preserve all existing rows. |
| Duplicate appointments | A capacity-two reschedule could produce two appointments for the same customer and slot; verification accepted it. | Guard creates and moves, excluding the moved row; independently detect introduced duplicates in every reconstructed state. Different-customer shared capacity stays valid. |
| Full-history verification | An invalid zero-length or offset-free intermediate destination could disappear after a final valid move. Boolean/float write versions compared equal to integers. | Validate every write's destination interval and exact integer versions. |
| Status reporting | Typed status could mask opposite text, and the first affirmative clause could hide a later contradictory clause. | Audit every supported affirmative status class; retain bounded clause-level negation and valid cancellation confirmations. |
| Episode reset | Default `reset(seed=2)` depended on which seed first initialized the environment. | Regenerate automatic scenarios from the requested seed, preserve explicit scenarios, and snapshot reset mode. |
| Time budgets | END at virtual time 7 could earn success despite a 6.5 deadline. | Apply the hard deadline to END/HANDOFF, preserve safety priority and in-flight write history, and independently check completion budgets. Exactly-on-deadline completion remains valid. |
| Configuration | Nonfinite/negative fault durations could poison clock/hash state; two individually finite extreme durations could overflow. | Validate finite JSON and typed positive limits. Bound horizon and individual fault duration to `2**40` virtual seconds, keeping an in-flight overshoot finite and unit ticks representable. |
| Corrections and identity | Account-code typo corrections could remain permanently unauthorized; late corrections retained stale operation/refusal markers. | Support an evaluator-owned `user.principal_customer_id`, never switch principals from a supplied code, reject unresolved identity-correction configurations clearly, and invalidate superseded outcome markers. |
| Worker recovery | A dead process broke its executor permanently, so retrying on that executor failed immediately. | Replace broken pools, retain completed results, and apply bounded per-task retries to unfinished work. |
| Trace integrity | Exact replay ignored corrupt aggregate measurements and supplied snapshots. Tool-call naming mixed attempts with dispatches. | Verify deterministic aggregates and all supplied snapshots. Trace v2 records dispatched `tool_calls` and attempted `tool_decisions`; v1 loading/replay remains supported. Wall time is checked for validity, not exact equality. |

Regression coverage is in [native environment cases](../tests/test_postrelease_native.py),
[state/verifier invariants](../tests/test_postrelease_invariants.py) and
[worker/replay cases](../tests/test_postrelease_rollouts.py).
The audit also independently recomputed all 28 existing evaluation summaries and
validated their original source/report/checkpoint fingerprints without finding a
measurement or PPO-math defect. Historical training, checkpoints and prior
evaluations were retained; the corrected source required refreshed replay and
evaluation data, as recorded below.

## Corrected build and evaluation evidence

The frozen code at `4fd3e16` has a clean fresh Astra medium rereview. Local release
verification passed **372 tests** (97 additional cases), lint/type checks, real
CLI training/evaluation, five exact replays and a 200-episode run using four
observed worker processes. Wheel/sdist data and external-directory CLI checks
also passed. All 28 evaluations were refreshed with the original five checkpoints:
there are no new outcome, resource or return differences against `0.1.1`.
Registered manifests and checkpoint bytes are unchanged, and historical evidence
remains archived.

The [quality record](../artifacts/validation/quality.json),
[package check](../artifacts/validation/package_check.json),
[four-worker benchmark](../artifacts/validation/parallel/benchmark.json) and
[five replay verdicts](../examples/replay_verdicts.json) retain the captured
results. [Evaluation provenance](../artifacts/experiments/evaluation_release.json)
records the refreshed reports and preserved hashes. The original PPO weights and
learning curves came from training build `837af56`; this reevaluation performed
no additional gradient training. Held-out shaped-PPO task success remained
50/200, 0/200 and 0/200 across seeds 11, 29 and 47. These results do not establish
stable learning or robust transfer to new conversational tasks.

## Historical review verdict

Fresh Astra medium issued a clean overall verdict for candidate
`0e691c441f82c527e2b02bf702bf6a1fdd874534`, source `4fd3e16`: **no remaining
material complaints established**. Independent final checks recomputed all 28
summaries, compared all 3,360 current/prior episode records on the recorded keys
with zero differences, verified report/checkpoint/archive/source fingerprints,
and exactly replayed all five regenerated v2 traces. The reviewer inspected the
372-test, packaging and serial/process parity evidence and approved the exact
candidate for the private publication requested at that time. It subsequently
approved exact review-record commit `298546759cf4fe5ce26e0af68904ee3b81a5ba8b`.
Later documentation polish preserves those historical identifiers and findings;
it does not change the evaluated implementation. This records review scope
rather than asserting that tests prove the absence of every possible future bug.
