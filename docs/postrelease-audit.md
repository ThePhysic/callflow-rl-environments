# Follow-up audit of the first release

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

Regression coverage is in `tests/test_postrelease_native.py`,
`tests/test_postrelease_invariants.py` and `tests/test_postrelease_rollouts.py`.
The audit also independently recomputed all 28 existing evaluation summaries and
validated their original source/report/checkpoint fingerprints without finding a
measurement or PPO-math defect. Historical training, checkpoints and prior
evaluations are retained; updated source needs refreshed replay/evaluation data.

Fresh Astra rereview and final evidence validation are pending. This document
does not claim a general natural-language truth judge, production user realism,
or that any finite test suite proves absence of all possible bugs.
