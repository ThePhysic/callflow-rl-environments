# Ten golden obligations

`scenarios/fixtures/golden_cases.json` contains deterministic seed/archetype inputs
and expected outcomes. The regression suite runs the same observation-only
heuristic adapter used by ordinary rollouts and verifies final database state.
These are small review anchors; randomized stateful/adversarial tests cover more
seeds and action orders.

| Case | Expected outcome | Required evidence |
| --- | --- | --- |
| Direct booking | completed | Exactly one active correct reservation; confirmation, truthful report, END |
| All facts missing | completed | Identity, service, location and window clarified before commit |
| Timeout after commit | completed | State inspection or identical idempotent retry; one write |
| Booking conflict | completed | Refresh candidate and confirm an admissible alternative |
| No availability | handoff | Database unchanged; explicit handoff; no success claim |
| User declines | refused | Database unchanged; decline respected and communicated before END |
| Reschedule | completed | Original reservation ID retained; atomic slot change; no duplicate |
| Cancel | completed | Correct existing reservation cancelled with authorization and confirmation |
| User corrects time | completed | Old preference discarded, stale approval unusable, final slot matches correction |
| Repeated DST hour | completed | Later UTC-offset occurrence selected; earlier wall-time match insufficient |

A passed database predicate alone is insufficient if the agent reports a false
status or leaves an unauthorized side effect. Conversely a justified handoff or
refusal is recorded separately from completed appointment success.
