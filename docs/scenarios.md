# Synthetic scenarios and evaluation integrity

Every customer, account code, business location, slot and reservation is authored
synthetic data. No recordings, production calendars, credentials, personal calls or
licensed external demonstrations are part of this repository.

## Versioned contract

`ScenarioSpec` version `1` is evaluator input, never policy input. It contains the
scenario ID, archetype, split, seed, public task, private goal, initial world, user
truth model, fault/event schedules and turn/time/tool limits. `to_dict` and
`from_dict` preserve a JSON representation. Public observations contain only facts
the user has supplied, dialogue/tool history, published policy and tool schemas,
visible candidates/reservations and remaining budgets.

The private goal names the target customer, operation, authorized service, allowed
locations and final acceptable slot IDs. `outcome` is `completed`, `handoff` or
`refused`. A goal is never edited by an agent action. Scheduled user corrections
are authored together with the final private goal: initial preferences can be
obsolete, but the verifier still checks the final accepted constraints.

The world contains two synthetic customer accounts, four versioned slots, and zero
or more original reservations. Account insertion order varies; selecting the first
record is not a customer-resolution strategy. All timestamps contain an explicit
UTC offset. The DST case distinguishes `2026-11-01T01:30:00-07:00` from
`2026-11-01T01:30:00-08:00` with distinct slot IDs; this release does not infer an
IANA time zone from a place name or claim natural-language date parsing.

## Authored archetypes and frozen partitions

The catalog contains **34 authored archetypes**. Review the explicit branches in
`generator.py` and obligations in `fixtures/golden_cases.json`; counts alone are not
evidence of realistic human conversation. These are controlled state-machine
variants, not sampled real calls or an LLM user population.

| Split | Archetypes | Fixed instances | Purpose |
| --- | ---: | ---: | --- |
| train | 20 | 160 | Training configurations and in-distribution evaluation |
| dev | 4 | 80 | Development/checkpoints; no gradient updates |
| test | 8 | 200 | Frozen held-out behavior evaluation |
| transfer | 2 | 40 | Genuinely new calendar patterns, no gradient updates |

Split ownership is assigned to the archetype and behavior class before seeds are
chosen. No archetype or named behavior class crosses partitions. Shared operation
types (book/reschedule/cancel) are intentional: this measures behavioral transfer
within one domain, not transfer to an unrelated business domain.

Train covers required-fact clarification, flexible alternatives, no capacity,
declines, early interruptions, latency, conflicts, precommit and postcommit
timeouts, atomic rescheduling, cancellation and explicit handoff policy. Dev
holds out relative-date phrasing, argument rejection, a full reschedule calendar,
and ambiguous cancellation. Test holds out window/location corrections, stale
availability reads, duplicate requests after a lost response, late cancellation
interruption, UTC-offset disambiguation and a prohibited service. Correction events
and stale reads are absent from training; shared timeout recovery remains in
training. Dev/test public task paraphrase patterns are also absent from training.

Transfer adds (1) a calendar with interleaved service/location distractors and a
later acceptable candidate, and (2) a capacity-two slot already occupied by one
other customer. Training calendars have one service/location and capacity one.
These tests do not establish open-domain linguistic or real-voice generalization.

The catalog exposes challenge slices `clean`, `fault`, `timeout_after`,
`ambiguous`, `policy`, `interruption`, `timezone` and `transfer`. They are evaluator
metadata only. An instance may appear in multiple slices, so slice denominators
must not be summed as if they were disjoint.

## Frozen suite integrity and reproducibility

`manifests/*.jsonl` stores full initial worlds and schedules. `index.json` records
suite version `1.0.0`, counts, owned behaviors and SHA-256 digests. `load_suite`
checks each digest and returns fresh objects. A seed change alone does not move a
training template into a held-out split. `generate` rejects cross-split requests.

```console
python -m callflow_rl.scenarios.freeze --check
```

The explicit maintenance command `--write` regenerates manifests. It changes the
evaluation definitions, so any such change needs a suite-version/review decision
and fresh experiment evidence. Evaluation never regenerates the test set on load.

Generation uses a local `random.Random(seed)`. Episode simulation derives isolated
RNG streams during reset; loading or running another episode cannot advance this
generator's state. Latency is virtual rather than a wall-clock sleep.

## Future demonstration annotation

Real interaction ingestion is outside the shipped simulator. An authored dialogue
can be annotated with the following manual record before any future conversion:

| Field | Required content |
| --- | --- |
| provenance | Synthetic author or lawful rights/consent, redaction status |
| intent | Book/reschedule/cancel/handoff, target account reference |
| known_facts | Turn-indexed user facts, uncertainties and corrections |
| decision_points | Observable state and permissible actions at each decision |
| alternatives | Accepted/rejected slots and explicit confirmation evidence |
| effects | Tool arguments, before/after rows, errors and idempotency key |
| success_predicate | Independent final-state constraints and forbidden effects |
| counterfactuals | Valid alternative actions and expected continuations |

Do not include raw real calls or personal data merely because this schema exists.
