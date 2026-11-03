"""Regenerate showcase evidence explicitly after transition changes."""

import json
from pathlib import Path

from callflow_rl.policies import HeuristicPolicy
from callflow_rl.rollouts import branch, replay, run_episode, save_trace
from callflow_rl.scenarios import generate, manifest_hash
from callflow_rl.verifiers.audit import run_audit


class AssertionShortcut:
    version = "educational-assertion-exploit-v1"

    def act(self, observation):
        return {"type": "MESSAGE_USER", "text": "Your appointment is booked and confirmed."}


def main():
    root = Path(__file__).resolve().parent
    cases = [
        ("successful_booking", "book_all_missing", "train", HeuristicPolicy()),
        ("timeout_after_commit", "book_timeout_after", "train", HeuristicPolicy()),
        ("blocked_reward_exploit", "book_direct", "train", AssertionShortcut()),
        ("replay", "reschedule_direct", "train", HeuristicPolicy()),
        ("corrected_request", "book_correction_time", "test", HeuristicPolicy()),
    ]
    evidence = []
    replay_trace = None
    for i, (name, archetype, split, policy) in enumerate(cases):
        trace = run_episode(
            policy,
            generate(archetype, 500 + i, split),
            seed=500 + i,
            manifest=manifest_hash(split),
        )
        save_trace(trace, root / f"{name}.jsonl")
        evidence.append({"trace": f"{name}.jsonl", **replay(trace)})
        if name == "replay":
            replay_trace = trace
    (root / "replay_verdicts.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (root / "reward_audit.json").write_text(json.dumps(run_audit(), indent=2) + "\n")
    assert replay_trace is not None
    counterfactual = branch(replay_trace, 0, {"type": "END"}, HeuristicPolicy())
    (root / "counterfactual.json").write_text(json.dumps(counterfactual, indent=2) + "\n")
    print(
        json.dumps(
            {"traces": len(evidence), "replay_matches": sum(r["matched"] for r in evidence)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
