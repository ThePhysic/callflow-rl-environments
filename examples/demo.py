"""Short offline demo: clarification, lost commit response, independent audit."""

from callflow_rl.policies import HeuristicPolicy
from callflow_rl.rollouts import replay, run_episode
from callflow_rl.rollouts.trace import view
from callflow_rl.scenarios import generate
from callflow_rl.verifiers.audit import run_audit


def main():
    for name in ("book_all_missing", "book_timeout_after"):
        trace = run_episode(HeuristicPolicy(), generate(name, 42), seed=42)
        print(view(trace))
        print(f"Exact replay matched: {replay(trace)['matched']}")
    audit = run_audit()
    print(f"Reward shortcuts rejected: {audit['case_count']} / {audit['case_count']}")
    print(f"Audit passed: {audit['all_passed']}")


if __name__ == "__main__":
    main()
