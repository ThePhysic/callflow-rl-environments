# Contributing

Use Python 3.13 and uv 0.13.5 (Python 3.12+ is supported by the package contract).
Install dependencies with package access; the deterministic checks and demos
then run locally without paid API keys or a GPU.

```console
uv sync --frozen --python 3.13
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest -q
uv run --no-sync callflow validate --suite core
uv run --no-sync callflow audit-rewards --suite adversarial
```

Keep fixtures synthetic. Do not add account credentials, raw customer calls,
real calendars, personal records or paid-service dependencies to the demo.
Keep generated local outputs under `artifacts/local/` rather than overwriting
tracked research evidence.

For a change, describe the behavior or documentation problem, keep the patch
focused, and report the checks run and any checks that could not run. For a bug
fix, prefer a failing regression that demonstrates the problem before changing
the implementation. Check relative links and runnable examples for documentation
edits; do not regenerate experiments just to refresh documentation.

[DEVELOPMENT.md](DEVELOPMENT.md) describes API boundaries, extension points,
behavioral regression/replay gates, frozen-suite maintenance and evidence
preservation. Follow those contracts before changing environment semantics,
reward rules, scenario manifests or published evaluations. The
[experiment protocol](docs/experiments.md) distinguishes training new weights
from reevaluating saved ones and explains train/development-only selection.
