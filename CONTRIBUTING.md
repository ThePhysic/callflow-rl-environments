# Development

Use Python 3.13 and uv 0.13.5 (Python 3.12+ is supported by the package contract).

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
Changes to tool/state transitions require contract, verifier and replay checks.
Freeze scenario manifests explicitly; do not regenerate a test suite silently.
Training, hyperparameter selection and checkpoint selection must use train/dev
only. Record test evaluations as measurements, without tuning against failures.
Changing a verifier or transition rule invalidates affected experiment evidence:
rerun it and document the new source/manifests rather than retaining stale claims.

The intentionally vulnerable reward comparator is educational and separate
from the default environment. New shaping rewards must be bounded and resistant
to repeated queries/clarifications. Prefer a failing adversarial regression case
before changing correctness rules.
