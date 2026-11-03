"""Small, explicit uncertainty estimators without normal-approximation edge cases."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Two-sided 95% Wilson interval; null denominators are not silently successes."""
    if total <= 0:
        return [0.0, 1.0]
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def bootstrap_seed_means(values: Sequence[float], seed: int = 9173) -> list[float]:
    """Bootstrap independent training seeds, not pooled episodes.

    With three seeds the interval is descriptive and coarse. Reports retain the
    individual seed values so uncertainty is not overstated.
    """
    if not values:
        return [0.0, 1.0]
    rng = np.random.default_rng(seed)
    array: np.ndarray = np.asarray(values, dtype=np.float64)
    means = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]
