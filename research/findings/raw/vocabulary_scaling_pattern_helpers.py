"""Pure helpers for pattern-grounded compositional symbols.

The K-of-N sparse pattern that defines a concept's identity on the
trained substrate is a list of K pool-neuron indices; the
pattern-grounded symbol-derivation step needs the corresponding binary
indicator vector over the whole N-neuron pool to feed into the same
fixed deriver the activity-grounded path uses. This is a trivial
function but it is isolated in its own module so the dedicated
adversarial reviewer can confirm by name that the symbol's input is
the pattern indicator and nothing else.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def pattern_vector(pattern: Iterable[int], n_pool: int) -> np.ndarray:
    """Return the binary 0/1 indicator vector for a K-of-N sparse
    pattern: 1.0 at every neuron index in ``pattern``, 0.0 elsewhere.

    Raises ValueError if any index in ``pattern`` is outside the
    half-open range ``[0, n_pool)`` -- a guard against silent
    out-of-bounds writes (numpy would otherwise raise IndexError, but
    a clear ValueError surfaces the substrate / runner mismatch the
    moment it appears).
    """
    n = int(n_pool)
    v = np.zeros(n, dtype=np.float64)
    for idx in pattern:
        i = int(idx)
        if i < 0 or i >= n:
            raise ValueError(
                f"pattern index {i} out of range [0, {n})")
        v[i] = 1.0
    return v
