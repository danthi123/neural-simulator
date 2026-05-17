"""Pure held-out per-token NLL for a NgramTeacher -- the EXACT formula
from the grounded probe ba1jyepwf (held-out ppl ~14-15). Combine with
subword_lm_gate_core.perplexity for ppl. Pure numpy/stdlib;
CPU-unit-testable."""
from __future__ import annotations
import math
import numpy as np


def ngram_heldout_nll(teacher, ids):
    """Per-token NLL over `ids`: for i in range(2, len(ids)),
    nll_i = -log(max(teacher.soft_dist((ids[i-2], ids[i-1]))[ids[i]],
                     1e-12)). ids shorter than 3 -> []. The 1e-12 floor
    clamps zero-prob (never +inf), matching the grounded probe."""
    ids = list(ids)
    n = len(ids)
    if n < 3:
        return []
    out = []
    for i in range(2, n):
        q = np.asarray(teacher.soft_dist((ids[i - 2], ids[i - 1])),
                       dtype=np.float64)
        p = float(q[ids[i]]) if 0 <= ids[i] < q.shape[0] else 0.0
        out.append(-math.log(max(p, 1e-12)))
    return out
