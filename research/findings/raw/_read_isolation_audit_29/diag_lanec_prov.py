import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")
import numpy as np

from research.runners._laneC_source_provenance_opponent_derisk import (
    ProvenanceBrain, make_paired_patterns, _encode_all, PROVENANCES, N_PAIRS,
)

seed = 42
brain = ProvenanceBrain(seed)
patterns = make_paired_patterns(seed)
_encode_all(brain, patterns, learning=True)

pat_a = patterns["perceived"][0]
pat_b = patterns["generated"][1]

r1 = brain.recall(pat_a)
r2 = brain.recall(pat_a)
identical = (r1 == r2)

r3 = brain.recall(pat_a)
_ = brain.recall(pat_b)
r4 = brain.recall(pat_a)
order_identical = (r3 == r4)

print(json.dumps({
    "repeat_read": {"r1": r1, "r2": r2, "identical": identical},
    "order_dependence": {"r3": r3, "r4": r4, "identical": order_identical},
}, indent=2))
