import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")
import numpy as np

from research.runners._laneC_source_provenance_opponent_derisk import (
    ProvenanceBrain, make_paired_patterns, _encode_all,
)

seed = 42
brain = ProvenanceBrain(seed)
patterns = make_paired_patterns(seed)
_encode_all(brain, patterns, learning=True)

pat_a = patterns["perceived"][0]
w0 = np.asarray(brain._bridge.cp_connections.data).copy()
r1 = brain.recall(pat_a)
w1 = np.asarray(brain._bridge.cp_connections.data).copy()
r2 = brain.recall(pat_a)
w2 = np.asarray(brain._bridge.cp_connections.data).copy()
print("w0==w1:", np.array_equal(w0, w1), "max diff:", float(np.abs(w0-w1).max()))
print("w1==w2:", np.array_equal(w1, w2), "max diff:", float(np.abs(w1-w2).max()))
print("enable_ou_process:", brain._bridge.core_config.enable_ou_process)
print("enable_hebbian_learning:", brain._bridge.core_config.enable_hebbian_learning)
print(r1, r2)
