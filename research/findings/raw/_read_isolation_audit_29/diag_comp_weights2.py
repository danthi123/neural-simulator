import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")
import numpy as np

from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, build_battery

seed = 42
org = ComprehensionProductionOrgan(seed=seed)
org.ensure_built()
comp = org.comp
b = comp.bridge

w0 = np.asarray(b.cp_connections.data).copy()
items = build_battery(seed, n_per_cond=6)
(_lab, _tag, n0, v, n1) = items[0]
r1 = org.read_margin(n0, v, n1)
w1 = np.asarray(b.cp_connections.data).copy()

diff_idx = np.where(np.abs(w0 - w1) > 1e-6)[0]
print("n changed synapses:", len(diff_idx), "of", len(w0))
print("sample changed idx:", diff_idx[:10].tolist())
print("sample deltas:", (w1[diff_idx[:10]] - w0[diff_idx[:10]]).tolist())

# map synapse idx to pre/post region via connections row/col
conn = b.cp_connections
rows, cols = conn.nonzero() if hasattr(conn, "nonzero") else (None, None)
rm = b.region_manager
rid = rm.region_indices_dict()
def region_of(neuron_idx):
    for name, idxs in rid.items():
        if neuron_idx in set(np.asarray(idxs).tolist()):
            return name
    return "?"

if rows is not None:
    for i in diff_idx[:5]:
        pre, post = int(rows[i]), int(cols[i])
        print(f"synapse {i}: pre={pre}({region_of(pre)}) post={post}({region_of(post)}) w0={w0[i]:.4f} w1={w1[i]:.4f}")
