import sys; sys.path.insert(0, "/home/dant123/Projects/sim")
"""Measure the LAYER-2 weight-map pedestal + contrast directly (layer 1 is known: ~1.6x on a ~5x pedestal).
Needed as the BASELINE against which any pedestal-lowering mechanism must be scored."""
import os; os.environ.setdefault("SIM_BACKEND","numpy")
import numpy as np
import research.runners._gap4_btsp_rung3_downstream_read_derisk as m
from sim.backend import to_host

m.set_map_density(4)
SEED = 200
sb, pos, cells, l2 = m.build(SEED, l2_w0=150.0)

def ca1_to_l2_weights(sb):
    """Rows = CA1 cells, value = mean |w| on that cell's pathway into L2."""
    W = sb.cp_connections
    dat = np.asarray(to_host(W.data)); ind = np.asarray(to_host(W.indices)); ptr = np.asarray(to_host(W.indptr))
    out = []
    for c in range(m.N_CELL):
        src = set(int(i) for i in cells[c]); tgt = set(int(i) for i in l2)
        vals = []
        for r in range(len(ptr)-1):
            if r not in src: continue
            for k in range(ptr[r], ptr[r+1]):
                if int(ind[k]) in tgt: vals.append(abs(float(dat[k])))
        out.append(np.mean(vals) if vals else 0.0)
    return np.array(out)

w_pre = ca1_to_l2_weights(sb)
ca1_pre, _ = m.run_lap(sb, pos, cells, l2, ca1_targets=m.CELL_TARGETS, bin_steps=200, record=True)
sb.core_config.enable_btsp = True; sb.core_config.btsp_learning_rate = 0.02
m.run_lap(sb, pos, cells, l2, ca1_targets=m.CELL_TARGETS, bin_steps=200)
w_mid = ca1_to_l2_weights(sb)
tgt_bin = 7
m.run_lap(sb, pos, cells, l2, ca1_targets=None, l2_plateau_bin=tgt_bin, bin_steps=200)
w_post = ca1_to_l2_weights(sb)

print("\n=== LAYER-2 (CA1->L2) weight map, seed 200 ===")
print("per-CA1-cell mean |w| into L2:")
for c in range(m.N_CELL):
    print(f"  cell {c} (field bin {[0,1,3,7,11][c]:2d}): pre={w_pre[c]:8.4f}  post-stage1={w_mid[c]:8.4f}  post-stage2={w_post[c]:8.4f}")
peak, mean = w_post.max(), w_post.mean()
print(f"\nCONTRAST (peak/mean) after learning : {peak/mean:.3f}x")
print(f"PEDESTAL rise (mean post / mean pre) : {w_post.mean()/max(w_pre.mean(),1e-9):.3f}x")
print(f"peak={peak:.4f}  mean={mean:.4f}  min={w_post.min():.4f}  spread(max-min)={peak-w_post.min():.4f}")
print(f"\nLayer 1 reference (from the record): contrast ~1.6x on a pedestal raised ~5x (0.600 -> 2.9)")
