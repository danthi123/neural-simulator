import os, sys
os.environ["SIM_BACKEND"] = "numpy"
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from research.findings.raw.direction_7_bridge_builder import build_direction_7_bridge_A_nouns
from research.findings.raw.direction_8_bridge_builder import build_direction_8_bridge_A_nouns

b7 = build_direction_7_bridge_A_nouns(seed=42, n_lang_input=512, n_per_pool=20, n_fs_per_pool=4, weak_dynamics=True, verbose=False)
b8 = build_direction_8_bridge_A_nouns(seed=42, n_lang_input=1024, n_per_pool=20, n_fs_per_pool=4, weak_dynamics=True, verbose=False)

def local_edges(bridge, pool_region):
    rm = bridge.region_manager
    li = list(rm.indices("language_input")); li_start = li[0]; li_set = {g:g-li_start for g in li}
    pool = list(rm.indices(pool_region)); pool_start = pool[0]; pool_set = {g:g-pool_start for g in pool}
    csr = bridge.cp_connections
    indptr = np.asarray(csr.indptr); indices = np.asarray(csr.indices)
    edges = set()
    for pre in li:  # row = pre = lang_input
        for off in range(int(indptr[pre]), int(indptr[pre+1])):
            post = int(indices[off])
            if post in pool_set:
                edges.add((li_set[pre], pool_set[post]))
    return edges

# Test determinism first: same builder, same seed, twice -> identical?
b7b = build_direction_7_bridge_A_nouns(seed=42, n_lang_input=512, n_per_pool=20, n_fs_per_pool=4, weak_dynamics=True, verbose=False)
e7 = local_edges(b7, "noun_pool_APPLE")
e7b = local_edges(b7b, "noun_pool_APPLE")
print(f"determinism check (D7 twice): {'IDENTICAL' if e7==e7b else 'NONDETERMINISTIC'} ({len(e7)} vs {len(e7b)})")

e8 = local_edges(b8, "noun_pool_APPLE")
inter = e7 & e8; union = e7 | e8
jac = len(inter)/max(1,len(union))
print(f"D7 APPLE edges: {len(e7)}  D8 APPLE edges: {len(e8)}")
print(f"intersection: {len(inter)}  Jaccard: {jac:.3f}")
print()
if jac > 0.99:
    print("MATCH -> raw-weight transplant CLEAN")
elif jac > 0.5:
    print(f"PARTIAL ({jac:.0%}) -> lossy transplant (intersection only)")
else:
    print(f"DIVERGED ({jac:.0%}) -> raw-weight transplant NOT viable; connectivity differs across n_lang")
