"""Diff the bridge state left by RANK 1's _prepare (reactivates) vs RANK 2's _prepare_sequence (doesn't),
both at n_mem=1, seed 42 -> reveal the SECOND divergence blocking reactivation."""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np

from research.runners._gap5_spontaneous_reactivation_derisk import _prepare as r1_prepare, CFG as R1CFG
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence as r2_prepare, SEQ_CFG

# match configs for n_mem=1: RANK1 CFG has n_mem=1; RANK2 SEQ_CFG override n_mem=1, within_events=30, no chain
r1cfg = dict(R1CFG); r1cfg["n_mem"] = 1
r2cfg = dict(SEQ_CFG); r2cfg["n_mem"] = 1; r2cfg["within_events"] = 30; r2cfg["chain_fwd"] = 0; r2cfg["chain_rev"] = 0

print("building RANK1 _prepare (n_mem=1, seed 42)...", flush=True)
p1 = r1_prepare(42, r1cfg, do_encode=True)
print(f"  RANK1 w_within={p1['w_within']:.2f}", flush=True)
print("building RANK2 _prepare_sequence (n_mem=1, seed 42)...", flush=True)
p2 = r2_prepare(42, r2cfg, do_encode=True)
print(f"  RANK2 w_within={p2['w_within']:.2f}", flush=True)

c1 = p1["bridge"].core_config
c2 = p2["bridge"].core_config

# compare all config attrs that could affect recall/reactivation
keys = sorted(set([k for k in vars(c1)]) | set([k for k in vars(c2)]))
print("\n=== CONFIG DIFFERENCES (RANK1 vs RANK2) ===")
ndiff = 0
for k in keys:
    v1 = getattr(c1, k, "<MISSING>"); v2 = getattr(c2, k, "<MISSING>")
    try:
        same = bool(v1 == v2)
    except Exception:
        same = str(v1) == str(v2)
    if not same and not callable(v1):
        # skip huge arrays / irrelevant
        s1, s2 = str(v1), str(v2)
        if len(s1) < 60 and len(s2) < 60:
            print(f"  {k}: RANK1={s1}  |  RANK2={s2}")
            ndiff += 1
print(f"  ({ndiff} scalar config diffs)")

# compare key bridge-level recall state
b1, b2 = p1["bridge"], p2["bridge"]
print("\n=== BRIDGE RECALL-STATE ===")
for attr in ["coincidence_k_threshold"]:
    print(f"  cfg.{attr}: R1={getattr(c1,attr,'?')}  R2={getattr(c2,attr,'?')}")
for arr in ["cp_plasticity_rate_gain", "cp_transmission_gain"]:
    a1 = getattr(b1, arr, None); a2 = getattr(b2, arr, None)
    def summ(a):
        if a is None: return "None"
        import numpy as _np
        h = _np.asarray(a)
        return f"min={float(h.min()):.3g} max={float(h.max()):.3g} mean={float(h.mean()):.3g}"
    print(f"  {arr}: R1[{summ(a1)}]  R2[{summ(a2)}]")

# within-weight recurrent DRIVE proxy (mean * n_within_edges / n_assy_cells)
print(f"\n  within edges: R1={p1['within_flat'].size}  R2={p2['within_flat'].size}")
print(f"  assembly size: R1={len(p1['assemblies'][0])}  R2={len(p2['assemblies'][0])}")
