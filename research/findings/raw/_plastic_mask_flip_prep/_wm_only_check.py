import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import json
import numpy as np


def _to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


import research.runners.worldmodel_production_organ as WM

seed = 42
org = WM.get_organ(seed=seed)
org.ensure_built()
b = org._st["bridge"]
nnz = b.cp_connections.nnz
w0 = _to_host(b.cp_connections.data)[:nnz].copy()
for _ in range(10):
    r = org.read_surprise(1, -1)
w1 = _to_host(b.cp_connections.data)[:nnz]
dw = np.abs(w1 - w0)
mask = getattr(b, "cp_synapse_plastic_mask", None)
out = {
    "runner": "research/findings/raw/_plastic_mask_flip_prep/_wm_only_check.py",
    "seed": seed,
    "backend": "numpy",
    "organ": "worldmodel",
    "shared_pool": bool(getattr(org, "_shared", None) is not None),
    "nnz": int(nnz),
    "enable_hebbian_learning": bool(b.core_config.enable_hebbian_learning),
    "enable_stdp": bool(b.core_config.enable_stdp),
    "whole_bridge_max_dw": float(dw.max()) if dw.size else 0.0,
    "sample_reply": {k: r[k] for k in ("surprise_hz", "surprised")},
    "enforced_flag_read": os.environ.get("BRAIN_ENFORCE_PLASTIC_MASK", "").strip().lower()
    not in ("", "0", "false", "no", "off"),
}
if mask is not None:
    m = _to_host(mask)[:nnz].astype(bool)
    out["mask_present"] = True
    out["n_frozen"] = int((~m).sum())
    out["n_plastic"] = int(m.sum())
    out["frozen_max_dw"] = float(dw[~m].max()) if (~m).any() else 0.0
    out["plastic_max_dw"] = float(dw[m].max()) if m.any() else 0.0
else:
    out["mask_present"] = False
print(json.dumps(out, indent=2))
