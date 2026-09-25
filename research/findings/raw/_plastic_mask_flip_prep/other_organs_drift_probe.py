"""Vikunja #203 flip-prep, coordinator follow-up (2026-09-24): does each of affect/curiosity/surprise/
world-model/metacognition drift under its OWN normal-use read, flag OFF, one seed (42), a small number of
reads each?

Each organ is built via its OWN production `get_organ(seed=42)` entry point (the same one webapp/server.py
calls) -- NOT a bespoke standalone construction -- so this measures the ACTUAL production code path,
including whichever pool (or lack of one) that path resolves to today. `cp_synapse_plastic_mask` is sliced
to `[:nnz]` (the mask array is capacity-padded like every other per-synapse gate array in this codebase --
see sim/bridge.py's `_ensure_gate_capacity` docstring -- so an unsliced compare against `nnz`-sized weight
deltas silently over/under-counts).

CPU/numpy throughout. Run flag OFF then flag ON as two separate processes (BRAIN_ENFORCE_PLASTIC_MASK unset
vs =1), since several of these organs share ONE process-global wave3-pool bridge
(research.runners.onebrain_wave3_pool_production) once any of them builds it.
"""
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def _to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _drift(bridge, w0):
    nnz = bridge.cp_connections.nnz
    w1 = _to_host(bridge.cp_connections.data)[:nnz]
    dw = np.abs(w1 - w0)
    mask = getattr(bridge, "cp_synapse_plastic_mask", None)
    out = {"nnz": int(nnz), "whole_bridge_max_dw": float(dw.max()) if dw.size else 0.0,
          "enable_hebbian_learning": bool(bridge.core_config.enable_hebbian_learning),
          "enable_stdp": bool(bridge.core_config.enable_stdp)}
    if mask is not None:
        m = _to_host(mask)[:nnz].astype(bool)
        out["mask_present"] = True
        out["n_frozen"] = int((~m).sum())
        out["n_plastic"] = int(m.sum())
        out["frozen_max_dw"] = float(dw[~m].max()) if (~m).any() else 0.0
        out["plastic_max_dw"] = float(dw[m].max()) if m.any() else 0.0
    else:
        out["mask_present"] = False
    return out


def probe_affect(seed):
    import research.runners.affect_production_organ as AF
    org = AF.get_organ(seed=seed)
    org.ensure_built()
    b = org.bridge
    w0 = _to_host(b.cp_connections.data)[:b.cp_connections.nnz].copy()
    for _ in range(10):
        r = org.read_differential(0.8)
    d = _drift(b, w0)
    d["organ"] = "affect"
    d["shared_pool"] = bool(getattr(org, "_shared", None) is not None)
    d["sample_reply"] = {k: r[k] for k in ("differential", "pos_rate", "neg_rate")}
    return d


def probe_curiosity(seed):
    import research.runners.curiosity_production_organ as CU
    org = CU.get_organ(seed=seed)
    org.ensure_built()
    b = org.bridge
    w0 = _to_host(b.cp_connections.data)[:b.cp_connections.nnz].copy()
    for _ in range(10):
        r = org.judge()
    d = _drift(b, w0)
    d["organ"] = "curiosity"
    d["shared_pool"] = bool(getattr(org, "_shared", None) is not None)
    d["sample_reply"] = {k: r[k] for k in ("novelty", "want_hz", "curious")}
    return d


def probe_surprise(seed):
    import research.runners.surprise_production_organ as SU
    org = SU.get_organ(seed=seed)
    org.ensure_built()
    b = org.bridge
    w0 = _to_host(b.cp_connections.data)[:b.cp_connections.nnz].copy()
    for _ in range(10):
        r = org.judge("dog", "chase", "cat", "bone")
    d = _drift(b, w0)
    d["organ"] = "surprise"
    d["shared_pool"] = bool(getattr(org, "_shared", None) is not None)
    d["sample_reply"] = {k: r[k] for k in ("surprise_hz", "surprised")}
    return d


def probe_worldmodel(seed):
    import research.runners.worldmodel_production_organ as WM
    org = WM.get_organ(seed=seed)
    org.ensure_built()
    b = org.bridge
    w0 = _to_host(b.cp_connections.data)[:b.cp_connections.nnz].copy()
    for _ in range(10):
        r = org.read_surprise(1, -1)
    d = _drift(b, w0)
    d["organ"] = "worldmodel"
    d["shared_pool"] = bool(getattr(org, "_shared", None) is not None)
    d["sample_reply"] = {k: r[k] for k in ("surprise_hz", "surprised")}
    return d


def probe_metacog(seed):
    import research.runners.metacog_production_organ as MC
    org = MC.get_organ(seed=seed)
    org.ensure_built()
    b = org.bridge
    w0 = _to_host(b.cp_connections.data)[:b.cp_connections.nnz].copy()
    for _ in range(10):
        r = org.judge(0.8)
    d = _drift(b, w0)
    d["organ"] = "metacog"
    d["shared_pool"] = bool(getattr(org, "_shared", None) is not None)
    d["sample_reply"] = {k: r[k] for k in list(r.keys())[:3]}
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    enforced = os.environ.get("BRAIN_ENFORCE_PLASTIC_MASK", "").strip().lower() not in ("", "0", "false", "no", "off")
    results = {}
    for name, fn in (("affect", probe_affect), ("curiosity", probe_curiosity), ("surprise", probe_surprise),
                     ("worldmodel", probe_worldmodel), ("metacog", probe_metacog)):
        try:
            results[name] = fn(a.seed)
        except Exception as e:
            results[name] = {"organ": name, "error": f"{type(e).__name__}: {e}"}
    out = {"seed": a.seed, "enforced_flag_read": enforced, "organs": results}
    js = json.dumps(out, indent=2)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(js)
    print(js)
