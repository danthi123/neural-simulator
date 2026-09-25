"""Positive control for Vikunja #203 flip-prep, requested by the coordinator (2026-09-24 follow-up).

Reproduces research/findings/raw/_read_isolation_audit_29/diag_comp_drift_accum.py's ORIGINAL protocol
EXACTLY (same seed, same battery call, same single item, same 30-read loop, same wnorm measurement against
w_init) -- the script that produced the 13.8->56.1 max-weight-drift finding
(research/findings/raw/_read_isolation_audit_29/audit_29runners.json) -- so this run is a genuine
before/after on the SAME instrument, not a new one. The only addition: an --arm flag that sets/unsets
BRAIN_ENFORCE_PLASTIC_MASK before sim.bridge's first import (must happen before ANY sim import, matching
this project's env-flag convention), so the same protocol can be compared flag-off vs flag-on.

Usage (numpy/CPU, matching the original diagnostic and this branch's CPU-only rule):
    SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python \\
        research/findings/raw/_plastic_mask_flip_prep/positive_control_drift.py --arm=off \\
        --out=research/findings/raw/_plastic_mask_flip_prep/positive_control_off_s42.json
    SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= PYTHONPATH=. BRAIN_ENFORCE_PLASTIC_MASK=1 .venv/bin/python \\
        research/findings/raw/_plastic_mask_flip_prep/positive_control_drift.py --arm=on \\
        --out=research/findings/raw/_plastic_mask_flip_prep/positive_control_on_s42.json
"""
import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def run(seed, arm):
    assert arm in ("off", "on")
    enforced = os.environ.get("BRAIN_ENFORCE_PLASTIC_MASK", "").strip().lower() not in ("", "0", "false", "no", "off")

    from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, build_battery

    org = ComprehensionProductionOrgan(seed=seed)
    org.ensure_built()
    b = org.comp.bridge
    items = build_battery(seed, n_per_cond=6)
    (_lab, _tag, n0, v, n1) = items[0]

    # Locate the sel_*<->sel_FS_* pathways by name, exactly the ones the original audit named, so this
    # control also reports the NAMED-pathway breakdown, not just the whole-bridge max.
    mask = getattr(b, "cp_synapse_plastic_mask", None)
    mask_host = np.asarray(mask) if mask is not None else None
    rm = getattr(b, "region_manager", None)
    named = {}
    if rm is not None:
        try:
            idx_by_region = {r.name: set(int(i) for i in rm.indices(r.name)) for r in rm.regions()}
            coo = b.cp_connections.tocoo(copy=False)
            row, col = coo.row, coo.col
            for fr, to in (("sel_agent", "sel_FS_agent"), ("sel_patient", "sel_FS_patient"),
                          ("sel_FS_agent", "sel_patient"), ("sel_FS_patient", "sel_agent")):
                if fr in idx_by_region and to in idx_by_region:
                    frs, tos = idx_by_region[fr], idx_by_region[to]
                    sel = np.array([(int(r) in frs) and (int(c) in tos) for r, c in zip(row, col)], dtype=bool)
                    named[f"{fr}->{to}"] = np.where(sel)[0]
        except Exception as e:
            named = {"_error": str(e)}

    w_init = np.asarray(b.cp_connections.data).copy()
    margins, wnorms = [], []
    named_wnorms = {k: [] for k in named if k != "_error"}
    for i in range(30):
        m = org.read_margin(n0, v, n1)
        margins.append(m)
        w = np.asarray(b.cp_connections.data)
        wnorms.append(float(np.abs(w - w_init).max()))
        for k, idx in named.items():
            if k == "_error" or not isinstance(idx, np.ndarray) or idx.size == 0:
                continue
            named_wnorms[k].append(float(np.abs(w[idx] - w_init[idx]).max()))

    frozen_final = None
    if mask_host is not None:
        w_final = np.asarray(b.cp_connections.data)
        frozen_idx = np.where(~mask_host[:w_final.size])[0]
        frozen_final = float(np.abs(w_final[frozen_idx] - w_init[frozen_idx]).max()) if frozen_idx.size else 0.0

    return {
        "arm": arm,
        "enforced_flag_read": enforced,
        "seed": seed,
        "item0": [n0, v, n1],
        "margins": margins,
        "max_weight_drift_from_init": wnorms,
        "final_max_weight_drift_whole_bridge": wnorms[-1] if wnorms else None,
        "final_frozen_only_max_drift": frozen_final,
        "named_pathway_drift_over_reads": named_wnorms,
        "named_pathway_error": named.get("_error"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", choices=["off", "on"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.seed, a.arm)
    js = json.dumps(res, indent=2)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(js)
    print(js)
