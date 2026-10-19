#!/usr/bin/env python3
"""Amendment 2 calibration: pick I_tonic by the physiological rule (docs/drafts/bg-commit/WORKLOG.md,
rule declared 2026-10-19 before this run).  Flag-ON, seed 9, one bridge; per grid point: 200-step
equilibration (discarded, appended to the WORKLOG before this run) then 1000 ms measured.
Scan 1: GPe grid {25,50,100,200,400,700,1000} pA with STN held 0.
Scan 2: STN grid with GPe held at the scan-1 pick (0 if none in band... rule: pick = smallest IN band).
No bar is read: only per-neuron Hz vs the declared bands (GPe 20-60, STN 10-30).
"""
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["BRAIN_BG_TONIC_DRIVE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from sim.backend import to_host  # noqa: E402
from research.runners._vocal_action_selector_gate import (  # noqa: E402
    build_selector_bridge, selector_config, _set_equal_tonic_current, _step, _indices, CHANNELS,
)

GRID = [25, 50, 100, 200, 400, 700, 1000]
WARMUP, MEASURE = 200, 1000
GPE_BAND, STN_BAND = (20.0, 60.0), (10.0, 30.0)


def rates(bridge, n_gpe, n_stn):
    """Per-neuron mean Hz over MEASURE steps (spikes / neurons / ms * 1000)."""
    gpe_spk = {c: 0 for c in CHANNELS}
    stn_spk = 0
    gpe_idx = {c: _indices(bridge, f"gpe_{c}") for c in CHANNELS}
    stn_idx = _indices(bridge, "selector_stn")
    for _ in range(WARMUP):
        _step(bridge)
    for _ in range(MEASURE):
        _step(bridge)
        f = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for c in CHANNELS:
            gpe_spk[c] += int(f[gpe_idx[c]].sum())
        stn_spk += int(f[stn_idx].sum())
    gpe_hz = [gpe_spk[c] / (n_gpe * MEASURE) * 1000.0 for c in CHANNELS]
    stn_hz = stn_spk / (n_stn * MEASURE) * 1000.0
    return gpe_hz, stn_hz


def main():
    cfg = selector_config("v2")
    bridge = build_selector_bridge(9, cfg)          # seed 9 (first dev seed), declared
    n_gpe, n_stn = int(cfg.n_gpe), int(cfg.n_stn)

    def set_currents(gpe_pA, stn_pA):
        os.environ["BRAIN_BG_TONIC_GPE_PA"] = str(float(gpe_pA))
        os.environ["BRAIN_BG_TONIC_STN_PA"] = str(float(stn_pA))
        _set_equal_tonic_current(bridge, cfg)

    results = {"rule": "smallest pA in band; no bar read", "seed": 9, "warmup": WARMUP,
               "measure_ms": MEASURE, "grid": GRID, "scan_gpe": [], "scan_stn": []}

    gpe_pick = 0.0
    for pA in GRID:
        set_currents(pA, 0.0)
        gpe_hz, stn_hz = rates(bridge, n_gpe, n_stn)
        in_band = GPE_BAND[0] <= float(np.mean(gpe_hz)) <= GPE_BAND[1]
        results["scan_gpe"].append({"pA": pA, "gpe_hz": [round(h, 2) for h in gpe_hz],
                                    "stn_hz": round(stn_hz, 2), "in_band": bool(in_band)})
        if in_band and gpe_pick == 0.0:
            gpe_pick = float(pA)
    if gpe_pick == 0.0:
        gpe_pick = 0.0                              # no pick in declared grid -> extension is DECLARED first

    stn_pick = 0.0
    for pA in GRID:
        set_currents(gpe_pick, pA)
        gpe_hz, stn_hz = rates(bridge, n_gpe, n_stn)
        in_band = STN_BAND[0] <= float(stn_hz) <= STN_BAND[1]
        results["scan_stn"].append({"pA": pA, "gpe_hz": [round(h, 2) for h in gpe_hz],
                                    "stn_hz": round(stn_hz, 2), "in_band": bool(in_band)})
        if in_band and stn_pick == 0.0:
            stn_pick = float(pA)

    results["gpe_pick"] = gpe_pick
    results["stn_pick"] = stn_pick
    results["gpe_in_band"] = any(r["in_band"] for r in results["scan_gpe"])
    results["stn_in_band"] = any(r["in_band"] for r in results["scan_stn"])
    out = "/opt/data/cache/scratch/bg_tonic_calibration.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results, indent=1))
    print("WROTE", out)


if __name__ == "__main__":
    main()