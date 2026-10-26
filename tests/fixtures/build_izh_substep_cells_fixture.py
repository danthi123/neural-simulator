"""Build tests/fixtures/izh_substep_cells.npz: the recorded per-step input of the three BG-organ MSNs whose
explicit-Euler (dt = 1 ms) membrane ran away (research/findings/2026-10-26-izh-membrane-substeps-derisk.md).

Source: the observe-only organ probe of the 2026-10-26 BG breach diagnosis (probe_breach.py --arm intact,
research/bg-tonic-loop @ 3f1249c79, SIM_BACKEND=numpy, BRAIN_BG_NEURAL_SEAM=1 BRAIN_BG_COMMIT=1 op 4,
BRAIN_BG_PRESET_CELLS=1 BRAIN_BG_BOUNDED_COND=1 BRAIN_BG_TONIC_DRIVE=1 GPe/STN 100/100 pA; dev seeds 72/75/77;
build + the four hsb select_once events = 2755 steps). Its trace.npz files hold, per step and neuron, the
decayed conductances the bounded kernel used (g_e, g_i) and the other current terms (external, OU, slow-NMDA
recurrent). For each breaching cell this script keeps g_e, g_i and I_other = ext + ou + I_nmda_rec as float32,
the cell's drawn parameters, and the organ's own read-outs (v_min, spike count, first step below -150 mV), so
tests/test_izh_membrane_substeps.py can replay the cell open-loop through a one-neuron SimulationBridge.

Usage: python tests/fixtures/build_izh_substep_cells_fixture.py <diag runs dir> [out.npz]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

CELLS = ((72, 439, "IZH2007_STRIATAL_MSN_D2"), (75, 145, "IZH2007_STRIATAL_MSN_D1"),
         (77, 418, "IZH2007_STRIATAL_MSN_D1"))
PARAMS = ("a", "b", "C", "k", "vr", "vt", "vpeak", "c_reset", "d_increment")


def build(runs_dir: str, out_path: str) -> dict:
    arrays, meta = {}, {"source": "probe_breach.py --arm intact (diag_1026/bg), research/bg-tonic-loop @ 3f1249c79",
                        "backend": "numpy", "dt_ms": 1.0, "cells": {}}
    for seed, idx, preset in CELLS:
        d = os.path.join(runs_dir, f"s{seed}_intact")
        Z = np.load(os.path.join(d, "trace.npz"))
        M = json.load(open(os.path.join(d, "meta.json")))
        key = f"s{seed}"
        other = Z["ext"][:, idx].astype(np.float32) + Z["ou"][:, idx].astype(np.float32)
        if "I_nmda_rec" in Z.files:
            other = other + Z["I_nmda_rec"][:, idx].astype(np.float32)
        arrays[f"{key}_g_e"] = Z["g_e"][:, idx].astype(np.float32)
        arrays[f"{key}_g_i"] = Z["g_i"][:, idx].astype(np.float32)
        arrays[f"{key}_I_other"] = other.astype(np.float32)
        v = Z["v"][:, idx]
        meta["cells"][key] = {
            "seed": seed, "idx": idx, "region": M["regions"][int(Z["reg_of"][idx])], "preset": preset,
            "params": {p: float(Z["par_" + p][idx]) for p in PARAMS},
            "v0": float(Z["v_in"][0, idx]), "u0": float(Z["u_in"][0, idx]), "n_steps": int(v.shape[0]),
            "organ": {"v_min": float(v.min()), "spikes": int(Z["fired"][:, idx].sum()),
                      "first_step_lt_m150": int(np.argmax(v < -150)) if (v < -150).any() else None},
        }
    arrays["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    np.savez_compressed(out_path, **arrays)
    return meta


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "izh_substep_cells.npz")
    m = build(sys.argv[1], out)
    print(json.dumps(m, indent=1))
    print("wrote", out, os.path.getsize(out), "bytes")
