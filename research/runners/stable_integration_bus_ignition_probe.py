"""BUS WORKSPACE IGNITION, OLD vs STABLE INTEGRATOR (research/stable-integration).

The attribution probe (research/findings/2026-10-26-stable-integration-dev7.md) shows the GNW ignition-bus workspace
(`_p1_2_workspace_deliberation_loop_derisk.build_workspace_bridge`, 4 self-recurrent NMDA assemblies of 80 cells
sharing one FS pool) decides recall verification: on the stable integrator (SIM_STABLE_INTEGRATION=1) it stops
verifying. This probe measures the organ in isolation, exactly as the bus reads it (`_ignite_and_read`: restore the
quiescent snapshot, drive the slots for DRIVE_STEPS, free-run FREE_STEPS, late-window rate per slot; a slot is ignited
iff that rate >= IGNITE_FRAC * SOLO_PLATEAU = 1/6), on both integrators, across the assembly's recurrent weight
(production 30), so the drive constant the stable integrator needs is a measured number, not a guess.

Conditions per (integrator, attractor_weight): `solo` (target slot only at IGNITE_PA), `wta` (target at IGNITE_PA,
3 distractors at DISTRACTOR_FRAC), `none` (no drive: false-ignition control), `lesion` (assembly loops zeroed, the
organ's own workspace-silence anti-cheat). Per condition: late rates per slot, winner, n_ignited, and the workspace
membrane range over the whole read (cells below -100 mV or above the spike peak count as out of range).
No sim/ or webapp/ edit; dev seeds only (gate seeds refused).

  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' .venv/bin/python -m research.runners.stable_integration_bus_ignition_probe \
      --seed 7 --out research/findings/raw/_stable_integration/bus_ignition_s7.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

ENV = "SIM_STABLE_INTEGRATION"
GATE_SEEDS = (42, 43, 44, 100, 101, 102)
WEIGHTS = (30.0, 45.0, 60.0, 90.0, 120.0)
V_LOW = -100.0


def _read(P, bridge, xp, slots_dev, snap, drives):
    """`_ignite_and_read` with a membrane watch: the same drive/free schedule and late-window rate read, plus the
    min/max membrane and the out-of-range cell count of the workspace slots over every step of the read."""
    from sim.backend import to_host
    bridge.cp_external_input_current[:] = 0.0
    P._full_restore(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    allidx = xp.concatenate(slots_dev)
    n_all = int(allidx.shape[0])
    oor = np.zeros(n_all, dtype=bool)
    vmin, vmax = np.inf, -np.inf
    late_v, late_hi = [], []

    def _watch():
        nonlocal vmin, vmax
        v = to_host(bridge.cp_membrane_potential_v[allidx]).astype(np.float64)
        vmin = min(vmin, float(v.min()))
        vmax = max(vmax, float(v.max()))
        oor[:] |= (v < V_LOW) | (v > 60.0)

    for _ in range(P.DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        for s_dev, d in zip(slots_dev, drives):
            if d > 0.0:
                bridge.cp_external_input_current[s_dev] = xp.float32(d)
        bridge._run_one_simulation_step()
        _watch()
    bridge.cp_external_input_current[:] = 0.0
    late_start = P.FREE_STEPS - max(1, P.FREE_STEPS // 3)
    counts = [0] * len(slots_dev)
    free_counts = [0] * len(slots_dev)
    for t in range(P.FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        _watch()
        for i, s_dev in enumerate(slots_dev):
            c = int(to_host(bridge.cp_firing_states[s_dev].astype(xp.float64).sum()))
            free_counts[i] += c
            if t >= late_start:
                counts[i] += c
        if t >= late_start:
            v0 = to_host(bridge.cp_membrane_potential_v[slots_dev[0]]).astype(np.float64)
            late_v.append(float(v0.mean()))
            late_hi.append(float((v0 > -20.0).mean()))
    denom = float((P.FREE_STEPS - late_start) * P.ASSEMBLY_SIZE)
    rates = [c / denom for c in counts]
    thr = P.IGNITE_FRAC * P.SOLO_PLATEAU
    w = int(np.argmax(rates))
    return {"drives": [float(d) for d in drives], "late_rates": [round(r, 4) for r in rates],
            "free_rates": [round(c / float(P.FREE_STEPS * P.ASSEMBLY_SIZE), 4) for c in free_counts],
            "winner": w, "winner_rate": round(rates[w], 4), "n_ignited": int(sum(r >= thr for r in rates)),
            "ignited": bool(rates[w] >= thr), "v_min": round(vmin, 2), "v_max": round(vmax, 2),
            "n_oor_cells": int(oor.sum()), "n_cells": n_all,
            "slot0_late_mean_v": round(float(np.mean(late_v)), 2),
            "slot0_late_frac_above_m20mV": round(float(np.mean(late_hi)), 4)}


def run(seed, weights=WEIGHTS, fs_weights=None, arms=("off", "on")):
    """`fs_weights` (optional) also sweeps the workspace_fs -> workspace inhibition (production FS_TO_WS_WEIGHT),
    set on the builder module for the build and restored after."""
    from research.runners import _p1_2_workspace_deliberation_loop_derisk as P
    from tools.lab import attributable_to, lever
    fs_prod = P.FS_TO_WS_WEIGHT
    fs_list = list(fs_weights) if fs_weights else [fs_prod]
    out = {"runner": "research/runners/stable_integration_bus_ignition_probe.py", "seed": int(seed),
           "ignite_threshold": round(P.IGNITE_FRAC * P.SOLO_PLATEAU, 4), "solo_plateau": round(P.SOLO_PLATEAU, 4),
           "ignite_pa": P.IGNITE_PA, "distractor_frac": P.DISTRACTOR_FRAC, "drive_steps": P.DRIVE_STEPS,
           "free_steps": P.FREE_STEPS, "production_attractor_weight": P.DEFAULT_ATTRACTOR_WEIGHT,
           "weights": list(weights), "production_fs_to_ws_weight": fs_prod, "fs_to_ws_weights": fs_list,
           "arms": {}}
    saved = os.environ.get(ENV)
    try:
        for arm in arms:
            if arm == "on":
                os.environ[ENV] = "1"
            else:
                os.environ.pop(ENV, None)
            rows = []
            for fsw, w in [(f, x) for f in fs_list for x in weights]:
                P.FS_TO_WS_WEIGHT = fsw
                row = {"attractor_weight": w, "fs_to_ws_weight": fsw}
                for lesion in (False, True):
                    bridge, xp, slots_dev, snap = P.build_workspace_bridge(int(seed), attractor_weight=w,
                                                                          lesion=lesion)
                    cfg = bridge.core_config
                    rb = {"requested": bool(bridge.stable_integration_requested),
                          "bounded": bool(cfg.enable_bounded_conductance_integration),
                          "substeps": int(cfg.izh_membrane_substeps),
                          "rule": cfg.izh_substep_refractory_rule}
                    if rb["requested"] != (arm == "on"):
                        raise RuntimeError("bridge read the wrong switch value")
                    n = len(slots_dev)
                    if lesion:
                        row["lesion"] = _read(P, bridge, xp, slots_dev, snap, [P.IGNITE_PA] + [0.0] * (n - 1))
                    else:
                        row["readback"] = rb
                        row["solo"] = _read(P, bridge, xp, slots_dev, snap, [P.IGNITE_PA] + [0.0] * (n - 1))
                        row["wta"] = _read(P, bridge, xp, slots_dev, snap,
                                           [P.IGNITE_PA] + [P.IGNITE_PA * P.DISTRACTOR_FRAC] * (n - 1))
                        row["none"] = _read(P, bridge, xp, slots_dev, snap, [0.0] * n)
                rows.append(row)
                P.FS_TO_WS_WEIGHT = fs_prod
                # whose is the sustained late rate? the fraction NOT present in the loop-lesion control and in the
                # no-drive control (tools.lab.attributable_to; None = no effect in either arm, a null)
                lbl = "%s fs=%g w=%g solo late rate" % (arm, fsw, w)
                row["attributable_to_loops_vs_lesion"] = attributable_to(
                    lbl + " vs loop lesion", row["solo"]["winner_rate"], row["lesion"]["winner_rate"])
                row["attributable_to_drive_vs_none"] = attributable_to(
                    lbl + " vs no drive", row["solo"]["winner_rate"], row["none"]["winner_rate"])
                print("[bus ignition] %s fs=%4.1f w=%5.1f solo %.3f (oor %d, vmin %.1f) wta win=%d %.3f n_ign=%d none %.3f "
                      "lesion %.3f" % (arm, fsw, w, row["solo"]["winner_rate"], row["solo"]["n_oor_cells"],
                                       row["solo"]["v_min"], row["wta"]["winner"], row["wta"]["winner_rate"],
                                       row["wta"]["n_ignited"], row["none"]["winner_rate"],
                                       row["lesion"]["winner_rate"]), flush=True)
            out["arms"][arm] = rows
        if "off" in out["arms"] and "on" in out["arms"]:
            # the manipulation itself: the two arms must have run different integrators, else the A/B is void
            lever("membrane integrator (bounded, substeps)",
                  (out["arms"]["off"][0]["readback"]["bounded"], out["arms"]["off"][0]["readback"]["substeps"]),
                  (out["arms"]["on"][0]["readback"]["bounded"], out["arms"]["on"][0]["readback"]["substeps"]))
    finally:
        P.FS_TO_WS_WEIGHT = fs_prod
        if saved is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = saved
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=",".join(str(w) for w in WEIGHTS))
    ap.add_argument("--fs-weights", default="", help="comma list; default = the production FS_TO_WS_WEIGHT only")
    ap.add_argument("--arms", default="off,on")
    a = ap.parse_args()
    if a.seed in GATE_SEEDS:
        sys.exit("GATE seed %d refused: dev-seed diagnostic only" % a.seed)
    d = run(a.seed, weights=[float(x) for x in a.weights.split(",")],
            fs_weights=[float(x) for x in a.fs_weights.split(",") if x.strip()] or None,
            arms=tuple(x.strip() for x in a.arms.split(",") if x.strip()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(d, f, indent=1)
    print("[bus ignition] wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
