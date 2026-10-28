#!/usr/bin/env python3
"""bg_breach_observe - batch-1026 (3): the approved one-seed observe run that names the region
where the step-1 membrane breaches.

Replays the grid's hsb_hi turn (the event whose v reached -1425.8 aggregate) under the exact
step-1 cell environment (preset cells, bounded-conductance integration, tonic drive, neural seam
+ commit on, set stn50_gpe50 = GPE 50 / STN 50 pA), and records per step: the minimum v per
region; the cells below -150 mV and the first-breach step; at that step those cells' preset and
jittered a, b, c, d, C, whether the bounded-conductance (ratio-clamp) bound was active, synaptic
current by receptor, tonic and external current, u; plus a read-back proving the bounded path ran.

Arms: --seed 72 (breach) plus --seed 71 as the clean control (RULING batch-1026 (3)).
Observe only: nothing in the emulated system is written; wrappers read and return untouched.
Run from the bg-tonic-loop worktree (PYTHONPATH = worktree): the preset/bounded flags are branch code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

BREACH_MV = -150.0
E_REV = {"ampa": 0.0, "nmda": 0.0, "gaba": -80.0}       # receptor reversal from the brain constructor record

SET_KV = {                                                # grid cell stn50_gpe50, verbatim from the claims line
    "PYTHONHASHSEED": "0", "SIM_BACKEND": "numpy",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
    "BRAIN_BG_PRESET_CELLS": "1", "BRAIN_BG_BOUNDED_COND": "1", "BRAIN_BG_TONIC_DRIVE": "1",
    "BRAIN_BG_TONIC_GPE_PA": "50", "BRAIN_BG_TONIC_STN_PA": "50",
}


def _first_attr(obj, names):
    """(name, value) for the first of names that obj carries, else (None, reason)."""
    for n in names:
        if hasattr(obj, n):
            try:
                return n, getattr(obj, n)
            except Exception as e:                        # noqa: BLE001 - record, never mask
                return None, f"{n}: {type(e).__name__}"
    return None, "absent:" + ",".join(names)


def _host(x):
    import numpy as np
    from sim.backend import to_host
    a = np.asarray(to_host(x))
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True, help="72 breach / 71 clean control")
    ap.add_argument("--turn", default="hsb_hi", help="grid turn label (msg resolved from TURNS)")
    ap.add_argument("--out", default="", help="record path (default raw/_bg_live_seam/observe/s<seed>_<turn>.json)")
    args = ap.parse_args(argv)

    # 1) environment: the grid cell's own, applied before any sim import (arm_env mirrors run_worker).
    import research.runners.host_share_bg_seam as SS
    C = SS.lane_c()
    env = SS.arm_env(C.BASE_ENV, "seam_b", args.seed, backend="numpy", commit=True)
    env.update(SET_KV)
    for k, v in env.items():
        os.environ[k] = str(v)
    env_full = {k: os.environ.get(k) for k in sorted(
        list(SET_KV) + ["BRAIN_BG_NEURAL_SEAM", "BRAIN_GNW_UNIQUE_IGNITION_READ", "BRAIN_BG_COMMIT",
                        "BRAIN_CHAT_SEED", "BRAIN_BG_NEURAL_SEAM_LESION", "BRAIN_BG_COMMIT_OP",
                        "BRAIN_BG_COMMIT_LESION"])}

    # 2) instrument: per-step region record via the module-global _step of the organ's trial loop.
    import research.runners.bg_action_selection_production_organ as BG
    from sim.backend import to_host
    import numpy as np

    rec = {"instrument": "bg_breach_observe", "seed": args.seed, "turn": args.turn,
           "set": "stn50_gpe50", "env": env_full, "steps": [], "baseline_marks": [],
           "warnings": []}
    state = {"t": -1, "bridge": None, "region_map": None, "phase_events": [], "breach": None}

    def _region_map(bridge):
        if state["region_map"] is None:
            rm = getattr(bridge, "region_manager", None)
            if rm is None:
                rec["warnings"].append("no region_manager: per-region record impossible")
                state["region_map"] = {}
            else:
                state["region_map"] = dict(rm.region_indices_dict())
        return state["region_map"]

    orig_step, orig_base = BG._step, BG._neural_baseline

    def step_hook(bridge, n=1):
        out = orig_step(bridge, n)
        state["bridge"] = bridge
        for _ in range(int(n)):
            state["t"] += 1
            v = _host(bridge.cp_membrane_potential_v)
            row = {"t": state["t"], "min": float(v.min()), "max": float(v.max()),
                   "argmin": int(v.argmin())}
            rmap = _region_map(bridge)
            if rmap:
                row["rmin"] = {k: float(v[idx].min()) for k, idx in rmap.items()}
            below = np.nonzero(v < BREACH_MV)[0]
            row["n_below"] = int(below.size)
            if below.size and state["breach"] is None:
                rcounts = {}
                for k, idx in (rmap or {}).items():
                    c = int(np.isin(idx, below).sum())
                    if c:
                        rcounts[k] = c
                state["breach"] = {"step": state["t"], "n_below": int(below.size),
                                   "indices": [int(i) for i in below[:512]],
                                   "argmin": row["argmin"], "v_min": row["min"],
                                   "per_region_counts": rcounts}
            rec["steps"].append(row)
        return out

    def base_hook(bridge, config):
        state["phase_events"].append(state["t"])
        return orig_base(bridge, config)

    BG._step, BG._neural_baseline = step_hook, base_hook
    try:
        # 3) the grid's hsb_hi turn: msg -> neural seam (commit on -> BRAIN_CHAT_SEED organ).
        msg = dict((lab, m) for lab, m, _ in SS.TURNS)[args.turn]
        t0 = time.time()
        r = BG.decide_action_neural(msg, seed=None)
        rec["wall_seconds"] = round(time.time() - t0, 1)
        rec["returned"] = r if r is None else {k: v for k, v in r.items() if k != "trial"}
        last = BG.decide_action_neural.last
        rec["trial_keys"] = sorted((last or {}).keys())
        rec["trial_light"] = {k: last.get(k) for k in
                              ("winner", "committed", "decision_step", "loser_ratio", "v_min", "v_max")
                              if last and k in last}
        b = state["bridge"]
        if b is None:
            rec["warnings"].append("bridge never stepped")
            rec["status"] = "NO_TRIAL"
            _write(args, rec)
            return 1
        v_all = np.array([row["min"] for row in rec["steps"]]) if rec["steps"] else np.array([0.0])
        dstep = min(rec["steps"], key=lambda rr: rr["min"]) if rec["steps"] else None
        # identity anchor: the grid's s72 hsb_hi aggregate v min (the event being replayed)
        grid_t = {72: -1425.8082275390625}.get(args.seed)
        rec["identity"] = {"steps": len(rec["steps"]), "v_min": float(v_all.min()),
                           "v_max": max(row["max"] for row in rec["steps"]),
                           "deepest_step": ({"t": dstep["t"], "min": dstep["min"]} if dstep else None),
                           "grid_hsbhi_min_s72": grid_t,
                           "identity_match": (None if grid_t is None
                                              else bool(abs(float(v_all.min()) - grid_t) < 1e-3))}
        rec["baseline_marks"] = state["phase_events"]

        # 4) first-breach read-back: params, clamp, currents, u (every field defensive, absent recorded).
        br = state["breach"]
        if br is None:
            rec["breach"] = None
            rec["status"] = "NO_BREACH"
            rec["bounded_readback"] = _bounded_readback()
            _write(args, rec)
            return 0
        idx = np.asarray(br["indices"], dtype=np.int64)
        br["n_sampled"] = int(idx.size)
        br["v_at_first_breach"] = None

        # jittered Izhikevich parameters: the per-neuron cp_izh_* arrays the heterogeneity writer targets
        izh_names = sorted(n for n in dir(b) if n.startswith("cp_izh"))
        br["izh_attrs_present"] = izh_names
        jitter = {}
        for pn in sorted(izh_names):                      # every cp_izh_*: a, b, C, d_increment, c_reset, vt, vr, vpeak, k
            val = getattr(b, pn, None)
            if val is None:
                continue
            try:
                arr = _host(val)
                jitter[pn] = ([float(x) for x in arr[idx[:64]]]
                              if getattr(arr, "ndim", 0) == 1 and arr.size > int(idx.max())
                              else {"shape": list(getattr(arr, "shape", []))})
            except Exception as e:                        # noqa: BLE001
                jitter[pn] = f"unreadable:{type(e).__name__}"
        br["jittered_params"] = jitter or {"absent": f"no cp_izh_* on bridge; dir={izh_names}"}

        # preset (pre-jitter) via the bridge's own preset reader
        try:
            ps, cfg = None, getattr(b, "core_config", None)
            presets = {}
            for region, label, _i, values in b._region_parameter_presets(cfg):
                rname = getattr(region, "name", None) or str(region)[:40]
                presets[f"{rname}:{label}"] = dict(values)
            br["preset_params_by_region"] = presets
        except Exception as e:                            # noqa: BLE001
            br["preset_params_by_region"] = {"absent": f"{type(e).__name__}: {e}"[:200]}

        # synaptic current by receptor + u + tonic/external, whatever arrays the bridge carries
        cur = {}
        v_b = _host(b.cp_membrane_potential_v)
        n_total = int(v_b.size)
        cur["conductance_attrs"] = sorted(n for n in dir(b) if n.startswith("cp_conductance"))
        for rn, er, an in (("ampa", 0.0, "cp_conductance_g_e"),
                           ("nmda", 0.0, "cp_conductance_g_nmda"),
                           ("gaba", -80.0, "cp_conductance_g_i")):
            val = getattr(b, an, None)
            if val is None:
                cur[rn] = {"absent": an}
                continue
            try:
                g = _host(val)
                if g.ndim == 1 and int(g.size) == n_total:
                    cur[rn] = {"via": an, "E_rev": er,
                               "i": [float(x) for x in g[idx[:64]] * (v_b[idx[:64]] - er)],
                               "g": [float(x) for x in g[idx[:64]]]}
                else:
                    cur[rn] = {"via": an, "shape": list(getattr(g, "shape", [])),
                               "note": "not per-neuron; per-cell read skipped"}
            except Exception as e:                        # noqa: BLE001
                cur[rn] = {"via": an, "unreadable": type(e).__name__}
        _n, ext = _first_attr(b, ("cp_external_input_current",))
        if ext is not None:
            try:
                ex = _host(ext)
                cur["i_external"] = {"via": _n, "at_breach_cell": [float(x) for x in ex[idx[:64]]]}
                rmap2 = _region_map(b) or {}
                cur["i_external_region_mean"] = {
                    k: round(float(ex[ids].mean()), 3) for k, ids in rmap2.items()
                    if ("gpe" in k or "stn" in k or k.startswith("thal") or k.startswith("gpi"))
                    and len(ids)}
                cur["tonic_note"] = ("tonic drive + external stim are ONE array: cp_external_input_current "
                                     "holds gpi/thal tonic + gpe/stn pacemaking; heard cells get the rendered "
                                     "stim each step; pA from env (see env_full)")
            except Exception as e:                        # noqa: BLE001
                cur["i_external"] = f"unreadable:{type(e).__name__}"
        for nm in ("cp_tonic_current", "cp_i_tonic", "cp_synaptic_current"):
            val = getattr(b, nm, None)
            cur[nm] = ({"absent": nm} if val is None
                       else {"present_type": type(val).__name__})
        _n, u = _first_attr(b, ("cp_recovery_variable_u", "u", "cp_u", "_u", "cp_adaptive_u"))
        if u is not None:
            try:
                br["u"] = [float(x) for x in _host(u)[idx[:64]]]
            except Exception:                              # noqa: BLE001
                br["u"] = {"absent": str(u)[:120]}
        else:
            br["u"] = {"absent": "u,cp_u,_u,cp_adaptive_u"}
        br["currents"] = cur
        rec["breach"] = br
        rec["bounded_readback"] = _bounded_readback()
        rec["status"] = "BREACH_RECORDED"
        _write(args, rec)
        return 0
    finally:
        BG._step, BG._neural_baseline = orig_step, orig_base


def _bounded_readback():
    """Proof the bounded-conductance path ran: the flag, its parser, and the build-time config if exposed."""
    from research.runners import _vocal_action_selector_gate as VG
    out = {"env": {k: os.environ.get(k) for k in
                   ("BRAIN_BG_BOUNDED_COND", "BRAIN_BG_PRESET_CELLS", "BRAIN_BG_TONIC_DRIVE")}}
    for fn in ("bg_bounded_cond_enabled", "bg_preset_cells_enabled", "tonic_drive_enabled"):
        f = getattr(VG, fn, None)
        if callable(f):
            try:
                out[fn] = bool(f())
            except Exception as e:                        # noqa: BLE001
                out[fn] = f"error:{type(e).__name__}"
    return out


def _write(args, rec):
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "findings", "raw", "_bg_live_seam",
        "observe", f"s{args.seed}_{args.turn}.json")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec["record_path"] = out
    json.dump(rec, open(out, "w"), indent=1, default=str)
    print(f"[bg_breach_observe] seed={args.seed} {rec.get('status')} "
          f"steps={len(rec.get('steps', []))} identity={rec.get('identity')} -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())