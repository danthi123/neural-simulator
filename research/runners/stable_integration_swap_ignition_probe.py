"""THOUGHT-SWAP WORKSPACE IGNITION, OLD vs STABLE INTEGRATOR (research/stable-recal).

The stable-integration attribution (research/findings/2026-10-26-stable-integration-dev7.md) shows the GNW thought-swap
workspace (`_gnw_neural_swap_intention_derisk.build`, the substrate of webapp/gnw_thought_swap.py and of the #85
swap-drives lead) decides whether a topic swap ignites, in both directions: on the old integrator it ignites, on the
stable set (SIM_STABLE_INTEGRATION=1) it does not. Its ignition criterion is inherited from Rung-2
(`_gnw_rung2_competitive_access_derisk._ignited`: a pattern is ignited iff its late-window private-core rate >=
IGNITE_FRAC x SOLO_PLATEAU = 0.5 x 1/3), i.e. half the period-3 limit cycle. This probe measures the organ ALONE, exactly
as the organ reads it, on both integrators, so its re-calibration is read off measured curves, never off replies:

  --curve       ignition curve: pattern 0 driven alone at each drive (`_drive`, DRIVE_STEPS) on the organ's tonic
                thal, then `_read_private_rates` (READ_FREE_STEPS free, late third) and a long hold read
                (REIGNITE_HOLD free steps: does the ignited state sustain on its own loop?), per integrator and per
                recurrent weight; with the loop-lesion (w_rec 0) and no-drive controls; plus the workspace membranes
                over the whole read (out of range = the census's rule: v < -110 mV, v > vpeak + 5 mV, non-finite).
  --inhibition  the same read across the divisive-normalization inhibition norm_pool -> workspace (production 16).
  --sustain     long free-run read (1000 steps): does the driven pattern self-sustain, is the ignited state steady
                enough for the organ's 15-step late-window read, does the undriven workspace stay quiescent (no
                spontaneous ignition), across the recurrent weight and the tonic thalamic support.
  --primitives  the vacancy-gate primitive (empty admits, occupied vetoes, the held content survives its own gate) and
                the mismatch-detector primitive (the organ's own `--calibrate` bars) at a calibration, sweeping one
                constant.
  --collapse    the eviction effector alone: a held pattern under a fixed per-spike utilization boost for
                EVICT_STEPS, so the boost a swap needs is measured on each integrator.
  --organ       the organ's own GO gate (`evaluate_seed`: neural-triggered swap, non-salient hold, match hold,
                trigger lesion, timing, re-ignite, two-swap reversibility, no host reset, determinism) at a named
                calibration on a named integrator (dev seeds only).

No sim/ or webapp/ edit; module constants are set for a measurement and restored after. Dev seeds only (gate seeds
refused).

  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' .venv/bin/python -m research.runners.stable_integration_swap_ignition_probe \
      --curve --seed 7 --out research/findings/raw/_stable_recal/swap_ignition_curve_s7.json
"""
from __future__ import annotations

import argparse
import contextlib
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
V_LOW_MV = -110.0                 # the membrane census's out-of-range rule (research/runners/membrane_census.py)
VPEAK_MARGIN_MV = 5.0
CURVE_DRIVES = (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0)
WEIGHTS = (34.0, 5.0, 10.0, 15.0, 20.0, 25.0, 45.0, 60.0)


@contextlib.contextmanager
def _integrator(arm):
    saved = os.environ.get(ENV)
    if arm == "on":
        os.environ[ENV] = "1"
    else:
        os.environ.pop(ENV, None)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = saved


@contextlib.contextmanager
def _module_constants(mod, **kv):
    """Set module constants for one measurement and restore them after (the organ reads them as module globals)."""
    before = {k: getattr(mod, k) for k in kv}
    for k, v in kv.items():
        setattr(mod, k, v)
    try:
        yield before
    finally:
        for k, v in before.items():
            setattr(mod, k, v)


def _build(S_mod, seed, arm, w_rec, **kw):
    np.random.seed(int(seed))                  # the production organ seeds its private RNG timeline the same way
    with _integrator(arm):
        S = S_mod.build(seed=int(seed), w_rec=float(w_rec), **kw)
    b = S["bridge"]
    if bool(b.stable_integration_requested) != (arm == "on"):
        raise RuntimeError("bridge read the wrong switch value")
    cfg = b.core_config
    S["readback"] = {"requested": bool(b.stable_integration_requested),
                     "bounded": bool(cfg.enable_bounded_conductance_integration),
                     "substeps": int(cfg.izh_membrane_substeps), "rule": cfg.izh_substep_refractory_rule}
    return S


def _read_free_steps():
    """READ_FREE_STEPS of the module whose `_read_private_rates` the organ uses (the active-overwrite runner)."""
    from research.runners import _gnw_active_overwrite_derisk as AO
    return int(AO.READ_FREE_STEPS)


class _Watch:
    """Membrane watch over a set of cells for one read: out-of-range cells (census rule), min/max v."""

    def __init__(self, S, idx_host):
        from sim.backend import to_host
        self.to_host = to_host
        self.b = S["bridge"]
        self.idx = S["xp"].asarray(np.asarray(idx_host, dtype=np.int64))
        self.vpeak = to_host(self.b.cp_izh_vpeak[self.idx]).astype(np.float64)
        self.oor = np.zeros(len(idx_host), dtype=bool)
        self.vmin, self.vmax = np.inf, -np.inf

    def __call__(self):
        v = self.to_host(self.b.cp_membrane_potential_v[self.idx]).astype(np.float64)
        self.vmin = min(self.vmin, float(np.nanmin(v)) if np.isfinite(v).any() else -np.inf)
        self.vmax = max(self.vmax, float(np.nanmax(v)) if np.isfinite(v).any() else np.inf)
        self.oor |= (v < V_LOW_MV) | (v > self.vpeak + VPEAK_MARGIN_MV) | ~np.isfinite(v)

    def summary(self):
        return {"n_oor_cells": int(self.oor.sum()), "n_cells": int(self.oor.size), "v_min": round(self.vmin, 2),
                "v_max": round(self.vmax, 2)}


def _free_rates(S_mod, S, std, n_free, watch, late_v=None):
    """`_read_private_rates` with a membrane watch: free-run n_free steps (tonic thal on, no pattern drive), late third
    per-pattern private-core rate."""
    from sim.backend import to_host
    b, xp = S["bridge"], S["xp"]
    late_start = n_free - max(1, n_free // 3)
    counts = [0] * len(S["privates"])
    for t in range(n_free):
        S_mod._ws_step(b, xp, S["thal"], S_mod.THAL_TONIC_PA, std)
        watch()
        if t >= late_start:
            for i, p in enumerate(S["privates"]):
                counts[i] += int(to_host(b.cp_firing_states[p].astype(xp.float64).sum()))
            if late_v is not None:
                v0 = to_host(b.cp_membrane_potential_v[S["privates"][0]]).astype(np.float64)
                late_v.append((float(v0.mean()), float((v0 > -20.0).mean())))
    return [counts[i] / float((n_free - late_start) * int(p.shape[0])) for i, p in enumerate(S["privates"])]


def read_ignition(S_mod, S, std, drives, hold_steps=None):
    """The organ's own ignition read (restore the settled snapshot, drive the patterns for DRIVE_STEPS, free-run
    READ_FREE_STEPS, late-window private rates) plus a long hold read and the membrane watch."""
    b, xp = S["bridge"], S["xp"]
    S_mod._full_restore(b, S["snap"])
    std.reset()
    watch = _Watch(S, S["ws_used"])
    dm = [(S["patterns"][k], float(d)) for k, d in enumerate(drives) if d > 0.0]
    for _ in range(S_mod.DRIVE_STEPS):
        S_mod._ws_step(b, xp, S["thal"], S_mod.THAL_TONIC_PA, std, drive_map=dm)
        watch()
    late_v = []
    read = _free_rates(S_mod, S, std, _read_free_steps(), watch, late_v)
    hold = _free_rates(S_mod, S, std, int(hold_steps if hold_steps is not None else S_mod.REIGNITE_HOLD), watch)
    out = {"drives": [float(d) for d in drives], "read_rates": [round(r, 4) for r in read],
           "hold_rates": [round(r, 4) for r in hold], "slot0_late_mean_v": round(float(np.mean([x[0] for x in late_v])), 2),
           "slot0_late_frac_above_m20mV": round(float(np.mean([x[1] for x in late_v])), 4)}
    out.update(watch.summary())
    return out


def run_curve(seed, weights=WEIGHTS, drives=CURVE_DRIVES, arms=("off", "on"), norm_ws=None, thal=None):
    from research.runners import _gnw_neural_swap_intention_derisk as SM
    from research.runners._gnw_recurrence_weaken_swap_derisk import MultiLoopSTD
    from tools.lab import attributable_to, lever
    out = {"runner": "research/runners/stable_integration_swap_ignition_probe.py --curve", "seed": int(seed),
           "ignite_threshold_production": round(SM.IGNITE_FRAC * SM.SOLO_PLATEAU, 4),
           "solo_plateau_production": round(SM.SOLO_PLATEAU, 4), "drive_steps": SM.DRIVE_STEPS,
           "read_free_steps": _read_free_steps(), "hold_steps": SM.REIGNITE_HOLD, "drives": list(drives),
           "weights": list(weights), "production_w_rec": SM.W_REC, "production_norm_ws_w": SM.NORM_WS_W,
           "production_establish_pa": SM.ESTABLISH_PA, "production_salient_pa": SM.SALIENT_PA,
           "thal_tonic_pa": SM.THAL_TONIC_PA, "norm_ws_w": (SM.NORM_WS_W if norm_ws is None else float(norm_ws)),
           "v_low_mV": V_LOW_MV, "vpeak_margin_mV": VPEAK_MARGIN_MV, "arms": {}}
    nws = SM.NORM_WS_W if norm_ws is None else float(norm_ws)
    thal_pa = SM.THAL_TONIC_PA if thal is None else float(thal)
    out["thal_tonic_pa"] = thal_pa
    with _module_constants(SM, NORM_WS_W=nws, THAL_TONIC_PA=thal_pa):
        for arm in arms:
            rows = []
            for w in weights:
                S = _build(SM, seed, arm, w)
                std = MultiLoopSTD(S["bridge"], S["xp"], S["ws_used"], S["patterns_host"])
                curve = [dict(drive=d, **read_ignition(SM, S, std, [d, 0.0, 0.0])) for d in drives]
                none = read_ignition(SM, S, std, [0.0, 0.0, 0.0])
                SL = _build(SM, seed, arm, 0.0)
                stdl = MultiLoopSTD(SL["bridge"], SL["xp"], SL["ws_used"], SL["patterns_host"])
                lesion = read_ignition(SM, SL, stdl, [max(drives), 0.0, 0.0])
                top = max(curve, key=lambda c: c["read_rates"][0])
                row = {"w_rec": float(w), "readback": S["readback"], "curve": curve, "none": none,
                       "lesion_at_max_drive": lesion,
                       "attributable_to_loops_vs_lesion": attributable_to(
                           "%s w=%g top read rate vs loop lesion" % (arm, w), top["read_rates"][0],
                           lesion["read_rates"][0]),
                       "attributable_to_drive_vs_none": attributable_to(
                           "%s w=%g top read rate vs no drive" % (arm, w), top["read_rates"][0],
                           none["read_rates"][0])}
                rows.append(row)
                print("[swap curve] %s nws=%g thal=%g w=%5.1f %s | none %.3f lesion %.3f" % (
                    arm, nws, thal_pa, w, " ".join("%g:%.3f/%.3f(oor%d)" % (c["drive"], c["read_rates"][0], c["hold_rates"][0],
                                                               c["n_oor_cells"]) for c in curve),
                    none["read_rates"][0], lesion["read_rates"][0]), flush=True)
            out["arms"][arm] = rows
    if "off" in out["arms"] and "on" in out["arms"]:
        lever("membrane integrator (bounded, substeps)",
              (out["arms"]["off"][0]["readback"]["bounded"], out["arms"]["off"][0]["readback"]["substeps"]),
              (out["arms"]["on"][0]["readback"]["bounded"], out["arms"]["on"][0]["readback"]["substeps"]))
    return out


def run_collapse(seed, calib, arms=("off", "on"), boosts=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12,
                                                           0.16)):
    """The eviction effector alone (the recurrence-weaken swap's STD on the held pattern's own loop) at calibration
    `calib`: hold pattern 0 (ESTABLISH_PA x DRIVE_STEPS), then a FIXED per-spike utilization boost on every loop for
    EVICT_STEPS (no proposal, no mm), then the organ's read: does pattern 0 stay ignited? The boost at which the held
    pattern collapses is the knee the mismatch read-out (eff_boost = min(MAX_BOOST, BOOST_GAIN x mm rate)) has to clear
    for a salient mismatch and stay under for a non-salient one."""
    from research.runners import _gnw_neural_swap_intention_derisk as SM
    from research.runners._gnw_recurrence_weaken_swap_derisk import MultiLoopSTD
    out = {"runner": "research/runners/stable_integration_swap_ignition_probe.py --collapse", "seed": int(seed),
           "calibration": calib, "evict_steps": SM.EVICT_STEPS, "arms": {}}
    for arm in arms:
        np.random.seed(int(seed))
        with _integrator(arm), organ_calibration(calib):
            S = SM.build(seed=int(seed), w_rec=float(calib.get("w_rec", SM.W_REC)))
            b, xp = S["bridge"], S["xp"]
            std = MultiLoopSTD(b, xp, S["ws_used"], S["patterns_host"])
            rows = []
            for bst in boosts:
                SM._full_restore(b, S["snap"])
                std.reset()
                SM._drive(b, xp, S["thal"], SM.THAL_TONIC_PA, std, [(S["patterns"][0], SM.ESTABLISH_PA)],
                          n=SM.DRIVE_STEPS)
                pre = SM._read_private_rates(b, xp, S["thal"], SM.THAL_TONIC_PA, S["privates"], std)
                watch = _Watch(S, S["ws_used"])
                for kk in range(SM.N_PATTERNS):
                    std.set_boost(kk, float(bst))
                xmin = 1.0
                for _ in range(int(SM.EVICT_STEPS)):
                    SM._ws_step(b, xp, S["thal"], SM.THAL_TONIC_PA, std)
                    watch()
                    xmin = min(xmin, std.x_mean(0))
                std.clear_boost()
                post = _free_rates(SM, S, std, _read_free_steps(), watch)
                rows.append({"boost": float(bst), "pre_rates": [round(r, 4) for r in pre],
                             "post_rates": [round(r, 4) for r in post], "xA_min": round(xmin, 4),
                             "held": bool(SM._ignited(post[0])), **watch.summary()})
            cfg = b.core_config
            out["arms"][arm] = {"readback": {"requested": bool(b.stable_integration_requested),
                                             "bounded": bool(cfg.enable_bounded_conductance_integration),
                                             "substeps": int(cfg.izh_membrane_substeps)}, "rows": rows}
        print("[swap collapse] %s s%d %s" % (arm, seed, " ".join("%g:%.3f(x%.2f)" % (r["boost"], r["post_rates"][0],
                                                                                   r["xA_min"]) for r in rows)),
              flush=True)
    return out


SUSTAIN_WINDOWS = ((30, 45), (100, 200), (300, 400), (600, 700), (900, 1000))


def run_sustain(seed, weights, thals, drives, norm_ws=None, arms=("off", "on"), steps=1000):
    """Does the driven pattern SELF-SUSTAIN on its own loop, and does the undriven workspace stay quiescent? Pattern 0
    driven for DRIVE_STEPS at each drive (0 = the no-drive / spontaneous-ignition control), then `steps` free steps
    (the organ's longest no-drive stretch is the 320-step eviction window + the 45-step read + the 150-step hold read):
    per window the private-core rate of all three patterns, pattern 0's mean membrane and fraction above -20 mV (the
    depolarization-block read used for the bus), and the workspace's out-of-range cells over the whole run. Swept over
    the recurrent weight and the tonic thalamic support (THAL_TONIC_PA, production 700) at one normalization weight."""
    from sim.backend import to_host
    from research.runners import _gnw_neural_swap_intention_derisk as SM
    from research.runners._gnw_recurrence_weaken_swap_derisk import MultiLoopSTD
    nws = SM.NORM_WS_W if norm_ws is None else float(norm_ws)
    out = {"runner": "research/runners/stable_integration_swap_ignition_probe.py --sustain", "seed": int(seed),
           "norm_ws_w": nws, "weights": list(weights), "thal_tonic_pa": list(thals), "drives": list(drives),
           "steps": int(steps), "windows": [list(w) for w in SUSTAIN_WINDOWS], "drive_steps": SM.DRIVE_STEPS,
           "production": {"w_rec": SM.W_REC, "norm_ws_w": SM.NORM_WS_W, "thal_tonic_pa": SM.THAL_TONIC_PA},
           "rows": []}
    for arm in arms:
        for thal in thals:
            with _module_constants(SM, NORM_WS_W=nws, THAL_TONIC_PA=float(thal)):
                for w in weights:
                    S = _build(SM, seed, arm, w)
                    b, xp = S["bridge"], S["xp"]
                    std = MultiLoopSTD(b, xp, S["ws_used"], S["patterns_host"])
                    priv = [to_host(p) for p in S["privates"]]
                    for d in drives:
                        SM._full_restore(b, S["snap"])
                        std.reset()
                        watch = _Watch(S, S["ws_used"])
                        dm = [(S["patterns"][0], float(d))] if d > 0 else None
                        for _ in range(SM.DRIVE_STEPS):
                            SM._ws_step(b, xp, S["thal"], float(thal), std, drive_map=dm)
                            watch()
                        r = np.zeros((steps, 3))
                        v0 = np.zeros(steps)
                        hi0 = np.zeros(steps)
                        for t in range(steps):
                            SM._ws_step(b, xp, S["thal"], float(thal), std)
                            watch()
                            f = to_host(b.cp_firing_states).astype(bool)
                            vv = to_host(b.cp_membrane_potential_v)
                            r[t] = [f[p].mean() for p in priv]
                            v0[t] = vv[priv[0]].mean()
                            hi0[t] = (vv[priv[0]] > -20.0).mean()
                        win = [{"window": [a, e], "rates": [round(float(x), 4) for x in r[a:e].mean(0)],
                                "slot0_mean_v": round(float(v0[a:e].mean()), 2),
                                "slot0_frac_above_m20mV": round(float(hi0[a:e].mean()), 4)}
                               for a, e in SUSTAIN_WINDOWS if e <= steps]
                        row = {"arm": arm, "w_rec": float(w), "thal_tonic_pa": float(thal), "drive": float(d),
                               "readback": S["readback"], "windows": win, **watch.summary()}
                        out["rows"].append(row)
                        print("[swap sustain] %s nws=%g thal=%g w=%g d=%g %s oor%d" % (
                            arm, nws, thal, w, d, " ".join("[%d-%d]%.3f/%.3f/%.3f v%.0f" % (
                                x["window"][0], x["window"][1], x["rates"][0], x["rates"][1], x["rates"][2],
                                x["slot0_mean_v"]) for x in win), row["n_oor_cells"]), flush=True)
    return out


# calibration keys -> the swap module's constants the organ reads at call time (build + run_intention_swap)
ORGAN_CONSTS = {"establish_pa": "ESTABLISH_PA", "max_boost": "MAX_BOOST", "thal_tonic_pa": "THAL_TONIC_PA",
                "norm_ws_w": "NORM_WS_W", "w_ws_occ": "W_WS_OCC", "w_occ_gate": "W_OCC_GATE",
                "w_gate_ws": "W_GATE_WS", "boost_window": "BOOST_WINDOW", "timing_window": "TIMING_WINDOW"}
_INT_CONSTS = ("boost_window", "timing_window")
# calibration keys -> `build` keyword arguments (their defaults are bound at definition time)
ORGAN_BUILD_KW = {"w_pat_pred": "w_pat_pred", "w_pred_mm": "w_pred_mm"}


@contextlib.contextmanager
def organ_calibration(calib):
    """Apply a swap-workspace calibration for the duration of a measurement: the swap module's call-time constants,
    the Rung-2 ignition criterion (`SOLO_PLATEAU`, read by `_ignited` / `_margin`) and the swap module's import-time
    threshold copy (`IGNITE_THRESH`, the timing read), and the `build` keywords; everything restored after."""
    from research.runners import _gnw_neural_swap_intention_derisk as SM
    from research.runners import _gnw_rung2_competitive_access_derisk as R2
    sm_consts = {ORGAN_CONSTS[k]: (int(v) if k in _INT_CONSTS else float(v)) for k, v in calib.items()
                 if k in ORGAN_CONSTS}
    r2_consts = {}
    if "solo_plateau" in calib:
        r2_consts["SOLO_PLATEAU"] = float(calib["solo_plateau"])
        sm_consts["IGNITE_THRESH"] = R2.IGNITE_FRAC * float(calib["solo_plateau"])
    kw = {ORGAN_BUILD_KW[k]: float(v) for k, v in calib.items() if k in ORGAN_BUILD_KW}
    orig_build = SM.build

    def _build_cal(*a, **k):
        for key, val in kw.items():
            k.setdefault(key, val)
        return orig_build(*a, **k)

    with _module_constants(R2, **r2_consts), _module_constants(SM, build=_build_cal, **sm_consts):
        yield


def _gate_and_mm_primitives(SM, S, pa, nonsalient_pa):
    """The two primitives the swap is composed of, read exactly as their own calibrations read them:
    VACANCY GATE (`_gnw_neural_vacancy_gate_derisk.run_calibrate`, here at the swap's own proposal drive into gate_k):
      empty  = empty workspace + 45 steps of proposal into gate_1 -> the organ's read: pattern 1 must ignite alone;
      held   = pattern 0 established (ESTABLISH_PA x DRIVE_STEPS) + 45 steps of proposal into gate_1 -> pattern 1 must
               stay out and pattern 0 must hold;
      self   = pattern 0 established + 45 steps of proposal into gate_0 (the held content proposed again, a MATCH): a
               vetoed gate must leave the held pattern ignited (the swap's match-hold needs it).
    MISMATCH DETECTOR (`_gnw_neural_swap_intention_derisk.run_calibrate`): with pattern 0 held, the mm population rate
      for a salient mismatch (> 0.10), a salient match (< 0.02, the pred veto) and a non-salient mismatch (< 0.05)."""
    b, xp, thal = S["bridge"], S["xp"], S["thal"]
    T = SM.THAL_TONIC_PA

    def _est():
        SM._full_restore(b, S["snap"])
        SM._drive(b, xp, thal, T, None, [(S["patterns"][0], SM.ESTABLISH_PA)], n=SM.DRIVE_STEPS)

    def _gate_read(k, held):
        if held:
            _est()
        else:
            SM._full_restore(b, S["snap"])
        during = []
        for _ in range(45):
            SM._ws_step(b, xp, thal, T, None, drive_map=[(S["gate_slices"][k], float(pa))])
            during.append(SM._pop_rate(S, S["gate_slices"][k]))
        rates = SM._read_private_rates(b, xp, thal, T, S["privates"], None)
        return [round(float(r), 4) for r in rates], round(float(np.mean(during)), 4)

    def _mm(k, p):
        _est()
        rs = []
        for _ in range(45):
            SM._ws_step(b, xp, thal, T, None, drive_map=[(S["mm_slices"][k], float(p))])
            rs.append(SM._pop_rate(S, S["mm_all"]))
        return round(float(np.mean(rs[-45 // 2:])), 4)

    ign = SM._ignited
    empty, g_empty = _gate_read(1, False)
    held, g_held = _gate_read(1, True)
    selfm, g_self = _gate_read(0, True)
    mm = {"salient_mismatch": _mm(1, pa), "salient_match": _mm(0, pa), "nonsalient_mismatch": _mm(1, nonsalient_pa)}
    gate_ok = bool(ign(empty[1]) and sum(ign(x) for x in empty) == 1 and not ign(held[1]) and ign(held[0]))
    self_ok = bool(ign(selfm[0]) and sum(ign(x) for x in selfm) == 1)
    mm_ok = bool(mm["salient_mismatch"] > 0.10 and mm["salient_match"] < 0.02 and mm["nonsalient_mismatch"] < 0.05)
    return {"gate_empty_rates": empty, "gate_empty_gate_rate": g_empty, "gate_held_rates": held,
            "gate_held_gate_rate": g_held, "gate_self_rates": selfm, "gate_self_gate_rate": g_self,
            "gate_primitive_holds": gate_ok, "held_content_survives_own_gate": self_ok, "mm": mm,
            "mm_primitive_holds": mm_ok}


def run_primitives(seed, arm, calib, sweep_key, sweep_values):
    """The vacancy-gate and mismatch-detector primitives on integrator `arm` at calibration `calib`, sweeping one
    calibration key (an ORGAN_CONSTS / ORGAN_BUILD_KW key) over `sweep_values`."""
    from research.runners import _gnw_neural_swap_intention_derisk as SM
    out = {"runner": "research/runners/stable_integration_swap_ignition_probe.py --primitives", "seed": int(seed),
           "arm": arm, "calibration": calib, "sweep_key": sweep_key, "rows": []}
    for v in sweep_values:
        c = dict(calib)
        c[sweep_key] = v
        np.random.seed(int(seed))
        with _integrator(arm), organ_calibration(c):
            S = SM.build(seed=int(seed), w_rec=float(c.get("w_rec", SM.W_REC)))
            r = _gate_and_mm_primitives(SM, S, float(c.get("salient_pa", SM.SALIENT_PA)),
                                        float(c.get("nonsalient_pa", SM.NONSALIENT_PA)))
        r[sweep_key] = v
        out["rows"].append(r)
        print("[swap primitives] %s s%d %s=%g | empty %s (gate %.3f) held %s (gate %.3f) self %s (gate %.3f) | mm %s"
              " | gate_ok=%s self_ok=%s mm_ok=%s" % (arm, seed, sweep_key, v, r["gate_empty_rates"],
                                                    r["gate_empty_gate_rate"], r["gate_held_rates"],
                                                    r["gate_held_gate_rate"], r["gate_self_rates"],
                                                    r["gate_self_gate_rate"], r["mm"], r["gate_primitive_holds"],
                                                    r["held_content_survives_own_gate"], r["mm_primitive_holds"]),
              flush=True)
    return out


def run_organ(seed, arm, calib):
    """The organ's own GO gate (`evaluate_seed`) at calibration `calib` on integrator `arm`. calib keys: w_rec,
    solo_plateau, salient_pa, nonsalient_pa, boost_gain, evict_steps (evaluate_seed arguments) plus ORGAN_CONSTS and
    ORGAN_BUILD_KW (missing = production)."""
    from research.runners import _gnw_neural_swap_intention_derisk as SM
    np.random.seed(int(seed))
    with _integrator(arm), organ_calibration(calib):
        r = SM.evaluate_seed(int(seed), proposal_pa=float(calib.get("salient_pa", SM.SALIENT_PA)),
                             nonsalient_pa=float(calib.get("nonsalient_pa", SM.NONSALIENT_PA)),
                             boost_gain=float(calib.get("boost_gain", SM.BOOST_GAIN)),
                             evict_steps=int(calib.get("evict_steps", SM.EVICT_STEPS)),
                             w_rec=float(calib.get("w_rec", SM.W_REC)), verbose=True)
    return {"runner": "research/runners/stable_integration_swap_ignition_probe.py --organ", "seed": int(seed),
            "arm": arm, "calibration": calib, "result": r}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="off,on")
    ap.add_argument("--weights", default=",".join("%g" % w for w in WEIGHTS))
    ap.add_argument("--drives", default=",".join("%g" % d for d in CURVE_DRIVES))
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--inhibition", default="", help="comma list of norm_pool->workspace weights (curve per value)")
    ap.add_argument("--collapse", action="store_true", help="eviction-effector collapse knee at --organ calib")
    ap.add_argument("--organ", default="", help="JSON calibration dict for the organ's own GO gate")
    ap.add_argument("--sustain", action="store_true", help="long free-run sustain/baseline read (see run_sustain)")
    ap.add_argument("--thals", default="700", help="comma list of THAL_TONIC_PA values for --sustain")
    ap.add_argument("--norm-ws", type=float, default=None, help="norm_pool->workspace weight for --sustain")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--primitives", default="", help="'key=v1,v2,...': gate + mismatch primitives at --organ calib")
    a = ap.parse_args()
    if a.seed in GATE_SEEDS:
        sys.exit("GATE seed %d refused: dev-seed diagnostic only" % a.seed)
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    weights = [float(x) for x in a.weights.split(",") if x.strip()]
    drives = [float(x) for x in a.drives.split(",") if x.strip()]
    if a.primitives:
        key, vals = a.primitives.split("=", 1)
        calib = json.loads(a.organ) if a.organ else {}
        d = {"arms": {arm: run_primitives(a.seed, arm, calib, key, [float(x) for x in vals.split(",")])
                      for arm in arms}}
    elif a.sustain:
        d = run_sustain(a.seed, weights, [float(x) for x in a.thals.split(",") if x.strip()], drives,
                        norm_ws=a.norm_ws, arms=arms, steps=a.steps)
    elif a.collapse:
        d = run_collapse(a.seed, json.loads(a.organ) if a.organ else {}, arms=arms)
    elif a.organ:
        d = {"arms": {arm: run_organ(a.seed, arm, json.loads(a.organ)) for arm in arms}}
    elif a.inhibition:
        d = {"runner": "research/runners/stable_integration_swap_ignition_probe.py --inhibition", "seed": a.seed,
             "by_norm_ws_w": {x: run_curve(a.seed, weights, drives, arms, norm_ws=float(x))
                              for x in a.inhibition.split(",") if x.strip()}}
    elif a.curve:
        d = run_curve(a.seed, weights, drives, arms, norm_ws=a.norm_ws,
                      thal=(float(a.thals) if a.thals != "700" else None))
    else:
        ap.error("choose --curve, --inhibition, --sustain, --primitives, --collapse or --organ")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(d, f, indent=1, default=str)
    print("[swap ignition] wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
