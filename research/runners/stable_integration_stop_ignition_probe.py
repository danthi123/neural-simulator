"""GLOBAL-STOP WORKSPACE IGNITION, OLD vs STABLE INTEGRATOR (research/stable-recal).

The stop-lead attribution (research/findings/2026-10-27-stable-recal-dev78.md, "Who decides the stop lead") shows that
the global-stop distributed-overwrite workspace (`_gnw_distributed_overwrite_workspace_derisk.build_overwrite_bridge`,
the substrate of webapp/gnw_global_stop.py and of the "Setting the held thread aside" lead) decides that lead in both
directions, and only clears while 250 to 270 of its cells are out of the valid membrane range. Its ignition criterion
is Rung-2's (`_gnw_rung2_competitive_access_derisk._ignited`: a pattern is ignited iff its late-window private-core
rate >= IGNITE_FRAC x SOLO_PLATEAU = 0.5 x 1/3) and its conflict sensor's reference is half of the same plateau
(MARGIN_REF = SOLO_PLATEAU x 0.5). This probe measures the organ ALONE, exactly as the organ reads it, on both
integrators, so its re-calibration is read off measured curves, never off replies:

  --curve       ignition curve: pattern 0 driven alone at each drive (`_drive`, DRIVE_STEPS) on the organ's tonic
                thal, then the organ's read (`_read_private_rates`: READ_FREE_STEPS free, late third) and a hold read
                (POST_FREE_STEPS more free steps); the same with all three patterns driven together (the conflict
                state the stop clears); per integrator and recurrent weight, with the loop-lesion (w_rec 0) and
                no-drive controls and the workspace membranes over the whole read (out of range = the census's rule:
                v < -110 mV, v > vpeak + 5 mV, non-finite).
  --inhibition  the same curve across the divisive-normalization weight norm_pool -> workspace (production 16).
  --sustain     long free-run read (default 1000 steps) after pattern 0 (or all three, --co) is driven: does the
                ignited state self-sustain, how reliably does the organ's 15-step late window read it, does the undriven
                workspace stay quiescent (drive 0 = the spontaneous-ignition control); across weight and thal support.
  --stopwin     the stop effector alone: co-ignite the held contents (STRONG drive, DRIVE_STEPS), read, apply a FIXED
                per-spike depression boost on the shared recurrence for PULSE_DURATION steps, read again, intact and
                with the normalization lesion. The organ's rule puts the full-conflict boost inside the window where
                the intact workspace clears and the lesioned one does not (production: ~0.13 inside [0.09, 0.15]).
  --margins     the conflict sensor's two levels (confident single-content margin, co-ignited conflict margin),
                repeated, at a calibration: the organ's MARGIN_REF is "halfway between" them.
  --organ       the organ's own GO gate (`evaluate_seed`: global stop, selectivity, divisive-norm and STD load-bearing,
                scramble, conflict-off, continuous stop without restore, byte-identical substrate, determinism) at a
                named calibration on a named integrator.

A calibration is a JSON dict: build keys w_rec, norm_ws_w, ws_norm_w, thal_tonic_pa, thal_ws_w; module keys strong_pa
(the per-pattern establishment drive, production 2 x IGNITE_PA = 5000), solo_plateau (the stop organ's ignited
plateau, production 1/3; criterion IGNITE_FRAC x plateau), margin_ref (default 0.5 x solo_plateau, the organ's rule);
evaluate_seed keys boost_scale, pulse_duration. Missing keys = production. The calibration is applied by patching the
module constants for one measurement (Rung-2's SOLO_PLATEAU only inside this process, for the stop organ's reads) and
restoring them after. No sim/ or webapp/ edit. Dev seeds only (gate seeds refused).

  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' .venv/bin/python -m research.runners.stable_integration_stop_ignition_probe \
      --curve --seed 7 --out research/findings/raw/_stable_recal/stop_ignition_curve_s7.json
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

from research.runners.stable_integration_swap_ignition_probe import (  # noqa: E402  (shared instruments)
    _integrator, _module_constants, _Watch, V_LOW_MV, VPEAK_MARGIN_MV, GATE_SEEDS, READ15,
)

CURVE_DRIVES = (250.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 8000.0)
WEIGHTS = (34.0, 5.0, 10.0, 15.0, 20.0, 25.0, 45.0, 60.0)
STOP_BOOSTS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.15, 0.18, 0.22, 0.27, 0.33, 0.40, 0.50)
BUILD_KEYS = {"w_rec": "w_rec", "norm_ws_w": "norm_ws_w", "ws_norm_w": "ws_norm_w", "thal_tonic_pa": "thal_tonic_pA",
              "thal_ws_w": "thal_ws_w"}
EVAL_KEYS = ("boost_scale", "pulse_duration", "boost_gain")


def _dm():
    from research.runners import _gnw_distributed_overwrite_workspace_derisk as DM
    return DM


def build_kw(calib):
    return {BUILD_KEYS[k]: float(v) for k, v in calib.items() if k in BUILD_KEYS}


def thal_of(calib):
    return float(calib.get("thal_tonic_pa", _dm().THAL_TONIC_PA))


@contextlib.contextmanager
def stop_calibration(calib):
    """Apply a stop-workspace calibration's call-time constants for one measurement: the stop module's STRONG_PA and
    MARGIN_REF (module globals read at call time) and Rung-2's SOLO_PLATEAU (read by `_ignited`, which the stop
    module's `_margin` / `_verdict_label` call); all restored after."""
    from research.runners import _gnw_rung2_competitive_access_derisk as R2
    DM = _dm()
    dm_consts, r2_consts = {}, {}
    if "strong_pa" in calib:
        dm_consts["STRONG_PA"] = float(calib["strong_pa"])
    if "solo_plateau" in calib:
        r2_consts["SOLO_PLATEAU"] = float(calib["solo_plateau"])
        dm_consts["MARGIN_REF"] = 0.5 * float(calib["solo_plateau"])
    if "margin_ref" in calib:
        dm_consts["MARGIN_REF"] = float(calib["margin_ref"])
    with _module_constants(R2, **r2_consts), _module_constants(DM, **dm_consts):
        yield


def _build(seed, arm, calib, **over):
    """The organ as production builds it: np.random seeded with the organ seed (webapp/gnw_global_stop.py seeds its
    private RNG timeline the same way before the first build), then `build_overwrite_bridge`."""
    DM = _dm()
    kw = build_kw(calib)
    kw.update(over)
    np.random.seed(int(seed))
    with _integrator(arm):
        b, xp, pats, privs, thal, ws_used, snap, handles = DM.build_overwrite_bridge(seed=int(seed), **kw)
    if bool(b.stable_integration_requested) != (arm == "on"):
        raise RuntimeError("bridge read the wrong switch value")
    cfg = b.core_config
    return {"bridge": b, "xp": xp, "patterns": pats, "privates": privs, "thal": thal, "ws_used": ws_used, "snap": snap,
            "handles": handles,
            "readback": {"requested": bool(b.stable_integration_requested),
                         "bounded": bool(cfg.enable_bounded_conductance_integration),
                         "substeps": int(cfg.izh_membrane_substeps), "rule": cfg.izh_substep_refractory_rule}}


def _free_rates(S, std, thal_pa, n_free, watch, late_v=None):
    """`_read_private_rates` with a membrane watch: free-run n_free steps (tonic thal on, no pattern drive), late third
    per-pattern private-core rate."""
    from sim.backend import to_host
    DM = _dm()
    b, xp = S["bridge"], S["xp"]
    late_start = n_free - max(1, n_free // 3)
    counts = [0] * len(S["privates"])
    for t in range(n_free):
        DM._ws_step(b, xp, S["thal"], thal_pa, std)
        watch()
        if t >= late_start:
            for i, p in enumerate(S["privates"]):
                counts[i] += int(to_host(b.cp_firing_states[p].astype(xp.float64).sum()))
            if late_v is not None:
                v0 = to_host(b.cp_membrane_potential_v[S["privates"][0]]).astype(np.float64)
                late_v.append((float(v0.mean()), float((v0 > -20.0).mean())))
    return [counts[i] / float((n_free - late_start) * int(p.shape[0])) for i, p in enumerate(S["privates"])]


def _n_ign(rates):
    DM = _dm()
    return int(sum(1 for r in rates if DM._ignited(r)))


def read_ignition(S, std, drives, thal_pa):
    """The organ's own ignition read (restore the settled snapshot, drive for DRIVE_STEPS, READ_FREE_STEPS free, late
    third private-core rates) plus a POST_FREE_STEPS hold read and the membrane watch."""
    DM = _dm()
    b, xp = S["bridge"], S["xp"]
    DM._full_restore(b, S["snap"])
    std.reset()
    watch = _Watch(S, S["ws_used"])
    dm = [(S["patterns"][k], float(d)) for k, d in enumerate(drives) if d > 0.0]
    for _ in range(DM.DRIVE_STEPS):
        DM._ws_step(b, xp, S["thal"], thal_pa, std, drive_map=dm)
        watch()
    late_v = []
    read = _free_rates(S, std, thal_pa, DM.READ_FREE_STEPS, watch, late_v)
    hold = _free_rates(S, std, thal_pa, DM.POST_FREE_STEPS, watch)
    out = {"drives": [float(d) for d in drives], "read_rates": [round(r, 4) for r in read],
           "hold_rates": [round(r, 4) for r in hold], "n_ignited_read": _n_ign(read), "n_ignited_hold": _n_ign(hold),
           "slot0_late_mean_v": round(float(np.mean([x[0] for x in late_v])), 2),
           "slot0_late_frac_above_m20mV": round(float(np.mean([x[1] for x in late_v])), 4)}
    out.update(watch.summary())
    return out


def run_curve(seed, calib, weights=WEIGHTS, drives=CURVE_DRIVES, arms=("off", "on")):
    from tools.lab import attributable_to, lever
    from research.runners import _gnw_rung2_competitive_access_derisk as R2
    DM = _dm()
    thal_pa = thal_of(calib)
    out = {"runner": "research/runners/stable_integration_stop_ignition_probe.py --curve", "seed": int(seed),
           "calibration": calib, "drive_steps": DM.DRIVE_STEPS, "read_free_steps": DM.READ_FREE_STEPS,
           "hold_steps": DM.POST_FREE_STEPS, "drives": list(drives), "weights": list(weights),
           "production": {"w_rec": DM.W_REC, "norm_ws_w": DM.NORM_WS_W, "ws_norm_w": DM.WS_NORM_W,
                          "thal_tonic_pa": DM.THAL_TONIC_PA, "thal_ws_w": DM.THAL_WS_W, "strong_pa": DM.STRONG_PA,
                          "ignite_pa": DM.IGNITE_PA, "solo_plateau": R2.SOLO_PLATEAU,
                          "ignite_threshold": round(R2.IGNITE_FRAC * R2.SOLO_PLATEAU, 4),
                          "margin_ref": round(DM.MARGIN_REF, 4)},
           "v_low_mV": V_LOW_MV, "vpeak_margin_mV": VPEAK_MARGIN_MV, "arms": {}}
    with stop_calibration(calib):
        out["ignite_threshold_used"] = round(R2.IGNITE_FRAC * R2.SOLO_PLATEAU, 4)
        for arm in arms:
            rows = []
            for w in weights:
                S = _build(seed, arm, calib, w_rec=float(w))
                std = DM.WorkspaceDepression(S["bridge"], S["xp"], S["ws_used"])
                curve = [dict(drive=d, **read_ignition(S, std, [d, 0.0, 0.0], thal_pa)) for d in drives]
                co = [dict(drive=d, **read_ignition(S, std, [d, d, d], thal_pa)) for d in drives]
                none = read_ignition(S, std, [0.0, 0.0, 0.0], thal_pa)
                SL = _build(seed, arm, calib, w_rec=0.0)
                stdl = DM.WorkspaceDepression(SL["bridge"], SL["xp"], SL["ws_used"])
                lesion = read_ignition(SL, stdl, [max(drives), 0.0, 0.0], thal_pa)
                top = max(curve, key=lambda c: c["read_rates"][0])
                rows.append({"w_rec": float(w), "readback": S["readback"], "curve": curve, "co_drive": co,
                             "none": none, "lesion_at_max_drive": lesion,
                             "attributable_to_loops_vs_lesion": attributable_to(
                                 "%s w=%g top read rate vs loop lesion" % (arm, w), top["read_rates"][0],
                                 lesion["read_rates"][0]),
                             "attributable_to_drive_vs_none": attributable_to(
                                 "%s w=%g top read rate vs no drive" % (arm, w), top["read_rates"][0],
                                 none["read_rates"][0])})
                print("[stop curve] %s s%d %s w=%5.1f solo %s | co %s | none %.3f lesion %.3f" % (
                    arm, seed, json.dumps(calib), w,
                    " ".join("%g:%.3f/%.3f(oor%d)" % (c["drive"], c["read_rates"][0], c["hold_rates"][0],
                                                      c["n_oor_cells"]) for c in curve),
                    " ".join("%g:n%d/%d(oor%d)" % (c["drive"], c["n_ignited_read"], c["n_ignited_hold"],
                                                    c["n_oor_cells"]) for c in co),
                    none["read_rates"][0], lesion["read_rates"][0]), flush=True)
            out["arms"][arm] = rows
    if "off" in out["arms"] and "on" in out["arms"]:
        lever("membrane integrator (bounded, substeps)",
              (out["arms"]["off"][0]["readback"]["bounded"], out["arms"]["off"][0]["readback"]["substeps"]),
              (out["arms"]["on"][0]["readback"]["bounded"], out["arms"]["on"][0]["readback"]["substeps"]))
    return out


SUSTAIN_WINDOWS = ((30, 45), (100, 200), (300, 400), (600, 700), (900, 1000))


def _read15(r, k=0, start=100):
    """Read reliability of pattern k's sustained state: its private rate in consecutive READ15-step windows from step
    `start` on (the organ reads a 15-step late window). plateau = the mean; frac_below_half = the fraction of windows
    under half of it (the ignition criterion is half the plateau)."""
    if r.shape[0] < start + READ15:
        return None
    wins = np.array([r[a:a + READ15, k].mean() for a in range(start, r.shape[0] - READ15 + 1, READ15)])
    plat = float(wins.mean())
    return {"plateau": round(plat, 4), "min": round(float(wins.min()), 4),
            "p10": round(float(np.percentile(wins, 10)), 4),
            "frac_below_half": round(float(np.mean(wins < plat / 2.0)), 4) if plat > 0 else None,
            "n_windows": int(len(wins))}


def run_sustain(seed, calib, weights, thals, drives, arms=("off", "on"), steps=1000, co=False):
    """Pattern 0 (or, with co=True, all three patterns: the conflict state) driven for DRIVE_STEPS at each drive (0 =
    the spontaneous-ignition control), then `steps` free steps: per window the private-core rates of all three
    patterns, pattern 0's mean membrane and fraction above -20 mV, the 15-step read reliability per pattern, and the
    workspace's out-of-range cells over the whole run. Swept over the recurrent weight and the tonic thal support."""
    from sim.backend import to_host
    DM = _dm()
    out = {"runner": "research/runners/stable_integration_stop_ignition_probe.py --sustain", "seed": int(seed),
           "calibration": calib, "weights": list(weights), "thal_tonic_pa": list(thals), "drives": list(drives),
           "co_drive": bool(co), "steps": int(steps), "windows": [list(w) for w in SUSTAIN_WINDOWS],
           "drive_steps": DM.DRIVE_STEPS, "rows": []}
    for arm in arms:
        for thal in thals:
            for w in weights:
                c = dict(calib, thal_tonic_pa=float(thal))
                S = _build(seed, arm, c, w_rec=float(w))
                b, xp = S["bridge"], S["xp"]
                std = DM.WorkspaceDepression(b, xp, S["ws_used"])
                priv = [to_host(p) for p in S["privates"]]
                for d in drives:
                    DM._full_restore(b, S["snap"])
                    std.reset()
                    watch = _Watch(S, S["ws_used"])
                    dm = ([(p, float(d)) for p in S["patterns"]] if co else [(S["patterns"][0], float(d))]) \
                        if d > 0 else None
                    for _ in range(DM.DRIVE_STEPS):
                        DM._ws_step(b, xp, S["thal"], float(thal), std, drive_map=dm)
                        watch()
                    r = np.zeros((steps, 3))
                    v0 = np.zeros(steps)
                    hi0 = np.zeros(steps)
                    for t in range(steps):
                        DM._ws_step(b, xp, S["thal"], float(thal), std)
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
                           "readback": S["readback"], "windows": win,
                           "read15": [_read15(r, k) for k in range(3)],
                           "first_45_late15": [round(float(x), 4) for x in r[30:45].mean(0)],
                           **watch.summary()}
                    out["rows"].append(row)
                    print("[stop sustain] %s s%d thal=%g w=%g d=%g co=%s %s oor%d | read15 p0 %s" % (
                        arm, seed, thal, w, d, co, " ".join("[%d-%d]%.3f/%.3f/%.3f v%.0f" % (
                            x["window"][0], x["window"][1], x["rates"][0], x["rates"][1], x["rates"][2],
                            x["slot0_mean_v"]) for x in win), row["n_oor_cells"], row["read15"][0]), flush=True)
    return out


def _stop_trial(S, std, thal_pa, contents, boost, pulse, strong_pa):
    """Co-ignite `contents` (strong_pa x DRIVE_STEPS), the organ's read (pre), a FIXED per-spike depression boost on
    the shared recurrence for `pulse` steps, then the organ's post read (POST_FREE_STEPS). Returns rates, n_ignited
    pre / post (the calibration's criterion), the pre margin and the membrane watch."""
    DM = _dm()
    b, xp = S["bridge"], S["xp"]
    DM._full_restore(b, S["snap"])
    std.reset()
    watch = _Watch(S, S["ws_used"])
    for _ in range(DM.DRIVE_STEPS):
        DM._ws_step(b, xp, S["thal"], thal_pa, std, drive_map=[(S["patterns"][c], float(strong_pa)) for c in contents])
        watch()
    pre = _free_rates(S, std, thal_pa, DM.READ_FREE_STEPS, watch)
    _w, margin, n_pre = DM._margin(pre)
    std.boost = float(boost)
    xmin = 1.0
    for _ in range(int(pulse)):
        DM._ws_step(b, xp, S["thal"], thal_pa, std)
        watch()
        xmin = min(xmin, float(std.x[std.ws_used].mean()))
    std.boost = 0.0
    post = _free_rates(S, std, thal_pa, DM.POST_FREE_STEPS, watch)
    _w2, _m2, n_post = DM._margin(post)
    return {"boost": float(boost), "pre_rates": [round(r, 4) for r in pre], "post_rates": [round(r, 4) for r in post],
            "n_pre": int(n_pre), "n_post": int(n_post), "pre_margin": round(float(margin), 4),
            "x_min_mean": round(xmin, 4), "cleared": bool(n_pre >= 2 and n_post == 0), **watch.summary()}


def run_stopwin(seed, calib, arms=("off", "on"), boosts=STOP_BOOSTS, contents_list=((0, 1, 2), (0, 1)), repeats=1):
    """The divisive-norm load-bearing window of the stop effector at calibration `calib`: per integrator, per held
    set (all three = the organ's gate; two = what production usually holds), the fixed boost at which the intact
    workspace clears and at which the normalization-lesioned one clears. With repeats > 1 every boost is run that many
    times in a row on the same bridge (each trial restores the settled snapshot; the OU noise stream runs on, so the
    trials differ only by noise), giving a clearing fraction per boost and the spread of the conflict margin the
    organ's sensor reads (pre_margin) at this drive."""
    DM = _dm()
    thal_pa = thal_of(calib)
    pulse = int(calib.get("pulse_duration", DM.PULSE_DURATION))
    out = {"runner": "research/runners/stable_integration_stop_ignition_probe.py --stopwin", "seed": int(seed),
           "calibration": calib, "boosts": list(boosts), "pulse_duration": pulse, "arms": {}}
    with stop_calibration(calib):
        strong = float(DM.STRONG_PA)
        out["strong_pa_used"] = strong
        out["margin_ref_used"] = float(DM.MARGIN_REF)
        for arm in arms:
            res = {}
            for label, lesion in (("intact", False), ("norm_lesion", True)):
                S = _build(seed, arm, calib, norm_lesion=lesion)
                std = DM.WorkspaceDepression(S["bridge"], S["xp"], S["ws_used"])
                for contents in contents_list:
                    key = "%s_n%d" % (label, len(contents))
                    rows = [_stop_trial(S, std, thal_pa, contents, bst, pulse, strong)
                            for bst in boosts for _rep in range(int(repeats))]
                    clears = [r["boost"] for r in rows if r["cleared"]]
                    per_boost = {}
                    for bst in boosts:
                        rr = [r for r in rows if r["boost"] == float(bst)]
                        per_boost["%g" % bst] = {
                            "n_trials": len(rr), "n_cleared": int(sum(r["cleared"] for r in rr)),
                            "n_pre_ge2": int(sum(r["n_pre"] >= 2 for r in rr)),
                            "n_post_hist": {str(k): int(sum(r["n_post"] == k for r in rr)) for k in range(4)},
                            "max_oor": int(max(r["n_oor_cells"] for r in rr))}
                    margins = [r["pre_margin"] for r in rows]
                    res[key] = {"rows": rows, "readback": S["readback"], "per_boost": per_boost,
                                "min_clearing_boost": (min(clears) if clears else None),
                                "clears_at": sorted(set(clears)),
                                "pre_margin": {"mean": round(float(np.mean(margins)), 4),
                                               "p90": round(float(np.percentile(margins, 90)), 4),
                                               "max": round(float(np.max(margins)), 4)}}
                    print("[stop window] %s s%d %s %s | %s | margin %s" % (
                        arm, seed, json.dumps(calib), key,
                        " ".join("%s:%d/%d%s" % (k, v["n_cleared"], v["n_trials"],
                                                 "" if v["n_pre_ge2"] == v["n_trials"] else "(pre<2:%d)" % (
                                                     v["n_trials"] - v["n_pre_ge2"]))
                                 for k, v in per_boost.items()), res[key]["pre_margin"]), flush=True)
            out["arms"][arm] = res
    return out


def run_margins(seed, calib, arms=("on",), repeats=16, contents_list=((0, 1, 2), (0, 1))):
    """The conflict sensor's two levels at calibration `calib`, read exactly as the organ reads them: the CONFIDENT
    single-content margin (`run_confident`: pattern 0 at STRONG_PA x DRIVE_STEPS, CONF_HOLD_STEPS hold, the 45-step
    read) and the CONFLICT margin (`run_conflict_stop`'s pre read: the held contents at STRONG_PA x DRIVE_STEPS, the
    45-step read), `repeats` trials each (the noise stream runs on between trials). The organ's MARGIN_REF is stated
    as "halfway between single-content and co-ignition": these are the two levels it sits between."""
    DM = _dm()
    thal_pa = thal_of(calib)
    out = {"runner": "research/runners/stable_integration_stop_ignition_probe.py --margins", "seed": int(seed),
           "calibration": calib, "repeats": int(repeats), "arms": {}}

    def _summ(xs):
        return {"mean": round(float(np.mean(xs)), 4), "sd": round(float(np.std(xs)), 4),
                "min": round(float(np.min(xs)), 4), "max": round(float(np.max(xs)), 4),
                "p10": round(float(np.percentile(xs, 10)), 4), "p90": round(float(np.percentile(xs, 90)), 4)}

    with stop_calibration(calib):
        out["strong_pa_used"] = float(DM.STRONG_PA)
        out["margin_ref_used"] = float(DM.MARGIN_REF)
        for arm in arms:
            S = _build(seed, arm, calib)
            b, xp = S["bridge"], S["xp"]
            std = DM.WorkspaceDepression(b, xp, S["ws_used"])
            conf = [DM.run_confident(b, xp, S["patterns"], S["privates"], S["thal"], S["snap"], std, thal_pa)
                    for _ in range(int(repeats))]
            res = {"confident": {"margin": _summ([c["margin"] for c in conf]),
                                 "n_ignited_hist": {str(k): int(sum(c["n_ignited"] == k for c in conf))
                                                    for k in range(4)},
                                 "winner_is_target": int(sum(c["winner"] == 0 for c in conf)),
                                 "rates": [c["rates"] for c in conf]}}
            for contents in contents_list:
                rows = []
                for _ in range(int(repeats)):
                    DM._full_restore(b, S["snap"])
                    std.reset()
                    DM._drive(b, xp, S["thal"], thal_pa, std, [(S["patterns"][c], DM.STRONG_PA) for c in contents])
                    pre = DM._read_private_rates(b, xp, S["thal"], thal_pa, S["privates"], std)
                    _w, m, n = DM._margin(pre)
                    rows.append({"rates": [round(r, 4) for r in pre], "margin": round(float(m), 4), "n": int(n)})
                res["conflict_n%d" % len(contents)] = {
                    "margin": _summ([r["margin"] for r in rows]),
                    "n_ignited_hist": {str(k): int(sum(r["n"] == k for r in rows)) for k in range(4)}, "rows": rows}
            out["arms"][arm] = res
            print("[stop margins] %s s%d %s | confident %s n%s | %s" % (
                arm, seed, json.dumps(calib), res["confident"]["margin"], res["confident"]["n_ignited_hist"],
                " | ".join("%s %s n%s" % (k, v["margin"], v["n_ignited_hist"]) for k, v in res.items()
                           if k.startswith("conflict"))), flush=True)
    return out


def run_organ(seed, arm, calib):
    """The organ's own GO gate (`evaluate_seed`) at calibration `calib` on integrator `arm`, plus the membrane read of
    its global-stop trial (the workspace's out-of-range cells during the conflict + stop, with the organ's own
    boost)."""
    DM = _dm()
    ev = {k: (int(calib[k]) if k == "pulse_duration" else float(calib[k])) for k in EVAL_KEYS if k in calib}
    with stop_calibration(calib):
        np.random.seed(int(seed))
        with _integrator(arm):
            r = DM.evaluate_seed(int(seed), margin_ref=float(DM.MARGIN_REF), verbose=True,
                                 **{BUILD_KEYS[k]: float(v) for k, v in calib.items() if k in BUILD_KEYS}, **ev)
        S = _build(seed, arm, calib)
        std = DM.WorkspaceDepression(S["bridge"], S["xp"], S["ws_used"])
        stop = r["measurements"]["stop"]
        trial = _stop_trial(S, std, thal_of(calib), (0, 1, 2), stop["boost"],
                            int(calib.get("pulse_duration", DM.PULSE_DURATION)), float(DM.STRONG_PA))
    return {"runner": "research/runners/stable_integration_stop_ignition_probe.py --organ", "seed": int(seed),
            "arm": arm, "calibration": calib, "result": r, "stop_trial_membranes": trial}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="off,on")
    ap.add_argument("--calib", default="", help="JSON calibration dict (missing keys = production)")
    ap.add_argument("--weights", default=",".join("%g" % w for w in WEIGHTS))
    ap.add_argument("--drives", default=",".join("%g" % d for d in CURVE_DRIVES))
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--inhibition", default="", help="comma list of norm_pool->workspace weights (curve per value)")
    ap.add_argument("--sustain", action="store_true")
    ap.add_argument("--co", action="store_true", help="--sustain with all three patterns driven (the conflict state)")
    ap.add_argument("--thals", default="", help="comma list of thal tonic drives for --sustain (default: calib's)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--stopwin", action="store_true")
    ap.add_argument("--boosts", default=",".join("%g" % x for x in STOP_BOOSTS))
    ap.add_argument("--repeats", type=int, default=1, help="--stopwin trials per boost")
    ap.add_argument("--held", default="012,01", help="--stopwin held sets, e.g. '012,01'")
    ap.add_argument("--margins", action="store_true", help="confident vs conflict sensor margins (--repeats)")
    ap.add_argument("--organ", action="store_true")
    a = ap.parse_args()
    if a.seed in GATE_SEEDS:
        sys.exit("GATE seed %d refused: dev-seed diagnostic only" % a.seed)
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    calib = json.loads(a.calib) if a.calib else {}
    weights = [float(x) for x in a.weights.split(",") if x.strip()]
    drives = [float(x) for x in a.drives.split(",") if x.strip()]
    if a.margins:
        d = run_margins(a.seed, calib, arms=arms, repeats=a.repeats,
                        contents_list=tuple(tuple(int(c) for c in g) for g in a.held.split(",") if g.strip()))
    elif a.organ:
        d = {"arms": {arm: run_organ(a.seed, arm, calib) for arm in arms}}
    elif a.stopwin:
        d = run_stopwin(a.seed, calib, arms=arms, boosts=[float(x) for x in a.boosts.split(",") if x.strip()],
                        repeats=a.repeats,
                        contents_list=tuple(tuple(int(c) for c in g) for g in a.held.split(",") if g.strip()))
    elif a.sustain:
        thals = [float(x) for x in a.thals.split(",") if x.strip()] or [thal_of(calib)]
        d = run_sustain(a.seed, calib, weights, thals, drives, arms=arms, steps=a.steps, co=a.co)
    elif a.inhibition:
        d = {"runner": "research/runners/stable_integration_stop_ignition_probe.py --inhibition", "seed": a.seed,
             "by_norm_ws_w": {x: run_curve(a.seed, dict(calib, norm_ws_w=float(x)), weights, drives, arms)
                              for x in a.inhibition.split(",") if x.strip()}}
    elif a.curve:
        d = run_curve(a.seed, calib, weights, drives, arms)
    else:
        ap.error("choose --curve, --inhibition, --sustain, --stopwin or --organ")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(d, f, indent=1, default=str)
    print("[stop ignition] wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
