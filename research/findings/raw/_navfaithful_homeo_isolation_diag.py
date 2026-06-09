"""Diagnostic: WHY per-region-critic-only homeostasis underperforms GLOBAL homeostasis on the
nav-faithful critic wiring. NOT the verdict (the verdict is the graceful-FAIL of the de-risk).

Three conditions on the SAME deterministic-nav wiring (dense afferent -> MSN-D1 critic), all
with OU/conductance-noise OFF, training the value-leads-reward protocol identically:
  (A) homeostasis fully OFF       -> the original FAIL (critic can't fire; the baseline)
  (B) per-region critic ONLY      -> the protected edit under test (critic mask, afferent fixed)
  (C) GLOBAL homeostasis ON        -> the forensic's "re-flip homeostasis ON" (ALL regions adapt,
                                       incl. the dense afferent -> it fires harder -> drives the
                                       critic harder). This is the ceiling the forensic reported.

Reports the trained critic V(near) firing rate + near-ensemble weight growth for each, isolating
whether the per-region-only shortfall is because the AFFERENT (not just the critic) needs the
adapted threshold to push the under-active MSN into a firing range. Run under SIM_BACKEND=numpy.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np

from research.runners.snc_stageb_critic_probe_navfaithful import (
    _build_navfaithful_bridge, _grid_prefs, grid_place_code_drive,
)
from research.runners.snc_stageb_critic_probe_place import (
    _drive_place, _calibrate_da_threshold, _mean_pathway_weight,
    _clear_eligibility, _ensemble_overlap, _idx, _host,
)


def _force_global_homeostasis(bridge):
    """Turn GLOBAL homeostasis ON post-build (the forensic's condition C). This adapts ALL
    regions' thresholds, including the dense afferent — so it does NOT replicate the
    deterministic regime; it is ONLY a diagnostic of the firing ceiling."""
    bridge.core_config.enable_homeostasis = True


def run_condition(seed, *, mode, n_train=40, hold_steps=40,
                  vs_place_drive_pa=800.0, vs_place_sigma=4.0,
                  snc_tonic_pa=180.0, snc_reward_gain=300.0,
                  p_near_xy=(26.571, 26.571), grid_size=32):
    from sim.backend import get_backend
    xp, _ = get_backend()
    critic_homeo = (mode == "per_region_critic")
    bridge, cfg = _build_navfaithful_bridge(
        seed, grid_size=grid_size, vs_place_to_strio_weight=0.2,
        gabab=True, gabab_propagation_strength=0.02, critic_homeostasis=critic_homeo)
    if mode == "global":
        _force_global_homeostasis(bridge)
    regions = ("vs_place_context", "striosome_value", "snc",
               "sensor_place_readout", "cortex_N", "cortex_E", "cortex_S", "cortex_W")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}
    idx_map["place"] = idx_map["vs_place_context"]
    n_vs = len(_host(idx_map["vs_place_context"]))
    vs_prefs = _grid_prefs(n_vs, grid_size)
    near_vec = grid_place_code_drive(p_near_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)
    near_set_drive = np.asarray(near_vec, dtype=np.float64)
    k = max(1, int(round(0.25 * len(near_set_drive))))
    near_set = set(int(i) for i in np.argsort(near_set_drive)[-k:])
    _calibrate_da_threshold(bridge, cfg, idx_map, snc_tonic_pa, xp)
    w_near_init = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
    v_curve = []
    for t in range(n_train):
        _drive_place(bridge, idx_map, None, {"snc": snc_tonic_pa}, hold_steps, xp)
        _clear_eligibility(bridge)
        snc_r, strio_r, da = _drive_place(
            bridge, idx_map, near_vec, {"snc": snc_tonic_pa + snc_reward_gain}, hold_steps, xp)
        v_curve.append(strio_r)
    w_near_final = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
    # Final V(near): drive near, measure critic rate (no reward) over a read window.
    _, v_read, _ = _drive_place(bridge, idx_map, near_vec, {"snc": snc_tonic_pa}, hold_steps, xp)
    v_late = float(np.mean(v_curve[-max(1, n_train // 5):]))
    return dict(mode=mode, seed=seed, v_late=v_late, v_read=v_read,
                w_near_init=w_near_init, w_near_final=w_near_final,
                w_growth=w_near_final / max(w_near_init, 1e-6),
                global_homeo=bool(bridge.core_config.enable_homeostasis),
                per_region_mask=(getattr(bridge, "cp_homeostasis_neuron_mask", None) is not None))


if __name__ == "__main__":
    seeds = [42, 43, 44]
    print(f"{'mode':<22}{'seed':>5}{'V_late(Hz)':>12}{'V_read(Hz)':>12}{'w_growth':>10}"
          f"{'glob_homeo':>12}{'critic_mask':>13}")
    for mode in ("off", "per_region_critic", "global"):
        for s in seeds:
            r = run_condition(s, mode=mode)
            print(f"{r['mode']:<22}{r['seed']:>5}{r['v_late']:>12.2f}{r['v_read']:>12.2f}"
                  f"{r['w_growth']:>10.2f}{str(r['global_homeo']):>12}{str(r['per_region_mask']):>13}")
