"""READ-ONLY blocker-1 nav-loop isolation (2026-06-08, deep-research subagent).

The free-running probe (_gabab_flag_isolation_probe.py) showed enable_gabab is
INERT when the network is driven by a constant cortex current with NO reward-hold
loop (P0==P1==P2, all active). But the FAILING config runs the full
run_moving_goal_episode (reward-hold sub-loop, visual retina drive, action
selection). So the silence is an interaction of the FLAG with the full nav loop.

This probe runs the REAL run_moving_goal_episode (short) under conditions that
decouple the FLAG from the CRITIC SYNAPSE inside the live nav loop:

  (D) critic OFF, enable_gabab FORCED ON at runtime (monkeypatch each step).
      => the GABA_B per-step block executes, but cp_gabab_synapse_mask is None
         (no pathway tagged), so the masked-matvec increment branch is SKIPPED
         (guarded by `cp_gabab_synapse_mask is not None`). Only the decay+current
         on a zero conductance runs. If D SILENCES the net like cond A, the bare
         FLAG + reward-loop interaction is the silencer (mask irrelevant). If D is
         FINE, the 407-synapse mask matvec is implicated.

Compare against the recovered diag: A (critic ON, gabab ON) = SILENT (snc 0);
B (critic ON, gabab forced off) = WORKS (snc 8.16, dist 0.5).

NO sim/ edits. Monkeypatch on _run_one_simulation_step sets cfg.enable_gabab
each step (read at line ~5499 every step). Pure runtime flip; build is critic-OFF.
SERIAL, GPU, short.
"""
import os
import sys
import json

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sim.bridge import SimulationBridge
import research.runners.g11_bg_runner as g11

_ACTIONS = ["N", "E", "S", "W"]
_PROBE = {"force_gabab_on": False, "snc_idx": None, "rows": []}
_orig_step = SimulationBridge._run_one_simulation_step


def _patched_step(self):
    # Force enable_gabab ON at runtime (read fresh at the GABA_B block each step).
    # With the critic OFF build, cp_conductance_g_gabab is None UNLESS we also
    # allocate it. The block is guarded by `cp_conductance_g_gabab is not None`,
    # so to truly exercise the flag path we must ensure the conductance exists.
    if _PROBE["force_gabab_on"]:
        try:
            self.core_config.enable_gabab = True
            # Lazily allocate the slow conductance + reversal if the critic-OFF
            # build left them None (so the per-step GABA_B block actually runs).
            if getattr(self, "cp_conductance_g_gabab", None) is None:
                import cupy as cp
                n = int(self.cp_membrane_potential_v.shape[0])
                self.cp_conductance_g_gabab = cp.zeros(n, dtype=cp.float32)
                self.cp_gabab_reversal_per_neuron = cp.full(n, -90.0, dtype=cp.float32)
                # cache the decay (read each step from _cached_decay_gabab)
                self._cached_decay_gabab = float(cp.exp(-self.core_config.dt_ms / 150.0))
                # leave cp_gabab_synapse_mask = None => increment branch skipped
        except Exception as e:
            print("  [patch] alloc failed:", e, flush=True)

    _orig_step(self)

    if _PROBE["snc_idx"] is None and getattr(self, "region_manager", None) is not None:
        try:
            idx = self.region_manager.indices("snc")
            _PROBE["snc_idx"] = np.asarray(idx.get() if hasattr(idx, "get") else idx)
        except Exception:
            _PROBE["snc_idx"] = None
    if _PROBE["snc_idx"] is not None:
        fs = self.cp_firing_states
        fs_h = fs.get() if hasattr(fs, "get") else np.asarray(fs)
        g = getattr(self, "cp_conductance_g_gabab", None)
        gmax = float((g.get() if hasattr(g, "get") else np.asarray(g)).max()) if g is not None else float("nan")
        _PROBE["rows"].append({
            "snc": float(fs_h[_PROBE["snc_idx"]].mean()),
            "n_fired": int(fs_h.sum()),
            "gabab_all_max": gmax,
        })


def run(force_gabab_on, seed=42, n_steps=150):
    label = "D critic-OFF gabab-FORCED-ON" if force_gabab_on else "ctrl critic-OFF gabab-OFF"
    print(f"\n{'='*70}\n  {label}\n{'='*70}", flush=True)
    _PROBE["force_gabab_on"] = force_gabab_on
    _PROBE["snc_idx"] = None
    _PROBE["rows"] = []
    out_path = os.path.join(_HERE, f"_gabab_navloop_nav_s{seed}.json")
    kwargs = dict(
        out_path=out_path, seed=seed, n_steps=n_steps, grid_size=32,
        start_pos=(1, 1), goal_pos=(28, 28),
        enable_bg_lateral_inhibition=True, enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True, enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True, enable_pfc=True, enable_pfc_nmda=True,
        enable_visual_cortex=True, visual_cortex_action_warmup_steps=100,
        spiking_snc=True, enable_neural_critic=False,  # CRITIC OFF
        verbose=False, progress_print_interval=0,
    )
    results = g11.run_moving_goal_episode(**kwargs)
    phase_stats = results.get("phase_stats", [])
    sum_final_q = float(sum(p["final_quarter_mean_distance"] for p in phase_stats)) if phase_stats else float("nan")
    snc_rate_log = results.get("snc_rate_log", [])
    rows = _PROBE["rows"]
    snc_step_mean = float(np.mean([r["snc"] for r in rows])) if rows else float("nan")
    n_fired_mean = float(np.mean([r["n_fired"] for r in rows])) if rows else float("nan")
    gabab_max = float(np.nanmax([r["gabab_all_max"] for r in rows])) if rows else float("nan")
    n_snc_fired = int(sum(1 for r in rows if r["snc"] > 0))
    out = {
        "label": label, "force_gabab_on": force_gabab_on,
        "snc_rate_log_mean": float(np.mean(snc_rate_log)) if snc_rate_log else None,
        "sum_final_quarter_distance": sum_final_q,
        "n_steps_at_goal": results.get("n_steps_at_goal"),
        "snc_step_mean": snc_step_mean, "n_fired_step_mean": n_fired_mean,
        "n_steps_logged": len(rows), "n_steps_snc_fired": n_snc_fired,
        "gabab_all_max": gabab_max,
    }
    print(f"  -> snc_rate_log_mean={out['snc_rate_log_mean']}  sumFinalQ={out['sum_final_quarter_distance']:.2f}  "
          f"snc_step_mean={out['snc_step_mean']:.4f}  n_fired_mean={out['n_fired_step_mean']:.1f}  "
          f"snc_fired={out['n_steps_snc_fired']}/{len(rows)}  gabab_all_max={out['gabab_all_max']:.4f}", flush=True)
    return out


def main():
    SimulationBridge._run_one_simulation_step = _patched_step
    all_out = {}
    try:
        all_out["ctrl"] = run(force_gabab_on=False)   # critic OFF, no flag (sanity)
        all_out["D"] = run(force_gabab_on=True)        # critic OFF, flag forced ON
    finally:
        SimulationBridge._run_one_simulation_step = _orig_step
    out_path = os.path.join(_HERE, "_gabab_navloop_isolation_result.json")
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nWROTE {out_path}")
    print(f"\n{'='*70}\n  VERDICT (compare to recovered diag A=silent, B=works)\n{'='*70}")
    for c in ("ctrl", "D"):
        o = all_out[c]
        print(f"  {o['label']:<32} snc_rate_log={o['snc_rate_log_mean']}  sumFinalQ={o['sum_final_quarter_distance']:.2f}  "
              f"snc_step_mean={o['snc_step_mean']:.4f}  gabab_max={o['gabab_all_max']:.4f}")


if __name__ == "__main__":
    main()
