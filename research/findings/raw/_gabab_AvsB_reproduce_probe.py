"""READ-ONLY: directly reproduce the recovered-diag A-vs-B result (2026-06-08).

Load-bearing claim under test: with the critic BUILT (enable_neural_critic=True,
407 GABA_B synapses), the live nav loop is SILENT when enable_gabab stays ON
(cond A) but WORKS when enable_gabab is forced OFF each step (cond B). My
isolation probes showed the GABA_B per-step block is numerically inert when the
critic doesn't fire (g=0 => I_gabab=0), which CONTRADICTS a pure-flag silencer.
So re-run A and B back-to-back, identical builds, only the runtime flag differs,
and log snc / total-firing every step + the GABA_B conductance max.

If A is silent and B active => genuine (if opaque) flag interaction, real blocker 1.
If both active => the original diag's 'A silent' was a transient/confound, and
blocker 1 (as a flag instability) does NOT reproduce.

NO sim/ edits. Monkeypatch flips cfg.enable_gabab each step for cond B only.
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

_PROBE = {"force_gabab_off": False, "snc_idx": None, "rows": []}
_orig_step = SimulationBridge._run_one_simulation_step


def _patched_step(self):
    if _PROBE["force_gabab_off"]:
        try:
            self.core_config.enable_gabab = False
        except Exception:
            pass
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


def run(label, force_gabab_off, seed=42, n_steps=150):
    print(f"\n{'='*70}\n  {label}\n{'='*70}", flush=True)
    _PROBE["force_gabab_off"] = force_gabab_off
    _PROBE["snc_idx"] = None
    _PROBE["rows"] = []
    out_path = os.path.join(_HERE, f"_gabab_AvsB_nav_s{seed}.json")
    kwargs = dict(
        out_path=out_path, seed=seed, n_steps=n_steps, grid_size=32,
        start_pos=(1, 1), goal_pos=(28, 28),
        enable_bg_lateral_inhibition=True, enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True, enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True, enable_pfc=True, enable_pfc_nmda=True,
        enable_visual_cortex=True, visual_cortex_action_warmup_steps=100,
        spiking_snc=True, enable_neural_critic=True,  # CRITIC ON (407 GABA_B synapses)
        verbose=False, progress_print_interval=0,
    )
    results = g11.run_moving_goal_episode(**kwargs)
    phase_stats = results.get("phase_stats", [])
    sum_final_q = float(sum(p["final_quarter_mean_distance"] for p in phase_stats)) if phase_stats else float("nan")
    snc_rate_log = results.get("snc_rate_log", [])
    rows = _PROBE["rows"]
    out = {
        "label": label, "force_gabab_off": force_gabab_off,
        "snc_rate_log_mean": float(np.mean(snc_rate_log)) if snc_rate_log else None,
        "sum_final_quarter_distance": sum_final_q,
        "n_steps_at_goal": results.get("n_steps_at_goal"),
        "snc_step_mean": float(np.mean([r["snc"] for r in rows])) if rows else float("nan"),
        "n_fired_step_mean": float(np.mean([r["n_fired"] for r in rows])) if rows else float("nan"),
        "n_steps_snc_fired": int(sum(1 for r in rows if r["snc"] > 0)),
        "n_steps_logged": len(rows),
        "gabab_all_max": float(np.nanmax([r["gabab_all_max"] for r in rows])) if rows else float("nan"),
        "critic_weight_initial": results.get("critic_weight_initial"),
    }
    print(f"  -> snc_rate_log_mean={out['snc_rate_log_mean']}  sumFinalQ={out['sum_final_quarter_distance']:.2f}  "
          f"snc_step_mean={out['snc_step_mean']:.4f}  n_fired_mean={out['n_fired_step_mean']:.1f}  "
          f"snc_fired={out['n_steps_snc_fired']}/{len(rows)}  gabab_all_max={out['gabab_all_max']:.4f}", flush=True)
    return out


def main():
    SimulationBridge._run_one_simulation_step = _patched_step
    all_out = {}
    try:
        all_out["A"] = run("A critic-ON gabab-ON (as-built)", force_gabab_off=False)
        all_out["B"] = run("B critic-ON gabab-FORCED-OFF", force_gabab_off=True)
    finally:
        SimulationBridge._run_one_simulation_step = _orig_step
    out_path = os.path.join(_HERE, "_gabab_AvsB_reproduce_result.json")
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nWROTE {out_path}")
    print(f"\n{'='*70}\n  VERDICT\n{'='*70}")
    for c in ("A", "B"):
        o = all_out[c]
        print(f"  {o['label']:<34} snc_rate_log={o['snc_rate_log_mean']}  sumFinalQ={o['sum_final_quarter_distance']:.2f}  "
              f"snc_step_mean={o['snc_step_mean']:.4f}  gabab_max={o['gabab_all_max']:.4f}")
    a, b = all_out["A"], all_out["B"]
    a_silent = (a["snc_rate_log_mean"] or 0) < 1.0
    b_works = (b["snc_rate_log_mean"] or 0) > 3.0
    if a_silent and b_works:
        print("\n  => REPRODUCES the diag: A silent, B works. Blocker 1 is a genuine flag interaction.")
    elif (a["snc_rate_log_mean"] or 0) > 3.0 and b_works:
        print("\n  => DOES NOT reproduce: A also works. The diag 'A silent' was a transient/confound.")
    else:
        print("\n  => MIXED/other; inspect numbers.")


if __name__ == "__main__":
    main()
