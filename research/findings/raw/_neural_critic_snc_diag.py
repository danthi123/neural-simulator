"""Throwaway diagnostic for the --enable-neural-critic SNc-silencing bug
(2026-06-08). Root-causes why the neural-critic config silences the SNc.

Runs the flagship A+E+G v2.5 nav stack at REDUCED n_steps under 3 conditions,
logging EVERY step (not just reward windows): per-step firing rates of
cortex_it / striosome_value / snc / motor_*, the GABA_B conductance magnitude
on the SNc neurons (mean/max), and a NaN check on the membrane potential.

  (A) neural-critic ON (the failing config).
  (B) neural-critic ON but cfg.enable_gabab forced OFF at runtime (decisive
      isolation: critic region + cortex_it->striosome_value + host -V drop all
      still on, but no GABA_B current). If SNc fires in (B), GABA_B over-
      inhibition is the silencer.
  (C) Stage A (--spiking-snc only) positive control.

NO sim/ edits. Pure runner-side instrumentation via a monkeypatch on
SimulationBridge._run_one_simulation_step. Does NOT change the runtime
behavior of the --enable-neural-critic flag itself.
"""
import os
import sys
import json
import argparse

# Force GPU (CuPy) backend — decisive run, not a smoke.
os.environ.setdefault("SIM_BACKEND", "cupy")
# Determinism: g11 reads sys.argv for "--deterministic" at import to set this
# env var before cupy import. We set it directly (the diagnostic doesn't pass
# the flag through argv) so seed-to-seed noise is tight.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

# Repo root on path (this file lives in research/findings/raw/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim import SimulationBridge
import research.runners.g11_bg_runner as g11

# ---------------------------------------------------------------------------
# Per-step probe state (module global; reset per condition).
# ---------------------------------------------------------------------------
_PROBE = {
    "rows": [],            # one dict per step
    "force_gabab_off": False,
    "snc_idx": None,
    "it_idx": None,
    "striov_idx": None,
    "motor_idx": {},       # action -> host idx array
    "step_count": 0,
    "max_steps": 10**9,
}

_ACTIONS = ["N", "E", "S", "W"]

_orig_step = SimulationBridge._run_one_simulation_step


def _patched_step(self):
    # Resolve region indices once (first call) from the bridge's region manager.
    if _PROBE["snc_idx"] is None and getattr(self, "region_manager", None) is not None:
        try:
            names = set(self.region_manager.regions_by_name.keys()) \
                if hasattr(self.region_manager, "regions_by_name") else None
        except Exception:
            names = None

        def _idx(name):
            try:
                idx = self.region_manager.indices(name)
            except Exception:
                return None
            idx = np.asarray(idx.get() if hasattr(idx, "get") else idx)
            return idx if idx.size else None

        _PROBE["snc_idx"] = _idx("snc")
        _PROBE["it_idx"] = _idx("cortex_it")
        _PROBE["striov_idx"] = _idx("striosome_value")
        for a in _ACTIONS:
            _PROBE["motor_idx"][a] = _idx(f"motor_{a}")

    # Decisive isolation (condition B): force GABA_B current OFF at runtime.
    # Line ~5499 of bridge.py reads getattr(cfg, "enable_gabab", False) EACH
    # step, so flipping this on the live core_config disables the GABA_B
    # increment + current block without touching the allocated conductance.
    if _PROBE["force_gabab_off"]:
        try:
            self.core_config.enable_gabab = False
        except Exception:
            pass

    _orig_step(self)

    if _PROBE["step_count"] >= _PROBE["max_steps"]:
        _PROBE["step_count"] += 1
        return

    # ---- probe AFTER the step ----
    fs = self.cp_firing_states
    fs_h = fs.get() if hasattr(fs, "get") else np.asarray(fs)

    def _rate(idx):
        if idx is None or idx.size == 0:
            return float("nan")
        return float(fs_h[idx].mean())  # fraction firing this step

    row = {
        "it": _rate(_PROBE["it_idx"]),
        "striov": _rate(_PROBE["striov_idx"]),
        "snc": _rate(_PROBE["snc_idx"]),
    }
    for a in _ACTIONS:
        row[f"motor_{a}"] = _rate(_PROBE["motor_idx"][a])

    # GABA_B conductance magnitude on the SNc neurons.
    g = getattr(self, "cp_conductance_g_gabab", None)
    if g is not None and _PROBE["snc_idx"] is not None:
        gsnc = g[ _PROBE["snc_idx"] ]
        gsnc_h = gsnc.get() if hasattr(gsnc, "get") else np.asarray(gsnc)
        row["gabab_snc_mean"] = float(gsnc_h.mean())
        row["gabab_snc_max"] = float(gsnc_h.max())
        # also the whole-array max so we see if anything else accumulates
        g_all = g.get() if hasattr(g, "get") else np.asarray(g)
        row["gabab_all_max"] = float(g_all.max())
    else:
        row["gabab_snc_mean"] = float("nan")
        row["gabab_snc_max"] = float("nan")
        row["gabab_all_max"] = float("nan")

    # NaN check on membrane potential.
    v = self.cp_membrane_potential_v
    v_h = v.get() if hasattr(v, "get") else np.asarray(v)
    row["v_nan"] = bool(np.isnan(v_h).any())
    row["v_snc_mean"] = (
        float(v_h[_PROBE["snc_idx"]].mean()) if _PROBE["snc_idx"] is not None
        else float("nan")
    )

    _PROBE["rows"].append(row)
    _PROBE["step_count"] += 1


def _summary(rows):
    """Aggregate the per-step rows into means + a sparse trace."""
    if not rows:
        return {"n_steps_logged": 0}
    keys = ["it", "striov", "snc", "motor_N", "motor_E", "motor_S", "motor_W",
            "gabab_snc_mean", "gabab_snc_max", "gabab_all_max",
            "v_snc_mean"]
    out = {"n_steps_logged": len(rows)}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[f"{k}_mean"] = float(finite.mean()) if finite.size else float("nan")
        out[f"{k}_max"] = float(finite.max()) if finite.size else float("nan")
    out["any_v_nan"] = bool(any(r["v_nan"] for r in rows))
    # how many steps had ANY snc firing
    snc_vals = np.asarray([r["snc"] for r in rows], dtype=float)
    out["n_steps_snc_fired"] = int(np.sum(snc_vals > 0))
    return out


def _flagship_kwargs(seed, n_steps, spiking_snc, enable_neural_critic):
    """The flagship A+E+G v2.5 nav config + the spiking-SNc / neural-critic
    flags, as run_moving_goal_episode kwargs. Mirrors the documented recipe:
      --moving-goal --goal-schedule multi --deterministic
      --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry
      --enable-striatal-pv-fsi --enable-cluster-a-closed-loop
      --enable-cluster-e-topography --enable-dlpfc-wm --enable-pfc-nmda
      --enable-visual-cortex --visual-cortex-action-warmup-steps 600
      --grid-size 32
    Most other params keep run_moving_goal_episode defaults.
    """
    # multi goal schedule (mirrors --goal-schedule multi); compress the warmup
    # so a short run still exercises the action gate (warmup 600 would never
    # open in a 200-step run; use a small warmup so cortex_it->cortex_X opens).
    # NOTE: this is a DIAGNOSTIC; the SNc-silence mechanism is independent of
    # the warmup (the critic GABA_B fires from step 0 regardless of the action
    # gate, which only gates cortex_it->cortex_X, not cortex_it->striosome_value).
    # NOTE on parameter names (verified against run_moving_goal_episode):
    #   --enable-msn-lateral-inhibition -> enable_bg_lateral_inhibition
    #   --enable-striatal-pv-fsi        -> enable_striatal_fsis
    #   --enable-dlpfc-wm / --pfc       -> enable_pfc
    #   --deterministic is NOT a kwarg; it sets a CUBLAS env var at import via
    #     the argv check at g11 module top. Determinism is not required to
    #     root-cause the (deterministic) GABA_B accumulation; we omit it.
    out_path = os.path.join(_HERE, f"_neural_critic_snc_diag_nav_s{seed}.json")
    return dict(
        out_path=out_path,
        seed=seed,
        n_steps=n_steps,
        grid_size=32,
        start_pos=(1, 1),
        goal_pos=(28, 28),
        enable_bg_lateral_inhibition=True,
        enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        enable_pfc_nmda=True,
        enable_visual_cortex=True,
        visual_cortex_action_warmup_steps=100,  # compressed for the short run
        spiking_snc=spiking_snc,
        enable_neural_critic=enable_neural_critic,
        verbose=True,
        progress_print_interval=50,
    )


def run_condition(label, seed, n_steps, spiking_snc, enable_neural_critic,
                  force_gabab_off):
    print(f"\n{'='*72}\n  CONDITION {label}: spiking_snc={spiking_snc} "
          f"neural_critic={enable_neural_critic} force_gabab_off={force_gabab_off}\n"
          f"{'='*72}", flush=True)
    # reset probe
    _PROBE["rows"] = []
    _PROBE["force_gabab_off"] = force_gabab_off
    _PROBE["snc_idx"] = None
    _PROBE["it_idx"] = None
    _PROBE["striov_idx"] = None
    _PROBE["motor_idx"] = {}
    _PROBE["step_count"] = 0
    _PROBE["max_steps"] = n_steps * 200  # log generously (covers stim windows)

    kwargs = _flagship_kwargs(seed, n_steps, spiking_snc, enable_neural_critic)
    results = g11.run_moving_goal_episode(**kwargs)

    # nav metric: sum of per-phase final-quarter mean distance
    phase_stats = results.get("phase_stats", [])
    sum_final_q = float(sum(p["final_quarter_mean_distance"] for p in phase_stats)) \
        if phase_stats else float("nan")

    snc_rate_log = results.get("snc_rate_log", [])
    striov_rate_log = results.get("striov_rate_log", [])

    probe = _summary(_PROBE["rows"])

    out = {
        "label": label,
        "spiking_snc": spiking_snc,
        "enable_neural_critic": enable_neural_critic,
        "force_gabab_off": force_gabab_off,
        "n_steps": n_steps,
        # runner's own reward-window readouts:
        "snc_rate_log_mean": float(np.mean(snc_rate_log)) if snc_rate_log else None,
        "snc_rate_log_max": float(np.max(snc_rate_log)) if snc_rate_log else None,
        "striov_rate_log_mean": float(np.mean(striov_rate_log)) if striov_rate_log else None,
        "striov_rate_log_max": float(np.max(striov_rate_log)) if striov_rate_log else None,
        "critic_weight_initial": results.get("critic_weight_initial"),
        "critic_weight_final": results.get("critic_weight_final"),
        "sum_final_quarter_distance": sum_final_q,
        "n_steps_at_goal": results.get("n_steps_at_goal"),
        # per-step probe (the decisive evidence):
        "probe": probe,
    }
    print(f"\n[COND {label}] probe summary:")
    print(json.dumps(out, indent=2, default=str), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "_neural_critic_snc_diag_result.json"))
    ap.add_argument("--conditions", type=str, default="A,B,C",
                    help="comma list of A/B/C to run")
    args = ap.parse_args()

    SimulationBridge._run_one_simulation_step = _patched_step

    conds = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    all_out = {}
    try:
        if "A" in conds:
            all_out["A"] = run_condition(
                "A", args.seed, args.n_steps,
                spiking_snc=True, enable_neural_critic=True, force_gabab_off=False)
        if "B" in conds:
            all_out["B"] = run_condition(
                "B", args.seed, args.n_steps,
                spiking_snc=True, enable_neural_critic=True, force_gabab_off=True)
        if "C" in conds:
            all_out["C"] = run_condition(
                "C", args.seed, args.n_steps,
                spiking_snc=True, enable_neural_critic=False, force_gabab_off=False)
    finally:
        SimulationBridge._run_one_simulation_step = _orig_step

    with open(args.out, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nWROTE {args.out}", flush=True)

    # Final compact verdict table
    print(f"\n{'='*72}\n  VERDICT TABLE\n{'='*72}")
    hdr = f"{'cond':<5} {'snc_step_mean':>13} {'snc_fired/steps':>16} {'gabab_snc_mean':>15} {'gabab_snc_max':>14} {'v_nan':>6} {'sumFinalQ':>10} {'snc_rate_log':>13}"
    print(hdr)
    for c in ("A", "B", "C"):
        if c not in all_out:
            continue
        o = all_out[c]
        p = o["probe"]
        print(f"{c:<5} {p.get('snc_mean', float('nan')):>13.5f} "
              f"{str(p.get('n_steps_snc_fired'))+'/'+str(p.get('n_steps_logged')):>16} "
              f"{p.get('gabab_snc_mean_mean', float('nan')):>15.4f} "
              f"{p.get('gabab_snc_max_max', float('nan')):>14.4f} "
              f"{str(p.get('any_v_nan')):>6} "
              f"{o.get('sum_final_quarter_distance', float('nan')):>10.2f} "
              f"{str(o.get('snc_rate_log_mean')):>13}")


if __name__ == "__main__":
    main()
