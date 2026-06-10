"""De-risk 5b for the navigation+conversational single-instance unification (roadmap step 2, STRICT path).

Per `docs/plans/2026-06-10-nav-conv-single-instance-unification-design.md` §5b: the literal crux of the
owner-chosen STRICT (RF co-resident) merge. The resonate-and-fire (RF) composer stores its complex phasor
state Z = re + i*im in the SAME two arrays the Izhikevich navigation neurons use — `re` in
`cp_membrane_potential_v`, `im` in `cp_recovery_variable_u` (`sim/bridge.py:5380-5381`). The neuron-model
dynamics dispatch in `_run_one_simulation_step` is a SINGLE GLOBAL branch on `cfg.neuron_model_type`
(`sim/bridge.py:5870` Izhikevich / ... / RESONATE_AND_FIRE), so on any step EITHER the Izhikevich branch
runs for ALL neurons OR the RF branch does — never both.

5b (as-is) demonstrates the KILL this implies: place a unit phasor in `v`/`u` (the RF state) and run ONE
Izhikevich `_run_one_simulation_step` — the phasor is destroyed (the Izhikevich `+140` drive sends `v` far
past the RF unit circle, then spike-resets). This CONFIRMS that true RF+Izhikevich co-residence needs the
protected `sim/` edit, and it PINS the edited-version PASS criterion (below).

PASS CRITERION FOR THE EDITED VERSION (built only after this kill + owner byte-review; the design's
recommended edit is SEPARATE RF state arrays + a per-neuron model mask + a masked dual-dynamics step):
  - On a mixed bridge, ONE `_run_one_simulation_step` advances the RF slice by RF dynamics (its phase
    read-back byte-matches a pure-`rf_resonate_steps` reference) AND the Izhikevich slice by Izhikevich
    dynamics (matches a pure-Izhikevich control).
  - With `enable_mixed_neuron_models=False`, a RESONATE_AND_FIRE-only bridge and an IZHIKEVICH-only bridge
    are BYTE-UNCHANGED vs today (the guard).
This file's `rf_reference_one_step()` is exactly that reference; the corruption case is what the edit must
fix. Run on GPU (CuPy).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host
from research.runners.rf_phasor_composer import _build_rf_bridge   # Izhikevich-init then flip to RF


def build_izh_bridge(n: int, seed: int = 42) -> SimulationBridge:
    """The SAME construction as `_build_rf_bridge` but STAYS Izhikevich (no flip) — the navigation model."""
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge   # stays IZHIKEVICH


def rf_reference_one_step(z0: np.ndarray, period: int, seed: int = 42):
    """The CORRECT one-step RF evolution: kick the phasor onto an RF bridge, advance ONE RF step, read v/u."""
    n = z0.shape[0]
    ref = _build_rf_bridge(n, seed=seed)
    ref.rf_kick(z0, period=period, lam=0.0)          # lam=0 -> magnitude exactly preserved (clean reference)
    ref.rf_resonate_steps(1)                          # one RF dynamics step (the fast path = one _rf_advance_one)
    re = to_host(ref.cp_membrane_potential_v).copy()
    im = to_host(ref.cp_recovery_variable_u).copy()
    return re, im


def main():
    ap = argparse.ArgumentParser(description="Unification de-risk 5b: RF/Izhikevich shared-v/u corruption (as-is)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--period", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/derisk_5b_rf_izh.json")
    args = ap.parse_args()

    xp, backend = get_backend()
    n, period = args.n, args.period
    print(f"[5b] backend={backend} n={n} period={period} seed={args.seed}")

    # known unit phasors: phases 0, 1/n, ..., (n-1)/n  ->  |z0| == 1 exactly
    phases0 = np.arange(n) / n
    z0 = np.exp(2j * np.pi * phases0)
    mag0 = float(np.mean(np.abs(z0)))

    # --- (1) the correct RF one-step evolution (reference) ---
    ref_re, ref_im = rf_reference_one_step(z0, period=period, seed=args.seed)
    ref_mag = np.sqrt(ref_re ** 2 + ref_im ** 2)
    ref_mag_mean = float(ref_mag.mean())

    # --- (2) the AS-IS shared-bridge corruption: a phasor in v/u, ONE Izhikevich step ---
    izh = build_izh_bridge(n, seed=args.seed)
    izh.cp_membrane_potential_v[:] = xp.asarray(z0.real, dtype=izh.cp_membrane_potential_v.dtype)
    izh.cp_recovery_variable_u[:] = xp.asarray(z0.imag, dtype=izh.cp_recovery_variable_u.dtype)
    before_re = to_host(izh.cp_membrane_potential_v).copy()
    before_im = to_host(izh.cp_recovery_variable_u).copy()
    izh._run_one_simulation_step()                    # ONE Izhikevich step over the phasor-laden v/u
    after_re = to_host(izh.cp_membrane_potential_v).copy()
    after_im = to_host(izh.cp_recovery_variable_u).copy()
    after_mag = np.sqrt(after_re ** 2 + after_im ** 2)
    after_mag_mean = float(after_mag.mean())

    # how far the Izhikevich step took the state AWAY from the correct RF rotation
    dev_from_rf = float(np.max(np.abs(np.concatenate([after_re - ref_re, after_im - ref_im]))))
    re_changed = float(np.max(np.abs(after_re - before_re)))

    # KILL = the Izhikevich step did NOT preserve the unit phasor (magnitude blew up) AND it does not match
    # the RF rotation -> v/u cannot be time-shared between the two models in one step loop, as-is.
    corrupted = (after_mag_mean > 5.0 * mag0) and (dev_from_rf > 1.0)

    print("\n=== De-risk 5b verdict (AS-IS: expect the documented KILL) ===")
    print(f"initial unit phasor |z0| mean           : {mag0:.4f}")
    print(f"RF reference after 1 RF step  |z| mean   : {ref_mag_mean:.4f}  (preserved ~1 => RF rotates cleanly)")
    print(f"Izhikevich after 1 step       |z| mean   : {after_mag_mean:.4f}  (blown up => phasor destroyed)")
    print(f"max|Izh.v - before.v| (re moved)         : {re_changed:.3f}")
    print(f"max deviation Izh-state vs RF-rotation    : {dev_from_rf:.3f}")
    print(f"\n[5b] {'KILL CONFIRMED' if corrupted else 'UNEXPECTED'} — a phasor in v/u is destroyed by one "
          f"Izhikevich step.")
    print("[5b] => RF + Izhikevich CANNOT share v/u in one global step dispatch as-is; the protected sim/ edit "
          "(separate RF state arrays + per-neuron model mask + masked dual-dynamics step) is genuinely required.")
    print("[5b] edited-version PASS criterion: on a mixed bridge, the RF slice's phase read-back byte-matches "
          "this rf_reference, the Izh slice matches a pure-Izh control, and mixed-OFF is byte-identical to baseline.")

    result = {
        "derisk": "5b_rf_izh_coexistence_as_is",
        "backend": backend,
        "n": n, "period": period, "seed": args.seed,
        "mag0": mag0,
        "rf_ref_mag_mean": ref_mag_mean,
        "izh_after_mag_mean": after_mag_mean,
        "re_changed_max": re_changed,
        "dev_from_rf_max": dev_from_rf,
        "kill_confirmed": corrupted,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[5b] wrote {args.out}")
    # 5b is a KILL demo: 'success' = the corruption is confirmed (exit 0 when kill_confirmed).
    raise SystemExit(0 if corrupted else 1)


if __name__ == "__main__":
    main()
