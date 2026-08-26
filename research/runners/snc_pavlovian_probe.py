"""Spiking-SNc Stage A — Pavlovian de-risk harness (omission-dip falsifier).

Design: docs/plans/2026-06-08-spiking-snc-actor-critic-design.md §4.1 / §4.4.
Owner standard (CLAUDE.md "Standing standard: BRAIN-BASED ONLY"): the dopamine
reward-prediction error must be the SNc's FIRING, not a host formula. This
harness INSTRUMENTS that firing on a tiny 2-cue Pavlovian schedule, separate
from the nav runner, so the spiking RPE is verified directly.

What it tests (STAGE A scope)
-----------------------------
Stage A's value V is the HOST global reward EMA (an explicit scaffold; Stage B
replaces it with a neural striosome critic). The two canonical Schultz-1998
dopamine signatures are:
  (i)  CUE-SHIFT  — the burst migrates from the US to the CS as the CS acquires
       value. This REQUIRES a *state-dependent* value (the CS state must
       acquire value), so a host GLOBAL-EMA value CANNOT produce it. >>>
       OUT OF SCOPE for Stage A; it needs Stage B's neural critic. <<<
  (ii) OMISSION DIP — when an expected reward is WITHHELD, the SNc fires BELOW
       its tonic baseline at the expected-reward time. This needs only that
       the value is non-zero (reward was expected) AND that the DA rule is
       SIGNED (two-sided). The signed `from_region_firing_signed` rule (the one
       protected sim/ edit) is exactly what makes the dip possible — the
       pre-existing one-sided `from_region_firing` physically cannot dip.

So this harness is the STAGE-A falsifier: it must show the OMISSION DIP (the SNc
rate on omitted trials drops below the rewarded-then-baseline tonic). It also
runs a `--snc-probe` calibration sweep (§4.4) that confirms the SNc windowed
rate is MONOTONE in delta = r - V (burst on +RPE, tonic at 0, dip on -RPE)
BEFORE the gains are trusted as a teaching signal.

Everything here is the SAME mechanism the runner uses (the SNc pool driven via
cp_external_input_current; the rate read from cp_firing_states) so the probe
matches the deployed config (the project's "probes must match deployed config"
rule). CPU-friendly: runs under SIM_BACKEND=numpy.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners.snc_pavlovian_probe --seed 42
    SIM_BACKEND=numpy python -m research.runners.snc_pavlovian_probe --snc-probe
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _build_snc_bridge(seed: int, n_dopamine: int = 30, snc_tonic_pa: float = 220.0,
                      snc_da_sensitivity: float = 8.0):
    """Build a minimal bridge with the spiking `snc` pool + the signed
    `dopamine` modulator (the exact Stage-A wiring, no gridworld).

    The pool is IZH2007_DOPAMINE (slow tonic, bursts on phasic input). The
    dopamine modulator's production rule reads the snc firing via the new
    `from_region_firing_signed` rule so the concentration tracks the RPE.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import (
        NeuromodulatorConfig, ModulatorTarget, ProductionRule,
    )

    cfg = CoreSimConfig()
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = False
    # The silent SNc placeholder, exactly as the runner builds it (enums.py:665;
    # SNc lacks KCC2 -> E_inh override -55 mV).
    cfg.brain_regions = [
        BrainRegion(
            name="snc",
            n_neurons=n_dopamine,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
            syn_reversal_potential_i_override=-55.0,
        ),
    ]
    cfg.region_pathways = []
    # The Stage-A dopamine modulator: production = from_region_firing_signed
    # over ['snc'] (the protected sim/ edit). threshold = the tonic firing
    # FRACTION the windowed-rate EMA settles to at rest.
    snc_tonic_firing_fraction = 0.30
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine",
            baseline=0.5,
            decay_tau_ms=200.0,
            concentration_min=0.0,
            concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate",
                                     scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(
                rule_type="from_region_firing_signed",
                sensitivity=float(snc_da_sensitivity),
                threshold=float(snc_tonic_firing_fraction),
                window_ms=200.0,
                source_regions=["snc"],
            )],
        )
    ]

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _snc_indices(bridge):
    """Host int index array for the snc pool."""
    import numpy as np
    return np.asarray(bridge.region_manager.indices("snc"), dtype=np.int64)


def _drive_snc_and_count(bridge, snc_idx, I_snc, n_steps, xp):
    """Drive the snc pool with constant external current I_snc for n_steps,
    stepping the bridge (which also advances the dopamine concentration via the
    signed rule), and return (total_spikes, mean_da_conc). The rate is READ from
    cp_firing_states — measured from spikes, not a formula."""
    # Write the SNc drive once; the bridge does not reset cp_external_input_current
    # between steps here, so it persists across the window.
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[snc_idx] = xp.float32(I_snc)
    total = 0
    da_sum = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        total += int(bridge.cp_firing_states[snc_idx].sum())
        da_sum += float(bridge.neuromodulator_manager.get_concentration("dopamine"))
    return total, da_sum / max(n_steps, 1)


def run_calibration_probe(seed, *, snc_tonic_pa, snc_reward_gain, snc_value_gain,
                          snc_da_sensitivity, hold_steps, verbose=True):
    """§4.4 calibration: sweep (r, V) and report the SNc windowed rate so we can
    confirm it is MONOTONE in delta = r - V (burst on +RPE, tonic at 0, dip on
    -RPE). This MEASURES that the spiking RPE is well-formed rather than
    hand-asserting the gains."""
    from sim.backend import get_backend
    xp, _ = get_backend()

    bridge, _cfg = _build_snc_bridge(seed, snc_tonic_pa=snc_tonic_pa,
                                     snc_da_sensitivity=snc_da_sensitivity)
    snc_idx = xp.asarray(_snc_indices(bridge), dtype=xp.int64) \
        if hasattr(xp, "asarray") else _snc_indices(bridge)
    snc_idx_host = _snc_indices(bridge)
    snc_idx = xp.asarray(snc_idx_host)

    # (r, V) grid spanning the three regimes. delta = r - V.
    cases = [
        ("+RPE  (r=+1, V=0)", +1.0, 0.0),
        ("+RPE  (r=+1, V=0.5)", +1.0, 0.5),
        ("zero  (r=+1, V=1)", +1.0, 1.0),    # fully predicted -> delta 0
        ("tonic (r=0,  V=0)", 0.0, 0.0),     # nothing happening
        ("-RPE  (r=0,  V=1)", 0.0, 1.0),     # omission: expected but withheld
        ("-RPE  (r=-1, V=1)", -1.0, 1.0),    # worse than predicted (aversive)
    ]
    rows = []
    for label, r, V in cases:
        I_snc = (snc_tonic_pa
                 + snc_reward_gain * max(0.0, r)
                 - snc_value_gain * max(0.0, V))
        spikes, da = _drive_snc_and_count(bridge, snc_idx, I_snc, hold_steps, xp)
        rate_hz = spikes / max(len(snc_idx_host), 1) / (hold_steps * 1e-3)
        rows.append({"label": label, "r": r, "V": V, "delta": r - V,
                     "I_snc_pA": I_snc, "snc_spikes": spikes,
                     "snc_rate_hz": rate_hz, "da_conc": da})
        if verbose:
            print(f"  {label:24s} delta={r - V:+.1f}  I={I_snc:7.1f}pA  "
                  f"spikes={spikes:4d}  rate={rate_hz:6.2f}Hz  DA={da:.3f}")
    return rows


def run_pavlovian(seed, *, snc_tonic_pa, snc_reward_gain, snc_value_gain,
                  snc_da_sensitivity, hold_steps, n_train_trials,
                  verbose=True):
    """The Stage-A Pavlovian schedule. A CS predicts a US (reward) after a
    delay. The host EMA tracks the expected reward V (the Stage-A SCAFFOLD).
    We compare the SNc rate on:
      - REWARDED trials (US delivered): r=+1 at the expected time.
      - OMITTED trials  (US withheld):  r=0 at the expected time, but V>0
        (reward was expected) -> the SNc must DIP below tonic.

    Stage-A testable signature = the OMISSION DIP (omitted-trial SNc rate <
    rewarded-then-tonic baseline). The cue-shift (burst migrating CS<-US) is
    OUT OF SCOPE — it needs Stage B's state-dependent neural critic.
    """
    from sim.backend import get_backend
    xp, _ = get_backend()

    bridge, _cfg = _build_snc_bridge(seed, snc_tonic_pa=snc_tonic_pa,
                                     snc_da_sensitivity=snc_da_sensitivity)
    snc_idx_host = _snc_indices(bridge)
    snc_idx = xp.asarray(snc_idx_host)

    def snc_rate(spikes):
        return spikes / max(len(snc_idx_host), 1) / (hold_steps * 1e-3)

    # Host EMA value V (Stage-A scaffold), exactly the runner's reward_ema.
    reward_ema = 0.0
    ema_decay = 0.9  # ~tau 10 trials, matches the runner default

    rewarded_rates = []
    omitted_rates = []
    baseline_rates = []   # the SNc rate at the tonic floor (no reward window)

    # --- Acquisition: CS->US rewarded trials so V (=reward_ema) rises ---
    for t in range(n_train_trials):
        V = max(0.0, reward_ema)
        # Tonic / inter-trial: SNc at its pacemaker floor.
        b_spk, _ = _drive_snc_and_count(bridge, snc_idx, snc_tonic_pa, hold_steps, xp)
        baseline_rates.append(snc_rate(b_spk))
        # US delivered (r=+1): I = tonic + k_r*1 - k_v*V.
        r = 1.0
        I_snc = snc_tonic_pa + snc_reward_gain * max(0.0, r) - snc_value_gain * V
        spk, da = _drive_snc_and_count(bridge, snc_idx, I_snc, hold_steps, xp)
        rewarded_rates.append(snc_rate(spk))
        # Update the host EMA (V learns to predict the reward).
        reward_ema = ema_decay * reward_ema + (1 - ema_decay) * r
        if verbose and (t < 3 or t == n_train_trials - 1):
            print(f"  [acq t={t:02d}] V={V:.3f} rewarded_rate={snc_rate(spk):6.2f}Hz "
                  f"(baseline {snc_rate(b_spk):6.2f}Hz) DA={da:.3f}")

    V_learned = max(0.0, reward_ema)

    # --- Probe: OMISSION trials (US withheld, r=0) at the now-learned V ---
    n_probe = 5
    for t in range(n_probe):
        V = V_learned
        # Tonic baseline window (for the dip comparison).
        b_spk, _ = _drive_snc_and_count(bridge, snc_idx, snc_tonic_pa, hold_steps, xp)
        baseline_rates.append(snc_rate(b_spk))
        # Omission: r=0 but V>0 -> I = tonic - k_v*V (inhibition with no reward).
        r = 0.0
        I_snc = snc_tonic_pa + snc_reward_gain * max(0.0, r) - snc_value_gain * V
        spk, da = _drive_snc_and_count(bridge, snc_idx, I_snc, hold_steps, xp)
        omitted_rates.append(snc_rate(spk))
        if verbose:
            print(f"  [omit t={t:02d}] V={V:.3f} OMITTED_rate={snc_rate(spk):6.2f}Hz "
                  f"(baseline {snc_rate(b_spk):6.2f}Hz) DA={da:.3f}")

    import statistics as _st
    mean_rewarded = _st.mean(rewarded_rates) if rewarded_rates else 0.0
    mean_omitted = _st.mean(omitted_rates) if omitted_rates else 0.0
    mean_baseline = _st.mean(baseline_rates) if baseline_rates else 0.0

    # The Stage-A falsifier: the omitted-trial rate must DIP below the tonic
    # baseline (the SNc fires LESS than its spontaneous rate when an expected
    # reward is withheld). This is the omission/dip signature and it requires
    # the SIGNED rule + the inhibitory value drive.
    omission_dip = mean_omitted < mean_baseline
    # And the rewarded trials should BURST above tonic (sanity on the +branch).
    reward_burst = mean_rewarded > mean_baseline

    return {
        "seed": seed,
        "V_learned": V_learned,
        "mean_baseline_rate_hz": mean_baseline,
        "mean_rewarded_rate_hz": mean_rewarded,
        "mean_omitted_rate_hz": mean_omitted,
        "omission_dip": bool(omission_dip),
        "reward_burst": bool(reward_burst),
        "rewarded_rates_hz": rewarded_rates,
        "omitted_rates_hz": omitted_rates,
        "note_cue_shift": ("CUE-SHIFT is OUT OF SCOPE for Stage A (host "
                           "global-EMA value cannot produce it; needs Stage B's "
                           "state-dependent neural striosome critic)."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snc-tonic-pa", type=float, default=220.0)
    ap.add_argument("--snc-reward-gain", type=float, default=400.0)
    ap.add_argument("--snc-value-gain", type=float, default=400.0)
    ap.add_argument("--snc-da-sensitivity", type=float, default=8.0)
    ap.add_argument("--hold-steps", type=int, default=40,
                    help="Reward-hold window length in steps (wider = smoother "
                         "rate estimate on the small SNc pool).")
    ap.add_argument("--n-train-trials", type=int, default=20)
    ap.add_argument("--snc-probe", action="store_true",
                    help="Run the §4.4 calibration sweep (monotone-in-delta "
                         "check) instead of the Pavlovian schedule.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.snc_probe:
        print(f"[snc-probe seed={args.seed}] calibration sweep "
              f"(monotone-in-delta = r - V):")
        rows = run_calibration_probe(
            args.seed, snc_tonic_pa=args.snc_tonic_pa,
            snc_reward_gain=args.snc_reward_gain,
            snc_value_gain=args.snc_value_gain,
            snc_da_sensitivity=args.snc_da_sensitivity,
            hold_steps=args.hold_steps,
        )
        result = {"mode": "calibration_probe", "rows": rows}
    else:
        print(f"[snc-pavlovian seed={args.seed}] Stage-A omission-dip falsifier "
              f"(tonic={args.snc_tonic_pa}pA k_r={args.snc_reward_gain} "
              f"k_v={args.snc_value_gain}):")
        result = run_pavlovian(
            args.seed, snc_tonic_pa=args.snc_tonic_pa,
            snc_reward_gain=args.snc_reward_gain,
            snc_value_gain=args.snc_value_gain,
            snc_da_sensitivity=args.snc_da_sensitivity,
            hold_steps=args.hold_steps,
            n_train_trials=args.n_train_trials,
        )
        result["mode"] = "pavlovian"
        print()
        print(f"  baseline (tonic) rate : {result['mean_baseline_rate_hz']:6.2f} Hz")
        print(f"  rewarded (US)    rate : {result['mean_rewarded_rate_hz']:6.2f} Hz  "
              f"(burst above tonic: {result['reward_burst']})")
        print(f"  OMITTED          rate : {result['mean_omitted_rate_hz']:6.2f} Hz  "
              f"(DIP below tonic:   {result['omission_dip']})")
        print()
        verdict = "PASS" if result["omission_dip"] else "FAIL"
        print(f"  Stage-A OMISSION-DIP falsifier: {verdict} "
              f"(omitted {result['mean_omitted_rate_hz']:.2f}Hz "
              f"{'<' if result['omission_dip'] else '>='} "
              f"baseline {result['mean_baseline_rate_hz']:.2f}Hz)")
        print(f"  {result['note_cue_shift']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
