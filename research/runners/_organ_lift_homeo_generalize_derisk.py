"""THROWAWAY de-risk (do NOT commit): does the per-region homeostasis enabler GENERALIZE from
the MINIMAL limbic organ to the FULL nav-validated reward/critic critic?

Context: 2026-06-18-merged-config-homeostasis-boundary-RESOLVED.md root-caused the merged-config
boundary to `enable_homeostasis`. The minimal limbic organ (cue->striosome->snc, +reward_us)
was lifted onto the merge by setting BrainRegion.enable_homeostasis=True per-region (the threshold
select bridge.py:6320-6323 gives ONLY masked neurons the ~-42mV threshold; the synaptic-scaling
foot-gun is gated by the SEPARATE cfg.enable_synaptic_scaling which stays OFF). The next direction
(AUTONOMOUS_STATE CYCLE 208) is to lift the FULL nav critic (build_bg_brain_regions
enable_neural_critic + spiking_reward_us) using this same enabler.

THE QUESTION: when build_bg_brain_regions builds the nav critic regions (striosome_value, snc,
reward_us, vs_place_context), and we give the critic-relevant regions enable_homeostasis=True while
keeping GLOBAL cfg.enable_homeostasis=False + cfg.enable_synaptic_scaling=False, does the SNc reach
its validated operating point (BURSTS on a reward_us drive >=3x tonic, and the striosome critic
FIRES so the GABA_B value subtraction can operate) — restored toward the global-homeostasis-on
standalone level vs the broken (homeostasis-off) state?

NOTE on a sharper edge than the prompt frames: in build_bg_brain_regions, the existing
`enable_critic_homeostasis` kwarg sets enable_homeostasis on the AFFERENT (vs_place_context) +
CRITIC (striosome_value) ONLY — NOT on snc or reward_us (g11_bg_runner.py:1230,1252,1283 vs
:1133-1142 snc, :1158-1163 reward_us). But the RESOLVED finding's f-I sweep showed the SNc is the
saturating operating point that benefits from homeostasis (the merged minimal organ set
enable_homeostasis=True on ALL 4 incl. limbic_snc, nav_conv_merged_bridge.py:591). So this de-risk
tests THREE homeostasis configs to isolate which regions need the mask:
  (A) NONE                          : the broken merged-default state (the boundary).
  (B) critic-only (the kwarg)       : enable_critic_homeostasis=True (afferent+critic).
  (C) critic + snc + reward_us      : (B) + post-hoc enable_homeostasis=True on snc, reward_us.

CPU-friendly. Run under SIM_BACKEND=numpy.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Region keys the FULL nav critic builds (with enable_neural_critic + spiking_reward_us, no
# neural_place_selforg / convergent_upstate -> the vs_place_context afferent path).
CRITIC_AFFERENT = "vs_place_context"


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def build_nav_critic(seed, *, homeo_config="none", n_cortex=100, upstate=False,
                     drive_w=12.0):
    """Build the FULL nav critic regions (build_bg_brain_regions enable_neural_critic +
    spiking_reward_us) into a bare merged-regime CoreSimConfig.

    homeo_config in {"none","critic","critic_snc_us"} selects the per-region homeostasis mask.
    upstate=True adds the convergent-excitation A1 arm (vs_place_drive) so the MSN-D1 critic can
    FIRE from its afferent (the init plastic weight 0.20 is too weak; the validated nav design
    grows it via STDP OR uses the up-state — the MSN-D1's vr=-80mV makes it hard to fire from a
    weak distributed afferent without one of these).
    Returns (bridge, cfg, region_names).
    """
    from research.runners.g11_bg_runner import build_bg_brain_regions
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.enums import NeuronModel
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    # build the FULL nav critic. enable_critic_homeostasis controls whether the builder sets
    # enable_homeostasis on the afferent+critic (the existing kwarg).
    crit_homeo = homeo_config in ("critic", "critic_snc_us")
    regions, pathways = build_bg_brain_regions(
        n_cortex=int(n_cortex),
        enable_neural_critic=True,
        spiking_reward_us=True,
        enable_critic_homeostasis=crit_homeo,
        enable_convergent_upstate=bool(upstate),
        vs_place_drive_to_value_weight=float(drive_w),
    )
    region_names = [r.name for r in regions]

    # (C) ALSO set enable_homeostasis on snc + reward_us (the saturating DA cell + its US afferent)
    # — the minimal merged organ set it on ALL 4 (incl. limbic_snc). This is a POST-HOC set on the
    # returned BrainRegion objects (so it works WITHOUT a build_bg_brain_regions kwarg).
    if homeo_config == "critic_snc_us":
        for r in regions:
            if r.name in ("snc", "reward_us"):
                r.enable_homeostasis = True

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    # MERGED REGIME (the regime under test): global homeostasis OFF, synaptic scaling OFF.
    cfg.enable_homeostasis = False
    cfg.enable_synaptic_scaling = False
    # heterogeneity OFF to match the merged bridge (it runs het off for nav/conv determinism; the
    # RESOLVED finding's BUG note: het-on silently overwrites per-region izh params -> snc would run
    # as jittered RS, not DOPAMINE). The merged bridge runs het off so limbic_snc is a correct DA.
    cfg.enable_parameter_heterogeneity = False

    # learning machinery (we won't train here — drive the critic directly — but keep it consistent).
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.reward_learning_rate = 0.0
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 40.0

    # GABA_B/GIRK (the resolved-finding merged params).
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.22
    cfg.gabab_conductance_max = 0.0

    # the SNc-derived dopamine modulator over ['snc'] (the merged builder pattern, source 'snc').
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [NeuromodulatorConfig(
        name="dopamine", baseline=0.5, decay_tau_ms=200.0, concentration_min=0.0, concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
        production_rules=[ProductionRule(rule_type="from_region_firing_signed", sensitivity=8.0,
                                         threshold=0.30, window_ms=200.0, source_regions=["snc"])])]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg, region_names


def _idx(bridge, name, xp):
    import numpy as np
    return xp.asarray(np.asarray(bridge.region_manager.indices(name), dtype=np.int64))


def _settle(bridge, xp, n_steps=80):
    bridge.cp_external_input_current[:] = 0.0
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * bridge.core_config.dt_ms


def _rate(bridge, idx, region_idx_map, drives, xp, n_steps=60):
    """Drive {region: pA}, step, return {region: Hz}."""
    bridge.cp_external_input_current[:] = 0.0
    for nm, pa in drives.items():
        bridge.cp_external_input_current[region_idx_map[nm]] = xp.float32(pa)
    counts = {nm: 0 for nm in region_idx_map}
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * bridge.core_config.dt_ms
        for nm, gi in region_idx_map.items():
            counts[nm] += int(_host(bridge.cp_firing_states[gi]).sum())
    dur_s = n_steps * 1e-3
    return {nm: counts[nm] / max(len(_host(gi)), 1) / dur_s for nm, gi in region_idx_map.items()}


def run_config(seed, homeo_config, *, snc_tonic_pa=160.0, us_drive_pa=400.0, cue_drive_pa=800.0,
               n_cortex=100, verbose=True):
    from sim.backend import get_backend
    xp, backend = get_backend()
    bridge, cfg, region_names = build_nav_critic(seed, homeo_config=homeo_config, n_cortex=n_cortex)
    if verbose:
        print(f"\n=== homeo_config={homeo_config!r} (seed {seed}, backend {backend}) ===")
        # which regions carry the homeostasis mask?
        masked = [r.name for r in bridge.region_manager.regions()
                  if getattr(r, "enable_homeostasis", False)]
        print(f"  regions with enable_homeostasis=True: {masked}")

    # OU on (the limbic op point was pinned with OU; merged sets it off for nav determinism but
    # re-enables for the limbic read — match that).
    cfg.enable_ou_process = True
    cfg.ou_std_current_pA = 100.0
    cfg.homeostasis_threshold_adapt_rate = 0.0   # freeze the homeostatic threshold during the probe

    region_idx_map = {nm: _idx(bridge, nm, xp)
                      for nm in ("snc", "striosome_value", "reward_us", CRITIC_AFFERENT)}

    def measure(drives):
        _settle(bridge, xp)
        return _rate(bridge, None, region_idx_map, drives, xp)

    # (1) SNc f-I: tonic vs reward_us-driven burst.
    base = measure({"snc": snc_tonic_pa})
    unpred = measure({"reward_us": us_drive_pa, "snc": snc_tonic_pa})
    burst_ratio = unpred["snc"] / max(base["snc"], 1e-6)

    # (2) striosome critic fires from its afferent (so GABA_B value subtraction can operate).
    crit = measure({CRITIC_AFFERENT: cue_drive_pa, "snc": snc_tonic_pa})
    strio_hz = crit["striosome_value"]

    # (3) value subtraction: drive critic afferent + reward_us together -> predicted burst < unpred.
    pred = measure({CRITIC_AFFERENT: cue_drive_pa, "reward_us": us_drive_pa, "snc": snc_tonic_pa})
    gap = unpred["snc"] / max(pred["snc"], 1e-6)

    if verbose:
        print(f"  SNc tonic={base['snc']:6.1f}Hz  reward_us-burst={unpred['snc']:6.1f}Hz "
              f"=> burst {burst_ratio:5.2f}x (>=3: {burst_ratio >= 3.0})")
        print(f"  striosome critic fires from {CRITIC_AFFERENT}: {strio_hz:6.1f}Hz "
              f"(reward_us afferent {unpred['reward_us']:.0f}Hz, fires: {strio_hz >= 1.0})")
        print(f"  value subtraction: unpred {unpred['snc']:.1f} vs predicted {pred['snc']:.1f} Hz "
              f"=> gap {gap:5.2f} (pred<unpred: {pred['snc'] < unpred['snc']})")
    return dict(homeo_config=homeo_config, seed=seed, region_names=region_names,
                snc_tonic_hz=base["snc"], snc_burst_hz=unpred["snc"], burst_ratio=burst_ratio,
                strio_hz=strio_hz, predicted_hz=pred["snc"], gap=gap,
                reward_us_hz=unpred["reward_us"], afferent_hz=crit[CRITIC_AFFERENT])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cortex", type=int, default=100)
    ap.add_argument("--snc-tonic-pa", type=float, default=160.0)
    ap.add_argument("--us-drive-pa", type=float, default=400.0)
    ap.add_argument("--cue-drive-pa", type=float, default=800.0)
    args = ap.parse_args()

    rk = dict(snc_tonic_pa=args.snc_tonic_pa, us_drive_pa=args.us_drive_pa,
              cue_drive_pa=args.cue_drive_pa, n_cortex=args.n_cortex)

    # first build prints the region names so we KNOW what the full nav critic builds.
    print("[organ-lift homeo generalize de-risk] building the FULL nav critic "
          "(enable_neural_critic=True, spiking_reward_us=True)")
    results = {}
    for hc in ("none", "critic", "critic_snc_us"):
        r = run_config(args.seed, hc, **rk)
        results[hc] = r

    print("\n\n========================= SUMMARY =========================")
    print("  full nav critic region names:")
    print(f"    {results['none']['region_names']}")
    print("\n  SNc burst (reward_us-driven / tonic) by homeostasis config:")
    for hc in ("none", "critic", "critic_snc_us"):
        r = results[hc]
        print(f"    {hc:16s}: tonic={r['snc_tonic_hz']:6.1f}  burst={r['snc_burst_hz']:6.1f}Hz  "
              f"ratio={r['burst_ratio']:5.2f}x  | striosome={r['strio_hz']:6.1f}Hz  "
              f"| value gap={r['gap']:5.2f}")

    # verdict: does the enabler RESTORE the SNc f-I (burst >=3x) AND make the critic fire?
    none_ratio = results["none"]["burst_ratio"]
    best = max(("critic", "critic_snc_us"), key=lambda h: results[h]["burst_ratio"])
    best_ratio = results[best]["burst_ratio"]
    best_strio = results[best]["strio_hz"]
    restored = best_ratio >= 3.0 and best_strio >= 1.0
    print(f"\n  VERDICT: homeostasis-off burst={none_ratio:.2f}x; best-with-homeostasis "
          f"({best})={best_ratio:.2f}x, striosome={best_strio:.1f}Hz")
    print(f"  => enabler {'RESTORES' if restored else 'does NOT restore'} the SNc f-I + critic firing "
          f"(GO: {restored})")


if __name__ == "__main__":
    main()
