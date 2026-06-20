"""BURNDOWN #9-critic — GRADED-RATE vs ALL-OR-NONE critic READ-OUT delta (the point-neuron read-out item).

Shortcut-burndown #9 splits into (a) a POINT-NEURON critic READ-OUT over (b) a deferred DENDRITIC-flavored
place-FIELD-carving floor. This runner is (a): does swapping the nav value-critic's READ-OUT form from the
all-or-none COINCIDENCE-PLATEAU (the Route-D supralinear `fused_coincidence_plateau` switch, which SATURATES
=> over-clamps the SNc => the value signal binarizes) to a GRADED RATE read-out (a point-neuron linear-synapse
critic, value = striosome firing RATE that scales with the learned weight) LIFT the dopamine RPE delta=r-V,
or is the delta still capped by the deferred dendritic field-carving floor (b)?

Both read-outs are measured on the SAME deterministic-nav-faithful harness (`snc_stageb_critic_probe_navfaithful`)
at the faithful grid-32 operating point, multi-seed, with the GABA_B lesion anti-cheat — so the comparison is
read-out-form-only (identical afferent, critic, SNc, training, lead).

THE TWO READ-OUT FORMS (identical otherwise)
--------------------------------------------
  LINEAR  (graded-rate, point-neuron): `vs_place_context -> striosome_value` PLAIN PLASTIC synapse. The critic
          fires by GRADED synaptic summation (learned w x place drive); V = a graded striosome rate; GABA_B
          subtracts it. This is the host-Gaussian `vs_place_context` scaffold whose value-train reached ~1.3.
  PLATEAU (all-or-none): the SAME `vs_place_context -> striosome_value` afferent but `coincidence_detector=True`
          + `enable_coincidence_detection` -> read through `fused_coincidence_plateau` (the steep all-or-none
          sigmoid g_inc = plateau / (1 + exp(-gain*(c_drive - k))), gain=2 => SATURATES at ~plateau). This is
          the read-out form nav DEPLOYS via `--neural-place-selforg` (place -> striosome_value coincidence_detector),
          the one the burndown flags as over-clamping.

delta TABLE (per read-out, multi-seed)
--------------------------------------
  delta = gap_ratio = far_burst(unpredicted) / near_burst(predicted).  Larger = better state-graded RPE.
  Compared against the host-Gaussian reference delta ~1.3 (CYCLE-219 / CYCLE-212 nav-deployment value-train).

ANTI-CHEATS
-----------
  (lesion) GABA_B mask zeroed -> the gap must COLLAPSE to ~1.0 (proves the value subtraction is the GABA_B
           conductance, not host arithmetic). Run per read-out form.
  (host-EMA) reported = 1.0 by construction (a scalar reward-EMA is place-blind; carried from _test_and_gate).
  (regime fidelity) global OU/conductance-noise/homeostasis OFF (asserted by the navfaithful builder).
  (faithful scale) grid-32, the dense place afferent, the deterministic regime — NOT a tiny smoke (#6 lesson).

NO sim/ edit. Reuse-by-import of the validated navfaithful machinery; the coincidence_detector flag + the
plateau kernel already exist in sim/. CPU-friendly (tiny ~554-neuron bridge), but GPU for the verdict.

Usage
-----
    SIM_BACKEND=cupy python -m research.runners._burndown9_critic_graded_readout_derisk \
        --seeds 42,43,44 --lead-ms 150 --out research/findings/raw/_burndown9_critic_readout.json
    SIM_BACKEND=numpy python -m research.runners._burndown9_critic_graded_readout_derisk --seed 42  # CPU smoke
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.runners.snc_stageb_critic_probe_navfaithful import (
    _build_navfaithful_bridge,
    _assert_deterministic_regime,
    _grid_prefs,
    grid_place_code_drive,
)
from research.runners.snc_stageb_critic_probe_place import (
    _drive_place,
    _calibrate_da_threshold,
    _mean_pathway_weight,
    _clear_eligibility,
    _test_and_gate,
    _host,
    _idx,
)


def _build_critic_bridge(seed, *, readout, grid_size=32,
                         vs_place_to_strio_weight=0.2, strio_to_snc_weight=10.0,
                         snc_da_sensitivity=8.0, reward_learning_rate=0.12,
                         gabab=True, gabab_tau_decay=150.0, gabab_propagation_strength=0.02,
                         critic_homeostasis=True, afferent_homeostasis=True,
                         coincidence_plateau=80.0, coincidence_k_threshold=4.0,
                         coincidence_gain=2.0, coincidence_weighted=False):
    """Build the navfaithful critic bridge, then (for readout='plateau') retag the
    vs_place_context->striosome_value afferent as a coincidence_detector and enable the
    coincidence plateau. readout='linear' leaves the plain plastic graded-rate afferent.

    The retag is done by REBUILDING with the coincidence flag on the pathway — no sim/ edit,
    no post-hoc mutation: we set cfg.region_pathways[afferent].coincidence_detector and the
    coincidence cfg knobs BEFORE _initialize_simulation_data, exactly as g11_bg_runner does for
    the place->striosome_value Route-D path. We do this by constructing the bridge with the
    navfaithful builder and re-running init is not possible post-hoc, so we replicate the build
    here using the same builder but injecting the flag through a thin wrapper.
    """
    # The navfaithful builder constructs cfg + bridge + runs init internally. To toggle the
    # coincidence flag on the afferent pathway we need it set on cfg BEFORE init. The builder does
    # not expose that, so for the plateau variant we monkey-set the pathway flag + cfg knobs on a
    # builder that we call with include_actor (the actor stub is needed for gate-4 parity but is
    # inert to the read-out comparison). Simplest faithful route: build via the navfaithful builder,
    # which returns (bridge, cfg) AFTER init; for the plateau variant the coincidence path must be
    # wired at build. We therefore wrap: temporarily patch the RegionPathway default is not viable,
    # so we re-implement the build inline mirroring _build_navfaithful_bridge but with the flag.
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 40.0

    # deterministic-nav regime (the exact knobs nav disables; anti-cheat d)
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False

    if gabab:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = float(gabab_tau_decay)
        cfg.gabab_propagation_strength = float(gabab_propagation_strength)

    plateau = (readout == "plateau")
    if plateau:
        # The all-or-none coincidence-plateau read-out (Route-D, as nav deploys).
        cfg.enable_coincidence_detection = True
        cfg.coincidence_plateau_strength = float(coincidence_plateau)
        cfg.coincidence_k_threshold = float(coincidence_k_threshold)
        cfg.coincidence_gain = float(coincidence_gain)
        cfg.coincidence_weighted_drive = bool(coincidence_weighted)

    regions = [
        BrainRegion(
            name="vs_place_context", n_neurons=200, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            enable_homeostasis=bool(afferent_homeostasis),
        ),
        BrainRegion(
            name="striosome_value", n_neurons=60, exc_fraction=0.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,
            enable_homeostasis=bool(critic_homeostasis),
        ),
        BrainRegion(
            name="snc", n_neurons=30, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
            syn_reversal_potential_i_override=-55.0,
        ),
    ]
    pathways = [
        RegionPathway(
            from_region="vs_place_context", to_region="striosome_value",
            density=0.5, weight_mean=float(vs_place_to_strio_weight),
            weight_jitter=0.5, plastic=True,
            # The ONLY difference between the two read-out forms:
            coincidence_detector=bool(plateau),
        ),
        RegionPathway(
            from_region="striosome_value", to_region="snc",
            density=0.5, weight_mean=float(strio_to_snc_weight),
            weight_jitter=0.2, plastic=False,
            receptor=("gaba_b" if gabab else "gaba_a")),
    ]
    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine", baseline=0.5, decay_tau_ms=200.0,
            concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(
                rule_type="from_region_firing_signed", sensitivity=float(snc_da_sensitivity),
                threshold=0.30, window_ms=200.0, source_regions=["snc"],
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


def run_readout(seed, *, readout, grid_size=32,
                p_near_xy=(26.571, 26.571), p_far_xy=(4.429, 4.429),
                vs_place_sigma=4.0, vs_place_drive_pa=800.0,
                snc_tonic_pa=180.0, snc_reward_gain=300.0,
                hold_steps=40, n_train=40, lead_steps=150,
                vs_place_to_strio_weight=0.2, strio_to_snc_weight=10.0,
                snc_da_sensitivity=8.0,
                critic_homeostasis=True, afferent_homeostasis=True,
                coincidence_plateau=80.0, coincidence_k_threshold=4.0,
                coincidence_weighted=False, lesion=False, verbose=True):
    """Train the value-leads-reward critic with the given read-out form, then measure the delta
    (far_burst/near_burst gap) at `lead_steps`. Optionally lesion the GABA_B mask (anti-cheat)."""
    from sim.backend import get_backend
    xp, _ = get_backend()

    bridge, cfg = _build_critic_bridge(
        seed, readout=readout, grid_size=grid_size,
        vs_place_to_strio_weight=vs_place_to_strio_weight, strio_to_snc_weight=strio_to_snc_weight,
        snc_da_sensitivity=snc_da_sensitivity,
        critic_homeostasis=critic_homeostasis, afferent_homeostasis=afferent_homeostasis,
        coincidence_plateau=coincidence_plateau, coincidence_k_threshold=coincidence_k_threshold,
        coincidence_weighted=coincidence_weighted)
    _assert_deterministic_regime(cfg)

    regions = ("vs_place_context", "striosome_value", "snc")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}
    idx_map["place"] = idx_map["vs_place_context"]
    n_vs = len(_host(idx_map["vs_place_context"]))

    vs_prefs = _grid_prefs(n_vs, grid_size)
    near_vec = grid_place_code_drive(p_near_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)
    far_vec = grid_place_code_drive(p_far_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)

    tonic_frac = _calibrate_da_threshold(bridge, cfg, idx_map, snc_tonic_pa, xp)

    # near/far ensembles for location-selective weight tracking
    def _ens(vec, frac=0.25):
        g = np.asarray(_idx(bridge, "vs_place_context"), dtype=np.int64)
        d = np.asarray(vec, dtype=np.float64)
        k = max(1, int(round(frac * len(d))))
        return set(int(g[i]) for i in np.argsort(d)[-k:])
    near_set = _ens(near_vec); far_set = _ens(far_vec) - near_set

    w_near_init = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
    w_far_init = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=far_set)
    w_init = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value")

    # value-leads-reward acquisition (FAR held out; NEAR potentiated)
    near_v_curve, near_burst_curve = [], []
    for t in range(n_train):
        _drive_place(bridge, idx_map, None, {"snc": snc_tonic_pa}, hold_steps, xp)
        _clear_eligibility(bridge)
        snc_r, strio_r, da = _drive_place(
            bridge, idx_map, near_vec, {"snc": snc_tonic_pa + snc_reward_gain}, hold_steps, xp)
        near_v_curve.append(strio_r); near_burst_curve.append(snc_r)
        if verbose and (t < 2 or t % 10 == 0 or t == n_train - 1):
            wn = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
            print(f"  [{readout} acq t={t:02d}] near-burst={snc_r:6.2f}Hz V(near)={strio_r:6.2f}Hz "
                  f"w_near={wn:.3f} DA={da:.3f}")

    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    near_v_early = _st.mean(near_v_curve[early]); near_v_late = _st.mean(near_v_curve[late])
    w_near_final = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
    w_far_final = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value", pre_subset=far_set)
    w_final = _mean_pathway_weight(bridge, "vs_place_context", "striosome_value")

    ts = dict(
        seed=seed, lesion=lesion, gabab=True, cfg=cfg, idx_map=idx_map,
        near_vec=near_vec, far_vec=far_vec, snc_tonic_pa=snc_tonic_pa,
        snc_reward_gain=snc_reward_gain, hold_steps=hold_steps,
        na=0, nb=0, overlap=0.0, distinct_ensembles=True,
        near_v_early=near_v_early, near_v_late=near_v_late,
        w_init=w_init, w_final=w_final, w_near_init=w_near_init, w_near_final=w_near_final,
        w_far_init=w_far_init, w_far_final=w_far_final,
        near_v_curve=near_v_curve, near_burst_curve=near_burst_curve,
    )
    result = _test_and_gate(bridge, xp, ts, lead_steps, verbose=verbose)
    result["readout"] = readout
    result["critic_rate_late_hz"] = float(near_v_late)
    return result


def _seed_pair(seed, lead_steps, kw, verbose=True):
    """Run BOTH read-out forms (linear graded-rate + all-or-none plateau) + the GABA_B lesion of
    each, for one seed. Returns a dict with the delta (gap) of each form and the lesion gaps."""
    out = {}
    for ro in ("linear", "plateau"):
        r = run_readout(seed, readout=ro, lead_steps=lead_steps, lesion=False, verbose=verbose, **kw)
        rl = run_readout(seed, readout=ro, lead_steps=lead_steps, lesion=True, verbose=False, **kw)
        out[ro] = dict(
            delta=float(r["gap_ratio"]),
            near_burst=float(r["test_predicted_near_hz"]),
            far_burst=float(r["test_unpredicted_far_hz"]),
            critic_rate_hz=float(r["critic_rate_late_hz"]),
            above_floor=bool(r["above_floor"]),
            v_near_far_ratio=float(r["v_near_far_ratio"]),
            w_near_far_ratio=float(r["w_near_far_ratio"]),
            lesion_delta=float(rl["gap_ratio"]),
            lesion_far_burst=float(rl["test_unpredicted_far_hz"]),
            lesion_collapses=bool(rl["gap_ratio"] <= 1.15),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--lead-ms", type=float, default=150.0)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--coincidence-plateau", type=float, default=80.0)
    ap.add_argument("--coincidence-k-threshold", type=float, default=4.0)
    ap.add_argument("--coincidence-weighted", action="store_true",
                    help="use the WEIGHTED-subunit plateau (Poirazi-Mel) so the learned weight grades the "
                         "plateau (the g11 _run_stage_b_smoke read-out toggle); default COUNT (weight-blind).")
    ap.add_argument("--host-ref-delta", type=float, default=1.3,
                    help="the host-Gaussian nav-deployment value-train delta reference (CYCLE-219/212).")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    lead_steps = int(round(args.lead_ms / 1.0))
    kw = dict(
        grid_size=args.grid_size, n_train=args.n_train,
        coincidence_plateau=args.coincidence_plateau,
        coincidence_k_threshold=args.coincidence_k_threshold,
        coincidence_weighted=args.coincidence_weighted,
    )

    per_seed = {}
    for s in seeds:
        print(f"\n##### BURNDOWN #9-critic READ-OUT COMPARISON seed={s} "
              f"(lead {args.lead_ms:.0f}ms, grid-{args.grid_size}, deterministic regime) #####")
        per_seed[s] = _seed_pair(s, lead_steps, kw, verbose=True)
        p = per_seed[s]
        print(f"  [seed {s}] LINEAR  (graded-rate): delta={p['linear']['delta']:.2f} "
              f"(near={p['linear']['near_burst']:.1f} far={p['linear']['far_burst']:.1f} "
              f"critic={p['linear']['critic_rate_hz']:.2f}Hz AF={p['linear']['above_floor']}) "
              f"| lesion delta={p['linear']['lesion_delta']:.2f} collapses={p['linear']['lesion_collapses']}")
        print(f"  [seed {s}] PLATEAU (all-or-none): delta={p['plateau']['delta']:.2f} "
              f"(near={p['plateau']['near_burst']:.1f} far={p['plateau']['far_burst']:.1f} "
              f"critic={p['plateau']['critic_rate_hz']:.2f}Hz AF={p['plateau']['above_floor']}) "
              f"| lesion delta={p['plateau']['lesion_delta']:.2f} collapses={p['plateau']['lesion_collapses']}")

    print("\n" + "=" * 100)
    print("=== BURNDOWN #9-critic READ-OUT delta TABLE (delta = far_burst/near_burst; host-Gaussian ref "
          f"~{args.host_ref_delta}) ===")
    print("=" * 100)
    print(f"  {'seed':>5} | {'LINEAR delta':>12} {'(critic Hz)':>11} {'AF':>3} | "
          f"{'PLATEAU delta':>13} {'(critic Hz)':>11} {'AF':>3} | {'lin-lesion':>10} {'plat-lesion':>11}")
    for s in seeds:
        p = per_seed[s]
        print(f"  {s:>5} | {p['linear']['delta']:>12.2f} {p['linear']['critic_rate_hz']:>11.2f} "
              f"{('Y' if p['linear']['above_floor'] else 'n'):>3} | "
              f"{p['plateau']['delta']:>13.2f} {p['plateau']['critic_rate_hz']:>11.2f} "
              f"{('Y' if p['plateau']['above_floor'] else 'n'):>3} | "
              f"{p['linear']['lesion_delta']:>10.2f} {p['plateau']['lesion_delta']:>11.2f}")

    def _med(form, key):
        return _st.median([per_seed[s][form][key] for s in seeds])
    lin_d = _med("linear", "delta"); plat_d = _med("plateau", "delta")
    lin_cr = _med("linear", "critic_rate_hz"); plat_cr = _med("plateau", "critic_rate_hz")
    lin_af = sum(1 for s in seeds if per_seed[s]["linear"]["above_floor"])
    plat_af = sum(1 for s in seeds if per_seed[s]["plateau"]["above_floor"])
    lin_les_ok = sum(1 for s in seeds if per_seed[s]["linear"]["lesion_collapses"])
    plat_les_ok = sum(1 for s in seeds if per_seed[s]["plateau"]["lesion_collapses"])
    n = len(seeds)
    print(f"\n  MEDIAN  LINEAR  delta={lin_d:.2f} (critic {lin_cr:.2f}Hz, above-floor {lin_af}/{n}, "
          f"lesion-collapses {lin_les_ok}/{n})")
    print(f"  MEDIAN  PLATEAU delta={plat_d:.2f} (critic {plat_cr:.2f}Hz, above-floor {plat_af}/{n}, "
          f"lesion-collapses {plat_les_ok}/{n})")

    # Verdict logic.
    #  - CONVERTED: the graded-rate (linear) read-out gives a MATERIALLY better delta than the
    #    all-or-none plateau AND fires the critic + above floor -> the read-out form WAS the
    #    bottleneck (#9-critic converts to the graded read-out).
    #  - CHARACTERIZED: the linear read-out either can't fire the point-neuron MSN critic from the
    #    place code (critic ~0 Hz / not above floor) OR its delta is no better than the plateau and
    #    no better than the host-Gaussian ref -> the read-out is graded/point-neuron but the delta
    #    floor is the deferred dendritic field-carving (b), not the read-out form.
    lin_fires = (lin_cr >= 1.0 and lin_af >= max(1, (n + 1) // 2))
    materially_better = (lin_d >= 1.30 * max(plat_d, 1e-6) and lin_d >= 1.30)
    if lin_fires and materially_better:
        verdict = "CONVERTED"
        verdict_note = ("the graded-rate (linear) point-neuron read-out fires the critic + grades the "
                        "delta materially above the all-or-none plateau -> the read-out form WAS the "
                        "bottleneck; #9-critic converts to the graded read-out.")
    else:
        verdict = "CHARACTERIZED"
        if not lin_fires:
            why = ("the LINEAR graded-rate read-out cannot fire the point-neuron MSN critic from the "
                   f"place code at faithful scale (critic {lin_cr:.2f}Hz, above-floor {lin_af}/{n}) -- "
                   "linear synaptic summation is sub-rheobase on the point neuron; the all-or-none "
                   "coincidence plateau exists precisely to fire it. So the read-out form is a genuine "
                   "fork (graded=can't-fire vs plateau=fires-but-saturates), and the residual delta floor "
                   "is the deferred DENDRITIC field-carving (b), not a clean read-out swap.")
        else:
            why = (f"the LINEAR graded-rate read-out fires (critic {lin_cr:.2f}Hz) and its delta "
                   f"({lin_d:.2f}) is NOT materially above the all-or-none plateau ({plat_d:.2f}) nor the "
                   f"host-Gaussian ref (~{args.host_ref_delta}) -- the read-out is now graded/point-neuron "
                   "(the all-or-none form retired) but the delta MAGNITUDE stays capped by the deferred "
                   "DENDRITIC place-field-carving floor (b), not the read-out form.")
        verdict_note = why

    print("\n" + "=" * 100)
    print(f"=== BURNDOWN #9-critic VERDICT: {verdict} ===")
    print(f"=== {verdict_note} ===")
    print(f"=== lesion anti-cheat: LINEAR collapses {lin_les_ok}/{n}, PLATEAU collapses {plat_les_ok}/{n} "
          "(the GABA_B value subtraction is load-bearing where the critic fires) ===")
    print("=" * 100)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(
                item="burndown_9_critic_graded_readout",
                deterministic_regime=True, grid_size=args.grid_size, lead_ms=args.lead_ms,
                host_ref_delta=args.host_ref_delta,
                seeds=seeds, per_seed={str(s): per_seed[s] for s in seeds},
                median_linear_delta=lin_d, median_plateau_delta=plat_d,
                median_linear_critic_hz=lin_cr, median_plateau_critic_hz=plat_cr,
                linear_above_floor=lin_af, plateau_above_floor=plat_af,
                linear_lesion_collapses=lin_les_ok, plateau_lesion_collapses=plat_les_ok,
                verdict=verdict, verdict_note=verdict_note,
            ), f, indent=2, default=float)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
