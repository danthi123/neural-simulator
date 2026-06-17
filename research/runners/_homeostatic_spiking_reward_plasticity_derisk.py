"""Spiking homeostatic reward -> plasticity link de-risk — does the NEURAL drive-reduction reward drive
reward-modulated plasticity? This closes the last unvalidated link in the motivational core's learning path.

Established already:
  * the reward STRUCTURE is learnable (rate-proxy tabular Q, 6 seeds — 2026-06-17-homeostatic-drive-rl-cheap-first-GO.md);
  * the 2-pool SPIKING drive + the neural reward r = −Δ(hunger conc) work on real spikes (3 seeds —
    2026-06-17-homeostatic-spiking-drive-mechanism-GO.md);
  * reward-modulated STDP learning from a scalar `current_reward_signal` is validated project-wide (g9/g11/nav).

The one link left to verify directly: that the NEURAL drive-reduction reward, fed into `current_reward_signal`,
actually STRENGTHENS a co-active synapse (eligibility × reward -> Δw). Tested FUNCTIONALLY (no host weight read):
co-fire cue->motor (tags eligibility), apply the neural reward, then drive the cue ALONE -> does the motor fire
MORE (the pathway strengthened)? Rewarded vs unrewarded vs lesion.

GATE (>=3 seeds): the cue-evoked motor rate INCREASES after rewarded training (Δ > 0 by a clear margin) and the
increase is much larger than UNREWARDED (no drive-reduction -> r≈0) and LESION (drive frozen -> r≈0). ⇒ the
neural drive-reduction reward drives learning; the motivational core's full learning path is brain-based.

Run: SIM_BACKEND=numpy python -m research.runners._homeostatic_spiking_reward_plasticity_derisk --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def build_bridge(seed, n=60):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.15
    cfg.reward_eligibility_tau_ms = 500.0
    cfg.current_reward_signal = 0.0
    cfg.stdp_w_max = 30.0
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    cfg.brain_regions = [
        BrainRegion(name="cue", n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=rs),
        BrainRegion(name="motor", n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=rs),
        BrainRegion(name="agrp", n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=rs),
    ]
    # cue -> motor: plastic, reward-modulated. Small initial weight so a functional change is visible.
    cfg.region_pathways = [
        RegionPathway(from_region="cue", to_region="motor", density=0.6, weight_mean=20.0, weight_jitter=2.0,
                      plastic=True),
    ]
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="hunger", baseline=0.0, decay_tau_ms=100.0, concentration_min=0.0, concentration_max=3.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+0.0)],  # read-only here
            production_rules=[ProductionRule(rule_type="from_region_firing_signed", sensitivity=100.0,
                                             threshold=0.005, window_ms=100.0, source_regions=["agrp"])],
        )
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _idx(bridge, name):
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _conc(bridge):
    mgr = bridge.neuromodulator_manager
    try:
        return float(mgr.get_concentration("hunger"))
    except Exception:
        return 0.0


def _probe_cue_motor(bridge, cue, motor, xp, B, drive=400.0, steps=80):
    """Drive cue ALONE; return the motor firing fraction it evokes (the functional strength of cue->motor).
    STDP is FROZEN during the probe so the read-out itself does not modify the weights it is measuring."""
    was = bridge.core_config.enable_stdp
    bridge.core_config.enable_stdp = False
    m_spk = 0
    for _ in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[xp.asarray(cue)] = drive
        bridge._run_one_simulation_step()
        m_spk += int(B.to_host(bridge.cp_firing_states[xp.asarray(motor)]).sum())
    bridge.core_config.enable_stdp = was
    return m_spk / (len(motor) * steps)


def run_condition(seed, mode):
    """mode in {rewarded, unrewarded, lesion}. Returns (pre_motor_rate, post_motor_rate)."""
    import sim.backend as B
    xp, _ = B.get_backend()
    bridge, cfg = build_bridge(seed)
    cue, motor, agrp = _idx(bridge, "cue"), _idx(bridge, "motor"), _idx(bridge, "agrp")

    pre = _probe_cue_motor(bridge, cue, motor, xp, B)

    # TRAIN: repeat co-fire(cue+motor) [tags eligibility] -> set the neural reward -> let it apply.
    for _ in range(40):
        # 1) co-fire with cue clearly LEADING motor: drive cue alone for a few ms (cue spikes), THEN add motor so
        #    the motor spikes a few ms AFTER the cue -> delta_t = t_post - t_pre > 0 -> clean LTP (POSITIVE
        #    eligibility on cue->motor). A simultaneous drive (the earlier version) gave net-LTD timing.
        for _ in range(12):
            for _ in range(3):                        # cue leads
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[xp.asarray(cue)] = 400.0
                bridge._run_one_simulation_step()
            for _ in range(3):                        # motor follows (post lags pre -> LTP)
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[xp.asarray(cue)] = 400.0
                bridge.cp_external_input_current[xp.asarray(motor)] = 350.0
                bridge._run_one_simulation_step()
        # 2) compute the NEURAL reward as the drive-reduction. Hungry (agrp driven) -> conc up; then "eat"
        #    (agrp off) -> conc decays -> r = conc_before - conc_after > 0 (rewarded). For unrewarded/lesion, the
        #    agrp is never driven, so there is no drive to reduce -> r ≈ 0.
        if mode == "rewarded":
            for _ in range(250):                    # build the drive (hungry) -> a full hunger conc
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[xp.asarray(agrp)] = 400.0
                bridge._run_one_simulation_step()
            c_before = _conc(bridge)
            for _ in range(120):                    # eat: agrp off -> drive decays fully
                bridge.cp_external_input_current[:] = 0.0
                bridge._run_one_simulation_step()
            c_after = _conc(bridge)
            r = max(0.0, c_before - c_after)
        elif mode == "lesion":
            for _ in range(370):                    # drive frozen: agrp never driven -> no drive, no reduction
                bridge.cp_external_input_current[:] = 0.0
                bridge._run_one_simulation_step()
            r = 0.0
        else:  # unrewarded: same elapsed time, no reward
            for _ in range(370):
                bridge.cp_external_input_current[:] = 0.0
                bridge._run_one_simulation_step()
            r = 0.0
        # 3) apply the reward to the eligible (cue->motor) synapses. A fixed reward GAIN maps the small drive-
        #    reduction concentration delta to a usable reward signal (a reward sensitivity; the SIGN + presence are
        #    what the neural drive sets — gain is constant across conditions, so it cannot manufacture the contrast).
        cfg.current_reward_signal = float(r) * 8.0
        for _ in range(10):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()
        cfg.current_reward_signal = 0.0

    post = _probe_cue_motor(bridge, cue, motor, xp, B)
    return pre, post, r


def run_seed(seed):
    pre_r, post_r, r_used = run_condition(seed, "rewarded")
    pre_u, post_u, _ = run_condition(seed, "unrewarded")
    pre_l, post_l, _ = run_condition(seed, "lesion")
    d_rew = post_r - pre_r
    d_unr = post_u - pre_u
    d_les = post_l - pre_l
    out = {"seed": seed, "reward_used": r_used,
           "rewarded": {"pre": pre_r, "post": post_r, "delta": d_rew},
           "unrewarded": {"pre": pre_u, "post": post_u, "delta": d_unr},
           "lesion": {"pre": pre_l, "post": post_l, "delta": d_les}}
    out["go"] = bool(d_rew > 0.02 and d_rew >= 2.0 * max(abs(d_unr), abs(d_les), 1e-6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_homeostatic_spiking_reward_plasticity.json")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("[spiking homeostatic reward->plasticity] does the NEURAL drive-reduction reward strengthen a co-active "
          "cue->motor synapse?\n  GATE: cue-evoked motor rate rises after REWARDED training, >> unrewarded + lesion.\n",
          flush=True)
    results = []
    for seed in a.seeds:
        r = run_seed(seed)
        results.append(r)
        print(f"  [seed {seed}] reward r={r['reward_used']:.2f} | cue->motor rate Δ: rewarded "
              f"{r['rewarded']['delta']:+.3f} ({r['rewarded']['pre']:.3f}->{r['rewarded']['post']:.3f}) | "
              f"unrewarded {r['unrewarded']['delta']:+.3f} | lesion {r['lesion']['delta']:+.3f} || "
              f"{'GO' if r['go'] else 'NO'}", flush=True)

    n_go = sum(r["go"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if n_go == len(results):
        print(f"  GO ({n_go}/{len(results)} seeds): the NEURAL drive-reduction reward drives reward-modulated "
              "plasticity — a co-active cue->motor synapse STRENGTHENS (cue-evoked motor firing rises) ONLY when "
              "the intrinsic reward is delivered; unrewarded + lesion show no strengthening. ⇒ the motivational "
              "core's full learning path is brain-based on spikes: a self-generated homeostatic drive produces a "
              "neural reward that teaches the synapses. The artificial-life motivational core is de-risked end-to-end.",
              flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(results)} seeds): the neural reward does not cleanly drive the "
              "plasticity — localize (reward magnitude, eligibility timing/tau, learning rate, the co-fire window). "
              "Honest boundary.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
