"""Spiking homeostatic-drive MECHANISM de-risk — the brain-based realization's low-risk half.

The cheapest-first GO (2026-06-17-homeostatic-drive-rl-cheap-first-GO.md, 6 seeds) de-risked the reward STRUCTURE
at the rate-proxy/algorithm level: an intrinsic drive-reduction reward is a learnable training signal. This is the
first BRAIN-BASED increment — the drive + reward on REAL SPIKES (the next, bigger increment is the spiking BG
cascade learning the policy from this neural reward).

THE BRAIN-BASED CLAIM TESTED HERE: a 2-pool SPIKING drive (hypothalamic AgRP=hunger / POMC=satiety; catalog
O.05/O.06) driven by the body's energy DEFICIT (an interoceptive current — the legitimate body→sensory boundary)
(1) encodes the deficit in its firing, and (2) a neuromodulator sourced from that firing via the EXISTING
`from_region_firing_signed` rule yields the intrinsic reward r = −Δ(drive concentration) — drive REDUCTION ->
positive reward (Keramati & Gutkin, eLife 2014). NO host distance/reward formula: r is read from spikes.

GATES (>=3 seeds, on REAL spikes):
  (1) corr(deficit, AgRP firing rate) >= +0.9          -- the spiking drive encodes the body's deficit.
  (2) drive = AgRP_rate - POMC_rate is monotone in deficit (push-pull: hunger up, satiety down with deficit).
  (3) eating (deficit drop) -> the hunger modulator DROPS -> r = -Δ(modulator) > 0 by a clear margin.
ANTI-CHEATS:
  * LESION the AgRP drive (zero the interoceptive current) -> AgRP firing ~0 -> the modulator floors -> r ~ 0
    (the reward rides the neural drive, not a host term).
  * r is read from the modulator concentration (driven by cp_firing_states), NOT any host deficit value.

Run: SIM_BACKEND=numpy python -m research.runners._homeostatic_spiking_drive_mechanism_derisk --seeds 42 43 44
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


def build_drive_bridge(seed, n_pool=60, da_sensitivity=100.0, tonic_frac=0.005):
    """A 2-pool spiking drive (agrp, pomc) + a `hunger` modulator sourced from agrp firing via
    from_region_firing_signed (the same proven path as the spiking SNc dopamine)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

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
    cfg.brain_regions = [
        BrainRegion(name="agrp", n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="pomc", n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
    ]
    cfg.region_pathways = []
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="hunger", baseline=0.0, decay_tau_ms=100.0, concentration_min=0.0, concentration_max=3.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(rule_type="from_region_firing_signed", sensitivity=float(da_sensitivity),
                                             threshold=float(tonic_frac), window_ms=100.0, source_regions=["agrp"])],
        )
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _drive_and_read(bridge, deficit, n_steps=200, i_scale=300.0, lesion=False):
    """Drive agrp ∝ deficit, pomc ∝ surplus(=1−deficit); run n_steps; return (agrp_rate, pomc_rate, hunger_conc).
    Resets the hunger concentration to baseline first so each measurement is INDEPENDENT (no carryover from a
    prior condition's decaying concentration — the residual that otherwise contaminates the lesion read)."""
    import sim.backend as B
    xp, _ = B.get_backend()
    rm = bridge.region_manager
    mgr0 = bridge.neuromodulator_manager
    if mgr0 is not None:
        try:
            mgr0._concentrations["hunger"] = 0.0   # clean start for an independent per-condition measurement
        except Exception:
            pass
    agrp = np.asarray(rm.indices("agrp"), dtype=np.int64)
    pomc = np.asarray(rm.indices("pomc"), dtype=np.int64)
    i_agrp = 0.0 if lesion else i_scale * max(0.0, deficit)
    i_pomc = i_scale * max(0.0, 1.0 - deficit)
    a_spikes = p_spikes = 0
    conc = 0.0
    for _ in range(n_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[xp.asarray(agrp)] = i_agrp
        bridge.cp_external_input_current[xp.asarray(pomc)] = i_pomc
        bridge._run_one_simulation_step()
        fs = bridge.cp_firing_states
        a_spikes += int(B.to_host(fs[xp.asarray(agrp)]).sum())
        p_spikes += int(B.to_host(fs[xp.asarray(pomc)]).sum())
    mgr = bridge.neuromodulator_manager
    if mgr is not None:
        try:
            conc = float(mgr.get_concentration("hunger"))
        except Exception:
            conc = 0.0
    a_rate = a_spikes / (len(agrp) * n_steps)
    p_rate = p_spikes / (len(pomc) * n_steps)
    return a_rate, p_rate, conc


def run_seed(seed):
    bridge = build_drive_bridge(seed)
    deficits = np.linspace(0.0, 1.0, 6)
    a_rates, p_rates, concs = [], [], []
    for d in deficits:
        a, p, c = _drive_and_read(bridge, float(d))
        a_rates.append(a); p_rates.append(p); concs.append(c)
    a_rates, p_rates, concs = np.array(a_rates), np.array(p_rates), np.array(concs)
    drive = a_rates - p_rates
    corr = float(np.corrcoef(deficits, a_rates)[0, 1]) if a_rates.std() > 1e-9 else 0.0
    monotone = bool(np.all(np.diff(drive) >= -1e-6))

    # eating: drive at high deficit (hungry) then drop to low deficit (fed); the hunger modulator must DROP -> r>0.
    _, _, c_hungry = _drive_and_read(bridge, 0.9)
    _, _, c_fed = _drive_and_read(bridge, 0.1)
    r_eat = c_hungry - c_fed                                   # intrinsic reward = drive (modulator) reduction

    # lesion: zero the agrp interoceptive drive -> firing ~0 -> modulator floors -> r ~ 0
    _, _, c_les_h = _drive_and_read(bridge, 0.9, lesion=True)
    _, _, c_les_f = _drive_and_read(bridge, 0.1, lesion=True)
    a_les, _, _ = _drive_and_read(bridge, 0.9, lesion=True)
    r_les = c_les_h - c_les_f

    out = {"seed": seed, "corr_deficit_agrp": corr, "drive_monotone": monotone,
           "agrp_rates": a_rates.tolist(), "drive_signal": drive.tolist(),
           "conc_hungry": c_hungry, "conc_fed": c_fed, "r_eat": r_eat,
           "agrp_rate_lesion": a_les, "r_lesion": r_les}
    out["check1_corr"] = corr >= 0.9
    out["check2_monotone"] = monotone
    out["check3_reward"] = r_eat > 0.2
    out["anti_lesion"] = (a_les < 0.02) and (abs(r_les) < 0.1)
    out["go"] = bool(out["check1_corr"] and out["check2_monotone"] and out["check3_reward"] and out["anti_lesion"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_homeostatic_spiking_drive_mechanism.json")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("[spiking homeostatic-drive mechanism] does a 2-pool SPIKING drive encode the deficit + yield the neural "
          "drive-reduction reward?\n  GATES: corr(deficit,AgRP firing)>=0.9 | drive monotone | eating->r>0 | "
          "lesion->silent+r~0.\n", flush=True)
    results = []
    for seed in a.seeds:
        r = run_seed(seed)
        results.append(r)
        print(f"  [seed {seed}] corr {r['corr_deficit_agrp']:+.2f} | monotone {r['drive_monotone']} | "
              f"eat: hungry-conc {r['conc_hungry']:.2f} -> fed {r['conc_fed']:.2f} (r {r['r_eat']:+.2f}) || "
              f"lesion: AgRP rate {r['agrp_rate_lesion']:.3f}, r {r['r_lesion']:+.2f} || "
              f"{'GO' if r['go'] else 'NO'}", flush=True)

    n_go = sum(r["go"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if n_go == len(results):
        print(f"  GO ({n_go}/{len(results)} seeds): a 2-pool SPIKING drive encodes the body's deficit (corr >=0.9), "
              "the push-pull drive signal is monotone in deficit, and eating (deficit drop) DROPS the hunger "
              "modulator -> a positive intrinsic reward read from spikes (no host term); lesioning the drive "
              "silences it and zeroes the reward. The brain-based drive + neural reward WORK on real spikes. "
              "⇒ next increment: the spiking BG cascade learning the policy from this neural reward (the full loop).",
              flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(results)} seeds): the spiking drive/reward mechanism does not robustly "
              "hold — localize (drive current scale, modulator sensitivity/threshold, tonic rate). Honest boundary.",
              flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
