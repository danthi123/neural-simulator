"""Phase 1 (cheat B) ON-BRIDGE realization de-risk. The numpy argmax cleanup is replaced by the NEF-cleanup analogue
(the structure the rate composer's cleared cleanup used): a SPIKING concept-neuron bank whose firing rate is driven
by the matched-filter score Re(S* rec) (the phase-correlation -- the bridge's complex synapse matvec gives c=S* rec;
the real part is the cosine-sum score), then argmax-over-FIRING (a readout of the spiking output, exactly as the NEF
cleanup does). GATE: the spiking-bank winner == the numpy-argmax winner on the composer's REAL noisy unbinds,
multi-seed. The matvec (score) is == numpy by construction; this de-risks whether the SPIKING firing + readout
preserves the argmax despite spike noise.
"""
import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from sim.bridge import SimulationBridge
from sim.backend import to_host
from research.runners.rf_phasor_composer import RFPhasorComposer


def build_concept_bank(V, seed):
    cfg = CoreSimConfig()
    cfg.num_neurons = int(V)
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
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def spiking_bank_cleanup(rec_phases, codebook, bank, scale, window=120):
    """Matched-filter score (= the bridge's complex matvec real part) -> drive the spiking concept bank -> firing ->
    argmax-over-firing (the NEF-cleanup readout)."""
    words = list(codebook)
    S = np.stack([np.exp(2j * np.pi * codebook[w]) for w in words], axis=1)   # D x V
    c = S.conj().T @ np.exp(2j * np.pi * np.asarray(rec_phases))              # V complex scores (the matvec)
    scores = np.maximum(c.real, 0.0)                                          # rectified cosine-sum score (a current)
    import sim.backend as _b
    xp, _ = _b.get_backend()
    bank.cp_external_input_current[:] = xp.asarray(scores * scale, dtype=bank.cp_external_input_current.dtype)
    firing = np.zeros(len(words))
    for _ in range(window):
        bank._run_one_simulation_step()
        firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
    bank.cp_external_input_current[:] = 0.0
    return words[int(np.argmax(firing))], int(firing.sum())


def numpy_argmax_cleanup(rec_phases, codebook):
    words = list(codebook)
    sims = [float(np.mean(np.cos(2.0 * np.pi * (rec_phases - codebook[w])))) for w in words]
    return words[int(np.argmax(sims))]


def run(seed, D, scale):
    comp = RFPhasorComposer(seed=seed, D=D, period=200)
    comp.store("dog", "go", "north"); comp.store("cat", "run", "south"); comp.store("river", "look", "apple")
    bank = build_concept_bank(len(comp.concepts), seed)
    n = n_match = 0
    for (a, v, p), cph in zip([("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")],
                              [c for _, c in comp.kb]):
        for role in ("agent", "action", "patient"):
            rec = comp._unbind_phases(cph, role)
            w_np = numpy_argmax_cleanup(rec, comp.concepts)
            w_sp, _tot = spiking_bank_cleanup(rec, comp.concepts, bank, scale)
            n += 1
            n_match += int(w_sp == w_np)
    return n_match, n


if __name__ == "__main__":
    for scale in (20.0, 50.0, 100.0, 200.0):
        rows = []
        for seed in (42, 43, 44):
            m, n = run(seed, 256, scale)
            rows.append((seed, m, n))
        tot_m = sum(m for _, m, _ in rows); tot_n = sum(n for _, _, n in rows)
        print(f"scale={scale}: spiking-bank == argmax {tot_m}/{tot_n}  "
              + "  ".join(f"s{s}:{m}/{n}" for s, m, n in rows), flush=True)
