"""gap#1 fully-synaptic RF-phase transduction — cheap-first RUNG 1 (removes the last host read `rf_read_phases`).

The deployed RF encode reads the phase on the HOST (`rf_read_phases`) and charges cp_ssm_state. The fully-spiking form:
the RF spike DRIVES a downstream synapse whose DECAYING CONDUCTANCE, sampled at period-end, encodes the spike TIMING
(= the phase) — a biological latency/decay read, no host phase read. RUNG 1 asks the minimal question: does a
decaying-conductance read of the RF spike preserve the VALUE (monotonic + invertible in phase, high corr)? If yes, the
fully-synaptic transduction is feasible (the downstream neuron's conductance IS the value); RUNG 2 wires it on-bridge.

Mechanism: the RF neuron spikes at `cp_rf_spike_step` s in [0,period). A synapse pulse at s, decayed to period-end
(tau), gives g = exp(-(period - s)/tau) = a monotone function of the latency (period - s) = a monotone function of the
phase p = (period - s)/period. So g = exp(-p*period/tau) -- strictly monotone in p, hence invertible: the value is
RECOVERABLE from the decayed conductance by a fixed (log) read-out, NO host phase read. NO sim/ edit (reads public
cp_rf_spike_step; the on-bridge RUNG 2 uses a real conductance synapse)."""
import sys; sys.path.insert(0, "/home/dant123/Projects/sim")
import os; os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.enums import NeuronModel
from sim.backend import to_host


def build_rf(n, seed=42):
    cfg = CoreSimConfig(); cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = seed; cfg.dt_ms = 1.0; cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity", "enable_structural_plasticity",
              "enable_homeostasis", "enable_reward_modulation", "enable_watts_strogatz", "enable_neuromodulator_subsystem",
              "enable_brain_region_framework"):
        if hasattr(cfg, f): setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    b.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return b


PERIOD = 200
N = 128
VMIN, VMAX = 0.0, 24.0                      # the deployed inject range (relu(+/-v)/(1-decay))
b = build_rf(N, seed=42)
vals = np.linspace(VMIN, VMAX, N)
PLO, PHI = 0.05, 0.95
p = PLO + (PHI - PLO) * (vals - VMIN) / (VMAX - VMIN)
z = np.exp(1j * 2 * np.pi * p).astype(np.complex64)
b.rf_kick(z, period=PERIOD)
for _ in range(PERIOD + 8):
    b.cp_external_input_current[:] = 0.0
    b._run_one_simulation_step()

spike_step = np.asarray(to_host(b.cp_rf_spike_step)).astype(np.float64)   # in [0, period); the SPIKE TIMING
# --- the HOST phase read (the current deployed path, for reference) ---
p_host = ((PERIOD - spike_step) % PERIOD) / PERIOD

# --- the FULLY-SYNAPTIC decaying-conductance read: a synapse pulse at spike_step, decayed to period-end ---
# g = exp(-(period - spike_step)/tau) = a strictly-monotone function of the latency -> monotone in phase.
for tau in (60.0, 120.0, 240.0):
    latency = PERIOD - spike_step                                        # steps from the spike to period-end
    g = np.exp(-latency / tau)                                           # the decayed conductance a downstream neuron reads
    # recover the value by the FIXED inverse (a log read-out over the conductance -- a fixed synaptic nonlinearity):
    #   g = exp(-latency/tau) => latency = -tau*ln(g) => phase = (period-latency)/period => value
    lat_rec = -tau * np.log(np.clip(g, 1e-12, 1.0))
    p_rec = lat_rec / PERIOD                                             # phase == latency/PERIOD (rf_read_phases convention)
    v_rec = (np.clip(p_rec, PLO, PHI) - PLO) / (PHI - PLO) * (VMAX - VMIN) + VMIN
    corr_g = np.corrcoef(g, vals)[0, 1]                                  # is the raw conductance monotone in the value?
    corr_rec = np.corrcoef(v_rec, vals)[0, 1]                            # does the fixed inverse recover the value?
    rms = np.sqrt(np.mean((v_rec - vals) ** 2))
    # value-band bias (the M0 property: must stay unbiased)
    lo = (v_rec - vals)[vals <= VMIN + 0.2 * (VMAX - VMIN)]
    hi = (v_rec - vals)[vals >= VMIN + 0.8 * (VMAX - VMIN)]
    print(f"tau={tau:>5.0f}: corr(g,val)={corr_g:+.4f} (monotone if |corr|~1)  |  recovered corr={corr_rec:.4f} "
          f"rms={rms:.3f}  band-bias lo={lo.mean():+.3f} hi={hi.mean():+.3f}")

print(f"\nhost-phase-read corr (reference) = {np.corrcoef(p_host, vals)[0,1]:.4f}")
print("VERDICT: if the recovered corr ~ the host-read corr, a DECAYING-CONDUCTANCE latency read reproduces the value "
      "=> the fully-synaptic transduction is feasible (RUNG 2 = wire the RF spike -> a real conductance synapse on-bridge).")
