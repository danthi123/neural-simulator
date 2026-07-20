"""gap#1 fully-synaptic RF transduction — RUNG 2 (on-bridge): the RF spike drives a REAL conductance synapse whose
decayed conductance encodes the value, ON the substrate (no host `rf_read_phases`). Confirms RUNG 1's math is produced
by a genuine synapse: encoder RF neurons resonate; a diagonal SLOW-NMDA synapse encoder_i->readout_i carries each RF
spike into readout_i's g_nmda; at period-end g_nmda[i] = w*exp(-(period-spike_step_i)/tau_nmda) = RUNG 1's decaying-
conductance read = monotone in the value. Reading g_nmda (a standard on-bridge conductance read, like reading
cp_ssm_state) + the fixed log inverse recovers the value. NO host phase read; NO sim/ edit (inject_explicit_wiring +
reads public cp_conductance_g_nmda)."""
import sys; sys.path.insert(0, "/home/dant123/Projects/sim")
import os; os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.enums import NeuronModel
from sim.backend import to_host

N = 64
PERIOD = 200
VMIN, VMAX = 0.0, 24.0
PLO, PHI = 0.05, 0.95

cfg = CoreSimConfig(); cfg.num_neurons = 2 * N
cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
cfg.seed = 42; cfg.dt_ms = 1.0; cfg.connections_per_neuron = 0; cfg.num_traits = 1
for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity", "enable_structural_plasticity",
          "enable_homeostasis", "enable_reward_modulation", "enable_watts_strogatz", "enable_neuromodulator_subsystem",
          "enable_brain_region_framework"):
    if hasattr(cfg, f): setattr(cfg, f, False)
cfg.ou_std_current_pA = 0.0
cfg.enable_nmda = True                                            # slow-NMDA synapse persists over the period
b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
b._initialize_simulation_data(called_from_playback_init=False)

# diagonal SLOW-NMDA synapse: encoder i (0..N) -> readout i (N..2N). The RF spike carries the value into the readout's g_nmda.
enc = np.arange(N); rdt = np.arange(N) + N
plan = {"enc2rdt": {"pre_indices": enc.tolist(), "post_indices": rdt.tolist(),
                    "initial_weights": [30.0] * N, "plastic": False, "conn_type": "enc2rdt",
                    "exc_receptors": ["nmda_slow"] * N}}
b.inject_explicit_wiring(plan)

# switch to RF + kick ONLY the encoders with the value-phases (readouts unkicked -> their g_nmda is purely synaptic)
b.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
vals = np.linspace(VMIN, VMAX, N)
p = PLO + (PHI - PLO) * (vals - VMIN) / (VMAX - VMIN)
z_full = np.zeros(2 * N, np.complex64); z_full[enc] = np.exp(1j * 2 * np.pi * p).astype(np.complex64)
mask = np.zeros(2 * N, bool); mask[enc] = True
try:
    b.rf_kick(z_full, period=PERIOD, neuron_mask=mask)
except TypeError:
    b.rf_kick(z_full, period=PERIOD)                             # older signature (no mask): kicks all; readouts get z=0
for _ in range(PERIOD + 8):
    b.cp_external_input_current[:] = 0.0
    b._run_one_simulation_step()

# READ the readout g_nmda (a standard on-bridge conductance read) -> the synaptically-transduced value
g_nmda = np.asarray(to_host(b.cp_conductance_g_nmda)).astype(np.float64)[rdt]
spike_step = np.asarray(to_host(b.cp_rf_spike_step)).astype(np.float64)[enc]     # for the host reference only
p_host = ((PERIOD - spike_step) % PERIOD) / PERIOD
corr_g = np.corrcoef(g_nmda, vals)[0, 1]
print(f"[RUNG2 on-bridge] readout g_nmda: nonzero={int((g_nmda>1e-9).sum())}/{N} range=[{g_nmda.min():.4f},{g_nmda.max():.4f}] "
      f"| corr(g_nmda, value) = {corr_g:+.4f}  (monotone if |corr|~1 => the SYNAPSE transduces the value)")

# recover the value from g_nmda by the fixed log inverse (biological log-compressive read-out). g = w*exp(-lat/tau).
if (g_nmda > 1e-9).sum() >= N // 2 and abs(corr_g) > 0.5:
    gg = np.clip(g_nmda / max(g_nmda.max(), 1e-9), 1e-9, 1.0)
    # fit the monotone map g->value by isotonic-free linear-in-log (tau + w unknown -> a 2-param fit is a fixed read-out)
    x = np.log(gg)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    v_rec = A @ coef
    corr_rec = np.corrcoef(v_rec, vals)[0, 1]; rms = np.sqrt(np.mean((v_rec - vals) ** 2))
    lo = (v_rec - vals)[vals <= VMIN + 0.2 * (VMAX - VMIN)]; hi = (v_rec - vals)[vals >= VMIN + 0.8 * (VMAX - VMIN)]
    print(f"  fixed log read-out recovers value: corr={corr_rec:.4f} rms={rms:.3f} (range {VMAX-VMIN}) "
          f"band-bias lo={lo.mean():+.3f} hi={hi.mean():+.3f}")
    go = corr_rec > 0.9 and rms < 0.15 * (VMAX - VMIN)
    print(f"\n  => RUNG 2 {'GO — the RF spike drives a REAL synapse whose conductance encodes the value ON-BRIDGE (no host phase read)' if go else 'partial — inspect (tune weight/tau/read)'}")
else:
    print("  => g_nmda not carrying the value (nonzero/corr too low) — inspect the synapse (weight, nmda persistence, mask/kick, decay over the period)")
print(f"  (host phase-read reference corr = {np.corrcoef(p_host, vals)[0,1]:.4f})")
