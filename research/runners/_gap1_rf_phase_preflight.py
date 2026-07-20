"""gap#1 RF PHASE pre-flight (M0 reframe): is the RF phase delivery of a continuous value UNBIASED across the value
range (symmetric error), unlike the rate code's value-dependent dead-zone? If yes -> corr ~0.82 suffices (M0 curve)
and the full RF encode is worth building. Cheapest decisive check; NO sim/ edit."""
import sys; sys.path.insert(0,"/home/dant123/Projects/sim")
import os; os.environ.setdefault("SIM_BACKEND","cupy")
import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.enums import NeuronModel
from sim.backend import to_host

def build_rf(n, seed=42):
    cfg = CoreSimConfig(); cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = seed; cfg.dt_ms = 1.0; cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp","enable_hebbian_learning","enable_short_term_plasticity","enable_structural_plasticity",
              "enable_homeostasis","enable_reward_modulation","enable_watts_strogatz","enable_neuromodulator_subsystem",
              "enable_brain_region_framework"):
        if hasattr(cfg,f): setattr(cfg,f,False)
    cfg.ou_std_current_pA = 0.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    b.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return b

PERIOD = 200
N = 128
b = build_rf(N, seed=42)
# encode: value v in [-3,3] -> phase p=(v+3)/6 in [0,1) -> complex kick z=exp(i*2*pi*p) (unit magnitude, value in PHASE)
VMIN, VMAX = -3.0, 3.0
vals = np.linspace(VMIN, VMAX, N)                          # a grid of test values, one per neuron
p = (vals - VMIN) / (VMAX - VMIN)
z = np.exp(1j * 2 * np.pi * p)
b.rf_kick(z, period=PERIOD)
for _ in range(PERIOD + 8):
    b.cp_external_input_current[:] = 0.0
    b._run_one_simulation_step()
p_read = b.rf_read_phases()                                # [N] in [0,1)
v_hat = p_read * (VMAX - VMIN) + VMIN
# circular error in value units (phase wraps)
err = v_hat - vals
err = (err + (VMAX-VMIN)/2) % (VMAX-VMIN) - (VMAX-VMIN)/2  # wrap to [-3,3]
print(f"RF phase reconstruction over {N} values in [{VMIN},{VMAX}], period={PERIOD}:")
print(f"  mean error (BIAS) = {err.mean():+.4f}   rms error = {np.sqrt((err**2).mean()):.4f}")
print(f"  corr(v_hat, v) = {np.corrcoef(v_hat, vals)[0,1]:.4f}")
# THE KEY: is the error VALUE-DEPENDENT (like the rate-code dead-zone) or symmetric/constant?
lo = err[vals < -1]; mid = err[np.abs(vals) <= 1]; hi = err[vals > 1]
print(f"\n  error by value band (VALUE-DEPENDENCE = the rate-code bias signature):")
print(f"    small |v|<=1 (dead-zone risk): mean {mid.mean():+.4f} rms {np.sqrt((mid**2).mean()):.4f}")
print(f"    v<-1 : mean {lo.mean():+.4f} rms {np.sqrt((lo**2).mean()):.4f}")
print(f"    v>1  : mean {hi.mean():+.4f} rms {np.sqrt((hi**2).mean()):.4f}")
_bias_spread = abs(mid.mean() - lo.mean()) + abs(mid.mean() - hi.mean())
print(f"\n  => bias-spread across bands = {_bias_spread:.4f}  ({'~SYMMETRIC/UNBIASED (greenlight RF)' if _bias_spread < 0.2 else 'VALUE-DEPENDENT (same failure mode as rate code)'})")
