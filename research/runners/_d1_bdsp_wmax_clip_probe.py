import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
from sim.backend import get_backend, to_host
from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronModel
xp,_=get_backend()
def build(wmax, fw=40.0, in_hi=1200.0):
    cfg=CoreSimConfig(); cfg.dt_ms=1.0; cfg.enable_brain_region_framework=True
    cfg.neuron_model_type=NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name="GENERIC_UNSTRUCTURED"
    cfg.brain_regions=[BrainRegion(name="inp",n_neurons=20,exc_fraction=1.0,internal_density=0.0),
                       BrainRegion(name="hid",n_neurons=60,exc_fraction=1.0,internal_density=0.0)]
    cfg.region_pathways=[RegionPathway(from_region="inp",to_region="hid",density=1.0,weight_mean=fw,weight_jitter=0.3,plastic=True)]
    for f in ("enable_short_term_plasticity","enable_hebbian_learning","enable_homeostasis","enable_structural_plasticity","enable_reward_modulation","enable_stdp","enable_input_divisive_norm","enable_nmda"):
        setattr(cfg,f,False)
    cfg.enable_bdsp=True; cfg.bdsp_p0=0.30; cfg.burst_isi_threshold_ms=6.0; cfg.bdsp_learning_rate=0.0
    cfg.bdsp_w_max=wmax
    cfg.ou_std_current_pA=0.0; cfg.seed=42; cfg.heterogeneity_seed=42; cfg.ou_seed=42; cfg.actual_seed_used=42
    sb=SimulationBridge(core_config=cfg,gpu_config=GPUConfig(),viz_config=VisualizationConfig(),runtime_state=RuntimeState())
    sb._initialize_simulation_data(); rm=sb.region_manager
    inp=np.asarray(list(rm.indices("inp")),int); hid=np.asarray(list(rm.indices("hid")),int); n=sb.core_config.num_neurons
    w0=float(np.abs(np.asarray(to_host(sb.cp_connections.data))).mean())
    rng=np.random.RandomState(0); rates=[]
    for p in range(8):
        on=rng.choice(len(inp),len(inp)//2,replace=False)
        sb.cp_membrane_potential_v[:]=(sb.cp_izh_c_reset if getattr(sb,'cp_izh_c_reset',None) is not None else -65.0); sb.cp_recovery_variable_u[:]=0.0
        drive=np.zeros(n,np.float32); drive[inp[on]]=in_hi; hr=0.0
        for _ in range(60):
            sb.cp_external_input_current[:]=xp.asarray(drive); sb._run_one_simulation_step()
            hr+=float(np.asarray(to_host(sb.cp_firing_states[hid])).mean())
        rates.append(hr/60)
    w1=float(np.abs(np.asarray(to_host(sb.cp_connections.data))).mean())
    return np.array(rates),w0,w1
for wmax in (2.0, 50.0, 200.0):
    r,w0,w1=build(wmax); print(f"  bdsp_w_max={wmax:>6}: hidden std={r.std():.4f} mean={r.mean():.4f} | weight |w| {w0:.1f}->{w1:.1f}")
print("\nif raising bdsp_w_max preserves the forward weight (|w| stays ~40) AND the hidden fires input-dependently -> the fix.")
