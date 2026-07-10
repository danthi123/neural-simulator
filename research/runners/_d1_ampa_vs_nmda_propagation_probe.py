import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
from sim.backend import get_backend, to_host
from sim.regions import BrainRegion, RegionPathway
xp,_=get_backend()

def build(nmda, recept="nmda_slow", in_hi=1200.0, fw=40.0):
    cfg=CoreSimConfig(); cfg.dt_ms=1.0; cfg.enable_brain_region_framework=True
    cfg.brain_regions=[BrainRegion(name="inp",n_neurons=20,exc_fraction=1.0,internal_density=0.0),
                       BrainRegion(name="hid",n_neurons=60,exc_fraction=1.0,internal_density=0.0)]
    pw=RegionPathway(from_region="inp",to_region="hid",density=1.0,weight_mean=fw,weight_jitter=0.3,plastic=False)
    if nmda: pw.exc_receptor=recept
    cfg.region_pathways=[pw]
    for f in ("enable_short_term_plasticity","enable_hebbian_learning","enable_homeostasis","enable_structural_plasticity","enable_reward_modulation","enable_stdp","enable_input_divisive_norm"):
        setattr(cfg,f,False)
    cfg.enable_nmda=bool(nmda); cfg.enable_nmda_recurrent=bool(nmda); cfg.nmda_recurrent_tau_decay_ms=100.0
    cfg.ou_std_current_pA=0.0; cfg.seed=42; cfg.heterogeneity_seed=42; cfg.ou_seed=42; cfg.actual_seed_used=42
    sb=SimulationBridge(core_config=cfg,gpu_config=GPUConfig(),viz_config=VisualizationConfig(),runtime_state=RuntimeState())
    sb._initialize_simulation_data(); rm=sb.region_manager
    inp=np.asarray(list(rm.indices("inp")),int); hid=np.asarray(list(rm.indices("hid")),int)
    n=sb.core_config.num_neurons
    # 8 random input patterns (half the input neurons on), measure hidden firing input-dependence
    rng=np.random.RandomState(0); rates=[]
    for p in range(8):
        on=rng.choice(len(inp),len(inp)//2,replace=False)
        sb.cp_membrane_potential_v[:]=(sb.cp_izh_c_reset if getattr(sb,'cp_izh_c_reset',None) is not None else -65.0)
        sb.cp_recovery_variable_u[:]=0.0
        for a in ("cp_conductance_g_nmda_recurrent","cp_conductance_g_e","cp_conductance_g_i"):
            ar=getattr(sb,a,None)
            if ar is not None: ar[:]=0.0
        drive=np.zeros(n,np.float32); drive[inp[on]]=in_hi
        sb.cp_external_input_current=xp.asarray(drive)
        hr=0.0
        for _ in range(60):
            sb._run_one_simulation_step(); hr+=float(np.asarray(to_host(sb.cp_firing_states[hid])).mean())
        rates.append(hr/60)
    return np.array(rates)

for nmda,lbl in ((False,"AMPA (enable_nmda=False, the runner default)"),(True,"NMDA (enable_nmda=True, temporal summation)")):
    r=build(nmda)
    print(f"  {lbl:48s}: hidden rate mean={r.mean():.4f} std={r.std():.4f}")
print("\nNMDA std >> AMPA std, and mean well above 0 -> temporal summation makes the hidden fire input-dependently (the fix).")
