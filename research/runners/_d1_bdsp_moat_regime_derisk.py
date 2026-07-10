import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
from sim.backend import get_backend, to_host
from sim.regions import BrainRegion, RegionPathway
xp, _ = get_backend()

def build(out_drive, couple, g, apical, seed=42):
    cfg = CoreSimConfig(); cfg.dt_ms = 1.0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [BrainRegion(name="inp", n_neurons=15, exc_fraction=1.0, internal_density=0.0),
                         BrainRegion(name="out", n_neurons=15, exc_fraction=1.0, internal_density=0.0)]
    cfg.region_pathways = [RegionPathway(from_region="inp", to_region="out", density=0.6, weight_mean=1.0, weight_jitter=0.1, plastic=True)]
    cfg.enable_bdsp = True; cfg.bdsp_learning_rate = 0.05; cfg.burst_isi_threshold_ms = 6.0; cfg.bdsp_p0 = 0.30
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.ou_std_current_pA = 0.0
    cfg.seed = seed; cfg.heterogeneity_seed = seed; cfg.ou_seed = seed; cfg.actual_seed_used = seed
    cfg.bdsp_apical_couples_soma = bool(couple); cfg.bdsp_apical_soma_g = float(g)
    br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(), viz_config=VisualizationConfig(), runtime_state=RuntimeState())
    br._initialize_simulation_data(); n = br.core_config.num_neurons; rm = br.region_manager
    inp = np.asarray(list(rm.indices("inp")), int); out = np.asarray(list(rm.indices("out")), int)
    drive = np.zeros(n, np.float32); drive[inp] = 900.0; drive[out] = out_drive
    br.cp_external_input_current = xp.asarray(drive)
    ap = np.zeros(n, np.float32); ap[out] = apical; br.cp_bdsp_apical_drive = xp.asarray(ap)
    w0 = np.array(np.asarray(to_host(br.cp_connections.data)))
    for _ in range(400): br._run_one_simulation_step()
    dw = float(np.abs(np.asarray(to_host(br.cp_connections.data)) - w0).sum())
    E = float(np.asarray(to_host(br.cp_bdsp_E[out])).mean()); B = float(np.asarray(to_host(br.cp_bdsp_B[out])).mean())
    return dw, E, B

print("does a SPARSER output-drive regime give B<E (burst a fraction of events) + a clean moat?")
print(f"  {'out_drive':>9} {'E_rest':>7} {'B_rest':>7} {'B/E':>6} | {'moat dw':>8} {'credit dw(g80)':>14} {'sep':>6}")
for od in (300, 450, 600, 700, 800):
    _, E, B = build(od, False, 0.0, 0.0)                    # rest rates
    m, _, _ = build(od, True, 80.0, 0.0)                    # moat (apical off, coupling on)
    c, _, _ = build(od, True, 80.0, 300.0)                  # credit (apical on, coupling on)
    print(f"  {od:>9} {E:>7.4f} {B:>7.4f} {B/max(E,1e-9):>6.2f} | {m:>8.3f} {c:>14.3f} {c/max(m,1e-9):>5.2f}x")
print()
print("if a drive gives B<E AND moat dw ~0 while credit dw stays high -> the moat is a REGIME issue (fixable), not a rule wall.")
