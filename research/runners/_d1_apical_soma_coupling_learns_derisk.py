"""The downstream gate: with the apical->soma coupling ON, does the committed FF rule now get DIRECTED credit?
A 2-region input->output net, plastic input->output, enable_bdsp + the coupling. Apical-ON (300pA on output) =
directed credit -> weights should move; apical-OFF (0) = the P0 moat -> weights should NOT move. Deterministic
(ou_std=0). The pre-edit boundary had this INVERTED (moat_smaller=False: apical-off moved weights MORE)."""
import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
from sim.backend import get_backend, to_host
from sim.regions import BrainRegion, RegionPathway
xp, _ = get_backend()


def learns(couple, g, apical_pA, seed=42):
    cfg = CoreSimConfig(); cfg.dt_ms = 1.0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [BrainRegion(name="inp", n_neurons=15, exc_fraction=1.0, internal_density=0.0),
                         BrainRegion(name="out", n_neurons=15, exc_fraction=1.0, internal_density=0.0)]
    cfg.region_pathways = [RegionPathway(from_region="inp", to_region="out", density=0.6,
                                         weight_mean=1.0, weight_jitter=0.1, plastic=True)]
    cfg.enable_bdsp = True; cfg.bdsp_learning_rate = 0.05; cfg.burst_isi_threshold_ms = 6.0; cfg.bdsp_p0 = 0.30
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.ou_std_current_pA = 0.0
    cfg.seed = seed; cfg.heterogeneity_seed = seed; cfg.ou_seed = seed; cfg.actual_seed_used = seed
    cfg.bdsp_apical_couples_soma = bool(couple); cfg.bdsp_apical_soma_g = float(g)
    br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(), viz_config=VisualizationConfig(), runtime_state=RuntimeState())
    br._initialize_simulation_data(); n = br.core_config.num_neurons
    rm = br.region_manager
    inp = np.asarray(list(rm.indices("inp")), int); out = np.asarray(list(rm.indices("out")), int)
    drive = np.zeros(n, np.float32); drive[inp] = 900.0; drive[out] = 800.0
    br.cp_external_input_current = xp.asarray(drive)
    ap = np.zeros(n, np.float32); ap[out] = apical_pA
    br.cp_bdsp_apical_drive = xp.asarray(ap)
    w0 = np.array(np.asarray(to_host(br.cp_connections.data)))
    for _ in range(400):
        br._run_one_simulation_step()
    w1 = np.array(np.asarray(to_host(br.cp_connections.data)))
    return float(np.abs(w1 - w0).sum())


for tag, couple, g in (("PRE-EDIT path (couple OFF)", False, 0.0), ("SURPASS (couple ON, g=40)", True, 40.0),
                       ("SURPASS (couple ON, g=80)", True, 80.0)):
    dw_credit = learns(couple, g, 300.0)     # apical ON -> directed credit
    dw_moat = learns(couple, g, 0.0)         # apical OFF -> the P0 moat
    ok = dw_credit > dw_moat + 1e-6
    print(f"  {tag:28s}: dw_credit={dw_credit:8.3f}  dw_moat={dw_moat:8.3f}  moat_RIGHT(credit>moat)={ok}")
print()
print("PRE-EDIT boundary: moat INVERTED (credit < moat). SURPASS goal: credit >> moat (apical-directed credit moves")
print("weights; apical-off = the P0 moat = ~no move). If moat_RIGHT=True with coupling on, the FF rule now learns directed.")
