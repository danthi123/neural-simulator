"""DETERMINISTIC re-measurement (OU noise OFF) of: (1) byte-identity (flag-on g=0 == flag-off, exactly);
(2) the boundary (is B really flat vs apical on the pure enable_bdsp path?); (3) the surpass (does the coupling
raise B + hold the moat). The D1 Stage-A detector was nondeterministic (unseeded OU noise, +-0.09 on B); pinning
ou_std=0 makes it exact.
"""
import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
from sim.backend import get_backend, to_host
xp, _ = get_backend()


def detector(couple, g, seed=42):
    cfg = CoreSimConfig(); cfg.num_neurons = 40; cfg.dt_ms = 1.0
    cfg.enable_bdsp = True; cfg.burst_isi_threshold_ms = 6.0; cfg.bdsp_p0 = 0.30
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.ou_std_current_pA = 0.0                      # <-- kill the OU noise -> deterministic
    cfg.seed = seed; cfg.heterogeneity_seed = seed; cfg.ou_seed = seed; cfg.actual_seed_used = seed
    cfg.bdsp_apical_couples_soma = bool(couple); cfg.bdsp_apical_soma_g = float(g)
    br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(), viz_config=VisualizationConfig(), runtime_state=RuntimeState())
    br._initialize_simulation_data(); n = cfg.num_neurons

    def run(apical_pA, steps=400):
        drive = np.zeros(n, np.float32); drive[:20] = 900.0
        br.cp_external_input_current = xp.asarray(drive)
        ap = np.zeros(n, np.float32); ap[:20] = apical_pA
        br.cp_bdsp_apical_drive = xp.asarray(ap)
        for _ in range(steps):
            br._run_one_simulation_step()
        B = float(np.asarray(to_host(br.cp_bdsp_B[:20])).mean())
        vs = float(np.asarray(to_host(br.cp_membrane_potential_v[:20])).mean())
        return B, vs
    B0, vs0 = run(0.0); B1, vs1 = run(300.0)
    return {"B_rest": round(B0, 5), "B_apical": round(B1, 5), "B_rises": bool(B1 > B0 + 1e-3),
            "soma_depol": round(vs1 - vs0, 2)}


print("(0) determinism check (ou_std=0): stage detector twice, must be identical:")
a = detector(False, 0.0); b = detector(False, 0.0)
print(f"    run1 B_rest={a['B_rest']}  run2 B_rest={b['B_rest']}  identical={a['B_rest']==b['B_rest']}")
print()
print("(1) BYTE-IDENTITY: flag-OFF vs flag-ON-with-g=0 must be EXACTLY equal (g=0 -> the coupling term is 0):")
off = detector(False, 0.0); on0 = detector(True, 0.0)
print(f"    flag-off  : B_rest={off['B_rest']} B_apical={off['B_apical']}")
print(f"    flag-on g0: B_rest={on0['B_rest']} B_apical={on0['B_apical']}")
print(f"    BYTE-IDENTICAL: {off==on0}")
print()
print("(2) THE BOUNDARY (deterministic): is B flat vs apical on the pure enable_bdsp path?")
print(f"    flag-off: B_rest={off['B_rest']} -> B_apical={off['B_apical']}  B_rises={off['B_rises']}  (boundary = flat)")
print()
print("(3) THE SURPASS: gain sweep (coupling ON) -- does apical raise B while rest stays put (moat)?")
print(f"    {'g':>5} | {'B_rest':>8} {'B_apical':>9} {'B_rises':>8} | {'soma_depol':>10}")
for g in (5, 10, 20, 40, 80, 160):
    r = detector(True, g)
    print(f"    {g:>5} | {r['B_rest']:>8} {r['B_apical']:>9} {str(r['B_rises']):>8} | {r['soma_depol']:>10}")
