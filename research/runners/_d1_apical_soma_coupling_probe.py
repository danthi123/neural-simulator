"""Cheapest-first de-risk of the surpass: does the existing two-compartment electrotonic coupling let the BDSP apical
raise MEASURED bursts B? Test the config matrix {two_compartment_dap} x {coincidence_detection} on a BDSP detector.
If any config restores B_rises=True, the surpass is a CONFIG COMPOSE (no sim/ edit). If none do, a sim/ edit is needed.
"""
import numpy as np
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
from sim.backend import get_backend, to_host
xp, _ = get_backend()


def probe(two_comp, coinc, apical_g_couple=1.0, seed=42):
    cfg = CoreSimConfig()
    cfg.num_neurons = 40
    cfg.dt_ms = 1.0
    cfg.enable_bdsp = True
    cfg.burst_isi_threshold_ms = 6.0
    cfg.bdsp_p0 = 0.30
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_two_compartment_dap = bool(two_comp)
    cfg.enable_coincidence_detection = bool(coinc)
    cfg.apical_g_couple = float(apical_g_couple)
    cfg.seed = seed                      # cfg.seed is what ACTUALLY seeds the substrate
    cfg.actual_seed_used = seed
    try:
        br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(),
                              viz_config=VisualizationConfig(), runtime_state=RuntimeState())
        br._initialize_simulation_data()
    except Exception as e:
        return {"err": repr(e)[:120]}
    n = cfg.num_neurons

    def run(apical_pA, steps=400):
        drive = np.zeros(n, np.float32); drive[:20] = 900.0
        br.cp_external_input_current = xp.asarray(drive)
        ap = np.zeros(n, np.float32); ap[:20] = apical_pA
        br.cp_bdsp_apical_drive = xp.asarray(ap)
        for _ in range(steps):
            br._run_one_simulation_step()
        B = float(np.asarray(to_host(br.cp_bdsp_B[:20])).mean())
        va = float(np.asarray(to_host(br.cp_v_apical[:20])).mean()) if br.cp_v_apical is not None else float("nan")
        vs = float(np.asarray(to_host(br.cp_membrane_potential_v[:20])).mean())
        return B, va, vs

    B0, va0, vs0 = run(0.0)
    B1, va1, vs1 = run(300.0)
    return {"B_rest": round(B0, 4), "B_apical": round(B1, 4), "B_rises": bool(B1 > B0 + 1e-4),
            "v_apical_rest": round(va0, 1), "v_apical_ap": round(va1, 1),
            "v_soma_rest": round(vs0, 1), "v_soma_ap": round(vs1, 1)}


print("config matrix (BDSP detector; does apical=300pA raise measured B?):")
print(f"  {'two_comp':>9} {'coinc':>6} | {'B_rest':>7} {'B_apical':>9} {'B_rises':>8} | {'vsoma_rest':>10} {'vsoma_ap':>9} {'vapical_ap':>11}")
for tc in (False, True):
    for co in (False, True):
        r = probe(tc, co)
        if "err" in r:
            print(f"  {str(tc):>9} {str(co):>6} | ERROR: {r['err']}")
        else:
            print(f"  {str(tc):>9} {str(co):>6} | {r['B_rest']:>7} {r['B_apical']:>9} {str(r['B_rises']):>8} | "
                  f"{r['v_soma_rest']:>10} {r['v_soma_ap']:>9} {r['v_apical_ap']:>11}")
print()
print("baseline (pure enable_bdsp): B_rises=False. If ANY row shows B_rises=True, the surpass is a CONFIG COMPOSE.")
print("Watch v_soma_ap: if it depolarizes with the apical (vs v_soma_rest), the electrotonic coupling is reaching the soma.")
