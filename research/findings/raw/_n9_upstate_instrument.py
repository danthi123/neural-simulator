"""N9 up-state INSTRUMENT (CuPy): why is the critic silent? Measure
A1/A2 afferent firing rate AND critic g_e / membrane V at a few A1 weights."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp, _bk = get_backend()
print("backend:", _bk, flush=True)

from research.findings.raw._n9_upstate_calib import build, _idx, _grid_prefs, place_code, _host


def instrument(bridge, drive_idx, ctx_idx, crit_idx, dvec, n_steps=120, warmup=40, label=""):
    drive_cp = xp.asarray(drive_idx); ctx_cp = xp.asarray(ctx_idx); crit_cp = xp.asarray(crit_idx)
    dvec_cp = xp.asarray(dvec, dtype=xp.float32)
    n_drive = len(drive_idx); n_crit = len(crit_idx)
    ge_l, v_l, aff_l, crit_spk = [], [], [], 0
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        bridge.cp_external_input_current[drive_cp] = dvec_cp
        bridge.cp_external_input_current[ctx_cp] = dvec_cp
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * bridge.core_config.dt_ms
        ge_l.append(float(_host(bridge.cp_conductance_g_e[crit_cp]).mean()))
        v_l.append(float(_host(bridge.cp_membrane_potential_v[crit_cp]).mean()))
        aff_l.append(float(_host(bridge.cp_firing_states[drive_cp]).sum()) / max(n_drive, 1))
        if t >= warmup:
            crit_spk += int(bridge.cp_firing_states[crit_cp].sum())
    h = n_steps // 2
    aff_rate = np.mean(aff_l[h:]) / 1e-3
    crit_rate = crit_spk / max(n_crit, 1) / ((n_steps - warmup) * 1e-3)
    print(f"  [{label}] aff_fire_rate={aff_rate:7.1f}Hz  critic g_e(last-half)={np.mean(ge_l[h:]):.4f}  "
          f"critic V(last-half mean)={np.mean(v_l[h:]):.2f}mV  Vmax={max(v_l):.2f}mV  critic_rate={crit_rate:.2f}Hz",
          flush=True)


if __name__ == "__main__":
    SEED = 42; GRID = 32; SIGMA = 4.0; DRIVE_MAX = 800.0
    NEAR = (26.571, 26.571)
    # Check the CSR slice actually installed: how many A1 synapses onto the critic, mean weight.
    for w in [6.0, 12.0, 30.0]:
        bridge, cfg = build(SEED, a1_weight=w)
        drive_idx = _idx(bridge, "vs_place_drive"); ctx_idx = _idx(bridge, "vs_place_context")
        crit_idx = _idx(bridge, "striosome_value")
        coo = bridge.cp_connections.tocoo()
        rows = _host(coo.row); cols = _host(coo.col); data = _host(coo.data)
        # rows=? cols=? check both orientations
        pre = set(int(i) for i in drive_idx); post = set(int(i) for i in crit_idx)
        m1 = np.array([(int(r) in pre and int(c) in post) for r, c in zip(rows, cols)])
        m2 = np.array([(int(r) in post and int(c) in pre) for r, c in zip(rows, cols)])
        which = "rows=pre,cols=post" if m1.sum() > m2.sum() else "rows=post,cols=pre"
        m = m1 if m1.sum() > m2.sum() else m2
        print(f"\nA1 w={w}: A1->critic synapses n={int(m.sum())} ({which}) "
              f"mean_w={float(data[m].mean()) if m.any() else float('nan'):.3f} "
              f"sum_w_per_critic~={float(data[m].sum())/max(len(crit_idx),1):.1f}", flush=True)
        prefs = _grid_prefs(len(drive_idx), GRID)
        near_vec = place_code(NEAR, prefs, DRIVE_MAX, SIGMA)
        instrument(bridge, drive_idx, ctx_idx, crit_idx, near_vec, label=f"A1 w={w} NEAR")
        del bridge
