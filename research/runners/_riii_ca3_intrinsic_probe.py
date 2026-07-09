"""R-iii a0 probe (CYCLE 1093): WHY does the co-resident ca3 fire intrinsically-tonic (0.5 rate, input-independent)
on the merged nav/conv bridge while the STANDALONE completion ca3 (same IZH2007_HIPPO_PYRAMIDAL type) stays quiescent?
Build BOTH bridges, compare the ca3 Izhikevich params (C/k/vr/vt/vpeak/a/b/c/d) side by side, and step each 60x with
ZERO input (no cue, no clamp) to measure the intrinsic firing + the membrane trajectory. This localizes the tonic
firing to a specific param/config difference (fixable) vs a deep global-config conflict. GPU. NO sim/ edit.

Run: SIM_BACKEND=cupy python -m research.runners._riii_ca3_intrinsic_probe
"""
from __future__ import annotations
import numpy as np


def _ca3_params(bridge, ca3_idx):
    from sim.backend import to_host
    i0 = int(ca3_idx[0])
    g = lambda arr: float(to_host(arr[i0:i0 + 1])[0]) if arr is not None else float("nan")
    return dict(C=g(bridge.cp_izh_C), k=g(bridge.cp_izh_k), vr=g(bridge.cp_izh_vr), vt=g(bridge.cp_izh_vt),
                vpeak=g(bridge.cp_izh_vpeak), a=g(bridge.cp_izh_a), b=g(bridge.cp_izh_b),
                c=g(bridge.cp_izh_c_reset), d=g(bridge.cp_izh_d_increment))


def _intrinsic_fire(bridge, cp, ca3_idx, steps=60):
    """Step with ZERO input (no cue, no clamp) -> pure intrinsic firing. Returns (mean spikes/cell, v-of-cell0 first 6)."""
    from sim.backend import to_host
    dev = cp.asarray(ca3_idx, dtype=cp.int64)
    acc = cp.zeros(len(ca3_idx), dtype=cp.float32)
    vtraj = []
    for s in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        acc += bridge.cp_firing_states[dev].astype(cp.float32)
        if s < 6:
            vtraj.append(float(to_host(bridge.cp_membrane_potential_v[dev[:1]])[0]))
    return float(to_host(cp.mean(acc))), vtraj


def main():
    from sim.backend import get_backend
    cp, _ = get_backend()

    print("=== MERGED bridge (co_resident_hippo_memory=True) ===", flush=True)
    from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
    mb, _h = build_merged_nav_conv_bridge(seed=42, co_resident_hippo_memory=True, hippo_n_ca3=500, hippo_n_ca1=120)
    m_ca3 = np.asarray(list(mb.region_manager.indices("ca3")), dtype=np.int64)
    m_par = _ca3_params(mb, m_ca3)
    m_fire, m_v = _intrinsic_fire(mb, cp, m_ca3)
    print(f"  ca3 Izh params: {m_par}", flush=True)
    print(f"  intrinsic firing (zero input, 60 steps): {m_fire:.1f} spikes/cell | v[cell0] first6={['%.1f'%x for x in m_v]}", flush=True)
    del mb

    print("\n=== STANDALONE completion bridge (_build) ===", flush=True)
    from research.runners._riii_ca3_coincidence_completion_derisk import _build
    sb = _build(42, n_ca3=500, ca3_density=0.5, ca3w=6.0, coincidence=True, two_comp=True, apical_R=50.0,
                k_thresh=20.0, plateau_strength=300.0, weighted=True, train=True, hebb_rate=True, hebb_lr=10.0,
                hebb_decay=0.0, coact_thresh=0.001, ca3_fb_inhib=120.0, hebb_max=120.0)
    s_ca3 = np.asarray(list(sb.region_manager.indices("ca3")), dtype=np.int64)
    s_par = _ca3_params(sb, s_ca3)
    s_fire, s_v = _intrinsic_fire(sb, cp, s_ca3)
    print(f"  ca3 Izh params: {s_par}", flush=True)
    print(f"  intrinsic firing (zero input, 60 steps): {s_fire:.1f} spikes/cell | v[cell0] first6={['%.1f'%x for x in s_v]}", flush=True)

    print("\n=== PARAM DIFF (merged - standalone) ===", flush=True)
    for kk in m_par:
        d = m_par[kk] - s_par[kk]
        flag = "  <-- DIFFERS" if abs(d) > 1e-6 else ""
        print(f"  {kk}: merged={m_par[kk]:.4f} standalone={s_par[kk]:.4f} diff={d:+.4f}{flag}", flush=True)
    print(f"\n  VERDICT: merged intrinsic={m_fire:.1f} vs standalone intrinsic={s_fire:.1f} "
          f"({'MERGED IS TONIC -> find the differing param/config above' if m_fire > 5 * max(0.1, s_fire) else 'both similar -> tonic is elsewhere'})", flush=True)


if __name__ == "__main__":
    main()
