"""Micro-probe: build the neural-critic bridge and inspect the GABA_B mask.
Which synapses are tagged receptor=gaba_b? What are their PRE/POST regions?
Is cp_gabab_reversal_per_neuron set on the SNc? Does the mask align?

Also: drive the SNc tonic alone for a few steps with enable_gabab ON vs OFF
and measure SNc firing + I_gabab, to see if GABA_B silences the SNc even with
the critic silent.
"""
import os, sys
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cupy as cp
from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
import research.runners.g11_bg_runner as g11


def build_critic_bridge():
    regions, pathways = g11.build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        pfc_enable_nmda=True,
        enable_bg_lateral_inhibition=True,
        enable_visual_cortex=True,
        enable_neural_critic=True,
    )
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = 150.0
    # critic GABA_B settings exactly as the runner sets them
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.105
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, regions


def main():
    bridge, regions = build_critic_bridge()
    rm = bridge.region_manager

    def idx(name):
        try:
            i = rm.indices(name)
        except Exception:
            return np.array([], dtype=int)
        return np.asarray(i.get() if hasattr(i, "get") else i)

    snc = idx("snc")
    striov = idx("striosome_value")
    print(f"snc idx: {snc.min()}..{snc.max()} (n={snc.size})")
    print(f"striosome_value idx: {striov.min()}..{striov.max()} (n={striov.size})")

    # --- GABA_B mask inspection ---
    mask = bridge.cp_gabab_synapse_mask
    print(f"\ncp_gabab_synapse_mask is None? {mask is None}")
    if mask is not None:
        mask_h = mask.get()
        nnz = bridge.cp_connections.nnz
        print(f"mask len={mask_h.size}, nnz={nnz}, n_tagged(total)={int(mask_h.sum())}, "
              f"n_tagged(<nnz)={int(mask_h[:nnz].sum())}")
        # Which synapses are tagged? Get their PRE (row) / POST (col).
        coo = bridge.cp_connections.tocoo()
        rows = coo.row.get(); cols = coo.col.get()
        # CSR data order vs coo order may differ; the mask is aligned to
        # cp_connections.data (CSR order). Build CSR-order row index.
        csr = bridge.cp_connections
        indptr = csr.indptr.get(); indices = csr.indices.get()
        csr_rows = np.zeros(nnz, dtype=np.int64)
        for r in range(len(indptr) - 1):
            csr_rows[indptr[r]:indptr[r+1]] = r
        csr_cols = indices
        tagged = np.where(mask_h[:nnz])[0]
        tr = csr_rows[tagged]; tc = csr_cols[tagged]
        print(f"tagged PRE (row) range: {tr.min()}..{tr.max()}  unique={np.unique(tr).size}")
        print(f"tagged POST (col) range: {tc.min()}..{tc.max()}  unique={np.unique(tc).size}")
        # Are PRE all in striosome_value and POST all in snc?
        pre_in_striov = np.isin(tr, striov).mean()
        post_in_snc = np.isin(tc, snc).mean()
        print(f"fraction tagged PRE in striosome_value: {pre_in_striov:.3f}")
        print(f"fraction tagged POST in snc: {post_in_snc:.3f}")
        # What regions do tagged PRE/POST actually belong to?
        def region_of(i):
            for r in regions:
                ix = idx(r.name)
                if ix.size and i >= ix.min() and i <= ix.max() and i in set(ix.tolist()):
                    return r.name
            return "?"
        print("sample tagged PRE regions:", [region_of(int(x)) for x in tr[:5]])
        print("sample tagged POST regions:", [region_of(int(x)) for x in tc[:5]])

    # --- E_gabab per-neuron on SNc ---
    rev = bridge.cp_gabab_reversal_per_neuron
    print(f"\ncp_gabab_reversal_per_neuron is None? {rev is None}")
    if rev is not None:
        rev_h = rev.get()
        print(f"E_gabab on SNc: {np.unique(rev_h[snc])}")
        print(f"E_gabab global unique values: {np.unique(rev_h)} (counts: "
              f"{[int((rev_h==v).sum()) for v in np.unique(rev_h)]})")
        # Which neurons have E_gabab = -90 (the GABA_B target reversal)?
        tgt = np.where(np.abs(rev_h - (-90.0)) < 1e-3)[0]
        print(f"n neurons with E_gabab=-90: {tgt.size}; all in snc? "
              f"{np.isin(tgt, snc).all() if tgt.size else 'n/a'}")

    # --- Decisive: drive ONLY the SNc tonic, critic silent, gabab ON vs OFF ---
    # If GABA_B (with g=0) silences SNc, something else is wrong; if SNc fires
    # the same both ways, the silencing in nav is NOT this block.
    print("\n=== SNc-tonic-only test (critic silent) ===")
    for gabab_on in (True, False):
        # fresh state
        bridge.cp_membrane_potential_v[:] = -65.0
        if bridge.cp_conductance_g_gabab is not None:
            bridge.cp_conductance_g_gabab[:] = 0.0
        bridge.cp_conductance_g_e[:] = 0.0
        bridge.cp_conductance_g_i[:] = 0.0
        bridge.cp_prev_firing_states[:] = False
        bridge.cp_firing_states[:] = False
        bridge.core_config.enable_gabab = gabab_on
        snc_spikes = 0
        gabab_snc_max = 0.0
        for s in range(200):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[cp.asarray(snc)] = cp.float32(220.0)
            bridge._run_one_simulation_step()
            snc_spikes += int(bridge.cp_firing_states[cp.asarray(snc)].sum())
            if bridge.cp_conductance_g_gabab is not None:
                gabab_snc_max = max(gabab_snc_max,
                                    float(bridge.cp_conductance_g_gabab[cp.asarray(snc)].max()))
        print(f"  gabab_on={gabab_on}: snc_spikes over 200 steps = {snc_spikes} "
              f"(rate~{snc_spikes/snc.size/200*1000:.1f}Hz), gabab_snc_max={gabab_snc_max:.4f}")


if __name__ == "__main__":
    main()
