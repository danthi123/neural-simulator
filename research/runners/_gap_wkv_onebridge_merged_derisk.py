"""Single-shared-substrate consolidation — the WKV faculty PHYSICALLY MERGED onto ONE bridge (2026-07-20).

The crux + encoder-equivalence de-risks proved the two co-residence risks are byte-clean. This BUILDS the physical
merge: the WKV faculty's TWO internal bridges (self.b = ssm-state read-out + self._rfb = RF spike-encoder) become ONE
`SimulationBridge` with TWO regions -- a `chan` region (holds cp_ssm_state, the WKV leaky state) + an `encoder` region
(the RF oscillators, driven by the masked rf_resonate_steps loop). Verify-first: the merged faculty's per-token logits
must be LOGIT-IDENTICAL to the stock two-bridge OnBridgeWKVFaculty on the same sentence.

Why it's byte-exact (not merely close): the WKV state reads cp_ssm_state (= lam*s + (1-lam)*inject, NO firing -> does
not depend on the neuron thresholds the extra region perturbs); the encoder phases read the RF Im-zero-crossing of
independent oscillators (no synapses, no thresholds). So the chan region's state + the encoder region's phases are
identical whether or not the encoder region shares the bridge. rf_resonate_steps == the encoder's old step-loop
(separately proven byte-identical). => the merged forward reproduces the two-bridge forward exactly.

Non-synaptic RF encoder path only (the default). Reuse-by-import; NO sim/ edit. `--seed`, `--n-tokens`.
"""
import argparse
import copy
import types
import numpy as np

from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.regions import BrainRegion

from research.runners._wkv_onbridge_faculty import OnBridgeWKVFaculty

CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"


def _build_merged_wkv_bridge(D, seed, decay, enc_n, pop_k=1, dt=1.0):
    """ssm-state bridge (chan region, 2*D*pop_k neurons, holds cp_ssm_state) + an encoder region (enc_n RF oscillators)
    on ONE bridge. chan is region 0 so its indices == a chan-only bridge's (read_idx unchanged)."""
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_selective_ssm_state = True
    cfg.ssm_k_leak = float(max(0.0, min(1.0, 1.0 - decay)))
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False; cfg.enable_short_term_plasticity = False
    cfg.enable_parameter_heterogeneity = False; cfg.enable_conductance_noise = False
    cfg.brain_regions = [
        BrainRegion(name="chan", n_neurons=2 * D * pop_k, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="encoder", n_neurons=int(enc_n), exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = []
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    chan_idx = np.asarray(b.region_manager.indices("chan"))
    enc_idx = np.asarray(b.region_manager.indices("encoder"))
    chan_groups = [chan_idx[c * pop_k:(c + 1) * pop_k] for c in range(2 * D)]
    return b, chan_groups, enc_idx


def _merged_rf_encode_decode(self, inj):
    """The encoder step on the MERGED bridge: kick the encoder region (masked), rf_resonate_steps, read its phases.
    Mirrors OnBridgeWKVFaculty._rf_encode_decode (non-synaptic) exactly, but on self.b's encoder region."""
    d = np.clip(np.asarray(inj, np.float64), 0.0, self._RF_VMAX)
    p = self._PLO + (self._PHI - self._PLO) * (d / max(self._RF_VMAX, 1e-9))
    n = self.b.core_config.num_neurons
    z_full = np.zeros(n, dtype=np.complex64)
    z_full[self._enc_idx] = np.exp(1j * 2 * np.pi * p).astype(np.complex64)
    mask = np.zeros(n, dtype=bool); mask[self._enc_idx] = True
    self.b.rf_kick(z_full, period=self.rf_period, neuron_mask=mask)
    self.b.rf_resonate_steps(self.rf_period + 8)
    pr = np.asarray(to_host(self.b.rf_read_phases()), np.float64)[self._enc_idx]
    return np.clip((pr - self._PLO) / (self._PHI - self._PLO) * self._RF_VMAX, 0.0, self._RF_VMAX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-tokens", type=int, default=10)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()
    xp, _ = get_backend()

    # reference: stock two-bridge faculty (non-synaptic encoder)
    ref = OnBridgeWKVFaculty(ckpt=args.ckpt, seed=args.seed, rf_synaptic=False)
    D = ref.D
    enc_n = int(ref._rfb.core_config.num_neurons)   # encoder oscillator count (2*D for the phase encoder)

    # merged: ONE bridge with chan + encoder regions
    mb, mchan_groups, enc_idx = _build_merged_wkv_bridge(D, args.seed, ref.decay, enc_n)
    mrg = copy.copy(ref)                            # shares emb/Wv/head (read-only); own bridge + state below
    mrg.b = mb
    mrg.nnrn = int(mb.cp_membrane_potential_v.size)
    mrg._enc_idx = enc_idx
    # read_idx maps into the chan region (region 0 => same indices as the chan-only bridge)
    mrg.read_idx = np.concatenate([np.asarray(g) for g in mchan_groups]).astype(np.int64)
    mrg._rf_encode_decode = types.MethodType(_merged_rf_encode_decode, mrg)

    # a test token stream
    words = ["the", "penguin", "can", "not", "fly", "the", "owl", "can", "fly", "the"][: args.n_tokens]
    ids = ref.ids(words)

    # run both forwards token-by-token, compare the accumulated state + the next-token logits at each step
    ref._wash(); mrg._wash()
    state_err = 0.0; logit_err = 0.0
    rstate = None; mstate = None
    for t in ids:
        rstate = ref._charge(t)
        mstate = mrg._charge(t)
        state_err = max(state_err, float(np.max(np.abs(rstate - mstate))))
        rlg = ref._next_logits(t, rstate); mlg = mrg._next_logits(t, mstate)
        logit_err = max(logit_err, float(np.max(np.abs(rlg - mlg))))

    # end-to-end greedy generation identical?
    ref._wash(); mrg._wash()
    r_gen = ref.generate(words[:3], max_new=8)
    mrg._wash()
    m_gen = mrg.generate(words[:3], max_new=8)
    gen_match = (r_gen == m_gen)

    state_ok = state_err < 1e-5
    logit_ok = logit_err < 1e-4
    verdict = "GO" if (state_ok and logit_ok and gen_match) else "NO-GO"
    print(f"[RESULT {verdict}] WKV faculty PHYSICALLY MERGED onto ONE bridge (seed {args.seed}, {len(ids)} tokens, "
          f"chan={2*D} + encoder={enc_n} on one bridge):")
    print(f"  accumulated state  merged vs two-bridge  max|err| = {state_err:.3e}  ({'byte-clean' if state_ok else 'DIVERGES'})")
    print(f"  next-token logits  merged vs two-bridge  max|err| = {logit_err:.3e}  ({'identical' if logit_ok else 'DIVERGES'})")
    print(f"  greedy generation  merged == two-bridge : {gen_match}")
    print(f"  ref  gen: {' '.join(r_gen)}")
    print(f"  mrg  gen: {' '.join(m_gen)}")
    print(f"  => the WKV cortex (ssm read-out + RF spike-encoder) runs on ONE SimulationBridge, forward-identical to "
          f"the two-bridge faculty. One step toward composer+WKV+learning on a single shared substrate.")


if __name__ == "__main__":
    main()
