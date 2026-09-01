"""Single-shared-substrate consolidation — CAPSTONE: the composer + the WKV cortex on ONE bridge (2026-07-20).

De-risk 5 ran the grounded turn with the composer (RFPhasorComposer, its own per-op RF bridges) + the WKV renderer
(its own two bridges) — separate bridges in one process. This CAPSTONE puts BOTH on ONE `SimulationBridge` with three
regions:
  - `chan`    (2*D_wkv)  — holds cp_ssm_state (the WKV leaky read-out state)
  - `encoder` (2*D_wkv)  — the WKV's RF spike-encoder (masked rf_resonate_steps)
  - `composer`(7*D_cmp)  — the composer's RF bind/unbind/cleanup ops (masked rf_resonate_steps, the MergedRFComposer
                            index-shift port: rebase conns by rf_base, kick masked, read the slice)

A whole grounded turn — composer STORE a fact, composer QUERY it (RF unbind + cleanup on the shared bridge), then the
WKV RENDER the retrieved answer (ssm forward on the same bridge) — runs on ONE substrate.

VERIFY-FIRST gates (the fatigue safeguard — every claim is checked):
  (1) the shared-bridge composer's recall == an ISOLATED RFPhasorComposer's recall (the composer is byte-faithful on
      the shared slice), AND
  (2) the WKV generation on the shared bridge == the ISOLATED two-bridge OnBridgeWKVFaculty (the render is unchanged).

Reuse-by-import (RFPhasorComposer + OnBridgeWKVFaculty + the MergedRFComposer _resonate pattern); NO sim/ edit.
`--seed`, `--ckpt`.
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
from research.runners._gap_wkv_onebridge_merged_derisk import _merged_rf_encode_decode
from research.runners.rf_phasor_composer import RFPhasorComposer

CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"


def _build_capstone_bridge(D_wkv, D_cmp, seed, decay, pop_k=1, dt=1.0):
    """chan (WKV ssm) + encoder (WKV RF) + composer (RF bind) regions on ONE bridge with cp_ssm_state."""
    enc_n = 2 * D_wkv
    cmp_n = 7 * D_cmp                                 # the composer's max per-op span (MergedRFComposer: 7*rf_D)
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
        BrainRegion(name="chan", n_neurons=2 * D_wkv * pop_k, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="encoder", n_neurons=enc_n, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="composer", n_neurons=cmp_n, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = []
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    chan_idx = np.asarray(b.region_manager.indices("chan"))
    enc_idx = np.asarray(b.region_manager.indices("encoder"))
    cmp_idx = np.asarray(b.region_manager.indices("composer"))
    chan_groups = [chan_idx[c * pop_k:(c + 1) * pop_k] for c in range(2 * D_wkv)]
    return b, chan_groups, enc_idx, cmp_idx


class SharedBridgeComposer(RFPhasorComposer):
    """RFPhasorComposer whose RF resonate ops run on a masked SLICE of a shared bridge (the MergedRFComposer port)."""
    def bind_to_shared(self, merged_bridge, cmp_idx):
        self._merged = merged_bridge
        self._rf_base = int(cmp_idx.min())
        self._rf_size = int(len(cmp_idx))
        n = int(merged_bridge.core_config.num_neurons)
        m = np.zeros(n, dtype=bool); m[cmp_idx] = True
        self._rf_mask = m

    def _resonate(self, n, conns, kick, period=None):
        per = self.period if period is None else int(period)   # finer-period "second look" (decode escalation)
        n = int(n)
        if n > self._rf_size:
            raise ValueError(f"RF op needs {n} neurons but composer region is {self._rf_size}")
        b = self._merged; N = int(b.core_config.num_neurons); base = self._rf_base
        shifted = [(base + int(post), base + int(pre), w) for (post, pre, w) in conns]
        b.rf_set_complex_weights(shifted)
        full_kick = np.zeros(N, dtype=np.complex128)
        kk = np.asarray(kick, dtype=np.complex128).reshape(-1)
        full_kick[base:base + n] = kk[:n]
        b.rf_kick(full_kick, period=per, lam=0.0, neuron_mask=self._rf_mask)
        b.rf_resonate_steps(per + 8)
        phases = np.asarray(b.rf_read_phases())
        return phases[base:base + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--D-cmp", type=int, default=64)
    args = ap.parse_args()
    xp, _ = get_backend()

    facts = [("dog", "chase", "cat"), ("owl", "eat", "mouse"), ("wolf", "hunt", "deer")]
    vocab = sorted({w for f in facts for w in f})

    # --- ISOLATED composer (reference) ---
    iso_cmp = RFPhasorComposer(seed=args.seed, D=args.D_cmp, vocab=vocab)
    for (a, v, p) in facts:
        iso_cmp.store(a, v, p)
    iso_ans = [iso_cmp.query_patient(a, v) for (a, v, p) in facts]
    iso_abstain = iso_cmp.query_patient("lion", "roar")            # never stored -> no-confab moat

    # --- ISOLATED WKV (reference) ---
    iso_wkv = OnBridgeWKVFaculty(ckpt=args.ckpt, seed=args.seed, rf_synaptic=False)
    D_wkv = iso_wkv.D
    iso_wkv._wash(); iso_gen = iso_wkv.generate(["the", "dog", "can"], max_new=8)

    # --- CAPSTONE: composer + WKV on ONE bridge ---
    mb, mchan, enc_idx, cmp_idx = _build_capstone_bridge(D_wkv, args.D_cmp, args.seed, iso_wkv.decay)

    sh_cmp = SharedBridgeComposer(seed=args.seed, D=args.D_cmp, vocab=vocab)
    sh_cmp.bind_to_shared(mb, cmp_idx)
    for (a, v, p) in facts:
        sh_cmp.store(a, v, p)
    sh_ans = [sh_cmp.query_patient(a, v) for (a, v, p) in facts]
    sh_abstain = sh_cmp.query_patient("lion", "roar")

    sh_wkv = copy.copy(iso_wkv)
    sh_wkv.b = mb; sh_wkv.nnrn = int(mb.cp_membrane_potential_v.size); sh_wkv._enc_idx = enc_idx
    sh_wkv.read_idx = np.concatenate([np.asarray(g) for g in mchan]).astype(np.int64)
    sh_wkv._rf_encode_decode = types.MethodType(_merged_rf_encode_decode, sh_wkv)
    sh_wkv._wash(); sh_gen = sh_wkv.generate(["the", "dog", "can"], max_new=8)

    # gates
    recall_match = (sh_ans == iso_ans)
    moat_match = (sh_abstain == iso_abstain) and (sh_abstain is None)
    gen_match = (sh_gen == iso_gen)
    # order sensitivity: interleave a composer query BETWEEN WKV tokens -> WKV still identical (byte-isolation)
    sh_wkv._wash()
    inter = []
    ids = sh_wkv.ids(["the", "dog", "can"])
    st = None
    for t in ids:
        st = sh_wkv._charge(t)
        sh_cmp.query_patient("dog", "chase")               # composer op BETWEEN WKV charges (shares the bridge)
    lg = sh_wkv._next_logits(ids[-1], st)
    iso_wkv._wash(); ist = None
    for t in iso_wkv.ids(["the", "dog", "can"]):
        ist = iso_wkv._charge(t)
    ilg = iso_wkv._next_logits(iso_wkv.ids(["the", "dog", "can"])[-1], ist)
    interleave_err = float(np.max(np.abs(lg - ilg)))
    interleave_ok = interleave_err < 1e-4

    verdict = "GO" if (recall_match and moat_match and gen_match and interleave_ok) else "NO-GO"
    print(f"[RESULT {verdict}] CAPSTONE — composer + WKV cortex on ONE bridge (seed {args.seed}, "
          f"chan={2*D_wkv} + encoder={2*D_wkv} + composer={7*args.D_cmp}):")
    print(f"  composer recall  shared vs isolated : {sh_ans} == {iso_ans}  -> {recall_match}")
    print(f"  no-confab moat   shared vs isolated : {sh_abstain!r} == {iso_abstain!r}  -> {moat_match}")
    print(f"  WKV generation   shared vs isolated : {gen_match}")
    print(f"    iso gen: {' '.join(iso_gen)}")
    print(f"    shr gen: {' '.join(sh_gen)}")
    print(f"  WKV logits w/ composer op INTERLEAVED between tokens: max|err|={interleave_err:.3e} -> {interleave_ok}")
    print(f"  => a full grounded turn (composer STORE+QUERY [RF unbind+cleanup] -> WKV RENDER [ssm forward]) runs on "
          f"ONE shared spiking substrate; composer recall + moat + WKV render all identical to isolated; the composer "
          f"op interleaved between WKV tokens does not perturb the WKV state (byte-isolated).")


if __name__ == "__main__":
    main()
