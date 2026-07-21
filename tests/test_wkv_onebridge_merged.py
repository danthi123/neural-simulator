"""CI guard: the WKV faculty's two internal bridges (ssm read-out + RF spike-encoder) MERGE onto ONE bridge
(chan region + encoder region), forward-identical to the stock two-bridge faculty (2026-07-20).

Single-shared-substrate consolidation. GPU-only + needs the grounded-ft ckpt; skips otherwise."""
import os
import numpy as np
import pytest

from sim.backend import is_gpu_backend

_CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"

pytestmark = pytest.mark.skipif(
    not is_gpu_backend() or not os.path.exists(_CKPT),
    reason="merged-WKV de-risk is GPU-only and needs the grounded-ft ckpt")


def test_merged_wkv_forward_identical_to_two_bridge():
    import copy
    import types
    from research.runners._wkv_onbridge_faculty import OnBridgeWKVFaculty
    from research.runners._gap_wkv_onebridge_merged_derisk import (
        _build_merged_wkv_bridge, _merged_rf_encode_decode)
    from sim.backend import to_host  # noqa: F401 (used inside _merged_rf_encode_decode)

    ref = OnBridgeWKVFaculty(ckpt=_CKPT, seed=42, rf_synaptic=False)
    D = ref.D
    enc_n = int(ref._rfb.core_config.num_neurons)
    mb, mchan, enc_idx = _build_merged_wkv_bridge(D, 42, ref.decay, enc_n)
    mrg = copy.copy(ref)
    mrg.b = mb; mrg.nnrn = int(mb.cp_membrane_potential_v.size); mrg._enc_idx = enc_idx
    mrg.read_idx = np.concatenate([np.asarray(g) for g in mchan]).astype(np.int64)
    mrg._rf_encode_decode = types.MethodType(_merged_rf_encode_decode, mrg)
    # bridge-identity: the merged faculty uses a DISTINCT bridge, not a secret reuse of ref's bridges
    assert mrg.b is not ref.b and mrg.b is not ref._rfb, "merged must use a distinct bridge"

    ids = ref.ids(["the", "penguin", "can", "not", "fly", "the"])
    ref._wash(); mrg._wash()
    for t in ids:
        rs = ref._charge(t); ms = mrg._charge(t)
        assert float(np.max(np.abs(rs - ms))) < 1e-5, "merged ssm state diverged"
        rl = ref._next_logits(t, rs); ml = mrg._next_logits(t, ms)
        assert float(np.max(np.abs(rl - ml))) < 1e-4, "merged logits diverged"

    ref._wash(); r = ref.generate(["the", "penguin", "can"], max_new=6)
    mrg._wash(); m = mrg.generate(["the", "penguin", "can"], max_new=6)
    assert r == m, f"merged generation differs: {r} vs {m}"

    # LESION (load-bearing): scrambling mb's encoder output must DIVERGE the generation -> the encoder region genuinely
    # feeds the WKV state (not a dead passenger fed by ref._rfb).
    _rng = np.random.default_rng(999); _perm = {}

    def _scramble(self, inj):
        d = _merged_rf_encode_decode(self, inj)
        p = _perm.get(len(d))
        if p is None:
            p = _rng.permutation(len(d)); _perm[len(d)] = p
        return d[p]
    les = copy.copy(mrg); les.b = mb
    les._rf_encode_decode = types.MethodType(_scramble, les)
    les._wash(); l = les.generate(["the", "penguin", "can"], max_new=6)
    assert l != r, "encoder lesion did NOT change the output -- encoder region is not load-bearing (dead-passenger confound)"
