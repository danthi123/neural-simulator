"""CI guard: the CAPSTONE — the composer (RF bind/unbind/cleanup) + the WKV cortex (ssm read-out + RF encoder) run a
full grounded turn on ONE SimulationBridge (three regions), byte-identical to isolated (2026-07-20).

Single-shared-substrate consolidation, flagship. GPU-only + needs the grounded-ft ckpt; skips otherwise."""
import os
import copy
import types
import numpy as np
import pytest

from sim.backend import is_gpu_backend

_CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"

pytestmark = pytest.mark.skipif(
    not is_gpu_backend() or not os.path.exists(_CKPT),
    reason="capstone de-risk is GPU-only and needs the grounded-ft ckpt")


def _setup():
    from research.runners._wkv_onbridge_faculty import OnBridgeWKVFaculty
    from research.runners.rf_phasor_composer import RFPhasorComposer
    from research.runners._gap_wkv_onebridge_merged_derisk import _merged_rf_encode_decode
    from research.runners._gap_onebridge_capstone_derisk import (
        _build_capstone_bridge, SharedBridgeComposer)
    facts = [("dog", "chase", "cat"), ("owl", "eat", "mouse")]
    vocab = sorted({w for f in facts for w in f})
    iso_cmp = RFPhasorComposer(seed=42, D=64, vocab=vocab)
    for a, v, p in facts:
        iso_cmp.store(a, v, p)
    iso_ans = [iso_cmp.query_patient(a, v) for a, v, _ in facts]
    iso_wkv = OnBridgeWKVFaculty(ckpt=_CKPT, seed=42, rf_synaptic=False)
    D_wkv = iso_wkv.D
    iso_wkv._wash(); iso_gen = iso_wkv.generate(["the", "dog", "can"], max_new=6)

    mb, mchan, enc_idx, cmp_idx = _build_capstone_bridge(D_wkv, 64, 42, iso_wkv.decay)
    sh_cmp = SharedBridgeComposer(seed=42, D=64, vocab=vocab)
    sh_cmp.bind_to_shared(mb, cmp_idx)
    for a, v, p in facts:
        sh_cmp.store(a, v, p)
    sh_ans = [sh_cmp.query_patient(a, v) for a, v, _ in facts]
    sh_abstain = sh_cmp.query_patient("lion", "roar")
    sh_wkv = copy.copy(iso_wkv)
    sh_wkv.b = mb; sh_wkv.nnrn = int(mb.cp_membrane_potential_v.size); sh_wkv._enc_idx = enc_idx
    sh_wkv.read_idx = np.concatenate([np.asarray(g) for g in mchan]).astype(np.int64)
    sh_wkv._rf_encode_decode = types.MethodType(_merged_rf_encode_decode, sh_wkv)
    sh_wkv._wash(); sh_gen = sh_wkv.generate(["the", "dog", "can"], max_new=6)
    return iso_ans, sh_ans, sh_abstain, iso_gen, sh_gen


def test_composer_recall_and_moat_on_shared_bridge():
    iso_ans, sh_ans, sh_abstain, _, _ = _setup()
    assert sh_ans == iso_ans, f"composer recall differs on shared bridge: {sh_ans} vs {iso_ans}"
    assert sh_abstain is None, "no-confab moat broken on shared bridge"


def test_wkv_render_unchanged_on_shared_bridge():
    _, _, _, iso_gen, sh_gen = _setup()
    assert sh_gen == iso_gen, f"WKV render differs on shared bridge: {sh_gen} vs {iso_gen}"
