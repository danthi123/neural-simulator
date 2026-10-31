"""C2 production patient relay through the composer's patient-inhibition bridge (BRAIN_C2_COMPOSER_PATIENT_INHIB, default OFF).

(a) flag-off byte-identity: the real `OneBrainComposer._ensure_sequencer` on a stub buffer (stubbed fabric builders, the
    real score bridge and the real `make_block_drives`) gives the drive hash the pre-edit code gave (GOLDEN_DRIVES, computed
    by running `origin/main`'s one_brain_composer.py on the same fixture), calls `block_cleanup_scores` without
    `with_patient`, and leaves no `_seq_patient_*` attribute.
(b) flag-on: the agent/action drives are unchanged; the patient drive of each block lights exactly the pools the
    composer's inhibition bridge fires (`composer_patient_inhib_read`), one pool on the fixture; the lesion knob
    (BRAIN_C2_COMPOSER_PATIENT_INHIB_W_I2E=0) lights several again.
(c) the step-0 read (c) follows the flag: flag ON it reports the pools PRODUCTION lit (a planted production drive is read
    back verbatim), flag OFF it is the old divnorm relay.
(d) no new `.kb` read: the flag-on drive build reads `.kb` exactly as often as the flag-off one.
(e) `block_cleanup_scores(with_patient=True)` returns the same agent/action arrays plus the patient row of the same read.
"""
import hashlib
import os
import types

import numpy as np
import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _c2_patient_inhib as CI

V = 120
AG = np.zeros(V)
AG[5:8] = [9e5, 2e5, 1e5]
AX = np.zeros(V)
AX[10:12] = [8e5, 3e5]
PATIENT = np.zeros(V)
PATIENT[:3] = np.array([1.0, 0.3, 0.2]) * 2.4e6            # the block-0-shaped row: flag-off relay fires several pools
GOLDEN_DRIVES = "43441e410595158d"                          # origin/main 15db27a73 code, this fixture (see module doc)


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]


class CountingKB(list):
    """A `.kb` that counts entry reads (indexing, slicing, iteration)."""
    reads = 0

    def __getitem__(self, i):
        CountingKB.reads += 1
        return list.__getitem__(self, i)

    def __iter__(self):
        CountingKB.reads += 1
        return list.__iter__(self)


_SCORE_SB = {}


def _score_bridge(**_kw):
    if "sb" not in _SCORE_SB:
        from research.runners._phaseC_S5_divnorm_derisk import build_divnorm_score_bridge
        _SCORE_SB["sb"] = build_divnorm_score_bridge(seed=7, V=V, enable_divnorm=True, sigma=1.0, gain=0.11)
    return _SCORE_SB["sb"]


def stub_fns(calls):
    """`_seq_imports()` with the sequencer fabric stubbed (not under test) and the real score bridge + make_block_drives."""
    from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import make_block_drives

    def bcs(buf, b, with_patient=False):
        calls.append(with_patient)
        r = (AG * (1 + 0.1 * b), AX * (1 + 0.1 * b))
        return r + (PATIENT.copy(),) if with_patient else r

    return dict(build_sequencerK_bridge=lambda **k: ("seq", {"K": k["K"]}),
                build_sequencerK_reduced_bridge=lambda **k: ("seqR", {"K": k["K"]}),
                reduced_cue_vocab=lambda facts, K: (["a"], ["x"], {"a": 0}, {"x": 0}),
                build_divnorm_score_bridge=_score_bridge, block_cleanup_scores=bcs, make_block_drives=make_block_drives)


def stub_buf(reduced=True, K=2):
    kb = CountingKB([({"agent": "a", "action": "x", "patient": "w0"}, 0)] * K)
    return types.SimpleNamespace(kb=kb, V=V, seed=7, sequencer_sigma=1.0, sequencer_gain=0.11, sequencer_input_gain=1.0,
                                 enable_seq_vocab_shrink=reduced, _seq=None, _seq_score=None, _seq_K=None, _seq_drives=None,
                                 _seq_dirty=True, _seq_cleanup_conns_cache=None, _seq_cuevocab_sig=None)


def build(monkeypatch, buf, K=2):
    from research.runners import one_brain_composer as OBC
    calls = []
    monkeypatch.setattr(OBC, "_seq_imports", lambda: stub_fns(calls))
    CountingKB.reads = 0
    OBC.OneBrainComposer._ensure_sequencer(buf, K)
    return calls, CountingKB.reads


def _drives_hash(drives):
    return _h(np.concatenate([np.asarray(x, dtype=float).ravel() for pair in drives for x in pair]))


def test_flag_off_drives_byte_identical_and_no_patient_relay(monkeypatch):
    monkeypatch.delenv(CI.COMPOSER_PATIENT_INHIB_ENV, raising=False)
    buf = stub_buf()
    calls, _ = build(monkeypatch, buf)
    assert calls == [False, False]                                   # never asked for the patient row
    assert all(len(d) == 2 for d in buf._seq_drives)                 # (dA, dX) pairs, as every consumer unpacks
    assert _drives_hash(buf._seq_drives) == GOLDEN_DRIVES
    assert not hasattr(buf, "_seq_patient_drives") and not hasattr(buf, "_seq_patient_relay")
    assert not hasattr(buf, "_seq_score_inhib")                      # no inhibition bridge built


def test_flag_on_patient_drive_is_the_bridge_fired_pools(monkeypatch):
    monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_ENV, "1")
    monkeypatch.delenv(CI.COMPOSER_PATIENT_INHIB_W_I2E_ENV, raising=False)
    buf = stub_buf()
    calls, _ = build(monkeypatch, buf)
    assert calls == [True, True]
    assert _drives_hash(buf._seq_drives) == GOLDEN_DRIVES            # agent/action drives unchanged
    want, _inh = CI.composer_patient_inhib_read(buf, PATIENT)
    for dP, rel in zip(buf._seq_patient_drives, buf._seq_patient_relay):
        lit = [int(j) for j in np.flatnonzero(dP > 0)]
        assert lit == [int(j) for j in np.flatnonzero(want > 0)] == [0]      # exactly the pool(s) the bridge fires
        assert set(np.unique(dP)) <= {0.0, CI.PATIENT_HI_PA}
        assert rel["relay"] == "composer_patient_inhib" and rel["inh_spikes"] > 0 and rel["w_i2e"] == 2.0
    # the flag-off relay of the same row lights several pools (the defect this fixes)
    from research.runners._phaseC_S5_divnorm_derisk import onbridge_divnorm_drive
    assert int((onbridge_divnorm_drive(buf._seq_score, V, PATIENT, 1.0)[1] > 0).sum()) > 1


def test_flag_on_lesion_knob_lights_several(monkeypatch):
    monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_ENV, "1")
    monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_W_I2E_ENV, "0")
    buf = stub_buf()
    build(monkeypatch, buf)
    assert all(int((dP > 0).sum()) > 1 for dP in buf._seq_patient_drives)
    assert all(r["w_i2e"] == 0.0 for r in buf._seq_patient_relay)


@pytest.mark.parametrize("reduced", [True, False])
def test_no_new_kb_read(monkeypatch, reduced):
    monkeypatch.delenv(CI.COMPOSER_PATIENT_INHIB_ENV, raising=False)
    _, off = build(monkeypatch, stub_buf(reduced))
    monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_ENV, "1")
    _, on = build(monkeypatch, stub_buf(reduced))
    assert on == off == (1 if reduced else 0)                        # only the existing reduced-vocab self.kb[:K] read


def _read_c_buf(monkeypatch):
    from research.runners import _phaseB_onebrain_sequencer_derisk as PB, _phaseC_task1_cleanup_probe as PT
    buf = stub_buf(reduced=False)
    buf._seq_score = _score_bridge()
    buf.words = ["w%d" % i for i in range(V)]
    buf.main_roles = ["agent", "action", "patient"]
    buf.c_base = 0
    buf._silent_reserve = lambda j: False
    rows = np.zeros(3 * V)
    rows[2 * V:] = PATIENT
    buf.b = types.SimpleNamespace(cp_membrane_potential_v=rows)
    z = np.zeros(V)
    monkeypatch.setattr(PB, "block_cleanup_scores", lambda b, k: (z, z))
    monkeypatch.setattr(PT, "block_cleanup_scores_full", lambda b, k: [z, z, PATIENT])
    return buf


def test_read_c_follows_the_flag(monkeypatch):
    from research.runners import c2_hops_step0 as S0
    from research.runners._phaseC_S5_divnorm_derisk import onbridge_divnorm_drive
    monkeypatch.delenv(S0.PATIENT_INHIB_ENV, raising=False)
    monkeypatch.delenv(CI.COMPOSER_PATIENT_INHIB_ENV, raising=False)
    buf = _read_c_buf(monkeypatch)
    rows = [{"patient": "w0"}, {"patient": "w0"}]
    off = S0.read_c(buf, rows)
    assert "error" not in off, off.get("traceback")
    want = onbridge_divnorm_drive(buf._seq_score, V, PATIENT, 1.0)[1]
    assert "patient_relay" not in off
    for blk in off["blocks"]:                                        # flag off == the old relay, several pools
        assert blk["fired_pools"] == [buf.words[j] for j in range(V) if want[j] > 0] and blk["n_fired"] > 1
    # flag ON: read (c) reports what PRODUCTION lit. Plant a production drive that differs from a recomputation (w5) and
    # check the read returns it, with the re-check flagging the disagreement.
    monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_ENV, "1")
    planted = np.zeros(V)
    planted[5] = CI.PATIENT_HI_PA
    acc = np.zeros(V)
    acc[5] = 3.0
    buf._ensure_sequencer = lambda K: None
    buf._seq_patient_drives = [planted, planted]
    buf._seq_patient_relay = [dict(relay="composer_patient_inhib", w_i2e=2.0, inh_spikes=1.0, acc=acc)] * 2
    on = S0.read_c(buf, rows)
    assert "error" not in on, on.get("traceback")
    assert on["patient_relay"]["source"].startswith("production")
    for blk in on["blocks"]:
        assert blk["fired_pools"] == ["w5"] and blk["relay_recheck_same_pools"] is False
    # with the real production drives the re-check agrees and exactly the decoded patient fires
    del buf._ensure_sequencer
    from research.runners import one_brain_composer as OBC
    buf._ensure_sequencer = lambda K: OBC.OneBrainComposer._ensure_sequencer(buf, K)
    buf._seq_dirty = True
    calls = []
    monkeypatch.setattr(OBC, "_seq_imports", lambda: stub_fns(calls))
    on = S0.read_c(buf, rows)
    assert "error" not in on, on.get("traceback")
    for blk in on["blocks"]:
        assert blk["fired_pools"] == ["w0"] and blk["fires_decoded_patient"] and blk["relay_recheck_same_pools"]


def test_guard_removes_patient_attrs_a_read_created():
    from research.runners import stage2_exit_gate as SG
    buf = types.SimpleNamespace(b=None)
    with SG.substrate_read_guard(buf) as info:
        buf._seq_patient_drives = [1]
        buf._seq_patient_relay = [2]
    assert not hasattr(buf, "_seq_patient_drives") and not hasattr(buf, "_seq_patient_relay")
    assert "optional_composer_attrs_restored" not in info
    buf._seq_patient_drives = ["kept"]
    with SG.substrate_read_guard(buf):
        buf._seq_patient_drives = ["changed"]
    assert buf._seq_patient_drives == ["kept"]


def test_block_cleanup_scores_with_patient_on_a_real_composer():
    from research.runners.one_brain_composer import OneBrainComposer
    from research.runners._phaseB_onebrain_sequencer_derisk import block_cleanup_scores
    from sim.backend import to_host
    c = OneBrainComposer(seed=7, D=64, vocab=["cat", "dog", "fish", "eat", "chase", "bird"], k_max=4,
                         enable_batched=False, enable_rf_cudagraph=False)
    c.store("cat", "eat", "fish")
    c.store("dog", "chase", "cat")
    ag, ax = block_cleanup_scores(c, 1)
    ag2, ax2, ap = block_cleanup_scores(c, 1, with_patient=True)
    assert np.array_equal(ag, ag2) and np.array_equal(ax, ax2)
    iP = c.main_roles.index("patient")
    mem = np.asarray(to_host(c.b.cp_membrane_potential_v)).astype(float)
    assert np.array_equal(ap, np.maximum(mem[c.c_base + iP * c.V:c.c_base + (iP + 1) * c.V], 0.0))
    assert c.words[int(np.argmax(ap))] == "cat"                      # test-side check only: the stored patient leads
