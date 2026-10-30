"""C2 composer patient-read inhibition (owner ruling c2-s0d-score-1029): `BRAIN_C2_COMPOSER_PATIENT_INHIB`, research-only, default OFF.

(a) flag-off golden: the composer's own score bridge (`build_divnorm_score_bridge` via `_seq_imports`), its on-bridge drive
    and `make_block_drives` hash to values computed from the code BEFORE this build's edit; with the flag unset the composer
    builds no inhibition bridge (`patient_inhib_bridge()` is None). The sequencer path is unchanged in both states (the flag
    only adds a SEPARATE bridge).
(b) flag-on: on a small fixture buf the patient read fires exactly one pool where the flag-off read fires several, and
    the lesion arm (w_i2e=0) fires several again, so the inhibition is load-bearing.
(c) engagement: the interneurons fire (recorded spike count > 0) and fire before the pools.
"""
import hashlib
import os
import types

import numpy as np
import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _c2_patient_inhib as CI

V = 120      # ~ the real composer vocabulary size (finding: divisor D~3.5e3 at a 2.4e6 leader)
SCORES = np.zeros(V)
SCORES[:4] = np.array([1.0, 0.9, 0.7, 0.4]) * 1e6          # flag-off fixture: four pools fire (golden)
PATIENT = np.zeros(V)
PATIENT[:3] = np.array([1.0, 0.3, 0.2]) * 2.4e6             # a block-0-shaped patient row: leader, runner-up ratio 0.3


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]


GOLDEN_PARAMS = {   # computed from the code BEFORE the edit (seed 7, V=120, sigma 1.0, gain 0.11)
    "cp_izh_a": "14d1f57c1bf6fef3", "cp_izh_b": "697d82dbda514bcc", "cp_izh_c_reset": "06594aa6c41af7d8",
    "cp_izh_d_increment": "2592631dfa77c1ea", "cp_izh_C": "f04af201fe27fff4", "cp_izh_k": "ca8ca602999c41aa",
    "cp_izh_vt": "4a3bc3065d8b9e06", "cp_izh_vr": "9ede99b26e78b717"}
GOLDEN_ACC = (4, "08fc958f4fa0ea4e", "53ce2020796f1dc1")        # n pools fired, acc hash, drive hash
GOLDEN_DRIVES = "8b6ba3b711da1e52"


def _score_bridge():
    from research.runners.one_brain_composer import _seq_imports
    return _seq_imports()["build_divnorm_score_bridge"](seed=7, V=V, enable_divnorm=True, sigma=1.0, gain=0.11)


def _buf():
    buf = types.SimpleNamespace(seed=7, V=V, sequencer_sigma=1.0, sequencer_gain=0.11, sequencer_input_gain=1.0)
    buf._seq_score = _score_bridge()
    return buf


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(CI.COMPOSER_PATIENT_INHIB_ENV, raising=False)
    assert CI.composer_patient_inhib_enabled() is False
    for v in ("0", "", "off", "no"):
        monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_ENV, v)
        assert CI.composer_patient_inhib_enabled() is False
    monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_ENV, "1")
    assert CI.composer_patient_inhib_enabled() is True


def test_flag_off_golden_composer_score_bridge(monkeypatch):
    """(a) flag unset: the composer's score bridge, its drive and the sequencer's block drives equal the pre-edit hashes."""
    monkeypatch.delenv(CI.COMPOSER_PATIENT_INHIB_ENV, raising=False)
    from research.runners.one_brain_composer import _seq_imports, OneBrainComposer
    from research.runners._phaseC_S5_divnorm_derisk import onbridge_divnorm_drive
    from sim.backend import to_host
    sb = _score_bridge()
    assert sb.core_config.num_neurons == 2400
    assert not getattr(sb.core_config, "input_divisive_norm_external_only", False)
    assert not sb.core_config.enable_bounded_conductance_integration
    for name, want in GOLDEN_PARAMS.items():
        assert _h(to_host(getattr(sb, name))) == want, name
    drive, acc = onbridge_divnorm_drive(sb, V, SCORES, 1.0)
    assert (int((acc > 0).sum()), _h(acc), _h(drive)) == GOLDEN_ACC
    bs = [(SCORES, SCORES * 0.5), (SCORES[::-1].copy(), SCORES * 0.2)]
    dr, _ = _seq_imports()["make_block_drives"](sb, V, bs, input_gain=1.0, retreat="divnorm", peak_mult=1.0)
    assert _h(np.concatenate([np.asarray(x, dtype=float).ravel() for pair in dr for x in pair])) == GOLDEN_DRIVES
    # the composer method is a no-op when the flag is unset: no bridge built, nothing cached
    comp = types.SimpleNamespace(_seq_score=sb)
    assert OneBrainComposer.patient_inhib_bridge(comp) is None
    assert not hasattr(comp, "_seq_score_inhib")


def test_flag_on_builds_a_separate_bridge_and_leaves_the_score_bridge_alone(monkeypatch):
    monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_ENV, "1")
    from research.runners.one_brain_composer import OneBrainComposer
    buf = _buf()
    before = {n: _h(getattr(buf._seq_score, n)) for n in GOLDEN_PARAMS}
    inh = OneBrainComposer.patient_inhib_bridge(buf)
    assert inh is not buf._seq_score and inh is OneBrainComposer.patient_inhib_bridge(buf)     # cached
    cfg = inh.core_config
    assert cfg.input_divisive_norm_external_only and cfg.enable_bounded_conductance_integration
    assert {n: _h(getattr(buf._seq_score, n)) for n in GOLDEN_PARAMS} == before
    assert not buf._seq_score.core_config.input_divisive_norm_external_only


def test_flag_on_exactly_one_pool_lesion_and_off_fire_more(monkeypatch):
    """(b) + (c)."""
    monkeypatch.setenv(CI.COMPOSER_PATIENT_INHIB_ENV, "1")
    from research.runners._phaseC_S5_divnorm_derisk import onbridge_divnorm_drive
    buf = _buf()
    off = onbridge_divnorm_drive(buf._seq_score, V, PATIENT, 1.0)[1]
    assert int((off > 0).sum()) > 1                                       # flag-off: several pools
    tr_on, tr_les = {}, {}
    on, inh_on = CI.composer_patient_inhib_read(buf, PATIENT, trace=tr_on)
    les, inh_les = CI.composer_patient_inhib_read(buf, PATIENT, w_i2e=0.0, trace=tr_les)
    assert [int(j) for j in np.flatnonzero(on > 0)] == [0]               # exactly one pool: the leader
    assert int((les > 0).sum()) > 1                                       # lesion (w_i2e=0): several again
    assert int((off > 0).sum()) > int((on > 0).sum()) and int((les > 0).sum()) > int((on > 0).sum())
    assert inh_on > 0 and inh_les > 0                                     # (c) the interneurons fire in both
    assert 0 <= tr_on["first_spike_step_pinh"] <= min(tr_on["first_spike_step"].values())   # FS fires first
    assert tr_on["max_g_i"] > 10.0 and tr_les["max_g_i"] < 0.01 * tr_on["max_g_i"]             # the inhibition only acts when the synapses are on
