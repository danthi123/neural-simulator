"""C2 step 0b (ruling c2-s0-result-1029): local feedback inhibition among the patient word pools of the score bridge.

Default OFF and byte-identical when off: with C2S0_PATIENT_INHIB unset, `read_c` takes the existing path (same n_fired /
fired_pools as a direct `onbridge_divnorm_drive` relay of the same patient row, no `inhib` / `step0b` keys). The lesion arm
builds the same population with zeroed I->E weights. The scorer applies the two registered STOP lines to planted blocks."""
import os
import types

import numpy as np
import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import c2_hops_step0 as S0


def test_flag_default_is_off(monkeypatch):
    monkeypatch.delenv(S0.PATIENT_INHIB_ENV, raising=False)
    assert S0.patient_inhib_env() == "off"
    monkeypatch.setenv(S0.PATIENT_INHIB_ENV, "bogus")
    with pytest.raises(ValueError):
        S0.patient_inhib_env()


def test_score_step0b_stop_lines():
    def blk(on, lesion, dec="a"):
        return {"inhib": {"on": dict(fired_pools=on, n_fired=len(on), fires_decoded_patient=on == [dec], inh_spikes=3.0),
                          "lesion": dict(fired_pools=lesion, n_fired=len(lesion), fires_decoded_patient=lesion == [dec],
                                         inh_spikes=3.0)}}
    ok = S0.score_step0b([blk(["a"], ["a", "b", "c"]), blk(["a"], ["a", "b"])])
    assert ok["stops"] == [] and ok["arms"]["on"]["one_correct"] == 2 and ok["arms"]["lesion"]["multi"] == 2
    same = S0.score_step0b([blk(["a"], ["a"])])
    assert same["stops"] == ["S0B_lesion_equals_on"]
    bad = S0.score_step0b([blk(["a", "b"], ["a", "b"]), blk([], ["a"]), blk(["z"], ["a"])])
    assert "S0B_on_not_one_correct_pool" in bad["stops"]
    a = bad["arms"]["on"]
    assert (a["multi"], a["zero"], a["wrong"]) == (1, 1, 1)


def _stub_buf(V=6):
    from research.runners._phaseC_S5_divnorm_derisk import build_divnorm_score_bridge
    words = ["w%d" % i for i in range(V)]
    buf = types.SimpleNamespace(
        kb=[({}, 0)] * 2, V=V, words=words, main_roles=["agent", "action", "patient"], seed=7, sequencer_sigma=1.0,
        sequencer_gain=0.11, sequencer_input_gain=1.0, c_base=0, enable_sparse_index=False, enable_batched=False)
    buf._silent_reserve = lambda j: False
    buf._seq_score = build_divnorm_score_bridge(seed=7, V=V, enable_divnorm=True, sigma=buf.sequencer_sigma,
                                                gain=buf.sequencer_gain)
    rows = np.zeros(3 * V)
    rows[2 * V:3 * V] = [0.0, 900.0, 700.0, 0.0, 300.0, 0.0]
    buf.b = types.SimpleNamespace(cp_membrane_potential_v=rows)
    return buf, rows[2 * V:3 * V].copy()


def test_off_path_reproduces_the_current_read(monkeypatch):
    from research.runners import _phaseB_onebrain_sequencer_derisk as PB, _phaseC_task1_cleanup_probe as PT
    from research.runners._phaseC_S5_divnorm_derisk import onbridge_divnorm_drive
    monkeypatch.delenv(S0.PATIENT_INHIB_ENV, raising=False)
    buf, prow = _stub_buf()
    z = np.zeros(buf.V)
    monkeypatch.setattr(PB, "block_cleanup_scores", lambda b, k: (z, z))
    monkeypatch.setattr(PT, "block_cleanup_scores_full", lambda b, k: [z, z, prow])
    rows = [{"patient": "w1"}, {"patient": "w1"}]
    want = onbridge_divnorm_drive(buf._seq_score, buf.V, prow, buf.sequencer_input_gain)[1]
    out = S0.read_c(buf, rows)
    assert "error" not in out, out.get("traceback")
    assert "step0b" not in out and "patient_inhib" not in out
    for blk in out["blocks"]:
        assert "inhib" not in blk
        assert blk["n_fired"] == int((want > 0).sum())
        assert blk["fired_pools"] == [buf.words[j] for j in range(buf.V) if want[j] > 0]


def test_on_and_lesion_arms_record_engagement(monkeypatch):
    from research.runners import _phaseB_onebrain_sequencer_derisk as PB, _phaseC_task1_cleanup_probe as PT
    monkeypatch.setenv(S0.PATIENT_INHIB_ENV, "all")
    buf, prow = _stub_buf()
    z = np.zeros(buf.V)
    monkeypatch.setattr(PB, "block_cleanup_scores", lambda b, k: (z, z))
    monkeypatch.setattr(PT, "block_cleanup_scores_full", lambda b, k: [z, z, prow])
    out = S0.read_c(buf, [{"patient": "w1"}, {"patient": "w1"}])
    assert "error" not in out, out.get("traceback")
    assert out["patient_inhib"]["mode"] == "all" and set(out["step0b"]["arms"]) == {"on", "lesion"}
    for blk in out["blocks"]:
        for arm in ("on", "lesion"):
            r = blk["inhib"][arm]
            assert {"fired_pools", "n_fired", "inh_spikes"} <= set(r)
        # the lesion arm has the same population and the same excitation as `on`; engagement is a recorded float
        assert blk["inhib"]["lesion"]["inh_spikes"] >= 0.0


def test_ff_arm_specs_and_trace(monkeypatch):
    kn = S0.patient_inhib_knobs()
    spec = S0._inhib_arm_specs("ff", dict(kn, sweep_ff_gain=[1e-3]))
    assert {"stable_off", "ff", "ff_lesion", "ff@0.001", "ffr", "ffr_lesion"} == set(spec)
    assert all(v["stable"] for v in spec.values())                       # every 0c arm on the stable-integration bridge
    assert spec["ff_lesion"]["w_i2e"] == 0.0 and spec["ff_lesion"]["ff_gain"] == spec["ff"]["ff_gain"] > 0
    assert spec["stable_off"]["ff_gain"] == 0.0 and spec["stable_off"]["w_i2e"] == 0.0
    assert set(S0._inhib_arm_specs("off", kn)) == set()                  # default off builds nothing
    from research.runners._phaseC_S5_divnorm_derisk import build_divnorm_score_bridge
    ref = build_divnorm_score_bridge(seed=7, V=6, enable_divnorm=True, sigma=1.0, gain=0.11)
    sb = S0.build_inhib_score_bridge(7, 6, 1.0, 0.11, w_i2e=5.0, ref_sb=ref, stable=True)
    assert sb.core_config.enable_bounded_conductance_integration and sb.core_config.izh_membrane_substeps == 4
    tr = {}
    p = np.zeros(6)
    p[:3] = [2e6, 8e5, 5e5]
    S0.onbridge_inhib_drive(sb, 6, p, 1.0, ff_gain=3e-4, trace=tr)
    assert tr["ff_current_pA"] == pytest.approx(3e-4 * 3.3e6) and tr["first_spike_step_pinh"] >= 0
    assert {"first_spike_step", "max_abs_v", "max_g_i"} <= set(tr)


def test_fs_population_is_silent_without_input():
    from research.runners._phaseC_S5_divnorm_derisk import build_divnorm_score_bridge
    ref = build_divnorm_score_bridge(seed=7, V=6, enable_divnorm=True, sigma=1.0, gain=0.11)
    sb = S0.build_inhib_score_bridge(7, 6, 1.0, 0.11, w_i2e=320.0, ref_sb=ref)
    acc, inh = S0.onbridge_inhib_drive(sb, 6, np.zeros(6), 1.0)
    assert inh == 0.0 and acc.sum() == 0.0          # starts at rest (vr), not at the spike-reset potential
