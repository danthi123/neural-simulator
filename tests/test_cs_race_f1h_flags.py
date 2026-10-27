"""CI guard -- cs-race f1h STEP 0 runner flags (packet research/coordination/cs-race-f1h-PACKET.md).

Pins four things about research/runners/cs_race_f1.py, CPU/numpy, one seed (41), tiny circuit:
  1. the new flags (--trace intrinsic, --soma-g, --apical-pulse) parse, and invalid combinations are refused;
  2. FLAGS OFF is the f1g circuit unchanged: golden calibration numbers captured from the runner BEFORE the f1h edit
     (merge 6c88b7860 of research/cs-race @ 2e6c85598), no BDSP config, no v_apical key in the record, same neuron count;
  3. with --trace intrinsic the apical drive is non-zero ONLY on mention-slot cells, ONLY on the steps the word's
     somatic current is on (the word's own duration), at the --apical-pulse amplitude; amplitude 0 (the LESION arm)
     writes nothing; the frozen gap4 GO values are what the config actually holds;
  4. the latched apical state does not leak across trials (cp_v_apical is not in the quiescent snapshot, so the
     circuit restores it itself).
"""
import os
import subprocess
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import research.runners.cs_race_f1 as M  # noqa: E402

SEED = 41
_P_F1G = dict(M._BASE, W_GIS=30.0, W_MM=10.0, W_MI=20.0, W_IM=20.0, NIM=8, ENCODE_WORDS=1)


@pytest.fixture()
def ment150(monkeypatch):
    monkeypatch.setattr(M, "MENT_PA", 150.0)
    return 150.0


def _grid3():
    return dict(M.F1_GRID[3])


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "research.runners.cs_race_f1", *args],
        cwd=REPO, capture_output=True, text=True, timeout=120,
        env=dict(os.environ, SIM_BACKEND="numpy"))


# ---------------------------------------------------------------- 1. flags parse
def test_flags_in_help_and_default_off():
    r = _cli("--help")
    assert r.returncode == 0
    for flag in ("--trace", "--soma-g", "--apical-pulse", "--w-im"):
        assert flag in r.stdout
    for key in ("TRACE_INTRINSIC", "SOMA_G", "APICAL_PA"):
        assert not M._BASE[key], key          # every new default is OFF


@pytest.mark.parametrize("argv", [
    ("--soma-g", "0.004"),                                   # needs --trace intrinsic
    ("--apical-pulse", "120"),                               # needs --trace intrinsic
    ("--trace", "intrinsic"),                                # needs --soma-g
    ("--trace", "intrinsic", "--soma-g", "0.004", "--w-mm", "10"),   # W_MM must be 0
    ("--trace", "recurrent"),                                # unknown trace
])
def test_invalid_flag_combinations_refused(argv):
    r = _cli("--calibrate", "--seeds", "41", *argv)
    assert r.returncode == 2, (argv, r.stderr[-300:])


# ---------------------------------------------------------------- 2. flags-off identity
def test_flags_off_matches_pre_f1h_golden(ment150):
    circ = M.RaceCircuit(3, SEED, dict(_P_F1G), _grid3())
    assert circ._ap_on is False
    bcfg = circ._bridge.core_config
    assert not getattr(bcfg, "enable_bdsp", False)
    assert not getattr(bcfg, "bdsp_apical_bistable", False)
    assert not getattr(bcfg, "bdsp_apical_couples_soma", False)
    assert circ._bridge.cp_bdsp_apical_drive is None
    assert circ._bridge.cp_v_apical is None
    assert circ._n == 346
    rec = circ.cal(seed=SEED, item="c1", word="vole")
    assert "v_apical" not in rec
    # golden: captured from the runner before the f1h edit (idB_f1g.json, seed 41, W_MM 10 W_IM 20 ment-pa 150)
    assert rec["pulsed_rate_hz"] == 0.7692
    assert rec["pulse_phase_spikes"] == 35
    assert rec["ments_spikes"] == [0, 0, 4, 0, 0, 0, 0, 0, 0]
    assert rec["all_engram_spikes"] == [0, 0, 1]
    assert rec["window_ms"] == [150, 800]


# ---------------------------------------------------------------- 3. apical drive only on mention cells
def _intrinsic(apical_pa, soma_g=0.004):
    p = dict(M._BASE, W_GIS=30.0, W_MI=20.0, W_IM=40.0, NIM=8, ENCODE_WORDS=1,
             TRACE_INTRINSIC=1, SOMA_G=soma_g, APICAL_PA=apical_pa)
    return M.RaceCircuit(3, SEED, p, _grid3())


def test_intrinsic_config_is_the_frozen_gap4_go(ment150):
    circ = _intrinsic(120.0)
    c = circ._bridge.core_config
    assert c.enable_bdsp and c.bdsp_apical_bistable and c.bdsp_apical_couples_soma
    assert c.bdsp_apical_soma_g == pytest.approx(0.004)
    for k, v in M.GAP4_GO.items():
        assert getattr(c, k) == pytest.approx(v), k
    # the gap4 6/6 GO record: self_regen 2.0, v_hold -35, kir_g 1.0, BDSP learning off, pulse 120 pA
    assert (M.GAP4_GO["coincidence_plateau_self_regen"], M.GAP4_GO["coincidence_plateau_v_hold"],
            M.GAP4_GO["apical_kir_g"], M.GAP4_GO["bdsp_learning_rate"], M.GAP4_GO_PULSE_PA) == (
        2.0, -35.0, 1.0, 0.0, 120.0)
    assert not circ.p.get("W_MM", 0.0)               # no recurrence at all
    with pytest.raises(ValueError):                  # W_MM != 0 with the intrinsic trace is refused
        M.RaceCircuit(3, SEED, dict(circ.p, W_MM=10.0), _grid3())


def test_apical_drive_nonzero_only_on_mention_cells_during_the_word(ment150):
    circ = _intrinsic(120.0)
    b, xp = circ._bridge, circ._xp
    vole = circ.ment[M.MENTION_WORDS.index("vole")]
    all_ment = np.concatenate(circ.ment)
    nonment = np.setdiff1d(np.arange(circ._n), all_ment)
    seen = []
    orig = b._run_one_simulation_step

    def spy():
        seen.append(np.asarray(xp.asarray(b.cp_bdsp_apical_drive)).copy())
        return orig()

    b._run_one_simulation_step = spy
    circ.cal(seed=SEED, item="c1", word="vole")         # one 150 ms word, then 150 + 500 ms free
    on = [i for i, d in enumerate(seen) if np.any(d != 0)]
    assert on == list(range(M.MENTION_MS))                # exactly the word's own 150 ms window
    for i in on:
        d = seen[i]
        assert np.all(d[nonment] == 0.0)                  # nothing off the mention layer
        assert np.all(d[vole] == 120.0)                   # the heard word's slot, full amplitude
        other = np.setdiff1d(all_ment, vole)
        assert np.all(d[other] == 0.0)                    # no other word slot
    assert all(not np.any(seen[i]) for i in range(M.MENTION_MS, len(seen)))   # 0 elsewhere


def test_lesion_arm_writes_no_apical_drive(ment150):
    circ = _intrinsic(0.0)                              # --trace intrinsic without --apical-pulse
    b, xp = circ._bridge, circ._xp
    mx = [0.0]
    orig = b._run_one_simulation_step

    def spy():
        mx[0] = max(mx[0], float(np.abs(np.asarray(xp.asarray(b.cp_bdsp_apical_drive))).max()))
        return orig()

    b._run_one_simulation_step = spy
    rec = circ.cal(seed=SEED, item="c1", word="vole")
    assert mx[0] == 0.0
    # no pulse -> no plateau: the pulsed slot never leaves the KIR down-state (C6 would read UNDEFINED, not FAIL)
    assert rec["v_apical"]["run_max"]["vole"] < rec["v_apical"]["v_hold"]


# ---------------------------------------------------------------- 4. the plateau does not leak across trials
def test_latched_plateau_is_reset_between_trials(ment150):
    circ = _intrinsic(120.0)
    rec = circ.cal(seed=SEED, item="c1", word="vole")
    va = rec["v_apical"]
    assert va["run_max"]["vole"] > va["v_hold"]           # the pulse took the apical past v_hold
    assert va["window"]["vole"]["min"] > va["v_hold"]     # and it HELD through the whole 650 ms window
    assert va["window"]["frog"]["max"] < va["v_hold"]     # unpulsed slot stays in the down-state
    # a second, different item on the same circuit must start from the down-state again
    rec2 = circ.cal(seed=SEED, item="c2b")
    assert max(w["max"] for w in rec2["v_apical"]["window"].values()) < va["v_hold"]
