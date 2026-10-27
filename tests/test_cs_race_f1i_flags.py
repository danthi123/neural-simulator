"""CI guard -- cs-race f1i runner flags (packet research/coordination/cs-race-f1i-PACKET.md, Method).

Pins, CPU/numpy, seed 41 (a dev seed), the f1h circuit at soma-g 40 / W_IM 40:
  1. the new flags (--som, --som-ap, --som-lesion, --som-nofac) parse, default OFF, and invalid combinations are refused;
  2. FLAGS OFF is the f1h runner unchanged: record hashes of cal(c3), cal(c1) and one race() equal the ones captured
     from the f1h runner + engine BEFORE the f1i edit (8e13b44cc), so the SOM code path is never entered;
  3. with --som: 8 SOM cells appended and declared inhibitory; the realized per-synapse STP type and U / tau_f / tau_d
     read back as wanted (engram ee = the row's globals on type 0, slot->SOM = the MWT98 triple on type 1); the apical
     routing mask is exactly the SOM->mention pathway and g_ap lives on the mention cells; step errors re-raise;
  4. every trial starts with the slot->SOM STP at rest (u 0, x 1) and g_ap 0, read back per trial (S1, S3), and the
     race record splits the STP state by pathway (S4);
  5. --som-lesion: the cs_som gate reads 0.0 at measurement time and g_ap is 0 at every recorded ms, while SOM cells
     still fire; --som-nofac: slot->SOM stp_disabled at W_MS * U.
"""
import hashlib
import json
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
_P_F1H = dict(M._BASE, W_GIS=30.0, W_MI=20.0, W_IM=40.0, NIM=8, TRACE_INTRINSIC=1, SOMA_G=40.0, APICAL_PA=120.0)
_P_F1I = dict(_P_F1H, SOM_W_MS=215.0, SOM_W_SA=0.25)
# Captured from the f1h runner + engine at 8e13b44cc (research/cs-race-f1i-packet, before the f1i edit), numpy, with
# MENT_PA 60 and exactly the calls in _f1h_records below.
_GOLD_F1H = {
    "c3": "5b0f9b9479183955d94a7b2480751b8e5eceec4dd8f68d2011e37831bf5dc73b",
    "c1": "dad0752b086caf8f178cc566b41409c17973277de57877004632e114b65c299d",
    "race": "76538878ed6699f39c5674be94c2467945344793c5a36b98760502eaf6b01ae6",
}


@pytest.fixture()
def ment60(monkeypatch):
    monkeypatch.setattr(M, "MENT_PA", 60.0)
    return 60.0


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


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
    for flag in ("--som", "--som-ap", "--som-lesion", "--som-nofac"):
        assert flag in r.stdout
    for key in ("SOM_W_MS", "SOM_W_SA", "SOM_LESION", "SOM_NOFAC"):
        assert not M._BASE[key], key
    assert M._BASE["NSOM"] == 8
    assert M.SOM_STP == dict(U=0.049, tau_f=1797.0, tau_d=399.0)


@pytest.mark.parametrize("argv", [
    ("--som", "215", "--som-ap", "0.25"),                                       # needs --trace intrinsic
    ("--trace", "intrinsic", "--soma-g", "40", "--som", "215"),                 # needs --som-ap
    ("--trace", "intrinsic", "--soma-g", "40", "--som-ap", "0.25"),             # --som-ap needs --som
    ("--trace", "intrinsic", "--soma-g", "40", "--som-lesion"),                 # needs --som
    ("--trace", "intrinsic", "--soma-g", "40", "--som-nofac"),                  # needs --som
    ("--trace", "intrinsic", "--soma-g", "40", "--som", "0", "--som-ap", "1"),  # W_MS must be > 0
])
def test_invalid_combinations_refused(argv):
    r = _cli("--calibrate", "--seeds", "41", *argv)
    assert r.returncode == 2, (argv, r.stdout[-400:], r.stderr[-400:])


def test_circuit_refuses_som_without_trace():
    with pytest.raises(ValueError, match="TRACE_INTRINSIC"):
        M.RaceCircuit(3, SEED, dict(M._BASE, SOM_W_MS=215.0, SOM_W_SA=0.25), _grid3())


# ---------------------------------------------------------------- 2. flags off == the f1h runner
def _f1h_records(p):
    c = M.RaceCircuit(3, SEED, dict(p), _grid3())
    r3 = c.cal(seed=SEED, item="c3", word="vole", word2="frog")
    r1 = c.cal(seed=SEED, item="c1", word="vole")
    rr = c.race(seed=SEED, cue_words=("lynx", "chase"), mention_words=("vole",), encode=("vole", "frog"))
    return c, {"c3": _h(r3), "c1": _h(r1), "race": _h(rr)}


def test_flags_off_records_equal_pre_f1i_runner(ment60):
    c, hashes = _f1h_records(_P_F1H)
    assert c.som is None and c._som_on is False
    assert c._bridge.core_config.enable_apical_gaba_routing is False
    assert c._bridge.core_config.enable_per_type_stp is False
    assert c._bridge.cp_conductance_g_apical_gaba is None
    assert hashes == _GOLD_F1H


# ---------------------------------------------------------------- 3. the SOM build
def test_som_build_readback(ment60):
    base_n = M.RaceCircuit(3, SEED, dict(_P_F1H), _grid3())._n
    c = M.RaceCircuit(3, SEED, dict(_P_F1I), _grid3())
    b = c._bridge
    assert c._n == base_n + 8 and c.som.size == 8 and int(c.som[0]) == base_n
    cfg = b.core_config
    assert cfg.enable_apical_gaba_routing and cfg.enable_per_type_stp and cfg.enable_bdsp
    assert b.strict_step_errors is True
    rb = c.som_readback
    assert rb["problems"] == []
    assert rb["realized_stp"]["ee"] == {"conn_type": [0], "U": [round(float(cfg.stp_U), 6)],
                                        "tau_f": [float(cfg.stp_tau_f)], "tau_d": [200.0]}
    assert rb["realized_stp"]["msom"] == {"conn_type": [1], "U": [0.049], "tau_f": [1797.0], "tau_d": [399.0]}
    assert rb["apical_targets_are_mention_cells"] is True
    assert rb["n_msom"] == rb["n_sap"] == 8 * len(M.MENTION_WORDS) * M.W_SLOT
    assert int(b.cp_conductance_g_apical_gaba.shape[0]) == c._n
    inh = np.asarray(b.cp_traits)[c.som]
    assert np.all(np.isin(inh, cfg.inhibitory_trait_indices))
    assert rb["msom_stp_disabled"] is False


# ---------------------------------------------------------------- 4. per-trial resets + STP split
def test_trial_resets_and_stp_split(ment60):
    c = M.RaceCircuit(3, SEED, dict(_P_F1I), _grid3())
    r1 = c.cal(seed=SEED, item="c3", word="vole", word2="frog")
    s = r1["som"]
    assert s["reset_readback"] == {"msom_u_max": 0.0, "msom_x_min": 1.0, "g_ap_max": 0.0, "gate_cs_som": 1.0}
    assert sum(s["som_spikes_per_ms"]) > 0                      # the pool is driven
    assert s["g_ap_max_run"] > 0.0                              # and reaches the apical
    assert set(s["per_word_ms"]) == {"vole", "frog"}
    T = len(s["som_spikes_per_ms"])
    assert all(len(v) == T for d in s["per_word_ms"].values() for v in d.values())
    # the trial left facilitated synapses + conductance behind; the next trial starts at rest again
    b = c._bridge
    assert float(np.asarray(b.cp_stp_u)[c._msom_pos].max()) > 0.0
    r2 = c.cal(seed=SEED, item="c1", word="vole")
    assert r2["som"]["reset_readback"] == s["reset_readback"]
    rr = c.race(seed=SEED, cue_words=("lynx", "chase"), mention_words=("vole",), encode=("vole", "frog"))
    assert set(rr["stp"]) == {"ee", "msom"}
    assert c._som_resets[-1]["msom_u_max"] == 0.0 and len(c._som_resets) == 4   # build + 3 trials


# ---------------------------------------------------------------- 5. lesion and nofac arms
def test_lesion_gate_closed_at_measurement_and_g_ap_zero(ment60):
    c = M.RaceCircuit(3, SEED, dict(_P_F1I, SOM_LESION=1), _grid3())
    c.cal(seed=SEED, item="c1", word="vole")                    # a prior trial must not re-open it
    r = c.cal(seed=SEED, item="c3", word="vole", word2="frog")
    s = r["som"]
    assert s["lesion"] is True
    assert s["gate_cs_som_at_measurement"] == 0.0
    assert s["reset_readback"]["gate_cs_som"] == 0.0
    assert all(v == 0.0 for d in s["per_word_ms"].values() for v in d["g_ap_max"])
    assert s["g_ap_max_run"] == 0.0
    assert sum(s["som_spikes_per_ms"]) > 0                      # SOM cells still driven and firing


def test_nofac_is_static_at_resting_efficacy(ment60):
    c = M.RaceCircuit(3, SEED, dict(_P_F1I, SOM_NOFAC=1), _grid3())
    b = c._bridge
    assert c.som_readback["msom_stp_disabled"] is True
    w = np.asarray(b.cp_connections.data)[c._msom_pos]
    np.testing.assert_allclose(w, 215.0 * 0.049, rtol=1e-6)
    c_fac = M.RaceCircuit(3, SEED, dict(_P_F1I), _grid3())
    np.testing.assert_allclose(np.asarray(c_fac._bridge.cp_connections.data)[c_fac._msom_pos], 215.0, rtol=1e-6)
