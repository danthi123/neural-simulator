"""CI guard -- cs-race f1j runner flags (packet research/coordination/cs-race-f1j-PACKET.md, Method and S9-S14).

Pins, CPU/numpy, seed 41 (a calibration seed), the f1i corner cell (base cell soma-g 40 / W_IM 40 + the f1i SOM path at
W_SA 0.125, W_MS 215) at the step-0 mention current (150 pA):
  1. the new flags (--nmda-g0, --nmda-tau, --tau-gi) parse, default OFF, and every flag-dependency error is refused
     (CLI exit 2; RaceCircuit raises);
  2. FLAGS OFF is the f1i runner unchanged: record hashes of cal(c3), cal(c1) and one race() equal the ones captured from
     the f1i runner + engine BEFORE the f1j edit (origin/main 9f2b083c4), TRACE and LESION;
  3. --tau-gi 10 hash-equals no flag (the engine default; trial records unchanged);
  4. --nmda-g0 0 (APV) leaves every f1i field of every trial record equal to flags off; its f1j fields read g_N == 0.0
     at every ms while the recorded gating s reaches s_full on each pulsed word by its last pulse ms (S11);
  5. the reset reads back 0.0 before the snapshot and at every trial start (S9), and the runner source assigns
     cp_apical_nmda_s ONLY in _reset_nmda (S12);
  6. with --nmda-g0 G > 0: the realized config reads back (S14 a), g_N at ms 149 == G (S14 b) and decays by
     exp(-1/T) per ms after it (S14 c), every undriven mention cell's g_N is 0.0 at every ms (S10), and the c3 older
     slot's g_N at Y_on + 150 is recorded; --tau-gi 12 reads back syn_tau_g_i 12 and decay_i exp(-1/12) (S13);
  7. one CLI run writes the f1j echo and read-backs into the calibration record.
"""
import ast
import copy
import hashlib
import json
import math
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
_P_F1I = dict(M._BASE, W_GIS=30.0, W_MI=20.0, W_IM=40.0, NIM=8, TRACE_INTRINSIC=1, SOMA_G=40.0, APICAL_PA=120.0,
              ENCODE_WORDS=1, SOM_W_MS=215.0, SOM_W_SA=0.125)
# Captured from the f1i runner + engine at origin/main 9f2b083c4 (before the f1j edit), numpy, MENT_PA 150, with
# exactly the calls in _records below.
_GOLD_F1I = {
    0: {"c3": "a124310c410431ec966fb498e103c6eb02319575736b129d3219c7dd6bc23fba",
        "c1": "d74ec78f6e0efe121d9f72812a8354de8010a4184e5d46d17bec77ff24125364",
        "race": "44703c87e59e8fbdff33c48bf672aeefb3fa9b3ffafbd84414ee49ca8f70ed6c"},
    1: {"c3": "ba8c01e2b9df5a1fa6032b0d2953ad396e125c7187cedd298a3efefb7d64d15e",
        "c1": "08c3c16b37aa10572e49c998d230eb179c056e3ed2ad23830658e930a3f46ffc",
        "race": "bc02438df12236a32926d3b3d5a1072731c991e258562e0eae49a57d9d7a8ecb"},
}
F1J_TRIAL_KEYS = ("nmda",)


@pytest.fixture()
def ment150(monkeypatch):
    monkeypatch.setattr(M, "MENT_PA", 150.0)
    return 150.0


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _grid3():
    return dict(M.F1_GRID[3])


def _cli(*args, timeout=180):
    return subprocess.run(
        [sys.executable, "-m", "research.runners.cs_race_f1", *args],
        cwd=REPO, capture_output=True, text=True, timeout=timeout,
        env=dict(os.environ, SIM_BACKEND="numpy"))


def _records(p):
    c = M.RaceCircuit(3, SEED, dict(p), _grid3())
    r3 = c.cal(seed=SEED, item="c3", word="vole", word2="frog")
    r1 = c.cal(seed=SEED, item="c1", word="vole")
    rr = c.race(seed=SEED, cue_words=("lynx", "chase"), mention_words=("vole",), encode=("vole", "frog"))
    return c, {"c3": r3, "c1": r1, "race": rr}


def _strip(rec):
    return {k: v for k, v in rec.items() if k not in F1J_TRIAL_KEYS}


# ---------------------------------------------------------------- 1. flags parse
def test_flags_in_help_and_default_off():
    r = _cli("--help")
    assert r.returncode == 0
    for flag in ("--nmda-g0", "--nmda-tau", "--tau-gi"):
        assert flag in r.stdout
    assert M._BASE["NMDA_ON"] == 0 and M._BASE["NMDA_G0"] == 0.0
    assert M._BASE["NMDA_TAU"] == 0.0 and M._BASE["TAU_GI"] == 0.0


_SOM = ("--trace", "intrinsic", "--soma-g", "40", "--apical-pulse", "120", "--som", "215", "--som-ap", "0.125")


@pytest.mark.parametrize("argv", [
    ("--trace", "intrinsic", "--soma-g", "40", "--apical-pulse", "120", "--nmda-g0", "8"),   # needs --som
    _SOM + ("--nmda-tau", "50"),                                                               # needs --nmda-g0
    _SOM + ("--nmda-g0", "-1"),                                                                # g0 >= 0
    _SOM + ("--nmda-g0", "8", "--nmda-tau", "0"),                                              # tau > 0
    _SOM + ("--tau-gi", "0"),                                                                  # tau_gi > 0
    ("--trace", "intrinsic", "--soma-g", "40", "--som", "215", "--som-ap", "0.125", "--nmda-g0", "8"),  # no pulse
])
def test_invalid_combinations_refused(argv):
    r = _cli("--calibrate", "--seeds", "41", *argv)
    assert r.returncode == 2, (argv, r.stdout[-400:], r.stderr[-400:])


@pytest.mark.parametrize("extra,match", [
    (dict(SOM_W_MS=0.0, SOM_W_SA=0.0), "SOM"),
    (dict(APICAL_PA=0.0), "APICAL_PA"),
    (dict(NMDA_G0=-1.0), "NMDA_G0"),
])
def test_circuit_refuses_bad_nmda(extra, match):
    with pytest.raises(ValueError, match=match):
        M.RaceCircuit(3, SEED, dict(dict(_P_F1I, NMDA_ON=1, NMDA_G0=8.0), **extra), _grid3())


# ---------------------------------------------------------------- 2. flags off == the f1i runner
@pytest.mark.parametrize("lesion", [0, 1])
def test_flags_off_records_equal_pre_f1j_runner(ment150, lesion):
    c, recs = _records(dict(_P_F1I, SOM_LESION=lesion))
    assert c._nmda_on is False and c._bridge.cp_apical_nmda_s is None
    assert c._bridge.core_config.enable_apical_nmda is False
    assert all("nmda" not in r for r in recs.values())
    assert {k: _h(v) for k, v in recs.items()} == _GOLD_F1I[lesion]


# ---------------------------------------------------------------- 3. --tau-gi 10 == no flag
def test_tau_gi_10_hash_equals_no_flag(ment150):
    c, recs = _records(dict(_P_F1I, TAU_GI=10.0))
    assert c._bridge.core_config.syn_tau_g_i == 10.0
    assert {k: _h(v) for k, v in recs.items()} == _GOLD_F1I[0]


# ---------------------------------------------------------------- 4. APV leaves every f1i field equal
def test_apv_leaves_every_f1i_field_equal(ment150):
    c, recs = _records(dict(_P_F1I, NMDA_ON=1, NMDA_G0=0.0, NMDA_TAU=50.0))
    assert c._bridge.cp_apical_nmda_s is not None
    assert {k: _h(_strip(v)) for k, v in recs.items()} == _GOLD_F1I[0]
    s_full = recs["c1"]["nmda"]["config"]["s_full"]
    for item, last_ms in (("c1", {"vole": 149}), ("c3", {"vole": 149, "frog": 449})):
        nm = recs[item]["nmda"]
        assert nm["config"]["apical_nmda_g0"] == 0.0 and nm["config"]["enable_apical_nmda"] is True
        for w, d in nm["per_word_ms"].items():
            assert all(v == 0.0 for v in d["g_n_max"]), (item, w)
            assert max(d["s_max"][:last_ms[w] + 1]) == pytest.approx(s_full, rel=1e-5), (item, w)
        assert nm["g_n_undriven_mention_max_run"] == 0.0 and nm["g_n_undriven_mention_nonzero_ms"] == []
    assert recs["race"]["nmda"]["per_word_run_max"]["vole"]["g_n"] == 0.0
    assert recs["race"]["nmda"]["per_word_run_max"]["vole"]["s"] == pytest.approx(s_full, rel=1e-5)


# ---------------------------------------------------------------- 5. S9 reset and S12 single write site
def test_reset_reads_back_zero_before_snapshot_and_every_trial(ment150):
    c, recs = _records(dict(_P_F1I, NMDA_ON=1, NMDA_G0=8.0, NMDA_TAU=50.0))
    assert len(c._nmda_resets) == 4                      # build (before the snapshot) + 3 trials
    assert all(rb == {"s_max": 0.0} for rb in c._nmda_resets)
    assert recs["c1"]["nmda"]["reset_readback"] == {"s_max": 0.0}
    assert float(np.asarray(c._bridge.cp_apical_nmda_s).max()) > 0.0     # the race left a conductance behind ...
    r = c.cal(seed=SEED, item="c1", word="vole")
    assert r["nmda"]["reset_readback"] == {"s_max": 0.0}                 # ... and the next trial starts at 0


def _assign_targets(node):
    out = []
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in tgts:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Attribute) and sub.attr == "cp_apical_nmda_s":
                    out.append(node.lineno)
    return out


def test_runner_writes_the_gating_only_in_the_reset():
    src = open(M.__file__).read()
    tree = ast.parse(src)
    writes = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                for ln in _assign_targets(node):
                    writes.setdefault(ln, set()).add(fn.name)
    top = [ln for node in tree.body for ln in _assign_targets(node)]
    assert not top
    assert writes and all("_reset_nmda" in names for names in writes.values()), writes
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "setattr":
            assert not any(isinstance(a, ast.Constant) and a.value == "cp_apical_nmda_s" for a in call.args)
    # no NMDA engine quantity other than the gating is touched either (the conductance is derived, never stored)
    assert "apical_nmda_gating_step" not in src


# ---------------------------------------------------------------- 6. S10, S13, S14 on a G > 0 record
def test_s14_s10_readbacks_on_a_trace_record(ment150):
    G, T = 16.0, 40.0
    c, recs = _records(dict(_P_F1I, NMDA_ON=1, NMDA_G0=G, NMDA_TAU=T))
    nm = recs["c1"]["nmda"]
    cfgr = nm["config"]
    assert cfgr["enable_apical_nmda"] is True and cfgr["apical_nmda_g0"] == G and cfgr["apical_nmda_tau_ms"] == T
    assert cfgr["apical_nmda_drive_ref"] == 120.0 and cfgr["s_len"] == c._n and cfgr["s_dtype"] == "float32"
    assert cfgr["syn_tau_g_i"] == 10.0 and cfgr["decay_i"] == pytest.approx(math.exp(-0.1), rel=1e-12)
    g = nm["per_word_ms"]["vole"]["g_n_max"]
    assert g[149] == pytest.approx(G, rel=1e-5)
    d = math.exp(-1.0 / T)
    tail = [(g[t + 1], g[t]) for t in range(149, len(g) - 1) if g[t] > 1e-30]
    assert len(tail) > 100
    assert all(abs(a / b / d - 1.0) <= 1e-5 for a, b in tail)
    assert nm["driven_words"] == ["vole"] and nm["n_driven_cells"] == 8
    for rec in (recs["c1"], recs["c3"]):
        assert rec["nmda"]["g_n_undriven_mention_max_run"] == 0.0
        assert rec["nmda"]["g_n_undriven_mention_nonzero_ms"] == []
        assert rec["nmda"]["g_n_undriven_all_cells_max_run"] == 0.0
    n3 = recs["c3"]["nmda"]
    assert n3["newer_offset_ms"] == 450
    assert n3["older_g_n_at_newer_offset"] == pytest.approx(G * math.exp(-301.0 / T), rel=1e-4)
    assert n3["older_g_n_at_newer_offset"] == n3["per_word_ms"]["vole"]["g_n_max"][450]


def test_tau_gi_12_reads_back(ment150):
    c = M.RaceCircuit(3, SEED, dict(_P_F1I, NMDA_ON=1, NMDA_G0=0.0, NMDA_TAU=50.0, TAU_GI=12.0), _grid3())
    rb = c.cal(seed=SEED, item="c1", word="vole")["nmda"]["config"]
    assert rb["syn_tau_g_i"] == 12.0 and rb["decay_i"] == pytest.approx(math.exp(-1.0 / 12.0), rel=1e-12)


# ---------------------------------------------------------------- 7. the CLI record
def test_cli_record_carries_the_f1j_echo_and_readbacks(tmp_path):
    out = tmp_path / "f1j_cli.json"
    r = _cli("--calibrate", "--seeds", "41", "--grid-row", "3", "--w-gis", "30", "--w-mi", "20", "--w-im", "40",
             "--nim", "8", "--encode-words", "--ment-pa", "150", *_SOM, "--tau-gi", "12", "--nmda-g0", "8",
             "--nmda-tau", "40", "--out", str(out), timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    rec = json.loads(out.read_text())
    assert rec["flags"]["f1j"] == {"nmda_g0": 8.0, "nmda_tau": 40.0, "tau_gi": 12.0}
    ps = rec["per_seed"][0]
    cr = ps["config_readback"]["f1j"]
    assert cr["apical_nmda_g0"] == 8.0 and cr["apical_nmda_tau_ms"] == 40.0 and cr["syn_tau_g_i"] == 12.0
    for clause in ("clause_a", "clause_b"):
        assert ps["c5"]["f1j_nmda"][clause]["reset_readback"] == {"s_max": 0.0}
    for trial in [ps["c1"], ps["c2b"]] + ps["c3"]:
        assert trial["nmda"]["reset_readback"] == {"s_max": 0.0}
        assert trial["nmda"]["config"]["syn_tau_g_i"] == 12.0
    assert ps["c2b"]["nmda"]["driven_words"] == [] and ps["c2b"]["nmda"]["n_undriven_mention_cells"] == 72
    stripped = copy.deepcopy(rec)
    stripped["flags"].pop("f1j")
    assert "f1j" not in json.dumps(stripped["flags"])
