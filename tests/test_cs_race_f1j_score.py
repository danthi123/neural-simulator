"""CI guard -- the cs-race f1j STEP-0 scorer (research/runners/cs_race_f1j_score.py; packet
research/coordination/cs-race-f1j-PACKET.md, Step 0).

Planted fixtures, never sampled values:
  1. names: the artifact names the scorer asks for are exactly the --out paths of the 1042 wave 1-2 lines;
  2. the landed f1i reference (skipped when absent): 45 early / 6 late / 21 pass C6 records and 23 defined corner pairs,
     seed 50 frog>vole the undefined one, re-derived from the artifacts;
  3. real runner records (CPU numpy, seed 41, one CLI run per arm, in parallel): every guard, flag, schedule, STOP-ID
     and consistency read is clean on APV, TRACE, LESION, IPSC-12 and K1-NOFAC; the APV record minus its f1j fields hash
     equals the landed f1i record (identity (iii) on one record);
  4. planted guard failures on copies of the real records: S9, S10, S11 (g_N, s, missing s), S13, S14 (a)/(b)/(c),
     a missing c5 race record; flag mismatches;
  5. C7(c) rewritten: the f1i peak reading fails a late peak whose RISE precedes the termination, the rewrite passes
     it; a silent SOM cell and a rise after the termination fail; with no termination the spikes and a g_ap > 0 pass;
  6. rescue with attribution on planted records: rescue, IPSP removal (floor and APV fraction), the 0.3189 second
     reading, APV passing too, attribution UNDEFINED (no fall ms), invalid artifact UNDEFINED; K1 (a) and (b), and
     K1 UNDEFINED (never "not falsified") when K1-NOFAC / APV / GRID records are missing or a K1 C6 is UNDEFINED;
  7. selection order, F1-F4 and the decision tree (branch 1 alone);
  8. score_full end to end on a fake loader (no artifact): a clean world selects t40_g4 at W_MS 215 (branch 9); F1, F2,
     F4, STOP-SMOKE and STOP-ATTRIB fire on their planted worlds; the seed-50 frog>vole label; a missing GRID record
     (at the rescue bar) and a missing IPSC-12 record make the read UNDEFINED, never F1 / F3, STOP-ID still first; a
     missing NOMEM record leaves the decision and makes the NOMEM count UNDEFINED;
  9. identity mode end to end on planted artifacts built from the landed f1i records (skipped when absent): PASS; a
     missing artifact is UNDEFINED, never a pass; an S9 read-back fault is STOP-ID; a mismatch is HOST-CLASS PENDING
     across host classes and STOP-ID on the same class;
 10. P0': the needles exist in this checkout, and a missing file or needle is STOP-ENGINE.
"""
import copy
import json
import math
import os
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import cs_race_f1i_score as F1I  # noqa: E402
from research.runners import cs_race_f1j_lines as LN  # noqa: E402
from research.runners import cs_race_f1j_score as S  # noqa: E402
from tests import test_cs_race_f1i_score as T1  # noqa: E402  (the f1i planted-record builders)

F1I_ROOT = os.path.join(REPO, S.F1I_DIR)
HAVE_F1I = os.path.exists(os.path.join(F1I_ROOT, "cal_sa0p125_ms215_s41.json"))
need_f1i = pytest.mark.skipif(not HAVE_F1I, reason="f1i step-0 artifacts not in this checkout")
PIN = "f" * 40
UP, DOWN = T1.UP, T1.DOWN


# ── 1. names ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _scorer_names():
    names = [S.art_name("identity", s) for s in S.IDENTITY_SEEDS]
    names += [S.art_name("apv", s, sa, ms) for sa, ms in S.F1I_CELLS for s in S.SEEDS]
    names += [S.art_name("k1nofac", s, sa, ms) for sa, ms in S.K1_CELLS for s in S.SEEDS]
    for arm in ("cal", "lesion", "ipsc12"):
        names += [S.art_name(arm, s, 0.125, ms, t, g) for t, g in S.NMDA_CELLS for ms in S.PRIMARY_MS for s in S.SEEDS]
    names += [S.art_name("grid", s, sa, ms, t, g) for t, g in S.NMDA_CELLS for sa, ms in LN.GRID_CELLS
              for s in S.SEEDS]
    names += [S.art_name("ipsc12_apv", s, 0.125, ms) for ms in S.PRIMARY_MS for s in S.SEEDS]
    return names


def test_scorer_names_are_the_line_outputs(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text(LN.render(LN.wave1() + LN.wave2()))
    outs = {os.path.basename(k) for k in F1I.parse_lines(str(p))}
    names = _scorer_names()
    assert len(names) == len(set(names)) == 1042
    assert set(names) == outs
    assert S.art_name("nomem", 41, 0.125, 304, None, 8) == "nomem_g8_sa0p125_ms304_s41.json"


def test_arm_spec():
    assert S.arm_spec("identity")[0] is None
    assert S.arm_spec("apv") == ({"nmda_g0": 0.0, "nmda_tau": 50.0}, False, False, 10.0, 0.0, 50.0)
    assert S.arm_spec("k1nofac")[2] is True
    assert S.arm_spec("ipsc12_apv")[0] == {"nmda_g0": 0.0, "nmda_tau": 50.0, "tau_gi": 12.0}
    assert S.arm_spec("lesion", 40, 4) == ({"nmda_g0": 4.0, "nmda_tau": 40.0}, True, False, 10.0, 4.0, 40.0)
    assert S.arm_spec("ipsc12", 75, 16)[0] == {"nmda_g0": 16.0, "nmda_tau": 75.0, "tau_gi": 12.0}
    assert S.arm_spec("nomem", None, 8)[0] == {"nmda_g0": 8.0, "nmda_tau": 2.0}


# ── 2. the landed f1i reference ────────────────────────────────────────────────────────────────────────────────────

@need_f1i
def test_f1i_reference_counts_rederived():
    ref = S.f1i_reference(F1I_ROOT)
    assert ref["counts"] == {"early": 45, "late": 6, "pass": 21, "corner_defined": 23} and ref["ok"]
    assert ref["corner"]["undefined"] == [S.CORNER_UNDEF_PAIR]
    assert sorted(ref["c6"]["late"]) == sorted([(0.125, 304, 46), (0.125, 429, 46), (0.125, 429, 49), (0.125, 429, 51),
                                                (0.25, 429, 49), (0.25, 429, 51)])
    assert all(ref["c6"]["drop_ms"][k] < 200 for k in ref["c6"]["early"])


def test_f1i_reference_missing_root_is_not_ok(tmp_path):
    ref = S.f1i_reference(str(tmp_path))
    assert not ref["ok"] and ref["c6"]["problems"] and ref["corner"]["problems"]


# ── 3. real runner records ─────────────────────────────────────────────────────────────────────────────────────────

_BASE_ARGV = ["--calibrate", "--seeds", "41", "--grid-row", "3", "--w-gis", "30", "--w-mi", "20", "--w-im", "40",
              "--nim", "8", "--encode-words", "--ment-pa", "150", "--trace", "intrinsic", "--soma-g", "40",
              "--apical-pulse", "120"]
_REAL = {  # tag: (arm, W_SA, W_MS, tauN, gN0, extra flags)
    "apv": ("apv", 0.125, 215, None, None, ["--nmda-g0", "0", "--nmda-tau", "50"]),
    "cal": ("cal", 0.125, 215, 40, 4, ["--nmda-g0", "4", "--nmda-tau", "40"]),
    "lesion": ("lesion", 0.125, 215, 40, 4, ["--som-lesion", "--nmda-g0", "4", "--nmda-tau", "40"]),
    "ipsc12": ("ipsc12", 0.125, 215, 40, 4, ["--tau-gi", "12", "--nmda-g0", "4", "--nmda-tau", "40"]),
    "k1nofac": ("k1nofac", 0.125, 429, None, None, ["--som-nofac", "--nmda-g0", "0", "--nmda-tau", "50"]),
}


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    d = tmp_path_factory.mktemp("f1j_real")
    env = dict(os.environ, SIM_BACKEND="numpy", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    procs = {}
    for tag, (_arm, sa, ms, _t, _g, extra) in _REAL.items():
        argv = _BASE_ARGV + ["--som", str(ms), "--som-ap", "%g" % sa] + extra + ["--out", str(d / (tag + ".json"))]
        procs[tag] = subprocess.Popen([sys.executable, "-m", "research.runners.cs_race_f1"] + argv, cwd=REPO, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = {}
    for tag, p in procs.items():
        log, _ = p.communicate(timeout=600)
        assert p.returncode == 0, (tag, log[-2000:])
        with open(d / (tag + ".json")) as f:
            out[tag] = json.load(f)
    return out


def test_real_records_read_clean(real):
    for tag, (arm, sa, ms, t, g, _x) in _REAL.items():
        rec = real[tag]
        ps = rec["per_seed"][0]
        assert S.flag_problems(rec, arm, 41, sa, ms, t, g) == [], tag
        assert F1I.schedule_problems(ps) == [], tag
        assert F1I.stop_id_problems(rec, "lesion" if arm == "lesion" else "trace") == [], tag
        assert S.guard_problems(rec, arm, t, g) == [], tag
        assert F1I.consistency(ps) == [], tag
    it = S.seed_items_f1j(real["cal"]["per_seed"][0], real["lesion"]["per_seed"][0])
    assert all(v is not None for v in it["older_g_n_at_newer_offset"])


@need_f1i
def test_real_apv_minus_f1j_fields_equals_the_landed_f1i_record(real):
    with open(os.path.join(F1I_ROOT, "cal_sa0p125_ms215_s41.json")) as f:
        ref = json.load(f)
    stripped = S.strip_f1j(real["apv"])
    assert F1I.canon_hash(stripped) == F1I.canon_hash(ref), S._first_diffs(stripped, ref)
    assert F1I.canon_hash(real["apv"]) != F1I.canon_hash(ref)          # the f1j fields are there before stripping


def test_strip_f1j_removes_only_the_f1j_fields(real):
    st = S.strip_f1j(real["cal"])
    text = json.dumps(st)
    assert '"nmda"' not in text and '"f1j"' not in text and '"f1j_nmda"' not in text
    plain = T1.make_ps()
    assert S.strip_f1j({"per_seed": [plain]}) == {"per_seed": [plain]}
    assert real["cal"]["flags"]["f1j"] == {"nmda_g0": 4.0, "nmda_tau": 40.0}          # the input is not mutated


# ── 4. planted guard failures ──────────────────────────────────────────────────────────────────────────────────────

def _g(real, tag, mutate):
    rec = copy.deepcopy(real[tag])
    mutate(rec["per_seed"][0])
    arm, _sa, _ms, t, g, _x = _REAL[tag]
    return S.guard_problems(rec, arm, t, g)


def _has(probs, *needles):
    return any(all(n in p for n in needles) for p in probs)


def test_s9_reset_readback(real):
    def m(ps):
        ps["c3"][1]["nmda"]["reset_readback"] = {"s_max": 0.001}
    assert _has(_g(real, "cal", m), "c3[1]", "S9")

    def m2(ps):
        ps["c5"]["f1j_nmda"]["clause_b"]["reset_readback"] = {"s_max": 0.5}
    assert _has(_g(real, "cal", m2), "clause_b", "S9")


def test_s10_undriven(real):
    def m(ps):
        ps["c1"]["nmda"]["g_n_undriven_mention_nonzero_ms"] = [[300, 0.01]]
    assert _has(_g(real, "cal", m), "c1", "S10")

    def m2(ps):
        ps["c2b"]["nmda"]["g_n_undriven_all_cells_max_run"] = 1e-9
    assert _has(_g(real, "cal", m2), "c2b", "S10")

    def m3(ps):
        ps["c5"]["f1j_nmda"]["clause_a"]["g_n_undriven_all_cells_max_run"] = 0.2
    assert _has(_g(real, "cal", m3), "clause_a", "S10")


def test_s11_on_apv(real):
    def m(ps):
        ps["c1"]["nmda"]["per_word_ms"]["vole"]["g_n_max"][400] = 1e-6
    assert _has(_g(real, "apv", m), "c1", "S11 APV g_N")

    def m2(ps):
        w = ps["c3"][0]["v_apical"]["pulsed_words"][1]
        s = ps["c3"][0]["nmda"]["per_word_ms"][w]["s_max"]
        for t in range(len(s)):
            s[t] = min(s[t], 0.9)
    assert _has(_g(real, "apv", m2), "c3[0]", "did not reach s_full")

    def m3(ps):
        del ps["c1"]["nmda"]["per_word_ms"]["vole"]["s_max"]
    assert _has(_g(real, "apv", m3), "S11 s field missing", "UNDEFINED")

    def m4(ps):
        k = next(iter(ps["c5"]["f1j_nmda"]["clause_a"]["per_word_run_max"]))
        ps["c5"]["f1j_nmda"]["clause_a"]["per_word_run_max"][k]["g_n"] = 0.1
    assert _has(_g(real, "apv", m4), "clause_a", "S11")

    def m5(ps):
        ps["c1"]["nmda"]["per_word_ms"]["frog"] = copy.deepcopy(ps["c1"]["nmda"]["per_word_ms"]["vole"])
    assert _has(_g(real, "apv", m5), "S11 pulsed words")


def test_s13_and_s14(real):
    def m(ps):
        ps["c1"]["nmda"]["config"]["syn_tau_g_i"] = 12.0
    assert _has(_g(real, "cal", m), "c1", "S13")
    rec = real["ipsc12"]
    assert S.guard_problems(rec, "cal", 40, 4) != []                    # an IPSC-12 record read as TRACE: S13 fires
    assert _has(S.guard_problems(real["cal"], "cal", 40, 8), "S14(a)")  # the stem's gN0 differs from the realized one

    def m2(ps):
        ps["c1"]["nmda"]["per_word_ms"]["vole"]["g_n_max"][S.C1_LAST] *= 1.001
    assert _has(_g(real, "cal", m2), "S14(b)")

    def m3(ps):
        ps["c1"]["nmda"]["per_word_ms"]["vole"]["g_n_max"][300] *= 1.01
    assert _has(_g(real, "cal", m3), "S14(c)")

    def m4(ps):
        ps["c1"]["nmda"]["per_word_ms"]["vole"]["g_n_max"] = []
    assert _has(_g(real, "cal", m4), "S14 g_N series missing", "UNDEFINED")

    def m5(ps):
        ps["c3"][2]["nmda"]["config"]["apical_nmda_drive_ref"] = 100.0
    assert _has(_g(real, "cal", m5), "c3[2]", "S14(a)")


def test_missing_records_are_stop_id(real):
    def m(ps):
        del ps["c5"]["f1j_nmda"]["clause_a"]
        ps["c2b"].pop("nmda")
        ps.pop("config_readback")
    probs = _g(real, "cal", m)
    assert _has(probs, "c5 clause_a: no f1j_nmda record") and _has(probs, "c2b: no nmda record")
    assert _has(probs, "config_readback.f1j missing")


def test_flag_problems_planted(real):
    rec = copy.deepcopy(real["cal"])
    assert _has(S.flag_problems(rec, "cal", 41, 0.125, 215, 50, 4), "flags.f1j")
    assert _has(S.flag_problems(rec, "lesion", 41, 0.125, 215, 40, 4), "som_lesion")
    assert _has(S.flag_problems(rec, "cal", 45, 0.125, 215, 40, 4), "seed mismatch")
    assert _has(S.flag_problems(rec, "cal", 41, 0.125, 304, 40, 4), "cell mismatch")
    assert _has(S.flag_problems(real["apv"], "k1nofac", 41, 0.125, 215), "som_nofac")
    assert _has(S.flag_problems(real["apv"], "identity", 41, 0.125, 215), "identity record carries f1j flags")
    assert S.guard_problems(real["apv"], "identity") == []


# ── 5. C7(c) rewritten ─────────────────────────────────────────────────────────────────────────────────────────────

def _pair0(ps_kw, lesion_kw=None):
    it = S.seed_items_f1j(T1.make_ps(pair_kw={0: ps_kw}), T1.make_lesion(lesion_kw))
    return it, it["c7_pairs"][0]


def test_c7c_rewrite_passes_a_late_peak_whose_rise_precedes_termination():
    it, p = _pair0({"late_peak": True})
    assert p["c_f1i_peak_reading"] is False and p["c"] is True and p["status"] == S.PASS
    assert p["g_ap_onset_ms"] == 330 and p["termination_ms"] == 360 and p["g_ap_peak_ms"] == 400
    assert it["C7"] == S.PASS
    assert F1I.seed_items(T1.make_ps(pair_kw={0: {"late_peak": True}}), T1.make_lesion())["C7"] == S.FAIL


def test_c7c_rewrite_fails_silent_cell_and_rise_after_termination():
    _it, p = _pair0({"silent_cell": 3})
    assert p["c"] is False and p["c_fail_reason"] == "a SOM cell silent in [300, 600)" and p["status"] == S.FAIL
    _it, p = _pair0({"older_end": 325})
    assert p["termination_ms"] == 325 and p["g_ap_onset_ms"] == 330 and p["a"] is True
    assert p["c"] is False and p["c_fail_reason"] == "no g_ap increase after Y_on before the termination ms"
    assert p["status"] == S.FAIL


def test_c7c_without_termination():
    _it, p = _pair0({"older_end": None})
    assert p["termination_ms"] is None and p["c"] is True and p["a"] is False and p["status"] == S.FAIL
    ps = T1.make_ps(pair_kw={0: {"older_end": None}})
    g = ps["c3"][0]["som"]["per_word_ms"]["vole"]["g_ap_max"]
    for t in range(len(g)):
        g[t] = 0.0
    p = S.c7_pair_f1j(ps["c3"][0], None, "vole", "frog")
    assert p["c"] is False and p["c_fail_reason"] == "older g_ap never > 0 in [300, 950)"


def test_c7_undefined_pair_stays_undefined():
    it = S.seed_items_f1j(T1.make_ps(pair_kw={1: {"older_latched": False}}), T1.make_lesion())
    assert it["c7_pairs"][1]["status"] == S.UNDEF and it["C7"] == S.UNDEF
    it = S.seed_items_f1j(T1.make_ps(), None)
    assert it["C7"] == S.UNDEF                                     # no LESION record: (d) undefined, never a pass


def test_summarize_trace_cell_reports_both_c_readings():
    rows = {s: S.seed_items_f1j(T1.make_ps(pair_kw={0: {"late_peak": True}}), T1.make_lesion()) for s in S.SEEDS}
    cs = S.summarize_trace_cell(rows)
    assert cs["C7"]["full"] and cs["c7_sub"]["c"] == 24 and cs["c7_sub"]["c_f1i_peak_reading"] == 16
    assert len(cs["c7_timing"]) == 24 and cs["older_g_n_at_newer_offset"] is None


# ── 6. rescue with attribution, K1 ─────────────────────────────────────────────────────────────────────────────────

def _c6_fail(ps, drop=None):
    """C6 FAIL: the c1 plateau first below v_hold at DROP; with DROP None, an unpulsed word above v_hold instead (no
    fall ms)."""
    c1 = ps["c1"]
    if drop is None:
        c1["v_apical"]["window"]["frog"] = {"min": DOWN, "max": UP}
        return ps
    pw = c1["som"]["per_word_ms"]["vole"]
    v = T1._series(900, DOWN, [(10, drop, UP)])
    pw["v_apical_min"], pw["v_apical_max"] = v, list(v)
    c1["v_apical"]["window"]["vole"] = {"min": round(min(v[150:800]), 4), "max": round(max(v[150:800]), 4)}
    return ps


def _g_ap(ps, val, lo=150, hi=200):
    ps["c1"]["som"]["per_word_ms"]["vole"]["g_ap_max"] = T1._series(900, 0.0, [(120, 150, 0.3), (lo, hi, val)])
    return ps


def _rec(ps):
    return {"per_seed": [ps]}


class FakeLoader:
    def __init__(self, recipe, root="/nonexistent"):
        self.recipe, self.root = recipe, root
        self.cache, self.invalid, self.stop_id, self.consistency = {}, {}, {}, {}

    def get(self, arm, seed, sa, ms, tau=None, g0=None):
        name = S.art_name(arm, seed, sa, ms, tau, g0)
        if name not in self.cache:
            self.cache[name] = self.recipe(arm, seed, sa, ms, tau, g0)
            if self.cache[name] is None:
                self.invalid[name] = ["missing (planted)"]
        return self.cache[name]


def test_rescue_attribution_rows():
    trace = {41: _g_ap(T1.make_ps(), 0.5), 45: _g_ap(T1.make_ps(), 0.2), 46: _g_ap(T1.make_ps(), 0.319),
             47: _g_ap(T1.make_ps(), 0.45), 48: _g_ap(T1.make_ps(), 0.5), 49: _g_ap(T1.make_ps(), 0.5),
             50: None, 51: _g_ap(_c6_fail(T1.make_ps(), 170), 0.5)}
    apv = {41: _c6_fail(T1.make_ps(), 170), 45: _c6_fail(T1.make_ps(), 170), 46: _c6_fail(T1.make_ps(), 170),
           47: _g_ap(_c6_fail(T1.make_ps(), 170), 0.6), 48: T1.make_ps(), 49: _c6_fail(T1.make_ps(), None),
           50: _c6_fail(T1.make_ps(), 170), 51: _c6_fail(T1.make_ps(), 170)}

    def recipe(arm, s, sa, ms, tau, g0):
        src = apv if arm == "apv" else trace
        return _rec(src[s]) if src[s] is not None else None
    L = FakeLoader(recipe)
    rr = S.rescue_reads(L, [(0.125, 215, s) for s in S.SEEDS], 40, 4)
    st = {r["seed"]: r["status"] for r in rr["rows"]}
    assert st == {41: "rescue", 45: "rescued by IPSP removal", 46: "rescued by IPSP removal",
                  47: "rescued by IPSP removal", 48: "passes in APV too (not a rescue)",
                  49: "rescued, attribution UNDEFINED (APV fall ms absent)", 50: "UNDEFINED (invalid artifact)",
                  51: "not rescued"}
    alt = {r["seed"]: r.get("alt_floor_0p3189") for r in rr["rows"]}
    assert alt[46] == "rescue" and alt[45] == "rescued by IPSP removal" and alt[47] == "rescued by IPSP removal"
    assert rr["count"] == 1 and rr["count_alt_floor_0p3189"] == 2
    assert [r["seed"] for r in rr["removal"]] == [45, 46, 47]
    assert [r["seed"] for r in rr["undefined"]] == [49, 50]
    r41 = rr["rows"][0]
    assert r41["g_ap_max_trace"] == 0.5 and r41["g_ap_max_apv"] == 0.3 and r41["apv_fall_ms"] == 170
    assert r41["som_tail_trace"] == S._som_tail(_rec(trace[41]))
    assert set(L.cache) == {S.art_name("cal", s, 0.125, 215, 40, 4) for s in S.SEEDS} | {
        S.art_name("apv", s, 0.125, 215) for s in S.SEEDS}


def test_rescue_window_is_capped_at_the_apv_fall_ms():
    tr = _g_ap(T1.make_ps(), 0.5, lo=171, hi=200)        # the TRACE g_ap only after the APV twin fell (170)
    ap = _c6_fail(T1.make_ps(), 170)
    L = FakeLoader(lambda arm, *a: _rec(ap if arm == "apv" else tr))
    rr = S.rescue_reads(L, [(0.125, 215, 41)], 40, 4)
    assert rr["rows"][0]["status"] == "rescued by IPSP removal"
    assert rr["rows"][0]["g_ap_max_trace"] == 0.0 and rr["rows"][0]["g_ap_max_apv"] == 0.3


def _som_after(ps, n, t0=210):
    sp = ps["c1"]["som"]["som_spikes_per_ms"]
    for i in range(n):
        sp[t0 + i] += 1
    return ps


def test_k1_falsifier_reads():
    def world(late_apv_som, nofac_late):
        def recipe(arm, s, sa, ms, tau, g0):
            if arm == "apv" and (sa, ms) == (0.125, 429) and s in (46, 49, 51):
                return _rec(_som_after(_c6_fail(T1.make_ps(), 300), late_apv_som))
            if arm == "k1nofac" and (sa, ms) == (0.125, 429) and s in (41, 45, 46, 47)[:nofac_late]:
                return _rec(_som_after(_c6_fail(T1.make_ps(), 320), 5))
            return _rec(T1.make_ps())
        return S.k1_reads(FakeLoader(recipe))
    k = world(late_apv_som=4, nofac_late=1)
    assert k["apv"] == {"late": 3, "early": 0, "n": 16} and k["nofac"]["late"] == 1
    assert not k["falsified_a"] and not k["falsified_b"] and not k["H_K1_falsified"]
    assert len(k["late_fails_apv_grid"]) == 3
    k = world(late_apv_som=0, nofac_late=1)                  # (a): a late APV fail with 0 SOM spikes in [200, fall)
    assert k["falsified_a"] and k["H_K1_falsified"] and len(k["late_fails_with_0_som_after_200"]) == 3
    k = world(late_apv_som=4, nofac_late=3)                  # (b): NOFAC as many late fails as APV
    assert k["falsified_b"] and not k["falsified_a"] and k["H_K1_falsified"]


def test_k1_missing_records_are_undefined_never_not_falsified():
    def world(missing, c6_undef=()):
        def recipe(arm, s, sa, ms, tau, g0):
            if (arm, sa, ms, s) in missing or (arm == "k1nofac" and "all" in missing):
                return None
            if arm == "apv" and (sa, ms) == (0.125, 429) and s in (46, 49, 51):
                return _rec(_som_after(_c6_fail(T1.make_ps(), 300), 4))
            ps = T1.make_ps()
            if (arm, sa, ms, s) in c6_undef:
                ps["c1"]["v_apical"]["run_max"]["vole"] = -50.0      # the plateau never rose: C6 UNDEFINED
            return _rec(ps)
        return S.k1_reads(FakeLoader(recipe))
    k = world({"all"})                                       # every K1-NOFAC record missing
    assert k["nofac"]["n"] == 0 and k["b_undefined"]["k1nofac"]["invalid"] == 16
    assert k["falsified_b"] is None and k["falsified_a"] is False and k["H_K1_falsified"] is None
    k = world({("apv", 0.125, 429, 41)})                     # one APV record at a K1 cell missing
    assert k["apv"]["n"] == 15 and k["falsified_b"] is None
    assert k["a_records"]["invalid"] == 1 and k["falsified_a"] is None and k["H_K1_falsified"] is None
    k = world(set(), c6_undef={("k1nofac", 0.25, 429, 50)})  # a valid K1-NOFAC record with C6 UNDEFINED
    assert k["nofac"]["n"] == 15 and k["b_undefined"]["k1nofac"]["c6_undefined"] == 1 and k["falsified_b"] is None
    k = world(set(), c6_undef={("apv", 0.5, 215, 41)})       # (a): a C6-UNDEFINED record is no late fail, (a) reads
    assert k["a_records"]["c6_undefined"] == 1 and k["falsified_a"] is False and k["falsified_b"] is False
    assert k["H_K1_falsified"] is False

    def hit(arm, s, sa, ms, tau, g0):                        # (a) hit on the valid records, a GRID record missing
        if arm == "grid" and (sa, ms, s) == (0.5, 429, 51) and (tau, g0) == (40, 4):
            return None
        if arm == "apv" and (sa, ms) == (0.125, 429) and s == 46:
            return _rec(_c6_fail(T1.make_ps(), 300))         # 0 SOM spikes in [200, 300)
        return _rec(T1.make_ps())
    k = S.k1_reads(FakeLoader(hit))
    assert k["falsified_a"] is True and k["a_records"]["invalid"] == 1 and k["H_K1_falsified"] is True


# ── 7. selection, falsifiers, decision tree ────────────────────────────────────────────────────────────────────────

def _ncell(working=(), rescue=None, eraser=None, kept=None):
    out = {}
    for k in S.NMDA_CELLS:
        out[k] = {"working": k in working, "rescue": {"count": (rescue or {}).get(k, 45 if k in working else 0)},
                  "eraser": (eraser or {}).get(k, 23), "kept": (kept or {}).get(k, 21)}
    return out


def test_selection_order():
    tc = {(t, g, ms): {"selectable": True} for t, g in S.NMDA_CELLS for ms in S.PRIMARY_MS}
    w, cands, sel = S.select_cell(tc, _ncell(working=[(75, 4), (40, 8)]))
    assert sel == (215, 4, 75)                               # W_MS 215 first, then the smallest gN0, then tauN
    assert cands[:2] == [(215, 4, 75), (215, 8, 40)]
    tc[(75, 4, 215)]["selectable"] = False
    assert S.select_cell(tc, _ncell(working=[(75, 4), (40, 8)]))[2] == (215, 8, 40)
    tc[(40, 8, 215)]["selectable"] = False
    assert S.select_cell(tc, _ncell(working=[(75, 4), (40, 8)]))[2] == (304, 4, 75)
    nc = _ncell(working=[(75, 4), (40, 8)], kept={(75, 4): 20})
    assert S.select_cell(tc, nc)[2] == (304, 8, 40)          # kept 20/21 is never selected
    assert S.select_cell(tc, _ncell())[2] is None


def test_falsifiers():
    ip = {k: {"pass": True} for k in S.NMDA_CELLS}
    nc = _ncell(rescue={k: 39 for k in S.NMDA_CELLS})
    assert S.falsifiers(nc, ip, [])["F1"]
    assert S.falsifiers(_ncell(eraser={k: 22 for k in S.NMDA_CELLS}), ip, [])["F1"]     # no eraser-23 cell: fires
    nc = _ncell(working=[(40, 4)], rescue={k: 30 for k in S.NMDA_CELLS if k != (40, 4)})
    fz = S.falsifiers(nc, ip, [(40, 4)])
    assert not fz["F1"] and fz["F2"] and not fz["F3"] and not fz["F4"]
    ip_fail = dict(ip)
    ip_fail[(40, 4)] = {"pass": False}
    fz = S.falsifiers(nc, ip_fail, [(40, 4)])
    assert fz["F3"]
    fz = S.falsifiers(_ncell(working=[(40, 16), (75, 16)]), ip, [(40, 16), (75, 16)])
    assert fz["F4"] and not fz["F2"]
    fz = S.falsifiers(_ncell(), ip, [])
    assert not (fz["F2"] or fz["F3"] or fz["F4"])            # no working cell: F2-F4 cannot fire


def test_branches():
    no = {"F1": False, "F2": False, "F3": False, "F4": False}
    kw = dict(stop_engine=False, stop_id=False, fz=no, selected=True, stop_attrib=False, stop_smoke=False)
    assert S.branches(**kw)[0].startswith("9:")
    assert S.branches(**dict(kw, stop_id=True)) == ["1: STOP-ENGINE / STOP-ID: instrument; nothing else is read"]
    assert S.branches(**dict(kw, stop_engine=True, fz=dict(no, F1=True))) == [
        "1: STOP-ENGINE / STOP-ID: instrument; nothing else is read"]
    b = S.branches(**dict(kw, fz=dict(no, F2=True, F4=True)))
    assert [x[:2] for x in b] == ["3:", "5:"]                 # both recorded, no branch 9
    assert S.branches(**dict(kw, stop_smoke=True))[0].startswith("8:")
    b = S.branches(**dict(kw, selected=False, stop_attrib=True))
    assert [x[:2] for x in b] == ["6:", "7:"]
    assert S.branches(**dict(kw, selected=False, fz=dict(no, F1=True)))[0].startswith("2:")


# ── 8. score_full on a fake loader ─────────────────────────────────────────────────────────────────────────────────

ALL_KEYS = [(sa, ms, s) for sa, ms in S.F1I_CELLS for s in S.SEEDS]
OTHERS = [k for k in ALL_KEYS if k[:2] != S.CORNER]
EARLY, LATE = OTHERS[:45], OTHERS[45:51]
PASSING = [k for k in ALL_KEYS if k[:2] == S.CORNER] + OTHERS[51:]


def _fake_ref():
    defined = [(s, x, y) for s in S.SEEDS for x, y in S.PAIRS if (s, x, y) != S.CORNER_UNDEF_PAIR]
    return {"c6": {"early": EARLY, "late": LATE, "pass": PASSING, "undefined": [], "drop_ms": {}, "problems": []},
            "corner": {"defined": defined, "undefined": [S.CORNER_UNDEF_PAIR], "problems": []},
            "counts": {"early": 45, "late": 6, "pass": 21, "corner_defined": 23}, "ok": True}


class World:
    """A planted step-0 world: each arm's record from a few prototypes (shared, read-only)."""

    def __init__(self, trace_g=None, trace_ps=None, apv_ps=None, c5=None, missing=None):
        self.missing = missing or (lambda arm, key, tau, g0: False)
        self.clean = _g_ap(T1.make_ps(), 0.5)
        self.lesion = T1.make_lesion()
        self.trace_g = trace_g or (lambda t, g, key: 0.5)
        self.trace_ps = trace_ps
        self.apv_ps = apv_ps
        self.c5 = c5
        self.memo = {}

    def _trace(self, t, g, key):
        if self.trace_ps is not None:
            ps = self.trace_ps(t, g, key)
            if ps is not None:
                return ps
        val = self.trace_g(t, g, key)
        if ("t", val) not in self.memo:
            ps = _g_ap(T1.make_ps(), val)
            if self.c5:
                ps["c5"] = dict(self.c5)
            self.memo[("t", val)] = ps
        return self.memo[("t", val)]

    def recipe(self, arm, s, sa, ms, tau, g0):
        key = (sa, ms, s)
        if self.missing(arm, key, tau, g0):
            return None                                      # an invalid or missing artifact (FakeLoader lists it)
        if arm in ("cal", "grid", "ipsc12", "nomem"):
            return _rec(self._trace(tau, g0, key))
        if arm == "lesion":
            return _rec(self.lesion)
        if arm == "apv":
            if self.apv_ps is not None:
                ps = self.apv_ps(key)
                if ps is not None:
                    return _rec(ps)
            if key in EARLY:
                return _rec(self.memo.setdefault("ae", _c6_fail(T1.make_ps(), 170)))
            if key in LATE:
                return _rec(self.memo.setdefault("al", _som_after(_c6_fail(T1.make_ps(), 300), 4)))
            return _rec(self.memo.setdefault("ac", T1.make_ps()))
        return _rec(self.memo.setdefault("other", T1.make_ps()))


def _score(monkeypatch, tmp_path, world, f1i_lesions=False, nomem=None, ident_iii=None):
    monkeypatch.setattr(S, "Loader", lambda root, lines, pin: FakeLoader(world.recipe, root))
    monkeypatch.setattr(S, "f1i_reference", lambda root: _fake_ref())
    clean = {"equal": True, "problems": []}
    monkeypatch.setattr(S, "identity_i", lambda L, root: {s: dict(clean) for s in S.IDENTITY_SEEDS})
    monkeypatch.setattr(S, "identity_iii", lambda L, root: {"x": dict(ident_iii or clean)})
    monkeypatch.setattr(S, "p0_engine", lambda pin: {"sim/bridge.py": "sim/bridge.py:1"})
    monkeypatch.setattr(S, "host_summary", lambda paths: "host summary: planted")
    if f1i_lesions:
        for s in S.SEEDS:
            for ms in S.PRIMARY_MS:
                (tmp_path / F1I.art_name("lesion", s, 0.125, ms)).write_text(json.dumps(_rec(T1.make_lesion())))
    res = S.score_full(str(tmp_path), str(tmp_path), None, PIN, nomem_g0=nomem)
    S.render(res)                                            # the RESULT renders on every world
    return res


def test_full_clean_world_selects_t40_g4_ms215(monkeypatch, tmp_path):
    res = _score(monkeypatch, tmp_path, World(), f1i_lesions=True, nomem=[8])
    d = res["decision"]
    assert d["selected"] == "t40_g4_sa0p125_ms215" and d["verdict"] == "CELL SELECTED: t40_g4_sa0p125_ms215"
    assert d["branch"].startswith("9:") and d["stops"] == [] and d["falsifiers"] == []
    assert len(d["working"]) == 9 and d["selectable"][0] == "t40_g4_ms215"
    for c in res["nmda_cells"].values():
        assert c["rescue"]["count"] == 45 and c["late"]["count"] == 6 and c["kept"] == 21 and c["c6"] == 72
        assert c["eraser"] == 23 and c["grid_c7_defined"] == 216 and c["grid_c7a"] == 216 and c["working"]
        assert c["c6_apv"] == 72 - 51 and math.isclose(c["c6_fraction_attributable_to_nmda"], (72 - 21) / 72)
    assert d["selected_cell_attribution"]["apv_fails_trace_passes_items"] == []
    assert d["selected_cell_attribution"]["label"] is None
    assert res["k1"]["apv"] == {"late": 0, "early": 16, "n": 16}       # the K1 cells' APV records are early fails here
    assert res["nomem"]["g8"]["rescue"] == 45 and set(res["nomem"]["g8"]["beside_working_cells"]) == {
        "t40_g8", "t50_g8", "t75_g8"}
    assert res["identity_ii"] == {"lesion_records": 144, "clean": 144}


def test_full_seed50_label(monkeypatch, tmp_path):
    bad50 = T1.make_ps(pair_kw={1: {"newer_dies_at": 500}})

    def apv_ps(key):
        return bad50 if key == (0.125, 215, 50) else None
    res = _score(monkeypatch, tmp_path, World(apv_ps=apv_ps), f1i_lesions=True)
    a = res["decision"]["selected_cell_attribution"]
    assert a["apv_fails_trace_passes_pairs"] == [(50, "frog", "vole", S.FAIL)]
    assert a["apv_fails_trace_passes_items"] == [(50, "C7", S.FAIL)]
    assert a["label"] == "NMDA-attributed on 1/24 pairs; C6 rescue evidence is GRID-only"
    assert a["step1_apv_arm_also_at_ms304"] is True


def test_full_f1_fires_when_rescue_is_ipsp_removal(monkeypatch, tmp_path):
    six = set(EARLY[:6])
    res = _score(monkeypatch, tmp_path, World(trace_g=lambda t, g, key: 0.319 if key in six else 0.5))
    d = res["decision"]
    for c in res["nmda_cells"].values():
        assert c["rescue"]["count"] == 39 and c["rescue"]["count_alt_floor_0p3189"] == 45
        assert len(c["rescue"]["removal"]) == 6 and not c["working"]
    assert d["falsifiers"] == ["F1"] and d["branch"].startswith("2:") and d["STOP_CAL"]
    assert d["verdict"] == "STOP: STOP_CAL, STOP_FALS, F1"


def test_full_f2_and_f4(monkeypatch, tmp_path):
    res = _score(monkeypatch, tmp_path, World(trace_g=lambda t, g, key: 0.5 if t == 40 else 0.1))
    d = res["decision"]
    assert d["working"] == ["t40_g4", "t40_g8", "t40_g16"] and d["falsifiers"] == ["F2"]
    assert d["branch"].startswith("3:") and d["selected"] == "t40_g4_sa0p125_ms215" and d["STOP_FALS"]
    res = _score(monkeypatch, tmp_path, World(trace_g=lambda t, g, key: 0.5 if g == 16 else 0.1))
    assert res["decision"]["falsifiers"] == ["F4"] and res["decision"]["branch"].startswith("5:")


def test_full_stop_smoke_and_stop_attrib(monkeypatch, tmp_path):
    res = _score(monkeypatch, tmp_path, World(c5={"clause_a_delta_pa": 1.0, "clause_b_delta_pa": 14.0}),
                 f1i_lesions=True)
    d = res["decision"]
    assert d["STOP_SMOKE_global"] and d["STOP_SMOKE_selected"] and d["branch"].startswith("8:")
    silent = _g_ap(T1.make_ps(pair_kw={i: {"silent_cell": 2} for i in range(3)}), 0.5)
    res = _score(monkeypatch, tmp_path, World(trace_ps=lambda t, g, key: silent if key[:2] in (
        (0.125, 215), (0.125, 304)) else None))
    d = res["decision"]
    assert d["STOP_ATTRIB"] and d["STOP_CAL"] and [b[:2] for b in d["branches_fired"]] == ["6:", "7:"]
    assert "C7" in d["stop_cal_reading"]["failing_at_every_working_cell"]


GRID_EARLY = [k for k in EARLY if S._tg_arm(*k[:2]) == "grid"]


def test_full_missing_grid_record_is_undefined_never_f1(monkeypatch, tmp_path):
    """One GRID early record missing at every NMDA cell, on a world whose rescue sits exactly at the bar (40/45):
    read as a shortfall it would be 39 and fire F1 (branch 2, the method banked as not closing). It is UNDEFINED."""
    five = set(EARLY[:5])
    gone = GRID_EARLY[0]
    w = World(trace_g=lambda t, g, key: 0.319 if key in five else 0.5,
              missing=lambda arm, key, tau, g0: arm == "grid" and key == gone)
    res = _score(monkeypatch, tmp_path, w, f1i_lesions=True)
    d = res["decision"]
    assert len(res["invalid"]) == 9
    assert d["UNDEFINED"] is True and d["invalid_or_missing"] == 9
    assert d["verdict"] == ("UNDEFINED: re-run the 9 invalid or missing artifact(s), then re-score; not a pass, "
                            "not a falsifier")
    assert d["branches_fired"] == [d["verdict"]] and d["stops"] == [] and d["falsifiers"] == []
    assert d["falsifier_reads"] is None and d["STOP_FALS"] is None and d["STOP_CAL"] is None
    assert d["selected"] is None and d["working"] is None and not d["STOP_ID"]
    for c in res["nmda_cells"].values():
        assert c["rescue"]["count"] == 39 and c["rescue"]["invalid"] == 1 and c["invalid_trace_grid"] == 1
    assert "VERDICT: UNDEFINED" in S.render(res)
    # the shortfall alone would fire F1: falsifiers() refuses to read it
    with pytest.raises(ValueError, match="incomplete data"):
        S.falsifiers({k: res["nmda_cells"][LN.nmda_tag(*k)] for k in S.NMDA_CELLS},
                     {k: res["ipsc12"][LN.nmda_tag(*k)] for k in S.NMDA_CELLS}, [])
    # six cal / GRID early records missing (the skeptic's probe): UNDEFINED, not "STOP: STOP_CAL, STOP_FALS, F1"
    six = set(EARLY[4:10])
    res = _score(monkeypatch, tmp_path, World(missing=lambda arm, key, tau, g0: arm in ("cal", "grid")
                                              and key in six), f1i_lesions=True)
    assert len(res["invalid"]) == 54 and res["decision"]["UNDEFINED"] and res["decision"]["falsifiers"] == []
    # STOP-ID (branch 1) still comes first: nothing else is read
    stop = {"equal": False, "problems": [], "verdict": "STOP-ID (same host class pool)"}
    res = _score(monkeypatch, tmp_path, w, f1i_lesions=True, ident_iii=stop)
    d = res["decision"]
    assert d["STOP_ID"] and d["UNDEFINED"] is False and d["falsifier_reads"] is None and d["falsifiers"] == []
    assert d["branches_fired"] == ["1: STOP-ENGINE / STOP-ID: instrument; nothing else is read"]


def test_full_missing_ipsc12_record_is_undefined_never_f3(monkeypatch, tmp_path):
    """One IPSC-12 seed missing at every NMDA cell: read as a C6 shortfall (7/8 at W_MS 215) it fires F3. UNDEFINED."""
    w = World(missing=lambda arm, key, tau, g0: arm == "ipsc12" and key == (0.125, 215, 41))
    res = _score(monkeypatch, tmp_path, w, f1i_lesions=True)
    d = res["decision"]
    assert len(res["invalid"]) == 9 and d["UNDEFINED"] and d["falsifiers"] == [] and d["STOP_FALS"] is None
    assert d["verdict"].startswith("UNDEFINED: re-run the 9 invalid or missing artifact(s)")
    for t, ip in res["ipsc12"].items():
        assert ip["invalid"] == 1 and ip["c6_ms215"] == 7 and not ip["pass"]
    assert "F3" not in d["verdict"] and not d["selected"]


def test_full_missing_nomem_record_leaves_the_decision_and_undefines_nomem(monkeypatch, tmp_path):
    w = World(missing=lambda arm, key, tau, g0: arm == "nomem" and key == EARLY[0])
    res = _score(monkeypatch, tmp_path, w, f1i_lesions=True, nomem=[8])
    assert res["decision"]["verdict"] == "CELL SELECTED: t40_g4_sa0p125_ms215"
    n8 = res["nomem"]["g8"]
    assert n8["rescue"] is None and n8["invalid"] == 1 and n8["rescue_on_valid_records"] == 44


def test_full_with_nothing_on_disk_is_stop_id(tmp_path):
    if not HAVE_F1I:
        pytest.skip("f1i step-0 artifacts not in this checkout")
    res = S.score_full(str(tmp_path), F1I_ROOT, None, PIN)
    d = res["decision"]
    assert d["STOP_ID"] and d["branches_fired"] == ["1: STOP-ENGINE / STOP-ID: instrument; nothing else is read"]
    assert len(res["invalid"]) == 1042 and "STOP_ID" in d["stops"]
    assert "DECISION" in S.render(res)


# ── 9. identity mode end to end ────────────────────────────────────────────────────────────────────────────────────

def _transplant(base, donor):
    r = copy.deepcopy(base)
    ps, dp = r["per_seed"][0], donor["per_seed"][0]
    r["flags"]["f1j"] = copy.deepcopy(donor["flags"]["f1j"])
    ps["config_readback"]["f1j"] = copy.deepcopy(dp["config_readback"]["f1j"])
    ps["c5"]["f1j_nmda"] = copy.deepcopy(dp["c5"]["f1j_nmda"])
    for k in ("c1", "c2b"):
        ps[k]["nmda"] = copy.deepcopy(dp[k]["nmda"])
    for c, dc in zip(ps["c3"], dp["c3"]):
        c["nmda"] = copy.deepcopy(dc["nmda"])
    return r


def _sidecar(path, argv, sha=PIN):
    with open(str(path) + ".prov.json", "w") as f:
        json.dump({"git_sha": sha, "git_dirty": False, "source_kind": "git_archive",
                   "source_manifest_verified_at_start": True, "source_manifest_verified_at_exit": True,
                   "argv": ["/x/research/runners/cs_race_f1.py"] + argv}, f)


@pytest.fixture(scope="module")
def wave1_root(tmp_path_factory, real):
    if not HAVE_F1I:
        pytest.skip("f1i step-0 artifacts not in this checkout")
    d = tmp_path_factory.mktemp("f1j_wave1")
    lines_path = d / "lines.txt"
    lines_path.write_text(LN.render(LN.wave1()))
    lines = F1I.parse_lines(str(lines_path))
    root = d / "root"
    root.mkdir()

    def put(name, rec):
        p = root / name
        p.write_text(json.dumps(rec))
        _sidecar(p, lines[S.STEP0_DIR + "/" + name])

    for s in S.IDENTITY_SEEDS:
        with open(os.path.join(F1I_ROOT, F1I.art_name("trace", s, *S.CORNER))) as f:
            put(S.art_name("identity", s), json.load(f))
    for sa, ms in S.F1I_CELLS:
        for s in S.SEEDS:
            with open(os.path.join(F1I_ROOT, F1I.art_name("trace", s, sa, ms))) as f:
                base = json.load(f)
            put(S.art_name("apv", s, sa, ms), _transplant(base, real["apv"]))
            if (sa, ms) in S.K1_CELLS:
                nf = _transplant(base, real["k1nofac"])
                nf["flags"]["som_nofac"] = True
                put(S.art_name("k1nofac", s, sa, ms), nf)
    return {"root": root, "lines": str(lines_path), "lines_map": lines}


def test_identity_mode_passes_on_planted_wave1(wave1_root):
    res = S.score_identity(str(wave1_root["root"]), F1I_ROOT, wave1_root["lines"], PIN)
    assert res["invalid"] == {} and res["stop_id_problems"] == {} and res["consistency"] == {}
    assert res["f1i_reference_ok"] and all(v["equal"] for v in res["identity_i"].values())
    assert len(res["identity_iii"]) == 72 and all(v["equal"] for v in res["identity_iii"].values())
    assert res["verdict"] == "IDENTITY PASS: queue wave 2 (952 lines), no ruling needed"
    assert "VERDICT: IDENTITY PASS" in S.render(res)
    assert S.main(["--mode", "identity", "--pin", PIN, "--root", str(wave1_root["root"]), "--f1i-root", F1I_ROOT,
                   "--lines", wave1_root["lines"]]) in (0, None)


def _swap(path, rec):
    with open(path) as f:
        old = f.read()
    with open(path, "w") as f:
        f.write(json.dumps(rec))
    return old


def test_identity_mode_undefined_and_stop_id(wave1_root):
    root = wave1_root["root"]
    apv = root / S.art_name("apv", 47, 0.25, 304)
    os.rename(str(apv), str(apv) + ".away")
    try:
        res = S.score_identity(str(root), F1I_ROOT, wave1_root["lines"], PIN)
        assert not res["STOP_ID"] and res["verdict"].startswith("IDENTITY UNDEFINED")
        assert S.art_name("apv", 47, 0.25, 304) in res["undefined"]
    finally:
        os.rename(str(apv) + ".away", str(apv))
    nf = root / S.art_name("k1nofac", 46, 0.125, 429)
    with open(nf) as f:
        rec = json.load(f)
    rec["per_seed"][0]["c3"][0]["nmda"]["reset_readback"] = {"s_max": 0.01}
    old = _swap(nf, rec)
    try:
        res = S.score_identity(str(root), F1I_ROOT, wave1_root["lines"], PIN)
        assert res["STOP_ID"] and res["verdict"] == "STOP-ID: post and STOP"
        assert list(res["stop_id_problems"]) == [S.art_name("k1nofac", 46, 0.125, 429)]
    finally:
        with open(nf, "w") as f:
            f.write(old)


def test_identity_iii_host_class_rule(wave1_root, monkeypatch):
    root = wave1_root["root"]
    p = root / S.art_name("apv", 41, 0.125, 215)
    with open(p) as f:
        rec = json.load(f)
    rec["per_seed"][0]["c1"]["word_rates_hz"]["vole"] += 0.5
    old = _swap(p, rec)
    try:
        def run(c_new, c_ref):
            monkeypatch.setattr(S, "_host", lambda path: ("h", c_ref if path.startswith(F1I_ROOT) else c_new))
            L = S.Loader(str(root), F1I.parse_lines(wave1_root["lines"]), PIN)
            iii = S.identity_iii(L, F1I_ROOT, cells=[(0.125, 215)], seeds=[41])
            return iii, S._identity_stop({}, iii)
        iii, (stop, pending) = run("workstation", "pool")
        row = iii[S.art_name("apv", 41, 0.125, 215)]
        assert row["equal"] is False and row["verdict"].startswith("HOST-CLASS PENDING") and not stop
        assert pending == [S.art_name("apv", 41, 0.125, 215)] and row["diffs"]
        iii, (stop, pending) = run("pool", "pool")
        assert iii[S.art_name("apv", 41, 0.125, 215)]["verdict"] == "STOP-ID (same host class pool)" and stop
        iii, (stop, _p) = run("unknown", "unknown")
        assert not stop                                      # an unknown class is never "the same class"
    finally:
        with open(p, "w") as f:
            f.write(old)


def test_identity_stop_undefined_rule():
    rows = {"a": {"equal": None, "problems": ["artifact unreadable"]}}
    assert S._identity_stop(rows, {}) == (False, [])
    assert S._identity_stop(rows, {}, undefined_is_stop=True) == (True, [])
    assert S._identity_stop({"b": {"equal": True, "problems": ["x"]}}, {})[0] is True


def test_artifact_validity_off_pin_and_argv(wave1_root, tmp_path):
    lines = wave1_root["lines_map"]
    name = S.art_name("apv", 41, 0.125, 215)
    src = wave1_root["root"] / name
    dst = tmp_path / name
    dst.write_text(src.read_text())
    _sidecar(dst, lines[S.STEP0_DIR + "/" + name], sha="0" * 40)
    L = S.Loader(str(tmp_path), lines, PIN)
    assert L.get("apv", 41, 0.125, 215) is None and any("git_sha" in p for p in L.invalid[name])
    _sidecar(dst, lines[S.STEP0_DIR + "/" + name][:-2] + ["--out", "x.json"])
    L = S.Loader(str(tmp_path), lines, PIN)
    assert L.get("apv", 41, 0.125, 215) is None and any("argv" in p for p in L.invalid[name])


def test_read_lines_refuses_repeated_wave3_outputs(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text(LN.render(LN.wave1()[:3]))
    b = tmp_path / "b.txt"
    b.write_text(LN.render(LN.wave1()[2:4]))
    with pytest.raises(ValueError):
        S.read_lines(str(a), str(b))
    c = tmp_path / "c.txt"
    c.write_text(LN.render(LN.wave1()[3:5]))
    assert len(S.read_lines(str(a), str(c))) == 5


# ── 10. P0' ────────────────────────────────────────────────────────────────────────────────────────────────────────

_P0_NEEDLES = {"sim/config.py": "enable_apical_nmda: bool = False",
               "sim/bridge.py": "def apical_nmda_gating_step(",
               "research/runners/cs_race_f1.py": "def _reset_nmda(self):",
               "tests/test_apical_nmda.py": "def test_replay_match_single_slot(",
               "tests/test_cs_race_f1j_flags.py": "def test_runner_writes_the_gating_only_in_the_reset(",
               "research/biology/apical-nmda-conductance.md": "id: apical-nmda-conductance"}


def test_p0_needles_exist_in_this_checkout():
    for path, needle in _P0_NEEDLES.items():
        with open(os.path.join(REPO, path)) as f:
            assert needle in f.read(), path


def test_p0_engine_reads_file_and_line(monkeypatch):
    texts = {p: "x\n%s\n" % n for p, n in _P0_NEEDLES.items()}
    monkeypatch.setattr(S, "_git_show", lambda rev, path: texts.get(path))
    out = S.p0_engine(PIN)
    assert out == {p: "%s:2" % p for p in _P0_NEEDLES}
    texts["sim/bridge.py"] = "nothing here\n"
    del texts["tests/test_apical_nmda.py"]
    out = S.p0_engine(PIN)
    assert out["sim/bridge.py"] is False and out["tests/test_apical_nmda.py"] is None
    assert all(v is None for v in S.p0_engine(None).values())
    assert math.isclose(S.RESCUE_GAP_FLOOR, 0.32) and math.isclose(S.RESCUE_GAP_FLOOR_ALT, 0.3189)
