"""CI guard -- the cs-race f1i STEP-0 scorer (research/runners/cs_race_f1i_score.py; packet
research/coordination/cs-race-f1i-PACKET.md, Step 0).

Synthetic records in the runner's record shape (cal() with --som), so every item is planted, not sampled:
  1. a clean record scores PASS on C0-C4, C6, C7 (every pair) and C8, with the C7(a) latency read from the per-ms max;
  2. PLANTED C7 FAILS, one per sub-item: the older plateau never ends (a); the newer plateau dies (b); one SOM cell
     silent, or the older g_ap peak after the termination ms (c); the lesion leaves the older plateau down, or the
     rate form of C3 passes under the lesion (d). Each fails the pair and the seed;
  3. C7(0): an older plateau that never latched makes the pair UNDEFINED, never a pass (and the seed UNDEFINED unless
     another pair FAILs);
  4. C8 counts SOM spikes only in [pulse end + 50, pulse end + 650): the volley's own tail is excluded, 1 Hz per cell is
     the bar;
  5. the cell rule (8/8 defined passes), selection order (smallest W_SA, then W_MS), UNATTRIBUTED, STOP-ATTRIB and the
     branches B1-B3 by the packet's text; an invalid artifact is UNDEFINED;
  6. artifact validity: a sidecar off the pin or whose argv differs from the pool line is refused; identity hashing
     ignores only wall_s;
  7. the landed step-0 data (skipped when absent): no invalid artifact, no STOP-ID read-back problem, no runner field
     that disagrees with its re-derivation, identity (i) equal, and the C1 / C3 / C6 counts the driver posted.
"""
import copy
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import cs_race_f1i_score as S  # noqa: E402

VH = S.V_HOLD
UP, DOWN = -24.15, -77.28


def _series(T, base, segs=()):
    out = [base] * T
    for lo, hi, v in segs:
        for t in range(lo, min(hi, T)):
            out[t] = v
    return out


def _som(T, per_word, events, lesion=False):
    spikes = [0] * T
    for ms, _c in events:
        spikes[ms] += 1
    return {"n_som": 8, "lesion": lesion, "nofac": False, "w_ms": 215.0, "w_sa": 0.125,
            "gate_cs_som_at_measurement": 0.0 if lesion else 1.0,
            "g_ap_max_run": 0.0 if lesion else max([0.0] + [max(d["g_ap_max"]) for d in per_word.values()]),
            "som_spikes_per_ms": spikes, "som_cell_events": [list(e) for e in events], "per_word_ms": per_word,
            "reset_readback": {"g_ap_max": 0.0, "gate_cs_som": 0.0 if lesion else 1.0, "msom_u_max": 0.0,
                               "msom_x_min": 1.0},
            "stp_trial_start": {"stp_reset": False, "ee": None, "msom": None},
            "build_readback": {"problems": [], "apical_targets_are_mention_cells": True, "g_ap_len": 354,
                               "realized_stp": {"msom": {"U": [0.049], "tau_f": [1797.0], "tau_d": [399.0],
                                                         "conn_type": [1]},
                                                "ee": {"U": [0.3], "tau_f": [150.0], "tau_d": [200.0],
                                                       "conn_type": [0]}}}}


def _volley(lo, hi, n_per_cell=2, cells=range(8)):
    ev = []
    step = max(1, (hi - lo) // (n_per_cell * 8 + 1))
    t = lo
    for _ in range(n_per_cell):
        for c in cells:
            ev.append((t, c))
            t = min(t + step, hi - 1)
    return sorted(ev)


def _rates(counts, ms_len):
    return {w: round(counts.get(w, 0) / (8 * ms_len / 1000.0), 4) for w in S.MENTION_WORDS}


def _trial_common(word_counts, ms_len, window, pulses):
    rates = _rates(word_counts, ms_len)
    return {"window_ms": list(window), "window_len_ms": ms_len,
            "ments_spikes": [word_counts.get(w, 0) for w in S.MENTION_WORDS], "word_rates_hz": rates,
            "pulse_events": [{"word": w, "t0": t0, "t1": t1 - 1, "n_cells": 8, "ms": t1 - t0, "min_cell_spikes": 5,
                              "total_spikes": 40, "per_cell_spikes": [5] * 8} for w, t0, t1 in pulses]}


def make_c1(*, plateau=True, c8_spikes=0, tail_spikes=0, lesion=False):
    T = 900
    vmin = _series(T, DOWN, [(10, T, UP)] if plateau else [(10, 150, UP)])
    g = _series(T, 0.0) if lesion else _series(T, 0.0, [(120, 160, 0.3)])
    ev = _volley(60, 150) + [(150 + i, i % 8) for i in range(tail_spikes)] + [(250 + 40 * i, i % 8)
                                                                               for i in range(c8_spikes)]
    som = _som(T, {"vole": {"v_apical_min": vmin, "v_apical_max": list(vmin), "g_ap_max": g}}, ev, lesion=lesion)
    r = _trial_common({"vole": 39}, 650, S.C1_WIN, [("vole", 0, 150)])
    vw = {w: {"min": DOWN, "max": DOWN} for w in S.MENTION_WORDS}
    vw["vole"] = {"min": round(min(vmin[150:800]), 4), "max": round(max(vmin[150:800]), 4)}
    r.update({"item": "c1", "pulsed_rate_hz": r["word_rates_hz"]["vole"], "unpulsed_max_hz": 0.0,
              "burst_ms": None, "latch_ms": None, "som": som,
              "v_apical": {"v_hold": VH, "window": vw, "run_max": {w: (18.3 if w == "vole" else DOWN)
                                                                    for w in S.MENTION_WORDS}}})
    return r


def make_c2b(lesion=False):
    T = 2100
    return {"item": "c2b", "max_word_hz": 0.0, "som": _som(T, {}, [], lesion=lesion)}


def make_c3(older, newer, *, older_latched=True, older_end=360, newer_dies_at=None, silent_cell=None,
            late_peak=False, lesion=False, rate_pass=True):
    T = 1200
    if not older_latched:
        o = _series(T, DOWN)
    elif older_end is None:
        o = _series(T, DOWN, [(10, T, UP)])
    else:
        o = _series(T, DOWN, [(10, older_end, UP)])
    n = _series(T, DOWN, [(310, T if newer_dies_at is None else newer_dies_at, UP)])
    g_o = _series(T, 0.0) if lesion else _series(T, 0.0, [(330, 345, 0.3), (345, 356, 0.6)] + (
        [(400, 410, 0.9)] if late_peak else []))
    cells = [c for c in range(8) if c != silent_cell]
    ev = _volley(320, 450, cells=cells)
    som = _som(T, {older: {"v_apical_min": o, "v_apical_max": list(o), "g_ap_max": g_o},
                   newer: {"v_apical_min": n, "v_apical_max": list(n), "g_ap_max": list(g_o)}}, ev, lesion=lesion)
    counts = {newer: 28} if rate_pass else {newer: 28, older: 28}
    r = _trial_common(counts, 350, S.C3_WIN, [(older, 0, 150), (newer, 300, 450)])
    vw = {w: {"min": DOWN, "max": DOWN} for w in S.MENTION_WORDS}
    vw[older] = {"min": round(min(o[600:950]), 4), "max": round(max(o[600:950]), 4)}
    vw[newer] = {"min": round(min(n[600:950]), 4), "max": round(max(n[600:950]), 4)}
    r.update({"item": "c3", "word1_rate_hz": r["word_rates_hz"][older], "word2_rate_hz": r["word_rates_hz"][newer],
              "som": som, "v_apical": {"v_hold": VH, "window": vw}})
    return r


def make_ps(seed=41, pair_kw=None, lesion=False, c1_kw=None):
    pair_kw = pair_kw or {}
    c3s = [make_c3(x, y, lesion=lesion, **(pair_kw.get(i) or {})) for i, (x, y) in enumerate(S.PAIRS)]
    ps = {"seed": seed, "c1": make_c1(lesion=lesion, **(c1_kw or {})), "c2b": make_c2b(lesion), "c3": c3s,
          "c5": {"clause_a_delta_pa": 12.0, "clause_b_delta_pa": 14.0},
          "c3_apical": [{"older": x, "newer": y, "older_plateau_ended": c["v_apical"]["window"][x]["max"] < VH}
                        for (x, y), c in zip(S.PAIRS, c3s)]}
    return ps


def make_lesion(pair_kw=None):
    """A clean lesion: the older plateau holds through the window and rate-C3 fails (rate_pass False puts the older
    word at 10 Hz)."""
    kw = {i: {"older_end": None, "rate_pass": False} for i in range(3)}
    for i, d in (pair_kw or {}).items():
        kw[i] = dict(kw[i], **d)
    return make_ps(pair_kw=kw, lesion=True)


# ── 1. a clean record ───────────────────────────────────────────────────────────────────────────────────────────────

def test_clean_record_passes_every_item():
    it = S.seed_items(make_ps(), make_lesion())
    for k in S.ITEMS_SEL:
        assert it[k] == S.PASS, (k, it[k])
    for p in it["c7_pairs"]:
        assert p["status"] == S.PASS
        assert p["termination_ms"] == 360 and p["latency_ms"] == 60
        assert p["g_ap_peak_ms"] == 345 and p["g_ap_onset_ms"] == 330
    assert S.consistency(make_ps()) == []


# ── 2. planted C7 failures ──────────────────────────────────────────────────────────────────────────────────────────

def _pair0(ps_kw, lesion_kw=None):
    it = S.seed_items(make_ps(pair_kw={0: ps_kw}), make_lesion(lesion_kw))
    return it, it["c7_pairs"][0]


def test_c7a_fails_when_older_plateau_never_ends():
    it, p = _pair0({"older_end": None})
    assert p["termination_ms"] is None and p["a"] is False
    assert p["status"] == S.FAIL and it["C7"] == S.FAIL
    # B3 reading: SOM fired and g_ap > 0 with the plateau holding -> (c) is read without the timing clause
    assert p["c"] is True


def test_c7a_fails_when_termination_is_later_than_300_ms():
    it, p = _pair0({"older_end": 640})
    assert p["latency_ms"] == 340 and p["a"] is False and it["C7"] == S.FAIL


def test_c7b_fails_when_newer_plateau_dies():
    it, p = _pair0({"newer_dies_at": 500})
    assert p["b"] is False and p["status"] == S.FAIL and it["C7"] == S.FAIL


def test_c7c_fails_when_a_som_cell_is_silent():
    it, p = _pair0({"silent_cell": 3})
    assert p["c"] is False and p["c_fail_reason"].startswith("a SOM cell silent")
    assert p["c_alt_pooled"] is True          # pooled reading still has >= 8 spikes
    assert it["C7"] == S.FAIL


def test_c7c_fails_when_g_ap_peaks_after_termination():
    it, p = _pair0({"late_peak": True})
    assert p["g_ap_peak_ms"] == 400 and p["termination_ms"] == 360
    assert p["c"] is False and p["c_fail_reason"] == "older g_ap peak at or after the termination ms"
    assert p["diag_onset_precedes_term"] is True   # diagnostic only; the bar text says PEAK
    assert it["C7"] == S.FAIL


def test_c7d_fails_when_lesion_lets_older_plateau_end():
    it, p = _pair0({}, {0: {"older_end": 700}})
    assert p["d"] is False and it["C7"] == S.FAIL


def test_c7d_fails_when_rate_c3_passes_under_lesion():
    it, p = _pair0({}, {0: {"rate_pass": True}})
    assert p["lesion_rate_c3"] is True and p["d"] is False and it["C7"] == S.FAIL


def test_c7d_undefined_without_lesion_record():
    it = S.seed_items(make_ps(), None)
    assert all(p["d"] is None and p["status"] == S.UNDEF for p in it["c7_pairs"])
    assert it["C7"] == S.UNDEF


# ── 3. C7(0) ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_unlatched_older_makes_pair_undefined_not_pass():
    it = S.seed_items(make_ps(pair_kw={1: {"older_latched": False}}), make_lesion())
    assert it["c7_pairs"][1]["status"] == S.UNDEF
    assert it["C7"] == S.UNDEF


def test_undefined_pair_with_a_failing_pair_is_fail():
    it = S.seed_items(make_ps(pair_kw={1: {"older_latched": False}, 2: {"newer_dies_at": 500}}), make_lesion())
    assert it["C7"] == S.FAIL


# ── 4. C8 window ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_c8_window_and_bar():
    assert S.seed_items(make_ps(c1_kw={"tail_spikes": 40}), make_lesion())["C8"] == S.PASS   # the volley's tail
    it = S.seed_items(make_ps(c1_kw={"c8_spikes": 4}), make_lesion())
    assert it["c8_rate_hz"] == round(4 / 4.8, 4) and it["C8"] == S.PASS
    it = S.seed_items(make_ps(c1_kw={"c8_spikes": 5}), make_lesion())
    assert it["c8_rate_hz"] > 1.0 and it["C8"] == S.FAIL
    assert it["C8_alt_poolsum"] == S.FAIL


def test_c6_undefined_when_plateau_never_reached_v_hold():
    ps = make_ps()
    ps["c1"]["v_apical"]["run_max"]["vole"] = -50.0
    assert S.seed_items(ps, make_lesion())["C6"] == S.UNDEF
    ps = make_ps(c1_kw={"plateau": False})
    it = S.seed_items(ps, make_lesion())
    assert it["C6"] == S.FAIL and it["c6_drop_ms"] == 150


def test_termination_ms_edges():
    T = 1000
    assert S.termination_ms(_series(T, DOWN), VH) == S.Y_ON
    assert S.termination_ms(_series(T, DOWN, [(0, 949, UP)]), VH) == 949
    assert S.termination_ms(_series(T, DOWN, [(0, 950, UP)]), VH) is None
    assert S.termination_ms(_series(T, DOWN, [(0, 400, UP), (500, 501, UP)]), VH) == 501


# ── 5. cell rule, selection, attribution, branches ─────────────────────────────────────────────────────────────────

def _rows(n_fail_c7=0, undefined=False, c8_fail=False):
    good = S.seed_items(make_ps(), make_lesion())
    rows = {}
    for i, s in enumerate(S.SEEDS):
        r = copy.deepcopy(good)
        if i < n_fail_c7:
            r = S.seed_items(make_ps(pair_kw={0: {"late_peak": True}}), make_lesion())
        if c8_fail:
            r["C8"] = S.FAIL
        rows[s] = r
    if undefined:
        rows[S.SEEDS[-1]] = None
    return rows


def test_cell_rule_and_selection_order():
    cells = {(sa, ms): S.summarize_cell(_rows(n_fail_c7=1)) for sa in S.W_SA for ms in S.W_MS}
    cells[(0.25, 215)] = S.summarize_cell(_rows())
    cells[(0.5, 215)] = S.summarize_cell(_rows())
    d = S.decide(cells)
    assert d["selected"] == "sa0p25_ms215" and not d["STOP_CAL"]
    cells[(0.25, 215)] = S.summarize_cell(_rows(undefined=True))       # an invalid artifact is not a pass
    assert cells[(0.25, 215)]["C7"]["full"] is False
    assert S.decide(cells)["selected"] == "sa0p5_ms215"


def test_unattributed_and_stop_attrib():
    cells = {(sa, ms): S.summarize_cell(_rows(n_fail_c7=8)) for sa in S.W_SA for ms in S.W_MS}
    for c in cells.values():
        assert c["C3"]["full"] and c["unattributed"] and not c["selectable"]
        assert c["c7_sub"]["c_fail_reasons"] == {"older g_ap peak at or after the termination ms": 8}
    d = S.decide(cells)
    assert d["STOP_CAL"] and d["STOP_ATTRIB"] and "STOP_ATTRIB" in d["stops"]
    assert d["B3"] is False      # (c) is not passing, so B3 cannot fire


def test_branches_b1_b2_b3():
    b1 = {(sa, ms): S.summarize_cell({s: S.seed_items(make_ps(pair_kw={0: {"newer_dies_at": 500}}), make_lesion())
                                      for s in S.SEEDS}) for sa in S.W_SA for ms in S.W_MS}
    d = S.decide(b1)
    assert d["B1"] and not d["B3"]
    b2 = {(sa, ms): S.summarize_cell(_rows(c8_fail=True)) for sa in S.W_SA for ms in S.W_MS}
    assert S.decide(b2)["B2"]
    b3 = {(sa, ms): S.summarize_cell({s: S.seed_items(make_ps(pair_kw={i: {"older_end": None} for i in range(3)}),
                                                      make_lesion()) for s in S.SEEDS})
          for sa in S.W_SA for ms in S.W_MS}
    d = S.decide(b3)
    assert d["B3"] and not d["B1"]


# ── 6. validity and identity ───────────────────────────────────────────────────────────────────────────────────────

def _write_artifact(tmp, name, rec, argv_tail, sha=S.PIN, dirty=False):
    p = tmp / name
    p.write_text(json.dumps(rec))
    (tmp / (name + ".prov.json")).write_text(json.dumps({
        "git_sha": sha, "git_dirty": dirty, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
        "source_manifest_verified_at_exit": True, "argv": ["/x/research/runners/cs_race_f1.py"] + argv_tail}))


def test_load_artifact_refuses_off_pin_and_argv_mismatch(tmp_path):
    name = "cal_sa0p125_ms215_s41.json"
    rel = S.STEP0_DIR + "/" + name
    argv = ["--calibrate", "--seeds", "41", "--out", rel]
    lines = {rel: argv}
    _write_artifact(tmp_path, name, {"x": 1}, argv)
    assert S.load_artifact(str(tmp_path), name, lines)[1] == []
    _write_artifact(tmp_path, name, {"x": 1}, argv, sha="0" * 40)
    assert any("git_sha" in p for p in S.load_artifact(str(tmp_path), name, lines)[1])
    _write_artifact(tmp_path, name, {"x": 1}, argv[:-2] + ["--som", "304", "--out", rel])
    assert any("argv" in p for p in S.load_artifact(str(tmp_path), name, lines)[1])
    assert S.load_artifact(str(tmp_path), "missing.json", lines)[0] is None


def test_parse_lines_reads_the_runner_argv(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("cd x && bash tools/memcap.sh 2 -- env A=1 .venv/bin/python -u -m research.runners.cs_race_f1 "
                 "--calibrate --seeds 41 --out research/x.json\twhy\n")
    assert S.parse_lines(str(f)) == {"research/x.json": ["--calibrate", "--seeds", "41", "--out", "research/x.json"]}


def test_canon_hash_ignores_only_wall_s():
    a = {"k": 1, "wall_s": 3.0}
    assert S.canon_hash(a) == S.canon_hash({"k": 1, "wall_s": 9.0}) == S.canon_hash({"k": 1})
    assert S.canon_hash(a) != S.canon_hash({"k": 2})


def test_stop_id_problems_catch_an_open_lesion_gate():
    rec = {"grid": {"U": 0.3, "tau_f": 150.0}, "per_seed": [make_ps(lesion=True)]}
    assert S.stop_id_problems(rec, "lesion") == []
    rec["per_seed"][0]["c1"]["som"]["gate_cs_som_at_measurement"] = 1.0
    assert any("gate_cs_som" in p for p in S.stop_id_problems(rec, "lesion"))


# ── 7. the landed data ──────────────────────────────────────────────────────────────────────────────────────────────

_ROOT = os.path.join(REPO, S.STEP0_DIR)


@pytest.mark.skipif(not os.path.exists(os.path.join(_ROOT, "cal_sa0p125_ms215_s41.json")),
                    reason="f1i step-0 artifacts not in this checkout")
def test_landed_step0_reads_clean():
    res = S.score(_ROOT, os.path.join(REPO, S.F1H_DIR), os.path.join(REPO, S.LINES))
    assert res["invalid"] == {} and res["stop_id_problems"] == {} and res["consistency"] == {}
    assert all(v["hash_equal_minus_wall_s"] for v in res["identity_i"].values())
    assert res["identity_ii"] == {"lesion_records": 72, "clean": 72}
    tot = {k: sum(c[k][S.PASS] for c in res["cells"].values()) for k in ("C1", "C3", "C6")}
    assert tot == {"C1": 24, "C3": 18, "C6": 21}     # the driver's [f1i-0] counts, scored independently


def test_lesion_attribution_fraction():
    att = S.summarize_cell(_rows())["c7_sub"]["attribution"]
    assert att["older_plateau_ended"] == {"trace": 24, "lesion": 0, "n_pairs": 24, "fraction_attributable": 1.0}
    assert att["rate_c3_pair_pass"]["fraction_attributable"] == 1.0
    rows = {s: S.seed_items(make_ps(), make_lesion({i: {"older_end": 500} for i in range(3)})) for s in S.SEEDS}
    att = S.summarize_cell(rows)["c7_sub"]["attribution"]
    assert att["older_plateau_ended"]["lesion"] == 24 and att["older_plateau_ended"]["fraction_attributable"] == 0.0
