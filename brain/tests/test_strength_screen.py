# Capability: Memory strength and forgetting
# Record: brain/research/2026-11-27-fable-review-dg-port.md
# Expected-fail: none (harness plumbing on the stub cortex and exact verdicts on synthetic records; no model weights)
"""strength_screen: on a stub life with night snapshots the sims stage reproduces every banked fetch, the NULL rebuild equals
the snapshot strengths and arm A on every row, the RS rebuild from an OFF life equals a life lived with the Brain switch on
(no fetch boosts in either), and the verdict is exact on synthetic counts with every clause failing alone."""
import copy
import json
import os

import pytest

from brain.experiments import strength_screen as SS
from brain.hippocampus import Hippocampus
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)

SEEDS = (319, 320, 321)


def _life(tmp_path, name, days=4, extra=None):
    from brain.experiments import life_kw
    snaps = tmp_path / name / "snaps"
    argv = ["--seed", "7", "--days", str(days), "--full-probe-nights", "1", "--frozen-probe",
            "--homes", str(tmp_path / name / "h"), "--out", str(tmp_path / name / "o" / "life_s7"),
            "--night-snapshots", str(snaps)]
    if extra:
        argv += ["--brain-kw-extra", json.dumps(extra)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    return json.load(open(str(tmp_path / name / "o" / "life_s7") + ".json")), str(snaps)


def _sims(tmp_path, res, snaps, nights):
    out = str(tmp_path / "scr")
    SS.run_sims(res, snaps, StubCortex(), out, weekend=False, nights=nights, log=lambda *a: None)
    return out


# ------------------------------------------------------------------ the real path on a stub life
def test_guard():
    with pytest.raises(SystemExit):
        SS.guard({})
    SS.guard({"POOL_NODE": "1"})
    SS.guard({"BRAIN_ALLOW_W3090": "1"})


def test_sims_and_null_rebuild_reproduce_a_stub_life(tmp_path):
    res, snaps = _life(tmp_path, "a")
    d = _sims(tmp_path, res, snaps, range(1, 5))
    units = [json.load(open(os.path.join(d, f"sims_s7_n{n}.json"))) for n in range(1, 5)]
    assert all(not u["void"] and u["rows"] and u["commit"] is not None and u["host"] for u in units)
    assert all(len(r["sims"]) == u["store"] for u in units for r in u["rows"])
    assert any(r["cls"] == "told" and r["right_set"] for u in units for r in u["rows"])   # not vacuous
    assert sum(r["a_idx"] is not None for u in units for r in u["rows"]) > 0
    with pytest.raises(SystemExit, match="GATED"):            # the full arm set never runs before HEADROOM
        SS.run_screen(d, 7, None)
    trip = SS.run_screen(d, 7, None, tripwire_only=True)
    assert trip["null_mismatch"] == 0 and trip["state_mismatch"] == 0 and trip["arms"] == ["NULL"]
    assert not os.path.exists(os.path.join(d, "scr_s7.json")) and all("rows" not in n for n in trip["nights"])
    scr = SS.run_screen(d, 7, "HEADROOM")
    assert scr["null_mismatch"] == 0 and scr["state_mismatch"] == 0
    assert scr["night_records"]["RS"] and all("g" in r and "changed" in r for r in scr["night_records"]["RS"])
    assert all("engagement" in n and "tanh_right" in n["engagement"] for n in scr["nights"])
    for n in scr["nights"]:
        for r in n["rows"]:
            assert r["NULL"] == r["A"]
            assert 0.0 <= r["SHAM"]["right"] <= 1.0
    # resume: the same inputs and commit are not recomputed
    SS.screen_seed, real = (lambda *a, **k: pytest.fail("recomputed")), SS.screen_seed
    try:
        assert SS.run_screen(d, 7, "HEADROOM")["input_sha"] == scr["input_sha"]
    finally:
        SS.screen_seed = real
    # a banked fetch that the snapshot cannot reproduce voids the night
    bad = copy.deepcopy(res)
    for p in bad["probes"]:
        if p["arm"] == "intact-frozen" and p["night"] == 2:
            p["rows"][-1]["fetch"] = {"idx": 999}
    d2 = str(tmp_path / "scr2")
    SS.run_sims(bad, snaps, StubCortex(), d2, weekend=False, nights=[2], log=lambda *a: None)
    u = json.load(open(os.path.join(d2, "sims_s7_n2.json")))
    assert u["void"] and u["void_rows"] == [len(u["rows"]) - 1]


def test_rs_rebuild_equals_a_life_with_the_switch_on(tmp_path, monkeypatch):
    """No fetch boosts in either life (FREEZE_USE): the offline RS store from the OFF life's ledger is the store the Brain
    switch builds night by night."""
    monkeypatch.setattr(Hippocampus, "FREEZE_USE", True)
    res_off, snaps_off = _life(tmp_path, "off")
    _, snaps_on = _life(tmp_path, "on", extra={"replay_strengthen": SS.ARM_CFG["RS"]["boost"],
                                               "strength_setpoint": SS.ARM_CFG["RS"]["setpoint"]})
    d = _sims(tmp_path, res_off, snaps_off, range(1, 5))
    units = [json.load(open(os.path.join(d, f"sims_s7_n{n}.json"))) for n in range(1, 5)]
    ledger = json.load(open(os.path.join(d, "ledger_s7.json")))
    hist = SS.history(ledger, units[0]["hc"]["USE_BOOST"], False)
    assert hist["events"] == [] and sum(hist["slept"].values()) >= 3
    stores = SS.simulate(hist, SS.ARM_CFG["RS"], units[0]["hc"], units[0]["sleep"], {u["night"]: u["t_battery"] for u in units})
    on = SS.ledger_of(snaps_on)
    changed = 0
    for n in range(1, 5):
        rows = on[str(n)]["rows"]
        assert len(rows) == len(stores[n])
        for e, r in zip(stores[n], rows):
            assert e.strength == pytest.approx(r["strength"], rel=1e-6) and abs(e.last_used - r["last_used"]) < 1.0
        off_rows = ledger["snapshots"][str(n)]["rows"]
        changed += sum(abs(a["strength"] - b["strength"]) > 1e-6 for a, b in zip(rows, off_rows))
    assert changed > 0, "the switch should change some strength (not vacuous)"


# ------------------------------------------------------------------ rebuild guards on synthetic ledgers
def _ledger():
    t1 = SS.day_start(1, False)
    row = lambda f, s, c, lu, u, p=False: {"file": f, "strength": s, "created": c, "last_used": lu, "uses": u, "permanent": p}
    p0 = row("e0.pt", 3.0, t1 + 0.001, t1 + 0.002, 0, True)
    a1 = row("e1.pt", 1.0, t1 + 3600, t1 + 3600.001, 0)
    return {"weekend": False, "snapshots": {
        "1": {"rows": [p0, a1], "meta": {"sleeps": 1}},
        "2": {"rows": [p0, dict(a1, strength=1.5, uses=1, last_used=SS.day_start(2, False) + 60),
                       row("e2.pt", 1.0, SS.day_start(2, False) + 100, SS.day_start(2, False) + 100.001, 0)],
              "meta": {"sleeps": 2}}}}


def test_history_events_and_guards():
    h = SS.history(_ledger(), 0.5, False)
    assert h["events"] == [{"file": "e1.pt", "count": 1, "t": SS.day_start(2, False) + 60, "day": 2, "approx": False}]
    assert h["slept"] == {1: True, 2: True} and h["sleeps_before"] == {1: 0, 2: 1}
    bad = _ledger()
    bad["snapshots"]["2"]["rows"][1]["strength"] = 1.7          # not salience + boosts
    with pytest.raises(SS.ReconstructionError):
        SS.history(bad, 0.5, False)
    bad = _ledger()
    bad["snapshots"]["2"]["rows"][0], bad["snapshots"]["2"]["rows"][1] = bad["snapshots"]["2"]["rows"][1], bad["snapshots"]["2"]["rows"][0]
    with pytest.raises(SS.ReconstructionError):                  # not append-only
        SS.history(bad, 0.5, False)
    bad = _ledger()
    bad["snapshots"]["2"]["meta"]["sleeps"] = 3
    with pytest.raises(SS.ReconstructionError):
        SS.history(bad, 0.5, False)
    bad = _ledger()
    bad["snapshots"]["2"]["rows"][1]["uses"] = -1
    bad["snapshots"]["2"]["rows"][1]["strength"] = 0.5
    with pytest.raises(SS.ReconstructionError):
        SS.history(bad, 0.5, False)


def test_fetch_lesion_drops_the_banked_boost():
    h = SS.history(_ledger(), 0.5, False)
    hc = {k: getattr(Hippocampus, k) for k in ("FAMILIARITY", "FAMILIARITY_SPONTANEOUS", "MARGIN", "TAU_DAYS", "DECAY", "USE_BOOST")}
    sl = {"steps": 10, "sleep_seed": 1, "n_generic": 3, "sleep_extra": 0}
    b = {2: SS.day_start(3, False)}
    null = SS.simulate(h, SS.ARM_CFG["NULL"], hc, sl, b)[2]
    les = SS.simulate(h, {"fetch": False, "boost": None, "setpoint": None, "reset": True}, hc, sl, b)[2]
    assert [e.strength for e in null] == [3.0, 1.5, 1.0] and [e.strength for e in les] == [3.0, 1.0, 1.0]
    assert les[1].last_used < null[1].last_used


# ------------------------------------------------------------------ seed_counts on a synthetic screen record
def test_seed_counts_ov_sham_and_void(tmp_path):
    rows = [{"i": 0, "cls": "told", "right_set": [1], "A": None, "NULL": None, "RS": 1, "SCO": None, "FBL": 2,
             "SHAM": {"right": 0.25, "fire": 1.0, "wrong": 0.75, "new_wrong_on_A_right": 0.0}},
            {"i": 1, "cls": "told", "right_set": [2], "A": 2, "NULL": 2, "RS": 1, "SCO": 2, "FBL": 2,
             "SHAM": {"right": 0.5, "fire": 1.0, "wrong": 0.5, "new_wrong_on_A_right": 0.5}},
            {"i": 2, "cls": "fire", "right_set": None, "A": 3, "NULL": 3, "RS": None, "SCO": 3, "FBL": None,
             "SHAM": {"right": 0.0, "fire": 0.5, "wrong": 0.5, "new_wrong_on_A_right": 0.0}}]
    scr = {"seed": 5, "arms": ["NULL", "RS", "SCO", "FBL"], "null_mismatch": 0, "state_mismatch": 0, "approx_events": 0,
           "nights": [{"night": 5, "void": False, "rows": rows}, {"night": 6, "void": True, "rows": rows}]}
    cd = tmp_path / "ceil"
    cd.mkdir()
    json.dump({"rows": [{"cls": "told", "ok": True}, {"cls": "told", "ok": True}, {"cls": "fire", "ok": True}]},
              open(cd / "u_s5_n5_ORACLE.json", "w"))
    m = SS.seed_counts(scr, str(cd))
    a = m["arms"]
    assert m["void_nights"] == 1 and m["missing_oracle_nights"] == []
    assert {k: a["A"][k] for k in ("told_n", "right", "wrong", "none", "fire_n", "fires", "new_wrong_on_A_right", "ov")} == \
        {"told_n": 2, "right": 1, "wrong": 0, "none": 1, "fire_n": 1, "fires": 1, "new_wrong_on_A_right": 0, "ov": 1}
    assert a["RS"]["right"] == 1 and a["RS"]["wrong"] == 1 and a["RS"]["new_wrong_on_A_right"] == 1 and a["RS"]["fires"] == 0
    assert a["RS"]["ov"] == 1 and a["FBL"]["right"] == 1 and a["FBL"]["ov"] == 1
    assert a["SHAM"]["right"] == pytest.approx(0.75) and a["SHAM"]["ov"] == pytest.approx(0.75)
    assert a["SHAM"]["fires"] == pytest.approx(0.5) and a["SHAM"]["new_wrong_on_A_right"] == pytest.approx(0.5)
    os.remove(cd / "u_s5_n5_ORACLE.json")
    assert SS.seed_counts(scr, str(cd))["missing_oracle_nights"] == [5]


# ------------------------------------------------------------------ exact verdicts
def _arm(right, ov=0, fires=0, nw=0, nonH=None, clusters=None):
    clusters = clusters or {"h": right - 8, "n1": 4, "n2": 4}
    return {"told_n": 70, "right": right, "wrong": 0, "none": 0, "fire_n": 120, "fires": fires, "new_wrong_on_A_right": nw,
            "ov": ov, "right_nonH": 8 if nonH is None else nonH, "cluster_right": clusters, "new_none_on_A_right": 0}


def _eng(**over):
    """Base engagement: every gate holds."""
    e = {"RS_dec_vs_null": 10, "SCO_dec_vs_null": 3, "FBL_dec_vs_RS": 2, "SHAM_dec_vs_null": 4.0, "fbl_store_vs_rs": 5,
         "sham_floored": 0, "sham_slots": 100, "tanh_right": {"NULL": [0.3, 0.36, 0.4], "RS": [0.5, 0.6, 0.7]},
         "night_records": {"RS": {"nights": 8, "clamped": 0, "g_off1": 8}, "SCO": {"nights": 8, "clamped": 0, "g_off1": 8}}}
    e.update(over)
    return e


def _meas(**over):
    """Base: passes every clause. Pooled A right 42, RS 66 (need 53); non-H A 24, RS 42 (+18 >= 6); clusters 9 wins, 0 losses;
    ov A 30, RS 48 (need 38); fires A 240, RS 90 (max 120); SCO 45 (RS - SCO 21 >= 5); SHAM 42 (24 >= 10); FBL 60 (18 >= 12);
    every engagement gate holds (RS changes 30 decisions pooled, never clamped)."""
    out = {}
    for j, s in enumerate(SEEDS):
        arms = {"A": _arm(14, 10, 80), "RS": _arm(22, 16, 30, nonH=14, clusters={"h": 8, "n1": 7, "n2": 7}),
                "SCO": _arm(15), "SHAM": _arm(14), "FBL": _arm(20), "NULL": _arm(14, 10, 80)}
        eng = _eng()
        for k, v in over.items():
            arm, field = k.split("__")
            val = v[j] if isinstance(v, (list, tuple)) else v
            if arm == "ENG":
                eng[field] = val
            else:
                arms[arm][field] = val
        out[s] = {"arms": arms, "void_nights": 0, "missing_oracle_nights": [], "null_mismatch": 0, "state_mismatch": 0,
                  "approx_events": 0, "engagement": eng}
    return out


def test_verdict_base_passes():
    v = SS.verdict(_meas(), "HEADROOM")
    assert v["status"].startswith("PASS") and v["kills"] == [] and v["control_fails"] == []
    assert v["checks"]["need"] == 53 and v["checks"]["ov_need"] == 38 and v["checks"]["fire_max"] == 120


@pytest.mark.parametrize("over,clause", [
    ({"RS__right": [18, 17, 17], "RS__cluster_right": {"h": 4, "n1": 7, "n2": 7}},
     "P1"),                                                    # 52 < 53; C1 7, C2 10, C3 18 >= 5 all hold
    ({"RS__right_nonH": 8, "RS__cluster_right": {"h": 14, "n1": 4, "n2": 4}}, "P1b"),   # the whole gain from one H item
    ({"RS__right_nonH": 17, "RS__cluster_right": {"h": 5, "n1": 14, "n2": 3}}, "P1c"),  # 3 wins, 6 losses
    ({"RS__right": [30, 24, 12]}, "P2"),                       # pooled 66, seed 3 falls by 2
    ({"RS__ov": [13, 12, 12]}, "P3"),                          # 37 < 38
    ({"RS__fires": [41, 40, 40]}, "P4"),                       # 121 > 120
    ({"RS__new_wrong_on_A_right": [2, 0, 0]}, "P5"),
])
def test_each_pass_clause_fails_alone(over, clause):
    v = SS.verdict(_meas(**over), "HEADROOM")
    assert v["status"].startswith("KILLED") and [k.split()[0] for k in v["kills"]] == [clause] and v["control_fails"] == []
    assert clause in v["status"]


@pytest.mark.parametrize("over,clause", [
    ({"SCO__right": [21, 21, 20]}, "C1"),                      # RS - SCO = 4
    ({"SHAM__right": [19, 19, 19]}, "C2"),                     # RS - SHAM = 9
    ({"FBL__right": [18, 18, 17]}, "C3"),                      # 53 - 42 = 11 < 12
])
def test_each_control_clause_fails_alone(over, clause):
    v = SS.verdict(_meas(**over), "HEADROOM")
    assert v["status"].startswith("NOT-ATTRIBUTED") and v["kills"] == [] and [c[:2] for c in v["control_fails"]] == [clause]


@pytest.mark.parametrize("status", [None, "NO-HEADROOM", "UNDEFINED", "INCONCLUSIVE: x"])
def test_gated_unless_the_ceiling_says_headroom(status):
    v = SS.verdict(_meas(), status)
    assert v["status"].startswith("GATED") and "checks" in v and v["checks"] == {}


def test_inconclusive_cases():
    m = _meas()
    del m[321]
    assert SS.verdict(m, "HEADROOM")["status"].startswith("INCONCLUSIVE")
    m = _meas()
    m[320]["void_nights"] = 3
    assert SS.verdict(m, "HEADROOM")["status"].startswith("INCONCLUSIVE")
    m = _meas()
    m[320]["null_mismatch"] = 1
    assert "tripwire" in SS.verdict(m, "HEADROOM")["status"]
    m = _meas()
    m[319]["missing_oracle_nights"] = [7]
    assert "ORACLE" in SS.verdict(m, "HEADROOM")["status"]
    for arm in SS.CONTROLS:
        m = _meas()
        del m[319]["arms"][arm]
        assert SS.verdict(m, "HEADROOM")["status"].startswith("INCONCLUSIVE"), arm


def test_score_reads_files_and_the_ceiling_status(tmp_path):
    d = tmp_path / "o"
    d.mkdir()
    rows = [{"i": 0, "cls": "told", "right_set": [1], "A": 1, **{a: 1 for a in SS.ARM_CFG},
             "SHAM": {"right": 1.0, "fire": 1.0, "wrong": 0.0, "new_wrong_on_A_right": 0.0}}]
    for s in SEEDS:
        json.dump({"seed": s, "arms": list(SS.ARM_CFG), "null_mismatch": 0, "state_mismatch": 0, "approx_events": 0,
                   "commit": "c1", "sims_commits": ["c1"], "night_records": {},
                   "nights": [{"night": 5, "void": False, "rows": rows}]},
                  open(d / f"scr_s{s}.json", "w"))
        json.dump({"rows": [{"cls": "told", "ok": True}]}, open(d / f"u_s{s}_n5_ORACLE.json", "w"))
    assert SS.score(str(d), SEEDS, str(d), None)["status"].startswith("GATED")
    json.dump({"status": "HEADROOM"}, open(d / "ceil.json", "w"))
    v = SS.score(str(d), SEEDS, str(d), SS.ceiling_status_of(str(d / "ceil.json")))
    assert v["status"].startswith("UNDEFINED") and "G1" in v["status"] and v["checks"]["RS_right"] == 3   # RS = A: inert
    j = json.load(open(d / "scr_s320.json"))
    j["commit"] = "c2"
    json.dump(j, open(d / "scr_s320.json", "w"))
    assert "mixed commits" in SS.score(str(d), SEEDS, str(d), "HEADROOM")["status"]


# ------------------------------------------------------------------ review fixes (11-27): increment sham, gate, seeds, dirs
def _ep(s, lu=0.0, perm=False):
    from brain.hippocampus import Episode
    return Episode(text="", delta=None, key=None, strength=s, created=0.0, last_used=lu, permanent=perm)


def test_sham_permutes_only_the_increment():
    """RS = NULL (no increment): the increment sham IS the NULL store, so RS - SHAM = 0 and C2 must fail; the old
    whole-vector shuffle would have scrambled the store's own structure and passed C2 for free."""
    import random
    hc, now = Hippocampus(), 10 * 86400.0
    null = [_ep(3.0, now), _ep(0.2, now), _ep(1.1, now - 86400), _ep(5.0, perm=True)]
    eff = [hc.effective_strength(e, now) for e in null[:3]]
    same = [_ep(e.strength, e.last_used, e.permanent) for e in null]
    for k in range(5):
        out = SS._sham_store(same, null, now, hc, random.Random(k))
        assert [e.strength for e in out[:3]] == pytest.approx(eff) and out[3] is same[3]
    # with a real increment the sham keeps NULL's base and moves only the add-on: multiset of (sham - base) = increments
    rs = [_ep(e + d, now) for e, d in zip(eff, (0.5, 0.0, 2.0))] + [same[3]]
    cnt = {}
    out = SS._sham_store(rs, null, now, hc, random.Random(3), cnt)
    assert sorted(round(o.strength - b, 9) for o, b in zip(out[:3], eff)) == [0.0, 0.5, 2.0] and cnt["floored"] == 0
    # a negative increment landing on a small base floors at 0 and is counted
    rs = [_ep(eff[0] - 2.5, now), _ep(eff[1], now), _ep(eff[2], now), same[3]]
    cnt = {}
    for k in range(10):
        out = SS._sham_store(rs, null, now, hc, random.Random(k), cnt)
        assert min(e.strength for e in out[:3]) >= 0
    assert cnt["floored"] > 0
    with pytest.raises(SS.ReconstructionError):
        SS._sham_store(rs, null[:3], now, hc, random.Random(0))


def test_gated_score_returns_the_status_alone(tmp_path):
    d = tmp_path / "o"
    d.mkdir()
    json.dump({"seed": 319, "arms": ["RS"], "nights": []}, open(d / "scr_s319.json", "w"))
    for st in (None, "NO-HEADROOM", "INCONCLUSIVE: x"):
        v = SS.score(str(d), SEEDS, [str(d)], st)
        assert v["status"].startswith("GATED") and set(v) == {"status", "ceiling_status", "checks"} and v["checks"] == {}
        assert "per_seed" not in v and "report_only" not in v


def test_unregistered_seed_sets_are_inconclusive():
    for seeds in ((319, 320), (319, 320, 321, 322), (42, 43, 44)):
        assert SS.verdict(_meas(), "HEADROOM", seeds)["status"].startswith("INCONCLUSIVE"), seeds
    m = _meas()
    m[322] = copy.deepcopy(m[319])
    assert "unregistered" in SS.verdict(m, "HEADROOM")["status"]


def test_score_reads_one_ceiling_dir_per_seed(tmp_path, capsys):
    d = tmp_path / "o"
    d.mkdir()
    rows = [{"i": 0, "cls": "told", "right_set": [1], "A": 1, **{a: 1 for a in SS.ARM_CFG},
             "SHAM": {"right": 1.0, "fire": 1.0, "wrong": 0.0, "new_wrong_on_A_right": 0.0}}]
    dirs = []
    for s in SEEDS:
        json.dump({"seed": s, "arms": list(SS.ARM_CFG), "null_mismatch": 0, "state_mismatch": 0, "approx_events": 0,
                   "commit": "c1", "sims_commits": ["c1"], "night_records": {},
                   "nights": [{"night": 5, "void": False, "rows": rows}]},
                  open(d / f"scr_s{s}.json", "w"))
        cd = tmp_path / f"mc_s{s}"
        cd.mkdir()
        json.dump({"rows": [{"cls": "told", "ok": True}]}, open(cd / f"u_s{s}_n5_ORACLE.json", "w"))
        dirs.append(str(cd))
    json.dump({"status": "HEADROOM"}, open(tmp_path / "ceil.json", "w"))
    v = SS.main(["score", "--dir", str(d), "--seeds", "319", "320", "321", "--ceiling-dir", *dirs,
                 "--ceiling-verdict", str(tmp_path / "ceil.json")])
    assert v["status"].startswith("UNDEFINED") and v["checks"]["A_ov"] == 3
    json.dump({"rows": [{"cls": "told", "ok": False}]}, open(tmp_path / "mc_s320" / "u_s319_n5_ORACLE.json", "w"))
    with pytest.raises(SystemExit):
        SS.score(str(d), SEEDS, dirs, "HEADROOM")


def test_a_ledger_of_another_commit_stops_sims(tmp_path, monkeypatch):
    out = tmp_path / "scr"
    out.mkdir()
    json.dump({"seed": 7, "weekend": False, "snapshots": {}, "commit": "other"}, open(out / "ledger_s7.json", "w"))
    monkeypatch.setattr(SS, "provenance", lambda dev: {"commit": "mine", "host": "h", "device": dev})
    with pytest.raises(SystemExit, match="another commit"):
        SS.run_sims({"seed": 7}, str(tmp_path / "snaps"), StubCortex(), str(out), weekend=False, nights=[5],
                    log=lambda *a: None)


# ------------------------------------------------------------------ review 11-27 (second): engagement gates, P1b/P1c, no-peek
@pytest.mark.parametrize("over,gate", [
    ({"ENG__RS_dec_vs_null": 3}, "G1"),                                   # 9 pooled < 10: RS barely moved a decision
    ({"ENG__night_records": {"RS": {"nights": 8, "clamped": 3, "g_off1": 8},
                             "SCO": {"nights": 8, "clamped": 0, "g_off1": 8}}}, "G1"),     # 9/24 clamped > 1/4
    ({"ENG__tanh_right": {"NULL": [0.4, 0.5, 0.6], "RS": [0.4, 0.5, 0.6]}}, "G4"),    # the blamed quantity did not move
    ({"ENG__night_records": {"RS": {"nights": 8, "clamped": 0, "g_off1": 8},
                             "SCO": {"nights": 8, "clamped": 0, "g_off1": 3}}}, "G2"),     # SCO's g ~ 1 on most nights
    ({"ENG__SCO_dec_vs_null": 0}, "G2"),
    ({"ENG__fbl_store_vs_rs": 0}, "G3"),                                  # the lesion removed nothing
    ({"ENG__SHAM_dec_vs_null": 0.0}, "G3"),
    ({"ENG__sham_floored": 6}, "G3"),                                     # 18/300 > 5%
])
def test_an_inert_arm_is_undefined(over, gate):
    v = SS.verdict(_meas(**over), "HEADROOM")
    assert v["status"].startswith("UNDEFINED") and gate in v["status"]
    assert [g for g, x in v["checks"]["gates"].items() if not x["holds"]] == [gate]


def test_inert_method_is_undefined_even_when_a_p_clause_fails():
    v = SS.verdict(_meas(**{"RS__right": [14, 14, 14], "ENG__RS_dec_vs_null": 0}), "HEADROOM")
    assert v["status"].startswith("UNDEFINED") and "G1" in v["status"] and v["kills"]     # never KILLED by an inert arm


def test_an_inert_control_does_not_block_a_p_clause_kill():
    """P clauses read no control arm, so a P kill with an engaged method stays KILLED; an inert control blocks only the
    verdicts that read it (PASS, NOT-ATTRIBUTED)."""
    v = SS.verdict(_meas(**{"RS__fires": [41, 40, 40], "ENG__SCO_dec_vs_null": 0}), "HEADROOM")
    assert v["status"].startswith("KILLED") and "P4" in v["status"] and v["gate_fails"]
    v = SS.verdict(_meas(**{"SCO__right": [21, 21, 20], "ENG__fbl_store_vs_rs": 0}), "HEADROOM")
    assert v["status"].startswith("UNDEFINED") and "G3" in v["status"]


def test_missing_engagement_is_inconclusive():
    m = _meas()
    m[320]["engagement"] = None
    assert "engagement record missing" in SS.verdict(m, "HEADROOM")["status"]


def _scr_one(rows, nights=None):
    return {"seed": 5, "arms": ["NULL", "RS", "SCO", "FBL"], "null_mismatch": 0, "state_mismatch": 0, "approx_events": 0,
            "night_records": {"RS": [{"g": 1.5, "clamped": False, "counted": 2, "changed": 3}]},
            "nights": nights or [{"night": 5, "void": False, "rows": rows,
                                  "engagement": {"sham_floored": 1, "sham_slots": 40, "fbl_store_vs_rs": 2,
                                                 "tanh_right": {"NULL": [0.3], "RS": [0.6]}}}]}


def test_seed_counts_clusters_nonH_identity_and_engagement(tmp_path):
    sh = {"right": 0.0, "fire": 0.0, "wrong": 0.0, "new_wrong_on_A_right": 0.0, "none_on_A_right": 1.0, "diff_null": 0.5}
    rows = [{"i": 0, "cls": "told", "kind": "H", "item": "pixel", "cue": "cat?", "right_set": [1], "A": 1, "NULL": 1,
             "RS": None, "SCO": 1, "FBL": 1, "SHAM": sh},
            {"i": 1, "cls": "told", "kind": "rot", "item": "volvo", "cue": "car?", "right_set": [2], "A": None, "NULL": None,
             "RS": 2, "SCO": None, "FBL": 2, "SHAM": sh},
            {"i": 2, "cls": "identity", "kind": "H", "item": "name", "cue": "who?", "right_set": None, "A": 0, "NULL": 0,
             "RS": 0, "SCO": 0, "FBL": 0, "SHAM": sh}]
    cd = tmp_path / "c"
    cd.mkdir()
    json.dump({"rows": [{"cls": "told", "ok": True, "item": "pixel", "you": "cat? "},
                        {"cls": "told", "ok": True, "item": "volvo", "you": "car?"}, {"cls": "identity"}]},
              open(cd / "u_s5_n5_ORACLE.json", "w"))
    m = SS.seed_counts(_scr_one(rows), str(cd))
    rs = m["arms"]["RS"]
    assert rs["cluster_right"] == {"pixel": 0.0, "volvo": 1.0} and rs["right_nonH"] == 1 and rs["nonH_n"] == 1
    assert rs["new_none_on_A_right"] == 1 and rs["identity_n"] == 1 and rs["identity_fetches"] == 1
    e = m["engagement"]
    assert e["RS_dec_vs_null"] == 2 and e["SCO_dec_vs_null"] == 0 and e["FBL_dec_vs_RS"] == 1
    assert e["SHAM_dec_vs_null"] == pytest.approx(1.5) and e["sham_floored"] == 1 and e["sham_slots"] == 40
    assert e["night_records"]["RS"]["g_off1"] == 1 and e["tanh_right"]["RS"] == [0.6]
    # a ceiling unit whose row i is another battery row stops the score (NB5)
    json.dump({"rows": [{"cls": "told", "ok": True, "item": "volvo", "you": "car?"},
                        {"cls": "told", "ok": True, "item": "pixel", "you": "cat?"}]}, open(cd / "u_s5_n5_ORACLE.json", "w"))
    with pytest.raises(SystemExit, match="item/cue"):
        SS.seed_counts(_scr_one(rows), str(cd))
    json.dump({"rows": [{"cls": "told", "ok": True, "item": "pixel", "you": "cat?"}]}, open(cd / "u_s5_n5_ORACLE.json", "w"))
    with pytest.raises(SystemExit, match="told rows"):
        SS.seed_counts(_scr_one(rows), str(cd))
    with pytest.raises(ValueError):                                          # NB6: an empty ceiling dir is an error
        SS.seed_counts(_scr_one(rows), [])
    with pytest.raises(ValueError):
        SS.score(str(tmp_path), SEEDS, [], "HEADROOM")


def test_full_screen_cli_refuses_without_headroom(tmp_path):
    d = tmp_path / "o"
    d.mkdir()
    with pytest.raises(SystemExit, match="GATED"):
        SS.main(["screen", "--dir", str(d), "--seed", "319"])
    json.dump({"status": "NO-HEADROOM"}, open(tmp_path / "v.json", "w"))
    with pytest.raises(SystemExit, match="GATED"):
        SS.main(["screen", "--dir", str(d), "--seed", "319", "--ceiling-verdict", str(tmp_path / "v.json")])


def test_homeostat_record_flags_the_clamp():
    from brain import replay_strength as RSM
    hc = Hippocampus()
    eps = [_ep(0.01, 0.0), _ep(0.02, 0.0)]
    r = RSM.homeostatic_scale(eps, hc.effective_strength, 1.0, 0.0)
    assert r["clamped"] and r["g"] == RSM.SCALE_MAX and r["g_raw"] > RSM.SCALE_MAX
    eps = [_ep(1.0, 0.0), _ep(2.0, 0.0)]
    m = sum(hc.effective_strength(e, 0.0) for e in eps) / 2
    r = RSM.homeostatic_scale(eps, hc.effective_strength, 1.5 * m, 0.0)
    assert not r["clamped"] and r["g"] == pytest.approx(r["g_raw"])
