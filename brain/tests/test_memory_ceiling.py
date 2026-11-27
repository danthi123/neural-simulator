"""Exact-verdict and plumbing tests for brain/experiments/memory_ceiling.py on synthetic records (no model, no snapshots)."""
import copy
import json

import pytest

from brain.experiments import memory_ceiling as MC

own2 = lambda t: t.startswith("my")           # voice scorer stub
SEEDS = (319, 320, 321)


def units(rows_by_seed):
    """rows_by_seed: {seed: [(item, kind, ok, brain)]} -> one unit per seed (night 5)."""
    return [{"seed": s, "night": 5, "arm": "x", "n_rows": len(r), "reproduces_banked": 0,
             "rows": [{"cls": "told", "kind": k, "item": it, "ok": ok, "brain": b, "you": "q"} for it, k, ok, b in r]}
            for s, r in rows_by_seed.items()]


def keys_for(spec):
    return {it: (it,) for r in spec.values() for (it, *_rest) in r}


def meas_of(spec_by_arm):
    allspec = {}
    for sp in spec_by_arm.values():
        for s, r in sp.items():
            allspec.setdefault(s, []).extend(r)
    lk = keys_for(allspec)
    return {a: MC.arm_measures(units(sp), lk, own2=own2) for a, sp in spec_by_arm.items()}


def base_spec(o_ok=lambda s, j: 1, off_ok=lambda s, j: 0, sham_ok=lambda s, j: 0, o_brain=lambda s, j, it: it,
              off_brain=lambda s, j, it: it):
    """8 rows per seed: items i0..i7, kinds H x4 then rotating x4."""
    def mk(ok, brain):
        return {s: [(f"i{j}", "H" if j < 4 else "rotating", ok(s, j), brain(s, j, f"i{j}")) for j in range(8)]
                for s in SEEDS}
    a = mk(lambda s, j: 0, lambda s, j, it: it)
    return {"OFF": mk(off_ok, off_brain), "A": a, "ORACLE": mk(o_ok, o_brain), "SHAM": mk(sham_ok, lambda s, j, it: it)}


def st(spec):
    return MC.verdict(meas_of(spec))


def test_headroom_all_clauses_hold():
    v = st(base_spec())
    assert v["status"] == "HEADROOM", v
    assert v["checks"]["oracle_minus_off"] == 24 and all(v["checks"]["headroom_clauses"].values())


def test_h1_alone_fails_undefined():
    # ORACLE 24, OFF 12 (rows j<4 and a spread): O-OFF 12 in [8, 15), O-SHAM 24, rest hold
    v = st(base_spec(off_ok=lambda s, j: 1 if j < 4 else 0))
    c = v["checks"]["headroom_clauses"]
    assert v["status"] == "UNDEFINED" and [k for k, x in c.items() if not x] == ["H1 oracle-off >= 15"]
    assert v["checks"]["oracle_minus_off"] == 12


def test_h2_alone_fails_undefined():
    # ORACLE ok: all H rows + one nonH row per seed -> pooled 15, nonH 3
    v = st(base_spec(o_ok=lambda s, j: 1 if (j < 4 or j == 4) else 0))
    c = v["checks"]["headroom_clauses"]
    assert v["checks"]["oracle_minus_off"] == 15 and v["checks"]["oracle_minus_off_nonH"] == 3
    assert v["status"] == "UNDEFINED" and [k for k, x in c.items() if not x] == ["H2 non-H >= 6"]


def test_h3_alone_fails_undefined():
    v = st(base_spec(o_ok=lambda s, j: 0 if s == 321 else 1))
    c = v["checks"]["headroom_clauses"]
    assert v["checks"]["oracle_minus_off"] == 16
    assert v["status"] == "UNDEFINED" and [k for k, x in c.items() if not x] == ["H3 positive on every seed"]


def test_h4_alone_fails_undefined():
    big = {s: [("big", "rotating", 1, "big")] * 20 + [(f"o{j}", "H", 0, "x") for j in range(4)] for s in SEEDS}
    off = {s: [("big", "rotating", 0, "big")] * 20 + [(f"o{j}", "H", 1, "x") for j in range(4)] for s in SEEDS}
    zero = {s: [("big", "rotating", 0, "big")] * 20 + [(f"o{j}", "H", 0, "x") for j in range(4)] for s in SEEDS}
    v = st({"OFF": off, "A": zero, "ORACLE": big, "SHAM": zero})
    c = v["checks"]["headroom_clauses"]
    assert v["checks"]["cluster"]["wins"] == 3 and v["checks"]["cluster"]["losses"] == 12
    assert v["status"] == "UNDEFINED" and [k for k, x in c.items() if not x] == ["H4 cluster wins > losses"]


def test_h5_alone_fails_undefined():
    v = st(base_spec(sham_ok=lambda s, j: 1 if j < 4 else 0))   # SHAM 12: O-SHAM 12 in [8, 15)
    c = v["checks"]["headroom_clauses"]
    assert v["checks"]["oracle_minus_sham"] == 12
    assert v["status"] == "UNDEFINED" and [k for k, x in c.items() if not x] == ["H5 oracle-sham >= 15"]


def test_h6_alone_fails_undefined():
    # ORACLE names the fact in the user's voice on 6 rows, OFF never: gap 6 > 5
    ob = lambda s, j, it: f"my {it}" if (s == 319 and j < 6) else it
    v = st(base_spec(o_brain=ob))
    c = v["checks"]["headroom_clauses"]
    assert v["checks"]["uv_knows_oracle_minus_off"] == 6
    assert v["status"] == "UNDEFINED" and [k for k, x in c.items() if not x] == ["H6 user-voice not above OFF by more than 5"]
    # exactly 5 is allowed
    ob5 = lambda s, j, it: f"my {it}" if (s == 319 and j < 5) else it
    assert st(base_spec(o_brain=ob5))["status"] == "HEADROOM"


def test_boundaries_exact():
    # O-OFF exactly 15 with nonH 6 passes; 14 does not
    def o(n_extra):   # 12 H rows + n_extra nonH rows in total
        seen = {"n": 0}
        def f(s, j):
            if j < 4:
                return 1
            if seen["n"] < n_extra and j == 4:
                seen["n"] += 1
                return 1
            return 0
        return f
    v = st(base_spec(o_ok=o(3)))
    assert v["checks"]["oracle_minus_off"] == 15
    v = st(base_spec(o_ok=o(2)))
    assert v["checks"]["oracle_minus_off"] == 14 and v["status"] == "UNDEFINED"


def test_no_headroom_either_clause_alone():
    # O-OFF = 7 (OFF 0), O-SHAM = 7 too -> NO-HEADROOM
    v = st(base_spec(o_ok=lambda s, j: 1 if (s == 319 and j < 7) else 0))
    assert v["status"] == "NO-HEADROOM" and v["checks"]["oracle_minus_off"] == 7
    # O-OFF large but SHAM nearly as good (O-SHAM 7): NO-HEADROOM from the sham clause alone
    v = st(base_spec(sham_ok=lambda s, j: 1 if j < 6 and s != 321 else (1 if s == 321 and j < 5 else 0)))
    assert v["checks"]["oracle_minus_off"] == 24 and v["checks"]["oracle_minus_sham"] == 24 - 17 and v["status"] == "NO-HEADROOM"
    # O-OFF exactly 8 is not NO-HEADROOM (UNDEFINED), 7 is
    v8 = st(base_spec(o_ok=lambda s, j: 1 if (s == 319 and j < 8) else 0))
    assert v8["checks"]["oracle_minus_off"] == 8 and v8["status"] == "UNDEFINED"


def test_inconclusive_when_control_missing_or_seeds_short():
    sp = base_spec()
    for arm in MC.CONTROLS:
        part = {k: v for k, v in sp.items() if k != arm}
        v = MC.verdict(meas_of(part))
        assert v["status"].startswith("INCONCLUSIVE") and arm in v["status"]
    two = {a: {s: r for s, r in d.items() if s != 321} for a, d in sp.items()}
    assert MC.verdict(meas_of(two))["status"].startswith("INCONCLUSIVE")
    # an extra arm absent (K-REFIT, rekey) does not block the verdict
    assert st(sp)["status"] == "HEADROOM"


# ---------------------------------------------------------------- Stage 0
def diag_row(night, i, cls, score_best, margin_w, ww, k_idx, k_right, a_idx=None, a_right=False, raw_w=None, raw_right=None,
             question=True):
    r = {"night": night, "i": i, "cls": cls, "item": "x", "kind": "H", "cue": "c", "banked_idx": a_idx,
         "K_raw_winner": ww if raw_w is None else raw_w, "K_sim_raw_max": 0.3,
         "K_diag": {"weighted_winner": ww, "score_best": score_best, "margin_w": margin_w, "w_raw": 0.5,
                    "score_raw": score_best, "question": question},
         "A": {"idx": a_idx, "text": None, "right_episode": a_right},
         "K": {"idx": k_idx, "text": None, "right_episode": k_right}}
    if cls != "fire":
        r["K_raw_right"] = bool(raw_right) if raw_right is not None else bool(k_right)
    return r


def synth_diag():
    rows = [
        diag_row(5, 0, "told", 0.30, 0.10, 4, 4, True, a_idx=9),                    # fires registered, right
        diag_row(5, 1, "told", 0.10, 0.10, 7, None, False, a_idx=9, raw_w=7, raw_right=True),   # K no-fetch, raw right
        diag_row(5, 2, "told", 0.10, 0.10, 8, None, False, a_idx=9, raw_w=2),       # unknown rightness
        diag_row(5, 3, "told", 0.30, 0.01, 6, None, False, a_idx=3, a_right=True),  # margin blocks at any bar
    ] + [diag_row(5, 10 + j, "fire", sc, 0.1, 1, (1 if sc >= 0.165 else None), False, a_idx=1)
         for j, sc in enumerate([0.05, 0.08, 0.12, 0.20])]                          # A fires on all 4
    return {"seed": 319, "nights": [{"night": 5, "rows": rows, "void": False}]}


def test_stage0_bar_rule_and_counts():
    out = MC.stage0([synth_diag()])
    assert out["registered_reproduces_K"] is True
    assert out["A_fires_on_fire_rows"] == 4
    # smallest bar with K fires <= 4: bar 0.0 fires on all four -> 4 <= 4
    assert out["bars"]["A_matched"]["bar"] == 0.0
    # half A = 2: need fires <= 2: bar 0.08 -> scores 0.08, 0.12, 0.20 fire (3); 0.085 -> 2
    assert out["bars"]["half_A"]["bar"] == 0.085
    t = out["arms"]["K-refit@A_matched"]
    assert (t["right"], t["wrong"], t["fired_unknown_right"], t["no_fetch"], t["fires"]) == (2, 0, 1, 1, 4)
    h = out["arms"]["K-refit@half_A"]
    assert h["fires"] == 2
    assert (h["right"], h["wrong"], h["fired_unknown_right"], h["no_fetch"]) == (2, 0, 1, 1)
    assert out["decisions_A_matched"]["319"]["5"]["0"] == 4 and "3" not in out["decisions_A_matched"]["319"]["5"]


def test_stage0_statement_ratio_and_void_nights_skipped():
    d = synth_diag()
    d["nights"].append({"night": 6, "void": True, "rows": [diag_row(6, 0, "told", 0.9, 0.5, 1, 1, True)]})
    assert MC.stage0([d])["arms"]["A"]["told_n"] == 4
    rec = {"question": False, "score_best": 0.40, "margin_w": 0.1}
    assert MC.fires(rec, 0.12) is False and MC.fires(rec, 0.12, stmt=0.45) is False
    assert MC.fires({**rec, "score_best": 0.46}, 0.12) is True       # 0.12 * 0.45/0.12 = 0.45
    assert MC.fires({**rec, "margin_w": 0.019}, 0.0) is False


def test_stage0_missing_fields_raises():
    d = synth_diag()
    del d["nights"][0]["rows"][0]["K_diag"]
    with pytest.raises(ValueError):
        MC.stage0([d])


def test_stage0_on_banked_files_reproduces_registered():
    import os
    d = "brain/experiments/results/2026-11-25-routing-screen2"
    fs = [f"{d}/s2diag_s{s}.json" for s in (319, 320, 321)]
    if not all(os.path.exists(f) for f in fs):
        pytest.skip("banked s2diag files absent")
    out = MC.stage0([json.load(open(f)) for f in fs])
    assert out["registered_reproduces_K"] is True
    assert (out["arms"]["A"]["right"], out["arms"]["A"]["wrong"], out["arms"]["A"]["fires"]) == (44, 136, 239)
    assert (out["arms"]["K@registered"]["right"], out["arms"]["K@registered"]["fires"]) == (49, 67)


# ---------------------------------------------------------------- plumbing
def test_guard(monkeypatch):
    with pytest.raises(SystemExit):
        MC.guard({})
    with pytest.raises(SystemExit):
        MC.guard({"POOL_NODE": "0"})
    MC.guard({"POOL_NODE": "1"})
    MC.guard({"BRAIN_ALLOW_W3090": "1"})


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = str(tmp_path / "a" / "x.json")
    MC.atomic_write(p, {"a": 1})
    assert json.load(open(p)) == {"a": 1} and not (tmp_path / "a" / "x.json.tmp").exists()


def test_pick_sham_matches_tercile_and_age():
    told = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"}
    eff = [0.1, 0.2, 1.0, 1.1, 2.0, 2.1]
    created = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # right = 3 (item d, middle tercile with index 2); sham must be another item, same tercile, nearest age -> 2
    assert MC.pick_sham(3, "d", told, eff, created) == 2
    # never the row's own item even when it is the nearest
    assert MC.pick_sham(0, "a", told, eff, created) == 1
    assert MC.pick_sham(0, "a", {0: "a"}, [1.0], [0.0]) is None


def test_forced_map_per_arm():
    rows = [{"i": 0, "cls": "told", "a_idx": 5, "right_idx": 1, "sham_idx": 2, "k_idx": 3},
            {"i": 1, "cls": "fire", "a_idx": 4, "right_idx": None, "sham_idx": None, "k_idx": None}]
    assert MC.forced_map("ORACLE", rows, {}) == {0: 1} and MC.forced_map("ORACLE-rekey", rows, {}) == {0: 1}
    assert MC.forced_map("SHAM", rows, {}) == {0: 2}
    assert MC.forced_map("K-REFIT", rows, {}) == {0: 3, 1: None}
    # defect 5/7: no stored right episode or sham -> A's fetch kept; identity rows keep A's fetch under K-REFIT
    rows2 = [{"i": 0, "cls": "told", "a_idx": 5, "right_idx": None, "sham_idx": None, "k_idx": None},
             {"i": 1, "cls": "identity", "a_idx": 4, "right_idx": None, "sham_idx": None, "k_idx": None}]
    assert MC.forced_map("ORACLE", rows2, {}) == {} and MC.forced_map("SHAM", rows2, {}) == {}
    assert MC.forced_map("K-REFIT", rows2, {}) == {0: None}
    assert MC.forced_map("A", rows, {}) == {}


def test_run_resume_and_provenance(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(MC, "night_decisions", lambda *a, **k: {"night": a[1], "rows": [], "void": False, "void_rows": []})
    monkeypatch.setattr(MC, "run_arm", lambda res, n, snap, trunk, cx, work, wk, arm, dec: (calls.append((n, arm)) or
                        {"seed": res["seed"], "night": n, "arm": arm, "seconds": 0.0, "rows": []}))
    snaps = tmp_path / "snaps"
    (snaps / "n5").mkdir(parents=True)
    out = str(tmp_path / "out")
    res = {"seed": 319}
    MC.run(res, str(snaps), "T", None, str(tmp_path / "w"), out, True, "", nights=range(5, 6), arms=("OFF", "A"), log=lambda *a: None)
    assert calls == [(5, "OFF"), (5, "A")]
    u = json.load(open(f"{out}/u_s319_n5_OFF.json"))
    assert {"host", "commit", "device"} <= set(u)
    MC.run(res, str(snaps), "T", None, str(tmp_path / "w"), out, True, "", nights=range(5, 6), arms=("OFF", "A", "SHAM"), log=lambda *a: None)
    assert calls[2:] == [(5, "SHAM")]          # finished units are skipped
    assert not [f for f in __import__("os").listdir(out) if f.endswith(".tmp")]


def test_void_night_runs_no_arms(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(MC, "night_decisions", lambda *a, **k: {"night": 5, "rows": [], "void": True, "void_rows": [3]})
    monkeypatch.setattr(MC, "run_arm", lambda *a, **k: calls.append(1))
    (tmp_path / "s" / "n5").mkdir(parents=True)
    MC.run({"seed": 319}, str(tmp_path / "s"), "T", None, str(tmp_path / "w"), str(tmp_path / "o"), True, "",
           nights=range(5, 6), log=lambda *a: None)
    assert calls == []


def test_handread_is_blind_and_deterministic():
    spec = base_spec()
    by = {a: units(sp) for a, sp in spec.items()}
    for a, us in by.items():
        for u in us:
            u["arm"] = a
            for r in u["rows"]:
                r["brain"] = f"{a.lower()}-reply"
    sheet, key = MC.handread(by, n=10)
    assert len(sheet) == 10 and sheet == MC.handread(by, n=10)[0]
    blob = json.dumps(sheet)
    assert all(a not in blob for a in ("OFF", "ORACLE", "SHAM"))
    assert sorted(set(key.values())) == sorted(by)
    for e in sheet:
        for lab, text in e["replies"].items():
            assert text == key[f'{e["id"]}:{lab}'].lower() + "-reply"


def test_sign_p():
    assert MC.sign_p(0, 0) == 1.0 and MC.sign_p(10, 0) == pytest.approx(2 / 1024)


# ---------------------------------------------------------------- pre-data review fixes
def test_pick_sham_never_the_right_episode_even_if_mismapped():
    told = {0: "a", 1: "a", 2: "b"}           # episode 1 mapped to the wrong item; right episode is 1 for row item "b"
    assert MC.pick_sham(1, "b", told, [1.0, 1.0, 1.0], [0.0, 1.0, 2.0]) != 1


def test_verdict_inconclusive_on_mismatched_night_sets():
    spec = base_spec()
    meas = meas_of(spec)
    assert MC.verdict(meas)["status"] == "HEADROOM"
    for a in MC.CONTROLS:
        m = copy.deepcopy(meas)
        m[a]["units"] = m[a]["units"] + [[319, 6]]
        assert MC.verdict(m)["status"].startswith("INCONCLUSIVE"), a
    m = copy.deepcopy(meas)
    m["SHAM"]["told_n"] += 1
    assert MC.verdict(m)["status"].startswith("INCONCLUSIVE")
    # all arms agree with each other but not with the non-void decision files
    assert MC.verdict(meas, expected_units=[(s, n) for s in SEEDS for n in (5, 6)])["status"].startswith("INCONCLUSIVE")
    assert MC.verdict(meas, expected_units=[(s, 5) for s in SEEDS])["status"] == "HEADROOM"


def test_score_inconclusive_on_mixed_commits_and_writes_handread(tmp_path):
    spec = base_spec()
    for a, sp in spec.items():
        for s, rows in sp.items():
            u = units({s: rows})[0]
            u.update(arm=a, commit="c1", host="h", device="cpu", stage0_sha="x")
            for r in u["rows"]:
                r["brain"] = a
            MC.atomic_write(str(tmp_path / f"u_s{s}_n5_{a}.json"), u)
    for s in SEEDS:
        MC.atomic_write(str(tmp_path / f"dec_s{s}_n5.json"), {"seed": s, "night": 5, "void": False, "commit": "c1", "stage0_sha": "x"})
    lk = keys_for(spec["OFF"])
    v = MC.score(str(tmp_path), lk)
    assert v["status"] in ("HEADROOM", "UNDEFINED", "NO-HEADROOM") and (tmp_path / "handread_sheet.json").exists()
    assert (tmp_path / "handread_key.json").exists()
    MC.atomic_write(str(tmp_path / "dec_s319_n5.json"), {"seed": 319, "night": 5, "void": False, "commit": "c2", "stage0_sha": "x"})
    assert MC.score(str(tmp_path), lk, write_handread=False)["status"].startswith("INCONCLUSIVE: mixed")


def test_arm_measures_engagement_and_fire_kinds():
    u = [{"seed": 319, "night": 5, "arm": "ORACLE", "rows": [
        {"cls": "told", "kind": "H", "item": "a", "ok": 1, "brain": "a", "forced": True, "engaged": True, "right_idx": 3},
        {"cls": "told", "kind": "H", "item": "b", "ok": 0, "brain": "x", "forced": False, "engaged": False, "right_idx": None},
        {"cls": "fire", "kind": "never", "fetched": True}, {"cls": "fire", "kind": "greet"}]}]
    m = MC.arm_measures(u, {"a": ("a",), "b": ("b",)}, own2=lambda t: False)
    assert m["engaged"] == {"told_forced": 1, "told_engaged": 1, "told_no_right_episode": 1}
    assert m["fire_by_kind"]["never"]["fetched"] == 1 and set(m["fire_never"]) == {"never"} and m["units"] == [[319, 5]]


def test_knows_is_the_banked_scorer():
    from brain.experiments import voice_split as VS
    ledger = {"i": {"keys": ["Nova", "blue"]}}
    for t in ("My nova is here", "bluebird", "the BLUE one", "nothing"):
        row = {"item": "i", "brain": t}
        assert MC._knows(row, {"i": ["Nova", "blue"]}) == VS.knows(row, ledger)


class _FakeHC:
    def __init__(self):
        self.episodes = [type("E", (), {"text": f"t{i}", "delta": None})() for i in range(3)]
        self.last_fetch = None


def _run_with_fake_brain(monkeypatch, cues, rows_b):
    """Drive run_arm's recall closure through a fake open_snapshot / run_battery."""
    import types
    hc = _FakeHC()
    b = types.SimpleNamespace(hippocampus=hc, cortex=None, base=None, release=lambda: None, set_lesions=lambda **k: None)
    monkeypatch.setattr(MC.RR, "banked_rows", lambda res, n: rows_b)
    monkeypatch.setattr(MC.RR, "open_snapshot", lambda *a, **k: b)
    monkeypatch.setattr(MC.RR, "_qs", lambda rows: rows)
    import brain.eval.life as life
    monkeypatch.setattr(life, "run_battery", lambda bb, qs, led, night: [
        (bb.hippocampus.recall(None, None, c), {"brain": "r", "recalled": hc.episodes[1].text if i == 0 else None})[1]
        for i, c in enumerate(cues)])
    dec = {"rows": [{"i": i, "cls": "told", "a_idx": None, "right_idx": 1, "sham_idx": 2, "k_idx": None, "kind": "H"}
                    for i in range(len(rows_b))]}
    return MC.run_arm({"seed": 1, "ledger": {}, "brain_kw": {}}, 5, "snap", "T", None, "/tmp/mc_fake_work", True, "ORACLE", dec)


def test_recall_closure_checks_cue_and_call_count(monkeypatch):
    rows_b = [{"you": "q0", "brain": "x", "kind": "H"}, {"you": "q1", "brain": "x", "kind": "H"}]
    u = _run_with_fake_brain(monkeypatch, ["q0", "q1"], rows_b)
    assert u["recall_calls"] == 2 and u["rows"][0]["engaged"] is True and u["rows"][1]["engaged"] is False
    assert u["rows"][0]["forced_idx"] == 1
    with pytest.raises(RuntimeError, match="not battery row"):
        _run_with_fake_brain(monkeypatch, ["q0", "WRONG"], rows_b)
    with pytest.raises(RuntimeError, match="recall calls"):
        _run_with_fake_brain(monkeypatch, ["q0"], rows_b)
