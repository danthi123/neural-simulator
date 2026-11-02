"""tools/b2d_score.py: the B2d scorer's S0/S1/S2 levels and A1.2 rule 7 (organ error -> class C), on synthetic
shard trees (no real data, no brain build).

Governing text: research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md ("Three scores",
"Self-describing fields at H", "Scoring code"). The three acceptance checks that section names for the scorer:
  1. identity replay -- for THIS implementation, S0 is a verbatim `lb.json` read, so the only exact check possible
     is that S0 reproduces the file; see `test_s0_reproduces_lb_json_verbatim`. (b2d_score.py's own docstring states
     why a harness-level S1 replay is out of scope for this file.)
  2. synthetic cells: an SD-only diff gives S1 pass; a null-noisy answer gives S2 UNDEFINED:null-noise, never 0; a
     missing .rep0 gives class E; an all-SD row gives UNDEFINED:sd-only; open-ended-generation is never replayed.
  3. a test that fails if the scorer ever imports/calls the harness (`_spawn_arm_raw` / load_bearing_fraction).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import b2d_score as bd  # noqa: E402

PIN = "f" * 40
HOST = "node-x"
PRIMARY = "a" * 64
HOST_SHA = {HOST: PRIMARY}


def _prov(extra_env=None, sha=PIN, host=HOST, **kw):
    env = {"SIM_BACKEND": "numpy"}
    env.update(extra_env or {})
    d = {"git_sha": sha, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
         "source_manifest_verified_at_exit": True, "env": env, "host": host}
    d.update(kw)
    return d


def _turn(answer="the wolf bites the apple", abstained=False, recalled_svo=None, extra=None):
    t = {"answer": answer, "abstained": abstained, "recalled_svo": recalled_svo or ["wolf", "bite", "apple"]}
    t.update(extra or {})
    return {"well": t}


def _cell(tmp, fac="row-a", verdict="regressed", lb=True, null_clean=True, kind="neural-lesion", turn="well",
         diffs=None, ia=None, ib=None, le=None, le0=None, lb_prov=None, arm_prov=None):
    """Build one shard directory with 4 arm files (each a {turn: {...}} dict) + lb.json."""
    d = tmp / fac
    d.mkdir(parents=True, exist_ok=True)
    arm_env = {"BRAIN_CHAT_SEED": "42"}
    bodies = {"intact_a_well.json": ia or _turn(), "intact_b_well.json": ib or _turn(),
             ("lesion_%s.json" % fac.replace("-", "_")): le or _turn(),
             ("lesion_%s.json.rep0" % fac.replace("-", "_")): le0 or (le or _turn())}
    for name, body in bodies.items():
        (d / name).write_text(json.dumps(body))
        (d / (name + ".prov.json")).write_text(json.dumps(arm_prov or _prov(arm_env)))
    if lb:
        rep = {"per_faculty": [{"faculty": fac, "kind": kind, "verdict": verdict, "load_bearing": lb and
                                verdict == "regressed", "turn": turn, "diffs": diffs or [],
                                "null_control_clean": null_clean, "lesion_reproduced": True}]}
        (d / "lb.json").write_text(json.dumps(rep))
        (d / "lb.json.prov.json").write_text(json.dumps(lb_prov or _prov()))
    return str(d)


# ───────────────────────────────────────────── get_path ─────────────────────────────────────────────

def test_get_path_resolves_dotted_and_reports_absence():
    assert bd.get_path({"a": {"b": 1}}, "a.b") == (True, 1)
    assert bd.get_path({"a": {"b": 1}}, "a.c") == (False, None)
    assert bd.get_path({"a": 1}, "a.b") == (False, None)      # "a" is not a dict -> can't descend


# ───────────────────────────────────────────── judge_dir / rule 7 ─────────────────────────────────────────────

def test_good_cell_is_defined(tmp_path):
    r = bd.judge_dir(_cell(tmp_path), "row-a", PIN, HOST_SHA, PRIMARY)
    assert r["class"] is None and r["fails"] == [] and r["turn"] == "well"


def test_rule7_organ_error_on_probe_turn_forces_class_c(tmp_path):
    """THE B2d ADDITION: an 'error' key in an organ dict on the row's OWN probe turn, in any arm, is class C --
    never a silent 'regressed', matching the comprehension-monitor crash this session's fix targets."""
    le = _turn(extra={"comprehension": {"on": True, "error": "ValueError: shape mismatch"}})
    d = _cell(tmp_path, verdict="regressed", le=le, le0=le)
    frow = {"fields": {"comprehension.comprehended": {"class": "SD"}}, "kept": [], "sd_only": True,
           "probe_turn": "well"}
    r = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY, fields_row=frow)
    assert r["class"] == "C"
    assert any("rule7" in x for x in r["fails"])
    # both the lesion arm and its .rep0 carry the same synthetic error body here -> 2 hits, same organ
    assert len(r["rule7_organ_error"]) >= 1
    assert all(h["organ"] == "comprehension" for h in r["rule7_organ_error"])


def test_rule7_ignores_an_organ_error_on_a_DIFFERENT_turn(tmp_path):
    """Rule 7 is scoped to the row's own probe turn -- an error on some other turn in the same arm file must not
    invalidate a cell whose probe turn is clean."""
    le = {"well": _turn()["well"], "other_turn": {"comprehension": {"error": "boom"}}}
    d = _cell(tmp_path, verdict="regressed", le=le, le0=le)
    frow = {"fields": {"comprehension.comprehended": {"class": "SD"}}, "kept": [], "sd_only": True,
           "probe_turn": "well"}
    r = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY, fields_row=frow)
    assert r["class"] is None
    assert r["organ_errors"] and not r["rule7_organ_error"]    # reported, but did not trigger rule 7


def test_rule7_ignores_an_organ_error_outside_the_rows_field_scope(tmp_path):
    """An error in an organ this row's compared fields never touch (out of `field_organs`) does not trigger rule 7
    -- e.g. a comprehension crash should not invalidate an unrelated affect-drives-response cell."""
    le = _turn(extra={"comprehension": {"error": "boom"}})
    d = _cell(tmp_path, verdict="regressed", le=le, le0=le)
    frow = {"fields": {"affect_drives.acted": {"class": "N"}}, "kept": ["affect_drives.acted"],
           "sd_only": False, "probe_turn": "well"}
    r = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY, fields_row=frow)
    assert r["class"] is None
    assert r["organ_errors"] and not r["rule7_organ_error"]


def test_rule7_a_null_error_field_is_not_an_organ_error(tmp_path):
    """REGRESSION (2026-10-06, NEEDS [transitive-spiking-reach]): webapp/server.py's `transitive_trace` records
    `"error": null` on every matched turn. A null error is "no error"; with fields_row=None (scope = every organ,
    the way tools/iw2_score.py and tools/s2p_score.py call judge_dir) it must neither be reported nor fire rule 7."""
    tr = {"route": "spiking_reach", "lesioned": False, "error": None, "reach": {"reached": True}}
    ia = _turn(extra={"transitive_trace": tr})
    le = _turn(extra={"transitive_trace": dict(tr, lesioned=True)})
    d = _cell(tmp_path, verdict="regressed", ia=ia, ib=ia, le=le, le0=le)
    r = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    assert r["organ_errors"] == [] and r["rule7_organ_error"] == []
    assert r["class"] is None and bd.final_state(r, None, None, None) == "DEFINED"


def test_rule7_still_fires_on_a_real_error_in_the_same_trace(tmp_path):
    """The null-error fix must not hide a genuine one: a swallowed exception recorded in the same trace shape, in
    one lesion arm only, still forces class C (scope None and an explicit field scope alike)."""
    tr = {"route": "spiking_reach", "lesioned": True, "error": "RuntimeError: reach build failed"}
    ok = _turn(extra={"transitive_trace": dict(tr, error=None, lesioned=False)})
    bad = _turn(extra={"transitive_trace": tr})
    good_le = _turn(extra={"transitive_trace": dict(tr, error=None)})
    for frow in (None, {"fields": {"transitive_trace.lesioned": {"class": "N"}}, "kept": ["transitive_trace.lesioned"],
                        "sd_only": False, "probe_turn": "well"}):
        sub = tmp_path / ("scoped" if frow else "unscoped")
        d = _cell(sub, verdict="regressed", ia=ok, ib=ok, le=good_le, le0=bad)
        r = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY, fields_row=frow)
        assert r["class"] == "C" and any("rule7" in x for x in r["fails"])
        assert [h["organ"] for h in r["rule7_organ_error"]] == ["transitive_trace"]
        assert "RuntimeError" in r["rule7_organ_error"][0]["error"]


def test_rule7_a_falsy_but_non_null_error_still_counts(tmp_path):
    """Only None means "no error". An empty string or False is ambiguous, so it invalidates (never hidden)."""
    for i, val in enumerate(("", False)):
        le = _turn(extra={"comprehension": {"error": val}})
        d = _cell(tmp_path / ("v%d" % i), verdict="regressed", le=le, le0=le)
        r = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
        assert r["class"] == "C", val


def test_missing_rep0_is_class_e_rule6(tmp_path):
    d = _cell(tmp_path)
    os.remove(os.path.join(d, "lesion_row_a.json.rep0"))
    os.remove(os.path.join(d, "lesion_row_a.json.rep0.prov.json"))
    r = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    assert r["class"] is None                 # judge_dir's own A1.2 rules don't require rep0 to exist
    # S1/S2 must catch the absence themselves (see test_s2_missing_rep0_is_undefined_class_e below)


# ─────────────────────────────────────────────────── S1 ───────────────────────────────────────────────────

def test_s1_sd_field_diff_gives_pass_not_regressed(tmp_path):
    """A synthetic cell whose ONLY diff is in an SD field (not in `kept`) must read S1 pass -- this is the whole
    point of S1."""
    ia = _turn(extra={"x": {"reason": "neutral_hold", "acted": True}})
    le = _turn(extra={"x": {"reason": "lesion_collapsed", "acted": True}})   # only .reason differs; .acted doesn't
    d = _cell(tmp_path, verdict="regressed", diffs=[{"field": "x.reason", "on": "neutral_hold",
                                                      "off": "lesion_collapsed"}], ia=ia, ib=ia, le=le, le0=le)
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    frow = {"fields": {"x.reason": {"class": "SD"}, "x.acted": {"class": "N"}}, "kept": ["x.acted"],
           "sd_only": False, "probe_turn": "well"}
    verdict, detail = bd.score_s1(canon, frow)
    assert verdict == "pass"
    assert detail["field_status"]["x.acted"] == "no-diff"


def test_s1_kept_field_genuine_diff_is_regressed(tmp_path):
    ia = _turn(extra={"x": {"mode": "neutral"}})
    le = _turn(extra={"x": {"mode": "rest"}})
    d = _cell(tmp_path, le=le, le0=le, ia=ia, ib=ia)
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    frow = {"fields": {"x.mode": {"class": "N"}}, "kept": ["x.mode"], "sd_only": False, "probe_turn": "well"}
    verdict, detail = bd.score_s1(canon, frow)
    assert verdict == "regressed" and detail["load_bearing_fields"] == ["x.mode"]


def test_s1_all_sd_row_is_undefined_sd_only_never_zero(tmp_path):
    d = _cell(tmp_path)
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    frow = {"fields": {"x.reason": {"class": "SD"}}, "kept": [], "sd_only": True, "probe_turn": "well"}
    verdict, detail = bd.score_s1(canon, frow)
    assert verdict == "UNDEFINED:sd-only"
    assert detail["kept"] == []


def test_s1_field_absent_in_all_arms_is_not_applicable_and_does_not_block_pass(tmp_path):
    """An UNSURE field the organ never emits in either arm (per the fields-file audit) must not make the WHOLE
    cell unresolved when another kept field cleanly resolves to no-diff."""
    ia = _turn(extra={"x": {"acted": True}})   # "x.on" is never present in any arm
    d = _cell(tmp_path, ia=ia, ib=ia, le=ia, le0=ia)
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    frow = {"fields": {"x.acted": {"class": "N"}, "x.on": {"class": "UNSURE"}}, "kept": ["x.acted", "x.on"],
           "sd_only": False, "probe_turn": "well"}
    verdict, detail = bd.score_s1(canon, frow)
    assert verdict == "pass"
    assert detail["field_status"]["x.on"] == "not-applicable"
    assert detail["field_status"]["x.acted"] == "no-diff"


def test_s1_null_noise_is_undefined_never_zero(tmp_path):
    ia = _turn(extra={"x": {"mode": "neutral"}})
    ib = _turn(extra={"x": {"mode": "rest"}})    # the null itself differs -> noisy, not a real effect
    le = _turn(extra={"x": {"mode": "rest"}})
    d = _cell(tmp_path, ia=ia, ib=ib, le=le, le0=le)
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    frow = {"fields": {"x.mode": {"class": "N"}}, "kept": ["x.mode"], "sd_only": False, "probe_turn": "well"}
    verdict, _ = bd.score_s1(canon, frow)
    assert verdict == "UNDEFINED:s1-noise"


def test_s1_open_ended_generation_is_not_replayed_equals_s0(tmp_path):
    d = _cell(tmp_path, fac="open-ended-generation", verdict="pass")
    canon = bd.judge_dir(d, "open-ended-generation", PIN, HOST_SHA, PRIMARY)
    # score() special-cases this row via fields["not_replayed"]; score_s1 itself is never called for it in score(),
    # but confirm judge_dir's own S0 read is what score() would fall back to.
    assert canon["verdict"] == "pass"


# ─────────────────────────────────────────────────── S2 ───────────────────────────────────────────────────

def test_s2_regressed_on_answer_change(tmp_path):
    ia = _turn(answer="yes")
    le = _turn(answer="no")
    d = _cell(tmp_path, ia=ia, ib=ia, le=le, le0=le)
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    verdict, detail = bd.score_s2(canon, {"probe_turn": "well"})
    assert verdict == "regressed" and detail["answer_only"] is True


def test_s2_null_noise_is_undefined_never_zero(tmp_path):
    ia = _turn(answer="yes")
    ib = _turn(answer="no")     # the two intact arms disagree -> the null itself is noisy
    le = _turn(answer="no")
    d = _cell(tmp_path, ia=ia, ib=ib, le=le, le0=le)
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    verdict, _ = bd.score_s2(canon, {"probe_turn": "well"})
    assert verdict == "UNDEFINED:null-noise"


def test_s2_lesion_noise_is_undefined(tmp_path):
    ia = _turn(answer="yes")
    le = _turn(answer="no")
    le0 = _turn(answer="maybe")     # the lesion does not reproduce against its own repeat
    d = _cell(tmp_path, ia=ia, ib=ia, le=le, le0=le0)
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    verdict, _ = bd.score_s2(canon, {"probe_turn": "well"})
    assert verdict == "UNDEFINED:lesion-noise"


def test_s2_pass_when_reply_unchanged(tmp_path):
    d = _cell(tmp_path)   # every arm's turn body is the default _turn() -- identical everywhere
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    verdict, detail = bd.score_s2(canon, {"probe_turn": "well"})
    assert verdict == "pass" and detail["changed_fields"] == []


def test_s2_missing_rep0_is_undefined_missing_arm(tmp_path):
    d = _cell(tmp_path)
    os.remove(os.path.join(d, "lesion_row_a.json.rep0"))
    os.remove(os.path.join(d, "lesion_row_a.json.rep0.prov.json"))
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    verdict, _ = bd.score_s2(canon, {"probe_turn": "well"})
    assert verdict == "UNDEFINED:missing-arm-or-turn"


# ────────────────────────────────────────────── S0 (identity) ──────────────────────────────────────────────

def test_s0_reproduces_lb_json_verbatim(tmp_path):
    """The only 'identity' check this file's S0 can offer: it must read exactly what `lb.json` itself says, never
    recompute it."""
    d = _cell(tmp_path, verdict="pass")
    canon = bd.judge_dir(d, "row-a", PIN, HOST_SHA, PRIMARY)
    on_disk = json.load(open(os.path.join(d, "lb.json")))["per_faculty"][0]
    assert canon["verdict"] == on_disk["verdict"]
    assert canon["load_bearing"] == on_disk["load_bearing"]
    assert canon["diffs"] == on_disk["diffs"]


# ─────────────────────────────────────────── never touches the harness ───────────────────────────────────────────

def test_never_imports_or_calls_the_harness():
    """Acceptance check #3 from the pre-registration: the scorer must never call `_spawn_arm_raw` (i.e. never build
    a brain), and never import the harness module at all. Checks actual CODE (a call site, and the AST's own
    import statements), not the module's prose docstring -- which legitimately DISCUSSES `_spawn_arm_raw` and
    `research.runners.load_bearing_fraction` by name (as the thing this file deliberately does NOT do) without ever
    calling or importing either."""
    import ast
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "b2d_score.py")
    src = open(path).read()
    assert "_spawn_arm_raw(" not in src                     # no call site anywhere, prose or code
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any("load_bearing_fraction" in (n.name or "") for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "load_bearing_fraction" not in (node.module or "")


# ─────────────────────────────────────────────── read_fields / field_organs ───────────────────────────────────────────────

def test_field_organs_collects_top_level_keys():
    row = {"fields": {"affect_drives.reason": {}, "affect_drives.acted": {}, "affect.valence_sign": {}}}
    assert bd.field_organs(row) == ["affect", "affect_drives"]


def test_field_organs_none_row_is_empty():
    assert bd.field_organs(None) == []


# ─────────────────────────────────────────────────── score() end to end ───────────────────────────────────────────────────

def test_score_end_to_end_on_a_small_synthetic_battery(tmp_path):
    base = tmp_path / "shards"
    tag = "synthtag"
    (base / tag).mkdir(parents=True)
    for seed in (42, 43):
        sdir = base / tag / ("s%d" % seed)
        sdir.mkdir()
        ia = _turn(answer="yes")
        le = _turn(answer="no")
        _cell(sdir, fac="row-a", verdict="regressed", diffs=[{"field": "answer", "on": "yes", "off": "no"}],
              ia=ia, ib=ia, le=le, le0=le)
    fields = {"pin": PIN, "reply_fields": ["answer", "abstained", "recalled_svo"], "not_replayed": {},
             "rows": {"row-a": {"fields": {"answer": {"class": "R"}}, "kept": [], "sd_only": True,
                                "probe_turn": "well"}}}
    fpath = tmp_path / "fields.json"
    fpath.write_text(json.dumps(fields))
    res = bd.score(str(base), tag, PIN, [42, 43], fields_path=str(fpath))
    assert res["s0"]["robust_core_n"] == 1
    assert res["s2"]["robust_core_n"] == 1          # answer differs cleanly on both seeds
    assert res["s1"]["robust_core_n"] == 0          # row-a is sd_only for S1 -> never counted
    assert "row-a" in res["sd_only_rows"]


def test_s2_engaged_excludes_cells_s0_marks_not_exercised(tmp_path):
    """FAILURE_LOG 2026-11-02: s2 scored a not-exercised cell 'pass'; s2_engaged leaves it UNDEFINED."""
    base = tmp_path / "shards"
    tag = "engtag"
    (base / tag).mkdir(parents=True)
    same = _turn(answer="yes")
    for seed in (42, 43):
        sdir = base / tag / ("s%d" % seed)
        sdir.mkdir()
        _cell(sdir, fac="row-a", verdict="regressed", diffs=[{"field": "answer", "on": "yes", "off": "no"}],
              ia=_turn(answer="yes"), ib=_turn(answer="yes"), le=_turn(answer="no"), le0=_turn(answer="no"))
        _cell(sdir, fac="row-b", verdict="not-exercised", lb=True, ia=same, ib=same, le=same, le0=same)
    fields = {"pin": PIN, "reply_fields": ["answer", "abstained", "recalled_svo"], "not_replayed": {},
              "rows": {f: {"fields": {"answer": {"class": "R"}}, "kept": [], "sd_only": True, "probe_turn": "well"}
                       for f in ("row-a", "row-b")}}
    fpath = tmp_path / "fields.json"
    fpath.write_text(json.dumps(fields))
    res = bd.score(str(base), tag, PIN, [42, 43], fields_path=str(fpath))
    assert res["s2"]["per_seed"]["42"]["n_exercised"] == 2          # the unengaged row is counted as a pass
    assert res["s2"]["per_seed"]["42"]["fraction"] == 0.5
    assert res["s2_engaged"]["per_seed"]["42"]["n_exercised"] == 1  # excluded
    assert res["s2_engaged"]["per_seed"]["42"]["fraction"] == 1.0
    assert res["s2_engaged"]["robust_core_n"] == 1
