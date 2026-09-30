"""Integration wave 1 (2026-09-29): the three production-default anchors, the flip script and the defaults guard.

Pins, without building a brain:
  * BYTE-IDENTICAL-OFF: while each anchor is False, its reader returns exactly what the pre-anchor expression
    `os.environ.get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")` returned, for every env value tried.
  * DEFAULT-ON SEMANTICS after a flip: ON when unset, OFF only for an explicit 0/false/no/off/empty.
  * ONE READER PER FLAG: no source file outside the anchored modules reads the anchored flags with os.environ (the
    FAILURE_LOG 2026-08-27 inner-gate-defaults-off class: a flip that lights one reader and leaves another off).
  * tools/integration_wave1_flip.py applies and reverts each packet's flip exactly, on a temporary copy of the repo
    files it edits, and leaves the ledger parseable.
  * tools/assert_iw1_defaults.py accepts/refuses revisions and environments as documented.
"""
import importlib
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ENV_VALUES = [None, "", "0", "1", "true", "TRUE", " on ", "yes", "off", "no", "false", "garbage", "2"]
_OLD_ON = ("1", "true", "yes", "on")
_EXPLICIT_OFF = ("0", "false", "no", "off", "")


def _old(v):
    return (v if v is not None else "").strip().lower() in _OLD_ON


def _set(monkeypatch, name, v):
    if v is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, v)


D6_READERS = [("BRAIN_D6_HEBBIAN_STORE", "hebbian_store_enabled"),
              ("BRAIN_D6_ENGRAM_VOCAB", "engram_vocab_enabled"),
              ("BRAIN_D6_ENGRAM_READTIME", "engram_readtime_enabled")]


@pytest.mark.parametrize("flag,fn", D6_READERS)
@pytest.mark.parametrize("v", ENV_VALUES)
def test_d6_readers_byte_identical_while_anchor_off(monkeypatch, flag, fn, v):
    from research.runners import d6_hebbian_store as m
    monkeypatch.setattr(m, "_D6_LEARN_DEFAULT_ON", False)   # the pre-flip semantics, whatever the repo state
    _set(monkeypatch, flag, v)
    assert getattr(m, fn)() is _old(v)


@pytest.mark.parametrize("flag,fn", D6_READERS)
@pytest.mark.parametrize("v", ENV_VALUES)
def test_d6_readers_default_on_semantics_after_flip(monkeypatch, flag, fn, v):
    from research.runners import d6_hebbian_store as m
    monkeypatch.setattr(m, "_D6_LEARN_DEFAULT_ON", True)
    _set(monkeypatch, flag, v)
    want = not (v is not None and v.strip().lower() in _EXPLICIT_OFF)
    assert getattr(m, fn)() is want


def test_d6_anchor_does_not_govern_the_lesion_or_prune(monkeypatch):
    from research.runners import d6_hebbian_store as m
    monkeypatch.setattr(m, "_D6_LEARN_DEFAULT_ON", True)
    monkeypatch.delenv("BRAIN_D6_HEBBIAN_FREEZE", raising=False)
    monkeypatch.delenv("BRAIN_D6_ENGRAM_PRUNE", raising=False)
    assert m.hebbian_freeze_lesioned() is False
    assert m.engram_prune_enabled() is False
    with pytest.raises(KeyError):
        m.d6_flag("BRAIN_D6_HEBBIAN_FREEZE")


def test_webapp_d6_chat_reader_follows_the_anchor(monkeypatch):
    from research.runners import d6_hebbian_store as m
    from webapp import d6_hebbian_chat as c
    monkeypatch.delenv("BRAIN_D6_HEBBIAN_STORE", raising=False)
    monkeypatch.setattr(m, "_D6_LEARN_DEFAULT_ON", False)
    assert c.d6_hebbian_enabled() is False
    monkeypatch.setattr(m, "_D6_LEARN_DEFAULT_ON", True)
    assert c.d6_hebbian_enabled() is True


MULTIREF_READERS = [("BRAIN_MULTIREF_FOCUS_BIND", "multiref_focus_bind_enabled", "_MULTIREF_FOCUS_BIND_DEFAULT_ON")]


@pytest.mark.parametrize("flag,fn,anchor", MULTIREF_READERS)
@pytest.mark.parametrize("v", ENV_VALUES)
def test_multiref_readers_byte_identical_while_anchor_off(monkeypatch, flag, fn, anchor, v):
    from research.runners import d6_multiref_wm_production_organ as m
    monkeypatch.setattr(m, anchor, False)
    _set(monkeypatch, flag, v)
    assert getattr(m, fn)() is _old(v)


@pytest.mark.parametrize("flag,fn,anchor", MULTIREF_READERS)
@pytest.mark.parametrize("v", ENV_VALUES)
def test_multiref_readers_default_on_semantics_after_flip(monkeypatch, flag, fn, anchor, v):
    from research.runners import d6_multiref_wm_production_organ as m
    monkeypatch.setattr(m, anchor, True)
    _set(monkeypatch, flag, v)
    want = not (v is not None and v.strip().lower() in _EXPLICIT_OFF)
    assert getattr(m, fn)() is want


@pytest.mark.parametrize("v", ENV_VALUES)
def test_c1_reader_off_default_and_after_flip(monkeypatch, v):
    """The C1 anchor predates this wave (2026-09-02); its reader already treats unset as the anchor value and any
    set value by the truthy list. Pinned here in both anchor states so the flip's semantics are explicit."""
    from research.runners import onebrain_xedge_surprise_worldmodel_production as m
    monkeypatch.setattr(m, "_XEDGE_SWM_DEFAULT_ON", False)
    _set(monkeypatch, "BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL", v)
    assert m.xedge_surprise_worldmodel_enabled() is _old(v)
    monkeypatch.setattr(m, "_XEDGE_SWM_DEFAULT_ON", True)
    want = True if v is None else (v.strip().lower() in _OLD_ON)
    assert m.xedge_surprise_worldmodel_enabled() is want


def test_c1_explicit_off_values_disable_after_flip(monkeypatch):
    from research.runners import onebrain_xedge_surprise_worldmodel_production as m
    monkeypatch.setattr(m, "_XEDGE_SWM_DEFAULT_ON", True)
    for v in _EXPLICIT_OFF:
        monkeypatch.setenv("BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL", v)
        assert m.xedge_surprise_worldmodel_enabled() is False


# ── one reader per flag ─────────────────────────────────────────────────────────────────────────────────────────
_ANCHORED = ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_ENGRAM_VOCAB", "BRAIN_D6_ENGRAM_READTIME",
             "BRAIN_MULTIREF_FOCUS_BIND", "BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL")
_READER_HOMES = {"research/runners/d6_hebbian_store.py", "research/runners/d6_multiref_wm_production_organ.py",
                 "research/runners/onebrain_xedge_surprise_worldmodel_production.py"}


def test_no_production_file_reads_an_anchored_flag_directly():
    """A production-path module (webapp/, sim/, research/runners/ minus experiment runners that SET the flags for their
    arms) may not read an anchored flag with os.environ/getenv: it must call the anchored function."""
    pat = re.compile(r"(os\.environ\.get|os\.getenv|environ\[)\(?\s*['\"](%s)['\"]" % "|".join(_ANCHORED))
    offenders = []
    for top in ("webapp", "sim", os.path.join("research", "runners")):
        for root, _dirs, files in os.walk(os.path.join(REPO, top)):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(root, fn), REPO)
                if rel in _READER_HOMES:
                    continue
                # experiment runners/probes set these flags per arm; they are not production readers
                base = os.path.basename(rel)
                if base.startswith("_") or base in ("load_bearing_fraction.py", "d6_learn_through_use_lb.py",
                                                    "d6_capacity_curve.py", "content_provenance_probe.py",
                                                    "ai_teacher_experiment.py", "d6_offpath_parity.py",
                                                    "lexicon_frame_junction.py"):
                    continue
                src = open(os.path.join(REPO, rel), encoding="utf-8", errors="replace").read()
                for m in pat.finditer(src):
                    offenders.append("%s: %s" % (rel, m.group(0)))
    assert not offenders, offenders


# ── the flip script ───────────────────────────────────────────────────────────────────────────────────────────
def _copy_tree(tmp_path):
    """Copy the files the flip edits, then bring the copy to the all-off state (after a real flip commit the repo
    itself is partly on; these tests always exercise apply from off)."""
    flip = importlib.import_module("tools.integration_wave1_flip")
    for rel in flip.files_touched():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(REPO, rel), dst)
    on = [f for f, st in flip.status(str(tmp_path)).items() if st == "on"]
    if on:
        flip.revert(str(tmp_path), on)
    return flip


@pytest.mark.parametrize("faculty", ["d6-hebbian-write", "wm-focus-bind", "xedge-surprise-worldmodel"])
def test_flip_apply_then_revert_is_exact(tmp_path, faculty):
    flip = _copy_tree(tmp_path)
    before = {rel: (tmp_path / rel).read_text() for rel in flip.files_touched()}
    flip.apply(str(tmp_path), [faculty], date="2026-10-01")
    after = {rel: (tmp_path / rel).read_text() for rel in flip.files_touched()}
    assert after != before
    spec = flip.PACKETS[faculty]
    anchor_src = (tmp_path / spec["anchor_file"]).read_text()
    assert re.search(r"\b%s\s*=\s*True\b" % spec["anchor"], anchor_src)
    assert not re.search(r"\b%s\s*=\s*False\b" % spec["anchor"], anchor_src)
    import yaml
    led = yaml.safe_load((tmp_path / flip.LEDGER).read_text())
    row = [r for r in led["rows"] if r["key"] == spec["ledger_row"]][0]
    assert any(a["assign"] == spec["anchor"] and a["file"] == spec["anchor_file"] for a in row["default_anchor"])
    assert ("subflag_flip_iw1_%s" % spec["slug"]) in row
    if spec.get("row_module"):
        import ast
        tree = ast.parse((tmp_path / spec["row_module"]).read_text())
        assert not any(isinstance(nd, ast.Assign) and any(getattr(t, "id", "") == "REQUIRED_ENV" for t in nd.targets)
                       for nd in tree.body)
    assert flip.status(str(tmp_path))[faculty] == "on"
    flip.revert(str(tmp_path), [faculty])
    assert {rel: (tmp_path / rel).read_text() for rel in flip.files_touched()} == before
    assert flip.status(str(tmp_path))[faculty] == "off"


def test_flip_all_three_then_revert_one(tmp_path):
    flip = _copy_tree(tmp_path)
    flip.apply(str(tmp_path), list(flip.PACKETS), date="2026-10-01")
    assert set(flip.status(str(tmp_path)).values()) == {"on"}
    flip.revert(str(tmp_path), ["wm-focus-bind"])
    st = flip.status(str(tmp_path))
    assert st == {"d6-hebbian-write": "on", "wm-focus-bind": "off", "xedge-surprise-worldmodel": "on"}


def test_flip_refuses_a_double_apply(tmp_path):
    flip = _copy_tree(tmp_path)
    flip.apply(str(tmp_path), ["wm-focus-bind"], date="2026-10-01")
    with pytest.raises(SystemExit):
        flip.apply(str(tmp_path), ["wm-focus-bind"], date="2026-10-01")


def test_repo_anchor_state_matches_the_ledger():
    """Each packet is on in source iff its ledger row carries the flip's anchor entry and note (Check A of the PI gate
    enforces the literal once the anchor is in the ledger; this also catches a ledger note left behind by a revert)."""
    import yaml
    flip = importlib.import_module("tools.integration_wave1_flip")
    led = yaml.safe_load(open(os.path.join(REPO, flip.LEDGER)))
    rows = {r["key"]: r for r in led["rows"]}
    for f, st in flip.status(REPO).items():
        spec = flip.PACKETS[f]
        row = rows[spec["ledger_row"]]
        has_anchor = any(a.get("assign") == spec["anchor"] for a in (row.get("default_anchor") or []))
        has_note = ("subflag_flip_iw1_%s" % spec["slug"]) in row
        assert has_anchor == has_note == (st == "on"), (f, st, has_anchor, has_note)


# ── the guard ────────────────────────────────────────────────────────────────────────────────────────────────
def _on_now():
    flip = importlib.import_module("tools.integration_wave1_flip")
    return sorted(f for f, st in flip.status(REPO).items() if st == "on")


def test_guard_accepts_the_repo_state_and_refuses_brain_flags():
    g = importlib.import_module("tools.assert_iw1_defaults")
    on = _on_now()
    assert g.problems(expect_on=on, environ={"SIM_BACKEND": "numpy", "LB_PMEM_DRIVE_PROBE": "1"}, root=REPO) == []
    assert g.problems(expect_on=on, environ={"BRAIN_MULTIREF_FOCUS_BIND": "1"}, root=REPO)
    other = [p for p in g.ANCHORS if p not in on]
    wrong = on[1:] if on else other[:1]          # any expectation that differs from the repo state
    assert g.problems(expect_on=wrong, environ={}, root=REPO)


def test_guard_cli_exit_codes():
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_")}
    on = _on_now()
    ok = subprocess.run([sys.executable, os.path.join(REPO, "tools", "assert_iw1_defaults.py"),
                         "--expect-on=%s" % ",".join(on)], cwd=REPO, env=env)
    assert ok.returncode == 0
    flipped_all = ",".join(sorted(importlib.import_module("tools.assert_iw1_defaults").ANCHORS))
    wrong = "" if on else flipped_all
    bad = subprocess.run([sys.executable, os.path.join(REPO, "tools", "assert_iw1_defaults.py"),
                          "--expect-on=%s" % wrong], cwd=REPO, env=env)
    assert bad.returncode == 1


# ── the two new LBF row modules ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mod", ["d6_hebbian_write", "xedge_surprise_worldmodel"])
def test_row_module_selftest(mod):
    m = importlib.import_module("research.runners.lbf_rows.%s" % mod)
    r = m.selftest()
    assert r["ok"], r


@pytest.mark.parametrize("mod,key", [("d6_hebbian_write", "d6-hebbian-write"),
                                     ("xedge_surprise_worldmodel", "xedge-surprise-worldmodel")])
def test_row_is_opt_in_before_the_flip(monkeypatch, mod, key):
    """At production defaults the row stays out of the registry (REQUIRED_ENV unmet); with the flags set it enters."""
    from research.runners import lbf_rows
    m = importlib.import_module("research.runners.lbf_rows.%s" % mod)
    if not hasattr(m, "REQUIRED_ENV"):
        pytest.skip("flipped: REQUIRED_ENV lifted, the row is in the headline registry")
    for k in m.REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)
    lesions, probes = {}, []
    rep = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[m])
    assert key not in lesions and not probes
    for k, v in m.REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    lesions, probes = {}, []
    lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[m])
    assert key in lesions and [p[0] for p in probes] == [key]


# ── the production-default identity tool (pure parts; the brain-building arms run in the HANDOFF) ─────────────────
def test_identity_arm_env():
    from research.runners import iw1_default_identity as t
    flags = [f for fl in t.PACKET_FLAGS.values() for f in fl]
    assert set(t.arm_env("default").values()) == {None}
    assert set(t.arm_env("off").values()) == {"0"} and set(t.arm_env("off")) == set(flags)
    on = t.arm_env("on:wm-focus-bind+xedge-surprise-worldmodel")
    assert on["BRAIN_MULTIREF_FOCUS_BIND"] == "1" and on["BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL"] == "1"
    assert on["BRAIN_D6_HEBBIAN_STORE"] is None
    with pytest.raises(SystemExit):
        t.arm_env("on:nope")
    assert set(t.PACKET_FLAGS) == set(importlib.import_module("tools.integration_wave1_flip").PACKETS)


def test_identity_compare_verdicts():
    from research.runners import iw1_default_identity as t
    row = lambda sha, ans="x": {"status": 200, "sha_canon": sha, "answer": ans, "abstained": False,  # noqa: E731
                                "recalled_svo": None}
    a = {"arm": "default", "seed": 7, "n_turns": 2, "turns": [row("a"), row("b")]}
    b = {"arm": "off", "seed": 7, "n_turns": 2, "turns": [row("a"), row("b")]}
    c = {"arm": "on:wm-focus-bind", "seed": 7, "n_turns": 2, "turns": [row("a"), row("c", "y")]}
    assert t.compare(a, b, "same")["ok"] is True
    assert t.compare(a, c, "same")["ok"] is False
    v = t.compare(a, c, "differs")
    assert v["ok"] is True and v["differing_turns"] == [1] and v["reply_differing_turns"] == [1]
    assert t.compare(a, b, "differs")["ok"] is False
    bad = dict(b, seed=8)
    assert t.compare(a, bad, "same")["verdict"] == "UNDEFINED"


# ── the registered scorer tools/iw1_score.py, on synthetic shard trees ─────────────────────────────────────────
PIN = "a" * 40


def _sidecar(env):
    return {"git_sha": PIN, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
            "source_manifest_verified_at_exit": True, "host": "node1", "env": env}


def _write_cell(root, tag, seed, fac, turn, ia, ib, le, le0, kind="neural-lesion", env=None, arm_env=None,
                verdict="regressed"):
    import json as _json
    d = os.path.join(root, tag, "s%d" % seed, fac)
    os.makedirs(d, exist_ok=True)
    env = dict({"SIM_BACKEND": "numpy"}, **(env or {}))
    lb = {"per_faculty": [{"faculty": fac, "kind": kind, "verdict": verdict, "load_bearing": verdict == "regressed",
                           "turn": turn, "diffs": [], "null_control_clean": True, "lesion_reproduced": True}]}
    _json.dump(lb, open(os.path.join(d, "lb.json"), "w"))
    _json.dump(_sidecar(env), open(os.path.join(d, "lb.json.prov.json"), "w"))
    for name, reply in (("intact_a_x.json", ia), ("intact_b_x.json", ib), ("lesion_x.json", le),
                        ("lesion_x.json.rep0", le0)):
        _json.dump({turn: {"answer": reply, "abstained": reply is None, "recalled_svo": None}},
                   open(os.path.join(d, name), "w"))
        _json.dump(_sidecar(dict(arm_env or {})), open(os.path.join(d, name + ".prov.json"), "w"))


def test_iw1_score_R_pass_fail_undefined(tmp_path):
    sc = importlib.import_module("tools.iw1_score")
    flags = {"BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_ENGRAM_VOCAB": "1", "BRAIN_D6_ENGRAM_READTIME": "1"}
    for s in sc.SEEDS:
        _write_cell(str(tmp_path), "R", s, "d6-hebbian-write", "iw1d6_probe", "deer", "deer", None, None,
                    env=flags, arm_env=dict(flags, BRAIN_D6_HEBBIAN_FREEZE="1"))
    res, code = sc.score_R("d6-hebbian-write", "R", PIN, base=str(tmp_path))
    assert code == sc.PASS and res["n_reply_load_bearing"] == 6, res
    # a seed whose lesion leaves the reply unchanged -> FAIL
    _write_cell(str(tmp_path), "R", 44, "d6-hebbian-write", "iw1d6_probe", "deer", "deer", "deer", "deer",
                env=flags, arm_env=flags)
    assert sc.score_R("d6-hebbian-write", "R", PIN, base=str(tmp_path))[1] == sc.FAIL
    # a noisy null (intact_a != intact_b) -> UNDEFINED, never a pass
    _write_cell(str(tmp_path), "R", 44, "d6-hebbian-write", "iw1d6_probe", "deer", "wolf", None, None,
                env=flags, arm_env=flags)
    assert sc.score_R("d6-hebbian-write", "R", PIN, base=str(tmp_path))[1] == sc.UNDEF
    # a BRAIN_* key outside the packet's own flags -> the cell fails provenance -> UNDEFINED
    _write_cell(str(tmp_path), "R", 44, "d6-hebbian-write", "iw1d6_probe", "deer", "deer", None, None,
                env=dict(flags, BRAIN_PMEM_OP_STABILIZER="1"), arm_env=flags)
    assert sc.score_R("d6-hebbian-write", "R", PIN, base=str(tmp_path))[1] == sc.UNDEF


def test_iw1_score_B_no_regression_and_regression(tmp_path):
    sc = importlib.import_module("tools.iw1_score")
    for s in sc.SEEDS:
        _write_cell(str(tmp_path), "base", s, "affect-coloring", "emo", "warm", "warm", "flat", "flat")
        _write_cell(str(tmp_path), "flip", s, "affect-coloring", "emo", "warm", "warm", "flat", "flat")
        _write_cell(str(tmp_path), "flip", s, "d6-hebbian-write", "iw1d6_probe", "deer", "deer", None, None)
    res, code = sc.score_B("base", PIN, "flip", PIN, ["d6-hebbian-write"], base=str(tmp_path))
    assert code == sc.PASS, res
    _write_cell(str(tmp_path), "flip", 101, "affect-coloring", "emo", "warm", "warm", "warm", "warm", verdict="pass")
    res, code = sc.score_B("base", PIN, "flip", PIN, ["d6-hebbian-write"], base=str(tmp_path))
    assert code == sc.FAIL and [r["faculty"] for r in res["regressions"]] == ["affect-coloring"] * 2, res
    # attribution: the all-off run at F reads the same drop -> excused (not caused by the flips)
    for s in sc.SEEDS:
        _write_cell(str(tmp_path), "alloff", s, "affect-coloring", "emo", "warm", "warm", "flat", "flat")
    _write_cell(str(tmp_path), "alloff", 101, "affect-coloring", "emo", "warm", "warm", "warm", "warm", verdict="pass")
    a, acode = sc.score_A(res, "alloff", PIN, base=str(tmp_path))
    assert acode == sc.PASS and a["faculties"]["affect-coloring"]["verdict"] == "excused", a
    # ... but if the all-off run restores it, the flips caused the drop
    _write_cell(str(tmp_path), "alloff", 101, "affect-coloring", "emo", "warm", "warm", "flat", "flat")
    a, acode = sc.score_A(res, "alloff", PIN, base=str(tmp_path))
    assert acode == sc.FAIL and a["faculties"]["affect-coloring"]["verdict"] == "flip-caused", a


def test_iw1_score_B_reference_undefined_cell_is_reported_not_scored(tmp_path):
    sc = importlib.import_module("tools.iw1_score")
    for s in sc.SEEDS:
        _write_cell(str(tmp_path), "base", s, "affect-coloring", "emo", "warm", "warm", "flat", "flat")
        _write_cell(str(tmp_path), "flip", s, "affect-coloring", "emo", "warm", "warm", "flat", "flat")
    _write_cell(str(tmp_path), "base", 44, "affect-coloring", "emo", "warm", "cold", "flat", "flat")  # noisy null
    res, code = sc.score_B("base", PIN, "flip", PIN, [], base=str(tmp_path))
    assert code == sc.PASS, res


# ── a defect found while screening wave-1 candidates (research/FAILURE_LOG.md 2026-09-29, learned-referent route) ──
_CORPUS = os.path.join(REPO, "data", "corpus", "tinystories.txt")


@pytest.mark.skipif(not os.path.exists(_CORPUS), reason="needs data/corpus/tinystories.txt (the lexicon's corpus)")
@pytest.mark.xfail(strict=True, reason="KNOWN DEFECT 2026-09-29: the learned-referent route admits wh-words as "
                   "referents ('what', 'who'); strict, so the fix turns this into a failure that removes the marker")
def test_learned_referent_route_admits_no_wh_word(monkeypatch):
    monkeypatch.setenv("BRAIN_LEARNED_REFERENT_LEXICON", "1")
    monkeypatch.delenv("BRAIN_LEARNED_REFERENT_LESION", raising=False)
    from research.runners import d6_multiref_wm_production_organ as m
    for text in ("what does the dog chase", "who are we talking about?", "what should we have for dinner"):
        refs = m.extract_referents(text)
        assert not ({"what", "who"} & set(refs)), (text, refs)
