"""Integration wave 2 (2026-09-30): the three production-default anchors, the flip script, the guards and the scorer.

Same shape as tests/test_integration_wave1_anchors.py (branch research/integration-wave1), with the wave-1 review's
lessons pinned from the start. Pins, without building a brain:
  * BYTE-IDENTICAL-OFF: while each anchor is False, its reader returns exactly what the pre-anchor expression returned,
    for every env value tried (false-belief-chat, transitive-chat); neural-selfid's anchor predates this wave and its
    reader's semantics are pinned in both anchor states.
  * DEFAULT-ON SEMANTICS after a flip: ON when unset, OFF only for an explicit 0/false/no/off/empty.
  * ONE READER PER FLAG: no production file outside the anchored module reads an anchored flag with os.environ.
  * tools/integration_wave2_flip.py applies and reverts each packet exactly on a temporary copy, ledger parseable.
  * tools/assert_iw2_defaults.py and tools/assert_iw2_node.py accept/refuse as documented.
  * the LBF rows: shapes, opt-in before the flip.
  * the identity runner: arm envs, the D0/D_p/S_p comparison verdicts, the degraded-brain refusal.
  * tools/iw2_score.py on synthetic shard trees: R PASS/FAIL/UNDEFINED, the degenerate-intact guard (UNDEFINED), the
    intact-content check (FAIL), B/A attribution, A never passing a B with undefined criteria.
  * the job lines: memory-gated, node-preflighted, shape-checked; the HANDOFF queue reasons: a prereg path that
    resolves (no trailing ';'), a dispatchable mem_gb (<= 11) and a max_age_h (<= 240) on every line.
"""
import importlib
import json
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


def _set(monkeypatch, name, v):
    if v is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, v)


# ── the two new anchors: byte-identical while off, default-on semantics after the flip ──────────────────────────────
NEW_ANCHORS = [
    # (module, anchor, reader, flag, the pre-anchor expression's default for an unset flag)
    ("webapp.false_belief_chat", "_FALSE_BELIEF_CHAT_DEFAULT_ON", "false_belief_chat_enabled",
     "BRAIN_FALSE_BELIEF_CHAT", "0"),
    ("webapp.reasoning_transitive_chat", "_TRANSITIVE_CHAT_DEFAULT_ON", "transitive_chat_enabled",
     "BRAIN_TRANSITIVE_CHAT", ""),
]


@pytest.mark.parametrize("mod,anchor,fn,flag,unset_default", NEW_ANCHORS)
@pytest.mark.parametrize("v", ENV_VALUES)
def test_new_readers_byte_identical_while_anchor_off(monkeypatch, mod, anchor, fn, flag, unset_default, v):
    m = importlib.import_module(mod)
    monkeypatch.setattr(m, anchor, False)
    _set(monkeypatch, flag, v)
    old = (v if v is not None else unset_default).strip().lower() in ("1", "true", "on", "yes")
    assert getattr(m, fn)() is old


@pytest.mark.parametrize("mod,anchor,fn,flag,unset_default", NEW_ANCHORS)
@pytest.mark.parametrize("v", ENV_VALUES)
def test_new_readers_default_on_semantics_after_flip(monkeypatch, mod, anchor, fn, flag, unset_default, v):
    m = importlib.import_module(mod)
    monkeypatch.setattr(m, anchor, True)
    _set(monkeypatch, flag, v)
    want = not (v is not None and v.strip().lower() in _EXPLICIT_OFF)
    assert getattr(m, fn)() is want


@pytest.mark.parametrize("v", ENV_VALUES)
def test_selfid_reader_both_anchor_states(monkeypatch, v):
    """The SELFID anchor predates this wave (rank-13, 2026-09-05): unset -> the anchor; any set value -> ON unless it is
    an explicit 0/false/off/no/"" (so 'garbage' reads ON in both states). Pinned so the flip's semantics are explicit."""
    from research.runners import brain_chat_tui as m
    _set(monkeypatch, "BRAIN_NEURAL_SELFID", v)
    set_on = v is not None and v.strip().lower() not in _EXPLICIT_OFF
    monkeypatch.setattr(m, "_NEURAL_SELFID_DEFAULT_ON", False)
    assert m._neural_selfid_enabled() is (False if v is None else set_on)
    monkeypatch.setattr(m, "_NEURAL_SELFID_DEFAULT_ON", True)
    assert m._neural_selfid_enabled() is (True if v is None else set_on)


def test_selfid_flip_leaves_anaphora_abstain_off(monkeypatch):
    """The packet flips SELFID ALONE: BRAIN_NEURAL_ANAPHORA_ABSTAIN (the seed-44 regression of the 2026-09-05 pair flip)
    keeps its own False anchor."""
    from research.runners import brain_chat_tui as m
    monkeypatch.delenv("BRAIN_NEURAL_ANAPHORA_ABSTAIN", raising=False)
    monkeypatch.setattr(m, "_NEURAL_SELFID_DEFAULT_ON", True)
    assert m._neural_anaphora_abstain_enabled() is False
    flip = importlib.import_module("tools.integration_wave2_flip")
    assert all(s["anchor"] != "_NEURAL_ANAPHORA_ABSTAIN_DEFAULT_ON" for s in flip.PACKETS.values())


def test_transitive_scope_guard_declines_closed_class_questions(monkeypatch):
    """The wave-2 scope guard (before any gate data): a 'does <w1> <w2> <w3>?' question with a determiner/pronoun/
    wh-word token is not the transitive shape and falls through (None) even with the flag on; bare-noun questions are
    unchanged."""
    from webapp import reasoning_transitive_chat as T
    for q in ("does the cat sleep?", "does it rain now?", "does a dog bark?", "does he like cats?",
              "does dog chase it?", "does what precede e3?"):
        assert T.parse_transitive_question(q) is None, q
    assert T.parse_transitive_question("does e0 precede e3?") == ("e0", "precede", "e3")
    assert T.parse_transitive_question("does dog chase cat?") == ("dog", "chase", "cat")
    assert T.parse_transitive_question("does cat precede river?") == ("cat", "precede", "river")
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    assert T.resolve_transitive_query(object(), "does the cat sleep?") is None      # falls through to the ordinary path


# ── one reader per flag ─────────────────────────────────────────────────────────────────────────────────────────
_ANCHORED = ("BRAIN_FALSE_BELIEF_CHAT", "BRAIN_TRANSITIVE_CHAT", "BRAIN_NEURAL_SELFID")
_READER_HOMES = {"webapp/false_belief_chat.py", "webapp/reasoning_transitive_chat.py",
                 "research/runners/brain_chat_tui.py"}
# experiment runners/probes that SET these flags for their arms (they are not production readers)
_SETTERS = ("load_bearing_fraction.py", "reasoning_transitive_chat_gate.py", "iw2_default_identity.py")


def test_no_production_file_reads_an_anchored_flag_directly():
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
                base = os.path.basename(rel)
                if base.startswith("_") or base in _SETTERS:
                    continue
                src = open(os.path.join(REPO, rel), encoding="utf-8", errors="replace").read()
                for m in pat.finditer(src):
                    offenders.append("%s: %s" % (rel, m.group(0)))
    assert not offenders, offenders


# ── the flip script ───────────────────────────────────────────────────────────────────────────────────────────
PACKETS = ["false-belief-chat", "transitive-chat", "neural-selfid"]


def _copy_tree(tmp_path):
    """Copy the files the flip edits, then bring the copy to the all-off state."""
    flip = importlib.import_module("tools.integration_wave2_flip")
    for rel in flip.files_touched():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(REPO, rel), dst)
    on = [f for f, st in flip.status(str(tmp_path)).items() if st == "on"]
    if on:
        flip.revert(str(tmp_path), on)
    return flip


def test_packet_lists_agree():
    flip = importlib.import_module("tools.integration_wave2_flip")
    guard = importlib.import_module("tools.assert_iw2_defaults")
    score = importlib.import_module("tools.iw2_score")
    ident = importlib.import_module("research.runners.iw2_default_identity")
    assert sorted(flip.PACKETS) == sorted(guard.ANCHORS) == sorted(score.PACKETS) == sorted(ident.PACKET_FLAGS) \
        == sorted(PACKETS)
    for p, spec in flip.PACKETS.items():
        assert guard.ANCHORS[p] == (spec["anchor_file"], spec["anchor"])
        assert tuple(score.PACKETS[p]["flags"]) == spec["flags"] == ident.PACKET_FLAGS[p]


@pytest.mark.parametrize("faculty", PACKETS)
def test_flip_apply_then_revert_is_exact(tmp_path, faculty):
    flip = _copy_tree(tmp_path)
    before = {rel: (tmp_path / rel).read_text() for rel in flip.files_touched()}
    flip.apply(str(tmp_path), [faculty], date="2026-10-01")
    after = {rel: (tmp_path / rel).read_text() for rel in flip.files_touched()}
    assert after != before
    spec = flip.PACKETS[faculty]
    anchor_src = (tmp_path / spec["anchor_file"]).read_text()
    assert re.search(r"^%s = True$" % spec["anchor"], anchor_src, re.M)
    assert not re.search(r"^%s = False$" % spec["anchor"], anchor_src, re.M)
    import yaml
    led = yaml.safe_load((tmp_path / flip.LEDGER).read_text())
    row = [r for r in led["rows"] if r["key"] == spec["ledger_row"]][0]
    assert any(a["assign"] == spec["anchor"] and a["file"] == spec["anchor_file"] for a in row["default_anchor"])
    assert ("subflag_flip_iw2_%s" % spec["slug"]) in row
    assert row["on_by_default"] in (True, "YES")          # a sub-rung of a row that is already on by default
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
    flip.revert(str(tmp_path), ["transitive-chat"])
    assert flip.status(str(tmp_path)) == {"false-belief-chat": "on", "transitive-chat": "off", "neural-selfid": "on"}


def test_flip_refuses_a_double_apply(tmp_path):
    flip = _copy_tree(tmp_path)
    flip.apply(str(tmp_path), ["neural-selfid"], date="2026-10-01")
    with pytest.raises(SystemExit):
        flip.apply(str(tmp_path), ["neural-selfid"], date="2026-10-01")


def test_flip_passes_the_production_integration_gate_check_a(tmp_path):
    """After a flip of all three, the PI gate's anchor check (Check A) reads every new default_anchor as consistent."""
    flip = _copy_tree(tmp_path)
    flip.apply(str(tmp_path), list(flip.PACKETS), date="2026-10-01")
    import yaml
    led = yaml.safe_load((tmp_path / flip.LEDGER).read_text())
    for f, spec in flip.PACKETS.items():
        row = [r for r in led["rows"] if r["key"] == spec["ledger_row"]][0]
        for a in row["default_anchor"]:
            if a["assign"] != spec["anchor"]:
                continue
            src = (tmp_path / a["file"]).read_text()
            assert len(re.findall(r"^%s = %s$" % (a["assign"], a["on_value"]), src, re.M)) == a["count"]
            assert not re.findall(r"^%s = %s$" % (a["assign"], a["off_value"]), src, re.M)


def test_repo_anchor_state_matches_the_ledger():
    import yaml
    flip = importlib.import_module("tools.integration_wave2_flip")
    led = yaml.safe_load(open(os.path.join(REPO, flip.LEDGER)))
    rows = {r["key"]: r for r in led["rows"]}
    for f, st in flip.status(REPO).items():
        spec = flip.PACKETS[f]
        row = rows[spec["ledger_row"]]
        has_anchor = any(a.get("assign") == spec["anchor"] for a in (row.get("default_anchor") or []))
        has_note = ("subflag_flip_iw2_%s" % spec["slug"]) in row
        assert has_anchor == has_note == (st == "on"), (f, st, has_anchor, has_note)


# ── the guards ───────────────────────────────────────────────────────────────────────────────────────────────
def _on_now():
    flip = importlib.import_module("tools.integration_wave2_flip")
    return sorted(f for f, st in flip.status(REPO).items() if st == "on")


def test_guard_accepts_the_repo_state_and_refuses_brain_flags():
    g = importlib.import_module("tools.assert_iw2_defaults")
    on = _on_now()
    assert g.problems(expect_on=on, environ={"SIM_BACKEND": "numpy", "LB_PMEM_DRIVE_PROBE": "1"}, root=REPO) == []
    assert g.problems(expect_on=on, environ={"BRAIN_TRANSITIVE_CHAT": "1"}, root=REPO)
    other = [p for p in g.ANCHORS if p not in on]
    wrong = on[1:] if on else other[:1]
    assert g.problems(expect_on=wrong, environ={}, root=REPO)
    assert g.problems(expect_on=["nope"], environ={}, root=REPO)


def test_guard_cli_exit_codes():
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_")}
    on = _on_now()
    ok = subprocess.run([sys.executable, os.path.join(REPO, "tools", "assert_iw2_defaults.py"),
                         "--expect-on=%s" % ",".join(on)], cwd=REPO, env=env)
    assert ok.returncode == 0
    wrong = "" if on else ",".join(sorted(importlib.import_module("tools.assert_iw2_defaults").ANCHORS))
    bad = subprocess.run([sys.executable, os.path.join(REPO, "tools", "assert_iw2_defaults.py"),
                          "--expect-on=%s" % wrong], cwd=REPO, env=env)
    assert bad.returncode == 1


def test_node_guard_refuses_without_corpus_or_ltm(tmp_path):
    n = importlib.import_module("tools.assert_iw2_node")
    ok, msg = n.check(root=str(tmp_path), resolver=lambda: (True, "/x/wikidata_100k"))
    assert not ok and "corpus" in msg
    (tmp_path / "data" / "corpus").mkdir(parents=True)
    for f in n.CORPUS_FILES:
        (tmp_path / "data" / "corpus" / f).write_text("x")
    assert n.check(root=str(tmp_path), resolver=lambda: (True, "/x/wikidata_100k")) == (True,
                                                                                         "corpus ok; ltm_bundle=wikidata_100k")
    assert n.check(root=str(tmp_path), resolver=lambda: (True, None))[0] is False
    assert n.check(root=str(tmp_path), resolver=lambda: (False, None))[0] is False


# ── the LBF rows ───────────────────────────────────────────────────────────────────────────────────────────────
ROWS = [("tom_false_belief_chat", "tom-false-belief"), ("reasoning_transitive_chat", "reasoning-transitive-chat"),
        ("neural_selfid", "neural-selfid")]


@pytest.mark.parametrize("mod,key", ROWS)
def test_row_module_selftest(mod, key):
    m = importlib.import_module("research.runners.lbf_rows.%s" % mod)
    r = m.selftest()
    assert r["ok"], r
    assert [p[0] for p in m.EXTRA_PROBES] == [key]


@pytest.mark.parametrize("mod,key", ROWS)
def test_row_is_opt_in_before_the_flip(monkeypatch, mod, key):
    from research.runners import lbf_rows
    m = importlib.import_module("research.runners.lbf_rows.%s" % mod)
    if not hasattr(m, "REQUIRED_ENV"):
        pytest.skip("flipped: REQUIRED_ENV lifted, the row is in the headline registry")
    for k in m.REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)
    lesions, probes = {}, []
    lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[m])
    assert key not in lesions and not probes
    for k, v in m.REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    lesions, probes = {}, []
    rep = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[m])
    assert key in lesions and [p[0] for p in probes] == [key], rep
    assert lesions[key]["kind"] == "neural-lesion"


def test_control_rows_are_opt_in_by_a_probe_flag_and_never_lifted():
    """The permuted controls (research/runners/lbf_rows/iw2_controls.py) must never enter a production headline battery:
    opt-in by LB_IW2_CONTROLS=1 only, never a flip target, and each control row is the scorer's control for a packet."""
    from research.runners import lbf_rows
    m = importlib.import_module("research.runners.lbf_rows.iw2_controls")
    assert m.selftest()["ok"], m.selftest()
    flip = importlib.import_module("tools.integration_wave2_flip")
    assert all("iw2_controls" not in (s["row_module"] or "") for s in flip.PACKETS.values())
    score = importlib.import_module("tools.iw2_score")
    assert sorted(s["control"]["row"] for s in score.PACKETS.values()) == sorted(m.EXTRA_LESIONS)
    for p, spec in score.PACKETS.items():   # each control's lesion is its packet's lesion flag
        assert m.EXTRA_LESIONS[spec["control"]["row"]]["flag"] in spec["lesion_flags"], p
    lesions, probes = {}, []
    old = os.environ.pop("LB_IW2_CONTROLS", None)
    try:
        lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[m])
    finally:
        if old is not None:
            os.environ["LB_IW2_CONTROLS"] = old
    assert not lesions and not probes


def test_row_expectation_helpers_agree_with_the_scorer():
    """The rows' dev-read helpers (tools.lab.lever) and the registered scorer state the same predictions."""
    ns = importlib.import_module("research.runners.lbf_rows.neural_selfid")
    ctl = importlib.import_module("research.runners.lbf_rows.iw2_controls")
    score = importlib.import_module("tools.iw2_score")
    yes, no = {"answer": "the brain uses the spikes", "abstained": False}, {"answer": "I don't know.", "abstained": True}
    assert ns.expectation_holds(yes, no)["held"] is True
    assert ns.expectation_holds(yes, yes)["held"] is False and ns.expectation_holds(yes, yes)["moved"] is False
    for p, spec in score.PACKETS.items():
        c = spec["control"]
        e = ctl.EXPECT[c["row"]]
        assert e["abstains"] == c["intact_abstains"] and e["must"] == c["intact_must_contain"] \
            and e["must_not"] == c["intact_must_not_contain"], p
    box = {"answer": "Sally will look in the box for the marble.", "abstained": False}
    assert ctl.expectation_holds("tom-true-belief-ctl", box, box)["held"] is True
    assert ctl.expectation_holds("tom-true-belief-ctl", box, no)["held"] is False       # the lesion moved it
    assert ctl.expectation_holds("selfid-nofact-ctl", no, no)["held"] is True
    assert ctl.expectation_holds("selfid-nofact-ctl", yes, no)["held"] is False


def test_row_probe_messages_match_the_scorer():
    """The scorer's echo guard and content check must use the probe turn each row really plays."""
    score = importlib.import_module("tools.iw2_score")
    for mod, key in ROWS:
        m = importlib.import_module("research.runners.lbf_rows.%s" % mod)
        label = m.EXTRA_PROBES[0][1]
        msg = [t[1] for t in m.EXTRA_TURNS if t[0] == label][0]
        spec = [s for s in score.PACKETS.values() if s["row"] == key][0]
        assert spec["probe_message"] == msg, (key, msg)


# ── the production-default identity tool (pure parts) ────────────────────────────────────────────────────────
def test_identity_arm_env_and_script():
    from research.runners import iw2_default_identity as t
    flags = [f for fl in t.PACKET_FLAGS.values() for f in fl]
    assert set(t.arm_env("default").values()) == {None}
    assert set(t.arm_env("off").values()) == {"0"} and set(t.arm_env("off")) == set(flags)
    on = t.arm_env("on:false-belief-chat+neural-selfid")
    assert on["BRAIN_FALSE_BELIEF_CHAT"] == "1" and on["BRAIN_NEURAL_SELFID"] == "1"
    assert on["BRAIN_TRANSITIVE_CHAT"] is None
    with pytest.raises(SystemExit):
        t.arm_env("on:nope")
    tags = [x[1] for x in t.TURNS]
    for p in t.PACKET_FLAGS:
        assert p in tags, p                               # every packet is reached by >= 1 turn
    assert t.SCOPE in tags
    # every segment starts with a reset and with scope turns (so a scope difference is never carried-over state)
    for i, (_m, tag, reset) in enumerate(t.TURNS):
        if reset:
            assert tag == t.SCOPE, i
    assert t.TURNS[0][2] is True
    assert t.arm_file_name("on:a+b") == "on_a_b"


def _row(sha, ans="x", tag="scope"):
    return {"status": 200, "sha_canon": sha, "answer": ans, "abstained": False, "recalled_svo": None, "for": tag}


_ONE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _arm(turns, **kw):
    base = {"arm": "default", "seed": 7, "n_turns": len(turns), "turns": turns, "corpus_present": True,
            "ltm_bundle": "wikidata_100k", "script_sha": "s", "threads": dict(_ONE)}
    base.update(kw)
    return base


def test_identity_compare_verdicts():
    from research.runners import iw2_default_identity as t
    a = _arm([_row("a"), _row("b", tag="transitive-chat")])
    b = _arm([_row("a"), _row("b", tag="transitive-chat")], arm="off")
    c = _arm([_row("a"), _row("c", "y", tag="transitive-chat")], arm="on:transitive-chat")
    d = _arm([_row("z", "w"), _row("c", "y", tag="transitive-chat")], arm="on:transitive-chat")
    assert t.compare(a, b, "same")["verdict"] == "IDENTICAL"
    assert t.compare(a, c, "same")["ok"] is False
    v = t.compare(a, c, "differs", "transitive-chat")
    assert v["ok"] is True and v["own_differing_turns"] == [1] and v["reply_differing_turns"] == [1]
    assert t.compare(a, b, "differs", "transitive-chat")["ok"] is None          # vacuous -> UNDEFINED
    assert t.compare(a, c, "differs", "neural-selfid")["ok"] is False           # changes a turn it should not reach
    assert t.compare(a, d, "differs", "transitive-chat")["ok"] is False         # out of scope
    assert t.compare(a, dict(b, seed=8), "same")["verdict"] == "UNDEFINED"
    assert t.compare(a, dict(b, script_sha="other"), "same")["verdict"] == "UNDEFINED"
    assert t.compare(a, dict(b, corpus_present=False), "same")["verdict"] == "UNDEFINED"
    assert t.compare(a, dict(b, ltm_bundle="wikidata_core_15k"), "same")["verdict"] == "UNDEFINED"
    assert t.compare(a, dict(b, threads=dict(_ONE, OMP_NUM_THREADS="2")), "same")["verdict"] == "UNDEFINED"
    assert t.compare(a, {k: v for k, v in b.items() if k != "threads"}, "same")["verdict"] == "UNDEFINED"
    with pytest.raises(SystemExit):
        t.compare(a, c, "differs")


def test_identity_compare_cli_exit_codes(tmp_path):
    a = _arm([_row("a"), _row("b", tag="transitive-chat")])
    b = _arm([_row("a"), _row("b", tag="transitive-chat")], arm="on:transitive-chat")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a)); pb.write_text(json.dumps(b))
    run = lambda *x: subprocess.run([sys.executable, "-m", "research.runners.iw2_default_identity", "--compare",  # noqa
                                     str(pa), str(pb)] + list(x), cwd=REPO, capture_output=True).returncode
    assert run("--expect", "same") == 0
    assert run("--expect", "differs", "--packet", "transitive-chat") == 3       # vacuous = UNDEFINED, never a pass


def test_identity_arm_refuses_multithreaded_blas(tmp_path, monkeypatch):
    """A byte-identity arm must be bit-reproducible: refuse (exit 3) before any build unless all thread vars are 1."""
    from research.runners import iw2_default_identity as t
    monkeypatch.chdir(tmp_path)
    for k in t.THREAD_VARS:
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(SystemExit) as e:
        t.run_arm("default", 7, str(tmp_path / "out.json"))
    assert e.value.code == 3 and not (tmp_path / "out.json").exists()


def test_identity_job_lines_pin_single_threaded_blas(tmp_path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    r = _mk("identity", head, "default", env_extra={"IW2_JOBS_OUTD": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    ln = (tmp_path / ("identity_%s.txt" % head[:7])).read_text()
    for k in ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1"):
        assert k in ln, ln


def test_identity_arm_refuses_to_build_without_the_corpus(tmp_path, monkeypatch):
    from research.runners import iw2_default_identity as t
    monkeypatch.chdir(tmp_path)
    for k in t.THREAD_VARS:
        monkeypatch.setenv(k, "1")
    monkeypatch.delenv("IW2_ALLOW_DEGRADED", raising=False)
    before = dict(os.environ)
    with pytest.raises(SystemExit) as e:
        t.run_arm("default", 7, str(tmp_path / "out.json"))
    assert e.value.code == 3
    assert not (tmp_path / "out.json").exists()
    assert dict(os.environ) == before


# ── the registered scorer tools/iw2_score.py, on synthetic shard trees ─────────────────────────────────────────
PIN = "a" * 40


def _sidecar(env):
    return {"git_sha": PIN, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
            "source_manifest_verified_at_exit": True, "host": "node1", "env": env}


def _write_cell(root, tag, seed, fac, turn, arms, env=None, arm_env=None, verdict="regressed"):
    """One shard dir: lb.json + sidecar and the four arm files; `arms` = {"ia": reply, "le": reply} (full dicts)."""
    d = os.path.join(root, tag, "s%d" % seed, fac)
    os.makedirs(d, exist_ok=True)
    env = dict({"SIM_BACKEND": "numpy"}, **(env or {}))
    lb = {"per_faculty": [{"faculty": fac, "kind": "neural-lesion", "verdict": verdict,
                           "load_bearing": verdict == "regressed", "turn": turn, "diffs": [],
                           "null_control_clean": True, "lesion_reproduced": True}]}
    json.dump(lb, open(os.path.join(d, "lb.json"), "w"))
    json.dump(_sidecar(env), open(os.path.join(d, "lb.json.prov.json"), "w"))
    for name, key in (("intact_a_x.json", "ia"), ("intact_b_x.json", "ib"), ("lesion_x.json", "le"),
                      ("lesion_x.json.rep0", "le")):
        json.dump({turn: arms.get(key, arms["ia"])}, open(os.path.join(d, name), "w"))
        json.dump(_sidecar(dict(arm_env or {})), open(os.path.join(d, name + ".prov.json"), "w"))


def _r(ans, abstained=False, svo=None):
    return {"answer": ans, "abstained": abstained, "recalled_svo": svo}


_ABSTAIN = _r("I don't know about that.", True)
FB = {"BRAIN_FALSE_BELIEF_CHAT": "1"}
FB_INTACT = _r("Sally will look in the basket for the marble.")
FB_LESION = _r("Sally will look in the box for the marble.")


FB_CTL = {"ia": FB_LESION, "le": FB_LESION}          # the witnessed-move control: the true location, lesion-inert


def _fill(tmp_path, tag="R", arms=None, env=FB, arm_env=None, verdict="regressed", control=True):
    sc = importlib.import_module("tools.iw2_score")
    aenv = arm_env if arm_env is not None else dict(env, BRAIN_FALSE_BELIEF_LESION="1")
    for s in sc.SEEDS:
        _write_cell(str(tmp_path), tag, s, "tom-false-belief", "tom_fb", arms or {"ia": FB_INTACT, "le": FB_LESION},
                    env=env, arm_env=aenv, verdict=verdict)
        if control:
            _write_cell(str(tmp_path), tag, s, "tom-true-belief-ctl", "iw2ctl_tb", FB_CTL,
                        env=dict(env, LB_IW2_CONTROLS="1"), arm_env=aenv, verdict="pass")
    return sc


def test_iw2_score_R_pass(tmp_path):
    sc = _fill(tmp_path)
    res, code = sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))
    assert code == sc.PASS and res["n_reply_load_bearing"] == 6, res


def test_iw2_score_R_control_missing_is_undefined_and_violated_is_a_fail(tmp_path):
    """The permuted control is part of the bar: a missing control cell -> UNDEFINED (never a pass); a control whose
    intact reply names the stale location (the organ ignores witnessing) or whose reply the lesion changes -> FAIL."""
    sc = _fill(tmp_path)
    shutil.rmtree(os.path.join(str(tmp_path), "R", "s100", "tom-true-belief-ctl"))
    res, code = sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))
    assert code == sc.UNDEF and res["control"]["n_defined"] == 5, res["control"]
    sc = _fill(tmp_path)
    _write_cell(str(tmp_path), "R", 100, "tom-true-belief-ctl", "iw2ctl_tb", {"ia": FB_INTACT, "le": FB_INTACT},
                env=dict(FB, LB_IW2_CONTROLS="1"), arm_env=FB, verdict="pass")
    res, code = sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))
    assert code == sc.FAIL and res["control"]["per_seed"]["100"]["verdict"] == "control-violated", res["control"]
    sc = _fill(tmp_path)
    _write_cell(str(tmp_path), "R", 100, "tom-true-belief-ctl", "iw2ctl_tb", {"ia": FB_LESION, "le": FB_INTACT},
                env=dict(FB, LB_IW2_CONTROLS="1"), arm_env=FB)
    assert sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))[1] == sc.FAIL


def test_iw2_score_R_abstaining_control(tmp_path):
    """transitive-chat / neural-selfid controls predict an abstention: an intact 'yes' is a control violation."""
    sc = importlib.import_module("tools.iw2_score")
    TR = {"BRAIN_TRANSITIVE_CHAT": "1"}
    yes = _r("I believe the cat precedes the river, but I reasoned that myself rather than being told it directly.")
    for s in sc.SEEDS:
        _write_cell(str(tmp_path), "T", s, "reasoning-transitive-chat", "transitive_nonadjacent",
                    {"ia": yes, "le": _ABSTAIN}, env=TR, arm_env=TR)
        _write_cell(str(tmp_path), "T", s, "transitive-reversed-ctl", "iw2ctl_rv", {"ia": _ABSTAIN, "le": _ABSTAIN},
                    env=dict(TR, LB_IW2_CONTROLS="1"), arm_env=TR, verdict="pass")
    assert sc.score_R("transitive-chat", "T", PIN, base=str(tmp_path))[1] == sc.PASS
    _write_cell(str(tmp_path), "T", 42, "transitive-reversed-ctl", "iw2ctl_rv", {"ia": yes, "le": _ABSTAIN},
                env=dict(TR, LB_IW2_CONTROLS="1"), arm_env=TR)
    res, code = sc.score_R("transitive-chat", "T", PIN, base=str(tmp_path))
    assert code == sc.FAIL and res["control"]["per_seed"]["42"]["verdict"] == "control-violated", res["control"]


def test_iw2_score_R_fail_when_the_lesion_leaves_the_reply(tmp_path):
    sc = _fill(tmp_path)
    _write_cell(str(tmp_path), "R", 44, "tom-false-belief", "tom_fb", {"ia": FB_INTACT, "le": FB_INTACT}, env=FB,
                arm_env=FB, verdict="pass")
    assert sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))[1] == sc.FAIL


def test_iw2_score_R_undefined_on_noise_or_a_stray_brain_key(tmp_path):
    sc = _fill(tmp_path)
    _write_cell(str(tmp_path), "R", 44, "tom-false-belief", "tom_fb",
                {"ia": FB_INTACT, "ib": _r("Sally will look in the drawer."), "le": FB_LESION}, env=FB, arm_env=FB)
    assert sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))[1] == sc.UNDEF
    _write_cell(str(tmp_path), "R", 44, "tom-false-belief", "tom_fb", {"ia": FB_INTACT, "le": FB_LESION},
                env=dict(FB, BRAIN_PMEM_OP_STABILIZER="1"), arm_env=FB)
    assert sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))[1] == sc.UNDEF


@pytest.mark.parametrize("intact", [
    _r(""), _r(None), _ABSTAIN,                                         # empty, absent, an abstention
    _r("Sally puts the marble in the basket. Sally leaves the room. Anne moves the marble to the box. "
       "Where will Sally look for the marble?"),                          # an echo of the probe
])
def test_iw2_score_R_degenerate_intact_never_passes(tmp_path, intact):
    sc = _fill(tmp_path)
    _write_cell(str(tmp_path), "R", 101, "tom-false-belief", "tom_fb", {"ia": intact, "le": FB_LESION}, env=FB,
                arm_env=FB)
    res, code = sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))
    assert code == sc.UNDEF and res["per_seed"]["101"]["state"] == "UNDEFINED:degenerate-intact", res["per_seed"]["101"]


def test_iw2_score_R_wrong_intact_content_is_a_fail(tmp_path):
    """A non-degenerate intact reply that states the WRONG content (the true location: no false belief) while the
    lesion changes it is not the capability: DEFINED `intact-wrong`, R FAIL (never a pass, never silently undefined)."""
    sc = _fill(tmp_path)
    _write_cell(str(tmp_path), "R", 43, "tom-false-belief", "tom_fb", {"ia": FB_LESION, "le": _ABSTAIN}, env=FB,
                arm_env=FB)
    res, code = sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))
    assert code == sc.FAIL and res["per_seed"]["43"]["reply_verdict"] == "intact-wrong", res["per_seed"]["43"]


def test_iw2_score_R_intact_abstain_equal_to_lesion_stays_a_fail(tmp_path):
    sc = _fill(tmp_path, arms={"ia": _ABSTAIN, "le": _ABSTAIN}, verdict="pass")
    assert sc.score_R("false-belief-chat", "R", PIN, base=str(tmp_path))[1] == sc.FAIL


def _std(tmp_path, tag, fac, turn, ia, le, seeds=None, verdict="regressed"):
    sc = importlib.import_module("tools.iw2_score")
    for s in (seeds or sc.SEEDS):
        _write_cell(str(tmp_path), tag, s, fac, turn, {"ia": _r(ia), "le": _r(le)}, verdict=verdict)


def test_iw2_score_B_no_regression_regression_and_attribution(tmp_path):
    sc = importlib.import_module("tools.iw2_score")
    _std(tmp_path, "base", "affect-coloring", "emo", "warm", "flat")
    _std(tmp_path, "flip", "affect-coloring", "emo", "warm", "flat")
    _fill(tmp_path, tag="flip", env={}, arm_env={})
    res, code = sc.score_B("base", PIN, "flip", PIN, ["false-belief-chat"], base=str(tmp_path))
    assert code == sc.PASS and res["new_rows_s2_at_flip"] == {"false-belief-chat": 6}, res
    _std(tmp_path, "flip", "affect-coloring", "emo", "warm", "warm", seeds=[101], verdict="pass")
    res, code = sc.score_B("base", PIN, "flip", PIN, ["false-belief-chat"], base=str(tmp_path))
    assert code == sc.FAIL and [r["faculty"] for r in res["regressions"]] == ["affect-coloring"] * 2, res
    _std(tmp_path, "alloff", "affect-coloring", "emo", "warm", "flat")
    _std(tmp_path, "alloff", "affect-coloring", "emo", "warm", "warm", seeds=[101], verdict="pass")
    a, acode = sc.score_A(res, "alloff", PIN, base=str(tmp_path))
    assert acode == sc.PASS and a["faculties"]["affect-coloring"]["verdict"] == "excused", a
    _std(tmp_path, "alloff", "affect-coloring", "emo", "warm", "flat", seeds=[101])
    a, acode = sc.score_A(res, "alloff", PIN, base=str(tmp_path))
    assert acode == sc.FAIL and a["faculties"]["affect-coloring"]["verdict"] == "flip-caused", a


def test_iw2_score_B_F1_row_absent_is_undefined_and_wrong_intact_is_a_regression(tmp_path):
    sc = importlib.import_module("tools.iw2_score")
    _std(tmp_path, "base", "affect-coloring", "emo", "warm", "flat")
    _std(tmp_path, "flip", "affect-coloring", "emo", "warm", "flat")
    res, code = sc.score_B("base", PIN, "flip", PIN, ["false-belief-chat"], base=str(tmp_path))
    assert code == sc.UNDEF and any("absent" in u for u in res["undefined"]), res
    _fill(tmp_path, tag="flip", env={}, arm_env={})
    _write_cell(str(tmp_path), "flip", 43, "tom-false-belief", "tom_fb", {"ia": FB_LESION, "le": _ABSTAIN},
                arm_env={})
    res, code = sc.score_B("base", PIN, "flip", PIN, ["false-belief-chat"], base=str(tmp_path))
    assert code == sc.FAIL and [r["level"] for r in res["regressions"]] == ["F1"], res


def test_iw2_score_B_reference_undefined_cell_is_reported_not_scored(tmp_path):
    sc = importlib.import_module("tools.iw2_score")
    _std(tmp_path, "base", "affect-coloring", "emo", "warm", "flat")
    _std(tmp_path, "flip", "affect-coloring", "emo", "warm", "flat")
    _write_cell(str(tmp_path), "base", 44, "affect-coloring", "emo",
                {"ia": _r("warm"), "ib": _r("cold"), "le": _r("flat")})            # noisy null in the reference
    res, code = sc.score_B("base", PIN, "flip", PIN, [], base=str(tmp_path))
    assert code == sc.PASS and res.get("reference_undefined_faculties"), res


def test_iw2_score_A_never_passes_a_B_with_undefined_cells(tmp_path):
    sc = importlib.import_module("tools.iw2_score")
    _std(tmp_path, "base", "affect-coloring", "emo", "warm", "flat")
    _std(tmp_path, "flip", "affect-coloring", "emo", "warm", "flat")
    _std(tmp_path, "base", "curiosity-followup", "cur", "q", "x")
    _std(tmp_path, "flip", "curiosity-followup", "cur", "q", "x")
    _std(tmp_path, "alloff", "affect-coloring", "emo", "warm", "flat")
    _std(tmp_path, "flip", "affect-coloring", "emo", "warm", "warm", seeds=[101], verdict="pass")
    _std(tmp_path, "alloff", "affect-coloring", "emo", "warm", "warm", seeds=[101], verdict="pass")
    _write_cell(str(tmp_path), "flip", 44, "curiosity-followup", "cur", {"ia": _r("q"), "ib": _r("r"), "le": _r("x")})
    res, code = sc.score_B("base", PIN, "flip", PIN, [], base=str(tmp_path))
    assert code == sc.FAIL and res["undefined"], res
    a, acode = sc.score_A(res, "alloff", PIN, base=str(tmp_path))
    assert a["faculties"]["affect-coloring"]["verdict"] == "excused", a
    assert acode == sc.UNDEF and a["verdict"] == "UNDEFINED", a


# ── the job lines and the HANDOFF queue reasons ────────────────────────────────────────────────────────────────
POOL_MAX_DISPATCHABLE_GB = 11     # an idle 15 GB mini-PC node's budget, min(MemAvailable - 3, MemTotal - 2)
MAX_AGE_CAP_H = 240               # tools/pool_autodispatch.sh POOL_JOB_MAX_AGE_CAP_H


def _mk(*args, env_extra=None, inp=None):
    mk = os.path.join(REPO, "research", "packets", "integration-wave2", "make_jobs.sh")
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run(["bash", mk] + list(args), cwd=REPO, env=env, capture_output=True, text=True, input=inp)


def _shape_ok(ln):
    sh = subprocess.run(["bash", os.path.join(REPO, "tools", "queue_job_shape_check.sh"), ln], cwd=REPO,
                        capture_output=True, text=True)
    return sh.returncode == 0, sh.stderr


def test_make_jobs_identity_lines_are_memory_gated_and_dispatchable(tmp_path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    r = _mk("identity", head, "default", "off", env_extra={"IW2_JOBS_OUTD": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    lines = (tmp_path / ("identity_%s.txt" % head[:7])).read_text().splitlines()
    assert len(lines) == 2
    for ln in lines:
        m = re.search(r" && bash tools/mem_ok\.sh (\d+) \d+ && bash tools/memcap\.sh (\d+) -- env SIM_BACKEND=numpy ", ln)
        assert m, ln
        assert int(m.group(1)) <= POOL_MAX_DISPATCHABLE_GB, ln
        assert "--seed 7 " in ln and "iw2_default_identity" in ln
        ok, err = _shape_ok(ln)
        assert ok, (ln, err)


def test_make_jobs_lb_wrappers_add_the_node_preflight_and_the_memory_gate():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    sample = ("cd ~/derisk-pool/revisions/%s && mkdir -p x && env SIM_BACKEND=numpy .venv/bin/python -u -m "
              "research.runners.load_bearing_fraction --only neural-selfid --seed 42 --out x/lb.json" % head)
    src = open(os.path.join(REPO, "research", "packets", "integration-wave2", "make_jobs.sh")).read()
    assert src.count("| nodeguard | capwrap") == 5, "every LB job file (R x3, flip, alloff/bisect) is node-guarded"
    w = _mk("capwrap", "8", "12", inp=sample + "\n")
    assert w.returncode == 0, w.stderr
    ln = w.stdout.strip()
    ln = re.sub(r"^cd (\S+) && ", r"cd \1 && .venv/bin/python tools/assert_iw2_node.py && ", ln)   # what nodeguard adds
    assert re.search(r" && bash tools/mem_ok\.sh 8 2 && bash tools/memcap\.sh 12 -- env SIM_BACKEND=numpy ", ln), ln
    ok, err = _shape_ok(ln)
    assert ok, (ln, err)


def _handoff_reasons():
    k = os.path.join(REPO, "research", "packets", "integration-wave2")
    reasons = []
    for root, _dirs, files in os.walk(k):
        for fn in files:
            if fn == "HANDOFF.md":
                reasons += re.findall(r'R="([^"]*)"', open(os.path.join(root, fn), encoding="utf-8").read())
    return reasons


def test_handoff_queue_reasons_resolve_and_are_dispatchable():
    """tools/pool_queue.sh's GATE-SEED GUARD extracts `prereg[[:space:]]+[^[:space:]]+` and checks it with git cat-file:
    a path glued to ';' is refused (the wave-1 defect). Every reason must also declare a mem_gb the dispatcher can hand
    a 15 GB node (<= 11) and a max_age_h (<= 240) so the dispatcher does not skip it as debris after 12 h."""
    reasons = _handoff_reasons()
    assert len(reasons) >= 6, reasons
    for r in reasons:
        m = re.search(r"prereg[ \t]+(\S+)", r)
        assert m, r
        assert os.path.isfile(os.path.join(REPO, m.group(1))), (r, m.group(1))
        assert m.group(1).endswith("PREREGISTRATION.md"), r
        gb = re.search(r"mem_gb=(\d+)", r)
        assert gb and int(gb.group(1)) <= POOL_MAX_DISPATCHABLE_GB, r
        age = re.search(r"max_age_h=(\d+)", r)
        assert age and 12 < int(age.group(1)) <= MAX_AGE_CAP_H, r


def test_identity_declared_mem_matches_make_jobs():
    """The HANDOFF's identity reasons declare the same mem_gb as make_jobs.sh's IDENT_NEED (the measured need)."""
    src = open(os.path.join(REPO, "research", "packets", "integration-wave2", "make_jobs.sh")).read()
    need = int(re.search(r"^IDENT_NEED=(\d+)", src, re.M).group(1))
    ident = [r for r in _handoff_reasons() if "step D" in r or "step F" in r]
    assert ident, "no identity reasons found"
    for r in ident:
        assert int(re.search(r"mem_gb=(\d+)", r).group(1)) == need, r


def test_flip_passes_the_real_production_integration_gate(tmp_path, monkeypatch):
    """Run the REAL PI gate checks (A anchors, C ratchet, D retire status) on a copy of the repo files after flipping all
    three packets, so a gate/script disagreement shows up here, not at the local model's section-6 commit."""
    import yaml
    flip = importlib.import_module("tools.integration_wave2_flip")
    pi = importlib.import_module("tools.gates.production_integration")
    led = yaml.safe_load(open(os.path.join(REPO, flip.LEDGER)))
    files = set(flip.files_touched())
    for r in led["rows"]:
        for a in r.get("default_anchor") or []:
            files.add(a["file"])
        if r.get("scaffold_symbol"):
            files.add(r["scaffold_symbol"]["file"])
    for rel in files:
        if os.path.exists(os.path.join(REPO, rel)):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.path.join(REPO, rel), tmp_path / rel)
    on = [f for f, st in flip.status(str(tmp_path)).items() if st == "on"]
    if on:
        flip.revert(str(tmp_path), on)
    monkeypatch.setattr(pi, "_ROOT", str(tmp_path))
    before = pi._load_ledger((tmp_path / flip.LEDGER).read_text())
    base_a, base_c, base_d = pi._check_anchors(before), pi._check_ratchet(before), pi._check_retire_status(before)
    flip.apply(str(tmp_path), list(flip.PACKETS), date="2026-10-01")
    after = pi._load_ledger((tmp_path / flip.LEDGER).read_text())
    assert pi._check_anchors(after) == base_a, pi._check_anchors(after)
    assert pi._check_ratchet(after) == base_c
    assert pi._check_retire_status(after) == base_d
