# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
"""brain.eval.life on the stub cortex: 2 days of 3 turns, a fake night (no model weights)."""
import hashlib
import json
import os

import pytest

import brain.brain as bb
import brain.eval.life as life
from brain import clock
from brain.brain import Brain
from brain.eval import life_bank as LB
from brain.eval import life_table
from brain.eval.weekly import SESSION_1
from brain.tests.test_brain_key_switch import KW, StubCortex

BKW = {k: v for k, v in KW.items() if k != "sleep_on_restart"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    nights = []

    def fake_replay(plastic, base, eps, generic, **k):
        nights.append(len(eps))
        return []
    monkeypatch.setattr(bb, "replay_into_cortex", fake_replay)
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: [], "B_k": {}, "B_v": {}})())
    real = life.build_life

    def short(seed, days, split="dev"):
        sched, ledger = real(seed, days, split)
        for d in sched:
            d["turns"] = d["turns"][:3]     # 3 turns a day keeps the stub life fast
        return sched, ledger
    monkeypatch.setattr(life, "build_life", short)
    yield nights
    clock.set(None)


def _run(tmp_path, name, days, **kw):
    return life.run(StubCortex(), 7, days, "dev", str(tmp_path / name / "h"), str(tmp_path / name / "o" / "life_s7"),
                    kw.pop("full", {1}), brain_kw=BKW, log=lambda *_: None, **kw)


def _hash(root):
    out = {}
    for d, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f != ".lock":
                with open(os.path.join(d, f), "rb") as fh:
                    out[os.path.relpath(os.path.join(d, f), root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _replies(res, last_nosleep_night=2):
    """Lived and probed replies, ordered by (line, day, arm); a nosleep probe exists only at the horizon's last night."""
    lived = sorted((x["line"], x["day"], r["you"], r["brain"], str(r["recalled"])) for x in res["lived"] for r in x["rows"])
    probes = sorted((p["line"], p["night"], p["arm"], r["brain"], str(r["recalled"])) for p in res["probes"] for r in p["rows"]
                    if p["line"] == "trunk" or p["night"] == last_nosleep_night)
    return lived, probes


def test_day_one_is_weekly_session_one_and_build_is_deterministic():
    s1, l1 = life.build_life(310, 3)
    assert [t["text"] for t in s1[0]["turns"]] == SESSION_1[:3]
    stats = LB.split_stats()
    assert 75 <= stats["items"] <= 100


def test_full_build_deterministic_and_variants_opposite():
    import importlib
    m = importlib.reload(life)      # un-patched build_life
    try:
        s1, l1 = m.build_life(310, 7)
        s2, l2 = m.build_life(310, 7)
        assert s1 == s2 and l1 == l2
        assert [t["text"] for t in s1[0]["turns"]] == SESSION_1
        assert m.build_life(311, 7)[0] != s1
        inc = {k: v["variant"] for k, v in l1.items() if k.startswith("inc:")}
        sys_v = {v for k, v in inc.items() if "sysadmin" in k}
        home_v = {v for k, v in inc.items() if "home" in k}
        assert len(sys_v) == len(home_v) == 1 and sys_v != home_v
        assert m.build_life(310, 3)[0] == s1[:3]        # day d does not depend on the horizon
    finally:
        importlib.reload(life)


def test_two_days_night_counter_clock_and_probes(tmp_path, _env):
    res = _run(tmp_path, "a", 2)
    home = str(tmp_path / "a" / "h" / "trunk")
    with open(os.path.join(home, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["sleeps"] == 2 == len([n for n in res["nights"] if n["line"] == "trunk" and n["slept"]])
    assert len(_env) == 2                                    # the night replayed twice
    assert meta["last_sleep"] == pytest.approx(life.day_start(2) + 14 * 3600, abs=1)
    with open(os.path.join(home, "episodes.json")) as fh:
        idx = json.load(fh)
    assert all(r["created"] <= meta["last_sleep"] for r in idx)           # nothing pending after the night
    day2 = [r for r in idx if r["created"] >= life.day_start(2)]
    assert day2 and all(r["created"] < life.day_start(2) + 14 * 3600 for r in day2)
    arms = {(p["night"], p["arm"]) for p in res["probes"]}
    assert arms == {(1, a) for a in life.ARMS} | {(2, "intact"), (2, "hippocampus-off")}   # full nights {1}
    p1 = next(p for p in res["probes"] if p["night"] == 2 and p["arm"] == "intact")
    kinds = [r["kind"] for r in p1["rows"]]
    assert kinds.count("H") == 6 and kinds.count("never") == 3 and kinds.count("neardup") == 2
    assert sum(k.startswith("open-") for k in kinds) == 6
    off = next(p for p in res["probes"] if p["night"] == 2 and p["arm"] == "hippocampus-off")
    assert all(r["recalled"] is None for r in off["rows"])             # no hippocampus, no recall
    assert os.path.exists(str(tmp_path / "a" / "o" / "life_s7.md")) and "morning after night 2" in open(str(tmp_path / "a" / "o" / "life_s7.md")).read()
    life_table.render([res])                                           # the table reads it


def test_probes_leave_the_trunk_byte_identical(tmp_path):
    res = _run(tmp_path, "b", 2)
    home = str(tmp_path / "b" / "h" / "trunk")
    before = _hash(home)
    sim = life.SimClock(life.day_start(3))
    clock.set(sim.now)
    ledger = res["ledger"]
    out = life.probe(home, StubCortex(), 2, tuple(life.ARMS), life.battery(7, 2, ledger), ledger, sim, 7, BKW,
                     str(tmp_path / "b" / "w"))
    assert len(out) == 4 and _hash(home) == before
    assert not os.path.exists(str(tmp_path / "b" / "w" / "probe_L2_intact"))      # the copy is deleted after use


def test_resume_equals_non_resume(tmp_path):
    whole = _run(tmp_path, "w", 2, nosleep_arm=True)
    # a life of 1 day, then resumed to 2
    _run(tmp_path, "r", 1, nosleep_arm=True)
    resumed = _run(tmp_path, "r", 2, nosleep_arm=True, resume=True)
    assert _replies(whole) == _replies(resumed)
    key = lambda r: sorted((n["line"], n["night"]) for n in r["nights"])  # noqa: E731
    assert key(resumed) == key(whole)


def test_resume_after_a_cut_between_save_and_night(tmp_path):
    whole = _run(tmp_path, "w2", 2)
    real = life.live_day
    state = {"cut": True}

    def cutting(home, cortex, day, ledger, sim, seed, kw, sleeps, on_lived=None, weekend=False):
        def hook(rows, live_s):
            on_lived(rows, live_s)
            if day["day"] == 2 and state["cut"]:
                state["cut"] = False
                raise RuntimeError("cut")
        return real(home, cortex, day, ledger, sim, seed, kw, sleeps, on_lived=hook, weekend=weekend)
    life.live_day = cutting
    try:
        with pytest.raises(RuntimeError):
            _run(tmp_path, "c", 2)
    finally:
        life.live_day = real
    res = _run(tmp_path, "c", 2, resume=True)
    with open(str(tmp_path / "c" / "h" / "trunk" / "meta.json")) as fh:
        assert json.load(fh)["sleeps"] == 2                       # the cut night was slept on resume
    assert _replies(whole) == _replies(res)


def _scrub(x):
    """A life result without timings, host, the record-only stack `provenance` (prereg test2 T1) and the Brain-kwarg echo
    (the only places an explicit-off flag may show)."""
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items() if k not in ("seconds", "live_seconds", "host", "brain_kw", "provenance")}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x


def test_every_new_flag_explicitly_off_equals_flags_absent(tmp_path):
    """Prereg 2026-11-19 section 4 test 13 on the stub cortex (the 0.1B version is run by hand)."""
    a = _run(tmp_path, "absent", 2)
    b = life.run(StubCortex(), 7, 2, "dev", str(tmp_path / "off" / "h"), str(tmp_path / "off" / "o" / "life_s7"), {1},
                 brain_kw={**BKW, "use_gate": None, "sleep_downscale": None}, log=lambda *_: None, frozen_probe=False,
                 pruned_a=False, cue_ladder=False, interview=False, lang_guard=False)
    assert json.dumps(_scrub(a), sort_keys=True) == json.dumps(_scrub(b), sort_keys=True)
    assert "frozen_probe" not in a and all("fetch" not in r for p in a["probes"] for r in p["rows"])
    assert all("max_strength_open" not in p for p in a["probes"])


def test_use_gate_and_frozen_probe_life(tmp_path):
    res = life.run(StubCortex(), 7, 2, "dev", str(tmp_path / "on" / "h"), str(tmp_path / "on" / "o" / "life_s7"), {1},
                   brain_kw={**BKW, "use_gate": "raw_win"}, log=lambda *_: None, frozen_probe=True)
    arms = {(p["night"], p["arm"]) for p in res["probes"]}
    assert (1, "intact-frozen") in arms and (2, "intact-frozen") in arms
    assert all("fetch" in r for p in res["probes"] for r in p["rows"])
    assert all("max_strength_open" in p and "factor_norms" in p for p in res["probes"])
    fz = next(p for p in res["probes"] if p["arm"] == "intact-frozen" and p["night"] == 2)
    assert fz["summary"]["H_n"] == 6
    assert any("fetch" in r for x in res["lived"] for r in x["rows"])
    text = life_table.render([res])
    assert "M-use / M-scale read-outs" in text and "boosts granted/denied" in text


def test_provenance_recorded_and_record_only(tmp_path, monkeypatch):
    """Prereg test2 Amendment T1 device rule: life.py records the stack at life start (fields only); the replies of a
    life are the same whatever the env-derived fields say."""
    import torch
    monkeypatch.setenv("AWS_AMI_ID", "ami-test")
    monkeypatch.setenv("BRAIN_CODE_COMMIT", "fallback-commit")
    a = _run(tmp_path, "pa", 1)
    p = a["provenance"]
    assert set(p) == {"cuda_precision", "torch", "torch_cuda", "gpu_name", "tf32_matmul", "tf32_cudnn", "code_commit", "aws_ami_id",
                      "frozen_sha256", "brain_tree_dirty", "brain_code_clean_env"}
    assert set(p["frozen_sha256"]) <= set(life.FROZEN_FILES) and "brain/eval/life.py" in p["frozen_sha256"]
    assert p["frozen_sha256"]["brain/eval/life_read_spec.py"] == open(
        os.path.join(life.REPO_ROOT, "brain/eval/life_read_spec.sha256")).read().split()[0]
    assert p["brain_tree_dirty"] in (True, False, None)
    assert p["torch"] == torch.__version__ and p["torch_cuda"] == torch.version.cuda and p["gpu_name"] is None   # cpu life
    assert p["tf32_matmul"] == torch.backends.cuda.matmul.allow_tf32 and p["aws_ami_id"] == "ami-test" and p["code_commit"]
    with open(str(tmp_path / "pa" / "o" / "life_s7") + ".json") as fh:
        assert json.load(fh)["provenance"] == p
    monkeypatch.delenv("AWS_AMI_ID")
    b = _run(tmp_path, "pb", 1)
    assert b["provenance"]["aws_ami_id"] is None and _replies(a) == _replies(b)


def test_registered_split_refuses_dirty_tree_or_no_git(tmp_path, monkeypatch):
    """Prereg test2 section 3: --split test2 refuses to START from a dirty brain/ tree, or without git unless
    BRAIN_CODE_COMMIT and BRAIN_CODE_CLEAN=1 (AWS git-archive launch). Nothing is written before the refusal."""
    monkeypatch.delenv("BRAIN_CODE_COMMIT", raising=False)
    monkeypatch.delenv("BRAIN_CODE_CLEAN", raising=False)
    monkeypatch.setattr(life, "_git_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="dirty"):
        life.run(StubCortex(), 7, 1, "test2", str(tmp_path / "r" / "h"), str(tmp_path / "r" / "o" / "life_s7"), {1},
                 brain_kw=BKW, log=lambda *_: None)
    assert not (tmp_path / "r").exists()
    monkeypatch.setenv("BRAIN_CODE_COMMIT", "abc")
    monkeypatch.setenv("BRAIN_CODE_CLEAN", "1")
    with pytest.raises(RuntimeError, match="dirty"):        # the override never covers a tree git sees as dirty
        life.registered_guard("test2", {"brain_tree_dirty": True})
    life.registered_guard("test2", {"brain_tree_dirty": None})   # no git + both env: allowed
    monkeypatch.delenv("BRAIN_CODE_CLEAN")
    with pytest.raises(RuntimeError, match="git unavailable"):
        life.registered_guard("test2", {"brain_tree_dirty": None})
    monkeypatch.setenv("BRAIN_CODE_CLEAN", "1")
    monkeypatch.delenv("BRAIN_CODE_COMMIT")
    with pytest.raises(RuntimeError, match="git unavailable"):
        life.registered_guard("test2", {"brain_tree_dirty": None})
    life.registered_guard("test2", {"brain_tree_dirty": False})  # clean tree: allowed
    for split in ("dev", "test"):                                 # unregistered splits: unchanged
        life.registered_guard(split, {"brain_tree_dirty": True})
        life.registered_guard(split, {"brain_tree_dirty": None})


def test_dev_split_runs_from_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(life, "_git_dirty", lambda: True)
    res = _run(tmp_path, "d", 1)
    assert res["provenance"]["brain_tree_dirty"] is True


def test_registered_guard_also_runs_on_resume(tmp_path, monkeypatch):
    """Prereg test2 section 3: the dirty-tree guard covers --resume too (the arg split or the saved split is test2);
    the refusal writes nothing. A dev resume from a dirty tree is unchanged."""
    monkeypatch.delenv("BRAIN_CODE_COMMIT", raising=False)
    monkeypatch.delenv("BRAIN_CODE_CLEAN", raising=False)
    _run(tmp_path, "g", 1)
    o = str(tmp_path / "g" / "o" / "life_s7") + ".json"
    with open(o) as fh:
        saved = json.load(fh)
    saved["split"] = "test2"                     # a registered life, as saved
    with open(o, "w") as fh:
        json.dump(saved, fh)
    before = _hash(str(tmp_path / "g"))
    monkeypatch.setattr(life, "_git_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="dirty"):
        _run(tmp_path, "g", 2, resume=True)
    monkeypatch.setattr(life, "_git_dirty", lambda: None)   # git-archive tree without the env pair
    with pytest.raises(RuntimeError, match="git unavailable"):
        _run(tmp_path, "g", 2, resume=True)
    assert _hash(str(tmp_path / "g")) == before              # nothing written by either refusal
    monkeypatch.setattr(life, "_git_dirty", lambda: True)
    _run(tmp_path, "d", 1)
    res = _run(tmp_path, "d", 2, resume=True)                # dev resume from a dirty tree: allowed, as before
    assert len(res["nights"]) == 2
