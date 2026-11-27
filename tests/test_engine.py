"""tools/engine: chaining, retry, resume, pick/verdict invocation, dirty refusal, dry-run, on a fake local node; plus the SHY rules."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.engine import core
from tools.engine.core import Engine, make_nodes

FAKE = '''
import argparse, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); ap.add_argument("--v", type=float, default=0)
ap.add_argument("--fail-once"); ap.add_argument("--fail-always", action="store_true"); ap.add_argument("--out")
ap.add_argument("--wait-for"); ap.add_argument("--garbage", action="store_true")
a = ap.parse_args()
open(os.environ["ENGINE_TEST_LOG"], "a").write(f"{a.seed} {a.v}\\n")
import time
while a.wait_for and not os.path.exists(a.wait_for): time.sleep(0.05)
if a.fail_always: sys.exit(3)
if a.garbage:
    open(a.out, "w").write("not json"); sys.exit(0)
if a.fail_once and not os.path.exists(a.fail_once):
    open(a.fail_once, "w").close(); sys.exit(2)
json.dump({"seed": a.seed, "v": a.v}, open(a.out, "w"))
'''
MOD = '''
import json, os
from pathlib import Path
def _log(ctx, s):
    Path(ctx["res"]).mkdir(parents=True, exist_ok=True); open(Path(ctx["res"]) / "calls.txt", "a").write(s + "\\n")
def pick(dev, ctx):
    _log(ctx, "pick " + ",".join(sorted(dev)))
    if os.path.exists(os.environ["ENGINE_CRASH"]): raise RuntimeError("crash")
    best = max(dev.values(), key=lambda r: r["v"])
    return None if best["v"] < 0 else {"params": f"--v {best['v']}"}
def verdict(six, pick, ctx):
    _log(ctx, "verdict " + ",".join(map(str, sorted(six))))
    m = sum(r["v"] for r in six.values()) / len(six)
    return {"verdict": "GO" if m > 3 else "NEGATIVE", "summary": f"mean {m}"}
'''


def spec_yaml(name, grid, nxt=None, mx=3, seeds=(1, 2), extra="", question="q1"):
    g = "".join(f'  {k}: "{v}"\n' for k, v in grid.items())
    on = "".join(f"    {k}: {v}\n" for k, v in (nxt or {}).items())
    return (f"name: {name}\nquestion: {question}\nprereg: p.md\nmodule: m.py\n{'' if extra.startswith('six_params') else 'pick: pick' + chr(10)}verdict: verdict\n"
            f'cmd: \'"$PY" brain/fake.py --seed {{seed}} {{params}} --out {{out}}\'\ndev_seed: 7\n'
            f"dev_grid:\n{g or '  {}'}\nseeds: {list(seeds)}\nmemory_gb: 1\nladder:\n  max_len: {mx}\n  next:\n{on or '    {}'}\n{extra}")


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "brain").mkdir(parents=True)
    (repo / "brain/fake.py").write_text(FAKE)
    sp = repo / "tools/engine/specs"
    sp.mkdir(parents=True)
    (sp / "m.py").write_text(MOD)
    monkeypatch.setenv("ENGINE_TEST_LOG", str(tmp_path / "invocations.txt"))
    monkeypatch.setenv("ENGINE_CRASH", str(tmp_path / "crash"))

    def add(name, **kw):
        (sp / f"{name}.yaml").write_text(spec_yaml(name, **kw))

    def commit():
        for c in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
            subprocess.run(["git", "-C", str(repo), *c], check=True, capture_output=True)

    def engine(name, nodes=2, poll=0.02, **kw):
        return Engine(name, make_nodes("local", [f"n{i}" for i in range(nodes)], tmp_path / "nodes"), repo=repo, poll=poll,
                      log=lambda *_: None, **kw)

    def calls():
        p = tmp_path / "invocations.txt"
        return p.read_text().split("\n")[:-1] if p.exists() else []

    return type("E", (), dict(repo=repo, add=add, commit=commit, engine=engine, calls=calls, tmp=tmp_path))


def test_chain_dev_pick_six_verdict_next_and_digest(env):
    env.add("a", grid={"lo": "--v 1", "hi": "--v 5"}, nxt={"GO": "b"})
    env.add("b", grid={}, nxt={"NEGATIVE": "c"}, extra="six_params: '--v 1'\n")
    env.add("c", grid={}, extra="six_params: '--v 9'\n")
    env.commit()
    assert env.engine("a").run() == "GO"                  # a GO -> b NEGATIVE (mean 1) -> c GO (mean 9), end of chain
    res = env.repo / core.RES
    va = json.loads((res / "a/verdict.json").read_text())
    assert va["verdict"] == "GO" and va["pick"]["params"] == "--v 5.0" and len(va["commit"]) == 40
    assert (res / "a/dev/hi.json").exists() and (res / "a/six/s2.json").exists()
    assert (res / "a/calls.txt").read_text().split("\n")[:-1] == ["pick hi,lo", "verdict 1,2"]   # pick and verdict each invoked once
    dg = (res / "ENGINE_DIGEST.md").read_text().splitlines()
    assert "variant 1 of max 3 (a) | GO" in dg[2] and "next: b" in dg[2] and "variant 2 of max 3 (b) | NEGATIVE" in dg[3]
    assert "variant 3 of max 3 (c) | GO" in dg[4] and "next: none" in dg[4]
    st = json.loads(next(res.glob("ENGINE_STATE_*.json")).read_text())
    assert st["variants"]["q1"] == ["a", "b", "c"]


def test_ladder_stops_at_max_len(env):
    env.add("a", grid={"x": "--v 1"}, nxt={"NEGATIVE": "b"}, mx=1)
    env.add("b", grid={}, extra="six_params: '--v 9'\n")
    env.commit()
    env.engine("a").run()
    assert "ladder limit 1 reached" in (env.repo / core.RES / "ENGINE_DIGEST.md").read_text()
    assert not (env.repo / core.RES / "b").exists()


def test_negative_dev_skips_six_seeds(env):
    env.add("a", grid={"x": "--v -1"})
    env.commit()
    assert env.engine("a").run() == "NEGATIVE-DEV"
    assert not (env.repo / core.RES / "a/six").exists() and len(env.calls()) == 1


def test_retry_once_then_succeeds_on_second_attempt(env):
    env.add("a", grid={"x": f"--v 5 --fail-once {env.tmp}/marker"})
    env.commit()
    assert env.engine("a").run() == "GO"
    assert env.calls().count("7 5.0") == 2                    # failed once, retried once
    st = json.loads(next((env.repo / core.RES).glob("ENGINE_STATE_*.json")).read_text())
    assert st["specs"]["a"]["jobs"]["dev/x"]["attempts"] == 2 and st["specs"]["a"]["jobs"]["dev/x"]["status"] == "done"


def test_failing_twice_gives_error_and_stops(env):
    env.add("a", grid={"x": "--fail-always"}, nxt={"ERROR": "b"})
    env.add("b", grid={}, extra="six_params: '--v 9'\n")
    env.commit()
    assert env.engine("a").run() == "ERROR"
    assert len(env.calls()) == 2 and not (env.repo / core.RES / "b").exists()


def test_resume_after_crash_does_not_redo_finished_jobs(env):
    env.add("a", grid={"lo": "--v 1", "hi": "--v 5"})
    env.commit()
    (env.tmp / "crash").write_text("")
    with pytest.raises(RuntimeError):
        env.engine("a").run()                                  # dev done, pick crashed
    assert len(env.calls()) == 2
    (env.tmp / "crash").unlink()
    assert env.engine("a").run() == "GO"                       # a fresh Engine = a rebooted driver
    assert len(env.calls()) == 2 + 2                           # only the two six-seed jobs ran; dev not redone
    again = env.engine("a").run()                              # finished chain: nothing runs
    assert again == "GO" and len(env.calls()) == 4


def test_dirty_brain_refused_and_dry_run_prints_plan(env):
    env.add("a", grid={"x": "--v 1"}, nxt={"GO": "b"})
    env.commit()
    (env.repo / "brain/new.py").write_text("x=1")
    with pytest.raises(SystemExit, match="dirty"):
        env.engine("a").run()
    out = []
    Engine("a", make_nodes("local", ["n0"], env.tmp / "nodes"), repo=env.repo, dry=True, log=out.append).run()
    assert any("dev   x" in o and "--v 1" in o for o in out) and not list((env.repo / core.RES).glob("ENGINE_STATE*"))


def test_node_must_fit_memory(env):
    env.add("a", grid={"x": "--v 1"}, extra="hosts: [nope]\n")
    env.commit()
    with pytest.raises(SystemExit, match="no node fits"):
        env.engine("a").run()


def _state(env):
    return json.loads(next((env.repo / core.RES).glob("ENGINE_STATE_*.json")).read_text())


def _prep(e):
    """Engine ready to call launch() directly (state, tree, spec, one pending job)."""
    e.st = e.load()
    e.st["commit"], e.st["tree"] = e.commit()
    return e.root, {"key": "dev/x", "seed": 7, "params": "--v 5", "status": "pending", "attempts": 0, "node": None, "jd": None}


def test_resume_survives_unrelated_and_results_commits_but_not_brain_change(env):
    env.add("a", grid={"lo": "--v 1", "hi": "--v 5"})
    env.commit()
    (env.tmp / "crash").write_text("")
    with pytest.raises(RuntimeError):
        env.engine("a").run()
    st0 = _state(env)
    assert len(st0["commit"]) == 40 and len(st0["tree"]) == 40
    (env.tmp / "crash").unlink()
    (env.repo / "README").write_text("unrelated")
    (env.repo / "brain/experiments/results/a").mkdir(parents=True, exist_ok=True)
    git = lambda *c: subprocess.run(["git", "-C", str(env.repo), "-c", "user.name=t", "-c", "user.email=t@t", *c], check=True, capture_output=True)
    git("add", "-A", "-f"); git("commit", "-qm", "unrelated + results")          # HEAD moves, brain/ code does not
    assert env.engine("a").run() == "GO" and len(env.calls()) == 4
    assert _state(env)["tree"] == st0["tree"]
    (env.repo / "brain/fake.py").write_text(FAKE + "\n# changed")
    git("commit", "-qam", "brain change")
    with pytest.raises(SystemExit, match="use --reset"):
        env.engine("a").run()


def test_crash_during_launch_does_not_duplicate_job(env):
    env.add("a", grid={"x": f"--v 5 --wait-for {env.tmp}/go"})
    env.commit()
    e1 = env.engine("a", nodes=1)
    real = e1.nodes[0].sh
    def crash_after_launch(script, timeout=900):
        r = real(script, timeout)
        if "nohup" in script:
            raise RuntimeError("driver crashed after the job started")
        return r
    e1.nodes[0].sh = crash_after_launch
    with pytest.raises(RuntimeError):
        e1.run()
    assert _state(env)["specs"]["a"]["jobs"]["dev/x"]["status"] == "launching"
    (env.tmp / "go").write_text("")
    assert env.engine("a", nodes=1).run() == "GO"
    assert env.calls().count("7 5.0") == 1                      # reattached, not relaunched


def test_job_script_exits_if_pid_alive(env):
    env.add("a", grid={})
    env.commit()
    e = env.engine("a", nodes=1)
    spec, job = _prep(e)
    job["params"] = f"--v 5 --wait-for {env.tmp}/go"
    assert e.launch(e.nodes[0], spec, job)
    jd = job["jd"]
    job["attempts"] = 0                                         # same jd again, as after a crash that lost the state
    for _ in range(100):
        if (env.tmp / "invocations.txt").exists():
            break
        time.sleep(0.05)
    assert e.launch(e.nodes[0], spec, job)
    time.sleep(0.5)
    assert job["jd"] == jd and env.calls() == ["7 5.0"]
    (env.tmp / "go").write_text("")


def test_killed_job_is_dead_then_retried(env):
    env.add("a", grid={"x": f"--v 5 --wait-for {env.tmp}/go"})
    env.commit()
    e, out = env.engine("a", nodes=1), []
    t = threading.Thread(target=lambda: out.append(e.run()))
    t.start()
    pidf = None
    for _ in range(200):
        pidf = next((env.tmp / "nodes").glob("n0/brain-lane/engine/jobs/*/*_a0/pid"), None)
        if pidf and pidf.read_text().strip():
            break
        time.sleep(0.05)
    os.killpg(int(pidf.read_text()), signal.SIGKILL)            # no rc is ever written: status must say dead
    time.sleep(0.3)
    (env.tmp / "go").write_text("")
    t.join(30)
    assert out == ["GO"] and env.calls().count("7 5.0") == 2
    assert _state(env)["specs"]["a"]["jobs"]["dev/x"]["attempts"] == 2


def test_job_alive_across_engine_restart_is_reattached(env):
    env.add("a", grid={"x": f"--v 5 --wait-for {env.tmp}/go"})
    env.commit()
    e1 = env.engine("a", nodes=1)
    def boom(*a):
        raise RuntimeError("driver died")
    e1.status = boom
    with pytest.raises(RuntimeError):
        e1.run()
    assert _state(env)["specs"]["a"]["jobs"]["dev/x"]["status"] == "running"
    (env.tmp / "go").write_text("")
    assert env.engine("a", nodes=1).run() == "GO"
    assert env.calls().count("7 5.0") == 1


def test_ssh_failures_mark_dead_after_20(env):
    env.add("a", grid={})
    env.commit()
    e = env.engine("a", nodes=1)
    logs = []
    e.log = logs.append
    n = e.nodes[0]
    n.sh = lambda *a, **k: (255, "")
    job = {"key": "dev/x", "jd": "j", "t0": 0}
    assert [e.status(n, job) for _ in range(19)] == ["running"] * 19
    assert e.status(n, job) == "dead" and any("ssh failures" in x for x in logs)


def test_pull_failure_retries_without_consuming_attempt_but_parse_failure_does(env):
    env.add("a", grid={"x": "--v 5"})
    env.commit()
    e = env.engine("a", nodes=1)
    real, n = e.nodes[0].get, [0]
    def flaky(remote, local):
        n[0] += 1
        if n[0] <= 4:                                           # 3 tries of round one, 1 of round two
            raise OSError("rsync blip")
        return real(remote, local)
    e.nodes[0].get = flaky
    assert e.run() == "GO"
    assert env.calls().count("7 5.0") == 1 and _state(env)["specs"]["a"]["jobs"]["dev/x"]["attempts"] == 1
    env.add("g", grid={"x": "--v 5 --garbage"})
    env.commit()
    assert env.engine("g").run() == "ERROR"
    assert env.calls().count("7 5.0") == 1 + 2                  # parse failure consumed both attempts


def test_stale_results_never_reach_pick(env):
    env.add("a", grid={"lo": "--v 1", "hi": "--v 5"})
    env.commit()
    stale = env.repo / core.RES / "a/dev"
    stale.mkdir(parents=True)
    (stale / "stale.json").write_text(json.dumps({"v": 99}))
    assert env.engine("a").run() == "GO"
    assert (env.repo / core.RES / "a/calls.txt").read_text().split("\n")[0] == "pick hi,lo"


def test_never_runs_on_workstation_and_w3090_rules(env, monkeypatch):
    monkeypatch.setattr(core.socket, "gethostname", lambda: "wk-test")
    env.add("a", grid={"x": "--v 1"})
    env.commit()
    with pytest.raises(SystemExit, match="never runs jobs here"):
        Engine("a", [core.Node("wk-test", 23)], repo=env.repo, log=lambda *_: None)
    assert [n.name for n in make_nodes("ssh")] == ["pool40", "pool41", "pool42"]            # off by default
    nodes = make_nodes("ssh", allow_w3090=True)
    w = nodes[-1]
    assert (w.name, w.kind, w.local) == ("w3090", "w3090", True) and nodes[0].kind == "ssh"
    e = Engine("a", nodes, repo=env.repo, log=lambda *_: None)
    spec, job = _prep(e)
    sc = e.job_script(w, spec, job, "r", "jd")
    assert "BRAIN_DEVICE=cuda" in sc and "exit 96" not in sc
    assert "exit 96" in e.job_script(nodes[0], spec, job, "r", "jd") and "BRAIN_DEVICE" not in e.job_script(nodes[0], spec, job, "r", "jd")
    spec.cmd = "x {device}"
    assert spec.render(1, "", "o", "cuda") == "x cuda" and spec.render(1, "", "o") == "x cpu"
    for used, apps, ok in (("500", "", True), ("3000", "", False), ("500", "123", False)):
        monkeypatch.setattr(core.subprocess, "run", lambda a, **k: subprocess.CompletedProcess(
            a, 0, used if "memory.used" in a[1] else apps, ""))
        assert w.usable() is ok
    monkeypatch.setattr(core.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert w.usable() is False and nodes[0].usable() is True


def test_w3090_only_used_when_gpu_idle(env, monkeypatch):
    env.add("a", grid={"x": "--v 5"})
    env.commit()
    w = core.Node("w3090", 23, root=env.tmp / "nodes/w", py=sys.executable, memcap=False, kind="w3090")
    w._home = str(env.tmp / "nodes/w")
    e = Engine("a", [w], repo=env.repo, poll=0.02, log=lambda *_: None)
    idle = [False, False, True]
    monkeypatch.setattr(core, "gpu_idle", lambda: idle.pop(0) if idle else True)
    assert e.run() == "GO" and idle == []                       # waited out two busy checks before launching


# ---- the registered SHY rules
def _arm(recall, kl, norm=1.0, fr=1):
    return {"recall": recall, "generic_kl": kl, "adapter_norm": norm, "false_recall": fr}


def test_shy_pick_rule():
    from tools.engine.specs import local_sleep_shy as m   # noqa  (namespace import of the spec module)
    ctx = {"repo": str(core.REPO)}
    mk = lambda lr, arms: {"lr": lr, "feedback": "transport", "arms": arms}
    dev = {"t3": mk(0.3, {"local@g0.8": _arm(4, 0.09), "local@g0.4": _arm(3, 0.03), "local@g0.25": _arm(3, 0.02), "local@g1": _arm(4, 0.1)}),
           "t1": mk(0.1, {"local@g0.6": _arm(3, 0.035)})}
    p = m.pick(dev, ctx)
    assert (p["lr"], p["g"]) == (0.1, 0.6) and "--downscale-grid 1,0.6" in p["params"]   # recall 3 ties -> larger g; g=1 (recall 4) never
    assert m.pick({"t": mk(0.3, {"local@g0.8": _arm(1, 0.01)})}, ctx) is None             # recall under floor 2


def test_shy_verdict_clauses():
    from tools.engine.specs import local_sleep_shy as m
    def seed(loc, loc1, grad1, gradg, shuf=0, ns=0):
        return {"arms": {"local@g0.4": loc, "local@g1": loc1, "gradient@g1": grad1, "gradient@g0.4": gradg,
                         "local_shuffled@g0.4": _arm(shuf, 0.01), "no_sleep@g1": _arm(ns, 0.0)}}
    pk = {"g": 0.4}
    good = {s: seed(_arm(4, 0.03), _arm(4, 0.09), _arm(5, 0.032), _arm(4, 0.03)) for s in range(6)}
    assert m.verdict(good, pk, {})["verdict"] == "GO-SHY"
    plain = {s: seed(_arm(4, 0.03), _arm(4, 0.03), _arm(5, 0.032), _arm(4, 0.03)) for s in range(6)}
    assert m.verdict(plain, pk, {})["verdict"] == "GO-PLAIN-LOCAL"
    bad = {s: seed(_arm(1, 0.03), _arm(4, 0.09), _arm(5, 0.032), _arm(4, 0.03)) for s in range(6)}
    assert m.verdict(bad, pk, {})["verdict"] == "NEGATIVE"
    noref = {s: seed(_arm(4, 0.03), _arm(4, 0.09), _arm(1, 0.032), _arm(1, 0.03)) for s in range(6)}
    assert m.verdict(noref, pk, {})["verdict"] == "UNDEFINED"


@pytest.mark.parametrize("name,mut,verdict,failed", [
    ("a", dict(ns=3), "NEGATIVE", ["a"]),                                              # no_sleep too close to local
    ("b", dict(shuf_tie=True), "NEGATIVE", ["b"]),                                     # shuffled ties/beats local in 2 of 6 seeds
    ("c", dict(grad1_kl=0.02), "NEGATIVE", ["c"]),                                     # local KL > 1.25x gradient@1
    ("d", dict(loc_fr=3), "NEGATIVE", ["d"]),                                          # false recall above gradient + 0.5
    ("e2", dict(gradg_kl=0.02), "NEGATIVE", ["e"]),                                    # local KL > 1.25x gradient@g
    ("e1", dict(gradg_recall=10), "NEGATIVE", ["a", "e"]),                             # L < 0.5 Gg cannot fail without (a) also failing
    ("s", dict(shuf=2), "UNDEFINED", None),                                            # shuffled mean > 1.0
    ("f", dict(loc1_kl=0.03), "GO-PLAIN-LOCAL", ["f"]),
])
def test_shy_each_verdict_clause_fails_alone(name, mut, verdict, failed):
    from tools.engine.specs import local_sleep_shy as m
    def seed(i):
        loc = _arm(4, 0.03, fr=mut.get("loc_fr", 1))
        shuf = mut.get("shuf", 0)
        if mut.get("shuf_tie") and i < 2:
            loc, shuf = _arm(1, 0.03), 1                                               # tie is not a beat; mean recall stays >= 2
        return {"arms": {"local@g0.4": loc, "local@g1": _arm(4, mut.get("loc1_kl", 0.09)),
                         "gradient@g1": _arm(5, mut.get("grad1_kl", 0.032)),
                         "gradient@g0.4": _arm(mut.get("gradg_recall", 4), mut.get("gradg_kl", 0.03)),
                         "local_shuffled@g0.4": _arm(shuf, 0.01), "no_sleep@g1": _arm(mut.get("ns", 0), 0.0)}}
    v = m.verdict({i: seed(i) for i in range(6)}, {"g": 0.4}, {})
    assert v["verdict"] == verdict
    if failed is not None:
        assert [k for k, ok in v["clauses"].items() if not ok] == failed


def test_shy_pick_none_gives_negative_dev_through_engine(env):
    from tools.engine.specs import local_sleep_shy as m
    mk = lambda kl: {"lr": 0.1, "feedback": "transport", "arms": {"local@g0.6": _arm(3, kl)}}
    assert m.pick({"t": mk(10.0)}, {"repo": str(core.REPO)}) is None                  # drift above 1.25x pinned gradient KL
    env.add("a", grid={"x": "--v -1"})
    env.commit()
    env.engine("a").run()
    assert "NEGATIVE-DEV" in (env.repo / core.RES / "ENGINE_DIGEST.md").read_text()
