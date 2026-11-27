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
ap.add_argument("--wait-for"); ap.add_argument("--garbage", action="store_true"); ap.add_argument("--rc", type=int)
a = ap.parse_args()
open(os.environ["ENGINE_TEST_LOG"], "a").write(f"{a.seed} {a.v}\\n")
import time
while a.wait_for and not os.path.exists(a.wait_for): time.sleep(0.05)
if a.fail_always: print("BOOM fake failure", file=sys.stderr); sys.exit(3)
if a.rc:
    open(a.out, "w").write('{"partial": 1}'); sys.exit(a.rc)
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
        kw.setdefault("launch_checks", False)          # the real-smoke/model/kind checks have their own tests below
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


@pytest.mark.parametrize("rc,runs", [(137, 1), (3, 2)])
def test_memcap_kill_is_not_retried_other_failures_are(env, rc, runs):
    env.add("a", grid={"x": f"--rc {rc}"})
    env.commit()
    assert env.engine("a").run() == "ERROR"
    assert len(env.calls()) == runs
    v = json.loads((env.repo / core.RES / "a/verdict.json").read_text())
    assert ("not retried" in v["summary"]) == (rc == 137) and f"rc={rc}" in v["summary"]
    assert json.loads((env.repo / core.RES / "a/dev/x.json.fail1.json").read_text()) == {"partial": 1}   # partial out pulled


def test_six_seed_data_checked_before_any_dev_job(env):
    _yaml(env, "r", BASE.format(n="r") + "pick: pick\ndev_seed: 7\ndev_grid: {x: '--v 5'}\nseeds: [1, 2]\n"
          "requires: ['data/s{seed}']\n")
    env.commit()
    for n in ("n0", "n1"):
        for s in (7, 1):
            (env.tmp / "nodes" / n / "data" / f"s{s}").mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit, match=r"seed 2: required data missing.*data/s2"):
        env.engine("r").run()
    assert env.calls() == []                                    # the dev job never launched
    (env.tmp / "nodes/n1/data/s2").mkdir(parents=True)
    assert env.engine("r").run() == "GO"


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
    e = Engine("a", [w], repo=env.repo, poll=0.02, log=lambda *_: None, launch_checks=False)
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


# ---- placement, requires, vars, status, HOLD, stale rc (11-27)
def _yaml(env, name, body):
    (env.repo / "tools/engine/specs" / f"{name}.yaml").write_text(body)


BASE = ("name: {n}\nquestion: q\nprereg: p.md\nmodule: m.py\nverdict: verdict\n"
        "cmd: '\"$PY\" brain/fake.py --seed {{seed}} {{params}} --out {{out}}'\nmemory_gb: 1\nladder: {{max_len: 1, next: {{}}}}\n")


def test_placement_pins_each_seed_to_its_node(env):
    _yaml(env, "p", BASE.format(n="p") + "six_params: '--v 5'\nseeds: [1, 2, 3]\nplacement: {1: n1, 2: n0, 3: [n1]}\n")
    env.commit()
    assert env.engine("p").run() == "GO"
    jobs = _state(env)["specs"]["p"]["jobs"]
    assert {k: j["node"] for k, j in jobs.items()} == {"six/s1": "n1", "six/s2": "n0", "six/s3": "n1"}


def test_placement_on_absent_node_refused_before_any_launch(env):
    _yaml(env, "p", BASE.format(n="p") + "six_params: '--v 5'\nseeds: [1, 2]\nplacement: {2: pool99}\n")
    env.commit()
    with pytest.raises(SystemExit, match="seed 2 is placed on"):
        env.engine("p").run()
    assert env.calls() == []


def test_kr_placement_alias_and_bad_alias():
    sp = core.Spec.find("mix_night")
    assert sp.placement == {319: ["w3090"], 320: ["w3090"], 321: ["w3090"]}     # Amendment S2: the W 3090
    sp.apply({"placement": "@kr"})
    assert sp.placement == {319: ["pool40"], 320: ["pool41"], 321: ["pool42"]}
    with pytest.raises(ValueError, match="placement"):
        sp.apply({"placement": "@nope"})


def test_requires_missing_everywhere_stops_with_the_list(env):
    _yaml(env, "r", BASE.format(n="r") + "six_params: '--v 5'\nseeds: [1]\nrequires: ['data/s{seed}/x', data/common]\n")
    env.commit()
    with pytest.raises(SystemExit, match=r"required data missing.*data/s1/x.*data/common"):
        env.engine("r").run()
    assert env.calls() == []


def test_requires_routes_unpinned_seed_to_the_node_with_data(env):
    _yaml(env, "r", BASE.format(n="r") + "six_params: '--v 5'\nseeds: [1, 2]\nrequires: ['data/s{seed}']\n")
    env.commit()
    for n, s in (("n1", 1), ("n1", 2)):
        (env.tmp / "nodes" / n / "data" / f"s{s}").mkdir(parents=True, exist_ok=True)
    assert env.engine("r").run() == "GO"
    assert {j["node"] for j in _state(env)["specs"]["r"]["jobs"].values()} == {"n1"}


def test_pinned_seed_with_data_missing_on_its_node_is_refused(env):
    _yaml(env, "r", BASE.format(n="r") + "six_params: '--v 5'\nseeds: [1]\nplacement: {1: n0}\nrequires: ['data/s{seed}']\n")
    env.commit()
    (env.tmp / "nodes/n1/data/s1").mkdir(parents=True)           # data on the other node does not help a pinned seed
    with pytest.raises(SystemExit, match="n0: \\['data/s1'\\]"):
        env.engine("r").run()


def test_vars_seed_vars_and_tree_tokens(env):
    _yaml(env, "v", BASE.format(n="v") + "six_params: '--v {val}'\nseeds: [1, 2]\nvars: {val: 1}\nseed_vars: {2: {val: 7}}\n")
    env.commit()
    assert env.engine("v").run() == "GO"                         # mean (1 + 7) / 2 = 4 > 3
    assert sorted(env.calls()) == ["1 1.0", "2 7.0"]
    sp = core.Spec.find("v", env.repo)
    sp.cmd = "x {seed} {tree} {val} {commit} {job}"
    assert sp.render(2, "", "o", tree="abc", commit="c1", job="s2") == "x 2 abc 7 c1 s2" and sp.render(1, "", "o") == "x 1  1  "


@pytest.mark.parametrize("extra,ok_draft,match", [
    ("status: TODO\n", False, "status TODO"),
    ("status: REGISTERED\nvars: {a: TODO}\n", False, "a TODO is left"),
    ("status: DRAFT\n", True, "is DRAFT"),
])
def test_todo_and_draft_specs_refused(env, extra, ok_draft, match):
    _yaml(env, "t", BASE.format(n="t") + "six_params: '--v 5'\nseeds: [1]\n" + extra)
    env.commit()
    with pytest.raises(SystemExit, match=match):
        env.engine("t").run()
    assert env.calls() == []
    if ok_draft:
        assert env.engine("t", allow_draft=True).run() == "GO"
    else:
        with pytest.raises(SystemExit, match=match):
            env.engine("t", allow_draft=True).run()


def test_bad_status_rejected(env):
    _yaml(env, "t", BASE.format(n="t") + "six_params: '--v 5'\nseeds: [1]\nstatus: MAYBE\n")
    with pytest.raises(ValueError, match="status"):
        core.Spec.find("t", env.repo)


def test_pick_stop_gives_hold_dev_and_no_six(env):
    (env.repo / "tools/engine/specs/h.py").write_text("def pick(dev, ctx):\n    return {'stop': 'too slow'}\n"
                                                      "def verdict(six, pick, ctx):\n    raise AssertionError\n")
    _yaml(env, "h", BASE.format(n="h").replace("module: m.py", "module: h.py") + "pick: pick\ndev_seed: 7\n"
          "dev_grid: {x: '--v 1'}\nseeds: [1, 2]\n")
    env.commit()
    assert env.engine("h").run() == "HOLD-DEV"
    assert env.calls() == ["7 1.0"] and "too slow" in (env.repo / core.RES / "ENGINE_DIGEST.md").read_text()


def test_reset_at_same_tree_reruns_jobs_not_stale_rc(env):
    env.add("a", grid={"x": "--v 5"})
    env.commit()
    assert env.engine("a").run() == "GO" and len(env.calls()) == 3
    assert env.engine("a").run(reset=True) == "GO"
    assert len(env.calls()) == 6                                # same jd reused: old rc/out cleared, the jobs really ran


def test_job_script_host_flags(env):
    env.add("a", grid={})
    env.commit()
    nodes = make_nodes("ssh", allow_w3090=True)
    e = Engine("a", nodes, repo=env.repo, log=lambda *_: None)
    spec, job = _prep(e)
    assert "POOL_NODE=1" in e.job_script(nodes[0], spec, job, "r", "jd")
    assert "BRAIN_ALLOW_W3090=1" in e.job_script(nodes[-1], spec, job, "r", "jd")
    loc = env.engine("a", nodes=1)
    loc.st = e.st
    assert "ENGINE_LOCAL_TEST=1" in loc.job_script(loc.nodes[0], spec, job, "r", "jd")


# ---- smoke
SMOKE_FILES = ("tools/engine/__init__.py", "tools/engine/bundle.py", "tools/engine/smoke_job.py",
               "tools/engine/specs/engine_smoke.yaml", "tools/engine/specs/engine_smoke.py")


@pytest.fixture
def smoke_env(env):
    for f in SMOKE_FILES:
        (env.repo / f).parent.mkdir(parents=True, exist_ok=True)
        (env.repo / f).write_text((core.REPO / f).read_text())
    for i in range(3):
        (env.tmp / "nodes" / f"n{i}" / "brain-lane").mkdir(parents=True, exist_ok=True)
    env.commit()
    nodes = make_nodes("local", ["n0", "n1", "n2"], env.tmp / "nodes")
    return env, nodes


def test_smoke_end_to_end_on_local_nodes(smoke_env):
    env, nodes = smoke_env
    rep = core.smoke(nodes, repo=env.repo, poll=0.05, log=lambda *_: None, imports="json", sleep_s=0.5)
    assert rep["ok"] and rep["verdict"] == "SMOKE-OK", rep
    assert set(rep["checks"]) == {"crash_after_launch", "placement", "reattached_not_relaunched", "pulled", "shipped",
                                  "resume_finished_is_noop", "verdict_ok"}
    assert json.loads((env.repo / core.RES / "engine_smoke/smoke_report.json").read_text())["ok"]
    rep2 = core.smoke(nodes, repo=env.repo, poll=0.05, log=lambda *_: None, imports="json", sleep_s=0.2)
    assert rep2["ok"]                                            # a second smoke starts clean (own state and job dirs)


def test_smoke_fails_on_missing_import(smoke_env):
    env, nodes = smoke_env
    rep = core.smoke(nodes[:1], repo=env.repo, poll=0.05, log=lambda *_: None, imports="json,no_such_mod_zz", sleep_s=0)
    assert not rep["ok"] and rep["verdict"] == "SMOKE-FAIL" and rep["per_seed"]["1"]["imports"] is False


@pytest.mark.parametrize("bad", ["imports", "once", "commit", None])
def test_smoke_verdict_each_clause_fails_alone(bad):
    from tools.engine.specs import engine_smoke as m
    r = {"imports": {"torch": {"ok": bad != "imports"}}, "count": 2 if bad == "once" else 1,
         "commit": "x" if bad == "commit" else "c"}
    v = m.verdict({1: r}, None, {"commit": "c"})
    assert v["verdict"] == ("SMOKE-OK" if bad is None else "SMOKE-FAIL")
    assert [k for k in ("imports", "once", "commit") if not v["per_seed"][1][k]] == ([bad] if bad else [])
    assert m.verdict({}, None, {"commit": "c"})["verdict"] == "SMOKE-FAIL"


def test_smoke_job_guard_and_bundle(tmp_path, monkeypatch):
    from tools.engine import bundle, smoke_job
    for k in ("POOL_NODE", "BRAIN_ALLOW_W3090", "ENGINE_LOCAL_TEST"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="refuses"):
        smoke_job.main(["--seed", "1", "--out", str(tmp_path / "o.json"), "--imports", "json"])
    monkeypatch.setenv("POOL_NODE", "1")
    (tmp_path / "a").mkdir()
    smoke_job.main(["--seed", "1", "--out", str(tmp_path / "a/o.json"), "--imports", "json"])
    assert json.loads((tmp_path / "a/o.json").read_text())["count"] == 1
    d = tmp_path / "u"
    d.mkdir()
    with pytest.raises(SystemExit):
        bundle.main([str(d), str(tmp_path / "b.json")])
    assert not (tmp_path / "b.json").exists()
    (d / "u_A0_s1_n13.json").write_text('{"kl": 1}')
    (d / "u_x.json.tmp").write_text("partial")
    (d / "other.json").write_text("{}")
    bundle.main([str(d), str(tmp_path / "b.json"), "--t0", "0"])
    b = json.loads((tmp_path / "b.json").read_text())
    assert list(b["units"]) == ["u_A0_s1_n13.json"] and list(b["mtime"]) == ["u_A0_s1_n13.json"]


def test_dev_memory_gb_caps_dev_jobs_and_requires_local(env):
    _yaml(env, "m", BASE.format(n="m").replace("memory_gb: 1", "memory_gb: 13") + "pick: pick\ndev_seed: 7\n"
          "dev_grid: {x: '--v 5'}\ndev_memory_gb: 21\nseeds: [1]\nrequires_local: [docs/prereg.md]\n")
    env.commit()
    e = Engine("m", make_nodes("ssh"), repo=env.repo, log=lambda *_: None, launch_checks=False)
    spec, job = _prep(e)
    assert "memcap.sh 21 --" in e.job_script(e.nodes[0], spec, {**job, "key": "dev/x"}, "r", "jd")
    assert "memcap.sh 13 --" in e.job_script(e.nodes[1], spec, {**job, "key": "six/s1"}, "r", "jd")
    assert spec.mem("dev") == 21 and spec.mem("six") == 13
    with pytest.raises(SystemExit, match="requires_local missing.*docs/prereg.md"):
        e.check_runnable(spec)
    (env.repo / "docs").mkdir()
    (env.repo / "docs/prereg.md").write_text("x")
    e.check_runnable(spec)


def test_ship_key_carries_commit_and_tools_hash(env):
    env.add("a", grid={})
    env.commit()
    e = env.engine("a", nodes=1)
    _prep(e)
    k = e.ship_key()
    assert k.startswith(e.st["tree"][:12] + "_" + e.st["commit"][:8] + "_")
    e.st["commit"] = "f" * 40
    assert e.ship_key() != k                                    # one COMMIT file per shipped dir


# ---- the mix-night spec (brain/research/2026-11-27-mix-night-prereg.md, REGISTERED 11-27 after pre-data review 2)
def test_mix_night_spec_renders_per_seed_and_is_runnable_once_registered():
    from brain.experiments import mix_night_screen as M
    e = Engine("mix_night", make_nodes("ssh"), dry=True, log=lambda *_: None)
    sp = e.root
    assert sp.status == "REGISTERED" and sp.dev_seed == 319 and sp.seeds == [319, 320, 321]
    assert (sp.memory_gb, sp.mem("dev", 319), sp.mem("dev", 321)) == (21, 21, 21) and sp.vars["offframe_sha256"] == M.OFFFRAME_SHA256
    assert (core.REPO / sp.vars["offframe"]).exists()
    c = sp.render(320, "A0 A1", "/o.json", tree="t" * 12, commit="c" * 12, job="s320")
    assert "--seed 320" in c and "snaps_s320 " in c and "--foreign-snaps $KR/foreign_s321" in c
    assert "mix_night/cccccccccccc/s320" in c and "for X in A0 A1;" in c and "tools.engine.bundle $D /o.json --text '*.txt'" in c
    assert f"--offframe-sha256 {M.OFFFRAME_SHA256}" in c and "TF=$D/time_${A}.$(date +%s).txt; ${TIMEV:-/usr/bin/time} -v -o $TF" in c and "detcheck --dir $D --seed 320" in c
    req = [sp.fill(p, 321) for p in sp.requires]
    assert "brain-lane/route_repo/_scratch/route/kr/foreign_s319/n13/episodes.json" in req and any("snaps_s321/n12" in r for r in req)
    pre = core.REPO / "brain/research/2026-11-27-mix-night-prereg.md"
    if not pre.exists():
        with pytest.raises(SystemExit, match="requires_local missing"):
            e.check_runnable(sp)
    sp.requires_local = []
    e.check_runnable(sp)                                         # registered: no --allow-draft needed


TIME_V = "\tMaximum resident set size (kbytes): {rss}\n\tElapsed (wall clock) time (h:mm:ss or m:ss): {wall}\n"


def _step0(rss_gb=8.0, wall="40:00", a2_gb=None, det="rc 0", seeds=(319, 320, 321), mtime=None, resumed=False):
    """Synthetic step-0 bundles of every node: A0r, A0 n13, detcheck.txt, A2 n13, with /usr/bin/time -v reports."""
    dev = {}
    for S in seeds:
        texts = {f"time_A0r.100.txt": TIME_V.format(rss=int(rss_gb * 1048576), wall=wall),
                 f"time_A0.200.txt": TIME_V.format(rss=int(rss_gb * 1048576), wall=wall),
                 f"time_A2.300.txt": TIME_V.format(rss=int((a2_gb or rss_gb * 0.9) * 1048576), wall=wall)}
        if resumed:                                             # a later attempt that resumed the units: tiny RSS
            texts["time_A2.900.txt"] = TIME_V.format(rss=100000, wall="0:05")
        if det is not None:
            texts["detcheck.txt"] = f"{{}}\n{det}\n"
        units = {f"u_{A}_s{S}_n{n}.json": {} for A, n in (("start", 12), ("A0r", 13), ("A0", 13), ("A2", 13))}
        dev[f"step0_s{S}"] = {"units": units, "texts": texts, "mtime": dict(mtime or {})}
    return dev


@pytest.mark.parametrize("kw,expect", [
    ({}, "go"),                                       # cap ceil(10.4) + 1 = 12 <= 13, 40 min
    ({"rss_gb": 9.2}, "go"),                          # cap ceil(11.96) + 1 = 13: fits exactly
    ({"rss_gb": 9.3}, "cap 14 GB > the six-phase"),   # A0r alone over: ceil(12.09) + 1 = 14 > 13
    ({"a2_gb": 9.3}, "cap 14 GB > the six-phase"),    # A2 alone over (A0r 8.0): the cap takes the larger arm
    ({"a2_gb": 9.3, "resumed": True}, "cap 14"),      # a resumed attempt's tiny report never lowers the measured peak
    ({"wall": "45:00"}, "go"),                        # exactly 45 min passes
    ({"wall": "0:45:01"}, "min > 45"),                # h:mm:ss form, 45 min 1 s
    ({"det": "rc 3"}, "determinism pair MISS"),       # the det pair alone
    ({"det": None}, "determinism unread"),            # no detcheck.txt alone
    ({"seeds": (319, 321)}, "step 0 s320 left no"),   # one node's step 0 absent alone
])
def test_mix_night_step0_pick(kw, expect):
    from tools.engine.specs import mix_night as m
    p = m.pick(_step0(**kw), {"memory_gb": 13})
    if expect == "go":
        assert p["params"].split() == ["A0r", "A0L", "A0", "A0e", "A0k", "A1", "A2", "A3", "A4", "A5s", "A5f"] and "stop" not in p
        assert set(p["memcap"]) == {319, 320, 321} and len(set(p["memcap"].values())) == 1
    else:
        assert "params" not in p and expect in p["stop"], p
    dev = _step0()
    dev["step0_s320"]["texts"].pop("time_A2.300.txt")             # a missing RSS report alone
    assert "report (RSS) for ['A2']" in m.pick(dev, {"memory_gb": 13})["stop"]
    dev = _step0(mtime={"u_start_s319_n12.json": 0.0, "u_A0r_s319_n13.json": 46 * 60.0})
    assert "min > 45" in m.pick(dev, {"memory_gb": 13})["stop"]      # with unit mtimes the night + reading is A0r - start
    dev = _step0(mtime={"u_start_s319_n12.json": 0.0, "u_A0r_s319_n13.json": 30 * 60.0}, wall="2:00:00")
    assert m.pick(dev, {"memory_gb": 13})["step0_min"] == 30.0
    assert set(m.ALL_ARMS.split()) == set(__import__("brain.experiments.mix_night_screen", fromlist=["ARMS"]).ARMS)


def test_mix_night_per_seed_caps_reach_the_jobs(env):
    sp = core.Spec.find("mix_night")
    assert sp.mem("dev", 319) == 21 and sp.mem("dev", 320) == 21 and sp.mem("six", 320) == 21
    pk = json.loads(json.dumps({"memcap": {319: 12, 320: 9, 321: 40}}))   # through the JSON state: string keys
    assert (sp.mem("six", 319, pk), sp.mem("six", 320, pk), sp.mem("six", 321, pk)) == (12, 9, 21)   # never above memory_gb
    assert [(n, s) for n, s, _ in sp.dev_items()] == [("step0_s319", 319), ("step0_s320", 320), ("step0_s321", 321)]


def test_mix_night_dry_run_fills_seed_vars_per_seed():
    out = []
    Engine("mix_night", make_nodes("ssh"), dry=True, log=out.append).run()
    six = [o for o in out if "-> six seeds" in o]
    assert len(six) == 3 and all("{foreign}" not in o for o in six)
    assert "--seed 321" in six[2] and "foreign_s319 " in six[2]
    dev = [o for o in out if o.lstrip().startswith("dev   step0")]
    assert len(dev) == 3 and all("A0r A0@13 DET A2@13" in o for o in dev) and "(s320)" in dev[1]


_PLC = {"placebo_pool": 100, "placebo_excluded": 64, "placebo_overlap": 0}
_ENG = {"A1": {"ce_keep_w": 1.0, "mean_ce_keep": 2.1}, "A2": {"keep_per_step": [2]},      # what an engaged arm records
        "A3": {"keep_n": 256, "keep_n_loop": 256}, "A4": {"kl_weight": 6.0}, "A5s": _PLC, "A5f": _PLC}


def _unit(arm, seed, night, kl, off=1.0, nll=1.0, refusal=0.0, name=0.0):
    return {"arm": arm, "seed": seed, "night": night, "kl": kl, "off_kl": off, "refusal": refusal, "name": name,
            "sleep_info": dict(_ENG.get(arm, {})),
            "ret": {s: {"n": 20, "nll": nll} for s in ("new", "mid", "old")}, "commit": "c", "device": "cpu",
            "precision": "fp32", "model": "1.5b", "host": f"h{seed}", "offframe_sha256": "h"}


def _mix_six(cand_kl, a0r_kl=0.2, commit_of=None):
    six = {}
    for s in (319, 320, 321):
        us = [_unit("start", s, 12, 0.1), _unit("A0", s, 13, 0.2), _unit("A0r", s, 13, a0r_kl), _unit("A0L", s, 13, 0.2),
              _unit("A0", s, 15, 1.0), _unit("A0e", s, 15, 1.01), _unit("A0k", s, 15, 0.99),
              _unit("A5s", s, 15, 0.5), _unit("A5f", s, 15, 0.6)]
        us += [_unit(X, s, 15, cand_kl.get(X, 1.0), off=0.5 if X in cand_kl else 1.0) for X in ("A1", "A2", "A3", "A4")]
        if commit_of and s == commit_of:
            us[-1]["commit"] = "other"
        six[s] = {"units": {f"u_{u['arm']}_s{u['seed']}_n{u['night']}.json": u for u in us}}
    return six


@pytest.fixture
def lives(tmp_path):
    out = []
    for s in (319, 320, 321):
        p = tmp_path / f"lifeKR_s{s}.json.json"
        p.write_text(json.dumps({"seed": s, "nights": [{"line": "trunk", "night": 13, "lang_guard": {"kl": 0.2}}]}))
        out.append(p.name)
    return out


@pytest.mark.parametrize("cand,a0r,commit_of,word,status", [
    ({"A3": 0.5}, 0.2, None, "PASS", {"A1": "KILL", "A2": "KILL", "A3": "PASS", "A4": "KILL"}),
    ({}, 0.2, None, "KILL", {"A1": "KILL", "A2": "KILL", "A3": "KILL", "A4": "KILL"}),
    ({"A3": 0.8}, 0.2, None, "INCONCLUSIVE", {"A1": "KILL", "A2": "KILL", "A3": "INCONCLUSIVE", "A4": "KILL"}),  # P1 mean 0.8 > 0.75 alone
    ({"A3": 0.5}, 0.3, None, "INCONCLUSIVE", "INCONCLUSIVE: A0r differs"),                                     # instrument void alone
    ({"A3": 0.5}, 0.2, 320, "INCONCLUSIVE", "INCONCLUSIVE: mixed provenance"),                                 # one unit's commit alone
])
def test_mix_night_verdict_word(tmp_path, lives, cand, a0r, commit_of, word, status):
    from tools.engine.specs import mix_night as m
    ctx = {"res": str(tmp_path / "res"), "repo": str(tmp_path), "vars": {"lives": lives}}
    six = _mix_six(cand, a0r, commit_of)
    six[320]["texts"] = {"time_A3.5.txt": TIME_V.format(rss=12345, wall="1:00"), "time_A3.9.txt": TIME_V.format(rss=99, wall="1")}
    stale = tmp_path / "res/units/u_A9_s319_n13.json"           # a same-commit leftover of an earlier run
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    v = m.verdict(six, {"params": "x"}, ctx)
    assert v["verdict"] == word and v["lives_read"] == "3 of 3" and not stale.exists()
    assert v["rss_kb"] == {319: {}, 320: {"A3": 12345}, 321: {}}
    if isinstance(status, dict):
        assert v["candidates"] == status and v["placebo"] == "BRACKETED"
    else:
        assert all(s.startswith(status) for s in v["candidates"].values())
    assert (tmp_path / "res/units/u_A3_s321_n15.json").exists()


def test_mix_night_verdict_without_lives_is_void(tmp_path):
    from tools.engine.specs import mix_night as m
    v = m.verdict(_mix_six({"A3": 0.5}), {}, {"res": str(tmp_path / "r"), "repo": str(tmp_path), "vars": {"lives": ["nope.json"]}})
    assert v["verdict"] == "ERROR" and v["lives_read"] == "0 of 1" and "K-sane unread" in v["summary"] and "candidates" not in v
    sp = core.Spec.find("mix_night")
    assert set(sp.vars["lives"]) <= set(sp.requires_local)      # and the engine refuses to launch without them


# ---- the local-rule v2 specs (brain/research/2026-11-27-local-rule-v2-prereg.md, REGISTERED 11-27)
def test_local_rule_v2_specs_render_and_chain():
    a, b = core.Spec.find("local_rule_v2_01b"), core.Spec.find("local_rule_v2_04b")
    assert (a.status, b.status, a.memory_gb, b.memory_gb) == ("REGISTERED", "REGISTERED", 8, 13) and a.question == b.question
    assert len(a.dev_grid) == 10 and a.dev_grid == {k: v for k, v in b.dev_grid.items()}
    assert a.dev_grid["lr0.3_tau0.99"] == "--arms v2 --lr 0.3 --tau 0.99 --tag lr0.3_tau0.99"
    c = a.render(7, a.dev_grid["ref"], "/o.json", commit="c" * 12, job="ref")
    assert "local_rule_v2_01b/cccccccccccc/ref" in c and "--model $BRAIN_MODEL_DIR/rwkv7-g1d-0.1b" in c
    assert "--arms no_sleep,gradient --tag ref --out-dir $D" in c and "tools.engine.bundle $D /o.json --glob '*.json'" in c
    for v in ("GO-V2", "GO-PLAIN-LOCAL", "GO-NONSPECIFIC", "NEGATIVE", "NEGATIVE-DEV"):    # every defined verdict
        assert a.next_for(v) == "local_rule_v2_04b", v
    for v in ("UNDEFINED", "ERROR", "HOLD-DEV"):                      # the prereg decision table: diagnose 01b first
        assert a.next_for(v) is None, v
    assert b.next_for("GO-V2") is None
    out = []
    Engine("local_rule_v2_01b", make_nodes("ssh"), dry=True, log=out.append).run()
    assert any("== local_rule_v2_04b" in o for o in out)


def _fake_v2_main(monkeypatch, pick_out=None, score_out=None, refuse=None):
    from brain.experiments import local_rule_v2 as V
    seen = []

    def main(argv):
        seen.append(argv)
        if refuse:
            raise SystemExit(refuse)
        o = argv[argv.index("--out") + 1]
        Path(o).write_text(json.dumps(pick_out if argv[0] == "pick" else score_out))
    monkeypatch.setattr(V, "main", main)
    return seen


@pytest.mark.parametrize("hp,expect", [({"verdict": "PICKED", "pick": {"lr": 0.1, "tau": 0.9}}, "params"),
                                       ({"verdict": "NEGATIVE-DEV", "pick": None}, None),
                                       ({"verdict": "UNDEFINED-DEV", "pick": None}, "stop")])
def test_local_rule_v2_pick_delegates_to_the_harness(tmp_path, monkeypatch, hp, expect):
    from brain.experiments import local_rule_v2 as V
    from tools.engine.specs import local_rule_v2 as m
    seen = _fake_v2_main(monkeypatch, pick_out=hp)
    (tmp_path / "dev_units").mkdir()
    (tmp_path / "dev_units/lr9_tau9_s7_v2.json").write_text("{}")   # a same-commit leftover of an older grid
    dev = {"ref": {"units": {"ref_s7_gradient.json": {"x": 1}}}, "lr0.1_tau0.9": {"units": {"lr0.1_tau0.9_s7_v2.json": {"y": 2}}}}
    p = m.pick(dev, {"res": str(tmp_path)})
    assert seen[0][:2] == ["pick", "--dir"] and (tmp_path / "dev_units/lr0.1_tau0.9_s7_v2.json").exists()
    assert not (tmp_path / "dev_units/lr9_tau9_s7_v2.json").exists()
    if expect == "params":
        assert p["params"] == f"--arms {','.join(V.ARMS)} --lr 0.1 --tau 0.9 --tag six" and p["harness_pick"] == hp
    elif expect is None:
        assert p is None
    else:
        assert "UNDEFINED-DEV" in p["stop"] and "params" not in p
    _fake_v2_main(monkeypatch, refuse="pick refused: v2 cells [] != the 9 registered (lr, tau)")
    assert "v2 cells" in m.pick(dev, {"res": str(tmp_path)})["stop"]


def test_local_rule_v2_verdict_delegates_to_the_harness(tmp_path, monkeypatch):
    from tools.engine.specs import local_rule_v2 as m
    hp = {"verdict": "PICKED", "pick": {"lr": 0.1, "tau": 0.9}}
    seen = _fake_v2_main(monkeypatch, score_out={"verdict": "GO-V2", "clauses": {"a": True, "f": False}})
    v = m.verdict({42: {"units": {"six_s42_v2.json": {"z": 3}}}}, {"harness_pick": hp}, {"res": str(tmp_path)})
    assert v["verdict"] == "GO-V2" and "failed clauses ['f']" in v["summary"]
    a = seen[0]
    assert a[:2] == ["score", "--dir"] and a[a.index("--tag") + 1] == "six"
    assert json.loads(Path(a[a.index("--pick") + 1]).read_text()) == hp and (tmp_path / "six_units/six_s42_v2.json").exists()
    _fake_v2_main(monkeypatch, refuse="missing units: [('sham', 42)]")
    v = m.verdict({}, {"harness_pick": hp}, {"res": str(tmp_path)})
    assert v["verdict"] == "ERROR" and "missing units" in v["summary"]


# ---- the real smoke stage, model naming, node kind check, RAM placement (engine-first launches)
TIMESHIM = """#!/bin/bash
shift; out=$2; shift 2; "$@"; rc=$?
printf '\\tMaximum resident set size (kbytes): %s\\n\\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.50\\n' "$FAKE_RSS_KB" > "$out"
exit $rc
"""
CHECKED = BASE.replace("cmd: '", "model: m.pth\ncmd: 'test -f {{model}} && ") + "six_params: '--v 5'\nseeds: [1]\n"


@pytest.fixture
def checked(env, monkeypatch):
    shim = env.tmp / "timeshim"
    shim.write_text(TIMESHIM)
    shim.chmod(0o755)
    monkeypatch.setenv("ENGINE_TIME", str(shim))
    monkeypatch.setenv("FAKE_RSS_KB", str(int(3.5 * 1048576)))
    for n in ("n0", "n1"):
        (env.tmp / "nodes" / n / "brain-lane/models").mkdir(parents=True)
        (env.tmp / "nodes" / n / "brain-lane/models/m.pth").write_text("w")

    def engine(name, smoke="{params: '--v 5'}"):
        _yaml(env, name, CHECKED.format(n=name) + (f"smoke: {smoke}\n" if smoke else ""))
        env.commit()
        e = env.engine(name, launch_checks=True)
        e.imports = ("json",)
        return e
    return engine


def test_smoke_measures_rss_and_sets_the_memcap(env, checked):
    e = checked("a")
    assert e.run() == "GO" and env.calls() == ["1 5.0", "1 5.0"]          # smoke, then the six-seed job
    sm = _state(env)["specs"]["a"]["smoke"]
    assert abs(sm["peak_rss_gb"] - 3.5) < 1e-9 and sm["wall_s"] == 1.5 and sm["memcap_gb"] == 6   # ceil(1.3 x 3.5 + 1)
    assert e.mem(e.root, "six", 1) == 6 and e.mem(e.root, "smoke", 1) == 1
    assert json.loads((env.repo / core.RES / "a/verdict.json").read_text())["smoke"]["memcap_gb"] == 6


def test_placement_skips_nodes_too_small_for_the_measured_cap(env, checked):
    (env.repo / "tools/engine/nodes.json").write_text('{"n0": {"ram_gb": 6}}')   # 6 < 6 + 2: n0 only fits the smoke
    e = checked("a")
    assert e.run() == "GO"
    jobs = _state(env)["specs"]["a"]["jobs"]
    assert jobs["smoke/smoke"]["node"] == "n0" and jobs["six/s1"]["node"] == "n1"
    assert e.ram(e.nodes[0]) == 6 and e.ram(e.nodes[1]) == 23


def test_smoke_failure_stops_the_chain_with_its_log_tail(env, checked):
    e = checked("a", smoke="{params: '--fail-always'}")
    assert e.run() == "ERROR-SMOKE" and len(env.calls()) == 1          # never retried, no dev/six job
    dg = (env.repo / core.RES / "ENGINE_DIGEST.md").read_text()
    assert "ERROR-SMOKE" in dg and "BOOM fake failure" in dg and "rc=3" in dg


def test_missing_model_smoke_or_kind_refuses(env, checked):
    for n in ("n0", "n1"):
        (env.tmp / "nodes" / n / "brain-lane/models/m.pth").unlink()
    with pytest.raises(SystemExit, match="required data missing.*brain-lane/models/m.pth"):
        checked("a").run()
    assert env.calls() == []
    _yaml(env, "b", BASE.format(n="b") + "six_params: '--v 5'\nseeds: [1]\nsmoke: {params: '--v 5'}\n")
    env.commit()
    with pytest.raises(SystemExit, match="must name its model"):
        env.engine("b", launch_checks=True).run()
    with pytest.raises(SystemExit, match="no `smoke"):
        checked("c", smoke=None).run()
    (env.tmp / "nodes/n0/brain-lane/models/m.pth").write_text("w")
    e = checked("d")
    e.imports = ("no_such_mod_zz",)
    with pytest.raises(SystemExit, match="no such node is in use"):
        e.run()


# ---- preconditions blocks: every verdict.json / pick.json carries what earned it (tools.verdict.Verdict, commit gate class V)
HOOK_MOD = MOD + '''
def preconditions(results, ctx):
    ok = all(r["v"] >= 0 for r in results["dev"].values())
    return [("dev values nonnegative", ok, "spec hook"), ("hook saw stage", ctx["stage"] in ("pick", "final"), ctx["stage"])]
'''


def _gate(path):
    from tools.gates import verdict_preconditions as G
    return G.check([str(path)])


def _read(env, name, f):
    return json.loads((env.repo / core.RES / name / f).read_text())


def _names(rec):
    return {c["name"]: c["ok"] for c in rec["preconditions"]}


def test_verdict_and_pick_carry_preconditions_and_pass_the_gate(env):
    (env.repo / "tools/engine/specs/m.py").write_text(HOOK_MOD)
    env.add("a", grid={"lo": "--v 1", "hi": "--v 5"})
    env.commit()
    assert env.engine("a").run() == "GO"
    v, p = _read(env, "a", "verdict.json"), _read(env, "a", "pick.json")
    for rec in (v, p):
        n = _names(rec)
        assert rec["preconditions_status"] == "GO" and all(ok is True for ok in n.values())
        assert {"job dev/lo present and parsed", "job dev/hi present and parsed", "dev values nonnegative", "hook saw stage"} <= set(n)
        assert any(d["process"] == "real smoke stage" for d in rec["disabled_processes"])   # no smoke stage here: scope, not a pass
    assert {"job six/s1 present and parsed", "job six/s2 present and parsed"} <= set(_names(v))
    assert "job six/s1 present and parsed" not in _names(p)
    assert v["verdict"] == "GO" and p["verdict"] == "PICKED"
    assert _gate(env.repo / core.RES / "a/verdict.json") == [] and _gate(env.repo / core.RES / "a/pick.json") == []


def test_negative_dev_and_error_files_carry_the_block_too(env):
    env.add("n", grid={"lo": "--v -1"})
    env.add("e", grid={"x": "--fail-always"})
    env.commit()
    assert env.engine("n").run() == "NEGATIVE-DEV"
    v, p = _read(env, "n", "verdict.json"), _read(env, "n", "pick.json")
    assert v["verdict"] == "NEGATIVE-DEV" and p["verdict"] == "NEGATIVE-DEV" and "job six/s1 present and parsed" not in _names(v)
    assert _gate(env.repo / core.RES / "n/verdict.json") == [] and _gate(env.repo / core.RES / "n/pick.json") == []
    assert env.engine("e").run() == "ERROR"
    ve = _read(env, "e", "verdict.json")
    assert ve["verdict"] == "ERROR" and _names(ve)["job dev/x present and parsed"] is False
    assert _gate(env.repo / core.RES / "e/verdict.json") == []        # an ERROR asserts no result


def test_unmet_spec_precondition_makes_the_verdict_undefined_and_the_gate_still_passes(env):
    (env.repo / "tools/engine/specs/m.py").write_text(MOD + '''
def preconditions(results, ctx):
    return [("arm engaged", ctx["stage"] == "pick", "fails at the final stage only"),
            ("never measured", None if ctx["stage"] == "final" else True, "")]
''')
    env.add("a", grid={"hi": "--v 5"})
    env.commit()
    assert env.engine("a").run().startswith("UNDEFINED")
    v = _read(env, "a", "verdict.json")
    assert v["verdict"] == "UNDEFINED (GO)" and v["verdict_raw"] == "GO" and v["preconditions_status"] == "UNDEFINED"
    assert any("never measured" in r for r in v["undefined_reasons"])
    assert _gate(env.repo / core.RES / "a/verdict.json") == []
    # the gate refuses the same file if it were to assert GO over the failed block (the relabel is what keeps it passing)
    bad = {**v, "verdict": "GO"}
    (env.tmp / "bad.json").write_text(json.dumps(bad))
    assert _gate(env.tmp / "bad.json")


def test_failed_pick_precondition_stops_before_the_six_seeds(env):
    (env.repo / "tools/engine/specs/m.py").write_text(MOD + '''
def preconditions(results, ctx):
    return [("pick-stage check", ctx["stage"] != "pick", "")]
''')
    env.add("a", grid={"hi": "--v 5"})
    env.commit()
    assert env.engine("a").run() == "UNDEFINED-DEV"
    assert not (env.repo / core.RES / "a/six").exists()
    assert _read(env, "a", "pick.json")["preconditions_status"] == "UNDEFINED"


@pytest.mark.parametrize("bad,name", [("smoke_job", "smoke rc 0"), ("smoke_peak", "smoke measured peak RSS"),
                                      ("dev_file", "job dev/hi present and parsed"), ("six_file", "job six/s1 present and parsed")])
def test_each_core_precondition_fails_alone(env, bad, name):
    """Direct block test: a finished state is built, then exactly one thing is broken; only that check may fail."""
    env.add("a", grid={"hi": "--v 5"}, seeds=(1,), extra="model: m.pth\nsmoke: {params: '--v 5'}\n")
    env.commit()
    e = env.engine("a", launch_checks=True)
    e.st = e.load()
    e.st.update(commit="c" * 40, tree="t" * 40)
    spec = e.spec("a")
    out = e.res / "a"
    (out / "dev").mkdir(parents=True)
    (out / "six").mkdir()
    (out / "dev/hi.json").write_text('{"v": 5}')
    (out / "six/s1.json").write_text('{"v": 5}')
    job = lambda k, st="done": {"key": k, "seed": 1, "status": st}
    ss = {"phase": "done", "pick": {"params": "--v 5"}, "six_reached": True, "smoke": {"peak_rss_gb": 2.0},
          "jobs": {"smoke/smoke": job("smoke/smoke"), "dev/hi": job("dev/hi"), "six/s1": job("six/s1")}}
    names = ("smoke rc 0", "smoke measured peak RSS", "job dev/hi present and parsed", "job six/s1 present and parsed")
    assert [c for c in e.precondition_block(spec, ss, "final").decide(True, False)["preconditions"] if c["ok"] is not True] == []
    if bad == "smoke_job":
        ss["jobs"]["smoke/smoke"]["status"] = "failed"
    elif bad == "smoke_peak":
        ss["smoke"] = None
    elif bad == "dev_file":
        (out / "dev/hi.json").write_text("not json")
    else:
        (out / "six/s1.json").unlink()
    d = e.precondition_block(spec, ss, "final").decide(True, False)
    assert d["status"] == "UNDEFINED"
    not_ok = {c["name"] for c in d["preconditions"] if c["ok"] is not True}
    assert not_ok == {name}, not_ok
    assert set(names) <= {c["name"] for c in d["preconditions"]}


def test_smoke_chain_block_has_smoke_and_measured_peak(env, checked):
    e = checked("a")
    assert e.run() == "GO"
    v = _read(env, "a", "verdict.json")
    n = _names(v)
    assert n["smoke rc 0"] is True and n["smoke measured peak RSS"] is True and not v["disabled_processes"]
    assert _gate(env.repo / core.RES / "a/verdict.json") == []
    sm = checked("b", smoke="{params: '--fail-always'}")
    assert sm.run() == "ERROR-SMOKE"
    vb = _read(env, "b", "verdict.json")
    assert _names(vb)["smoke rc 0"] is False and _gate(env.repo / core.RES / "b/verdict.json") == []


def test_mix_night_runs_on_w3090_only_amendment_s2(monkeypatch):
    """Amendment S2: --nodes w3090 --allow-w3090 gives the GPU node only; every seed is placed there and fits its caps
    (smoke, step 0, six at 21 GB within the 23 GB budget); the job script runs cuda with BRAIN_ALLOW_W3090=1."""
    with pytest.raises(SystemExit, match="allow-w3090"):
        make_nodes("ssh", ["w3090"])
    nodes = make_nodes("ssh", ["w3090"], allow_w3090=True)
    assert [(n.name, n.kind) for n in nodes] == [("w3090", "w3090")]
    assert [n.name for n in make_nodes("ssh", ["pool40", "w3090"], allow_w3090=True)] == ["pool40", "w3090"]
    e = Engine("mix_night", nodes, log=lambda *_: None)
    sp = e.root
    monkeypatch.setattr(e, "kind_ok", lambda n: True)
    for phase in ("smoke", "dev", "six"):
        for s in sp.seeds:
            assert [n.name for n in e.allowed(sp, s, e.fits(sp, phase, s))] == ["w3090"]
    e.st = {"tree": "t" * 40, "commit": "c" * 40, "specs": {}}
    sc = e.job_script(nodes[0], sp, {"key": "smoke/smoke", "seed": 319, "params": "A0r"}, "r", "jd")
    assert "--device cuda" in sc and "BRAIN_ALLOW_W3090=1" in sc and "POOL_NODE" not in sc and "memcap.sh 21 " in sc
    assert "$BRAIN_MODEL_DIR/rwkv7-g1l-1.5b-20261031-ctx34304.pth" in sc and "${BRAIN_MODEL_DIR:-$HOME/brain-lane/models}" in sc


# ---- w3090 smoke without GNU time: tools/engine/peak_rss.py (time -v compatible report), pool nodes keep /usr/bin/time
def test_w3090_job_script_uses_peak_rss_launcher_pool_keeps_gnu_time(env):
    nodes = make_nodes("ssh", allow_w3090=True)
    env.add("a", grid={"x": "--v 1"})
    env.commit()
    e = Engine("a", nodes, repo=env.repo, log=lambda *_: None)
    spec, job = _prep(e)
    job = {**job, "key": "smoke/smoke"}
    w, pool = nodes[-1], nodes[0]
    sw, sp = e.job_script(w, spec, job, "r", "jd"), e.job_script(pool, spec, job, "r", "jd")
    assert '"$PY" -m tools.engine.peak_rss -v -o' in sw and "/usr/bin/time" not in sw and 'TIMEV="$PY -m tools.engine.peak_rss"' in sw
    assert "/usr/bin/time -v -o" in sp and "peak_rss" not in sp and "TIMEV" not in sp
    assert "tools/engine/peak_rss.py" in core.SHIP_TOOLS


def _peak(tmp_path, code, extra_env=None, monkeypatch=None):
    from tools.engine import peak_rss
    out = tmp_path / "time.txt"
    rc = peak_rss.main(["-v", "-o", str(out), "--", sys.executable, "-c", code])
    return rc, out.read_text()


def test_peak_rss_report_is_parsed_by_timev_and_passes_the_exit_code(tmp_path, monkeypatch):
    from tools.engine import peak_rss
    monkeypatch.setattr(peak_rss, "gpu_used_mib", lambda: None)
    rc, txt = _peak(tmp_path, "b = bytearray(200*1024*1024); b[::4096] = b'x' * len(b[::4096]); import time; time.sleep(0.8); raise SystemExit(7)")
    assert rc == 7 and "Exit status: 7" in txt
    m = core.timev(txt)
    assert m and 0.19 < m["peak_rss_gb"] < 1.5 and 0.5 < m["wall_s"] < 30 and "Peak GPU" not in txt


def test_peak_rss_covers_grandchildren_and_gpu_peak(tmp_path, monkeypatch):
    from tools.engine import peak_rss
    used = iter([300, 300, 5300, 9300, 9300] + [300] * 50)
    monkeypatch.setattr(peak_rss, "gpu_used_mib", lambda: next(used))
    monkeypatch.setattr(peak_rss, "POLL_S", 0.05)
    code = ("import subprocess, sys, time; subprocess.run([sys.executable, '-c', 'b = bytearray(150*1024*1024); "
            "b[::4096] = b\"x\" * len(b[::4096])']); time.sleep(0.6)")
    rc, txt = _peak(tmp_path, code)
    assert rc == 0 and core.timev(txt)["peak_rss_gb"] > 0.14      # the waited grandchild's RSS counts
    assert "Peak GPU memory (MiB): 9000 (nvidia-smi sampled" in txt


def test_peak_rss_prefers_the_jobs_torch_peak_file(tmp_path, monkeypatch):
    from tools.engine import peak_rss
    monkeypatch.setattr(peak_rss, "gpu_used_mib", lambda: 300)
    code = "import os; open(os.environ['ENGINE_GPU_PEAK_FILE'], 'w').write(str(2048 * 1048576))"
    rc, txt = _peak(tmp_path, code)
    assert rc == 0 and "Peak GPU memory (MiB): 2048 (torch max_memory_allocated)" in txt

