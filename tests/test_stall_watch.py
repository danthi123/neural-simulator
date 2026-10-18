"""tools/stall_watch.py (RULING stall-alarm-1018, owner-approved 2026-10-18): a pool or GPU line whose output is still
absent after 2x its expected wall (the longest of the canary wall and the last 8 completed runs, else 6 h) is a STALL.
Reviewed 2026-10-18: the expected wall is no longer the newest run alone or a sidecar mtime, an output older than the claim
is not this line's, the GPU START is found whatever the job wrote after it, stamps are read in the zone they were written
in, a stale gpu.running is UNKNOWN, a crash is not a STALL (exit 3 is). Fixture files only: a fake
pool.running, canary table and provenance sidecars under tmp_path, a stub `ssh` that answers the liveness scan from a
file, GPU status/log from text files. No node, no GPU, no network."""
import base64
import calendar
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stall_watch as SW  # noqa: E402

NOW = 1_790_000_000.0                      # a fixed "now" (2026-09-21); every fixture time is relative to it
H = 3600
SHA = "a1b2c3d4e5f6a7b8c9d0a1b2c3d4e5f6a7b8c9d0"
OUT = "research/findings/raw/_x/out_s42.json"
RUNNER = "research.runners.fake_runner"


def stamp(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def job(runner=RUNNER, out=OUT, sha=SHA, seed=42, reason=r"corpus:\ nothing\ covers\ this\ mem_gb=8"):
    """A pool.running job field as pool_autodispatch.sh's pop_job writes it (POOL_CHECKED_REASON=<%q token> prefix)."""
    pin = "cd ~/derisk-pool/revisions/%s && " % sha if sha else ""
    return "POOL_CHECKED_REASON=%s %s.venv/bin/python -m %s --seed %d --out %s" % (reason, pin, runner, seed, out)


def b64(j):
    return base64.b64encode(j.encode()).decode()


class Env:
    """A scratch data root plus the stub ssh; `.run(*argv)` calls stall_watch.main and returns (rc, stdout lines)."""

    def __init__(self, tmp_path, capsys):
        self.root = tmp_path / "data"
        (self.root / "research" / "queue").mkdir(parents=True)
        self.stub = tmp_path / "stub"
        self.stub.mkdir()
        self.capsys = capsys
        self.running = self.root / "research" / "queue" / "pool.running"
        self.canaries = self.root / "research" / "queue" / "pool_canaries.tsv"
        ssh = tmp_path / "fake_ssh"
        ssh.write_text("#!/usr/bin/env bash\n"
                       "node=${@: -2:1}\n"                                  # args end: ... <node> <remote command>
                       f"[ -e {self.stub}/$node.down ] && exit 255\n"
                       f"[ -e {self.stub}/$node.live ] && cat {self.stub}/$node.live\n"
                       "echo POOL_LIVENESS_SCAN_OK\n")
        ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
        self.ssh = str(ssh)

    def claim(self, node, ago_h, j, live=True):
        with open(self.running, "a") as fh:
            fh.write("%s\t%s\t%s\n" % (stamp(NOW - ago_h * H), node, j))
        if live:
            with open(self.stub / (node + ".live"), "a") as fh:
                fh.write("JOB_B64=%s\n" % b64(j))

    def canary(self, module, wall, verdict="PASS", day="2026-10-10"):
        new = not self.canaries.exists()
        with open(self.canaries, "a") as fh:
            if new:
                fh.write("# recorded_at\tmodule\tsha\tverdict\tnode\tseed_kind\tvia\tevidence\n")
            fh.write("%sT00:00:00-04:00\t%s\t%s\t%s\tpool41\tgate\tpool_canary.sh run (plain ssh)\t"
                     "rc=0 wall=%ds json=1 errors=0 unreadable=0 tracebacks=0 id=x\n" % (day, module, SHA, verdict, wall))

    def sidecar(self, name, runner_path, started_ago_h, wall_s=None, ns=None, mtime_ago_h=None):
        """A provenance sidecar (+ artifact carrying wall_s, or v2 ns fields, or an mtime)."""
        art = self.root / "research" / "findings" / "raw" / "_prov" / (name + ".json")
        art.parent.mkdir(parents=True, exist_ok=True)
        doc = {"wall_s": wall_s} if wall_s else {}
        art.write_text(json.dumps(doc))
        side = {"run_id": name, "runner": runner_path, "started": time.strftime("%Y-%m-%dT%H:%M:%S",
                time.localtime(NOW - started_ago_h * H)), "artifact": str(art.relative_to(self.root))}
        if ns:
            side.update({"schema": "sim-run-provenance-v2", "started_utc_ns": 10**18, "ended_utc_ns": 10**18 + int(ns * 1e9)})
        p = Path(str(art) + ".prov.json")
        p.write_text(json.dumps(side, indent=1))
        if mtime_ago_h is not None:
            t = NOW - mtime_ago_h * H
            os.utime(p, (t, t))

    def run(self, *extra):
        argv = ["--root", str(self.root), "--ssh", self.ssh, "--gpu", "none", "--now", str(NOW),
                "--running", str(self.running), "--canaries", str(self.canaries)] + list(extra)
        rc = SW.main(argv)
        return rc, self.capsys.readouterr().out.splitlines()


@pytest.fixture()
def env(tmp_path, capsys):
    return Env(tmp_path, capsys)


def stall_lines(lines):
    return [ln for ln in lines if ln.startswith("STALL ")]


# ---------------------------------------------------------------- the required cases
def test_no_stall_prints_clean_and_exits_zero(env):
    env.canary(RUNNER, 2 * H)
    env.claim("pool41", 3, job())                     # 3 h in, limit 2 x 2 h = 4 h
    rc, lines = env.run()
    assert (rc, lines) == (0, ["stall_watch: CLEAN"])



def test_claims_on_a_retired_node_are_skipped(env, monkeypatch):
    # 2026-10-18 first run on the server: 85 claims from the retired 09-24 nodes pool1/pool2 printed two permanent
    # "unreachable" UNKNOWN lines every heartbeat. Only nodes in the live list can hold a stalled line.
    monkeypatch.setenv("POOL_NODES", "pool41 pool42")
    env.canary(RUNNER, 2 * H)
    env.claim("pool1", 50, job(), live=False)         # retired, unreachable, long overdue: skipped
    (env.stub / "pool1.down").write_text("")          # without the skip this prints "node pool1 unreachable"
    env.claim("pool41", 3, job())
    rc, lines = env.run()
    assert (rc, lines) == (0, ["stall_watch: CLEAN"])
    j = job()
    env.claim("pool42", 5, j)                         # a live node still stalls
    rc, lines = env.run()
    assert rc == SW.STALL_RC and len(stall_lines(lines)) == 1 and " node=pool42 " in stall_lines(lines)[0]

def test_stall_by_canary_wall(env):
    env.canary(RUNNER, 2 * H)
    j = job()
    env.claim("pool41", 5, j)                         # 5 h in > 4 h
    rc, lines = env.run()
    s = stall_lines(lines)
    assert rc == SW.STALL_RC == 3 and len(s) == 1 and "stall_watch: CLEAN" not in lines
    ln = s[0]
    assert " node=pool41 " in ln and " runner=%s " % RUNNER in ln
    assert " elapsed=5h00m " in ln and " expected=2h00m (canary 7200s 2026-10-10) " in ln
    assert " pin=%s " % SHA[:9] in ln and " out=%s" % OUT in ln


def test_a_failed_or_walless_canary_is_not_used(env):
    env.canary(RUNNER, 60, verdict="FAIL")            # a FAIL's wall says nothing about a healthy line
    env.claim("pool41", 7, job())                     # so the 6 h default applies: 7 h < 12 h
    assert env.run() == (0, ["stall_watch: CLEAN"])
    env.canary(RUNNER, 1 * H)                         # a later PASS is the one that counts: limit 2 h -> stall
    rc, lines = env.run()
    assert rc == 3 and "canary 3600s" in stall_lines(lines)[0]


def test_stall_by_the_longest_of_the_last_completed_runs(env):
    env.sidecar("old", "research/runners/fake_runner.py", started_ago_h=300, wall_s=1 * H)
    env.sidecar("new", "research/runners/fake_runner.py", started_ago_h=200, wall_s=2 * H)
    env.claim("pool42", 5, job())                     # the longest of the two recorded runs is 2 h: limit 4 h
    rc, lines = env.run()
    s = stall_lines(lines)
    assert rc == 3 and len(s) == 1
    assert " expected=2h00m (longest of 2 runs 2h00m " in s[0] and "(artifact wall_s)" in s[0]
    env.sidecar("v2", "research/runners/fake_runner.py", started_ago_h=100, ns=3 * H)          # a newer v2 sidecar: 3 h
    rc, lines = env.run()
    assert rc == 0 and lines == ["stall_watch: CLEAN"]                                          # 5 h < 2 x 3 h = 6 h


def test_the_newest_run_alone_is_not_the_expected_wall(env):
    """Review 2026-10-18: the newest run is often a smoke (11 min) of a runner whose real lines take 69 min."""
    for i, h in enumerate((1.1, 1.15, 1.0, 0.2)):                  # oldest ... a short smoke is the NEWEST
        env.sidecar("r%d" % i, "research/runners/fake_runner.py", started_ago_h=100 - 10 * i, wall_s=h * H)
    env.claim("pool41", 2.0, job())                                 # 2 h in: a smoke-only wall (12 min) would call it a stall
    assert env.run() == (0, ["stall_watch: CLEAN"])                 # limit 2 x 1.15 h = 2.3 h
    env.claim("pool42", 3.0, job(seed=43))
    rc, lines = env.run()
    assert rc == 3 and "node=pool42" in stall_lines(lines)[0] and "longest of 4 runs 1h09m" in stall_lines(lines)[0]


def test_a_sidecar_mtime_is_not_a_duration(env):
    """Review 2026-10-18: mtime minus `started` was right for 12% of real sidecars (zones, checkout stamps)."""
    env.sidecar("w", "research/runners/fake_runner.py", started_ago_h=50, mtime_ago_h=48.5)    # 1.5 h by the old rule
    env.claim("pool42", 4, job())                                                              # 4 h: old rule said STALL
    assert env.run() == (0, ["stall_watch: CLEAN"])                                            # the 6 h default holds
    env.claim("pool41", 13, job(seed=43))
    rc, lines = env.run()
    assert rc == 3 and "expected=6h00m (default 6h)" in stall_lines(lines)[0]


def test_a_canary_under_the_floor_is_a_startup_check_not_a_wall(env):
    env.canary(RUNNER, 4)                              # the real table's only wall= value is 4 s
    env.claim("pool41", 8 / 60, job())                 # 8 min in
    env.claim("pool42", 5, job(seed=43))               # 5 h in: under the 6 h default's 12 h limit
    assert env.run() == (0, ["stall_watch: CLEAN"])
    env.sidecar("a", "research/runners/fake_runner.py", started_ago_h=50, wall_s=2 * H)
    rc, lines = env.run()                              # a recorded 2 h run is a real wall; the 4 s canary does not shrink it
    assert rc == 3 and "node=pool42" in stall_lines(lines)[0] and "longest of 1 run 2h00m" in stall_lines(lines)[0]
    env.canary(RUNNER, 5 * H)                          # a real canary wall longer than any recorded run wins
    assert env.run()[0] == 0


def test_default_six_hours(env):
    env.claim("pool41", 11, job())                    # 11 h < 12 h
    assert env.run() == (0, ["stall_watch: CLEAN"])
    env.claim("pool42", 13, job(seed=43))             # 13 h > 12 h
    rc, lines = env.run()
    s = stall_lines(lines)
    assert rc == 3 and len(s) == 1 and "node=pool42" in s[0] and "expected=6h00m (default 6h)" in s[0]


def test_output_file_present_means_not_a_stall(env):
    env.canary(RUNNER, 1 * H)
    env.claim("pool41", 9, job())
    (env.root / OUT).parent.mkdir(parents=True)
    (env.root / OUT).write_text("{}")
    assert env.run() == (0, ["stall_watch: CLEAN"])
    # two --out files in one line: STALL until BOTH exist
    two = job() + " && .venv/bin/python -m %s --out research/findings/raw/_x/second.json" % RUNNER
    env.claim("pool42", 9, two)
    rc, lines = env.run()
    assert rc == 3 and "node=pool42" in stall_lines(lines)[0]
    (env.root / "research/findings/raw/_x/second.json").write_text("{}")
    assert env.run()[0] == 0


def test_an_output_older_than_the_claim_is_the_previous_runs_not_this_lines(env):
    """Review 2026-10-18: 79 of 81 re-queued --out paths already existed from the earlier run."""
    env.canary(RUNNER, 1 * H)
    env.claim("pool41", 9, job())
    out = env.root / OUT
    out.parent.mkdir(parents=True)
    out.write_text("{}")
    os.utime(out, (NOW - 30 * 24 * H, NOW - 30 * 24 * H))            # written 30 days ago, claim is 9 h old
    rc, lines = env.run()
    assert rc == 3 and "node=pool41" in stall_lines(lines)[0]
    os.utime(out, (NOW - 2 * H, NOW - 2 * H))                         # written during this claim
    assert env.run() == (0, ["stall_watch: CLEAN"])
    os.utime(out, (NOW - 9 * H - 60, NOW - 9 * H - 60))               # a minute before the claim: inside the clock slack
    assert env.run()[0] == 0


def test_a_claim_that_ended_is_a_failure_not_a_stall(env):
    env.canary(RUNNER, 1 * H)
    env.claim("pool41", 9, job(), live=False)         # not in the node's live JOB_B64 set
    assert env.run() == (0, ["stall_watch: CLEAN"])


# ---------------------------------------------------------------- unreachable sources: UNKNOWN, never a crash
def test_missing_pool_running_is_unknown_not_a_crash_and_not_clean(env):
    rc, lines = env.run()
    assert rc == 0 and len(lines) == 1 and lines[0].startswith("stall_watch: UNKNOWN pool: cannot read ")
    assert "stall_watch: CLEAN" not in lines


def test_unreachable_node_is_unknown_and_other_nodes_still_report(env):
    env.canary(RUNNER, 1 * H)
    env.claim("pool41", 9, job(), live=True)
    env.claim("pool42", 9, job(seed=43), live=True)
    (env.stub / "pool41.down").write_text("")
    rc, lines = env.run()
    assert rc == 3                                    # pool42's stall is still found
    assert len(stall_lines(lines)) == 1 and "node=pool42" in stall_lines(lines)[0]
    unk = [ln for ln in lines if "UNKNOWN" in ln]
    assert len(unk) == 1 and "node pool41 unreachable" in unk[0] and "1 claim(s)" in unk[0]


def test_only_unreachable_nodes_exit_zero_and_never_claim_clean(env):
    env.canary(RUNNER, 1 * H)
    env.claim("pool41", 9, job())
    (env.stub / "pool41.down").write_text("")
    rc, lines = env.run()
    assert rc == 0 and lines and all("UNKNOWN" in ln for ln in lines)


def test_missing_ssh_binary_and_garbage_files_do_not_crash(env, tmp_path):
    env.canary(RUNNER, 1 * H)
    env.claim("pool41", 9, job())
    rc = SW.main(["--root", str(env.root), "--ssh", str(tmp_path / "no-such-ssh"), "--gpu", "none", "--now", str(NOW),
                  "--running", str(env.running), "--canaries", str(env.canaries)])
    out = env.capsys.readouterr().out
    assert rc == 0 and "UNKNOWN node pool41 unreachable" in out
    env.running.write_text("not\ta\tvalid row\n\x00\xff garbage\n\n")
    env.canaries.write_text("\x00\xff\n")
    rc, lines = env.run()
    assert rc == 0 and lines == ["stall_watch: CLEAN"]
    assert env.run("--canaries", str(env.root / "nope.tsv"))[0] == 0               # a missing canary table: fall back


def test_a_dead_future_dated_claim_is_unknown(env):
    env.claim("pool41", -3, job())                    # 3 h in the future: a timezone mismatch
    rc, lines = env.run()
    assert rc == 0 and len(lines) == 1 and "dated in the future" in lines[0]


# ---------------------------------------------------------------- pool.running details
def test_only_the_latest_claim_of_a_repeated_job_counts_and_old_claims_age_out(env):
    env.canary(RUNNER, 1 * H)
    j = job()
    env.claim("pool41", 40, j)                         # an earlier attempt at the same job
    env.claim("pool41", 1, j)                          # re-queued 1 h ago: 1 h < 2 h
    assert env.run() == (0, ["stall_watch: CLEAN"])
    env.claim("pool42", 24 * 40, job(seed=44))         # 40 days: outside --max-age-days 30
    assert env.run() == (0, ["stall_watch: CLEAN"])
    rc, lines = env.run("--max-age-days", "60")
    assert rc == 3 and "node=pool42" in stall_lines(lines)[0]


def test_the_expected_wall_has_a_floor(env):
    env.sidecar("tiny", "research/runners/fake_runner.py", started_ago_h=50, wall_s=10)       # a 10 s run
    env.claim("pool41", 8 / 60, job())                 # 8 min in: under 2 x 300 s
    assert env.run() == (0, ["stall_watch: CLEAN"])
    env.claim("pool42", 12 / 60, job(seed=43))         # 12 min in: over
    rc, lines = env.run()
    assert rc == 3 and "expected=5m (longest of 1 run 0m" in stall_lines(lines)[0] and "+floor)" in stall_lines(lines)[0]


def test_expected_wall_is_computed_once_per_runner(env, monkeypatch):
    """Review 2026-10-18: it was recomputed per claim (143 sidecar scans, 30 s cold)."""
    calls = []
    real = SW.Provenance.last_runs
    monkeypatch.setattr(SW.Provenance, "last_runs", lambda self, p, n=1: calls.append(p) or real(self, p, n))
    for i in range(12):
        env.claim("pool41", 20 + i, job(seed=100 + i))
    rc, lines = env.run()
    assert rc == 3 and len(stall_lines(lines)) == 12 and len(calls) == 1


def test_runner_pin_and_out_are_read_from_the_job_text(env):
    f = SW.features(job(runner="research.runners.lbf_rows.word_role_parse", out="a/b.json", sha=SHA[:10]))
    assert (f["module"], f["runner_path"], f["pin"], f["outs"]) == (
        "research.runners.lbf_rows.word_role_parse", "research/runners/lbf_rows/word_role_parse.py", SHA[:10], ["a/b.json"])
    f = SW.features("cd ~/derisk-pool/revisions/%s-1760000000 && python research/runners/x_y.py --out=o.json;" % SHA)
    assert (f["module"], f["pin"], f["outs"]) == ("research.runners.x_y", SHA, ["o.json"])   # a GPU worktree's -<epoch>
    f = SW.features(r"POOL_CHECKED_REASON=$'a\'b c' tools/memcap.sh 8 -- python -m research.runners.zz --seed 1")
    assert f["module"] == "research.runners.zz" and f["pin"] is None and f["outs"] == []
    assert SW.features("pytest -x tests/test_a.py")["runner"] == "pytest"


def test_unpinned_line_prints_pin_none_and_age_na(env):
    env.claim("pool41", 13, job(sha=None))
    rc, lines = env.run()
    assert rc == 3 and " pin=none pin_age_commits=n/a " in stall_lines(lines)[0]


def test_pin_age_counts_commits_on_main_since_the_pin(env, tmp_path):
    g = lambda *a: subprocess.run(["git", "-C", str(env.root), "-c", "commit.gpgsign=false", *a], check=True,  # noqa: E731
                                  capture_output=True, text=True,
                                  env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x.invalid",
                                           GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@x.invalid")).stdout.strip()
    g("init", "-q", "-b", "main")
    shas = []
    for i in range(4):
        (env.root / ("f%d" % i)).write_text(str(i))
        g("add", "f%d" % i)
        g("commit", "-q", "-m", "c%d" % i)
        shas.append(g("rev-parse", "HEAD"))
    env.claim("pool41", 13, job(sha=shas[0]))                    # 3 commits behind main
    env.claim("pool42", 13, job(sha="deadbeef" * 5, seed=43))    # not in this repository
    rc, lines = env.run()
    by_node = {ln.split("node=")[1].split()[0]: ln for ln in stall_lines(lines)}
    assert rc == 3 and " pin_age_commits=3 " in by_node["pool41"] and " pin=%s " % shas[0][:9] in by_node["pool41"]
    assert " pin_age_commits=unknown " in by_node["pool42"]


# ---------------------------------------------------------------- the GPU queue
GPU_STATUS_RUNNING = "== gpu_queue ==\nstate: running\ncurrent: %s (pid 4242)\nqueued: 0\ndispatcher: up (pid 1)\n"
GPU_STATUS_IDLE = "== gpu_queue ==\nstate: running\ncurrent: (idle)\nqueued: 0\n"


def gpu(env, tmp_path, status, log, *extra):
    (tmp_path / "gpu_status.txt").write_text(status)
    (tmp_path / "gpu_log.txt").write_text(log)
    return env.run("--no-pool", "--gpu", "auto", "--gpu-status-cmd", "cat %s" % (tmp_path / "gpu_status.txt"),
                   "--gpu-log-cmd", "cat %s" % (tmp_path / "gpu_log.txt"), *extra)


def test_gpu_current_job_past_twice_its_expected_wall_is_a_stall(env, tmp_path):
    gj = "cd ~/derisk-pool/revisions/%s-1789000000 && .venv/bin/python -m %s --seed 7 --out %s" % (SHA, RUNNER, OUT)
    log = ("%s START: some earlier job\nmodel noise\n%s DONE(rc=0): some earlier job\n%s START: %s\nstep 1 of 99\n"
           % (stamp(NOW - 40 * H), stamp(NOW - 39 * H), stamp(NOW - 14 * H), gj))
    rc, lines = gpu(env, tmp_path, GPU_STATUS_RUNNING % gj[:100], log)
    s = stall_lines(lines)
    assert rc == 3 and len(s) == 1 and " node=gpu " in s[0] and " elapsed=14h00m " in s[0]
    assert " expected=6h00m (default 6h) " in s[0] and " pin=%s " % SHA[:9] in s[0]
    env.canary(RUNNER, 8 * H)                                         # a canary wall of 8 h: 14 h < 16 h
    assert gpu(env, tmp_path, GPU_STATUS_RUNNING % gj[:100], log)[0] == 0


def test_gpu_idle_or_finished_or_unreadable(env, tmp_path):
    gj = "cd /w && python -m %s --out %s" % (RUNNER, OUT)
    done = "%s START: %s\n%s DONE(rc=0): %s\n" % (stamp(NOW - 30 * H), gj, stamp(NOW - 20 * H), gj)
    assert gpu(env, tmp_path, GPU_STATUS_IDLE, "%s START: %s\n" % (stamp(NOW - 30 * H), gj)) == (0, ["stall_watch: CLEAN"])
    rc, lines = gpu(env, tmp_path, GPU_STATUS_RUNNING % gj, done)     # running per status, but its START was DONE'd
    assert rc == 0 and "has no START line" in lines[0]
    rc, lines = env.run("--no-pool", "--gpu", "auto", "--gpu-status-cmd", "false")
    assert rc == 0 and lines == ["stall_watch: UNKNOWN gpu: GPU status command failed (rc 1)"]
    rc, lines = env.run("--no-pool", "--gpu", "auto", "--gpu-status-cmd", str(tmp_path / "no-such-command"))
    assert rc == 0 and len(lines) == 1 and lines[0].startswith("stall_watch: UNKNOWN gpu: GPU status command failed")
    rc, lines = gpu(env, tmp_path, "garbage with no current line\n", "")
    assert rc == 0 and "no `current:` line" in lines[0]


def test_gpu_local_queue_files(env, tmp_path, monkeypatch):
    q = tmp_path / "gpuq"
    q.mkdir()
    gj = "cd /w && python -m %s --out %s" % (RUNNER, OUT)
    (q / "gpu.running").write_text("%d\t%s\n" % (os.getpid(), gj))                   # a live pid: this test process
    (q / "gpu_queue.log").write_text("%s START: %s\n" % (stamp(NOW - 14 * H), gj))
    monkeypatch.setenv("GPU_QUEUE_DIR", str(q))
    rc, lines = env.run("--no-pool", "--gpu", "auto")
    assert rc == 3 and " node=gpu " in stall_lines(lines)[0]
    (q / "gpu.running").unlink()                                   # the job ended: idle
    assert env.run("--no-pool", "--gpu", "auto")[0] == 0
    monkeypatch.setenv("GPU_QUEUE_DIR", str(tmp_path / "no-such-dir"))
    rc, lines = env.run("--no-pool", "--gpu", "auto")
    assert rc == 0 and "UNKNOWN gpu: no GPU queue dir" in lines[0]


# ---------------------------------------------------------------- the command line
def test_cli_exit_codes_and_single_clean_line(tmp_path):
    root = tmp_path / "data"
    (root / "research" / "queue").mkdir(parents=True)
    running = root / "research" / "queue" / "pool.running"
    running.write_text("")
    base = [sys.executable, str(ROOT / "tools" / "stall_watch.py"), "--root", str(root), "--gpu", "none",
            "--running", str(running)]
    r = subprocess.run(base, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout == "stall_watch: CLEAN\n"
    r = subprocess.run(base + ["--running", str(tmp_path / "nope")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "UNKNOWN pool" in r.stdout and "CLEAN" not in r.stdout
    r = subprocess.run(base + ["--bogus"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2                                        # argparse's own usage error, not a crash trace


# ---------------------------------------------------------------- review 2026-10-18: the findings, one test each
class zone:
    """Run a block with this process's TZ set (and put it back): the reader's zone in the zone tests."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.old = os.environ.get("TZ")
        os.environ["TZ"] = self.name
        time.tzset()

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = self.old
        time.tzset()


def utc_stamp(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))


def test_pool_stamps_are_read_in_the_zone_they_were_written_in(env):
    """The dispatcher writes UTC; the reader runs in New York. Explicit --pool-tz, and inference from the file's mtime."""
    real_now = time.time()
    j = job()
    wrote = real_now - 20 * H
    env.running.write_text("%s\tpool41\t%s\n" % (utc_stamp(wrote), j))
    os.utime(env.running, (wrote, wrote))                   # the last row was appended at its stamp, as the dispatcher does
    (env.stub / "pool41.live").write_text("JOB_B64=%s\n" % b64(j))
    with zone("America/New_York"):
        for extra in (("--pool-tz", "UTC"), ()):             # told, then inferred
            rc = SW.main(["--root", str(env.root), "--ssh", env.ssh, "--gpu", "none", "--now", str(real_now),
                          "--running", str(env.running), "--canaries", str(env.canaries), *extra])
            out = env.capsys.readouterr().out
            assert rc == 3 and " elapsed=20h00m " in out, (extra, out)
        # a file touched long after its last row says nothing about the zone: no shift is invented
        os.utime(env.running, (real_now, real_now))
        rc = SW.main(["--root", str(env.root), "--ssh", env.ssh, "--gpu", "none", "--now", str(real_now),
                      "--running", str(env.running), "--canaries", str(env.canaries)])
        out = env.capsys.readouterr().out
        assert " elapsed=20h00m " not in out and (" elapsed=15h" in out or " elapsed=16h" in out), out


def test_infer_shift_only_trusts_whole_hours_close_to_the_mtime():
    assert SW.infer_shift(1000.0, 1000.0) == 0.0 and SW.infer_shift(1100.0, 1000.0) == 0.0
    assert SW.infer_shift(1000.0 + 4 * H, 1000.0) == 4 * H and SW.infer_shift(1000.0 - 5 * H, 1000.0) == -5 * H
    assert SW.infer_shift(1000.0 + 4 * H + 30, 1000.0) == 4 * H
    assert SW.infer_shift(1000.0 + 4 * H + 300, 1000.0) == 0.0         # 5 min off a whole hour: a touched file
    assert SW.infer_shift(1000.0 + 5.5 * H, 1000.0) == 0.0             # half-hour zones are never inferred
    assert SW.infer_shift(1000.0 + 20 * H, 1000.0) == 0.0              # beyond any zone


def gpu_events(*pairs):
    """gpu_remote.sh events output: the START/DONE lines then the `now:` line of a workstation clock in UTC."""
    lines = ["%s %s: %s" % (utc_stamp(ts), kind, jb) for ts, kind, jb in pairs]
    real = NOW
    lines.append("now: %d %s" % (real, utc_stamp(real)))
    return "\n".join(lines) + "\n"


def test_gpu_remote_stamps_use_the_workstations_own_clock_line(env, tmp_path):
    gj = "cd /w && python -m %s --out %s" % (RUNNER, OUT)
    ev = gpu_events((NOW - 14 * H, "START", gj))
    with zone("America/New_York"):                                  # the reader's zone differs from the log's (UTC)
        rc, lines = gpu(env, tmp_path, GPU_STATUS_RUNNING % gj, ev)
    assert rc == 3 and " elapsed=14h00m " in stall_lines(lines)[0]
    with zone("Asia/Tokyo"):
        rc, lines = gpu(env, tmp_path, GPU_STATUS_RUNNING % gj, ev)
    assert rc == 3 and " elapsed=14h00m " in stall_lines(lines)[0]
    rc, lines = gpu(env, tmp_path, GPU_STATUS_RUNNING % gj, ev, "--gpu-tz", "UTC")   # explicit zone: same answer
    assert rc == 3 and " elapsed=14h00m " in stall_lines(lines)[0]


def test_gpu_start_far_back_in_a_chatty_log_is_still_found(env, tmp_path, monkeypatch):
    """Review 2026-10-18: a job that wrote 800k lines after its START pushed it out of a 3000-line tail."""
    q = tmp_path / "gpuq"
    q.mkdir()
    gj = "cd /w && python -m %s --out %s" % (RUNNER, OUT)
    (q / "gpu.running").write_text("%d\t%s\n" % (os.getpid(), gj))
    with open(q / "gpu_queue.log", "w") as fh:
        fh.write("%s START: older job\n%s DONE(rc=0): older job\n" % (stamp(NOW - 50 * H), stamp(NOW - 40 * H)))
        fh.write("%s START: %s\n" % (stamp(NOW - 14 * H), gj))
        for i in range(60000):
            fh.write("step %d of 99999 loss 0.123456 some chatty training output\n" % i)
    monkeypatch.setenv("GPU_QUEUE_DIR", str(q))
    rc, lines = env.run("--no-pool", "--gpu", "auto")
    assert rc == 3 and " elapsed=14h00m " in stall_lines(lines)[0], lines
    # the reader itself, with a chunk smaller than a line, pairs DONE with START from the end
    text = SW.local_events(str(q / "gpu_queue.log"), gj[:100], chunk=64)
    assert [ln.split(" ", 2)[2][:5] for ln in text.splitlines()] == ["START"]
    assert SW.gpu_current("current: %s (pid 1)" % gj[:100], text, lambda t: SW.parse_ts(t))[0] == gj
    two = tmp_path / "two.log"
    two.write_text("%s START: a\n%s DONE(rc=0): a\nnoise\n%s START: a\n%s START: b\n"
                   % (stamp(1), stamp(2), stamp(3), stamp(4)))
    got = SW.local_events(str(two), "a", chunk=16).splitlines()                 # stops at the OPEN start of a (not the paired one)
    assert [g.split(" ", 2)[2] for g in got] == ["START: a", "START: b"]


def test_gpu_remote_events_subcommand_survives_a_chatty_log(tmp_path):
    """The workstation side: gpu_remote.sh events greps START/DONE lines out of a log whose START is 100k lines back."""
    import shutil
    home = tmp_path / "home"
    q = home / "Projects" / "sim" / "research" / "queue"
    q.mkdir(parents=True)
    with open(q / "gpu_queue.log", "w") as fh:
        fh.write("2026-10-17 01:00:00 START: cd /w && python -m x --out y\n")
        for i in range(100000):
            fh.write("noise line %d\n" % i)
        fh.write("2026-10-17 02:00:00 DONE(rc=0): cd /w && python -m x --out y\n2026-10-18 03:00:00 START: job2\n")
        fh.write("noise\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ssh = bindir / "ssh"
    ssh.write_text("#!/usr/bin/env bash\nHOME=%s exec bash -s\n" % home)
    ssh.chmod(0o755)
    res = subprocess.run(["bash", str(ROOT / "tools" / "gpu_remote.sh"), "events", "2"], capture_output=True, text=True,
                         env={**os.environ, "PATH": "%s:%s" % (bindir, os.environ["PATH"])}, timeout=60)
    assert res.returncode == 0, res.stderr
    out = res.stdout.splitlines()
    assert out[0].endswith("DONE(rc=0): cd /w && python -m x --out y") and out[1].endswith("START: job2")
    assert out[2].startswith("now: ") and SW.NOW_RE.search(res.stdout)
    assert shutil.which("grep")
    assert subprocess.run(["bash", str(ROOT / "tools" / "gpu_remote.sh"), "events", "x"], capture_output=True,
                          timeout=60).returncode == 2


def test_a_dead_gpu_pid_or_a_down_dispatcher_is_a_stale_record_not_a_stall(env, tmp_path, monkeypatch):
    q = tmp_path / "gpuq"
    q.mkdir()
    gj = "cd /w && python -m %s --out %s" % (RUNNER, OUT)
    dead = subprocess.Popen(["true"])
    dead.wait()
    (q / "gpu.running").write_text("%d\t%s\n" % (dead.pid, gj))
    (q / "gpu_queue.log").write_text("%s START: %s\n" % (stamp(NOW - 20 * H), gj))
    monkeypatch.setenv("GPU_QUEUE_DIR", str(q))
    rc, lines = env.run("--no-pool", "--gpu", "auto")
    assert rc == 0 and len(lines) == 1 and "UNKNOWN gpu: stale gpu.running" in lines[0] and "CLEAN" not in lines[0]
    down = GPU_STATUS_RUNNING % gj + "dispatcher: DOWN (run: tools/gpu_queue.sh start)\n"
    rc, lines = gpu(env, tmp_path, down, "%s START: %s\n" % (stamp(NOW - 20 * H), gj))
    assert rc == 0 and "UNKNOWN gpu: stale gpu.running" in lines[0]


def test_stale_queue_files_do_not_make_the_server_read_a_gpu_it_does_not_have(env, tmp_path, monkeypatch):
    """Review 2026-10-18: any stale gpu_queue.log in the data root switched 'auto' to local and hid the workstation."""
    q = env.root / "research" / "queue"
    (q / "gpu_queue.log").write_text("%s START: old\n" % stamp(NOW - 99 * H))
    (q / "gpu_queue.dpid").write_text("999999\n")
    calls = []

    def fake(argv, timeout, cwd=None):
        calls.append(argv)
        return 0, ("current: (idle)\n" if argv[-1] == "status" else "now: %d %s\n" % (NOW, stamp(NOW))), ""

    monkeypatch.setattr(SW, "run_cmd", fake)
    monkeypatch.delenv("GPU_QUEUE_DIR", raising=False)
    rc = SW.main(["--root", str(env.root), "--no-pool", "--gpu", "auto", "--now", str(NOW)])
    assert rc == 0 and any(a[-2:] == ["events", "200"] or a[-1] == "status" for a in calls), calls
    assert any("gpu_remote.sh" in " ".join(a) for a in calls)
    (q / "gpu.running").write_text("%d\tx\n" % os.getpid())                  # a live job here: now it IS local
    calls.clear()
    SW.main(["--root", str(env.root), "--no-pool", "--gpu", "auto", "--now", str(NOW)])
    assert not calls


# ---------------------------------------------------------------- the exit codes
def test_exit_codes_distinguish_a_stall_from_a_crash(tmp_path):
    root = tmp_path / "data"
    (root / "research" / "queue").mkdir(parents=True)
    running = root / "research" / "queue" / "pool.running"
    running.write_text("%s\tpool41\t%s\n" % (stamp(time.time() - 30 * H), job(sha=None)))
    (tmp_path / "fake_ssh").write_text("#!/usr/bin/env bash\necho JOB_B64=%s\necho POOL_LIVENESS_SCAN_OK\n" % b64(job(sha=None)))
    (tmp_path / "fake_ssh").chmod(0o755)
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "stall_watch.py"), "--root", str(root), "--gpu", "none",
                        "--running", str(running), "--ssh", str(tmp_path / "fake_ssh")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 3 and r.stdout.startswith("STALL ")                  # a stall is 3, never 1
    crash = subprocess.run([sys.executable, "-c", "import nonexistent_module_for_review"], capture_output=True, text=True)
    assert crash.returncode == 1                                                  # a traceback is 1: not a stall code
