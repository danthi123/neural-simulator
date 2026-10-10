"""The pool-job sandbox (2026-09-30 security review, HIGH #2: "pool-submit is a LAN foothold").

Three layers, all hermetic (no real pool node, no real queue):
  1. tools/pool_autodispatch.sh with a STUB ssh: the POOL_SANDBOX switch (on refuses an unready node and leaves the
     job queued; auto launches unsandboxed with a loud warning; off never probes), agent-marked lines are never
     popped for an unsandboxed launch, the sandboxed launch carries tools/pool_job_sandbox.sh as its wrapper, and the
     mode record the live loop publishes.
  2. tools/pool_job_sandbox.sh with a FAKE bwrap / systemd-run / timeout that record their argv: the exact binds (the
     revision read-only, only research/findings/raw writable -- an agent line only its _agent_jobs/<id>), the outer
     memory scope, and every refusal (rc 126, never an unsandboxed fallback).
  3. The same wrapper with the REAL bwrap (skipped where it cannot run): a job really runs, its output lands, and
     from inside the same sandbox the LAN (192.168.0.10:445) and the internet are unreachable, sudo fails, the
     revision and everything outside the output area are read-only, and ~/.ssh does not exist.
"""
from __future__ import annotations

import base64
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = ROOT / "tools" / "pool_autodispatch.sh"
WRAPPER = ROOT / "tools" / "pool_job_sandbox.sh"
SHA = "0123456789abcdef0123456789abcdef01234567"
AGENT_ID = "0123456789ab-89abcdef"


def _exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


# ------------------------------------------------------------------------------------------ 1. the dispatcher

def _dispatch_env(tmp: Path, probe_answer: str, queue_lines: list[str], mode: str | None) -> tuple[dict, Path, Path]:
    """A stub ssh for fill_node: the node_is_idle metrics probe says idle; the sandbox probe answers `probe_answer`;
    an `ssh -f` launch is only logged (the job is never run)."""
    bin_dir = tmp / "bin"
    log = tmp / "ssh.log"
    log.write_text("")
    _exe(bin_dir / "ssh", f"""echo "$*" >> "{log}"
case "$*" in
  *MemAvailable*) echo "8 1 0 20 1 32"; exit 0 ;;
  *POOL_SANDBOX_ACTION=probe*) echo "{probe_answer}"; exit 0 ;;
esac
exit 0
""")
    now = int(time.time())
    queue = tmp / "pool.queue"
    queue.write_text("".join(f"{now}\t{line}\n" for line in queue_lines))
    (tmp / "tmp").mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "TMPDIR": str(tmp / "tmp"),
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp / "pool.running"),
        "POOL_RESERVATIONS_PATH": str(tmp / "resv"),
        "POOL_EXTRA_NODES_FILE": str(tmp / "extra_nodes"),
        "POOL_SSH_CONFIG": str(tmp / "does-not-exist"),
        "POOL_RUNNER_MEM_PATH": str(tmp / "runner_mem.tsv"),
        "POOL_DISPATCH_LAUNCH_SLEEP": "0",
        "POOL_SKIP_FAST_FAIL_CHECK": "1",
        "POOL_JOBS_PER_NODE": "3",
    }
    env.pop("POOL_SANDBOX", None)
    if mode is not None:
        env["POOL_SANDBOX"] = mode
    return env, log, queue


def _fill(tmp: Path, probe_answer: str, queue_lines: list[str], mode: str | None):
    env, log, queue = _dispatch_env(tmp, probe_answer, queue_lines, mode)
    res = subprocess.run(["bash", str(DISPATCHER), "--fill-node", "pool1"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=60)
    launches = [ln for ln in log.read_text().splitlines() if "JOB_B64=" in ln]
    probes = [ln for ln in log.read_text().splitlines() if "POOL_SANDBOX_ACTION=probe" in ln]
    return res, launches, probes, queue


def _launched_job(launch: str) -> str:
    return base64.b64decode(launch.split("JOB_B64='")[1].split("'")[0]).decode()


def _launched_wrapper(launch: str) -> str:
    return base64.b64decode(launch.split("WRAPPER_B64='")[1].split("'")[0]).decode()


OWNER_LINE = f"cd ~/derisk-pool/revisions/{SHA} && echo owner  #checked:owner line mem_gb=1"
AGENT_LINE = (f"AGENT_JOB_SRC=sim-agent cd ~/derisk-pool/revisions/{SHA} && mkdir -p "
              f"research/findings/raw/_agent_jobs/{AGENT_ID} && echo agent  #checked:agent line mem_gb=1")
READY = "POOL_SANDBOX_READY bwrap=0.6.1"
NOT_READY = "POOL_SANDBOX_NOT_READY reason=bwrap-not-installed"


def test_render_sandboxed_launch_swaps_only_the_wrapper(tmp_path: Path) -> None:
    job = "cd ~/derisk-pool/revisions/%s && echo hi" % SHA
    env = {**os.environ, "HOME": str(tmp_path)}
    legacy = subprocess.run(["bash", str(DISPATCHER), "--render-remote-command", job], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=True).stdout
    sandboxed = subprocess.run(["bash", str(DISPATCHER), "--render-remote-command", job, "sandbox"], cwd=ROOT,
                               env=env, capture_output=True, text=True, check=True).stdout
    assert "POOL_SANDBOX_ACTION" not in legacy
    assert "POOL_SANDBOX_ACTION='run' setsid bash -c" in sandboxed
    assert _launched_wrapper(sandboxed) == WRAPPER.read_text()
    assert _launched_job(sandboxed) == _launched_job(legacy) == job
    for token in ("cd ~/derisk-pool/sim && POOL_JOB_ID='", "POOL_JOB_MEM_GB='", "OMP_NUM_THREADS='1'",
                  "NUMEXPR_NUM_THREADS='1'", "</dev/null >/dev/null 2>&1 & exit 0"):
        assert token in legacy and token in sandboxed


def test_on_mode_gives_an_unready_node_nothing_and_keeps_the_job_queued(tmp_path: Path) -> None:
    res, launches, probes, queue = _fill(tmp_path, NOT_READY, [OWNER_LINE], "on")
    assert launches == [], "POOL_SANDBOX=on launched on a node that cannot sandbox"
    assert len(probes) == 1
    assert "⛔ POOL_SANDBOX=on: pool1 is not sandbox-ready" in res.stderr
    assert "echo owner" in queue.read_text(), "the job was popped (and so lost) instead of staying queued"


def test_on_mode_launches_sandboxed_on_a_ready_node(tmp_path: Path) -> None:
    res, launches, probes, queue = _fill(tmp_path, READY, [OWNER_LINE, AGENT_LINE], "on")
    assert len(launches) == 2 and len(probes) == 1, (launches, res.stderr)   # probed once per node per cycle
    for launch in launches:
        assert "POOL_SANDBOX_ACTION='run'" in launch
        assert _launched_wrapper(launch) == WRAPPER.read_text()
    assert "UNSANDBOXED" not in res.stderr
    assert "echo owner" not in queue.read_text() and "echo agent" not in queue.read_text()


def test_auto_mode_launches_unsandboxed_with_a_loud_warning(tmp_path: Path) -> None:
    res, launches, probes, _ = _fill(tmp_path, NOT_READY, [OWNER_LINE], None)   # auto is the default
    assert len(launches) == 1 and len(probes) == 1
    assert "POOL_SANDBOX_ACTION" not in launches[0]
    assert "bash -c \"$job\" > autodispatch.out" in _launched_wrapper(launches[0])   # the legacy wrapper
    assert "⚠ UNSANDBOXED launch on pool1 (POOL_SANDBOX=auto)" in res.stderr
    assert "is not sandbox-ready" in res.stderr


def test_auto_mode_sandboxes_a_ready_node(tmp_path: Path) -> None:
    res, launches, _, _ = _fill(tmp_path, READY, [OWNER_LINE], "auto")
    assert len(launches) == 1 and "POOL_SANDBOX_ACTION='run'" in launches[0]
    assert "UNSANDBOXED" not in res.stderr


def test_off_mode_never_probes(tmp_path: Path) -> None:
    res, launches, probes, _ = _fill(tmp_path, READY, [OWNER_LINE], "off")
    assert probes == [] and len(launches) == 1
    assert "POOL_SANDBOX_ACTION" not in launches[0]
    assert "UNSANDBOXED launch on pool1 (POOL_SANDBOX=off)" in res.stderr


@pytest.mark.parametrize("mode", [None, "off"])
def test_an_agent_line_is_never_popped_for_an_unsandboxed_launch(tmp_path: Path, mode: str | None) -> None:
    res, launches, _, queue = _fill(tmp_path, NOT_READY, [AGENT_LINE, OWNER_LINE], mode)
    assert [_launched_job(ln) for ln in launches] and all("echo agent" not in _launched_job(ln) for ln in launches)
    assert any("echo owner" in _launched_job(ln) for ln in launches), "the owner line behind it was not served"
    assert "echo agent" in queue.read_text(), "the agent line was dropped instead of left queued"


def test_an_unknown_mode_fails_closed_to_on(tmp_path: Path) -> None:
    env = {**os.environ, "POOL_SANDBOX": "yes-please"}
    res = subprocess.run(["bash", str(DISPATCHER), "--sandbox-mode"], cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True)
    assert res.stdout.strip() == "on" and "fail closed" in res.stderr


def test_the_dispatcher_publishes_its_effective_mode(tmp_path: Path) -> None:
    """The live loop's per-cycle mode record (write_sandbox_state). Its first consumer, tools/agent_gateway.sh, was
    retired with the sim-agent harness (07981b7de); the record stays as the dispatcher's observable mode."""
    state = tmp_path / "research" / "queue" / ".pool_sandbox_state"
    for mode in ("auto", "off", "on"):
        env = {**os.environ, "POOL_SANDBOX": mode, "POOL_SANDBOX_STATE_PATH": str(state)}
        subprocess.run(["bash", str(DISPATCHER), "--write-sandbox-state"], cwd=ROOT, env=env, check=True)
        assert state.read_text().startswith("mode=%s ts=" % mode)


def test_check_fast_fail_reports_a_sandbox_refusal_loudly(tmp_path: Path) -> None:
    job = "echo refused"
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "ssh", f'printf "v2\\t%s\\t126\\t%s\\n" "$(date +%s)" "{_b64(job)}"\n')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "POOL_SSH_CONFIG": str(tmp_path / "none")}
    res = subprocess.run(["bash", str(DISPATCHER), "--check-fast-fail", "pool1", job, str(int(time.time()) - 5)],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert "FAST-FAIL: pool1 rc=126" in res.stderr and "sandbox refused" in res.stderr


# ---------------------------------------------------------------------- 2. the wrapper, with a fake bwrap

def _node(tmp: Path) -> tuple[Path, Path]:
    """A fake node HOME: ~/.ssh with a key, the shared checkout, one provisioned revision, sim-data."""
    home = tmp / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE KEY\n")
    (home / "derisk-pool" / "sim" / "research" / "findings" / "raw").mkdir(parents=True)
    rev = home / "derisk-pool" / "revisions" / SHA
    (rev / "research" / "findings" / "raw").mkdir(parents=True)
    (rev / "tools").mkdir()
    (rev / ".venv" / "bin").mkdir(parents=True)
    (rev / ".venv" / "bin" / "python").symlink_to(shutil.which("python3") or "/usr/bin/python3")
    for name in ("memcap.sh", "mem_ok.sh"):
        shutil.copy(ROOT / "tools" / name, rev / "tools" / name)
    (home / "Projects" / "sim-data" / "knowledge_bundles").mkdir(parents=True)
    return home, rev


def _fakes(tmp: Path, systemd_ok: bool = True) -> tuple[Path, Path]:
    """Recording stand-ins: bwrap (argv one per line, then exit 0 without running anything), systemd-run and timeout
    (record, then exec what follows), systemctl (answers is-system-running)."""
    bin_dir = tmp / "fakebin"
    rec = tmp / "rec"
    rec.mkdir()
    _exe(bin_dir / "bwrap", f"""[ "${{1:-}}" = --version ] && {{ echo "bubblewrap 0.6.1"; exit 0; }}
printf '%s\\n' "$@" > "{rec}/bwrap.argv"
exit 0
""")
    _exe(bin_dir / "systemd-run", f"""printf '%s\\n' "$@" > "{rec}/systemd-run.argv"
{'' if systemd_ok else 'exit 1'}
while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done; shift
[ $# -gt 0 ] || exit 0
exec "$@"
""")
    _exe(bin_dir / "systemctl", "exit %d\n" % (0 if systemd_ok else 1))
    _exe(bin_dir / "timeout", f"""printf '%s\\n' "$@" > "{rec}/timeout.argv"
while [ $# -gt 0 ]; do case "$1" in --*) shift ;; *) shift; break ;; esac; done
exec "$@"
""")
    return bin_dir, rec


def _run_wrapper(home: Path, job: str, bin_dir: Path | None, bwrap: str | None = None,
                 action: str = "run") -> tuple[subprocess.CompletedProcess, Path]:
    cwd = home / "derisk-pool" / "sim"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("POOL_", "XDG_RUNTIME_DIR"))}
    env.update({"HOME": str(home), "JOB_B64": _b64(job), "POOL_SANDBOX_ACTION": action,
                "POOL_JOB_ID": "1790000000000-42", "POOL_JOB_MEM_GB": "7", "OMP_NUM_THREADS": "1"})
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
    if bwrap is not None:
        env["POOL_SANDBOX_BWRAP"] = bwrap
    res = subprocess.run(["bash"], input=WRAPPER.read_text(), cwd=cwd, env=env, capture_output=True, text=True,
                         timeout=120)
    return res, cwd


def _status(cwd: Path) -> tuple[str, str]:
    row = (cwd / "job_status.log").read_text().splitlines()[-1].split("\t")
    assert row[0] == "v2" and row[1].isdigit()
    return row[2], base64.b64decode(row[3]).decode()


def _pairs(argv: list[str], flag: str) -> list[tuple[str, str]]:
    return [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == flag]


def test_owner_line_binds_the_revision_read_only_and_only_its_raw_writable(tmp_path: Path) -> None:
    home, rev = _node(tmp_path)
    bin_dir, rec = _fakes(tmp_path)
    job = (f"cd ~/derisk-pool/revisions/{SHA} && bash tools/mem_ok.sh 7 1 && mkdir -p research/findings/raw/x "
           f"&& bash tools/memcap.sh 9 -- env SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.r "
           f"--out research/findings/raw/x/y.json")
    res, cwd = _run_wrapper(home, job, bin_dir, bwrap=str(bin_dir / "bwrap"))
    assert _status(cwd) == ("0", job), res.stderr
    argv = (rec / "bwrap.argv").read_text().splitlines()
    for flag in ("--unshare-all", "--die-with-parent", "--new-session", "--clearenv"):
        assert flag in argv
    assert not any(a.startswith("--share") for a in argv), "a namespace was re-shared (e.g. --share-net)"
    assert argv[-4:] == ["--", "/bin/bash", "-c", job]
    assert ("/proc" in argv[argv.index("--proc") + 1]) and argv[argv.index("--dev") + 1] == "/dev"
    tmpfs = [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"]
    assert tmpfs == ["/tmp", str(home)]
    assert _pairs(argv, "--bind") == [(str(rev / "research/findings/raw"),) * 2], "writable set is not just raw/"
    ro = dict(_pairs(argv, "--ro-bind"))
    assert ro[str(rev)] == str(rev) and ro["/usr"] == "/usr"
    assert not any(src in ("/", str(home), str(home / ".ssh"), str(home / "derisk-pool"),
                           str(home / "derisk-pool" / "sim")) for src in ro), ro
    assert (str(home / "Projects" / "sim-data"), str(home / "Projects" / "sim-data")) in _pairs(argv, "--ro-bind-try")
    env = dict(_pairs(argv, "--setenv"))
    assert env["PATH"].startswith("/opt/pool-sandbox/bin:") and env["HOME"] == str(home)
    assert env["POOL_JOB_ID"] == "1790000000000-42" and env["POOL_SANDBOX"] == "1"
    assert env["POOL_SANDBOX_MEMCAP_ENFORCED"] == "1" and env["POOL_SANDBOX_OUTER_MEMORY_MAX"] == "9G"
    assert argv[argv.index("--chdir") + 1] == str(rev)
    outer = (rec / "systemd-run.argv").read_text().splitlines()
    assert outer[:3] == ["--user", "--scope", "--quiet"] and "MemoryMax=9G" in outer and "MemorySwapMax=0G" in outer
    assert not (rec / "timeout.argv").exists(), "an owner line got the agent-only wall-clock timeout"
    assert not list((home / "derisk-pool" / ".sandbox-jobs").iterdir()), "the per-job shim dir was left behind"


def test_agent_line_gets_only_its_own_output_dir_and_the_resource_bounds(tmp_path: Path) -> None:
    home, rev = _node(tmp_path)
    bin_dir, rec = _fakes(tmp_path)
    job = (f"POOL_CHECKED_REASON=x AGENT_JOB_SRC=sim-agent cd ~/derisk-pool/revisions/{SHA} && mkdir -p "
           f"research/findings/raw/_agent_jobs/{AGENT_ID} && SIM_NO_PROVENANCE=1 .venv/bin/python -u -m "
           f"research.runners.r --out research/findings/raw/_agent_jobs/{AGENT_ID}/o.json")
    res, cwd = _run_wrapper(home, job, bin_dir, bwrap=str(bin_dir / "bwrap"))
    assert _status(cwd)[0] == "0", res.stderr
    argv = (rec / "bwrap.argv").read_text().splitlines()
    out = rev / "research" / "findings" / "raw" / "_agent_jobs" / AGENT_ID
    assert _pairs(argv, "--bind") == [(str(out), str(out))]
    assert out.is_dir()
    outer = (rec / "systemd-run.argv").read_text().splitlines()
    assert "MemoryMax=8G" in outer and "TasksMax=512" in outer
    assert (rec / "timeout.argv").read_text().splitlines()[:2] == ["--kill-after=30s", "12h"]


@pytest.mark.parametrize("job, why", [
    (f"AGENT_JOB_SRC=sim-agent cd ~/derisk-pool/revisions/{SHA} && touch research/findings/raw/_agent_jobs/.ssh/x",
     "must name exactly one"),
    (f"AGENT_JOB_SRC=sim-agent cd ~/derisk-pool/sim && echo research/findings/raw/_agent_jobs/{AGENT_ID}",
     "exactly one revision"),
    ("cd ~/derisk-pool/revisions/fedcba9876543210fedcba9876543210fedcba98 && echo x", "none of the revisions"),
])
def test_refusals_never_run_the_job(tmp_path: Path, job: str, why: str) -> None:
    home, _ = _node(tmp_path)
    bin_dir, rec = _fakes(tmp_path)
    res, cwd = _run_wrapper(home, job, bin_dir, bwrap=str(bin_dir / "bwrap"))
    assert _status(cwd) == ("126", job)
    assert not (rec / "bwrap.argv").exists(), "the sandbox was started for a refused line"
    assert "REFUSED" in (cwd / "autodispatch.out").read_text() and why in (cwd / "autodispatch.out").read_text()


def test_no_bwrap_means_refused_not_unsandboxed(tmp_path: Path) -> None:
    home, _ = _node(tmp_path)
    marker = tmp_path / "ran"
    res, cwd = _run_wrapper(home, f"touch {marker}", None, bwrap=str(tmp_path / "no-such-bwrap"))
    assert _status(cwd)[0] == "126" and not marker.exists()
    assert "not installed" in (cwd / "autodispatch.out").read_text()


def test_a_memcap_line_without_systemd_user_is_refused_like_memcap_itself(tmp_path: Path) -> None:
    home, _ = _node(tmp_path)
    bin_dir, rec = _fakes(tmp_path, systemd_ok=False)
    res, cwd = _run_wrapper(home, f"cd ~/derisk-pool/revisions/{SHA} && bash tools/memcap.sh 4 -- true", bin_dir,
                            bwrap=str(bin_dir / "bwrap"))
    assert _status(cwd)[0] == "3" and not (rec / "bwrap.argv").exists()


def test_a_symlinked_output_area_is_refused(tmp_path: Path) -> None:
    home, rev = _node(tmp_path)
    bin_dir, rec = _fakes(tmp_path)
    raw = rev / "research" / "findings" / "raw"
    raw.rmdir()
    raw.symlink_to(home / ".ssh")
    res, cwd = _run_wrapper(home, f"cd ~/derisk-pool/revisions/{SHA} && echo x", bin_dir, bwrap=str(bin_dir / "bwrap"))
    assert _status(cwd)[0] == "126" and not (rec / "bwrap.argv").exists()


def test_probe_reports_readiness(tmp_path: Path) -> None:
    home, _ = _node(tmp_path)
    ok = _exe(tmp_path / "okbin" / "bwrap", '[ "$1" = --version ] && { echo "bubblewrap 0.6.1"; exit 0; }\n'
              'echo SANDBOX_INSIDE_OK\n')
    res, _ = _run_wrapper(home, "", None, bwrap=str(ok), action="probe")
    assert res.stdout.strip() == "POOL_SANDBOX_READY bwrap=0.6.1"
    res, _ = _run_wrapper(home, "", None, bwrap=str(tmp_path / "missing"), action="probe")
    assert res.stdout.strip() == "POOL_SANDBOX_NOT_READY reason=bwrap-not-installed"
    bad = _exe(tmp_path / "badbin" / "bwrap", 'echo "bwrap: setting up uid map: Permission denied" >&2; exit 1\n')
    res, _ = _run_wrapper(home, "", None, bwrap=str(bad), action="probe")
    assert res.stdout.startswith("POOL_SANDBOX_NOT_READY reason=bwrap-failed: bwrap: setting up uid map")


# ---------------------------------------------------------------------------------- 3. the REAL bwrap

def _real_bwrap_usable() -> bool:
    if shutil.which("bwrap") is None:
        return False
    return subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                           "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--", "/bin/true"],
                          capture_output=True).returncode == 0


needs_bwrap = pytest.mark.skipif(not _real_bwrap_usable(), reason="real bubblewrap cannot run here")

PROBE = r"""
import json, os, socket, subprocess
r = {}
def conn(host, port):
    try:
        socket.create_connection((host, port), timeout=3).close(); return "OPEN"
    except OSError as e:
        return "blocked: %s" % e.strerror
r["lan_smb"] = conn("192.168.0.10", 445)
r["internet"] = conn("1.1.1.1", 443)
try:
    socket.getaddrinfo("example.com", 443); r["dns"] = "RESOLVED"
except OSError as e:
    r["dns"] = "blocked: %s" % e
r["sudo_rc"] = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode if os.path.exists("/usr/bin/sudo") else None
r["nnp"] = [l.split()[1] for l in open("/proc/self/status") if l.startswith("NoNewPrivs")][0]
def can_write(p):
    try:
        open(p, "w").write("x"); return True
    except OSError:
        return False
r["write_revision"] = can_write("tools/planted.sh")
r["write_usr"] = can_write("/usr/planted")
r["write_home"] = can_write(os.path.expanduser("~/planted"))
r["ssh_visible"] = os.path.exists(os.path.expanduser("~/.ssh"))
r["shared_checkout_visible"] = os.path.exists(os.path.expanduser("~/derisk-pool/sim"))
r["pids"] = sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
r["env_pool_job_id"] = os.environ.get("POOL_JOB_ID")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(r, open(OUT, "w"))
"""


@needs_bwrap
def test_real_sandbox_owner_line(tmp_path: Path) -> None:
    import json
    import shlex
    home, rev = _node(tmp_path)
    out_rel = "research/findings/raw/_selftest/report.json"
    code = "OUT = %r\n%s" % (out_rel, PROBE)
    job = f"cd ~/derisk-pool/revisions/{SHA} && .venv/bin/python -c {shlex.quote(code)}"
    res, cwd = _run_wrapper(home, job, None)
    rc, _ = _status(cwd)
    assert rc == "0", (cwd / "autodispatch.out").read_text()
    r = json.loads((rev / out_rel).read_text())                       # (a) the result landed on the node
    assert r["lan_smb"].startswith("blocked") and r["internet"].startswith("blocked"), r   # (b) no network
    assert r["dns"].startswith("blocked"), r
    assert r["nnp"] == "1" and r["sudo_rc"] in (None, 1), r           # (c) no new privileges; sudo fails
    assert not r["write_revision"] and not r["write_usr"], r          # (d) read-only outside the output area
    assert not (rev / "tools" / "planted.sh").exists()
    assert not (home / "planted").exists(), "a write to the sandbox HOME reached the node's real HOME"
    assert not r["ssh_visible"] and not r["shared_checkout_visible"], r
    assert r["pids"] and max(r["pids"]) < 10, "the job can see the node's other processes: %s" % r["pids"]
    assert r["env_pool_job_id"] == "1790000000000-42"


@needs_bwrap
def test_real_sandbox_agent_line_can_write_only_its_own_dir(tmp_path: Path) -> None:
    import json
    import shlex
    home, rev = _node(tmp_path)
    out_rel = f"research/findings/raw/_agent_jobs/{AGENT_ID}/o.json"
    code = ("import json\nr = {}\n"
            "def w(p):\n    try:\n        open(p, 'w').write('x'); return True\n    except OSError:\n        return False\n"
            "r['sibling'] = w('research/findings/raw/other.json')\n"
            f"r['own'] = w({out_rel!r})\n"
            f"json.dump(r, open({out_rel!r}, 'w'))\n")
    job = (f"AGENT_JOB_SRC=sim-agent cd ~/derisk-pool/revisions/{SHA} && mkdir -p "
           f"research/findings/raw/_agent_jobs/{AGENT_ID} && .venv/bin/python -c {shlex.quote(code)}")
    # tests/conftest.py puts an inert systemd-run on PATH for every test (no test may start a real unit), so the outer
    # scope here is the recording fake that execs its command; its argv is pinned by the fake-bwrap test above.
    bin_dir, rec = _fakes(tmp_path)
    (bin_dir / "bwrap").unlink()                    # ...but the REAL bwrap
    res, cwd = _run_wrapper(home, job, bin_dir)
    rc, _ = _status(cwd)
    assert rc == "0", (cwd / "autodispatch.out").read_text()
    assert "MemoryMax=8G" in (rec / "systemd-run.argv").read_text()
    r = json.loads((rev / out_rel).read_text())
    assert r == {"sibling": False, "own": True}
    assert not (rev / "research" / "findings" / "raw" / "other.json").exists()

def test_corpus_extra_is_bound_read_only_and_never_through_a_symlink(tmp_path: Path) -> None:
    """Owner 2026-10-09: ~/derisk-pool/corpus_extra (prereg'd corpora outside any revision) is visible read-only;
    a symlinked corpus_extra is not bound at all, and the writable set is unchanged."""
    home, rev = _node(tmp_path)
    corpus = home / "derisk-pool" / "corpus_extra"
    corpus.mkdir()
    bin_dir, rec = _fakes(tmp_path)
    job = f"cd ~/derisk-pool/revisions/{SHA} && .venv/bin/python -c 1"
    res, cwd = _run_wrapper(home, job, bin_dir, bwrap=str(bin_dir / "bwrap"))
    assert _status(cwd) == ("0", job), res.stderr
    argv = (rec / "bwrap.argv").read_text().splitlines()
    assert (str(corpus), str(corpus)) in _pairs(argv, "--ro-bind")
    assert _pairs(argv, "--bind") == [(str(rev / "research/findings/raw"),) * 2]
    # a symlinked corpus_extra is skipped, not followed
    corpus.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    corpus.symlink_to(elsewhere)
    (rec / "bwrap.argv").unlink()
    res, cwd = _run_wrapper(home, job, bin_dir, bwrap=str(bin_dir / "bwrap"))
    argv = (rec / "bwrap.argv").read_text().splitlines()
    assert not any(str(corpus) in a or str(elsewhere) in a for a in argv), argv
