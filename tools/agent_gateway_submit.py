#!/usr/bin/env python3
"""tools/agent_gateway_submit.py -- H2 fix (2026-09-29 security review). Replaces FREE-FORM job submission
(a single quoted shell command, forwarded almost verbatim) with a STRUCTURED request the gateway itself
validates and builds the actual command line from -- no text from the container ever reaches a shell as-is.

    agent_gateway_submit.py gpu  --sha <40-hex> --runner research.runners.<name> [--prereg <path>] -- <args...>
    agent_gateway_submit.py pool --sha <40-hex> --runner research.runners.<name> [--prereg <path>]
                                  --checked <reason> -- <args...>

Called from tools/agent_gateway.sh with an ALREADY shlex-split argument list (tools/agent_gateway_parse.py,
H1) -- so this never re-parses a raw string, and never invokes a shell on anything the caller wrote.

VALIDATION (all of this refuses before anything is queued or executed):
  - --sha is exactly 40 hex characters AND is reachable from a `refs/remotes/{origin,gitea}/*` ref (never an
    unpushed/local-only commit -- the container's own branch pushes are what land there, see
    docs/AGENT_HARNESS_SETUP.md's gitea branch-protection step (M2): only research/draft-* is ever pushable, so
    a sha reachable from a remote ref is, transitively, a sha the branch-protection rules already allowed).
  - --runner matches `research\\.runners\\.[A-Za-z0-9_]+` AND that module file exists in the repo AT THAT SHA
    (git cat-file -e <sha>:research/runners/<name>.py) -- never merely "exists on the local checkout's HEAD".
  - every argument token matches `^[A-Za-z0-9_./=:,+@%-]+$` (no shell metacharacters at all: no spaces, quotes,
    `$`, backticks, `;`, `|`, `&`, redirection, globs -- so even if something downstream ever concatenated
    these into a string, there would be nothing for an attacker's token to DO).
  - any `--out`/`--out=` argument is reduced to its BASENAME and relocated under a fresh, per-job subdirectory
    of `research/findings/raw/_agent_jobs/` (2026-09-29 re-review, LOW: a job must not write the shared raw/
    directory directly) -- the caller's directory component is discarded outright, never merely validated.
  - a GATE seed (42/43/44/100/101/102) requires a `--prereg <path>` that exists as a COMMITTED file at the
    pinned --sha (tools/gate_seed_guard.py, shared with tools/local_llm/gpu_add.sh -- CLAUDE.md non-negotiable).

gpu: the runner executes SANDBOXED via bubblewrap (see build_bwrap_argv) against a detached checkout the
gateway itself creates at .claude/worktrees/agent-jobs/<sha> (guaranteed clean + pinned -- this SUPERSEDES
gpu_add.sh's own "cd into a committed, clean checkout" check, which is why this calls tools/gpu_queue.sh
directly rather than through gpu_add.sh: gpu_add.sh's checkout guarantee is redundant here (this script gives a
STRONGER one by construction) and its own shape check -- the job must literally start with `cd <path> && ...`
-- cannot be satisfied by a job that starts with `bwrap ...` by design. The gate-seed/prereg guard is still the
identical, SHARED check (tools/gate_seed_guard.py), not a second copy.

pool: builds the standard provisioned-revision job line (`cd ~/derisk-pool/revisions/<sha> && ...`, the same
shape tools/pool_provision.sh --isolated / tools/pool_queue.sh / tools/pool_autodispatch.sh already use
throughout) and forwards it to tools/pool_queue.sh add, which re-applies its OWN full guard suite (shape/
gate-seed/node-validation/--checked control-character rejection) independently -- this script's own validation
above is not a substitute for that, it is what makes the CONSTRUCTED command line trustworthy in the first
place. Pool nodes are NOT sandboxed (see docs/AGENT_HARNESS_SETUP.md's "pool nodes are dedicated compute" note,
M2/H2) -- they are dedicated, disposable compute with no credentials of their own beyond what
tools/pool_provision.sh deploys, not shared infrastructure a malicious job could pivot from the way the
workstation's own GPU process could.
"""
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid

ROOT = os.environ.get("AGENT_GATEWAY_SUBMIT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# gate_seed_guard.py ships alongside THIS script in the real codebase and is resolved from ITS OWN directory,
# never from the (test-overridable) ROOT above -- ROOT is the repository being validated/queued AGAINST (a
# scratch repo in tests), which has no reason to carry a copy of this codebase's own tooling.
GATE_SEED_GUARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_seed_guard.py")
GPU_QUEUE_SH = os.environ.get("AGENT_GATEWAY_GPU_QUEUE_SH", os.path.join(ROOT, "tools", "gpu_queue.sh"))
POOL_QUEUE_SH = os.environ.get("AGENT_GATEWAY_POOL_QUEUE_SH", os.path.join(ROOT, "tools", "pool_queue.sh"))
POOL_PROVISION_SH = os.environ.get("AGENT_GATEWAY_POOL_PROVISION_SH", os.path.join(ROOT, "tools", "pool_provision.sh"))
BWRAP = os.environ.get("AGENT_GATEWAY_BWRAP", "bwrap")   # TEST-ONLY override; NEVER set in production
HOME = os.environ.get("AGENT_GATEWAY_HOME", os.path.expanduser("~"))
JOBS_DIR = os.path.join(ROOT, ".claude", "worktrees", "agent-jobs")
GIT_FETCH_REMOTE = os.environ.get("AGENT_GATEWAY_GITEA_REMOTE", "gitea")

# RESOURCE + AVAILABILITY BOUNDS (2026-09-29 re-review, M2) -- none of bwrap's namespace isolation caps CPU/
# memory/process-count or wall-clock runtime; these bound what a sandboxed job can do to the shared workstation
# even while it runs, and how much of the shared queue/disk ANY number of submissions can claim.
SYSTEMD_RUN = os.environ.get("AGENT_GATEWAY_SYSTEMD_RUN", "systemd-run")   # TEST-ONLY override
TIMEOUT_BIN = os.environ.get("AGENT_GATEWAY_TIMEOUT_BIN", "timeout")       # TEST-ONLY override
GPU_JOB_TIMEOUT_H = os.environ.get("AGENT_GATEWAY_GPU_JOB_TIMEOUT_H", "4")
GPU_JOB_MEMORY_MAX = os.environ.get("AGENT_GATEWAY_GPU_JOB_MEMORY_MAX", "20G")
GPU_JOB_TASKS_MAX = os.environ.get("AGENT_GATEWAY_GPU_JOB_TASKS_MAX", "512")
AGENT_JOB_MARKER = "AGENT_JOB_SRC=sim-agent"   # embedded in every job line THIS script builds, so the queue-
                                                # depth cap counts only agent-submitted jobs, never unrelated
                                                # work the owner/other tooling queued directly.
MAX_QUEUED_AGENT_JOBS = int(os.environ.get("AGENT_GATEWAY_MAX_QUEUED_JOBS", "20"))
MAX_AGENT_WORKTREES = int(os.environ.get("AGENT_GATEWAY_MAX_WORKTREES", "20"))
GPU_QUEUE_PATH = os.environ.get("AGENT_GATEWAY_GPU_QUEUE_PATH", os.path.join(ROOT, "research", "queue", "gpu.queue"))
POOL_QUEUE_PATH = os.environ.get("AGENT_GATEWAY_POOL_QUEUE_PATH", os.path.join(ROOT, "research", "queue", "pool.queue"))
POOL_PROVISION_LOCK = os.environ.get("AGENT_GATEWAY_POOL_PROVISION_LOCK",
                                      os.path.join(ROOT, "research", "queue", ".agent_pool_provision.lock"))
FLOCK_BIN = os.environ.get("AGENT_GATEWAY_FLOCK_BIN", "flock")   # TEST-ONLY override
# PER-JOB output root (2026-09-29 re-review, LOW): a job must never write into the SHARED research/findings/raw/
# directly -- every submission gets its own subdirectory here, and only THAT subdirectory is ever bound
# writable into the sandbox. A (not-yet-written) harvest step is expected to copy real results out of these
# subdirectories into their permanent home; this script's job is only to make sure a job can never reach
# anything BUT its own subdirectory.
AGENT_JOBS_OUT_ROOT = os.path.join(ROOT, "research", "findings", "raw", "_agent_jobs")
NVIDIA_DEVICES = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]

SHA_RE = re.compile(r'^[0-9a-f]{40}$')
RUNNER_RE = re.compile(r'^research\.runners\.[A-Za-z0-9_]+$')
ARG_RE = re.compile(r'^[A-Za-z0-9_./=:,+@%-]+$')
OUT_RE = re.compile(r'^--out(?:=(.*))?$')
SAFE_BASENAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')


def die(msg, code=1):
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


def git(*args, cwd=ROOT):
    # -c core.hooksPath=/nonexistent on EVERY git invocation this script makes (2026-09-29 re-review, L2): a
    # `git worktree add` (or any other git operation) otherwise runs whatever hooks ambient config points at --
    # this repo is a shared checkout with many worktrees, so a hooksPath set for some OTHER purpose (or a
    # future config change) must never be able to run code as a side effect of validating/staging an
    # agent-submitted job. Read-only calls (cat-file/rev-parse/merge-base/for-each-ref) don't trigger hooks
    # anyway; setting this unconditionally on the one shared helper means no future call here can forget it.
    return subprocess.run(["git", "-c", "core.hooksPath=/nonexistent", "-C", cwd, *args],
                           capture_output=True, text=True)


# ------------------------------------------------------------------------------------------------- validation

def parse_common(kind, argv):
    """Parses the shared `--sha S --runner R [--prereg P] [--checked C] -- args...` shape. `--checked` is
    required for kind == "pool", forbidden for "gpu" (pool_queue.sh's own contract)."""
    sha = runner = prereg = checked = None
    i = 0
    args = None
    while i < len(argv):
        a = argv[i]
        if a == "--sha":
            i += 1; sha = argv[i] if i < len(argv) else die("--sha needs a value")
        elif a == "--runner":
            i += 1; runner = argv[i] if i < len(argv) else die("--runner needs a value")
        elif a == "--prereg":
            i += 1; prereg = argv[i] if i < len(argv) else die("--prereg needs a value")
        elif a == "--checked":
            i += 1; checked = argv[i] if i < len(argv) else die("--checked needs a value")
        elif a == "--":
            args = argv[i + 1:]
            break
        else:
            die(f"unexpected token before '--': {a!r}")
        i += 1
    if args is None:
        die("missing '--' separator before the runner arguments")
    if not sha:
        die("--sha is required")
    if not runner:
        die("--runner is required")
    if kind == "pool" and not checked:
        die("--checked '<what the record says>' is required for pool-submit (tools/pool_queue.sh's own gate)")
    if kind == "gpu" and checked is not None:
        die("--checked is not accepted by gpu-submit")
    return sha, runner, prereg, checked, args


def _fetch_draft_branches():
    # Best-effort, SCOPED fetch (2026-09-29 re-review, L5): without this, gpu-submit/pool-submit only ever saw
    # whatever refs/remotes/gitea/* happened to already be cached locally -- a sha the container just pushed to
    # its own research/draft-* branch was refused as "not reachable" until the OWNER happened to run a `git
    # fetch` themselves, or (if not even the commit OBJECT was present yet) failed the earlier cat-file check
    # outright. Scoped to research/draft-* ONLY -- the one namespace the container's own deploy key can push to
    # (docs/AGENT_HARNESS_SETUP.md's gitea branch-protection step, M2) -- never a bare `git fetch` of everything.
    # Hooks disabled (core.hooksPath=/nonexistent, same as every other git call here). Failure (gitea
    # unreachable, no such remote, timeout) is silent: the checks below simply fall back to whatever is
    # already cached, exactly like before this fetch existed -- this only ever makes reachability MORE
    # permissive of genuinely-pushed work, never less strict.
    git_bin = os.environ.get("AGENT_GATEWAY_GIT_BIN", "git")   # TEST-ONLY override; NEVER set in production
    try:
        subprocess.run(
            [git_bin, "-c", "core.hooksPath=/nonexistent", "-C", ROOT, "fetch", "--quiet", GIT_FETCH_REMOTE,
             "refs/heads/research/draft-*:refs/remotes/%s/research/draft-*" % GIT_FETCH_REMOTE],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        pass


def validate_sha(sha):
    if not SHA_RE.match(sha):
        die(f"--sha must be exactly 40 hex characters, got: {sha!r}")
    _fetch_draft_branches()
    if git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        die(f"--sha {sha} is not a commit this repo has")
    refs_res = git("for-each-ref", "--format=%(refname)", "refs/remotes/origin", "refs/remotes/gitea")
    refs = [r for r in refs_res.stdout.splitlines() if r.strip()]
    for ref in refs:
        if git("merge-base", "--is-ancestor", sha, ref).returncode == 0:
            return
    die(f"--sha {sha} is not reachable from any refs/remotes/{{origin,gitea}}/* ref "
        f"(an unpushed/local-only commit is never a valid job pin)")


def validate_runner(sha, runner):
    if not RUNNER_RE.match(runner):
        die(f"--runner must match research.runners.<name>, got: {runner!r}")
    name = runner.rsplit(".", 1)[-1]
    relpath = f"research/runners/{name}.py"
    if git("cat-file", "-e", f"{sha}:{relpath}").returncode != 0:
        die(f"{relpath} does not exist at --sha {sha}")
    return relpath


def validate_args(args):
    for a in args:
        if not ARG_RE.match(a):
            die(f"argument does not match the allowed pattern ^[A-Za-z0-9_./=:,+@%-]+$: {a!r}")


def rewrite_out_args(args, job_out_dir):
    """A job must never write into the SHARED research/findings/raw/ directly (2026-09-29 re-review, LOW) --
    every `--out`/`--out=` value the caller supplies is reduced to its BASENAME (any directory component,
    including a `..` traversal attempt, is simply discarded, never trusted or even inspected for validity) and
    relocated under this job's OWN, freshly-created output subdirectory. There is deliberately no "reject a bad
    --out path" step any more: a path cannot escape what it is never allowed to keep in the first place."""
    out = []
    i = 0
    while i < len(args):
        a = args[i]
        m = OUT_RE.match(a)
        if m:
            if m.group(1) is not None:
                value = m.group(1)
            else:
                i += 1
                if i >= len(args):
                    die("--out needs a value")
                value = args[i]
            base = os.path.basename(value.rstrip("/")) or "out"
            if not SAFE_BASENAME_RE.match(base) or base in (".", ".."):
                die(f"--out's filename is not safe: {value!r}")
            out.append("--out=%s" % os.path.join(job_out_dir, base))
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def validate_gate_seed(sha, args, prereg):
    joined = " ".join(args + (["--prereg", prereg] if prereg else []))
    py = subprocess.run([sys.executable, GATE_SEED_GUARD, joined], capture_output=True, text=True)
    if py.returncode != 0:
        die(f"gate-seed guard could not run: {py.stderr.strip()}")
    hit, found_prereg = py.stdout.split(None, 1)
    found_prereg = found_prereg.strip()
    if hit != "1":
        return
    prereg_path = prereg or (found_prereg if found_prereg != "-" else None)
    if not prereg_path:
        die("this job runs a GATE seed (42/43/44/100/101/102) but carries no --prereg <path> -- "
            "CLAUDE.md non-negotiable, guardrail table row 'ran GATE seeds without a committed prereg' (2026-09-28)")
    prereg_rel = prereg_path.lstrip("./")
    if git("cat-file", "-e", f"{sha}:{prereg_rel}").returncode != 0:
        die(f"--prereg {prereg_path!r} is not a file committed at --sha {sha}")


# --------------------------------------------------------------------------------------------------------- gpu

def ensure_bwrap():
    if shutil.which(BWRAP) is None:
        die("bubblewrap ('bwrap') is not installed on this workstation -- gpu-submit refuses to run an "
            "agent-authored job unsandboxed. Owner: install it (e.g. `sudo pacman -S bubblewrap` on this "
            "Arch/CachyOS workstation, or `sudo apt-get install bubblewrap` on Debian/Ubuntu) and retry.")


def _remove_worktree(path):
    git("worktree", "remove", "--force", path)
    subprocess.run(["rm", "-rf", path])


def _cleanup_old_worktrees(keep, exempt=None):
    """Caps how many .claude/worktrees/agent-jobs/<sha> checkouts can accumulate (2026-09-29 re-review, M2):
    each is a full checkout, and every gpu-submit that names a new --sha creates one -- with no bound, an agent
    that keeps varying --sha (or a long-lived harness) could fill the workstation's disk one checkout at a
    time. Removes the OLDEST (by mtime) worktrees under JOBS_DIR beyond `keep`, via `git worktree remove` (never
    a bare rm -- keeps the repo's own worktree registration consistent). `exempt` (the sha about to be
    created/reused this call) is never removed even if it is technically the oldest."""
    if not os.path.isdir(JOBS_DIR):
        return
    entries = []
    for name in os.listdir(JOBS_DIR):
        if exempt and name == exempt:
            continue
        p = os.path.join(JOBS_DIR, name)
        if os.path.isdir(p):
            try:
                entries.append((os.path.getmtime(p), p))
            except OSError:
                continue
    entries.sort()   # oldest first
    n_over = len(entries) - (keep - 1 if exempt else keep)   # room for `exempt` itself if it's being created
    for _mtime, p in entries[:max(0, n_over)]:
        _remove_worktree(p)


def ensure_worktree(sha):
    path = os.path.join(JOBS_DIR, sha)
    if os.path.isdir(path):
        head = git("rev-parse", "HEAD", cwd=path)
        if head.returncode == 0 and head.stdout.strip() == sha:
            _ensure_venv_symlink(path)
            return path
        # stale/wrong -- remove the registration and the directory, then recreate cleanly below.
        _remove_worktree(path)
    _cleanup_old_worktrees(MAX_AGENT_WORKTREES, exempt=sha)
    os.makedirs(JOBS_DIR, exist_ok=True)
    res = git("worktree", "add", "--detach", path, sha)
    if res.returncode != 0:
        die(f"could not create the job checkout at {path}: {res.stderr.strip()}")
    _ensure_venv_symlink(path)
    return path


def _ensure_venv_symlink(path):
    # matches the convention every OTHER worktree in this repo already uses (`.venv -> <repo>/.venv`, verified
    # 2026-09-29 across .claude/worktrees/*): `git worktree add` does not create this itself (`.venv` is
    # gitignored), so `.venv/bin/python` inside a freshly created worktree would otherwise not exist at all.
    # The symlink is only ever a POINTER; build_bwrap_argv ro-binds the REAL target (ROOT/.venv) at its own
    # absolute path, so resolving through this symlink inside the sandbox finds real content.
    link = os.path.join(path, ".venv")
    target = os.path.join(ROOT, ".venv")
    if not os.path.islink(link) and not os.path.exists(link):
        try:
            os.symlink(target, link)
        except OSError:
            pass


def build_bwrap_argv(checkout, job_out_dir, inner_argv, home=None, repo_venv=None):
    """SANDBOX for an agent-authored GPU job, rebuilt FROM EMPTY (2026-09-29 re-review -- HIGH-1: the previous
    `--ro-bind / /` design left the HOST's real /proc visible with no PID namespace, so a same-uid process's
    /proc/<pid>/environ (live ANTHROPIC/AWS/GH tokens from the dispatcher, local-llm, Claude Code) was readable,
    and left /run/user/<uid>/bus + DBUS_SESSION_BUS_ADDRESS reachable, so `systemctl --user` could start a
    transient unit that runs OUTSIDE the sandbox entirely -- both are real escapes, not merely theoretical).
    Nothing is trusted by default any more; every path below was individually verified NEEDED (a real `cupy`
    array op run through this exact argv; the six sandbox-escape properties below are pinned by real-bwrap
    tests in tests/test_agent_gateway_submit_sandbox_security.py, and a real GPU compute smoke through the
    actual gpu_queue.sh was run manually -- see this commit's message for the result):

      - `--unshare-all` (net, pid, ipc, uts, cgroup, AND user -- verified this still sees the GPU: the device
        nodes are world-rw, so namespace-mapped uids don't affect opening them) + `--die-with-parent` (a job can
        never outlive gpu_queue.sh's own process tree) + `--new-session` (detaches from any controlling
        terminal/session, e.g. TIOCSTI-style tricks).
      - `--clearenv` + an explicit allow-list (PATH, a fresh empty HOME, LANG, SIM_BACKEND, CUDA_VISIBLE_DEVICES,
        thread-count env vars) -- no ANTHROPIC_*/AWS_*/GH_* token, no DBUS_SESSION_BUS_ADDRESS, no SSH_AUTH_SOCK
        is ever even in this process's environment to read, from itself or (now impossible anyway) another pid.
      - a FRESH `--proc /proc` (this process's own, now-isolated pid namespace only -- confirmed: only its own
        pid 1/2 are visible) and a FRESH `--dev /dev`, then `--dev-bind` ONLY the four nvidia device nodes that
        exist on this box -- never `/dev` itself, never `/run` or `/run/user/*` (both simply do not exist inside
        the sandbox at all: dbus/systemd --user are unreachable, not merely denied).
      - RO-BIND ONLY: `/usr` (+ `/lib`, `/lib64` as the SAME symlinks this Arch-based host already uses, so the
        dynamic linker and `python3.11` resolve), `/sys` (verified REQUIRED -- cupy's device enumeration fails
        with `cudaErrorOperatingSystem` without it; nothing narrower was found to work), and the repo's shared
        `.venv` (cupy + its bundled CUDA runtime libraries; only `libcuda.so.1` itself, the kernel-matched
        driver library, comes from `/usr`). Never `/`, `/home`, `/etc`, `/proc`, `/run`.
      - Writable: ONLY the job's own per-job output subdirectory (2026-09-29 re-review, LOW -- never the shared
        research/findings/raw/ directly; see rewrite_out_args). The checkout itself is RO-BOUND (2026-09-29
        re-review, L1) -- a job has no reason to modify the pinned revision it runs against; scratch space, if
        a runner needs any outside its declared --out, is the already-writable --tmpfs /tmp.
    """
    home = home or HOME
    repo_venv = repo_venv or os.path.join(ROOT, ".venv")
    argv = [
        BWRAP,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "HOME", "/tmp/home",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "SIM_BACKEND", "cupy",
        "--setenv", "CUDA_VISIBLE_DEVICES", "0",
        "--setenv", "OMP_NUM_THREADS", "1",
        "--setenv", "OPENBLAS_NUM_THREADS", "1",
        "--setenv", "MKL_NUM_THREADS", "1",
        "--setenv", "NUMEXPR_NUM_THREADS", "1",
        "--dir", "/tmp/home",
        "--proc", "/proc",
        "--dev", "/dev",
    ]
    for dev in NVIDIA_DEVICES:
        if os.path.exists(dev):
            argv += ["--dev-bind", dev, dev]
    argv += [
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", "/sys", "/sys",
    ]
    if os.path.isdir(repo_venv):
        argv += ["--ro-bind", repo_venv, repo_venv]
    argv += ["--tmpfs", "/tmp"]
    argv += ["--ro-bind", checkout, checkout]
    argv += ["--bind", job_out_dir, job_out_dir]
    argv += ["--chdir", checkout]
    argv += ["--"] + inner_argv
    return argv


def _count_agent_jobs(queue_path):
    if not os.path.exists(queue_path):
        return 0
    try:
        return open(queue_path, errors="ignore").read().count(AGENT_JOB_MARKER)
    except OSError:
        return 0


def _enforce_queue_cap(queue_path, kind):
    # 2026-09-29 re-review, M2: an unbounded number of agent-submitted jobs could otherwise fill the shared
    # queue (and, for GPU, the workstation's own resources once dispatched) indefinitely -- counted by
    # AGENT_JOB_MARKER so this never counts the owner's own, unrelated queue entries.
    n = _count_agent_jobs(queue_path)
    if n >= MAX_QUEUED_AGENT_JOBS:
        die(f"refusing to queue another {kind} job: {n} agent-submitted jobs are already queued "
            f"(cap {MAX_QUEUED_AGENT_JOBS}) -- wait for some to drain before submitting more")


def wrap_gpu_job_with_resource_limits(job_id, bwrap_argv):
    """2026-09-29 re-review, M2: bwrap's namespace isolation caps WHAT a job can reach, never HOW MUCH memory/
    processes it can consume or for how long it can run -- a job that simply allocates until OOM, fork-bombs
    within its own pid namespace, or hangs forever would still degrade (or indefinitely occupy) the shared
    workstation/GPU queue. `systemd-run --user --scope` runs the sandboxed job as a transient cgroup-scoped
    unit (synchronously -- `--scope`, not `--unit`, so gpu_queue.sh's own `wait` on the dispatched pid still
    works exactly as before) with a hard MemoryMax/TasksMax; `timeout` wraps the WHOLE scope from outside it, so
    a hung job is killed by wall-clock time regardless of what is happening inside the sandbox.
    `--setenv=AGENT_JOB_SRC=sim-agent` embeds AGENT_JOB_MARKER in the literal job line so the queue-depth cap
    can count agent-submitted jobs specifically."""
    unit = "sim-agent-gpu-%s" % job_id
    return [
        TIMEOUT_BIN, "--kill-after=30s", "%sh" % GPU_JOB_TIMEOUT_H,
        SYSTEMD_RUN, "--user", "--scope", "--quiet",
        "--unit=%s" % unit,
        "--setenv=AGENT_JOB_SRC=sim-agent",
        "-p", "MemoryMax=%s" % GPU_JOB_MEMORY_MAX,
        "-p", "TasksMax=%s" % GPU_JOB_TASKS_MAX,
        "--",
    ] + bwrap_argv


def cmd_gpu(argv):
    sha, runner, prereg, _checked, args = parse_common("gpu", argv)
    validate_sha(sha)
    validate_runner(sha, runner)
    validate_args(args)
    validate_gate_seed(sha, args, prereg)
    _enforce_queue_cap(GPU_QUEUE_PATH, "GPU")
    ensure_bwrap()
    checkout = ensure_worktree(sha)
    job_id = "%s-%s" % (sha[:12], uuid.uuid4().hex[:8])
    job_out_dir = os.path.join(AGENT_JOBS_OUT_ROOT, job_id)
    os.makedirs(job_out_dir, exist_ok=True)
    args = rewrite_out_args(args, job_out_dir)
    venv_python = os.path.join(checkout, ".venv", "bin", "python")
    inner_argv = [venv_python, "-u", "-m", runner] + args
    bwrap_argv = build_bwrap_argv(checkout, job_out_dir, inner_argv)
    wrapped_argv = wrap_gpu_job_with_resource_limits(job_id, bwrap_argv)
    job_line = " ".join(shlex.quote(a) for a in wrapped_argv)
    res = subprocess.run(["bash", GPU_QUEUE_SH, "add", job_line])
    return res.returncode


# -------------------------------------------------------------------------------------------------------- pool

def cmd_pool(argv):
    sha, runner, prereg, checked, args = parse_common("pool", argv)
    validate_sha(sha)
    validate_runner(sha, runner)
    validate_args(args)
    validate_gate_seed(sha, args, prereg)
    _enforce_queue_cap(POOL_QUEUE_PATH, "pool")
    # PER-JOB output subdir, same rationale as gpu-submit (2026-09-29 re-review, LOW): a job must not write the
    # shared research/findings/raw/ directly. Pool jobs run on a REMOTE node against its own
    # ~/derisk-pool/revisions/<sha> checkout, so this is a RELATIVE path (resolved against that remote
    # checkout), not the local absolute AGENT_JOBS_OUT_ROOT gpu-submit uses -- the existing pool result-sync
    # mechanism (tools/pool_sync.sh) is what eventually copies it back into this repo.
    job_id = "%s-%s" % (sha[:12], uuid.uuid4().hex[:8])
    job_out_dir_rel = "research/findings/raw/_agent_jobs/%s" % job_id
    args = rewrite_out_args(args, job_out_dir_rel)
    # BEST-EFFORT, NON-BLOCKING provisioning of this revision ahead of dispatch (2026-09-29, H2: "build the
    # standard provisioned job line ... from the structured request"). The job line below pins to
    # ~/derisk-pool/revisions/<sha>, the exact layout tools/pool_provision.sh --isolated creates; the dispatcher
    # (tools/pool_autodispatch.sh's revision_available_cached) already tolerates "not yet provisioned" by
    # skipping the job for later cycles, so this never needs to block the submit call on a real network/rsync
    # round trip -- and in tests AGENT_GATEWAY_POOL_PROVISION_SH points at a stub that never touches a real node.
    # FLOCK, ONE PROVISIONING AT A TIME (2026-09-29 re-review, M2): several pool-submit calls arriving close
    # together used to each spawn their OWN concurrent `pool_provision.sh` background process -- a thundering
    # herd of simultaneous rsync/ssh sessions against the SAME pool nodes. `flock LOCKFILE cmd...` (not `-n`)
    # serializes them: this Popen call still returns immediately (the wait happens INSIDE the backgrounded
    # process, never blocking cmd_pool itself), but only one provisioning run against the lock actually
    # executes at a time; the rest queue up on the same lock in whatever order they arrived.
    os.makedirs(os.path.dirname(POOL_PROVISION_LOCK), exist_ok=True)
    subprocess.Popen([FLOCK_BIN, POOL_PROVISION_LOCK, "bash", POOL_PROVISION_SH, "--revision", sha, "--isolated"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    job_line = "%s cd ~/derisk-pool/revisions/%s && mkdir -p %s && SIM_NO_PROVENANCE=1 SIM_BACKEND=numpy .venv/bin/python -u -m %s %s" % (
        AGENT_JOB_MARKER, sha, job_out_dir_rel, runner, " ".join(args))
    res = subprocess.run(["bash", POOL_QUEUE_SH, "add", job_line, "--checked", checked])
    return res.returncode


# ------------------------------------------------------------------------------------------------------- main

def main(argv):
    if len(argv) < 1 or argv[0] not in ("gpu", "pool"):
        print("usage: agent_gateway_submit.py {gpu|pool} --sha <40hex> --runner research.runners.<name> "
              "[--prereg <path>] [--checked <reason>] -- <args...>", file=sys.stderr)
        return 2
    kind, rest = argv[0], argv[1:]
    return cmd_gpu(rest) if kind == "gpu" else cmd_pool(rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
