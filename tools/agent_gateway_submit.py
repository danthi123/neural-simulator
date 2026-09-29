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
  - any `--out`/`--out=` argument resolves (realpath, no `..` component) under `research/findings/raw/`.
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
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys

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

SHA_RE = re.compile(r'^[0-9a-f]{40}$')
RUNNER_RE = re.compile(r'^research\.runners\.[A-Za-z0-9_]+$')
ARG_RE = re.compile(r'^[A-Za-z0-9_./=:,+@%-]+$')
OUT_RE = re.compile(r'^--out(?:=(.*))?$')


def die(msg, code=1):
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


def git(*args, cwd=ROOT):
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True)


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


def validate_sha(sha):
    if not SHA_RE.match(sha):
        die(f"--sha must be exactly 40 hex characters, got: {sha!r}")
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


def find_out_paths(args):
    outs = []
    i = 0
    while i < len(args):
        a = args[i]
        m = OUT_RE.match(a)
        if m:
            if m.group(1) is not None:
                outs.append(m.group(1))
            else:
                i += 1
                if i >= len(args):
                    die("--out needs a value")
                outs.append(args[i])
        i += 1
    return outs


def validate_out_paths(outs):
    allowed = os.path.realpath(os.path.join(ROOT, "research", "findings", "raw"))
    for o in outs:
        if ".." in o.split("/"):
            die(f"--out path contains '..': {o!r}")
        resolved = os.path.realpath(o if os.path.isabs(o) else os.path.join(ROOT, o))
        if resolved != allowed and not resolved.startswith(allowed + os.sep):
            die(f"--out path does not resolve under research/findings/raw/: {o!r}")


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


def ensure_worktree(sha):
    path = os.path.join(JOBS_DIR, sha)
    if os.path.isdir(path):
        head = git("rev-parse", "HEAD", cwd=path)
        if head.returncode == 0 and head.stdout.strip() == sha:
            return path
        # stale/wrong -- remove the registration and the directory, then recreate cleanly below.
        git("worktree", "remove", "--force", path)
        subprocess.run(["rm", "-rf", path])
    os.makedirs(JOBS_DIR, exist_ok=True)
    res = git("worktree", "add", "--detach", path, sha)
    if res.returncode != 0:
        die(f"could not create the job checkout at {path}: {res.stderr.strip()}")
    return path


def build_bwrap_argv(checkout, out_dir, inner_cmd):
    """SANDBOX for an agent-authored GPU job (2026-09-29 security review, H2): the agent can author runner code,
    so the job runs with `--ro-bind / /` (the whole host filesystem, READ-ONLY) plus tmpfs masks over every
    credential store this workstation account has (SSH keys, AWS/gitea/npm/docker/rust credentials, the GNOME
    keyring, Claude Code's own session/auth files, GPG), writable binds ONLY for the job's own pinned checkout
    and the shared output directory, no network (`--unshare-net`), only the /dev/nvidia* device nodes
    `--dev-bind`ed in (so cupy can see the GPU and nothing else under /dev), and `--die-with-parent` so a job
    can never outlive gpu_queue.sh's own process tree. This does not make an agent-authored job TRUSTED -- it
    bounds what a malicious one can reach: no credentials, no network egress, no write access outside its own
    checkout/output dir, and no discovery of MORE of the host to escalate from before this container of jobs is
    itself only ever launched by the ALREADY-restricted tools/agent_gateway.sh trust boundary.
    """
    home = HOME
    tmpfs_targets = [
        os.path.join(home, ".ssh"),
        os.path.join(home, ".aws"),
        os.path.join(home, ".claude-config"),
        os.path.join(home, ".config"),   # no CUDA-specific subpath found under ~/.config on this workstation
                                          # (verified 2026-09-29, ~/.cupy and ~/.nv are separate TOP-LEVEL dirs,
                                          # already left readable via the blanket --ro-bind / /); if one ever
                                          # appears, carve out a --ro-bind for that specific subpath here first.
        os.path.join(home, ".gnupg"),
        os.path.join(home, ".local", "share", "keyrings"),
        # additional credential-bearing dirs found on this workstation (2026-09-29 audit, not exhaustive):
        os.path.join(home, ".docker"),
        os.path.join(home, ".cargo"),
        os.path.join(home, ".pki"),
        os.path.join(home, ".npm"),
    ]
    file_masks = [
        # Claude Code's own session/auth files and the Bitwarden ssh-agent socket are FILES, not directories --
        # tmpfs only masks directories, so these are individually bind-masked with /dev/null instead.
        os.path.join(home, ".claude.json"),
        os.path.join(home, ".claude.json.backup"),
        os.path.join(home, ".bitwarden-ssh-agent.sock"),
        os.path.join(home, ".git-credentials"),
    ]
    argv = [BWRAP, "--ro-bind", "/", "/", "--die-with-parent", "--unshare-net"]
    for t in tmpfs_targets:
        if os.path.isdir(t):
            argv += ["--tmpfs", t]
    for f in file_masks:
        if os.path.exists(f):
            argv += ["--bind", "/dev/null", f]
    argv += ["--bind", checkout, checkout]
    argv += ["--bind", out_dir, out_dir]
    for dev in sorted(glob.glob("/dev/nvidia*")):
        argv += ["--dev-bind", dev, dev]
    argv += ["--unsetenv", "AWS_ACCESS_KEY_ID", "--unsetenv", "AWS_SECRET_ACCESS_KEY",
             "--unsetenv", "ANTHROPIC_API_KEY", "--unsetenv", "GH_TOKEN", "--unsetenv", "SSH_AUTH_SOCK"]
    argv += ["--", "bash", "-c", inner_cmd]
    return argv


def cmd_gpu(argv):
    sha, runner, prereg, _checked, args = parse_common("gpu", argv)
    validate_sha(sha)
    validate_runner(sha, runner)
    validate_args(args)
    validate_out_paths(find_out_paths(args))
    validate_gate_seed(sha, args, prereg)
    ensure_bwrap()
    checkout = ensure_worktree(sha)
    out_dir = os.path.join(ROOT, "research", "findings", "raw")
    os.makedirs(out_dir, exist_ok=True)
    inner = "cd %s && SIM_BACKEND=cupy .venv/bin/python -u -m %s %s" % (
        shlex.quote(checkout), shlex.quote(runner), " ".join(shlex.quote(a) for a in args))
    bwrap_argv = build_bwrap_argv(checkout, out_dir, inner)
    job_line = " ".join(shlex.quote(a) for a in bwrap_argv)
    res = subprocess.run(["bash", GPU_QUEUE_SH, "add", job_line])
    return res.returncode


# -------------------------------------------------------------------------------------------------------- pool

def cmd_pool(argv):
    sha, runner, prereg, checked, args = parse_common("pool", argv)
    validate_sha(sha)
    validate_runner(sha, runner)
    validate_args(args)
    validate_out_paths(find_out_paths(args))
    validate_gate_seed(sha, args, prereg)
    # BEST-EFFORT, NON-BLOCKING provisioning of this revision ahead of dispatch (2026-09-29, H2: "build the
    # standard provisioned job line ... from the structured request"). The job line below pins to
    # ~/derisk-pool/revisions/<sha>, the exact layout tools/pool_provision.sh --isolated creates; the dispatcher
    # (tools/pool_autodispatch.sh's revision_available_cached) already tolerates "not yet provisioned" by
    # skipping the job for later cycles, so this never needs to block the submit call on a real network/rsync
    # round trip -- and in tests AGENT_GATEWAY_POOL_PROVISION_SH points at a stub that never touches a real node.
    subprocess.Popen(["bash", POOL_PROVISION_SH, "--revision", sha, "--isolated"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    job_line = "cd ~/derisk-pool/revisions/%s && SIM_NO_PROVENANCE=1 SIM_BACKEND=numpy .venv/bin/python -u -m %s %s" % (
        sha, runner, " ".join(args))
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
