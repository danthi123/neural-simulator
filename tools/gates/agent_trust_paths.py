"""agent_trust_paths (class TP, BLOCK) -- an AGENT-authored change to code that runs with the owner's trust reaches
main without an explicit review.

WHY (2026-09-30 security review of the sim-agent harness, MEDIUM). Everything srv-host-0 runs as dant123 (member
of `docker` and `sudo`, i.e. root-equivalent) or as root at boot is built from `/home/dant123/data/sim-agent/src`,
which tracks GitHub main: the ctl forced command (tools/agent_harness_ctl.sh), update.sh, the Dockerfile and
entrypoint, the nft and systemd files. On the workstation, core.hooksPath=tools/githooks and .claude/settings.json
hooks execute tools/githooks/** (and what those run: tools/gates/**, tools/check_docs.py, ...) and .claude/** at
the owner's next commit or Claude session; tools/agent_gateway*.{sh,py} is the workstation's forced command. The
agent's ONLY route into main is a `research/draft-*` branch the owner merges, and nothing flagged a diff to those
paths.

WHAT IT BLOCKS. Hook mode (`check(paths)`, via pre-commit / pre-merge-commit): while a merge (MERGE_HEAD), a
conflicted cherry-pick (CHERRY_PICK_HEAD) or a squash merge (SQUASH_MSG) is being committed, if any incoming
commit is reachable from a `research/draft-*` ref (refs/heads/research/draft-* or refs/remotes/<remote>/
research/draft-*, or the merge message names one) AND the incoming side changes a TRUSTED path, the commit is
blocked until the owner sets SIM_AGENT_TRUST_ACK to that exact incoming sha (full, or at least 12 hex digits) --
one acknowledgement per reviewed tip, never a standing waiver. Commits that are not agent-originated are not this
gate's business.

WHAT IT CANNOT SEE, AND THE CONTROL FOR THAT. A fast-forward, a rebase and a clean cherry-pick run no commit hook
at all. And a merge applies the incoming tree BEFORE the hook runs, so an agent change to this gate, to
tools/gates/__init__.py or to tools/githooks/** is already the code that executes. The hook is therefore a
tripwire for the honest path; the CONTROL is the review CLI run from main BEFORE merging, which executes main's
own (trusted) copy:

    .venv/bin/python -m tools.gates.agent_trust_paths gitea/research/draft-<topic>   # exit 1 = review needed

and, on the server, `git -C .../src diff HEAD..origin/main -- <the same paths>` before `merge --ff-only`
(docs/AGENT_HARNESS_SETUP.md). Never check out an agent branch on the workstation with hooks enabled
(`git -c core.hooksPath=/dev/null ...`), as tools/agent_gateway_submit.py already does for its job worktrees.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

NAME = "agent-trust-paths"
CLASS_ID = "TP"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GIT_TIMEOUT = 60

# Paths whose code runs with the owner's (or root's) trust without the owner invoking it by hand. A path matches
# when it equals an entry or starts with an entry that ends in "/" (or with a prefix entry marked "*").
TRUST_PATHS = (
    "tools/agent_container/",          # the image, entrypoint, update.sh, host nft/systemd/sbin files (server)
    "tools/agent_harness_ctl.sh",      # the server's ctl forced command
    "tools/agent_gateway*",            # the workstation's forced command + its parser/submitter
    "tools/githooks/",                 # runs at every commit on the workstation
    "tools/gates/",                    # imported by the pre-commit hook (this gate included)
    "tools/check_docs.py", "tools/claim_check.py", "tools/biology_check.py",   # run by the pre-commit hook
    "tools/rag/update_indexes.py",     # run by the post-commit hook
    "tools/live_state.py", "tools/product*", "tools/hermes/sync_skills.sh",  # run by .claude hooks
    "tools/push_both.sh",              # the mandated push path
    ".claude/",                        # Claude Code settings, hooks and skills: the owner's next session
)
_DRAFT_REF = re.compile(r"^refs/(?:heads|remotes/[^/]+)/research/draft-")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def is_trust_path(path):
    for entry in TRUST_PATHS:
        if entry.endswith("*"):
            if path.startswith(entry[:-1]):
                return True
        elif entry.endswith("/"):
            if path.startswith(entry):
                return True
        elif path == entry:
            return True
    return False


def _git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=_GIT_TIMEOUT)
    return r.stdout if r.returncode == 0 else ""


def _acked(incoming, ack):
    ack = (ack or "").strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{12,40}", ack)) and all(sha.startswith(ack) for sha in incoming)


def decide(agent_commits, changed_paths, incoming, ack):
    """Pure decision: problems (empty = pass) for agent-originated incoming commits that change trusted paths."""
    if not agent_commits:
        return []
    trusted = sorted(p for p in changed_paths if is_trust_path(p))
    if not trusted or _acked(incoming, ack):
        return []
    tip = incoming[0] if incoming else "?"
    shown = ", ".join(trusted[:12]) + (" ... (+%d more)" % (len(trusted) - 12) if len(trusted) > 12 else "")
    return ["incoming commit(s) from a research/draft-* branch (the agent's) change trusted path(s): %s. These run "
            "with your trust (server forced command/image/firewall, git hooks, Claude Code hooks). Review: git diff "
            "HEAD...%s -- %s ; then commit with SIM_AGENT_TRUST_ACK=%s (that exact sha only)."
            % (shown, tip[:12], " ".join(trusted[:12]), tip)]


def _incoming(repo):
    """(kind, [incoming shas]) for a merge/cherry-pick/squash in progress in `repo`, else (None, [])."""
    git_dir = _git(repo, "rev-parse", "--absolute-git-dir").strip()
    if not git_dir:
        return None, []
    for head in ("MERGE_HEAD", "CHERRY_PICK_HEAD"):
        path = os.path.join(git_dir, head)
        if os.path.exists(path):
            shas = [l.strip() for l in open(path) if _SHA.match(l.strip())]
            if shas:
                return head, shas
    path = os.path.join(git_dir, "SQUASH_MSG")
    if os.path.exists(path):
        shas = [m.group(1) for m in re.finditer(r"^commit ([0-9a-f]{40})$", open(path, errors="replace").read(),
                                                   re.MULTILINE)]
        if shas:
            return "SQUASH_MSG", shas
    return None, []


def _agent_commits(repo, incoming, merge_msg):
    new = set(_git(repo, "rev-list", *incoming, "--not", "HEAD").split())
    if not new:
        new = set(incoming)
    if re.search(r"research/draft-", merge_msg or ""):
        return new
    drafts = [r for r in _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes").split()
              if _DRAFT_REF.match(r)]
    if not drafts:
        return set()
    return new & set(_git(repo, "rev-list", *drafts, "--not", "HEAD").split())


def check_repo(repo, ack=None):
    ack = os.environ.get("SIM_AGENT_TRUST_ACK", "") if ack is None else ack
    kind, incoming = _incoming(repo)
    if not incoming:
        return []
    git_dir = _git(repo, "rev-parse", "--absolute-git-dir").strip()
    msg_path = os.path.join(git_dir, "MERGE_MSG")
    merge_msg = open(msg_path, errors="replace").read() if kind == "MERGE_HEAD" and os.path.exists(msg_path) else ""
    agent = _agent_commits(repo, incoming, merge_msg)
    if not agent:
        return []
    if kind == "MERGE_HEAD":
        changed = set()
        for sha in incoming:
            base = _git(repo, "merge-base", "HEAD", sha).strip()
            changed |= set(_git(repo, "diff", "--name-only", base or "HEAD", sha).split("\n"))
    else:   # a cherry-pick or squash: what is being committed is the staged change
        changed = set(_git(repo, "diff", "--cached", "--name-only", "HEAD").split("\n"))
    changed.discard("")
    return decide(agent, changed, incoming, ack)


def check(paths):
    return check_repo(_ROOT)


def selftest():
    """FAILING DIRECTION FIRST: an agent commit touching a trusted path, unacknowledged, MUST be caught."""
    bad = []
    tip = "a" * 40
    if not decide({tip}, {"tools/githooks/pre-commit", "README.md"}, [tip], ""):
        bad.append("did NOT catch an agent change to tools/githooks/pre-commit")
    if not decide({tip}, {"tools/agent_container/server-stack/host/etc/nftables.d/sim-agent.nft"}, [tip], ""):
        bad.append("did NOT catch an agent change to the host firewall file")
    if not decide({tip}, {".claude/settings.json"}, [tip], "b" * 40):
        bad.append("accepted an acknowledgement for a DIFFERENT sha")
    if not decide({tip}, {"tools/agent_gateway_submit.py"}, [tip], tip[:6]):
        bad.append("accepted a 6-digit prefix as an acknowledgement")
    if decide({tip}, {"tools/agent_harness_ctl.sh"}, [tip], tip[:12]):
        bad.append("FALSE POSITIVE: ignored a valid acknowledgement of the exact sha")
    if decide({tip}, {"research/runners/_draft_x.py", "docs/drafts/x.md"}, [tip], ""):
        bad.append("FALSE POSITIVE: flagged an agent change to ordinary paths")
    if decide(set(), {"tools/githooks/pre-commit"}, [tip], ""):
        bad.append("FALSE POSITIVE: flagged a non-agent change")
    for p in ("tools/agent_gateway.sh", "tools/agent_gateway_parse.py", "tools/gates/coverage.py"):
        if not is_trust_path(p):
            bad.append("%s is not treated as trusted" % p)
    if is_trust_path("tools/gates_extra.py") or is_trust_path("research/findings/x.md"):
        bad.append("a path outside the trusted set matched")
    return bad


def _review(repo, ref):
    """Pre-merge review: every trusted path the ref changes relative to HEAD (merge-base..ref), any origin."""
    base = _git(repo, "merge-base", "HEAD", ref).strip()
    if not base:
        print("agent_trust_paths: cannot find a merge base between HEAD and %s" % ref)
        return 2
    changed = [p for p in _git(repo, "diff", "--name-only", base, ref).split("\n") if p]
    trusted = sorted(p for p in changed if is_trust_path(p))
    if not trusted:
        print("agent_trust_paths: %s changes no trusted path (%d file(s) changed)" % (ref, len(changed)))
        return 0
    tip = _git(repo, "rev-parse", ref).strip()
    print("agent_trust_paths: %s (%s) changes %d TRUSTED path(s) -- read every line before merging:" % (
        ref, tip[:12], len(trusted)))
    for p in trusted:
        print("  " + p)
    print("\n  git diff %s..%s -- %s" % (base[:12], tip[:12], " ".join(trusted)))
    print("\nThen merge WITHOUT fast-forward so the commit hook re-checks it, acknowledging this exact tip:")
    print("  SIM_AGENT_TRUST_ACK=%s git merge --no-ff %s" % (tip, ref))
    return 1


def main(argv):
    repo = _ROOT
    args = list(argv)
    if len(args) >= 2 and args[0] == "--repo":
        repo, args = args[1], args[2:]
    if len(args) != 1:
        print("usage: python -m tools.gates.agent_trust_paths [--repo <path>] <ref-to-merge>")
        return 2
    return _review(repo, args[0])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
