#!/usr/bin/env python3
"""tools/inspection_digest.py -- the weekly Claude-inspection digest for the local-model-first transition
(docs/plans/2026-09-29-local-model-transition-PLAN.md section 6, "Inspection digest").

Claude no longer reads the whole history when it checks in on the local model; it reads THIS: one Markdown
report, target under 15K tokens, covering everything that changed since the last inspection -- commits per
branch, new findings (with their verdict line and whether PROVISIONAL or re-derived), NEEDS-CLAUDE items from
the runbook, new FAILURE_LOG rows, and a pool/GPU utilization summary.

    .venv/bin/python tools/inspection_digest.py                  # since research/coordination/last_inspection.txt, else 7 days
    .venv/bin/python tools/inspection_digest.py --since <sha>
    .venv/bin/python tools/inspection_digest.py --since 2026-09-25
    .venv/bin/python tools/inspection_digest.py --mark            # after reading it, record main's HEAD as the new marker

--mark only records the marker; it never gates on having actually read the output, so run it once per real
inspection, not on a dry run used just to check size/sections.

Every section is best-effort and caps its own length (commit lines, findings, failure rows) so the whole report
stays small regardless of how much happened; a capped section says how much was left out rather than silently
truncating. research/queue/dispatch.log carries no per-line date (only a time-of-day), so its section is always
"most recent N lines", not a since-filtered range -- documented in the section itself, not glossed over.
"""
from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER_FILE = ROOT / "research" / "coordination" / "last_inspection.txt"
RUNBOOK = ROOT / "research" / "coordination" / "LOCAL_LLM_RUNBOOK.md"
FAILURE_LOG = ROOT / "research" / "FAILURE_LOG.md"
FINDINGS_DIR = ROOT / "research" / "findings"
DISPATCH_LOG = ROOT / "research" / "queue" / "dispatch.log"
GPU_LOG = ROOT / "research" / "queue" / "gpu_queue.log"

DEFAULT_LOOKBACK_DAYS = 7
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
MAX_BRANCH_COMMIT_LINES = 200
MAX_FINDINGS_SHOWN = 40
MAX_FAILURE_ROWS_SHOWN = 30
MAX_NEEDS_CLAUDE_SHOWN = 40
MAX_GATE_FAILURE_COMMITS = 20
DISPATCH_TAIL_LINES = 400
GPU_LOG_GREP_TIMEOUT_S = 60


# ----------------------------------------------------------------------------------------------- git helpers

def git(*args, timeout=30):
    try:
        res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout, check=False)
        return res.returncode, res.stdout
    except subprocess.TimeoutExpired:
        return 1, ""


def is_sha(s):
    return bool(SHA_RE.fullmatch(s))


def sha_commit_date(sha):
    rc, out = git("show", "-s", "--format=%cs", sha)
    out = out.strip()
    return out if rc == 0 and out else None


# ------------------------------------------------------------------------------------------------ since / date

def resolve_since(explicit_since):
    """Returns (since, label, since_is_sha)."""
    if explicit_since:
        return explicit_since, f"--since {explicit_since}", is_sha(explicit_since)
    if MARKER_FILE.exists():
        sha = MARKER_FILE.read_text().strip()
        if sha:
            return sha, f"last inspection ({sha[:12]})", True
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    return cutoff, f"{DEFAULT_LOOKBACK_DAYS} days ago ({cutoff}; no {MARKER_FILE.name} yet)", False


def since_date(since, since_is_sha):
    """A YYYY-MM-DD floor usable for date-based filtering (findings/needs-claude/failure-log/gpu-log),
    regardless of whether `since` is a sha or a date."""
    if since_is_sha:
        d = sha_commit_date(since)
        return d or "1970-01-01"
    # explicit_since may itself be a date string, or something git --since already understands (e.g. "7 days
    # ago"); best-effort: if it already looks like YYYY-MM-DD, use it directly.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", since):
        return since
    return "1970-01-01"   # unrecognized form -- do not silently drop everything; show more rather than less


# ------------------------------------------------------------------------------------------ commits per branch

def list_target_branches():
    rc, out = git("branch", "--list", "main", "research/*", "--format=%(refname:short)")
    names = [line.strip() for line in out.splitlines() if line.strip()]
    main_b = [b for b in names if b == "main"]
    draft_b = sorted(b for b in names if b.startswith("research/draft-"))
    other_b = sorted(b for b in names if b.startswith("research/") and not b.startswith("research/draft-"))
    return main_b, draft_b, other_b


def branch_tip_date(branch):
    rc, out = git("log", "-1", "--format=%cs", branch)
    return out.strip() if rc == 0 else ""


def branch_commits(branch, since, since_is_sha, floor_date):
    # Non-main branches: exclude anything already reachable from main (--not main), so a feature branch shows
    # only ITS OWN new commits instead of repeating the whole shared history it was cut from -- without this, a
    # branch cut from yesterday's main re-listed 100+ commits it shares with main for zero new information.
    exclude = [] if branch == "main" else ["--not", "main"]
    if since_is_sha:
        rc, out = git("log", "--oneline", "--no-decorate", f"{since}..{branch}", *exclude)
        if rc != 0:
            rc, out = git("log", "--oneline", "--no-decorate", f"--since={floor_date}", branch, *exclude)
    else:
        rc, out = git("log", "--oneline", "--no-decorate", f"--since={since}", branch, *exclude)
    lines = [l[:160] for l in out.splitlines() if l.strip()]
    return lines


def render_commits_section(since, since_is_sha, floor_date):
    out = ["## 1. Commits per branch\n"]
    main_b, draft_b, other_b = list_target_branches()
    budget = MAX_BRANCH_COMMIT_LINES
    truncated_branches = []

    def render_group(title, branches):
        nonlocal budget
        rows = []
        for b in branches:
            if branch_tip_date(b) < floor_date:
                continue   # the branch's own tip predates the window -- it cannot have newer commits
            commits = branch_commits(b, since, since_is_sha, floor_date)
            if not commits:
                continue
            if budget <= 0:
                truncated_branches.append(b)
                continue
            take = commits[:max(1, min(len(commits), budget))]
            budget -= len(take)
            more = len(commits) - len(take)
            rows.append(f"- **{b}** ({len(commits)} commit{'s' if len(commits) != 1 else ''}):")
            for c in take:
                rows.append(f"    - `{c}`")
            if more > 0:
                rows.append(f"    - ... {more} more (run `git log {b}` locally)")
        if rows:
            out.append(f"### {title}\n")
            out.extend(rows)
            out.append("")

    render_group("main", main_b)
    render_group("research/draft-*", draft_b)
    render_group("other research/*", other_b)
    if truncated_branches:
        out.append(f"_(commit-line budget exhausted; {len(truncated_branches)} more branch(es) with commits "
                    f"not shown: {', '.join(truncated_branches[:15])}{'...' if len(truncated_branches) > 15 else ''})_\n")
    if len(out) == 1:
        out.append("_(no commits on main/research/* since the window start)_\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------- findings

def new_finding_files(floor_date):
    rc, out = git("log", f"--since={floor_date}", "--branches=main", "--branches=research/*",
                   "--diff-filter=A", "--name-only", "--pretty=format:", "--",
                   "research/findings/*.md")
    if rc != 0:
        return []
    paths = sorted({line.strip() for line in out.splitlines() if line.strip().endswith(".md")})
    return paths


def summarize_finding(rel_path):
    fname = Path(rel_path).name
    status_tokens = []
    for tok in ("PREREGISTRATION", "AMENDMENT", "NO-GO", "GO", "DESIGN", "NOT-GATEABLE"):
        if tok in fname.upper():
            status_tokens.append(tok)
    text = ""
    abs_path = ROOT / rel_path
    try:
        text = abs_path.read_text(errors="replace")[:4000]
    except OSError:
        pass
    provisional = "provisional" in text.lower()
    frontmatter = {}
    if text.startswith("---"):
        for line in text.splitlines()[1:30]:
            if line.strip() == "---":
                break
            if ":" in line:
                k, _, v = line.partition(":")
                frontmatter[k.strip()] = v.strip()
    return {
        "path": rel_path,
        "filename_tags": status_tokens,
        "status": frontmatter.get("status", ""),
        "claim_check": frontmatter.get("claim_check", ""),
        "prereg": frontmatter.get("prereg", ""),
        "provisional": provisional,
    }


def render_findings_section(floor_date):
    out = ["## 2. New findings\n"]
    paths = new_finding_files(floor_date)
    if not paths:
        out.append("_(no new files under research/findings/*.md since the window start)_\n")
        return "\n".join(out)
    shown = paths[:MAX_FINDINGS_SHOWN]
    for p in shown:
        info = summarize_finding(p)
        tags = "/".join(info["filename_tags"]) or "(untagged)"
        mark = "PROVISIONAL" if info["provisional"] else "not marked provisional (re-derived, or predates the convention)"
        bits = [f"tags={tags}", mark]
        if info["status"]:
            bits.append(f"status={info['status']}")
        if info["claim_check"]:
            bits.append(f"claim_check={info['claim_check']}")
        if info["prereg"]:
            bits.append(f"prereg={info['prereg']}")
        out.append(f"- `{p}` -- {'; '.join(bits)}")
    if len(paths) > len(shown):
        out.append(f"\n_(... {len(paths) - len(shown)} more new finding file(s) not shown)_")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------- NEEDS CLAUDE

NEEDS_CLAUDE_ITEM_RE = re.compile(r"^-\s*(\d{4}-\d{2}-\d{2})\b")


def needs_claude_items(floor_date):
    if not RUNBOOK.exists():
        return []
    text = RUNBOOK.read_text(errors="replace")
    m = re.search(r"^##\s*NEEDS CLAUDE\s*$", text, flags=re.MULTILINE)
    if not m:
        return []
    rest = text[m.end():]
    m2 = re.search(r"^##\s", rest, flags=re.MULTILINE)
    section = rest[: m2.start()] if m2 else rest
    items = []
    for line in section.splitlines():
        im = NEEDS_CLAUDE_ITEM_RE.match(line.strip())
        if im and im.group(1) >= floor_date:
            items.append(line.strip()[:300])
    return items


def render_needs_claude_section(floor_date):
    out = ["## 3. NEEDS-CLAUDE items (research/coordination/LOCAL_LLM_RUNBOOK.md)\n"]
    items = needs_claude_items(floor_date)
    if not items:
        out.append("_(none dated on/after the window start)_\n")
        return "\n".join(out)
    for it in items[:MAX_NEEDS_CLAUDE_SHOWN]:
        out.append(it if it.startswith("-") else f"- {it}")
    if len(items) > MAX_NEEDS_CLAUDE_SHOWN:
        out.append(f"\n_(... {len(items) - MAX_NEEDS_CLAUDE_SHOWN} more not shown -- read the runbook section directly)_")
    out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------- FAILURE_LOG

FAILURE_ROW_DATE_RE = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|")


def failure_log_rows(floor_date):
    if not FAILURE_LOG.exists():
        return []
    rows = []
    for line in FAILURE_LOG.read_text(errors="replace").splitlines():
        m = FAILURE_ROW_DATE_RE.match(line)
        if not m or m.group(1) < floor_date:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        date = parts[1].strip()
        gate = parts[-2].strip()
        failure = "|".join(parts[2:-2]).strip()
        rows.append((date, failure[:280], gate[:160]))
    return rows


def render_failure_log_section(floor_date):
    out = ["## 4. New research/FAILURE_LOG.md rows\n"]
    rows = failure_log_rows(floor_date)
    if not rows:
        out.append("_(none dated on/after the window start)_\n")
        return "\n".join(out)
    for date, failure, gate in rows[:MAX_FAILURE_ROWS_SHOWN]:
        out.append(f"- **{date}** -- {failure} _(gate: {gate})_")
    if len(rows) > MAX_FAILURE_ROWS_SHOWN:
        out.append(f"\n_(... {len(rows) - MAX_FAILURE_ROWS_SHOWN} more row(s) not shown)_")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------------------- pool / GPU utilization

def dispatch_log_summary():
    if not DISPATCH_LOG.exists():
        return None
    try:
        res = subprocess.run(["tail", "-n", str(DISPATCH_TAIL_LINES), str(DISPATCH_LOG)],
                              capture_output=True, text=True, timeout=15)
        lines = res.stdout.splitlines()
    except Exception:
        return None
    dispatched, refused, restarts = 0, 0, 0
    per_node = {}
    for line in lines:
        if " <- " in line:
            dispatched += 1
            m = re.search(r"\]\s*\d\d:\d\d:\d\d\s+(\S+)\s+<-", line)
            if m:
                per_node[m.group(1)] = per_node.get(m.group(1), 0) + 1
        elif "REFUSING" in line:
            refused += 1
        elif "started" in line and "pool-dispatch" in line:
            restarts += 1
    return {"lines_scanned": len(lines), "dispatched": dispatched, "refused": refused,
            "dispatcher_restarts": restarts, "per_node": per_node}


def gpu_log_summary(floor_date):
    if not GPU_LOG.exists():
        return None
    pattern = r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} (START|DONE)"
    try:
        res = subprocess.run(["grep", "-aE", pattern, str(GPU_LOG)], capture_output=True, text=True,
                              timeout=GPU_LOG_GREP_TIMEOUT_S)
        lines = res.stdout.splitlines()
    except Exception:
        return None
    starts, done_ok, done_fail = 0, 0, 0
    for line in lines:
        date = line[:10]
        if date < floor_date:
            continue
        if " START:" in line:
            starts += 1
        elif " DONE(rc=0):" in line:
            done_ok += 1
        elif " DONE(rc=" in line:
            done_fail += 1
    return {"starts": starts, "done_ok": done_ok, "done_fail": done_fail}


def render_utilization_section(floor_date):
    out = ["## 5. Pool / GPU utilization\n"]
    d = dispatch_log_summary()
    if d is None:
        out.append("- dispatch.log: not found.")
    else:
        nodes = ", ".join(f"{n}={c}" for n, c in sorted(d["per_node"].items()))
        out.append(f"- pool dispatch.log (no per-line date -- last {d['lines_scanned']} lines only): "
                    f"{d['dispatched']} dispatched ({nodes or 'no node breakdown'}), {d['refused']} refused, "
                    f"{d['dispatcher_restarts']} dispatcher restart(s) seen in that window.")
    g = gpu_log_summary(floor_date)
    if g is None:
        out.append("- gpu_queue.log: not found.")
    else:
        out.append(f"- GPU gpu_queue.log since {floor_date}: {g['starts']} job(s) started, "
                    f"{g['done_ok']} finished rc=0, {g['done_fail']} finished with a non-zero rc.")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------------- gate-failure-noted commits

def gate_failure_commits(since, since_is_sha, floor_date):
    grep_pattern = r"gate.*\(fail\|block\)\|\(fail\|block\).*gate"
    if since_is_sha:
        rc, out = git("log", "--oneline", "--no-decorate", "-i", "--grep", grep_pattern,
                       "--branches=main", "--branches=research/*", f"{since}..HEAD")
        if rc != 0:
            rc, out = git("log", "--oneline", "--no-decorate", "-i", "--grep", grep_pattern,
                           "--branches=main", "--branches=research/*", f"--since={floor_date}")
    else:
        rc, out = git("log", "--oneline", "--no-decorate", "-i", "--grep", grep_pattern,
                       "--branches=main", "--branches=research/*", f"--since={since}")
    lines = [l[:160] for l in out.splitlines() if l.strip()]
    return lines


def render_gate_failure_section(since, since_is_sha, floor_date):
    out = ["## 6. Commits noting a gate failure\n"]
    out.append("_(best-effort: no dedicated gate-failure log exists; this scans commit messages for "
                "\"gate\" co-occurring with \"fail\"/\"block\". Authoritative gate state is "
                "docs/FAILURE_GATE_MATRIX.md + research/FAILURE_LOG.md, section 4 above.)_\n")
    commits = gate_failure_commits(since, since_is_sha, floor_date)
    if not commits:
        out.append("_(none found)_\n")
        return "\n".join(out)
    for c in commits[:MAX_GATE_FAILURE_COMMITS]:
        out.append(f"- `{c}`")
    if len(commits) > MAX_GATE_FAILURE_COMMITS:
        out.append(f"\n_(... {len(commits) - MAX_GATE_FAILURE_COMMITS} more not shown)_")
    out.append("")
    return "\n".join(out)


# -------------------------------------------------------------------------- agent-container Claude Code version

AGENT_CTL_HOST = "homelab-agent-ctl"
NPM_VERSION_TIMEOUT_S = 10
SSH_STATE_TIMEOUT_S = 10
# VOLUME_WARN_GB (2026-09-29 security review, M3): "the digest that warns when the volumes exceed a threshold".
# homelab-server is a SHARED box (gitea/nextcloud/home-assistant/etc. all on it too); an agent-authored job that
# writes heavily to /work/sim (the repo volume) or piles up Claude Code versions/state under /home/agent (the
# home volume) has no other backstop (named volumes are not covered by compose.yaml's own log-size cap, M3's
# other half). 20 GB is a generous multiple of a normal repo+venvs checkout on this box -- comfortably above
# routine growth, low enough to catch a genuine runaway well before it threatens the shared disk.
VOLUME_WARN_GB = 20.0


def query_container_claude_version():
    # a thin, monkeypatchable wrapper (tests/test_inspection_digest.py stubs this directly -- never a real ssh
    # call in a test) around `ssh homelab-agent-ctl state`'s `claude_version:` line.
    try:
        res = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", AGENT_CTL_HOST, "state"],
                              capture_output=True, text=True, timeout=SSH_STATE_TIMEOUT_S)
        for line in res.stdout.splitlines():
            if line.startswith("claude_version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def query_container_volume_usage_gb():
    # Same `ssh homelab-agent-ctl state` call as query_container_claude_version (tools/agent_harness_ctl.sh's
    # own `repo_volume_gb:`/`home_volume_gb:` lines, 2026-09-29) -- a SEPARATE ssh round trip here (rather than
    # sharing one call between the two query_* functions) so each stays independently monkeypatchable, matching
    # the existing test seam. Returns {"repo": float|None, "home": float|None}; a line reading "unknown" (the
    # volume could not be found/parsed) or a missing line both map to None.
    result = {"repo": None, "home": None}
    try:
        res = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", AGENT_CTL_HOST, "state"],
                              capture_output=True, text=True, timeout=SSH_STATE_TIMEOUT_S)
        for line in res.stdout.splitlines():
            if line.startswith("repo_volume_gb:"):
                v = line.split(":", 1)[1].strip()
                result["repo"] = float(v) if v != "unknown" else None
            elif line.startswith("home_volume_gb:"):
                v = line.split(":", 1)[1].strip()
                result["home"] = float(v) if v != "unknown" else None
    except Exception:
        pass
    return result


def query_latest_claude_code_version():
    # likewise monkeypatchable; never a real npm call in a test.
    try:
        res = subprocess.run(["npm", "view", "@anthropic-ai/claude-code", "version"],
                              capture_output=True, text=True, timeout=NPM_VERSION_TIMEOUT_S)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def render_agent_container_version_section():
    # owner 2026-09-29 (docs/plans/2026-09-29-local-model-transition-PLAN.md section 9's harness-currency
    # addendum): surface a lag between the sim-agent container's installed Claude Code build and the latest
    # release, so it is visible without a separate check-in. Best-effort both ways -- an unreachable server, or
    # no network for the npm registry lookup, SKIPS that half rather than failing the whole digest.
    out = ["## 7. Agent-container Claude Code version\n"]
    installed = query_container_claude_version()
    latest = query_latest_claude_code_version()

    if installed is None:
        out.append("_(could not reach the sim-agent harness -- `ssh homelab-agent-ctl state` did not answer; "
                    "skipped)_\n")
    else:
        out.append(f"- installed (via the harness's `claude --version`): `{installed}`")
    if latest is None:
        out.append("_(could not resolve the latest published version -- `npm view "
                    "@anthropic-ai/claude-code version` failed or timed out, likely offline; skipped)_\n")
    else:
        out.append(f"- latest published (npm `@anthropic-ai/claude-code`): `{latest}`")
    if installed is not None and latest is not None:
        installed_ver = installed.split()[0] if installed.split() else installed
        if installed_ver == latest:
            out.append("- **up to date.**")
        else:
            out.append(f"- **LAG: installed `{installed_ver}` != latest `{latest}`.** "
                        "`agent start` runs `claude update` + a smoke test on every start "
                        "(tools/agent_container/bin/claude-safe-update.sh); if this persists across several "
                        "inspections, either the smoke test keeps failing (check ~/claude_update.log on the "
                        "container) or the harness has not been started recently.")

    # --- volume disk usage (2026-09-29 security review, M3) --------------------------------------------------
    usage = query_container_volume_usage_gb()
    repo_gb, home_gb = usage.get("repo"), usage.get("home")
    if repo_gb is None and home_gb is None:
        out.append("- volume disk usage: _(could not reach the sim-agent harness, or the volumes were not "
                    "found by `docker system df -v` -- skipped)_")
    else:
        def _fmt(name, gb):
            if gb is None:
                return f"- {name} volume: unknown"
            flag = " **-- OVER the %.0f GB warn threshold**" % VOLUME_WARN_GB if gb > VOLUME_WARN_GB else ""
            return f"- {name} volume: {gb:.2f} GB{flag}"
        out.append(_fmt("repo", repo_gb))
        out.append(_fmt("home", home_gb))
        if (repo_gb or 0) > VOLUME_WARN_GB or (home_gb or 0) > VOLUME_WARN_GB:
            out.append("  **A volume is over the warn threshold** -- homelab-server is a SHARED box; investigate "
                        "before it threatens disk space for gitea/nextcloud/home-assistant/etc. An optional "
                        "size-capped volume (a loop-mounted image file, so growth hits ENOSPC inside the loop "
                        "device instead of the shared filesystem) is documented in docs/AGENT_HARNESS_SETUP.md "
                        "for the owner to opt into.")
    out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------- main

def build_digest(explicit_since):
    since, label, since_is_sha = resolve_since(explicit_since)
    floor_date = since_date(since, since_is_sha)
    parts = [
        f"# Inspection digest -- since {label} (floor date {floor_date})\n",
        f"_Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        f"Run with `--mark` after reading this to record the new checkpoint._\n",
        render_commits_section(since, since_is_sha, floor_date),
        render_findings_section(floor_date),
        render_needs_claude_section(floor_date),
        render_failure_log_section(floor_date),
        render_utilization_section(floor_date),
        render_gate_failure_section(since, since_is_sha, floor_date),
        render_agent_container_version_section(),
    ]
    text = "\n".join(parts)
    approx_tokens = len(text) // 4
    parts.append(f"\n---\n_Approx. size: {len(text)} chars (~{approx_tokens} tokens)._\n")
    return "\n".join(parts)


def mark(main_sha=None):
    if main_sha is None:
        rc, out = git("rev-parse", "main")
        if rc != 0:
            print("could not resolve main's HEAD; marker not written", file=sys.stderr)
            return 1
        main_sha = out.strip()
    MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    MARKER_FILE.write_text(main_sha + "\n")
    print(f"marker written: {main_sha}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None, help="a sha or a date (YYYY-MM-DD); default: last_inspection.txt, else 7 days ago")
    ap.add_argument("--mark", action="store_true", help="also record main's current HEAD as the new checkpoint")
    args = ap.parse_args()

    print(build_digest(args.since))

    if args.mark:
        return mark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
