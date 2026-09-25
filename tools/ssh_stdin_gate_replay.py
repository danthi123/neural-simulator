"""Replay the CLASS SR gate (tools/gates/ssh_stdin_in_read_loop.py) over git history.

For each of the last N commits that touch a `*.sh` file, emulate the pre-commit hook: the staged set is the commit's
Added/Copied/Modified/Renamed `*.sh` files against its FIRST parent, the corpus (which scripts drain their own stdin)
is the commit's own tree, and every file is read from the commit's blobs. Records which commits the gate would have
BLOCKED and on which lines, so each block can be reviewed as a true or a false positive. Optionally runs an older
revision of the gate on the same inputs for comparison (`--compare-old <rev>`), both as that version was wired (only
ADDED files reach it) and with modified files included.

    python tools/ssh_stdin_gate_replay.py --n 2000 --all-refs --compare-old a497db159 \
        --out research/coordination/sr_gate_replay_2026-09-25.json

Read-only: it runs `git rev-list`, `git diff-tree`, `git ls-tree` and `git cat-file` and writes only --out.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import tools.gates.ssh_stdin_in_read_loop as sr  # noqa: E402


def _git(args, data=None):
    r = subprocess.run(["git"] + args, cwd=_ROOT, input=data, capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace")[:300]))
    return r.stdout


# The manual review of every block this replay produced, written down as rules so a re-run re-applies it: each
# rule is (file, text on the flagged line, verdict). A block no rule matches is reported as UNREVIEWED.
_REVIEW = (
    ("research/coordination/b2b_queue_next_wave.sh", '"$QUEUE_TOOL" add',
     "TRUE POSITIVE, incident 1: pool_queue.sh's probes (no -n) read the job-file loop's stdin (fixed 6406924ee)"),
    ("tools/pool_autodispatch.sh", 'ssh "${SSH_F[@]}"',
     "TRUE POSITIVE, incident 2: revision_available() ssh without -n, called from pop_job's while-read loop "
     "(fixed 096dfdae0)"),
    ("tools/aws_idle_stop.sh", 'ssh -i "$state_key"',
     "TRUE POSITIVE, incident 3: per-instance ssh probes in `while read iid ... done <<<\"$ids\"` (fixed on "
     "research/aws-pool-stop-start-safety)"),
    ("tools/aws_idle_stop.sh", "tools/pool_sync.sh",
     "TRUE POSITIVE, incident 3 (lane A WIP a144f6d46): pool_sync.sh, which runs ssh on its stdin, called in the "
     "same loop; that lane's own review flagged it HIGH #2 and 75dfb7960 fixed it"),
)


def _classify(problem):
    where = problem.split(" -- ")[0].rsplit(":", 1)[0]
    near = problem.split("Near: ", 1)[-1]
    for path, needle, verdict in _REVIEW:
        if where == path and needle in near:
            return verdict
    return "UNREVIEWED"


def _tally(rows):
    out = {}
    for r in rows:
        for p in r["problems"]:
            v = _classify(p)
            out[v] = out.get(v, 0) + 1
    return out


def _load_old(rev):
    src = _git(["show", "%s:tools/gates/ssh_stdin_in_read_loop.py" % rev]).decode("utf-8")
    mod = types.ModuleType("sr_old")
    mod.__file__ = os.path.join(_ROOT, "tools", "gates", "ssh_stdin_in_read_loop.py")
    exec(compile(src, "sr_old@%s" % rev, "exec"), mod.__dict__)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--all-refs", action="store_true", help="every ref (unmerged branches too), not only --rev")
    ap.add_argument("--compare-old", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    t0 = time.time()
    head = _git(["rev-parse", a.rev]).decode().strip()
    refs = ["--all"] if a.all_refs else [head]
    commits = _git(["rev-list", "-n", str(a.n), "--full-history"] + refs + ["--", "*.sh"]).decode().split()
    old = _load_old(a.compare_old) if a.compare_old else None

    blob_text = {}      # blob sha -> text
    blob_summary = {}   # blob sha -> summary (or None when the parser raised)
    parse_errors = {}
    rows = []
    old_rows = []
    n_files = 0
    rsync_loop_sites = []
    stray_blobs = {}
    for idx, c in enumerate(commits):
        meta = _git(["log", "-1", "--format=%H%x00%P%x00%ad%x00%s", "--date=iso-strict", c]).decode().rstrip("\n")
        full, parents, date, subject = meta.split("\x00", 3)
        parent = parents.split()[0] if parents.split() else None
        if parent:
            diff = _git(["diff-tree", "-r", "-z", "--no-commit-id", "--name-status", "--diff-filter=ACMR",
                         parent, c, "--", "*.sh"]).decode("utf-8", "replace").split("\0")
        else:
            diff = _git(["diff-tree", "-r", "-z", "--no-commit-id", "--name-status", "--root", "--diff-filter=ACMR",
                         c, "--", "*.sh"]).decode("utf-8", "replace").split("\0")
        touched = {}
        k = 0
        while k < len(diff) - 1:
            status = diff[k]
            if not status:
                k += 1
                continue
            if status[0] in "RC":
                touched[diff[k + 2]] = status[0]
                k += 3
            else:
                touched[diff[k + 1]] = status[0]
                k += 2
        if not touched:
            continue
        tree = _git(["ls-tree", "-r", "-z", c]).decode("utf-8", "replace").split("\0")
        files = {}
        for rec in tree:
            if "\t" not in rec:
                continue
            m, path = rec.split("\t", 1)
            parts = m.split()
            if len(parts) >= 3 and parts[1] == "blob" and path.endswith(".sh"):
                files[path] = parts[2]
        need = [sha for sha in set(files.values()) if sha not in blob_text]
        if need:
            got = sr._cat_blobs(_ROOT, need)
            blob_text.update(got)
        summaries = {}
        for path, sha in files.items():
            if sha not in blob_summary:
                try:
                    blob_summary[sha] = sr._summarize(blob_text.get(sha, ""))
                except Exception as e:  # counted, never fatal: the gate reports these as LOUD problems
                    blob_summary[sha] = None
                    parse_errors[sha] = "%s %s: %s: %s" % (c[:10], path, type(e).__name__, e)
            summaries[path] = blob_summary[sha]
            if blob_summary[sha] is not None and blob_summary[sha].strays:
                stray_blobs.setdefault(sha, "%s %s: %d unmatched token(s)" % (c[:10], path, blob_summary[sha].strays))
        known = sr._known_scripts(summaries)
        texts = {p: blob_text.get(files[p], "") for p in files}
        cands = sorted(p for p in touched if p in files)
        n_files += len(cands)
        problems = sr._report(texts, cands, True)
        for p in cands:
            s = summaries.get(p)
            if s is not None:
                for sites in [s.top_sites] + list(s.func_sites.values()):
                    for st, kd, ln, nm in sites:
                        if st == "loop" and kd == "rsync":
                            rsync_loop_sites.append("%s %s:%d" % (c[:10], p, ln))
        if problems:
            rows.append({"commit": full, "date": date, "subject": subject[:160],
                         "touched": {p: touched[p] for p in cands}, "problems": problems,
                         "review": sorted({_classify(p) for p in problems})})
        if old is not None:
            added_only, acmr = [], []
            for p in cands:
                hits = old._find_unprotected(texts[p])
                if hits:
                    line = ["%s:%d %s" % (p, h[0], h[1]) for h in hits]
                    acmr.extend(line)
                    if touched[p] == "A":
                        added_only.extend(line)
            if acmr:
                old_rows.append({"commit": full, "subject": subject[:120], "as_wired_added_only": added_only,
                                 "with_modified": acmr})
        if (idx + 1) % 250 == 0:
            print("  %d/%d commits, %d blocked, %.0fs" % (idx + 1, len(commits), len(rows), time.time() - t0),
                  file=sys.stderr)

    out = {
        "provenance": {
            "cmd": "python tools/ssh_stdin_gate_replay.py " + " ".join(argv if argv is not None else sys.argv[1:]),
            "script": "tools/ssh_stdin_gate_replay.py",
            # gates/device_and_cost: static analysis of git history on the host CPU; no simulator backend runs
            "device": "cpu (host python; no simulator backend)",
            "gate_at": head,
            "git_sha": _git(["rev-parse", "HEAD"]).decode().strip(),
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        },
        "refs": "all" if a.all_refs else head,
        "commits_touching_sh": len(commits),
        "oldest_commit": commits[-1] if commits else None,
        "staged_sh_files_checked": n_files,
        "distinct_blobs_parsed": len(blob_summary),
        "parse_errors": sorted(parse_errors.values()),
        "blobs_with_unmatched_tokens": sorted(stray_blobs.values()),
        "commits_blocked": len(rows),
        "blocks_by_review_verdict": _tally(rows),
        "blocked": rows,
        "rsync_in_read_loop_sites_not_flagged": rsync_loop_sites,
        "seconds": round(time.time() - t0, 1),
    }
    if old is not None:
        out["old_gate"] = {
            "rev": a.compare_old,
            "commits_blocked_as_wired_added_only": sum(1 for r in old_rows if r["as_wired_added_only"]),
            "commits_blocked_with_modified": len(old_rows),
            "blocked": old_rows,
        }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print("replayed %d commits (%d staged .sh files, %d blobs) in %.0fs: %d blocked, %d parse errors -> %s"
          % (len(commits), n_files, len(blob_summary), time.time() - t0, len(rows), len(parse_errors), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
