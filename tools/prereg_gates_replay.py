#!/usr/bin/env python3
"""Replay the two pre-registration ORDER gates over real history: CLASS PRA (`gates/prereg_amendment_order`, an
amendment to an existing prereg landing with run data) and CLASS PR (`gates/prereg_before_run`, a prereg ADDED with
run data). A BLOCKING gate must not false-block real commits, and this is the measurement that says so.

PRA. Every commit in the window goes through the gate's own `_evaluate` on its real trees. Merges are judged against
EVERY parent -- what a `git commit` / `git merge --continue` with MERGE_HEAD present sees; a clean auto-merge is not
checked at all, so this is the stricter reading. With `--baseline REV`, the gate as it stood at REV is replayed on
the same commits and every verdict difference is listed, and the amendment-entry lists of every prereg file at REF
are compared between the two grammars. `merges_blocked_head_only` counts merges whose tree blocks when judged against
the first parent alone -- what a clean `git merge` would have hit had the gate not skipped it.

PR. Non-merge verdicts do not depend on which git command runs the hook once history is written (a commit's parent
IS what `--amend` is now judged against), so they are listed once. For merges: the trees the pre-2026-09-25 gate
blocked during a CLEAN `git merge` (HEAD = first parent and no MERGE_HEAD yet, so its merged-in-unchanged exemption
could not fire), against a conflicted merge finished by `git commit` (exemption applies).

Truth labels are not computed here: a block is a TRUE positive only when read by hand (the gate's docstring records
that reading). The replay reports verdicts, not correctness.

usage: .venv/bin/python tools/prereg_gates_replay.py [--n 2500] [--ref origin/main] [--baseline REV] [--out F.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import tools.gates.prereg_amendment_order as pra  # noqa: E402
import tools.gates.prereg_before_run as pr  # noqa: E402

_ENV = pra._stripped_env()


def _run(*args):
    r = subprocess.run(["git", *args], cwd=_ROOT, env=_ENV, capture_output=True, timeout=120)
    return r.stdout.decode("utf-8", "replace"), r.returncode


def _gate_at(rev):
    src, rc = _run("show", "%s:tools/gates/prereg_amendment_order.py" % rev)
    if rc != 0:
        raise SystemExit("no tools/gates/prereg_amendment_order.py at %s" % rev)
    mod = types.ModuleType("pra_at_%s" % rev)
    mod.__file__ = os.path.join(_ROOT, "tools", "gates", "prereg_amendment_order.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _name_status(a, b):
    out, _ = _run("diff", "--name-status", a, b)
    d = {}
    for ln in out.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2:
            d[parts[-1]] = parts[0]
    return d


def replay_pra(shas, base_mod):
    rows, n_merges, head_only = [], 0, []
    for sha in shas:
        parents = _run("rev-parse", sha + "^@")[0].split()
        if not parents:
            continue
        if len(parents) > 1:
            n_merges += 1
            if pra._evaluate(_ROOT, _ENV, parents[:1], target=sha):
                head_only.append(sha[:10])
        pre = [pra._changes(_ROOT, _ENV, p, sha, pra._PREREG_SPECS, True) for p in parents]
        modified = [p for p in sorted(set(pre[0]).intersection(*pre[1:]))
                    if pra._PREREG_RE.match(p) and pre[0][p][0] in ("M", "R") and all(c[p][0] != "D" for c in pre)]
        if not modified:
            continue
        raw = [pra._changes(_ROOT, _ENV, p, sha, pra._RAW_SPECS, False) for p in parents]
        raw_written = sorted(p for p in set(raw[0]).intersection(*raw[1:])
                             if pra._RAW_RE.match(p) and all(c[p][0] != "D" for c in raw))
        if not raw_written:
            continue
        verdict = pra._evaluate(_ROOT, _ENV, parents, target=sha)
        row = {"sha": sha[:10], "merge": len(parents) > 1, "subject": _run("log", "-1", "--format=%s", sha)[0].strip()[:100],
               "n_data": sum(1 for p in raw_written if pra._is_data_artifact(p)),
               "block": bool(verdict), "problem": (verdict or [""])[0][:200]}
        if base_mod is not None:
            row["baseline_block"] = bool(base_mod._evaluate(_ROOT, _ENV, parents, target=sha))
        rows.append(row)
    return rows, n_merges, head_only


def entry_diffs(ref, base_mod):
    files = [f for f in _run("ls-tree", "-r", "--name-only", ref)[0].split("\n") if pra._PREREG_RE.match(f)]
    diffs = []
    for f in files:
        t = _run("show", "%s:%s" % (ref, f))[0]
        en = [(e["key"], e["line"]) for e in pra._Doc(t).entries]
        eo = [(e["key"], e["line"]) for e in base_mod._Doc(t).entries]
        if en != eo:
            diffs.append({"file": f, "only_baseline": [x for x in eo if x not in en],
                          "only_current": [x for x in en if x not in eo]})
    return len(files), diffs


def replay_pr(shas):
    nonmerge, clean_merge, conflicted_merge = [], [], []
    for sha in shas:
        parents = _run("rev-parse", sha + "^@")[0].split()
        if not parents:
            continue
        d = _name_status(parents[0], sha)
        added = [p for p, st in d.items() if st.startswith("A")]
        if not any(pr._PREREG.match(p) for p in added) or not any(pr._RAW.match(p) for p in d):
            continue

        def read(p, sha=sha):
            return _run("show", "%s:%s" % (sha, p))[0]

        if len(parents) == 1:
            if pr._problems(added, list(d), read):
                nonmerge.append(sha[:10])
            continue
        if pr._problems(added, list(d), read):
            clean_merge.append(sha[:10])
        theirs_same = []
        for p in added:
            a, rc = _run("rev-parse", "-q", "--verify", "%s:%s" % (parents[1], p))
            if rc == 0 and a.strip() == _run("rev-parse", "-q", "--verify", "%s:%s" % (sha, p))[0].strip():
                theirs_same.append(p)
        if pr._problems([p for p in added if p not in theirs_same], list(d), read):
            conflicted_merge.append(sha[:10])
    return nonmerge, clean_merge, conflicted_merge


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--baseline", default=None, help="a revision whose PRA gate is replayed for verdict differences")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    tip = _run("rev-parse", a.ref)[0].strip()
    shas = _run("rev-list", "-n", str(a.n), tip)[0].split()
    base_mod = _gate_at(a.baseline) if a.baseline else None
    rows, n_merges, head_only = replay_pra(shas, base_mod)
    out = {"ref": a.ref, "tip": tip, "gate_rev": _run("rev-parse", "HEAD")[0].strip(), "n_commits": len(shas),
           "n_merges": n_merges,
           "pra": {"n_candidates": len(rows), "n_block": sum(r["block"] for r in rows),
                   "blocks": [r["sha"] for r in rows if r["block"]],
                   "merges_blocked": [r["sha"] for r in rows if r["merge"] and r["block"]],
                   "merges_blocked_head_only": head_only, "rows": rows}}
    if base_mod is not None:
        n_files, ediffs = entry_diffs(tip, base_mod)
        out["pra"].update({"baseline": a.baseline, "baseline_n_block": sum(r["baseline_block"] for r in rows),
                           "verdict_diffs": [r["sha"] for r in rows if r["block"] != r["baseline_block"]],
                           "n_prereg_files": n_files, "entry_diffs": ediffs})
    nonmerge, clean_merge, conflicted = replay_pr(shas)
    out["pr"] = {"nonmerge_blocks": nonmerge, "merges_blocked_during_a_clean_merge_before_2026_09_25": clean_merge,
                 "merges_blocked_with_the_merge_head_exemption": conflicted}
    text = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    p = out["pra"]
    print("PRA: %d commits (%d merges) of %s @ %s -> %d candidates, %d blocks, %d merges blocked (%d head-only)"
          % (len(shas), n_merges, a.ref, tip[:10], p["n_candidates"], p["n_block"], len(p["merges_blocked"]),
             len(head_only)))
    if base_mod is not None:
        print("PRA vs %s: %d baseline blocks, %d verdict diffs, %d/%d prereg files with changed entries"
              % (a.baseline, p["baseline_n_block"], len(p["verdict_diffs"]), len(p["entry_diffs"]), p["n_prereg_files"]))
    print("PR: %d non-merge blocks; merges blocked on a clean merge before the fix %d, with MERGE_HEAD exemption %d"
          % (len(nonmerge), len(clean_merge), len(conflicted)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
