#!/usr/bin/env python3
"""b2b_score.py -- the committed, re-derivable scorer for Battery B2b (tag b2b0924-base).

Governing text: research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md, Amendment 1
(A1.2 validity, A1.3 cause classes, A1.4 re-runs, A1.5 verdict, A1.6 corpus, A1.9 comparison with B2a) and
Amendment 2 (the three torn cells).

WHY A SEPARATE SCORER. `tools/lb_shard.py aggregate --pin` checks A1.2 rules 1-3 and computes the fraction, but
the prereg warns (the "Validity of a cell" section) that the aggregate lists an `arm-build-failed` cell as present
and not load-bearing, and it does not check rule 4 (corpus host), rule 5 (verdict class), the re-run log, the
A1.5 final states or the A1.9 comparison. This file does all of them itself: it does NOT import lb_shard, so its
numbers are an independent derivation, and `--registered-aggregate` cross-checks them against lb_shard's output
number by number (a mismatch fails a precondition).

Inputs (all read-only): the job file (the cell list), the corpus-hash table, the re-run log, dispatch.log (hosts),
the b2b0924-base shard tree, the attempt-1 tree, and the b2a0924 shard tree (A1.9). Output: one JSON with the
verdict, a `preconditions` block (gates/verdict_preconditions), per-cell states, a per-cell manifest (every file's
sha256, the verdict, the host), the reported quantities and the B2a comparison.

Usage:
  python tools/b2b_score.py --root /home/dant123/Projects/sim [--coord-root <this checkout>] \
      --registered-aggregate research/findings/raw/_load_bearing/_shards/b2b0924-base/aggregate.json \
      --out research/findings/raw/_load_bearing/_b2b0924_score/b2b_score.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys

F = "a308f1e09babcc9ed096c3c8046d00040391368c"
M1 = "9db7613296c3d02a161b36fb15da3983188b7902"
TAG = "b2b0924-base"
B2A_TAG = "b2a0924"
SEEDS = [42, 43, 44, 100, 101, 102]
COVERABLE = ("neural-lesion", "whether-disable")
MEASURED = ("regressed", "pass", "not-exercised")
EXERCISED = ("regressed", "pass")
# A1.3 class C verdicts (code at F failed on the row); `unmapped-*` is a prefix
C_VERDICTS = ("arm-build-failed", "lesion-knob-missing", "lesion-not-effective", "artifact-missing",
              "artifact-unreadable", "seed-missing-in-artifact", "noisy", "noisy-null-control")
SHARDS = "research/findings/raw/_load_bearing/_shards"
ATTEMPT1 = "research/findings/raw/_load_bearing/_b2b0924_attempt1"
JOBS = "research/coordination/b2b0924_base_jobs.txt"
CORPUS = "research/coordination/b2b0924_corpus_sha256.tsv"
RERUNS = "research/coordination/b2b0924_reruns.tsv"
DISPATCH = "research/queue/dispatch.log"
# Amendment 2: for these three cells the true first attempt is a dispatcher-popped fragment (class E, evidence in
# research/findings/raw/_dispatcher_fragment_audit/node_evidence/); what sits in _b2b0924_attempt1/ is the voided
# SAME-HOST F run, reported as such and never scored.
AMENDMENT2_CELLS = {"s43/d5-consolidate": "pool2", "s42/causal-whatif": "pool1",
                    "s42/affect-appraisal-interoceptive": "pool2"}


def _j(path):
    with open(path) as fh:
        return json.load(fh)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_cells(root):  # root = the coordination root
    """(faculty, seed, out_dir) for every line of the job file, in file order."""
    cells = []
    for line in open(os.path.join(root, JOBS)):
        m = re.search(r"--only (\S+) --seed (\d+) .*--out (\S+)/lb\.json", line)
        if m:
            cells.append((m.group(1), int(m.group(2)), m.group(3)))
    return cells


def read_corpus(root):
    """node name -> hostname, hostname -> sha256, and the primary checkout's sha256 (A1.6)."""
    node_host, host_sha, primary = {}, {}, None
    for row in open(os.path.join(root, CORPUS)).read().splitlines()[1:]:
        p = row.split("\t")
        if len(p) < 4:
            continue
        if p[0] == "primary-checkout":
            primary = p[3]
        else:
            node_host[p[0]] = p[1]
            host_sha[p[1]] = p[3]
    return node_host, host_sha, primary


def read_reruns(root):
    rows = {}
    path = os.path.join(root, RERUNS)
    if not os.path.exists(path):
        return rows
    lines = open(path).read().splitlines()
    hdr = lines[0].split("\t")
    for ln in lines[1:]:
        if not ln.strip():
            continue
        rows[ln.split("\t")[0]] = dict(zip(hdr, ln.split("\t")))
    return rows


def read_dispatch_hosts(root, tag):
    """cell key 's<seed>/<faculty>' -> list of dispatch node names, in log order, for one tag."""
    out = {}
    path = os.path.join(root, DISPATCH)
    if not os.path.exists(path):
        return out
    pat = re.compile(r"^\[pool-dispatch\] \S+ (\S+) <- .*--out research/findings/raw/_load_bearing/_shards/%s/"
                     r"s(\d+)/([^/ ]+)/lb\.json" % re.escape(tag))
    for ln in open(path, errors="replace"):
        m = pat.search(ln)
        if m:
            out.setdefault("s%s/%s" % (m.group(2), m.group(3)), []).append(m.group(1))
    return out


def sidecar_fails(pj, pin, is_lb):
    f = []
    if pj.get("git_sha") != pin:
        f.append("git_sha=%s" % pj.get("git_sha"))
    if pj.get("source_kind") != "git_archive":
        f.append("source_kind=%s" % pj.get("source_kind"))
    if pj.get("source_manifest_verified_at_start") is not True:
        f.append("manifest_start=%s" % pj.get("source_manifest_verified_at_start"))
    if pj.get("source_manifest_verified_at_exit") is not True:
        f.append("manifest_exit=%s" % pj.get("source_manifest_verified_at_exit"))
    env = pj.get("env") or {}
    brain = sorted(k for k in env if k.startswith("BRAIN_"))
    if is_lb:
        if env.get("SIM_BACKEND") != "numpy":
            f.append("SIM_BACKEND=%s" % env.get("SIM_BACKEND"))
        if brain:
            f.append("BRAIN keys %s" % brain)
    elif [k for k in brain if k != "BRAIN_CHAT_SEED"]:
        f.append("arm BRAIN keys %s" % brain)
    return f


ARM_RE = re.compile(r"^(intact_a_|intact_b_|lesion_).*\.json(\.rep\d+)?$")


def judge_dir(d, fac, pin, host_sha=None, primary_sha=None):
    """Apply A1.2 rules 1-5 to one shard directory. Returns a dict with class (None if DEFINED)."""
    r = {"dir": d, "faculty": fac, "exists": os.path.isdir(d), "fails": [], "class": None, "kind": None,
         "verdict": None, "load_bearing": None, "host": None, "files": {}, "multiref_error_turns": 0,
         "n_multiref_turns": 0, "organ_errors": []}
    if not r["exists"]:
        r["fails"].append("E: shard directory absent")
        r["class"] = "E"
        return r
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            r["files"][fn] = _sha(p)
    lb = os.path.join(d, "lb.json")
    if not os.path.exists(lb):
        r["fails"].append("E: no lb.json")
    else:
        rep = _j(lb)
        rows = [p for p in rep.get("per_faculty", []) if p.get("faculty") == fac]
        if len(rows) != 1:
            r["fails"].append("E: lb.json holds %d rows for %s" % (len(rows), fac))
        else:
            p = rows[0]
            r.update(kind=p.get("kind"), verdict=p.get("verdict"), load_bearing=p.get("load_bearing"),
                     null_control_clean=p.get("null_control_clean"), unreliable=bool(rep.get("UNRELIABLE")))
        if not os.path.exists(lb + ".prov.json"):
            r["fails"].append("E rule1: lb.json has no sidecar")
        else:
            lp = _j(lb + ".prov.json")
            r["host"] = lp.get("host")
            r["fails"] += ["E rule1/3 lb.json: " + x for x in sidecar_fails(lp, pin, True)]
            if host_sha is not None:
                hs = host_sha.get(lp.get("host"))
                if hs is None:
                    r["fails"].append("E rule4: host %s not in the corpus table" % lp.get("host"))
                elif hs != primary_sha:
                    r["fails"].append("E rule4: host %s corpus hash %s != primary" % (lp.get("host"), hs[:12]))
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".prov.json") or not ARM_RE.match(fn):
            continue
        sc = os.path.join(d, fn + ".prov.json")
        if not os.path.exists(sc):
            r["fails"].append("E rule2: arm file %s has no sidecar" % fn)
            continue
        pj = _j(sc)
        r["fails"] += ["E rule2/3 %s: %s" % (fn, x) for x in sidecar_fails(pj, pin, False)]
        if r["host"] and pj.get("host") != r["host"]:
            r["fails"].append("E rule2: %s host %s != lb.json host %s" % (fn, pj.get("host"), r["host"]))
        try:
            arm = _j(os.path.join(d, fn))
        except Exception:
            continue
        if isinstance(arm, dict):
            for t in arm.values():
                if isinstance(t, dict) and isinstance(t.get("multiref"), dict):
                    r["n_multiref_turns"] += 1
                    if "error" in t["multiref"]:
                        r["multiref_error_turns"] += 1
                # REPORTED, not a validity rule (A1.2 rule 5 reads only lb.json): an organ that raised inside an
                # arm and was caught into `<organ>.error`. Found 2026-09-29: comprehension-monitor's lesion arm
                # raises a ValueError on every seed, so its "load-bearing" diff is the missing field, not a lesion
                # effect on the reply.
                if isinstance(t, dict):
                    for k, v in t.items():
                        if k != "multiref" and isinstance(v, dict) and "error" in v:
                            r["organ_errors"].append({"arm": fn, "organ": k, "error": str(v["error"])[:160]})
    if any(x.startswith("E") for x in r["fails"]):
        r["class"] = "E"
    elif r["kind"] in COVERABLE:
        v = r["verdict"]
        if r.get("unreliable"):
            r["fails"].append("C: report UNRELIABLE")
        if r.get("null_control_clean") is False:
            r["fails"].append("C: null_control_clean False")
        if v not in MEASURED:
            r["fails"].append("verdict %s" % v)
        if r["fails"]:
            is_c = r.get("unreliable") or r.get("null_control_clean") is False or (
                v in C_VERDICTS or (isinstance(v, str) and v.startswith("unmapped-")))
            r["class"] = "C" if is_c else "I"
    return r


def final_state(canon, attempt1, first_host, rerun_host):
    """A1.5 final state of one coverable cell from its canonical (last) attempt and its moved-aside attempt 1.

    Reading of A1.5 used here (its three "otherwise" clauses overlap for a C-then-E pair, so they are applied by
    the class of the LAST attempt): C -> FAIL when the first attempt is also C on a different host, else
    C-UNCONFIRMED; I -> UNDECIDED; E -> MISSING."""
    if canon["class"] is None:
        return "DEFINED"
    if canon["class"] == "C":
        if (attempt1 is not None and attempt1["class"] == "C" and first_host and rerun_host
                and first_host != rerun_host):
            return "FAIL"
        return "C-UNCONFIRMED"
    if canon["class"] == "I":
        return "UNDECIDED"
    return "MISSING"


def score(root, registered=None, b2a=True, coord_root=None):
    """root: the checkout holding the shard trees and dispatch.log (the primary checkout, where pool_sync lands
    results). coord_root: the checkout holding the committed coordination files (job file, corpus table, re-run
    log); defaults to root."""
    coord_root = coord_root or root
    cells = read_cells(coord_root)
    node_host, host_sha, primary_sha = read_corpus(coord_root)
    reruns = read_reruns(coord_root)
    disp = read_dispatch_hosts(root, TAG)
    out_cells, kind_of = {}, {}
    for fac, seed, out_dir in cells:
        key = "s%d/%s" % (seed, fac)
        canon = judge_dir(os.path.join(root, out_dir), fac, F, host_sha, primary_sha)
        a1_dir = os.path.join(root, ATTEMPT1, TAG, "s%d" % seed, fac)
        attempt1 = judge_dir(a1_dir, fac, F, host_sha, primary_sha) if os.path.isdir(a1_dir) else None
        rr = reruns.get("%s/%s" % (TAG, key))
        first_host = rerun_host = None
        if rr:
            first_host = rr.get("first_host", "").split(" ")[0]
            rerun_host = rr.get("rerun_host", "").split(" ")[0]
        if canon["kind"]:
            kind_of[fac] = canon["kind"]
        dnodes = disp.get(key, [])
        out_cells[key] = {"faculty": fac, "seed": seed, "canonical": canon, "attempt1": attempt1,
                          "rerun_log": rr, "first_host": first_host, "rerun_host": rerun_host,
                          "dispatch_nodes": dnodes}
    for key, c in out_cells.items():
        c["kind"] = kind_of.get(c["faculty"])
        c["coverable"] = c["kind"] in COVERABLE
        c["state"] = final_state(c["canonical"], c["attempt1"], c["first_host"], c["rerun_host"]) \
            if c["coverable"] else "reported (%s)" % c["canonical"]["verdict"]
        # A1.4 host checks: the canonical sidecar host must be the node the log says ran it
        exp_node = c["rerun_host"] or (c["dispatch_nodes"][-1] if c["dispatch_nodes"] else None)
        exp_host = node_host.get(exp_node) if exp_node else None
        c["host_check"] = {"expected_node": exp_node, "expected_host": exp_host,
                           "sidecar_host": c["canonical"]["host"],
                           # a cell with no lb.json sidecar has no host to check; it is already not DEFINED
                           "ok": (exp_host is None) or (c["canonical"]["host"] is None)
                           or (exp_host == c["canonical"]["host"])}
        if c["rerun_log"]:
            c["rerun_on_different_host"] = bool(c["first_host"] and c["rerun_host"] and
                                                c["first_host"] != c["rerun_host"])
    cov = {k: c for k, c in out_cells.items() if c["coverable"]}
    facs = sorted({c["faculty"] for c in cov.values()})
    per_seed = {}
    for s in SEEDS:
        dc = [c for c in cov.values() if c["seed"] == s and c["state"] == "DEFINED"]
        ex = [c for c in dc if c["canonical"]["verdict"] in EXERCISED]
        lbc = [c for c in ex if c["canonical"]["load_bearing"] is True]
        per_seed[str(s)] = {"n_defined": len(dc), "n_exercised": len(ex), "n_load_bearing": len(lbc),
                            "load_bearing_fraction": (len(lbc) / len(ex)) if ex else None}
    per_fac = {}
    for f in facs:
        vs = [cov["s%d/%s" % (s, f)] for s in SEEDS]
        nd = [v for v in vs if v["state"] != "DEFINED"]
        n_lb = sum(1 for v in vs if v["state"] == "DEFINED" and v["canonical"]["load_bearing"] is True)
        per_fac[f] = {"kind": kind_of.get(f), "n_load_bearing": n_lb if not nd else [n_lb, n_lb + len(nd)],
                      "verdicts": {str(v["seed"]): v["canonical"]["verdict"] for v in vs},
                      "states": {str(v["seed"]): v["state"] for v in vs}}
    lo = lambda v: v if isinstance(v, int) else v[0]   # noqa: E731
    hi = lambda v: v if isinstance(v, int) else v[1]   # noqa: E731
    robust = sorted(f for f, v in per_fac.items() if lo(v["n_load_bearing"]) == 6)
    union = sorted(f for f, v in per_fac.items() if lo(v["n_load_bearing"]) > 0)
    open_membership = sorted(f for f, v in per_fac.items() if isinstance(v["n_load_bearing"], list)
                             and (hi(v["n_load_bearing"]) == 6 or lo(v["n_load_bearing"]) == 0))
    fr = [v["load_bearing_fraction"] for v in per_seed.values() if v["load_bearing_fraction"] is not None]
    states = {}
    for c in cov.values():
        states.setdefault(c["state"], []).append("s%d/%s" % (c["seed"], c["faculty"]))
    fail_cells = sorted(states.get("FAIL", []))
    n_defined = len(states.get("DEFINED", []))
    if fail_cells:
        verdict = "FAIL"
    elif n_defined == len(cov) == 216:
        verdict = "COMPLETE"
    else:
        verdict = "INCOMPLETE"
    res = {
        "tag": TAG, "pin": F, "seeds": SEEDS, "rule": "B2b prereg Amendment 1 (A1.2-A1.5, A1.9) + Amendment 2",
        # device (gates/device_and_cost): A1.2 rule 1 admits a cell only if its lb.json sidecar records
        # env.SIM_BACKEND=numpy, so every DEFINED cell ran on numpy (CPU), one BLAS/OMP thread per the job line
        "backend": "numpy", "device": "cpu",
        "backend_source": "env.SIM_BACKEND of every DEFINED cell's lb.json sidecar (A1.2 rule 1); %d of %d coverable "
                          "cells DEFINED" % (n_defined, len(cov)),
        "verdict": verdict,
        "counts": {"n_job_lines": len(cells), "n_coverable_faculties": len(facs), "n_coverable_cells": len(cov),
                   "n_defined": n_defined, "states": {k: len(v) for k, v in states.items()}},
        "not_defined": {k: {"state": c["state"], "fails": c["canonical"]["fails"]}
                        for k, c in cov.items() if c["state"] != "DEFINED"},
        "fail_cells": fail_cells,
        "per_seed": per_seed,
        "mean_fraction": statistics.fmean(fr) if fr else None,
        "sd_fraction_pop": statistics.pstdev(fr) if fr else None,
        "robust_core": robust, "robust_core_n": len(robust), "union": union, "union_n": len(union),
        "membership_open": open_membership,
        "per_faculty": per_fac,
        "r3_learned_referent": {str(c["seed"]): c["canonical"]["verdict"] for c in out_cells.values()
                                if c["faculty"] == "learned-referent"},
        "reported_rows": {k: c["canonical"]["verdict"] for k, c in out_cells.items() if not c["coverable"]},
        "multiref_error_turns": {k: c["canonical"]["multiref_error_turns"] for k, c in out_cells.items()
                                 if c["canonical"]["multiref_error_turns"]},
        "multiref_error_turns_total": sum(c["canonical"]["multiref_error_turns"] for c in out_cells.values()),
        "multiref_turns_total": sum(c["canonical"]["n_multiref_turns"] for c in out_cells.values()),
        "organ_error_cells": {k: c["canonical"]["organ_errors"] for k, c in out_cells.items()
                              if c["canonical"]["organ_errors"]},
        "load_bearing_with_lesion_arm_organ_error": sorted(
            k for k, c in out_cells.items() if c["canonical"]["load_bearing"] is True
            and any(e["arm"].startswith("lesion_") for e in c["canonical"]["organ_errors"])),
        "hosts": sorted({c["canonical"]["host"] for c in out_cells.values() if c["canonical"]["host"]}),
        "host_check_failures": {k: c["host_check"] for k, c in out_cells.items() if not c["host_check"]["ok"]},
        "reruns": {k: {"first_host": c["first_host"], "rerun_host": c["rerun_host"],
                       "different_host": c.get("rerun_on_different_host"), "state": c["state"],
                       "attempt1_kept_never_scored": None if c["attempt1"] is None else {
                           "what": ("voided same-host F run (Amendment 2); the true first attempt was a dispatcher "
                                    "fragment, class E" if k in AMENDMENT2_CELLS else "first attempt"),
                           "dir": os.path.relpath(c["attempt1"]["dir"], root),
                           "class_as_found": c["attempt1"]["class"], "verdict": c["attempt1"]["verdict"],
                           "fails": c["attempt1"]["fails"], "files": sorted(c["attempt1"]["files"])}}
                   for k, c in out_cells.items() if c["rerun_log"] or c["attempt1"] is not None},
        "manifest": {k: {"dir": os.path.relpath(c["canonical"]["dir"], root), "state": c["state"],
                         "verdict": c["canonical"]["verdict"], "load_bearing": c["canonical"]["load_bearing"],
                         "host": c["canonical"]["host"], "sha256": c["canonical"]["files"]}
                     for k, c in out_cells.items()},
    }
    if b2a:
        res["a1_9_b2a_comparison"] = compare_b2a(root, res, out_cells)
    pre = [
        {"name": "job file lists 258 cells (43 rows x 6 seeds)", "ok": len(cells) == 258, "value": len(cells)},
        {"name": "36 coverable faculties, 216 coverable cells", "ok": len(facs) == 36 and len(cov) == 216,
         "value": [len(facs), len(cov)]},
        {"name": "corpus table read: primary hash and 4 node hosts", "ok": bool(primary_sha) and len(host_sha) >= 4,
         "value": {"primary": primary_sha, "hosts": sorted(host_sha)}},
        {"name": "every canonical sidecar host matches its dispatch or re-run node (A1.4)",
         "ok": not res["host_check_failures"], "value": res["host_check_failures"]},
        {"name": "every logged re-run ran on a host different from its first attempt (A1.4 b)",
         "ok": all(v["different_host"] for v in res["reruns"].values() if v["different_host"] is not None),
         "value": {k: [v["first_host"], v["rerun_host"]] for k, v in res["reruns"].items()}},
    ]
    if registered is not None:
        mism = cross_check(registered, res)
        pre.append({"name": "matches tools/lb_shard.py aggregate --pin F number by number", "ok": not mism,
                    "value": mism})
    res["preconditions"] = pre
    if not all(p["ok"] is True for p in pre):
        res["verdict"] = "UNDEFINED -- a scorer precondition failed (%s); cell-level verdict would read %s" % (
            ", ".join(p["name"] for p in pre if p["ok"] is not True), verdict)
    return res


def cross_check(agg, res):
    mism = []
    prov = agg.get("provenance") or {}
    if prov.get("pin") != F or prov.get("n_invalid") != 0:
        mism.append("registered aggregate provenance: pin %s, n_invalid %s" % (prov.get("pin"), prov.get("n_invalid")))
    for k, rk in (("robust_core_n", "robust_core_n"), ("union_n", "union_n")):
        if agg.get(k) != res.get(rk):
            mism.append("%s %s vs %s" % (k, agg.get(k), res.get(rk)))
    if sorted(agg.get("robust_core") or []) != res["robust_core"]:
        mism.append("robust_core members differ")
    for a, b in ((agg.get("mean_fraction"), res["mean_fraction"]), (agg.get("sd_fraction"), res["sd_fraction_pop"])):
        if a is None or b is None or abs(a - b) > 1e-12:
            mism.append("fraction stat %r vs %r" % (a, b))
    for s in SEEDS:
        a = (agg.get("per_seed") or {}).get(str(s)) or (agg.get("per_seed") or {}).get(s) or {}
        b = res["per_seed"][str(s)]
        if (a.get("n_exercised"), a.get("n_load_bearing")) != (b["n_exercised"], b["n_load_bearing"]):
            mism.append("seed %s exercised/load-bearing %s/%s vs %s/%s" % (
                s, a.get("n_exercised"), a.get("n_load_bearing"), b["n_exercised"], b["n_load_bearing"]))
    for f, v in res["per_faculty"].items():
        a = (agg.get("per_faculty") or {}).get(f, {}).get("n_load_bearing")
        if a != v["n_load_bearing"]:
            mism.append("%s n_load_bearing %s vs %s" % (f, a, v["n_load_bearing"]))
    if agg.get("incomplete_faculties"):
        mism.append("registered aggregate lists incomplete faculties %s" % agg.get("incomplete_faculties"))
    return mism


def compare_b2a(root, res, out_cells):
    """A1.9: B2b-base against B2a (M1). A B2a cell enters only if its sidecars meet A1.2 rules 1-3 with M1."""
    b2a_hosts = read_dispatch_hosts(root, B2A_TAG)
    b2a = {}
    for d in glob.glob(os.path.join(root, SHARDS, B2A_TAG, "s*", "*")):
        if not os.path.isdir(d):
            continue
        seed, fac = int(os.path.basename(os.path.dirname(d))[1:]), os.path.basename(d)
        r = judge_dir(d, fac, M1)   # rules 1-3 only (B2a has no corpus table and no host field)
        rules13 = [x for x in r["fails"] if x.startswith("E")]
        b2a["s%d/%s" % (seed, fac)] = {"faculty": fac, "seed": seed, "kind": r["kind"], "verdict": r["verdict"],
                                       "load_bearing": r["load_bearing"], "valid_rules_1_3": not rules13,
                                       "fails": rules13, "class": r["class"],
                                       "host_from_dispatch_log": (b2a_hosts.get("s%d/%s" % (seed, fac)) or [None])[-1]}
    b2a_cov_facs = sorted({v["faculty"] for v in b2a.values() if v["kind"] in COVERABLE})
    b2b_facs = sorted(res["per_faculty"])
    common = [f for f in b2b_facs if f in b2a_cov_facs]
    added = [f for f in b2b_facs if f not in b2a_cov_facs]
    per_fac, diffs = {}, []
    for f in common:
        a_cells = [b2a.get("s%d/%s" % (s, f)) for s in SEEDS]
        a_valid = [c for c in a_cells if c and c["valid_rules_1_3"] and c["class"] is None]
        a_n = sum(1 for c in a_valid if c["load_bearing"] is True)
        per_fac[f] = {"b2a_n_load_bearing": a_n if len(a_valid) == 6 else [a_n, a_n + 6 - len(a_valid)],
                      "b2b_n_load_bearing": res["per_faculty"][f]["n_load_bearing"]}
        for s in SEEDS:
            a = b2a.get("s%d/%s" % (s, f))
            b = out_cells["s%d/%s" % (s, f)]
            both_defined = bool(a and a["valid_rules_1_3"] and a["class"] is None and b["state"] == "DEFINED")
            if a is None or a["load_bearing"] != b["canonical"]["load_bearing"] or a["verdict"] != b["canonical"]["verdict"]:
                diffs.append({"cell": "s%d/%s" % (s, f), "b2a": None if a is None else [a["verdict"], a["load_bearing"]],
                              "b2b": [b["canonical"]["verdict"], b["canonical"]["load_bearing"]],
                              "b2a_host_dispatch_log": None if a is None else a["host_from_dispatch_log"],
                              "b2b_host_sidecar": b["canonical"]["host"],
                              "attributed_to_code_M1_to_F": both_defined})
    b2a_excluded = {k: v["fails"] for k, v in b2a.items() if v["kind"] in COVERABLE and not v["valid_rules_1_3"]}
    lo = lambda v: v if isinstance(v, int) else v[0]   # noqa: E731
    return {
        "b2a_pin": M1,
        "b2a_coverable_faculties": len(b2a_cov_facs),
        "b2a_coverable_cells_valid_rules_1_3": sum(1 for v in b2a.values() if v["kind"] in COVERABLE and v["valid_rules_1_3"]),
        "b2a_cells_excluded": b2a_excluded,
        "i_same_rows": {"n_rows": len(common), "per_faculty": per_fac,
                        "faculties_differing": sorted(f for f, v in per_fac.items()
                                                      if v["b2a_n_load_bearing"] != v["b2b_n_load_bearing"])},
        "ii_all_rows": {"n_rows": len(b2b_facs), "added_rows": {f: res["per_faculty"][f]["n_load_bearing"] for f in added}},
        "per_cell_differences": diffs,
        "b2b_over_same_rows": {
            "robust_core_n": sum(1 for f in common if lo(res["per_faculty"][f]["n_load_bearing"]) == 6),
            "union_n": sum(1 for f in common if lo(res["per_faculty"][f]["n_load_bearing"]) > 0)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="checkout holding the shard trees and dispatch.log (where pool_sync lands results)")
    ap.add_argument("--coord-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="checkout holding the committed job file, corpus table and re-run log (default: this "
                         "file's own checkout)")
    ap.add_argument("--registered-aggregate", default=None,
                    help="lb_shard.py aggregate output to cross-check (path relative to --root or absolute)")
    ap.add_argument("--no-b2a", action="store_true", help="skip the A1.9 comparison")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    reg = None
    if a.registered_aggregate:
        p = a.registered_aggregate if os.path.isabs(a.registered_aggregate) else os.path.join(a.root, a.registered_aggregate)
        reg = _j(p)
    res = score(a.root, reg, b2a=not a.no_b2a, coord_root=a.coord_root)
    summary = {k: res[k] for k in ("verdict", "counts", "not_defined", "fail_cells", "robust_core_n", "union_n",
                                   "mean_fraction", "sd_fraction_pop", "membership_open")}
    summary["preconditions_failed"] = [p["name"] for p in res["preconditions"] if p["ok"] is not True]
    print(json.dumps(summary, indent=1, default=str))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str, sort_keys=False)
        print("wrote %s" % a.out)
    return 0 if not summary["preconditions_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
