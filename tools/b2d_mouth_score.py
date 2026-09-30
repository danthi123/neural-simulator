#!/usr/bin/env python3
"""b2d_mouth_score.py -- the registered scorer for B2d's mouth-channel arm (tag b2d0929-mouth).

Governing text: research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md, "Mouth-channel arm",
and its "Amendment 1" section. The mouth arm runs the three
affect rows (affect-coloring, affect-appraisal-interoceptive, affect-drives-response) through the same probe turns as
the stub arm, with LB_ARM_RENDERER=production (the reply rendered by the production mouth, QwenRenderer, on cupy). It
is reported separately and never averaged into B2d's S0/S1/S2.

Read-only, like tools/b2d_score.py: it reads arm files on disk, never imports the harness, never builds a brain.

Per cell (row, seed), in this order; the first rule that fails decides the state:
  E  (validity, re-run once per A1.4): lb.json and its sidecar exist; every sidecar (lb.json and the four arm files)
     reads git_sha = --pin in full, source_kind git_archive, both manifest checks true, env.SIM_BACKEND = cupy and
     env.LB_ARM_RENDERER = production; lb.json's env holds no BRAIN_* key and an arm's none but BRAIN_CHAT_SEED;
     every arm sidecar names the lb.json host; the four arm files (intact_a, intact_b, lesion, lesion.rep0) exist
     (rule 6); with --corpus, the host's corpus hash is recorded and equals the tree's own.
  C  (rule 7): an `error` key in the probe-turn parent dict of any compared field, in any arm.
  M  (not the mouth, re-run once): on the probe turn of any of the four arms, the response's `renderer` is not
     QWEN_RENDERER_NAME, or its `_arm_renderer.mode` is not "production" (e.g. CUDA was unavailable and the handler
     resolved the GPU-free stub, or the job ran without the mode).
  MOUTH-SILENT (final, not re-run): the Qwen generation entry points ran 0 times on the probe turn in all four arms.
     The reply on this turn never came from the mouth (an abstention template, a host clause), so the mouth arm
     cannot measure this row on this turn; the row is named, never counted as 0 and never as a pass.
  S2m: tools/b2d_score.py's S2 definition verbatim (answer, abstained, recalled_svo on the probe turn; null floor
     intact_a = intact_b, lesion floor lesion = lesion.rep0; UNDEFINED:null-noise / UNDEFINED:lesion-noise never 0).

Per row: n_reply_load_bearing over the DEFINED cells, n_mouth_silent, n_undefined; the row reading
  MOUTH-LOAD-BEARING (all six DEFINED and reply-load-bearing), MOUTH-PARTIAL (1-5 of the DEFINED),
  MOUTH-NOT-LOAD-BEARING (all six DEFINED, none), MOUTH-SILENT (all six mouth-silent),
  MIXED (anything else once every cell is final), PENDING (a cell is not final).
Battery: COMPLETE when every cell is final (DEFINED, MOUTH-SILENT or FAIL = class C/M on two hosts or two attempts),
INCOMPLETE otherwise. With --stub-tag, the stub arm's S2 for the same (row, seed) is printed beside it (read with
tools/b2d_score.py's own judge_dir/score_s2), so the only variable between the two columns is the renderer (and the
backend the production mouth needs).

Exit code: 0 when the scorer ran and no precondition failed, 1 when a precondition failed (fields pin != --pin, or a
--code-root blob mismatch), 2 on bad arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2d_score as bd  # noqa: E402

TAG_DEFAULT = "b2d0929-mouth"
ROWS = ("affect-coloring", "affect-appraisal-interoceptive", "affect-drives-response")
QWEN_RENDERER_NAME = "off-bridge Qwen-0.5B (spiking forward)"
MODE = "production"
ARM_KEYS = ("intact_a", "intact_b", "lesion", "lesion.rep0")


def mouth_sidecar_fails(pj, pin, is_lb):
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
    if env.get("SIM_BACKEND") != "cupy":
        f.append("SIM_BACKEND=%s" % env.get("SIM_BACKEND"))
    if env.get("LB_ARM_RENDERER") != MODE:
        f.append("LB_ARM_RENDERER=%s" % env.get("LB_ARM_RENDERER"))
    brain = sorted(k for k in env if k.startswith("BRAIN_"))
    if is_lb and brain:
        f.append("BRAIN keys %s" % brain)
    elif not is_lb and [k for k in brain if k != "BRAIN_CHAT_SEED"]:
        f.append("arm BRAIN keys %s" % brain)
    return f


def judge_mouth_cell(d, fac, pin, fields_row, host_sha=None, primary_sha=None):
    """One mouth cell. Returns a dict with `state` (DEFINED / MOUTH-SILENT / E / C / M), `fails`, the probe-turn
    renderer read-out per arm, and (when DEFINED) `s2` = (verdict, detail) from tools/b2d_score.py's score_s2."""
    r = {"dir": d, "faculty": fac, "fails": [], "state": None, "host": None, "turn": None, "arms_present": [],
         "renderer": {}, "qwen_calls": {}, "lb": None}
    turn = (fields_row or {}).get("probe_turn")
    r["turn"] = turn
    if not os.path.isdir(d):
        r["fails"].append("E: shard directory absent")
        r["state"] = "E"
        return r
    lb = os.path.join(d, "lb.json")
    if not os.path.exists(lb):
        r["fails"].append("E: no lb.json")
    else:
        rep = bd._j(lb)
        rows = [p for p in rep.get("per_faculty", []) if p.get("faculty") == fac]
        if len(rows) == 1:
            p = rows[0]
            r["lb"] = {"verdict": p.get("verdict"), "load_bearing": p.get("load_bearing"), "turn": p.get("turn"),
                       "null_control_clean": p.get("null_control_clean"), "diffs": p.get("diffs") or []}
        else:
            r["fails"].append("E: lb.json holds %d rows for %s" % (len(rows), fac))
        if not os.path.exists(lb + ".prov.json"):
            r["fails"].append("E rule1: lb.json has no sidecar")
        else:
            lp = bd._j(lb + ".prov.json")
            r["host"] = lp.get("host")
            r["fails"] += ["E rule1/3 lb.json: " + x for x in mouth_sidecar_fails(lp, pin, True)]
            if host_sha is not None and primary_sha is not None:
                hs = host_sha.get(lp.get("host"))
                if hs is None:
                    r["fails"].append("E rule4: host %s not in the corpus table" % lp.get("host"))
                elif hs != primary_sha:
                    r["fails"].append("E rule4: host %s corpus hash %s != tree" % (lp.get("host"), hs[:12]))
    arms = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".prov.json") or not bd.ARM_RE.match(fn):
            continue
        sc = os.path.join(d, fn + ".prov.json")
        if not os.path.exists(sc):
            r["fails"].append("E rule2: arm file %s has no sidecar" % fn)
            continue
        pj = bd._j(sc)
        r["fails"] += ["E rule2/3 %s: %s" % (fn, x) for x in mouth_sidecar_fails(pj, pin, False)]
        if r["host"] and pj.get("host") != r["host"]:
            r["fails"].append("E rule2: %s host %s != lb.json host %s" % (fn, pj.get("host"), r["host"]))
        try:
            arm = bd._j(os.path.join(d, fn))
        except Exception:
            r["fails"].append("E: arm file %s unreadable" % fn)
            continue
        kind = bd._arm_kind(fn)
        key = (kind + ".rep0") if (kind == "lesion" and fn.endswith(".rep0")) else kind
        if key and key not in arms and not fn.endswith(tuple(".rep%d" % i for i in range(1, 10))):
            arms[key] = arm
    r["arms_present"] = sorted(arms)
    missing = [k for k in ARM_KEYS if k not in arms or not isinstance(arms[k], dict) or turn not in arms[k]]
    if missing:
        r["fails"].append("E rule6: arm file or probe turn %r missing for %s" % (turn, missing))
    if any(x.startswith("E") for x in r["fails"]):
        r["state"] = "E"
        return r
    # rule 7: an organ error in the parent dict of any compared field, on the probe turn, in any arm
    scope = set(bd.field_organs(fields_row))
    for k in ARM_KEYS:
        t = arms[k][turn]
        for organ, v in t.items():
            if organ in scope and isinstance(v, dict) and "error" in v:
                r["fails"].append("C rule7: %s organ %s error %s" % (k, organ, str(v["error"])[:160]))
    if r["fails"]:
        r["state"] = "C"
        return r
    # M: the probe turn must have been rendered by the production mouth, in production mode, in all four arms
    for k in ARM_KEYS:
        t = arms[k][turn]
        ar = t.get("_arm_renderer") or {}
        r["renderer"][k] = t.get("renderer")
        r["qwen_calls"][k] = ar.get("qwen_generate_calls")
        if t.get("renderer") != QWEN_RENDERER_NAME:
            r["fails"].append("M: %s renderer %r != %r" % (k, t.get("renderer"), QWEN_RENDERER_NAME))
        if ar.get("mode") != MODE or not isinstance(ar.get("qwen_generate_calls"), int):
            r["fails"].append("M: %s _arm_renderer %r" % (k, ar))
    if r["fails"]:
        r["state"] = "M"
        return r
    if all(r["qwen_calls"][k] == 0 for k in ARM_KEYS):
        r["state"] = "MOUTH-SILENT"
        return r
    canon = {"arms": arms, "turn": turn}
    r["s2"] = bd.score_s2(canon, fields_row)
    r["state"] = "DEFINED"
    return r


def final_state(cell, attempt1):
    """A1.5 carried to the mouth arm: DEFINED / MOUTH-SILENT are final; class C or M on two attempts is FAIL."""
    s = cell["state"]
    if s in ("DEFINED", "MOUTH-SILENT"):
        return s
    if attempt1 is not None and attempt1["state"] == s and s in ("C", "M"):
        return "FAIL"
    return {"E": "MISSING", "C": "C-UNCONFIRMED", "M": "M-UNCONFIRMED"}.get(s, "UNDECIDED")


def row_reading(cells):
    finals = [c["final"] for c in cells]
    if any(f not in ("DEFINED", "MOUTH-SILENT", "FAIL") for f in finals):
        return "PENDING"
    defined = [c for c in cells if c["final"] == "DEFINED"]
    lb = [c for c in defined if c["s2"][0] == "regressed"]
    ex = [c for c in defined if c["s2"][0] in ("regressed", "pass")]
    n = len(cells)
    if all(f == "MOUTH-SILENT" for f in finals):
        return "MOUTH-SILENT"
    if len(ex) == n:
        return "MOUTH-LOAD-BEARING" if len(lb) == n else ("MOUTH-NOT-LOAD-BEARING" if not lb else "MOUTH-PARTIAL")
    if lb:
        return "MOUTH-PARTIAL"
    return "MIXED"


def score(base, tag, pin, seeds, fields_path, rows=ROWS, attempt1_dir=None, code_root=None, corpus_path=None,
          stub_tag=None):
    fields = bd.read_fields(fields_path)
    frows = fields.get("rows") or {}
    node_host, host_sha = bd.read_corpus(corpus_path) if corpus_path else ({}, {})
    primary_sha = bd.primary_corpus_sha(code_root)
    out = {"tag": tag, "pin": pin, "seeds": seeds, "rows": list(rows), "renderer_required": QWEN_RENDERER_NAME,
           "backend": "cupy", "cells": {}, "per_row": {}}
    for fac in rows:
        frow = frows.get(fac)
        if frow is None:
            raise SystemExit("no fields entry for %s in %s" % (fac, fields_path))
        cells = []
        for s in seeds:
            d = os.path.join(base, tag, "s%d" % s, fac)
            c = judge_mouth_cell(d, fac, pin, frow, host_sha or None, primary_sha)
            a1 = None
            if attempt1_dir:
                a1d = os.path.join(attempt1_dir, tag, "s%d" % s, fac)
                if os.path.isdir(a1d):
                    a1 = judge_mouth_cell(a1d, fac, pin, frow, host_sha or None, primary_sha)
            c["final"] = final_state(c, a1)
            c["seed"] = s
            if stub_tag:
                sd = os.path.join(base, stub_tag, "s%d" % s, fac)
                sj = bd.judge_dir(sd, fac, pin, fields_row=frow)
                c["stub"] = {"class": sj["class"],
                             "s2": bd.score_s2(sj, frow)[0] if sj["class"] is None else None}
            out["cells"]["s%d/%s" % (s, fac)] = c
            cells.append(c)
        defined = [c for c in cells if c["final"] == "DEFINED"]
        out["per_row"][fac] = {
            "probe_turn": frow.get("probe_turn"),
            "n_defined": len(defined),
            "n_reply_load_bearing": sum(1 for c in defined if c["s2"][0] == "regressed"),
            "n_reply_exercised": sum(1 for c in defined if c["s2"][0] in ("regressed", "pass")),
            "n_mouth_silent": sum(1 for c in cells if c["final"] == "MOUTH-SILENT"),
            "n_fail": sum(1 for c in cells if c["final"] == "FAIL"),
            "not_final": sorted("s%d" % c["seed"] for c in cells
                                if c["final"] not in ("DEFINED", "MOUTH-SILENT", "FAIL")),
            "reading": row_reading(cells),
        }
    all_cells = list(out["cells"].values())
    out["verdict"] = ("COMPLETE" if all(c["final"] in ("DEFINED", "MOUTH-SILENT", "FAIL") for c in all_cells)
                      else "INCOMPLETE")
    out["rerun_cells"] = sorted(k for k, c in out["cells"].items()
                                if c["final"] in ("MISSING", "C-UNCONFIRMED", "M-UNCONFIRMED", "UNDECIDED"))
    pre = []
    mism = bd.check_code_root(fields, code_root, pin)
    pre.append({"name": "fields pin == --pin and code-root blobs match", "ok": not mism, "value": mism})
    out["preconditions"] = pre
    if not all(p["ok"] for p in pre):
        out["verdict"] = "UNDEFINED -- a scorer precondition failed; cell-level verdict would read %s" % out["verdict"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default=TAG_DEFAULT)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=bd.SEEDS_DEFAULT)
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    ap.add_argument("--base", default="research/findings/raw/_load_bearing/_shards")
    ap.add_argument("--fields", default="research/coordination/b2d_decision_fields.json")
    ap.add_argument("--code-root", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--attempt1-dir", default=None)
    ap.add_argument("--stub-tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = score(a.base, a.tag, a.pin, a.seeds, a.fields, rows=a.rows, attempt1_dir=a.attempt1_dir,
                code_root=a.code_root, corpus_path=a.corpus, stub_tag=a.stub_tag)
    summary = {"verdict": res["verdict"], "per_row": res["per_row"], "rerun_cells": res["rerun_cells"],
               "preconditions_failed": [p["name"] for p in res["preconditions"] if not p["ok"]]}
    for k, c in sorted(res["cells"].items()):
        line = {"state": c["final"], "s2": (c.get("s2") or [None])[0], "qwen_calls": c.get("qwen_calls"),
                "fails": c["fails"][:2]}
        if "stub" in c:
            line["stub_s2"] = c["stub"]["s2"]
        summary.setdefault("cells", {})[k] = line
    print(json.dumps(summary, indent=1, default=str))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print("wrote %s" % a.out)
    return 1 if summary["preconditions_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
