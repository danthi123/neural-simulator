#!/usr/bin/env python3
"""summarize_dev7.py -- the dev-seed-7 record of the B2d packet, computed from the files (no brain build).

  python research/packets/b2d-mouth-renderer/dev7/summarize_dev7.py --shards <dir holding the dev tags> \
      --probe <dir holding the mood-render probe arms> --out <summary.json>

1. Stub-path identity (the prereg's acceptance check for the renderer change): for each of the three mouth rows, every
   file of the cell at the change's commit (tag b2d0929-stubid-dev7-e8a0d0c) against the same cell at its parent on
   main (tag b2d0929-stubid-dev7-f5e3732): arm files byte for byte, lb.json per-faculty verdict/diffs equal.
2. The mouth arm at dev seed 7 (tag b2d0929-mouth-dev7): tools/b2d_mouth_score.py's cell states with a fields file
   whose pin is set to the dev commit (the committed file keeps H until repin.py apply).
3. The mood-render probe (research/runners/_b2d_mood_render_probe.py --compare), if its arm files exist.
"""
import argparse
import filecmp
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import b2d_mouth_score as M  # noqa: E402

P_DEV = "e8a0d0cb1a32ba5ecdd867398caa0d21b0ce66c1"
M_REF = "f5e3732fed28f70d849a5e4d6cdc6ef4d206f3f8"
ROWS = ("affect-coloring", "affect-appraisal-interoceptive", "affect-drives-response")


def identity(shards):
    out = {}
    for row in ROWS:
        a = os.path.join(shards, "b2d0929-stubid-dev7-%s" % P_DEV[:7], "s7", row)
        b = os.path.join(shards, "b2d0929-stubid-dev7-%s" % M_REF[:7], "s7", row)
        if not (os.path.isdir(a) and os.path.isdir(b)):
            out[row] = {"state": "missing", "have": [os.path.isdir(a), os.path.isdir(b)]}
            continue
        arms = sorted(f for f in os.listdir(a) if not f.endswith(".prov.json") and f != "lb.json")
        arms_b = sorted(f for f in os.listdir(b) if not f.endswith(".prov.json") and f != "lb.json")
        byte_eq = {f: (f in arms_b and filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False))
                   for f in arms}
        la = json.load(open(os.path.join(a, "lb.json"))) if os.path.exists(os.path.join(a, "lb.json")) else {}
        lb = json.load(open(os.path.join(b, "lb.json"))) if os.path.exists(os.path.join(b, "lb.json")) else {}
        pick = lambda r: [{k: p.get(k) for k in ("faculty", "verdict", "load_bearing", "diffs", "null_control_clean",  # noqa
                                                 "lesion_reproduced")} for p in r.get("per_faculty", [])]
        out[row] = {"arm_files": arms, "arm_files_same_set": arms == arms_b,
                    "arm_files_byte_identical": all(byte_eq.values()) and arms == arms_b, "per_file": byte_eq,
                    "lb_rows_equal": pick(la) == pick(lb), "lb_row": pick(la)}
    return out


def mouth(shards):
    fields = json.load(open(os.path.join(ROOT, "research", "coordination", "b2d_decision_fields.json")))
    fields["pin"] = P_DEV
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(fields, fh)
        fpath = fh.name
    tree = os.path.expanduser("~/derisk-pool/revisions/%s" % P_DEV)
    res = M.score(shards, "b2d0929-mouth-dev7", P_DEV, [7], fpath, code_root=tree)
    os.unlink(fpath)
    cells = {k: {"final": c["final"], "fails": c["fails"], "s2": (c.get("s2") or [None])[0],
                 "s2_changed_fields": ((c.get("s2") or [None, {}])[1] or {}).get("changed_fields"),
                 "qwen_calls": c.get("qwen_calls"), "renderer": c.get("renderer"), "source": c.get("source"),
                 "answers": None}
             for k, c in res["cells"].items()}
    for k in cells:
        row = k.split("/", 1)[1]
        d = os.path.join(shards, "b2d0929-mouth-dev7", "s7", row)
        turn = res["cells"][k].get("turn")
        ans = {}
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if fn.endswith(".prov.json") or fn == "lb.json" or fn == "PIN.txt":
                continue
            t = json.load(open(os.path.join(d, fn))).get(turn) or {}
            ans[fn] = t.get("answer")
        cells[k]["answers"] = ans
    return {"verdict": res["verdict"], "preconditions": res["preconditions"], "per_row": res["per_row"],
            "cells": cells}


def probe(pdir):
    if not all(os.path.exists(os.path.join(pdir, "%s.json" % a)) for a in ("intact_a", "intact_b", "lesion")):
        return {"state": "missing"}
    out = os.path.join(pdir, "compare.json")
    r = subprocess.run([sys.executable, "-m", "research.runners._b2d_mood_render_probe", "--compare", pdir, "--out", out],
                       cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, SIM_NO_PROVENANCE="1"))
    if r.returncode != 0:
        return {"state": "compare failed", "stderr": r.stderr[-500:]}
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"backend": "n/a (reads dev arm files)", "device": "n/a", "execution": "summary of dev seed 7 files",
           "stub_identity": identity(a.shards), "mouth_dev7": mouth(a.shards), "mood_render_probe": probe(a.probe)}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps({"stub_identity": {r: v.get("arm_files_byte_identical") for r, v in res["stub_identity"].items()},
                      "mouth_states": {k: (v["final"], v["s2"], v["qwen_calls"]) for k, v in
                                       res["mouth_dev7"]["cells"].items()},
                      "probe": res["mood_render_probe"].get("reading", res["mood_render_probe"].get("state"))},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
