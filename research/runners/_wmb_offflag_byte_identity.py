"""Byte-identity assertion (IN DATA): LB_WMB_HOLDQUERY_PROBE, LB_WMB_CONTENT_PROBE and BRAIN_MULTIREF_LESION_SCOPE
all OFF vs the pinned pre-change SHA.

Compares two completed `load_bearing_fraction --only wm-binding-advanced --repeats 2` output directories -- one run
from an extracted tree at the pinned SHA (default 0c265b93d, the origin/main this fix round merged last; this lane's
changes are absent there), one from this branch with every flag unset -- by EXACT sha256 of every arm file and of the
per-faculty record, and hashes `PROBE_TURNS` / `FACULTY_PROBES` as imported from each tree. No brain build.
Pre-registrations: research/findings/2026-09-23-wm-binding-holdquery-adequate-probe-PREREGISTRATION.md and
research/findings/2026-09-24-wm-binding-ordinary-content-probe-PREREGISTRATION.md.

  .venv/bin/python -m research.runners._wmb_offflag_byte_identity --pinned-dir D1 --pinned-tree T1 \
      --branch-dir D2 --branch-tree T2 --out research/findings/raw/_load_bearing/wmb_holdquery/offflag_byte_identity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ARM_FILES = ["intact_a_hold_held.json", "intact_b_hold_held.json", "lesion_wm_binding_advanced.json",
             "lesion_wm_binding_advanced.json.rep0"]

_ROSTER_SNIPPET = (
    "import json,hashlib;from research.runners.onebrain_regression_battery import PROBE_TURNS,FACULTY_PROBES;"
    "h=lambda o:hashlib.sha256(json.dumps(o,sort_keys=True,default=str).encode()).hexdigest();"
    "print(json.dumps({'PROBE_TURNS':h(PROBE_TURNS),'FACULTY_PROBES':h(FACULTY_PROBES),"
    "'n_probe_turns':len(PROBE_TURNS)}))")


def _sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def _canon(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def _record(d):
    rep = json.load(open(os.path.join(d, "lbf.json")))
    return [p for p in rep["per_faculty"] if p["faculty"] == "wm-binding-advanced"][0]


def _roster(tree):
    out = subprocess.run([sys.executable, "-c", _ROSTER_SNIPPET], cwd=tree, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": tree})
    return json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {"error": out.stderr[-500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned-dir", required=True)
    ap.add_argument("--pinned-tree", required=True)
    ap.add_argument("--pinned-sha", default="0c265b93d")
    ap.add_argument("--branch-dir", required=True)
    ap.add_argument("--branch-tree", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {}
    for f in ARM_FILES:
        sp, sb = _sha_file(os.path.join(a.pinned_dir, f)), _sha_file(os.path.join(a.branch_dir, f))
        arms[f] = {"pinned": sp, "branch": sb, "identical": bool(sp is not None and sp == sb)}
    rp, rb = _record(a.pinned_dir), _record(a.branch_dir)
    record = {"pinned": _canon(rp), "branch": _canon(rb), "identical": _canon(rp) == _canon(rb),
              "verdict_pinned": rp.get("verdict"), "verdict_branch": rb.get("verdict")}
    ros_p, ros_b = _roster(a.pinned_tree), _roster(a.branch_tree)
    roster = {"pinned": ros_p, "branch": ros_b, "identical": ("error" not in ros_p and ros_p == ros_b)}
    identical = all(v["identical"] for v in arms.values()) and record["identical"] and roster["identical"]
    art = {"runner": "research.runners._wmb_offflag_byte_identity", "pinned_sha": a.pinned_sha,
           "flag": "LB_WMB_HOLDQUERY_PROBE + LB_WMB_CONTENT_PROBE + BRAIN_MULTIREF_LESION_SCOPE (unset in both runs)", "compare": "exact sha256",
           "arm_files": arms, "per_faculty_record": record, "roster": roster,
           "byte_identical_off": identical}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(art, open(a.out, "w"), indent=2)
    print(json.dumps({"byte_identical_off": identical, "arms": {k: v["identical"] for k, v in arms.items()},
                      "record": record["identical"], "roster": roster["identical"]}))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
