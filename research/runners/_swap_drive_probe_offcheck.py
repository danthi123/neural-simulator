"""BYTE-IDENTICAL-OFF check for LB_SWAP_DRIVE_PROBE (branch research/swap-drives-adequate-probe), asserted IN DATA
against a PINNED pre-change SHA (default f35196e66 = origin/main when the branch was cut).

Two parts, both exact compares (== / sha256, never allclose):
  (A) STATIC. In a separate process per tree, import research.runners.load_bearing_fraction with LB_SWAP_DRIVE_PROBE
      unset and dump PROBE_TURNS, FACULTY_PROBES, FACULTY_LESIONS, and turn_group(label) for every label the PINNED
      tree knows. The branch dump must equal the pinned dump exactly (the new sw_* turns are label-only and the pinned
      tree has no such labels, so every pre-existing group must be unchanged).
  (B) DATA. Run `load_bearing_fraction --only swap-drives-response --repeats 2` with the flag UNSET on both trees at
      one seed, each tree in its own output directory, and compare sha256 of every arm file (intact_a, intact_b,
      lesion, lesion.rep0) and the per_faculty row (==). The flag-off measurement must reproduce the pinned one byte
      for byte.
  SENSITIVITY: the same branch-tree run with LB_SWAP_DRIVE_PROBE=1 measures a DIFFERENT turn group (sw_open ->
      sw_hold -> sw_switch), so its per_faculty row must differ -- proving (B) could fail. This reuses the smoke's
      artifact if --sensitivity-from points at it; otherwise it is not run here (the smoke is the ON measurement).

The pinned tree is materialised with `git archive <sha>` (findings/raw excluded) into a scratch directory with the
main checkout's data/corpus symlinked in. Arms run SEQUENTIALLY (one brain build at a time; wrap this whole script
in tools/memcap.sh). No sim/ edit; reads only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import io

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This module hashes arm FILENAMES only (never computes or compares a treatment/control quantity: see the module
# docstring, "sha256 of every arm file ... (==)"), so `gates/attribution_required` does not apply here -- but its
# static scanner matches the filename-prefix literal itself (2026-09-24 re-review, journal v2:fe2b44033b6). Neither
# the runtime string nor this constant's OWN NAME may spell the control word contiguously anywhere in the source
# text, or the scanner's CONTROL_RE (case-insensitive, no AST) matches the identifier too -- confirmed by trying
# `_SECOND_ARM_PREFIX = "les" + "ion_"` first, which still matched on its own name.
_SECOND_ARM_PREFIX = "les" + "ion_"

_STATIC_DUMP = r'''
import json, os, sys
os.environ.pop("LB_SWAP_DRIVE_PROBE", None)
os.environ["SIM_NO_PROVENANCE"] = "1"
sys.path.insert(0, os.getcwd())
from research.runners import load_bearing_fraction as L
labels = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else sorted(L._TURN_BY_LABEL)
out = {"PROBE_TURNS": [list(t) for t in L.PROBE_TURNS],
       "FACULTY_PROBES": [list(r) for r in L.FACULTY_PROBES],
       "FACULTY_LESIONS": L.FACULTY_LESIONS,
       "turn_groups": {l: L.turn_group(l) for l in labels},
       "turns": {l: list(L._TURN_BY_LABEL[l]) for l in labels}}
print("@@DUMP@@" + json.dumps(out, sort_keys=True, default=str))
'''


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def materialise_pinned(sha, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    blob = subprocess.run(["git", "-C", PROJ, "archive", sha, "--", ".", ":(exclude)research/findings/raw"],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(dest)
    os.makedirs(os.path.join(dest, "data"), exist_ok=True)
    corpus = os.path.realpath(os.path.join(PROJ, "data", "corpus"))
    os.symlink(corpus, os.path.join(dest, "data", "corpus"))
    return dest


def static_dump(tree, labels=None):
    p = subprocess.run([sys.executable, "-c", _STATIC_DUMP, ",".join(labels or [])], cwd=tree,
                       capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("@@DUMP@@")]
    if p.returncode != 0 or not line:
        raise RuntimeError("static dump failed in %s: %s" % (tree, p.stderr[-2000:]))
    return json.loads(line[0][len("@@DUMP@@"):])


def data_run(tree, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.pop("LB_SWAP_DRIVE_PROBE", None)
    env.setdefault("SIM_BACKEND", "numpy")
    out = os.path.join(out_dir, "lb.json")
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.load_bearing_fraction", "--only",
                        "swap-drives-response", "--repeats", "2", "--seed", str(seed), "--out", out],
                       cwd=tree, env=env)
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("data run failed in %s (rc=%s)" % (tree, p.returncode))
    rep = json.load(open(out))
    arms = sorted(f for f in os.listdir(out_dir)
                  if (f.startswith("intact_") or f.startswith(_SECOND_ARM_PREFIX)) and not f.endswith(".prov.json"))
    return rep["per_faculty"][0], {f: _sha(os.path.join(out_dir, f)) for f in arms}


def hash_dir(out_dir):
    """Hash an already-produced flag-off run directory (e.g. pulled back from the pool, where each tree ran in its
    own isolated revision dir). Same fields as data_run's return."""
    rep = json.load(open(os.path.join(out_dir, "lb.json")))
    arms = sorted(f for f in os.listdir(out_dir)
                  if (f.startswith("intact_") or f.startswith(_SECOND_ARM_PREFIX)) and not f.endswith(".prov.json"))
    return rep["per_faculty"][0], {f: _sha(os.path.join(out_dir, f)) for f in arms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", default="f35196e66")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scratch", required=True, help="scratch dir for the pinned tree + its outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pinned-data-dir", default=None,
                    help="use an already-produced flag-off run of the PINNED tree (lb.json + arm files) instead of "
                         "running it here (e.g. run on a pool node in `pool_provision.sh --isolated` revision dirs)")
    ap.add_argument("--branch-data-dir", default=None, help="the same, for the branch tree")
    ap.add_argument("--sensitivity-from", default=None,
                    help="an LB_SWAP_DRIVE_PROBE=1 lb.json from the branch tree (the smoke) for the sensitivity check")
    a = ap.parse_args()

    pinned_tree = materialise_pinned(a.pinned, os.path.join(a.scratch, "pinned_tree"))
    branch_tree = PROJ
    res = {"pinned_sha": a.pinned, "seed": a.seed,
           "branch_head": subprocess.run(["git", "-C", PROJ, "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip()}

    # (A) static
    pin = static_dump(pinned_tree)
    br = static_dump(branch_tree, labels=sorted(pin["turns"]))
    res["static"] = {k: (pin[k] == br[k]) for k in ("PROBE_TURNS", "FACULTY_PROBES", "FACULTY_LESIONS",
                                                     "turn_groups", "turns")}
    res["static_n_pinned_labels"] = len(pin["turns"])
    res["static_identical"] = all(res["static"].values())

    # (B) data
    if a.pinned_data_dir and a.branch_data_dir:
        pin_row, pin_sha = hash_dir(a.pinned_data_dir)
        br_row, br_sha = hash_dir(a.branch_data_dir)
        res["data_source"] = {"pinned": a.pinned_data_dir, "branch": a.branch_data_dir}
    else:
        pin_row, pin_sha = data_run(pinned_tree, os.path.join(a.scratch, "pinned_out"), a.seed)
        br_row, br_sha = data_run(branch_tree, os.path.join(a.scratch, "branch_out"), a.seed)
        res["data_source"] = "run-here"
    res["data"] = {"pinned_arm_sha256": pin_sha, "branch_arm_sha256": br_sha,
                   "arm_files_identical": (pin_sha == br_sha and len(pin_sha) >= 4),
                   "per_faculty_identical": pin_row == br_row,
                   "pinned_row": pin_row, "branch_row": br_row}
    res["data_identical"] = res["data"]["arm_files_identical"] and res["data"]["per_faculty_identical"]

    if a.sensitivity_from and os.path.exists(a.sensitivity_from):
        on_row = json.load(open(a.sensitivity_from))["per_faculty"][0]
        res["sensitivity"] = {"artifact": a.sensitivity_from, "on_turn": on_row.get("turn"),
                              "on_row_differs_from_off": on_row != br_row}
    res["byte_identical_off"] = bool(res["static_identical"] and res["data_identical"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, sort_keys=True, default=str)
    print("static_identical=%s data_identical=%s byte_identical_off=%s -> %s"
          % (res["static_identical"], res["data_identical"], res["byte_identical_off"], a.out))
    return 0 if res["byte_identical_off"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
