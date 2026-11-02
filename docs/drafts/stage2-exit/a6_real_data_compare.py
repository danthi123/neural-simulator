"""Amendment A6 (DRAFT 2026-11-01): what the amended scorer returns vs the current one on the REAL Stage-2 data.

Reads, for each data set, the aggregate of (1) the PRE-A6 runner (the file at commit 25e360740, passed as --old-runner),
(2) this runner at level A5 (must be byte-identical to (1)), (3) this runner at level A6. Data sets: the 10-26 gate
(six gate seeds, research/findings/raw/_stage2_exit/gate) and the 11-01 dev preview at 41d76bdbe (dev seeds 1-3,
--preview-root; 105/105 cells: the three seed-3 cells not yet committed on the workstation branch were read from the
home server's landed copies, sha256 in the output). Nothing is run; no verdict is filed.

    git show 25e360740:research/runners/stage2_exit_gate.py > _scratch/s2retake/old_gate.py
    .venv/bin/python docs/drafts/stage2-exit/a6_real_data_compare.py --old-runner _scratch/s2retake/old_gate.py \
        --preview-root _scratch/s2retake/preview_full --out docs/drafts/stage2-exit/a6_real_data_compare.json
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, _REPO)
from research.runners import stage2_exit_gate as NEW  # noqa: E402


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-runner", required=True)
    ap.add_argument("--preview-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("old_gate", a.old_runner)
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    sets = {"gate_1026": ("research/findings/raw/_stage2_exit/gate", (42, 43, 44, 100, 101, 102)),
            "preview_1101": (a.preview_root, (1, 2, 3))}
    res = {"backend": "numpy", "device": "cpu", "analysis_only": True, "old_runner_sha256": _sha(a.old_runner), "new_runner_sha256": _sha(NEW.__file__),
           "preview_seed3_server_copies_sha256": {n: _sha(os.path.join(a.preview_root, "s3", n))
                                                  for n in ("emo__intact_b.json", "inner__intact_a.json", "inner__intact_b.json")
                                                  if os.path.exists(os.path.join(a.preview_root, "s3", n))}}
    for name, (root, seeds) in sets.items():
        with contextlib.redirect_stdout(io.StringIO()):
            o = OLD.aggregate(root, seeds=seeds)
            n5 = NEW.aggregate(root, seeds=seeds)
            n6 = NEW.aggregate(root, seeds=seeds, amendment_level="A6")
        res[name] = {
            "seeds": list(seeds), "n_missing_shards": len(o["missing_shards"]),
            "A5_aggregate_identical_to_pre_A6_runner": json.dumps(o, sort_keys=True, default=str) == json.dumps(n5, sort_keys=True, default=str),
            "A5_report_identical_to_pre_A6_runner": OLD.report(o) == NEW.report(n5),
            "pre_A6": {"stage": o["verdict"], "criteria": o["criteria"],
                       "C2": {k: o["groups"]["infer"][k] for k in ("verdict", "why", "n_defined", "n_correct", "n_permuted_tracks")}},
            "A6": {"stage": n6["verdict"], "criteria": n6["criteria"],
                   "C2": {k: n6["groups"]["infer"][k] for k in ("verdict", "why", "n_defined", "n_correct", "n_permuted_tracks")},
                   "C2_seeds": {str(s): {"permuted_acquisition": r.get("permuted_acquisition"), "tracks": r.get("tracks"),
                                         "tracks_as_scored_pre_A6": r.get("tracks_as_scored_pre_A6"),
                                         "undefined_reasons": r.get("undefined_reasons")}
                                for s, r in n6["groups"]["infer"]["seeds"].items()},
                   "preconditions_unmet": n6["undefined_reasons"]}}
        print(name, res[name]["A5_aggregate_identical_to_pre_A6_runner"], res[name]["pre_A6"]["criteria"], res[name]["A6"]["criteria"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
