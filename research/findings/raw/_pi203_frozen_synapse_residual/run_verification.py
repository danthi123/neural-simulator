#!/usr/bin/env python
"""Vikunja #203 residual verification: 5 rules x 6 seeds x 2 arms (flag OFF / flag ON).

Writes research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json.
Run from the repo root:  PYTHONPATH=. python research/findings/raw/_pi203_frozen_synapse_residual/run_verification.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tests._plastic_mask_beyond_hebbian_scenario import run  # noqa: E402

RULES = ["scaling", "structural", "reward", "rebuild", "combined"]
SEEDS = [42, 43, 44, 100, 101, 102]
OUT = os.path.join(os.path.dirname(__file__), "verification_record.json")

runs = []
for rule in RULES:
    for enforce in (False, True):
        for seed in SEEDS:
            r = run(rule, enforce=enforce, seed=seed)
            runs.append(r)
            print(f"{rule:10s} enforce={enforce} seed={seed} "
                  f"frozen_delta={r['frozen_max_delta']:.4f} control_delta={r['control_max_delta']:.4f} "
                  f"mis={r['mask_mislabelled_after_rebuild']} new={r['new_synapses_total']}/"
                  f"{r['new_synapses_labelled_plastic']} fired={r['fired_any']} sha={r['sha'][:12]}")

summary = {}
for rule in RULES:
    off = [r for r in runs if r["rule"] == rule and not r["enforce"]]
    on = [r for r in runs if r["rule"] == rule and r["enforce"]]
    summary[rule] = {
        "off_frozen_delta_min": min(r["frozen_max_delta"] for r in off),
        "off_frozen_delta_max": max(r["frozen_max_delta"] for r in off),
        "on_frozen_delta_max": max(r["frozen_max_delta"] for r in on),
        "on_frozen_delta_min": min(r["frozen_max_delta"] for r in on),
        "off_control_delta_min": min(r["control_max_delta"] for r in off),
        "on_control_delta_min": min(r["control_max_delta"] for r in on),
        "off_mislabelled": [r["mask_mislabelled_after_rebuild"] for r in off],
        "on_mislabelled": [r["mask_mislabelled_after_rebuild"] for r in on],
        "off_new_plastic": [r["new_synapses_labelled_plastic"] for r in off],
        "on_new_plastic": [r["new_synapses_labelled_plastic"] for r in on],
        "all_fired": all(r["fired_any"] for r in off + on),
    }

record = {
    "script": "research/findings/raw/_pi203_frozen_synapse_residual/run_verification.py",
    "cmd": "PYTHONPATH=. python research/findings/raw/_pi203_frozen_synapse_residual/run_verification.py",
    "provenance_note": "CPU/numpy backend (SIM_BACKEND=numpy, CUDA_VISIBLE_DEVICES empty). Each run is one "
                       "6-neuron bridge, 30 reads x 20 steps. Flag arm toggled by BRAIN_ENFORCE_PLASTIC_MASK; "
       "the OFF arm touches no new code path.",
    "commit": os.environ.get("GIT_COMMIT", ""),
    "runs": runs,
    "summary": summary,
}
with open(OUT, "w") as f:
    json.dump(record, f, indent=2)
print(json.dumps(summary, indent=2))
