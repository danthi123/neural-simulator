#!/usr/bin/env python
"""Vikunja #203 residual battery: does cp_synapse_plastic_mask actually freeze a declared-non-plastic synapse
at EVERY write site to cp_connections.data, or only at the Hebbian sites the 2026-09-02 fix reached?

Six rules, each isolating one write site in sim/bridge.py:
  scaling   homeostatic synaptic scaling          structural  structural-plasticity elimination
  reward    reward-modulation update + clip       rebuild     mask remap across a CSR re-sort
  compaction the DEFERRED CSR compaction (_compact_synapse_arrays) after elimination -- a second re-ordering
  combined  scaling + structural + reward on ONE bridge (the integration case)

Arms: flag OFF (BRAIN_ENFORCE_PLASTIC_MASK unset -> no new branch reachable) vs flag ON.
Each run is a small numpy bridge (6 neurons; 8 for the compaction rule, which needs an eliminated set sitting
BETWEEN surviving ones so the re-packing actually shifts indices), 30 reads x 20 steps, with a frozen population (declared plastic=False,
NO named plasticity_gate, so only the mask can protect it) and a control population (plastic=True) that MUST
move for a zero in the frozen column to mean anything.

Run:  PYTHONPATH=. CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy python -m research.runners.pi203_frozen_synapse_battery \
        --out research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._plastic_mask_beyond_hebbian_scenario import run  # noqa: E402

RULES = ["scaling", "structural", "reward", "rebuild", "compaction", "combined", "hebbian"]
SEEDS = [42, 43, 44, 100, 101, 102]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.environ.setdefault("SIM_BACKEND", "numpy")

    runs, t0 = [], time.time()
    for rule in RULES:
        for enforce in (False, True):
            for seed in seeds:
                r = run(rule, enforce=enforce, seed=seed)
                r["elapsed_seconds"] = round(time.time() - t0, 2)
                runs.append(r)
                print(f"{rule:10s} enforce={enforce} seed={seed} "
                      f"frozen_delta={r['frozen_max_delta']:.4f} control_delta={r['control_max_delta']:.4f} "
                      f"mis={r['mask_mislabelled_after_rebuild']} end_mis={r['mask_mislabelled_at_end']} "
                      f"new={r['new_synapses_total']}/{r['new_synapses_labelled_plastic']} "
                      f"fired={r['fired_any']} sha={r['sha'][:12]}", flush=True)

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
            "off_mislabelled_at_end": [r["mask_mislabelled_at_end"] for r in off],
            "on_mislabelled_at_end": [r["mask_mislabelled_at_end"] for r in on],
            "off_new_plastic": [r["new_synapses_labelled_plastic"] for r in off],
            "on_new_plastic": [r["new_synapses_labelled_plastic"] for r in on],
        }

    record = {
        "runner": "research.runners.pi203_frozen_synapse_battery",
        "script": "research/runners/pi203_frozen_synapse_battery.py",
        "cmd": " ".join(sys.argv),
        "backend": os.environ.get("SIM_BACKEND", "numpy"),
        "device": "CPU",
        "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
        "elapsed_seconds": round(time.time() - t0, 2),
        "cost_projection": f"{len(runs)} runs x 30 reads x 20 steps of a 6-neuron numpy bridge; measured total "
                           "wall time is elapsed_seconds above. No GPU used: no GPU-hour tail.",
        "provenance_note": "CPU/numpy (SIM_BACKEND=numpy, CUDA_VISIBLE_DEVICES empty). Arms toggled by "
                           "BRAIN_ENFORCE_PLASTIC_MASK; cfg.enforce_plastic_mask_beyond_hebbian is the config "
                           "mirror of the same gate. cfg.seed is passed explicitly per run.",
        "rules": RULES,
        "seeds": seeds,
        "runs": runs,
        "summary": summary,
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[battery] {len(runs)} runs in {record['elapsed_seconds']}s -> {args.out}")


if __name__ == "__main__":
    main()
