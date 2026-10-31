#!/usr/bin/env python
"""Backend confirmation for the #203 freeze: does `cp_synapse_plastic_mask` gate the writes on the CUPY path too?

WHY THIS RUN EXISTS. Every number behind the #203 merge and its follow-up was measured on numpy/CPU:
research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json (72 runs),
verification_record_v2.json (84 runs), byte_identity_record.json (42/42), write_path_attribution_v2.json (16 cells).
The Claude merge_review made the default flip conditional on a cupy confirmation, and the first GPU job queued for
this follow-up printed "NumPy backend" for all 84 of its runs -- because tests/_plastic_mask_beyond_hebbian_scenario.py
set `os.environ["SIM_BACKEND"] = "numpy"` HARD. An env var that overrules the caller is exactly the trap
tools/lab.assert_backend exists for (tools/lab.py:799). So the scenario now uses setdefault and this runner records
the backend the engine ACTUALLY resolved, from sim.backend.get_backend(), not the one asked for.

WHAT IT MEASURES (dev seeds only for the organ: 7, 8; the generalization seeds are untouched by this instrument):
  (A) the seven-rule scenario battery on the requested backend, flag OFF vs ON, per seed: the frozen population's
      max delta, the control population's delta (non-vacuity: the control MUST move), mask mislabelling;
  (B) the production comprehension organ (research.runners.comprehension_production_organ), 30 successive reads of
      one sentence, OFF vs ON: coordinate-keyed drift of the synapses the mask declares frozen vs the ones it
      declares plastic.

The write tap (research/runners/pi203_write_path_tap.py) is a numpy ndarray subclass; it cannot wrap a cupy buffer.
So on cupy the per-site write table is NOT available and this runner does not claim it -- it measures the drift of
the frozen population by coordinate, which is the same invariant the tap test asserts, one level up.

Run:  SIM_BACKEND=cupy PYTHONPATH=. python -m research.runners.pi203_cupy_backend_confirmation \
        --out research/findings/raw/_pi203_frozen_synapse_residual/verification_record_cupy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from tools.lab import assert_backend, attributable_to, term_budget, project_cost  # noqa: E402

RULES = ["scaling", "structural", "reward", "rebuild", "compaction", "combined", "hebbian"]
SEEDS = [42, 43, 44, 100, 101, 102]
READS = 30
SENTENCE = "the monkey grab the box"
TEACHER_FLAGS = {
    "BRAIN_LEARNED_ANIMACY_CUE": "1",
    "BRAIN_LEARNED_VERB_SELECTS": "1",
    "BRAIN_LEARNED_ANIMACY_LESION": "0",
    "BRAIN_LEARNED_VERB_SELECTS_LESION": "0",
}


def _npify(x):
    """cupy -> numpy without the implicit-conversion error; numpy passes through unchanged."""
    if hasattr(x, "get") and not isinstance(x, np.ndarray):
        return np.asarray(x.get())
    return np.asarray(x)


def _coo(bridge):
    """The organ's weight store as a COO triple, whatever the backend handed us (scipy sparse on numpy, a
    dense/cupy buffer if the GPU path stores it that way). One place, so _coords and _frozen_coords cannot
    disagree about what a synapse IS."""
    csr = bridge.cp_connections
    if hasattr(csr, "tocoo"):
        return csr.tocoo()
    from scipy.sparse import coo_matrix
    return coo_matrix(np.asarray(csr)).tocoo()


def _coords(bridge) -> dict:
    """(pre, post) coordinate -> weight. Identity-based, so a CSR re-ordering cannot fake a zero."""
    coo = _coo(bridge)
    return {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}


def _frozen_coords(bridge) -> set:
    mask = getattr(bridge, "cp_synapse_plastic_mask", None)
    if mask is None:
        return set()
    mask = _npify(mask)
    coo = _coo(bridge)
    return {(int(r), int(c)) for k, (r, c) in enumerate(zip(coo.row, coo.col))
            if k < mask.size and not mask[k]}


def _organ_arm(enforce: bool, seed: int) -> dict:
    for k, v in TEACHER_FLAGS.items():
        os.environ[k] = v
    os.environ["BRAIN_ENFORCE_PLASTIC_MASK"] = "1" if enforce else "0"

    from research.runners.comprehension_production_organ import ComprehensionProductionOrgan

    organ = ComprehensionProductionOrgan(seed=seed)
    organ.ensure_built()
    bridge = organ.comp.bridge
    before = _coords(bridge)
    frozen = _frozen_coords(bridge)
    for _ in range(READS):
        organ.judge(SENTENCE)
    after = _coords(bridge)
    common = set(before) & set(after)
    frozen_drift = {cc: abs(after[cc] - before[cc]) for cc in common if cc in frozen}
    plastic_drift = {cc: abs(after[cc] - before[cc]) for cc in common if cc not in frozen}
    return {
        "seed": seed, "enforce": enforce, "reads": READS,
        "n_synapses": len(after), "n_frozen": len(frozen),
        "frozen_max_abs_drift": max(frozen_drift.values(), default=0.0),
        "frozen_synapses_drifted": sum(1 for d in frozen_drift.values() if d > 1e-9),
        "plastic_max_abs_drift": max(plastic_drift.values(), default=0.0),
        "plastic_synapses_drifted": sum(1 for d in plastic_drift.values() if d > 1e-9),
        "reply": organ.judge(SENTENCE),
        "reply_margin": (organ.judge(SENTENCE) or {}).get("margin"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_pi203_frozen_synapse_residual/verification_record_cupy.json")
    ap.add_argument("--backend", default="cupy", help="backend this confirmation is FOR (cupy)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--skip-organ", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    os.environ.setdefault("SIM_BACKEND", args.backend)
    assert_backend(args.backend, "the #203 freeze must be shown to gate the writes on this path, not only on CPU")
    from sim.backend import get_backend
    _, resolved = get_backend()
    if resolved != args.backend:
        raise SystemExit(f"⛔ backend mismatch: asked {args.backend!r}, engine resolved {resolved!r}. "
                         "This run would be a CPU run wearing a GPU label.")

    t0 = time.time()
    runs = []
    for rule in RULES:
        for enforce in (False, True):
            for seed in seeds:
                from tests._plastic_mask_beyond_hebbian_scenario import run
                r = run(rule, enforce=enforce, seed=seed)
                r["backend_resolved"] = resolved
                runs.append(r)
                print(f"{rule:10s} enforce={enforce} seed={seed} "
                      f"frozen_delta={r['frozen_max_delta']:.4f} control_delta={r['control_max_delta']:.4f} "
                      f"mis={r['mask_mislabelled_after_rebuild']} end_mis={r['mask_mislabelled_at_end']} "
                      f"fired={r['fired_any']} sha={r['sha'][:12]}", flush=True)
            off = [r for r in runs if r["rule"] == rule and not r["enforce"]]
            on = [r for r in runs if r["rule"] == rule and r["enforce"]]
            if off and on:
                attributable_to(
                    f"{rule} frozen drift on the {resolved} backend (OFF arm vs ON arm)",
                    max(r["frozen_max_delta"] for r in off),
                    max(r["frozen_max_delta"] for r in on),
                )

    gross = sum(r["frozen_max_delta"] for r in runs if r["enforce"]) + sum(
        r["control_max_delta"] for r in runs if r["enforce"])
    term_budget("gross weight drift surviving the flag ON, split by WHO is allowed to move: the mask-declared "
               "frozen population (must be 0) vs the mask-declared plastic population (moving is the design, "
               "not a leak)", gross if gross > 0 else 1e-12,
               {"frozen-population delta (ON arm, the leak this measures)": sum(r["frozen_max_delta"] for r in runs if r["enforce"]),
                "control-population delta (ON arm, plastic by declaration)": sum(r["control_max_delta"] for r in runs if r["enforce"])})

    organ_arms = []
    if not args.skip_organ:
        for seed in (7, 8):
            for enforce in (False, True):
                a = _organ_arm(enforce, seed)
                organ_arms.append(a)
                print(f"organ seed={seed} enforce={enforce} frozen={a['n_frozen']} "
                      f"frozen_drift={a['frozen_max_abs_drift']:.6f} drifted={a['frozen_synapses_drifted']} "
                      f"plastic_drift={a['plastic_max_abs_drift']:.6f}", flush=True)
        for seed in (7, 8):
            off = next(a for a in organ_arms if a["seed"] == seed and not a["enforce"])
            on = next(a for a in organ_arms if a["seed"] == seed and a["enforce"])
            attributable_to(f"organ frozen drift seed {seed} on {resolved} (OFF arm vs ON arm)",
                            off["frozen_max_abs_drift"], on["frozen_max_abs_drift"])

    elapsed = round(time.time() - t0, 2)
    record = {
        "runner": "research.runners.pi203_cupy_backend_confirmation",
        "script": "research/runners/pi203_cupy_backend_confirmation.py",
        "cmd": " ".join(sys.argv),
        "backend_requested": args.backend,
        "backend_resolved": resolved,
        "device": "GPU (cupy)" if resolved == "cupy" else "CPU",
        "elapsed_seconds": elapsed,
        "cost_projection": f"{len(runs)} scenario runs x 30 reads x 20 steps of a 6-neuron bridge"
                           f"{' + %d organ arms x %d reads' % (len(organ_arms), READS) if organ_arms else ''}; "
                           f"measured wall time {elapsed}s on this backend. GPU minutes: elapsed/60.",
        "provenance_note": f"Backend resolved by sim.backend.get_backend() == {resolved}; the scenario now "
                           "setdefaults SIM_BACKEND so an explicit GPU ask is honoured (it previously forced numpy).",
        "rules": RULES, "seeds": seeds, "runs": runs, "organ_arms": organ_arms,
        "note_on_tap": "The write tap is a numpy ndarray subclass and cannot wrap a cupy buffer; the per-site "
                       "write table is not measured here. What IS measured is the coordinate-keyed drift of the "
                       "frozen population, the same invariant the tap-based test asserts.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[cupy-confirm] {len(runs)} scenario runs + {len(organ_arms)} organ arms in {elapsed}s -> {args.out}")


if __name__ == "__main__":
    main()
