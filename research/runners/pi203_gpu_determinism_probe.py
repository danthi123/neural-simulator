#!/usr/bin/env python
"""Is the cupy path reproducible run-to-run? This decides whether "byte-identical" means anything on the GPU.

WHY THIS RUN EXISTS (2026-10-31, Vikunja #203 follow-up). `tests/test_plastic_mask_beyond_hebbian.py` and
`tests/test_comprehension_frozen_flat.py` assert exact SHA/digest equality (flag-OFF == pre-fix code; tapped ==
untapped). On numpy all 42 comparisons held (research/findings/raw/_pi203_frozen_synapse_residual/byte_identity_record.json).
Run under `SIM_BACKEND=cupy` every one of them failed (research/findings/raw/_pi203_frozen_synapse_residual/cupy_frozen_flat_pytest_v3.txt:
"flag-OFF is NOT byte-identical to pre-fix for rule scaling: off=c7936d... prefix=60cf4a..."), and the
instrument-inertness test failed too. Two explanations are live and they lead to opposite conclusions:
  (a) the tap or the fix genuinely changes the numbers on the GPU path, or
  (b) the GPU path is not bit-reproducible between two runs of IDENTICAL code, in which case an exact-equality
      assertion is void on that backend no matter what the code does.
Only a control that compares identical code to itself can separate them. This runner does exactly that:

  A = organ seed 7, untapped, 25 battery items            digest
  B = organ seed 7, untapped, same items, built again     digest      <- A==B means the backend is reproducible
  C = organ seed 7, tapped                                digest      <- A==C means the tap is inert

and the same for the seven-rule scenario: run(rule, enforce=False, seed) twice, compare SHA.

Run:  SIM_BACKEND=cupy PYTHONPATH=. python -m research.runners.pi203_gpu_determinism_probe \
        --out research/findings/raw/_pi203_frozen_synapse_residual/gpu_determinism_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.lab import assert_backend, attributable_to, lever, project_cost  # noqa: E402

TEACHER_FLAGS = {
    "BRAIN_LEARNED_ANIMACY_CUE": "1",
    "BRAIN_LEARNED_VERB_SELECTS": "1",
    "BRAIN_LEARNED_VERB_SELECTS_LESION": "0",
    "BRAIN_LEARNED_ANIMACY_LESION": "0",
}
SEED = 7


def _organ_digest(tap: bool) -> dict:
    from research.runners import pi203_write_path_tap as tapmod
    from research.runners._comprehension_learned_cues_joint_arm import all_items
    from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, extract_transitive

    for k, v in TEACHER_FLAGS.items():
        os.environ[k] = v
    os.environ["BRAIN_ENFORCE_PLASTIC_MASK"] = "0"
    organ = ComprehensionProductionOrgan(seed=SEED)
    organ.ensure_built()
    bridge = organ.comp.bridge
    if tap:
        tapmod.install_tap(bridge)
    out = []
    for label, n0, v, n1 in all_items():
        text = f"the {n0} {v} the {n1}"
        tr = extract_transitive(text)
        out.append((label, organ.judge(text), organ.competent(*tr) if tr else None))
    return {"digest": tapmod._digest(out), "n_items": len(out),
            "tap_blind": tapmod.tap_is_blind(bridge) if tap else False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cupy")
    ap.add_argument("--rules", default="hebbian,reward,scaling")
    ap.add_argument("--prereg", default="research/findings/2026-09-24-plastic-mask-flip-PREREGISTRATION.md")
    ap.add_argument("--out", default="research/findings/raw/_pi203_frozen_synapse_residual/gpu_determinism_probe.json")
    args = ap.parse_args()
    assert_backend(args.backend, "the question is whether THIS backend reproduces itself")

    t0 = time.time()
    first_unit_s = None
    A = _organ_digest(tap=False)
    first_unit_s = round(time.time() - t0, 2)
    project_cost(f"{args.backend} organ build + 25-item battery", 1, 3, first_unit_s)
    B = _organ_digest(tap=False)
    C = _organ_digest(tap=True)

    from tests._plastic_mask_beyond_hebbian_scenario import run
    rule_pairs = {}
    for rule in args.rules.split(","):
        r1 = run(rule, enforce=False, seed=42)
        r2 = run(rule, enforce=False, seed=42)
        rule_pairs[rule] = {"sha_1": r1["sha"], "sha_2": r2["sha"],
                           "identical": r1["sha"] == r2["sha"],
                           "frozen_delta": r1["frozen_max_delta"], "control_delta": r1["control_max_delta"]}

    organ_identical = A["digest"] == B["digest"]
    tap_identical = A["digest"] == C["digest"]
    lever(f"{args.backend} reproducibility of identical code (A vs B)", 0.0 if organ_identical else 1.0, 0.0,
          required=False)
    attributable_to("reply digest difference: tap installed (C) vs not installed (A)",
                    0.0 if tap_identical else 1.0, 0.0 if organ_identical else 1.0)

    record = {
        "runner": "research.runners.pi203_gpu_determinism_probe",
        "script": "research/runners/pi203_gpu_determinism_probe.py",
        "cmd": " ".join(sys.argv),
        "backend_requested": args.backend,
        "prereg": args.prereg,
        "elapsed_seconds": round(time.time() - t0, 2),
        "organ_digest_A_untapped": A,
        "organ_digest_B_untapped_again": B,
        "organ_digest_C_tapped": C,
        "identical_code_identical_run": organ_identical,
        "tap_inert": tap_identical,
        "scenario_rule_repeats": rule_pairs,
        "summary": f"backend {args.backend}: identical code twice -> {'IDENTICAL' if organ_identical else 'DIFFERENT'} "
                   f"(A {A['digest'][:12]} vs B {B['digest'][:12]}); tap vs no tap -> "
                   f"{'IDENTICAL' if tap_identical else 'DIFFERENT'} (A vs C {C['digest'][:12]}); "
                   f"scenario rules repeated: {sum(1 for v in rule_pairs.values() if v['identical'])}/{len(rule_pairs)} "
                   "identical SHAs.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(record["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
