"""LOAD-BEARING FRACTION -- the OPEN-ENDED-GENERATION DISTRIBUTIONAL RULER, required-verify wrapper.

BRANCH: research/lbf-oeg-distributional-integration. Runs `load_bearing_fraction.run(only=["open-ended-
generation"])` with `LB_OPEN_ENDED_DISTRIB_PROBE=1` (the flag that branch adds to research/runners/
load_bearing_fraction.py) and wraps the result in a `tools.verdict.Verdict` preconditions block, per the project's
own discipline (an artifact asserting a verdict must carry what earned it -- tools/gates/verdict_preconditions.py
BLOCKS a bare one). `load_bearing_fraction.py` itself stays a REUSABLE INSTRUMENT (its `run()`/`main()` output shape
is unchanged for every other faculty and every other caller); this thin wrapper is where ONE particular measurement
earns a GO, mirroring the existing project convention of a separate `_..._verify.py` around a shared instrument
(e.g. `onebrain_merge_verify.py` around `onebrain_merge_framework.py`).

THE CLAIM this verifies: 'open-ended-generation' reads load-bearing via the DISTRIBUTIONAL ruler (the _followon2
draw-many plausible-fraction-of-novel lesion, 6-seed GO'd separately) rather than the single-turn field-diff (which
reads an honest treat=0 -- finding 2026-09-21-open-ended-generation-single-turn-not-load-bearing-spiking-
plausibility-gate-masks-draw.md), with a CLEAN null (an independent rebuild at the identical seed reads EXACTLY
equal, per CLAUDE.md's cfg.seed determinism guarantee -- not a statistical closeness bar).

Run (numpy CPU, memcap'd -- the "brain build" here is the _followon2 world's small unwired Izhikevich WTA banks,
NOT the webapp tiny-demo brain; this script never touches brain_chat / _spawn_arm):
  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' tools/memcap.sh 10 -- .venv/bin/python \
      -m research.runners._lbf_open_ended_distributional_verify \
      --out research/findings/raw/_load_bearing/_oed_distrib_verify/oeg_distributional_verify_GO.json
  # NOTE: this script sets LB_OPEN_ENDED_DISTRIB_PROBE=1 itself (before importing load_bearing_fraction, which
  # reads the flag at import time) -- do not also pass it on the command line; it is a no-op there either way.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
# THE WHOLE POINT of this verify: exercise the distributional ruler. Set BEFORE importing load_bearing_fraction --
# that module reads this env var ONCE at import time into a module-level constant, so setting it after import (or
# relying on the caller's shell env when something else imported the module first) would silently fall through to
# the single-turn path. The explicit re-check right after the import (below) fails loudly if that ever happens.
os.environ["LB_OPEN_ENDED_DISTRIB_PROBE"] = "1"

from research.runners.load_bearing_fraction import run, LB_OPEN_ENDED_DISTRIB  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_load_bearing/_oed_distrib_verify/"
                                     "oeg_distributional_verify_GO.json")
    args = ap.parse_args()

    if not LB_OPEN_ENDED_DISTRIB:
        raise RuntimeError(
            "LB_OPEN_ENDED_DISTRIB_PROBE did not take effect on research.runners.load_bearing_fraction -- it was "
            "imported (by something else, this process) before this script's os.environ assignment ran, or the "
            "flag string failed to parse. Refusing to write a verify artifact that would have silently measured "
            "the single-turn ruler instead of the distributional one this script exists to verify.")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    report = run(out_dir=out_dir, only=["open-ended-generation"], repeats=1, seed=args.seed)
    row = report["per_faculty"][0]
    print(json.dumps({k: row[k] for k in ("faculty", "verdict", "load_bearing", "measurement_ruler",
                                          "treatment_diffs", "control_diffs", "null_control_clean",
                                          "attributable_fraction")}, indent=2, default=str))

    # THE ATTRIBUTION CALL (tools.lab discipline, gates/attribution_required): this verify layer itself handles a
    # treatment (intact-vs-lesion)/control (intact-vs-intact-rebuild, the null) pair, so it asks the question again
    # HERE rather than only trusting the value load_bearing_fraction.py already computed -- a fresh call whose
    # result is then cross-checked (v.reads) against what the row reports, not merely re-quoted.
    from tools.lab import attributable_to
    attributable_fraction = attributable_to(
        "open-ended-generation load-bearing (distributional, verify-layer re-check)",
        row.get("treatment_diffs"), row.get("control_diffs"))

    from tools.verdict import Verdict
    v = Verdict("open-ended-generation load-bearing via the DISTRIBUTIONAL ruler (LB_OPEN_ENDED_DISTRIB_PROBE)")
    v.require("measured via the distributional ruler (not the single-turn field-diff)",
              row.get("measurement_ruler"), expect="distributional")
    v.require("the arm build succeeded (shared _followon2 world + 3 independent samplers)",
              row.get("verdict") != "arm-build-failed", expect=True)
    v.require("the NULL control is CLEAN (intact-vs-intact-rebuild, exact cfg.seed determinism)",
              row.get("null_control_clean"), expect=True)
    v.floor("treatment effect (|intact - lesion| plausible-fraction-of-novel) beats a zero floor",
           row.get("treatment_diffs"), floor=0.0)
    v.reads("attributable_fraction", row, used=attributable_fraction,
           note="this verify layer's OWN attributable_to() call must agree with what measure_faculty() reported")
    v.require("the observed effect is (near-)fully attributable to the lesion, not the null",
              attributable_fraction, expect=lambda f: f is not None and f >= 0.99)
    v.require("the lesion knob (ablate_likelihood on SpikingWTASampler) is confirmed present",
              row.get("flag_resolves"), expect=True)
    v.require("the categorical read-out is 'regressed' (changed under lesion)",
              row.get("verdict"), expect="regressed")
    v.require("load_bearing == True", row.get("load_bearing"), expect=True)
    decided = v.decide(go=bool(row.get("load_bearing") is True))

    payload = {"runner": "research.runners._lbf_open_ended_distributional_verify",
               "what_this_verifies": ("open-ended-generation reads load-bearing via the _followon2 distributional "
                                      "plausible-fraction-of-novel lesion, per the branch's remap of "
                                      "load_bearing_fraction.measure_faculty() for LB_OPEN_ENDED_DISTRIB_PROBE=1, "
                                      "with a clean intact-vs-intact-rebuild null control."),
               "seed": args.seed, "faculty_row": row, "battery_report": report}
    payload.update(decided)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2, default=str)
    print("\nVERDICT:", decided["status"])
    print("wrote", args.out)
    return 0 if decided["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
