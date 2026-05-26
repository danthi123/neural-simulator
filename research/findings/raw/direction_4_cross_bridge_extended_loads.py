"""Direction 4 extended-load envelope wrapper — characterise the
bio_brain_regions 5-bridge dedicated-pool substrate's cross-bridge
FHRR composition envelope at loads {4, 6, 7}, complementing the
existing production probe at LOADS={2, 3, 5}.

CPU-only by design (operates on Task 5 activity caches already on
disk; no retraining, no GPU). Pure reuse-by-import of
`direction_4_cross_bridge_probe.run_cross_bridge_probe`: the parent
function accepts an arbitrary `loads` list, so this wrapper only
overrides that argument and writes a separate output JSON so the
existing production result file is untouched.

The extended loads complement the production grid:
- L={2,3,5} (existing): 1.000 / 1.000, 1.000 / 1.000, 1.000 / 0.977
- L={4,6,7} (this wrapper): characterise the high-L envelope to find
  where OI first dips below the 0.80 bar (or whether it stays above
  through L=7, the N_GAMMA_SLOTS ceiling).

Comparison anchors:
- OPTION 3 V=16 load-ceiling map (2026-05-24): bio_brain_regions
  single-bridge substrates PASS at every load L=2..7 multi-seed; L=7
  OI 0.900 / 0.895 / 0.935 across OPTION 3 / HIPPO / DLPFC.
- Cross-bridge G.20 sparse 160-concept union (2026-05-24 / pillar
  n=95): OI ceiling between L=4 and L=5; L=5 multi-seed 0.770-0.790
  just below the 0.80 bar; L=7 0.16 (chance).
- D5 hybrid + pillar n=95: both at L=5 0.790.

D4's substrate is the 5-bridge bio_brain_regions ensemble (V=80
union), so the expected envelope sits BETWEEN the single-bridge V=16
ceiling (PASSes L=7) and the cross-bridge V=160 ceiling (fails L=5)
in capacity terms.

DISCIPLINE: protected/frozen/moat unchanged; no autograd; CPU-only;
bar UNCHANGED at 0.80 multi-seed throughout; reuse-by-import only;
existing probe module byte-unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse-by-import only — no modifications to the existing probe.
from research.findings.raw.direction_4_cross_bridge_probe import (
    run_cross_bridge_probe,
    SEEDS,
)


# The extended loads complementing the production probe's LOADS=[2,3,5].
EXTENDED_LOADS = [4, 6, 7]


def main():
    ap = argparse.ArgumentParser(
        description="Direction 4 extended-load envelope wrapper "
                    "(L={4,6,7}; complements the production L={2,3,5} "
                    "probe; CPU-only; reuse-by-import only)",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="use 'smoke' tag activity caches (reduced scale; numbers "
             "NOT propagated as a result)",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="seeds to probe; default [42, 43, 44]",
    )
    ap.add_argument(
        "--loads", type=int, nargs="+", default=None,
        help="loads to probe; default [4, 6, 7] (the extended set)",
    )
    ap.add_argument(
        "--cache-dir", default=None,
        help="override per-bridge activity cache directory "
             "(defaults to research/findings/raw/direction_4_cache)",
    )
    ap.add_argument(
        "--out", default=None,
        help="output JSON path "
             "(default: direction_4_cross_bridge_production_extended.json)",
    )
    args = ap.parse_args()

    tag = "smoke" if args.smoke else "full"
    loads = list(args.loads) if args.loads is not None else list(EXTENDED_LOADS)
    seeds = list(args.seeds) if args.seeds is not None else list(SEEDS)

    t0 = time.time()
    result = run_cross_bridge_probe(
        seeds=seeds,
        loads=loads,
        tag=tag,
        cache_dir=args.cache_dir,
        verbose=True,
    )
    total = time.time() - t0

    # The result.verdict from the production probe is reported on
    # _DIRECTION_4_LOADS=(2,3,5). For an EXTENDED-LOAD characterisation
    # over {4,6,7}, compute_verdict still applies — but the meaning is
    # "does the substrate PASS the frozen bar at THIS load grid?" rather
    # than the canonical Direction 4 verdict (which uses the frozen
    # {2,3,5}). The result dict labels both for clarity.
    result["framing"] = (
        "DESCRIPTIVE_EXTENDED_LOAD_CHARACTERISATION "
        "(NOT a re-verdict on canonical Direction 4 LOADS={2,3,5}; "
        "pillar n=108 stands)"
    )

    default_out = os.path.join(
        _HERE,
        "direction_4_cross_bridge_production_extended.json"
        if tag == "full"
        else "direction_4_cross_bridge_smoke_extended.json",
    )
    out_path = args.out or default_out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        "\nWrote " + out_path
        + " (wall " + ("%.1f" % total) + "s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
