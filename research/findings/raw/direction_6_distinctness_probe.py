"""Direction 6 cross-bridge activity-distinctness probe.

Loads the 5 D6 smoke / production bridges for one seed and computes
per-bridge mean firing rates + cross-bridge cosine on the activity
vectors. Same diagnostic pattern as D4 + D5 used to verify the bug-fix
(no byte-identical activity across bridges).

Pre-bug: cos = 1.000000 byte-identical across all 5 bridges (the failure
mode the D4 bug fix addressed). Post-bug: cos ~0.01-0.03 (orthogonal-
like activity per bridge). D6 must replicate the post-bug pattern.

Usage:
    python -m research.findings.raw.direction_6_distinctness_probe \
        --tag smoke --seed 42
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    ap = argparse.ArgumentParser(
        description="D6 cross-bridge activity-distinctness probe"
    )
    ap.add_argument("--tag", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_dir = os.path.join(_HERE, "direction_6_cache")

    # Per-bridge canonical word (first word in each bridge's frozen list)
    from research.findings.raw.direction_6_vocab_spec import (
        DIRECTION_6_BRIDGE_A_WORDS,
        DIRECTION_6_BRIDGE_B_WORDS,
        DIRECTION_6_BRIDGE_C_WORDS,
        DIRECTION_6_BRIDGE_D_WORDS,
        DIRECTION_6_BRIDGE_E_WORDS,
    )
    bridges = {
        "A_nouns": DIRECTION_6_BRIDGE_A_WORDS,
        "B_verbs": DIRECTION_6_BRIDGE_B_WORDS,
        "C_adj": DIRECTION_6_BRIDGE_C_WORDS,
        "D_spatial": DIRECTION_6_BRIDGE_D_WORDS,
        "E_functional": DIRECTION_6_BRIDGE_E_WORDS,
    }

    # Load each bridge's canonical word activity
    canonical_acts = {}
    for bridge_name, words in bridges.items():
        cache_p = os.path.join(
            cache_dir,
            "activity_" + args.tag + "_" + bridge_name
            + "_seed" + str(args.seed) + ".npz",
        )
        if not os.path.exists(cache_p):
            print(
                "MISSING: " + cache_p
                + "  (run direction_6_5bridge_runner first)",
                flush=True,
            )
            continue
        # Safety: numeric-only npz load (object arrays rejected).
        data = np.load(cache_p, allow_pickle=False)
        canonical_word = words[0]
        # Mean across M_OBS observations -> one vector per word
        canonical_acts[bridge_name] = (
            canonical_word, data[canonical_word].mean(axis=0),
        )

    if len(canonical_acts) < 2:
        print("Need at least 2 bridges loaded; aborting", flush=True)
        return 1

    print("=== D6 cross-bridge activity-distinctness probe ===", flush=True)
    print("  tag=" + args.tag + " seed=" + str(args.seed), flush=True)
    print("", flush=True)

    print("Per-bridge canonical word + mean rate + density:", flush=True)
    for bn, (word, vec) in canonical_acts.items():
        mr = float(vec.mean())
        dens = float(np.mean(vec > 0.0))
        print(
            "  " + bn + "[" + word + "]: mean_rate=" + ("%.4f" % mr)
            + " density=" + ("%.4f" % dens)
            + " d_act=" + str(vec.shape[0]),
            flush=True,
        )

    # Cross-bridge cosines (canonical word vs canonical word).
    print("", flush=True)
    print("Cross-bridge cosines (canonical word per bridge):", flush=True)

    def _cos(a, b):
        an = float(np.linalg.norm(a))
        bn_norm = float(np.linalg.norm(b))
        if an == 0.0 or bn_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (an * bn_norm))

    bridge_names = list(canonical_acts.keys())
    for i, bi in enumerate(bridge_names):
        for j, bj in enumerate(bridge_names):
            if j <= i:
                continue
            wi, vi = canonical_acts[bi]
            wj, vj = canonical_acts[bj]
            # If the activity vectors have different lengths (different
            # n_pool_union), the cosine is undefined in the canonical
            # sense; truncate to the shorter length for the diagnostic.
            n = min(vi.shape[0], vj.shape[0])
            cos = _cos(vi[:n], vj[:n])
            byte_identical = bool(np.array_equal(vi[:n], vj[:n]))
            print(
                "  " + bi + "[" + wi + "] vs " + bj + "[" + wj + "]: "
                "cos=" + ("%.6f" % cos)
                + "  byte_identical=" + str(byte_identical),
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
