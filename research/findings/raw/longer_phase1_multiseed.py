"""Multi-seed expansion of the longer-Phase-1 training.

Single-seed (42) at 800 events/word produced 15/16 = 93.8% direct
binding accuracy on the 16-word task -- WELL ABOVE the 0.80
trustworthy bar (commit `1a8b384`). This multi-seed expansion trains
seeds 43 and 44 at the same 800ev recipe so multi-seed direct
binding can be validated.

PROTOCOL:
1. Train seeds 43 and 44 sequentially at 800ev (~138 min each;
   ~276 min total).
2. Save each checkpoint to `research/findings/raw/unified_per_regime/phase1_800ev/`.
3. Caller will then run the 16-word direct-binding diagnostic on all
   3 seeds (already cached: seed 42 + new seeds 43 + 44).
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the training function from the seed-42 diagnostic.
from importlib import util as _import_util
_diag_path = os.path.join(_HERE, "longer_phase1_diagnostic.py")
_spec = _import_util.spec_from_file_location("_long", _diag_path)
_long = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_long)
train_longer_phase1 = _long.train_longer_phase1


SEEDS = [43, 44]
EVENTS_PER_WORD = 800
OUT_CACHE_DIR = "research/findings/raw/unified_per_regime/phase1_800ev"


def main():
    overall_start = time.time()
    for seed in SEEDS:
        print(f"\n=== Training seed {seed} at {EVENTS_PER_WORD} events/word ===")
        t_seed = time.time()
        cache_path = train_longer_phase1(seed, EVENTS_PER_WORD, OUT_CACHE_DIR)
        elapsed = (time.time() - t_seed) / 60.0
        print(f"  seed {seed} done; {elapsed:.1f} min; saved {cache_path}")

    overall_elapsed = (time.time() - overall_start) / 60.0
    print(f"\nMulti-seed training complete; {overall_elapsed:.1f} min total")
    print(
        f"Cached 800ev checkpoints now available at "
        f"{OUT_CACHE_DIR} for seeds 42 (existing), 43, 44."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
