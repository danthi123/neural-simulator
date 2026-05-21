"""Phase-1 training-event curve diagnostic.

The 8-arc convergent ceiling rests on the claim that 200 events/word
is the LOCAL OPTIMUM for compositional retrieval at N=3 (the 6th arc:
3-seed mean 0.458; seed-42 0.571). Tested data points on the same
substrate so far:

| Phase-1 ev/word | N=3 full_acc (seed 42) | Source |
|-----------------|------------------------|--------|
| 200ev           | 0.571                  | 6th arc decisive |
| 800ev           | 0.143                  | longer-Phase-1 (commit 1926cfe) |

We have NEVER tested SHORTER Phase-1. Critical-period biology (CLS
schema-vs-binding tradeoff; McClelland 2013) predicts that GENTLER
training preserves compositional flexibility further. The cheap-first
probe is at 100ev (single seed 42) to test the hypothesis that
shorter helps compositional. Decision rule (pre-registered before
the run):

- If 100ev N=3 full_acc >= 0.571: shorter Phase-1 may be a better
  sweet-spot; expand multi-seed (seeds 43/44 at 100ev) for honest
  validation.
- If 100ev N=3 full_acc strictly < 0.571: the 6th arc 200ev sweet-
  spot is empirically confirmed below AS WELL AS above, strengthening
  the 8-arc convergent ceiling claim.

REUSE: this script reuses `train_longer_phase1` from
`longer_phase1_diagnostic.py` (byte-unchanged); only overrides the
`n_train_events` argument and the output cache directory.

PROTOCOL:
1. Train seed 42 at 100 events/word into a new cache dir
   `research/findings/raw/unified_per_regime/phase1_100ev/`.
2. Caller then invokes the 6th arc runner pointing at that cache dir:

       python -m research.runners.generative_replay_pfc_frame_runner \
           --seeds 42 --loads 3 \
           --phase1-cache-dir research/findings/raw/unified_per_regime/phase1_100ev \
           --ckpt research/findings/raw/phase1_100ev_decisive.ckpt \
           --out research/findings/raw/phase1_100ev_decisive.json
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the existing training helper byte-unchanged
from importlib import util as _import_util
_diag_path = os.path.join(_HERE, "longer_phase1_diagnostic.py")
_spec = _import_util.spec_from_file_location("_long", _diag_path)
_long = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_long)
train_longer_phase1 = _long.train_longer_phase1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Single seed for the cheap-first probe (default 42)."
    )
    parser.add_argument(
        "--events-per-word", type=int, default=100,
        help="Events per word for Phase-1 training (default 100; "
             "below the 200ev 6th arc sweet-spot)."
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Output cache dir (default phase1_{events-per-word}ev)."
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = (
            f"research/findings/raw/unified_per_regime/"
            f"phase1_{args.events_per_word}ev"
        )

    cache_path = train_longer_phase1(args.seed, args.events_per_word, cache_dir)
    print(f"\nCheckpoint ready: {cache_path}")
    print(
        f"\nNext step: invoke the 6th arc decisive eval pointing at this cache:\n"
        f"  python -m research.runners.generative_replay_pfc_frame_runner \\\n"
        f"      --seeds {args.seed} --loads 3 \\\n"
        f"      --phase1-cache-dir {cache_dir} \\\n"
        f"      --ckpt research/findings/raw/phase1_{args.events_per_word}ev_decisive.ckpt \\\n"
        f"      --out research/findings/raw/phase1_{args.events_per_word}ev_decisive.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
