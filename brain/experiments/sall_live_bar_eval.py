# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-03-sall-live-bar-refit.md (pre-registration this implements)
"""Run the UNCHANGED sall_live_probe with Brain.KEY_BARS["S_all"] set to a refit pair for this process only.

  python -m brain.experiments.sall_live_bar_eval --familiarity F --margin M -- --home <dir> --seed 100 --out <json>
"""
import argparse
import sys

from brain.brain import Brain
from brain.experiments import sall_live_probe

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--familiarity", type=float, required=True)
    ap.add_argument("--margin", type=float, required=True)
    a, rest = ap.parse_known_args()
    Brain.KEY_BARS["S_all"] = (a.familiarity, a.margin)
    sys.argv = [sys.argv[0]] + [x for x in rest if x != "--"]
    sall_live_probe.main()
