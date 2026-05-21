"""Per-word attractor analysis (Direction I).

Reads existing per-word JSON outputs from the 5000-step silent-interval
runs at seeds 43 and 44 at 800ev. Identifies which specific words
showed accuracy GAINS (seed 43: trajectory peaked at -15.4% gain) and
which words showed accuracy LOSSES (seed 44: peaked at +15.4% loss).
Cross-references to see if the attractor-sensitive vocabulary is
SHARED across the conjugate seeds (same words swing) or DISJOINT
(different words swing).

Pure analysis; no GPU; reuse-only.
"""
from __future__ import annotations

import json
import os


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_pre_post_per_word(payload, ev=800):
    for ev_result in payload["per_ev_results"]:
        if ev_result["ev_per_word"] != ev:
            continue
        # We need the per-word results from the diagnostic's per_word
        # field. The silent_interval_persistence_probe stores per-ev
        # entries; each entry has pre_n_correct + post_n_correct but
        # not per-word. We need to re-extract from the diagnostic
        # output (test_one_checkpoint result is not stored verbosely
        # in the persistence JSON).
        return ev_result
    return None


def main():
    # Read seed-43 5000ev JSON
    seed43_5k = _load(
        "research/findings/raw/silent_interval_length_sweep_seed43_800ev_5000.json"
    )
    seed44_5k = _load(
        "research/findings/raw/silent_interval_length_sweep_seed44_800ev_5000.json"
    )

    s43 = _extract_pre_post_per_word(seed43_5k, ev=800)
    s44 = _extract_pre_post_per_word(seed44_5k, ev=800)

    print("=== PER-EV SUMMARY (5000-step silent interval; 800ev) ===")
    print(f"  Seed 43: pre={s43['pre_n_correct']}/16 -> post={s43['post_n_correct']}/16 "
          f"(forgetting={s43['forgetting_pct']:.1f}%)")
    print(f"  Seed 44: pre={s44['pre_n_correct']}/16 -> post={s44['post_n_correct']}/16 "
          f"(forgetting={s44['forgetting_pct']:.1f}%)")

    # The persistence probe DOES NOT store per-word details in its JSON.
    # The per-word details are only printed to the log file.
    # Need to parse the log files instead.
    print("\n=== Per-word output not in JSON; need to parse from log file. ===")
    print("=== Manual extraction from logs follows. ===\n")

    return 0


if __name__ == "__main__":
    main()
