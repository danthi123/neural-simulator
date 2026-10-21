#!/usr/bin/env python3
"""Entry gate for step-0b Part B (PACKET.md, RULING bg-tonic-loop-partA-1020b).

A0 must reproduce the dev refile's persisted ``tonic_rates_hz`` on all 96 bg_neural
commit records EXACTLY, or the step-0 read STOPs before anything else is examined.

Records live in the three seam arm artifacts (seam_b / seam_readout /
seam_lesion_value), under ``events`` -> <event kind> -> [ {.., tonic_rates_hz, ..} ].
The refile baseline: seeds 10-16 at ``<refile-root>/s<seed>/s<seed>/``; seed 9's
canary artifacts at ``<s9-canary-root>/s9/s9/`` (the refile root's s9 dir is empty).

Usage:
    python -m research.runners._bg_loop_entry_gate \
        --partb-root research/findings/raw/_bg_live_seam/loop_b/<tip> \
        [--refile-root research/findings/raw/_bg_live_seam/tonic_dev/8177576] \
        [--s9-canary-root research/findings/raw/_pool_canary/<canary>/_bg_live_seam/tonic_dev/8177576] \
        [--self-test]         # compares the refile against itself; must PASS 96/96

Exit codes: 0 = gate PASS (96/96 exact), 1 = STOP (missing or mismatched records),
2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from tools.lab import attributable_to  # noqa: E402  (attribution for the gate's pair)

SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
SEAM_ARMS = ("seam_b", "seam_readout", "seam_lesion_value")
DEFAULT_REFILE = ("research/findings/raw/_bg_live_seam/tonic_dev/8177576")
DEFAULT_S9 = ("research/findings/raw/_pool_canary/"
              "host_share_bg_seam_8177576ae_pool40_20261019T133006/"
              "_bg_live_seam/tonic_dev/8177576")
EXPECTED_RECORDS = 96  # 8 seeds x 3 seam arms x 4 event kinds


def _record_path(root: str, seed: int, seam_arm: str) -> str:
    return os.path.join(root, "s%d" % seed, "s%d" % seed, seam_arm + ".json")


def _tonic_by_event(path: str) -> dict:
    """event kind -> tonic_rates_hz for one seam arm artifact."""
    with open(path) as fh:
        art = json.load(fh)
    out = {}
    for kind, entries in (art.get("events") or {}).items():
        if not isinstance(entries, list):
            continue
        for ent in entries:
            if isinstance(ent, dict) and ent.get("tonic_rates_hz") is not None:
                out[kind] = ent["tonic_rates_hz"]
                break
    return out


def _baseline_path(refile_root: str, s9_root: str, seed: int, seam_arm: str) -> str:
    root = s9_root if seed == 9 else refile_root
    return _record_path(root, seed, seam_arm)


def run_gate(partb_root: str, refile_root: str, s9_root: str,
             s9_for_partb: bool = False) -> int:
    compared = 0
    missing = []
    mismatched = []
    for seed in SEEDS:
        for seam_arm in SEAM_ARMS:
            base_path = _baseline_path(refile_root, s9_root, seed, seam_arm)
            if s9_for_partb and seed == 9:
                cand_path = _baseline_path(refile_root, s9_root, seed, seam_arm)
            else:
                cand_path = _record_path(partb_root, seed, seam_arm)
            try:
                base = _tonic_by_event(base_path)
            except (OSError, ValueError):
                missing.append(("baseline", base_path))
                continue
            try:
                cand = _tonic_by_event(cand_path)
            except (OSError, ValueError):
                missing.append(("partb-a0", cand_path))
                continue
            if set(base) != set(cand):
                mismatched.append((seed, seam_arm, "event kinds",
                                   sorted(base), sorted(cand)))
                continue
            for kind in sorted(base):
                compared += 1
                if base[kind] != cand[kind]:
                    mismatched.append((seed, seam_arm, kind,
                                       base[kind], cand[kind]))
    print("entry gate: compared=%d/%d missing=%d mismatched=%d"
          % (compared, EXPECTED_RECORDS, len(missing), len(mismatched)))
    for tag, path in missing:
        print("  MISSING[%s] %s" % (tag, path))
    for row in mismatched[:20]:
        print("  MISMATCH seed=%s arm=%s kind=%s\n    baseline=%r\n    partb   =%r"
              % row)
    if len(mismatched) > 20:
        print("  ... %d more mismatches" % (len(mismatched) - 20))
    if compared == EXPECTED_RECORDS and not missing and not mismatched:
        print("gate PASS: A0 reproduces the refile exactly on all 96 records")
        return 0
    # Attribution stamp: this gate computes a baseline/candidate pair, so state whose
    # numbers the comparison rests on (tools.lab attribution is mandatory for pairs).
    n_clean = compared - len(mismatched)
    n_bad = EXPECTED_RECORDS - n_clean
    attributable_to("entry-gate exact-match records vs failures",
                    n_clean, n_bad)
    print("gate STOP: Part B is not readable until A0 matches the refile")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partb-root", default=None)
    ap.add_argument("--refile-root", default=DEFAULT_REFILE)
    ap.add_argument("--s9-canary-root", default=DEFAULT_S9)
    ap.add_argument("--self-test", action="store_true",
                    help="compare the refile against itself (expect PASS 96/96)")
    args = ap.parse_args(argv)
    try:
        if args.self_test:
            return run_gate(args.refile_root, args.refile_root,
                            args.s9_canary_root, s9_for_partb=True)
        if not args.partb_root:
            print("--partb-root required unless --self-test", file=sys.stderr)
            return 2
        return run_gate(args.partb_root, args.refile_root, args.s9_canary_root)
    except OSError as exc:
        print("entry gate IO error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
