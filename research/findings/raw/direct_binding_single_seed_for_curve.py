"""Direct binding 16-word diagnostic on a single cache (driver for curve probe).

Reuses test_one_checkpoint from direct_binding_phase1_comparison.py
byte-unchanged. Single seed, single cache dir, prints n_correct / 16.

Used by Direction B Probe-2 + future probes to measure direct-binding
capability at an arbitrary Phase-1 training-event count.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from importlib import util as _import_util
_diag_path = os.path.join(_HERE, "direct_binding_phase1_comparison.py")
_spec = _import_util.spec_from_file_location("_db", _diag_path)
_db = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_db)
test_one_checkpoint = _db.test_one_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    label = args.label or f"seed{args.seed} {args.cache_dir}"
    result = test_one_checkpoint(args.seed, args.cache_dir, label)

    print(f"\n=== DIRECT BINDING RESULT ===")
    print(f"  seed: {args.seed}")
    print(f"  cache: {args.cache_dir}")
    print(f"  n_correct/n_total: {result['n_correct']}/{result['n_total']}")
    print(f"  accuracy: {100.0*result['accuracy']:.1f}%")

    bar = 0.80
    print(f"  bar {bar:.2f}: {'PASS' if result['accuracy'] >= bar else 'FAIL'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "seed": args.seed,
                "cache_dir": args.cache_dir,
                "label": label,
                "n_correct": result["n_correct"],
                "n_total": result["n_total"],
                "accuracy": result["accuracy"],
                "per_word": result["per_word"],
                "bar_0p80_pass": bool(result["accuracy"] >= bar),
            }, f, indent=2)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
