"""Increment-3 capacity-scan gate: honest REAL-vs-PERMUTED verdict.

This orchestrator does NOT train. The controller runs
`research.runners.scaled_generator_train` TWICE to distinct ckpt paths:
  - REAL    : real local-English corpus
  - PERMUTED: same trainer + `--permute-corpus` (chars shuffled; same
              char distribution, sequential structure destroyed)

The honest "capacity scan" question this answers:

  Does a much-larger student beat its own PERMUTED-corpus control by a
  REAL margin (>= 10% lower held-out loss)?

  PASS => the scaled student learns genuine sequential structure, so
          capacity WAS the bottleneck (worth pushing the local path).
  FAIL => even a maxed local char-SNN cannot beat its permuted control,
          so self-contained local fluent generation is out of reach on
          this hardware -- an honest, decision-relevant negative.

The gate bar is FIXED at >= 10% (real_end <= 0.90 * perm_end). It is
NOT softened anywhere.

`verdict()` is a PURE function (no file IO) so it is CPU-unit-tested.
`main()` does the only IO: it READS the two already-produced
checkpoints via the verified `sim.train_checkpoint.load_checkpoint`
(DRY -- no checkpoint logic reimplemented here) and emits the verdict.

Usage:
    python -m research.runners.scaled_capacity_gate \\
        --real-ckpt research/findings/raw/g11_bg/scaled_gen_real.ckpt.npz \\
        --perm-ckpt research/findings/raw/g11_bg/scaled_gen_perm.ckpt.npz \\
        --baseline-end 4.18 \\
        --out research/findings/raw/g11_bg/scaled_capacity_gate.json

Exit codes:
    0 -- both ckpts present, verdict computed + written.
    2 -- at least one ckpt not ready yet (training not done); the
         controller polls on this.
"""
from __future__ import annotations

import argparse

# Fixed gate: REAL held-out loss must be at least this fraction BELOW
# the PERMUTED control's. 0.90 == "real beats permuted by >= 10%".
_GATE_FRACTION = 0.90

_PASS_MSG = ("scaled student learns real structure "
             "-> capacity WAS the bottleneck")
_FAIL_MSG = ("even a maxed local char-SNN does not beat its permuted "
             "control -> self-contained local fluent generation out of "
             "reach on this hardware (honest)")


def verdict(real_loss_hist, perm_loss_hist, baseline_end=None):
    """Pure honest REAL-vs-PERMUTED capacity verdict (no file IO).

    Parameters
    ----------
    real_loss_hist : sequence of float
        Per-epoch loss history of the REAL-corpus run.
    perm_loss_hist : sequence of float
        Per-epoch loss history of the PERMUTED-corpus control run.
    baseline_end : float or None
        Optional Inc-1 tiny-config end loss (~4.18) for context only.
        Does NOT affect the GATE.

    Returns
    -------
    dict
        real_end, perm_end, pct_below_permuted, gate (bool),
        GATE ("PASS"/"FAIL"), gate_message, baseline_end,
        vs_baseline_pct (only when baseline_end is given).
        pct_below_permuted = 100 * (perm_end - real_end) / perm_end
        (positive == real is lower/better than permuted).
        gate = real_end <= 0.90 * perm_end.
    """
    real_end = float(real_loss_hist[-1])
    perm_end = float(perm_loss_hist[-1])

    pct_below_permuted = 100.0 * (perm_end - real_end) / perm_end
    gate = real_end <= _GATE_FRACTION * perm_end

    out = {
        "real_end": real_end,
        "perm_end": perm_end,
        "pct_below_permuted": pct_below_permuted,
        "gate_threshold_fraction": _GATE_FRACTION,
        "gate": bool(gate),
        "GATE": "PASS" if gate else "FAIL",
        "gate_message": _PASS_MSG if gate else _FAIL_MSG,
        "baseline_end": baseline_end,
    }
    if baseline_end is not None:
        be = float(baseline_end)
        # Positive == real_end is below (better than) the tiny baseline.
        out["vs_baseline_pct"] = 100.0 * (be - real_end) / be
    return out


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--real-ckpt", type=str, required=True,
        help="Checkpoint .npz produced by the REAL-corpus trainer run.")
    ap.add_argument(
        "--perm-ckpt", type=str, required=True,
        help="Checkpoint .npz produced by the trainer run with "
             "--permute-corpus (anti-cheat control).")
    ap.add_argument(
        "--baseline-end", type=float, default=4.18,
        help="Inc-1 tiny-config end loss for context only "
             "(does NOT affect the gate). Default 4.18.")
    ap.add_argument(
        "--out", type=str,
        default="research/findings/raw/g11_bg/scaled_capacity_gate.json",
        help="Where to write the verdict JSON.")
    return ap


def main():
    # Lazy imports keep `import research.runners.scaled_capacity_gate`
    # instant (no numpy at module-import time).
    import json

    from sim.train_checkpoint import load_checkpoint

    args = _build_arg_parser().parse_args()

    real_ck = load_checkpoint(args.real_ckpt)
    if real_ck is None:
        print("ckpt not ready: %s" % args.real_ckpt)
        return 2
    perm_ck = load_checkpoint(args.perm_ckpt)
    if perm_ck is None:
        print("ckpt not ready: %s" % args.perm_ckpt)
        return 2

    real_hist = real_ck["loss_history"]
    perm_hist = perm_ck["loss_history"]
    if not real_hist:
        print("ckpt not ready: %s (empty loss_history)" % args.real_ckpt)
        return 2
    if not perm_hist:
        print("ckpt not ready: %s (empty loss_history)" % args.perm_ckpt)
        return 2

    v = verdict(real_hist, perm_hist, baseline_end=args.baseline_end)

    print("=" * 64)
    print("SCALED GENERATOR CAPACITY SCAN -- honest REAL vs PERMUTED")
    print("=" * 64)
    print("  REAL    end loss : %.4f  (epoch %d, %d epochs trained)"
          % (v["real_end"], real_ck["epoch"], len(real_hist)))
    print("  PERMUTED end loss: %.4f  (epoch %d, %d epochs trained)"
          % (v["perm_end"], perm_ck["epoch"], len(perm_hist)))
    print("  REAL is %.2f%% below PERMUTED control (need >= 10.00%%)"
          % v["pct_below_permuted"])
    if v.get("vs_baseline_pct") is not None:
        print("  REAL is %.2f%% below the Inc-1 baseline (%.2f) "
              "[context only]"
              % (v["vs_baseline_pct"], v["baseline_end"]))
    print("-" * 64)
    print("  GATE: %s  -- %s" % (v["GATE"], v["gate_message"]))
    print("=" * 64)

    with open(args.out, "w") as fh:
        json.dump(v, fh, indent=2)
    print("[written] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
