"""Sweep eval-time parameters on a saved bridge checkpoint.

Runs text_reeval.py with multiple combinations of (drive_pA,
n_reset_steps), saves each result to a separate JSON, and produces
a summary table.

This isolates eval-methodology effects from training effects: same
trained network, different readout. Cheap (~3min per re-eval) compared
to full training (~75min).

Usage:
  python -m research.runners.text_reeval_sweep \\
      research/findings/raw/g11_bg/text_eval_R3R6_v2.simstate.h5 \\
      --output-dir research/findings/raw/g11_bg/sweep_v2/ \\
      --drives 200 300 400 500 \\
      --resets 100 200 400

Output:
- One JSON per (drive, reset) combo: {output_dir}/sweep_d{D}_r{R}.json
- Summary CSV: {output_dir}/summary.csv
- Summary printed to stdout
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str, help="path to .simstate.h5")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--drives", type=float, nargs="+",
                    default=[200.0, 300.0, 400.0, 500.0])
    ap.add_argument("--resets", type=int, nargs="+",
                    default=[100, 200, 400])
    ap.add_argument("--n-eval-image-word", type=int, default=100)
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1],
                    help="eval-rng seeds; each combo runs once per seed")
    ap.add_argument("--include-legacy-block", action="store_true",
                    help="also run a legacy block-ordered eval at default "
                    "drive/reset, for comparison")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        ap.error(f"checkpoint not found: {ckpt}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # collected (drive, reset, seed, label, iw_acc, wa_acc)
    t_start = time.time()
    n_combos = len(args.drives) * len(args.resets) * len(args.seeds)
    if args.include_legacy_block:
        n_combos += 1
    done = 0

    for seed in args.seeds:
        for drive in args.drives:
            for reset in args.resets:
                done += 1
                tag = f"d{int(drive)}_r{reset}_s{seed}"
                out_file = out_dir / f"sweep_{tag}.json"
                print(f"\n[{done}/{n_combos}] drive={drive} reset={reset} "
                      f"seed={seed} -> {out_file.name}", flush=True)
                cmd = [
                    sys.executable, "-m", "research.runners.text_reeval",
                    str(ckpt),
                    "--drive-pA", str(drive),
                    "--n-reset-steps", str(reset),
                    "--n-eval-image-word", str(args.n_eval_image_word),
                    "--n-eval-word-action", str(args.n_eval_word_action),
                    "--seed", str(seed),
                    "--out-stats", str(out_file),
                ]
                rc = subprocess.run(cmd, check=False)
                if rc.returncode != 0:
                    print(f"  WARN: returncode {rc.returncode}", flush=True)
                    rows.append({
                        "drive_pA": drive, "n_reset_steps": reset,
                        "seed": seed, "label": tag, "interleave": True,
                        "iw_correct": None, "iw_n": None, "iw_acc": None,
                        "wa_correct": None, "wa_n": None, "wa_acc": None,
                    })
                    continue
                d = json.loads(out_file.read_text())
                iw = d["image_to_word_eval"]
                wa = d["word_to_action_eval"]
                rows.append({
                    "drive_pA": drive, "n_reset_steps": reset,
                    "seed": seed, "label": tag, "interleave": True,
                    "iw_correct": iw["correct"], "iw_n": iw["n_trials"],
                    "iw_acc": iw["accuracy"],
                    "wa_correct": wa["correct"], "wa_n": wa["n_trials"],
                    "wa_acc": wa["accuracy"],
                })
                print(f"  IW={iw['correct']}/{iw['n_trials']}={iw['accuracy']:.1%} "
                      f"WA={wa['correct']}/{wa['n_trials']}={wa['accuracy']:.1%}",
                      flush=True)

    # Optional legacy comparison at default drive/reset
    if args.include_legacy_block:
        done += 1
        tag = f"LEGACY_BLOCK_d200_r100_s1"
        out_file = out_dir / f"sweep_{tag}.json"
        print(f"\n[{done}/{n_combos}] LEGACY_BLOCK d=200 r=100 s=1 -> "
              f"{out_file.name}", flush=True)
        cmd = [
            sys.executable, "-m", "research.runners.text_reeval",
            str(ckpt),
            "--drive-pA", "200",
            "--n-reset-steps", "100",
            "--n-eval-image-word", str(args.n_eval_image_word),
            "--n-eval-word-action", str(args.n_eval_word_action),
            "--seed", "1",
            "--legacy-block-eval",
            "--out-stats", str(out_file),
        ]
        rc = subprocess.run(cmd, check=False)
        if rc.returncode == 0:
            d = json.loads(out_file.read_text())
            iw = d["image_to_word_eval"]
            wa = d["word_to_action_eval"]
            rows.append({
                "drive_pA": 200.0, "n_reset_steps": 100,
                "seed": 1, "label": tag, "interleave": False,
                "iw_correct": iw["correct"], "iw_n": iw["n_trials"],
                "iw_acc": iw["accuracy"],
                "wa_correct": wa["correct"], "wa_n": wa["n_trials"],
                "wa_acc": wa["accuracy"],
            })
            print(f"  IW={iw['correct']}/{iw['n_trials']}={iw['accuracy']:.1%} "
                  f"WA={wa['correct']}/{wa['n_trials']}={wa['accuracy']:.1%}",
                  flush=True)

    # Summary CSV
    summary_csv = out_dir / "summary.csv"
    fieldnames = ["drive_pA", "n_reset_steps", "seed", "label", "interleave",
                  "iw_correct", "iw_n", "iw_acc",
                  "wa_correct", "wa_n", "wa_acc"]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSweep complete in {time.time() - t_start:.0f}s.")
    print(f"Summary CSV: {summary_csv}")

    # Pretty-printed table sorted by best max(iw, wa) accuracy
    print("\n" + "=" * 78)
    print(f"  {'tag':<30}  {'IW':<10}  {'WA':<10}  {'best':<8}")
    print("-" * 78)
    sorted_rows = sorted(
        [r for r in rows if r["iw_acc"] is not None],
        key=lambda r: max(r["iw_acc"] or 0, r["wa_acc"] or 0),
        reverse=True,
    )
    for r in sorted_rows:
        iw_str = f"{r['iw_correct']}/{r['iw_n']}={r['iw_acc']:.1%}"
        wa_str = f"{r['wa_correct']}/{r['wa_n']}={r['wa_acc']:.1%}"
        best = max(r["iw_acc"], r["wa_acc"])
        print(f"  {r['label']:<30}  {iw_str:<10}  {wa_str:<10}  {best:.1%}")
    print("=" * 78)


if __name__ == "__main__":
    main()
