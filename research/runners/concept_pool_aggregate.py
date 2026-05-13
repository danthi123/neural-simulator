"""concept_pool_aggregate — multi-seed analysis for concept_pool_demo.

Reads seed42.json ... seed46.json (or any subset) and produces:
- Mean / std / range across seeds for n_pass / 10
- Per-word PASS rate across seeds (which words are robust vs fragile?)
- Per-seed verdict (GO if >= 8/10, PARTIAL 5-7, FAIL <5)
- Multi-seed verdict (GO if >= 3 seeds PASS)

Output: JSON summary + console table.

Use after launch_multiseed.ps1 completes, or any time you have multiple
seed JSONs in the same directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def aggregate(json_paths):
    """Aggregate per-seed results."""
    results = []
    per_word_pass = {}
    for path in json_paths:
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:
            print(f"[WARN] could not parse {path}: {e}")
            continue
        seed = data["seed"]
        n_pass = data["n_pass"]
        n_words = data["n_words"]
        wc = data["wall_clock_s"]
        verdict = (
            "GO" if n_pass >= 8
            else "PARTIAL" if n_pass >= 5
            else "FAIL"
        )
        results.append({
            "seed": seed, "n_pass": n_pass, "n_words": n_words,
            "wall_clock_s": wc, "verdict": verdict,
        })
        # Per-word PASS tracking
        for word, res in data.get("results", {}).items():
            if word not in per_word_pass:
                per_word_pass[word] = []
            target = res["target_rate"]
            max_off = res["max_off_target"]
            passed = bool(target > max_off)
            per_word_pass[word].append(passed)
    return results, per_word_pass


def format_table(results, per_word_pass):
    out = []
    out.append("=== Multi-seed aggregation ===\n")
    out.append(f"{'seed':>6s} {'n_pass':>8s} {'wall_clock_min':>14s} {'verdict':>10s}")
    out.append("-" * 45)
    for r in results:
        out.append(
            f"{r['seed']:>6d} {r['n_pass']:>3d}/{r['n_words']:<3d} "
            f"{r['wall_clock_s']/60:>14.1f} {r['verdict']:>10s}"
        )
    out.append("-" * 45)
    if results:
        passes = [r["n_pass"] for r in results]
        out.append(f"{'mean':>6s} {sum(passes)/len(passes):>3.1f}/{results[0]['n_words']:<3d}")
        if len(passes) > 1:
            out.append(f"{'std':>6s} {stdev(passes):>3.2f}")
        out.append(f"{'range':>6s} {min(passes)}-{max(passes)}/{results[0]['n_words']}")
        n_go = sum(1 for r in results if r["verdict"] == "GO")
        n_partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
        n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
        out.append(f"\nSeeds: {n_go} GO + {n_partial} PARTIAL + {n_fail} FAIL")

        multi_verdict = (
            "MULTI-SEED GO" if n_go >= 3
            else "MULTI-SEED PARTIAL" if (n_go + n_partial) >= 3
            else "MULTI-SEED FAIL"
        )
        out.append(f"\n>>> {multi_verdict} <<<\n")

    if per_word_pass:
        out.append("\nPer-word PASS rate across seeds:")
        for word, passes in per_word_pass.items():
            rate = sum(passes) / len(passes)
            marker = "robust" if rate >= 0.8 else "fragile" if rate <= 0.4 else "mixed"
            out.append(
                f"  {word:10s}: {sum(passes)}/{len(passes)} "
                f"({rate * 100:.0f}%) [{marker}]"
            )
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed concept_pool_demo results.")
    parser.add_argument("--dir", type=str,
                         default="research/findings/raw/g11_bg/concept_pool_demo",
                         help="Directory containing seed*.json files")
    parser.add_argument("--pattern", type=str, default="seed*.json",
                         help="Glob pattern (default 'seed*.json'). "
                         "For v7-only: 'seed*_v7.json'.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output JSON path for aggregated summary")
    args = parser.parse_args()

    json_files = sorted(Path(args.dir).glob(args.pattern))
    if not json_files:
        print(f"No files matching '{args.pattern}' in {args.dir}")
        return 1

    print(f"Aggregating {len(json_files)} seed results from {args.dir}\n")
    results, per_word_pass = aggregate(json_files)
    print(format_table(results, per_word_pass))

    if args.out:
        out_path = Path(args.out)
        summary = {
            "n_seeds": len(results),
            "per_seed": results,
            "per_word_pass_count": {
                w: sum(passes) for w, passes in per_word_pass.items()
            },
            "per_word_pass_rate": {
                w: sum(passes) / len(passes)
                for w, passes in per_word_pass.items()
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
