"""Increment-2 controlled anti-cheat distillation gate.

Trains the SAME Increment-1 student char-net on three corpora, all
identical config / seed, **teacher absent at eval** (only the
previously-cached teacher TEXT is read):

  REAL-baseline : Increment-1 corpus (repo's own English findings prose)
  DISTILLED     : teacher-generated corpus (Kim & Rush 2016 data distill)
  PERMUTED      : char-shuffled distilled corpus (anti-cheat control)

All three sized to the SAME char budget so the comparison isolates
corpus *content*, not size. Reuses cortex_pretraining.train_shakespeare
(DRY — no BPTT reimplementation), same tmpfile pattern as
generator_baseline_smoke.

Gate (falsifiable; failed gate = honest NEGATIVE, never tuned/stubbed):
  - DISTILLED end-loss <= 0.90 * REAL-baseline end-loss (real margin), AND
  - DISTILLED end-loss <  0.90 * PERMUTED end-loss (learned real
    structure, not a size/noise artifact).
PASS => teacher-distilled data gives the student's OWN self-contained
weights a genuine lift. FAIL => data distillation didn't help at PoC
scale (mechanism/path still sound; honest negative).
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from research.runners.local_corpus import load_local_corpus
from research.runners.build_distill_corpus import DISTILL_PATH


def _ascii(text: str, n: int) -> str:
    keep = "".join(c for c in text if 32 <= ord(c) < 127 or c == "\n")
    return keep[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/g11_bg/distill_gate.json")
    a = ap.parse_args()
    from research.runners.cortex_pretraining import train_shakespeare

    distilled_raw = Path(DISTILL_PATH).read_text(encoding="utf-8",
                                                  errors="ignore")
    baseline_raw = load_local_corpus()
    # same char budget for all three -> isolates content, not size
    budget = min(len(_ascii(distilled_raw, 10**9)),
                 len(_ascii(baseline_raw, 10**9)), 110_000)
    distilled = _ascii(distilled_raw, budget)
    baseline = _ascii(baseline_raw, budget)
    rng = np.random.default_rng(a.seed)
    pc = list(distilled)
    rng.shuffle(pc)
    permuted = "".join(pc)
    print(f"[distill-gate] budget={budget} chars each; "
          f"epochs={a.epochs} hidden=[{a.hidden}] seed={a.seed}",
          flush=True)

    def run(text: str, label: str) -> list:
        with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(text)
            p = fh.name
        try:
            res = train_shakespeare(
                seed=a.seed, T=a.seq_len, hidden_layers=[a.hidden],
                epochs=a.epochs, batch_size=32,
                n_train_samples=a.n_samples, corpus_path=p,
                print_every=max(1, a.epochs // 3), verbose=True)
            lh = [float(x) for x in (res["loss_history"]
                                     if isinstance(res, dict) else res)]
            print(f"[{label}] loss {lh[0]:.4f} -> {lh[-1]:.4f}",
                  flush=True)
            return lh
        finally:
            os.unlink(p)

    base_lh = run(baseline, "REAL-baseline")
    dist_lh = run(distilled, "DISTILLED")
    perm_lh = run(permuted, "PERMUTED")
    b1, d1, p1 = base_lh[-1], dist_lh[-1], perm_lh[-1]
    beats_baseline = d1 <= 0.90 * b1
    beats_permuted = d1 < 0.90 * p1
    gate = bool(beats_baseline and beats_permuted)
    summary = {
        "baseline_end": b1, "distilled_end": d1, "permuted_end": p1,
        "distilled_vs_baseline_pct": 100.0 * (b1 - d1) / b1 if b1 else 0,
        "distilled_vs_permuted_pct": 100.0 * (p1 - d1) / p1 if p1 else 0,
        "criterion_beats_baseline_10pct": beats_baseline,
        "criterion_beats_permuted_10pct": beats_permuted,
        "GATE": "PASS" if gate else "FAIL",
        "budget_chars": budget, "epochs": a.epochs,
        "hidden": a.hidden, "seed": a.seed,
    }
    print("\n=== INCREMENT-2 DISTILLATION GATE ===", flush=True)
    print(f"  REAL-baseline end {b1:.4f} | DISTILLED end {d1:.4f} | "
          f"PERMUTED end {p1:.4f}", flush=True)
    print(f"  DISTILLED vs baseline: "
          f"{summary['distilled_vs_baseline_pct']:+.1f}%  | "
          f"vs permuted: {summary['distilled_vs_permuted_pct']:+.1f}%",
          flush=True)
    print(f"  GATE: {summary['GATE']}"
          + ("  (teacher-distilled data lifts the student's OWN "
             "self-contained weights)" if gate
             else "  -- HONEST NEGATIVE (data distillation did not "
             "lift at PoC scale; do not paper over)"), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "baseline_loss_history": base_lh,
               "distilled_loss_history": dist_lh,
               "permuted_loss_history": perm_lh},
              open(a.out, "w"), indent=2)
    print(f"  -> {a.out}", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
