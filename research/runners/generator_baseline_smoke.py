"""Increment-1 foundation gate: the sim's OWN ported Phase-2 generator
must demonstrably learn REAL local text on `main`, beating a
permuted-character control (anti-cheat: real sequential structure,
not noise-fitting / memorization artifact).

Reuses the validated `cortex_pretraining.train_shakespeare` trainer
(DRY — no BPTT reimplementation). Zero network: corpus is the repo's
own English via `local_corpus`, written to a temp file the existing
loader reads. No LLM, no templates.

Gate (falsifiable):
  - REAL: final loss <= 0.7 * initial loss (substantial reduction), AND
  - REAL final loss < PERMUTED final loss by a clear margin
    (>=10% lower) -> learned real structure, not just fitting noise.
A failed gate is an HONEST finding (port/infra regressed), not papered.
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from research.runners.local_corpus import load_local_corpus


def _ascii_slice(text: str, n_chars: int) -> str:
    """Printable-ASCII filter (Shakespeare-like ~<100 vocab, fast) +
    deterministic length cap for a quick smoke."""
    keep = "".join(c for c in text if 32 <= ord(c) < 127 or c == "\n")
    return keep[:n_chars]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chars", type=int, default=180_000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    from research.runners.cortex_pretraining import train_shakespeare

    real = _ascii_slice(load_local_corpus(), a.n_chars)
    rng = np.random.default_rng(a.seed)
    perm_chars = list(real)
    rng.shuffle(perm_chars)             # destroy sequential structure
    perm = "".join(perm_chars)
    print(f"[smoke] corpus {len(real):,} chars (ascii); "
          f"vocab~{len(set(real))}; epochs={a.epochs} "
          f"hidden=[{a.hidden}]", flush=True)

    def run(text: str, label: str):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(text)
            path = fh.name
        try:
            res = train_shakespeare(
                seed=a.seed, T=a.seq_len, hidden_layers=[a.hidden],
                epochs=a.epochs, batch_size=32,
                n_train_samples=a.n_samples, corpus_path=path,
                print_every=max(1, a.epochs // 3), verbose=True,
            )
            lh = res["loss_history"] if isinstance(res, dict) else res
            lh = [float(x) for x in lh]
            print(f"[{label}] loss {lh[0]:.4f} -> {lh[-1]:.4f}",
                  flush=True)
            return lh
        finally:
            os.unlink(path)

    real_lh = run(real, "REAL")
    perm_lh = run(perm, "PERMUTED")

    r0, r1 = real_lh[0], real_lh[-1]
    p1 = perm_lh[-1]
    reduced = r1 <= 0.7 * r0
    beats_perm = r1 < 0.9 * p1
    gate = bool(reduced and beats_perm)
    summary = {
        "real_loss_start": r0, "real_loss_end": r1,
        "real_reduction_pct": 100.0 * (r0 - r1) / r0 if r0 else 0.0,
        "permuted_loss_end": p1,
        "real_below_permuted_pct": 100.0 * (p1 - r1) / p1 if p1 else 0.0,
        "criterion_real_reduced_30pct": reduced,
        "criterion_real_beats_permuted_10pct": beats_perm,
        "GATE": "PASS" if gate else "FAIL",
        "epochs": a.epochs, "hidden": a.hidden, "seed": a.seed,
    }
    print("\n=== INCREMENT-1 FOUNDATION GATE ===", flush=True)
    print(f"  REAL {r0:.4f}->{r1:.4f} "
          f"({summary['real_reduction_pct']:.1f}% reduction)", flush=True)
    print(f"  PERMUTED end {p1:.4f}; REAL is "
          f"{summary['real_below_permuted_pct']:.1f}% below PERMUTED",
          flush=True)
    print(f"  GATE: {summary['GATE']} "
          f"(real learns REAL structure on main, anti-cheat-controlled)"
          if gate else
          f"  GATE: {summary['GATE']} — HONEST NEGATIVE (do not paper over)",
          flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": summary,
                    "real_loss_history": real_lh,
                    "permuted_loss_history": perm_lh},
                   open(a.out, "w"), indent=2)
        print(f"  -> {a.out}", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
