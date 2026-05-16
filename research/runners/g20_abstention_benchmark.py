"""Hallucination-resistance / abstention benchmark for a sparse G.20
ensemble.

Prior benchmarks measured retrieval of *encoded* associations. The
single most important property for a trustworthy conversational system
is the opposite: when there is NO answer, does it correctly abstain,
or confabulate a spurious high-confidence associate?

Method: split N concepts into ENCODED (a real cross-bridge `A is B`
is created) and CONTROL (nothing encoded -- only the per-concept
training tag exists). Query every concept's top associate confidence
(the firing rate of the #1 non-self associate). If the system is
trustworthy there is a CONFIDENCE GAP: encoded queries return a high
rate (real associate fired), control queries return only the noise
floor -> a threshold can cleanly separate "I know" from "I don't
know". If the distributions overlap, the system hallucinates
(can't tell knowing from not-knowing).

Reports separability: % of ENCODED whose top-rate exceeds the MAX
control top-rate (clean-separation rate), plus the rate
distributions. This is the permuted-label-control discipline applied
to abstention.

Reuses validated SharedPoolMember + _query_top (DRY).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from research.runners.g20_multibridge import (
    SharedPoolMember, read_vocab_file,
)
from research.runners.g20_xbridge_benchmark import (
    _query_top, sample_xbridge_pairs,
)


def split_encoded_control(pairs: List[Tuple[int, str, int, str]],
                           seed: int
                           ) -> Tuple[list, list]:
    """Half the sampled cross-bridge pairs become ENCODED (we will
    `remember A is B`); the other half's A-words are CONTROL probes
    (queried but never associated). Deterministic. Pure."""
    rng = np.random.RandomState(seed * 71 + 3)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    half = len(idx) // 2
    enc = [pairs[i] for i in idx[:half]]
    ctrl = [pairs[i] for i in idx[half:]]
    return enc, ctrl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridges", nargs="+", required=True)
    p.add_argument("--vocab-files", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=8192)
    p.add_argument("--n-shared-pool", type=int, default=2000)
    p.add_argument("--sparsity", type=float, default=0.007)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--sparse", action="store_true")
    p.add_argument("--n-pairs", type=int, default=40,
                    help="sampled; ~half encoded, ~half control")
    p.add_argument("--exclude-idx", type=int, default=12)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    members = []
    for bp, vp, nm in zip(args.bridges, args.vocab_files, args.names):
        members.append(SharedPoolMember(
            bridge_path=bp, vocab=read_vocab_file(vp), name=nm,
            n_lang_input=args.n_lang_input,
            n_shared_pool=args.n_shared_pool, sparsity=args.sparsity,
            sparse=args.sparse, pattern_size=args.pattern_size))
    print(f"=== abstention benchmark: {args.n_pairs} probes, "
          f"seed {args.seed} ===", flush=True)
    for m in members:
        m.load(args.seed)
    by_name = {m.name: m for m in members}

    exclude = None if args.exclude_idx < 0 else args.exclude_idx
    pairs = sample_xbridge_pairs(
        [m.vocab for m in members], args.n_pairs, args.seed, exclude)
    enc, ctrl = split_encoded_control(pairs, args.seed)
    print(f"  {len(enc)} ENCODED, {len(ctrl)} CONTROL", flush=True)

    # Control top-confidence BEFORE any encoding (only per-concept tags)
    ctrl_rates = []
    for ba, wa, bb, wb in ctrl:
        top = _query_top(members, wa)
        ctrl_rates.append(top[0][1] if top else 0.0)

    # Encode the ENCODED set, then measure their top-confidence
    for ba, wa, bb, wb in enc:
        ma, mb = members[ba], members[bb]
        tag = f"{wa}_{wb}"
        ma.encode_partial(wa, tag)
        mb.encode_partial(wb, tag)
        for m in (ma, mb):
            if tag not in m.encoded_tags:
                m.encoded_tags.append(tag)
    enc_rates, enc_correct = [], 0
    for ba, wa, bb, wb in enc:
        top = _query_top(members, wa)
        if not top:
            enc_rates.append(0.0)
            continue
        enc_rates.append(top[0][1])
        if top[0][0] == wb:
            enc_correct += 1

    enc_rates = np.array(enc_rates, float)
    ctrl_rates = np.array(ctrl_rates, float)
    max_ctrl = float(ctrl_rates.max()) if len(ctrl_rates) else 0.0
    # Clean-separation: encoded whose confidence beats EVERY control
    clean = int((enc_rates > max_ctrl).sum())
    # Threshold-free separability: AUC via Mann-Whitney U
    if len(enc_rates) and len(ctrl_rates):
        wins = sum(1.0 for e in enc_rates for c in ctrl_rates
                   if e > c) + 0.5 * sum(
                   1.0 for e in enc_rates for c in ctrl_rates if e == c)
        auc = wins / (len(enc_rates) * len(ctrl_rates))
    else:
        auc = 0.0

    summary = {
        "n_encoded": len(enc), "n_control": len(ctrl),
        "enc_rate_mean": float(enc_rates.mean()) if len(enc_rates) else 0,
        "enc_rate_min": float(enc_rates.min()) if len(enc_rates) else 0,
        "ctrl_rate_mean": float(ctrl_rates.mean()) if len(ctrl_rates) else 0,
        "ctrl_rate_max": max_ctrl,
        "clean_separation_rate": clean / max(len(enc_rates), 1),
        "separability_auc": auc,
        "encoded_top1_correct": enc_correct / max(len(enc), 1),
        "seed": args.seed,
    }
    print(f"\n=== RESULTS (hallucination resistance) ===", flush=True)
    print(f"  encoded  top-rate: mean {summary['enc_rate_mean']:.0f} "
          f"min {summary['enc_rate_min']:.0f}", flush=True)
    print(f"  control  top-rate: mean {summary['ctrl_rate_mean']:.0f} "
          f"max {summary['ctrl_rate_max']:.0f}", flush=True)
    print(f"  clean-separation: {clean}/{len(enc_rates)} = "
          f"{100*summary['clean_separation_rate']:.1f}% "
          f"(encoded conf > ALL control)", flush=True)
    print(f"  separability AUC: {auc:.3f}  "
          f"(1.0 = perfectly abstainable, 0.5 = hallucinates)",
          flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": summary,
                    "enc_rates": enc_rates.tolist(),
                    "ctrl_rates": ctrl_rates.tolist()},
                   open(args.out, "w"), indent=2)
        print(f"  -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
