"""Dynamical failure-signature probe (research; short GPU, no retrain).

The seed-quality NEGATIVE result localized the failure to the
DYNAMICAL regime (not static pattern overlap). This instruments the
recall dynamics of a KNOWN-failing index vs a robust one on an
EXISTING 320 bridge (idx-12 = 'ball' empirically failed at the
64-tier; idx-0 = 'apple' robust). It does NOT retrain or fix
anything -- it characterizes WHY, to correctly target the flagged
recovery.

For each probed per-concept tag: stim it, record per-step summed
firing of EVERY sparse pattern, and classify the failure mode:
  - UNDER-RECALL: the concept's own pattern barely fires.
  - COMPETITIVE CAPTURE: a DIFFERENT pattern dominates (and which).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from research.runners.concept_pool_sparse_distributed import (
    build_sparse_pool_bridge, generate_sparse_patterns,
)
from research.runners.g20_multibridge import read_vocab_file


def probe_tag(bridge, tag, pattern_arrs, steps=100):
    """Stim `tag`, return per-step (n_steps x n_patterns) firing sums."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    bridge.stimulate_tag(tag, drive_pA=1500.0)
    traj = np.zeros((steps, len(pattern_arrs)), dtype=np.float32)
    for s in range(steps):
        bridge._run_one_simulation_step()
        for j, parr in enumerate(pattern_arrs):
            f = bridge.cp_firing_states[parr]
            v = f.sum() if hasattr(f, "sum") else 0
            traj[s, j] = float(v.item() if hasattr(v, "item") else v)
    bridge.clear_tag_drive(tag)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    return traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridge", default="research/findings/raw/g11_bg/"
                   "g20_sparse_bridges_320/bridgeA_nouns_sparse64.simstate.h5")
    p.add_argument("--vocab", default="research/findings/raw/g11_bg/"
                   "g20_bridgeA_nouns_vocab64.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fail-idx", type=int, default=12)
    p.add_argument("--robust-idx", type=int, default=0)
    p.add_argument("--n-concepts", type=int, default=64)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    vocab = read_vocab_file(args.vocab)
    pats = generate_sparse_patterns(args.n_concepts, 2000, 100, args.seed)
    bridge = build_sparse_pool_bridge(
        seed=args.seed, n_lang_input=8192, n_shared_pool=2000,
        n_lang_output=8192, verbose=False)
    bridge.load_checkpoint(args.bridge)

    from sim.backend import get_backend
    cp, _ = get_backend()
    shared = list(bridge.region_manager.indices("shared_concept_pool"))
    parrs = [cp.asarray([shared[k] for k in pat], dtype=cp.int64)
             for pat in pats]

    out = {"bridge": args.bridge, "seed": args.seed, "probes": {}}
    for label, idx in (("robust", args.robust_idx),
                       ("failing", args.fail_idx)):
        word = vocab[idx]
        traj = probe_tag(bridge, word, parrs)
        cum = traj.sum(axis=0)                 # cumulative per pattern
        winner = int(np.argmax(cum))
        self_c = float(cum[idx])
        win_c = float(cum[winner])
        rank = int((cum > cum[idx]).sum()) + 1  # 1 = self wins
        # last-20-step steady firing of self
        steady_self = float(traj[-20:, idx].mean())
        mode = ("SELF-WINS" if winner == idx else
                f"CAPTURED-by-idx{winner}({vocab[winner]})")
        print(f"[{label}] '{word}' idx={idx}: self_cum={self_c:.0f} "
              f"winner=idx{winner}({vocab[winner]}) win_cum={win_c:.0f} "
              f"self_rank={rank} steady_self={steady_self:.2f} -> {mode}",
              flush=True)
        out["probes"][label] = {
            "word": word, "idx": idx, "self_cum": self_c,
            "winner_idx": winner, "winner_word": vocab[winner],
            "winner_cum": win_c, "self_rank": rank,
            "steady_self": steady_self, "mode": mode}

    r, f = out["probes"]["robust"], out["probes"]["failing"]
    print("\n=== DYNAMICAL SIGNATURE ===", flush=True)
    if f["self_rank"] == 1:
        verdict = ("failing idx self-WINS here too -> failure is "
                   "marginal/competitive (close race), not gross")
    elif f["self_cum"] < 0.3 * r["self_cum"]:
        verdict = ("UNDER-RECALL: failing idx self-pattern barely "
                   f"fires ({f['self_cum']:.0f} vs robust "
                   f"{r['self_cum']:.0f})")
    else:
        verdict = (f"COMPETITIVE CAPTURE by idx{f['winner_idx']} "
                   f"('{f['winner_word']}') -- self fires "
                   f"({f['self_cum']:.0f}) but loses to "
                   f"{f['winner_cum']:.0f}")
    print("  " + verdict, flush=True)
    out["verdict"] = verdict
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"  -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
