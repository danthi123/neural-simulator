"""Inc-3 HELD-OUT evaluator -- the pre-registered anti-cheat metric.

WHY THIS EXISTS (honest correction): the scaled trainer records only
TRAINING loss per epoch. Both REAL and PERMUTED drove training loss
to ~0.06 -- a ~600K-param net memorizing 2000 fixed windows (the
PERMUTED control, with zero sequential structure, memorized equally
well). Comparing TRAINING loss REAL-vs-PERMUTED is a memorization
artifact, NOT evidence of learned generalizable structure. The Inc-3
plan pre-registered HELD-OUT loss for exactly this reason. This module
computes that held-out loss on windows PROVABLY DISJOINT from the
2000 the net trained on, so the fixed >=10% gate is applied to the
right quantity. The gate bar is NOT changed -- only the measured
quantity is corrected to what the plan specified.

Disjointness is exact, not probabilistic: `make_seq_dataset` draws one
`rng.integers(0, n_chars-seq_len-1)` per sample, so the training start
indices are reconstructed deterministically from the training seed,
and held-out starts are rejected unless every training start is
> seq_len away (zero character overlap with any training window).

Pure helpers (`reconstruct_train_starts`, `select_heldout_starts`) are
CPU-unit-tested. `main()` loads a trained checkpoint via the verified
`sim.train_checkpoint.load_checkpoint`, rebuilds the SAME corpus the
net trained on (incl. `--permute-corpus`), evaluates mean cross-entropy
on the disjoint held-out windows, and writes a small JSON.

Usage:
    python -m research.runners.scaled_heldout_eval \\
        --ckpt research/findings/raw/g11_bg/scaled_gen_real.ckpt.npz \\
        --corpus all --seed 42 --T 96 --n-train-samples 2000 \\
        --n-heldout 2000 --heldout-seed 12345 \\
        --out research/findings/raw/g11_bg/scaled_gen_real.heldout.json
Add --permute-corpus when evaluating the PERMUTED control checkpoint
(rebuilds the identical shuffled corpus the control trained on).
"""
from __future__ import annotations

import argparse


def reconstruct_train_starts(n_chars: int, seq_len: int,
                             n_train_samples: int, seed: int) -> list:
    """Replay make_seq_dataset's start-index stream EXACTLY.

    make_seq_dataset(text, tok, seq_len, n_samples, rng) draws, per
    sample, `start = int(rng.integers(0, n_chars - seq_len - 1))` and
    nothing else from rng. The trainer used
    `rng = np.random.default_rng(seed)` then called it once. So the
    training start set is fully determined by (n_chars, seq_len,
    n_train_samples, seed). n_chars == len(corpus) for a char
    tokenizer (1 id per char).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    hi = n_chars - seq_len - 1
    return [int(rng.integers(0, hi)) for _ in range(n_train_samples)]


def select_heldout_starts(n_chars: int, seq_len: int, train_starts,
                          n_heldout: int, rng) -> list:
    """Sample `n_heldout` start indices with ZERO character overlap
    with any training window.

    A held-out start s is accepted only if |s - ts| > seq_len for
    every training start ts (so [s, s+seq_len] shares no character
    with any [ts, ts+seq_len]). Deterministic given `rng`. Raises
    ValueError if the disjoint space is too small (never silently
    returns a contaminated set -- that would make the gate a cheat).
    """
    import numpy as np
    hi = n_chars - seq_len - 1
    if hi <= 0:
        raise ValueError("corpus too short for seq_len")
    train_sorted = np.array(sorted(set(int(t) for t in train_starts)))

    def _clean(s: int) -> bool:
        # nearest training start must be strictly more than seq_len away
        i = np.searchsorted(train_sorted, s)
        for j in (i - 1, i):
            if 0 <= j < len(train_sorted):
                if abs(int(train_sorted[j]) - s) <= seq_len:
                    return False
        return True

    out: list = []
    seen = set()
    # Cap attempts so an impossible request fails fast instead of
    # spinning forever.
    max_attempts = max(50 * n_heldout, 200_000)
    attempts = 0
    while len(out) < n_heldout and attempts < max_attempts:
        attempts += 1
        s = int(rng.integers(0, hi))
        if s in seen:
            continue
        if _clean(s):
            seen.add(s)
            out.append(s)
    if len(out) < n_heldout:
        raise ValueError(
            "insufficient disjoint held-out space: got %d/%d clean "
            "starts in %d attempts (corpus likely too small or "
            "train coverage too dense)"
            % (len(out), n_heldout, attempts))
    return out


def _build_corpus(which: str, permute: bool, seed: int) -> str:
    """Rebuild the EXACT corpus the checkpoint trained on (same loader,
    same optional deterministic permutation as scaled_generator_train).
    """
    import numpy as np
    from research.runners.scaled_generator_train import _load_corpus
    corpus = _load_corpus(which)
    if permute:
        perm_rng = np.random.default_rng(seed)
        chars = list(corpus)
        perm_rng.shuffle(chars)
        corpus = "".join(chars)
    return corpus


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--corpus", type=str, default="all")
    ap.add_argument("--permute-corpus", action="store_true", default=False,
                    help="Rebuild the shuffled corpus (use for the "
                         "PERMUTED control checkpoint).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Training seed (to reconstruct train starts).")
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--n-train-samples", type=int, default=2000,
                    help="Must match the trainer's --n-samples.")
    ap.add_argument("--n-heldout", type=int, default=2000)
    ap.add_argument("--heldout-seed", type=int, default=12345)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--out", type=str, required=True)
    return ap


def main():
    import json

    import numpy as np

    from sim.bptt_snn import cross_entropy_loss_np
    from sim.bptt_snn_gpu import (
        LIFLayerXP, forward_unroll_xp, _get_backend,
    )
    from sim.char_tokenizer import CharTokenizer
    from sim.train_checkpoint import load_checkpoint

    args = _build_arg_parser().parse_args()

    ck = load_checkpoint(args.ckpt)
    if ck is None:
        print("ckpt not ready: %s" % args.ckpt)
        return 2
    if not ck.get("loss_history"):
        print("ckpt has empty loss_history (not trained): %s" % args.ckpt)
        return 2

    corpus = _build_corpus(args.corpus, args.permute_corpus, args.seed)
    tok = CharTokenizer(corpus)
    V = tok.vocab_size
    ids = np.array(tok.encode(corpus), dtype=np.int64)
    n_chars = len(ids)
    T = args.T

    train_starts = reconstruct_train_starts(
        n_chars, T, args.n_train_samples, args.seed)
    ho_starts = select_heldout_starts(
        n_chars, T, train_starts, args.n_heldout,
        np.random.default_rng(args.heldout_seed))

    # Build held-out one-hot windows + last-position targets (same
    # convention the trainer uses: target = y[:, -1]).
    n = len(ho_starts)
    X = np.zeros((n, T, V), dtype=np.float32)
    ytgt = np.zeros((n,), dtype=np.int64)
    for i, s in enumerate(ho_starts):
        w = ids[s:s + T + 1]
        for t in range(T):
            X[i, t, w[t]] = 1.0
        ytgt[i] = w[T]

    # Rebuild the trained net (sizes inferred from saved weights;
    # same LIF hyperparams as the trainer: threshold 1.0, leak 0.95).
    saved_w = ck["weights"]
    xp, is_gpu = _get_backend(prefer_gpu=True)
    layer_sizes = ([saved_w[0].shape[0]]
                   + [w.shape[1] for w in saved_w])
    layers = []
    for li, W_host in enumerate(saved_w):
        W = xp.asarray(W_host) if is_gpu else np.asarray(W_host)
        layers.append(LIFLayerXP(W_in=W, n_post=layer_sizes[li + 1],
                                  threshold=1.0, leak=0.95))

    # Forward-only eval in batches; mean CE over held-out windows.
    total_loss = 0.0
    nb = 0
    B = max(1, args.eval_batch)
    for s0 in range(0, n, B):
        s1 = min(s0 + B, n)
        xb = X[s0:s1].transpose(1, 0, 2)
        if is_gpu:
            xb = xp.asarray(xb, dtype=xp.float32)
        st = forward_unroll_xp(xb, layers, xp=xp)
        logits = st["spikes"][-1].sum(axis=0)
        logits_np = logits.get() if is_gpu else logits
        for k in range(s1 - s0):
            total_loss += cross_entropy_loss_np(
                logits_np[k:k + 1], int(ytgt[s0 + k]))
        nb += (s1 - s0)
    heldout_loss = float(total_loss / max(1, nb))

    result = {
        "ckpt": args.ckpt,
        "permuted": bool(args.permute_corpus),
        "heldout_loss": heldout_loss,
        "n_heldout": n,
        "n_train_samples": args.n_train_samples,
        "vocab_size": V,
        "ln_V": float(np.log(V)),
        "layer_sizes": layer_sizes,
        "trained_epochs": ck["epoch"],
        "final_train_loss": float(ck["loss_history"][-1]),
        "T": T,
        "heldout_seed": args.heldout_seed,
        "zero_overlap_with_train": True,
    }
    print("=" * 64)
    print("HELD-OUT EVAL (pre-registered anti-cheat metric)")
    print("=" * 64)
    print("  ckpt            : %s" % args.ckpt)
    print("  permuted control: %s" % result["permuted"])
    print("  trained epochs  : %d  final TRAIN loss: %.4f"
          % (result["trained_epochs"], result["final_train_loss"]))
    print("  held-out windows: %d  (zero char overlap w/ %d train)"
          % (n, args.n_train_samples))
    print("  HELD-OUT loss   : %.4f   (ln V = %.4f = uniform chance)"
          % (heldout_loss, result["ln_V"]))
    print("=" * 64)

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("[written] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
