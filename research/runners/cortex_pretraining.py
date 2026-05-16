"""Phase 2.1 cortex pretraining via surrogate-gradient backprop.

ONLY ON path-f-hybrid BRANCH.

Numpy-based BPTT trainer for ABC task validation. Uses the
sim.bptt_snn reference implementation (CPU). Once this validates
loss decrease, we'll port the framework to CuPy for GPU and
larger corpora (Phase 2.2 Tiny Shakespeare).

Per Phase 2.1 design at
docs/plans/2026-05-06-Phase-2.1-surrogate-grad-design.md.

Toy task: predict next token in 'ABCABC...' cycle.
- Input one-hot of token at position t
- Target class of token at position t+1
- Architecture: 3 -> 32 -> 3 LIF stack
- Loss: cross-entropy on cumulative output spikes (rate code)

Pass criterion: training loss decreases by >= 50% over N epochs.
This validates BPTT framework end-to-end before GPU port.

Usage:
    python -m research.runners.cortex_pretraining \\
        --task abc --T 30 --epochs 100 --hidden 32 --lr 0.01 --seed 42
"""
from __future__ import annotations

import argparse
import time
import numpy as np


def train_abc(
    seed: int = 42,
    T: int = 30,
    hidden: int = 32,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 0.01,
    n_train_samples: int = 200,
    print_every: int = 10,
    verbose: bool = True,
):
    """Train SNN on ABC task via numpy BPTT.

    Returns dict with loss history.
    """
    from sim.bptt_snn import (
        LIFLayer, forward_unroll, backward_unroll,
        cross_entropy_loss_np, softmax_grad_np, make_abc_dataset,
    )

    rng = np.random.default_rng(seed)

    # Build dataset
    X, y = make_abc_dataset(
        n_samples=n_train_samples, seq_len=T + 1, rng=rng,
    )

    # Build 2-layer SNN: 3 -> hidden -> 3
    layers = [
        LIFLayer(W_in=rng.normal(0, 0.5, (3, hidden)).astype(np.float32),
                 n_post=hidden, threshold=1.0, leak=0.95),
        LIFLayer(W_in=rng.normal(0, 0.5, (hidden, 3)).astype(np.float32),
                 n_post=3, threshold=1.0, leak=0.95),
    ]

    if verbose:
        print(f"Training ABC task")
        print(f"  Architecture: 3 -> {hidden} -> 3 LIF")
        print(f"  T={T}, batch={batch_size}, lr={lr}, epochs={epochs}")
        print(f"  Samples: {n_train_samples}")
        print()

    loss_history = []
    n_batches = max(1, n_train_samples // batch_size)
    t0 = time.time()

    for epoch in range(epochs):
        perm = rng.permutation(n_train_samples)
        X_shuf = X[perm]
        y_shuf = y[perm]

        epoch_loss = 0.0
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, n_train_samples)
            B = end - start
            x_batch = X_shuf[start:end].transpose(1, 0, 2).astype(np.float32)
            target_batch = y_shuf[start:end, -1]  # last position target

            state = forward_unroll(x_batch, layers)
            logits = state["spikes"][-1].sum(axis=0)

            batch_loss = 0.0
            output_grad_accum = np.zeros_like(state["spikes"][-1])
            for s_idx in range(B):
                loss_s = cross_entropy_loss_np(
                    logits[s_idx:s_idx+1], int(target_batch[s_idx])
                )
                batch_loss += loss_s
                grad_s = softmax_grad_np(
                    logits[s_idx:s_idx+1], int(target_batch[s_idx])
                )
                output_grad_accum[:, s_idx, :] = grad_s[0]

            batch_loss /= B
            epoch_loss += batch_loss

            weight_grads, _ = backward_unroll(
                x_batch, layers, state, output_grad_accum
            )

            for li, layer in enumerate(layers):
                layer.W_in -= lr * weight_grads[li]

        epoch_loss /= n_batches
        loss_history.append(float(epoch_loss))

        if verbose and (epoch + 1) % print_every == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f} "
                  f"({elapsed:.1f}s)")

    if verbose:
        print(f"\nTraining complete ({time.time()-t0:.1f}s)")
        print(f"  Initial loss: {loss_history[0]:.4f}")
        print(f"  Final loss:   {loss_history[-1]:.4f}")
        if loss_history[0] > 0:
            reduction = (1 - loss_history[-1]/loss_history[0]) * 100
            print(f"  Reduction:    {reduction:.0f}%")

    return {
        "loss_history": loss_history,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "reduction_pct": ((1 - loss_history[-1]/loss_history[0]) * 100
                          if loss_history[0] > 0 else 0.0),
    }


def train_shakespeare(
    seed: int = 42,
    T: int = 32,
    hidden_layers: list = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.005,
    n_train_samples: int = 1000,
    corpus_path: str = "data/tinyshakespeare.txt",
    print_every: int = 5,
    backend: str = "auto",
    verbose: bool = True,
):
    """Train SNN on Tiny Shakespeare next-char task.

    Uses GPU-aware bptt_snn_gpu for CuPy backend when available.

    Args:
        hidden_layers: list of hidden layer sizes. Default [128, 128]
            gives a 4-layer net (input -> 128 -> 128 -> output).
        backend: 'auto', 'cpu', or 'gpu'.
    """
    from sim.bptt_snn_gpu import (
        LIFLayerXP, forward_unroll_xp, backward_unroll_xp,
        _get_backend,
    )
    from sim.bptt_snn import cross_entropy_loss_np, softmax_grad_np
    from sim.char_tokenizer import (
        load_tiny_shakespeare, CharTokenizer, make_seq_dataset,
    )

    if hidden_layers is None:
        hidden_layers = [128, 128]

    # Backend
    if backend == "cpu":
        import numpy as xp
        is_gpu = False
    elif backend == "gpu":
        import cupy as xp
        is_gpu = True
    else:  # auto
        xp, is_gpu = _get_backend(prefer_gpu=True)

    rng = np.random.default_rng(seed)

    # Load corpus
    corpus = load_tiny_shakespeare(path=corpus_path)
    if verbose:
        print(f"Corpus: {len(corpus):,} chars")
    tok = CharTokenizer(corpus)
    if verbose:
        print(f"Vocab size: {tok.vocab_size}")

    V = tok.vocab_size
    # Build dataset (numpy first; convert to xp as needed)
    X_np, y_np = make_seq_dataset(
        corpus, tok, seq_len=T, n_samples=n_train_samples, rng=rng,
    )
    if is_gpu:
        X = xp.asarray(X_np, dtype=xp.float32)
        y = xp.asarray(y_np, dtype=xp.int64)
    else:
        X = X_np
        y = y_np

    # Build N-layer SNN: V -> hidden_layers... -> V
    layer_sizes = [V] + hidden_layers + [V]
    if verbose:
        print(f"Architecture: {' -> '.join(str(s) for s in layer_sizes)} LIF")
    # Init: for one-hot inputs only one neuron fires per timestep,
    # so we need weights large enough that a single active input
    # can push voltage above threshold. std=2.0 works for ABC (3 ->
    # hidden) and Shakespeare (66 -> hidden).
    layers = []
    for li in range(len(layer_sizes) - 1):
        n_pre = layer_sizes[li]
        n_post = layer_sizes[li + 1]
        # First layer: large init for one-hot drive.
        # Later layers: smaller init for sparse-spike input.
        std = 2.0 if li == 0 else 0.5
        W_np = rng.normal(0, std,
                           (n_pre, n_post)).astype(np.float32)
        if is_gpu:
            W = xp.asarray(W_np)
        else:
            W = W_np
        layers.append(LIFLayerXP(
            W_in=W, n_post=n_post, threshold=1.0, leak=0.95,
        ))

    if verbose:
        print(f"  Backend: {'GPU (CuPy)' if is_gpu else 'CPU (numpy)'}")
        print(f"  T={T}, batch={batch_size}, lr={lr}, epochs={epochs}")
        print(f"  Samples: {n_train_samples}")
        print()

    loss_history = []
    n_batches = max(1, n_train_samples // batch_size)
    t0 = time.time()

    for epoch in range(epochs):
        # Shuffle
        perm = rng.permutation(n_train_samples)
        if is_gpu:
            X_shuf = X[xp.asarray(perm)]
            y_shuf = y[xp.asarray(perm)]
        else:
            X_shuf = X[perm]
            y_shuf = y[perm]

        epoch_loss = 0.0
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, n_train_samples)
            B = end - start
            x_batch = X_shuf[start:end].transpose(1, 0, 2)
            target_batch = y_shuf[start:end, -1]  # last-position target

            state = forward_unroll_xp(x_batch, layers, xp=xp)
            # Logits: cumulative output spikes (across time, last layer)
            logits = state["spikes"][-1].sum(axis=0)  # (B, V)
            if is_gpu:
                logits_np = logits.get()
            else:
                logits_np = logits

            # Per-sample loss + grad (numpy for simplicity)
            batch_loss = 0.0
            tgt_np = target_batch if not is_gpu else target_batch.get()
            output_grad_np = np.zeros((T, B, V), dtype=np.float32)
            for s_idx in range(B):
                loss_s = cross_entropy_loss_np(
                    logits_np[s_idx:s_idx+1], int(tgt_np[s_idx])
                )
                batch_loss += loss_s
                grad_s = softmax_grad_np(
                    logits_np[s_idx:s_idx+1], int(tgt_np[s_idx])
                )
                # Distribute uniformly over time
                output_grad_np[:, s_idx, :] = grad_s[0]

            batch_loss /= B
            epoch_loss += batch_loss

            if is_gpu:
                output_grad = xp.asarray(output_grad_np)
            else:
                output_grad = output_grad_np

            weight_grads, _ = backward_unroll_xp(
                x_batch, layers, state, output_grad, xp=xp,
            )

            # SGD update
            for li, layer in enumerate(layers):
                layer.W_in -= lr * weight_grads[li]

        epoch_loss /= n_batches
        loss_history.append(float(epoch_loss))

        if verbose and (epoch + 1) % print_every == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f} "
                  f"({elapsed:.1f}s)")

    if verbose:
        print(f"\nTraining complete ({time.time()-t0:.1f}s)")
        print(f"  Initial loss: {loss_history[0]:.4f}")
        print(f"  Final loss:   {loss_history[-1]:.4f}")
        if loss_history[0] > 0:
            reduction = (1 - loss_history[-1]/loss_history[0]) * 100
            print(f"  Reduction:    {reduction:.0f}%")

    return {
        "loss_history": loss_history,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "vocab_size": V,
        "n_layers": len(layers),
        "is_gpu": is_gpu,
        "_layers": layers,           # for save_checkpoint
        "_vocab": tok.vocab,         # for save_checkpoint
    }


def save_checkpoint(result: dict, path: str):
    """Save Phase 2.2 SNN trained weights to .npz + sidecar metadata.json.

    Two files written:
      <path>.npz: numerical arrays (weights, layer_sizes, thresholds, leaks)
      <path>.metadata.json: vocab + scalar metadata (avoids pickle)

    Used by Phase 2.3 to load pretrained cortex into SimulationBridge.
    """
    from pathlib import Path
    import json
    layers = result.get("_layers")
    vocab = result.get("_vocab")
    if layers is None:
        raise ValueError("save_checkpoint: result dict must include "
                         "_layers (returned by train_shakespeare).")
    npz = {}
    for i, layer in enumerate(layers):
        W = layer.W_in
        try:
            W = W.get()  # cupy -> numpy
        except AttributeError:
            pass
        npz[f"W_layer_{i}"] = W.astype(np.float32)
    npz["n_layers"] = np.array(len(layers))
    npz["layer_sizes"] = np.array(
        [layers[0].W_in.shape[0]] + [layer.n_post for layer in layers]
    )
    npz["thresholds"] = np.array([layer.threshold for layer in layers])
    npz["leaks"] = np.array([layer.leak for layer in layers])
    npz["loss_history"] = np.array(result.get("loss_history", []))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(p), **npz)
    if vocab is not None:
        meta_path = p.with_suffix(".metadata.json")
        meta_path.write_text(json.dumps({"vocab": list(vocab)}))


def load_checkpoint(path: str) -> dict:
    """Load Phase 2.2 checkpoint (.npz + sidecar .metadata.json)."""
    from pathlib import Path
    import json
    p = Path(path)
    data = np.load(str(p))
    n_layers = int(data["n_layers"])
    W_layers = [data[f"W_layer_{i}"] for i in range(n_layers)]
    vocab = None
    meta_path = p.with_suffix(".metadata.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        vocab = meta.get("vocab")
    return {
        "W_layers": W_layers,
        "layer_sizes": data["layer_sizes"].tolist(),
        "thresholds": data["thresholds"].tolist(),
        "leaks": data["leaks"].tolist(),
        "vocab": vocab,
        "loss_history": (data["loss_history"].tolist()
                          if "loss_history" in data.files else []),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["abc", "shakespeare"], default="abc")
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=32,
                    help="ABC task: hidden layer size (single layer)")
    ap.add_argument("--hidden-layers", type=str, default="128,128",
                    help="Shakespeare task: comma-sep hidden layer sizes")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--n-train-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--corpus-path", type=str, default="data/tinyshakespeare.txt")
    ap.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--out-checkpoint", type=str, default=None,
                    help="Save trained SNN weights to .npz (for Phase 2.3)")
    args = ap.parse_args()

    if args.task == "abc":
        result = train_abc(
            seed=args.seed,
            T=args.T,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_train_samples=args.n_train_samples,
            print_every=args.print_every,
            verbose=True,
        )
    elif args.task == "shakespeare":
        hidden_layers = [int(x) for x in args.hidden_layers.split(",")]
        result = train_shakespeare(
            seed=args.seed,
            T=args.T,
            hidden_layers=hidden_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_train_samples=args.n_train_samples,
            corpus_path=args.corpus_path,
            print_every=args.print_every,
            backend=args.backend,
            verbose=True,
        )
    else:
        raise NotImplementedError(f"Task {args.task} not yet implemented")

    if args.out_checkpoint and "_layers" in result:
        save_checkpoint(result, args.out_checkpoint)
        print(f"Saved checkpoint: {args.out_checkpoint}")

    if args.out_stats:
        import json
        from pathlib import Path
        # Strip non-serializable _layers / _vocab from JSON
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(clean, indent=2))
        print(f"Saved stats: {args.out_stats}")


if __name__ == "__main__":
    main()
