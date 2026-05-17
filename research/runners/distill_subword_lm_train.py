"""Generator-D distillation trainer. BYTE-MIRROR of the VALIDATED
research/runners/scaled_subword_lm_train.py:train_subword_lm with the
SOLE behavioral change: the per-sample target is the NgramTeacher's
DENSE soft distribution (knowledge distillation via soft cross-entropy)
instead of the hard one-hot next token. The teacher context is
recovered from the SAME permuted batch window the logits came from
(backend-safe), so ctx aligns with logits regardless of shuffle/
backend. Everything else (init std 2.0/0.5, threshold 1.0, leak 0.95,
forward/backward unroll, sim.train_checkpoint kill-safe resume,
OOM-halving, KeyboardInterrupt clean exit) is IDENTICAL to the
validated trainer. Teacher is TRAINING-TIME ONLY; runtime artifact =
SNN weights .npz + BPE .json. ASCII only (Windows cp1252)."""
from __future__ import annotations

import argparse
import time
import numpy as np


def _build_layers(layer_sizes, rng, is_gpu, xp):
    from sim.bptt_snn_gpu import LIFLayerXP
    layers = []
    for li in range(len(layer_sizes) - 1):
        n_pre = layer_sizes[li]
        n_post = layer_sizes[li + 1]
        std = 2.0 if li == 0 else 0.5
        W_np = rng.normal(0, std, (n_pre, n_post)).astype(np.float32)
        W = xp.asarray(W_np) if is_gpu else W_np
        layers.append(LIFLayerXP(W_in=W, n_post=n_post,
                                 threshold=1.0, leak=0.95))
    return layers


def _host(W):
    try:
        return W.get()
    except AttributeError:
        return np.asarray(W)


def train_distill_subword_lm(
    seed: int = 42,
    corpus_path: str = "data/tinyshakespeare.txt",
    vocab_size: int = 512,
    hidden_layers=None,
    T: int = 32,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.005,
    n_train_samples: int = 1000,
    ckpt_path: str = "research/findings/raw/g11_bg/gen_d.ckpt.npz",
    bpe_path: str = "research/findings/raw/g11_bg/gen_d.bpe.json",
    backend: str = "auto",
    print_every: int = 5,
    verbose: bool = True,
):
    """Kill-safe scaled subword-SNN DISTILLATION trainer. Re-running
    the SAME command resumes from the last completed epoch (atomic
    checkpoint); KeyboardInterrupt flushes a final checkpoint and
    returns cleanly so the user can free the GPU and resume later."""
    import os
    from pathlib import Path
    from sim.bptt_snn_gpu import (
        forward_unroll_xp, backward_unroll_xp, _get_backend,
    )
    from sim.soft_xent import soft_xent_loss, soft_xent_grad
    from sim.ngram_teacher import NgramTeacher
    from sim.char_tokenizer import make_seq_dataset            # UNMODIFIED
    from sim.bpe_tokenizer import BPETokenizer
    from sim.train_checkpoint import (
        save_checkpoint, load_checkpoint, resume_epoch,
    )

    if hidden_layers is None:
        hidden_layers = [128, 128]

    if backend == "cpu":
        import numpy as xp
        is_gpu = False
    elif backend == "gpu":
        import cupy as xp
        is_gpu = True
    else:
        xp, is_gpu = _get_backend(prefer_gpu=True)

    corpus = Path(corpus_path).read_text(encoding="utf-8")
    if verbose:
        print("[gen-d] corpus: %d chars" % len(corpus))

    if os.path.exists(bpe_path):
        tok = BPETokenizer.load(bpe_path)
        if verbose:
            print("[gen-d] loaded cached BPE (%d vocab) %s"
                  % (tok.vocab_size, bpe_path))
    else:
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=vocab_size)
        Path(bpe_path).parent.mkdir(parents=True, exist_ok=True)
        tok.save(bpe_path)
        if verbose:
            print("[gen-d] trained BPE (%d vocab) -> %s"
                  % (tok.vocab_size, bpe_path))
    V = tok.vocab_size

    # NET-NEW: the distillation teacher (training-time only). Built
    # ONCE on the full train token stream; dense soft target per ctx.
    teacher = NgramTeacher()
    teacher.train(tok.encode(corpus), vocab_size=V)
    if verbose:
        print("[gen-d] trigram teacher trained (vocab %d)" % V)

    data_rng = np.random.default_rng(seed)
    X_np, y_np = make_seq_dataset(corpus, tok, seq_len=T,
                                  n_samples=n_train_samples, rng=data_rng)
    if is_gpu:
        X = xp.asarray(X_np, dtype=xp.float32)
        y = xp.asarray(y_np, dtype=xp.int64)
    else:
        X, y = X_np, y_np

    layer_sizes = [V] + list(hidden_layers) + [V]
    if verbose:
        print("[gen-d] arch: %s LIF  backend=%s  T=%d batch=%d lr=%s "
              "epochs=%d samples=%d"
              % (" -> ".join(str(s) for s in layer_sizes),
                 "GPU" if is_gpu else "CPU", T, batch_size, lr,
                 epochs, n_train_samples))

    init_rng = np.random.default_rng(seed * 7919 + 1)
    layers = _build_layers(layer_sizes, init_rng, is_gpu, xp)

    ckpt = load_checkpoint(ckpt_path)
    start_epoch = resume_epoch(ckpt)
    loss_history = []
    if ckpt is not None:
        for li in range(len(layers)):
            Wi = ckpt["weights"][li]
            layers[li].W_in = xp.asarray(Wi) if is_gpu else np.asarray(Wi)
        loss_history = list(ckpt["loss_history"])
        if verbose:
            print("[gen-d] RESUMED from epoch %d (ckpt %s)"
                  % (start_epoch, ckpt_path))

    n_batches = max(1, n_train_samples // batch_size)
    t0 = time.time()
    cur_batch = batch_size

    def _flush(ep):
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_path, ep,
                        [_host(l.W_in) for l in layers],
                        np.random.default_rng(seed).bit_generator.state,
                        loss_history)

    try:
        for epoch in range(start_epoch, epochs):
            perm = np.random.default_rng(
                seed * 100003 + epoch).permutation(n_train_samples)
            if is_gpu:
                Xs, ys = X[xp.asarray(perm)], y[xp.asarray(perm)]
            else:
                Xs, ys = X[perm], y[perm]

            epoch_loss = 0.0
            bi = 0
            while bi < n_batches:
                start = bi * cur_batch
                if start >= n_train_samples:
                    break
                end = min(start + cur_batch, n_train_samples)
                B = end - start
                try:
                    x_b = Xs[start:end].transpose(1, 0, 2)
                    st = forward_unroll_xp(x_b, layers, xp=xp)
                    logits = st["spikes"][-1].sum(axis=0)
                    logits_np = logits.get() if is_gpu else logits
                    # NET-NEW: teacher ctx from the SAME permuted batch
                    # window the logits came from (backend-safe argmax
                    # of the permuted Xs slice -> ctx aligns with
                    # logits regardless of shuffle/backend).
                    Xb = Xs[start:end]
                    Xb_ids = (Xb.get() if is_gpu
                              else np.asarray(Xb)).argmax(axis=2)  # (B,T)
                    bl = 0.0
                    og = np.zeros((T, B, V), dtype=np.float32)
                    for s in range(B):
                        if T >= 2:
                            ctx = (int(Xb_ids[s, T - 2]),
                                   int(Xb_ids[s, T - 1]))
                        else:
                            ctx = (int(Xb_ids[s, T - 1]),)
                        q = teacher.soft_dist(ctx)     # dense soft target
                        bl += soft_xent_loss(logits_np[s:s+1], q)
                        og[:, s, :] = soft_xent_grad(
                            logits_np[s:s+1], q)[0]
                    bl /= B
                    epoch_loss += bl
                    og_x = xp.asarray(og) if is_gpu else og
                    wg, _ = backward_unroll_xp(x_b, layers, st, og_x,
                                               xp=xp)
                    for li, layer in enumerate(layers):
                        layer.W_in -= lr * wg[li]
                    bi += 1
                except Exception as e:                # OOM -> halve batch
                    if "memory" in type(e).__name__.lower() or \
                       "OutOfMemory" in type(e).__name__:
                        if cur_batch <= 1:
                            raise
                        cur_batch = max(1, cur_batch // 2)
                        n_batches = max(
                            1, n_train_samples // cur_batch)
                        if verbose:
                            print("[gen-d] OOM -> batch halved to %d"
                                  % cur_batch, flush=True)
                        continue
                    raise

            epoch_loss /= n_batches
            loss_history.append(float(epoch_loss))
            _flush(epoch)
            if verbose and (epoch + 1) % print_every == 0:
                print("[gen-d] epoch %d/%d loss=%.4f (%.1fs)"
                      % (epoch + 1, epochs, epoch_loss,
                         time.time() - t0), flush=True)
    except KeyboardInterrupt:
        done = (len(loss_history) - 1) if loss_history else 0
        _flush(max(0, done))
        if verbose:
            print("[gen-d] INTERRUPTED -> checkpoint flushed; re-run "
                  "the same command to resume.", flush=True)
        return {
            "loss_history": loss_history,
            "initial_loss": loss_history[0] if loss_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
            "vocab_size": V, "n_layers": len(layers),
            "is_gpu": is_gpu, "interrupted": True,
            "bpe_path": bpe_path, "ckpt_path": ckpt_path,
            "_layers": layers, "_tok": tok, "_teacher": teacher,
        }

    if verbose and loss_history:
        print("[gen-d] done (%.1fs) init=%.4f final=%.4f"
              % (time.time() - t0, loss_history[0],
                 loss_history[-1]), flush=True)
    return {
        "loss_history": loss_history,
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "vocab_size": V, "n_layers": len(layers), "is_gpu": is_gpu,
        "interrupted": False, "bpe_path": bpe_path,
        "ckpt_path": ckpt_path, "_layers": layers, "_tok": tok,
        "_teacher": teacher,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus-path", type=str,
                    default="data/tinyshakespeare.txt")
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--hidden-layers", type=str, default="128,128")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--n-train-samples", type=int, default=1000)
    ap.add_argument("--ckpt-path", type=str,
                    default="research/findings/raw/g11_bg/gen_d.ckpt.npz")
    ap.add_argument("--bpe-path", type=str,
                    default="research/findings/raw/g11_bg/gen_d.bpe.json")
    ap.add_argument("--backend", choices=["auto", "cpu", "gpu"],
                    default="auto")
    ap.add_argument("--print-every", type=int, default=5)
    a = ap.parse_args()
    hl = [int(x) for x in a.hidden_layers.split(",") if x.strip()]
    train_distill_subword_lm(
        seed=a.seed, corpus_path=a.corpus_path,
        vocab_size=a.vocab_size, hidden_layers=hl, T=a.T,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        n_train_samples=a.n_train_samples, ckpt_path=a.ckpt_path,
        bpe_path=a.bpe_path, backend=a.backend,
        print_every=a.print_every, verbose=True)


if __name__ == "__main__":
    raise SystemExit(main())
