"""Increment-3: scaled char-level BPTT-SNN generator, kill-safe resumable.

A much larger char-level SNN than the toy Inc-1 net, trained on ALL local
English text (repo findings prose + the cached teacher-distilled corpus),
that the user can KILL AT ANY INSTANT to free the GPU for gaming and
resume simply by re-running the exact same command.

Design (DRY -- reuses verified cores; nothing reimplemented):
  - sim.train_checkpoint   : atomic per-epoch save / load / resume_epoch
  - sim.bptt_snn_gpu       : LIFLayerXP, forward/backward_unroll_xp, backend
  - sim.bptt_snn           : cross_entropy_loss_np, softmax_grad_np
  - sim.char_tokenizer     : CharTokenizer, make_seq_dataset
  - research.runners.local_corpus.load_local_corpus
  - research.runners.build_distill_corpus.clean_corpus
  - research/datasets/distill_corpus.txt (optional teacher English)

The per-epoch training loop is a direct mirror of
`research.runners.cortex_pretraining.train_shakespeare` (same backend
handling, same init std=2.0 first layer / 0.5 later, same
forward_unroll_xp -> per-sample CE loss/grad -> backward_unroll_xp ->
SGD weight update). The ONLY additions here are: multi-layer from CLI,
the full-corpus loader, per-epoch atomic checkpoint, auto-resume, OOM
auto-halving of the batch, and KeyboardInterrupt -> clean checkpointed
exit.

Usage:
    python -m research.runners.scaled_generator_train \\
        --hidden "512,512,512" --T 96 --batch 64 --epochs 400 \\
        --n-samples 2000 --seed 42 --lr 0.005 \\
        --ckpt research/findings/raw/g11_bg/scaled_gen.ckpt.npz

Kill it (Ctrl-C / SIGKILL) any time; re-run the SAME command to resume
from the last completed epoch.
"""
from __future__ import annotations

import argparse

# Cap corpus length so an epoch stays a sane wall-clock duration. Real
# English with sequential structure; 1M ascii chars is plenty for a
# char-level next-char generator at this scale.
_CORPUS_CHAR_CAP = 1_000_000


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--hidden", type=str, default="512,512,512",
                    help="Comma-separated hidden layer sizes, "
                         'e.g. "512,512,512".')
    ap.add_argument("--T", type=int, default=96,
                    help="BPTT unroll length (sequence length).")
    ap.add_argument("--batch", type=int, default=64,
                    help="Batch size (auto-halved on OOM).")
    ap.add_argument("--epochs", type=int, default=400,
                    help="Total epochs to train to (inclusive target).")
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="Number of (input,target) windows sampled "
                         "from the corpus.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument(
        "--ckpt", type=str,
        default="research/findings/raw/g11_bg/scaled_gen.ckpt.npz",
        help="Checkpoint .npz path (atomic, kill-safe).")
    ap.add_argument(
        "--corpus", type=str, default="all",
        help='"all" = clean_corpus(local repo English + distill corpus '
             "if present), capped to %d ascii chars."
             % _CORPUS_CHAR_CAP)
    ap.add_argument(
        "--permute-corpus", action="store_true", default=False,
        help="Anti-cheat control: deterministically shuffle the cleaned "
             "corpus characters (seeded by --seed) BEFORE tokenization. "
             "Destroys sequential structure while preserving the exact "
             "character distribution. Use a distinct --ckpt path.")
    return ap


def _load_corpus(which: str) -> str:
    """Build the training corpus string.

    "all" -> clean_corpus(load_local_corpus() + "\\n\\n" + distill if it
    exists), capped to _CORPUS_CHAR_CAP ascii chars for sane epoch time.
    Any other value is treated as a literal path to a UTF-8 text file.
    """
    from research.runners.build_distill_corpus import (
        DISTILL_PATH, clean_corpus,
    )

    if which == "all":
        from research.runners.local_corpus import load_local_corpus
        parts = [load_local_corpus()]
        # Teacher-distilled English may not exist -- handle gracefully.
        try:
            if DISTILL_PATH.exists():
                parts.append(DISTILL_PATH.read_text(
                    encoding="utf-8", errors="ignore"))
        except OSError:
            pass
        raw = "\n\n".join(p for p in parts if p)
    else:
        from pathlib import Path
        raw = Path(which).read_text(encoding="utf-8", errors="ignore")

    cleaned = clean_corpus(raw)
    if len(cleaned) > _CORPUS_CHAR_CAP:
        cleaned = cleaned[:_CORPUS_CHAR_CAP]
    return cleaned


def _is_oom(exc: BaseException) -> bool:
    """True if exc is a CuPy OOM or a RuntimeError mentioning OOM."""
    name = type(exc).__name__
    if name == "OutOfMemoryError":  # cupy.cuda.memory.OutOfMemoryError
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return ("out of memory" in msg) or ("oom" in msg)
    return False


def main() -> int:
    # Lazy imports so `import research.runners.scaled_generator_train`
    # stays light and instant (no numpy/cupy at module import time).
    import time

    import numpy as np

    from sim.bptt_snn import cross_entropy_loss_np, softmax_grad_np
    from sim.bptt_snn_gpu import (
        LIFLayerXP, backward_unroll_xp, forward_unroll_xp, _get_backend,
    )
    from sim.char_tokenizer import CharTokenizer, make_seq_dataset
    from sim.train_checkpoint import (
        load_checkpoint, resume_epoch, save_checkpoint,
    )

    args = _build_arg_parser().parse_args()

    hidden_list = [int(x) for x in args.hidden.split(",") if x.strip()]
    if not hidden_list:
        raise SystemExit("--hidden must give at least one layer size")

    # Backend (CuPy GPU if available, else numpy) -- same call
    # train_shakespeare uses.
    xp, is_gpu = _get_backend(prefer_gpu=True)
    backend_name = "GPU (CuPy)" if is_gpu else "CPU (numpy)"

    # --- Corpus + tokenizer + dataset (built once, seeded) -----------
    corpus = _load_corpus(args.corpus)
    print("[corpus] %d ascii chars (%s)"
          % (len(corpus),
             "capped" if len(corpus) == _CORPUS_CHAR_CAP else "full"))
    if args.permute_corpus:
        # Anti-cheat control: destroy sequential structure while keeping
        # the exact character distribution. Deterministic given --seed.
        perm_rng = np.random.default_rng(args.seed)
        chars = list(corpus)
        perm_rng.shuffle(chars)
        corpus = "".join(chars)
        print("[permute-corpus] anti-cheat control: "
              "corpus characters shuffled")
    tok = CharTokenizer(corpus)
    V = tok.vocab_size
    print("[vocab] V=%d" % V)

    # Dataset RNG is seeded; the SAME rng object is what we checkpoint
    # so a resumed run continues the exact shuffle stream.
    rng = np.random.default_rng(args.seed)
    X_np, y_np = make_seq_dataset(
        corpus, tok, seq_len=args.T, n_samples=args.n_samples, rng=rng,
    )
    if is_gpu:
        X = xp.asarray(X_np, dtype=xp.float32)
        y = xp.asarray(y_np, dtype=xp.int64)
    else:
        X = X_np
        y = y_np

    layer_sizes = [V] + hidden_list + [V]
    print("[arch] %s LIF  (backend: %s)"
          % (" -> ".join(str(s) for s in layer_sizes), backend_name))

    # --- Resume or fresh init ----------------------------------------
    ck = load_checkpoint(args.ckpt)
    loss_history: list = []

    if ck is not None:
        # Rebuild layers from saved host weight arrays.
        saved_w = ck["weights"]
        if len(saved_w) != len(layer_sizes) - 1:
            raise SystemExit(
                "[resume] checkpoint has %d weight arrays but current "
                "--hidden implies %d layers. Use a matching --hidden "
                "or a fresh --ckpt path."
                % (len(saved_w), len(layer_sizes) - 1))
        layers = []
        for li, W_host in enumerate(saved_w):
            W = xp.asarray(W_host) if is_gpu else np.asarray(W_host)
            layers.append(LIFLayerXP(
                W_in=W, n_post=layer_sizes[li + 1],
                threshold=1.0, leak=0.95,
            ))
        loss_history = list(ck["loss_history"])
        rng.bit_generator.state = ck["rng_state"]
        start = resume_epoch(ck)
        print("[resume] checkpoint found at epoch %d -> resuming at "
              "epoch %d (loss_history len=%d)"
              % (ck["epoch"], start, len(loss_history)))
    else:
        # Fresh init -- mirror train_shakespeare exactly: first layer
        # std=2.0 (one-hot drive needs strong weights), later std=0.5.
        layers = []
        for li in range(len(layer_sizes) - 1):
            n_pre = layer_sizes[li]
            n_post = layer_sizes[li + 1]
            std = 2.0 if li == 0 else 0.5
            W_np = rng.normal(0, std, (n_pre, n_post)).astype(np.float32)
            W = xp.asarray(W_np) if is_gpu else W_np
            layers.append(LIFLayerXP(
                W_in=W, n_post=n_post, threshold=1.0, leak=0.95,
            ))
        start = 0
        print("[fresh] no checkpoint -> training from epoch 0")

    if start > args.epochs:
        print("[done] checkpoint epoch %d already >= target epochs %d; "
              "nothing to do." % (start - 1, args.epochs))
        print("[summary] epochs_done=%d final_loss=%s ckpt=%s"
              % (start - 1,
                 ("%.4f" % loss_history[-1]) if loss_history else "n/a",
                 args.ckpt))
        return 0

    print("[train] T=%d batch=%d lr=%s n_samples=%d  epochs %d..%d"
          % (args.T, args.batch, args.lr, args.n_samples,
             start, args.epochs))

    def _host(W):
        """Device array -> host numpy (cupy.asnumpy if GPU, else pass)."""
        if is_gpu:
            import cupy as cp
            return cp.asnumpy(W)
        return np.asarray(W)

    def _save(epoch: int):
        host_w = [_host(layer.W_in) for layer in layers]
        save_checkpoint(args.ckpt, epoch, host_w,
                        rng.bit_generator.state, loss_history)

    def _run_one_epoch(epoch: int, B0: int) -> int:
        """Train a single epoch. Returns the batch size actually used
        (may be smaller than B0 if OOM forced a halving). Mirrors the
        train_shakespeare inner loop step-for-step."""
        B_cur = B0
        while True:
            try:
                perm = rng.permutation(args.n_samples)
                if is_gpu:
                    X_shuf = X[xp.asarray(perm)]
                    y_shuf = y[xp.asarray(perm)]
                else:
                    X_shuf = X[perm]
                    y_shuf = y[perm]

                n_batches = max(1, args.n_samples // B_cur)
                epoch_loss = 0.0
                for bi in range(n_batches):
                    s0 = bi * B_cur
                    s1 = min(s0 + B_cur, args.n_samples)
                    B = s1 - s0
                    x_batch = X_shuf[s0:s1].transpose(1, 0, 2)
                    target_batch = y_shuf[s0:s1, -1]

                    state = forward_unroll_xp(x_batch, layers, xp=xp)
                    logits = state["spikes"][-1].sum(axis=0)  # (B, V)
                    logits_np = logits.get() if is_gpu else logits

                    batch_loss = 0.0
                    tgt_np = (target_batch.get()
                              if is_gpu else target_batch)
                    output_grad_np = np.zeros(
                        (args.T, B, V), dtype=np.float32)
                    for s_idx in range(B):
                        loss_s = cross_entropy_loss_np(
                            logits_np[s_idx:s_idx + 1],
                            int(tgt_np[s_idx]))
                        batch_loss += loss_s
                        grad_s = softmax_grad_np(
                            logits_np[s_idx:s_idx + 1],
                            int(tgt_np[s_idx]))
                        output_grad_np[:, s_idx, :] = grad_s[0]

                    batch_loss /= B
                    epoch_loss += batch_loss

                    output_grad = (xp.asarray(output_grad_np)
                                   if is_gpu else output_grad_np)
                    weight_grads, _ = backward_unroll_xp(
                        x_batch, layers, state, output_grad, xp=xp)

                    for li, layer in enumerate(layers):
                        layer.W_in -= args.lr * weight_grads[li]

                epoch_loss /= n_batches
                loss_history.append(float(epoch_loss))
                return B_cur

            except (Exception, RuntimeError) as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    raise
                new_B = max(4, B_cur // 2)
                print("[OOM] halving batch %d-> %d (retrying epoch %d)"
                      % (B_cur, new_B, epoch))
                if is_gpu:
                    try:
                        import cupy as cp
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:  # noqa: BLE001
                        pass
                if new_B == B_cur:
                    # Can't shrink further -- re-raise so we don't spin.
                    raise
                B_cur = new_B

    # --- Training loop with kill-safety ------------------------------
    t0 = time.time()
    cur_batch = max(4, int(args.batch))
    last_done = start - 1
    try:
        for epoch in range(start, args.epochs + 1):
            cur_batch = _run_one_epoch(epoch, cur_batch)
            _save(epoch)
            last_done = epoch
            print("[epoch %d] loss=%.4f batch=%d (%.1fs) [ckpt saved]"
                  % (epoch, loss_history[-1], cur_batch,
                     time.time() - t0))
    except KeyboardInterrupt:
        # User pausing to game -- persist the last completed epoch and
        # exit cleanly. (The epoch in flight when Ctrl-C hit is NOT
        # counted; we save the last fully-completed one.)
        _save(last_done if last_done >= 0 else 0)
        print("[paused -> checkpoint saved, re-run to resume] "
              "(last completed epoch=%d)" % last_done)
        return 0

    final_loss = ("%.4f" % loss_history[-1]) if loss_history else "n/a"
    print("[done] trained through epoch %d in %.1fs"
          % (last_done, time.time() - t0))
    print("[summary] epochs_done=%d final_loss=%s ckpt=%s"
          % (last_done, final_loss, args.ckpt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
