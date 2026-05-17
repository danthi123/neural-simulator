"""Generator-F kill-safe PyTorch BPTT trainer for sim.tiny_transformer
.TinyGPT. Return-dict contract mirrors the validated
scaled_subword_lm_train.train_subword_lm so the gate consumes it the
same way. Kill-safe atomic resume = the sim.train_checkpoint os.replace
idiom adapted for torch state (tmp + os.replace; the user games/
resumes). Self-contained at runtime (artifact = TinyGPT weights +
BPE JSON). ASCII only (Windows cp1252)."""
from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F


def train_tiny_gpt(
    seed: int = 42,
    corpus_path: str = "data/tinyshakespeare.txt",
    vocab_size: int = 512,
    d_model: int = 256,
    n_layer: int = 4,
    n_head: int = 4,
    block_size: int = 128,
    steps: int = 12000,
    batch_size: int = 64,
    lr: float = 3e-4,
    ckpt_path: str = "research/findings/raw/g11_bg/gen_f.ckpt",
    bpe_path: str = "research/findings/raw/g11_bg/gen_f.bpe.json",
    device: str = "auto",
    print_every: int = 500,
    verbose: bool = True,
):
    from pathlib import Path
    from sim.bpe_tokenizer import BPETokenizer
    from sim.tiny_transformer import TinyGPT

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    corpus = Path(corpus_path).read_text(encoding="utf-8")
    if os.path.exists(bpe_path):
        tok = BPETokenizer.load(bpe_path)
        if verbose:
            print("[gen-f] loaded cached BPE (%d) %s"
                  % (tok.vocab_size, bpe_path), flush=True)
    else:
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=vocab_size)
        Path(bpe_path).parent.mkdir(parents=True, exist_ok=True)
        tok.save(bpe_path)
        if verbose:
            print("[gen-f] trained BPE (%d) -> %s"
                  % (tok.vocab_size, bpe_path), flush=True)
    V = tok.vocab_size

    data = torch.tensor(tok.encode(corpus), dtype=torch.long)
    if data.numel() < block_size + 2:
        raise ValueError("corpus too short for block_size")
    data = data.to(device)

    model = TinyGPT(vocab_size=V, d_model=d_model, n_layer=n_layer,
                    n_head=n_head, block_size=block_size,
                    dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, steps))

    start_step = 0
    loss_history = []
    ckf = ckpt_path + ".pt"
    if os.path.exists(ckf):
        try:
            st = torch.load(ckf, map_location=device)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optim"])
            sched.load_state_dict(st["sched"])
            start_step = int(st["step"])
            loss_history = list(st.get("loss_history", []))
            if st.get("torch_rng") is not None:
                torch.set_rng_state(
                    st["torch_rng"].to("cpu")
                    if hasattr(st["torch_rng"], "to")
                    else st["torch_rng"])
            if verbose:
                print("[gen-f] RESUMED from step %d (%s)"
                      % (start_step, ckf), flush=True)
        except Exception as e:
            if verbose:
                print("[gen-f] ckpt unreadable (%s) -- fresh: %s"
                      % (ckf, e), flush=True)
            start_step = 0
            loss_history = []

    n = data.numel()
    cur_bs = batch_size
    t0 = time.time()

    def _rand_batch(bs):
        ix = torch.randint(0, n - block_size - 1, (bs,),
                           device=device)
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
        return x, y

    def _flush(step):
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = ckf + ".tmp"
        torch.save({"model": model.state_dict(),
                    "optim": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "step": step,
                    "loss_history": loss_history,
                    "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, ckf)

    if verbose:
        print("[gen-f] arch V=%d d=%d L=%d H=%d blk=%d dev=%s "
              "steps=%d bs=%d lr=%s"
              % (V, d_model, n_layer, n_head, block_size, device,
                 steps, batch_size, lr), flush=True)

    model.train()
    interrupted = False
    try:
        step = start_step
        while step < steps:
            try:
                x, y = _rand_batch(cur_bs)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, V), y.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                opt.step()
                sched.step()
                loss_history.append(float(loss.item()))
                step += 1
                if verbose and step % print_every == 0:
                    print("[gen-f] step %d/%d loss=%.4f (%.0fs)"
                          % (step, steps, loss_history[-1],
                             time.time() - t0), flush=True)
                if step % print_every == 0:
                    _flush(step)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() and cur_bs > 1:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    cur_bs = max(1, cur_bs // 2)
                    if verbose:
                        print("[gen-f] OOM -> batch halved to %d"
                              % cur_bs, flush=True)
                    continue
                raise
    except KeyboardInterrupt:
        interrupted = True
        _flush(step)
        if verbose:
            print("[gen-f] INTERRUPTED -> checkpoint flushed; "
                  "re-run to resume.", flush=True)

    _flush(min(step, steps))
    model.eval()
    init_loss = loss_history[0] if loss_history else None
    fin_loss = loss_history[-1] if loss_history else None
    if verbose and loss_history:
        print("[gen-f] done (%.0fs) init=%.4f final=%.4f"
              % (time.time() - t0, init_loss, fin_loss), flush=True)
    return {
        "loss_history": loss_history,
        "initial_loss": init_loss,
        "final_loss": fin_loss,
        "vocab_size": V,
        "n_layer": n_layer,
        "device": device,
        "interrupted": interrupted,
        "bpe_path": bpe_path,
        "ckpt_path": ckpt_path,
        "_model": model,
        "_tok": tok,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus-path", type=str,
                    default="data/tinyshakespeare.txt")
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-path", type=str,
                    default="research/findings/raw/g11_bg/gen_f.ckpt")
    ap.add_argument("--bpe-path", type=str,
                    default="research/findings/raw/g11_bg/gen_f.bpe.json")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--print-every", type=int, default=500)
    a = ap.parse_args()
    train_tiny_gpt(
        seed=a.seed, corpus_path=a.corpus_path,
        vocab_size=a.vocab_size, d_model=a.d_model,
        n_layer=a.n_layer, n_head=a.n_head,
        block_size=a.block_size, steps=a.steps,
        batch_size=a.batch_size, lr=a.lr, ckpt_path=a.ckpt_path,
        bpe_path=a.bpe_path, device=a.device,
        print_every=a.print_every, verbose=True)


if __name__ == "__main__":
    raise SystemExit(main())
