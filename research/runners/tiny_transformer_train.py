"""Generator-F kill-safe PyTorch BPTT trainer for sim.tiny_transformer
.TinyGPT. Return-dict contract mirrors the validated
scaled_subword_lm_train.train_subword_lm so the gate consumes it the
same way. Kill-safe atomic resume = the sim.train_checkpoint os.replace
idiom adapted for torch state (tmp + os.replace; the user games/
resumes). Self-contained at runtime (artifact = TinyGPT weights +
BPE JSON). ASCII only (Windows cp1252)."""
from __future__ import annotations

import argparse
import math
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
    dropout: float = 0.0,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    heldout_path: str | None = None,
    heldout_every: int = 0,
    bpe_fit_chars: int = 8_000_000,
):
    """Train a TinyGPT generator (kill-safe atomic resume).

    Regularization / schedule knobs (ADDITIVE; defaults reproduce the
    original behaviour for the 3.4M toy):
      - ``dropout``       attn+MLP+embedding dropout (0.0 = off, as before).
                          A bigger model on a SMALL corpus MUST set this
                          (e.g. 0.1-0.2) or it memorizes the train set
                          (train ppl -> ~1, held-out ppl blows up).
      - ``weight_decay``  AdamW weight decay (0.01 = the prior default).
      - ``warmup_steps``  linear LR warmup 0->lr over this many steps,
                          then cosine-decay to 0 over the REMAINING steps
                          (0 = pure cosine over all steps, as before).
      - ``heldout_path``  optional text file; if set with ``heldout_every>0``
                          the train loop prints a held-out token-NLL ppl
                          every ``heldout_every`` steps (a generalization
                          probe so overfitting is visible DURING training,
                          not only at the downstream gate).
      - ``bpe_fit_chars`` fit the BPE on the first N chars of the corpus
                          (default 8M). BPE merges are frequency-stable, so a
                          sample yields the same vocab FAR faster on a large
                          corpus (the pure-Python fit cost scales with the
                          unique-word count); the model still TRAINS on the full
                          corpus. 0 = fit on the full corpus (the prior
                          behaviour). For corpora <= the cap this is a no-op.
    """
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
        # BPE merges are frequency-statistics: fitting on an ~8M-char SAMPLE
        # yields the same vocab as the full corpus but in minutes not hours
        # (the pure-Python fit cost scales with unique-word count x merges, and
        # a large-vocab corpus like SimpleWiki has far more unique words than
        # TinyStories). The model still TRAINS on the full corpus below.
        # corpus[:bpe_fit_chars] is a no-op for corpora <= the cap; 0 = full.
        fit_text = corpus[:bpe_fit_chars] if bpe_fit_chars else corpus
        tok.train(fit_text, vocab_size=vocab_size)
        Path(bpe_path).parent.mkdir(parents=True, exist_ok=True)
        tok.save(bpe_path)
        if verbose:
            print("[gen-f] trained BPE (%d) on %d/%d chars -> %s"
                  % (tok.vocab_size, len(fit_text), len(corpus), bpe_path), flush=True)
    V = tok.vocab_size

    def _encode_cached(text, cache_path):
        # Encoding a large corpus with the pure-Python per-word BPE apply is the
        # second big pre-training cost (after the BPE fit). Two speedups, both
        # OUTPUT-IDENTICAL to tok.encode(text): (1) memoize per UNIQUE word --
        # natural-language corpora repeat words heavily, so this cuts the
        # BPE-applies from O(total words) down to O(unique words); (2) cache the
        # token array to disk so pause/resume skips the re-encode entirely.
        if os.path.exists(cache_path):
            arr = np.load(cache_path)
            if verbose:
                print("[gen-f] loaded cached tokens (%d) %s" % (arr.size, cache_path), flush=True)
            return arr
        wc, ids = {}, []
        for w in text.split():
            e = wc.get(w)
            if e is None:
                e = [tok._sym_to_id.get(s, 0) for s in tok._encode_word(w)]
                wc[w] = e
            ids.extend(e)
        arr = np.asarray(ids, dtype=np.int32)
        np.save(cache_path, arr)
        if verbose:
            print("[gen-f] encoded+cached %d tokens (%d unique words) -> %s"
                  % (arr.size, len(wc), cache_path), flush=True)
        return arr

    data = torch.tensor(_encode_cached(corpus, bpe_path + ".traintokens.npy"), dtype=torch.long)
    if data.numel() < block_size + 2:
        raise ValueError("corpus too short for block_size")
    data = data.to(device)

    # Optional held-out tensor for an in-loop generalization (overfit) probe.
    ho_data = None
    if heldout_path and int(heldout_every) > 0 and os.path.exists(heldout_path):
        try:
            ho_text = Path(heldout_path).read_text(encoding="utf-8")
            ho_ids = _encode_cached(ho_text, bpe_path + ".heldtokens.npy")
            if len(ho_ids) > block_size + 2:
                ho_data = torch.tensor(ho_ids, dtype=torch.long, device=device)
                if verbose:
                    print("[gen-f] held-out probe: %d tokens from %s (every %d steps)"
                          % (ho_data.numel(), heldout_path, int(heldout_every)),
                          flush=True)
        except Exception as e:
            if verbose:
                print("[gen-f] held-out probe disabled (%s)" % e, flush=True)

    @torch.no_grad()
    def _heldout_ppl(n_windows=24):
        """Mean per-token held-out NLL -> ppl over n_windows random blocks."""
        if ho_data is None:
            return None
        was_training = model.training
        model.eval()
        nho = ho_data.numel()
        gg = torch.Generator(device="cpu").manual_seed(seed + 777)
        tot_nll = 0.0; tot_tok = 0
        for _ in range(n_windows):
            i = int(torch.randint(0, nho - block_size - 1, (1,), generator=gg).item())
            xb = ho_data[i:i + block_size][None, :]
            yb = ho_data[i + 1:i + 1 + block_size][None, :]
            lg = model(xb)
            nll = F.cross_entropy(lg.reshape(-1, V), yb.reshape(-1),
                                  reduction="sum")
            tot_nll += float(nll.item()); tot_tok += yb.numel()
        if was_training:
            model.train()
        import math as _m
        return _m.exp(tot_nll / max(1, tot_tok))

    model = TinyGPT(vocab_size=V, d_model=d_model, n_layer=n_layer,
                    n_head=n_head, block_size=block_size,
                    dropout=float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(weight_decay))
    # LR schedule: optional linear warmup (0->lr over warmup_steps) then
    # cosine decay to 0 over the remaining steps. warmup_steps=0 collapses
    # to the original pure-cosine-over-all-steps schedule.
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0 and warmup_steps < steps:
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0,
            total_iters=warmup_steps)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, steps - warmup_steps))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warm, cos], milestones=[warmup_steps])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, steps))

    start_step = 0
    loss_history = []
    heldout_history = []
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
                    cur_lr = opt.param_groups[0]["lr"]
                    print("[gen-f] step %d/%d loss=%.4f lr=%.2e (%.0fs)"
                          % (step, steps, loss_history[-1], cur_lr,
                             time.time() - t0), flush=True)
                if ho_data is not None and int(heldout_every) > 0 \
                        and step % int(heldout_every) == 0:
                    hp = _heldout_ppl()
                    heldout_history.append((step, hp))
                    if verbose and hp is not None:
                        print("[gen-f]   [held-out] step %d train_ppl~%.2f "
                              "HELD-OUT_ppl=%.2f"
                              % (step, math.exp(min(20.0, loss_history[-1])), hp),
                              flush=True)
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
    final_heldout_ppl = _heldout_ppl(n_windows=48) if ho_data is not None else None
    if verbose and loss_history:
        msg = ("[gen-f] done (%.0fs) init=%.4f final=%.4f"
               % (time.time() - t0, init_loss, fin_loss))
        if final_heldout_ppl is not None:
            msg += " HELD-OUT_ppl=%.2f" % final_heldout_ppl
        print(msg, flush=True)
    return {
        "loss_history": loss_history,
        "heldout_history": heldout_history,
        "final_heldout_ppl": final_heldout_ppl,
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
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="attn+MLP+emb dropout (set 0.1-0.2 for a big "
                         "model on a small corpus to avoid memorization)")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="linear LR warmup 0->lr over N steps, then cosine")
    ap.add_argument("--heldout-path", type=str, default=None,
                    help="optional held-out text for an in-loop overfit probe")
    ap.add_argument("--heldout-every", type=int, default=0,
                    help="print held-out ppl every N steps (needs --heldout-path)")
    a = ap.parse_args()
    train_tiny_gpt(
        seed=a.seed, corpus_path=a.corpus_path,
        vocab_size=a.vocab_size, d_model=a.d_model,
        n_layer=a.n_layer, n_head=a.n_head,
        block_size=a.block_size, steps=a.steps,
        batch_size=a.batch_size, lr=a.lr, ckpt_path=a.ckpt_path,
        bpe_path=a.bpe_path, device=a.device,
        print_every=a.print_every, dropout=a.dropout,
        weight_decay=a.weight_decay, warmup_steps=a.warmup_steps,
        heldout_path=a.heldout_path, heldout_every=a.heldout_every,
        verbose=True)


if __name__ == "__main__":
    raise SystemExit(main())
