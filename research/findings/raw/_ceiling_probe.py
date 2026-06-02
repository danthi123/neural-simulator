"""Single-seed DECISIVE ceiling probe for the subword spiking LM -- the owner's "3090 generative ceiling".

Generator-S (2026-05-17) tested a subword spiking LM (surrogate-grad BPTT) on real TinyStories at hidden
256,256 -> honest NEGATIVE (held-out ppl 117K-388K, token-soup, worse than random). The owner's note: VRAM is
NOT the limit (compute/speed is), so push hidden width up. This probe extends Generator-S to bigger hidden
(using VRAM headroom) and measures whether SCALE rescues the spiking architecture, vs the Generator-F
transformer reference (held-out ppl ~6.1, coherent) and the uniform-random floor (vocab_size).

Cheap-first / staged: single seed first. If a big model STILL produces token-soup (ppl >> vocab), the ceiling
is NEGATIVE and no 3-seed run is needed for a clear negative (same logic as the 50M-cosine smoke). If it BEATS
random (ppl < vocab) or approaches the transformer, scale further + run the full 3-seed gate. Reuse-by-import
(train_subword_lm + _heldout_nll + generate + corpus_fetch + gate_core.perplexity); no protected-module change;
no new autograd. GPU/CuPy.

  python -m research.findings.raw._ceiling_probe --hidden 2048,2048 --vocab 1024 --T 48 --epochs 30 --n-train 12000
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--hidden", type=str, default="2048,2048")
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--max-corpus-mb", type=int, default=8)
    ap.add_argument("--gen-tokens", type=int, default=60)
    ap.add_argument("--eval-positions", type=int, default=400)
    ap.add_argument("--tag", type=str, default="probe")
    a = ap.parse_args()
    hidden = [int(x) for x in a.hidden.split(",") if x.strip()]

    from research.runners.corpus_fetch import fetch_corpus, split_corpus
    from research.runners.scaled_subword_lm_train import train_subword_lm
    from research.runners.subword_lm_generate import generate
    from research.runners.subword_lm_gate_core import perplexity
    from research.runners.subword_lm_gate import _heldout_nll
    from sim.bptt_snn_gpu import _get_backend

    xp, is_gpu = _get_backend()
    cinfo = fetch_corpus(name="tinystories", max_bytes=int(a.max_corpus_mb) * 1_000_000)
    train_text, heldout_text = split_corpus(cinfo["text"], heldout_frac=0.1)
    print(f"[ceiling-probe:{a.tag}] backend={'GPU' if is_gpu else 'CPU'} corpus={cinfo['corpus_used']} "
          f"degraded={cinfo['degraded']} n_chars={cinfo['n_chars']} | V={a.vocab} hidden={hidden} T={a.T} "
          f"epochs={a.epochs} n_train={a.n_train} batch={a.batch}", flush=True)
    cdir = Path(f"research/findings/raw/_ceiling_ckpt_{a.tag}"); cdir.mkdir(parents=True, exist_ok=True)
    tr_file = str(cdir / "train.txt"); Path(tr_file).write_text(train_text, encoding="utf-8")

    t0 = time.time()
    rr = train_subword_lm(seed=a.seed, corpus_path=tr_file, vocab_size=a.vocab, hidden_layers=hidden,
                          T=a.T, epochs=a.epochs, batch_size=a.batch, lr=a.lr, n_train_samples=a.n_train,
                          ckpt_path=str(cdir / f"real_s{a.seed}.npz"),
                          bpe_path=str(cdir / f"real_s{a.seed}.bpe.json"), backend="auto", verbose=True)
    tok, lay = rr["_tok"], rr["_layers"]
    V = tok.vocab_size
    train_min = (time.time() - t0) / 60.0

    nll = _heldout_nll(lay, tok, heldout_text, a.T, xp, a.eval_positions)
    ppl = perplexity(nll)
    grng = np.random.default_rng(a.seed * 13 + 5)
    prompt = " ".join(heldout_text.split()[:8])
    gen_ids, gen_txt = generate(lay, tok, prompt, a.gen_tokens, a.T, xp=xp, rng=grng, temperature=1.0)

    n_params = sum(int(np.prod(l.W_in.shape)) for l in lay)
    beats = ppl < V
    print(f"\n[RESULT seed {a.seed}] params={n_params / 1e6:.1f}M  train_min={train_min:.1f}", flush=True)
    print(f"  held-out perplexity = {ppl:.1f}", flush=True)
    print(f"  uniform-random floor (vocab {V}) = {V}  -> "
          f"{'BEATS random' if beats else 'WORSE than random (token-soup)'} ({V / max(ppl, 1e-9):.3f}x)", flush=True)
    print(f"  transformer reference (Generator-F) = 6.1  -> gap {ppl / 6.1:.0f}x worse", flush=True)
    print(f"  prompt:    {prompt!r}", flush=True)
    print(f"  generated: {gen_txt!r}", flush=True)
    print(f"  VERDICT: {'BEATS-RANDOM (scale up + run 3-seed gate)' if beats else 'TOKEN-SOUP (ceiling NEGATIVE at this scale)'}",
          flush=True)


if __name__ == "__main__":
    main()
