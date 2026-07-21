# LM-training performance de-risk — the naive path is FAR from optimal (~8-28× headroom); realize it BEFORE the big compute

**2026-07-21.** Owner (before committing training compute): confirm the training is as optimized as reasonable, to
maximize compute value. It is NOT yet — measured on the 3090.

## Real 3090 throughput by model size (fp32, B=16, Python-loop recurrence — the NAIVE path)
| d/L | params | tok/s | VRAM | ~1wk tokens (tok/param) |
|---|---|---|---|---|
| 768/12 | 34M | 4439 | 1.9GB | 2.7B (79) |
| 1024/16 | 67M | 2789-3072 | 3.2GB | 1.7B (25, Chinchilla-optimal) |
| 1280/20 | 119M | 1837 | 4.9GB | 1.1B (9) |
| 1536/24 | 195M | 1173 | 7.2GB | 0.7B |
VRAM is NOT the constraint (2-7GB of 24GB) — big headroom for batch-scaling.

## Optimization levers (67M, measured)
| config | tok/s | vs baseline |
|---|---|---|
| fp32, B=16 (baseline) | 3072 | 1.0× |
| + bf16 | 4267 | 1.4× |
| + bf16, B=64 | 7537 | 2.5× |
| + bf16, B=128 (VRAM 15.6GB) | 8609 | **2.8×** |
| + torch.compile | **HANGS** (unrolls the 256-step Python loop) | — |
- **Easy wins (bf16 + batch-scaling into the free VRAM): ~2.8×, zero algorithmic risk.**
- **torch.compile HANGS on the Python-loop recurrence** → the loop is the bottleneck AND torch.compile can't fuse it.
  ⇒ the big lever is a **CHUNKED/PARALLEL SCAN** (replace the 256-step loop with a chunked matmul — the standard SSM/
  linear-attention optimization, Mamba/RWKV/GLA; the DIAGONAL decay makes it a straightforward chunked form; ~3-10× on
  the recurrence + makes torch.compile work). Correctness gate: chunked output MUST equal the Python-loop output.

## ⇒ ~8-28× total headroom transforms the compute value
The 67M/1-week budget: 1.7B tokens (Chinchilla-MINIMAL) naive → ~5B (well-trained) with easy wins → ~15-50B (SmolLM-level
over-trained, genuinely fluent) with the chunked scan. SAME wall-clock, categorically better model. **⇒ realize the
optimizations (bf16 + batch + chunked scan) BEFORE committing the big compute.** Probes:
`_lmtrain_throughput_probe.py`, `_lmtrain_optim_probe.py`.
