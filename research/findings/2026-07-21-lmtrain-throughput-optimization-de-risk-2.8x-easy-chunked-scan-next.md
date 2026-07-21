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

## ⇒ CHUNKED SCAN IMPLEMENTED + VERIFIED (2026-07-21): ~30× total, correctness-gated
`research/runners/_lmtrain_chunked_scan.py` (`gate`, `speed`). CORRECTNESS GATE PASSED (controller-verified
independently): chunked_ssm is NUMERICALLY IDENTICAL to the Python loop — max|chunked-loop| = **4.77e-07** (<<1e-4)
across C∈{8,16,32} × seeds; block-level 2.38e-07. Speed at 67M (d1024/L16/T256):
| variant | tok/s | vs naive |
|---|---|---|
| naive fp32/B16/loop | 3043 | 1.0× |
| bf16/B64/loop (easy) | 7538 | 2.5× |
| bf16/B64/**chunked** | 49027 | 16.1× |
| bf16/B64/**chunked+compile** | **~90000** (verified 89811) | **~30×** |
| loop+compile | HANGS (>200s, T=256 unroll — never compiles) | — |
- **torch.compile WORKS on the chunked (53s one-time) where it HANGS on the loop.** B=64 is the sweet spot (best tok/s,
  6.7GB VRAM); B=128 = same throughput (compute-bound). C=16 (any of 8/16/32 exact — free knob).
- **⇒ ~30× total → 1-week 67M budget: naive 1.84B tokens → optimized ~55B tokens (~40× Chinchilla = SmolLM-level
  over-trained, genuinely fluent); OR Chinchilla-optimal (1.3B) in ~4 hours; OR a well-trained model in ~1 day (7.8B).**
  The training path is now near the reasonable optimization ceiling BEFORE the big compute. NO `sim/` edit.
