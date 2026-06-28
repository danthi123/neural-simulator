# 100M Generative C2 Run — driving guide (the 3-day GPU burn)

The decisive **C2 generative grow-without-forget** experiment at ~88.6M parameters on SimpleWiki.
This is the first *clean* run of the scale-up: the old "30M scale wall" turned out to be two bugs —
a broken fine-tune learning rate (3e-4 instead of 1e-5) and an overfit base model (40k steps on the
tiny 8 MB TinyStories) — both fixed in this recipe. ~3 days on the 3090. **Resumable:** kill it
anytime, re-run the same command, it continues from the last checkpoint.

## The command (start / resume — identical)
```powershell
$env:SIM_BACKEND='cupy'
python -m research.runners._genseq_C2_scaleup_runner --d-model 768 --n-layers 12 --n-heads 12 `
  --vocab-size 2048 --block-size 512 --batch-size 16 --ft-batch 8 --steps 450000 `
  --dropout 0.1 --weight-decay 0.1 --warmup-steps 1000 --heldout-every 1000 --corpus simplewiki `
  --out research/findings/raw/_genseq_C2_scaleup_100M.json `
  --run-dir research/findings/raw/c2_scaleup_100M
```
It was launched detached (survives a closed terminal); live log at `research/findings/raw/_c2_100M_live.log`.

## Monitor (anytime, safe — no GPU contention)
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # ~80-100% once training
Get-Content research\findings\raw\_c2_100M_live.log -Tail 20                # heldout ppl every 1000 steps
```
The first ~10–30 min is a one-time, CPU-bound BPE tokenization of the 127M-char corpus (GPU sits at
~1%, model loaded — this is normal). Then GPU utilization jumps to ~80–100% and the held-out
perplexity starts descending (target ~6–10).

## Pause / resume / stop (e.g. to game)
- **Pause / free the GPU:**
  ```powershell
  Get-Process python | Stop-Process -Force
  ```
  It checkpoints periodically and flushes on interrupt, so you lose at most a little progress.
- **Resume:** re-run the exact command above — it auto-continues from the last checkpoint, and skips
  any already-completed stage via its `.DONE.json` marker in the run-dir.
- Safe to kill/resume as often as you like; the run-dir (`research/findings/raw/c2_scaleup_100M`) holds
  all state (checkpoint `genf.ckpt.pt`, tokenizer `genf.bpe.json`, the stage markers).

## What you get
Three stages, each resumable via its marker:
1. **TRAIN** the 88.6M Gen-F on SimpleWiki (~the bulk of the 3 days) → `genf.ckpt.pt`.
2. **C1** — the on-bridge spiking-consolidation check (held-out ppl on the spiking forward ≈ the ANN, ppl_ratio ≈ 1.0).
3. **C2** — the grow-without-forget loop (fine-tune to a new in-band shift with generative self-replay; measure retention of the original).

The decisive output lands in `research/findings/raw/_genseq_C2_scaleup_100M.json`: whether the corrected
recipe holds at 100M. A **GO** retires the "scale wall"; a clean **NEGATIVE-with-the-fixed-recipe** is
still a far stronger capacity finding than the two contaminated priors. Either way you get the project's
largest trained spiking-consolidatable generative model.

When I'm back (usage reset), I'll read the C2 verdict + the trained model's perplexity and write it up.
