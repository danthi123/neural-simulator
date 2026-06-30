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

**Authoritative progress — use this (survives reboots).** The detached run's live log can get
*orphaned* by a PC reboot (its stdout redirect breaks), so the log may show a stale step. The
checkpoint is the source of truth — read the real step + recent loss anytime:
```powershell
python -c "import torch; d=torch.load(r'research/findings/raw/c2_scaleup_100M/genf.ckpt.pt', map_location='cpu', weights_only=True); print('step', d['step'], '/ 450000  (', round(100*d['step']/450000,1), '%)  recent loss', [round(float(x),3) for x in d['loss_history'][-5:]])"
```
Confirm it's actively training — GPU busy AND the checkpoint timestamp advancing (it saves every ~500 steps):
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # ~80-100% once training
Get-Item research\findings\raw\c2_scaleup_100M\genf.ckpt.pt | Select-Object LastWriteTime
```

**Live log (secondary — may be stale after a reboot):**
```powershell
Get-Content research\findings\raw\_c2_100M_live.log -Tail 20                # step / held-out ppl stream
```
On a *fresh* start the first ~10–30 min is a one-time CPU-bound BPE tokenization (GPU ~1%, normal);
then GPU jumps to ~80–100% and held-out perplexity descends (target ~6–10). On a *resume* the BPE +
tokens load from cache instantly, so training re-engages within ~1 min.

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
