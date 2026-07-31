---
type: plan
status: live
date: 2026-05-16
---

# Generator Increment 3 — scaled capacity scan, checkpointable/resumable

> **For Claude:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. User-directed (2026-05-16): "scale to whatever max reasonably fits my hardware; runnable so I can pause without killing it for gaming." Autonomous design calls documented below.

**Goal:** Honestly answer "is the generator's failure at PoC scale a *capacity* problem?" by training the largest student that reasonably fits a 24 GB RTX 3090, with **per-epoch checkpoint + auto-resume** so it can be killed anytime (to free the GPU for gaming) and continued by re-running the same command.

**Honest framing:** This is a **transparently-reported capacity scan**, not config-cranking to retro-pass the failed Increment-2 distillation gate. The reported metric is the same anti-cheat one (held-out loss + REAL-vs-PERMUTED control). Outcome is reported whichever way it lands. If a maxed local student still fails the permuted control, that is a profound, honest finding: self-contained fluent generation is likely out of reach on a single 3090 under the no-cheating/local-only constraints — which is decision-relevant for the user.

## Autonomous design calls (rationale)

1. **Stay char-level (minimal-change capacity test).** The cleanest
   controlled test of "is it capacity?" holds tokenizer/metric fixed
   and only scales the model + training. A word/subword model is a
   larger architectural change → a *separate later* increment if this
   scan says capacity helps. Same anti-cheat metric as Inc-1/2 (no
   moving goalposts).
2. **Scale = deeper + wider + longer + more data, within 24 GB.**
   BPTT stores per-timestep activations, so memory ≈
   batch·T·Σlayer·layers. Target config (parameterised, OOM-safe):
   hidden_layers `[512, 512, 512]`, T=96, batch=64, corpus = all
   local English available (distilled ∪ repo findings ∪ docs),
   epochs = large (e.g. 400) with per-epoch checkpoint. The runner
   **auto-halves batch on CUDA OOM** and records the config actually
   used. "Max reasonable" = biggest that fits with headroom and
   trains in resumable chunks, not the biggest that technically loads.
3. **Pause = checkpoint + auto-resume, not a hung pause state.**
   Per-epoch atomic checkpoint (`.npz`: weights + rng + epoch +
   loss_history) to a fixed path. On start: if checkpoint exists,
   load and continue from the next epoch (deterministic). Killing
   the process (Ctrl-C / closing it to game) frees all GPU memory;
   re-running the identical command resumes with ≤1 epoch lost.
   Also trap KeyboardInterrupt → checkpoint-then-exit for a clean
   manual pause. Robust + simple; no special daemon.

## Tasks (TDD; pure checkpoint logic is CPU-testable)

### Task 1: Checkpoint/resume core (pure, CPU TDD)

**Files:** Create `sim/train_checkpoint.py`; Test `tests/test_train_checkpoint.py`

Pure functions (no GPU): `save_checkpoint(path, epoch, weights, rng_state, loss_history)` (atomic: write `path+'.tmp'` then `os.replace`), `load_checkpoint(path) -> dict | None` (None if absent), `resume_epoch(ckpt) -> int`. Weights are a list of numpy arrays (the SNN layer W matrices) — backend-agnostic (cupy arrays → `to_host` before save; numpy passthrough).

Tests: save→load round-trips weights/epoch/rng exactly; load missing→None; atomic (no partial file on simulated mid-write); resume_epoch returns ckpt['epoch']+1; deterministic RNG continuation (same rng_state → same next draws).

### Task 2: Scaled, resumable trainer

**Files:** Create `research/runners/scaled_generator_train.py`

Reuses Inc-1 infra (`local_corpus`, `char_tokenizer`, `bptt_snn_gpu` LIF layers + forward/backward unroll) — DRY, do NOT reimplement BPTT. Adds: multi-layer config from CLI (`--hidden 512,512,512 --T 96 --batch 64 --epochs 400 --corpus all`), per-epoch `save_checkpoint` to `research/findings/raw/g11_bg/scaled_gen.ckpt.npz`, auto-resume from it on start, CUDA-OOM→halve-batch retry loop, KeyboardInterrupt→checkpoint+exit. Corpus `all` = `clean_corpus(load_local_corpus() + distill_corpus + docs)`. Prints per-epoch loss + `[ckpt saved epoch N]`. No unit test (orchestration; validated by the gate + the checkpoint unit tests).

### Task 3: Resumability smoke (CPU-fast)

**Files:** Create `tests/test_scaled_resume_smoke.py` (or a tiny scripted check)

Run trainer for 2 epochs (tiny config, CPU/quick), kill, re-run for 2 more → assert it resumed at epoch 3 (not 1) and the checkpoint epoch advanced. Proves the pause/resume guarantee the user asked for, cheaply, before the long run.

### Task 4: The honest capacity-scan gate (long, resumable, GPU)

**Files:** Create `research/runners/scaled_capacity_gate.py`

Train the maxed student to convergence/epoch-budget on the full local corpus (resumable — run in chunks around gaming). Then the **same anti-cheat metric**: held-out loss of the scaled student vs a permuted-corpus control at the same scaled config. Report: REAL end loss, PERMUTED end loss, % below, and vs the Inc-1 tiny baseline. **Gate question (transparent, pre-registered):** does the *scaled* student beat its permuted control by a real margin (≥10%)? PASS ⇒ capacity *was* the bottleneck (honest, important). FAIL ⇒ even a maxed local char-SNN cannot learn robust structure → self-contained local fluent generation is out of reach on this hardware (honest, decision-relevant). Either way: propagate honestly (findings + capability_status), no spin, no goalpost-moving.

## Notes for the executor

- **Anti-cheat:** the metric is fixed (held-out loss + permuted control). Do NOT change the metric or the ≥10% bar after seeing results. A maxed-and-still-FAIL is a real finding to report, not iterate away.
- **Resumability is a user requirement, not optional** — Task 3 must prove kill→re-run resumes before Task 4's long run starts.
- OOM handling must be real (catch `cupy.cuda.memory.OutOfMemoryError`/RuntimeError, halve batch, record actual config) — never silently shrink the model without logging it.
- DRY: reuse Inc-1 BPTT/corpus/tokenizer. YAGNI: no word tokenizer, no distillation here — pure capacity scan.
- ASCII-only prints (Windows cp1252).
- Long run: launch via background/`run_in_background`; the user kills it to game and I (or they) re-run to resume.
