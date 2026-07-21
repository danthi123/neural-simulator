# Autonomous, incremental, resumable LM-training workflow — design + de-risk plan (2026-07-21)

Owner directive: plan + de-risk a workflow that trains the substrate-native WKV language cortex **incrementally** —
train → checkpoint → benchmark/test → continue → repeat — fully autonomously, resumable across sessions (build on
checkpoints as compute allows), pausable on demand, with the owner able to check in on progress. Toward "converse like a
small LLM about anything." This is the gap#1 SCALE lever (gap#1's mechanism is investigation-complete; closure = feeding
it real data at real scale).

## 0. Closeness assessment (the gate for committing compute)
- **Gaps that matter for BASE fluency are resolved as MECHANISMS.** #1 (open generation) = complete-as-mechanism +
  characterized (the scale wall + the tractable scale path); #2 (open-ended) and #3 (multi-fact synthesis) largely
  COLLAPSE INTO #1 at scale (a fluent generative LM is open-ended + synthesizes by construction). #3-multi-referent is
  closed-spiking already. ⇒ **training is UNBLOCKED by the remaining gap-work.**
- **The remaining gap-work is largely ORTHOGONAL to the base training** — it is the MEMORY / LEARNING-FROM-CONVERSATION
  system (gap#5 emergent-DG episodic memory [chain demonstrated this session], gap#4-unsupervised online learning,
  gap#2 emergence-bar binder). These give #4 (learning from YOUR chats) — a scaled-corpus model does NOT get that from
  scale. So: TRAIN (base fluency) and continue the GAP-WORK (memory/online-learning) IN PARALLEL; neither blocks the other.
- **Fluid/natural abstaining (replace the hard binary moat):** a MODERATE build on existing validated pieces, NOT a deep
  frontier — reasonably close (weeks, not months). We HAVE a calibrated confidence signal (the Bogacz-Brown familiarity
  gate + composer match-confidence, 6-seed GO). Today it's THRESHOLDED (binary gate → "I don't know"). Fluid = map it to
  a GRADED HEDGING LEVEL that conditions generation (high→assertive; medium→"I think…"; low→"I'm not certain, but…";
  very-low→graceful soft-abstain), via a confidence-conditioned generation fine-tune (the same "format fine-tune" lever
  as EMERGE-57 / the RA render tune). For GROUNDED facts the gate is calibrated → close. For OPEN generation, the LM's
  own token entropy/logprob gives a usable (imperfect) confidence → graded hedging is achievable; PERFECT calibration
  (never confidently-wrong) is the field's open hallucination problem, unsolved for everyone — so the honest target is
  "graded hedging, much better than binary," not "perfect uncertainty-awareness."
- **Recommendation:** de-risk the workflow NOW (cheap, low-compute), start the first real training increment when you
  commit compute; run the fluid-abstain + gap-work in parallel. Don't gate training on "all gaps closed" — they're
  orthogonal.

## 1. Architecture (reuse existing patterns)
Precedent to adapt: `research/runners/develop_run.py` + `scripts/develop.ps1` + `DEVELOP_RUN.md` (the resumable,
pause-sentinel, per-increment-bundle, Monitored 3-day self-driven develop-run launcher) — the SAME shape, retargeted
from the artificial-life day-loop to the LM train-loop. Plus `sim/lineage.py` `BridgeLineage` (atomic save + history +
metadata) for the checkpoint lineage.

**Components:**
1. **Corpus pipeline** (`lm_corpus.py`): download + tokenize a broad corpus (FineWeb-Edu / DCLM per the gap#1 gate;
   start with a curated slice), a STABLE tokenizer (BPE — reuse `sim/bpe_tokenizer.py`, frozen vocab), and a STREAMING
   loader with a persistent read-CURSOR (so resume continues from the exact token position, not re-shuffling seen data).
2. **Model:** the multi-layer WKV (`_emerge_wkv_lm_derisk.py` `--n-layers` refactor from this session), scaled via
   `--d-model`/`--n-layers`; a config file pins the architecture per lineage. Start ~150M for the first real run.
3. **Training loop:** BPTT-on-GPU (the pragmatic scaffold; the biological LOCAL-rule readout is a separate track —
   `_gap_grounded_wkv_local_readout.py` — not on the critical path for the scale run). AdamW + cosine LR + warmup.
   Gradient checkpointing (VRAM) when the model needs it.
4. **Checkpointing (THE load-bearing piece):** every N steps atomically save {model weights, optimizer state, LR-scheduler
   state, data read-cursor, RNG states (torch+numpy+python), step count, tokens-seen}. A resume MUST continue the EXACT
   trajectory (verified — de-risk #1). Keep a rolling history + a "best-by-val" pointer.
5. **Benchmark/test harness** (`lm_benchmark.py`): at each checkpoint — held-out perplexity (a FIXED held-out shard) +
   deep-context NLL (the `eval_perdepth` we already have) + N generation samples (fixed prompts, so the owner sees
   qualitative progress) + optionally a tiny knowledge/reasoning probe. Append to a progress JSONL + a human-readable log.
6. **Autonomous loop** (`lm_train_run.py`): while not-paused and budget-remaining → train a chunk → checkpoint → benchmark
   → log → repeat. Resumable (loads the latest checkpoint on start). A PAUSE sentinel file (stop at the next checkpoint
   boundary, zero work lost — for gaming/other use). A coverage-complete Monitor armed alongside (done/crash/hang).
7. **Progress check-in:** a progress JSONL + a rendered curve (perplexity vs tokens) + the latest sample transcript, all
   readable any time; optionally surfaced in the webapp. The owner runs `lm_train_run.py start/pause/resume/status`.

## 2. De-risk ladder (cheapest-decisive first — validate BEFORE the big compute)
1. **RESUME CORRECTNESS (load-bearing):** train K steps → checkpoint → kill → resume → the next K steps' loss trajectory
   is BIT-CLOSE to an uninterrupted 2K-step run. If this fails, incremental "build on checkpoints" is broken. TINY model,
   minutes. **← starting this now.**
2. **THROUGHPUT at scale:** measure tokens/sec for the multi-layer WKV at ~50M/150M/300M on the 3090 → firm the
   compute→quality mapping (the 1-day/1-week/1-month token budgets) + the VRAM ceiling (gradient-checkpointing threshold).
3. **CORPUS pipeline:** download a FineWeb-Edu slice, tokenize, stream with a resumable cursor → tokens/sec of the data
   path (must not bottleneck the GPU) + the vocab freeze.
4. **BENCHMARK stability:** the held-out ppl + deep-NLL + fixed-prompt samples give a stable, monotone-ish progress
   signal at successive checkpoints (not noisy).
5. **THE DECISIVE SCALING RUN (the first real increment, biggest compute):** ~150M WKV on FineWeb-Edu, run to ~a few B
   tokens, watch whether broad-domain ppl collapses from ~121 toward the 20-40 range (does the WKV track the scaling
   curve?). THIS is the go/no-go on "converse like a small LLM is a training run away vs needs the scaffold."


## De-risk status (2026-07-21 — all LOAD-BEARING pieces validated)
- ✅ **Resume correctness** (model+optimizer+RNG): bit-exact (loss diff 0.00). `_lmtrain_resume_correctness_derisk.py`
- ✅ **Resumable data cursor** (stream continues exactly across restarts + epoch rollovers): 10/10 + 11/11.
  `_lmtrain_stream_cursor_derisk.py`
- ✅ **Throughput/VRAM**: measured; VRAM not the constraint. `_lmtrain_throughput_probe.py`
- ✅ **Optimization**: ~30× (bf16 + batch + chunked-scan[correctness-gated 4.77e-07] + torch.compile), 3043→~90000 tok/s.
  `_lmtrain_chunked_scan.py`, `_lmtrain_optim_probe.py`
- ⏳ **Remaining (ASSEMBLY, not de-risk)**: (a) benchmark harness (wrap `eval_perdepth` + fixed-prompt samples); (b)
  FineWeb-Edu download+tokenize (setup); (c) the autonomous train→ckpt→benchmark→continue loop + launcher (adapt
  `develop_run.py`). All the RISKY pieces are proven; the rest is integration + a data download.

## 3. Open choices for the owner (decision points)
- **Corpus:** FineWeb-Edu (educational-quality, best per-token) vs DCLM (reasoning/diversity) vs a mix (SmolLM2's 60/40).
  Default: FineWeb-Edu slice to start.
- **First model size:** ~150M (Chinchilla-optimal for a ~1-week budget) vs smaller (~50M, faster feedback) to start.
- **Tokenizer:** a fresh BPE on the corpus (freeze it — it must be stable across all increments).
- **When to commit the big compute** (the decisive scaling run) vs finishing the cheap de-risks first.
- **BPTT scaffold vs the biological local-rule** for the scale run (BPTT is pragmatic; the local rule is the end-state,
  a parallel track — the BPTT-retirement finding shows the readout is local-rule-learnable, but full local-rule at scale
  is a separate arc).

## 4. Fluid-abstain (parallel track, moderate build)
Confidence-conditioned generation: (1) expose the graded confidence (familiarity-gate margin for grounded; LM
entropy/logprob for open) as a scalar; (2) fine-tune the generator to hedge conditioned on it (assertive/hedged/soft-
abstain), interleaved to avoid forgetting; (3) verify calibration (hedge-rate rises as confidence drops; no confident-
wrong on held-out; graceful soft-abstain replaces the flat refusal). Reuses the format-fine-tune lever. Weeks; independent
of the base training.
