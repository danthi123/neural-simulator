---
type: plan
status: live
date: 2026-05-09
---

# Phase 2 Resumption Plan — 10M-param scale-up + conversational scaffolding

**Date:** 2026-05-09 EDT
**Status:** DESIGN
**Branch:** path-f-hybrid (Phase 2 work; main untouched)
**Goal:** Resolve the strategic uncertainty Phase 2.3a left open: does cortex pretraining transfer to word-action binding at non-toy scale? Then build conversational scaffolding regardless of outcome.

---

## Background: where Phase 2 currently sits

- **Phase 2.1 (path-f-hybrid)**: ABC task PASSES. Surrogate-grad BPTT + ATan surrogate validated. 2-layer SNN (3 → 32 → 3) on ABCABC... 100% loss reduction.
- **Phase 2.2 (path-f-hybrid)**: Tiny Shakespeare PASSES at toy scale. 4-layer SNN (66 → 256 → 256 → 66, 134K params), 200 epochs, loss 12.18 → 1.016 (92% reduction), 11 min on 3090.
- **Phase 2.3a (path-f-hybrid)**: NEGATIVE at toy scale. Pretrained 134K-param cortex used as adapter for Bridge: 22% W→A vs 28% random init vs 33% Phase 1.4 baseline. Char-level next-char features didn't transfer to word-action binding.
- **Conclusion at session end 2026-05-07**: Phase 2 paused. Recommendation: build on Phase 1.4 architecture for conversational demo, OR scale Phase 2 ~1000x for real transfer.

The "or" is the strategic gap. **Phase 2.3a NEGATIVE at 134K params doesn't necessarily mean Phase 2 thesis is wrong — it might just mean we needed more params.** This plan tests whether 10M params (75× larger, still 3090-feasible) is enough for transfer.

---

## Three tracks running in parallel

### Track 1: Phase 2.2b — 10M-param scale-up (~14 hr single training run)

**Goal:** Train a 10M-param SNN cortex on Tiny Shakespeare. Direct continuation of Phase 2.2 at meaningful scale.

**Architecture:**
- 4-layer SNN (matches Phase 2.2): input → hidden_1 → hidden_2 → hidden_3 → output
- Hidden width: 2048 (vs Phase 2.2's 256, 8× wider)
- 66-char vocab (Tiny Shakespeare; matches Phase 2.2)
- Embedding: 66 → 2048 (135K params)
- Hidden layers: 2048 → 2048 (4.2M params each, ×3 layers = 12.6M params)
- Output projection: 2048 → 66 (135K params)
- **Total: ~13M params**

Memory budget (back-of-envelope):
- Params: 13M × 4 bytes = 52 MB
- Activations (BPTT, T=32, batch=32): 32 × 32 × 2048 × 4 bytes × 4 layers = ~32 MB
- Gradients: 52 MB
- Optimizer (Adam): 104 MB
- SNN state (membrane V, refractory, surrogate cache): ~50 MB
- **Total: ~300 MB GPU memory** (well within 24GB)

**Training:**
- Tiny Shakespeare (1.1MB, 1.1M chars)
- T = 32 timesteps unroll
- Batch size 32
- 200 epochs (matches Phase 2.2)
- Adam, lr=0.003 (matches Phase 2.2)
- ATan surrogate (matches Phase 2.2)

**Wall-clock estimate:**
- Phase 2.2 at 134K params: 11 min / 200 epochs
- Compute scales with params at fixed corpus + epochs: 75× more params
- Wall clock: 11 min × 75 = **~14 hrs single overnight run on 3090**
- (likely less since 134K underutilized 3090; could be 8-10 hrs)

**Pass criterion:**
- Loss reduction matches Phase 2.2's pattern (≥80% reduction; 12.18 → ≤2.5 ish)
- Sample text generation produces locally-coherent chunks
- Doesn't NaN or gradient-explode

**Deliverable:**
- Checkpoint at `research/findings/raw/path_f/shakespeare_pretrained_10M.npz`
- Findings doc: `2026-05-XX-Phase-2.2b-10M-params-RESULT.md`

---

### Track 2: Phase 2.3b — Test if 10M-param transfer works (~3-6 hrs)

**Goal:** Use the 10M-param checkpoint as adapter into Phase 1.4 BRANCH A. Repeat the Phase 2.3a methodology at 75× scale.

**Architecture (mirrors Phase 2.3a):**
- Pretrained 10M-param cortex → adapter linear layer → existing Bridge `language_input` region
- All Phase 1.4 BRANCH A config: embodied Hebbian co-firing, NMDA, topographic prior, motor FS
- Train 4-word vocab (north/east/south/west) at 200 events/word
- Eval W→A on Phase 1.4 protocol

**Comparisons:**
| Condition | Architecture | Expected from Phase 2.3a | New result |
|---|---|---|---|
| Random init cortex | 10M params, untrained | (was 28% at 134K random) | TBD |
| Pretrained cortex | 10M params, trained on Shakespeare | (was 22% at 134K pretrained) | **CRITICAL** |
| Phase 1.4 baseline | (no cortex pretraining, just biology) | 33% (validated 5/6 + 6/6) | reference |

**Pass criteria (multiple thresholds):**
- **Strong PASS**: pretrained 10M ≥ Phase 1.4 baseline (33%+) AND pretrained ≥ random + 5pp
  - Means: bigger features transferred, scaling works, Phase 2 thesis rescued
- **Weak PASS**: pretrained 10M > random init by 5pp+
  - Means: pretraining helps SOMEWHAT at this scale, larger scale would help more
- **NEGATIVE**: pretrained ≤ random
  - Means: char-level pretraining genuinely doesn't transfer to word-action binding regardless of scale; need word-level tokenization OR much larger scale

**Decision tree post-2.3b:**

```
                     [Phase 2.3b result]
                            |
            +---------------+---------------+
            |               |               |
       Strong PASS       Weak PASS      NEGATIVE
            |               |               |
   2.3c integration   try 50M params    investigate why
   + chat_repl wiring  (1 week run)     - word tokenization?
                                         - bigger needed?
```

**Wall-clock:** ~3 hrs train + ~30 min eval = ~3.5 hrs single seed. Multi-seed (3 seeds): ~10 hrs.

**Deliverable:**
- Findings doc: `2026-05-XX-Phase-2.3b-10M-cortex-transfer-RESULT.md`
- If PASS: launch Phase 2.3c integration

---

### Track 3: Conversational scaffolding (parallel CPU work, ~2-3 days engineering)

**Goal:** Make the biology-only path produce a genuine conversational artifact even WITHOUT Phase 2. Independent of Track 1+2 outcome.

**Components:**

#### 3a. Online vocab learning in chat_repl
- User types unknown word → sim assigns to a motor pool via STDP
- Persists across sessions via existing `--save-bridge` checkpoint
- Already partially possible (chat_repl trains and saves), but training is batch
- Need: per-turn STDP gate toggle (`set_plasticity_gate("language_input_to_motor", 1.0)`) during dialogue, with motor target inferred from user's stated intent

Implementation: ~1 day
- Add `--learn` flag to chat_repl
- Each turn: if user says "swerve = west", drive motor_W teacher pulse + open STDP gate, run inference loop, close gate
- Save checkpoint after each binding

#### 3b. Multi-turn dialog state via PFC working memory
- PFC NMDA bistability holds context ~500ms
- Wire `dlpfc_wm` region to integrate user's recent utterances
- Use as gating signal for motor selection (per Tier 2.3 design but at simpler vocab scale)

Implementation: ~1-2 days

#### 3c. Generative output decoder
- `language_output` region's spike pattern → token via cosine match against vocab
- Multi-step: at each step, pick top-1 token, drive it back as input, generate next
- Bounded by vocab (4/8/12/16 words for now); chooses "which word the sim emits"

Implementation: ~1 day

**Combined deliverable:**
- chat_repl_v2 with online learning + dialog state + generative output
- Demo script:
  ```
  > Spike, what direction did I just go?
  Sim: north (last vocab event was "north")
  > Spike, "swerve" means left.
  Sim: ok (binds swerve → motor_W)
  > Spike, swerve.
  Sim: motor_W activated (recognizes new word)
  ```
- This isn't GPT-class, but it's MEANINGFULLY conversational — user can teach the sim new words and it remembers them across sessions

**Why this is valuable independent of Phase 2:**
- Even if Phase 2 succeeds, conversational scaffolding still needs to be built
- Validates the biology path's UX limit (what's interactive but doesn't generate novel text?)
- Provides a concrete artifact for the user to interact with

---

## Overall sequencing (after Phase 1.5 multi-seed completes)

| Hours | Track | Task |
|---|---|---|
| 0 | (now) | Wait for Phase 1.5 multi-seed completion |
| 0-2 | 3a | Build online vocab learning in chat_repl (CPU work in parallel with Phase 1.5) |
| 2-16 | 1 | Phase 2.2b training (overnight, 14 hrs on 3090) |
| 16-24 | 3b | Build dialog state during 2.2b training (parallel CPU) |
| 24-28 | 2 | Phase 2.3b transfer test single-seed |
| 28-32 | (analysis) | Aggregate, write findings, decision-tree branch |
| 32+ | (branch) | Either 2.3c integration OR investigation of negative result |

Total ~32 hrs from end of current session for the BIG strategic question to resolve.

**During and after**: Track 3 builds steadily, regardless of Phase 2 outcome.

---

## What changes in the master plan

Currently master plan says:
> Phase 2 PAUSED per the 2026-05-07 toy-scale negative finding.

After this plan:
> Phase 2 RESUMED at 10M-param scale (Phase 2.2b) per 2026-05-09 strategic decision. Scope: ~14 hr training run + 3-6 hr transfer test. Decision tree determines whether to scale further (50M-100M params, 1+ week local training) OR pivot to biology-only conversational path.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 10M Phase 2.2b NaNs / gradient explodes | Medium (8× wider hidden than tested) | Lower lr (3e-4 → 1e-4); gradient clipping at 1.0 |
| 10M Phase 2.3b still NEGATIVE | Likely 40% | Pivot to 50M scale OR word-level tokenization OR conversational scaffolding only |
| Training takes 30+ hrs instead of 14 | Medium | Can checkpoint periodically; resume on next day if PC needed |
| Conversational scaffolding (Track 3) breaks Phase 1.4 BRANCH A | Low | Develop on a feature branch; merge only if regression tests still pass |

---

## Per autonomous-runs principle #6 (anti-shortcut discipline)

This plan deliberately runs Phase 2 at 10M-param scale BEFORE committing to FineWeb-Edu (~$300-500 cloud budget). The intermediate scale (10M) is a known unknown — the answer to "does scaling fix Phase 2.3a's negative" is the strategic information we need.

If 10M PASSES at single-seed → multi-seed → Phase 2 thesis rescued, scale further when budget allows.

If 10M NEGATIVE → Phase 2 thesis is in serious doubt at any reasonable scale. Pivot to biology-only conversational path with current architecture.

Either outcome is decision-relevant. The current "Phase 2 paused after toy-scale negative" leaves the strategic question unresolved indefinitely.

---

## Specific file changes needed (path-f-hybrid branch)

1. **`research/runners/cortex_pretraining.py`** — already exists, extend for 10M arch:
   - Add `--hidden-width 2048` flag (currently `--hidden-layers "256,256"`)
   - Verify GPU backend handles 13M params (CuPy memory pool)
   - Add gradient clipping option

2. **`research/runners/run_continual_with_pretrained_features.py`** — already exists, mirror Phase 2.3a:
   - Take `--checkpoint shakespeare_pretrained_10M.npz`
   - Compare pretrained vs random vs Phase 1.4 baseline

3. **`research/runners/chat_repl.py`** (main branch, scaffolding work):
   - Add `--learn` flag for online vocab learning (Track 3a)
   - Refactor to support multi-turn dialog state (Track 3b)
   - Add cosine-match generative decoder (Track 3c)

4. **Webapp presets** (main branch):
   - `phase_2_2b_shakespeare_10M` (10M cortex pretraining)
   - `phase_2_3b_transfer_10M` (transfer test)
   - `chat_repl_v2_online_learning` (Track 3 deliverable)

5. **Tests:**
   - Verify path-f-hybrid existing 27 tests still pass
   - Add tests for new Phase 2.2b CLI flags
   - Add tests for chat_repl --learn mode (CPU helpers)

---

## Validation order

1. Phase 2.2b smoke (1 epoch, ~5 min): does training run cleanly at 10M params?
2. Phase 2.2b full (200 epochs, ~14 hrs overnight): loss reduction matches expectation?
3. Phase 2.3b transfer test single-seed: does pretrained > random?
4. Phase 2.3b multi-seed (if single PASS): does it generalize?
5. Track 3 milestones (parallel): online learning / dialog state / generative output

---

## Cost summary

- **Compute:** all on 3090, ~14 hrs Track 1 + ~10 hrs Track 2 = ~24 hrs GPU time
- **Cloud budget:** $0 (local-only)
- **Engineering time:** ~3-4 days work for Track 3 + validation
- **Total wall-clock to strategic resolution:** ~32 hrs (~1.5 days continuous)
