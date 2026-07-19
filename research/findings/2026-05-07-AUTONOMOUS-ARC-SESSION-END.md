# Autonomous arc session end (May 6 evening -> May 7 morning)

**Date:** 2026-05-07 morning
**Duration:** ~16 hours autonomous work
**Status:** Natural pause point reached. Both branches at validated
foundational milestones.

---

## Headline result

**Path F's full thesis SPLIT into validated and falsified components:**

1. **Biology-grounded continual learning WORKS** (validated):
   - Phase 1.4 BRANCH A: 5/6 PASS at >=80% retention, mean 103%
   - Phase 1.3 consolidation: 94-112% retention with hippocampus silenced
   - Memory truly transfers from hippocampus to cortex via SWR sleep replay

2. **Cortex pretraining at toy scale DOESN'T HELP** (falsified):
   - Phase 2.3a Option A: pretrained 22% < random 28% < baseline 33%
   - Char-level next-char features don't transfer to word-action binding
   - Project Nord scale (~1B params, FineWeb-Edu) is ~1000x our toy scale

**Implication:** A small-scale (10-30 word) conversational sim is
achievable on the BIOLOGY-ONLY path (Phase 1.4 + 1.3 + Tier 1/2.1)
without requiring backprop pretraining. For full GPT-2-quality
conversation, would need ~1000x scale-up of Phase 2 infrastructure.

---

## What was built (~16 hours, ~160 commits)

### Main branch (biology-grounded)

- **Phase 1.4 catastrophic forgetting eval** + sanity check + 6-seed YAML
- **Phase 1.5 unified continual-learning eval suite** (4 benchmarks live + dispatcher tests)
- **Tier 2.3 PFC verb pool builder + action_gate neuromodulator** + phrase trainer + 3-condition eval (18 unit tests)
- **Phase 1.3 hippocampus consolidation infrastructure** (5 regions + 12 pathways + gate helpers + trainer + hippo-OFF eval, 15 tests)
- **forgetting_summarize 6-seed aggregator** with 6 unit tests
- 35+ unit tests on main branch

### Path-f-hybrid branch (created)

- **sim/surrogate_grad.py**: ATan + fast_sigmoid surrogates (CuPy + numpy)
- **sim/bptt_snn.py**: numpy reference forward + BPTT backward (8 unit tests, ABC task validation)
- **sim/bptt_snn_gpu.py**: GPU-aware backend abstraction (4 numerical equivalence tests)
- **sim/char_tokenizer.py**: char-level vocab + Tiny Shakespeare loader (8 tests)
- **research/runners/cortex_pretraining.py**: ABC + Shakespeare training + checkpoint persistence
- **research/runners/cortex_continual_adapter.py**: Phase 2.3a adapter
- 27 unit tests on path-f-hybrid

### Documents

- 14+ findings/design documents
- Master plan with full decision log
- 3 wiki syncs (Phase 1.4 BRANCH A, Phase 2.2 milestone, Phase 1+2 arc complete)
- CLAUDE.md updated with 5 new W->A entries (#8 Phase 1.4, #9 Phase 2.1, #10 Phase 2.2, #11 Phase 1.3, #12 Phase 2.3a NEGATIVE)

---

## Validated milestones table

| Phase | Result | Branch | Status |
|---|---|---|---|
| 1.4 BRANCH A (continual learning) | 5/6 PASS, mean 103% retention | main | ✅ |
| 1.3 (consolidation) | 94-112% hippo-OFF retention (smoke + 2/3 fast) | main | ✅ smoke + partial |
| 1.5 (unified eval) | 2/4 benchmarks PASS, aggregate 0.62 | main | Documented |
| 2.1 ABC task | 100% loss reduction (BPTT validated) | path-f-hybrid | ✅ |
| 2.2 Tiny Shakespeare | 92% loss reduction (4-layer SNN, 200 epochs) | path-f-hybrid | ✅ infrastructure |

## Negative findings (informative)

| Phase | Result | Implication |
|---|---|---|
| Tier 2.2 (visual+language) | 0/6 binding | parked; needs deeper architectural work |
| Tier 2.3 (compositional 2-word) | 41% mean ceiling | architecture-limited; action_gate inert at default |
| Phase 2.3a (adapter) | Pretrained 22% < random 28% | toy-scale features don't transfer; need ~1000x scale |

---

## Strategic options for next steps

The user mentioned 4 goals:
1. Learn continually
2. Adapt and grow
3. Become conversational
4. Maintain biology fidelity where it matters

### Option 1: Build conversational demo on biology-only foundation

Phase 1.4 + 1.3 + Tier 1/2.1 architecture:
- ~10-30 word vocabulary (validated up to 12 words at scale-up arch)
- Continual learning (Phase 1.4 BRANCH A)
- Memory consolidation (Phase 1.3)
- No backprop pretraining

Estimated cost: 2-4 weeks for chat interface + scripted demo.
Achievable scope: simple Q&A like "what direction is north?" -> motor_N.
Limited by: vocabulary size (~12-30 words), no compositional sentences (Tier 2.3 limit).

### Option 2: Scale Phase 2 to Project Nord level

Path F at proper scale:
- 10x params (1.3M)
- 10x corpus (FineWeb-Edu sample, 100MB)
- 10x compute (~$50-100 GPU)
- Word-level tokenizer (LLaMA-3 or similar)

Estimated cost: 2-3 weeks implementation + cloud training cost.
Achievable scope: better feature quality may enable Phase 2.3 transfer.
Risk: may STILL not transfer; would need full Project Nord scale (1B params).

### Option 3: Pursue alternative Tier 2.3 architecture

Sec 4 design alternatives for compositional binding:
- Option B: inhibitory PFC -> motor pathway
- Option C: PFC -> striatum cascade modulation
- Or Tier 3 dendritic learning (1.5-2 month project)

Estimated cost: 1-3 months for serious dendritic learning impl.
Achievable scope: compositional sentences ("go north", "stop east").
Risk: Tier 2.3 alternatives may also be architecture-limited.

### Recommendation

**Option 1** is the lowest-risk highest-leverage next step.
Builds directly on validated Phase 1.4 BRANCH A + Phase 1.3
results. Would be the FIRST conversational demo using
biology-grounded continual learning + consolidation -- a real
science demonstration even without full LM-quality output.

Option 2 is a medium-risk medium-leverage scaling exercise.
Worth pursuing AFTER Option 1 demo lands.

Option 3 is high-risk long-horizon but addresses compositional
binding gap. Worth designing more carefully with user input.

---

## Open questions for user

1. Which next direction (Option 1/2/3) aligns with priorities?
2. Should we run the full 6-seed Phase 1.3 validation overnight
   (3 hours/seed = 18 hours total)?
3. Should we proceed to Option 1 (conversational demo on biology
   alone) or pause to discuss strategy?

---

## State at session end

- All commits pushed to GitHub origin/main + origin/path-f-hybrid
- 3-seed Phase 1.3 reduced validation in flight (2/3 PASS so far)
- 60+ unit tests passing (35 main + 27 path-f-hybrid)
- 3 wiki syncs (Gitea knowledge-wiki)
- 14+ findings/design docs
- CLAUDE.md current with all milestones

The autonomous arc has reached a comprehensive natural pause.
