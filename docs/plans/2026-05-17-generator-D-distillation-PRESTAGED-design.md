# Generator-D — Knowledge-Distillation Spiking LM (PRE-STAGED FAIL-branch design)

> **Status: PRE-STAGED, not active.** Written while the decisive
> Generator-S gate runs, so the continuous autonomous arc never
> stalls on the verdict (autonomous-runs "pre-stage A/B branches"
> discipline). ACTIVATES ONLY IF Generator-S FAILs its pre-registered
> gate. If Generator-S PASSes, this is shelved and Generator-C (spiking
> cortex onto the validated grounded-memory arch) is designed instead.
> Pre-staging is NOT pre-judging — the Generator-S verdict comes solely
> from its own frozen-bar gate.

## Why this is genuinely different (NOT config-cranking Generator-S)

Generator-S tests: can a subword spiking net learn coherent
generation from the **raw corpus next-token signal** (hard, sparse —
one-hot target per position). The converged diagnosis across the
whole arc is **training-signal poverty**. Generator-S removes corpus
poverty (real public corpus) but the *signal shape* is still a sparse
hard target a spiking net must fit through surrogate-grad BPTT.

Generator-D changes the **signal shape itself**: a competent **local
open-weights teacher** (training-time only) provides a DENSE
soft-target next-token distribution; the spiking net is trained to
match it (knowledge distillation — Hinton 2015, already in the
project's scientific basis). Dense soft targets are the strongest
known training signal for making a small model competent — it is
exactly how small open models are made fluent. This is a different
training objective and signal source, not a hyperparameter of
Generator-S, so it is not config-cranking a terminated mechanism.

## Constraints (firm — restated)

- **Self-contained at RUNTIME (post-training):** the teacher is used
  ONLY during training to produce soft targets. The shipped artifact
  is the trained spiking-net weights + the static BPE merge table.
  No teacher, no corpus, no external dependency at inference. This
  satisfies the user's firm "self-contained post-training" constraint
  (the teacher is a training-time resource of the same class as the
  user-authorized public training corpus).
- **Local hardware only:** teacher must run on the RTX 3090 (e.g.
  Qwen2.5-0.5B / Llama-3.2-1B / Phi-3-mini, 4-bit — all 3090-feasible
  and open-weights). No cloud.
- If the user deems an open-weights teacher out of scope (only a
  *corpus* was explicitly named), the in-constraints fallback is
  **Generator-D'**: self-distillation / sequence-level objectives over
  the corpus (no external teacher) — still a different signal shape
  than S. Recorded here so the arc continues either way; the user can
  redirect but the arc does NOT stop to ask (standing directive).

## Mechanism

1. Reuse the validated subword BPE + the scaled spiking-SNN BPTT
   trainer (Generator-S Tasks 1,4 — UNMODIFIED, DRY).
2. Net-new (small): a training-time teacher adapter
   `research/runners/teacher_logits.py` — loads a local open-weights
   model (transformers/llama.cpp, 4-bit, 3090), returns per-position
   next-token log-probs over the SHARED BPE vocab (vocab-projection /
   tokenizer-bridge handled once, cached to disk so the teacher runs
   at most once per corpus — kill-safe, resumable, then the teacher is
   never needed again).
3. Distillation loss = KL(teacher_soft || student_softmax) (+ optional
   small CE on the true token), backprop through the SAME validated
   surrogate-grad BPTT. Only the loss/grad term is new; the unroll is
   reused UNMODIFIED.

## Pre-registered gate (REUSE Generator-S's frozen bars UNMODIFIED)

Generator-D is judged by the **IDENTICAL, already-frozen,
adversarially-hardened `subword_lm_gate_core`** (bars
0.20/1.5/0.5/0.20, ≥3 seeds, BPE-invariant word-shuffle control) —
the gate is mechanism-agnostic, so NO new bar is introduced or tuned.
A new thin runner `generator_d_gate.py` swaps only the trainer's loss
(distillation vs corpus-CE); everything scored identically. Held-out
perplexity must beat the word-shuffle control by ≥20%, generalize
(≤1.5× train), generate non-degenerately (distinct-trigram ≥0.5,
≤20% verbatim copy), multi-seed ≥3. PASS ⇒ scale + Generator-C
integration. FAIL ⇒ honest propagation + next pre-staged mechanism;
NO config-crank.

## Scope / honest ceiling

Cheap decisive slice (same falsify-cheaply discipline): small teacher,
bounded corpus slice, the SAME ~3090-feasible scale Generator-S used.
Honest ceiling: small-LM coherent generation, NOT GPT-class; the
deliverable framing never overclaims. A maxed FAIL is a real finding,
propagated, triggering the next genuinely-different mechanism.

## Pre-staged successor (so the arc never stalls)

- Generator-D PASS ⇒ Generator-C (corpus/distill-pretrained spiking
  cortex wired onto the validated grounded-memory + no-confabulation
  arch; the unification).
- Generator-D FAIL ⇒ Generator-E candidates already in view:
  (i) latent/continuous-target sequence prediction (predict the
  teacher's hidden state, not just token dist); (ii) a different
  catalog-grounded sequence substrate (e.g. reservoir/ESN readout
  trained on the corpus). Each gets its own pre-registered gate via
  the same UNMODIFIED gate_core. The arc continues autonomously.
