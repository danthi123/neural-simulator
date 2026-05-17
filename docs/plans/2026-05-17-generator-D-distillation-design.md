# Generator-D — Soft-Target Distillation into a Spiking LM — Design (ACTIVE)

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). Continuous autonomous arc
> (user 2026-05-17: work the arc a week, no stopping/asking, no
> config-cranking a terminated mechanism, self-contained at RUNTIME,
> local 3090, public training resources authorized). Supersedes the
> pre-staged skeleton `2026-05-17-generator-D-distillation-PRESTAGED-
> design.md` with the grounded mechanism.

## Why genuinely different (NOT config-cranking Generator-S)

Generator-S (subword spiking LM, one-hot next-token target on a real
corpus) FAILED its decisive multi-seed gate: held-out ppl ~10^5,
200x-758x WORSE than uniform-random — the spiking net could not learn
next-token prediction from a **hard, sparse one-hot target** at
feasible local scale (the anti-cheat caught the relative-only gate's
false PASS; honest NEGATIVE propagated; gate_core hardened with a
frozen absolute-competence floor).

The diagnosed bottleneck is **signal density / spiking learnability**,
NOT corpus quality (TinyStories is excellent) and NOT tokenization
(subword already fixed char-confusion). Generator-D changes the
**training signal shape itself**: replace the one-hot hard target with
a competent teacher's **dense soft next-token distribution** and train
the spiking student by KL (Hinton-2015 knowledge distillation — the
"dark knowledge" in the full distribution is the documented mechanism
that makes small/hard-to-train models learnable). Different objective
(KL to a dense distribution) + different signal source (a teacher),
not a hyperparameter of Generator-S → not config-cranking a
terminated mechanism.

## Evidence grounding (falsify-cheaply, done BEFORE this design)

A smoothed back-off **trigram teacher over our own BPE vocab** (vocab
513), trained on the TinyStories train split, achieves **held-out
perplexity 14.3** vs uniform-random 513 — **36x better than random**
(recorded probe `ba1jyepwf`). So the distillation TARGET is a
genuinely competent LM and carries real transferable signal. It is
pure stdlib `Counter` (zero new deps), zero external weights,
unambiguously in-constraints (a statistical model of the
user-authorized corpus — the SAME class of training-time resource as
"using the corpus"), and shares the student's BPE vocab → NO
vocab-bridge complexity.

## Thesis

Train the validated spiking SNN (surrogate-grad BPTT, reused
UNMODIFIED) to match, per position, the trigram teacher's dense
soft next-token distribution over the shared BPE vocab via a KL /
soft-cross-entropy loss. Runtime artifact = the trained spiking-net
weights + the static BPE merge table ONLY (teacher discarded after
training — self-contained at RUNTIME; the teacher is training-time
only, exactly like the corpus). Honest scope: small-LM coherent
generation, NOT GPT-class; framing never overclaims.

## Architecture (net-new is small; validated core + hardened gate reused UNMODIFIED)

Reuse UNMODIFIED (DRY): `sim.bpe_tokenizer`,
`sim.bptt_snn_gpu.{forward_unroll_xp,backward_unroll_xp}`,
`research.runners.scaled_subword_lm_train` loop SHAPE,
`sim.train_checkpoint` kill-safe, `research.runners.corpus_fetch`,
`research.runners.subword_lm_generate`, and the **hardened
`subword_lm_gate_core`** (frozen bars 0.20/1.5/0.5/0.20 + the new
frozen absolute-competence floor `_GS_ABS_COMPETENCE_PPL_RATIO=1.0`,
≥3 seeds — byte-UNTOUCHED; Generator-D is judged by the SAME gate, no
new bar).

Net-new (small, well-scoped, pure-testable where possible):
1. `research/runners/ngram_teacher.py` — pure: train a smoothed
   back-off trigram over BPE-encoded train ids; `soft_dist(ctx)` →
   a length-V probability vector (the dense target). Add-k + back-off
   exactly as the grounded probe (ppl 14.3). CPU-unit-testable
   (proper distribution, sums to 1, beats uniform on a toy corpus,
   deterministic).
2. `research/runners/distill_subword_lm_train.py` — a DRY mirror of
   `scaled_subword_lm_train` with the ONLY change being the loss/grad:
   instead of `cross_entropy_loss_np`/`softmax_grad_np` against a
   one-hot target, use **soft cross-entropy** `-(sum_w q_w log p_w)`
   and its gradient `softmax(logits) - q` (q = teacher dist). This is
   the SAME backward_unroll_xp BPTT; only the output-grad term
   changes (pure, unit-testable: `soft_xent_loss`/`soft_xent_grad`).
   Kill-safe resume / OOM-halving / KeyboardInterrupt reused verbatim.
3. `research/runners/generator_d_gate.py` — thin runner: identical to
   `subword_lm_gate.py` orchestration EXCEPT (a) trains the student
   with the distillation trainer, (b) **passes
   `uniform_ppl=tok.vocab_size` to `gs_verdict`** (REQUIRED — the
   hardened gate_core is fail-closed without it; this is the
   controller follow-up the gate-core hardening flagged), (c) records
   the teacher's own held-out ppl in the JSON for transparency (NOT a
   gate metric — the gate judges the STUDENT's held-out ppl only).

## Data flow

corpus_fetch (cached TinyStories) → split → BPE (shared) → trigram
teacher on train ids → student SNN trained by KL-to-teacher-soft-dist
via the validated BPTT (kill-safe) → student held-out perplexity +
word-shuffle-control student + train ppl + generation metrics →
hardened `gs_verdict(..., uniform_ppl=V)` → `gs_aggregate_multiseed`
→ JSON. Teacher discarded post-training (runtime = student only).

## Pre-registered gate (the SAME hardened gate_core; bars FROZEN, never tuned)

The student (spiking net, teacher gone) must, multi-seed ≥3:
1. **Absolute-competence (the bar Generator-S failed):** held-out ppl
   < uniform-random (vocab_size) — `_GS_ABS_COMPETENCE_PPL_RATIO=1.0`,
   fail-closed without `uniform_ppl`.
2. Beat the BPE-invariant word-shuffle control by ≥20%
   (`_GS_PPL_MARGIN=0.20`).
3. Generalize: held-out ≤ 1.5× train (`_GS_GENERALIZATION_MAX`).
4. Non-degenerate generation: distinct-trigram ≥0.5; ≤20% verbatim
   train copy (catches "just memorized/copied the teacher").
All bars FROZEN/byte-untouched, NEVER tuned post-hoc. Same mandatory
post-run anti-cheat smell-test as Generator-S (a nominal PASS is
scrutinized harder than a FAIL; absolute numbers sanity-checked vs
random BEFORE propagation). PASS ⇒ a self-contained spiking LM
genuinely learned language via distillation → scale + Generator-C
(wire onto the validated grounded-memory arch). FAIL ⇒ honest
propagation + immediately the next pre-staged mechanism; NO
config-crank, NO stop.

## Honest ceiling / risks (no overclaiming)

- The teacher being competent (ppl 14.3) does NOT imply the spiking
  student can absorb it — surrogate-grad BPTT through LIF may still
  fail to fit a dense target at feasible scale. A maxed FAIL here is a
  deep, decision-relevant negative about spiking-substrate
  learnability (points to non-spiking / different-substrate
  directions for Generator-E), and is propagated honestly, not
  iterated.
- Anti-cheat: the gate judges the STUDENT on HELD-OUT (teacher never
  at gate time); the verbatim-copy bar + word-shuffle control catch
  trivial teacher-copying; the absolute-competence floor catches the
  Generator-S-style vacuous-relative-bars trap. Teacher is
  training-time only (self-contained at runtime).
- Cheap decisive slice (falsify-cheaply): same ~3090-feasible scale
  Generator-S used (vocab 512, hidden 256,256, T 32, ~40 ep, 2000
  samples); kill-safe; ASCII-only; my wall-clock hand-estimates are
  unreliable, the measured per-epoch cost (~3.5s) is.

## Pre-staged successors (arc never stalls)

- Generator-D PASS ⇒ Generator-C (distill-pretrained spiking cortex
  wired onto the validated grounded-memory + no-confabulation arch).
- Generator-D FAIL ⇒ Generator-E candidates already in view:
  (i) predict the teacher's CONTINUOUS hidden/logit vector
  (regression target, even denser than a categorical soft dist);
  (ii) a non-spiking but still-catalog-grounded sequence substrate
  (e.g. an echo-state/reservoir readout) trained on the corpus —
  testing whether the spiking constraint itself is the bottleneck.
  Each gets the SAME hardened gate_core, pre-registered, no stop.

## Scientific basis (catalog)

Hinton/Vinyals/Dean 2015 knowledge distillation (dark knowledge);
Kim & Rush 2016 sequence-level KD lineage; Kneser-Ney/back-off n-gram
LM (the teacher); Neftci 2019 surrogate-grad BPTT (the student, reused
validated); Pulvermüller distributed cortical word ensembles
(Generator-C integration target); Marr/McClelland CLS (no-forgetting,
Generator-C).

## Out of scope (YAGNI)

No external dependency at RUNTIME ever. No config-cranking Generator-S
or any terminated mechanism. No vocab-bridge (teacher shares the
student's BPE). No open-weights teacher in this slice (the n-gram
teacher is competent, zero-dep, zero-ambiguity; an open-weights
teacher is a deferred stronger variant only if the n-gram-distilled
student PASSES competence but plateaus below coherence). The
pre-registered multi-seed gate decides; FAIL → next pre-staged
mechanism immediately.
