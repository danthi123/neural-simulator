# Generator-D — distillation into a spiking LM: honest NEGATIVE (decisive, clean; the spiking substrate itself is now the implicated wall)

## TL;DR

Generator-D — knowledge distillation of a competent trigram teacher's
DENSE soft next-token distribution into the spiking SNN via soft
cross-entropy through the validated surrogate-grad BPTT (the
genuinely-different attack on Generator-S's diagnosed bottleneck,
"the spiking net could not learn from a hard one-hot target") —
**FAILED its pre-registered multi-seed gate, 0/3, clean.** The
HARDENED gate_core (with the post-Generator-S absolute-competence
floor) functioned exactly as designed: it correctly rejected all 3
seeds (`absolute_competence_beats_random=False`) — there was NO
false-PASS to catch (the Generator-S gate hole is closed and
verified). **Honest, decision-relevant signal (reported, NOT spun as
a pass):** dense soft-target distillation improved the student's
absolute held-out perplexity **~146x** vs Generator-S (best
held-out ppl **804** vs Generator-S's 117,716; seed 42 within
**1.57x** of uniform-random) — the dense "dark-knowledge" signal IS
dramatically more learnable by the spiking substrate than a hard
one-hot target. But it still does **not** clear the pre-registered
absolute-competence floor (must beat uniform-random ~= vocab 512;
best 804 > 512), 0/3 multi-seed. Honest verdict: **NEGATIVE.** Real
TinyStories (cache-hit, `degraded=False`), FIXED pre-registered
config, clean multi-seed run, frozen bars byte-untouched. This
sharply re-localizes the bottleneck to the **surrogate-grad LIF
spiking substrate itself** at feasible local scale.

## The decisive gate result (recorded; smell-tested; NO re-run, NO bar-tuning)

`generator_d_gate.py`, seeds 42/43/44, FIXED pre-registered config
(vocab 512, hidden 256,256, T 32, 40 epochs, batch 32, 2000 samples),
real TinyStories (7.99M chars, cache-hit, `degraded=False`),
BPE-invariant word-shuffle control, HARDENED `subword_lm_gate_core`
(0.20/1.5/0.5/0.20 + `_GS_ABS_COMPETENCE_PPL_RATIO=1.0`, >=3 seeds —
byte-untouched), `gs_verdict` passed `uniform_ppl=512`:

| seed | student held-out ppl | vs random (512) | shuffled-ctl ppl | train ppl | teacher ppl | abs-competence | verdict |
|---|---|---|---|---|---|---|---|
| 42 | 804 | 1.57x WORSE | 965 | 2785 | 15.27 | False | FAIL |
| 43 | 2367 | 4.62x WORSE | 2118 | 2245 | 15.27 | False | FAIL |
| 44 | 3088 | 6.03x WORSE | 3320 | 2966 | 15.27 | False | FAIL |

Aggregate: n_seeds 3, n_pass **0/3** -> **GATE: FAIL**. Mandatory
anti-cheat smell-test (scrutinize a verdict harder than its face
value, recomputed from the recorded JSON, no GPU re-run): 0/3 seeds
beat uniform-random; best student ppl 804 is still 1.57x WORSE than
random. The hardened floor `absolute_competence_beats_random=False`
correctly rejects all three. There is NO false-PASS here (clean FAIL);
the Generator-S-class gate hole is closed and verified working. Bars
NEVER tuned; verdict integrity-clean.

## What is genuinely new (honest signal, not overclaimed)

Generator-D is **not** a flat repeat of Generator-S. The dense
soft-target distillation signal moved the spiking student ~146x
closer to competence in absolute perplexity (Generator-S best
held-out ppl ~117,716, ~230x worse than random; Generator-D best
~804, ~1.57x worse than random). The competent teacher (held-out ppl
**15.27** vs random 512) demonstrably carried real transferable
signal, and the student absorbed most of it — seed 42 came within a
factor of 1.6 of the random floor. This is the closest any mechanism
in the conversational-generation arc has come to held-out competence.
But "closest" is **not** a pass: the pre-registered floor is "beat
uniform-random", multi-seed >=3, and Generator-D clears it on 0/3.
The honest conclusion is a NEGATIVE with a precisely-characterized,
decision-relevant residual: the bottleneck is no longer signal
poverty (distillation closed ~99.3% of the absolute-ppl gap) and not
the teacher (competent) — it is the **surrogate-grad LIF spiking
substrate's learnability at feasible local scale**.

## The converged picture (now 9 honest negatives)

Self-contained-at-runtime generative *production* of coherent text
does not work on this substrate/hardware under the no-cheating/local
constraints across: char-level self-distilled (Inc-1/2/3),
controllers/predictors over an order-blind pool (G1/G1.5/P),
order-intrinsic deterministic readback (order-intrinsic), a subword
spiking LM on a real corpus with a hard one-hot target (Generator-S),
and now the SAME spiking LM with the strongest known training signal
(dense soft-target knowledge distillation from a competent teacher;
Generator-D). The corpus removed signal poverty; distillation removed
the hard-target handicap (~146x absolute improvement) — yet a modest
surrogate-grad LIF spiking LM still does not reach better-than-random
held-out generation at feasible local scale. The validated,
multi-seed, anti-cheat-validated **trustworthy grounded continual
memory with no-confabulation abstention** is untouched and remains
the deliverable.

## Anti-cheat discipline (maxed-integrity honest negative)

- HARDENED gate_core bars (0.20/1.5/0.5/0.20 + abs-competence floor
  1.0, >=3 seeds) byte-UNTOUCHED across the whole Generator-D arc
  (verified empty-diff); 650 never used; song_g1_core/bridge
  byte-untouched.
- The mandatory post-run smell-test was applied (a clean FAIL is
  still scrutinized): recomputed from the recorded JSON, no GPU
  re-run, no bar-tuning. The Generator-S-lesson floor caught what it
  was designed to catch (it would NOT have admitted a noise model;
  here there is no false-PASS — it is an honest FAIL).
- Build was subagent-driven TDD with two-stage review; the rigorous
  adversarial review of the load-bearing `soft_xent` core caught a
  real defense-in-depth gap (silent q/logits length mismatch), fixed
  by strengthening; an implementer correctly STOPPED on a
  controller test-literal error rather than fake-pass (corrected to
  match the validated precedent). soft_xent's faithful-CE-equivalence
  is pinned (loss diff 2.5e-16, grad diff exactly 0 vs the validated
  oracle) — the distill gradient is provably correct, so the FAIL is
  about the spiking substrate, not a numerical bug.
- Generator-D is NOT config-cranked. The pre-registered cheap-slice
  gate decided; this honest NEGATIVE is propagated and the arc
  proceeds to the next genuinely-different mechanism.

## Next (continuous autonomous arc — NOT a stop, NOT a config-crank)

The residual is now precisely localized, so Generator-E is sharply
motivated and decision-relevant: **isolate whether the SPIKING
constraint itself is the wall.** Run a NON-spiking but
catalog-grounded sequence substrate (an echo-state / reservoir
readout, or a minimal rate-based ANN of comparable FLOP/scale) on the
SAME real corpus, judged by the SAME HARDENED gate_core (no new bar).
- If the non-spiking substrate clears the absolute-competence floor
  where the spiking one did not -> the spiking substrate is the
  decision-relevant bottleneck for self-contained generation (a major
  finding: the conversational-generation path needs a non-spiking
  sequence model, or a fundamentally different spiking approach).
- If even the non-spiking substrate fails at this cheap scale -> the
  bottleneck is scale/approach, not spiking; informs a different
  Generator-F.
Either outcome is decision-relevant. Design pre-staged in the
Generator-D design doc's successor section; full Generator-E
design->plan->build->gate proceeds immediately, no stop/ask.

## Files

- Mechanism: `sim/ngram_teacher.py`, `sim/soft_xent.py`,
  `research/runners/distill_subword_lm_train.py`,
  `research/runners/generator_d_gate.py`
- Gate (HARDENED, frozen, byte-untouched):
  `research/runners/subword_lm_gate_core.py`
- Evidence: `research/findings/raw/g11_bg/generator_d_gate.json`,
  `_generator_d_gate.log`, `generator_d_grounding.json`
- Design/plan: `docs/plans/2026-05-17-generator-D-distillation-{design,implementation}.md`
- Prior arc: the 8 prior generator/order-intrinsic/Generator-S
  NEGATIVE findings
