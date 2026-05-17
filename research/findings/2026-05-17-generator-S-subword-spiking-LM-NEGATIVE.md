# Generator-S — subword spiking LM on a real corpus: honest NEGATIVE (the anti-cheat discipline caught a false PASS pre-propagation)

## TL;DR

Generator-S — the genuinely-different mechanism selected after the
7-negative arc (a subword spiking LM trained by surrogate-grad BPTT on
a REAL public corpus, attacking the converged root cause
"training-signal poverty" with the now-authorized corpus) — **FAILED
its pre-registered multi-seed gate.** The gate runner *nominally
printed PASS (3/3)*, but that PASS was a **gate-design-hole artifact**,
caught by the project's mandatory anti-cheat smell-test BEFORE any
propagation: held-out perplexity was **117K / 124K / 388K** on a
~512-token vocab, i.e. **230x-758x WORSE than uniform-random next-token
perplexity (~512)** on all 3 seeds. The spiking net did not learn
language at all; generation is incoherent token-soup ("when ated Ft.
loved inina from gh,"). The nominal PASS occurred only because the
pre-registered bars were *relative* (held-out vs shuffled-control vs
train) and **lacked an absolute-competence floor** — when the real,
shuffled-control, AND train models are ALL astronomically bad, "ho <=
0.8*ctl AND ho <= 1.5*tr" is vacuously satisfied, and
distinct-trigram=1.0 / verbatim-copy=0.0 are trivially true for random
noise (every trigram distinct, nothing copied — *because it is
gibberish, not text*). Honest verdict: **NEGATIVE.** Corpus was the
real TinyStories (downloaded, `degraded=False`), config was the FIXED
pre-registered slice, run was clean (no traceback), multi-seed. This
is a maxed-integrity honest negative; the validated grounded-memory +
no-confabulation asset stands untouched.

## The decisive gate result (recorded data; recomputed, NO re-run, NO bar-tuning-toward-pass)

`subword_lm_gate.py`, seeds 42/43/44, **FIXED pre-registered config**
(vocab 512, hidden 256,256, T 32, 40 epochs, batch 32, 2000 samples),
**real TinyStories** (7.99M chars, `degraded=False`), BPE-invariant
word-shuffle control, FIXED bars `subword_lm_gate_core`
0.20/1.5/0.5/0.20 byte-untouched, >=3 seeds:

| seed | held-out ppl | shuffled-ctl ppl | train ppl | ho / uniform(512) | beats random? |
|---|---|---|---|---|---|
| 42 | 117,716 | 224,392 | 133,303 | 230x WORSE | NO |
| 43 | 124,282 | 188,015 | 236,538 | 243x WORSE | NO |
| 44 | 387,991 | 488,165 | 396,377 | 758x WORSE | NO |

Uniform-random next-token perplexity for a 512-vocab is ~512 (mean NLL
ln 512 = 6.24). All three seeds' held-out perplexity is ~10^5, mean
NLL ~11.7-12.9 — the model is **~2x worse per-token than guessing
uniformly at random**. Generation samples (all 3 seeds) are
incoherent fragments, not English. distinct-trigram=1.0 / copy=0.0
confirm it: pure noise is maximally "distinct" and copies nothing.

**Honest verdict:** with the obviously-required absolute-competence
precondition (a language model that has learned anything must at
minimum beat uniform-random; otherwise the relative comparisons are
vacuous), **0/3 seeds qualify -> FAIL**. This was computed PURELY from
the already-recorded `generator_s_gate.json` (no GPU re-run, no
chasing a number), and the correction is strictly **stricter** (it
rejects a noise model the holed gate admitted) — the only
integrity-valid direction. Same documented honest-correction class as
the order-intrinsic PRE-GATE CATEGORY-ERROR box, C1/C2,
Inc-3-held-out, P-bias, Task-5-retarget.

## What this is (honest mechanism, no spin)

The spiking SNN trained by surrogate-grad BPTT on next-subword-token
prediction **did not learn coherent generation at the cheap-decisive
slice scale on this hardware**. The training loop is the
multi-seed-validated Phase-2.1/2.2 core (loss *does* decrease on
TRAIN, as in Phase 2.2) — but, exactly as the Inc-3 lesson predicted,
**a decreasing train loss is not held-out language competence**: the
held-out perplexity is astronomically bad and the generated text is
noise. Subword tokenization + a real corpus removed two prior
confounds (char-level confusion, corpus poverty) but did NOT make a
modest spiking LM generate coherent held-out text at this scale.

The pre-registered gate's relative-only bars had a design hole (no
absolute-competence floor). The hole was caught by the mandatory
anti-cheat smell-test ("if it seems too easy, surface the suspicion
and check") BEFORE propagation — the discipline functioned exactly as
designed (the project's permuted-label-control history is built on
catching precisely this class of false positive). A less rigorous
process would have shipped a noise model as a "conversational
breakthrough." It did not.

## Anti-cheat discipline (maxed-integrity honest negative)

- FIXED bars 0.20/1.5/0.5/0.20 + >=3 seeds in `subword_lm_gate_core`
  NEVER tuned; byte-untouched (verified). 650 never used.
- The FALSE PASS was NOT propagated. The honest NEGATIVE was derived
  from recorded data with NO GPU re-run and NO bar-tuning-toward-pass;
  the correction only ever makes the gate reject more (noise), never
  pass more.
- Pre-registered FIXED config (real corpus, not the degraded
  fallback); clean multi-seed run; falsify-cheaply grounding had
  validated the pipeline end-to-end first (its toy FAIL was correctly
  NOT propagated).
- Gate-design lesson recorded forward: any LM gate needs a
  pre-registered **absolute-competence floor** (held-out ppl <
  uniform-random ~= vocab_size) so relative bars cannot be satisfied
  vacuously by mutually-terrible models. This floor is being added to
  `subword_lm_gate_core` as an ADDITIVE, STRENGTHENING, frozen
  pre-registered constant for the next mechanism's gate (it CANNOT
  flip Generator-S — Generator-S is already the honest FAIL by this
  obviously-correct sanity check on recorded data; and a
  reject-more-only change can never be "tuned toward a pass").
- Build was subagent-driven TDD with two-stage review; the adversarial
  review had already hardened the gate core (C1/I1) — yet this
  *relative-only* hole remained, which is exactly why the
  post-hoc anti-cheat smell-test on the actual numbers is mandatory
  and non-skippable. Honest finding, propagated, not iterated away.

## The converged picture (now 8 honest negatives)

Self-contained-at-runtime generative *production* of coherent text
does not work on this substrate/hardware under the no-cheating/local
constraints — across char-level self-distilled (Inc-1/2/3),
controllers/predictors over an order-blind pool (G1/G1.5/P),
order-intrinsic deterministic readback (order-intrinsic), and now a
subword spiking LM on a real public corpus at the cheap-decisive scale
(Generator-S). The corpus removed signal *poverty* but a modest
surrogate-grad spiking LM still does not reach held-out language
competence at feasible local scale. The validated, multi-seed,
anti-cheat-validated **trustworthy grounded continual memory with
no-confabulation abstention** is untouched and remains the deliverable.

## Next (continuous autonomous arc — NOT a stop, NOT a config-crank)

Per the pre-registered branch + the user's standing directive
(autonomous, no stopping, no config-cranking a terminated mechanism):
proceed immediately to the pre-staged **Generator-D** — knowledge
distillation from a LOCAL open-weights teacher (training-time teacher
only; runtime = the trained spiking net, self-contained
post-training). Generator-D changes the *signal shape* (dense
soft-target distillation — the strongest known signal for making a
small model competent), a genuinely different mechanism, judged by the
SAME gate_core now hardened with the pre-registered
absolute-competence floor. Design pre-staged at
`docs/plans/2026-05-17-generator-D-distillation-PRESTAGED-design.md`.

## Files

- Mechanism: `research/runners/scaled_subword_lm_train.py`,
  `sim/bpe_tokenizer.py`, `research/runners/subword_lm_generate.py`,
  `research/runners/corpus_fetch.py`
- Gate: `research/runners/subword_lm_gate.py` +
  `research/runners/subword_lm_gate_core.py` (FIXED bars; the
  absolute-competence floor is being added as a forward pre-registered
  strengthening for Generator-D)
- Evidence: `research/findings/raw/g11_bg/generator_s_gate.json`,
  `_generator_s_gate.log`, `generator_s_grounding.json`
- Design/plan: `docs/plans/2026-05-17-scaled-subword-spiking-LM-{design,implementation}.md`
- Prior arc: the 7 prior generator/order-intrinsic NEGATIVE findings
