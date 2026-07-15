# Controlled-lag recurrent e-prop — the copy task's horizon-extension is a MEMORY-TIMESCALE ARTIFACT (the anti-cheats caught it on a controlled task); the discriminator must require RECURRENT COMPUTATION

**Date:** 2026-07-14 · **Status:** de-risk BUILT + eligibility grad-check-validated; the COPY-task result is an honest artifact (not genuine credit-driven learning); the recurrent-computation task is the pinned next step. numpy CPU; NO `sim/` edit.

## Context

The emergence-engine recurrent-language-cortex frontier: get a spiking recurrent cortex to LEARN long-range structure from a stream via a biological LOCAL rule. The prior attempt was adversarially REFUTED (`2026-07-14-eprop-recurrent-synthesis-CONTROLS-REFUTED`): the WikiText "deep-context" win was a credit-direction-INDEPENDENT memory-timescale artifact. The deep-research gate (Bellec-2020 e-prop, read in depth) diagnosed: the eligibility was already correct (faithful ALIF 2-component form); the failure was the TASK (a gameable next-token-CE metric with no controlled long-range dependency) + running the decisive controls on the short-horizon LIF arm. The gate's fix: a **delayed cued-recall (copy) task with a controlled dependency length T** on the **ALIF** arm, with genuine-learning anti-cheats.

## What was built

`research/runners/_reslm_controlled_lag_eprop_derisk.py` — the copy task `[STORE, x, f_1..f_T, RECALL, x]` (score predicting x at the RECALL position; provably beats no n-gram — the last n<T tokens are fillers/RECALL, independent of x) + an ALIF e-prop trainer carrying the **finite-difference-validated** Bellec-2020 2-component eligibility (`grad_check_alif` PASS: eps_h/eps_a vs local FD exact to ~1e-9; the e-prop total-grad vs full-FD residual is the expected off-diagonal-truncation, converging at tight spectral radius) + all credit-direction arms + the RECALL-position metric + a T-sweep. Reuse-by-import of the validated `RateReservoir` (ALIF).

## The result (seed 42)

- **T=5:** fixed=0.950 ≈ plastic=1.000 — the fixed ALIF reservoir's intrinsic adaptation memory ALREADY holds x for 5 steps → no discrimination (trivially within the fixed horizon). n-gram 0.190 ≈ chance 0.167 ✓; cue_scramble 0.115 (collapses) ✓ — the task is sound.
- **T=15 (fixed partial 0.365), lr_rec=0.01:** plastic=0.140 (WORSE than fixed, below chance) — the e-prop W_rec updates DESTABILIZE the good reservoir dynamics at the default lr.
- **T=15, lr_rec=0.0002 (tuned):** plastic=**0.730** EXCEEDS fixed=0.365 — BUT the genuine-learning **anti-cheats FAIL**: symmetric (TRUE gradient)=**0.140** HURTS (should be the ceiling ≥ plastic), sign_flip=**0.520** does NOT collapse to chance. ⇒ the SAME memory-timescale-artifact signature as the WikiText refutation, now on a controlled task.

## Diagnosis (honest)

The single-cue COPY task is solved by the ALIF **adaptation state directly** (the read-out reads `[h, a]`, and `a` holds x), so a learned W_rec is UNNECESSARY within the fixed horizon and only PERTURBS the good dynamics — a bidirectional memory-timescale nudge (any structured W_rec change helps a bit; the random-FA direction happens to help more, its negation helps less, the true gradient over-optimizes the shallow loss and HURTS recall). This is exactly the artifact class the anti-cheats were built to catch — and they caught it. **The de-risk WORKED: a single-cue hold task does NOT elicit genuine credit-driven recurrent learning; W_rec learning is not load-bearing for it.**

## The pinned next step (the real discriminator)

The task must REQUIRE recurrent computation a memory-timescale nudge cannot fake: the answer must be a **nonlinear function of temporally-separated cues** that the learned recurrence MUST compute — e.g. **delayed-XOR** (STORE x1 … x2 … RECALL → x1⊕x2) or **evidence-accumulation** (multiple weak cues → majority). On such a task the fixed reservoir cannot solve it by holding (no single value to hold), so genuine plastic learning (if it works) yields the true signature: **plastic > fixed AND symmetric ≥ plastic AND sign_flip collapses AND cue_scramble collapses**. This is Bellec's actual store-recall/evidence-accumulation regime (which e-prop+ALIF solves) — vs the copy task, which the ALIF alone solves. Build the XOR/accumulation task mode into the runner + re-run the arms.

## The recurrent-computation (XOR / modular-sum) task — TESTED: a genuine e-prop limitation, NOT under-training

Added the `xor` task mode: `[STORE, x1, f_1..f_T, x2, RECALL, y]` with y=(x1+x2) mod K — the answer is a NONLINEAR
function of TWO temporally-separated cues (a linear read-out over [held-x1, recent-x2] provably cannot compute it), so
the learned RECURRENCE must combine them and a memory-nudge cannot fake it. Result (seed 42, K=4, chance 0.25):

- **T=2/T=5, n=100:** fixed ≈ plastic ≈ symmetric ≈ sign_flip ≈ 0.33 — ALL arms flat, barely above chance. No learning
  signal; W_rec learning does nothing useful; the true-gradient (symmetric) does NOT beat fixed.
- **Under-training RULED OUT (n=200, epochs=50):** fixed=0.270 ≈ plastic=0.260 ≈ symmetric=0.265 ≈ chance — MORE
  scale/epochs does NOT learn it, and the TRUE-GRADIENT direction is still at chance.

⇒ **the modular-sum recurrent computation is NOT learnable here, and it is NOT under-training** — even the true
spatial-gradient direction (symmetric e-prop) is at chance. This is the genuine limitation the deep-research gate
flagged from Bellec-2020 + Marschall-Savin: **e-prop is a pure DIAGONAL approximation of RTRL that zeroes the
off-diagonal cross-neuron influence — exactly the credit needed to COMBINE two temporally-separated cues.** The ALIF
long-memory is validated for HOLDING a cue (the copy task) but does not supply the cross-neuron credit for COMBINING
cues. (Consistent with the field: long-range/compositional recurrent learning from a local/transport-free rule is
genuinely open at LRA scale; SnAp keeps a rank-reduced non-diagonal approximation, FPTT+LTC reaches ~784 steps but is
not transport-free — the gate's ranked ceilings.)

## Bottom line (comprehensive)

The controlled-lag de-risk is BUILT + grad-check-validated, and it COMPREHENSIVELY characterizes the emergence-engine
recurrent-language-cortex frontier's core difficulty on this substrate: the biological local rule (transport-free
e-prop + ALIF) learns **NEITHER** (a) a genuine single-cue HOLD — the copy task is solved by the ALIF adaptation and
plastic e-prop is only a memory-timescale ARTIFACT (the anti-cheats caught it: symmetric hurts, sign_flip doesn't
collapse), **NOR** (b) a genuine cross-cue RECURRENT COMPUTATION — the XOR/modular-sum is a flat null even with the
true-gradient direction and more scale (e-prop's diagonal-RTRL truncation drops the cross-neuron credit). The
anti-cheats + the under-training tiebreaker make this a clean, honest characterization, not a tuning miss.

## POSITIVE CONTROL (evidence-accumulation) — the implementation is VALIDATED; e-prop's W_rec learning is not load-bearing on this substrate

Added the `accum` task (Bellec's OWN validated e-prop+ALIF task): a stream of T LEFT/RIGHT cues + fillers, RECALL → the
MAJORITY side (recurrent INTEGRATION, not cross-cue combination). Result (seed 42, T=5, chance 0.5): **fixed=0.875,
plastic=0.805, symmetric=0.820, sign_flip=0.835** (all well above chance); cue_scramble=0.425 (collapses ✓);
ngram=0.510 (chance ✓). ⇒ **the substrate + implementation CORRECTLY solve a recurrent-integration task** — so the XOR
null is a genuine nonlinear-combination limitation, NOT a broken implementation (the positive control validates the
negative). BUT the **FIXED reservoir already solves accumulation** (linear integration is native to a random recurrent
net; the read-out thresholds the running sum) → W_rec learning is not needed for it either.

## The comprehensive, validated picture (3 tasks)

| task | what it needs | fixed reservoir | e-prop plastic | verdict |
|---|---|---|---|---|
| **copy** (single-cue HOLD) | hold x for T | 0.95 (ALIF holds it) | artifact (sym HURTS, sign_flip helps) | fixed-solved; plastic = memory-nudge artifact |
| **accum** (INTEGRATE) | count/majority | 0.875 (native linear integration) | 0.805 ≈ fixed | fixed-solved; validates the implementation |
| **xor** (nonlinear COMBINE) | (x1+x2) mod K | 0.25 = chance | 0.26 = chance (not under-training) | NEITHER fixed nor e-prop can do it |

⇒ **on this reservoir substrate with a direct-read read-out, e-prop's recurrent W_rec learning is NOT demonstrably
load-bearing:** the fixed reservoir natively handles the LINEAR recurrent functions (hold via ALIF, integrate via the
recurrent sum), and NEITHER the fixed reservoir NOR e-prop-learned W_rec supplies the NONLINEAR cross-cue combination
(XOR) — because e-prop's diagonal-RTRL truncation drops exactly the off-diagonal cross-neuron credit that combination
needs. This is a clean, positive-control-validated characterization of the frontier's core difficulty.

## ⚠️ SELF-CAUGHT CORRECTION (2026-07-14, same session — the discipline caught a wrong conclusion BEFORE building the wrong mechanism)

Before building SnAp-1 (the "less-truncated credit rule" I had pinned as the fix for the XOR null), I tested the
research gate's alternative hypothesis MYSELF (a0): **is the XOR null a LINEAR-READOUT limitation, not a
recurrent-credit limitation?** The one-line test — a 2-layer MLP read-out on the FIXED (untrained-W_rec) reservoir:

- **T=2, T=5: MLP acc = 1.000** (chance 0.25) on BOTH seeds → the fixed reservoir's features ALREADY CONTAIN both cues;
  a LINEAR softmax simply cannot COMBINE them (XOR isn't linearly separable), but a nonlinear read-out solves it
  perfectly. ⇒ **the XOR "e-prop limitation" at short lag was a LINEAR-READOUT limitation — the recurrent W_rec credit
  was NEVER the bottleneck, and SnAp-1 is the WRONG mechanism.**
- **T=15: 0.680; T=30: 0.295; T=60: 0.265 ≈ chance** → beyond the ALIF preservation horizon (~5-15 tokens) even a
  nonlinear read-out fails because the distal cue x1 is genuinely LOST from the fixed reservoir's features.

**⇒ CORRECTED conclusion.** The earlier "e-prop can't learn the cross-cue recurrent computation" framing was WRONG
(it conflated the read-out limitation with a recurrent-credit limitation). The accurate picture: the fixed ALIF
reservoir SUPPLIES the features for both HOLD (copy) and COMBINE (XOR) up to its intrinsic preservation horizon
(~5-15 tokens); a nonlinear read-out extracts them; and e-prop's recurrent W_rec learning is genuinely NOT
load-bearing for any of this. **The one genuine frontier is HORIZON EXTENSION** — can LEARNING (recurrent W_rec)
extend the distal-cue preservation BEYOND the fixed ALIF horizon (so XOR at T=30+ becomes solvable)? My copy-task
result (plastic degrades/nudges rather than extends) suggested no, but that used a LINEAR read-out — the properly-
framed open test is **plastic recurrent learning + a nonlinear read-out at T=15-30 (does learning push the horizon
out?)**. This is the corrected, precise open question. SnAp-1 is NOT indicated (the short-T bottleneck was the
read-out); the horizon-extension question is about the ALIF/recurrent MEMORY, not the cross-neuron combination credit.

**Methodology win:** testing the alternative hypothesis (the nonlinear-readout reframe) BEFORE building the pinned
mechanism caught a wrong conclusion + saved the SnAp-1 build — the same adversarial-self-check discipline that caught
the WikiText over-claim. The finding above (the copy artifact, the accum positive control, the grad-check) all STAND;
only the XOR *interpretation* is corrected.

**Pinned next (corrected):** (1) the horizon-extension test — plastic recurrent e-prop + a NONLINEAR read-out at
T=15-30 (does learning extend the fixed ALIF's ~15-token preservation horizon?); (2) a harder task where the fixed
reservoir does NOT already supply the features (learned OUTPUT-GATING); (3) accept the ALIF-horizon as the substrate's
memory limit and scale the adaptation windows. The honest field state stands: long-range recurrent MEMORY from a
biological local rule is the genuine open frontier — now precisely characterized (readout vs recurrent-memory
disentangled) on a positive-control-validated harness.
