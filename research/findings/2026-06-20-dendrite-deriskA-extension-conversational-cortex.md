# Dendrite de-risk A — EXTENSION to the conversational learned cortex: honest NEGATIVE (the graded-read-out unlock does NOT generalize to this wall; the conversational cortex is a per-input-NORMALIZATION problem, a DIFFERENT dendritic unlock) (2026-06-20)

**Item:** extend the GO dendrite de-risk A (`2026-06-20-dendrite-derisk-A-graded-plateau-readout.md`, `ac92a79d`)
from the nav value-critic to the OTHER instance the dendrite scoping (`2acebf6b`, controller-verified) classed
as the same family — the conversational "learned graded cortex" (D2 Phase 2), the other "graded read-out of a
distributed code." **Question:** does the SAME graded dendritic-plateau read-out recover a graded/faithful
read-out of the LEARNED concept code where the point-neuron read-out (D2 Phase 2) returned an honest NEGATIVE?

> **Verdict: NEGATIVE — unanimous 6/6 seeds, all controls clean, NO `sim/` edit (Stage 0 reuse-by-import).**
> The graded dendritic-plateau read-out (de-risk A's mechanism) on the RAW conversational concept-code sits at
> the point-neuron floor (L1 fixed-W +0.044, L2 unsupervised-learned-W +0.039 ≈ the point-neuron all-or-none
> read-out +0.037 ≈ the raw-profile cosine ~0), where the host PPMI+SVD ceiling is +0.959. **The de-risk-A
> graded-read-out unlock does NOT generalize to this wall.** The attribution ladder localizes the real cause:
> the conversational cortex NEGATIVE is a **COMMON-MODE / per-input-NORMALIZATION** problem — only the SEPARATE
> per-hub divisive normalization (the D2 Phase-1 per-presynaptic-source dendritic gain) recovers it (the
> normalized residual read LINEARLY reaches +0.832; a graded read-out on top neither helps nor is load-bearing).
> **⇒ a decisive, build-saving result: the graded read-out and the per-input normalization are DISTINCT
> dendritic unlocks, not interchangeable.** De-risk A's GO stands for the nav value-critic; it does NOT transfer
> to the conversational cortex, because the nav place-code input has no common mode while the conversational
> concept code is common-mode-dominated.

## Why the two instances are NOT the same problem (the mechanistic root)

The dendrite scoping put both under "a graded read-out of a distributed code," but the two distributed codes
differ in a load-bearing way:
- **Nav value-critic (de-risk A, GO):** the input is the **grid-32 place population code** — already
  decorrelated, no common mode. The read-out's ONLY job is to express a graded VALUE continuum (near>mid>far)
  that the point-neuron somatic spike rate provably can't (0 sub-rheobase, or saturated all-or-none). The
  graded plateau read-out is exactly the right tool, and it works.
- **Conversational cortex (D2 Phase 2, this extension):** the input is the **concept×hub co-occurrence count
  code**, which is **common-mode-dominated** — high-frequency COMMON hubs ("said","day","big": every concept
  connects to them strongly) dominate every concept's profile, so the raw-profile cosine recovers ~0
  (Pearson +0.044). The binding constraint is **removing the common mode** (a per-INPUT normalization), NOT
  expressing a graded read-out. A graded read-out of a common-mode-dominated input is still common-mode-
  dominated.

So the graded-read-out unlock (the de-risk-A mechanism) addresses the value-continuum problem but is the wrong
tool for the common-mode problem. The right dendritic tool for the common-mode is the **per-presynaptic-source
divisive gain** — a DIFFERENT dendritic computation (D2 Phase 1; built + verified, but its *necessity* on the
spiking substrate was itself an honest NEGATIVE because the spiking threshold + temporal integration substitute
for it — `2026-06-14-D2-phase1-DONE-phase2-frontier.md`).

## The D2 Phase-2 NEGATIVE this builds on

The task: recover an a-priori category structure `S_true` from the concept×hub count matrix whose common hubs
dominate. The decisive metric: **Pearson(cos(codes), S_true)** — a graded category-similarity fidelity vs the
host PPMI+SVD ceiling (+0.96 on the counts). The point-neuron forward-pass read-out (random projection +
Izhikevich dynamics) gave codes **anti-correlated** with `S_true` (≈ −0.07) at every setting; the deeper limit
is the spiking rheobase threshold **silencing the low-count category hubs**. The graded read-out (smooth,
non-saturating, no hard threshold) is the natural candidate fix for the *silencing* — which is exactly why this
extension was the right next test, and exactly why the NEGATIVE is informative: the silencing was not the only,
nor the binding, failure mode.

## The ladder — all read-outs read the IDENTICAL common-mode-dominated count code (anti-cheat d: no host-norm smuggling)

| read-out | what it is | mean Pearson(cos,S_true) | mean held-out gen (chance 0.125) |
|---|---|---|---|
| HOST ceiling | PPMI+SVD on the counts (labelled instrument; the data carries it) | **+0.959** | — |
| **L0 POINT-NEURON all-or-none** | Heaviside threshold of `v_basal` (the somatic-spike read-out — the validity gate) | **+0.037** (fails) | 0.20 |
| **L1 GRADED plateau, fixed-W** | the de-risk-A mechanism `V=σ((v_basal−θ)/slope)`, fixed-random W, NO norm, NO learning | **+0.044** | 0.24 |
| **L2 GRADED plateau, learned-W** | + W learned UNSUPERVISED (local Urbanczik-Senn, self-apical, NO category label) | **+0.039** | 0.22 |
| L3 per-hub NORM → linear | the SEPARATE D2-Phase-1 per-input divisive gain, residual read LINEARLY | **+0.832** | 1.000 |
| L3 per-hub NORM → graded | + the graded plateau read-out on top of the normalized residual | +0.628 | 1.000 |

**The de-risk-A graded read-out (L1/L2) sits at the point-neuron floor (~+0.04 ≈ L0 +0.037 ≈ raw +0.044), where
the host is +0.959.** Only the per-hub NORMALIZATION (L3) recovers the structure — and there the GRADED read-out
is not the load-bearing piece (L3-linear +0.832 ≥ L3-graded +0.628; the graded read-out neither helps nor is
needed once the common mode is removed).

## Per-seed (faithful, 6 seeds; the synthetic D1/D2 regime — calibrated so the point neuron fails + the host carries)

| seed | host | L0 (pn) | L1 (grd-fix) | L2 (grd-lrn) | best-graded | best-graded all-or-none lesion | L3-norm-linear | L3-norm-graded |
|---|---|---|---|---|---|---|---|---|
| 42  | +0.957 | +0.025 | +0.037 | +0.041 | +0.041 | +0.037 | +0.863 | +0.640 |
| 43  | +0.957 | +0.008 | +0.031 | +0.040 | +0.040 | +0.026 | +0.827 | +0.613 |
| 44  | +0.964 | +0.038 | +0.044 | +0.056 | +0.056 | +0.042 | +0.843 | +0.646 |
| 100 | +0.956 | +0.064 | +0.057 | +0.018 | +0.057 | +0.064 | +0.811 | +0.624 |
| 101 | +0.964 | +0.054 | +0.058 | +0.046 | +0.058 | +0.054 | +0.844 | +0.633 |
| 102 | +0.956 | +0.035 | +0.036 | +0.035 | +0.036 | +0.035 | +0.804 | +0.612 |
| **mean** | **+0.959** | **+0.037** | **+0.044** | **+0.039** | **+0.048** | **+0.043** | **+0.832** | **+0.628** |

## ANTI-CHEAT table (the de-risk-A battery + the D2 controls — 6 seeds)

| anti-cheat | result | reading |
|---|---|---|
| **(a)** POINT-NEURON CONTROL (L0) re-asserted IN-RUN and FAILS (the validity gate) | **6/6** fail (mean +0.037 ≤ 0.15) | the regime is correctly calibrated; the substrate genuinely can't from the raw counts |
| **(b)** HOST CEILING carries (the data has the structure) | **6/6** (mean +0.959 ≥ 0.30) | a failure is the MECHANISM, not the data |
| **(c)** GRADED-NESS LOAD-BEARING (the best graded arm's all-or-none lesion collapses it) | **0/6** load-bearing | the graded-ness is NOT load-bearing here (L1 ≈ L2 ≈ its own all-or-none lesion ≈ +0.04) — the de-risk-A discriminator FAILS to fire because there is no graded signal to carry |
| **(d)** NO HOST-NORMALIZATION SMUGGLING (L0/L1/L2 read the RAW counts; only L3 normalizes) | enforced by construction | the graded read-out is tested on the genuine common-mode-dominated code, not a pre-whitened one |
| **(e)** PERMUTED-S collapses (shuffle same-category → recovery ~0) | **6/6** clean (L3-linear & L1-graded ≈ 0) | the L3 recovery is the real category structure, not cosine geometry |
| **(f)** S_true A-PRIORI (constructed block, never data-derived) + L3 gains LEARNED ONLINE (converge, track hub frequency +1.00) + multi-seed | 6/6 | the test + the L3 control are honest |
| **graded BEATS the point neuron** (the GO test) | **0/6** | the graded read-out does not beat the point neuron — it does not generalize to this wall |

## Honest scope + what is / isn't claimed

1. **What is NEGATIVE:** the de-risk-A **graded dendritic-plateau read-out** does NOT recover the conversational
   concept-code's graded category-similarity from the raw common-mode-dominated counts — it sits at the point-
   neuron floor (~+0.04 vs host +0.959), and its graded-ness is not load-bearing (the de-risk-A anti-cheat (b)
   discriminator does not fire — L1 ≈ L2 ≈ the all-or-none lesion). This is a clean, unanimous, multi-seed
   NEGATIVE for the graded-read-out unlock on THIS wall.
2. **The attribution is decisive (the real cause):** the conversational cortex NEGATIVE is a **COMMON-MODE /
   per-input-NORMALIZATION** problem — only the SEPARATE per-hub divisive normalization (the D2 Phase-1
   per-presynaptic-source dendritic gain) recovers it (L3-linear +0.832, gen 1.000). The graded read-out is
   the wrong dendritic tool; the per-input normalization is the right one.
3. **What is NOT retracted:** de-risk A's GO stands for the **nav value-critic** (the place-code input has no
   common mode; the graded read-out's job there is the value continuum, and it works 6/6). This extension does
   not weaken that; it bounds its scope.
4. **The build-saving value:** this **sharpens the dendrite's capability map** — the dendrite affords at least
   TWO distinct unlocks (a graded analog READ-OUT, and a per-input NORMALIZATION) and they are **not
   interchangeable**. Greenlighting Stage 1 (the graded-plateau bridge read-out) on the strength of de-risk A
   would NOT have fixed the conversational cortex — this extension establishes that the conversational wall
   needs the *other* dendritic mechanism (already built as D2 Phase 1, whose on-substrate *necessity* is itself
   an honest NEGATIVE — the spiking threshold + temporal integration substitute for it). So the deep
   conversational frontier remains a *learned* graded embedding that doesn't rate-code raw counts one-shot, not
   a graded read-out.
5. **Why the unsupervised-learned arm (L2) doesn't rescue it (the honest reason):** de-risk A had a LEGITIMATE
   external teacher (the SNc reward delta, location-selective). The conversational code structure must emerge
   WITHOUT category labels (the brain gets no `S_true`), so the only legitimate apical signal is self-generated
   — and a self-supervised local rule on the raw common-mode-dominated input cannot, on its own, discover the
   common-mode-removal that the structure requires (L2 +0.039, at the floor). Injecting `S_true` as the apical
   teacher would smuggle the answer (a cheat) — so it was not done.
6. **The Stage-0/Stage-1 boundary held: NO `sim/` edit.** The whole ladder is composed from existing modules
   (`sim/dendritic_neuron.DendriticLayer`, `sim/dendritic_plasticity.urbanczik_senn_update`) + the D1/D2 count
   builder + the host PPMI lens. No `sim/` edit was needed or made.

## Reproduce

```bash
# Faithful 6-seed (CPU numpy; fast ~2 s total): the ladder + anti-cheats + the NEGATIVE verdict
SIM_BACKEND=numpy python -u -m research.runners._dendrite_deriskA_extension_conversational_cortex \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_dendrite_deriskA_ext_conv_cortex.json
# CPU smoke (single seed):
SIM_BACKEND=numpy python -u -m research.runners._dendrite_deriskA_extension_conversational_cortex --seed 42
```

Raw: `research/findings/raw/_dendrite_deriskA_ext_conv_cortex.json` (+ `_..._smoke.json`). Runner:
`research/runners/_dendrite_deriskA_extension_conversational_cortex.py`.
