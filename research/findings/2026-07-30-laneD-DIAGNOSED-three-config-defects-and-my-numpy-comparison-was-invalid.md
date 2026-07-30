# Lane D DIAGNOSED: three config defects saturate every synapse — and my "numpy toys sit at ceiling" reading was invalid

**Date:** 2026-07-30 · Every claim below was read out of opened source with file:line, not inferred.
**Corrects:** [`2026-07-30-density1-destroys-learning-structure-and-laneD-6seed-NEGATIVE.md`](2026-07-30-density1-destroys-learning-structure-and-laneD-6seed-NEGATIVE.md) §2.

## ⛔ FIRST, THE CORRECTION I OWE

I recorded lane D's 6/6 NEGATIVE as *"the board's own methodology lesson repeating exactly: off-substrate toys sit
at ceiling while the substrate spreads the same configs."* **That is WRONG.** It was never the same mechanism.

The numpy GO carried FIVE ingredients, **none** of which was ported to the bridge:
1. **ZCA whitening** — its own comment (`_b1_v1_selforg_rf_derisk.py:250-252`) says *"without this a Hebbian rule
   learns the dominant low-frequency mode, not oriented structure; documented failure"*.
2. **SIGNED zero-mean patches AND weights** — header `:97-104`: *"learning on the non-negative ON/OFF cone with
   weak inhibition collapses to all-positive blobs -- the documented failure"*. That is EXACTLY the on-bridge
   regime (`hebbian_min_weight=0.0`, `stdp_w_min=0.0`, `w_init=np.abs()`).
3. per-filter **L2 renormalization** every batch (zero-sum competition).
4. a per-sample quantile **sparse soft-threshold** (competitive k-WTA).
5. **Foldiak anti-Hebbian lateral inhibition.**

Two of these are off because of a **DEFAULT ARGUMENT** — `--n-inh` defaults to **0** — while the runner's own
docstring (`:176-180`) names lateral inhibition *"the ingredient the numpy mechanism A used for orientation
SELECTIVITY"*. The other three have no on-bridge counterpart at all. This is the standing **"an absent flag means
DEFAULT, not OFF"** trap, and I walked into it while writing the finding that quoted the runner.

⇒ The comparison was invalid, so the negative says nothing about substrate-vs-numpy. Two different mechanisms.

## Defect A — an unreachable set-point turns synaptic scaling into a uniform saturator

`sim/bridge.py:8703-8705`: `scale_factors = 1 + synaptic_scaling_rate*(homeostasis_target_rate - activity_ema)`,
applied **multiplicatively and uniformly to every synapse of a postsynaptic cell**. Gated only on
`_homeostasis_gated`, **not** on spiking — it runs all 40,000 steps.

The runner sets `homeo_target=0.012` and `syn_scaling_rate=0.02` (20× the project default 0.001, which
`sim/config.py:576` annotates *"Slow scaling rate (operates on seconds timescale)"*). Measured V1 firing rate is
**0.0004-0.0010**, so the rate error is permanently **+0.0112** — a 1.000224× upward multiplier every step,
forever. Net over 40k steps ≈ 3.50e3 against a ceiling only 2.33× above init ⇒ **every retina→V1 synapse pins at
`hebbian_max_weight=70` by step ~4,150**, i.e. for 90% of development.

The error can never close: threshold homeostasis moves the spike threshold only −1.79 mV against a 25 mV range.
**The two homeostats are mismatched by ~3 orders of magnitude, and the effective one is the non-selective one.**

**Why it reads "worse than random":** the RF is read as the SIGNED difference `W[ON] − W[OFF]`. Both channels
pinned at 70 ⇒ difference identically 0 ⇒ structure-tensor trace < 1e-9 ⇒ OSI returns **exactly 0.0**. Random init
scores ~0.17 by chance. Learning does not fail to build — **it deletes the chance structure**.

**The odd signature is fully explained as an artifact.** OSI mean falls 0.169→0.069 while `frac>0.5` RISES
0.0035→0.0165: a bimodal population, ~70% identically flat and ~30% retaining a few unsaturated pixels that a
structure tensor reads as maximally anisotropic. Two independent statistics give the same survivor fraction
(0.29 and 0.33). **It is not emerging orientation.**

**The decisive discriminator:** learned and shuffle-control converge to within 0.011-0.019 OSI on all 6 seeds, and
BOTH sit 2-3× below random init. If the rule simply could not extract orientation, the arms would differ. They do
not, because the destroying force is **input-blind**. Their near-equality is not evidence about the mechanism —
it is evidence that neither arm ran one.

## Defect B — the rule's fixed point does not depend on the input

`sim/bridge.py:7697-7698`: `delta = hebbian_learning_rate * coact_j * (hebbian_max_weight - w_j)`.
Set `dw=0`: `coact_j` factors BOTH the drive and the decay, so it **cancels**, giving `w_j* = hebbian_max_weight`
for every gated synapse regardless of drive. Coactivity sets only the RATE of approach, never the destination.
**A rule whose fixed point is a constant cannot represent a graded receptive field, at any operating point.**

The numpy Oja rule by contrast: `dW = (a.T@X − (a*a).sum(0)[:,None]*W)/B` — potentiation carries the
synapse-specific `x_j` while decay carries only per-unit `a²`, so `w_j* = <a·x_j>/<a²>` = the input correlation
= **the RF shape itself**. That is the entire difference.

Corollary: there is **no input-specific depression anywhere in this path**. Both decreasing terms
(`bridge.py:7744` uniform array decay; `:8711-8719` per-post multiplier) preserve each RF's relative profile
exactly. Nothing in the configuration can carve an RF's SHAPE — only its scale.

## Defect C — the constructive force was gated off by config

`sim/bridge.py:7692-7694` gates on `trace[row]*trace[col] > hebbian_coactivity_thresh`. Runner sets
`coact_thresh=0.03` against its own `homeo_target=0.012` — so even at PERFECT homeostasis the max mean coactivity
(0.012) is **2.5× below the threshold**; at the measured 0.0007 it is **43× below**. Total learning signal
delivered: ~32 spikes per V1 cell across ~98 afferents for all of development.
**Input-blind forces executed ~40,000 times; the input-specific force ~100 times.**

## Symmetry — no orientation basis can form even in principle

All 32 V1 cells sharing a retinotopic position get an IDENTICAL support list
(`build_isotropic_support:104-133`), zero lateral inhibition, and a FULL-FIELD grating, so they receive identical
drive. Nothing breaks their interchangeability.

## Biology says the stimulus is also wrong (Kandel 6e Ch 49, opened, ~760 lines read)

p.1218: *"Only when the optic nerves were stimulated asynchronously were ocular dominance columns established."*
p.1227: *"When all of the axons fire in synchrony, the tectum cannot determine which axons are neighbors."*
The B1 stimulus is the **strobe condition** — a full-field grating at 100% duty cycle with no blank — whereas real
retinal waves are LOCAL propagating foci with silent periods lasting minutes. Kandel also puts input-specific
WEAKENING first, with homeostatic compensation only *"over the following few days"*; the bridge inverts this,
running its homeostat at ~10× the cadence of its correlational term.

## Status and next

The adversarial round returned `survived: false` (3/3 refuted) on the synthesized *recommendation*, so no fix is
adopted here. The DIAGNOSIS above is a separate matter: it is read from opened source with line numbers and is
independently checkable, and the two arithmetic predictions (saturation by step ~4,150; survivor fraction ~0.3
from two statistics) are falsifiable by direct measurement of the weight matrix.

**Before any mechanism work: measure whether the weights are in fact pinned at 70.** That single readout confirms
or kills Defect A, and Defects A-C are configuration, not substrate — so the capability is untouched, exactly as
NO-DEFER requires.
