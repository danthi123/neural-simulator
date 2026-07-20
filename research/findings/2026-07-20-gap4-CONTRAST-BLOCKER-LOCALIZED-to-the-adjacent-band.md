# gap#4 — the contrast blocker LOCALIZED: it is ADJACENT-BAND only, and my depression is on the wrong lag axis

Output of the research gate on contrast (`wf_00b96ef1-b8a`, 4 sources read in depth), plus a zero-GPU re-score it
prescribed. **This converts "contrast is the blocker" into a specific, mechanistically-named defect with a
biologically-attested fix.**

## The gate's analytic model reproduces our measurement from our own code

Derived from the committed rule, not fitted to the result:
- eligibility trace peak/mean over a field window `W` is exactly `(W/tau)/(1 - e^{-W/tau})` → **2.3-4.1x**, the
  analytic ceiling of the current rule;
- the saturating update integrates to `w(E) = w_max - (w_max - w0)*exp(-c*E)` — a **compressive** map — and at the
  `c` that matches our observed pedestal predicts **peak/mean 1.6-2.0x**, which is what we measured (1.73x).

Falsifiable prediction it makes for free: **contrast falls monotonically with drive**, so training longer or harder
makes contrast *worse*. Untested; cheap.

## ⚠️ My reported metric was the wrong one — and the correction cuts BOTH ways

Milstein's own reconstruction of a CORRECT BTSP map gives **peak/mean 1.4-1.9x** while **peak/min is 6-8x**,
because a correct map is a narrow deep trough beside a peak on an untouched far field; his readout
(eLife 73046 Eqs 6-7) explicitly **subtracts the uniform baseline** before measuring the ramp.

**So our 1.73x peak/mean is INSIDE the range a correct map produces — it was never evidence of failure.**

Re-scoring the SAME layer-2 map (no new run):

| metric | ours | Milstein reference |
|---|---|---|
| peak/mean | **1.734x** | 1.4-1.9x ✔ in range |
| peak/min | **4.035x** | 6-8x — below |
| trough exists? | **YES** — 2/5 cells depressed below baseline (-43%, -35%) | yes |
| peak above baseline | **+131%** | — |

*(Metric-change discipline: this is not goalpost-moving. The metric was prescribed by an independent research gate,
citing Milstein's own equations, BEFORE it saw our numbers — and I report BOTH metrics, not just the flattering one.
Note it also makes us look WORSE on peak/min, which is the point of an honest metric.)*

## THE DEFECT, LOCALIZED — the deficit is entirely ADJACENT-BAND

The trough is real but **in the wrong place**. Ordering cells by distance from the peak:

| distance from peak | weight vs baseline | response contrast |
|---|---|---|
| adjacent (1 field away) | **ELEVATED** +80%, +33% | **1.21x** ← the deficit |
| far (2 fields away) | **DEPRESSED** -43%, -35% | **2.60x** ← already healthy |

**The weight map and the response show the identical signature.** Far-field contrast is fine. What is missing is
contrast against the *neighbours* — and neighbours are exactly what localizes a field.

**Cause (the gate's diagnosis, confirmed by the map):** my thresholded rule `max(theta - E, 0)/theta` depresses
**low-eligibility** synapses — which are the FAR ones — lowering a distant floor roughly uniformly. Milstein's
depression fires in a band **adjacent to the peak** (`alpha- < Omega < alpha+`, a depression-only window BETWEEN two
thresholds), carving a flanking trough. **I built the depression on the wrong part of the lag axis.**

## A second self-inflicted error the gate caught

`fused_btsp_hetero_update`'s docstring cites **Oja** as its competitive precedent. Oja's rule is **multiplicative** —
the family Miller-MacKay and Triesch both show **PRESERVES** ratios rather than sharpening them. The citation
inverts the result it was invoked to support. Every depression form tried so far (`1-E`, `max(theta-E,0)/theta`) is a
function of the synapse's **own** eligibility, so **nothing in the committed rule reads any other synapse's state,
and therefore nothing in it can cancel a common mode.** That is the analytic reason the pedestal survived both gates.

## Ranked next steps (from the gate, cheap-first)

1. **Rank 0c (zero GPU, do first):** measure the afferent adjacent-bin cosine. If a smooth position code gives
   cos ~0.99, no local rule can carve a sharp field and the arc is mis-scoped — the repo already has a validated
   5-module grid code that took adjacent cos 0.9921 → 0.7379. **FAILS IF cos <= 0.8** (input is fine, rule is the blocker).
2. **Rank 2 (top build): zero-DC difference-of-exponentials kernel.** A kernel with zero DC gain **cannot build a
   pedestal — algebraically, not by tuning**: `drive = E_fast - a_dep * E_slow`, with
   `a_dep = [tau_p(1-e^{-W/tau_p})]/[tau_d(1-e^{-W/tau_d})]`. Additive, default-off, byte-identical at `a_dep=0`.
3. **Rank 3: split-threshold (Milstein) depression band** — the adjacent-band trough this map is missing.
4. **Rank 4: mean-subtracted increment** (Miller-MacKay subtractive normalization) — `sum_j dw_ij = 0` by
   construction, no tuning. Needs a CSR row reduction (rank-1/common-mode, which this repo's own 2026-06-15 finding
   establishes is point-neuron-legal and already ships as `enable_input_mean_adapt`) — NOT the off-diagonal operation
   Mikulasch-Priesemann forbids. **Genuinely untried.**
5. **Rank 5 (prerequisite, not a fix): replacing eligibility traces.** `E <- max(E*lam, fired)` instead of
   `E <- E*lam + (1-lam)*fired`; the current trace is dominated by spike COUNT, not timing (one spike contributes
   ~5e-4), so a distant input that fires often outscores a near input that fires once. **The amplitude change is a
   ~2000x effective-learning-rate change and eta MUST be rescaled**, or this is the purest instance of the R2b
   confound the repo already burned.

## Scoring the prediction filed in advance (`29c6f897`)

Filed before any research ran: *"the fix must LOWER THE PEDESTAL via depression of non-coincident inputs —
bidirectional BTSP (Milstein 2021) — not more readout sensitivity."*

**PARTIALLY CONFIRMED.** Right family (Milstein bidirectional BTSP is the top-ranked biological mechanism) and right
about the readout (independently excluded twice). **Wrong about the sufficiency of "depression of non-coincident
inputs"** — that is precisely what was implemented, and it targets the FAR field. The correction the prediction
missed: the depression must be **positionally targeted at the band ADJACENT to the peak**, not at low-eligibility
inputs generally. A prediction that was directionally right and mechanistically incomplete.

---

## Rank 0c EXECUTED (zero GPU) — the arc is correctly scoped; the INPUT is fine at both layers

The gate's criterion: adjacent-bin afferent cos ~0.99 ⇒ no local rule can carve a sharp field (arc mis-scoped);
cos <= 0.8 ⇒ input is fine and the RULE is the blocker.

| layer | afferent | adjacent-bin cos | verdict |
|---|---|---|---|
| L1 | position pools (disjoint one-hot) | **0.0000** (min 0, max 0) | maximally separable |
| L2 | the LEARNED CA1 population code | **0.7436** (min 0.500, max 0.833) | below the 0.8 bar |

**Both pass. The input is not the blocker at either layer — the rule is.** This arc is correctly scoped, and the
alternative hypothesis (a smooth position code needing the repo's 5-module grid fix) is REFUTED for this setup.

**A positive finding falls out of it:** the repo's validated grid-code work took adjacent cos from 0.9921 to
**0.7379**. BTSP's own learned CA1 map lands at **0.7436** — essentially the same decorrelation, achieved by the
plasticity rule itself rather than by an engineered input code. **The rule DOES decorrelate; what it does not do is
carve the adjacent-band trough.** Those are separable properties and only the second is missing.

⇒ Diagnosis complete and singular: **everything required is in place EXCEPT positionally-targeted adjacent-band
depression.** Proceed to build it (Rank 3, Milstein split-threshold band), with Rank 2's zero-DC kernel as the
fallback whose pedestal-cancellation is algebraic rather than tuned.
