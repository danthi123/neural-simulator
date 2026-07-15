# Fresh deep-credit gate (5-agent Workflow, a-1 + external 2025-26 field + biology + reuse): feedforward spiking deep credit is ALREADY GO (e-prop + population coding) — the genuine open frontier is RECURRENT OFF-DIAGONAL cross-neuron temporal credit, with a cheapest-first 2-stage plan (recalibrate e-prop → then MDGL)

**Date:** 2026-07-15 · **Status:** deep-research gate DONE (read-only, adversarial-verify-style multi-lens). Corrects a STALE a-1: my earlier "learned-feedback-on-spikes = next" was inside the exhausted FA family — but the scout surfaced the 2026-07-14 e-prop SURPASS I'd missed. NO build yet — this scopes the next de-risk.

## The correction (the scout caught my stale premise)

I had re-anchored the deep-credit frontier as "FA family exhausted → the fix is learned feedback (Kolen-Pollack), untried on-spikes." **The scout of our OWN record corrected it:** the FA-family boundary was **SURPASSED on 2026-07-14 by e-prop** (transport-free forward eligibility + membrane surrogate + DFA): it trains the depth-2 compositional-inheritance CLASSIFICATION task, and **ports to the production Izhikevich bridge with population coding to the LIF ceiling** (K=1 0.47 → K=8 0.877 ≈ LIF 0.89), anti-cheat-clean, NO `sim/` edit. *"The parked 'spikes can't do deep credit / SNR wall' verdict is COMPREHENSIVELY REFUTED."* ⇒ **feedforward multi-layer spiking deep credit is NOT a boundary — it is GO.** Don't re-derive it; don't re-run the exhausted family (FA/burstprop/microcircuit/graded/DECOLLE/pool-k/node-perturbation[retired]/BurstCCN-preset — all 0/6 at cheap scale, mapped).

## The genuine open frontier (narrower, recurrent-side)

The e-prop GO is feedforward + on a toy 3-4 orders below language scale (necessary-not-sufficient). The **RECURRENT** form (e-prop training `W_rec`) was BUILT then REFUTED (`2026-07-14-eprop-recurrent-synthesis-CONTROLS-REFUTED` + my own `2026-07-15` selective-SSM trigram-bound finding): the "beats bigram/deep-context" win was a credit-direction-INDEPENDENT memory-timescale artifact. The open question: can a spiking recurrent cortex learn a genuine cross-neuron temporal dependency that e-prop's **diagonal** RTRL (which zeroes the off-diagonal ∂hₖ/∂hⱼ term) cannot?

## The ranked NEW mechanism classes (external 2025-26 field)

1. **MDGL — cell-type-specific one-hop cross-neuron modulation** (Liu et al PNAS 2021, `github.com/Helena-Yuhan-Liu/MDGL-main`; 2026 neuromod-diffusion successor arXiv:2603.08949). Adds the FIRST off-diagonal term the whole family omits: each neuron emits a cell-type-specific neuropeptide-like learning signal to its DIRECT synaptic partners → a synapse's update sees its postsynaptic partners' loss-contribution. Single-phase, transport-free, spiking-compatible. **Reuses `sim/neuromodulators.py` (scope-by-group broadcast) + the e-prop port; NO `sim/` edit** for the de-risk. Maps onto the project's shared spiking-DA volume-transmission core.
2. **Forward gradient / activity-perturbation** (first-order JVP + local losses; Ren-Kornblith-Liao-Hinton ICLR 2023) — distinct from the retired zeroth-order node-perturbation; unbiased; small-hidden = favorable variance. A feedforward second bet.
3. **Forward-Forward SNN** (2026; beats backprop on temporal SHD) — no error signal; local-greedy caveat (says more about task local-separability than deep credit).
4. **FPTT-spiking** (forward-prop-through-time) — the horizon-extension lever; compose with MDGL (temporal × spatial).
5. **SoftHebb** — unsupervised feature side, not task-directed deep credit.
- **RULED OUT (single-phase constraint):** predictive-coding/iPC, µPC/EqProp (need a settling/relaxation phase).

## The biology prior (decisive, shapes the plan)

The biology lens converges: cortex likely does NOT compute the off-diagonal Jacobian online — it approximates via **REPLAY** (the project's own validated SWR-replaces-BPTT lever), long NMDA-plateau within-dendrite eligibility, and neuromodulator volume-transmission. And *"Temporal Credit Is Free"* (arXiv:2603.28750) argues recurrent-eligibility failures are **CALIBRATION** (decay ~85× too slow; recurrent-weight gradients ~100× too small for plain SGD → fixed by Adam-style per-parameter normalization), NOT a missing circuit — which directly predicts our own recurrent-e-prop refutation (memory-timescale artifact, true-gradient hurts).

## THE cheapest-first de-risk (the next build)

**Task instrument correction:** the feedforward semantic-inheritance task cannot discriminate MDGL (e-prop already saturates it → the "a simpler mechanism doesn't also pass" anti-cheat is unsatisfiable). Use a **DELAYED-CUE temporal task** — two cues separated in time, combined at readout beyond the fixed-reservoir/ALIF horizon, where diagonal e-prop provably can't and BPTT hits 1.0 (= my own pinned horizon-extension test).

- **STAGE 0 (must run FIRST; ~1hr rate-level numpy; NO new mechanism):** recalibrate the existing recurrent e-prop — **the ONE variable = Adam per-parameter gradient normalization + a swept trace-decay** — on the delayed-cue task. If recalibrated diagonal e-prop reaches BPTT-comparable accuracy → **the open residual dissolves for free; STOP; no new class.**
- **STAGE 1 (only if a gap survives Stage 0):** add MDGL's one-hop cross-neuron term on top of the RECALIBRATED e-prop (recalibration on BOTH arms so it's not a confound). Rate-level first; port to spikes only if it beats the recalibrated baseline.
- **Anti-cheats (all four):** permuted-label→chance; **zero/shuffle the cross-neuron term → collapses byte-for-byte to the recalibrated diagonal-e-prop baseline** (this IS the like-for-like "simpler mechanism doesn't also pass" control); positive-control BPTT=1.0; and the decisive memory-timescale-artifact control (`sign_flip==plastic` must HURT, + a matched-trace-decay diagonal baseline + a tuned/interpolated n-gram must all lose to MDGL).

## Verdict + priority

Spiking cheap-scale deep credit is NOT a scale-only boundary: feedforward is GO; the recurrent off-diagonal frontier is worth ONE cheap-first de-risk with the even-cheaper recalibration gate in front. **Priority conditional (load-bearing):** even a full delayed-cue GO is 3-4 orders below language scale, and it does NOT touch the natural-language perplexity ceiling (a data/scale wall — all models lose to a tuned n-gram at tractable scale, per my `2026-07-15` findings). Per ROADMAP §12 the deep-credit rule is OFF the open-generation-ladder critical path — but it IS the emergence-engine ENABLER per the emergence bar (a substrate that learns recurrent structure from a stream). ⇒ **BUILD Stage 0 now** (cheapest-first; it may dissolve the frontier for free), gated behind the anti-cheats.

Full workflow transcript: the `wf_419ab83d-70a` journal. Sources cited inline (verify "Temporal Credit Is Free" + MDGL PNAS 2021 in depth before Stage 1).
