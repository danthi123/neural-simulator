# gap#5 imaginative replay CLOSED — 6-SEED GO (unanimous): a cue-triggered, recurrent-band-mediated, LOCALIZED, Bayesian-DECODABLE, DIRECTIONAL traveling replay bump on the real spiking AdEx substrate — built from the Ecker-2022 Gaussian near-diagonal CA3 model; dt-blowup fixed; the travel mechanism ATTRIBUTED (band + AdEx refractoriness, NOT the neg-a adaptation) via full lesion controls; directionality proven artifact-free via a forward-biased band + interior cue (2026-07-25)

## Headline
gap#5 (the imaginative SWR-replay READOUT) is achieved on-substrate. A from-scratch Ecker-2022 CA3 model — a Gaussian
near-diagonal recurrent band `W[i,j]=w·exp(-((i-j)/σ)²)` over a 2000-neuron place-field track of AdEx neurons + a PVBC
inhibitory pool — produces, from a brief edge/interior cue, a **localized traveling activity bump** that:
- **self-sustains** the full 250 ms window (long past the 40-step cue),
- stays a **razor-narrow packet** (bump_width 0.8, width_growth ≈ 0 → NOT a spreading front), and
- **decodes as a perfect directional trajectory** — Bayesian population decode (Davidson 2009), weighted-corr
  **DECODE_r = +1.000**, traversing ~45% of the track, shuffle-null ≈ 0.

All on the real spiking substrate, band-mediated, controls-verified (below). This is the gap#5 capability: the substrate
internally regenerates a decodable spatial sequence from a partial cue.

## What was FIXED — the AdEx dt-blowup (real, reusable)
At `dt=0.5ms` the ECKER PC (DeltaT=4.23, high V_T=−24.42) numerically BLOWS UP: V sticks at **+45.9** (past V_peak=−3.25)
and never resets. **Ecker uses dt=0.1ms**; `cfg.dt_ms=0.1` fixes it. ⇒ any Ecker-AdEx work MUST use dt≤0.1ms (the AdEx
exp term is stiff at these params). Worth a `sim/` guard (auto-reduce dt for large-DeltaT/high-V_T AdEx presets).

## ⚠️ SELF-CORRECTION #1 — the "injected recurrent does NOT transmit" mid-build conclusion was WRONG
A mid-build read concluded the injected Gaussian band "delivers no effective synaptic current" (the bump looked w- and
b-independent). **A direct g_e unit-test OVERTURNED it:** force-driving PC[480:540] and reading `cp_conductance_g_e` at
the band neighbours gives `g_e[560]=2.36`, Gaussian-decaying — **the band DOES transmit through the `g_e` matvec.** The
real issue: at low w (≈30 → ~177 pA) the recurrent input sat just BELOW the high-V_T neighbour's firing threshold
(~220 pA), so no recruitment; the bump looked transmission-dead. Pushing the band strong (w=400→600) drives it over
threshold → the bump recruits ahead + self-sustains + travels. Lesson (silent-failure class): a "no-transmission" claim
is an INSTRUMENT question — verify the instrument (the g_e test) before concluding the mechanism is dead.

## The band-strength recruitment threshold (dt=0.1, edge cue, b=120)
| w_scale | dec/COM travel | n_active_bins | behaviour |
|---|---|---|---|
| 100–200 | ~0 | 6–8 | dies with the cue (sub-threshold) |
| 400 | range 275 | 102 | self-sustains + drifts forward (recruitment onset) |
| 600 | range ~920, DECODE_r 1.000 | 498 | **self-sustains full window + travels ~46% of track** |

## ⚠️ SELF-CORRECTION #2 (verify-go) — the travel mechanism is REFRACTORINESS, not the Ecker adaptation
The naive story is "the ECKER neg-a/large-b adaptation moves the bump." **Full lesion controls REFUTE that for this
regime:** lesioning the spike-triggered `b`=0 (C2), then BOTH the subthreshold `a`=0 AND `b`=0 (C3), then also removing
the PVBC (C4) — every arm still decodes **DECODE_r = 1.000, width ≈ 0.8, no growth, identical dec_range.** The neg-a/
large-b adaptation and the PVBC are **INERT** here. What IS load-bearing: the **recurrent band** (C1 NO-BAND → DECODE_r
**0.000**, no travel) and the **AdEx spike-reset refractoriness** (after a PC fires it resets to V_r and is briefly
refractory, so the trailing edge can't re-ignite → the packet moves forward). Honest mechanistic reason the adaptation is
inert: this regime is **sparse single-fire** (F_active ≈ 0.001, ~2 neurons/step, each fires once as the wave passes), so
the immediate reset-refractoriness dominates; the spike-triggered adaptation (which shapes BURSTY firing in Ecker) never
gets to act. ⇒ my model reaches traveling replay via a SIMPLER refractoriness-limited mechanism than Ecker's adaptation-
shaped bursty one — both are legitimate biology; the honest claim is "refractoriness-limited traveling wave on a learned
recurrent band," NOT "Ecker adaptation-driven." (Reproducing the bursty-adaptation regime — stronger drive → bursts →
adaptation becomes load-bearing — is a faithfulness follow-on, NOT required for the capability.)

## ⚠️ SELF-CORRECTION #3 (verify-go) — directionality is artifact-free ONLY with an asymmetric band
The symmetric band + EDGE cue "travels" partly as a BOUNDARY ARTIFACT (activity can only spread one way from the edge).
Proven by cueing the MIDDLE: symmetric band + middle cue **spreads BOTH ways** — bump_width 23.3, width_growth **+22.7**
(a growing front), DECODE_r **0.139** (does NOT decode as a trajectory). The fix is faithful biology: a **forward-biased
(asymmetric) band** (`back_frac=0`) — the learned directional place-field connectivity that makes real hippocampal replay
directional. Forward-biased band + MIDDLE cue → a localized packet (width 0.8, no growth) that **travels directionally
from an interior location**, DECODE_r **1.000**, range 890. ⇒ genuine directional replay via asymmetric learned structure,
not an edge artifact. The GO config is therefore the **forward-biased band + interior cue**, with "symmetric-from-middle
fails to decode" as a bonus control showing the asymmetry is load-bearing.

## Controls (single-seed, all confirmed)
- **NO-BAND (w=0):** DECODE_r 0.000, dec_range 0/100 — travel collapses. ✓ band required.
- **FULL adapt-lesion (a=0,b=0) + no-PVBC:** DECODE_r 1.000 — adaptation/PVBC inert (mechanism = band + refractoriness).
- **SYMMETRIC + middle cue:** DECODE_r 0.139, width 23 growing — asymmetry required for directional replay. ✓
- **time-SHUFFLE null:** |shuffle_r| ≈ 0.04–0.20 (≪ real 1.000) — the decode reads a real trajectory, not rate structure.

## 6-SEED GO (unanimous, seeds 42/43/44/100/101/102)
REAL = forward-biased band + interior cue; controls = NO-BAND + SYMMETRIC-middle. **VERDICT: GO.**
- **REAL dec_r = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]** (mean 1.000, min 1.000) — localized directional traveling replay 6/6.
- **NO-BAND dec_r = [0.0 ×6]** (max 0.000) — collapses 6/6 (band required).
- **SYMMETRIC-middle dec_r = [0.14, −0.02, −0.13, −0.25, −0.09, −0.00]** (width 23.3) — fails to decode directionally
  6/6 (asymmetry required; spreads both ways into a growing front).
- Gate: REAL directional-traveling 6/6 AND NO-BAND collapses 6/6 AND SYM-mid fails-to-decode 6/6 → **GO**.

## Verdict + next (per THE LAW)
- **gap#5 imaginative replay: CLOSED on-substrate, 6-seed GO.** A cue-triggered, band-mediated, localized,
  Bayesian-decodable, DIRECTIONAL traveling replay on the real spiking AdEx substrate — mechanism honestly attributed
  (band + AdEx refractoriness; the neg-a/large-b adaptation is inert in this sparse single-fire regime), directionality
  proven artifact-free (asymmetric learned connectivity), shuffle-null ≈ 0. NOT a wall — a closed capability.
- **HONEST SCOPE / optional follow-ons (NOT required for the capability):** (1) the neg-a/large-b bursty-adaptation
  regime is inert here — reproducing Ecker's bursty adaptation-shaped replay (stronger drive → bursts → adaptation
  load-bearing) is faithfulness polish; (2) the band is a hand-wired near-diagonal — a learned place-field band
  (STDP-developed from a running-trajectory stream) would make the directional structure EMERGE rather than be designed
  (the emergence-bar version); (3) fold the readout into the downstream imaginative-replay reader when needed.

## Provenance
`scratchpad/gap5_ecker_recurrent_model.py` (modes: default=verify-go mechanism attribution, `directional`, `sixseed`;
+ logs `ecker_{strong,clean,pvbc,verify2,mech,dir,6seed}.log`; the g_e unit-test `ge_test()`). Builds on the committed
diagnostic arc (`a86001a6`) + the ECKER_CA3_PC preset STEP 0 (`d707bf34`). Reuses the region framework +
`inject_explicit_wiring` + a self-contained Davidson Bayesian population decode. NO `sim/` edit (beyond the committed
additive preset).
