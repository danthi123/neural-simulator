---
type: finding
status: contributing
date: 2026-08-10
mechanism: ca3-completion
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: LEVER A on the gap#5 composition seam — raise the DG detonator-gain / readout-threshold so the EMERGENTLY-SELECTED CA3 assembly grows from the ~23-cell regime toward the ~72-cell regime where the 2026-08-10 formation GO's bistable window lives, WITHOUT hand-setting membership, then run the committed end-to-end completion and ask whether a LARGER emergent assembly completes cue-specifically. The lever is a pure SELECTION front-end change (NO change to the completion instrument): the emergent assembly = the natural >=theta CA3 set from the R1 recovered-at-scale sparse mossy-detonator (n_ca3=2000, d0.02/w3000, acw12, drv2000, theta lowered 0.15->0.10). theta=0.10 is the size-matched, separation-PRESERVING growth point chosen from a 5-knob separation-cost sweep (raising mossy_weight grows ONE input to 101 while others stay ~14/23 = non-uniform; mossy_density=0.05 grows all to ~116 but roughly doubles the pairwise Jaccard = separation cost; theta=0.10 grows to ~72 uniform at a pairwise Jaccard essentially equal to the baseline — the numeric table is in the body). The grown membership is fed via the ADDITIVE `assemblies_ext` seam into the committed slow-NMDA reverberatory BTSP-formation+completion instrument (`_gap5_btsp_forms_nmda_slow_reverberatory_derisk.run_seed`), swept over completion recurrent density {0.06,0.08} x the E%-max FEEDFORWARD divisive-norm basket (`ca3_ff_inhib=400`, the size-aware fix) at ca3_fb_inhib=60, wmax=5000. A SCALE-vs-STRUCTURE control runs the SAME completion on 3 UNIFORM-RANDOM disjoint 72-cell assemblies at n_ca3=2000 (seed 42, densities 0.06/0.08/0.12) to disentangle "the emergent set's structure is uncompletable" from "no recurrent-attractor bistable window exists at this scale for ANY 72-cell assembly". Anti-cheats: EMERGENT membership 6/6 (mossy-LESION -> assemblies collapse to size 0 = DG-derived; Jaccard vs the random-permutation pre-assigned set <=0.06; sizes are the grown emergent ~63-99, mossy-lesion `attributable_to` on intact-vs-lesion size); the CONTROL WITHOUT the lever = the ~23-cell baseline, which STILL fails (established 6/6 in the 2026-08-10 size-aware PARTIAL, the lever is load-bearing on SIZE = it verifiably grew the assembly); plasticity FROZEN at recall; perm via the FF-basket; silent-rest nocue; OU OFF (deterministic seam); genuine_formation asserted per BTSP row (within grows from `fused_btsp_update`, cross_dw ~ 0). Runner: `research/runners/_gap5_leverA_detonator_gain_larger_assembly_derisk.py` (-m provenance sidecar records backend+argv+SHA). SIM_BACKEND=cupy. NO `sim/` edit. GO gate (6-seed): a grown-emergent + BTSP-formed assembly gives held_cue>=0.20 AND held_cue>=3*held_perm AND held_cue>=3*held_nocue AND held_nocue<=0.10 (genuine) on >=5/6 seeds.
---

# Gap #5 completion seam, LEVER A — the DG detonator-gain lever GROWS the emergent assembly to the ~72-cell formation-reference regime WITH pattern-separation PRESERVED (mean 76 cells, pairwise Jaccard essentially unchanged from baseline), but the completion does NOT reach the cue-specific bistable GO (0/6): cue-completion and rest-self-ignition stay COUPLED through the shared within-assembly recurrent gain at EVERY density, and a uniform-RANDOM 72-cell control at n_ca3=2000 self-ignites IDENTICALLY — so the residual is NOT assembly SIZE (the size-aware finding's diagnosis) but the recurrent-attractor SELF-DRIVE coupling at the composition SCALE, and the FF-basket is the wrong lever for it

The 2026-08-10 size-aware-FF PARTIAL
(`2026-08-10-gap5-size-aware-FF-completion-closes-permuted-cue-specificity-but-small-emergent-assembly-bistability-residual-PARTIAL.md`)
closed the afferent-specificity half of the gap#5 composition seam (perm 0.13 -> 0.00 via the E%-max feedforward
divisive-norm basket, load-bearing) but pinned the remaining bistability residual to assembly **SIZE**: the ~23-cell
emergently-selected assembly had no recurrent-attractor operating point that is simultaneously cue-ignitable
(cue >= 0.20) and rest-silent (nocue <= 0.10), while the ~72-cell UNIFORM pre-assigned assemblies of the
2026-08-10 formation GO
(`2026-08-10-gap5-BTSP-emergently-forms-the-slow-nmda-reverberatory-attractor-6seed-GO-preassigned-assemblies.md`)
did. It named two next levers; **this finding builds lever (a): raise the DG detonator-gain so the emergent assembly
grows into the formation's viable-size regime.** The outcome is a decisive, precisely-diagnosed NEGATIVE that
**re-diagnoses the residual**: growing the emergent assembly to ~72 cells (with separation preserved) does NOT open the
bistable window, and — critically — neither does a uniform-RANDOM 72-cell assembly at the same scale. The residual is
the recurrent SELF-drive coupling at n_ca3=2000, not size.

## The lever WORKS on its own terms — the emergent assembly grows to ~72, uniform, separation PRESERVED
<!--derived-->
A 5-knob separation-cost sweep (seed 42, `research/findings/raw/_gap5_e2e/leverA_explore_s42.json`) shows the honest
size-vs-separation tension, and that a size-matched separation-preserving growth point exists:

| knob | value | mean size | pairwise Jaccard (separation cost) |
|---|---|---|---|
| baseline (R1) | theta 0.15 | 24 | 0.054 |
| readout threshold | theta 0.10 | 70 | **0.063** |
| readout threshold | theta 0.08 | 104 | 0.070 |
| recurrent amp | amp_ca3w 16 | 52 | 0.067 |
| mossy density | 0.03 | 37 | 0.056 |
| mossy density | 0.05 | 116 | 0.108 |
| mossy weight | 12000 | 46 | 0.041 (but sizes [101,14,23] = NON-uniform) |

- **theta=0.10 is the clean growth point:** it grows the emergent assembly to the ~72-cell regime UNIFORMLY (all three
  co-stored assemblies grow together) with pairwise Jaccard 0.063 — essentially the baseline 0.054, so the DG's
  pattern-separation is PRESERVED at the target size. (Raising `mossy_weight` grows ONE input's assembly while the
  others stay tiny = non-uniform, the "which DG codes concentrate" fragility of the 2026-07-18 BOUNDARY; raising
  `mossy_density` past ~0.03 grows all three but degrades separation. theta lowering is the size-matched knob.)
- **6-seed: the lever is verifiably ENGAGED and emergent, 6/6.** Grown emergent assembly mean size **76** cells (per
  seed 68/72/71/84/79/82) at pairwise Jaccard mean **0.062** — the ~72-cell formation-reference regime with separation
  preserved. Emergent-membership anti-cheat holds 6/6: the mossy-LESION collapses every assembly to size **0** (the
  DG->CA3 detonation is load-bearing = membership is DG-derived, `attributable_to` intact-vs-lesion is the full size),
  and Jaccard vs the readout's random-permutation pre-assigned set is 0.006-0.056 (not a hand-set mask).

## But the completion does NOT reach the cue-specific bistable GO — 0/6, cue and nocue rise TOGETHER
<!--derived-->
Feeding the grown ~72-cell emergent membership into the committed completion (density {0.06,0.08} x FF-basket 400,
fb=60), 6-seed (`research/findings/raw/_gap5_e2e/leverA_detonator_gain_6seed.json`): **grown-emergent completion GO
0/6**, status NEGATIVE. Per-seed BTSP arm (mean held over the 3 co-stored assemblies):

| seed | sizes | d0.06 cue/perm/nocue | d0.08 cue/perm/nocue |
|---|---|---|---|
| 42 | [77,60,67] | 0.215 / 0.026 / 0.214 | 0.303 / 0.116 / 0.319 |
| 43 | [69,79,67] | 0.228 / 0.187 / 0.237 | 0.274 / 0.084 / 0.287 |
| 44 | [99,62,51] | 0.236 / 0.020 / 0.300 | 0.304 / 0.110 / 0.358 |
| 100 | [78,85,88] | 0.316 / 0.215 / 0.327 | 0.366 / 0.257 / 0.371 |
| 101 | [63,91,84] | 0.194 / 0.099 / 0.180 | 0.301 / 0.091 / 0.286 |
| 102 | [79,88,80] | 0.286 / 0.257 / 0.261 | 0.297 / 0.262 / 0.337 |

- **cue and nocue are LOCKED together on every seed and every density.** Wherever the cue reaches >= 0.20 the rest
  state is already self-igniting at nocue ~ 0.18-0.37: the mean **min nocue at cue >= 0.20 is 0.271** (>> the 0.10
  bar). Raising density from 0.06 -> 0.08 lifts the cue AND the nocue by the same amount — they share the within-assembly
  recurrent gain, exactly as at the ~23-cell size. **Growing the assembly to ~72 did not decouple them; if anything the
  larger assembly self-ignites HARDER** (seed-42 single-input density curve, grown ~72:
  `research/findings/raw/_gap5_e2e/` scratch: d0.03 cue/nocue 0.068/0.069 -> d0.06 0.222/0.235 -> d0.12 0.388/0.418 —
  the whole cue/nocue pair slides up together with density, never separating).
- **genuine_formation = True on every BTSP row** (within grows from the plateau-gated one-shot, cross_dw ~ 0), so this
  is a READOUT/bistability seam, not dead formation. The FF-basket suppresses the permuted cue (perm 0.02-0.12 on the
  clean-structure seeds) but cannot touch the SELF-drive coupling — perm is afferent, nocue is intrinsic.

## The DECISIVE diagnostic — a uniform-RANDOM 72-cell assembly at n_ca3=2000 self-ignites IDENTICALLY (the residual is SCALE, not SIZE, not structure)
<!--derived-->
The scale-vs-structure control (seed 42, `struct_control` block of the 6-seed JSON) runs the SAME completion on 3
UNIFORM-RANDOM disjoint 72-cell assemblies at n_ca3=2000 — the exact clean structure + exact size that gave the 6/6
formation GO **at n_ca3=400**:

| assembly | d0.06 cue/perm/nocue | d0.08 cue/perm/nocue | d0.12 cue/perm/nocue | GO |
|---|---|---|---|---|
| uniform-random-72 @ n_ca3=2000 | 0.195 / 0.000 / 0.189 | 0.289 / 0.000 / 0.253 | 0.369 / 0.000 / 0.326 | 0/3 |

- **The random-72 control fails IDENTICALLY to the emergent-72:** cue and nocue rise together at every density
  (0.195/0.189 -> 0.369/0.326), no bistable window, GO=False. The FF-basket kills perm to a clean 0.000 (a perfectly
  uniform random set is the ideal case for afferent divisive-norm), yet the SELF-drive coupling is untouched.
- **Therefore the failure is NOT the emergent assembly's structure and NOT its size.** A clean random 72-cell assembly —
  the formation GO's own winning configuration — self-ignites at n_ca3=2000. The formation GO's wide bistable window was
  a property of the n_ca3=400 SCALE (where 72 cells = 18% of CA3 and the fb_inhib=60 / 500-basket feedback inhibition
  balanced the self-drive). At the composition scale (n_ca3=2000, forced by emergent-DG selection which is 6/6 only at
  2000), the FIXED completion operating point (fb_inhib=60) has no cue-vs-rest window for ANY 72-cell assembly.
- **This re-diagnoses the size-aware finding's residual.** "The DG produces assemblies an order of magnitude too small"
  was the visible correlate at 23 cells, but growing to the reference 72 (this finding) still fails, and the reference
  72 as a clean random set also fails — so the load-bearing cause is the recurrent-attractor SELF-drive coupling at the
  n_ca3=2000 scale, of which small-assembly-weakness was one face. Lever A is a genuine, verified SIZE lever; the seam
  is simply not a size seam.

## Status + the next mechanism (per THE LAW — a wall is a verdict on a METHOD)
<!--derived-->
**NEGATIVE (honest, load-bearing, decisive) — the DG detonator-gain lever grows the emergent assembly to the ~72-cell
formation-reference regime with pattern-separation PRESERVED (mean 76 cells, pairwise Jaccard 0.062; emergent-membership
6/6; the lever is verifiably engaged), but the grown-emergent completion does NOT reach the cue-specific bistable GO
(0/6): cue-completion and rest-self-ignition stay coupled through the shared within-assembly recurrent gain at every
density, and a uniform-RANDOM 72-cell control at n_ca3=2000 self-ignites identically — so the residual is the
recurrent-attractor SELF-DRIVE coupling at the composition SCALE, NOT assembly size.** This is NOT gap#5 closure. It
CONVERTS the size-aware finding's "assembly too small" residual into a sharper, scale-anchored one and eliminates SIZE
as the lever.

The quantified residual: at n_ca3=2000 with the fixed fb_inhib=60 completion, min nocue at cue >= 0.20 is 0.271 (needs
<= 0.10); the coupling is intrinsic to a recurrent-POPULATION attractor and is NOT separable by any afferent lever (the
FF-basket removes perm but not nocue). Two named next mechanisms, both one-brain, ordered by leverage:

1. **INTRINSIC per-cell dendritic READOUT bistability for the completion read** (the size-aware finding's lever (b), now
   the PRIMARY): a plateau/dAP-latched high state held PER held-out cell (size- AND scale-independent) instead of a
   recurrent-population attractor, so cue-ignition and rest-silence decouple by construction and no longer depend on the
   within-assembly recurrent gain or the network scale. This is a READOUT bistability that HOLDS a completion state —
   explicitly NOT the two-compartment/BDSP/burstprop deep-CREDIT-assignment rule, which is tested-NEGATIVE for hidden
   credit on spikes (`research/findings/2026-05-17-dendritic-credit-assignment-NEGATIVE.md`,
   `research/findings/2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS-deep-research.md`). (The
   2026-07-18 dendritic "learned CLOSED" completion characterization is ⛔ RETRACTED [self-sustaining + Wang confound];
   this names the readout-bistability MECHANISM, not that result. Run `bash tools/before_you_build.sh` before building.)
2. **SCALE-AWARE feedback inhibition** — since the random-72 control shows the window closes purely from n_ca3 400 ->
   2000 at fixed fb_inhib=60, a feedback-inhibition gain that scales with n_ca3 (or with the active population, the
   companion homeostatic process the fixed constant replaced) may re-open the population-attractor window without a new
   readout mechanism. This is the cheaper test and directly probes the "what did we replace with a constant?" reframe
   (the fb basket is the proxy); it is a bounded lever (it does not decouple cue from nocue, only re-centers them), so
   (1) remains the primary path to a robust decoupling.

Artifacts (SIM_BACKEND=cupy; provenance sidecar records backend + argv + git SHA):
`research/findings/raw/_gap5_e2e/leverA_detonator_gain_6seed.json` (6-seed grown-emergent completion + the
scale-vs-structure random-72 control), `research/findings/raw/_gap5_e2e/leverA_explore_s42.json` (the 5-knob
size-vs-separation-cost sweep). Runner: `research/runners/_gap5_leverA_detonator_gain_larger_assembly_derisk.py`. The
`ca3_ff_inhib`/`assemblies_ext` seams are ADDITIVE (byte-identical when None):
`research/runners/_gap5_btsp_forms_nmda_slow_reverberatory_derisk.py`. NO `sim/` edit.

### Sources
- de Almeida L., Idiart M., Lisman J.E. *A second function of gamma frequency oscillations: an E%-max winner-take-all mechanism selects which cells fire.* J. Neurosci. 29:7497-7503 (2009).
- Pouille F., Scanziani M. *Enforcement of temporal fidelity in pyramidal cells by somatic feed-forward inhibition.* Science 293:1159-1163 (2001).
- Guzman S.J., Schlogl A., Frotscher M., Jonas P. *Synaptic mechanisms of pattern completion in the hippocampal CA3 network.* Science 353:1117-1123 (2016).
- Wang X-J. *Probabilistic decision making by slow reverberation in cortical circuits.* Neuron 36:955-968 (2002).
- Bittner K.C., Milstein A.D., Grienberger C., Romani S., Magee J.C. *Behavioral time scale synaptic plasticity underlies CA1 place fields.* Science 357:1033-1036 (2017).
