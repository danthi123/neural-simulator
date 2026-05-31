# Denoiser arc (biologize shortcut-2, the oracle lookup): HONEST NEGATIVE. Temporal integration denoises the VARIANCE (CV ~ 1.63/sqrt(k)) but the activity-grounded symbol has a FUNDAMENTAL SEPARABILITY limit that neither temporal integration NOR the attractor cleanup can fix -- the attractor actually COLLAPSES on the poorly-separated activity symbols (near-chance). The oracle lookup's irreducible value is the ORTHOGONALITY the substrate's raw activity lacks. THREE independent arcs (integrated-loop, D-arc capacity, this denoiser arc) now converge on the SAME prescription: the substrate needs a DG-style PATTERN-SEPARATION stage.

**Date:** 2026-05-31
**Status:** Honest NEGATIVE for biologizing shortcut-2 (the oracle symbol lookup) via temporal integration +/- the attractor cleanup, under the frozen 0.80 bar at loads {2,3,5}, multi-seed. The scientific deliverable (honest negative under strict biology). Points to the next mechanism (DG pattern separation) and converges with two prior arcs.

## The arc + the honest correction chain

The phase-coded FHRR composition is validated; the last engineered shortcut is the ORACLE LOOKUP (a fixed clean symbol per concept). The May-22 activity-level integration (derive the symbol from raw per-neuron activity) was NEGATIVE: single-observation activity is too noisy (CV ~1.63). It re-specified the path: a denoiser. This arc tested the two named denoisers.

1. TEMPORAL INTEGRATION (mean of k observations). Cheap-first (16 obs) looked VIABLE but was OPTIMISTIC (vocab/storage observation overlap inflated it). The rigorous 64-observation distinct measure (no substrate confound -- RECOG_CACHE=phase1_800ev constant for both captures) showed: CV falls EXACTLY as 1.63/sqrt(k) (variance reduction real) but composition PLATEAUS below the bar for higher loads -- L=2 0.834 only at k=32, L=3 0.694, L=5 0.575 at k=32 (CV 0.294). The residual is symbol QUALITY/separability, NOT variance.

2. ATTRACTOR CLEANUP (ResonateFireTPAM, the coupled shortcut-3 denoiser; the May-22 insight that a biological attractor grounds AND denoises). Swapping ONLY the cleanup (simple argmax -> annealed attractor settle, validated params theta 0.1->0.5 over 12 iters) at the best denoiser point (k=32, 64-obs activity):

| load | simple argmax | attractor cleanup | delta |
|---|---|---|---|
| L=2 | 0.833 | 0.228 | -0.606 |
| L=3 | 0.687 | 0.244 | -0.443 |
| L=5 | 0.564 | 0.261 | -0.303 |

The attractor cleanup is CATASTROPHICALLY WORSE -- near-chance (~0.23-0.26 for 8 fillers). It does not fix the residual; it is DESTROYED by it.

## Why -- and the scrutiny (this is not a usage bug)

The attractor near-chance was scrutinized (a validated TPAM at near-chance is surprising). A sanity check fed CLEAN orthogonal vocabulary + noisy versions through the IDENTICAL attractor usage: it recovered the correct pattern 40/40 = 100% at noise 0.00, 0.05, 0.10, AND 0.20. So the usage is correct; the attractor works perfectly on separable patterns. The near-chance on activity-derived symbols is REAL.

The mechanism: a Hopfield-type attractor (W = S S*, S = stored patterns) requires SEPARABLE stored patterns. The activity-derived symbols (random projection of the substrate's per-neuron population activity) are intrinsically OVERLAPPING -- so the recurrent drive never cleanly exceeds threshold toward one basin, the settle COLLAPSES to silence, and the readout is near-chance. The attractor amplifies the overlap rather than cleaning it. Simple argmax (nearest vocabulary by cosine) is actually the BEST available cleanup for these poorly-separated symbols, and even it gets only L=2 to bar.

## Honest verdict

Biologizing shortcut-2 by grounding the composition symbol in the substrate's own activity FAILS at the {2,3,5} bar, for a now-precisely-characterized reason: temporal integration removes the trial VARIANCE (CV ~ 1/sqrt(k), confirmed) but the activity-grounded symbol has a FUNDAMENTAL SEPARABILITY limit, and neither temporal integration nor the attractor cleanup can manufacture separability that the substrate's activity does not contain. The oracle lookup's irreducible value is precisely the ORTHOGONALITY of its codes -- the substrate's raw per-neuron activity is not orthogonal enough for higher-load compositional binding. This is an honest, biology-translatable boundary: a brain cannot use raw population activity as a compositional symbol; it must first ORTHOGONALIZE (pattern-separate) it.

## The convergence (the strategic finding)

THREE independent arcs now converge on the SAME prescription:
- Integrated-loop arc (2026-05-30): the wm binding-retrieval capability needs a stable-AND-lesionable selectivity carrier; the recommended mechanism was DG pattern-separation.
- D-arc capacity geometry (2026-05): dedicated-pool concept geometry erodes under training/common-mode noise; the resolution prescribed pattern-separation (DG orthogonalizes so more concepts fit without interference).
- This denoiser arc (2026-05-31): the activity-grounded composition symbol is separability-limited; it needs orthogonalization before composition.

All three point to a DG-style PATTERN-SEPARATION stage as the missing substrate mechanism. The project HAS a validated DG (the hippocampal trisynaptic loop; P1 validated D.12 pattern separation: DG cosine 0.218 from input cosine 0.800, a 58pp orthogonalization). So the genuine next arc is: insert DG pattern-separation between the substrate's raw activity and the composition-symbol derivation, then re-test whether the DG-separated activity grounds a composable symbol. This is a deeper arc (a decision point), but it is the convergent, biology-grounded, three-arcs-agree prescription.

## Discipline

No protected/frozen/moat/sim/runner module modified (throwaway probes only; spiking_phasor_fhrr / resonate_fire_fhrr / activity_level_integration reused by import byte-unchanged). No bars moved. No autograd. The optimistic cheap-first was honestly corrected by the rigorous confirmation (the grounding discipline caught it before it was a claimed result), and the surprising attractor near-chance was scrutinized (sanity check) before being concluded. Honest NEGATIVE = the deliverable. The cheap-first-before-build discipline held: no big build was committed on the optimistic premise.

## Premise confirmed quantitatively (free, cached): concept activity overlaps at the EXACT P1 DG-input regime

Measured the inter-concept cosine of the captured concept-pool mean activity (the deriver's input,
64-obs cache, 3 seeds):
- BETWEEN different concepts (fillers): cosine ~0.818/0.829/0.815 (max ~0.85)
- BETWEEN different concepts (cues): ~0.80
- WITHIN a concept (split-half, trial-to-trial): ~0.90

So different concepts are only ~0.08 cosine apart from being the SAME concept -- the activity-grounded
symbols are barely separable. This is WHY composition crosstalks at L>=3. And ~0.82 is almost exactly
the P1 D.12 DG pattern-separation INPUT cosine (0.800), where the validated DG produced output cosine
0.218 (58pp orthogonalization). So the convergent prescription makes a FALSIFIABLE QUANTITATIVE
PREDICTION: routing the concept activity through DG should separate it from ~0.82 toward ~0.22, making
the symbols separable enough to compose. The DG arc's cheap-first gate is exactly this measurement.
