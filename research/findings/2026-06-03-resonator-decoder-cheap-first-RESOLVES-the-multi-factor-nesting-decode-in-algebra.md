# Direction A cheap-first probe RESOLVES (primary): a resonator-network decoder factors multi-factor FHRR products our single-shot decode cannot — the missing decode stage for the NESTING wall

**Date:** 2026-06-03
**Status:** ✅ RESOLVES (primary claim) in the FHRR ALGEBRA (numpy), per the pre-registered three-state gate +
smell-test. Honest secondary negative on the noise-injection sub-claim. Decisive next test = the SPIKING port.
**Probe:** `research/findings/raw/_resonator_capacity_probe.py` (frozen bars; recorded
`_resonator_capacity_probe_recorded.txt`).
**Origin:** the 2026-06-03 deep-research synthesis (owner-directed) identified the resonator network
(Frady-Kent-Olshausen-Sommer 2020) + noise injection (Kymn 2024) as the genuinely-untried, biology-faithful
mechanism for our characterized wall. Check-existing-first confirmed **no resonator exists in the codebase**
(grep empty); our decode is single-shot (`batched_phase_similarity`). Cheap-first numpy probe before any
spiking build (standing discipline).

## What it tested + why it maps to OUR wall

Our composition hit a **NESTING wall**: the hierarchical-320 (a 2nd binding level) scored **0.000** (the
"multi-hop SNR wall"), which forced the flat-distinct single-binding-level workaround. Decoding a nested
bound structure = **factoring a product** `C = x1 ⊗ x2 ⊗ … ⊗ xF` of F unknown factors, each from a codebook
of size M (search space M^F). A **single-shot decode cannot factor a product** (it would need F−1 factors
already). A **resonator network** searches M^F *in superposition* by iterating unbind ↔ codebook-cleanup.
This probe asks: on OUR FHRR code type (unit-magnitude complex phasors — the same algebra
`resonate_fire_fhrr.py` realizes), does the resonator factor multi-factor products the single-shot decode
cannot? (Algebra first; spiking next.)

## Result (F=3, D=1024, 60 trials/M; frozen)

| M (codebook size) | resonator | resonator+noise(0.30) | single-shot control |
|---|---|---|---|
| 4 | 1.00 | 1.00 | 1.00 |
| 8 | 1.00 | 1.00 | 0.95 |
| 16 | 1.00 | 1.00 | **0.07** |
| 32 | 1.00 | 1.00 | **0.00** |
| 48 (follow-up) | 1.00 | ~1.00 | — |
| 64 | 0.35 | 0.15 | 0.00 |
| 128 | 0.00 | 0.00 | 0.00 |

- **Smell-test PASS:** resonator @ M=4 = 1.00 (≥0.95) — implementation valid.
- **Resonator operational capacity ≈ M=48–56** per factor (the 4→128 sweep showed ≥0.90 through M=32; a
  follow-up confirmed M=48 ≈ 1.00; the edge is 48–64). At M=32 the search space is **32³ = 32,768
  combinations** decoded at 100%.
- **The single-shot control reproduces the failure decisively:** 0.95 at M=8 → **0.07 at M=16 → 0.00 at
  M=32**. Single-shot decode cannot factor multi-factor products beyond M≈8; the resonator handles 48–56.
  The sharp crossover at M=16 (single-shot 0.07 vs resonator 1.00) is the load-bearing evidence.

**VERDICT (primary): RESOLVES.** The resonator network is the missing decode stage for multi-factor /
nested composition, working on our FHRR codes in the algebra, where single-shot decode completely fails.

## Honest secondary negative: noise injection did NOT replicate the ≥50× claim on our codes

At the frozen 0.30 noise, success was **neutral within capacity** (M≤48 unchanged) and **harmful past it**
(M=64: 0.15 vs 0.35). A follow-up sweep at the capacity edge (M=48; noise ∈ {0, .03, .06, .1, .15, .2, .3})
showed **no meaningful benefit at any level** (all 0.98–1.00). The Kymn-2024 ≥50× operational-capacity
extension is regime-specific (it rescues resonators that fall into spurious limit-cycles); **our
well-conditioned random FHRR phasor codes apparently do not fall into those cycles**, so noise has nothing
to rescue. This is an honest secondary boundary; it does NOT affect the primary result (the resonator
itself is the unlock). Noise may yet matter in the spiking substrate (where stochasticity is intrinsic) —
that is a question for the port, not a re-crank here.

## What this means + the honest caveat

The resonator unlocks **multi-factor / nested decode in the FHRR ALGEBRA** — the prerequisite the
flat-distinct workaround was built to avoid. But the prior 2026-05-22 finding was "composition trivial in
ALGEBRA, impossible in SUBSTRATE." This probe shows the resonator works in the algebra; **whether it
survives the SPIKING resonate-and-fire realization (iterated `rf_unbind` + `ResonateFireTPAM.cleanup`, under
spiking phase noise) is the decisive next test** and directly addresses that caveat. Capacity is D-bounded
(theory: ~quadratic in D); M≈48 at D=1024, so larger vocab/nesting needs larger D — the capacity-safe lever.

## Pre-registered next step (cheap-first, then build)

**Port the resonator to the spiking resonate-and-fire substrate** (`resonate_fire_fhrr.py`): the iterative
loop becomes `rf_unbind` of the other estimates + `ResonateFireTPAM.cleanup` (the spiking codebook
auto-associator we already have), repeated to convergence. Pre-register a three-state gate measuring the
same multi-factor recovery in spikes vs the single-shot spiking decode control. RESOLVES → the resonator is
real on our biology-faithful substrate and unlocks nested composition (re-enabling the hierarchical
structures the flat-distinct workaround avoided); BOUNDARY → the "substrate fails" caveat extends to the
resonator (an honest negative tightening the wall). Reuse-by-import; no protected-module change.

## Files / evidence

- Probe + frozen bars: `research/findings/raw/_resonator_capacity_probe.py`
- Recorded output: `research/findings/raw/_resonator_capacity_probe_recorded.txt`
- Substrate to port onto: `research/runners/resonate_fire_fhrr.py` (`rf_unbind`, `ResonateFireTPAM.cleanup`)
- Research origin: `research/findings/2026-06-03-deep-research-how-the-field-gets-past-our-generative-conversation-wall.md`
</content>
