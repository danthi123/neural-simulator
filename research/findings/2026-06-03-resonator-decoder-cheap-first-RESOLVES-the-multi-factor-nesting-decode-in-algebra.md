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

## ✅ SPIKING PORT ALSO RESOLVES (the decisive test — addresses "substrate fails")

`research/findings/raw/_spiking_resonator_probe.py` ran the SAME iterative resonator factorization using the
GENUINE spiking operations from `resonate_fire_fhrr.py` (`rf_unbind` resonate-and-fire phase-subtract +
`rf_resonate` readout + soft codebook projection), F=3, D=256, frozen bars:

| M | spiking resonator | single-shot control |
|---|---|---|
| 4 | 1.00 | 1.00 |
| 8 | 1.00 | **0.07** |
| 16 | 1.00 | **0.00** |
| 32 | 0.07 | 0.00 |

- Smell-test PASS (M=4 = 1.00). **Spiking operational capacity M=16** (16³ = 4,096-combination search
  decoded 100% IN SPIKES); single-shot control collapses (0.07 at M=8 → 0.00 at M=16). **VERDICT: RESOLVES.**
- **The resonator decode SURVIVES the resonate-and-fire substrate.** The prior "composition trivial in
  algebra, impossible in substrate" caveat does NOT extend to the resonator — the iterative search is robust
  to the genuine spiking unbind. (Spiking capacity M=16 at D=256 vs algebra M≈48 at D=1024 is the expected
  D-scaling — resonator capacity ~quadratic in D; larger D → larger M, the capacity-safe lever.)

**Net: a genuinely-new, biology-faithful mechanism (the resonator network), found via the owner-directed
deep research, validated cheap-first in BOTH the algebra AND the spiking substrate, gets past our
characterized multi-factor/nesting decode wall** — the one that scored 0.000 (hierarchical-320) and forced
the flat-distinct workaround. This is a real path forward, not a wall.

## D-scaling characterization (cheap-first step done)

| D | capacity M (max M with ≥0.90, F=3) | search space M³ |
|---|---|---|
| 256 | 16 | 4,096 |
| 512 | 32 | 32,768 |
| 1024 | 48 | 110,592 |
| 2048 | 64 | 262,144 |

Capacity **scales with D** (linear at low D, gently sub-linear ~D^0.67 higher). **Realistic nesting (per-slot
fan-out ≤ 64) already works at our 320-substrate's D≈2000**; full M=320-per-slot would need D≈22K (large but
GPU-feasible). This is a capacity-*safe* lever (more D → more capacity), not a hard ceiling — the resonator
path scales. So **Direction A cheap-first is fully characterized: algebra RESOLVE + spiking RESOLVE +
D-scaling — a genuinely-new, biology-faithful, scalable mechanism past the nesting-decode wall.**

## ⚠️ Real-codes transfer test: the resonator needs PHASOR codes, NOT the real-Hadamard 320 substrate (key architectural finding)

`research/findings/raw/_resonator_real320_probe.py` ran the resonator on the **actual 320 concept codes** (the
`_flatdist320_codes.npz` cache the conversational agent uses): dense real (float64, D=2000, Hadamard-bound).

| substrate (D=2000, M=16, F=3) | resonator success |
|---|---|
| **real-Hadamard 320 codes, MULTIPLY unbind** | **0.00** |
| **real-Hadamard 320 codes, DIVIDE unbind** (true Hadamard inverse) | **0.00** |
| **PHASOR codes (same D=2000, M=16)** | **1.00** |

**This is fundamental, not a bug** (both unbind variants fail; the phasor control at the identical D/M
RESOLVES). The dense real-Hadamard binding is **not cleanly invertible** (`a⊙b` then `·a` = `a²⊙b ≠ b`;
elementwise division blows up on the dense near-zero elements), so the resonator's iterative unbind↔cleanup
cannot converge. **This is precisely WHY the 320 substrate cannot nest** — its codes/binding fundamentally
forbid clean factorization, which is what forced the flat-distinct single-binding workaround. The resonator
cannot fix that substrate.

**The resonator's nesting-unlock applies to the PHASOR substrate** (`resonate_fire_fhrr` / `spiking_phasor_
fhrr` — the resonate-and-fire phasor FHRR layer, itself validated 2026-05-22 as a "working compositional
layer"), where binding is clean phase-arithmetic (invertible) and the resonator RESOLVES (algebra + spiking +
scalable). So nested conversational composition is achievable — on the phasor substrate, not the
real-Hadamard 320 agent.

## ✅ PAYOFF DEMONSTRATED: nested-fact understanding works on phasor FHRR

`research/findings/raw/_resonator_nested_fact_probe.py` ran the decisive capability test on a GENUINE
SEMANTIC nested fact (phasor FHRR, D=1024, M=16/kind, 40 trials):

```
fact = AGENT⊗noun + ACTION⊗verb + PATIENT⊗( adj ⊗ noun )      ("dog chase (big cat)")
```

The patient slot's filler is itself a bound product (`adj ⊗ noun` = "big cat") — the nesting a flat substrate
cannot decode. Result:

| decoder | recovers the attributed patient |
|---|---|
| **resonator** (factor `adj⊗noun` → adjective + noun) | **1.00** (both adj AND noun) |
| single-shot flat decode (clean up vs the noun vocab) | **0.07** (≈ chance 0.062 — the 0.000-class failure) |

**VERDICT: RESOLVES.** The resonator decodes the nested attributed fact at **100%**, recovering the patient
as BOTH its adjective and noun, **crosstalk-robust** (the bundle of three role-bindings does not break it),
where the flat single-shot decode (the 320-substrate approach) is **at chance**. So **nested-fact
understanding — an SVO fact whose slot is itself a structured entity — genuinely works on the phasor FHRR
substrate**, decoded by the spiking-validated resonator. This is the first concrete capability past the
nesting wall that scored 0.000 and forced the flat-distinct workaround.

## Pre-registered next steps (corrected by the real-codes finding)

1. ~~D-scaling sweep~~ — DONE: capacity scales with D; realistic nesting at D≈2000.
2. ~~Integrate into the real-Hadamard 320 pipeline~~ — **REJECTED by the real-codes test**: that substrate's
   binding is non-invertible; the resonator cannot work there. Nesting on the 320 agent's codes is impossible.
3. **The real payoff build: nested composition on the PHASOR substrate.** Build (or extend the validated
   `resonate_fire_fhrr` compositional layer to) a NESTED structure — a fact-about-a-fact / attributed concept
   / embedded clause — encoded in phasor FHRR, decoded by the spiking resonator. Pre-register a frozen gate:
   does the phasor+resonator pipeline decode a genuine 2-level nested fact (where single-shot gives the
   documented 0.000-class failure)? This is the decisive capability test, on the substrate where the
   resonator actually works. The no-confab moat + abstention carry over (the phasor TPAM already has them).

(The noise sub-claim stays a documented secondary negative.)

## Honest framing of where Direction A stands

The owner-directed deep research found a genuinely-new mechanism (the resonator) and cheap-first validation
established, precisely: it **resolves the multi-factor/nesting decode on the phasor FHRR substrate** (algebra
+ spiking + scalable), and it **does not and cannot apply to the real-Hadamard 320 substrate** (non-invertible
binding — the structural reason that substrate is single-binding-only). This is a real, scoped path past the
nesting wall: nested composition is buildable on the phasor substrate. It is NOT a drop-in upgrade to the
current 320 agent — it implies a substrate choice (phasor FHRR for composition that needs nesting). An honest,
scoped advance, not an over-claim.

## Files / evidence

- Probe + frozen bars: `research/findings/raw/_resonator_capacity_probe.py`
- Recorded output: `research/findings/raw/_resonator_capacity_probe_recorded.txt`
- Substrate to port onto: `research/runners/resonate_fire_fhrr.py` (`rf_unbind`, `ResonateFireTPAM.cleanup`)
- Research origin: `research/findings/2026-06-03-deep-research-how-the-field-gets-past-our-generative-conversation-wall.md`
</content>
