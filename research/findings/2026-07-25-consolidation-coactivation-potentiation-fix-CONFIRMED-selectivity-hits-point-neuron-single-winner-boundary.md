# Consolidation de-risk (1-seed indicator): the CO-ACTIVATION replay fix CLEARS the A1 frozen-wire failure (directional potentiation confirmed) + the dedicated attractor region ignites/holds, BUT selective one-of-N binding hits the point-neuron single-dominant-winner boundary (robust to potentiation strength) — the research-predicted bounded negative that names the SFA-eviction / dendritic surpass (2026-07-25)

## What this establishes (first concrete de-risk of the consolidation frontier)
Building the research-gate's recommended Option-1 de-risk (`acd06561`: a DEDICATED strong Wang-2002 attractor region +
CO-ACTIVATION replay), on the extended `nmda_compositional_consolidation.py` harness (dedicated `comp_attr_<s>` slots +
`coactivation_replay`, both additive/default-off, `65881837`/`f19b78d0`). 1-seed indicator (seed 42, offline — Phase-1
train SKIPPED as it stalls >300s on the 8860-neuron substrate; co-activation drives the concept pools by index so Phase-1
isn't needed for the mechanism check):

1. **The co-activation potentiation fix WORKS (directional) — the A1 frozen-wire failure is cleared.** Zero-init the
   plastic `ca1→slot` wire (like A1's `ca1_concept_weight=0`), run replay: with **coactivate=ON** (drive CA3 tag +
   reinstate the fact's noun/adj pools + its target slot) the wire potentiates off zero (Δ **+0.0019** @30cyc, **+0.0057**
   @100cyc); with **coactivate=OFF** (CA3-only, the A1 mode) it stays **frozen at exactly 0.0500** (Δ +0.0000). This is the
   load-bearing potentiation half of the surpass: the A1 non-potentiation ("wire frozen because CA3-only drive never fires
   the pools → no post-spike for STDP") is fixed by co-activating the target. (A silent bug found + fixed en route: the
   pool regions are UPPERCASE `noun_pool_APPLE` but facts are lowercase; a swallowed KeyError made co-activation a silent
   no-op → ON==OFF, until the `.upper()` fix — the classic broad-except silent-failure pattern.)
2. **The dedicated attractor region IGNITES + HOLDS.** After replay, cueing each fact's tag ignites the `comp_attr` slots
   3/3 (the nmda_slow self-loop holds them).
3. **BUT selective one-of-N binding FAILS — the single-dominant-winner boundary.** A dominant slot captures multiple facts
   (@30cyc: facts 0,1 → slot 2; @100cyc: facts 0,2 → slot 0). SELECTIVE (fact i → its own slot i) = **1/3 with
   co-activation vs 1/3 without = chance, at BOTH 30 and 100 cycles.** More potentiation does NOT buy selectivity — the WTA
   on point neurons collapses to one dominant winner regardless of the ca1→slot routing strength.

## DECISIVE diagnostic — the failure is WTA-dominance, NOT engram overlap (⇒ SFA is the confirmed lever)
Ruled out the upstream alternative (that the per-fact ca1 engrams overlap, so no routing could be selective): stimulating
each fact's tag and reading the ca1 firing pattern gives **DISTINCT** engrams — pairwise Jaccard overlap **0.000 / 0.111 /
0.083** (8/14/12 active ca1 neurons per tag, near-disjoint). So the ca1→slot routing HAS distinct inputs to route on; the
selectivity failure is purely the attractor/WTA collapsing to a dominant winner. ⇒ **SFA-eviction is confirmed as the
correct next lever** (not engram separation).

## The bounded negative — precisely the research-predicted risk
This is exactly the outcome the scoping doc flagged as the live risk: *"a strong point-neuron NMDA attractor with lateral
inhibition may latch to one dominant sub-assembly rather than N selective ones."* It is the same **P0.3 saturation
boundary** (`2026-07-24-P0.3-affect-state-region-6seed-GO.md`: on point neurons the NMDA attractor ignites low + saturates
→ a single bistable winner, not a graded/selective assembly). The potentiation-strength sweep (30 vs 100 cycles, both
chance-selectivity) confirms it is NOT a tuning shortfall in the routing — it is the point-neuron attractor dynamics.

## SFA-eviction lever EXHAUSTED (9-config sweep, seed 42) + the 6-seed CORRECTION (the boundary is SEED-VARIABLE, at-chance-mean)
Injected spike-frequency adaptation directly on the slot neurons (runner-side, no `sim/` edit: `cp_izh_d_increment` +
`cp_izh_a`) and swept it **at seed 42**: no-SFA + a grid of d∈{50,100,200,300} × a∈{0.02,0.05} → every config SELECTIVE
≤1/3; BEST 1/3 @ no-SFA (strong SFA d=500/a=0.01 over-suppresses to 0/3 ignition). No SFA regime helps *at that seed*.
**⚠️ 6-SEED CORRECTION (the honest refinement — my initial "systematically confirmed single-winner / all ≤chance" was a
1-seed OVER-CLAIM; seed 42 is a low-selectivity seed):** running the no-SFA baseline across **6 seeds** gives SELECTIVE =
**[1, 2, 1, 0, 1, 2], mean 1.17/3 ≈ chance (1.0/3)** — it is **SEED-VARIABLE**: 2/6 seeds reach 2/3, three are at chance
1/3, one at 0/3, and it **NEVER robustly reaches the ≥⌈N/2⌉ GO bar (2/3 on only 2/6 seeds; never 3/3).** ⇒ the corrected
verdict: the point-neuron regime is **at-chance-mean, seed-variable, NOT robustly one-of-N** — the boundary stands as
"point neurons do not ROBUSTLY grade into selective compositional assemblies," but it is NOT the clean "always
single-winner" the seed-42 sweep suggested. The SFA sweep itself was seed-42-only; the baseline is seed-variable. The
dendritic surpass remains well-motivated (point neurons never robustly clear the GO bar) and is the confirmed next lever.

## Verdict + next (per THE LAW — the residual is precisely mapped + the surpass named, NOT a wall)
- **CONFIRMED (1-seed):** co-activation replay clears the A1 frozen-wire failure (directional potentiation) + the
  dedicated region ignites/holds. **BOUNDARY (1-seed, robust to cycle-count):** selective one-of-N binding hits the
  point-neuron single-dominant-winner pathology.
- **The surpass — SFA-eviction TRIED + systematically EXHAUSTED (9-config sweep, all ≤chance) → the DENDRITIC line/bump
  attractor is the confirmed lever** (P0.3's named ultimate lever for graded/selective point-neuron assemblies; the
  standing dendritic-cortex project priority). A point-neuron adaptation+WTA does NOT grade into N selective assemblies.
- **NEXT:** (1) **6-seed** confirm this systematic bounded negative (1-seed sweep is a strong indicator, not yet the
  6-seed bar); (2) build the **dendritic line/bump attractor** on the slots (`sim/dendritic_*` — a two-compartment /
  plateau mechanism that supports a graded, one-of-N selective assembly where the point-neuron NMDA+WTA saturates), then
  re-test selectivity + HOLD + the full anti-cheat suite (no-region / no-NMDA / no-replay / no-co-activation /
  hippo-lesion-after / permuted-tag / control-outperforms) on a CACHED Phase-1 substrate. Full GO-gate in the scoping
  finding. The co-activation potentiation fix stands CONFIRMED regardless.

## Provenance
`scratchpad/consol_coact_smoke.py` (harness `nmda_compositional_consolidation.py`: `build_substrate` +comp_attractor
region, `coactivation_replay`; logs `consol_coact{4..9}.log`). Reuses the harness + the research-gate scoping (`acd06561`).
NO `sim/` edit. GPU. 1-seed indicator (offline, Phase-1 skipped for the mechanism check).
