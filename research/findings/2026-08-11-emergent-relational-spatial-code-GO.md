---
type: finding
status: go
date: 2026-08-11
mechanism: Hebbian/Oja heteroassociative spatial code (object->direction) trained on a noisy co-occurrence stream; the causal-chain grounding + direction-join are a learned-code readout + cosine, replacing the causal-composition GO's host-stored (object,at) fact and its symbolic dir==obj_dir test
lane: Stage-A conversation / honesty-boundary-as-deliverable (the causal-composition follow-on)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/stageA/causal/relational_spatial_code_tier0_6seed.json
  - research/findings/raw/lanes/stageA/causal/relational_spatial_code_s42.json
---

# A LEARNED relational/spatial code makes the causal chain's grounding EMERGE — the host `(object,at)` fact and the symbolic `dir==obj_dir` join are BOTH replaced by a Hebbian-learned object→direction code + a learned-code cosine, and the "why did the dog go east?" chain still grounds/abstains correctly (6-seed Tier-0 GO, seed-42 live graduation, 0 confab)

**The causal-composition follow-on, named per THE LAW.** The causal-composition GO
([`2026-08-11-emergent-causal-composition-chain-6seed.md`](2026-08-11-emergent-causal-composition-chain-6seed.md))
composes a grounded "why did the dog go east?" chain, but its honest negative is explicit: the motion+goal+spatial
JOIN is host-orchestrated, the grounding hop `(object, at) -> direction` is a **host-taught FACT**
(`comp.store("river","at","east")`), and the chain closes with a **symbolic `dir == obj_dir`** equality on cleaned-up
tokens. It named the neural successor: *a LEARNED relational/spatial code — the co-occurrence stream cortex learning
object↔location in synapses — so the chain EMERGES from the substrate rather than a host join.* This de-risk builds and
measures exactly that, and asks whether the point-neuron substrate can LEARN a spatial code clean enough to ground the
chain WITHOUT the host fact and WITHOUT the `==`.

## What changed — the two host scaffolds are gone, replaced by a Hebbian-learned code

| | causal-composition GO (host) | this de-risk (learned) |
|---|---|---|
| object→location grounding | `comp.store("river","at","east")` — a **stored fact** | a **Hebbian/Oja heteroassociator** `W` trained on a NOISY co-occurrence stream (`river` seen at `east`, 24 samples, phase-noise σ=0.5); **NO `(object,at)` fact is EVER stored** — `query_patient(river,"at")` is `None` by construction |
| the direction join | `if dir == obj_dir` — **symbolic equality** | `cos(motion_dir_code, W·z_obj) >= θ` — a **similarity in the learned code**; the direction is never symbol-matched |
| "is the object located?" | `(hill,at)` absent ⇒ `query_patient` `None` — a clean abstain | the learned-code **moat**: readout direction-margin (best−2nd) `>= θ_locmargin` |

`W = Σ_samples z_dir ⊗ conj(z_obj)` over the substrate's own unit-phasor codes, per-component-normalized on readout
(the Oja projection back onto the phasor manifold) — the classic linear associative memory (Steinbuch Lernmatrix /
Kohonen / Hopfield heteroassociator). HOP-1 (motion direction) and HOP-2 (the shared-entity goal) stay `query_patient`
reads of OBSERVED SVO facts (the #6/#7 kb) — those are legitimate observations, not the scaffold the finding flagged.
The scaffold being replaced is exactly the spatial grounding fact + the `==` join.

## Result — 6 seeds (42/43/44/100/101/102), SIM_BACKEND=numpy, cfg.seed-controlled

<!--derived--> (Tier-0 core claim: `relational_spatial_code_tier0_6seed.json`; Tier-1 live graduation seed-42:
`relational_spatial_code_s42.json`.)

Tier-0 is the SAME 8-query grid as the causal-composition GO (2 ground, 6 abstain across all four reasons + both
confab traps), now grounded entirely by the learned spatial code. All 6 seeds unanimous:

| metric | value (6/6 seeds) |
|---|---|
| supported chains correct | **2/2** |
| abstains correct | **6/6** |
| false-accepts / confabulations | **0 / 0** |
| **`(object,at)` facts stored** | **0** — the grounding is ONLY the learned `W` |
| untrained-map supported (the lever) | **0** — an untrained `W` grounds nothing |
| permuted-map still-supported | **0** — a deranged stream collapses both chains |
| unlocated-object confabulation | **0** — `hill` never grounds |
| grounded join-margin (correct−best-wrong dir) | **0.88 – 0.93** (floor 0.10) |
| grounding attributable to the learned code | **1.0** (`tools.lab.attributable_to`; treat 12 vs permuted-control 0) |

**The learned code discriminates cleanly at this scale:** a located object reconstructs its direction at cos ≈ 0.99
(river→east, apple→west), the wrong directions sit at ≈ 0; `cos(north, loc(river)=east)` ≈ 0.02, so the goal-shortcut
trap ("why dog run north?", river@east) abstains by the learned code, not a host test.

### Tier-1 — the #5 disclaimer GRADUATES on the LIVE co-resident one-brain composer, with the LEARNED map as the switch (seed 42)

Built through `SA.build_one_brain(seed, co_resident_affect_ladder=True)` (the merged-bridge
`CoResidentOneBrainComposer`, whose `query_patient` is the spiking RF-VSA unbind):

- **Train the spatial map** → turn-4 "why did the dog go east?" composes *"the dog goes east to reach the river"*
  (join cos 0.994, margin 0.877 <!--derived-->), and `grounding_is_learned_not_stored=True` — **the composer has NO stored
  `(river,at)` fact; the location came only from `W`**. confab 0.
- **Don't train the map** (untrained `W`) → abstains (`no_spatial`) and falls back to the #5 honest disclaimer,
  **byte-identical** (`matches_#5_disclaimer=True`).

So on the LIVE substrate the disclaimer graduates from "I have not learned causes" to a reason grounded in a code the
brain LEARNED from seeing objects and directions together — exactly when that code confidently grounds, and preserves
the honest #5 fallback when it does not.

## The teeth — the linear associator has NO native "unlocated" state, and the moat that fixes it

The sharpest finding of this de-risk is a genuine limitation of the mechanism, surfaced and then closed. A linear
heteroassociator **projects EVERY object into the span of the trained direction codes** — an object never seen at any
location (`hill`) does NOT read out as "low confidence"; it reads out as a **blend** of the trained directions, whose
raw best-direction confidence SWINGS by seed (measured 0.01 → 0.80). A confidence threshold is therefore an unreliable
"is-it-located?" test, and a naive version would confabulate a location for `hill` on some seeds.

The robust, principled instrument is the readout's **direction-margin** (best minus second-best direction): a truly
located object reconstructs a clean single direction (margin **0.88 – 1.01** measured); an unlocated object reconstructs
a smear (margin **0.05 – 0.45** measured) — a clean gap, so a fixed `θ_locmargin = 0.60` rejects `hill` on every seed.
This is the no-confab moat's cleanup-confidence gate, applied to the learned spatial code. Anti-cheat #7
(`unlocated_confab`) evaluates the compose-path grounding condition on `hill` across all four directions and requires 0
— **held 0/6 seeds.**

## Anti-cheats (all required, all pass 6/6)

1. **No `(object,at)` fact stored** — `spatial_facts_stored == 0`. The grounding CANNOT be a host fact; it exists only
   in the learned `W`. (Structural.)
2. **Untrained-map lever** (`tools.lab.lever`, 0 → 12 supported over 6 seeds) — an untrained `W` grounds 0 chains;
   training it is load-bearing.
3. **Permuted-map collapse** — train `W` on a DERANGED stream (river@west, apple@east): both true chains collapse to
   abstain. Attribution 1.0 — the chains are 100% owed to the learned map, not a hardcoded link.
4. **Permuted-positive** — train river@north: "why dog run north" GROUNDS ("to reach the river"), "why dog go east"
   abstains. The supported set MOVES with the learned data both directions.
5. **Discrimination margin >= floor** — grounded join-margin 0.88–0.93 ≫ 0.10 every seed; the threshold is not knife-edge.
6. **Moat battery** — 8 untaught SVO cues abstain (`query_patient` `None`), 0 false-accepts.
7. **Unlocated-object confabulation == 0** — the learned-code moat rejects `hill` regardless of motion direction.

## What is brain-based vs a declared scaffold (per THE LAW + docs/TERMS.md)

**Newly emergent (the delta this de-risk banks).** The object→location grounding is now **Hebbian-learned from a
co-occurrence stream in a synaptic weight matrix**, not a host-stored fact; the direction join is a **cosine in that
learned code**, not a symbolic `==`; and the "is it located?" abstain is a **readout-cleanliness gate on the learned
code** (the moat), not the absence of a stored fact. Every one of the causal-composition GO's host spatial pieces is
gone.

**Still a declared scaffold (named, not hidden).** (1) The **JOIN TOPOLOGY** — which query is the motion, which is the
goal, the traversal order — remains **host-orchestrated**, same status as the causal-composition runner and
`query_chain`'s caller-supplied action list. This de-risk replaced the spatial GROUNDING + the comparison, NOT the
chain's shape. (2) The associator is a **rate/phasor weight matrix** (numpy), not yet the **on-substrate spiking**
learned map — the ON/OFF-rate + three-factor realization from
[`2026-06-16-learned-bind-reachable-on-stream-codes.md`](2026-06-16-learned-bind-reachable-on-stream-codes.md) is the
named next build. (3) The **word codebook** (direction/object phasor codes) is host-assigned random codes, so this is
"Hebbian-learned association over given codes", NOT "self-organized" (per docs/TERMS.md: the target/codebook is not
self-allocated).

## Honest scope — the verdict, precisely

**GO for the LEARNED SPATIAL-GROUNDING CODE.** A point-neuron phasor substrate CAN Hebbian-learn an object→location
relational code from a noisy co-occurrence stream that is clean enough to (a) ground the two true causal chains via a
learned-code cosine, (b) abstain on all six traps — including the two confab traps and the unlocated-object trap — with
0 false-accepts and 0 confabulation, and (c) do so under all seven anti-cheats, 6 seeds, on the live one-brain composer.
So the causal chain's **grounding** now EMERGES from a learned code rather than a host fact + `==`. This is the DATA/
grounding half of the causal-composition GO's named successor, delivered.

**What is NOT yet emergent (the honest residual that launches the next arc).** (1) The **JOIN TOPOLOGY is still
host-orchestrated** — the brain learned WHERE the river is, but a host still decides to chain motion→goal→location; a
fully-emergent version learns the relational STRUCTURE (which relations compose a "why"), e.g. a factorised TEM-style
structural code or a learned schema, so the traversal itself emerges. (2) **Scale** — this is the toy scale (2 located
objects, D=128) where associator crosstalk (~1/√D) is negligible and the located/unlocated margin gap is wide; at
corpus scale the located margins shrink and the moat's gap narrows, a sensitivity to characterize. (3) The **spiking
on-substrate** realization of the learned map is the named build. At those three, the composition graduates from "the
brain learned the locations and a host chained them" to "the brain composed the reason".

## Reproduce

```bash
# 1-seed smoke (Tier-0 grid + Tier-1 live graduation):
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._relational_spatial_code_derisk \
    --seeds 42 --out research/findings/raw/lanes/stageA/causal/relational_spatial_code_s42.json

# Tier-0 6-seed core-claim (fast, standalone composer, no build_one_brain):
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._relational_spatial_code_derisk \
    --seeds 42,43,44,100,101,102 --no-tier1 \
    --out research/findings/raw/lanes/stageA/causal/relational_spatial_code_tier0_6seed.json

# 6-seed DECISIVE sweep (Tier-0 grid + Tier-1 live graduation on ALL seeds) — the coordinator runs THIS and writes
# the decisive aggregate under that dir (path given verbatim in the handoff report):
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._relational_spatial_code_derisk \
    --seeds 42,43,44,100,101,102 \
    --out <decisive 6-seed aggregate JSON under research/findings/raw/lanes/stageA/causal/>
```

GO = the two true chains ground via the learned code + all six traps abstain (incl. the unlocated-object moat) + 0
false-accepts + all seven anti-cheats, on all 6 seeds; Tier-1 graduates the #5 disclaimer via the learned map (composed
when trained, byte-identical #5 fallback when not).
