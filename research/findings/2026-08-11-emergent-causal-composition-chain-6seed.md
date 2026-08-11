---
type: finding
status: go
date: 2026-08-11
mechanism: goal-directed causal composition (motion+goal+spatial-location join) over the RF-VSA fact store, every edge a query_patient moat read
lane: Stage-A conversation / honesty-boundary-as-deliverable (the INTEGRATION #5 follow-on)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/stageA/causal/causal_composition_chain_tier0_6seed.json
  - research/findings/raw/lanes/stageA/causal/causal_composition_chain_s42.json
---

# Emergent causal composition — a "why did the dog go east?" answer COMPOSES two stored facts + a spatial grounding into a moat-verified causal chain, and ABSTAINS to the #5 disclaimer on the confab traps (6-seed GO, 0 false-accepts)

**The #5 follow-on, named per THE LAW.** INTEGRATION #5
([`2026-08-10-INTEGRATION-5-honest-causal-query-disclaimer-turn4-6seed.md`](2026-08-10-INTEGRATION-5-honest-causal-query-disclaimer-turn4-6seed.md))
made turn 4 of the live chat ("why did the dog go east?") HONESTLY DISCLOSE that the brain has no causal faculty —
it confirms the stored fact via the no-confab moat (`comp.query_patient("dog","go") -> east`) and declines to invent
a reason. That finding named the truly-emergent answer as the follow-on: COMPOSE stored facts into a grounded causal
CHAIN — "dog goes east" + "dog looks at river" + [river is east] ⇒ "to reach the river". This de-risk builds and
measures exactly that composition, and asks whether the substrate can chain a causal answer WITHOUT confabulating.

## The composition — a goal-directed JOIN, every edge a `query_patient` moat read

For "why did AGENT MOTION?" the answer is composed entirely from three moat reads over the flat SVO store:

```
HOP 1  dir     = comp.query_patient(agent, motion_verb)     # where the agent moved (a stored direction)
HOP 2  obj     = comp.query_patient(agent, goal_verb)       # the agent's OWN goal-object  (SHARED-ENTITY join)
HOP 3  obj_dir = comp.query_patient(obj,   locative_verb)   # where that object is located
COMPOSE  iff  dir == obj_dir  ->  "the AGENT moves <dir> to reach the <obj>"    else  ABSTAIN (#5 disclaimer)
```

Because every link is a `query_patient` read (abstain → None), a composed answer asserts ONLY moat-confirmed facts:
**0 confabulation by construction**. The de-risk's teeth are the moat's DISCRIMINATION — it must compose only when
the chain genuinely grounds, and abstain on the two confab traps a "why" invites. The abstain reply reuses the #5
`_honest_causal_answer` byte-for-byte, so an unsupported "why" degrades to exactly the #5 behaviour, not to silence
or invention.

## Result — 6 seeds (42/43/44/100/101/102), SIM_BACKEND=numpy, cfg.seed-controlled

<!--derived--> (Tier-0 core claim: `causal_composition_chain_tier0_6seed.json`; Tier-1 live + full seed-42:
`causal_composition_chain_s42.json`.)

Tier-0 is an 8-query grid over a toy causal world (motion facts + goal facts + spatial-location facts). Two queries
GROUND (the chain composes); six must ABSTAIN, spanning all four abstain reasons and both confab traps.

| grid query | outcome (6/6 seeds) | why |
|---|---|---|
| why dog go | **SUPPORTED → "to reach the river"** | (dog,go)→east · (dog,look)→river · (river,at)→east ✓ |
| why cat go | **SUPPORTED → "to reach the apple"** | (cat,go)→west · (cat,look)→apple · (apple,at)→west ✓ |
| why dog run | **ABSTAIN** (dir_mismatch) | GOAL-SHORTCUT trap: river is dog's goal but river is EAST; the dog ran NORTH |
| why cat run | **ABSTAIN** (dir_mismatch) | GOAL-SHORTCUT trap: apple@west, cat ran south |
| why fish go | **ABSTAIN** (no_goal) | SPATIAL-SHORTCUT trap: river IS east, but the fish has no stored goal |
| why bird go | **ABSTAIN** (no_spatial) | bird's goal 'hill' has no stored location to ground the direction |
| why dog come | **ABSTAIN** (unstored_motion) | `query_patient(dog,come)` → None |
| why cat stop | **ABSTAIN** (unstored_motion) | `query_patient(cat,stop)` → None |

Aggregate, all 6 seeds unanimous: **supported_correct 2/2 · abstain_correct 6/6 · false_accepts 0 · confab 0 ·
every-edge-moat 2/2 · moat-battery false-accepts 0/8 · permuted-spatial still-supported 0**. GO = 6/6.

**The SUPPORTED composed answer (verbatim, seed 42, identical template all seeds):** *"I know the dog goes east —
that fact is stored, and my no-confab moat confirms it ((dog, go) -> east). This time I can say WHY, because two more
stored facts COMPOSE into a grounded reason: (dog, look) -> river, and (river, at) -> east. So the dog goes east to
reach the river. Every link in that chain is moat-confirmed — I composed it from what I stored, I did not invent it."*

**The confab-trap ABSTAIN (verbatim, "why dog run"):** the #5 disclaimer — *"I know the dog runs north … But I have
no stored reason WHY: I have learned associations, not causes … and I will not invent one."* The dog's goal (river)
is known, but river is EAST and the dog ran NORTH, so the chain does not ground — a naive reasoner answers "to reach
the river"; the moat abstains.

### Tier-1 — the #5 turn-4 disclaimer GRADUATES on the LIVE co-resident one-brain composer (seed 42; 6-seed = the command below)

Built through `SA.build_one_brain(seed, co_resident_affect_ladder=True)` — the real merged-bridge
`CoResidentOneBrainComposer`, whose `query_patient` is the spiking RF-VSA unbind on the co-resident substrate:

- **WITH the spatial grounding stored** (river@east, apple@west added to the 6 curated facts) → turn-4 "why did the
  dog go east?" produces the composed chain above (supported=True, obj=river), confab=0.
- **WITHOUT the spatial grounding** (the #5 world) → the composition abstains (no_spatial) and falls back to the #5
  honest disclaimer, **byte-identical** (`matches_#5_disclaimer=True`).

So the disclaimer graduates to a moat-verified composed reason exactly when the grounding exists, and preserves the
honest #5 fallback when it does not — on the LIVE substrate, not just the standalone composer.

## Anti-cheats (all required, all pass 6/6) — adapted from the multi-hop `query_chain` GO's control set

1. **Permuted-spatial collapse.** Derange the (object,at,dir) grounding (river@west, apple@east). Both
   originally-grounded chains COLLAPSE to abstain (`permuted_spatial_still_supported=0`). The chain READS the stored
   grounding; it is not a hardcoded "dog→river" link. Attribution (`tools.lab.attributable_to`,
   `grounding_attribution`): the composed chains are 100% attributable to the stored spatial grounding (treatment 2
   vs permuted-control 0), not to a hardcoded link.
2. **Permuted-spatial POSITIVE.** A derangement that grounds a DIFFERENT query (river@north) makes "why dog run
   north" become SUPPORTED → "to reach the river", while "why dog go east" now abstains. The supported SET moves with
   the data both directions — not a memorised 2-row answer key.
3. **Goal-shortcut trap = 0 false-accepts.** The dir_mismatch rows (agent has a known goal, moved elsewhere) never
   compose (`goal_shortcut_false_accepts=0`).
4. **Spatial-shortcut trap = 0 false-accepts.** "why fish go east" never grabs the river just because the river is
   east — the join requires the agent's OWN goal (`spatial_shortcut_false_accepts=0`).
5. **Moat battery.** 8 untaught cues all abstain (`query_patient`→None), 0 false-accepts.
6. **Every-edge-is-a-moat-read.** Every triple in every composed chain reads back via `query_patient`; a supported
   answer whose links were not all moat reads would count as a confab — none do (confab=0).

## What is brain-based vs a declared scaffold (per THE LAW + docs/TERMS.md)

**Substrate / mechanism.** Every FACT in every chain is a spiking RF-VSA unbind + cleanup (`query_patient`), and the
moat's abstain-vs-compose DECISION is driven entirely by those reads. 0 mis-bind / 0 false-accept holds at this
scale (≤13 facts, |V|=21), consistent with the `query_chain` GO's clean regime (40 concepts, D-independent).

**Declared scaffold (named, not hidden).** The composition POLICY — the motion+goal+spatial JOIN shape — is a HOST
route, same status as `query_chain`'s caller-supplied action list and the #5 `why`+known-cue trigger. So are the
`comp.store` writes of the spatial-location facts (the composer-as-idealization / host-taught-storage shortcut, the
same status #6 gives its corpus-mined facts) and the answer TEMPLATE (same status as the #5 disclaimer template).

## Honest scope — the DATA path is de-risked; the RELATIONAL STRUCTURE is the named honest negative

**This is NOT emergent causal reasoning, and it does not claim to be.** The de-risked claim is narrow and real: the
point-neuron RF-VSA substrate can reliably SUPPLY every fact of a grounded goal-directed causal chain (0 mis-bind, 0
false-accept, 6-seed), so a composed "why" answer is never a confabulation, and the moat correctly discriminates a
grounded chain from the two confab traps — abstaining to the #5 disclaimer whenever the facts do not support it. That
de-risks the DATA path for the #5 follow-on.

**What is NOT yet emergent (the honest negative that launches the next arc).** The toy substrate stores FLAT
`(agent,action)→patient` associations with NO relational / causal / spatial graph and NO intervention model; the
motion+goal+spatial JOIN is host-orchestrated, and the "(object, at) → direction" grounding is a host-taught fact,
not a learned spatial code. **Next mechanism (per THE LAW): a LEARNED relational / spatial code** — a factorised
relation binder (Whittington TEM, catalog note; the generalize-a-relation-to-unseen-items path the `query_chain` GO
flagged as out of its scope), or the co-occurrence stream cortex learning the (object→location) and (motion→goal)
relations in synapses — so the causal chain EMERGES from the substrate's own structure rather than a host join. At
that point the composition graduates from "the host chained the brain's stored facts" to "the brain composed the
reason", and the #5 disclaimer graduates from "I have not learned causes" to a self-generated grounded causal chain.

## Reproduce

```bash
# 1-seed smoke (Tier-0 grid + Tier-1 live graduation):
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._causal_composition_chain_derisk \
    --seeds 42 --out research/findings/raw/lanes/stageA/causal/causal_composition_chain_s42.json

# 6-seed DECISIVE sweep (Tier-0 grid + Tier-1 live graduation on all seeds) — the coordinator runs this and writes
# the decisive aggregate to research/findings/raw/lanes/stageA/causal/ (path given verbatim in the handoff report):
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._causal_composition_chain_derisk \
    --seeds 42,43,44,100,101,102 --out <decisive 6-seed aggregate JSON under that dir>
```

The multiseed core-claim artifact committed here (`causal_composition_chain_tier0_6seed.json`) is Tier-0 only
(the fast standalone-composer grid); the decisive command above ALSO runs Tier-1 (`build_one_brain`) on every seed.

GO = correct causal chains when the facts support it + honest #5 disclaimer/abstain otherwise + moat 0 false-accepts,
all 6 seeds.
