---
type: finding
status: go
date: 2026-08-13
mechanism: causal-forward-model grounded in the production RF-VSA fact store (why/what-if over real learned facts)
verdict: GO (6/6 seeds, runner-level de-risk; NOT yet wired/integrated to production)
lane: T1-4 · Learned causal forward model — grounding the events in the real fact graph
artifacts:
  - research/findings/raw/_causal_forward_model_grounded_6seed.json
verification: >
  substrate seeded (cfg.seed set on all three RNGs via the reused build_forward_model); the runner's
  OWN tools.verdict.Verdict returns GO with all 15 [ok] checks; the grounding lesion HOLDS at
  measurement by construction (the fact is dropped from the composer at BUILD, before any training,
  so the missing causal edge can never form); the edge lesion HOLDS (STDP + reward modulation frozen
  before the edges are zeroed); the instrument was read in both directions (D_rate 98-100 Hz vs
  offchain 0 Hz; do(C)->Y 164-167 Hz vs do(X)->Y 0 Hz) from cp_firing_states block-rate reads, not a
  summary; every fact is a spiking RF-VSA query_patient unbind, every prediction a cp_firing_states
  argmax — no host transition table, no host formula computes the prediction, the cause, or the moat.
---

# The learned causal forward model, GROUNDED in the brain's real fact store — a real-fact "why did X" / "what happens if X" answered by the spiking substrate, moat-safe (6/6-seed GO, 2026-08-13)

## Result
The learned causal forward model (`2026-08-12`, 6/6 GO) predicted directed n-way state transitions
and ran a Pearl DO-intervention on spikes — but over TOY host block-drives (A/B/C/D bare indices),
so it could not answer a real conversational "why" / "what-if" over the brain's ACTUAL learned
facts. That finding named its own next rung: *"events are delivered as block drive … grounding them
in the emergent relational code is the follow-on that makes the states themselves learned."* This
de-risk closes that rung: the forward model's event population is now wired to the **production
RF-VSA fact store** — the `RFPhasorComposer` whose `query_patient(agent,action)->patient` is the
no-confab moat the live chat recall uses — so every event IS a real `(agent,action)->patient` fact,
the causal machinery runs over the REAL fact graph, and the why/what-if ANSWERS are real recalled
facts, moat-confirmed. **6/6 seeds (42/43/44/100/101/102), SIM_BACKEND=numpy.**

The real-fact causal world (every state is a real stored SVO fact):
- **CHAIN** A=(dog,go,east) -> B=(dog,reach,river) -> D=(dog,drink,water), taught as ADJACENT pairs
  so A->D is NEVER a taught edge — the "what happens if the dog goes east?" consequence must be a
  substrate rollout.
- **CONFOUND** C=(sun,rise,sky) is a COMMON CAUSE of X=(bird,sing,dawn) and Y=(dog,wake,morning); X
  is observed just before Y, so temporal-order STDP tags a SPURIOUS X->Y (the bird's song does NOT
  wake the dog; the sunrise does).

Two real-fact answers the substrate produces (verbatim, seed 42, identical across seeds):
- **What-if:** *"If the dog goes east, it will drink water — a consequence I rolled forward through
  dog reaching the river, and my no-confab moat confirms (dog,drink)->water is a fact I stored."*
  (D fires at 98-100 Hz via B; every off-chain event at 0 Hz; the direct A->D weight stays at init
  0.2 — the consequence is a rollout, not a recalled A->D edge.)
- **Why:** *"The dog wakes (morning) because the sun rises — that cause survives a DO-probe (forcing
  the sun to rise makes the dog wake; forcing the bird to sing does NOT), so it is a cause not a mere
  correlation, and (sun,rise)->sky is a fact I stored."* (why-cause = C on all 6 seeds; do(C)->Y
  164-167 Hz vs do(X)->Y 0 Hz.)

Runner: `research/runners/_causal_forward_model_grounded_derisk.py` (NEW; reuse-by-import of the toy
de-risk's primitives, NO `sim/` edit). Artifact: `research/findings/raw/_causal_forward_model_grounded_6seed.json`.

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._causal_forward_model_grounded_derisk \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_causal_forward_model_grounded_6seed.json
# fast 1-seed pipeline check: --smoke  (intact GO | edge-lesion collapse | ground-lesion collapse | corr_only wrong)
```

## The grounding — three load-bearing bindings to the production composer (NOT a relabeling of the toy)
1. **The event set is DERIVED from the composer.** The event blocks are enumerated by QUERYING the
   composer (`query_patient` — the spiking RF unbind): a candidate `(agent,action,patient)` becomes
   an event ONLY if the brain's recall confirms it. All 6 candidate facts are moat-recalled on every
   seed (6/6); the 8-cue untaught moat battery abstains (0 false-accepts, 6/6).
2. **The causal curriculum is GATED by recall.** A causal episode (fact_i then fact_j) is taught
   ONLY when BOTH endpoints are moat-recalled. This is the **GROUNDING LESION** (distinct from the
   toy edge-lesion): drop a fact from the composer and its event vanishes, so every causal edge
   touching it never forms. Drop D=(dog,drink,water) -> events 5/6, the B->D edge never trains, the
   **what-if collapses** (D no longer predicted, 3/3). Drop C=(sun,rise,sky) -> the C->Y edge never
   trains, the **why-cause collapses** (3/3). The composer is load-bearing — this is real grounding,
   not a toy under a new label.
3. **The answers are real recalled facts, moat-safe.** The what-if successor and the why-cause are
   each mapped BACK to a fact and CONFIRMED by `query_patient`. A predicted consequence that is not a
   moat-confirmed fact is REJECTED — **0 confabulation across all 6 seeds**. The organ reads/notices;
   it never manufactures a fact.

## Mechanism (brain-based; carried from the toy de-risk, now over real-fact events)
One recurrent EVENT population, one assembly per real fact; cross-block edges weak + plastic. The
edges are DIRECTED + CAUSAL via (1) temporal-order STDP (Mehta-Blum-Abbott causal window — the
teacher renders each episode as an ordered pair, so block-i fires before block-j -> the asymmetric
window tags i->j, depresses j->i) and (2) phasic dopamine three-factor plasticity
(`reward_defer_stdp_weight_update`: STDP timing tags the eligibility, a DA signal converts it). The
**why** read is a spiking DO-probe: for each candidate cause i, do(i) (HOLD i) and read the target's
firing rate; the cause is the argmax (C=164-167 Hz, all others 0). The **DO-intervention** prune
(Pearl do-calculus, teacher-scaffolded): forcing do(X) in isolation fires Y via the spurious edge,
a NEGATIVE DA depresses it; the genuine C->Y and the chain are never driven by X here, so they
survive — the invariance-across-interventions principle (Peters/Scholkopf) on spikes.

## Anti-cheats (all pass 6/6 or 3/3)
- **Edge lesion (load-bearing)** — zero the learned cross-block edges (learning frozen first, holds):
  forward prediction collapses (acc 1.0 -> 0.5), unseen D -> 0, why -> undefined. 3/3.
- **GROUNDING lesion (load-bearing, the NEW grounding teeth)** — drop a fact from the COMPOSER: drop
  D collapses the what-if (3/3); drop C collapses the why-cause (3/3). Distinct from the edge lesion:
  it proves the COMPOSER (the real fact store), not just the forward edges, is load-bearing.
- **Correlation-only (the DO-prune is load-bearing, for BOTH the intervention AND the why)** — run
  WITHOUT the interventional phase: the spurious X->Y survives, so do(X) WRONGLY fires Y (161-167 Hz,
  Xcause=True) AND **why(Y) WRONGLY reads X (the bird sang) on all 3 seeds** — the model concludes the
  bird's song wakes the dog. WITH the intervention, do(X)->Y = 0 and why(Y) = C. The cause-vs-
  correlation separation is 106% attributable to the DO-prune (the control moves OPPOSITE). 3/3.
- **Shuffle (structure, not a fixed template)** — relabel which real facts are causally linked: the
  forward prediction scored vs the TRUE chain fails (acc 0.0). The model learns the structure it is
  SHOWN. 3/3.
- **Moat no-confab** — every predicted consequence is a moat-confirmed real fact; the moat battery
  abstains on 8 untaught cues (0 false-accepts). 6/6.
- **Brain-based** — every fact is a spiking RF-VSA `query_patient` unbind; every prediction/cause is
  a `cp_firing_states` block-rate read; no host argmax over a stored transition table.

## What is brain-based vs the declared boundary (per THE LAW + docs/TERMS.md)
Neural + learned: every fact (spiking RF-VSA unbind + cleanup), every prediction (evt block-rate
argmax), the forward-simulation (the substrate's own directed propagation — D via B, A->D unlearned),
the transition edges (STDP + DA three-factor). Declared boundary (carried from the toy de-risk): the
teacher renders the event drive, the temporal ORDER of each episode, and the phasic-DA SIGN (the
environment/teacher reinforcement; the brain's dopamine channel converts it to a weight change). This
is **NOT** claimed as `closed`, `wired`, `integrated`, or `fully spiking` — it is a runner-level GO;
the read-out is a spiking argmax and the DA sign is teacher-delivered.

## Honest boundary + the next rungs (first-class deliverables, not caveats)
- **Grounding-by-DERIVATION, not yet grounding-by-shared-SUBSTRATE.** The events are DERIVED from +
  gated by the composer's moat recall (and the answers re-confirmed by it), but the composer's unbind
  SPIKES do not yet directly DRIVE the event blocks in one merged bridge. Wiring the composer's
  recall spikes as the event drive (one substrate end-to-end) is the named next rung.
- **The DA sign is teacher-delivered.** Driving it from a spiking mismatch unit (E2's surprise read
  -> a from_reward/from_novelty DA) so the prediction-error is itself neural is the named next rung
  (shared with the toy de-risk).
- **First-order transitions only** (state -> next). History-dependent transitions need the HTM-TM
  high-order predictor (EMERGE-15 GO) — a composition, not a new wall.
- **The fact set + causal curriculum are teacher-rendered** (the environment boundary), same status
  as the composer's host-taught `store` writes and the composition-chain's host JOIN policy. The
  causal EDGES between facts are still delivered as ordered episodes; learning WHICH facts causally
  connect from raw co-occurrence (a learned relational/spatial code, TEM-style) remains the deeper
  arc the `2026-08-11` composition-chain finding named.

## Path to a production spiking "why / what-if" organ (a wiring spec — NOT yet wired)
The production turn (`webapp/server.py` `/api/brain-chat` -> the `ChatBrain`) can gain a co-resident
grounded forward-model organ, exactly as this runner assembles it:
1. **Build once, alongside the composer.** At `ChatBrain` build, instantiate the forward-model event
   population; enumerate its events by `comp.query_patient` over the composer's stored facts (the
   event set == the brain's recalled facts). Train the directed edges from the teacher's / corpus's
   ordered causal episodes (the same DA-gated temporal-order STDP this runner uses).
2. **On "what happens if <agent> <action>?"** — cue the fact's event, roll the substrate forward,
   read the spiking successor event, map it back to a fact, and CONFIRM via `comp.query_patient`
   (moat-safe: reject any predicted consequence the composer cannot confirm). Emit the real fact.
3. **On "why did <agent> <action>?"** — read the directed edge INTO the fact's event as the argmax
   DO-probe predecessor (the cause), confirm it via `comp.query_patient`, and confirm it survives a
   DO-probe (do(cause) fires the effect; a spurious correlate does not). Emit the real cause fact.
4. **Moat-safe by construction** — every emitted fact is a `query_patient` read; an unconfirmed
   prediction abstains to the honest `_honest_causal_answer` disclaimer (the INTEGRATION #5 fallback),
   so the organ NEVER manufactures a fact. Wiring it default-on + LESION-verified (disable the
   forward edges -> the default why/what-if CHANGES) is the integration follow-on that would earn the
   `wired`/`on-by-default`/`integrated` terms this de-risk does not claim.

## Provenance
`research/runners/_causal_forward_model_grounded_derisk.py` (modes: intact; edge_lesion;
ground_lesion_D; ground_lesion_C; corr_only; shuffle; `--smoke`). Reuse-by-import of
`_causal_forward_model_derisk` (build/step/train/reads/lesion) + `rf_phasor_composer.RFPhasorComposer`
(the production fact store). Uses `tools.lab.attributable_to` + `tools.verdict.Verdict`. NO `sim/`
edit. CPU/numpy (the ~180-neuron forward model + the small RF composer get no GPU benefit — the
E2/causal precedent, same scale, ran 6-seed on numpy CPU).
