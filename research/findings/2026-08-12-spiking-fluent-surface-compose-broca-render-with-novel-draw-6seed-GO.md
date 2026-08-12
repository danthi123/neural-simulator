---
type: finding
status: contributing
date: 2026-08-12
mechanism: brain-native fluent surface (A1 burn-down) — COMPOSE the spiking BROCA frame-and-slot render (order = the per-pool spiking-RATE ranking on a real Izhikevich SimulationBridge, EMERGE-59/61) WITH the brain's novel-content DRAW (weighted resample over its learned association graph, the _burndown_3E GO proposer mechanism), so a GENERATED novel-but-plausible hypothesis gets a GRAMMATICAL, FAITHFUL, moat-safe SURFACE produced on firing neurons — replacing the agrammatic host f-string "perhaps bear walk foot"
lane: E · Language / integration-first (retiring the external Qwen mouth, A1)
verdict: 6-seed GO (42/43/44/100/101/102). Arm A (hedged transitive "perhaps the <S> <V-3sg> the <O>"): grammatical 0.993 / faithful 0.993 (the INDEPENDENT re-parse recovers the DRAWN SVO — the moat), hypothesis-flag 1.0, 0 confab leaks; every control collapses (PERMUTED 0.0, NO-LEARNING 0.0, EQUAL-DRIVE 0.0, WRONG-CONTENT recover 0.0 vs main 0.993); genuinely spiking (order = the SimulationBridge per-pool rate ranking, bridge advanced real spikes every seed, NO torch/transformers in-process). Arm B (coordinating connective "... and ..." over two independently-drawn SVOs): grammatical 0.99 / faithful 0.99 (BOTH clauses recovered), permuted collapses 0.0. The composition of two GO pieces (never previously wired) yields a brain-native spiking mouth that renders a DRAWN SVO grammatically-and-faithfully — the agrammatic host f-string can be retired for the GENERATE channel's transitive hypotheses. HONEST BOUNDARY: the DRAW's content SAMPLING is host bookkeeping over the brain's learned graph (the fully-spiking SWR-CA3 draw, _followon1, is a banked HONEST_NEGATIVE — a separate residual); the RENDER-ORDER is on spikes; the A->W spell is the identity-surface callback (concept_speak_demo).
artifacts:
  - research/runners/_spiking_fluent_surface_derisk.py
  - research/findings/raw/_spiking_fluent_surface_derisk.json
verification: _spiking_fluent_surface_derisk --derisk, seeds 42/43/44/100/101/102, SIM_BACKEND=numpy. go True, arm_a_go True, arm_b_go True. Arm A grammatical/faithful mean 0.993 (min 0.958 seed 101), hyp-flag 1.0, confab_leaks_total 0, all controls 0.0 (perm/nolearn/equal/wrong-content), all_spiked True, no_torch_all True. Arm B grammatical/faithful mean 0.99 (min 0.938 seed 101), permuted 0.0. Spiking-authenticity probe: graded primacy drive -> monotone per-pool rates 0.429>0.4>0.39>0.237>0.171>0.098 recovering the taught order [0..5]; EQUAL drive -> ~flat rates (spread 0.081) -> order collapses (the graded spiking drive carries the order, read from real spikes, not a host argsort).
---

# Spiking fluent surface (A1): the brain's GENERATED hypothesis rendered grammatically + faithfully ON SPIKES

## The exact fluency boundary this composition crosses (mapped from the record, not assumed)

The best brain-native SPIKING surface today already renders a SINGLE, GIVEN clause across the core relational
schema on firing neurons — order = the per-pool spiking-RATE ranking on a real Izhikevich `SimulationBridge`,
every word via the A->W read-out, productive 3sg inflection, position-independent (EMERGE-61 wash-out), gate-first
no-confab moat:

- property "the owl can fly" (F_MODAL), negated modal "the penguin does not fly" (F_NEGMOD) — EMERGE-59/60/61 GO;
- transitive "the dog eats the cat" (C_TRANS), spatial "the owl runs to the pond" (C_PPGOAL), ditransitive "the
  dog gives the cat a bone" (C_DITRANS, 7 slots) — the 2026-07-08 schema-breadth-complete GO (EMERGE-74/77).

SEPARATELY, the brain GENERATES novel-but-plausible propositions from its OWN learned association graph
(`_burndown_3E_brain_owns_generation`, 6-seed GO) — but it surfaces each hypothesis with an **agrammatic HOST
F-STRING** (`say_hypothesis`: `f"perhaps {a} {ac} {p}"` -> "perhaps bear walk foot": no determiner, no
subject-verb agreement, no clause). **The two GO pieces had never been composed.** So the precise boundary: the
brain's *generated* content had NO grammatical spiking surface; the fluent prose of the production `/api/brain-chat`
turn is the external Qwen-0.5B off-bridge mouth (the A1 shortcut). Open *arbitrary* prose (a from-scratch spiking
sequence-LM) is a separate, banked deep-context BOUNDARY (2026-07-11 R1-stream-eprop-longrange), not this lever.

## The lever de-risked — compose the spiking BROCA render with the novel-content DRAW

`research/runners/_spiking_fluent_surface_derisk.py` composes them: the DRAW proposes a novel SVO over the brain's
learned graph; the spiking BROCA renders it. Two arms, each rendered on firing neurons, GO-gated by grammaticality
(an INDEPENDENT held-out grammar check + moat re-parse) + faithfulness (the re-parse recovers the DRAWN content) +
genuinely-spiking (order from the `SimulationBridge` rate ranking; no external transformer):

- **Arm A — hedged transitive** "perhaps the <S> <V-3sg> the <O>" (determiner + transitive SVO clause + epistemic
  hedge). Samples: "perhaps the tove hunts the plim", "perhaps the seal spots the hare", "perhaps the lynx snerks
  the vole". The "perhaps" is BOTH the discourse marker and the hypothesis flag (moat: a guess, never a known fact).
- **Arm B — coordinating connective** "the <S> <V-3sg> the <O> and the <S> <V-3sg> the <O>" (two independently-drawn
  clauses joined by "and"). Samples: "the seal spots the hare and the wug prowls the mole", "the tove hunts the plim
  and the zib zonks the newt". Each clause is an INDEPENDENT spiking production (the EMERGE-61 wash-out).

The RENDER-ORDER is genuinely on spikes: the learned per-clause primacy gradient -> graded external current into
slot pools on the real bridge -> the per-pool spiking-RATE ranking = the emission order (rate-coded competitive
queuing; the validated `_phaseB_serial_order_spiking` read-out). Words spelled by the A->W read-out (identity-surface
callback here, the EMERGE-59 precedent). Vocab is arbitrary (a mix of real + pseudo-words, varied per seed) to make
"vocab-agnostic" concrete; verb morphology is regular/invertible so the independent parser can de-inflect the
surface (comprehension knows the lexicon).

## Result — 6-seed GO (42/43/44/100/101/102, CPU numpy)

Raw artifact (headline + per-seed + the spiking-authenticity probe + the earned `preconditions` block):
`research/findings/raw/_spiking_fluent_surface_derisk.json`.

| metric | Arm A (hedged transitive) | Arm B (connective "and") |
|---|---|---|
| grammatical (independent check) | **0.993** (min 0.958) | **0.99** (min 0.938) |
| faithful / moat re-parse recovers drawn SVO | **0.993** (min 0.958) | **0.99** (both clauses) |
| hypothesis-flag ("perhaps") / 0 confab | 1.0 / **0 leaks** | n/a |
| PERMUTED-order control | **0.0** | **0.0** |
| NO-LEARNING control | **0.0** | — |
| EQUAL-DRIVE control (no graded primacy) | **0.0** | — |
| WRONG-CONTENT re-parse (moat is discriminative) | **0.0** | — |
| genuinely spiking (bridge spiked / no torch) | **True / True** | True / True |

Every anti-cheat collapses to chance/zero while the main is ~1.0: the correct word order is produced by the LEARNED
gradient READ OUT THROUGH SPIKES (permute the taught order, or don't learn it, or drive every pool equally, and
grammaticality goes to 0.0), and the moat re-parse is discriminative (re-parsing surface(A) against a different
SVO(B) recovers 0.0, not a pass-through). The spiking-authenticity probe confirms the mechanism directly: the
graded drive yields monotone per-pool rates (0.429>0.4>0.39>0.237>0.171>0.098) recovering the taught order,
while EQUAL drive flattens them (spread 0.081) so the order collapses — the order is READ FROM REAL SPIKES.

## Verdict

**6-seed GO, both arms.** A brain-native SPIKING mouth that renders a DRAWN novel SVO **grammatically AND faithfully
(moat-safe)** on firing neurons IS feasible — the agrammatic host f-string "perhaps bear walk foot" can be RETIRED
for the GENERATE channel's transitive hypotheses, and a coordinating connective ("... and ...") is in reach on the
same spiking read-out. This is a concrete step toward retiring the external Qwen mouth (A1): the *generated* content
now has a grammatical spiking surface, not just the *recalled* content.

## Honest scope + residuals (the deliverable boundaries, not caveats)

- **The DRAW's content SAMPLING is host bookkeeping over the brain's LEARNED graph** (weighted resample over
  learned co-occurrence — the `_burndown_3E` GO mechanism, self-contained here). It is NOT spiking. The FULLY-spiking
  draw (SWR-gated CA3 resampler, `_followon1_spiking_generative_sampler`) is a **banked HONEST_NEGATIVE** (it did not
  match the host sample-loop quality) — a SEPARATE residual. So the composed pipeline is
  [host-sampled-over-the-brain's-learned-graph DRAW] -> [SPIKING-order BROCA render]; the de-risked NEW piece is the
  spiking render of an arbitrary drawn hypothesis + the moat/faithfulness.
- **The A->W spell is the identity-surface callback** (the EMERGE-59 precedent); its own spiking validation is
  `concept_speak_demo` (100% multi-seed). Wiring the real trained-bridge A->W read-out is the GPU follow-on.
- **Seed 101 is a thin margin** (Arm A 0.958, Arm B 0.938): one drawn item's 6-slot order swaps two adjacent
  content slots — the known competitive-queuing read-out tail (EMERGE-61's wash-out reduces but does not fully
  eliminate it at 6 tightly-packed ranks with WTA_NOISE=0.25). EMERGE-77's 8-pool 2-stage bias-calibrated read is
  the mechanism that closed the same tail for the 7-slot ditransitive; applying it here would lift the min above 0.99.
- **The two determiner slots are identical words** ("the"), so a DET-DET order swap is invisible to the surface and
  counts as grammatical (both orderings produce the identical correct string) — a harmless degeneracy; the
  content-carrying slots (SUBJ/VERB/OBJ) ARE order-observable and are what the controls collapse.
- **Reproducibility:** the main metrics were stable across runs, but the near-zero WRONG-CONTENT control showed a
  sub-threshold flip on one borderline seed until the process-global RNGs were pinned per seed (the CLAUDE.md seed
  trap: `cfg.seed` seeds the substrate, but a razor-edge spiking tie-break also reads the process-global stream).
  With `_pin_global(seed)` the de-risk is byte-reproducible across two consecutive processes (wrong-content stably
  0.0). The verdict never depended on it — the control was sub-threshold either way.
- **NO `sim/` edit.** Reuse-by-import: the EMERGE-59 spiking order read-out, the EMERGE-57 frame-aware 3sg
  inflection, the EMERGE-61 (scoped) wash-out.

## The single most promising next lever toward retiring the Qwen mouth

**Wire this composed spiking render onto the production `/api/brain-chat` GENERATE channel** (in place of the
`say_hypothesis` host f-string), lesion-verified and moat-preserved, then broaden the frame inventory the DRAW can
feed (property/negation/ditransitive already render on spikes) so the GENERATE channel's fluent surface is
brain-native — reducing the Qwen mouth's remaining job to open *arbitrary* prose (the banked deep-context wall),
which then becomes the one clearly-scoped residual for A1.

## Provenance
- Reused by import: `_emerge59_spiking_broca_frame_slots_derisk` (`build_slot_bridge`, `slot_pool_rates`,
  primacy/wash params), `_emerge57_ra_refinetune_emerge_frames_derisk` (`emerge_v3`),
  `_emerge61_spiking_broca_order_robustness_derisk` (`_snapshot_state`/`_restore_state`, scoped wash-out).
- Composed with: the `_burndown_3E_brain_owns_generation` DRAW mechanism (6-seed GO) + `_genfrontier_b2_generative_replay`.
- Prior boundary mapped: `_followon1_spiking_generative_sampler` (fully-spiking SWR-CA3 draw, HONEST_NEGATIVE);
  the fluent-mouth-is-Qwen integration finding (2026-08-12); the deep-context spiking-LM wall (2026-07-11
  R1-stream-eprop-longrange).
- Catalog/literature: G.12 Broca frame-and-slot (Bock & Levelt 1994; Levelt-Roelofs-Meyer 1999); G.07/H.19
  competitive queuing (Grossberg 1978; Bullock-Rhodes 2003; Kornysheva et al. 2019); G.09 constructive generative
  replay.
