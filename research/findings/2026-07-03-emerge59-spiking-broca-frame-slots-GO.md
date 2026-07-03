# EMERGE-59 RUNG A — SIMULATE BROCA on spikes: EMERGE's reply frames rendered FLUENTLY with LEARNED function-word + inflection SLOTS — 6-seed GO

**2026-07-03. CPU/numpy de-risk (seconds). Reuse-by-import; NO `sim/` edit.** The first genuine "simulate Broca, don't
bolt on an LLM" step: EMERGE's fixed reply frames are rendered FLUENTLY on the SPIKING substrate by a frame-and-slot
grammatical encoder, so the 21M ANN generator can be RETIRED for those frames. Per the research gate
`research/findings/2026-07-03-simulate-broca-generator-replacement-research-gate.md` (Rung A, the recommended cheapest
first step). Runner `research/runners/_emerge59_spiking_broca_frame_slots_derisk.py`; test
`tests/test_emerge59_spiking_broca_frame_slots.py` (7 tests, CPU/numpy, offline); raw
`research/findings/raw/_emerge59_spiking_broca_frame_slots.json`.

## What this closes (the ~25% residual named by the research gate)

~75% of fluent production in this console is ALREADY spiking-realizable by composing validated pieces: the emergent
LEXICON (EMERGE-30..55 discovered codes = lemmas), the A→W read-out (concept-pool → word, `concept_speak_demo`, 100%
multi-seed), the FrameCQ frame-CONDITIONED serial-order generator on real spikes
(`_phaseB_serial_order_multiframe_derisk`, 6/6 GO, cross-frame control 0.000), and the EMERGE gate-first no-confab moat
(EMERGE-56/57/58). Rung A supplies the truly-missing part — the closed-class GRAMMATICAL FURNITURE, exactly BROCA's
catalogued job (feature-catalog **G.12**, Kandel 6e Ch 55 pp 1382–1384: agrammatism = retained noun selection but LOST
function-word / grammatical-morphology use):

- **(R1) FUNCTION-WORD slots** — the / can / does / not (articles, modals, auxiliaries, negators) — frame furniture, not
  content lemmas selected upstream (Bock & Levelt 1994 grammatical encoding: frames = syntactic structures with slots
  labelled by the grammatical classes of the lemmas that fill them).
- **(R2) MORPHOLOGICAL INFLECTION** — 3sg `-s` (fly bare vs walks 3sg), a per-slot morphological read-out selected by the
  frame's slot tag (Levelt-Roelofs-Meyer 1999 phonological-encoding stage; `emerge_v3` frame-aware, reused from
  EMERGE-57).

## The mechanism (Rung A)

Each EMERGE reply frame is an ordered set of TYPED slots (frame-and-slot grammatical encoding — Levelt/Bock/Dell). The
three frames EMERGE actually emits (from its EMERGE-54/57/58 gate decision):

| frame | reply | slots |
|---|---|---|
| `F_MODAL`  (affirm ability, inherited)     | "the owl can fly"          | `[det:the, SUBJ, FUNC:can, VERB:bare]` |
| `F_INTR`   (intransitive exception, cancel) | "the penguin walks"        | `[det:the, SUBJ, VERB:3sg]` |
| `F_NEGMOD` (negated modal, deny ability)    | "the penguin does not fly" | `[det:the, SUBJ, FUNC:does, FUNC:not, VERB:bare]` |

- **Order LEARNED, produced ON SPIKES.** The per-frame slot-order gradient `prim[frame]` is LEARNED from the frame
  TEMPLATE (the order-teacher), extending the 6/6-GO FrameCQ. The learned primacy gradient becomes GRADED EXTERNAL
  CURRENT into the frame's slot pools on a real `SimulationBridge` (6 driven, non-attractor pools, Izhikevich, dt=1.0);
  the per-pool spiking RATE ranking = the emission order (rate-coded competitive queuing, the validated
  `_phaseB_serial_order_spiking` read-out; Grossberg 1978 / Bullock-Rhodes 2003; catalog G.07/H.19; the biological
  ordinal-template evidence Kornysheva et al. 2019, bioRxiv 383364). A tuning note: the primacy current range was widened
  to 1800→300 pA (below the f-I saturation shoulder) so up to 5 adjacent slots separate cleanly in rate, and the teacher
  is presented 12× per frame so the accumulated primacy gradient is well-separated relative to the choice-layer noise
  (the tie-break-stability the CQ literature flags as the read risk).
- **Every slot spelled by A→W.** Function-word AND content slots are spelled by the SAME A→W read-out (function words are
  just more lemmas in the emergent lexicon — Bock-Levelt frame furniture). The A→W read-out's own spiking validation is
  `concept_speak_demo` (100% multi-seed); it is passed as a pluggable callback (the `neural_serial_order_renderer`
  precedent) so this de-risk validates the frame-and-slot MECHANISM substrate-agnostically for the spelling, on-spikes
  for the order. The inflection surface is picked by the frame slot's morphology tag (`emerge_v3`, frame-aware). **NO
  host f-string assembles the sentence** — the ORDER is the spiking rate ranking, the WORDS are the A→W read-out.
- **Gate-first no-confab moat (load-bearing).** The BRAIN decides answer-vs-abstain BEFORE the producer runs; on ABSTAIN
  the producer is NEVER invoked (0 productions, asserted via a `production_count` counter mirroring EMERGE-56's
  `render_call_count == 0`).

## Results — 6-seed GO (seeds 42/43/44/100/101/102, CPU)

| metric | value |
|---|---|
| **MAIN held-out** order / exact-slot / own-word | **0.993 / 0.995 / 0.998** |
| PERMUTED-slot-order (b1) order / **exact** | 0.269 / **0.000** |
| NO-LEARNING (b4) order | 0.262 |
| CROSS-FRAME (b2) word-match vs own 0.998 | **0.433** |
| FUNCTION-WORD-ABLATION (b3) grammaticality main → ablated | **1.00 → 0.00** |
| MOAT — producer calls on abstains (total) | **0** |

Every anti-cheat collapses:
- **PERMUTED-slot-order** teaches a fixed WRONG order per frame → **0.000 exact-slot match** (right multiset, wrong
  order → no correct sentence).
- **NO-LEARNING** (untrained primacy) → chance order 0.262 (main beats it by +0.73).
- **CROSS-FRAME** (render one frame's content under a DIFFERENT frame's surface) → 0.433 vs the frame's own 0.998: the
  same content is ordered/worded DIFFERENTLY per frame (frame-specific — the seed of syntax; e.g. "the owl can fly" vs
  "the owl does not fly").
- **FUNCTION-WORD-ABLATION** (drop the learned FUNC slots) → grammaticality 1.00→0.00: the function words are
  LEARNED-slot-supplied, not host-inserted (removing the slots yields agrammatic output — precisely the G.12 Broca
  signature).
- **MOAT** holds BY CONSTRUCTION: 0 producer invocations on abstains (the gate short-circuits before the producer); the
  positive control confirms an ANSWER DOES invoke it (the counter is meaningful).

### Sample rendered transcript (on spikes, seed 42)

```
you> can an owl fly?         broca> the owl can fly            [INHERIT (affirm-modal); producer INVOKED]
you> can a penguin fly?      broca> the penguin walks          [CANCEL  (intransitive); producer INVOKED]
you> can a penguin fly?[deny]broca> the penguin does not fly   [DENY    (negated-modal); producer INVOKED]
you> can a zzz fly?          broca> I don't know.              [MOAT    (abstain); producer NOT invoked]
```

The inflection is correct: `fly` stays bare inside `can` / `does not`; `walks` (already-3sg) is NOT double-inflected
(no "walkses"); a base intransitive verb inflects to 3sg in F_INTR (run → runs).

## Verdict

**GO.** EMERGE's fixed reply frames are rendered FLUENTLY on the SPIKING substrate by a frame-and-slot grammatical
encoder with LEARNED function-word + inflection slots, the gate-first no-confab moat intact (0 productions on abstains),
and all four anti-cheats collapsing, 6 seeds. **⇒ the 21M ANN is RETIRED for EMERGE's BOUNDED frame inventory
(ability-affirm / intransitive-exception / negated-modal) — Broca is SIMULATED for these frames on spikes, the first
genuine "simulate Broca, don't bolt on an LLM" step.**

## Honest scope + the exact next mechanism (not a wall)

- **This renders the BOUNDED EMERGE frame inventory, NOT open prose.** The three frames above are exactly what EMERGE-57/58
  needed the 21M for. Rung A retires the ANN for THOSE frames.
- **Open arbitrary generation (R4) is the separate deferred wall** — the from-scratch spiking LM is ~4 orders too small
  (`2026-05-07-Phase-2.3a-NEGATIVE`); 2024–2026 fully-spiking LMs are still off-substrate-backprop-trained + sub-scale.
  Per the master directive this is an UNDISCOVERED MECHANISM sequenced AFTER Rung A, not an endpoint (research-gate
  Move 4 + Rung C: a small learned recurrent spiking sentence-producer, Chang-Dell-Bock dual-path, teacher = the
  structured fact's own sequence).
- **Two honest reuse boundaries in this de-risk** (each a bounded follow-on, NOT a wall): (i) the A→W SPELL is passed as
  a callback (its own spiking validation is `concept_speak_demo`, 100% multi-seed) — wiring the real trained-bridge A→W
  read-out into this producer is the GPU follow-on; (ii) function words are added to the emergent lexicon as closed-class
  lemmas here (frame furniture) — learning/self-organizing the function-word POOLS from experience (per
  `feedback_spiking_structure_must_self_organize`) is a deeper follow-on shared with Rung B (neural frame SELECTION via
  the dlPFC content Control, closing R3 — the host `if`-based frame selection here).

## Wiring the next rung

Rung A's `BrocaProducer` takes an EMERGE gate decision `(gate, subject, verb, polarity | negated_modal)` and produces the
fluent surface on spikes behind the gate-first moat — the SAME decision shape EMERGE-56/58's adapter already emits. The
production wire (EMERGE-60 candidate) is to route `_emerge58`'s gate decision into `BrocaProducer` in place of the 21M
`FTFaculty` for the three frames, keeping the 21M only for anything outside this inventory (R4). NO `sim/` edit needed.

## Provenance
- Machinery reused: `research/runners/song_g1_core.py` (`score_order` / `permuted_order_controls`),
  `_phaseB_serial_order_multiframe_derisk.py` (FrameCQ), `_phaseB_serial_order_spiking_derisk.py` (driven-pool bridge +
  rate read), `neural_serial_order_renderer.py` (the pluggable-A→W precedent), `_emerge57_ra_refinetune_emerge_frames_derisk.py`
  (`emerge_v3` frame-aware inflection), `_emerge56/58` (gate-first moat + the gate-decision shape).
- Catalog: **G.12** Broca (agrammatism = retained noun selection, lost function-word/morphology use), **G.07 / H.19** SMA
  serial order, **G.10** language-as-hierarchical-symbolic-system.
- Literature: Levelt-Roelofs-Meyer 1999 (*BBS*); Bock & Levelt 1994 (grammatical encoding); Dell 1986 (*Psych Rev*);
  Kornysheva et al. 2019 (bioRxiv 383364, neural competitive queuing); Grossberg 1978 / Bullock-Rhodes 2003 (CQ).
- Memory: `project_master_directive_relentless_biological_emergence`, `feedback_spiking_structure_must_self_organize`,
  `feedback_brain_based_only_standard`, `feedback_moat_not_hard_lossy_memory_ok`.
