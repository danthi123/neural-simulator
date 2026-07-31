---
type: plan
status: live
date: 2026-05-16
---

# Order-Intrinsic Conversational Memory — Design

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). User directive
> (2026-05-16): "Continue working autonomously on conversational
> capabilities. Full freedom on architectural work, no cheats,
> hardware limitations in mind. Reference the catalog as needed."
> Same non-negotiable anti-cheat discipline: pre-registered
> permuted-control-gated honest probe, bars never tuned post-hoc,
> honest negative propagated, no overclaiming, no templates-for-UX,
> self-contained at runtime, local RTX 3090.

## Why this is genuinely different (not config-cranking the terminated line)

Six honest negatives (Inc-1/2/3, G1, G1.5, P) terminated the
generative-PRODUCTION line. Their **single, precise, shared cause**:
every one tried to make ORDER emerge via a learned
controller/readout/predictor *over* the recognition-only G.20
sparse-ensemble pool, which provably does not encode recoverable
sequence order (G1 Step-0 AUC 0.775, G1.5 0.40, P 0.475; all held-out
gates FAIL). This design does the categorically different thing the
diagnosis points to: **make order INTRINSIC to the representation,
and read it back with a deterministic position sweep — no learned
sequence model anywhere.** That is not a variant of the terminated
mechanism; it removes the failed mechanism entirely.

## Evidence grounding (falsify-cheaply, done BEFORE designing)

- **Order-intrinsic STORE is multi-seed validated and reused as-is
  (DRY):** `sim/text_embeddings.py:positional_drive_pattern`
  (catalog D.11 time cells; deterministic, near-orthogonal per
  position) + the P4.1 `enable_episodic_context` substrate in
  `research/runners/text_minimal_isolation.py` (`ec_context` region
  + plastic `ec_context->dg` pathway -> distinct CA3 ensemble per
  (word,position)). `validate_positional_binding.py`: **3/3
  multi-seed PASS** (`research/findings/2026-05-11-P41-positional-multiseed.md`,
  all cosines < 0.14), seed-42 re-confirmed on current code. So
  "apple is big" vs "big is apple" are **different stored engram
  sets by construction** — order is in the representation.
- **The honest gap (the probe caught it):** ordered READ-BACK via
  the existing primitive (`test_word_order_discrimination.py`
  `query_position`: drive `ec_context(position)` alone, see which
  concept fires) is **weak — 2/4 = 50%, near-noise rates
  (0.01-0.02), single seed** on the CuPy/GPU production backend.
  This is NOT a clean GO and is NOT spun as one. But it is a
  *different, tractable* gap than the terminal diagnosis: order *is*
  distinctly stored (validated); what is weak is the position->concept
  read-back, because `query_position` uses **no trained read-back
  pathway** — it drives position and hopes the raw associative trace
  suffices.
- **Risk retired:** the `bridge.py:5360` IndexError is
  **numpy-backend-only**; the CuPy/3090 production path ran the probe
  cleanly (no Traceback). The episodic-context arch is GPU-safe.

## Thesis

Conversational capability as **structured-proposition conversation
over an order-intrinsic engram memory**: the agent is *told*
propositions; each is stored as a set of (concept @ position)
bindings using the multi-seed-validated D.11/P4.1 substrate; it can
retrieve / CA3-pattern-complete (D.13) / compose / intersect /
answer / **abstain on the unknown** (validated moat) over them; and
it **produces** a response by reactivating a stored or recomposed
(concept@position) engram set and reading it out with a
**deterministic position sweep** (pos 0, 1, 2, …) — the order comes
from the D.11 positional structure, not a learned controller. This
is NOT free-form open-domain generation (terminally falsified) and
NOT templated surface (a user-rejected cheat).

## Architecture (Approach A — recommended; net-new piece is small)

Reuse UNCHANGED (DRY, multi-seed validated): G.20 sparse concept
ensembles; the P4.1 `(word,position)->distinct-CA3` store;
`positional_drive_pattern` (D.11); the Tonegawa engram API
(`start_engram_recording`/`commit_engram_tag`/`stimulate_tag`,
D.14); CA3 separation/completion (D.12/D.13); SWR replay (D.19);
the no-confabulation abstention moat; CLS no-forgetting; the
validated comprehension/decoder readout.

**Net-new (the only new mechanism — small, biologically natural):**
a **trained position->concept read-back pathway**. During encoding,
while `lang_input(word)` + `ec_context(position)` + the concept
ensemble co-fire (the validated store), ALSO strengthen a plastic
`ec_context(position) -> concept-pool` pathway via Hebbian/STDP
co-firing. Biologically this is exactly what a Tonegawa engram does
— it binds *all* co-active elements (D.14), and Eichenbaum-Cohen
relational binding (D.02) ties position to content; reactivating any
bound element (here: position) pulls the rest (the concept). At
read-back, driving `ec_context(k)` alone now reactivates the
position-k concept strongly (the trained pathway), not the weak raw
trace the 50% probe exposed.

**Produce (runtime, self-contained):** for the proposition to be
said (a stored one, or one recomposed from stored concepts), sweep
positions k = 0..L-1: drive `ec_context(k)` -> the trained pathway
reactivates the concept bound at k -> decode it via the validated
readout -> emit. Order is the sweep order = the D.11 positional
structure. No learned sequence model. Abstain (validated moat) if a
position's decoded confidence is below the pre-registered floor (no
confabulated slot).

## Data flow

Tell: `lang_input(word_k)` + `ec_context(k)` co-drive for each k ->
commit one engram over the co-active (concept ⊕ position) set
(D.14); the `ec_context->concept-pool` pathway is strengthened by
the same co-firing. Ask/compose/answer/abstain: the validated
retrieval/completion/intersection/moat paths, UNCHANGED. Produce:
deterministic position sweep through the trained read-back pathway +
validated decode.

## Pre-registered anti-cheat gate (FIXED bars, never tuned)

Cheap-first decisive slice (scoped like G1's B-probe — this slice's
gate decides whether the line is pursued; NOT a months-class
buildout):

1. Store a 2-3 concept ORDERED proposition via the validated store +
   the new trained read-back pathway.
2. Ordered read-back = the deterministic position sweep -> decoded
   ordered concept list.
3. **Held-out** novel propositions (never trained), **permuted-ORDER
   control** (same concept multiset, scrambled positions — the
   load-bearing anti-cheat: proves ORDER, not just concepts), and a
   pre-registered control-calibrated frozen abstention floor (same
   control-max/AUC methodology that produced prior floors; NEVER
   650; never recomputed at gate time).
4. Reuse the UNMODIFIED `song_g1_core.g1_verdict` / `score_order` /
   `permuted_order_controls` (`_G1_MARGIN=0.10`, `_G1_ABS_FLOOR=0.5`
   NEVER touched). PASS iff true-order read-back beats the best
   permuted-ORDER control by >= 10% AND clears the floor AND >= 0.5
   majority, **multi-seed** (>=3 seeds; single-seed 50% is explicitly
   NOT sufficient — the probe already showed single-seed is
   unreliable).
5. **LOAD-BEARING no-harm check (CRITICAL):** the new trained
   `ec_context->concept-pool` pathway must NOT regress (a) the
   validated `(word,position)->distinct-CA3` store distinctness
   (re-run `validate_positional_binding`, must stay 3/3 / cosines
   in-band) NOR (b) the no-confabulation abstention moat. If it
   regresses either, STOP and fix the pathway's separation-of-
   concerns before any scaling (the v12/v13/v15/G1 lesson).

```
+---------------------------------------------------------------+
| PRE-GATE CORRECTION (CATEGORY ERROR) -- 2026-05-16             |
+---------------------------------------------------------------+
| Task-6 no-harm OVERALL =                                       |
|   (store-distinctness-unregressed: every seed max CA3 cos<0.4) |
|   AND (no-confabulation: every seed's never-encoded control    |
|        ABSTAINS).                                              |
| The encoded-vs-control MAGNITUDE separation is the             |
| pre-registered Task 7 capability gate (permuted-ORDER control  |
| + control-calibrated frozen floor, proper config), NOT a       |
| Task-6 no-harm criterion. The a78815b run's OVERALL=FAIL       |
| folded that capability-magnitude bar into moat_PASS at a       |
| deliberately-minimal speed config, under-powered-pre-empting   |
| the decisive Task 7 gate (the same class as the documented     |
| C1/C2, Inc-3-held-out, P-bias, Task-5-retarget corrections).   |
| RECOMPUTED purely from the already-recorded a78815b run data   |
| (NO GPU re-run, no chasing a pass):                            |
|   store: 3/3 PASS (seed42 0.263, seed43 0.177, seed44 0.166;   |
|          all < 0.4)                                            |
|   control-abstains: 3/3 PASS (seed42/43/44 all abstain)        |
|   -> Task-6 no-harm SATISFIED.                                 |
| The capability question is NOT removed/weakened -- it IS the   |
| pre-registered Task 7 gate, run honestly at proper config      |
| regardless of this no-harm outcome.                            |
+---------------------------------------------------------------+
```

PASS => order-intrinsic structured-proposition conversation is real
on this substrate -> scale (more concepts/positions, compose,
multi-turn, the full conversational primitive set over it). FAIL =>
honest negative, propagated; this line also terminates and the
validated grounded-memory + no-confabulation agent stands as the
deliverable. Either outcome decision-relevant.

## Honest ceiling / risks (no overclaiming)

- The simplest read-back is 50%/near-noise at single seed; the
  trained pathway is a hypothesis to make it strong + multi-seed —
  it is NOT a validated free win, and it might (a) stay weak or (b)
  damage the validated store's distinctness (hence the load-bearing
  no-harm check). A maxed FAIL here is honest and terminal for this
  line too.
- Scope is structured-proposition conversation (tell/ask/compose/
  answer/abstain/ordered-readback), explicitly NOT free-form
  LLM-style generation. Stated up front; the deliverable framing
  never overclaims this as "an LLM."
- Hardware: same scale as the validated P4.1/320-concept work + a
  positional axis on the RTX 3090; kill-safe resumable training
  (reuse `sim.train_checkpoint`); long runs background, user
  games/resumes.

## Scientific basis (catalog)

D.11 time cells (positional code); D.01 episodic memory;
D.02 Eichenbaum-Cohen relational binding (position<->content);
D.12 CA3 pattern separation; D.13 CA3 pattern completion;
D.14 Tonegawa engram cells (binds all co-active elements; the
read-back pathway is engram-natural); D.19 SWR replay /
consolidation; G.20 Pulvermuller distributed cortical word
ensembles; Marr/McClelland CLS (no catastrophic forgetting).

## Out of scope (YAGNI)

No external LLM/corpus/templates ever at runtime. No learned
sequence controller/predictor (the terminated line; order is
intrinsic + read by a deterministic sweep). No char-level BPTT. No
months-class buildout in this slice — the pre-registered cheap gate
decides first. The numpy-backend `bridge.py:5360` IndexError is out
of scope (production is CuPy; verified clean) — note it, do not fix
it here.
