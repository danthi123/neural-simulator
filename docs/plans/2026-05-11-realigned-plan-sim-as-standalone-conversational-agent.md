# Realigned plan: sim as standalone conversational agent (catalog-grounded v3)

**Date:** 2026-05-11 (post-user-checkin v3)
**Status:** ACTIVE — primary path, supersedes v2 and earlier Path 3 framing
**Author:** autonomous arc, after three user clarifications (no LLM,
biology-first workflow, consult catalog)

## Goal

Biology-grounded spiking neural simulator as a **standalone**
conversational agent. **No external LLM, ever.** Fully local. Eventually
capable of multi-word, multi-turn, semantically meaningful conversation.

## Methodology — biology-first, catalog-grounded

Per `.claude/skills/continual-autonomous-work/SKILL.md` Rule 8:

1. State capability
2. Test against existing architecture
3. **Consult the research catalog FIRST:**
   - `E:/Documents/Projects/sim-catalog/references/feature-catalog.md`
   - `E:/Documents/Projects/sim-catalog/references/biology-buildout-roadmap.md`
   - `E:/Documents/Projects/sim-catalog/references/textbooks/` (Kandel 6e + specialty PDFs)
4. Implement the catalog-cited mechanism
5. Test, validate, repeat

## What the previous "realigned plan v2" got wrong

v2 proposed a "Step 0 — add semantic_hub region" invented from
Patterson 2007 hub-and-spoke memory. The catalog already had the
better framing:

- **G.11 dual-stream language model** (Kandel Ch 55 pp 1380–1387) —
  ventral stream is the missing semantic substrate
- **G.13 Wernicke's area** (Kandel Ch 55 pp 1384–1385) — explicitly
  lists "semantic memory store" as a prerequisite
- **D.01 episodic memory** (Kandel Ch 52 pp 1296–1302) — concept
  binding via medial temporal lobe + association cortices
- **D.02 relational binding (Eichenbaum–Cohen)** — items-in-context

All four entries are `Sim status: missing`. The buildout roadmap
sequenced the prerequisites:

- **T1.A — Hippocampal trisynaptic loop** (Month 1): DG → CA3 → CA1
  proper, with DG pattern-separation + CA3 attractor completion
- **T1.B — SWR-driven sequential replay** (Month 2): time-compressed
  consolidation sequences
- **T1.C — Engram-tagging API** (Month 2, parallel): tag + stimulate
  active ensembles by name

This is the **catalog-grounded** sequence the user's "concepts ≠
motor pools" insight pointed at all along. Defer to the roadmap.

## Capabilities table (catalog-grounded, where we are, where we need to go)

| Capability | Catalog entry | Sim status | Distance |
|---|---|---|---|
| Hippocampus proper (DG/CA3/CA1 trisynaptic) | D.03 (Kandel Ch 52) | partial (stub CA3) | Tier 1 / month 1 |
| Pattern separation (DG sparsifies overlaps) | D.12 | missing | gated on T1.A |
| Pattern completion (CA3 attractor) | D.13 | missing | gated on T1.A |
| Engram tagging (Tonegawa-style ensemble ID) | D.14 + T1.C | missing | 2-3 days bridge code |
| SWR sequential replay (time-compressed) | D.19 + T1.B | partial (gates exist, no compression) | Tier 1 / month 2 |
| Episodic encoder | D.01 | missing | gated on T1.A+B+C |
| Relational binding (items-in-context) | D.02 | missing | gated on D.01 |
| Ventral semantic stream | G.11 | missing | gated on D.01 |
| Wernicke's area (auditory→semantic) | G.13 | missing | gated on G.11 |
| Broca's area (speech production + syntax) | G.12 | missing | gated on G.13 |
| Direction-word binding (motor pool) | (existing) | ✅ Tier 2.1 5/6 W→A | done — but only for action-words |
| 12-word direction vocab | (existing) | ✅ 3/3 GO scaled arch | done |
| 16-word direction vocab | (existing) | partial (1/1 smoke) | small (multi-seed pending) |
| 64+ word vocab | (existing) | NEGATIVE 2026-05-11 | architecture-limited; gated on D.01+G.* |
| Compositional 2-word phrases | (Tier 2.3) | partial (39.8% mean) | architecture-limited; gated on D.01+G.12 |
| Sentence-level understanding | gated chain | missing | gated on T1.A→T1.C→D.01→G.11→G.13 |
| Reasoning | gated chain | missing | gated on all of above + G.* expansion |
| Hundreds-of-words conversation | gated chain | missing | gated on all of above |

## Step plan (deferring to roadmap T1.A → T1.B + T1.C → D.01 → G.*)

### Step P1 — Hippocampal trisynaptic loop (roadmap T1.A, ~Month 1)

**Catalog:** D.03 (Kandel Ch 52 pp 1310-ish), D.12 pattern separation,
D.13 pattern completion. Roadmap T1.A spec.

**Build:** 3 new BrainRegions — DG, CA3, CA1 — wired as
`EC → DG → CA3 → CA1 → output` with `EC → CA1` direct path and
`CA3 → CA3` recurrent attractor. Existing primitives only (no new
GPU code).

**Validate:**
- Pattern separation: present 2 similar place inputs → verify DG
  outputs decorrelate (cosine drop)
- Pattern completion: train CA3 on (cue, context) pair → partial cue
  reactivates full pattern
- Place-field stability across trials in CA1 readout

**Effort:** 1-2 weeks for working circuit; 4-6 weeks to full validation.

### Step P2 — Engram-tagging API (roadmap T1.C, ~Month 2 parallel)

**Catalog:** D.14 engram cells (Tonegawa et al). Roadmap T1.C spec.

**Build:** ~50 LOC bridge addition:
- `bridge.tag_active_ensemble(name, threshold_hz, window_ms)` —
  snapshot which neurons fired above threshold during a window
- `bridge.stimulate_tag(name, drive_pA)` — drive only tagged neurons
- Persist tags across simulation steps

**Validate:**
- Train on "context A → reward"; tag the active ensemble
- Place agent in context B, drive tagged neurons → verify
  reward-conditioned behavior emerges (Liu 2012 inception-of-fear
  paradigm)

**Effort:** 2-3 days for bridge code; 1-2 weeks for first experiment.

**This is the primitive that lets concepts live as tagged ensembles
independent of motor pools.** "Apple" becomes the name of a tagged
ensemble in CA3, not a motor target. Recall = stimulate the tag.

### Step P2 update — engram-tagging API SHIPPED 2026-05-11

`bridge.start_engram_recording / commit_engram_tag / stimulate_tag`
plus persistence through save/load. 12 unit tests pass.

See `sim/bridge.py` (200 LOC added) and
`tests/test_engram_tagging.py` (14 tests including 2 skip-pending-
integration). Commits: `29513ac` (API), `a3acb9c` (persistence).

### Step P1 update — multi-seed mixed, two-concept test running

Catalog D.12 + D.13 validated single-seed (cosine 0.748). Multi-seed
(42, 43, 44):
- D.12 (DG separation): 3/3 PASS, robust
- D.13 (CA3 completion at cos > 0.7 absolute): 1/3 (42=0.748,
  43=0.676, 44=0.679) — autoassociator IS working, just below my
  arbitrary > 0.7 cutoff.

Running `validate_two_concept_discrimination.py` (3 seeds) — the
biology-faithful RELATIVE criterion (same-concept cos >> cross-
concept cos). Results determine whether P1 is good enough for
the conversational-sim use case.

### Step P3 update — DESIGN SHIPPED 2026-05-11

`docs/plans/2026-05-11-P3-swr-replay-design.md` (commit `5012a9d`).
Two-stage plan: P3.1 concept replay (cheap, uses P2 tags) then P3.2
sequence replay (deferred until P4 produces sequences). Gates on
P1 final pass.

### Step P4 update — DESIGN SHIPPED 2026-05-11

`docs/plans/2026-05-11-P4-episodic-encoder-design.md` (commit
`1de75f0`). Adds `ec_context` region with positional embedding;
trains item-in-context bindings in CA3 so (apple, pos_1) is distinct
from (apple, pos_3). Foundation for word-order-dependent meaning.

### Step P5 update — DESIGN SHIPPED 2026-05-11

`docs/plans/2026-05-11-P5-ventral-semantic-stream-design.md` (commit
`8cf7e14`). Adds `semantic_cortex` + `wernicke` regions; replaces
the invented `semantic_hub` from v2 with the catalog-grounded
ventral language stream (Hickok & Poeppel, Kandel Ch 55).

### Step P6 update — DESIGN SHIPPED 2026-05-11

`docs/plans/2026-05-11-P6-brocas-grammar-design.md` (commit
`88b1124`). Adds `broca` + `motor_speech` regions; replaces the
failed Tier 2.3 PFC verb pool with the catalog-grounded Broca's
design. Solves composition at the syntactic level, not via motor
gain modulation.

### Step P3 (original heading kept for backref) — SWR sequential replay augment (roadmap T1.B, ~Month 2)

**Catalog:** D.19 SWRs, N.04 ripple-coupled replay. Roadmap T1.B.

**Build:** augment existing sleep-replay infrastructure to generate
time-compressed (10-20×) place-cell sequences during NREM windows,
phase-locked to slow oscillation surrogate and nested by spindle
envelopes.

**Validate:**
- Ripple events show 10-20× temporal compression of waking sequences
- Downstream weight changes during sleep vs no-sleep on a memory task
- Replicate "blocking SWRs impairs spatial learning" (Girardeau 2009)

**Effort:** 2-3 weeks (composes onto T1.A).

### Step P4 — Episodic encoder + relational binding (D.01 + D.02)

**Catalog:** D.01 (Kandel Ch 52 pp 1296-1302), D.02 Eichenbaum-Cohen.

**Build:** wire item-stream (perirhinal analog) + context-stream
(parahippocampal analog) → hippocampus CA1. Store items-in-context;
support transitive inference via overlapping events.

**This unlocks abstract concepts beyond motor-pool grounding.**

### Step P5 — Ventral semantic stream + Wernicke's (G.11 + G.13)

**Catalog:** G.11 (Kandel Ch 55 pp 1380-1387 Hickok & Poeppel), G.13
(Kandel Ch 55 pp 1384-1385).

**Build:** semantic cortex region (Wernicke's analog) receiving
language_input and routing to/from hippocampal engram tags.
Bidirectional: word → concept (comprehension), concept → word
(recall).

### Step P6 — Broca's area + compositional syntax (G.12)

**Catalog:** G.12 (Kandel Ch 55 pp 1382-1384). Replaces the failed
Tier 2.3 PFC verb pool with the Broca's-grounded design.

**Build:** left posterior inferior frontal gyrus analog supporting
grammatical processing + speech production. Two-word phrase
composition validation.

### Steps P7+ — Sentence-level + reasoning + conversation

Long-horizon. Each gated on prior steps. Specific catalog entries
TBD as we approach.

## Tier 3 (long horizon, decide explicitly per roadmap)

- **T3.A compartmental neurons** (apical-basal dendrites) — was
  "Step 4 dendritic learning rewrite" in v2. Per roadmap: decide
  explicitly when limitations bite. Don't pre-commit.
- **T3.C muscle output / Hill-type model** — gateway to embodied
  speech production. Big architectural decision.

## What about the in-vivo binding fix experiment I was about to run?

**Off-axis until P1/P2 land.** "Fix in-vivo binding to motor pools"
tries to scale a fundamentally-limited architecture. Real biology
binds new concepts as **tagged hippocampal ensembles** (D.14 + T1.C),
which then consolidate via SWR replay (D.19 + T1.B) to distributed
cortical representations (D.01 + G.11). All currently `Sim status:
missing`.

The n_events sweep result (0/4, 1/4, ??/4 at 200/400/800) is still a
useful baseline — it documents the motor-pool-bound architecture's
ceiling. Let it finish for the record. Then pivot to T1.A.

## What gets carried forward from prior work

KEEP:
- BridgeLineage (continuous learning infrastructure)
- BridgeMemory API (works as-is for direction-word vocab; will extend
  to engram-tagged concepts once T1.C lands)
- Synapse tiering + auto-growth + NumPy backend (scaling infra)
- Dashboard chat panel (UI works against any future API)
- Phase 1.3 consolidation work (partial T1.B foundation — needs the
  compression augment)
- continual-autonomous-work skill (meta-tooling)

KEEP-AS-SECONDARY (no active work):
- OllamaLLM adapter + SIM_LLM_BACKEND env var (vector-DB-for-LLMs
  path, not primary)

DEPRECATED:
- Phase 3.3 design (real LLM integration)
- "Realigned plan v2" Step 0 semantic_hub invention (replaced by
  catalog-grounded P1-P5)
- In-vivo binding fix runner (off-axis until P1/P2)

## Local-only commitment

Every step runs on local hardware (RTX 3090 or CPU). No cloud, no
external LLM, ever.

## Realistic timeline

Following the roadmap's 12-month sequencing:

| Roadmap month | Phase here | Capability unlocked |
|---|---|---|
| 1 | P1 | Trisynaptic loop working |
| 2 | P2 + P3 | Engram tags + SWR compression |
| 3-4 | P4 | Episodic + relational binding |
| 5-6 | P5 | Semantic memory store |
| 7-9 | P6 | Compositional 2-word phrases |
| 10-12 | P7+ | Sentence-level approaches |
| 12+ | T3 decisions | Compartmental neurons, muscle, etc. |

Realistic horizon for "demonstrable conversational sim with
composition + abstract concepts": **6-12+ months autonomous pace.**
Each phase has explicit catalog-cited validation criteria.

## Resume directive

The n_events sweep finishes on its own (let the 800-level wrap for
the record). Next concrete action after the sweep: start **P1 —
hippocampal trisynaptic loop** per roadmap T1.A. The build uses
existing `BrainRegion` + `RegionPathway` primitives; no new GPU
code. Catalog entries D.03 + D.12 + D.13 are the spec.

Pause for user confirmation before starting P1, since this is the
first month-scale step of the realigned plan.
