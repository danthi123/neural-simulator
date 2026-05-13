# Sequential composition for concept pools — design note

**Date:** 2026-05-13
**Status:** Design only. Not implemented; queued for after v8 batch.

## The problem

v7 Phase 2 composition test:
- Sequential composition (drive word_a 50 steps then word_b 50 steps): **0/6 PASS**
- Co-firing composition (merged drive): **2/6 PASS**

Sequential composition needs pool_a to remain firing during word_b's
drive window so both targets are active. With weak dynamics
(0.05/0.3/0.8, chosen to fix v2c canon bias amplification), pools
have insufficient self-sustaining recurrent activity. Drive removal
collapses firing within ~25ms.

Trade-off:
- Canon dynamics (0.10/2.0/4.0): NMDA bistability holds firing 100-150ms,
  but at biological scale with 12 pools, off-target pools self-sustain
  with random bias (v2c 0/12 result).
- Weak dynamics: cleaner cross-category isolation (v7 6.7/12 mean),
  but no NMDA persistence → no sequential composition.

## Design options

### Option A — Per-region NMDA tau override (small change)

Add a `concept_pool_nmda_tau_ms` parameter to `build_biological_brain_regions`.
Default = cfg.nmda_tau (~150ms). Set per-region for concept pools to
~50-100ms — long enough for cross-word persistence, short enough to
prevent indefinite accumulation.

Pro: targeted intervention; doesn't change canon for other regions.
Con: NMDA tau is currently a global cfg parameter; would need bridge
modification to support per-region tau.
Estimate: 1-2 days.

### Option B — Explicit working-memory pool (Tier 2.3 pattern)

Use the existing `enable_dlpfc_verb` infrastructure. Wire concept
pools through a dedicated PFC-like region that maintains persistent
activity via NMDA bistability. lang_input → pool → dlpfc → lang_output.

Pro: reuses existing infrastructure (Tier 2.3).
Con: Tier 2.3 was stuck at 34-40%; adds another bottleneck pathway.
Composition would only work for the verb kind.
Estimate: 2-3 days.

### Option C — Inject sustaining current during composition

Phase 2 test could explicitly hold word_a's drive at reduced amplitude
during word_b's window, simulating attentional maintenance. Not a
true architectural fix; eval-side hack.

Pro: easy to test (no architecture change).
Con: doesn't generalize to autonomous composition; eval-only.
Estimate: hours.

### Option D — Add a separate "echo pool" per concept

Each pool has a small (~50 neuron) high-recurrent "echo" satellite
that gets driven by the main pool, has canon dynamics + strong NMDA,
and feeds back to the main pool. Echo pools self-sustain for ~150ms;
main pools stay weak for clean isolation.

Pro: best of both worlds (weak main pool + bistable echo).
Con: doubles pool count; doesn't map to known biology cleanly.
Estimate: 3-4 days.

## Recommendation

**Option A (per-region NMDA tau)** is the cleanest biology-faithful
fix. Cortex NMDA tau IS region-specific in biology (Wang 2002 PFC
~150ms; sensory ~50ms). Setting concept pools to 80-100ms gives a
real bistability window without amplifying random bias.

Implementation order:
1. Add `cp_nmda_tau` per-neuron array (instead of cfg scalar)
2. In RegionPathway / BrainRegion, accept `nmda_tau_ms_override`
3. Bridge initialization applies per-neuron tau from region overrides
4. Test: v9 = v7 + concept_pool_nmda_tau_ms=80
5. Compare Phase 2 sequential PASS (v7 0/6 → v9 ?)

## Decision criteria

- If v8 (concept_to_language_output fix only) gives A→W PASS rate
  >= 4/12, that validates the weight fix; sequential composition
  can be tackled in v9.
- If v8 also shows Phase 1 improvement, the weight fix may indirectly
  help bistability too (via stronger reciprocal feedback) — sequential
  might improve without architecture change.
- If v8 doesn't help, Option A is the next step.

## Update 2026-05-13 evening (post-v9 breakthrough)

V8 result: A→W still 0/12.
V9 result: A→W 12/12 PASS (reciprocal topographic bias was the fix,
not the weight magnitude).

v10 = Option A in flight: --nmda-tau-decay-ms 250.0 (extends from
default 100ms toward Wang 2002 PFC NMDA range 100-300ms).

### Catalog biology grounding (referenced 2026-05-13)

**G.06 PFC working memory — sustained delay-period activity**
(Kandel 6e Ch 34 pp 827–842): "dorsolateral PFC; recurrent excitation
supports persistent firing across delay; modulated by D1." This is
EXACTLY the mechanism v10 NMDA tau extension is targeting — though
v10 applies it generically across all concept pools rather than
specifically to PFC.

**G.08 Working memory in prefrontal cortex — persistent activity
for active maintenance** (Kandel 6e Ch 52 pp 1292–1294): "Maintains
transient, goal-relevant representations across delays (seconds).
DMS-task PFC neurons hold 'what' (object), 'where' (location), and
'what+where' conjunctions during the delay period (Rainer/Asaad/Miller 1998)."

This is the bidirectional binding pattern we want — verb_pool_GO
holds "what" while motor_N processes "where" for the "go north"
phrase. The catalog notes Cluster G sim status: partial — PFC region
exists (60 neurons recurrent) but single-compartment, no DMS-style
delay-period mixed selectivity. Concept pool architecture extends
the partial PFC capability to non-direction concepts.

### v11 fallback if v10 partial

If v10 NMDA tau 250ms gives sequential <4/6, fall back to **Option B
(explicit working-memory pool)** integrating the existing
`enable_dlpfc_verb` infrastructure. The dlpfc_verb region (200
neurons, internal_density 0.15, NMDA bistable) can act as a
persistent holding stage between concept pools and motor output:

  language_input → verb_pool_X → dlpfc_verb → motor_X

This is exactly Tier 2.3's pattern (which got stuck at 34-40% on
phrase composition, but architecturally proven for persistence).

## Update 2026-05-13 late (post-v12/v13/v14)

**v10 (NMDA tau=250ms on ALL pools):** NEGATIVE on isolation. Canon-amplifies-
bias path resurfaces — all pools self-sustain on random structural bias
once NMDA tau is extended. Sequential persistence achieved but Phase 1
W→A collapsed from 11/16 to 4/16.

**v11 (NMDA tau=250ms on verb pools only):** PARTIAL. Verb pools held
for ~150ms (3x baseline), but Phase 1 W→A dropped to 9/16 — verb pools
still over-activated by stray lang_input drive. Composition seq 1/6.

**v12 (verb_pool ↔ dlpfc_verb bidirectional):** NEGATIVE on isolation.
Bidirectional feedback creates a leakage pathway: any pool's stray
activity → dlpfc_verb via the broad concept→dlpfc pathway → back to
verb_pool → all pools mildly active. Phase 1 collapsed to 6/16.

**v13 (per-kind NMDA: verb=canon, motor/noun/adj=weak):** PARTIAL.
verb_pool_X persistence +3x but Phase 1 isolation -5x (other words
activate verb pools due to dlpfc back-feed in canon NMDA bistability).
Sequential composition still ~1/6.

**v14 (orthogonal drive codes, no holding mechanism):** Strong Phase 1
(15/16 single-seed, 12-15/16 multi-seed) + perfect A→W (16/16 unanimous
across seeds 42-45). Confirms orthogonal codes solve the cross-word
overlap interference. **But no architectural change for sequential
composition.** Composition test still ~co-fire-only.

## v15 design — unidirectional verb_pool → dlpfc_verb → motor gating

**Catalog grounding:** G.08 (BG-thalamic gating of PFC) + G.06 (PFC
delay-period activity). The biology: PFC receives content from
sensory/concept areas via FEEDFORWARD pathways and maintains it via
INTERNAL recurrence + bistability. PFC then gates downstream motor
selection via thalamo-cortical-striatal loops. Critically, PFC does
NOT broadcast directly back to all upstream concept areas — that
would erase selectivity. PFC's outputs go through gated pathways
(BG, motor cortex, downstream effectors).

**v15 architecture:**

  language_input → verb_pool_X (existing, plastic)
  verb_pool_X → dlpfc_verb (forward only, plastic) — NEW WIRING
  dlpfc_verb internal recurrence (canon dynamics + NMDA bistability)
  dlpfc_verb → motor_X (forward only, plastic, gated) — NEW WIRING
  NO dlpfc_verb → verb_pool feedback (delete v12 wiring)

**Mechanism:**
1. "go" drive → verb_pool_GO fires → forward propagates to dlpfc_verb
2. dlpfc_verb sustains via internal NMDA bistability (~100-200ms hold)
3. "north" drive 50ms later → motor_N fires from lang_input pathway
4. dlpfc_verb still firing → dlpfc → motor STDP strengthens during
   the co-fire window → "go north" pair learned
5. At inference: "go" sustains dlpfc → primes motor selection bias
6. "north" alone: lang_input → motor_N (dominant) without prior dlpfc
   activation → clean isolation preserved

**Why this should work where v12/v13 failed:**
- Forward-only verb_pool → dlpfc prevents back-leakage to other pools
- dlpfc internal recurrence (200 neurons, density 0.15, canon dynamics)
  provides the bistability v10/v11 tried to add globally
- dlpfc→motor weights LEARNED, so spurious activations don't
  preferentially route to any one motor

**Implementation plan:**
1. Add `enable_dlpfc_verb_unidirectional` flag to `build_biological_brain_regions`
2. When true + enable_dlpfc_verb + enable_verb_pools:
   - Add verb_pool_X → dlpfc_verb pathway per verb (forward, plastic)
   - Add dlpfc_verb → motor_X pathway per direction (forward, plastic,
     gated by `dlpfc_verb_to_motor`)
   - DO NOT add dlpfc → verb_pool feedback
3. Run training: open `verb_pool_to_dlpfc` gate during verb word events;
   open `dlpfc_verb_to_motor` gate during co-firing windows in compose test
4. Test v15a: train + Phase 1 W→A (should preserve v14's 12-15/16)
5. Test v15b: train compose pairs + measure sequential PASS (target ≥ 4/6)

**Estimated cost:** 1-2 days code, 8-12 hr training + eval for one seed.

**Risk:** if dlpfc internal recurrence is too weak (lower density, weak
weights), no hold. If too strong, dlpfc self-sustains indefinitely and
loses selectivity. Tier 2.3 canon settings (density 0.15, weight 3.0/4.0,
NMDA on) are the starting point.

## Notes

This design note is intentionally NOT a plan — it's an option
inventory for the next architectural decision. v15 (unidirectional
verb→dlpfc→motor) is the most promising path forward; the v12 failure
was due to bidirectional feedback breaking isolation, not the basic
dlpfc-as-working-memory concept.
