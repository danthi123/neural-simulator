---
type: finding
status: scoping
date: 2026-08-27
mechanism: onebrain-completeness-audit
lane: onebrain-integration
---

# One-brain COMPLETENESS audit — is the live integration the FULL one-brain, or are organs still unwired? What is left, WHY, and what to work on alongside

Status: SCOPING / INVENTORY (a read-only classification of the live `/api/brain-chat` turn against the merge
framework + the production-integration ledger; NO code changed, NO default flipped, NO `sim/`/`webapp/` edit).
Answers the owner's question: the one-brain integration is NOT the full one-brain — it is ONE frozen learned
cross-region synapse (behind a default-OFF flag) sitting inside a brain whose every other organ→organ
influence is still HOST-ORCHESTRATED. This doc enumerates the organs, classifies each interaction, states the
honest fraction, gives the concrete blocker per class, and ranks the next 5-8 edges + the single framework
investment that makes them cheap.

## 1. THE HONEST HEADLINE — one learned cross-synapse vs an all-host connectome

Of all organ→organ interaction in the live default brain, **learned cross-region synapse = exactly ONE
directed edge** (d6 working-memory referent → comprehension role competition), wired by PART 1
(`onebrain_xedge_production.py`, flag `BRAIN_ONEBRAIN_XEDGE`) and **FROZEN** (grown once, plasticity gated to
0). It is behind a **DEFAULT-OFF** flag, so in the config the owner actually gets, the count of live learned
cross-organ synapses is **ZERO**; with the flag ON it is **ONE** (frozen, and — the PART 1 caveat — its drive is
SUB-DECISION: it shifts the comprehension margin lesion-attributably by 0.009-0.044 <!--derived--> but does not
yet cross the `comprehended` boundary, so it is content-neutral at the wording level). Those figures restate the
PART 1 GO finding `2026-08-27-onebrain-xedge-production-frozen-GO.md` (cited in the Files section) — not
measurements this audit produced.

Everything else — every other cross-organ influence in the pipeline — flows through **host orchestration**: the
turn handler calls each organ's `get_organ()`, reads its scalar/verdict into a host Python variable, and passes
that to the next stage. Even the two default-ON merge POOLS add **ZERO cross-synapse**: they are byte-identical
CO-RESIDENCY (many organs, one `cp_membrane_potential_v`, no synapse spanning two slices) — a substrate-
consolidation claim, not an interaction claim (their own docstrings: "NO cross-organ synapse is added on this
rung").

**Fraction:** ~17 live faculty organs + the recall composer / parser / dlPFC planner / moat / mouth glue. A
fully-interacting one-brain is a directed graph of dozens of such edges. We have **1** in production (default-
OFF, frozen, sub-decision) and **3** de-risked on the research framework (R1, R3-v3, R4). So the learned-cross-
synapse share of the eventual connectome is **single-digit-percent at most, and 0% by default today**. **The
full one-brain is a large remaining arc, not near-done** — the integration milestone that landed is the FIRST
brick of the wall, proven load-bearing, not the wall.

## 2. INVENTORY — every live organ, classified by how it interacts with OTHER organs

Classes: **(a)** learned cross-region synapse IN PRODUCTION · **(b)** learned cross-edge GO on the FRAMEWORK,
not in production · **(c)** co-resident in a merge pool with ZERO cross-synapse · **(d)** STANDALONE bridge,
interacts only via host orchestration · **(e)** not migrated into the framework at all (Group B/C seam).

| Organ (live getter, `webapp/server.py`) | Framework status | Interaction class | Note |
|---|---|---|---|
| d6 multi-referent WM (`_get_multiref_organ:3303`) | GROUP_A migrated | **(a)** when flag ON, else **(d)** | per-session; takes the process xedge pool's d6 slice as `shared=` only if `xedge_enabled()` (default OFF) |
| comprehension (`_get_comprehension_organ:2970`) | GROUP_A migrated | **(a)** when flag ON, else **(d)** | process singleton; the xedge cross-edge's POST target (`sel_agent`/`sel_patient`) |
| surprise (`_get_surprise_organ:2976`) | pool-organ (SURPRISE) | **(c)** | pool #1 co-resident (`merge_enabled()` default ON), zero cross-synapse |
| world-model (`_get_worldmodel_organ:2988`) | pool-organ (WORLDMODEL) | **(c)** | pool #1 co-resident, zero cross-synapse |
| recall composer + parser | in pool #1 | **(c)** | co-resident on pool #1's shared bridge (`onebrain_merge_production.py`, composer-in-pool#1 slice), zero cross-synapse |
| metacog (`_get_metacog_organ:2982`) | pool-organ (METACOG) | **(c)** | pool #2 co-resident (framework `merge_organs([METACOG,PRAGMATIC])`, `merge2_enabled()` default ON), zero cross-synapse |
| pragmatic (`_get_pragmatic_organ:3007`) | pool-organ (PRAGMATIC) | **(c)** | pool #2 co-resident, zero cross-synapse |
| self_schema (`_get_self_schema_organ:3013`) | GROUP_A migrated | **(d)** | standalone bridge (`shared=None`); R4's cross-edge SOURCE, not wired to production |
| source_provenance (`_get_source_provenance_organ:3000`) | GROUP_A migrated | **(d)** | standalone; R4's cross-edge TARGET; prod wrapper does not plumb `shared=` |
| curiosity (`_get_curiosity_organ:2994`) | GROUP_A migrated | **(d)** | standalone; organ-read now closed on a pool but not routed onto one in prod |
| causal what-if (`_get_causal_organ:3348`) | GROUP_A migrated | **(d)** | per-session standalone |
| prospective memory (`_get_pmem_organ:3364`) | GROUP_A migrated | **(d)** | per-session standalone (largest organ, ~1720 neurons) |
| activity-silent WM (`_get_silent_wm_organ:3328`) | not in REGISTRY | **(d)** | per-session standalone |
| affect / affect-coloring (`_get_affect_organ:2964`) | — | **(d)** | standalone; a STRUCTURAL pool exclusion (region-name collision + global OU/neuromod cfg) |
| affective ToM (Group-B `affective_tom`) | GROUP_A_DEFERRED | **(e)** | needs an OU + neuromodulator-subsystem seam; not a shared-pool slice |
| d5 episodic (`_get_episodic_organ_existing:3284`) | GROUP_A_DEFERRED | **(e)** | heavy own-pool (~2000-neuron CA3 + apical dAP + BTSP + slow-NMDA); Group-C seam |
| d3 discourse register | GROUP_A_DEFERRED | **(e)** | multi-bridge (4 FS-WTA discretizers + host rate-RNN); needs a multi-bridge seam |
| reconsolidation (`_get_reconsolidation_organ:3341`) | GROUP_A_DEFERRED | **(e)** | owns no circuit; reuses surprise's D2 slice + rewrites the composer store |
| non-contradiction (`_get_noncontradiction_organ:3357`) | GROUP_A_DEFERRED (`b3`) | **(e)** | STATELESS; rides the live composer's polarity recall; nothing to co-locate |
| repair | GROUP_A_DEFERRED | **(e)** | no class; functions composing the comprehension organ; migrates when it does |

**Counts.** (a) production learned cross-synapse: **1 edge / 2 organs** (d6→comprehension, + da_credit as the
pool's third member), and only when `BRAIN_ONEBRAIN_XEDGE` is set (default OFF). (b) framework-GO edges NOT in
production: **3** — R1 (d6-WM→comprehension, two-factor Hebbian), R3-v3 (d6-WM+DA-credit→comprehension,
functional drive — PART 1's frozen edge derives from this), R4 (self_schema→source_provenance); R1 and R3-v3 are
the SAME organ pair under two rules. (c) co-resident, zero cross-synapse: **2 pools / 6 slices** — pool #1
{surprise, world-model, composer, parser}, pool #2 {metacog, pragmatic}. (d) standalone host-orchestrated:
**~9 organs** (self_schema, source_provenance, curiosity, causal, pmem, silent-WM, affect, + d6/comprehension in
the default-OFF config). (e) not migrated (Group B/C): **6 keys** — affective_tom, d5_episodic,
d3_discourse_event_register, reconsolidation, b3_noncontradiction, repair.

The 7 GROUP_A organs (causal_whatif, comprehension, self_schema, source_provenance, curiosity,
prospective_memory, d6_multiref_wm) + the 4 pool organs (SURPRISE/WORLDMODEL/METACOG/PRAGMATIC) are all
`supports_shared=True` and ORGAN-READ CLOSED 6-seed GO on a pool — i.e. migration-READY — but only 2 of the 7
GROUP_A organs (d6, comprehension) are on a production pool at all, and only via the default-OFF xedge flag.

## 3. WHY each unwired interaction is NOT wired (the concrete blocker per class)

- **(a) Only ONE edge exists because cross-edges are NON-DECLARATIVE.** `OrganDescriptor` has NO `cross_edges`
  field and `merge_organs` has NO `cross_edges=` param (grep of the framework: zero matches — the DESIGN §2
  `cross_edges: [(src,dst,w0,rule,gate)]` is an ASPIRATION). Each edge is a bespoke ~37-46 KB research runner
  that hand-wires the synapse via `inject_explicit_wiring` + a whitelisted plasticity gate. Adding an interaction
  is a from-scratch build, not a registry row — so we have one, not ten.
- **(b) The framework-GO edges (R1/R3-v3/R4) are not in production** for four stacked reasons the SCOPING doc
  established: (i) no PRODUCTION pool co-locates the source+target organs (prod keeps them on separate bridges,
  `shared=None`), so no substrate a synapse could span; (ii) the cross-edge wiring lives in a research runner,
  in no production path; (iii) LIFECYCLE mismatch — d6 is PER-SESSION, comprehension is PROCESS-SHARED; one pool
  cannot be both (PART 1 resolves this narrowly: a process pool + d6 keeps its per-session codebook, taking only
  the shared spiking slice); (iv) NO live credit signal — the edge is grown from a host teacher schedule the
  live turn does not have, so the first wire-in must ship a PRE-GROWN FROZEN edge (exactly what PART 1 does).
  R4 additionally: the production `SourceProvenanceHonestyMonitor` does not forward `shared=` (a production
  edit), which is why R3-v3, not R4, was the first wire-in.
- **(c) The merge pools carry zero cross-synapse BY DESIGN.** Pools #1/#2 were built to prove SUBSTRATE
  consolidation (one `cp_membrane_potential_v`) with byte-identity + answer-preservation; a genuine cross-synapse
  was explicitly the "named next step", never added on those rungs. Co-residency is a prerequisite for wiring,
  not wiring.
- **(d) The standalone migrated organs are not routed onto a pool.** curiosity/source_provenance/self_schema/
  causal/pmem are all `supports_shared=True` and organ-read GO, but every production `get_organ` still builds
  `shared=None`. Wiring them needs a production pool that instantiates them + the `shared=` attach (the
  `metacog_production_organ` template) — plus, for a genuine edge, a declared cross-synapse (blocker a).
- **(e) The Group-B/C organs are not migrated at all**, each waiting on a specific engine seam: affective_tom
  needs an OU-process + neuromodulator-subsystem seam; d5_episodic is a heavy own-pool (CA3 + apical dAP + BTSP +
  slow-NMDA reverberation) needing a Group-C own-pool seam; d3_discourse is multi-bridge (four FS-WTA
  discretizers + a host rate-RNN transition) needing a multi-bridge seam; reconsolidation/b3/repair own no
  circuit and migrate WHEN their host organ (surprise / composer / comprehension) does.
- **THE OPEN BIOLOGICAL QUESTION underneath all of it.** The real brain is NOT all-to-all — WHICH connectivity is
  even correct is unresolved, so "wire everything to everything" is wrong, not just expensive. The
  biologically-motivated, directed next edges (each a real cortico-cortical / neuromodulatory projection) are:
  **affect→tone/mouth** (limbic coloring of speech prosody), **surprise→world-model / surprise→episodic
  encoding** (VTA-hippocampal novelty-gated plasticity, Lisman-Grace), **self_schema→source_provenance** (R4,
  authorship→"is this my own thought"), **WM→comprehension** (R1, dlPFC top-down bias on role assignment —
  DONE as PART 1), and **comprehension/self_schema→metacog speech-confidence** (DESIGN R4). These are the
  candidate connectome, not an exhaustive mesh.

## 4. PRIORITIZED ROADMAP to the full one-brain (the "work alongside" answer)

Ranked by biological motivation × existing de-risk × production-readiness. Independent lanes are flagged so they
can run alongside each other.

1. **Strengthen + default-flip the ONE existing edge** (d6→comprehension, `BRAIN_ONEBRAIN_XEDGE`). Highest
   readiness — already wired + 6-seed GO + frozen. Blocker: the drive is SUB-DECISION (does not cross the
   `comprehended` boundary) + a declared positional-proxy residual (no semantic referent→pool binding). Work:
   route through R3-v3's balanced `amb_read` / strengthen the weight so a held referent visibly flips a pronoun;
   reconcile the d6 lifecycle; then an owner-gated default-ON flip. No framework work. (Production lane.)
2. **Make cross-edges DECLARATIVE — the single highest-leverage framework investment.** Add a `cross_edges`
   field to `OrganDescriptor` + a `cross_edges=` param to `merge_organs` (the DESIGN §2 aspiration; zero today),
   backed by the `_apply_gain0_freeze` inversion that already exists (whitelist the declared edge as the sole
   gain-1 plastic synapse). This turns every future edge from a bespoke ~40 KB runner into a REGISTRY ROW under
   the F-gate. It makes edges 3-8 cheap instead of bespoke and carries zero production risk. (Framework lane —
   fully independent, start now.)
3. **R4 self_schema→source_provenance into production** ("is this my own thought" self-monitoring — directly on
   the honesty-boundary deliverable). GO on the framework. Needs: a production pool co-locating the two organs +
   extending the production `SourceProvenanceHonestyMonitor` to forward `shared=`. High biological + honesty
   value, GO de-risk, medium readiness (one wrapper edit). (Production lane; cheaper AFTER step 2.)
4. **R2 three-factor (neuromod-gated) upgrade of the d6→comprehension edge.** Establishes the neuromod-gated-
   plasticity backbone every later edge reuses AND is the path to live-learning (PART 2 — the frozen edge learns
   from raw dialogue). Needs the neuromodulator-subsystem pool seam (curiosity's deferral, now partly resolved by
   the per-neuron OU stream). (Framework + production.)
5. **R3 (DESIGN) surprise→episodic/provenance ENCODING gate** — the canonical novelty-gated-plasticity edge
   (Lisman-Grace VTA-hippocampal loop). Ties the pool-#1 surprise organ to the encoding organs. Needs folding
   surprise into the shared pool + reconciling its Hebbian-on config via per-region gating. (Depends on step 6.)
6. **Migrate d5_episodic (Group-C own-pool seam)** so the memory hub can accept cross-edges (R3/R5 target it).
   Heavy but independent: a ~2000-neuron CA3 + apical dAP + BTSP + slow-NMDA seam. (Migration lane — independent
   CPU de-risk, run alongside.)
7. **affect→tone/mouth edge** — highest CONVERSATIONAL payoff (affect drives HOW it speaks, memory #84), but
   blocked twice: affective_tom needs the OU/neuromod seam (not migrated) AND no mouth/tone organ is registered
   at all (a new organ must be built first). High value, low readiness — sequence after the neuromod seam
   (step 4) lands. (Migration + new-organ lane.)
8. **Reciprocal / multi-edge loops** (DESIGN §6) — once several directed edges hold their F-gate, allow
   comprehension→WM feedback etc. under F3 stability + the lesion-recovers-migration invariant. This is the
   genuinely-interacting one-brain end state; it is the LAST step, not an early one.

**Parallelizable alongside** (disjoint lanes, no serial dependency): step 2 (declarative cross_edges, pure
framework) · step 6 + the affective_tom OU/neuromod seam + the d3_discourse multi-bridge seam (three independent
Group-B/C migration de-risks) · step 1 (strengthen the existing edge, production) · step 3's provenance-wrapper
`shared=` edit. The single highest-leverage item remains **step 2** — every edge after it is a registry row, not
a bespoke runner.

## Files (this audit — read-only, no code changed)

Live path traced: `webapp/server.py` — `_build_chat_brain:3621`, `brain_reply:4106`, the organ getters
`_get_affect_organ:2964`/`_get_comprehension_organ:2970`/`_get_surprise_organ:2976`/`_get_metacog_organ:2982`/
`_get_worldmodel_organ:2988`/`_get_curiosity_organ:2994`/`_get_source_provenance_organ:3000`/
`_get_pragmatic_organ:3007`/`_get_self_schema_organ:3013`/`_get_multiref_organ:3303` (+ the xedge attach
`:3310-3323`)/`_get_silent_wm_organ:3328`/`_get_reconsolidation_organ:3341`/`_get_causal_organ:3348`/
`_get_noncontradiction_organ:3357`/`_get_pmem_organ:3364`/`_get_episodic_organ_existing:3284`;
`webapp/brain_reply.py`.

Framework + edges: `research/runners/onebrain_merge_framework.py` (`OrganDescriptor:49` — no `cross_edges`
field; `merge_organs:479` — no `cross_edges=` param; `GROUP_A:1951`, `GROUP_A_DEFERRED:2016`, `REGISTRY:2067`);
`research/runners/onebrain_xedge_production.py` (PART 1 — `xedge_enabled:57` default OFF, `XedgeProductionPool`,
frozen d6→comprehension); `_onebrain_integration_r1_wm_comprehension.py`,
`_onebrain_integration_r3v3_functional_drive.py`, `_onebrain_integration_r4_selfschema_provenance.py`.

Production pools: `research/runners/onebrain_merge_production.py` (`_MERGE_DEFAULT_ON`, pool #1 —
surprise+world-model+composer+parser, zero cross-synapse); `research/runners/onebrain_merge_production2.py`
(`_MERGE2_DEFAULT_ON`, pool #2 — metacog+pragmatic via the framework, zero cross-synapse).

Ledger + scoping/design: `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (rows `one-brain-substrate:110`,
`onebrain-merge-organs:134`); `research/findings/2026-08-27-onebrain-production-integration-SCOPING.md`;
`research/findings/2026-08-27-onebrain-integration-phase-DESIGN.md`.

Evidentiary basis for "de-risked" (the R-edge GOs this audit's premise rests on):
`research/findings/raw/_onebrain_integration_r1_wm_comprehension_6seed.json`,
`research/findings/raw/_onebrain_integration_r3v3_functional_drive_6seed.json`,
`research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed.json`;
`research/findings/2026-08-27-onebrain-xedge-production-frozen-GO.md` (PART 1 wire-in),
`research/findings/2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md`.

- `research/findings/2026-08-27-onebrain-completeness-audit.md` — this doc.
