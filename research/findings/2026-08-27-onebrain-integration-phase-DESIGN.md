---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# One-brain INTEGRATION phase — DESIGN: the FUNCTIONAL gate + LEARNED cross-region edges that go beyond migration byte-identity

Status: DESIGN (a plan + a 6/6 feasibility SMOKE; no production GO claimed). This is the phase AFTER the
migration-safety gate. The merge framework (`research/runners/onebrain_merge_framework.py` +
`onebrain_merge_verify.py`) has closed ORGAN-READ byte-identity for the Group-A organs on ONE shared
`SimulationBridge` — but byte-identity-in-ISOLATION is the MIGRATION gate (organs co-resident, ZERO
cross-region synapses), and it is ANTAGONISTIC to the actual one-brain GOAL: organs that INTERACT via
cross-region synapses so one faculty's activity shapes another's. This doc defines (1) the FUNCTIONAL gate
that replaces byte-identity once an edge is added, (2) how those cross-edges FORM by LEARNING (the emergence
bar), (3) the FIRST interaction to build, (4) the ordered roadmap + the production flip, (5) the risks + the
honest boundary. It is grounded in the record (§0) — this is not a new invention; the project already has a
cross-region-interaction GO to build ON.

The smoke (`research/runners/_onebrain_integration_crossedge_smoke.py`, artifact
`research/findings/raw/_onebrain_integration_crossedge_smoke_6seed.json`) proves the crux on the REAL
substrate, 6/6 seeds, numpy CPU. NO `sim/` edit.

## 0. PRIOR ART — what the record already establishes (build ON this, do not re-derive)

The owner's OPERATIONAL DEFINITION of functional integration is already written
(`research/findings/2026-06-23-functional-one-brain-integration-scoping.md` §1.1): a cross-region influence
carried by NEURONS + SYNAPSES on the merged bridge, such that an isolated half cannot reproduce the behavior.
Its two DISQUALIFIERS: (a) co-location without coupling (disjoint frozen slices, zero cross-synapses); (b) a
HOST round-trip between regions (`b=to_host(A.read()); B.kick(f(b))`). That doc also names the WORKING
TEMPLATE — `spoken_instruction_nav.py`: a parser ensemble's FIRING opens a transmission gate (`COMMAND_GATE`)
on a route into `cortex_X`, 6-seed GO, lesion-confirmed — and it enumerates the anti-cheat controls this
design inherits (§1 below).

Two SYNAPTIC coupling primitives already exist on the shared bridge (both zero `sim/` edit at use-site):

- A TRANSMISSION GATE opened by a control pool's firing: `set_transmission_gate`/`_apply_gate_couplings`
  (`sim/bridge.py:3179`,`:3229`) + `couple_gate_to_pool` (`sim/bridge.py:3205`). The cross-region influence is
  a 0/1 gate STATE derived from firing — gain modulation of a pre-existing route (the `COMMAND_GATE` shape).
- A synaptic ROUTE injected into the shared `cp_connections` via `inject_explicit_wiring`
  (`sim/bridge.py:4199`) — an actual pathway between two region slices.

The integration level was ALREADY reached, as a single-pathway proof-of-concept:

- `research/findings/2026-08-11-INTEGRATION-7-burndown1-one-brain-merge-coresidency-6seed.md` — CO-RESIDENCY:
  the e-prop acquisition net became disjoint slices of the ONE conversational bridge (one `cp_connections`),
  6/6 GO — but with ZERO cross-synapses. Explicitly "NOT yet cross-region synaptic INTERACTION".
- `research/findings/2026-08-11-cross-region-synaptic-interaction-true-one-brain-6seed.md` — **status: GO**:
  it injected a GENUINE synaptic pathway `conv-cue → eprop_in` on the merged bridge (runner
  `_i7_cross_region_synaptic_derisk.py`) so acquisition input arrives via SYNAPSES, not a host hand-off, and it
  is LOAD-BEARING 6/6 (lesion the pathway → the acquisition input collapses). This is the existing
  cross-region-interaction GO.

What the 2026-08-11 GO did NOT do — and where THIS design advances it (three deltas):

1. It HAND-INJECTED the pathway at a fixed weight. It is a migrated wire, not a SELF-ORGANIZED one — it does
   not meet the mission's emergence bar (a cross-synapse must FORM/LEARN from experience, not be a hand-set
   weight matrix). §2 makes the cross-edge LEARN.
2. It was ONE bespoke pathway in ONE runner. The framework's 7/7 organ-read migration
   (`2026-08-27-onebrain-merge-framework-organ-read-engine-seams.md`) now puts many organs on ONE pool, so the
   cross-edge becomes a DECLARATIVE descriptor field over any organ pair, gated by a general FUNCTIONAL gate —
   the O(N)→O(1) generalization of the single pathway.
3. It coupled an ACQUISITION-INPUT plumbing pathway (cue → the learner's input). §3 picks a COGNITIVE
   faculty→faculty coupling (working-memory → comprehension) whose emergent behavior — reference resolution —
   is conversationally load-bearing, the kind of cross-faculty behavior that IS the one-brain payoff.

So: co-residency (byte-identity, this framework, 7/7) is the safe base; a single hand-wired load-bearing
pathway (2026-08-11 GO) is the interaction proof-of-concept; the general LEARNED N-organ interacting one-brain
under a functional gate is the goal this doc designs.

## 1. THE FUNCTIONAL GATE that replaces byte-identity (concrete + measurable)

Integration BREAKS byte-identity by construction: a cross-synapse makes organ B's read depend on organ A, so
B's read is no longer identical to B-alone. The gate therefore SWITCHES from bit-identity to FUNCTION. It is
per-cross-edge, and it subsumes the 2026-06-23 anti-cheat controls (§4 there). An edge is admitted only if all
four hold, 6 seeds:

- **F1 — FACULTY-STILL-WORKS (each organ keeps its own task metric).** With the edge present, every organ's
  standalone end-to-end metric stays on the correct side of its decision boundary: comprehension keeps its
  well-vs-ill separation (calib well ≈0.46 vs ill ≈0.06, threshold ≈0.32; `read_margin` ≥ threshold on
  well-formed items — from `2026-08-27-onebrain-merge-framework-organ-read-extension.md`); d6 keeps
  `all_recovered=True` with `hold_alive_min ≥ 0.32`; surprise keeps its 22.8x confirm/contradict separation;
  metacog `type2_auc ≥ 0.67`. Co-residence + the edge may PERTURB the numbers but must not cross the boundary
  on the organ's OWN task.
- **F2 — INTERACTION-IS-REAL (the vary-then-lesion crux, BOTH directions; memory
  `feedback_faculties_must_drive_not_observe`).** (i) VARY the SOURCE organ's state → the TARGET organ's
  read/answer must demonstrably CHANGE (a measured Δ above a declared floor). (ii) LESION the cross-edge (zero
  its weight, or hold its gate shut) → the change must VANISH (Δ→~0). This proves the coupling CAUSED the
  change, not a confound. A faculty verdict merely stashed as metadata with no downstream effect is the HOLLOW
  integration the gate exists to reject. This is also the 2026-06-23 "lesion = the interaction vanishes" +
  "both-brains-required" control, made two-sided.
- **F3 — NO-RUNAWAY (stability).** The added recurrence must not destabilize the pool: per-region firing rate
  stays in a physiological band across a long multi-turn stepping burst (the read_isolation stress case), the
  edge weight CONVERGES (bounded by `hebbian_max_weight`, `sim/config.py:777`) rather than diverging, and the
  pool stays alive (no silence, no seizure) with the edge live.
- **F4 — MOAT/HONESTY preserved (the HARD invariant).** The interaction may only REWEIGHT competition among
  options the substrate already has evidence for; it must NOT manufacture a fact, flip an abstain to a
  false-accept, override strong disambiguating evidence, or assert phenomenality. Concretely: on a
  no-evidence input the read stays undecided (no bias can create a winner from silence); on a CLEAR item the
  bias cannot flip the decision; the no-confab moat's `is None`/"unknown" assertions still hold at every
  source-organ state. (2026-06-23 §4.4; `feedback_moat_not_hard_lossy_memory_ok` — kept where free, never
  traded for a number.)

Provenance anti-cheat (inherited): the cross-region coupling must be a synaptic ROUTE or a firing-derived GATE
STATE — never `cp_external_input_current[Bidx] = f(to_host(A))`. The only legitimate host writes stay the
environment (sensory render) and the body (motor act). A LESION-RECOVERS-MIGRATION invariant makes the gate
falsifiable and keeps the migration safety net: with every cross-edge lesioned (gain/weight 0), the integrated
pool must return to the migrated pool's byte-identity — proving the cross-edges are the ONLY change and no
migration regression hid inside the integration work.

## 2. THE EMERGENCE BAR — how cross-edges FORM by LEARNING (not hand-wired weight matrices)

The mission burns down host scaffold; a hand-set cross-region weight matrix IS scaffold (2026-08-11's GO wired
one by hand — the residual this closes). The faithful cross-edge is SEEDED near-zero and GROWS from experience
via the substrate's OWN plasticity, gated by the relevant neuromodulator. Every mechanism already exists on the
shared substrate:

- **Two-factor associative (Hebbian / rate-window BCM).** `enable_hebbian_learning` +
  `_apply_branchless_hebbian` (`sim/bridge.py:1181`) and the RATE-WINDOW co-activity trace (`sim/bridge.py:9767`,
  `hebbian_rate_window`): two neurons BOTH active over a ~10-step window potentiate the synapse between them
  regardless of exact-step alignment — the associative rule for asynchronously-firing assemblies. Bounded by
  `hebbian_max_weight` (`sim/config.py:777`). This is the biological substrate of LEARNED cortico-cortical
  associations — Miyashita paired-associate neurons in inferotemporal cortex acquire pair-coding by experience
  (Kandel PNS 6e, ch. on associative recall) — and it is what grows the FIRST edge in §3.
- **Three-factor (neuromodulator-gated STDP).** `enable_stdp` + `_apply_branchless_stdp` (`sim/bridge.py:1009`)
  with the eligibility trace (`sim/bridge.py:10074`) gated by `enable_reward_modulation` (`sim/config.py:908`):
  pre × post × dopamine = weight change, tag decays τ≈500 ms (`docs/biology.md:280`; Frémaux & Gerstner 2016;
  Schultz). This is how a cross-edge learns SELECTIVELY — only when a modulator (novelty/surprise, relevance,
  reward) marks the co-activity as worth wiring. The modulator can be delivered as a firing-gated plasticity
  gate (`set_transmission_gate`/`couple_gate_to_pool`, `sim/bridge.py:3179`,`:3205`) so the THIRD factor is
  itself a spiking pool, not a host scalar.
- **One-shot / plateau (BTSP, BDSP).** `enable_btsp`/`fused_btsp_update` (`sim/kernels.py:1485`),
  `enable_bdsp`/`fused_bdsp_update` (`sim/kernels.py:1451`): for cross-edges that must form from a SINGLE
  salient episode (a taught fact, a one-trial binding) rather than accumulated co-activity.

The confinement primitive that makes a LEARNED cross-edge co-residence-safe already exists and is proven in the
framework: `cp_plasticity_rate_gain` (`sim/bridge.py:710`,`:4384`) is a per-synapse 0/1 multiplier on every
plasticity rule. The framework's `_apply_gain0_freeze` (`onebrain_merge_framework.py:314`) sets it to 0 on every
intra-organ edge and ASSERTS no edge crosses a frozen boundary (`cross = row_in ^ col_in`, `:319`).
causal_whatif's `_freeze_non_evt` (`:1127`) is the exact template already used in the migration: "gain-0 freeze
of every edge that is NOT [X]-internal ... only the [X] edges stay plastic, so the STDP tags + DA-gated
three-factor updates are confined". The integration extension is a ONE-LINE inversion of that assert: the
declared cross-edge is WHITELISTED as the sole plastic edge (gain 1), everything else frozen (gain 0). So the
cross-edge is the ONLY synapse in the whole pool that learns — it self-organizes from experience while every
migrated organ's internals stay frozen and byte-stable. The descriptor gains a field
`cross_edges: [(src_region, dst_region, w0≈0, rule, modulator_gate)]`; adding an interaction is a registry ROW
under the F-gate, not bespoke wiring — the same declarative move the framework made for migration.

Honest scaffold note (tracked, not hidden): even a LEARNED edge still has host-chosen TOPOLOGY (which regions
connect) and a curated EXPERIENCE stream at first. The faithful end state has the pairing itself emerge from raw
dialogue (the teacher just talks; the edge forms where activity correlates). This is a `scaffold_residual`
carried per cross-edge, on the same burn-down as the framework's block-diagonal masks and gain-0 freezes — the
declarative merge must not ossify a permanent host-wired connectome.

## 3. THE FIRST INTERACTION — working-memory referent → comprehension role competition (WM-guided reference resolution)

Chosen pre→post: **d6_multiref_wm (a held-referent assembly, region `w{k}`) → comprehension
(`sel_agent`/`sel_patient` Wong-Wang accumulators)**. Both organs are already ORGAN-READ CLOSED on ONE pool
(byte-identity + non-degenerate reads, `2026-08-27-onebrain-merge-framework-organ-read-extension.md`), their
configs UNION with no MergeConflict (both frozen + NMDA-on; `_D6_CONFIG`/`_COMPREHENSION_CONFIG` in the
framework), and both reads are clean frozen forward passes — the minimal-moving-parts base for the first edge.

Why biologically. dlPFC working memory exerts TOP-DOWN bias on temporal/parietal comprehension circuits;
active maintenance of a discourse referent biases syntactic/semantic role assignment (interactive-activation /
predictive parsing). The mechanism is a LEARNED cortico-cortical association — the same Miyashita paired-
associate class the substrate's Hebbian rule implements (Kandel PNS 6e), plus corticofugal top-down feedback
(auditory/visual cortex feedback sharpening lower-area responses, Kandel PNS 6e). It is a directed PFC→
association-cortex projection, exactly the shape of a single learned cross-edge.

Why conversationally (the #1 fluency frontier, comprehension side). This is pronoun / reference resolution:
"The dog chased the cat. It was fast." — the WM-active referent biases who the ambiguous continuation binds to.
A brain that resolves it/she/they against what it is HOLDING converses coherently across turns. Comprehension
is the one speech-facing organ already co-resident + closed on the pool (affect and metacog live in other
pools, Group-C/pool-2), so this is the best-motivated first edge that touches the mouth/comprehension frontier
without a new pool seam.

- **Pre / post regions.** pre = a d6 slot pool `w{k}` holding the active referent (region flags
  `_D6_REGION_FLAGS`, `onebrain_merge_framework.py:~739`); post = comprehension `sel_agent`/`sel_patient`
  (`_spiking_comprehension_monitor_derisk.py:13`). Directed `w{k} → sel`.
- **Plasticity rule.** Seed the `w{k}→sel` edge-set at w≈0 (`cp_plasticity_rate_gain=1` on it, 0 everywhere
  else). GROW it by rate-window Hebbian (`hebbian_rate_window`) over experiential episodes where a referent is
  held in WM while its ground-truth role fires — the edge learns which held referent maps to which role slot.
  Two-factor for the first de-risk; the three-factor upgrade (a relevance/attention modulator gating WHEN it
  learns) is roadmap R2 (§4), deferred only because the neuromod subsystem is a known pool seam (curiosity's
  deferral), not because two-factor is non-emergent — a Hebbian-grown edge already meets the emergence bar.
- **Functional-gate metric.** F1: comprehension still separates well/ill (margin ≥ ~0.32 on well items) and d6
  still recovers all referents, edge present. F2: on an AMBIGUOUS item (balanced cues, baseline margin ≈0),
  VARY which referent is loaded into d6 → the sel margin shifts toward the WM-held referent (signed Δ above a
  floor); LESION the `w{k}→sel` edge → Δ→~0. F3: rate band + weight convergence over the load→hold→comprehend
  burst. F4: a no-cue (silent) item stays undecided regardless of WM; a CLEAR agent-cue item is NOT flipped by
  a wrong WM referent (the bias only tilts genuine ambiguity).
- **The vary-then-lesion test (F2, concrete).** conditions = {referent-0 in WM, referent-1 in WM, no referent}
  × {edge intact, edge lesioned}, on the same ambiguous transitive. PASS = sign(Δmargin) follows the held
  referent with the edge intact, and |Δmargin| < floor with it lesioned, 6 seeds.
- **Smallest 6-seed de-risk.** A tiny numpy pool: the d6 slot pools + comprehension sel/cue slices (a few
  hundred neurons) co-resident, one plastic `w{k}→sel` edge, ~K experiential episodes to grow it, then the
  vary-then-lesion battery + the F1/F4 controls. Seeds 42,43,44,100,101,102. numpy CPU → routes to
  `tools/sweep_pool.sh` (mini-PC); zero GPU, zero agent tokens (cost-routing).

FEASIBILITY SMOKE (done; the crux, on the REAL substrate). `research/runners/_onebrain_integration_crossedge_smoke.py`
builds two small region slices on a real `SimulationBridge`, injects ONE near-zero PLASTIC cross-edge (the sole
plastic synapse; every structural edge `plastic=False`), and runs the emergence + F2 core: co-drive src+tgt so
the edge GROWS by rate-window Hebbian, then vary (drive src on/off, weak "ambiguous" drive into tgt) and lesion.
Result (`research/findings/raw/_onebrain_integration_crossedge_smoke_6seed.json`, 6/6 PASS): the edge grows from
0.05 to mean ≈9.7 (LEARNED, not hand-set — emergence), tgt read shifts by Δ≈+0.10 spikes/neuron/step when src is
active (LOAD-BEARING, F2-i), and lesioning the grown edge collapses Δ to 0.0 exactly (LESION-REMOVES, F2-ii).
This is the abstract stand-in for `w{k}→sel`; it proves the mechanism (learned cross-edge, load-bearing,
lesionable) is real on the substrate before the organ-specific build.

## 4. ORDERED ROADMAP (interactions after the first) + where the production flip fits

1. **R1 — d6 WM → comprehension (this doc, two-factor Hebbian).** The first learned faculty→faculty edge; the
   full 6-seed organ de-risk (§3), building on the smoke.
2. **R2 — upgrade R1's edge to THREE-FACTOR (neuromodulator-gated).** A relevance/attention (ACh) or
   novelty/surprise modulator gates WHEN the WM→role edge is allowed to learn (`enable_reward_modulation` +
   the eligibility trace, delivered via a firing-gated plasticity gate). This establishes the neuromod-gated-
   plasticity backbone every later edge reuses; it needs the neuromod-subsystem pool seam (curiosity's declared
   deferral) — the reason it is R2 not R1.
3. **R3 — surprise → episodic/provenance ENCODING gate (the canonical novelty-gated-plasticity edge).**
   Prediction-error (surprise) as the third factor gating encoding of source_provenance / d6 (Lisman-Grace
   VTA-hippocampal loop; the DA encoding-gain hook `encoding_gain_fn` is already de-risked GO per the 2026-06-23
   scoping, and surprise-gated plasticity on the live turn is a QUEUED item, `docs/BURN_DOWN_LIST.md:146`).
   Requires folding surprise (pool #1) into the shared pool — reconcile its Hebbian-on config via per-region
   gating (surprise's own edges plastic, the rest frozen).
4. **R4 — comprehension / self_schema → metacog SPEECH-CONFIDENCE.** Comprehension margin + self_schema
   authorship drive a confidence read that sets speech hedging/tone — directly speech-fluency-relevant. Needs
   metacog folded from pool-2 into the shared pool (the param-het-ON-vs-OFF config conflict + the
   workspace/meta_schema name-collision reconciliation, `onebrain_merge_production2.py`).
5. **R5 — curiosity → attention / comprehension threshold.** Curiosity modulator lowers the comprehension
   decision threshold on an attended topic (reuses R2's neuromod seam).
6. **Reciprocal / multi-edge loops.** Once several directed edges hold their F-gate, allow reciprocal coupling
   (e.g. comprehension → WM: a resolved referent updates what WM holds) under F3's stability monitor + the
   lesion-recovers-migration invariant, so the pool becomes a genuinely interacting one-brain, not a DAG of
   one-way biases.

PRODUCTION FLIP (retire MergedSubstrate*, integrate into `server.py` `brain_chat`). Fits AFTER R1–R3 (or R4)
hold their F-gate across 6 seeds AND the whole-pool multi-turn chat stays stable (F3) with the moat intact
(F4). Flip criterion, concrete: (i) F1–F4 pass for EVERY cross-edge in the integrated pool; (ii) the
lesion-recovers-migration invariant holds (integration added only the declared edges); (iii) the LIVE chat
demonstrably USES an interaction (e.g. a pronoun visibly resolves to the WM-held referent, and the resolution
vanishes when the cross-edge is lesioned in the live turn — the drive-couplings discipline, memory #84/#85);
(iv) the honesty battery shows no moat regression. Then `MergedSubstrate`/`MergedSubstrate2` become a thin
`merge_organs([...], cross_edges=[...])` call wired into `brain_chat`, and the bespoke pools retire (the
twopool 6/6 byte-identity is the evidence the framework reproduces production).

## 5. RISKS + THE HONEST BOUNDARY (and how the gate catches each)

- **Instability / runaway (added recurrence excites the pool into silence or seizure).** Caught by F3 (rate
  band + weight convergence). Mitigation: `hebbian_max_weight`/`stdp_w_max` caps, E/I balance via the target's
  lateral-inhibition FS pool, homeostasis where it does not break determinism. If an edge cannot satisfy F3 it
  is NO-GO — bank the method, try a weaker rule / a gated (transmission-gate) coupling instead of a raw synapse.
- **Emergence does not happen (the Hebbian edge fails to converge to a USEFUL mapping — co-activity too noisy,
  or it learns the wrong association).** This is the honest-negative path and F2 is its instrument: if the
  grown edge does not shift the target margin, or the lesion does not remove the shift, the interaction METHOD
  is REFUTED (a hollow edge), banked — not deferred. The smoke shows the mechanism CAN converge cleanly (6/6);
  the organ-specific de-risk tests whether the SPECIFIC pairing does.
- **Moat leak (a bias manufactures a fact / flips an abstain).** Caught by F4 (no-evidence stays undecided;
  clear items are not flipped; the `is None`/"unknown" assertions hold at every source state). The
  additive-honest constraint every organ carries stays binding through integration.
- **Byte-identity is LOST by design — losing the ability to DETECT a migration regression is the real danger.**
  Mitigation: the lesion-recovers-migration invariant (§1) keeps the byte-identity migration gate as a
  separate, still-runnable lane (lesion all cross-edges → the pool must be byte-identical to the migrated pool).
  Integration and migration verification coexist rather than one destroying the other.
- **The FP-determinism floor (from `2026-08-27-onebrain-merge-framework-multiturn-stateful-read.md`): at N≈4968
  the strict bit-identity tips for long-integration reads.** The functional gate uses TOLERANCED decision
  boundaries (margins, rate bands), NOT bit-identity, so it is ROBUST to that floor — the functional gate is
  the correct instrument PAST the determinism floor, which is another reason integration must switch metrics.
- **The honesty boundary (unchanged, a DELIVERABLE not a caveat).** Every interaction is an additive, honest,
  functional read-out; none asserts phenomenal experience. A cross-edge that only makes the brain's behavior
  more coherent (reference resolution, novelty-gated memory, confidence-tuned speech) is exactly the
  cross-faculty behavior the one-brain bet predicts — measured, lesion-verified, moat-safe.

## Files (this design)

- `research/findings/2026-08-27-onebrain-integration-phase-DESIGN.md` — this doc.
- `research/runners/_onebrain_integration_crossedge_smoke.py` — the feasibility smoke (learned cross-edge,
  vary-then-lesion; numpy CPU; NO `sim/` edit).
- `research/findings/raw/_onebrain_integration_crossedge_smoke_6seed.json` — the 6/6 smoke artifact.
