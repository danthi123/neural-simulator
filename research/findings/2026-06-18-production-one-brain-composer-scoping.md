# The production `OneBrainComposer` — turning the 4 GO de-risks into the integrated agent, with the two open problems scoped (2026-06-18)

**Scoping type:** read-only. No code edited, no experiments run. One design/scoping document.
**Audience:** the controller (owner). Plain language; every term defined once; no undefined acronyms.
**Companion to** `2026-06-18-one-brain-integrated-pipeline-scoping.md` (the broad architecture + the incremental
plan). This doc is the narrower PRODUCTION-CLASS build spec: it picks up from the four GO de-risks and drills into
the two architectural problems they did NOT solve (the multi-fact store on a persistent bridge; the parser front-end)
plus the winner-take-all-selection question, and names the FIRST cheap-first de-risk concretely.

## Terms (defined once)
- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated neurons with one step loop.
- **op** — one conversational operation (bind / unbind / bundle / cleanup / store / retrieve / abstain / parse).
- **RF** — resonate-and-fire (`NeuronModel.RESONATE_AND_FIRE`): the composer's spiking neuron. Its state is a complex
  number `Z = re + i·im` stored in the bridge's `v` (membrane, `cp_membrane_potential_v`) and `u`
  (recovery, `cp_recovery_variable_u`) arrays. A concept/role is a **phasor** vector — a phase angle in `[0,1)` per
  dimension; the phase, not the magnitude, carries the information.
- **FHRR** — Fourier Holographic Reduced Representation: the vector-symbolic algebra the composer realizes
  (bind = complex multiply = phase add; unbind = multiply by the conjugate; bundle = sum), run THROUGH complex
  synapses (`rf_set_complex_weights` installs sparse `cp_rf_w_re`/`cp_rf_w_im`; each step `_rf_advance_one` does the
  complex matvec `W·z` and adds it to the rotating state).
- **register** — a D-neuron RF slice holding one phasor vector as its `Z` state.
- **host** — Python/numpy on the CPU, plus the host↔device transfers it forces.
- **host round-trip** — reading spiking state OFF the bridge (`rf_read_phases` → numpy), computing/routing in numpy,
  then driving the result BACK on (`rf_kick`). The thing the integration removes.
- **weight-store** — the existing spiking memory-in-synapses: a composite phasor held in complex synaptic weights of a
  persistent RF bridge (`RFPhasorComposer._store_substrate`/`_retrieve_substrate`, validated == numpy at parity).
- **transmission gate** — a per-synapse current multiplier in `[0,1]` (`cp_transmission_gain`); `set_transmission_gate`
  opens/closes a pre-wired route. NOTE (verified §5 risk B): it gates the Izhikevich matrix `cp_connections`, NOT the
  RF complex matvec.
- **plasticity gate** — a per-synapse weight-update multiplier (`cp_plasticity_rate_gain`); `=0.0` freezes a
  population's weights even under global learning.
- **the moat** — the no-confabulation abstention behaviour: a query returns `None`/`"unknown"` when no stored fact
  matches, rather than inventing one. Must NEVER be weakened (owner: keep it where free; it stays correct throughout).

---

## 1. Diagnosis — what the four GO de-risks proved, and the three things the production build still needs

### 1.1 What is already PROVEN (register→register on ONE persistent bridge, no host round-trip)
Four de-risks this session, each 3 seeds × 2 D (6/6), each == its numpy oracle, each with collapsing anti-cheats:

| de-risk | runner | proved |
|---|---|---|
| synaptic phase handoff | `_phaseB_onebrain_register_handoff_derisk.py` | `unbind(bind(role,filler),role)` chains register→register on one bridge with **no `rf_read_phases` between the two ops**; recovers the filler 1.000 == the host two-call pipeline; permuted-role 0.051 + severed-route lesion 0.077 collapse. |
| fact store + query | `_phaseB_onebrain_fact_store_query_derisk.py` | a full 2-role fact `bind(agent_role,agent)+bind(action_role,action)` settles into a stored register **C** over a 3-window settle (bind→bundle→query, no read-out between), and BOTH roles unbind out 1.000 == host; lesioned binds collapse. |
| cleanup on-bridge | `_phaseB_onebrain_cleanup_onbridge_derisk.py` | concept-score neurons read the recovered **Q** register through `conj(codebook)` complex synapses, membrane (re) = the matched-filter score; argmax == the numpy cleanup 1.000; random-codebook control collapses (0.17). |
| moat on-bridge | `_phaseB_onebrain_moat_onbridge_derisk.py` | the cleanup's **peak score** IS the familiarity signal: a stored, correctly-cued fact → HIGH peak (answer); an unbound query → LOW peak (abstain); clean separation 6/6 at a measured (not tuned) midpoint threshold. |

So `bind → bundle → store-in-register → unbind → cleanup-matched-filter → moat-peak-read` all run on one persistent
bridge, register→register, with the FHRR complex synapses carrying the value from each op to the next — **no numpy
between ops**. The conversational *core* (store a fact, query a role, clean up, abstain) is demonstrated on one brain.

### 1.2 What the de-risks did NOT solve — the three production gaps

**(GAP A) The MULTI-FACT store.** The de-risks store ONE fact in register C and query it immediately. C is RF
*register state* (`v`/`u`), which a work-register reset erases — and the `fact_store_query` finding states the
load-bearing insight directly (CYCLE 168-169): *"WORK registers must be RESET between facts; stored facts must live in
SYNAPSES, not register state … so a register reset can't erase them — the operand-vs-store split."* A real knowledge
base has MANY facts, all persisting across turns and across the per-op work-register resets. The de-risks have no
multi-fact store on the persistent bridge: their store is one register, and their KB-scan (`_unbind_all_phases`/
`_cleanup_all`) is still the *per-op* numpy batched scan over `self.kb` (a numpy list).

**(GAP B) The PARSER front-end.** The de-risks HOST-set the operand-register kicks (`kick[0:D] = za_fill` etc., i.e. the
agent code is written in from numpy). The production pipeline must drive the operand registers from the PARSER's role
firing — comprehension is synaptic. The de-risks contain no parser; the operands arrive as host kicks.

**(GAP C, the WTA-selection question) — is the host argmax legitimate, or does it need the spiking WTA?** The cleanup
and moat de-risks END with a host `np.argmax` over the on-bridge concept-neuron membranes (`re[D:D+V]`). Reading "which
unit scored highest" off a finalized membrane vector is the same KIND of read-out as "which motor pool fired" — under
the BRAIN-BASED-ONLY standard a final argmax-over-a-neural-readout is a defensible body/output read, NOT a cognitive
host computation (the cleanup/moat themselves are on-substrate; only the *selection read* is host). **So the host
argmax is a legitimate read-out and is NOT a blocker for the production build.** However, the validated spiking
alternative already exists: `RFPhasorComposer._spiking_cleanup` stage 2 (`rf_phasor_composer.py:241-256`) drives a
co-resident Izhikevich concept bank with the input-normalized scores and reads **argmax-over-FIRING** — i.e. the winner
is the unit that *fires*, exactly "read which pool fired". Folding that in makes the selection fully spiking and is the
cleaner end state, and per-neuron-model RF/Izhikevich coexistence already shipped (task #11, the masked-RF-ops). **The
HARD sub-problem, IF the spiking WTA is wanted on the SAME persistent bridge:** today `_spiking_cleanup` couples the RF
matched-filter to the Izhikevich WTA by a HOST hop — it reads `re[D:D+V]` to numpy, normalizes, and writes
`cp_external_input_current` on a *separate* Izhikevich bank. To make that on-bridge with no host hop, the RF
concept-score neurons' membrane must drive the co-resident Izhikevich WTA neurons' input current **synaptically** — an
**RF-membrane → Izhikevich-current coupling** that does not exist in the step loop today (the RF branch writes `v`/`u`
as the complex state; the Izhikevich branch reads `v` as voltage and `cp_external_input_current` as drive — there is no
path from "an RF neuron's re" to "an Izhikevich neuron's input current"). This is the one genuinely-new neuron-model
coupling the fully-spiking selection would need; it is FLAGGED here as the hardest optional sub-problem, and it is
**off the critical path** because the host argmax read-out is legitimate. Recommendation: ship the production
`OneBrainComposer` with the host argmax read-out (brain-based-compliant), and treat the RF→Izhikevich-coupled spiking
WTA as a later optional biologization, scoped separately if/when wanted.

### 1.3 The honest framing of "exactness"
This build changes WHERE the ops run (persistent on-bridge, synaptic handoff) — it does NOT change WHAT they compute.
The bind/unbind stay the exact-inverse FHRR algebra (the principled Eliasmith Semantic-Pointer idealization; the
audit's row-2 idealization). De-idealizing the algebra into a *learned* binder is the SEPARATE learned-binder arc
(`OnBridgeLearnedComposer`, CYCLE 150 6-seed GO) — orthogonal to and compatible with this integration (a learned binder
is also a set of synapses on the persistent bridge). The production `OneBrainComposer` is the FHRR pipeline made
persistent + synaptic + single-bridge; numpy stays the test oracle.

---

## 2. Target `OneBrainComposer` architecture

A drop-in sibling/replacement of `RFPhasorComposer` exposing the SAME agent API (`store`, `query_patient`,
`query_agent`, `ask_yes_no`, the moat-as-`None`) so `BrainConversationalAgent` can swap it in by `composer_kind`,
with the WHOLE pipeline on ONE persistent bridge that is **never rebuilt per op**.

### 2.1 The end state in two sentences
All conversational regions live as disjoint, permanently-allocated neuron-index slices on ONE persistent
`SimulationBridge`: a parser slice, a phasor workspace of RF register slices (operand-A, operand-B, role bank,
bound-A, bound-B, accumulator/composite), a PERSISTENT plastic fact-store region, cleanup concept-score neurons, a
moat read, and (optionally) a spiking winner-take-all slice. A `store(fact)` and a `query(role)` flow end-to-end as
spikes through complex synapses — the parser's role firing routes each comprehended word's concept code into the
correct operand register, the FHRR complex synapses carry bind/bundle/unbind from one register to the next WITHIN the
persistent step loop, the new fact is written into the persistent fact-store's complex weights, a query unbinds the
cued probe against the WHOLE stored set, the concept-score neurons fire the match, the moat peak-read gates
abstain-vs-answer — with the host doing only text I/O (the sentence in; the body's final `" ".join` of
already-spelled, already-ordered words out).

### 2.2 Regions on the one persistent bridge
| slice | role | substrate | state lives in |
|---|---|---|---|
| `parse_conj` (6) + `parse_role` (3·R) | comprehension (position×voice → role) | Izhikevich, Hebbian-learned (the existing `BridgeParser`) | `v`/`u` as voltage |
| `op_A`, `op_B` (D each) | the two FHRR operands | RF registers | `v`/`u` as complex `Z` |
| `role_bank` (D) | the active role phasor (parser-gated) | RF register | `v`/`u` as `Z` |
| `bnd_A`, `bnd_B`, `acc` (D each) | bound + bundled composite under construction | RF registers | `v`/`u` as `Z` |
| **`kb_store`** (the fact memory) | every stored fact = a composite in **complex synaptic weights** | RF weight-store, GROWING | `cp_rf_w_re`/`cp_rf_w_im` (array-disjoint from `cp_connections`) |
| `Q_*` (D each, per cued role) | the recovered probe per query | RF registers (separate per query — CYCLE 168 insight) | `v`/`u` as `Z` |
| `clean_score` (V) | matched-filter scores (concept neuron j accumulates `conj(code_j)·Q`) | RF complex synapse → membrane = score | `v` (re) = the score |
| `fam_read` | the moat: peak-score threshold | host peak-read (or a small Izhikevich threshold pop) | — / Izhikevich |
| `clean_wta` (V, OPTIONAL) | spiking selection (argmax-over-firing) | co-resident Izhikevich, driven by the scores | `v`/`u` as voltage |

These coexist because their state is in **disjoint arrays**: Izhikevich slices use `v`/`u` as voltage; RF slices use
`v`/`u` as the complex phasor (the masked-RF-ops co-residence); the FHRR memory lives in `cp_rf_w_re`/`cp_rf_w_im`,
which the Izhikevich step never touches. The merged-bridge work proved Izhikevich+RF slices coexist byte-isolated
(`tests/test_rf_neuron_mask_coexistence.py`, `test_merged_rf_composer_coresident.py`).

**Design rule, pinned (verified §5 risk F):** persistent RF MEMORY = synapses (`kb_store`, immune to Izhikevich steps);
transient RF COMPUTE = masked `v`/`u` registers (re-kicked per op). This is the same guarantee `MergedRFComposer`
relies on.

### 2.3 End-to-end spike flow — `store(fact)` (no host round-trips; host = text in)
1. The parser comprehends "dog go north" → fires `agent=dog, action=go, patient=north` (already neural).
2. Per role R, the parser's role-ensemble firing **opens a transmission-gated route** that drives `role_bank` with
   role R's phasor while the word's concept code drives `op_B`; the FHRR diagonal synapse `op_B ⊗ role_bank → bnd_*`
   binds THROUGH synapses (the de-risk's bind, but the operands ARRIVE synaptically). This is the `hear_synaptic`
   precedent, extended from the rate-coded coincidence banks to RF registers.
3. The binds bundle into `acc` (unit complex synapses summing into `acc`) over a settle window.
4. `acc` is **written into the persistent `kb_store`** as a NEW fact's trigger→readout complex weights (the
   `_store_substrate` mechanism, appended to the ONE persistent store, not a throwaway per-fact bridge). The fact is
   now memory-in-synapses; no numpy array holds it; a work-register reset cannot erase it (GAP A's resolution).

### 2.4 End-to-end spike flow — `query_patient(agent, action)` (no host round-trips; host = `" ".join` out)
1. Build the cue probe (agent+action role-bound) in `op_A` (driven from the cue's concept codes — in production via
   the parser comprehending the question; for a programmatic API call, a host kick of the cue codes is a legitimate
   text-in boundary).
2. The persistent `kb_store` complex synapses fan the probe against EVERY stored fact (the resident equivalent of the
   block-diagonal `_unbind_all_phases`), unbinding the cued role → drives `clean_score`.
3. `clean_score` concept neurons fire the matched filter (membrane re = score; the on-bridge cleanup de-risk).
4. `fam_read` peak-reads the scores → fires "answer" only if the peak is above threshold, else "abstain" (the
   moat-on-bridge de-risk; abstention happens BEFORE any rendering, so the moat is never weakened).
5. On "answer", the winning concept (host argmax read-out, or `clean_wta`'s firing) keys the renderer; the host does
   only the final word-join (the body).

### 2.5 The crux mechanisms (and what is new vs reused)
- **Register→register synaptic handoff in one resonate window** — PROVEN (step-1/step-2 de-risks). The handoff is
  "which complex synapses are installed", not a host re-kick: op N's output register IS op N+1's input register,
  connected by an FHRR synapse; the downstream complex matvec consumes the complex *state* `z` (`W·z`, not `W·phase`),
  so chaining is native AS LONG AS both registers resonate in the same window (or a GPU-side multi-window settle —
  also proven). NEW here only in being scaled to the full persistent layout, not in the primitive.
- **The growing persistent fact-store** — the genuinely-new GAP-A piece (see §3.1).
- **The parser→operand-register routing** — the genuinely-new GAP-B piece (see §3.2).

---

## 3. The two open problems, scoped concretely

### 3.1 GAP A — the multi-fact synapse-store on the SAME persistent bridge

**The store mechanism already exists and is validated.** `_store_substrate` (`rf_phasor_composer.py:406-416`) holds a
composite in a `(1+D)` RF bridge's `trigger(neuron 0) → readout(1..D)` complex weights `conns = [(1+k, 0, zc[k])]`;
`_retrieve_substrate` (`:418-426`) fires the trigger (unit phasor) → the readout neurons reconstruct the composite IN
PHASE. It is the Crawford-Eliasmith memory-in-weights; validated == numpy at parity (the Phase-2 fact-store-query GO).
Today it builds a SEPARATE `(1+D)` bridge per fact and is default-off.

**The production question: how do K facts live in ONE persistent store region, register-reset-safe, and queryable
on-substrate?** Two concrete layout options, cheaper-first:

- **(A1, the safe first cut) Pre-allocated max-K block of triggers.** Reserve a `kb_store` region of `K_max` triggers
  + a shared `D`-neuron readout, all RF. Storing fact `i` installs `conns_i = [(trig_base + i, ..., readout_k? )]` —
  concretely, each fact gets its OWN trigger neuron and its own `trigger_i → readout_block_i` (D readout neurons),
  i.e. the de-risk's `(1+D)` store TILED `K_max` times into one region (`K_max·(1+D)` neurons). A store is then "fire
  fact i's trigger to reconstruct composite i", and these complex weights are NEVER advanced by an Izhikevich step
  (array-disjoint), so they persist across turns and across work-register resets — **GAP A's core resolution.** Wastes
  the unused `K_max − K` triggers; trivially correct; no CSR-rebuild hazard.
- **(A2, the scaling cut, deferred) Append to the complex CSR.** Grow `cp_rf_w_re`/`cp_rf_w_im` by appending fact i's
  rows. The RF complex weights are a SEPARATE CSR from `cp_connections`, so the
  `set_pathway_weights(add_missing=True)` CSR-resort hazard documented at `unified_brain_bridge.py:106-116` (which
  stales gate-index maps) is CONTAINED to the RF store and does not touch the Izhikevich wiring or the gates. Still,
  A2 needs a careful append (and `rf_set_complex_weights` currently builds the CSR *fresh* and REPLACES — `:5509-5510`
  — so an append path is a small new method, a likely flagged `sim/` edit). Defer A2 until A1 hits a capacity wall.

**How a query is routed to scan/match the stored facts on-substrate.** The de-risks' batched scan
(`_unbind_all_phases`, `rf_phasor_composer.py:274-291`) already does the right COMPUTATION — unbind a role from ALL K
composites in ONE resonate over K isolated 2D-blocks, equal to K separate unbinds, ONE launch — but it builds a
throwaway `2·K·D` bridge per query from the numpy `kb`. The mapping onto persistent storage (A1):
1. Fire all K triggers (or the cue selects which to fire — but the FHRR way is fire-all, the store IS the
   superposition-search) → the K readout blocks reconstruct the K composites IN PLACE on the persistent bridge.
2. A resident block-diagonal unbind synapse (the cued role's `conj` diagonal, tiled per block) unbinds the cued role
   from each reconstructed composite → drives a per-block recovered register.
3. Those drive the shared `clean_score` matched-filter (the cleanup de-risk), and the moat peak-read + a first-match
   pick selects the answering fact.

So the resident store IS a separate plastic (complex-weight) region the query unbinds against, and it is
register-reset-safe BECAUSE the facts are in synapses, not register state. (The first-match/string-equality moat of
the numpy path becomes the on-bridge peak-read + the per-block score comparison — already the moat-on-bridge de-risk's
mechanism, just over K blocks.)

**Honest risk for GAP A:** phase coherence as K grows (the per-block reconstruct→unbind→score is a 3-op chain per
block, and K blocks resonate together). Mitigation: the multi-window settle (proven to 5 ops at 1.000); fallback: a
per-block settle micro-schedule (GPU-side, no host hop) or cap K per persistent bridge and shard. Store capacity /
cross-talk at many facts is the OTHER GAP-A risk (the superposition-search SNR), addressed in §5.

### 3.2 GAP B — the parser front-end (drive the operand registers from the parser's role firing)

**The precedent exists and is validated.** `UnifiedBrainBridge.hear_synaptic` (`unified_brain_bridge.py:447-507`) +
`_op_synaptic` (`:509-...`) already route the parser's role selection into the composer's role bank through
transmission gates with NO numpy `{role: word}` dict: per word, drive the parser conjunction `(position, voice)` →
the parser's role ensemble fires → `couple_gate_to_indices(bridge, "role_route_<R>", parser.role_idx[R])`
(`unified_brain_bridge.py:441`) opens that role's gate → role R's pattern reaches the role bank, while the word's
concept code drives the fill bank ungated. The GATE PRE-WARM discipline (`_op_synaptic` docstring, `:516-527`) is the
load-bearing timing: a PRE-WINDOW runs until the parser FIRES and the coupling OPENS the gate (the gate genuinely
opens from the parser's firing, not set by hand), THEN a READOUT window holds the parser-opened gate while the bind
accumulates — the biologically-correct order **comprehend → latch the route → compose**.

**The production question: how does this drive RF operand registers (not the rate-coded coincidence banks)?** The
adaptation:
1. Reuse `BridgeParser` verbatim (it is already neural; `role_of(position, voice)` → role; `role_idx[R]` → the role
   ensemble indices, `brain_conversational_agent.py:54`). The parser slice is appended as framework regions exactly as
   `UnifiedBrainBridge`/`MergedNavConvAgent` already do.
2. Per comprehended word: drive the parser conjunction → its role ensemble fires → `couple_gate_to_pool` /
   `couple_gate_to_indices` opens `role_route_<R>` → a pre-wired route drives `role_bank` (the RF role register) with
   role R's phasor, while the word's concept code drives `op_B` (the RF fill register). The FHRR `op_B ⊗ role_bank →
   bnd_*` synapse binds.
3. Use the gate-prewarm two-window latch (`_op_synaptic`'s pattern) so the route is held open across the RF resonate
   window — the WM-latch discipline the unified bridge already validated (`unified_brain_bridge.py:65-79,509-592`).

**The one real adaptation vs `hear_synaptic`:** `hear_synaptic` routes a ROLE *current* into a rate-coded ON/OFF
coincidence bank; here the route must drive a ROLE *phasor* into an RF register's complex state. Two sub-options:
(B-i) the route's synaptic current sets the RF register's `Z` (an RF "kick via synapse" — the role-bank register is
driven to the role phasor by a gated complex synapse from a role-source register, all RF); or (B-ii) a small bridge
helper that, on the gate-open signal, performs the masked `rf_kick` of `role_bank` to role R's phasor (a controlled
re-kick, still no host *decision* — the parser's spikes choose R; the kick value is role R's fixed code). B-i is the
purer all-synaptic form; B-ii is the smaller first step and is the natural FIRST cheap-first de-risk if GAP B is
tackled before GAP A (see §4). **Honest risk for GAP B:** the gate-coupling EMA timing (the parser must fire and latch
WITHIN the pre-window cap, `ROLE_GATE_PREWARM_CAP_STEPS`) co-existing with the RF resonate cadence (dt, period) — a
new co-residence regime (Izhikevich parser firing + RF register resonating in one loop), validated incrementally.

---

## 4. Incremental build plan (cheap-first, each individually verifiable; the moat preserved throughout)

Ordered so each step is verifiable against the numpy oracle, with an anti-cheat, the moat never weakened. CuPy for the
real runs; numpy retained as the TEST ORACLE; 6-seed for any variable effect; an exact/identity (null) effect needs
parity at 3 seeds × 2 D (the merged-bridge byte-identity precedent). Picks up from the 4-GO state (steps 1, 2, 3a, 3b
DONE). Each step writes a findings doc; a NEGATIVE step stops the cascade and the prior default (host orchestration for
that piece) stays until the step is GO.

**STEP A0 (prerequisite, mostly done — confirm, don't rebuild).** The production flag-flips are 6-seed GO already
(`2026-06-18-production-fully-brain-based-flag-flips-GO.md`): `enable_spiking_cleanup`+`enable_substrate_store`+
`enable_neural_render` ON, the full who/what/yes-no/moat/describe matrix == oracle, moat 6/6, CI-guarded
(`tests/test_production_spiking_flags.py`). This is the per-op fully-spiking baseline the integration builds on.

**STEP A1 — FIRST CHEAP-FIRST DE-RISK (recommended): the MULTI-FACT persistent synapse-store on ONE bridge.**
*Why this first:* GAP A is the SMALLER independent step — it reuses the validated `_store_substrate`/`_retrieve_
substrate` mechanism, needs NO parser, and extends the already-GO fact-store-query de-risk by exactly one new property
(many facts, persistent, register-reset-safe). GAP B (the parser front-end) depends on more moving parts (parser slice
+ gate-coupling + RF-kick-via-gate timing), so it is the second step.
- **Build:** one persistent RF bridge with a pre-allocated `K_max`-fact store region (layout A1: `K_max` tiled
  `(1+D)` trigger→readout blocks) + the work registers (op_A/op_B/role_bank/bnd/acc/Q). STORE several facts in
  sequence: bind→bundle into `acc` (the GO step-2 chain), then write `acc` into the next free store block's complex
  weights. **Crucially, RESET the work registers between facts** (`cp_membrane_potential_v[:]=0`,
  `cp_recovery_variable_u[:]=0` — the CYCLE-168 insight) and verify EARLIER facts are unchanged (they live in
  synapses, so the reset cannot touch them). Then QUERY: fire all stored triggers → reconstruct → resident
  block-diagonal unbind of the cued role → `clean_score` → moat peak-read + first-match.
- **GO criterion:** for a K-fact store (start K=5, then K=16/32), every stored fact's cued role recalls the correct
  filler == the numpy `RFPhasorComposer` (`_encode`+`_store`+`_scan_first_match`+`unbind`) for **3 seeds × 2 D**
  (this is an exact/identity effect → parity, not distribution), AND the work-register-reset invariant holds (after
  storing fact i and resetting, facts 0..i−1 still recall correctly).
- **Anti-cheats:** (i) **store-block lesion** — zero one fact's store-block complex weights → that fact's recall
  collapses (the on-substrate store is load-bearing, not residual register state); (ii) **register-reset stress** —
  reset the work registers AFTER all stores and re-query → all K facts STILL recall (proves the facts are in
  synapses, not register state — the GAP-A guarantee); (iii) **the moat battery** — an absent-cue query peak-reads
  BELOW threshold → abstain (the moat cannot be weakened); a present-cue → answer.
- **If NEGATIVE:** if recall degrades as K grows → it is phase-coherence/cross-talk in the K-block
  reconstruct→unbind→score chain (the §5 store-capacity risk). Fallback: a per-block settle micro-schedule (GPU-side,
  no host hop), or cap K per persistent bridge + shard (multi-bridge, the validated 320-concept scaling route). If
  the register-reset invariant fails → a store-block is accidentally aliasing a work register (a layout/index bug) →
  fix the slice disjointness. The host-orchestrated `self.kb` numpy store stays the default until A1 is GO.

**STEP A2 — the PARSER front-end drives the operand registers synaptically (GAP B).** Append the `BridgeParser` slice;
wire `role_route_<R>` gates from the parser's role ensembles (`couple_gate_to_indices`, the `hear_synaptic`
precedent); drive `role_bank`/`op_B` from the parser's firing (sub-option B-ii first: a gate-open-triggered masked
`rf_kick` of `role_bank` to role R's phasor; then B-i, the all-synaptic role-source register). GO = `hear(sentence)`
stores the comprehended fact into the persistent A1 store with NO host `{role:word}` dict, and the stored fact queries
back == the host. Anti-cheat: the parser-route lesion collapses comprehension (no store); a permuted parser→role
mapping stores the wrong roles.

**STEP A3 — the full `OneBrainComposer` class + the agent capability matrix.** Wrap A1+A2 as a `RFPhasorComposer`
sibling exposing `store`/`query_patient`/`query_agent`/`ask_yes_no`/moat; swap it into `BrainConversationalAgent` via
`composer_kind="onebrain"`. GO = the full `tests/test_brain_conversational_agent.py` + `test_rf_*` suites pass on the
persistent agent == the numpy oracle, moat intact (the three `is None` no-confab assertions). Anti-cheat = the
existing suite's moat assertions + a co-residence check (the parser/store/cleanup slices are byte-isolated).

**STEP A4 (OPTIONAL biologization) — the spiking WTA selection on the same bridge.** Only if the fully-spiking
selection is wanted (the host argmax is already brain-based-compliant, §1.2 GAP C). Fold `_spiking_cleanup` stage 2 in
as a co-resident Izhikevich `clean_wta` slice driven by the `clean_score` membranes. This requires the
**RF-membrane → Izhikevich-current coupling** (the flagged hard sub-problem, §1.2) — a likely `sim/` edit adding a
path from an RF score-neuron's `re` to an Izhikevich neuron's `cp_external_input_current` in the step loop. GO =
selection == host argmax, moat intact. If NEGATIVE → the host argmax read-out stays (legitimate).

**STEP A5 — make it the default + megakernel the persistent loop + retire numpy from the runtime.** Run the persistent
loop's resonate windows through `enable_rf_cudagraph` (the within-window fusion, `bridge.py:5558`); measure a full
`what_does` turn end-to-end vs the per-op path (target from the profile: ~0.8 s/turn → tens of ms). GO =
answer-identical AND real-time-grade latency. Then deprecate-then-retire the numpy production paths (the per-op
orchestration, the `self.kb` list, the numpy cleanup/moat), keeping numpy as the test oracle.

---

## 5. Honest risks + fallbacks

| # | risk | mitigation / fallback | verified? |
|---|---|---|---|
| A | **Phase coherence as the chain lengthens to many facts.** The K-block reconstruct→unbind→score is a 3-op chain per block, K blocks resonate together; phase can degrade. | Multi-window settle (proven to 5 ops at 1.000, step-3a). Fallback: per-block settle micro-schedule (GPU-side, no host hop); or cap K per persistent bridge + SHARD across bridges (the validated 320-concept multi-bridge scaling route). | settle proven; sharding precedent exists |
| B | **Transmission gates do NOT gate the RF complex matvec.** | CONFIRMED at `bridge.py:5526-5529`: the RF matvec `cp_rf_w_re @ re − cp_rf_w_im @ im` has NO `cp_transmission_gain` factor (that factor multiplies `cp_connections`, the Izhikevich matrix, at `:5773-5776`). So "open/close an RF route" is NOT free via the gate. **Mitigation (preferred, no edit):** sequence RF ops by which complex weights are INSTALLED (install = open, absent = closed — the natural FHRR way; this is exactly how the de-risks' multi-window settle works — `rf_set_complex_weights` REPLACES per window). The parser-gated routing (GAP B) gates the *Izhikevich* role-source → role-bank drive (where the gate DOES apply), then the role-bank-as-RF-register feeds the FHRR synapse — so the gate sits on the legitimate (Izhikevich) side. | VERIFIED in source |
| C | **The per-op RF "reset" trackers are whole-array.** `rf_kick` re-inits `cp_rf_prev_im`/`cp_rf_fired`/`cp_rf_spike_step` for the WHOLE array (`bridge.py:5481-5484`) even under a `neuron_mask` — only the `v`/`u` writes are masked (`:5474-5480`). On a persistent shared bridge, re-initialising one register's trackers would reset a still-settling register or the store's trackers. | **Flagged `sim/` edit (small, owner pre-approved):** mask the tracker re-init too (~6 lines; default `neuron_mask=None` = current behaviour; `test_rf_*` asserts bit-identity). Low risk. (The store itself is in synapses, not trackers, so the store is safe regardless; this only affects co-resident transient registers.) | VERIFIED whole-array at `:5481-5484` |
| D | **Store capacity / cross-talk at many facts (the superposition-search SNR).** Firing all K triggers reconstructs K composites; the matched-filter must still separate the right fact. FHRR superposition has a known capacity ceiling. | Start K small (5), scale to 16/32 with parity as the gate; the per-block (not summed) reconstruct keeps facts ISOLATED in separate blocks (A1's tiling — NOT a single summed superposition), so the per-fact unbind is clean and capacity is bounded by neuron count, not SNR. Fallback: shard across bridges (validated). The moat peak-read provides the abstain safety even under cross-talk. | layout choice mitigates |
| E | **The RF→Izhikevich coupling, IF the spiking WTA is built (STEP A4).** No path exists today from an RF neuron's `re` to an Izhikevich neuron's input current in the step loop. | Off the critical path: the host argmax read-out is brain-based-compliant (§1.2 GAP C). If wanted, a flagged `sim/` edit adds the coupling; if NEGATIVE, host argmax stays. | VERIFIED no such path in the two step-loop branches |
| F | **RF memory must survive Izhikevich steps.** RF and Izhikevich share `v`/`u`; one Izhikevich step destroys an idle RF phasor (the 5b KILL finding). | The `kb_store` lives in complex SYNAPSES (`cp_rf_w_re`/`cp_rf_w_im`), array-disjoint from `cp_connections` → Izhikevich steps never touch it. Only transient RF *register* `v`/`u` is at risk, and those are re-kicked per op. Design rule pinned (§2.2). Same guarantee `MergedRFComposer` relies on (`nav_conv_merged_bridge.py:843-847`). | VERIFIED array-disjoint |
| G | **dt / NMDA constraints (inherited from the dlPFC merge, only relevant if the dlPFC planner co-resides).** The dlPFC WM latch needs NMDA and survives dt=1.0 only at the genuinely-NMDA-dependent attractor weight ≈30, with a per-neuron NMDA mask isolating NMDA to the dlPFC slice; the dlPFC regime runs OU noise OFF while parser+composer run OU ON (the unified bridge toggles OU per-read). | The core `OneBrainComposer` (parser+registers+store+cleanup+moat) does NOT need the dlPFC — defer dlPFC co-residence to a later step. If/when added, reuse the unified-bridge per-region NMDA mask + per-read OU toggle (`unified_brain_bridge.py:82-120,727-732`). | mechanism exists/tested |
| H | **Bigger persistent step touches more neurons than a per-op bridge (latency shape).** | The latency profile says cost is launch-bound, not neuron-count-bound (D 128→2048 was flat), so a bigger-N persistent step stays launch-bound + the megakernel (1 launch/step) bounds it — MEASURE at STEP A5. Fallback: a masked-megakernel path (the megakernel currently bails to the loop when a mask is set, `bridge.py:5565-5567`) — a flagged `sim/` edit if the persistent+co-resident loop needs it fast. | profile measured |

**Overall fallback posture:** every step's NEGATIVE has a defined retreat (per-block settle / sharding for store
coherence; weight-install-as-gate for RF sequencing; host-moat-and-argmax stay if a neural version can't hold; the
megakernel'd per-op path is the ~17–24× latency ceiling already shipped). The biology research is NOT blocked — it
validates at small K where the per-op path's latency is tolerable — so a stalled step parks the integration without
parking the science. **The moat is preserved at every step:** abstention is the peak-read threshold (validated
6/6, measured-not-tuned midpoint) computed BEFORE any rendering; no step weakens it, and STEP A1's anti-cheat (ii)+(iii)
re-assert it under the new multi-fact store.

---

## 6. Reusable machinery (file:line + what each contributes) — all verified against source this session

| machinery | file:line | contributes |
|---|---|---|
| **the 4 GO de-risk runners** | `research/runners/_phaseB_onebrain_{register_handoff,fact_store_query,cleanup_onbridge,moat_onbridge}_derisk.py` | the proven register→register handoff, fact store+query, cleanup, moat — the chain the production class scales up |
| **`RFPhasorComposer` (the API + the FHRR ops)** | `rf_phasor_composer.py:61` | the API surface to mirror (`store`/`query_patient`/`query_agent`/`ask_yes_no`/moat); `_bind`/`_unbind_phases`/`_bundle`/`_encode` the op kernels |
| **`_store_substrate` / `_retrieve_substrate`** | `rf_phasor_composer.py:406,418` | memory-in-complex-synapses (Crawford-Eliasmith weight-store) — the kernel of the persistent `kb_store`, to be made resident + growing (GAP A) |
| **`_unbind_all_phases` / `_cleanup_all` / `_scan_first_match`** | `rf_phasor_composer.py:274,293,305` | the batched store-scan COMPUTATION (block-diagonal unbind + matched-filter cleanup + first-match) — maps onto the resident store's query routing |
| **`_spiking_cleanup` (stage 1 matched-filter + stage 2 Izhikevich WTA)** | `rf_phasor_composer.py:208,241-256` | the kernel of `clean_score` + the OPTIONAL `clean_wta`; stage-2 is the argmax-over-FIRING spiking selection (GAP C) |
| **`UnifiedBrainBridge`** | `unified_brain_bridge.py:286` | the precedent for parser+composer+dlPFC as disjoint slices on ONE bridge; the build-order discipline (fixed pops first, train last) |
| **`hear_synaptic` + `_op_synaptic` + the gate pre-warm latch** | `unified_brain_bridge.py:447,509,516-527` | THE synaptic parser→composer hand-off (parser firing → gated route → role bank, NO numpy `{role:word}` dict) + the comprehend→latch→compose timing — the GAP-B precedent |
| **`couple_gate_to_indices` / `couple_gate_to_pool` / `_apply_gate_couplings`** | `unified_brain_bridge.py:441`; `bridge.py:3085,3108` | drive a transmission gate from a control population's FIRING in-substrate (no host read) — how the parser's spikes sequence the routing (GAP B) |
| **`BridgeParser`** | `brain_conversational_agent.py:28` (`role_idx` `:54`, `role_of` `:123`) | the (position×voice)→role Hebbian parser (already neural) — reused verbatim as the parser slice |
| **`set_transmission_gate` / `cp_transmission_gain`** | `bridge.py:3059`; `:2440-2461` (gain), `:5773-5776` (applied to `cp_connections`) | the route-gating primitive — applies to the Izhikevich (role-source) side, NOT the RF matvec (risk B) |
| **`set_plasticity_gate` / `cp_plasticity_rate_gain`** | `bridge.py:3033` | freeze fixed populations on a learning bridge (the parser learns; the FHRR + store edges held at 0.0) |
| **`rf_kick(neuron_mask=)` + masked `_rf_advance_one`** | `bridge.py:5448,5512` | the RF-on-a-slice co-residence (RF + Izhikevich on one bridge, byte-isolated) — owner-approved + tested; the masked `v`/`u` writes are at `:5474-5480,5542-5547`; the WHOLE-ARRAY tracker re-init (risk C) at `:5481-5484` |
| **`rf_set_complex_weights`** | `bridge.py:5493` | install the FHRR complex synapses (sparse CSR; REPLACES per call — install-as-gate, risk B); an APPEND variant is the GAP-A/A2 flagged edit |
| **the megakernel (`enable_rf_cudagraph`) + `rf_resonate_steps`** | `bridge.py:5551,5558,5565`; `config.py` (`enable_rf_cudagraph`) | the within-window fusion (1 CUDA launch/step) for the persistent loop's resonate windows (STEP A5); bails to the loop under a neuron mask (`:5565-5567` — risk H) |
| **`MergedRFComposer`** | `nav_conv_merged_bridge.py:832` (`_resonate` `:863`; the store-safe array-disjointness `:843-847`) | the existing co-resident RF composer (proves RF ops on a merged slice); its per-op `_resonate` is what the persistent loop replaces |
| **the familiarity gate (the validated neural moat)** | `ordered_position_wm.py:113-131` (`_match_strength`, `read_slot`); `2026-06-11-familiarity-gate-v320-GO.md` | the validated abstain-by-match-strength mechanism — the production `fam_read` if a neural (vs host peak-read) moat is wanted |
| **`inject_explicit_wiring`** | `bridge.py:2273` | the wholesale wiring entry point all slice layouts build on |
| **the latency profile + CUDA-graph adoption** | `2026-06-17-scaling-profile-3090-latency-is-the-wall-not-vram.md`; `2026-06-17-rf-megakernel-resonate-GO.md` | proves the resonate loop is ~98% of cost and the megakernel gives ~17–24× (answer-identical, adopted) — the persistent loop's per-window cost is bounded |

---

### Bottom line
The four GO de-risks prove the conversational CORE (bind→bundle→store→unbind→cleanup→moat) runs register→register on
ONE persistent bridge with zero host round-trips. The production `OneBrainComposer` (an `RFPhasorComposer` API-sibling)
needs three things the de-risks left open: (A) a MULTI-FACT store that lives in PLASTIC COMPLEX SYNAPSES on the same
persistent bridge (so a work-register reset can't erase it) and is queried by an on-substrate superposition-scan; (B)
a PARSER front-end that routes each comprehended word's concept code into the correct operand register via the
parser's role firing + gated routes (the `hear_synaptic` precedent); and (C) the selection read-out, where the host
argmax over the on-bridge concept-membrane scores is already brain-based-compliant (the spiking Izhikevich WTA is an
OPTIONAL later biologization needing a new RF-membrane→Izhikevich-current coupling — the one flagged hard sub-problem,
off the critical path). The recommended FIRST cheap-first de-risk is **GAP A — the multi-fact persistent synapse-store
on one bridge** (smaller, parser-free, reuses the validated `_store_substrate`): build a pre-allocated K_max
trigger→readout store region, store K facts, RESET the work registers between facts, and query each back — GO =
per-fact recall == the numpy oracle (3 seeds × 2 D), the work-register-reset invariant holds (earlier facts unchanged),
with anti-cheats = store-block lesion (recall collapses), register-reset stress (all K still recall → facts are in
synapses), and the moat battery (absent cue abstains). The top risk is phase coherence / store cross-talk as K grows
(mitigations: per-block isolated tiling keeps facts separate, multi-window settle proven to 5 ops, shard across bridges
as the fallback). Two small default-preserving `sim/` edits are likely (mask the RF spike-tracker re-init; an RF-CSR
append for A2 scaling); one real constraint to design around (transmission gates don't gate the RF matvec → sequence
RF ops by weight-install, gate only the Izhikevich role-source side). numpy stays the test oracle throughout; the moat
is preserved at every step.
