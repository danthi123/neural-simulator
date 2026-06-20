# Tier-2 HEADLINE build — the PERSISTENT INTEGRATED SPIKING LOOP: architecture scoping (2026-06-19)

**Type:** read-only deep-research + architecture scoping. NO code edited, NO experiments run. One design document.
**Decision this scopes:** the owner's "real one brain" — turn the whole conversational pipeline into ONE persistent,
interacting spiking loop where the operations hand off **as spikes through synapses with NO host round-trips between
them**, AND the *control flow that sequences the operations* lives ON the substrate (a spiking sequencer replacing the
Python orchestrator). Roadmap memory `project_one_brain_integrated_pipeline_and_cleanup`.
**Audience:** the controller (owner). Plain language; every term defined once; no undefined acronyms.

> **Relationship to the 2026-06-18 scoping doc** (`2026-06-18-one-brain-integrated-pipeline-scoping.md`). That doc
> traced the host round-trips against the *inner* `RFPhasorComposer` and proposed a 6-step build whose new primitive
> was "register→register synaptic phase hand-off." **That primitive is now GO** (`2026-06-18-one-brain-register-handoff-GO.md`,
> 6/6). This doc supersedes it on two fronts the prior treatment left thin: (1) it re-maps the host round-trips against
> the **actual production composer, `OneBrainComposer`** (the agent's `composer_kind="onebrain"`, the 320-demo default
> as of CYCLE 190), which is a different, *more advanced* code path than the inner `RFPhasorComposer` the prior doc
> traced; and (2) it makes **the on-substrate SEQUENCER** — the spiking control-flow that decides which op runs next
> and routes the matched fact, the part the owner's framing calls out as "host orchestration is doing real work" — the
> centre of the design, naming a concrete biological mechanism (Eliasmith's Spaun neural production system /
> basal-ganglia–thalamocortical routing) that the codebase has *already half-built* but not yet used as a sequencer.

---

## Terms (defined once)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated neurons with one step loop.
- **op** — one conversational operation: comprehend (parse), bind, unbind, bundle, store, retrieve/scan, cleanup,
  abstain (the no-confabulation decision), generate-word-order, dialogue-plan.
- **RF** — resonate-and-fire: the spiking neuron model the composer's algebra runs on
  (`NeuronModel.RESONATE_AND_FIRE`). Its state is a complex number `Z = re + i·im` held in the bridge's `v` (membrane)
  and `u` (recovery) arrays. A concept/role is a **phasor** vector: a phase angle in `[0,1)` per dimension. The angle,
  not the magnitude, carries the information (the magnitude-invariant first-spike readout).
- **FHRR** — Fourier Holographic Reduced Representation: the vector-symbolic algebra the composer realizes
  (bind = complex multiply = phase-add; unbind = multiply by the conjugate; bundle = sum). It runs THROUGH complex
  synapses: `rf_set_complex_weights` installs a sparse complex matrix `cp_rf_w_re` / `cp_rf_w_im`, and each step
  `_rf_advance_one` does the complex matvec `u = W·z` and adds it to the rotating state.
- **host** — Python / numpy on the CPU, plus the host↔device (CPU↔GPU) transfers it forces.
- **host round-trip** — reading spiking state OFF the bridge to the host (`rf_read_phases` → numpy phases; or
  `to_host(cp_membrane_potential_v)`), computing/routing in numpy, then driving the result BACK onto the bridge
  (`rf_kick`, or installing fresh weights with `rf_set_complex_weights`). The thing this re-architecture removes.
- **host orchestration / the orchestrator** — the Python control flow that decides *which op runs next* and routes the
  result (e.g. `for got in self._read_blocks(): if got["agent"]==agent and got["action"]==action: return ...`). This is
  distinct from a host round-trip: a round-trip moves *data*; orchestration moves *control*. The owner's framing names
  this ("host orchestration is doing real work").
- **transmission gate** — a per-synapse current multiplier in `[0,1]` (`cp_transmission_gain`) that scales a route's
  synaptic CURRENT at runtime; `set_transmission_gate(name, v)` opens/closes a pre-wired route.
- **plasticity gate** — a per-synapse weight-update multiplier (`cp_plasticity_rate_gain`); `=0.0` freezes a
  population's weights even under global learning. Protects fixed populations on a learning bridge.
- **gate↔pool coupling** — `couple_gate_to_pool` / `_apply_gate_couplings`: a transmission gate driven IN-SUBSTRATE by
  the firing rate (EMA) of a control population, with no host read. Disinhibit pool X → its activity opens route gate G.
- **the moat** — the no-confabulation abstention behaviour: a query returns `None` / `"unknown"` when no stored fact
  matches, rather than inventing one. Per `feedback_moat_not_hard_lossy_memory_ok` it is a *plus kept where free*, not
  a hard gate — but a demo must NEVER *silently weaken* it (lowering the false-accept floor is a regression).
- **WM latch** — a working-memory latch: a recurrent attractor (here NMDA-dependent) that HOLDS a routing decision
  active across a downstream read, so "comprehend → latch the route → compose" works without the route flickering shut.

---

## 1. The op-graph + the host round-trips (the map of what must be converted)

### 1.0 What is the production object, and what is already on one bridge

The production conversational agent is `BrainConversationalAgent` (`brain_conversational_agent.py:146`). Its
flagship composer is `OneBrainComposer` (`composer_kind="onebrain"`, `brain_conversational_agent.py:174`;
`consolidated_320_conversation_demo` defaults to it, CLAUDE.md CYCLE 190). **`OneBrainComposer`
(`one_brain_composer.py:87`) is already a persistent co-resident bridge**: ONE bridge built once
(`build_coresident_bridge`, `:62`), holding the parser slice `[0:P]`, the RF work registers, a **persistent
fact-store in complex synapses** (`store_conns` → `cp_rf_w_re/im`), per-block + batched Q registers, and cleanup
slices — all as disjoint index ranges. The bridge is NOT rebuilt per op (that was the `RFPhasorComposer._bridge_cache`
regime the prior doc traced; `OneBrainComposer` already left it behind). The megakernel (`enable_rf_cudagraph=True`,
`:104`) and the CSR cache (`enable_csr_cache=True`, `:112`) are both on by default.

**So "one persistent bridge" is DONE. What is NOT done is two things, and they are different:**

1. **Between-op data hand-offs are still host round-trips.** Every op ends by reading phases / membrane to numpy
   (`rf_read_phases`, `to_host(cp_membrane_potential_v)`) and the next op re-installs weights + re-kicks from a numpy
   operand. The bound product never flows register→register through a synapse *within the production query path* — even
   though that hand-off is proven possible (`2026-06-18-one-brain-register-handoff-GO.md`).
2. **The control flow that sequences the ops is host Python.** A `for`-loop over stored blocks, a string-equality
   cue-match, an early `return` — the orchestrator. There is no spiking mechanism that picks "now do unbind-agent, then
   cleanup, then test-the-match, then either answer or move to the next block."

These are the two axes of "the real one brain." Axis 1 (data hand-off) is the larger line-count but the *easier*
problem (the primitive is GO). Axis 2 (the on-substrate SEQUENCER) is smaller line-count but the *deep* problem.

### 1.1 The op-graph for the two load-bearing turns

**`hear("dog go north")` — comprehend + store** (`OneBrainComposer.hear`, `:197`):

```
parser.role_of(pos,voice)  →  bind agent  →  bind action  →  bind patient  →  bundle  →  write store block
  (Izhikevich, on-bridge)      \________________ _compose_phases (:229) ________________/      _write_block (:251)
```

**`what_does("dog","go")` — query + render** (`query_patient`, `:504` → `_read_blocks`/`_read_all_blocks`, `:354`):

```
fire ALL K triggers → unbind 4 roles (block-diagonal) → cleanup (matched filter) → argmax → host cue-match → render patient
   (batched read, one resonate over the persistent store; megakernel; cached CSRs)        \__ the orchestrator __/
```

### 1.2 Every hand-off, marked ALREADY-SYNAPTIC vs HOST-MEDIATED

Traced against `OneBrainComposer` (the production path), not the inner `RFPhasorComposer`.

| # | hand-off | where | status | what makes it so |
|---|---|---|---|---|
| H0 | parser role-firing → which bind to perform | `hear` `:201-203` | **HOST-MEDIATED (control)** | `roles=[parser.role_of(pos,voice)...]; rmap={...}` — the parser fires on the bridge, but the host READS the role assignment and builds a `{role:word}` dict, then calls `_store_fact`. |
| H1 | filler word → its phasor → `fill_i` register | `_compose_phases` `:239-246` | **HOST-MEDIATED (data)** | `kick[...] = zf` builds a numpy complex kick from `comp.concepts[word]`, then `rf_kick`. The concept code enters from the host. |
| H2 | `fill_i` → `bound_i` (the bind) | `_compose_phases` `:243,246-247` | **ALREADY-SYNAPTIC** | `binds += [(bound_i, fill_i, role_phasor)...]; rf_set_complex_weights(binds)` — the bind is a diagonal complex synapse; the product STAYS on the bridge. |
| H3 | `bound_i` → `acc` (the bundle) | `_compose_phases` `:244,248` | **ALREADY-SYNAPTIC** | `bundle += [(acc, bound_i, 1.0)...]` — unit complex synapses sum into `acc`, on-bridge. |
| H4 | `acc` phasor → the store block weights | `_compose_phases` end `:249` → `_write_block` `:251` | **HOST-MEDIATED (data)** | `rf_read_phases()[acc...]` reads `acc` to numpy, then `_write_block` installs `(trig+1+k, trig, zc[k])` into `store_conns`. **This is the bind→store hand-off, and it is a host round-trip.** |
| H5 | store block → reconstruct composite (fire trigger) | `_read_all_blocks` `:374-379` | **ALREADY-SYNAPTIC** | `kick[trigger]=1.0; rf_kick; resonate` — the store reconstructs all K composites in one resonate, on-bridge. |
| H6 | composite → unbind each role → Q register | `_read_all_blocks` `:380` (cached unbind CSR) | **ALREADY-SYNAPTIC** | the unbind is a conj-role diagonal complex synapse; product → batched Q register, on-bridge. |
| H7 | Q register → cleanup matched-filter → membrane | `_read_all_blocks` `:381` (cached clean CSR) | **ALREADY-SYNAPTIC** | the cleanup is a conj-codebook complex synapse; `Re(c_k)` lands on the cleanup neurons' membrane, on-bridge. |
| H8 | cleanup membrane → winning word (argmax) | `_decode_batched_mem` `:417-432` | **HOST-MEDIATED (data)** | `mem = to_host(cp_membrane_potential_v); np.argmax(scores)` — the WTA selection is a numpy argmax. (`RFPhasorComposer._spiking_cleanup` `:242` is the validated SPIKING-WTA replacement, but it is per-op and not wired into `OneBrainComposer`'s batched path.) |
| H9 | decoded {role:word} dicts → cue-match → answer/abstain | `_scan` `:442`, `query_patient` `:508`, `ask_yes_no` `:519` | **HOST-MEDIATED (control + moat)** | `for got in self._read_blocks(): if all(got[r]==want ...): return got[answer_role]` — **this is the orchestrator AND the moat.** Python decides which block matches, which op to run next, and whether to answer or `return None`. |
| H10 | matched block → render word order | `query_patient`/`render_fact` `:513,539` | **HOST-MEDIATED (data)** | `" ".join(words[o] for o in order)` (or `order_fn` = the spiking serial-order renderer, opt-in). The final emission. |
| H11 | (clauses) outer patient → re-kick clause → inner unbind | `_decode_clause` `:448-492` | **HOST-MEDIATED (data)** | `clause_phases = rf_read_phases()[...]; kick2[...] = _to_phasor(clause_phases)` — the intermediate clause composite is read to host and re-kicked between the two unbind hops. |
| H12 | dialogue plan → next associate | `elaborate` (agent) / `_assoc_graph` | **HOST-MEDIATED (control + data)** | the association graph is rebuilt host-side from the kb dicts; the dlPFC Control is the validated `SpikingSpreadingController` (the *spreading* is on-bridge, but the graph build + the tie-break read are host). |

**Count for a flat `what_does` turn:** the data hand-offs **inside** the algebra (H2,H3,H5,H6,H7) are already synaptic;
the **host round-trips that remain** are H1 (operand in), H4 (bind→store), H8 (cleanup argmax), H10 (render), and — the
one the owner's framing targets — **H9, the orchestrator + moat**. A clause turn adds H11. The latency profile
(`2026-06-19-latency-cudagraph-arc-scoping.md`) shows the resonate is now ~3–40% of a query and the dominant residual
is the per-op CSR *rebuild* (H1/H6/H7's weight construction) — which the CSR cache (already in `OneBrainComposer`) and
this integration both attack by making the operators **resident**.

**Why this map differs from the prior doc.** The 2026-06-18 doc counted "3 host round-trips" against
`RFPhasorComposer`'s per-op-bridge path. `OneBrainComposer` already made H2,H3,H5,H6,H7 synaptic and the store
persistent — so the *data* problem is mostly solved already inside the algebra. **The residual is concentrated in H4
(bind→store), H8 (argmax), and H9 (the orchestrator+moat) — and H9 is the deep one.**

---

## 2. The mechanism (the crux): synaptic hand-off + the on-substrate sequencer

Two mechanisms, in increasing difficulty.

### 2.1 Closing the residual DATA hand-offs (H1, H4, H8, H11) — the SOLVED pattern, extended

The register→register synaptic hand-off is **GO** (`2026-06-18-one-brain-register-handoff-GO.md`): on one persistent
bridge, install the bind synapse AND the unbind synapse *together*, kick the operand once, resonate, read only the
final register — `unbind(bind(role,filler),role)` recovers the filler at 1.000 == the host two-call pipeline, with the
permuted-role and severed-route anti-cheats collapsing (0.051 / 0.077). The mechanism: the downstream complex matvec
`W·z` consumes the upstream register's **complex state** `z` (not its read-out phase), so a phasor flows
register→register through a complex synapse without ever leaving the bridge. The "two-window settle" variant (let `acc`
settle, then install the next op's synapse and run a 2nd resonate — still no host read between) is the clean default and
the mitigation for phase coherence across a longer chain.

Applying this to the residual hand-offs:
- **H4 (bind→store), the highest-value single conversion.** Today `_compose_phases` ends by reading `acc` to numpy and
  `_write_block` installs it as the store block's weights. The synaptic version: install the bundle→store synapse so
  `acc`'s settled phasor drives the store block's readout neurons, and capture the block by the *weight-write being part
  of the same resonate*, OR — the simplest cut — keep `acc` resident and have the store block READ `acc` directly when
  queried (collapse H4+H5). Either way the composite never round-trips to host.
- **H8 (cleanup argmax → spiking WTA).** The replacement EXISTS and is validated: `_spiking_cleanup` stage 2
  (`rf_phasor_composer.py:242`) drives an Izhikevich winner-take-all bank from the input-normalized matched-filter
  scores and reads **argmax-over-firing**, not argmax-over-membrane. It is == numpy argmax multi-seed
  (`2026-06-05-phase1-tpam-cleanup-derisk-GO.md`). The integration wires this bank as a **resident** slice driven by
  H7's cleanup membrane, so the winner is a spiking event, not a host argmax.
- **H11 (clause re-kick).** The two unbind hops can chain register→register (the GO primitive) with a two-window settle
  between them, the same way H2→H3 already chain — removing the `rf_read_phases`+`kick2` between hops.

These are extensions of a proven primitive. The honest hard part is **phase coherence over the longest chain** (bind ×3
→ bundle → store → unbind → cleanup in one persistent loop); the two-window settle is the mitigation and a phase-latch
the fallback (§5).

### 2.2 The on-substrate SEQUENCER (H9, H0, H12) — the genuinely deep part

This is the crux the owner's framing names, and the part the prior doc under-specified. The data hand-offs move a
phasor through a synapse; the **sequencer** must move *control*: decide which op fires next, route the matched block to
the answer register, and gate answer-vs-abstain — all of which `_scan`'s Python `for/if/return` does today.

**The biological mechanism: a neural production system on a basal-ganglia–thalamocortical routing fabric (Eliasmith,
Spaun).** Spaun ("Semantic Pointer Architecture Unified Network", Eliasmith et al. 2012) sequences its cognitive ops
with **no host orchestrator**: its action-selection is a *neural production system* implementing rules of the form
**"IF cortical area X matches vector `a`, THEN route the vector from area X4 to area X5"** — and ALL the control-like
steps (compare-with, infer, route information) are carried by a biologically-plausible basal-ganglia (BG) model. The
rules become fixed synaptic weights between cortex, BG, and thalamus; the BG output **disinhibits** the thalamic
channel for the selected action, which **opens a cortical route**, which performs the next op. This is exactly the
"control as spikes" the integrated loop needs (Stewart-Choo-Eliasmith 2012; the SPA action-selection = the BG winner-
take-all over rule-utilities; the routing = thalamic disinhibition opening cortical gates).

**The deeper motor-sequencing mechanism (the autonomous transitions): Logiaco-Abbott-Escola 2021.** BG output silences
some thalamic units and disinhibits others; a *small* set of disinhibited thalamic neurons **controls cortical
dynamics** to produce a specific motif, and a **"preparatory" thalamocortical network produces fast transitions between
any pair of learned motifs**. This is the biology of stringing ops into a sequence without an external loop — the
"which op next" transitions are themselves a learned thalamocortical trajectory.

**The codebase has already BUILT the routing fabric — but not yet used it as a sequencer.** Two runners are the
load-bearing precedent:
- `gated_compose_bg_demo.py` (`build_bg_gated_bridge` + `bind_via_bg` + `couple_all_route_gates`): the BG disinhibits a
  thalamic gate-control pool `thal_X_Y`; that thalamic ACTIVITY opens the cortical route gate `g_X_Y` **in-substrate**
  via `couple_gate_to_pool` (no runner read). Binding flows BG-disinhibition → thalamic activity → cortical route gate
  → routing — exactly Spaun's IF-match-THEN-route, validated 3-seed on real spikes.
- `gated_sequence_demo.py` (`produce_sequence`): the BG steps through a PLAN, disinhibiting one thalamic pool at a time,
  producing an *ordered* motor sequence with temporal variable binding (the same verb re-binds to a different motor at
  a later position — impossible for grown weights). **Its honest-scope note (`gated_sequence_demo.py:13-16`) is the
  exact gap this arc closes:** "the SEQUENCER here is an external plan-loop (the BG selection order is given);
  autonomous cortical sequence generation with preparatory transitions (Logiaco-Abbott-Escola Option C) is the further
  build."

So the on-substrate sequencer mechanism is concrete and partly-built:

> **A fixed library of "op rules" wired as BG/thalamocortical channels.** Each conversational op (unbind-agent,
> unbind-action, cleanup, test-match, render, advance-to-next-block) is a thalamic gate-control pool whose
> disinhibition opens that op's route. A small spiking **sequencer** (the production system) holds the current
> "program state" in a WM latch (the dlPFC NMDA attractor pattern, already on the bridge) and, conditioned on the
> *result* of the current op (e.g. the cleanup WTA winner, or the familiarity gate's answer/abstain), disinhibits the
> next op's channel. The condition→action mapping is the IF-match-THEN-route rule, realized as BG utility weights — the
> BG winner-take-all selects the next op the way the project's nav BG cascade already selects an action
> (`g11_bg_runner`, catalog A, the closed action-selection loop). The "test-match" op IS the moat: its rule is "IF the
> cleanup winner's role-words == the cue AND the familiarity gate fires `answer`, THEN route the patient register to
> the output channel; ELSE disinhibit the advance-to-next-block channel; if no block remains, fire abstain."

This reuses three things the project already validated: (a) the BG action-selection cascade (nav, catalog A), now
selecting *cognitive ops* instead of motor directions (= Spaun's claim that BG action-selection IS cognitive control);
(b) `couple_gate_to_pool` as the disinhibition→route primitive; (c) the dlPFC NMDA WM latch as the program-state
register and the familiarity gate as the answer/abstain decision. **The sequencer = "the nav BG cascade, but the
actions are ops, and the result of each op conditions the next selection."**

**The honest framing of what's genuinely new.** The data hand-off is GO; the routing fabric is built; the BG selector
is built. What is NOT yet built or validated is **conditioning the next BG selection on the RESULT of the current op**
(the cleanup winner / the moat decision) — i.e. the *closed loop* where an op's spiking output drives the sequencer's
next choice. In `gated_sequence_demo` the plan is host-given precisely because this closed loop is the unbuilt piece.
That closed loop — result-conditioned op-selection — is the deep part of this whole arc.

---

## 3. Reuse-vs-new + the likely sim/ edits

### 3.1 What transfers (reuse-by-import)

| machinery | file:line | contributes |
|---|---|---|
| `OneBrainComposer` (the persistent co-resident bridge + store-in-synapses + cached operators) | `one_brain_composer.py:87` | the substrate is ALREADY persistent; H2/H3/H5/H6/H7 already synaptic; the megakernel + CSR cache on by default. The starting point, not a rebuild. |
| register→register synaptic hand-off (the GO primitive) | `2026-06-18-one-brain-register-handoff-GO.md`; `_phaseB_onebrain_register_handoff_derisk.py` | the proven mechanism for H4/H11 (data hand-off with no host read). |
| `_spiking_cleanup` (matched filter + Izhikevich WTA) | `rf_phasor_composer.py:242` | the validated spiking replacement for H8 (argmax → spiking winner-take-all). |
| `gated_compose_bg_demo` (`build_bg_gated_bridge`, `bind_via_bg`, `couple_all_route_gates`) | `gated_compose_bg_demo.py:22,55,86` | THE on-substrate routing fabric: BG disinhibition → thalamic activity → cortical route gate, in-substrate. The sequencer's routing layer. |
| `gated_sequence_demo` (`produce_sequence`) | `gated_sequence_demo.py:26` | the BG-stepped ordered sequence with temporal re-binding; its honest-scope note is the exact gap (result-conditioned selection) to close. |
| `couple_gate_to_pool` / `couple_gate_to_indices` / `_apply_gate_couplings` | `bridge.py:3141,3164`; `unified_brain_bridge.py:123` | drive a gate from a control pool's FIRING, no host read — the disinhibition→route primitive. |
| `set_transmission_gate` / `cp_transmission_gain` | `bridge.py:3115,407` | pre-wire a route, hold it closed, open on a control signal — op-sequencing on `cp_connections` (caveat §5: NOT the RF complex matvec). |
| `hear_synaptic` + `_op_synaptic` + the WM-latch gate pre-warm | `unified_brain_bridge.py:447,509,65-79` | the validated parser→composer synaptic hand-off (H0) + how to HOLD a gate open across a downstream read ("comprehend → latch → compose"). |
| the BG action-selection cascade (nav) | `g11_bg_runner.py` (`build_bg_brain_regions`); catalog A | the winner-take-all selector to repurpose as the OP selector (Spaun's BG-as-cognitive-control). |
| the dlPFC NMDA WM latch + per-neuron NMDA mask | `unified_brain_bridge.py:82-120`; `bridge.py:6148` (`cp_nmda_neuron_mask`) | the program-state register; how to keep NMDA on the dlPFC slice only while parser/composer stay NMDA-free. |
| the learned familiarity gate (the neural moat) | `ordered_position_wm.py:120-131`; `2026-06-11-familiarity-gate-v320-GO.md` | the spiking answer-vs-abstain decision (the moat as a match-strength threshold) — the "test-match" op's decision. |
| `NeuralSerialOrderRenderer` (competitive queuing) | `neural_serial_order_renderer.py:50` | the spiking word-order emitter (H10), opt-in already. |
| `set_plasticity_gate` / `cp_plasticity_rate_gain` | `bridge.py:3089` | freeze the fixed populations (FHRR + store + dlPFC + route weights at 0.0) on the global-Hebbian parser bridge. |
| `merge_population_into_shared_bridge` | `unified_brain_bridge.py:151` | accumulate new slices into one re-injected union plan without clobbering existing wiring. |
| the masked megakernel (`rf_megastep`, `use_mask`) | `bridge.py:5640-5707` | ALREADY DONE (the prior doc flagged it as a needed edit; it shipped as A5 lever 3). The persistent loop's resonate windows run through it, masked. |
| the CSR cache (query-invariant operators built once) | `one_brain_composer.py:315,345,354` | ALREADY DONE; the resident-operator pattern this integration generalizes. |

### 3.2 The likely sim/ edits (flag each for byte-review; owner OK on justified sim/ edits)

This IS the biggest Tier-2 build, and it is the one most likely to need substantial `sim/` work. Honest list:

1. **Mask the RF spike-tracker re-init in `rf_kick`** (`bridge.py:5537-5540`). `cp_rf_prev_im`, `cp_rf_fired`,
   `cp_rf_spike_step` are re-initialised whole-array even under a neuron mask. On a persistent loop, re-kicking ONE
   register's trackers would reset a still-settling register or the store's readout. **Edit:** mask the three tracker
   writes the same way the `v`/`u` writes are already masked (`:5534-5536`). ~6 lines, default `None` = byte-identical,
   `test_rf_*` pins bit-identity. **Low risk, flagged for byte-review.** (Carried over from the prior doc; still needed.)
2. **A per-RF-synapse transmission gain (or accept weight-install-as-gate).** `cp_transmission_gain` multiplies
   `cp_connections` (the real-valued Izhikevich matrix, `bridge.py:5846`); the RF complex matvec `cp_rf_w_re/im @ z`
   (`bridge.py:5584`) has NO gain multiply. So "open/close an RF route" is NOT free via the existing gate. **Option (a),
   preferred, NO edit:** sequence RF ops by which complex weights are *installed* (install = open, absent = closed — the
   natural FHRR way; this is what `OneBrainComposer` already does op-to-op). **Option (b), an edit:** add a per-RF-synapse
   gain mirroring `cp_transmission_gain` into `_rf_advance_one` + the megakernel, so the SEQUENCER can gate RF routes the
   same way it gates Izhikevich routes. If the sequencer must open/close RF op-routes *dynamically per program-step on a
   fixed installed weight set* (the cleanest design), option (b) becomes necessary. **Flag as a real fork:** (a) avoids
   the edit but ties op-sequencing to weight-installs (a host action unless the install itself is made resident);
   (b) is a byte-reviewed `sim/` edit (~15 lines + the kernel) that makes the RF matvec gateable so the spiking sequencer
   controls it directly. This is the single biggest sim/-edit decision in the arc.
3. **The result→sequencer closed loop may need a new per-step hook** analogous to `_apply_gate_couplings`, that reads an
   op's spiking *result* (the cleanup WTA winner, the familiarity-gate firing) and disinhibits the next op's channel. If
   it can be built entirely from existing `couple_gate_to_pool` couplings (the result pool drives the next gate), NO
   edit. If the *conditional* (IF winner==cue THEN route-A ELSE route-advance) needs more than a firing-rate threshold —
   a genuine comparison/match in spikes — it needs either the Spaun-style BG utility comparison (reuse the nav cascade,
   NO sim/ edit) or a small new coupling type. **Flag: most likely NO new sim/ edit if the conditional is realized as a
   BG winner-take-all over rule-utility pools (the validated nav mechanism); flag for byte-review only if a new coupling
   primitive proves necessary.**
4. **Possibly: multiple independent RF register groups that re-init/settle on staggered schedules within one step
   loop.** The current `_rf_advance_one` advances the whole masked RF field as one phasor system; the persistent loop's
   micro-schedule (settle `acc`, then install+settle the next op) may want per-register-group masks. If the two-window
   pattern (install weights → resonate → install next → resonate) suffices (it does for the GO 2-op chain), NO edit. If
   a longer chain needs genuinely concurrent independent register groups, a sub-mask on the resonate is a possible edit.
   **Flag: likely avoidable via the two-window pattern; re-assess after the first multi-op de-risk.**

**Honest assessment of size.** Edits 1 is small and near-certain. Edit 2 (the RF gate) is the substantive fork and the
most likely real `sim/` change. Edits 3 and 4 are *probably* avoidable by reusing the BG cascade + two-window pattern,
but the sequencer is unbuilt territory and may surface one of them. **This is the biggest Tier-2 build precisely because
the sequencer (axis 2) is genuinely new — the data hand-off (axis 1) is mostly mop-up of a proven primitive.**

---

## 4. Phased arc + the cheapest-first de-risk

Ordered so each phase is independently verifiable against the numpy oracle, with an anti-cheat, and the moat never
weakened. CuPy for real runs; 6-seed for any variable effect; a numpy-equal/identity step needs only parity (the
merged-bridge byte-identity-from-3-seeds precedent applies to exact/null effects). Each phase writes a findings doc; a
NEGATIVE phase STOPS the cascade and the prior default (host orchestration for that piece) stays until GO.

> **Pick the cheapest, most load-bearing host round-trip to convert FIRST.** Two candidates: H4 (bind→store, a DATA
> hand-off) and H9 (the orchestrator, CONTROL). **H4 is the cheaper de-risk** (it is a direct extension of the GO
> register-hand-off primitive; the oracle is exact; no new mechanism). But H4 is NOT the load-bearing one — H9 (the
> sequencer) is the part that "host orchestration is doing real work." **Resolution: de-risk H4 FIRST as the cheap
> confidence-builder (it proves the data axis end-to-end on the production composer), THEN immediately de-risk the
> sequencer's smallest possible loop (Phase B) — because the sequencer is where the arc lives or dies.**

**PHASE 0 (prerequisite, mostly done — confirm, don't rebuild).** Flip the residual op-level spiking flags into the
production `OneBrainComposer` query path: route H8 through `_spiking_cleanup`'s Izhikevich WTA (it exists, opt-in in
`RFPhasorComposer`; wire it into `OneBrainComposer._decode_batched_mem`), and route the moat (H9's decision) through the
learned familiarity gate. GO = the full conversational suite (`test_one_brain_composer_agent.py` 11/11 +
`test_brain_conversational_agent.py`) passes with these on, moat intact (the `is None` no-confab assertions). This makes
every op fully-spiking BEFORE removing the host hand-offs, so any Phase-1+ regression isolates to the *integration*.

**PHASE A — CHEAP-FIRST: the bind→store DATA hand-off (H4) on `OneBrainComposer`, NO host round-trip.** The smallest
de-risk that proves a between-op hand-off works fully synaptically on the production composer.
- **The hand-off:** `_compose_phases` currently ends `rf_read_phases()[acc]` → `_write_block`. Replace with: keep `acc`
  resident and install the `acc → store-block-readout` synapse so the composite enters the persistent store WITHOUT a
  host read — OR collapse H4+H5 (the store block reads `acc` directly at query time). Build it on the existing persistent
  bridge; the store block, once written, is queried by the existing H5→H7 synaptic path.
- **GO bar:** a stored fact, written via the synaptic H4, recalls each role to the correct filler **== the current
  `_compose_phases`+`_write_block`+query path, for ≥100% of a K=8 fact set, 3 seeds × {D=64,128}** (exact/identity →
  parity not distribution). AND the no-confab moat abstains on an unstored cue (must hold).
- **Anti-cheat:** (i) **severed-route lesion** — zero the `acc→store` synapse → recall collapses (proves the on-bridge
  hand-off is load-bearing, not residual numpy state); (ii) **permuted-store** — write fact A's `acc` into block B →
  block B recalls A's fillers (proves the synaptic write carries the content); (iii) the read is SPIKING (goes through
  `rf_resonate_steps` + the cleanup), the only host op being the final argmax/render.
- **Why first:** cheapest (direct extension of the GO primitive), exact oracle, isolates the data axis on the *real*
  production object, no sequencer yet. If NEGATIVE (phase coherence over bind→bundle→store fails), the fallback is the
  two-window settle, then a phase-latch; if even that fails, the data axis is the ceiling and the sequencer is moot.

**PHASE B — THE LOAD-BEARING DE-RISK: the smallest result-conditioned op-SELECTION on the substrate (the sequencer
kernel).** This is where the arc lives or dies; do it immediately after A.
- **The minimal sequencer:** ONE conditional — "IF the cleanup WTA winner for the cue role matches the cue, THEN route
  the patient register to the output; ELSE disinhibit the next-block channel." Realize the conditional as a BG
  winner-take-all over two rule-utility pools (match → answer-route; no-match → advance-route), driven by the cleanup
  WTA firing, with the answer/advance route opened by `couple_gate_to_pool` (the validated disinhibition→gate
  primitive). The program-state (which block is current) held in a small WM latch. **Two blocks, one cue** is enough to
  prove the closed loop: block 0 doesn't match → the substrate ADVANCES to block 1 → block 1 matches → the substrate
  ANSWERS — with NO Python `for`/`if`/`return` deciding it.
- **GO bar:** the substrate selects the correct answering block AND the substrate ABSTAINS when neither block matches
  (the absent-cue case), **== the host `_scan` for a 2-block store, 6 seeds**. The moat decision (answer-vs-abstain) is
  the familiarity gate's firing, not a host string-equality.
- **Anti-cheat = the moat battery + a sequencer-lesion:** (i) an absent-cue query MUST fire `abstain` (the
  next-block-advance exhausts and the answer-route never opens) — the moat must NOT be weakened (HARD guard: false-accept
  on unstored stays ≈0); (ii) **lesion the result→sequencer coupling** → the sequencer can't advance/answer → it must
  fail SAFE (abstain), never silently confabulate; (iii) a **permuted-rule** control (wire match→advance, no-match→answer)
  must INVERT the behaviour (proves the BG selection carries the conditional, not a leak).
- **Why this is the crux:** it is the FIRST result-conditioned op-selection — the exact piece `gated_sequence_demo`
  leaves host-given. If it walls (a point-neuron BG can't reliably condition the next op on the cleanup result), the
  honest finding is "the sequencer is the substrate boundary; the megakernel'd op-at-a-time with host control flow is
  the ceiling" — a real, biology-translatable deliverable (it maps where on-substrate control flow breaks).

**PHASE C — the full who/what turn sequenced on the substrate.** Extend Phase B to K blocks + all four roles + the
patient render, so `what_does` / `who_does` / `ask_yes_no` run end-to-end with the BG sequencer driving op-selection and
the host doing only text-in / `" ".join`-out. GO = the full conversational matrix == oracle, 6 seeds, moat intact.
Anti-cheat: the parser-route lesion (H0) collapses comprehension; the sequencer-lesion fails safe.

**PHASE D — comprehension drives the loop synaptically (H0) + clauses (H11).** Wire the parser's role firing to open the
gated routes that drive the operand registers (the `hear_synaptic` precedent), and chain the clause unbind hops
register→register. GO = `hear` stores and clause queries decode == oracle, moat intact.

**PHASE E — dlPFC + order generator on the same loop (H10, H12), then make it the DEFAULT + measure.** Bring `elaborate`
and the neural renderer onto the one persistent loop; run the loop through the masked megakernel; measure a full turn
vs the op-at-a-time path. GO = answer-identical AND real-time-grade latency. This is where the cleanup payoff (numpy
exits the production runtime, stays as the test oracle) lands.

---

## 5. Honest risk + the clean stop

**The single biggest way this is harder than it looks: the on-substrate SEQUENCER (Phase B), not the data hand-off.**
The owner's framing is right — host orchestration is doing real work. The data hand-off looks like the bulk (most of
the line-count and the dual code paths), but its primitive is GO and the oracle is exact; it is *engineering*. The
sequencer is *research*: it requires an op's spiking RESULT (the cleanup winner, the familiarity decision) to condition
the next BG op-selection, **closed-loop, in spikes** — the exact piece every existing demo (`gated_sequence_demo`) left
host-given. A point-neuron BG cascade selecting *cognitive ops* (Spaun's thesis) is plausible — the project's nav BG
cascade already does result-driven action selection — but conditioning on a *match comparison* (IF winner==cue) rather
than a raw drive is the unproven step. **If a point-neuron sequencer can't hold the match-conditional reliably across K
blocks (the comparison washes out in noise, or the WM program-state drifts), the arc walls at Phase B.**

**The clean cheap-first GO vs NEGATIVE:**
- **GO** (Phase A): the bind→store synaptic hand-off recalls == the host path (parity, 3 seeds × 2 D), anti-cheats
  collapse, moat intact → the data axis is viable on the production composer; proceed to the sequencer.
- **GO** (Phase B): the substrate selects the answering block AND abstains == the host `_scan` for a 2-block store
  (6 seeds), the moat battery holds (false-accept ≈0, lesion fails safe, permuted-rule inverts) → the on-substrate
  sequencer is viable; the real one brain is reachable.
- **NEGATIVE** (Phase A): phase coherence over bind→bundle→store fails even with the two-window settle + phase-latch →
  the data hand-off is the ceiling; the megakernel'd op-at-a-time path (which already gives the validated speedups) is
  the honest production form. Deliverable: maps the phase-coherence boundary of multi-op RF chains.
- **NEGATIVE** (Phase B): the point-neuron BG can't condition op-selection on the cleanup match reliably (or it can
  only by weakening the moat — REJECT) → the sequencer is the substrate boundary; host control flow stays. Deliverable
  (high value): pinpoints exactly where on-substrate cognitive control flow breaks on point neurons — a biology-
  translatable insight directly on the top-level goal (`project_actual_goal_artificial_life_brain_analogue`).

**Other risks (bounded, with retreats):**
- The RF complex matvec is not gated by `cp_transmission_gain` (§3.2 edit 2) — the real fork. Retreat: sequence RF ops
  by weight-install (no edit) before adding the RF-gate edit.
- The dlPFC's OU-noise-OFF regime co-existing with the composer's OU-ON regime in ONE continuously-running loop is a new
  regime (the unified bridge toggles OU per-read; the persistent loop must preserve per-region regimes). Retreat: the
  per-read toggle exists and is tested; validate co-residence at Phase E.
- RF-vs-Izhikevich co-residence: the store lives in complex synapses (`cp_rf_w_re/im`), array-disjoint from
  `cp_connections`, so Izhikevich steps never touch the *memory*; only transient RF register `v`/`u` is at risk and
  those are re-kicked per op. Design rule (pins the architecture): **persistent RF memory = synapses; transient RF
  compute = masked `v`/`u`.** Low risk (the merged-bridge co-residence already relies on this).

**The unlocks (note, do NOT scope here).** A persistent integrated loop is the *prerequisite* for the emergent features
the op-at-a-time host loop can't have — graceful degradation, neuromodulatory mood (a diffuse gain shaping the whole
turn's confidence), reconsolidation-in-the-loop — already deep-researched and ranked
(`2026-06-18-emergent-one-brain-features-research.md`, graceful degradation #1). Those are downstream of this build, not
part of it.

**The clean stop posture:** every phase's NEGATIVE has a defined retreat, and the biology/conversation science is NOT
blocked by this arc (it validates at small K where op-at-a-time latency is tolerable). A stalled phase parks the
*integration* without parking the *science*. Phase A (cheap) and Phase B (load-bearing) together settle whether the
real one brain is reachable; everything after is engineering on a proven mechanism.

---

### Bottom line

`OneBrainComposer` already gives ONE persistent co-resident bridge with the FHRR algebra (bind/bundle/unbind/cleanup)
synaptic on-bridge, a persistent store in complex synapses, the megakernel and CSR cache resident — so "one persistent
bridge" is **done**. The residual host work concentrates in **H4** (bind→store, a DATA round-trip), **H8** (argmax → a
spiking WTA that already exists), and — the deep one — **H9, the ORCHESTRATOR + moat**: the Python `for/if/return` that
sequences the ops and gates answer-vs-abstain. The data hand-off is mop-up of a **GO** primitive
(`2026-06-18-one-brain-register-handoff-GO.md`). The on-substrate SEQUENCER is the genuinely new, deep part: a **neural
production system on a basal-ganglia–thalamocortical routing fabric (Spaun; Logiaco-Abbott-Escola transitions)**, of
which the routing layer is **already built** (`gated_compose_bg_demo` / `couple_gate_to_pool`) and the BG selector is
already built (the nav cascade) — the unbuilt piece is **conditioning the next op-selection on the current op's spiking
result**, exactly the gap `gated_sequence_demo` flags. The cheapest-first de-risk is **Phase A** (the bind→store
synaptic hand-off, GO = parity with the host path, 3 seeds × 2 D, anti-cheats = severed-route + permuted-store); the
load-bearing de-risk immediately after is **Phase B** (the smallest result-conditioned op-selection — a 2-block
who/what scan sequenced by the substrate, GO = == the host `_scan`, 6 seeds, the moat battery holding). Likely `sim/`
edits: masking the RF spike-tracker re-init (small, near-certain) and — the real fork — a per-RF-synapse transmission
gain so the spiking sequencer can gate RF routes directly (else sequence by weight-install). This is the biggest Tier-2
build because the sequencer (control-as-spikes) is genuinely new; an HONEST NEGATIVE at Phase B (a point-neuron BG can't
condition op-selection on a match without weakening the moat) is itself the deliverable — it maps where on-substrate
cognitive control flow breaks.

Sources: Stewart, Choo & Eliasmith (2012) "Spaun: A Perception-Cognition-Action Model Using Spiking Neurons"
(compneuro.uwaterloo.ca); the Semantic Pointer Architecture (compneuro.uwaterloo.ca/research/spa); Logiaco, Abbott &
Escola (2021) "Thalamic control of cortical dynamics in a model of flexible motor sequencing," *Cell Reports* 35(9)
(cell.com/cell-reports); catalog A (closed BG action-selection loop), G.07 (pre-SMA internally-generated sequences),
E.03 (population coding), C.14 (LC-NE inverted-U); the codebase precedents cited inline.
