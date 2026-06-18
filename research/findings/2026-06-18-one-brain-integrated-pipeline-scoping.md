# The REAL "one brain": the whole conversational pipeline as ONE persistent, interacting spiking loop — architecture scoping (2026-06-18)

**Scoping type:** read-only. No code edited, no experiments run. One design/scoping document.
**Audience:** the controller (owner). Plain language; every term defined once; no undefined acronyms.

## Terms (defined once)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated neurons with one step loop.
- **op** — one conversational operation: bind, unbind, bundle, cleanup, store, retrieve, abstain
  (the no-confabulation decision), parse, generate word-order, dialogue-plan.
- **RF** — resonate-and-fire: the spiking neuron model the composer uses (`NeuronModel.RESONATE_AND_FIRE`).
  Its state is a complex number `Z = re + i·im` stored in the bridge's `v` (membrane) and `u` (recovery)
  arrays. A concept/role is a **phasor** vector: a phase angle in `[0,1)` per dimension. The angle, not the
  magnitude, carries the information (the magnitude-invariant readout).
- **FHRR** — Fourier Holographic Reduced Representation: the vector-symbolic algebra the composer realizes
  (bind = complex multiply = phase add; unbind = multiply by the conjugate; bundle = sum). It runs THROUGH
  complex synapses: `rf_set_complex_weights` installs a sparse complex matrix `cp_rf_w_re` / `cp_rf_w_im`,
  and each step `_rf_advance_one` does the complex matvec `u = W·z` and adds it to the rotating state.
- **host** — Python / numpy code on the CPU (and the host↔device transfers it forces).
- **host round-trip** — reading spiking state OFF the bridge to the host (`rf_read_phases` →
  numpy phases), computing or routing in numpy, then driving the result BACK onto the bridge
  (`rf_kick`). The thing this re-architecture removes.
- **transmission gate** — a per-synapse current multiplier in `[0,1]` (`cp_transmission_gain`) that scales a
  route's synaptic CURRENT at runtime; `set_transmission_gate(name, v)` opens/closes a pre-wired route.
- **plasticity gate** — a per-synapse weight-update multiplier (`cp_plasticity_rate_gain`); `=0.0` freezes a
  population's weights even under global learning. Used to protect fixed populations on a learning bridge.
- **the moat** — the no-confabulation abstention behaviour: a query returns `None` / `"unknown"` when no
  stored fact matches, rather than inventing one. Must NEVER be weakened (owner: it is a plus, biologize it
  where free; it stays correct here throughout).

---

## 1. Diagnosis — the current op-at-a-time, host-orchestrated pipeline (what we are removing)

The production conversational agent (`BrainConversationalAgent`, default `composer_kind="rf"`,
`brain_conversational_agent.py:151`) delegates every fact op to `RFPhasorComposer`
(`rf_phasor_composer.py:61`). **Every op is its own host-orchestrated micro-episode** built around the same
five-line motif, `RFPhasorComposer._resonate` (`rf_phasor_composer.py:105-116`):

```
b = self._bridge_cache.get(n)        # reuse a per-OP bridge sized to THIS op's neuron count n
b.rf_set_complex_weights(conns)      # rebuild the complex synapse matrix FRESH for this op (host→device)
b.rf_kick(kick, period, lam=0.0)     # write Z=re+i·im into v/u from a numpy kick vector (host→device)
b.rf_resonate_steps(period+8)        # run the 208-step resonate loop
return np.asarray(b.rf_read_phases())# read first-spike steps back to numpy phases (device→host)
```

Each op thus: (i) selects/builds a **per-op bridge** keyed by neuron count (`self._bridge_cache`,
`:102`), (ii) **rebuilds the complex weights** (`rf_set_complex_weights`, `bridge.py:5493` — a full CSR
rebuild on the GPU), (iii) **kicks** the operand in as a numpy complex vector (`rf_kick`, `bridge.py:5448`,
host→device transfer), (iv) runs the **208-step resonate loop** (`rf_resonate_steps`, `bridge.py:5551`),
(v) **reads phases back to the host** (`rf_read_phases`, `bridge.py:5486`, device→host). The numpy result is
then handed to the next op by Python.

### Trace one `agent.what_does("dog", "go")` turn, op-by-op, naming every host boundary

`what_does` → `composer.query_patient("dog", "go")` (`rf_phasor_composer.py:446`). On the default fast path
(`enable_substrate_store=False`, `enable_spiking_cleanup=False`) it uses the **batched store scan**
(`_can_batch_scan`, `:269`). For a fact store ("knowledge base", `self.kb`) of K facts:

| step | call | host boundary |
|---|---|---|
| A | `_scan_first_match(agent="dog", action="go")` `:305` | host loop over cue roles |
| A.1 | for role `agent`: `_unbind_all_phases(comps, "agent")` `:274` | builds a **block-diagonal** kick over all K composites in numpy, ONE `_resonate` over `2·K·D` neurons → **1 host round-trip** (kick in, phases out) |
| A.2 | `_cleanup_all(rec)` `:293` | nearest-concept = a numpy complex matmul `rec_z @ conj(codebook)ᵀ` then `argmax` — **host compute** (no bridge) |
| A.3 | for role `action`: `_unbind_all_phases` + `_cleanup_all` again | **a 2nd host round-trip** + a 2nd numpy argmax |
| A.4 | `mask &= (words == cue)` then `idx[0]` `:309-314` | host string-equality match + first-index pick — **this is the moat decision** |
| B | matched index `i`; `_render(comp, "patient", fact["patient"])` `:153` | host dispatch on stored structure |
| B.1 | `_unbind_phases(comp, "patient")` `:170` | **a 3rd host round-trip** (the `_render`'s own unbind) |
| B.2 | if the patient is a concept: `_cleanup(rec)` `:258` → numpy `argmax` `:262-263` | **host compute** |
| B.3 | if the patient is an embedded clause: 3 MORE `_unbind_phases` (agent/action/patient of the inner clause) + 3 `_cleanup` `:161-167` | **3 more host round-trips** + 3 argmaxes |
| C | attribute roles: `unbind(comp, "attribute"...)` `:459` | **+1 host round-trip per attribute** |

**Count for a flat (non-clause, non-attribute) `what_does` turn with the batched scan: 3 host round-trips**
onto/off the bridge (one per unbound role: agent, action during the scan; patient during render), each
followed by a numpy cleanup, plus the host moat-equality. With an embedded-clause patient it is **6**, and
the per-op (non-batched) path is **2 round-trips per fact checked** (it blows up linearly — the latency
profile measured an abstention at K=1000 as *minutes*). The single-op latency profile
(`2026-06-17-scaling-profile-3090-latency-is-the-wall-not-vram.md`) measured **~160 ms per round-trip**, of
which **97.7% is the 208-step resonate loop** issuing ~3,000–4,000 tiny sequential GPU kernel launches.

### What specifically makes this op-at-a-time and host-mediated

1. **Per-op bridge churn.** `self._bridge_cache` holds a *separate* bridge per neuron count
   (`rf_phasor_composer.py:102,108-112`). A bind uses a `2·D`-neuron bridge; a bundle of L bound vectors
   uses `(L+1)·D`; the batched scan uses `2·K·D`. The composite is never resident anywhere persistent —
   it is recomputed from numpy each time it is needed.
2. **The between-op handoff is numpy, not synapses.** Op N reads phases to the host
   (`rf_read_phases`), op N+1 re-kicks them (`rf_kick`). The intermediate vector lives in a Python
   variable. The **megakernel fuses WITHIN an op** (`enable_rf_cudagraph`, `bridge.py:5558` — one CUDA
   launch replaces the ~15-kernel step), but it does NOTHING for the **between-op** handoff: it still ends
   in `rf_read_phases` → numpy → next op.
3. **The fact store is a numpy list.** `self.kb = []` (`:99`); `store` appends a numpy phasor array (`:333`);
   every query indexes that Python list. (The spiking weight-store `_store_substrate`/`_retrieve_substrate`,
   `:406,418`, exists but is default-off and *still per-op* — it builds a throwaway `1+D` bridge per fact.)
4. **The cleanup is numpy.** `_cleanup` (`:258-263`) is a numpy cosine + `argmax`. (The spiking
   `_spiking_cleanup`, `:208`, exists but is default-off and *still per-op* — it builds/reuses a `D+V`
   bridge per call.)
5. **The moat is host string-equality.** `_scan_first_match` (`:305-314`) and the per-fact loops
   (`:441-444`, `:461-468`, `:493-497`) decide abstention by `unbound_word == cue_word` in Python, then
   `return None`.
6. **The renderer's word-order is a host f-string** (`render_fact` `:520`, `_render` `:167`). (The neural
   competitive-queuing renderer, `enable_neural_render`, exists but default-off.)
7. **Dialogue planning builds a throwaway bridge per `elaborate` call**
   (`brain_conversational_agent.py:285`: `SpikingSpreadingController(graph, seed)` constructs a fresh
   bridge every time the graph changes).

**Even the existing "one-bridge" work does NOT close this.** `UnifiedBrainBridge` (parser+composer+dlPFC on
one bridge) and `MergedNavConvAgent` (nav+conv on one bridge) put the regions' NEURONS on one bridge as
disjoint slices — but the conversational OPS still run op-at-a-time through `RFPhasorComposer._resonate` /
`MergedRFComposer._resonate`. `MergedRFComposer._resonate` (`nav_conv_merged_bridge.py:863-880`) is the
clearest proof: it shifts the op's indices into the `rf` slice, but still does
`rf_set_complex_weights` → `rf_kick(neuron_mask=rf_mask)` → `rf_resonate_steps` → `rf_read_phases` → numpy →
next op, **per op**. The co-residence solved *which neurons* an RF op uses; it did NOT remove the host
round-trips BETWEEN ops. That removal is this arc.

---

## 2. Target architecture — the persistent co-resident interacting loop

### 2.1 The end state in two sentences

All conversational regions live as disjoint, permanently-allocated neuron-index slices on **ONE
persistent `SimulationBridge` that is never rebuilt per op**: a parser slice, a "phasor workspace" of
several RF register slices (operand-A, operand-B, role banks, accumulator/bundle, cleanup codebook,
fact-store), a cleanup winner-take-all slice, a familiarity-gate slice (the moat), a serial-order generator
slice, and the dlPFC dialogue-planning loop. A fact-store and a query flow **end-to-end as spikes through
synapses** — comprehension fires the parser, the parser opens gated routes that drive the operand registers,
the FHRR complex synapses carry bind/unbind/bundle from one register to the next WITHIN the persistent step
loop, the cleanup slice fires the winning concept, the familiarity gate decides abstain-vs-answer, and the
order generator emits the word sequence — with the host doing only I/O (text in, the body's final
`" ".join` of already-spelled, already-ordered words out).

### 2.2 Regions on the one persistent bridge

| slice | role | substrate |
|---|---|---|
| `parse_conj` (6) + `parse_role` (3·R) | comprehension (word-position×voice → role) | Izhikevich, Hebbian-learned (the existing `BridgeParser`, already neural) |
| `op_A`, `op_B` (D each) | the two FHRR operands of a bind/unbind | RF registers |
| `role_bank` (D) | the active role phasor (driven by a parser-gated route) | RF register |
| `acc` (D) | the running bundle / current composite under construction | RF register |
| `kb_store` (K·D, persistent) | the fact store: each fact a composite held in **complex synaptic weights** | RF weight-store |
| `clean_codebook` (V) + `clean_wta` (V) | matched-filter cleanup + spiking winner-take-all | RF matched filter → Izhikevich WTA |
| `fam_gate` (small) | the moat: a Bogacz-Brown familiarity/novelty threshold on match-strength | Izhikevich threshold population |
| `order_gen` | serial-order word emission (competitive queuing) | the existing `NeuralSerialOrderRenderer`, rate-coded |
| `cortex_ctx` + `dlpfc_wm` (dlPFC) | dialogue planning (spreading activation, working-memory latch) | Izhikevich + NMDA (per-neuron NMDA mask) |

These coexist because their state lives in **disjoint arrays**: the Izhikevich slices use `v`/`u` as voltage;
the RF slices use `v`/`u` as the complex phasor (the masked-RF-ops co-residence, below); the FHRR memory
lives in `cp_rf_w_re`/`cp_rf_w_im`, which are **array-disjoint** from `cp_connections` (the Izhikevich
synapse matrix). The merged-bridge work already proved Izhikevich and RF slices coexist byte-isolated
(`tests/test_rf_neuron_mask_coexistence.py`, `test_merged_rf_composer_coresident.py`).

### 2.3 How a store flows as spikes, end-to-end

`hear("dog go north")` → parser fires `agent=dog, action=go, patient=north` (already neural). For each role:
the parser's role ensemble firing **opens a transmission-gated route** (`role_route_<R>`) that drives the
`role_bank` RF register with that role's phasor, while the word's concept code drives `op_B`; the FHRR
diagonal complex synapse `role_bank ⊗ op_B → acc` performs the bind THROUGH synapses (the same complex
matvec `_bind` does today, but the operands ARRIVE synaptically and the product STAYS on the bridge). Three
binds accumulate into `acc` (bundle = unit complex synapses summing into `acc`). The composite is then
**written into the persistent `kb_store`** as a new fact's trigger→readout complex weights (the
`_store_substrate` mechanism, but appended to the ONE persistent store, not a throwaway bridge). The fact is
now memory-in-synapses; no numpy array holds it.

This is exactly the `UnifiedBrainBridge.hear_synaptic` precedent (`unified_brain_bridge.py:447-507`) —
which already routes the parser's role selection to the composer's role bank through transmission gates with
zero numpy `{role: word}` dict — **extended** so the bound product is written to a persistent store and the
operands are RF registers rather than the rate-coded coincidence banks.

### 2.4 How a query flows as spikes, end-to-end

`what_does("dog", "go")`: drive `op_A` = the cue's role-bound probe; the persistent `kb_store` complex
synapses fan the probe against every stored fact (block-diagonal, the `_unbind_all_phases` batched pattern
but resident); the unbind product drives `clean_codebook` (matched filter = the conjugate-codebook complex
synapse, already what `_spiking_cleanup` stage 1 does, `:208-236`); `clean_wta` fires the winning concept
(spiking winner-take-all, `_spiking_cleanup` stage 2, `:241-256`); the `fam_gate` reads the match strength
and **fires "answer" only if above threshold, else "abstain"** (the moat as a neural familiarity gate — the
validated `OrderedPositionWM.read_slot` mechanism, `ordered_position_wm.py:120-131`); on "answer", the
winning concept keys `order_gen` to emit the word. The host does only the final `" ".join` (the body).

### 2.5 The synaptic hand-off mechanism between ops (the crux) — and the honest hard parts

**(a) RF phasor state lives in `v`/`u` — CONFIRMED solved for co-residence.** `rf_kick(neuron_mask=)`
(`bridge.py:5448,5471-5480`) and `_rf_advance_one`'s masked write-back (`bridge.py:5537-5548`) already
restrict RF reads/writes to a slice, leaving co-resident Izhikevich slices' `v`/`u` byte-untouched. The
remaining new requirement: **multiple RF register slices that DON'T clobber each other within one step.** The
current `_rf_advance_one` rotates/advances the *whole* masked set as one phasor field — fine, because all RF
registers can advance together in one resonate window (the complex matvec `W·z` routes between them by the
sparse synapse structure). The op-to-op handoff becomes "which complex synapses are installed", not a host
re-kick.

**(b) Routing one op's phase output to the next op's input SYNAPTICALLY (not via
`rf_read_phases`→host→`rf_kick`).** This is the heart of the re-architecture and the genuinely new piece.
Today op N's output is read to numpy and op N+1 re-kicks it. In the target, op N's output register IS op
N+1's input register, connected by an FHRR complex synapse, so the value never leaves the bridge. The bind
chain `op_B —(role_bank diagonal)→ acc —(unit)→ kb_store` is a single resonate over the persistent bridge
with the right complex weights installed. **Honest hard part:** the RF readout is a *first-spike-time* code
(`rf_read_phases` recovers phase from the step at which Im crosses zero, `bridge.py:5486-5491`). For a phasor
to drive the NEXT register synaptically, the downstream complex matvec must consume the *complex state* `z`,
which IS what `_rf_advance_one` does (`W·z`, not `W·phase`) — so chaining register→register through complex
synapses is natively supported AS LONG AS both registers resonate in the same window. The risk is **phase
coherence across a multi-op chain in one resonate window**: each register must settle before the next reads
it, or use a staged set of complex-weight installs within the persistent step loop (a "micro-schedule" that
is GPU-side, not host round-trips). The first de-risk (§4) tests exactly this for the shortest chain.

**(c) Keeping the per-op "reset" from disturbing other regions.** Today each op begins with `rf_kick`
(re-initialising `v`/`u`, `cp_rf_prev_im`, `cp_rf_fired`, `cp_rf_spike_step`, `bridge.py:5481-5484`) — a
clean slate. On a persistent shared bridge, a register must be re-initialised WITHOUT touching other
registers. The masked `rf_kick(neuron_mask=)` already does the masked-write part; the spike trackers
(`cp_rf_fired`, `cp_rf_spike_step`, `cp_rf_prev_im`) are currently whole-array (`bridge.py:5481-5484`,
non-masked) — **this is a likely small `sim/` edit: mask the tracker re-init too**, so re-initialising
register R's trackers doesn't reset a still-settling register or the persistent store's. Flag: ~6 lines in
`rf_kick`, byte-reviewed, default-preserving (mask `None` = current behaviour).

**(d) The cleanup / store / moat as PERSISTENT regions vs rebuilt per op.** Today even the opt-in spiking
versions rebuild a bridge per op (`_spiking_cleanup` reuses a `D+V` bridge but re-installs weights each
call; `_store_substrate` builds a fresh `1+D` bridge per fact). In the target, `clean_codebook`/`clean_wta`,
`fam_gate`, and `kb_store` are **wired once at construction** and driven by routes — the codebook complex
synapses are fixed (the conjugate-codebook matched filter), the WTA is fixed Izhikevich, the store GROWS
(append a fact = install that fact's complex weights into the persistent `kb_store` block; no rebuild of
anything else). **Honest hard part:** a growing store on a persistent bridge means either pre-allocating a
max-K block (simple; wastes some capacity) or appending to the complex CSR (the `set_pathway_weights(add_missing=True)`
CSR-rebuild hazard the unified bridge documents at `unified_brain_bridge.py:106-116` — a rebuild resorts the
matrix and stales gate-index maps). The RF complex weights are a SEPARATE CSR (`cp_rf_w_re/im`), so the
hazard is contained to the RF store, but the pre-allocate-max-K route is the safer first cut.

**The honest framing of the "exactness".** This re-architecture changes WHERE the ops run (persistent
on-bridge, synaptic handoff) — it does NOT change WHAT they compute. The bind/unbind stay the exact-inverse
FHRR algebra (the principled Eliasmith Semantic-Pointer idealization, audited in
`2026-06-18-conversational-brain-based-only-audit.md` row 2). De-idealizing the algebra into a *learned*
binder is the SEPARATE learned-binder arc (`OnBridgeLearnedComposer`); it is orthogonal to and compatible
with this integration (a learned binder is also a set of synapses on the persistent bridge).

---

## 3. Reusable machinery (file:line + what each contributes)

| machinery | file:line | contributes |
|---|---|---|
| **`UnifiedBrainBridge`** | `unified_brain_bridge.py:286` | the precedent for parser+composer+dlPFC as disjoint slices on ONE bridge; the build-order discipline (fixed pops first, train last) |
| **`hear_synaptic` + `_op_synaptic`** | `unified_brain_bridge.py:447,509` | THE synaptic parser→composer hand-off precedent: parser firing → gated route → composer role bank, **no numpy `{role:word}` dict**. The pattern the whole pipeline generalizes. |
| **the working-memory latch / gate pre-warm** | `unified_brain_bridge.py:65-79,509-592` | how to hold a gate open across a downstream read ("comprehend → latch → compose") — the timing discipline for chaining ops on one bridge |
| **transmission gates** | `bridge.py:3059` (`set_transmission_gate`), `:2440-2461` (`cp_transmission_gain`) | pre-wire a route, hold it closed, open it on a control signal — the op-sequencing primitive (caveat in §6: gates `cp_connections`, NOT the RF complex matvec) |
| **`couple_gate_to_pool` / `couple_gate_to_indices` / `_apply_gate_couplings`** | `bridge.py:3085,3108`; `unified_brain_bridge.py:123` | drive a gate from a control population's FIRING, in-substrate (no host read) — how the parser's spikes sequence the composer |
| **`set_plasticity_gate` / `cp_plasticity_rate_gain`** | `bridge.py:3033` | freeze fixed populations on a learning bridge (the parser learns; the FHRR + store + dlPFC edges held at 0.0) |
| **the megakernel (`enable_rf_cudagraph`) + `rf_resonate_steps`** | `bridge.py:5551,5558,5605` | the WITHIN-op fusion (1 CUDA launch/step); the persistent loop's resonate windows run through it for the ~11–100× per-window speedup |
| **`rf_kick(neuron_mask=)` + masked `_rf_advance_one`** | `bridge.py:5448,5512` | the RF-on-a-slice co-residence (RF and Izhikevich on one bridge, byte-isolated) — already owner-approved + tested |
| **`MergedRFComposer`** | `nav_conv_merged_bridge.py:832` | the existing co-resident RF composer (proves RF ops on a merged slice); the thing whose per-op `_resonate` we replace with the persistent loop |
| **`merge_population_into_shared_bridge`** | `unified_brain_bridge.py:151` | accumulate populations into one re-injected union plan (a runner helper, NOT a bridge method) — how new slices are added without clobbering existing wiring |
| **`_store_substrate` / `_retrieve_substrate`** | `rf_phasor_composer.py:406,418` | memory-in-complex-synapses (Crawford-Eliasmith weight-store) — the kernel of the persistent `kb_store`, to be made resident + growing |
| **`_spiking_cleanup`** | `rf_phasor_composer.py:208` | matched-filter (complex synapse) + Izhikevich WTA — the kernel of the persistent `clean_codebook`/`clean_wta` |
| **the familiarity gate** | `ordered_position_wm.py:120-131`; `2026-06-11-familiarity-gate-v320-GO.md` | the validated neural moat (match-strength threshold → abstain) — the persistent `fam_gate` |
| **`NeuralSerialOrderRenderer`** | `neural_serial_order_renderer.py:50` | the spiking word-order emitter (competitive queuing) — the persistent `order_gen` |
| **`SpikingSpreadingController` + `_SharedDlpfcContext`** | `content_selection_spiking.py:315,371`; `unified_brain_bridge.py:240` | the dlPFC dialogue planner on a SHARED slice (the unified bridge already runs it without a throwaway bridge) |
| **`inject_explicit_wiring`** | `bridge.py:2273` | the wholesale wiring entry point all of the above build on |
| **the latency profile + CUDA-graph prototype** | `2026-06-17-scaling-profile-3090-latency-is-the-wall-not-vram.md`; `_phaseB_resonate_cudagraph_prototype.py` | proves the resonate loop is 98% of cost and the graph fix gives 11× (measured) → the persistent loop's per-window cost is bounded |

---

## 4. Incremental build plan (cheap-first, each a GO/NEGATIVE de-risk; moat preserved throughout)

Ordered so each step is individually verifiable against the numpy oracle, with an anti-cheat, and the moat
never weakened. CuPy for the real runs; 6-seed for any variable effect; a numpy-equal step needs only
parity (the merged-bridge precedent for byte-identity from 3 seeds applies to exact/null effects).

**STEP 0 (prerequisite, already largely done — confirm, don't rebuild).** Flip the four biologization flags
ON in the production agent (`enable_spiking_cleanup`, `enable_substrate_store`, `enable_neural_render`, and
route the moat through the familiarity gate) per the audit's "flip-the-flag" recommendations
(`2026-06-18-conversational-brain-based-only-audit.md`). This makes every op individually fully-spiking
*before* removing the host handoffs — so a regression in STEP 1+ is isolated to the integration, not the
op's neural-ness. GO = the full conversational suite (`tests/test_brain_conversational_agent.py` +
`test_rf_*`) passes with all flags on, moat intact (the three `is None` assertions).

**STEP 1 — FIRST CHEAP-FIRST DE-RISK: two adjacent RF ops hand off synaptically with NO host round-trip.**
- **The two ops:** `bind` then `unbind` of the SAME role — i.e. `unbind(bind(role, filler), role)` should
  recover `filler`. Concretely: install, on ONE persistent bridge, `op_B (filler) —(role diagonal)→ acc`
  (bind) AND `acc —(conj-role diagonal)→ op_out` (unbind), kick `op_B` once, run ONE resonate window, read
  `op_out`'s phases. **No `rf_read_phases` between the bind and the unbind** — the bound composite stays on
  the bridge as `acc`'s phasor state and feeds the unbind synapse directly.
- **The test:** `op_out` cleans up to the original `filler` for all V vocabulary fillers and a fixed role.
- **GO criterion:** recovery == the current two-call `_bind`→`_unbind_phases`→`_cleanup` host pipeline for
  **≥ V/V (100%) of fillers, 3 seeds × 2 D** (this is an exact/identity effect, so parity not distribution).
- **Anti-cheat:** (i) a **permuted-role** unbind (unbind with the WRONG conjugate role) must FAIL to recover
  (collapses to chance) — proves the synaptic route carries the role binding, not a leak; (ii) a
  **severed-route** lesion (zero the bind→acc complex synapse) must collapse recovery — proves the on-bridge
  handoff is load-bearing, not the residual kick.
- **Why this first:** it isolates exactly the new primitive (register→register synaptic phase handoff in one
  resonate window, hard-part §2.5b) at the smallest scale, against a known oracle, with no store/parser/moat
  involved. If NEGATIVE, the fallback (§6) is a staged GPU-side micro-schedule (settle `acc`, then a 2nd
  resonate window for the unbind, still no host round-trip) before declaring the synaptic handoff infeasible.

**STEP 2 — three-op bind→bundle→store on the persistent bridge.** Extend STEP 1: bind three role-filler
pairs and bundle them into `acc`, then write `acc` into a persistent `kb_store` block (the resident
`_store_substrate`). GO = the stored composite, read back by firing its trigger, unbinds each role to the
correct filler == the numpy `_encode`+`_store`. Anti-cheat: a store-block lesion collapses retrieval; a
permuted-store (wrong fact's weights) retrieves the wrong fillers.

**STEP 3 — query: probe → resident store → cleanup → moat, all on-bridge.** Drive a cue probe against the
persistent `kb_store` (block-diagonal unbind, the resident `_unbind_all_phases`), into the persistent
`clean_codebook`/`clean_wta`, into the `fam_gate`. GO = `query_patient` / `query_agent` answers AND abstains
== the numpy oracle for a K-fact store, 6 seeds. **Anti-cheat = the moat battery:** an absent-cue query must
fire "abstain" (the `fam_gate` below threshold), a present-cue must fire "answer"; a lesioned `fam_gate`
must NOT silently accept (the moat cannot be weakened — if the neural gate can't hold abstention, this step
is NEGATIVE and the host moat stays until it can).

**STEP 4 — comprehension drives the query/store synaptically (full parser→pipeline).** Wire the parser's
role firing to open the gated routes that drive `op_B`/`role_bank` (the `hear_synaptic` precedent), so
`hear(sentence)` stores and `what_does(...)` queries with the parse arriving as spikes. GO = the full
`test_brain_conversational_agent.py` suite passes on the persistent agent, moat intact. Anti-cheat: the
parser-route lesion collapses comprehension (no store).

**STEP 5 — the dlPFC + order generator on the same persistent bridge.** Bring `cortex_ctx`/`dlpfc_wm`
(`elaborate`) and `order_gen` (`describe` word order) onto the one bridge (the unified-bridge dlPFC pattern;
the renderer is rate-coded and small). GO = `elaborate` and neural-`describe` == oracle, moat intact.

**STEP 6 — make it the default + megakernel the persistent loop.** Run the persistent loop's resonate
windows through `enable_rf_cudagraph` (the within-window fusion) and measure a full `what_does` turn
end-to-end vs the current op-at-a-time path. GO = answer-identical AND a real-time-grade latency (target
from the profile: ~0.8 s/turn → tens of ms). This is where the cleanup payoff (§5) is unlocked.

Each step writes a findings doc; a NEGATIVE step stops the cascade and the prior default (host orchestration
for that op) stays until the step is GO.

---

## 5. The cleanup payoff — what can RETIRE once this lands (numpy stays as the TEST ORACLE)

Per `project_one_brain_integrated_pipeline_and_cleanup`: once the persistent fully-spiking pipeline is the
DEFAULT, numpy leaves the **production runtime** but stays in **tests**. Deprecate-then-retire, not big-bang.
Retirement candidates, by confidence:

**High confidence (the dual code paths the persistent loop subsumes):**
- The per-op host orchestration in `RFPhasorComposer`: the numpy fast paths in `_cleanup`/`_cleanup_all`
  (`rf_phasor_composer.py:258-303`), the numpy `kb` list + `store` append (`:99,333`), the host
  `_scan_first_match` moat equality (`:305`), the host f-string in `_render`/`render_fact` (`:167,520`), and
  the `self._bridge_cache` per-op-bridge machinery (`:102,108`). All become the persistent on-bridge regions
  → their numpy bodies move to tests as the oracle.
- The opt-in flag scaffolding once the spiking path is the only path: `enable_spiking_cleanup`,
  `enable_substrate_store`, `enable_neural_render`, `enable_learned_assoc`, `composer_kind` branch
  (`brain_conversational_agent.py:151-208`) collapse to one code path.
- `_store_substrate`/`_retrieve_substrate` and `_spiking_cleanup` **as per-op constructs**
  (`rf_phasor_composer.py:208,406,418`) — superseded by their resident versions (the mechanism is reused,
  the throwaway-bridge-per-op form retires).

**Medium confidence (the legacy / reference paths the audit + CLAUDE.md already mark as non-production):**
- The **rate-coded `CoreSimComposer`** legacy composer (the ±1 Hadamard, opponency-bounded) — already an
  explicit opt-in (`composer_kind="rate"`), explicitly superseded by the RF default (CLAUDE.md "OPPONENCY
  ESCAPED"). Retire from production; keep as oracle.
- The **reference-only standalone numpy phasor simulators** `spiking_phasor_fhrr.py` +
  `resonate_fire_fhrr.py` (and the unified agents that import them: `nested_composition_agent`,
  `spiking_unified_agent`, `unified_agent_*`) — already carry a NUMPY-REFERENCE header (CLAUDE.md). They are
  the FHRR validation ceiling → exactly "keep as test oracle, out of production".

**Lower confidence (orchestration shims that the persistent bridge replaces architecturally):**
- `UnifiedBrainBridge`'s per-op `_op_synaptic` two-window machinery (`unified_brain_bridge.py:509`) and
  `MergedRFComposer._resonate`'s per-op `_resonate` (`nav_conv_merged_bridge.py:863`) — both are
  op-at-a-time shims that the persistent loop subsumes; they likely become thin wrappers or retire once the
  persistent agent is the surface. (Keep `UnifiedBrainBridge` as the parser/dlPFC slice-layout reference.)

**The DEEPEST cleanup is architectural, not line-count** (owner's load-bearing nuance): the persistent
substrate REPLACES the host orchestration layer (the per-op build/kick/read/handoff dance), with the host
reduced to text-in / `" ".join`-out. That is the win, not deleting N files.

---

## 6. Honest risks + the fallback per step

1. **Phase coherence across a multi-op chain in ONE resonate window (highest risk).** The RF readout is a
   first-spike-time code; chaining register→register synaptically assumes the downstream register reads a
   *settled* upstream phasor within the same window. If a bound composite hasn't settled before the unbind
   reads it, the recovery degrades. **Fallback:** a GPU-side **micro-schedule** — settle `acc` over its
   window, THEN install the unbind synapse and run a 2nd window — still NO host round-trip (just a staged
   set of weight installs in the persistent loop). STEP 1 tests this directly; if even the micro-schedule
   fails for the 2-op chain, the synaptic-handoff thesis is NEGATIVE and the megakernel'd op-at-a-time path
   (which already gives ~11–100×) is the honest ceiling.

2. **Transmission gates do NOT gate the RF complex matvec.** Confirmed by reading the step loop:
   `cp_transmission_gain` multiplies `cp_connections` (the real-valued Izhikevich matrix, `bridge.py:5773-5776`),
   while the RF complex matvec `cp_rf_w_re/im @ z` is applied SEPARATELY in `_rf_advance_one`
   (`bridge.py:5526-5529`) with **no transmission-gain multiply**. So "open/close an RF route" is NOT free
   via the existing gate. **Mitigation / flagged `sim/` edit:** either (a) sequence RF ops by which complex
   weights are *installed* (install = open, absent = closed — the natural FHRR way, no gate needed), or
   (b) add a per-RF-synapse gain multiply mirroring `cp_transmission_gain` (a small, byte-reviewed,
   default-preserving `sim/` edit). Option (a) is preferred and avoids the edit; this is a real
   constraint to design around, not a blocker.

3. **Per-op "reset" coupling (the spike trackers are whole-array).** `rf_kick` re-inits `cp_rf_fired` /
   `cp_rf_spike_step` / `cp_rf_prev_im` for the WHOLE array (`bridge.py:5481-5484`), even under a neuron
   mask. On a persistent bridge, re-initialising one register's trackers would reset a still-settling
   register or the store. **Mitigation / flagged `sim/` edit:** mask the tracker re-init too (~6 lines,
   default `None` = current behaviour, byte-reviewed; `test_rf_*` asserts bit-identity). Low risk, owner
   pre-approved sim/ edits when justified.

4. **A giant persistent step loop vs many small per-op loops (latency shape).** The current per-op bridges
   are sized to the op (`2D`, `(L+1)D`); a persistent bridge holds ALL slices, so each resonate STEP touches
   more neurons. The latency profile says cost is **launch-bound, not neuron-count-bound** (D 128→2048 was
   flat), so a bigger-N persistent step should stay launch-bound and the megakernel (1 launch/step) keeps it
   bounded — but this must be MEASURED (STEP 6). **Fallback:** if the persistent step is unexpectedly
   compute-bound, slice-restrict the resonate to the active registers (a masked megakernel variant). Risk:
   the megakernel currently bails to the loop when a neuron mask is set (`bridge.py:5565-5567`) — a
   masked-megakernel path may be needed (flagged `sim/` edit) to keep the persistent+co-resident loop fast.

5. **The dt / NMDA constraints inherited from the dlPFC merge.** The dlPFC working-memory latch needs NMDA
   and survives dt=1.0 only at the genuinely-NMDA-dependent attractor weight ≈30 (the unified-bridge
   crux, `unified_brain_bridge.py:82-120`), with a per-neuron NMDA mask isolating NMDA to the dlPFC slice.
   The persistent bridge inherits this: one global `cfg.enable_nmda` + `cp_nmda_neuron_mask` set to the dlPFC
   slice only. Also: the dlPFC validated regime runs OU background-noise OFF (it corrupts the bistable
   attractors), while parser+composer run OU ON — the unified bridge toggles OU per-read
   (`unified_brain_bridge.py:727-732`); the persistent loop must preserve that per-region regime. Risk:
   medium; the mechanism exists and is tested, but co-residence of OU-ON (composer) and OU-OFF (dlPFC) in
   one continuously-running loop is a new regime to validate (STEP 5).

6. **RF-vs-Izhikevich co-residence limits.** RF and Izhikevich share `v`/`u`; one Izhikevich step destroys
   an idle RF phasor (the 5b KILL finding). The composer is stateless-per-op (re-kicks), so this is
   harmless WHEN ops are discrete. In a PERSISTENT loop where the RF store must PERSIST between turns, the
   `kb_store` must NOT be advanced by Izhikevich steps. **Mitigation:** the store lives in complex SYNAPSES
   (`cp_rf_w_re/im`), which Izhikevich steps never touch (array-disjoint from `cp_connections`) — so the
   *memory* is safe; only transient RF register `v`/`u` is at risk, and those are re-kicked per op anyway.
   This is the same guarantee `MergedRFComposer` relies on (`nav_conv_merged_bridge.py:843-847`). Low risk,
   but it pins a design rule: **persistent RF memory = synapses, transient RF compute = masked `v`/`u`.**

**Overall fallback posture:** every step's NEGATIVE has a defined retreat (micro-schedule for handoff;
weight-install-as-gate for sequencing; host-moat-stays for the gate; megakernel'd op-at-a-time as the
latency ceiling). The biology research is NOT blocked by this arc — it validates at small K where op-at-a-time
latency is tolerable — so a stalled step parks the integration without parking the science.

---

### Bottom line
A flat `what_does` turn currently makes **3 host round-trips** (6 with an embedded clause; 2/fact on the
non-batched path), each ~160 ms and 98% resonate-loop launch overhead. The target replaces the
build/kick/read/handoff dance with ONE persistent bridge whose disjoint RF + Izhikevich slices hand off
**as spikes through complex synapses within the step loop**, host reduced to text-in / join-out. The
machinery exists (the `hear_synaptic` synaptic-parser→composer hand-off, transmission gates +
gate-coupling, the masked-RF co-residence, the resident `_store_substrate`/`_spiking_cleanup`/familiarity-gate
kernels, the megakernel) and is reuse-by-import; the genuinely new primitive is **register→register synaptic
phase handoff in one resonate window**, which is the first cheap-first de-risk (`unbind(bind(role,filler),role)`
on one persistent bridge, GO = 100% recovery == the two-call host oracle, anti-cheats = permuted-role +
severed-route lesion). Two small, default-preserving `sim/` edits are likely (mask the RF spike-tracker
re-init; possibly a masked-megakernel path); one real constraint to design around (transmission gates don't
gate the RF complex matvec → sequence ops by weight-install instead). The cleanup payoff: numpy exits the
production runtime (the dual paths, the legacy rate composer, the reference phasor sims, the per-op
orchestration) and stays as the test oracle.
