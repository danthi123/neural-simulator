# Tier-2 Phase C — the FULL who/what conversational turn as ONE persistent integrated spiking loop: build design (2026-06-19)

**Type:** read-only design / build plan. NO code edited, NO experiments run. One design document.
**Decision this designs:** realize the FULL end-to-end conversational who/what turn (comprehend → store → scan →
answer/abstain) as ONE persistent interacting spiking loop on the production `OneBrainComposer`, with the host
`for/if/return` orchestrator (`_scan`) GONE — the substrate sequencing the ops on its own spiking match result.
**Roadmap memory:** `project_one_brain_integrated_pipeline_and_cleanup` (the owner's "real one brain"; co-location ≠
integration — the whole loop must hand off as spikes through synapses, no host control between ops).
**Audience:** the controller (owner) + the executing subagent. Plain language; every term defined once.

> **This is engineering on two proven mechanisms, not new research.** Both foundational pieces are GO + controller-verified:
> - **Phase A — the bind→store DATA hand-off (H4) is SYNAPTIC** (`2026-06-19-onebrain-bindstore-handoff-derisk.md`, commit
>   `21bec31c`): the composite flows `acc → store-block-readout` through a unit complex synapse; recall == host (6/6),
>   lesion collapses, permuted carries content, moat 0 breaches, NO `sim/` edit.
> - **Phase B — the on-substrate SEQUENCER (H9) is GO** (`2026-06-19-onebrain-sequencer-derisk.md`, commit `6043101b`):
>   a point-neuron basal-ganglia/thalamocortical circuit sequences the who/what scan on the spiking match result
>   (gated disinhibition `couple_gate_to_pool` + a BG production rule), replacing the host `_scan`; ==host 6/6, moat 0
>   false-accepts, sequencer-lesion fails safe, permuted-rule inverts, NO `sim/` edit. The decisive fix was resetting
>   per-query membrane to the Izhikevich resting potential (~−65 mV), NOT 0 mV.
>
> Phase C **composes** these into one loop. The Phase B subagent's own assessment: "Phase C is engineering on a proven
> mechanism." The one genuinely-novel seam — making the result→sequencer coupling itself on-substrate (Phase B still
> reads the cleanup scores to host and re-drives them onto a *separate* sequencer bridge) — is scoped below as a
> deliberate fork with an honest-negative escape.

---

## Terms (defined once)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated neurons with one step loop.
- **op** — one conversational operation: comprehend (parse), bind, bundle, store, reconstruct, unbind, cleanup,
  match-the-cue, answer/abstain, render.
- **RF** — resonate-and-fire: the spiking neuron model the composer's algebra runs on (`NeuronModel.RESONATE_AND_FIRE`).
  Its state is a complex number `Z = re + i·im` held in the bridge's `v` (membrane) and `u` (recovery) arrays. A
  concept/role is a **phasor** vector: a phase angle per dimension; the angle, not the magnitude, carries the info.
- **FHRR** — the vector-symbolic algebra the composer realizes (bind = complex multiply = phase-add; unbind = multiply
  by the conjugate; bundle = sum). It runs THROUGH complex synapses (`cp_rf_w_re`/`cp_rf_w_im`).
- **host** — Python/numpy on the CPU (plus the CPU↔GPU transfers it forces).
- **host round-trip (DATA)** — reading spiking state OFF the bridge to host (`rf_read_phases`,
  `to_host(cp_membrane_potential_v)`), computing in numpy, driving the result BACK on (`rf_kick`). Moves *data*.
- **host orchestration / the orchestrator (CONTROL)** — the Python control flow that decides *which op runs next* and
  gates answer-vs-abstain: `_scan`'s `for got in self._read_blocks(): if all(got[r]==want...): return got[answer_role]`.
  Moves *control*, not data. This is what Phase B replaced and what Phase C must keep replaced end-to-end.
- **transmission gate** — a per-synapse current multiplier in `[0,1]` (`cp_transmission_gain`) that scales a route's
  synaptic CURRENT at runtime; `set_transmission_gate(name, v)` opens/closes a pre-wired route. Multiplies
  `cp_connections` (the Izhikevich real-valued matrix), NOT the RF complex matvec.
- **gate↔pool coupling** — `couple_gate_to_pool` / `_apply_gate_couplings`: a transmission gate driven IN-SUBSTRATE by
  the firing-rate EMA of a control population, no host read. Disinhibit pool X → its activity opens route gate G.
- **the moat (no-confab abstention)** — a query returns `None`/`"unknown"` when no stored fact matches, rather than
  inventing one. **Phase C treats the moat as a HARD gate: 0 false-accepts on an unstored cue. Never weakened.**
- **the sequencer KERNEL (Phase B's result)** — the 2-block who/what scan sequenced on the substrate: present the cue +
  each block's decoded word-lines, settle a spiking gated-match cascade, read the spiking match pools `m0`/`m1`, apply a
  BG production rule (`m0`→answer block 0 priority; else `m1`→answer block 1; else abstain). No Python `for/if/return`.

---

## 1. Scope — the smallest END-TO-END persistent loop first (cheap-first), with IN/OUT bounded

### 1.1 What "the full who/what turn as one loop" means here

A `what_does(agent, action)` turn today is: `hear` (parse + bind + bundle + write store block) → `_read_blocks` (fire
all triggers → unbind 4 roles → cleanup → host argmax) → `_scan` (host `for/if/return` cue-match → answer/abstain) →
render (host `" ".join`). Phase A made the bind→store hand-off (write step) synaptic; Phase B replaced the `_scan`
control with a spiking sequencer **on a separate Izhikevich bridge, with the cleanup scores still read to host and
re-driven**. Phase C's job: run the whole turn — comprehend, store, reconstruct/unbind/cleanup, match, answer/abstain —
as ONE persistent loop where **the residual host CONTROL glue is gone** and the result→sequencer hand-off is on-substrate
(or, if that seam walls, honestly bounded).

### 1.2 The cheap-first first loop (the minimum to call Phase C reached)

**A K=2 store, the who/what query + abstain, the WHOLE turn sequenced by the substrate, on the REAL co-resident
`OneBrainComposer` bridge (the parser + RF registers + persistent store + the sequencer all DISJOINT INDEX SLICES on
ONE bridge), with the host doing only: text-in (the sentence string) and the mechanical body-read of which spiking
channel won (→ the emitted patient label).** Concretely:

1. comprehend `dog go north` + `cat run river` via the on-bridge parser (already synaptic on `OneBrainComposer`);
2. store each fact via the Phase-A synaptic bind→store hand-off (the composite never round-trips to host);
3. for a query `(dog, go)`: reconstruct+unbind+cleanup all blocks on-bridge → **the cleanup result drives the sequencer
   slice on the SAME bridge (the novel seam, §2.3)** → the spiking gated-match cascade settles → the BG production rule
   selects {answer block 0 / answer block 1 / abstain} → the won channel maps to the patient (`north`);
4. an absent cue `(fox, go)` → no match fires → abstain (`None`), false-accept = 0.

**This is the cheapest END-TO-END loop that exercises comprehend→store→scan→answer/abstain with the host control
GONE.** It composes Phase A (step 2) + Phase B (step 3's sequencer) + the on-bridge parser (step 1), and adds exactly
one new piece: the result→sequencer coupling realized ON the one bridge (§2.3).

### 1.3 IN vs OUT (explicit)

**IN (Phase C cheap-first → full):**
- comprehend (the on-bridge `BridgeParser`, already synaptic) — IN as the front-end (reuse, not re-derive).
- the Phase-A synaptic bind→store hand-off — IN (fold into the loop).
- who / what query (`query_patient` / `query_agent`) + abstain — IN (the sequencer drives it).
- the cheap-first first loop at **K=2**, then extend to **K∈{4,8}** with all 4 roles (agent/action/patient/polarity).
- the result→sequencer hand-off made on-substrate (§2.3) — IN, as the novel seam (with a host-coupling fallback).

**OUT (later phases / kept host or kept on the numpy oracle):**
- **negation / yes-no** beyond the polarity tag already in the 4-role bundle — the *yes/no MAPPING* (AFFIRM→yes) is a
  trivial body read, but routing the polarity through the sequencer's match is a Phase-D extension; the cheap-first
  loop reads who/what only. (The polarity ROLE is still bound/stored — it's in the bundle — just not sequencer-gated.)
- **the patient RENDER** as a spiking word-order emission (`NeuralSerialOrderRenderer`, H10) — OUT; the cheap-first loop
  emits the single patient *word* (one body read), not a multi-word ordered sentence. Render is a later phase.
- **recursive embedded CLAUSES** (`_decode_clause`, H11 — a chained 2-hop unbind with a host re-kick between hops) — OUT.
- **multi-hop reasoning** (`query_chain`) and **multi-turn anaphora** (`MultiTurnAgent`) — OUT (they iterate the single
  turn; once the single turn is one loop, they compose, but they are not Phase C).
- **reconsolidation** (`update_on_mismatch`) — OUT (stays as-is; it reuses `_compose_phases`/`_write_block`, which
  Phase A already made synaptic).
- the residual DATA hand-offs **H1 (operand-in)** and **H8 (cleanup argmax)** — H8's spiking-WTA replacement
  (`_spiking_cleanup`) is a PHASE-0 confirm, not new; H1 (the concept code entering from host at parse time) is the
  parser's job and stays as the legitimate text-in boundary. The cheap-first loop does NOT require H1 to be synaptic
  (the parser supplies the operand; that is comprehension, the legitimate sensory boundary).

**The honest framing of "host doing real work" after Phase C cheap-first:** the host does text-in, supplies the concept
codes at parse time (comprehension/sensory boundary — legitimate per the BRAIN-BASED-ONLY standard, like a retina
rendering input), and reads which spiking channel won (the body read, like the nav cascade reading the winning motor
pool). The CONTROL FLOW between ops — which op next, match-the-cue, answer-vs-abstain — is on the substrate. That is the
owner's "real one brain" target for the who/what turn.

---

## 2. Architecture — how Phase A + Phase B + the parser compose into ONE persistent loop

### 2.1 The starting substrate (already done — confirm, don't rebuild)

`OneBrainComposer` (`one_brain_composer.py:87`) is ALREADY ONE persistent co-resident bridge built once
(`build_coresident_bridge`, `:62`): the parser slice `[0:P]` (Izhikevich, voltage in `v`/`u`), the RF work registers
(`fill_*`, `bound_*`, `acc`), the persistent fact-store in complex synapses (`store_conns` → `cp_rf_w_re/im`), the
per-block + batched Q registers, and the cleanup slices — all disjoint index ranges. The RF ops are masked to their
slice (`rf_mask`, `:160`); the Izhikevich parser slice's `v`/`u` is byte-isolated from RF ops (the masked
`_rf_advance_one`, `bridge.py:5601`). H2/H3/H5/H6/H7 (bind/bundle/reconstruct/unbind/cleanup) are already synaptic
on-bridge. The masked megakernel (`enable_rf_cudagraph`) and the CSR cache (`enable_csr_cache`) are on by default.

**So "one persistent bridge with the algebra synaptic" is DONE.** Phase C adds: (a) the sequencer SLICE as a third
disjoint index range on this SAME bridge, (b) the Phase-A bind→store hand-off folded in, (c) the result→sequencer
coupling on-bridge, (d) the host `_scan` removed from the query path.

### 2.2 The seams (where each piece hands off as spikes)

The full who/what turn, op by op, with the hand-off mechanism at each seam:

| seam | from → to | mechanism | status |
|---|---|---|---|
| S0 | sentence → role firing (comprehend) | the on-bridge `BridgeParser` fires the role per word; the role selects each bind | ALREADY-SYNAPTIC (`OneBrainComposer.hear`) — reuse |
| S1 | parser role + filler code → `fill_i` register | host builds the kick from `comp.concepts[word]` (H1) | HOST (legitimate text-in/sensory boundary) — kept |
| S2 | `fill_i` → `bound_i` → `acc` (bind ×n + bundle) | diagonal + unit complex synapses; the product STAYS on-bridge | ALREADY-SYNAPTIC (`_compose_phases` H2/H3) — reuse |
| **S3** | **`acc` → store block readout (the WRITE)** | **the Phase-A `acc → store-readout` unit complex synapse; capture at the synaptic terminus** | **PHASE A — fold in** |
| S4 | store block → reconstruct (fire trigger) → unbind 4 roles → cleanup → membrane | fire all triggers; block-diagonal unbind + cleanup complex synapses; `Re(c)` on the cleanup neurons' membrane | ALREADY-SYNAPTIC (`_read_all_blocks` H5/H6/H7) — reuse |
| **S5** | **cleanup membrane (the OP RESULT) → the sequencer's decoded word-lines** | **the result→sequencer coupling (§2.3) — the ONE novel seam** | **PHASE C (new)** |
| **S6** | **cue + decoded word-lines → spiking gated-match → BG production rule → {ans0/ans1/abstain}** | **Phase-B gated disinhibition (`couple_gate_to_pool`) + the BG WTA, now a SLICE on the one bridge** | **PHASE B — port onto the one bridge** |
| S7 | won BG channel → the emitted patient word | the mechanical body read (which channel fired → that block's patient) | HOST (body read) — kept |

The host `_scan` (`one_brain_composer.py:442`) is REMOVED from the query path: its `for`/`if`/`return` is replaced by
S5→S6 (the substrate computes the match and selects the answer/abstain channel in spikes).

### 2.3 The ONE novel seam (S5): the result→sequencer coupling, made on-substrate

This is the only genuinely-new mechanism in Phase C, and it is the heart of "one loop". In **Phase B** the cleanup
result was read to host (`block_cleanup_scores` → `mem` numpy → `scores_to_drive` → `cur[idx(...)]` = an external
current pattern) and driven onto a **SEPARATE** Izhikevich sequencer bridge. That is a host DATA round-trip in the
middle of the loop — the very thing "one brain" removes. Phase C must make the cleanup spiking RESULT drive the
sequencer's decoded word-lines **on the same bridge, without a host read**.

Two options, scoped as a deliberate fork (cheapest-first; option (a) is the default, (b) the escape):

- **Option (a) — DEFAULT, NO `sim/` edit: synaptic projection cleanup→decoded-word-line.** The cleanup neurons (the
  `bat_c_base` V-block per role per block; `Re(c)` lands on their membrane) and the sequencer's decoded word-line pools
  (`d{b}{role}_w`) are BOTH on the one bridge. Wire a **fixed projection** from each cleanup neuron `c_base + ri·V + j`
  to the corresponding decoded word-line pool `d{b}{role}_{j}` (an Izhikevich route on `cp_connections`, gated like any
  Phase-B route). The cleanup membrane is a RATE-coded score (the winner ~1e6 vs runner-up ~4e5, per Phase B's probe);
  the projection must turn that graded score into a SPIKING drive on the decoded word-line. **The mechanism that already
  does rate→spike on this substrate is the same one Phase B used downstream: the decoded line crosses threshold when its
  cleanup score is high.** Because the cleanup membrane is RF state (`v`/`u` complex), the cleanest realization is a
  thresholded read: the cleanup score → a small driver pool whose firing gates the decoded word-line, mirroring the
  spiking-WTA cleanup (`_spiking_cleanup`, `rf_phasor_composer.py:242`, validated == numpy argmax). **The honest open
  question this seam de-risks: can a fixed on-bridge projection convert the cleanup's graded result into the SAME
  decoded-word-line drive Phase B fed by host, cleanly enough that the gated-match cascade stays decisive (true match
  ~0.22 vs no-match ≤0.10) and the moat holds?** This is the FIRST thing Phase C must validate (Task 1, §5).

- **Option (b) — ESCAPE, the host coupling kept (honest-negative form): the cleanup score → decoded-line drive stays a
  host read.** If option (a) walls (the graded-to-spiking projection washes the match out, or needs a comparison the
  fixed projection can't do), Phase C falls back to Phase B's exact mechanism: read the cleanup scores to host, drive
  the sequencer slice. **This is NOT a failure of the arc** — it is the precise boundary "the result→sequencer DATA
  hand-off is the residual host op; the CONTROL (the match + answer/abstain) is on-substrate." It maps where the
  point-neuron loop's seam breaks, and the who/what CONTROL FLOW is still substrate-sequenced (Phase B's GO). The
  cheap-first loop is still reached (host text-in + host result-read + substrate control), just with one more host DATA
  read than the ideal. Record it as the honest deliverable; the ideal (option a) becomes a follow-on.

**Why this is the only novel seam:** S0/S2/S4 are already synaptic; S3 is Phase A; S6 is Phase B's mechanism re-homed
onto the one bridge (a wiring change, not a new mechanism); S1/S7 are the legitimate sensory/body boundaries. S5 is the
one place where an op's spiking RESULT must drive the next op's CONTROL circuit *without leaving the bridge* — and that
closed loop (result-conditioned op-selection in spikes) is exactly the piece the scoping (`§2.2`) and `gated_sequence_demo`
flag as "unbuilt." Phase B proved the SEQUENCER given the decoded drive; Phase C's novel contribution is supplying that
drive on-substrate (option a) or honestly bounding it (option b).

### 2.4 The step-loop micro-schedule (the persistent loop's "one continuous run")

A subtle but load-bearing architectural fact: **`rf_resonate_steps` deliberately bypasses `_run_one_simulation_step`**
(`bridge.py:5607-5612`) — it skips conductance, plasticity, recording, engram, AND **the gate-coupling hook
(`_apply_gate_couplings`)**. The RF algebra (reconstruct/unbind/cleanup) runs via `rf_resonate_steps`; the Izhikevich
sequencer's gated disinhibition runs via `_run_one_simulation_step` (which calls `_apply_gate_couplings`, `:6567`).
**These are two different step paths on the one bridge.** The persistent loop is therefore NOT a single
`while: _run_one_simulation_step()` for everything; it is a fixed micro-schedule that interleaves the two:

```
per turn (one continuous run on the one bridge):
  [comprehend] parser fires (Izhikevich _run_one_simulation_step window) -> roles selected
  [store]      _compose_phases (rf_resonate windows: bind, bundle) -> acc; Phase-A acc->store synapse -> block written
  [reconstruct+unbind+cleanup]  rf_resonate windows (fire triggers -> unbind -> cleanup) -> cleanup membrane = the result
  [S5 result->sequencer]  option(a): a short Izhikevich window (_run_one_simulation_step) where the cleanup->decoded
                          projection drives the decoded word-lines; option(b): host read + drive
  [S6 match+select]       Izhikevich settle window (_run_one_simulation_step, _apply_gate_couplings active): the cue +
                          decoded lines settle the gated-match cascade -> the BG WTA selects the channel
  [S7]  read the won channel (body read) -> the patient word
```

The "one persistent loop" is this fixed sequence of windows on ONE bridge whose state persists across them (the store
synapses persist; the work/Q/cleanup/sequencer registers are re-kicked/reset per turn). **This is the integration: no
host control DECIDES the sequence of ops or the answer — the windows run in a fixed schedule, and the only branch (which
block answers / abstain) is the spiking BG WTA, not a Python `if`.** The fixed-schedule framing is honest: a turn's op
ORDER (comprehend, store, read, match, select) is a fixed program (the same for every who/what turn), exactly as a
cortical-subcortical loop runs a fixed processing pipeline; the *data-dependent* control (which fact matches, answer vs
abstain) is the part that must be — and now is — on the substrate.

### 2.5 What is removed (the residual host glue dissolved)

- **`_scan` (`one_brain_composer.py:442-446`)** — the `for got in self._read_blocks(): if all(...): return` is REMOVED
  from `query_patient`/`query_agent` and replaced by S5→S6→S7. (`_read_blocks` itself stays — it IS the on-bridge
  reconstruct/unbind/cleanup; only the host cue-match `for/if/return` after it is removed.)
- **`_decode_batched_mem`'s host argmax (H8, `:417-432`)** — the cleanup→answer no longer goes through `np.argmax` for
  the cue-match decision; the spiking match cascade decides. (The argmax may remain for the FINAL body read of the
  patient word at S7 — that is a legitimate body read of the won block's patient, not a control decision; or route it
  through `_spiking_cleanup`'s WTA per the PHASE-0 confirm.)

---

## 3. Reuse-by-import vs `sim/` edit

### 3.1 Reuse-by-import (the default — almost everything)

| machinery | file:line | contributes |
|---|---|---|
| `OneBrainComposer` (persistent co-resident bridge; store-in-synapses; cached operators; megakernel + CSR cache) | `one_brain_composer.py:87` | the substrate — already persistent; S0/S2/S4 already synaptic. The starting point. |
| `SynapticH4Composer` (the Phase-A bind→store hand-off, the thin subclass) | `_phaseB_onebrain_bindstore_handoff_derisk.py:68` | S3 — fold its `_compose_phases`-leaves-`acc`-resident + `_write_block`-routes-synaptically into the loop composer. |
| the Phase-B sequencer (`build_sequencer_bridge`, `wire_sequencer_couplings`, `reset_sequencer_state`, `run_sequencer`) | `_phaseB_onebrain_sequencer_derisk.py:117,215,229,254` | S6 — the gated-match + BG-WTA mechanism. Phase C ports the SLICE onto the one bridge (not a separate bridge). |
| `couple_gate_to_pool` / `_apply_gate_couplings` | `bridge.py:3141,3164` | the disinhibition→route primitive; drives the match gates from cue/agent-match firing in-substrate. |
| `set_transmission_gate` / `cp_transmission_gain` | `bridge.py:3115,407` | pre-wire the sequencer's Izhikevich match routes, gate them; multiplies `cp_connections` (the sequencer slice is Izhikevich). |
| `reset_sequencer_state` (membrane→`cp_izh_c_reset` ≈ −65 mV; gate/EMA clear) | `_phaseB_onebrain_sequencer_derisk.py:229` | THE per-query housekeeping the Phase-B GO depended on (the resting-reset fix). Phase C MUST keep this discipline for the sequencer slice. |
| `block_cleanup_scores` / `scores_to_drive` | `_phaseB_onebrain_sequencer_derisk.py:64,96` | the option-(b) host coupling (the escape) AND the spec of what the option-(a) on-bridge projection must reproduce. |
| `_spiking_cleanup` (matched filter + Izhikevich WTA) | `rf_phasor_composer.py:242` | the PHASE-0 spiking-WTA confirm for H8; the rate→spike template for the S5 cleanup→decoded projection (option a). |
| the on-bridge `BridgeParser` (comprehension is synaptic) | `brain_conversational_agent.py` (`BridgeParser`); `OneBrainComposer.hear:197` | S0 — the front-end, reused verbatim. |
| the masked `rf_kick` / `_rf_advance_one` (RF ops sliced; Izhikevich slice byte-isolated) | `bridge.py:5504,5601` | the co-residence guarantee (the sequencer Izhikevich slice's `v`/`u` is untouched by RF ops). |
| `set_plasticity_gate` / `cp_plasticity_rate_gain` | `bridge.py:3089` | freeze the fixed populations (the sequencer routes + store) at `0.0` on the global-Hebbian parser bridge (the parser trains; everything else is fixed). |
| `couple_gate_to_indices` | `unified_brain_bridge.py:123` | the index-based gate-coupling variant (the sequencer slice's control pools resolve by index on the merged bridge). |

### 3.2 The `sim/` edit picture — flag each for byte-review

**The honest assessment: Phase C cheap-first most likely needs ZERO `sim/` edit, with ONE small near-certain edit if
the loop runs multiple stores/reads back-to-back, and ONE deferred fork that the cheap-first loop avoids.**

1. **(LIKELY NEEDED, SMALL) Mask the RF spike-tracker re-init in `rf_kick` (`bridge.py:5537-5540`).** `cp_rf_prev_im`,
   `cp_rf_fired`, `cp_rf_spike_step` are re-initialised whole-array even under a `neuron_mask` (lines 5537/5538/5540).
   On a persistent loop that re-kicks ONE register group (e.g. the next store's `fill_*`) while ANOTHER RF group (the
   store readouts, a still-settling Q register) must hold its state, the whole-array tracker reset would clobber the
   holding group. **Phase A explicitly found this edit was NOT needed for its hand-off** (the route window's resonate
   makes the readout cross from the kick-reset phase-0 baseline — `2026-06-19-onebrain-bindstore-handoff-derisk.md:42-45`),
   so the SINGLE store→read sequence is safe today. **The edit becomes necessary only if the loop's micro-schedule
   re-kicks an RF group while a disjoint RF group must retain its trackers across the kick.** EDIT (if needed): mask the
   three tracker writes the same way `v`/`u` are already masked (lines 5534-5536) — ~6 lines, default `None` =
   byte-identical, `test_rf_*` pins bit-identity. **Low risk; flag for byte-review; defer until a multi-op de-risk shows
   it's required (the cheap-first K=2 loop may not hit it, exactly as Phase A didn't).**

2. **(DEFERRED FORK, the cheap-first loop AVOIDS it) A per-RF-synapse transmission gain.** `cp_transmission_gain`
   multiplies `cp_connections` (the Izhikevich matrix, `bridge.py:5846`); the RF complex matvec (`cp_rf_w_re/im @ z`,
   `bridge.py:5584`) has NO gain multiply. So "dynamically open/close an RF route on a fixed installed weight set" is not
   free. **The scoping flagged this as the single biggest sim/-edit decision in the whole arc.** **Phase B already
   obviated it** for the SEQUENCER (`2026-06-19-onebrain-sequencer-derisk.md:120-123`): the sequencer gates Izhikevich
   routes (already gated by `cp_transmission_gain`), driven by the cleanup result; it never needs to gate an RF route.
   **Phase C inherits this:** S6 is Izhikevich (gateable today), and S2/S3/S4 sequence RF ops by *which complex weights
   are installed* (the natural FHRR way `OneBrainComposer` already uses op-to-op — `rf_set_complex_weights`). **So the
   cheap-first loop needs NO RF gate.** The edit becomes a real fork ONLY if a later phase wants the spiking sequencer to
   dynamically gate an RF op-route per program-step on a fixed RF weight set (not in scope for the who/what turn).
   **Verdict: NOT needed for Phase C; flagged here only because the scoping named it as the arc's biggest potential
   edit, and Phase B + Phase C both route around it.**

3. **(MOST LIKELY NOT NEEDED) A new per-step coupling for the S5 result→sequencer closed loop.** Option (a) (§2.3) wires
   cleanup→decoded-word-line as a fixed `cp_connections` projection gated by the existing primitives — NO new coupling
   type. The existing `couple_gate_to_pool` drives gates from a control pool's firing; if the cleanup→decoded projection
   is a direct synaptic route (cleanup neuron → decoded pool, gated normally), no new hook is needed. **A new sim/ edit
   here would be required only if the S5 conversion needs a comparison the fixed projection + existing couplings can't
   express** (e.g. a divisive normalization of the cleanup scores before they drive the decoded lines). **Flag: re-assess
   after Task 1; if a new coupling primitive proves necessary, it is a small additive `sim/` edit for byte-review; the
   default expectation is NO edit (reuse `cp_connections` + the existing gates).**

**Summary of the `sim/`-edit call:** the cheap-first Phase C loop is reuse-by-import with NO `sim/` edit expected
(matching Phase A and Phase B, both NO `sim/` edit). The one small edit (mask the `rf_kick` tracker re-init) is deferred
until a multi-op de-risk demonstrates it's required, and is then minimal/additive/default-preserving/isolated-commit for
byte-review. The big fork (RF gate) is confirmed NOT needed for the who/what turn (Phase B + Phase C route around it).

---

## 4. The GO bar (the moat is the HARD gate) + the honest-negative framing

### 4.1 The GO bar

Phase C is GO when, on the **REAL co-resident `OneBrainComposer`** (the production object — `composer_kind="onebrain"`):

1. **==host on the capability matrix.** The substrate-sequenced who/what turn (the full loop: comprehend→store→
   reconstruct/unbind/cleanup→match→answer/abstain) returns the SAME answer as the host `_scan` path for every
   `query_patient` / `query_agent` query, on the cheap-first **K=2** store, extending to **K∈{4,8}**. **Multi-seed: 6
   seeds** (the standing rule for any variable effect; the cleanup/match cascade is noise-sensitive, so it is a
   distribution, not an exact identity — 6 seeds).
2. **The no-confab MOAT — the HARD gate — holds: 0 false-accepts.** Every unstored/absent cue (absent agent, absent
   action, cross = agent-of-block-0 + action-of-block-1) abstains (the BG WTA selects the `abstain` channel; the emitted
   patient is `None`). **A single false-accept at any seed is a Phase-C FAIL** (per the moat-as-hard-gate rule;
   `feedback_moat_not_hard_lossy_memory_ok` keeps the moat where free — and here it is free, so it is not weakened).
3. **The full existing CI suite still passes verbatim.** `tests/test_one_brain_composer_agent.py` (11 tests incl. the
   three `is None` no-confab assertions) and `tests/test_brain_conversational_agent.py` must pass UNCHANGED when the
   loop composer is used (the default `rf`/`onebrain` paths byte-unregressed; the loop is opt-in until GO).

### 4.2 The decisive lesion + anti-cheats (cut the loop → fails safe)

- **Sequencer-lesion fails SAFE.** Sever the S5 result→sequencer coupling (the decoded word-lines get zero drive) → the
  match can't fire → the loop ABSTAINS on a PRESENT cue, never confabulates a wrong block. (Phase B's lesion, now on the
  one bridge.) The decisive control: cut the result→op conditioning and the substrate fails safe, not unsafe.
- **Store-lesion collapses recall.** Sever the Phase-A `acc→store` synapse → recall collapses (Phase A's lesion, in the
  loop) — proves the on-bridge store hand-off is load-bearing, not a residual host write.
- **Permuted-rule INVERTS.** Swap the match→answer production rule (m0→ans1, m1→ans0) → the block-0 cue routes to ans1,
  the block-1 cue to ans0 — proves the BG selection carries the conditional, not a fixed scan order. (Phase B's anti-cheat.)
- **Permuted-store carries content.** Synaptically route a distinct fact into a block, read it directly → it holds the
  routed fact (Phase A's anti-cheat).
- **Parser-route lesion collapses comprehension** (when the front-end is exercised): cut the parser→bind routing → `hear`
  stores garbage → the loop can't answer (proves S0 is load-bearing).

### 4.3 The honest-negative framing (a clean failure IS a valid deliverable)

Per the top-level goal (`project_actual_goal_artificial_life_brain_analogue`), an honest negative maps where a
persistent point-neuron loop breaks — a biology-translatable insight. The two clean failure modes and what each delivers:

- **NEGATIVE at the S5 on-bridge coupling (option a walls).** A fixed projection can't convert the cleanup's graded
  result into a clean decoded-word-line drive (the match washes out, or it needs a comparison the projection can't do).
  **Deliverable:** the result→sequencer DATA hand-off is the residual host op on the point-neuron substrate; the CONTROL
  (match + answer/abstain) is on-substrate (Phase B's GO stands). The loop is reached via option (b) (host result-read +
  substrate control). This maps the precise seam where an op's spiking output can't drive the next op's control circuit
  on-substrate — exactly the closed-loop boundary the scoping anticipated.
- **NEGATIVE at the multi-op persistent loop (register cross-talk / phase coherence over the longer chain, or the
  micro-schedule corrupts a holding RF group).** **Deliverable:** the megakernel'd op-at-a-time path (Phase B's separate
  sequencer bridge, already GO) is the honest production form; the single-persistent-bridge integration is the ceiling
  here. Maps the phase-coherence / register-isolation boundary of a multi-op RF+Izhikevich loop on one bridge.

**The moat is NEVER the negotiable axis:** if the only way to pass the ==host bar is by weakening the moat (a
false-accept), Phase C is a FAIL, not a softer GO. The negatives above are about WHERE the integration seam breaks, not
about trading away abstention.

---

## 5. Cheap-first, TDD-friendly task breakdown (subagent-executable; each task ends GREEN)

Ordered so each task is independently verifiable against the numpy oracle / the host path, with an anti-cheat, and the
moat checked at every step. CuPy for the real co-resident runs (the parser trains on the CuPy substrate;
`SIM_BACKEND=cupy`); the pure-algebra parity steps can run numpy (the exact oracle path). Each task writes a findings
doc (or extends the loop runner's JSON); a NEGATIVE task STOPS the cascade with its escape (§4.3) recorded. **Build the
smallest loop → ==host → add a seam → re-gate.** Each task is a `research/runners/_phaseB_onebrain_*_derisk.py` runner
(the established naming) + a `tests/test_*` assertion where it pins a capability.

**Task 0 — PHASE-0 confirm (no new mechanism): the residual op-level spiking flags into the production query path.**
Wire H8 through `_spiking_cleanup`'s Izhikevich WTA (it exists, opt-in in `RFPhasorComposer`) for the FINAL patient
body-read at S7, and confirm the existing moat (the `is None` assertions) holds. **GREEN:** `test_one_brain_composer_agent.py`
11/11 with the spiking-WTA body read on; moat intact. (This isolates any later regression to the *integration*, not the
op-level spiking.) *Smallest, no new mechanism — start here.*

**Task 1 — THE NOVEL SEAM (S5): the result→sequencer coupling on-substrate (option a), in ISOLATION.** A K=2 store; for
each block, drive the cleanup (the real `_read_block` op) and project its cleanup membrane → the sequencer's decoded
word-lines via a FIXED on-bridge `cp_connections` projection (no host read of the scores). **GREEN:** the on-bridge
decoded-word-line drive reproduces the Phase-B host-driven decoded pattern closely enough that the spiking gated-match
stays decisive — true match ≥ ~0.22, no-match ≤ ~0.10 — on a present cue vs an absent cue, **3 seeds** (parity of the
match cascade vs Phase B's host-coupled cascade). **Anti-cheat:** lesion the projection → decoded lines silent → no
match (fails safe). **If NEGATIVE → option (b)** (record the seam boundary; the loop proceeds with the host result-read,
§4.3) and skip to Task 2 using the host coupling. *This is where the arc's novel claim lives or dies; do it early.*

**Task 2 — THE CHEAP-FIRST FIRST LOOP (K=2): the WHOLE who/what turn on ONE bridge, host control GONE.** Compose Task 1's
S5 (option a, or option b if Task 1 went negative) + the Phase-B sequencer SLICE on the SAME `OneBrainComposer` bridge +
the Phase-A bind→store hand-off. The micro-schedule (§2.4) runs comprehend(parser)→store(synaptic)→reconstruct/unbind/
cleanup→S5→S6→S7, with NO host `_scan`. **GREEN:** the loop's `query_patient`/`query_agent` == the host `_scan` for both
present cues + abstains on 3 absent/cross cues, **6 seeds**, false-accepts = 0. **Anti-cheats:** sequencer-lesion fails
safe; store-lesion collapses; permuted-rule inverts; permuted-store carries content. *This is the cheap-first END-TO-END
loop — reaching GREEN here = Phase C cheap-first reached.*

**Task 3 — EXTEND to K∈{4,8} + all 4 roles.** Grow the store to K=4 then K=8; bind/store/match all four roles
(agent/action/patient/polarity in the bundle); the sequencer's match margin must stay decisive as K grows (more blocks →
more leak lines — the Phase-B scope note flags re-verifying the margin at larger K). **GREEN:** ==host on who/what +
abstain at K=4 and K=8, 6 seeds, moat 0 false-accepts; the match margin (worst-leak vs threshold vs true) reported and
still separated. **Anti-cheats:** the full battery at K=8. *Scaling the kernel.*

**Task 4 — FOLD INTO `OneBrainComposer` (opt-in) + CI guard.** Make the loop an opt-in mode on `OneBrainComposer`
(`integrated_loop=True`, default off = byte-identical to today), so `BrainConversationalAgent(composer_kind="onebrain")`
can use it. **GREEN:** `test_one_brain_composer_agent.py` (11) + `test_brain_conversational_agent.py` pass VERBATIM with
the flag OFF (byte-unregressed); a NEW `tests/test_onebrain_integrated_loop.py` asserts the K=2 who/what + moat with the
flag ON (GPU-gated, skips gracefully without GPU/the concept cache, like the sibling tests). *Pins the capability + the
no-regression.*

**Task 5 (only if Task 2/3 surface it) — the `rf_kick` tracker mask `sim/` edit.** If the K=4/8 micro-schedule re-kicks
an RF group while a disjoint RF group must hold its trackers (§3.2 edit 1), make the minimal masked-tracker edit (~6
lines), default `None` = byte-identical. **GREEN:** `test_rf_*` bit-identity pins the default path; the loop's
multi-op schedule now isolates the holding group. **Isolated commit for byte-review.** *Deferred until demonstrated
necessary — Phase A did NOT need it; the cheap-first loop may not either.*

Each task is bite-sized and ends green; Task 1 (the novel seam) and Task 2 (the first loop) are the load-bearing pair —
Task 1 settles whether S5 is on-substrate or host-coupled; Task 2 settles whether the whole turn runs as one loop with
the host control gone. Tasks 3–5 are scaling + integration + the conditional edit.

---

## 6. Risks (bounded, with retreats)

- **Register-reset / cross-talk across a MULTI-op persistent loop (the central new risk).** The K=32 store was
  register-reset-safe (the GAP-A de-risk), and Phase A's single store→read sequence is safe (it did NOT need the
  tracker-mask edit). But the FULL loop chains MORE ops on the one bridge per turn (parser → store → reconstruct →
  unbind → cleanup → S5 → S6), and the RF register groups (work, Q, cleanup) + the Izhikevich groups (parser, sequencer)
  must each retain/reset their state at the right window boundary. **Retreat:** the two-window settle (let a group settle,
  then install the next op's weights and resonate — no host read between) is the Phase-A/register-handoff mitigation; the
  `rf_kick` tracker-mask edit (Task 5) is the fallback if a re-kick clobbers a holding group; the merged-bridge
  co-residence rule (**persistent RF memory = synapses (`cp_rf_w_re/im`, array-disjoint from `cp_connections`); transient
  RF compute = masked `v`/`u`, re-kicked per op**) bounds the risk to transient registers only — the STORE (the memory)
  is never at risk.
- **The membrane-reset-to-rest discipline (the Phase-B fix) must be preserved for the sequencer slice.** Phase B's GO
  depended on resetting the per-query membrane to `cp_izh_c_reset` (≈ −65 mV), NOT 0 mV (0 mV is above threshold → every
  neuron spikes spuriously → a false-match leak). On the co-resident bridge the RF ops reset `v`/`u` to 0 (the RF kick
  baseline) — but the SEQUENCER slice is Izhikevich and MUST be reset to −65 mV, not 0. **Retreat / design rule:** the
  per-turn reset is SLICE-AWARE — RF slices reset to the RF baseline (0, the kick); the Izhikevich sequencer slice resets
  to `cp_izh_c_reset` (`reset_sequencer_state`'s exact discipline, applied to the sequencer index range). This is a
  housekeeping invariant, not a mechanism; getting it wrong is a moat-leak, so it is checked by the moat gate at every
  task.
- **GPU vs CPU.** The on-bridge parser trains on the CuPy substrate (the real co-resident runs are GPU-only,
  `SIM_BACKEND=cupy`; numpy is the tiny-smoke/CI path). The pure-algebra parity sub-steps (Task 1's match-cascade parity
  vs Phase B, the bind→store identity) can run numpy (the exact oracle). Match Phase A/B: numpy for the exact/identity
  parity, CuPy for the integrated co-resident loop. The 6-seed rule applies to the variable (noise-sensitive)
  match/cleanup effects; the exact bind→store identity needs only parity (3 seeds × 2 D, the merged-bridge precedent).
- **The cleanup membrane is RF state, not Izhikevich firing.** S5's option-(a) projection reads the cleanup neurons'
  RF membrane (`Re(c)`, a graded score) to drive Izhikevich decoded word-lines. The RF "membrane" is the complex `v`;
  whether a fixed Izhikevich-style synaptic route can cleanly threshold it is the open question Task 1 settles.
  **Retreat:** option (b) (host read of the score, Phase B's exact mechanism) — the loop still reaches the cheap-first
  bar with the control on-substrate and one host DATA read at S5.

---

## Bottom line

`OneBrainComposer` already gives ONE persistent co-resident bridge with the FHRR algebra synaptic and the store in
complex synapses; **Phase A** made the bind→store hand-off synaptic (GO) and **Phase B** proved the on-substrate
sequencer (GO). **Phase C composes them into the full who/what turn as ONE persistent loop** with the host `_scan`
removed: comprehend (the on-bridge parser, S0) → store (Phase-A synaptic, S3) → reconstruct/unbind/cleanup (already
synaptic, S4) → **the result→sequencer coupling on-substrate (S5, the one novel seam, option a — with a host-coupling
escape, option b)** → the Phase-B gated-match + BG-WTA as a SLICE on the one bridge (S6) → the body read of the won
channel (S7). The cheap-first first loop is **K=2 who/what + abstain, the whole turn substrate-sequenced, host doing only
text-in + the body read**; OUT of cheap-first scope: render, clauses, multi-hop, multi-turn, reconsolidation, yes-no
gating (later phases). **Reuse-by-import with NO `sim/` edit expected** (matching Phase A and B); the ONE small edit
(mask the `rf_kick` tracker re-init) is deferred until a multi-op de-risk shows it's required, then minimal/additive/
default-preserving/isolated-commit for byte-review; the big RF-gate fork the scoping flagged is **confirmed NOT needed**
(Phase B + Phase C route around it via Izhikevich gating + weight-install op sequencing). **The GO bar:** ==host on the
who/what matrix at K∈{2,4,8}, 6 seeds, on the real `OneBrainComposer`, with **the no-confab moat as the HARD gate (0
false-accepts — never weakened)**; the decisive lesion (cut the S5 coupling → fails safe) + the full anti-cheat battery
(store-lesion collapses, permuted-rule inverts, permuted-store carries content). The task breakdown is five bite-sized
TDD steps (Phase-0 confirm → the novel S5 seam in isolation → the cheap-first K=2 loop → extend to K=8 + 4 roles → fold
in + CI guard, with the conditional `sim/` edit last), Task 1 + Task 2 the load-bearing pair. An HONEST NEGATIVE is a
valid deliverable: at S5 (the on-bridge result→sequencer coupling walls → the residual host op is the DATA read, the
CONTROL is on-substrate) or at the multi-op loop (register cross-talk → the op-at-a-time path is the production form) —
each maps exactly where a persistent point-neuron conversational loop breaks, directly on the top-level
artificial-life/brain-analogue goal.

Sources: Phase A (`2026-06-19-onebrain-bindstore-handoff-derisk.md`, commit `21bec31c`); Phase B
(`2026-06-19-onebrain-sequencer-derisk.md`, commit `6043101b`); the pre-registration
(`2026-06-19-tier2-persistent-integrated-loop-scoping.md`); the production composer (`one_brain_composer.py`); the gate
primitives (`bridge.py:3141,3164,3115` — `couple_gate_to_pool`, `_apply_gate_couplings`, `set_transmission_gate`); the
masked RF ops (`bridge.py:5504,5601`); the CI guard (`tests/test_one_brain_composer_agent.py`); Stewart-Choo-Eliasmith
(2012) Spaun (BG action-selection = cognitive control); Logiaco-Abbott-Escola (2021) thalamic control of cortical
dynamics (the gated routing fabric); catalog A (the closed BG action-selection loop).
