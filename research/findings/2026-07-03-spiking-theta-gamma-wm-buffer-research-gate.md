# Research gate: the SPIKING theta-gamma WM-buffer realization of the RANK-3 stack (EMERGE-85 follow-on)

**Date:** 2026-07-03
**Type:** READ-ONLY deep-research gate (no code edit / no GPU / no git). Scopes the SPIKING realization of the EMERGE-85
rate-level theta-gamma multiplexed WM buffer + stack-match, cheap-first, on the project's `SimulationBridge`.
**Trigger:** EMERGE-85 GO closed the RANK-3 recursion boundary at the RATE level (a functional theta-gamma buffer +
mirror-pair stack-match). The non-negotiable directive is FULLY-SPIKING on the ONE brain. EMERGE-85's own HONEST_NOTE
pre-registers "the SPIKING theta-gamma port is the follow-on rung." This gate is that rung's cheap-first scope.

**BOTTOM LINE (verdict up front):** the project **already has a validated, genuinely-spiking theta-gamma / Lisman-Idiart
ordered-WM realized on the resonate-and-fire (RF) substrate** — `research/runners/ordered_position_wm.py`
(`OrderedPositionWM`, promoted PRODUCTION module, CYCLE 135 GO), whose encode/read run on real RF spikes via
`bridge.rf_kick` / `rf_resonate_steps` / `rf_read_phases` (`NeuronModel.RESONATE_AND_FIRE`; the Im zero-crossing IS the
spike, phase = the gamma slot). The genuine residual is TINY and precisely located: EMERGE-85's **mirror-pair stack-MATCH**
(`WMBuffer.feature`, `_emerge85:90-91`, `f[k]=1.0 if idx[k]==idx[N-1-k]`) is the ONLY piece still a host `==` array
comparison; the multiplexed BUFFER itself already has a spiking realization. The cheapest genuinely-spiking rung is to
**replace EMERGE-85's `WMBuffer` with `OrderedPositionWM` slot-reads + realize the per-pair equality as a spiking
phase-COINCIDENCE** (two phasors agree ⇒ their unbind lands on a familiar concept; the composer's own familiarity/cleanup
gate is that coincidence), then re-run EMERGE-84's exact task + anti-cheats. **NO `sim/` edit is required for this rung.**

---

## MOVE 1 — ISOLATE + QUANTIFY the genuine residual

### What EMERGE-85 is, mechanically (from the code)
`_emerge85_wm_buffer_recursion_derisk.py` re-runs EMERGE-84's nested subject-verb pair-matching grammaticality task
(`_emerge84._gen`, center-embedding, verbs reversed, multiset-preserving swap for the ungrammatical case so the count
shortcut is dead). Its `WMBuffer.feature(toks)` (`_emerge85:81-94`) does exactly four rate/array things:

| # | Piece | EMERGE-85 code | What it is |
|---|-------|----------------|-----------|
| (a) | **Ordinal slot assignment** (the multiplex) | `idx = [_NUM_IDX[w] for w in toks if w in _NUM_IDX][:capacity]` (`:82`) | each number-marker token pushed into the next ordered slot; a running-ordinal = the theta-gamma multiplex |
| (b) | **Per-slot number storage** | the `idx` list holds `sng`/`plu` per slot | one item held per slot, unfading up to `capacity` |
| (c) | **Mirror-pair stack MATCH** | `f[k] = 1.0 if idx[k]==idx[N-1-k] else 0.0` (`:90-91`) | verb j (slot N-1-k) matches its top-of-stack subject j (slot k) — a **LIFO pop + equality** |
| (d) | **Depth read-out** | `f[dim]=n_pairs; f[-1]=1.0`; ridge over `f` (`:92-93,104-105`) | linear read-out over the per-pair agreements → grammaticality |

The `buffer_slot_scramble` control (`:83-86`) shuffles the slot order → destroys the mirror structure → collapses (proves
the ORDERED slots are load-bearing = the stack, not a bag). GO at d*≤3, BOUNDARY at capacity (depth 4, 10 markers > 8
slots) — the biologically-faithful ~2-3 center-embedding human limit.

### Which pieces the project ALREADY has spiking machinery for
- **(a) ordinal slot assignment + (b) per-slot storage** → `OrderedPositionWM.encode_sequence` (`ordered_position_wm.py:102-111`):
  each item bound to a per-seed **gamma-slot POSITION phasor** (`roles["pos0..pos{N-1}"]`, `:91-93`, N_SLOTS=7 = the
  Lisman-Idiart ceiling) and bundled into ONE composite. The bind/bundle run on the RF substrate
  (`_bind`→`_resonate`→`rf_kick`/`rf_resonate_steps`, `rf_phasor_composer.py:234-243,169-182`). Reading slot k =
  `read_slot(composite,'posk')` = spiking `unbind` + familiarity gate + cleanup (`:120-131`). **This IS a spiking
  theta-gamma multiplexed buffer** — validated 6-seed, recall 1.00 at loads {2,3,5}, order-control-FLIPS 1.00, moat clean
  (`_phaseB_ordered_wm_position_binding.json`: recall_means all 1.0, order_control_flip_mean 1.0, moat principled-pass).
- **(c) mirror-pair equality** → NO direct spiking op yet, BUT the substrate primitive exists: two phasors are equal iff
  `bind(conj(A), B)` (i.e. `unbind`) yields the zero/identity phasor whose familiarity-cosine to a reference is maximal —
  i.e. **phase coincidence = the composer's own `_match_strength` / familiarity gate** (`ordered_position_wm.py:113-118`;
  `cleanup_separated`, Bogacz-Brown). Equality-by-coincidence is a phase read, not a host `==`.
- **(d) read-out** → the RANK-3 decision is already realizable as the project's validated spiking WTA/familiarity read-out
  (the `_cleanup` / `_izh_bank` WTA in `rf_phasor_composer.py:303+`, and the on-spike A→W read-out used across EMERGE-59..71).

### The precise, quantified residual
**Exactly ONE array line is the genuine residual: `_emerge85:90-91` — the host `idx[k]==idx[N-1-k]` mirror-pair equality.**
Everything else (the ordinal multiplex, the per-slot storage, the ordered-slot read, the slot-scramble control) is already
realized as genuine RF spikes by `OrderedPositionWM`. The residual is a per-pair PHASE COINCIDENCE, for which the substrate
already has the primitive (spiking unbind + familiarity match). This is the EMERGE-30..71 pattern again: "most of the
blocker is already spiking; the genuine residual is a single local op."

---

## MOVE 2 — REFRAME via "how does the brain actually do theta-gamma WM (Lisman-Idiart) + the stack pop?"

### The Lisman-Idiart theta-gamma multiplex (catalog N.15, `feature-catalog.md:992-999`)
- **The mechanism:** a theta carrier (4–12 Hz) nests ~7±2 gamma sub-cycles (40–100 Hz); **each gamma cycle carries one
  item-assembly**; the theta period sets the buffer span (≈7±2 = Miller's number); re-firing on successive theta cycles
  maintains the items. Order = **gamma phase within the theta cycle**. (Bz 2006 Cycle 12 pp. 350–353, Fig. 12.6 p. 352;
  Lisman & Idiart 1995; Bragin et al. 1995; Chrobak & Buzsáki 1998.) **Item maintenance is NMDA-mediated
  after-depolarization / persistent activity** — the same slow-NMDA machinery the project already has
  (`cp_conductance_g_nmda_recurrent`, per-region NMDA mask `bridge.py:320-323,1250-1257`).
- **Sim status per the catalog (`:995`):** "missing — the project has neither theta nor gamma generators in the locale
  path. Adding nested oscillators is straightforward in the NM framework (sinusoidal `excitability_drive` at theta + a
  faster modulator with theta-phase-modulated amplitude)." **This is the ONLY place a `sim/`/driver addition would enter —
  and it is optional (see the two-track reframe below).**
- **Capacity-limit validation (`:999`):** count distinct gamma-locked assemblies per theta cycle (≤9); recall drops
  sharply past the gamma/theta ratio. EMERGE-85's capacity-8 BOUNDARY at depth 4 is exactly this prediction; the spiking
  version must reproduce it.

### The biological "stack pop + match" (the verb matching its top-of-stack subject)
Center-embedding is LIFO: subjects pushed on gamma slots 0..N/2-1, verbs popped in REVERSE (verb j at mirror slot
N-1-j). The verb's agreement check is a **phase-based COINCIDENCE**: the verb's number-assembly must fire IN THE SAME
gamma slot (or with the same phase-relationship) as its matching subject's number-assembly. In VSA/phasor terms (the
substrate's language): `agree(k) = familiarity( unbind(bind(pos_k, num_subj_k), bind(pos_{N-1-k}, num_verb_{N-1-k})) )`
peaks iff the two numbers coincide — the composer's already-spiking unbind + familiarity gate. Biologically this is
**gamma binding-by-synchrony** (catalog N.19, `:1028-1032`: neurons in the same gamma cycle are co-grouped; the gamma
cycle and the STDP window are matched) applied to the two mirror slots. So the "stack pop + match" is a **gamma-slot
phase coincidence between the mirror-paired assemblies** — NOT a host `==`, and NOT a new mechanism: it is the
familiarity/cleanup gate the composer already runs (Bogacz-Brown), read on the mirror-pair unbind.

### The reframe (which hypothesis to test)
EMERGE-85's `WMBuffer` and the project's `OrderedPositionWM` are the SAME Lisman-Idiart mechanism at two abstraction
levels: `WMBuffer` uses literal integer slots + `==`; `OrderedPositionWM` uses **spiking phasor position-slots + spiking
unbind** (order = phasor identity, the VSA analogue of gamma phase). The catalog's "literal nested theta/gamma
oscillators" is the FULLER biological realization but is NOT required for the RANK-3 stack capability — the
position-binding phasor WM is already the substrate's validated, genuinely-spiking realization of the multiplex. So the
right cheap-first hypothesis is: **"does the already-spiking `OrderedPositionWM` (ordinal multiplex + spiking unbind) +
a spiking phase-coincidence stack-match reproduce EMERGE-85's RANK-3 surpass + capacity boundary?"** — NOT "can we learn
the buffer" (which fails) and NOT "must we first build literal theta/gamma oscillators" (a larger, separable N.15 build).

---

## MOVE 3 — RANK cheap-first spiking de-risks (the ladder rate → spike → on-bridge)

### RUNG 1 (RECOMMENDED — the cheapest genuinely-spiking rung; NO `sim/` edit)
**Swap EMERGE-85's `WMBuffer` for `OrderedPositionWM` + a spiking phase-coincidence stack-match; re-run EMERGE-84's task
+ anti-cheats.** Concretely, a new `_emerge86_spiking_wm_buffer_recursion_derisk.py`:
1. Reuse `m84._discover` / `_gen` / `_count_multiset_baseline_acc` and `m85._TEST_DEPTHS` verbatim (SAME task, SAME
   count-shortcut defeat, SAME depths 1..4 to expose surpass + capacity boundary).
2. Build `OrderedPositionWM(seed, D=128, vocab=['sng','plu'], n_slots=8)` (capacity 8 = EMERGE-85's `_CAPACITY`; on the RF
   substrate). `encode_sequence([number-markers in stream order])` = the spiking ordinal multiplex (a,b) — genuine RF
   spikes via `rf_kick`/`rf_resonate_steps`/`rf_read_phases`.
3. **Stack-match as spiking coincidence (c):** for each mirror pair k, `read_slot(comp,'pos{k}')` and
   `read_slot(comp,'pos{N-1-k}')` (spiking unbind + cleanup → `sng`/`plu`); the agreement bit = the two cleaned-up reads
   equal. (Even stronger/purer: `agree(k) = familiarity(unbind by pos_k of (comp bound-with pos_{N-1-k}))` peaks on
   coincidence — a single phase read, no host label compare; use this as the load-bearing variant.) The per-pair
   agreements + `n_pairs` feed the SAME ridge read-out EMERGE-85 uses (d).
4. **Anti-cheats (reuse EMERGE-85's, adapted):** (i) **slot-scramble** — shuffle the position-slot assignment before
   encode (the RF analogue of `slot_scramble_rng`) → the mirror structure dies → collapse (ordered slots load-bearing);
   (ii) **count-multiset baseline** stays at chance by construction; (iii) **capacity-overflow** — depth 4 (10 markers >
   8 slots) MUST boundary (the honest bounded-recursion limit, NOT forced GO); (iv) **unbind-LESION** — replace the
   spiking unbind with a random-phasor read → collapse (proves the read is the mechanism); (v) **empty-slot MOAT** — a
   read of a never-bound slot ABSTAINS (the composer's familiarity gate, already 6/6 in `OrderedPositionWM`).
- **GO bar (6-seed, mirroring EMERGE-85):** spiking buffer d* STRICTLY > the plain reservoir's d*=2 (from EMERGE-84);
  count defeated; slot-scramble + unbind-lesion collapse; moat clean; capacity boundary at depth 4 EXPECTED.
- **Reused machinery:** `OrderedPositionWM` (spiking multiplex+read, VALIDATED), the RF substrate (`rf_kick`/
  `rf_resonate_steps`/`rf_read_phases`, `NeuronModel.RESONATE_AND_FIRE`), the familiarity/cleanup gate (the coincidence),
  EMERGE-84 task + EMERGE-85 harness. **NO `sim/` edit** (pure reuse-by-import; RF ops already committed).
- **Cost:** CPU/numpy (the RF composer runs there; each op a small RF bridge), ~minutes/seed. Single-variable: the ONLY
  thing changed vs EMERGE-85 is `WMBuffer` → spiking `OrderedPositionWM`+coincidence. This is the tightest rung that is
  genuinely SPIKING (not another rate array) and directly closes the pre-registered residual.

### RUNG 2 (optional, larger — the LITERAL temporal theta/gamma nesting; needs an additive `sim/` driver)
If a reviewer wants the FULLER N.15 realization (a real theta carrier at 4–12 Hz nesting gamma sub-cycles in TIME, items
bound to gamma PHASE rather than to a phasor position-code): add a nested-oscillator DRIVE. The catalog says this is
"straightforward in the NM framework" (`:995`). Reusable pieces already present:
- **Theta carrier:** the `StimulusManager` **already produces a SINUSOIDAL current** (`experiment/stimulus.py:154-156`,
  `frequency_hz`/`amplitude_pA`/`phase_offset_rad`, config `sim/config.py:823-824`) — a ready ENVIRONMENT-side theta
  injector into a pool. Or a neuromodulator `excitability_drive` target driven by a `manual`/sinusoidal production rule
  (`sim/neuromodulators.py:546-583,636-647`).
- **Gamma:** the existing FS-interneuron gamma (`CORTEX_GAMMA_FS_NETWORK` profile `sim/profiles.py:263-265`; the
  `gamma-oscillations` benchmark) — a real PING/ING gamma pool (catalog N.19).
- **Slot persistence:** the per-region NMDA mask + slow-NMDA recurrent (`bridge.py:320-323,1250-1257,259-266`) for the
  Lisman after-depolarization item maintenance.
- **`sim/` edit needed?** Only a THIN, ADDITIVE, default-OFF driver: a theta-phase variable that phase-modulates the
  gamma pool's amplitude/gating (a nested-oscillator `excitability_drive` whose amplitude follows theta phase) — a
  faithful-biology addition, byte-identical when off (the NM/StimulusManager pattern is already additive). This is a
  SEPARABLE N.15 build, NOT on the RANK-3 critical path; RUNG 1 already delivers the spiking RANK-3 capability without it.
  Defer unless the deliverable specifically demands literal time-domain gamma nesting.

### RUNG 3 (scale/confirm) — after RUNG 1 GO, GPU multi-seed confirm at production D + the `SpikingBrocaConsole` wire-in
(the RANK-3 stack made available to the reply-side producer for center-embedded constructions), following the EMERGE-60/66
console-wire pattern. Composition of already-de-risked pieces; not a research gate.

---

## MOVE 4 — VERDICT

**SURPASSABLE, and cheaply — the spiking realization is ~95% already built.** The RANK-3 theta-gamma WM buffer is NOT a
new-mechanism problem: the project's `OrderedPositionWM` (`research/runners/ordered_position_wm.py`, PRODUCTION, 6-seed
GO) is already a genuinely-spiking Lisman-Idiart ordered-WM on the RF substrate (`rf_kick`/`rf_resonate_steps`/
`rf_read_phases`, `NeuronModel.RESONATE_AND_FIRE`). The genuine residual is a SINGLE host line — EMERGE-85's mirror-pair
`==` (`_emerge85:90-91`) — and the substrate already has its spiking primitive (unbind + familiarity/cleanup coincidence,
the Bogacz-Brown gate).

**THE SINGLE RECOMMENDED NEXT DE-RISK (start building immediately):**
> Write `research/runners/_emerge86_spiking_wm_buffer_recursion_derisk.py` that reuses EMERGE-84's task
> (`m84._discover`/`_gen`/`_count_multiset_baseline_acc`) and EMERGE-85's harness/depths, but **replaces `WMBuffer` with
> a spiking `OrderedPositionWM(seed, D=128, vocab=['sng','plu'], n_slots=8)`** for the ordinal multiplex + per-slot read,
> and realizes the **mirror-pair stack-match as a spiking phase-COINCIDENCE** (the composer's unbind + familiarity gate on
> each mirror pair), feeding the SAME ridge read-out. Anti-cheats: **slot-scramble** (RF-analogue of EMERGE-85's, must
> collapse), **count-multiset** (chance by construction), **unbind-lesion** (random-phasor read, must collapse),
> **empty-slot moat** (must abstain), **capacity-overflow at depth 4** (must BOUNDARY — the honest bounded-recursion
> limit, do NOT force GO). GO bar (6-seed): spiking-buffer d* STRICTLY > reservoir d*=2, count defeated, scramble +
> lesion collapse, moat clean, capacity boundary honest.

- **Cheapest genuinely-spiking rung:** yes — it changes ONLY the residual, reuses a VALIDATED spiking WM, needs **NO
  `sim/` edit**, runs CPU in minutes/seed, and is single-variable against EMERGE-85.
- **Why not "learn the buffer":** rejected — the reframe (MOVE 2) shows the mechanism is a fixed multiplex + a coincidence
  match, both structural primitives the substrate has; "can the buffer be learned" is the wrong hypothesis (the EMERGE-16
  learned-bind pattern: structure DEVELOPS/is-primitive, it isn't task-learned).
- **Why not RUNG 2 first:** the literal time-domain theta/gamma-oscillator nesting (catalog N.15 "missing") is the FULLER
  biological realization but a SEPARABLE, larger build (a thin additive default-off `sim/` oscillator driver) that is NOT
  on the RANK-3 critical path — RUNG 1 already delivers the spiking RANK-3 stack. RUNG 2 is a good later "make the
  multiplex literally temporal" deepening, gated separately.
- **Honest scope carried forward:** the RANK-3 recursion stays BOUNDED at the buffer capacity (~2-3 center-embeddings, the
  human limit) — the spiking version must reproduce that boundary, not surpass it. The VSA position-phasor is the
  substrate's phase-code (order = phasor identity ≈ gamma phase); RUNG 2 would make the phase literal in time. The
  composer's exact-inverse FHRR bind remains the principled idealization (the separate learned-cortex frontier), unchanged
  by this rung.

**Files cited:** `research/runners/_emerge85_wm_buffer_recursion_derisk.py` (`:81-94` the residual, `:90-91` the one
`==` line), `research/runners/_emerge84_reservoir_stack_recursion_derisk.py` (task), `research/runners/ordered_position_wm.py`
(the spiking WM, `:102-131` encode/read, `:113-118` familiarity), `research/runners/rf_phasor_composer.py` (`:169-182`
`_resonate`→`rf_kick`/`rf_resonate_steps`, `:234-301` bind/bundle/unbind), `sim/bridge.py` (`:5656-5866` RF ops,
`:320-323,1250-1257` per-region NMDA), `experiment/stimulus.py` (`:154-156` SINUSOIDAL theta injector),
`sim/neuromodulators.py` (`:546-583` `excitability_drive`), `sim/profiles.py` (`:263-265` gamma FS pool),
catalog `E:/Documents/Projects/sim-catalog/references/feature-catalog.md` (N.15 `:992-999`, N.19 `:1028-1032`).
Prior spiking theta-gamma art: `research/findings/raw/direction_E_theta_gamma_numpy_probe.py` (algebra ceiling GO),
`_phaseB_ordered_wm_position_binding.json` (the spiking WM GO), `research/findings/2026-06-17-ordered-wm-position-binding-derisk.md`.
