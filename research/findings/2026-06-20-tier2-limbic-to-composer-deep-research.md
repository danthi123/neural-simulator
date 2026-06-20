# Tier-2 (#6 limbic → composer) deep-research gate — the route is ALREADY BUILT both ways; this gate FIRES AGAINST A RE-SCOPE

**Date:** 2026-06-20
**Type:** read-only deep-research + catalog/code-verification gate (NO code, NO experiments — one findings doc). The
standing opening move for the TRUE ONE BRAIN line (owner `feedback_move_everything_to_shared_spiking_substrate`;
roadmap `project_post_conversational_roadmap_tiers` Tier 2). Branch `main`, read-only throughout.
**The question put to this gate:** "what is the minimal, biology-grounded mechanism by which the shared spiking
dopamine/salience core MODULATES the conversational composer … on the existing substrate? What is the minimal NEW
route (and is it a `sim/` edit)?"
**Top-line verdict:** **The limbic → composer route is NOT an open gap. It is already built TWO ways, both validated,
one production-wired on the merged bridge.** The prompt's premise — "DA does NOT currently reach the composer; a new
`_rf_lambda`/cleanup-gain route is needed (likely a `sim/` edit)" — is **STALE**: that gap was closed across CYCLE
206–CYCLE 315 by the prior `2026-06-19-tier2-limbic-to-composer-scoping` arc, which **explicitly rejected the
`_rf_lambda` route** in favour of two cleaner mechanisms requiring **no `sim/` edit at the recommended level**. Per the
owner's standing instruction for this gate ("if MORE is already done than expected, SAY SO and fire against a
re-scope — like the primary just did"), this gate fires that way. The genuinely-open Tier-2 frontier is a *different*
roadmap item — **the persistent integrated spiking loop** (the on-substrate sequencer replacing host orchestration),
whose Phases A + B are also already GO and whose Phase C (one novel seam) is the real remaining build. §6 phases it.

---

## 1. Verified state — what is built vs the genuine gap (file:line + finding evidence)

### 1.1 The shared limbic / dopamine core REACHES the composer — already, two ways

There are **two distinct, mechanistically-different DA → composer routes**, both validated. They are complementary
(one acts at recall, one at encoding), not duplicates.

**Route A — the READ-side DA salience-gate (PRODUCTION-WIRED on the merged bridge, GO).**
- Code (verified): `MergedNavConvAgent.__init__` gains `enable_da_salience_gate=False` (+ knobs `da_gate_g0=0.06`,
  `da_gate_k=2.0`, `da_gate_cap=0.25`) at `research/runners/nav_conv_merged_bridge.py:1219-1250`; the three helpers
  `_da_confidence_gate` (`:1316-1332`), `_gated_out` (`:1334-1356`), `_role_cleanup_scores` (`:1358-1365`).
- Mechanism (verified at `:1316-1356`): before each conversational READ op the agent reads the shared spiking
  dopamine off **its own merged bridge** — `nm.get_concentration("dopamine")` (`:1328`) — maps it
  *clamped-to-sharpen* via the de-risk's `da_to_gate` (imported verbatim, `:1322`):
  `g_eff = clip(g0, g_cap, g0 + k·(DA − DA_baseline))`, then ABSTAINS on a noise-dominated cue read when
  `min(margin(agent), margin(action)) < g_eff` (`:1355`, the exact `OneBrainComposer._margin`). At DA baseline
  `g_eff = g0` ⇒ the gate floor ⇒ **byte-identical** read path; only a salient/high-DA turn raises it (capped at the
  inverted-U ceiling `g_cap`). **A higher gate can ONLY tighten abstention ⇒ moat-safe by construction.**
- Validation: `research/findings/2026-06-18-DA-salience-gate-production-wireup-GO.md` — GPU smoke PASS (default-OFF
  byte-identity; ON + a co-resident limbic core: gate rises `0.060 → 0.250` monotone with spiking DA `0.500 → 0.843`,
  **0 false-accepts at BOTH DA levels**, clean recall preserved); regression GREEN
  (`tests/test_nav_conv_merged_agent.py` 8/8 + `tests/test_nav_conv_step2b_coresident.py` 7/7); **NO `sim/` edit**
  (`git diff --stat -- sim/` empty), +92/−4 in the one runner. Underlying 6-seed de-risk:
  `2026-06-18-DA-composer-precision-derisk-GO.md` (commit `b76959be`).

**Route B — the WRITE-side DA-gated encoding strength (DE-RISKED numpy 6/6; NOT yet deployed on the merged bridge).**
- Code (verified): `RFPhasorComposer.__init__(..., encoding_gain_fn=None)` at
  `research/runners/rf_phasor_composer.py:62-80`; applied in `_store_substrate` at `:440-453` —
  `g = 1.0 if self.encoding_gain_fn is None else float(self.encoding_gain_fn())` (`:449`), then
  `conns = [(1+k, 0, complex(g) * zc[k]) ...]` (`:450`). Plus the default-preserving read-damage knobs
  `_retrieve_noise`/`_retrieve_read_floor`/`_retrieve_lam`/`_retrieve_kick_mag` (`:87-100`, applied `:455-478`) so the
  graceful-degradation knee can be reached.
- Mechanism (verified): the RF phase read-out has a hard **magnitude floor** (`sim/bridge.py:5589`,
  `_rf_mag2 > _rf_floor2`; the megakernel mirror at `:5640`): a readout neuron whose `|Z|` decays below the floor never
  registers the `im` up-crossing → reads phase 0 (garbage) → contributes nothing to the cleanup matched filter. So a
  **per-fact** encoding gain `g·M / σ` (M = base readout magnitude, σ = common read noise) is **differential, not a
  vacuous global scalar**: a rewarded (g>1) fact's readout neurons stay above the floor where a neutral (g=1) fact's
  drop below ⇒ the rewarded fact wins the cue-match scan under shared read damage.
- Validation: `research/findings/2026-06-19-dopamine-encoding-gain-derisk.md` — **GO**, 6 seeds (42/43/44/100/101/102),
  numpy/CPU. WITHIN-FACT gain lift +6/12 (neutral fact g1 4/6→g2 6/6; rewarded fact g1 2/6→g2 6/6); DA-lesion null
  (both g=1) kills the differential; permuted (gain swapped) flips the advantage to the other fact; monotonic in g
  (g0.5→1→1.5→2→3); **moat 6/6 in EVERY condition** including the high-gain regimes; regression byte-identical
  (`encoding_gain_fn=None`). Commits `b4ae63b0` (composer knobs) + `5928465f` (runner). **NO `sim/` edit.**

**The shared DA SOURCE the composer reads — also built + co-resident.**
- The shared `dopamine` neuromodulator is registered on the merged bridge over the spiking-SNc pool via
  `from_region_firing_signed` (the signed RPE production rule), at `nav_conv_merged_bridge.py:849-866`
  (`co_resident_limbic` / `co_resident_nav_critic`, source `["limbic_snc"]` / `["snc"]`) and `:889-895`
  (`co_resident_td_cueshift`, source `["td_snc"]`); threshold 0.0 ⇒ neutral-at-rest. The production rule
  `from_region_firing_signed` is `sim/neuromodulators.py:774-817` (EMA of SNc firing − tonic = δ, signed). So
  `get_concentration("dopamine")` **is** the spiking reward-prediction error.
- The limbic core itself is lifted onto the merged bridge (`co_resident_limbic`, 46 regions, nav-inert, default-off
  byte-preserved; the δ=r−V spiking mechanism confirmed co-resident) — `2026-06-18-merged-limbic-core-lift.md`. Its
  standalone Schultz RPE battery is GO 6/6 (`2026-06-18-limbic-core-rpe-battery-GO.md`). **One characterized boundary**
  (honest): the FULL multi-gate RPE *arithmetic* (burst ratio ≥3× AND the GABA_B value-subtraction together) does not
  hold on the het-off merged config — the SNc's effective synaptic response is ~6–10× weaker in the full network; the
  path forward is on-merge critic LEARNING (not per-region heterogeneity, which was de-risked + FALSIFIED). **This does
  NOT block the limbic→composer routes:** both routes consume the DA *concentration*, which is present and readable
  (the salience-gate's GPU smoke already drove it to two operating points on the merged `limbic_snc`).

### 1.2 The "genuine gap" the prompt names — verified, and verified ALREADY-CLOSED

The prompt's flagged gap is real *as a fact about the RF op path* but **already-addressed as a design decision**:

- **The fact (verified):** the RF resonate fast path **does** bypass `_run_one_simulation_step`, where the
  neuromodulator subsystem lives. `rf_resonate_steps` (`sim/bridge.py:5607-5627`) is a bare `for _ in range(n):
  self._rf_advance_one()` loop (or the megakernel `:5672`); the neuromodulator `manager.step(self)` is called only
  INSIDE `_run_one_simulation_step` (`sim/bridge.py:6821`), as are `compute_synaptic_gain_multiplier` (`:5814`),
  `compute_plasticity_rate_multiplier` (`:6918`), `compute_plasticity_gate_values` (`:6828`). A full-file search of
  `sim/neuromodulators.py` for `rf|phasor|resonate|composer|cleanup` returns **ZERO matches** — the NM subsystem has
  no awareness of the RF composer. So a *live, per-resonate-step* NM coupling into the RF dynamics does not exist.
- **One nuance worth recording:** `compute_excitability_drive_per_neuron` (`sim/neuromodulators.py:562-632`) DOES write
  to `total_input_current_pA` for in-scope neurons, and RF state reuses `cp_membrane_potential_v`/`cp_recovery_variable_u`
  — but only on the `_run_one_simulation_step` path, NOT the bypassed `rf_resonate_steps` fast loop the composer uses,
  so even this does not reach the composer's ops in production.
- **Why this gap does NOT need closing (the prior scoping's explicit decision):** routing a live DA concentration into
  `_rf_lambda` (the decay term, `sim/bridge.py:5524`/`5574`) would (a) require threading NM state into the bypassed
  fast loop (the flagged `sim/` edit) AND (b) modulate *every* op including the unbind/cleanup of **already-stored**
  facts — a global gain whose functional consequence is muddy and which risks the moat. The `2026-06-19` scoping
  **explicitly rejected `_rf_lambda`** (its §2.4: "modulating the decay would change the phase read-out … and would
  modulate already-stored facts on every op — wrong functional target and a moat risk"). The two built routes (A:
  read-side gate; B: write-side encoding gain) reach the composer functionally **without** touching the bypassed loop —
  Route A reads the concentration at the agent layer between ops; Route B bakes the gain into the stored weights at
  encoding. **Both are `sim/`-edit-free at the deployed level.**

**⇒ The verified state: the limbic → composer route is BUILT (A wired on the merged bridge + GO; B de-risked numpy +
GO), the DA source is co-resident, and the only "gap" (live NM into the bypassed RF loop) was deliberately designed
AROUND, not left open.**

---

## 2. The mechanism (which composer knob, mapped to the gain-theory biology)

The two built routes map cleanly onto two well-separated pieces of biology — *both* are legitimate "DA modulates the
cortex," at different stages:

- **Route A (recall precision) = DA → cortical GAIN (Servan-Schreiber/Cohen gain theory).** The salience gate sharpens
  the *recall* decision: higher DA → a stricter confidence threshold → the marginal (noise-dominated) recall tail
  abstains, the confident reads pass. This is the **Servan-Schreiber, Cohen & Steingard (Science 1990)** result —
  catecholamine/DA acts as a *gain* change that raises the network-level signal-to-noise ratio (improves signal
  detection of the assembly as a whole), realized cellularly by **Thurley, Senn & Lüscher (J Neurophysiol 99:2985,
  2008)** — DA increases the f-I-curve *gain* of prefrontal pyramidal neurons (via reducing the slow AHP) and shifts
  the input-output to lower inputs. Mapping a higher-gain (sharper) read to a stricter abstention is the
  "attention-as-precision" reading: salient context → higher precision on the read-out → reject the low-precision tail.
  Catalog grounding: **C.32 Component-1 (salience/detection)** (`feature-catalog.md:615`) — the DA detection component
  "amplifies … downstream sensory gain on any 'potentially important' event"; the inverted-U cap (`da_gate_cap`) is the
  **C.05** NE/arousal Yerkes-Dodson inverted-U (`feature-catalog.md:714`).
- **Route B (encoding strength) = DA-gated LTP magnitude (Lisman-Grace hippocampal-VTA loop).** The encoding gain
  scales how strongly a fact is *written*: a rewarded/salient fact is encoded above the RF magnitude floor (a stable
  trace), a neutral fact at unit magnitude (a degradable trace). This is **Lisman & Grace 2005** (the hippocampal-VTA
  loop *controls the entry of information into long-term memory* — VTA DA enhances LTP for novel/salient information)
  and **Lemon & Manahan-Vaughan 2006** (D1/D5 gates the acquisition of novel information via hippocampal LTP/LTD).
  Catalog grounding: **D.16 Place-field stability requires attention + D1/D5 dopamine + late-LTP** (`feature-catalog.md:1272`,
  verbatim: "Inattentive exploration → fields form but degrade in 3–6 hours. Attended exploration … → fields stable for
  days"; Kandel 6e Ch 54 pp 1366–1367) — the composer's magnitude floor is the direct analogue of "an un-attended
  trace degrades below a usable level." The shared DA itself is **C.04 Dopamine (reward/WM)** (`feature-catalog.md:692`)
  driven as the **C.28/C.32** RPE (`feature-catalog.md:574`/`615`).

So the chain is exact and TWICE-instantiated: **shared spiking SNc firing → `dopamine` concentration (signed RPE) →
(A) sharper recall gate at READ, or (B) stronger encoding magnitude at WRITE → the salient fact is preferentially
recalled.** Same motivational brain, both halves; the limbic core reaches the conversational cortex.

---

## 3. Reuse vs the minimal NEW piece (the `sim/`-edit call)

For the limbic → composer route there is **essentially nothing new to build** — it is reuse-by-import, already done:

| piece | status | file:line / finding |
|---|---|---|
| DA source on the merged bridge (`dopamine` over spiking SNc) | BUILT | `nav_conv_merged_bridge.py:849-866` (`co_resident_limbic`/`nav_critic`), `:889-895` (`td_cueshift`); `from_region_firing_signed` `neuromodulators.py:774` |
| NM subsystem (hold + update + read) | BUILT, reusable hinge | `neuromodulators.py:228` (`get_concentration`), `:249` (`step`) |
| **Route A** (read-side salience gate) | **PRODUCTION-WIRED + GO** | `nav_conv_merged_bridge.py:1316-1356`; `2026-06-18-DA-salience-gate-production-wireup-GO.md` |
| **Route B** (write-side encoding gain) | **DE-RISKED numpy + GO**; deploy pending | `rf_phasor_composer.py:62-80,440-478`; `2026-06-19-dopamine-encoding-gain-derisk.md` |
| the no-confab moat under modulation | held 6/6 (B) + 0-FA both DA levels (A) | both findings above |

**The single residual increment for THIS route (small, optional, NOT a new mechanism):** deploy Route B on the merged
bridge — i.e. pass `encoding_gain_fn = lambda: clip(1.0 + k_DA·(b.neuromodulator_manager.get_concentration("dopamine")
− 0.5), 0.5, 3.0)` to the `MergedNavConvAgent`'s composer, so a fact heard while the spiking SNc is bursting is
encoded stronger. This is a **runner-side wire-up, NO `sim/` edit** (the gain is a composer-layer multiply on the
weights handed to the existing `rf_set_complex_weights`). Two caveats to handle honestly:
1. The composer on the merged path is `RFPhasorComposer`/`MergedRFComposer` with `enable_substrate_store=True`; the
   `encoding_gain_fn` already lives on `RFPhasorComposer._store_substrate` (`:449`). The production `OneBrainComposer`
   path (`_write_block`, `one_brain_composer.py:261-267`) does **NOT** yet have the gain hook (it writes
   `complex(zc[k])` unconditionally) — so deploying Route B on the OneBrain default would need the same one-line gain
   added there too (still no `sim/` edit, additive default-off).
2. The de-risk used a *probe* DA value and *injected* read damage to reach the degradation knee; in deployment the
   damage is the real superposition/noise load. The deploy wire-up should re-confirm the differential survives at the
   real merged-bridge read damage (a small GPU smoke, §4).

**The `sim/`-edit verdict for the route: NONE required.** The prompt's anticipated `_rf_lambda`/cleanup-gain `sim/`
edit is **explicitly NOT the right mechanism** (it would modulate the read of already-stored facts and risk the moat —
the prior scoping rejected it). The only `sim/` edit anywhere in this neighbourhood is the *deferred, optional*
fully-on-substrate refinement: an additive default-`1.0` `kick_gain` on `rf_kick` / `weight_gain` on
`rf_set_complex_weights` (byte-identical when absent) — needed ONLY if a later phase wants the encoding gain to *emerge
from live SNc firing during an on-bridge store op* rather than be read as a scalar at the composer layer. NOT required
for the route, the de-risk, or the deploy.

---

## 4. The recommended cheap-first de-risk (the ONE thing left for this route)

Because both routes are already GO at their validated levels, the only open experiment is **the Route-B deployment
smoke on the merged bridge** — proving the WRITE-side DA gain works with the REAL shared `dopamine` (not a probe) and
the REAL read damage (not injected):

- **Setup (GPU, `SIM_BACKEND=cupy`):** a `MergedNavConvAgent(co_resident_limbic=True)` with the composer's
  `encoding_gain_fn` wired to the merged `dopamine`. Drive the shared `limbic_snc` to two operating points (the
  salience-gate smoke's recipe): DA-low (tonic) and DA-high (salient burst). `hear` a fact at each DA level (so the
  encoding gain differs by the real DA), then query both after a common read.
- **The decisive metric:** the fact heard at DA-high is recalled correctly at a read-damage level where the fact heard
  at DA-low abstains/mis-recalls — i.e. **the same within-fact / matched-cue differential the numpy de-risk showed,
  now driven by the spiking SNc on the merged bridge.** Quantify with the composer's `_margin` (rewarded-block margin
  stays above the gate; neutral-block margin collapses at the same damage).
- **Multi-seed plan:** 6 seeds (42/43/44/100/101/102), the standing rule (the cleanup/read is noise-sensitive — a
  distribution, not an exact identity). GO = the differential holds ≥5/6, the DA-lesion (hold DA at baseline for both
  hears) kills it, the moat is intact at both DA levels.

(Route A needs nothing further — it is already production-wired + GPU-smoke-validated. If the controller wants Route A
at 6-seed numbers rather than the committed de-risk, that is a confirmation run, not a de-risk.)

---

## 5. The anti-cheats (the moat is the HARD constraint here)

The hardest constraint, exactly as the prompt flags: **DA modulation must NOT manufacture false-accepts.** Both routes
are designed to make this structurally impossible, and the de-risks already verify it:

| anti-cheat | what it rules out | for Route A | for Route B |
|---|---|---|---|
| **Moat-safe by construction** | "DA lowering a threshold breaches the moat" | a higher gate can ONLY *tighten* abstention (clamped below at `g0`; `da_to_gate` floor) ⇒ DA never *loosens* the moat — structural | the gain scales an *already-stored* fact's magnitude; an *unstored* cue has no block to amplify ⇒ no fact to confabulate — structural |
| **Unstored-cue abstention at MAX gain (HARD gate)** | "the modulation broke the moat empirically" | 0 false-accepts at DA_low AND DA_high (the salience-gate smoke) | moat 6/6 in EVERY condition incl. g=2 (the encoding de-risk) — any breach at any gain = NEGATIVE, not a tunable |
| **DA-lesion (hold DA at baseline)** | "the effect is fact content/order, not DA" | gate = `g0` floor ⇒ byte-identical read | both facts g=1 ⇒ the within-fact differential vanishes (the de-risk's decisive control) |
| **Permuted-gain** | "one fact is intrinsically more robust" | n/a (read-side; the lesion+monotonicity cover it) | swap the high gain to the other fact ⇒ the advantage FOLLOWS the gain (verified) |
| **Value-conflict / regression** | "the default path drifted" | regression GREEN, default-OFF byte-identical | `encoding_gain_fn=None` byte-identical, regression suites pass |

The moat operating-point is honest: Route B's de-risk runs at read-damage σ=260 (the moat-safe knee for D=64/2 facts),
NOT the maximal-differential σ where the *damage itself* (not the gain) starts to breach the moat on one seed — per the
HARD-gate rule ("any moat breach at any setting is a NEGATIVE, not a tunable"). The deploy smoke (§4) must verify the
real merged-bridge damage sits below that knee, or report the boundary honestly.

---

## 6. Honest top-line + phased build

**Top-line: the limbic → composer route is ACHIEVABLE on the existing substrate and is ALREADY ACHIEVED** — Route A
(read-side salience gate) is production-wired on the merged bridge and GO; Route B (write-side encoding gain) is
de-risked numpy 6/6 and one runner-side wire-up away from deployment; the shared DA source is co-resident; and the
`_rf_lambda`/cleanup-gain `sim/` edit the prompt anticipated is **deliberately NOT the mechanism** (rejected as a moat
risk). It needs **no `sim/` edit** and does **not** need the deferred dendritic substrate. So this gate, like the
conversational-primary gate before it, **fires AGAINST a redundant re-build of #6** and toward the *actual* open
Tier-2 frontier.

**The actual open Tier-2 frontier (a DIFFERENT roadmap item): the PERSISTENT INTEGRATED SPIKING LOOP** — the
on-substrate sequencer that removes the host `for/if/return` orchestrator from the conversational turn (owner
`project_one_brain_integrated_pipeline_and_cleanup`: co-location ≠ integration). Its foundational pieces are ALSO
already GO:
- **Phase A** — the bind→store DATA hand-off (H4) is synaptic (`acc → store-block-readout` complex synapse) —
  `2026-06-19-onebrain-bindstore-handoff-derisk.md` GO (3 seeds × 2 D = 6/6, lesion collapses, permuted carries
  content, moat 0 breaches, NO `sim/` edit; commit `21bec31c`).
- **Phase B** — the on-substrate SEQUENCER (H9, the deep result-conditioned op-selection that `gated_sequence_demo`
  left host-given) — `2026-06-19-onebrain-sequencer-derisk.md` GO (6/6 seeds 42–47, ==host, moat 0 false-accepts,
  sequencer-lesion fails safe, permuted-rule inverts, NO `sim/` edit; commit `6043101b`).
- **Phase C** — the FULL who/what turn as ONE persistent loop, host `_scan` removed — **DESIGNED**
  (`2026-06-19-tier2-phaseC-integrated-loop-design.md`), with exactly **one novel seam** left: S5, the result→sequencer
  coupling made on-substrate (option a, NO `sim/` edit expected; option b = host-coupling escape if it walls). Phase C
  is "engineering on two proven mechanisms" (its own framing), 5 bite-sized TDD tasks, the moat the HARD gate.

**Recommended next action (the gate's deliverable to the controller):**
1. **Present that #6 (limbic → composer) is DONE** (Route A wired + GO; Route B de-risked + GO) — surface the milestone,
   exactly as the primary-parser gate surfaced "already built." Owner decides whether to spend the small Route-B deploy
   smoke (§4) now or treat #6 as closed at its current strong state.
2. **Pivot the Tier-2 build to the persistent integrated loop's Phase C** (the genuine open frontier), starting at its
   Task 1 (the S5 on-bridge result→sequencer coupling in isolation, the one place the arc's novel claim lives or dies)
   — Phases A + B are GO, so this is the load-bearing remaining de-risk of the whole "real one brain."

**Phasing if #6 deployment is wanted first (cheap, parallelizable with Phase-C Task 1):**
- **#6-deploy (small):** wire Route B's `encoding_gain_fn` to the merged `dopamine` on `MergedNavConvAgent` (+ the
  one-line gain in `OneBrainComposer._write_block` if the OneBrain default is the target), run the §4 6-seed GPU smoke,
  moat the HARD gate. NO `sim/` edit. → #6 fully deployed both halves (read + write).
- **Phase-C Task 1 (load-bearing):** the S5 seam in isolation (§5 of the Phase-C design) — the real Tier-2 frontier.

**Where it would hit a wall (honest):** the limbic → composer route does NOT wall (it's built). The persistent loop's
Phase C *could* wall at the S5 on-bridge coupling (a fixed projection can't convert the cleanup's graded result into a
clean spiking decoded-line drive) — but that is a bounded, designed-for honest-negative (option b retreat keeps the
CONTROL on-substrate with one host DATA read), and it is a *different* roadmap item from #6. Neither item needs the
deferred dendritic substrate.

---

## 7. Discipline confirmation

- **READ-ONLY:** no code edited, no experiments run. Stayed on branch `main` throughout.
- **Trust-but-verify:** every load-bearing claim cited to actual code (`nav_conv_merged_bridge.py:1316-1356`,
  `rf_phasor_composer.py:440-478`, `sim/bridge.py:5589/5607/6821`, `sim/neuromodulators.py:774`,
  `one_brain_composer.py:261-267`), the actual catalog text (D.16 `feature-catalog.md:1272`; C.32
  `feature-catalog.md:615`; C.04 `:692`; C.05 `:714`), and verified primary literature (Servan-Schreiber/Cohen Science
  1990; Thurley/Senn/Lüscher J Neurophysiol 2008; Lisman-Grace 2005; Lemon & Manahan-Vaughan 2006). The two prior GO
  findings' verdicts were read in full and confirmed (not taken from headlines).
- **The re-scope call is the deliverable:** like the conversational-primary gate, this gate found the named direction
  ALREADY BUILT and fires against a redundant rebuild, pointing the Tier-2 build at the genuinely-open persistent-loop
  Phase C.

### Key sources
- Servan-Schreiber, Cohen & Steingard (1990) "A Network Model of Catecholamine Effects: Gain, Signal-to-Noise Ratio,
  and Behavior," *Science* 249:892. — DA as gain → network SNR.
- Thurley, Senn & Lüscher (2008) "Dopamine increases the gain of the input-output response of rat prefrontal pyramidal
  neurons," *J Neurophysiol* 99:2985. — cellular DA→f-I gain.
- Lisman & Grace (2005) "The Hippocampal-VTA Loop," *Neuron* 46:703; Lemon & Manahan-Vaughan (2006) *J Neurosci*
  26:7723. — DA gates entry into long-term memory / D1-D5 gates LTP magnitude.
- Catalog: D.16 (`feature-catalog.md:1272`), C.32 (`:615`), C.04 (`:692`), C.05 (`:714`), C.28 (`:574`). Kandel 6e Ch
  54 pp 1366–1367.
- Project findings: `2026-06-18-DA-salience-gate-production-wireup-GO.md`, `2026-06-19-dopamine-encoding-gain-derisk.md`,
  `2026-06-19-tier2-limbic-to-composer-scoping.md`, `2026-06-18-merged-limbic-core-lift.md`,
  `2026-06-18-limbic-core-rpe-battery-GO.md`, `2026-06-19-onebrain-bindstore-handoff-derisk.md`,
  `2026-06-19-onebrain-sequencer-derisk.md`, `2026-06-19-tier2-phaseC-integrated-loop-design.md`,
  `2026-06-19-tier2-persistent-integrated-loop-scoping.md`.
