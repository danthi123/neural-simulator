# Tier-2 — the persistent INTEGRATED SPIKING LOOP (the REAL one-brain headline): deep-research scoping (2026-06-30)

**Type:** read-only deep-research scoping / diagnosis + ranked options. NO code/sim/GPU edit; NO GPU job launched (a
DA-recall-vigor GPU acceptance run was in flight — `bvlajgc53`, CYCLE 717/718 — and was NOT touched). One findings doc.
**Decision this scopes:** Tier-2's next + bigger piece (`project_post_conversational_roadmap_tiers` Tier 2;
`project_one_brain_integrated_pipeline_and_cleanup`) — make the whole conversational pipeline ONE continuously-running
spiking loop where the between-op hand-offs are SYNAPTIC (spikes through synapses), not host-orchestrated op-at-a-time.
**Standing opening move:** deep-research + catalog review FIRST at a new direction (the SURPASS sharpening: ISOLATE +
QUANTIFY the genuine residual — usually most is already done; #6 and Tier 1 both turned out largely-done on inspection).
**Audience:** the controller (owner) + the executing subagent. Plain language; every term defined once.

---

## TOP-LINE (the honest call the owner asked for)

**The persistent integrated spiking loop is LARGELY ALREADY BUILT — and was de-risked GO on 2026-06-19/20 (Phase C).**
The headline who/what conversational turn ALREADY runs as one persistent loop on the production `OneBrainComposer`, with
the host control orchestrator (`_scan`) GONE, the per-query normalization closed on-bridge, and the no-confab moat
intact (6-seed, 0 false-accepts). This is the SAME re-frame that hit #6 (CYCLE 711) and Tier 1 (CYCLE 709): the comfortable
"this is the big unbuilt Tier-2 piece" verdict does NOT survive inspection. Three pieces were already shipped:

1. **`persistent_loop=True` (default since 2026-06-24 "Closure 2/5")** — the INTRA-query op hand-offs (unbind→cleanup,
   clause hop-1→hop-2, reconsolidation PE) are spike-resident: each register holds a clean unit phasor handed off
   register→register via `_dev_rekick_into` (an on-device read-phase + re-kick, NO `to_host` of the phasor). Byte-identical
   to the host round-trip it replaced (`_burndown_I1a_op_handoff_probe`, max|dphase|=0). `one_brain_composer.py:184,607-641`.
2. **`integrated_loop=True` (default in the flagship `consolidated_320_conversation_demo`)** — the CUE-MATCH CONTROL
   (which stored block answers / answer-vs-abstain) is on-substrate: a spiking basal-ganglia/thalamocortical K-way
   sequencer replaces the host `for/if/return` (`_scan`). `_seq_block` (`one_brain_composer.py:1029-1060`). The host
   `_scan` `for got in _read_blocks(): if all(...): return` is REMOVED from the integrated query path.
3. **S5 (the last host DATA read, the per-query peak-normalization) CLOSED on-bridge** via the existing
   `input_divisive_norm` Carandini-Heeger primitive — and WIRED INTO PRODUCTION (`_ensure_sequencer` builds the divnorm
   score bridge `enable_divnorm=True` and feeds `make_block_drives(..., retreat="divnorm")`, `one_brain_composer.py:1009,
   1016,1023`). The host `scores.max()` read is RETIRED for the cleanup→sequencer seam. `2026-06-20-S5-divisive-norm-derisk.md`.

**Phase C Task 2 GO (2026-06-19, `2026-06-19-phaseC-task2-wholeturn-loop.md`):** the whole who/what turn (comprehend →
store → reconstruct/unbind/cleanup → match → answer/abstain) runs as ONE persistent loop, host `_scan` GONE, ==host 6/6
seeds at K=2, **moat 0 false-accepts every seed**, full anti-cheat battery (sequencer-lesion fails safe, store-lesion
collapses, permuted-rule inverts, permuted-store carries content). NO `sim/` edit.

**⇒ The genuine residual is NOT "build the integrated loop" — it is FOUR narrow, specific gaps**, none of which is the
big mechanism-class build the title suggests. They are precisely isolated + quantified in §2. The single cheapest +
highest-leverage one is **R1: collapse the THREE sub-bridges (composer / sequencer / divnorm-score) into ONE so the
cleanup→sequencer DATA hand-off is intra-bridge spikes, not a host array marshalled BETWEEN bridges** — closing the last
honest host DATA seam in the deployed loop. The honest bound on R1's hardest sub-part is mapped in §2.1.

---

## 1. DIAGNOSIS — what EXISTS toward the persistent integrated loop (cited file:line)

### 1.1 The substrate is already ONE persistent co-resident bridge with the algebra synaptic

`OneBrainComposer` (`one_brain_composer.py:87`, built via `build_coresident_bridge`) is ONE persistent
`sim.bridge.SimulationBridge` built once: the parser slice `[0:P]` (Izhikevich, `v`/`u`), the RF work registers
(`fill_*`/`bound_*`/`acc`), the persistent fact-store in COMPLEX SYNAPSES (`store_conns → cp_rf_w_re/im`), the per-block
+ batched Q registers, the cleanup slices — all DISJOINT index ranges. The store (the MEMORY) lives in synapses
(`cp_rf_w_re/im`, array-disjoint from `cp_connections`); the bind/bundle/reconstruct/unbind/cleanup ops run THROUGH
complex synapses (`rf_set_complex_weights` + `rf_resonate_steps`). The masked megakernel (`enable_rf_cudagraph`) + CSR
cache (`enable_csr_cache`) are default-on. **"One persistent bridge with the FHRR algebra synaptic and the store in
synapses" is DONE** (design doc `2026-06-19-tier2-phaseC-integrated-loop-design.md` §2.1).

### 1.2 The seam taxonomy (S0–S7) and what is synaptic vs host — VERIFIED against the code

The full who/what turn, op by op (the design's S0–S7, confirmed by direct read of `one_brain_composer.py` +
`_phaseC_task2_wholeturn_loop.py` + the divnorm wiring):

| seam | from → to | mechanism (as DEPLOYED) | SYNAPTIC / HOST | file:line |
|---|---|---|---|---|
| S0 | sentence → role firing (comprehend) | on-bridge `BridgeParser` fires the role per word; the role selects each bind | SYNAPTIC (the parser FIRES; host only maps the fired role→filler dict) | `hear` `:405-413` |
| S1 | parser role + filler code → `fill_i` kick | host builds the kick from `comp.concepts[word]` | HOST — legitimate text-in/sensory boundary (kept) | `_store_fact` `:389` |
| S2 | `fill_i` → `bound_i` → `acc` (bind ×n + bundle) | diagonal + unit complex synapses; the product STAYS on-bridge | SYNAPTIC | `_compose_phases` `:538-558` |
| S3 | `acc` → store block (the WRITE) | the composite phases → block-readout complex synapses | SYNAPTIC (the store IS synapses; one `rf_read_phases` of `acc` per store = the consolidation read, not a between-op hand-off) | `_compose_phases:558` → `_write_block:560-577` |
| S4 | store → fire trigger → unbind 4 roles → cleanup → membrane | block-diagonal unbind + cleanup complex synapses; `Re(c)` on the cleanup neurons | SYNAPTIC (the batched read fires all triggers in 3 resonate windows; `persistent_loop` re-kicks the Q regs as clean unit phasors between unbind→cleanup) | `_read_all_blocks` `:771-857`, `_loop_rekick:633-641` |
| **S5** | **cleanup membrane (the OP RESULT) → the sequencer's decoded word-lines** | **the per-query normalization is the on-bridge `input_divisive_norm` divide (Option 4); the threshold = the Izhikevich rheobase** | **on-bridge normalization (control), BUT the score array is read to host + re-driven onto a SEPARATE score bridge — see R1** | `_ensure_sequencer:1009,1016,1023`; `block_cleanup_scores` does `to_host(cp_membrane_potential_v)` (`_phaseB_onebrain_sequencer_derisk.py:86`) |
| **S6** | **cue + decoded lines → spiking gated-match → BG WTA → {ans block / abstain}** | **the Phase-B gated-disinhibition match cascade + BG first-match-priority WTA** | **SYNAPTIC (the CONTROL is on-substrate — the host `_scan` is GONE)** | `_seq_block:1041-1060` (integrated_loop path) |
| S7 | won BG channel → the emitted word | the mechanical body read (which channel fired → that block's answer role) | HOST body read — legitimate (like the nav cascade reading the winning motor pool) | `_seq_block:1060` → `decision_to_block` |

**The host `_scan` is GONE from the integrated query path** (`_seq_block`'s `if not self.integrated_loop:` host branch
at `:1034-1040` is the test-oracle/CPU escape; the `integrated_loop=True` branch `:1041-1060` is the flagship default).
The intra-query op hand-offs (S2→S3, S4's unbind→cleanup, the clause hop-1→hop-2, the reconsolidation PE) are all
spike-resident under `persistent_loop=True`.

### 1.3 The cross-region SYNAPTIC routes (the OTHER "one brain" axis) — already built both directions

- **Route A (language→action), 6-seed GO** (`spoken_instruction_nav.py`): the parser's action-ensemble FIRING opens the
  `command_route` transmission gate IN-SUBSTRATE via `couple_gate_to_indices` (`:395-403`) — the gate opens iff the
  control population's firing-rate EMA crosses threshold; **no Python reads a value and sets the gate**. The gated route's
  per-synapse current is scaled every step on GPU (`bridge.py:6009`, `effective_connections_matrix.data *
  cp_transmission_gain`). The per-step perception-render-in / action-readout-out are HOST (the legitimate sensory/body
  boundaries).
- **Route B (perception→memory), 6-seed GO** (`navigate_to_see_then_answer.py`): live `cortex_it` firing → engram-tag →
  recall via the TRAINED `cortex_it→language_output` synapses (`stimulate_tag` fires the bound ensemble → synaptic drive
  reaches `language_output`; the firing-to-firing link is on-bridge; the engram commit + the output read are host API).
- **Route C / step-3 (perception→compose)** (`navigate_to_compose_then_answer.py`): the DEFAULT `gen_spikes` mode grounds
  via the LEARNED `gen_perception→gen_concept` convergence (synaptic, NMDA-integrated); the concept SPIKES are read to host
  + a fixed projection formats them to a phasor written to `composer.concepts[obj]` (the host marshalling = R4). The
  `host_m` mode (`angle(host_M @ rate)`) is the documented host-projection closure.

### 1.4 The merged-bridge step loop — host-orchestrated by design (the navigation half)

`run_moving_goal_episode` (`g11_bg_runner.py` / `nav_conv_merged_bridge.py`) is a Python `for step in range(n_steps)`
loop: render perception → `bridge._run_one_simulation_step()` → read sel/motor spike counts → decide move → reward. The
action SELECTION is on-substrate (the spiking-WTA accumulator + commit-burst, default-on since 2026-06-19); the
per-step perception-in / body-out / reward-in are HOST (the sensory/body boundaries). This is the NAVIGATION loop, not
the conversational loop; it is a fixed episode schedule by design (one timestep = one `_run_one_simulation_step`), and
its host involvement is the legitimate environment/body boundary, NOT a between-cognitive-op host hand-off.

---

## 2. The GENUINE RESIDUAL — precisely isolated + quantified (the SURPASS practice)

Most of "the persistent integrated loop" is already built (§1). The residual is FOUR narrow gaps. For each: WHAT exactly
is still host, WHERE (file:line), HOW BIG, and whether it is a between-op cognitive hand-off (the target) vs a legitimate
sensory/body boundary (out of scope).

### R1 — the cleanup→sequencer DATA hand-off crosses HOST because the sequencer is a SEPARATE sub-bridge (the one genuine remaining host DATA seam in the deployed who/what loop)

**WHAT.** Phase C Task 1 ruled out a fixed `cp_connections` projection carrying the graded cleanup score (option a WALLED
— the point-neuron graded-magnitude limit: winner AND runner-up both suprathreshold, so a binary spike destroys the
relative magnitude). Task 2 + the S5 divnorm de-risk then closed the per-query NORMALIZATION on-bridge (Option 4). BUT
the deployed `_ensure_sequencer` builds the sequencer and the divnorm-score pool as **SEPARATE `SimulationBridge`
objects** (`build_sequencerK_bridge`, `build_divnorm_score_bridge`), distinct from the composer's own
`OneBrainComposer.b`. So the loop is: composer-bridge cleanup → `block_cleanup_scores` does
`to_host(cp_membrane_potential_v)` (`_phaseB_onebrain_sequencer_derisk.py:86`) → `make_block_drives` re-drives the score
array onto the divnorm-score sub-bridge → the sequencer sub-bridge settles. **The CONTROL is on-substrate and the
per-query normalization is on-bridge, but the cleanup-score VECTOR is physically marshalled host-side between two
bridges.** This is the precise residual: it is NOT a host COMPUTATION (the normalization + match + select are all
spiking) — it is a host DATA TRANSFER between co-resident-in-principle slices that are currently separate bridge objects.

**WHERE.** `one_brain_composer.py:1009/1016` (the separate `build_divnorm_score_bridge`), `:1022` (`block_cleanup_scores`
per block → host), `:1023` (`make_block_drives` re-drives). `_phaseB_onebrain_sequencer_derisk.py:86` (the `to_host`).

**HOW BIG.** ONE score-vector read per query (K blocks × V words, read once per cue-match), the only host DATA round-trip
left in the integrated who/what query path. It is the SAME seam the design doc named "option b, the residual host DATA
read" — the divnorm de-risk closed the *normalization computation* on-bridge but the *cross-bridge data marshalling*
remains because the score pool is a separate bridge. The honest design framing ("one number is read to host between the
cleanup and the sequencer", Task 2 finding) is now downgraded to "one score VECTOR is marshalled between two bridges."

**WHY IT WASN'T DONE.** The S5 divnorm de-risk explicitly listed the follow-on "Wire Option 4 into the integrated loop's
cleanup→sequencer seam so the deployed who/what turn has zero host round-trips" as NOT on its path. `_ensure_sequencer`
wired the divnorm NORMALIZATION (the hard part) but kept the score pool + sequencer on separate bridges (the easy part
deferred). Folding them into the composer's bridge as disjoint index slices is the close.

**Between-op cognitive hand-off?** YES — this is exactly the target (an op's spiking RESULT driving the next op's CONTROL
circuit). The hardest sub-part (the graded-score → spike conversion) is ALREADY solved (the divnorm + placed-rheobase
threshold). What remains is the wiring: build the sequencer + divnorm-score pools as disjoint index slices on the SAME
`OneBrainComposer.b`, and route `cleanup-neuron → divnorm-score-pool → sequencer-cue-line` as on-bridge synapses (the
divnorm pool's firing replaces the host `make_block_drives`).

### R2 — the macro-schedule is a FIXED PYTHON PROGRAM of discrete op-windows with `_zero_rf_v_u()` resets; there is no NEURAL latch holding state across windows

**WHAT.** Even with R1 closed, the loop is NOT one continuously-running attractor; it is a fixed sequence of op-WINDOWS
(`per turn: parser window → bind/bundle resonate windows → reconstruct/unbind/cleanup resonate windows → S5 window → S6
settle window → S7 read`, design §2.4), with `_zero_rf_v_u()` resets between RF ops (`one_brain_composer.py:603`, called
at the head of every read). The op ORDER is a Python program (the same fixed schedule every turn). The design doc is
HONEST about this: "the persistent loop is NOT a single `while: _run_one_simulation_step()` for everything; it is a fixed
micro-schedule that interleaves the two [RF + Izhikevich] step paths… a turn's op ORDER is a fixed program… exactly as a
cortical-subcortical loop runs a fixed processing pipeline." The DATA-DEPENDENT branch (which fact matches / abstain) IS
on the substrate (S6 BG WTA); the op SEQUENCING is a fixed Python schedule. Relatedly, the parser→compose WM "latch" that
holds the routing gate open across the downstream read is a **Python two-window gate-hold** (drive-window then
pause-the-coupling-to-hold-the-gate-value), NOT a self-holding neural attractor (`2026-06-04-one-bridge-unification-
step2-DONE.md`; the genuine NMDA-attractor latch is deferred).

**WHERE.** `_zero_rf_v_u` `:603`; the per-op window schedule is implicit in `_read_block`/`_read_all_blocks`/`_decode_clause`
(each does `_zero_rf_v_u()` → kick → resonate → unbind-resonate → cleanup-resonate). The micro-schedule is documented in
design §2.4. The Python gate-hold latch is in the `_op_synaptic` two-window protocol (the merged-bridge step-2 hand-off).

**HOW BIG.** This is a FRAMING/architecture residual, not a host COMPUTATION leak. The two `rf_resonate_steps` step paths
genuinely bypass `_run_one_simulation_step` (and thus `_apply_gate_couplings`) by design (`bridge.py:5607-5612`), so a
single `while: _run_one_simulation_step()` for everything is not even the right target on this substrate. The honest
question is whether a fixed op-schedule with a Python `for` over the windows is acceptable "one brain" (the design argues
yes — a fixed pipeline order is biological; only the data-dependent control must be neural, and it is). The harder version
(a genuine neural latch + a neural op-sequencer that orders the windows from substrate state) is the deep, high-variance
extension — and the design + the 2026-05-18/19 "integrated-loop necessity instrument" findings already flag that a fully
self-sequencing loop runs into the point-neuron graded-credit / WM-selectivity walls (5 convergent negatives,
`2026-05-19-FIFTH-convergent-...integrated-loop-necessity...biologically-unsatisfiable-by-the-CLS-division-of-labor.md`).

**Between-op cognitive hand-off?** PARTIAL — the op-ORDER being a Python schedule is the residual; but it is the
biologically-defensible "fixed pipeline" part, NOT the data-dependent control (which is neural). This is the disguised-
boundary candidate: the comfortable verdict "a fixed micro-schedule IS one brain" should be SURPASS-tested (§4 R2 option),
but the prior 5-convergent-negative on the self-sequencing instrument means it is likely a characterized boundary, not a
cheap win — flag-not-chase unless the owner prioritizes the deep version.

### R3 — S5's on-bridge closure is validated only at D=64/K=2; the PRODUCTION scale (D=2048/V=320, K up to 32) is unvalidated

**WHAT.** The S5 divnorm GO is at D=64, K=2, 12-word vocab, 3 seeds (`2026-06-20-S5-divisive-norm-derisk.md` "HONEST
SCOPE"). The named follow-on "Scale check at D=2048 / V=320 — the per-query peaks grow but the saturated ratio is
scale-free, so the same `gain≈0.05` operating point SHOULD transfer; confirm the firing-band placement holds and the moat
stays 0-FA at scale" was NOT done. The flagship `consolidated_320_conversation_demo` runs `integrated_loop=True` at V=320
(CYCLE 190 GO 3/3) — BUT that GO predates the divnorm wiring and the demo's recall leans partly on the host `query_agent`
scan (purity #4 R-A1 finding: even at V=320 `query_patient` over-abstains 2/8 on the K=8 demo set). So the divnorm-S5
firing-band placement at production scale is genuinely unconfirmed.

**WHERE.** `_phaseC_S5_divnorm_derisk.py` (D=64 only); the named follow-on in `2026-06-20-S5-divisive-norm-derisk.md`
"Follow-on (NOT on this de-risk's path)". The V=320 over-abstention is `4b9bf261` / `d5ce2882` (purity #4 R-A1/R-B).

**HOW BIG.** A validation gap, not a build. The mechanism is scale-free in theory (the saturated divnorm ratio is a
dimensionless rank-preserving number); the risk is the firing-band BASIN narrowing at V=320 (more cleanup lines → the
mean-pool divisor shifts) needing the NEF input-norm FS pool (Option 1, `2026-06-05-composer-cleanup-NEF-GO.md`, validated
27/27 @ D=2048) as the graded-pool fallback. Quantified: a 6-seed GPU run at D=2048/V=320/K∈{8,32} confirming ==host +
moat 0-FA, with the NEF-FS pool as the named fallback if the mean-pool basin is too narrow.

**Between-op cognitive hand-off?** N/A — this is a scale-validation of an already-closed seam. Pairs naturally with R1
(validate R1's intra-bridge S5 at production scale in the same run).

### R4 — the cross-region grounding (perception→compose) marshals the concept code host-side

**WHAT.** Route C's DEFAULT `gen_spikes` mode: the LEARNED `gen_perception→gen_concept` convergence is synaptic (the
load-bearing transform), but the gen_concept SPIKES are read to host and a fixed projection formats them to a phasor
written to `composer.concepts[obj]` (`navigate_to_compose_then_answer.py:218,223-228`). This is the perception→memory
DATA hand-off — the percept's neural code crosses host to enter the composer's codebook.

**WHERE.** `navigate_to_compose_then_answer.py:183-228` (`read_gen_concept_spikes` + `gen_grounded_phases`), `:420`
(the codebook write).

**HOW BIG.** One concept-code read per perceived object (in-episode grounding). It is the cross-region analogue of R1
(an op's spiking RESULT crossing host to become the next op's operand). Lower priority than R1 because it is on the
nav-perception→compose route (not the core who/what conversational turn), and the load-bearing transform (the learned
convergence) is already synaptic — only the code-formatting read is host.

**Between-op cognitive hand-off?** YES (cross-region), but on a secondary route. Naturally folds into R1's "result→operand
on-bridge" mechanism if R1 is built generically.

### R5 — the per-op cleanup body-read (`to_host(cp_membrane_potential_v)` + `_select`) — a LEGITIMATE body-read boundary (mostly out of scope)

**WHAT.** Every read ends with `mem = to_host(cp_membrane_potential_v)` + `_select` (host argmax, OR the spiking NEF/WTA
`_spiking_select` when `enable_spiking_cleanup=True`, default-on in the flagship). When the spiking WTA is on, the SELECT
is on-substrate (argmax-over-FIRING, a body read of which concept-neuron won); the residual `to_host` is reading which
neuron fired = the legitimate body-read boundary (like the nav cascade reading the winning motor pool).

**WHERE.** `_select:684-690`, `_spiking_select:652-682`, the `to_host` at `:803/838/1108`.

**HOW BIG.** Per-op, but it is the BODY-READ boundary by the BRAIN-BASED-ONLY standard (host reads which neuron fired;
the brain did the selection). With `enable_spiking_cleanup=True` (flagship default) this is already spiking. Effectively
CLOSED (the remaining `to_host` is the legitimate "read which channel won" boundary). Listed for completeness; not a
target.

### Residual summary

| residual | what is still host | between-op cognitive hand-off? | size | status |
|---|---|---|---|---|
| **R1** | cleanup-score VECTOR marshalled host-side BETWEEN the composer bridge + a separate sequencer/divnorm bridge | YES (the target) — but the hard graded→spike part is ALREADY solved (divnorm) | 1 vector/query | **the cheap, high-leverage close: fold the sub-bridges into one** |
| R2 | the op-ORDER is a fixed Python schedule; the WM "latch" is a Python gate-hold | PARTIAL (the defensible fixed-pipeline part) | architecture/framing | likely a characterized boundary (5 convergent prior negatives); flag-not-chase |
| R3 | — (validation gap) | N/A | D=64→D=2048 | scale-validate R1's S5 (fold into R1's run) |
| R4 | perceived concept-code read host-side (gen_spikes) | YES (cross-region, secondary route) | 1 code/object | folds into R1's mechanism; lower priority |
| R5 | read which cleanup neuron fired | NO (legitimate body read; spiking-WTA select is on-substrate) | per-op | effectively closed |

**⇒ The genuine residual that is BOTH a real between-op hand-off AND cheaply closable is R1.** R2 is the deep,
likely-bounded extension; R3 is a scale-validation that rides R1; R4/R5 are secondary/closed.

---

## 3. THE BIOLOGY (catalog-first)

Searched `sim-catalog/references/feature-catalog.md` (clusters A–Q) + Kandel 6e notes. The mechanisms that bear on a
persistent integrated spiking loop (between-area synaptic relay; sustained recurrent activity; routing without host
control):

| cluster/ID | title | mechanism | Kandel | sim status (per catalog) |
|---|---|---|---|---|
| **A.04** | BG output disinhibition is SELECTIVE — competitive WTA at GPi/SNr | the "selected" channel = strongest striatal inhibition → GPi/SNr silenced → target released; **"selection is an emergent property of the entire REENTRANT network"** | Ch 38 | the S6 sequencer IS this (BG first-match WTA) — built |
| **A.05** | REENTRANT cortico-BG-thalamo-cortical loops — parallel channels | Alexander/DeLong: motor/cognitive/limbic loops topographically segregated cortex→STR→GPi/SNr→thalamus→cortex; the closed reentrant loop is the substrate for sustained, self-sequencing activity | Ch 38 pp 943–948 | partial (the conv loop uses the BG-WTA motif for op-selection; the full thalamo-cortical reentry for op-SEQUENCING is the R2 deep version) |
| **A.07** | subcortical BG loops — superior colliculus, brainstem | the BG→SC orienting loop (the routing fabric for "which target") | Ch 38 | the nav SC is built (spiking SC); the conv analogue = the sequencer |
| **G.06** | PFC working memory — SUSTAINED delay-period activity | dlPFC recurrent excitation → persistent firing across a delay when the stimulus is absent; D1-DA modulated | Ch 34 pp 827–842 | partial (60-neuron recurrent PFC; the genuine self-holding NMDA-attractor LATCH for R2 is the deferred piece) |
| **G.08** | working memory in PFC — persistent activity for active maintenance | dlPFC/vlPFC holds "what"/"where"/conjunctions across delays (seconds) | Ch 52 pp 1292–1294 | partial (the WM latch is currently a Python gate-hold, R2) |
| **D.05** | CA3 recurrent collaterals — autoassociative ATTRACTOR | LTP-modifiable recurrent excitation → pattern completion: a partial cue converges to the full stored pattern (theta-paced) | Ch 54 pp 1342,1360–1361 | partial (the store IS an attractor-like complex-synapse memory; the explicit recurrent-attractor convergence is not separately built) |
| (C.12 region) | Dehaene GLOBAL WORKSPACE / access | selective gating of a representation to a frontoparietal global workspace; ignition = a sustained reverberant broadcast | Ch (consciousness) | the conceptual frame for "one loop"; the sequencer + transmission gates are the routing primitives |

**The key biological read:** the project's S6 sequencer is exactly the catalog's **A.04/A.05 reentrant BG-thalamo-cortical
selection-by-disinhibition** — selection as an emergent property of a reentrant loop, NOT a host `if`. That mechanism is
BUILT. The R1 close (cleanup→sequencer on-bridge) is the cortico-striatal relay (A.06 topography: the cleanup membrane is
the "cortical" result; the divnorm-score pool → sequencer is the cortico-striatal projection). The R2 deep version (a
self-holding WM latch + a self-sequencing loop) maps to G.06/G.08 sustained PFC activity + A.05 full reentry — and the
project's own 2026-05-19 findings already flag that a FULLY self-necessitating integrated loop hits the point-neuron
graded-credit / CLS-division-of-labor wall (the honest boundary). **So the catalog confirms: the routing + selection
biology is built (A.04/A.05/A.07); the residual R1 is a cortico-striatal relay wiring; the R2 deep version is the
sustained-PFC-attractor frontier the project has already mapped as hard.**

---

## 4. RANKED CHEAP-FIRST OPTIONS for closing the genuine residual

Each: the mechanism, the reusable machinery, the expected behavioral signature, the anti-cheat. All reuse-by-import; NO
`sim/` edit expected (matching Phase A/B/C). The moat is the HARD gate everywhere (0 false-accepts — never weakened).

### OPTION 1 (TOP, RECOMMENDED) — fold the sequencer + divnorm-score pools into the composer's ONE bridge as disjoint index slices, so the cleanup→sequencer DATA hand-off is intra-bridge spikes (closes R1)

- **Mechanism.** Build the K-way sequencer + the divnorm-score pool as DISJOINT index ranges on `OneBrainComposer.b`
  (today they are separate `SimulationBridge` objects). Wire `cleanup-neuron (c_base+ri·V+j) → divnorm-score-pool →
  sequencer-decoded-line` as on-bridge synaptic routes (the divnorm pool's firing IS the decoded-line drive — it replaces
  the host `make_block_drives`). The per-query normalization (already on-bridge via `input_divisive_norm`) + the placed
  rheobase threshold convert the graded cleanup score to a spiking decoded-line drive WITHOUT the host `to_host` +
  `make_block_drives` array marshalling. The hard graded→spike conversion is ALREADY solved (the divnorm de-risk); this is
  the wiring to make it intra-bridge.
- **Reusable machinery.** `build_divnorm_score_bridge` + `build_sequencerK_bridge` (re-home as slices, not bridges);
  `couple_gate_to_indices`/`couple_gate_to_pool` (`bridge.py:3141,3164`) for the disinhibition→route on the merged slice;
  the masked RF ops + `cp_izh_c_reset` slice-aware reset (the design's housekeeping invariant); `_ensure_sequencer` (the
  existing build/cache hook — extend to build the slices on `self.b`); `couple_gate_to_indices` (`unified_brain_bridge.py:123`)
  for index-based coupling on the merged bridge.
- **Expected behavioral signature.** The integrated `query_patient`/`query_agent` == host on who/what (the SAME answers as
  today), with ZERO `to_host` of the cleanup score between cleanup and sequencer (the loop has zero host DATA round-trips
  in the query path; only S1 text-in + S7 body-read remain, both legitimate boundaries).
- **Anti-cheat.** (1) **==host 6 seeds** at K=2, extend K∈{4,8}; (2) **MOAT 0 false-accepts** (HARD) on every absent/cross
  cue; (3) **lesion the cleanup→score-pool route → decoded lines silent → ABSTAIN** (fails safe, never confabulates); (4)
  **OFF == byte-identical** (the separate-bridge path stays the revertible escape); (5) **permuted-store carries content**;
  (6) the slice-aware reset invariant checked by the moat gate (the Izhikevich sequencer slice resets to `cp_izh_c_reset`
  ≈ −65 mV, NOT 0 — the Phase-B discipline; getting it wrong is a moat-leak, so the moat gate catches it).
- **`sim/` edit?** Likely NONE (reuse `cp_connections` + the existing gate couplings + `input_divisive_norm`). The ONE
  contingent edit (the design's §3.2 edit 1: mask the `rf_kick` spike-tracker re-init at `bridge.py:5537-5540` so a re-kick
  of one RF group doesn't clobber a holding disjoint RF group) is DEFERRED until a multi-op de-risk shows it's required —
  Phase C Task 2 did NOT need it at K=2; the slice-folding may surface it at K=8. If needed: ~6 lines, default `None` =
  byte-identical, `test_rf_*` pins bit-identity, isolated commit for byte-review.

### OPTION 2 (the scale-validation, RIDES Option 1) — validate the folded S5 at production scale D=2048/V=320 (closes R3)

- **Mechanism.** Run Option 1's folded loop at the production composer scale (D=2048, V=320, K∈{8,32}), confirming the
  divnorm firing-band placement holds (the saturated ratio is scale-free in theory) and the moat stays 0-FA.
- **Reusable machinery.** The flagship `consolidated_320_conversation_demo` harness (`--composer onebrain
  --integrated-loop`); the NEF input-norm FS pool (`_spiking_cleanup_nef.py`, Option 1 of the S5 deep research, validated
  27/27 @ D=2048) as the named graded-pool FALLBACK if the mean-pool divnorm basin is too narrow at V=320.
- **Expected signature.** ==host on the 320-concept who/what matrix, moat 0-FA, divnorm firing-band placement reported
  (winner supra-rheobase, runner-up sub-rheobase) at V=320.
- **Anti-cheat.** Moat 0-FA HARD; the firing-band placement reported per seed; the NEF-FS fallback gated on the mean-pool
  basin failing. 6 seeds.
- **`sim/` edit?** NONE (Option 1's wiring + the existing NEF-FS pool if the fallback is reached).

### OPTION 3 (the cross-region analogue, secondary) — make Route C's perceived-concept hand-off intra-bridge (closes R4)

- **Mechanism.** Apply Option 1's "an op's spiking RESULT drives the next op's operand on-bridge" generically to the
  perception→compose route: route the gen_concept SPIKES into the composer's `fill_*` register via on-bridge synapses
  (instead of `read_gen_concept_spikes` → host projection → codebook write).
- **Reusable machinery.** The grounding projection (`gen_grounded_phases`); the `_dev_rekick_into` register-handoff
  primitive; the merged-bridge co-residence.
- **Expected signature.** The held-out compose == today (1.000), with the percept→composer code crossing as spikes not a
  host array. **Anti-cheat:** lesion the route → compose collapses; moat 0-FA; 6 seeds.
- **`sim/` edit?** Likely NONE; lower priority (secondary route).

### OPTION 4 (the DEEP, likely-bounded extension — SURPASS-test before accepting R2 as a boundary) — a self-sequencing loop + a neural WM latch

- **Mechanism.** Replace the fixed Python op-schedule with substrate-driven op-sequencing (a thalamo-cortical reentrant
  loop that, from the match RESULT, drives the NEXT op-window's gate) + a genuine NMDA-attractor WM latch (self-holding,
  no Python gate-hold). This is the R2 residual.
- **Honest prior.** The project's 2026-05-18/19 "integrated-loop necessity instrument" arc returned **5 convergent
  negatives** (the self-necessitating loop hits the point-neuron graded-credit + WM-selectivity + CLS-division-of-labor
  walls). So this is LIKELY a characterized boundary, not a cheap win.
- **The SURPASS move (mandatory before accepting "a fixed micro-schedule IS one brain" as a boundary).** ISOLATE the
  genuine residual: the op-ORDER being a fixed pipeline is biologically DEFENSIBLE (a cortical-subcortical loop runs a
  fixed pipeline; only the data-dependent control must be neural, and it IS — the BG WTA). The genuinely-irreducible part
  is whether the WM latch can be a self-holding neural attractor vs a Python gate-hold — and the catalog (G.06/G.08
  sustained PFC) + the NMDA-attractor latch the project ALREADY validated at dt=1.0 (the 2026-06-04 step-3 dlPFC merge)
  suggest the LATCH is closable (a small targeted de-risk: replace the two-window gate-hold with the validated NMDA
  attractor on the routing gate). The full self-SEQUENCING loop is the harder, likely-bounded part.
- **Recommendation.** Do NOT build the full self-sequencing loop now (5 prior negatives = a characterized boundary;
  research-gate it if ever prioritized). A SMALL carve-out — the neural WM latch (replace the Python gate-hold with the
  validated NMDA attractor) — is a defensible later de-risk, NOT on the cheap-first path.
- **Anti-cheat.** If ever attempted: the latch HOLDS the gate across the read without Python pausing the coupling (lesion
  the recurrent attractor → the gate flickers shut → the read starves); moat 0-FA; 6 seeds.

### Ranking rationale

Option 1 closes the ONE genuine remaining host DATA seam in the deployed who/what loop, the hard part (graded→spike) is
already solved (divnorm), it is reuse-by-import with NO `sim/` edit expected, and it directly delivers the owner's "real
one brain" (zero host round-trips between cognitive ops in the query path). Option 2 rides it (validate at production
scale). Option 3 is the secondary cross-region analogue. Option 4 (R2) is the deep, likely-bounded extension — its WM-latch
sub-part is a defensible small later de-risk; its full self-sequencing form is a characterized boundary (5 prior negatives).

---

## 5. THE SINGLE RECOMMENDED CHEAP-FIRST DE-RISK

**Fold the K-way sequencer + the divnorm-score pool into the `OneBrainComposer` bridge as disjoint index slices, and route
the cleanup→score-pool→sequencer hand-off as on-bridge synapses — closing R1 (the last host DATA seam in the integrated
who/what query path). Validate at the cheap-first K=2 first (the Phase-C bar), then K∈{4,8}.** (Option 1.)

**Why this cut.** It is the smallest experiment that closes the genuine residual the SURPASS analysis isolated; the hard
sub-part (the per-query-normalized graded→spike conversion) is ALREADY de-risked GO (the divnorm + placed-rheobase, both
backends, moat 0-FA); the remaining work is wiring the sub-bridges into one (matching the merged-bridge co-residence the
project already does for nav+conv+composer); and it directly delivers "zero host round-trips between cognitive ops in the
who/what query path." It reuses the exact Phase-C Task-2 harness (`_phaseC_task2_wholeturn_loop.py` `LoopComposer`) — swap
the option-(b) host `scores_to_drive` for the on-bridge cleanup→divnorm-score-pool route.

**Pre-registered GO bar (the moat is the HARD gate):**
1. **==host** on who/what at K=2 (extend K∈{4,8}), **6 seeds** (the cleanup/match cascade is noise-sensitive — a
   distribution, not an exact identity).
2. **MOAT 0 false-accepts** (HARD) — every absent-agent / absent-action / cross cue abstains (the BG WTA selects the
   abstain channel). A single false-accept at any seed is a FAIL (never traded for a pass).
3. **ZERO `to_host` of the cleanup score between cleanup and sequencer** — assert the score-pool firing (not a host array)
   drives the decoded lines (the close itself).
4. The full anti-cheat battery (Option 1's): cleanup→score-pool LESION fails safe (abstain on a present cue); store-lesion
   collapses recall; permuted-rule inverts; permuted-store carries content; OFF==byte-identical (the separate-bridge path
   is the revertible escape); the slice-aware Izhikevich reset (`cp_izh_c_reset`) checked by the moat gate.

**Honest-negative escape.** If folding the score pool onto the composer bridge surfaces a register-isolation / phase-
coherence break on the longer chain (the design §6 central risk — a re-kick clobbers a holding RF group), the deliverable
is the precise seam boundary (the separate-bridge op-at-a-time path is the production form; the single-bridge integration
is the ceiling) + the contingent `rf_kick` tracker-mask `sim/` edit (design §3.2 edit 1, ~6 lines, default-byte-identical,
isolated commit for byte-review) as the targeted fix to try. Either way the moat is never weakened to manufacture a pass.

**Backend.** `SIM_BACKEND=cupy` for the real co-resident run (the parser trains on the CuPy substrate); the pure-algebra
parity sub-steps can run numpy (the exact oracle). 6 seeds for the variable match/cleanup effects.

**NOTE on GPU:** a DA-recall-vigor GPU acceptance run is currently in flight (`bvlajgc53`). This de-risk is the NEXT build
AFTER that acceptance lands + the owner steers — it is NOT to be launched concurrently (GPU contention).

---

## 6. Bottom line

The persistent integrated spiking loop — the REAL one-brain headline — is **LARGELY ALREADY BUILT and was de-risked GO
on 2026-06-19/20 (Phase C)**: the who/what conversational turn runs as ONE persistent loop on the production
`OneBrainComposer` with the host control orchestrator (`_scan`) GONE (`integrated_loop=True`, the spiking BG sequencer),
the intra-query op hand-offs spike-resident (`persistent_loop=True`, byte-identical to host), and the per-query
normalization closed on-bridge + WIRED INTO PRODUCTION (`_ensure_sequencer` builds the `input_divisive_norm` score pool).
Phase C Task 2 is GO 6-seed, moat 0-FA, full anti-cheat battery. This is the SAME re-frame that hit #6 and Tier 1: the
comfortable "big unbuilt piece" verdict does not survive inspection. The GENUINE residual is FOUR narrow gaps (§2), of
which exactly ONE is both a real between-op hand-off AND cheaply closable: **R1 — the cleanup→sequencer DATA hand-off
crosses host because the sequencer + divnorm-score pools are SEPARATE sub-bridges**; the hard graded→spike sub-part is
already solved (divnorm), so the close is wiring them into the composer's ONE bridge as disjoint slices
(reuse-by-import, NO `sim/` edit expected). R3 (scale-validate at D=2048/V=320) rides R1; R4 (the cross-region grounding
hand-off) is the secondary analogue; R2 (a self-sequencing loop + a neural WM latch) is the deep, likely-bounded extension
the project has ALREADY mapped as hard (5 convergent negatives) — its WM-latch sub-part is a defensible small later
de-risk, its full form a characterized boundary. **Recommended cheap-first de-risk: Option 1 (fold the sub-bridges into
one, close R1), K=2 first then K∈{4,8}, 6 seeds, moat 0-FA HARD, OFF==byte-identical, with the honest-negative escape +
the contingent `rf_kick`-tracker-mask `sim/` edit byte-reviewed if a multi-op register break surfaces.** Catalog: the
routing + selection biology (A.04/A.05/A.07 reentrant BG-thalamo-cortical selection-by-disinhibition) is BUILT (the S6
sequencer IS it); R1 is the cortico-striatal relay wiring; the R2 deep version maps to sustained-PFC G.06/G.08 +
full-reentry A.05 (the mapped-hard frontier).

Sources (verified against the actual text): the production composer (`one_brain_composer.py:87,184,405-413,538-558,
607-641,1009-1027,1029-1060`); the Phase-C design (`2026-06-19-tier2-phaseC-integrated-loop-design.md`); Task 1 / the S5
wall (`2026-06-19-phaseC-task1-S5-seam-derisk.md`); Task 2 / the whole-turn loop GO (`2026-06-19-phaseC-task2-wholeturn-
loop.md`); the S5 deep research (`2026-06-19-S5-on-bridge-normalization-deep-research.md`); the S5 divnorm close
(`2026-06-20-S5-divisive-norm-derisk.md`); the pre-build scoping (`2026-06-19-tier2-persistent-integrated-loop-scoping.md`);
the cross-region routes (`spoken_instruction_nav.py:395-403`, `navigate_to_see_then_answer.py`,
`navigate_to_compose_then_answer.py:183-228`); the merged-bridge step loop (`nav_conv_merged_bridge.py` /
`g11_bg_runner.py` `run_moving_goal_episode`); the WM-latch Python gate-hold (`2026-06-04-one-bridge-unification-step2-
DONE.md`); the self-necessitating-loop negatives (`2026-05-19-FIFTH-convergent-...integrated-loop-necessity...md`); the
V=320 over-abstention (purity #4 R-A1/R-B, commits `d5ce2882`/`4b9bf261`); the gate primitives (`bridge.py:3141,3164,3115`,
`unified_brain_bridge.py:123`); the masked RF ops + the contingent tracker-mask edit (`bridge.py:5504,5537-5540,5601-5612`);
the catalog (`sim-catalog/references/feature-catalog.md` A.04/A.05/A.07/G.06/G.08/D.05); Stewart-Choo-Eliasmith (2012)
Spaun (BG action-selection = cognitive control); Logiaco-Abbott-Escola (2021) thalamic control of cortical dynamics;
Carandini-Heeger normalization; Niv-2007 (the #6 vigor context).
