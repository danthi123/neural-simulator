# Scoping: rung-4 — the `sim/` two-compartment HTM Temporal Memory port (dAP-plateau neuron / permanence segments / per-column WTA)

**2026-07-02 (read-only deep-research + code-audit subagent; NO code edited, NO experiment run, NO commit).** This is
the standing research gate BEFORE the protected `sim/` mechanism build that realizes the fully-spiking, unsupervised,
self-organizing HTM Temporal Memory (Bouhadjar-Diesmann 2022's spiking model) on the real `SimulationBridge`. It maps
what to REUSE vs ADD, the cheap-first single-variable gated de-risk order (riskiest piece first), the RESONATE_AND_FIRE
guarded-`NeuronModel` precedent as the template, and honest risks. File:line cited for every load-bearing claim; APIs
verified against the actual code.

---

## 1. Goal + what is de-risked

**Goal (rung-4):** realize the numpy HTM Temporal Memory faithfully on the ONE spiking substrate (`sim/`), not a numpy
toy — the master-directive single-spiking-substrate path.

**What is DE-RISKED (all numpy, all GO, all committed — the port composes proven mechanisms, it does not invent one):**
- **EMERGE-9b** (`research/runners/_emerge9b_htm_faithful_derisk.py`): the faithful multi-segment HTM-TM self-organizes
  context-specific high-order prediction on overlapping sequences, unsupervised + local + no-teacher; 6 seeds, up to 8
  sequences, shared context to 24 steps (`2026-07-02-emerge9b-htm-faithful-GO.md`).
- **EMERGE-9c** (`_emerge9c_spiking_tm_derisk.py`): SPIKING INFERENCE reproduces it exactly — LIF somas + a distal
  dendrite PLATEAU (dAP) that pre-depolarizes predicted cells + per-column WTA inhibition (dAP-primed cells win the
  spiking competition → sparse context-specific firing; unpredicted → burst = mismatch). `SpikingTM._spiking_winners`
  (`_emerge9c…:49-71`) is the reference to port. (`2026-07-02-emerge9c-spiking-tm-rung3b-GO.md`.)
- **EMERGE-9d** (`_emerge9d_spiking_learning_derisk.py`): SPIKING LEARNING reproduces it — the Bouhadjar THREE-TERM
  permanence rule (`SpikingLearnTM.train_sequence_spiking`, `_emerge9d…:42-99`): (1) STDP-windowed potentiation to
  prior-WINNER spikes; (2) constant presynaptic depression; (3) dAP-rate homeostasis (per-cell low-pass predictive
  rate `z`; potentiation scaled by `(z*-z)`). 1.000 across 6 seeds, scales to 8 sequences, dAP-lesion collapses
  everywhere (`2026-07-02-emerge9d-fully-spiking-tm-rung3d-GO.md`).

So both inference and learning are fully spiking, local, unsupervised, self-organizing — validated in numpy. The numpy
reference (`SpikingLearnTM`) is the numerical ORACLE the `sim/` port must match (spiking-vs-numpy branch-acc parity).

**Scope discipline (non-negotiable):** every `sim/` edit is ADDITIVE + default-off + byte-identical-when-off + guarded,
cheap-first, single-variable, gated — the same discipline as `enable_graded_dendritic_plateau`,
`enable_coincidence_detection`, `enable_rf_cudagraph`, and the RF `neuron_mask`. `sim/` edits ARE fair game for faithful
biology (the protected-module caution is anti-CHEAT, not anti-biology; `feedback_dendritic_substrate_fair_game`).

---

## 2. The three pieces — REUSE vs NEW vs the minimal guarded change

### 2.1 The two-compartment dAP-plateau neuron (the riskiest new piece)

The HTM "predictive" state = a distal-dendrite PLATEAU (dAP) that sub-threshold-depolarizes the soma so a predicted
cell fires FIRST on matching feedforward input (and wins the per-column WTA). In the numpy reference this is one line:
`drive = i_input + plateau * primed` (`_emerge9c…:61`), i.e. a per-cell additive depolarizing current on the
dAP-primed cells that lowers their effective threshold in the LIF competition.

| Need | Substrate machinery that supplies it | file:line | Reuse verdict |
|---|---|---|---|
| **The dAP plateau CURRENT** — a sustained, self-limiting depolarizing current on the "predicted" (distally-driven) cells | `fused_graded_dendritic_plateau(g, g_rise, decay, decay_rise, v, E_e, mg_conc, c_weighted, center, slope, plateau_strength)` — a GENTLE centered logistic on the WEIGHTED distal drive `c_weighted = Σ w_eff_j·x_j`, dual-exp Mg²⁺-self-limiting NMDA-plateau current toward `E_e`; wired guarded, byte-inert when off | `sim/kernels.py:280-330`; wired at `sim/bridge.py:6460-6498` (guarded by `enable_graded_dendritic_plateau`) | **REUSE as the dAP transfer.** This is precisely the biophysics of the HTM dAP: a distal (apical) synaptic drive → a graded, self-limiting NMDA-plateau depolarizing current. In the HTM the dAP is closer to all-or-none (a segment is active or not), so the STEEPER sibling `fused_coincidence_plateau` (`sim/kernels.py:252-278`, all-or-none switch `1/(1+exp(-gain·(c-k_thresh)))`) is the CLOSER match to the `act_th`-thresholded segment; graded is the fallback. Both take the same restricted-matvec `c_weighted` and the same routing mask — no new kernel needed for the plateau itself. |
| **The SOMA that spikes (events)** | Izhikevich 2007 fused dynamics + threshold/reset (`fused_izhikevich2007_dynamics_update`; the 3-branch spike-threshold + `cp_izh_c_reset` / `cp_izh_d_increment` reset) | `sim/kernels.py:31-45`; `sim/bridge.py:6595-6660` | **REUSE the spiking engine unchanged.** The dAP plateau current is ADDED to `total_input_current_pA` exactly as the graded-plateau block does today (`bridge.py:6498`), so a dAP-primed soma reaches threshold FIRST. LIF (a leaky-integrate soma, the Bouhadjar/Stuck-Wang-Naud minimal model) is a special case; the existing Izhikevich soma is a strictly-richer, already-present substitute. |
| **Co-resident special dynamics via a neuron mask** | the RF `neuron_mask` pattern: `rf_kick(neuron_mask=)`, `_rf_advance_one` masks all v/u writes to the RF slice; `_rf_neuron_mask` None ⇒ byte-identical | `sim/bridge.py:5646-5763` (esp. the `_rf_mask` write-back branch `:5751-5761`) | **REUSE the pattern (the design template).** The TM columns live on their own neuron slice; the dAP-plateau routing (the coincidence mask) already restricts the plateau to the routed (distal-afferent) neurons — so no new masking is even required if the TM is built as a routed graded/coincidence pathway. |

**What is genuinely NEW (small):** a config `enable_htm_dap` (or reuse `enable_graded_dendritic_plateau` /
`enable_coincidence_detection` directly with a distal-tagged pathway) so the plateau is driven by a DISTAL pathway that
does NOT itself force the feedforward spike — the plateau is ADDITIVE-on-top (already true today: the block adds
`I_graded_plateau` to `total_input_current_pA`, it does not replace the feedforward current). **A dedicated
`cp_v_apical` state array is NOT needed for the FIRST de-risk** — the graded/coincidence-plateau kernel already carries
the apical-drive read-out (`c_weighted` = the restricted matvec over the distal synapses) and injects it as a somatic
current, which is functionally the HTM dAP (a sub-threshold pre-depolarization that biases the WTA). A separate
`cp_v_apical` register (a true two-compartment membrane) is the FAITHFULNESS upgrade for a later rung (§5), only needed
if a spiking BOUNDARY shows the current-injection dAP is insufficient. **Verdict: for rung-4 the dAP-plateau neuron is
mostly a WIRING convention over existing kernels, not a new `NeuronModel`** — a much smaller delta than the
`TWO_COMPARTMENT_BURST` neuron scoped for Burstprop (`2026-07-01-spiking-burst-substrate-scoping.md §1.2`), because the
HTM dAP is a scalar depolarizing current, not a full second membrane with BAC coupling.

### 2.2 Distal segments as plastic permanence pathways

Each cell has multiple distal SEGMENTS; each segment = synapses to ONE prior-context SDR with PERMANENCE values
(silent until `perm_conn`; a segment is ACTIVE when `≥ act_th` of its CONNECTED synapses are from currently-active
cells — `HTM._seg_conn_active`, `_emerge9b…:76-77`). Permanences evolve by the three-term rule (`_emerge9d…:75-99`).

| Need | Substrate machinery | file:line | Reuse verdict |
|---|---|---|---|
| **Per-synapse plastic "permanence" weights on a distal pathway** | `RegionPathway(from_region, to_region, density, weight_mean, weight_jitter, plastic=True, plasticity_gate=..., coincidence_detector=...)` → CSR synapses in `cp_connections`; the framework path wraps `inject_explicit_wiring` | `sim/regions.py:251-353` (RegionPathway); `sim/bridge.py:2196` (`inject_explicit_wiring`) | **REUSE the pathway machinery.** A distal segment's synapses are a set of CSR entries whose weight IS the permanence; `perm_conn` = the connected threshold read at plateau time; `act_th` = the coincidence `k_threshold` (`coincidence_k_threshold`, `config.py`). The segment→cell structure is a routing mask + a per-post `k_thresh`. |
| **The presynaptic eligibility trace (term-1 potentiation "prior-symbol spikes within the window")** | `cp_eligibility_trace` (per-synapse; decayed by `fused_eligibility_trace_decay`) | `sim/bridge.py:504, 725`; `sim/kernels.py:406-417` | **REUSE.** The trace of the prior-symbol WINNER spikes is exactly a short-window presynaptic eligibility trace — the substrate quantity the three-term potentiation needs. |
| **The per-synapse plasticity GATE (freeze/thaw, and — critically — where a custom rule slots in)** | `cp_plasticity_rate_gain` (per-synapse float, allocated only if any synapse is gated; the STDP / eligibility / Hebbian / clip paths multiply by it, so gain=0 truly freezes) | `sim/bridge.py:2529-2557` (allocation), plus the gated clip paths (`bridge.py:6673/6990/7253` per CLAUDE.md) | **REUSE the gate + write scaffold; the RULE is new.** The three-term permanence update is a NEW fused kernel that writes `cp_connections.data` on the distal (coincidence-tagged) synapses, gated by `cp_plasticity_rate_gain`, slotted at the plasticity stage beside STDP. |
| **The dAP-rate homeostasis term (per-cell low-pass predictive rate `z`; potentiation scaled by `(z*-z)`)** | `fused_homeostasis_update` maintains a per-neuron activity EMA already; the neuromodulator/plasticity-rate-multiplier infra scales learning per-region/per-synapse | `sim/kernels.py:347-361`; `sim/neuromodulators.py` (`plasticity_rate` target) | **REUSE the EMA idiom; the `z`-scaling is new but tiny.** `z` = a per-cell low-pass of the plateau (predictive) event, a direct analogue of the firing-rate EMA the homeostasis kernel computes; feeding `(z*-z)` as a per-post multiplier into the new permanence kernel is a small addition. |

**What is genuinely NEW:** ONE fused permanence kernel `fused_htm_permanence_update` (shaped like
`fused_stdp_weight_update`, `sim/kernels.py:364-404`) implementing the three terms — (1) potentiation to
prior-WINNER-eligible presynapses scaled by the per-post homeostasis factor `(0.5+0.5·max(0,z*-z))` (the numpy `hfac`,
`_emerge9d…:77`), (2) constant presynaptic depression, (3) segment/synapse GROWTH to prior winners (the numpy
`connect_grow`, `_emerge9d…:80-83`). Growth (adding synapses to a segment) maps to the bridge's structural-plasticity /
synaptogenesis path or to pre-allocating a dense distal pathway at `perm=0` and letting the rule raise permanences from
zero (the cleaner, cheaper first cut — no runtime CSR growth). **Verdict: one new kernel + the per-cell `z` EMA;
everything else (traces, gates, CSR weights, homeostasis EMA idiom) is reuse.**

### 2.3 Per-column WTA inhibition

Each column = a subpopulation of E cells + an inhibitory neuron enforcing sparse firing (`k_win` winners). In the numpy
reference: `_spiking_winners` runs the LIF competition, and once `k_win` cells have fired the inhibition clamps the rest
(`_emerge9c…:59-70`).

| Need | Substrate machinery | file:line | Reuse verdict |
|---|---|---|---|
| **Per-column FS-interneuron WTA (fire k winners, suppress the rest)** | The navigation cascade's FS lateral-inhibition / WTA microcircuit: `cortex_FS_*` pools, `motor→FS→motor` lateral inhibition, `thal_FS` TRN-style WTA, `str_FS` PV-FSI feedforward WTA — all built as `RegionPathway`s (E pool → FS interneuron → back onto the E pool) | `research/runners/g11_bg_runner.py:645-663` (`enable_motor_lateral_inhibition`, `motor_to_fs_weight`, `n_motor_fs_per_action`; `enable_thal_lateral_inhibition`); the SC-opponent path drives `cortex_FS` (`g11_bg_runner.py:332-469`) | **REUSE verbatim (config, not code).** A per-column WTA is one FS interneuron per column with `column_E → col_FS → column_E` inhibitory edges (the exact `motor→FS→motor` / `str_FS` pattern). `k_win` sparsity emerges from the FS drive strength + threshold, the same way the nav WTA yields a single clean winner. This is a wiring recipe over existing `RegionPathway` + `BrainRegion` machinery — **no `sim/` edit at all.** |

**What is genuinely NEW:** nothing in `sim/` — a per-column WTA is a wiring pattern the concept-pool + nav runners
already build (the FS-within-kind pattern in `concept_pool` and the `motor_FS` WTA). The only work is a builder that
lays out M columns × nE cells + M FS interneurons + the intra-column inhibitory edges.

---

## 3. The cheap-first, single-variable, gated de-risk ORDER

Mirror the EMERGE / RF discipline: numpy-oracle-checkable first; each stage additive/default-off/single-variable, each
with anti-cheats; multi-seed 42/43/44. **Do not build the full TM net until the dAP-plateau neuron is validated in
isolation.**

### Stage A — a SINGLE dAP-plateau neuron reproduces the numpy dAP "fire-first" behavior (the riskiest piece; cheapest)
Build one column (nE cells) on a real `SimulationBridge` with a distal (coincidence/graded-plateau-tagged) pathway
carrying a known "predicted" pattern. Drive the column with feedforward input; measure spike ORDER and rates.
- **GO (single-variable = the dAP plateau on/off):** (i) a dAP-primed cell (distal segment active) crosses threshold
  FIRST / at a lower feedforward drive than an unprimed cell — the `plateau·primed` fire-first effect; (ii) the effect
  is MONOTONE in the plateau strength and ~zero at plateau-off (rest); (iii) the primed cell's spike does NOT require
  the plateau alone (no spurious firing from the dAP without feedforward — the multiplexing/no-confab invariant).
- **Anti-cheats:** dAP-LESION (plateau strength = 0, i.e. `enable_*_plateau` off) ⇒ no fire-first advantage (must
  collapse — the dAP is load-bearing); a bare feedforward drive with NO distal match ⇒ no plateau, no advantage (no
  spurious priming); jitter/desynchrony of the distal inputs must not fabricate a plateau (the coincidence anti-rate
  property, already validated for `fused_coincidence_plateau`).
- **Cost:** hours; CPU/numpy-backend-checkable. **This is the FIRST cheap step and the go/no-go for the whole port** —
  it tests the ONE genuinely-new biophysical behavior (distal pre-depolarization biasing the somatic competition) in
  isolation, before any TM wiring. The dAP plateau kernel already exists and is byte-inert when off, so this is a
  wiring + anti-cheat exercise, not a kernel build.

### Stage B — a per-column WTA + dAP produces the numpy SPIKING-INFERENCE selection (EMERGE-9c parity, learning frozen)
Build the full M-column × nE-cell TM with per-column FS WTA (§2.3) and the dAP-plateau distal pathway; INSTALL the
permanences learned by the numpy `SpikingLearnTM` (a fixed, known segment structure — freeze plasticity, `plasticity_gate=0`);
run the EMERGE-9c overlapping-sequence branch-prediction task.
- **GO (multi-seed 42/43/44):** the on-bridge spiking selection reproduces the numpy `run_sequence_spiking` branch-acc
  (≥0.90, == the EMERGE-9c discrete parity within tolerance) — dAP-primed cells win the on-bridge WTA and select the
  same context-specific SDR; an UNPREDICTED column BURSTS (many cells fire, the mismatch signal).
- **Anti-cheats:** dAP-lesion (plateau off) → branch-acc collapses to the Markov floor + the column bursts everywhere;
  WTA-lesion (FS off) → no sparse selection (the whole column fires, context lost); permuted permanences → chance;
  no-teacher/untrained permanences → chance.
- **Match-to-numpy:** the on-bridge branch-acc within a tolerance band of `SpikingTM.branch_acc_spiking` on the same
  task/seeds; a gap localizes LIF-vs-Izhikevich soma or WTA-timing differences.
- **Cost:** the wiring stage. Small net (M·nE ~ hundreds of neurons — EMERGE-9c ran nE=16, M≈10). CPU-smoke (1 seed)
  to shake out wiring, then multi-seed. This isolates INFERENCE (learning frozen) — the second single variable.

### Stage C — the three-term permanence rule LEARNS the segments on-bridge (EMERGE-9d parity; the decisive stage)
Turn ON the new `fused_htm_permanence_update` (thaw `cp_plasticity_rate_gain`) + the per-cell `z` homeostasis EMA;
present the sequences; let the permanences + segment growth self-organize FROM SCRATCH on the bridge (no installed
weights). Run the exact EMERGE-9d task/splits/seeds.
- **GO (multi-seed 42/43/44):** on-bridge self-organized branch-acc ≥0.90, == the numpy EMERGE-9d GO (1.000 at n_seq=2)
  within tolerance; dAP-lesion collapses; scales to n_seq=4/8 (as the numpy did).
- **Anti-cheats (each must hold):** dAP-lesion → Markov floor; no-teacher (present no prior-winner eligibility) → no
  net learning; permuted-sequence → chance; the locality assert (no forward-weight transpose in the rule — the numpy
  `used_transpose` stays False); a full-context oracle confirms task-learnability (else INCONCLUSIVE, not a verdict).
- **Cost:** the expensive stage (per-step integration × epochs × seeds). GPU genuinely helps here (step-loop-bound;
  the RF megakernel precedent `bridge.py:5833` is the launch-bound lever IF slow — but do NOT pre-optimize; smoke → GPU
  multi-seed → measure → fuse only if needed). This turns the CONFIRMED numpy spiking-learning result into a real
  `sim/` result.

### Stage D — scale / persistence / real sequences (only after Stage C GO)
Longer shared context, more sequences, a real-vocabulary/corpus-fragment stream on one substrate (the COMMUNICATION
target from the 9d finding), persistence via `BridgeLineage`. Honest scaling limits documented, not overclaimed.

---

## 4. The precedent — how RESONATE_AND_FIRE was added guarded (the template)

The RF `NeuronModel` is the exact, owner-approved template for adding a guarded new spiking behavior additively:
1. **Enum value, opt-in only.** `NeuronModel.RESONATE_AND_FIRE` added to the enum with a comment "Opt-in only;
   Izhikevich/HH/AdEx unaffected" (`sim/enums.py:11-15`). *If the HTM port ever needs a true two-compartment membrane
   (§5), a `NeuronModel.TWO_COMPARTMENT_DAP` follows this template — but rung-4 likely does not need it (§2.1).*
2. **A dispatch BRANCH in the neuron-dynamics step, byte-inert unless selected.** The RF branch is
   `elif cfg.neuron_model_type == NeuronModel.RESONATE_AND_FIRE.name:` at `sim/bridge.py:6781-6795`, sitting after the
   IZHIKEVICH / HODGKIN_HUXLEY / ADEX branches — the four models are mutually exclusive, so a default (Izhikevich)
   config never reaches the RF code.
3. **New state carried in EXISTING arrays where possible + lazy-init.** RF reuses `v`/`u` for the complex state (no new
   membrane arrays); trackers (`cp_rf_prev_im`, `cp_rf_fired`, `cp_rf_spike_step`) lazy-init on first step
   (`bridge.py:6789-6794`). *The HTM analogue: reuse `cp_conductance_g_graded_plateau` for the dAP; a per-cell `z` EMA
   is the only genuinely-new state array.*
4. **A `neuron_mask` slices the special op for CO-RESIDENCE, `None` = byte-identical.** `rf_kick(neuron_mask=)` +
   `_rf_advance_one` write v/u only for masked neurons; `_rf_neuron_mask is None` short-circuits to the byte-identical
   path (`bridge.py:5672-5678, 5751-5761`). *The HTM plateau is already routed by a per-synapse mask, so co-residence
   is free.*
5. **Opt-in fast path, default-off = byte-identical.** `enable_rf_cudagraph` megakernel; `use_mask==0` short-circuits
   (`bridge.py:5780-5786`). *The HTM permanence kernel follows the same guarded-flag idiom
   (`enable_graded_dendritic_plateau` / `enable_coincidence_detection` are the direct precedents,
   `config.py:173, 230`).*

The upshot: rung-4 rides an EVEN LIGHTER version of this precedent than Burstprop would, because the dAP-plateau kernel
and the plasticity-gate/eligibility scaffold already exist and are already guarded — the port is mostly a wiring
convention + one new plasticity kernel + one per-cell EMA, not a new `NeuronModel` with a new membrane ODE.

---

## 5. Honest risks — where the `sim/` port may diverge from numpy, and how the ladder catches it

1. **Current-injection dAP vs a true second compartment.** The cheap first cut injects the dAP as a somatic current
   (reusing the plateau kernel) rather than a separate `cp_v_apical` membrane. RISK: the continuous-time bridge dynamics
   may not reproduce the numpy discrete-substep "fire-first" ordering crisply (the plateau's ~80ms tail could over-prime
   or the WTA could tie). CAUGHT BY: Stage A's monotone-fire-first check + Stage B's match-to-numpy branch-acc band.
   MITIGATION (a later rung, only if it bites): a true `cp_v_apical` register + `NeuronModel.TWO_COMPARTMENT_DAP` (the
   RF-template §4) — a bigger but well-precedented delta.
2. **All-or-none vs graded plateau for the segment threshold.** The HTM segment is `act_th`-thresholded (nearer
   all-or-none); the graded plateau is smooth. RISK: graded may blur the segment on/off distinction. CAUGHT BY: Stage A;
   MITIGATION: use `fused_coincidence_plateau` (the STEEP switch, `sim/kernels.py:252`) instead of the graded sibling —
   both already exist, a one-flag swap.
3. **WTA `k_win` timing / sparsity.** The FS-WTA sparsity (`k_win` winners) is an operating point (FS drive × threshold);
   too strong → single winner (loses the k-of-N SDR), too weak → column doesn't sparsify (context lost). CAUGHT BY:
   Stage B's WTA-lesion + branch-acc parity; tuned like the nav `motor_to_fs_weight` sweet spot (`g11_bg_runner.py:649`).
4. **Segment GROWTH (adding synapses to a segment).** Runtime CSR growth is heavier than the numpy dict growth. CAUGHT
   BY: Stage C. MITIGATION (cheaper first cut): pre-allocate a dense distal pathway at `perm=0` and let the three-term
   rule raise permanences from zero (no runtime growth) — the "grow from zero-init" pattern the concept-pool/v16 work
   established; only fall back to structural synaptogenesis if the fixed distal fan-in is capacity-limited.
5. **`z` homeostasis cold-start (the numpy decisive fix).** EMERGE-9d found a pure-homeostatic allocation has a
   circular cold-start (`z=0` for all cells) and fixed it by bootstrapping with the committed-segment metric +
   homeostasis MODULATING (never fully gating: `0.5+0.5·(z*-z)`, `_emerge9d…:77`). The port MUST carry this exact
   modulation-not-gating form, or allocation never bootstraps. CAUGHT BY: Stage C's untrained/no-teacher anti-cheat.
6. **What could force a BIGGER `sim/` change:** if Stage A shows the current-injection dAP cannot reproduce the
   fire-first ordering (risk 1), the port needs a genuine two-compartment `NeuronModel` (`cp_v_apical` + a coupled
   somatic-apical ODE + the RF-template dispatch branch) — still additive/guarded, but ~120-180 lines + a state array,
   the same magnitude as the Burstprop `TWO_COMPARTMENT_BURST` scope (`2026-07-01-spiking-burst-substrate-scoping.md`).
   A spiking BOUNDARY where the numpy GO'd is itself a real finding (localizes the substrate limit), not a stop.

---

## 6. VERDICT — the exact first cheap-first de-risk to build

**Build Stage A first: a SINGLE dAP-plateau neuron (one column) on a real `SimulationBridge`, reusing the existing
`fused_coincidence_plateau` / `fused_graded_dendritic_plateau` kernel as the distal dAP, and show the ONE genuinely-new
behavior in isolation — a distally-primed cell fires FIRST / at lower feedforward drive than an unprimed cell, monotone
in plateau strength, zero at plateau-off, with the dAP-lesion anti-cheat collapsing the advantage (multi-seed 42/43/44,
CPU).** It is the riskiest piece, the cheapest to test, needs NO `sim/` edit (the plateau kernel already exists and is
byte-inert when off — this is a wiring + anti-cheat exercise), and it sizes/de-risks the whole port. Stage B (WTA +
frozen permanences → EMERGE-9c parity) and Stage C (the new `fused_htm_permanence_update` three-term rule →
EMERGE-9d parity) follow, each additive/default-off/single-variable/anti-cheated.

---

## Sources (cited)
- Bouhadjar, Diesmann, et al. *Sequence learning, prediction, and replay in networks of spiking neurons.* PLoS Comput
  Biol 2022 (the verified three-term permanence rule Eq. 1 — potentiation windowed + presynaptic depression + dAP-rate
  homeostasis; the spiking HTM Temporal Memory).
- Hawkins & Ahmad. *Why neurons have thousands of synapses, a theory of sequence memory in neocortex.* Front Neural
  Circuits 2016 (the multi-segment HTM-TM algorithm; distal segments = one-context SDRs; predictive = dAP).
- Larkum. BAC / distal-apical dendritic plateau (dAP). Catalog `sim-catalog/references/feature-catalog.md` **G.02**
  (active dendrites; "single-compartment everywhere today; ~10× compute per neuron"); Kandel 6e Ch 13 pp 293-298.
- Project de-risk artifacts (the CONFIRMED numpy mechanism + the oracle to match): `research/runners/_emerge9b_htm_faithful_derisk.py`,
  `_emerge9c_spiking_tm_derisk.py`, `_emerge9d_spiking_learning_derisk.py`; findings
  `research/findings/2026-07-02-emerge9{b,c,d}-*.md`; the prior two-compartment precedent map
  `2026-07-01-spiking-burst-substrate-scoping.md`.
- Substrate reuse points (file:line, this repo): `sim/kernels.py:252-278` (coincidence plateau — the steep dAP),
  `:280-330` (graded plateau — the smooth dAP), `:364-404` (STDP kernel shape for the permanence rule), `:347-361`
  (homeostasis EMA for `z`), `:406-417` (eligibility decay); `sim/bridge.py:6460-6498` (graded-plateau wiring —
  additive current), `:6595-6660` (Izhikevich soma dynamics + reset), `:6781-6795` (RESONATE_AND_FIRE guarded branch —
  the template), `:5646-5763` (RF neuron_mask co-residence pattern), `:2196` (inject_explicit_wiring), `:2529-2557`
  (per-synapse plasticity gate allocation), `:504, 725` (eligibility trace); `sim/regions.py:251-353` (RegionPathway +
  plasticity/transmission gates + coincidence_detector), `:31-130` (BrainRegion + per-region izh type/NMDA);
  `sim/enums.py:7-15` (NeuronModel enum); `sim/config.py:35` (neuron_model_type), `:173/230` (enable_coincidence_detection
  / enable_graded_dendritic_plateau guarded-flag precedent), `:231-235` (graded-plateau operating point);
  `research/runners/g11_bg_runner.py:645-663` (motor/thal FS-WTA lateral inhibition — the per-column WTA recipe).

**Do NOT commit — the controller reviews + commits.**
