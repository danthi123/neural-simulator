# Scoping: carrying Burstprop deep credit assignment onto the SPIKING SimulationBridge — the two-compartment burst pyramidal

**2026-07-01 (read-only deep-research + code-audit subagent; NO code edited, NO experiment run, NO commit).** This is
the standing research gate BEFORE the protected `sim/` mechanism build for **spiking** burst-multiplexed dendritic
credit assignment. The rate/numpy result is CONFIRMED twice (EMERGE-1b Burstprop GO, held-out 0.796 / probe 0.989,
`2026-07-01-emerge1b-burstprop-MECHANISM-CONFIRMED-partial.md`; EMERGE-3 Sacramento-Senn microcircuit GO 0.961,
`2026-07-01-emerge3-microcircuit-GO.md`), both faithful + no-weight-transport + all anti-cheats. Burstprop is the
cleaner (fully-from-scratch) mechanism and the primary candidate to carry onto the substrate — the "one brain, single
spiking substrate, emergent" path. This doc: **what to reuse vs add · the minimal faithful spiking mechanism · the
cheap-first validation ladder + gates · cost/feasibility · honest risks.** File:line + catalog IDs + papers cited.

---

## 0. TL;DR verdict (feasibility)

**FEASIBLE, and smaller than "10× compute per neuron" implies for the FIRST cheap step** — because the substrate
already has (a) a graded analog apical-drive read-out on-bridge (`fused_graded_dendritic_plateau`), (b) an
event-vs-burst-decodable short-term-plasticity substrate (per-type STF/STD, `stp_*`), (c) Izhikevich neurons that
BURST natively (per-neuron `cp_izh_d_increment`), (d) a masked/sliced-op precedent for co-resident special dynamics
(the RF `neuron_mask` pattern), and (e) a per-synapse eligibility + plasticity-gain hook where a burst-dependent
update slots in. The genuine NEW `sim/` addition is a **two-compartment spiking pyramidal `NeuronModel` (soma + apical)
+ a burst detector (event/burst multiplexing) + a burst-dependent plasticity rule (BDSP) + a fixed-random apical
feedback pathway** — but the two-compartment membrane is a small ODE delta on top of Izhikevich, the burst detector is
an ISI counter, and the feedback pathway REUSES the existing pathway machinery. Recommend the ladder start with a
**single spiking two-compartment neuron reproducing event/burst multiplexing (Stage A)** — days of work, CPU/numpy-
reference-checkable — before any deep-net or GPU build.

---

## 1. DIAGNOSIS — reuse vs new (with file:line)

### 1.1 What the substrate ALREADY gives (REUSE — cite the exact points)

| Burstprop need | Substrate machinery that supplies it | file:line | Reuse verdict |
|---|---|---|---|
| **(a) a 2nd (apical) dendritic compartment** — graded top-down drive read-out | `fused_graded_dendritic_plateau` — a GENTLE centered logistic on the WEIGHTED distributed drive `c_w=Σ w_eff_j·x_j`, dual-exp Mg²⁺-self-limiting NMDA-plateau current toward E_e; the on-bridge "GRADED ANALOG read-out of a distributed code the point-neuron soma provably can't be" | `sim/kernels.py:280-330`; wired at `sim/bridge.py:6460-6498` (guarded by `enable_graded_dendritic_plateau`, byte-inert when off) | **REUSE as the apical-drive transfer.** This is *most* of the apical compartment: a separate synaptic drive → a graded analog value injected as current. Delta: it currently ADDS to the somatic current; Burstprop needs the apical value to (i) set burst probability and (ii) NOT drive the feedforward spike (multiplexing). See §2.2. |
| **(b) event vs burst DEMULTIPLEXING on one axon** (Naud-Sprekeler 2018: STD→event rate, STF→burst rate) | per-connection-type short-term plasticity: `stp_U/tau_d/tau_f`, `enable_per_type_stp`, `stp_U_per_type=[E→E,E→I,I→E,I→I]`, facilitation `tau_f` + depression `tau_d`; Tsodyks-Markram `fused_stp_decay_recovery` | `sim/config.py:285-293, 572-578`; `sim/kernels.py:332-345` | **REUSE — the STF/STD substrate exists.** A DEPRESSING (STD) synapse transmits the event rate E; a FACILITATING (STF) synapse transmits the burst rate B. This is the biological demultiplexer the owner's standard demands; it is ALREADY per-type. Delta: the burst channel needs a *per-pathway* STF tag, not just the 4 global E/I types — a config/wiring extension, not a kernel. |
| **(c) BURSTS from a single neuron** | Izhikevich 2007 with per-neuron reset `cp_izh_c_reset` + recovery jump `cp_izh_d_increment` — the `d`-param bursting presets (`IZH2007_STN_BURST`, `IZH2007_HIPPO_PYRAMIDAL` IB-like, `IZH2007_THALAMIC_RETICULAR` LTS) | `sim/bridge.py:6595-6660` (dynamics + reset); `sim/enums.py:29-31`; presets `DefaultIzhikevichParamsManager` | **REUSE the spiking engine; the BURST DEFINITION is new.** Izhikevich already emits ≥2-spike bursts. What is new is *detecting* an event (isolated spike OR first-of-burst) vs a burst (2nd spike, ISI<θ) per neuron and forming the rates E, P, B — an ISI-counter on `cp_firing_states`. See §2.3. |
| **(d) fixed-random top-down feedback pathway (no weight transport)** | `RegionPathway` + `inject_explicit_wiring` (a separate physical projection, its own weights, never derived from forward W); the framework path IS a wrapper around `inject_explicit_wiring` | `sim/bridge.py:2196` (`inject_explicit_wiring`), `sim/regions.py` (`RegionPathway`); plasticity-gate to FREEZE it (`set_plasticity_gate`, `cp_plasticity_rate_gain=0`) `sim/bridge.py:2529-2557` | **REUSE verbatim.** The apical feedback `Y_l` (l+1→l onto the apical compartment) is just a `RegionPathway` targeting the apical drive, held FIXED (plasticity gate 0 / init-and-freeze). The rate de-risk's fixed-random `Y` (drawn from a separate seed stream, asserted independent of forward W, `_emerge1b_burstprop_derisk.py:82-84`) maps directly to a frozen pathway. No-weight-transport is STRUCTURAL (separate array), exactly as the RF complex weights are array-disjoint. |
| **(e) the plasticity WRITE hook** (BDSP: `dw ∝ E_pre·(B_post − P̄_post·E_post)`) | per-synapse `cp_eligibility_trace` (E_pre trace), `cp_plasticity_rate_gain` (per-synapse gate), reward-modulated three-factor update path, `fused_eligibility_trace_decay` | `sim/bridge.py:504, 725, 842, 2529-2557`; `sim/kernels.py:406-417` | **REUSE the trace + gate + write scaffold; the RULE is new.** The presynaptic eligibility trace `E_tilde_j` is exactly `cp_eligibility_trace`. The post-factor `(B_post − P̄_post·E_post)` (the burst-deviation-from-baseline) is a new per-neuron quantity fed into a new fused BDSP kernel, gated by `cp_plasticity_rate_gain`. Slots beside STDP at the plasticity stage of `_run_one_simulation_step`. |
| **(f) co-resident special dynamics via a neuron mask** | the RF `neuron_mask` pattern: `rf_kick(neuron_mask=)`, `_rf_advance_one` masks all v/u writes to the RF slice; `_rf_neuron_mask` None ⇒ byte-identical; the masked megakernel honors it (`use_mask==0` short-circuits) | `sim/bridge.py:5646-5763, 5801-5837` | **REUSE the pattern (the design template).** The two-compartment pyramidal is a `NeuronModel` addressing its own neuron slice; a `_burst_neuron_mask` restricts the second-compartment state + burst plasticity to that slice, leaving co-resident Izhikevich/RF byte-identical. This is the owner-approved "slice the ops, default None = byte-identical" precedent. |
| **(g) rate reference to validate against** | `sim/dendritic_neuron.py` `DendriticLayer` (two-compartment basal/apical, fixed-random `B_apical`, Larkum BAC threshold-lowering, rate); `sim/dendritic_mlp.py`; `research/runners/_emerge1b_burstprop_derisk.py` `BurstpropMLP` (the CONFIRMED rate mechanism) | `sim/dendritic_neuron.py:20-59`; `_emerge1b_burstprop_derisk.py:65-129` | **REUSE as the numerical oracle** the spiking version must match (rate-limit → spiking → within tolerance). |

### 1.2 The precise NEW `sim/` addition (what is genuinely missing)

Catalog **G.02** (active dendrites, Larkum BAC firing) is explicitly **"missing — single-compartment everywhere …
Compartmental neurons would be a major addition (~10× compute per neuron at minimum)"** (`feature-catalog.md:2644-2652`;
Kandel 6e Ch 13 pp 293-298). The genuine deltas, smallest → largest:

1. **A two-compartment spiking pyramidal `NeuronModel`** — soma (basal-driven, spikes) + apical (top-down-driven,
   sets burst probability), with **BAC coupling** (apical depolarization near a somatic spike → a burst). The
   authoritative spiking reference (Stuck, Wang & Naud 2025 — Burstprop, bioRxiv 2024, §below) confirms the minimal faithful model is a
   **two-compartment LIF WITHOUT adaptation** — the simplest dynamics that capture burst-dependent credit assignment.
   → NEW: a `NeuronModel.TWO_COMPARTMENT_BURST` (or extend Izhikevich with an apical state); state arrays
   `cp_v_apical`, an ISI/burst counter; a fused two-compartment kernel. Est. ~120-180 lines + state alloc.
2. **A burst detector / event-burst rate estimator** — per neuron, from `cp_firing_states`: event = isolated spike OR
   first spike of a burst; burst = 2nd spike with ISI < θ_burst; low-pass to event rate E, burst rate B, burst prob
   P=B/E, and the slow EMA baseline P̄. → NEW: a small stateful counter + EMA buffers. Est. ~40-60 lines.
3. **The BDSP fused plasticity kernel** — `dw = η · E_pre_trace ⊙ (B_post − P̄_post·E_post)`, gated by
   `cp_plasticity_rate_gain`, slotted at the plasticity stage. → NEW kernel mirroring `fused_stdp_weight_update`'s
   shape. Est. ~30-50 lines.
4. **The apical-feedback wiring convention** — a `RegionPathway` (or a per-pathway flag `targets_apical=True`) so a
   top-down projection lands on `cp_v_apical` instead of the somatic current, held fixed (plasticity gate 0). → mostly
   config/wiring reuse; a small routing branch in the conductance-injection block. Est. ~30-50 lines.
5. **(optional, faithfulness) STF-tagged credit synapses** — a per-pathway STF flag so the credit pathway reads the
   BURST rate while the feedforward pathway (STD) reads the EVENT rate (Naud-Sprekeler demultiplexing). → config
   extension over the existing per-type STP. Est. ~20-40 lines. Deferrable to a later faithfulness stage (the rate
   model computes E and B directly; the spiking Stage A can read both from the same detector without literal STF).

All ADDITIVE + default-off + guarded (the `enable_*` + None-mask idiom already pervasive in `bridge.py`), so a default
config is byte-identical and Izhikevich/HH/AdEx/RF are untouched — the same discipline as `enable_graded_dendritic_
plateau`, `enable_coincidence_detection`, `enable_rf_cudagraph`, and the RF `neuron_mask`.

---

## 2. THE MINIMAL FAITHFUL SPIKING MECHANISM

### 2.1 The two channels on the substrate (faithful to Payeur 2021 + Naud-Sprekeler 2018)
- **Event rate E** (feedforward) = the neuron's ordinary spiking, low-passed. Reuse the Izhikevich soma unchanged: a
  somatic spike is an EVENT. E is read by DEPRESSING (STD) downstream synapses (feedforward pathway).
- **Burst probability P** (top-down credit) = a monotone function of the APICAL potential around a baseline P0
  (P=σ(β·v_apical); P0=0.5 at v_apical=0 ⇒ ZERO net plasticity at rest — the no-spurious-learning moat). On the
  substrate: the apical potential is the graded-plateau read-out of the top-down feedback drive.
- **Burst rate B = P·E**, read by FACILITATING (STF) downstream synapses (credit pathway). BAC realizes it: an apical
  depolarization coincident with a somatic spike promotes a 2nd spike (a burst) → B rises with apical drive.

### 2.2 The apical compartment — REUSE `fused_graded_dendritic_plateau`, retargeted
The apical potential `v_apical,l` is driven by the layer-above burst-coded error through the fixed-random feedback
`Y_l` (a frozen `RegionPathway`): the descending signal is the layer-above burst-rate deviation `b_{l+1} =
E_{l+1}·(P_{l+1}−P̄_{l+1})` (rate de-risk `_emerge1b_burstprop_derisk.py:118-122`). On-bridge, the graded-plateau
kernel already turns a weighted distributed drive into a graded analog value with the right (NMDA-plateau) biophysics
(`sim/kernels.py:280-330`). The **one change vs today**: route the plateau output to a per-neuron `cp_v_apical`
register (which sets P via the burst detector) INSTEAD of adding it to `total_input_current_pA` at
`sim/bridge.py:6498` — i.e. the apical drive controls BURSTING, it does not itself force somatic spikes (the
multiplexing invariant: the credit channel must not corrupt the feedforward pass). β is the apical→burst gain
(mirrors the rate model's `beta`, `BurstpropMLP.__init__` `beta=1.0`).

### 2.3 The burst detector — new, small, on `cp_firing_states`
Per neuron maintain `last_spike_step` and a burst flag. On a somatic spike: if `(step − last_spike_step) < θ_burst`
AND the apical is depolarized (BAC), mark this spike a BURST-member (increment a burst counter at the 2nd spike);
else mark it an EVENT. Low-pass event count → E, burst count → B, P=B/max(E,ε), and P̄ = EMA(P) (the per-unit baseline;
init P̄=P0 for an unbiased first step, mirroring `BurstpropMLP.pbar` `_emerge1b…:85`). This is the spiking realization
of the rate model's `p=σ(β·v_api)`, `b=p·e`, `pbar` EMA.

### 2.4 The BDSP update — new kernel at the plasticity stage
`dw_ij = η · Ẽ_pre_j · (B_i − P̄_i·E_i)` = `η · Ẽ_pre_j · E_i·(P_i − P̄_i)`. `Ẽ_pre_j` = `cp_eligibility_trace`
(REUSE); `(B_i − P̄_i·E_i)` = the new per-neuron burst-deviation post-factor; gated by `cp_plasticity_rate_gain`
(REUSE). A fused kernel shaped like `fused_stdp_weight_update` (`sim/kernels.py:364-404`). Three-factor, fully LOCAL
(post burst rate + post event rate + pre trace) — no unit reads another's weights.

### 2.5 The feedback pathway — a frozen fixed-random `RegionPathway`
`Y_l` : layer l+1 → layer l apical (a `RegionPathway`, weights init fixed-random from a separate seed stream, held
by `set_plasticity_gate(name, 0)` / init-and-freeze). Never derived from any forward W (structural no-weight-transport,
exactly as the rate `Y` `_emerge1b…:82-84` and the RF complex weights are array-disjoint from `cp_connections`).

### 2.6 No-weight-transport + biological-locality guarantees ON the substrate
- **No weight transport (structural):** `Y_l` is a separate physical CSR pathway, never a function of the forward
  weights; assert it is never written after init (byte-check across steps) and never equals a forward W/Wᵀ (inherited
  from the rate self-check `_emerge1b…:159`).
- **Locality:** every quantity in BDSP is pre/post-synaptic to the synapse; the apical drive is a local dendritic
  read-out; the burst detector is per-neuron. No global error, no reverse-mode graph, no autodiff (same fence the rate
  modules keep, `sim/dendritic_mlp.py:1-24`).
- **No-spurious-learning moat:** at rest (no teaching) v_apical=0 ⇒ P≡P0 ⇒ P−P̄≈0 ⇒ dw≈0 (the `no_teaching_null`
  gate, made physical by P0).

---

## 3. VALIDATION LADDER (cheap-FIRST; each with GO gate + anti-cheats)

Mirror the EMERGE-1b discipline: numpy-reference-checkable first, GPU only where it earns its keep. **Do not build a
deep spiking net until Stage A passes.**

### Stage A — a SINGLE spiking two-compartment neuron reproduces event/burst MULTIPLEXING (cheapest; CPU/numpy)
Build the two-compartment kernel + burst detector; drive the basal compartment with a known rate and the apical with a
known top-down current; measure E, B, P.
- **GO:** (i) E tracks basal drive and is ~INVARIANT to apical drive (feedforward channel uncorrupted by credit);
  (ii) P (=B/E) is monotone in apical drive with P≈P0 at v_apical=0; (iii) an STD synapse's transmitted rate tracks E,
  an STF synapse's tracks B (Naud-Sprekeler demultiplexing) — OR, if literal STF is deferred, the detector's E and B
  are separable to <5% cross-talk. Multi-seed (42/43/44).
- **Anti-cheats:** apical-lesion (v_apical≡0) ⇒ P≡P0, B=P0·E (no credit modulation); a pure-basal drive with no apical
  gives P≈P0 (no spurious bursting); jitter/desynchrony must not fabricate bursts (the coincidence anti-rate property).
- **Cost:** hours; CPU. **This is the FIRST cheap step and the go/no-go for the whole substrate build.**

### Stage B — a small spiking Burstprop NET reproduces the EMERGE-1b rate result ON the substrate (the decisive stage)
Wire N_BITS → hidden → hidden → 2 two-compartment pyramidal layers on ONE `SimulationBridge`, fixed-random apical
feedback pathways, BDSP on the feedforward synapses; run the EXACT EMERGE-1 depth-2 task/splits/seeds
(`make_task`, the threshold-of-5-pair-XORs over 10 bits, 65/35 held-out, the level-1 XOR linear probe — reuse
`_emerge1_deep_dendritic_representation_derisk` VERBATIM, `_emerge1b…:50-51`).
- **GO (multi-seed 42/43/44):** spiking-Burstprop held-out **≥0.75 AND > a spiking vanilla-FA control + 0.10 AND >
  apical-lesion floor + 0.10**; the level-1 XOR **probe ≥0.70** (the intermediate features emerged); train-vs-heldout
  gap SHRINKS vs FA. The decisive within-net contrast: SAME spiking net/seed/init/feedback, only the rule differs.
- **Anti-cheats (each must hold):** (1) apical/feedback lesion (Y=0) → no-credit floor + probe ~0.5; (2) wrong-sign
  (negate the teaching) → ≤ chance+0.05 (anti-learns); (3) **no-teaching-null** (target detached → P≡P0) → no net
  learning, weights ~unchanged (the P0 moat, physical); (4) permuted-label → held-out ~chance; (5) a spiking-friendly
  oracle/reference (the rate BurstpropMLP or a fenced spiking-backprop) confirms task-learnability ≥0.80 (else
  INCONCLUSIVE, not a mechanism verdict); (6) no-weight-transport asserted (Y never written / never = forward W).
- **Match-to-rate check:** the spiking net's held-out + probe should be within a tolerance band of the rate
  BurstpropMLP on the same task (rate 0.796 / probe 0.989) — a spiking-vs-rate gap localizes burst-estimation noise.
- **Cost:** the expensive stage. Small net (hundreds of neurons — Stuck-Wang-Naud 2025 reach MNIST at this scale), many
  spiking steps × epochs. GPU genuinely helps here (the tiny rate de-risk was CPU-minutes; the spiking net is
  step-loop-bound). Start CPU-smoke (1 seed, reduced epochs) to shake out the wiring, then GPU multi-seed.

### Stage C — SCALE / persistence (only after Stage B GO)
Wider hidden (Payeur: the burst-rate estimate sharpens with ensemble/width — the rate result needed width 384 to clear
+0.10, `emerge1b…UPDATE`), a harder/deeper task, and the emergence question on a real experience stream (the
one-brain, learn-from-experience target). Confirm no catastrophic forgetting / persistence via `BridgeLineage`.
- **GO:** the depth benefit persists at scale; multi-seed; the honest scaling limits (ImageNet-gap regime) documented,
  not overclaimed.
- **Cost:** the months-scale substrate program; sized by the Stage-B GPU wall-clock.

---

## 4. COST / FEASIBILITY (honest, on one RTX 3090)

- **Compute per neuron:** catalog G.02 quotes **~10× per neuron** for compartmental neurons. The two-compartment LIF
  (Stuck-Wang-Naud 2025, no adaptation) is the LOW end — roughly 2× the somatic ODE (a second compartment) + a burst counter
  + the apical matvec. The graded-plateau apical read-out is already a per-step restricted matvec (`bridge.py:6463-
  6482`), so the apical drive cost is a KNOWN, already-paid pattern, not new.
- **VRAM:** trivial at Stage A/B scale (hundreds → low-thousands of neurons; the project routinely runs 50K-117K-neuron
  bridges). Not a VRAM-bound problem — this is a **wall-clock (step-loop) problem**, matching the memory
  `feedback_long_local_runs_ok_confirm_cloud_cause` (measure throughput/ETA; cloud only for a genuine >24GB wall,
  which this is not).
- **Where GPU helps NOW vs the tiny rate de-risks:** the rate de-risks are matmul-only, CPU-minutes. The spiking net is
  a per-step integration loop over epochs — the RF megakernel precedent (`bridge.py:5801-5831`, collapse ~15-20
  CuPy kernels/step into 1) shows the launch-bound lever if Stage B is slow; but do NOT pre-optimize — Stage A/B-smoke
  first, then measure, then (if needed) a fused two-compartment megakernel.
- **Feasibility verdict:** Stage A is days; Stage B (the decisive one) is the real build — a `sim/` `NeuronModel` +
  kernels + wiring convention, additive/guarded, plus a GPU multi-seed run. Sized and de-risked by Stage A. This is a
  legitimate protected `sim/` mechanism build for faithful biology (per `feedback_dendritic_substrate_fair_game` +
  `feedback_move_everything_to_shared_spiking_substrate`), not a shortcut.

---

## 5. HONEST RISKS — where a faithful SPIKING version might diverge from the rate model, and how the ladder catches it

1. **Burst-rate estimation noise in spikes.** The rate model uses a well-estimated fraction P; single-neuron spiking
   bursts are noisy (Payeur's clean theory assumes an ENSEMBLE — the spec's honest limit 1.2(b); the rate result
   itself needed width 384). RISK: at small width the spiking P is too noisy → the credit signal is corrupted →
   Stage B misses the +0.10 gate even though the rate mechanism GO'd. CAUGHT BY: Stage A's E/B separability metric +
   Stage B's match-to-rate band; MITIGATION is width/ensemble (a scaling knob, not a mechanism failure) — exactly the
   rate result's own resolution.
2. **Multiplexing cross-talk (the load-bearing invariant).** If the apical drive leaks into the feedforward spike
   (e.g. the plateau current is still summed at the soma), E is corrupted by credit and the whole scheme collapses.
   CAUGHT BY: Stage A GO condition (i) — E must be ~invariant to apical drive. This is WHY Stage A precedes the net.
3. **Timing / ISI-threshold sensitivity.** The burst definition (ISI<θ_burst) and the BAC coupling window are new free
   parameters; a wrong θ_burst makes every spike a "burst" (P saturates) or none (P≈0). CAUGHT BY: Stage A's monotone-
   P-in-apical-drive check + the jitter anti-cheat (desynchrony must not fabricate bursts).
4. **P0 / no-spurious-learning moat.** If the apical rest value ≠ the P0 that zeroes BDSP, the net learns garbage at
   rest. CAUGHT BY: the `no_teaching_null` anti-cheat (Stage B (3)) — the direct physical test that P0 is right.
5. **STF/STD demultiplexing fidelity (if implemented literally).** Facilitation/depression time constants must
   actually separate B from E at the operating rate. CAUGHT BY: Stage A condition (iii). MITIGATION: defer literal
   STF/STD to a faithfulness stage; Stage A/B can read E and B from the detector directly (still on-substrate, the
   biological justification cited per the owner's standard, not faked — spec §1.5).
6. **Rate-limit vs true integration (the microcircuit caveat).** The rate GO'd; a spiking version is a stricter test.
   A spiking BOUNDARY where the rate GO'd is itself a real finding (localizes the substrate limit), per the master
   directive — build-informative, not a stop. The confirming Sacramento-Senn microcircuit (EMERGE-3 GO) is the
   fallback second spiking mechanism if Burstprop-spiking is variance-limited.

---

## 6. RECOMMENDATION (for the controller to plan the build)

Build **Stage A first** (single spiking two-compartment neuron + burst detector; reproduce event/burst multiplexing;
CPU; the go/no-go). It sizes and de-risks everything. The `sim/` addition is: a `NeuronModel.TWO_COMPARTMENT_BURST`
(or an Izhikevich-apical extension) with `cp_v_apical` + an ISI/burst counter (REUSE the graded-plateau kernel for the
apical drive, the Izhikevich soma for events, the RF `neuron_mask` pattern for co-residence) + a `fused_bdsp_update`
kernel (REUSE `cp_eligibility_trace` + `cp_plasticity_rate_gain`) + a fixed-random apical `RegionPathway` convention —
all additive/guarded/default-off, byte-identical when unused. Then **Stage B** (the decisive small spiking Burstprop
net on the EMERGE-1 depth-2 task, GPU multi-seed) turns the CONFIRMED rate result into a spiking-substrate result.

---

## Sources (cited)
- Payeur, Guerguiev, Zenke, Richards, Naud. *Burst-dependent synaptic plasticity can coordinate learning in
  hierarchical circuits.* Nat Neurosci 2021. DOI 10.1038/s41593-021-00857-x; preprint bioRxiv 2020.03.30.015511.
  [BDSP rule, event/burst multiplexing, apical→P, recurrent linearization.]
- Naud & Sprekeler. *Sparse bursts optimize information transmission in a multiplexed neural code.* PNAS 2018.
  [Event rate = feedforward / burst prob = feedback; STD decodes events, STF decodes bursts — the demultiplexer.]
- Stuck, Wang & Naud. *A Burst-Dependent Algorithm for Neuromorphic On-Chip Learning of Spiking Neural Networks.* (author correction 2026-07-01: previously mis-cited as "Que, Naud" — verified Stuck/Wang/Naud via IOP + WebSearch)
  bioRxiv 2024.07.19.604308 (v1/v2); IOP Neuromorph. Comput. Eng. 2025 (2634-4386/adb511). **[THE authoritative
  SPIKING Burstprop: two-compartment LIF WITHOUT adaptation = the minimal faithful model; fully-spiking
  communication of both feedforward + learning signals; MNIST at hundreds of neurons — the direct Stage-A/B
  reference.]** https://www.biorxiv.org/content/10.1101/2024.07.19.604308v2.full.pdf
- Greedy, Zhang, Najafi, Bengio, Richards, Costa. *Single-phase deep learning in cortico-cortical networks*
  (BurstCCN). NeurIPS 2022; arXiv 2206.11769. [Rate-model formalization + scale critique; the feedback-alignment
  quality is the residual gap.]
- Sacramento, Costa, Bengio, Senn. *Dendritic cortical microcircuits approximate the backpropagation algorithm.*
  NeurIPS 2018; arXiv 1810.11393. [The confirming second mechanism — EMERGE-3 GO — a fallback spiking arm.]
- Larkum. BAC firing / two-layer apical-basal coincidence. Catalog `sim-catalog/references/feature-catalog.md`
  **G.02** (active dendrites, dendritic spikes; "~10× compute per neuron"; single-compartment everywhere today);
  Kandel 6e Ch 13 pp 293-298.
- Project rate-de-risk artifacts (the CONFIRMED mechanism + the reference to match): `sim/dendritic_mlp.py`,
  `sim/dendritic_neuron.py`, `sim/dendritic_plasticity.py`, `research/runners/_emerge1b_burstprop_derisk.py`,
  `research/findings/2026-07-01-burst-multiplexed-dendritic-credit-assignment-spec.md`,
  `2026-07-01-emerge1b-burstprop-MECHANISM-CONFIRMED-partial.md`, `2026-07-01-emerge3-microcircuit-GO.md`.
- Substrate reuse points (file:line, this repo): `sim/kernels.py:280-330` (graded plateau), `:332-345` (STP),
  `:364-417` (STDP + eligibility decay); `sim/bridge.py:6460-6498` (graded-plateau wiring), `:6595-6660` (Izhikevich
  dynamics + bursting reset), `:5646-5837` (RF neuron_mask + megakernel co-residence pattern), `:2196`
  (inject_explicit_wiring), `:2529-2557` (per-synapse plasticity gate), `:504-949` (eligibility trace);
  `sim/config.py:285-293, 572-578` (per-type STP); `sim/enums.py:7-15` (NeuronModel enum).

**Do NOT commit — the controller reviews + commits.**
