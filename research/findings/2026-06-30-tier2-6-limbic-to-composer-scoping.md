# Tier-2 #6 (limbic / dopamine / neuromodulator → conversational composer) — deep-research scoping

**Date:** 2026-06-30 (CYCLE 710 — Tier-1 just closed; first Tier-2 piece is #6, the standing TOP "one self" directive)
**Type:** READ-ONLY deep-research scoping (catalog-first → Kandel → project code → literature). NO code/`sim/`/GPU edits.
One findings doc on `main`. Trust-but-verify: every load-bearing claim cited to source read this session.
**Roadmap item:** `project_post_conversational_roadmap_tiers` Tier-2 #6 — let the SHARED reward/value/dopamine/neuromod
state (the SAME limbic brain that drives navigation) MODULATE the conversational composer. Aligns with
`feedback_move_everything_to_shared_spiking_substrate` / `project_one_brain_integrated_pipeline_and_cleanup`.

---

## 0. Executive summary (the headline + the honest re-frame)

**The prompt framed #6 as a NEW mechanism class. It is NOT — #6 is FAR more done than "scoping," and re-scoping from
zero would repeat ~12 prior findings.** The standing deep-research gate still fired correctly (a new modulation TARGET
is a new mechanism), but the deliverable's real value is to (a) report what's already BUILT + CLOSED, (b) ISOLATE the
genuine residual, and (c) RANK the *next* modulation target (the SURPASS move per the CLAUDE.md sharpening — the
comfortable "encoding-gain is closed" verdict is the START of this research, not the end).

**What EXISTS + is CLOSED (verified against source this session):**
- **Route A — READ-side DA salience PRECISION gate: PRODUCTION-WIRED + GO.** The shared spiking dopamine tightens the
  composer's no-confab abstention threshold (`MergedNavConvAgent`, `nav_conv_merged_bridge.py:1305/1458-1508`); the gate
  rises 0.060→0.250 monotone with DA 0.50→0.84, **0 false-accepts at both DA levels**, regression GREEN
  (`2026-06-18-DA-salience-gate-production-wireup-GO.md`).
- **Route B — WRITE-side DA-gated ENCODING strength: CLOSED as real-but-modest, multi-seed-confirmed.** The fact's
  stored complex-phasor magnitude is scaled by the live spiking DA at encoding (`encoding_gain_fn`, in BOTH composers).
  Deployment smoke ran on the merged bridge with the REAL spiking SNc (`2026-06-22-tier2-routeB-deployment-smoke-LATENT.md`)
  → the mechanism WORKS end-to-end (g 1.08→1.69 from the real SNc) but is behaviorally **LATENT** at the deployed D=128
  read model; the decisive content-matched 6-seed isolation (`2026-06-22-tier2-routeB-content-matched-6seed-MODEST.md`)
  found the DA encoding effect **REAL but MODEST** (diff-in-diff +0.306 mean at N=48, 6/6 positive, moat 0-FA every seed).
- **The limbic WRITE-side is load-bearing on the consolidated one brain at GPU scale.** CYCLE 536 wired the co-resident
  `OneBrainComposer` onto the merged bridge with the DA encoding-gain activated (`_consolidation_probe2_limbic.json`:
  neural DA 294.7 Hz → g 1.423 scales the stored magnitude EXACTLY; lesion pins 1.0; moat intact; gates 15/15).
- **The shared DA SOURCE is genuinely co-resident + spiking.** `dopamine` = the SIGNED RPE over the spiking SNc
  (`from_region_firing_signed`, `sim/neuromodulators.py:774-817`), registered on the merged bridge via
  `co_resident_limbic`/`co_resident_nav_critic`/`co_resident_td_cueshift`. Same `get_concentration("dopamine")` the nav
  cascade consumes — so the "one self" SOURCE half is done.

**THE GENUINE RESIDUAL (the next modulation target):** both Route-B findings, independently, name the SAME next lever —
**DA must modulate the READ/RETRIEVAL, not (only) the encoding.** Encoding-strength-as-noise-robustness is a uniform
per-fact magnitude scalar that the D=128 matched-filter read averages out → modest. The read model is far more sensitive
to **(i) salience-gated RECALL** (DA scales retrieval vigor / which fact wins the cue-scan) and **(ii) DA-gated
RECONSOLIDATION** (a surprising correction, gated by salience, is rewritten more strongly). These are the larger-effect
"one self" routes the closed encoding-gain arc explicitly points to.

**Recommended cheap-first de-risk (§5):** **DA-gated RECALL vigor** — the shared spiking dopamine biases *which stored
fact wins the cue-match scan / how vigorously it is retrieved* (a value/salience prior on retrieval, NOT on encoding),
validated by-FUNCTION (a task where the value/DA prior is load-bearing for the read), moat 0-FA as a HARD gate. Distant
second: DA-gated reconsolidation (richer, but couples to the prediction-error rewrite — more moving parts).

---

## 1. The diagnosis — what EXISTS vs what's MISSING (file:line)

### 1.1 The declarative neuromodulator subsystem — `sim/neuromodulators.py`

The hook abstraction is right for volume-transmission neuromodulation (catalog C.21: NM as a scalar field, not
per-synapse). Dataclasses: `NeuromodulatorConfig` (name/baseline/decay_tau_ms/concentration_min/max/targets/production_rules),
`ModulatorTarget` (target_type/scope/sensitivity), `ProductionRule` (rule_type/sensitivity/threshold/window_ms).

**Built-in target_types (the exact bridge-state hook points):**
| target_type | Effect on bridge state | compute method |
|---|---|---|
| `synaptic_gain` | multiplies `effective_synaptic_strength` | `compute_synaptic_gain_multiplier()` (~:301) |
| `plasticity_rate` | multiplies STDP amplitudes + `reward_learning_rate` | `compute_plasticity_rate_multiplier()` (~:323) |
| `excitability_drive` | adds pA to `total_input_current_pA` | `compute_excitability_drive_pA()` (~:546) |
| `plasticity_gate` | per-pathway weight-update gate [0,1] | `compute_plasticity_gate_values()` (~:512) |
| `plasticity_window_gate` | INVERSE gate (high conc BLOCKS plasticity) | `compute_plasticity_window_gate_multiplier()` (~:461) |
| `transmission_gate` | per-pathway synaptic CURRENT scale [0,1] | (in `sim/bridge.py`, not neuromodulators.py) |

**Built-in production_rules:** `manual`, `from_reward`, `from_error_persistence` (tonic NE-like), `from_surprise`
(phasic NE), `pause_on_reward` (ACh-style), `from_region_firing` / `from_region_firing_signed` (the **spiking SnC
dopamine** — signed, sub-tonic dips), `from_action_specific_reward` (per-action DA). **Scopes:** `all` / `trait:N` /
`group:NAME` / `gate:NAME` / `action:idx` / `plastic_only`.

**Integration:** `self.neuromodulator_manager.step(self)` is called ONLY inside `_run_one_simulation_step`
(`sim/bridge.py:7040`, with the gain/rate/gate multipliers applied on that same path). Allocated in
`apply_simulation_configuration_core()` (~`:746`).

**Diagnosis (a):** NO target_type modulates the COMPOSER's internal decision knobs (gain/threshold/vigor). The NM
subsystem reaches neuron dynamics + STDP + eligibility — not the composer's cue-scan / cleanup / abstention. A
whole-file search of `sim/neuromodulators.py` for `rf|phasor|resonate|composer|cleanup` returns ZERO. (The deployed #6
routes reach the composer at the AGENT/composer LAYER, reading `get_concentration("dopamine")` between ops, NOT via a
target_type — see §1.4 on why.)

### 1.2 The navigation dopamine / value machinery (the SOURCE)

- **Spiking SNc** — `g11_bg_runner.py` builds an `snc` pool (~:1365-1395, the spiking-dopamine `from_region_firing_signed`
  source); the dopamine concentration drives `plasticity_rate` in STDP (the shipped nav precedent). `rpe_dopamine=True`
  enables it.
- **TD value critic** — `sim/td_value_critic.py` (`run_pavlovian` ~:56) is a standalone numpy validator of the
  TD(λ) critic; the SPIKING critic role is realized by the SNc firing + the merged `co_resident_*` slice (CYCLE 536's
  `striosome_value` critic → shared `dopamine`). Catalog O.18-supplemental maps this exactly: critic = striosome V(s),
  actor = matrix H(s,a), δ = VTA/SNc DA — the project has the matrix side + (co-resident) the V side.
- **NOT a reusable shared builder:** the SNc/critic setup is inline in the runner + the merged-bridge `co_resident_*`
  kwargs; there is no standalone "limbic core" function importable by an arbitrary composer. (Not a blocker — the merged
  bridge IS where the composer co-resides, and that bridge has the source.)

### 1.3 The conversational composers — the knobs a modulator COULD turn (file:line)

**`RFPhasorComposer` (`research/runners/rf_phasor_composer.py`):**
- `encoding_gain_fn=None` (`:64`/`:93`) — **Route B (DONE).** Explicitly headed "Tier-2 #6 … DOPAMINE-GATED ENCODING
  STRENGTH (Lisman-Grace hippocampal-VTA loop; Kandel D.16)". Read at store time, scales the stored complex-phasor
  magnitude in `_store_substrate` (the differential comes from the RF magnitude FLOOR, `sim/bridge.py:5731`/`:5804`).
- de-risk read-damage knobs `_retrieve_lam`/`_retrieve_kick_mag`/`_retrieve_floor`/`_retrieve_noise` (`:100-112`) — the
  graceful-degradation knee tooling.
- `period=200` (`:67`) — resonate-window duration = a **recall-vigor / WM-persistence** knob (number of resonate steps
  the read integrates over; the candidate for salience-gated recall).
- `enable_spiking_cleanup` (`:62`) — spiking WTA vs numpy argmax cleanup (the **competition gain** surface).
- reconsolidation: `update_on_mismatch` + `_calibrate_pe_labile` (~`:466`) — the prediction-error labilization gate (the
  **DA-gated reconsolidation** target).

**`OneBrainComposer` (`research/runners/one_brain_composer.py`) — the PRODUCTION-default composer:**
- `confidence_gate=0.0` (`:114`, applied `:728-729`: `min(agent_margin, action_margin) < gate → return None`) — **the
  no-confab MOAT / abstention threshold** = a signal-detection criterion (catalog G.15). This is the knob Route A drives
  (read-side); a value-DEPENDENT criterion is the salience-gated-recall lever's natural form.
- `encoding_gain_fn=None` (`:116`, applied in `_write_block` `:342/:351-352`) — Route B (DONE), byte-identical when None.
- `period`, `enable_spiking_cleanup` (`:115`), `integrated_loop` (`:116`), `persistent_loop=True` (`:117`) — same family
  of vigor/gain knobs.
- reconsolidation: `update_on_mismatch` / `_calibrate_pe_labile` (~`:605`) — the DA-gated-reconsolidation target.

**`CoreSimComposer` (`research/runners/core_sim_composition.py`):** `enable_spiking_cleanup` (the NEF cleanup threshold)
— the original cleanup surface; superseded by the OneBrain/RF production path for this work.

### 1.4 `BrainConversationalAgent` + the dialogue planner

- `BrainConversationalAgent.__init__` (`research/runners/brain_conversational_agent.py:175-184`) takes a `composer` +
  `composer_kind` + capability flags + `speak_value_Q` — **NO dopamine/neuromodulator/value-critic parameter** (confirmed
  absent). The agent is composer-agnostic; it delegates composition. So the modulation is wired at the COMPOSER /
  merged-agent layer, not the generic agent constructor. (Note `speak_value_Q` is the value plumbing for the
  communicable-brain choose-to-speak appraisal — adjacent, see §6.)
- **Dialogue planning** (`elaborate`, `:717-731`) uses `SpikingSpreadingController`
  (`research/runners/content_selection_spiking.py`) over a learned Hebbian association graph. The bias toward which
  associate wins is STRUCTURAL (graph edge weights) + the initial spread strength — **no scalar dialogue-bias knob, no DA
  input today.** (Candidate target, lower-ranked — §4.)

### 1.5 Why the deployed routes reach the composer WITHOUT a per-step NM coupling (the audit-flagged bypass)

The composer's RF ops run on the production-fast path `rf_resonate_steps` (`sim/bridge.py:5749`) / `_rf_advance_one`
(`:5710`) / the megakernel (`:5814`), which **bypass `_run_one_simulation_step`** — so `manager.step()` (`:7040`) and the
NM target_type multipliers never reach the RF dynamics (the complex matvec `cp_rf_w_re/im @ z` is array-disjoint from
`cp_connections`). This is REAL but **designed-around, not an open hole:** Route A reads the DA concentration at the
agent layer BETWEEN ops; Route B bakes the gain into the stored weights AT ENCODING. Both are `sim/`-edit-free. The prior
arc EXPLICITLY rejected routing live DA into `_rf_lambda`/cleanup-gain inside the bypassed loop — it would (a) need a
`sim/` edit into the fast loop AND (b) modulate the unbind/cleanup of *already-stored* facts = a global gain that **risks
the moat** (`2026-06-19`/`2026-06-20` deep-research both reject it). **The recommended salience-gated-recall de-risk
respects this:** it operates at the composer/agent read layer (a value prior on the cue-scan), not inside the bypassed
RF loop.

---

## 2. The biology (catalog-FIRST, then Kandel, then literature)

Cluster C = "Dopamine & neuromodulation" (27 entries); Cluster G = "Working memory / PFC / cortical integration"; Cluster
O = "Emotion, reward, motivation". The load-bearing entries for "limbic state modulates language/composition/recall/WM":

| Entry | Mechanism (1-line) | Citation |
|---|---|---|
| **C.04** | DA (VTA→PFC mesocortical) modulates PFC working-memory SNR via D1; tonic=motivational state, phasic=RPE | Kandel 6e Ch 16 p371-376; Ch 43 |
| **C.05 / C.14** | NE (LC) "increases SNR by suppressing background firing AND enhancing selective response"; Aston-Jones inverted-U: phasic=focused, tonic=exploration | Kandel 6e Ch 40 p999-1006 |
| **C.19** | Basal-forebrain ACh enhances cortical pyramidal responsiveness (Hasselmo encode-vs-retrieve gating); selective-attention amplification | Kandel 6e Ch 40 p1003-1006 |
| **C.20** | Tonic (1-5 Hz pacemaker, maintains plasticity competence) vs phasic (burst = RPE/salience) DA — TWO axes the project fuses into one scalar | Kandel 6e Ch 40 p1001-1002; Schultz98 |
| **C.22** | Schultz RPE: VTA→NAc+PFC+**hippocampus**; phasic DA = δ, drives three-factor plasticity (the teaching signal) | Kandel 6e Ch 43; Schultz-Dayan-Montague 1997 |
| **D.16** | Place-field STABILITY requires attention + **D1/D5 dopamine** + late-LTP (inattentive → degrade in hours; attended/D1 → stable for days) | Kandel 6e Ch 54 p1366-1367 |
| **G.06 / G.08** | PFC WM persistent delay-period activity, **D1-modulated**; recurrent attractor / NMDA bistability | Kandel 6e Ch 34 p827; Ch 52 p1292 |
| **G.15** | Signal-detection decision rule — **a threshold/criterion on noisy evidence**; criterion = prior prob + cost of hit/miss/FA | Kandel 6e Ch 56 p1393-1395 |
| **G.16** | Drift-diffusion bounded accumulation; bound height = speed/accuracy; **"O (DA scaling of drift rate)"** (the catalog's own cross-cluster tag) | Kandel 6e Ch 56 p1399-1404 |
| **O.16** | NAc reward hub: VTA-DA + mPFC-goal + hippo-context + BLA-valence + LHb-aversion CONVERGE — the limbic→cortex integration node | Kandel 6e Ch 43 p1067-1068 |
| **O.18** | Reward-modulated stimulus-outcome learning, striatum↔HC; supplemental gives the actor-critic anatomical map (critic=striosome V, δ=SNc DA) | Kandel 6e Ch 52 p1303-1305; Sutton-Barto Ch 11 |
| **O.19** | vmPFC/OFC encode subjective value; **"Value modulates DRIFT RATE of accumulator — same drift-diffusion math as perceptual decisions"**; DA itself codes subjective value | Kandel 6e Ch 56 p1406-1409; Schultz16 |

**The four canonical computational models (literature, controller-verified):**
1. **D1 inverted-U on PFC-WM (Vijayraghavan/Arnsten 2007, Nat Neuro nn1846; Durstewitz-Seamans dual-state):** low D1R
   "enhances spatial tuning by **suppressing responses to nonpreferred directions**" (a SNR/gain sculpting of the
   attractor); too-high D1R erodes tuning. Dual-state: D1-dominated = high energy barrier = robust stabilization;
   D2-dominated = fast flexible switching. → maps to **cleanup competition gain** + **WM persistence**.
2. **Aston-Jones–Cohen adaptive-gain (Annu Rev Neurosci 2005):** LC-NE modulates GLOBAL CORTICAL GAIN; phasic =
   exploit/focus (sharpen the decision), tonic = explore/disengage. → maps to **cleanup gain** + **abstention threshold**.
3. **Niv 2007 tonic-DA & response vigor (Psychopharmacology):** tonic DA = average reward rate = opportunity cost →
   scales **response vigor** (how fast/vigorously to respond). → maps directly to **recall vigor** (the recommended de-risk).
4. **Lisman-Grace hippocampal-VTA loop (Neuron 2005):** hippocampal NOVELTY → (subiculum→NAc→VP→) VTA → DA back into
   hippocampus → enhances LTP → **DA gates the entry of information into long-term memory**, combined with salience+goal.
   → the biological grounding for Route B (encoding) AND for salience-gated RECALL (the loop reads goal/salience to
   decide what is retrieved/strengthened).

**Reframe via "how does real biology do this?" (the SURPASS move):** biology does NOT make a memory robust by a uniform
magnitude scalar (Route B's modest mechanism). It (i) DA-gates which memories enter/persist (Lisman-Grace, D.16), and
(ii) value/DA scales the DRIFT RATE / vigor of the retrieval decision (O.19, Niv 2007, G.16). The project's read model
(matched-filter cue-scan) is exactly a decision/accumulation over stored facts — so **the biologically-correct DA lever
is on the RETRIEVAL decision (vigor/criterion), not the stored magnitude.** This is why encoding-gain came back modest
and why salience-gated recall is the higher-leverage next target.

---

## 3. RANKED cheap-first options — WHAT the shared DA/limbic state should modulate in the composer

For each: the biology, the reusable machinery, the expected BEHAVIORAL signature, and the validate-by-FUNCTION anti-cheat
(the R4→R5 lesson — a task where modulated≠lesioned BEHAVIORALLY, not a task that ignores the signal). The moat is a HARD
gate in every option (modulating a threshold must NOT create false-accepts).

### Option 1 (RECOMMENDED) — DA-gated RECALL vigor / a value prior on the cue-scan (READ-side)
- **Biology:** Niv 2007 (tonic DA → response vigor); O.19/G.16 (value scales the accumulator drift rate); Lisman-Grace
  (salience/goal gates retrieval). Catalog G.15 (criterion) + G.16 ("DA scaling of drift rate").
- **What it modulates:** how vigorously / how far down the candidate list the cue-match scan reads, OR a value/salience
  PRIOR added to each stored fact's match score (the retrieval competition), driven by `get_concentration("dopamine")`.
  Concretely: scale `period` (resonate-window vigor) and/or add a value-weighted bias to the per-fact `_margin` score the
  scan ranks (`OneBrainComposer._margin`), with the moat criterion unchanged (a value prior can only RE-RANK among facts
  that already clear the familiarity gate — it cannot manufacture a match for an unstored cue).
- **Reusable machinery:** the merged DA source (`get_concentration("dopamine")` on the merged bridge), the existing
  `_margin`/cue-scan in `OneBrainComposer`, `period` knob, the Route-A `_da_confidence_gate`/`da_to_gate` helper pattern
  (`nav_conv_merged_bridge.py:1458-1474`) — reuse-by-import, NO `sim/` edit (read-layer, respects §1.5).
- **Behavioral signature:** a task with TWO competing stored facts where the value/DA prior is the tie-breaker — the
  high-value (high-DA-at-store-or-cue) fact is retrieved where, without the prior, the read is ambiguous/abstains. The
  read model is more sensitive to a retrieval-competition bias than to a uniform encoding magnitude (the explicit Route-B
  diagnosis: "they act where the read model is more sensitive than a uniform per-fact magnitude scalar").
- **Anti-cheat (validate-by-FUNCTION):** the task MUST require the value prior — design a value-CONFLICTED cue (two facts
  both clear familiarity; only the value/DA prior disambiguates), mirroring the nav R5 value-driven-CHOICE design (the
  6-seed GO that fixed the R4 value-irrelevant artifact). Controls: **DA-LESION** (hold DA at baseline → the prior
  vanishes → the read reverts to ambiguous/content-only); **EQUAL-value discriminator** (when the two facts have equal
  value, the prior is neutral — the R5 validate-by-function control); **PERMUTED** (swap which fact is high-value → the
  retrieval advantage follows the value, not the content); **MOAT (HARD)** an unstored cue abstains at every DA level.
- **`sim/` edit:** NONE (read-layer composer/agent wire-up). **Why ranked #1:** highest expected behavioral effect (acts
  where the read is sensitive), reuses the proven Route-A read-layer pattern, respects the bypass (§1.5), moat-safe by
  construction (re-ranks within the familiarity-gated set).

### Option 2 — DA-gated RECONSOLIDATION (a surprising correction, gated by salience, rewrites more strongly)
- **Biology:** Lisman-Grace (DA gates LTP for novel/salient events) + reconsolidation (a reactivated memory is labile;
  prediction-error opens the labilization window). The natural pairing of the existing PE-gated rewrite with limbic
  salience.
- **What it modulates:** the labilization threshold in `update_on_mismatch` / `_calibrate_pe_labile`
  (`one_brain_composer.py` ~`:605`; `rf_phasor_composer.py` ~`:466`) — high DA (surprise/salience) lowers the threshold so
  a corrective restatement rewrites the fact more strongly; low DA leaves the fact stable.
- **Reusable machinery:** the reconsolidation hooks ALREADY exist + are validated (the PE-gated rewrite is
  multi-seed-GO); the DA source exists. Reuse-by-import; the modulation is a runner-side scalar into the existing labile
  gate, NO `sim/` edit.
- **Behavioral signature:** a turn-2 correction issued under high DA (surprising) overwrites the stored fact (later query
  returns the corrected value); the SAME correction under low DA leaves more of the original (graded rewrite).
- **Anti-cheat:** **DA-LESION** (baseline DA for both corrections → equal rewrite strength); **PERMUTED** (high DA on the
  other correction → the stronger rewrite follows DA); **MOAT (HARD)** the rewrite never creates a fact for an unstored
  cue (reconsolidation acts only on a REACTIVATED block); a re-statement at low DA must still RESTABILIZE (no spurious
  erasure). Validate-by-FUNCTION: the task must make the correction's *strength* observable (graded query margin), not a
  binary already-handled case.
- **`sim/` edit:** NONE. **Why ranked #2:** biologically rich + larger-than-encoding effect, BUT couples to the PE rewrite
  (more moving parts than a read-side prior) and the "graded rewrite strength" metric is subtler to make load-bearing.

### Option 3 — DA / NE gain on the cleanup competition (binding/cleanup SNR)
- **Biology:** D1 inverted-U (low D1 sculpts the attractor by suppressing nonpreferred responses = a WTA sharpening);
  Aston-Jones NE global cortical gain; Durstewitz-Seamans D1-dominated robust-attractor state. Catalog C.04/C.05/G.06.
- **What it modulates:** the spiking-WTA cleanup competition sharpness (`enable_spiking_cleanup` path's
  inhibition/threshold) — high DA → sharper WTA → more decisive cleanup; an inverted-U so too-high erodes it.
- **Reusable machinery:** the spiking NEF/WTA cleanup; BUT the cleanup runs on the bypassed RF/cleanup op path (§1.5), so
  a LIVE per-step DA coupling would need a `sim/` edit into the fast loop — the very route rejected twice as a MOAT risk
  (a global gain on the cleanup of already-stored facts).
- **Anti-cheat:** would need the inverted-U behavioral signature (cleanup accuracy peaks at intermediate DA), DA-LESION,
  MOAT-HARD. Hard to make moat-safe because the cleanup sits between unbind and the answer.
- **`sim/` edit:** REQUIRED (into the bypassed loop) → **DEFERRED.** Ranked #3: the cleanest biological mapping
  (D1-attractor-gain) but the highest moat risk + the only one needing a protected edit. Revisit only if the read-side
  options plateau.

### Option 4 — DA-gated dialogue-planning content-selection bias
- **Biology:** mesocortical DA to PFC biases goal-relevant content selection (C.04 goal-gating; O.16 mPFC-goal +
  BLA-valence convergence into NAc → which content is salient).
- **What it modulates:** the spread strength / which associate wins in `SpikingSpreadingController` (`elaborate`) —
  higher DA → broader/value-weighted spread → richer or more value-relevant elaboration.
- **Reusable machinery:** the dialogue planner exists, but takes NO DA input and exposes NO scalar bias knob (§1.4) — more
  build than the read-side options.
- **Anti-cheat:** a task where the value prior changes WHICH associate is elaborated (DA-lesion → content-only spread;
  permuted → the elaboration follows the value); MOAT not directly at risk (elaboration is generative-from-graph, not a
  fact assertion) but on-topic-ness must hold.
- **`sim/` edit:** NONE, but needs a new bias surface on the planner. Ranked #4: real "one self" expression (mood biases
  what you choose to talk about) but lower-leverage for the core who/what loop + more plumbing.

**Summary ranking:** 1 (DA-gated recall vigor, read-side, NO `sim/` edit, highest leverage + moat-safe) > 2 (DA-gated
reconsolidation, NO `sim/` edit, rich but coupled) > 4 (dialogue-bias, NO `sim/` edit, lower leverage) > 3 (cleanup gain,
`sim/` edit + moat risk, DEFERRED).

---

## 4. The single recommended de-risk + anti-cheat controls

**Build:** a CPU-first → GPU two-rung **DA-gated RECALL-vigor** de-risk (Option 1). Reuse-by-import, NO `sim/` edit.

- **Rung 1 (CPU/numpy, isolate the read-layer plumbing — seconds/seed):** on a `MergedNavConvAgent` (or a standalone
  `OneBrainComposer` with the merged DA source mocked via `manager.set_concentration("dopamine", ·)`), add a value/DA
  prior to the cue-scan ranking: `score'_i = score_i + beta * DA * value_i` where `value_i` is the fact's stored
  value/salience and the prior re-ranks ONLY within the set that clears the familiarity/moat criterion. Confirm the prior
  re-ranks two value-conflicted facts at high DA and is neutral at baseline DA; confirm an unstored cue still abstains
  (moat).
- **Rung 2 (GPU, `SIM_BACKEND=cupy`, the real claim):** drive the shared `limbic_snc` to two operating points (the
  Route-A `_settle_snc` recipe: tonic 80 pA → DA≈0.50; salient 600 pA → DA≈0.84). Run the value-conflicted-cue task; the
  high-value fact is retrieved under high DA where the read is otherwise ambiguous/abstains, driven by the REAL spiking
  SNc on the merged bridge.

**The single load-bearing question:** *Does a value/salience prior carried by the shared spiking dopamine determine which
of two familiarity-cleared stored facts is retrieved — load-bearing for the read (DA-lesion kills it, equal-value is
neutral, permuted follows DA), with the no-confab moat intact?*

**Anti-cheat controls (validate-by-FUNCTION — the R4→R5 lesson):**
| Control | Rules out | Expected |
|---|---|---|
| **Value-CONFLICTED cue (the task)** | "the task doesn't need the value prior" (the R4 artifact) | two facts both clear familiarity; ONLY the value/DA prior disambiguates the retrieval (mirrors the nav R5 value-driven-CHOICE) |
| **DA-LESION (hold DA at baseline)** | "the effect is content/order, not DA" | the prior vanishes → the read reverts to content-only/ambiguous (the decisive control) |
| **EQUAL-value discriminator** | "the prior fires regardless / it's not value-specific" | when the two facts have equal value, the prior is neutral (the R5 validate-by-function discriminator that stays neutral) |
| **PERMUTED value** | "one fact is intrinsically more retrievable" | the retrieval advantage FOLLOWS the high-value fact, not the content |
| **No-confab MOAT (HARD)** | "the value prior manufactured a false-accept" | an unstored cue returns `None` at BOTH DA levels, EVERY seed. Structural: the prior re-ranks only WITHIN the familiarity-gated set → an unstored cue has nothing to re-rank. Any breach = NEGATIVE, not a tunable. |
| **Brain-based-only** | "the prior is a host scalar, not a spiking population's effect" | the prior magnitude = `get_concentration("dopamine")` = the SIGNED RPE from the spiking SNc (`from_region_firing_signed`); SAME status as the shipped nav `dopamine→plasticity_rate` precedent. Honest: the prior is APPLIED by a host multiply at the read layer (the fully-spiking ideal — the prior EMERGING in the cue-scan competition — is the follow-on, not claimed). |
| **Regression GREEN** | "the default path drifted" | prior OFF (beta=0 / no value) ⇒ byte-identical; `test_one_brain_composer_agent` + `test_nav_conv_merged_agent` 8/8 + `test_nav_conv_step2b_coresident` 7/7 pass VERBATIM. |

**GO / NEGATIVE bar (multi-seed, pre-registered):**
- **GO:** on the merged bridge with the REAL shared `dopamine`, the value-conflicted retrieval is disambiguated by the
  DA-carried value prior on **≥5/6 seeds** (42/43/44/100/101/102); **DA-LESION kills it** (every seed); **EQUAL-value is
  neutral** (every seed); the effect **follows the value** (permuted); **moat 0-FA at both DA levels** (every seed);
  **regression GREEN**. ⇒ the shared spiking dopamine demonstrably modulates a conversational RETRIEVAL decision — a
  larger, read-sensitive effect than the closed encoding-gain, both halves driven by the same limbic core.
- **NEGATIVE (an honest deliverable):** EITHER the value prior cannot disambiguate without breaching the moat on some
  seed (HARD gate → NEGATIVE, not a tunable), OR the read model is insensitive even to a retrieval-competition bias (then
  the honest finding is "DA-gated recall is also latent at the deployed read model; the one-self read-side levers are
  exhausted; the next is the persistent-integrated-loop's online retrieval or the dendritic read" — points at the
  emergent-feature follow-ons, NOT a failure). Either maps what DA can/can't do on the deployed RF read substrate.

---

## 5. Why NOT re-do the encoding route + why this is the SURPASS move

The comfortable verdict was "Tier-2 #6 reward→memory: closed as real-but-modest." Per the CLAUDE.md SURPASS sharpening,
that is a DISGUISED boundary (a "characterized limit" that quietly ends the #6 investigation). The four SURPASS moves:
1. **ISOLATE + QUANTIFY the residual:** the genuine residual is NOT "DA can't reach the composer" (it does — Routes A+B
   both built, the limbic write-side load-bearing at GPU scale). It is "the DA effect on the deployed read is MODEST
   because encoding-strength-as-noise-robustness is the WRONG lever for a matched-filter read." Quantified: diff-in-diff
   +0.306 (encoding) vs the read model's far-higher sensitivity to a retrieval bias (the Route-B findings' own
   diagnosis).
2. **REFRAME via real biology:** biology gates RETRIEVAL/persistence (Lisman-Grace, D.16) and scales the decision DRIFT
   RATE / vigor (O.19, Niv 2007, G.16) — NOT a uniform stored magnitude. We tested the wrong hypothesis (encoding
   magnitude); the read-side decision is the biologically-correct + behaviorally-sensitive lever.
3. **RANK cheap-first SURPASS mechanisms:** §3 — DA-gated recall vigor (read-side, NO `sim/` edit) is the cheapest path
   PAST the modest encoding result.
4. **Verdict:** #6 is SURPASSABLE and CHEAPLY (a read-layer value prior, reuse-by-import) — the encoding boundary was
   real but is the START, not the end.

---

## 6. Adjacent in-flight work to keep aligned (not scope, but to dedupe)

- **The communicable-brain "choose-to-speak" value/salience appraisal** (`_value_salience_appraisal_scoping.md`,
  AUTONOMOUS_STATE ~CYCLE 537+) uses the SAME shared `dopamine`/value axis — `speak_value_Q` is already plumbed into the
  agent. A DA-gated-recall de-risk should reuse, not duplicate, that value plumbing (the limbic core is the common
  source). #6's recall-vigor lever and the choose-to-speak appraisal are the read-side and output-side faces of the same
  "one self" value broadcast.
- **The persistent integrated spiking loop** (a separate Tier-2 item, Phase C designed) is what makes salience-gated
  recall happen ONLINE during a live turn — a GO here is the per-op slice; the loop is the continuous realization.

---

## 7. Discipline + key references

- **READ-ONLY:** no code edited, no experiments, no GPU. Branch `main` throughout. Strict git add (ONLY this doc; the
  pre-existing modified findings JSONs NOT staged). Pushed origin + gitea.
- **Trust-but-verify:** every load-bearing claim cited to source read this session — `sim/neuromodulators.py`
  (target_types/production_rules/`:774-817`/`:228`/`:231`), `sim/bridge.py` (`:5710/5731/5749/5804/5814/7040`),
  `research/runners/rf_phasor_composer.py` (`:62/64/93/100-112/466`), `one_brain_composer.py`
  (`:114/116/342/351-352/605/728-729`), `brain_conversational_agent.py` (`:175-184/717-731`),
  `nav_conv_merged_bridge.py` (`:1305/1458-1508`), `g11_bg_runner.py` (`~:1365-1395`), `sim/td_value_critic.py`
  (`~:56`); catalog `feature-catalog.md` entries C.04/C.05/C.14/C.19/C.20/C.21/C.22/D.16/G.06/G.08/G.15/G.16/O.16/O.18/O.19.
- **Prior #6 findings (the lineage this CONTINUES, not re-scopes):** `2026-06-19-tier2-limbic-to-composer-scoping.md`
  (original scope), `2026-06-20-tier2-limbic-to-composer-deep-research.md` (route built two ways),
  `2026-06-18-DA-salience-gate-production-wireup-GO.md` (Route A), `2026-06-19-dopamine-encoding-gain-derisk.md` +
  `2026-06-20-tier2-6-routeB-onebrain-wireup.md` (Route B build), `2026-06-22-tier2-limbic-composer-next-step.md`
  (the deploy-smoke plan), `2026-06-22-tier2-routeB-deployment-smoke-LATENT.md` +
  `2026-06-22-tier2-routeB-content-matched-6seed-MODEST.md` (the CLOSED encoding result + the explicit next-lever
  pointer), AUTONOMOUS_STATE CYCLE 536 (limbic write-side load-bearing at GPU scale) + CYCLE 710 (the dispatch of this
  doc).

## 8. One-line recommendation

**#6's encoding/read-precision routes are BUILT (Route A GO, Route B closed-real-but-modest, limbic write-side
load-bearing at GPU scale); the genuine next "one self" lever — pointed to by both Route-B findings + the biology
(Niv-vigor / O.19-drift-rate / Lisman-Grace-retrieval) — is DA-gated RECALL VIGOR: a value/salience prior carried by the
shared spiking dopamine that biases WHICH stored fact wins the cue-match scan, validated on a value-CONFLICTED retrieval
task (DA-lesion kills it, equal-value neutral, permuted follows DA, moat 0-FA HARD), reuse-by-import, NO `sim/` edit,
≥5/6 seeds.**
