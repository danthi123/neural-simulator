# Spiking VALUE / SALIENCE APPRAISAL — the brain's decision to SPEAK — SCOPING (read-only, 2026-06-24)

**Role:** read-only deep-research + scoping subagent. NO edits, NO runs, NO webapp. No `tail`/`grep -f`/`while-sleep` watchers.
**Trigger / research-gate (d) new-mechanism-class:** the natural next mechanism after the communicable-brain Probe 1 GO
(`research/findings/2026-06-24-communicable-brain-probe1-GO.md`). The frontier scoping flagged this as **"the ONE genuine
new mechanism" (Option 5)** — a spiking value/salience read-out that lets the brain CHOOSE to speak: rank candidate
propositions by **salience/value** (not just raw plausibility), and decide **speak-vs-stay-silent** by a **value threshold**,
grounded in the brain's reward/value/dopamine system (the limbic/DA core that option-A activates).
**North-star:** owner #1 — the whole pipeline as ONE persistent interacting spiking loop
(`project_one_brain_integrated_pipeline_and_cleanup`, `feedback_move_everything_to_shared_spiking_substrate`); the
reward/value/dopamine limbic core is the highest-leverage shared system. **Hard constraints honored throughout:**
BIOLOGICAL (neurons/synapses/their communication); SINGLE BRAIN, SINGLE SUBSTRATE; the brain does the cognition (LLM
fluency-only); the moat **RELAXES** (speak-while-flagging) — it is NOT removed.

> **HEADLINE (load-bearing for the controller).** Probe 1 emits a flagged hypothesis only where one proposition is
> graph-plausible, taking the **single highest-plausibility** topic-relevant proposal and otherwise abstaining-from-opinion
> (~70% of topics, 6–9/30 emissions). It has **no APPRAISAL** — no spiking read-out of "is this worth saying / which of my
> candidate thoughts is most salient," and no value-grounded speak threshold. Reading the actual code, the gap is **narrow,
> and it is exactly where the option-A limbic core pays off:** every input the appraisal would CONSUME (a graded value/DA
> scalar from the spiking `snc`/`striosome_value` critic; a graded plausibility + a lesionable familiarity signal; a
> candidate SET from the b2 proposer) and everything it would DRIVE INTO (a spiking accumulator→commit-burst, or a
> firing-coupled "speak" transmission gate) **already exists and is GO.** The **genuine new part is a single spiking
> appraisal pool** whose firing encodes "how worth-saying is this candidate," driven by {value, plausibility, familiarity},
> feeding a sel→commit speak-vs-silent threshold — and doing it brain-based (a neural pool's firing, NOT a host
> `if score > threshold`). The catalog confirms the **appraisal-network layer is genuinely un-catalogued** (a real NEW
> mechanism), while the value half and the commit-at-threshold half are well-grounded reusable primitives.

---

## 1. DIAGNOSIS — what a spiking value/salience appraisal IS, what we have, what's genuinely new

### 1.1 What the mechanism IS, biologically

The "decision to speak" is a **value-gated, evidence-accumulating commit** sitting on top of three cooperating biological systems:

1. **The salience network (appraisal).** Anchored in the **anterior insula (AI)** + **dorsal anterior cingulate cortex
   (dACC)**, the salience network monitors internal + external candidates and **assigns salience (importance)**, prioritizing
   what reaches the executive/output networks and switching between the internally-oriented Default-Mode Network (DMN) and
   the externally-oriented Frontoparietal/Central-Executive Network (the Menon/Seeley **triple-network** model; the right AI is
   the switching bottleneck — Sridharan/Menon 2008; phase-transfer-entropy iEEG replication 2024-25 confirms directed AI→DMN/FPN
   flow during tasks). **This is the literal "which of my candidate thoughts is salient enough to act on" layer.**

2. **The DA value system (worth).** **OFC/vmPFC** encode a scalar **subjective value** across options; the **striatum**
   biologically realizes **action-value Q(s,a)**; **phasic dopamine** is the reward-prediction error (RPE), and **tonic
   dopamine** sets **response vigor / willingness to respond** as the average-reward opportunity cost (Niv 2007). Critically,
   **value modulates the DRIFT RATE** of the decision accumulator (same drift-diffusion math as a perceptual decision —
   Kandel 6e Ch 56, O.19). And per **Berridge** incentive-salience, mesolimbic DA is the **"wanting"** signal (motivational
   magnitude attributed to a representation), distinct from "liking" — the cleanest biological framing for "how much do I
   *want* to say this." DA Component-1 (the 60–90 ms unselective **salience/detection pulse**, Schultz 2016) is the
   intensity/novelty driver; Component-2 is the utility-RPE.

3. **The commit / speak-initiation (threshold).** **Drift-diffusion / bounded evidence accumulation** (Kandel 6e Ch 56;
   LIP/parietal accumulator; G.15–G.18) integrates evidence to a **threshold** = the **decision/commit**. For SPEECH
   specifically, the **pre-SMA / SMA** ignites the speech motor "go" via the frontal aslant tract, and a 2025 paper
   (*Evidence accumulation in the pre-supplementary motor area and insula drives confidence and changes of mind*, PMC12311133)
   shows **evidence accumulation + confidence overlap anatomically in pre-SMA + insula** — i.e. the same accumulator that
   commits the decision also produces the confidence and gates the speak-vs-withhold. The "urge to speak" is the point at
   which the accumulator commits.

So: **appraise candidates → value/salience modulates the drift → accumulate → commit-at-threshold = speak (the most salient
candidate), else stay silent.** The brain volunteers a view when accumulated worth crosses the bound.

### 1.2 What the project ALREADY has (reusable) — all GO

| System | Where (project) | Spiking? | Reuse for the appraisal |
|---|---|---|---|
| **Spiking DA / limbic core** — `snc` SNc reward-burst + `striosome_value` MSN-D1 value critic (GABA_B/GIRK); the `dopamine` neuromodulator via `from_region_firing_signed` over `[snc]` (signed → DA can dip BELOW baseline = negative RPE) | `research/runners/nav_conv_merged_bridge.py` (`co_resident_nav_critic` default-ON, `:1022-1035`); minimal `co_resident_limbic` organ `:811-839`; `sim/neuromodulators.py` (`NeuromodulatorManager.get_concentration`, `from_region_firing_signed` `:774`, `_default_dopamine_config`) | **YES** — SNc *firing* is the RPE; validated Schultz battery 6/6 (`2026-06-18-limbic-core-rpe-battery-GO.md`) | the **VALUE axis**: `nm.get_concentration("dopamine")` and/or the `striosome_value` firing = "how rewarding/salient is this to say." The signed-firing rule is the ready-made way to turn an appraisal pool's firing into a graded scalar. |
| **`_da_encoding_gain` / `_da_confidence_gate`** — DA→per-fact gain (Lisman-Grace D.16) and DA→cue-role confidence sharpening (Vijayraghavan/Arnsten inverted-U), both DA-correct + de-risked GO | `nav_conv_merged_bridge.py:1828` / `:1808`; read-side gate default-ON (`enable_da_salience_gate` `:1471`, 6/6 GO) | YES (read DA scalar; act on conv cortex) | the **template** for "the shared DA reaches the conversational cortex" — the appraisal extends the read-side from a *confidence sharpener* to a *speak-decision driver*. |
| **b2 generative-replay PROPOSER** — invents the candidate propositions + scores each | `research/runners/_genfrontier_b2_generative_replay_derisk.py` (`GenerativeReplayProposer`; `propose(n) -> {accepted:[triples], plausible_fraction_of_novel,...}`; `_plausible`/`_strong_plausible`/`_contradicts`/`_weight_partner`) | host bookkeeping; the PPMI graph + RF store are the brain | the **OUTPUT the appraisal ranks**: `propose()` already returns a *set* of accepted plausible triples; the appraisal ranks that set + decides emit. |
| **Plausibility (PPMI) + graded-confidence read** — the learned relatedness signal | `build_plausibility(corpus, vocab)` (PPMI, `_genfrontier_b2...py:104`); `plausibility_score(P,row,a,ac,p)` + `hedge_for(...)` (`_communicable_brain_probe1...py:126`) | host (learned over real TinyStories) | the **CONTENT-SUPPORT axis** — already read out as graded confidence (Probe 1 spearman 1.00). |
| **Bogacz-Brown familiarity gate** — learned, **lesionable** novelty/"do I know enough" signal | `AntiHebbianFamiliarity` (`cortex_learned_cleanup_derisk.py:126`: `imprint`/`novelty = ‖x‖²−xᵀWx`/`lesion`); `RelationalFamiliarityGate` (`familiarity_gate_v320_validation.py:74`, V=320 GO, 0 moat-breaches) | rate-form of a spiking anti-Hebbian net; validated alongside the host moat | the **CONFIDENCE / metacognitive axis** + a built-in **lesion anti-cheat** (W:=0 collapses separation). |
| **Spiking accumulator → commit-burst → omnipause (OPN)** — the WTA decision circuit; the **library default** since 2026-06-19 | `g11_bg_runner.py` (`sel_{N,E,S,W}` Wang-2002 NMDA integrator `n_sel_per_action=40`, `sel_recurrent_weight=0.3` Usher-McClelland leak `:2286`; `commit_{N,E,S,W}` Lo-Wang all-or-none burst held below threshold by tonic `commit_OPN` `:2316`; `urgency_max_pA=180.0` Cisek collapsing bound); via merged `run_moving_goal_episode(readout_source="spiking_wta", ...)` | **YES, fully spiking** (read is a body-side count over the window) | the **literal template for the speak-vs-silent THRESHOLD**: a single (or speak/silent opponent-pair) sel→commit→OPN micro-circuit whose evidence input = the appraisal; the brain **commits to speaking** when accumulated salience crosses the OPN-held bound (== how it commits a move). |
| **`command_route` firing-gated decision** — a parser ensemble's FIRING opens a synaptic transmission gate (the WHEN supplied by spikes) | `spoken_instruction_nav.py` (`couple_command_gate` `:395`, `couple_gate_to_indices` EMA-over-threshold; 6-seed GO `2026-06-10-spoken-instruction-nav-GO.md`) | YES (spiking-firing-driven gate state) | the **alternative read-out**: a "speak" transmission gate opened by the appraisal pool's FIRING (no host threshold) — gates the render/emit path on. |
| **td_value_critic** (host reference) + `td_critic_gate._da_modulator_from_delta` | `sim/td_value_critic.py` (TD(λ), `delta=r+γV(s')−V(s)`); `research/runners/td_critic_gate.py` | host numpy (reference/oracle) | the **value ORACLE/teacher** to validate the spiking appraisal against; shows δ→DA wiring. NOT the on-substrate signal (the on-bridge TD-read is contaminated by place-code magnitude — `2026-06-22-shortcut5b-td-read-derisk.md`). |

### 1.3 The genuinely NEW part (small, but real)

The catalog review is decisive on what is novel:
- **The salience-network appraisal layer is ENTIRELY ABSENT** from the catalog (no "salience network" entry, no anterior-insula
  entry, no dACC entry, no von Economo entry; DMN exists only inside G.09 imagination, FPN only inside G.20 global-workspace).
  A salience read-out that **ranks candidates and gates the output network** is a NEW mechanism.
- **Confidence / metacognition is ABSENT** ("confidence" appears once, as a byproduct of the G.16 drift-diffusion bound; no
  metacognition / feeling-of-knowing / familiarity entry — the project's own Bogacz-Brown gate is not catalogued).
- **C.23 "Heterogeneous DA Subpopulations — reward, aversion, SALIENCE VTA cells" is flagged `missing`** (Kandel 6e Ch 43
  pp 1068-1069), as is **C.32 "Two-component DA — detection (salience, Component 1) + utility-RPE"** (`partial`), and **C.27
  wanting-vs-liking (incentive salience)** (`missing`). These are the closest catalogued primitives to "spiking salience."
- **C.30 actor-critic / a separable learned V(s) critic** is `partial` and repeatedly cited as the single highest-leverage
  missing piece (the project's `striosome_value` MSN-D1 critic is the on-substrate realization the appraisal reads).
- **No catalog entry frames the decision/urge to SPEAK** — it is a COMPOSITION of catalogued primitives (G.07 internally-generated
  initiation + G.15 yes/no criterion + G.16/G.17 accumulator + O.19/C.32 value-salience drive + G.20 workspace gate).

**⇒ The genuine new mechanism = a spiking APPRAISAL pool** (a salience-network analogue, AI/dACC-flavored) that maps the
already-computed {value (DA), plausibility, familiarity} signals onto a graded "worth-saying" firing rate, and feeds a spiking
sel→commit "speak gate" (the existing accumulator template). Everything it consumes and drives into is already GO; the new
science is **the appraisal read-out itself + the value-grounded speak threshold**, realized in spikes.

---

## 2. RANKED biology-grounded options (cheap-first) for the spiking appraisal read-out

> Every option is single-substrate (co-resident on the merged "one brain" / a slice; or the standalone Probe-1 brain for the
> cheapest CPU de-risk), reuses the limbic DA core + the b2 proposer, and keeps the moat RELAXED-NOT-REMOVED (a "speak" gate
> only adds an emission channel; the known-fact channel stays hard-gated; flagged-hypothesis-only). Ordered cheapest/most-
> reusing → deepest.

### Option A [CHEAPEST — the recommended de-risk; "salience drift" on the existing commit circuit]
**Mechanism.** A small **spiking appraisal/salience pool** (one per candidate proposition, or a graded population coding the
*best* candidate's worth) whose drive = a weighted sum of the brain's own signals, injected as **drift** into the **existing
spiking sel→commit→OPN accumulator** (one "speak" accumulator vs. an opponent "stay-silent" pool). The **value/salience axis**
is the shared **`dopamine`** concentration (and/or the `striosome_value` firing) read off the merged bridge — DA scales the
drift rate (the catalogued O.19/C.32 "value modulates drift" + Niv-2007 tonic-DA vigor: more DA → speaks more readily); the
**content-support axis** is the b2 `plausibility_score`; the **confidence axis** is the `AntiHebbianFamiliarity` novelty
(low-novelty/high-familiarity of the *topic neighborhood* → more drift). The **speak commit** fires when the accumulator
crosses the OPN-held bound (Lo-Wang); else the silence pool wins → no emission (the honest "no view"). The hedge band is read
off the *committed* candidate's plausibility (Probe-1's `hedge_for`), now CALIBRATED against the accumulator's balance-of-
evidence (pre-SMA-confidence, PMC12311133).
**Why cheapest:** reuses the GO accumulator verbatim (the merged `sel/commit/OPN` template), the GO DA scalar, the GO b2 proposer,
the GO familiarity gate. The ONLY new pool is the appraisal-drive pool. NO `sim/` edit expected (the accumulator + masked RF +
neuromodulator subsystem are shipped; the drive is `cp_external_input_current` on the appraisal pool).
**Biology:** salience network assigns salience → modulates the FPN/output accumulator (Menon triple-network); value modulates
drift (O.19); DA = wanting/vigor (Berridge; Niv 2007); commit = drift-diffusion bound (G.16/Lo-Wang).
**Risk:** the three axes must be *commensurable* (a scaling decision); mitigated by the cheap CPU pre-pass calibration Probe-1
already does for the hedge bands.

### Option B [firing-coupled "SPEAK gate" — the `command_route` pattern]
**Mechanism.** Instead of feeding an accumulator, the appraisal pool's **FIRING** directly opens a synaptic **`speak_route`
transmission gate** (reuse `couple_gate_to_indices`, EMA-over-threshold): when the appraisal pool fires above threshold (its
drive = the same {DA, plausibility, familiarity} sum), the gate opens and the render/emit path runs; below threshold the gate
stays closed → silence. The threshold IS the speak decision; "how readily the brain volunteers a view" is the gate threshold +
the appraisal-pool gain.
**Why next:** also reuses-by-import (the `command_route` firing-gated-decision is GO 6-seed); slightly less biologically rich
than a true accumulator (no RT/confidence dynamics) but mechanically the simplest spiking gate. Good as a **second read-out**
to cross-check Option A (do they agree on which topics get spoken?).
**Biology:** SMA/pre-SMA "go" ignition of speech (frontal aslant tract); a threshold-on-firing speak initiator.
**Risk:** a hard gate has no graded confidence; pair with the plausibility-derived hedge for the flag.

### Option C [an explicit spiking salience-network triad analogue]
**Mechanism.** A small **anterior-insula/dACC-flavored salience module**: an AI-analogue pool that receives the candidate set's
{value, plausibility, familiarity}, computes a **biased competition** (winner = the most salient candidate via lateral inhibition
— Rutishauser α>1 WTA), and a dACC-analogue that gates between a "DMN" (internally-generated proposer / abstain-from-opinion)
state and an "output" (speak) state — the triple-network switch. The winner drives the speak commit (Option A/B).
**Why deeper:** this is the closest to the actual salience-network biology and the most general (it natively does
**candidate-ranking** as biased competition, not just a scalar threshold), but it is more new wiring (a second WTA + a switch
pool). Best **after** Option A proves the value-grounded speak-decision works, as the "do it properly as a salience network" upgrade.
**Biology:** Menon/Seeley triple-network; AI biased-competition salience assignment; dACC DMN↔FPN switch.
**Risk:** the switch dynamics are the genuinely-new part; defer until the cheaper read-outs are GO.

### Option D [DEEP / DEFER — DA threads the appraisal-pool DYNAMICS; tonic-DA vigor as an explicit set-point]
**Mechanism.** Make the shared DA modulate the appraisal pool's **excitability / gain** directly (not just its drift) — tonic
DA as the global "willingness to volunteer" set-point (Niv-2007 vigor), phasic DA (the SNc burst on a salient candidate) as the
per-candidate boost (Component-1 detection pulse, Schultz 2016). This is the I-7-c-flavored deep `sim/` edit (modulate the pool's
f-I or the RF λ/ω). **DEFER** — the read-side DA scalar gate (Option A's drift) + the encoding-gain magnitude are the cheap, GO,
moat-safe pieces and are sufficient for "the limbic core grounds the speak decision." Flag for a later purity arc.

---

## 3. RECOMMENDED cheap-first de-risk (smallest falsification)

**Probe: "Does a value/salience read-out make the brain speak MORE where it has genuine support AND stay silent where it
doesn't, beating the plausibility-only baseline — calibrated, moat-preserved, and lesion-confirmed-as-the-BRAIN's-value-system?"**
(Option A, CPU-first on the Probe-1 standalone brain; reuse the `_communicable_brain_probe1_whatdoyouthink` harness verbatim and
extend it.)

- **Build (reuse-by-import, NO `sim/` edit):** the Probe-1 brain (PPMI cortex + `RFPhasorComposer` store + `GenerativeReplayProposer`
  + `BrainConversationalAgent` parse + familiarity gate). Add ONE appraisal read-out: for each topic, the proposer returns its
  **candidate set** (Probe-1 currently keeps only the best — expose all accepted topic-relevant triples); compute each candidate's
  worth = a weighted combination of {DA-value, plausibility, familiarity}; rank by worth; **decide emit** via a spiking
  sel→commit "speak" accumulator (Option A) whose drift = the top candidate's worth (CPU: a small Izhikevich speak/silence pair,
  or — for the CPU-only smoke — the on-substrate merged `sel/commit` micro-circuit driven by the worth, exercised GPU-side as the
  GO bar). The **value axis** on the standalone brain is a stand-in for the merged `dopamine` (the GPU follow-on uses the real
  shared SNc/`striosome_value`).
- **GO bars (multi-seed ≥3; controller runs 6-seed if GO):**
  1. **SPEAKS MORE WITH SUPPORT, BEATS PLAUSIBILITY-ONLY.** The appraisal emits on **more** topics than the plausibility-only
     baseline (Probe-1's "single highest-plausibility, else abstain") **where genuine graph support exists**, WITHOUT emitting on
     topics that lack support (the abstain-from-opinion set the baseline correctly skips). Quantify: emissions rise (e.g. 7.7/30 →
     higher) AND every new emission is GROUNDED (the shuffled-PPMI-graph control collapses its worth ≥3×; the appraisal-driven
     emission set's grounded-advantage ≥ the baseline's). The appraisal *adds* emissions on supported topics, not on noise.
  2. **CALIBRATED.** The stated confidence/hedge tracks the committed candidate's worth (spearman ≥ 0.5; the high-worth bin
     carries more of the INDEPENDENT b2 `_strong_plausible` property — the Probe-1 non-tautological calibration check, extended
     to the worth ranking).
  3. **MOAT RELAXED-NOT-REMOVED (HARD).** 0 known-fact-channel leaks (a who/what query on every emitted un-stored proposition
     still ABSTAINS; every emission flagged). The speak gate adds an emission channel; the known-fact channel is byte-unchanged.
- **The decisive anti-cheat (LESION the value system):** **pin the value axis (DA / the appraisal-value input) to baseline →
  the speak-decision COLLAPSES to the plausibility-only baseline (or below)** — i.e. with the brain's value system lesioned, the
  appraisal adds nothing and the brain reverts to flat plausibility-ranking. This proves the *extra* emissions are driven by the
  BRAIN's VALUE SYSTEM, not a host re-ranking. (Mirrors the Probe-1 lesion 46/46 + the familiarity-gate `lesion()` precedent.)
- **Cost:** CPU, minutes (the standalone-brain pre-pass). The GPU follow-on wires the SAME appraisal onto the merged "one brain"
  (the real shared `dopamine` from the spiking SNc; the real `sel/commit/OPN` accumulator) and re-runs the bars + the live-nav
  co-residence gate (byte-identity of the conversational reads, nav score Δ=0).

**If NEGATIVE** (the value axis doesn't add grounded emissions / collapses a control / breaks calibration), report it PRECISELY —
that localizes whether the gap is the value SIGNAL (the on-substrate value-read contamination, `2026-06-22-shortcut5b`), the
commensurability of the three axes, or the threshold. An honest negative IS the deliverable under the BRAIN-BASED-ONLY standard.

---

## 4. Anti-cheats (every probe inherits these)

1. **The appraisal is the BRAIN's value system, not a host threshold — LESION it.** Pin DA / the value input to baseline → the
   speak-decision collapses to the plausibility-only baseline. The *extra*, value-driven emissions MUST require the brain's
   value system. (Plus the familiarity-gate `lesion()` and the Probe-1 proposal-lesion both remain available.) **This is the
   load-bearing anti-cheat** — it is what makes the appraisal the brain's, not a Python `if score > threshold` (the
   `feedback_brain_based_only_standard` bar). The GPU version lesions the shared SNc → DA pinned baseline.
2. **GROUNDEDNESS load-bearing.** The shuffled-PPMI-graph control must collapse the appraisal-driven emission set's worth ≥3×
   (the Probe-1 grounded control, applied to the emission set the appraisal selects) — the LEARNED structure drives which
   topics get spoken, not a template/string artifact.
3. **MOAT RELAXED-NOT-REMOVED (HARD).** 0 known-fact-channel leaks at every step; every emission flagged as a hypothesis; the
   speak gate is an additive emission channel; the known-fact `what_does`/`is_it_true` channel stays hard-gated + byte-identical.
   (Per `feedback_moat_not_hard_lossy_memory_ok`: kept-where-free; here it is free — the appraisal only decides *whether/which*
   to volunteer a flagged view.)
4. **CALIBRATION (non-tautological).** The high-worth bin must carry more of an INDEPENDENT graph-support property (b2
   `_strong_plausible`) the worth-score does not directly read — a flat curve = the appraisal is decorative.
5. **SINGLE-SUBSTRATE / PROVENANCE.** The PPMI cortex + RF composer + the spiking SNc/`striosome_value` critic + the accumulator
   are the brain; the host does only recombination bookkeeping + routes which assembly fired; the fluency faculty is surface form
   only. The limbic→appraisal coupling is a **scalar read of the spike-derived `dopamine` concentration** (legitimate
   neuromodulatory-broadcast boundary — a global concentration the substrate produces via SNc firing), NOT a host computation of
   value written into a region. GPU version: confirm no new `cp_external_input_current[<appraisal>] = f(to_host(<value>))` host
   copy; the value enters via the neuromodulator concentration only.
6. **BYTE-IDENTITY of the existing pipeline (GPU).** Adding the appraisal must not regress the conversational reads (byte-identical
   at REST / DA==baseline) or the nav score (Δ=0; the composer's complex synapses are array-disjoint from `cp_connections`).

---

## 5. VERDICT + how it composes with option-A + owner-steer flags

**Verdict: a GENUINE NEW MECHANISM with a small footprint — the appraisal READ-OUT + the value-grounded speak threshold — built
almost entirely by COMPOSING already-GO pieces.** Unlike the option-A consolidation (which was "compose-already-proven-pieces,
one contained port"), this introduces a *new cognitive capability* (the brain deciding what is worth saying) that the catalog
confirms is **un-catalogued** (the salience-network + confidence layers are absent; the closest primitives — C.23/C.32 DA-salience,
C.27 wanting, C.30 actor-critic — are `missing`/`partial`). BUT the new part is **narrow**: a single spiking appraisal pool whose
drive = {DA-value, plausibility, familiarity} (all GO signals), feeding the EXISTING spiking sel→commit→OPN accumulator (GO) or a
firing-coupled speak gate (GO), with the b2 proposer's candidate set (GO) as input and the moat relaxed-not-removed (the Probe-1
contract). The smallest falsification is CPU, reuse-by-import, NO `sim/` edit — and it has a sharp lesion anti-cheat (pin the
value system → the speak-decision collapses to plausibility-only) that makes it the BRAIN's value system, not a host threshold.

**How it composes with option-A's limbic core (it depends on it).** The appraisal's **value axis IS the option-A shared
`dopamine`** (the spiking SNc reward-burst + `striosome_value` MSN-D1 critic, default-ON on the merged bridge). The CPU Probe-1
de-risk uses a value stand-in; the production appraisal reads the real merged-bridge `dopamine` — the SAME scalar the option-A
read-side `_da_confidence_gate` already reads. So: **option-A brings the limbic core onto the shared substrate (read-side
confidence sharpening + write-side encoding gain); this appraisal extends the read-side from "sharpen the confidence gate" to
"decide whether/which to SPEAK," grounded in that same DA.** Cleanest sequencing: land option-A (co-resident `OneBrainComposer` +
the limbic write-side) first, then the CPU appraisal de-risk on the standalone brain, then wire the appraisal onto the merged
one-brain so the speak-decision is value-grounded in the live spiking DA.

**Owner-steer flags (genuine forks):**
1. **THE VALUES CALL (load-bearing, owner's to set) — how readily should the brain volunteer a view?** The speak-vs-silent
   threshold / appraisal-pool gain is a *personality/values* parameter, not a science bar. Low threshold → the brain speaks
   often (assertive, more flagged hypotheses, more risk of low-support emissions); high threshold → it mostly abstains (cautious,
   stays near the current ~30% but better-ranked). Recommend exposing it as a single calibratable "talkativeness" gain with a
   conservative default (speaks somewhat more than Probe-1's 7.7/30 but only on grounded topics), and letting the owner tune it —
   this is exactly the "how readily the brain volunteers a view" call the prompt flags.
2. **The value axis on the STANDALONE Probe-1 brain (CPU de-risk).** The standalone brain has no spiking SNc; the cheap de-risk
   needs a value stand-in (e.g. the b2 `_strong_plausible`/coherence as a proxy worth, or a scalar mocking the merged `dopamine`).
   Recommend a transparent stand-in for the CPU smoke, with the GPU follow-on using the REAL shared `dopamine` (so the lesion
   anti-cheat is genuine — pinning the real SNc). Flag: the CPU value stand-in must NOT be the same quantity as the plausibility
   axis (else the "value adds beyond plausibility" test is circular) — use the DA proxy as a DISTINCT signal (e.g. a
   reward/novelty-tagged value the brain learned, separate from PPMI relatedness).
3. **Read-out: accumulator (Option A) vs firing-gate (Option B) vs salience-triad (Option C).** Recommend Option A first (richest
   biology + literal reuse of the GO accumulator + native confidence), Option B as a cross-check, Option C as the "do it properly
   as a salience network" upgrade after A is GO.
4. **Candidate-ranking vs scalar-threshold.** Probe-1 takes the single best candidate; the appraisal can either (a) keep
   best-candidate + add only the speak threshold (cheapest), or (b) genuinely RANK the candidate set by worth and emit the
   top-1 (or top-k) — the "which of my candidate thoughts is most salient" framing. Recommend (b) (it is the stated goal and the
   b2 proposer already returns the set); (a) is the fallback if (b)'s ranking doesn't beat best-by-plausibility.
5. **DEEP/DEFER — Option D (DA threads the appraisal-pool dynamics / tonic-DA vigor set-point).** The only deep `sim/`-edit
   follow-on; deferred behind the cheap GO drift/gate read-outs. Owner call on priority vs. the Tier-3 artificial-life horizon.

---

### Key references
- **Probe 1 context:** `research/findings/2026-06-24-communicable-brain-probe1-GO.md` (the appraisal flagged as the next mechanism);
  `research/runners/_communicable_brain_probe1_whatdoyouthink.py` (`WhatDoYouThinkTurn`, `plausibility_score`, `hedge_for`).
- **b2 proposer:** `research/runners/_genfrontier_b2_generative_replay_derisk.py` (`GenerativeReplayProposer`, `build_plausibility`);
  `2026-06-23-genfrontier-b2-generative-replay-derisk.md` (GO 6-seed).
- **Limbic core (option-A):** `research/findings/raw/_consolidation_onebrain_limbic_scoping.md`;
  `research/runners/nav_conv_merged_bridge.py` (`co_resident_nav_critic`, `_da_encoding_gain` `:1828`, `_da_confidence_gate`
  `:1808`, `enable_da_salience_gate` `:1471`); `sim/neuromodulators.py` (`from_region_firing_signed` `:774`); `snc_pavlovian_probe.py`;
  `2026-06-18-limbic-core-rpe-battery-GO.md`; `2026-06-19-dopamine-encoding-gain-derisk.md`.
- **Accumulator / commit:** `research/runners/g11_bg_runner.py` (`sel/commit/commit_OPN`, `urgency`);
  `2026-06-19-spiking-decision-default-on-GO.md`; `research/runners/spoken_instruction_nav.py` (`couple_command_gate`);
  `2026-06-10-spoken-instruction-nav-GO.md`.
- **Value / familiarity (reference):** `sim/td_value_critic.py`; `research/runners/td_critic_gate.py`;
  `2026-06-22-shortcut5b-td-read-derisk.md` (on-bridge value-read contamination); `cortex_learned_cleanup_derisk.py`
  (`AntiHebbianFamiliarity`); `familiarity_gate_v320_validation.py`; `2026-06-11-familiarity-gate-v320-GO.md`.
- **Catalog:** `sim-catalog/references/feature-catalog.md` — C.23 (DA-salience, `missing`, Ch 43 pp 1068-1069), C.27 (wanting/incentive
  salience, `missing`), C.30 (actor-critic / V(s) critic, `partial`), C.32 (two-component DA: salience + utility-RPE, `partial`),
  O.19 (vmPFC/OFC subjective value modulates accumulator drift, Ch 56 pp 1406-1409), O.22 (striatal action-value), G.15–G.18
  (signal-detection / drift-diffusion / LIP accumulator, Ch 56), G.07 (pre-SMA internally-generated initiation), G.20 (global-
  workspace threshold-crossing). **ABSENT (NEW):** salience network / AI / dACC; confidence/metacognition; the decision-to-speak.
- **Literature:** Sridharan/Menon/Seeley triple-network (AI/dACC salience + DMN↔FPN switch; iEEG replication 2024-25);
  Kandel 6e Ch 56 (value modulates drift rate; LIP accumulator); *Evidence accumulation in pre-SMA + insula drives confidence and
  changes of mind* (PMC12311133, 2025); SMA speech-initiation via the frontal aslant tract (bioRxiv 2023.04.04.535557);
  Berridge incentive-salience "wanting" (Psychopharmacology 2007; Robinson-Berridge An. Rev. Psychol. 2025); Niv 2007 tonic-DA
  vigor / opportunity cost (Psychopharmacology); Schultz 2016 two-component DA (detection + utility-RPE).
