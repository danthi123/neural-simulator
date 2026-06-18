# Emergent "one-brain" features — deep-research scoping (2026-06-18)

**Type:** standing-opening-move research + recommendation (read-only; NO build/edit here).
**Frontier:** the integrated one-brain conversational loop (`OneBrainComposer` /
`BrainConversationalAgent(composer_kind="onebrain")`) now runs the WHOLE who/what turn as ONE
persistent interacting spiking loop. The owner asks: what EMERGENT brain-like features does that
integration now make *demonstrable* that the separate host-orchestrated pieces could not?
Four candidates: (1) graceful degradation, (2) neuromodulation-in-the-loop, (3)
reconsolidation-in-the-loop, (4) generalization across the integrated pipeline.

**Top-level goal (load-bearing):** artificial life with a proper brain analogue +
biology-translatable insights. An HONEST NEGATIVE under strict biology IS the deliverable.
"Brain-based only": cognition = neurons/synapses/their communication; host code legitimate only
for the environment (sensory render) and body (motor act). The no-confab moat is a *plus, kept
where free* — not a hard gate (per `feedback_moat_not_hard_lossy_memory_ok`), but it must NEVER
be silently *weakened* by a demo (a demo that lowers the false-accept floor is a regression, not
a feature).

---

## 0. The load-bearing substrate fact (shapes the entire ranking)

Before diagnosing the four features, one fact about the actual code determines which are cheap and
which are duds.

**The composer's bind/unbind/bundle/cleanup ops run through the dedicated RF resonate loop
(`bridge.rf_resonate_steps` → `_rf_advance_one` / the `rf_megastep` CUDA kernel,
`sim/bridge.py:5512-5631`), which deliberately SKIPS `_run_one_simulation_step`.** The RF step is a
pure complex-phasor dynamical system:

```
z_new = decay · rotate(z, ω) + W · z          # decay = exp(_rf_lambda); W = cp_rf_w_re/im (complex synapses)
spike = upward Im-zero-crossing of z_new       # the phase readout
```

The neuromodulator subsystem's effects (`compute_synaptic_gain_multiplier`,
`compute_excitability_drive_pA`, `compute_plasticity_gate_values`) are applied INSIDE
`_run_one_simulation_step` (`sim/bridge.py:5758-5773`, `6050-6055`, `6763-6773`). **Therefore a
global neuromodulator does NOT automatically reach the composer's RF ops.** Two consequences:

- **The PARSER (`BridgeParser`, Izhikevich, slice `[0:P]`) DOES run `_run_one_simulation_step`** (its
  `role_of` / `_train` loop calls `self.bridge._run_one_simulation_step()`,
  `brain_conversational_agent.py:120,133`). So neuromodulation reaches *comprehension* natively, with
  zero new substrate, if the subsystem is enabled on the co-resident bridge.
- **The composer's RF dynamics expose their OWN biologically-meaningful gain/precision knobs that do
  NOT require the neuromodulator subsystem:** the resonate decay `_rf_lambda` (a per-step magnitude
  leak = a precision/gain control on the phasor), the cleanup drive/threshold
  (`RFPhasorComposer._cleanup_drive_pA`, the spiking-cleanup WTA), and the abstention decision (the
  cue-match scan / the learned familiarity gate). These are the real levers for "neuromodulation OF
  the conversational loop" even though they are not wired as `NeuromodulatorConfig` today.

This is why the ranking below puts **graceful degradation FIRST** (it perturbs the substrate that is
already load-bearing in every query, no new wiring) and treats "neuromodulation as a declared
`NeuromodulatorConfig` shaping the composer" as the higher-risk, higher-cost option.

The composer ALSO already has the perturbation surface graceful-degradation needs for free: the
persistent store lives in complex synapses (`store_conns` → `cp_rf_w_re/im`), and the work/cleanup
registers are addressable neuron slices. A lesion = zeroing a fraction of those synapses; dropout =
masking a fraction of the RF slice; synaptic noise = jitter on `cp_rf_w_re/im`. All are
host-orchestrated *perturbations of the brain* (legitimate — the experimenter lesioning tissue), and
the *response* (degrade-vs-abstain) is computed entirely by the spiking substrate (legitimate — the
brain-based standard holds).

---

## 1. Diagnosis — what the integrated loop makes demonstrable, per feature

### (1) GRACEFUL DEGRADATION

**What's newly demonstrable.** When the pipeline was separate host-orchestrated pieces, "robustness"
was untestable as a *system* property: a host `dict.get()` lookup either returns the value or
`KeyError`s — there is no graceful middle. Now the whole turn is a distributed phasor computation:
the fact lives as a superposition across `D=128` complex synapses; recall is a matched-filter +
WTA cleanup over a `V`-way codebook; abstention is a margin test. So you can now ask the
*characteristic* question of a distributed neural memory: under graded lesion / dropout / synaptic
noise, does accuracy fall off *gracefully* and does the residual error convert to *abstention*
(the moat catching low-confidence reads) rather than *confabulation* (confident wrong answers)?
That is a system-level emergent property that did not exist for the host pieces.

**Biological mechanism + catalog.** This is the canonical virtue of **distributed/population
coding**: **E.03 Population coding & vector averaging** ("represented by the *distribution of
activity* across many broadly-tuned neurons; downstream vector sum … extracts the value. **Robust to
noise and single-neuron loss**", catalog L1372) and **D.05 CA3 recurrent collaterals —
autoassociative attractor** / **D.13 pattern completion** (partial/noisy cue → attractor converges to
the stored pattern; Marr 1971 autoassociator, catalog L1137-1148, L1240). The FHRR codebook +
matched-filter cleanup IS a one-shot autoassociative clean-up (the resonator-network reading of
Frady-Sommer); graded degradation of a holographic/VSA memory is a textbook prediction (the
"holographic" property: information is spread, so damage degrades all items slightly rather than
erasing some). The moat-catches-the-residual behaviour maps to a **confidence/familiarity readout**
gating the answer — the project already has the validated learned **Bogacz-Brown familiarity gate**
(CLAUDE.md "familiarity-gate-v320-GO") as the brain-based abstention mechanism.

**Honest expectation.** The argmax cleanup is winner-take-all, so for SMALL lesions accuracy may be
*flat* (the matched filter still wins) and only fall once the lesion crosses a threshold — the
interesting result is the *shape* of the dose-response and whether the fall-off routes to abstention.
A NEGATIVE (a cliff, or silent confabulation under noise) is itself a strong deliverable: it would
map the substrate's robustness boundary and motivate the attractor-cleanup ladder
(CLAUDE.md "spike-native robustness ladder: …c population redundancy + attractor cleanup").

### (2) NEUROMODULATION-IN-THE-LOOP

**What's newly demonstrable.** With one persistent loop, a single global scalar can shape the
*whole* turn's behaviour at once — the defining signature of volume-transmission neuromodulation
(**C.21**, "scalar field rather than per-synapse signal", catalog L894-905). The demonstrable claim
is a **gain/precision → confidence trade-off**: turning one "neuromodulator" knob moves the operating
point along a recall-vs-abstention (hit-rate vs false-accept) curve — i.e. an **inverted-U / SNR**
control over the conversational read-out, exactly the **LC-NE** signature.

**Biological mechanism + catalog.** **C.05 / C.13 Norepinephrine (LC):** "Increases SNR by
simultaneously suppressing background firing and enhancing selective response … Yerkes-Dodson
inverted-U" (catalog L713-717). **C.14 LC-NE:** "Aston-Jones inverted-U: tonic-mode high → exploration
/ labile; phasic → focused; very low → drowsy" (catalog L818). **E.15 Visual attention — top-down
gain:** "Attention multiplies firing rates, sharpens tuning, reduces noise correlations … the
neuromodulator subsystem CAN apply scope-targeted gain (synaptic_gain target)" (catalog L1516-1517).
This is the deepest "translatable insight" candidate: a SINGLE diffuse gain signal that tunes the
precision of an entire cognitive read-out (the Yu-Dayan / Aston-Jones expected-uncertainty framing).

**Honest expectation + the substrate caveat (§0).** Two routes, very different cost:
- **Route A (composer-intrinsic gain — cheap, real):** the RF resonate decay `_rf_lambda` and the
  cleanup drive/threshold ARE the gain/precision controls of the composer; sweeping them and showing a
  monotone hit-rate↔false-accept trade-off (an SNR/inverted-U curve) is a legitimate "gain modulates
  the loop" demonstration, no new wiring. The honest caveat: today these are *constructor constants*,
  not a *declared diffuse neuromodulator* — so the claim is "the loop HAS a gain knob with the
  inverted-U signature", not "a `NeuromodulatorConfig` drives it".
- **Route B (a declared `NeuromodulatorConfig` shaping the loop — higher risk):** to make a *declared*
  modulator shape the composer, it must reach the RF ops, which today it does NOT (§0). It WOULD reach
  the **parser** for free (enable `enable_neuromodulator_subsystem` + an `excitability_drive` /
  `synaptic_gain` NE config; the parser runs the full step). So a defensible Route-B demo is
  "NE-gain shapes COMPREHENSION robustness" (does the parser's role assignment degrade gracefully and
  is its operating point NE-tunable?). Making NE shape the composer's *cleanup* would need a sim/ edit
  (route the gain into the RF/cleanup path) — out of "cheap-first" scope; flag as a follow-on.

### (3) RECONSOLIDATION-IN-THE-LOOP

**What's newly demonstrable.** `update_on_mismatch` already exists and is de-risked 6/6 as a *one-off
call* (`one_brain_composer.py:397`, `rf_phasor_composer.py:372`). What the *persistent* loop newly
makes demonstrable is the *systems-memory* behaviour around it: because the store is persistent and
the prediction-error gate is auto-calibrated FROM THE CURRENT FACT SET (`_calibrate_pe_labile`,
which reads all stored blocks each time), the labilization boundary *moves as the brain learns more*.
So across many facts/turns you can now demonstrate the boundary conditions reconsolidation is famous
for: (a) **prediction-error necessity** — a re-statement (PE≈0) re-stabilizes, a correction (PE high)
rewrites IN PLACE with no duplicate (`count_facts`); (b) **interference / set-size effects** — does
the auto-calibrated gate stay well-separated as `K` grows (the same-vs-different PE distributions
must remain bimodal), or does a denser store blur the gate and cause mis-rewrites / failed updates?
(c) the **moat under reconsolidation** — a never-stored corrective cue must still ABSTAIN.

**Biological mechanism + catalog.** **J.27 Memory reconsolidation** ("reactivating a long-term memory
makes it transiently labile … retrieval re-stabilizes through the same mechanism as initial storage",
catalog L3789-3797). **J.34 Memory imperfections as features** ("Reconsolidation makes retrieval
inherently editable … adaptive prioritization of generalizable structure", catalog L3903-3909). The
in-place PE-gated update is the Osan-Tort-Amaral 2011 mismatch-gated attractor update + Sevenster 2013
PE-necessity (cited in the composer docstring). The *set-size / interference* boundary is the novel
systems-level question the persistent loop unlocks.

**Honest expectation.** The single-fact mechanism is GO; the *risk* is that the cheap, high-value
deliverable here is the **boundary-conditions sweep** (does it survive interference as K grows), which
is genuinely informative either way — a NEGATIVE ("gate blurs above K=N → mis-rewrites") maps a real
substrate limit and the no-confab moat / familiarity-gate is the obvious mitigation.

### (4) GENERALIZATION

**What's newly demonstrable.** The project already has a comprehensive generalization arc (PPMI stream
cortex + cross-modal convergence + the visual-similarity capstone, CLAUDE.md). On the INTEGRATED loop,
the marginal new question is narrow: when the composer is fed the *learned, similarity-structured*
grounded codes (`OneBrainComposer(grounded_codes=…)`, already plumbed to the inner RF composer,
`one_brain_composer.py:81`), does the END-TO-END persistent loop still (a) bind/recall correctly AND
(b) preserve the codes' category structure through bind→store→unbind→cleanup, i.e. does a query with a
*near-synonym / same-category* cue degrade toward the right neighbour rather than randomly?

**Biological mechanism + catalog.** Convergence-zone semantics (ATL hub-and-spoke,
Patterson-Lambon-Ralph; Garagnani-Pulvermüller spiking precedent) + **E.03 population coding** (graded,
similarity-preserving codes). But note the project's own settled caveat (CLAUDE.md, CYCLE 88;
`_step3_correlated_percept_boundary`): "the compose algebra TOLERATES correlation up to ≈0.98 … this is
compose-ROBUSTNESS to correlation, NOT generalization-across-similar-concepts." So the honest framing is
that the loop does NOT *add* generalization — generalization lives in the codes. This makes (4) the
LEAST novel of the four for THIS integration question (it re-demonstrates the already-validated arc end
to end), hence ranked last.

---

## 2. Ranked options (leverage × cheapness-to-demonstrate on the EXISTING OneBrainComposer/agent)

| Rank | Feature / sub-option | Demonstrable claim | Reuses (existing machinery) | Effort | Dud-risk |
|---|---|---|---|---|---|
| **1** | **Graceful degradation (lesion/dropout/noise dose-response on the persistent store)** | Accuracy falls off GRACEFULLY with lesion fraction, and residual error routes to ABSTENTION (moat), not confabulation — the distributed-memory signature absent for a host dict. | `OneBrainComposer.store_conns`/`cp_rf_w_re/im`; `_read_blocks`/`_scan`; the no-confab moat (`is None`); the learned familiarity gate (optional). NO new substrate. | **Low** (a perturbation harness around the built composer + a dose-response sweep). | **Low-med** — even a *cliff* or a *confabulation-under-noise* result is a publishable substrate-boundary deliverable. |
| **2** | **Reconsolidation interference / set-size boundary (the persistent-loop systems behaviour)** | PE-necessity holds AND the auto-calibrated labilization gate stays bimodal/well-separated as K grows (or maps the K at which it blurs); moat abstains on never-stored corrective cues. | `update_on_mismatch`, `_calibrate_pe_labile`, `count_facts`, `_patient_prediction_error`. NO new substrate. | **Low-med** (a multi-fact / multi-correction protocol + measure the same/diff PE distributions vs K). | **Low** — informative either way; sharpens a known mechanism. |
| **3a** | **Neuromodulation, Route A (composer-intrinsic gain → SNR/inverted-U)** | Sweeping the RF gain/precision (decay `_rf_lambda` and/or cleanup drive/threshold) traces a monotone recall↔false-accept (SNR) curve with an inverted-U optimum — gain modulates the WHOLE loop's confidence. | RF resonate (`rf_resonate_steps`/`_rf_lambda`); `RFPhasorComposer._cleanup_drive_pA`/`_cleanup_window`; the moat. NO new substrate. | **Low-med** (sweep a knob; the honesty caveat = it's a constant, not a declared modulator). | **Med** — risk the curve is flat/degenerate over the usable range; mitigated by also sweeping injected synaptic noise. |
| **3b** | **Neuromodulation, Route B (a DECLARED NE `NeuromodulatorConfig` shaping COMPREHENSION)** | Enabling an NE modulator (`excitability_drive`/`synaptic_gain`) on the co-resident bridge tunes the PARSER's role-assignment robustness with the inverted-U signature. | `sim/neuromodulators.py` (`NeuromodulatorConfig`, `from_error_persistence`); `enable_neuromodulator_subsystem`; `BridgeParser` (runs the full step). NO sim/ edit (parser path only). | **Med** (enable subsystem on the co-resident bridge; verify the parser sees the drive; sweep baseline). | **Med-high** — the parser is robust/saturated (drive 2500 pA), so the modulation window may be narrow; and it touches comprehension, not the headline composer. |
| **4** | **Generalization end-to-end on the integrated loop** | With grounded similarity-structured codes, the persistent loop binds/recalls AND a same-category cue degrades toward the right neighbour. | `OneBrainComposer(grounded_codes=…)`; the existing PPMI/grounded-codes arc. | **Med** (needs the grounded-code fixture; largely re-runs a validated arc). | **High (as a NOVEL result)** — the loop adds nothing; generalization is in the codes (project's own settled caveat). |

**Recommendation:** **Rank 1 (graceful degradation)** is the single highest leverage × cheapness item:
it is the one feature that (i) is genuinely NEWLY demonstrable as a *system* property of the integrated
loop, (ii) needs zero new substrate (perturb what is already load-bearing in every query), (iii) directly
exercises the two deepest brain-analogue virtues the project cares about — distributed coding robustness
(E.03) and the no-confab moat — and (iv) yields a clean quantitative, anti-cheatable result, with an
HONEST NEGATIVE that is itself a deliverable. Rank 2 (reconsolidation interference) is the natural
immediate follow-on (same "perturb the persistent loop" shape, also zero new substrate).

---

## 3. Recommended cheap-first de-risk (the single demonstration to run first)

**Demonstration: lesion/noise dose-response of the integrated one-brain recall, with abstention routing.**

**Setup (all on the EXISTING built composer — reuse-by-import, NO sim/ edit):**
- Build `c = OneBrainComposer(seed=42, D=128, vocab=…)` (or via
  `BrainConversationalAgent(composer_kind="onebrain")`). Store a fixed fact set (e.g. K=8 distinct
  SVO facts at production D, as in the 320-scale demo's 8-fact set).
- **Perturbation knobs (host-applied to the BRAIN, then the spiking substrate computes the response):**
  1. **Synaptic lesion** — zero a fraction `p ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5}` of the store synapses:
     mask `p·len(c.store_conns)` entries of `cp_rf_w_re/im` (or rebuild `store_conns` with that fraction
     dropped) BEFORE the query. (Graded ablation of stored tissue.)
  2. **Synaptic noise** — add Gaussian jitter of increasing σ to `cp_rf_w_re/im` (phase/magnitude noise
     on the holographic trace).
  3. **Neuron dropout** — mask a fraction of the RF readout/cleanup slice (`c.rf_mask` sub-indices) so
     those units are silent during the read.
- **Measure, per perturbation level, over all stored facts (and a matched set of UNSTORED query cues):**
  - **Recall accuracy** = fraction of stored `query_patient(agent,action)` returning the correct patient.
  - **Abstention rate on stored** = fraction returning `None`/`unknown` (graceful "I'm not sure").
  - **Confabulation rate** = fraction returning a CONFIDENT WRONG answer (a wrong patient, not `None`).
  - **Moat integrity** = false-accept rate on UNSTORED cues (must stay ~0; the moat must not break under
    perturbation — this is the must-not-regress guard).

**What GO looks like (quantitative bar):**
- **Monotone, graceful fall-off:** recall accuracy decreases monotonically and *smoothly* with
  perturbation (no single-step cliff from ~1.0 → ~chance); concretely, at least one intermediate
  perturbation level sits in a genuine middle band (e.g. recall in [0.4, 0.9]) rather than the curve
  being a step function.
- **Error → abstention, not confabulation:** as recall drops, the **abstention** rate rises and the
  **confabulation** rate stays low (target: confabulation ≤ ~0.1 across the swept range, i.e. most lost
  recall converts to `None`, not to confident-wrong) — the moat catching the residual.
- **Moat preserved:** false-accept on unstored cues stays ≈ 0 at every perturbation level (the
  must-not-weaken guard).
- **Distributed signature:** small lesions (p≤0.1) degrade ALL facts a little (mean accuracy dips
  slightly) rather than erasing a SUBSET cleanly — the holographic/population property (E.03).

**What an HONEST NEGATIVE would mean (and that it is still a deliverable):**
- A **cliff** (accuracy stays ~1.0 then collapses to chance at one threshold) → the argmax cleanup is
  brittle; maps the substrate's robustness boundary and motivates the **attractor-cleanup ladder**
  (population redundancy + iterative clean-up) as the next build. Deliverable.
- **Silent confabulation** (lost recall returns confident WRONG answers, confabulation rate climbs) →
  the moat does not catch low-confidence reads under this perturbation; pinpoints exactly where the
  familiarity/abstention threshold must be made perturbation-robust. Deliverable (and high-value: it is
  the moat's failure mode under damage).
- Either negative is a real, biology-translatable finding about how a point-neuron FHRR memory fails —
  precisely the "honest negative under strict biology IS the deliverable" mandate.

---

## 4. Anti-cheat controls (proving the effect is REAL and brain-based, not a host artifact)

1. **Dose-response monotonicity (not a single point).** Require the FULL curve across ≥5 perturbation
   levels, not a single "it degrades" datapoint — a real distributed-memory degradation is graded; a
   host artifact (e.g. an exception swallowed into a default) would be a step. Report the curve.
2. **Scrambled-lesion control.** For matched lesion fraction `p`, compare (a) random synapse subset vs
   (b) a STRUCTURED lesion (e.g. all synapses of one stored block). If "graceful degradation" were a
   host bookkeeping artifact, the two would look the same; for a genuine distributed code, random
   lesions degrade ALL facts slightly while a block-structured lesion erases ONE fact — the contrast is
   the signature that the code is genuinely distributed.
3. **The response is SPIKING, not host.** Assert that recall under perturbation is computed by the RF
   substrate (the read goes through `rf_resonate_steps` + the cleanup membrane/firing readout), and that
   the ONLY host operations are (i) applying the lesion to `cp_rf_w_re/im` (experimenter damaging tissue
   — legitimate) and (ii) reading argmax off the cleanup membrane (a readout of spiking output —
   legitimate, the same readout the validated pipeline already uses). No host re-implements the recall.
4. **Moat-not-weakened guard (HARD).** False-accept rate on UNSTORED cues must be reported at EVERY
   perturbation level and must not rise above the intact baseline. A demo that achieves "graceful
   degradation" by lowering the abstention threshold (trading false-accepts for hit-rate) is a
   REGRESSION, not the feature, and must be rejected. (Per `feedback_moat_not_hard_lossy_memory_ok`: the
   moat may be *traded* deliberately for a learned-lossy capability, but it must never be *silently*
   weakened by an unrelated robustness demo.)
5. **Intact positive control.** At p=0 the composer must reproduce its validated recall (1.00 at the
   tested scale) — proving the harness itself is not degrading the baseline.
6. **Noise-vs-lesion convergence.** The three independent perturbations (synaptic lesion, synaptic
   noise, neuron dropout) should each produce a graceful curve. If only one does, the "graceful
   degradation" claim is mechanism-specific, not a general distributed-code property — report which.

(For the Rank-2 reconsolidation follow-on, the matching anti-cheats are: PE-necessity — a re-statement
must NOT rewrite (`wrote=False`); the gate must be CALIBRATED FROM THE DATA (`_calibrate_pe_labile`,
not tuned to the probe); a permuted-cue control (a correction with a wrong agent/action) must ABSTAIN;
and `count_facts` must stay 1 after a true correction, 2 only if a naive append bug exists. For the
Rank-3 neuromodulation follow-on: the effect must TRACK the gain knob monotonically with a no-modulation
control and a permuted-target control, and must not be reproducible by a host scalar threshold.)

---

## 5. Reusable machinery summary (the exact files/functions/params the build reuses)

**Composer / agent (the loop under test):**
- `research/runners/one_brain_composer.py` — `OneBrainComposer`: `.store()`/`.hear()`,
  `.query_patient()`/`.query_agent()`/`.ask_yes_no()`, `.update_on_mismatch()`, `.count_facts()`,
  `._read_blocks()`/`._scan()`. **Perturbation surface:** `self.store_conns` (the persistent fact
  synapses), `self.rf_mask` (the RF slice), `self.b.cp_rf_w_re` / `self.b.cp_rf_w_im` (the complex
  store weights to lesion/noise), `self.k_max`, `self.D`.
- `research/runners/brain_conversational_agent.py` — `BrainConversationalAgent(composer_kind="onebrain")`
  (`.what_does`/`.who_does`/`.is_it_true`/`.describe`/`.reason_chain`); `BridgeParser` (the Izhikevich
  comprehension slice that DOES run `_run_one_simulation_step`).
- `research/runners/rf_phasor_composer.py` — `RFPhasorComposer` (the inner composer + the TEST ORACLE
  for parity): the RF gain knobs `_cleanup_drive_pA`, `_cleanup_window`; `_calibrate_pe_labile`;
  `enable_spiking_cleanup`; `grounded_codes` interface.

**Substrate knobs (RF dynamics — the composer-intrinsic gain/precision, NO neuromodulator subsystem):**
- `sim/bridge.py` — `rf_resonate_steps` (5551), `_rf_advance_one` (5512), the `rf_megastep` kernel
  (5584); the gain/precision parameters `_rf_lambda` (decay), `_rf_omega`, `_rf_floor`;
  `cp_rf_w_re`/`cp_rf_w_im` (the complex synaptic store); `rf_kick`/`rf_read_phases`/
  `rf_set_complex_weights`; the co-residence mask `_rf_neuron_mask`.

**Neuromodulator subsystem (Route B / declared-modulator follow-on only):**
- `sim/neuromodulators.py` — `NeuromodulatorConfig`, `ModulatorTarget` (`synaptic_gain` /
  `excitability_drive` / `plasticity_rate` target types), `ProductionRule` (`from_error_persistence`
  for NE, `manual` for a probe), the default NE/DA/ACh config helpers; `NeuromodulatorManager`
  (`compute_synaptic_gain_multiplier`, `compute_excitability_drive_pA/_per_neuron`,
  `set_concentration` for manual probes).
- `sim/bridge.py` — `enable_neuromodulator_subsystem` gate + the apply sites (5758-5773 synaptic_gain,
  6050-6055 excitability_drive, 6763-6773 step+gates). **Caveat (§0): these are on the
  `_run_one_simulation_step` path → they reach the PARSER, NOT the RF composer ops.**

**Catalog grounding (cite in the build's findings doc):**
- E.03 (population coding, robust to noise/loss) · D.05 + D.13 (CA3 autoassociator / pattern
  completion) — graceful degradation.
- C.05 / C.13 / C.14 (LC-NE SNR + Aston-Jones inverted-U) · E.15 (attention gain) · C.21
  (volume-transmission scalar field) — neuromodulation.
- J.27 (reconsolidation) · J.34 (memory imperfections as features) — reconsolidation-in-the-loop.

**Validated controls/mechanisms to reuse as anti-cheat scaffolds:**
- The no-confab moat (`is None` / `"unknown"` returns throughout the composer) + the learned
  Bogacz-Brown familiarity gate (CLAUDE.md "familiarity-gate-v320-GO") — the abstention-routing measure.
- The 320-scale 8-fact demo protocol (`consolidated_320_conversation_demo.py`) — the intact positive
  control + fact-set fixture.
