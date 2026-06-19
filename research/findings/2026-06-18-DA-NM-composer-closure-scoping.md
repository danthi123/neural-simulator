# DA/NM → the conversational composer — the "one self" closure (roadmap #6) SCOPING

**Date:** 2026-06-18
**Type:** READ-ONLY deep-research + catalog-review scoping (no code edited; no GPU). The standing
opening move for a new direction (CLAUDE.md "deep research + catalog review FIRST").
**Direction:** TRUE-ONE-BRAIN roadmap **#6** — let the SHARED spiking dopamine/salience signal (the
spiking SNc, already on the merged "one brain") actually MODULATE the conversational composer, so the
limbic core reaches the cortex on BOTH halves (navigation AND conversation), not just navigation. This
is the deepest "one self" step: the same dopamine the BG actor learns from also shapes how the
composer recalls/encodes.

---

## 0. TL;DR for the controller

- **The obstacle is real and precisely localized.** The composer's resonate-and-fire (RF) ops
  (`rf_resonate_steps` → `_rf_advance_one` / the `rf_megastep` megakernel) run their OWN dynamics loop
  and **bypass `_run_one_simulation_step`**, where the neuromodulator subsystem steps
  (`bridge.py:6764-6765`). So the `dopamine` modulator (which IS already built + driven by the spiking
  `limbic_snc`/`snc` on the merged bridge, `nav_conv_merged_bridge.py:757-761`) reaches the parser but
  **not** the RF bind/cleanup ops. The DA scalar exists; it just never touches the resonate dynamics or
  the cleanup read-out.
- **The biologically-grounded target is settled by the catalog + literature.** DA's **Component 1 =
  unselective detection/salience** (catalog **C.32**) "amplifies learning rate and downstream **sensory
  gain** on any potentially important event **before identification**." In PFC, D1 stimulation follows
  an **inverted-U** that at the optimal level **sharpens tuning by suppressing responses to nonpreferred
  directions** (Vijayraghavan/Arnsten 2007) — i.e. DA literally **sharpens a winner-take-all read-out**.
  And the **Lisman-Grace hippocampal-VTA loop** (2005): a **novelty** signal drives VTA DA → DA in
  hippocampus **enhances LTP and gates entry of information into long-term memory** = novelty-gated
  ENCODING. These map cleanly onto the composer's two knobs: the **cleanup gain/sharpness** (recall
  confidence) and the **encoding/labilization gate** (what gets written / reconsolidated).
- **TOP-RANKED option: salience-gated cleanup SHARPNESS, implemented as a DA-driven `confidence_gate`
  (and a parallel DA-driven cleanup-resonate gain), NOT a DA-raised abstention-lowering gain.** The
  composer **already has** the exact lever — `OneBrainComposer(confidence_gate=g)` — built last cycle as
  the graceful-degradation fix. The minimal, moat-SAFE closure is to make `g` **rise** with DA/salience
  (a salient/novel turn ⇒ a *more decisive* read = the inverted-U sharpening), which **tightens** the
  abstention boundary, never loosens it. This is the one option whose moat risk is **structurally
  inverted** (it can only make the moat stricter), so it is the right cheap-first target.
- **Cheap-first de-risk (numpy/CPU, no GPU):** drive a (toy) spiking `limbic_snc` to a high-salience and
  a low-salience level, read `get_concentration("dopamine")`, map it to the composer's cleanup
  sharpness, and show (a) salient ⇒ measurably more decisive correct recall (higher margin / better
  recall under noise), (b) **the no-confab moat is unchanged-or-stricter at EVERY salience level** (zero
  new false-accepts on unstored cues), (c) a **lesion** (sever `snc→dopamine`, i.e. zero the DA source)
  abolishes the modulation. Frozen GO bar below.
- **`sim/` edit required?** **NO for the top-ranked option** — it is a composer-runner-layer read of
  `bridge.neuromodulator_manager.get_concentration("dopamine")` between ops, scaling the EXISTING
  `confidence_gate`/cleanup knob. A `sim/` edit is **only** needed for the deeper variant (threading the
  DA scalar into the RF resonate `_rf_lambda`/megakernel so DA changes the dynamics mid-resonate); that
  minimal edit is sketched in §7 but is **deferred** below the host-layer de-risk.

---

## 1. Diagnosis — the precise obstacle + the biologically-grounded target

### 1a. The mechanism gap (verified in code)

The neuromodulator subsystem steps **inside** `_run_one_simulation_step`:
- `sim/bridge.py:6764-6765` — `self.neuromodulator_manager.step(self)` (decay + production rules,
  including `from_region_firing_signed` reading the SNc firing).
- Its effects are applied **within the same step**: `compute_plasticity_rate_multiplier`
  (`:6863`), `compute_synaptic_gain_multiplier` (`:5760/:5773`),
  `compute_excitability_drive_pA` (`:6052`), `compute_plasticity_gate_values` (`:6773`).

The composer's resonate ops **do not** call `_run_one_simulation_step`:
- `rf_resonate_steps` (`bridge.py:5551`) **"skips the full `_run_one_simulation_step` machinery
  (conductance / plasticity / recording / engram / gate couplings / stats)"** — it loops
  `_rf_advance_one` (`:5512`), or, with `enable_rf_cudagraph`, the `rf_megastep` CUDA kernel
  (`:5616`). Neither path consults `neuromodulator_manager` at all.
- So a composer query — `kick → rf_set_complex_weights → rf_resonate_steps(period+8) → read membrane`
  (`one_brain_composer.py:_read_block`/`_read_all_blocks`; `rf_phasor_composer.py:_resonate`) —
  **never sees the DA concentration**. The cleanup scores are read straight off
  `cp_membrane_potential_v` with a fixed argmax/margin; nothing modulates them.

**Conclusion:** the DA signal **is produced** on the merged bridge (the spiking `limbic_snc`/`snc` →
`dopamine` modulator, `nav_conv_merged_bridge.py:757-761` — roadmap #1/#2 are DONE), but it is **never
read by the conversational composer**. The audit's hypothesis (`...roadmap.md` §3 #6) is confirmed: the
RF ops are off the NM path. The audit's framing that this "likely needs a `sim/` edit (a new
`_rf_lambda`/cleanup-gain route)" is **correct only for the deep dynamics variant**; the cleanup-gain
route can be closed entirely at the composer-runner layer (§2, §7).

### 1b. The biologically-grounded target (what DA *should* do to the composer)

Two canonical DA→cortex functions, both directly mappable onto a composer knob:

**(i) Salience-gated GAIN/SHARPENING of the read-out (recall confidence).** Catalog **C.32**
(Schultz-2016 two-component DA): **Component 1** (60–90 ms, unselective detection/salience) "amplifies
learning rate and **downstream sensory gain** on any potentially important event before identification."
And the mechanism of that gain in cortex is the **D1 inverted-U** (Vijayraghavan/Arnsten 2007, PMID
17277774): at the *optimal* DA level, D1 stimulation **"enhances spatial tuning by suppressing responses
to nonpreferred directions"** — i.e. DA **sharpens** a tuned/competitive read-out (suppresses the
runner-ups), which is *exactly* a WTA-margin sharpening. Catalog **E.15** (top-down attentional gain):
"attention **multiplies firing rates ~20–30%**, **sharpens tuning, reduces noise correlations**, modeled
as a gain field." ⇒ **A salient/novel conversational turn should make the composer's cleanup MORE
decisive** (a sharper argmax, a larger margin, more robust recall under noise) — the recall-confidence
hook.

**(ii) Novelty/salience-gated ENCODING (what gets written / reconsolidated).** The **Lisman-Grace
hippocampal-VTA loop** (2005, PMID 15924857): the hippocampus detects **novelty** → (via
subiculum/accumbens/VP) drives **VTA DA firing (with salience + goal info)** → DA release in hippocampus
**"produces an enhancement of LTP and learning"** → the loop **"regulates the entry of information into
long-term memory."** ⇒ **DA/novelty should gate the composer's WRITE/labilization** (a novel fact
encodes more strongly; the reconsolidation labilization gate, `update_on_mismatch`, is DA/PE-gated). This
is the encoding hook (the reconsolidation already computes a phase-PE; DA is the missing neuromodulatory
gate on it).

**The biological "value" of doing this (the directive's "one self"):** in the brain the *same* midbrain
DA population teaches the BG actor AND modulates cortical gain/encoding (mesocortical projection,
catalog C.04). Wiring the shared spiking SNc to the composer makes the artificial agent's recall +
encoding **modulated by the same motivational/salience state** that drives its navigation — one limbic
core, two cortical consumers. That is the functional content of "one self," and it is the emergent-feature
#3 (neuromodulation) the emergent-features scoping flagged.

---

## 2. Ranked biologically-grounded options

Each: the biology anchor → the composer knob → the moat risk → the cheapness.

### OPTION A (TOP-RANKED) — salience-gated cleanup SHARPENING via a DA-driven `confidence_gate`
- **Biology:** C.32 Component-1 gain + Vijayraghavan/Arnsten D1 inverted-U **"sharpens tuning by
  suppressing nonpreferred responses"** + E.15 attentional gain. DA at the optimal level ⇒ a sharper,
  more-decisive read-out.
- **Knob:** `OneBrainComposer.confidence_gate` (`one_brain_composer.py:84`, `_read_block:211`,
  `_read_all_blocks:262`, `_margin:175`) **already exists** — it blanks a block whose cleanup margin
  `(peak−runner_up)/peak` is below `g` (→ abstain). Make `g` **rise with DA**: `g_eff = g0 + k·(DA −
  DA_baseline)`. High salience ⇒ higher gate ⇒ only **confident** blocks answer (more decisive); a
  faint/noise-dominated block is rejected. (A parallel, equivalent realization: scale the cleanup
  *resonate* drive or the per-concept threshold with DA so the winner's margin grows — the
  `rf_phasor_composer._cleanup` / `_spiking_cleanup._cleanup_drive_pA` analog.)
- **Moat risk: STRUCTURALLY INVERTED (this is why it ranks first).** Raising `confidence_gate` can only
  make abstention **stricter** — it converts marginal reads to abstain, it can NEVER turn an abstain
  into a false-accept. The graceful-degradation finding (`2026-06-18-emergent-graceful-degradation-derisk.md`)
  already proved `g=0.15` **closes** moat leaks + flat-uncertain confabulations while preserving the
  functional regime. So DA-raised `g` is moat-SAFE by construction. The ONLY thing to guard is that DA
  does not *lower* `g` below `g0` (clamp `g_eff ≥ g0`), and that the salient-recall claim is a
  decisiveness/robustness gain, not a fabrication.
- **Cheapness: HIGHEST.** Reuse-by-import, composer-runner layer, **NO `sim/` edit**. The DA scalar is
  `bridge.neuromodulator_manager.get_concentration("dopamine")`, read between ops.
- **Why first:** maximal biology fidelity (the canonical Component-1/D1-sharpening) × the one option
  whose moat risk is provably one-directional (stricter) × zero protected-code change × an existing knob.

### OPTION B — value/novelty-gated ENCODING (the reconsolidation labilization threshold)
- **Biology:** Lisman-Grace novelty→VTA→hippocampal-LTP gating of memory entry (PMID 15924857); the
  encoding half of C.32 Component-1 ("amplifies **learning rate**"). A novel/salient fact writes more
  strongly; a mismatch under high DA labilizes + rewrites.
- **Knob:** the reconsolidation `pe_labile` gate (`update_on_mismatch`,
  `one_brain_composer.py:417` / `rf_phasor_composer.py:372`) — currently auto-calibrated from the
  facts' own PE distribution. Make the labilization threshold DA-gated: a high-salience correction
  lowers the bar to rewrite (encodes the new fact); a low-salience restatement does not. (Or, more
  simply: scale the *write strength* — the store-block weight magnitude — with DA at `_write_block`.)
- **Moat risk: MODERATE — this is the riskier direction.** Encoding-gating touches WHAT is written, so a
  mis-set DA gate could (a) over-write (a low-PE restatement spuriously rewrites — a memory corruption,
  not a confab-on-query, but still a fidelity loss) or (b) under-write. It does NOT directly create
  query-time confabulation, but it is a write-side fidelity knob, so it needs the
  count_facts/no-duplicate + restabilize-on-low-PE guards the reconsolidation already has, plus a
  DA-OFF == current-behavior control.
- **Cheapness: MODERATE.** Composer-runner layer, NO `sim/` edit, but the de-risk is a reconsolidation
  battery (more moving parts than the cleanup-margin read).
- **Why second:** strong biology (the Lisman-Grace loop is the textbook novelty-encoding mechanism) but
  the moat risk is two-directional (write fidelity), so it is the natural *follow-on* once Option A
  lands the read-side hook.

### OPTION C — DA-modulated dlPFC gain (the dialogue planner's working memory)
- **Biology:** D1 inverted-U in PFC WM is literally measured on **dlPFC delay-period firing**
  (Vijayraghavan/Arnsten 2007) — DA sets the gain/stability of the NMDA attractor that holds the
  dialogue topic. C.04 mesocortical DA→PFC WM.
- **Knob:** the dlPFC dialogue loop on the merged bridge (`_build_dlpfc_loop_population`,
  `cortex_ctx`/`dlpfc_wm`, NMDA) — DA would scale the WM attractor gain (e.g. via the EXISTING NM
  `excitability_drive` scope=`group:dlpfc_wm` or `synaptic_gain`, which **already** flow through
  `_run_one_simulation_step` because the dlPFC IS a normal Izhikevich region, NOT an RF-bypass region).
- **Moat risk: LOW-but-INDIRECT.** The dlPFC drives `elaborate`/topic-selection, not the no-confab
  cue-match; modulating its gain changes dialogue planning quality (inverted-U: too much DA →
  perseveration, too little → drift), not the abstention decision. Low moat risk, but it is the LEAST
  load-bearing for "DA reaches the *composer*" — the dlPFC is already on the NM path, so this is the
  *easiest* but addresses a different organ than the bind/cleanup the directive names.
- **Cheapness: HIGH** and **already wired-able with ZERO new code** (declare a `dopamine` target
  `excitability_drive scope=group:dlpfc_wm` — the NM subsystem applies it natively because the dlPFC is
  not RF). But it does **not** close the stated gap (DA → the RF composer ops).
- **Why third:** trivially achievable and biologically apt, but it modulates the planner, not the
  bind/cleanup the directive points at. Good as a **free add-on** alongside Option A (one extra
  `ModulatorTarget`), not the headline closure.

### Recommendation
**Do Option A as the cheap-first de-risk (read-side, moat-safe-by-construction).** Carry Option C as a
zero-cost companion `ModulatorTarget` (dlPFC gain) since it rides the existing NM path. Defer Option B
(encoding/reconsolidation gating) to the follow-on once A's read-side hook is validated. Defer the deep
RF-dynamics `sim/` edit (§7) below all of these — it is only needed if a continuous DA-modulated
resonate decay is wanted beyond the discrete cleanup-margin gate.

---

## 3. What existing project machinery is reusable (named)

- **The DA signal source (already on the merged bridge):** `nav_conv_merged_bridge.py:748-761` builds the
  `dopamine` `NeuromodulatorConfig` with `from_region_firing_signed` over `["limbic_snc"]` (minimal
  organ) or `["snc"]` (full nav critic) — driven by the **spiking** SNc firing. The DA scalar is read by
  `bridge.neuromodulator_manager.get_concentration("dopamine")` (`sim/neuromodulators.py:228`). Threshold
  0.0 ⇒ neutral-at-rest (quiescent SNc → DA = baseline → multiplier ~1.0), so it cannot suppress the
  conversational plasticity — confirmed in the merge comment (`:738-745`).
- **The composer knob (already built):** `OneBrainComposer(confidence_gate=g)` +
  `_margin`/`_read_block`/`_read_all_blocks` (`one_brain_composer.py:84,175,211,262`). Default 0.0 =
  byte-identical. The graceful-degradation finding validated it
  (`2026-06-18-emergent-graceful-degradation-derisk.md`): `g=0.15` closes flat-uncertain
  confabs/leaks, `g=0.30` over-blanks — so the operating band is known.
- **The cleanup primitives (for the alternative resonate-gain realization):**
  `rf_phasor_composer._cleanup` / `_spiking_cleanup` (`_cleanup_drive_pA=60.0`, `_cleanup_window=120`,
  `rf_phasor_composer.py:79-80,208`) — the spiking NEF cleanup whose drive/threshold could be DA-scaled.
- **The NM subsystem (for Option C, zero new code):** `sim/neuromodulators.py`
  `ModulatorTarget(target_type="excitability_drive"|"synaptic_gain", scope="group:dlpfc_wm")` +
  `compute_excitability_drive_per_neuron` (`:562`) — already applied inside `_run_one_simulation_step`
  for normal regions; the dlPFC qualifies.
- **The reconsolidation machinery (for Option B):** `update_on_mismatch` + `_calibrate_pe_labile` +
  `count_facts` (`one_brain_composer.py:388,417,440`; `rf_phasor_composer.py:351,372,397`), de-risked
  6/6 (`2026-06-17-reconsolidation-update-derisk-GO.md`).
- **The RPE-battery harness style (for the salience-source validation):**
  `research/runners/sc_n5_rpe_probe.py` / `snc_pavlovian_probe.py` (drive a US/cue, read SNc firing,
  lesion the source) — the same drive-SNc-and-read pattern the de-risk reuses to set the two salience
  levels.
- **The merge builder + masked RF co-residence:** `nav_conv_merged_bridge.build_merged_nav_conv_bridge`
  (the `co_resident_limbic` + `co_resident_rf` path; the `limbic` handles at `:862-866` expose the
  limbic-slice bases so a probe can drive `limbic_reward_us` and read `limbic_snc`).

---

## 4. The recommended cheap-first de-risk (numpy/CPU; the falsifiable probe + frozen GO bar)

**Goal:** falsify, cheaply (numpy/CPU `SIM_BACKEND=numpy`), the load-bearing claim that **a DA/salience
signal — sourced from a spiking SNc — modulating the composer's cleanup sharpness produces a
biologically-meaningful, useful recall effect (a salient turn ⇒ a more-decisive correct recall) WITHOUT
breaking (and ideally STRENGTHENING) the no-confab moat.**

**Probe (`research/runners/_da_composer_salience_cleanup_derisk.py`, to be built):**
1. Build a small composer (`OneBrainComposer(D=64, k_max=8)` or the `RFPhasorComposer` analog, numpy
   path). Store K=6–8 SVO facts.
2. **Salience source = spiking, not host.** Stand up a tiny spiking `snc`-like pool (reuse the limbic
   slice via the merged builder's `co_resident_limbic` handles, OR a standalone 40-neuron Izhikevich pool
   driven to two firing levels) + the `dopamine` `from_region_firing_signed` modulator. Step it a few ms
   to set DA at two operating points: **DA_high** (drive the pool hard → high firing → DA above baseline =
   a "salient/novel turn") and **DA_low** (quiescent → DA ≈ baseline). Read `DA =
   get_concentration("dopamine")`.
3. **Map DA → the composer knob (read-side):** `g_eff = max(g0, g0 + k·(DA − DA_baseline))` (clamped so
   DA can only sharpen). Set `composer.confidence_gate = g_eff`.
4. Under **added cleanup noise** (a fixed complex jitter on the store reconstruction, the same dial the
   graceful-degradation probe used), measure, at DA_high vs DA_low:
   - **recall correctness** on the K stored cues,
   - **mean cleanup margin** `_margin` on the correct reads (the decisiveness),
   - **moat false-accept rate** on 5 UNSTORED cues (must stay 0),
   - **abstention rate** (graceful: lost recall → abstain, not confab).
5. **Controls (all in the same run):**
   - **DA-OFF / lesion:** zero the DA source (or sever `snc→dopamine`) → `g_eff` collapses to `g0` →
     the modulation **vanishes** (the salient-vs-non-salient difference disappears). Decisive that the
     effect is DA-driven and neural.
   - **Permuted-DA:** shuffle which turn is "salient" vs not → the recall-decisiveness advantage must
     NOT track the shuffled label (it must track the true SNc firing).
   - **Host-scalar anti-cheat:** the DA must come from the SNc *firing* (`from_region_firing_signed`),
     not a host constant; the lesion control proves this.

**Frozen GO bar (pre-registered):**
- (a) **Useful recall effect:** at DA_high vs DA_low, under matched cleanup noise, recall is **≥ as
  good** AND the mean correct-read margin is **higher by ≥ 1.3×** (salient ⇒ more decisive) — OR, at a
  noise level where DA_low recall has degraded, DA_high recall is **≥ +1 fact** recovered. (Either the
  decisiveness *or* the robustness limb suffices; both reported.)
- (b) **Moat HELD-OR-STRICTER (the hard gate):** moat false-accepts on the 5 unstored cues = **0 at
  EVERY DA level** (DA_high included). Confabulation on stored cues does **not increase** with DA. (This
  is the load-bearing safety claim; by construction a raised `g` cannot loosen it, and the probe must
  show it empirically.)
- (c) **Lesion abolishes the effect:** with the DA source severed, the DA_high−DA_low recall/margin
  difference is **within ±5%** (the modulation is gone) — proving it is the synaptic SNc, not a re-hidden
  host scalar.
- **Multi-seed:** ≥ 5/6 seeds (42–47) pass (a)+(b); the lesion control (c) is mechanistic so 3 clean
  seeds is conclusive there.

**What an honest NEGATIVE means (the deliverable):**
- If **(a) fails** (DA sharpening buys no measurable recall/decisiveness improvement) → the honest
  finding is "the composer's cleanup is already saturated/decisive in the functional regime, so a
  salience-gain has no headroom there; the salience hook is only load-bearing in the *degrading* regime
  (the graceful-degradation tail) or for *encoding* (Option B), not for routine recall." That cleanly
  re-points the work to Option B (encoding-gating) — a real result, not a failure.
- If **(b) ever fails** (a DA level raises false-accepts) → that would mean the implementation lowered
  `g` somewhere (a bug) — STOP and fix; do not weaken the moat. (Per the owner's `feedback_moat_not_hard_lossy_memory_ok`,
  the moat is a plus-not-a-hard-gate, but here it is FREE to keep — Option A cannot need to trade it —
  so any moat loss in Option A is a bug, not a design trade.)

---

## 5. Anti-cheat controls it needs (all must hold)

1. **The modulation is NEURAL (DA from the spiking SNc firing, not a host scalar).** DA =
   `from_region_firing_signed` over the spiking `limbic_snc`/`snc` pool. The **lesion** control (sever
   `snc→dopamine` / zero the DA source) MUST abolish the modulation — a host-constant version would be
   lesion-insensitive. (The host residual is legitimately limited to *presenting the cue/turn* and
   *reading the cleanup argmax/margin off the membrane* — cognition stays spikes.)
2. **The no-confab MOAT holds-or-strengthens at EVERY DA level (the critical risk control).** The named
   danger — "salience-gating recall could lower the abstention threshold and cause confabulation on
   unstored cues" — is structurally avoided in Option A (DA only *raises* `g`), and the probe must
   EMPIRICALLY show zero new false-accepts at DA_high. (If a future variant instead *lowered* the gate
   on salience, that variant would be REJECTED on this control.)
3. **Lesion / permuted controls that collapse.** Lesion the DA source → modulation gone; permute the
   salient-label → the decisiveness advantage does not track the shuffle.
4. **DA-OFF == byte-identical baseline.** With DA at baseline (no salient drive), `g_eff = g0`, so the
   composer behaves exactly as the chosen `confidence_gate` default (0.0 ⇒ the byte-identical current
   composer). No always-on perturbation.
5. **Honest-negative framing.** A null recall effect (a) is reported as "salience-gain has no headroom in
   the routine-recall regime → the hook lives in encoding (Option B) / the degrading regime," not buried.

---

## 6. Is a `sim/` edit required? (NO for the recommendation; the minimal one for the deep variant)

**For the TOP-RANKED Option A: NO `sim/` edit.** The closure is entirely at the composer-runner layer:
between composer ops, read `bridge.neuromodulator_manager.get_concentration("dopamine")` and set
`composer.confidence_gate` (or the cleanup drive). The DA signal is already produced on the merged bridge
by the spiking SNc (`nav_conv_merged_bridge.py:757-761`); only the *consumer* (the composer runner) is
new, and it is host glue that reads a spiking-derived scalar and sets a host knob — no protected
dynamics change. **For Option C (dlPFC gain): also NO new code** — one extra `ModulatorTarget` on the
existing `dopamine` config (the NM subsystem already applies `excitability_drive`/`synaptic_gain` to
normal regions inside `_run_one_simulation_step`).

**The deep variant (DA changes the RF resonate dynamics continuously) DOES need a minimal `sim/` edit** —
documented here for completeness, but **deferred** below the host-layer de-risk.

---

## 7. The MINIMAL `sim/` edit IF the deep RF-dynamics variant is later pursued (byte-level sketch)

If, after Option A, a *continuous* DA modulation of the resonate itself is wanted (DA changes the
decay/sharpness mid-resonate, not just the post-hoc cleanup margin), the RF dynamics read **one scalar**:
`self._rf_lambda` (the per-step decay, used in BOTH the loop and the megakernel). The minimal, additive,
default-preserving route is a **DA-gain multiplier on the RF decay (or on the cleanup matched-filter
drive)**, threaded so that `None`/unset ⇒ byte-identical.

- **Exact sites:**
  - `sim/bridge.py:5518` — `_rf_advance_one`: `_rf_decay = float(np.exp(getattr(self, "_rf_lambda",
    -3.0e-4)))`. The decay is the only "gain" knob in the rotate step.
  - `sim/bridge.py:5636` — `_rf_resonate_steps_megakernel`: `decay = cp.float32(np.exp(getattr(self,
    "_rf_lambda", -3.0e-4)))` (the SAME scalar passed to the `rf_megastep` kernel).
- **Byte-level sketch (additive, guarded — does NOT touch the kernel source or the matvec):**
  introduce an optional `self._rf_da_gain` (default `None`). In BOTH sites, after computing the base
  decay, multiply the *synaptic-input contribution* (NOT the rotation — keep the phase exact) by the gain
  — i.e. gain the matvec term `(self.cp_rf_w_re @ _rf_re − …)` by `(1 + g)` when `_rf_da_gain` is set.
  Concretely in `_rf_advance_one` (`:5526-5529`), wrap the `if getattr(self, "cp_rf_w_re", None) is not
  None:` block's additions with a `_g = getattr(self, "_rf_da_gain", None); _gain = 1.0 if _g is None
  else (1.0 + _g)` factor on the matvec terms. For the megakernel, pass `_rf_da_gain` as a new
  `float gain` kernel arg defaulting to a path that, when the host sets `_rf_da_gain=None`, supplies
  `1.0` (the `||`-short-circuit/`use_*` pattern the mask already uses at `:5607,5643` — a `use_gain`
  int flag, `gain=1.0` when 0 ⇒ **byte-identical**). The composer sets `bridge._rf_da_gain = k·(DA −
  DA_baseline)` (clamped ≥ 0) before the cleanup resonate.
- **Why gain the matvec, not the rotation:** the rotation `exp(λ+iω)` carries the *phase* (the FHRR
  information); scaling it would corrupt the phase read-out. The synaptic matvec is the matched-filter
  *score* accumulation — gaining it sharpens the winner's magnitude (the inverted-U sharpening) without
  touching phase. This keeps every existing RF op (bind/unbind/bundle) byte-identical when
  `_rf_da_gain=None`.
- **Controller review hooks:** default `None` ⇒ both paths bit-identical (the `getattr(..., None)`
  guard + the `use_gain==0` kernel short-circuit, mirroring the shipped `_rf_neuron_mask` /
  `use_mask` pattern at `:5537-5547,5607,5642-5644`); a golden test `== the no-gain path` at gain 0,
  plus the masked-megakernel golden already in `tests/test_rf_megakernel.py`. This is the SAME
  additive-default-preserving discipline as the A5-lever-3 masked megakernel edit (CYCLE 185), so the
  precedent + review pattern exist.

**Bottom line on the edit:** **not required for the recommended path.** The `sim/` edit is a *deferred,
optional* deepening (continuous resonate-gain) with a known-minimal, byte-preserving sketch — to be built
only if Option A's discrete cleanup-margin gate proves insufficient.

---

## 8. Catalog + literature anchors (cited)

- **C.32** Two-component DA — Component 1 (detection/salience, 60–90 ms) "amplifies learning rate and
  **downstream sensory gain** on any potentially important event **before identification**"; Component 2
  (value/utility RPE). *The salience component is the conversational gain hook.* (Schultz16-NRN
  pp. 4–11; Schultz16-JNT pp. 681–682.)
- **C.04** Dopamine — mesocortical DA→PFC WM (the same DA population teaches the BG actor AND modulates
  cortex); tonic DA *enables* cortical processing. (Kandel 6e Ch 16.)
- **E.15** Visual attention — top-down gain modulation: attention **multiplies firing rates ~20–30%**,
  **sharpens tuning, reduces noise correlations**, "modeled as a gain field." *The cleanup-sharpening
  anchor.* (Kandel 6e Ch 25.)
- **C.33** PPN → SNc reward driver (the `reward_us` afferent), for the salience source. (Schultz16-JNT
  pp. 684–686.)
- **C.30** Actor-critic — SNc δ broadcast updates BOTH the actor AND (via the same scalar) downstream
  consumers — the structural justification for one DA broadcast reaching two cortical organs.
- **Vijayraghavan, Wang, Birnbaum, Williams & Arnsten 2007**, *Nat Neurosci* 10:376–384 — **"Inverted-U
  dopamine D1 receptor actions on prefrontal neurons engaged in working memory"**: optimal D1
  **"enhances spatial tuning by suppressing responses to nonpreferred directions"** (the WTA-sharpening
  mechanism); excess D1 erodes tuning (the inverted-U ceiling = the moat-safety reason DA must *raise*,
  not blindly maximize, the gate). [DOI](https://doi.org/10.1038/nn1846) (PMID 17277774). *Based on
  articles retrieved from PubMed.*
- **Lisman & Grace 2005**, *Neuron* 46:703–713 — **"The hippocampal-VTA loop: controlling the entry of
  information into long-term memory"**: a **novelty** signal (with salience + goal info) drives VTA DA →
  DA in hippocampus **"produces an enhancement of LTP and learning"** → **regulates entry of information
  into long-term memory.** *The novelty/salience-gated ENCODING anchor (Option B).*
  [DOI](https://doi.org/10.1016/j.neuron.2005.05.002) (PMID 15924857). *Based on articles retrieved from
  PubMed.*
- **Moore, Zhou, Potapenko, Kim & Antic 2010**, *Brain Res* 1370:1–15 — brief phasic DA pulses
  transiently change PFC pyramidal firing (D1 → depression of firing, D2-spillover → +30% excitability),
  dose-dependent inverted-U; confirms phasic DA acts on PFC pyramidal gain on the ~0.5 s / 40 s
  timescales relevant to a per-turn modulation. [DOI](https://doi.org/10.1016/j.brainres.2010.10.111)
  (PMID 21059342). *Based on articles retrieved from PubMed.*

*(Attribution: the four journal references above were retrieved via PubMed; DOIs linked inline per
PubMed's terms.)*

---

## 9. Honest scope / non-claims

- This scopes the **read-side recall hook (Option A)** as the cheap-first move because it is the one with
  a structurally-inverted (stricter) moat risk and zero protected-code change. It does **not** claim the
  encoding hook (Option B) or the deep RF-dynamics edit (§7) — those are ranked follow-ons with their own
  (higher) moat/edit costs.
- The DA signal is **already produced** on the merged bridge by the spiking SNc (roadmap #1/#2 DONE,
  `nav_conv_merged_bridge.py`); this direction is purely about making the **composer consume** it. No new
  limbic organ is invented.
- The "useful effect" GO bar is deliberately two-limbed (decisiveness OR robustness) because the routine
  functional regime may already be saturated — in which case the honest finding re-points to Option B,
  which is itself a result.
- Per `feedback_moat_not_hard_lossy_memory_ok`: the moat is a plus, not a hard gate — but in Option A it
  is FREE to keep (the mechanism can only tighten it), so any moat loss here is a bug, not a trade.

---

## 10. EXACT NEXT

Build the **Option-A cheap-first de-risk** (numpy/CPU): a small composer + a tiny spiking `snc` pool +
the `dopamine` `from_region_firing_signed` modulator; map `DA → confidence_gate` (clamped to only
sharpen); measure recall-decisiveness + the moat at DA_high vs DA_low under matched cleanup noise, with
the lesion + permuted-DA + host-scalar anti-cheats; pre-register the §4 frozen GO bar; report PASS or the
honest "no recall headroom → re-point to encoding-gating (Option B)" negative. On PASS, wire the
composer-runner read into the merged-bridge conversational path (and add the zero-cost Option-C dlPFC
`ModulatorTarget`); reuse-by-import, NO `sim/` edit. Only if A is insufficient, build the §7 minimal
`_rf_da_gain` resonate-gain edit (byte-preserving, default `None`).
