# Tier-2 #6 (limbic → composer) — where the arc left off + the concrete next cheap-first de-risk

**Date:** 2026-06-22
**Type:** read-only deep-research CONTINUATION (NO edits, NO experiments, NO GPU). One findings doc. Branch `main`,
read-only throughout. The scoping for #6 ALREADY EXISTS (do not re-scope) — this doc reads it + the two later docs,
**verifies where the arc actually left off** against source, and presents the **concrete next cheap-first de-risk step**.
**Roadmap item:** Tier-2 #6 — "the one self": let the SHARED spiking dopamine/salience/value (the SAME limbic brain that
drives navigation) MODULATE the conversational composer. Parent: `2026-06-18-full-spikeification-shared-substrate-roadmap.md`
§3 #6 (line 108–111).

---

## 0. Executive summary (the 12-line version)

- **Where it left off — #6 is FAR more done than "scoping."** Both mechanisms by which the shared dopamine reaches the
  composer are BUILT. **Route A (READ-side salience precision gate): PRODUCTION-WIRED on the merged bridge + GO.**
  **Route B (WRITE-side DA-gated encoding strength): de-risked numpy 6/6 GO, AND now wired into BOTH composers**
  (`RFPhasorComposer` + the production `OneBrainComposer`) behind a default-off `encoding_gain_fn`. The shared DA SOURCE
  (`dopamine` over the spiking SNc, = the signed RPE) is co-resident on the merged bridge.
- **The genuine GAP (the one open experiment):** Route B has **never been run on the merged bridge with the REAL shared
  `dopamine` and the REAL read damage** — only with a *probe* DA value + *injected* read noise (numpy, D=64). No
  deployment smoke runner exists (`research/runners/*routeB*` / `*encoding_gain*` deploy → none; only the de-risk).
- **The audit-flagged risk — CONFIRMED from source, but it is DESIGNED-AROUND, not an open hole.** `rf_resonate_steps`
  (`sim/bridge.py:5749`) and `_rf_advance_one` (`:5710`) DO bypass `_run_one_simulation_step` (where `manager.step()`
  lives, `:7040`); the NM subsystem never reaches the RF op path. Both deployed routes consume the DA *concentration*
  at the composer/agent layer (Route A reads it between ops; Route B bakes it into the stored weights at encoding), so
  neither needs the bypassed loop. The prior arc EXPLICITLY rejected the `_rf_lambda`/cleanup-gain route as a moat risk.
- **The cheapest next #6 de-risk:** the **Route-B deployment smoke on the merged bridge** — wire `encoding_gain_fn` to
  the live merged `dopamine`, `hear` a fact at DA-low vs DA-high, show the high-DA fact recalled where the low-DA fact
  abstains under the REAL read damage. **NO `sim/` edit** (a runner-side wire-up; the composer hook already exists).
- **Anti-cheats:** moat 0-FA at both DA levels (HARD gate); DA-lesion (hold DA at baseline for both hears) kills the
  differential; the effect must beat the DA-lesion null AND follow the gain (permuted); the gain VALUE is a spiking
  population's concentration (brain-based-compliant, the existing `dopamine→plasticity_rate` precedent).
- **GO bar:** the within-fact differential holds ≥5/6 seeds (42/43/44/100/101/102) on the merged bridge, DA-lesion null,
  moat intact at both DA levels, regression GREEN (`encoding_gain_fn=None` byte-identical).

---

## 1. WHERE the Tier-2 #6 arc left off (BUILT vs GAP, verified against source)

The #6 work spans four findings, in order. The earliest (`2026-06-19-tier2-limbic-to-composer-scoping.md`) is the
scoping the prompt refers to; **three later docs supersede its "what's built" picture** (the scoping recommended a
de-risk that has since RUN + GONE GO, and the wire-up has since shipped both routes). The current state:

### 1.1 Route A — READ-side DA salience precision gate: **PRODUCTION-WIRED + GO**

`MergedNavConvAgent` (`research/runners/nav_conv_merged_bridge.py`) — the deployed merged agent that owns BOTH the
limbic core AND the composer — gates each conversational read on the shared spiking dopamine.

- **Verified code:** the flag `enable_da_salience_gate=False` (+ knobs `da_gate_g0=0.06`, `da_gate_k=2.0`,
  `da_gate_cap=0.25`) at `nav_conv_merged_bridge.py:1305`; the three helpers `_da_confidence_gate` (`:1458-1474`),
  `_gated_out` (`:1476-1498`), `_role_cleanup_scores` (`:1500-1508`); the gate applied at the four read ops
  `what_does`/`who_does`/`describe`/`is_it_true` (`:1525`, `:1531`, `:1537`, `:1545`).
- **Mechanism (verified `:1458-1498`):** before a read, `_da_confidence_gate` reads `nm.get_concentration("dopamine")`
  (`:1470`) off the agent's OWN merged bridge, maps it via the de-risk's `da_to_gate` (imported verbatim, `:1464`):
  `g_eff = clip(g0, g_cap, g0 + k·(DA − DA_baseline))`. `_gated_out` short-circuits to `False` when `g_eff <= g0+1e-12`
  (`:1485-1486`, the no-modulation floor ⇒ **byte-identical read path**); otherwise it abstains on a noise-dominated
  cue read (`min(margin(agent), margin(action)) < g_eff`, `:1497`, the EXACT `OneBrainComposer._margin`). **A higher
  gate can ONLY tighten abstention ⇒ moat-safe by construction.**
- **Validation:** `2026-06-18-DA-salience-gate-production-wireup-GO.md` — GPU smoke PASS: default-OFF byte-identity;
  ON + co-resident limbic, the gate rises `0.060 → 0.250` monotone with spiking DA `0.500 → 0.843`, **0 false-accepts at
  BOTH DA levels**, clean recall preserved; regression GREEN (`tests/test_nav_conv_merged_agent.py` 8/8 +
  `tests/test_nav_conv_step2b_coresident.py` 7/7); NO `sim/` edit (+92/−4 in one runner). Underlying 6-seed de-risk:
  `2026-06-18-DA-composer-precision-derisk-GO.md`.

### 1.2 Route B — WRITE-side DA-gated encoding strength: **de-risked numpy 6/6 GO + wired into BOTH composers**

The fact's stored complex-phasor magnitude is scaled by a DA-driven encoding gain at store time (Lisman-Grace
hippocampal-VTA loop / Kandel D.16: DA makes a memory trace stable vs degradable).

- **Verified code (RF composer):** `RFPhasorComposer.__init__(..., encoding_gain_fn=None)` at `rf_phasor_composer.py`,
  applied in `_store_substrate`: `g = 1.0 if self.encoding_gain_fn is None else float(self.encoding_gain_fn())`, then
  `conns = [(1+k, 0, complex(g) * zc[k]) ...]`. (`store` → `_store_substrate` when `enable_substrate_store=True`,
  `rf_phasor_composer.py:448`.)
- **Verified code (production OneBrain composer):** `OneBrainComposer.__init__(..., encoding_gain_fn=None)`
  (`one_brain_composer.py:116`, stored `:175`); `_write_block` (`:342`) applies the IDENTICAL gain:
  `g = 1.0 if self.encoding_gain_fn is None else float(self.encoding_gain_fn())` (`:351`),
  `block_conns = [(trig + 1 + k, trig, complex(g) * zc[k]) for k in range(D)]` (`:352`). `complex(1.0)*zc == complex(zc)`
  exactly for finite IEEE ⇒ the `None` default is GENUINELY byte-identical.
- **Why it is DIFFERENTIAL (not a vacuous global scalar):** the RF phase read-out has a hard **magnitude floor**
  (`sim/bridge.py:5731` in the loop, `:5804` in the megakernel: `(_rf_mag2 > _rf_floor2)` gates the up-crossing). A
  readout neuron whose `|Z|²` decays below `_rf_floor²` never spikes → reads phase 0 (garbage) → contributes nothing to
  the cleanup matched filter. So under shared read damage, a rewarded (g>1) fact's readout neurons stay above the floor
  where a neutral (g=1) fact's drop below ⇒ the rewarded fact wins the cue-match scan. The floor × noise interaction is
  the nonlinearity.
- **Validation (numpy de-risk):** `2026-06-19-dopamine-encoding-gain-derisk.md` — **GO, 6 seeds** (42/43/44/100/101/102):
  WITHIN-FACT gain lift +6/12 (neutral g1 4/6→g2 6/6; rewarded g1 2/6→g2 6/6); DA-lesion null (both g=1) kills the
  differential; permuted (gain swapped) flips the advantage to the other fact; monotonic in g; **moat 6/6 in EVERY
  condition** incl. g=2; regression byte-identical. Commits `b4ae63b0` (RF composer knobs) + `5928465f` (runner).
- **OneBrain wire-up:** `2026-06-20-tier2-6-routeB-onebrain-wireup.md` — the gain mirrored into `OneBrainComposer`
  (+2 CI tests: default-OFF byte-identity + `g=1.5` lifts recall with the moat 0-FA). NO `sim/` edit.

### 1.3 The shared DA SOURCE the composer reads — co-resident, verified

- **`dopamine` is registered over the spiking SNc via `from_region_firing_signed`** (the SIGNED RPE production rule,
  `sim/neuromodulators.py:774-817`: EMA of SNc firing − tonic, no `max(0,·)` clamp → burst→conc>baseline (LTP), tonic→0,
  dip→conc<baseline (LTD)). So `get_concentration("dopamine")` (`neuromodulators.py:228`) **IS** the spiking RPE.
- **On the merged bridge:** registered by `co_resident_limbic` / `co_resident_nav_critic` / `co_resident_td_cueshift`
  (the limbic/critic/TD slice), threshold 0.0 ⇒ neutral-at-rest. The limbic core lift itself is GO-structural
  (`2026-06-18-merged-limbic-core-lift.md`: 46 regions, nav-inert, default-off byte-preserved; δ=r−V confirmed
  co-resident). **ONE characterized boundary** (honest, does NOT block #6): the FULL multi-gate RPE *arithmetic* (burst
  ratio ≥3× AND the GABA_B value-subtraction together) does not hold on the het-off merged config — the SNc's effective
  synaptic response is ~6–10× weaker in the full network. **This does not block either composer route** — both consume
  the DA *concentration*, which is present + readable (Route A's GPU smoke already drove it to two operating points on
  the merged `limbic_snc`). The boundary is for the critic's *internal arithmetic*, not the DA broadcast the composer reads.

### 1.4 The PRECISE remaining GAP

> **The only thing #6 has NOT done: prove Route B works on the MERGED BRIDGE with the REAL shared `dopamine` (not a probe
> scalar) and the REAL superposition/read damage (not injected numpy noise).** Route A is fully production-validated;
> Route B is validated only in the isolated numpy de-risk (D=64, 2 facts, probe DA, injected σ=260). The composer-side
> hook exists in BOTH composers; the merged-bridge DA source exists; what is missing is the **runner-side wire-up + the
> 6-seed GPU deployment smoke** that closes the loop end-to-end. No such runner exists in the tree.

---

## 2. The AUDIT-FLAGGED RISK — verified against source (CONFIRMED, but designed-around)

**The claim under test (from the prompt + roadmap §3 #6 line 111):** the composer's RF op path bypasses
`_run_one_simulation_step`, so the neuromodulator subsystem (`compute_synaptic_gain_multiplier` /
`compute_plasticity_rate_multiplier` / `manager.step()`) does NOT reach the composer's complex-synapse ops.

**Verdict: CONFIRMED — the bypass is real (verified line-by-line):**

- **`rf_resonate_steps` (`sim/bridge.py:5749`)** — its own docstring (`:5750-5754`): *"Run `n_steps` of the RF resonate
  dynamics DIRECTLY (the production-fast path) — skips the full `_run_one_simulation_step` machinery (conductance /
  plasticity / recording / engram / gate couplings / stats)…"* Body: `for _ in range(int(n_steps)): self._rf_advance_one()`
  (`:5768-5769`), OR the megakernel `_rf_resonate_steps_megakernel` (`:5814`) when `cfg.enable_rf_cudagraph` is on.
- **`_rf_advance_one` (`sim/bridge.py:5710`)** — uses `_rf_lambda`/`_rf_omega`/`_rf_floor` directly (`:5716-5718`); the
  complex matvec is `cp_rf_w_re/im @ z` (`:5726-5727`), array-disjoint from `cp_connections`. **No `manager.step`, no
  `compute_*_multiplier`, no NM read anywhere in this method.**
- **`manager.step(self)` is called ONLY inside `_run_one_simulation_step`** (`sim/bridge.py:7040`); the synaptic-gain /
  plasticity-rate / plasticity-gate multipliers are likewise applied only on that path. A whole-file search of
  `sim/neuromodulators.py` for `rf|phasor|resonate|composer|cleanup` returns **ZERO** matches — the NM subsystem has no
  awareness of the RF composer.
- **One nuance:** `compute_excitability_drive_per_neuron` DOES write `total_input_current_pA`, and RF reuses
  `cp_membrane_potential_v`/`cp_recovery_variable_u` — but ONLY on the `_run_one_simulation_step` path, NOT the bypassed
  `rf_resonate_steps` loop the composer uses. So even that does not reach the composer's ops in production.

**⇒ The risk is REAL as a fact about the op path, but it is DESIGNED-AROUND, not an open hole.** A *live, per-resonate-
step* NM coupling into the RF dynamics does not exist — and the prior arc deliberately did NOT build one. Routing a live
DA concentration into `_rf_lambda` (the audit's first guess) would (a) require threading NM state into the bypassed fast
loop (a `sim/` edit) AND (b) modulate *every* op including the unbind/cleanup of **already-stored** facts — a global gain
whose functional consequence is muddy and which **risks the moat**. Both deployed routes reach the composer functionally
**without** touching the bypassed loop: **Route A reads the concentration at the agent layer between ops; Route B bakes
the gain into the stored weights at encoding.** Both are `sim/`-edit-free at the deployed level. The `2026-06-19` scoping
(§2.4) and the `2026-06-20` deep-research (§1.2) both explicitly reject `_rf_lambda` for exactly these reasons.

---

## 3. The cheapest-first de-risk for #6 (the ONE open experiment)

Because the composer-side hook is ALREADY wired into both composers and the merged DA source ALREADY exists, the only
open experiment is **the Route-B deployment smoke on the merged bridge** — proving the WRITE-side DA gain works with the
REAL shared `dopamine` (not a probe) and the REAL read damage (not injected). This is the §4 plan from the `2026-06-20`
deep-research, not yet run.

### 3.1 The single load-bearing question

*Does a fact heard while the shared spiking SNc is bursting (high DA, so the encoding gain `g>1`) get recalled at a
read-damage level where a fact heard at DA baseline (`g≈1`) abstains/mis-recalls — driven by the REAL merged `dopamine`,
with the no-confab moat intact?*

### 3.2 The wire-up (NO `sim/` edit — runner-side; the composer hook exists)

The two-rung structure keeps it cheapest-first:

- **Rung 1 (CPU/numpy, isolate the deploy plumbing — seconds/seed):** build a `MergedNavConvAgent` with the composer's
  `encoding_gain_fn` wired to the merged `dopamine`:
  ```python
  b = agent._merged_bridge
  g_fn = lambda: float(np.clip(1.0 + k_DA * (b.neuromodulator_manager.get_concentration("dopamine") - 0.5), 0.5, 3.0))
  ```
  Pass `g_fn` into the composer (the OneBrain default `_write_block` hook or the `RFPhasorComposer`/`MergedRFComposer`
  `_store_substrate` hook — whichever the deployed merged path uses). To isolate "does the deploy plumbing read DA
  correctly," drive the gain by `manager.set_concentration("dopamine", ·)` (`neuromodulators.py:231`) directly — two
  levels (baseline 0.5 and a salient burst ~0.85). Confirm `g` differs between the two `hear`s and the high-DA fact's
  stored block edges have larger `|w|`.
- **Rung 2 (GPU, `SIM_BACKEND=cupy`, the real claim):** `MergedNavConvAgent(co_resident_limbic=True)` (or
  `co_resident_nav_critic`); drive the shared `limbic_snc` to two operating points (the salience-gate smoke's recipe:
  tonic 80 pA → DA≈0.50; salient 600 pA → DA≈0.84). `hear` a fact at each DA level (so the encoding gain differs by the
  REAL spiking DA), then query both after a common read at the merged-bridge's real superposition/read damage.

### 3.3 The decisive metric

The fact heard at DA-high is recalled correctly at a read-damage level where the fact heard at DA-low abstains/mis-recalls
— the same WITHIN-FACT / matched-cue differential the numpy de-risk showed, now driven by the spiking SNc on the merged
bridge. Quantify with the composer's `_margin` (`OneBrainComposer._margin`): the high-DA block's cue-role margin stays
above the recall threshold; the low-DA block's collapses at the same damage.

### 3.4 Honest caveat to re-confirm in deployment

The numpy de-risk used a probe DA and *injected* read noise (σ=260) to reach the graceful-degradation knee. **In
deployment the damage is the REAL superposition/noise load** — the smoke must verify the real merged-bridge read damage
**sits at/below the moat-safe knee** (σ≈260 for D=64/2 facts in the de-risk), or report the boundary honestly (per the
HARD-gate rule, §4). If the real damage is *below* the knee (facts reconstruct cleanly even at g=1), the differential
will be small and the honest finding is "the deploy read damage is too gentle to exercise the floor — the gain is
correct but behaviorally latent at this load," which points at the emergent follow-ons (§5) rather than a NEGATIVE.

---

## 4. The anti-cheat controls (the moat is the HARD constraint)

| Control | What it rules out | Expected |
|---|---|---|
| **No-confab MOAT (HARD gate)** | "DA modulation manufactured a false-accept" | an UNSTORED cue (`query_patient("river","run")`) returns `None` at BOTH DA levels, EVERY seed. **Structural:** the gain scales an *already-stored* fact's magnitude; an unstored cue has no block to amplify ⇒ no fact to confabulate. Any breach at any DA level = NEGATIVE, not a tunable. |
| **DA-LESION (hold DA at baseline for both hears)** | "the effect is fact content/order, not DA" | both facts `g≈1` ⇒ the within-fact differential VANISHES (the de-risk's decisive control). |
| **Beats the lesion null** | "the differential is just content asymmetry" | the within-fact gain lift (g-high − g-low on the SAME fact) must exceed the lesion's content-only between-fact gap. |
| **PERMUTED gain** (high DA on the OTHER fact's hear) | "one fact is intrinsically more robust" | the advantage FOLLOWS the DA-gated fact, not the content. |
| **Brain-based-only standard** | "the modulation is a host scalar, not a spiking population's effect" | the gain VALUE = `get_concentration("dopamine")` = the SIGNED RPE produced by the spiking SNc (`from_region_firing_signed`, `neuromodulators.py:774`); what it scales is synaptic weight magnitude at encoding (the DA-gated-LTP analogue). SAME status as the shipped nav `dopamine→plasticity_rate` precedent. Be explicit (per the de-risk §5): the gain is APPLIED by a host multiply at the composer layer; the fully-spiking ideal (gain EMERGING from live SNc firing *during* an on-bridge store op, via the deferred `rf_kick(kick_gain=...)` `sim/` edit) is the follow-on, NOT claimed. |
| **Regression GREEN** (default-OFF byte-identity) | "the default path drifted" | `encoding_gain_fn=None` ⇒ `tests/test_one_brain_composer_agent.py` (15) + `tests/test_brain_conversational_agent.py` + the merged-agent suites (`tests/test_nav_conv_merged_agent.py` 8/8 + `tests/test_nav_conv_step2b_coresident.py` 7/7) pass VERBATIM. |

---

## 5. The GO / NEGATIVE bar (multi-seed, pre-registered)

- **GO:** on the MERGED bridge with the REAL shared `dopamine`, the fact heard at DA-high is recalled where the fact
  heard at DA-low abstains/mis-recalls, on **≥5/6 seeds** (42/43/44/100/101/102, the standing multi-seed rule — the
  cleanup/read is noise-sensitive, a distribution not an exact identity); the **DA-lesion kills the differential**
  (every seed); the **moat is intact (0 false-accepts) at BOTH DA levels** (every seed); the effect **follows the gain**
  (permuted); **regression GREEN**. ⇒ the shared spiking dopamine demonstrably modulates a conversational-composer
  operation (fact encoding strength) end-to-end on the one brain, both halves driven by the same limbic core.
- **NEGATIVE (an honest deliverable):** EITHER the real merged-bridge read damage is too gentle to exercise the RF
  floor (the gain is correct but behaviorally latent at the deployed load — report the σ-knee mismatch, point at the
  emergent follow-ons §5.1), OR the differential cannot be obtained without the read damage itself breaching the moat on
  some seed (per the HARD gate, that is a NEGATIVE not a tunable). Either is a real finding mapping what DA-gated
  encoding can/can't do on the deployed RF substrate.

### 5.1 What a GO unlocks (note, NOT scope)

- **online salience-gated recall** during a live conversational turn (the persistent integrated loop, a *different*
  Tier-2 item — its Phases A+B are GO, Phase C designed: `2026-06-19-tier2-phaseC-integrated-loop-design.md`);
- **tonic-DA mood/state-dependence** biasing what is encoded/recalled across a session;
- **DA-gated reconsolidation:** modulate the labilization threshold in the composer's `update_on_mismatch` /
  `_calibrate_pe_labile` (`rf_phasor_composer.py:466`; `one_brain_composer.py:605`) so a *surprising* correction is
  encoded more strongly — the natural pairing of the existing prediction-error-gated rewrite with the limbic salience.

---

## 6. Summary recommendation (one line)

**#6's mechanisms are BUILT** (Route A production-wired + GO; Route B de-risked numpy 6/6 + wired into both composers);
the **one open experiment** is the **Route-B deployment smoke on the merged bridge** — wire `encoding_gain_fn` to the
live merged `dopamine` (a runner-side wire-up, **NO `sim/` edit**, the composer hook already exists), `hear` a fact at
DA-low vs DA-high, and show the high-DA fact recalled where the low-DA fact abstains under the REAL read damage, ≥5/6
seeds, DA-lesion null, moat 0-FA at both DA levels, regression GREEN. The deferred `rf_kick(kick_gain=...)` /
`rf_set_complex_weights(weight_gain=...)` `sim/` edit (additive, default-`1.0`, byte-identical when absent) is the
*later* fully-on-substrate refinement (the gain emerging from live SNc firing during an on-bridge store op), **NOT
required** for this de-risk; the audit-flagged `_rf_lambda`/cleanup-gain route is explicitly the WRONG mechanism (a moat
risk, rejected twice).

---

## 7. Key file:symbol references (verified this session)

- **Route A (read-side gate, production-wired):** `nav_conv_merged_bridge.py:1305` (flag + knobs), `:1458-1474`
  (`_da_confidence_gate`), `:1476-1498` (`_gated_out`, floor short-circuit `:1485-1486`, the `_margin` gate `:1497`),
  `:1500-1508` (`_role_cleanup_scores`), `:1525/1531/1537/1545` (the four read ops). Validation:
  `2026-06-18-DA-salience-gate-production-wireup-GO.md`, `2026-06-18-DA-composer-precision-derisk-GO.md`.
- **Route B (write-side encoding gain, both composers):** `one_brain_composer.py:116` (param), `:175` (stored), `:342`
  (`_write_block`), `:351-352` (the gain `complex(g)*zc[k]`); `rf_phasor_composer.py` `encoding_gain_fn` +
  `_store_substrate` (`store` at `:448`). Validation: `2026-06-19-dopamine-encoding-gain-derisk.md` (numpy 6/6 GO);
  wire-up `2026-06-20-tier2-6-routeB-onebrain-wireup.md`. De-risk runner: `_phaseB_dopamine_encoding_gain_derisk.py`.
- **The audit-flagged bypass (CONFIRMED):** `rf_resonate_steps` (`sim/bridge.py:5749`, docstring `:5750-5754`, loop
  `:5768`), `_rf_advance_one` (`:5710`, `_rf_lambda` use `:5716`, matvec `:5726-5727`),
  `_rf_resonate_steps_megakernel` (`:5814`); `manager.step` ONLY at `:7040` (inside `_run_one_simulation_step` `:5857`).
  The magnitude floor: `:5731` (loop) / `:5804` (megakernel) `(_rf_mag2 > _rf_floor2)`.
- **DA source / NM subsystem:** `from_region_firing_signed` (`sim/neuromodulators.py:774-817`, signed RPE),
  `get_concentration` (`:228`), `set_concentration` (`:231`). Merged registration: `co_resident_limbic` /
  `co_resident_nav_critic` / `co_resident_td_cueshift` on `nav_conv_merged_bridge.py` (the limbic lift,
  `2026-06-18-merged-limbic-core-lift.md`).
- **Scoping lineage:** `2026-06-19-tier2-limbic-to-composer-scoping.md` (the original scoping; recommended the de-risk
  now RUN+GO), `2026-06-20-tier2-limbic-to-composer-deep-research.md` (the re-scope gate: route built two ways; §4 = the
  deploy smoke this doc operationalizes), `2026-06-20-tier2-6-routeB-onebrain-wireup.md` (Route B into `OneBrainComposer`).
- **Roadmap parent:** `2026-06-18-full-spikeification-shared-substrate-roadmap.md` §3 #6 (lines 108–111; line 111
  anticipated the `_rf_lambda`/cleanup-gain `sim/` edit — the very route the later arc rejected) + §3 #1 (the limbic
  core, the deployment dependency).

---

## 8. Discipline confirmation

- **READ-ONLY:** no code edited, no experiments run, no GPU. Stayed on branch `main` throughout.
- **Trust-but-verify:** every load-bearing claim cited to actual source read this session — `nav_conv_merged_bridge.py`
  (`:1305/1458-1508/1525-1546`), `one_brain_composer.py` (`:116/175/342/351-352`), `rf_phasor_composer.py` (`:448`,
  `_store_substrate` gain), `sim/bridge.py` (`:5710/5731/5749/5768/5804/5814/7040`), `sim/neuromodulators.py`
  (`:228/231/774-817`). The four #6 findings' verdicts read in full (not from headlines); the audit-flagged bypass
  verified line-by-line; the "no deploy runner exists" gap verified by glob (`*routeB*`/`*encoding_gain*` deploy → none).
- **The deliverable** is the CONCRETE NEXT STEP (the Route-B merged-bridge deployment smoke), not a re-scope — the
  mechanisms are built; this is the one open experiment that closes #6 end-to-end, with its `sim/`-edit scope (NONE for
  the de-risk), anti-cheats, and GO bar pre-registered.
