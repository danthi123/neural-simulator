# Tier 2 #6 scoping — wire the shared limbic (dopamine / neuromodulator) state INTO the conversational composer

**Date:** 2026-06-19
**Type:** deep-research + catalog-review scoping (READ-ONLY; no build, no runs). The standing opening move for the
TRUE ONE BRAIN line (owner top directive `feedback_move_everything_to_shared_spiking_substrate`).
**Roadmap item:** Tier 2 #6 — the deepest "one self": let the SAME motivational/limbic brain that drives
navigation actually REACH the conversational composer, so dopamine/neuromodulator state modulates BOTH halves.
This is item **#6** of `2026-06-18-full-spikeification-shared-substrate-roadmap.md` ("Wire the shared DA/NM onto
the conversational composer — limbic ↔ cortex closure"), the one item that body flags as "probably touches
protected code (a new `_rf_lambda`/cleanup-gain route)".

---

## 0. Executive summary

- **The functional target (the crux, recommended):** *dopamine gates fact ENCODING STRENGTH at `store`/bind
  time.* A rewarded / salient fact is written into the persistent fact-store complex weights with a larger
  magnitude; a neutral fact is written at unit magnitude. Because the resonate-and-fire (RF) phase read-out has a
  hard **magnitude floor** (`_rf_floor`: a phasor whose magnitude decays below the floor never spikes → its phase
  reads as garbage), a stronger-encoded fact reconstructs reliably and **wins the cue-match scan where a neutral
  fact (same cue strength) degrades**. This is a real functional consequence — not a global gain with no effect —
  and it is exactly the **Lisman-Grace hippocampal-VTA loop** biology (dopamine gates the *entry of information
  into long-term memory*; Kandel catalog **D.16**: D1/D5 dopamine + attention is what makes a memory trace
  *stable* rather than degradable).
- **The RF-op entry point (where the DA signal enters):** the composer's store path —
  `OneBrainComposer._write_block` (the persistent on-bridge fact store) and `RFPhasorComposer._store_substrate`
  (the standalone fact store) — multiplies the stored composite phasor `zc[k]` by an **encoding gain `g`** before
  building the trigger→readout complex weights (`complex(g * zc[k])`). `g = 1 + k_DA·(DA_conc − DA_baseline)`,
  read from the shared `dopamine` neuromodulator at store time. **No per-step coupling is needed** — the salience
  is *baked into the weights at encoding*, which is both the cleanest engineering and the most faithful biology
  (LTP-magnitude gating, not a runtime modulation of the read).
- **Why NOT `_rf_lambda` / a per-step cleanup-gain (the audit's first guess):** those are the *runtime* knobs of
  the RF dynamics, and the composer deliberately runs each op with `lam=0.0` (no decay) inside the fast
  `rf_resonate_steps` loop that *bypasses* `_run_one_simulation_step` (where the neuromodulator subsystem applies
  its effects). Routing a live NM concentration into `_rf_lambda` per op would (a) require threading NM state into
  the bypassed fast loop (the flagged sim/ edit) AND (b) modulate *every* op (including unbind/cleanup of
  *already-stored* facts) — a global gain whose functional consequence is muddy and which risks the moat. The
  **store-time encoding gain is strictly better**: it is a single multiply at write time, touches only the store
  path (composer-layer, **NO `sim/` edit at all** for the recommended first target), has a clear functional
  readout, and matches the encoding-gates-LTP biology precisely.
- **The cheapest-first de-risk:** numpy/CPU. Store a NEUTRAL fact and a REWARDED fact (same cue strength), then
  damage both equally (noise on the read) and show the rewarded fact is recalled where the neutral one abstains —
  with the DA signal **load-bearing** (lesion DA → no differential), the **no-confab moat intact** (modulation
  never makes it confabulate an unstored fact), multi-seed. GO bar + controls in §3.
- **The flagged sim/ edit — only if/when the store moves fully on-substrate:** if a later phase wants the
  encoding gain itself to be *driven by live SNc firing during an on-bridge store op* (rather than read as a
  scalar at the composer layer), THEN the minimal `sim/` edit is an **encoding-gain argument on `rf_kick` /
  `rf_set_complex_weights`** (a `kick_gain` / `weight_gain` scalar, additive, default `1.0` = byte-identical),
  NOT a `_rf_lambda` route. Scoped in §2.4 for the owner's byte-review, but **NOT required for the de-risk or the
  first functional target.**

---

## 1. The mechanism map (the crux)

### 1.1 What the composer's `store` actually does (verified file:symbol)

The production conversational composer is `OneBrainComposer` (`research/runners/one_brain_composer.py`), with
`RFPhasorComposer` (`research/runners/rf_phasor_composer.py`) as its inner engine + the numpy test oracle.

**Store path (OneBrainComposer), verified:**
- `hear()` / `store()` → `_store_fact` → `_store_composite(fillers, roles)` (`one_brain_composer.py:264`).
- `_store_composite` → `_compose_phases(fillers, roles)` (binds each role-filler through a diagonal complex
  synapse, bundles → the composite phasor PHASES) → `_write_block(i, zc)` (`one_brain_composer.py:251`).
- `_write_block` builds the persistent trigger→readout weights:
  ```python
  trig = self.store_base + i * self.block
  block_conns = [(trig + 1 + k, trig, complex(zc[k])) for k in range(D)]   # ← THE ENTRY POINT
  ```
  i.e. fact `i` is a `(1+D)` block whose trigger neuron, when fired, reconstructs the composite phasor on its `D`
  readout neurons *in phase*. `zc = exp(2πi·phase)` so **|zc[k]| = 1** today — every fact is encoded at unit
  magnitude. The block weights live in `cp_rf_w_re` / `cp_rf_w_im` (the complex synapse arrays), array-disjoint
  from `cp_connections`.

**Recall path (verified):** `_read_all_blocks` / `_read_block` fires a stored trigger (`kick[trig] = 1.0`),
resonates (`rf_resonate_steps`), unbinds all roles in parallel, cleans up (matched filter on the codebook), and
the **first block whose cue roles (agent+action) match answers** — `query_patient` / `query_agent` /
`ask_yes_no`. The no-confab moat: an unstored cue matches no block → returns `None`/`"unknown"`.

**Standalone (RFPhasorComposer):** the equivalent persistent store is `_store_substrate(comp_phases)`
(`rf_phasor_composer.py:406`), which builds a `(1+D)` trigger→readout bridge with `conns = [(1+k, 0, zc[k]) ...]`
— same unit-magnitude write. (The default fast path holds the composite as a numpy array in `self.kb`; the
on-substrate store is the `enable_substrate_store=True` path. The functional target applies to the on-substrate
store; the numpy-array path would scale the array magnitude identically.)

### 1.2 The magnitude floor — why an encoding gain has a real functional consequence

The RF read-out is **magnitude-invariant in PHASE but magnitude-GATED in whether a spike happens at all.** In
`_rf_advance_one` (`sim/bridge.py:5568`) and the megakernel (`sim/bridge.py:5640`):
```python
_rf_mag2 = _rf_re_new**2 + _rf_im_new**2
_rf_crossed = (~fired) & (prev_im < 0) & (im_new >= 0) & (_rf_mag2 > _rf_floor2)   # ← floor gate
```
A neuron whose complex magnitude `|Z|²` has decayed below `_rf_floor²` **never registers an up-crossing → never
spikes → `spike_step` stays at the default `period` → `rf_read_phases` returns phase 0** (garbage), and that
readout neuron contributes nothing to the cleanup matched filter. So:

- **Unit-magnitude fact + read noise / competing superposition** → some readout neurons drop below the floor →
  the recovered phasor is partially garbled → the cleanup margin shrinks → under enough damage the cue-match
  scan misses (the documented graceful-degradation boundary, `2026-06-18-emergent-graceful-degradation-derisk.md`).
- **Higher-magnitude (rewarded) fact** → the readout neurons stay comfortably above the floor under the same
  damage → reconstructs reliably → the cleanup margin stays high → it wins the scan.

This is the load-bearing point: an **encoding gain is not a global no-op scalar** (which would scale every fact
equally and change nothing relative), it differentially protects the *salient* fact's reconstruction against the
*shared* read noise floor — a genuine "rewarded facts are recalled when neutral ones aren't" effect. (The
`confidence_gate` margin already computed in `_decode_batched_mem`/`_margin` is the natural read-out of this
"how well did this block reconstruct".)

### 1.3 The DA source to read (verified)

The shared limbic dopamine state is the `dopamine` neuromodulator concentration, **driven by the spiking SNc
pool's firing** via the `from_region_firing_signed` production rule (`sim/neuromodulators.py:774`). In the nav
runner this is registered when `spiking_snc=True` (`g11_bg_runner.py:4304`):
```python
NeuromodulatorConfig(name="dopamine", baseline=0.5, decay_tau_ms=200.0, ...,
    production_rules=[ProductionRule(rule_type="from_region_firing_signed",
        sensitivity=snc_da_sensitivity, threshold=snc_tonic_firing_fraction, source_regions=["snc"])])
```
So the DA concentration **is** the spiking reward-prediction error (SNc firing − tonic = δ). The
just-built multi-cue learner (`2026-06-19-multicue-learning-firm-and-neural-reward.md`, Part 2) confirms a
**standalone spiking-SNc DA RPE pool** (a `snc` region, `IZH2007_DOPAMINE`, `n_snc=40`, reusing the nav pattern)
whose firing tracks the signed graded RPE 6/6 seeds — a ready, co-residable DA *source* independent of the
gridworld.

**The read at store time:** `g = 1 + k_DA · (DA_conc − DA_baseline)`, where
`DA_conc = bridge.neuromodulator_manager.get_concentration("dopamine")` (the same call the bridge already makes at
`bridge.py:6894`). A rewarded fact (SNc bursting, DA above baseline) → `g > 1` (encoded stronger); a neutral fact
(DA at baseline) → `g = 1`; an aversive/omission (DA dip) → `g < 1` (encoded weaker / more forgettable). Clip `g`
to a sane band (e.g. `[0.5, 3.0]`) to stay in the floor-relevant magnitude regime.

### 1.4 The biology (concretely, not "dopamine modulates it")

- **Lisman & Grace 2005 (Neuron), "The Hippocampal-VTA Loop: Controlling the Entry of Information into Long-Term
  Memory."** The hippocampus detects novelty → a novelty/salience signal (via subiculum→accumbens→VP→VTA) →
  VTA dopamine release back into the hippocampus → **enhancement of LTP** → the information is gated into
  long-term storage. The loop's *function* is to decide *which* information gets the strong-encoding (long-term)
  treatment. That is precisely "DA at encoding scales how strongly the fact is written."
- **Lemon & Manahan-Vaughan 2006 (J Neurosci 26:7723), "Dopamine D1/D5 Receptors Gate the Acquisition of Novel
  Information through Hippocampal LTP and LTD."** D1/D5 activation lowers the threshold for inducing both LTP and
  LTD for novel information — dopamine literally sets *whether* a synaptic change of meaningful magnitude happens.
- **Kandel catalog D.16 (Place-field stability requires attention + D1/D5 dopamine + late-LTP)**, Kandel 6e Ch 54
  pp 1366–1367: inattentive/un-rewarded encoding → traces form but **degrade in 3–6 hours**; attended/rewarded
  (D1-gated) encoding → **stable for days**; a PKA-inhibitor (blocking late-LTP) mimics the inattentive
  phenotype. The composer's magnitude floor is the direct analogue of "trace degrades below a usable level": a
  unit-magnitude fact is the degradable trace, a DA-boosted fact is the stable (late-LTP) trace.
- **Catalog Schultz16 component-1 salience** (feature-catalog.md ~line 702): the DA signal is graded by intensity,
  reward context, generalization, and **novelty** — so reading DA at store time gives a *salience-graded*
  encoding gain, not a binary reward flag.

So the chain is exact: **shared spiking SNc firing → `dopamine` concentration (signed RPE) → store-time encoding
gain `g` on the fact's complex weights → the salient fact reconstructs above the RF floor where a neutral one
degrades → it is preferentially recalled.** Same motivational brain, both halves; the limbic core reaches the
cortex.

---

## 2. Reuse-vs-new

### 2.1 What transfers unchanged (reuse-by-import)

| Component | File:symbol | Role in #6 |
|---|---|---|
| The DA **source** | `g11_bg_runner.py:4304` (`spiking_snc` → `dopamine` via `from_region_firing_signed` over `snc`); standalone pool in `_phaseB_multicue_spikingRPE_*` per `2026-06-19-multicue-...md` | Provides `DA_conc` (= spiking RPE). The de-risk can use the *standalone* spiking-SNc pool (already validated, co-residable) so it does not depend on the full nav merge. |
| The **NM subsystem** | `sim/neuromodulators.py` (`NeuromodulatorManager.get_concentration`, `from_region_firing_signed`, `step`) | Holds + updates the `dopamine` concentration from SNc firing. **Already correct + reusable** — the roadmap calls it "the correct, reusable hinge for the limbic core." Read it with the existing `get_concentration("dopamine")` (the bridge already does this at `bridge.py:6894`). |
| The **RF ops** | `sim/bridge.py` `rf_kick` / `rf_set_complex_weights` / `rf_resonate_steps` / `_rf_advance_one` / `rf_megastep` | Unchanged for the recommended target — the gain is applied to the *connection weights handed to* `rf_set_complex_weights`, not to the RF dynamics. |
| The composer store + recall + moat | `one_brain_composer.py` `_write_block` / `_store_substrate` / `_read_*`; `rf_phasor_composer.py` `_store_substrate` | The single change is the gain factor inside `_write_block` / `_store_substrate`. Recall + the cue-match scan + abstention are untouched. |
| The **confidence/margin** read-out | `one_brain_composer.py` `_margin` / `_decode_batched_mem` / `confidence_gate` | Already measures "how cleanly did this block reconstruct" — the natural metric for the de-risk (a rewarded block's margin stays high under damage; a neutral one's collapses). |
| Regression gates | `tests/test_one_brain_composer_agent.py` (11 tests, incl. the no-confab moat), `tests/test_rf_megakernel.py`, `tests/test_rf_neuron_mask_coexistence.py`, `tests/test_brain_conversational_agent.py` | The default `g = 1.0` (no DA / subsystem off) must keep these byte-identical. |

### 2.2 The MINIMAL new wiring (recommended first target — composer-layer, NO `sim/` edit)

A single additive, default-off knob on the composer:

- `OneBrainComposer(..., encoding_gain_fn=None)` and `RFPhasorComposer(..., encoding_gain_fn=None)`: an optional
  callable `() -> float` returning the store-time gain `g` (read from the shared `dopamine` concentration). Default
  `None` ⇒ `g = 1.0` ⇒ **byte-identical** to today.
- Inside `_write_block` (OneBrain) and `_store_substrate` (RFPhasor): when set, multiply the composite phasor by
  `g` before building `block_conns` — `complex(g * zc[k])`. (The numpy-array fast path in `RFPhasorComposer.kb`
  would store `g * comp_phasor` analogously, but the *substrate* store is the brain-based target.)
- The agent / runner that owns both the merged bridge and the composer passes
  `encoding_gain_fn = lambda: 1.0 + k_DA * (b.neuromodulator_manager.get_concentration("dopamine") - 0.5)` (clipped).

This is **reuse-by-import only**: no protected `sim/` code changes; the gain is a composer-layer multiply on the
weights handed to the existing `rf_set_complex_weights`. It satisfies the owner's "host code legit only for env +
body" bar at the *recommended* level too, because the gain VALUE is produced by the spiking SNc/`dopamine`
concentration (neural), and the multiply is the synaptic-weight-magnitude analogue of DA-gated LTP — the same
status the existing nav `dopamine→plasticity_rate` modulation already has (a neural concentration scaling a
synaptic change).

### 2.3 Honest framing of the recommended target's brain-based status

The encoding gain `g` is *applied* at the composer layer (a host multiply), but its **value is the spiking SNc's
firing-derived DA concentration**, and what it scales is **synaptic weight magnitude at encoding** — the direct
analogue of dopamine-gated LTP (D.16 / Lisman-Grace). This is the SAME status as the already-shipped nav
`dopamine → plasticity_rate` path (`neuromodulators.py` `compute_plasticity_rate_multiplier`, applied at
`bridge.py:6919`): a neural concentration scales a synaptic change. So it is brain-based-compliant by the project's
existing precedent. The fully-spiking ideal (§2.4) is the *encoding gain itself emerging from live SNc firing
during an on-bridge store op*; that is the deeper follow-on, not the first target.

### 2.4 The flagged `sim/` edit — scoped, but DEFERRED (only for the fully-on-substrate phase)

IF a later phase wants the gain to be driven by **live SNc firing during the actual on-bridge store op** (the RF
store runs inside `rf_resonate_steps`, which bypasses `_run_one_simulation_step` where NM lives), the minimal
edit is **NOT** a `_rf_lambda` route. It is an **encoding-gain scalar on the kick / weight install**:

- `rf_kick(..., kick_gain=1.0)`: multiply the injected `kick` by `kick_gain` (scales the stored composite's
  magnitude at write). Additive arg; `kick_gain=1.0` ⇒ the `kick[:] = _kick_re/_kick_im` writes are byte-identical.
- and/or `rf_set_complex_weights(connections, weight_gain=1.0)`: multiply `w_re`/`w_im` by `weight_gain` before
  building the CSR. `weight_gain=1.0` ⇒ byte-identical.

These are **additive + default-`1.0` + byte-identical when absent** (the owner's byte-review bar), and they touch
the *write* of the RF state, not the resonate dynamics (`_rf_lambda`/`_rf_omega` stay untouched, so the phase
read-out math is unchanged). **This edit is NOT required for the de-risk or the first functional target** (§2.2
does it entirely at the composer layer). It is scoped here so the owner can see the *worst case* is one small,
byte-reviewable arg — not a refactor of the bypassed fast loop. **Explicitly reject** the audit's `_rf_lambda`
guess: modulating the decay would change the phase read-out (the resonate is what *produces* the phase) and would
modulate already-stored facts on every op — wrong functional target and a moat risk.

---

## 3. The cheapest-first de-risk

**The single load-bearing question:** *does the shared dopamine state MODULATE a composer operation in a
load-bearing, NEURAL way — i.e. is a rewarded fact preferentially recalled over a neutral one (same cue strength)
BECAUSE of the DA-gated encoding, with the no-confab moat intact?*

### 3.1 Minimal test (numpy/CPU first; `RFPhasorComposer` with `enable_substrate_store=True`, small D)

1. Build one composer (small `D`, e.g. 64; `seed` swept). Use a **standalone spiking-SNc DA pool** (the
   `2026-06-19-multicue` standalone `snc` region) as the DA source, OR — for the *cheapest* first cut — inject the
   DA concentration directly (`manager.set_concentration("dopamine", ·)`) to isolate the *composer-side*
   mechanism before adding the spiking source. (Two rungs: rung-1 isolates "does encoding-gain → differential
   recall work"; rung-2 sources the gain from real SNc firing.)
2. Store a **NEUTRAL** fact (`dog go north`) with DA at baseline (`g≈1`) and a **REWARDED** fact (`cat eat apple`)
   with DA bursting (`g>1`, e.g. `g≈2`). **Crucially, cue strengths are matched** — both are normal 3-role SVO
   facts; the *only* difference is the encoding gain.
3. **Damage both equally:** add common read-side phasor noise (e.g. corrupt a fixed fraction of readout phases, or
   add a fixed superposition load) calibrated to sit at the graceful-degradation knee where a unit-gain fact
   *starts* to fail the cue-match.
4. Query both (`query_patient("dog","go")` and `query_patient("cat","eat")`).

### 3.2 GO bar (pre-registered)

- **Differential recall:** the REWARDED fact is recalled correctly at a damage level where the NEUTRAL fact
  **abstains / mis-recalls**, on **≥ 5/6 seeds** (the project's multi-seed bar). Quantify with the
  `confidence_gate` margin: rewarded-block margin > gate, neutral-block margin < gate, at the same damage.
- **DA is LOAD-BEARING (lesion control):** with DA held at baseline for *both* facts (`g=1` for both — "lesion
  the DA gate"), the differential **vanishes** (both abstain or both recall ~equally) on every seed. This is the
  anti-cheat that proves the effect is the DA-gated encoding, not an artifact of fact content/order.
- **No-confab MOAT intact:** an **unstored** cue (`query_patient("river","run")`) still returns `None` under
  *every* gain setting (including the high-gain rewarded regime) on every seed. The modulation must NEVER make the
  composer confabulate a fact it never stored. (Verify the abstention rate is unchanged from the `g=1` baseline.)
- **Monotonicity (sanity, not a hard gate):** recall reliability of a single fact under fixed damage rises
  monotonically with `g` across `g ∈ {0.5, 1.0, 1.5, 2.0, 3.0}` — confirming `g` is a genuine encoding-strength
  knob, not a threshold artifact.
- **Regression GREEN:** with `encoding_gain_fn=None`, `tests/test_one_brain_composer_agent.py` (11, incl. moat) +
  `tests/test_brain_conversational_agent.py` pass **verbatim** (default byte-identity).

### 3.3 Controls summary (the anti-cheats)

| Control | What it rules out | Expected |
|---|---|---|
| **Lesion DA** (both facts `g=1`) | "the effect is fact content/order, not DA" | differential vanishes |
| **Neutral-vs-rewarded, matched cue strength** | "the rewarded fact just had an easier cue" | only the gain differs ⇒ the gain is the cause |
| **Unstored-cue abstention at high gain** | "modulation broke the moat" | still `None`, every seed |
| **Permuted gain assignment** (apply the high gain to the *other* fact) | "one fact is intrinsically more robust" | the robustness follows the GAIN, not the fact |
| **Multi-seed (≥6)** | seed-luck | signature holds 5–6/6 |

CPU/numpy first (a `D=64` composer + a tiny SNc pool is seconds/seed). Only lift to GPU + the merged bridge once
the numpy de-risk is GO.

---

## 4. Honest risk + the clean GO vs NEGATIVE

### 4.1 The biggest mislead to guard against

- **A "modulation" that is really a global gain with no functional consequence.** If you scale *every* fact's
  encoding equally (or the damage is below the floor knee so even unit-gain facts reconstruct perfectly), you get
  a number that "changes" but no differential recall — a vacuous win. **Defense:** the lesion control + the
  matched-cue neutral-vs-rewarded contrast + calibrating the damage to the graceful-degradation knee. The GO
  *requires* a differential that disappears under lesion.
- **A modulation that weakens the moat.** A large gain that bleeds a fact's energy into neighboring blocks, or a
  gain-driven lowering of an abstention threshold, could make the composer "recall" something for an unstored cue.
  **Defense:** the unstored-cue abstention control is a HARD gate — any moat breach at any gain setting is a
  NEGATIVE, not a tunable. (Per `feedback_moat_not_hard_lossy_memory_ok`, the moat is a plus we keep where free;
  here it is free, so we keep it — a modulation that *requires* breaking it is the wrong design.)
- **Over-claiming the brain-based status.** The recommended first target applies `g` at the composer layer (a
  host multiply of a neural-valued gain). Be explicit (§2.3) that this matches the existing nav
  `dopamine→plasticity_rate` precedent; do NOT claim the gain *emerges* from spikes until the §2.4 on-substrate
  phase is built. Honest scope is the deliverable.

### 4.2 Clean GO vs NEGATIVE

- **GO:** rewarded fact recalled where neutral abstains, ≥5/6 seeds; differential vanishes under DA lesion; moat
  intact at every gain; regression green. ⇒ the shared dopamine state demonstrably modulates a composer operation
  in a load-bearing, neural-valued way. Then lift onto the merged bridge (read the merged `dopamine` modulator
  the limbic core drives — depends on #1, the shared limbic core landing on the merge).
- **NEGATIVE (an honest deliverable):** the encoding gain produces no differential recall (the floor/cleanup is
  too robust or too brittle to be salience-graded on the point-neuron substrate), OR it cannot be made to work
  without weakening the moat. Either is a real finding mapping what DA-gated encoding can/can't do on the RF
  substrate — log it; the alternative functional targets (e.g. salience-gated *cleanup* margin, or a DA-gated
  reconsolidation labilization threshold) become the next candidates, but those are explicitly *not* scoped here.

### 4.3 Tier-2 follow-ons this UNLOCKS (note, not scope)

Once the shared limbic state reaches the composer, the emergent-features finding's items become reachable (do NOT
scope here):
- **the persistent integrated loop** — salience modulating recall *online* during a live conversational turn;
- **mood / state-dependence** — a tonic DA/neuromodulator level biasing what is encoded/recalled across a session;
- **reconsolidation gating** — DA/novelty modulating the labilization threshold in `update_on_mismatch`
  (`one_brain_composer.py:605`), so a *surprising* correction is encoded more strongly (the natural pairing of
  the existing prediction-error-gated rewrite with the limbic salience signal).

**Dependency:** #6 depends on **#1** (the shared spiking reward/value/dopamine limbic core landing on the merged
bridge) to read a *merged* `dopamine` modulator in deployment. The de-risk in §3 does **not** need #1 (it uses a
standalone SNc pool / direct concentration injection), so it can run now; the *deployment* wiring waits on #1.

---

## 5. Key file:symbol references (verified this session)

- DA source: `research/runners/g11_bg_runner.py:4304` (`spiking_snc` → `dopamine` modulator via
  `from_region_firing_signed` over `["snc"]`); standalone pool per `2026-06-19-multicue-learning-firm-and-neural-reward.md` Part 2.
- NM subsystem: `sim/neuromodulators.py` — `NeuromodulatorManager.get_concentration` (`:228`),
  `_compute_production` `from_region_firing_signed` (`:774`), `compute_plasticity_rate_multiplier` (`:323`).
- NM application + the bypass: `sim/bridge.py:6819` (`neuromodulator_manager.step` inside
  `_run_one_simulation_step`), `:6894` (`get_concentration("dopamine")`), `:6919`
  (`compute_plasticity_rate_multiplier`). The RF fast path that bypasses it: `rf_resonate_steps` (`:5607`),
  `_rf_advance_one` (`:5568`), `_rf_resonate_steps_megakernel` (`:5672`), `rf_megastep` (`:5640`).
- The RF-op entry point for the gain: `one_brain_composer.py:251` (`_write_block`), `:264` (`_store_composite`),
  `:229` (`_compose_phases`); `rf_phasor_composer.py:406` (`_store_substrate`), `:317` (`store`).
- The magnitude floor (why the gain has a functional consequence): `sim/bridge.py:5589` / `:5662`
  (`_rf_mag2 > _rf_floor2`).
- The flagged sim/ edit (deferred): `rf_kick` (`sim/bridge.py:5504`), `rf_set_complex_weights` (`:5549`) — add a
  default-`1.0` `kick_gain` / `weight_gain`.
- Catalog: D.16 (`sim-catalog/references/feature-catalog.md:1272`), Kandel 6e Ch 54 pp 1366–1367.
- Biology: Lisman & Grace 2005 (Neuron 46:703); Lemon & Manahan-Vaughan 2006 (J Neurosci 26:7723).
- Roadmap parent: `research/findings/2026-06-18-full-spikeification-shared-substrate-roadmap.md` §3 #6, §3 #1.

---

## 6. Recommendation (one line)

Build the §3 **store-time DA-gated encoding-strength de-risk** numpy/CPU first (a NEUTRAL vs a REWARDED fact,
matched cue strength, common read damage → the rewarded fact recalled where the neutral abstains; DA lesion kills
the differential; moat intact; ≥5/6 seeds) — entirely at the composer layer via an additive default-off
`encoding_gain_fn` (**NO `sim/` edit**). On GO, lift onto the merged bridge (reading the shared `dopamine`
modulator) after #1 lands the limbic core; the `rf_kick(kick_gain=...)` `sim/` edit (byte-identical default) is
the *deferred* fully-on-substrate refinement, not a prerequisite.
