# Tier 2 #6 de-risk — the shared dopamine state functionally modulates the conversational composer (encoding strength): GO

**Date:** 2026-06-19
**Type:** cheap-first de-risk (CPU/numpy, INLINE, multi-seed). The deepest "one self" rung of the TRUE ONE BRAIN line
(owner directive `feedback_move_everything_to_shared_spiking_substrate`): let the SAME motivational/limbic brain that
drives navigation actually REACH the conversational composer, so dopamine state modulates BOTH halves.
**Pre-registered by:** `research/findings/2026-06-19-tier2-limbic-to-composer-scoping.md` (commit `eeae7264`).
**Verdict:** **GO.** The dopamine-driven encoding gain gives a load-bearing, content-controlled, moat-safe recall
lift. The DA-lesion control kills the differential; the no-confab moat holds 6/6 in every condition; the default is
byte-identical.

---

## 0. The question (load-bearing)

Does scaling a fact's stored complex-phasor magnitude by a dopamine-driven encoding gain
`g = 1 + k_DA·(DA − DA_baseline)` (applied AT STORE TIME — the Lisman-Grace hippocampal-VTA loop; Kandel D.16:
dopamine makes a memory trace STABLE vs degradable) make a REWARDED fact (g>1) recallable where a NEUTRAL fact
(g=1, MATCHED cue strength) degrades under common read damage — **in a load-bearing, neural way** (a DA-lesion kills
the differential), **without weakening the no-confab moat**?

## 1. The verified mechanism

The resonate-and-fire (RF) phase read-out has a hard **magnitude floor** (`sim/bridge.py:5589`,
`_rf_mag2 > _rf_floor2`): a readout neuron whose complex magnitude `|Z|` decays below the floor never registers the
`im` up-crossing → its `spike_step` stays at the default → `rf_read_phases` returns phase 0 (garbage), contributing
nothing to the cleanup matched filter.

Under **common, gain-independent** additive read noise of fixed σ, a readout neuron's per-neuron SNR is `g·M / σ`
(M = the base readout magnitude, ~208 after the 208-step resonate integration; the gain scales it to `g·M`). So a
higher-gain (rewarded) fact has higher per-neuron SNR → cleaner recovered phase → it survives the floor and the
cleanup recalls it; a unit-gain (neutral) fact's low-SNR neurons drop below the floor → garbled phase → the cleanup
mis-recalls. **The floor × noise interaction is the nonlinearity that makes a per-fact encoding gain DIFFERENTIAL,
not a vacuous global scalar.** This is the direct analogue of Kandel D.16's "an un-rewarded trace degrades below a
usable level; a D1-gated (rewarded) trace stays stable."

## 2. What was built (composer-layer; NO `sim/` edit; default-OFF = byte-identical)

Commit `b4ae63b0` — `research/runners/rf_phasor_composer.py`:
- `RFPhasorComposer(..., encoding_gain_fn=None)`: an optional callable `() -> float` read AT STORE TIME (a probe DA
  value here; the live shared `dopamine` SNc modulator in deployment). When set, `_store_substrate` multiplies the
  fact's composite phasor written into the persistent substrate weights by the per-fact gain `g`
  (`complex(g) * zc[k]`). `None` → `g=1.0` for every fact → the **byte-identical** unit-magnitude write.
- Default-preserving de-risk read-damage knobs (so the graceful-degradation knee can be reached): `_retrieve_noise`
  (common, gain-INDEPENDENT additive complex read noise on the recovered readout phasor, with the RF magnitude floor
  applied to the noisy phasor), `_retrieve_read_floor`, `_retrieve_lam`, `_retrieve_kick_mag`. All default to the
  current `_retrieve_substrate` behaviour exactly (`noise=0` → off → unchanged).

Commit `5928465f` — `research/runners/_phaseB_dopamine_encoding_gain_derisk.py` + the raw result.

NO `sim/` edit anywhere: the gain is a composer-layer multiply on the written complex weight; the read damage is a
composer-layer perturbation of the recovered phasor. Recall / the cue-match scan / abstention are untouched.

## 3. The de-risk design (content-controlled)

Two facts on one substrate-store composer (`enable_substrate_store=True`, D=64): a NEUTRAL fact `dog go north` and a
REWARDED fact `cat eat apple` — **matched cue strength** (both plain 3-role SVO; the only intended difference is the
encoding gain). Plus an UNSTORED cue `river run` (the moat probe). 6 seeds (42/43/44/100/101/102). Common read damage
σ = **260** = the moat-safe knee for D=64 / two facts.

**The KEY measure is the WITHIN-FACT paired contrast** — each fact recalled at g=1 AND at g=2 with its CONTENT held
fixed (so the content-robustness confound a between-fact comparison carries is removed):
- REAL (neu fact g=1, rew fact g=2) → `neu_fact@g1` + `rew_fact@g2`
- PERMUTED (the gain swapped to the OTHER fact: neu fact g=2, rew fact g=1) → `neu_fact@g2` + `rew_fact@g1`
- LESION (both g=1) → the within-fact null (no fact gets a gain → no differential between them)

## 4. Results (6 seeds, deterministic)

| Measure | Result |
|---|---|
| **WITHIN-FACT gain lift — NEUTRAL fact** (`dog go north`) | g1 **4/6 → g2 6/6** (+2) |
| **WITHIN-FACT gain lift — REWARDED fact** (`cat eat apple`) | g1 **2/6 → g2 6/6** (+4) |
| **TOTAL within-fact gain lift** | **+6/12** |
| LESION within-fact null (both g=1, between-fact diff = content only) | 2 (the gain lift +6 exceeds it) |
| BETWEEN-FACT (REAL: neu g=1, rew g=2) | rewarded **6/6** vs neutral **4/6** |
| **MONOTONICITY** (single fact, fixed damage) | g0.5 **1** → g1.0 5 → g1.5 6 → g2.0 6 → g3.0 6 — **monotonic** |
| **MOAT intact (HARD gate)** | **6/6 in EVERY condition** (real 6/6, permuted 6/6, lesion 6/6) |
| **REGRESSION** (`encoding_gain_fn=None` == g=1) | **6/6 byte-identical** |

**The decisive control (DA-lesion):** with the gain removed (both facts g=1), there is **no between-fact
differential** beyond the residual content asymmetry — i.e. the rewarded-vs-neutral advantage exists ONLY when the
gain is applied, and it **follows the gain** (the permuted control: swapping the high gain to the neutral-slot fact
flips the advantage to that fact). This proves the effect is the DA-gated encoding, not fact content or order.

**The moat (HARD gate):** the unstored cue `river run` abstains (returns `None`) 6/6 in every condition, **including
the high-gain (g=2) regimes** — a higher encoding gain does NOT make the composer confabulate a stored fact. The
abstention rate is unchanged from the g=1 baseline.

## 5. Honest scope + the one characterized boundary

- **The damage knee is moat-bounded.** Above σ≈280 the within-fact differential grows further, but the heavy read
  damage ITSELF (not the gain) begins to breach the moat on one seed (seed 42 returned a stored patient for the
  unstored cue at σ=350, while the SAME composer abstains correctly at σ=0). Per the HARD gate ("any moat breach at
  any setting is a NEGATIVE, not a tunable"), the de-risk operates at σ=260 where the moat is fully intact 6/6 AND
  the within-fact gain lift is strong (+6/12). This is the honest operating point, not the maximal-differential one.
- **Brain-based status (per the scoping §2.3, explicit).** The gain `g` is APPLIED at the composer layer (a host
  multiply), but its VALUE is the spiking SNc's firing-derived `dopamine` concentration, and what it scales is
  **synaptic weight magnitude at encoding** — the direct analogue of dopamine-gated LTP (Lisman-Grace / Kandel
  D.16). This is the SAME status as the already-shipped nav `dopamine → plasticity_rate` path (a neural
  concentration scaling a synaptic change). The fully-spiking ideal (the gain EMERGING from live SNc firing during
  an on-bridge store op) is the deferred follow-on, NOT claimed here.
- **Validated at D=64, 2 facts, numpy.** The deployment lift (read the merged `dopamine` modulator the shared limbic
  core drives) waits on #1 (the limbic core landing on the merged bridge).

## 6. Verdict + the deployment follow-on

**GO.** The shared dopamine state demonstrably modulates a conversational-composer operation (fact encoding strength)
in a load-bearing, content-controlled, neural-valued, moat-safe way: turning the DA gain from g=1 to g=2 on the SAME
fact lifts its recall under common read damage (+6/12 across both facts), the DA-lesion kills the differential, the
effect follows the gain (permuted), it is monotonic in g, the no-confab moat holds 6/6 everywhere, and the default
is byte-identical.

**Recommended follow-ons:**
1. **Deployment wiring (depends on #1):** pass
   `encoding_gain_fn = lambda: 1.0 + k_DA·(bridge.neuromodulator_manager.get_concentration("dopamine") − 0.5)`
   (clipped to `[0.5, 3.0]`) on the merged nav+conv bridge, so a fact heard while the spiking SNc is bursting (a
   rewarded/salient utterance) is encoded stronger than one heard at DA baseline. The DA SOURCE already exists
   (the standalone spiking-SNc RPE pool, `2026-06-19-multicue-learning-firm-and-neural-reward.md` Part 2, and the
   nav `spiking_snc` → `dopamine` route).
2. **Unlocked emergent features** (now reachable; the scoping §4.3): salience-gated recall online during a turn;
   tonic-DA mood/state-dependence biasing what is encoded across a session; DA-gated reconsolidation (modulate the
   labilization threshold in `update_on_mismatch` so a *surprising* correction is encoded more strongly — the
   natural pairing of the existing prediction-error-gated rewrite with the limbic salience signal).
3. **The deferred fully-on-substrate `sim/` edit** (the scoping §2.4): an additive default-`1.0` `kick_gain` on
   `rf_kick` / `weight_gain` on `rf_set_complex_weights` if a later phase wants the gain driven by live SNc firing
   during the on-bridge store op (byte-identical when absent; NOT required for this de-risk).

## 7. Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_dopamine_encoding_gain_derisk
# -> within-fact gain lift +6/12, LESION null 2, moat 6/6 all, regression 6/6, monotonic -> GO
```

## 8. Commits

- `b4ae63b0` — add the default-OFF `encoding_gain_fn` + read-damage knobs to `RFPhasorComposer` (byte-identical default).
- `5928465f` — the de-risk runner + the raw result.

Both on `main`, pushed to `origin` + `gitea`. Composer + agent regression tests pass
(`tests/test_one_brain_composer_agent.py`, `tests/test_brain_conversational_agent.py`,
`tests/test_rf_phasor_composer.py`: 53 passed, 22 skipped — the skips are GPU-gated).
