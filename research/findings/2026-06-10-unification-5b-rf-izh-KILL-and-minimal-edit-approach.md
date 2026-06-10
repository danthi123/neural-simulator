# Unification de-risk 5b — RF/Izhikevich shared-v/u corruption CONFIRMED; the minimal strict edit is sliced RF ops (2026-06-10)

**Roadmap step 2 (consolidate navigation + conversational onto one brain), STRICT (RF co-resident) path,
cheapest-first de-risk 5b** (`docs/plans/2026-06-10-nav-conv-single-instance-unification-design.md` §5b).

## The kill (as-is)

The resonate-and-fire (RF) composer stores its complex phasor state `Z = re + i*im` in the SAME two arrays the
Izhikevich navigation neurons use — `re` in `cp_membrane_potential_v`, `im` in `cp_recovery_variable_u`
(`sim/bridge.py:5380-5381`). The neuron-model dispatch in `_run_one_simulation_step` is a single GLOBAL branch
on `cfg.neuron_model_type` (`:5870`). Probe
`research/runners/derisk_unification_5b_rf_izh_coexistence.py` (GPU): put a unit phasor in `v`/`u`, run ONE
step under each model.

| State | mean \|z\| |
|---|---|
| initial unit phasor | 1.0000 |
| after 1 **RF** step (reference, `rf_resonate_steps(1)`) | **1.0000** (rotates cleanly, magnitude preserved) |
| after 1 **Izhikevich** `_run_one_simulation_step` | **16.28** (phasor destroyed; dev 16.8 from the RF rotation) |

**KILL CONFIRMED.** One Izhikevich step (its `+140` drive then spike-reset) sends the phasor off the unit
circle. RF and Izhikevich cannot time-share `v`/`u` in one global step dispatch as-is — the protected `sim/`
edit is genuinely required. This empirically reproduces the design's §2 verdict.

**Edited-version PASS criterion (pinned):** on a mixed bridge, one step advances the RF slice by RF dynamics
(its phase read-back byte-matches `rf_reference_one_step`) AND the Izhikevich slice by Izhikevich dynamics
(matches a pure-Izhikevich control); with the feature OFF, RESONATE_AND_FIRE-only and IZHIKEVICH-only bridges
are byte-identical to today.

## The important refinement: the minimal strict edit is SLICED RF OPS, not a core-step-loop dual-dispatch

Thinking about how the production RF composer actually uses the substrate (`rf_phasor_composer.py`) changes
the right edit:

1. **The composer is STATELESS across operations.** Each bind/unbind op is atomic: `rf_set_complex_weights`
   (fresh) → `rf_kick` (re-initialise the phasor) → `rf_resonate_steps` → `rf_read_phases`. It never relies on
   the phasor persisting across operations — it re-kicks every time.
2. **The composer's stored memory lives in COMPLEX SYNAPSES** (`cp_rf_w_re` / `cp_rf_w_im`), which are
   array-disjoint from BOTH `v`/`u` AND the navigation's real-valued `cp_connections`. The Izhikevich step
   touches only `v`/`u` and `cp_connections.data` — never the complex weights.

**Consequence:** the RF slice's `v`/`u` does NOT need to survive a navigation `_run_one_simulation_step`. If
nav's Izhikevich step clobbers the RF slice's `v`/`u` between composer ops, the next op re-kicks it — no harm,
and the stored memory (complex weights) is untouched. The ONLY real requirement for co-residence is the
reverse: a composer op must not clobber the NAVIGATION slice's `v`/`u`, because today `rf_kick` /
`_rf_advance_one` / `rf_resonate_steps` operate on the WHOLE bridge.

So the minimal, correct, strict-satisfying edit is to **slice the RF ops to the RF neurons**:
- Add an optional neuron-slice (mask/indices) to `rf_kick`, `_rf_advance_one`, `rf_resonate_steps`,
  `rf_read_phases`. When set, they read/write ONLY the RF slice's `v`/`u` and trackers; the navigation slice
  is untouched. Default (no slice) = whole bridge = **byte-identical to the composer's current standalone
  usage** (the composer's per-op bridges are 100% RF, so they pass no slice and behave exactly as today).
- This touches ONLY the RF-specific methods — it does NOT modify the Izhikevich / HH / AdEx dynamics or the
  global step-loop dispatch, so those paths stay byte-unchanged by construction. Far lower blast radius than
  the per-neuron dual-dynamics step the design §2.3 sketched (which over-engineers for a composer that doesn't
  need the RF slice stepped by the main loop at all).

**Co-residence orchestration on the merged bridge:** navigation runs `_run_one_simulation_step` (harmlessly
clobbering the idle RF slice's `v`/`u`); when the agent converses, the composer runs its ops on the RF slice
(`rf_*` with the RF mask), leaving nav's `v`/`u` intact. Runner-side wiring constraint: the RF neurons carry
only complex synapses (no `cp_connections` out-edges into nav), so their incidental Izhikevich firing between
ops injects nothing into navigation — already true of the composer's `connections_per_neuron=0` bridges.

This is the edit to build (smaller + safer than the LATEST-6 dual-dispatch scoping), byte-prove (mask-off ==
baseline byte-identical; RF-only + Izh-only bridges unchanged), de-risk (composer ops on the RF slice of a
mixed Izhikevich bridge reproduce the standalone composer's bind/unbind while nav is unaffected), and present
for owner byte-review before relying.
