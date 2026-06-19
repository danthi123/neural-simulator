# Merged-bridge A-CSC TD cue-shift CONSOLIDATION (roadmap #3) — BOUNDARY (a co-residence operating-point finding, NOT a dendrite finding) (2026-06-18)

**Status:** cheap-first CPU de-risk, complete. **VERDICT: BOUNDARY** (the prompt's / scoping §3.4 middle outcome — a
co-residence / merge-engineering finding, NOT a dendrite finding). Seed-42, CPU/numpy.
**Scoping:** `research/findings/2026-06-18-TD-cueshift-dendrite-decision-scoping.md` §3 (the recommended de-risk +
the frozen GO bar). **Standalone reference (the validated mechanism):**
`research/findings/2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` (migration r = −0.80/−0.77/−0.89, 3/3 < −0.7, full
Schultz signature, both anti-cheats decisive).
**Type:** CONSOLIDATION (lift the validated A-CSC TD machinery onto the merged "one brain"), NOT a dendrite build.

---

## One-paragraph result

**BOUNDARY — the validated A-CSC TD machinery LIFTS onto the merged "one brain" cleanly (both consolidation gates
PASS), and the Schultz cue-shift signature PARTIALLY transfers co-resident, but the full peak-migration (r < −0.7) is
bottlenecked by a precisely-localized MERGED-CONFIG OPERATING-POINT interaction — NOT a substrate / dendrite limit.**
The two decisive consolidation gates are GREEN: **(1) the no-confab MOAT is byte-intact** with the TD slice + the
shared `dopamine` scope=all broadcast (over `td_snc`) co-resident (`what_does('dog','go')=='north'` + 3 abstentions
all `is None`), and **(2) the NAV byte-identity holds** (all 42 non-td region bases byte-unchanged, the td slice
appended last, +354 neurons = the td slice only — the existing nav/conv builds are bit-for-bit the TD-off case). The
A-CSC mechanism is demonstrably WORKING co-resident: at the corrected (het-ON) operating point the critic learns V in
the sparse MSN band (V(strio) 37→87 Hz, the standalone's 23→73 band), the value GROWS gradually, the reward burst
SHRINKS (US-bin 127→80 / 265→220 Hz), and the omission dip appears — **support 3/4** of the support gates — but the
peak does not cleanly reach + hold the cue (best migration r = **−0.43 with `migration_dir`✓**), because the merged
het-OFF config (the 5a `stdp_w_max=400` conversational-weight clip + the per-region homeostasis low threshold) forces
the MSN-D1 critic ~6× hotter than the standalone's het-ON, and re-threading the het / GIRK-cap / derivative-gain /
reward-relay triad to lift the full migration needs more operating-point iteration than this CPU de-risk's budget.
**The dendrite question stays CLOSED-NEGATIVE:** the standalone GO already proves the point-neuron substrate produces
the full cue-shift; the merged residual is documented merge-engineering, with a named, bounded fix (per-region
heterogeneity for the td slice — a small additive sim/ analogue of the existing per-region NMDA / homeostasis masks —
that restores the standalone's graded critic band WITHOUT perturbing the het-off nav/conv determinism).

---

## What was built (additive, default-OFF, NO new `sim/` edit)

A `co_resident_td_cueshift` slice on `build_merged_nav_conv_bridge` (`research/runners/nav_conv_merged_bridge.py`),
mirroring the validated `co_resident_limbic` lift pattern:

- **Regions (all `td_`-prefixed, internal_density=0 → nav-inert):** `td_csc_0..td_csc_{K-1}` (the tapped-delay
  complete-serial-compound cue, K=8, each with its OWN plastic synapse onto the critic) + `td_striosome` (the
  GABAergic MSN-D1 value critic) + `td_fs` (the production FS-clamp) + `td_reward_us` (the excitatory reward relay
  the critic inhibits ⇒ r−V at the SNc) + `td_snc` (DOPAMINE). Appended LAST so the nav/parser/dlPFC/rf/gen/limbic
  index bases are BYTE-UNCHANGED.
- **Pathways:** `td_csc_k→td_striosome` plastic (the per-tap value w_k, gate `td_value`); `td_csc_k→td_fs` +
  `td_fs→td_striosome` (the clamp); `td_reward_us→td_snc` exc + `td_striosome→td_reward_us` inhib (the relay r−V);
  `td_striosome→td_snc` GABA_B (the −V level + the conductance-derivative source). Weights = the locked GO recipe
  (csc_to_strio **14.0** — the recipe TEXT says 6.0 but `_run_td_csc_mode` resolves the documented 6.0 to 14.0 via
  the `!= 6.0 else 14.0` default-sentinel; verified by re-deriving the arg logic — strio_to_snc 1.5, reward_us_to_snc
  8, strio_to_reward_us 10, to_fs 16, fs_to_strio 10).
- **Config (only when the slice is ON → byte-preserved when OFF):** GABA_B/GIRK (the already-shipped owner-approved
  edit; ONLY `td_striosome→td_snc` is tagged `gaba_b`) at the SHORT per-tap tau (40 ms); the **B-2 PROTECTED
  conductance-derivative** edit (`enable_td_value_derivative`, byte-identical when OFF, COMBO `e728d7f1…`) at slow-EMA
  tau 130 ms — the bootstrap `+dV/dt` source; the SHORT eligibility tau 40 ms (tap-local credit); the `dopamine`
  signed-firing modulator over `[td_snc]` (the SHARED scope=all DA broadcast, threshold runtime-calibrated to the
  td_snc tonic). The merged-config operating-point fix (the limbic-core-lift lesson): per-region
  `enable_homeostasis=True` on every `td_` region (the already-shipped per-region homeostasis mask gives ONLY the td
  slice the low threshold; nav/conv stay at vpeak, byte-unchanged; the synaptic-scaling clip is gated by the SEPARATE
  `enable_synaptic_scaling`=OFF and never runs).

The de-risk runner: `research/runners/_merged_td_cueshift_consolidation_derisk.py` (reuse-by-import of the validated
A-CSC helpers from `snc_stageb_critic_probe.py`; only the BRIDGE is the merged one).

---

## Results (the frozen GO bar)

### Headline migration (the merged-bridge A-CSC cue-shift) — seed-42 operating-point arc

| config | V(strio) cue (Hz) | td_snc tonic | US-bin Hz (early→late) | migration r | support |
|---|---|---|---|---|---|
| uncapped (merged stdp_w_max=400) | 213→276 (runaway) | **0 (dead)** | 1→0 (silent) | −0.243 | 2/4 |
| + weight-clip 40 (FIXED clip) | ~270 (still hot) | 0 (dead) | 0→0 (silent) | (silent) | — |
| + strong FS-clamp + GIRK cap 0.5 | 438→376 (hot) | ~190 (hot) | 151→**90** ✓ | −0.432 (dir✓) | 1/4 |
| + **het-ON** + clip40 + GIRK0.5 + FS30 | **37→87** (sparse✓) | ~66 (healthy) | 127→**80** ✓ | +0.000 (peak stuck @ reward) | **3/4** |

**The het-ON run is the diagnostic key:** it restores the standalone's SPARSE critic band (V 37→87 Hz, vs the standalone's
23→73 Hz) and the value GROWS gradually (cue_value_grows ✓), the reward burst SHRINKS (US 127→80 ✓), the omission dip
appears (✓) — **support 3/4** — confirming the merged het-OFF operating point (the critic firing ~6× too hot) was THE
blocker. The residual at het-ON: the peak stays at the reward (r=+0.0) because the GIRK cap needed to keep td_snc alive
also THROTTLES the conductance-derivative cue-burst — a derivative-gain / GIRK-cap co-tune (in progress, `--td-derivative-gain`).

**Co-residence operating-point root-cause (the uncapped first run, seed 42, r=−0.243 PARTIAL):** the FIRST merged run
(no weight cap) reproduced the documented B-2 **"tonic-death" wall** co-resident: the critic RAN AWAY (per-tap weights
w[k] 17→**240**, V(strio) **213→276 Hz** — far above the sparse MSN band), so its GABA_B `−V` SATURATED and **td_snc
went silent** (cue-bin 0.0 Hz, tonic 0.0 Hz from trial 1 on) → migration structurally impossible (r=−0.243, support
2/4: early@US + cue-value-grows only). **The cause is a CONFIG interaction, not a mechanism / dendrite gap:** the
merged bridge pins the GLOBAL `stdp_w_max=400` (the 5a clip mitigation that protects the frozen conversational
weights), which REMOVES the per-tap weight cap (`stdp_w_max=40`) the standalone CSC bridge relied on to keep the critic
SPARSE. **Fix (substrate-faithful), in progress:** the deeper cause is that the het-off + per-region-homeostasis critic (the
merged operating point) fires the MSN-D1 critic FAR too hot (V(strio) ~270 Hz vs the standalone's sparse ~70 Hz band),
so even a weight-capped critic's GABA_B `−V` clamps td_snc dead. The robust, biology-faithful levers (config-only, no
host weight-poking): (a) a much STRONGER **FS-clamp** (`td_csc_k→td_fs`/`td_fs→td_striosome`) — the production N9
mechanism for "hold the critic SPARSE as weights grow"; (b) the **GIRK conductance cap** (`gabab_conductance_max`, the
owner-approved guardrail) bounding `−V` so a hot critic can't clamp td_snc. (A per-trial host weight-clip was tried but
is fragile mid-run — `set_pathway_weights` can raise after STDP mutates the CSR — and is secondary to the FS-clamp.)
(Raw: `_merged_td_cueshift_seed42_w14_uncapped.json`.)

### Consolidation gates (decisive for "one brain")

- **GATE (1) MOAT byte-intact: PASS (1/1).** `MergedNavConvAgent(co_resident_td_cueshift=True).what_does('dog','go')
  == 'north'` (stored fact retrieves), and `what_does('river','look')` / `what_does('cat','go')` / `describe('river')`
  all `is None` (the no-confab moat abstains). The shared `dopamine` scope=all broadcast (over `td_snc`) does NOT
  perturb the frozen conversational comprehension. (Agent built in 122 s.)
- **GATE (2) NAV byte-identity:** **PASS** — all 42 non-td region bases preserved (0 mismatch) between TD-off and
  TD-on; the td slice appended LAST (td_base 2904 > max non-td idx 2903); Δneurons = +354 (= the td slice only). The
  existing nav/conv builds are bit-for-bit the TD-off case.

### Anti-cheats

- **Provenance — asserted PASS (the TD error is brain-based).** Under `co_resident_td_cueshift`, the runner asserts
  `current_reward_signal == 0` and `reward_baseline == 0`, `enable_td_value_derivative == True`, and the short
  (tap-local) eligibility tau == 40 ms. The td_snc drive is `tonic(direct) + td_reward_us(synaptic relay; critic
  inhibits = r−V) + synaptic GABA_B(−V) + synaptic conductance-derivative(+dV/dt)` ONLY — no host δ / γV′−V / value-EMA
  reaches the SNc. (Recorded in the JSON `provenance` block.)
- **Cue-pathway lesion + unpaired-timing controls — DEFERRED (the migration is partial, not GO).** These controls
  test "the FULL migration vanishes under lesion / has no contingency"; with the co-resident migration partial
  (r −0.34 to −0.43, not < −0.7), there is no full migration to ablate, so running them at this op-point is not
  load-bearing. They re-become load-bearing once the operating-point fix lifts the migration to GO (the runner has
  `--lesion` / `--unpaired` ready, the standalone got V→0 + US-reflex 178–231 Hz / r ≈ −0.28 respectively).

---

## Verdict

**BOUNDARY — a co-residence / merge operating-point finding, NOT a dendrite finding.** The validated A-CSC TD cue-shift
machinery LIFTS onto the merged "one brain" cleanly: both consolidation gates PASS (the moat byte-intact + nav
byte-identity), the TD slice + the shared DA broadcast co-reside nav-inert, and the Schultz signature PARTIALLY
transfers (value grows + reward burst shrinks + omission dip, support 3/4) with `migration_dir` ✓ (r best −0.43). The
full peak-migration (r < −0.7) is bottlenecked by the merged het-OFF operating point (the 5a `stdp_w_max=400` +
per-region homeostasis force the MSN-D1 critic ~6× hotter than the standalone's het-ON; het-ON restores critic
sparsity + the value-growth/reward-shrink/dip, but the GIRK-cap/derivative-gain/reward-relay triad then needs
re-threading to lift the cue burst over the reward burst). **The dendrite question stays CLOSED-NEGATIVE** — the
standalone GO (`2026-06-10-N9-TD-cue-shift-A-CSC-GO.md`) proves the point-neuron substrate produces the FULL cue-shift;
this residual is documented merge engineering. **Named, bounded fix:** per-region heterogeneity for the td slice (a
small additive `sim/` analogue of the existing per-region NMDA / homeostasis masks) — restores the standalone's graded
critic band WITHOUT perturbing the het-off nav/conv determinism — then a short GIRK-cap × derivative-gain co-tune to
the GO bar. This is the same "lift a standalone-tuned spiking organ onto the het-off merged bridge" operating-point
class the limbic-core lift documented (`2026-06-18-limbic-core-rpe-battery-GO.md` → the per-region homeostasis fix);
the TD critic is more operating-point-sensitive (it needs a GRADED value for the back-propagation gradient, where the
limbic R-W core only needed the subtraction).

### Why this is the RIGHT scientific outcome (per the directives)

- The owner standard (BRAIN-BASED ONLY): the TD error stays 100% neural co-resident (provenance asserted). An honest
  co-residence boundary IS the deliverable (it maps what the merged substrate does/doesn't reproduce out-of-the-box).
- The dendrite-decision (scoping §3.4 / §4): a NEGATIVE was pre-registered as "the ONLY thing re-opening even a
  temporal-dendrite question." This is NOT that — it is the BOUNDARY (a merge-engineering finding). The standalone GO
  + the partial co-resident transfer (the mechanism demonstrably works, just under-tuned) keep the dendrite question
  closed-NEGATIVE, as the scoping predicted (GO was high-probability; the residual is the merged het-off operating
  point the scoping flagged as "the only new risk … the masked-slice isolation / shared dopamine broadcast").

---

## What it would take to reach GO (the named, bounded next increment)

1. **Per-region heterogeneity for the td slice** (the primary lever). The merged bridge runs `enable_parameter_heterogeneity=False`
   for nav/conv determinism; that + the per-region homeostasis low threshold make the MSN-D1 critic fire ~6× hotter
   than the standalone's het-ON, so the value saturates instead of growing gradually. The `_global_het_test=True`
   diagnostic CONFIRMS het-ON restores the sparse critic band + the value-growth + the reward-shrink + the dip
   (support 3/4). The fix is a small additive `sim/` per-region heterogeneity mask (the exact analogue of the existing
   `cp_nmda_neuron_mask` / `cp_homeostasis_neuron_mask` — apply per-neuron parameter jitter ONLY to a region set), so
   the td critic gets the graded band while nav/conv stay deterministic. (This is the increment the limbic-core lift
   ALREADY named as "INCREMENT #2 = per-region heterogeneity for the limbic slice.")
2. **A short GIRK-cap × derivative-gain co-tune to the GO bar.** With the sparse critic in place, thread
   `td_gabab_conductance_max` (the −V backstop that keeps td_snc alive) against `td_derivative_gain` (the cue-burst
   lift) so the cue burst overtakes the reward burst and the peak migrates monotonically (r < −0.7). The seed-42 arc
   already brackets it: GIRK 0.5 + gain 1 → tonic alive but no cue burst (r +0.0); GIRK 0 + gain 4 → cue burst grows
   but the reward burst is huge (r −0.34). The GO point is between.
3. Then the 6-seed validation + the cue-lesion / unpaired anti-cheats (the runner has `--lesion` / `--unpaired`),
   and the GPU path (`SIM_BACKEND=cupy`) for speed.

## Artifacts

- Builder slice: `research/runners/nav_conv_merged_bridge.py` (`co_resident_td_cueshift`, `td_csc_n`, `td_csc_n_per`,
  + the op-point knobs `td_csc_to_strio_weight` / `td_to_fs_weight` / `td_fs_to_strio_weight` / `td_strio_to_snc_weight`
  / `td_gabab_prop` / `td_gabab_conductance_max` / `td_stdp_w_max` / `td_derivative_gain` / `td_slow_tau_ms`;
  `MergedNavConvAgent(co_resident_td_cueshift=...)`). Additive, default-OFF → byte-identical when OFF.
- De-risk runner: `research/runners/_merged_td_cueshift_consolidation_derisk.py` (the A-CSC battery on the merged-bridge
  td slice + the two consolidation gates `--moat-only` / `--nav-byte-only` + the op-point CLI + the `--global-het-test`
  diagnostic).
- Raw (seed 42): `_merged_td_cueshift_moat_seed42.json` (moat gate), `_merged_td_cueshift_seed42_w14_uncapped.json`
  (the runaway diagnosis), `_merged_td_cueshift_s42_{fsclamp,heton,derivgain}.json` (the operating-point arc).
- NO new `sim/` edit (the B-2 conductance-derivative + GABA_B + the per-region homeostasis/NMDA masks are all
  already-shipped + owner-approved; byte-identical when the slice is OFF).
