# Merged-bridge A-CSC TD cue-shift CONSOLIDATION (roadmap #3) — &lt;VERDICT&gt; (2026-06-18)

**Status:** cheap-first CPU de-risk. **PLACEHOLDER — numbers filled when the seed-42 + multi-seed runs land.**
**Scoping:** `research/findings/2026-06-18-TD-cueshift-dendrite-decision-scoping.md` §3 (the recommended de-risk +
the frozen GO bar). **Standalone reference (the validated mechanism):**
`research/findings/2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` (migration r = −0.80/−0.77/−0.89, 3/3 < −0.7, full
Schultz signature, both anti-cheats decisive).
**Type:** CONSOLIDATION (lift the validated A-CSC TD machinery onto the merged "one brain"), NOT a dendrite build.

---

## One-paragraph result

&lt;FILL: the verdict + the headline migration r per seed on the merged bridge + the two consolidation gates&gt;

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

### Headline migration (the merged-bridge A-CSC cue-shift)

&lt;FILL: per-seed migration r table&gt;

**Co-residence operating-point root-cause (the uncapped first run, seed 42, r=−0.243 PARTIAL):** the FIRST merged run
(no weight cap) reproduced the documented B-2 **"tonic-death" wall** co-resident: the critic RAN AWAY (per-tap weights
w[k] 17→**240**, V(strio) **213→276 Hz** — far above the sparse MSN band), so its GABA_B `−V` SATURATED and **td_snc
went silent** (cue-bin 0.0 Hz, tonic 0.0 Hz from trial 1 on) → migration structurally impossible (r=−0.243, support
2/4: early@US + cue-value-grows only). **The cause is a CONFIG interaction, not a mechanism / dendrite gap:** the
merged bridge pins the GLOBAL `stdp_w_max=400` (the 5a clip mitigation that protects the frozen conversational
weights), which REMOVES the per-tap weight cap (`stdp_w_max=40`) the standalone CSC bridge relied on to keep the critic
SPARSE. **Fix (substrate-faithful):** re-clip ONLY the `td_value`-gated synapses to a LOCAL cap per trial (a
weight-BOUND, not a host computation of value/reward/δ — the cue-shift stays 100% neural) + the GIRK conductance cap
(`gabab_conductance_max`, the owner-approved guardrail) as the −V backstop. (Raw: `_merged_td_cueshift_seed42_w14_uncapped.json`.)

### Consolidation gates (decisive for "one brain")

- **GATE (1) MOAT byte-intact: PASS (1/1).** `MergedNavConvAgent(co_resident_td_cueshift=True).what_does('dog','go')
  == 'north'` (stored fact retrieves), and `what_does('river','look')` / `what_does('cat','go')` / `describe('river')`
  all `is None` (the no-confab moat abstains). The shared `dopamine` scope=all broadcast (over `td_snc`) does NOT
  perturb the frozen conversational comprehension. (Agent built in 122 s.)
- **GATE (2) NAV byte-identity:** **PASS** — all 42 non-td region bases preserved (0 mismatch) between TD-off and
  TD-on; the td slice appended LAST (td_base 2904 > max non-td idx 2903); Δneurons = +354 (= the td slice only). The
  existing nav/conv builds are bit-for-bit the TD-off case.

### Anti-cheats

- **Cue-pathway lesion:** &lt;FILL&gt; — migration vanishes, the US reflex survives.
- **Unpaired-timing control:** &lt;FILL&gt; — no migration (discriminating).
- **Provenance:** the td_snc drive is `tonic(direct) + td_reward_us(synaptic relay; critic inhibits = r−V) +
  synaptic GABA_B(−V) + synaptic conductance-derivative(+dV/dt)` ONLY — `current_reward_signal == 0`, no host δ /
  γV′−V / value-EMA (asserted in the runner).

---

## Verdict

&lt;FILL: GO / BOUNDARY / NEGATIVE + the dendrite-decision implication&gt;

---

## Artifacts

- Builder slice: `research/runners/nav_conv_merged_bridge.py` (`co_resident_td_cueshift`, `td_csc_n`, `td_csc_n_per`;
  `MergedNavConvAgent(co_resident_td_cueshift=...)`).
- De-risk runner: `research/runners/_merged_td_cueshift_consolidation_derisk.py`.
- Raw: `research/findings/raw/_merged_td_cueshift_*.json`.
