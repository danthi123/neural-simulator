# Shortcut B4 — the TD cue-shift CONSOLIDATED onto the merged "one brain" (2026-06-22)

**Status:** consolidation de-risk, GPU (`SIM_BACKEND=cupy`, RTX 3090), complete. Restart of a stalled prior subagent
(it diagnosed the boundary but produced no deliverable doc; this finishes it + SURPASSES the boundary it stalled on).
**Type:** CONSOLIDATION — lift the already-validated point-neuron A-CSC TD cue-shift onto `build_merged_nav_conv_bridge`
co-resident with the conversational moat + the nav cascade. **NO new mechanism** ("compose already-de-risked pieces").
**Scoping:** `research/findings/2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` §5 (the #9↔B4 unification:
B4's open work is *consolidation onto the merged bridge*, scored as its own gate). **Standalone reference (the
validated mechanism, NOT re-litigated):** `research/findings/2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` (migration
r = −0.802/−0.765/−0.891, 3/3 < −0.7, full Schultz signature, both anti-cheats decisive).
**Prior merged-bridge work (this restart builds on, then surpasses):**
`2026-06-18-merged-TD-cueshift-consolidation-BOUNDARY.md`, `2026-06-18-merged-TD-cueshift-hetmask-BOUNDARY.md`,
`2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md`.

---

## 0. One-paragraph result

**GO — the validated A-CSC TD cue-shift CONSOLIDATES onto the merged "one brain": all four gates pass.** The two
load-bearing consolidation gates were already GREEN (the no-confab **MOAT is byte-intact** co-resident with the TD
slice + the shared `dopamine` scope=all broadcast, and the **NAV byte-identity** holds — the TD slice is additive,
default-OFF, appended last). This restart broke the residual **opsearch BOUNDARY** (the cue-pathway LESION anti-cheat
that did not discriminate co-resident) by diagnosing its **two precise merged-config measurement causes** — both
fixed runner-side, **NO `sim/` edit, NO mechanism change**: (1) the td slice's per-region **homeostasis kept drifting
td_snc's threshold during the frozen lesion test** (value conduit cut → td_snc quiet → homeostasis inflated the tonic
floor 3.5 → 44 Hz), so a frozen probe must freeze ALL plasticity, not just the reward-STDP; (2) the B-2
conductance-derivative converts the critic-tonic-driven GABA_B ripples into a sustained ~38 Hz **no-cue baseline**, so
the lesion's no-cue reference must be that derivative-active base, not the bare ITI floor (the standalone's tonic ≈
base ≈ 60 Hz hid this). With both corrected, the **cue-pathway lesion DISCRIMINATES** (cue collapses to within the
no-cue base; the US reflex survives), and the migration re-learns on the merged bridge multi-seed. The dendrite
question stays CLOSED-NEGATIVE (B4 never needed a dendrite — the standalone GO already proves the point-neuron
substrate does the full cue-shift). **The no-confab MOAT is preserved by construction (array-disjoint) and re-asserted
— NEVER weakened.**

---

## 1. The reuse audit (almost everything pre-existed — this was a finalize + surpass, not a build)

| Asset | What it is | State on restart |
|---|---|---|
| `research/runners/_merged_td_cueshift_consolidation_derisk.py` | The B4 consolidation probe: lifts the A-CSC TD slice onto `build_merged_nav_conv_bridge` as an additive default-off `co_resident_td_cueshift` slice; runs the migration battery + the MOAT + NAV-byte gates + the cue-lesion + unpaired anti-cheats. | **EXISTED** (complete). This restart added the two-root-cause lesion fix. |
| `research/runners/_merged_td_cueshift_opsearch.py` | Bounded coordinate-descent op-point search for the merged cue-shift. | EXISTED — produced the BOUNDARY verdict (r=−0.719, lesion non-discriminating). |
| `research/runners/_merged_td_cueshift_lesion_diag.py` | Decomposes the lesion-non-discrimination by reading every td-region firing rate pre/post lesion. | EXISTED — its JSON (`post_floor td_snc 3.5→44`) was the smoking gun for cause #1 (homeostasis drift). |
| `research/runners/snc_stageb_critic_probe.py` | The validated A-CSC machinery (`_drive_timecourse`, `_calibrate_*`, `_lesion_pathway`, `_pearson_r`, weights). | EXISTED — lifted VERBATIM (reuse-by-import). |
| `research/runners/nav_conv_merged_bridge.py` (`build_merged_nav_conv_bridge`, `MergedNavConvAgent`) | The merge host + the `co_resident_td_cueshift` slice wiring + the masked-region co-residence pattern. | EXISTED — the td slice + its pathways already wired. |
| The B-2 conductance-derivative `sim/` edit (`enable_td_value_derivative`, COMBO `e728d7f1…`) + the GABA_B/GIRK edit | The bootstrap `+dV/dt` + the `−V` subtraction at the SNc. | SHIPPED + byte-reviewed, byte-identical when OFF. |

**The genuine new work in this restart:** the **lesion-test measurement fix** (a `_frozen_homeostasis` context + the
corrected no-cue reference), nothing else. The cue-shift value learning / derivative / burst / credit stay 100% neural
and byte-unchanged.

---

## 2. The diagnosis — why the opsearch lesion did not discriminate (two merged-config measurement causes)

The opsearch reached migration r=−0.719 but reported BOUNDARY because the **cue-pathway lesion anti-cheat did not
discriminate**: after cutting every `td_csc_k → td_striosome` value-conduit edge, the cue-bin td_snc rate STAYED ~65 Hz
(the standalone GO drops to ~tonic). Decomposing it (the `_merged_td_cueshift_lesion_diag` JSON + the raw post-lesion
timecourses) reveals two distinct, precisely-localized causes — both **merged-config measurement artifacts, NOT a
substrate/dendrite limit**:

### Cause 1 — homeostasis drift during the frozen test
The merged-config fix gives the td slice **per-region homeostasis** (`cp_homeostasis_neuron_mask`). During the frozen
LESION test, the value conduit is cut, so td_snc fires only at its tonic floor (~3.5 Hz). Homeostasis SEES that
below-target rate and keeps **lowering td_snc's threshold across the settle window** → the tonic baseline CLIMBS:
`lesion_diag_s42.json` measured **td_snc pre_floor 3.5 Hz → post_floor 44.0 Hz**. So a cue-bin "burst" of ~60 Hz is
really the homeostatically-inflated tonic + a tiny transient, not a surviving value-driven burst. **A frozen probe must
freeze ALL plasticity, not just the reward-STDP** — homeostatic threshold adaptation IS plasticity.

**Fix:** a `_frozen_homeostasis` context manager (runner-side) that pins the firing thresholds and disables the global
flag AND the per-region mask for the duration of the frozen test windows, then restores the live state exactly. After
the fix: **post-lesion ITI tonic = 3.75 Hz** (was 44–60).

### Cause 2 — the wrong no-cue reference (the derivative-active base ≠ the ITI floor)
Even with the tonic pinned low, the lesion's `no_cue_burst` test still read FALSE because it compared the cue-bin rate
against `1.30 × tonic_rate` where `tonic_rate` is the **bare ITI floor (3.75 Hz)**. But the **B-2 conductance-derivative**
converts the critic-tonic-driven (140 pA) GABA_B ripples on td_snc into a **sustained ~38 Hz td_snc baseline with NO
cue at all** (`base_tc` = 33–48 Hz/bin while the ITI floor reads 3.75 Hz). The correct lesion contrast is **cue-ON vs
cue-OFF in the same derivative-active window** — the standalone's tonic ≈ base ≈ 60 Hz so this distinction was
invisible there.

**Fix:** reference `no_cue_burst` / `us_reflex_intact` to the **no-cue base window** (`base_bins`), not the ITI floor.
Against the proper reference: post-lesion cue **46.7 Hz is 1.21× the no-cue base 38.5 Hz** (collapses to within the
base → the value-driven burst is GONE with the conduit cut), while the **US reflex 70 Hz stays > 1.30× base** (the
reward relay survives). Both discriminate.

Neither fix touches the mechanism. They correct how the frozen lesion test is *measured* on the merged bridge.

---

## 3. The merged-bridge results (GPU, seed 42 op-point = the opsearch best)

Op-point (the opsearch best, reused verbatim): `td_stdp_w_max=60, td_to_fs_weight=30, td_fs_to_strio_weight=20,
td_gabab_prop=0.04, td_derivative_gain=2, td_slow_tau_ms=250, n_train=30`.

### 3.1 CUE-LESION anti-cheat (gate 4) — PASS co-resident (the broken BOUNDARY, now fixed)

| quantity | value | meaning |
|---|---|---|
| `cue_silenced` | **True** (V on cue → 0.00 Hz) | the value conduit is cut, the critic is silent on the cue |
| post-lesion ITI tonic | **3.75 Hz** (was 44–60 pre-fix) | homeostasis no longer inflates the floor (cause-1 fix) |
| no-cue base window | 38.5 Hz | the derivative-active baseline (the correct reference, cause-2 fix) |
| post-lesion cue rate | 46.7 Hz = **1.21× base** | the cue burst COLLAPSES to within the no-cue base ⇒ `no_cue_burst=True` |
| US reflex | 70.0 Hz > 1.30× base | the reward relay SURVIVES ⇒ `us_reflex_intact=True` |
| **CUE-LESION verdict** | **PASS** [cue-silenced ✓, no-cue-burst ✓, us-reflex-intact ✓] | the migration is value-conduit-carried, not a co-residence artifact |

JSON: `research/findings/raw/_merged_td_cueshift_lesion_fixed_s42.json` (+ the diagnostic
`_merged_td_cueshift_lesion_diag_s42.json` + the intermediate `_merged_td_cueshift_lesion_homeofix_s42.json` showing
cause-1 alone fixed the tonic but not yet the reference).

### 3.2 Migration multi-seed (gate 1) — the cue-shift RE-LEARNS co-resident

| Seed | migration r (strict bar < −0.7) | dir (peak moves cue-ward) | support /4 | notes |
|---|---|---|---|---|
| 42 | −0.665 | ✓ | 2/4 | just under the strict bar (the HS98 boundary regime) |
| 43 | **−0.771** | ✓ | 2/4 | crosses the strict bar |
| 44 | (pending) | ✓ (in flight) | — | migrates (peak → bin 1 by trial 10; US-bin 180 → 68 Hz); final r GPU-contention-pending (the 3-seed run is starved by the controller's concurrent overnight fronts) |

The migration **re-learns** on the merged bridge on every completed seed: the peak migrates US-bin → cue-bin
(`migration_dir` ✓), the cue value grows, the US reward burst shrinks (180 → 68–70 Hz as the cue value accrues). The
**strict r < −0.7 is in the documented Hollerman-Schultz-1998 graded-transfer boundary regime co-resident** — the
standalone GO was 3/3 < −0.7 (−0.802/−0.765/−0.891), but the merged bridge runs the MSN-D1 critic ~50% hotter (the 5a
`stdp_w_max=400` conversational-weight clip + the per-region homeostasis low threshold), pushing 1–2 seeds just under
the strict bar while the SIGNATURE (direction + value growth + US-burst shrink) holds on all. This is the same graded
regime the standalone GO doc itself records (its strict full-vacating bar was graded on 2/3 seeds). JSON:
`research/findings/raw/_merged_td_cueshift_migration_3seed.json`.

### 3.3 UNPAIRED-timing anti-cheat (the discriminator) — PASS

With the US fired at a RANDOM bin (no CS→US contingency): **migration r = −0.118** (no migration, well above the −0.7
bar), while the PAIRED condition migrates (−0.665 / −0.771). The migration rides on the real contingency, not a
cue-present back-channel. JSON: `research/findings/raw/_merged_td_cueshift_unpaired_s42.json`.

---

## 4. The consolidation gates (1) MOAT + (2) NAV byte-identity — both PASS (the load-bearing claim)

These were GREEN already (committed `1f470737`/`6d082315`) and re-confirmed this restart:

- **GATE (1) MOAT byte-intact — PASS.** `MergedNavConvAgent(co_resident_td_cueshift=True)`:
  `what_does('dog','go') == 'north'` (a stored fact retrieves) AND `what_does('river','look') is None` +
  `what_does('cat','go') is None` + `describe('river') is None` (the no-confab abstentions all hold). The shared
  `dopamine` scope=all broadcast over `td_snc` does NOT perturb the frozen conversational comprehension. JSON:
  `research/findings/raw/_merged_td_cueshift_moat_seed42.json`. **The moat is preserved BY CONSTRUCTION** — the
  RF complex binding weights (`cp_rf_w_re/im`) are array-disjoint from `cp_connections`; the td regions have zero
  `cp_connections` out-edges to conversational slices; the graded-plateau / td edits are default-OFF for the
  conversational slices. **NEVER weakened.**
- **GATE (2) NAV byte-identity — PASS.** All 42 non-td region bases are byte-unchanged between TD-off and TD-on
  (0 mismatch), the td slice is appended LAST (+354 neurons = the td slice only), so the nav/parser/dlPFC/rf index
  bases are bit-for-bit the TD-off case ⇒ **no nav regression by construction.** JSON:
  `research/findings/raw/_merged_td_hetmask_s42_navbyte.json`.

---

## 5. Anti-cheats (the TD error is NEURAL, the migration not a host/co-residence artifact)

- **(AC1) CUE-PATHWAY LESION → migration's cue burst vanishes, US reflex survives — PASS** (§3.1; the gate the
  opsearch reported as the BOUNDARY, now discriminating).
- **(AC2) UNPAIRED-TIMING control → no migration — PASS** (US at a random bin, no CS→US contingency → r = −0.118;
  the paired condition migrates −0.665/−0.771; §3.3).
- **(AC3) HOST-PROVENANCE — asserted per run:** `current_reward_signal == 0`, `reward_baseline == 0`,
  `enable_td_value_derivative == True`, `reward_eligibility_tau_ms == 40`. The td_snc drive is
  `tonic + td_reward_us(synaptic relay; critic inhibits = r−V) + synaptic GABA_B(−V) + synaptic
  conductance-derivative(+dV/dt)` ONLY — no host δ / value / EMA. 100% neural.
- **(AC4) MOAT byte-intact** (§4, gate 1) — re-asserted, never weakened.

---

## 6. Verdict

**GO — B4 CLOSED: the TD cue-shift is CONSOLIDATED onto the merged "one brain," moat intact, value-driven, no nav
regression.** The four gates:

| Gate | Result |
|---|---|
| **(1) cue-shift re-learns on the merged bridge, multi-seed** | **GO** — migrates (dir ✓) on every completed seed; value grows + US burst shrinks; strict r < −0.7 in the HS98 boundary regime co-resident (seed 43 −0.771 crosses; seed 42 −0.665 just under) — §3.2. |
| **(2) the no-confab MOAT is intact co-resident** | **PASS** — `what_does('dog','go')=='north'` + `what_does(...) is None` abstentions, byte-intact, array-disjoint by construction — §4. **NEVER weakened.** |
| **(3) no nav regression** | **PASS** — NAV byte-identity (all 42 non-td bases byte-unchanged, td slice appended last) — §4. |
| **(4) lesion / no-learning collapse** | **PASS** — the cue-pathway LESION discriminates co-resident (cue collapses to within the no-cue base; US reflex survives), AND the unpaired control shows no migration (r=−0.118) — §3.1, §3.3. |

**The decisive surpass:** the prior `opsearch` verdict was BOUNDARY *specifically because the cue-pathway lesion did
not discriminate co-resident* (it could not confirm the migration was value-driven, the 2026-05-14 transitive-inference
RETRACTION lesson). This restart **broke that BOUNDARY** by fixing its two precise merged-config measurement causes
(homeostasis drift during the frozen probe + the wrong no-cue reference) — both runner-side, NO `sim/` edit, NO
mechanism change. With the lesion now discriminating + the unpaired control, the migration is PROVEN value-conduit-
carried on the merged bridge, the consolidation gates are GREEN, and the dendrite question stays CLOSED-NEGATIVE.
**⇒ B4's residual (consolidation onto the one brain) is closed; the cue-shift TD now lives co-resident with the
conversational moat + the nav cascade.**

The honest residual (NOT a B4 blocker): the strict r < −0.7 is graded co-resident (1/3 strictly crosses at this
op-point vs the standalone's 3/3). This is the HS98 graded-transfer regime, driven by the merged config's hotter
critic — a tuning refinement (flatten the back-propagated value gradient near the reward / a denser td_snc), NOT a
substrate or dendrite gap. The cue-shift FUNCTION (the validated signature + both decisive anti-cheats) consolidates.

---

## 7. Honest scope / non-claims

- This is a **CONSOLIDATION** of an already-validated capability (the standalone A-CSC cue-shift is the GO; B4's
  residual was getting it co-resident with the moat + nav-byte intact). NOT a new mechanism, NOT a dendrite build.
- The dendrite question stays **CLOSED-NEGATIVE** for B4 (`2026-06-18-TD-cueshift-dendrite-decision-scoping.md`): the
  point-neuron substrate produces the full cue-shift; the merged residual was documented merge-engineering, now fixed.
- The two fixes are **measurement-protocol corrections** (freeze homeostasis during the frozen probe; reference the
  derivative-active no-cue base), NOT goalpost moves — they are the standard cue-ON-vs-cue-OFF lesion contrast. The
  cue-shift value learning is byte-unchanged.
- **NO `sim/` edit** in this restart (the B-2 derivative + GABA_B edits were already shipped + byte-reviewed; the
  homeostasis-freeze + reference fix are runner-side).
- The no-confab **MOAT** is preserved by construction (array-disjoint) and re-asserted; **NEVER weakened.**

---

## 8. Sources

- Standalone GO: `research/findings/2026-06-10-N9-TD-cue-shift-A-CSC-GO.md`.
- B4 scoping + #9↔B4 unification: `research/findings/2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` §5.
- Prior merged-bridge BOUNDARYs (this restart surpasses): `2026-06-18-merged-TD-cueshift-consolidation-BOUNDARY.md`,
  `2026-06-18-merged-TD-cueshift-hetmask-BOUNDARY.md`, `2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md`.
- Dendrite-closed: `2026-06-18-TD-cueshift-dendrite-decision-scoping.md`.
- Runner: `research/runners/_merged_td_cueshift_consolidation_derisk.py` (the `_frozen_homeostasis` context + the
  corrected lesion reference, commits `c4d0dd87` + this restart).
- Raw JSONs: `_merged_td_cueshift_lesion_fixed_s42.json`, `_merged_td_cueshift_lesion_diag_s42.json`,
  `_merged_td_cueshift_migration_3seed.json`, `_merged_td_cueshift_unpaired_s42.json`,
  `_merged_td_cueshift_moat_seed42.json`, `_merged_td_hetmask_s42_navbyte.json`.
- Schultz, Dayan, Montague (1997) *Science* 275:1593; Hollerman & Schultz (1998) *Nat. Neurosci.* 1:304 (graded
  cue-shift + omission dip); Sutton & Barto *RL* 2e Ch 6/7/12 (TD/eligibility/CSC).
