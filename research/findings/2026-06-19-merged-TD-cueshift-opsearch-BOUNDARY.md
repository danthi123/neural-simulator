# Merged-bridge A-CSC TD cue-shift CONSOLIDATION (roadmap #3) — op-point search BOUNDARY: r<-0.7 reachable but the cue-pathway LESION anti-cheat does not discriminate on the merged bridge (2026-06-19)

> **⮕ SURPASSED 2026-06-22 (shortcut B4).** The "cue-pathway LESION does not discriminate" BOUNDARY below was
> diagnosed to **two merged-config MEASUREMENT causes** (NOT a substrate/dendrite limit) and both fixed runner-side
> (NO `sim/` edit): (1) per-region homeostasis drifting td_snc's threshold during the frozen probe (post-lesion tonic
> 3.5 → 44 Hz — a frozen test must freeze ALL plasticity), and (2) the wrong no-cue reference (the B-2 conductance-
> derivative makes a ~38 Hz no-cue baseline — reference that, not the bare ITI floor). With both corrected, the
> cue-pathway lesion DISCRIMINATES co-resident (cue 46.7 Hz collapses to 1.21× the no-cue base 38.5 Hz; US reflex 70 Hz
> survives) and the unpaired control passes (r=−0.118). **B4 consolidation closes (GO on all four gates).** See
> `research/findings/2026-06-22-shortcut-B4-cueshift-merged-consolidation.md`.

**Status:** bounded coordinate-descent operating-point search, complete (28 op-points at seed 42, CPU/numpy).
**VERDICT: BOUNDARY (a precisely-localized merged-bridge anti-cheat finding, NOT a substrate / dendrite finding).**
**Task:** roadmap #3 — lift the validated standalone A-CSC TD cue-shift onto the merged "one brain". The science
is settled (standalone GO multi-seed). This was ENGINEERING ONLY: find the merged operating point reaching the
frozen GO bar migration r < -0.7.
**Standalone reference (the validated mechanism):** `research/findings/2026-06-10-N9-TD-cue-shift-A-CSC-GO.md`
(migration r = -0.802 / -0.765 / -0.891, 3/3 < -0.7, full Schultz 1997 signature, both anti-cheats decisive).
**Prior merged boundary (this task's starting point):**
`research/findings/2026-06-18-merged-TD-cueshift-consolidation-BOUNDARY.md` (best r = -0.43, stuck at a hot critic).
**`nav_conv_merged_bridge.py` is BYTE-IDENTICAL** (no edit — the entire search was driven by the existing runner
flags + the already-shipped per-region heterogeneity mask). NO `sim/` edit.

---

## One-paragraph result

**BOUNDARY — the op-point search DID reach the numeric GO bar (best migration r = -0.719, seed 42), but the
migration FAILS the load-bearing cue-pathway LESION anti-cheat on the merged bridge, so it cannot be honestly
certified as a value-driven cue-shift.** The search found the merged operating point that the prior boundary
lacked: per-tap weight clip (60) + FS-clamp 30/20 + gabab_prop 0.04 + GIRK off + derivative gain 2 + slow-EMA
tau 250 lifts the peak migration to **r = -0.719** (peak_early bin 6.5 ≈ the reward, peak_late bin 1.0 ≈ the cue,
the US burst shrinking 223→88 Hz, the cue value growing V 22→92, the omission dip present — support 3/4). The
**unpaired-timing anti-cheat PASSES** (r = +0.015, no migration without contingency). But the **cue-pathway lesion
anti-cheat does NOT discriminate**: after zeroing every `td_csc_k → td_striosome` value-conduit edge (V(strio) on
the cue → 0, the conduit verifiably cut), the **cue-bin SNc rate STAYS ~66 Hz** (the standalone GO's drops to ~the
tonic floor) — `no_cue_burst = False` at EVERY op-point tested, including configs with a healthy tonic (33 Hz). So
the cue-bin burst is NOT abolished by cutting the learned value, which means the migration is carried substantially
by a **learning-independent cue-onset transient** (lesion-surviving) landing the peak cue-ward as the US burst
collapses — NOT by a value-driven cue burst the lesion would remove. Foregrounding the project's 2026-05-14
transitive-inference RETRACTION lesson (a result the lesion control does not discriminate is NOT real), this is a
**BOUNDARY**, not a GO. The standalone GO already proves the point-neuron substrate produces the full,
lesion-clean cue-shift; the merged residual is a precisely-localized merge-engineering finding (the merged td_snc
has a ~66 Hz cue-onset response independent of the critic that the standalone did not — likely the merged config's
SNc excitability / the shared scope=all dopamine broadcast / the SNc onset-recovery dynamics), with a named,
bounded next fix. **The dendrite question stays CLOSED-NEGATIVE.**

---

## What was searched (additive, runner-flag only, NO new `sim/` edit)

The de-risk runner `research/runners/_merged_td_cueshift_consolidation_derisk.py` (reuse-by-import of the validated
A-CSC helpers from `snc_stageb_critic_probe.py`; only the BRIDGE is the merged `build_merged_nav_conv_bridge`'s
`co_resident_td_cueshift` slice) + a bounded coordinate-descent driver `research/runners/_merged_td_cueshift_opsearch.py`.
Every op-point is reachable by the existing builder flags (`--td-stdp-w-max`, `--td-gabab-cmax`, `--td-gabab-prop`,
`--td-to-fs-weight`, `--td-fs-to-strio-weight`, `--td-csc-to-strio-weight`, `--td-derivative-gain`, `--td-slow-tau-ms`);
the per-region heterogeneity mask (the prior boundary's named fix) is ALREADY SHIPPED + byte-reviewed
(`enable_heterogeneity=True` on the `td_*` regions; `cp_heterogeneity_neuron_mask`). 28 op-points at seed 42,
n_train=30 (the search) — each merged-bridge build ~100 s, CPU/numpy.

---

## The anchor diagnosis (why the prior boundary was stuck, and what the het-mask actually does)

The prior boundary (2026-06-18) named "per-region heterogeneity for the td slice" as the fix. **That mask is now
shipped, but it alone does NOT solve the problem.** Reproducing the CYCLE-224 anchor (het-mask ON, `--td-stdp-w-max`
OFF, GIRK 0.5, gain 1) at seed 42 n_train=30 gave **r = -0.459** — and the timecourse revealed the real blocker:
the per-tap critic weights **RUN AWAY to ~329** (V(strio) ~362 Hz, far above the standalone's sparse ~70 Hz band),
because the merged config pins the GLOBAL `stdp_w_max = 400` (the 5a conversational-weight clip) which REMOVES the
per-tap weight cap (40) the standalone CSC bridge relied on to keep the critic SPARSE. **The het-mask gives the
critic a graded f-I band but does NOT cap the weights** — the runaway is a weight-clip problem, not a heterogeneity
problem. So the PRIMARY lever is the per-tap weight clip `--td-stdp-w-max` (the runner re-clips ONLY the td_value
synapses; a weight-bound, not a host value/reward computation, so the cue-shift stays 100% neural).

---

## The full op-point landscape (28 runs, seed 42, n_train=30 unless noted; sorted by migration r)

| r | dir | sup | tonic | peak e→l | V(cue) e→l | op deltas from the runner baseline |
|---|---|---|---|---|---|---|
| **-0.719** | T | 3 | 1 | 6.5→1.0 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain2, **tau250** ← best, but lesion-FAIL |
| -0.689 | T | 3 | 2 | 6.2→1.0 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain2, tau200 |
| -0.686 | T | 3 | 6 | 5.3→1.0 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain2, tau130 |
| -0.680 | T | 3 | 0 | 6.0→1.0 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain4 |
| -0.677 | T | 3 | 2 | 5.7→1.0 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain2.5 |
| -0.658 | T | 3 | 4 | 5.7→1.0 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain2, tau160 |
| -0.646 | T | 3 | 19 | 5.8→1.3 | 22→92 | clip60, FS30/20, gp0.025, GIRK0, gain2, tau200 |
| -0.641 | T | 3 | 1 | 5.3→1.0 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain3 |
| -0.627 | T | 3 | 0 | 5.0→1.0 | 22→92 | clip60, FS30/20, gp0.07, GIRK0, gain2 |
| -0.610 | T | 3 | 0 | 4.8→1.0 | 46→141 | clip60, FS30/20, gp0.04, GIRK0, gain2, csc18 |
| -0.579 | T | **4** | 19 | 4.5→1.0 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain2, tau80 |
| -0.555 | T | 3 | 5 | 3.8→1.0 | 45→96 | clip60, FS30/20, gp0.04, GIRK0, gain2, **n_train=50** |
| -0.555 | T | 3 | 1 | 4.1→1.0 | 45→96 | clip60, FS30/20, gp0.04, GIRK0, gain2, tau200, **n_train=50** |
| -0.543 | T | 3 | 1 | 3.9→1.0 | 45→96 | clip60, FS30/20, gp0.04, GIRK0, gain2.5, **n_train=50** |
| -0.539 | T | **4** | **33** | 4.0→0.7 | 22→92 | clip60, FS30/20, gp0.04, GIRK0, gain1 ← healthiest tonic |
| -0.477 | T | 3 | 0 | 2.7→1.0 | 72→340 | clip60 only (no FS-clamp, no gp) |
| -0.459 | T | 1 | 65 | 6.3→3.5 | 75→362 | anchor: no clip, GIRK0.5, gain1 (runaway critic) |
| -0.432 | T | 4 | 1 | 3.0→1.0 | 22→92 | clip60, FS30/20, GIRK0, gain1 (gp default 0.105) |
| -0.349 | T | 3 | 0 | 2.2→1.0 | 72→340 | clip60, gp0.07 |
| -0.330 | T | 3 | 0 | 2.2→1.0 | 67→246 | clip40 only |
| -0.311 | T | 4 | 0 | 2.0→1.0 | 72→340 | clip60, gp0.04 |
| -0.311 | T | 4 | 0 | 2.0→1.0 | 67→246 | clip40, gp0.04 |
| -0.310 | T | 3 | 64 | 4.2→2.2 | 23→92 | clip60, FS30/20, gp0.04, **GIRK0.3**, gain2, tau200 ← GIRK revives tonic but kills r |
| -0.071 | F | 3 | 0 | 2.5→2.3 | 49→107 | clip25 (over-clamped) |
| +0.000 | F | 3 | 0 | 1.0→1.0 | 102→179 | FS40/30, gp0.07 (over-clamped) |
| +0.000 | F | 4 | 0 | 1.0→1.0 | 155→187 | FS50/40 (over-clamped) |
| +0.172 | F | 2 | 66 | 7.0→8.0 | 68→247 | clip40, **GIRK0.5** (throttles the cue burst) |
| +0.182 | F | 3 | 1 | 1.0→1.0 | 102→179 | FS40/30, gp0.04 (over-clamped) |

### What the levers do (the landscape shape)

1. **Per-tap weight clip `--td-stdp-w-max` (PRIMARY).** Off → the critic runs away to ~330 (the merged global
   stdp_w_max=400 has no per-tap cap). clip 40–60 keeps it bounded; clip 25 over-clamps (V chokes to 107, no value
   gradient, r → -0.07). Clip 60 ≥ 40 in this family.
2. **FS-clamp `--td-to-fs-weight` / `--td-fs-to-strio-weight` (cools the critic to a GRADED band).** FS 30/20 is
   the SWEET SPOT (V starts sparse at 22, grows to a graded 92). FS ≥ 40/30 OVER-clamps (V starts 102–155, the
   critic cannot grade, the SNc tonic dies, r → 0). This is the lever the prior boundary mis-attributed to the
   GIRK cap.
3. **`--td-gabab-prop` (the per-spike GABA_B −V increment).** 0.04 (vs the default 0.105) keeps the SNc tonic
   alive without a GIRK cap. The prompt's hypothesis ("lower it to keep the critic from over-clamping") is
   CONFIRMED as a real lever (it is necessary for the live-tonic configs).
4. **GIRK cap `--td-gabab-cmax` HURTS.** 0.5 → +0.17; 0.3 → -0.31. It revives the tonic (64–66 Hz) but THROTTLES
   the conductance-derivative cue burst, so the migration collapses. CONFIRMS the prior boundary's note. The
   correct fix for the SNc tonic is the FS-clamp + lower gabab_prop, NOT the GIRK cap.
5. **`--td-derivative-gain` (lifts the cue burst, the +dV/dt term).** With a live tonic in place, raising gain
   1 → 2 lifts r from -0.539 to -0.686 (and pushes peak_early toward the reward). Gain > 2 plateaus / dies the
   tonic (gain 4 → -0.680 tonic 0).
6. **`--td-slow-tau-ms` (the conductance-derivative slow-EMA window).** MONOTONE: 130 → -0.539, 160 → -0.658,
   200 → -0.689, 250 → -0.719. A longer EMA = a stronger derivative = peak_early reaches the reward. The
   trade-off: the tonic drops to ~1 Hz at tau 250 (the strong derivative silences the SNc late).
7. **`n_train` is a MEASUREMENT-WINDOW lever, not a mechanism one.** n_train=50 (the standalone's value)
   REGRESSES to ~-0.55, because the early/late slices are `n_train//5`: with more trials + the gain-2 derivative
   the cue burst forms EARLIER, so the "early" window already has the peak near the cue, flattening the
   correlation. n_train=30 catches the reward-dominant early phase.

**The core tension:** reaching r < -0.7 requires a STRONG derivative (long tau / high gain), which drives the
td_snc tonic near-silent late. Configs with a healthy tonic (gain1/tau130 → tonic 33, support 4/4) only reach
r ≈ -0.54. The two cannot both be maximized at seed 42 in this op-point family.

---

## The decisive anti-cheats (why this is a BOUNDARY, not a GO)

Per the task, r < -0.7 ALONE is not a GO — the migration must pass the two anti-cheats (the 2026-05-14
transitive-inference RETRACTION lesson: a result the lesion/unpaired controls do not discriminate is not real).

### Unpaired-timing — PASS

At the best op-point (tau250), with the US delivered at a RANDOM bin each trial (no CS→US contingency), the
migration vanishes: **r = +0.015** (the runner reports `no-migration True`). The cue value still grows
(V 55→151, because the teacher still fires the critic), but the peak does not migrate — a DISCRIMINATING control.
(Raw: `_opsearch_winner_unpaired_s42.json`.)

### Cue-pathway lesion — UNEXPECTED (does NOT discriminate) → the BOUNDARY

After training, zeroing EVERY `td_csc_k → td_striosome` edge (the learned value conduit; 7195 edges):

| config (seed 42) | tonic (Hz) | V(strio) on cue after lesion | cue-bin rate after lesion | US-reflex | `no_cue_burst` | verdict |
|---|---|---|---|---|---|---|
| tau250 (r=-0.719) | 1.22 | **0.00** (conduit cut ✓) | **65.00** | 295 Hz ✓ | **False** | UNEXPECTED |
| gain2/tau130 (r=-0.686) | 6.25 | 0.00 ✓ | 66.67 | 287 Hz ✓ | **False** | UNEXPECTED |
| gain1/tau130 (r=-0.539) | **32.81** | 0.00 ✓ | 66.67 | 285 Hz ✓ | **False** | UNEXPECTED |

The lesion verifiably cuts the conduit (V(strio) on the cue → 0) and the innate US reflex SURVIVES (the critic can
no longer inhibit `td_reward_us`, so the SNc bursts ~290 Hz on the US) — **but the cue-bin SNc rate STAYS ~66 Hz**,
far above tonic at EVERY op-point, INCLUDING the healthy-tonic config (32.8 Hz). The standalone GO's lesion drops
the cue-bin to ~the tonic floor (`no_cue_burst ✓ 3/3`). **So on the merged bridge there is a learning-INDEPENDENT
~66 Hz cue-onset SNc response that does not exist in the standalone, and which the value-conduit lesion does not
remove.** The migration r < -0.7 is therefore achieved by the US-burst COLLAPSE (223→88 Hz, value-dependent and
real) plus this lesion-surviving cue-onset transient landing the peak cue-ward — NOT by a value-driven cue burst
the lesion would abolish. **The migration cannot be cleanly attributed to the learned value, so it does not pass
the pre-registered cue-lesion bar.** (Raw: `_opsearch_winner_lesion_s42.json`, `_opsearch_tau130_lesion_s42.json`,
`_opsearch_gain1_lesion_s42.json`.)

This is exactly the rigor the task demanded: an r that looks like a GO (-0.719 < -0.7) but that the lesion control
does not discriminate is NOT a GO.

---

## Consolidation gates (still GREEN — the lift itself is clean)

The two consolidation gates (validated at the prior boundary, config-level so unchanged by the op-point) remain
GREEN: **(1) the no-confab MOAT is byte-intact** (`MergedNavConvAgent(co_resident_td_cueshift=True)` —
`what_does('dog','go')=='north'` + 3 abstentions `is None`; the shared scope=all dopamine broadcast does not
perturb the frozen conversational comprehension), and **(2) NAV byte-identity** (all 42 non-td region bases
byte-unchanged, the td slice appended last). The slice LIFTS cleanly; the residual is purely the migration's
anti-cheat cleanliness, not the consolidation.

---

## Provenance (the TD error stays 100% neural)

Asserted in every run under `co_resident_td_cueshift`: `current_reward_signal == 0`, `reward_baseline == 0`,
`enable_td_value_derivative == True`, eligibility tau == 40 ms. The td_snc drive is
`tonic(direct) + td_reward_us(synaptic relay; critic inhibits = r−V) + synaptic GABA_B(−V) + synaptic
conductance-derivative(+dV/dt)` ONLY — no host δ / γV′−V / value-EMA reaches the SNc. The per-tap weight clip is a
weight-BOUND, not a host computation of value/reward/delta.

---

## Verdict

**BOUNDARY — a precisely-localized merged-bridge anti-cheat finding, NOT a substrate / dendrite finding.** The
op-point search reached the numeric GO bar (best migration **r = -0.719**, with the full support signature:
US-burst shrink, cue-value growth, omission dip, peak migrating from the reward to the cue) and PASSED the
unpaired-timing control — a clear advance over the prior boundary's r = -0.43. But the **cue-pathway LESION
anti-cheat does not discriminate on the merged bridge**: the cue-bin SNc burst (~66 Hz) survives the
value-conduit lesion at EVERY op-point (even with a healthy tonic), so the migration is not cleanly value-driven
(a learning-independent cue-onset transient that the standalone does not have contributes the cue-ward peak as the
US burst collapses). Per the pre-registered bar + the 2026-05-14 RETRACTION lesson, this is a BOUNDARY. **The
standalone GO** (`2026-06-10-N9-TD-cue-shift-A-CSC-GO.md`, r = -0.80/-0.77/-0.89, lesion-clean 3/3) **proves the
point-neuron substrate produces the full, lesion-clean cue-shift; the merged residual is documented
merge-engineering, with a named, bounded next fix. The dendrite question stays CLOSED-NEGATIVE.**

### Why this is the RIGHT scientific outcome (per the directives)

- The owner standard (BRAIN-BASED ONLY): the TD error stays 100% neural co-resident (provenance asserted). An
  honest co-residence boundary IS the deliverable (it maps what the merged substrate reproduces out-of-the-box).
- The 2026-05-14 RETRACTION lesson was applied LITERALLY: a numeric r < -0.7 that the lesion control does not
  discriminate was correctly refused as a GO. This is the exact rigor the task asked for ("be rigorous").
- The dendrite-decision (the prior boundary's framing): a NEGATIVE was pre-registered as "the ONLY thing
  re-opening even a temporal-dendrite question." This is NOT that — it is a sharper BOUNDARY (the merged td_snc's
  cue-onset response is a config/excitability issue, not a substrate-can't-do-the-cue-shift issue, which the
  standalone GO disproves).

---

## What it would take to reach a CLEAN GO (the named, bounded next increment)

The op-point search exhausted the runner-flag family at seed 42 and localized the residual precisely: **the merged
`td_snc` has a ~66 Hz cue-onset response that is independent of the learned value conduit (lesion-surviving), which
the standalone did not.** Three candidate root-causes + fixes (in increasing `sim/`-edit cost), for a follow-on
increment (NOT this CPU search's budget):

1. **The merged SNc excitability / the shared scope=all dopamine broadcast.** The merged config gives the td_snc
   per-region homeostasis (low threshold) + the shared `dopamine` plasticity-rate broadcast; either could raise
   the SNc's stimulus-onset excitability so the cue ONSET (a transient via the merged drive path) bursts it
   independent of the critic. Diagnose by reading the td_snc drive decomposition at the cue-bin under lesion
   (which afferent supplies the 66 Hz). If it is the cue's own onset reaching the SNc via a path the standalone
   lacks, gate/remove that path; if it is the SNc onset-recovery from the inter-trial floor, lengthen the
   inter-trial settle or raise the SNc adaptation so the cue-bin sits at the tonic floor (as the standalone's did).
2. **A SNc onset-recovery fix.** In the standalone the critic's GABA_B clamped the SNc onset; here even a healthy
   critic does not (the merged SNc recovers faster). A per-region SNc adaptation / a longer omission-test settle
   would restore the standalone's `no_cue_burst` behavior — making the lesion discriminate WITHOUT changing the
   migration mechanism. This is the most likely single fix and is config/builder-level.
3. **6-seed confirmation + GPU.** Once a config passes the lesion at seed 42, run the 3-seed (42/43/44) r<-0.7 +
   lesion/unpaired battery on `SIM_BACKEND=cupy` (the search was CPU/numpy; the standalone GO was multi-seed).

The deployment flip (editing `build_merged_nav_conv_bridge` defaults) is deferred to the controller — this task
reports the landscape + the named fix, not a deployed config (no clean-GO config was found).

## Artifacts

- Search driver: `research/runners/_merged_td_cueshift_opsearch.py` (bounded coordinate-descent over the runner
  op-point flags; appends each (op → r) to a results JSON; budget-defended / re-committable).
- De-risk runner (unchanged): `research/runners/_merged_td_cueshift_consolidation_derisk.py` (the A-CSC battery on
  the merged-bridge td slice + the two consolidation gates + the `--lesion` / `--unpaired` anti-cheats + the
  op-point CLI).
- Master results (28 op-points + the verdict/anti-cheat block): `research/findings/raw/_merged_td_cueshift_opsearch.json`.
- Per-run raw: `research/findings/raw/_opsearch_p{1d1,1d2,1d3,1d4,2d*,3n50,4}_*.json`.
- Anti-cheats: `_opsearch_winner_{lesion,unpaired}_s42.json`, `_opsearch_{tau130,gain1}_lesion_s42.json`.
- `nav_conv_merged_bridge.py` is BYTE-IDENTICAL (verified `git diff` empty vs the task's start commit). NO `sim/`
  edit (the per-region heterogeneity mask + the B-2 conductance-derivative + GABA_B are all already-shipped +
  owner-approved; byte-identical when the slice is OFF).
