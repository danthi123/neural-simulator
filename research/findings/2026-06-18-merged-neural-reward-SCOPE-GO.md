# TRUE-ONE-BRAIN roadmap #2 — NEURAL reward on the merged bridge: SCOPE + cheap-first DE-RISK = **GO**

**Date:** 2026-06-18
**Type:** SCOPE resolution + cheap-first (numpy/CPU) de-risk. Roadmap item #2 of the spike-ification roadmap
(`research/findings/2026-06-18-full-spikeification-shared-substrate-roadmap.md` §3 #2). NO `sim/` edit — runner-only
(additive plumbing through the merged builder + a new de-risk runner).

**Goal of #2:** make the merged "one brain" nav episode source its reward `r` **NEURALLY** (from
`sc_rostral→reward_us` firing — the N5 SC proximity/goal-salience approach reward), retiring the host
Manhattan/sign(Δ-eccentricity) reward formula. Together with the already-committed value-train (learned V,
commit `6fe74bc5`), this makes δ = r − V **fully synaptic** on the one brain.

**Bottom line: GO** to proceed to the route+6-seed build. The spiking SC reward chain **composes co-resident on the
merged bridge** (moat intact), and `sc_rostral→reward_us` **sources a graded proximity reward the SNc bursts on**;
the **lesion collapses it** (the reward IS the synaptic SC proximity, not a re-hidden host scalar). The one nuance
is an **operating-point** finding (the standalone-tuned SC weights fire too weakly het-off — the documented
merge-lift boundary), which is exactly what the 6-seed build tunes.

---

## (1) SCOPE answer — does #2 need the full vision→SC chain, and how big is the build?

**Yes, #2 needs the full vision→SC chain — and it ALREADY EXISTS end-to-end inside `g11_bg_runner`.** Two
load-bearing structural facts:

1. **The SC chain is built by `build_bg_brain_regions(enable_spiking_sc=True, enable_spiking_sc_approach=True,
   spiking_reward_us=True)`** — `sc_retina`(2048) → `sc_map`(256, Mexican-hat via `sc_fs`) → `sc_rostral`(24,
   foveal-centre proximity readout) → `reward_us`(40, PPN-like) → `snc`(DA). The `sc_rostral→reward_us` pathway
   (the neural reward `r`) is declared at `g11_bg_runner.py:2541-2544`. The `sc_map→sc_rostral` proximity readout +
   `retina→sc_map` retinotopy are wired **post-init** by `install_spiking_sc_wiring` (`g11_bg_runner.py:201`,
   line 274-298 for the `sc_map→sc_rostral` foveal Gaussian).

2. **The SC region build is NESTED inside `if enable_visual_cortex:`** — `g11_bg_runner.py:2488` (`if
   enable_spiking_sc:`, indent 8) sits inside `:2428` (`if enable_visual_cortex:`, indent 4). So **the SC chain
   ONLY builds when `enable_visual_cortex=True`** (verified by indentation + an empirical build smoke: with
   `enable_spiking_sc=True` but `enable_visual_cortex=False`, the bridge had 56 populations and `KeyError:
   'sc_retina'`). The CLAUDE.md "Requires enable_visual_cortex" note is literally true at the region-build level.
   ⇒ #2 forwards BOTH flags: the full vision hierarchy (`retina`/V1/V2/`cortex_it`) AND the SC chain. (`cortex_it`
   is also the value-train critic's perceived-state afferent, so this is consistent.) The SC `sc_retina` is its
   OWN egocentric eye, so the SC is self-contained on it; the visual hierarchy is co-built but the SC reward path
   does not depend on the V1→IT pathway firing.

**Build size = SMALL, runner-only.** `run_moving_goal_episode` (the merged nav gate's entry,
`g11_bg_runner.py:3065`) **already accepts** all the needed parameters (`enable_visual_cortex:3493`,
`enable_spiking_sc:3502`, `enable_spiking_sc_approach:3504`, `spiking_reward_us:3258`, `enable_neural_critic:3252`,
`spiking_snc:3237`, `reward_us_drive_pa:3262`) and forwards them to `build_bg_brain_regions` (`:3866`). The SC
retina is driven each nav step at `:6619-6624`. So the episode-loop machinery for the SC reward is **already wired
end-to-end** — #2 is mostly turning the flags on for the merged nav gate.

**The ONE genuine code change** (a runner edit, NOT `sim/`): the deployed reward routing at
`g11_bg_runner.py:7140-7149`. Even with `spiking_reward_us + enable_neural_critic`, the `else` branch
(`:7147-7149`) drives `reward_us` with **`reward_us_drive_pa * max(0, reward)` = the HOST `reward` scalar**
(computed at `:6901-6946`). The `if`-branch that would zero the host write (`:7140`) checks for `"approach_n5"`,
**a region that no longer exists** (dropped per `:2532` — "the earlier slow-channel `sc_rostral_slow/approach_n5`
was dropped"). So the dead `approach_n5` branch must become a live **`sc_rostral`** branch: when
`enable_spiking_sc_approach`, ZERO the host `reward_us` write and let `sc_rostral→reward_us` (firing graded with
proximity from the SC retina) carry the reward. That is the load-bearing one-line-block edit #2 delivers.

---

## (2) COMPOSITION result — does the spiking SC compose on the merged bridge? Moat intact?

**YES, both.** Building `MergedNavConvAgent(co_resident_nav_critic=True, nav_critic_spiking_sc=True)` (the new
additive default-off kwarg, plumbed through `build_merged_nav_conv_bridge` → `build_bg_brain_regions`):

- **Co-residence (no collision):** the merged bridge has **54 regions, 9468 neurons, 915,559 synapses**, with
  `sc_retina / sc_map / sc_fs / sc_rostral / reward_us / snc` ALL present alongside the conversational
  `parse_conj / parse_role / cortex_ctx / dlpfc_wm` AND the nav critic `cortex_N / striosome_value`. No region-name
  or index-slice conflict (the SC regions are appended into the nav block; the parser/dlPFC/composer slices are
  unchanged).
- **No-confab MOAT intact:** on the bridge WITH the SC chain, `agent.hear("dog go north")` then
  `what_does("dog","go")` = **`'north'`** (a known fact resolves) AND `what_does("river","look")` = **`None`** (an
  unstored cue abstains — no confabulation). The shared `dopamine` modulator (threshold-0, neutral-at-rest) does
  not perturb conversation.

The additive kwarg is **byte-preserving**: the plain default build is 42 regions / 2904 neurons (no SC);
`co_resident_nav_critic=True` without the SC flag has `reward_us` present + `sc_retina` absent (== the value-train
build); only `nav_critic_spiking_sc=True` adds the SC chain.

---

## (3) cheap-first DE-RISK result — graded proximity reward? lesion-collapses?

Reusing the **validated** `sc_n5_rpe_probe.py` mechanism (corr(distance, SNc) = −0.99 standalone) but driving the
merged bridge's `sc_retina` with `render_egocentric_goal` at varying agent→goal proximities, measuring the SYNAPTIC
`reward_us` + `snc` firing (the only host input is the SNc tonic pacemaker; `r` flows purely through
`sc_retina → sc_map → sc_rostral → reward_us → snc`):

| agent ecc | SNc Hz (INTACT) | reward_us Hz (INTACT) | SNc Hz (LESION) | reward_us Hz (LESION) |
|---:|---:|---:|---:|---:|
| 7 (far) | 113.3 | 0.0 | 113.3 | 0.0 |
| 6 | 96.7 | 0.0 | 96.7 | 0.0 |
| 5 | 98.3 | 0.0 | 98.3 | 0.0 |
| 4 | 91.7 | 0.0 | 91.7 | 0.0 |
| **2 (close)** | **171.7** | **61.7** | 93.3 | 0.0 |

- **GRADED proximity reward:** corr(ecc, reward_us) = **−0.81** (closer → bigger, the right sign); reward_us
  **61.7 Hz close vs 0 far**. PASS (≤ −0.5).
- **SNc bursts on the close goal:** **171.7 Hz vs 118.3 tonic = 1.45×**, produced SYNAPTICALLY by reward_us
  (not a host write). PASS.
- **LESION collapses it (decisive anti-cheat):** zeroing `sc_rostral→reward_us` → reward_us **61.7 → 0.0 Hz**, the
  SNc burst vanishes (corr flips to **+0.76**, the residual tonic-only response). PASS. This proves the reward IS
  the synaptic SC proximity, not a re-hidden host scalar.

**The operating-point nuance (honest):** at the *standalone-probe* SC weights (`w_ret_sc=80`, `w_sc_rec=6`,
`sc_rostral→reward_us=14`, retina 2500 pA) the chain is **starved het-off**: `sc_map` fires ~2 Hz and `reward_us`
**never crosses threshold** (the documented "standalone-tuned organ fires ~6–10× weaker co-resident, het-off
merged bridge" boundary — `2026-06-18-merged-limbic-core-lift.md`). The de-risk restores it with a **merged-tuned
operating point** (`w_ret_sc=160`, `w_sc_rec=12`, `sc_rostral→reward_us=40`, retina 3500 pA). Even tuned, the
gradient is currently **compressed into the closest bins** (`sc_map` only forms a bump at ecc ≤ ~4 het-off, so
reward_us fires only when close) rather than spanning the full distance range. Spreading the gradient across more
distance bins is the substance of the 6-seed operating-point tuning (the established alternative fix is **per-region
homeostasis on `sc_map`/`sc_rostral`** — the same `nav_critic_homeostasis_mask` mechanism the critic uses; the SC
regions are NOT currently in the homeostasis mask, only `reward_us`/`snc` are).

---

## (4) VERDICT — **GO** to the route + 6-seed build

The mechanism is **de-risked GO** at the cheap-first level:
- COMPOSITION GO (SC chain co-resident on the one brain, moat intact).
- The neural reward is **synaptic, graded by proximity (corr −0.81), and the SNc bursts on it (1.45×)**.
- The **lesion collapses it** (the decisive load-bearing anti-cheat for a *reward*) — single-seed but mechanistic
  (3-clean conclusive per the standing rule for lesion gates).

The only caveat is an **operating-point tuning** (not a substrate wall, not a composition failure): the SC bump is
weak het-off and the gradient compresses into the closest bins. This is the well-trodden merge-lift tuning the
6-seed build addresses. **Scope is CONTAINED (runner-only, no `sim/` edit).** Proceed.

---

## (5) Recommended build path (for the controller) + any sim/ edit needed

**No `sim/` edit needed.** Runner-only, three steps:

1. **(DONE this de-risk)** Plumb `nav_critic_spiking_sc` through `build_merged_nav_conv_bridge` + `MergedNavConvAgent`
   (additive, default-off → forwards `enable_visual_cortex` + `enable_spiking_sc` + `enable_spiking_sc_approach`
   into the `co_resident_nav_critic` `build_bg_brain_regions` call). Byte-preserving on the default path.

2. **The reward-routing edit (`g11_bg_runner.py:7140-7149`)** — make the dead `approach_n5` branch a live
   `sc_rostral` branch: when `enable_spiking_sc_approach` (and the SC retina is being driven), **ZERO the host
   `reward_us` write** (`:7148-7149`) so `reward_us` is driven PURELY by `sc_rostral→reward_us`. (Keep the host
   path as the default/scaffold when `enable_spiking_sc_approach` is OFF — byte-preserved.) This is the line that
   retires the host reward.

3. **The merged nav gate (`nav_conv_merged_bridge.py:nav_on_merged_smoke` / its `run_moving_goal_episode` call)** —
   pass `enable_visual_cortex=True, enable_spiking_sc=True, enable_spiking_sc_approach=True, spiking_reward_us=True,
   enable_neural_critic=True, spiking_snc=True` through to `run_moving_goal_episode` (it already accepts them all).
   Tune the SC operating point for het-off (start from the de-risk's `w_ret_sc=160`/`w_sc_rec=12`/`ros_us=40`/retina
   3500, or add per-region homeostasis on `sc_map`/`sc_rostral`). Validate 6 seeds:
   - the RPE battery (graded reward corr ≤ −0.5, SNc burst ≥ 1.3×, **lesion `sc_rostral→reward_us` collapses it**) —
     the reward is the dependent variable (the N5 lesson: validate a reward by its teaching signal, not the nav A/B);
   - nav-not-regressed (the merged nav score with the neural reward vs the host reward — and the honest negative IF
     it changes nav behavior is itself the deliverable, the orient-solvable caveat);
   - the conversational moat unaffected (the `is None` abstention assertions still pass).

---

## (6) Files + commits

- **`research/runners/_merged_neural_reward_scope_derisk.py`** (NEW) — the cheap-first de-risk runner (composition
  + moat + the graded-proximity RPE battery + the `sc_rostral→reward_us` lesion). `SIM_BACKEND=numpy python
  research/runners/_merged_neural_reward_scope_derisk.py`.
- **`research/runners/nav_conv_merged_bridge.py`** (EDIT, additive default-off) — `nav_critic_spiking_sc` kwarg on
  `build_merged_nav_conv_bridge` + `MergedNavConvAgent` (forwards `enable_visual_cortex`/`enable_spiking_sc`/
  `enable_spiking_sc_approach` into the `co_resident_nav_critic` `build_bg_brain_regions` call).
- **`research/findings/2026-06-18-merged-neural-reward-SCOPE-GO.md`** (NEW, this doc).

Commit SHAs recorded in the commit that ships this doc.
