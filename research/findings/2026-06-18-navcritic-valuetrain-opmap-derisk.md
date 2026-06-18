# De-risk: a NON-SATURATED graded-δ operating point for the MERGED nav critic — GO (the gap was a SILENT critic, not SNc saturation) (2026-06-18, CYCLE 209 follow-on)

Cheap-first numpy/CPU de-risk. Throwaway runners `research/runners/_navcritic_valuetrain_opmap_derisk.py`
+ `_navcritic_msn_fi_probe.py` (NOT committed). Builds the MERGED nav+conv bridge
`build_merged_nav_conv_bridge(seed=42, co_resident_nav_critic=True)` (45 regions, 3224 neurons; the
full nav critic `snc` / `striosome_value` MSN-D1 / `reward_us` US-afferent / `vs_place_context`
dense afferent all co-resident; moat builds clean) and sweeps the SNc operating point in-process.

## 1. VERDICT — GO, and it re-diagnoses the residual the prompt was chasing

**A non-saturated, GRADED-δ operating point EXISTS on the merged nav critic. But the weak ~1.05 gap
the prior finding flagged was NEVER an SNc-saturation problem — it was a SILENT CRITIC.** When the
critic actually fires (driven above its ~600 pA MSN-D1 rheobase, the trained-V proxy), the GABA_B
δ=r−V is strongly graded at every non-saturated SNc point:

| homeostasis mask | SNc tonic | GIRK cap | SNc burst (unpred) | predicted (critic fires) | **gap** | critic Hz |
|---|---|---|---|---|---|---|
| `critic_no_snc` {reward_us,critic} | 120 | 0.0 (uncapped) | 157.5 | 17.5 | **9.0** | 83 |
| `critic_no_snc` | 160 | 0.0 | 165.0 | 25.0 | **6.6** | 83 |
| `critic_no_snc` | 210 | 0.0 | 172.5 | 25.0 | **6.9** | 82 |
| `critic_only` {critic} | 160 | 0.0 | 97.5 | 5.0 | **19.5** | 94 |
| `critic_only` | 120 | 0.0 | 87.5 | 20.0 | **4.4** | 90 |
| `critic_no_snc` | 160 | 1.0 (finite cap) | 162.5 | 112.5 | **1.44** | 86 |

(`predA` / `strioA` = the afferent-driven path — `vs_place_context` at the INIT weight 0.20 — fires
the critic 0.0 Hz, so `gapA ≈ 1.0` everywhere. That is the EXPECTED untrained state: the value-train's
whole job is to grow that weight. `predD` / `strioD` = the critic driven DIRECTLY at 1000 pA = the
"V-learned" proxy.)

**The decisive two-run contrast:**
- **Run 1** (critic driven at 500 pA = SUB-rheobase): critic 0.0 Hz at every point, gap ≈ 1.0
  everywhere — exactly the prior finding's "weak ~1.05" symptom. It looked like SNc saturation
  because the SNc was indeed pinned ~435 Hz under the builder's all-3-masked config, but the gap
  would have been ~1.0 even at a low SNc tonic, because there was no V to subtract.
- **Run 2** (critic driven at 1000 pA = supra-rheobase, the trained-V proxy): gap jumps to 4–19×.
  **The firing critic's GABA_B drops the SNc reward-burst from 157→17 Hz (9×) — that IS δ=r−V on the
  real merged bridge.** The same subtraction would even work under the builder's saturating SNc (run 1
  proved the SNc was never the blocker); the non-saturating masks just give it cleaner headroom.

**The prompt's KEY INSIGHT (homeostasis on the critic but NOT the SNc) works for the SNc f-I exactly
as hypothesized** — it de-saturates the SNc reward-burst 435 → 157–172 Hz (`critic_no_snc`, SNc at
vpeak) → 87–100 Hz (`critic_only`, SNc + reward_us both at vpeak). So if a clean non-saturated SNc
operating point is wanted (it is, for graceful graded RPE), `critic_no_snc` or `critic_only` is the
mask — but it is not *required* for the subtraction to be graded; it is required for the SNc not to
be pinned at its ceiling. **Recommended: `critic_only` mask + SNc tonic ≈ 160 + GIRK cap 0.0** (SNc
burst ~97 Hz non-saturated, critic ~94 Hz, gap ~19; the SNc + reward_us run at vpeak so the f-I is
the standalone-Stage-B-like regime, only the under-active MSN-D1 critic gets the homeostasis boost).

## 2. The MSN-D1 critic f-I (the load-bearing fact, `_navcritic_msn_fi_probe.py`)

The striosome critic is `IZH2007_STRIATAL_MSN_D1` (`vr=−80 mV`, `vt=−25 mV`). Its f-I, with vs
without the per-region homeostasis mask, is **threshold-INSENSITIVE** (the prior `_organ_lift` finding,
lines 40–43, confirmed verbatim):

```
drive   homeo=OFF        homeo=ON
200pA   0.0 Hz (v -70)   0.0 Hz (v -70)
400pA   0.0 Hz (v -57)   0.0 Hz (v -57)
600pA  16.5 Hz (v +39)  17.1 Hz (v -32)   <- rheobase ~600 pA, mask-INDEPENDENT
800pA  60.6 Hz          53.5 Hz
1200   110.6 Hz         104.6 Hz
1800   167.7 Hz         170.0 Hz
```

The homeostasis mask shifts only the post-spike v_peak (+40 vs −31 mV — a cosmetic reset target), NOT
the rheobase: the MSN-D1's rheobase is set by the depolarization needed to reach `vt=−25 mV`, which
the lowered spike threshold does not change. **So the per-region homeostasis is f-I-IRRELEVANT for the
MSN-D1 critic** — the critic fires from synaptic DRIVE (≥600 pA), which only the trained
`vs_place_context→striosome_value` weight (or the up-state arm, or a teacher) supplies. This is the
same wall the prior finding and the 2026-06-08 calibration NEGATIVE both hit; here it is re-confirmed
on the merged bridge and quantified (rheobase ~600 pA; 80–100 Hz at ~1000 pA).

⇒ The critic CAN fire ⇒ **V IS learnable.** The op-map's silent `strioA=0.0` at the init weight 0.20
is not a wall — it is the untrained starting point, and the value-train grows the weight until the
place volley delivers ≥600 pA-equivalent (the runner's validated pipeline does exactly this).

## 3. The value-train plan (the concrete next step to learn V on the merged bridge)

The substrate is sound; the clean δ=r−V is gated on TRAINING the critic weight, not on the
homeostasis enabler. Reuse the `g11_bg_runner` value-train machinery VERBATIM:

1. **Operating point (set on the merged bridge before value-train):** mask = `critic_only`
   (`striosome_value` only; `snc` + `reward_us` at vpeak so the SNc f-I is the non-saturating
   Stage-B regime), SNc tonic ≈ 160 pA, `gabab_conductance_max = 0.0` (uncapped — the subtraction is
   sharpest; the finite cap 1.0 gives a gentler gap ~1.44 if a softer RPE is wanted later). NOTE: the
   merged builder currently masks ALL THREE {snc, reward_us, striosome_value}
   (`nav_conv_merged_bridge.py:525`); to adopt `critic_only`, drop `snc`+`reward_us` from that
   post-hoc set — a one-line builder change, no `sim/` edit.

2. **Critic afferent — the merged path uses `vs_place_context` (dense Gaussian), NOT the `place`
   self-org pool.** So the merged value-train does NOT need the place-code self-org (`_run_place_selforg`,
   `selforg_steps=2000`) — that is the `neural_place_selforg=True` path. The merged
   `co_resident_nav_critic` build is the `enable_neural_critic + spiking_reward_us` (no `selforg`) path,
   whose afferent `vs_place_context` is **drive-injected each step with a grid-32 Gaussian place code**
   (`g11_bg_runner.py:1841-1859`). Two options, in order of cheapness:
   - **(cheapest) direct-afferent value-train:** drive `vs_place_context` with the goal's Gaussian
     place code at the GOAL and a far place code AWAY, open the `value_input` gate, run the
     pair-then-reward DA-gated STDP loop (`_run_place_value_training`, lines 5342-5481 — the protocol
     is afferent-agnostic; it just needs `region_indices_cp["striosome_value"]` + `["snc"]` + the
     afferent drive). The STDP grows `vs_place_context→striosome_value` from 0.20 until the goal-place
     volley fires the critic ≥600 pA-equivalent (≈80 Hz) → V(goal) ≫ V(far). The runner's validated
     `--value-train-trials 40 --value-train-stdp-w-max 40` (the soft-bound that stops the MSN
     saturating, `g11_bg_runner.py:5377-5388`) transfers directly.
   - **(if direct-afferent under-fires)** add the convergent up-state arm (`enable_convergent_upstate`,
     `vs_place_drive→striosome_value`, the dense non-plastic up-state, `g11_bg_runner.py:1860-1871`)
     so the critic is in a location-gated up-state from init and the STDP refines on top — the
     validated nav-deployment recipe.

3. **Reuse VERBATIM from `g11_bg_runner`:** `_run_place_value_training` (the pair-then-reward DA-gated
   STDP loop), `_n9_calibrate_da_threshold` (sets the dopamine rule threshold to the tonic SNc fraction
   so a burst→+LTP gate), `_n9_reset_snc_subtraction_state` (per-trial GABA_B/GIRK reset),
   `value_train_stdp_w_max=40` (the critic soft-bound). The merged bridge already has the `dopamine`
   neuromodulator over `["snc"]` (`nav_conv_merged_bridge.py:699-706`), the GABA_B
   `striosome_value→snc` route, and the `value_input` plasticity gate — all wired by
   `co_resident_nav_critic`.

4. **Then re-measure the δ:** after value-train, the afferent path (`gapA`/`strioA`) should fire the
   critic AT THE GOAL (no direct drive) and the gap should grade with the perceived place — that is the
   real learned δ=r−V, the (b) residual closed. The de-risk shows the substrate delivers it the moment
   the critic fires.

## 4. The key insight tested (per the prompt #3): homeostasis on the CRITIC but NOT the SNc

**Tested — it works for the SNc f-I, and is the recommended op-point lever, BUT it is NOT the fix for
the gap** (the gap fix is firing the critic):
- `critic_no_snc` ({reward_us, striosome_value}; SNc at vpeak): SNc reward-burst 157–172 Hz
  (non-saturated, vs builder 435 Hz). reward_us STILL bursts the SNc from vpeak (the US→DA reflex
  arc does not need the f-I boost — reward_us at vpeak drives ~157 Hz). ✅
- `critic_only` ({striosome_value} only; SNc + reward_us both at vpeak): SNc burst 87–100 Hz (clearly
  non-saturated), critic 90–98 Hz, gap 4–19. **This is the cleanest config** — only the under-active
  MSN-D1 critic gets the homeostasis boost (and even that is f-I-irrelevant to it, §2 — so in practice
  `critic_only` ≈ NO homeostasis mask at all on the critic firing; the critic fires from synaptic
  drive). The SNc stays at its native Stage-B f-I.
- So the prompt's insight is CORRECT — the SNc does not need the f-I boost when `reward_us` drives it;
  only the critic + its afferent matter — but the deeper truth is the homeostasis mask is moot for the
  MSN-D1 anyway; what fires the critic is the trained afferent weight, and what de-saturates the SNc is
  leaving it at vpeak.

## 5. Anti-cheats + honest residuals

- **The graded gap is via the SYNAPTIC GABA_B, not host arithmetic:** the subtraction only appears when
  the critic FIRES (`strioD ~83-98 Hz` → `predD` collapses; `strioA = 0` → `predA = unpred`, no
  subtraction). The δ tracks the critic's spiking output through the `striosome_value→snc`
  `receptor="gaba_b"` route — the same anti-cheat the limbic battery uses. The GABA_B-lesion gate
  (zero `cp_gabab_synapse_mask`) → gap collapses, already validated by
  `_merged_limbic_coresident_validate.validate_arithmetic` for the minimal organ; the full critic
  inherits the same route.
- **The "direct 1000 pA critic drive" is a TRAINED-V PROXY, not the trained weight.** This de-risk does
  NOT run the value-train (per scope); it proves that GIVEN a firing critic, the SNc op point yields a
  graded δ. The actual V-learning (growing `vs_place_context→striosome_value` to ~80 Hz at the goal) is
  the value-train step (§3), separable from this op-map. The f-I probe (§2) confirms the firing regime
  the value-train must reach (≥600 pA / ~80 Hz), which `value_train_stdp_w_max=40` is sized for.
- **The afferent at INIT weight 0.20 fires the critic 0 Hz** — the documented "sparse/weak afferent
  can't fire the MSN at init" boundary, faithfully inherited co-resident. NOT a new wall; it is the
  value-train's premise.
- **Homeostatic threshold-adapt frozen** (`homeostasis_threshold_adapt_rate=0.0`) during the probe, so
  the multi-condition sweep isn't drift-contaminated (the prior finding's `freeze_lr` analogue). OU on
  (sigma 100) to match the pinned op-point regime; learning frozen (`reward_learning_rate=0`).
- **het OFF** (matches the merged bridge): `snc` builds as a correct `IZH2007_DOPAMINE` (bridge log
  confirms), not a jittered RS (the prior finding's latent `_apply_parameter_heterogeneity` bug).
- **Saturation flag honest:** the `builder` all-3-masked config (run 1) pins the SNc ~435 Hz, but that
  did not cause the weak gap (the critic was silent); it would, however, give a binary δ once the
  critic fires (no headroom), so the non-saturating mask is still the right production choice.

## 6. Load-bearing file:line

- `research/runners/nav_conv_merged_bridge.py:517-526` — the `co_resident_nav_critic` build
  (`build_bg_brain_regions(enable_neural_critic=True, spiking_reward_us=True,
  enable_critic_homeostasis=True)` + the post-hoc mask of `snc`/`reward_us`/`striosome_value`); **:525
  is the one line to change** to adopt the `critic_only` mask (drop `snc`,`reward_us`). `:693-706` —
  the GABA_B + `dopamine` modulator over `["snc"]` (transfers verbatim).
- `sim/bridge.py:6318-6325` — the threshold-select (`cp.where(mask, adapted, vpeak)`); branch 2 gives
  masked neurons the low threshold. `:1227-1245` — builds `cp_homeostasis_neuron_mask`. (Confirmed
  f-I-irrelevant for the MSN-D1, §2 — `vt=−25 mV` rheobase, not threshold-set.)
- `research/runners/g11_bg_runner.py:5342-5481` — `_run_place_value_training` (the value-train loop to
  reuse); `:1841-1859` — the `vs_place_context` dense afferent (drive-injected); `:1874-1879` — the
  `vs_place_context→striosome_value` plastic init weight 0.20 the STDP grows; `:1880-1892` — the
  `striosome_value→snc` GABA_B route; `:5377-5388` — `value_train_stdp_w_max=40` (the critic
  soft-bound); `:1227 IZH2007_STRIATAL_MSN_D1` critic / `:1140 IZH2007_DOPAMINE` snc.
- `research/findings/2026-06-18-organ-lift-homeo-generalize-derisk.md:34-56` — the prior "weak ~1.05
  gap / SNc saturates" residual this de-risk re-diagnoses (the gap was the silent critic).

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._navcritic_valuetrain_opmap_derisk --seed 42   # the op-map (run 2: critic driven 1000pA, graded gaps)
SIM_BACKEND=numpy python -m research.runners._navcritic_msn_fi_probe                          # the MSN-D1 f-I (rheobase ~600pA, mask-independent)
```
