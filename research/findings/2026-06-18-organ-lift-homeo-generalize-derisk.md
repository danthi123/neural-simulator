# De-risk: does the per-region homeostasis enabler GENERALIZE from the minimal limbic organ to the FULL nav critic? (2026-06-18, CYCLE 208 follow-on)

Cheap-first numpy/CPU de-risk. Throwaway runner `research/runners/_organ_lift_homeo_generalize_derisk.py`
(NOT committed). Builds the FULL nav critic STANDALONE via
`build_bg_brain_regions(enable_neural_critic=True, spiking_reward_us=True, n_cortex=100)` into a
bare merged-regime config (global `enable_homeostasis=False`, `enable_synaptic_scaling=False`,
`enable_gabab=True` reversal −90/prop 0.22, dt=1.0, Izhikevich, het OFF, the SNc-derived `dopamine`
modulator over `['snc']`) and probes the SNc + critic f-I under three per-region homeostasis configs.

## 1. VERDICT — split, and sharper than the prompt framed it

**(a) The SNc f-I restoration GENERALIZES — clean GO, BUT the existing `enable_critic_homeostasis`
kwarg is INSUFFICIENT; `snc` + `reward_us` need the mask too.**

| homeostasis config (regions masked) | SNc tonic | SNc reward_us-burst | burst ratio |
|---|---|---|---|
| `none` (the broken merged default) | 31.7 Hz | 111.7 Hz | **3.53×** |
| `critic` = the existing kwarg (`vs_place_context`+`striosome_value`) | 31.7 Hz | 111.7 Hz | **3.53×** (byte-identical to none — the kwarg does not touch the SNc) |
| `critic_snc_us` (+ post-hoc `snc`,`reward_us`) | 81.7 Hz | 446.7 Hz | **5.47×** |

The homeostasis enabler reproduces the RESOLVED-finding f-I mechanism on the full nav SNc exactly:
masking `snc`+`reward_us` lifts tonic 31.7→81.7 Hz and the reward burst 111.7→446.7 Hz (the
~−42 mV homeostatic threshold vs +40 mV `vpeak` = the large DA-cell f-I gain). Verified the mask is
applied: `striosome_value` thr=−41.4 mV, `snc` thr=−42.6 mV, `reward_us` thr=−42.2 mV, all
`mask-all-True`, with global `enable_homeostasis=False` and `mask is None=False`
(`bridge.py:6320-6323` branch 2). **THE KEY GENERALIZATION GAP:** in `build_bg_brain_regions` the
`enable_critic_homeostasis` kwarg sets `enable_homeostasis` ONLY on the afferent + critic
(`g11_bg_runner.py:1230,1252,1283`) — NOT on `snc`/`reward_us` (`:1133-1142`,`:1158-1163`). The
RESOLVED minimal organ set it on ALL 4 incl. `limbic_snc` (`nav_conv_merged_bridge.py:591`), and the
SNc is the saturating operating point that benefits — so the full lift must ALSO mask `snc`+`reward_us`
(a post-hoc `r.enable_homeostasis=True` on the returned regions, no `build_bg_brain_regions` kwarg
required).

**(b) The critic FIRING + the clean GABA_B value subtraction co-resident hit the SAME narrow-window /
saturation boundary the RESOLVED finding flagged (lines 76-78) — NOT a free GO.**

- The MSN-D1 critic (`striosome_value`) does **NOT** fire from its afferent at the init plastic
  weight (0.20) at ANY config (0.0 Hz) — and critically, **the per-region homeostasis mask on the
  critic does NOT change its f-I**: direct-drive `striosome_value` 200/339/500/800 pA → identical
  0/0/1/51-57 Hz with vs without the mask. The MSN-D1's `vr=−80 mV`/`vt=−25 mV` make its f-I
  threshold-INSENSITIVE (it barely reaches −41 mV at 500 pA), unlike the DA cell that swings through
  −42 mV readily. This is the documented "the SPARSE place code can't fire the MSN critic at ANY
  weight" boundary; the dense `vs_place_context` at init weight 0.20 inherits it.
- The critic DOES fire via its validated drive mechanisms: the **convergent-excitation up-state arm**
  (`enable_convergent_upstate`, `vs_place_drive`→striosome) fires it 235-454 Hz, and a **trained-proxy
  weight** (aff_w 2.0 = the STDP-grown regime) fires it 6.5-105 Hz. So the critic-firing path is intact
  and matches the nav design — it just is NOT supplied by the homeostasis threshold.
- The GABA_B value subtraction shows the **correct DIRECTION** (predicted < unpredicted) when the
  up-state critic fires, but the gap is **WEAK** (~1.05-1.06) because the homeostasis-boosted SNc
  **saturates** at ~438 Hz, leaving the GABA_B little headroom. At lower SNc operating points the
  up-state afferents (600 pA) leak enough excitation that `pred > unpred` (gap < 1.0) — the
  "too-strong cue over-drives the striosome and flips the subtraction" caveat, now via the up-state.
  A clean graded gap on the full nav critic needs the **trained sparse afferent** (not the saturating
  up-state) AND a **lower, non-saturated SNc operating point** — exactly the narrow window the
  RESOLVED finding hit on the minimal organ (it reached gap 1.34 only at `tonic=160/us=400/cue=800`
  with a STRONG direct `cue→striosome` weight-10 afferent that fires the MSN without the up-state).

**Bottom line:** the homeostasis enabler **generalizes for the SNc f-I** (the prompt's load-bearing
question) provided `snc`+`reward_us` are masked too — GO. The **full δ=r−V arithmetic** co-resident is
**the same narrow-window boundary** as the minimal organ, gated on the MSN-D1 critic reaching a
moderate (non-saturated) firing rate, which on the full nav critic requires the TRAINED sparse
afferent (the up-state saturates the loop). This is an **honest, expected residual**, not a new wall —
it is the documented nav-critic firing constraint, faithfully inherited co-resident.

## 2. Merged-integration plan (the concrete minimal change to `nav_conv_merged_bridge.py`)

The merged builder's `build_bg_brain_regions(...)` call is `nav_conv_merged_bridge.py:508`.

1. **Pass the Stage-B kwargs** to lift the full nav critic (replacing the default-kwargs nav build):
   ```python
   nav_regions, nav_pathways = build_bg_brain_regions(
       n_cortex=100, ...existing...,
       enable_neural_critic=True,
       spiking_reward_us=True,
       enable_critic_homeostasis=True,      # masks vs_place_context + striosome_value
       # (the trained-weight regime; the runner's value-train pipeline grows vs_place_to_value 0.2->~0.58)
   )
   ```
2. **ALSO mask `snc` + `reward_us`** (the f-I gap the kwarg leaves) — post-hoc, no new kwarg needed:
   ```python
   for r in nav_regions:
       if r.name in ("snc", "reward_us"):
           r.enable_homeostasis = True
   ```
   (Alternatively, add a small `enable_snc_homeostasis` kwarg to `build_bg_brain_regions` that masks
   `snc`+`reward_us` symmetrically — cleaner, but the post-hoc set is sufficient and avoids a `sim/`-
   adjacent builder edit.)
3. **The GABA_B + dopamine-modulator config** the merged builder already sets for `co_resident_limbic`
   (`nav_conv_merged_bridge.py:671-683`) transfers verbatim — but **re-point the modulator's
   `source_regions` from `["limbic_snc"]` to `["snc"]`** (the full nav critic's DA cell is named `snc`,
   not `limbic_snc`). Global `enable_homeostasis=False` + `enable_synaptic_scaling=False` stay
   (the foot-gun guard `nav_conv_merged_bridge.py:690-692` still holds: the synaptic-scaling clip is
   gated by the separate `cfg.enable_synaptic_scaling`, never run).
4. **`co_resident_limbic` → fold into the nav critic, keep the minimal organ as a fallback.** The full
   nav critic SUPERSEDES the 4-region minimal `limbic_*` organ (it IS the same actor-critic topology
   at nav scale, wired to the perceived place/reward streams). Recommendation: make the full critic a
   new opt-in path (e.g. `co_resident_nav_critic=True`) and **keep `co_resident_limbic` as the
   minimal-organ fallback** (it is already validated GO 3.09× co-resident and is a cleaner CI smoke
   that doesn't require the value-train pipeline). Do NOT run both at once (two DA `snc`/`limbic_snc`
   pools + two `dopamine` modulators would double-count the scope=`all` plasticity broadcast).

**Honest scope of the merge step:** the SNc f-I lifts cleanly; the critic V-learning + the clean value
subtraction require the runner's **value-train pipeline** (the DA-gated STDP that grows
`vs_place_context→striosome_value`) to run on the merged bridge — that is the LEARNING increment (#2 in
the validate runner's roadmap), separable from this f-I lift. The de-risk confirms the substrate is
sound; the clean arithmetic is gated on training, not on the homeostasis enabler.

## 3. Anti-cheats + honest residuals

- **The lesion gates' DOPAMINE-adaptation / state-settle confound (carried from the RESOLVED finding)
  is PRESENT on the full critic too** — the homeostasis-boosted SNc is highly excitable and saturates,
  so a multi-condition battery with inter-condition silent settles leaves the DA cell in a
  lower-firing state by the late (lesion) windows; the reward-lesion read overshoots below the early
  baseline (the RESOLVED finding's exact mechanism). The clean-gate fix is the documented
  **re-baseline-per-condition** protocol (measure the tonic floor immediately before each lesioned
  read), a test-protocol refinement. I froze `homeostasis_threshold_adapt_rate=0.0` during the probe
  (removes the threshold-drift component) but the adaptation/settle-state component remains.
- **The narrow operating window is REAL and inherited:** the homeostasis boost trades SNc f-I gain for
  saturation headroom. The full nav critic's window is even narrower than the minimal organ's because
  its afferent is the WEAK plastic `vs_place_context` (init 0.20) — the critic only fires via the
  up-state (which saturates the loop and flips the subtraction at low SNc tonic) or via trained
  weights. The clean δ=r−V on the full critic is gated on TRAINING + a low SNc operating point.
- **No moat / no nav-inertness risk introduced here** (this de-risk built the critic STANDALONE, not
  co-resident). The co-residence + nav-inertness + moat gates are the merge runner's job
  (`_merged_limbic_coresident_validate.py` already PASSES them for the minimal organ); the full critic
  must re-pass them (its `striosome_value→snc` GABA_B + `reward_us→snc` are the only new edges, all
  internal to the critic slice).
- **het OFF is correct** (matches the merged bridge): the RESOLVED finding's latent
  `_apply_parameter_heterogeneity` bug (`bridge.py:~2032`) would otherwise overwrite the `snc`
  IZH2007_DOPAMINE params with RS-centered jitter, silently running the DA cell as RS. Confirmed: with
  het off, `snc` builds as a correct IZH2007_DOPAMINE (the bridge log: `Region 'snc' (10 neurons):
  using Izh type IZH2007_DOPAMINE`).

## 4. Load-bearing file:line

- `sim/bridge.py:6318-6325` — the threshold-select: global-off + per-region mask → `cp.where(mask,
  adapted_thresholds, vpeak)`. Branch 2 is what gives ONLY masked neurons the low threshold.
- `sim/bridge.py:1227-1245` — builds `cp_homeostasis_neuron_mask` from regions with
  `enable_homeostasis=True`; `:1369-1375` allocates `cp_neuron_firing_thresholds` ∈ [−55,−30] uniform
  (mid ~−42.5 mV) — the adapted threshold the masked neurons receive.
- `research/runners/g11_bg_runner.py:1133-1142` (`snc`, no homeostasis), `:1158-1163` (`reward_us`, no
  homeostasis), `:1230 / :1252 / :1283` (`enable_critic_homeostasis` masks ONLY afferent+critic),
  `:1874-1879` (`vs_place_context→striosome_value` plastic init weight `vs_place_to_value_weight=0.2`),
  `:344` (the 0.20 init the STDP grows to ~0.58), `:1866-1871` (`enable_convergent_upstate` A1 arm).
- `research/runners/nav_conv_merged_bridge.py:508` (the `build_bg_brain_regions` call to modify),
  `:591` (the minimal organ masks `limbic_snc`), `:671-683` (the GABA_B + `dopamine` modulator config
  to reuse, re-pointing `source_regions` to `["snc"]`), `:690-692` (the foot-gun guard that still
  holds).
- `research/findings/2026-06-18-merged-config-homeostasis-boundary-RESOLVED.md:50-78` — the f-I
  mechanism + the narrow-window/saturation caveat this de-risk confirms generalizes.

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._organ_lift_homeo_generalize_derisk --seed 42
```
