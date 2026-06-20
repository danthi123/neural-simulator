# Tier-3 SPIKING persistent living loop — the co-resident interoceptive drive on the merged one-brain

**Date:** 2026-06-20

> ## CONTROLLER VERDICT (seed 42, GPU; controller-verified `_tier3_spiking_living_loop_s42.json`)
> **SPLIT — the spiking DRIVE CONVERTS; the spiking SURVIVAL POLICY = an honest NEGATIVE (the dendrite wall, surfaced in the capstone).**
> - **DRIVE converts (brain-based GO):** the interoceptive drive is now ACTUAL co-resident neurons on the merged
>   bridge whose firing tracks the body deficit at **corr 0.995** (lived); it is LOAD-BEARING — lesion it and the
>   agent does worse (`mean_energy` intact 0.135 > lesion 0.034). The no-confab **moat held** (byte-frozen in vivo).
>   Persistence OK. **NO `sim/` edit** (reuse-by-import of the `co_resident_limbic` lift template).
> - **SURVIVAL is a NEGATIVE:** the intact agent does NOT keep itself alive (`min_energy` 0.0, 915 crash steps) and —
>   the decisive datum — does WORSE than a yoked-RANDOM policy (intact mean_energy 0.135 **< yoke 0.212**). So the
>   fixed spiking BG-cascade food-seeking is worse than random for survival.
> - **DIAGNOSIS (why):** the rate-proxy living loop (GO 6/6) used a *learned* Q-policy that LEARNED effective
>   food-seeking. The spiking substrate **cannot learn that policy** — the spiking actor-critic credit-assignment is
>   the 3rd documented NEGATIVE (2026-06-19, "resolves toward the dendrite"), and the fixed spiking nav (with the #6
>   re-orient cost) cannot substitute. ⇒ a fully-brain-based living agent's SURVIVAL needs the **learned spatial
>   policy = the dendrite substrate (Tier-4)**. This UNIFIES the frontier: the same dendrite wall that blocks the nav
>   read-outs (#6/#9) blocks the living agent's survival policy. The DRIVE half is genuinely brain-based now; the
>   POLICY half is the owner-deferred dendrite call. Honest scope: the brain-based deliverable (the boundary maps the
>   substrate cost), NOT a faked GO. 6-seed confirm + the cross-modal "one animal" DA→composer demo are follow-ons.

**Status:** _(filled in below from the GPU de-risk)_
**Runner:** `research/runners/_tier3_spiking_living_loop_derisk.py`
**Builder kwarg:** `conv_extra_regions_pathways(co_resident_drive=True)` +
`build_merged_nav_conv_bridge(co_resident_drive=True)` (`research/runners/nav_conv_merged_bridge.py`)
**Backend:** `SIM_BACKEND=cupy` (the merged bridge is GPU-only)

## Goal

The owner's top directive — *"move every bit of the sim possible onto the shared spiking substrate; true one
brain"*. The **rate-proxy persistent living loop is GO 6/6** (`2026-06-20-tier3-persistent-living-loop-derisk.md`,
commit `8d236b5f`): the agent keeps itself alive from a self-generated intrinsic drive-reduction reward (NO
external goal) and persists across a reset; all anti-cheats collapse. That probe's hunger DRIVE was a **host
rate-proxy** (the AgRP/POMC `TwoPoolDrive` Python class).

This de-risk **lifts that drive onto the merged one-brain bridge as ACTUAL CO-RESIDENT SPIKING NEURONS**, so the
agent keeps itself alive via a SPIKING drive on the shared substrate — the noted follow-on in the rate-proxy
finding's GO line.

## What is new vs the rate-proxy (the brain-based delta)

The interoceptive hunger DRIVE is now real spikes **on the same bridge the agent navigates**. The merged nav+conv
bridge is built by `run_moving_goal_episode` via `extra_regions` (parser + dlPFC + the new drive slice):

- `conv_extra_regions_pathways(co_resident_drive=True)` appends a **2-pool SPIKING drive slice** — `drive_agrp`
  (hunger) / `drive_pomc` (satiety); hypothalamic AgRP/POMC, catalog O.05/O.06; validated mechanism
  `2026-06-17-homeostatic-spiking-drive-mechanism-GO.md` (corr(deficit,AgRP)≥0.9) — co-resident with the nav
  cascade + parser + dlPFC. **ZERO out-edges** → maximally nav-inert (like the `rf` composer slice).
- Each living step, the body's energy **DEFICIT** is injected as an **interoceptive current** into `drive_agrp`
  (∝ deficit) and `drive_pomc` (∝ surplus) — the legitimate body→sensory boundary — and the **SPIKING HUNGER** is
  READ as the `drive_agrp` **FIRING RATE** off `cp_firing_states` (NOT a host deficit value).
- That spiking hunger **GATES the reward** of the VALIDATED BG-cascade learner (the episode's
  `homeostatic_hook`): `reward *= hunger`; food relocates on an "eat"; an **intrinsic drive-reduction reward**
  (Keramati-Gutkin). The reward `r` rides the NEURAL drive, not a host distance term.

So the survival decision — "keep yourself alive" — is driven by a SPIKING interoceptive drive on the **same
shared substrate** as the validated spiking-WTA nav action selection (Wang-2002) and the conversational
parser/dlPFC.

## Implementation (NO `sim/` edit)

Templated on the proven `co_resident_limbic` lift (prior = NO sim/ edit). The drive slice:

- **Builder:** two `BrainRegion`s (`drive_agrp`, `drive_pomc`), `internal_density=0`, **no pathways at all**
  (driven by current, read by firing). `enable_homeostasis=True` **per-region** = the merged-config
  operating-point fix (the limbic-core-lift lesson): the already-shipped per-region homeostasis mask
  (`sim/bridge.py:1227-1245/:6320`) gives ONLY the drive slice the low spike threshold while nav/conv stay at
  vpeak (byte-unchanged); the synaptic-scaling clip (gated by the SEPARATE `enable_synaptic_scaling`, OFF here)
  never runs → the frozen-weight foot-gun is untouched. The **global** `cfg.enable_homeostasis` stays `False`.
- **Default-OFF byte-identical.** Appended LAST, so the nav/parser/dlPFC/rf index bases are byte-unchanged.
- Added in BOTH the framework builder (`build_merged_nav_conv_bridge`) and the episode-path
  `conv_extra_regions_pathways` (the latter is the one the living loop uses, since the navigated bridge is built
  inside `run_moving_goal_episode`).

**No `sim/` edit was needed** — purely the additive builder kwarg + the runner.

## GPU smoke (mechanics — confirmed)

`--smoke` (1 seed, intact, grid 6, 80 steps, drive_window 60):

- Merged bridge built with the drive slice: **68 regions / 7437 neurons**; the log reports
  `Homeostasis per-region mask: 2 regions enabled (120 neurons)` = exactly the `drive_agrp`+`drive_pomc` pools.
- **corr(deficit, AgRP firing) = +1.00** — the co-resident SPIKING drive encodes the body deficit on the
  navigated bridge.
- **The no-confab MOAT held in vivo:** the **720 parser synapses are BYTE-IDENTICAL** across the live nav run
  (frozen under the reward-STDP + dopamine + the co-resident drive stressor), and the parser still parses
  voice-invariantly (`active_agent=dog`, `passive_agent=dog`).
- Body life-state **persists** across the reset; the hook fires; the agent eats; the reward is
  spiking-hunger-gated.

## Full 3-seed de-risk (survival + anti-cheats)

_(Results table + verdict — filled in from `research/findings/raw/_tier3_spiking_living_loop.json`.)_

| seed | mode | corr(deficit,AgRP) | eats (post-wean) | min-E post | crashes | persist | moat |
|------|------|--------------------|------------------|------------|---------|---------|------|
| _TBD_ |

### Anti-cheat collapse

- **DRIVE-LESION** (zero the interoceptive current → `drive_agrp` silent → hunger floors → reward attenuated): _TBD_
- **YOKED-RANDOM** (the spiking hunger replaced by a shuffled signal of matched marginal): _TBD_
- **REWARD-PROVENANCE:** `r` is the spiking-hunger-gated drive reduction read from `cp_firing_states`; asserted
  by construction that NO `r = f(distance_to_food)` host term exists in the gating.
- **NO-PERSISTENCE:** the persisted resume carries the mid-life deficit; a cold start would be full-energy.

## Verdict

_(GO vs HONEST-NEGATIVE — filled in.)_

## Honest scope

This realizes the **DRIVE** in spikes. The **LEARNED SPATIAL POLICY** under the cascade stays the deferred
dendrite wall (Tier-4); survival (not spatial optimality) is the discriminator — the rate-proxy already showed
survival is GO without a converged spatial policy. If the validated spiking-nav cost (the ~16% commit-timing
floor of the spiking-WTA readout, `2026-06-19-spiking-decision-default-on-GO.md`) makes survival underperform the
rate-proxy, that maps the substrate cost — the brain-based deliverable.

## Reproduce

```bash
# smoke (GPU mechanics: corr + moat byte-frozen + persist):
SIM_BACKEND=cupy python -m research.runners._tier3_spiking_living_loop_derisk --smoke

# multi-seed survival + anti-cheat de-risk (validated reuse-probe dynamics; drive read every 5 steps):
SIM_BACKEND=cupy python -m research.runners._tier3_spiking_living_loop_derisk \
    --seeds 42 43 44 --n-steps 1800 --grid-size 8 --deplete 0.004 --refill 0.6 \
    --drive-window 40 --drive-read-every 5 \
    --out research/findings/raw/_tier3_spiking_living_loop.json
```

## Commits

- `f47dc10d` — builder kwarg (`co_resident_drive` on the framework builder +
  `conv_extra_regions_pathways`) + the runner + smoke OK.
- `c2b886ba` — lesion-zeroes-reward semantics + the `--drive-read-every` drive-read subsampling.
- _(survival + anti-cheat results commit — TBD)_
