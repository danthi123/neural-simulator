# Consolidation dendritic op-sweep — INTERIM (seed-42 indicator, sweep IN PROGRESS): the c_drive non-separation is the WRITE being non-selective (linear g_e median 1.025) AND the bistable plateau saturating (g_coincidence uniformly 1.000); the attractor-OFF-write fix is REFUTED (2026-07-25)

**Status: INTERIM / seed-42 indicator.** The full 480-config sweep runs on the 3090 (seed 42, ~3h) + the mini-PC pool
(6-seed, ~30-40h, downtime). This note captures the root-cause analysis from the first 58 seed-42 cells so Tuesday's
continuation has the diagnosis + the next test. NOT a 6-seed verdict.

## What the op-sweep shows so far (58/480 seed-42 configs, k=2/3/4)
Every config: the reported (plateau `g_coincidence`) c_drive own/other ratio ≈ **1.000** (max 1.026), selective ignition
≤ chance. The dendritic plateau does NOT route selectively at ANY operating point tried — consistent with the design's
prior CLIFF finding, now measured directly across the grid.

## Root cause — a two-part decomposition (the c_drive probe records BOTH conductances per fact)
The `cdrive_probe` stores, under each fact's tag, the mean plateau conductance (`g_coincidence`) AND the linear
excitatory conductance (`g_e`) at each slot. Splitting them:
- **Plateau `g_coincidence` own/other: uniformly 1.000** (max 1.026, 0/168 fact-obs > 1.3). The bistable/all-or-none
  plateau ignites FULLY even at the weaker non-own input → it ERASES any input gradient (the saturation the P0.3
  boundary predicts).
- **Linear `g_e` own/other: median 1.025** (mean 2.49 was OUTLIER-DRIVEN — a few near-zero-denominator 251× spikes;
  the median is the honest statistic). Only **3/58 configs** have ≥2/3 facts linearly-selective, and those are
  degenerate (one fact at ratio 0.0). All three sit at the **sr=0 / k=4 / wta=5** corner (op049/051/052).

⇒ **The co-activation write does NOT robustly localize `ca1_i→slot_i`** (linear median 1.025 = flat), AND the plateau
saturates on top. The write is the PRIMARY issue; the plateau saturation is a real secondary one. This CONFIRMS the
design's core diagnosis ("the STDP-written ca1→slot selectivity is too weak to give the plateau structure to route
on") — now with a direct linear-vs-plateau measurement, not inferred from the ignition cliff.

## Refuted this session
- **Attractor-OFF write hypothesis (REFUTED, seed 42):** the idea that the write is non-selective because the
  co-activation runs with the attractor ON (recurrent spread makes all slots fire → potentiates ca1_i→all slots). Test:
  co-activation with `attractor_on=False`. Result: c_drive ratio 1.0 either way (ON and OFF) — the recurrent-spread is
  NOT the cause. `research/runners/_consol_write_selectivity_probe.py`.
- **My own mean-vs-median over-read (self-corrected):** the linear g_e MEAN (2.49) first looked like "the write IS
  selective"; the MEDIAN (1.025) shows it is mostly flat. Checked the median before concluding.

## NEXT (per THE LAW — the residual is mapped, the surpass named)
1. **A genuinely selective WRITE** is the load-bearing need (not a plateau operating point — no op point can fix a
   non-selective write). The design's Option-3 (BTSP one-shot) was chicken-and-egg (needs a selective plateau to gate
   the write, which needs a selective write). The named deeper mechanism is the **dendritic LINE/BUMP attractor** (a
   graded moving bump over the slots, Ecker/continuous-attractor style) rather than N independent point-plateaus.
2. **A GRADED (non-saturating) readout:** `enable_graded_dendritic_plateau` — if a selective write existed, the graded
   plateau would preserve the g_e gradient the bistable plateau erases. Test alongside a selective write, not alone.
3. **The sr=0/k=4/wta=5 corner** (op049/051/052) is a WEAK lead (marginal + degenerate) — check at 6 seeds when the
   full sweep lands; likely noise, but the only linear-selectivity signal in the grid.
4. The full 6-seed sweep (pool) + the 480-config seed-42 (GPU) confirm the plateau-op-point negative robustly.

## Provenance
`_consol_dendritic_opsweep.py` (GPU seed-42 preview, `research/findings/raw/consol_opsweep_gpu/`) +
`_consol_write_selectivity_probe.py`. Interim analysis of 58/480 seed-42 cells. Reuse-by-import, NO sim/ edit.
Part of the [downtime 3-lane compute](2026-07-25-consolidation-opsweep-downtime-MANIFEST.md).
