---
type: finding
status: contributing
date: 2026-09-25
lane: F · gap#4 deep credit (the crux lane; plan step S19 / GPU step G5)
mechanism: bdsp-sliding-burst-baseline (research/biology/bdsp-sliding-burst-baseline.md) -- AMENDMENT 6's dev
  battery (C25-C27, 36 dev-seed-7 shards, pool) and AMENDMENT 7's C26 full-size GPU transfer run, scored against
  the registered rules exactly.
seeds: [7]
seed-waiver: dev seed 7 only, on both the pool battery and the GPU run. This is a dev-seed de-risk under the
  pre-registration, not a 6-seed verdict, and neither run is a gate row. No evaluation seed (42+) has run; the
  runner's evaluation-seed guard (AMENDMENT 5 D) still has no committed EVALUATION CONFIG section naming a
  fingerprint and seed list, so it continues to refuse one.
verdict: UNDEFINED on interpretability at both scales, and DECISIVE on the drift question. C26's sliding
  burst-probability baseline (`cfg.bdsp_pbar_ratio_tau_ms`, hidden layers) removes the one-sided ff_0/ff_1
  weight-clamp saturation the 2026-09-24 census found load-bearing -- 0 of 3 dev replicates and 0 of 3 full-size
  replicates now cross the 10%-at-bound threshold, against 3 of 3 dev replicates before the fix. At dev size the
  transport_ceiling arm's training accuracy also clears its own training chance on 3 of 3 replicates (AMENDMENT 6
  criterion ii). Neither result transfers into an interpretable readout: held-out accuracy never clears chance
  with the registered headroom on any replicate at either scale (0 of 3 interpretable, dev and full-size alike),
  and the training-fit gain itself does not transfer to full size (1 of 3 replicates there, against 3 of 3 at
  dev). Per AMENDMENT 6's own decision procedure this is the "(i) and (ii) hold, (iii) fails" branch: the
  collapse is repaired, the ceiling is still not interpretable at the dev budget. Per AMENDMENT 7's decision
  procedure the full-size run is NOT DEFINED, so C26 is reported UNDEFINED at full size, never NO-GO. C25 (the
  clamp relaxed to 48, no ratio baseline) still collapses -- 2 of 3 replicates still cross the census threshold,
  worse than at the earlier 3-epoch smoke -- confirming the drift, not the wall's position, caused the earlier
  saturation. C27 (ratio baseline on all layers, including output) reads within noise of C26 on every registered
  criterion, so the output layer's own baseline is not the missing piece.
runner: research/runners/_gap4_transport_ceiling_readout_derisk.py
prereg: research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md (AMENDMENT 6, items
  C26/C27/C25 and their Decision/Criteria; AMENDMENT 7, the C26 full-size GPU run and its Decision; the 2026-09-25
  Erratum on tau_avg)
artifacts:
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r0_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r0_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r0_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r1_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r1_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r1_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r1_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r2_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r2_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r2_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/s7_r2_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r0_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r0_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r0_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r1_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r1_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r1_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r1_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r2_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r2_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r2_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/ckpt/s7_r2_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r0_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r0_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r0_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r1_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r1_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r1_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r1_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r2_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r2_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r2_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C27/ckpt/s7_r2_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/C26_s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C26/C26_s7_r0_transport_ceiling.json.prov.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40.json.prov.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r0_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r0_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r0_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r1_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r1_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r1_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r1_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r2_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r2_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r2_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40_ckpt/s7_r2_transport_ceiling.json
external: none new this round (carried from AMENDMENT 6: Payeur et al. 2020/2021, bioRxiv 2020.03.30.015511;
  Bienenstock, Cooper & Munro 1982; van Rossum, Bi & Turrigiano 2000; Royer & Pare 2003).
builds_on:
  - research/findings/2026-09-24-gap4-transport-ceiling-bound-census-clamp-load-bearing-fullsize-UNDEFINED.md
  - research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md
  - research/biology/bdsp-sliding-burst-baseline.md
---

# gap#4 sliding baseline: the clamp drift is fixed at both scales; readout interpretability is not (dev seed 7)

**Headline.** AMENDMENT 6 registered three dev configs (C25/C26/C27) to test whether Payeur's sliding
burst-probability baseline removes the one-sided weight-clamp saturation the 2026-09-24 census found load-bearing,
and AMENDMENT 7 registered a full-size GPU transfer check for the winning config (C26) ahead of the dev battery
landing. Both are complete now: all 36 dev-seed-7 pool shards (C25/C26/C27, 4 arms x 3 replicates each) and all 12
full-size GPU shards (C26, H64/pool16, 40 epochs). The drift question has a clean answer: C26's ratio baseline
stops the saturation at dev size (0 of 3 replicates cross the census threshold, against 3 of 3 before the fix) AND
at full size (0 of 3), and C26's transport_ceiling arm fits its own training set above chance on 3 of 3 dev
replicates -- a real change, not noise. The interpretability question does not move: held-out accuracy never
clears chance with the registered headroom at either scale, and the training-fit gain itself collapses when the
net scales up (1 of 3 replicates at full size). This is exactly the "(i) and (ii) hold, (iii) fails" branch
AMENDMENT 6 pre-registered, and exactly an UNDEFINED (never NO-GO) full-size read under AMENDMENT 7's own rule.
<!--derived-->

## Provenance (checked before scoring)

All 36 dev-seed-7 pool shards read `git_sha: feaca2fdf4561d43275250763a54dab18f870298` (AMENDMENT 6's registered
commit) and `git_dirty: false`, with `source_kind: git_archive` and a matching `source_manifest_sha256` in every
`.prov.json` sidecar -- a clean, verified checkout, not the live pool tree. <!--derived--> All 36 landed against
AMENDMENT 6's own 36-shard manifest (3 configs x 3 replicates x 4 arms); none are missing, and none are stray
duplicates. <!--derived-->

The 12 full-size GPU shards read `git_sha: 04e6def34e12b595b9197d28a4b6885bc90d796d` (the A10_PIN_SHA AMENDMENT 7
names) in every shard. <!--derived--> 11 of the 12 per-arm shards, and the final `--aggregate-only` write, read
`git_dirty: true`; only the very first shard written (`s7_r0_fixed_fa.json`, by file mtime) reads `git_dirty:
false`. This is the checkpointing process's own output accumulating in the tree, not a code change: a byte-diff
of `research/runners/_gap4_transport_ceiling_readout_derisk.py`, `sim/bridge.py` and `sim/config.py` between the
pinned commit `04e6def34` and the live files in that detached worktree is empty on all three files. <!--derived-->
The runner's own argv (`gpu_c26_s7_e40.json.prov.json`) shows `--pbar-ratio-tau-ms 5000 --pbar-ratio-layers
hidden` and the C21 flags, `--aggregate-only` for the final write, matching AMENDMENT 7's registered config; every
shard's own `backend` field reads `cupy`. <!--derived--> The aggregate write's mtime is 11:34:10 EDT 2026-09-25
(close to the task's "~11:45" note; either reflects the same run). <!--derived-->

No prereg_amendment / evaluation-seed guard was invoked by any of these 48 shards (`prereg_amendment: null` in
every config dict), and every run is at `seeds: [7]` -- dev only, never an evaluation seed, consistent with
AMENDMENT 5 D and AMENDMENT 7's own statement that it registers no EVALUATION CONFIG section. <!--derived-->

## Dev battery (C25/C26/C27): per-replicate table

Pool, numpy, H32/pool4, 2 hidden layers, 30 epochs, train subsample 400, seed 7, replicates 0/1/2 (task seeds
7/10014/20021). Chance = 0.166667 (1/6 of 54 held-out items); training chance is the replicate's own majority-class
rate (0.1825/0.170/0.1625). `p(held)` is the one-sided binomial upper-tail probability for the transport_ceiling
arm's held-out correct count against chance; `p(train)` is the shard's own stored `train_binom_p` for that arm.
<!--derived-->

| config | rep | frozen held | fixed_fa held | micro_i held | ceiling held | p(held) | headroom | interp | ceiling p(train) | census fires (ff0/ff1 >=10% at bound) |
|---|---|---|---|---|---|---|---|---|---|---|
| C25 (clamp 48, no ratio) | r0 | 0.130 | 0.204 | 0.185 | 0.241 | 0.104 | 0.111 | false | 0.2771 | false (0.0%/0.7%) |
| C25 | r1 | 0.185 | 0.167 | 0.185 | 0.167 | 0.557 | -0.019 | false | 0.6752 | **true (1.1%/33.6%)** |
| C25 | r2 | 0.204 | 0.167 | 0.148 | 0.185 | 0.412 | -0.019 | false | 0.8765 | **true (49.3%/20.9%)** |
| C26 (ratio baseline, hidden) | r0 | 0.148 | 0.148 | 0.130 | 0.167 | 0.557 | 0.019 | false | 3.6e-07 | false (1.0%/0.4%) |
| C26 | r1 | 0.148 | 0.278 | 0.241 | 0.204 | 0.282 | 0.056 | false | 0.0115 | false (0.5%/0.1%) |
| C26 | r2 | 0.185 | 0.167 | 0.148 | 0.093 | 0.959 | -0.093 | false | 1.8e-04 | false (0.1%/0.2%) |
| C27 (ratio baseline, all layers) | r0 | 0.074 | 0.167 | 0.167 | 0.148 | 0.698 | 0.074 | false | 2.7e-12 | false (0.4%/0.2%) |
| C27 | r1 | 0.148 | 0.222 | 0.222 | 0.130 | 0.818 | -0.019 | false | 0.9548 | false (1.0%/0.3%) |
| C27 | r2 | 0.167 | 0.222 | 0.130 | 0.185 | 0.412 | 0.019 | false | 8.6e-06 | false (0.4%/0.3%) |

<!--derived-->

**Aggregate over the 3 dev replicates, applying AMENDMENT 6's registered criteria exactly:**

| config | (i) census fires on <=1/3? | (ii) ceiling train-fit p<0.05 on >=2/3? | (iii) n_interpretable >=2/3 (Rule B, DEFINED)? |
|---|---|---|---|
| C25 | **FAILS** (2/3 fire) | **FAILS** (0/3) | FAILS (0/3, NOT DEFINED) |
| C26 | **HOLDS** (0/3 fire) | **HOLDS** (3/3) | FAILS (0/3, NOT DEFINED) |
| C27 | **HOLDS** (0/3 fire) | **HOLDS** (2/3) | FAILS (0/3, NOT DEFINED) |

<!--derived-->

**Decision, applying AMENDMENT 6's registered text exactly.** C26: "(i) and (ii) hold, (iii) fails: the collapse
is repaired and the ceiling is still not interpretable at the dev budget; the finding reports the budget
question" -- this is that branch. The registered budget reference (AMENDMENT 4's rate-backprop oracle) reads 0.81
held-out after 4000 updates and 0.94 after 8000 at H32, per
`research/findings/raw/gap4/transport_ceiling_readout/round4_rev8f16994/diag_oracle_online_budget_s7.json`; C26's
30 epochs x 400 items is 12000 credit-phase presentations, more than that budget in raw count, yet the readout's
own local rule does not reach a comparable held-out read -- so the residual is not simply "too few presentations"
in the oracle's sense (see "What this does not show" below). <!--derived-->

C25 (secondary reading): "if C25 still collapses, the drift, not the wall's position, caused the collapse." C25
collapses on (i) (2/3 replicates still cross the census threshold, mean |w| on the affected pathways reaching
20.9-44.3 against the 3.1-5.4 build mean -- WORSE than the +-12-clamped runs, because a looser bound lets the same
one-sided drive travel further before anything stops it) and on (ii) (0/3 train-fit). A bigger constant does not
fix this; the drift is the cause, confirming the 3-epoch smoke's own reading at the full 30-epoch budget. <!--derived-->

C27 (secondary reading): "C27 against C26 says whether the output layer's baseline matters." C27 passes (i) and
(ii) like C26 (2/3 vs C26's 3/3 on (ii), inside dev noise at n=3) and fails (iii) identically (0/3). Mean
transport_ceiling held-out across the 3 replicates is 0.154 for both C26 and C27 -- indistinguishable. <!--derived-->
Extending the ratio baseline to the output layer's own preset baseline does not change the outcome either
direction; the output layer's baseline is not the missing piece.

## Full-size GPU (C26 only, AMENDMENT 7)

Cupy, H64/pool16, 2 hidden layers, 40 epochs, train subsample 400, seed 7, replicates 0/1/2 (task seeds
7/10014/20021), same C26 flags. The runner's own `--aggregate-only` pass
(`research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu/gpu_c26_s7_e40.json`) reports, verbatim:
`n_ceiling_clears_chance: 0`, `n_interpretable: 0`, `n_fa_wall: 3`, `n_surpass: 1`,
`status: "UNDEFINED (interpretable on 0/3 replicates < 2; ceiling clears chance on 0, AMENDMENT 5 also needs
headroom >= 0.05)"`. Its own per-replicate `ceiling_binom_p` values (0.905, 0.104, 0.818) match an independent <!--derived-->
one-sided-binomial recomputation from the ckpt shards' `inherit_heldout`/`n_inh`/`chance` fields to the printed
precision. <!--derived-->

<!--derived-->

| rep | frozen held | fixed_fa held | micro_i held | ceiling held | p(held) | headroom | interp | ceiling p(train) | census fires |
|---|---|---|---|---|---|---|---|---|---|
| r0 | 0.204 | 0.167 | 0.167 | 0.111 | 0.905 | -0.093 | false | 0.9997 | false (2.5%/6.4%) |
| r1 | 0.241 | 0.148 | 0.222 | 0.241 | 0.104 | 0.000 | false | 1.95e-05 | false (0.08%/0.03%) |
| r2 | 0.222 | 0.111 | 0.130 | 0.130 | 0.818 | -0.093 | false | 0.575 | false (0.05%/0.02%) |

**Applying AMENDMENT 7's registered criteria exactly:** census does NOT read load-bearing (0/3 >= 10% threshold,
same threshold as AMENDMENT 5 H found the clamp load-bearing at dev size on 3/3) -- the drift fix generalizes to
full size. Training-fit holds on only 1 of 3 replicates (r1, `p=1.95e-05`; r0 and r2 are at or below their own
training chance) -- against the dev battery's 3/3, this does NOT hold at the registered ">=2/3" bar. Rule B:
0/3 interpretable, seed 7 is NOT DEFINED. Per AMENDMENT 7's Decision text: "Any other outcome (not DEFINED, or
DEFINED but the census still fires, or DEFINED but the training-fit check fails): C26 does NOT transfer at full
size, reported as UNDEFINED (never as a NO-GO on C26 itself...)." That is the outcome here -- NOT DEFINED, with
the training-fit check also failing -- so C26 is UNDEFINED at full size under the registered rule, not a NO-GO.

## What this does show

<!--derived-->

1. **The census/drift fix generalizes across scale.** At dev size, ff_0/ff_1 fire the 10%-at-bound census on 3/3
   replicates before the fix (2026-09-24 finding) and 0/3 after it (this finding, C26). At full size the same
   census reads 0/3 with no prior full-size measurement to compare against (the earlier full-size run used the
   preset baseline throughout and was never separately censused), but the raw `mean_abs_w` on the affected
   pathways (2.2-6.2 across all 12 full-size shards) sits close to the 3.1-5.4 build mean and far below the +-12
   clamp -- the same signature the dev fix produces. <!--derived-->
2. **The output-silencing side-effect the pre-fix smoke reported is also gone, at least at full size.** The
   2026-09-24 pre-fix census reported the hidden-learning arms' output layer going "nearly silent" (held-out mean
   read 0.001-0.09 against 0.17-0.19 in frozen). Under C26, `mean_output_rate_heldout` for transport_ceiling reads
   0.216/0.020/0.120 at dev size (mixed: silenced only on r1) and 0.129/0.226/0.221 at full size (not silenced on
   any replicate, and above the frozen arm's own 0.127/0.151/0.145 on 2 of 3). <!--derived--> This bears directly
   on `research/biology/bdsp-sliding-burst-baseline.md`'s own registered next-companion candidate ("event-rate
   homeostasis... if the sliding baseline removes the drift but the output still falls silent, this is the next
   companion"): at full size the output does NOT fall silent, so this data does not support event-rate
   homeostasis as the explanation for the residual UNDEFINED read.
3. **A real, reproducible training-set fit appears at dev size that was absent before the fix.** C26's
   transport_ceiling arm clears its own training chance at `p<0.05` on 3/3 dev replicates (best: `p=3.6e-07`),
   where the pre-fix instrument (2026-09-24 diagnosis) read at or below training chance in every arm. C27
   replicates this (2/3). C25 (clamp relaxed, no ratio baseline) does NOT (0/3) -- the fit is specific to the
   sliding-baseline mechanism, not to any change in the clamp.

## What this does not show

<!--derived-->

1. **Held-out generalization.** 0 of 3 replicates are interpretable under Rule B at either scale, for any of
   C25/C26/C27. The transport_ceiling arm's held-out reads (0.093-0.241 dev, 0.111-0.241 full-size) scatter around
   chance (0.167) with no config, replicate, or scale showing a clean, reproducible separation.
2. **Transfer of the training-fit gain.** C26's 3/3 dev train-fit drops to 1/3 at full size, despite MORE
   training steps (H64/pool16, 40 epochs vs H32/pool4, 30 epochs). This is the opposite of what a pure
   budget/capacity story predicts (more neurons and more epochs should make a weak local rule's job easier, not
   harder), so "not enough training" alone does not explain the full-size collapse in training fit.
3. **Whether extending the ratio baseline to the output layer helps or hurts.** C27 is statistically
   indistinguishable from C26 on every registered criterion at n=3 dev replicates.

## The next method, per THE LAW (companion process first)

The registered decision text for this exact branch says the finding "reports the budget question," and does not
itself name a further lever -- so this section applies the project's own reframe: *what does the real system run
alongside BDSP, here, that the runner still replaces with a constant?* Two candidates named in
`research/biology/bdsp-sliding-burst-baseline.md`'s own `companion_processes` list are checked against this
data first, before proposing anything new:

- **Event-rate homeostasis (rung 2, `status: proxied`).** Ruled OUT as the explanation by item 2 above: the
  output layer is not silenced at full size, yet the readout is still uninterpretable there. This companion
  process is not indicated by the new data.
- **Plasticity gated to the teaching period (`status: not_implemented`).** This is the one still standing.
  Payeur et al.'s own rule carries a prefactor `M` that is "1 when the teaching signal is present and 0
  otherwise" -- BDSP updates ONLY during the credit-bearing part of a trial. The runner's C21-derived operating
  point (kept unchanged by AMENDMENT 6, which deliberately touched only the baseline) instead runs BDSP on every
  step, including the forward settle before any teaching signal exists. With a SLOW sliding baseline, those
  settle-step updates are no longer washed out by a fast (~20 ms) baseline the way the old EMA implicitly
  suppressed them (AMENDMENT 3's own diagnosis of why the legacy EMA held training below chance) -- the sliding
  baseline instead lets a small, systematically-signed contribution accumulate on every step of every
  presentation, teaching or not. This is a plausible, evidence-grounded account of "fits weakly, does not
  generalize": the credit-phase signal (the part that actually carries class information) is diluted by an
  equal-or-larger volume of settle-phase updates that carry none, on every one of the settle steps
  (`settle_steps=40`) against 25 credit steps (`credit_steps=25`) per training example. <!--derived--> This
  candidate is not yet built or run; it is the direct, literature-named next lever this record already flagged
  before these results landed, not a new hypothesis invented after the fact.

**Proposed next dev check (NOT run by this finding; to be registered as an amendment before it runs).** Gate BDSP
plasticity to the credit phase only (`M=1` during credit steps, `M=0` during settle/ISI), on top of C26's ratio
baseline, dev seed 7, same H32/pool4/30-epoch battery, 4 arms x 3 replicates, before any full-size or evaluation
run. If gating restores or improves held-out interpretability, the settle-phase noise is confirmed as the
residual; if it does not, the credit signal's own strength (the softmax/gain host residual, or the readout's
training budget in a sense not captured by the oracle's SGD-update count) becomes the next candidate.

## Erratum carried forward

The pre-registration's 2026-09-25 Erratum (tau_avg = 5 s is the bioRxiv **v1** XOR value; v2 states 2 s, both
inside the ~1-10 s range that is C26/C27's actual justification for 5000 ms) applies unchanged to this data; no
registered rule, config, or threshold used in this finding depends on which version's value is quoted.

## Flip-candidate status

Not applicable. Nothing here is a production default; this is instrument/lever diagnosis inside the gap#4 crux
lane, still upstream of any dev config that clears the interpretability gate. No config across C0-C27 has cleared
Rule B's interpretability gate at dev or full size on this task. The evaluation-seed guard (AMENDMENT 5 D) still
refuses without a committed EVALUATION CONFIG section, and no anti-cheat battery has been run because nothing has
qualified to run it against yet.

## Functional read-outs only

Every number above is a spiking-network readout accuracy, a synaptic weight statistic, or a binomial/count
statistic over those readouts. Nothing here is a claim about experience.
