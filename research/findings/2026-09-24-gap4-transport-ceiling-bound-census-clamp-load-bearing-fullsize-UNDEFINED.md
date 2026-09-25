---
type: finding
status: contributing
date: 2026-09-24
lane: F · gap#4 deep credit (the crux lane; plan step S19 / GPU step G5)
mechanism: gap4-transport-ceiling-readout-instrument (research/runners/_gap4_transport_ceiling_readout_derisk.py) --
  AMENDMENT 5 H (the bound census) and AMENDMENT 5 F (the full-size GPU dev run), read under the registered rules.
seeds: [7]
seed-waiver: dev seed 7 only. This is a dev-seed de-risk under the pre-registration, not a 6-seed verdict. No
  evaluation seed (42+) has run; the runner's evaluation-seed guard (AMENDMENT 5 D) still has no committed
  EVALUATION CONFIG section to satisfy, so it continues to refuse one.
verdict: UNDEFINED at both rungs, and DECISIVE for the next lever. Rung 0 (bound census, dev net H32/pool 4, C21
  flags, 3 replicates): the +-12 BDSP weight clamp is LOAD-BEARING under the registered rule -- in the
  transport_ceiling arm, at least 10% of ff_0's synapses AND at least 10% of ff_1's synapses sit at +-w_max on 3 of
  3 replicates (registered threshold: >=2 of 3, either pathway). Rung 3 (full-size GPU dev run, H64/pool 16, 40
  epochs, same C21 flags): 0 of 3 replicates are interpretable under rule B (ceiling never clears chance on any
  replicate, one-sided binomial p about 0.56 throughout) -- seed 7 is NOT DEFINED (needs >=2/3), so C21 does not
  transfer to full size and the full-size instrument is also UNDEFINED. Per the pre-registration's own decision
  procedure (AMENDMENT 5 H), the next lever is the BDSP weight bound itself (the companion process the static
  clamp replaced), ahead of spiking lateral inhibition and output-unit homeostasis.
runner: research/runners/_gap4_transport_ceiling_readout_derisk.py
prereg: research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md (AMENDMENT 5, items H
  and F)
artifacts:
  - research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/ckpt/s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/ckpt/s7_r1_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/ckpt/s7_r2_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/ckpt/s7_r0_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/ckpt/s7_r0_fixed_fa.json
  - research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/ckpt/s7_r0_micro_inengine.json
  - research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/census_s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/census_s7_r0_transport_ceiling.json.prov.json
  - research/findings/raw/gap4/transport_ceiling_readout/gpu/gpu_s7_e40.json
  - research/findings/raw/gap4/transport_ceiling_readout/gpu/gpu_s7_e40.json.prov.json
  - research/findings/raw/gap4/transport_ceiling_readout/gpu/gpu_s7_e40_ckpt/s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/gpu/gpu_s7_e40_ckpt/s7_r1_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/gpu/gpu_s7_e40_ckpt/s7_r2_transport_ceiling.json
external: none new this round (carried from the pre-registration: Bellec et al. 2020 Nat Commun 11:3625; Mazurek et
  al. 2026 Front Neurosci; Carandini & Heeger 2012 Nat Rev Neurosci 13:51).
builds_on:
  - research/findings/2026-09-24-gap4-transport-ceiling-instrument-diagnosed-dev-seed7-UNDEFINED.md
  - research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md
---

# gap#4 transport ceiling: the bound census says the clamp IS load-bearing; the full-size run does not transfer (dev seed 7)

**Headline.** Two runs were still open against the 2026-09-24 instrument-diagnosis finding: the bound census
(AMENDMENT 5 H) and the full-size GPU dev run (AMENDMENT 5 F). Both are now complete on dev seed 7. The census
answers the question it was built to answer: yes, the static +-12 weight clamp is load-bearing in the arm that
matters (transport_ceiling), so the registered decision procedure names the weight bound as the next lever, ahead
of lateral inhibition and output homeostasis. The full-size run answers a different question (does C21 transfer to
the 2026-09-15 net size) and the answer is no -- at H64/pool 16 every hidden-learning arm (fixed_fa,
micro_inengine, transport_ceiling) lands EXACTLY at chance on every one of 3 replicates, a cleaner collapse than at
dev size, where the transport ceiling reached its best held-out read on one replicate (rung 0's r0, cited below). <!--derived-->

## Rung 0: the bound census (AMENDMENT 5 H)

Runner: `_gap4_transport_ceiling_readout_derisk.py`, C21 flags (H32/pool 4, 30 epochs, `--bdsp-w-max 12`), 4 arms x 3
replicates, dev seed 7, revision `afe2b323871b7363b0db119c05241ef147a84922` (a descendant of `23d98e71f`, the
commit that introduced AMENDMENT 5 and the `bound_census()` method the census reads; `git diff` of the runner file
between that revision and the latest AMENDMENT-5 fix commit on main is empty, so the code is byte-identical to
what is on main today). Every shard: `git_dirty: false`, `source_kind: git_archive` with a verified manifest sha256
at both start and exit. Artifacts: one shard per (replicate, arm), e.g.
`research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/ckpt/s7_r0_transport_ceiling.json`,
with its r1/r2 and frozen/fixed_fa/micro_inengine siblings in the same directory (12 shards total).

**Consistency check (required by the pre-registration before the census counts).** The r0 shards must reproduce the
earlier dev run's r0 reads exactly. They do: census r0 frozen `inherit_heldout` 0.12962962962962962 and
`ff_weight_moved` 9063.4296875 match the dev run's r0 frozen row (0.130, ff-moved 9063.4); census r0 <!--derived-->
transport_ceiling `inherit_heldout` 0.2222222222222222 and `ff_weight_moved` 162621.1640625 match the dev run's r0
ceiling row (0.222, ff-moved 162621.2). The census is not VOID. <!--derived-->

**Per-pathway fraction of feedforward synapses at +-`bdsp_w_max` (12.0), transport_ceiling arm, END of training**
(from `bound_census_end` in each shard; `ff_0` and `ff_1` are the two hidden-post pathways, `ff_2` is the
output-post pathway):

| replicate (task seed) | ff_0 frac at bound | ff_1 frac at bound | ff_2 frac at bound | frac_at_bound_all_ff |
|---|---|---|---|---|
| r0 (7) | 0.796875 | 0.22723388671875 | 0.0 | 0.2888671875 |
| r1 (10014) | 0.794921875 | 0.7489013671875 | 0.0 | 0.6223828125 |
| r2 (20021) | 0.18033854166666666 | 0.2479248046875 | 0.0 | 0.1911328125 |

At the START of training every pathway reads 0.0 (no synapse at the bound; `mean_abs_w` 3.09-3.83 against the +-12
clamp), so the saturation is a training effect, not an initialization artifact.

**Negative control (frozen arm, no hidden credit).** `bound_census_end` for frozen, r0: ff_0 0.0, ff_1 0.0 (the
readout-only arm never moves the hidden feedforward weights, so it cannot saturate them). The other two
hidden-learning arms saturate similarly to the ceiling: fixed_fa r0 ff_0 0.2348090277777778 / ff_1
0.1790771484375; micro_inengine r0 ff_0 0.2267795138888889 / ff_1 0.17230224609375. The saturation is specific to
arms that actually train the hidden feedforward weights, which is the expected direction for a real clamp effect
(not a runner bug).

**Decision, applying the registered rule exactly.** "The clamp is load-bearing if, in the transport_ceiling arm, at
least 10% of the synapses of a hidden-post pathway (ff_0 or ff_1) end at +-w_max on at least 2 of 3 replicates."
ff_0 clears 10% on 3 of 3 replicates (0.797, 0.795, 0.180) and ff_1 clears 10% on 3 of 3 replicates (0.227, 0.749, <!--derived-->
0.248). Both pathways clear on all three replicates, well past the registered >=2/3 threshold. **The clamp is <!--derived-->
load-bearing.** Per AMENDMENT 5 H, the next lever is the BDSP weight bound itself (the companion process the
static clamp replaced), before lateral inhibition or output homeostasis.

## Rung 3: the full-size GPU dev run (AMENDMENT 5 F)

Runner: `_gap4_transport_ceiling_readout_derisk.py`, C21 flags at full size (H64/pool 16, 40 epochs, same
`--bdsp-w-max 12`), 4 arms x 3 replicates, dev seed 7, `SIM_BACKEND=cupy`. Config fingerprint
`713efa9804a3dbb0`, matching across every shard and the aggregate. Started 15:54:59 EDT 2026-09-24 (per the
pre-registration's own record of the launch, `research/queue/_a9_gap4_tc_gpu.sh` at pin `10479d530`); the
per-replicate checkpoint shards finish between 17:30 and 21:00 EDT the same day (about 5 hours wall clock for one
seed, 4 arms, with partial overlap on one GPU). Each per-shard checkpoint carries no `git_sha` field (that
instrumentation was added by AMENDMENT 5, i.e. after this shard-writing code path was pinned at `10479d530`,
consistent with the pre-registration's own statement that this run started before AMENDMENT 5). The aggregate file
`gpu_s7_e40.json` was regenerated by a later `--aggregate-only` pass (`run_id 1790298038-2545526`, started
21:00:38 EDT) whose own `.prov.json` records `git_sha: 10479d530` (the prereg's own named pin for this run) with
`git_dirty: true` and no `source_manifest` (unlike the census, this run was not launched from a verified git-archive
checkout). **Caveat, stated plainly:** the aggregate step's dirty flag is a genuine provenance gap relative to the
census's clean git-archive run. It does not change the numbers below, which are read directly off the individual
training shards (`gpu_s7_e40_ckpt/s7_r{0,1,2}_transport_ceiling.json`, no `git_sha` field, predating that
instrumentation) and are independently recomputable: `ceiling_binom_p` is the one-sided binomial upper-tail
probability for `k = round(inherit_heldout * n_inh)` successes out of `n_inh = 54` at `p = chance = 1/6`, and
matches the aggregate's own report to the printed precision in every replicate. No claim below depends on the
`--aggregate-only` pass having run at a specific commit.

**Per-replicate result** (`inherit_heldout`, chance = 0.16666666666666666 = 9 of 54 held-out items):

| replicate (task seed) | frozen | fixed_fa | micro_inengine | transport_ceiling | ceiling binom p | headroom | fa_wall |
|---|---|---|---|---|---|---|---|
| r0 (7) | 0.16666666666666666 | 0.16666666666666666 | 0.16666666666666666 | 0.16666666666666666 | 0.5565130652800339 | 0.0 | true |
| r1 (10014) | 0.24074074074074073 | 0.16666666666666666 | 0.16666666666666666 | 0.16666666666666666 | 0.5565130652800339 | -0.07407407407407407 | true |
| r2 (20021) | 0.24074074074074073 | 0.16666666666666666 | 0.16666666666666666 | 0.16666666666666666 | 0.5565130652800339 | -0.07407407407407407 | true |

**r1 and r2 read identically in this table -- verified as independent runs, not a duplicate shard.** Held-out
accuracy is discrete (k of 54 items), so ties across independently-drawn task replicates are possible; here every
arm bar frozen sits exactly at the majority-class count (9 of 54) in both replicates, and frozen ties at 13 of 54
in both. The two replicates are genuinely distinct draws: different `task_seed` (10014 vs 20021), different oracle
ceiling (`oracle_heldout` 0.8888888888888888 vs 0.9259259259259259 in the two shards, matching
`gpu_s7_e40.json`'s `per_seed.7.replicates[1].oracle` and `[2].oracle`), and different `ff_weight_moved` in the
underlying shards (transport_ceiling: 10419652.75 for r1 vs 9406836.75 for r2). The tie is a property of a
54-item discrete accuracy floor at chance, not a re-run artifact.

Every hidden-learning arm (fixed_fa, micro_inengine, transport_ceiling) reads EXACTLY at chance on every replicate
(9 of 54 held-out items correct, the majority-class count). `n_fa_wall = 3` of 3, `n_surpass = 0`. `train_acc` for
the ceiling sits at 0.1725/0.17/0.1625 against training-chance 0.1825/0.170/0.1625 (task seeds 7/10014/20021) --
at or below chance in every replicate, the same collapse item 5 of the diagnosis finding named at dev size, now
total rather than partial.

**Decision, applying the registered rule exactly.** Rule B: "A replicate is interpretable only if the ceiling
clears chance (one-sided binomial p < 0.05) AND its headroom over frozen is at least 0.05." No replicate clears
chance at all (p 0.557 on all three, nowhere near 0.05), so `n_interpretable = 0`. Rule B also needs a seed <!--derived-->
DEFINED iff >=2 of 3 replicates are interpretable; **0 of 3 << 2 of 3, so seed 7 is NOT DEFINED at full size.**
Per AMENDMENT 5 F: "UNDEFINED keeps the instrument at dev, and the next lever applies." C21 does not transfer to
the 2026-09-15 net size. The cupy wall time is 4.268-6.693 ms/step across arms and replicates (`ms_per_step` in
each shard), for the evaluation-budget record AMENDMENT 5 F also asked for.

## What this changes

Both open rungs are now closed as UNDEFINED, and neither is a dead end: the census gives a positive, actionable
answer (the clamp is load-bearing) and the full-size run gives a negative-but-informative one (the dev
calibration's best config does not transfer, and it fails in the SAME direction -- collapse toward the readout's
own baseline class -- only more completely). Read together: at full size, with the identical +-12 clamp, the same
saturation dynamic the census measured at dev size is the leading candidate for why nothing separates from chance
at all (the census was not itself run at full size, so this is the next thing to check, not yet measured).

The pre-registration's own next-rungs list (in the diagnosis finding) ordered the bound census (0) ahead of
lateral inhibition (1) and output homeostasis (2), with the full-size run (3) run in parallel because it had
already been launched. That order is now confirmed by rung 0's own result: the bound is load-bearing, so it is the
next lever to build, not lateral inhibition or output homeostasis.

**The exact next dev command (NOT run by this finding; a proposed calibration round, to be registered as an
amendment before it runs).** The minimal isolating test is the same small net and the same C21 flags with only the
clamp relaxed, to check whether the collapse is specifically the numeric bound (before building the biological
companion process -- synaptic scaling / weight normalization -- that a real weight bound would sit alongside):

```
.venv/bin/python -m research.runners._gap4_transport_ceiling_readout_derisk \
  --seeds 7 --hidden 32 --pool-k 4 --train-subsample 400 --epochs 30 \
  --read-quantity spikes --settle-steps 40 --read-window 30 --read-gain 20 --isi-steps 0 \
  --eval-frozen --spi-silence-outside-credit --no-structural-plasticity --no-ff-stp \
  --ff-w-init 40 --propagation-strength 0.5 --tonic-h-pA 225 --tonic-o-pA 250 --pbar-alpha 0 \
  --lr 5 --hidden-lr-gain 0.2 --bdsp-w-max 48 \
  --replicates 0 1 2 --arms frozen fixed_fa micro_inengine transport_ceiling \
  --out <new round6 directory, chosen when this is registered as an amendment>/calib_C25_s7.json
```

(`--bdsp-w-max 48` is a 4x relaxation of the clamp; everything else is C21 unchanged.) If the ceiling still cannot
clear chance with the clamp far away, the bound is excluded and rungs 1-2 (lateral inhibition, output homeostasis)
move up. If it clears, the weight bound is confirmed as the mechanism and the real lever is a biological
replacement (synaptic scaling), not simply a bigger constant.

## Flip-candidate status

Not applicable. Nothing here is a production default; this is instrument/lever diagnosis inside the gap#4 crux
lane, still upstream of any dev config that clears the interpretability gate. Distance to the owner's flip bar
(6-seed GO + SOUND review + no-regression battery + production-default validation) remains large: no config in
C0-C25 (once C25 above is run) has cleared the interpretability gate at dev size on this task; the evaluation-seed
guard (AMENDMENT 5 D) still refuses without a committed EVALUATION CONFIG section; and no anti-cheat battery
(apical lesion, freeze-spi, no-transport guard, AST guard, two-build seed check) has been run because nothing has
qualified to run it against yet.

## Functional read-outs only

Every number above is a spiking-network readout accuracy, a synaptic weight statistic, or a binomial test over
those readouts. Nothing here is a claim about experience.
