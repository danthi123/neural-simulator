---
type: preregistration
status: preregistered
date: 2026-09-24
lane: F · gap#4 deep credit (the crux lane; plan step S19 / GPU step G5)
mechanism: READ-REGIME repair of the gap#4 transport-ceiling instrument. A longer output read (the time-averaged
  pooled event rate over the last W steps of a longer settle, instead of the ~10 ms low-pass snapshot), a stronger
  output error read (divisive-normalization gain g on the softmax over output rates), an inter-stimulus relaxation
  interval (no stimulus, no teaching) after each credit phase, plasticity frozen during evaluation reads, and the
  in-engine interneuron rate silenced outside the credit phase. Runner:
  research/runners/_gap4_transport_ceiling_readout_derisk.py (new; no sim/ edit; every new knob defaults to the
  legacy value). Arms at ONE cfg.seed: frozen (reservoir: hidden apical 0, only the readout learns), fixed_fa,
  micro_inengine, transport_ceiling. R=3 task replicates per seed for FA-wall coverage.
seeds: dev 7 (calibration + dev run, no pre-registered weight); evaluation 42 (de-risk, 1 seed, G5 GPU) then
  43/44/100/101/102 overnight under this same document.
verdict: PREREGISTERED -- no evaluation-seed run exists at commit time. The seed-7 calibration below is run AFTER
  this commit, and an amendment fixing the evaluation config is committed BEFORE seed 42 is queued.
external: Bellec et al. (2020) Nat Commun 11:3625 (e-prop; readouts are leaky INTEGRATOR neurons with their own
  membrane time constant); Mazurek et al. (2026) Front Neurosci "Operational manifolds in spiking neural networks"
  (accuracy depends on the inference integration horizon); Carandini & Heeger (2012) Nat Rev Neurosci 13:51
  (divisive normalization as a canonical cortical computation). Search recorded with tools/record_external_search.sh.
  AMENDMENT 6 adds: Payeur et al. (2020) bioRxiv doi:10.1101/2020.03.30.015511 / (2021) Nat Neurosci 24:1010
  doi:10.1038/s41593-021-00857-x (the slow moving-average baseline); Bienenstock, Cooper & Munro (1982) J Neurosci
  2:32 (the sliding threshold); van Rossum, Bi & Turrigiano (2000) J Neurosci 20:8812 and Royer & Pare (2003) Nature
  422:518 (the alternatives weighed).
builds_on:
  - research/findings/2026-09-15-gap4-inengine-selfpredict-interneuron-UNDEFINED-transport-ceiling-foreclosed.md
  - research/findings/raw/gap4/_aggregate_5seed.json
  - research/findings/2026-08-11-gap4-ALLIN-ARC-SUMMARY-a-spiking-deep-credit-WALL-was-a-hyperparameter-READ-BEFORE-RE-ATTACKING.md
---

# gap#4 transport-ceiling readout lever — PRE-REGISTRATION

## Why (what the record says, and what it does not)

<!--derived-->
(Numbers in this section are quoted, rounded to 3 places, from research/findings/raw/gap4/_aggregate_5seed.json and
from the per-arm lines that research/queue/gpu_queue.log printed for the 2026-09-10 seed-42 and seed-43 runs.)

The 2026-09-15 run returned UNDEFINED on 5 seeds: the transport_ceiling arm (feedback Y := pooled forward W
transposed, the labeled weight-transport cheat) never cleared chance (0.093-0.167 vs chance 0.167 on 54 held-out
inheritance items), so nothing beneath it was readable. The finding names the next lever: a stronger or longer
readout, plus more FA-wall coverage per seed (n_fa_wall was 0-1 per seed).

The seed-42/43 lines of research/queue/gpu_queue.log (2026-09-10) add a fact the finding did not state: TRAIN
accuracy was at chance in EVERY arm, the frozen reservoir (only the readout learns) at 0.163 and the ceiling at
0.152 on 9 classes. The output readout does not fit the training set even when the hidden layers are frozen. So
the instrument fails upstream of deep credit: the read, or the readout's learning, cannot carry the class.

## The companion-process question (asked before "what biology surpasses this")

What does the real system run alongside this that the runner replaced with a constant?

1. **Integration time of the read.** The read is `cp_bdsp_E` at the last settle step: a per-step low-pass with
   `bdsp_rate_tau = 0.90`, i.e. about the last 10 ms of spikes (a few spikes per neuron at rates of 0.05-0.2 per
   ms). Decision readouts integrate evidence over 100s of ms, and e-prop's readout is a leaky integrator with its own
   time constant. Lever **W**: average the pooled event rate over the last W steps of a settle of S steps.
2. **Normalization of the output population.** The error that trains the readout is `softmax(rates) - onehot`. With
   rates in [0, 0.3] over 9 classes the softmax is within a few percent of uniform, so the error hardly depends on
   what the network currently outputs. The readout then learns class centroids, which cannot separate classes
   built from XOR pairs. Lever **g**: a normalization gain, `softmax(g * rates)`, so the error shrinks when the
   right class already dominates. This is the output population's divisive normalization. It is a parameter of the
   host credit computation that the whole arc already declares as a host residual; it adds no new host step.
3. **An inter-trial interval.** The burst-probability baseline `Pbar` is an EMA with alpha 0.05 per step (about
   20 steps). After each 25-step credit phase the apical relaxes (tau 15 ms) while `Pbar` lags, so `P - Pbar` goes
   NEGATIVE during the next example's settle, whose input is driving `Etilde_pre`. The update then takes a negative
   copy of the teaching signal paired with the WRONG input. Lever **isi**: I steps with no stimulus and no teaching
   after each credit phase, so the negative lobe lands on spontaneous activity, not the next example.
4. **Two instrument fixes, not levers.** (a) The in-engine interneuron keeps projecting `cp_spi_int_rate` after the
   credit phase (the engine re-forms `cp_bdsp_int_drive` every step, while the runner zeroes the raw top-down), so
   micro_inengine carried an uncancelled negative apical into the next forward pass and into evaluation reads. The
   runner-supplied `micro` arm does not. Fix: zero the interneuron rate outside the credit phase
   (`--spi-silence-outside-credit`). (b) BDSP plasticity runs on every step, including evaluation forwards. Fix:
   `bdsp_learning_rate = 0` during evaluation reads (`--eval-frozen`), restored afterwards.

## Arms (all at ONE `cfg.seed` per seed; the same neurons in every arm)

| arm | hidden credit | role |
|---|---|---|
| frozen | none (hidden apical 0; only the output layer learns) | the credit-independent floor |
| fixed_fa | fixed random Y | the frozen-signal baseline to beat |
| micro_inengine | fixed Y top-down minus the in-engine learned interneuron cancellation | the mechanism under test |
| transport_ceiling | Y := pooled forward W transposed each example | the interpretability ceiling (a labeled cheat) |

The runner-supplied `micro` arm is dropped (already tested; not needed for this question).

## Replicates (FA-wall coverage)

R = 3 task replicates per seed. Replicate r uses task seed `seed` for r = 0 (the exact task of the 2026-09-15
run) and `seed + 10007*r` for r >= 1; train-order and subsample RNGs are keyed the same way. The substrate
`cfg.seed = seed` in every replicate and arm. Each (seed, replicate, arm) is checkpointed to its own JSON on
completion; a restarted run skips completed shards (config fingerprint must match).

## Metrics (per replicate; chance = the majority-class rate of the held-out inheritance items, as before)

- `inherit_heldout(arm)`: held-out inheritance accuracy (54 items per replicate), the arc's metric.
- `train_acc(arm)`: accuracy on the training subsample.
- `decode_h2(arm)` (instrument only, host ridge regression on the pooled top-hidden read, fit on train, scored on
  held-out): is the class present in the hidden read at all? Never used as a pathway or in any gate below.
- `ceiling_clears_chance`: transport_ceiling held-out above chance with a one-sided binomial p < 0.05 over the
  held-out items.
- `headroom = transport_ceiling - frozen`; `deep_credit_share(arm) = (arm - frozen) / headroom`, reported only when
  `headroom >= 0.05` (else UNDEFINED, never 0).
- `fa_wall`: `fixed_fa <= frozen + 0.02`. `n_fa_wall` = the number of replicates (of 3) where it holds.
- `surpass`: `micro_inengine > fixed_fa + 0.05`.

## Dev calibration (seed 7 only; carries no pre-registered weight)

Net: hidden 32, pool_k 4, 2 hidden layers, epochs 10, train subsample 400, replicate 0, arms frozen and
transport_ceiling, `SIM_BACKEND=numpy`. Configs:

| id | settle S | read window W | gain g | isi I | eval-frozen, spi-silence |
|---|---|---|---|---|---|
| C0 | 40 | 0 (legacy snapshot) | 1 | 0 | off (legacy) |
| C1 | 100 | 80 | 1 | 0 | on |
| C2 | 40 | 0 | 20 | 0 | on |
| C3 | 100 | 80 | 20 | 0 | on |
| C4 | 100 | 80 | 20 | 60 | on |

**Selection rule (fixed now).** Among C1-C4, keep configs whose transport_ceiling clears chance (binomial p < 0.05)
AND has headroom >= 0.05 over frozen. Choose the largest headroom; within 0.02 of it, choose the fewest steps per
example. If none qualifies, the instrument stays UNDEFINED at dev: bank that with the measured ceilings, do NOT
queue seed 42, and G5 runs the SlotBinder fallback (plan step G5).

**Dev run (after selection).** The chosen config on seed 7, all 4 arms x 3 replicates, same small net. It reports
n_fa_wall, ceiling definedness and the per-step wall time used for the cupy extrapolation.

## Evaluation config (fixed by an AMENDMENT before seed 42 is queued)

Net hidden 64, pool_k 16, 2 hidden layers, train subsample 400 (the 2026-09-15 net), `SIM_BACKEND=cupy`, the knobs
of the selected config. Epochs: fixed in the amendment from the dev learning curve and the measured budget, and
stated with its reason. Transfer risk, stated now: the calibration is at a smaller net; a config that clears at
dev may not clear at H64/K16. That outcome is itself reported as UNDEFINED, not as NO-GO.

## Verdict rules

**Seed 42 alone (status: de-risk, 1 seed).** No GO is claimed from one seed. Reported: per-replicate arm table,
`n_fa_wall`, `ceiling_clears_chance`, `headroom`, `deep_credit_share` for fixed_fa and micro_inengine, the surpass
count, `decode_h2`. The seed is UNDEFINED if the ceiling clears chance on fewer than 2 of 3 replicates.

**Six seeds (42, 43, 44, 100, 101, 102), for the overnight completion.** GO iff all of:
1. the ceiling clears chance on >= 2/3 replicates on >= 5/6 seeds (else UNDEFINED, the interpretability gate);
2. `micro_inengine > fixed_fa + 0.05` (replicate mean) on >= 5/6 seeds;
3. `deep_credit_share(micro_inengine) > deep_credit_share(fixed_fa)` on >= 5/6 of the seeds where both are defined;
4. anti-cheats, run on each seed where 1 and 2 hold: apical lesion (hidden Y zeroed) and freeze-spi (interneuron
   never learns) each fall below micro_inengine by >= 0.05; the ceiling's no-transport guard FAILS; the AST guard
   finds no forward-weight read in the credit path; two builds at one cfg.seed give identical firing thresholds.

NO-GO iff 1 holds and 2 fails. The FA-wall-regime claim is made only on seeds with `n_fa_wall = 3`; seeds with
fewer are reported as "FA-wall coverage incomplete", and the unconditional comparison in 2 still stands.

## Declared host residuals (brain-based-only standard)

The credit projection (softmax of output rates, the error, Y @ error, the transport copy) and the argmax read-out
are host computations, as in every gap#4 runner; the gain g is a parameter of that residual. The ridge decode is an
instrument. The forward pass, the feedforward weight changes (BDSP kernel) and the in-engine interneuron learning
are on the spiking substrate. Functional read-outs only; nothing here is a claim about experience.

## Byte-identity (to be shown in data)

With every new knob at its legacy value (S 40, W 0, g 1, I 0, both fixes off) the new net class must produce the
same forward-weight bytes and the same reads as `Gap4InEngineNet` after a few training examples on seed 7; the
runner's `--identity-selftest` writes that comparison (weight hash, read hash) to an artifact. No sim/ file changes,
so production behaviour on main is untouched.

## AMENDMENT 1 (2026-09-24 ~12:45 EDT, before round 2 runs; dev seed 7 only)

**What round 1 showed (dev data, no pre-registered weight).** Artifacts:
research/findings/raw/gap4/transport_ceiling_readout/round1_rev9654a99/ (commit 9178f9449).

<!--derived-->
- C0-C3: the transport ceiling reads 0.056 held-out and about 0.05 train in every config; the frozen readout 0.074-0.093
  held-out and 0.048-0.087 train. Train accuracy is BELOW the 1/9 chance of 9 classes, so the readout learns the wrong way.
  [Erratum, AMENDMENT 5 A: the training chance is the majority-class rate, 0.1825 here, not 1/9; the reading stands.]
- Diagnostic `diag_eread_monotonic_s7.json`: the BDSP event read `E` (isolated or first-of-burst spikes) is
  NON-MONOTONIC in drive. Extra output current 0 -> +1600 pA raises the total spike rate 86 -> 404 Hz but lowers `E`
  0.050 -> 0.002. At the default tonic drive the output layer sits at the peak of `E`, so any LTP onto an output
  neuron LOWERS its read. That explains below-chance training accuracy, and it is independent of the window W and
  the gain g, which is why C1-C3 did not move it.

**The added lever (still "a stronger readout").** `--read-quantity spikes`: the read counts EVERY somatic spike in the
window (events plus burst spikes), per step, which is monotonic in drive. A downstream neuron receives every spike, so
this is the quantity a spiking readout integrates. The BDSP kernel itself is unchanged (it still uses `E` internally).
**Companion process found in the same pass:** the default-on synapse elimination (`enable_structural_plasticity`)
zeroes weights below 0.05, which includes every NEGATIVE signed feedforward weight, at 5e-7 per step; over the
2026-09-15 runs (about 1.06M steps per arm) that is roughly a third of the negative weights. `--no-structural-plasticity`
turns it off; `--eval-frozen` now also pauses it during evaluation reads (it was editing weights during reads).

**Round-2 grid** (same small net, seed 7, epochs 10, subsample 400, replicate 0, arms frozen and transport_ceiling;
all with `--eval-frozen --spi-silence-outside-credit`):

| id | settle S | window W | read | gain g | isi I | structural plasticity |
|---|---|---|---|---|---|---|
| C5 | 100 | 80 | spikes | 1 | 0 | on (default) |
| C6 | 100 | 80 | spikes | 20 | 0 | on |
| C7 | 100 | 80 | spikes | 20 | 60 | on |
| C8 | 40 | 30 | spikes | 20 | 0 | on |
| C9 | 100 | 80 | spikes | 20 | 60 | off |

**Selection rule: unchanged**, applied over C1-C9 (C4's round-1 result included when it lands). If nothing
qualifies, the prereg's fallback stands: the instrument is UNDEFINED at dev and seed 42 is not queued.

## AMENDMENT 2 (2026-09-24 ~13:15 EDT, before round 3 runs; dev seed 7 only)

**What round 2 and the diagnostics showed (dev data, no pre-registered weight).** Artifacts:
research/findings/raw/gap4/transport_ceiling_readout/round2_rev5c3a865/ (commit b66314f92).

<!--derived-->
- The spike read removed the below-chance training accuracy (C8 frozen train 0.110, ceiling 0.098) but nothing
  learned: the ceiling stayed at 0.056-0.074 held-out. [Erratum, AMENDMENT 5 A: against the training chance 0.1825
  both are still below chance, so the spike read did NOT remove the below-chance training accuracy.]
- **The forward pathway does not transmit.** `diag_transmit_scan_*`: with the default Tsodyks-Markram short-term
  depression ON, H1/H2/output rates do not change when `ff_w_init` goes 4 -> 40 or `propagation_strength`
  0.05 -> 0.5, at any tonic level; at tonic 0 the input layer fires at 0.096/ms and H1 stays at 0.005/ms. Scaling
  the H2->out weights x0, x1, x3, x10 leaves the output read unchanged. With STP bypassed on the feedforward
  synapses, ff 40 / ps 0.5 transmits (tonic 0: H1 0.035/ms; tonic 0.5: H2 0.083/ms).
- `diag_lr_scan_*`: at lr 8 the readout weights move by |dw| 1.0 (|w| 0.38) and training accuracy stays 0.08.
- C9 (no synapse elimination): the frozen arm's total weight movement is 2.1, against about 1270 with elimination on,
  so the "ff-moved" totals of every earlier run are mostly elimination, not BDSP learning.

**Reading.** In this read-regime no arm's learning can reach the output: the transport ceiling was never
askable, which is the mechanism behind the 2026-09-15 UNDEFINED. The engine's own STP block documents the same
effect for sustained stimulus-driven firing (an effective multiplier near 0.07 at U=0.15, tau_d=200 ms).

**Round-3 grid: the operating point** (same small net, seed 7, epochs 10, subsample 400, replicate 0, arms frozen and
transport_ceiling). Common: `--read-quantity spikes --settle-steps 40 --read-window 30 --read-gain 20 --isi-steps 0
--eval-frozen --spi-silence-outside-credit --no-structural-plasticity --no-ff-stp --ff-w-init 40
--propagation-strength 0.5 --bdsp-w-max 12`. `--no-ff-stp` bypasses the STP factor on the explicit feedforward
synapses only; the recurrent background keeps STP.

| id | tonic (hidden / output pA) | lr |
|---|---|---|
| C10 | 225 / 250 | 0.05 |
| C11 | 225 / 250 | 1.0 |
| C12 | 225 / 250 | 5.0 |
| C13 | 0 / 0 | 1.0 |
| C14 | 450 / 500 | 1.0 |

**Selection rule: unchanged**, over C1-C14. Fallback unchanged. The per-arm learning rate follows the arc's
meta-lesson #1 (one shared lr is an unfair A/B); if the evaluation config uses one lr for all arms, the amendment
fixing it says so and why.

## AMENDMENT 3 (2026-09-24 ~13:20 EDT, before round 4 runs; dev seed 7 only)

**What round 3 showed (dev data).** Artifacts: research/findings/raw/gap4/transport_ceiling_readout/round3_rev7dfb386/
(commit 4dcbec29f).

<!--derived-->
- With feedforward STP bypassed, the ceiling begins to fit the training set as lr rises (lr 1: train 0.180; lr 5:
  train 0.203; 1/9 chance), while held-out stays 0.074-0.130 (chance 0.167). Nothing qualifies yet.
  [Erratum, AMENDMENT 5 A: the training chance is 0.1825; 0.180 is at chance and 0.203 is not above it (p 0.17).]
- The frozen readout's argmax collapses onto one or two output units (often the never-taught class 8): baseline rate
  differences between output units outweigh the learned class selectivity.

**Companion process named for round 4: the burst-probability baseline.** `Pbar` is an EMA (alpha 0.05/step) that
returns to p0 after every teaching transient, so the integral of `P - Pbar` over a presentation is close to zero, and
the negative lobe lands on the next example's input. `--pbar-alpha 0` presets the baseline at p0 (BurstCCN's preset
form); an inter-stimulus interval is the protocol alternative. Longer training is the third candidate.

**Round-4 grid** (seed 7, same small net, subsample 400, replicate 0, arms frozen and transport_ceiling). Common:
round 3's common flags plus `--tonic-h-pA 225 --tonic-o-pA 250`.

| id | lr | pbar alpha | isi | epochs |
|---|---|---|---|---|
| C15 | 5 | 0 | 0 | 10 |
| C16 | 20 | 0.05 | 0 | 10 |
| C17 | 20 | 0 | 0 | 10 |
| C18 | 5 | 0.05 | 0 | 30 |
| C19 | 5 | 0 | 40 | 10 |

**Selection rule: unchanged**, over C1-C19; for equal headroom (within 0.02) the cheaper config means fewer total
training steps (epochs x steps per example). Fallback unchanged.

## AMENDMENT 4 (2026-09-24 ~13:30 EDT, before round 5 runs; dev seed 7 only)

**What round 4 showed (dev data).** Artifacts: research/findings/raw/gap4/transport_ceiling_readout/round4_rev8f16994/
(commit 1d8fcd669).

<!--derived-->
- The preset baseline (`--pbar-alpha 0`) lets the frozen readout fit the training set (C15, C17: train 0.265).
- The ceiling reaches 0.204 held-out at C15 (not significant on 54 items), with train 0.168, BELOW the frozen
  readout's train 0.265: at lr 5 on every layer the hidden weights move fast (ff-moved 127707) and the readout loses
  ground. The step size is shared across layers of very different fan-in, which the arc's meta-lesson #1 warns about.
- Rate backprop oracle, online batch 1, lr 0.05 (`diag_oracle_online_budget_s7.json`): held-out 0.81 at H32 after
  4000 updates (10 epochs x 400) and 0.94 after 8000; at lr 0.3 it never clears. Ten dev epochs sit at the exact
  gradient's own threshold, so the dev budget, not only the rule, can hold the ceiling at chance.

**Round-5 grid** (seed 7, same small net, subsample 400, replicate 0, arms frozen and transport_ceiling). Common:
round 4's common flags plus `--pbar-alpha 0 --isi-steps 0`. `--hidden-lr-gain` scales the BDSP step on synapses onto
hidden neurons through the committed per-synapse plasticity gain; the output pathway keeps the full lr.

| id | lr (output) | hidden lr gain | epochs |
|---|---|---|---|
| C20 | 5 | 1.0 | 30 |
| C21 | 5 | 0.2 | 30 |
| C22 | 5 | 0.05 | 30 |
| C23 | 20 | 0.05 | 30 |
| C24 | 5 | 0.2 | 20 |

**Selection rule: unchanged**, over C1-C24 (cost = epochs x steps per example). Fallback unchanged.

## AMENDMENT 5 (2026-09-24 ~17:20 EDT, review fix round; before the transmission diagnostic and the bound census run, and before any evaluation amendment)

Why: the independent review of the lane (verdict "not safe to merge yet") found wrong chance references, an overstated
causal claim, a lesion that was not a matched cut, an evaluation-seed guard that could not fail, a NO-GO readable with
no headroom, and an unmeasured weight clamp. Everything below is fixed now, before any evaluation seed has run. No
evaluation seed has run at the commit of this amendment.

<!--derived-->
(Numbers in item A are computed, not read from a stored run artifact: the per-seed/per-replicate training-chance
figures are the majority-class count of the training subsample built by the runner's own task construction --
`_task()` in `research/runners/_gap4_transport_ceiling_readout_derisk.py`, `train_chance =
np.bincount(yb, minlength=k).max() / len(yb)` over the subsample `np.random.default_rng(task_seed * 13 + 1)` draws
from `make_task_semantic_inheritance(task_seed, ...)`, `task_seed = seed if r == 0 else seed + 10007 * r` -- and are
reproducible by calling that code directly, with no training run needed. No evaluation seed has run at this
amendment's commit, so none of these could come from a run artifact. The re-quoted C8 / ceiling-train / 2026-09-15
figures below repeat AMENDMENT 2, AMENDMENT 3 and the Why section above, each already under its own `<!--derived-->`
marker there -- round2_rev5c3a865, round3_rev7dfb386, and `research/queue/gpu_queue.log`'s 2026-09-10 seed-42/43
lines respectively.)

**A. Training chance (erratum to AMENDMENTS 2 and 3).** Training accuracy is compared with the majority-class rate of
the 400-item training subsample, the same chance definition this prereg uses for held-out items. It is not 1/9: class 8
has no training items on this task. Seed 7: r0 0.1825, r1 0.170, r2 0.1625. Evaluation seeds (r0/r1/r2): 42
0.1825/0.1725/0.195; 43 0.190/0.1975/0.170; 44 0.1775/0.155/0.1825; 100 0.165/0.1725/0.185; 101 0.195/0.165/0.1575; 102
0.170/0.1625/0.1725. Corrected readings (replicate 0, chance 0.1825): AMENDMENT 2's C8 frozen 0.110 and ceiling 0.098
are still BELOW chance, so the spike read did not remove below-chance training accuracy. AMENDMENT 3's ceiling train
0.180 (lr 1) is at chance and 0.203 (lr 5) is not above it (binomial p 0.17). The Why section's "at chance" for the
2026-09-15 train accuracies (0.163, 0.152) stands, since seed 42 r0 and seed 43 r0 have majority rates 0.1825 and 0.190.
The runner now writes `train_chance` and `train_binom_p` into every shard.

**B. Rule 1 (interpretability gate) now needs headroom.** A replicate is interpretable only if the ceiling clears
chance (one-sided binomial p < 0.05) AND its headroom over frozen is at least 0.05. A seed is DEFINED iff at least 2 of
3 replicates are interpretable; this applies to the seed-42 de-risk and to the six-seed rule 1 (at least 5 of 6 seeds
DEFINED). NO-GO still needs rule 1 to hold, so a negative can no longer be read off an instrument with no headroom. The
runner reports `n_interpretable` beside `n_ceiling_clears_chance`.

**C. Rule 4 apical lesion is a matched cut.** In `micro_inengine_lesion` the runner zeroes the interneuron rate
`cp_spi_int_rate` as well as the hidden Y, so the engine-formed cancellation `int_drive` is zero and the lesioned top
hidden apical receives nothing. Before this fix the untrained cancellation kept driving it (review probe: 17.35 mV max
|v_apical - rest|; this runner's `--lesion-selftest`, tiny net: 49.2 mV unmatched, 0.0 mV matched). Every lesion shard
records `lesion_hidden_apical_max_abs_dev_mV` over all hidden neurons during training; the lesion counts as held
(docs/TERMS.md "lesion") only if that is at most 1e-6 mV. Lesion shards without the matched-lesion tag are never
resumed or aggregated.

**D. Evaluation-seed guard.** An evaluation seed runs only if `--prereg-amendment` names a file whose COMMITTED content
(git HEAD, or a manifest-verified git archive on the pool) holds a markdown heading that starts with "AMENDMENT" and
contains the words "EVALUATION CONFIG", with, inside that section, a line `evaluation-config-fingerprint: <16 hex>`
equal to the run's config fingerprint (`--print-fingerprint`) and a line `evaluation-seeds: ...` listing every
evaluation seed requested. This file has no such section, so passing it refuses. `--guard-selftest` shows the refusals.

**E. Calibration tie-break.** `select_calibration` breaks ties by total training steps (epochs x steps per example), as
AMENDMENTS 3 and 4 registered; the code had used steps per example only. Re-running it over C0-C24 must still give
NONE QUALIFIES (nothing qualified, so the tie-break never applied).

**F. Dev runs not registered before they ran (disclosed, no pre-registered weight).** (1) The dev run at C21, 4 arms x
3 replicates (rev 1ec6635), and (2) the full-size timing burst, both reported in the 2026-09-24 finding. (3) The
full-size GPU dev run on seed 7, `research/queue/_a9_gap4_tc_gpu.sh` at pin 10479d530 (C21 flags at H64/pool 16, 40
epochs, 4 arms x 3 replicates), started 15:54:59 EDT, before this amendment; its output had not been read when this was
committed. What it decides: whether C21 transfers to the 2026-09-15 net size. Seed 7 is read under rule B (DEFINED iff
at least 2 of 3 replicates interpretable), by re-aggregating its shards with this revision's runner (same fingerprint;
the shards carry every field rule B needs). DEFINED makes C21-at-full-size the candidate for an evaluation-config
amendment (section D), which is still required. UNDEFINED keeps the instrument at dev, and the next lever applies. It
also measures the cupy ms per step for the evaluation budget.

**G. Transmission diagnostic with a noise reference (dev seed 7).** Runner
`research/runners/_gap4_tc_transmission_noise_ref_diag.py`, 24 training items, reads event (legacy) and spikes (W 30),
the 2x2 of feedforward STP (on / bypassed) x gain (ff 4, propagation 0.05 / ff 40, propagation 0.5). Full size
(H64/pool 16) at tonic 1.0, the 2026-09-15 operating point; dev size (H32/pool 4) at tonic 1.0 and 0.5. Per layer:
effect/noise (mean |intact - cut| over mean |intact - intact|) and the across-item reliability of each unit's read.
What it decides: the STP attribution. If at full size only STP-bypassed-plus-gain reads effect/noise above 2 with item
reliability above 0.6 (spike read), the 2026-09-15 cause is recorded as two-factor (default STP at the legacy gain);
if the legacy variant already shows item reliability above 0.3, "the legacy net barely transmits" is withdrawn for
the full-size net.

**H. Bound census (dev seed 7).** Review issue: at C21 the hidden-learning arms end at a mean |w| of 8.7-10.1 against
the +-12 bound, so the clamp may own the weight change. Run: C21 flags on the small net (H32/pool 4, 30 epochs), 4 arms x
3 replicates, one pool job per shard, at the revision that carries this amendment. Each shard records, per feedforward
pathway, the fraction of synapses at (or beyond) +-bdsp_w_max at build and after training, mean and max |w|, and
tools.lab.bound_check at build. Consistency check: the r0 shards must reproduce the dev run's r0 reads exactly (the
census only reads weights), else the census is VOID. Decision: the clamp is load-bearing if, in the transport_ceiling
arm, at least 10% of the synapses of a hidden-post pathway (ff_0 or ff_1) end at +-w_max on at least 2 of 3
replicates. Then the next lever is the bound (the companion process that the static clamp replaced) before lateral
inhibition or output homeostasis. Otherwise the clamp is excluded as the cause of the residual and the next-lever order
stands.

## AMENDMENT 6 (2026-09-25 ~02:10 EDT, the clamp's companion process; before any run of C25-C27; dev seed 7 only)

**Why.** AMENDMENT 5 H's census found the +-12 clamp load-bearing
(`research/findings/2026-09-24-gap4-transport-ceiling-bound-census-clamp-load-bearing-fullsize-UNDEFINED.md`). Read
again from its shards (`research/findings/raw/gap4/transport_ceiling_readout/bound_census_revafe2b32/ckpt/s7_r*_*.json`),
the saturation is ONE-SIDED and common to every arm that trains the hidden weights:

<!--derived-->
- In fixed_fa, micro_inengine and transport_ceiling, 17-80% of ff_0 and ff_1 end at +w_max and at most 0.2% at w_min,
  on all 3 replicates. Mean |w| of those two pathways goes from 3.83 / 3.09 at build to 9.7-11.7. In the frozen arm
  both stay at 3.83 / 3.09 (two decimals) with no synapse at either bound.
- In the same arms the output layer goes nearly silent: its mean held-out read is 0.001-0.09, against 0.17-0.19 in
  the frozen arm, and the ridge decode of the top hidden layer on its own training items falls from 0.42-0.48
  (frozen) to 0.27-0.32.

**What the real system runs alongside this, that the runner replaced with a constant.** The BDSP rule's source
(Payeur et al., bioRxiv 2020.03.30.015511 v1; Nat Neurosci 2021) sets the baseline to "a moving average of the
proportion of events that are bursts in postsynaptic neuron i, with a slow (~ 1 – 10 s) time scale", and states
why: "To ensure a finite growth of synaptic weights". Its Methods use the ratio of two exponential moving averages
(burst train over event train), with tau_avg 5 s in the XOR task. C21 presets the baseline to the constant p0 = 0.3
(`--pbar-alpha 0`), because the engine's only moving baseline was an EMA of the instantaneous P at 0.05 per step
(about 20 ms), which averaged each teaching transient away (AMENDMENT 3). Mechanism, stated as the hypothesis this
amendment tests: with p0 = 0.3 the burst-probability sigmoid is convex, so a credit that is zero on average still
raises mean P above p0. A fixed baseline then turns that excess into potentiation on every active synapse,
whatever the sign of the credit, and the clamp catches it. The engine unit test
(`tests/test_bdsp_pbar_ratio.py::test_ratio_baseline_cancels_the_one_sided_drive_a_zero_mean_apical_gives_the_preset_baseline`)
shows the direction in the engine: under a zero-mean apical current the hidden neurons' summed E*(P - Pbar) is 54.6
with the preset baseline and 3.2 with the ratio baseline at tau 200 ms. <!--derived-->
Biology record: `research/biology/bdsp-sliding-burst-baseline.md` (it also records the two alternatives weighed and
not chosen: weight-dependent soft bounds, and heterosynaptic conservation of total weight).

**The lever (engine + runner, both additive and default-off).** `cfg.bdsp_pbar_ratio_tau_ms` (sim/config.py,
sim/bridge.py): when above 0, each masked neuron's baseline is Pbar = EMA(B_post) / EMA(E), both with time constant
tau, where B_post is the burst factor the kernel already uses (E*P under graded credit). The EMAs start from an event
rate of 0.05 with Pbar = p0. Runner flags: `--pbar-ratio-tau-ms` (default 0, off) and `--pbar-ratio-layers hidden|all`
(default hidden: the hidden neurons, whose pathways the census names; the output keeps C21's preset baseline).
The +-12 clip stays in the kernel as a backstop, so the census still measures whether it binds.
Byte-identity, shown in data: with the knob at its default, a short BDSP training run hashes to the values recorded
from the engine before the edit (`tests/test_bdsp_pbar_ratio.py`), the runner's `--identity-selftest` still passes in
all four arms, and `--print-fingerprint` reproduces the census fingerprint bd108215d75223ed and the full-size GPU
fingerprint 713efa9804a3dbb0.

**Configs (C21 flags plus the listed change; H32/pool 4, 30 epochs, seed 7, 4 arms x 3 replicates, one pool job
per shard, numpy, at the revision carrying this amendment).**

| id | change from C21 | role |
|---|---|---|
| C26 | `--pbar-ratio-tau-ms 5000` (hidden) | PRIMARY: the companion process, tau_avg 5 s as in the source's XOR task |
| C27 | `--pbar-ratio-tau-ms 5000 --pbar-ratio-layers all` | secondary: the source's form on every neuron |
| C25 | `--bdsp-w-max 48` (no ratio baseline) | control: the relaxation the census finding proposed; a bigger constant, no process |

(C25 is the label the census finding reserved for exactly this command; it was never registered or run until now.)

**Criteria for C26 (the dev check).**
- (i) Bound census. Count the replicates on which, in the transport_ceiling arm at END of training, ff_0 or ff_1 has
  at least 10% of its synapses at +-w_max (the AMENDMENT 5 H quantity). (i) holds iff that count is at most 1 of 3,
  i.e. the rule that found the clamp load-bearing no longer fires. The fraction within 10% of either bound
  (`frac_near_w_max`, `frac_near_w_min`) and the mean signed weight are reported beside it for every arm.
- (ii) Hidden learning no longer collapses the output: the transport_ceiling arm's training accuracy is above its
  replicate's training chance with one-sided binomial p < 0.05 (the shard's `train_binom_p`) on at least 2 of 3
  replicates. The same test is reported for fixed_fa and micro_inengine.
- (iii) Rule B (AMENDMENT 5 B) on the ceiling: seed 7 is DEFINED iff at least 2 of 3 replicates are interpretable
  (ceiling above chance at binomial p < 0.05 AND headroom over frozen at least 0.05).

**Decision (C26).** All three hold: C26 qualifies as the dev config, and the next step is a full-size dev run on the
GPU (H64/pool 16, 40 epochs, seed 7) registered by its own amendment, before any EVALUATION CONFIG amendment.
(i) holds and (ii) fails: the drift is gone but the output still collapses, so the clamp is excluded as the cause of
the collapse and the next lever is event-rate homeostasis (the source's own H/G terms; the diagnosis finding's rung
2), then output lateral inhibition (rung 1). (i) and (ii) hold, (iii) fails: the collapse is repaired and the ceiling
is still not interpretable at the dev budget; the finding reports the budget question (the online rate oracle reads
0.81 held-out after 4000 updates and 0.94 after 8000, per
`research/findings/raw/gap4/transport_ceiling_readout/round4_rev8f16994/diag_oracle_online_budget_s7.json`).
(i) fails: the ratio baseline at 5 s does not stop the saturation; C25 and C27 are read, and the next lever is
weight dependence or heterosynaptic conservation.

**Secondary readings (no gate).** C27 against C26 says whether the output layer's baseline matters. C25 against C26:
if C25 also passes (ii) and (iii), a larger constant would have been enough and the finding says so; if C25 still
collapses, the drift, not the wall's position, caused the collapse. For C25 the census is also reported at |w| >= 12
(`frac_abs_w_ge_c21_bound`), the C21 clamp's magnitude.

**Smoke (declared now, no decision weight).** Before queueing: locally, under `tools/memcap.sh`, the C21 flags at 3
epochs, replicate 0, the transport_ceiling arm only, for C21 itself, C25, C26 and C27. It shows whether the flag
changes the census at a short budget. Its artifacts go to `research/findings/raw/gap4/transport_ceiling_readout/companion_smoke_rev<sha>/`.

**Unchanged.** Evaluation seeds stay locked: the runner's guard still needs a committed EVALUATION CONFIG amendment
registering the fingerprint, and this amendment is not one. Declared host residuals are as in the parent document.
The mask that chooses the hidden neurons is runner configuration. The ratio EMAs are per-neuron state in the engine
step, as the old EMA was. Functional read-outs only.
