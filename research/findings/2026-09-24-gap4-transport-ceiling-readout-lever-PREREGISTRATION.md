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
