---
type: finding
status: provisional
date: 2026-10-26
lane: Stage-2 C4 (discrete-emotion read-out), integration-first; repair of the seed-101 failure of the Stage-2 attempt
mechanism: emotion_category_organ built with the engine's own opt-in exponential-Euler guard cfg.enable_bounded_conductance_integration, behind BRAIN_EMOTION_CATEGORY_BOUNDED (default OFF)
integration_faculty: emotion-category
flip: no - default stays OFF; this is a de-risk of a repair, not a flip
seeds: [301-360 held-out, 1-8 dev, 401-420 fresh, 303-306 stress, 301-310 quench, 101 handler]
artifacts:
  - research/findings/raw/_emo_bounded/aggregate.json
verdict: PROVISIONAL, organ level plus one seed through the handler. With the flag on the organ named all four categories on 60/60 held-out, 8/8 dev and 20/20 fresh seeds (flag off reproduces the pinned 58/60 and misses 1 of 20 fresh), no neuron left the physiological range, and the five pinned failures all pass. Default OFF; nothing flipped.
---

# Emotion-category organ: bounded conductance integration (de-risk, default OFF)

**What was built.** `BRAIN_EMOTION_CATEGORY_BOUNDED` (default OFF; 1/true/yes/on enables) in
`research/runners/emotion_category_organ.py`. When on, `_build` sets `cfg.enable_bounded_conductance_integration = True`
at `CoreSimConfig` construction, the repo's existing opt-in guard (`sim/config.py`; kernel
`fused_bounded_conductance_decay_and_current`, `sim/kernels.py`). Nothing else changes: no `sim/` edit, no host table,
no English rule, no capacitance truncation. Flag unset, every pre-existing field, the trained weights and the dynamical state are identical to main; each read gains
five engagement keys (bounded_flag, bounded_env, bounded_readback, oor_count, oor_count_session). A whole-record
byte-compare of an emo/feel turn across the merge therefore differs by exactly those keys (expected drift).
Every read now carries engagement fields in the `emotion_category` block: `bounded_flag` (what the organ resolved),
`bounded_env` (the raw env string it saw), `bounded_readback` (the bridge's own `core_config` value),
`oor_count` (distinct organ neurons below -110 mV during the read) and `oor_count_session` (the same, over the
session's observes and the read). The census is a read-only min over `v` after each step (about 2 microseconds per
step, 2.5-5% of a bare 59-microsecond step, under 0.5 ms per observe), so it is always computed. `oor_count` alone is
blind to the defect, because the appraisal ensembles are silent during the read (flag-off maximum 0); the session
count is the informative one (flag-off maximum 6).

**Mechanism, corrected label.** Conductance-term overshoot of the explicit-Euler step, not an Izhikevich quadratic
`v_crit`. For `dv = g_i (E_i - v) dt / C`, the step multiplies the distance to `E_i` by `(1 - g_i dt / C)`, which
overshoots and grows once `g_i dt / C` exceeds about 2 (g in nS, dt in ms, C in pF). Seed 101's heterogeneity draw
gives an off-channel appraisal neuron C = 53.64 pF (mean near 100); when the joy channel wins, its inhibitory pool
drives that neuron's `g_i` to 100-136 nS (ratio 1.9-2.5), `v` oscillates with growing amplitude (-96, -24, -116, +18,
-137 mV), the neuron fires 5 spurious spikes, and they recruit the sadness quench pool, which silences the joy rungs
before the hold window. The bounded kernel solves the linear ODE over dt exactly, so one step can only move `v`
toward `E_i`.

## Arms (CPU numpy)

Evidence: `research/findings/raw/_emo_bounded/aggregate.json` (built by `research/runners/emotion_category_bounded_aggregate.py`;
it carries every tally below and the sha256 of each input). The raw per-seed records (about 20 MB) stay in the
workstation scratch directory, uncommitted; the drivers that wrote them are in `docs/drafts/emo-bounded-cond/`.

All organ-level arms run `research.runners.emotion_category_dev.run_seed` (the 10-case battery) from this branch with the
flag taken from the environment, as production reads it. A driver-side wrapper (no repo edit) recorded `v` after every
step of every bridge a seed builds, training included.

| arm | flag OFF | flag ON |
|---|---|---|
| (a) held-out 301-360, all 10 cases | 58/60 (308: permuted, teach_swap, ladder_only; 348: intact, shuffle), the pinned result | 60/60 |
| (a) dev 1-8, all 10 cases | 8/8 | 8/8 |
| intact named latch count (bar >= 36) | held-out min 69, median 86, max 104 | held-out min 61, median 73, max 87; dev min 69 |
| co_commit reads | 0 of 2584 | 0 of 2584 |
| census, every step incl. training | 19/60 held-out seeds, 68 neurons, min v -174.3 mV; dev 5/8 seeds, 19 neurons, min -146.6 mV, max +68 mV | 0/68 seeds, 0 neurons; min v -89.9 mV |
| engagement | flag False, env "0", read-back False on 2584 reads | flag True, env "1", read-back True on 2584 reads; `oor_count_session` max 0 |

- Flag OFF reproduces the pinned 58/60, and its per-seed records (categories, slot, readout and gate counts, all 10
  cases) are identical to the pinned records produced by the Stage-2 pin code on 60 of 60 held-out seeds.
- An excursion is common (19/60) and needed, not sufficient: 17 of those 19 still pass; both failing seeds (308, 348)
  are among them. The failure needs the spurious spikes to recruit the opposing quench pool.
- The bounded step lowers the intact latch counts (paired median -13 spikes over 58 seeds x 4 channels, 86 to 73),
  all far above the 36 bar.
- (c) Pinned-failure replay, organ-level protocol (intact + permuted): seeds 101, 202, 242, 308, 348 pass 0/5 flag off
  (`oor_count_session` 1, 5, 1, 2, 2) and 5/5 flag on (0, slots 66-87). Seed 101 is a gate seed, used here only at the
  organ level.
- (d) Fresh seeds 401-420, never used in the diagnosis, same battery: flag on 20/20 (latch min 61, median 73, max 83;
  0 co_commit; 0 excursions; min v -90.7 mV). Flag off 19/20: seed 406 fails five cases (intact, permuted, shuffle,
  teach_swap, baseline), an unselected natural failure (1 in 20, near the 4% rate) that the flag removes; 9/20 flag-off
  seeds had an excursion (min v -180.8 mV).
- (f) Injection stress: one outlier appraisal neuron (seed 101's a/b/d, C = 39.6 or 53.6 pF) placed in each of the four
  ensembles, baseline-passing seeds 303-306, intact + permuted protocol. C = 39.6: flag off 0/16, flag on 16/16. C = 53.6: flag off
  13/16, flag on 16/16. Distinct organ neurons below -110 mV in one run: flag off up to 288 (the outlier's inhibitory
  targets cascade), flag on 0.
- (h) Quench necessity, flag on, seeds 301-310, quench gate `emo_quench` (the `ql_*` quench-to-rung synapses) clamped to 0
  after development; read-back of the lesion: 5760 synapses, gain min = max = 0.0 (open arm 1.0). Quench open: all four
  situations named correctly on 10/10 seeds. Quench closed: the four-situation protocol fails on 10/10 seeds. Of the 30
  later reads (fear, anger, sadness) 24 co-latch and name None, 5 name a stale earlier channel alone (joy; seeds 305, 306,
  308, 309), and 1 names its own channel alone (seed 309, sadness). So the strict wording "co-latch or name None" holds on
  24 of 30 reads, not on all 30; what holds on every seed is that the sequence is no longer read correctly without the
  quench. Flag off, same seeds: 26 / 3 / 1 and the protocol fails 10/10. The quench is still load-bearing under the fix.
- Determinism: re-running flag-on seeds 305 and 1 reproduced the first run's records exactly.

## Handler level, seed 101, group emo (the owed step)

The Stage-2 runner `research/runners/stage2_exit_gate.py` is not on main (it lives on `research/stage2-exit-gate`, pin
9bdb92ee8). The emotion organ file is identical at the pin and at main, so these four shards ran from a scratch
worktree at the pin with this branch's organ diff applied (the file is identical to the branch's), not from the
branch worktree. `--shard --seed 101 --group emo
--arm {intact_a, permuted} --allow-gate-seed` against the committed prereg, flag set or unset, run serially under
`mem_ok.sh 10` and `memcap.sh 11`; peak RSS 8.0-8.1 GB, 516-547 s each, source `tiny-demo +LTM`, no errored turn, no
organ error. Seed 101 is a gate seed; these are diagnostic shards, not the gate.

| feel turn (intact_a) | flag OFF | flag ON |
|---|---|---|
| after joy | no category (latch 0; `oor_count_session` 1) | **joy**, latch 77 |
| after fear | fear, 84 | fear, 73 |
| after anger | anger, 80 | anger, 67 |
| after sadness | sadness, 93 | sadness, 75 |

- Flag off reproduces the Stage-2 failure at the handler: `feel_joy` names nothing. Flag on, `feel_joy` names joy with
  latch 77 (bar >= 36). Slot counts match the organ-level protocol for seed 101 exactly (77, 73, 67, 75).
- Permuted exposure (affect words swapped): flag on names sadness, anger, fear, **joy** (the sadness frame names joy, latch
  77; latches 74, 68, 73); flag off names sadness, anger, fear, **no category** on the last turn (latches 90, 76, 86, 0).
- Engagement on every feel turn, flag on: `bounded_flag` True, `bounded_env` "1", `bounded_readback` True,
  `oor_count_session` 0. Flag off: False, None, False, and `oor_count_session` 1 from the joy (intact) or fear (permuted)
  read on, the out-of-range neuron of the diagnosis.
- Non-feel turns (the 8 situation exposures): reply text byte-identical flag on vs off on both arms (8/8 and 8/8; the
  whole response JSON identical too). On the 5 feel turns the text before the category sentence is identical on both
  arms. Answer rate over all 13 turns of each arm (`tools/answer_rate.py` library; its paired CLI reads a different
  shard layout): 5/13 on and off on both arms, no FLIP-HURTS-ANSWERS.


## Caveats

- The pinned failure rate is about 4% of builds (held-out 301-360: 58/60), so a 6/6 gate pass alone proves little. The
  evidence here is the paired organ-level comparison (60 + 8 seeds) and the injection stress, not a gate re-run.
- Seeds 101, 202, 242, 308 and 348 are design seeds for this repair (the failures it was built to remove). Held-out
  301-360 is not untouched either: the diagnosis compared three candidate fixes (this guard, a capacitance floor,
  capacitance-matched synaptic scaling) on those 60 seeds before this branch existed, so the 60/60 here is a re-run of the
  chosen fix from this branch's code, not a fresh draw. The fresh draw (d), seeds 401-420, is the out-of-design check.
- Organ level plus one seed (101) through the handler. No six-seed handler run, no flip, default OFF.
- Dev seeds 1-8 do not discriminate: flag off passes 8/8 although 5 of the 8 builds have an out-of-range excursion. The
  excursion is common and harmless on most draws; only the draws where it recruits the opposing quench pool fail.
- The flag-off organ-level arms set the env to "0" explicitly; the flag-off handler shards leave it unset. Both resolve to
  the unbounded step (read-back False).
- The bounded kernel changes every neuron's conductance step in the organ, so development (training) runs bounded too
  and the learned weights differ slightly from the unbounded build.
- The 2-D sign sub-read (fear reads "neutral") is a separate host dead-zone issue and is untouched.
