---
type: finding
status: live
date: 2026-08-09
mechanism: noise-driven-unsupervised-stdp-sleep
lane: breadth-crux
---

# HONEST NEGATIVE — noise-driven UNSUPERVISED-STDP sleep (Bazhenov 2022) does not beat supervised self-replay on the teacher-loop breadth crux

STATUS: NEGATIVE (adversarially verified; 6-seed). A first-class deliverable: it maps what unsupervised STDP
can/can't do on this substrate.
Date: 2026-08-09
Runner: `research/runners/_teacher_loop_noise_stdp_sleep_derisk.py`
Aggregate artifact (all per-seed + mean numbers below trace here): `research/findings/raw/teacher_loop_noise_stdp_sleep_aggregate.json`
Per-seed raws: `research/findings/raw/teacher_loop_noise_stdp_sleep_s42.json` ... `research/findings/raw/teacher_loop_noise_stdp_sleep_s47.json`

## Question
<!--derived (prior-banked self-replay numbers + external Bazhenov citation; motivation, not this run's measurement)-->
Our best mechanism against catastrophic forgetting in the sequential teacher-loop is IN-RUN SELF-REPLAY, which is
SUPERVISED (the taught class label rides along with each self-generated engram replay). It retains ~0.85 @ N=10
but degrades to ~0.45 @ N=20. Bazhenov's SNN sleep (Golden, Delanois, Sanda, Bazhenov 2022, PLOS Comput Biol
18(11):e1010628) proposes a MECHANISTICALLY DIFFERENT consolidation: after sequential acquisition, SILENCE the
cue, drive the network with broadband POISSON NOISE, and switch the readout plasticity from supervised e-prop to
UNSUPERVISED (Hebbian) spike-timing STDP — moving the synapses to the INTERSECTION of the task manifolds that
satisfies ALL facts at once, with NO class label. Does it beat supervised self-replay in the overlapping N=20
regime?

## Method (brain-based, one spiking substrate, NO sim/ edit)
- Substrate: `OnBridgeEpropNet` (production Izhikevich spiking bridge), reused by import; the readout is a Bellec
  leaky readout whose H_last->out synapses live in `cp_connections`.
- NOISE-STDP SLEEP (the arm under test): after teaching each fact, SILENCE the cue (no percept) and drive broadband
  Poisson-noise current — injected at the input slice, propagating through the ALREADY-TRAINED input->hidden
  weights to reactivate the readout-upstream hidden units. The substrate SPIKES; those spikes propagate through the
  readout synapses and the output neurons SPIKE; PAIR-BASED spike-timing STDP (LTP pre-before-post, LTD
  post-before-pre) then updates the readout synapse weights DIRECTLY in `cp_connections.data`. NO label, NO target,
  NO gradient, NO host weight-average — noise -> spikes -> STDP on synapses.
- Arms (same net build / seed / per-fact teaching budget — the ONLY difference is the consolidation phase):
  NOSLEEP (the wall), SELFREPLAY (the supervised baseline, imported verbatim from the sleep-replay derisk),
  NOISESTDP (this).
- 6 seeds 42-47; N=10 and N=20 measured in one n_max=20 curriculum (retention of facts 0..9 at milestone 10,
  facts 0..19 at milestone 20). SIM_BACKEND=numpy.

## Results (frac_recalled; chance = 1/N)

Per-seed (representative, seed 42):

| arm         | N=10        | N=20        |
|-------------|-------------|-------------|
| nosleep     | 0.10        | 0.05        |
| selfreplay  | 0.80        | 0.45        |
| noisestdp   | 0.10        | 0.05        |

6-SEED MEANS (42-47):

| arm         | N=10 mean   | N=20 mean   |
|-------------|-------------|-------------|
| nosleep     | 0.100       | 0.058       |
| selfreplay  | 0.550       | 0.417       |
| noisestdp   | 0.083       | 0.058       |

- noisestdp − selfreplay @ N=20: **−0.359** <!--derived (difference of the two means above)--> (negative)
- noisestdp − nosleep @ N=20:    **+0.000** (identical to the wall)
- seeds where noisestdp >= selfreplay: N=10 **0/6**, N=20 **0/6**

Per-seed N=20 (nosleep / selfreplay / noisestdp): s42 .05/.45/.05, s43 .05/.30/.05, s44 .05/.40/.05,
s45 .10/.40/.10, s46 .05/.45/.05, s47 .05/.50/.05.

The noise-STDP sleep lands ON the no-sleep wall at BOTH N=10 and N=20 — it recovers nothing (at N=10 its 0.083
mean is even marginally BELOW no-sleep's 0.100: on two seeds the sleep erased the one recency fact), and is
nowhere near supervised self-replay. The self-replay baseline reproduces the banked numbers (0.55 @ N=10 mean,
0.417 @ N=20 mean; per-seed 0.30-0.50 @ N=20), so the comparison is like-for-like.

## The sleep was NOT inert — it ran fully and moved a lot of weight
This is what gives the negative teeth: the mechanism executed end-to-end and was ANYTHING but starved.
- mean output spikes per sleep cycle: 33.77 across seeds <!--derived (range across per-seed values)--> (29 to 43)
  — the output neurons spiked healthily under noise.
- total |readout weight moved| by STDP: 48277 mean across seeds <!--derived (range across per-seed values)-->
  (25223 to 96123) — the unsupervised STDP actively and massively rewrote the readout synapses in cp_connections.
- Yet retention stayed EXACTLY at the no-sleep wall (Δ vs nosleep = +0.000 @ N=20, all 6 seeds). The unsupervised
  spike-timing STDP rewrites the readout weights heavily but does NOT reconstruct the discriminative class
  structure: driven by broadband noise, the Hebbian rule reinforces whatever the readout currently favours (the
  recency-dominant class) rather than the joint intersection. There is no beneficial-then-lost effect for the
  sleep to be load-bearing for — the sleep phase is NEURAL and ACTIVE but retention-inert (and at N=10 it
  occasionally erases the last fact).

## Anti-cheats (all verified)
- (e) `git diff main -- sim/` EMPTY — all wiring is runner-side, reuse-by-import.
- cfg.seed set (via the net's `CoreSimConfig.seed`, NOT `actual_seed_used`); substrate byte-identical across two
  builds at one seed (firing-threshold hash identical).
- NEURAL: the sleep phase drives real substrate spikes (`_run_one_simulation_step`) and writes STDP deltas into
  `cp_connections.data` via the same position map / plastic mask / rate gain the e-prop path uses — reported
  `mean_out_spikes_per_cycle` (33.77 mean) and `total_abs_weight_moved` (48277 mean) are both large and non-zero.
- UNSUPERVISED: `_noise_stdp_sleep_consolidate` has NO env / y / cls / label / target parameter (AST identifier
  scan of the function body is clean). The byte-identical label-independence check reads 0.0e+00 max-diff on ALL
  6 seeds: two identically-built + identically-acquired nets run the identical sleep, one with a scrambled label
  array in scope — the resulting readout weights are byte-identical, so a label provably cannot enter the sleep
  update.
- LOAD-BEARING baseline: NOSLEEP (== sleep lesioned) forgets to ~1/N (the wall), and SELFREPLAY reproduces the
  banked 0.80/0.45 — the harness is faithful.
- immediate acquisition stays perfect in NOISESTDP (~0.99) — the sleep does not destroy the just-taught fact.

## Verdict
NO-GO for beating supervised self-replay, 6/6 seeds. On this substrate, Bazhenov-style noise-driven UNSUPERVISED
spike-timing STDP does not close catastrophic forgetting in the sequential teacher-loop: it matches the no-sleep
wall (Δ +0.000 @ N=20) and falls −0.359 <!--derived (difference of means)--> below supervised self-replay at N=20 (0/6 seeds beat it at N=10 or N=20).
The mechanism is neural, unsupervised, and active (33.77 output spikes/cycle, 48277 weight moved on average <!--derived-->), so this
is a property of the MECHANISM on this substrate, not an inert/starved setup.

## Why (the mechanistic read, for the next lever)
Bazhenov's result depends on the noise reactivating the DISTINCT stored attractors so unsupervised STDP can find
their intersection. Here the decision layer is a NON-spiking leaky-readout whose H_last->out synapses are weak and
whose hidden code for different referents overlaps; broadband Poisson noise driven through it produces output
spikes dominated by the recency-favoured class, so Hebbian STDP amplifies the dominant readout rather than the
intersection. The label-carrying signal that self-replay supplies (which target each replayed engram belongs to)
is exactly what disambiguates the overlapping facts — and it is precisely what the unsupervised sleep discards.
This localizes the residual: the missing companion process is a NEURAL error/target signal during offline
consolidation (self-replay's supervised e-prop provides it), not more/faster unsupervised plasticity.
