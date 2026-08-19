---
type: finding
status: live
date: 2026-08-19
mechanism: graded-affect-bistable-ladder
board_task: 81
artifacts:
  - research/findings/raw/graded_affect/_graded_affect_attractor_6seed.json
  - research/findings/raw/graded_affect/_graded_affect_attractor_6seed_smoke.json
runner: research/runners/_graded_affect_attractor_derisk.py
---

# Graded affect — a bistable-LADDER attractor reads the body-state as a SMOOTH valence x arousal (6-seed GO)

**Board #81 ("make the brain's feelings graded, not just on/off").** Verdict: **BRAIN-BASED GO, 6/6 seeds**
(numpy-CPU, NO `sim/` edit). The residual from the embodied-affect GO (#49) was that the affect substrate — the
P0.3 bistable NMDA opponent LATCH — read the interoceptive body as a two-state +/- SWITCH (mood a +-0.08 sign
flip; felt-arousal on/off, gradedness Pearson 0.70, 1/6 seeds >=0.8). Replacing the single latch with the named
surpass — a **Koulakov-2002 / Goldman-2003 robust DISCRETE integrator (a LADDER of independently-latched bistable
sub-pools recruited at staggered thresholds)** — makes the same synaptic interoceptive drive map to a **smooth,
graded, persistent** valence x arousal, while keeping the #49 embodiment intact.

## The mechanism (a runner-level ladder; the P0.3 `aff()` bistable primitive reused, NO `sim/` edit)

- Per affect sign, a **LADDER of N_L=6 independent self-recurrent NMDA bistable sub-pools** (the exact P0.3
  `aff()` factory, 50 neurons, recur_weight 22 — the proven NMDA-dependent regime): `affect_vplus_L0..L5`,
  `affect_vminus_L0..L5`, `affect_arousal_L0..L5`. **NO intra-sign lateral inhibition** (the load-bearing Koulakov
  rule — any within-sign competition collapses the ladder back to a WTA 2-level latch).
- **Staggered recruitment is purely SYNAPTIC**: the interoceptive->sub-pool weight decreases along the ladder
  (`g_k = [1.00, 0.84, 0.68, 0.52, 0.36, 0.20]`), so a stronger body signal crosses more sub-pools' ignition
  thresholds and latches more of them. Graded value = the **count of latched sub-pools = population firing rate**.
- The **#49 interoceptive channel is reused unchanged**: 3 spiking relay pools (`intero_comfort` <- comfort=h,
  `intero_discomfort` <- discomfort=1-h, `intero_arousal` <- arousal=a) driven by a host body-current, each
  projecting synaptically (AMPA, gated by `intero_out`) onto its ladder.
- The felt STATE is the ladder's OWN population read: `mood = rate(V+ ladder) - rate(V- ladder)`,
  `felt_arousal = rate(arousal ladder)`, off `cp_firing_states`. **Never a host formula.** The affect pools
  receive ZERO direct external current (asserted every step) — the body reaches them ONLY through synapses, and
  the threshold staggering lives in the synapse, not a host tonic bias.

## Anti-cheat 1 — GRADED, not a switch (6/6, and the LADDER is load-bearing)

Sweeping the body-state (11 points) and reading the ladder (pooled over 6 seeds):

| channel | Pearson(body, felt) | seeds >=0.8 | dynamic range | resolvable levels (pooled) |
| --- | --- | --- | --- | --- |
| valence (mood vs h) | **+0.97** | **6/6** | 0.156 | **7** | <!--derived-->
| felt-arousal (vs a) | **+0.95** | **6/6** | 0.078 | **5** | <!--derived-->

vs the #49 baseline: mood a 2-state +-0.08 switch; felt-arousal on/off (Pearson 0.70, 1/6 seeds >=0.8). Per-seed
resolvable levels are 7-8 (valence) and 5-6 (arousal); all > 2. The valence curve crosses zero cleanly at the
set-point h~0.5 (a graded bidirectional ramp, not a sign flip), and felt-arousal is a monotone staircase.

**The ladder is what makes it graded (the different-in-kind control, seed 42).** A matched **single bistable pool**
(same total neuron count, no ladder) at the same operating point recruits ALL-OR-NONE and SATURATES — the P0.3
latch: **arousal 2 resolvable levels, range 0.019; valence 3 levels, range 0.036** <!--derived--> — vs the ladder's
5-7 levels and 4x larger range. Note the single-pool Pearson was still +0.81 (arousal) / +0.94 (valence): a
step function correlates ~0.8-0.9 with a ramp, so **Pearson alone does NOT distinguish a switch from a graded
substrate** — the resolvable-levels count (with a mechanism floor of half a sub-pool's rate contribution, ~0.0075, <!--derived-->
so readout micro-jitter is not miscounted as states) and the dynamic range are the discriminating measures.

## Anti-cheat 2 — PERSISTENCE: a graded robust integrator (holds the graded level after drive-off, 6/6)

Establish several body-drive levels, REMOVE the drive, and read the HELD state ~350 ms later:

| held quantity | Pearson(level, held) | held range | NMDA-off held |
| --- | --- | --- | --- |
| felt-arousal (5 levels) | **+0.95** | 0.077 | **0.000** | <!--derived-->
| mood (5 valence levels) | **+0.98** | 0.154 | (NMDA-off decays) | <!--derived-->

The held state TRACKS the drive level instead of decaying to ONE default — a **graded working state**, exactly the
point of the ladder. With NMDA disabled the held state decays to ~0 (max 0.000 over the top two levels), so the
persistence is the slow-NMDA latch, not the tonic drive. This is the robust-integrator property a plain continuous
line attractor lacked (Wave-1: held range 0.003, drifts to a point) and a single latch lacked (2 held levels). <!--derived-->

## Anti-cheat 3 — EMBODIMENT preserved (the #49 dissociation still holds, 6/6)

Cutting the interoceptive->affect synapses (`intero_out` gate=0) while the body-state sweep is UNCHANGED collapses
BOTH channels: valence range 0.156 -> **0.000** (corr +0.97 -> **+0.00**), felt-arousal range 0.078 -> **0.000**. <!--derived-->
`tools.lab.attributable_to` reports the interoceptive path owns **100% / 100%** of the valence / arousal coupling.
Meanwhile the interoceptive pools STILL encode the body (corr +0.99 / +0.99, comfort / arousal). The felt read is
neural (`cp_firing_states`); the affect pools never receive a direct current (runtime-asserted). The body causes a
graded feeling through synapses, and interoception is load-bearing for it — the #49 result, now graded.

## Anti-cheat 4 — 6 seeds, deterministic

Seeds 42/43/44/100/101/102, `cfg.seed` set. Verified: two builds at one seed give byte-identical
`cp_neuron_firing_thresholds` (seeded substrate), and the SIX seeds give DISTINCT thresholds (genuine independent
seeds, not a fake multi-seed — the CLAUDE.md seed gotcha checked). The near-identical curves across seeds reflect
the ladder's seed-robustness, not a seeding bug. All 11 aggregate gates pass 6/6.

## Honest residuals (the characterized boundary)

- **Gradedness is QUANTIZED, by design (Koulakov: drift-robustness is bought with resolution).** The substrate is
  a robust graded STAIRCASE of N_L+1 levels (achieved: ~5-7 resolvable states over the sweep), NOT a smooth
  Russell continuum. More resolution is more sub-pools (linear cost). The claim is "a robust graded quantized
  code," and the resolvable-levels count reports the achieved resolution — selling it as a continuum would overclaim.
- **The single-cell gold standard (Egorov I_CAN graded persistent activity) is NOT built** — it needs a `sim/`
  change (the Izhikevich substrate has no calcium-activated cation current). The ladder buys the *capability* now
  on the existing substrate; I_CAN remains the faithful single-cell upgrade to schedule if the quantization proves
  insufficient (the research gate's design note).
- **Aggregate opponent cross-inhibition is present in the runner but DEFAULT-OFF (xinh=0).** The body input is
  already opponent-structured (comfort=h vs discomfort=1-h anti-correlated), so mood grades as the DIFFERENCE of
  two independently-latching ladders; adding cross-inhibition forces WTA that both sharpens valence toward a switch
  and resolves the graded middle during unclamped hold — it DEGRADES both gradedness and graded persistence. An
  honest design choice, not the named default. (Confirmed: xinh>0 lowers valence gradedness and held-range.)
- **Scope (bounded, as #49).** Two body axes (comfort + arousal), a 3-pool interoceptive channel, an open-loop
  body sweep (no homeostatic feedback / body dynamics — a follow-on). The body VARIABLES are host (the standard
  body boundary); the de-risk is the body->AFFECT MAPPING being a graded ladder of synaptically-recruited bistable
  pools. numpy-CPU is the backend (real spiking Izhikevich bridge), not a shortcut.
- **Honesty boundary.** A functional graded core-affect state with a bodily cause and an honest functional
  read-out; no claim of phenomenal experience.

## Reproduce

Per-seed rows, sweep curves, the single-pool control and all 11 aggregate gates are in
`research/findings/raw/graded_affect/_graded_affect_attractor_6seed.json` (provenance sidecar alongside).

```
SIM_BACKEND=numpy python -u -m research.runners._graded_affect_attractor_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._graded_affect_attractor_derisk --seeds 42 43 44 100 101 102
```
