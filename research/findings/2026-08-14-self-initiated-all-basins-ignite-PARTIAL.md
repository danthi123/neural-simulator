---
type: finding
status: partial
date: 2026-08-14
mechanism: dmn-all-basins-ignite-adaptation-driven-itinerancy-STP
lane: F-self-initiation-DMN
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_self_initiated_all_basins_ignite_derisk.py
artifacts:
  - research/findings/raw/_self_initiated_all_basins_ignite_derisk.json
builds_on:
  - research/findings/2026-08-13-self-initiation-multibasin-GO.md
  - research/findings/2026-08-13-self-initiated-utterance-GO.md
  - research/findings/2026-08-13-self-initiated-spontaneous-thought-GO.md
  - research/findings/2026-07-22-gap5-RANK1-spontaneous-reactivation-6seed-GO.md
  - research/findings/2026-07-23-gap5-replay-candidate1-intrinsic-fatigue-alone-NEGATIVE-pivot-to-STD.md
---

# All-basins-ignite via adaptation-driven itinerancy (ca3->ca3 short-term depression) does NOT close the multibasin/utterance "3 of 4 basins ignite" residual -- and the SOLO diagnosis CORRECTS the cause: the tail basin fails even in ISOLATION (absolute-threshold, NOT competition lock-in), so STP (a competition-reshaper) cannot rescue it; STP additionally SELF-IGNITES the bistable down-state without noise. STP is BANKED; the named next rung is a CONNECTIVITY-aware pattern separation / scale lever

<!--derived-->

**One-line verdict:** the multibasin (2026-08-13) and utterance (2026-08-13) de-risks left one residual -- only ~3 of
4 DISJOINT CA3 basins ever ignite under the noise-driven wander, so the wander (and the mouth) covers 3 concepts,
never the full store. This rung's SURPASS was short-term synaptic DEPRESSION on the ca3->ca3 recurrent (Tsodyks-Markram,
U=0.5/tau_d=300/tau_f=50) to fatigue the current winner -> release the shared feedback inhibition -> a RESTED basin
wins the next noise race -> the wander samples ALL N (winnerless competition; Christoff et al. 2016 DMN dynamic
framework). It does NOT work, and the reason is decisive and re-directs the effort: the SOLO isolation diagnosis (each
basin run with every competitor's recurrence zeroed) shows the tail basin fails to ignite EVEN ALONE (seed 42: basins
0/1/2 ignite solo at member 0.35-0.38, dwell 286/186/103; basin 3 fails, dwell 1, member 0.12 vs random 0.04). Across
all 6 seeds this is UNANIMOUS: the GO substrate ignites exactly 3/4 basins (ignition-reliability for 4/4 = 0/6) and the
SAME basin -- the LAST-encoded index 3 -- fails solo on 6/6 seeds (solo member 0.12-0.15), across 6 DIFFERENT random
partitions, so the weakness is SYSTEMATIC to being encoded last, not per-seed random variance. It is
ABSOLUTE-THRESHOLD, not competition lock-in -- there is nothing for STP to out-compete; the basin simply cannot
complete its own attractor. Two further levers were falsified on the same tail: a tonic intrinsic-excitability boost
(60-120 pA on the basin's cells -> member stuck ~0.12) and a within-recurrent WEIGHT boost (1.5x/2x/3x -> member
0.12-0.15). STP on this bistable dendritic-plateau substrate ALSO destabilises: with STP on the recurrent and NO noise
the down-state SELF-IGNITES (10^5-10^6 spikes, apical latched), so the internally-triggered / noise-seeded property
breaks. NOT a wall: STP is BANKED (it is the correct fair-sampling fix for competition lock-in, which is NOT the
failure mode here); the named next rung is a CONNECTIVITY-aware pattern separation / scale lever (larger n_ca3, or a
partition that GUARANTEES each disjoint basin sufficient internal recurrent density) so every basin can complete. Per
THE LAW: method banked, next named; closure carried forward, not deferred. FUNCTIONAL CORRELATE only, no phenomenal
claim.

## The mechanism attempted (the spec's SURPASS) and why STD is NOT the banked chain-ordering negative

The wander is a hard winner-take-all on a SHARED feedback-inhibition pool (ca3_pv_basket, ca3_fb_inhib=20) with FIXED
recurrent efficacy and NO fatigue. The hypothesis (correct wall-reframe: "what companion process did we replace with a
constant?"): biology runs short-term synaptic depression alongside recurrent attractor dynamics, fatiguing the active
assembly so it YIELDS. Turn it off -> hard WTA -> lock-in -> the same ~3 basins win every race. The fix was STP on the
ca3->ca3 recurrent (riii `_build` enable_stp; Romani & Tsodyks 2015; Ecker 2022).

**This is NOT the banked STD negative.** `2026-07-23-gap5-replay-candidate1-intrinsic-fatigue-alone-NEGATIVE-pivot-to-STD.md`
banked E->E STD NEGATIVE for a DIFFERENT objective -- directing a forward A->B->C ORDER on a LEARNED inter-assembly
CHAIN, where STD depressed the stored chain links and destroyed the directional store (fwd 0.333->0.000 <!--derived--> quoted from that finding). Here the
basins are DISJOINT (max pairwise overlap 0) with NO learned inter-basin chain -- there is nothing directional to
destroy. STD depressing the WITHIN-assembly recurrence of the current winner is exactly the fatigue wanted. So STD is
banked-negative for chain-ORDERING and was untested + mechanistically favourable for basin-FAIR-SAMPLING. This rung
TESTED it for fair sampling. It fails for a NEW reason (absolute-threshold tail + substrate destabilisation), not the
chain-destruction the prior finding measured.

## The decisive result -- the SOLO isolation diagnosis (STP-independent, clean)

The runner's SOLO diagnosis encodes the full n_mem=4 DISJOINT store (identical seed/partition) then, for each basin m,
zeroes every OTHER basin's within-recurrence (they cannot complete or draw the shared inhibition) so basin m runs
UNCONTESTED on its own encoded weights. The competition premise predicts every basin ignites solo (the tail loses only
UNDER competition). It does not hold: at seed 42, basins 0/1/2 ignite solo (dwell 286/186/103, member 0.35-0.38 vs
random ~0.04) but basin 3 fails (dwell 1, member 0.12). The solo dwell gradient 286/186/103/1 also shows a WIDE
intrinsic-ignitability spread across equal-size, identically-encoded disjoint basins -- the random partition alone
produces the inequality. Because the tail fails with ZERO competitors, competition is not the bottleneck, and STP
(which only reshapes competition) cannot be the fix. This REFUTES the spec's expected dominant cause (A, competition
lock-in) and establishes cause (B, absolute-threshold) for this substrate/operating point. 6-seed artifact
`research/findings/raw/_self_initiated_all_basins_ignite_derisk.json` (rest 3000,
solo-steps 3000): baseline n_ig 3/4 every seed; solo fail-basin == [3] every seed; the failing basin is always the
LAST-encoded (the BTSP encode loop runs basins 0..3 sequentially), matching the multibasin GO finding's report that
"the 4th disjoint basin is consistently weakly ignitable" -- now shown to be a SOLO (absolute-threshold) failure of the
last-encoded basin, not a competition artifact.

## Multi-lever falsification of the tail (AUXILIARY seed-42 probes, solo isolation)

<!--derived-->
These are seed-42 auxiliary feasibility probes (NOT in the committed 6-seed diagnostic artifact; reproducible via the
runner's `_solo_ignition` + `_scale_within_assembly` + `_steered_rest` bias). They test whether ANY simple gain lever
rescues the tail's SOLO completion before naming the next rung.

- **Intrinsic-excitability boost (the spec's §4 lever, intrinsic variant).** A tonic depolarising bias of 60 and 120 pA
  on basin-3's cells leaves member ~0.12-0.13 (unrescued); NO-NOISE stays silent at those biases. A tonic bias makes
  cells fire more INDEPENDENTLY but does not create the COINCIDENT recurrent drive pattern completion needs (summed
  weighted drive over coincident recurrent inputs must cross the plateau k_thresh=40), so it cannot rescue completion.
- **Within-recurrent WEIGHT boost (the §4 lever, synaptic-scaling variant / uniform compensation).** Scaling basin-3's
  within-assembly recurrence by 1.5x / 2x / 3x leaves member 0.123 / 0.148 / 0.128 -- tripling the weights barely
  moves it. The failure is NOT weight magnitude; it is that this 240-cell random subset lacks the internal recurrent
  CONNECTIVITY (which cells connect to which) + favourable heterogeneous dendritic-plateau thresholds to sustain a
  completing attractor.
- **Uniform tonic-utilisation compensation of STP (comp = base x k).** comp>1 monotonically HURTS (rest 2500: comp 1.0
  -> 3/4, comp 2.0 -> 1/4, comp 3.0 -> 0/4): a stronger base over-drives firing -> faster STP depletion -> the
  attractors collapse. There is no compensation that restores the operating point AND helps the tail.

## STP is INCOMPATIBLE with this bistable dendritic-plateau substrate (a substrate-integration hazard, documented)

Two integration failures, both measured deterministically (seed 42):

1. **Build-time STP corrupts the one-shot encode.** Building the bridge with enable_stp=True (STP allocated at build)
   and running the BTSP plateau-gated encode gives within-assembly weight w~17 vs the GO store's w~103 -- a 6x weaker,
   DIFFERENT store -- EVEN with the runtime application flag held OFF during the encode. STP is a fast recall-time
   state, not part of the plateau-driven consolidation, so the runner encodes the EXACT GO substrate (store
   byte-identical every condition, ~1% GPU build-to-build non-determinism) and allocates the STP fast state POST-ENCODE.
2. **STP self-ignites the bistable down-state without noise.** With STP on the recurrent and NO Poisson noise the
   network self-ignites (~5x10^5 CA3 spikes over 800 steps on 6/6 seeds -- 495916/506074/511077/513179/513360/519136
   -- apical latched far above plateau_v_hold=-35), whether STP is applied to all synapses OR restricted to only the
   ca3->ca3 excitatory recurrent (cp_stp_disabled_mask carving inhibition + mossy + inter-region). The carefully-tuned
   KIR bistable down-state is not robust to STP on this substrate -- the internally-triggered / noise-seeded anti-cheat
   fails. (This also means STP-on "member ~0.27" wander numbers are DIFFUSE runaway, not clean coherent itinerancy, and
   must not be read as partial ignition.)

Both are engineering hazards banked with the mechanism; they are MOOT for the scientific conclusion because the SOLO
diagnosis already refutes STP's premise -- a perfectly-stable STP would still not reach 4/4, since the tail cannot
complete even uncontested.

## What is banked, what is named next

- **BANKED:** ca3->ca3 short-term DEPRESSION as the all-basins-ignite fix (it is the correct fair-sampling fix for
  competition lock-in, which is NOT the failure mode here); the intrinsic-excitability homeostatic lever (does not aid
  completion); the synaptic-weight-scaling lever (weight magnitude is not the bottleneck); the uniform
  tonic-utilisation compensation (monotonically hurts).
- **NAMED NEXT RUNG:** make EVERY disjoint basin ignite SOLO before any competition question is asked -- an
  ignition-reliability precondition the current sequential BTSP encode does not meet (the last-encoded basin fails on
  6/6 seeds). Two convergent levers, given the failure is the LAST-ENCODED basin (systematic, not random) yet is robust
  to 3x weight-scaling: (i) per-basin ENCODE / ignitability EQUALIZATION -- interleave the basins' BTSP drive instead of
  encoding them sequentially, or add a homeostatic encode that drives each basin to a COMMON solo-ignition criterion
  (single scalar target; per-basin equalization emerges; the spec §4 homeostatic framing, but applied to the ENCODE not
  a tonic bias -- since a tonic bias does not aid completion); (ii) a CONNECTIVITY-aware pattern separation / scale lever
  -- draw each basin so its cells have guaranteed sufficient internal recurrent density, raise n_ca3 for shared-basket
  headroom, and/or a stronger synchronous within-ensemble encode (Kopsick W_HIGH / dg_ffi_weight; spec §4 last-resort
  scale levers).

## Honesty boundary + anti-cheats that held

FUNCTIONAL CORRELATE only, no phenomenal claim. The DISJOINT store is genuinely pattern-separated (max pairwise
overlap 0 every seed). The stored weights are byte-frozen during every wander (conn.data array_equal True; STP lives in
cp_stp_u/cp_stp_x). The GO substrate baseline reproduces the multibasin residual (3/4 ignite). The SOLO diagnosis is
reported, not gated (spec anti-cheat #10) so the reader sees competition-lockin vs absolute-threshold directly. The
STP-off store byte-matches the multibasin/utterance GO substrate (post-encode STP allocation leaves the encode
untouched).
