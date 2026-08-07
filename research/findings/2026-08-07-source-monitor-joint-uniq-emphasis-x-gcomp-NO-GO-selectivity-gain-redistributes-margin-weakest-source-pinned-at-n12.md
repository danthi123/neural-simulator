# Source-monitor JOINT knob (uniq_emphasis × g_comp): NO-GO (smoke) — a label-free selectivity gain REDISTRIBUTES margin across co-resident sources but the weakest source is pinned; n=12 at overlap 0.2 is near a substrate boundary

Status: NO-GO (smoke). Backend: numpy (deterministic).
Runner: `research/runners/_laneC_source_monitor_attractor_joint.py`.
Artifact: `research/findings/raw/laneC_source_monitor_attractor_joint/smoke_650_651_overlap0.2.json`.
Prior rung (recall-only): `2026-08-07-source-monitor-attractor-competition-NO-GO-single-gcomp-knob-wta-does-not-track-cue-joint-storage-separation-knob-needed.md`.
Scoping: `research/findings/raw/_source_monitor_attractor_competition_scoping.md`.

## Step-1 diagnosis: the parent's weight-magnitude read is INVERTED; the honest signal already works feedforward
Instrumented seed 650, g_comp=1, `seen` cued (decoding the learned `episode→source` synapses; CSR is pre→post,
bridge.py:3827). Findings, which CORRECT the parent's hypothesis ("shared core carries inflated weight the uniq cells
cannot overcome"):

1. In SUM the uniq→source learned weight DOMINATES core→source (|w| core/uniq ratio 0.07–0.18): uniq is the LARGER
   signal, not the weaker. (Per-synapse the core is more potentiated — mixed-episode boost — but there are only 2 core
   cells vs 10 uniq cells per pure pattern, so the uniq sum wins.)
2. In PURE FEEDFORWARD (g_comp=0) the honest cue signal WORKS: under the `seen` cue the `seen` pop receives the most
   TOTAL input (6190, of which 4902 is uniq-driven); the rivals heard/self get ONLY core-driven input (2477 / 1588,
   uniq-driven 0). So `seen` wins every cue and `all_dominant_correct=True` (seed 650).
3. Competition INVERTS it: at g_comp=1 the `seen` cue makes `heard` win (rate 0.207 vs seen 0.102) and the `heard` cue
   makes `self` win. The lateral-inhibition WTA latches whichever rival has the strongest CORE-driven / mixed-boosted
   recurrent drive and quenches the correctly-cued source. The failure is that the competition reads INTRINSIC assembly
   strength (dominated by the shared core + the mixed boost of core→seen/heard), NOT the cue-specific uniq advantage.

This confirms `uniq_emphasis` is the right lever and clarifies WHY: down-weight the shared core's synaptic efficacy so
the honest uniq-driven signal, not the core-driven rival drive, dominates.

## The honest joint knob
`uniq_emphasis` applies a per-PRESYNAPTIC-cell selectivity gain `b(p)**(-uniq_emphasis)` to the learned
`episode→source` synapses, where `b(p)` = the presynaptic cell's cumulative cross-source fan-out breadth (# distinct
source pops it acquired a learned weight to: shared core → b=3, source-unique → b=1). Broadly-projecting (core) inputs
are down-weighted RELATIVE to selective (uniq) inputs. It uses NO source label (`b(p)` is a property of the cell's own
fan-out, symmetric, available without knowing the cued source), re-weights synapses at FIXED overlap (never touches
`make_overlapping_episode_patterns`), and `uniq_emphasis=0` is a no-op. Swept against the recall-side `g_comp`
(slow-NMDA recurrent excitation + GABA-A lateral inhibition) = the JOINT knob. NO `sim/` edit.

## Smoke result (calib 650/651, uniq_emphasis {0,0.5,1,2} × g_comp {0,1}, overlap 0.2, core=2/12)
Decisive gate: `min_s M_s ≥ 0.15` AND `all_dominant_correct` True on EVERY source incl. the weakest.

| seed | ue | g_comp | min M | dom_ok | per-source margins {seen, heard, self} |
|------|----|--------|-------|--------|----------------------------------------|
| 650 | 0.0 | 0.0 | +0.0367 | True | 0.037 / 0.137 / 0.046 |
| 650 | 0.5 | 0.0 | +0.0508 | True | 0.084 / 0.167 / 0.051 |
| 650 | 1.0 | 0.0 | +0.0283 | True | 0.112 / 0.081 / 0.028 |
| 650 | 2.0 | 0.0 | +0.0117 | True | 0.143 / 0.080 / 0.012 |
| 651 | 0.0 | 0.0 | −0.0567 | False | 0.051 / 0.146 / −0.057 |
| 651 | 2.0 | 0.0 | −0.0767 | False | 0.125 / 0.126 / −0.077 |
| 650 | * | 1.0 | ≤ −0.056 | False (all) | competition inverts at every ue |
| 651 | * | 1.0 | ≤ −0.075 | False (all) | competition inverts at every ue |

The knob is ACTIVE and CONFIRMS the diagnosis direction: `uniq_emphasis` lifts the cued source's OWN margin
monotonically (seed 650 `seen`: 0.037 → 0.084 → 0.112 → 0.143, nearly the floor). BUT it is a SEE-SAW REDISTRIBUTION:
lifting the weakest source comes out of the shared budget of the others (seed 650 `heard`: 0.137 → 0.080), so the MIN
margin never clears 0.15 — best `min M` among all-dominant-correct rows = **0.0508** (3× below floor). `self_generated`
(never mixed-boosted, lowest uniq weight) is the binding constraint: for seed 651 it stays NEGATIVE at EVERY (ue,
g_comp) and `uniq_emphasis` makes it no better (removing core also removes self's own core-driven support). And
`g_comp>0` still inverts the ranking at every `uniq_emphasis` (`all_dominant_correct=False` in all 8 competition
cells) — the joint does not rescue the recall-side WTA.

## Anti-cheats (all reported, all hold)
- OVERLAP UNCHANGED (the key honesty proof): the pattern/core hash is constant per seed across every arm
  (`overlap_unchanged_across_arms=True`), and the shared core still fires in every cue
  (`overlap_intact_core_fires_every_cue=True`, `core_active_in_cue == core.size` for all cues). The knob re-weighted
  synaptic EFFICACY at fixed overlap — it did NOT reduce co-residency (no goalpost move).
- (a) `uniq_emphasis=0 ∧ g_comp=0` is byte-identical to the attractor NO-GO (`byte_identical_null_at_0_0=True`;
  min M seed 650 +0.0367 / seed 651 −0.0567 reproduce it exactly).
- (b) HONESTY: source-afferent current=0 AND firing=0 at recall on every source; the gain carries no source term.
  Non-vacuity: at the feedforward arm a forced afferent moves the winner (under strong competition it can override the
  forced afferent — expected, not a broken guard).
- (c) no source's own-recall rate collapses. (d) zero-learned-weight control stays strict=False (no instrument artifact).

## Verdict + honest residual (capability NOT abandoned)
NO-GO. The method — a label-free storage-side selectivity gain, alone or jointly with recall-side attractor
competition — is banked. The residual is not a tuning gap but a CONSERVATION constraint: at n=12 neurons/source pop with
2 shared-core cells and a mixed episode that structurally disadvantages `self_generated`, the honest per-source margin
the learned weights afford (feedforward ceiling ~0.04–0.15) cannot be pushed to a UNIFORM 0.15 across all three
co-resident sources — lifting one source's margin necessarily draws from the shared firing budget of the others, and the
mixed-boost pins the weakest source (negative for seed 651). This strongly suggests **n=12 at overlap 0.2 is near a
genuine substrate boundary**: the discriminating capacity is set by the population size and the mixed-episode asymmetry,
not by the recall dynamics. The honest next locus is the SUBSTRATE SIZE / the mixed-episode weakness (a larger
`n_source_memory` giving each co-resident source its own headroom, and/or removing the structural penalty on
`self_generated`), NOT another recall-side or single-storage-side re-weighting — those have now been shown (6 linear
levers + recall-only attractor + this joint knob) to redistribute a fixed budget rather than enlarge it.
