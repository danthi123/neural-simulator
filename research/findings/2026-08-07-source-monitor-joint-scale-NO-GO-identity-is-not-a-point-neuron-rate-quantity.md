---
type: finding
status: negative
date: 2026-08-07
mechanism: source-monitor-attractor-competition-x-storage-separation-x-scale
runner: research/runners/_laneC_source_monitor_joint_scale.py
builds-on: research/findings/2026-08-06-source-monitor-coresidency-v6-development-NO-GO-redistributive-win-does-not-generalize.md
spec: research/findings/raw/_crossgap_identity_discrimination_synthesis.md
artifacts:
  - research/findings/raw/laneC_source_monitor_joint_scale/smoke_650_651.json
  - research/findings/raw/laneC_source_monitor_joint_scale/smoke_650_651.json.prov.json
---

# The joint attractor-competition x storage-separation x SCALE de-risk is a NO-GO at n=96: same-core co-resident source identity is not a point-neuron rate quantity

**Verdict: JOINT_SCALE_NOGO_DECISIVE_BOUNDARY.** The one decisive cross-gap de-risk named by the identity-discrimination synthesis (`research/findings/raw/_crossgap_identity_discrimination_synthesis.md`) -- attractor competition (`g_comp`, fixed recurrent-E : lateral-I ratio) x storage-side separation (`uniq_emphasis`) x source-memory SCALE (`n_source` swept 12 -> 48 -> 96 with episode + assembly sizes raised proportionally at FIXED overlap fraction) -- does NOT clear its frozen GO gate at ANY point in the grid, including CA3 scale (n=96). This is the DECISIVE boundary verdict the synthesis pre-registered: on a point-neuron **rate** substrate, the identity of same-core co-resident sources is not recoverable by within-substrate attractor competition, even with redundant sparse assemblies. It is a verdict on the point-neuron-rate METHOD, not on the source-monitoring CAPABILITY. The honest next substrate is dendritic compartments (per the synthesis; NOT built here).

## What was tested (the un-run decisive config)

`_laneC_source_monitor_joint_scale.py` reuses -- BY REFERENCE, no `sim/` edit, prior runners intact -- the joint mechanism from `_laneC_source_monitor_attractor_joint.evaluate_joint` (per-presynaptic-cell selectivity gain `b(p)**(-uniq_emphasis)` on the learned `episode->source` synapses x within-population slow-NMDA recurrent excitation + GABA-A lateral inhibition, one knob `g_comp`) and adds the ONE variable never properly tested: SCALE.

Scale is `F = n_source / 12`. Every population (episode, source-afferent, source-memory, aPFC, ACC, interneuron) is multiplied by F; every fan-in structural weight (recurrent-E, lateral-I, source->interneuron, afferent->source, source->aPFC/ACC, Hebbian cap) is divided by F to **preserve the per-neuron operating point** (density stays 1.0, so without the 1/F the total synaptic current would scale with F and every neuron would saturate -- a raw-current confound, not a redundancy test). This is the honest scale test: same per-neuron drive, F x more neurons sharing the code, which is the CA3-GO's stated route to reliable winner selection.

Grid: `n_source in {12, 48, 96}` x `uniq_emphasis in {0, 1, 2}` x `g_comp in {0, 1}` x seeds `{650, 651}` (36 cells). numpy, deterministic, ~15 min total.

## Result: no cell clears; scale does not rescue the competition

`both_seed_go_cells = []`, `any_GO_rows = 0`, `any_scale_helped = False`. No cell satisfies the frozen three-part GO (`min_margin_M >= 0.15` AND `min_margin_M > min_margin_L` AND `all_dominant_correct` on every source incl. the weakest `self_generated`) on either seed.

**The competition mechanism (`g_comp = 1`) -- the only path that can satisfy `min M > min L` -- fails the decisive anti-cheat `all_dominant_correct = False` in ALL 18 cells at EVERY scale including n=96**, with a NEGATIVE `min_margin_M` in every one. The lateral-inhibition WTA reads INTRINSIC assembly strength (dominated by the shared core + the mixed-episode boost of `core->seen/heard`), not the cue-specific uniq advantage, so it latches whichever rival has the largest core drive and quenches the correctly-cued source. Scale added redundancy but did NOT change this: `min_margin_M` at (n12/n48/n96) for the representative ue=0 cell is (-0.105/-0.013/-0.048) seed 650 and (-0.168/-0.049/-0.089) seed 651 -- still negative at n=96.

**Scale DID modestly help the pure feedforward case (`g_comp = 0`), but on a path that cannot pass the gate.** At ue=0, feedforward `all_dominant_correct` is seed-fragile at n=12 (True/False across 650/651) and stabilizes to True/True at both n=48 and n=96 -- the redundancy averaging the near-threshold winner-flip, exactly the CA3-GO mechanism. But (i) the feedforward margins stay ~0.01-0.06, an order of magnitude below the 0.15 floor, at every scale; and (ii) `g_comp = 0` has M == L by construction (competition off in both arms, byte-identical null verified at all three scales), so it can NEVER satisfy `min M > min L`. The uniq-cell signal that pure feedforward exploits is real but weak, and scale does not lift it to the floor.

### The load-bearing question -- did n>=48 change anything vs n=12?

For the **competition** mechanism (the GO path): **No.** `all_dominant_correct = False` and `min_margin_M < 0` at n=12, n=48 AND n=96 -- scale changed nothing decisive. For **feedforward winner-correctness**: a modest yes (dom_ok stabilized True/True at n=96 for all `uniq_emphasis`), but far below the floor and on a path that cannot pass `min M > min L`. The CA3-scale escape hypothesis (redundant sparse assemblies -> reliable competitive winner selection) is falsified for this substrate: redundancy stabilizes the feedforward rank but does not make the attractor competition track the cue rather than the intrinsic core-strength.

## Honesty guards (all held) + co-residency anti-cheat (proven)

- Overlap fraction held EXACTLY constant across scale: realized `core_size / episode_pattern_size = 0.1667` at n=12 (2/12), n=48 (8/48) and n=96 (16/96). The scale did NOT reduce co-residency to look better (the goalpost-moving cheat). `overlap_intact_core_fires_every_cue = True` in all 36 cells.
- Recall stays EPISODE-ONLY: source-afferent current == 0 AND firing == 0 at recall in all 36 cells.
- Competition module is parameter-symmetric across sources (no source term).
- `g_comp = 0` + `uniq_emphasis = 0` is byte-identical to the attractor NO-GO at all three scales.
- Zero-learned-weight instrument control is strict=False everywhere (no stepping-history artifact).
- No source's own-recall rate collapses.
- Non-vacuity: a forced source afferent moves the winner in every `g_comp = 0` cell. Honest caveat: in several `g_comp > 0` cells the forced afferent does NOT move the winner -- the competition WTA latches so hard on the core-strongest assembly that even a forced afferent input cannot override it. This is consistent with (and a further symptom of) the failure mode, not a honesty breach: the actual recall measurement is still afferent-silent.

## Why this is the boundary, and the next substrate (NOT built here)

The synthesis' proof-by-construction: on a point-neuron rate substrate the only distinguishing signal (the unique-cell subset) sums LINEARLY with the shared core into ONE soma rate, and every downstream op on that rate is an aggregate that has already discarded the categorical label. Attractor competition operates on that same aggregate and therefore amplifies the largest-total-drive (core-dominated) assembly, not the cued one -- which is exactly what all 18 `g_comp > 0` cells show, unchanged by 8x scale. Identity is a nominal label, not a magnitude, and no rate-level mechanism (sparsity, participation ratio, gini, top-share, E/I, rate set-point, symmetric GABA, attractor WTA, storage-side selectivity gain, scale) can defend a between-source contrast that a firing-rate aggregate cannot represent.

Per the synthesis, the honest next-substrate direction is **dendritic compartments**: unique-source afferents cluster on one branch, shared core on another, and only the branch with coincident core+unique input crosses its NMDA-plateau nonlinearity -- an identity-specific nonlinear AND whose discriminating variable is WHICH branch plateaued (re-created fresh by input geometry each recall), not a stored soma rate-level. That build is deliberately NOT opened here; this de-risk was the pre-registered gate on whether the cheaper rate-substrate route could close, and it is now closed as a NO-GO. HONESTY CONDITION for the next substrate: the branch assignment of unique-source afferents must SELF-ORGANIZE (BCM branch-sculpting), never be host-wired per source.
