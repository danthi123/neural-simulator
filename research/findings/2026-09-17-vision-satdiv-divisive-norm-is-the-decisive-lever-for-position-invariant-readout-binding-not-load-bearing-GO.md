---
type: finding
status: verified
date: 2026-09-17
mechanism: isolate what carries the fully-spiking position-invariant object readout under the full anti-cheat protocol
  (held-out position + scramble-null). Two clean single-variable comparisons on the config-B spiking front end (LIF
  S1->C1, fixed random S2 bank) -> C2 spike code -> signed linear discriminant spike-ported as LIF class somata with
  feedforward inhibition + temporal integration over LIF glimpses -> spiking WTA: (1) saturating divisive normalization
  (satdiv) ON vs OFF at matched n_glimpses/ridge; (2) conjunctive binding topologies vs a width-matched flat pool at
  matched n_glimpses/ridge
integration_faculty: perception (vision object readout)
lane: perception (vision configural / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO for the position-invariant spiking object readout, with divisive normalization the DECISIVE lever. Two
  results. (A) POSITIVE + isolated — saturating divisive normalization at the C2 stage is what carries the GO: with
  n_glimpses and ridge held fixed under the full anti-cheats, satdiv ON is GO (majority of seeds clear the config-C
  floor AND beat random, learning load-bearing every seed, scramble-null clean every seed), and turning satdiv OFF (to
  z-norm or no-norm) collapses it to a clear NO-GO / PARTIAL and drops held-out learned accuracy toward the flat-pool
  level. So the confound (the GO arm also ran more glimpses + higher ridge than the controls) is RESOLVED: at matched
  glimpses/ridge, satdiv is decisively load-bearing. (B) NEGATIVE + isolated — conjunctive BINDING is NOT the lever: at
  matched glimpses/ridge a width-matched flat pool matches or beats every conjunctive-binding topology (product, bind-
  arm, attention-gated, attention-gated-soft, triple, topographic-C2); only a competitive-binding arm edges the flat
  control by a single seed, still sub-GO. This REFINES the 2026-09-03 "configural-binding PARTIAL 3/6 — anti-cheats
  confirm genuine signal": the signal is genuine but NOT binding-specific (a width-matched flat pool reproduces it).
  Advances the 2026-09-03 satdiv-BORDERLINE to a GO with the lever isolated. Wires nothing; a research de-risk.
runner: research/runners/_vision_lindiscrim_readout_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/norm_z_nglim6_r1p0_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/norm_none_nglim6_r1p0_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/conjbind_competitive_n1152_heldoutpos_scramblenull_c2basistopo_6seed.json
external: NO-EXTERNAL-NEEDED — a single-variable isolation within an already-biology-bound mechanism (divisive
  normalization = Carandini & Heeger 2012 canonical computation; signed FF-inhibition discriminant + LIF WTA =
  Fremaux & Gerstner 2016 / Pouget 2000 / Brunel 2004, cited in the runner). The honest negative on binding is the
  deliverable.
builds_on:
  - research/findings/2026-09-03-vision-configural-binding-PARTIAL-3of6-below-5of6-gate-anticheats-confirm-genuine-signal.md
  - research/findings/2026-09-03-vision-satdiv-divisive-norm-readout-BORDERLINE.md
---

# Vision position-invariant readout — divisive normalization is the decisive lever; conjunctive binding is not (GO 6/6)

The perception lane had two open, adjacent 2026-09-03 verdicts: configural-binding PARTIAL (3/6, "anti-cheats confirm
genuine signal") and satdiv-divisive-norm BORDERLINE. This overnight batch runs the two single-variable isolations that
adjudicate BOTH, all under the full anti-cheat protocol (held-out position + scramble-null, config-B spiking front end).

## (A) Divisive normalization IS the lever — the confound resolved
<!--derived-->
From research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_6seed.json,
research/findings/raw/lanes/perception/norm_z_nglim6_r1p0_heldoutpos_scramblenull_6seed.json, and
research/findings/raw/lanes/perception/norm_none_nglim6_r1p0_heldoutpos_scramblenull_6seed.json —
all three arms share n_glimpses=6, ridge=1.0, and the full anti-cheats; only the C2 normalization differs:
- s2-norm satdiv (ON): capability-GO 5/6, beats-config-C-floor 5/6, learning-load-bearing 6/6, scramble-null 6/6 ->
  LINDISCRIM-READOUT-GO. Held-out learned spiking-WTA accuracy ~0.477 (vs flat-pool ~0.278, random ~0.233).
- s2-norm z (satdiv OFF): capability-GO 0/6, beats-floor 1/6, load-bearing 4/6 -> PARTIAL. Held-out learned ~0.373.
- s2-norm none (satdiv OFF): capability-GO 1/6, beats-floor 1/6, load-bearing 3/6 -> PARTIAL. Held-out learned ~0.347.

Turning satdiv off, with glimpses and ridge held fixed, collapses the GO (5/6 -> 0-1/6) and drops held-out learned
accuracy toward the flat-pool floor. The earlier GO arm confounded satdiv with more glimpses (6 vs 2) and a higher ridge
(1.0 vs 0.5); this isolation removes that confound and shows the normalization itself is decisive.

## (B) Conjunctive binding is NOT the lever
<!--derived-->
From research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_heldoutpos_scramblenull_6seed.json and
research/findings/raw/lanes/perception/conjbind_competitive_n1152_heldoutpos_scramblenull_c2basistopo_6seed.json (with
the product/bind-arm/attention-gated/triple sibling arms in the same directory) — all arms share n_glimpses=2,
ridge=0.5, and the full anti-cheats; only the C2 code differs (flat width-matched pool vs conjunctive-binding
topologies):
- width-matched flat pool (conj-bind none, the null): capability-GO 3/6, load-bearing 6/6, scramble-null 6/6.
- conjunctive-binding topologies: product 0/6, bind-arm 0/6, attention-gated 2/6, attention-gated-soft NO-GO, triple
  0/6, topographic-C2 competitive 0/6 — all at or below the flat control; only a competitive-binding arm reaches 4/6
  (one seed above the flat control), still under the 5/6 GO bar.

A width-matched flat pool matches or beats every binding topology, so the position-invariant signal the 2026-09-03
finding attributed to configural binding is genuine but NOT binding-specific.

## Robustness (independent seeds 200-205)
<!--derived-->
From research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_seeds200-205.json —
the satdiv GO arm re-run on six fresh held-out seeds (200-205) is capability-GO 6/6, learning-load-bearing 6/6,
scramble-null 6/6 (LINDISCRIM-READOUT-GO). So the position-invariant readout GO holds on 11 of 12 seeds across the two
independent seed sets — the divisive-normalization lever is robust, not a seed-42-102 artifact.

## What it means

The fully-spiking position-invariant object readout is real and GO under held-out-position + scramble-null anti-cheats,
and it is carried by **saturating divisive normalization** at the C2 stage (a canonical Carandini-Heeger computation),
composed with temporal integration over LIF glimpses and a signed feedforward-inhibition discriminant read by a spiking
WTA — NOT by conjunctive binding, which adds nothing robust over a width-matched flat pool. This closes two 2026-09-03
verdicts at once: it promotes satdiv from BORDERLINE to GO with the lever isolated, and it refines the configural-binding
PARTIAL to an honest negative on binding-as-the-mechanism. The next lever for the lane is not another binding topology
(that method is banked) but pushing the divisive-normalization operating point / S2 template learning, which the
2026-09-01 readout-side-exhausted finding already pointed at. A wall defers a METHOD (conjunctive binding), never the
capability (position-invariant object readout), which is GO.
