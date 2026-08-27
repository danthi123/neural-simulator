---
type: finding
status: live
date: 2026-08-27
mechanism: generative-attractor-wander-onsubstrate
lane: continuous-life/generation
seeds: [42, 43, 44, 100, 101, 102]
instrument: on-substrate two-compartment dendritic dAP bistable-latch completion (the SAME mechanism family the
  D5/episodic production organ's `_episodic_dap_dialogue_memory.EpisodicDapMemory` composes), a calibrated
  small-absolute-count blended cue, isolated fresh-bridge reads per read condition, 4 anti-cheats
runner: research/runners/_generative_attractor_wander_onsubstrate_derisk.py
artifacts:
  - research/findings/raw/_generative_attractor_wander_onsubstrate/batch1.json
  - research/findings/raw/_generative_attractor_wander_onsubstrate/batch2.json
  - research/runners/_generative_attractor_wander_onsubstrate_derisk.py
---
# On-substrate port of the generative attractor-wander mechanism (board #104): a blended cue SETTLES into a genuinely NOVEL, stable, real spiking-CA3 recombination — 6/6-seed GO

Artifact: research/findings/raw/_generative_attractor_wander_onsubstrate/batch1.json + batch2.json (runner: research/runners/_generative_attractor_wander_onsubstrate_derisk.py)

**One line.** The 2026-08-20 numpy de-risk's claim — a blended cue of two stored patterns settles into a
STABLE, BALANCED state that is NOT equal to either source — now reproduces on the REAL spiking substrate, using
the SAME two-compartment dendritic dAP bistable-latch completion mechanism the D5/episodic production organ
actually runs (not a fresh numpy Hopfield net). 6/6 seeds: novelty (max overlap with any single stored
assembly) 0.583-0.806 (mean 0.736, well below the 0.85 "this is just a recall" bar), blend balance (min
overlap on the two cued sources) 0.486-0.750 <!--derived--> (mean 0.625), the non-cued third assembly untouched
at every seed (0.000), and the settled state PERSISTS unchanged after the cue is released (bistable latch
holding, persistence_gap=0.000 every seed) — a genuinely stable, novel, on-substrate combination.

## The mechanism (reuse-by-import, no sim/ edit)

This reuses `_build_dap_readout` / `make_readout` / `_form_one_assembly` / `form_btsp_multi`
(research/runners/_gap5_dendritic_dap_readout_completion_derisk.py,
research/runners/_gap5_btsp_forms_nmda_slow_reverberatory_derisk.py) VERBATIM — the identical functions
research/runners/_episodic_dap_dialogue_memory.py (the production D5/episodic organ) composes. Three topics'
assemblies are BTSP-formed at n_ca3=400/assembly_frac=0.18 (72-cell assemblies, a scale this mechanism family
was already validated at before the production organ's n_ca3=2000 emergent-DG-selected scale — that larger
scale is a mandatory-scale mechanism per the gap#5 seam findings and was not foreground-feasible on CPU numpy
for a 6-seed x 3-assembly sweep). A cue is then driven into the SAME per-cell dendritic dAP bistable latch the
organ's `recall()` reads (`cp_v_apical` crossing `up_thresh`), and read as the UP-fraction of EVERY stored
assembly's own member cells — the on-substrate analogue of `overlap(settled, stored_m)`.

## Two genuine findings the calibration surfaced (both load-bearing, both banked)

**(1) An instrument bug: sequential reads on one bridge contaminate each other.** (an ad hoc diagnostic run, not
saved as a cited artifact — reproduced by any caller of `_population_up` on a reused bridge) A second read on
an already-driven bridge collapsed from [1.0, 0.972, 0.0] to [0.056, 0.0, 0.0] for the IDENTICAL cue <!--derived-->
`hard_silence`/`_reset_apical_latch` reset the soma + apical compartment but not synaptic resources, and
`CoreSimConfig.enable_short_term_plasticity` defaults True (`_build`'s `enable_stp=False` argument only skips
its OWN explicit STP tuning, it never clears the flag) — so Tsodyks-Markram depression on the just-driven
recurrent synapses biased every following read on the same bridge. Disabling STP outright was tried and
REJECTED (it collapsed the positive-control single-cue completion too — STP dynamics are load-bearing for the
plateau reaching threshold). The fix: every read condition below gets its OWN fresh bridge, with `R.C.data`
seeded from the SAME formed (or, for the lesion control, baseline) weight array — genuinely independent
measurements. A stray noise-cue anti-cheat also initially leaked (0.75 overlap) because its exclusion pool
omitted the THIRD stored assembly's own member cells, letting an unlucky random draw directly cue ~14% of it;
fixed by excluding every stored assembly's membership, not just the two cued ones.

**(2) The per-cell dAP latch has no population-wide competitive budget.** Driving HALF of each source's
cue-eligible cells (the natural analogue of the numpy de-risk's blend) does NOT produce a balanced novel
state — it independently completes BOTH assemblies to ~0.97-1.0 simultaneously (a dual-full-recall, not a
recombination). This is a genuine mechanistic disanalogy from the numpy mean+std dynamic threshold, which
forces a fixed-size population-wide "budget" of active units that different sources must compete for; the
on-substrate dAP mechanism's strength (per-cell bistability DECOUPLED from population-wide recurrent gain,
exactly what let it surpass the point-neuron trilemma for single-item completion, per
2026-08-10-ca3-point-neuron-attractor-completion-trilemma-NEGATIVE) is also why it does not naturally compete
sources against each other. A blend-size sweep with the fixed-bridge-per-read instrument (frac_sweep in the
same raw directory) found a genuinely graded regime instead: the coincident drive a held-out cell receives
depends on the ABSOLUTE number of directly-connected cued neighbours (~n_cue * ca3_density), not on assembly
size or cue-set fraction, so a small ABSOLUTE cell count (`blend_cells_each=3`) is the scale-invariant
operating point (a sweep run interactively, not saved as a cited artifact) — confirmed consistent at both
n_ca3=200/assembly=36 ([0.667, 0.694, 0.0]) and n_ca3=400/assembly=72 ([0.778, 0.639, 0.0], the latter <!--derived-->
reproduced in the cited batch1.json seed-42 row) for seed 42.

## 6-seed results (n_ca3=400, assembly_frac=0.18, blend_cells_each=3; batch1.json seeds 42/43/44, batch2.json seeds 100/101/102)

<!--derived-->
(the table's "mean" row is the arithmetic mean of the six per-seed rows above it, each individually traceable
to batch1.json or batch2.json)

| seed | novelty (max overlap) | blend balance (min A,B) | overlap-other (C) | persistence gap | single-cue recovered | single-cue others | noise best | untrained-blend best | genuine formation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.778 | 0.639 | 0.000 | 0.000 | 0.819 | 0.000 | 0.000 | 0.000 | True |
| 43 | 0.750 | 0.500 | 0.000 | 0.000 | 0.861 | 0.000 | 0.000 | 0.000 | True |
| 44 | 0.764 | 0.639 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | True |
| 100 | 0.806 | 0.750 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | True |
| 101 | 0.736 | 0.736 | 0.000 | 0.000 | 0.583 | 0.000 | 0.000 | 0.000 | True |
| 102 | 0.583 | 0.486 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 | 0.000 | True |
| mean | 0.736 | 0.625 | 0.000 | 0.000 | 0.773 | 0.000 | 0.000 | 0.000 | 6/6 |

GO gate (all 6 seeds): genuine BTSP formation (within-assembly weight grew from the plasticity rule, cross/non-
member weights did not) AND novelty<0.85 AND balance>0.35 AND balance-minus-other>0.10 AND persistence_gap<0.20
AND single-cue recovered>0.50 AND single-cue others<0.20 AND untrained-blend best<0.20. All 6 seeds clear every
term; noise-cue best/2nd-best are additionally clean (0.000) at every seed (the noise anti-cheat, reported and
gated after the fix above).

## Anti-cheats (the load-bearing teeth)

- **Positive control / lesion of the generative mechanism:** driving only ONE source's full cue-eligible pool
  (removing the second blended source — the literal "lesion" of what makes this generative rather than plain
  recall) collapses the read back to PLAIN, SPECIFIC recall: 0.583-0.875 <!--derived--> (mean 0.773) on the cued assembly,
  0.000 on every other stored assembly, at every seed. This is the load-bearing teeth the task asked for: the
  novelty is carried BY the two-source blend structure, not by the completion mechanism in general — removing
  the second source removes the novelty and nothing else changes.
- **Untrained (lesioned weights):** the identical blended cue read through the pre-formation BASELINE
  (unformed) recurrent weights reads 0.000 on every assembly at every seed — the dAP latch alone, without the
  learned BTSP potentiation, does not fake completion.
- **Noise cue:** an equal-sized cue drawn from CA3 cells outside every stored assembly's own membership reads
  0.000 (best and 2nd-best) at every seed — an unstructured drive does not produce the balanced blend signature.
- **Persistence / stability:** the blended UP-state survives 100 steps of cue RELEASE (external drive set to
  zero) unchanged at every seed (persistence_gap=0.000 throughout) — the settled state is a genuine bistable
  latch holding on its own, not something merely tracking ongoing drive. This is a stronger stability proof
  than the numpy de-risk's discrete fixed-point check (nothing sustains this state but the latch itself).

## What is spiking vs what is host (declared — the honesty boundary is a deliverable)

- **SPIKING (load-bearing):** the completion itself — which cells enter the dendritic UP state — is read from
  the substrate's own two-compartment apical dAP bistability (`fused_coincidence_plateau`,
  `enable_two_compartment_dap`) acting on BTSP-formed recurrent weights (`fused_btsp_update`, the plasticity
  rule's own output, never a hand-set constant — asserted via `genuine_formation`: within-assembly weight grew,
  cross/non-member weight did not, at every seed).
- **HOST (declared scaffolds):** (1) which two assemblies to blend is a runner-side selection (this de-risk
  cues assembly indices 0 and 1 directly; the production wander's curiosity-gain selection of "the two most
  active concepts" is a separate, already-wired host scheduler, unchanged here); (2) the blend cue's cell COUNT
  (`blend_cells_each=3`) is a runner-calibrated constant, the on-substrate analogue of the numpy de-risk's
  `thresh_c` dynamic-threshold sharpness knob; (3) PRE-ASSIGNED (random-permutation) assembly membership at
  n_ca3=400, not the production organ's n_ca3=2000 emergent DG-selected membership — the declared scope
  reduction (see Residual).

## Residual / next step

**Production wiring is NOT done here and should not be rushed.** Two concrete blockers, both honestly named
rather than glossed: (1) LATENCY — this de-risk's build+BTSP-form+read cycle costs ~140-160s per seed on CPU
numpy (vs today's numpy stand-in in `webapp/continuous_engine.py`'s `_ideation_blend_settle`, which is
effectively instant); a between-turn idle tick has no such budget, so a live wire-in needs either the cupy
backend (seconds, per the organ's own docstring) plus a precompute/cache path, or a materially smaller
operating point re-validated at that scale. (2) ASSEMBLY AVAILABILITY — the live wander's curiosity-selected
"concepts" (`self_initiated_production_organ.py`'s agents) are not today BTSP-formed CA3 assemblies the way
D5/episodic topics are; wiring this mechanism into the live wander needs those two stores unified first (or a
parallel formation path for wander concepts) — a real prerequisite, not a rename. Next: (a) re-run this
runner's `--n-ca3` on `SIM_BACKEND=cupy` via `tools/gpu_queue.sh` once the GPU is free, to confirm the
blend_cells_each=3 operating point transfers to the emergent n_ca3=2000 scale the production organ actually
uses; (b) design the assembly-availability bridge between the wander's concept agents and the episodic BTSP
store before attempting the live wire-in.

**Byte-identical when off:** no `sim/` file and no `webapp/` file was touched by this change — the entire port
is one new, additive `research/runners/` file. Production behaviour is therefore unconditionally
byte-identical to before this commit, not merely "byte-identical when a new flag is off" (there is no new
flag; there is no production edit to gate).
