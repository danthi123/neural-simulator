---
type: biology
id: conjunction-competitive-selection
mechanism: An overcomplete population of candidate coincidence-binding (conjunction) units competes via lateral inhibition / k-WTA over unlabeled training presentations; only the units whose drive survives the competition often enough are kept in the final bank -- the population SELF-SELECTS which conjunctions are informative for the task's actual input statistics, instead of a fixed-random draw freezing an arbitrary sample of the combinatorial space forever.
status: de-risking
last_verified: 2026-09-09
current_finding: research/findings/2026-09-09-vision-configural-binding-competitive-selection-NEXT-MECHANISM-PREREGISTERED.md
current_status: "BUILT + decisive 6-seed run LANDED (--conj-select competitive in _vision_lindiscrim_readout_derisk.py): LINDISCRIM-READOUT-PARTIAL-beat4/6-lb6/6, the lane's best result to date (RATE_lin_ceiling_held 0.4288 vs the pairwise arm's 0.3403; learning_load_bearing PERFECT 6/6 vs the pairwise arm's 4/6) but not yet a task GO (needs beat>=5/6, landed 4/6 -- one seed missed the margin by 0.0025, one seed is a clear miss). Read as PROGRESS at this mechanism's default operating point, not a banked NO-GO; the named next rung is sweeping --conj-select-overcomplete/--conj-select-kwta-frac, not a new mechanism. Additive + default-OFF: --conj-select fixed (default) is byte-identical to every prior run of this file (verified by re-running the triple-order smoke and diffing every computed field)."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "pattern separation results from the divergence"
    note: "EXTERNAL. The structural motif this mechanism generalizes: 'pattern separation results from the divergence of entorhinal inputs onto a larger number of granule cells in the dentate gyrus' -- expand FIRST (an overcomplete candidate population), THEN sparsify/select. research/biology/dg-ca3-sparse-index.md already establishes this motif for LTM retrieval routing (fixed random sparse projection + per-band k-WTA); this entry applies the SAME expand-then-select structure to a SENSORY feature population (conjunctive binding units) instead of a memory-routing index, with the selection driven by actual training-data competition rather than a fixed hash."
  - path: research/runners/_vision_lindiscrim_readout_derisk.py
    anchor: "Foldiak 1991 / Kohonen 1982-style winner-relative competitive-learning gate"
    note: "LOCAL. The SAME competitive-learning primitive already established one function up in this file: _bcm_learn_s2_templates's competitive_frac restricts a WEIGHT UPDATE to the top-k-by-current-drive templates each presentation (Foldiak 1991 trace/competitive learning; Kohonen 1982 self-organizing competitive nets), found NECESSARY there to stop independent learners collapsing onto one shared direction. This entry's mechanism (_select_conjunctions_competitive) reuses the IDENTICAL top-k-by-current-drive competition rule, applied to SELECT which fixed-random candidate units survive into the bank, rather than to gate a weight update -- selection-by-competition instead of learning-by-competition, the same underlying winner-relative-inhibition arithmetic."
  - path: research/biology/affective-marker-lateral-inhibition-wta.md
    anchor: "mutual/reciprocal lateral inhibition"
    note: "LOCAL. The general form this project already established for a spiking competitive circuit: 'each assembly recruits its own fast-spiking-interneuron sub-pool that cross-inhibits every OTHER assembly (mutual/reciprocal lateral inhibition)' (itself grounded in Grossberg 1973's on-center/off-surround competitive-network motif and Douglas & Martin 2004's canonical cortical microcircuit). That entry selects a DISCRETE OUTPUT WORD among 6 assemblies at read time; this entry instead selects WHICH STRUCTURAL UNITS exist in a feature bank, from training-set competition -- a different point in the pipeline, the same lateral-inhibition/k-WTA competitive primitive."
implemented_by:
  - research/runners/_vision_lindiscrim_readout_derisk.py
findings:
  - research/findings/2026-09-09-vision-configural-binding-competitive-selection-NEXT-MECHANISM-PREREGISTERED.md
---

# Let the conjunction bank compete for itself, instead of freezing a random draw

**The wall this answers.** The pairwise S2.5 configural-binding conjunction bank (`_bind_conjunctions`,
`coincidence-binding.md`) samples its `(a,b,Delta)` triples uniformly at random, ONCE per seed, and freezes them
forever (`research/findings/2026-09-04-vision-crossing-heldout-scramble-anticheat-PARTIAL.md`,
`lb4/6`). The natural escalation -- sample THIRD-ORDER `(a,b,c;Delta1,Delta2)` quadruples instead, on the theory
that a higher-order feature is strictly more specific -- landed WORSE at matched budget and stayed worse at 4x
width (`research/findings/2026-09-09-vision-configural-binding-triple-order-conjunction-NEXT-MECHANISM-
PREREGISTERED.md`): a fixed-random draw over a combinatorial space that grows faster than the unit budget dilutes
coverage no matter how it is widened. **The missing companion process (the wall-reframe question) is competition
itself** -- nothing in the fixed-random scheme lets an overcomplete candidate population fight over which of its
members actually fire informatively on this task's real images; it just accepts whatever a seed happened to draw.

**The mechanism.** Sample a candidate bank several times larger than the target width (the SAME
`_make_conjunction_bank`/`_make_conjunction_bank_triple` fixed-random sampler, unchanged, just asked for more
units -- the DG-style expand-first step). Drive every candidate unit on the TRAINING patches only (never held or
scrambled -- the same anti-leakage discipline `_bcm_learn_s2_templates` follows). Run a lateral-inhibition /
k-WTA competition ACROSS candidates, per `(image, location)` presentation: only the top fraction survive, the
rest are zeroed (a Foldiak 1991 / Kohonen 1982 winner-relative competitive-learning gate, the SAME
arithmetic `_bcm_learn_s2_templates`'s `competitive_frac` already validated on this substrate for weight
updates, applied here to a selection decision instead). Accumulate each candidate's cumulative SURVIVING
(post-inhibition) drive across every training presentation; keep the top `conj_n` candidates. The result is a
conjunction bank whose membership is determined by which candidates actually WON competitions against real
training input, not by which indices a random-number generator happened to draw.

**Why this is not a new primitive.** The coincidence AND (`_bind_conjunctions`) is completely unchanged --
`coincidence-binding.md` still grounds it. What changes is only WHICH `(a,b,Delta)` triples get to compute that
AND at all, decided by a competitive process this project has already validated twice (the affect-marker WTA's
mutual lateral inhibition, and the S2-BCM competitive-learning gate) -- generalized here to a THIRD point in the
pipeline: selecting fixed-random STRUCTURE rather than a discrete output or a learned weight.

**The honesty boundary.** No labels are used in the selection (it runs on `tr_c1`/training IMAGES, not
`tr_cls`), so a capability gain here is not label-leakage -- but it IS an additional pass over the training
SET (not a single stimulus), which a biological circuit would realize as a developmental/early-experience
epoch of competitive refinement rather than a single-trial computation; this is the SAME idealization the S2
BCM mechanism already carries (`_bcm_learn_s2_templates`, multiple epochs over a small stimulus set), not a
new one. The candidate sampler, offset range, and k-WTA fraction remain de-risk operating-point knobs, not
biology-REQUIRED constants (no `constraints_config` is bound, matching `_bcm_learn_s2_templates`'s own entry
convention).
