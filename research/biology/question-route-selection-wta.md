---
type: biology
id: question-route-selection-wta
mechanism: >
  Which comprehension-ROUTE handles an incoming question (a fronted-relation construction, a curated
  multi-word-relation/idiom construction, a definitional-copula construction, or the elsewhere/DEFAULT generic
  SVO parse) is decided by N=4 competing excitatory assemblies, each with its own fast-spiking-interneuron
  sub-pool that cross-inhibits every OTHER assembly (mutual/reciprocal lateral-inhibition WTA, generalized from
  the already-validated N=6 affect-marker circuit to N=4 route candidates). Each assembly is driven by a
  host-extracted LEXICAL-CUE current (the SAME regex/shape evidence the host router already computes as a
  matched-filter read of the surface string -- legitimate sensory-style feature extraction, not the decision);
  the DEFAULT (generic-SVO) assembly additionally carries a constant baseline "elsewhere" drive so it wins
  whenever no construction-specific cue is genuinely recognized, and is overridden (recognition-gated blocking)
  only when a specific construction's cue drive is strong enough to win the lateral-inhibition race -- the
  Pinker-Ullman words-and-rules default/elsewhere principle, realized here as a population competition instead
  of a Python if/elif priority cascade.
status: de-risking
last_verified: 2026-09-09
current_finding: research/findings/2026-09-09-rank14-question-route-selection-wta-derisk-GO.md
current_status: >
  BUILT + decisive 6-seed run LANDED (`_rank14_question_route_selection_derisk.py`, reusing
  `_affect_marker_wta_derisk.py`'s `_build_bridge`/`_pool_rates` N-pool cross-inhibition primitive verbatim, N=4
  route channels). See the current_finding for the measured parity/lesion/attribution/sweep numbers and verdict.
  PRODUCTION WIRE-IN LANDED default-OFF (2026-09-09, `spiking_qroute_selection_organ.py` +
  `brain_chat_tui.ChatBrain._extract_route`/`_spiking_route_decision`, flag `BRAIN_SPIKING_QROUTE`): focused verify
  GO (byte-identical-off; load-bearing-on incl. ambiguous + lesion, 6 seeds). The wire-in builds a FRESH bridge per
  decision (not one reused bridge) -- the tight ambiguous margin is sensitive to Izhikevich adaptation carryover
  across arbitrary question sequences (measured 0.14->0.009 on reuse). The FLIP to default-ON awaits an integrated
  AWS-CPU no-regression soak. Still `status: de-risking` (not production-ON).
sources:
  - path: /home/dant123/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "becomes went rather than goed and break becomes broke"
    note: >
      EXTERNAL. Kandel 6e Ch 55 (Language), p.1373 -- the SAME anchor `dual-route-past-tense-recognition-gated-
      blocking.md` already uses. Grounds the DEFAULT/elsewhere half of this mechanism: a regular/default
      procedure applies UNLESS a specific stored/recognized exception is genuinely retrieved above criterion.
      Here the "regular rule" is the generic-SVO comprehension route (already the substrate's own neural
      BridgeParser, `ChatBrain._extract_route`'s CHOOSE branch) and the "stored exceptions" are the three
      construction-specific routes (relation-fronted / kb-relation / definitional-copula) that should win ONLY
      when their own cue is genuinely present, exactly mirroring how "went" blocks "-ed" only on recognized
      retrieval and a novel stem leaves the default rule unopposed.
  - path: research/biology/affective-marker-lateral-inhibition-wta.md
    anchor: "mutual/reciprocal lateral inhibition"
    note: >
      LOCAL. The N-way (there N=6 valence + N=2 arousal) per-assembly-FSI cross-inhibition WTA architecture this
      entry's mechanism reuses WHOLESALE (same builder function, `_build_bridge`, imported not reimplemented) at
      N=4 route candidates instead of N=6 affect registers. That entry's own citation chain (Grossberg 1973
      on-center/off-surround competitive network; Douglas & Martin 2004 canonical cortical microcircuit;
      generalized there from the 2-channel `bg_action_selection_production_organ.py` SPEAK-vs-STAY-SILENT
      circuit) grounds the competitive-selection PRIMITIVE itself; this entry is a further generalization of the
      SAME primitive to a different decision (which comprehension construction applies) with a different
      per-channel drive (discrete lexical-cue evidence instead of a Gaussian population-vector tuning curve).
  - path: research/runners/_affect_marker_wta_derisk.py
    anchor: "lateral-inhibition WTA motif, generalized from the 2-channel `_vocal_action_selector_gate` precedent"
    note: >
      LOCAL. The exact reused builder (`_build_bridge(seed, n_pools, prefix)`): n_pools excitatory assemblies,
      each with its own dedicated FSI sub-pool that inhibits every OTHER assembly (never itself) -- imported
      verbatim by `_rank14_question_route_selection_derisk.py`, not reimplemented, so a change to the primitive
      has one source of truth.
implemented_by:
  - research/runners/_rank14_question_route_selection_derisk.py
  - research/runners/spiking_qroute_selection_organ.py
  - research/runners/_rank14_qroute_wirein_verify.py
findings:
  - research/findings/2026-09-09-rank14-question-route-selection-wta-derisk-GO.md
  - research/findings/2026-09-09-rank14-question-route-selection-wta-wirein-default-off.md
---

# Which question-comprehension route wins is a spiking competition, not a Python if/elif cascade

**What this retires (scaffold-retirement-backlog rank-14, "NL question-routing host comprehension").**
`research/runners/brain_chat_tui.py`'s `ChatBrain._extract_route` decides, in host Python, which of four
comprehension routes handles an incoming question by testing three regex-shaped special cases in a FIXED
priority order (`_relation_fronted_route` -> `_kb_relation_question_route` -> `_definitional_copula_route`,
guarded by `len(content) <= 1`) and falling through to the generic (already-neural) SVO parse only when none
matched. Each individual regex test is a legitimate matched-filter read of the surface string (the same honesty
class as any other host-side sensory feature extraction in this codebase); what was NOT neural is the
COMBINATION -- deciding, when evidence for more than one construction could in principle be present, which one
actually wins, and doing so unconditionally in a fixed textual order rather than by the STRENGTH of the
recognized cue.

**The mechanism.** Four small excitatory assemblies (RELFRONT, KBREL, DEFCOP, GENERIC), each with its own
fast-spiking cross-inhibition sub-pool (`_affect_marker_wta_derisk._build_bridge`, reused verbatim with
`n_pools=4`), compete under mutual/reciprocal lateral inhibition. Each of the three construction-specific
assemblies is driven by a current proportional to whether its OWN host regex genuinely matched the question (0
or a fixed drive); GENERIC additionally receives a constant BASELINE "elsewhere" current so it wins by default
when no construction cue fires. The assembly whose rate clears the runner-up by a dead margin after the network
settles IS the route decision -- read off `cp_firing_states`, never a host `if`/`elif`.

**Why this is the right generalization of an established pattern, not a new primitive.** The N-way
cross-inhibition WTA circuit is imported, not reimplemented, from the ALREADY-GO'd + production-flipped N=6
affect-marker selector (`affective-marker-lateral-inhibition-wta.md`), itself a generalization of the 2-channel
SPEAK-vs-STAY-SILENT basal-ganglia action selector. What is new here is only the per-channel DRIVE semantics
(discrete lexical-cue evidence rather than a Gaussian population-vector tuning curve over a shared continuous
axis) and the DEFAULT-channel baseline current, which realizes the Pinker-Ullman "elsewhere" case documented in
`dual-route-past-tense-recognition-gated-blocking.md` -- a rule/default assembly that is always minimally driven
and is overridden only by a genuinely recognized, sufficiently strong competing cue.

**Honesty boundary.** The regex feature EXTRACTION (does the fronted-relation shape match; does one of the 29
curated kb-relation/idiom shapes match; does the definitional-copula shape match) stays host code -- the same
class of scaffold `_relation_fronted_route`/`_kb_relation_question_route`/`_definitional_copula_route` already
are, and this entry does not retire THEM (that is a separate, larger residual: teaching the substrate to
recognize these constructions from corpus statistics rather than a curated regex table, out of scope here). What
moves onto the substrate is only the DISPATCH -- which route's evidence wins when more than one could apply, and
whether the default is genuinely unopposed when none does.
