# Generative replay + PFC-held compositional frame architecture design (6th arc after the 5-architecture convergent ceiling)

> **For Claude / autonomous continuation:** This is the **design** for the
> 6th architecture in the gating-based composition design line. After
> approval, writing-plans produces the TDD implementation plan, then
> subagent-driven-development builds Task 0..Task 5 with dedicated
> adversarial review + controller-only decisive run. Mirrors the prior
> 5 arcs' (Stage-1 + SPEAR + Pirazzini + Unified + Theta-gamma)
> discipline exactly.

## Status

Pre-registered NEW design grounded in three established facts:

1. **5-architecture convergent ceiling (commit `1bbc165`, both remotes)**:
   Stage-1, SPEAR, Pirazzini, Unified per-regime monitor, and
   Theta-gamma all failed decisively at biological scale with different
   mechanism-level signatures. Gating-based composition + cue-suppression-
   during-retrieve are empirically exhausted on the v14/v16+hippocampus
   substrate.

2. **Theta-gamma finding (commit `1bbc165`)**: cue-suppression-during-
   retrieve is biologically backwards. It violates the encoding-
   specificity principle (Tulving 1973). The cue is BOTH a noise source
   AND useful encoding-context; at biological scale the context-loss
   outweighs the noise-removal -> NEGATIVE per_regime_advantage. So the
   6th arc REMOVES the cue-suppression and keeps the cue present during
   retrieve.

3. **Standing direction (user-directed; design doc
   `docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md`
   section 2b)**: the catalog-grounded conversational direction has
   THREE load-bearing pieces: (a) theta-gamma mode-unification [tested,
   FAIL]; (b) GENERATIVE REPLAY (proposing-and-pattern-completing
   compositional hypotheses during NREM-equivalent cycles); (c) PFC-HELD
   COMPOSITIONAL FRAME (the prefrontal working memory holding the
   ordered compositional structure across queries via NMDA bistable
   attractors). This 6th arc adds (b) + (c) as augmenting mechanisms,
   removing (a)'s cue-suppression.

## 1. The capability under test (falsifiable)

Compositional retrieval that emits the bound adjective when the noun
is cued, where the substrate is REPEATEDLY EXPOSED to its own engram
tags via generative replay BEFORE the eval queries, AND a PFC
compositional frame primes the substrate during eval to expect
compositional readout. Frozen bars identical to the prior 5 arcs
(`_GR_FULL_MIN=0.80, _GR_UNIFORM_CTRL_MAX=0.10, _GR_DIRECT_RETAIN_MIN=0.80,
_GR_ABSTAIN_CORRECT_MIN=0.90, _GR_SCALE_TOL=0.10, _GR_LADDER=(2,3,5),
_GR_MIN_SEEDS=3`); module-local `_GR_*` constants distinct from
`_TG_*` and `_PR_*` even though values match.

The experimental contrast:
- **FULL arm**: engram tag encode + N cycles generative replay + PFC
  compositional frame active during query + cue PRESENT during retrieve
- **UNIFORM_CTRL arm**: engram tag encode + 0 cycles generative replay
  (replay disabled) + PFC compositional frame disabled + cue PRESENT
  during retrieve (cue parity preserved; encoding-specificity respected)

If the augmenting mechanisms (replay + PFC-frame) genuinely help
compositional retrieval, FULL_acc > UNIFORM_CTRL_acc with
per_regime_advantage >= 0.70 at smallest-N rung.

## 2. The mechanism being added (load-bearing; grounded in the 5-architecture ceiling + standing design)

### Generative replay loop (REUSED subsystem; no protected modification)

The project has a VALIDATED `run_concept_replay_phase` subsystem at
`research/runners/consolidation_trainer.py:43`. It cycles through engram
tags, stimulating each repeatedly (default `n_replays_per_tag=20`)
during a NREM-equivalent ACh-low + SWR-burst window. The replay
strengthens the CA3-CA1 pathway via STDP at the bound-adj-tagged
neurons.

For the 6th arc, this is invoked between encode and eval:
1. Encode the N compositional pairs as engram tags (REUSED `_encode_facts`)
2. **NEW**: Run `run_concept_replay_phase` with the encoded tag names
   for K = 20 cycles per tag (the validated default)
3. Eval queries as before, but the replay-strengthened pathway should
   now produce stronger bound-adj firing during retrieve

The UNIFORM_CTRL arm skips step 2 (K=0 replay cycles).

### PFC-held compositional frame (REUSED subsystem; no protected modification)

The project has a VALIDATED `dlpfc_verb` region in
`build_biological_brain_regions(..., enable_dlpfc_verb=True)` with 300
neurons + Cluster-G v2.5 per-region NMDA bistable attractors (the
validated NMDA subsystem flag `enable_pfc_nmda`). At biological scale
the dlpfc_verb region holds working-memory state via NMDA bistability.

For the 6th arc, the dlpfc_verb region is BRIEFLY DRIVEN at eval start
(per-query) to set a compositional-frame attractor; the NMDA
bistability holds the frame across the query's encode + retrieve
windows; the frame primes the substrate to expect compositional
readout.

Implementation route (reuse-by-import; no protected/frozen module
touched):
- `bridge.cp_external_input_current[dlpfc_verb slice] += pfc_frame_pA`
  during the eval-query window
- `pfc_frame_pA` derived from the cued noun (so the frame is content-
  primed, mirroring the biological PFC compositional structure)
- The UNIFORM_CTRL arm skips this write (dlpfc_verb stays unprimed)

## 3. What the 6th arc REMOVES (per the 5-architecture findings)

- **Cue suppression during retrieve** (theta-gamma's mechanism;
  REMOVED per encoding-specificity violation finding). The cue is
  PRESENT during retrieve in both arms. Both arms see the same cue;
  the SOLE difference is the augmenting mechanisms (replay + PFC-frame
  ON in full; OFF in uniform_ctrl).
- **The phase-multiplexed theta cycle** (encode/gap/retrieve phases).
  The 6th arc uses the simpler single-phase eval (drive cue + measure
  lang_output) like the unified arc; the theta-cycle was structurally
  active but produced an anti-effect.
- **The substrate-specific compositional gate at 0.198**: STAYS active
  (per the 4-times-validated substrate-and-protocol-specific principle;
  unchanged byte-stable).

## 4. Inventory of reused subsystems (byte-unchanged)

- `build_biological_brain_regions(..., enable_hippocampus_consolidation=True,
  enable_dlpfc_verb=True, enable_pfc_nmda=True, enable_noun_pools=True,
  enable_verb_pools=True, enable_adjective_pools=True)` -- the SAME
  unified substrate; same Phase-1 cached checkpoints
- `encode_concept_pair` (`compose_concept_engram.py`) -- encoding
- `run_concept_replay_phase(bridge, tag_names, n_replays_per_tag=20)`
  (`consolidation_trainer.py:43`) -- generative replay (NEW load-bearing
  reuse for this arc)
- `_compositional_query_ranked` (`unified_per_regime_monitor_runner.py:804`)
  -- readout
- `per_regime_monitor_core.per_regime_monitor_verdict` -- REFERENCE
  ONLY; the new arc has its own `generative_replay_pfc_frame_core.py`
  with identical-value but module-local `_GR_*` constants
- The 4 calibrated abstention moats byte-unchanged
- The neuromodulator subsystem byte-unchanged
- `sim/bridge.py` byte-unchanged

The genuine net-new code: ~600-800 line runner that wires replay +
PFC-frame into the unified-substrate eval loop + a frozen
capability-verdict module (transcription with `_PR_*` -> `_GR_*` rename).

## 5. Pre-registered next staged step

After this design ships:

- Task 0: grounding pin test (RED until Tasks 1 + 2)
- Task 1: net-new frozen capability-verdict module
  `research/runners/generative_replay_pfc_frame_core.py` (transcribe
  from `per_regime_monitor_core.py`; rename `_PR_*` -> `_GR_*`; verdict
  function rename; 17+ adversarial test cases)
- Task 2: net-new runner
  `research/runners/generative_replay_pfc_frame_runner.py` (~600-800
  lines mirroring `unified_per_regime_monitor_runner.py` structure but
  ADD replay phase between encode + eval; ADD PFC-frame priming during
  eval queries; REMOVE theta cycle; reuse-by-import only; no autograd)
- Task 3: 10th consecutive dedicated adversarial review (subagent;
  specific exploit-class probes: replay-effect probe; PFC-frame-effect
  probe; false-PASS vector; byte-unchanged audit; no autograd;
  encoding-specificity preserved)
- Task 4: no-harm verification
- Task 5: controller-only decisive run + smell-test + honest propagation

## 6. Honest ceiling (binding throughout)

- A PASS would be the FIRST architecture in the 6-arc series to clear
  the frozen bars; biology-grounded compositional retrieval at small
  loads; NOT yet fluent open-ended language.
- A FAIL extends the convergent ceiling to SIX architectures.
- If the 6th arc also fails, the gating-based composition design line
  is structurally exhausted under the project's currently-validated
  subsystems. The terminal biology-translatable finding would be: the
  v14/v16+hippocampus substrate cannot produce reliable compositional
  retrieval via any combination of (gating + theta-multiplexing +
  cue-suppression + replay-augmentation + PFC-frame-priming) when the
  underlying retrieval mechanism is "drive cue + measure lang_output
  cosine" -- a deeper SUBSTRATE design refinement would be required
  (e.g., per-region inhibitory normalisation to suppress cross-pathway
  interference; or replace the cosine-cosine readout with something
  more selective).

## 7. Discipline pins (mirrors prior 5 arcs)

- NO bar change anywhere; the new `_GR_*` constants are set in
  advance and NEVER tuned in response to results.
- NO protected file modification; the protected set byte-empty diff vs
  `e8a99a2` must continue to hold across every commit.
- NO autograd / no torch / no LLM call.
- NO declare-unfit; NO hand-back; NO config-crank.
- Mandatory dedicated adversarial review BEFORE no-harm BEFORE decisive
  run.
- Honest propagation EVERY outcome both remotes.
- The autonomous next-action tool call is always in the same turn.
- The 4 substrate-and-protocol-specific calibrated moats stay
  byte-stable.
- The no-confabulation moat (`abstention_gate.py` + tests) stays
  byte-identical and 7/7 green.

## 8. Next-step pointer (writing-plans)

After approval of this design, writing-plans produces the TDD
implementation plan at
`docs/plans/2026-05-20-generative-replay-PFC-frame-implementation.md`.
The plan transcribes Tasks 0..5 mirroring the theta-gamma plan's
structure. Then subagent-driven-development executes the plan.
