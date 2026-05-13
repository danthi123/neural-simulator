# Concept pool architecture arc — 2026-05-13 day summary

**Status at session end**: v2b smoke in flight (~30 min), full v2 validation pending result.

## User mandate

2026-05-12: "those scaling axes are 100% what need to be given our full focus
currently, as the blocker for reaching conversational capabilities... it
needs concepts, composition, and diversity."

## Today's progress (25 commits)

### Architecture built (4 motor + N noun + M verb + optional adjective)
- `sim/research/runners/text_minimal_isolation.py`:
  `build_biological_brain_regions` extended with:
  - `enable_noun_pools` + `noun_pool_names` + `n_noun_per_pool` + `n_noun_fs_per_pool`
  - `enable_verb_pools` + `verb_pool_names` + `n_verb_per_pool` + `n_verb_fs_per_pool`
  - `enable_adjective_pools` (3rd kind, opt-in)
  - `_add_concept_kind()` helper applies Tier 1 recipe per kind
  - FS WITHIN-kind only (allows composition across kinds)
  - Reciprocal pool→language_output for A→W readout

### Three Phase demos
- `research/runners/concept_pool_demo.py` — Phase 1 cross-category isolation
- `research/runners/concept_compose_demo.py` — Phase 2 sequential + co-fire
- `research/runners/concept_speak_demo.py` — Phase 3 A→W readout

### Supporting tooling
- `research/runners/concept_pool_repl.py` — interactive shell (hear/speak/compose)
- `research/runners/concept_weight_probe.py` — post-training weight diagnostic
- `research/runners/concept_pool_aggregate.py` — multi-seed analysis

### Tests (25 PASS, CPU-only NumPy backend)
- `tests/test_concept_pool_architecture.py` (18 unit)
- `tests/test_concept_pool_bridge.py` (7 integration)

### Webapp wire-up
- `webapp/server.py` PRESETS["concept_pool_demo|compose|speak"]
- `webapp/static/index.html` launcher dropdown entries
- `webapp/static/ui.js` "Concept pool architecture" category

### Documentation
- `CLAUDE.md` concept pool section added
- `docs/plans/2026-05-13-concept-pool-FS-design-note.md` design lessons
- `research/findings/2026-05-13-concept-pool-architecture-Phase1.md` findings

## Iteration log

### v1: 10 pools (4 motor + 4 noun + 2 verb)
- 200 events, 4096 lang_input, 500 per pool
- Topographic 1.5/0.7 (2.14× ratio)
- **Result: 0/10 PASS**
- **Diagnosis**: verb_pool_COME dominated 9/10 words (firing 2.8-3.2 vs target 0.8-2.8)
- **Root cause**: FS within-kind asymmetry. 2 verb pools means 1 cross-FS
  edge per verb_FS (vs 3 for 4-pool kinds). 1/3 FS suppression → verb
  pools fire freely on all stimuli.

### v2: 12 pools (4 motor + 4 noun + 4 verb)
- Added 2 more verbs (STOP, LOOK) for FS symmetry
- Topographic 2.0/0.5 (4× ratio)
- **Discovered second bug**: training opens all 6 plasticity gates,
  letting off-target pathways accumulate STDP. Fixed to open only the
  target kind's 2 gates per word.
- Initially "hung" 11 min on first word, killed prematurely
- Re-launched as smoke test, first word completed in 174s (just slow,
  not hung)

### v2b smoke (in flight): smaller config to validate
- 200 per pool, 2048 lang_input, 50 events
- Target-only gating + 4 verb pools + tighter topographic
- ETA ~30 min total

## Wall-clock realities

v1: 1314s (~22 min) at 500 per pool × 10 words × 200 events
v2 smoke: ~35 min predicted at 200 per pool × 12 words × 50 events
v2 full: ~2-3 hr predicted at 500 per pool × 12 words × 200 events
Multi-seed (4 seeds × full): ~10 hr

GPU is the bottleneck. Per-step overhead doesn't shrink proportionally
at smaller pools, so smoke isn't much faster than full per word.

## Next steps

Conditional on v2b smoke result:

**If smoke PASS (≥8/12)**: launch full v2b with --save-bridge.
After full PASSes, multi-seed launch via launch_multiseed.ps1.

**If smoke PARTIAL (5-7/12)**: probe weights, iterate on:
- Stronger topographic (3.0/0.3 = 10x)
- Longer training (200 events at smaller scale)
- Longer reset between events (NMDA decay)

**If smoke FAIL (<5/12)**: deeper architectural rework:
- FS strength scaling (Strategy B from design note)
- Lower internal recurrent gain
- Cross-kind weak FS

## Files for handoff

Status JSON: `research/findings/raw/g11_bg/concept_pool_demo/`
- `seed42.json` — v1 FAIL result
- `seed42_smoke_v2b.log` — v2b smoke training log (in flight)

Launchers:
- `launch_multiseed.ps1` — automates seeds 43-46 after seed 42 PASSes
- `launch_v3_nmda_fix.ps1` — fallback with longer NMDA-decay reset
- `post_seed42_analysis.ps1` — automated verdict + next-step recommendation
