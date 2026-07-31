---
type: plan
status: live
date: 2026-05-20
---

# 8th arc: dedicated compositional-readout region design

> **For Claude / autonomous continuation:** This is the **design** for
> the 8th architecture in the substrate-level refinement direction
> (direction A from the day-consolidated summary at commit `351366f`).
> Mirrors the prior 7 arcs' discipline.

## Status

Pre-registered NEW design grounded in:

1. **6th arc local-optimum confirmation (commit `f004da5`)**: the gating
   + augmenting composition design line is asymptotically exhausted at
   N=3 full_acc = 0.458; the substrate's cosine-readout-via-engram-tag-
   stim mechanism has a CEILING around this value.

2. **Localised substrate bottleneck (commit `0ef9b6e`)**: bridge-state
   perturbations from input augmenting mechanisms (cue-suppression,
   amplified-tag, persistent-PFC) are absorbed by downstream FS
   interneuron normalisation + abstention-gate thresholding before
   reaching the gated answer. The READOUT is the bottleneck.

3. **Original localisation finding (commit `110f7cd`)**: the cued-noun's
   diffuse `lang_input` drive dominates the engram tag's selective
   bound-adj drive at the lang_output cosine readout. The spelling
   readout is the wrong signal for compositional retrieval.

## 1. The mechanism being added (load-bearing)

**Dedicated compositional-readout region** (the genuine net-new piece):

A new region added to the substrate, trained specifically on compositional
outputs (not spelling). The region's pattern is read out via a new
gated pipeline, BYPASSING the lang_output spelling-cosine pathway.

### Architecture

- Region name: `composition_readout` (200 neurons; cortical RS type;
  mirrors the existing concept-pool architecture)
- New plastic pathway: `engram_tag_stim_region -> composition_readout`
  (sparse; STDP-learning during the new training pre-stage)
- Existing plastic pathway: `composition_readout -> lang_output_alt`
  (decoded to compositional outputs; e.g., "big", "hot", "cold")
- Training data: (cue noun, bound adj) -> teach `composition_readout`
  to fire a pattern correlated with the adj's existing target pool

### Training pre-stage (NEW, controller-driven)

For each (cue, bound-adj) pair in the training vocab:
1. Drive `lang_input(cue)` and `lang_input(adj)` simultaneously
2. Open the new `engram_tag_to_composition_readout` plasticity gate
3. Run STDP for ~500 ms
4. Snapshot the resulting weights

This pre-stage runs ONCE before the decisive eval; produces a
substrate that has learned cue->bound-adj compositional mappings via
the dedicated readout pathway.

### Eval pipeline

- Drive cue noun (encoding-specificity respected; cue PRESENT)
- Stim the engram tag (selectively activates `composition_readout`)
- Measure `composition_readout` firing pattern (not lang_output)
- Decode by cosine match to stored compositional patterns

## 2. Experimental contrast

- **FULL arm**: NEW pipeline -- drive cue + tag stim + read
  `composition_readout` + cosine match
- **UNIFORM_CTRL arm**: OLD pipeline -- drive cue + tag stim + read
  `lang_output` + cosine match (the 6th arc's baseline; full=0.458)

If FULL > UNIFORM_CTRL by per_regime_advantage >= 0.70 at smallest-N,
the new readout mechanism closes the gap. If FULL is below or near
UNIFORM_CTRL, the new readout doesn't help and the substrate's
underlying retrieval mechanism (engram tag stim) is the deeper
limit.

## 3. Implementation route (disciplined; no protected modification)

Net-new substrate-builder function `build_substrate_with_composition_readout`
in `research/runners/8th_arc_composition_readout_runner.py` (or a
separate substrate-builder helper module). This function INTERNALLY
calls `build_biological_brain_regions` byte-unchanged and then ADDS
the new region + pathways via the existing brain-region framework
API:

```python
def build_substrate_with_composition_readout(seed, ...):
    """Build the unified substrate + add a new composition_readout
    region. The base substrate is build_biological_brain_regions
    byte-unchanged; the new region + pathways are added on top via
    the existing brain-region framework API.
    """
    bridge = build_biological_brain_regions(seed=seed, ...)  # REUSED
    # Add new region (additive; no protected modification)
    bridge.add_region("composition_readout", n_neurons=200,
                       neuron_type="IZH2007_RS_CORTICAL_PYRAMIDAL")
    # Add new pathways (additive)
    bridge.add_pathway("engram_tag_stim_region",
                        "composition_readout",
                        weight_mean=0.5, plasticity_gate="comp_readout_training")
    return bridge
```

If `bridge.add_region` / `add_pathway` aren't part of the public API,
the new substrate-builder uses lower-level brain-region framework
APIs (RegionManager.add_region; etc.) WITHOUT modifying the protected
files.

If neither path is feasible without protected modification, an
alternative: SKIP the new region entirely and instead REPURPOSE an
existing region (e.g., `dlpfc_verb` for verb composition; `adjective_pool_*`
for adjective composition) by training a new gated pathway INTO it.
This doesn't add new regions; just net-new pathways via existing
framework hooks.

## 4. Frozen verdict module + bars

Mirrors prior 7 arcs. New module-local constants `_CR_*` (CR =
Composition Readout) with VALUES IDENTICAL to all prior arcs:
- `_CR_FULL_MIN = 0.80`
- `_CR_UNIFORM_CTRL_MAX = 0.10` (uniform_ctrl is the 6th arc's
  lang_output pipeline; at the 6th arc's 0.458 it would NOT be below
  0.10 -- this is the load-bearing experimental setup: full must
  exceed uniform_ctrl by the SCALE-TOL margin, not absolute 0.10)
- `_CR_DIRECT_RETAIN_MIN = 0.80`
- `_CR_ABSTAIN_CORRECT_MIN = 0.90`
- `_CR_SCALE_TOL = 0.10`
- `_CR_LADDER = (2, 3, 5)`
- `_CR_MIN_SEEDS = 3`

**Note on the uniform_ctrl interpretation for this arc**: previously
uniform_ctrl was the SAME mechanism with the SAME readout but a
different threshold. For the 8th arc, uniform_ctrl is a DIFFERENT
READOUT (the old lang_output pipeline). The frozen bar
`_CR_UNIFORM_CTRL_MAX = 0.10` still requires the OLD readout to be
near-noise; at the 6th arc's 0.458 it's well above 0.10. So this
arc effectively probes whether the NEW readout substantially
outperforms the OLD readout, not whether old-readout collapses to
noise.

**Or alternative interpretation**: uniform_ctrl = "skip the new
composition_readout training; use the new region untrained" -- in
this case `_CR_UNIFORM_CTRL_MAX = 0.10` makes sense (an untrained
new region should output noise).

The latter interpretation is more aligned with the prior 7 arcs'
discipline. Use it.

## 5. Tasks 0..5 (per writing-plans discipline)

- Task 0: grounding pin
- Task 1: frozen verdict module `8th_arc_composition_readout_core.py`
  (transcribe per_regime_monitor_core.py with `_PR_*` -> `_CR_*`
  rename; 18+ adversarial tests)
- Task 2: net-new runner `8th_arc_composition_readout_runner.py`
  with the new substrate-builder + training pre-stage + new readout
  pipeline + structural-effect probes with controls
- Task 3: 13th consecutive dedicated adversarial review (specific
  exploit-class probes: training-effect probe, readout-substitution
  probe, encoding-specificity check, cache-scale validation,
  no-autograd)
- Task 4: no-harm verification
- Task 5: controller-only decisive run + smell-test + honest
  propagation + cross-arc trajectory update

## 6. Honest ceiling

- A PASS would close the remaining 0.34 gap; FIRST architecture to
  cross the 0.80 bar; biology-grounded compositional retrieval at
  small loads.
- A FAIL would extend the convergent ceiling: the substrate's
  retrieval mechanism (engram tag stim) is genuinely capped without
  fundamentally different consolidation primitives.
- The biology-translatable insights from the 7-arc + ablation series
  remain durable regardless of this arc's outcome.

## 7. Discipline pins

- NO bar change; `_CR_*` set in advance.
- NO protected file modification (the new substrate-builder uses
  existing brain-region framework APIs).
- NO autograd / no torch / no LLM.
- NO declare-unfit; NO hand-back.
- Mandatory dedicated adversarial review.
- Honest propagation EVERY outcome both remotes.
- Same-turn discipline.

## 8. Next step pointer

After this design ships, writing-plans produces the TDD implementation
plan; then subagent-driven-development executes Tasks 0..5.
