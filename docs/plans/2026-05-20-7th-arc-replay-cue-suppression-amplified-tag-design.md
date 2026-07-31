---
type: plan
status: live
date: 2026-05-20
---

# 7th arc: targeted cue-suppression-during-REPLAY + amplified engram-tag stim + persistent PFC-frame (design)

> **For Claude / autonomous continuation:** This is the **design** for
> the 7th architecture, empirically motivated by the cross-arc
> trajectory analysis (commit `9693685`) showing 35% gap-closure across
> the 6-arc series. After approval, writing-plans produces the TDD
> implementation plan, then subagent-driven-development builds Task 0..5
> with adversarial review and controller-only decisive run.

## Status

Pre-registered NEW design grounded in three established facts:

1. **Cross-arc trajectory analysis (commit `9693685`)** shows
   progressive improvement at N=3 across the three decisively-run arcs:
   Unified 0.274 -> Theta-gamma 0.280 -> 6th arc 0.458. The 6th arc
   closed the gap to 0.80 by 35%. The trajectory is real; the
   "convergent ceiling" framing was too pessimistic.

2. **Theta-gamma finding (commit `1bbc165`)**: cue-suppression-during-
   RETRIEVE is biologically backwards (violates Tulving 1973 encoding-
   specificity). BUT cue-suppression-during-REPLAY is fundamentally
   different: the replay phase aims to consolidate the engram tag's
   selective bound-adj drive; the cue's contribution there is
   contamination of the consolidation signal, not encoding-context.

3. **Localisation finding (commit `110f7cd`)**: the substrate emits
   strong-but-wrong top words because the cued-noun's diffuse
   lang_input drive dominates the engram tag's selective bound-adj
   drive. Amplifying the engram-tag stim during retrieve directly
   addresses this; persistent PFC-frame priming reinforces the
   compositional structure.

## 1. The capability under test (falsifiable; same frozen bars)

Compositional retrieval with the augmenting mechanisms targeted by
empirical evidence. Frozen bars identical to all prior arcs (module-
local `_TC_*` constants distinct from `_GR_*/_TG_*/_PR_*`):
`_TC_FULL_MIN=0.80, _TC_UNIFORM_CTRL_MAX=0.10, _TC_DIRECT_RETAIN_MIN=
0.80, _TC_ABSTAIN_CORRECT_MIN=0.90, _TC_SCALE_TOL=0.10,
_TC_LADDER=(2,3,5), _TC_MIN_SEEDS=3`. (TC = Targeted Cue suppression.)

Experimental contrast:
- **FULL arm**: encode + replay with cue-SUPPRESSED + AMPLIFIED engram-
  tag stim during retrieve + PERSISTENT PFC-frame priming (50-step
  stim instead of 10; NMDA bistability holds across full retrieve
  window) + cue PRESENT during retrieve
- **UNIFORM_CTRL arm**: encode + replay with cue PRESENT during replay
  + baseline 1x tag stim during retrieve + brief 10-step PFC-frame +
  cue PRESENT during retrieve (cue parity preserved; encoding-
  specificity respected in both arms; only the augmenting
  modifications differ)

If the targeted mechanisms continue the 35% gap-closure trajectory,
the 7th arc could reach ~0.60-0.65 full_acc at N=3. If it reaches
>= 0.80, the bar is met. If it plateaus, the substrate's retrieval
mechanism is asymptotically capped.

## 2. The mechanisms being added (load-bearing; grounded in prior findings)

### Mechanism 1: Cue-suppression DURING REPLAY (not retrieve)

The 6th arc used `run_concept_replay_phase` which stimulates the
engram tag selectively but DOES NOT suppress the cue. If the cue's
lang_input pathway is active during replay, the replay-induced STDP
strengthens BOTH the engram tag's selective bound-adj drive AND the
cue's diffuse contamination. By suppressing the cue ONLY during the
replay window:
- Replay strengthens ONLY the engram tag's selective pathway
- The cue's contamination is NOT consolidated
- The retrieve phase sees the cue (encoding-specificity respected)
  but the substrate has been replay-strengthened to favor the bound-
  adj over the contamination

Implementation: in the runner's compositional eval, wrap the
`run_concept_replay_phase` call with cue-suppression on the cortico-
hippocampal input pathway (lang_to_ec gate) via the existing
plasticity_gate or external-input mechanism. This is net-new wiring
in the 7th arc runner; no protected modification.

### Mechanism 2: Amplified engram-tag stim during retrieve

The 6th arc used the default tag drive_pA. Per the localisation
finding, the cue's diffuse drive dominates the tag's selective drive.
Increasing the tag drive_pA from baseline 1500 -> 3000 or 5000 amplifies
the bound-adj selective signal. This requires either:
- A new wrapper around `_compositional_query_ranked` that accepts an
  amplified drive_pA kwarg (net-new code in the runner; the helper
  itself stays byte-unchanged)
- OR a copy of the helper with the amplitude as a parameter

The disciplined route: write a local wrapper in the 7th arc runner.

### Mechanism 3: Persistent PFC-frame priming

The 6th arc used PFC_FRAME_STIM_STEPS=10 (brief drive). NMDA
bistability should hold the frame for longer, but 10 steps may be too
brief to actually trigger the bistability. Extending to 50 steps gives
the NMDA dynamics more time to lock into the attractor; the frame
then holds across the full retrieve window without further driving.

### Mechanism 4: Higher n_replays_per_tag (consolidation strength)

The 6th arc used n_replays_per_tag=20. Increasing to 50 strengthens
the consolidation signal per tag. Combined with cue-suppression
during replay (mechanism 1), this targets the consolidation directly
at the engram tag without cue contamination.

## 3. Inventory of reused subsystems (byte-unchanged)

- `build_biological_brain_regions(...)` -- same unified substrate
- `encode_concept_pair` -- encoding
- `run_concept_replay_phase` -- replay (mechanism 1 wraps it with
  cue-suppression; doesn't modify it)
- `_compositional_query_ranked` -- retrieve readout (mechanism 2
  builds a local wrapper for amplified-tag-stim version)
- `_seed_query_rng` / `_restore_query_rng` -- RNG isolation pattern
  from theta-gamma `e6b17da`
- Cache-scale validation pattern from `13f73e8`
- 4 calibrated abstention moats byte-unchanged
- `dlpfc_verb` region + NMDA subsystem byte-unchanged

The genuine net-new code: ~700-1000 line runner that adds the 4
targeted mechanisms to the eval loop + a frozen verdict module
(transcription with rename `_PR_*` -> `_TC_*`).

## 4. Pre-registered next staged step

- Task 0: grounding pin (RED until Tasks 1 + 2)
- Task 1: frozen verdict module
  `research/runners/targeted_cue_suppression_replay_core.py` with
  `_TC_*` constants; 18+ adversarial test cases.
- Task 2: net-new runner
  `research/runners/targeted_cue_suppression_replay_runner.py`
  (~700-1000 lines mirroring 6th arc structure; ADD cue-suppression-
  during-replay wrapper; ADD amplified-tag-stim wrapper; EXTEND PFC-
  frame stim window; INCREASE n_replays_per_tag).
- Task 3: 12th consecutive dedicated adversarial review.
- Task 4: no-harm verification.
- Task 5: controller-only decisive run + smell-test + honest
  propagation.

## 5. Honest ceiling (binding throughout)

- A PASS would be the FIRST architecture in the 7-arc series to clear
  the frozen bars. Biology-grounded compositional retrieval at small
  loads; NOT yet fluent open-ended language.
- A partial result (e.g., full_acc ~ 0.60 at N=3, gap closed further
  but not to 0.80) would extend the trajectory analysis: each arc
  closes ~35% of the remaining gap; this is a quantitative biology-
  translatable signature (consistent with progressive consolidation
  in real CLS theory).
- A FAIL with no further gap-closure (e.g., 0.46 again) would extend
  the convergent ceiling to 7 architectures with the trajectory
  flattening; the substrate's retrieval mechanism is asymptotically
  capped and deeper refinement is required.

## 6. Discipline pins (mirrors prior 6 arcs)

- NO bar change; `_TC_*` constants set in advance, NEVER tuned.
- NO protected file modification; protected set byte-empty diff vs
  `e8a99a2` holds.
- NO autograd / no torch / no LLM call.
- NO declare-unfit; NO hand-back; NO config-crank.
- Mandatory dedicated adversarial review BEFORE no-harm BEFORE
  decisive run.
- Honest propagation EVERY outcome both remotes.
- Same-turn autonomous next-action discipline.
- 4 substrate-and-protocol-specific calibrated moats byte-stable.
- No-confab moat 7/7 byte-identical.

## 7. Next-step pointer

After approval (committed), writing-plans produces the TDD
implementation plan
(`docs/plans/2026-05-20-7th-arc-replay-cue-suppression-amplified-tag-implementation.md`).
Then subagent-driven-development executes Tasks 0..5.
