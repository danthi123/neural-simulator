# Unified per-regime monitor + per-regime encoding architecture: the design the previous stage's nuanced FAIL prescribes

**Status:** Design (autonomous; no hand-back). Supersedes the per-regime
metacognitive-monitor stage, which closed as an honest FAIL **with the
FIRST mechanistically-validated per-regime separation in the project**
(uniform_ctrl=0 vs full>0 across all 9 cells; seed 43 N=2 hit 25%) but
which the verdict module correctly failed because `direct_retain_acc =
0.0` could not clear the v14/v16-multi-event-calibrated 650 direct gate
-- because the runner used one-shot pair encoding (same as Stage-1 /
SPEAR / Pirazzini), not the v14/v16-validated multi-event Phase-1 W->A
training that 650 was calibrated against.

**Date:** 2026-05-20

**Plain-language commitment:** ordinary scientific terms, each defined
once; no internal codenames or letter-number labels are load-bearing;
catalog identifiers in parentheses for traceability.

---

## 1. Why this direction, and what the previous stage's nuanced FAIL taught

Four stages now: Stage-1 static, SPEAR rhythm-multiplexed synaptic_gain,
Pirazzini disinhibition + Hasselmo ACh, and Per-regime metacognitive
monitor. The TRIPLE convergent ceiling (first three) empirically
localised the *threshold* as the rate-limiter. The per-regime monitor
stage proved the *threshold separation works* (the first time any
compositional architecture in the project produced any non-zero
full-system accuracy above the abstention baseline; uniform_ctrl=0
control collapsed correctly across all cells). But the architecture
still FAILed because direct queries collapsed against the
v14/v16-calibrated 650 -- the runner's encoding regime (one-shot pair
engrams) is fundamentally different from v14/v16's multi-event direct
training, so 650 doesn't apply to its direct queries.

**The biology-translatable insight (the dual localisation):** per-regime
metacognitive monitors are NECESSARY but NOT SUFFICIENT -- they also
require regime-appropriate ENCODING. This matches
complementary-learning-systems theory exactly: cortical multi-event
schema learning for direct concepts (the v14/v16 88.75% multi-seed
recipe); hippocampal one-shot relational binding for compositional
content (the engram API path).

This stage builds the unified architecture: **Phase-1 multi-event W->A
training BEFORE the compositional one-shot encoding**, with the existing
650 direct gate + the calibrated 5.69 compositional gate both byte-
unchanged, the per-regime routing as in the previous stage.

## 2. The reframe (explicit)

Unit of analysis: a **single bridge** that hosts BOTH (a) v14/v16-style
multi-event-trained direct concept binding (calibrated to clear the 650
direct moat) AND (b) hippocampal one-shot compositional engram tags
(calibrated to clear the 5.69 compositional moat). Per-query-type
routing as before. The same four conjunctive verdict bars (full >=
0.80, uniform_ctrl <= 0.10, direct_retain >= 0.80, abstain_correct >=
0.90) -- the existing `per_regime_monitor_core` verdict module REUSED
byte-unchanged.

There is no necessity partition; no new frozen verdict module needed;
no new abstention gate needed. Both calibrated moats stay frozen as
they are. The only net-new is the orchestrating runner.

## 3. Inventory of validated subsystems to reuse byte-unchanged (all 100% reuse)

Reuse-by-import only; no edits to any protected/frozen/validated module
or the moats:

- **Phase-1 multi-event W->A training (validated 88.75% multi-seed):**
  `research/runners/concept_pool_demo.py`:
  - `build_concept_bridge(seed, n_lang_input=2048, n_per_pool=200,
    n_fs_per_pool=24, enable_adjective=True, weak_dynamics=True,
    enable_direct_verb_to_motor=True, ...)` (:76) -- builds the
    untrained v16 substrate.
  - `apply_concept_topographic_bias(bridge, n_lang_input=2048,
    topographic_factor=3.0, off_target_factor=0.3, sparsity=0.05,
    apply_reciprocal=True, orthogonal_codes=True, ...)` (:298) -- the
    Pulvermueller-style topographic prior (v14 fix).
  - `train_word_to_pool(bridge, word, target_pool_region,
    n_events=200, ...)` (:583) -- per-word multi-event Phase-1
    binding training (200 events, STDP-driven; opens target-kind
    gates, freezes others).
  - `run_concept_pool_demo(seed, n_train_events=200, n_lang_input=2048,
    n_per_pool=200, n_fs_per_pool=24, weak_dynamics=True,
    interleaved=True, topographic_factor=3.0, off_target_factor=0.3,
    enable_adjective=True, orthogonal_codes=True, sparsity=0.05,
    enable_direct_verb_to_motor=True, save_bridge=<h5>, ...)` (:808)
    -- the full pipeline; the validated recipe is the kwargs above.
- **Bridge state persistence (HDF5; byte-stable at same seed):**
  `SimulationBridge.save_checkpoint(filepath)` / `.load_checkpoint(filepath)`
  (sim/bridge.py:5917 / :6092). The Phase-1-trained bridge is saved
  once per seed and loaded into each evaluation run -- expensive
  training amortised across the decisive run + any future stages.
- **Direct-retrieval W->A readout (validated):** `measure_pool_firing(bridge, word,
  all_pool_regions, stim_steps=100, reset_steps=50, drive_pA=200.0,
  sparsity=0.05, n_lang_input=2048, orthogonal_codes=True,
  n_words_for_orthogonal=16, word_to_idx=<dict>)` (concept_pool_demo.py:744)
  -- drives `language_input(word)` at the trained-substrate config,
  reads per-pool firing rates. The raw firing-rate confidence on the
  word's correct pool IS the quantity the 650 direct gate is
  calibrated on (encoded ~796 vs control ~584).
- **Compositional one-shot encoding (validated):** `encode_concept_pair(
  bridge, concept_a, concept_b, tag_name, ...)` (compose_concept_engram.py:101)
  + engram API on the bridge (`start_engram_recording`, `commit_engram_tag`).
- **Compositional readout + 5.69 gate:** the per-regime monitor stage's
  `_compositional_query_confidence` pattern (per_regime_monitor_runner.py)
  -- raw firing-rate confidence at `lang_output` via
  `lang_output_pattern_during_*` + the calibrated 5.69 gate.
- **Per-regime routing controller:** the existing per-regime runner
  pattern (the FAILed stage's runner) -- byte-unchanged reuse of
  the routing logic; only the encoding pre-stage is new.
- **Capability-verdict module (REUSED byte-unchanged):**
  `research/runners/per_regime_monitor_core.py` with bars
  `_PR_FULL_MIN=0.80, _PR_UNIFORM_CTRL_MAX=0.10, _PR_DIRECT_RETAIN_MIN=0.80,
  _PR_ABSTAIN_CORRECT_MIN=0.90, _PR_LADDER=(2,3,5), _PR_MIN_SEEDS=3`.
  The same conjunctive bars; the same VOID/FAIL/PASS semantics; the
  same rung-shape contract. No new frozen module.
- **Both abstention moats (byte-unchanged):** `abstention_gate.py`
  (`DEFAULT_THRESHOLD = 650.0`, calibrated on v14/v16 multi-event;
  7/7 tests) + `abstention_gate_compositional.py`
  (`COMPOSITIONAL_THRESHOLD = 5.688725490196079`, calibrated on
  full-scale held-out compositional readout; 7/7 tests).
- **Existing chaining pattern (validated):**
  `compose_concept_engram.py` already chains `bridge.load_checkpoint(
  <phase-1-trained.h5>)` + `encode_concept_pair(...)` -- the canonical
  pattern this new runner mirrors exactly. See its commit history for
  the validated recipe.

## 4. What is genuinely net-new (bounded precisely)

ONE new runner file + ONE test file. The runner orchestrates:

1. **Phase-1 multi-event training (once per seed; cached as a
   checkpoint):** if a Phase-1 checkpoint for the seed exists at the
   expected path, load it; otherwise call the validated
   `run_concept_pool_demo` with the v14/v16 recipe + save_bridge to
   the expected path. The resulting bridge has all direct concepts
   trained to v14/v16-calibrated ~796 confidence (the basis for 650).
2. **Compositional one-shot encoding (per rung):** load the cached
   Phase-1 checkpoint into a fresh bridge per (seed, N), open the
   cross_pool_concept gate (per the `compose_concept_engram.py`
   pattern), encode N compositional (noun, adj) pairs via the engram
   API, close the gate.
3. **Per-query-type routing (per query):** direct queries through
   `measure_pool_firing` -> raw rate -> 650 direct gate;
   compositional queries through the existing compositional readout
   path -> raw rate -> 5.69 compositional gate.
4. **Three measurement arms per (seed, N):** full (per-regime
   routing); uniform_ctrl (both gates at 650; the decisive built-in
   control); direct_retain (direct-only accuracy under per-regime,
   subset of the `full` run).
5. **Emit rungs in the existing `per_regime_monitor_core` shape;
   call the existing verdict module unchanged.**

No new verdict module; no new gate module; no new encoding mechanism;
no new substrate construction. The entirety of the net-new is the
runner's orchestration logic + a smell-test test file. Estimated
runner size: ~400-600 lines (smaller than the prior runners because
so much is reused via imports).

## 5. Three concrete architectures, honest ceilings, falsify-cheaply-first

- **A -- minimal unified (RECOMMENDED first; cheaply de-riskable).**
  Exactly the architecture above. Validated v14/v16 Phase-1 training
  + validated one-shot compositional encoding + per-regime routing
  through both calibrated moats. Tests whether the dual-encoding
  hypothesis the previous stage's nuanced FAIL prescribed actually
  lifts direct_retain above 0.80 while compositional clears 5.69.
- **B -- A + per-regime calibration verification.** Add a smell-test
  routine that re-runs the calibration on the Phase-1-trained
  substrate (since the substrate's noise distribution may differ
  with a trained vs untrained substrate); verify the calibrated
  thresholds match the committed constants (MATCH status). Staged
  only if A passes.
- **C -- B + the Tse 2007 schema-acceleration variant.** Add a
  small held-out compositional set whose facts are CONSISTENT with
  the Phase-1-trained schema (e.g. "apple is RED" where the schema
  already encodes apple); compare to a control of inconsistent
  facts. Tests whether schema-consistency accelerates one-shot
  compositional binding above the 5.69 threshold (Tse 2007;
  Sommer 2022). Staged only if B passes.

**Recommendation:** build A first under the pre-registered REUSED
fixed-bar verdict. B and C are staged follow-ons.

## 6. Pre-registered gate, falsify-cheaply-first, anti-cheat

- **REUSED frozen capability-verdict module**
  `research/runners/per_regime_monitor_core.py` -- byte-unchanged
  since its Task-1 creation `c1626e0`. Bars verbatim:
  `_PR_FULL_MIN=0.80, _PR_UNIFORM_CTRL_MAX=0.10,
  _PR_DIRECT_RETAIN_MIN=0.80, _PR_ABSTAIN_CORRECT_MIN=0.90,
  _PR_SCALE_TOL=0.10, _PR_LADDER=(2,3,5), _PR_MIN_SEEDS=3`. No new
  bars.
- **Decisive built-in control:** uniform_ctrl_acc <= 0.10 (same as
  the previous stage; the per-regime separation must be the
  measurable differentiator regardless of encoding regime).
- **Falsify-cheaply-first:** a fast tiny-synth smoke confirms the
  pipeline structurally works (Phase-1 training shrunk to a few
  events; compositional encoding shrunk to one pair); toy numbers
  explicitly NOT a result.
- **Anti-cheat (non-negotiable; carry forward all prior lessons):**
  OPAQUE tag names; raw firing-rate moat input (not synthetic
  arithmetic); per-query-type routing in the runner's source; the
  cross_pool_concept gate is opened ONLY during compositional
  encoding then closed (per `compose_concept_engram.py` pattern, not
  always-on); dedicated adversarial review before no-harm (primary
  mandate: is the Phase-1 training genuinely happening and producing
  the v14/v16-calibrated ~796 direct-pool-firing-rate confidence;
  is the cross_pool_concept gate genuinely closed outside the
  encoding window; can a degenerate / over-permissive scenario score
  PASS via runner + verdict end-to-end; are both moats fed their
  calibrated quantities; are subsystems byte-unchanged); mandatory
  smell-test scrutinising a nominal PASS HARDER than a FAIL.

## 7. Honest ceiling (stated up front, never spun)

A clean scrutinised success = the unified architecture clears ALL
FOUR conjunctive bars simultaneously: full >= 0.80 (the per-regime
architecture produces correct answers above the regime-appropriate
threshold for both query types); uniform_ctrl <= 0.10 (per-regime
separation is the differentiator; a single 650 threshold applied
uniformly collapses); direct_retain >= 0.80 (Phase-1 multi-event
training restores direct retrieval to v14/v16-calibrated levels);
abstain_correct >= 0.90 (trustworthy property holds). This would
be the FIRST clean scrutinised PASS in the project's compositional-
capability arc.

Explicitly NOT fluent open-ended language, NOT an LLM, NOT a
threshold relaxation. The genuine durable contribution of this
stage, regardless of outcome, is a faithful, adversarially-hardened,
fixed-bar test of whether the dual-encoding + dual-monitor
architecture the previous stage's nuanced FAIL prescribed actually
yields the capability that the threshold-only per-regime monitor
could not.

## 8. Components / data flow / error handling / testing (for the plan)

- **Components:** validated `concept_pool_demo` Phase-1 training
  (reused); HDF5 bridge checkpoints (reused); validated
  `encode_concept_pair` compositional encoding (reused);
  per-query-type routing controller (NEW; ~one orchestration
  function); the existing per-regime verdict module (reused);
  both calibrated moats (reused, byte-unchanged); kill-safe
  checkpoint pattern (reused).
- **Data flow:** per seed: Phase-1-train-or-load -> save -> per
  rung: load -> compositional one-shot encode (gate-opened) ->
  per query: route by type -> readout via validated path -> gate
  via appropriate moat -> answer-or-abstain; per cell: aggregate
  full / uniform_ctrl / direct_retain / abstain_correct;
  aggregate to rungs; call verdict.
- **Error handling:** instrument-validity FIRST (the reused
  verdict module already enforces this); kill-safe/resumable via
  the reused checkpoint module.
- **Testing:** runner-tests pin: tiny-synth smoke runs end-to-end
  (Phase-1 training is shrunk to a few events; compositional
  encoding is one pair per rung); the runner produces well-formed
  rungs the verdict accepts; no torch/autograd; opaque tags; both
  moats fed calibrated quantities; uniform_ctrl differs from full
  ONLY in the threshold-routing decision (no other plumbing
  difference). The dedicated adversarial review independently
  re-verifies all of the above + that the cross_pool_concept gate
  is genuinely closed outside the encoding window.

---

**Next:** writing-plans for this design (Task 0 pin; Task 1 the
net-new unified runner reusing concept_pool_demo Phase-1 +
compose_concept_engram pattern + existing per-regime verdict; Task 2
dedicated adversarial review; Task 3 no-harm; Task 4 controller-only
decisive run + smell-test + honest propagation), then subagent-
driven-development under the REUSED frozen verdict module's fixed
bars, honest propagation of every outcome to both remotes,
iterating following the biology -- autonomous, no hand-back.
