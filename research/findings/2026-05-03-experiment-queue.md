# Candidate experiments — post-batch queue

After H1 + H4 batches complete (~16:30 EDT), here are the next
experiments ranked by expected information gain. Pick one based on
H4's outcome (see `swr_decision.py`).

---

## If H4 isolation gives 80%+ (cascade interference confirmed)

### #1 — Reverse curriculum (H5)
Train language→motor pathway in isolation (H4-style) FIRST, then
freeze those weights and unfreeze the cascade for full training.
Tests whether we can keep H4's high W→A while gaining navigation.

Implementation:
* Add `--phase0-isolation N` flag to text_train_curriculum
* Phase 0: paired-stim training of language→motor only
* Phase 1+: standard, with `bridge.set_plasticity_gate("language_input_to_motor", 0.0)` to freeze
* Phase 3 (optional): unfreeze + SWR replay

Expected runtime: ~30 min × 6 seeds = 3 hours. Code: ~50 lines.

### #2 — Cascade-silenced eval
Add an alternative eval that DISABLES the cascade pathways at eval
time so motor_X readout reflects ONLY the direct PFC bypass. Lets us
measure each pathway's contribution separately.

Implementation:
* Add `evaluate_word_to_action_bypass_only(bridge)` that temporarily
  zeros the cortex_X→str_D1/D2_X pathway weights during eval, then
  restores after.

This isn't a new experiment — it's a measurement enhancement that
re-evaluates existing models. Cheap.

---

## If H4 isolation gives 50-79% (moderate, ambiguous)

### #1 — H1 balanced replay (already running — wait for result)
Will tell us whether the bias amplification is THE primary issue or
just one of several.

### #2 — Phase3 replays sweep
Run with `--phase3-replays = {50, 100, 200}` at seed 42 to find the
optimum dose. Tests whether smaller doses don't over-amplify.

Implementation: just CLI args. No code change.

Expected runtime: ~3 × 30 min × 1 seed = 1.5 hours.

### #3 — Reduced motor drive during replay
`--motor-replay-drive-pA 0` so replay drives only language_input +
language_output, not motor. Lets the existing motor-receptive
synapses learn from STDP but doesn't artificially boost motor pools.

Implementation: just CLI args (the kwarg already exists).

---

## If H4 isolation gives ~28% (architecture limit)

### #1 — Larger language regions
`--text-n-input-neurons 512 --text-n-output-neurons 512` gives 2x
discrimination capacity. With sparsity 0.1 fixed, more active neurons
per word (52 instead of 26) → less random overlap.

Implementation: just CLI args (kwargs exist).

Expected runtime: ~70 min × 6 seeds = 7 hours per condition.

### #2 — Larger motor pools
`--n-motor-per-action 50` gives 5x readout neurons. Population-mean
firing rate has lower variance → cleaner discrimination.

Implementation: just CLI args.

Expected runtime: similar to #1.

### #3 — Population-vector decoding
Instead of `argmax` over motor pool means, use cosine-similarity
between motor pool firing vector and a target reference vector.
Implementation requires adding decoding option to evaluate_word_to_action.

Expected runtime: just re-eval existing models (cheap).

### #4 — Different sparsity
`--sparsity 0.05` (12 active per word) or `0.20` (52 active). Very
sparse codes are cleaner but smaller signal; less sparse is more
robust but more overlap. Tunable.

Implementation: vocab_to_drive_pattern already takes sparsity, but
need to add a CLI flag to text_train_curriculum.

---

## Cross-cutting biology improvements (regardless of H4 outcome)

### Theta-gamma binding for word codes
Real Wernicke/Broca regions show theta-gamma cross-frequency coupling.
Add a 8 Hz theta rhythm that gates language_input firing windows;
within each window, gamma cycles bind specific cell assemblies.

Substantial implementation (~200 lines in bridge + sim_text_embeddings).
Low confidence in payoff but high biological fidelity.

### Multiple plasticity time scales
Current STDP is single-timescale. Real synapses have:
* Fast STDP (millisecond)
* Late-LTP (minutes — protein synthesis)
* Synaptic tagging (hours)

Implementing late-LTP would let consolidation-via-replay work as it
does in real brains (replay during sleep tags eligible synapses;
they then strengthen over time).

Substantial: bridge changes to track tagging + slow weight updates.
Likely high payoff but deep work.

### Recurrent connectivity in language regions
Currently `language_input` has `plastic_internal=False` (reservoir).
Real Broca's has tight cell-assembly recurrence (Pulvermüller's
"word webs"). Setting plastic_internal=True with appropriate weight
controls would let cell-assembly structure emerge through training.

Risk: runaway activity. Need sufficient inhibition + STDP balance.

---

## Frontend improvements (CPU-only, can run during GPU experiments)

### Real per-region firing rate in 3D viz (Phase 2 of viz design)
Bridge instrumentation to publish per-region rates via data_bus →
log → webapp parse → render REAL activity instead of synthesized.
Big change to bridge.py, ~50 lines.

### Run comparison side-by-side
Pick 2 runs in Runs tab → see Brain viz of both stepped together,
synchronized scrubbers. Useful for comparing baseline vs SWR for
same seed.

### Webapp performance
At 1000+ runs the Runs tab list rendering slows. Add virtualization
or pagination.

---

## My recommendation for first thing tomorrow

Run `swr_decision.py`. Pick the "winning branch" and pursue ONE
experiment from it. If H4 strongly favors a direction, take the
top-recommended path. If H4 is ambiguous, do the H3 phase3-replays
sweep first since it's cheap.

Don't try to test multiple architectural changes in parallel — they
all need GPU.
