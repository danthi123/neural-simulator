---
type: plan
status: live
date: 2026-05-22
---

# Vocabulary scaling -- implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development) to implement this plan
> task-by-task.

**Goal:** Test whether the biologized grounded-composition pipeline,
validated at a 16-concept vocabulary, still clears the frozen 0.80
compositional bar at a 64-concept vocabulary -- using the project's
validated G.20 sparse-distributed substrate.

**Architecture:** A new runner captures per-neuron concept-population
activity from a 64-concept G.20 sparse bridge, then runs the existing
biologized grounded-composition pipeline (longer-integration recognition
+ common-mode-removed grounded symbols + resonate-and-fire FHRR +
attractor clean-up) on that captured activity. The pipeline is reused
byte-unchanged where possible; the one genuine change is generalising
its concept taxonomy from the v14/v16 16-pool layout to an arbitrary
N-concept layout (the G.20 sparse vocabulary has no noun/verb/adjective
split). Cue and filler roles are assigned by a fixed partition of the
64 concepts, not by the v14/v16 pool-name prefixes.

**Tech stack:** numpy + the project's CuPy/numpy backend; the validated
G.20 sparse builder; the biologized FHRR/attractor modules. No
automatic differentiation.

**Design doc:** `docs/plans/2026-05-22-vocabulary-scaling-design.md`.

**Discipline:** reuse-by-import only; no protected/frozen/moat module
modified; the frozen 0.80 bar never tuned; controller verifies the
protected set is byte-empty in every commit-scoped diff; honest
propagation of every outcome to both remotes.

---

### Task 0: Grounding pin test

**Files:**
- Create: `tests/test_vocabulary_scaling_pin.py`

**Step 1: Write the pin test.** A test that imports the new runner
module (`research/findings/raw/vocabulary_scaling_run.py`) and asserts
its key constants exist and are sane: `N_CONCEPTS == 64`, `BAR == 0.80`,
`LOADS == [2, 3, 5]`. It will fail until Task 2 creates the module.

**Step 2: Run it, expect failure** (`ModuleNotFoundError`). This is the
intended Task-0 state -- it goes green only after Task 2.

**Step 3: Commit.**

---

### Task 1: 64-concept G.20 sparse bridge builder

**Files:**
- Create: `research/findings/raw/vocabulary_scaling_substrate.py`
- Test: `tests/test_vocabulary_scaling_substrate.py`

**Step 1: Write the failing test.** Assert that
`build_64_concept_sparse_bridge(seed)` returns a bridge plus a list of
64 distinct concept words, and that the bridge has the expected
per-concept sparse pool structure.

**Step 2: Run it, verify it fails.**

**Step 3: Implement.** A thin wrapper that reuses the validated G.20
sparse builder (`build_sparse_pool_bridge` / the `g20_multibridge
--sparse` path) byte-unchanged to construct one 64-concept sparse
bridge with a fixed 64-word vocabulary (reuse an existing G.20 vocab
spec; check `g20_vocab_spec_*` for a 64-concept list). Reuse-by-import;
do not modify the G.20 builder.

**Step 4: Run the test, verify it passes.**

**Step 5: Commit.** Controller verifies the protected set is byte-empty
in the commit-scoped diff.

---

### Task 2: Per-neuron activity capture + the pipeline run

**Files:**
- Create: `research/findings/raw/vocabulary_scaling_run.py`
- Test: `tests/test_vocabulary_scaling_run.py`

**Step 1: Write the failing test.** Assert the runner exposes
`capture_concept_activity(bridge, words, ...)` returning an
(M, n_neurons) array per word, and a `run_one_seed(seed, smoke=True)`
that on a tiny smoke vocabulary returns a result dict with
`per_load` integrated/composition-only accuracies.

**Step 2: Run it, verify it fails.**

**Step 3: Implement.** The runner: (a) builds the 64-concept sparse
bridge (Task 1); (b) captures M per-neuron concept-population activity
observations per concept by driving each concept word and recording
the concept-population firing-rate vector -- mirror `capture_activity`
in `activity_level_integration.py`; (c) generalises the
biologized grounded-composition pipeline to N concepts -- import the
pipeline's stages from `biologized_grounded_composition.py` and feed
the 64-concept captured activity, with cue/filler roles assigned by a
fixed partition of the 64 concepts (not pool-name prefixes); (d)
measures integrated + composition-only accuracy against the frozen
0.80 bar, loads {2,3,5}; (e) a `--smoke` mode (tiny vocab, few
observations -- toy numbers, NOT a result); plain ASCII output;
kill-safe per-seed activity cache.

**Step 4: Run the smoke, verify the test passes.** Task 0's pin test
goes green here.

**Step 5: Commit.** Controller verifies the protected set byte-empty.

---

### Task 3: Adversarial review (before the decisive run)

Dedicated independent reviewer (fresh agent) of the new runner: is the
G.20 sparse builder genuinely reused unchanged; is the activity capture
faithful; can a broken run score a PASS; is the frozen bar movable; any
autodiff. STRENGTHEN-only fixes. This precedes the decisive GPU run.

---

### Task 4 (CONTROLLER-ONLY): the decisive run

Not a subagent task. A grounding smoke run first (toy numbers not
reported), then the decisive multi-seed GPU substrate capture +
pipeline run (seeds 42/43/44), monitored to actual completion. Then the
mandatory smell-test (scrutinise a nominal PASS HARDER than a FAIL;
recompute from the recorded JSON; confirm recognition is reported
separately; no re-run, no bar change). Then honest propagation -- a
findings document, a capability_status entry, the schema test green,
commit and push to both remotes. Then continue autonomously per the
design doc's sequence.
