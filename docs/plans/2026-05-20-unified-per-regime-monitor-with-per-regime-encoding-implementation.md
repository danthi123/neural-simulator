---
type: plan
status: live
date: 2026-05-20
---

# Unified per-regime monitor + per-regime encoding — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans, task
> by task. Owner standing instruction pre-selects same-session subagent-
> driven execution (one fresh subagent per task; failing-test ->
> minimal-impl -> run -> commit; controller trust-but-verify every
> diff). Task 3 (no-harm) and Task 4 (decisive run) are CONTROLLER-ONLY.
> The discipline's frozen-verdict module + both calibrated abstention
> moats are REUSED byte-unchanged -- no new core module needed.

**Goal:** Test whether the unified architecture (Phase-1 multi-event
direct training BEFORE compositional one-shot encoding + per-regime
routing through both calibrated moats) clears all four conjunctive
verdict bars simultaneously: `full >= 0.80`, `uniform_ctrl <= 0.10`,
`direct_retain >= 0.80`, `abstain_correct >= 0.90`. The previous
per-regime stage's nuanced FAIL prescribed this stage precisely: the
direct_retain bar collapsed because the runner used one-shot pair
encoding without the v14/v16-calibrated Phase-1 multi-event training
that 650 was calibrated against.

**Architecture:** 100% reuse-by-import. Concrete pieces:
- Validated v14/v16 Phase-1 W->A training via `run_concept_pool_demo` /
  `build_concept_bridge` / `apply_concept_topographic_bias` /
  `train_word_to_pool` (concept_pool_demo.py).
- HDF5 bridge checkpoints via `bridge.save_checkpoint` /
  `bridge.load_checkpoint` (byte-stable at same seed).
- Compositional one-shot encoding via `encode_concept_pair` +
  engram API (`start_engram_recording`, `commit_engram_tag`,
  `stimulate_tag`).
- Both calibrated moats: `abstention_gate.gate(., 650.0)` for direct;
  `abstention_gate_compositional.gate(., 5.6887...)` for compositional.
- Existing per-regime verdict module `per_regime_monitor_core.py` --
  same rung shape, same four conjunctive bars, byte-unchanged.

Net-new = one orchestration runner + one test file.

**Protected set (byte-unchanged across every commit; controller
verifies):** every `*_core.py` incl. `per_regime_monitor_core.py`;
`abstention_gate.py` + its test; `abstention_gate_compositional.py` +
its test; the previously-validated per-regime runner; every Stage-1
/ SPEAR / Pirazzini runner + tests; `text_minimal_isolation.py`;
`compose_concept_engram.py`; `concept_pool_demo.py`;
`consolidation_trainer.py`; `validate_trisynaptic_loop.py`;
`sim/bridge.py`; `sim/regions.py`; `sim/neuromodulators.py`;
`sim/train_checkpoint.py`; `sim/backend.py`; `sim/kernels.py`.

---

## Task 0: Grounding pin

**Files:** Create `tests/test_unified_per_regime_pin.py`.

```python
"""Grounding pin; intentionally RED until Task 1 lands the runner."""
import importlib

def test_unified_per_regime_runner_importable():
    m = importlib.import_module(
        "research.runners.unified_per_regime_monitor_runner"
    )
    assert hasattr(m, "run_unified_per_regime_monitor")
```

Run -> FAIL. Commit (`test: grounding pin for unified per-regime
monitor + encoding stage (red until Task 1)`). Verify protected set
byte-empty + push both remotes.

---

## Task 1: The unified runner (LOAD-BEARING; reuse-only orchestration)

**Files:** Create `research/runners/unified_per_regime_monitor_runner.py`;
Test `tests/test_unified_per_regime_monitor_runner.py`.

**Behavioural spec (orchestrates reused pieces; no new logic):**

`run_unified_per_regime_monitor(seeds, loads=(2,3,5), tiny_synth=False,
phase1_cache_dir="research/findings/raw/unified_per_regime/phase1/",
out_path=None, ckpt=None) -> dict`. CLI:
`--seeds`, `--loads`, `--tiny-synth`, `--phase1-cache-dir`, `--out`,
`--ckpt`.

Per seed (in order):
1. **Phase-1 training (cached):** if
   `{phase1_cache_dir}/seed{seed}.simstate.h5` exists, skip; else
   call `run_concept_pool_demo(seed=seed,
   n_train_events=200 (or 4 if tiny_synth),
   n_lang_input=2048 (or 256 if tiny_synth),
   n_per_pool=200 (or 24 if tiny_synth), n_fs_per_pool=24 (or 6),
   weak_dynamics=True, interleaved=True, topographic_factor=3.0,
   off_target_factor=0.3, enable_adjective=True,
   orthogonal_codes=True, sparsity=0.05,
   enable_direct_verb_to_motor=True,
   save_bridge=cache_path, ...)`. Verify the resulting checkpoint
   exists.

Per (seed, N) cell:
2. **Build a fresh bridge with the same recipe + load Phase-1
   checkpoint:** mirror the `compose_concept_engram.py` Phase-1-load
   pattern exactly.
3. **Compositional one-shot encoding:** generate N compositional
   facts deterministically from the seed (different from any
   calibration set via the per-regime-runner's `seed + 10000`
   convention); open the cross_pool_concept gate; call
   `encode_concept_pair(bridge, a, b, tag_name=f"ep_{i}", ...)` per
   pair; close the gate.
4. **Per-query routing:** generate a query set (direct queries from
   the trained vocab + compositional queries from the encoded
   pairs + an ungroundable control set); for each query:
   - direct -> `measure_pool_firing(bridge, word, all_pool_regions, ...)`
     -> ranked list (concept, rate, tag) -> `gate_direct(ranked, 650.0)`
     -> answer or abstain.
   - compositional -> the existing per-regime runner's
     compositional readout (raw firing-rate confidence at lang_output)
     -> `gate_compositional(ranked, COMPOSITIONAL_THRESHOLD)` -> answer
     or abstain.
5. **Three measurement arms (per cell):** full (per-regime routing),
   uniform_ctrl (both gates at 650), direct_retain (direct-only
   accuracy under per-regime).
6. **Emit rungs in the existing per_regime_monitor_core shape; call
   `per_regime_monitor_verdict(rungs)` unchanged.**

**Anti-cheat (carry forward all prior lessons):** OPAQUE tag names
(`f"ep_{i}"`); raw firing-rate moat input via the validated paths;
cross_pool_concept gate opened only during compositional encoding
then closed (per the validated `compose_concept_engram.py` pattern);
no torch/autograd; no protected-file edits; the `phase1_cache_dir`
contains the Phase-1-trained bridges (HDF5; byte-stable at same
seed; reusable across decisive runs).

**Step 1: Write failing test** -- tiny-synth runs end-to-end (Phase-1
training shrunk; compositional encoding one pair per rung);
verdict accepts well-formed rungs (may legitimately FAIL on toy);
no torch/autograd; opaque tags; both moats fed calibrated quantities;
the `phase1_cache_dir` exists post-run.
**Step 2: Run-to-fail.** **Step 3: Implement minimally** orchestrating
the reused pieces. **Step 4: Run-to-pass** + verify existing per-regime
+ Pirazzini + SPEAR + Stage-1 + moats suites still green (no
regression). **Step 5: Commit** (`feat: net-new unified per-regime
monitor + per-regime encoding runner (Phase-1 cached + compositional
one-shot; reuse-only; no autograd)`). Controller verifies protected
set byte-empty + push both remotes.

---

## Task 2: Dedicated adversarial review

Fresh adversarial reviewer (mirror Stage-1 / SPEAR / Pirazzini /
Per-regime reviews, each of which found real load-bearing defects).
Primary mandate:
- Phase-1 training is GENUINELY happening (not skipped/stubbed);
  the resulting checkpoint exists; loading it into a fresh bridge
  reproduces the v14/v16-calibrated direct-pool-firing-rate
  confidence on direct queries (independent probe).
- The cross_pool_concept gate is genuinely closed outside the
  compositional encoding window.
- Both moats receive their calibrated quantities (raw firing rate);
  the direct path uses `measure_pool_firing`; the compositional
  path uses the validated `lang_output_pattern_during_*` path.
- The decisive built-in control (uniform_ctrl: both gates at 650)
  is faithfully "full minus only threshold routing".
- direct_retain is genuinely the direct-query subset of the same
  full run; not a separate measurement.
- A degenerate scenario (e.g. Phase-1 training silently skipped;
  cached checkpoint loaded but plasticity gates left open; cross-
  pool gate left open after encoding; calibrated moat fed wrong
  quantity) provably CANNOT score PASS via runner + frozen verdict
  end-to-end.
- All `*_core.py` + `abstention_gate.py` + `abstention_gate_compositional.py`
  byte-unchanged; no autograd anywhere.

STRENGTHEN-only fixes to non-protected files only; `review:`
commit prefix; re-review loop until CLEAR.

---

## Task 3: No-harm phase (controller-only)

Comprehensive protected-set diff (pre-Task-0 base .. HEAD) MUST be
empty for every protected path including the existing per-regime
verdict module + both moats; full suite green across Unified + Per-
regime + Pirazzini + SPEAR + Stage-1 + moats; no autograd shipped.
Commit no-harm evidence; push both remotes.

---

## Task 4: CONTROLLER-ONLY decisive run

Controller, same turn, never stopping on a promise:
1. Grounding tiny-synth (toy numbers explicitly NOT propagated).
2. Phase-1 training at full biological scale per seed (cached; the
   most expensive step, ~17 min/seed per CLAUDE.md's v14/v16 timing;
   3 seeds = ~50 min). Verify Phase-1 cached checkpoints exist; do
   a sanity probe (load one + check `measure_pool_firing` produces
   v14/v16-style ~796 confidence on a known-trained direct word).
3. Decisive evaluation multi-seed run at the frozen ladder (2,3,5),
   CuPy on RTX 3090, DURABLE capture, monitored to ACTUAL completion
   via a genuine completion waiter.
4. Mandatory smell-test scrutinising a nominal PASS HARDER than a
   FAIL (recompute verdict from single recording; no re-run, no bar
   change; confirm full >= 0.80 AND uniform_ctrl <= 0.10 AND
   direct_retain >= 0.80 AND abstain_correct >= 0.90 all
   simultaneously cleared; on any anomaly raise the bar of
   skepticism; a fourth-architecture PASS especially must clear an
   extra-skeptical review given the prior triple-convergent
   ceiling).
5. Honest propagation of EVERY outcome (findings doc + capability
   pillar + state file + commit + push BOTH remotes).
6. Autonomous next staged step per outcome.

**Honest ceiling (never overstated):** a clean scrutinised success
= the unified architecture clears ALL FOUR conjunctive bars
simultaneously (per-regime separation works; direct retrieval
preserved by Phase-1 training; trustworthy property held). This
would be the FIRST clean scrutinised PASS in the project's
compositional-capability arc. Explicitly NOT fluent open-ended
language, NOT an LLM, NOT a threshold relaxation. The orienting
goal is artificial life with a proper brain analogue; biology-
translatable insights are the deliverable.
