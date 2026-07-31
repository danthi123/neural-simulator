---
type: plan
status: live
date: 2026-05-23
---

# Biologized spiking theta-gamma mode-unification: TDD implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task (the owner's standing instruction pre-selects same-session execution; controller-built focused runner is acceptable for this scale per the trained-substrate-runner precedent, with the dedicated adversarial review providing the load-bearing safety gate).

**Goal:** Build the biologized spiking implementation of theta-gamma mode-unification on the project's substrate, per `docs/plans/2026-05-23-biologized-spiking-mode-unification-design.md`. PASS iff BOTH order-bearing AND order-invariant readouts multi-seed-mean ≥ 0.80 at every compositional load {2, 3, 5} on one trained bridge (bridgeA_nouns, 32 concepts). Frozen 0.80 bar; K=16 PASS recipe; multi-seed (42, 43, 44).

**Architecture:** A focused single-runner extension. Reuses the trained-substrate runner's `train_substrate`, the vocab-scaling pipeline's `capture_concept_activity`, the FHRR-biologization arc's `ResonateFireFHRR` + `ResonateFireTPAM` + `make_deriver` + `phases_to_spikes`, and Task 1 of the 160-ensemble arc's `bridge_vocab_and_patterns` helper — all byte-unchanged. Genuinely-new code: (i) `gamma_slot_positions(seed, n_slots, n_dim)` helper that returns 7 deterministic spiking-phasor position symbols; (ii) sequence encoding via `ResonateFireFHRR.encode([(grounded_symbol, position) for k])`; (iii) two readout decoders — per-slot attractor settle for order-bearing, per-slot unbind + marginal similarity-sum for order-invariant; (iv) the orchestration loop.

**Tech Stack:** Python + numpy + CuPy (GPU for substrate train + capture; CPU for the FHRR pipeline). Reuse-by-import only. No autograd. Frozen 0.80 bar unchanged.

---

### Task 0: Grounding pin

**Files:**
- Create: `tests/test_biologized_spiking_mode_unification_pin.py`

**Step 1: Write the failing test**

Pin the frozen 0.80 bar, the K=16 PASS recipe constants imported unchanged, the gamma-slot count (7, the Lisman-Idiart biologically grounded value), per-bridge concept count (32, matching the 160-ensemble per-bridge size), multi-seed grid {42, 43, 44}, loads {2, 3, 5}, FHRR phasor dim 512, and the runner module's public surface (red until Task 2 lands).

**Step 2: Run + commit**

`python -m pytest tests/test_biologized_spiking_mode_unification_pin.py -q`
Expected: 4 constant pins PASS, runner-module-exists FAILS (intentional).

```bash
git add tests/test_biologized_spiking_mode_unification_pin.py
git commit -m "mode-unif Task 0: grounding pin (red until Task 2 -- intentional)"
```

---

### Task 1: `gamma_slot_positions` helper

**Files:**
- Create: `research/findings/raw/biologized_spiking_mode_unification_helpers.py`
- Create: `tests/test_biologized_spiking_mode_unification_helpers.py`

**Step 1: Write the failing tests**

5 unit tests pinning: return list of length `n_slots`; each position is a numpy ndarray of length `n_dim` (the spike-phase representation); deterministic in seed (same seed → byte-identical positions); per-seed independence (different seeds → different positions); pairwise overlap near-orthogonal (mean abs phase-similarity below a small threshold across the 21 pairs of the 7 positions).

**Step 2: Run to verify they fail**

Expected: ModuleNotFoundError.

**Step 3: Write the helper**

```python
"""Pure helper for the biologized spiking mode-unification arc.

Returns N_GAMMA_SLOTS=7 deterministic per-seed spike-phase position
symbols, generated via the same FHRR primitive (random uniform phases
quantised to spike times) the validated SpikingPhasorFHRR + ResonateFireFHRR
use for vocabulary symbols. The positions represent the 7 gamma slots
within one theta cycle (Lisman-Idiart 1995).
"""
from __future__ import annotations
import numpy as np
from typing import List
from research.runners.spiking_phasor_fhrr import phases_to_spikes


def gamma_slot_positions(seed: int, n_slots: int, n_dim: int) -> List[np.ndarray]:
    """Return n_slots deterministic per-seed spike-phase position
    symbols, each of dimension n_dim. The positions are independently-
    seeded random uniform phases quantised to spike times via
    phases_to_spikes (the same mechanism the FHRR pipeline uses to
    construct any symbol)."""
    rng = np.random.default_rng(seed)
    return [phases_to_spikes(rng.uniform(0.0, 1.0, size=n_dim))
            for _ in range(int(n_slots))]
```

**Step 4: Run tests + commit**

```bash
git add research/findings/raw/biologized_spiking_mode_unification_helpers.py tests/test_biologized_spiking_mode_unification_helpers.py
git commit -m "mode-unif Task 1: gamma_slot_positions helper (deterministic per-seed, 5/5 unit tests)"
```

---

### Task 2: The runner

**Files:**
- Create: `research/findings/raw/biologized_spiking_mode_unification_runner.py`

**Step 1: Write the runner**

Public surface that the Task 0 pin checks for:
- `gamma_slot_positions` (re-exported)
- `run_one_seed(seed, smoke=False)` -> dict
- `main()`

Per (seed): build + train the bridge via the validated trained-substrate path (1 bridge = bridgeA_nouns at 32 concepts; reuses `train_substrate`); capture activity at M_OBS=16; derive grounded symbols (mean-centred consolidated activity → deriver → `phases_to_spikes`); build 7 deterministic gamma-slot position symbols; per trial sample an ordered K-tuple from the 32 vocab items; encode `C = ResonateFireFHRR.encode([(grounded[item_k], position_k) for k])`; run BOTH readouts on the SAME C:

- ORDER-BEARING: for each slot k, `unbound_k = net.query(C, position_k)`; `z, _ = tpam.settle_annealed(unbound_k, ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS, fast=True)`; recovered item = `argmax(abs(tpam.s.conj().T @ z))`. Compare K-tuple exactly.

- ORDER-INVARIANT: for each slot k, `unbound_k = net.query(C, position_k)` (cached, REUSING the same unbinds order-bearing computed — same C, same positions); for each vocab item w, `score[w] += phase_similarity(unbound_k, grounded[w])`. Top-K items by score, sorted by index. Compare set.

Multi-seed aggregate per load; verdict PASS iff BOTH readouts multi-seed-mean ≥ 0.80 at every load.

`--smoke` runs a tiny bridge + smaller vocab + few trials; toy numbers NOT propagated.

**Step 2: Run the Task 0 pin to verify it goes green**

`python -m pytest tests/test_biologized_spiking_mode_unification_pin.py -q`
Expected: 5/5 PASS (4 constant pins + module-exists).

**Step 3: Commit**

```bash
git add research/findings/raw/biologized_spiking_mode_unification_runner.py
git commit -m "mode-unif Task 2: runner (focused byte-reuse; gamma-slot encoding + 2 readouts on SAME C)"
```

---

### Task 3: Soundness tests

**Files:**
- Create: `tests/test_biologized_spiking_mode_unification.py`

**Step 1: Write the failing tests**

3-4 soundness tests pinning load-bearing structural properties:
- Gamma-slot positions are FIXED per seed (call helper twice, byte-identical).
- BOTH readouts share the SAME encoded C (the runner computes C once per trial; cannot bypass).
- The deriver seed pinned (DERIV_SEED = 90909 matching the FHRR-biologization arc).
- The frozen bar imported unchanged (BAR = 0.80).

**Step 2: Run + commit**

---

### Task 4: Dedicated adversarial review (BEFORE Task 5)

Dispatch fresh `general-purpose` agent with full tool access. Adversarial checks (per the design's Soundness Considerations):

1. No answer leak in DECODING: the true sequence is used in encoding (necessarily) but the decoders use only readout outputs.
2. The biologized pipeline is reused byte-unchanged (git diff confirms zero modifications to `resonate_fire_fhrr.py`, `spiking_phasor_fhrr.py`, `vocabulary_scaling_run_trained.py`, `vocabulary_scaling_run.py`, `pattern_separation_grounding_probe.py`, any `sim/*`, the no-confab moat).
3. Gamma-slot positions are FIXED per seed across all trials (no per-trial regeneration).
4. The substrate is genuinely trained (uses `train_substrate` from the trained-substrate runner, the K=16 PASS recipe).
5. Both readouts on the SAME C (verify the runner computes C exactly once per trial and both readouts receive it).
6. The frozen 0.80 bar is unchanged.
7. No autograd.
8. The per-slot order-bearing decoder genuinely uses `unbind(C, position_k)` alone as input (no oracle access to the true item at slot k).
9. The marginal-sum order-invariant decoder genuinely scores ALL 32 vocab items (no oracle restriction to the true items).
10. Capacity envelope respected: K ≤ 7 (gamma-slot ceiling); vocab = 32 (well inside the 256 ceiling); no artificial noise added.

Verdict CLEAR required before Task 5.

---

### Task 5: Controller-only decisive GPU run (NOT a subagent task)

**This is the controller's responsibility.**

**Step 1: Smoke (single-seed, reduced scale):**

`python research/findings/raw/biologized_spiking_mode_unification_runner.py --smoke`

Confirms end-to-end on the toy configuration.

**Step 2: Decisive run (harness-tracked background):**

`python research/findings/raw/biologized_spiking_mode_unification_runner.py > log 2>&1` with `run_in_background=true`. Expected ~2 hours GPU (substrate build + train + capture on 1 bridge × 3 seeds × ~35 min each, plus a few minutes CPU for pipeline composition).

**Step 3: When harness notifies completion:**

(a) Mandatory anti-cheat smell-test: recompute per-load means from per-seed independently; recompute captured pool density from the activity cache (must sit in the 32-concept-bridge regime ~0.04-0.06); per-seed variation check.

(b) Pre-registered reading: PASS iff BOTH readouts multi-seed-mean ≥ 0.80 at every load; NEGATIVE_* variants otherwise.

(c) Write findings doc (per-readout-per-load breakdown; oracle-adjacency caveats from the design preserved); update `webapp/capability_status.json` (new VALIDATED pillar on PASS, NEGATIVE pillar on miss); update AUTONOMOUS_STATE; commit + push BOTH remotes.

(d) On clean PASS: fresh dedicated adversarial review BEFORE the capability-pillar claim.

---

### Honest scope

A focused build on a validated foundation. Whatever the verdict, it is one further test in a continuing line; the algebra-PASS + characterisation stand. A PASS on one bridge justifies extending across the 5-bridge ensemble and ultimately connecting mode-unification to generative replay. A NEGATIVE is a clean biology-translatable finding: the algebra-PASS does not transfer to the biologized substrate even at substrate-friendly noise, and the failure mode sharpens which biological component needs refinement. Frozen bar never tuned; reuse-by-import only; no protected/frozen/moat module modified; no autograd; no-confab moat must stay 7/7 green throughout.
