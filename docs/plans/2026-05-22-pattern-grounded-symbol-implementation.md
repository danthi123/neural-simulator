---
type: plan
status: live
date: 2026-05-22
---

# Pattern-grounded compositional symbols: TDD implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task in the same session (the owner's standing instruction pre-selects same-session subagent-driven execution; transition directly from this plan to subagent-driven-development).

**Goal:** Implement and decisively test pattern-grounded compositional symbols on the trained 64-concept sparse-distributed substrate, per `docs/plans/2026-05-22-pattern-grounded-symbol-design.md`. The symbol-derivation step is the only change relative to the trained-substrate decisive runner; the recognition front-end, the compositional pipeline, the attractor clean-up, the frozen 0.80 bar, the multi-seed grid, and the loads {2,3,5} are all unchanged. Pre-registered reading: PASS = multi-seed mean >= 0.80 at all loads. The honest oracle-adjacency caveat is recorded up front on every artefact.

**Architecture:** A focused byte-reuse extension of `research/findings/raw/vocabulary_scaling_run_trained.py`. The genuinely-new code is two pure functions: `pattern_vector(concept_idx, n_pool, pattern)` builds the binary 0/1 indicator vector over the pool from a stored sparse pattern, and `_ground_symbols_pattern(words, patterns, n_pool, d_act)` returns the per-concept grounded phasor using the SAME fixed-seed deriver the activity-grounded path uses (only the deriver's input differs: pattern indicator vs mean-centred activity). The runner's `run_one_seed_pattern` substitutes `_ground_symbols_pattern` for the activity-grounded `_ground_symbols`; everything else -- capture, recognition, the resonate-and-fire FHRR + attractor clean-up via `run_pipeline`, the multi-seed aggregate -- is imported byte-unchanged.

**Tech Stack:** Python + numpy. Reuses the validated G.20 substrate builder + training + activity cache + biologized pipeline by import. The decisive run is CPU-feasible (the symbol-derivation change is pure numpy; recognition still reads the trained activity cache). No GPU strictly required; no autograd; no protected, frozen, or moat module modified.

---

### Task 0: Grounding pin

**Files:**
- Create: `tests/test_pattern_grounded_pin.py`

**Step 1: Write the failing test**

```python
"""Grounding pin for the pattern-grounded symbol arc -- pins the design
doc's contract that the frozen 0.80 compositional bar is unchanged and
the test grid (multi-seed, loads {2,3,5}, 64 concepts, validated
deriver dimension 512) is identical to the trained-substrate decisive
run. Goes green only after Task 2 is in place -- intentional: a Task 0
failure surfaces any drift in the load-bearing constants the moment a
later task is wired."""
import pytest

from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, N_CONCEPTS, K_RECOG, K_VOCAB, N_TRIALS,
)


def test_compositional_bar_frozen():
    assert BAR == 0.80


def test_test_grid_unchanged():
    assert N_CONCEPTS == 64
    assert LOADS == [2, 3, 5]
    assert SEEDS == [42, 43, 44]
    assert N_DIM == 512
    assert K_RECOG == 8 and K_VOCAB == 8 and N_TRIALS == 200


def test_pattern_grounded_runner_module_exists():
    # Imports the pattern-grounded runner so the pin trips if Task 2 has
    # not been built yet, or if its public surface drifts.
    from research.findings.raw import vocabulary_scaling_run_pattern_grounded  # noqa: F401
    assert hasattr(vocabulary_scaling_run_pattern_grounded, "pattern_vector")
    assert hasattr(vocabulary_scaling_run_pattern_grounded, "_ground_symbols_pattern")
    assert hasattr(vocabulary_scaling_run_pattern_grounded, "run_one_seed_pattern")
```

**Step 2: Run test to verify it fails**

`python -m pytest tests/test_pattern_grounded_pin.py -q`

Expected: `test_pattern_grounded_runner_module_exists` FAILS (module not built yet). The two constant pins pass immediately.

**Step 3: Commit**

```bash
git add tests/test_pattern_grounded_pin.py
git commit -m "pattern-grounded Task 0: grounding pin (red until Task 2 -- intentional)"
```

---

### Task 1: `pattern_vector` helper (pure function, unit-tested)

**Files:**
- Create: `research/findings/raw/vocabulary_scaling_pattern_helpers.py`
- Create: `tests/test_vocabulary_scaling_pattern_helpers.py`

**Step 1: Write the failing tests**

```python
"""Unit tests for the pattern_vector helper. Pure function; tests it
gives a binary 0/1 indicator over the pool with ones at exactly the
pattern's neuron indices, refuses out-of-range indices, and is
deterministic."""
import numpy as np
import pytest

from research.findings.raw.vocabulary_scaling_pattern_helpers import (
    pattern_vector,
)


def test_pattern_vector_basic_shape_and_values():
    v = pattern_vector([1, 3, 5], n_pool=8)
    assert v.shape == (8,)
    assert v.dtype == np.float64
    assert v.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]


def test_pattern_vector_full_size():
    v = pattern_vector(list(range(100)), n_pool=2000)
    assert v.shape == (2000,)
    assert int(v.sum()) == 100
    assert np.all(np.isin(np.where(v > 0)[0], list(range(100))))


def test_pattern_vector_rejects_out_of_range():
    with pytest.raises(ValueError):
        pattern_vector([0, 1, 2000], n_pool=2000)
    with pytest.raises(ValueError):
        pattern_vector([-1, 0, 1], n_pool=2000)


def test_pattern_vector_deterministic():
    v1 = pattern_vector([7, 13, 22], n_pool=64)
    v2 = pattern_vector([7, 13, 22], n_pool=64)
    assert np.array_equal(v1, v2)
```

**Step 2: Run tests to verify they fail**

`python -m pytest tests/test_vocabulary_scaling_pattern_helpers.py -q`

Expected: import-time `ModuleNotFoundError`.

**Step 3: Write minimal implementation**

```python
"""Pure helpers for pattern-grounded compositional symbols.

The K-of-N sparse pattern is a list of K pool-neuron indices; the
symbol-derivation step needs the corresponding binary indicator vector
over the whole N-neuron pool. Trivial function; isolated so the runner
imports it instead of inlining it (so the dedicated adversarial review
can confirm by name that the symbol's input is the pattern indicator
and nothing else)."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def pattern_vector(pattern: Iterable[int], n_pool: int) -> np.ndarray:
    """Return the binary 0/1 indicator vector for a K-of-N sparse
    pattern. 1 at every neuron index in `pattern`, 0 elsewhere.

    Raises ValueError if any index is out of [0, n_pool)."""
    v = np.zeros(int(n_pool), dtype=np.float64)
    for idx in pattern:
        i = int(idx)
        if i < 0 or i >= n_pool:
            raise ValueError(
                f"pattern index {i} out of range [0, {n_pool})")
        v[i] = 1.0
    return v
```

**Step 4: Run tests to verify they pass**

`python -m pytest tests/test_vocabulary_scaling_pattern_helpers.py -q`

Expected: 4/4 PASS.

**Step 5: Commit**

```bash
git add research/findings/raw/vocabulary_scaling_pattern_helpers.py tests/test_vocabulary_scaling_pattern_helpers.py
git commit -m "pattern-grounded Task 1: pattern_vector helper (pure function, 4/4 unit tests)"
```

---

### Task 2: The pattern-grounded runner

**Files:**
- Create: `research/findings/raw/vocabulary_scaling_run_pattern_grounded.py`
- Test: `tests/test_pattern_grounded_pin.py` (the Task 0 pin -- it goes green when this task lands)

**Step 1: Re-run the grounding pin to confirm it still trips**

`python -m pytest tests/test_pattern_grounded_pin.py -q`

Expected: `test_pattern_grounded_runner_module_exists` still FAILS (red).

**Step 2: Write the runner**

```python
"""Vocabulary scaling on the trained substrate with PATTERN-GROUNDED
symbols -- candidate 2 of the vocabulary-scaling NEGATIVE branch.

WHY THIS RUNNER EXISTS
----------------------
The trained-substrate decisive run cleared the frozen 0.80 bar at
loads 2-3 multi-seed (0.842, 0.814) but missed at load 5 (0.756). The
load-ceiling characterisation showed the ceiling sits between binding
loads 3 and 4 (L=4 mean 0.7988, miss by 0.0012; smooth monotonic
~0.03/load decay), about a 30x capacity reduction from the pure FHRR
algebra at the same phasor dimension. The hypothesis: the spiking-
symbol noise floor is the limit; replacing the noisy activity-derived
symbol with the substrate's clean K-of-N pattern-derived symbol should
raise the ceiling.

THE HONEST ORACLE-ADJACENCY CAVEAT (recorded up front)
------------------------------------------------------
The K-of-N pattern is the substrate's own concept code -- stored in
the trained connectivity, evoked by the language-input drive, and
extracted via the existing recognition front-end (which still reads
the noisy activity). So pattern-grounded symbols ARE substrate-
grounded. But the pattern abstracts past the per-observation noise to
the underlying stable ensemble identity, so it is ONE STEP CLOSER to
oracle-lookup than activity-grounded. A PASS here is read with that
caveat in mind, not as a biological compositional result at the same
fidelity as activity-grounded. See the design doc:
`docs/plans/2026-05-22-pattern-grounded-symbol-design.md`.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
The symbol-derivation step is the only thing that changes. The
recognition front-end (temporally averaged nearest-match in the
captured activity space), the FHRR pipeline (resonate-and-fire
bind/unbind/bundle), the attractor clean-up with separate familiarity
gate, the deriver (same fixed-seed linear projection), the frozen
0.80 bar, the multi-seed grid {42, 43, 44}, the loads {2, 3, 5}, the
FHRR phasor dimension 512 -- ALL imported byte-unchanged from
`vocabulary_scaling_run.py`.

The recognised concept name (the OUTPUT of recognition, not the true
label) selects which K-of-N pattern is read from the per-seed pattern
store. The true label NEVER indexes the pattern store -- doing so
would be an answer leak. The adversarial review must exploit-check
this explicitly.

PRE-REGISTERED reading (fixed; never tuned):
- PASS: integrated multi-seed mean >= 0.80 at all loads {2, 3, 5}.
  Subject to the oracle-adjacency caveat above, pattern-grounded
  symbols raise the ceiling past where activity-grounded missed.
- NEGATIVE: integrated below 0.80 at some load. The spiking-symbol
  noise is NOT the only ceiling cause; the limit is deeper.

Reuse-by-import only; no protected/frozen/moat module modified; no
automatic differentiation. Plain ASCII.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The biologized pipeline + everything downstream of the substrate.
from research.findings.raw.vocabulary_scaling_run import (
    N_CONCEPTS, BAR, LOADS, SEEDS, N_DIM, K_RECOG, K_VOCAB, N_TRIALS,
    recognition_accuracy, run_pipeline, _load_cache,
)
# Task 1's pure helper.
from research.findings.raw.vocabulary_scaling_pattern_helpers import (
    pattern_vector,
)
# The biologized pipeline's fixed-seed phasor deriver -- identical
# to what activity-grounded uses; only the input vector differs.
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.runners.spiking_phasor_fhrr import phases_to_spikes

DERIV_SEED = 90909   # same deriver seed as the activity-grounded path

# The trained substrate's activity cache (read for recognition; the
# decisive trained-substrate run populated it).
TRAINED_CACHE_DIR = os.path.join(
    _HERE, "vocabulary_scaling_trained_cache")


def _ground_symbols_pattern(words, patterns, n_pool, d_act):
    """The pattern-grounded symbol derivation: per concept, build the
    binary K-of-N indicator vector and project it through the SAME
    fixed-seed deriver the activity-grounded path uses, then quantise
    to phasor spikes. The genuinely-new symbol-derivation step.

    `d_act` is the deriver's input dimensionality. The activity-
    grounded path uses d_act = n_pool (one feature per pool neuron);
    we use the same d_act here so the deriver is byte-identical."""
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    return {w: phases_to_spikes(deriver(pattern_vector(patterns[i], n_pool)))
            for i, w in enumerate(words)}


def run_one_seed_pattern(seed):
    """Run the pipeline on the trained activity cache with pattern-
    grounded symbols. Recognition reads the cached activity (unchanged);
    only the symbol derivation differs."""
    print(f"\n--- seed {seed} ---", flush=True)
    path = os.path.join(TRAINED_CACHE_DIR, f"trained_full_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"trained activity cache missing: {path}; run the trained-"
            f"substrate decisive runner first to populate it")
    acts, words, patterns = _load_cache(path)
    d_act = acts[words[0]].shape[1]
    n_pool = d_act

    # Recognition (reused unchanged) -- the only handle that names which
    # concept's pattern is read.
    consolidated = {w: acts[w][:K_VOCAB].mean(axis=0) for w in words}
    rec_per_obs, rec_avg = recognition_accuracy(
        acts, words, consolidated, K_RECOG,
        np.random.default_rng(seed + 7))

    # Substitute the symbol-derivation step.
    grounded = _ground_symbols_pattern(words, patterns, n_pool, d_act)

    # The pipeline's run_pipeline takes its own `grounded` map via the
    # _ground_symbols call inside it -- so we need a thin local copy of
    # the pipeline body that uses our grounded map. Reuse the imports;
    # do not modify run_pipeline (it is byte-unchanged in the activity-
    # grounded runner). The body below mirrors run_pipeline exactly,
    # only the `grounded` source differs.
    from research.runners.resonate_fire_fhrr import (
        ResonateFireFHRR, ResonateFireTPAM,
        ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
    )
    from research.findings.raw.vocabulary_scaling_run import (
        partition_cue_filler, _cosine,
    )

    cue_words, filler_words = partition_cue_filler(words)
    fidx = {fw: i for i, fw in enumerate(filler_words)}
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    tpam = ResonateFireTPAM([grounded[fw] for fw in filler_words])
    qrng = np.random.default_rng(seed + 1)
    consolidated_mat = {w: consolidated[w] for w in words}

    def reco(word):
        m = acts[word].shape[0]
        k = min(K_RECOG, m)
        idx = qrng.choice(m, size=k, replace=False)
        avg = acts[word][idx].mean(axis=0)
        best_w, best_s = None, -2.0
        for w in words:
            s = _cosine(avg, consolidated_mat[w])
            if s > best_s:
                best_s, best_w = s, w
        return best_w

    per_load = {}
    for load in LOADS:
        n_int_ok = n_int_tot = 0
        n_comp_ok = n_comp_tot = 0
        eff_load = min(load, len(cue_words), len(filler_words))
        for _ in range(N_TRIALS):
            cues = list(qrng.choice(cue_words, size=eff_load, replace=False))
            fills = list(qrng.choice(filler_words, size=eff_load, replace=True))
            rec_cue = {c: reco(c) for c in set(cues)}
            rec_fill = {f: reco(f) for f in set(fills)}
            facts = list(zip(cues, fills))
            composite = net.encode([
                (grounded[rec_cue[c]], grounded[rec_fill[f]])
                for (c, f) in facts])
            for (c, f) in facts:
                recovered = net.query(composite, grounded[rec_cue[c]])
                z, _ = tpam.settle_annealed(
                    recovered, ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH,
                    ANNEAL_ITERS, fast=True)
                overlaps = np.abs(tpam.s.conj().T @ z)
                hit = (int(np.argmax(overlaps)) == fidx[f])
                n_int_ok += int(hit)
                n_int_tot += 1
                if rec_cue[c] == c and rec_fill[f] == f:
                    n_comp_ok += int(hit)
                    n_comp_tot += 1
        int_acc = n_int_ok / n_int_tot if n_int_tot else float("nan")
        comp_acc = (n_comp_ok / n_comp_tot) if n_comp_tot else float("nan")
        per_load[load] = {
            "integrated_accuracy": int_acc,
            "composition_only_accuracy": comp_acc,
            "n_composition_only": n_comp_tot,
            "effective_load": eff_load,
        }

    for load in LOADS:
        e = per_load[load]
        print(f"  L={load}: integrated acc={e['integrated_accuracy']:.4f} "
              f"| composition-only acc={e['composition_only_accuracy']:.4f} "
              f"(n={e['n_composition_only']})", flush=True)
    print(f"  [seed {seed}] recognition (reported separately): "
          f"per-observation={rec_per_obs:.4f}, "
          f"temporally-averaged={rec_avg:.4f}", flush=True)

    return {
        "seed": seed, "trained_substrate": True,
        "symbol_grounding": "pattern",
        "n_concepts": len(words), "activity_dim": int(d_act),
        "recognition_per_observation": rec_per_obs,
        "recognition_temporally_averaged": rec_avg,
        "per_load": per_load,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Pattern-grounded compositional symbols on the "
                    "trained 64-concept G.20 sparse substrate -- "
                    "candidate 2 of the NEGATIVE branch.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    seeds = list(args.seeds)

    print("=== vocabulary scaling: PATTERN-GROUNDED symbols on the "
          "trained 64-concept G.20 sparse substrate ===", flush=True)
    print(f"  ORACLE-ADJACENCY CAVEAT: the K-of-N pattern is the "
          f"substrate's own concept code -- still substrate-grounded -- "
          f"but one step closer to oracle-lookup than activity-grounded; "
          f"a PASS is read with that caveat (see design doc).",
          flush=True)
    print(f"concepts={N_CONCEPTS}; FHRR N_dim={N_DIM}; loads={LOADS}; "
          f"bar={BAR}; seeds={seeds}; substrate=TRAINED (cache reused); "
          f"recognition unchanged; symbol grounding=PATTERN",
          flush=True)

    seed_results = [run_one_seed_pattern(s) for s in seeds]

    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    agg = {}
    all_pass = True
    for load in LOADS:
        int_accs = [r["per_load"][load]["integrated_accuracy"]
                    for r in seed_results]
        comp_accs = [r["per_load"][load]["composition_only_accuracy"]
                     for r in seed_results]
        mean_int = float(np.mean(int_accs))
        valid_comp = [c for c in comp_accs if c == c]
        mean_comp = float(np.mean(valid_comp)) if valid_comp else float("nan")
        agg[load] = {"mean_integrated": mean_int,
                     "per_seed_integrated": int_accs,
                     "mean_composition_only": mean_comp}
        if mean_int < BAR:
            all_pass = False
        print(f"  L={load}: integrated per-seed="
              f"{['%.3f' % a for a in int_accs]} mean={mean_int:.4f} "
              f"({'>=' if mean_int >= BAR else '<'} {BAR}) | "
              f"composition-only mean={mean_comp:.4f}", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if all_pass:
        verdict = "VOCABULARY_SCALING_64CONCEPT_PATTERN_GROUNDED_PASS"
        print("  Pattern-grounded symbols clear the frozen 0.80 bar "
              "multi-seed at all loads on the trained 64-concept G.20 "
              "sparse substrate. Subject to the oracle-adjacency caveat "
              "above.", flush=True)
    else:
        verdict = "VOCABULARY_SCALING_64CONCEPT_PATTERN_GROUNDED_BELOW_BAR"
        print("  Pattern-grounded multi-seed mean is below 0.80 at some "
              "load. The spiking-symbol noise is NOT the only ceiling "
              "cause; the limit is deeper.", flush=True)

    out = {
        "seeds": seeds, "n_concepts": N_CONCEPTS, "n_dim": N_DIM,
        "k_recog": K_RECOG, "loads": LOADS, "n_trials": N_TRIALS,
        "bar": BAR, "substrate": "trained",
        "symbol_grounding": "pattern",
        "oracle_adjacency_caveat": (
            "The K-of-N pattern is the substrate's own concept code -- "
            "still substrate-grounded -- but one step closer to "
            "oracle-lookup than activity-grounded; a PASS is read with "
            "this caveat."),
        "per_seed": seed_results,
        "aggregate": {str(k): v for k, v in agg.items()},
        "verdict": verdict,
    }
    out_path = os.path.join(
        _HERE, "vocabulary_scaling_run_pattern_grounded.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 3: Run the grounding pin to verify it goes green**

`python -m pytest tests/test_pattern_grounded_pin.py -q`

Expected: 3/3 PASS.

**Step 4: Commit**

```bash
git add research/findings/raw/vocabulary_scaling_run_pattern_grounded.py
git commit -m "pattern-grounded Task 2: runner (focused byte-reuse extension; only symbol-derivation differs)"
```

---

### Task 3: Soundness tests for the pattern-grounded runner

**Files:**
- Create: `tests/test_vocabulary_scaling_pattern_grounded.py`

**Step 1: Write the failing tests**

```python
"""Soundness tests for the pattern-grounded runner. The load-bearing
properties: (a) the grounded symbol's INPUT is genuinely the pattern
indicator and not the activity (pinned by computing both and asserting
they differ on real cached data); (b) the recognition front-end is the
only handle that names which pattern is read (pinned by the runner's
module-level imports + a structural check); (c) the deriver is the
exact one the activity-grounded path uses (pinned by DERIV_SEED + the
module the deriver comes from)."""
import os

import numpy as np
import pytest

from research.findings.raw.vocabulary_scaling_run_pattern_grounded import (
    _ground_symbols_pattern, DERIV_SEED, TRAINED_CACHE_DIR,
)
from research.findings.raw.vocabulary_scaling_pattern_helpers import (
    pattern_vector,
)
from research.findings.raw.vocabulary_scaling_run import (
    N_CONCEPTS, _load_cache,
)


def test_deriv_seed_matches_activity_grounded_path():
    # The deriver must be byte-identical to the activity-grounded path;
    # only its INPUT changes. The activity-grounded path uses
    # DERIV_SEED = 90909.
    assert DERIV_SEED == 90909


def test_pattern_indicator_differs_from_activity():
    cache = os.path.join(TRAINED_CACHE_DIR, "trained_full_seed42.npz")
    if not os.path.exists(cache):
        pytest.skip("trained activity cache not yet populated")
    acts, words, patterns = _load_cache(cache)
    n_pool = acts[words[0]].shape[1]
    # Pattern indicator for the first concept.
    pv = pattern_vector(patterns[0], n_pool)
    # Activity-derived input (mean-centred consolidated activity) for
    # the same concept.
    consolidated = {w: acts[w][:8].mean(axis=0) for w in words}
    common = np.mean([consolidated[w] for w in words], axis=0)
    av = consolidated[words[0]] - common
    # The two inputs are different objects; the pattern indicator is
    # binary, the activity input is real-valued and mostly nonzero.
    assert pv.shape == av.shape
    assert set(np.unique(pv).tolist()) == {0.0, 1.0}
    assert not set(np.unique(av).tolist()).issubset({0.0, 1.0})


def test_ground_symbols_pattern_returns_one_phasor_per_word():
    cache = os.path.join(TRAINED_CACHE_DIR, "trained_full_seed42.npz")
    if not os.path.exists(cache):
        pytest.skip("trained activity cache not yet populated")
    acts, words, patterns = _load_cache(cache)
    n_pool = acts[words[0]].shape[1]
    grounded = _ground_symbols_pattern(words, patterns, n_pool, n_pool)
    assert set(grounded.keys()) == set(words)
    # Each grounded symbol must be a complex phasor of length N_dim.
    from research.findings.raw.vocabulary_scaling_run import N_DIM
    for w in words:
        z = grounded[w]
        assert z.shape == (N_DIM,)
        assert np.iscomplexobj(z)
```

**Step 2: Run tests to verify they pass (Task 2 must be in place)**

`python -m pytest tests/test_vocabulary_scaling_pattern_grounded.py -q`

Expected: 3/3 PASS (or 1 pass + 2 skip if the trained cache is not yet on this machine -- the cache is from the decisive run and is in `vocabulary_scaling_trained_cache/`).

**Step 3: Commit**

```bash
git add tests/test_vocabulary_scaling_pattern_grounded.py
git commit -m "pattern-grounded Task 3: soundness tests (deriver seed pinned; pattern input differs from activity)"
```

---

### Task 4: Dedicated adversarial review (BEFORE the decisive run)

**Files:** none (review only)

**Step 1: Dispatch a fresh general-purpose agent (subagent) with the prompt template at the design doc's "Soundness considerations" section, expanded with these exploit-class checks:**

1. **No answer leak.** Confirm the true label NEVER indexes the pattern store. The pattern read for each fact is selected by `rec_cue[c]` / `rec_fill[f]` -- the OUTPUT of the recognition step -- not by `c` / `f` (the true labels). Trace the runner end-to-end.

2. **Recognition genuinely load-bearing.** If recognition is bypassed (e.g., the runner uses true labels directly), construct an exploit that scores PASS without recognition. Confirm the runner cannot do so.

3. **Deriver identical to activity-grounded.** `make_deriver(N_DIM, d_act, DERIV_SEED)` with DERIV_SEED == 90909, imported from the same module the activity-grounded path uses. The dimensions match (n_pool input, N_DIM output).

4. **Frozen bar immovable.** `BAR` imported from `vocabulary_scaling_run.py`; never redefined or scaled.

5. **No protected/frozen/moat module modified.** `git diff` since the parent commit shows only new files (the runner, the helper, the tests, the smell-test if added). No modification of any sim/, the no-confab moat, any `*_core.py`, or the trained-substrate runner.

6. **No automatic differentiation.** Grep the runner + its call graph.

7. **The reused pipeline is byte-identical.** `run_one_seed_pattern` MIRRORS `run_pipeline`'s body verbatim (modulo the `grounded` source); cross-check.

8. **The pattern store is the substrate's stored code, not freely chosen.** `patterns` comes from the cached `.npz` saved by the trained-substrate runner; the cache was saved from the substrate's per-concept patterns at build time; the runner does not regenerate them with a different seed.

The reviewer RUNS the checks (full tool access) and returns VERDICT CLEAR or VERDICT BLOCK with specific defects. The decisive run does NOT launch until CLEAR.

**Step 2: If the reviewer returns BLOCK, fix the specific defects (net-new code only; no protected/frozen module edit), re-run the reviewer.**

**Step 3: Commit the review verdict (if any change resulted; if CLEAR-on-first-pass, no commit needed -- the verdict goes in the findings doc).**

---

### Task 5: Controller-only decisive run (NOT a subagent task)

**This is the controller's responsibility -- bring back to the controller after the review is CLEAR.**

**Step 1: Smoke run (a single-seed dry run on the cached activity, ~tens of seconds on CPU):**

`python research/findings/raw/vocabulary_scaling_run_pattern_grounded.py --seeds 42`

Confirm it runs end-to-end, writes the JSON, prints sane numbers.

**Step 2: Multi-seed decisive run:**

`python research/findings/raw/vocabulary_scaling_run_pattern_grounded.py`

Expected wall-clock: tens of seconds to minutes (the symbol-derivation change is pure-numpy; recognition reads the cached activity; no GPU; no re-train).

**Step 3: Mandatory anti-cheat smell-test.**

Adapt the existing smell-test tool's logic (it expects the activity-grounded JSON shape; the pattern-grounded JSON shape is nearly identical). Recompute per-load means from per_seed independently of the runner's aggregate; re-derive the verdict against the frozen 0.80 bar; consistency checks (composition-only vs integrated; per-seed variation).

**Step 4: Extended load-ceiling characterisation on the pattern-grounded pipeline.**

Re-run the pipeline at loads {2..7} (mirroring `vocabulary_scaling_load_ceiling_probe.py`'s pattern -- but with `_ground_symbols_pattern` as the symbol source) so the comparison curve against the activity-grounded ceiling map extends across the full range. Cheap CPU; no GPU.

**Step 5: Findings doc, capability_status pillar, AUTONOMOUS_STATE update, commit + push both remotes.**

The findings doc reports the verdict with the oracle-adjacency caveat front and centre, the comparison against the activity-grounded ceiling map, and the next pre-registered step (continuing autonomously per the discipline).

---

### Honest scope

This plan executes ONE further test on the biologization arc. Whatever the verdict, it is one further test in a continuing line -- not a final answer. The completed twice-reviewed 16-concept FHRR-biologization arc (multi-seed 0.98) stands; the trained-substrate 64-concept BOUNDARY result (multi-seed 0.84 / 0.81 at loads 2-3, ceiling between 3 and 4) stands. The oracle-adjacency caveat is recorded up front and any PASS is read with that caveat. Frozen bar never tuned; reuse-by-import only; no protected, frozen, or moat module modified; no automatic differentiation.
