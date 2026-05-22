"""Vocabulary scaling of the biologized compositional capability -- the
capture-and-run (Task 2 of
docs/plans/2026-05-22-vocabulary-scaling-implementation.md).

The biologized grounded-composition pipeline is validated and
adversarially-reviewed CLEAR at a 16-concept vocabulary (multi-seed
0.98 against the frozen 0.80 compositional bar). The one scaling axis
not yet addressed is the vocabulary itself. This runner asks the
pre-registered question: does that same pipeline still clear the frozen
0.80 bar at a 64-concept vocabulary, on the project's own validated
large-vocabulary substrate -- the catalog G.20 sparse-distributed
ensemble?

Pipeline (per seed):
1. BUILD: one 64-concept G.20 sparse bridge via Task 1's
   `build_64_concept_sparse_bridge` (a thin wrapper that reuses the
   validated G.20 sparse builder byte-unchanged). Each concept is a
   scattered K-of-N sparse pattern in ONE `shared_concept_pool` -- the
   Kanerva-SDM form, NOT v14/v16 separate pools.
2. CAPTURE: drive each concept's sparse pattern (orthogonal lang_input
   drive + teacher current on the concept's pool-pattern neurons -- the
   same way `concept_pool_sparse_distributed` itself defines a concept)
   and record the per-neuron firing-rate vector over the WHOLE shared
   concept pool. M observations differ by the bridge's genuine
   trial-to-trial OU-noise variability. This mirrors `capture_activity`
   in `activity_level_integration.py`; the only adaptation is that the
   G.20 "concept population" is the one shared pool, not a per-concept
   pool slice.
3. COMPOSE: run the biologized grounded-composition pipeline -- imported
   byte-unchanged from `biologized_grounded_composition.py`: the
   grounded symbol = common-mode-removed concept activity -> phasor (the
   `meancenter` deriver); resonate-and-fire FHRR bind/bundle/unbind; an
   annealed attractor (TPAM) clean-up. Generalised from its v14/v16
   16-pool taxonomy to the 64-concept layout: cue and filler roles are
   assigned by a FIXED partition of the 64 concepts (first 32 cues,
   last 32 fillers), NOT by v14/v16 pool-name prefixes.
4. MEASURE: integrated + composition-only accuracy against the frozen
   0.80 bar at loads {2,3,5}.
5. Recognition is reported SEPARATELY and honestly: with the G.20
   sparse substrate there are no per-concept pools, so recognition is a
   nearest-match in the captured activity space (cosine of a noisy
   observation's activity to the consolidated per-concept activity) --
   reported as its own number, not folded into composition accuracy.

PRE-REGISTERED reading (fixed; never tuned):
- PASS: integrated multi-seed mean >= 0.80 at all loads {2,3,5}. The
  biologized grounded compositional capability scales to a 64-concept
  vocabulary; proceed to the 160/320-concept ensemble.
- NEGATIVE: integrated below 0.80 at some load. The honest finding is
  which stage costs it -- composition (unlikely, per the load curve),
  recognition, or concept separability at 64 concepts.

Reuse-by-import only: the validated G.20 sparse builder (via Task 1's
wrapper), the activity-capture pattern, and the biologized
grounded-composition pipeline's stages -- all byte-unchanged. No
protected / frozen / moat module modified. No automatic differentiation.

`--smoke` shrinks the vocabulary subset + observation count for a fast
end-to-end check (toy numbers, NOT a result). A kill-safe per-seed
activity cache (a numeric-only .npz, loaded with allow_pickle=False)
lets a re-run skip the captured seeds. Plain ASCII.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Task 1's 64-concept G.20 sparse bridge builder (a thin wrapper that
# reuses the validated G.20 sparse builder byte-unchanged).
from research.findings.raw.vocabulary_scaling_substrate import (
    build_64_concept_sparse_bridge,
    sixty_four_concept_sparse_patterns,
    sixty_four_concept_vocabulary,
    DEFAULT_N_LANG_INPUT,
    DEFAULT_N_SHARED_POOL,
    DEFAULT_N_SHARED_FS,
    DEFAULT_PATTERN_SIZE,
)
# The biologized grounded-composition pipeline stages -- reused by
# import, byte-unchanged. `make_deriver` (the activity -> phasor
# deriver) comes via the pattern-separation probe, exactly as the
# biologized pipeline imports it.
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from research.runners.resonate_fire_fhrr import (
    ResonateFireFHRR, ResonateFireTPAM,
    ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
)

# =====================================================================
# Pre-registered constants (frozen in advance; Task 0's pin asserts
# exactly these). The 0.80 bar is the project's pre-registered frozen
# compositional bar and must NEVER be tuned.
# =====================================================================
N_CONCEPTS = 64
BAR = 0.80
LOADS = [2, 3, 5]

# Multi-seed set for the decisive run (matches the design doc).
SEEDS = [42, 43, 44]

# Pipeline parameters -- identical to the validated biologized
# grounded-composition pipeline (biologized_grounded_composition.py).
N_DIM = 512                  # FHRR phasor dimension
K_RECOG = 8                  # observations averaged for the recognition readout
K_VOCAB = 8                  # registration observations averaged for the symbol
N_TRIALS = 200               # composition trials per load (full run)
M_OBS = 16                   # activity observations captured per concept (full)
DERIV_SEED = 90909           # fixed -- the deriver is a fixed interface property

# Substrate drive parameters for the activity capture -- mirror the
# validated G.20 sparse drive (concept_pool_sparse_distributed): an
# orthogonal lang_input drive + a teacher current on the concept's
# sparse pool pattern. Observation only -- no plasticity gate is opened.
DRIVE_PA = 200.0             # lang_input orthogonal drive (G.20 default)
TEACHER_PA = 100.0           # teacher bias on the sparse pattern (G.20 capture)
# orthogonal_drive_pattern requires n_active = round(sparsity * n_lang)
# <= n_lang // n_cues. With 64 concepts the stride is n_lang/64, so the
# sparsity must shrink as the concept count rises -- exactly the
# documented G.20 practice: the 320-concept tier used sparsity 0.007
# because "orthogonal-drive needs n_active < stride" (CLAUDE.md G.20
# section). At the full tier (n_lang=8192, 64 cues -> stride 128) and
# the smoke tier (n_lang=512, 64 cues -> stride 8), 0.01 keeps a clean
# margin (n_active 82 < 128; 5 < 8). NOT a tuned parameter -- a fixed
# substrate-drive value chosen to satisfy the orthogonal-code geometry,
# the same way every G.20 scale picks its sparsity.
SPARSITY = 0.01              # orthogonal-drive sparsity (64-concept tier)
STIM_STEPS = 100             # firing accumulation window (full run)
RESET_STEPS = 20             # settle steps between observations (G.20 default)

CACHE_DIR = "research/findings/raw/vocabulary_scaling_cache"

# Smoke-scale substrate sizes -- a tiny G.20 sparse bridge for a fast
# structural / end-to-end check. Toy numbers, NOT a result.
SMOKE_N_LANG_INPUT = 512
SMOKE_N_SHARED_POOL = 512
SMOKE_N_SHARED_FS = 60
SMOKE_PATTERN_SIZE = 24
SMOKE_VOCAB = 12             # concepts used in the smoke subset
SMOKE_M_OBS = 4
SMOKE_K = 2
SMOKE_TRIALS = 40
SMOKE_STIM_STEPS = 30


# ---------------------------------------------------------------------
# Cue / filler partition -- a FIXED split of the concepts.
# ---------------------------------------------------------------------
def partition_cue_filler(words):
    """Assign cue and filler roles by a FIXED partition of the concept
    words: the first half are cues, the second half are fillers.

    The v14/v16 16-pool pipeline split cues from fillers by pool-name
    prefixes (``noun_pool_`` / ``verb_pool_`` vs ``adjective_pool_``).
    The G.20 sparse vocabulary has no such taxonomy -- every concept is
    an undifferentiated sparse pattern in one shared pool -- so the plan
    requires a fixed partition instead. First-half / last-half is the
    simplest fixed, deterministic, role-balanced split and does not
    depend on word identity, only on position in the fixed vocabulary.
    """
    words = list(words)
    half = len(words) // 2
    cues = words[:half]
    fillers = words[half:]
    return cues, fillers


# ---------------------------------------------------------------------
# Substrate build helpers.
# ---------------------------------------------------------------------
def build_smoke_bridge(seed):
    """Build a tiny 64-concept G.20 sparse bridge for the smoke / tests
    -- reduced sizes so it builds in a couple of seconds on the NumPy
    backend. The wrapper is Task 1's; only the sizes are reduced."""
    return build_64_concept_sparse_bridge(
        seed=seed,
        n_lang_input=SMOKE_N_LANG_INPUT,
        n_shared_pool=SMOKE_N_SHARED_POOL,
        n_shared_fs=SMOKE_N_SHARED_FS,
        pattern_size=SMOKE_PATTERN_SIZE,
        verbose=False,
    )


def smoke_sparse_patterns(seed):
    """The 64 per-concept sparse patterns for the smoke-scale pool
    (Task 1's pure-function pattern generator at the smoke pool size)."""
    return sixty_four_concept_sparse_patterns(
        seed,
        n_shared_pool=SMOKE_N_SHARED_POOL,
        pattern_size=SMOKE_PATTERN_SIZE,
    )


# ---------------------------------------------------------------------
# Per-neuron activity capture.
# ---------------------------------------------------------------------
def _capture_one(bridge, lang_input_idx, pool_idx, drive_gpu, pattern_arr,
                 cp, to_host, stim_steps, reset_steps, teacher_pA):
    """Drive one concept once and record the per-neuron firing-rate
    vector over the shared concept pool.

    Mirrors `capture_activity` in `activity_level_integration.py`: a
    reset window, then an external-input drive held for `stim_steps`
    while `cp_firing_states` is accumulated over the recorded
    population. The only adaptation for the G.20 sparse substrate: the
    drive is the validated G.20 sparse drive (an orthogonal lang_input
    drive PLUS a teacher current on the concept's sparse pool pattern --
    exactly how `concept_pool_sparse_distributed` defines a concept's
    substrate signature), and the recorded population is the whole
    `shared_concept_pool` rather than a per-concept pool slice."""
    n_total = int(bridge.cp_external_input_current.shape[0])
    ext = cp.zeros(n_total, dtype=cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    counts = cp.zeros(pool_idx.shape[0], dtype=cp.float64)
    for _ in range(stim_steps):
        ext.fill(0)
        ext[lang_input_idx] = drive_gpu
        ext[pattern_arr] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += bridge.cp_firing_states[pool_idx]

    bridge.cp_external_input_current[:] = 0.0
    return np.asarray(to_host(counts), dtype=np.float64) / float(stim_steps)


def capture_concept_activity(bridge, words, sparse_patterns, m_obs=M_OBS,
                             n_lang_input=None, n_words_for_orthogonal=None,
                             stim_steps=STIM_STEPS, reset_steps=RESET_STEPS,
                             drive_pA=DRIVE_PA, teacher_pA=TEACHER_PA,
                             sparsity=SPARSITY, verbose=False):
    """Capture `m_obs` per-neuron concept-population activity observations
    for every concept word.

    Returns ``{word: (M, n_pool) float64 array}`` -- for each concept,
    M per-neuron firing-rate observations over the whole
    `shared_concept_pool`. The M observations of one concept differ by
    the bridge's genuine trial-to-trial OU-noise variability (the
    capture opens no plasticity gate; the bridge state is unchanged).

    `words` and `sparse_patterns` are index-aligned: `sparse_patterns[i]`
    is concept `words[i]`'s scattered K-of-N pattern (pool-local indices).
    A reduced `words` list (a smoke subset) captures only those concepts;
    `n_words_for_orthogonal` keeps the orthogonal-code stride matched to
    the full 64-concept vocabulary so a subset's codes do not collide.
    """
    from sim.backend import get_backend, to_host
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()

    rm = bridge.region_manager
    lang_input_local = list(rm.indices("language_input"))
    shared_local = list(rm.indices("shared_concept_pool"))
    lang_input_idx = cp.asarray(lang_input_local, dtype=cp.int64)
    pool_idx = cp.asarray(shared_local, dtype=cp.int64)

    if n_lang_input is None:
        n_lang_input = len(lang_input_local)
    # Orthogonal codes use the full 64-concept stride so a smoke subset
    # of concepts keeps non-overlapping codes (the validated G.20
    # capture passes n_words_for_orthogonal = the full concept count).
    if n_words_for_orthogonal is None:
        n_words_for_orthogonal = N_CONCEPTS

    acts = {}
    for cidx, word in enumerate(words):
        pat = sparse_patterns[cidx]
        pattern_global = [shared_local[i] for i in pat]
        pattern_arr = cp.asarray(pattern_global, dtype=cp.int64)

        # Concept index in the FULL vocabulary -> its orthogonal code.
        drive_in = orthogonal_drive_pattern(
            cue_idx=cidx, n_cues=n_words_for_orthogonal,
            n_neurons=int(n_lang_input),
            drive_max_pA=drive_pA, sparsity=sparsity,
        )
        drive_gpu = cp.asarray(drive_in, dtype=cp.float32)

        rows = []
        for _ in range(m_obs):
            rows.append(_capture_one(
                bridge, lang_input_idx, pool_idx, drive_gpu, pattern_arr,
                cp, to_host, stim_steps, reset_steps, teacher_pA))
        acts[word] = np.asarray(rows, dtype=np.float64)
        if verbose:
            mean_rate = float(acts[word].mean())
            print(f"    captured '{word}': {m_obs} obs, "
                  f"mean rate={mean_rate:.4f}", flush=True)
    return acts


# ---------------------------------------------------------------------
# Recognition (reported separately + honestly).
# ---------------------------------------------------------------------
def _cosine(a, b):
    """Cosine similarity of two non-negative activity vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def recognition_accuracy(acts, words, consolidated, k_recog, rng):
    """Honest recognition readout for the G.20 sparse substrate.

    The v14/v16 pipeline reads recognition as a per-pool argmax. The
    G.20 sparse substrate has no per-concept pools -- every concept is a
    scattered pattern in ONE shared pool -- so recognition is instead a
    NEAREST-MATCH in the captured activity space: average `k_recog`
    noisy observations of a concept and pick the consolidated concept
    activity it is closest to (cosine). Reported as its own number; it
    is NOT folded into the composition accuracy.

    Returns (per_observation_accuracy, temporally_averaged_accuracy):
    the design doc compares these to the ~0.66 / ~0.93 measured at 16
    concepts."""
    consolidated_mat = {w: consolidated[w] for w in words}

    def nearest(vec):
        best_w, best_s = None, -2.0
        for w in words:
            s = _cosine(vec, consolidated_mat[w])
            if s > best_s:
                best_s, best_w = s, w
        return best_w

    # Per-observation: each single noisy observation, classified.
    n_obs_ok = n_obs_tot = 0
    for w in words:
        for i in range(acts[w].shape[0]):
            n_obs_ok += int(nearest(acts[w][i]) == w)
            n_obs_tot += 1
    per_obs = n_obs_ok / n_obs_tot if n_obs_tot else float("nan")

    # Temporally averaged: k_recog observations averaged, then classified.
    n_avg_ok = n_avg_tot = 0
    for w in words:
        m = acts[w].shape[0]
        k = min(k_recog, m)
        for _ in range(m):  # m random k-subsets, matching the per-obs count
            idx = rng.choice(m, size=k, replace=False)
            avg = acts[w][idx].mean(axis=0)
            n_avg_ok += int(nearest(avg) == w)
            n_avg_tot += 1
    temporally_avg = n_avg_ok / n_avg_tot if n_avg_tot else float("nan")
    return per_obs, temporally_avg


# ---------------------------------------------------------------------
# Kill-safe activity cache (numeric-only .npz; loaded allow_pickle=False).
# ---------------------------------------------------------------------
def _cache_path(tag, seed):
    return os.path.join(CACHE_DIR, f"{tag}_seed{seed}.npz")


def _load_cache(path):
    data = np.load(path, allow_pickle=False)
    words = [str(w) for w in data["__words__"]]
    acts = {w: data["act__" + w] for w in words}
    patterns = [list(int(x) for x in data["pat__%03d" % i])
                for i in range(len(words))]
    return acts, words, patterns


def _save_cache(path, acts, words, patterns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save = {"__words__": np.asarray(words)}
    for w in words:
        save["act__" + w] = acts[w]
    for i, pat in enumerate(patterns):
        save["pat__%03d" % i] = np.asarray(pat, dtype=np.int64)
    np.savez(path, **save)


def capture_seed(seed, smoke, m_obs, verbose=True):
    """Build the 64-concept G.20 sparse bridge for `seed` and capture
    per-neuron concept-population activity for every concept word.
    Cached to disk -- a re-run loads the cache and skips the capture."""
    tag = "smoke" if smoke else "full"
    path = _cache_path(tag, seed)
    if os.path.exists(path):
        acts, words, patterns = _load_cache(path)
        if verbose:
            print(f"  [seed {seed}] loaded cached activity "
                  f"({len(words)} concepts, "
                  f"{acts[words[0]].shape[0]} obs/concept)", flush=True)
        return acts, words, patterns

    t0 = time.time()
    if smoke:
        bridge, all_words = build_smoke_bridge(seed)
        all_patterns = smoke_sparse_patterns(seed)
        # Smoke uses only a small subset of the 64 concepts.
        words = list(all_words[:SMOKE_VOCAB])
        patterns = [all_patterns[i] for i in range(SMOKE_VOCAB)]
        stim_steps = SMOKE_STIM_STEPS
        n_lang_input = SMOKE_N_LANG_INPUT
    else:
        bridge, words = build_64_concept_sparse_bridge(
            seed=seed,
            n_lang_input=DEFAULT_N_LANG_INPUT,
            n_shared_pool=DEFAULT_N_SHARED_POOL,
            n_shared_fs=DEFAULT_N_SHARED_FS,
            pattern_size=DEFAULT_PATTERN_SIZE,
            verbose=verbose,
        )
        patterns = sixty_four_concept_sparse_patterns(
            seed,
            n_shared_pool=DEFAULT_N_SHARED_POOL,
            pattern_size=DEFAULT_PATTERN_SIZE,
        )
        stim_steps = STIM_STEPS
        n_lang_input = DEFAULT_N_LANG_INPUT

    if verbose:
        print(f"  [seed {seed}] capturing {m_obs} activity "
              f"observations/concept for {len(words)} concepts ...",
              flush=True)
    acts = capture_concept_activity(
        bridge, words, patterns, m_obs=m_obs,
        n_lang_input=n_lang_input, n_words_for_orthogonal=N_CONCEPTS,
        stim_steps=stim_steps, verbose=verbose)

    _save_cache(path, acts, words, patterns)
    if verbose:
        print(f"  [seed {seed}] captured + cached in "
              f"{time.time() - t0:.1f}s", flush=True)
    return acts, words, patterns


# ---------------------------------------------------------------------
# The biologized grounded-composition pipeline, generalised to N concepts.
# ---------------------------------------------------------------------
def _ground_symbols(consolidated, words, d_act):
    """The biologized pipeline's `meancenter` grounding, byte-equivalent
    to `biologized_grounded_composition.run_one_seed`'s meancenter
    branch: subtract the across-concept common-mode activity
    (subtractive normalisation -- a recognised cortical computation),
    then derive a phasor via the fixed `make_deriver` projection.

    `make_deriver` and `phases_to_spikes` are the byte-unchanged
    pipeline imports; only the concept SET differs (64 concepts here
    vs 16 in the original)."""
    common = np.mean([consolidated[w] for w in words], axis=0)
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    return {w: phases_to_spikes(deriver(consolidated[w] - common))
            for w in words}


def run_pipeline(seed, acts, words, loads, n_trials, k_recog, k_vocab):
    """Run the biologized grounded-composition pipeline on captured
    64-concept activity.

    This is `biologized_grounded_composition.run_one_seed`'s pipeline,
    generalised: the grounding (mean-centred activity -> phasor), the
    resonate-and-fire FHRR (`ResonateFireFHRR`), and the annealed
    attractor clean-up (`ResonateFireTPAM` + the fixed anneal schedule)
    are all reused by import, byte-unchanged. The single generalisation
    is the concept taxonomy: cue and filler roles come from a FIXED
    partition of the concepts (`partition_cue_filler`) instead of the
    v14/v16 pool-name prefixes.

    Returns per-load integrated + composition-only accuracy. "Clean"
    (for the composition-only metric) means a concept's averaged
    recognition readout resolved to itself -- the G.20 analogue of the
    original's per-pool-argmax-correct condition."""
    d_act = acts[words[0]].shape[1]
    consolidated = {w: acts[w][:k_vocab].mean(axis=0) for w in words}
    grounded = _ground_symbols(consolidated, words, d_act)

    cue_words, filler_words = partition_cue_filler(words)
    fidx = {fw: i for i, fw in enumerate(filler_words)}

    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    # Annealed attractor clean-up over the filler grounded symbols.
    tpam = ResonateFireTPAM([grounded[fw] for fw in filler_words])
    qrng = np.random.default_rng(seed + 1)

    # Recognition for the G.20 sparse substrate = nearest-match in the
    # captured activity space (no per-concept pools to argmax over).
    consolidated_mat = {w: consolidated[w] for w in words}

    def reco(word):
        """Recognise a word: average k_recog noisy observations, then
        return the consolidated concept the average is closest to."""
        m = acts[word].shape[0]
        k = min(k_recog, m)
        idx = qrng.choice(m, size=k, replace=False)
        avg = acts[word][idx].mean(axis=0)
        best_w, best_s = None, -2.0
        for w in words:
            s = _cosine(avg, consolidated_mat[w])
            if s > best_s:
                best_s, best_w = s, w
        return best_w

    per_load = {}
    for load in loads:
        n_int_ok = n_int_tot = 0
        n_comp_ok = n_comp_tot = 0
        # Load cannot exceed the available cue / filler counts.
        eff_load = min(load, len(cue_words), len(filler_words))
        for _ in range(n_trials):
            cues = list(qrng.choice(cue_words, size=eff_load,
                                    replace=False))
            fills = list(qrng.choice(filler_words, size=eff_load,
                                     replace=True))
            # Recognise each word once (the recognised concept is used
            # consistently for encode and query) -- mirrors the original.
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
    return per_load


def run_one_seed(seed, smoke=False):
    """Capture the 64-concept G.20 sparse activity for one seed and run
    the biologized grounded-composition pipeline on it.

    Returns a result dict with per-load integrated + composition-only
    accuracies, and -- reported separately and honestly -- the
    per-observation and temporally-averaged recognition accuracy."""
    print(f"\n--- seed {seed} ---", flush=True)
    m_obs = SMOKE_M_OBS if smoke else M_OBS
    k_recog = SMOKE_K if smoke else K_RECOG
    k_vocab = SMOKE_K if smoke else K_VOCAB
    n_trials = SMOKE_TRIALS if smoke else N_TRIALS

    acts, words, _patterns = capture_seed(seed, smoke, m_obs)

    # Recognition -- reported separately, never folded into composition.
    d_act = acts[words[0]].shape[1]
    consolidated = {w: acts[w][:k_vocab].mean(axis=0) for w in words}
    rec_per_obs, rec_avg = recognition_accuracy(
        acts, words, consolidated, k_recog, np.random.default_rng(seed + 7))

    per_load = run_pipeline(seed, acts, words, LOADS, n_trials,
                            k_recog, k_vocab)

    for load in LOADS:
        e = per_load[load]
        print(f"  L={load}: integrated acc={e['integrated_accuracy']:.4f} "
              f"| composition-only acc={e['composition_only_accuracy']:.4f} "
              f"(n={e['n_composition_only']})", flush=True)
    print(f"  [seed {seed}] recognition (reported separately): "
          f"per-observation={rec_per_obs:.4f}, "
          f"temporally-averaged={rec_avg:.4f}", flush=True)

    return {
        "seed": seed,
        "smoke": bool(smoke),
        "n_concepts_captured": len(words),
        "activity_dim": int(d_act),
        "m_obs": int(m_obs),
        "recognition_accuracy": rec_avg,
        "recognition_per_observation": rec_per_obs,
        "recognition_temporally_averaged": rec_avg,
        "per_load": per_load,
    }


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Vocabulary scaling: the biologized grounded-"
                    "composition pipeline on a 64-concept G.20 sparse "
                    "bridge, against the frozen 0.80 bar.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny vocab subset + few observations on a "
                         "reduced-scale G.20 sparse bridge -- a fast "
                         "end-to-end check; toy numbers, NOT a result")
    args = ap.parse_args()

    smoke = bool(args.smoke)
    seeds = [42] if smoke else SEEDS

    print("=== vocabulary scaling: 64-concept G.20 sparse substrate "
          "-> biologized grounded-composition pipeline ===", flush=True)
    if smoke:
        print("  *** SMOKE MODE: reduced-scale bridge + tiny vocab "
              "subset, toy numbers, NOT a result ***", flush=True)
    print(f"concepts={N_CONCEPTS}; FHRR N_dim={N_DIM}; recognition "
          f"K={SMOKE_K if smoke else K_RECOG}; loads={LOADS}; "
          f"bar={BAR}; seeds={seeds}; "
          f"grounding=meancenter (biologized pipeline, reused)", flush=True)

    seed_results = [run_one_seed(s, smoke=smoke) for s in seeds]

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

    rec_per_obs = float(np.mean([r["recognition_per_observation"]
                                 for r in seed_results]))
    rec_avg = float(np.mean([r["recognition_temporally_averaged"]
                             for r in seed_results]))
    print(f"\nrecognition (reported separately + honestly): "
          f"per-observation mean={rec_per_obs:.4f}, "
          f"temporally-averaged mean={rec_avg:.4f}", flush=True)
    print("  (compare to the ~0.66 / ~0.93 measured at 16 concepts)",
          flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if smoke:
        verdict = "SMOKE"
        print("  SMOKE run -- toy numbers, not propagated as a result.",
              flush=True)
    elif all_pass:
        verdict = "VOCABULARY_SCALING_64CONCEPT_PASS"
        print("  The biologized grounded-composition pipeline clears the "
              "frozen 0.80 bar multi-seed at all loads on a 64-concept "
              "G.20 sparse vocabulary -- the compositional capability "
              "scales beyond the 16-concept toy vocabulary. Proceed to "
              "the 160/320-concept ensemble.", flush=True)
    else:
        verdict = "VOCABULARY_SCALING_64CONCEPT_BELOW_BAR"
        print("  Integrated multi-seed mean is below 0.80 at some load. "
              "The honest finding is which stage costs it -- compare "
              "integrated vs composition-only, and the separately-"
              "reported recognition accuracy.", flush=True)

    out = {
        "seeds": seeds, "n_concepts": N_CONCEPTS, "n_dim": N_DIM,
        "k_recog": SMOKE_K if smoke else K_RECOG, "loads": LOADS,
        "n_trials": SMOKE_TRIALS if smoke else N_TRIALS, "bar": BAR,
        "grounding": "meancenter", "smoke": smoke,
        "per_seed": seed_results,
        "aggregate": {str(k): v for k, v in agg.items()},
        "recognition_per_observation_mean": rec_per_obs,
        "recognition_temporally_averaged_mean": rec_avg,
        "verdict": verdict,
    }
    tag = "smoke" if smoke else "full"
    out_path = f"research/findings/raw/vocabulary_scaling_run_{tag}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
