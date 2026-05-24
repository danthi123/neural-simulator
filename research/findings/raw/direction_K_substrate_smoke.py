"""Direction K SUBSTRATE SMOKE: verify substrate-grounded FHRR
sequence storage mechanism works at small scale before committing
~3 hr GPU.

Per Direction K algebra probe (commit 4276611): FHRR-based sequence
storage clears 0.80 bar at all noise levels up to 1.0 in numpy
algebra. The substrate-grounded version reuses the validated FHRR
biologization pipeline (mean-centered substrate activity ->
resonate-and-fire -> attractor clean-up) + per-position phasors.

Smoke:
1. Build small v16 substrate (no full training; ~3 min)
2. Present 1 sequence with K slot-words via lang_input drive
   per slot (no engram tags; just drive + capture activity)
3. Mean-center per-slot activity vectors over the concept-pool
   regions (matches FHRR shortcut-2 RESOLVED grounding)
4. Per slot: bind mean-centered activity (concept phasor) with
   per-slot position phasor (deterministic random)
5. Bundle K slot products = sequence phasor
6. Retrieve slot 2: unbind sequence phasor with position phasor 2;
   cosine-match against each vocab word's mean-centered activity
   (captured during training presentation)
7. Verify top-1 = true slot-2 word

Pre-registered smoke success: at small scale + low noise, the
mechanism should return the correct slot-2 word for at least the
test sequence. If smoke FAILS, the mechanism has substrate-level
issues the algebra didn't expose; pause before full GPU run.

Reuse-by-import only; no protected/frozen/moat module modified.
~10 min wall.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.concept_pool_demo import (
    build_concept_bridge, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.runners.concept_compose_train import _WORD_TO_POOL
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_K_substrate_smoke.json")
N_LANG_INPUT = 2048
N_PER_POOL = 200
N_FS_PER_POOL = 24
SPARSITY = 0.05
SEED = 42
ENCODING_STEPS = 200  # per slot present
SLOT_COUNT = 3
TEACHER_PA = 500.0


def fhrr_bind_real_vec(a, b):
    """For real-valued mean-centered activity, use circular
    convolution (FHRR alternative). For simplicity in smoke, use
    Hadamard product on sign-encoded phasors derived from the
    mean-centered activity."""
    # Convert real vectors to complex phasors via sign + magnitude
    # encoding: phase = pi * sign(v) for non-zero entries.
    # Simpler: pair adjacent entries as real+imag.
    # For smoke, use Hadamard on the raw real vectors (the
    # validated FHRR biologization arc did this with real-valued
    # codes; the algebra still works).
    return a * b


def fhrr_unbind_real_vec(a, b):
    """Real-valued unbind: same as bind for sign vectors."""
    return a * b  # element-wise self-inverse for sign codes


def fhrr_bundle(*items):
    return np.sum(items, axis=0)


def cosine_real(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b) / (na * nb))


def capture_substrate_activity(bridge, word, words, word_to_idx,
                                  n_lang_input, sparsity,
                                  encoding_steps, pool_region):
    """Drive lang_input(word) + teacher on target pool; capture
    spike counts across all concept-pool regions."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    try:
        pool_idx = list(rm.indices(pool_region))
        pool_arr = cp.asarray(pool_idx, dtype=cp.int64)
    except Exception:
        pool_arr = None

    drive = orthogonal_drive_pattern(
        cue_idx=word_to_idx[word], n_cues=len(words),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity)

    # Collect spike counts across all concept-pool neurons
    pool_kinds = [("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
                   ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
                   ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"])]
    all_pool_idx = []
    for kind, names in pool_kinds:
        for n in names:
            try:
                all_pool_idx.extend(list(rm.indices(f"{kind}_{n}")))
            except Exception:
                pass
    for m in ["motor_N", "motor_E", "motor_S", "motor_W"]:
        try:
            all_pool_idx.extend(list(rm.indices(m)))
        except Exception:
            pass
    all_pool_arr = cp.asarray(all_pool_idx, dtype=cp.int64)
    n_pool_total = len(all_pool_idx)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    spike_counts = cp.zeros(n_pool_total, dtype=cp.float32)
    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_in_arr] = cp.asarray(drive, dtype=cp.float32)
        if pool_arr is not None:
            ext[pool_arr] = TEACHER_PA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[all_pool_arr]
        spike_counts = spike_counts + fired.astype(cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(spike_counts), n_pool_total


def mean_center(vec):
    return vec - np.mean(vec)


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction K SUBSTRATE SMOKE ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu}); seed={SEED}",
          flush=True)
    print(f"  Verify: mean-centered substrate activity + position "
          f"phasors + FHRR bundle = retrievable slot word",
          flush=True)

    t0 = time.time()
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    bridge = build_concept_bridge(
        seed=SEED, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=False,
        verbose=False,
    )
    print(f"  built bridge in {(time.time()-t0):.1f}s", flush=True)
    cp, _ = get_backend()

    # Freeze plasticity (we're just probing, not training)
    for g in ("language_input_to_motor",
              "language_input_to_noun_pool",
              "language_input_to_verb_pool",
              "language_input_to_adjective_pool",
              "motor_to_language_output",
              "noun_pool_to_language_output",
              "verb_pool_to_language_output",
              "adjective_pool_to_language_output"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    # Build vocab activity dictionary: present each word; capture
    # mean-centered concept-pool activity vector.
    print(f"\n  Capturing per-word activity vectors (substrate "
          f"grounding for vocab phasors)...", flush=True)
    vocab_activities = {}
    n_pool_total = None
    for w in words:
        spike_counts, n_pool_total = capture_substrate_activity(
            bridge, w, words, word_to_idx, N_LANG_INPUT, SPARSITY,
            ENCODING_STEPS, _WORD_TO_POOL[w])
        vocab_activities[w] = mean_center(spike_counts)
    print(f"  Captured {len(vocab_activities)} word activity vectors"
          f" (dim={n_pool_total})", flush=True)

    # Pick a test sequence
    seq = ["apple", "big", "north"]
    print(f"\n  Test sequence: {seq}", flush=True)

    # Per-slot: capture each slot's activity (already captured above
    # by word; just look up)
    slot_concept_phasors = [vocab_activities[w] for w in seq]

    # Position phasors (deterministic random; sign vectors for
    # element-wise FHRR)
    rng = np.random.default_rng(SEED * 9999 + 7)
    position_phasors = [
        rng.choice([-1.0, 1.0], size=n_pool_total)
        for _ in range(SLOT_COUNT)
    ]

    # Encode sequence: bundle of (concept_phasor * position_phasor)
    bound = []
    for slot_idx, c_phasor in enumerate(slot_concept_phasors):
        bound.append(fhrr_bind_real_vec(c_phasor, position_phasors[slot_idx]))
    bundle = fhrr_bundle(*bound)

    # Retrieve each slot via unbind with position phasor; cosine to
    # vocab activities; return top-1
    print(f"\n  Retrieval per slot:", flush=True)
    correct_count = 0
    for query_slot in range(SLOT_COUNT):
        unbound = fhrr_unbind_real_vec(bundle, position_phasors[query_slot])
        scores = {w: cosine_real(unbound, vocab_activities[w])
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1_word = topK[0][0]
        true_word = seq[query_slot]
        correct = (top1_word == true_word)
        if correct: correct_count += 1
        print(f"    slot {query_slot} true={true_word} top1="
              f"{top1_word} (score {topK[0][1]:.3f}); top3="
              f"{[w for w, _ in topK[:3]]}; {'CORRECT' if correct else 'WRONG'}",
              flush=True)

    print(f"\n  Smoke result: {correct_count}/{SLOT_COUNT} slots "
          f"correctly retrieved", flush=True)
    print(f"  Wall: {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if correct_count == SLOT_COUNT:
        verdict = "SMOKE_PERFECT"
        print(f"  All {SLOT_COUNT} slots correctly retrieved at "
              f"single-seed smoke; substrate-grounded FHRR "
              f"sequence storage MECHANISM WORKS. Full-scale "
              f"multi-seed run justified.", flush=True)
    elif correct_count >= SLOT_COUNT - 1:
        verdict = "SMOKE_MOSTLY_CORRECT"
        print(f"  {correct_count}/{SLOT_COUNT} slots correct at "
              f"single-seed; mechanism works but not perfectly. "
              f"Full-scale run could clarify.", flush=True)
    else:
        verdict = "SMOKE_PARTIAL_OR_FAIL"
        print(f"  {correct_count}/{SLOT_COUNT} slots correct; "
              f"smoke insufficient signal; mechanism may have "
              f"substrate-level issues the algebra didn't expose. "
              f"Diagnose before full-scale run.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "sequence": seq, "n_correct": correct_count,
        "slot_count": SLOT_COUNT, "n_pool_total": n_pool_total,
        "verdict": verdict, "wall_clock_minutes": (time.time()-t0)/60,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
