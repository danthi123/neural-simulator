"""Direction A ENCODING SMOKE: 5-min verification that the corrected
encoding mechanism produces non-zero engram tags on an untrained bridge.

The earlier broken Direction A smoke used region_filter=["ca3"] (CA3
doesn't exist in this substrate) and no teacher current -> 0-neuron
engram tags. The corrected mechanism uses (a) concept pool region
filter and (b) teacher current on target pool per slot. On an
UNTRAINED bridge the teacher alone should ignite the target pools and
the engram tag should pick up >= ~50 of the top ~100 from the pool.

If n_tagged > 0 on every encoding -> recipe is sound, launch full
scale. If 0 -> deeper instrument problem to fix before GPU run.

~5 min wall on small substrate (no training).
"""
from __future__ import annotations
import os, sys, time
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
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _build_region_filter, _encode_sequence_with_ec_context,
    K_PAIRS, SLOT_COUNT,
)
from sim.backend import get_backend, is_gpu_backend


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction A encoding smoke (instrument-only) ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    seed = 42

    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)

    t0 = time.time()
    bridge = build_concept_bridge(
        seed=seed,
        n_lang_input=2048,
        n_per_pool=200,
        n_fs_per_pool=24,
        enable_adjective=True,
        weak_dynamics=True,
        enable_positional_context=True,
        n_ec_context=200,
        verbose=False,
    )
    print(f"  built UNTRAINED bridge in {(time.time()-t0):.1f}s",
          flush=True)
    rm = bridge.region_manager
    region_filter = _build_region_filter(rm)
    print(f"  region_filter ({len(region_filter)} regions): "
          f"{region_filter}", flush=True)
    if len(region_filter) == 0:
        print("  [FATAL] empty region_filter -> recipe broken at "
              "the substrate level", flush=True)
        return 1

    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=n_words,
        slot_count=SLOT_COUNT, vocab=words)
    print(f"  testing {K_PAIRS} sequences x {SLOT_COUNT} slots",
          flush=True)

    n_tagged_per_seq = []
    for seq_idx, seq in enumerate(sequences):
        t1 = time.time()
        tag, stats = _encode_sequence_with_ec_context(
            bridge, seq, words, seq_idx, region_filter)
        n_tagged = stats.get("n_tagged", 0)
        n_tagged_per_seq.append(n_tagged)
        elapsed = time.time() - t1
        print(f"  seq {seq_idx}: {list(seq)}; n_tagged={n_tagged}"
              f" ({elapsed:.1f}s)", flush=True)

    print(f"\n  TOTAL wall: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"\n=== VERDICT ===", flush=True)
    mean_tagged = float(np.mean(n_tagged_per_seq))
    min_tagged = int(np.min(n_tagged_per_seq))
    print(f"  n_tagged per seq: {n_tagged_per_seq}", flush=True)
    print(f"  mean = {mean_tagged:.1f}, min = {min_tagged}",
          flush=True)
    if min_tagged > 0:
        print(f"  RECIPE_SOUND: every sequence engrams >= {min_tagged} "
              f"neurons; full-scale launch is safe.", flush=True)
        return 0
    else:
        print(f"  RECIPE_BROKEN: {n_tagged_per_seq.count(0)}/{K_PAIRS}"
              f" sequences have 0-neuron engrams; further fix needed"
              f" before GPU run.", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
