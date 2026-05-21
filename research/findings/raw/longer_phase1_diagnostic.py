"""Longer-Phase-1 training diagnostic.

The 8-architecture convergent ceiling shows the substrate is
asymptotically capped at N=3 full_acc ~0.458 (6th arc local optimum)
with the standard 200-events-per-word Phase-1 training. This
diagnostic tests whether MORE Phase-1 training shifts the ceiling:
if longer-trained substrate produces higher full_acc at N=3, the
substrate benefits from more training; if not, the ceiling is
asymptotic under any reasonable Phase-1 regime.

Cheap-first: single seed (42); 4x training events (800 events/word
vs 200); ~70 min wall-clock estimate; if positive, commit to
multi-seed (3 seeds × 70 min = ~3.5 hr) and full 9th arc cycle.

PROTOCOL:
1. Generate longer-trained Phase-1 checkpoint at seed 42 into a new
   cache directory (preserves existing 200-event cache byte-stable).
2. Run the 6th arc decisive eval on the longer-trained checkpoint
   (single-seed; N=3 only).
3. Report whether N=3 full_acc > 0.50 (clear improvement above
   6th arc baseline 0.458) or ~0.458 (asymptotic).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

import research.runners.unified_per_regime_monitor_runner as urr
import research.runners.concept_pool_demo as cpd
from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _phase1_train_kwargs,
)


SEED = 42
EVENTS_PER_WORD = 800  # 4x the standard 200
OUT_CACHE_DIR = "research/findings/raw/unified_per_regime/phase1_800ev"


def train_longer_phase1(seed: int, n_train_events: int, cache_dir: str):
    """Replicate `_phase1_train_if_needed` logic with n_train_events
    overridden. Saves to cache_dir.
    """
    cache_path = _phase1_cache_path(cache_dir, seed)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Cache already exists at {cache_path}; skipping training.")
        return cache_path

    print(f"=== Phase-1 training at {n_train_events} events/word ===")
    print(f"Seed: {seed}; cache_dir: {cache_dir}")

    # tiny_synth=False (biological scale)
    train_kwargs = _phase1_train_kwargs(False)

    # Build substrate
    bridge = _build_bridge_with_phase1_recipe(int(seed), False)

    # word_to_idx mapping (matches v14/v16 ordering)
    all_words_ordered = (
        list(cpd.DIRECTION_VOCAB)
        + list(cpd.NOUN_VOCAB)
        + list(cpd.VERB_VOCAB)
        + list(cpd.ADJECTIVE_VOCAB)
    )
    word_to_idx = {w: i for i, w in enumerate(all_words_ordered)}
    n_words_total = len(all_words_ordered)

    # Topographic bias (REUSED)
    cpd.apply_concept_topographic_bias(
        bridge,
        n_lang_input=int(train_kwargs["n_lang_input"]),
        topographic_factor=float(train_kwargs["topographic_factor"]),
        off_target_factor=float(train_kwargs["off_target_factor"]),
        sparsity=float(train_kwargs["sparsity"]),
        orthogonal_codes=bool(train_kwargs["orthogonal_codes"]),
        n_words_for_orthogonal=int(n_words_total),
        word_to_idx=word_to_idx,
        skip_motor=False,
        verbose=False,
    )

    # (word, target_pool) schedule
    all_targets = []
    for word, action in cpd.DIRECTION_VOCAB.items():
        all_targets.append((word, f"motor_{action}"))
    for word, name in cpd.NOUN_VOCAB.items():
        all_targets.append((word, f"noun_pool_{name}"))
    for word, name in cpd.VERB_VOCAB.items():
        all_targets.append((word, f"verb_pool_{name}"))
    for word, name in cpd.ADJECTIVE_VOCAB.items():
        all_targets.append((word, f"adjective_pool_{name}"))

    print(f"Vocab: {len(all_targets)} (word, pool) targets")
    print(f"Total events: {len(all_targets)} × {n_train_events} = "
          f"{len(all_targets) * n_train_events}")

    # Interleaved training (OVERRIDDEN n_train_events)
    rng = np.random.default_rng(int(seed))
    buffer = []
    for word, target in all_targets:
        for _ in range(n_train_events):
            buffer.append((word, target))
    rng.shuffle(buffer)

    print(f"Training buffer: {len(buffer)} events; starting train loop...")
    t_start = time.time()
    last_print = t_start
    for i, (word, target) in enumerate(buffer):
        cpd.train_word_to_pool(
            bridge,
            word,
            target,
            n_events=1,
            reset_steps=50,
            n_lang_input=int(train_kwargs["n_lang_input"]),
            n_lang_output=int(train_kwargs["n_lang_input"]),
            sparsity=float(train_kwargs["sparsity"]),
            orthogonal_codes=bool(train_kwargs["orthogonal_codes"]),
            n_words_for_orthogonal=int(n_words_total),
            word_to_idx=word_to_idx,
            verbose=False,
        )
        now = time.time()
        if now - last_print > 60:
            elapsed = now - t_start
            rate = (i + 1) / elapsed
            eta = (len(buffer) - i - 1) / rate / 60.0
            print(f"  step {i+1}/{len(buffer)} "
                  f"({100.0*(i+1)/len(buffer):.1f}%); elapsed {elapsed/60.0:.1f}min; "
                  f"ETA {eta:.1f}min")
            last_print = now

    elapsed_total = time.time() - t_start
    print(f"Training complete; {elapsed_total/60.0:.1f}min wall-clock")

    # Save checkpoint
    print(f"Saving checkpoint to {cache_path}...")
    bridge.save_checkpoint(str(cache_path))
    print(f"Saved.")
    return cache_path


def main():
    cache_path = train_longer_phase1(SEED, EVENTS_PER_WORD, OUT_CACHE_DIR)
    print(f"\nLonger-Phase-1 checkpoint ready: {cache_path}")
    print(
        "\nNext step: invoke the 6th arc decisive eval pointing at this "
        "cache dir:\n"
        f"  python -m research.runners.generative_replay_pfc_frame_runner "
        f"--seeds {SEED} --phase1-cache-dir {OUT_CACHE_DIR} "
        f"--ckpt research/findings/raw/longer_phase1_decisive.ckpt "
        f"--out research/findings/raw/longer_phase1_decisive.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
