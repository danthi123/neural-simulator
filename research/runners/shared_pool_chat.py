"""Chat REPL for the catalog G.20 shared-pool bridge.

Loads a pre-trained G.20 bridge (from concept_pool_demo_shared.py) and
provides natural-language interaction using slice-firing discrimination
instead of v16's lang_output cosine pathway.

Capabilities:
- 'remember X is Y' -> create engram tag combining concepts X and Y
- 'what is X' -> stim tag(s) containing X, return associates
- 'is X Y?' -> exact tag match
- 'tags' -> list stored tags
- 'concepts' -> list known concepts (vocab)
- 'stim X' -> stim concept X's engram, show which other concepts also fire
- 'quit' -> save bridge state + exit

Usage:
  python -m research.runners.shared_pool_chat \\
      --load-bridge research/findings/raw/g11_bg/shared_pool_n32_bridge.h5 \\
      --vocab "apple,river,dog,cat,go,come,stop,look,big,small,hot,cold,\\
               tree,bird,sun,moon,walk,run,eat,sleep,red,blue,fast,slow,\\
               house,road,fire,water,give,take,find,lose"
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from research.runners.concept_pool_demo_shared import (
    build_shared_pool_bridge,
    apply_shared_pool_topographic_prior,
)


def stim_recall_slice_rates(bridge, tag_name: str, n_concepts: int,
                              slice_size: int, drive_pA: float = 1500.0,
                              stim_steps: int = 100) -> np.ndarray:
    """Stim tag, return firing rate per slice in shared_concept_pool."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    shared_indices = list(rm.indices("shared_concept_pool"))
    slice_arrs = [
        cp.asarray(shared_indices[i * slice_size:(i + 1) * slice_size],
                    dtype=cp.int64)
        for i in range(n_concepts)
    ]

    bridge.stimulate_tag(tag_name, drive_pA=drive_pA)
    slice_rates = np.zeros(n_concepts, dtype=np.float32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        for j, sarr in enumerate(slice_arrs):
            firing = bridge.cp_firing_states[sarr]
            s = firing.sum() if hasattr(firing, 'sum') else 0
            if hasattr(s, 'item'):
                s = s.item()
            slice_rates[j] += float(s)
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    return slice_rates


def encode_pair_engram(bridge, word_a: str, word_b: str, vocab: List[str],
                        slice_size: int, n_lang_input: int, sparsity: float,
                        encoding_steps: int = 200,
                        teacher_pA: float = 500.0,
                        top_k: int = 100) -> str:
    """Encode (a, b) as a new engram tag. Drives lang_input for both
    words + teacher current on both slices."""
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    rm = bridge.region_manager

    a_idx = vocab.index(word_a)
    b_idx = vocab.index(word_b)

    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    shared_indices = list(rm.indices("shared_concept_pool"))
    slice_a = cp.asarray(shared_indices[a_idx*slice_size:(a_idx+1)*slice_size],
                          dtype=cp.int64)
    slice_b = cp.asarray(shared_indices[b_idx*slice_size:(b_idx+1)*slice_size],
                          dtype=cp.int64)

    drive_a = orthogonal_drive_pattern(
        cue_idx=a_idx, n_cues=len(vocab),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    drive_b = orthogonal_drive_pattern(
        cue_idx=b_idx, n_cues=len(vocab),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    combined = cp.asarray(drive_a + drive_b, dtype=cp.float32)
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)

    tag = f"{word_a}_{word_b}"
    bridge.start_engram_recording(tag)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr] = combined
        ext[slice_a] = teacher_pA
        ext[slice_b] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(10):
        bridge._run_one_simulation_step()
    bridge.commit_engram_tag(
        tag, top_k=top_k,
        region_filter=["shared_concept_pool"],
    )
    return tag


# ---------------------------------------------------------------------------
# Sparse-distributed (Kanerva SDM) analogues. The sparse bridges store each
# concept as a scattered K-of-N random pattern in the shared pool instead of
# a contiguous slice. `sparse_patterns[i]` is the list of pool-local indices
# for concept i (regenerated deterministically from the training seed via
# concept_pool_sparse_distributed.generate_sparse_patterns). These mirror the
# contiguous-slice helpers above but index scattered patterns, and match the
# VALIDATED sparse capture recipe (teacher-bias 100 pA, 100-step window,
# n_cues = vocab length, top_k 150).
# ---------------------------------------------------------------------------

def stim_recall_sparse_rates(bridge, tag_name: str,
                              sparse_patterns: List[List[int]],
                              drive_pA: float = 1500.0,
                              stim_steps: int = 100) -> np.ndarray:
    """Sparse analogue of stim_recall_slice_rates: stim the tag, return
    accumulated firing per SPARSE PATTERN in shared_concept_pool.

    Mirrors concept_pool_sparse_distributed.eval_sparse_discrimination's
    inner loop so demo recall reproduces the training-time discrimination.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    shared_indices = list(rm.indices("shared_concept_pool"))
    pattern_arrs = [
        cp.asarray([shared_indices[k] for k in pat], dtype=cp.int64)
        for pat in sparse_patterns
    ]

    bridge.stimulate_tag(tag_name, drive_pA=drive_pA)
    rates = np.zeros(len(sparse_patterns), dtype=np.float32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        for j, parr in enumerate(pattern_arrs):
            firing = bridge.cp_firing_states[parr]
            s = firing.sum() if hasattr(firing, 'sum') else 0
            if hasattr(s, 'item'):
                s = s.item()
            rates[j] += float(s)
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    return rates


def encode_partial_pair_engram_sparse(bridge, word_to_drive: str,
                                       tag_name: str, vocab: List[str],
                                       sparse_patterns: List[List[int]],
                                       n_lang_input: int, sparsity: float,
                                       encoding_steps: int = 100,
                                       teacher_pA: float = 100.0,
                                       top_k: int = 150) -> str:
    """Sparse analogue of encode_partial_pair_engram (cross-bridge single
    word under a shared tag name). Drives lang_input(word) + teacher-bias
    on the word's SPARSE PATTERN, commits `tag_name`."""
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    rm = bridge.region_manager

    if word_to_drive not in vocab:
        raise ValueError(f"word '{word_to_drive}' not in vocab")
    word_idx = vocab.index(word_to_drive)

    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    shared_indices = list(rm.indices("shared_concept_pool"))
    pattern_global = [shared_indices[k] for k in sparse_patterns[word_idx]]
    pattern_arr = cp.asarray(pattern_global, dtype=cp.int64)

    drive = orthogonal_drive_pattern(
        cue_idx=word_idx, n_cues=len(vocab),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    drive_arr = cp.asarray(drive, dtype=cp.float32)
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)

    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr] = drive_arr
        ext[pattern_arr] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(10):
        bridge._run_one_simulation_step()
    bridge.commit_engram_tag(
        tag_name, top_k=top_k,
        region_filter=["shared_concept_pool"],
    )
    return tag_name


def encode_pair_engram_sparse(bridge, word_a: str, word_b: str,
                               vocab: List[str],
                               sparse_patterns: List[List[int]],
                               n_lang_input: int, sparsity: float,
                               encoding_steps: int = 100,
                               teacher_pA: float = 100.0,
                               top_k: int = 150) -> str:
    """Sparse analogue of encode_pair_engram (both words live in ONE sparse
    bridge). Drives lang_input for both + teacher-bias on both sparse
    patterns, commits tag '<a>_<b>'."""
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    rm = bridge.region_manager

    a_idx = vocab.index(word_a)
    b_idx = vocab.index(word_b)

    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    shared_indices = list(rm.indices("shared_concept_pool"))
    pat_a = cp.asarray([shared_indices[k] for k in sparse_patterns[a_idx]],
                        dtype=cp.int64)
    pat_b = cp.asarray([shared_indices[k] for k in sparse_patterns[b_idx]],
                        dtype=cp.int64)

    drive_a = orthogonal_drive_pattern(
        cue_idx=a_idx, n_cues=len(vocab),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    drive_b = orthogonal_drive_pattern(
        cue_idx=b_idx, n_cues=len(vocab),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    combined = cp.asarray(drive_a + drive_b, dtype=cp.float32)
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)

    tag = f"{word_a}_{word_b}"
    bridge.start_engram_recording(tag)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr] = combined
        ext[pat_a] = teacher_pA
        ext[pat_b] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(10):
        bridge._run_one_simulation_step()
    bridge.commit_engram_tag(
        tag, top_k=top_k,
        region_filter=["shared_concept_pool"],
    )
    return tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab", type=str, required=True,
                    help="Comma-separated vocab list")
    p.add_argument("--n-lang-input", type=int, default=8192)
    p.add_argument("--n-shared-pool", type=int, default=1600)
    p.add_argument("--n-shared-fs", type=int, default=200)
    p.add_argument("--slice-size", type=int, default=50)
    p.add_argument("--sparsity", type=float, default=0.03)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--teacher-pA", type=float, default=500.0)
    p.add_argument("--drive-pA", type=float, default=1500.0)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--scripted", type=str, default=None)
    p.add_argument("--friendly", action="store_true")
    args = p.parse_args()

    vocab = [w.strip() for w in args.vocab.split(",") if w.strip()]
    n_concepts = len(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    print(f"=== G.20 shared-pool chat REPL ===", flush=True)
    print(f"  Vocab: {n_concepts} concepts ({vocab[:5]}...)", flush=True)
    print(f"  Bridge: {args.load_bridge}", flush=True)
    print(flush=True)

    bridge = build_shared_pool_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_shared_pool=args.n_shared_pool,
        n_shared_fs=args.n_shared_fs,
        n_lang_output=args.n_lang_input,
        verbose=False,
    )
    bridge.load_checkpoint(args.load_bridge)
    print(f"[loaded bridge]", flush=True)

    # Restore tags from checkpoint
    restored = sorted([t["name"] for t in bridge.list_engram_tags()])
    encoded_tags: List[str] = list(restored)
    print(f"[restored {len(restored)} engram tag(s)]", flush=True)
    print(flush=True)

    def query_concept(word):
        """Stim word's own engram + any pair engrams; aggregate slice
        firing across all tags."""
        if word not in word_to_idx:
            if args.friendly:
                print(f"  I don't know '{word}'.", flush=True)
            else:
                print(f"  [unknown concept: {word}]", flush=True)
            return
        # Find tags containing this word
        # Word's own engram tag = word name (if exists)
        # Pair engram tags = 'word_other' or 'other_word'
        matches = [t for t in encoded_tags if word in t.split("_")]
        if word in encoded_tags and word not in matches:
            matches.append(word)
        if not matches:
            if args.friendly:
                print(f"  I don't know anything about '{word}' yet.",
                      flush=True)
            else:
                print(f"  [no tags contain '{word}']", flush=True)
            return

        aggregated = np.zeros(n_concepts, dtype=np.float32)
        for tag in matches:
            rates = stim_recall_slice_rates(
                bridge, tag, n_concepts=n_concepts,
                slice_size=args.slice_size,
                drive_pA=args.drive_pA, stim_steps=args.drive_steps,
            )
            aggregated += rates
        sorted_idx = np.argsort(-aggregated)
        top5 = [(vocab[i], float(aggregated[i])) for i in sorted_idx[:5]]
        # Filter out the query word itself
        associates = [(w, s) for w, s in top5 if w != word][:4]
        if args.friendly:
            if not associates:
                print(f"  '{word}' has no associates.", flush=True)
            else:
                summaries = [f"{w} ({s:.0f})" for w, s in associates]
                print(f"  {word.capitalize()} is associated with: "
                      f"{', '.join(summaries)}.", flush=True)
        else:
            print(f"  '{word}' associates (from {len(matches)} tag(s)):",
                  flush=True)
            for w, s in associates:
                print(f"    {w:12} {s:.0f}", flush=True)

    def dispatch(line):
        line = line.strip().lower()
        if not line or line in ("quit", "exit"):
            return "EXIT"
        if line in ("concepts", "vocab", "/vocab"):
            print(f"  concepts ({n_concepts}): {vocab}", flush=True)
            return None
        if line in ("tags", "/tags"):
            print(f"  tags ({len(encoded_tags)}): {encoded_tags}",
                  flush=True)
            return None
        if line.startswith("remember "):
            rest = line[len("remember "):].strip()
            if " is " in rest:
                a, b = rest.split(" is ", 1)
                a, b = a.strip(), b.strip()
            else:
                parts = rest.split()
                if len(parts) != 2:
                    print("  [usage: remember a is b]", flush=True)
                    return None
                a, b = parts
            if a not in word_to_idx:
                print(f"  [unknown concept: {a}]", flush=True)
                return None
            if b not in word_to_idx:
                print(f"  [unknown concept: {b}]", flush=True)
                return None
            tag = encode_pair_engram(
                bridge, a, b, vocab=vocab,
                slice_size=args.slice_size,
                n_lang_input=args.n_lang_input,
                sparsity=args.sparsity,
                encoding_steps=args.encoding_steps,
                teacher_pA=args.teacher_pA,
                top_k=args.top_k,
            )
            encoded_tags.append(tag)
            if args.friendly:
                print(f"  OK, I'll remember {a} is {b}.", flush=True)
            else:
                print(f"  [encoded tag '{tag}']", flush=True)
            return None
        if line.startswith("what is "):
            word = line[len("what is "):].strip()
            query_concept(word)
            return None
        if line.startswith("is "):
            rest = line.rstrip("?").strip()[len("is "):]
            parts = rest.split()
            if len(parts) == 2:
                tag = f"{parts[0]}_{parts[1]}"
                if tag in encoded_tags:
                    if args.friendly:
                        print(f"  Yes, {parts[0]} is {parts[1]}.",
                              flush=True)
                    else:
                        print(f"  YES (tag '{tag}' exists)", flush=True)
                else:
                    if args.friendly:
                        print(f"  I don't know.", flush=True)
                    else:
                        print(f"  UNKNOWN", flush=True)
            return None
        if line.startswith("stim "):
            word = line[len("stim "):].strip()
            if word not in encoded_tags:
                print(f"  [no tag named '{word}']", flush=True)
                return None
            rates = stim_recall_slice_rates(
                bridge, word, n_concepts=n_concepts,
                slice_size=args.slice_size,
                drive_pA=args.drive_pA, stim_steps=args.drive_steps,
            )
            sorted_idx = np.argsort(-rates)
            print(f"  Stim of '{word}' -> top 5 slices:", flush=True)
            for i in sorted_idx[:5]:
                print(f"    {vocab[i]:12} rate={rates[i]:.0f}",
                      flush=True)
            return None
        # Plain word -> query
        query_concept(line)
        return None

    print("Commands:")
    print("  remember a is b           Encode engram for (a, b)")
    print("  what is X                 Find associates of X")
    print("  <word>                    Same as 'what is'")
    print("  is X Y?                   Exact tag match")
    print("  stim <tag>                Stim a specific tag; show top slices")
    print("  concepts / vocab          List known concepts")
    print("  tags                      List stored tags")
    print("  quit                      Exit")
    print()

    if args.scripted:
        for inp in [s.strip() for s in args.scripted.split(",") if s.strip()]:
            print(f"\n> {inp}", flush=True)
            if dispatch(inp) == "EXIT":
                break
    else:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if dispatch(line) == "EXIT":
                break
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
