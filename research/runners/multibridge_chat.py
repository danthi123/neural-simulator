"""Multi-bridge chat REPL — scale vocabulary via ensemble of v16 bridges.

Each v16 bridge has 90% multi-seed multitag at 16-word vocab (12 concept
words). Multiple bridges with different vocabs = expanded total vocab
without architectural rework.

Mechanism:
- Each bridge has its own engram tags (in its own HDF5 checkpoint)
- Chat REPL maintains list of (bridge, vocab) pairs
- On 'remember X is Y', route to bridge containing X (or first bridge
  with capacity)
- On 'what is X', route to bridge containing X
- Tags get bridge-id prefix to avoid collision

For now, this uses the EXISTING v16 bridges (all same vocab). Real
expansion would require training separate bridges with different vocab
tables. This is a SCAFFOLDING showing the multi-bridge mechanism.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import research.runners.concept_pool_demo as cpd
from research.runners.compose_concept_engram import (
    lang_output_pattern_during_stim,
)
from sim.text_embeddings import orthogonal_drive_pattern
import numpy as np


def cosine_to_word_with_vocab(pattern, target_word, n_lang_out,
                                word_to_idx, n_words_for_orthogonal=16,
                                sparsity=0.05):
    """Cosine to spelling pattern using PER-BRIDGE word_to_idx.

    The original cosine_to_word in compose_concept_engram uses a global
    _WORD_TO_IDX, which makes it impossible to query across bridges with
    different vocabularies. This version takes the mapping as arg.
    """
    if target_word not in word_to_idx:
        return 0.0
    target_pat = orthogonal_drive_pattern(
        cue_idx=word_to_idx[target_word], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_out, drive_max_pA=1.0, sparsity=sparsity,
    )
    a = float(np.linalg.norm(pattern))
    b = float(np.linalg.norm(target_pat))
    if a == 0 or b == 0:
        return 0.0
    return float(np.dot(pattern, target_pat) / (a * b))


# Vocabulary sets. Each bridge maps its 12 concept words to its
# specific pool names. word_to_idx for orthogonal_drive_pattern uses
# the bridge's local index (0-15 with 4 motors at front).
SET1_VOCAB = {
    "word_to_idx": {
        "north": 0, "east": 1, "south": 2, "west": 3,
        "apple": 4, "river": 5, "dog": 6, "cat": 7,
        "go": 8, "come": 9, "stop": 10, "look": 11,
        "big": 12, "small": 13, "hot": 14, "cold": 15,
    },
    "word_to_pool": {
        "apple": "noun_pool_APPLE", "river": "noun_pool_RIVER",
        "dog": "noun_pool_DOG", "cat": "noun_pool_CAT",
        "go": "verb_pool_GO", "come": "verb_pool_COME",
        "stop": "verb_pool_STOP", "look": "verb_pool_LOOK",
        "big": "adjective_pool_BIG", "small": "adjective_pool_SMALL",
        "hot": "adjective_pool_HOT", "cold": "adjective_pool_COLD",
    },
    "concept_words": ["apple","river","dog","cat","go","come","stop","look",
                       "big","small","hot","cold"],
}
SET2_VOCAB = {
    "word_to_idx": {
        "north": 0, "east": 1, "south": 2, "west": 3,
        "tree": 4, "bird": 5, "sun": 6, "moon": 7,
        "walk": 8, "run": 9, "eat": 10, "sleep": 11,
        "red": 12, "blue": 13, "fast": 14, "slow": 15,
    },
    "word_to_pool": {
        "tree": "noun_pool_TREE", "bird": "noun_pool_BIRD",
        "sun": "noun_pool_SUN", "moon": "noun_pool_MOON",
        "walk": "verb_pool_WALK", "run": "verb_pool_RUN",
        "eat": "verb_pool_EAT", "sleep": "verb_pool_SLEEP",
        "red": "adjective_pool_RED", "blue": "adjective_pool_BLUE",
        "fast": "adjective_pool_FAST", "slow": "adjective_pool_SLOW",
    },
    "concept_words": ["tree","bird","sun","moon","walk","run","eat","sleep",
                       "red","blue","fast","slow"],
}


class BridgeMember:
    """One bridge in the ensemble. Holds bridge + vocab metadata
    (word→idx, word→pool, concept_words) so encode/query can use the
    correct per-bridge mappings.
    """
    def __init__(self, bridge_path, vocab_set, n_lang_input, n_per_pool,
                 n_fs_per_pool, sparsity, n_words_for_orthogonal,
                 encoding_steps, balanced_teacher_pA, top_k, name):
        self.bridge_path = bridge_path
        self.word_to_idx = vocab_set["word_to_idx"]
        self.word_to_pool = vocab_set["word_to_pool"]
        self.concept_words = vocab_set["concept_words"]
        self.vocab = set(self.concept_words)  # for routing
        self.n_lang_input = n_lang_input
        self.n_per_pool = n_per_pool
        self.n_fs_per_pool = n_fs_per_pool
        self.sparsity = sparsity
        self.n_words_for_orthogonal = n_words_for_orthogonal
        self.encoding_steps = encoding_steps
        self.balanced_teacher_pA = balanced_teacher_pA
        self.top_k = top_k
        self.name = name
        self.bridge = None
        self.region_filter = []
        self.encoded_tags = []

    def load(self, seed, vocab_set_module=None):
        """Lazily load the bridge from disk.

        vocab_set_module: optional module to import BEFORE building the
        bridge, which can monkey-patch concept_pool_demo's vocab tables.
        Required for set2 bridges where the bridge architecture uses
        the set2 pool names (TREE/BIRD/...) instead of v16 (APPLE/RIVER/...).
        """
        if self.bridge is not None:
            return
        if vocab_set_module:
            print(f"  [applying vocab patch from {vocab_set_module}]",
                  flush=True)
            __import__(vocab_set_module)
        print(f"  [loading bridge '{self.name}' from {self.bridge_path}]",
              flush=True)
        self.bridge = cpd.build_concept_bridge(
            seed=seed,
            n_lang_input=self.n_lang_input,
            n_per_pool=self.n_per_pool,
            n_fs_per_pool=self.n_fs_per_pool,
            enable_adjective=True,
            weak_dynamics=True,
            enable_direct_verb_to_motor=True,
            verbose=False,
        )
        self.bridge.load_checkpoint(self.bridge_path)
        rm = self.bridge.region_manager
        # Derive pool name set from this bridge's word_to_pool mapping
        pool_names = set(self.word_to_pool.values())
        for pool_name in pool_names:
            try:
                rm.indices(pool_name)
                self.region_filter.append(pool_name)
            except Exception:
                pass
        # Restore tags from checkpoint
        restored = sorted([t["name"] for t in self.bridge.list_engram_tags()])
        self.encoded_tags = list(restored)
        if restored:
            print(f"  [restored {len(restored)} engram tag(s) from {self.name}]",
                  flush=True)


def find_bridge_for_word(members, word):
    """Return the bridge member whose vocab contains the word, or None."""
    for m in members:
        if word in m.vocab:
            return m
    return None


def find_bridges_for_words(members, words):
    """Find bridges that have ALL words in their vocab. Returns first match."""
    for m in members:
        if all(w in m.vocab for w in words):
            return m
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridges", nargs="+", required=True,
                    help="List of bridge .simstate.h5 paths")
    p.add_argument("--names", nargs="+", default=None,
                    help="Optional names for bridges (else uses filenames)")
    p.add_argument("--vocab-sets", nargs="+", default=None,
                    help="Per-bridge vocab set: 'set1' or 'set2'. "
                    "Defaults to set1 for all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--n-words-for-orthogonal", type=int, default=16)
    p.add_argument("--encoding-steps", type=int, default=500)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--balanced-teacher-pA", type=float, default=500.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--scripted", type=str, default=None)
    args = p.parse_args()

    names = args.names or [Path(bp).stem for bp in args.bridges]
    if len(names) != len(args.bridges):
        print(f"ERROR: --names length must match --bridges length", flush=True)
        return
    vocab_set_names = args.vocab_sets or ["set1"] * len(args.bridges)
    if len(vocab_set_names) != len(args.bridges):
        print(f"ERROR: --vocab-sets length must match --bridges length", flush=True)
        return

    SET_NAME_TO_VOCAB = {"set1": SET1_VOCAB, "set2": SET2_VOCAB}
    SET_NAME_TO_PATCH_MODULE = {
        "set1": None,
        "set2": "research.runners.concept_pool_demo_set2",
    }

    members = []
    for path, name, vs in zip(args.bridges, names, vocab_set_names):
        if vs not in SET_NAME_TO_VOCAB:
            print(f"ERROR: unknown vocab set '{vs}'", flush=True)
            return
        m = BridgeMember(
            bridge_path=path,
            vocab_set=SET_NAME_TO_VOCAB[vs],
            n_lang_input=args.n_lang_input,
            n_per_pool=args.n_per_pool,
            n_fs_per_pool=args.n_fs_per_pool,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            encoding_steps=args.encoding_steps,
            balanced_teacher_pA=args.balanced_teacher_pA,
            top_k=args.top_k,
            name=name,
        )
        m.vocab_set_module = SET_NAME_TO_PATCH_MODULE[vs]
        members.append(m)

    print(f"=== Multi-bridge chat REPL ===")
    print(f"  Bridges: {[m.name for m in members]}")
    print(f"  Total vocab: {sum(len(m.vocab) for m in members)} word-slots "
          f"across {len(members)} bridges")
    print(f"  (Note: until bridges have different vocabs, this is "
          f"redundant capacity)")
    print()

    # Load all bridges upfront, applying their vocab set patches
    for m in members:
        m.load(args.seed, vocab_set_module=m.vocab_set_module)

    def encode_to_bridge(m, a, b):
        """Encode (a, b) pair on bridge member m using m's vocab mapping.

        Custom implementation (not encode_concept_pair) so we can pass
        m.word_to_idx + m.word_to_pool instead of global imports.
        """
        from sim.backend import get_backend
        cp, _ = get_backend()
        tag = f"{a}_{b}"
        if tag in m.encoded_tags:
            return f"already remembered in {m.name}"
        bridge = m.bridge
        rm = bridge.region_manager
        drive_a = orthogonal_drive_pattern(
            cue_idx=m.word_to_idx[a],
            n_cues=args.n_words_for_orthogonal,
            n_neurons=args.n_lang_input,
            drive_max_pA=200.0, sparsity=args.sparsity,
        )
        drive_b = orthogonal_drive_pattern(
            cue_idx=m.word_to_idx[b],
            n_cues=args.n_words_for_orthogonal,
            n_neurons=args.n_lang_input,
            drive_max_pA=200.0, sparsity=args.sparsity,
        )
        combined = cp.asarray(drive_a + drive_b, dtype=cp.float32)
        lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
        pool_a = cp.asarray(list(rm.indices(m.word_to_pool[a])), dtype=cp.int64)
        pool_b = cp.asarray(list(rm.indices(m.word_to_pool[b])), dtype=cp.int64)
        n_total = bridge.cp_external_input_current.shape[0]

        bridge.start_engram_recording(tag)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
        ext = cp.zeros(n_total, dtype=cp.float32)
        for _ in range(args.encoding_steps):
            ext.fill(0)
            ext[lang_arr] = combined
            ext[pool_a] = args.balanced_teacher_pA
            ext[pool_b] = args.balanced_teacher_pA
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()
        bridge.commit_engram_tag(tag, top_k=args.top_k,
                                    region_filter=m.region_filter)
        m.encoded_tags.append(tag)
        return tag

    def query_word(word):
        """Multi-tag retrieval across all bridges. Returns top associates
        aggregated across all bridges that have tags with this word."""
        # Check if any bridge has this word
        if not any(word in m.vocab for m in members):
            return {"matches": [], "results": []}
        all_results = []
        for m in members:
            if word not in m.vocab:
                continue
            matches = [t for t in m.encoded_tags if word in t.split("_")]
            for tag in matches:
                pat, n_lo = lang_output_pattern_during_stim(
                    m.bridge, tag, drive_pA=1500.0,
                    stim_steps=args.drive_steps,
                )
                # Use this bridge's word_to_idx + concept_words
                for w in m.concept_words:
                    if w == word:
                        continue
                    score = cosine_to_word_with_vocab(
                        pat, w, n_lo,
                        word_to_idx=m.word_to_idx,
                        n_words_for_orthogonal=args.n_words_for_orthogonal,
                        sparsity=args.sparsity,
                    )
                    all_results.append({
                        "word": w, "score": score, "tag": tag,
                        "bridge": m.name,
                    })
        # Aggregate: max score per associate word
        by_word = {}
        for r in all_results:
            if r["word"] not in by_word or r["score"] > by_word[r["word"]]["score"]:
                by_word[r["word"]] = r
        ranked = sorted(by_word.values(), key=lambda r: -r["score"])
        return {"results": ranked[:5]}

    def dispatch(line):
        line = line.strip().lower()
        if not line or line in ("quit", "exit"):
            return "EXIT"
        if line in ("tags", "/tags"):
            for m in members:
                print(f"  [{m.name}] tags: {m.encoded_tags}", flush=True)
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
                a, b = parts[0], parts[1]
            # Find a bridge with both words in vocab
            m = find_bridges_for_words(members, [a, b])
            if m is None:
                print(f"  [no bridge has both {a} and {b}]", flush=True)
                return None
            result = encode_to_bridge(m, a, b)
            print(f"  [{m.name}] {result}", flush=True)
            return None
        if line.startswith("what is "):
            word = line[len("what is "):].strip()
            r = query_word(word)
            if not r["results"]:
                print(f"  No bridge has anything about '{word}'.", flush=True)
            else:
                print(f"  [multi-bridge multitag] '{word}' associates:", flush=True)
                for entry in r["results"]:
                    print(f"    {entry['word']:8s} = {entry['score']:.3f} "
                          f"via {entry['bridge']}/{entry['tag']}", flush=True)
            return None
        # plain word -> multitag
        r = query_word(line)
        if not r["results"]:
            print(f"  No bridge has anything about '{line}'.", flush=True)
        else:
            for entry in r["results"]:
                print(f"    {entry['word']:8s} = {entry['score']:.3f} "
                      f"via {entry['bridge']}/{entry['tag']}", flush=True)
        return None

    print("Commands:")
    print("  remember a is b   Encode pair (routed to bridge with both words)")
    print("  what is X         Multi-bridge multitag retrieval")
    print("  <word>            Same as 'what is'")
    print("  tags              List tags across all bridges")
    print("  quit              Exit")
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
