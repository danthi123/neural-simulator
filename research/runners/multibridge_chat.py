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
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_concept_engram import (
    encode_concept_pair, lang_output_pattern_during_stim,
    cosine_to_word, _ALL_CONCEPTS,
)


class BridgeMember:
    """One bridge in the ensemble. Holds bridge + vocab metadata."""
    def __init__(self, bridge_path, vocab_words, n_lang_input, n_per_pool,
                 n_fs_per_pool, sparsity, n_words_for_orthogonal,
                 encoding_steps, balanced_teacher_pA, top_k, name):
        self.bridge_path = bridge_path
        self.vocab = set(vocab_words)
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

    def load(self, seed):
        """Lazily load the bridge from disk."""
        if self.bridge is not None:
            return
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
        for kind, names in [("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
                             ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
                             ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"])]:
            for n in names:
                try:
                    rm.indices(f"{kind}_{n}")
                    self.region_filter.append(f"{kind}_{n}")
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

    # v16 vocab (default for all bridges)
    DEFAULT_VOCAB = [
        "apple", "river", "dog", "cat",
        "go", "come", "stop", "look",
        "big", "small", "hot", "cold",
    ]

    names = args.names or [Path(bp).stem for bp in args.bridges]
    if len(names) != len(args.bridges):
        print(f"ERROR: --names length must match --bridges length", flush=True)
        return

    members = []
    for path, name in zip(args.bridges, names):
        m = BridgeMember(
            bridge_path=path,
            vocab_words=DEFAULT_VOCAB,  # for now all bridges share vocab
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
        members.append(m)

    print(f"=== Multi-bridge chat REPL ===")
    print(f"  Bridges: {[m.name for m in members]}")
    print(f"  Total vocab: {sum(len(m.vocab) for m in members)} word-slots "
          f"across {len(members)} bridges")
    print(f"  (Note: until bridges have different vocabs, this is "
          f"redundant capacity)")
    print()

    # Load all bridges upfront
    for m in members:
        m.load(args.seed)

    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < args.n_words_for_orthogonal]

    def encode_to_bridge(m, a, b):
        """Encode (a, b) pair on bridge member m."""
        tag = f"{a}_{b}"
        if tag in m.encoded_tags:
            return f"already remembered in {m.name}"
        encode_concept_pair(
            m.bridge, a, b, tag,
            encoding_steps=args.encoding_steps,
            drive_pA=200.0, sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            region_filter=m.region_filter, top_k=args.top_k,
            balanced_teacher_pA=args.balanced_teacher_pA,
            verbose=False,
        )
        m.encoded_tags.append(tag)
        return tag

    def query_word(word):
        """Multi-tag retrieval across all bridges. Returns top associates
        aggregated across all bridges that have tags with this word."""
        if word not in _WORD_TO_IDX:
            return {"matches": [], "results": []}
        all_results = []
        for m in members:
            matches = [t for t in m.encoded_tags if word in t.split("_")]
            for tag in matches:
                pat, n_lo = lang_output_pattern_during_stim(
                    m.bridge, tag, drive_pA=1500.0,
                    stim_steps=args.drive_steps,
                )
                for w in valid_concepts:
                    if w == word:
                        continue
                    score = cosine_to_word(
                        pat, w, n_lo,
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
