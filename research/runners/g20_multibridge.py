"""G.20 multi-bridge ensemble: scale beyond 32-concept single-bridge ceiling.

Loads N shared-pool (G.20) bridges, each with its own 32-concept vocab.
Routes queries to the bridge that has the queried concept(s).

With 5 bridges x 32 concepts = 160 unique concepts in a single ensemble.
Combined with path-2 morpheme tokenization (~6x combinatorial reach):
projected ~960 surface forms. Combined with path-3 hierarchy: +35
category nodes for taxonomic queries.

This is the scaling unlock for the path-1 BREAKTHROUGH:
- v16 ceiling: 16 concepts/bridge (3200 neurons, 77.5% multi-seed)
- G.20 single-bridge: 32 concepts (1600 neurons, 81.2% seed 42)
- G.20 multi-bridge: 5x32 = 160 concepts (8000 neurons total)

Usage:
  python -m research.runners.g20_multibridge \\
      --bridges bridge1.h5 bridge2.h5 \\
      --vocab-files vocab1.txt vocab2.txt \\
      --friendly --scripted "remember apple is big,what is apple"
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from research.runners.concept_pool_demo_shared import (
    build_shared_pool_bridge,
)
from research.runners.shared_pool_chat import (
    stim_recall_slice_rates,
    encode_pair_engram,
)


def read_vocab_file(path: str) -> List[str]:
    """Read a vocab file: comma-separated OR newline-separated."""
    text = Path(path).read_text().strip()
    # Strip comments
    lines = [l.split("#")[0].strip() for l in text.split("\n")]
    text = " ".join(lines).strip()
    # Split on comma or whitespace
    if "," in text:
        return [w.strip() for w in text.split(",") if w.strip()]
    return text.split()


class SharedPoolMember:
    """One G.20 bridge in the ensemble."""
    def __init__(self, bridge_path: str, vocab: List[str], name: str,
                 n_lang_input: int = 8192, n_shared_pool: int = 1600,
                 n_shared_fs: int = 200, slice_size: int = 50,
                 sparsity: float = 0.03, top_k: int = 100,
                 encoding_steps: int = 200, teacher_pA: float = 500.0,
                 drive_pA: float = 1500.0, drive_steps: int = 100):
        self.bridge_path = bridge_path
        self.vocab = list(vocab)
        self.vocab_set = set(vocab)
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.name = name
        self.n_lang_input = n_lang_input
        self.n_shared_pool = n_shared_pool
        self.n_shared_fs = n_shared_fs
        self.slice_size = slice_size
        self.sparsity = sparsity
        self.top_k = top_k
        self.encoding_steps = encoding_steps
        self.teacher_pA = teacher_pA
        self.drive_pA = drive_pA
        self.drive_steps = drive_steps
        self.bridge = None
        self.encoded_tags: List[str] = []

    def load(self, seed: int):
        if self.bridge is not None:
            return
        self.bridge = build_shared_pool_bridge(
            seed=seed,
            n_lang_input=self.n_lang_input,
            n_shared_pool=self.n_shared_pool,
            n_shared_fs=self.n_shared_fs,
            n_lang_output=self.n_lang_input,
            verbose=False,
        )
        self.bridge.load_checkpoint(self.bridge_path)
        self.encoded_tags = sorted(
            [t["name"] for t in self.bridge.list_engram_tags()])

    def n_concepts(self):
        return len(self.vocab)


def find_member_for_word(members: List[SharedPoolMember],
                          word: str) -> Optional[SharedPoolMember]:
    """Return the first member whose vocab contains the word."""
    for m in members:
        if word in m.vocab_set:
            return m
    return None


def find_member_for_pair(members: List[SharedPoolMember],
                          a: str, b: str) -> Optional[SharedPoolMember]:
    """Return the first member that has BOTH words. None if no single
    bridge has both."""
    for m in members:
        if a in m.vocab_set and b in m.vocab_set:
            return m
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridges", nargs="+", required=True,
                    help="Paths to G.20 shared-pool bridges")
    p.add_argument("--vocab-files", nargs="+", required=True,
                    help="Vocab file per bridge (comma or newline separated)")
    p.add_argument("--names", nargs="+", default=None,
                    help="Optional names; defaults to bridge filenames")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=8192)
    p.add_argument("--n-shared-pool", type=int, default=1600)
    p.add_argument("--slice-size", type=int, default=50)
    p.add_argument("--sparsity", type=float, default=0.03)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--drive-pA", type=float, default=1500.0)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--scripted", type=str, default=None)
    p.add_argument("--friendly", action="store_true")
    args = p.parse_args()

    if len(args.bridges) != len(args.vocab_files):
        print("ERROR: --bridges and --vocab-files must have same length",
              flush=True)
        sys.exit(1)

    names = args.names or [Path(b).stem for b in args.bridges]
    if len(names) != len(args.bridges):
        print("ERROR: --names length must match --bridges", flush=True)
        sys.exit(1)

    members = []
    total_vocab = set()
    for bridge_path, vocab_path, name in zip(
            args.bridges, args.vocab_files, names):
        vocab = read_vocab_file(vocab_path)
        m = SharedPoolMember(
            bridge_path=bridge_path, vocab=vocab, name=name,
            n_lang_input=args.n_lang_input,
            n_shared_pool=args.n_shared_pool,
            slice_size=args.slice_size, sparsity=args.sparsity,
            top_k=args.top_k, encoding_steps=args.encoding_steps,
            drive_pA=args.drive_pA, drive_steps=args.drive_steps,
        )
        members.append(m)
        total_vocab.update(vocab)

    print(f"=== G.20 multi-bridge ensemble ===", flush=True)
    print(f"  Bridges: {[m.name for m in members]}", flush=True)
    print(f"  Total unique vocab: {len(total_vocab)} concepts "
          f"across {len(members)} bridges", flush=True)
    print(flush=True)

    for m in members:
        print(f"  [loading {m.name} ({m.n_concepts()} concepts)]",
              flush=True)
        m.load(args.seed)
        print(f"    {m.n_concepts()} concepts, "
              f"{len(m.encoded_tags)} engram tags restored", flush=True)
    print(flush=True)

    def query_concept(word):
        """Find associates of `word` across all bridges."""
        m = find_member_for_word(members, word)
        if m is None:
            if args.friendly:
                print(f"  I don't know '{word}'.", flush=True)
            else:
                print(f"  [no bridge has '{word}']", flush=True)
            return
        # Find tags containing this word
        matches = [t for t in m.encoded_tags if word in t.split("_")]
        if not matches:
            if args.friendly:
                print(f"  I don't know anything about '{word}' yet.",
                      flush=True)
            else:
                print(f"  [{m.name}] no tags contain '{word}'", flush=True)
            return
        # Aggregate slice firing across all matching tags
        n = m.n_concepts()
        aggregated = np.zeros(n, dtype=np.float32)
        for tag in matches:
            rates = stim_recall_slice_rates(
                m.bridge, tag, n_concepts=n,
                slice_size=m.slice_size,
                drive_pA=m.drive_pA, stim_steps=m.drive_steps,
            )
            aggregated += rates
        sorted_idx = np.argsort(-aggregated)
        top5 = [(m.vocab[i], float(aggregated[i])) for i in sorted_idx[:5]]
        associates = [(w, s) for w, s in top5 if w != word][:4]
        if args.friendly:
            if not associates:
                print(f"  '{word}' has no associates yet.", flush=True)
            else:
                summaries = [f"{w} ({s:.0f})" for w, s in associates]
                print(f"  {word.capitalize()} is associated with: "
                      f"{', '.join(summaries)}.", flush=True)
        else:
            print(f"  [{m.name}] '{word}' associates "
                  f"(from {len(matches)} tag(s)):", flush=True)
            for w, s in associates:
                print(f"    {w:12} {s:.0f}", flush=True)

    def dispatch(line):
        line = line.strip().lower()
        if not line or line in ("quit", "exit"):
            return "EXIT"
        if line in ("concepts", "vocab", "/vocab"):
            for m in members:
                print(f"  [{m.name}] {m.n_concepts()} concepts: "
                      f"{m.vocab[:8]}{'...' if m.n_concepts() > 8 else ''}",
                      flush=True)
            print(f"  TOTAL: {len(total_vocab)} unique concepts",
                  flush=True)
            return None
        if line in ("tags", "/tags"):
            for m in members:
                print(f"  [{m.name}] {len(m.encoded_tags)} tags: "
                      f"{m.encoded_tags[:5]}{'...' if len(m.encoded_tags) > 5 else ''}",
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
            # Route: prefer single bridge with both words
            m_both = find_member_for_pair(members, a, b)
            if m_both is None:
                # Cross-bridge: encode in each bridge that has at least
                # one word
                encoded_in = []
                for m in members:
                    if a in m.vocab_set or b in m.vocab_set:
                        # Need a helper that does PARTIAL encoding;
                        # the existing encode_pair_engram requires BOTH
                        # words to be in the same vocab. So for cross-
                        # bridge we'd need a partial-encoding path. For
                        # this iteration, only intra-bridge encoding.
                        pass
                if args.friendly:
                    print(f"  Sorry, I can only remember pairs from the "
                          f"same vocab right now. '{a}' or '{b}' "
                          f"crosses bridges.", flush=True)
                else:
                    print(f"  [no single bridge has both '{a}' and '{b}']",
                          flush=True)
                return None
            tag = encode_pair_engram(
                m_both.bridge, a, b, vocab=m_both.vocab,
                slice_size=m_both.slice_size,
                n_lang_input=m_both.n_lang_input,
                sparsity=m_both.sparsity,
                encoding_steps=m_both.encoding_steps,
                teacher_pA=m_both.teacher_pA,
                top_k=m_both.top_k,
            )
            m_both.encoded_tags.append(tag)
            if args.friendly:
                print(f"  OK, I'll remember {a} is {b}.", flush=True)
            else:
                print(f"  [{m_both.name}] encoded tag '{tag}'", flush=True)
            return None
        if line.startswith("what is "):
            word = line[len("what is "):].strip()
            query_concept(word)
            return None
        if line.startswith("is "):
            rest = line.rstrip("?").strip()[len("is "):]
            parts = rest.split()
            if len(parts) == 2:
                a, b = parts
                m = find_member_for_pair(members, a, b)
                if m is None:
                    if args.friendly:
                        print(f"  I don't have a bridge with both {a} "
                              f"and {b}.", flush=True)
                    else:
                        print(f"  UNKNOWN (no bridge has both)",
                              flush=True)
                    return None
                tag = f"{a}_{b}"
                if tag in m.encoded_tags:
                    if args.friendly:
                        print(f"  Yes, {a} is {b}.", flush=True)
                    else:
                        print(f"  [{m.name}] YES (tag '{tag}')",
                              flush=True)
                else:
                    if args.friendly:
                        print(f"  I don't know.", flush=True)
                    else:
                        print(f"  [{m.name}] UNKNOWN", flush=True)
            return None
        # plain word
        query_concept(line)
        return None

    print("Commands:")
    print("  remember a is b      Encode (a, b) on bridge containing both")
    print("  what is X            Find associates of X")
    print("  <word>               Same as 'what is'")
    print("  is X Y?              Exact tag match")
    print("  concepts / vocab     List per-bridge vocab")
    print("  tags                 List per-bridge tags")
    print("  quit                 Exit")
    print()

    if args.scripted:
        for inp in [s.strip() for s in args.scripted.split(",")
                     if s.strip()]:
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
