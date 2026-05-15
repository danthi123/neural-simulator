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

# Path 2: morpheme tokenizer
try:
    from research.runners.subword_tokenizer import (
        tokenize_sentence as _tokenize_sentence,
    )
    _HAS_TOKENIZER = True
except ImportError:
    _HAS_TOKENIZER = False
    _tokenize_sentence = None

# Path 3: hierarchy
try:
    from research.runners.hierarchical_concepts import (
        get_ancestors as _get_ancestors,
        get_descendants as _get_descendants,
        is_a as _is_a,
        DEFAULT_HIERARCHY,
    )
    _HAS_HIERARCHY = True
except ImportError:
    _HAS_HIERARCHY = False
    DEFAULT_HIERARCHY = {}


def encode_partial_pair_engram(bridge, word_to_drive: str,
                                 tag_name: str, vocab,
                                 slice_size: int, n_lang_input: int,
                                 sparsity: float,
                                 encoding_steps: int = 200,
                                 teacher_pA: float = 500.0,
                                 top_k: int = 100) -> str:
    """Encode a SINGLE-word engram tag in this bridge using the given tag name.

    Used for cross-bridge encoding: when a pair (apple, big) spans two
    bridges, we encode apple on the noun bridge and big on the adjective
    bridge — each gets a tag named 'apple_big' (or 'big_apple'). The full
    pair name is preserved in tag names for cross-bridge query aggregation.
    """
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    rm = bridge.region_manager

    if word_to_drive not in vocab:
        raise ValueError(f"word '{word_to_drive}' not in vocab")
    word_idx = vocab.index(word_to_drive)

    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    shared_indices = list(rm.indices("shared_concept_pool"))
    slice_word = cp.asarray(
        shared_indices[word_idx * slice_size:(word_idx + 1) * slice_size],
        dtype=cp.int64)

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
        ext[slice_word] = teacher_pA
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
    p.add_argument("--tokenize", action="store_true",
                    help="Apply morpheme tokenization before parsing")
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
    if args.tokenize and _HAS_TOKENIZER:
        print(f"  Path 2: morpheme tokenization ENABLED", flush=True)
    if _HAS_HIERARCHY:
        print(f"  Path 3: hierarchy queries ENABLED", flush=True)
    print(flush=True)

    # Build known-roots set for tokenizer (all bridge vocabs)
    KNOWN_ROOTS = set(total_vocab) if args.tokenize else set()
    MARKERS = {"PAST", "PLURAL", "ing", "ed", "er", "est", "ly",
                "tion", "able", "ful", "less", "ness", "s", "es", "ies",
                "un", "re", "pre", "dis", "mis", "over", "under", "anti"}

    def _maybe_tokenize(rest):
        """Tokenize + strip markers if --tokenize active."""
        if not args.tokenize or not _HAS_TOKENIZER:
            return rest
        tokens = _tokenize_sentence(rest, KNOWN_ROOTS)
        roots = [t for t in tokens if t not in MARKERS]
        return " ".join(roots)

    for m in members:
        print(f"  [loading {m.name} ({m.n_concepts()} concepts)]",
              flush=True)
        m.load(args.seed)
        print(f"    {m.n_concepts()} concepts, "
              f"{len(m.encoded_tags)} engram tags restored", flush=True)
    print(flush=True)

    def query_concept(word):
        """Find associates of `word` across ALL bridges.

        Searches tag NAMES for `word` across every bridge (not just
        the bridge that has `word` in vocab). This catches cross-bridge
        partial encodings: 'apple_big' is in both bridgeA (which has
        apple) and bridgeC (which has big). Aggregating across both
        gives the full set of associates."""
        all_results = []
        for m in members:
            matches = [t for t in m.encoded_tags
                        if word in t.split("_")]
            for tag in matches:
                rates = stim_recall_slice_rates(
                    m.bridge, tag, n_concepts=m.n_concepts(),
                    slice_size=m.slice_size,
                    drive_pA=m.drive_pA, stim_steps=m.drive_steps,
                )
                sorted_idx = np.argsort(-rates)
                # Top firing concepts from THIS bridge
                for j in sorted_idx[:5]:
                    candidate = m.vocab[j]
                    if candidate == word:
                        continue
                    all_results.append({
                        "word": candidate,
                        "rate": float(rates[j]),
                        "tag": tag,
                        "bridge": m.name,
                    })

        if not all_results:
            if args.friendly:
                print(f"  I don't know anything about '{word}' yet.",
                      flush=True)
            else:
                print(f"  [no tags contain '{word}' across any bridge]",
                      flush=True)
            return
        # Aggregate: max rate per word across all bridges + tags
        by_word = {}
        for r in all_results:
            if (r["word"] not in by_word
                    or r["rate"] > by_word[r["word"]]["rate"]):
                by_word[r["word"]] = r
        ranked = sorted(by_word.values(), key=lambda r: -r["rate"])[:4]
        if args.friendly:
            summaries = [f"{r['word']} ({r['rate']:.0f})" for r in ranked]
            print(f"  {word.capitalize()} is associated with: "
                  f"{', '.join(summaries)}.", flush=True)
        else:
            n_tags = len(set(r["tag"] for r in all_results))
            print(f"  '{word}' associates (from {n_tags} tag(s) across "
                  f"{len(set(r['bridge'] for r in all_results))} bridges):",
                  flush=True)
            for r in ranked:
                print(f"    {r['word']:12} {r['rate']:.0f} "
                      f"via {r['bridge']}/{r['tag']}", flush=True)

    def dispatch(line):
        line = line.strip().lower()
        if not line or line in ("quit", "exit"):
            return "EXIT"
        # PATH 3: hierarchy queries FIRST (before tokenization, since
        # 'what mammals do you know?' shouldn't be tokenized)
        if _HAS_HIERARCHY:
            # 'is a X an Y?' or 'is X an animal?' patterns
            if line.startswith("is a ") or line.startswith("is an "):
                rest = line.rstrip("?").strip()
                # Remove leading 'is a' / 'is an'
                if rest.startswith("is an "):
                    rest = rest[len("is an "):]
                elif rest.startswith("is a "):
                    rest = rest[len("is a "):]
                # Strip articles
                STOP = {"a", "an", "the", "that"}
                parts = [w for w in rest.replace(" an ", " ")
                          .replace(" a ", " ").split() if w not in STOP]
                if len(parts) == 2:
                    a, b = parts[0], parts[1]
                    if a in DEFAULT_HIERARCHY or b in DEFAULT_HIERARCHY or any(
                        a == v or b == v for v in DEFAULT_HIERARCHY.values()):
                        ok = _is_a(a, b)
                        if args.friendly:
                            if ok:
                                print(f"  Yes, {a} is a kind of {b}.",
                                      flush=True)
                            else:
                                print(f"  No, {a} is not a kind of {b}.",
                                      flush=True)
                        else:
                            print(f"  is_a({a}, {b}) = {ok}", flush=True)
                        return None
            # 'what mammals do you know?' / 'what colors do you know?'
            if (line.startswith("what ")
                    and line.endswith("s do you know?")):
                category = line[len("what "):-len("s do you know?")]
                desc = _get_descendants(category)
                if not desc:
                    desc = _get_descendants(category + "s")
                if desc:
                    leaves = [d for d in desc
                              if d not in set(DEFAULT_HIERARCHY.values())]
                    items = leaves or desc
                    if args.friendly:
                        print(f"  Kinds of {category}: "
                              f"{', '.join(items[:8])}.", flush=True)
                    else:
                        print(f"  [descendants of {category}]: "
                              f"{', '.join(items)}", flush=True)
                    return None
        # PATH 2: tokenize remaining command content
        if args.tokenize and _HAS_TOKENIZER:
            for cmd_prefix in ("remember ", "is ", "what is ", "what "):
                if line.startswith(cmd_prefix):
                    content = line[len(cmd_prefix):]
                    tokenized = _maybe_tokenize(content)
                    if tokenized != content:
                        line = cmd_prefix + tokenized
                    break
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
            if m_both is not None:
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
                    print(f"  [{m_both.name}] encoded tag '{tag}'",
                          flush=True)
                return None
            # CROSS-BRIDGE PARTIAL ENCODING: tag name preserves the
            # full pair; each bridge encodes only the word it knows.
            # Query-time tag-name search finds these across bridges.
            tag_name = f"{a}_{b}"
            encoded_in = []
            for m in members:
                if a in m.vocab_set:
                    encode_partial_pair_engram(
                        m.bridge, a, tag_name, vocab=m.vocab,
                        slice_size=m.slice_size,
                        n_lang_input=m.n_lang_input,
                        sparsity=m.sparsity,
                        encoding_steps=m.encoding_steps,
                        teacher_pA=m.teacher_pA,
                        top_k=m.top_k,
                    )
                    m.encoded_tags.append(tag_name)
                    encoded_in.append((m.name, a))
                elif b in m.vocab_set:
                    encode_partial_pair_engram(
                        m.bridge, b, tag_name, vocab=m.vocab,
                        slice_size=m.slice_size,
                        n_lang_input=m.n_lang_input,
                        sparsity=m.sparsity,
                        encoding_steps=m.encoding_steps,
                        teacher_pA=m.teacher_pA,
                        top_k=m.top_k,
                    )
                    m.encoded_tags.append(tag_name)
                    encoded_in.append((m.name, b))
            if encoded_in:
                bridge_names = [n for n, _ in encoded_in]
                if args.friendly:
                    print(f"  OK, I'll remember {a} is {b}.", flush=True)
                else:
                    print(f"  [cross-bridge: '{tag_name}' encoded in "
                          f"{bridge_names}]", flush=True)
            else:
                if args.friendly:
                    print(f"  I don't know '{a}' or '{b}'.", flush=True)
                else:
                    print(f"  [no bridge has either '{a}' or '{b}']",
                          flush=True)
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
                tag = f"{a}_{b}"
                # Check ANY bridge for this exact tag (intra OR cross)
                hits = [(m.name, tag) for m in members
                         if tag in m.encoded_tags]
                if hits:
                    if args.friendly:
                        print(f"  Yes, {a} is {b}.", flush=True)
                    else:
                        bridge_names = [n for n, _ in hits]
                        print(f"  YES (tag '{tag}' in {bridge_names})",
                              flush=True)
                else:
                    if args.friendly:
                        print(f"  I don't know.", flush=True)
                    else:
                        print(f"  UNKNOWN (no bridge has tag '{tag}')",
                              flush=True)
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
