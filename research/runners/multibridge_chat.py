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
SET3_VOCAB = {
    "word_to_idx": {
        "north": 0, "east": 1, "south": 2, "west": 3,
        "house": 4, "road": 5, "fire": 6, "water": 7,
        "give": 8, "take": 9, "find": 10, "lose": 11,
        "tall": 12, "short": 13, "wet": 14, "dry": 15,
    },
    "word_to_pool": {
        "house": "noun_pool_HOUSE", "road": "noun_pool_ROAD",
        "fire": "noun_pool_FIRE", "water": "noun_pool_WATER",
        "give": "verb_pool_GIVE", "take": "verb_pool_TAKE",
        "find": "verb_pool_FIND", "lose": "verb_pool_LOSE",
        "tall": "adjective_pool_TALL", "short": "adjective_pool_SHORT",
        "wet": "adjective_pool_WET", "dry": "adjective_pool_DRY",
    },
    "concept_words": ["house","road","fire","water","give","take","find","lose",
                       "tall","short","wet","dry"],
}
SET4_VOCAB = {
    "word_to_idx": {
        "north": 0, "east": 1, "south": 2, "west": 3,
        "person": 4, "baby": 5, "ball": 6, "key": 7,
        "open": 8, "close": 9, "push": 10, "pull": 11,
        "happy": 12, "sad": 13, "full": 14, "empty": 15,
    },
    "word_to_pool": {
        "person": "noun_pool_PERSON", "baby": "noun_pool_BABY",
        "ball": "noun_pool_BALL", "key": "noun_pool_KEY",
        "open": "verb_pool_OPEN", "close": "verb_pool_CLOSE",
        "push": "verb_pool_PUSH", "pull": "verb_pool_PULL",
        "happy": "adjective_pool_HAPPY", "sad": "adjective_pool_SAD",
        "full": "adjective_pool_FULL", "empty": "adjective_pool_EMPTY",
    },
    "concept_words": ["person","baby","ball","key","open","close","push","pull",
                       "happy","sad","full","empty"],
}
SET5_VOCAB = {
    "word_to_idx": {
        "north": 0, "east": 1, "south": 2, "west": 3,
        "food": 4, "drink": 5, "hand": 6, "foot": 7,
        "speak": 8, "listen": 9, "read": 10, "write": 11,
        "new": 12, "old": 13, "clean": 14, "hard": 15,
    },
    "word_to_pool": {
        "food": "noun_pool_FOOD", "drink": "noun_pool_DRINK",
        "hand": "noun_pool_HAND", "foot": "noun_pool_FOOT",
        "speak": "verb_pool_SPEAK", "listen": "verb_pool_LISTEN",
        "read": "verb_pool_READ", "write": "verb_pool_WRITE",
        "new": "adjective_pool_NEW", "old": "adjective_pool_OLD",
        "clean": "adjective_pool_CLEAN", "hard": "adjective_pool_HARD",
    },
    "concept_words": ["food","drink","hand","foot","speak","listen","read","write",
                       "new","old","clean","hard"],
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


def normalize_possessive(rest):
    """Convert 'X's Y' to 'Y_of_X' tag form.

    Returns rewritten string if possessive found, else original.
    Examples:
      "apple's color is red"   -> "color_of_apple is red"
      "the dog's tail is long" -> "tail_of_dog is long"

    Used by both 'remember' encoding and 'is X Y?' queries so
    possessive forms map to a canonical tag-name representation.
    """
    import re
    m = re.match(r"^(\w+)'s\s+(\w+)(.*)$", rest)
    if m:
        owner, attr, tail = m.group(1), m.group(2), m.group(3)
        return f"{attr}_of_{owner}{tail}"
    return rest


def resolve_pronouns(rest, last_subject):
    """Replace pronouns (it/he/she/they) with last_subject if set.

    Single-token substitution only. Pronouns inside compound words are
    NOT replaced (e.g. 'kit' won't be touched).
    """
    if not last_subject:
        return rest
    parts = rest.split()
    out = []
    for w in parts:
        if w in ("it", "he", "she", "they"):
            out.append(last_subject)
        else:
            out.append(w)
    return " ".join(out)


def query_sentence_template(members, template):
    """Find sentences matching a tag-name template.

    Template is a list of strings where '*' is a wildcard. Returns
    list of dicts {wildcards, tag, bridge} matching the template
    across all bridges' encoded_tags.

    E.g. template=['*', 'ate', 'apple'] finds 'alice_ate_apple',
    'bob_ate_apple' across all bridges, with wildcards=['alice'] or
    wildcards=['bob']. Returns subjects in tag-name first-position.

    Pure function — easy to unit-test. Used by 'who X Y?' and 'what
    did X Y?' commands.
    """
    results = []
    for m in members:
        for tag in m.encoded_tags:
            parts = tag.split("_")
            if len(parts) != len(template):
                continue
            wildcards = []
            ok = True
            for tpart, ttok in zip(parts, template):
                if ttok == "*":
                    wildcards.append(tpart)
                elif tpart != ttok:
                    ok = False
                    break
            if ok:
                results.append({
                    "wildcards": wildcards,
                    "tag": tag, "bridge": m.name,
                })
    return results


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

    SET_NAME_TO_VOCAB = {
        "set1": SET1_VOCAB,
        "set2": SET2_VOCAB,
        "set3": SET3_VOCAB,
        "set4": SET4_VOCAB,
        "set5": SET5_VOCAB,
    }
    SET_NAME_TO_PATCH_MODULE = {
        "set1": None,
        "set2": "research.runners.concept_pool_demo_set2",
        "set3": "research.runners.concept_pool_demo_set3",
        "set4": "research.runners.concept_pool_demo_set4",
        "set5": "research.runners.concept_pool_demo_set5",
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

    def encode_to_bridge(m, *words):
        """Encode a variable-length word tuple as one engram on bridge m.

        Supports PARTIAL encoding: if only some of the words are in m's
        vocab, encode those pools + drive their lang_input patterns. The
        other words are ignored at this bridge but still recorded in the
        tag name. Multi-bridge querying recovers the cross-set
        relationship by aggregating tags with matching name across
        bridges.

        Tag name = words joined by '_' (e.g. 'alice_ate_apple'). Order
        info preserved in name regardless of bridge.

        Returns the tag name, 'already remembered' string, or None if
        the bridge knows zero of the words.
        """
        from sim.backend import get_backend
        cp, _ = get_backend()
        tag = "_".join(words)
        if tag in m.encoded_tags:
            return f"already remembered in {m.name}"
        bridge = m.bridge
        rm = bridge.region_manager
        known_words = [w for w in words
                        if w in m.word_to_idx and w in m.word_to_pool]
        if not known_words:
            return None

        n_total = bridge.cp_external_input_current.shape[0]
        lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)

        # Drive lang_input for each known word
        drives = []
        for w in known_words:
            drives.append(orthogonal_drive_pattern(
                cue_idx=m.word_to_idx[w],
                n_cues=args.n_words_for_orthogonal,
                n_neurons=args.n_lang_input,
                drive_max_pA=200.0, sparsity=args.sparsity,
            ))
        combined = cp.asarray(sum(drives), dtype=cp.float32)

        teacher_pools = [cp.asarray(
            list(rm.indices(m.word_to_pool[w])), dtype=cp.int64)
            for w in known_words]

        bridge.start_engram_recording(tag)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
        ext = cp.zeros(n_total, dtype=cp.float32)
        for _ in range(args.encoding_steps):
            ext.fill(0)
            ext[lang_arr] = combined
            for tp in teacher_pools:
                ext[tp] = args.balanced_teacher_pA
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
        aggregated across all bridges that have tags with this word
        (in tag name OR vocab). Cross-bridge encoded tags are searched
        in every bridge — useful for cross-set associations.
        """
        all_results = []
        for m in members:
            # Search ALL tags by name for cross-set support, not just
            # bridges where word is in vocab. This way set1's
            # 'sun_hot' tag (encoded via cross-set fallback) is found
            # even though set1 doesn't have 'sun' in its vocab.
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

    def query_sentence(template):
        """Thin wrapper around top-level query_sentence_template using
        the closure's `members`."""
        return query_sentence_template(members, template)

    def encode_sentence(words):
        """Multi-bridge sentence encoding.

        Tries each bridge: if a bridge knows any of the words, encode
        partial there. Tag name preserves full sentence order across
        all participating bridges.
        """
        m_all = find_bridges_for_words(members, words)
        if m_all is not None:
            result = encode_to_bridge(m_all, *words)
            print(f"  [{m_all.name}] {result}", flush=True)
            return
        encoded_in = []
        for m in members:
            if any(w in m.vocab for w in words):
                r = encode_to_bridge(m, *words)
                if r is not None:
                    encoded_in.append((m.name, r))
        if encoded_in:
            bridge_names = [n for n, _ in encoded_in]
            tag_name = encoded_in[0][1]
            print(f"  [cross-set: '{tag_name}' encoded in {bridge_names}]",
                  flush=True)
        else:
            print(f"  [no bridge knows any of {words}]", flush=True)

    STOPWORDS = {"the", "a", "an", "that", "in", "on", "at",
                  "to", "of", "with", "by"}
    # Anaphora / pronoun resolution: track last subject mentioned
    state = {"last_subject": None}

    def _normalize_possessive(rest):
        return normalize_possessive(rest)

    def _resolve_pronouns(rest):
        return resolve_pronouns(rest, state.get("last_subject"))

    def _parts(rest):
        """Tokenize + strip stopwords + lowercase."""
        return [w for w in rest.split() if w not in STOPWORDS]

    def _track_subject(words):
        """If first word looks like a subject, remember it for pronoun
        resolution. Heuristic: anything that's not a stopword and not
        'NOT'."""
        if words and words[0] not in STOPWORDS and words[0] != "NOT":
            state["last_subject"] = words[0]
        elif len(words) > 1 and words[1] not in STOPWORDS:
            state["last_subject"] = words[1]

    def _yes_no_query(words):
        """Return (matched_tags_list, bridge_names_list) for exact tag match.
        If at least one bridge contains tag joined(words) -> YES."""
        target_tag = "_".join(words)
        hits = []
        for m in members:
            if target_tag in m.encoded_tags:
                hits.append((m.name, target_tag))
        return hits

    def dispatch(line):
        line = line.strip().lower()
        if not line or line in ("quit", "exit"):
            return "EXIT"
        # CONJUNCTIONS: split on ' and ' and dispatch each clause
        # (only if not a 'remember' that handles multi-clause itself)
        if " and " in line and not line.startswith("remember "):
            sub_clauses = [c.strip() for c in line.split(" and ") if c.strip()]
            if len(sub_clauses) >= 2:
                for c in sub_clauses:
                    print(f"  [and] dispatching: '{c}'", flush=True)
                    if dispatch(c) == "EXIT":
                        return "EXIT"
                return None
        if line in ("tags", "/tags"):
            for m in members:
                print(f"  [{m.name}] tags: {m.encoded_tags}", flush=True)
            return None
        if line in ("vocab", "/vocab"):
            for m in members:
                print(f"  [{m.name}] concepts: {m.concept_words}", flush=True)
            return None
        # 'save' -> persist each bridge to its load path
        if line in ("save", "/save"):
            saved = []
            for m in members:
                try:
                    m.bridge.save_checkpoint(m.bridge_path)
                    saved.append((m.name, m.bridge_path,
                                   len(m.encoded_tags)))
                except Exception as e:
                    print(f"  [save failed {m.name}: {e}]", flush=True)
            if saved:
                print(f"  [saved {len(saved)} bridge(s)]:", flush=True)
                for n, p, nt in saved:
                    print(f"    {n}: {nt} tags -> {p}", flush=True)
            return None
        # 'help' -> show commands
        if line in ("commands", "/commands"):
            print("  remember/forget/about/is/who/what/tags/vocab/save/quit",
                  flush=True)
            return None
        # 'know about X' or 'what do you know about X' or 'about X'
        # -> list all tags containing X across all bridges
        if (line.startswith("about ")
            or line.startswith("know about ")
            or line.startswith("what do you know about ")
            or line.startswith("tell me about ")):
            for prefix in ("what do you know about ",
                            "tell me about ", "know about ", "about "):
                if line.startswith(prefix):
                    word = line[len(prefix):].rstrip("?").strip()
                    break
            if not word:
                print("  [usage: about <X> | know about <X> | tell me about <X>]",
                      flush=True)
                return None
            found = []
            for m in members:
                for t in m.encoded_tags:
                    parts = t.split("_")
                    if word in parts:
                        found.append((m.name, t))
            if not found:
                print(f"  [no tags mention '{word}']", flush=True)
            else:
                print(f"  [I know {len(found)} thing(s) about '{word}']:",
                      flush=True)
                for bn, t in found:
                    print(f"    {t} (via {bn})", flush=True)
            return None
        # 'forget <a> <b>' or 'forget <tag_name>' or 'forget about <X>'
        # Removes matching tag(s) from bridges. Note: only removes the
        # named index; doesn't undo synaptic plasticity.
        if line.startswith("forget"):
            rest = line[len("forget"):].strip()
            if rest.startswith("about "):
                word = rest[len("about "):].rstrip("?").strip()
                removed = []
                for m in members:
                    keep, drop = [], []
                    for t in m.encoded_tags:
                        if word in t.split("_"):
                            drop.append(t)
                        else:
                            keep.append(t)
                    for t in drop:
                        try:
                            m.bridge.delete_engram_tag(t)
                            removed.append((m.name, t))
                        except Exception as e:
                            print(f"  [warning: delete failed {t}: {e}]",
                                  flush=True)
                    m.encoded_tags = keep
                if not removed:
                    print(f"  [no tags about '{word}' to forget]", flush=True)
                else:
                    print(f"  [forgot {len(removed)} tag(s) about '{word}']:",
                          flush=True)
                    for bn, t in removed:
                        print(f"    {t} (from {bn})", flush=True)
                return None
            # 'forget a b' or 'forget a is b' -> remove specific tag
            parts = rest.replace(" is ", " ").split()
            if not parts:
                print("  [usage: forget <a> <b> OR forget about <X>]",
                      flush=True)
                return None
            tag = "_".join(parts)
            removed = []
            for m in members:
                if tag in m.encoded_tags:
                    try:
                        m.bridge.delete_engram_tag(tag)
                        m.encoded_tags.remove(tag)
                        removed.append(m.name)
                    except Exception as e:
                        print(f"  [warning: delete failed {tag}: {e}]",
                              flush=True)
            if removed:
                print(f"  [forgot '{tag}' from {removed}]", flush=True)
            else:
                print(f"  [no bridge has tag '{tag}']", flush=True)
            return None
        # Help command
        if line in ("help", "/help", "?"):
            print(__doc__ or "Multi-bridge chat REPL", flush=True)
            return None
        if line.startswith("remember "):
            rest = line[len("remember "):].strip()
            # Resolve pronouns FIRST (so 'it' -> last subject before splitting)
            rest = _resolve_pronouns(rest)
            # Conjunction inside remember: handle each clause
            if " and " in rest:
                sub_clauses = [c.strip() for c in rest.split(" and ")
                                 if c.strip()]
                for c in sub_clauses:
                    print(f"  [remember-and] '{c}'", flush=True)
                    dispatch(f"remember {c}")
                return None
            # Possessive normalization: "X's Y is Z" -> "Y_of_X is Z"
            rest = _normalize_possessive(rest)
            # COMPARISONS: 'X is bigger than Y' -> tag 'X_bigger_Y'
            if " than " in rest:
                # Find comparison marker (e.g. "bigger" before " than ")
                before, after = rest.split(" than ", 1)
                before_parts = before.split()
                if len(before_parts) >= 3 and before_parts[-2] == "is":
                    subj = before_parts[-3]
                    rel = before_parts[-1]
                    obj_parts = _parts(after)
                    if obj_parts:
                        obj = obj_parts[-1]
                        words = [subj, rel, obj]
                        _track_subject([subj, obj])
                        encode_sentence(words)
                        return None
            # TENSE markers: PAST / FUTURE prefix
            # 'remember the dog will eat apple' -> 'FUTURE_dog_eat_apple'
            # 'remember the dog ate apple' -> 'PAST_dog_eat_apple'
            #   (special-case 'ate' -> 'eat' for normalization)
            tense = None
            if " will " in rest:
                tense = "FUTURE"
                rest = rest.replace(" will ", " ")
            elif " did " in rest:
                tense = "PAST"
                rest = rest.replace(" did ", " ")
            # Verb past-form normalization (simple irregular table)
            PAST_TO_PRESENT = {
                "ate": "eat", "drank": "drink", "spoke": "speak",
                "ran": "run", "took": "take", "gave": "give",
                "found": "find", "lost": "lose", "saw": "see",
                "went": "go", "came": "come", "wrote": "write",
                "read": "read",  # same form
                "pushed": "push", "pulled": "pull",
                "opened": "open", "closed": "close",
                "slept": "sleep", "walked": "walk",
                "listened": "listen", "looked": "look",
                "stopped": "stop", "heard": "hear",
            }
            for past_form, present_form in PAST_TO_PRESENT.items():
                if f" {past_form} " in f" {rest} ":
                    # Normalize past form -> present + mark tense as PAST
                    rest = rest.replace(f" {past_form} ",
                                          f" {present_form} ")
                    rest = rest.replace(f"{past_form} ",
                                          f"{present_form} ", 1) if rest.startswith(f"{past_form} ") else rest
                    if tense is None:
                        tense = "PAST"
            # Negation: "remember the dog is not big" -> tag 'NOT_dog_big'
            negated = False
            if " is not " in rest:
                negated = True
                rest = rest.replace(" is not ", " is ")
            if " is " in rest:
                a, b = rest.split(" is ", 1)
                a, b = a.strip(), b.strip()
                # Strip articles from each side
                a_parts = _parts(a)
                b_parts = _parts(b)
                a = a_parts[-1] if a_parts else a
                b = b_parts[-1] if b_parts else b
                words = [a, b]
                if negated:
                    words = ["NOT"] + words
                if tense:
                    words = [tense] + words
                _track_subject([a, b])
                encode_sentence(words)
                return None
            # N-word sentences (drop articles/prepositions)
            parts = _parts(rest)
            if len(parts) < 2:
                print("  [usage: remember a is b OR remember <words ...>]",
                      flush=True)
                return None
            if tense:
                parts = [tense] + parts
            _track_subject(parts)
            encode_sentence(parts)
            return None
        # Yes/no questions: 'is the dog big?' -> exact tag match on dog_big
        # or negated: 'is the dog not big?' -> match 'NOT_dog_big'
        if line.startswith("is "):
            rest = line.rstrip("?").strip()[len("is "):].strip()
            # Resolve pronouns first
            rest = _resolve_pronouns(rest)
            # Possessive normalization
            rest = _normalize_possessive(rest)
            negated = False
            if " not " in rest:
                negated = True
                rest = rest.replace(" not ", " ")
            parts = _parts(rest)
            if len(parts) != 2:
                print("  [usage: 'is <X> <Y>?' or 'is <X> not <Y>?']",
                      flush=True)
                return None
            target = (["NOT"] + parts) if negated else parts
            # Also check opposite-truth tag: if 'dog is big' encoded and
            # query is 'is dog not big', report NO (we have the positive)
            hits = _yes_no_query(target)
            alt = _yes_no_query(
                (parts if negated else ["NOT"] + parts))
            if hits:
                print(f"  YES (matched '{hits[0][1]}' in {hits[0][0]})",
                      flush=True)
            elif alt:
                print(f"  NO (have opposite-truth: '{alt[0][1]}' in "
                      f"{alt[0][0]})", flush=True)
            else:
                print(f"  UNKNOWN (no tag matches)", flush=True)
            _track_subject(parts)
            return None
        # 'who X Y?' or 'who X Y' -> find subject of '*_X_Y'
        # Also matches PAST_ or FUTURE_-prefixed tags when X is past-form
        if line.startswith("who "):
            rest = line.rstrip("?").strip()[len("who "):].strip()
            STOPWORDS = {"the", "a", "an", "that"}
            parts = [w for w in rest.split() if w not in STOPWORDS]
            if len(parts) < 2:
                print("  [usage: who <verb> <obj>? or who <verb> <mod> <obj>?]",
                      flush=True)
                return None
            # Normalize past-form verbs to present (so 'who ate apple?'
            # matches 'PAST_dog_eat_apple')
            PAST_TO_PRESENT = {
                "ate": "eat", "drank": "drink", "spoke": "speak",
                "ran": "run", "took": "take", "gave": "give",
                "found": "find", "lost": "lose", "saw": "see",
                "went": "go", "came": "come", "wrote": "write",
                "pushed": "push", "pulled": "pull",
                "opened": "open", "closed": "close",
                "slept": "sleep", "walked": "walk",
                "listened": "listen", "looked": "look",
                "stopped": "stop", "heard": "hear",
            }
            past_query = any(p in PAST_TO_PRESENT for p in parts)
            future_query = "will" in parts
            parts = [PAST_TO_PRESENT.get(w, w) for w in parts
                      if w != "will"]
            # Build templates: try with PAST_/FUTURE_ prefix and bare
            templates = [["*"] + parts]
            if past_query:
                templates.append(["PAST", "*"] + parts)
            elif future_query:
                templates.append(["FUTURE", "*"] + parts)
            else:
                # Also try prefixed forms in case user asked w/o tense
                templates.append(["PAST", "*"] + parts)
                templates.append(["FUTURE", "*"] + parts)
            all_matches = []
            for tpl in templates:
                all_matches.extend(query_sentence(tpl))
            if not all_matches:
                print(f"  [no tag matches: *_{'_'.join(parts)}]", flush=True)
            else:
                subjects = sorted(set(r["wildcards"][-1]
                                       if r["wildcards"][0] in ("PAST", "FUTURE")
                                       else r["wildcards"][0]
                                       for r in all_matches))
                print(f"  [subjects of '{' '.join(parts)}']: "
                      f"{', '.join(subjects)}", flush=True)
                for r in all_matches:
                    print(f"    {r['tag']} (via {r['bridge']})", flush=True)
            return None
        # 'what did X Y?' -> find object of 'X_Y_*' or 'PAST_X_Y_*'
        if line.startswith("what did "):
            rest = line.rstrip("?").strip()[len("what did "):].strip()
            STOPWORDS = {"the", "a", "an", "that"}
            parts = [w for w in rest.split() if w not in STOPWORDS]
            if len(parts) < 2:
                print("  [usage: what did <subj> <verb>? or what did <subj> <verb> <mod>?]",
                      flush=True)
                return None
            # 'did' triggers past-form normalization (e.g. 'what did dog ate?'
            # has verb 'ate' which should normalize to 'eat')
            PAST_TO_PRESENT = {
                "ate": "eat", "drank": "drink", "spoke": "speak",
                "ran": "run", "took": "take", "gave": "give",
                "found": "find", "lost": "lose",
                "wrote": "write", "pushed": "push", "pulled": "pull",
                "opened": "open", "closed": "close", "slept": "sleep",
                "walked": "walk", "listened": "listen", "looked": "look",
                "stopped": "stop", "heard": "hear",
            }
            parts = [PAST_TO_PRESENT.get(w, w) for w in parts]
            # Try bare and PAST-prefixed templates
            templates = [
                parts + ["*"],
                ["PAST"] + parts + ["*"],
            ]
            all_matches = []
            for tpl in templates:
                all_matches.extend(query_sentence(tpl))
            if not all_matches:
                print(f"  [no tag matches: {'_'.join(parts)}_*]", flush=True)
            else:
                objects = sorted(set(r["wildcards"][-1]
                                      for r in all_matches))
                print(f"  [objects of '{' '.join(parts)}']: "
                      f"{', '.join(objects)}", flush=True)
                for r in all_matches:
                    print(f"    {r['tag']} (via {r['bridge']})", flush=True)
            return None
        # Relational queries: 'what is the color of apple?' or
        # 'what color is apple?' -> template ['color', 'of', 'apple', '*']
        # Tag form: 'color_of_apple_red' encoded via 'remember apple's color is red'
        if line.startswith("what is the ") and " of " in line:
            # 'what is the <attr> of <owner>?'
            rest = line.rstrip("?").strip()[len("what is the "):]
            if " of " in rest:
                attr, owner = rest.split(" of ", 1)
                attr = attr.strip()
                owner_parts = _parts(owner.strip())
                owner = owner_parts[-1] if owner_parts else owner.strip()
                template = [attr, "of", owner, "*"]
                matches = query_sentence(template)
                if not matches:
                    print(f"  [no tag matches: {attr}_of_{owner}_*]", flush=True)
                else:
                    values = sorted(set(r["wildcards"][0] for r in matches))
                    print(f"  [{attr} of {owner}]: {', '.join(values)}",
                          flush=True)
                    for r in matches:
                        print(f"    {r['tag']} (via {r['bridge']})", flush=True)
                return None
        # 'what <attr> is <X>?' -> template [attr, 'of', X, '*']
        if line.startswith("what "):
            rest = line.rstrip("?").strip()[len("what "):]
            if " is " in rest:
                attr, owner = rest.split(" is ", 1)
                attr = attr.strip()
                owner_parts = _parts(owner.strip())
                owner = owner_parts[-1] if owner_parts else owner.strip()
                # Only treat as relational if attr is a single word
                # (avoids interfering with 'what did X V?' / 'what is X')
                if attr and " " not in attr and attr not in (
                    "is", "did", "do", "can", "are"):
                    template = [attr, "of", owner, "*"]
                    matches = query_sentence(template)
                    if matches:
                        values = sorted(set(
                            r["wildcards"][0] for r in matches))
                        print(f"  [{attr} of {owner}]: {', '.join(values)}",
                              flush=True)
                        for r in matches:
                            print(f"    {r['tag']} (via {r['bridge']})",
                                  flush=True)
                        return None
            # Fall through to plain 'what is X' handling below
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
    print("  remember a is b               Encode pair")
    print("  remember a is not b           Negated pair (tag 'NOT_a_b')")
    print("  remember X's Y is Z           Possessive ('Y_of_X is Z')")
    print("  remember it/he/she/they is X  Pronoun resolves to last subject")
    print("  remember <w1> ... <wN>        N-word sentence (order in tag)")
    print("  remember X and Y              Chained: each clause encoded")
    print("  is X Y?                       YES/NO/UNKNOWN exact tag match")
    print("  is X not Y?                   Negated YES/NO query")
    print("  is X's Y Z?                   Possessive YES/NO query")
    print("  remember X is bigger than Y   Comparison ('X_bigger_Y' tag)")
    print("  remember X will V Y           Future tense ('FUTURE_X_V_Y' tag)")
    print("  remember X ate Y              Past tense ('PAST_X_eat_Y' tag)")
    print("  who <verb> <obj>?             Find subject of '*_verb_obj'")
    print("  what did <subj> <verb>?       Find object of 'subj_verb_*'")
    print("  what is the Y of X?           Relational: 'Y_of_X_*' tag")
    print("  what Y is X?                  Same (compact form)")
    print("  what is X                     Multi-bridge multitag retrieval")
    print("  about X / tell me about X     List all tags mentioning X")
    print("  forget a b                    Remove tag 'a_b' from bridges")
    print("  forget about X                Remove all tags mentioning X")
    print("  <word>                        Same as 'what is'")
    print("  X and Y                       Conjunction: each dispatched")
    print("  tags                          List tags across all bridges")
    print("  vocab                         List concept words per bridge")
    print("  save                          Persist all bridges to disk")
    print("  quit                          Exit")
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
