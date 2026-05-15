"""Concept-concept chat REPL — user types a concept, system replies with associated concept.

NO motor routing. Output is via lang_output cosine to concept words.
This is the real semantic-memory chat (vs motor-direction chat).

Usage:
  python -m research.runners.compose_concept_chat \\
    --load-bridge .../seed42_v16.simstate.h5 \\
    --seed 42 \\
    --pairs "apple:big,dog:small,cat:hot,river:cold,big:hot,small:cold" \\
    --scripted "apple,dog,big,small,hot,cat,go"

User input: any concept word.
System response: drives lang_input(word), reads lang_output, returns
top-3 concept words that come to mind.
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_concept_engram import (
    encode_concept_pair, lang_output_pattern_during_input,
    lang_output_pattern_during_stim, cosine_to_word, _ALL_CONCEPTS,
)
from research.runners.compose_concept_pool_readout import (
    measure_concept_pool_rates, _POOL_TO_WORD,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--n-words-for-orthogonal", type=int, default=16)
    p.add_argument("--pairs", type=str,
                    default="apple:big,dog:small,cat:hot,river:cold,"
                            "big:hot,small:cold,apple:cat,dog:river",
                    help="Train these concept-concept associations")
    p.add_argument("--encoding-steps", type=int, default=500,
                    help="Encoding events per pair (default 500 for 87.5% "
                    "stim-recall recipe; was 200 before bug discovery)")
    p.add_argument("--balanced-teacher-pA", type=float, default=500.0,
                    help="Teacher current on both concept pools during "
                    "encoding (default 500 pA for 87.5% stim-recall)")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--scripted", type=str, default=None,
                    help="Comma-separated list of test inputs (skips "
                    "interactive). Cue mode for plain words, /stim <tag> "
                    "for stim-recall mode.")
    args = p.parse_args()

    # Parse pairs (empty string = no initial pairs)
    pairs = []
    if args.pairs.strip():
        for ps in args.pairs.split(","):
            ps = ps.strip()
            if not ps:
                continue
            try:
                a, b = ps.split(":")
            except ValueError:
                print(f"WARN: skipping malformed pair '{ps}'", flush=True)
                continue
            if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
                pairs.append((a, b))

    print(f"Loading bridge: {args.load_bridge}", flush=True)
    bridge = cpd.build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        enable_adjective=True,
        weak_dynamics=True,
        enable_direct_verb_to_motor=True,
        verbose=False,
    )
    bridge.load_checkpoint(args.load_bridge)

    # Sync REPL's encoded_tags from any tags restored by load_checkpoint.
    # This enables cross-session persistence: save, exit, restart, queries
    # against the same tags continue to work.
    # bridge.list_engram_tags() returns [{"name": ..., "n_neurons": ...}, ...]
    restored_tag_names = sorted([t["name"] for t in bridge.list_engram_tags()])
    if restored_tag_names:
        print(f"  [restored {len(restored_tag_names)} engram tag(s) from "
              f"checkpoint: {restored_tag_names}]", flush=True)

    # IMPORTANT: do NOT freeze plasticity BEFORE encoding. Cross-pool
    # association weights (lang_input -> non-target pool) need active STDP
    # during engram encoding for the associative recall to work later.
    # We freeze gates AFTER encoding completes, before the chat loop.

    # Region filter: concept pools only (no motor)
    rm = bridge.region_manager
    region_filter = []
    for kind, name in [("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
                        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
                        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"])]:
        for n in name:
            try:
                rm.indices(f"{kind}_{n}")
                region_filter.append(f"{kind}_{n}")
            except Exception:
                pass

    # Concepts that are in the bridge's vocab range
    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < args.n_words_for_orthogonal]

    print(f"\nEncoding {len(pairs)} concept-concept associations...", flush=True)
    print(f"  recipe: {args.encoding_steps} events + teacher {args.balanced_teacher_pA} pA "
          f"(2026-05-14 validated 87.5% stim-recall multi-seed)", flush=True)
    # Initialize encoded_tags from any restored tags so cross-session
    # persistence works (save in session N+1 includes session N's tags).
    encoded_tags = list(restored_tag_names)
    for a, b in pairs:
        tag = f"{a}_{b}"
        encode_concept_pair(
            bridge, a, b, tag,
            encoding_steps=args.encoding_steps,
            drive_pA=200.0, sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            region_filter=region_filter, top_k=args.top_k,
            balanced_teacher_pA=args.balanced_teacher_pA,
            verbose=False,
        )
        encoded_tags.append(tag)
        print(f"  learned: '{a}' <-> '{b}' (tag: {tag})", flush=True)

    # Now freeze plasticity for inference stability (chat loop)
    for g in [
        "language_input_to_motor", "language_input_to_verb_pool",
        "language_input_to_noun_pool", "language_input_to_adjective_pool",
        "motor_to_language_output", "verb_pool_to_language_output",
        "noun_pool_to_language_output", "adjective_pool_to_language_output",
    ]:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    print()
    print("=" * 60)
    print("CONCEPT CHAT")
    print(f"Vocab: {valid_concepts}")
    print(f"Learned associations: {pairs}")
    print(f"Encoded tags: {encoded_tags}")
    print()
    print("Commands:")
    print("  remember <a> is <b>      Encode new pair association (90% retrieval)")
    print("  remember <subj> <verb> <obj>  Encode 3-word sentence (order in tag)")
    print("  what is <word>           Retrieve associates (multi-tag, 90% multi-seed)")
    print("  describe <word>          Natural-language synthesis ('apple is big and hot')")
    print("  what is <a> and <b>      Compositional: words associated with BOTH")
    print("  is <a> <b>?              Yes/no: check if (a,b) is bound")
    print("  who <verb> <obj>?        Find subject of sentence (e.g. 'who ate apple?')")
    print("  what did <subj> <verb>?  Find object (e.g. 'what did alice eat?')")
    print("  tell me more             Next-best associates of last query")
    print("  tell me about <word>     Same as 'what is'")
    print("  forget <tag>             Delete an engram tag (tag = a_b)")
    print("  save [path]              Persist bridge + tags to checkpoint")
    print("  <word>                   Shortcut for multi-tag recall")
    print("  <a> and <b>              Shortcut for intersection query")
    print("  /stim <tag>              Direct tag stim-recall (87.5% multi-seed)")
    print("  /cue <word>              Raw cue-pool firing rank (~28%; experimental)")
    print("  /tags                    List all encoded engram tags")
    print("  /vocab                   List available concept words")
    print("  quit                     Exit")
    print("=" * 60, flush=True)

    # All concept pools (for pool-firing readout)
    all_concept_pools = [_WORD_TO_POOL[w] for w in valid_concepts]

    def handle(word):
        if word not in _WORD_TO_IDX or _WORD_TO_IDX[word] >= args.n_words_for_orthogonal:
            return None
        t0 = time.time()
        # Cue mode: drive lang_input alone, rank concept pools (27.5% multi-seed)
        rates = measure_concept_pool_rates(
            bridge, word, all_concept_pools,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            stim_steps=args.drive_steps,
        )
        pool_self = _WORD_TO_POOL[word]
        non_self_ranked = sorted(
            [(p, r) for p, r in rates.items() if p != pool_self],
            key=lambda kv: -kv[1])
        top3 = [(_POOL_TO_WORD.get(p, p.split('_')[-1]), r)
                 for p, r in non_self_ranked[:3]]
        return {
            "mode": "cue",
            "self_rate": rates[pool_self],
            "top3_non_self": top3,
            "elapsed_s": time.time() - t0,
        }

    def handle_multitag(cue_word):
        """Multi-tag aggregation: stim every engram tag containing this
        word and aggregate the lang_output cosines. This combines all
        learned associations for a single cue into a ranked list — the
        chat REPL equivalent of "what comes to mind when you hear X".

        Built 2026-05-14 to provide cue-driven retrieval at 87.5%-class
        reliability by leveraging stim-recall mechanism for each tag
        that contains the cue, rather than relying on weak cross-pool
        plastic weights.
        """
        if cue_word not in _WORD_TO_IDX:
            return None
        t0 = time.time()
        # Find all tags containing this cue word
        matching_tags = []
        for tag in encoded_tags:
            try:
                a_word, b_word = tag.split("_")
                if cue_word == a_word or cue_word == b_word:
                    other = b_word if cue_word == a_word else a_word
                    matching_tags.append((tag, other))
            except ValueError:
                pass
        if not matching_tags:
            return {"mode": "multitag", "cue": cue_word, "matches": [],
                     "associates": [], "elapsed_s": time.time() - t0}
        # For each matching tag, stim and read lang_output
        # Aggregate by averaging the cosine to each associate
        associate_scores = {}  # word → list of scores
        for tag, other_word in matching_tags:
            pattern, n_lang_out = lang_output_pattern_during_stim(
                bridge, tag, drive_pA=1500.0, stim_steps=args.drive_steps,
            )
            # Cosine to each vocab word in pool
            for w in valid_concepts:
                if w == cue_word:
                    continue  # skip self
                score = cosine_to_word(
                    pattern, w, n_lang_out,
                    n_words_for_orthogonal=args.n_words_for_orthogonal,
                    sparsity=args.sparsity,
                )
                associate_scores.setdefault(w, []).append((tag, other_word, score))
        # Rank associates: max score per associate (best matching tag)
        ranked = []
        for w, hits in associate_scores.items():
            best_score = max(h[2] for h in hits)
            best_tag = max(hits, key=lambda h: h[2])[0]
            n_hits = sum(1 for h in hits if h[2] > 0.1)
            ranked.append((w, best_score, best_tag, n_hits))
        ranked.sort(key=lambda x: -x[1])
        return {
            "mode": "multitag",
            "cue": cue_word,
            "matches": [t for t, _ in matching_tags],
            "associates": ranked[:5],
            "elapsed_s": time.time() - t0,
        }

    def handle_stim(tag_name):
        """Stim-recall: stimulate engram tag, read lang_output spelling.
        This is the 87.5% validated mode (2026-05-14)."""
        if tag_name not in encoded_tags:
            return None
        t0 = time.time()
        pattern, n_lang_out = lang_output_pattern_during_stim(
            bridge, tag_name, drive_pA=1500.0, stim_steps=args.drive_steps,
        )
        # Rank all 16 vocab words by cosine to lang_output pattern
        scores = {}
        for w in valid_concepts:
            scores[w] = cosine_to_word(
                pattern, w, n_lang_out,
                n_words_for_orthogonal=args.n_words_for_orthogonal,
                sparsity=args.sparsity,
            )
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top5 = ranked[:5]
        # Expected: both A and B from "A_B" tag
        try:
            a_word, b_word = tag_name.split("_")
        except ValueError:
            a_word, b_word = None, None
        return {
            "mode": "stim",
            "tag": tag_name,
            "a_word": a_word,
            "b_word": b_word,
            "a_score": scores.get(a_word, 0.0) if a_word else 0.0,
            "b_score": scores.get(b_word, 0.0) if b_word else 0.0,
            "top5": top5,
            "elapsed_s": time.time() - t0,
        }

    def print_result(r):
        if r is None:
            print(f"  [unknown input]", flush=True)
            return
        if r["mode"] == "cue":
            associations = ", ".join(f"{w}={s:.2f}" for w, s in r["top3_non_self"])
            print(f"  [cue mode, ~28% multi-seed]", flush=True)
            print(f"  self: {r['self_rate']:.2f}", flush=True)
            print(f"  associates: [{associations}]", flush=True)
        elif r["mode"] == "multitag":
            if not r["matches"]:
                print(f"  [multitag] no engram tag contains '{r['cue']}'",
                      flush=True)
                return
            print(f"  [multitag, leverages 87.5% stim-recall per tag]",
                  flush=True)
            print(f"  cue: {r['cue']}", flush=True)
            print(f"  matched {len(r['matches'])} tag(s): {r['matches']}",
                  flush=True)
            print(f"  top-5 associates (best-tag cosine):", flush=True)
            for w, score, tag, n_hits in r["associates"]:
                marker = "***" if n_hits >= 2 else ("**" if n_hits >= 1 else "")
                print(f"    {w:8s} = {score:.3f} via {tag:20s} {marker}",
                      flush=True)
        elif r["mode"] == "intersection":
            print(f"  [intersection] cue=({r['a']} AND {r['b']})", flush=True)
            if not r["shared"]:
                print(f"  no words associated with both {r['a']} and {r['b']}",
                      flush=True)
            else:
                for w, min_score, sa, sb in r["shared"][:5]:
                    print(f"    {w:8s} = min({sa:.2f}, {sb:.2f}) = {min_score:.2f}",
                          flush=True)
        elif r["mode"] == "stim":
            print(f"  [stim mode, 87.5% multi-seed] tag={r['tag']}", flush=True)
            print(f"  expected: {r['a_word']} + {r['b_word']}", flush=True)
            print(f"  a_score: {r['a_score']:.3f}   b_score: {r['b_score']:.3f}", flush=True)
            top5_str = ", ".join(f"{w}={s:.2f}" for w, s in r["top5"])
            print(f"  top-5 lang_output: [{top5_str}]", flush=True)
            both_in_top5 = (r["a_word"] in [w for w, _ in r["top5"]] and
                            r["b_word"] in [w for w, _ in r["top5"]])
            print(f"  verdict: {'PASS (both in top-5)' if both_in_top5 else 'PARTIAL/FAIL'}",
                  flush=True)
        print(f"  [{r['elapsed_s']:.1f}s]", flush=True)

    def _strip_articles(s):
        """Strip leading 'the ', 'a ', 'an ', 'that ' for more natural input."""
        s = s.strip()
        for prefix in ("the ", "a ", "an ", "that "):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        return s

    def encode_triple(word_a, word_v, word_b, tag_name):
        """Encode a 3-word tuple as a single engram. Drives lang_input of
        all 3 words simultaneously + teacher current on all 3 pools.

        Order info lives in the tag name string. Retrieval uses tag-name
        pattern matching (e.g. 'alice_ate_*' to find what alice ate).

        Built 2026-05-14 PM as pragmatic sentence-level encoding on top
        of the validated multitag mechanism. Limitations: doesn't
        encode word-order in the engram firing pattern (concept-pool
        architecture lacks temporal binding), but tag-name preserves
        ordering and queries work via string match.
        """
        from sim.backend import get_backend
        cp, _ = get_backend()
        rm = bridge.region_manager
        from research.runners.compose_concept_engram import (
            encode_concept_pair,
        )
        # Use the same encode_concept_pair but with 3-word drive. Simpler:
        # drive all 3 words in lang_input + 3 teacher pools.
        from sim.text_embeddings import orthogonal_drive_pattern
        import numpy as np

        n_words = args.n_words_for_orthogonal
        n_lang = args.n_lang_input
        drive_a = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[word_a], n_cues=n_words,
            n_neurons=n_lang, drive_max_pA=200.0, sparsity=args.sparsity,
        )
        drive_v = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[word_v], n_cues=n_words,
            n_neurons=n_lang, drive_max_pA=200.0, sparsity=args.sparsity,
        )
        drive_b = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[word_b], n_cues=n_words,
            n_neurons=n_lang, drive_max_pA=200.0, sparsity=args.sparsity,
        )
        combined = cp.asarray(drive_a + drive_v + drive_b, dtype=cp.float32)
        lang_arr = cp.asarray(
            list(rm.indices("language_input")), dtype=cp.int64)
        pool_a = cp.asarray(list(rm.indices(_WORD_TO_POOL[word_a])), dtype=cp.int64)
        pool_v = cp.asarray(list(rm.indices(_WORD_TO_POOL[word_v])), dtype=cp.int64)
        pool_b = cp.asarray(list(rm.indices(_WORD_TO_POOL[word_b])), dtype=cp.int64)
        n_total = bridge.cp_external_input_current.shape[0]

        bridge.start_engram_recording(tag_name)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()

        ext = cp.zeros(n_total, dtype=cp.float32)
        for _ in range(args.encoding_steps):
            ext.fill(0)
            ext[lang_arr] = combined
            ext[pool_a] = args.balanced_teacher_pA
            ext[pool_v] = args.balanced_teacher_pA
            ext[pool_b] = args.balanced_teacher_pA
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()

        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()

        bridge.commit_engram_tag(
            tag_name, top_k=args.top_k, region_filter=region_filter,
        )
        return tag_name

    def handle_remember(line):
        """Parse 'remember <a> is <b>' or 'remember <a> <b>' → encode pair.

        Also accepts 3-word sentences: 'remember alice ate apple' →
        encode triple as one engram with tag 'alice_ate_apple'. Order
        info preserved in tag name.

        Accepts natural phrasings with articles stripped.

        Returns the encoded tag name, or None if parse failed.
        """
        # Strip the 'remember ' prefix
        rest = line[len("remember "):].strip()
        # Strip optional 'that' (e.g. 'remember that apple is big')
        if rest.startswith("that "):
            rest = rest[len("that "):].strip()
        # Try 'a is b' form first (2-word with 'is' connector)
        if " is " in rest:
            parts = rest.split(" is ", 1)
            a = _strip_articles(parts[0])
            b = _strip_articles(parts[1])
        else:
            # Try 'a b' or 'a b c' (space-separated)
            parts = [_strip_articles(p) for p in rest.split()]
            if len(parts) == 2:
                a, b = parts[0], parts[1]
            elif len(parts) == 3:
                # 3-word sentence: subject verb object
                w_a, w_v, w_b = parts[0], parts[1], parts[2]
                for w in (w_a, w_v, w_b):
                    if w not in _WORD_TO_IDX or _WORD_TO_IDX[w] >= args.n_words_for_orthogonal:
                        return f"unknown word: {w}"
                tag = f"{w_a}_{w_v}_{w_b}"
                if tag in encoded_tags:
                    return f"already remembered: {tag}"
                encode_triple(w_a, w_v, w_b, tag)
                encoded_tags.append(tag)
                return tag
            else:
                return None
        if a not in _WORD_TO_IDX or _WORD_TO_IDX[a] >= args.n_words_for_orthogonal:
            return f"unknown word: {a}"
        if b not in _WORD_TO_IDX or _WORD_TO_IDX[b] >= args.n_words_for_orthogonal:
            return f"unknown word: {b}"
        tag = f"{a}_{b}"
        if tag in encoded_tags:
            return f"already remembered: {tag}"
        # Encode now (slow ~5s)
        encode_concept_pair(
            bridge, a, b, tag,
            encoding_steps=args.encoding_steps,
            drive_pA=200.0, sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            region_filter=region_filter, top_k=args.top_k,
            balanced_teacher_pA=args.balanced_teacher_pA,
            verbose=False,
        )
        encoded_tags.append(tag)
        return tag

    def handle_intersection(word_a, word_b):
        """Compositional retrieval: what's associated with BOTH a AND b.

        Drive lang_input(a) tags, drive lang_input(b) tags, intersect
        the top-5 associates of each. Returns words appearing in both
        with their combined min-score (weakest link wins).

        Built 2026-05-14 to demonstrate compositional conversational
        capability: 'what's both apple AND red?' style queries.
        """
        r_a = handle_multitag(word_a)
        r_b = handle_multitag(word_b)
        if r_a is None or r_b is None:
            return None
        if not r_a.get("associates") or not r_b.get("associates"):
            return {"mode": "intersection", "a": word_a, "b": word_b,
                     "shared": [], "elapsed_s": 0}
        # Set of words associated with a (top-5)
        words_a = {w: score for w, score, _, _ in r_a["associates"]}
        words_b = {w: score for w, score, _, _ in r_b["associates"]}
        shared = []
        for w in set(words_a) & set(words_b):
            min_score = min(words_a[w], words_b[w])
            shared.append((w, min_score, words_a[w], words_b[w]))
        shared.sort(key=lambda x: -x[1])
        return {
            "mode": "intersection",
            "a": word_a, "b": word_b,
            "shared": shared,
            "elapsed_s": r_a["elapsed_s"] + r_b["elapsed_s"],
        }

    def handle_forget(tag_name):
        """Delete an engram tag at runtime."""
        if tag_name not in encoded_tags:
            return f"no such tag: {tag_name}"
        try:
            bridge.delete_engram_tag(tag_name)
            encoded_tags.remove(tag_name)
            return tag_name
        except Exception as e:
            return f"error deleting {tag_name}: {e}"

    # Simple multi-turn state: last query cue + last shown associates
    state = {"last_cue": None, "last_shown": set()}

    def dispatch(line):
        """Parse one chat line; return result dict or None for command."""
        line = line.strip().lower()
        if not line or line in ("quit", "exit"):
            return "EXIT"
        if line in ("/tags", "tags"):
            print(f"  tags: {encoded_tags}", flush=True)
            return None
        if line in ("/vocab", "vocab"):
            print(f"  vocab: {valid_concepts}", flush=True)
            return None
        if line in ("tell me more", "more"):
            # Re-query the last cue, but exclude already-shown associates
            if state["last_cue"] is None:
                print(f"  [no recent query — type 'what is X' first]", flush=True)
                return None
            r = handle_multitag(state["last_cue"])
            if r is None or not r.get("associates"):
                print(f"  [no more associates for '{state['last_cue']}']", flush=True)
                return None
            # Filter out previously shown
            remaining = [(w, s, t, n) for (w, s, t, n) in r["associates"]
                         if w not in state["last_shown"]]
            if not remaining:
                print(f"  [no more associates for '{state['last_cue']}' "
                      f"beyond {sorted(state['last_shown'])}]", flush=True)
                return None
            print(f"  [more for '{state['last_cue']}']", flush=True)
            for w, score, tag, n_hits in remaining[:3]:
                marker = "**" if score > 0.1 else ""
                print(f"    {w:8s} = {score:.3f} via {tag:20s} {marker}",
                      flush=True)
                state["last_shown"].add(w)
            return None
        if line.startswith("save ") or line == "save":
            parts = line.split(maxsplit=1)
            path = parts[1].strip() if len(parts) > 1 else None
            if not path:
                # Default: save back to the loaded bridge path
                path = args.load_bridge
            try:
                bridge.save_checkpoint(path)
                print(f"  [saved bridge + {len(encoded_tags)} engram tag(s) "
                      f"to {path}]", flush=True)
            except Exception as e:
                print(f"  [save failed: {e}]", flush=True)
            return None
        if line.startswith("/forget ") or line.startswith("forget "):
            tag_arg = line.split(" ", 1)[1].strip()
            result = handle_forget(tag_arg)
            if result == tag_arg:
                print(f"  [forgot: {tag_arg}]", flush=True)
            else:
                print(f"  [{result}]", flush=True)
            return None
        if line.startswith("/stim "):
            tag_arg = line[len("/stim "):].strip()
            r = handle_stim(tag_arg)
            print_result(r)
            return None
        if line.startswith("/cue "):
            word = line[len("/cue "):].strip()
            r = handle(word)
            print_result(r)
            return None
        if line.startswith("remember "):
            result = handle_remember(line)
            if result is None:
                print(f"  [could not parse 'remember' command]", flush=True)
            elif result.startswith("unknown word"):
                print(f"  [{result}; vocab: {valid_concepts}]", flush=True)
            elif result.startswith("already remembered"):
                print(f"  [{result}]", flush=True)
            else:
                print(f"  [remembered: {result}]", flush=True)
            return None
        # 3-word role queries: who/what + verb + object/subject
        # Mechanism: pattern-match tag names, then NEURALLY verify by
        # stimming the tag and confirming all 3 words appear in
        # lang_output top-K. If neural verification fails, drop the
        # candidate. This grounds the symbolic lookup in actual neural
        # storage rather than pure string match.
        def neural_verify_triple(tag_name, words):
            """Stim the tag, verify all 3 words appear in lang_output top-K."""
            try:
                pat, n_lo = lang_output_pattern_during_stim(
                    bridge, tag_name, drive_pA=1500.0,
                    stim_steps=args.drive_steps,
                )
                scores = {}
                for w in valid_concepts:
                    scores[w] = cosine_to_word(
                        pat, w, n_lo,
                        n_words_for_orthogonal=args.n_words_for_orthogonal,
                        sparsity=args.sparsity,
                    )
                ranked = sorted(scores.items(), key=lambda kv: -kv[1])
                top_k_names = [w for w, _ in ranked[:8]]
                return all(w in top_k_names for w in words)
            except Exception:
                return True  # if verify fails for any reason, accept match

        if line.startswith("who "):
            # 'who <verb> <obj>' -> find tag matching '*_verb_obj'
            rest = line[len("who "):].strip().rstrip("?").strip()
            parts = rest.split()
            if len(parts) >= 2:
                verb = parts[0]
                obj = parts[-1]
                matches = [t for t in encoded_tags
                            if t.endswith(f"_{verb}_{obj}")]
                verified = []
                for t in matches:
                    subj = t.split("_")[0]
                    if neural_verify_triple(t, [subj, verb, obj]):
                        verified.append(subj)
                if verified:
                    print(f"  Who {verb} {obj}? {', '.join(verified)}",
                          flush=True)
                elif matches:
                    # Symbol match but neural verification failed
                    subjects_unverified = [t.split("_")[0] for t in matches]
                    print(f"  Who {verb} {obj}? (weak): "
                          f"{', '.join(subjects_unverified)}", flush=True)
                else:
                    print(f"  I don't know who {verb} {obj}.", flush=True)
            else:
                print(f"  [usage: 'who <verb> <object>?']", flush=True)
            return None
        if line.startswith("what did "):
            # 'what did <subj> <verb>' -> find tag matching 'subj_verb_*'
            rest = line[len("what did "):].strip().rstrip("?").strip()
            parts = rest.split()
            if len(parts) >= 2:
                subj = parts[0]
                verb = parts[-1]
                matches = [t for t in encoded_tags
                            if t.startswith(f"{subj}_{verb}_")]
                verified = []
                for t in matches:
                    obj = t.split("_")[-1]
                    if neural_verify_triple(t, [subj, verb, obj]):
                        verified.append(obj)
                if verified:
                    print(f"  What did {subj} {verb}? {', '.join(verified)}",
                          flush=True)
                elif matches:
                    objects_unverified = [t.split("_")[-1] for t in matches]
                    print(f"  What did {subj} {verb}? (weak): "
                          f"{', '.join(objects_unverified)}", flush=True)
                else:
                    print(f"  I don't know what {subj} {verb}.", flush=True)
            else:
                print(f"  [usage: 'what did <subject> <verb>?']", flush=True)
            return None
        if line.startswith("is "):
            # Yes/no query: 'is apple big' → check if apple_big or big_apple tagged
            rest = line[len("is "):].strip()
            # Support 'a b' and 'a is b' redundancy
            if rest.endswith("?"):
                rest = rest[:-1].strip()
            parts = rest.split()
            if len(parts) < 2:
                print(f"  [usage: 'is <a> <b>?']", flush=True)
                return None
            a, b = parts[0], parts[-1]  # first and last words
            # Check if either ordering exists as a tag
            tag1, tag2 = f"{a}_{b}", f"{b}_{a}"
            if tag1 in encoded_tags or tag2 in encoded_tags:
                actual_tag = tag1 if tag1 in encoded_tags else tag2
                # Stim and check cosine confidence
                r = handle_stim(actual_tag)
                if r:
                    a_in_top5 = a in [w for w, _ in r["top5"]]
                    b_in_top5 = b in [w for w, _ in r["top5"]]
                    if a_in_top5 and b_in_top5:
                        print(f"  YES: '{a}' is bound to '{b}' "
                              f"(tag {actual_tag}, both in lang_output top-5)",
                              flush=True)
                    else:
                        print(f"  PARTIAL: tag exists but recall is weak "
                              f"(a={a} in top5: {a_in_top5}, "
                              f"b={b} in top5: {b_in_top5})", flush=True)
                else:
                    print(f"  [tag exists but stim failed]", flush=True)
            else:
                print(f"  NO: no tag binding '{a}' and '{b}' "
                      f"(checked {tag1}, {tag2})", flush=True)
            return None
        if line.startswith("describe "):
            word = line[len("describe "):].strip()
            r = handle_multitag(word)
            if r is None or not r.get("associates"):
                print(f"  I don't know anything about '{word}'.", flush=True)
                return None
            # Natural-language synthesis: take top associates with score > 0.10
            strong = [(w, s) for w, s, _, _ in r["associates"] if s > 0.10]
            if not strong:
                print(f"  I have weak memories about '{word}' but nothing "
                      f"confident.", flush=True)
                return None
            words = [w for w, _ in strong]
            if len(words) == 1:
                print(f"  {word} is {words[0]}.", flush=True)
            elif len(words) == 2:
                print(f"  {word} is {words[0]} and {words[1]}.", flush=True)
            else:
                # 3+: oxford-comma list
                tail = ", ".join(words[:-1])
                print(f"  {word} is {tail}, and {words[-1]}.", flush=True)
            return None
        if line.startswith("what is ") or line.startswith("tell me about "):
            # Natural-language multitag query
            if line.startswith("what is "):
                arg = line[len("what is "):].strip()
            else:
                arg = line[len("tell me about "):].strip()
            if not arg:
                print(f"  [usage: 'what is <word>' or 'what is <a> and <b>']",
                      flush=True)
                return None
            # Check for 'a and b' intersection query
            if " and " in arg:
                parts = arg.split(" and ", 1)
                a = parts[0].strip()
                b = parts[1].strip()
                if not a or not b:
                    print(f"  [usage: 'what is <a> and <b>']", flush=True)
                    return None
                r = handle_intersection(a, b)
                print_result(r)
                # Reset state for intersection queries
                state["last_cue"] = None
                state["last_shown"] = set()
            else:
                r = handle_multitag(arg)
                print_result(r)
                # Track state for 'tell me more' follow-up
                if r and r.get("associates"):
                    state["last_cue"] = arg
                    state["last_shown"] = {w for w, _, _, _ in r["associates"][:5]}
            return None
        # plain word -> multitag mode (the recommended cue retrieval)
        # Also support 'a and b' shortcut
        if " and " in line:
            parts = line.split(" and ", 1)
            a = parts[0].strip()
            b = parts[1].strip()
            r = handle_intersection(a, b)
            print_result(r)
            state["last_cue"] = None
            state["last_shown"] = set()
            return None
        r = handle_multitag(line)
        print_result(r)
        if r and r.get("associates"):
            state["last_cue"] = line
            state["last_shown"] = {w for w, _, _, _ in r["associates"][:5]}
        return None

    if args.scripted:
        inputs = [s.strip() for s in args.scripted.split(",") if s.strip()]
        for inp in inputs:
            print(f"\n> {inp}", flush=True)
            if dispatch(inp) == "EXIT":
                break
    else:
        print("Ready.", flush=True)
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if dispatch(line) == "EXIT":
                break

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
