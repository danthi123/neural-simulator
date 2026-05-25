"""Direction R-v3 launcher: capacity envelope extension at 256 / 384 / 512
associations on the Direction M 320-concept G.20 multi-bridge chat
deliverable.

Generates N random ("remember X is Y") commands + 20 query commands;
invokes g20_multibridge.py --sparse --scripted "..."; parses output
for top-1 / top-3 retrieval accuracy; writes per-N JSON.

Reuses (does NOT modify) the Direction M deliverable (validated 2026-
05-24): the cached G.20 sparse 5-bridge ensemble at the 320-concept
production tier (5 bridges x 64 concepts = 320 unique).

Pre-registered bar: top-3 >= 0.80 (project's frozen multi-seed bar);
per-N verdict tag DIRECTION_R_V3_PASS_AT_{N} or
DIRECTION_R_V3_BOUNDARY_AT_{N}.

Cost: ~10-25 min per cell on CPU/GPU; ~45-55 min for all 3 cells.

Discipline:
- Bar UNCHANGED at 0.80 (top-3 only is the formal bar; top-1 is
  informational characterization)
- No protected/frozen/moat modification
- No autograd
- Honest propagation EVERY outcome both remotes
- The envelope edge is itself the biology-translatable finding
  even if no single N PASSes (graceful capacity degradation per
  Brunel-Wang cortical attractor capacity ~0.14 N)
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.g20_vocab_spec_320 import (
    ALL_BRIDGES_64,
    ALL_WORDS_64,
)


BRIDGES_DIR = os.path.join(
    _REPO_ROOT, "research/findings/raw/g11_bg/g20_sparse_bridges_320"
)
BRIDGE_NAMES = [
    "bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
    "bridgeD_spatial", "bridgeE_functional",
]
BRIDGE_FILES = [
    os.path.join(BRIDGES_DIR, f"{name}_sparse64.simstate.h5")
    for name in BRIDGE_NAMES
]
VOCAB_FILES = [
    os.path.join(_REPO_ROOT, "research/findings/raw/g11_bg",
                 f"g20_{name}_vocab64.txt")
    for name in BRIDGE_NAMES
]

# Pre-registered N ladder for the envelope characterization
DIRECTION_R_V3_LADDER = [256, 384, 512]

# Pre-registered bar (matches project's frozen multi-seed bar)
DIRECTION_R_V3_TOP3_MIN = 0.80

# Number of queries per N (matches Direction R / R-extended pattern)
N_QUERIES = 20

# Default seed for reproducibility
DEFAULT_SEED = 42


def _bridge_for_word(word: str) -> str:
    """Return which bridge name contains the given word."""
    for bridge_name, vocab in ALL_BRIDGES_64.items():
        if word in vocab:
            return bridge_name
    raise ValueError(f"word {word!r} not found in any bridge")


def _generate_associations(n: int, seed: int):
    """Generate N random (a, b) pairs.

    For N <= 320 unique vocab: sample N distinct a's; b drawn
    separately ensuring a != b.

    For N > 320: relax the unique-a constraint (allow some words
    to have multiple associations). The capacity probe at N > vocab
    size tests how the substrate handles MULTIPLE associations per
    word (some words become polysemous).

    Returns list of (a, b) tuples.
    """
    rng = random.Random(seed)
    n_total = len(ALL_WORDS_64)
    if n <= n_total:
        a_samples = rng.sample(ALL_WORDS_64, n)
    else:
        # First use all 320 unique words once; then sample with replacement
        a_samples = list(ALL_WORDS_64)
        rng.shuffle(a_samples)
        extras = [rng.choice(ALL_WORDS_64) for _ in range(n - n_total)]
        a_samples += extras
    b_samples = []
    for a in a_samples:
        candidates = [w for w in ALL_WORDS_64 if w != a]
        b_samples.append(rng.choice(candidates))
    return list(zip(a_samples, b_samples))


def _build_scripted_command(pairs, queries):
    """Build a scripted command string for g20_multibridge.py.

    Format: 'remember a is b, remember c is d, ..., what is X, what is Y, ...'
    """
    parts = []
    for a, b in pairs:
        parts.append(f"remember {a} is {b}")
    for q in queries:
        parts.append(f"what is {q}")
    return ",".join(parts)


def _run_multibridge(scripted_cmd: str, seed: int, out_log: str):
    """Invoke g20_multibridge.py with the scripted command.

    Returns the path to the log file (where output was written).
    """
    cmd = [
        sys.executable, "-u", "-m", "research.runners.g20_multibridge",
        "--sparse",
        "--pattern-size", "100",
        "--n-shared-pool", "2000",
        "--n-lang-input", "8192",
        "--sparsity", "0.007",  # required for orthogonal-drive at 64 concepts/bridge
        "--seed", str(seed),
        "--bridges",
    ] + BRIDGE_FILES + [
        "--vocab-files",
    ] + VOCAB_FILES + [
        "--names",
    ] + BRIDGE_NAMES + [
        "--scripted", scripted_cmd,
    ]
    with open(out_log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False,
                       cwd=_REPO_ROOT)
    return out_log


def _parse_chat_output(log_path: str, pairs, queries):
    """Parse g20_multibridge log to extract per-query top-1 and top-3
    retrieval results. Compares against the ground truth from 'pairs'.

    g20_multibridge output for 'what is X' (actual 2026-05 format):
      > what is apple
        'apple' associates (from N tag(s) across M bridges):
          big          677 via bridgeC_adj/apple_big
          spoon        605 via bridgeA_nouns/apple
          angry        475 via bridgeC_adj/apple_big
          person       413 via bridgeA_nouns/apple

    The TOP-N is the ORDER of the associate lines (highest score first).
    Top-1 = first associate; top-3 = first 3 associates.

    Returns dict with: n_queries, top1_correct, top3_correct,
    top1_accuracy, top3_accuracy, per_query.
    """
    import re

    pair_dict = {}  # a -> list of b's that were associated with a
    for a, b in pairs:
        pair_dict.setdefault(a, []).append(b)

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()

    # Per-query response regex: matches the "> what is {q}\n" line,
    # then the "'{q}' associates ..." header, then per-associate lines
    # until a blank line OR next "> ".
    per_query = []
    top1_correct = 0
    top3_correct = 0
    for q in queries:
        expected = pair_dict.get(q, [])

        # Find the query block
        marker = f"> what is {q}\n"
        idx = log_text.find(marker)
        if idx == -1:
            per_query.append({
                "query": q,
                "expected": expected,
                "top1": None,
                "top3": [],
                "top1_correct": False,
                "top3_correct": False,
                "missing_in_log": True,
            })
            continue

        # Extract block from marker until next "> " line (start of next command)
        # or EOF
        rest = log_text[idx + len(marker):]
        next_cmd = rest.find("\n> ")
        block = rest[:next_cmd] if next_cmd >= 0 else rest

        # Parse associate lines: format is whitespace + word + whitespace + score + " via " + bridge/tag
        # Or for UNKNOWN response: "UNKNOWN: {q} ..." style
        # Use regex to extract associate words in order
        top_words = []
        # Look for the header line "'{q}' associates ..."
        header_match = re.search(
            r"'" + re.escape(q) + r"' associates \(from \d+ tag\(s\) across \d+ bridges\):",
            block,
        )
        if header_match is not None:
            # Parse subsequent lines
            after_header = block[header_match.end():]
            for line in after_header.split("\n"):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # Stop at next "Done." or next "> " or other section markers
                if (line_stripped.startswith(">")
                    or line_stripped == "Done."
                    or line_stripped.startswith("Commands:")):
                    break
                # Parse associate line: "word    score via bridge/tag"
                # Word is the first token; might include hyphen or apostrophe
                # but is alphanumeric in our vocab
                tokens = line_stripped.split()
                if len(tokens) >= 1:
                    candidate = tokens[0]
                    # Verify it's plausibly a vocab word (not "Done.", not a number)
                    if candidate and not candidate[0].isdigit():
                        top_words.append(candidate)

        # Handle case where query was UNKNOWN (no associates found)
        if not top_words:
            # Check for UNKNOWN-style response
            if "UNKNOWN" in block or "no tags" in block.lower() or "no engram" in block.lower():
                per_query.append({
                    "query": q,
                    "expected": expected,
                    "top1": None,
                    "top3": [],
                    "top1_correct": False,
                    "top3_correct": False,
                    "abstained": True,
                })
                continue
            else:
                per_query.append({
                    "query": q,
                    "expected": expected,
                    "top1": None,
                    "top3": [],
                    "top1_correct": False,
                    "top3_correct": False,
                    "no_associates_parsed": True,
                })
                continue

        top1 = top_words[0]
        top3 = top_words[:3]
        is_top1 = top1 in expected if expected else False
        is_top3 = any(w in expected for w in top3) if expected else False
        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1
        per_query.append({
            "query": q,
            "expected": expected,
            "top1": top1,
            "top3": top3,
            "top1_correct": is_top1,
            "top3_correct": is_top3,
        })

    return {
        "n_queries": len(queries),
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "top1_accuracy": (
            top1_correct / len(queries) if queries else 0.0
        ),
        "top3_accuracy": (
            top3_correct / len(queries) if queries else 0.0
        ),
        "per_query": per_query,
    }


def _compute_verdict(n: int, top3_accuracy: float) -> str:
    """Pre-registered per-N verdict tag."""
    if top3_accuracy >= DIRECTION_R_V3_TOP3_MIN:
        return f"DIRECTION_R_V3_PASS_AT_{n}"
    return f"DIRECTION_R_V3_BOUNDARY_AT_{n}"


def run_one_cell(n: int, seed: int = DEFAULT_SEED) -> dict:
    """Run the capacity probe at a single N value."""
    print(f"\n=== Direction R-v3 cell: N={n} (seed={seed}) ===", flush=True)
    pairs = _generate_associations(n, seed=seed)
    # Queries: pick 20 random a's from the encoded pairs
    query_rng = random.Random(seed * 7 + n)
    a_pool = list({a for a, _ in pairs})
    n_queries = min(N_QUERIES, len(a_pool))
    queries = query_rng.sample(a_pool, n_queries)
    scripted = _build_scripted_command(pairs, queries)

    out_log = os.path.join(
        _HERE, f"direction_R_v3_n{n}_seed{seed}.log"
    )
    print(f"  encoding {n} associations + querying {n_queries} words "
          f"via g20_multibridge --sparse", flush=True)
    print(f"  scripted command length: {len(scripted)} chars", flush=True)
    print(f"  log -> {out_log}", flush=True)

    _run_multibridge(scripted, seed=seed, out_log=out_log)

    print(f"  parsing log for retrieval results...", flush=True)
    parsed = _parse_chat_output(out_log, pairs, queries)
    verdict = _compute_verdict(n, parsed["top3_accuracy"])
    print(f"  N={n}: top1={parsed['top1_correct']}/{parsed['n_queries']}={parsed['top1_accuracy']:.3f} "
          f"top3={parsed['top3_correct']}/{parsed['n_queries']}={parsed['top3_accuracy']:.3f} "
          f"-> {verdict}", flush=True)

    return {
        "n": n,
        "seed": seed,
        "n_queries": parsed["n_queries"],
        "top1_correct": parsed["top1_correct"],
        "top3_correct": parsed["top3_correct"],
        "top1_accuracy": parsed["top1_accuracy"],
        "top3_accuracy": parsed["top3_accuracy"],
        "verdict": verdict,
        "per_query": parsed["per_query"],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Direction R-v3 capacity envelope launcher "
                    "(extends Direction R / R-extended to N=256/384/512)"
    )
    ap.add_argument("--n-values", type=str, default="256,384,512",
                    help="comma-separated N values (default 256,384,512)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--out", type=str,
        default=os.path.join(_HERE, "direction_R_v3_envelope.json"),
    )
    args = ap.parse_args()

    n_values = [int(s) for s in args.n_values.split(",")]
    print(f"=== Direction R-v3 envelope characterization ===", flush=True)
    print(f"  N values: {n_values}", flush=True)
    print(f"  seed: {args.seed}", flush=True)
    print(f"  bar: top3 >= {DIRECTION_R_V3_TOP3_MIN}", flush=True)
    print(f"  bridges: {len(BRIDGE_FILES)} G.20 sparse 320-concept "
          f"production tier (2026-05-16)", flush=True)
    print(f"  out: {args.out}", flush=True)

    import time
    t0 = time.time()
    cell_results = []
    for n in n_values:
        result = run_one_cell(n, seed=args.seed)
        cell_results.append(result)

    total_min = (time.time() - t0) / 60.0
    print(f"\n=== Direction R-v3 envelope summary ===", flush=True)
    print(f"{'N':>5}  {'top-1':>6}  {'top-3':>6}  verdict", flush=True)
    for r in cell_results:
        print(f"{r['n']:>5}  {r['top1_accuracy']:.3f}  "
              f"{r['top3_accuracy']:.3f}  {r['verdict']}", flush=True)
    print(f"Wall total: {total_min:.1f} min", flush=True)

    out = {
        "ladder": n_values,
        "seed": args.seed,
        "bar_top3_min": DIRECTION_R_V3_TOP3_MIN,
        "cells": cell_results,
        "wall_minutes": total_min,
        "bridges": BRIDGE_NAMES,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
