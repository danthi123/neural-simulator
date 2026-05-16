"""Build + cache a cleaned distillation corpus from the local teacher
model (sequence-level / data distillation, Kim & Rush 2016).

The teacher (Qwen2.5-0.5B-Instruct) generates English prose; this
module cleans it to ASCII and caches it to research/datasets/ as a
deliberately-committed training artifact. The student trains on this
cached corpus; the teacher is never needed at runtime.

`clean_corpus` is pure / deterministic / dependency-free (only `re`),
so it is unit-testable without torch/CUDA. `main()` performs the
one-time GPU generation.
"""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path

# repo_root = research/runners/build_distill_corpus.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DISTILL_PATH = _REPO_ROOT / "research" / "datasets" / "distill_corpus.txt"


def clean_corpus(raw: str) -> str:
    """Keep only printable-ASCII chars and newlines, collapse runs of
    3+ newlines to a blank-line separator, and strip ends.

    Pure and deterministic. No dependencies beyond `re`.
    """
    kept = [ch for ch in raw if (32 <= ord(ch) < 127) or ch == "\n"]
    text = "".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build + cache the teacher-distilled corpus.")
    parser.add_argument("--n-passages", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the corpus exists.")
    args = parser.parse_args()

    if DISTILL_PATH.exists() and not args.force:
        print("exists, skipping: " + str(DISTILL_PATH))
        return

    # Imported lazily so unit tests (clean_corpus) need no torch/CUDA.
    from research.runners.distill_teacher import generate_corpus

    print("generating teacher corpus: n_passages=%d max_new_tokens=%d "
          "seed=%d" % (args.n_passages, args.max_new_tokens, args.seed))
    raw = generate_corpus(n_passages=args.n_passages,
                          max_new_tokens=args.max_new_tokens,
                          seed=args.seed)
    cleaned = clean_corpus(raw)

    DISTILL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISTILL_PATH.write_text(cleaned, encoding="utf-8")

    print("wrote %s" % str(DISTILL_PATH))
    print("char count: %d" % len(cleaned))
    print("first 200 chars sample:")
    print(cleaned[:200])


if __name__ == "__main__":
    main()
