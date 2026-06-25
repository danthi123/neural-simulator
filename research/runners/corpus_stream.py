"""STREAMING corpus loader for the stream cortex (foundational-curriculum Step 0).

Why this exists (scoping `_foundational_curriculum_scaling_scoping.md` §3 piece 1): the develop loop's
`StreamCortex._load_token_stream` and the on-bridge derisk's `load_token_stream` were BOTH hardcoded to ONE
file `data/corpus/tinystories.txt` and BOTH did `text = fh.read()` -- loading the WHOLE corpus into one giant
string. That does not scale to a foundational corpus (BabyLM-10M..100M = hundreds of MB..GB of raw text). This
module replaces the whole-file read with a BOUNDED-MEMORY chunked stream:

  - `iter_stories(path, chunk_bytes)` -- a TRUE streaming generator. It reads the file in fixed-size byte chunks,
    carries the partial story across chunk boundaries, splits on the `<|endoftext|>` document delimiter, and
    yields ONE story's tokens (`list[str]`) at a time. Peak resident raw text = ~one chunk + one partial story,
    INDEPENDENT of total corpus size. THIS is the path the large-corpus scaling route uses (sequential
    consumption of a corpus far bigger than RAM).
  - `load_token_stream(path)` -- a backward-compatible drop-in that returns the SAME `list[list[str]]` the old
    whole-file loaders returned (so the existing permutation-based consumers -- which need `len()` + random
    access -- keep working VERBATIM), but builds that list via `iter_stories`, so it NEVER does a single
    whole-file `fh.read()`. (The token-list materialization is itself O(corpus) in RAM, but the *raw text* is
    never fully resident; the streaming `iter_stories` is the path that is bounded for arbitrarily large
    corpora.)

Tokenization is byte-identical to the prior loaders: lowercase is assumed already done in the cached corpus,
each story tokenized with `re.findall(r"[a-z]+", story)`, stories split on the literal `<|endoftext|>`.

Stdlib + numpy-free. NO `sim/` edit (this is a runner-side helper).
"""
from __future__ import annotations

import os
import re
from typing import Iterator, List, Optional, Sequence, Union

# The delimiter TinyStories (and the wired corpus) use between documents.
_DOC_DELIM = "<|endoftext|>"
_TOKEN_RE = re.compile(r"[a-z]+")

# The default corpus path -- the SAME hardcoded file the old loaders used, so the new `--corpus-path` arg
# defaults to byte-identical behavior (backward-compatible fallback). Resolved relative to the repo root
# (this file lives at <repo>/research/runners/corpus_stream.py).
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_CORPUS_PATH = os.path.join(_REPO_ROOT, "data", "corpus", "tinystories.txt")


def default_corpus_path() -> str:
    """The backward-compatible default corpus path (the file the hardcoded loaders read)."""
    return DEFAULT_CORPUS_PATH


def _tokenize(story: str) -> List[str]:
    """Tokenize ONE story exactly as the prior loaders did: `re.findall(r"[a-z]+", story)`."""
    return _TOKEN_RE.findall(story)


def iter_stories(path: Optional[str] = None, chunk_bytes: int = 1 << 20,
                 skip_empty: bool = True) -> Iterator[List[str]]:
    """STREAM stories from a corpus file WITHOUT loading the whole file into memory.

    Reads `path` in fixed `chunk_bytes` byte chunks; maintains a small carry buffer for the story that straddles
    a chunk boundary; splits on the `<|endoftext|>` document delimiter; yields each completed story's token list
    (`list[str]`). Peak resident raw text is bounded by ~`chunk_bytes` + the length of one document --
    INDEPENDENT of the total file size, so this scales to a corpus larger than RAM.

    Args:
        path: corpus file path (default: the wired TinyStories cache).
        chunk_bytes: bytes per read (default 1 MiB). Smaller exercises more boundary-splitting; the result is
            chunk-size-INDEPENDENT (the same stories regardless of chunk_bytes).
        skip_empty: if True (default), stories that tokenize to zero tokens are not yielded (matches the
            consumers, which drop empty token lists anyway).

    Yields:
        list[str]: the lowercased `[a-z]+` tokens of one story, in order.
    """
    if path is None:
        path = DEFAULT_CORPUS_PATH
    buf = ""
    # newline="" + errors="ignore" mirrors the prior loaders' read(encoding="utf-8", errors="ignore").
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            buf += chunk
            # Emit every COMPLETE story in the buffer; keep the trailing partial (no delimiter yet) for the
            # next chunk. This is what makes the read bounded: we never hold more than one chunk + one partial.
            parts = buf.split(_DOC_DELIM)
            buf = parts.pop()            # the last part may be an incomplete story -> carry it forward
            for story in parts:
                toks = _tokenize(story)
                if toks or not skip_empty:
                    yield toks
    # the final trailing story (text after the last delimiter, or a delimiter-free file)
    toks = _tokenize(buf)
    if toks or not skip_empty:
        yield toks


def load_token_stream(path: Optional[str] = None, chunk_bytes: int = 1 << 20,
                      max_stories: Optional[int] = None) -> List[List[str]]:
    """Backward-compatible loader: return `list[list[str]]` (one token list per story), the SAME shape the old
    whole-file `load_token_stream`/`_load_token_stream` returned -- but built via the chunked `iter_stories`
    generator, so the raw file is NEVER read whole into one string.

    The existing stream-cortex consumers do `rng.permutation(len(stories))` + index into `stories[si]`, so they
    need a materialized list (random access + length). This preserves that interface exactly while removing the
    whole-file `fh.read()`. For the large-corpus SCALING path (sequential, bounded memory) callers should use
    `iter_stories` directly.

    Args:
        path: corpus file path (default: the wired TinyStories cache).
        chunk_bytes: bytes per chunked read (forwarded to `iter_stories`).
        max_stories: optional cap on the number of stories materialized (bounds RAM for the materialized list on
            a very large corpus; None = all stories, matching the old behavior).
    """
    stories: List[List[str]] = []
    for i, toks in enumerate(iter_stories(path, chunk_bytes=chunk_bytes, skip_empty=False)):
        if max_stories is not None and i >= max_stories:
            break
        stories.append(toks)
    return stories


# ============================================================================================================
# COMBINED-CORPUS path (Rung-1 of the knowledge-scaling ladder, `_knowledge_scaling_first_chat_scoping.md` §3):
# stream / aggregate MULTIPLE corpus files (TinyStories for clean codes + Wikipedia for breadth) so the derived
# vocab + co-occurrence come from the UNION. PURELY ADDITIVE: the single-path `iter_stories`/`load_token_stream`
# above are byte-unchanged; these helpers compose them. Each file is split on its OWN `<|endoftext|>` document
# delimiter (a file with none -- e.g. the cached `wikitext.txt` -- streams as ONE document, which is correct for
# the seed-independent token-frequency + WINDOW-local co-occurrence the curriculum derivation uses).
# ============================================================================================================


def normalize_corpus_paths(paths: Union[None, str, Sequence[str]]) -> List[str]:
    """Resolve a corpus-path argument into a clean ordered list of file paths (the COMBINED-corpus contract).

    Accepts (so callers can keep a single ``--corpus-path`` string OR pass a combined ``--corpus-paths``):
      - ``None``                       -> ``[DEFAULT_CORPUS_PATH]`` (the byte-identical TinyStories fallback).
      - ``str``                        -> a comma- or os.pathsep-separated string is split; a bare path -> ``[path]``.
      - ``Sequence[str]`` (list/tuple) -> stripped of empties, order + duplicates preserved (deliberate: a corpus
        may be weighted by listing it twice; aggregation is order-independent for frequency but the materialized
        story list concatenates in the given order).

    A SINGLE bare path string returns ``[that_path]`` -> the single-corpus callers are byte-identical.
    """
    if paths is None:
        return [DEFAULT_CORPUS_PATH]
    if isinstance(paths, str):
        # split on comma OR the OS path separator; a bare single path (no delimiter) yields exactly [paths].
        raw = re.split(r"[,%s]" % re.escape(os.pathsep), paths)
        out = [p.strip() for p in raw if p.strip()]
        return out if out else [DEFAULT_CORPUS_PATH]
    # a sequence of paths
    out = [str(p).strip() for p in paths if str(p).strip()]
    return out if out else [DEFAULT_CORPUS_PATH]


def iter_stories_multi(paths: Union[None, str, Sequence[str]] = None, chunk_bytes: int = 1 << 20,
                       skip_empty: bool = True) -> Iterator[List[str]]:
    """STREAM stories across MULTIPLE corpus files, concatenated in order (each file split on its OWN
    `<|endoftext|>` delimiter via the unchanged single-file `iter_stories`). Peak resident raw text stays bounded
    by ~`chunk_bytes` + one document of ONE file (files are streamed sequentially, never concatenated in RAM).

    A single bare path (str without a delimiter) is exactly `iter_stories(path)` -- byte-identical. Yields one
    story's token list at a time, in file order.
    """
    for p in normalize_corpus_paths(paths):
        for toks in iter_stories(p, chunk_bytes=chunk_bytes, skip_empty=skip_empty):
            yield toks


def load_token_stream_multi(paths: Union[None, str, Sequence[str]] = None, chunk_bytes: int = 1 << 20,
                            max_stories: Optional[int] = None) -> List[List[str]]:
    """Backward-compatible COMBINED loader: return `list[list[str]]` (one token list per story) across MULTIPLE
    corpus files, concatenated in the given order. Built via `iter_stories_multi` (per-file chunked streaming, so
    no file is read whole into one string). `skip_empty=False` matches `load_token_stream` (consumers drop empty
    token lists themselves) so a SINGLE-path call is byte-identical to `load_token_stream`.

    Args:
        paths: a single path, a comma/os.pathsep-separated string, or a sequence of paths (default: TinyStories).
        chunk_bytes: bytes per chunked read (forwarded to `iter_stories`).
        max_stories: optional cap on TOTAL materialized stories across all files (bounds RAM; None = all).
    """
    stories: List[List[str]] = []
    for i, toks in enumerate(iter_stories_multi(paths, chunk_bytes=chunk_bytes, skip_empty=False)):
        if max_stories is not None and i >= max_stories:
            break
        stories.append(toks)
    return stories
