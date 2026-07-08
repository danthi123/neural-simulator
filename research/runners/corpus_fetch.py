"""Authorized public-corpus fetch + cache for Generator-S (Task 3).

Generator-S trains a subword spiking LM on a real public corpus and
tests whether it generates coherent held-out text. This module fetches
and caches an authorized public/open-source training corpus (the kind
open-weights LLMs use).

Network access is permitted ONLY at fetch time. The fetch is idempotent
(a non-empty cache is loaded, never re-downloaded) and degrades
gracefully offline to the local ``data/tinyshakespeare.txt`` corpus that
ships with the repository. A shakespeare-only run is an HONEST weaker
claim and the gate JSON must record ``degraded=True`` so downstream code
can report it accurately.

Stdlib only (urllib.request, os, pathlib, re). No external deps. ASCII
prints only. Never raises on network failure -- it degrades instead.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["clean_text", "split_corpus", "fetch_corpus"]

# Known public-corpus sources. Each maps a short name to a plain-text URL.
# - tinystories: TinyStories V2 GPT4 validation split (small, clean English)
# - wikitext   : wikitext-2 train split mirrored as plain text by the
#                pytorch/examples repo (the HF parquet is NOT plain text)
_KNOWN_SOURCES = {
    "tinystories": (
        "https://huggingface.co/datasets/roneneldan/TinyStories/"
        "resolve/main/TinyStoriesV2-GPT4-valid.txt"
    ),
    "wikitext": (
        "https://raw.githubusercontent.com/pytorch/examples/main/"
        "word_language_model/data/wikitext-2/train.txt"
    ),
}

# Local degraded fallback. Resolved relative to the repo root (this file
# lives at <repo>/research/runners/corpus_fetch.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_FALLBACK = _REPO_ROOT / "data" / "tinyshakespeare.txt"


def clean_text(raw: str) -> str:
    """Normalize text to printable ASCII with single-space whitespace.

    Keeps only printable ASCII (codepoints 32..126) plus newline; every
    other character (control chars, non-ASCII) is dropped. Any run of
    whitespace -- including the kept newlines -- collapses to a single
    space. The result is stripped. Fully deterministic so a BPE
    word-split + decode roundtrip is well-defined.
    """
    if not raw:
        return ""
    # Keep printable ASCII + newline; drop everything else (control,
    # non-ASCII). A dropped char that is ITSELF whitespace (e.g. tab
    # 0x09, carriage return 0x0D) is replaced by a space rather than
    # deleted, so it still separates adjacent words; the collapse step
    # below merges the run. Non-whitespace non-printables are deleted.
    kept = []
    for ch in raw:
        o = ord(ch)
        if 32 <= o <= 126 or ch == "\n":
            kept.append(ch)
        elif ch.isspace():
            kept.append(" ")
    # Collapse any whitespace run (kept newlines, spaces, and the
    # space-substituted tabs/CRs above) to a single space, then strip.
    # LOWERCASE to honor the downstream tokenizer contract: corpus_stream's
    # `re.findall(r"[a-z]+", ...)` assumes pre-lowercased text, so a capital
    # first letter (proper nouns, sentence-initial words) would otherwise be
    # dropped ("Lily"->"ily", "Once"->"nce"). Case-folding is the correct
    # normalization for the co-occurrence / small-LM consumers.
    return re.sub(r"\s+", " ", "".join(kept)).strip().lower()


def split_corpus(text: str, heldout_frac: float = 0.1) -> tuple[str, str]:
    """Deterministic contiguous train/heldout split.

    ``train`` is the first ``(1 - heldout_frac)`` fraction of the text by
    character length; ``heldout`` is the remaining tail. No randomness;
    the split is reproducible and the two halves are disjoint and
    contiguous (``train + heldout == text``).

    Guards (never crash):
      - text shorter than ~10 chars     -> ("", "") if empty, else (text, "")
      - heldout_frac not in (0, 1)      -> (text, "")
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    n = len(text)
    # Too short to split meaningfully -> everything is train.
    if n < 10:
        return text, ""
    # Invalid fraction -> everything is train.
    if not (0.0 < heldout_frac < 1.0):
        return text, ""
    cut = int(round(n * (1.0 - heldout_frac)))
    # Keep both sides non-empty for a sane split.
    if cut <= 0:
        cut = 1
    if cut >= n:
        cut = n - 1
    return text[:cut], text[cut:]


def _read_local_fallback() -> str:
    """Read + clean the bundled tinyshakespeare corpus (degraded path)."""
    try:
        raw = _LOCAL_FALLBACK.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return clean_text(raw)


def _degraded_result(name: str) -> dict:
    """Build the degraded (local shakespeare) result dict."""
    text = _read_local_fallback()
    return {
        "text": text,
        "path": str(_LOCAL_FALLBACK),
        "name": name,
        "degraded": True,
        "corpus_used": "tinyshakespeare(local-degraded)",
        "n_chars": len(text),
        "source": "local",
    }


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically (tmp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, path)


def fetch_corpus(
    name: str = "tinystories",
    max_bytes: int = 8_000_000,
    out_dir: str = "data/corpus",
    timeout: int = 30,
) -> dict:
    """Fetch + cache an authorized public corpus (idempotent, offline-safe).

    Behaviour:
      1. If ``<out_dir>/<name>.txt`` exists and is non-empty: load it and
         return immediately (NEVER re-download a cached corpus).
      2. Else, if ``name`` is a known source, stream-download at most
         ``max_bytes`` with the given ``timeout``, clean it, and write it
         atomically to the cache path.
      3. If ``name`` is unknown, treat it as a local file path and load
         that file (cleaned).
      4. On ANY network/URL error, OR if the resulting text is empty:
         degrade to the bundled ``data/tinyshakespeare.txt`` with
         ``degraded=True`` and ``corpus_used`` recording the honest
         weaker claim. Never raises on network failure.

    Returns a dict with keys: text, path, name, degraded, corpus_used,
    n_chars, source.
    """
    out_path = Path(out_dir) / f"{name}.txt"

    # 1. Idempotent cache hit -- never re-download.
    try:
        if out_path.is_file() and out_path.stat().st_size > 0:
            cached_raw = out_path.read_text(encoding="utf-8", errors="ignore")
            cached = clean_text(cached_raw)
            if cached:
                print(
                    "[corpus_fetch] cache hit: %s (%d chars) -- no download"
                    % (out_path, len(cached))
                )
                return {
                    "text": cached,
                    "path": str(out_path),
                    "name": name,
                    "degraded": False,
                    "corpus_used": name,
                    "n_chars": len(cached),
                    "source": "cache",
                }
            # Cache exists but cleans to empty -> fall through to refetch.
            print(
                "[corpus_fetch] cache present but empty after clean: %s"
                % out_path
            )
    except OSError as exc:
        print("[corpus_fetch] cache check failed (%s); attempting fetch" % exc)

    url = _KNOWN_SOURCES.get(name)

    if url is None:
        # Unknown name -> treat as a local file path.
        local_path = Path(name)
        try:
            if local_path.is_file():
                raw = local_path.read_text(encoding="utf-8", errors="ignore")
                cleaned = clean_text(raw)
                if cleaned:
                    print(
                        "[corpus_fetch] loaded local path: %s (%d chars)"
                        % (local_path, len(cleaned))
                    )
                    return {
                        "text": cleaned,
                        "path": str(local_path),
                        "name": name,
                        "degraded": False,
                        "corpus_used": name,
                        "n_chars": len(cleaned),
                        "source": "local",
                    }
            print(
                "[corpus_fetch] unknown name and no readable local file: %s "
                "-- degrading to local shakespeare" % name
            )
        except OSError as exc:
            print(
                "[corpus_fetch] local path read failed (%s) -- degrading" % exc
            )
        return _degraded_result(name)

    # 2. Known source -> attempt a streamed, bounded download.
    print("[corpus_fetch] downloading %s from %s" % (name, url))
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "neural-simulator-corpus-fetch/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_bytes)
        raw = data.decode("utf-8", errors="ignore")
        cleaned = clean_text(raw)
        if not cleaned:
            print(
                "[corpus_fetch] downloaded text empty after clean -- "
                "degrading to local shakespeare"
            )
            return _degraded_result(name)
        try:
            _atomic_write(out_path, cleaned)
            print(
                "[corpus_fetch] cached %s -> %s (%d chars)"
                % (name, out_path, len(cleaned))
            )
            written_path = str(out_path)
        except OSError as exc:
            # Download succeeded but we could not persist the cache.
            # Still return the fetched text; just note the source path.
            print("[corpus_fetch] cache write failed (%s); using in-memory" % exc)
            written_path = url
        return {
            "text": cleaned,
            "path": written_path,
            "name": name,
            "degraded": False,
            "corpus_used": name,
            "n_chars": len(cleaned),
            "source": url,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError, TimeoutError) as exc:
        print(
            "[corpus_fetch] network/url error (%s) -- degrading to "
            "local shakespeare" % exc
        )
        return _degraded_result(name)


if __name__ == "__main__":
    import json
    import sys

    _name = sys.argv[1] if len(sys.argv) > 1 else "tinystories"
    _res = fetch_corpus(name=_name)
    _summary = {k: v for k, v in _res.items() if k != "text"}
    _tr, _ho = split_corpus(_res["text"], heldout_frac=0.1)
    _summary["train_chars"] = len(_tr)
    _summary["heldout_chars"] = len(_ho)
    print(json.dumps(_summary, indent=2))
