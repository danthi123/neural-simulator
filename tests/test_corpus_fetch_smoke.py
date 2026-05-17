import importlib, inspect
import numpy as np
from research.runners.corpus_fetch import clean_text, split_corpus, fetch_corpus

def test_clean_text_collapses_whitespace_ascii_only():
    out = clean_text("Hello\n\n  world\tthere\x00!")
    assert out == "Hello world there!"            # ws->single space, ctrl dropped
    assert all(31 < ord(c) < 127 or c == " " for c in out)

def test_split_corpus_is_deterministic_disjoint_and_contiguous():
    text = "abcdefghij" * 10                       # 100 chars
    tr, ho = split_corpus(text, heldout_frac=0.1)
    assert len(tr) == 90 and len(ho) == 10
    assert tr + ho == text                          # contiguous, disjoint
    assert split_corpus(text, 0.1) == (tr, ho)      # deterministic
    # degenerate inputs do not crash
    assert isinstance(split_corpus("ab", 0.1), tuple)

def test_fetch_signature_and_offline_degrades_to_shakespeare(monkeypatch, tmp_path):
    # force every network attempt to fail -> must degrade to local
    # tinyshakespeare with degraded=True, NOT raise.
    import urllib.request
    def boom(*a, **k):
        raise OSError("network disabled for test")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    r = fetch_corpus(name="tinystories", max_bytes=10000,
                     out_dir=str(tmp_path))
    assert isinstance(r, dict)
    assert {"text","path","name","degraded","corpus_used",
            "n_chars","source"} <= set(r)
    # offline -> degraded fallback to local shakespeare (present in repo)
    assert r["degraded"] is True
    assert r["n_chars"] > 0 and isinstance(r["text"], str)

def test_fetch_is_idempotent_when_cached(tmp_path):
    p = tmp_path / "tinystories.txt"
    p.write_text("cached corpus text here .", encoding="utf-8")
    r = fetch_corpus(name="tinystories", out_dir=str(tmp_path))
    assert r["degraded"] is False
    assert "cached corpus text here" in r["text"]   # used cache, no download
