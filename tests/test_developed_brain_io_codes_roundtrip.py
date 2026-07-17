"""Regression: grounded-codes bundle round-trips concept words that collide with numpy kwargs.

2026-07-17. A concept literally named "file" (present in the corpus-mined curriculum vocab) crashed
the per-day develop-loop bundle save under numpy>=2 -- `np.savez_compressed(file, *args,
allow_pickle=..., **kwds)` reserves `file`/`allow_pickle`, so splatting `{**{'file': arr}}` raised
"savez_compressed() got multiple values for argument 'file'". The fix prefixes every concept key
"g:" on save (concepts are [a-z]+, so ":" never appears in a word) and strips it on load, with a
backward-compat fallback for old unprefixed bundles. These tests pin all three properties.
"""

from pathlib import Path

import numpy as np

from research.runners.developed_brain_io import _load_codes_npz


def _save_like_bundle(root, codes):
    """Mirror save_developed_brain's grounded_codes.npz write (the 'g:'-prefixed key format)."""
    np.savez_compressed(
        str(Path(root) / "grounded_codes.npz"),
        **{f"g:{w}": np.asarray(ph, dtype=np.float32) for w, ph in codes.items()},
    )


def test_reserved_name_concepts_roundtrip(tmp_path):
    # 'file' and 'allow_pickle' are the numpy savez reserved kwargs that used to crash the save.
    codes = {"file": [1, 2, 3, 4], "allow_pickle": [9, 9, 9, 9], "dog": [5, 6, 7, 8]}
    _save_like_bundle(tmp_path, codes)  # must not raise
    got = _load_codes_npz(tmp_path)
    assert set(got) == set(codes)
    assert list(got["file"]) == [1, 2, 3, 4]
    assert list(got["allow_pickle"]) == [9, 9, 9, 9]
    assert list(got["dog"]) == [5, 6, 7, 8]


def test_old_unprefixed_bundle_still_loads(tmp_path):
    # An old bundle saved raw word keys (no "g:" prefix); it never contained 'file' (that would have
    # crashed the old save), so backward compat only needs ordinary words to read unchanged.
    np.savez_compressed(
        str(tmp_path / "grounded_codes.npz"),
        cat=np.asarray([1.0, 2.0], np.float32), tree=np.asarray([3.0, 4.0], np.float32),
    )
    got = _load_codes_npz(tmp_path)
    assert set(got) == {"cat", "tree"}
    assert list(got["cat"]) == [1.0, 2.0]


def test_absent_bundle_is_empty(tmp_path):
    assert _load_codes_npz(tmp_path) == {}
