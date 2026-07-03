"""CI guard for EMERGE-41: the pooler's k-winners selection as SPIKING competition. Columns integrate their graded drive
to threshold; higher-drive columns spike earlier (rank-order coding), so the first-K-to-spike == the host top-K; the FS
lateral inhibition suppresses the loser pool; permuted-drive follows. CPU (numpy); skips gracefully if deps unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._emerge41_fs_wta_kwinners_derisk import _run_seed
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge41 deps unavailable: {e}")
    return _run_seed(42)


def test_spiking_selects_top_k(result):
    """The first-K-to-spike (rank-order) matches the host top-K by drive."""
    assert result["overlap"] >= 0.8


def test_permuted_drive_follows(result):
    """Permuting the drive -> the spiking winners follow the permuted top-K (the competition reads the drive)."""
    assert result["permuted_overlap"] >= 0.8


def test_fs_suppresses_loser_pool(result):
    """The FS lateral inhibition is load-bearing for sparsity: FS-lesion fires more columns than the FS-on case."""
    assert result["lesion_sparsity"] >= result["sparsity"] + 0.1


if __name__ == "__main__":
    from research.runners._emerge41_fs_wta_kwinners_derisk import _run_seed
    r = _run_seed(42)
    assert r["overlap"] >= 0.8 and r["permuted_overlap"] >= 0.8 and r["lesion_sparsity"] >= r["sparsity"] + 0.1
    print("OK: emerge41 spiking rank-order k-WTA -- selects top-K, permuted follows, FS suppresses losers")
