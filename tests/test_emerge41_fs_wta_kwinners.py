"""CI guard for EMERGE-41: the pooler's k-winners SELECTION as SPIKING RANK-ORDER (latency) coding. Columns integrate their
graded drive to threshold; higher-drive columns spike earlier, so the first-K-to-spike == the host top-K. A FLAT (uniform)
drive collapses the overlap to the tie-break floor (the SELECTION reads the graded drive). The FS lateral inhibition is
INERT for the selection (winner set ~identical FS-on vs FS-lesion) and only sparsifies the loser pool. CPU (numpy); skips
gracefully if deps unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._emerge41_fs_wta_kwinners_derisk import _run_seed, K_WIN, NCOL
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge41 deps unavailable: {e}")
    r = _run_seed(42)
    r["_chance"] = K_WIN / NCOL
    return r


def test_spiking_selects_top_k(result):
    """The first-K-to-spike (rank-order latency) matches the host top-K by drive."""
    assert result["overlap"] >= 0.8


def test_flat_drive_collapses(result):
    """Input-destruction: a FLAT (uniform, non-graded) drive removes the ranking signal, so the overlap with the host
    top-K collapses toward the tie-break floor (~chance). Isolates that the SELECTION reads the GRADED drive."""
    assert result["flat_overlap"] <= result["_chance"] + 0.15


def test_fs_inert_for_selection(result):
    """The FS is CAUSALLY INERT for WHICH columns win: the winner set is ~identical FS-on vs FS-lesion (its only effect
    is loser-pool sparsity, which is a separate, secondary contribution)."""
    assert result["lesion_winner_overlap"] >= 0.8


def test_fs_suppresses_loser_pool(result):
    """The FS lateral inhibition's ONLY effect: loser-pool sparsity -- FS-lesion fires more columns than the FS-on case."""
    assert result["lesion_sparsity"] >= result["sparsity"] + 0.1


if __name__ == "__main__":
    from research.runners._emerge41_fs_wta_kwinners_derisk import _run_seed, K_WIN, NCOL
    r = _run_seed(42); chance = K_WIN / NCOL
    assert r["overlap"] >= 0.8 and r["flat_overlap"] <= chance + 0.15
    assert r["lesion_winner_overlap"] >= 0.8 and r["lesion_sparsity"] >= r["sparsity"] + 0.1
    print("OK: emerge41 spiking rank-order k-winners -- selects top-K, flat-drive collapses, FS inert for selection + "
          "sparsifies losers")
