"""CI guard for the EMERGENCE-BAR close of the reservoir-generation ladder (2026-07-11): the ladder's category structure
is DISCOVERED from experience, not a hand class-label. Pins the two headline mechanism facts at seed 42 (kept as fast as
a couple of small reservoir builds; the full 6-seed results live in the runners + findings). Skip if deps missing.
  (1) STEP 1 — a FIXED random codon (Marr-Albus F.12) of category-structured features surfaces a category the ladder
      generalizes on; the DISCOVERY-SCRAMBLE (destroy the feature co-occurrence) collapses it. `nolearn` >> `scramble`.
  (2) PERCEPTION-GROUNDED — the category is discovered from REAL PERCEPTION (objects seen through Gabor/V1); the load-
      bearing PIXEL-SCRAMBLE control (destroy the visual similarity) collapses it. `main` > `scramble`.
Both are the LOAD-BEARING discovery controls the adversarial-verify workflow certified as the honest evidence.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pytest

pytest.importorskip("sim.bridge")
_ec = pytest.importorskip("research.runners._emerge_reservoir_lm_emergent_category_codes_derisk")


def test_step1_fixed_codon_of_feature_statistics_beats_discovery_scramble():
    """A fixed random codon (POOL_EPOCHS=0) of the category-structured features generalizes on the ladder; destroying the
       feature co-occurrence structure (scramble) collapses it. This is the honest step-1 mechanism (the competitive SP
       learning is NOT needed -- the fixed codon suffices)."""
    nl = _ec.run_arm(42, "nolearn", epochs=200, lr=0.05, n_pool=300)
    sc = _ec.run_arm(42, "scramble", epochs=200, lr=0.05, n_pool=300)
    assert nl["heldagent_cat_acc"] >= 0.80, nl
    assert nl["heldagent_cat_acc"] - sc["heldagent_cat_acc"] >= 0.30, (nl, sc)


def test_perception_grounded_category_beats_pixel_scramble():
    """The category is discovered from REAL PERCEPTION (Gabor/V1); the per-image pixel-scramble (destroy the visual
       similarity, isolating the visual shape as the cause) collapses it."""
    _pg = pytest.importorskip("research.runners._emerge_reservoir_lm_perception_grounded_codes_derisk")
    main = _pg.run_arm(42, "main", epochs=200, lr=0.05, n_pool=300)
    scr = _pg.run_arm(42, "scramble", epochs=200, lr=0.05, n_pool=300)
    assert main["heldagent_cat_acc"] > scr["heldagent_cat_acc"], (main, scr)
    assert main["heldagent_cat_acc"] >= 0.65, main


def test_untrained_readout_is_floor():
    """Sanity: the one-step-local-delta read-out is doing the learning (a frozen read-out emits nothing)."""
    ut = _ec.run_arm(42, "untrained", epochs=200, lr=0.05, n_pool=300)
    assert ut["heldagent_cat_acc"] == 0.0, ut
