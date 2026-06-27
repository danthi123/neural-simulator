"""Regression guard for TRANSITIVE INFERENCE via a learned 1-D ORDINAL MAP (Tier 2.3).

The principled redemption of the project's most-burned RETRACTION (the 2026-05-14 "90% transitive inference"
spreading-activation artifact). De-risked GO host-path 6 seeds + spiking-accumulator 3 seeds
(2026-06-27-tier2.3-transitive-ordinal-map-GO.md). This pins the load-bearing signature -- the one the
artifact provably could NOT fake:

  - HELD-OUT non-adjacent pairs (never trained) >> chance AND >> the stored-edge memorization floor;
  - THE SYMBOLIC-DISTANCE EFFECT: the decision MARGIN rises monotonically with ordinal distance (curve rho>0);
  - PERMUTED-order collapses + the TRUE order is rank-1 (uniquely best);
  - LESION (scramble the map) collapses to chance (mean over seeds);
  - the SPREADING-ACTIVATION controls FAIL their signature: symmetric co-occurrence is at chance on the ORDER
    2AFC, and directed transitive-closure has the WRONG (decreasing) margin curve -- proving the MAP, not
    edge-chaining, is responsible;
  - the no-confab MOAT: an item never placed on the map abstains (0-FA).

CPU/numpy host path (fast, portable). The spiking-accumulator accuracy-distance psychometric curve is validated
in the de-risk runner (--spiking-accumulator) and findings; this guard pins the structural controls.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners._transitive_ordinal_map_derisk import run_seed, ITEMS, ADJ_PAIRS, NONADJ_PAIRS, RANK

SEEDS = [42, 43, 44]


def _rows():
    return [run_seed(s, use_spiking=False) for s in SEEDS]


def test_held_out_beats_chance_and_memorization_floor():
    rows = _rows()
    held = float(np.mean([r["held_out_acc"] for r in rows]))
    mem = float(np.mean([r["mem_floor"] for r in rows]))
    assert held >= 0.8, f"held-out {held:.2f} not >> chance"
    assert held >= mem + 0.25, f"held-out {held:.2f} not >> mem-floor {mem:.2f} (lookup is at chance by construction)"


def test_symbolic_distance_effect_margin_rises():
    """The headline control: the map's margin (position gap) rises monotonically with ordinal distance -- the
    positive signature an edge-lookup / co-occurrence artifact cannot produce."""
    rows = _rows()
    assert all(r["rho_margin"] > 0.0 for r in rows), \
        f"margin-distance rho not positive on every seed: {[r['rho_margin'] for r in rows]}"
    # the margin curve is strictly increasing in distance (the host comparator is the noiseless oracle)
    for r in rows:
        mc = r["map_margin_curve"]
        ds = sorted(mc)
        assert all(mc[ds[i + 1]] > mc[ds[i]] for i in range(len(ds) - 1)), f"margin curve not increasing: {mc}"


def test_permuted_order_collapses_and_true_is_rank1():
    rows = _rows()
    perm = float(np.mean([r["perm_acc"] for r in rows]))
    assert perm <= 0.65, f"permuted order did not collapse ({perm:.2f})"
    assert all(r["true_rank1"] for r in rows), "TRUE order is not uniquely rank-1 across seeds"


def test_lesion_collapses():
    rows = _rows()
    les = float(np.mean([r["lesion_acc"] for r in rows]))
    assert les <= 0.65, f"lesion (scrambled map) did not collapse to chance (mean {les:.2f})"


def test_spreading_controls_fail_their_signature():
    """The retracted family must FAIL: symmetric co-occurrence at chance on the ORDER 2AFC; directed
    transitive-closure with the WRONG (decreasing) margin-distance curve."""
    rows = _rows()
    sym = float(np.mean([r["spread_sym_acc"] for r in rows]))
    dir_rho = float(np.mean([r["spread_dir_rho_margin"] for r in rows]))
    assert sym <= 0.65, f"symmetric spreading should be ~chance on order (got {sym:.2f})"
    assert dir_rho < 0.0, f"directed-closure margin should DECREASE with distance (the wrong curve); got {dir_rho:+.2f}"


def test_moat_abstains_on_unmapped_item():
    rows = _rows()
    assert all(r["moat_unmapped_abstains"] for r in rows), "moat breach: an unmapped item was answered"


def test_only_adjacent_pairs_are_trained():
    """Structural invariant: NONADJ_PAIRS (the tested pairs) are disjoint from ADJ_PAIRS (the trained pairs) --
    every scored pair is genuinely held-out (the exact discipline the 2026-05-14 retraction lacked)."""
    trained = set(ADJ_PAIRS) | {(b, a) for a, b in ADJ_PAIRS}
    for (x, y) in NONADJ_PAIRS:
        assert (x, y) not in trained and (y, x) not in trained, f"{(x, y)} is a trained pair, not held-out"
        assert abs(RANK[x] - RANK[y]) >= 2, f"{(x, y)} is adjacent (distance 1), not a held-out non-adjacent pair"
