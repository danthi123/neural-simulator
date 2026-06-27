"""Regression guard for REGIME-B B1 — a CORPUS-MINED ORDINAL RELATION AXIS.

The qualitatively-STRONGER successor to Tier 2.3 (transitive inference): there the ordinal axis was GIVEN
(hand-coded `ADJ_PAIRS`); here the SIZE axis is MINED FROM THE CORPUS over the brain's OWN learned vocab
(scalar-adjective co-occurrence) -> the SAME Tier 2.3 ordinal-map objective -> held-out unstated comparisons.
De-risked GO host-path 6 seeds + spiking-accumulator 3 seeds
(2026-06-27-regimeB-corpus-mined-axis-GO.md). This pins the load-bearing signatures, including the NEW decisive
regime-B control the GIVEN-structure capabilities could not have:

  - HELD-OUT unstated comparisons (graded vs an EXTERNAL ground-truth size order) >> chance AND >> mem-floor;
  - THE SYMBOLIC-DISTANCE EFFECT: decision MARGIN rises monotonically with ground-truth distance (curve rho>0);
  - ** PERMUTED-MINING **: mining a SCRAMBLED relation (size-adjectives relabelled onto random items) collapses
    to ~chance -> the corpus-attested premises, NOT the mining apparatus, carry the order (structure ACQUIRED);
  - PERMUTED-ORDER collapses + the mined order is in the TOP ~2% of orderings for predicting the GT held-out;
  - LESION (scramble the map) collapses to chance;
  - the SPREADING-ACTIVATION control (symmetric co-occurrence over the mined premises) is at chance on the ORDER;
  - PROVENANCE / no train-test leak: held-out pairs are never adjacent mined premises;
  - the no-confab MOAT: an item never placed on the axis abstains (0-FA).

Requires the corpus (data/corpus/simplewiki.txt) + the brain NPZ (bridges/firstchat/brainALL_w7000.npz_seed42.npz);
skips gracefully if either is absent (mirrors the on-brain tests' skip-if-cache-absent discipline). CPU/numpy.
The spiking-accumulator accuracy-distance psychometric curve is validated in the de-risk runner + findings; this
guard pins the structural + mining controls on the host path.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

from research.runners._regimeb_corpus_mined_axis_derisk import run_seed, GT_ORDER

_CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "corpus", "simplewiki.txt")
_NPZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bridges", "firstchat", "brainALL_w7000.npz_seed42.npz")
SEEDS = [42, 43, 44]

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_CORPUS) and os.path.exists(_NPZ)),
    reason="needs data/corpus/simplewiki.txt + bridges/firstchat/brainALL_w7000.npz_seed42.npz")


@pytest.fixture(scope="module")
def rows():
    # The mining is corpus-budget-dependent (the honest scope: at HALF the corpus the mined order degrades to
    # rho~0.19 with ground-truth and the held-out falls below the gate). Use the VALIDATED full-corpus operating
    # point (80MB) -- the de-risk's GO budget. Cheap regardless: the 80MB regex-tokenize is ~3.4s, paid ONCE
    # (mining cached across seeds inside run_seed; subsequent seeds ~0.1s). Total ~4s for 3 seeds.
    return [run_seed(s, _CORPUS, _NPZ, use_spiking=False, max_chars=80_000_000)[0] for s in SEEDS]


def test_held_out_beats_chance_and_memorization_floor(rows):
    held = float(np.mean([r["held_out_acc"] for r in rows]))
    mem = float(np.mean([r["mem_floor"] for r in rows]))
    assert held >= 0.7, f"held-out {held:.2f} not >> chance"
    assert held >= mem + 0.15, f"held-out {held:.2f} not >> mem-floor {mem:.2f}"


def test_symbolic_distance_effect_margin_rises(rows):
    """The headline control: the map's margin rises monotonically with ground-truth ordinal distance."""
    assert all(r["rho_margin"] > 0.0 for r in rows), \
        f"margin-distance rho not positive on every seed: {[r['rho_margin'] for r in rows]}"


def test_permuted_mining_collapses(rows):
    """THE decisive regime-B control (BOTH variants must collapse): (1) PERMUTE the mined scores across items
    (= size-adjectives attached to random items); (2) RE-MINE from the corpus with the size-adjectives RELABELLED
    onto RANDOM in-vocab words (the spec's exact 'random word pairs labelled bigger'). Either way the scrambled-
    relation axis must NOT predict the GT held-out -> the corpus-attested premises, not the apparatus, carry the
    order (structure ACQUIRED, not given -- the control Tier 2.3's GIVEN structure could not have)."""
    pm = float(np.mean([r["permuted_mining_acc"] for r in rows]))
    pm_relabel = float(np.mean([r["permuted_mining_relabel_acc"] for r in rows]))
    assert pm <= 0.62, f"PERMUTED-MINING (perm-score) did not collapse ({pm:.2f}) -- the mining is not load-bearing"
    assert pm_relabel <= 0.62, \
        f"PERMUTED-MINING (relabel-adjectives) did not collapse ({pm_relabel:.2f}) -- the mining is not load-bearing"


def test_permuted_order_collapses_and_mined_is_top2pct(rows):
    perm = float(np.mean([r["perm_order_acc"] for r in rows]))
    assert perm <= 0.65, f"permuted order did not collapse ({perm:.2f})"
    assert all(r["true_top2pct"] for r in rows), "mined order is not in the top ~2% of orderings (every seed)"


def test_lesion_collapses(rows):
    les = float(np.mean([r["lesion_acc"] for r in rows]))
    assert les <= 0.65, f"lesion (scrambled map) did not collapse to chance (mean {les:.2f})"


def test_spreading_control_fails(rows):
    sym = float(np.mean([r["spread_sym_acc"] for r in rows]))
    assert sym <= 0.65, f"symmetric spreading should be ~chance on the order 2AFC (got {sym:.2f})"


def test_provenance_no_leak(rows):
    """Every held-out pair was asserted (inside run_seed) to never be an adjacent mined premise."""
    assert all(r["no_leak"] for r in rows), "train/test LEAK: a held-out pair was a mined premise"


def test_moat_abstains_on_unmapped_item(rows):
    assert all(r["moat_unmapped_abstains"] for r in rows), "moat breach: an unmapped item was answered"


def test_mined_order_correlates_with_ground_truth(rows):
    """Sanity: the corpus-MINED order is positively rank-correlated with the EXTERNAL ground-truth size order
    (the mining recovers real size structure -- the precondition for the held-out inferences to beat chance)."""
    mined = rows[0]["mined_order"]
    gt_rank = {it: i for i, it in enumerate(GT_ORDER)}
    mined_gt = [gt_rank[it] for it in mined]
    rx = np.argsort(np.argsort(np.arange(len(mined)))).astype(float)
    ry = np.argsort(np.argsort(mined_gt)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    assert rho > 0.4, f"mined order not correlated with ground-truth size ({rho:+.2f})"
