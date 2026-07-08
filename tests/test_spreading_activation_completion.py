"""CI guard for spreading-activation semantic completion (open-world inference beyond stored structure, CYCLE
1052/1053): an ADJACENT unknown (no stored fact) gets a HEDGED best-guess property via its nearest learned-code
neighbour; a DERANGED neighbourhood collapses it to chance; a DISJOINT unknown still hard-abstains (moat preserved).
Locks in the mechanism against regression. numpy-only, offline (synthetic stream); the real-code check skips if the
Simple-Wiki defs cache is absent.
"""
import os
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
from research.runners._realcorpus_spreading_activation_completion_derisk import run_seed


def test_synthetic_completion_and_moat():
    """Synthetic 4-category stream: adjacent unknown guessed correctly + coverage; deranged collapses to chance;
    disjoint hard-abstains; confidence tracks tightness."""
    r = run_seed(42, real=False)
    assert r["acc"] >= 0.85, r                       # adjacent unknown -> correct category property
    assert r["cover"] >= 0.85, r                     # adjacent unknowns get a guess
    assert r["deranged"] < 0.45, r                   # shuffled codes -> chance (learned similarity is load-bearing)
    assert r["disjoint_abstain"] >= 0.85, r          # disjoint novel code -> hard-abstain (moat preserved)
    assert r["conf_gap"] > 0.15, r                   # confidence tracks neighbourhood tightness


_DEFS = "research/findings/raw/_simplewiki_defs.json"


@pytest.mark.skipif(not os.path.exists(_DEFS), reason="needs the cached Simple-Wiki defs (regenerable)")
def test_real_code_completion_and_moat():
    """Real Simple-Wiki codes, 4 well-separated domains (mammal/tree/vehicle/tool): the mechanism holds."""
    r = run_seed(42, real=True)
    assert r["acc"] >= 0.85, r
    assert r["cover"] >= 0.85, r
    assert r["deranged"] < 0.45, r
    assert r["disjoint_abstain"] >= 0.85, r
    assert r["conf_gap"] > 0.15, r
