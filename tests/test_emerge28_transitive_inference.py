"""CI guard for EMERGE-28 transitive inference: from only adjacent premises (A>B, B>C, C>D, D>E) the never-trained
non-adjacent relations (B>D, ...) are inferred by chaining overlapping premises into an integrated order on the spiking
HTM cortex. CPU (numpy); skips gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge28_transitive_inference_derisk import TransitiveProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge28 deps unavailable: {e}")
    return TransitiveProbe(seed=42, epochs=80)


def test_critical_internal_pair(probe):
    """B>D was NEVER trained (only B>C, C>D) and is unsolvable by associative strength -> the genuine TI signal."""
    assert probe.greater("B", "D") is True
    assert probe.greater("D", "B") is False


def test_all_nonadjacent_inferred(probe):
    """Every never-trained non-adjacent pair is inferred with the correct order."""
    from research.runners._emerge28_transitive_inference_derisk import NONADJ
    assert all(probe.judge(pair) for pair in NONADJ)


def test_broken_chain_collapses_internal():
    """Dropping the middle premise (C>D) makes B and D uncomparable -> B>D collapses (isolates transitive chaining)."""
    from research.runners._emerge28_transitive_inference_derisk import TransitiveProbe, PREMISES
    prem = [p for p in PREMISES if p != ("C", "D")]
    p = TransitiveProbe(seed=42, epochs=80, premises=prem)
    assert p.greater("B", "D") is False


if __name__ == "__main__":
    from research.runners._emerge28_transitive_inference_derisk import TransitiveProbe
    pr = TransitiveProbe(seed=42, epochs=80)
    test_critical_internal_pair(pr); test_all_nonadjacent_inferred(pr); test_broken_chain_collapses_internal()
    print("OK: emerge28 transitive inference -- non-adjacent inferred + critical B>D + broken-chain collapse")
