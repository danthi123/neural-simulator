"""Smoke gate for the co-resident CA3 pattern-completion + consolidation slice on the merged nav/conv one-brain
(`build_merged_nav_conv_bridge(..., co_resident_hippo_memory=True)`).

This is the CONSTRUCTION smoke ONLY (the controller runs the multi-seed formation + completion): it proves the
compact hippo subset (ca3 recurrent attractor + ca1 Schaffer read-out + ca3_pv_basket FS sparsifier) BUILDS and
CO-RESIDES as a disjoint slice, and that the DEFAULT-OFF path is BYTE-IDENTICAL (the additive flag changes nothing
when off — the coincidence/two-compartment-dAP cfg fields are all guarded by the flag).

GPU-gated (mirrors tests/test_riii_emergent_ca3_completion.py): the merged bridge + the coincidence-plateau regime
are cupy-validated; numpy is a tiny-smoke/CI path only. Kept SMALL/fast: a 6-word vocab + hippo_n_ca3=200.
"""
import pytest


def _gpu():
    try:
        from sim.backend import get_backend, is_gpu_backend
        get_backend()
        return is_gpu_backend()
    except Exception:
        return False


# a small vocab keeps the parser train pass + dlPFC fast (the build is the slow part, not the hippo slice)
_VOCAB = ["dog", "cat", "go", "come", "north", "south"]
_N_CA3 = 200
_N_CA1 = 120
_N_BASKET = max(8, int(0.25 * _N_CA3))   # the ca3_pv_basket sizing rule (== 50 at n_ca3=200)


def _totals(bridge):
    rm = bridge.region_manager
    return (int(rm.total_neurons()),
            int(bridge.cp_firing_states.shape[0]),
            int(bridge.cp_connections.nnz),
            frozenset(rm.region_indices_dict().keys()))


@pytest.mark.skipif(not _gpu(), reason="merged bridge + coincidence-plateau slice are cupy-validated (GPU-gated)")
def test_default_off_byte_identical():
    """The additive flag OFF changes NOTHING: total neuron count, region names, and cp_connections.nnz are
    byte-identical between the current default (flag defaults to False) and an explicit co_resident_hippo_memory=False."""
    from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
    b_default, _ = build_merged_nav_conv_bridge(seed=42, vocab=_VOCAB)
    b_flag_off, _ = build_merged_nav_conv_bridge(seed=42, vocab=_VOCAB, co_resident_hippo_memory=False)
    tot_d, nf_d, nnz_d, names_d = _totals(b_default)
    tot_f, nf_f, nnz_f, names_f = _totals(b_flag_off)
    assert tot_f == tot_d, (tot_f, tot_d)                        # same neuron count
    assert nf_f == nf_d, (nf_f, nf_d)                            # same allocated state array
    assert nnz_f == nnz_d, (nnz_f, nnz_d)                        # same wiring (byte-identical connectivity)
    assert names_f == names_d, (names_f ^ names_d)               # same region set
    assert "ca3" not in names_d, names_d                         # no hippo regions leaked into the default build


@pytest.mark.skipif(not _gpu(), reason="merged bridge + coincidence-plateau slice are cupy-validated (GPU-gated)")
def test_hippo_regions_present():
    """With co_resident_hippo_memory=True the ca3 / ca1 / ca3_pv_basket regions build with the right sizes and
    co-reside as a DISJOINT slice (their indices do not overlap the nav cascade / parser / dlPFC regions)."""
    from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
    b_on, _ = build_merged_nav_conv_bridge(seed=42, vocab=_VOCAB,
                                           co_resident_hippo_memory=True, hippo_n_ca3=_N_CA3, hippo_n_ca1=_N_CA1)
    rid = b_on.region_manager.region_indices_dict()

    # the three hippo regions are present at the expected sizes
    assert len(rid.get("ca3", [])) == _N_CA3, rid.get("ca3")
    assert len(rid.get("ca1", [])) == _N_CA1, rid.get("ca1")
    assert len(rid.get("ca3_pv_basket", [])) == _N_BASKET, rid.get("ca3_pv_basket")

    # the ca3 slice is DISJOINT from the nav cascade / parser / dlPFC (co-resident, not overlapping)
    ca3 = set(rid["ca3"])
    for other in ("cortex_N", "parse_conj", "parse_role", "dlpfc_wm"):
        assert other in rid, f"{other} missing from the merged bridge"
        assert ca3.isdisjoint(set(rid[other])), f"ca3 overlaps {other}"

    # and the three hippo regions are mutually disjoint
    ca1 = set(rid["ca1"]); basket = set(rid["ca3_pv_basket"])
    assert ca3.isdisjoint(ca1) and ca3.isdisjoint(basket) and ca1.isdisjoint(basket)

    # the coincidence-plateau config is switched ON (only when the flag is set)
    cc = b_on.core_config
    assert cc.enable_coincidence_detection is True
    assert cc.enable_two_compartment_dap is True
    assert cc.coincidence_weighted_drive is True
