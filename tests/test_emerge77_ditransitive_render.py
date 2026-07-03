"""CI for EMERGE-77 -- SURPASS the EMERGE-74 DITRANSITIVE capacity boundary: render "the dog gives the cat a bone"
(7 slots) ON SPIKES by making the FrameCQ slot-pool count CONFIGURABLE (a bounded, additive scale lever; default 6 =
byte-identical, 8 for the ditransitive producer) + a 2-stage per-pool bias-calibrated read.

CPU/numpy, offline. A small-stream smoke: (1) the DEFAULT-6 path is byte-identical (the n_slot_pools threading did NOT
change the shipped FrameSlotCQ); (2) the 8-pool build gives a wider slot bridge + re-spaced primacy; (3) the 7-slot
ditransitive renders EXACT on spikes at 8 pools; (4) all 7 named constructions render exact; (5) the 2-stage read is
load-bearing (the RAW/uncalibrated read fails to order the ditransitive on the saturating seeds); (6) position-
independence holds for the 7-slot frame; (7) the input-destruction controls collapse (permuted-corpus / cross-
construction / no-corpus); (8) the gate-first moat holds (0 productions on abstains). A smaller n_extra keeps it fast.
NO sim/ edit; the EMERGE-59/72/74 defaults are preserved.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FrameSlotCQ, N_SLOT_POOLS, PRIMACY_pA, FRAME_NAMES, N_PER, build_slot_bridge, slot_pool_rates,
)
from research.runners._emerge74_transitive_ditransitive_derisk import (  # noqa: E402
    CONSTRUCTION_NAMES, CONSTRUCTIONS, build_stream_svo, build_heldout_facts_svo,
)
from research.runners._emerge77_ditransitive_render_derisk import (  # noqa: E402
    DitransRegistry, DitransRegistryProducer, DITRANS_POOLS, _FITS_8, _emit_construction,
    _render_registry, _cross_construction, _position_independence, _default_byte_identity,
    RegistryBrocaProducer, decision,
)

_SEED = 42
_N_EXTRA = 3500     # smaller SVO stream for CI speed (the derisk uses 8000)


def _build(seed=_SEED, n_extra=_N_EXTRA):
    tokens = build_stream_svo(seed, n_extra=n_extra)
    reg = DitransRegistry(seed).build(tokens)
    return tokens, reg


# --------------------------------------------------------------------------------------------------------------------
# (1) BYTE-IDENTITY of the default-6 path (the load-bearing "no regression" property of the n_slot_pools threading).
# --------------------------------------------------------------------------------------------------------------------
def test_default_frameslotcq_is_byte_identical():
    """The n_slot_pools threading defaults to N_SLOT_POOLS=6; the default FrameSlotCQ prim init is BIT-IDENTICAL to the
    pre-edit `standard_normal(6)`, the pool count is 6, and the instance primacy IS the module PRIMACY_pA tuple."""
    assert _default_byte_identity(_SEED) is True
    cq = FrameSlotCQ(seed=_SEED)
    assert cq.n_slot_pools == N_SLOT_POOLS == 6
    assert cq.primacy_pA is PRIMACY_pA
    for i, fr in enumerate(FRAME_NAMES):
        ref = np.random.default_rng(_SEED * 13 + 5 + i).standard_normal(N_SLOT_POOLS) * 0.01
        assert np.array_equal(cq.prim[fr], ref), f"{fr} prim not bit-identical"


def test_configurable_pool_count_widens_bridge_and_primacy():
    """A per-instance n_slot_pools=8 widens the slot bridge (8*N_PER neurons) + re-spaces the primacy over 8 ranks
    (still spanning 1800..300 pA). The DEFAULT build (no arg) stays at 6 pools (6*N_PER)."""
    _b6, idx6 = build_slot_bridge(_SEED)                          # default -> 6 pools
    assert len(idx6) == N_SLOT_POOLS * N_PER
    _b8, idx8 = build_slot_bridge(_SEED, n_slot_pools=8)
    assert len(idx8) == 8 * N_PER
    cq8 = FrameSlotCQ(seed=_SEED, n_slot_pools=8)
    assert cq8.n_slot_pools == 8 and len(cq8.primacy_pA) == 8
    assert cq8.primacy_pA[0] == pytest.approx(1800.0) and cq8.primacy_pA[-1] == pytest.approx(300.0)
    assert cq8.prim["F_MODAL"].shape == (8,)


# --------------------------------------------------------------------------------------------------------------------
# (2) THE DITRANSITIVE RENDERS at 8 pools -- the boundary surpassed.
# --------------------------------------------------------------------------------------------------------------------
def test_ditransitive_now_fits_eight_pools():
    """The 7-slot ditransitive FITS the 8-pool substrate (EMERGE-74's 6-pool wall surpassed)."""
    assert len(CONSTRUCTIONS["C_DITRANS"]) == 7
    assert 7 > N_SLOT_POOLS                    # the EMERGE-74 wall was real (7 > 6)
    assert DITRANS_POOLS == 8 and _FITS_8["C_DITRANS"] is True
    assert all(_FITS_8[n] for n in CONSTRUCTION_NAMES)   # ALL named constructions fit 8 pools


def test_ditransitive_renders_exact_svo_surface_on_spikes():
    """The ditransitive renders 'the [subj] [verb]s the [iobj] a [theme]' on spikes at 8 pools (DET SUBJ VERB:3sg DET
    IOBJ DET OBJ) -- the boundary surpassed with ZERO further mining."""
    _tokens, reg = _build()
    cq = reg.render_cq(calibrate=True)
    fact = {"svo_subject": "wolf", "ditrans_verb": "give", "iobj": "cub", "theme": "bone"}
    words = _emit_construction(cq, "C_DITRANS", fact)
    assert words == ["the", "wolf", "gives", "the", "cub", "a", "bone"], words


def test_all_named_constructions_render_exact_on_eight_pools():
    """(a) ALL 7 named constructions render EXACT on the 8-pool substrate (the 5 EMERGE-72 + transitive + ditransitive).
    Moat 0 on abstains; an answer is produced."""
    _tokens, reg = _build()
    facts = build_heldout_facts_svo(_SEED, n=6)
    per, moat_calls, answer_produced = _render_registry(reg, facts, calibrate=True)
    fits = reg.registered_fits()
    n_exact = sum(1 for n in fits if per[n]["exact"] == pytest.approx(1.0))
    assert n_exact == len(CONSTRUCTION_NAMES), f"only {n_exact} rendered exact: {[(n, per[n]['exact']) for n in fits]}"
    assert per["C_DITRANS"]["exact"] == pytest.approx(1.0)
    assert moat_calls == 0 and answer_produced is True


# --------------------------------------------------------------------------------------------------------------------
# (3) THE 2-STAGE READ is load-bearing: the RAW (uncalibrated) read fails to order the ditransitive on the saturating
# seeds; the bias-calibrated read fixes it.
# --------------------------------------------------------------------------------------------------------------------
def test_two_stage_bias_calibration_is_load_bearing():
    """The RAW (uncalibrated) 8-rank read fails to order the ditransitive on a saturating seed (42); the 2-stage per-
    pool bias calibration recovers the correct order. This is the causal control -- the calibration is not decorative."""
    fact = {"svo_subject": "wolf", "ditrans_verb": "give", "iobj": "cub", "theme": "bone"}
    correct = ["the", "wolf", "gives", "the", "cub", "a", "bone"]
    _tokens, reg = _build()
    cq_cal = reg.render_cq(calibrate=True)
    assert _emit_construction(cq_cal, "C_DITRANS", fact) == correct
    cq_raw = reg.render_cq(calibrate=False)
    raw = _emit_construction(cq_raw, "C_DITRANS", fact)
    # seed 42 raw read flips the top adjacent ranks (subj/verb) -> not the correct surface (the read-out limit the
    # 2-stage read closes). We assert the raw read is WRONG here (the causal demonstration).
    assert raw != correct, f"raw (uncalibrated) read unexpectedly correct on seed 42: {raw}"


def test_pool_bias_calibration_equalizes_heterogeneity():
    """The per-pool bias vector (each pool's rate at a common reference current, mean-centred) has non-trivial spread
    (the fixed f-I heterogeneity the 2-stage read removes)."""
    cq = DitransRegistryProducer(seed=_SEED, registry_slots={}, n_slot_pools=8, calibrate=True)
    bias = cq._pool_bias_vector()
    assert bias.shape == (8,)
    assert float(bias.std()) > 0.005      # the heterogeneity is real (per-pool std ~0.02)


# --------------------------------------------------------------------------------------------------------------------
# (4) POSITION-INDEPENDENCE of the 7-slot ditransitive (the hardest frame for the EMERGE-61 adaptation tail).
# --------------------------------------------------------------------------------------------------------------------
def test_ditransitive_is_position_independent():
    """(c) the 7-slot ditransitive renders IDENTICALLY at emit-position 1/3/5 -- the EMERGE-61 wash-out holds at 8 pools
    for the HARDEST (longest) frame."""
    _tokens, reg = _build()
    fact = build_heldout_facts_svo(_SEED, n=1)[0]
    posindep, surfaces = _position_independence(reg, fact)
    assert posindep is True, f"ditransitive not position-independent: {surfaces}"


# --------------------------------------------------------------------------------------------------------------------
# (5) THE INPUT-DESTRUCTION CONTROLS collapse.
# --------------------------------------------------------------------------------------------------------------------
def test_permuted_corpus_collapses_the_registry():
    """(b1) PERMUTED-CORPUS collapses the registry -> 0 registered (the render is genuinely corpus-order-derived)."""
    tokens = build_stream_svo(_SEED, n_extra=_N_EXTRA)
    srng = np.random.default_rng(1234)
    reg_p = DitransRegistry(_SEED).build(tokens, shuffle_within=True, shuffle_rng=srng)
    assert reg_p.n_registered() == 0, f"permuted-corpus still registered {sorted(reg_p.registered)}"


def test_cross_construction_is_wrong():
    """(b2) CROSS-CONSTRUCTION: rendering construction A's fact through a DIFFERENT construction B's mined structure is
    WRONG (form-specific; the ditransitive through the transitive != the ditransitive)."""
    _tokens, reg = _build()
    facts = build_heldout_facts_svo(_SEED, n=4)
    cross = _cross_construction(reg, facts)
    assert cross < 0.30, f"cross-construction render {cross} not collapsed"


def test_no_corpus_yields_empty_registry():
    """(b3) no corpus -> no signatures -> no registry (empty)."""
    reg_empty = DitransRegistry(_SEED).build([])
    assert reg_empty.n_registered() == 0


# --------------------------------------------------------------------------------------------------------------------
# (6) THE GATE-FIRST MOAT holds at 8 pools.
# --------------------------------------------------------------------------------------------------------------------
def test_gate_first_moat_never_invokes_producer_on_abstain():
    """(d) the gate-first no-confab moat: an ABSTAIN never invokes the producer (0 productions); an ANSWER does. The
    positive-control ANSWER runs through the 8-pool calibrated emit (which the base emit could not do)."""
    _tokens, reg = _build()
    cq = reg.render_cq(calibrate=True)
    prod = RegistryBrocaProducer(cq)
    for _ in range(5):
        r = prod.speak(decision("ABSTAIN"))
        assert r["produced"] is False
    assert prod.production_count == 0
    r = prod.speak(decision("ANSWER", "F_MODAL", subject="owl", verb="fly"))
    assert r["produced"] is True
    assert prod.production_count == 1


def test_registry_mines_all_seven_named_constructions():
    """The 8-pool registry mines all 7 named constructions from the corpus (5 EMERGE-72 + transitive + ditransitive)."""
    _tokens, reg = _build()
    assert reg.n_registered() == len(CONSTRUCTION_NAMES) == 7, f"registered {reg.n_registered()}"
    for name in CONSTRUCTION_NAMES:
        assert name in reg.registered_fits(), f"{name} not registered/fitting at 8 pools"
