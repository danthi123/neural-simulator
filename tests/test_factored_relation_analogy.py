"""CI GUARD (Tier 2.1-A): FACTORED-RELATION analogy A:B::C:? must keep resolving over EXPLICIT factored relations,
with the full anti-cheat bar holding (the burned-capability discipline -- analogy/reasoning-over-composition is
exactly where this project has over-claimed; 2026-05-14 retraction). This pins the de-risked GO
(research/findings/2026-06-27-tier2.1A-factored-relation-analogy-GO.md) so it does not silently bit-rot.

The anti-cheats (held-out >> floor; permuted-relation collapses + TRUE rank-1; lesion collapses; scrambled-source
chance; no-confab moat) are asserted as part of the guard -- a regression that broke the transform's necessity or
the moat would flip these.

CPU-safe (numpy phase arithmetic == the spiking result). A GPU-gated test also confirms the analogy op runs
through the REAL RF spiking bind/conj (== numpy) when the CuPy substrate is present.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.factored_relation_analogy import (  # noqa: E402
    FactoredRelationAnalogy, build_knowledge_base)
from sim.backend import is_gpu_backend  # noqa: E402


# A clean BIJECTIVE relation grid (gender x semantic) for the anti-cheat assertions, mirroring the de-risk.
_PAIRS = [("king", "queen", "royal_hi"), ("prince", "princess", "royal_lo"), ("man", "woman", "person"),
          ("actor", "actress", "perform"), ("uncle", "aunt", "kin"), ("boy", "girl", "young")]


def _build(seed, use_spiking_bind=False, abstain_sim=0.5):
    kb = FactoredRelationAnalogy(seed=seed, D=256, use_spiking_bind=use_spiking_bind, abstain_sim=abstain_sim)
    for m, f, sem in _PAIRS:
        kb.register_item(m, GENDER="male", SEM=sem)
        kb.register_item(f, GENDER="female", SEM=sem)
    return kb


def _held_out_tests():
    """(a, b, c, d): (a,b) builds the transform; (c,d) is the HELD-OUT scored pair (d never builds T)."""
    pairs = [(m, f) for m, f, _ in _PAIRS]
    targets = [f for _m, f in pairs]
    tests = []
    for (c, d) in pairs:
        a, b = next((s, t) for (s, t) in pairs if s != c and t != d)
        tests.append((a, b, c, d, targets))
    return tests


def test_held_out_analogy_resolves_above_floor():
    kb = _build(42)
    tests = _held_out_tests()
    correct = floor = 0
    for (a, b, c, d, targets) in tests:
        # held-out analogy
        correct += (kb.analogy(a, b, c, candidates=targets) == d)
        # memorization floor: nearest TARGET to C's own code, no transform.
        sims = [float(np.mean(np.cos(2.0 * np.pi * (kb.item_code[c] - kb.item_code[w])))) for w in targets
                if w != c]
        flw = [w for w in targets if w != c]
        floor += (flw[int(np.argmax(sims))] == d)
    acc = correct / len(tests)
    floor_acc = floor / len(tests)
    assert acc >= 0.9, f"held-out analogy acc {acc:.3f} should be ~1.0"
    assert acc > floor_acc + 0.3, f"held-out {acc:.3f} must be >> floor {floor_acc:.3f}"


def test_permuted_relation_collapses_and_true_is_best():
    """Permuting the example pair (a random fake b) must collapse to ~chance AND the TRUE example must rank 1/k."""
    kb = _build(42)
    tests = _held_out_tests()
    true_acc = np.mean([kb.analogy(a, b, c, candidates=targets) == d for (a, b, c, d, targets) in tests])
    rng = np.random.default_rng(999)
    perm_accs = []
    for _ in range(5):
        pc = 0
        for (a, b, c, d, targets) in tests:
            zt = kb._to_phasor(rng.uniform(0.0, 1.0, kb.D)) * np.conj(kb._phasor_of(a))  # fake transform
            rec = (np.angle(zt * kb._phasor_of(c)) / (2.0 * np.pi)) % 1.0
            cand = [w for w in targets if w not in (a, b, c)]
            pc += (cand[int(np.argmax([np.mean(np.cos(2 * np.pi * (rec - kb.item_code[w]))) for w in cand]))] == d)
        perm_accs.append(pc / len(tests))
    assert true_acc == 1.0
    assert np.mean(perm_accs) < 0.35, f"permuted analogy {np.mean(perm_accs):.3f} should collapse to chance"
    assert all(pa < true_acc for pa in perm_accs), "TRUE example must be UNIQUELY best (rank 1)"


def test_lesion_skip_transform_collapses():
    """Lesion = apply B directly (skip the transform extraction) -> must collapse below the held-out accuracy."""
    kb = _build(42)
    tests = _held_out_tests()
    lc = 0
    for (a, b, c, d, targets) in tests:
        rec = kb.item_code[b]                       # T := B (no unbind)
        cand = [w for w in targets if w not in (a, b, c)]
        lc += (cand[int(np.argmax([np.mean(np.cos(2 * np.pi * (rec - kb.item_code[w]))) for w in cand]))] == d)
    assert lc / len(tests) < 0.5, "lesion (no transform) must collapse"


def test_scrambled_source_is_chance():
    """Scrambled items (unique random codes, no shared factored structure) -> analogy at chance."""
    kb = FactoredRelationAnalogy(seed=42, D=256)
    rng = np.random.default_rng(5555)
    pairs = [(m, f) for m, f, _ in _PAIRS]
    for m, f, _sem in _PAIRS:
        kb.item_attrs[m] = {"x": m}; kb.item_code[m] = rng.uniform(0.0, 1.0, kb.D)
        kb.item_attrs[f] = {"x": f}; kb.item_code[f] = rng.uniform(0.0, 1.0, kb.D)
    targets = [f for _m, f in pairs]
    sc = 0
    for (c, d) in pairs:
        a, b = next((s, t) for (s, t) in pairs if s != c and t != d)
        # scrambled codes carry no shared relation -> the transform is junk -> the answer is None (low-confidence
        # abstain) or a wrong item; either way NOT d for most queries.
        sc += (kb.analogy(a, b, c, candidates=targets) == d)
    assert sc <= 2, f"scrambled-source analogy resolved {sc}/{len(pairs)} -- should be ~chance/abstain"


def test_no_confab_moat_abstains():
    """The moat: an UNREGISTERED operand abstains (None); a LOW-confidence (un-grounded) analogy abstains."""
    kb = _build(42)
    assert kb.analogy("king", "queen", "NOT_AN_ITEM") is None           # unregistered C -> abstain
    assert kb.analogy("NOPE", "queen", "prince") is None                # unregistered A -> abstain
    # a high abstain gate makes even a correct-but-grounded analogy abstain ONLY when below threshold; with the
    # default gate a grounded analogy does NOT abstain:
    assert kb.analogy("king", "queen", "prince") == "princess"
    # an un-grounded transform (scrambled C code) sits below the gate -> abstains. Build a C with a random code:
    kb2 = _build(42, abstain_sim=0.5)
    kb2.item_attrs["junk"] = {"x": "junk"}
    kb2.item_code["junk"] = np.random.default_rng(1).uniform(0.0, 1.0, kb2.D)
    # king:queen on 'junk' -> rec is junk rotated by the gender offset, far from any registered target -> abstain
    assert kb2.analogy("king", "queen", "junk", candidates=["princess", "woman", "girl", "aunt"]) is None


def test_console_knowledge_base_resolves_and_abstains():
    """The production KB: bijective analogies resolve; is_a items (absent) + unregistered items abstain."""
    kb = build_knowledge_base(seed=42, D=256)
    assert kb.analogy("king", "queen", "prince") == "princess"
    assert kb.analogy("paris", "france", "rome") == "italy"
    assert kb.analogy("walk", "walked", "jump") == "jumped"
    assert kb.analogy("big", "bigger", "fast") == "faster"
    # is_a / taxonomy is NOT in the KB (the documented many-to-one boundary) -> abstain
    assert kb.analogy("dog", "mammal", "robin") is None


def test_multi_seed_held_out_unanimous():
    """6-seed: held-out analogy is unanimous 1.0 across the bijective families (the standing 6-seed rule)."""
    for seed in (42, 43, 44, 45, 46, 47):
        kb = _build(seed)
        tests = _held_out_tests()
        acc = np.mean([kb.analogy(a, b, c, candidates=targets) == d for (a, b, c, d, targets) in tests])
        assert acc == 1.0, f"seed {seed}: held-out analogy {acc:.3f} should be 1.0"


@pytest.mark.skipif(not is_gpu_backend(),
                    reason="the RF spiking bind needs the CuPy/GPU substrate")
def test_spiking_bind_matches_numpy():
    """The analogy op through the REAL RF spiking bind/conj (== numpy phase arithmetic) on the bijective families."""
    kb_np = _build(42, use_spiking_bind=False)
    kb_sp = _build(42, use_spiking_bind=True)
    tests = _held_out_tests()
    for (a, b, c, d, targets) in tests:
        assert kb_np.analogy(a, b, c, candidates=targets) == d
        assert kb_sp.analogy(a, b, c, candidates=targets) == d        # spiking == numpy == ground truth
    # the spiking moat still abstains on an unregistered operand
    assert kb_sp.analogy("king", "queen", "NOT_AN_ITEM") is None
