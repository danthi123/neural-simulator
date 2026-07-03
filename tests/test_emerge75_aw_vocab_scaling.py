"""CPU-testable guards for EMERGE-75 (A->W vocab scaling via a 3-bridge dispatch). The on-spikes A->W read-out is
GPU-only; these tests pin the CPU-side structure that does NOT need a GPU:
  * the overflow vocab is exactly 16 words (one 16-pool bridge) = 3 new function words + 13 object nouns;
  * the overflow words are DISJOINT from BRIDGE-A (content) and BRIDGE-F (function) -- so the dispatch is well-defined;
  * the overflow pool assignment binds all 16 words onto the 16 concept pools 1:1;
  * the de-risk facts draw their OBJECT from BRIDGE-C's 13 (so every rendered object spike-spells) and their pp_verb
    from a BRIDGE-A ability lemma (3sg-inflectable on spikes);
  * the EMERGE-72 registry (numpy) still mines the in-scope constructions (incl. the PP constructions with the OBJ slot);
  * the gate-first moat contract: RegistryBrocaProducer never produces on an abstain (token-spell path, CPU);
  * the inflection-aware verb-decode helper strips a 3sg ability-verb surface to its lemma.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._emerge75_aw_vocab_scaling_derisk as m75  # noqa: E402
import research.runners._emerge67_neural_spell_wirein_derisk as m67  # noqa: E402
import research.runners._emerge68_function_word_spell_derisk as m68  # noqa: E402
from research.runners._emerge72_construction_registry_derisk import (  # noqa: E402
    ConstructionRegistry, RegistryBrocaProducer, decision, build_stream, OBJ, DET, SUBJ, FUNC, VERB,
)


def test_overflow_vocab_is_16_words_one_bridge():
    assert len(m75._OVF_VOCAB16) == 16, m75._OVF_VOCAB16
    assert len(m75._OVF_FUNC) == 3 and len(m75._OVF_OBJ) == 13
    assert m75._OVF_VOCAB16 == m75._OVF_FUNC + m75._OVF_OBJ
    # exactly the NEW function words the EMERGE-72/73 constructions add (PP prepositions + copula)
    assert set(m75._OVF_FUNC) == {"to", "on", "is"}


def test_overflow_disjoint_from_bridge_a_and_f():
    """The dispatch is well-defined ONLY if the overflow words are in neither BRIDGE-A nor BRIDGE-F."""
    content = set(m67._AW_CONTENT)
    func = set(m68._FUNC_WORDS)
    ovf = set(m75._OVF_VOCAB16)
    assert ovf.isdisjoint(content), ovf & content
    assert ovf.isdisjoint(func), ovf & func
    # BRIDGE-A content and BRIDGE-F function are themselves disjoint (established by EMERGE-67/68)
    assert content.isdisjoint(func)


def test_overflow_pool_assignment_bijective():
    w2p, p2w, func_pools, obj_pools = m75._ovf_pool_assignment()
    assert len(w2p) == 16 and len(p2w) == 16
    assert set(w2p) == set(m75._OVF_VOCAB16)
    assert len(set(w2p.values())) == 16   # 16 distinct pools
    assert set(func_pools) == set(m75._OVF_FUNC)
    assert set(obj_pools) == set(m75._OVF_OBJ)


def test_swap_ovf_vocab_word_to_idx():
    idx = m75._swap_ovf_vocab()
    assert len(idx) == 16
    assert set(idx) == set(m75._OVF_VOCAB16)
    assert sorted(idx.values()) == list(range(16))


def test_facts_draw_object_from_bridge_c_and_ppverb_from_bridge_a():
    """Every de-risk fact's OBJECT must be a BRIDGE-C overflow noun (so it spike-spells) and its pp_verb a BRIDGE-A
    ability lemma (so the 3sg PP render decodes the lemma on spikes)."""
    for seed in (42, 43, 44, 100, 101, 102):
        facts = m75._facts(seed, n=8)
        assert len(facts) == 8
        for f in facts:
            assert f["obj"] in set(m75._OVF_OBJ), (seed, f["obj"])
            assert f["pp_verb"] in set(m67._AW_ABILITY), (seed, f["pp_verb"])
            assert f["subject"] in set(m67._AW_SUBJECTS)


def test_registry_mines_in_scope_constructions_numpy():
    """The EMERGE-72 registry (numpy) mines the in-scope constructions, including the PP constructions with an OBJ slot."""
    reg = ConstructionRegistry(42).build(build_stream(42))
    for name in ("F_MODAL", "F_INTR", "F_NEGMOD", "C_PPGOAL", "C_PPLOC"):
        assert name in reg.registered, (name, list(reg.registered))
    # the PP constructions carry the OBJ slot
    ppgoal_types = [t for (t, p) in reg.registered["C_PPGOAL"]]
    assert OBJ in ppgoal_types


def test_scope_is_emerge72_five_not_adjective():
    assert m75._SCOPE_CONSTRUCTIONS == ["F_MODAL", "F_INTR", "F_NEGMOD", "C_PPGOAL", "C_PPLOC"]
    # the EMERGE-73 adjective constructions are explicitly OUT of scope
    assert "C_ATTRIB" not in m75._SCOPE_CONSTRUCTIONS
    assert "C_PRED" not in m75._SCOPE_CONSTRUCTIONS


def test_moat_contract_token_spell_cpu():
    """The gate-first moat: RegistryBrocaProducer never produces on an abstain (token-spell path, no GPU needed)."""
    reg = ConstructionRegistry(42).build(build_stream(42))
    cq = reg.render_cq()
    prod = RegistryBrocaProducer(cq)   # default token spell
    for _ in range(5):
        r = prod.speak(decision("ABSTAIN"))
        assert r["produced"] is False and r["surface"] is None
    assert prod.production_count == 0
    # an ANSWER DOES produce (the counter is meaningful)
    ans = prod.speak(decision("ANSWER", construction="F_MODAL", subject="owl", verb="fly", obj="pond"))
    assert ans["produced"] is True and prod.production_count == 1


class _FakeContentDecoder:
    """A CPU stand-in for the content bridge decode: returns the lemma verbatim (a genuine spiking decode returns the
    trained word). Lets us test the inflection re-application logic without a GPU."""
    def __init__(self):
        self.word_to_pool = {w: f"pool_{w}" for w in m67._AW_CONTENT}

    def _decode(self, word):
        return (word, 1.0, 1.0, 100)


def test_inflection_aware_verb_helper():
    """The unified spell strips a 3sg ability-verb surface to its lemma; the non-3sg / intransitive words pass through."""
    # build a minimal object exercising ONLY the pure helper (no GPU): monkeypatch the engines with fakes.
    u = m75.UnifiedNeuralSpell75.__new__(m75.UnifiedNeuralSpell75)
    u.spell_calls = 0
    u.content = _FakeContentDecoder()
    u.func = None
    u.overflow = None
    u.content_words = set(m67._AW_CONTENT)
    u.func_words = set()
    u.overflow_words = set()
    # 'flies' -> lemma 'fly', apply_3sg True
    lemma, apply3 = u._lemma_and_inflection("flies")
    assert lemma == "fly" and apply3 is True
    # 'walks' is an intransitive already-3sg pool word (NOT an ability lemma inflection) -> pass through
    lemma2, apply3b = u._lemma_and_inflection("walks")
    assert apply3b is False
    # 'owl' -> not a verb -> pass through
    lemma3, apply3c = u._lemma_and_inflection("owl")
    assert lemma3 == "owl" and apply3c is False
    # spell('flies') re-inflects the fake-decoded lemma -> 'flies' (genuinely-spiking path re-applies morphology)
    assert u.spell("flies") == "flies"
    # spell('owl') decodes content verbatim
    assert u.spell("owl") == "owl"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
