"""CPU-safe tests for EMERGE-57 -- the RA re-fine-tune on EMERGE's grounded frames.

Covers the load-bearing CPU-testable pieces:
  (1) the FRAME-AWARE INFLECTION fix (emerge_v3) -- the 'walkses'/'flys' bug: already-3sg intransitives stay verbatim,
      irregulars inflect correctly (fly->flies), regular RA verbs match the RA table.
  (2) the EMERGE-frame example generator produces well-formed 'facts:...question:...answer:...' examples with the two
      frame families (ability 'the X can V .' + intransitive-exception 'the X <intr_3sg> .') and NO double-inflection.
  (3) the corpus builder writes a mixed EMERGE+RA(+TinyStories) corpus.
  (4) the GPU render+moat smoke -- SKIPPED if the EMERGE ckpt / torch+cuda is absent (the gate-first moat's
      renderer-not-invoked-on-abstain is a hard count when it runs).
"""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import (  # noqa: E402
    emerge_v3, _is_3sg_already, _make_emerge_example, build_emerge_corpus, _rng,
    _ABILITY_VERBS, _INTRANS_3SG, _KNOWN_INTRANS_3SG, EMERGE_FT_CKPT,
)


# ---------------------------------------------------------------------------------------------------------------------
# (1) THE FRAME-AWARE INFLECTION FIX -- the load-bearing 'walkses' bug fix.
# ---------------------------------------------------------------------------------------------------------------------
def test_emerge_v3_already_3sg_intransitive_not_reinflected():
    # the exact EMERGE exception verbs -- must stay verbatim (the bug produced 'walkses'/'lurkses')
    assert emerge_v3("walks") == "walks"
    assert emerge_v3("lurks") == "lurks"
    assert emerge_v3("swims") == "swims"
    assert emerge_v3("sits") == "sits"
    assert emerge_v3("sleeps") == "sleeps"
    # explicit already_3sg flag also holds
    assert emerge_v3("walks", already_3sg=True) == "walks"


def test_emerge_v3_irregular_and_regular():
    assert emerge_v3("fly") == "flies"        # NOT 'flys' (the other inflection bug)
    assert emerge_v3("carry") == "carries"
    assert emerge_v3("catch") == "catches"
    assert emerge_v3("swim") == "swims"
    assert emerge_v3("run") == "runs"
    assert emerge_v3("hop") == "hops"
    # regular RA base verbs match the RA fine-tune's own (base->3sg) surface exactly
    assert emerge_v3("eat") == "eats"
    assert emerge_v3("chase") == "chases"


def test_emerge_v3_idempotent_on_intransitives():
    # applying twice must not change an already-3sg verb (idempotence -- the double-inflection guard)
    for v in _INTRANS_3SG:
        assert emerge_v3(emerge_v3(v)) == emerge_v3(v)
        assert emerge_v3(v) == v, f"{v!r} was re-inflected to {emerge_v3(v)!r}"


def test_is_3sg_detection():
    assert _is_3sg_already("walks")
    assert _is_3sg_already("lurks")
    assert not _is_3sg_already("fly")
    assert not _is_3sg_already("swim")


def test_all_intrans_verbs_are_known_3sg():
    # every intransitive exception verb used by the generator must be recognised as already-3sg (else it double-inflects)
    for v in _INTRANS_3SG:
        assert v in _KNOWN_INTRANS_3SG, f"{v!r} missing from _KNOWN_INTRANS_3SG -> would double-inflect"


# ---------------------------------------------------------------------------------------------------------------------
# (2) THE EMERGE-FRAME EXAMPLE GENERATOR.
# ---------------------------------------------------------------------------------------------------------------------
def test_emerge_examples_wellformed_and_no_double_inflection():
    r = _rng(123)
    saw_ability, saw_exception, saw_abstain = False, False, False
    for _ in range(3000):
        ex = _make_emerge_example(r)
        assert ex.startswith("facts : ")
        assert " question : " in ex and " answer : " in ex
        # NO double-inflection anywhere (the bug produced 'walkses'/'flys')
        toks = ex.split()
        assert "walkses" not in toks and "lurkses" not in toks and "flys" not in toks
        for w in toks:
            if w.endswith("ses"):
                # the only legit -ses words in this vocab would be from re-inflecting a 3sg -> must not happen
                assert w[:-3] + "s" not in _KNOWN_INTRANS_3SG, f"double-inflection {w!r} in {ex!r}"
        ans = ex.split(" answer : ", 1)[1]
        if ans.startswith("yes"):
            saw_ability = True
        elif ans.startswith("no"):
            saw_exception = True
            # an exception answer must contain an already-3sg intransitive verb, verbatim
        elif "know" in ans or "not sure" in ans or "not say" in ans:
            saw_abstain = True
    assert saw_ability and saw_exception and saw_abstain, "generator must emit all three frame families"


def test_ability_frame_uses_bare_infinitive_and_can():
    """The ability/inheritance frame must be 'the X can <bare> .' (a modal + a bare infinitive, never inflected)."""
    r = _rng(7)
    n_can = 0
    for _ in range(2000):
        ex = _make_emerge_example(r)
        facts = ex.split(" question : ", 1)[0]
        if " can " in facts:
            n_can += 1
            # the word after 'can' in the fact must be a bare ability verb (in _ABILITY_VERBS), not a 3sg form
            for i, w in enumerate(facts.split()):
                if w == "can" and i + 1 < len(facts.split()):
                    nxt = facts.split()[i + 1]
                    assert nxt in _ABILITY_VERBS, f"non-bare verb {nxt!r} after 'can' in {facts!r}"
    assert n_can > 100, "should generate many ability frames"


# ---------------------------------------------------------------------------------------------------------------------
# (3) THE CORPUS BUILDER.
# ---------------------------------------------------------------------------------------------------------------------
def test_build_emerge_corpus(tmp_path):
    out = tmp_path / "emerge_corpus.txt"
    # ts path absent -> pure QA corpus (no TinyStories dependency for the unit test)
    stats = build_emerge_corpus(str(out), n_emerge=50, n_ra=30, tinystories_path="/nonexistent",
                                mix_ratio=0.0, seed=42)
    assert stats["n_emerge"] == 50 and stats["n_ra"] == 30
    text = out.read_text(encoding="utf-8")
    assert "facts :" in text and "answer :" in text
    assert "can " in text            # ability frames present
    assert "walkses" not in text and "flys" not in text


# ---------------------------------------------------------------------------------------------------------------------
# (4) GPU render + gate-first MOAT smoke -- skipped without the EMERGE ckpt / torch+cuda.
# ---------------------------------------------------------------------------------------------------------------------
def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not os.path.exists(EMERGE_FT_CKPT) or not _has_torch_cuda(),
                    reason="EMERGE re-fine-tuned ckpt or torch+CUDA absent (GPU smoke)")
def test_render_moat_smoke():
    from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import _render_derisk
    r = _render_derisk(seed=42, verbose=False)
    # THE LOAD-BEARING MOAT PROPERTY: the generator is NEVER invoked on an abstain.
    assert r["moat_render_calls_on_abstains"] == 0
    assert r["n_model_invoked_on_abstain"] == 0
    assert r["n_abstain"] >= 1
    # no double-inflection in any rendered answer
    assert r["n_double_inflection"] == 0
