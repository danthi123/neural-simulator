import pytest
from sim.grounded_decode import grounded_decode


class _SpyLM:
    """Raises if generation is attempted -- proves the abstain path
    NEVER invokes the LM (no-confab BY CONSTRUCTION)."""
    def __init__(self):
        self.called = False
    def __call__(self, *a, **k):
        self.called = True
        raise AssertionError("LM invoked on the abstain path!")


class _EchoLM:
    """Deterministic stand-in: 'generates' by echoing the retrieved
    prompt ids back (faithful by construction) so the policy wiring
    is testable without torch."""
    def generate_ids(self, prompt_ids, max_new):
        return list(prompt_ids[:max_new])


class _Tok:
    def encode(self, s):
        return [ord(c) % 50 for c in s]
    def decode(self, ids):
        return "".join(chr(65 + (i % 26)) for i in ids)


def test_abstain_path_never_touches_the_lm():
    spy = _SpyLM()
    r = grounded_decode([("dog", 100.0, "t1")], spy, _Tok(),
                        retrieved_text="dog is big",
                        query="what is dog", threshold=650.0)
    assert r["abstained"] is True and r["text"] is None
    assert spy.called is False


def test_empty_ranked_abstains_no_lm():
    spy = _SpyLM()
    r = grounded_decode([], spy, _Tok(), retrieved_text="",
                        query="q", threshold=650.0)
    assert r["abstained"] is True and spy.called is False


def test_grounded_path_decodes_conditioned_on_retrieved():
    r = grounded_decode([("dog", 900.0, "t1")], _EchoLM(), _Tok(),
                        retrieved_text="dog is big",
                        query="what is dog", threshold=650.0,
                        max_new=8)
    assert r["abstained"] is False
    assert isinstance(r["text"], str) and len(r["text"]) > 0
    assert r["retrieved"] == "dog is big"


def test_threshold_boundary_is_abstain():
    spy = _SpyLM()
    r = grounded_decode([("x", 650.0, "t")], spy, _Tok(),
                        retrieved_text="x y", query="q",
                        threshold=650.0)
    assert r["abstained"] is True and spy.called is False
