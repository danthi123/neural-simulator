"""Pure CPU tests for the constrained-realization policy. LOAD-BEARING:
(A) abstain path provably NEVER touches the lm (spy lm raises on ANY
attribute access) -- the no-confab-by-construction guarantee;
(B) faithfulness BY CONSTRUCTION -- the masked argmax is ALWAYS in the
allowed id set, even when a NON-allowed id has the global argmax.
torch-free, deterministic toy lm/tok."""
import random
import pytest
from sim.constrained_realize import constrained_realize


class _SpyLM:
    """Raises on ANY attribute access -> proves abstain never uses it."""
    def __getattribute__(self, name):
        raise AssertionError(
            "LM was touched on the abstain path (no-confab BY "
            "CONSTRUCTION violated): attr=%r" % name)


class _ToyTok:
    """Deterministic toy tokenizer: id = ord(first char) of each word;
    decode joins symbols. Vocab is implicit (ids are small ints)."""
    def encode(self, text):
        return [ord(w[0]) for w in str(text).split() if w]

    def decode(self, ids):
        return " ".join(chr(i) for i in ids)


class _ToyLM:
    """Returns FIXED logits: a NON-allowed id (999) always has the
    global max, then allowed ids in a deterministic order. Proves the
    mask works even when the unconstrained argmax is non-allowed."""
    def __init__(self, vocab=1024):
        self.vocab = vocab

    def logits(self, seq_ids):
        v = [0.0] * self.vocab
        v[999] = 100.0          # global argmax is NON-allowed
        for i in range(0, 200):
            v[i] = (i % 7) * 0.1
        return v


def test_abstain_path_never_touches_lm():
    r = constrained_realize(
        ranked=[], lm=_SpyLM(), tok=_ToyTok(),
        retrieved_text="", query="zarn",
        function_words=["is", "a"], threshold=650.0,
        no_repeat_ngram=3, max_new=10)
    assert r["abstained"] is True
    assert r["text"] is None
    assert r["retrieved"] == ""


def test_below_threshold_abstains_without_touching_lm():
    r = constrained_realize(
        ranked=[("zarn", 400.0, "none")], lm=_SpyLM(), tok=_ToyTok(),
        retrieved_text="", query="zarn",
        function_words=["is", "a"], threshold=650.0)
    assert r["abstained"] is True and r["text"] is None


def test_faithfulness_by_construction_argmax_always_in_allowed():
    tok = _ToyTok()
    lm = _ToyLM()
    retrieved = "max big dog"
    fw = ["is", "a"]
    allowed = set(tok.encode(retrieved)) | set(tok.encode("is")) \
        | set(tok.encode("a"))
    r = constrained_realize(
        ranked=[("max", 900.0, "kb")], lm=lm, tok=tok,
        retrieved_text=retrieved, query="max",
        function_words=fw, threshold=650.0,
        no_repeat_ngram=3, max_new=30)
    assert r["abstained"] is False
    out_ids = [ord(w[0]) for w in r["text"].split() if w]
    assert 999 not in out_ids
    assert all(i in allowed for i in out_ids), (out_ids, allowed)


def test_faithfulness_random_logits_fuzz():
    tok = _ToyTok()
    retrieved = "lily small red ball"
    fw = ["has", "a"]
    allowed = set(tok.encode(retrieved)) | set(tok.encode("has")) \
        | set(tok.encode("a"))

    class _RandLM:
        def logits(self, seq_ids):
            random.seed(len(seq_ids))
            return [random.uniform(-10, 10) for _ in range(1024)]

    r = constrained_realize(
        ranked=[("lily", 900.0, "kb")], lm=_RandLM(), tok=tok,
        retrieved_text=retrieved, query="lily",
        function_words=fw, threshold=650.0, max_new=40)
    out_ids = [ord(w[0]) for w in r["text"].split() if w]
    assert all(i in allowed for i in out_ids)


def test_no_repeat_ngram_blocks_immediate_loop():
    tok = _ToyTok()

    class _LoopLM:
        def logits(self, seq_ids):
            v = [0.0] * 1024
            v[ord("a")] = 5.0
            v[ord("b")] = 4.0
            return v

    r = constrained_realize(
        ranked=[("x", 900.0, "kb")], lm=_LoopLM(), tok=tok,
        retrieved_text="a b", query="x",
        function_words=[], threshold=650.0,
        no_repeat_ngram=2, max_new=20)
    ids = [ord(w[0]) for w in r["text"].split() if w]
    grams = list(zip(ids, ids[1:]))
    assert len(grams) == len(set(grams)) or len(ids) <= 2, ids


def test_coverage_stop_halts_once_all_content_ids_emitted():
    tok = _ToyTok()

    class _CovLM:
        def logits(self, seq_ids):
            v = [0.0] * 1024
            v[ord("m")] = 9.0
            v[ord("d")] = 8.0
            return v

    r = constrained_realize(
        ranked=[("max", 900.0, "kb")], lm=_CovLM(), tok=tok,
        retrieved_text="max dog", query="max",
        function_words=[], threshold=650.0,
        no_repeat_ngram=3, max_new=100)
    ids = [ord(w[0]) for w in r["text"].split() if w]
    assert ord("m") in ids and ord("d") in ids
    assert len(ids) < 100
