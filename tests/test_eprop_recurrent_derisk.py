"""CI guard for the e-prop recurrent-learning de-risk (finding 2026-07-11-eprop-...-REAL-WITH-SCOPE).

Locks the load-bearing QUALITATIVE signature of "making the reservoir's recurrent weights learn via random-feedback
e-prop (no BPTT)":
  1. plastic beats the SAME-SIZE fixed reservoir OVERALL (learning W_rec helps),
  2. the credit STRUCTURE is load-bearing: shuffle_elig (same-magnitude, scrambled-structure updates) ~= fixed,
     i.e. its effect is a small fraction of plastic's (the clean genuine-credit proof, not overfitting/gain-growth),
  3. zero_signal == fixed EXACTLY (no learning signal -> W_rec never moves).
Tiny/fast config; skips if the WikiText corpus is absent (offline CI). CPU/numpy.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")

CORPUS = "data/corpus/wikitext.txt"


def _run():
    from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
    from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
    from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir, train, per_depth_ce
    sents = load_sentences(CORPUS, 2500)
    seed = 42
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sents)); cut = int(0.8 * len(sents))
    tr = [sents[i] for i in idx[:cut]][:400]
    ev = [sents[i] for i in idx[cut:]][:150]
    vocab = Vocab.build(tr, V=120); V = vocab.size
    tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
    P_bi = fit_bigram(tr_ids, V)
    out = {}
    for mode in ("fixed", "plastic", "shuffle_elig", "zero_signal"):
        res = RateReservoir(V, 150, seed, alpha=0.3, spectral=1.1)
        W = train(res, tr_ids, V, 5, 0.02, 0.006, seed, mode=mode)
        _, agg, _ = per_depth_ce(res, W, ev_ids, P_bi)
        out[mode] = agg
    return out


@pytest.mark.skipif(not os.path.exists(CORPUS), reason="WikiText corpus absent (offline CI)")
def test_eprop_credit_signature():
    agg = _run()
    fixed = agg["fixed"]
    plastic_gain = fixed - agg["plastic"]            # >0 => plastic better
    shuffle_gain = fixed - agg["shuffle_elig"]
    # (1) learning W_rec helps overall
    assert plastic_gain > 0.0, f"plastic should beat fixed overall (gain={plastic_gain:.4f})"
    # (2) credit STRUCTURE load-bearing: shuffle's effect is a small fraction of plastic's
    assert abs(shuffle_gain) < 0.5 * plastic_gain, \
        f"shuffle_elig must ~= fixed (shuffle_gain={shuffle_gain:.4f} vs plastic_gain={plastic_gain:.4f})"
    # (3) zero learning signal -> W_rec frozen -> EXACTLY fixed
    assert abs(agg["zero_signal"] - fixed) < 1e-9, \
        f"zero_signal must == fixed exactly ({agg['zero_signal']} vs {fixed})"
