"""THE DECISIVE VALIDATED-SCALE TEST of the SELECTIVE coupling: does adding the per-neuron SELECTIVE channel to the batched
reservoir generator's read-out make the margin over the memoryless BAG-of-prefix GROW with data — where the FIXED
reservoir's margin SHRINKS (the reservoir-scale run CLOSED the fixed reservoir as Ueda-bounded / bag-matches-reservoir at
scale)? Per the a-1 null-discriminator finding (`2026-07-13-SSM-language-escalation-toy-scale-NULL-DISCRIMINATOR-...`), a
tractable-scale deep-tail-vs-bigram test is a null discriminator; the DECISIVE regime is the validated-scale margin-over-BAG
trend the batched infra measures. This reuses that infra (block-diagonal batched reservoir cache + train_readout/eval_ce +
the bag control + fixed vocab/eval) and AUGMENTS each cached per-token reservoir state with a per-neuron selective-SSM
channel c_t (computed cheaply from the token embedding), then compares margin_over_BAG for res-only vs res+sel.

The selective channel here is FIXED input-dependent (gate lam_t=sigmoid(w.E[tok]+forget_bias), w random) -- the cheap-first
'detached'-style first look (the Rung-3 detached control still beat the fixed reservoir). If a FIXED selective hold already
lifts margin_over_BAG and that lift GROWS with data, the LEARNED gate (Rung-2/3) only improves it. A NEGATIVE (fixed
selective doesn't lift the scale trend) would say the LEARNED gate is required at scale (the next rung), not that the
mechanism fails.

GATE (compare the TREND across scales, the reservoir-scale discipline): margin_over_BAG for res+sel > for res-only AND that
gap GROWS with n_train. Run the SAME scales the reservoir-scale run used (np=300, n_train in {1400,2800,5600,11200}).
Reuse-by-import; NO `sim/` edit. GPU-capable (low-med VRAM): SIM_BACKEND=cupy for the batched reservoir collection.

Run: E:/.../python.exe -m research.runners._reslm_batched_scale_selssm_derisk --n-pool 300 --n-train 2800 --seed 42
"""
import argparse, time, json, math
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import (
    Vocab, train_readout, eval_ce, fit_bigram, bigram_ce, _standardize_fit, _bag_cache, ACTIVE_MIN,
)
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
import research.runners._reslm_batched_reservoir_derisk as BR
from research.runners._reslm_batched_scale_run import _cache_batched

N_SSM = 200
D_SEL = 32
FORGET_BIAS = 2.5


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _sel_params(seed, V):
    rng = np.random.default_rng(seed * 47 + 3)
    E = rng.standard_normal((V, D_SEL)) * 0.8
    Win = rng.standard_normal((N_SSM, D_SEL)) / np.sqrt(D_SEL)
    w = rng.standard_normal((N_SSM, D_SEL)) / np.sqrt(D_SEL)
    b = np.full(N_SSM, FORGET_BIAS)
    return E, Win, w, b


def _augment_selective(cache, E, Win, w, b):
    """Concatenate a per-token FIXED selective-SSM channel c_t to each cached reservoir state. c_t = lam_t c_{t-1} +
    (1-lam_t) inj, lam_t = sigmoid(w.E[tok]+b), inj = Win.E[tok]; reset per sentence."""
    out = []
    for states, ids in cache:
        T = len(states)
        c = np.zeros(N_SSM); cs = np.zeros((T, N_SSM))
        for t in range(T):
            tok = ids[t] if t < len(ids) else ids[-1]
            u = E[tok]; inj = Win @ u; lam = _sig(w @ u + b)
            c = lam * c + (1.0 - lam) * inj; cs[t] = c
        out.append((np.hstack([np.asarray(states), cs]), ids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-sentences", type=int, default=20000)
    ap.add_argument("--n-train", type=int, default=2800)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--batch-m", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_reslm_scale_selssm.json")
    a = ap.parse_args()

    t0 = time.time()
    sents = load_sentences(a.corpus, a.n_sentences)
    perm = np.random.default_rng(a.seed).permutation(len(sents))
    ev = [sents[i] for i in perm[-a.n_eval:]]; pool = [sents[i] for i in perm[:-a.n_eval]]
    vocab = Vocab.build(pool, V=a.vocab); V = vocab.size
    in_dim = len(vocab.encode_seq(pool[0])[0])
    tr = pool[:a.n_train]
    tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]

    b, copy_res, W_in, snap = BR.build_batched(a.seed, a.n_pool, in_dim, a.batch_m)
    tc = time.time()
    tr_cache = _cache_batched(b, copy_res, W_in, snap, vocab, tr, a.batch_m)
    ev_cache = _cache_batched(b, copy_res, W_in, snap, vocab, ev, a.batch_m)
    collect_s = time.time() - tc

    E, Win, w, bg = _sel_params(a.seed, V)
    tr_aug = _augment_selective(tr_cache, E, Win, w, bg)
    ev_aug = _augment_selective(ev_cache, E, Win, w, bg)

    def _fit_eval(tr_c, ev_c, salt):
        m, s = _standardize_fit(tr_c)
        W = train_readout(tr_c, V, a.epochs, a.lr, np.random.default_rng(a.seed * 13 + salt), m, s)
        ce, acc, _ = eval_ce(W, m, s, ev_c, V)
        return ce

    res_ce = _fit_eval(tr_cache, ev_cache, 1)
    sel_ce = _fit_eval(tr_aug, ev_aug, 2)
    bag_ce = _fit_eval(_bag_cache(tr_cache, V), _bag_cache(ev_cache, V), 7)
    P_bi = fit_bigram(tr_ids, V); bi_ce, _, _ = bigram_ce(P_bi, ev_ids)

    m_res = bag_ce - res_ce; m_sel = bag_ce - sel_ce           # margin over the memoryless bag (the headline control)
    res = dict(n_pool=a.n_pool, n_train=len(tr), V=V, res_ce=round(res_ce, 4), sel_ce=round(sel_ce, 4),
               bag_ce=round(bag_ce, 4), bi_ce=round(bi_ce, 4),
               margin_res_over_bag=round(m_res, 4), margin_sel_over_bag=round(m_sel, 4),
               sel_lift=round(m_sel - m_res, 4), collect_s=round(collect_s, 1), total_s=round(time.time() - t0, 1))
    json.dump(res, open(a.out, "w"))
    print(f"[scale-selssm] np={a.n_pool} nt={len(tr)} V={V}: margin_over_BAG res={m_res:+.4f} sel={m_sel:+.4f} "
          f"(sel_lift={m_sel - m_res:+.4f}) | res_ce={res_ce:.3f} sel_ce={sel_ce:.3f} bag={bag_ce:.3f} bigram={bi_ce:.3f} "
          f"-> {'SEL LIFTS margin' if m_sel > m_res + 0.005 else 'sel~=res (no lift at this scale)'}", flush=True)


if __name__ == "__main__":
    main()
