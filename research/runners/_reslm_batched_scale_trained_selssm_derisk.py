"""THE DECISIVE validated-scale TRAINED-selective test (tractable): a FIXED echo-state reservoir (batched-collected, fast)
+ a TRAINED selective gate + read-out (cheap -- no O(n^2) reservoir e-prop) -> does a TRAINED selective channel lift
margin_over_BAG where the FIXED one HURT (`-scale-selssm-FIXED-gate-negative-...`)? The fixed-gate negative pinned that the
LEARNED gate is required at scale; this trains it, over the batched fixed-reservoir cache, and can reach larger V/data on GPU
(the validated-signal regime the a-1 null-discriminator finding named) because the gate+read-out training is cheap.

Everything transport-free: the read-out by the local delta rule, the gate by the forward-mode eligibility x FIXED RANDOM
FEEDBACK (no BPTT, no transport). Same simple trainer for BOTH arms (fair): res-only (read-out over the fixed reservoir h)
vs res+sel (read-out over [h, trained-c]) vs the memoryless BAG control. HEADLINE = margin_over_BAG for each; the decisive
read is whether res+sel's margin_over_BAG > res-only's AND grows with data (the reservoir-scale discipline).

Reuse-by-import: the batched reservoir cache (`_reslm_batched_scale_run._cache_batched` + `_reslm_batched_reservoir.build_batched`),
`Vocab`/`fit_bigram`/`bigram_ce`/`load_sentences`. NO `sim/` edit. GPU-capable (SIM_BACKEND=cupy for the batched collection).

Run: E:/.../python.exe -m research.runners._reslm_batched_scale_trained_selssm_derisk --n-pool 300 --n-train 2800 --seed 42
"""
import argparse, time, json, math
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram, bigram_ce, _bag_cache
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
import research.runners._reslm_batched_reservoir_derisk as BR
from research.runners._reslm_batched_scale_run import _cache_batched

N_SSM = 200
D_SEL = 32
FORGET_BIAS = 2.5
EPOCHS = 12
LR_RO = 0.01
LR_GATE = 0.2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def _softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()


def _joint_train_eval(tr_cache, ev_cache, V, augment, seed, E, Win, w0, b0, Bc):
    """Train a read-out over the FIXED reservoir states (augment=False) or over [h, TRAINED selective c] (augment=True),
    by the per-token local delta rule; when augment, co-train the gate (w,b) by the forward eligibility x random feedback.
    Same trainer for both arms (fair). Returns held-out next-token CE."""
    n_res = np.asarray(tr_cache[0][0]).shape[1]                  # states may be an ndarray (reservoir) or a list (bag)
    fdim = n_res + (N_SSM if augment else 0)
    Wro = np.zeros((V, fdim))
    w = w0.copy() if augment else None
    b = b0.copy() if augment else None
    for _ep in range(EPOCHS):
        for states, ids in tr_cache:
            T = min(len(states), len(ids))
            c = np.zeros(N_SSM); ew = np.zeros((N_SSM, D_SEL)); ec = np.zeros(N_SSM)
            for t in range(T - 1):
                h = states[t]
                if augment:
                    u = E[ids[t]]; inj = Win @ u; lam = _sig(w @ u + b)
                    c_prev = c; c = lam * c_prev + (1.0 - lam) * inj
                    dl = lam * (1.0 - lam); base = (c_prev - inj)
                    ew = lam[:, None] * ew + (dl * base)[:, None] * u[None, :]; ec = lam * ec + dl * base
                    feat = np.concatenate([h, c])
                else:
                    feat = h
                p = _softmax(Wro @ feat); err = p.copy(); err[ids[t + 1]] -= 1.0
                Wro -= LR_RO * np.outer(err, feat)
                if augment:
                    delta_c = Bc @ err
                    w -= LR_GATE * (delta_c[:, None] * ew); b -= LR_GATE * (delta_c * ec)
    ce = 0.0; cnt = 0
    for states, ids in ev_cache:
        T = min(len(states), len(ids)); c = np.zeros(N_SSM)
        for t in range(T - 1):
            h = states[t]
            if augment:
                u = E[ids[t]]; inj = Win @ u; lam = _sig(w @ u + b); c = lam * c + (1.0 - lam) * inj
                feat = np.concatenate([h, c])
            else:
                feat = h
            p = _softmax(Wro @ feat); ce += -math.log(max(p[ids[t + 1]], 1e-12)); cnt += 1
    return ce / cnt


def _bag_ce(tr_cache, ev_cache, V, seed):
    """CE of the same simple trainer over the memoryless BAG-of-prefix features (the headline control)."""
    trb = _bag_cache(tr_cache, V); evb = _bag_cache(ev_cache, V)
    return _joint_train_eval(trb, evb, V, False, seed, None, None, None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-sentences", type=int, default=30000)
    ap.add_argument("--n-train", type=int, default=2800)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--batch-m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_reslm_scale_trained_selssm.json")
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

    rng = np.random.default_rng(a.seed * 47 + 3)
    E = rng.standard_normal((V, D_SEL)) * 0.8
    Win = rng.standard_normal((N_SSM, D_SEL)) / np.sqrt(D_SEL)
    w0 = rng.standard_normal((N_SSM, D_SEL)) / np.sqrt(D_SEL); b0 = np.full(N_SSM, FORGET_BIAS)
    Bc = np.random.default_rng(a.seed * 191 + 11).standard_normal((N_SSM, V)) / np.sqrt(V)

    res_ce = _joint_train_eval(tr_cache, ev_cache, V, False, a.seed, E, Win, w0, b0, Bc)
    sel_ce = _joint_train_eval(tr_cache, ev_cache, V, True, a.seed, E, Win, w0, b0, Bc)
    bag_ce = _bag_ce(tr_cache, ev_cache, V, a.seed)
    P_bi = fit_bigram(tr_ids, V); bi_ce, _, _ = bigram_ce(P_bi, ev_ids)

    m_res = bag_ce - res_ce; m_sel = bag_ce - sel_ce
    res = dict(n_pool=a.n_pool, n_train=len(tr), V=V, res_ce=round(res_ce, 4), sel_ce=round(sel_ce, 4),
               bag_ce=round(bag_ce, 4), bi_ce=round(bi_ce, 4), margin_res_over_bag=round(m_res, 4),
               margin_sel_over_bag=round(m_sel, 4), sel_lift=round(m_sel - m_res, 4),
               sel_over_bigram=round(bi_ce - sel_ce, 4), collect_s=round(collect_s, 1), total_s=round(time.time() - t0, 1))
    json.dump(res, open(a.out, "w"))
    print(f"[trained-selssm-scale] np={a.n_pool} nt={len(tr)} V={V}: margin_over_BAG res={m_res:+.4f} sel={m_sel:+.4f} "
          f"(sel_lift={m_sel - m_res:+.4f}) | sel_over_bigram={bi_ce - sel_ce:+.4f} | res_ce={res_ce:.3f} sel_ce={sel_ce:.3f} "
          f"bag={bag_ce:.3f} bigram={bi_ce:.3f} -> {'TRAINED SEL LIFTS' if m_sel > m_res + 0.005 else 'no lift'}", flush=True)


if __name__ == "__main__":
    main()
