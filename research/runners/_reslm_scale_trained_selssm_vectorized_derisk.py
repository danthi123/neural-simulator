"""VECTORIZED (mini-batch) trained-selective scale trainer — the ENABLER for the validated regime (V=2000, large data) on
GPU. The per-token Python loop in `_reslm_batched_scale_trained_selssm_derisk` is the bottleneck (it caps nt~3200); this
processes a mini-batch of B sentences' token-t IN PARALLEL (padded + masked), so the whole trainer is O(tokens/B) Python
steps + big matmuls (GPU-friendly). Same mechanism: a FIXED echo-state reservoir (batched-collected) + a TRAINED selective
gate + read-out over [h, c], everything transport-free (read-out mini-batch delta; gate forward eligibility x FIXED RANDOM
FEEDBACK; no BPTT/transport). The read-out update is per-token-averaged-over-the-mini-batch (mini-batch SGD) — a DIFFERENT
optimizer path than the slow runner's online per-sentence SGD, so this REPRODUCES the qualitative result (margin_over_BAG
lift), not a bit-exact match.

VALIDATION GATE (run first): at the small scale where the slow runner gave sel_lift ~+0.62, this vectorized trainer must
reproduce a comparable sel_lift (the mechanism, not the exact SGD path). Then scale V/data toward the validated regime.

Reuse-by-import: the batched reservoir cache + Vocab/fit_bigram/bigram_ce/load_sentences. NO `sim/` edit. Backend-aware
(SIM_BACKEND=cupy -> the mini-batch matmuls run on GPU).

Run: E:/.../python.exe -m research.runners._reslm_scale_trained_selssm_vectorized_derisk --n-pool 200 --n-train 800 --vocab 120 --seed 42
"""
import argparse, time, json, math
import numpy as np

from collections import defaultdict
from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram, bigram_ce, _bag_cache
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import _bucket
import research.runners._reslm_batched_reservoir_derisk as BR
from research.runners._reslm_batched_scale_run import _cache_batched
from sim.backend import get_backend

N_SSM = 200
D_SEL = 32
FORGET_BIAS = 2.5
EPOCHS = 12
LR_RO = 0.02
LR_GATE = 0.4
MB = 64                    # mini-batch of sentences trained in parallel


def _pad_batch(sl):
    """Pad a slice of the cache (list of (states[T x n], ids)) into arrays: H [B x Lmax x n], ID [B x Lmax], M [B x Lmax]
    (1 where token t AND target t+1 are valid), Lmax = max sentence length in the slice."""
    B = len(sl); n = np.asarray(sl[0][0]).shape[1]
    Ls = [min(len(np.asarray(st)), len(ids)) for st, ids in sl]
    Lmax = max(Ls)
    H = np.zeros((B, Lmax, n), np.float32); ID = np.zeros((B, Lmax), np.int64); M = np.zeros((B, Lmax), np.float32)
    for b, (st, ids) in enumerate(sl):
        st = np.asarray(st); L = Ls[b]
        H[b, :L] = st[:L]; ID[b, :L] = ids[:L]
        M[b, :max(0, L - 1)] = 1.0                            # positions with a valid next-token target
    return H, ID, M


def _train_vec(xp, cache, V, mode, E, Win, w0, b0, Bc, rseed=0):
    """Vectorized mini-batch trainer. mode in {res, sel, noheld, rand}: res=read-out over h only; sel=trained selective gate
    (current token); noheld=selective with lam=0 (c=inj, pure current-token projection, gate NOT trained -> isolates the
    shallow readout fix); rand=trained gate but lam reads a RANDOM token (inj still the real token -> selective-specific
    control). Returns (Wro, w, b)."""
    use_sel = mode != "res"
    train_gate = mode in ("sel", "rand")
    n = int(np.asarray(cache[0][0]).shape[1]); fdim = n + (N_SSM if use_sel else 0)
    Wro = xp.zeros((V, fdim), xp.float32)
    w = xp.asarray(w0) if use_sel else None
    b = xp.asarray(b0) if use_sel else None
    E_x = xp.asarray(E) if use_sel else None
    Win_x = xp.asarray(Win) if use_sel else None
    Bc_x = xp.asarray(Bc) if use_sel else None
    rng = np.random.default_rng(rseed * 131 + 7)
    order = list(range(0, len(cache), MB))
    for _ep in range(EPOCHS):
        for i in order:
            sl = cache[i:i + MB]
            if len(sl) < 2:
                continue
            H, ID, Msk = _pad_batch(sl)
            Hx = xp.asarray(H); IDx = xp.asarray(ID); Mx = xp.asarray(Msk)
            Bn, Lmax, _ = Hx.shape
            rIDx = xp.asarray(rng.integers(0, V, size=(Bn, Lmax))) if mode == "rand" else None  # random-token gate stream
            c = xp.zeros((Bn, N_SSM), xp.float32)
            ew = xp.zeros((Bn, N_SSM, D_SEL), xp.float32); ec = xp.zeros((Bn, N_SSM), xp.float32)
            for t in range(Lmax - 1):
                m = Mx[:, t][:, None]                        # [B x 1] active mask
                h = Hx[:, t]                                 # [B x n]
                if use_sel:
                    u = E_x[IDx[:, t]]                       # [B x D] current-token embedding (the INJECTION)
                    inj = u @ Win_x.T                        # [B x N_ssm]
                    ug = E_x[rIDx[:, t]] if mode == "rand" else u   # gate input (random token for rand)
                    if mode == "noheld":
                        lam = xp.zeros((Bn, N_SSM), xp.float32)     # no hold: c = inj (current-token projection)
                    else:
                        lam = 1.0 / (1.0 + xp.exp(-xp.clip(ug @ w.T + b, -30, 30)))
                    c_prev = c; c = m * (lam * c_prev + (1.0 - lam) * inj) + (1.0 - m) * c
                    if train_gate:
                        dl = lam * (1.0 - lam); base = (c_prev - inj)
                        ew = m[:, :, None] * (lam[:, :, None] * ew + (dl * base)[:, :, None] * ug[:, None, :]) + (1.0 - m)[:, :, None] * ew
                        ec = m * (lam * ec + dl * base) + (1.0 - m) * ec
                    feat = xp.concatenate([h, c], axis=1)    # [B x fdim]
                else:
                    feat = h
                z = feat @ Wro.T                             # [B x V]
                z = z - z.max(axis=1, keepdims=True); p = xp.exp(z); p = p / p.sum(axis=1, keepdims=True)
                err = p
                tgt = IDx[:, t + 1]
                err[xp.arange(Bn), tgt] -= 1.0
                err = err * m                                # zero inactive rows
                nact = float(m.sum()) + 1e-6
                Wro -= LR_RO * (err.T @ feat) / nact         # mini-batch delta update
                if train_gate:
                    delta_c = err @ Bc_x.T                   # [B x V] @ [V x N_ssm] = [B x N_ssm]
                    w -= LR_GATE * xp.einsum("bi,bij->ij", delta_c, ew) / nact
                    b -= LR_GATE * (delta_c * 1.0).sum(axis=0) / nact
    return Wro, w, b


def _eval_ce(xp, Wro, w, b, cache, V, mode, E, Win, rseed=0):
    """Returns (overall_ce, by_depth_ce_dict). Per-depth via _bucket(position). Mirrors _train_vec's mode forward."""
    use_sel = mode != "res"
    E_x = xp.asarray(E) if use_sel else None; Win_x = xp.asarray(Win) if use_sel else None
    rng = np.random.default_rng(rseed * 131 + 7)
    ce = 0.0; cnt = 0; dce = defaultdict(float); dcnt = defaultdict(int)
    for i in range(0, len(cache), MB):
        sl = cache[i:i + MB]
        if not sl:
            continue
        H, ID, Msk = _pad_batch(sl)
        Hx = xp.asarray(H); IDx = xp.asarray(ID); Mx = xp.asarray(Msk)
        Bn, Lmax, _ = Hx.shape
        rIDx = xp.asarray(rng.integers(0, V, size=(Bn, Lmax))) if mode == "rand" else None
        c = xp.zeros((Bn, N_SSM), xp.float32)
        for t in range(Lmax - 1):
            m = Mx[:, t][:, None]; h = Hx[:, t]
            if use_sel:
                u = E_x[IDx[:, t]]; inj = u @ Win_x.T
                ug = E_x[rIDx[:, t]] if mode == "rand" else u
                lam = xp.zeros((Bn, N_SSM), xp.float32) if mode == "noheld" else 1.0 / (1.0 + xp.exp(-xp.clip(ug @ w.T + b, -30, 30)))
                c = m * (lam * c + (1.0 - lam) * inj) + (1.0 - m) * c
                feat = xp.concatenate([h, c], axis=1)
            else:
                feat = h
            z = feat @ Wro.T; z = z - z.max(axis=1, keepdims=True); p = xp.exp(z); p = p / p.sum(axis=1, keepdims=True)
            tgt = IDx[:, t + 1]
            nll = -xp.log(xp.clip(p[xp.arange(Bn), tgt], 1e-12, 1.0)) * Mx[:, t]
            ce += float(nll.sum()); cnt += int(Mx[:, t].sum())
            bd = _bucket(t + 1); dce[bd] += float(nll.sum()); dcnt[bd] += int(Mx[:, t].sum())
    by_depth = {k: dce[k] / dcnt[k] for k in dcnt if dcnt[k]}
    return ce / max(1, cnt), by_depth


def _bag_pack(cache, V):
    """bag features as (states-list, ids) matching _pad_batch's expectation (np.asarray of the states-list)."""
    return _bag_cache(cache, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-sentences", type=int, default=30000)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-pool", type=int, default=200)
    ap.add_argument("--vocab", type=int, default=120)
    ap.add_argument("--batch-m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_reslm_scale_trained_selssm_vec.json")
    a = ap.parse_args()

    xp, backend = get_backend()
    t0 = time.time()
    sents = load_sentences(a.corpus, a.n_sentences)
    perm = np.random.default_rng(a.seed).permutation(len(sents))
    ev = [sents[i] for i in perm[-a.n_eval:]]; pool = [sents[i] for i in perm[:-a.n_eval]]
    vocab = Vocab.build(pool, V=a.vocab); V = vocab.size
    in_dim = len(vocab.encode_seq(pool[0])[0])
    tr = pool[:a.n_train]
    tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]

    b, copy_res, W_in, snap = BR.build_batched(a.seed, a.n_pool, in_dim, a.batch_m)
    tr_cache = _cache_batched(b, copy_res, W_in, snap, vocab, tr, a.batch_m)
    ev_cache = _cache_batched(b, copy_res, W_in, snap, vocab, ev, a.batch_m)

    rng = np.random.default_rng(a.seed * 47 + 3)
    E = rng.standard_normal((V, D_SEL)).astype(np.float32) * 0.8
    Win = (rng.standard_normal((N_SSM, D_SEL)) / np.sqrt(D_SEL)).astype(np.float32)
    w0 = (rng.standard_normal((N_SSM, D_SEL)) / np.sqrt(D_SEL)).astype(np.float32); b0 = np.full(N_SSM, FORGET_BIAS, np.float32)
    Bc = (np.random.default_rng(a.seed * 191 + 11).standard_normal((N_SSM, V)) / np.sqrt(V)).astype(np.float32)

    # 4 arms with per-depth CE (the airtight controls the adversarial-verify named)
    bd = {}
    Wr, _, _ = _train_vec(xp, tr_cache, V, "res", E, Win, w0, b0, Bc, a.seed)
    res_ce, bd["res"] = _eval_ce(xp, Wr, None, None, ev_cache, V, "res", E, Win, a.seed)
    Ws, ws, bs = _train_vec(xp, tr_cache, V, "sel", E, Win, w0, b0, Bc, a.seed)
    sel_ce, bd["sel"] = _eval_ce(xp, Ws, ws, bs, ev_cache, V, "sel", E, Win, a.seed)
    Wn, _, _ = _train_vec(xp, tr_cache, V, "noheld", E, Win, w0, b0, Bc, a.seed)
    noh_ce, bd["noheld"] = _eval_ce(xp, Wn, None, None, ev_cache, V, "noheld", E, Win, a.seed)
    Wra, wr, br = _train_vec(xp, tr_cache, V, "rand", E, Win, w0, b0, Bc, a.seed)
    rnd_ce, bd["rand"] = _eval_ce(xp, Wra, wr, br, ev_cache, V, "rand", E, Win, a.seed)
    Wb, _, _ = _train_vec(xp, _bag_pack(tr_cache, V), V, "res", E, Win, w0, b0, Bc, a.seed)
    bag_ce, bd["bag"] = _eval_ce(xp, Wb, None, None, _bag_pack(ev_cache, V), V, "res", E, Win, a.seed)
    # per-depth ORDERED bigram (the proper long-range baseline, not the order-blind bag)
    P_bi = fit_bigram(tr_ids, V); bi_ce, _, _ = bigram_ce(P_bi, ev_ids)
    bdce = defaultdict(float); bdcnt = defaultdict(int)
    for ids in ev_ids:
        for t in range(len(ids) - 1):
            k = _bucket(t + 1); bdce[k] += -math.log(max(P_bi[ids[t], ids[t + 1]], 1e-12)); bdcnt[k] += 1
    bd["bigram"] = {k: bdce[k] / bdcnt[k] for k in bdcnt if bdcnt[k]}

    # AIRTIGHT deep-tail (d>=6, d>=10) comparisons: sel vs ORDERED bigram, vs noheld (current-token), vs rand (selective-specific)
    def _dtail(depths):
        out = {}
        for name in ("bigram", "noheld", "rand"):
            num = 0.0; den = 0
            for db in depths:
                if db in bd["sel"] and db in bd[name]:
                    # weight by eval token count via re-deriving from bigram counts
                    w_ = bdcnt.get(db, 1); num += (bd[name][db] - bd["sel"][db]) * w_; den += w_
            out[f"sel_beats_{name}"] = round(num / den, 4) if den else None
        return out
    dtail6 = _dtail(["6-9", "10-99"]); dtail10 = _dtail(["10-99"])

    m_res = bag_ce - res_ce; m_sel = bag_ce - sel_ce
    res = dict(backend=backend, n_pool=a.n_pool, n_train=len(tr), V=V, res_ce=round(res_ce, 4), sel_ce=round(sel_ce, 4),
               noheld_ce=round(noh_ce, 4), rand_ce=round(rnd_ce, 4), bag_ce=round(bag_ce, 4), bi_ce=round(bi_ce, 4),
               margin_sel_over_bag=round(m_sel, 4), sel_lift=round(m_sel - m_res, 4), sel_over_bigram=round(bi_ce - sel_ce, 4),
               by_depth=bd, deep_tail_d6=dtail6, deep_tail_d10=dtail10, total_s=round(time.time() - t0, 1))
    json.dump(res, open(a.out, "w"))
    print(f"[vec-airtight] [{backend}] np={a.n_pool} nt={len(tr)} V={V} in {res['total_s']}s: "
          f"sel_lift(vs bag)={m_sel - m_res:+.4f} sel_over_bigram(agg)={bi_ce - sel_ce:+.4f} | "
          f"DEEP d>=6: sel<bigram {dtail6['sel_beats_bigram']} sel<noheld {dtail6['sel_beats_noheld']} "
          f"sel<rand {dtail6['sel_beats_rand']} | DEEP d>=10: sel<bigram {dtail10['sel_beats_bigram']} "
          f"sel<rand {dtail10['sel_beats_rand']}", flush=True)


if __name__ == "__main__":
    main()
