"""ADVERSARIAL-VERIFY CONTROLS for the e-prop-trained recurrent language-cortex claim (the 5-skeptic
`GO_needs_more_controls` verdict on `2026-07-14-eprop-...-3seed-GO.md`). This runner DOES NOT try to make the claim pass;
it implements the 4 MUST-RUN controls to DECIDE whether ANYTHING survives, honestly. Reuse-by-import from the committed
runner (`_emerge_reservoir_lm_eprop_recurrent_derisk`) + the n-gram machinery; NO `sim/` edit; numpy CPU.

The claim under test: making W_rec plastic via transport-free e-prop LOWERS held-out CE at DEEP context buckets (target
absolute position t+1 >= 6) vs the frozen reservoir, 3/3 seeds, and "beats the bigram at deep context". The confounds the
skeptics named:
  (1) "DEEP" = the target's ABSOLUTE POSITION, not the length of the predictive dependency -> a plain higher-order n-gram
      measured at late positions would also beat a bigram there. => DISTAL-PREFIX SCRAMBLE decides position-vs-dependency.
  (2) "beats the bigram" rests on ONE bucket vs a deliberately-WEAK add-1 bigram at V=300. => position-matched TRIGRAM /
      4-gram / interpolated-trigram / add-k bigram + TOKEN-COUNT-WEIGHTED deep CE.
  (3) The shallow-hurt/deep-help CROSSOVER is the fingerprint of a memory-timescale RETUNING, which shuffle_elig/zero_signal
      do NOT rule out (they kill INCOHERENT motion; a coherent operating-point shift passes through). => a COHERENT
      credit-IRRELEVANT arm (`random_signal` = randomized-target delta; `random_dir` = fixed-direction magnitude-matched)
      + log ||W_rec||_F & spectral radius per epoch. If it reproduces the crossover -> retuning, not credit.
  (4) Is the random-feedback update aligned with the true gradient? => `symmetric` (weight-transport ceiling) + `sign_flip`
      (must HURT) arms + a running feedback-alignment COSINE between the FA update and the true-gradient update.

`train_ctrl` re-implements the committed `train` non-ALIF path VERBATIM for the shared modes (a startup faithcheck asserts
np.allclose(W_rec, W_out) vs the imported `train`), and ADDS the modes {random_signal, random_dir, sign_flip} + per-epoch
logging. So `fixed`/`plastic` reproduce the committed synth_s{42,43,44}.json numbers bit-for-bit (the anchor).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, _softmax, fit_bigram
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket
from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir, train as train_ref

OUT = Path("research/findings/raw/_eproplm_controls")
DEEP_BUCKETS = ("6-9", "10-99")            # the two "deep" buckets the headline averaged (target position t+1 >= 6)


# =====================================================================================================================
# train_ctrl -- verbatim re-impl of the committed non-ALIF `train` for the shared modes, + the new control arms + logging
# =====================================================================================================================
def train_ctrl(res, tr_ids, V, epochs, lr_out, lr_rec, seed, mode="plastic", wd=1e-3, wd_rec=0.0, log=False):
    """Modes: fixed, plastic, shuffle_elig, zero_signal, symmetric  (== committed `train`, verified allclose)
              sign_flip     -- L := -(B@delta): the wrong-direction FA update; MUST HURT.
              random_signal -- COHERENT credit-IRRELEVANT: L = B@delta_rand where delta_rand puts the +1 on a UNIFORMLY
                               RANDOM in-vocab token (same delta structure/magnitude, credit decoupled from the true target).
              random_dir    -- COHERENT credit-IRRELEVANT: L = (fixed random unit vector r) * ||B@delta|| (a single fixed
                               per-neuron direction, magnitude-matched to plastic's learning signal each step).
       log=True records per-epoch ||W_rec||_F, spectral radius, and the running feedback-alignment cosine (FA update vs the
       true-gradient W_out^T@delta update; computed cheaply via the shared-eligibility row-energy reduction)."""
    rng = np.random.default_rng(seed * 13 + 7)
    n = res.n; a = res.alpha
    W_out = rng.standard_normal((V, n)) * 0.01
    B = rng.standard_normal((n, V)) / np.sqrt(V)          # fixed random feedback (broadcast alignment) -- drawn to MATCH
    symmetric = (mode == "symmetric")
    r_dir = None
    if mode == "random_dir":
        r = rng.standard_normal(n); r_dir = r / (np.linalg.norm(r) + 1e-30)   # fixed coherent direction (per seed)
    trng = np.random.default_rng(seed * 971 + 3)          # separate stream for the randomized target (credit ablation)
    order = np.arange(len(tr_ids))
    ep_log = []
    for ep in range(epochs):
        rng.shuffle(order)
        cos_num = 0.0; cos_den_a = 0.0; cos_den_b = 0.0    # accumulate the FA cosine over the epoch (row-energy reduction)
        n_upd = 0
        for si in order:
            ids = tr_ids[si]
            if len(ids) < 2:
                continue
            h = np.zeros(n); e = np.zeros((n, n))
            for t in range(len(ids) - 1):
                h_prev = h
                x = res.W_in[:, ids[t]]
                pre = res.W_rec @ h_prev + x + res.b
                act = np.tanh(pre)
                h = (1 - a) * h_prev + a * act
                p = _softmax(W_out @ h)
                delta = -p; delta[ids[t + 1]] += 1.0
                W_out += lr_out * (np.outer(delta, h) - wd * W_out)
                if mode == "fixed":
                    continue
                psi = a * (1.0 - act * act)
                e = (1 - a)[:, None] * e + np.outer(psi, h_prev)   # fast forward-filtered eligibility
                if mode == "zero_signal":
                    continue
                # ---- learning signal L for each arm ----------------------------------------------------------------
                Lfa = B @ delta                                    # the random-feedback signal (plastic)
                if symmetric:
                    L = W_out.T @ delta
                elif mode == "sign_flip":
                    L = -Lfa
                elif mode == "random_signal":
                    dr = -p; dr[trng.integers(V)] += 1.0           # delta with the +1 on a RANDOM token (credit ablated)
                    L = B @ dr
                elif mode == "random_dir":
                    L = r_dir * float(np.linalg.norm(Lfa))         # fixed direction, magnitude-matched to plastic
                else:                                              # plastic / shuffle_elig
                    L = Lfa
                E_use = e
                if mode == "shuffle_elig":
                    E_use = e.reshape(-1)[rng.permutation(n * n)].reshape(n, n)
                res.W_rec += lr_rec * (L[:, None] * E_use)
                if wd_rec > 0.0:
                    res.W_rec -= lr_rec * wd_rec * res.W_rec
                # ---- running FA cosine: cos(L[:,None]*e , Lsym[:,None]*e) via row energies s_j = sum_i e[j,i]^2 -----
                if log:
                    Lsym = W_out.T @ delta
                    s = np.einsum("ji,ji->j", e, e)                # row energy (n,)
                    cos_num += float(np.sum(L * Lsym * s))
                    cos_den_a += float(np.sum(L * L * s))
                    cos_den_b += float(np.sum(Lsym * Lsym * s))
                    n_upd += 1
        if log:
            fro = float(np.linalg.norm(res.W_rec))
            try:
                sr = float(np.max(np.abs(np.linalg.eigvals(res.W_rec))))
            except Exception:
                sr = float("nan")
            cos = (cos_num / math.sqrt(cos_den_a * cos_den_b)) if (cos_den_a > 0 and cos_den_b > 0) else float("nan")
            ep_log.append({"epoch": ep, "W_rec_fro": round(fro, 3), "spectral_radius": round(sr, 4),
                           "fa_cosine": round(cos, 4) if cos == cos else None, "n_upd": n_upd})
    return W_out, ep_log


# =====================================================================================================================
# per-depth CE (reservoir) with sums for token-weighting; + distal-prefix scramble
# =====================================================================================================================
def reservoir_depth_sums(res, W_out, ev_ids):
    """Per-bucket reservoir CE SUMS + counts (so deep buckets can be TOKEN-WEIGHTED, not the unweighted 2-bucket mean)."""
    s = defaultdict(float); c = defaultdict(int)
    for ids in ev_ids:
        states = res.forward_states(ids)
        for t in range(len(ids) - 1):
            b = _bucket(t + 1)
            p = _softmax(W_out @ states[t])
            s[b] += -math.log(max(p[ids[t + 1]], 1e-12)); c[b] += 1
    return s, c


def reservoir_deep_scramble_sums(res, W_out, ev_ids, V, keep=4, scramble_seed=7000):
    """DISTAL-PREFIX SCRAMBLE (control 1). For each DEEP target (t+1 >= 6), replace tokens at positions 0..t-keep with random
       in-vocab tokens (destroying the distal prefix) while KEEPING the trailing `keep` tokens (positions t-keep+1..t) intact
       (the local n-gram window), then run the reservoir forward over the scrambled prefix and score position t. The random
       tokens are drawn deterministically per (sentence, t) so the fixed & plastic arms see the SAME scramble (paired margin).
       If the plastic-vs-fixed deep margin SURVIVES -> the gain is LOCAL structure measured late ('deep' is a misnomer). If it
       COLLAPSES -> genuine long-range dependency capture."""
    s = defaultdict(float); c = defaultdict(int)
    for si, ids in enumerate(ev_ids):
        L = len(ids)
        for t in range(L - 1):
            if (t + 1) < 6:
                continue                                     # deep only
            sc = list(ids)
            hi = t - keep                                    # replace positions 0..t-keep (inclusive)
            if hi >= 0:
                srng = np.random.default_rng(scramble_seed + si * 100003 + t)
                for j in range(hi + 1):
                    sc[j] = int(srng.integers(V))
            states = res.forward_states(sc[:t + 1])
            p = _softmax(W_out @ states[-1])
            b = _bucket(t + 1)
            s[b] += -math.log(max(p[ids[t + 1]], 1e-12)); c[b] += 1
    return s, c


# =====================================================================================================================
# n-gram baselines, scored PER-DEPTH-BUCKET at the matched eval positions (control 2)
# =====================================================================================================================
def fit_bigram_addk(tr_ids, V, k):
    c = np.zeros((V, V))
    for ids in tr_ids:
        for a, b in zip(ids, ids[1:]):
            c[a, b] += 1.0
    return (c + k) / (c.sum(1, keepdims=True) + k * V)


def _mle_counts(tr_ids, V):
    uni = np.zeros(V)
    bi = defaultdict(lambda: defaultdict(float)); bic = defaultdict(float)
    tri = defaultdict(lambda: defaultdict(float)); tric = defaultdict(float)
    for ids in tr_ids:
        for i, w in enumerate(ids):
            uni[w] += 1.0
            if i >= 1:
                bi[ids[i - 1]][w] += 1.0; bic[ids[i - 1]] += 1.0
            if i >= 2:
                key = (ids[i - 2], ids[i - 1]); tri[key][w] += 1.0; tric[key] += 1.0
    uni_p = (uni + 1.0) / (uni.sum() + V)
    return uni_p, bi, bic, tri, tric


def _interp_prob(w, c1, c2, uni_p, bi, bic, tri, tric, lam):
    l3, l2, l1 = lam
    pt = (tri[(c1, c2)].get(w, 0.0) / tric[(c1, c2)]) if tric.get((c1, c2), 0.0) > 0 else 0.0
    pb = (bi[c2].get(w, 0.0) / bic[c2]) if bic.get(c2, 0.0) > 0 else 0.0
    return l3 * pt + l2 * pb + l1 * uni_p[w]


def tune_interp_lambda(tr_ids, V, holdout_frac=0.1, seed=0):
    """Jelinek-Mercer deleted-interpolation trigram: tune (l3,l2,l1) on a held-out train slice (grid, maximize likelihood)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tr_ids)); cut = int((1 - holdout_frac) * len(tr_ids))
    fit_ids = [tr_ids[i] for i in idx[:cut]]; ho_ids = [tr_ids[i] for i in idx[cut:]]
    uni_p, bi, bic, tri, tric = _mle_counts(fit_ids, V)
    best = None
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for l3 in grid:
        for l2 in grid:
            if l3 + l2 > 0.98:
                continue
            l1 = 1.0 - l3 - l2
            if l1 < 0.02:
                continue
            tot = 0.0; nn = 0
            for ids in ho_ids:
                for t in range(1, len(ids) - 1):
                    p = _interp_prob(ids[t + 1], ids[t - 1], ids[t], uni_p, bi, bic, tri, tric, (l3, l2, l1))
                    tot += -math.log(max(p, 1e-12)); nn += 1
            ce = tot / max(1, nn)
            if best is None or ce < best[0]:
                best = (ce, (l3, l2, l1))
    # refit the counts on the FULL train with the tuned lambda
    uni_p, bi, bic, tri, tric = _mle_counts(tr_ids, V)
    return best[1], (uni_p, bi, bic, tri, tric)


def tune_bigram_addk(tr_ids, V, holdout_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tr_ids)); cut = int((1 - holdout_frac) * len(tr_ids))
    fit_ids = [tr_ids[i] for i in idx[:cut]]; ho_ids = [tr_ids[i] for i in idx[cut:]]
    best = None
    for k in (0.01, 0.03, 0.05, 0.1, 0.3, 1.0):
        P = fit_bigram_addk(fit_ids, V, k)
        tot = 0.0; nn = 0
        for ids in ho_ids:
            for a, b in zip(ids, ids[1:]):
                tot += -math.log(max(P[a, b], 1e-12)); nn += 1
        ce = tot / max(1, nn)
        if best is None or ce < best[0]:
            best = (ce, k)
    return best[1], fit_bigram_addk(tr_ids, V, best[1])


def ngram_depth_sums(ev_ids, V, P_bi1, P_bik, tri_add1, four_add1, interp):
    """Per-depth-bucket CE SUMS for each n-gram baseline at the matched eval positions. tri_add1/four_add1 are the committed
       add-1 backoff tables (ctx dicts); interp = (lambda,(uni_p,bi,bic,tri,tric)). Returns dict[name] -> (sums, counts)."""
    names = ["bigram_add1", "bigram_addk", "trigram_add1", "fourgram_add1", "trigram_interp"]
    S = {nm: defaultdict(float) for nm in names}; C = defaultdict(int)
    lam, (uni_p, bi, bic, tri, tric) = interp
    for ids in ev_ids:
        for t in range(len(ids) - 1):
            b = _bucket(t + 1); tgt = ids[t + 1]; C[b] += 1
            S["bigram_add1"][b] += -math.log(max(P_bi1[ids[t], tgt], 1e-12))
            S["bigram_addk"][b] += -math.log(max(P_bik[ids[t], tgt], 1e-12))
            # add-1 trigram (committed backoff-to-bigram at unseen context)
            if t >= 1 and (ids[t - 1], ids[t]) in tri_add1:
                row = tri_add1[(ids[t - 1], ids[t])]; pt = row[tgt] / row.sum()
            else:
                pt = P_bi1[ids[t], tgt]
            S["trigram_add1"][b] += -math.log(max(pt, 1e-12))
            # add-1 4-gram (backoff 3->2->bigram)
            if t >= 2 and (ids[t - 2], ids[t - 1], ids[t]) in four_add1:
                row = four_add1[(ids[t - 2], ids[t - 1], ids[t])]; pf = row[tgt] / row.sum()
            elif t >= 1 and (ids[t - 1], ids[t]) in tri_add1:
                row = tri_add1[(ids[t - 1], ids[t])]; pf = row[tgt] / row.sum()
            else:
                pf = P_bi1[ids[t], tgt]
            S["fourgram_add1"][b] += -math.log(max(pf, 1e-12))
            # interpolated trigram (Jelinek-Mercer, tuned lambda)
            c1 = ids[t - 1] if t >= 1 else ids[t]
            pi = _interp_prob(tgt, c1, ids[t], uni_p, bi, bic, tri, tric, lam)
            S["trigram_interp"][b] += -math.log(max(pi, 1e-12))
    return S, C, names, lam


# =====================================================================================================================
def _deep_weighted(sums, cnts):
    ss = sum(sums[b] for b in DEEP_BUCKETS if b in cnts); nn = sum(cnts[b] for b in DEEP_BUCKETS if b in cnts)
    return (ss / nn) if nn else float("nan"), nn


def _faithcheck():
    """Assert train_ctrl == the committed train on the shared modes (tiny synthetic corpus), before trusting the new arms."""
    rng = np.random.default_rng(0)
    V = 15; n = 20
    tr = [list(rng.integers(0, V, size=int(rng.integers(3, 9)))) for _ in range(12)]
    for mode in ("fixed", "plastic", "shuffle_elig", "zero_signal", "symmetric"):
        r1 = RateReservoir(V, n, 42); r2 = RateReservoir(V, n, 42)
        Wr, _ = train_ctrl(r1, tr, V, 3, 0.02, 0.002, 42, mode=mode)
        Wref = train_ref(r2, tr, V, 3, 0.02, 0.002, 42, mode=mode)
        assert np.allclose(r1.W_rec, r2.W_rec, atol=1e-10), f"W_rec mismatch in {mode}"
        assert np.allclose(Wr, Wref, atol=1e-10), f"W_out mismatch in {mode}"
    print("[faithcheck] train_ctrl == committed train on {fixed,plastic,shuffle_elig,zero_signal,symmetric}  ->  PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/wikitext.txt")
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--n-sentences", type=int, default=6000)
    ap.add_argument("--max-train-sents", type=int, default=1500)
    ap.add_argument("--max-eval-sents", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--spectral", type=float, default=1.1)
    ap.add_argument("--lr-out", type=float, default=0.02)
    ap.add_argument("--lr-rec", type=float, default=0.002)
    ap.add_argument("--scramble-keep", type=int, default=4)     # trailing local tokens preserved under distal scramble
    ap.add_argument("--modes", type=str, nargs="+",
                    default=["fixed", "plastic", "shuffle_elig", "zero_signal",
                             "random_signal", "random_dir", "symmetric", "sign_flip"])
    ap.add_argument("--no-faithcheck", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not args.no_faithcheck:
        _faithcheck()

    sents = load_sentences(args.corpus, args.n_sentences)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); per_seed = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(sents)); cut = int(0.8 * len(sents))
        tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
        ev = [sents[i] for i in idx[cut:]][:args.max_eval_sents]
        vocab = Vocab.build(tr, V=args.vocab); V = vocab.size
        tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
        P_bi1 = fit_bigram(tr_ids, V)                                  # committed add-1 bigram (the headline baseline)

        rec = {"V": V, "n_train": len(tr), "by_mode": {}, "scramble": {}, "ngram": {}}
        res_by_mode = {}; W_by_mode = {}
        for mode in args.modes:
            res = RateReservoir(V, args.n_pool, seed, alpha=args.alpha, spectral=args.spectral)
            do_log = mode in ("plastic", "symmetric", "sign_flip", "random_signal", "random_dir")
            W_out, ep_log = train_ctrl(res, tr_ids, V, args.epochs, args.lr_out, args.lr_rec, seed, mode=mode, log=do_log)
            s, c = reservoir_depth_sums(res, W_out, ev_ids)
            depth = {b: {"n": c[b], "ce": round(s[b] / c[b], 3)} for b in c}
            dwc, dwn = _deep_weighted(s, c)
            rec["by_mode"][mode] = {"aggregate_ce": round(sum(s.values()) / sum(c.values()), 3),
                                    "deep_weighted_ce": round(dwc, 4), "deep_n": dwn, "by_depth": depth,
                                    "ep_log": ep_log, "_sums": {b: s[b] for b in c}, "_cnts": {b: c[b] for b in c}}
            res_by_mode[mode] = res; W_by_mode[mode] = W_out

        # ---- control 1: distal-prefix scramble (fixed vs plastic, paired) --------------------------------------------
        for mode in ("fixed", "plastic"):
            if mode in res_by_mode:
                s, c = reservoir_deep_scramble_sums(res_by_mode[mode], W_by_mode[mode], ev_ids, V,
                                                    keep=args.scramble_keep)
                dwc, dwn = _deep_weighted(s, c)
                rec["scramble"][mode] = {"deep_weighted_ce": round(dwc, 4), "deep_n": dwn,
                                         "by_depth": {b: {"n": c[b], "ce": round(s[b] / c[b], 3)} for b in c}}

        # ---- control 2: position-matched n-gram baselines (add-1 bi/tri/4gram, add-k bi, interp tri) ------------------
        from research.runners._emerge_reservoir_lm_derisk import fit_trigram, fit_fourgram
        tri_add1 = fit_trigram(tr_ids, V); four_add1 = fit_fourgram(tr_ids, V)
        k_star, P_bik = tune_bigram_addk(tr_ids, V, seed=seed)
        lam, interp_counts = tune_interp_lambda(tr_ids, V, seed=seed)
        Sn, Cn, names, lam = ngram_depth_sums(ev_ids, V, P_bi1, P_bik, tri_add1, four_add1, (lam, interp_counts))
        for nm in names:
            dwc, dwn = _deep_weighted(Sn[nm], Cn)
            rec["ngram"][nm] = {"deep_weighted_ce": round(dwc, 4),
                                "by_depth": {b: round(Sn[nm][b] / Cn[b], 3) for b in Cn}}
        rec["ngram"]["_meta"] = {"addk_k": k_star, "interp_lambda": [round(x, 2) for x in lam]}
        per_seed[str(seed)] = rec

        # ---- console summary per seed --------------------------------------------------------------------------------
        fx = rec["by_mode"]["fixed"]; pl = rec["by_mode"]["plastic"]
        print(f"\n===== seed {seed}  V={V} =====", flush=True)
        print(f"  DEEP token-weighted CE (n={pl['deep_n']}):  fixed {fx['deep_weighted_ce']}  plastic {pl['deep_weighted_ce']}"
              f"  (plastic-minus-fixed {pl['deep_weighted_ce'] - fx['deep_weighted_ce']:+.4f})", flush=True)
        for mode in args.modes:
            if mode == "fixed":
                continue
            mm = rec["by_mode"][mode]
            d1 = mm["by_depth"].get("1", {}).get("ce", float("nan")) - fx["by_depth"]["1"]["ce"]
            ddeep = mm["deep_weighted_ce"] - fx["deep_weighted_ce"]
            lg = mm["ep_log"][-1] if mm["ep_log"] else {}
            print(f"    {mode:>14}: d1 vs fixed {d1:+.3f} | DEEP(wt) vs fixed {ddeep:+.4f} | agg {mm['aggregate_ce']}"
                  + (f" | ||W_rec|| {lg.get('W_rec_fro')} sr {lg.get('spectral_radius')} FAcos {lg.get('fa_cosine')}"
                     if lg else ""), flush=True)
        print(f"  [ctrl1 scramble] DEEP(wt) plastic-minus-fixed:  UNSCRAMBLED {pl['deep_weighted_ce'] - fx['deep_weighted_ce']:+.4f}"
              f"   SCRAMBLED {rec['scramble']['plastic']['deep_weighted_ce'] - rec['scramble']['fixed']['deep_weighted_ce']:+.4f}",
              flush=True)
        print(f"  [ctrl2 n-gram DEEP(wt)]: add1-bi {rec['ngram']['bigram_add1']['deep_weighted_ce']}  "
              f"addk-bi(k={k_star}) {rec['ngram']['bigram_addk']['deep_weighted_ce']}  "
              f"tri-add1 {rec['ngram']['trigram_add1']['deep_weighted_ce']}  "
              f"4g-add1 {rec['ngram']['fourgram_add1']['deep_weighted_ce']}  "
              f"tri-interp {rec['ngram']['trigram_interp']['deep_weighted_ce']}  "
              f"|| plastic {pl['deep_weighted_ce']}", flush=True)

    out = {"runner": "_emerge_reservoir_lm_eprop_CONTROLS_derisk", "corpus": args.corpus, "seeds": args.seeds,
           "n_pool": args.n_pool, "args": vars(args), "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    jpath = args.json or str(OUT / f"controls_s{'_'.join(map(str, args.seeds))}.json")
    Path(jpath).parent.mkdir(parents=True, exist_ok=True); Path(jpath).write_text(json.dumps(out, indent=2))
    print(f"\n-> {jpath} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
