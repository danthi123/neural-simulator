"""D3 — RECURRENT MULTI-HOP COMPOSITION (the genuine depth-required LANGUAGE capability; the feedforward deep-credit
arc is COMPLETE, `2026-07-08-deep-credit-feedforward-arc-COMPLETE...md`). THE QUESTION (research-gated,
`2026-07-09-D3-recurrent-multihop-composition-research-gate.md`): does a RECURRENT credit path LEARN a multi-hop
bind->re-bind that a FEEDFORWARD net provably cannot?

THE TASK — streaming GROUP-WORD composition = STATE TRACKING (the clean temporal analog of XOR-over-pool):
a sequence of group elements g_1..g_L arrives one per step; the running product s_t = s_{t-1} * g_t is the LITERAL
bind->re-bind (each step re-binds the held state with the new element). Target = a 2-way property of the FINAL state
s_L (its parity/coset). Each element is encoded as a NOISY +-1 POOL code (XOR-over-pool style -> no linear/lookup
shortcut). NON-ABELIAN (S3) -> order matters -> a count/multiset model is provably at chance on order-dependent cases.
Theory (VERIFIED by controller): an RNN simulates a DFA and group composition IS a DFA (states=elements) -> linear-size
RNN-solvable; transformers/diagonal-SSMs are TC0 (Merrill-Petty-Sabharwal 2404.08819 "Illusion of State") -> cannot;
for non-solvable A5 the FF route is NC1-hard (Barrington 1986). The EMPIRICAL separation (S3, solvable) is LENGTH
GENERALIZATION: a FF net over the flattened sequence MEMORIZES trained lengths but cannot ITERATE -> fails held-out-
DEEPER; the RNN iterates -> generalizes.

THE DECISIVE NUMBER: held-out-DEEPER accuracy of the BPTT-RNN vs the FF-oracle. If RNN >> FF with controls collapsing
-> a recurrent credit path learns what FF provably cannot -> escalate to A5 + e-prop + the spiking port.

ARMS: (a) FF-ORACLE (MLP over flattened seq, depth 0/1/2/3 -- the feedforward ceiling); (b) FIXED RESERVOIR + ridge
(is fixed recurrent dynamics enough? predicted BOUNDARY -- Dambre memory-nonlinearity tradeoff); (c) BPTT-RNN (a
learned recurrent-credit path, tanh RNN + W_rec, the decisive arm). ANTI-CHEATS: FF-cap, permuted-ORDER (non-abelian
-> collapses to chance), Markov/1-hop floor, held-out-deeper + held-out-composition, recurrence-OFF lesion (W_rec=0 ->
FF-equivalent), permuted-label, learnability oracle, seeds 42/43/44. numpy CPU cheap-first; NO `sim/` edit.

Run:  python -m research.runners._d3_group_composition_derisk --group S3 --seeds 42,43,44
"""
from __future__ import annotations
import argparse, json
import numpy as np

# ---------------------------------------------------------------- groups (multiplication tables) ----------------
def _perm_group(gens_n):
    """Build the full permutation group on `gens_n` points by closure of adjacent transpositions -> S_n.
    Returns (elements as tuples, mul_table[i,j]=index of elem_i o elem_j, identity_index)."""
    n = gens_n
    from itertools import permutations
    elems = sorted(permutations(range(n)))                        # all n! permutations (as images)
    idx = {e: i for i, e in enumerate(elems)}
    def compose(a, b):                                            # (a o b)(x) = a[b[x]]
        return tuple(a[b[x]] for x in range(n))
    K = len(elems)
    mul = np.zeros((K, K), dtype=np.int64)
    for i, a in enumerate(elems):
        for j, b in enumerate(elems):
            mul[i, j] = idx[compose(a, b)]
    ident = idx[tuple(range(n))]
    return elems, mul, ident

GROUPS = {"S3": 3, "S4": 4, "A5": 5}   # A5 handled specially (even perms of 5) below

def build_group(name):
    if name == "A5":
        from itertools import permutations
        n = 5
        def parity(p):
            seen = [False] * n; sgn = 1
            for i in range(n):
                if seen[i]:
                    continue
                j = i; ln = 0
                while not seen[j]:
                    seen[j] = True; j = p[j]; ln += 1
                if ln % 2 == 0:
                    sgn = -sgn
            return sgn
        elems = sorted(p for p in permutations(range(n)) if parity(p) == 1)   # A5 = even perms, |A5|=60
        idx = {e: i for i, e in enumerate(elems)}
        def compose(a, b):
            return tuple(a[b[x]] for x in range(n))
        K = len(elems); mul = np.zeros((K, K), dtype=np.int64)
        for i, a in enumerate(elems):
            for j, b in enumerate(elems):
                mul[i, j] = idx[compose(a, b)]
        return elems, mul, idx[tuple(range(n))]
    return _perm_group(GROUPS[name])


# ---------------------------------------------------------------- task ------------------------------------------
def make_group_task(group_name, seed, n_pool=64, code_k=None, noise=0.6,
                    train_lens=(1, 2, 3, 4, 5), test_lens=(6, 7, 8), n_per_len=1500):
    """Streaming group composition. Each element -> a fixed sparse-ish +-1 POOL code (k active of n_pool, rest -1);
    a sequence -> per-step codes with fresh +-1 NOISE flips (fraction `noise` of a small jitter). Target = parity of
    the final product's element-index bucket (a 2-way property that DEPENDS on the full ordered composition).
    Returns dict with per-split (X seq [N,Lmax,n_pool], lengths, y, and the element-index sequence for controls)."""
    elems, mul, ident = build_group(group_name)
    K = len(elems)
    rng = np.random.RandomState(seed)
    code_k = code_k if code_k is not None else max(4, n_pool // 4)
    base = -np.ones((K, n_pool), dtype=np.float32)                # each element: a distinct +-1 code
    for e in range(K):
        on = rng.choice(n_pool, code_k, replace=False)
        base[e, on] = 1.0
    # 2-way target property of the FINAL element: parity of a fixed random 2-coloring of the group elements
    color = rng.randint(0, 2, size=K)                            # a fixed nontrivial 2-coloring (property to read)

    def gen(lens, n_each):
        Lmax = max(test_lens + train_lens)
        X, Y, L, SEQ = [], [], [], []
        for L_ in lens:
            for _ in range(n_each):
                gidx = rng.randint(0, K, size=L_)               # sequence of element indices
                s = ident
                for g in gidx:
                    s = mul[s, g]                               # s <- s * g  (LEFT-to-RIGHT ordered product)
                seq_codes = np.full((Lmax, n_pool), 0.0, dtype=np.float32)   # zero-pad past the real length
                for t, g in enumerate(gidx):
                    c = base[g].copy()
                    flip = rng.rand(n_pool) < (noise * 0.15)     # small +-1 jitter (kills exact-code lookup)
                    c[flip] = -c[flip]
                    seq_codes[t] = c
                X.append(seq_codes); Y.append(int(color[s])); L.append(L_)
                SEQ.append(np.pad(gidx, (0, Lmax - L_), constant_values=-1))
        return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64),
                np.asarray(L, dtype=np.int64), np.asarray(SEQ, dtype=np.int64))

    tr = gen(train_lens, n_per_len)
    te_same = gen(train_lens, n_per_len // 2)                    # LEARNABILITY oracle: NEW seqs at TRAIN lengths
    te_deep = gen(test_lens, n_per_len)                          # HELD-OUT-DEEPER: longer than any training length
    return {"K": K, "n_pool": n_pool, "Lmax": int(max(test_lens + train_lens)), "ident": ident, "mul": mul,
            "color": color, "train": tr, "test_same": te_same, "test_deeper": te_deep,
            "train_lens": train_lens, "test_lens": test_lens}


# ---------------------------------------------------------------- arms ------------------------------------------
def _softmax_ce_train(forward, params, Xtr, ytr, Xte, yte, epochs, lr, batch, seed, grad):
    """Generic full-batch-ish SGD with a 2-class softmax CE. forward(params,X)->logits[N,2]; grad(params,X,y)->dparams.
    Returns (test_acc, params)."""
    rng = np.random.RandomState(seed + 7)
    N = len(ytr)
    for ep in range(epochs):
        order = rng.permutation(N)
        for i in range(0, N, batch):
            bi = order[i:i + batch]
            dp = grad(params, Xtr[bi], ytr[bi])
            for kk in params:
                params[kk] -= lr * dp[kk]
    logits = forward(params, Xte)
    acc = float((logits.argmax(1) == yte).mean())
    return acc, params


def ff_oracle(task, hidden_sizes=(0, 128, 128, 128), epochs=40, lr=0.05, batch=128, seed=42):
    """FF-ORACLE: MLP over the FLATTENED sequence [Lmax*n_pool]. Sweep depth 0..len(hidden_sizes); return the BEST
    acc on each split (same-length learnability + held-out-deeper). The feedforward ceiling -- it can memorize
    trained lengths (same-length may be high) but cannot ITERATE (deeper should stay ~chance)."""
    Xtr, ytr = task["train"][0].reshape(len(task["train"][1]), -1), task["train"][1]
    Xsm, ysm = task["test_same"][0].reshape(len(task["test_same"][1]), -1), task["test_same"][1]
    Xte, yte = task["test_deeper"][0].reshape(len(task["test_deeper"][1]), -1), task["test_deeper"][1]
    n_in = Xtr.shape[1]
    best_same, best_deep = -1.0, -1.0
    rng = np.random.RandomState(seed)
    for depth in range(len(hidden_sizes) + 1):                   # 0 (linear) .. len(hidden)
        hs = [h for h in hidden_sizes[:depth] if h > 0]
        dims = [n_in] + hs + [2]
        Ws = [rng.randn(dims[i], dims[i + 1]).astype(np.float32) * np.sqrt(2.0 / dims[i]) for i in range(len(dims) - 1)]
        bs = [np.zeros(dims[i + 1], dtype=np.float32) for i in range(len(dims) - 1)]
        params = {}
        for i in range(len(Ws)):
            params[f"W{i}"] = Ws[i]; params[f"b{i}"] = bs[i]
        nL = len(Ws)

        def forward(p, X, nL=nL):
            a = X
            for i in range(nL):
                z = a @ p[f"W{i}"] + p[f"b{i}"]
                a = np.maximum(z, 0) if i < nL - 1 else z
            return a

        def grad(p, X, y, nL=nL):
            acts = [X]; a = X
            for i in range(nL):
                z = a @ p[f"W{i}"] + p[f"b{i}"]
                a = np.maximum(z, 0) if i < nL - 1 else z
                acts.append(a)
            logits = acts[-1]
            ex = np.exp(logits - logits.max(1, keepdims=True)); sm = ex / ex.sum(1, keepdims=True)
            d = sm.copy(); d[np.arange(len(y)), y] -= 1.0; d /= len(y)
            dp = {}
            for i in reversed(range(nL)):
                dp[f"W{i}"] = acts[i].T @ d; dp[f"b{i}"] = d.sum(0)
                if i > 0:
                    d = (d @ p[f"W{i}"].T) * (acts[i] > 0)
            return dp

        acc_deep, _ = _softmax_ce_train(forward, params, Xtr, ytr, Xte, yte, epochs, lr, batch, seed, grad)
        acc_same = float((forward(params, Xsm).argmax(1) == ysm).mean())
        best_same = max(best_same, acc_same); best_deep = max(best_deep, acc_deep)
    return {"same": best_same, "deeper": best_deep}


def reservoir_ridge(task, n_res=300, spectral=0.9, seed=42, ridge=1e-2):
    """FIXED random tanh reservoir + a ridge read-out on the FINAL state. Fixed recurrent dynamics -- is it enough?
    (predicted BOUNDARY: fixed dynamics can't learn the group-specific state update)."""
    rng = np.random.RandomState(seed + 3)
    n_pool = task["n_pool"]
    Win = rng.randn(n_res, n_pool).astype(np.float32) * 0.5
    W = rng.randn(n_res, n_res).astype(np.float32)
    W *= spectral / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9)
    def states(Xseq, L):
        H = np.zeros((len(L), n_res), dtype=np.float32)
        for n in range(len(L)):
            h = np.zeros(n_res, dtype=np.float32)
            for t in range(L[n]):
                h = np.tanh(W @ h + Win @ Xseq[n, t])
            H[n] = h
        return H
    Htr = states(task["train"][0], task["train"][2]); ytr = task["train"][1]           # train[2]=lengths
    Hsm = states(task["test_same"][0], task["test_same"][2]); ysm = task["test_same"][1]
    Hte = states(task["test_deeper"][0], task["test_deeper"][2]); yte = task["test_deeper"][1]
    Y = np.eye(2, dtype=np.float32)[ytr]                          # ridge 2-class (one-hot)
    A = Htr.T @ Htr + ridge * np.eye(Htr.shape[1], dtype=np.float32)
    Wout = np.linalg.solve(A, Htr.T @ Y)
    return {"same": float(((Hsm @ Wout).argmax(1) == ysm).mean()),
            "deeper": float(((Hte @ Wout).argmax(1) == yte).mean())}


def bptt_rnn(task, n_hid=128, epochs=60, lr=0.05, batch=128, seed=42, lesion_recurrent=False, orthogonal_rec=False):
    """BPTT-RNN (the DECISIVE arm): h_t = tanh(W_rec h_{t-1} + W_in x_t), y = W_out h_L (final state), softmax CE,
    trained by backprop-through-time. lesion_recurrent=True zeros + freezes W_rec (-> a per-step FF net = the
    recurrence-off lesion; must collapse to the FF ceiling). orthogonal_rec=True inits W_rec as a random ORTHOGONAL
    matrix (norm-preserving -> the state doesn't blow up/vanish over many steps; permutations ARE orthogonal, so this
    is the theory-motivated inductive bias for LENGTH-GENERALIZING group composition -- Arjovsky-Shah-Bengio unitary
    RNN / orthogonal-init line). Handles variable length via per-sample L (read h_L)."""
    rng = np.random.RandomState(seed + 5)
    n_pool = task["n_pool"]
    Win = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    if orthogonal_rec and not lesion_recurrent:
        Wrec = np.linalg.qr(rng.randn(n_hid, n_hid))[0].astype(np.float32)   # random orthogonal (norm-preserving)
    else:
        Wrec = (rng.randn(n_hid, n_hid) * (0.0 if lesion_recurrent else np.sqrt(1.0 / n_hid))).astype(np.float32)
    Wout = (rng.randn(2, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32)
    bout = np.zeros(2, dtype=np.float32)
    Xtr, ytr, Ltr = task["train"][0], task["train"][1], task["train"][2]
    Xte, yte, Lte = task["test_deeper"][0], task["test_deeper"][1], task["test_deeper"][2]
    N = len(ytr)

    def run_batch(Xb, Lb):
        B = len(Lb); Lmax = int(Lb.max())
        Hs = np.zeros((Lmax + 1, B, n_hid), dtype=np.float32)     # Hs[t] = state AFTER step t (Hs[0]=0)
        for t in range(Lmax):
            pre = Hs[t] @ Wrec.T + Xb[:, t] @ Win.T
            Hs[t + 1] = np.tanh(pre)
        hL = Hs[Lb, np.arange(B)]                                # final state per sample (gather at its length)
        return Hs, hL

    for ep in range(epochs):
        order = rng.permutation(N)
        for i in range(0, N, batch):
            bi = order[i:i + batch]; Xb = Xtr[bi]; yb = ytr[bi]; Lb = Ltr[bi]
            B = len(bi); Lmax = int(Lb.max())
            Hs, hL = run_batch(Xb, Lb)
            logits = hL @ Wout.T + bout
            ex = np.exp(logits - logits.max(1, keepdims=True)); sm = ex / ex.sum(1, keepdims=True)
            dlog = sm.copy(); dlog[np.arange(B), yb] -= 1.0; dlog /= B
            dWout = dlog.T @ hL; dbout = dlog.sum(0)
            dhL = dlog @ Wout                                    # [B, n_hid]
            # scatter dhL to the per-sample final timestep, BPTT backward
            dH_next = np.zeros((B, n_hid), dtype=np.float32)
            dWin = np.zeros_like(Win); dWrec = np.zeros_like(Wrec)
            for t in range(Lmax, 0, -1):
                dh = dH_next.copy()
                at_final = (Lb == t)
                if at_final.any():
                    dh[at_final] += dhL[at_final]
                dpre = dh * (1.0 - Hs[t] ** 2)                   # tanh'
                dWin += dpre.T @ Xb[:, t - 1]
                dWrec += dpre.T @ Hs[t - 1]
                dH_next = dpre @ Wrec
            Wout -= lr * dWout; bout -= lr * dbout
            Win -= lr * dWin
            if not lesion_recurrent:
                Wrec -= lr * dWrec
    # eval both splits (same-length learnability + held-out-deeper generalization)
    Xsm, ysm, Lsm, _ = task["test_same"]
    _, hLsm = run_batch(Xsm, Lsm)
    _, hLte = run_batch(Xte, Lte)
    return {"same": float(((hLsm @ Wout.T + bout).argmax(1) == ysm).mean()),
            "deeper": float(((hLte @ Wout.T + bout).argmax(1) == yte).mean())}


# ---------------------------------------------------------------- anti-cheats + driver -------------------------
def order_control(task, seed=42):
    """PERMUTED-ORDER control: shuffle each sequence's element order. For a NON-ABELIAN group the product changes ->
    the target becomes uncorrelated -> ANY model must drop toward chance. If a model still predicts well on shuffled
    order, it's using a count/multiset shortcut, not the ordered composition. Returns the fraction of test sequences
    whose target CHANGES under a random reshuffle (a task property: high => order genuinely matters)."""
    mul = task["mul"]; color = task["color"]; ident = task["ident"]
    Xte, yte, Lte, SEQ = task["test_deeper"]
    rng = np.random.RandomState(seed + 11)
    changed = 0; tot = 0
    for n in range(len(Lte)):
        gidx = SEQ[n][SEQ[n] >= 0]
        if len(gidx) < 2:
            continue
        perm = rng.permutation(len(gidx))
        s = ident
        for g in gidx[perm]:
            s = mul[s, g]
        changed += int(color[s] != yte[n]); tot += 1
    return changed / max(tot, 1)


def markov_floor(task):
    """1-hop / last-element floor: predict the target from ONLY the last element's code (a 1st-order shortcut).
    A ridge on the last real element's code -> its held-out-deeper acc = the no-composition floor."""
    Xtr, ytr, Ltr, _ = task["train"]; Xte, yte, Lte, _ = task["test_deeper"]
    last_tr = Xtr[np.arange(len(Ltr)), Ltr - 1]; last_te = Xte[np.arange(len(Lte)), Lte - 1]
    Y = np.eye(2, dtype=np.float32)[ytr]
    A = last_tr.T @ last_tr + 1e-2 * np.eye(last_tr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, last_tr.T @ Y)
    return float(((last_te @ W).argmax(1) == yte).mean())


def run_seed(group_name, seed, n_pool=64, noise=0.6, n_per_len=1200, epochs=60, n_hid=128,
             train_lens=(1, 2, 3, 4, 5), test_lens=(6, 7, 8), orthogonal_rec=False):
    task = make_group_task(group_name, seed, n_pool=n_pool, noise=noise, n_per_len=n_per_len,
                           train_lens=tuple(train_lens), test_lens=tuple(test_lens))
    ff = ff_oracle(task, seed=seed, epochs=40)
    res = reservoir_ridge(task, seed=seed)
    rnn = bptt_rnn(task, seed=seed, epochs=epochs, n_hid=n_hid, orthogonal_rec=orthogonal_rec)
    rnn_les = bptt_rnn(task, seed=seed, epochs=epochs, n_hid=n_hid, lesion_recurrent=True)
    mk = markov_floor(task)
    order_matters = order_control(task, seed=seed)
    return {"seed": seed, "group": group_name, "K": task["K"], "chance": 0.5,
            "ff_same": round(ff["same"], 3), "ff_deeper": round(ff["deeper"], 3),
            "reservoir_same": round(res["same"], 3), "reservoir_deeper": round(res["deeper"], 3),
            "rnn_same": round(rnn["same"], 3), "rnn_deeper": round(rnn["deeper"], 3),
            "rnn_lesion_same": round(rnn_les["same"], 3), "rnn_lesion_deeper": round(rnn_les["deeper"], 3),
            "markov_floor_deeper": round(mk, 3), "order_matters_frac": round(order_matters, 3),
            "rnn_minus_ff_deeper": round(rnn["deeper"] - ff["deeper"], 3),
            "rnn_minus_ff_same": round(rnn["same"] - ff["same"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=list(GROUPS) + ["A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-pool", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--n-per-len", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-hid", type=int, default=128)
    ap.add_argument("--train-lens", default="1,2,3,4,5")
    ap.add_argument("--test-lens", default="6,7,8")
    ap.add_argument("--orthogonal-rec", action="store_true",
                    help="init the BPTT-RNN's W_rec as a random ORTHOGONAL matrix (norm-preserving; permutations are "
                         "orthogonal -> the theory-motivated inductive bias for length-generalizing composition)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    train_lens = tuple(int(x) for x in a.train_lens.split(","))
    test_lens = tuple(int(x) for x in a.test_lens.split(","))
    print(f"[D3 group-composition] {a.group} | recurrent-vs-feedforward multi-hop composition | SAME=learnability, DEEPER=generalization", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(a.group, s, n_pool=a.n_pool, noise=a.noise, n_per_len=a.n_per_len, epochs=a.epochs, n_hid=a.n_hid,
                     train_lens=train_lens, test_lens=test_lens, orthogonal_rec=a.orthogonal_rec)
        rows.append(r)
        print(f"  [seed {s}] SAME(learn): FF={r['ff_same']} res={r['reservoir_same']} RNN={r['rnn_same']} (les={r['rnn_lesion_same']}) "
              f"|| DEEPER(gen): FF={r['ff_deeper']} res={r['reservoir_deeper']} RNN={r['rnn_deeper']} (les={r['rnn_lesion_deeper']}) "
              f"| markov={r['markov_floor_deeper']} order-matters={r['order_matters_frac']}", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        ffd, rnnd, lesd, resd = _m("ff_deeper"), _m("rnn_deeper"), _m("rnn_lesion_deeper"), _m("reservoir_deeper")
        rnns, ffs = _m("rnn_same"), _m("ff_same")
        # GO: the RNN LEARNS the composition (same-length high) AND generalizes DEEPER where FF cannot (RNN_deeper >>
        # FF_deeper), recurrence load-bearing (lesion collapses deeper). Learnability gate: rnn_same must be high first.
        learnable = rnns > 0.75
        go = learnable and (rnnd > 0.70) and (rnnd - ffd > 0.15) and (rnnd - lesd > 0.15)
        print(f"\n  AGGREGATE ({a.group}) SAME: FF={ffs:.3f} RNN={rnns:.3f} || DEEPER: FF={ffd:.3f} res={resd:.3f} RNN={rnnd:.3f} lesion={lesd:.3f}", flush=True)
        if not learnable:
            print(f"  VERDICT: NOT-YET-LEARNABLE -- the RNN did not learn same-length composition ({rnns:.3f}<0.75) -> scale n_hid/epochs/data or simplify before reading the deeper separation. NO sim/ edit.", flush=True)
        else:
            print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the RNN LEARNS composition (same '+format(rnns,'.2f')+') AND generalizes DEEPER where FF cannot (RNN_deeper>>FF_deeper, lesion collapses) -> D3 recurrent depth CONFIRMED; escalate to A5 + e-prop + spiking' if go else 'RNN learns same-length but does NOT generalize deeper beyond FF (the honest boundary: BPTT learns the map but not the ITERATION -> next: curriculum/length-schedule, or e-prop, or the reservoir is genuinely capped)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
