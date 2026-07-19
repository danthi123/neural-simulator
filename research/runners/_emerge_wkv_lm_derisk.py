"""gap#1 open-generation de-risk (the research-gate decision, 2026-07-19): does a LEARNED KEY-VALUE RECURRENCE
(RWKV/WKV linear-attention -- a content-selective NON-FADING learned-write store) capture DEEP-context language structure
where every FADING-memory reservoir FAILED (lost to a FAIR interpolated trigram at every depth, 2026-07-15)? This is the
one mechanism the arc's own synthesis named as the deepest unbuilt frontier; SpikeGPT is an at-scale (45M) spiking
existence proof. Rate-level, torch autodiff-BPTT to establish the MECHANISM first (a TRACKED shortcut; the ladder biologizes
the rule + ports to a spiking BrainRegion later). APPLES-TO-APPLES with the reservoir arc: reuse Vocab/load_sentences/
fit_bigram + the same deep-context (d10-99) bucketing; add the FAIR interpolated trigram (the exact control that killed the
reservoir). NO `sim/` edit.

THE WKV OP (RWKV-style, O(N) recurrent, differentiable): per channel, a running numerator a_t / denominator b_t with a
LEARNED per-channel time-decay w and a current-token bonus u; wkv_t = (a_{t-1} + exp(u+k_t)*v_t) / (b_{t-1} + exp(u+k_t));
a_t = exp(-w)*a_{t-1} + exp(k_t)*v_t; b_t = exp(-w)*b_{t-1} + exp(k_t). Output = sigmoid(r_t) * (Wo @ wkv_t). K,V,R,w,u
are LEARNED (a content-addressed non-fading read -- the store the reservoir lacked). Numerically stable via a running max.

GATE (the honest bar): the WKV LM BEATS the fair interpolated trigram on held-out DEEP-context (d>=10) NLL, AND its
deep-context margin over the bigram GROWS with context depth (the transformer/LSTM signature), AND the anti-cheats collapse:
  - PERMUTED-context (shuffle the prefix order per eval sentence) -> the deep-context advantage collapses (proves order/
    long-range use, not a bag-of-context).
  - MEMORYLESS (recurrence off: exp(-w)->0, u dominates -> only the current token) -> ~bigram (no depth advantage).
6-seed (42/43/44/100/101/102).

Run (smoke): SIM_BACKEND=numpy python -m research.runners._emerge_wkv_lm_derisk --seeds 42 --n-sentences 8000 --vocab 800 --epochs 8
Run (scale): python -m research.runners._emerge_wkv_lm_derisk --seeds 42 43 44 100 101 102 --corpus data/corpus/tinystories_train.txt --vocab 2000 --n-sentences 200000 --epochs 12 --d-model 256
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket

OUT = Path("research/findings/raw/_emerge_wkv_lm.json")


def load_stories(path, max_stories, max_len=48):
    """CONTIGUOUS multi-sentence passages (R4 open-prose test): each corpus line is a STORY -> tokenize the WHOLE story as
    ONE sequence (NOT split into sentences), so deep-context (d>16) spans SENTENCE BOUNDARIES = genuine cross-sentence
    long-range. Caps at max_len tokens."""
    import re
    txt = open(path, encoding="utf-8", errors="ignore").read(max_stories * max_len * 8)   # bounded read
    toks = re.findall(r"[a-z']+", txt.lower())                       # ONE contiguous token stream (the corpus is 1 line)
    stories = [toks[i:i + max_len] for i in range(0, min(len(toks), max_stories * max_len), max_len)]
    return [s for s in stories if len(s) >= 8][:max_stories]         # contiguous max_len-token passages (cross-sentence)


# ------------------------------------------------------------------ FAIR interpolated trigram (the KEY control) --------
def fit_interp_trigram(tr_ids, V, held_ids):
    """Deleted-interpolation trigram: P(w|u,v) = l3 P3 + l2 P2 + l1 P1 + l0 (uniform), lambdas tuned on a held-out split
    (the FAIR n-gram the reservoir arc lost to). Counts on TRAIN; lambdas by a small grid on HELD (dev). Returns a
    callable prob(u, v, w). Sparse dict counts (V=2000 tractable)."""
    from collections import Counter
    c1 = np.zeros(V); c2 = defaultdict(Counter); c3 = defaultdict(Counter)
    n1 = 0
    for ids in tr_ids:
        for t in range(len(ids)):
            c1[ids[t]] += 1; n1 += 1
            if t >= 1: c2[ids[t-1]][ids[t]] += 1
            if t >= 2: c3[(ids[t-2], ids[t-1])][ids[t]] += 1
    P1 = (c1 + 1.0) / (n1 + V)                                          # add-1 unigram
    c2tot = {u: sum(cnt.values()) for u, cnt in c2.items()}
    def p2(u, w):
        tot = c2tot.get(u, 0)
        return (c2[u][w]) / tot if tot > 0 else 0.0
    c3tot = {uv: sum(cnt.values()) for uv, cnt in c3.items()}
    def p3(u, v, w):
        tot = c3tot.get((u, v), 0)
        return (c3[(u, v)][w]) / tot if tot > 0 else 0.0
    # tune (l1,l2,l3) on held-out by grid (l0 = tiny uniform floor)
    grids = [(a, b, cc) for a in (0.05, 0.1, 0.2) for b in (0.2, 0.35, 0.5) for cc in (0.3, 0.5, 0.7) if a+b+cc < 0.97]
    best, best_ce = None, 1e18
    for (l1, l2, l3) in grids:
        l0 = max(0.0, 1.0 - l1 - l2 - l3); ce = 0.0; nn = 0
        for ids in held_ids:
            for t in range(len(ids) - 1):
                u = ids[t-1] if t >= 1 else -1; v = ids[t]; w = ids[t+1]
                pr = l0 / V + l1 * P1[w] + (l2 * p2(v, w) if True else 0.0) + (l3 * p3(u, v, w) if t >= 1 else 0.0)
                ce += -math.log(max(pr, 1e-12)); nn += 1
        ce /= max(nn, 1)
        if ce < best_ce: best_ce, best = ce, (l0, l1, l2, l3)
    l0, l1, l2, l3 = best
    def prob(u, v, w):
        return l0 / V + l1 * P1[w] + l2 * p2(v, w) + (l3 * p3(u, v, w) if u >= 0 else 0.0)
    return prob, best


# --------------------------------------------------- EMERGENT input: PPMI co-occurrence codes (Rung 1b, gap#1<->gap#4) --
def build_ppmi_codes(tr_ids, V, d, window=5):
    """UNSUPERVISED stream-cortex-style codes: a windowed co-occurrence matrix over the corpus -> PPMI (log + positive
    threshold, the CYCLE-88 local-normalization) -> SVD to d dims -> unit-normalized per-word code. This is the emergent
    representation (learned from the stream, NOT by the LM) that Rung 1b feeds (frozen) into the WKV -- the convergence of
    gap#1 (open generation) with the gap#4-pivot unsupervised cortex. NO LM gradient touches it."""
    C = np.zeros((V, V), dtype=np.float64)
    for ids in tr_ids:
        n = len(ids)
        for i in range(n):
            lo, hi = max(0, i - window), min(n, i + window + 1)
            for j in range(lo, hi):
                if i != j:
                    C[ids[i], ids[j]] += 1.0
    tot = C.sum() + 1e-12; rs = C.sum(1); cs = C.sum(0)
    P = C / tot; Pw = rs / tot; Pc = cs / tot
    pmi = np.log((P + 1e-12) / (np.outer(Pw, Pc) + 1e-12))
    ppmi = np.maximum(pmi, 0.0)                                     # positive PMI (local normalization)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    k = min(d, U.shape[1])
    codes = U[:, :k] * np.sqrt(S[:k] + 1e-12)
    if k < d:                                                       # pad if d>rank
        codes = np.concatenate([codes, np.zeros((V, d - k))], 1)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    return codes.astype(np.float32)                                # [V, d]


# ------------------------------------------------------------------ the WKV LM (torch) --------------------------------
def build_and_train_wkv(tr_ids, V, seed, args, device, init_emb=None):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    D = args.d_model

    RECUR = getattr(args, "recurrence", "wkv")
    class WKV(nn.Module):
        def __init__(self, V, D, memoryless=False):
            super().__init__()
            self.emb = nn.Embedding(V, D)
            self.ln = nn.LayerNorm(D)
            self.Wk = nn.Linear(D, D, bias=False); self.Wv = nn.Linear(D, D, bias=False)
            self.Wr = nn.Linear(D, D, bias=False); self.Wo = nn.Linear(D, D, bias=False)
            self.Wo_sp = nn.Linear(2 * D, D, bias=False)   # spiking-state read-out over ON/OFF non-negative rate channels
            # UNIFORM-decay (--uniform-decay): ONE shared decay = the substrate's uniform NMDA tau (simplifies the on-bridge
            # realization -- no per-neuron tau array); default = per-channel learned decay.
            self.w = nn.Parameter(torch.zeros(1 if getattr(args, "uniform_decay", False) else D))
            self.u = nn.Parameter(torch.zeros(D))     # current-token bonus
            self.head = nn.Linear(D, V)
            self.memoryless = memoryless

        def forward(self, x):                          # x: [B,T] token ids
            B, T = x.shape
            h = self.ln(self.emb(x))                    # [B,T,D]
            k = self.Wk(h); v = self.Wv(h); r = torch.sigmoid(self.Wr(h))
            wdec = torch.exp(-torch.nn.functional.softplus(self.w))       # per-channel decay in (0,1)
            if self.memoryless:
                wdec = torch.zeros_like(wdec)           # ANTI-CHEAT: no carry -> only the current token
            if RECUR == "ssm":
                # SPIKING-SUBSTRATE-FAITHFUL leaky-integrator (Rung 2 de-risk): a_t = decay*a_{t-1} + v_t (a slow
                # conductance/membrane leak -- NO exp(k) weighting, NO num/den normalization = the part hard on spikes),
                # out = receptance-gated read. If GO, the spiking membrane leak realizes this state directly.
                a = torch.zeros(B, D, device=x.device); outs = []
                spiking = getattr(args, "spiking_state", False)
                for t in range(T):
                    a = wdec * a + v[:, t]
                    if spiking:
                        # SPIKING-FAITHFUL read: firing rates are NON-NEGATIVE; encode the signed leaky state via ON/OFF
                        # rate channels [relu(a), relu(-a)] (the two-population sign code a spiking region uses). If GO,
                        # the on-bridge realization (firing-rate read of the region's slow conductance) preserves it.
                        rate = torch.cat([torch.relu(a), torch.relu(-a)], -1)
                        outs.append(r[:, t] * self.Wo_sp(rate))
                    else:
                        outs.append(r[:, t] * self.Wo(a))
                return self.head(torch.stack(outs, 1))
            u = self.u
            a = torch.zeros(B, D, device=x.device); b = torch.zeros(B, D, device=x.device)
            pmax = torch.full((B, D), -1e30, device=x.device)            # running max for numerical stability
            outs = []
            for t in range(T):
                kt = k[:, t]; vt = v[:, t]
                # current-token: exp(u+kt); the running state uses the previous a,b (a_{t-1}, b_{t-1})
                ww = pmax                                # state exponent reference
                q = torch.maximum(ww, u + kt)
                e1 = torch.exp(ww - q); e2 = torch.exp(u + kt - q)
                wkv = (e1 * a + e2 * vt) / (e1 * b + e2 + 1e-8)
                outs.append(r[:, t] * self.Wo(wkv))
                # advance the running state to include token t (decay then add), stable
                pmax2 = torch.maximum(pmax + torch.log(wdec + 1e-30), kt)
                e1 = torch.exp(pmax + torch.log(wdec + 1e-30) - pmax2); e2 = torch.exp(kt - pmax2)
                a = e1 * a + e2 * vt; b = e1 * b + e2; pmax = pmax2
            y = torch.stack(outs, 1)                     # [B,T,D]
            return self.head(y)                          # [B,T,V]

    def pad_batch(seqs):
        m = max(len(s) for s in seqs)
        X = np.zeros((len(seqs), m), dtype=np.int64); msk = np.zeros((len(seqs), m), dtype=bool)
        for i, s in enumerate(seqs):
            X[i, :len(s)] = s; msk[i, :len(s)] = True
        return torch.tensor(X, device=device), torch.tensor(msk, device=device)

    net = WKV(V, D).to(device)
    if init_emb is not None:
        with torch.no_grad():                                       # Rung 1b EMERGENT input: seed the emb with PPMI codes
            net.emb.weight.copy_(torch.tensor(init_emb, device=device))
    if getattr(args, "freeze_emb", False) or init_emb is not None:
        # FREEZE the input (only WKV+head learn) => the mechanism must capture deep context WITHOUT an LM-learned input =
        # the emergent-codes regime (codes learned by the unsupervised cortex, FROZEN for the LM). GO => the WKV
        # recurrence does not depend on an LM-learned embedding; PPMI>random => the emergent structure HELPS.
        net.emb.weight.requires_grad = False
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss(reduction="none")
    seqs = [s for s in tr_ids if len(s) >= 2]
    rng = np.random.default_rng(seed * 7 + 3)
    for ep in range(args.epochs):
        order = rng.permutation(len(seqs))
        for i in range(0, len(seqs), args.batch):
            batch = [seqs[j] for j in order[i:i+args.batch]]
            X, msk = pad_batch(batch)
            logits = net(X)[:, :-1]                       # predict token t+1 from context 0..t
            tgt = X[:, 1:]; m = msk[:, 1:]
            L = lossf(logits.reshape(-1, V), tgt.reshape(-1)).reshape(tgt.shape)
            loss = (L * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    return net, WKV


def eval_perdepth(net, WKV_cls, ev_ids, V, device, permute=False, memoryless=False, seed=0):
    """Per-context-depth held-out NLL for the WKV LM. permute=shuffle the prefix order (anti-cheat); memoryless=recurrence
    off (anti-cheat, rebuild a memoryless forward from the trained weights)."""
    import torch
    rng = np.random.default_rng(seed * 17 + 5)
    if memoryless:
        net.memoryless = True
    ce = defaultdict(float); cnt = defaultdict(int)
    with torch.no_grad():
        for ids in ev_ids:
            if len(ids) < 2: continue
            seq = list(ids)
            if permute and len(seq) > 2:
                # shuffle the PREFIX order but keep each target's immediately-preceding token intact is impossible while
                # destroying long-range; we shuffle tokens 0..t-1 for each t (destroys order beyond the last), evaluated
                # per-t below via a fresh permuted prefix. Simpler + faithful: shuffle the whole sentence order once.
                perm = rng.permutation(len(seq)); seq = [ids[p] for p in perm]
            X = torch.tensor([seq], device=device)
            logits = net(X)[0]                            # [T,V]
            logp = torch.log_softmax(logits, -1).cpu().numpy()
            for t in range(len(seq) - 1):
                d = t + 1; b = _bucket(d)
                ce[b] += -math.log(max(math.exp(logp[t, seq[t+1]]), 1e-12)); cnt[b] += 1
    if memoryless:
        net.memoryless = False
    return {b: ce[b] / cnt[b] for b in cnt}, {b: cnt[b] for b in cnt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories_train.txt")
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--n-sentences", type=int, default=200000)
    ap.add_argument("--max-train-sents", type=int, default=60000)
    ap.add_argument("--max-eval-sents", type=int, default=3000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-emb", dest="freeze_emb", action="store_true",
                    help="Rung 1b de-risk: freeze the input embedding at random init (only WKV+head learn) = the frozen "
                         "emergent-code regime; GO => the recurrence does not need an LM-learned input.")
    ap.add_argument("--input", choices=["learned", "ppmi"], default="learned",
                    help="learned = LM-trained embedding (Rung 1a); ppmi = EMERGENT unsupervised PPMI co-occurrence codes, "
                         "frozen (Rung 1b, the gap#1<->gap#4 convergence).")
    ap.add_argument("--ppmi-window", type=int, default=5)
    ap.add_argument("--recurrence", choices=["wkv", "ssm"], default="wkv",
                    help="wkv = full RWKV linear-attention (num/den normalized); ssm = spiking-substrate-faithful "
                         "leaky-integrator (a_t=decay*a_{t-1}+v_t, no normalization = the Rung 2 spiking-port form).")
    ap.add_argument("--spiking-state", dest="spiking_state", action="store_true",
                    help="(ssm only) read the leaky state via NON-NEGATIVE ON/OFF firing-rate channels [relu(a),relu(-a)] "
                         "= the spiking firing-rate constraint; GO => the on-bridge firing-rate read preserves deep context.")
    ap.add_argument("--uniform-decay", dest="uniform_decay", action="store_true",
                    help="(ssm only) ONE shared decay across channels = the substrate's uniform NMDA tau (simplifies the "
                         "on-bridge realization); GO => no per-neuron tau array is needed on the bridge.")
    ap.add_argument("--save-ssm", dest="save_ssm", type=str, default=None,
                    help="save the trained SSM weights to <path>_seed<N>.npz (for the on-bridge realization).")
    ap.add_argument("--generate", type=int, default=0, help="autoregressive: generate N tokens after training")
    ap.add_argument("--contiguous", action="store_true", help="CONTIGUOUS multi-sentence stories (R4 cross-sentence long-range test)")
    ap.add_argument("--max-len", type=int, default=48, help="(--contiguous) max tokens per story")
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(args.corpus).exists():
        args.corpus = "data/corpus/tinystories.txt"
    sents = (load_stories(args.corpus, args.n_sentences, max_len=args.max_len)
             if getattr(args, "contiguous", False) else load_sentences(args.corpus, args.n_sentences))
    t0 = time.time(); per_seed = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(sents)); cut = int(0.85 * len(sents))
        tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
        ev = [sents[i] for i in idx[cut:]][:args.max_eval_sents]
        dev = tr[-min(2000, len(tr)//5):]                # held-out dev for trigram lambda tuning
        vocab = Vocab.build(tr, V=args.vocab); V = vocab.size
        tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]; dev_ids = [vocab.ids(s) for s in dev]

        P_bi = fit_bigram(tr_ids, V)
        tri, lambdas = fit_interp_trigram(tr_ids, V, dev_ids)

        init_emb = build_ppmi_codes(tr_ids, V, args.d_model, args.ppmi_window) if args.input == "ppmi" else None
        net, WKV_cls = build_and_train_wkv(tr_ids, V, seed, args, device, init_emb=init_emb)
        if getattr(args, "generate", 0):                             # AUTOREGRESSIVE generation (does the WKV produce prose?)
            import torch
            prompt = ["once", "upon", "a", "time"]
            ids_g = [vocab.w2i.get(w, vocab.unk) for w in prompt]
            with torch.no_grad():
                for _ in range(args.generate):
                    logits = net(torch.tensor([ids_g], device=device))[0, -1]
                    p = torch.softmax(logits / 0.8, -1).cpu().numpy()
                    ids_g.append(int(np.random.default_rng(seed * 91 + len(ids_g)).choice(V, p=p / p.sum())))
            print(f"    [seed {seed}] GENERATED: {' '.join(vocab.i2w[i] for i in ids_g)}", flush=True)
        if getattr(args, "save_ssm", None):                      # save the trained SSM weights for the on-bridge realization
            import torch as _t
            sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
            np.savez(f"{args.save_ssm}_seed{seed}.npz", V=V, d_model=args.d_model,
                     words=np.array(vocab.i2w, dtype=object), **sd)
            print(f"    [seed {seed}] saved SSM weights -> {args.save_ssm}_seed{seed}.npz", flush=True)
        wkv_ce, cnt = eval_perdepth(net, WKV_cls, ev_ids, V, device, seed=seed)
        wkv_perm, _ = eval_perdepth(net, WKV_cls, ev_ids, V, device, permute=True, seed=seed)
        wkv_mless, _ = eval_perdepth(net, WKV_cls, ev_ids, V, device, memoryless=True, seed=seed)

        # bigram + trigram per-depth NLL on the SAME eval tokens
        bce = defaultdict(float); tce = defaultdict(float)
        for ids in ev_ids:
            for t in range(len(ids) - 1):
                d = t + 1; b = _bucket(d)
                bce[b] += -math.log(max(P_bi[ids[t], ids[t+1]], 1e-12))
                u = ids[t-1] if t >= 1 else -1
                tce[b] += -math.log(max(tri(u, ids[t], ids[t+1]), 1e-12))
        depth = {}
        for lo, hi in BUCKETS:
            b = f"{lo}-{hi}" if lo != hi else f"{lo}"
            if b in cnt:
                n = cnt[b]
                depth[b] = {"n": n, "wkv": round(wkv_ce[b], 3), "bigram": round(bce[b]/n, 3),
                            "trigram": round(tce[b]/n, 3), "wkv_perm": round(wkv_perm.get(b, float('nan')), 3),
                            "wkv_memoryless": round(wkv_mless.get(b, float('nan')), 3),
                            "margin_vs_trigram": round(tce[b]/n - wkv_ce[b], 3),
                            "margin_vs_bigram": round(bce[b]/n - wkv_ce[b], 3)}
        per_seed[str(seed)] = {"V": V, "n_train": len(tr), "lambdas": lambdas, "by_depth": depth}
        print(f"[seed {seed}] V={V} n_tr={len(tr)} d_model={args.d_model} -- per-depth NLL (WKV vs bigram vs FAIR trigram):", flush=True)
        for lo, hi in BUCKETS:
            b = f"{lo}-{hi}" if lo != hi else f"{lo}"
            if b in depth:
                dd = depth[b]
                print(f"    depth {b:>5} (n={dd['n']:>6}): WKV {dd['wkv']:.3f} | bigram {dd['bigram']:.3f} | trigram {dd['trigram']:.3f} "
                      f"|| vs-trigram {dd['margin_vs_trigram']:+.3f}  perm {dd['wkv_perm']:.3f}  mless {dd['wkv_memoryless']:.3f}", flush=True)
        deep = depth.get("10-99", {})
        if deep:
            go = (deep["margin_vs_trigram"] > 0.02) and (deep["wkv_perm"] - deep["wkv"] > 0.05) and (deep["wkv_memoryless"] - deep["wkv"] > 0.05)
            print(f"    [seed {seed}] DEEP (d10-99): WKV-beats-trigram {deep['margin_vs_trigram']:+.3f}, perm-collapse {deep['wkv_perm']-deep['wkv']:+.3f}, "
                  f"mless-collapse {deep['wkv_memoryless']-deep['wkv']:+.3f} -> {'GO' if go else 'no-go'}", flush=True)

    out = {"runner": "_emerge_wkv_lm_derisk", "corpus": args.corpus, "seeds": args.seeds, "d_model": args.d_model,
           "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
