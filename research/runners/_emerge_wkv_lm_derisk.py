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
import argparse, hashlib, json, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket
from sim.bpe_tokenizer import BPETokenizer

DEFAULT_BPE_PATH = "bridges/wkv_ckpt/wkv_bpe8k.json"


class _BPEVocabAdapter:
    """Subword-tokenizer swap (additive, --tokenizer bpe): wraps a loaded BPETokenizer with the SAME interface the
    trainer already uses off `Vocab` (.ids/.i2w/.w2i/.unk/.size) so the WORD path below is untouched byte-for-byte
    and every downstream consumer (fit_bigram/fit_interp_trigram/build_and_train_wkv/--save-ssm/--generate) is a
    pure drop-in -- no other call site changes. `.ids(s)` re-joins the already-tokenized word list and re-splits
    it through the BPE merges (the tokenizer's own .encode contract), so 0% hard-OOV replaces the word vocab's
    <unk>-riddled top-K."""
    def __init__(self, bpe: BPETokenizer):
        self.bpe = bpe
        self.i2w = list(bpe.vocab)
        self.w2i = {t: i for i, t in enumerate(bpe.vocab)}
        self.unk = self.w2i.get("<UNK>", 0)
        self.size = bpe.vocab_size
        self._wcache = {}   # per-word BPE id-list cache (SPEED, result-preserving; see .ids)

    def ids(self, s):
        # SPEED (additive, RESULT-PRESERVING): byte-identical to `self.bpe.encode(" ".join(s))`, which does
        # `for w in text.split(): for sym in _encode_word(w): ids.append(_sym_to_id.get(sym, 0))`. _encode_word is
        # O(n_merges) PER WORD with no cache, so a repetitive corpus (Wikipedia) re-pays it for every occurrence.
        # We memoize the per-word id-list (a pure function of w) -> ~6.6x faster tokenization (verified IDENTICAL
        # output), which matters because the trainer re-tokenizes tr/ev/dev per seed. NO behavior change.
        wc = self._wcache; get = self.bpe._sym_to_id.get; enc = self.bpe._encode_word
        out = []
        for w in (" ".join(s)).split():
            c = wc.get(w)
            if c is None:
                c = [get(sym, 0) for sym in enc(w)]
                wc[w] = c
            out.extend(c)
        return out

OUT = Path("research/findings/raw/_emerge_wkv_lm.json")
TOKCACHE_DIR = Path("data/corpus/.tokcache")   # --tok-cache: cross-RUN (cross-process) BPE token-id cache, see below


def _tokcache_key(args):
    """Content-addressed cache key for the SEED-INDEPENDENT per-sentence BPE token-id lists (SPEED, --tok-cache,
    BPE mode only): `vocab.ids(s)` depends only on the sentence + the loaded tokenizer, never on the seed, so the
    whole tokenized corpus can be computed once and shared across every seed AND every process. EVERYTHING that
    can change the resulting `sents` list or the tokenizer's output MUST be in this key, or a stale disk cache
    silently poisons every downstream NLL (the exact failure mode this helper exists to prevent)."""
    corpus_p = Path(args.corpus)
    try:
        cst = corpus_p.stat(); corpus_size, corpus_mtime = cst.st_size, cst.st_mtime
    except OSError:
        corpus_size, corpus_mtime = -1, -1.0
    bpe_p = Path(args.bpe_path)
    try:
        bst = bpe_p.stat(); bpe_size, bpe_mtime = bst.st_size, bst.st_mtime
    except OSError:
        bpe_size, bpe_mtime = -1, -1.0
    key_obj = {
        "corpus": str(corpus_p.resolve()), "corpus_size": corpus_size, "corpus_mtime": corpus_mtime,
        "tokenizer": args.tokenizer, "bpe_path": str(bpe_p.resolve()), "bpe_size": bpe_size, "bpe_mtime": bpe_mtime,
        "n_sentences": args.n_sentences, "contiguous": bool(getattr(args, "contiguous", False)),
        "max_len": args.max_len,
    }
    return hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()[:24]


def _tokcache_load(key):
    """Disk-cache HIT path: load the ragged per-sentence BPE token-id lists saved by `_tokcache_save`. Returns
    None on any miss/corruption -- a bad or absent cache must degrade to re-tokenizing, never crash the run."""
    p = TOKCACHE_DIR / f"{key}.npz"
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as d:
            concat, offsets = d["ids_concat"], d["offsets"]
            return [concat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    except Exception as e:
        print(f"    [tok-cache] load failed ({e}) -- re-tokenizing", flush=True)
        return None


def _tokcache_save(key, sents_ids):
    """Persist the ragged per-sentence token-id lists as one concatenated int32 array + int64 offsets (an exact,
    lossless round-trip of a list-of-lists-of-int). Atomic (temp file + os.replace) so a killed run never leaves
    a corrupt/partial cache entry for the next run to (silently) load."""
    TOKCACHE_DIR.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(sents_ids) + 1, dtype=np.int64)
    for i, ids in enumerate(sents_ids):
        offsets[i + 1] = offsets[i] + len(ids)
    concat = np.empty(int(offsets[-1]), dtype=np.int32)
    pos = 0
    for ids in sents_ids:
        n = len(ids)
        if n:
            concat[pos:pos + n] = ids
        pos += n
    tmp = TOKCACHE_DIR / f".tmp.{key}.{os.getpid()}.npz"
    np.savez(tmp, ids_concat=concat, offsets=offsets)
    os.replace(tmp, TOKCACHE_DIR / f"{key}.npz")


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

    class WkvLayer(nn.Module):
        """DEPTH LEVER (2026-07-21, gap#1): one STACKABLE pre-norm residual WKV block, used for layers >=1 (the deeper
        layers on top of the base single block). Its recurrence is BYTE-FOR-BYTE the baseline 'wkv' branch (the num/den
        normalized RWKV linear-attention, exactly WKV.forward's default-branch loop), operating on a pre-norm'd input:
        delta = block(LN(h)); the caller does the residual add h = h + delta. Each layer owns its own Wk/Wv/Wr/Wo + w/u."""
        def __init__(self, D, uniform_decay):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            self.Wk = nn.Linear(D, D, bias=False); self.Wv = nn.Linear(D, D, bias=False)
            self.Wr = nn.Linear(D, D, bias=False); self.Wo = nn.Linear(D, D, bias=False)
            self.w = nn.Parameter(torch.zeros(1 if uniform_decay else D))
            self.u = nn.Parameter(torch.zeros(D))

        def forward(self, h):                          # h: [B,T,D] -> delta [B,T,D] (pre-norm residual block)
            B, T, D = h.shape
            z = self.ln(h)
            k = self.Wk(z); v = self.Wv(z); r = torch.sigmoid(self.Wr(z))
            wdec = torch.exp(-torch.nn.functional.softplus(self.w)); u = self.u
            a = torch.zeros(B, D, device=h.device); b = torch.zeros(B, D, device=h.device)
            pmax = torch.full((B, D), -1e30, device=h.device)
            outs = []
            for t in range(T):
                kt = k[:, t]; vt = v[:, t]
                q = torch.maximum(pmax, u + kt)
                e1 = torch.exp(pmax - q); e2 = torch.exp(u + kt - q)
                wkv = (e1 * a + e2 * vt) / (e1 * b + e2 + 1e-8)
                outs.append(r[:, t] * self.Wo(wkv))
                pmax2 = torch.maximum(pmax + torch.log(wdec + 1e-30), kt)
                e1 = torch.exp(pmax + torch.log(wdec + 1e-30) - pmax2); e2 = torch.exp(kt - pmax2)
                a = e1 * a + e2 * vt; b = e1 * b + e2; pmax = pmax2
            return torch.stack(outs, 1)

    class SsmDualNonnegLayer(nn.Module):
        """DEPTH LEVER (2026-09-03, gap#1 fluency arc -- the spiking mouth's next mechanism after the ssm/
        dual-nonneg NO-GO and the divnorm NO-GO; research/findings/2026-09-03-spiking-mouth-ssm-dualnonneg-
        fluency-NO-GO-first-brain-based-baseline.md named DEPTH as the untested architectural lever, since the
        exact-math wkv reaches fluency at n_layers=2 but ssm/dual-nonneg was capped at n_layers=1 by an assert).
        One STACKABLE pre-norm residual dual-nonneg leaky-integrator block, used for layers >=1 (the deeper
        layers on top of layer 0, the model's existing base dual-nonneg loop in WKV.forward). Mirrors WkvLayer's
        stacking pattern EXACTLY (pre-norm -> per-layer recurrence -> the caller adds the residual h = h + delta)
        -- but its recurrence is the SSM/dual-nonneg leaky integrator, byte-for-byte the base loop's per-step
        update (ap2 = wdec*ap2 + relu(v_t); an2 = wdec*an2 + relu(-v_t); out = r_t * Wo_sp([ap2, an2])), NOT
        WkvLayer's num/den-normalized wkv_t. This is deliberate, not an oversight: --recurrence ssm's entire
        reason to exist (see the Rung 2 comment below, ~"SPIKING-SUBSTRATE-FAITHFUL leaky-integrator") is that
        the SPIKING SUBSTRATE realizes a slow leaky conductance integral, not a normalized-attention read;
        stacking WkvLayer blocks on an ssm base would silently reintroduce the exp(k)-weighted num/den op this
        branch exists to avoid, defeating the substrate-faithfulness the whole --recurrence ssm family is for.
        Each layer owns its own Wv/Wr/Wo_sp + decay w (uniform or per-channel per --uniform-decay) -- independent
        state, exactly like WkvLayer owns its own Wk/Wv/Wr/Wo/w/u rather than sharing layer 0's. Deliberately
        OMITS the base loop's optional co-adaptation levers (--input-noise/--plateau-surrogate/--dual-nonneg-
        divnorm): WkvLayer is "BYTE-FOR-BYTE the baseline wkv branch's DEFAULT loop" (its own docstring), so this
        mirrors that -- the bare recurrence only, matching the plain wkv depth lever's scope."""
        def __init__(self, D, uniform_decay):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            self.Wv = nn.Linear(D, D, bias=False)
            self.Wr = nn.Linear(D, D, bias=False)
            self.Wo_sp = nn.Linear(2 * D, D, bias=False)
            self.w = nn.Parameter(torch.zeros(1 if uniform_decay else D))

        def forward(self, h):                          # h: [B,T,D] -> delta [B,T,D] (pre-norm residual block)
            B, T, D = h.shape
            z = self.ln(h)
            v = self.Wv(z); r = torch.sigmoid(self.Wr(z))
            wdec = torch.exp(-torch.nn.functional.softplus(self.w))
            ap2 = torch.zeros(B, D, device=h.device); an2 = torch.zeros(B, D, device=h.device)
            outs = []
            for t in range(T):
                ip = torch.relu(v[:, t]); im = torch.relu(-v[:, t])
                ap2 = wdec * ap2 + ip; an2 = wdec * an2 + im
                outs.append(r[:, t] * self.Wo_sp(torch.cat([ap2, an2], -1)))
            return torch.stack(outs, 1)

    class HippoLayer(nn.Module):
        """FIXED diagonal HiPPO-structured multi-timescale recurrence + LEARNED local read-out (--recurrence
        hippo, 2026-09-03, gap#1 fluency arc's next mechanism after dual-nonneg+depth+tokens closed to
        margin_vs_trigram -0.125 but stayed trigram-bound -- research/findings/2026-09-03-spiking-depth-tokens-
        closing-fluency-gap-milestone.md). Ports the FIXED-recurrence extract already validated 6-seed GO on the
        pure memory-horizon task (research/findings/2026-07-13-SSM-fixed-structured-multitimescale-reservoir-
        SURPASSES-fading-memory-ceiling-6seed-GO.md, runner _ssm_fixed_structured_reservoir_derisk.py) into THIS
        file's language-model harness, so it is finally tested apples-to-apples on real NEXT-TOKEN PREDICTION
        (not pure retention). The July SCOPE CORRECTION on that same finding found a BARE linear diagonal fails
        at language prediction specifically because prediction needs nonlinear conjunctions over context that a
        LINEAR read-out over a LINEAR reservoir cannot form; here the read-out is a full learned Wo Linear PLUS a
        per-token receptance GATE r_t=sigmoid(Wr(z_t)) (the same nonlinear-gating machinery WkvLayer/
        SsmDualNonnegLayer already use), giving the read-out the nonlinear-conjunction capacity the bare-linear
        July arm lacked, while keeping the recurrence itself (A, the transition dynamics) exactly as FIXED/
        unlearned as the July gate -- the emergence-compatible bar from the task spec: "no BPTT through the
        recurrence structure; only C/head learned."

        A-INIT (HiPPO-LegS approximation, diag, fast->slow), justified against the theory: true HiPPO-LegS's
        continuous-time state matrix (Gu, Dao, Ermon, Rudra, Re 2020, "HiPPO: Recurrent Memory with Optimal
        Polynomial Projections", NeurIPS) has eigenvalues with real part ~ -(2n+1)/2 for Legendre basis order
        n=0..D-1 -- a SPREAD of decay rates growing with basis order, i.e. a genuine multi-timescale family of
        leaky integrators, NOT one shared time constant. SpikingSSMs (arXiv:2408.14909, cited by the fresh-gate
        finding this class descends from) frames this spread as dendritic-inspired multi-timescale integration.
        We reuse the SAME practical diagonal approximation already validated on the memory-horizon task
        (_ssm_fixed_structured_reservoir_derisk.py._build_A, kind="multitimescale"): tau_i LOG-spaced across
        [tau_lo, tau_hi] (default 1.5..1000 steps), a_i = exp(-1/tau_i) in (~0.49, ~0.999) -- fast units (small
        tau, a near 0) forget almost every step (shallow/local context, the bigram-like end of the spectrum),
        slow units (large tau, a near 1) integrate over the whole sequence (deep context). `A` is a torch BUFFER
        (register_buffer, NOT nn.Parameter) -- gradient NEVER reaches it, so there is zero learned recurrent
        credit through the transition dynamics.

        B (the input coupling into the recurrence, `x_{t+1}=A x_t + B u_t`) is ALSO a FIXED frozen random
        projection (a buffer, not nn.Parameter) -- matching the July gate's own W_in (a random reservoir input
        matrix, never trained) and the task spec's explicit "only C/head learned" bar: A and B together ARE the
        fixed developmental structure; only the read-out C (`self.Wo` below) and the receptance gate `self.Wr`
        (a per-token modulation of what is READ from the fixed state, not a change to the state dynamics itself)
        are learned by gradient, on top of the shared embedding upstream of B (self.emb, common to every
        --recurrence arm in this file).

        PERMUTE-A anti-cheat (--hippo-permute-a): reassigns which CHANNEL gets which tau (same multiset of decay
        rates, shuffled labeling) -- structurally distinct from eval_perdepth's generic sequence-level --permute
        anti-cheat (which shuffles TOKEN order, already exercised for every --recurrence arm). Included per the
        task spec; NOTE the July runner's OWN docstring for this exact control (kind="permuted" in
        _ssm_fixed_structured_reservoir_derisk.py._build_A) calls it "the diagonal structure without the
        principled range; should NOT change a diagonal reservoir, a SANITY control" -- i.e. the July source
        expects near-IDENTICAL results (channel identity is arbitrary to a fully-connected linear read-out),
        not a collapse. We run it and report the honest empirical number either way; see the runner's report for
        the reconciliation.
        """
        def __init__(self, D, tau_lo=1.5, tau_hi=1000.0, permute_a=False):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            tau = torch.exp(torch.linspace(math.log(tau_lo), math.log(tau_hi), D))
            A = torch.exp(-1.0 / tau)                      # FIXED diagonal decay, HiPPO-LegS-approx fast->slow
            if permute_a:
                A = A[torch.randperm(D)]                   # structural anti-cheat: shuffle channel<->tau labeling
            self.register_buffer("A", A)
            Bmat = torch.randn(D, D) / math.sqrt(D)         # FIXED random input projection (never trained)
            self.register_buffer("B", Bmat)
            self.Wr = nn.Linear(D, D, bias=False)           # LEARNED receptance gate (read-out pathway)
            self.Wo = nn.Linear(D, D, bias=False)           # LEARNED local read-out C

        def forward(self, h, memoryless=False):             # h: [B,T,D] -> delta [B,T,D] (pre-norm residual block)
            Bsz, T, D = h.shape
            z = self.ln(h)
            u = z @ self.B.t()                              # B u_t (fixed random input coupling)
            r = torch.sigmoid(self.Wr(z))
            A_eff = torch.zeros_like(self.A) if memoryless else self.A   # ANTI-CHEAT: no carry -> current token only
            x = torch.zeros(Bsz, D, device=h.device)
            outs = []
            for t in range(T):
                x = A_eff * x + u[:, t]                     # x_{t+1} = A x_t + B u_t  (FIXED recurrence)
                outs.append(r[:, t] * self.Wo(x))            # LEARNED local read-out C
            return torch.stack(outs, 1)

    class AssocLayer(nn.Module):
        """CONTENT-ADDRESSABLE associative read (learned-key attention), gap#1's next mechanism CLASS after the
        exhausted linear-recurrence family: WKV's num/den normalization, the ssm/dual-nonneg leaky integrator,
        and the FIXED-diagonal HiPPO multi-timescale SSM (--recurrence hippo, above) all converged on the SAME
        ~-0.125 margin_vs_trigram bound (research/findings/2026-09-03-spiking-depth-tokens-closing-fluency-gap-
        milestone.md; the HiPPO run reproduced it 2-seed). research/findings/2026-07-15-selective-ssm-generator-
        trigram-bound-... diagnosed WHY: every linear-recurrence family compresses the whole prefix into a
        FIXED-size state through a projection the read-out cannot losslessly invert, so it can only ever
        APPROXIMATE which past token mattered. A trigram instead keeps the EXACT identity of the last two
        tokens. This class is the mechanism the July record named to close that gap: EXACT content-addressable
        recall of a SPECIFIC past position, not a lossy compressed summary of all of them.

        OWNER STEER (2026-09-03): pursue open-ended own-voice fluency FULLY; content-addressable/attention-like
        reads are ACCEPTED even against the retire-the-transformer grain, PROVIDED they are biologically framed.
        BIOLOGICAL ANCHOR (not a raw transformer bolted on for its own sake): this is framed as an associative-
        memory / content-addressable-recall operation -- the computation CA3's recurrent collaterals perform
        during hippocampal PATTERN COMPLETION (a partial/current cue retrieves a whole previously-stored pattern
        by similarity), and that a cortical associative-memory column performs when a current context "looks
        up" a previously experienced conjunction. Ramsauer et al. 2020 ("Hopfield Networks is All You Need",
        ICLR 2021) proved the MODERN HOPFIELD NETWORK's continuous energy-based update -- iterated pattern
        completion against a set of stored patterns -- is mathematically IDENTICAL to dot-product-softmax
        attention; so "learned keys queried by a current probe, softmax-weighted retrieval of the associated
        value" is simultaneously the standard attention formulation AND a one-shot (single-iteration) modern-
        Hopfield/associative-memory read. The keys/values (Wk, Wv) are LEARNED FROM EXPERIENCE exactly as a
        cortical associative memory's synaptic weights are shaped by what actually co-occurred, not designed or
        hand-set -- "the keys learned from experience" the task spec names.

        Mechanism per position t (CAUSAL: t only ever reads s <= t, matching the recurrent layers' strict
        causality -- no leakage from the future):
          q_t = Wq(z_t)                                  -- the current probe/cue
          k_s = Wk(z_s), v_s = Wv(z_s)  for all s <= t    -- learned keys/values over every PAST position
          score_{t,s} = (q_t . k_s) / sqrt(D)             -- scaled dot-product content-similarity
          alpha_{t,:} = softmax_s(score_{t,:}, causal-masked at s>t)   -- competitive normalization over
                                                              candidate past positions (the "which past token
                                                              matters right now" competition a lossy fixed-size
                                                              state cannot run)
          read_t = sum_{s<=t} alpha_{t,s} v_s             -- the associative recall itself
          delta_t = Wo(read_t)                            -- caller adds this to the residual stream, h = h+delta
        Pre-norm residual block, matching WkvLayer/HippoLayer's `forward(h) -> delta` contract exactly so it
        stacks identically under --n-layers and composes with --contiguous unchanged (--contiguous only changes
        how load_stories builds the input token sequence upstream of this block; the block itself is agnostic
        to where its input sequence came from).

        MEMORYLESS anti-cheat (shared eval_perdepth machinery, WKV.forward's `self.memoryless` flag): masks
        every query to attend ONLY to s==t (itself) -- "no carry, current token only", the same semantics the
        ssm/hippo branches give this flag (their recurrence-off state). If deep-context NLL collapses toward
        bigram-level under this, the class's advantage was genuinely coming from READING the past through the
        causal attention, not from the current-token projection alone. The generic sequence-level --permute
        anti-cheat (eval_perdepth's `permute=True`, shuffles the prefix order per eval sentence) also applies
        unmodified to this branch -- it operates on the input token order before the net ever sees it, so no
        assoc-specific plumbing is needed; a collapse there shows the causal ORDER (not just bag-of-past-tokens
        membership) is load-bearing to the read.

        NOT a hidden re-admission of the retired transformer: no multi-head split, no FFN sublayer, no
        post-attention LayerNorm, no learned temperature -- exactly the four projections plus one causal softmax
        the biological framing above calls for, plus (--recurrence assoc_t only, see TEMPORAL CODE below) a
        FIXED "when" signal on the read competition -- and nothing more of the standard transformer block.

        TEMPORAL CODE (2026-09-03, `temporal=True` -> --recurrence assoc_t, diagnosing the underfit measured on
        the bag-of-tokens `assoc` above: at full scale its TRAINING loss converged to ~4.79, WORSE than the
        recurrences' ~4.36, and deep-bucket margin_vs_trigram ~ -0.35, worse than the SSM family's -0.125
        bound). ROOT CAUSE: `assoc` (temporal=False, the ORIGINAL branch, still reachable and UNCHANGED) reads
        z_s for every past position s<=t with no signal that distinguishes "s was 1 token ago" from "s was 30
        tokens ago" -- the causal softmax over content alone reads the past as an unordered BAG of tokens (two
        prefixes that contain the same tokens in a different ORDER produce IDENTICAL keys, hence identical
        read), so word ORDER within the context window is invisible to it. A sequential recurrence (wkv/ssm/
        hippo) gets order for free because its state is built by literally stepping through the sequence; a
        content-addressable read has to be TOLD when, or it cannot reconstruct it from content alone. This is
        why the bag-of-tokens read underfit the recurrences it was meant to surpass.

        BIOLOGICAL ANCHOR for the "when" signal (distinct from, and layered on top of, the CA3 pattern-
        completion anchor for the read itself, above): hippocampal TIME CELLS (MacDonald, Lepage, Eden, Eichenbaum
        2011, Neuron 71:737-749, "Hippocampal 'Time Cells' Bridge the Gap in Memory for Discontiguous Events")
        fire at successive, overlapping latencies during an unfilled temporal gap, tiling elapsed time with a
        population code the way place cells tile space -- so a CA3 pattern-completion read in vivo is never a
        pure content match; it is always conjoined with this population's "when" signal, which is what lets an
        animal retrieve the events of an episode IN ORDER rather than as an unordered set. Howard & Kahana's
        Temporal Context Model (2002, J. Math. Psychol. 46:269-299) formalizes the companion mechanism at the
        systems level: a slowly-drifting temporal-context vector is bound to each item at encoding and used as
        part of the retrieval cue, so recall competition is driven by content-similarity CONJOINED WITH
        context/recency, not content alone -- exactly the missing conjunction diagnosed above. We realize this
        population code as a bank of fixed sinusoidal "time cells" (frequencies log-spaced across the sequence,
        each a smooth, overlapping, monotonically-phase-advancing function of elapsed position t -- MacDonald's
        tiling property in closed form) ADDED to the per-position representation feeding Wq/Wk ONLY (the read
        COMPETITION becomes when-and-what sensitive) and NOT Wv (the read VALUE stays pure content, matching
        the CA3 anchor: what is retrieved is a stored pattern, not a timestamp). This is deliberately NOT framed
        as a bare transformer positional encoding bolted on for engagement: it is fixed (a registered buffer,
        `nn.Parameter` never touches it, exactly like HippoLayer's fixed A/B above -- no gradient can warp the
        time code into an arbitrary learned signal), and it is added to a MEMORY RETRIEVAL competition (Q/K),
        never to the recalled content (V) -- the TCM/time-cell role, not a sequence-labeling trick.

        ANTI-CHEAT PREDICTION (falsifiable, checked in the CPU smoke): the existing generic --permute anti-cheat
        (eval_perdepth's permute=True, shuffles the prefix order before the net ever sees it) should degrade
        `assoc_t` MORE than the bag-of-tokens `assoc`, because `assoc_t` is now the one arm whose read genuinely
        depends on positional order -- `assoc` was already order-blind (nothing left for permutation to break).

        LEARNED RETRIEVAL GATE (2026-09-03, `--assoc-gate`, default OFF, composes with BOTH `assoc` and
        `assoc_t`): the 2026-07-11 learned-keys de-risk (research/findings/2026-07-11-LEARNED-keys-make-
        content-addressable-retrieval-load-bearing-...md) found the raw associative read is
        informative-but-NOISY -- content-addressing IS load-bearing (content << shuffle) but the raw retrieved
        feature is a net COST over the base read-out (content - base stays positive), and that finding names the
        fix explicitly: "a learned GATING of when to trust the retrieval; retrieval as a RESIDUAL CORRECTION,
        not a raw appended feature." Until now `read` was appended UNGATED (`hh = hh + blk(hh)`, and
        `blk(hh) = Wo(read)`, an unconditional residual). This adds a per-channel, input-conditioned trust gate
        `g_t = sigmoid(Wg(z_t)) in (0,1)^D` (z_t = the block's OWN pre-norm'd input, the same tensor that feeds
        Wq/Wk/Wv -- so "how much do I trust a recall here" is itself content-conditioned, not context-free) that
        scales the recalled value BEFORE the output mixing: `delta_t = Wo(g_t * read_t)`.

        BIO FRAMING: gating WHETHER a hippocampal/associative recall is trusted right now is not exotic --
        thalamocortical and neuromodulatory (ACh/NE) gating of what a cortical target admits from a hippocampal
        CA3 read-out is a standing motif (the same "open the channel only when the recall is useful" role
        LSTM/GRU input gates formalize computationally). This is a FORWARD-pass gain-control decision on the
        read, not a credit-assignment rule -- it does not touch how Wq/Wk/Wv/Wo/Wg themselves get their
        gradients.

        PLACEMENT CHOICE (pre-Wo on the read, not post-Wo on the block output): the read `read_t` lives in the
        SAME per-channel space as `v_t = Wv(z_s)` (V's native content channels), so a per-channel gate there
        reads literally as "trust THIS recalled content channel." Gating the block's OUTPUT instead (after Wo
        has already linearly mixed those channels together) would conflate "distrust this recalled feature"
        with "distrust this particular Wo-mixed combination of features" -- a less legible knob, and it would
        NOT compose as cleanly with the biological framing above (the gate belongs on the retrieved content
        itself, mirroring where an LSTM/GRU input gate sits -- on the candidate value BEFORE it is written in,
        not on the cell's already-mixed output).

        INIT-OPEN (Wg.weight = 0, Wg.bias = +2.0 -> g_t = sigmoid(2.0) ~ 0.88 at init, UNIFORMLY across
        channels/positions, independent of z_t until training moves the weight off zero): the standard forget-
        gate-bias-positive trick (Jozefowicz, Sutskever, Vinyals 2015, "An Empirical Exploration of Recurrent
        Network Architectures") applied to a trust gate instead of a forget gate -- at init the gate is ROUGHLY
        OPEN (close to the pre-gate ungated behavior, `g~0.88` not `g~1.0` so there is already a small headroom
        pushing training to explore closing it), so training only has to LEARN where to CLOSE the gate (down-
        weight channels where the recall is noisy), not first discover that opening it helps.

        BYTE-IDENTICAL WHEN OFF (the load-bearing guarantee): `self.Wg` is constructed ONLY when `gate=True` is
        passed in -- when `--assoc-gate` is unset, no `Wg` module exists at all, so it consumes ZERO init RNG
        draws and the forward path is the untouched original `return self.Wo(read)`. `assoc`/`assoc_t` without
        `--assoc-gate` are therefore bit-identical to before this addition; wkv/ssm/hippo never construct
        AssocLayer at all (see `assoc_layers`'s conditional construction below) and are unaffected regardless.
        """
        def __init__(self, D, temporal=False, gate=False):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            self.Wq = nn.Linear(D, D, bias=False)
            self.Wk = nn.Linear(D, D, bias=False)
            self.Wv = nn.Linear(D, D, bias=False)
            self.Wo = nn.Linear(D, D, bias=False)
            self.scale = 1.0 / math.sqrt(D)
            self.temporal = temporal
            if temporal:
                # FIXED (buffer, not nn.Parameter -- gradient never reaches it, matching HippoLayer's fixed A/B
                # discipline) sinusoidal "time cell" frequency ladder: log-spaced frequencies over the channel
                # dim, standard sin/cos construction (Vaswani et al. 2017 Sec 3.5) reused here as the closed-form
                # realization of MacDonald et al. 2011's overlapping-latency time-cell population code (see the
                # class docstring's TEMPORAL CODE section for the bio anchor + why this is not a bare
                # transformer PE). Depends only on D (not on T), so it is computed once here; the actual
                # position-dependent code is built per-forward from this ladder (sequence length T varies).
                div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
                self.register_buffer("_time_div", div_term)          # [ceil(D/2)] fixed frequency ladder
            # LEARNED RETRIEVAL GATE (--assoc-gate, see the class docstring's LEARNED RETRIEVAL GATE section):
            # constructed ONLY when gate=True -- when the flag is unset, `self` has NO `Wg` attribute, consumes
            # ZERO extra init-RNG draws, and `self.gate` is a plain False the forward branch below skips, so the
            # Wq/Wk/Wv/Wo/(_time_div) construction above and the whole forward computation are BYTE-IDENTICAL
            # to before this addition -- the required off-by-default guarantee.
            self.gate = gate
            if gate:
                self.Wg = nn.Linear(D, D, bias=True)
                nn.init.zeros_(self.Wg.weight)          # g_t starts UNIFORM across channels (no input dependence yet)
                nn.init.constant_(self.Wg.bias, 2.0)    # sigmoid(2.0)~0.88 -> roughly OPEN at init (see docstring)

        def forward(self, h, memoryless=False):        # h: [B,T,D] -> delta [B,T,D] (pre-norm residual block)
            B, T, D = h.shape
            z = self.ln(h)
            if self.temporal:
                # "when" signal (time cells / TCM temporal context, see docstring): a population of fixed
                # overlapping phase codes over elapsed position t, added to Q/K's input only -- z itself (hence
                # V = Wv(z) below) is untouched, so the associative VALUE stays pure content.
                pos = torch.arange(T, dtype=torch.float32, device=h.device).unsqueeze(1)     # [T,1]
                ang = pos * self._time_div.unsqueeze(0)                                       # [T,ceil(D/2)]
                time_code = torch.zeros(T, D, device=h.device, dtype=z.dtype)
                time_code[:, 0::2] = torch.sin(ang)
                time_code[:, 1::2] = torch.cos(ang[:, :time_code[:, 1::2].shape[1]])
                zt = z + time_code.unsqueeze(0)                       # [B,T,D], broadcast over the batch
            else:
                zt = z                                                # bag-of-tokens (unchanged, --recurrence assoc)
            q = self.Wq(zt); k = self.Wk(zt); v = self.Wv(z)
            scores = torch.einsum("btd,bsd->bts", q, k) * self.scale    # [B,T,T] raw content(+when)-similarity scores
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device))
            if memoryless:
                # ANTI-CHEAT: no carry -> current token only (mirrors ssm/hippo's memoryless semantics) --
                # restrict every query to attend ONLY to itself (s==t), collapsing the associative read to a
                # pure self-read (no content-addressable recall of any PAST position survives).
                causal = torch.eye(T, dtype=torch.bool, device=h.device)
            scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
            alpha = torch.softmax(scores, dim=-1)                        # [B,T,T] causal content-addressed weights
            read = torch.einsum("bts,bsd->btd", alpha, v)                # [B,T,D] weighted associative recall
            if self.gate:
                # LEARNED RETRIEVAL GATE (--assoc-gate): per-channel "trust this recall" gate, content-
                # conditioned on the block's own pre-norm'd input z (NOT zt -- the gate decision itself is a
                # pure-content read of the current position, matching Wv's convention of reading z not zt).
                # Gates the associative VALUE before Wo mixes channels (see PLACEMENT CHOICE in the class
                # docstring). Skipped entirely when gate=False -> untouched original `return self.Wo(read)`.
                g = torch.sigmoid(self.Wg(z))          # [B,T,D] in (0,1), init ~0.88 (roughly open, see docstring)
                read = g * read
            return self.Wo(read)

    class LinAttnLayer(nn.Module):
        """NORMALIZED HEBBIAN FAST-WEIGHT LINEAR ATTENTION (--recurrence linattn, 2026-09-03), the deployable-
        spiking successor to ssm/dual-nonneg -- see the full DESIGN doc,
        research/findings/2026-09-03-spiking-content-addressable-read-DESIGN.md, for the complete derivation,
        the deep-read of the spiking-LM literature (SpikeGPT/SpikeLM/SpikingSSMs/Spikformer/BiSpikCLM/
        WTA-Spiking-Transformer), and the biological anchors. Summary of the mechanism this class implements:

        `dual-nonneg` (SsmDualNonnegLayer above) discards RWKV's numerator/denominator NORMALIZATION -- the
        division by an accumulated decay-weighted denominator that gives `wkv` its content-addressed,
        softmax-like weighting over past tokens (diagnosed in research/findings/2026-09-03-spiking-mouth-ssm-
        dualnonneg-fluency-NO-GO-first-brain-based-baseline.md). This class restores BOTH coupled pieces the
        diagnosis names -- a content-dependent nonnegative WRITE GAIN phi(k_t) on each token's contribution, and
        a running DENOMINATOR trace that accumulates that same gain and DIVIDES the read by it -- AND adds
        genuine query-key content-addressing (phi(q)^T M) that even `wkv` lacks (wkv's `k` is a per-channel
        gain, not a q.k match). It is linear-attention in the fast-weight form (Katharopoulos, Vyas, Pappas &
        Fleuret 2020, "Transformers are RNNs", arXiv:2006.16236, Eqs. 7/10-12/18-20): O(T) recurrent, no T x T
        matrix (unlike `assoc`/`assoc_t`), so it is spike-deployable.

        Mechanism per position t (causal, D x D real-valued state):
          z_t = LN(h_t);  q_t=Wq(z_t), k_t=Wk(z_t), v_t=Wv(z_t);  phi(.) = elu(.)+1 by default (non-negative
          feature map, Katharopoulos Eq.7; --linattn-phi selects relu/exp/sparse alternatives, see AssocLayer's
          TEMPORAL CODE docstring section for the WTA-sparse-key precedent).
          WRITE:  M_t = lam (*) M_{t-1} + phi(k_t) (x) v_t        -- a real-valued outer-product KV trace (D x D)
                  zden_t = lam (*) zden_{t-1} + phi(k_t)          -- its running normalizer (the denominator trace)
          READ:   num_t = phi(q_t)^T M_t;  den_t = phi(q_t)^T zden_t;  read_t = num_t / (den_t + eps)
          delta_t = Wo(r_t (*) read_t), r_t = sigmoid(Wr(z_t))    -- caller adds this to the residual stream

        `lam = exp(-softplus(w)) in (0,1)` is the per-channel (or scalar, --uniform-decay) leak, identical in
        role to wkv's exp(-w). This STRICTLY GENERALIZES wkv (restrict M to its diagonal, Wq=Wk=I, phi=exp and
        it degenerates to wkv's per-channel num/den), so with usable capacity it cannot do worse than the wkv
        upper bound it descends from.

        BIOLOGY (DESIGN doc Sec 3, brief): the real-valued outer-product KV trace M is short-term synaptic
        plasticity / a fast-weight matrix -- graded, calcium-mediated synaptic facilitation holding working
        memory (Mongillo, Barak & Tsodyks 2008, Science 319:1543-1546, doi:10.1126/science.1150769; Ba, Hinton,
        Mnih, Leibo & Ionescu 2016, "Using Fast Weights to Attend to the Recent Past", arXiv:1610.06258). The
        Hebbian outer-product WRITE (phi(k) (x) v, pre x post) is CA3 recurrent-collateral autoassociation
        (Marr 1971; Treves & Rolls 1994; Rolls & Treves 1998) -- the same anchor AssocLayer's docstring uses for
        its causal-softmax read (Ramsauer et al. 2020 proves modern-Hopfield <-> attention equivalence). The
        DIVISION num/den is divisive normalization by shunting inhibition (Carandini & Heeger 1994, 2012, Nat
        Rev Neurosci 13:51-62) -- over the QUERY'S MATCH-MASS axis, not the channel population (the axis fix
        that distinguishes this from the already-refuted `--dual-nonneg-divnorm` channel-pool NO-GO, DESIGN
        doc Sec 4); honest caveat, Holt & Koch 1997 (Neural Comput. 9:1001) found pure somatic shunting is
        SUBTRACTIVE not divisive on firing rate -- the on-substrate realization is a later rung, in scope for an
        honest negative if it degrades.

        `--linattn-phi sparse`: a k-winners-take-all phi(k) approximates a hard content match -- biologically a
        sparse pattern-separated key (DG->CA3), the WTA-Spiking-Transformer's sparse-softmax limit (DESIGN doc
        Sec 2/5b).

        `--assoc-gate` (gate=True, reused from AssocLayer, same init-open trick: Wg.weight=0, Wg.bias=+2.0 ->
        g~0.88 at init): the 2026-07-11 learned-keys de-risk found the raw retrieved feature is
        informative-but-noisy and named the fix "a learned GATING of when to trust the retrieval, retrieval as
        a RESIDUAL CORRECTION" -- gates `read_t` before Wo, exactly as AssocLayer's LEARNED RETRIEVAL GATE does.

        `--no-linattn-norm` (norm=False): THE KEY ABLATION -- drops the `/ (den_t + eps)` division so the read
        is the raw unnormalized sum `num_t`, isolating whether the restored content-weighted normalization
        (not merely the outer-product q.k widening) is what is load-bearing (DESIGN doc Sec 6, the cheapest
        decisive CPU experiment).

        Pre-norm residual block, `forward(h, memoryless=False) -> delta`, IDENTICAL contract to
        WkvLayer/HippoLayer/AssocLayer, so it stacks under --n-layers and composes with --contiguous/
        --tokenizer bpe unchanged (DESIGN doc Sec 5c). MEMORYLESS anti-cheat: mirrors AssocLayer's semantics
        exactly ("no carry, current token only") -- when True, M/zden are never accumulated across positions;
        each position's read uses ONLY that position's own phi(k_t) (x) v_t / phi(k_t), so no past position can
        be recalled. The generic sequence-level --permute anti-cheat (eval_perdepth's permute=True) also
        applies unmodified, exactly as for every other --recurrence branch.

        BYTE-IDENTICAL WHEN OFF: this class and `self.linattn_layers` are constructed ONLY when
        `--recurrence linattn` is selected (see the ModuleList construction in WKV.__init__, and the
        RECUR=="linattn" forward dispatch below) -- when linattn is not selected, this class is defined but
        never instantiated, consuming ZERO init-RNG draws, so wkv/ssm/hippo/assoc/assoc_t are completely
        unaffected by this addition.
        """
        def __init__(self, D, uniform_decay=False, phi="elu", gate=False, norm=True,
                     div_mode="exact", div_g_leak=1e-6, div_k=1.0):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            self.Wq = nn.Linear(D, D, bias=False); self.Wk = nn.Linear(D, D, bias=False)
            self.Wv = nn.Linear(D, D, bias=False); self.Wr = nn.Linear(D, D, bias=False)
            self.Wo = nn.Linear(D, D, bias=False)
            self.w = nn.Parameter(torch.zeros(1 if uniform_decay else D))   # lam = exp(-softplus(w)), like wkv
            self.phi = phi
            self.norm = norm            # --no-linattn-norm ablation: False -> raw unnormalized sum (num_t only)
            self.gate = gate
            # --linattn-div {exact,shunt} (2026-09-03 Tier-1 de-risk, research/findings/2026-09-03-linattn-
            # spike-native-normalization-DESIGN.md Sec 3e/4): DEFAULT "exact" is BYTE-IDENTICAL to the pre-
            # existing `num/(den+eps)` division (see _divisive_read below -- mode="exact" is spelled identically
            # to the formula this replaces). "shunt" swaps in the Carandini-Heeger conductance-divisive-gain
            # rate-model form `num/(g_leak + k*den)` -- the spike-native realization this design specifies.
            # Training-side plumbing only (this arc's own Tier-1 de-risk runs the READ-SIDE swap on an already-
            # trained checkpoint via LinAttnReadout, research/runners/_wkv_fewspike_read_derisk.py, which needs
            # no retrain); wired here too so a Tier-2 retrain-in-the-loop (DESIGN Sec 4, "if Tier 1 needs the
            # read-in-the-loop retrain") has the identical flag/semantics available without another edit.
            self.div_mode = div_mode; self.div_g_leak = float(div_g_leak); self.div_k = float(div_k)
            if gate:
                self.Wg = nn.Linear(D, D, bias=True)
                nn.init.zeros_(self.Wg.weight); nn.init.constant_(self.Wg.bias, 2.0)   # init-open, reuse assoc-gate

        def _phi(self, x):
            if self.phi == "elu":  return torch.nn.functional.elu(x) + 1.0
            if self.phi == "relu": return torch.relu(x) + 1e-3
            if self.phi == "exp":  return torch.exp(x - x.amax(-1, keepdim=True))       # stable RWKV-like
            if self.phi == "sparse":                                                     # k-WTA sparse key
                kth = torch.topk(x, max(1, x.shape[-1] // 8), dim=-1).values[..., -1:]
                return torch.relu(x - kth)
            raise ValueError(self.phi)

        @staticmethod
        def _divisive_read(num, den, mode="exact", g_leak=1e-6, k=1.0, fI=None):
            """Spike-native num/den realization (DESIGN doc Sec 3e sketch, transcribed verbatim from the design's
            own `divisive_read` pseudocode). `mode="exact"` -> `num/(den+1e-6)`, spelled IDENTICALLY to the
            formula it replaces -- so div_mode="exact" (the default) is BYTE-IDENTICAL to every call site that
            predates this flag. `mode="shunt"` -> `num/(g_leak + k*den)`, the Carandini-Heeger conductance-
            divisive-gain rate-model form: `g_leak` (sigma) is the read neuron's leak conductance -- AT
            `g_leak=1e-6, k=1.0` this is algebraically IDENTICAL to "exact" (both reduce to `num/(den+1e-6)`),
            which is why the Tier-1 de-risk (no retrain) instead varies `fI`/quantization/`g_leak`/`k` around
            that point to test robustness, not the bare formula. `fI`, if given, is the read neuron's own
            monotone f-I transfer (rate saturation) applied AFTER the divisive gain (design Sec 3c effect 2)."""
            g = num / (den + 1e-6) if mode == "exact" else num / (g_leak + k * den)
            return fI(g) if fI is not None else g

        def forward(self, h, memoryless=False):        # h:[B,T,D] -> delta:[B,T,D] (pre-norm residual block)
            B, T, D = h.shape
            z = self.ln(h)
            q = self._phi(self.Wq(z)); k = self._phi(self.Wk(z)); v = self.Wv(z)
            r = torch.sigmoid(self.Wr(z))
            lam = torch.exp(-torch.nn.functional.softplus(self.w))          # (D,) or (1,)
            M = torch.zeros(B, D, D, device=h.device)                       # outer-product KV trace
            zden = torch.zeros(B, D, device=h.device)                       # normalizer trace
            outs = []
            for t in range(T):
                if memoryless:                                              # ANTI-CHEAT: current token only
                    M_r = torch.einsum("bd,be->bde", k[:, t], v[:, t]); zden_r = k[:, t]
                else:
                    M = lam.unsqueeze(-1) * M + torch.einsum("bd,be->bde", k[:, t], v[:, t])
                    zden = lam * zden + k[:, t]
                    M_r, zden_r = M, zden
                num = torch.einsum("bd,bde->be", q[:, t], M_r)              # phi(q)^T M
                den = torch.einsum("bd,bd->b", q[:, t], zden_r).unsqueeze(-1)   # phi(q)^T zden  (scalar/token)
                read = (self._divisive_read(num, den, mode=self.div_mode, g_leak=self.div_g_leak, k=self.div_k)
                        if self.norm else num)                             # --no-linattn-norm = raw sum ablation
                if self.gate: read = torch.sigmoid(self.Wg(z[:, t])) * read
                outs.append(self.Wo(r[:, t] * read))
            return torch.stack(outs, 1)

    class DeltaNetLayer(nn.Module):
        """ERROR-CORRECTIVE DELTA-RULE FAST-WEIGHT WRITE on the linattn substrate (--recurrence deltanet,
        2026-09-05, own-voice-fluency arc, DR-ladder rung 3). This is a WRITE-RULE FIX to LinAttnLayer's SAME
        D x D fast-weight KV trace -- NOT a new content-addressing key, and NOT a reproposal of the banked
        content-addressing family (assoc/assoc_t/learnkey/hippokey, all of which lose to trigram): the query,
        key, value, feature map phi, learned per-channel decay and output gate are IDENTICAL to LinAttnLayer;
        ONLY the line that UPDATES the KV trace M changes, from additive Hebbian to error-correcting delta.

        WHY (the measured failure this targets): LinAttnLayer's additive Hebbian write
            M_t = lam (*) M_{t-1} + phi(k_t) (x) v_t
        accumulates EVERY token's key->value binding into M with NO erasure, so as content diversity rises
        (narrow simplewiki -> broad wikitext-103) the trace saturates with mutually-interfering bindings and its
        norm grows unbounded. Measured directly: linattn CROSSES a fair interpolated-trigram on simplewiki
        (+0.05, 6/6; research/findings/2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-deployable-spiking-mouth-
        beats-trigram-6of6.md) but FALLS BELOW trigram on the broad wikitext-103 domain (margin_vs_trigram
        -0.29..-0.57 at depth>=2; research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json) -- exactly
        the interference / unbounded-norm signature.

        THE DELTA RULE (Widrow-Hoff error correction; erase-before-write; Schlag et al. 2021 "Linear
        Transformers Are Secretly Fast Weight Programmers" arXiv:2102.11174; Yang et al. 2024 "Gated Delta
        Networks: Improving Mamba2 with Delta Rule" arXiv:2412.06464). Per position t (key-major M [D_k, D_v]):
            k_hat_t = phi(k_t) / ||phi(k_t)||_2            (--delta-key-norm l2, default; the unit-norm key that
                                                            makes the erase an EXACT projection removal)
            M'      = lam (*) M_{t-1}                       (learned per-channel decay diag(w), exactly linattn's)
            v_old   = k_hat_t^T M'                          (the value CURRENTLY bound to this key, retrieved)
            M_t     = M' + beta * k_hat_t (x) (v_t - v_old) (subtract it, write ONLY the RESIDUAL)
            read_t  = phi(q_t)^T M_t                        (raw content read; NO num/den -- see NORMALIZATION)
            delta_t = Wo(r_t (*) read_t), r_t = sigmoid(Wr(z_t))    -- caller adds this to the residual stream
        Reading the just-written key back gives k_hat^T M_t = v_old + beta*||k_hat||^2 (v - v_old) = v_t at
        beta=1, ||k_hat||=1 -- EXACT error correction. Because each write REPLACES rather than ADDS the binding
        for its key, M's norm stays BOUNDED under interference (the delta rule's defining property, and the
        direct fix for linattn's measured unbounded-norm failure). lam = exp(-softplus(w)) is linattn's
        IDENTICAL learned per-channel (or scalar, --uniform-decay) decay, here also playing the gated-delta-
        rule's decay-gate role (Yang 2024's alpha_t; the erase reads the DECAYED prior state M', the consistent
        gated form). beta (--delta-beta, default 1.0) is the write strength (canonical unit-step Widrow-Hoff at
        1.0), a FIXED scalar -> this arm adds ZERO extra parameters over linattn.

        NORMALIZATION (why raw read, no num/den): the delta rule bounds the state at the WRITE (erase-before-
        write) -- a DIFFERENT and stronger mechanism than linattn's read-side num/den average. Keeping linattn's
        additive denominator (zden_t = lam*zden + phi(k)) alongside a delta-corrected M is INCOHERENT: zden
        would grow unbounded while M stays bounded, so the read num/den -> 0 over long contexts. The entire
        DeltaNet family (Schlag 2021, Yang 2024, RWKV-7 Peng et al. 2025 arXiv:2503.14456) therefore reads the
        fast weight RAW (o_t = S_t q_t), which the L2-normalized keys keep well-conditioned. The clean ISOLATION
        control for 'is the win the write rule or the loss of the denominator?' is the already-existing linattn
        --no-linattn-norm arm (additive write + raw read): deltanet vs that isolates the delta WRITE, since both
        read raw.

        BIOLOGY (brain-based-only, DESIGN-consistent with LinAttnLayer): the delta rule is LOCAL and weight-
        transport-free (the correction v_t - v_old is available at the synapse from the postsynaptic value
        cell's own activity minus the retrieved prediction), so it is the SPIKING-PORTABLE write rule, not a
        host-only trick. Erase-before-write is realizable by short-term synaptic plasticity: the presynaptic key
        drive both reads the currently-bound value (facilitated transmission) and DEPRESSES the pre-existing
        binding before the new value is potentiated (presynaptic-driven short-term / heterosynaptic depression).
        Widrow & Hoff 1960 (the classical local LMS/delta error-correcting synaptic rule; Dayan & Abbott 2001
        Ch.8). Same fast-weight = STP anchor as LinAttnLayer (Mongillo, Barak & Tsodyks 2008; Ba, Hinton et al.
        2016).

        OUR-RECORD PROVENANCE (this is NOT a refuted reproposal -- stated explicitly per the build scope): the
        delta-rule fast-weight store was scoped the cheap-first #1 next mechanism (bio HIGH, drop-in;
        "M += eta(v - M k)k^T, read v_hat = M q") on 2026-07-15 (research/findings/2026-07-15-emergence-engine-
        research-gate-horizon-frontier-is-a-nonfading-content-addressable-store-delta-rule-fastweight-is-1.md)
        but was NEVER BUILT for the LM. 2026-07-13 (research/findings/2026-07-13-input-magnitude-gating-robust-
        generic-deep-context-lift-but-NOT-content-selective-6seed.md) conditionally green-lit the content-
        selective / delta memory PENDING 'structured codes + real scale' (one-hot x random-W_in codes had no
        content-vs-filler structure for a selective write to exploit) -- a precondition linattn's LEARNED BPE
        embeddings at 13.5M tokens now MEET. The 2026-07-15 edge5-rung3 delta-write that WAS refuted
        (research/findings/2026-07-15-edge5-rung3-delta-write-PARTIAL-error-correction-refuted.md) is a
        DIFFERENT mechanism at a DIFFERENT scale: a store-side ONE-SHOT ON-BRIDGE potentiate/depress for a
        spiking discourse BINDER (KV=4/8 value pools, P<=8 binds) -- refuted BECAUSE it was 'too coarse to
        reproduce the numpy delta rule's ITERATIVE MATRIX error-correction' (that finding's own words). THIS arm
        IS that iterative matrix delta, at LM scale.

        BYTE-IDENTICAL WHEN OFF: this class and self.deltanet_layers are constructed ONLY when
        --recurrence deltanet is selected (see the ModuleList construction in WKV.__init__ and the
        RECUR=="deltanet" forward dispatch) -- defined-but-never-instantiated otherwise, consuming ZERO init-RNG
        draws, so wkv/ssm/hippo/assoc/assoc_t/linattn/learnkey/hippokey are completely unaffected at any
        --n-layers. Its __init__ mirrors LinAttnLayer's module set exactly (Wq/Wk/Wv/Wr/Wo/w + optional Wg), so
        deltanet is a clean structural sibling of linattn differing ONLY in the write rule.

        MEMORYLESS anti-cheat (identical semantics to LinAttn/Assoc): when True, M is NEVER carried across
        positions -- each position's read uses ONLY its own k_hat (x) v, so no past position can be recalled
        (collapses to ~bigram). The generic sequence-level --permute anti-cheat (eval_perdepth permute=True)
        applies unmodified. Both MUST collapse for the context/order use to be genuine.
        """
        def __init__(self, D, uniform_decay=False, phi="elu", gate=False, beta=1.0, key_norm="l2"):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            self.Wq = nn.Linear(D, D, bias=False); self.Wk = nn.Linear(D, D, bias=False)
            self.Wv = nn.Linear(D, D, bias=False); self.Wr = nn.Linear(D, D, bias=False)
            self.Wo = nn.Linear(D, D, bias=False)
            self.w = nn.Parameter(torch.zeros(1 if uniform_decay else D))   # lam = exp(-softplus(w)), like linattn
            self.phi = phi
            self.beta = float(beta)              # write strength (FIXED scalar -> zero extra params over linattn)
            self.key_norm = key_norm            # "l2" (exact erase) | "none" (task-literal unnormalized phi(k))
            self.gate = gate
            if gate:
                self.Wg = nn.Linear(D, D, bias=True)
                nn.init.zeros_(self.Wg.weight); nn.init.constant_(self.Wg.bias, 2.0)   # init-open, reuse assoc-gate

        def _phi(self, x):                       # identical feature map to LinAttnLayer._phi
            if self.phi == "elu":  return torch.nn.functional.elu(x) + 1.0
            if self.phi == "relu": return torch.relu(x) + 1e-3
            if self.phi == "exp":  return torch.exp(x - x.amax(-1, keepdim=True))       # stable RWKV-like
            if self.phi == "sparse":                                                     # k-WTA sparse key
                kth = torch.topk(x, max(1, x.shape[-1] // 8), dim=-1).values[..., -1:]
                return torch.relu(x - kth)
            raise ValueError(self.phi)

        def forward(self, h, memoryless=False):        # h:[B,T,D] -> delta:[B,T,D] (pre-norm residual block)
            B, T, D = h.shape
            z = self.ln(h)
            q = self._phi(self.Wq(z)); kk = self._phi(self.Wk(z)); v = self.Wv(z)
            r = torch.sigmoid(self.Wr(z))
            if self.key_norm == "l2":
                kk = kk / (kk.norm(dim=-1, keepdim=True) + 1e-6)            # unit-norm key -> EXACT erase
            lam = torch.exp(-torch.nn.functional.softplus(self.w))          # (D,) or (1,), identical to linattn
            M = torch.zeros(B, D, D, device=h.device)                       # outer-product KV trace [B,D_k,D_v]
            outs = []
            for t in range(T):
                if memoryless:                                             # ANTI-CHEAT: current token only, no carry
                    M_r = torch.einsum("bd,be->bde", kk[:, t], v[:, t])
                else:
                    Md = lam.unsqueeze(-1) * M                             # decayed prior state (gated-delta form)
                    v_old = torch.einsum("bd,bde->be", kk[:, t], Md)       # value CURRENTLY bound to this key
                    M = Md + self.beta * torch.einsum("bd,be->bde", kk[:, t], v[:, t] - v_old)  # write RESIDUAL
                    M_r = M
                read = torch.einsum("bd,bde->be", q[:, t], M_r)            # phi(q)^T M  (raw -- DeltaNet reads raw)
                if self.gate: read = torch.sigmoid(self.Wg(z[:, t])) * read
                outs.append(self.Wo(r[:, t] * read))
            return torch.stack(outs, 1)

    class LearnKeyLayer(nn.Module):
        """FIXED-CAPACITY LEARNED-KEY CONTENT-ADDRESSABLE MEMORY (--recurrence learnkey, 2026-09-04). Build-ahead
        fallback: gap#1's NAMED next mechanism CLASS after `linattn` in the roadmap's own fluency lineage
        ("structured HiPPO SSM -> content-addressable learned-key attention", MEMORY
        project_own_voice_fluency_pursue_fully_2026_09_03) -- prepared and smoke-tested, held in reserve for if
        the linattn PRODUCTION-SCALE sweep plateaus, per owner instruction. NOT a re-run of `assoc`/`assoc_t`
        under a new flag: those two ALREADY instantiated "content-addressable learned-key attention" in this
        file and both LOST to the -0.125 ssm/dual-nonneg floor (bag -0.347, ordered -0.147; see
        research/findings/2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-deployable-spiking-mouth-beats-trigram-
        6of6.md's comparison table) -- re-adding the identical mechanism under `learnkey` would be a hollow
        duplicate of an already-adjudicated NO-GO. This class is a STRUCTURALLY DISTINCT member of the same
        bio-anchored family, chosen specifically to fix the property diagnosed as disqualifying about
        assoc/assoc_t (unbounded, per-token, O(T^2) keys -- NOT spiking-realizable) while staying genuinely
        content-addressable+learned-key (unlike linattn's continuous, keyless kernel trace):

          - `assoc`/`assoc_t`: keys are DERIVED per past TOKEN (Wk(z_s), one new key for every position ever
            seen) -- the key SET grows with T, so the read is an O(T^2) softmax over an ever-growing population.
            No fixed spiking substrate can host an unboundedly growing set of key-selective units.
          - `linattn`: normalization restored, O(T) recurrent, spike-deployable -- but its "key" is a continuous
            D x D outer-product trace with no discrete addressable slots; there is no inspectable set of
            "keys" a downstream reader (or a biologist) could point to.
          - `learnkey` (this class): a FIXED, SMALL bank of M LEARNED key prototypes (`self.Kmem`, an
            nn.Parameter[M,D] -- genuinely learned by gradient descent over the whole corpus, like a real
            synaptic weight matrix shaped by experience, but INDEPENDENT of any single position's content,
            unlike assoc's per-token Wk(z_s)). M is fixed at construction time (--learnkey-slots, default 64)
            and never grows with T, so the read is O(T*M*D) -- linear in sequence length, exactly like linattn,
            but through a DISCRETE, addressable, inspectable codebook instead of a continuous compressed trace.

        BIOLOGICAL ANCHOR: a bounded population of memory-index units whose afferent weights (the M key
        prototypes) are shaped by experience, holding content written/retrieved by competitive similarity -- the
        classical associative-net formulation (Willshaw, Buneman & Longuet-Higgins 1969, "Non-holographic
        associative memory", Nature 222:960-962; Kanerva 1988, "Sparse Distributed Memory", MIT Press) and
        Marr's (1971) CA3 autoassociative-net theory realized with the BOUNDED cell population real CA3 and
        Kanerva's model both actually have (a fixed number of pyramidal cells / hard locations, not one per
        experienced token -- the property assoc/assoc_t's per-token key set lacks). Ramsauer et al. 2020
        ("Hopfield Networks is All You Need", ICLR 2021) frames competitive softmax retrieval against a FIXED,
        LEARNED set of stored patterns as the canonical modern-Hopfield network -- the textbook case this class
        implements directly, whereas `assoc`'s per-token keys are Ramsauer's DEGENERATE limit (one stored
        pattern per token, unbounded pattern count). `self.Kmem` is the literal "keys learned from experience"
        the roadmap names -- persistent across every position and every training example, not a per-token
        projection.

        Mechanism per position t (causal; M FIXED key prototypes shared across ALL positions/timesteps/layers-
        of-this-block; D = model width):
          z_t = LN(h_t)
          q_t = Wq(z_t)                        -- READ probe (what does the current prediction need)
          k_t = Wk(z_t)                        -- WRITE probe (which slot(s) does this token's content address)
          v_t = Wv(z_t)                        -- the value to be filed
          addr_w_t = softmax_m(k_t . Kmem_m / sqrt(D))     -- WRITE competition over the M FIXED prototypes
          addr_r_t = softmax_m(q_t . Kmem_m / sqrt(D))     -- READ competition over the SAME M prototypes
          S_t[m] = lam_m * S_{t-1}[m] + addr_w_t[m] * v_t          -- per-slot decayed Hebbian content trace
          Z_t[m] = lam_m * Z_{t-1}[m] + addr_w_t[m]                -- per-slot normalizer
          read_t = sum_m addr_r_t[m] * (S_t[m] / (Z_t[m] + eps))   -- content-addressable recall from the codebook
          delta_t = Wo(r_t * read_t), r_t = sigmoid(Wr(z_t))       -- caller adds this to the residual stream
        Genuine softmax (not a kernel feature map like linattn's phi): M is fixed and small, so a real
        normalized competition over M alternatives is cheap -- the property that makes M unbounded (assoc)
        expensive does not apply here. `lam = exp(-softplus(w))` is a PER-SLOT leak (one decay per memory slot,
        not per channel as in wkv/linattn -- a deliberate reading of --uniform-decay as "one shared consolidation
        rate across memory slots" rather than across channels, since decay here multiplies a per-slot trace).

        SPIKING-REALIZABLE (the property this class restores over assoc/assoc_t): M fixed populations under
        lateral-inhibition-style competition (softmax over a SMALL, CONSTANT set) is a standing, bounded cortical
        motif -- unlike an attention matrix that would need a new competing unit for every token ever seen. The
        division S/Z is shunting-inhibition-style divisive normalization, the SAME honest caveat already on
        record for linattn/dual-nonneg (Holt & Koch 1997, Neural Comput. 9:1001: pure somatic shunting measures
        SUBTRACTIVE not divisive on firing rate) -- the on-substrate realization of the division is a later rung,
        not claimed solved here.

        MEMORYLESS anti-cheat (shared convention, identical semantics to Assoc/LinAttn): when True, S/Z are
        NEVER accumulated across positions -- each position's read uses ONLY that position's own
        addr_w_t(*)v_t / addr_w_t as its slot trace, so no PAST position can be recalled. The generic
        sequence-level --permute anti-cheat (eval_perdepth's permute=True) also applies unmodified.
        NOT INCLUDED (honest scope): a "shuffle slot identity" structural control analogous to
        --hippo-permute-a is NOT meaningful here the way it is for HippoLayer -- HippoLayer's tau values carry a
        principled fast->slow ORDERING that permutation can meaningfully scramble, whereas `learnkey`'s M slots
        have no a-priori identity before training (an unordered learned set), so permuting their storage order
        post-training changes nothing about what any given slot has learned to represent.

        `--assoc-gate` (reused, same init-open convention as Assoc/LinAttn: Wg.weight=0, Wg.bias=+2.0 ->
        g~0.88 at init): gates read_t before Wo, identical wiring to AssocLayer's LEARNED RETRIEVAL GATE.

        Pre-norm residual block, `forward(h, memoryless=False) -> delta`, IDENTICAL contract to every other
        --recurrence class in this file, so it stacks under --n-layers and composes with --contiguous/
        --tokenizer bpe unchanged.

        HONEST SCOPE (this is UNTESTED at the time this class was written -- a prepared fallback, not a
        result): no numeric expectation is claimed. The GO gate this fallback is judged against (see --recurrence
        help + the main() report) is TWO-PART: (1) the universal per-arm bar every mechanism in this file must
        clear (margin_vs_trigram > 0.02 at deep context, AND both anti-cheats collapse), (2) SPECIFIC to this
        fallback's purpose -- mean margin_vs_trigram >= the already-measured linattn 6-seed baseline
        (+0.0505, --linattn-baseline-margin), because a fallback that merely re-clears the trigram bar without
        matching the current best deployable mechanism is not a reason to switch off linattn.

        BYTE-IDENTICAL WHEN OFF: this class and `self.learnkey_layers` are constructed ONLY when `--recurrence
        learnkey` is selected -- when not selected, defined but never instantiated, consuming ZERO init-RNG
        draws, so wkv/ssm/hippo/assoc/assoc_t/linattn are completely unaffected by this addition.
        """
        def __init__(self, D, M=64, uniform_decay=False, gate=False):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            self.Wq = nn.Linear(D, D, bias=False); self.Wk = nn.Linear(D, D, bias=False)
            self.Wv = nn.Linear(D, D, bias=False); self.Wr = nn.Linear(D, D, bias=False)
            self.Wo = nn.Linear(D, D, bias=False)
            self.Kmem = nn.Parameter(torch.randn(M, D) / math.sqrt(D))   # FIXED-COUNT learned key codebook
            self.w = nn.Parameter(torch.zeros(1 if uniform_decay else M))  # PER-SLOT leak (see docstring)
            self.scale = 1.0 / math.sqrt(D)
            self.M = M
            self.gate = gate
            if gate:
                self.Wg = nn.Linear(D, D, bias=True)
                nn.init.zeros_(self.Wg.weight); nn.init.constant_(self.Wg.bias, 2.0)   # init-open, reuse assoc-gate

        def forward(self, h, memoryless=False):        # h:[B,T,D] -> delta:[B,T,D] (pre-norm residual block)
            B, T, D = h.shape
            M = self.M
            z = self.ln(h)
            q = self.Wq(z); k = self.Wk(z); v = self.Wv(z); r = torch.sigmoid(self.Wr(z))
            addr_w_all = torch.softmax(torch.einsum("btd,md->btm", k, self.Kmem) * self.scale, dim=-1)  # [B,T,M]
            addr_r_all = torch.softmax(torch.einsum("btd,md->btm", q, self.Kmem) * self.scale, dim=-1)  # [B,T,M]
            lam = torch.exp(-torch.nn.functional.softplus(self.w))       # (M,) or (1,) per-slot decay
            S = torch.zeros(B, M, D, device=h.device); Z = torch.zeros(B, M, device=h.device)
            outs = []
            for t in range(T):
                aw = addr_w_all[:, t]                                     # [B,M] WRITE competition at t
                if memoryless:                                            # ANTI-CHEAT: current token only
                    S_r = torch.einsum("bm,bd->bmd", aw, v[:, t]); Z_r = aw
                else:
                    S = lam.unsqueeze(-1) * S + torch.einsum("bm,bd->bmd", aw, v[:, t])
                    Z = lam * Z + aw
                    S_r, Z_r = S, Z
                read = torch.einsum("bm,bmd->bd", addr_r_all[:, t], S_r / (Z_r.unsqueeze(-1) + 1e-6))
                if self.gate:
                    read = torch.sigmoid(self.Wg(z[:, t])) * read
                outs.append(self.Wo(r[:, t] * read))
            return torch.stack(outs, 1)

    class HippoAssocLayer(nn.Module):
        """STRUCTURED HiPPO SSM -> CONTENT-ADDRESSABLE LEARNED-KEY ATTENTION (--recurrence hippokey, 2026-09-05,
        own-voice-fluency arc). The LITERAL owner steer (MEMORY project_own_voice_fluency_pursue_fully_2026_09_03,
        "a structured HiPPO-style SSM -> content-addressable learned-key attention"), realized as the composition
        it names -- distinct from `learnkey` (which substituted a FIXED codebook and dropped the HiPPO SSM
        entirely) and from `assoc`/`assoc_t` (which keyed attention off the token-local residual stream z).

        WHY THIS IS A GENUINELY NEW MEMBER OF THE FAMILY, NOT A HOLLOW DUPLICATE OF THE assoc NO-GO (the crux):
        assoc/assoc_t both LOST to the -0.125 ssm floor (bag -0.347, ordered -0.147). The ordered-attention
        bound-investigation (research/findings/2026-09-03-ordered-attention-at-shared-fluency-bound-investigation-
        verdict.md) concluded content+order is NECESSARY-BUT-NOT-SUFFICIENT: the read machinery was fine, the KEYS
        were weak. That echoes the July diagnosis exactly (research/findings/2026-07-11-content-addressable-
        retrieval-needs-LEARNED-keys-...md): "the fading reservoir state is a BAD KEY", content-addressable
        retrieval is load-bearing only with keys carrying LEARNED long-range structure. assoc's keys are Wk(z_s)
        where z_s is dominated by the CURRENT token embedding + shallow context -- so assoc can match "same token
        near the same absolute position" (assoc_t's time code) but NOT "same deep multi-timescale CONTEXT". THIS
        class forms Q/K from the FIXED HiPPO multi-timescale SSM STATE x_s (a rich, order-aware summary of the
        whole prefix up to s across log-spaced timescales), keeping V as the token content Wv(z_s) -- the classic
        "match by context, retrieve the content" key/value split. It fixes BOTH diagnosed assoc failure modes at
        once: (a) BAD KEY -> the HiPPO state is a deep multi-timescale context code, not a shallow token read;
        (b) ORDER-BLINDNESS -> the state is built by literally stepping through the sequence, so it is inherently
        order-dependent (assoc's bag-of-tokens problem) WITHOUT needing assoc_t's added time-cell code.

        WHY IT PLAUSIBLY BREAKS THE TRIGRAM BOUND WHERE THE LINEAR-RECURRENCE FAMILY DID NOT: research/findings/
        2026-07-15-selective-ssm-generator-trigram-bound-... diagnosed the shared -0.125 wall as fixed-size-state
        COMPRESSION -- wkv/ssm/hippo/linattn all crush the whole prefix into a fixed-size state a read-out cannot
        losslessly invert, so they only APPROXIMATE which past token mattered, while a trigram keeps the EXACT
        identity of the last two. linattn (the current deployable mouth) crosses the trigram on the SIMPLE
        simplewiki domain (+0.0505 6/6) but FALLS BELOW it on the BROAD wikitext103 domain (2026-09-04, margin
        -0.29..-0.57 at depth>=2) -- the fixed-state compression bites hardest when the domain is broad. A causal
        softmax read keeps a per-position value for every past position (unbounded effective context, EXACT
        recall), so it is not subject to that compression bound -- and here the position it recalls is chosen by a
        DEEP multi-timescale HiPPO key, the structure a trigram cannot see. That is the unexhausted hypothesis
        this arm tests: does a multi-timescale HiPPO key make content-addressable recall load-bearing at long
        range on a broad domain?

        BIOLOGICAL ANCHOR (a real circuit, not a transformer bolted on -- the owner accepts attention-like reads
        IF bio-grounded): this is the ENTORHINAL -> CA3 pathway. The medial entorhinal cortex supplies a
        multi-timescale temporal-context / grid code -- hippocampal time cells tiling elapsed time (MacDonald et
        al. 2011), Howard & Kahana's slowly-drifting temporal-context vector (2002), and grid modules at multiple
        spatial scales (Stensola et al. 2012) -- exactly a bank of leaky integrators at log-spaced time constants,
        which IS the diagonal HiPPO-LegS multi-timescale approximation (Gu et al. 2020; the same A/B this file's
        HippoLayer already validates). CA3's recurrent collaterals then perform content-addressable autoassociative
        pattern completion (Marr 1971; Treves & Rolls 1994; Hasselmo's EC-context-cued CA3 retrieval), which
        Ramsauer et al. 2020 prove is one-shot modern-Hopfield <-> softmax-attention. So "HiPPO multi-timescale
        state (EC context) keying a content-addressable read (CA3 completion) over learned values" is the
        entorhinal-hippocampal memory circuit, the same two anchors HippoLayer and AssocLayer already carry, now
        COMPOSED as biology composes them (EC feeds the CA3 cue), not run in isolation.

        DEPLOYABILITY (honest, named not hidden): the read here is EXACT causal softmax -- O(T^2), a CEILING /
        capability instrument (like assoc/assoc_t, and like the BPTT-trained WKV that the local-rule read-out was
        only later shown to match, 2026-07-20). It is NOT yet spike-deployable. This first version answers the
        CAPABILITY question cheaply (does the HiPPO key break the bound?); IF GO, the spike-port rung is a
        HiPPO-keyed linattn kernel (feed X into LinAttnLayer's phi(q)/phi(k) fast-weight trace, inheriting its
        deployed LinAttnReadout machinery) or a fixed-slot read -- exactly the prove-on-instrument-then-port
        discipline the arc already used for wkv and linattn. IF NO-GO, we have cheaply learned the key was not the
        missing piece and the bound is deeper (objective/capacity), re-aiming the arc. Either way it is a
        first-class deliverable, deferring a METHOD, never the capability.

        A (the multi-timescale decay grid) and B (the input coupling) are FIXED buffers -- register_buffer, never
        nn.Parameter -- so NO learned recurrent credit flows through the transition dynamics (the emergence bar
        HippoLayer's docstring establishes: "only C/head learned"). Only Wq/Wk/Wv/Wo (and, with --assoc-gate, Wg)
        are learned by gradient, on top of the shared embedding upstream.

        MEMORYLESS anti-cheat (shared eval_perdepth machinery, self.memoryless): sets A_eff=0 (HiPPO carry off ->
        x_t = u_t, current token only) AND masks the read to s==t (self-only) -- both collapse the read to a pure
        current-token projection Wo(Wv(z_t)), so if deep-context NLL drops toward bigram the advantage genuinely
        came from reading the past through the HiPPO-keyed recall. The generic --permute anti-cheat (eval_perdepth
        permute=True, shuffles the prefix order upstream) applies unmodified and should degrade this arm strongly
        (both the HiPPO state and the read depend on order).

        BYTE-IDENTICAL WHEN OFF: this class and self.hippoassoc_layers are constructed ONLY when
        --recurrence hippokey is selected (see the ModuleList construction in WKV.__init__ and the RECUR=="hippokey"
        forward dispatch) -- when not selected it is defined but never instantiated, consuming ZERO init-RNG draws,
        so wkv/ssm/hippo/assoc/assoc_t/linattn/learnkey are completely unaffected by this addition at any
        --n-layers. Pre-norm residual block, forward(h, memoryless) -> delta, IDENTICAL contract to every other
        layer here, so it stacks under --n-layers and composes with --contiguous/--tokenizer bpe unchanged.
        """
        def __init__(self, D, tau_lo=1.5, tau_hi=1000.0, permute_a=False, gate=False):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            # FIXED HiPPO multi-timescale diagonal recurrence (A, B buffers) -- identical construction to HippoLayer
            # (fast->slow log-spaced decay grid + fixed random input projection; see HippoLayer's A-INIT docstring).
            tau = torch.exp(torch.linspace(math.log(tau_lo), math.log(tau_hi), D))
            A = torch.exp(-1.0 / tau)                        # FIXED diagonal decay, HiPPO-LegS-approx fast->slow
            if permute_a:
                A = A[torch.randperm(D)]                     # structural anti-cheat (shuffle channel<->tau labeling)
            self.register_buffer("A", A)
            Bmat = torch.randn(D, D) / math.sqrt(D)          # FIXED random input projection (never trained)
            self.register_buffer("B", Bmat)
            # LEARNED content-addressable read: Q/K over the HiPPO STATE (context match), V over the token content
            # z (assoc's convention -- retrieve the content that followed matching contexts). Four projections +
            # one causal softmax, exactly AssocLayer's read; the ONLY change is Q/K read x (the HiPPO state) not z.
            self.Wq = nn.Linear(D, D, bias=False)
            self.Wk = nn.Linear(D, D, bias=False)
            self.Wv = nn.Linear(D, D, bias=False)
            self.Wo = nn.Linear(D, D, bias=False)
            self.scale = 1.0 / math.sqrt(D)
            self.gate = gate                                 # --assoc-gate (learned retrieval trust gate), default OFF
            if gate:
                self.Wg = nn.Linear(D, D, bias=True)
                nn.init.zeros_(self.Wg.weight)               # g_t starts UNIFORM across channels (init-open trick)
                nn.init.constant_(self.Wg.bias, 2.0)         # sigmoid(2.0)~0.88 -> roughly open at init (reuse assoc-gate)

        def forward(self, h, memoryless=False):              # h:[B,T,D] -> delta:[B,T,D] (pre-norm residual block)
            B, T, D = h.shape
            z = self.ln(h)
            u = z @ self.B.t()                               # B u_t (fixed random input coupling, as HippoLayer)
            A_eff = torch.zeros_like(self.A) if memoryless else self.A   # ANTI-CHEAT: no carry -> current token only
            x = torch.zeros(B, D, device=h.device)
            xs = []
            for t in range(T):
                x = A_eff * x + u[:, t]                       # x_{t+1} = A x_t + B u_t (FIXED HiPPO multi-timescale state)
                xs.append(x)
            X = torch.stack(xs, 1)                            # [B,T,D] multi-timescale context code per position
            q = self.Wq(X); k = self.Wk(X); v = self.Wv(z)   # match by HiPPO context (Q/K over X), retrieve content (V over z)
            scores = torch.einsum("btd,bsd->bts", q, k) * self.scale     # [B,T,T] HiPPO-context content-similarity
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device))
            if memoryless:
                causal = torch.eye(T, dtype=torch.bool, device=h.device) # self-only: collapses read to current token
            scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
            alpha = torch.softmax(scores, dim=-1)                        # [B,T,T] causal content-addressed weights
            read = torch.einsum("bts,bsd->btd", alpha, v)               # [B,T,D] weighted associative recall
            if self.gate:
                read = torch.sigmoid(self.Wg(z)) * read                  # learned retrieval trust gate (pre-Wo, on the read)
            return self.Wo(read)

    class WKV(nn.Module):
        def __init__(self, V, D, memoryless=False, n_layers=1):
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
            # DEPTH LEVER (gap#1, 2026-07-21): the base block above (self.Wk/Wv/Wr/Wo/w/u) IS layer 0 = the exact original
            # single-layer WKV. Extra pre-norm residual layers stack ON TOP of layer 0's output (wkv branch only).
            # Constructed AFTER self.head so at n_layers=1 the ModuleList is EMPTY -> parameter-init RNG order + the forward
            # are BYTE-IDENTICAL to the original single-block model (the load-bearing reproduction guarantee).
            self.n_layers = n_layers
            self.extra = nn.ModuleList([WkvLayer(D, getattr(args, "uniform_decay", False)) for _ in range(n_layers - 1)])
            # DEPTH LEVER (2026-09-03, ssm/dual-nonneg): mirrors self.extra immediately above, but stacks
            # SsmDualNonnegLayer blocks (see that class's docstring) instead of WkvLayer blocks -- used only by
            # the --recurrence ssm --dual-nonneg branch below. Also EMPTY at n_layers=1 (range(0)), so it
            # consumes zero RNG draws at init regardless of --recurrence -> the n_layers=1 byte-identical
            # guarantee holds unconditionally, not just for the branch that happens to be selected.
            self.extra_ssm = nn.ModuleList([SsmDualNonnegLayer(D, getattr(args, "uniform_decay", False)) for _ in range(n_layers - 1)])
            # --recurrence hippo (2026-09-03): composed with --n-layers depth (see HippoLayer's docstring).
            # UNLIKE self.extra/self.extra_ssm (n_layers-1 EXTRA layers stacked on a separate inline "layer 0"
            # kept for wkv/ssm backward-compat), hippo has no legacy inline layer-0 to preserve byte-identically
            # -- ALL n_layers blocks are uniform HippoLayer instances in ONE list. Built ONLY when RECUR=="hippo"
            # (else an empty ModuleList, consuming ZERO extra RNG draws) so the wkv/ssm paths' parameter-init RNG
            # order -- hence their outputs -- is UNCHANGED by this addition at any --n-layers, not just n_layers=1.
            self.hippo_layers = nn.ModuleList([
                HippoLayer(D, tau_lo=getattr(args, "hippo_tau_lo", 1.5), tau_hi=getattr(args, "hippo_tau_hi", 1000.0),
                           permute_a=getattr(args, "hippo_permute_a", False))
                for _ in range(max(n_layers, 1))
            ]) if RECUR == "hippo" else nn.ModuleList()
            # --recurrence assoc / assoc_t (2026-09-03, assoc_t added same day): CONTENT-ADDRESSABLE associative
            # read, composed with --n-layers depth exactly like hippo_layers above (ALL n_layers blocks are
            # uniform AssocLayer instances in ONE list -- see AssocLayer's docstring for the full mechanism +
            # bio framing, including the TEMPORAL CODE section for assoc_t). Built ONLY when RECUR is "assoc" or
            # "assoc_t" (else an empty ModuleList, consuming ZERO extra RNG draws) so the wkv/ssm/hippo paths'
            # parameter-init RNG order -- hence their outputs -- is UNCHANGED by this addition at any
            # --n-layers, not just n_layers=1. `temporal=(RECUR=="assoc_t")` is the ONLY difference between the
            # two arms' AssocLayer construction (an extra fixed, non-RNG-consuming buffer, see AssocLayer.
            # __init__) -- RECUR=="assoc" (the original bag-of-tokens read) is therefore byte-identical to
            # before this addition: same Wq/Wk/Wv/Wo init order, same forward computation (temporal=False takes
            # the zt=z branch, identical to the pre-assoc_t code).
            # `gate=getattr(args, "assoc_gate", False)` (2026-09-03, --assoc-gate, see AssocLayer's LEARNED
            # RETRIEVAL GATE docstring section): default False -> every AssocLayer built here is IDENTICAL to
            # before the gate existed (no Wg, zero extra RNG draws, untouched forward). Only when --assoc-gate is
            # passed does each layer additionally build its own Wg -- composes with BOTH assoc and assoc_t.
            self.assoc_layers = nn.ModuleList([
                AssocLayer(D, temporal=(RECUR == "assoc_t"), gate=getattr(args, "assoc_gate", False))
                for _ in range(max(n_layers, 1))
            ]) if RECUR in ("assoc", "assoc_t") else nn.ModuleList()

            # --recurrence linattn (2026-09-03, DESIGN doc research/findings/2026-09-03-spiking-content-
            # addressable-read-DESIGN.md): NORMALIZED HEBBIAN FAST-WEIGHT LINEAR ATTENTION, the deployable-
            # spiking successor to ssm/dual-nonneg -- see LinAttnLayer's docstring for the full mechanism + bio
            # framing. Composed with --n-layers depth exactly like hippo_layers/assoc_layers above (ALL n_layers
            # blocks are uniform LinAttnLayer instances in ONE list). `gate=getattr(args, "assoc_gate", False)`
            # reuses the SAME learned-retrieval-gate flag AssocLayer uses (default False -> no Wg, byte-
            # identical forward). `norm=getattr(args, "linattn_norm", True)` wires --no-linattn-norm (THE KEY
            # ABLATION, default True=normalization ON). Built ONLY when RECUR=="linattn" (else an empty
            # ModuleList, consuming ZERO extra RNG draws) so the wkv/ssm/hippo/assoc/assoc_t paths'
            # parameter-init RNG order -- hence their outputs -- is UNCHANGED by this addition at any
            # --n-layers, not just n_layers=1.
            self.linattn_layers = nn.ModuleList([
                LinAttnLayer(D, uniform_decay=getattr(args, "uniform_decay", False),
                             phi=getattr(args, "linattn_phi", "elu"), gate=getattr(args, "assoc_gate", False),
                             norm=getattr(args, "linattn_norm", True),
                             div_mode=getattr(args, "linattn_div", "exact"),
                             div_g_leak=getattr(args, "linattn_div_gleak", 1e-6),
                             div_k=getattr(args, "linattn_div_k", 1.0))
                for _ in range(max(n_layers, 1))
            ]) if RECUR == "linattn" else nn.ModuleList()

            # --recurrence learnkey (2026-09-04, build-ahead fallback -- see LearnKeyLayer's docstring for the
            # full mechanism + why this is NOT a re-run of the already-NO-GO assoc/assoc_t): FIXED-CAPACITY
            # LEARNED-KEY content-addressable memory, gap#1's next mechanism class after linattn. Composed with
            # --n-layers depth exactly like hippo_layers/assoc_layers/linattn_layers above (ALL n_layers blocks
            # are uniform LearnKeyLayer instances in ONE list). `gate=getattr(args, "assoc_gate", False)` reuses
            # the SAME learned-retrieval-gate flag Assoc/LinAttn use. Built ONLY when RECUR=="learnkey" (else an
            # empty ModuleList, consuming ZERO extra RNG draws) so the wkv/ssm/hippo/assoc/assoc_t/linattn paths'
            # parameter-init RNG order -- hence their outputs -- is UNCHANGED by this addition at any --n-layers.
            self.learnkey_layers = nn.ModuleList([
                LearnKeyLayer(D, M=getattr(args, "learnkey_slots", 64),
                              uniform_decay=getattr(args, "uniform_decay", False),
                              gate=getattr(args, "assoc_gate", False))
                for _ in range(max(n_layers, 1))
            ]) if RECUR == "learnkey" else nn.ModuleList()

            # --recurrence hippokey (2026-09-05, own-voice-fluency arc): STRUCTURED HiPPO SSM -> CONTENT-
            # ADDRESSABLE LEARNED-KEY ATTENTION (see HippoAssocLayer's docstring). The LITERAL owner steer
            # ("a structured HiPPO-style SSM -> content-addressable learned-key attention"), distinct from
            # learnkey (a FIXED codebook, no HiPPO SSM) and from assoc/assoc_t (keys from the token-local
            # residual stream z, no multi-timescale context). Composed with --n-layers depth exactly like the
            # lists above (ALL n_layers blocks are uniform HippoAssocLayer instances in ONE list). `gate=getattr(
            # args, "assoc_gate", False)` reuses the SAME learned-retrieval-gate flag assoc/linattn/learnkey use;
            # tau_lo/tau_hi reuse hippokey's own args (default to HiPPO's 1.5..1000). Built ONLY when
            # RECUR=="hippokey" (else an empty ModuleList, consuming ZERO extra RNG draws) so every other arm's
            # parameter-init RNG order -- hence its outputs -- is UNCHANGED by this addition at any --n-layers.
            self.hippoassoc_layers = nn.ModuleList([
                HippoAssocLayer(D, tau_lo=getattr(args, "hippokey_tau_lo", 1.5),
                                tau_hi=getattr(args, "hippokey_tau_hi", 1000.0),
                                permute_a=getattr(args, "hippo_permute_a", False),
                                gate=getattr(args, "assoc_gate", False))
                for _ in range(max(n_layers, 1))
            ]) if RECUR == "hippokey" else nn.ModuleList()

            # --recurrence deltanet (2026-09-05, own-voice-fluency arc, DR-ladder rung 3): ERROR-CORRECTIVE
            # DELTA-RULE fast-weight WRITE on the linattn substrate (see DeltaNetLayer's docstring for the full
            # mechanism, the convergent external evidence, the bio framing, and the explicit distinction from
            # the refuted edge5-rung3 store-side write and the banked content-addressing family). Composed with
            # --n-layers depth exactly like linattn_layers above (ALL n_layers blocks are uniform DeltaNetLayer
            # instances in ONE list). Reuses --linattn-phi (same feature map -- deltanet IS linattn-with-delta-
            # write) and --assoc-gate (the SAME learned-retrieval-gate flag assoc/linattn use). Built ONLY when
            # RECUR=="deltanet" (else an empty ModuleList, consuming ZERO extra RNG draws) so every other arm's
            # (wkv/ssm/hippo/assoc/assoc_t/linattn/learnkey/hippokey) parameter-init RNG order -- hence its
            # outputs -- is UNCHANGED by this addition at any --n-layers.
            self.deltanet_layers = nn.ModuleList([
                DeltaNetLayer(D, uniform_decay=getattr(args, "uniform_decay", False),
                              phi=getattr(args, "linattn_phi", "elu"), gate=getattr(args, "assoc_gate", False),
                              beta=getattr(args, "delta_beta", 1.0),
                              key_norm=getattr(args, "delta_key_norm", "l2"))
                for _ in range(max(n_layers, 1))
            ]) if RECUR == "deltanet" else nn.ModuleList()

            # PREDICTIVE-CODING AUXILIARY OBJECTIVE (2026-09-03, --pred-aux-weight, own-voice-fluency arc). WHY:
            # the bound-investigation of the fluency arc (research/findings/2026-09-03-spiking-depth-tokens-
            # closing-fluency-gap-milestone.md and its completeness-critic) converged every --recurrence family
            # tried so far (wkv/ssm/hippo/assoc/assoc_t) on margin_vs_trigram ~ -0.13 to -0.15 at the ~20M-token
            # budget -- an ARCHITECTURE-invariant floor. The strongest same-budget EXTERNAL datapoint the search
            # surfaced is that the TRAINING OBJECTIVE, not architecture or data, is the dominant lever below
            # ~20M tokens: at a matched 10M-word budget a hybrid causal+masked-objective model reaches BLiMP
            # 0.794 vs a tuned n-gram's 0.633 and a plain causal LSTM's 0.661 (recurrence alone barely ties the
            # n-gram -- exactly this arc's own failure mode). We cannot adopt a bidirectional/MLM objective
            # outright (a causal model can never peek both sides at inference, and the deployable mouth is
            # causal + eventually spiking), so this ports the underlying INSIGHT -- richer prediction targets
            # densify the training signal -- into a strictly causal-compatible form.
            #
            # BIO FRAMING: cortical predictive coding (Rao & Ballard 1999, "Predictive coding in the visual
            # cortex", Nat Neurosci 2:79-87; see also Friston's hierarchical predictive-processing account) casts
            # cortex as continuously generating predictions of UPCOMING input at multiple levels/horizons, with
            # only the residual (prediction error) driving further processing -- not merely a single "next
            # token" read-out. This adds one FURTHER-AHEAD auxiliary read-out per --pred-aux-offsets entry
            # (default: t+2 only), each an independent nn.Linear(D, V) applied to the SAME per-position hidden
            # state `hidden` that self.head already reads (see `_out` below) -- so the shared recurrent
            # representation is pushed, via gradient, to encode structure useful for predicting further ahead
            # than the single-step causal objective alone requires, mirroring the multi-horizon character of
            # cortical prediction. ONE HEAD PER OFFSET (not one head shared across offsets): a shared read-out
            # would be asked to match two different token distributions (t+2 and t+3) from a single predicted
            # distribution at position t, an incoherent objective -- each offset gets its own linear read-out so
            # the objective stays well-posed.
            #
            # STRICTLY CAUSAL (the deployable-mouth-compatible part): hidden state at position t is built ONLY
            # from tokens 0..t (unchanged -- this addition does not touch the recurrence), so nothing about the
            # architecture or the forward pass looks into the future; only the auxiliary loss's TARGET (the
            # token actually at t+k) reaches ahead, exactly like the existing causal loss's target (t+1) already
            # does. Unlike an MLM objective, inference-time autoregressive generation (--generate) is completely
            # unaffected -- the aux heads are pure TRAINING-time regularizers on the representation, discarded at
            # generation time (generate() never calls net(x, aux=True)).
            #
            # BYTE-IDENTICAL WHEN OFF (the load-bearing guarantee, matching every other additive lever in this
            # file): `self.aux_heads` is an nn.ModuleDict built empty by default and populated ONLY when
            # `args.pred_aux_weight > 0.0`, and this construction is the LAST statement in __init__ (after
            # self.head/self.extra/self.extra_ssm/self.hippo_layers/self.assoc_layers, exactly the position
            # self.extra's own docstring establishes as required for RNG-order-preserving additive levers). When
            # --pred-aux-weight is unset/0.0, `self.aux_heads` stays a truly EMPTY ModuleDict -- zero extra
            # parameters, zero extra init-RNG draws -- so parameter init for every existing --recurrence arm is
            # completely undisturbed, and `_out()` below degenerates to exactly `return self.head(hidden)`, the
            # pre-existing return value at every one of this class's 6 return points.
            self.aux_heads = nn.ModuleDict()
            _pred_aux_w = float(getattr(args, "pred_aux_weight", 0.0) or 0.0)
            if _pred_aux_w > 0.0:
                for _k in (getattr(args, "pred_aux_offsets", None) or [2]):
                    self.aux_heads[str(int(_k))] = nn.Linear(D, V)

        def _out(self, hidden, aux):
            """Single exit point for every --recurrence branch's final per-position hidden state -> vocab
            logits, used by all 6 return points below (hippo/assoc/plateau-exact/ssm-dual-nonneg/ssm-plain/wkv)
            instead of each calling `self.head(...)` directly, so the predictive-coding auxiliary objective
            (see the docstring on `self.aux_heads` above) composes with EVERY branch from one place. `aux` is
            False at every existing call site unless the caller explicitly opts in (see build_and_train_wkv's
            training loop) -- eval/generate call `net(x)` with the forward default `aux=False`, so they are
            unaffected regardless of whether aux_heads were constructed. When aux is False, OR when aux_heads is
            empty (--pred-aux-weight unset), this is exactly `return self.head(hidden)` -- byte-identical to
            every branch's pre-existing `return self.head(...)` line."""
            logits = self.head(hidden)
            if aux and len(self.aux_heads) > 0:
                return logits, {k: ah(hidden) for k, ah in self.aux_heads.items()}
            return logits

        def forward(self, x, aux=False):
            # x: [B,T] token ids. aux=False (default, every pre-existing call site: eval_perdepth/--generate)
            # returns just the logits [B,T,V], byte-identical to before this addition. aux=True (only the
            # training loop, only when --pred-aux-weight>0) additionally returns a {offset_str: aux_logits}
            # dict from the SAME hidden state, via `_out` above -- see `self.aux_heads`'s docstring.
            B, T = x.shape
            h = self.ln(self.emb(x))                    # [B,T,D]
            if RECUR == "hippo":
                # FIXED HiPPO-structured multi-timescale recurrence + LEARNED local read-out (see HippoLayer).
                # Self-contained: does NOT touch self.Wk/Wv/Wr/w/u below (those stay unconditionally constructed
                # for wkv/ssm parity but are simply unused on this branch -- no gradient reaches them). Inserted
                # BEFORE any wkv/ssm-specific line so this is a pure insertion: when RECUR != "hippo" this `if`
                # is one skipped comparison and every line below runs byte-identically to before this change.
                hh = h
                for blk in self.hippo_layers:
                    hh = hh + blk(hh, memoryless=self.memoryless)
                return self._out(hh, aux)
            if RECUR in ("assoc", "assoc_t"):
                # CONTENT-ADDRESSABLE associative read (see AssocLayer's docstring for the full mechanism + bio
                # framing: CA3 pattern completion / cortical associative memory, Ramsauer et al. 2020's modern-
                # Hopfield<->attention equivalence; assoc_t additionally layers a hippocampal time-cell / TCM
                # "when" signal onto the read competition, see TEMPORAL CODE in the same docstring). Self-
                # contained: does NOT touch self.Wk/Wv/Wr/w/u below (unused on this branch, no gradient reaches
                # them). Inserted BEFORE any wkv/ssm-specific line, mirroring the hippo insertion immediately
                # above, so when RECUR is neither "assoc" nor "assoc_t" this is one skipped comparison and every
                # line below runs byte-identically to before this addition. RECUR=="assoc" itself is also
                # byte-identical to before assoc_t existed (see the assoc_layers construction comment above).
                hh = h
                for blk in self.assoc_layers:
                    hh = hh + blk(hh, memoryless=self.memoryless)
                return self._out(hh, aux)
            if RECUR == "linattn":
                # NORMALIZED HEBBIAN FAST-WEIGHT LINEAR ATTENTION (see LinAttnLayer's docstring). Self-
                # contained: does NOT touch self.Wk/Wv/Wr/w/u below (unused on this branch, no gradient reaches
                # them). Inserted BEFORE any wkv/ssm-specific line, mirroring the hippo/assoc insertions above,
                # so when RECUR != "linattn" this is one skipped comparison and every line below runs
                # byte-identically to before this addition.
                hh = h
                for blk in self.linattn_layers:
                    hh = hh + blk(hh, memoryless=self.memoryless)
                return self._out(hh, aux)
            if RECUR == "learnkey":
                # FIXED-CAPACITY LEARNED-KEY CONTENT-ADDRESSABLE MEMORY (see LearnKeyLayer's docstring). Self-
                # contained: does NOT touch self.Wk/Wv/Wr/w/u below (unused on this branch, no gradient reaches
                # them). Inserted BEFORE any wkv/ssm-specific line, mirroring the hippo/assoc/linattn insertions
                # above, so when RECUR != "learnkey" this is one skipped comparison and every line below runs
                # byte-identically to before this addition.
                hh = h
                for blk in self.learnkey_layers:
                    hh = hh + blk(hh, memoryless=self.memoryless)
                return self._out(hh, aux)
            if RECUR == "hippokey":
                # STRUCTURED HiPPO SSM -> CONTENT-ADDRESSABLE LEARNED-KEY ATTENTION (see HippoAssocLayer). Self-
                # contained: does NOT touch self.Wk/Wv/Wr/w/u below (unused on this branch, no gradient reaches
                # them). Inserted BEFORE any wkv/ssm-specific line, mirroring the hippo/assoc/linattn/learnkey
                # insertions above, so when RECUR != "hippokey" this is one skipped comparison and every line
                # below runs byte-identically to before this addition.
                hh = h
                for blk in self.hippoassoc_layers:
                    hh = hh + blk(hh, memoryless=self.memoryless)
                return self._out(hh, aux)
            if RECUR == "deltanet":
                # ERROR-CORRECTIVE DELTA-RULE fast-weight write on the linattn substrate (see DeltaNetLayer).
                # Self-contained: does NOT touch self.Wk/Wv/Wr/w/u below (unused on this branch, no gradient
                # reaches them). Inserted BEFORE any wkv/ssm-specific line, mirroring the hippo/assoc/linattn/
                # learnkey/hippokey insertions above, so when RECUR != "deltanet" this is one skipped comparison
                # and every line below runs byte-identically to before this addition.
                hh = h
                for blk in self.deltanet_layers:
                    hh = hh + blk(hh, memoryless=self.memoryless)
                return self._out(hh, aux)
            k = self.Wk(h); v = self.Wv(h); r = torch.sigmoid(self.Wr(h))
            wdec = torch.exp(-torch.nn.functional.softplus(self.w))       # per-channel decay in (0,1)
            if self.memoryless:
                wdec = torch.zeros_like(wdec)           # ANTI-CHEAT: no carry -> only the current token
            if RECUR == "ssm":
                # DEPTH LEVER (2026-09-03): --n-layers>1 is now ALSO implemented for --recurrence ssm
                # --dual-nonneg (the base loop below -- see the `for blk in self.extra_ssm` stacking after it).
                # --plateau-exact and the plain/--spiking-state/--nonneg-state branch at the bottom of this `if
                # RECUR == "ssm":` are UNCHANGED and still assert n_layers==1 inline, at the point each is
                # selected (they have their own realizability targets this lever does not touch).
                # SPIKING-SUBSTRATE-FAITHFUL leaky-integrator (Rung 2 de-risk): a_t = decay*a_{t-1} + v_t (a slow
                # conductance/membrane leak -- NO exp(k) weighting, NO num/den normalization = the part hard on spikes),
                # out = receptance-gated read. If GO, the spiking membrane leak realizes this state directly.
                a = torch.zeros(B, D, device=x.device); outs = []
                spiking = getattr(args, "spiking_state", False)
                sn = float(getattr(args, "state_noise", 0.0))
                _nn = getattr(args, "nonneg_state", False)
                if getattr(args, "dual_nonneg", False):
                    # DUAL non-negative: two POSITIVE leaky integrators relu(+v), relu(-v) = the plateau's ACTUAL realizable
                    # state (each a positive plateau at 0.98; the read-out uses both, no opponency difference-of-large-integrals).
                    _psg = getattr(args, "plateau_surrogate", False)
                    _pc = float(getattr(args, "plateau_sur_center", 1.0)); _psl = float(getattr(args, "plateau_sur_slope", 1.0))
                    ap2 = torch.zeros(B, D, device=x.device); an2 = torch.zeros(B, D, device=x.device); outs2 = []
                    _inz = float(getattr(args, "input_noise", 0.0))
                    # DIVISIVE-NORMALIZATION GATE (2026-09-03, additive, default OFF -- --dual-nonneg-divnorm):
                    # read once, used only in the base dual-nonneg loop below (NOT the plateau_exact branch,
                    # which returns early and is untouched). See the in-loop comment for the full justification.
                    _dnv = getattr(args, "dual_nonneg_divnorm", False)
                    _dnv_n = float(getattr(args, "divnorm_n", 2.0))
                    _dnv_sigma = float(getattr(args, "divnorm_sigma", 0.5))
                    _dnv_scale = float(getattr(args, "divnorm_scale", 1.0))
                    if getattr(args, "plateau_exact", False):
                        assert self.n_layers == 1, ("--n-layers>1 is only implemented for --recurrence ssm "
                            "--dual-nonneg's BASE loop, not --plateau-exact (a distinct exact on-bridge "
                            "transfer target -- see SsmDualNonnegLayer's docstring for the scope)")
                        # EXACT on-bridge plateau transfer (gap#1<->gap#4 convergence): the recurrence IS
                        # fused_graded_dendritic_plateau's read state -- V = relu(sigmoid(slope*(pathway_w*rate - center)) - floor),
                        # floor = sigmoid(-slope*center) so V(0)=0. The leaky-integral of V (strength absorbed by the read-out) is
                        # exactly what the on-bridge runner reads from cp_conductance_g_graded_plateau. Training end-to-end through
                        # THIS transfer makes the input map LEARN to keep values in the plateau's GRADED (non-saturating) range, so
                        # the deployed on-bridge state matches (corr->1.0) and the trigram-beating read-out applies -- the fix the
                        # post-hoc reservoir read-out cannot do (it inherits a saturating input map trained for raw integration).
                        import math as _m
                        _pxc = float(getattr(args, "px_center", 8.0)); _pxs = float(getattr(args, "px_slope", 0.33))
                        _pxw = float(getattr(args, "px_pathway_w", 16.0))
                        _floor = 1.0 / (1.0 + _m.exp(_pxs * _pxc))       # = sigmoid(-slope*center) = V at rate=0
                        for t in range(T):
                            ip = torch.relu(v[:, t]); im = torch.relu(-v[:, t])
                            if _inz > 0.0 and self.training:
                                ip = torch.relu(ip + _inz * torch.sqrt(ip + 1e-4) * torch.randn_like(ip))
                                im = torch.relu(im + _inz * torch.sqrt(im + 1e-4) * torch.randn_like(im))
                            Von = torch.relu(torch.sigmoid(_pxs * (_pxw * ip - _pxc)) - _floor)
                            Voff = torch.relu(torch.sigmoid(_pxs * (_pxw * im - _pxc)) - _floor)
                            ap2 = wdec * ap2 + Von; an2 = wdec * an2 + Voff
                            rate = torch.cat([ap2, an2], -1)
                            outs2.append(r[:, t] * self.Wo_sp(rate))
                        return self._out(torch.stack(outs2, 1), aux)
                    for t in range(T):
                        ip = torch.relu(v[:, t]); im = torch.relu(-v[:, t])
                        if _inz > 0.0 and self.training:
                            # INPUT-DELIVERY NOISE (end-to-end co-adaptation, gap#1<->gap#4): model the substrate's spiking
                            # firing-rate estimate noise (Poisson-like, std ~ sqrt(rate)) on the PER-TOKEN input BEFORE the leaky
                            # integral, so the WKV learns a NOISE-ROBUST recurrence tuned to the actual substrate input delivery.
                            ip = torch.relu(ip + _inz * torch.sqrt(ip + 1e-4) * torch.randn_like(ip))
                            im = torch.relu(im + _inz * torch.sqrt(im + 1e-4) * torch.randn_like(im))
                        if _psg:
                            # PLATEAU SURROGATE (rate-level co-adaptation): the input passes the plateau's own SIGMOID transfer
                            # (a differentiable model of fused_graded_dendritic_plateau's logistic) BEFORE integrating, so the
                            # WKV read + input map co-adapt to the plateau's realizable state. If GO -> the actual port beats.
                            ip = torch.sigmoid(_psl * (ip - _pc)); im = torch.sigmoid(_psl * (im - _pc))
                        ap2 = wdec * ap2 + ip; an2 = wdec * an2 + im
                        if _dnv:
                            # DIVISIVE-NORMALIZATION GATE (2026-09-03 convergent next mechanism after the ssm/
                            # dual-nonneg NO-GO -- research/findings/2026-09-03-spiking-mouth-ssm-dualnonneg-
                            # fluency-NO-GO-first-brain-based-only-baseline.md). DIAGNOSIS: dual-nonneg's ap2/an2
                            # are raw leaky SUMS (a_t = decay*a_{t-1} + v_t) with no accumulated denominator, so
                            # they discard RWKV's numerator/denominator normalization (wkv_t divides its running
                            # numerator by a running denominator b_t, making the read a content-weighted AVERAGE,
                            # not a raw sum -- the "cross-time competition" the diagnosis names as missing). A
                            # LITERAL per-channel temporal transfer of b_t degenerates here: dual-nonneg has no
                            # exp(k_t) content-weighting term to normalize against (b_t would reduce to
                            # decay*b+1, a channel-independent constant at steady state, i.e. a no-op divisor).
                            # So instead we transfer the SAME Carandini & Heeger (2012, Nat Rev Neurosci 13:51-62)
                            # semi-saturating divisive-normalization ratio the vision lane's `satdiv` readout
                            # already validated (BORDERLINE, 2026-09-03,
                            # research/runners/_vision_lindiscrim_readout_derisk.py::_apply_s2_norm) for the
                            # identical "affine-normalization-exhausted, need a bounded competitive ratio" failure
                            # mode: R_i = drive_i^n / (sigma^n + pool), pool = sum over a POPULATION of drive^n.
                            # PLACEMENT (justified against the RWKV analogy): applied to the dual-nonneg STATE
                            # (ap2, an2) immediately before the Wo_sp readout -- the same point in the pipeline
                            # where RWKV's wkv_t is normalized before its own Wo readout. POOL AXIS: the CHANNEL
                            # population (dim -1, size D) at the current timestep, computed SEPARATELY for the ON
                            # (ap2) and OFF (an2) populations (their own local pool, not mixed -- mirrors
                            # biological ON/OFF center-surround normalization, and matches satdiv's per-population
                            # pool convention). This makes each channel's contribution COMPETITIVE and BOUNDED: a
                            # channel that dominates its population's drive gets a near-full-scale response, a
                            # channel among many co-active channels is suppressed -- restoring content-SELECTIVE
                            # (which channel carries the informative signal right now), not merely channel-
                            # independent accumulation. This is a FORWARD-pass content-selection computation, NOT
                            # a credit-assignment rule -- explicitly DISTINCT from the already-refuted dendritic/
                            # two-compartment/BDSP/burstprop deep-credit line (that addressed hidden-credit-on-
                            # spikes via the frozen fixed-random feedback SIGNAL, not this topology; see
                            # research/findings/2026-07-22-gap4-real-issue-NOT-dendrites*.md). Default OFF (--dual-
                            # nonneg-divnorm) -> byte-identical to the pre-existing dual-nonneg path when unset.
                            _dp = torch.clamp(ap2, min=0.0).pow(_dnv_n); _dn = torch.clamp(an2, min=0.0).pow(_dnv_n)
                            _poolp = _dp.sum(-1, keepdim=True); _pooln = _dn.sum(-1, keepdim=True)
                            rp = _dnv_scale * _dp / ((_dnv_sigma ** _dnv_n) + _poolp + 1e-12)
                            rn = _dnv_scale * _dn / ((_dnv_sigma ** _dnv_n) + _pooln + 1e-12)
                            rate = torch.cat([rp, rn], -1)
                        else:
                            rate = torch.cat([ap2, an2], -1)
                        outs2.append(r[:, t] * self.Wo_sp(rate))
                    h = torch.stack(outs2, 1)             # [B,T,D] = layer-0 output (the base dual-nonneg loop, unchanged)
                    for blk in self.extra_ssm:             # DEPTH: pre-norm residual ssm/dual-nonneg layers (empty at n_layers=1)
                        h = h + blk(h)
                    return self._out(h, aux)
                assert self.n_layers == 1, ("--n-layers>1 (--recurrence ssm) is only implemented for "
                    "--dual-nonneg's base loop; the plain/--spiking-state/--nonneg-state branch below is "
                    "still n_layers==1 only")
                for t in range(T):
                    a = wdec * a + v[:, t]
                    if _nn:
                        a = torch.relu(a)                            # NON-NEGATIVE state -> plateau holds it directly (no opponency)
                    if sn > 0.0 and self.training:
                        # DEGRADE the state fidelity to simulate the on-bridge substrate (~0.55 corr): does the SpikeGPT-
                        # faithful architecture (spike-coded output) still beat the trigram from a LOSSY graded state? If GO,
                        # end-to-end training suffices on the substrate's real 0.55 state; if not, a >0.8 line-attractor is needed.
                        a = a + torch.randn_like(a) * (sn * a.detach().std())
                    if spiking:
                        # SPIKING-FAITHFUL read: firing rates are NON-NEGATIVE; encode the signed leaky state via ON/OFF
                        # rate channels [relu(a), relu(-a)] (the two-population sign code a spiking region uses). If GO,
                        # the on-bridge realization (firing-rate read of the region's slow conductance) preserves it.
                        rate = torch.cat([torch.relu(a), torch.relu(-a)], -1)
                        if getattr(args, "quantize_state", False):
                            # CO-ADAPT to the on-bridge realization: a SATURATING f-I (cap 0.5 = refractory) + STRAIGHT-
                            # THROUGH quantization to the T_STEP firing levels {0,1/ts..0.5} -- so the WKV learns a read
                            # ROBUST to the spiking noise/quantization/refractory (the ~0.11 on-bridge residual).
                            ts = float(getattr(args, "t_step", 6))
                            sat = 0.5 * torch.tanh(rate)             # saturating f-I (cap at the 0.5 refractory ceiling)
                            q = torch.round(sat * ts) / ts           # quantize to the on-bridge firing levels
                            rate = sat + (q - sat).detach()          # straight-through: forward quantized, backward smooth
                        y = r[:, t] * self.Wo_sp(rate)
                        if getattr(args, "spike_output", False):
                            # SpikeGPT-faithful: the STATE stays GRADED (no state-quantize needed), the block OUTPUT y_t is
                            # SPIKE-CODED (a per-token spike-rate). Straight-through signed quantization to 2*ts+1 firing
                            # levels (forward spiked, backward smooth) = the single per-token output binarization SpikeGPT
                            # absorbs via BPTT (non-compounding across the recurrence). This is the "spikes for I/O, graded
                            # local state" bar (biology + SpikeGPT), not the stricter self-imposed "state = firing rate".
                            ts = float(getattr(args, "t_step", 6))
                            ysat = 0.5 * torch.tanh(y)               # saturating output f-I
                            yq = torch.round(ysat * ts) / ts         # signed spike-rate levels {-0.5..0..0.5}
                            y = ysat + (yq - ysat).detach()          # straight-through
                        outs.append(y)
                    else:
                        outs.append(r[:, t] * self.Wo(a))
                return self._out(torch.stack(outs, 1), aux)
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
            h = torch.stack(outs, 1)                     # [B,T,D]  = layer-0 output (the original single block)
            for blk in self.extra:                       # DEPTH: pre-norm residual WKV layers on top (empty at n_layers=1)
                h = h + blk(h)
            return self._out(h, aux)                     # [B,T,V], or ([B,T,V], aux_logits) when aux=True

    def pad_batch(seqs):
        m = max(len(s) for s in seqs)
        X = np.zeros((len(seqs), m), dtype=np.int64); msk = np.zeros((len(seqs), m), dtype=bool)
        for i, s in enumerate(seqs):
            X[i, :len(s)] = s; msk[i, :len(s)] = True
        return torch.tensor(X, device=device), torch.tensor(msk, device=device)

    net = WKV(V, D, n_layers=getattr(args, "n_layers", 1)).to(device)
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

    # PREDICTIVE-CODING AUXILIARY OBJECTIVE (--pred-aux-weight, see self.aux_heads' docstring inside WKV.__init__
    # for the full bio framing + design rationale). use_pred_aux gates the training-loop wiring below; when it is
    # False (the default, --pred-aux-weight 0.0/unset), the training loop's forward call and loss computation are
    # spelled EXACTLY as before this addition (`fwd(X)` -> `causal_loss` -> `loss = causal_loss`), so this branch
    # is a pure byte-identical no-op. net.parameters() above already includes any constructed aux_heads
    # (Adam optimizes them jointly with everything else) -- when aux_heads is empty (weight<=0) this is an
    # empty-set no-op contribution to `params`.
    pred_aux_weight = float(getattr(args, "pred_aux_weight", 0.0) or 0.0)
    pred_aux_offsets = list(getattr(args, "pred_aux_offsets", None) or [2])
    use_pred_aux = pred_aux_weight > 0.0

    # ---- SPEED (additive, --compile; RESULT-PRESERVING): the per-step WKV recurrence launches ~2.6k tiny kernels/step
    # (the `for t in range(T)` loop x ~10 ops), so training is launch/CPU-bound (GPU ~24% idle). torch.compile fuses the
    # unrolled loop into a CUDA-graph -- but reduce-overhead needs STATIC shapes. We pad every batch to ONE global fixed
    # T (>= the longest sequence, so NO real token is ever truncated) and to a FULL batch (masked filler rows), giving a
    # single (batch, fixed_T) shape = one compiled graph. WHY this preserves the science EXACTLY (up to fp reassociation,
    # verified to NLL-parity): (a) the recurrence is strictly CAUSAL -- token t's output depends only on tokens <= t, so
    # right-padding after the real tokens cannot change any real token's output; (b) padded positions are masked out of
    # the loss (m = msk[:,1:]); (c) the masked filler ROWS add 0 to both the loss numerator and its token count, so the
    # scalar loss -- hence every gradient and Adam step -- matches the eager per-batch-max path. Off by default: the
    # eager branch below is byte-identical to the pre-optimization trainer. torch.compile is opt-in and NLL-parity-checked.
    fwd = net
    use_compile = bool(getattr(args, "compile", False))
    if use_compile and use_pred_aux:
        # NOT WIRED (honest scope limit, not a silent wrong-result risk): the aux forward path returns an extra
        # {offset: tensor} dict output that reduce-overhead's static CUDA-graph capture is not verified against;
        # rather than risk a silently-corrupted graph, refuse the combination until it is explicitly de-risked.
        # Neither the CPU smoke nor the ready-to-fire GPU commands for this lever use --compile, so this does not
        # block the arc; it only guards a combination nobody has exercised yet.
        raise NotImplementedError(
            "--compile + --pred-aux-weight>0 is not wired yet (the aux forward returns an extra dict output the "
            "reduce-overhead CUDA-graph capture path has not been verified against) -- use one or the other.")
    Xbuf = None
    if use_compile:
        fixed_T = max((len(s) for s in seqs), default=2)
        if int(getattr(args, "compile_fixed_t", 0) or 0):
            assert args.compile_fixed_t >= fixed_T, (
                f"--compile-fixed-t {args.compile_fixed_t} < longest training sequence {fixed_T}: padding to a smaller "
                f"T would TRUNCATE tokens (NOT result-preserving). Use >= {fixed_T}, or drop the flag to auto-size.")
            fixed_T = int(args.compile_fixed_t)
        Nseq = len(seqs)
        # pre-pad ALL sequences to [N, fixed_T] on-device ONCE (removes the per-batch numpy build + H2D copy per step)
        Xall = torch.zeros((Nseq, fixed_T), dtype=torch.int64, device=device)
        Mall = torch.zeros((Nseq, fixed_T), dtype=torch.bool, device=device)
        for _i2, _s in enumerate(seqs):
            _L2 = len(_s)
            Xall[_i2, :_L2] = torch.tensor(_s, dtype=torch.int64, device=device)
            Mall[_i2, :_L2] = True
        # persistent static input buffer so the CUDA-graph replays against a fixed address (cudagraph-safe)
        Xbuf = torch.zeros((args.batch, fixed_T), dtype=torch.int64, device=device)
        fwd = torch.compile(net, mode="reduce-overhead")
        print(f"    [compile] torch.compile(reduce-overhead) ON  fixed_T={fixed_T}  batch={args.batch}  (static shape)", flush=True)

    for ep in range(args.epochs):
        order = rng.permutation(len(seqs))
        _ep_loss = torch.zeros((), device=device); _ep_n = 0   # on-device accum -> ONE .item() sync per epoch (was per step)
        for i in range(0, len(seqs), args.batch):
            idx_b = order[i:i+args.batch]
            if use_compile:
                # gather a STATIC (batch, fixed_T) block; pad the final partial batch to a full `batch` with all-masked
                # filler rows (they contribute 0 loss and 0 count -> identical scalar loss -> identical gradients)
                ib = torch.as_tensor(np.asarray(idx_b), dtype=torch.long, device=device)
                nb = int(ib.shape[0]); pad_rows = args.batch - nb
                if pad_rows > 0:
                    ib = torch.cat([ib, torch.zeros(pad_rows, dtype=torch.long, device=device)])
                Xbuf.copy_(Xall[ib])                          # into the fixed-address buffer (cudagraph-safe)
                msk = Mall[ib].clone()
                if pad_rows > 0:
                    msk[nb:] = False                          # mask the duplicated filler rows out of the loss
                X = Xbuf
            else:
                batch = [seqs[j] for j in idx_b]
                X, msk = pad_batch(batch)                     # EAGER default path: byte-identical to the pre-opt trainer
            if use_pred_aux:
                full_logits, aux_logits_by_off = fwd(X, aux=True)
            else:
                full_logits = fwd(X)                          # UNCHANGED call site when the aux objective is off
            logits = full_logits[:, :-1]                      # predict token t+1 from context 0..t
            tgt = X[:, 1:]; m = msk[:, 1:]
            L = lossf(logits.reshape(-1, V), tgt.reshape(-1)).reshape(tgt.shape)
            causal_loss = (L * m).sum() / m.sum().clamp(min=1)
            loss = causal_loss
            if use_pred_aux:
                # ADDITIVE PREDICTIVE-CODING AUXILIARY LOSS (see self.aux_heads' docstring in WKV.__init__ for
                # the full bio framing): for each --pred-aux-offsets entry k, predict the token at t+k from the
                # SAME per-position hidden state the causal head reads (full_logits' aux counterpart), masked +
                # averaged exactly like the causal loss above, then averaged across offsets so --pred-aux-weight
                # is a single interpretable coefficient regardless of how many offsets are requested.
                _aux_terms = []
                _Tf = full_logits.shape[1]
                for _k_str, _ah_logits in aux_logits_by_off.items():
                    _k = int(_k_str)
                    if _Tf <= _k:
                        continue                              # this batch's (padded) T is too short for offset k
                    _pred_k = _ah_logits[:, :_Tf - _k]         # hidden@t predicts the token at t+k
                    _tgt_k = X[:, _k:]; _m_k = msk[:, _k:]
                    _Lk = lossf(_pred_k.reshape(-1, V), _tgt_k.reshape(-1)).reshape(_tgt_k.shape)
                    _aux_terms.append((_Lk * _m_k).sum() / _m_k.sum().clamp(min=1))
                if _aux_terms:
                    aux_loss = sum(_aux_terms) / len(_aux_terms)
                    loss = causal_loss + pred_aux_weight * aux_loss
            opt.zero_grad(); loss.backward(); opt.step()
            _ep_loss += causal_loss.detach(); _ep_n += 1      # reported mean_train_loss stays the CAUSAL
                                                                # component only, for fair cross-config comparison
        print(f"    [train] epoch {ep+1}/{args.epochs} mean_train_loss={(_ep_loss/max(1,_ep_n)).item():.4f}", flush=True)
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
    ap.add_argument("--tokenizer", choices=["word", "bpe"], default="word",
                    help="word (default) = per-word top-K vocab, byte-identical to pre-swap behavior. bpe = load a "
                         "pretrained subword BPE tokenizer (--bpe-path); V is then the TOKENIZER's vocab_size "
                         "(--vocab is ignored) so arbitrary chat-topic corpora become representable (measured 0% "
                         "hard-OOV for wkv_bpe8k.json) instead of <unk>-riddled word-salad.")
    ap.add_argument("--bpe-path", dest="bpe_path", type=str, default=DEFAULT_BPE_PATH,
                    help="(--tokenizer bpe) path to the pretrained BPE merges+vocab JSON (sim.bpe_tokenizer.BPETokenizer.load).")
    ap.add_argument("--n-sentences", type=int, default=200000)
    ap.add_argument("--max-train-sents", type=int, default=60000)
    ap.add_argument("--max-eval-sents", type=int, default=3000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=1,
                    help="DEPTH LEVER (gap#1): stack N pre-norm residual layers. 1 = the original single block "
                         "(byte-identical). >1 implemented for --recurrence wkv (the baseline branch) AND for "
                         "--recurrence ssm --dual-nonneg's base loop (2026-09-03, SsmDualNonnegLayer); other ssm "
                         "sub-paths (--plateau-exact, plain/--spiking-state/--nonneg-state) remain n_layers==1 only.")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pred-aux-weight", dest="pred_aux_weight", type=float, default=0.0,
                    help="PREDICTIVE-CODING AUXILIARY OBJECTIVE (2026-09-03, own-voice-fluency arc -- see the "
                         "WKV.__init__ 'PREDICTIVE-CODING AUXILIARY OBJECTIVE' docstring for the full bio "
                         "framing + design rationale). ADDITIVE to the causal next-token loss: "
                         "total_loss = causal_loss + pred_aux_weight * aux_loss, where aux_loss is a FURTHER-"
                         "AHEAD token-prediction loss (--pred-aux-offsets, default t+2) read off the SAME "
                         "per-position hidden state the causal head reads -- a causal-compatible stand-in for "
                         "cortical predictive coding (Rao & Ballard 1999): the cortex continuously predicts "
                         "UPCOMING input at multiple horizons, not just the single next token. Composes with "
                         "EVERY --recurrence choice (wkv/ssm/hippo/assoc/assoc_t) unmodified -- it only adds an "
                         "extra linear read-out head + an extra loss term, never touches the recurrence itself. "
                         "Default 0.0 = OFF = byte-identical to before this flag existed: the auxiliary head is "
                         "not even constructed (zero extra init-RNG draws) and the training loop's forward call "
                         "is spelled identically to the pre-existing code (see build_and_train_wkv).")
    ap.add_argument("--pred-aux-offsets", dest="pred_aux_offsets", type=int, nargs="+", default=[2],
                    help="(--pred-aux-weight>0 only) which future offsets k to predict (token at t+k from the "
                         "hidden state at position t), one independent linear read-out head per offset (NOT one "
                         "shared head across offsets -- a shared head would be asked to match two different "
                         "target distributions from one prediction, an incoherent objective). Default [2] = "
                         "predict 2 steps ahead only; pass e.g. --pred-aux-offsets 2 3 to add a t+3 head too.")
    ap.add_argument("--freeze-emb", dest="freeze_emb", action="store_true",
                    help="Rung 1b de-risk: freeze the input embedding at random init (only WKV+head learn) = the frozen "
                         "emergent-code regime; GO => the recurrence does not need an LM-learned input.")
    ap.add_argument("--input", choices=["learned", "ppmi"], default="learned",
                    help="learned = LM-trained embedding (Rung 1a); ppmi = EMERGENT unsupervised PPMI co-occurrence codes, "
                         "frozen (Rung 1b, the gap#1<->gap#4 convergence).")
    ap.add_argument("--ppmi-window", type=int, default=5)
    ap.add_argument("--recurrence", choices=["wkv", "ssm", "hippo", "assoc", "assoc_t", "linattn", "learnkey", "hippokey", "deltanet"], default="wkv",
                    help="wkv = full RWKV linear-attention (num/den normalized); ssm = spiking-substrate-faithful "
                         "leaky-integrator (a_t=decay*a_{t-1}+v_t, no normalization = the Rung 2 spiking-port form); "
                         "hippo = FIXED diagonal HiPPO-structured multi-timescale recurrence (A=fixed log-spaced "
                         "decay grid, B=fixed random input projection) + LEARNED local read-out C (see HippoLayer) "
                         "-- no learned recurrent credit through the transition dynamics, composable with --n-layers; "
                         "assoc = CONTENT-ADDRESSABLE associative read (learned-key causal attention, see "
                         "AssocLayer) -- EXACT recall of a specific past position via learned Wq/Wk/Wv/Wo, framed "
                         "as hippocampal CA3 pattern-completion / cortical associative memory (Ramsauer et al. "
                         "2020's modern-Hopfield<->attention equivalence), composable with --n-layers. This "
                         "bag-of-tokens read has NO signal that distinguishes token ORDER (diagnosed 2026-09-03: "
                         "it underfit the recurrences it was meant to surpass, ~4.79 vs ~4.36 train loss at "
                         "scale); assoc_t = the SAME associative read PLUS a fixed hippocampal time-cell / "
                         "Howard-Kahana temporal-context 'when' signal (sinusoidal, non-learned buffer) added to "
                         "the Q/K projections only (not V) so the read competition becomes order-sensitive -- "
                         "see AssocLayer's TEMPORAL CODE docstring section. RECUR=='assoc' itself is untouched "
                         "(byte-identical) by assoc_t's addition; use --recurrence assoc for the original "
                         "bag-of-tokens ablation arm. linattn = NORMALIZED HEBBIAN FAST-WEIGHT LINEAR ATTENTION "
                         "(see LinAttnLayer + research/findings/2026-09-03-spiking-content-addressable-read-"
                         "DESIGN.md) -- an O(T) recurrent real-valued D x D outer-product KV trace + running "
                         "denominator, read by phi(q)^T M / phi(q)^T zden (content-weighted, softmax-free); "
                         "the deployable-spiking successor to ssm/dual-nonneg, and a strict generalization of "
                         "wkv (restrict M to its diagonal, phi=exp, Wq=Wk=I -> degenerates to wkv's num/den). "
                         "learnkey = FIXED-CAPACITY LEARNED-KEY content-addressable memory (see LearnKeyLayer + "
                         "2026-09-04 build-ahead) -- a small FIXED bank of M learned key prototypes (a real "
                         "inspectable codebook, --learnkey-slots) queried by genuine softmax competition (O(T*M), "
                         "M constant, unlike assoc's O(T^2) per-token keys) and written via a decayed per-slot "
                         "Hebbian trace; the modern-Hopfield-with-FIXED-LEARNED-patterns case Ramsauer et al. "
                         "2020 describes, distinct from assoc/assoc_t's already-NO-GO per-token-key degenerate "
                         "case. Prepared as the NEXT mechanism class after linattn -- a ready-to-fire fallback "
                         "if the linattn production-scale sweep plateaus, not yet run at scale. "
                         "hippokey = STRUCTURED HiPPO SSM -> CONTENT-ADDRESSABLE LEARNED-KEY ATTENTION (see "
                         "HippoAssocLayer, 2026-09-05) -- the LITERAL owner steer: a FIXED HiPPO multi-timescale "
                         "diagonal SSM produces a per-position multi-timescale context code x_s, and a causal "
                         "softmax read forms Q/K over x_s (match by deep context) while V stays the token content "
                         "z (retrieve what followed matching contexts). Distinct from learnkey (a FIXED codebook, "
                         "no HiPPO) and from assoc/assoc_t (keys from the shallow token-local z, the 'bad key' the "
                         "July record diagnosed) -- it fixes BOTH the bad-key and the order-blindness that sat "
                         "assoc at the -0.147 bound, and its full per-position softmax recall is not subject to "
                         "the fixed-state compression trigram-bound the linear-recurrence family (wkv/ssm/hippo/"
                         "linattn) shares. Bio anchor: entorhinal multi-timescale context -> CA3 pattern "
                         "completion. Exact-softmax CEILING instrument (O(T^2), spike-port is a named next rung). "
                         "deltanet = ERROR-CORRECTIVE DELTA-RULE fast-weight WRITE on the linattn substrate (see "
                         "DeltaNetLayer, 2026-09-05, DR-ladder rung 3) -- a WRITE-RULE fix to linattn's SAME KV "
                         "trace (erase-before-write: retrieve the value currently bound to the incoming key, "
                         "subtract it, write only the residual; L2-normalized keys + linattn's learned per-channel "
                         "decay; raw read like the DeltaNet family). NOT a new content-addressing key and NOT the "
                         "banked assoc/learnkey family -- ONLY the M-update line differs from linattn (Widrow-Hoff; "
                         "Schlag 2021, Yang et al. Gated DeltaNet 2024, RWKV-7 2025). Targets linattn's measured "
                         "interference/unbounded-norm failure on the broad wikitext-103 domain; the linattn "
                         "--no-linattn-norm arm is the write-rule isolation control.")
    ap.add_argument("--assoc-gate", dest="assoc_gate", action="store_true",
                    help="(--recurrence assoc / assoc_t only) LEARNED RETRIEVAL GATE (2026-09-03, default OFF, "
                         "see AssocLayer's LEARNED RETRIEVAL GATE docstring section): a per-channel, input-"
                         "conditioned trust gate g_t=sigmoid(Wg(z_t)) that scales the associative read BEFORE Wo "
                         "(delta_t = Wo(g_t*read_t)) instead of appending it UNGATED, the fix the 2026-07-11 "
                         "learned-keys de-risk named for the raw read's informative-but-noisy net-cost-over-base "
                         "problem. Wg is initialized near-OPEN (weight=0, bias=+2.0, sigmoid(2.0)~0.88) so training "
                         "only has to learn where to CLOSE it. Off by default -> Wg is not even constructed, so "
                         "assoc/assoc_t are byte-identical to before this flag existed; composes with both arms. "
                         "Also reused by --recurrence linattn (LinAttnLayer), same init-open semantics.")
    ap.add_argument("--linattn-phi", dest="linattn_phi", choices=["elu", "relu", "exp", "sparse"], default="elu",
                    help="(--recurrence linattn only) the non-negative feature map phi(.) applied to Q/K before "
                         "the outer-product write + content-weighted read (see LinAttnLayer's docstring). "
                         "elu = elu(x)+1 (default, Katharopoulos et al. 2020 Eq.7); relu = relu(x)+1e-3; "
                         "exp = a numerically-stabilized exp(x-max) (RWKV-like, sharper matching); sparse = a "
                         "k-winners-take-all top-(D/8) rectified key (a hard content match, biologically a "
                         "sparse pattern-separated DG->CA3-style key -- the WTA-Spiking-Transformer's "
                         "sparse-softmax limit, DESIGN doc Sec 2/5b).")
    ap.add_argument("--no-linattn-norm", dest="linattn_norm", action="store_false", default=True,
                    help="(--recurrence linattn only) THE KEY ABLATION: drop the '/ (den_t + eps)' division so "
                         "the read is the raw unnormalized outer-product sum num_t, instead of the "
                         "content-weighted average num_t/den_t. Tests whether the restored RWKV-style "
                         "numerator/denominator normalization -- not merely the outer-product q.k widening -- "
                         "is what is load-bearing (research/findings/2026-09-03-spiking-content-addressable-"
                         "read-DESIGN.md Sec 6, the cheapest decisive CPU experiment). Default (flag unset) = "
                         "normalization ON = args.linattn_norm=True = the design's primary arm.")
    ap.add_argument("--learnkey-slots", dest="learnkey_slots", type=int, default=64,
                    help="(--recurrence learnkey only) M, the FIXED number of learned key/memory-slot "
                         "prototypes in the codebook (see LearnKeyLayer). Independent of sequence length T and "
                         "of --d-model D -- this is the whole point (spiking-realizable = a bounded population, "
                         "unlike assoc's per-token O(T) key count). Default 64.")
    ap.add_argument("--delta-beta", dest="delta_beta", type=float, default=1.0,
                    help="(--recurrence deltanet only) beta, the delta-rule write strength in "
                         "M_t = lam*M_{t-1} + beta*k_hat (x) (v - v_old) (see DeltaNetLayer). Default 1.0 = the "
                         "canonical unit-step Widrow-Hoff / exact error correction (reading the just-written key "
                         "back yields v exactly at beta=1, ||k_hat||=1). A FIXED scalar -- adds NO parameters, so "
                         "deltanet stays byte-identical-when-off and a clean structural sibling of linattn.")
    ap.add_argument("--delta-key-norm", dest="delta_key_norm", choices=["l2", "none"], default="l2",
                    help="(--recurrence deltanet only) key normalization for the delta write/erase. l2 (default) "
                         "= unit-normalize phi(k) so the erase is an EXACT projection removal (the DeltaNet-"
                         "faithful form with bounded state, Schlag 2021 / Yang et al. 2024). none = use phi(k) "
                         "unnormalized (the build scope's literal formula S_t <- S_{t-1}diag(w) + "
                         "beta*(v - S_{t-1}phi(k))phi(k)^T) -- error-correcting in direction but not unit-gain; "
                         "offered as an ablation, l2 is the principled + literature-faithful default.")
    ap.add_argument("--linattn-baseline-margin", dest="linattn_baseline_margin", type=float, default=0.0505,
                    help="(--recurrence learnkey/deltanet, REPORTING) the already-measured linattn 6-seed mean "
                         "deep-context margin_vs_trigram (research/findings/2026-09-03-OPEN-FLUENCY-"
                         "BREAKTHROUGH-linattn-deployable-spiking-mouth-beats-trigram-6of6.md: mean +0.0505, "
                         "min +0.039, max +0.060). Printed alongside each learnkey seed's own margin so the "
                         "report answers the actual question this fallback exists for -- not just 'does it beat "
                         "trigram' but 'is it AT LEAST AS GOOD AS the mechanism already in production candidacy' "
                         "-- without requiring a fresh linattn run in the same invocation. Purely a report-time "
                         "comparison constant; does not affect training or any other --recurrence choice.")
    ap.add_argument("--linattn-div", dest="linattn_div", choices=["exact", "shunt"], default="exact",
                    help="(--recurrence linattn only) SPIKE-NATIVE num/den REALIZATION (2026-09-03, research/"
                         "findings/2026-09-03-linattn-spike-native-normalization-DESIGN.md Sec 3e/4). "
                         "exact (default) = today's graded host divide num/(den+eps), BYTE-IDENTICAL to every "
                         "run before this flag existed. shunt = the Carandini-Heeger conductance-divisive-gain "
                         "rate-model form num/(g_leak + k*den) (--linattn-div-gleak/--linattn-div-k); at the "
                         "default g_leak=1e-6, k=1.0 this is algebraically IDENTICAL to exact (both reduce to "
                         "num/(den+1e-6)) -- the Tier-1 CPU de-risk (research/runners/"
                         "_linattn_shunt_gain_tier1_derisk.py) instead swaps the READ of an ALREADY-TRAINED "
                         "checkpoint (LinAttnReadout) and varies the f-I squash/quantization/g_leak/k around "
                         "that point; this training-side flag exists so a Tier-2 read-in-the-loop retrain can "
                         "reuse the identical mechanism without another edit.")
    ap.add_argument("--linattn-div-gleak", dest="linattn_div_gleak", type=float, default=1e-6,
                    help="(--linattn-div shunt only) g_leak (sigma): the read neuron's leak conductance -- the "
                         "shunt read's own epsilon. Sweeping this UP tests sigma-domination (the 'the clamp "
                         "owned 97%%' trap, CLAUDE.md): a den-driven divisor should track den, not sit at a "
                         "g_leak-dominated fixed gain.")
    ap.add_argument("--linattn-div-k", dest="linattn_div_k", type=float, default=1.0,
                    help="(--linattn-div shunt only) k: the norm-neuron-rate -> shunt-conductance scale factor "
                         "in num/(g_leak + k*den).")
    ap.add_argument("--hippo-tau-lo", dest="hippo_tau_lo", type=float, default=1.5,
                    help="(--recurrence hippo) fastest time constant (steps) in the log-spaced HiPPO-LegS-approx decay grid")
    ap.add_argument("--hippo-tau-hi", dest="hippo_tau_hi", type=float, default=1000.0,
                    help="(--recurrence hippo) slowest time constant (steps) in the log-spaced decay grid")
    ap.add_argument("--hippokey-tau-lo", dest="hippokey_tau_lo", type=float, default=1.5,
                    help="(--recurrence hippokey) fastest time constant (steps) in the HiPPO multi-timescale decay "
                         "grid whose state keys the content-addressable read (see HippoAssocLayer)")
    ap.add_argument("--hippokey-tau-hi", dest="hippokey_tau_hi", type=float, default=1000.0,
                    help="(--recurrence hippokey) slowest time constant (steps) in the HiPPO decay grid keying the read")
    ap.add_argument("--hippo-permute-a", dest="hippo_permute_a", action="store_true",
                    help="ANTI-CHEAT (structural, per the task spec): shuffle the per-channel tau assignment (same "
                         "multiset of decay rates, different channel labeling) -- distinct from eval_perdepth's "
                         "generic sequence-level --permute anti-cheat. NOTE: the July gate's own runner docstring "
                         "for this control (_ssm_fixed_structured_reservoir_derisk.py, kind='permuted') calls it a "
                         "SANITY control expected NOT to change a linear-readout reservoir's result, not a collapse.")
    ap.add_argument("--spiking-state", dest="spiking_state", action="store_true",
                    help="(ssm only) read the leaky state via NON-NEGATIVE ON/OFF firing-rate channels [relu(a),relu(-a)] "
                         "= the spiking firing-rate constraint; GO => the on-bridge firing-rate read preserves deep context.")
    ap.add_argument("--quantize-state", dest="quantize_state", action="store_true")
    ap.add_argument("--state-noise", dest="state_noise", type=float, default=0.0, help="degrade state fidelity (train-time noise) to simulate the on-bridge substrate")
    ap.add_argument("--input-noise", dest="input_noise", type=float, default=0.0, help="Poisson-like input-delivery noise (end-to-end co-adaptation to the substrate)")
    ap.add_argument("--plateau-surrogate", dest="plateau_surrogate", action="store_true", help="apply the plateau sigmoid transfer to the input (rate-level co-adaptation to the plateau)")
    ap.add_argument("--plateau-exact", dest="plateau_exact", action="store_true", help="EXACT on-bridge plateau transfer as the recurrence (gap#1<->gap#4): V=relu(sigmoid(slope*(pathway_w*rate-center))-floor); train end-to-end so the input map learns the plateau's GRADED range -> the deployed on-bridge read-out matches (corr->1) and beats the trigram (the post-hoc reservoir read-out cannot)")
    ap.add_argument("--px-center", dest="px_center", type=float, default=8.0, help="(plateau-exact) logistic center in weight units (match the on-bridge graded_plateau_center)")
    ap.add_argument("--px-slope", dest="px_slope", type=float, default=0.33, help="(plateau-exact) logistic slope (match graded_plateau_slope)")
    ap.add_argument("--px-pathway-w", dest="px_pathway_w", type=float, default=16.0, help="(plateau-exact) input->plateau coincidence weight (match the on-bridge per_pop pathway_w)")
    ap.add_argument("--plateau-sur-center", dest="plateau_sur_center", type=float, default=1.0)
    ap.add_argument("--plateau-sur-slope", dest="plateau_sur_slope", type=float, default=1.0)
    ap.add_argument("--dual-nonneg", dest="dual_nonneg", action="store_true", help="two positive leaky integrators = the plateau realizable state")
    ap.add_argument("--dual-nonneg-divnorm", dest="dual_nonneg_divnorm", action="store_true",
                    help="(--dual-nonneg only, base loop -- NOT --plateau-exact) Carandini & Heeger divisive-"
                         "normalization gate on the ON/OFF dual-nonneg state before the Wo_sp readout: R_i = "
                         "drive_i^n / (sigma^n + pool), pool = sum over the CHANNEL population of drive^n, "
                         "applied separately to ap2 (ON) and an2 (OFF). Restores RWKV-style content-addressed "
                         "normalization dual-nonneg otherwise lacks (2026-09-03 convergent-mechanism finding, "
                         "the SAME functional form as the vision lane's --s2-norm satdiv). FORWARD-pass content-"
                         "selection only -- not credit assignment. Default OFF -> byte-identical to the "
                         "pre-existing dual-nonneg path.")
    ap.add_argument("--divnorm-n", dest="divnorm_n", type=float, default=2.0,
                    help="(--dual-nonneg-divnorm only) exponent n in drive^n/(sigma^n+pool) (Heeger 1992 fits "
                         "~2; matches the vision lane's --s2-satdiv-n default).")
    ap.add_argument("--divnorm-sigma", dest="divnorm_sigma", type=float, default=0.5,
                    help="(--dual-nonneg-divnorm only) semi-saturation constant sigma (matches the vision "
                         "lane's --s2-satdiv-sigma default).")
    ap.add_argument("--divnorm-scale", dest="divnorm_scale", type=float, default=1.0,
                    help="(--dual-nonneg-divnorm only) output rescale so the bounded ratio lands in the "
                         "read-out's useful range (matches the vision lane's --s2-satdiv-scale default).")
    ap.add_argument("--nonneg-state", dest="nonneg_state", action="store_true", help="rectified non-negative leaky state a=relu(decay*a+v) -> the dendritic plateau holds it directly, no ON/OFF opponency")
    ap.add_argument("--spike-output", dest="spike_output", action="store_true",
                    help="SpikeGPT-faithful: GRADED state + SPIKE-CODED output y_t (straight-through), trained end-to-end")
    ap.add_argument("--uniform-decay", dest="uniform_decay", action="store_true",
                    help="(ssm only) ONE shared decay across channels = the substrate's uniform NMDA tau (simplifies the "
                         "on-bridge realization); GO => no per-neuron tau array is needed on the bridge.")
    ap.add_argument("--save-ssm", dest="save_ssm", type=str, default=None,
                    help="save the trained SSM weights to <path>_seed<N>.npz (for the on-bridge realization).")
    ap.add_argument("--gen-prompts", dest="gen_prompts", type=int, default=1, help="number of distinct prompts to generate from")
    ap.add_argument("--generate", type=int, default=0, help="autoregressive: generate N tokens after training")
    ap.add_argument("--contiguous", action="store_true", help="CONTIGUOUS multi-sentence stories (R4 cross-sentence long-range test)")
    ap.add_argument("--max-len", type=int, default=48, help="(--contiguous) max tokens per story")
    ap.add_argument("--compile", action="store_true",
                    help="SPEED (additive, RESULT-PRESERVING): torch.compile(net, mode='reduce-overhead') over STATIC "
                         "shapes -- every batch is padded to ONE global fixed T (>= the longest sequence, so NO token is "
                         "truncated) and to a full batch (masked filler rows), fusing the per-step WKV recurrence kernels "
                         "(~1.7x measured, GPU-bound instead of launch-bound). OFF by default so the eager path stays "
                         "byte-identical; --compile is verified NLL-parity to the eager path (GO verdict identical).")
    ap.add_argument("--compile-fixed-t", dest="compile_fixed_t", type=int, default=0,
                    help="(--compile) force the fixed padded T instead of auto-sizing to the longest training sequence; "
                         "MUST be >= the longest sequence (asserted) or real tokens would be truncated.")
    ap.add_argument("--tok-cache", dest="tok_cache", action="store_true", default=True,
                    help="(--tokenizer bpe only, default ON) SPEED (additive, RESULT-PRESERVING): tokenize the "
                         "full sentence pool ONCE (not per-seed -- vocab.ids(s) is seed-independent, only the "
                         "tr/ev/dev SPLIT is seed-dependent) and persist the token-id lists to a content-keyed "
                         "disk cache under data/corpus/.tokcache/ so a FRESH PROCESS (a new --seeds run) skips "
                         "re-tokenization entirely instead of re-paying the ~5-6min BPE pass. Verified "
                         "byte-identical to the per-seed path (--no-tok-cache).")
    ap.add_argument("--no-tok-cache", dest="tok_cache", action="store_false",
                    help="disable the once+disk tokenization cache; revert to per-seed re-tokenization.")
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(args.corpus).exists():
        args.corpus = "data/corpus/tinystories.txt"
    sents = (load_stories(args.corpus, args.n_sentences, max_len=args.max_len)
             if getattr(args, "contiguous", False) else load_sentences(args.corpus, args.n_sentences))
    t0 = time.time(); per_seed = {}
    # SPEED (additive, result-preserving): the BPE adapter is seed-INDEPENDENT (it just wraps the loaded tokenizer),
    # so build it ONCE and reuse across seeds -> its per-word tokenization cache persists, making seeds 2..N tokenize
    # at cache-hit speed (the whole sentence pool is shared across seeds). Identical vocab/ids to per-seed creation.
    _bpe_vocab = _BPEVocabAdapter(BPETokenizer.load(args.bpe_path)) if args.tokenizer == "bpe" else None

    # SPEED (additive, RESULT-PRESERVING, --tok-cache, default ON for BPE): `vocab.ids(s)` is SEED-INDEPENDENT (a
    # pure function of the sentence + the loaded tokenizer) -- only the tr/ev/dev SPLIT below is seed-dependent.
    # Tokenize the WHOLE sentence pool ONCE here and try a content-keyed disk cache first, so a fresh PROCESS (a
    # new --seeds invocation) skips the ~5-6min BPE pass entirely instead of re-paying it every run.
    # --no-tok-cache reverts to the original per-seed `vocab.ids(s)` calls in the loop below (both paths verified
    # byte-identical).
    _sents_ids_all = None
    if args.tokenizer == "bpe" and getattr(args, "tok_cache", True):
        _key = _tokcache_key(args)
        _tl0 = time.time()
        _cached = _tokcache_load(_key)
        if _cached is not None and len(_cached) == len(sents):
            _sents_ids_all = _cached
            print(f"    [tok-cache] HIT   key={_key} n={len(sents)} loaded in {time.time() - _tl0:.2f}s  "
                  f"{TOKCACHE_DIR / (_key + '.npz')}", flush=True)
        else:
            if _cached is not None:
                print(f"    [tok-cache] STALE key={_key} (cached len {len(_cached)} != {len(sents)}) -- re-tokenizing", flush=True)
            else:
                print(f"    [tok-cache] MISS  key={_key} -- tokenizing {len(sents)} sentences...", flush=True)
            _tt0 = time.time()
            _sents_ids_all = [_bpe_vocab.ids(s) for s in sents]
            _tokcache_save(_key, _sents_ids_all)
            print(f"    [tok-cache] saved key={_key} in {time.time() - _tt0:.1f}s  {TOKCACHE_DIR / (_key + '.npz')}", flush=True)

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(sents)); cut = int(0.85 * len(sents))
        tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
        ev = [sents[i] for i in idx[cut:]][:args.max_eval_sents]
        dev = tr[-min(2000, len(tr)//5):]                # held-out dev for trigram lambda tuning
        if args.tokenizer == "bpe":                      # additive subword swap -- see _BPEVocabAdapter docstring
            vocab = _bpe_vocab                           # ONE adapter (cache persists across seeds); ids are identical
        else:
            vocab = Vocab.build(tr, V=args.vocab)         # default path, UNCHANGED
        V = vocab.size
        if _sents_ids_all is not None:
            # --tok-cache: index the seed-independent token ids instead of re-tokenizing. Identical content/order
            # to `[vocab.ids(s) for s in tr]` etc. because _sents_ids_all[i] == vocab.ids(sents[i]) (a pure
            # function) and tr_idx/ev_idx reproduce the EXACT same permutation-slice-truncate arithmetic used to
            # build tr/ev above; dev_ids is the same suffix-of-tr_ids slice that dev was a suffix-of-tr slice.
            tr_idx = idx[:cut][:args.max_train_sents]
            ev_idx = idx[cut:][:args.max_eval_sents]
            tr_ids = [_sents_ids_all[i] for i in tr_idx]
            ev_ids = [_sents_ids_all[i] for i in ev_idx]
            dev_ids = tr_ids[-min(2000, len(tr_ids)//5):]
        else:
            tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]; dev_ids = [vocab.ids(s) for s in dev]

        P_bi = fit_bigram(tr_ids, V)
        tri, lambdas = fit_interp_trigram(tr_ids, V, dev_ids)

        init_emb = build_ppmi_codes(tr_ids, V, args.d_model, args.ppmi_window) if args.input == "ppmi" else None
        net, WKV_cls = build_and_train_wkv(tr_ids, V, seed, args, device, init_emb=init_emb)
        if getattr(args, "generate", 0):                             # AUTOREGRESSIVE generation (does the WKV produce prose?)
            import torch
            prompts = [["once", "upon", "a", "time"], ["the", "dog", "and", "the", "cat"],
                       ["one", "day", "a", "boy", "named", "tom"], ["she", "was", "very", "happy", "because"]]
            for prompt in prompts[:max(1, args.gen_prompts)]:
                ids_g = [vocab.w2i.get(w, vocab.unk) for w in prompt]
                with torch.no_grad():
                    for _ in range(args.generate):
                        logits = net(torch.tensor([ids_g], device=device))[0, -1]
                        p = torch.softmax(logits / 0.8, -1).cpu().numpy()
                        ids_g.append(int(np.random.default_rng(seed * 91 + len(ids_g)).choice(V, p=p / p.sum())))
                print(f"    [seed {seed}] GEN[{' '.join(prompt)}]: {' '.join(vocab.i2w[i] for i in ids_g)}", flush=True)
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
            if args.recurrence in ("learnkey", "deltanet"):
                # CANDIDATE-REPLACEMENT GO GATE (2026-09-04 learnkey / 2026-09-05 deltanet): the universal
                # per-arm bar above (>0.02 margin + anti-cheat collapse) is necessary but not sufficient for a
                # mechanism whose PURPOSE is to REPLACE linattn -- it must also at least match linattn's own
                # measured deep-context margin (--linattn-baseline-margin), not merely re-clear the trigram bar
                # assoc/assoc_t already failed. PASS THE RIGHT CONSTANT PER DOMAIN: for a broad wikitext-103 run
                # this is linattn's OWN wt103 floor (--linattn-baseline-margin -0.286, the d10-99 value in
                # research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json) -- the direction-test asks
                # whether deltanet LIFTS off that floor; for a simplewiki run it is linattn's +0.0505 6-seed mean
                # (the default). REPORT-TIME comparison against a command-line constant (not a fresh linattn run
                # in this invocation); the 6-seed verdict is the mean of this per-seed comparison across --seeds.
                go_vs_linattn = deep["margin_vs_trigram"] >= args.linattn_baseline_margin
                print(f"    [seed {seed}] vs-linattn-baseline: {args.recurrence} {deep['margin_vs_trigram']:+.3f} vs "
                      f"linattn {args.linattn_baseline_margin:+.3f} (Δ{deep['margin_vs_trigram']-args.linattn_baseline_margin:+.3f}) "
                      f"-> {'GO (matches/beats linattn)' if go_vs_linattn else 'short of linattn'}", flush=True)

    out = {"runner": "_emerge_wkv_lm_derisk", "corpus": args.corpus, "seeds": args.seeds, "d_model": args.d_model,
           "pred_aux_weight": args.pred_aux_weight, "pred_aux_offsets": args.pred_aux_offsets,
           "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
