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

        def forward(self, x):                          # x: [B,T] token ids
            B, T = x.shape
            h = self.ln(self.emb(x))                    # [B,T,D]
            k = self.Wk(h); v = self.Wv(h); r = torch.sigmoid(self.Wr(h))
            wdec = torch.exp(-torch.nn.functional.softplus(self.w))       # per-channel decay in (0,1)
            if self.memoryless:
                wdec = torch.zeros_like(wdec)           # ANTI-CHEAT: no carry -> only the current token
            if RECUR == "ssm":
                assert self.n_layers == 1, "--n-layers>1 is only implemented for --recurrence wkv (the baseline branch)"
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
                        return self.head(torch.stack(outs2, 1))
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
                    return self.head(torch.stack(outs2, 1))
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
            h = torch.stack(outs, 1)                     # [B,T,D]  = layer-0 output (the original single block)
            for blk in self.extra:                       # DEPTH: pre-norm residual WKV layers on top (empty at n_layers=1)
                h = h + blk(h)
            return self.head(h)                          # [B,T,V]

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
            logits = fwd(X)[:, :-1]                           # predict token t+1 from context 0..t
            tgt = X[:, 1:]; m = msk[:, 1:]
            L = lossf(logits.reshape(-1, V), tgt.reshape(-1)).reshape(tgt.shape)
            loss = (L * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            _ep_loss += loss.detach(); _ep_n += 1
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
                    help="DEPTH LEVER (gap#1): stack N pre-norm residual WKV layers. 1 = the original single block "
                         "(byte-identical). >1 only implemented for --recurrence wkv (the baseline branch).")
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

    out = {"runner": "_emerge_wkv_lm_derisk", "corpus": args.corpus, "seeds": args.seeds, "d_model": args.d_model,
           "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
