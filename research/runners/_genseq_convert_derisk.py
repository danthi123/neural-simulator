"""Generative-sequence frontier (Spine A) -- DECISIVE CONVERT de-risk.

Question (the ONE open question): does the working NON-spiking generator
Gen-F (sim.tiny_transformer.TinyGPT, the shipped 3.45M GPT), CONVERTED to
spikes TRAINING-FREE (2025 ANN->SNN: MBE/LAS/ECMT class -- keep weights
VERBATIM + a no-gradient calibration minibatch + spiking-rate
softmax/LayerNorm/GELU at T timesteps), STILL generate coherent NOVEL
text? Designed in
research/findings/2026-06-22-genseq-convert-scoping.md  3.

WHAT THIS IS (faithful + simple, the Rank-1 path of the scoping):
  A *spiking-rate* forward for TinyGPT at a timestep budget T. The
  rate-SNN / QCFS equivalence: a rate-coded IF population with threshold
  theta over T steps computes clip(round(x*T/theta),0,T)*theta/T -- a
  clip-floor QUANTIZATION of the activation to T levels within a
  CALIBRATED dynamic range. Under rate coding the LINEAR ops
  (embeddings, in_proj/out_proj, mlp, head) are EXACT (linearity commutes
  with rate averaging -- the standard ANN->SNN result), so only the three
  hard NONLINEAR ops cost anything, and each is realized as a
  T-level rate-quantization over a one-pass-calibrated range:
    * Softmax (attention): exp over T-quantized logits + T-quantized
      reciprocal normalizer (the most T-hungry op -- the named risk).
    * LayerNorm: mean/var exact (rate-linear/quadratic); the 1/sqrt(var)
      inv-sqrt over a T-quantized variance.
    * GELU (MLP): exact GELU over a T-quantized input range.
  T is THE knob (raise T = the standard latency/fidelity lever). The
  calibration is a single forward pass over a TinyStories minibatch
  recording each op's input range -- NO gradient on the model, weights
  copied VERBATIM. NO sim/ edit, NO bridge, NO training. PyTorch only.

WHAT THIS IS NOT: an on-bridge realization (that is the SEPARATE later
consolidation step, step-0-de-risked at 0.92). SIM_BACKEND is irrelevant
here (torch forward).

METRICS: the BYTE-UNMODIFIED Gen-F gate is reused VERBATIM
(generator_f_gate._heldout_nll / _generate + subword_lm_gate_core
gs_verdict bars 0.20/1.5/0.5/0.20 + abs-competence floor uniform_ppl=V).
The ANN model and the converted SPIKING model are scored by the SAME
gate functions (they call model(x)); the spiking model is a drop-in
callable. 3 seeds (42/43/44). Word-shuffle .ctl control reused on disk.

VERDICT (per scoping  3): GO = 3/3 seeds -- spiking ho-ppl < V (513) AND
spiking/ANN ppl-ratio <= ~1.2 AND clears the gate AND novel
(distinct>=0.5, copy<=0.20, beats the word-shuffle control) AND reads
coherent. If T=16 fails: SWEEP T (16->32->64), cheap-first.

ASCII only. Honest: ppl can be preserved while GENERATION degrades -- we
ALWAYS decode + record a sample; the controller reads it.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# The training-free spiking-rate (rate-SNN / QCFS clip-floor) quantizer.
# --------------------------------------------------------------------------
def _rate_quantize(x, lo, hi, T):
    """Rate-SNN realization of a real value over T timesteps within a
    CALIBRATED range [lo,hi]: the clip-floor activation a rate-coded IF
    population computes. Maps x -> one of T+1 evenly spaced levels in
    [lo,hi] (clip outside). This is EXACTLY the QCFS/rate-SNN expected
    activation at timestep budget T (Bu 2023; the rate-code wall). T->inf
    recovers the ANN value. Weights/biases are untouched and exact.
    """
    if lo is None or hi is None:
        # uncalibrated stat -> behave as the exact ANN op (no-op). The
        # runner always calibrates first; this only guards stray spike-
        # mode forwards before calibration.
        return x
    if hi <= lo:
        return x.clamp(min=lo, max=hi) if hi > lo else x.new_full(x.shape, lo)
    step = (hi - lo) / float(T)
    q = torch.round((x - lo) / step)
    q = q.clamp(min=0.0, max=float(T))
    return lo + q * step


class _CalibStat:
    """Per-op input-range record over the calibration minibatch. The SOTA
    calibrates activation RANGES (not exact min/max -- a single outlier
    token must not blow up the quantizer step). We use a SORT-FREE robust
    range = mean +/- K*std (clamped to the observed min/max), which is the
    standard fast activation-range calibration (QCFS/MBE use running
    max / mean+lambda*std). CRITICAL: this is O(n) GPU REDUCTIONS
    (mean/std/amin/amax) -- NOT torch.quantile, whose O(n log n) sort,
    called ~24x per window, was the ~120s calibration wall. NO gradient;
    GPU-resident (only 4 scalars sync per op per window)."""

    _K = 4.0  # mean +/- 4 std ~ covers ~99.99% of ~normal activations

    def __init__(self):
        self._lo_acc = 0.0
        self._hi_acc = 0.0
        self._k = 0
        self.lo = None
        self.hi = None

    def observe(self, x):
        v = x.detach().reshape(-1).float()
        if v.numel() == 0:
            return
        mu = v.mean()
        sd = v.std() if v.numel() > 1 else v.new_zeros(())
        amin = v.amin()
        amax = v.amax()
        # robust band, clamped to the actual observed extremes
        lo = torch.maximum(mu - self._K * sd, amin)
        hi = torch.minimum(mu + self._K * sd, amax)
        self._lo_acc += float(lo)  # 4 tiny scalar syncs per op per window
        self._hi_acc += float(hi)
        self._k += 1

    def finalize(self):
        if self._k == 0:
            self.lo, self.hi = 0.0, 1.0
            return
        self.lo = self._lo_acc / self._k
        self.hi = self._hi_acc / self._k
        if self.hi <= self.lo:
            self.hi = self.lo + 1e-6


class SpikingTinyGPT:
    """A spiking-rate (training-free ANN->SNN) forward of a *trained*
    TinyGPT. Reuses the ANN's weights VERBATIM. The three nonlinear ops
    are rate-quantized to T levels over CALIBRATED ranges; everything
    linear is exact. Quacks like the ANN model: callable model(idx) ->
    logits [B,T_seq,V], plus .eval()/.cfg, so the byte-unmodified gate
    (_heldout_nll / _generate) drives it unchanged.

    calibrate(...) runs ONE no-gradient pass to set the per-op ranges
    (mode='calib'); after that, forward runs in mode='spike'.
    """

    def __init__(self, ann, T, device):
        self.ann = ann
        self.ann.eval()
        self.T = int(T)
        self.device = device
        self.cfg = ann.cfg
        self.n_layer = ann.cfg["n_layer"]
        self.n_head = ann.cfg["n_head"]
        self.d = ann.cfg["d_model"]
        self.dh = self.d // self.n_head
        self.block_size = ann.cfg["block_size"]
        self._mode = "spike"
        # one calibration record per (layer, op-instance)
        self.stats = {}
        for li in range(self.n_layer):
            self.stats[("attn_logits", li)] = _CalibStat()
            self.stats[("attn_norm", li)] = _CalibStat()
            self.stats[("ln1_var", li)] = _CalibStat()
            self.stats[("ln2_var", li)] = _CalibStat()
            self.stats[("gelu", li)] = _CalibStat()
        self.stats[("lnf_var", 0)] = _CalibStat()

    def eval(self):
        self.ann.eval()
        return self

    # ----- the three nonlinear ops, rate-quantized in 'spike' mode -----
    def _layernorm(self, x, ln, key):
        # mean/var are rate-linear/quadratic -> EXACT. The 1/sqrt(var+eps)
        # is the nonlinear part: quantize the variance to T levels over
        # its calibrated range, then exact inv-sqrt (MBE/LAS inv-sqrt).
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        eps = ln.eps
        if self._mode == "calib":
            self.stats[key].observe(var)
        elif self._mode == "spike":
            st = self.stats[key]
            var = _rate_quantize(var, st.lo, st.hi, self.T)
        xn = (x - mu) / torch.sqrt(var + eps)
        return xn * ln.weight + ln.bias

    def _gelu(self, x, key):
        # exact GELU over a T-quantized input range (MBE: GELU by basis
        # over partitioned sub-intervals; the rate-quantized exact-GELU is
        # the faithful simplest form).
        if self._mode == "calib":
            self.stats[key].observe(x)
        elif self._mode == "spike":
            st = self.stats[key]
            x = _rate_quantize(x, st.lo, st.hi, self.T)
        return F.gelu(x)

    def _softmax_attn(self, q, k, v, li, n):
        # manual scaled-dot-product attention so the INTERNAL softmax is
        # replaced by a spiking-rate softmax. q,k,v: [B,H,n,dh].
        scale = 1.0 / math.sqrt(self.dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,n,n]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool,
                                     device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask[None, None], float("-inf"))
        # softmax = exp(s - rowmax) / sum. rowmax/subtraction is the
        # LAS incremental-max step (linear-ish); the exp + reciprocal are
        # the rate-hungry nonlinear parts -> rate-quantize BOTH.
        rowmax = scores.amax(dim=-1, keepdim=True)
        shifted = scores - rowmax  # in (-inf, 0]; finite entries only
        if self._mode == "calib":
            finite = shifted[torch.isfinite(shifted)]
            if finite.numel():
                self.stats[("attn_logits", li)].observe(finite)
            ex = torch.exp(shifted)
            ex = ex.masked_fill(mask[None, None], 0.0)
            denom = ex.sum(dim=-1, keepdim=True)
            self.stats[("attn_norm", li)].observe(denom)
            attn = ex / denom.clamp_min(1e-20)
        else:  # spike
            sl = self.stats[("attn_logits", li)]
            shifted_q = _rate_quantize(shifted, sl.lo, min(sl.hi, 0.0),
                                       self.T)
            shifted_q = shifted_q.masked_fill(mask[None, None],
                                              float("-inf"))
            ex = torch.exp(shifted_q)
            ex = ex.masked_fill(mask[None, None], 0.0)
            denom = ex.sum(dim=-1, keepdim=True)
            sn = self.stats[("attn_norm", li)]
            denom = _rate_quantize(denom, sn.lo, sn.hi, self.T)
            attn = ex / denom.clamp_min(1e-20)
        return torch.matmul(attn, v)  # [B,H,n,dh]

    def _block(self, x, li):
        blk = self.ann.blocks[li]
        n = x.size(1)
        B = x.size(0)
        # --- attention sub-layer (pre-norm residual) ---
        h = self._layernorm(x, blk.ln1, ("ln1_var", li))
        # nn.MultiheadAttention packs Q,K,V into in_proj_weight[3d,d].
        W = blk.attn.in_proj_weight        # [3d, d]
        b = blk.attn.in_proj_bias          # [3d]
        qkv = F.linear(h, W, b)            # [B,n,3d] -- linear, exact
        q, k, vv = qkv.split(self.d, dim=-1)
        # reshape to heads
        q = q.view(B, n, self.n_head, self.dh).transpose(1, 2)
        k = k.view(B, n, self.n_head, self.dh).transpose(1, 2)
        vv = vv.view(B, n, self.n_head, self.dh).transpose(1, 2)
        ao = self._softmax_attn(q, k, vv, li, n)        # [B,H,n,dh]
        ao = ao.transpose(1, 2).contiguous().view(B, n, self.d)
        ao = F.linear(ao, blk.attn.out_proj.weight,
                      blk.attn.out_proj.bias)            # linear, exact
        x = x + ao
        # --- MLP sub-layer (pre-norm residual) ---
        h2 = self._layernorm(x, blk.ln2, ("ln2_var", li))
        m = F.linear(h2, blk.mlp[0].weight, blk.mlp[0].bias)  # exact
        m = self._gelu(m, ("gelu", li))
        m = F.linear(m, blk.mlp[2].weight, blk.mlp[2].bias)   # exact
        return x + m

    def _forward(self, idx):
        n = idx.size(1)
        if n > self.block_size:
            raise ValueError("seq len %d > block_size %d"
                             % (n, self.block_size))
        pos = torch.arange(n, device=idx.device)
        x = self.ann.tok(idx) + self.ann.pos(pos)[None, :, :]  # exact
        for li in range(self.n_layer):
            x = self._block(x, li)
        x = self._layernorm(x, self.ann.lnf, ("lnf_var", 0))
        return self.ann.head(x)  # head is linear -> exact

    def __call__(self, idx):
        with torch.no_grad():
            return self._forward(idx)

    def calibrate(self, tok, text, n_windows=32, ids=None):
        """One no-gradient pass: record each nonlinear op's input range
        over up to n_windows block-size windows of TinyStories text.
        Pass pre-encoded `ids` to skip the (pure-Python, ~120s on the
        full corpus) BPE encode -- calibration needs only
        n_windows*block_size tokens, so encoding a small SLICE (or
        reusing cached ids) is the whole speed fix; the forwards
        themselves are ~10ms each."""
        if ids is None:
            # encode only enough text for the windows we need (a slice,
            # not the 7.2M-char corpus): ~ n_windows*block_size tokens.
            need_chars = max(20000, n_windows * self.block_size * 8)
            ids = tok.encode(text[:need_chars])
        bs = self.block_size
        if len(ids) < bs + 2:
            return
        self._mode = "calib"
        n_avail = (len(ids) - 1) // bs
        step = max(1, n_avail // max(1, n_windows))
        with torch.no_grad():
            cnt = 0
            for w in range(0, n_avail, step):
                s = w * bs
                x = torch.tensor(ids[s:s + bs], dtype=torch.long,
                                 device=self.device)[None]
                _ = self._forward(x)
                cnt += 1
                if cnt >= n_windows:
                    break
        for st in self.stats.values():
            st.finalize()
        self._mode = "spike"
        return cnt


def _ann_from_ckpt(ckpt_path, tok, device):
    """Load a shipped Gen-F checkpoint {model,optim,...} into a TinyGPT
    with weights VERBATIM."""
    from sim.tiny_transformer import TinyGPT
    st = torch.load(ckpt_path, map_location=device)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    V = tok.vocab_size
    m = TinyGPT(vocab_size=V, d_model=256, n_layer=4, n_head=4,
                block_size=128, dropout=0.0).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m, int(st.get("step", -1)) if isinstance(st, dict) else -1


def _sample_text(model, tok, prompt_ids, n_tokens, block_size, device,
                 seed):
    from research.runners.generator_f_gate import _generate
    ids = _generate(model, tok, prompt_ids, n_tokens, block_size,
                    device, seed)
    return ids, tok.decode(ids)


def run_seed(seed, T, ckpt_base, heldout_text, train_text, device,
             eval_positions, gen_tokens, calib_windows, n_layer=4,
             ann_gen=False):
    from sim.bpe_tokenizer import BPETokenizer
    from research.runners.generator_f_gate import _heldout_nll, _generate
    from research.runners.subword_lm_gate_core import (
        perplexity, distinct_ngram_ratio, verbatim_copy_fraction,
        gs_verdict)

    real_ckpt = "%s.s%d.real.pt" % (ckpt_base, seed)
    real_bpe = "%s.s%d.real.bpe.json" % (ckpt_base, seed)
    ctl_ckpt = "%s.s%d.ctl.pt" % (ckpt_base, seed)
    ctl_bpe = "%s.s%d.ctl.bpe.json" % (ckpt_base, seed)
    for p in (real_ckpt, real_bpe, ctl_ckpt, ctl_bpe):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    rtok = BPETokenizer.load(real_bpe)
    ctok = BPETokenizer.load(ctl_bpe)
    rmodel, rstep = _ann_from_ckpt(real_ckpt, rtok, device)
    cmodel, _ = _ann_from_ckpt(ctl_ckpt, ctok, device)
    V = rtok.vocab_size
    bs = rmodel.cfg["block_size"]

    # PERF: the pure-Python BPE encode is ~120s on the 7.2M-char corpus
    # and ~13s on the 800K heldout. The gate's _heldout_nll steps through
    # at most `eval_positions` windows and STOPS -- so encoding text
    # beyond (eval_positions+2)*block_size tokens is never scored.
    # Truncate the text we hand it to exactly that many CHARS-worth
    # (~8 chars/token slack) => identical windows, far cheaper encode.
    # (Math-identical: same first-N windows scored; nothing dropped that
    # the gate would have read.)
    cap_tok = (int(eval_positions) + 2) * bs
    cap_chars = cap_tok * 8
    ho_eval = heldout_text[:cap_chars]
    tr_eval = train_text[:cap_chars]

    # ---- ANN baselines (sanity: must reproduce ho-ppl ~6.1) ----
    ann_ho_ppl = perplexity(_heldout_nll(rmodel, rtok, ho_eval, bs,
                                         device, eval_positions))

    # ---- build + CALIBRATE the spiking model (no gradient) ----
    # calibrate on a slice (encodes only n_windows*block_size tokens).
    smodel = SpikingTinyGPT(rmodel, T=T, device=device)
    n_cal = smodel.calibrate(rtok, train_text, n_windows=calib_windows)

    # ---- SPIKING metrics (the gate, byte-unmodified, drives smodel) ----
    sp_ho_ppl = perplexity(_heldout_nll(smodel, rtok, ho_eval, bs,
                                        device, eval_positions))
    sp_tr_ppl = perplexity(_heldout_nll(
        smodel, rtok, tr_eval, bs, device, eval_positions))
    # word-shuffle control: convert the CTL model the SAME way (faithful:
    # the control must be the spiking control, not the ANN one).
    sctl = SpikingTinyGPT(cmodel, T=T, device=device)
    sctl.calibrate(ctok, train_text, n_windows=calib_windows)
    sp_ctl_ppl = perplexity(_heldout_nll(sctl, ctok, ho_eval, bs,
                                         device, eval_positions))

    # ---- generation + novelty (decode + record the actual text) ----
    prompt_ids = rtok.encode(" ".join(heldout_text.split()[:8]))
    gen_ids, gen_text = _sample_text(smodel, rtok, prompt_ids,
                                     gen_tokens, bs, device,
                                     seed * 13 + 5)
    # copy-reference: the gate builds the 8-gram set from the train ids.
    # The full 7.2M-char encode is ~120s; a large bounded slice
    # (~2M chars => ~700K tokens => ~700K 8-grams) is an ample verbatim-
    # copy reference and keeps the anti-copy bar load-bearing. Documented
    # bound; only STRENGTHENS detection if widened.
    tr_ids = rtok.encode(train_text[:2_000_000])
    distinct = distinct_ngram_ratio(gen_ids, n=3)
    copy_frac = verbatim_copy_fraction(gen_ids, tr_ids, n=8)
    # ANN's own generated sample (for side-by-side coherence read). OFF
    # by default -- it is NOT a gate input (only the SPIKING generation
    # is scored); it just doubles the slow autoregressive sampling. The
    # ANN ho-ppl already pins the ANN baseline.
    if ann_gen:
        _, ann_gen_text = _sample_text(rmodel, rtok, prompt_ids,
                                       gen_tokens, bs, device,
                                       seed * 13 + 5)
    else:
        ann_gen_text = "(ANN generation skipped; not a gate input)"

    v = gs_verdict(heldout_ppl=sp_ho_ppl, shuffled_ppl=sp_ctl_ppl,
                   train_ppl=sp_tr_ppl, distinct=distinct,
                   copy_frac=copy_frac, has_shuffled_control=True,
                   uniform_ppl=V)
    v["seed"] = seed
    # free heavy state before the next seed/T (the prior run died on a
    # heavy seed; keep each seed's footprint bounded).
    del tr_ids, smodel, sctl, rmodel, cmodel
    if device == "cuda":
        torch.cuda.empty_cache()
    ppl_ratio = (sp_ho_ppl / ann_ho_ppl
                 if (math.isfinite(ann_ho_ppl) and ann_ho_ppl > 0)
                 else float("inf"))
    rec = {
        "seed": seed, "T": T, "train_step": rstep,
        "ann_heldout_ppl": ann_ho_ppl,
        "spiking_heldout_ppl": sp_ho_ppl,
        "spiking_train_ppl": sp_tr_ppl,
        "spiking_ctl_ppl": sp_ctl_ppl,
        "uniform_ppl": V,
        "ppl_ratio_spiking_over_ann": ppl_ratio,
        "distinct_trigram": distinct,
        "verbatim_copy_frac": copy_frac,
        "n_calib_windows": n_cal,
        "spiking_gen_sample": gen_text[:400],
        "ann_gen_sample": ann_gen_text[:400],
        "verdict": v,
    }
    return v, rec


def main():
    from research.runners.corpus_fetch import fetch_corpus, split_corpus
    from research.runners.subword_lm_gate_core import (
        gs_aggregate_multiseed)

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--T", type=int, default=16,
                    help="spiking timestep budget (the fidelity knob)")
    ap.add_argument("--sweep-T", type=str, default="",
                    help="comma list to auto-escalate, e.g. 16,32,64")
    ap.add_argument("--corpus", type=str, default="tinystories")
    ap.add_argument("--max-corpus-mb", type=int, default=8)
    ap.add_argument("--ckpt-base", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_f_gate.ckpt")
    ap.add_argument("--eval-positions", type=int, default=2000)
    ap.add_argument("--gen-tokens", type=int, default=200)
    ap.add_argument("--calib-windows", type=int, default=32)
    ap.add_argument("--ann-gen", action="store_true",
                    help="also autoregressively sample the ANN (cosmetic "
                         "side-by-side; NOT a gate input; doubles slow "
                         "generation -- off by default)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/"
                            "_genseq_convert_derisk.json")
    a = ap.parse_args()

    device = ("cuda" if (a.device == "auto" and torch.cuda.is_available())
              else ("cpu" if a.device == "auto" else a.device))
    seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    t_list = ([int(x) for x in a.sweep_T.split(",") if x.strip()]
              if a.sweep_T.strip() else [int(a.T)])

    print("=" * 70, flush=True)
    print("GEN-SEQ CONVERT DE-RISK -- training-free ANN->SNN of Gen-F",
          flush=True)
    print(" (spiking-rate TinyGPT forward; weights VERBATIM; "
          "no gradient; no sim/ edit;", flush=True)
    print("  byte-unmodified Gen-F gate; T-sweep cheap-first; "
          "3 seeds; device=%s)" % device, flush=True)
    print("=" * 70, flush=True)

    if len(seeds) < 3:
        print("[NOT RUNNABLE] %d seed(s); >=3 MANDATORY." % len(seeds),
              flush=True)
        return 2

    cinfo = fetch_corpus(name=a.corpus,
                         max_bytes=int(a.max_corpus_mb) * 1_000_000)
    train_text, heldout_text = split_corpus(cinfo["text"],
                                            heldout_frac=0.1)
    print("[corpus] used=%s degraded=%s train=%d heldout=%d"
          % (cinfo["corpus_used"], cinfo["degraded"],
             len(train_text), len(heldout_text)), flush=True)

    t0 = time.time()
    sweep_results = []
    final_T = None
    final_agg = None
    final_records = None
    for T in t_list:
        print("\n" + "#" * 70 + "\n### T = %d\n" % T + "#" * 70,
              flush=True)
        per_seed_verdicts = []
        per_seed_records = []
        for seed in seeds:
            print("\n" + "-" * 60 + "\n[T=%d SEED %d]" % (T, seed)
                  + "\n" + "-" * 60, flush=True)
            v, rec = run_seed(
                seed=seed, T=T, ckpt_base=a.ckpt_base,
                heldout_text=heldout_text, train_text=train_text,
                device=device, eval_positions=a.eval_positions,
                gen_tokens=a.gen_tokens, calib_windows=a.calib_windows,
                ann_gen=a.ann_gen)
            per_seed_verdicts.append(v)
            per_seed_records.append(rec)
            print("[T=%d SEED %d] ann_ppl=%.3f spk_ppl=%.3f ratio=%.3f "
                  "ctl=%.3f tr=%.3f uni=%d distinct=%.3f copy=%.3f -> %s"
                  % (T, seed, rec["ann_heldout_ppl"],
                     rec["spiking_heldout_ppl"],
                     rec["ppl_ratio_spiking_over_ann"],
                     rec["spiking_ctl_ppl"], rec["spiking_train_ppl"],
                     rec["uniform_ppl"], rec["distinct_trigram"],
                     rec["verbatim_copy_frac"], v["GATE"]), flush=True)
            print("   [spiking sample] %s"
                  % rec["spiking_gen_sample"][:200], flush=True)

        agg = gs_aggregate_multiseed(per_seed_verdicts)
        # convert-specific extra bar: ppl-ratio <= 1.2, all seeds
        ratios = [r["ppl_ratio_spiking_over_ann"]
                  for r in per_seed_records]
        ratio_ok = all(math.isfinite(x) and x <= 1.2 for x in ratios)
        convert_go = bool(agg["GATE"] == "PASS" and ratio_ok)
        sweep_results.append({
            "T": T, "gate_aggregate": agg,
            "ppl_ratios": ratios, "ratio_bar_1.2_all_pass": ratio_ok,
            "CONVERT_GO": convert_go,
            "per_seed": per_seed_records})
        print("\n[T=%d AGGREGATE] gate=%s (n_pass=%d/%d) "
              "ratio<=1.2 all=%s -> CONVERT_%s"
              % (T, agg["GATE"], agg["n_pass"], agg["n_seeds"],
                 ratio_ok, "GO" if convert_go else "NO"), flush=True)
        if convert_go:
            final_T, final_agg, final_records = T, agg, per_seed_records
            break  # cheap-first: first T that GOes wins
        final_T, final_agg, final_records = T, agg, per_seed_records

    overall_go = any(s["CONVERT_GO"] for s in sweep_results)
    result = {
        "task": "Gen-seq CONVERT de-risk: training-free ANN->SNN of "
                "Gen-F (spiking-rate TinyGPT), does it still generate "
                "coherent novel text?",
        "method": ("rate-SNN/QCFS clip-floor quantization of softmax/"
                   "LayerNorm/GELU at T timesteps over calibrated "
                   "ranges; weights VERBATIM; no gradient; linear ops "
                   "exact; reference MBE/LAS/ECMT"),
        "corpus_used": cinfo["corpus_used"],
        "corpus_degraded": cinfo["degraded"],
        "seeds": seeds, "T_swept": t_list,
        "device": device,
        "config": {"eval_positions": a.eval_positions,
                   "gen_tokens": a.gen_tokens,
                   "calib_windows": a.calib_windows,
                   "ckpt_base": a.ckpt_base},
        "anti_cheat": {
            "gate": "BYTE-UNMODIFIED Gen-F gate (gs_verdict bars "
                    "0.20/1.5/0.5/0.20 + abs-competence floor "
                    "uniform_ppl=V); >=3 seeds unbypassable; bars "
                    "NEVER tuned",
            "control": "word-shuffle .ctl model CONVERTED the same way "
                       "(spiking control)",
            "convert_bar": "spiking/ANN ppl-ratio <= 1.2 ALL seeds "
                           "(the SOTA gets ~1.03; 1.2 = generous "
                           "first-port margin)",
            "honest_risk": "ppl can be preserved while GENERATION "
                           "degrades -> spiking_gen_sample recorded "
                           "every seed for the controller to read; no "
                           "sim/ edit; no training",
            "no_confab": "n/a (free LM, not the retrieval composer); "
                         "novelty = non-degenerate + not-copy + "
                         "beats-shuffle"},
        "sweep_results": sweep_results,
        "final_T": final_T,
        "OVERALL": "GO" if overall_go else "NO-GO",
        "CONVERT_GO": overall_go,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=str),
                           encoding="utf-8")

    print("\n" + "=" * 70, flush=True)
    print("CONVERT DE-RISK VERDICT", flush=True)
    print("=" * 70, flush=True)
    for s in sweep_results:
        print("  T=%d: gate=%s (%d/%d) ratio<=1.2_all=%s -> CONVERT_%s "
              "| ratios=%s"
              % (s["T"], s["gate_aggregate"]["GATE"],
                 s["gate_aggregate"]["n_pass"],
                 s["gate_aggregate"]["n_seeds"],
                 s["ratio_bar_1.2_all_pass"],
                 "GO" if s["CONVERT_GO"] else "NO",
                 ["%.3f" % x for x in s["ppl_ratios"]]), flush=True)
    print("  -> OVERALL: %s (final_T=%s)"
          % (result["OVERALL"], final_T), flush=True)
    print("  -> %s" % a.out, flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
