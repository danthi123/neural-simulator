"""
gap#1 / A1 — biologize the mouth's INPUT PROJECTION `v = Wv @ LN(emb[tid])` as a SIGNED GRADED-CONDUCTANCE synaptic
read on the spiking substrate, the LAST host matmul in the mouth forward path. Uses the SAME validated template that
biologized the OUTPUT projection `Wo_sp @ state` (`_wkv_graded_output_projection_derisk`, corr 0.984) and the read-out
`head_w @ h` — Dale-split signed weights -> net graded synaptic-current margin at rest — reused VERBATIM by import.

WHERE THIS SITS IN THE MOUTH PIPELINE (per-token `tid`, WKV leaky state `ap`/`an`):
    (1) v      = Wv @ LN(emb[tid])                       # INPUT PROJECTION  <<< THIS RUNNER (the last host matmul)
    (2) ap,an  = decay*ap+relu(v), decay*an+relu(-v)     # WKV leaky STATE   (SUBSTRATE slow-NMDA — state-GO)
    (3) r_h    = sigmoid(Wr @ LN(emb[tid]))              # receptance gate   (host)
    (4) h      = r_h * (Wo_sp @ [ap,an])                 # OUTPUT PROJECTION (SUBSTRATE graded read — projection-GO)
    (5) logits = head_w @ h + head_b                     # read-out          (SUBSTRATE graded read + bias pop — read-GO)
Steps (2),(4),(5) are already substrate graded reads (2026-08-13 read/projection/state GOs). Step (1), the LAST host
matmul on the forward path, is this runner's target. Closing it leaves only LN + the embedding lookup + the trained
WEIGHT VALUES (weights learnable via the resolved 2026-08-12 e-prop rule) between sensation and the winner.

THE MECHANISM — the SIGNED input case (the new subtlety vs the output projection):
  The output projection's input (the WKV state `[ap,an]`) was already NONNEG. The input projection's input
  `x = LN(emb[tid])` is SIGNED (~51% negative) AND `Wv` is ~50% negative. A carrier population's rate codes a NONNEG
  magnitude, so a signed-input signed-weight matmul needs the four-quadrant (Dale) decomposition. It reduces EXACTLY to
  the validated output-projection template by extending both:
      x = x_pos - x_neg        (x_pos = relu(x) >= 0, x_neg = relu(-x) >= 0)          [nonneg dual code, length 2D]
      Wv @ x = [Wv, -Wv] @ [x_pos, x_neg]   ==   Wv_ext @ xstate                       (Wv_ext = [Wv,-Wv], [D, 2D])
  So with `Wv_ext` [D,2D] as the projection weight and `xstate = [x_pos, x_neg]` [2D] as the nonneg input, the problem
  IS an output-projection read: `GradedOutputProjection` splits `Wv_ext = Wv_ext_pos - Wv_ext_neg` into an EXCITATORY
  (stc_e -> hpool, g_e) and INHIBITORY-SHADOW (stc_i -> hpool, g_i) half, drives two matched carrier populations by the
  SAME nonneg `xstate` drive, and reads each channel's `v` from the substrate's OWN net signed synaptic-current margin
  at rest `v_k = df_e*g_e[k] + df_i*g_i[k]  ~  Wv_ext_k @ xstate = Wv_k @ x`. Expanded, the four quadrants are:
      g_e ~ Wp@x_pos + Wn@x_neg   (excitatory);   g_i ~ Wn@x_pos + Wp@x_neg   (inhibitory)   (Wp=relu(Wv), Wn=relu(-Wv))
      v   ~ (Wp@x_pos + Wn@x_neg) - (Wn@x_pos + Wp@x_neg) = (Wp-Wn)@(x_pos-x_neg) = Wv @ x.
  This is the SAME class `GradedOutputProjection`, instantiated on a tiny shim whose `.Wo_sp = [Wv,-Wv]`, `.D = D`.
  A single drive_gain + inh:exc `ratio` (calibrated ONCE on seed 42 to a WIDE plateau) and a single output scalar
  `v_out_scale` (least-squares match of the graded margin to the host `v` magnitude, seed 42) are FIXED for 5 unseen
  seeds. Because `v = Wv@LN(emb[tid])` depends ONLY on `tid`, the substrate `v` is CACHED per token id per seed.

METRICS:
  (a) RECONSTRUCTION FIDELITY (the headline, direct analog of the 0.984 output projection):
      v_corr_signed        = per-position Pearson corr(v_substrate, Wv@LN(emb))  averaged   [HEADLINE]
      v_corr_positive_only = corr with the INHIBITORY SHADOW lesioned (df_e*g_e alone) -> MUST be far worse (both the
                             weight AND the input are ~50% signed, so all four quadrants are populated; sign load-bearing)
      v_cosine, v_corr_scramble (channel-decode permute -> ~0)
  (b) FULL-PIPELINE next-word fidelity vs the full HOST mouth (does adding the input-projection stage degrade the mouth):
      B1 subV_hostpipe : SUBSTRATE v -> HOST WKV state -> HOST proj+read -> next word   (isolates the input stage,
                         autoregressive; cheap via the v-cache). recov_argmax + argmax_agree vs the full host mouth.
      B2 fullsub arms  : feed SUBSTRATE v into the ALREADY-substrate pipeline (WKV state substrate + composed substrate
                         read), vs the SAME pipeline on HOST v (== the existing full-substrate pipeline, recov ~0.86).
                         recov(fullsub_subV) vs recov(fullsub_hostV) -> the delta IS the input-projection stage's cost.

ANTI-CHEATS (brain-based; each MUST collapse; the inhibitory shadow load-bearing 6/6):
  * signed vs positive-only : v_corr_signed >> v_corr_positive_only  (negative weights / inhibitory shadow load-bearing)
  * scramble (permute the hpool->channel decode)  -> v_corr ~ 0 AND downstream -> chance
  * zero-input (zero LN(emb))                      -> substrate v ~ 0 -> downstream -> chance (cache-immune)
  * provenance: v read off cp_conductance_g_e/g_i; host_rng_draws_on_read_path == 0; 0 host matmul on `v`.
  * 6 seeds 42/43/44/100/101/102 (smoke first); single fixed seed-42 operating point (drive_gain/ratio/v_out_scale).

HONEST SCOPE: this biologizes step (1) `Wv @ LN(emb)` ONLY. Still host after a GO: the LN, the embedding LOOKUP, the r_h
gate, and every trained WEIGHT VALUE (Wv/decay/Wo_sp/head; the LEARNING rule is the separate 2026-08-12 e-prop GO), plus
the fixed unit scalars. On a GO the mouth's ENTIRE MATMUL CHAIN (Wv -> state -> Wo_sp -> r_h-shunt -> head_w -> head_b)
is a substrate graded-conductance read; the named next rung is LN(emb) (and then the e-prop weight learning). Runner-only,
default-off, NO `sim/` edit — drives + reads public bridge arrays; cfg.seed-controlled substrate (CLAUDE.md seed trap).

Run (calib):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_input_projection_substrate_derisk \
                --calib --seeds 42
Run (smoke):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_input_projection_substrate_derisk \
                --smoke --seeds 42
Run (6-seed): SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_input_projection_substrate_derisk \
                --seeds 42,43,44,100,101,102 --full-substrate \
                --json research/findings/raw/_wkv_input_projection_substrate_6seed.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.runners._wkv_fewspike_read_derisk import (  # noqa: E402
    WKVReadout, _softmax, _native, _load_eval,
)
from research.runners._wkv_graded_output_projection_derisk import GradedOutputProjection  # noqa: E402
from tools.lab import lever, void_if  # noqa: E402


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


class _ProjShim:
    """Minimal `ro`-like namespace GradedOutputProjection reads (only `.D` and `.Wo_sp`)."""
    def __init__(self, D, Wo_sp):
        self.D = int(D)
        self.Wo_sp = Wo_sp


class GradedInputProjection:
    """Realize `v = Wv @ LN(emb[tid])` as a SIGNED GRADED-CONDUCTANCE read by the four-quadrant reduction
    `Wv@x = [Wv,-Wv] @ [relu(x),relu(-x)]`, delegating to the validated `GradedOutputProjection` on a shim whose
    output-projection weight is `Wv_ext=[Wv,-Wv]`. Caches the substrate `v` per token id (v depends only on tid)."""

    def __init__(self, ro, seed, drive_gain=200.0, syn_scale=12.0, ratio=0.5, ou_std=40.0,
                 read_window=150, settle_frac=0.2, v_out_scale=1.0):
        self.ro = ro
        self.D = int(ro.D)
        self.v_out_scale = float(v_out_scale)
        Wv = ro.Wv.astype(np.float64)                                   # [D, D]
        Wv_ext = np.concatenate([Wv, -Wv], axis=1)                      # [D, 2D]  Wv@x = Wv_ext @ [x_pos, x_neg]
        shim = _ProjShim(self.D, Wv_ext)
        # reuse the validated signed graded-conductance read VERBATIM (pop=carrier_pop=1, as the projection GO)
        self.gop = GradedOutputProjection(shim, seed, pop=1, carrier_pop=1, ou_std=ou_std,
                                          read_window=read_window, drive_gain=drive_gain,
                                          syn_scale=syn_scale, ratio=ratio, settle_frac=settle_frac)
        self._cache = {}                                               # tid -> (v_sub_raw[D], v_pos_raw[D])

    @property
    def n_host_rng_draws(self):
        return int(self.gop.n_host_rng_draws)

    def _xstate(self, tid):
        x = self.ro._ln(self.ro.emb[tid])                              # [D] SIGNED LN(emb)
        return np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)])   # [2D] nonneg dual code

    def graded_v_raw(self, tid, zero_input=False, scramble_perm=None, use_cache=True):
        """Return (v_sub_raw[D], v_pos_raw[D]) in the substrate's ARBITRARY conductance-margin units (corr-relevant;
        scale applied separately by graded_v). Cached for the clean (non-lesioned) read."""
        if use_cache and (not zero_input) and (scramble_perm is None) and tid in self._cache:
            return self._cache[tid]
        xs = self._xstate(tid)
        hpre, hpre_pos = self.gop._graded_hpre(xs, scramble_perm=scramble_perm, zero_state=zero_input)
        if use_cache and (not zero_input) and (scramble_perm is None):
            self._cache[tid] = (hpre, hpre_pos)
        return hpre, hpre_pos

    def graded_v(self, tid, zero_input=False, scramble_perm=None):
        """Substrate `v` mapped to the HOST v magnitude via the fixed seed-42 scalar (for feeding the recurrence)."""
        hpre, _ = self.graded_v_raw(tid, zero_input=zero_input, scramble_perm=scramble_perm)
        return self.v_out_scale * hpre


# ====================================================================================================================
# Calibration — drive_gain + inh:exc ratio (maximize v reconstruction corr) + v_out_scale (least-squares magnitude
# match), ALL fit ONCE on seed 42 then FIXED for the unseen seeds.
# ====================================================================================================================
def _sample_tids(ro, ev_ids, warmup, n_tok):
    tids = []
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        for t in range(warmup, len(ids) - 1):
            tids.append(int(ids[t]))
            if len(tids) >= n_tok:
                return tids
    return tids


def _fit_v_out_scale(s, ro, tids):
    """Global least-squares scalar mapping the substrate margin -> host v magnitude (fixed once)."""
    num = 0.0; den = 0.0
    for tid in tids:
        vh = ro.v_of(tid)
        vs, _ = s.graded_v_raw(tid, use_cache=False)
        num += float(vh @ vs); den += float(vs @ vs)
    return num / max(1e-12, den)


def _calibrate(ro, seed, ev_ids, args):
    """Sweep drive_gain x ratio maximizing v_corr; print the plateau + the fitted v_out_scale at the best point."""
    tids = _sample_tids(ro, ev_ids, args.warmup, args.calib_tokens)
    print(f"[calib] {len(tids)} tokens; drive_gain x ratio sweep (corr of substrate v vs host Wv@LN(emb)):", flush=True)
    best = None
    for gain in [float(x) for x in args.calib_gains.split(",")]:
        for ratio in [float(x) for x in args.calib_ratios.split(",")]:
            s = GradedInputProjection(ro, seed, drive_gain=gain, syn_scale=args.syn_scale, ratio=ratio,
                                      ou_std=args.ou_std, read_window=args.read_window, settle_frac=args.settle_frac)
            cs = []; cps = []
            for tid in tids:
                vh = ro.v_of(tid)
                vs, vp = s.graded_v_raw(tid, use_cache=False)
                cs.append(_corr(vs, vh)); cps.append(_corr(vp, vh))
            c = float(np.mean(cs)); cp = float(np.mean(cps))
            print(f"    gain={gain:6.1f} ratio={ratio:4.2f} -> corr_signed={c:.4f} corr_pos={cp:.4f} "
                  f"(signed-pos={c-cp:+.4f})", flush=True)
            if best is None or c > best[0]:
                best = (c, gain, ratio, cp, s, tids)
    c, gain, ratio, cp, s, tids = best
    vscale = _fit_v_out_scale(s, ro, tids)
    print(f"[calib] BEST corr_signed={c:.4f} at gain={gain} ratio={ratio} (pos_only={cp:.4f}); "
          f"v_out_scale={vscale:.4f}", flush=True)


# ====================================================================================================================
# (a) reconstruction + (b1) subV -> host-pipe downstream (cheap, cached; autoregressive host state driven by sub-v).
# ====================================================================================================================
def _eval_recon_downstream(seed, ro, s, ev_ids, warmup, n_eval_pos, n_ac):
    D = ro.D; V = ro.V; chance = 1.0 / V
    acc = dict(n=0, corr=0.0, corr_pos=0.0, cos=0.0, corr_scr=0.0,
               b1_agree=0.0, b1_mass=0.0, b1_mass_ax=0.0)
    corr_min = 1.0
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap_h = np.zeros(D); an_h = np.zeros(D)      # HOST-v state (the reference mouth path)
        ap_s = np.zeros(D); an_s = np.zeros(D)      # SUBSTRATE-v-driven HOST state (B1)
        for t in range(len(ids) - 1):
            tid = ids[t]
            v_host = ro.v_of(tid)
            v_sub_raw, v_pos_raw = s.graded_v_raw(tid)               # arbitrary units (corr scale-free)
            v_sub = s.v_out_scale * v_sub_raw                        # host-magnitude (for the recurrence)
            ap_h = ro.decay * ap_h + np.maximum(v_host, 0.0); an_h = ro.decay * an_h + np.maximum(-v_host, 0.0)
            ap_s = ro.decay * ap_s + np.maximum(v_sub, 0.0);  an_s = ro.decay * an_s + np.maximum(-v_sub, 0.0)
            if t < warmup:
                continue
            # (a) reconstruction of v at this token
            c = _corr(v_sub_raw, v_host); corr_min = min(corr_min, c)
            acc["corr"] += c
            acc["corr_pos"] += _corr(v_pos_raw, v_host)
            acc["cos"] += _cosine(v_sub_raw, v_host)
            scr = np.random.default_rng(seed * 83 + 3 + positions).permutation(D)
            acc["corr_scr"] += _corr(v_sub_raw[scr], v_host)
            # (b1) full HOST mouth reference vs substrate-v-driven host state
            lg_h = ro.logits(ap_h, an_h, tid).copy()
            if ro.unk_idx >= 0:
                lg_h[ro.unk_idx] = -1e30
            host_am = int(np.argmax(lg_h)); pfull = _softmax(lg_h)
            lg_s = ro.logits(ap_s, an_s, tid).copy()
            if ro.unk_idx >= 0:
                lg_s[ro.unk_idx] = -1e30
            win = int(np.argmax(lg_s))
            acc["b1_agree"] += float(win == host_am)
            acc["b1_mass"] += pfull[win]; acc["b1_mass_ax"] += pfull[host_am]
            acc["n"] += 1; positions += 1
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    void_if(acc["n"] == 0, "no evaluable positions (every eval sentence shorter than warmup+2) — undefined")
    n = max(1, acc["n"])

    # ---- downstream cache-immune anti-cheats on B1: zero-input v + scramble-v (fresh autoregressive slices) ----
    def _b1_corrupt(kind):
        ag = 0; nn = 0
        for ids in ev_ids:
            if len(ids) < warmup + 2:
                continue
            ap2 = np.zeros(D); an2 = np.zeros(D); aph = np.zeros(D); anh = np.zeros(D)
            for t in range(min(len(ids) - 1, warmup + 30)):
                tid = ids[t]; v_host = ro.v_of(tid)
                if kind == "zero":
                    vsr, _ = s.graded_v_raw(tid, zero_input=True); vs = s.v_out_scale * vsr
                else:  # scramble the channel decode of the substrate v
                    perm = np.random.default_rng(seed * 61 + 7 + nn).permutation(D)
                    vsr, _ = s.graded_v_raw(tid, scramble_perm=perm); vs = s.v_out_scale * vsr
                ap2 = ro.decay * ap2 + np.maximum(vs, 0.0); an2 = ro.decay * an2 + np.maximum(-vs, 0.0)
                aph = ro.decay * aph + np.maximum(v_host, 0.0); anh = ro.decay * anh + np.maximum(-v_host, 0.0)
                if t < warmup:
                    continue
                lg_h = ro.logits(aph, anh, tid).copy()
                if ro.unk_idx >= 0:
                    lg_h[ro.unk_idx] = -1e30
                ham = int(np.argmax(lg_h))
                lg2 = ro.logits(ap2, an2, tid).copy()
                if ro.unk_idx >= 0:
                    lg2[ro.unk_idx] = -1e30
                ag += int(int(np.argmax(lg2)) == ham); nn += 1
                if nn >= n_ac:
                    break
            if nn >= n_ac:
                break
        return ag / max(1, nn)

    b1_zero = _b1_corrupt("zero")
    b1_scr = _b1_corrupt("scramble")

    lever("input_proj_signed_vs_positive_corr", before=round(acc["corr_pos"] / n, 4),
          after=round(acc["corr"] / n, 4), required=False)
    lever("input_proj_b1_argmax_vs_zeroinput", before=round(b1_zero, 4),
          after=round(acc["b1_agree"] / n, 4), required=False)

    m = {
        "seed": seed, "V": V, "D": D, "n_positions": acc["n"], "chance_1_over_v": round(chance, 6),
        "v_corr_signed": round(acc["corr"] / n, 4),
        "v_corr_signed_min": round(corr_min, 4),
        "v_corr_positive_only": round(acc["corr_pos"] / n, 4),
        "v_cosine": round(acc["cos"] / n, 4),
        "v_corr_scramble": round(acc["corr_scr"] / n, 4),
        "b1_downstream_argmax_agree": round(acc["b1_agree"] / n, 4),
        "b1_downstream_recov_argmax": round((acc["b1_mass"] / n) / max(1e-9, acc["b1_mass_ax"] / n), 4),
        "b1_downstream_argmax_zeroinput": round(b1_zero, 4),
        "b1_downstream_argmax_scramble": round(b1_scr, 4),
        "host_rng_draws_on_read_path": int(s.n_host_rng_draws),
        "n_unique_tids_cached": len(s._cache),
    }
    return m


# ====================================================================================================================
# (b2) FULL-SUBSTRATE pipeline: feed SUBSTRATE v into the substrate WKV state + composed substrate read. Two arms sharing
# the SAME (host-v-calibrated) state affine + read machinery — only the per-token drive differs (host-v vs sub-v). The
# reference is the exact HOST mouth (host v -> host recurrence -> host read).
# ====================================================================================================================
_STATE_CALIB = None


def _eval_full_substrate(seed, ro, s, ev_ids, warmup, n_eval_pos, args):
    from research.runners._wkv_graded_recurrent_state_derisk import (
        GradedRecurrentState, _ref_advance, _fit_calib, _cal_state,
    )
    from research.runners._wkv_mouth_endtoend_substrate_read_derisk import _build_proj, _build_read

    D = ro.D; V = ro.V; chance = 1.0 / V

    def _rv(v):
        return np.concatenate([np.maximum(v, 0.0), np.maximum(-v, 0.0)])

    def _mk_state():
        return GradedRecurrentState(D, seed, t_step=args.t_step, carrier_pop=args.carrier_pop,
                                    ou_std=args.wk_ou_std, drive_gain=args.wk_drive_gain,
                                    drive_bias=args.wk_drive_bias, syn_w=args.wk_syn_w, ssm_decay=ro.decay)
    s_wk_h = _mk_state(); s_wk_s = _mk_state()
    global _STATE_CALIB
    if _STATE_CALIB is None:
        _STATE_CALIB = _fit_calib(s_wk_h, ro, ev_ids, args.warmup, min(600, args.n_eval_pos), ro.decay)
        print(f"[calib-state on seed {seed}] scale.mean={_STATE_CALIB[0].mean():.4f}", flush=True)
    scale, off = _STATE_CALIB
    proj = _build_proj(ro, seed, args)
    s_read = _build_read(ro, seed, "composed_biaspop", args, proj)
    s_read._arm = "composed_biaspop"

    accH = dict(n=0, agree=0.0, mass=0.0, mass_ax=0.0)   # fullsub_hostV (host v -> sub state -> sub read)
    accS = dict(n=0, agree=0.0, mass=0.0, mass_ax=0.0)   # fullsub_subV  (sub  v -> sub state -> sub read)
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        s_wk_h.reset_state(); s_wk_s.reset_state()
        ap_ref = np.zeros(D); an_ref = np.zeros(D)
        for t in range(len(ids) - 1):
            tid = ids[t]
            v_host = ro.v_of(tid)
            v_sub = s.graded_v(tid)
            gH = s_wk_h.advance(_rv(v_host))
            gS = s_wk_s.advance(_rv(v_sub))
            ap_ref, an_ref = _ref_advance(ro, ap_ref, an_ref, tid)     # exact host recurrence (the mouth ceiling state)
            if t < warmup:
                continue
            lg_h = ro.logits(ap_ref, an_ref, tid).copy()
            if ro.unk_idx >= 0:
                lg_h[ro.unk_idx] = -1e30
            host_am = int(np.argmax(lg_h)); pfull = _softmax(lg_h)
            cstH = _cal_state(gH, scale, off); cstS = _cal_state(gS, scale, off)
            wH = s_read.read_endtoend(cstH[:D], cstH[D:], tid)["win"]
            wS = s_read.read_endtoend(cstS[:D], cstS[D:], tid)["win"]
            accH["n"] += 1; accH["agree"] += float(wH == host_am)
            accH["mass"] += (pfull[wH] if wH >= 0 else 0.0); accH["mass_ax"] += pfull[host_am]
            accS["n"] += 1; accS["agree"] += float(wS == host_am)
            accS["mass"] += (pfull[wS] if wS >= 0 else 0.0); accS["mass_ax"] += pfull[host_am]
            positions += 1
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    void_if(accS["n"] == 0, "no evaluable full-substrate positions")

    def _fin(a):
        n = max(1, a["n"])
        return dict(n=a["n"], argmax_agree=round(a["agree"] / n, 4),
                    recov_argmax=round((a["mass"] / n) / max(1e-9, a["mass_ax"] / n), 4))
    mH = _fin(accH); mS = _fin(accS)
    lever("fullsub_recov_subv_vs_hostv", before=mH["recov_argmax"], after=mS["recov_argmax"], required=False)
    return {"fullsub_hostV": mH, "fullsub_subV": mS,
            "read_host_rng_draws": int(s_read.n_host_rng_draws)}


def _verdict(m, full_substrate):
    chance = m["chance_1_over_v"]
    checks = {
        # (a) HEADLINE: the substrate input projection reconstructs Wv@LN(emb) at high fidelity (like the 0.984 output proj)
        "v_corr_signed_ge_0.90": m["v_corr_signed"] >= 0.90,
        "v_corr_min_ge_0.85": m["v_corr_signed_min"] >= 0.85,
        # the inhibitory shadow (negative weights) is LOAD-BEARING (weight AND input both ~50% signed).
        "signed_beats_positive_only": m["v_corr_signed"] > m["v_corr_positive_only"] + 0.05,
        "scramble_at_zero": abs(m["v_corr_scramble"]) < 0.1,
        # (b1) the substrate v carries the LM signal end-to-end through the mouth (autoregressive host pipe).
        "b1_argmax_gt_10x_chance": m["b1_downstream_argmax_agree"] > 10 * chance,
        "b1_recov_ge_0.85": m["b1_downstream_recov_argmax"] >= 0.85,
        "b1_zeroinput_collapses": m["b1_downstream_argmax_zeroinput"] <= 0.34 * m["b1_downstream_argmax_agree"],
        "b1_scramble_collapses": m["b1_downstream_argmax_scramble"] <= 0.34 * m["b1_downstream_argmax_agree"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
    }
    if full_substrate and "fullsub_subV" in m:
        mS = m["fullsub_subV"]; mH = m["fullsub_hostV"]
        checks["fullsub_subv_recov_ge_0.70"] = mS["recov_argmax"] >= 0.70
        # not degraded by adding the input-projection stage (subV within tol of the host-v full pipeline ~0.86).
        checks["fullsub_not_degraded"] = mS["recov_argmax"] >= mH["recov_argmax"] - 0.10
    checks = {k: bool(v) for k, v in checks.items()}
    return bool(all(checks.values())), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--n-eval-pos-full", type=int, default=120)     # bounded positions for the expensive B2 arms
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n-anticheat", type=int, default=120)
    # ---- input-projection graded read operating point (calibrated ONCE on seed 42; see --calib) ----
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--drive-gain", type=float, default=450.0)     # seed-42 calib (WIDE plateau; corr 0.984)
    ap.add_argument("--syn-scale", type=float, default=12.0)
    ap.add_argument("--ratio", type=float, default=0.5)            # inh:exc; seed-42 best, flat 0.3-0.7
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--v-out-scale", type=float, default=0.0867)   # margin -> host v magnitude (seed-42 least-squares)
    # ---- B2 full-substrate: WKV state operating point ([WK] state-GO seed-42 values) ----
    ap.add_argument("--full-substrate", action="store_true", help="run the B2 full-substrate composition arms")
    ap.add_argument("--t-step", type=int, default=40)
    ap.add_argument("--carrier-pop", type=int, default=24)
    ap.add_argument("--wk-drive-gain", type=float, default=40.0)
    ap.add_argument("--wk-drive-bias", type=float, default=40.0)
    ap.add_argument("--wk-syn-w", type=float, default=2.0)
    ap.add_argument("--wk-ou-std", type=float, default=60.0)
    # ---- B2 composed read operating point ([CE] read-GO seed-42 values) ----
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--read-ratio", type=float, default=0.3, dest="ratio_read")
    ap.add_argument("--hb-k", type=float, default=0.5)
    ap.add_argument("--proj-drive-gain", type=float, default=120.0)
    ap.add_argument("--proj-syn-scale", type=float, default=12.0)
    ap.add_argument("--proj-ratio", type=float, default=0.5)
    ap.add_argument("--proj-out-scale", type=float, default=0.30)
    ap.add_argument("--bias-scale", type=float, default=0.14)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    # calib
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib-tokens", type=int, default=120)
    ap.add_argument("--calib-gains", type=str, default="120,200,300,450")
    ap.add_argument("--calib-ratios", type=str, default="0.3,0.5,0.7,1.0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_input_projection_substrate.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_eval_pos = min(args.n_eval_pos, 60)
        args.n_eval_pos_full = min(args.n_eval_pos_full, 40)
        args.n_anticheat = min(args.n_anticheat, 40)
        args.n_sentences = min(args.n_sentences, 2500)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    if args.calib:
        seed = seeds[0]
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        ro = WKVReadout(ckpt)
        ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, 64)
        _calibrate(ro, seed, ev_ids, args)
        return

    t0 = time.time()
    results = []
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        s = GradedInputProjection(ro, seed, drive_gain=args.drive_gain, syn_scale=args.syn_scale,
                                  ratio=args.ratio, ou_std=args.ou_std, read_window=args.read_window,
                                  settle_frac=args.settle_frac, v_out_scale=args.v_out_scale)
        t1 = time.time()
        m = _eval_recon_downstream(seed, ro, s, ev_ids, args.warmup, args.n_eval_pos, args.n_anticheat)
        if args.full_substrate:
            # the imported read chain reads args.ratio for its OWN ratio; temporarily set it to the read ratio.
            saved_ratio = args.ratio
            args.ratio = args.ratio_read
            m.update(_eval_full_substrate(seed, ro, s, ev_ids, args.warmup, args.n_eval_pos_full, args))
            args.ratio = saved_ratio
        go, checks = _verdict(m, args.full_substrate)
        m["go"] = go; m["checks"] = checks; m["secs"] = round(time.time() - t1, 1)
        results.append(m)
        fs = ""
        if "fullsub_subV" in m:
            fs = (f" | B2 fullsub_subV recov={m['fullsub_subV']['recov_argmax']} "
                  f"(hostV {m['fullsub_hostV']['recov_argmax']}, agree {m['fullsub_subV']['argmax_agree']})")
        print(f"[seed {seed}] v_corr={m['v_corr_signed']} (min {m['v_corr_signed_min']}, "
              f"pos {m['v_corr_positive_only']}, cos {m['v_cosine']}, scr {m['v_corr_scramble']}) | "
              f"B1 agree={m['b1_downstream_argmax_agree']} recov={m['b1_downstream_recov_argmax']} "
              f"(zin {m['b1_downstream_argmax_zeroinput']}, scr {m['b1_downstream_argmax_scramble']})"
              f"{fs} GO={go} ({sum(checks.values())}/{len(checks)}) ({m['secs']}s)", flush=True)
        if not go:
            print(f"    checks: {json.dumps(checks)}", flush=True)

    if results:
        arr = lambda k: [r[k] for r in results]  # noqa: E731
        summary = {
            "n_seeds": len(results), "go_count": int(sum(r["go"] for r in results)),
            "v_corr_signed_mean": round(float(np.mean(arr("v_corr_signed"))), 4),
            "v_corr_signed_min": round(float(np.min(arr("v_corr_signed_min"))), 4),
            "v_corr_positive_only_mean": round(float(np.mean(arr("v_corr_positive_only"))), 4),
            "v_cosine_mean": round(float(np.mean(arr("v_cosine"))), 4),
            "v_corr_scramble_mean": round(float(np.mean(arr("v_corr_scramble"))), 4),
            "signed_load_bearing_count": int(sum(
                r["v_corr_signed"] > r["v_corr_positive_only"] + 0.05 for r in results)),
            "b1_argmax_agree_mean": round(float(np.mean(arr("b1_downstream_argmax_agree"))), 4),
            "b1_recov_mean": round(float(np.mean(arr("b1_downstream_recov_argmax"))), 4),
        }
        if args.full_substrate and all("fullsub_subV" in r for r in results):
            summary["fullsub_subV_recov_mean"] = round(
                float(np.mean([r["fullsub_subV"]["recov_argmax"] for r in results])), 4)
            summary["fullsub_subV_recov_min"] = round(
                float(np.min([r["fullsub_subV"]["recov_argmax"] for r in results])), 4)
            summary["fullsub_hostV_recov_mean"] = round(
                float(np.mean([r["fullsub_hostV"]["recov_argmax"] for r in results])), 4)
    else:
        summary = {"n_seeds": 0, "go_count": 0}

    out = {"results": _native(results), "summary": _native(summary), "seeds": seeds,
           "drive_gain": args.drive_gain, "ratio": args.ratio, "v_out_scale": args.v_out_scale,
           "read_window": args.read_window, "full_substrate": bool(args.full_substrate),
           "plasticity_off": True, "elapsed_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(_native(summary), indent=2)}", flush=True)
    print(f"[done] {len(results)} rows, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
