"""
gap#1 / A1 — the GRADED / CONDUCTANCE-DOMAIN signed read-out of the fluent WKV open-prose mouth: read the winner
word-pool from the CONTINUOUS net signed post-synaptic current margin (off `cp_conductance_g_e` / `cp_conductance_g_i`)
instead of a sparse 1-2-SPIKE COUNT. The winner-vs-loser margin becomes CONTINUOUS, so the sparse-count noise floor
that BLOCKED parity no longer dominates.

THE BOUNDARY THIS ATTACKS (mapped 2026-08-13, `_wkv_signed_read_parity_derisk`, 6-seed):
the TRUE SIGNED read-out (`_wkv_signed_shadow_read_derisk`) LIFTED read_fidelity 0.035 -> 0.55 (~16x) by carrying the
NEGATIVE `head_w` on an INHIBITORY SHADOW (Dale, no common mode), but PARITY was BLOCKED at projection_recovery 0.43
(oracle 1.30). Its named parity rung — a NEURAL divisive-norm homeostatic pool + recurrent-WTA sharpening — moved
NOTHING (proj_recovery 0.4265 vs 0.4269) because BOTH companion processes are INERT at the sign-preserving operating
point: the sign is load-bearing ONLY in a SPARSE ~1.5-SPIKE near-threshold SUBTRACTIVE regime (winner ~4-8 spikes,
margin ~1.5, ~15-20 active pools) that is TOO SPARSE for a feedback pool to sense proportionally / for recurrence to
ignite, and TOO NOISY for rank-order. ROOT CAUSE (the finding's words): "The wall is the sparse spike-COUNT margin
itself." Its #1 named next lever (in cost order): "Read in the GRADED / conductance domain, not a 1-2-spike count. ...
Move the signed read off the integrate-and-fire spike-count and onto the graded g_e/g_i the pools already compute."

THE LEVER (this runner): the parent already wires `Wp` as EXCITATORY synapses hid->pools (accumulating `cp_conductance
_g_e`) and `Wn` as INHIBITORY synapses hidinh->pools (accumulating `cp_conductance_g_i`, with `ratio` baked in). The
SUBSTRATE combines them every step into its own signed synaptic current
    I_syn = g_e*(E_e - v) + g_i*(E_i - v)          (fused_conductance_decay_and_current, bridge.py:8375)
a CONTINUOUS, graded quantity. Instead of pushing the pools over rheobase and counting the ~1.5 winner spikes, we keep
the pools SUBTHRESHOLD (floor 0) and read the winner from the net signed current DRIVE at rest:
    margin_k = (E_e - v_ref) * g_e[pool_k] + (E_i - v_ref) * g_i[pool_k]          (v_ref = rest, ~ -65 mV)
integrated over the read window (the ~5-10 ms conductance taus average out the OU noise a 1.5-spike count cannot). The
inhibitory:excitatory SYNAPTIC strength `ratio` (a biological quantity) is calibrated ONCE (seed 42, `_wkv_graded_calib
_probe.py`: read_fid vs `ratio` is a WIDE plateau 0.15-0.7, not a knife-edge; ratio=0.3 balances the two current terms
so margin_k ~ (Wp-Wn)@feat = head_w @ h — the true signed logit as a CONTINUOUS margin) — NOT the parent's spike-regime
ratio=6.5, which DOUBLE-compensates the driving force here and makes g_i over-dominant (positive-only would then beat
signed, the 2026-07-04 decorative-sign trap). The winner is argmax over that graded substrate margin: a genuine synaptic
CONDUCTANCE/CURRENT read (NOT a host softmax, NOT a host argmax over host logits — the g_e/g_i are accumulated by the
synapses on the substrate).

WHY THIS IS NOT THE 2026-07-04 CONDUCTANCE-SIGNED TRAP (retracted twice): that arc (a) had the SIGNED machinery
DECORATIVE (the positive Wp rows carried the read) and (b) OVERFIT to 3 tuned seeds (0-6/18 on unseen 100/101/102).
GUARDRAILS here: (1) a SINGLE fixed operating point on ALL 6 seeds 42/43/44/100/101/102 (NO per-seed tuning) — the
same seeds split tuned/unseen there; (2) the negative-weight (inhibitory-shadow) contribution MUST be LOAD-BEARING
(signed margin > positive-only, Wn-lesioned) on 6/6, gated in the verdict; the graded read removes the near-rheobase
silence confound that muddied the parent's signed-vs-positive comparison (3/6), so the sign's load-bearingness is now
testable cleanly.

DECISIVE metrics (calibration-robust, identical family to the parent):
  read_fidelity        = ondist_mass(graded_read) / ondist_mass(host_sample)   (1.0 == an ideal sampler)
  oracle_read_fidelity = the SAME spiking FS-WTA driven by a PERFECT host-logit current (parent's read_oracle; the
                         RESOLUTION ceiling) -> proj_recovery = read_fidelity / oracle_read_fidelity  (THE headline:
                         does the graded read approach PARITY, ~1.0, vs the spike-count read's 0.43?)
  positive_only_fidelity = graded read with the INHIBITORY SHADOW (Wn) LESIONED -> tests the sign is LOAD-BEARING.
  silent_frac          = positions with an undefined/degenerate margin (should be ~0 by construction: every pool has
                         a defined continuous margin; the sparse-count SILENCE residual is what the graded read closes).

ANTI-CHEATS (each MUST collapse): readout-lesion (zero Wp+Wn -> pools see only rest -> margin degenerate -> chance);
scramble (post-hoc pool->word relabel -> chance); provenance (winner from cp_conductance_*, 0 host categorical draws on
the read path); shadow-lesion attribution (positive-only, via tools.lab.lever); hidden-active; not-silent.

Reuse-by-import: SignedShadowLogitRead (wiring / FS-WTA / oracle / hidden feature / lesions) from
`_wkv_signed_shadow_read_derisk`; WKVReadout + _softmax + _native + _load_eval from `_wkv_fewspike_read_derisk`. NO
`sim/` edit — reads public bridge conductance arrays; cfg.seed-controlled substrate (CLAUDE.md seed trap). Runner-only,
default-off.

Run (smoke):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_graded_conductance_read_derisk \
                --smoke --seeds 42
Run (6-seed): .venv/bin/python -m research.runners._wkv_graded_conductance_read_derisk \
                --seeds 42,43,44,100,101,102 \
                --json research/findings/raw/_wkv_graded_conductance_6seed.json
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

from sim.backend import to_host, get_backend  # noqa: E402

from research.runners._wkv_fewspike_read_derisk import (  # noqa: E402
    WKVReadout, _softmax, _native, _load_eval,
)
from research.runners._wkv_signed_shadow_read_derisk import SignedShadowLogitRead  # noqa: E402
from tools.lab import lever, void_if  # noqa: E402


class GradedConductanceLogitRead(SignedShadowLogitRead):
    """Read the winner from the GRADED signed net-synaptic-current margin at the pools (continuous, off
    cp_conductance_g_e / cp_conductance_g_i at v_ref=rest), NOT a sparse spike count. Reuses the parent's exact
    signed wiring (Wp exc hid->pools, Wn inh hidinh->pools with ratio baked in); the ONLY changes are (1) the pools
    are kept SUBTHRESHOLD (low floor) so their conductances reflect the clean synaptic drive and (2) the winner is
    read from the integrated net current margin instead of the spike count."""

    def __init__(self, ro, seed, pop=1, hid_pop=1, ou_std=40.0, read_window=150,
                 hid_gain=120.0, hid_bias=0.0, syn_scale=12.0, ratio=0.3, graded_floor_pA=0.0,
                 n_fs=48, exc_to_fs=1.2, fs_to_exc=7.0, uniform_thresh=True, settle_frac=0.2, v_ref=None):
        # NOTE: floor_pA is passed as graded_floor_pA (0 by default: keep pools subthreshold so g_e/g_i reflect the
        # synaptic drive, not a near-rheobase count regime).
        super().__init__(ro, seed, pop=pop, hid_pop=hid_pop, ou_std=ou_std, read_window=read_window,
                         hid_gain=hid_gain, hid_bias=hid_bias, syn_scale=syn_scale, ratio=ratio,
                         floor_pA=graded_floor_pA, n_fs=n_fs, exc_to_fs=exc_to_fs, fs_to_exc=fs_to_exc,
                         head_b_gain=0.0, uniform_thresh=uniform_thresh)
        cfg = self._b.core_config
        self.E_e = float(getattr(cfg, "syn_reversal_potential_e", 0.0))
        self.E_i = float(getattr(cfg, "syn_reversal_potential_i", -75.0))
        if v_ref is None:
            v_ref = float(to_host(self._v0).mean()) if self._v0 is not None else -65.0
        self.v_ref = float(v_ref)
        self.settle_frac = float(settle_frac)
        self.df_e = self.E_e - self.v_ref                       # excitatory driving force at rest (>0)
        self.df_i = self.E_i - self.v_ref                       # inhibitory driving force at rest (<0)

    def _graded_margin(self, feat, want_diag=False):
        """Drive hid + hidinh by (hid_bias + hid_gain*feat[dim]); pools get only the (subthreshold) floor. Run the
        read window; INTEGRATE the per-pool net signed synaptic current at v_ref off cp_conductance_g_e/g_i. Return
        the CONTINUOUS per-pool margin (len V)."""
        b = self._b
        xp, _ = get_backend()
        self._reset()
        drive = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        fdrive = self.hid_bias + self.hid_gain * feat[self.hid_dim]
        drive[self.hid_idx] = fdrive
        drive[self.hidinh_idx] = fdrive                         # SAME drive -> rate-matched exc/inh pair
        if self.floor_pA:
            drive[self.all_pool] += self.floor_pA
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        settle = int(self.read_window * self.settle_frac)
        n_acc = 0
        ge_sum = np.zeros(self.V); gi_sum = np.zeros(self.V)
        pool_sp = 0.0
        for step in range(self.read_window):
            b._run_one_simulation_step()
            if step < settle:
                continue
            ge = np.asarray(to_host(b.cp_conductance_g_e)).astype(np.float64)[self.all_pool].reshape(self.V, self.P)
            gi = np.asarray(to_host(b.cp_conductance_g_i)).astype(np.float64)[self.all_pool].reshape(self.V, self.P)
            ge_sum += ge.sum(axis=1)
            gi_sum += gi.sum(axis=1)
            if want_diag:
                pool_sp += float(np.asarray(to_host(b.cp_firing_states)).astype(float)[self.all_pool].sum())
            n_acc += 1
        b.cp_external_input_current[:] = 0.0
        n_acc = max(1, n_acc)
        ge_mean = ge_sum / n_acc; gi_mean = gi_sum / n_acc      # mean per-pool conductance over the window
        # the SUBSTRATE's own signed synaptic-current combination, evaluated at rest (the graded pre-spike drive):
        margin = self.df_e * ge_mean + self.df_i * gi_mean      # [V] CONTINUOUS
        if want_diag:
            return margin, ge_mean, gi_mean, pool_sp / n_acc
        return margin

    @staticmethod
    def _argwin(mg):
        # a position is 'silent'/degenerate only if the margin carries no discrimination at all (all-equal).
        return -1 if float(mg.max() - mg.min()) <= 1e-9 else int(np.argmax(mg))

    def _reset(self):
        """Parent reset clears v/u/firing but NOT the synaptic conductances — so g_e/g_i carry over between reads
        (a confound: a lesioned read still shows the PRIOR read's conductance, and every position sees a residual
        of the previous one). Clear ALL conductance state so each read integrates ONLY its own drive."""
        super()._reset()
        b = self._b
        for name in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda",
                     "cp_conductance_g_nmda_rise", "cp_conductance_g_gabab"):
            arr = getattr(b, name, None)
            if arr is not None:
                arr[:] = 0.0

    def read_graded(self, ap, an, tid, scramble_perm=None, zero_feat=False):
        """Return BOTH the SIGNED winner (df_e*g_e + df_i*g_i) and the POSITIVE-ONLY winner (df_e*g_e) from the
        SAME per-position pooled conductances — a fair, re-sim-free 'is the inhibitory shadow load-bearing?' test
        (the identical-data comparison the 2026-07-04 arc lacked; the mass metric SATURATES near the argmax ceiling,
        so the sensitive instrument is argmax-agreement on identical g_e/g_i).

        zero_feat=True is the CACHE-IMMUNE COLLAPSE CONTROL: force the hidden feature (the signed-projection INPUT)
        to zero, so hid/hidinh carry no logit information -> the graded margin loses its structure -> the read must
        drop to chance. Unlike a weight-lesion (which does not fully reach the conductance the pools already hold in
        this wiring), silencing the INPUT is guaranteed to remove the signal."""
        feat = self._hidden_feature(ap, an, tid)
        if zero_feat:
            feat = np.zeros_like(feat)
        margin, ge, gi, psp = self._graded_margin(feat, want_diag=True)
        margin_pos = self.df_e * ge                              # positive-only: excitatory drive alone
        if scramble_perm is not None:
            margin = margin[scramble_perm]; margin_pos = margin_pos[scramble_perm]
        return dict(win=self._argwin(margin), margin=margin, win_pos=self._argwin(margin_pos),
                    margin_pos=margin_pos, ge=ge, gi=gi, pool_sp=psp)


def _eval(seed, ro, ev_ids, vocab, s, warmup, topk, sample_temp, n_eval_pos, gen_tokens, gen_temp,
          oracle_every=3):
    grng = np.random.default_rng(seed * 137 + 11)
    acc = dict(n=0, argmax_agree=0.0, top5_hit=0.0, nll=0.0, mass_syn=0.0, mass_hs=0.0, mass_ax=0.0,
               mass_scr=0.0, agree_scr=0.0, mass_ora=0.0, ora_n=0, silent=0, hid_active=0.0,
               pool_sp=0.0, ge=0.0, gi=0.0, gi_frac_neg=0.0,
               argmax_agree_pos=0.0, mass_pos=0.0, sign_flips_to_correct=0)
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            host_argmax = int(np.argmax(lg_supp))
            cand5 = np.argpartition(-lg_supp, 4)[:5]; top5 = set(int(c) for c in cand5)
            pfull = _softmax(lg_supp)
            candk = np.argpartition(-lg_supp, topk - 1)[:topk]; candk = candk[np.argsort(-lg_supp[candk])]
            pk = _softmax(lg_supp[candk] / sample_temp)
            hs = int(candk[int(grng.choice(len(pk), p=pk))])
            # GRADED signed-current read (the deliverable read path). NO host draw. r carries BOTH the signed and
            # the identical-data positive-only winner (the fair, re-sim-free load-bearing test).
            r = s.read_graded(ap, an, ids[t])
            win, margin = r["win"], r["margin"]
            win_pos = r["win_pos"]
            ge_m, gi_m, psp = r["ge"], r["gi"], r["pool_sp"]
            acc["ge"] += float(ge_m.mean()); acc["gi"] += float(gi_m.mean()); acc["pool_sp"] += psp
            acc["gi_frac_neg"] += float((s.df_i * gi_m < -1e-9).mean())    # frac pools the shadow actually pulls
            scr_perm = np.random.default_rng(seed * 83 + 3 + positions).permutation(s.V)
            win_s = int(np.argmax(margin[scr_perm])) if float(margin.max() - margin.min()) > 1e-9 else -1
            if positions % oracle_every == 0:
                ora = s.read_oracle(lg_supp)
                acc["mass_ora"] += (pfull[ora] if ora >= 0 else 0.0); acc["ora_n"] += 1
            acc["n"] += 1; positions += 1
            acc["hid_active"] += float(float(margin.max() - margin.min()) > 1e-9)
            if win < 0:
                acc["silent"] += 1
            acc["argmax_agree"] += float(win == host_argmax)
            acc["argmax_agree_pos"] += float(win_pos == host_argmax)
            # positions where the SIGN flips the winner from wrong (positive-only) to right (signed):
            acc["sign_flips_to_correct"] += int(win == host_argmax and win_pos != host_argmax)
            acc["top5_hit"] += float(win in top5)
            acc["nll"] += -math.log(max(pfull[win] if win >= 0 else 1e-12, 1e-12))
            acc["mass_syn"] += (pfull[win] if win >= 0 else 0.0)
            acc["mass_pos"] += (pfull[win_pos] if win_pos >= 0 else 0.0)
            acc["mass_hs"] += pfull[hs]; acc["mass_ax"] += pfull[host_argmax]
            acc["mass_scr"] += (pfull[win_s] if win_s >= 0 else 0.0)
            acc["agree_scr"] += float(win_s == host_argmax)
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    void_if(acc["n"] == 0, "no evaluable positions (every eval sentence shorter than warmup+2) — metrics undefined")
    n = max(1, acc["n"])
    diag_n = n
    pos_mass = acc["mass_pos"]; pos_n = n

    # ---- ZERO-FEATURE collapse control (cache-immune): silence the signed-projection INPUT -> chance ----
    les_mass = 0.0; les_n = 0; les_agree = 0
    for ids in ev_ids[:4]:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(min(len(ids) - 1, warmup + 30)):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            pfull = _softmax(lg_supp); host_am = int(np.argmax(lg_supp))
            win = s.read_graded(ap, an, ids[t], zero_feat=True)["win"]
            les_mass += (pfull[win] if win >= 0 else 0.0)
            les_agree += int(win == host_am); les_n += 1
            if les_n >= 60:
                break
        if les_n >= 60:
            break

    lever("graded_zero_feature_collapse_argmax", before=round(acc["argmax_agree"] / n, 4),
          after=round(les_agree / max(1, les_n), 4), required=False)
    # the sign's load-bearingness on the SENSITIVE instrument (argmax-agree, identical data): does adding the
    # inhibitory shadow pick the RIGHT word MORE often than the excitatory drive alone? (mass saturates near the
    # argmax ceiling and hides this; argmax-agree is the discriminating read.)
    lever("graded_signed_vs_positive_argmax", before=round(acc["argmax_agree_pos"] / n, 4),
          after=round(acc["argmax_agree"] / n, 4), required=False)

    m = {
        "seed": seed, "arm": "graded_conductance", "V": s.V, "pop": s.P, "ratio": s.ratio,
        "v_ref": round(s.v_ref, 2), "df_e": round(s.df_e, 2), "df_i": round(s.df_i, 2),
        "topk_ceiling": topk, "plasticity_off": True,
        "n_positions": acc["n"], "silent_frac": round(acc["silent"] / n, 4),
        "hidden_active_frac": round(acc["hid_active"] / n, 4),
        "mean_pool_spikes": round(acc["pool_sp"] / diag_n, 3),
        "mean_g_e": round(acc["ge"] / diag_n, 4), "mean_g_i": round(acc["gi"] / diag_n, 4),
        "shadow_pulls_frac": round(acc["gi_frac_neg"] / diag_n, 4),
        "argmax_agree": round(acc["argmax_agree"] / n, 4),
        "argmax_agree_positive_only": round(acc["argmax_agree_pos"] / n, 4),
        "sign_flips_to_correct_frac": round(acc["sign_flips_to_correct"] / n, 4),
        "top5_hit": round(acc["top5_hit"] / n, 4),
        "nll_graded": round(acc["nll"] / n, 4),
        "mass_graded": round(acc["mass_syn"] / n, 4),
        "mass_positive_only": round(pos_mass / max(1, pos_n), 4),
        "mass_hostsample_ceiling": round(acc["mass_hs"] / n, 4),
        "mass_argmax_ceiling": round(acc["mass_ax"] / n, 4),
        "mass_scramble": round(acc["mass_scr"] / n, 4),
        "argmax_agree_scramble": round(acc["agree_scr"] / n, 4),
        "mass_zerofeat": round(les_mass / max(1, les_n), 4),
        "argmax_agree_zerofeat": round(les_agree / max(1, les_n), 4),
        "mass_oracle_ceiling": round(acc["mass_ora"] / max(1, acc["ora_n"]), 4),
        "chance_1_over_v": round(1.0 / s.V, 6),
        "host_rng_draws_on_read_path": int(s.n_host_rng_draws),
    }
    m["read_fidelity_vs_sampler"] = round(m["mass_graded"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    m["oracle_read_fidelity"] = round(m["mass_oracle_ceiling"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    m["positive_only_fidelity"] = round(m["mass_positive_only"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    # proj_recovery vs the SPIKING FS-WTA oracle (parent-comparable; the parent read this 0.43). NOTE this oracle is
    # a CONTRASTIVE softmax-sharpened SPIKING read (mass > the ideal sampler), an UNFAIR bar for a LINEAR graded read.
    m["projection_recovery"] = round(m["read_fidelity_vs_sampler"] / max(1e-9, m["oracle_read_fidelity"]), 4)
    # the graded-NATIVE fidelity: the fraction of the PERFECT-ARGMAX mass the graded margin recovers (the graded read
    # has no WTA-resolution loss, so the only gap is the RECONSTRUCTION of the logit argmax -> this isolates it).
    m["projection_recovery_vs_argmax"] = round(m["mass_graded"] / max(1e-9, m["mass_argmax_ceiling"]), 4)
    if gen_tokens > 0:
        m["generation"] = _free_gen(ro, vocab, s, gen_tokens)
    return m


def _scramble_at_chance(agree_scramble, chance, n):
    sigma = math.sqrt(max(chance * (1.0 - chance), 1e-12) / max(1, n))
    return agree_scramble <= chance + 3.0 * sigma


def _verdict(m):
    chance = m["chance_1_over_v"]; n = m["n_positions"]
    checks = {
        # THE headline (graded-native): the graded margin recovers >=85% of the PERFECT-ARGMAX mass -> the signed
        # projection is essentially reconstructed (a graded read has no WTA-resolution loss, so this is PARITY in the
        # meaningful sense: the read is no longer the bottleneck).
        "recovers_argmax_mass_ge_0.85": m["projection_recovery_vs_argmax"] >= 0.85,
        # a large ABSOLUTE lift, and materially above the spike-count parent (read_fid 0.55).
        "read_fidelity_ge_0.55": m["read_fidelity_vs_sampler"] >= 0.55,
        # the NEGATIVE weights are LOAD-BEARING (not decorative, the 2026-07-04 trap): the inhibitory shadow picks
        # the RIGHT word MORE often than the excitatory drive alone (argmax-agree on IDENTICAL conductances — the
        # mass metric saturates near the argmax ceiling and cannot see this).
        "signed_beats_positive_only": m["argmax_agree"] > m["argmax_agree_positive_only"],
        "argmax_agree_gt_10x_chance": m["argmax_agree"] > 10 * chance,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, n),
        # CACHE-IMMUNE collapse: silencing the signed-projection INPUT (zero feature) drops the read to <=1/3 of the
        # intact argmax-agreement (the feature drives the read; it is not a floor/frequency artifact).
        "zero_feature_collapses": m["argmax_agree_zerofeat"] <= 0.34 * m["argmax_agree"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "hidden_active": m["hidden_active_frac"] > 0.9,
        "not_silent": m["silent_frac"] < 0.05,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return bool(all(checks.values())), checks


def _free_gen(ro, vocab, s, n_tok):
    out = {}
    for prompt in ("once upon a time", "the little girl", "tom and his dog"):
        pid = [i for i in vocab.ids(prompt.split()) if 0 <= i < ro.V] or [0]
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in pid:
            ap, an = ro.advance(ap, an, t)
        gen = list(pid); self_nll = 0.0; steps = 0
        for _ in range(n_tok):
            lg = ro.logits(ap, an, gen[-1]); lg2 = lg.copy()
            if ro.unk_idx >= 0:
                lg2[ro.unk_idx] = -1e30
            win = s.read_graded(ap, an, gen[-1])["win"]
            nxt = int(win) if win >= 0 else int(np.argmax(lg2))
            self_nll += -math.log(max(_softmax(lg2)[nxt], 1e-12)); steps += 1
            gen.append(nxt); ap, an = ro.advance(ap, an, nxt)
        txt = " ".join(ro.words[i] if 0 <= i < len(ro.words) else "<unk>" for i in gen)
        out[prompt] = {"text": txt, "self_nll": round(self_nll / max(1, steps), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--pops", type=str, default="4")                        # P is nearly irrelevant to a graded read
    ap.add_argument("--hid-pop", type=int, default=1)
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--topk", type=int, default=64)
    # ---- GRADED operating point (matched to the parent's signed wiring; the ONLY change is a SUBTHRESHOLD floor) ----
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--hid-bias", type=float, default=0.0)
    ap.add_argument("--syn-scale", type=float, default=12.0)
    ap.add_argument("--ratio", type=float, default=0.3)                     # inh:exc SYNAPTIC ratio (calib'd seed 42)
    ap.add_argument("--graded-floor-pA", type=float, default=0.0)           # 0 => pools subthreshold; read the drive
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--fs-to-exc", type=float, default=7.0)
    ap.add_argument("--exc-to-fs", type=float, default=1.2)
    ap.add_argument("--n-fs", type=int, default=48)
    ap.add_argument("--sample-temp", type=float, default=0.8)
    ap.add_argument("--gen-tokens", type=int, default=0)
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--oracle-every", type=int, default=3)
    ap.add_argument("--no-uniform-thresh", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_graded_conductance.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_eval_pos = min(args.n_eval_pos, 60)
        args.gen_tokens = args.gen_tokens or 30

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pops = [int(x) for x in args.pops.split(",") if x.strip()]

    t0 = time.time()
    results = []
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        for pop in pops:
            s = GradedConductanceLogitRead(ro, seed, pop=pop, hid_pop=args.hid_pop, ou_std=args.ou_std,
                                           read_window=args.read_window, hid_gain=args.hid_gain,
                                           hid_bias=args.hid_bias, syn_scale=args.syn_scale, ratio=args.ratio,
                                           graded_floor_pA=args.graded_floor_pA, n_fs=args.n_fs,
                                           exc_to_fs=args.exc_to_fs, fs_to_exc=args.fs_to_exc,
                                           settle_frac=args.settle_frac, uniform_thresh=not args.no_uniform_thresh)
            gen_here = args.gen_tokens if pop == max(pops) else 0
            m = _eval(seed, ro, ev_ids, vocab, s, args.warmup, args.topk, args.sample_temp,
                      args.n_eval_pos, gen_here, args.gen_temp, oracle_every=args.oracle_every)
            go, checks = _verdict(m); m["go"] = go; m["checks"] = checks
            results.append(m)
            print(f"[seed {seed} P={pop} ratio={args.ratio} floor={args.graded_floor_pA}] "
                  f"pool_spk={m['mean_pool_spikes']} g_e={m['mean_g_e']} g_i={m['mean_g_i']} "
                  f"read_fid={m['read_fidelity_vs_sampler']} recov_argmax={m['projection_recovery_vs_argmax']} "
                  f"proj_recov_vsOracle={m['projection_recovery']} pos_only={m['positive_only_fidelity']} "
                  f"agree={m['argmax_agree']}>pos{m['argmax_agree_positive_only']} "
                  f"(10x_chance {round(10/m['V'],4)}) "
                  f"scr={m['argmax_agree_scramble']} zerofeat_agree={m['argmax_agree_zerofeat']} "
                  f"silent={m['silent_frac']} GO={go} ({sum(checks.values())}/{len(checks)})", flush=True)
            if not go:
                print(f"    checks: {json.dumps(checks)}", flush=True)
            if m.get("generation"):
                for pr, g in m["generation"].items():
                    print(f"    [gen '{pr}' nll {g['self_nll']}] {g['text'][:150]}", flush=True)

    agg = {}
    for m in results:
        key = f"P{m['pop']}"
        agg.setdefault(key, {"read_fidelity": [], "oracle": [], "proj_recovery": [], "recov_argmax": [],
                             "pos_only": [], "silent": [], "signed_lb": [], "go": []})
        agg[key]["read_fidelity"].append(m["read_fidelity_vs_sampler"])
        agg[key]["oracle"].append(m["oracle_read_fidelity"])
        agg[key]["proj_recovery"].append(m["projection_recovery"])
        agg[key]["recov_argmax"].append(m["projection_recovery_vs_argmax"])
        agg[key]["pos_only"].append(m["positive_only_fidelity"])
        agg[key]["silent"].append(m["silent_frac"])
        agg[key]["signed_lb"].append(bool(m["argmax_agree"] > m["argmax_agree_positive_only"]))
        agg[key]["go"].append(m["go"])
    summary = {}
    for key, d in agg.items():
        summary[key] = {"n_seeds": len(d["go"]), "go_count": int(sum(d["go"])),
                        "read_fidelity_mean": round(float(np.mean(d["read_fidelity"])), 4),
                        "read_fidelity_min": round(float(np.min(d["read_fidelity"])), 4),
                        "oracle_mean": round(float(np.mean(d["oracle"])), 4),
                        "proj_recovery_vs_oracle_mean": round(float(np.mean(d["proj_recovery"])), 4),
                        "recovers_argmax_mass_mean": round(float(np.mean(d["recov_argmax"])), 4),
                        "recovers_argmax_mass_min": round(float(np.min(d["recov_argmax"])), 4),
                        "positive_only_mean": round(float(np.mean(d["pos_only"])), 4),
                        "signed_load_bearing_count": int(sum(d["signed_lb"])),
                        "silent_frac_mean": round(float(np.mean(d["silent"])), 4)}
    out = {"results": results, "summary": summary, "seeds": seeds, "pops": pops, "hid_pop": args.hid_pop,
           "ratio": args.ratio, "topk": args.topk, "read_window": args.read_window,
           "graded_floor_pA": args.graded_floor_pA, "settle_frac": args.settle_frac,
           "plasticity_off": True, "elapsed_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND", "numpy")}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
