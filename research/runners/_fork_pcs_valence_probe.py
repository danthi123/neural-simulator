"""_fork_pcs_valence_probe — AGI-fork AFFECT arc, stage C1: the VALENCE-PRESENCE probe.

The "cheapest decisive falsifier" from the adversarial skeptic (design doc
`docs/plans/2026-09-07-fork-affect-selfaware-arc-design.md`, sections "DESIGN — emergent
AFFECT/valence" + "ADVERSARIAL SKEPTIC"). It asks a NARROW, pre-registered question of the
substrate's now-exposed, byte-identical valence read-outs (`sub.last_valence_rpe` = reward −
reward-head prediction; `sub.last_valence_adv` = actor-critic advantage; committed 691c0914):

  Is the substrate-native reward-PREDICTION-ERROR signal PRESENT in the shared recurrent core
  (population-decodable above 3 floors), SOUND per-unit (split-half-stable AND significant vs a
  CYCLIC-SHIFT null — NOT an i.i.d. shuffle), and does it show a genuine EXPECTATION-VIOLATION
  signature (reward omission at an anticipated-reward step drives the read-out negative)?

⚠️ WHAT THIS IS NOT (the honesty boundary + the dominant skeptic risk — the TAUTOLOGY trap).
`last_valence_adv` IS the policy-gradient multiplier by construction, so its presence/decodability
is EXPECTED and is NOT by itself evidence of an emergent faculty. Even `last_valence_rpe` contains
a `−rhat` term that is linear in h_t by construction, so part of its decodability is expected too.
Therefore this C1 probe tests only PRESENCE + SOUNDNESS + EXPECTATION-VIOLATION of the reward-PE
read-out. The LOAD-BEARING faculty claim (necessity/sufficiency via a channel the signal was NOT
already wired into) is a LATER stage and is deliberately NOT made here. Every read-out below is a
FUNCTIONAL instrument reading ("the substrate's reward-prediction-error read-out is negative here"),
never a phenomenal claim ("the agent feels bad").

REUSES (verbatim, does not modify) the proven anti-hollow machinery of
`research.runners._fork_pcs_emergence_derisk`: `_ridge_weights`, `_r2_with_floors`, `_beats_floors`,
`replay_untrained` (the untrained-reservoir floor), `_quantile_bin_edges`, the constants
`FLOOR_MARGIN=0.05` / `PLACE_SI_STABILITY_THRESH=0.30`, and the split-half Skaggs-SI tuning
formalism (re-implemented here with the required CYCLIC-SHIFT null instead of the file's i.i.d.
shuffle — the one place the design flags the emergence file's null as unsound for a temporally
autocorrelated signal like valence).

Run:
  # CPU smoke (end-to-end, 1 seed, tiny — confirms JSON has all fields, no crash):
  SIM_BACKEND=numpy PYTHONPATH=. python -m research.runners._fork_pcs_valence_probe \
      --seeds 42 --n-hidden 64 --n-train 3000 --out /tmp/_val_smoke.json
  # full 6-seed decisive run (GPU lane — queue it; 0 agent tokens):
  SIM_BACKEND=cupy python -m research.runners._fork_pcs_valence_probe \
      --seeds 42 43 44 100 101 102 --units rate --n-hidden 128 --n-train 200000 --value-weight 1.0 \
      --out research/findings/raw/_fork_pcs_valence_presence_6seed.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig
from research.runners.fork_pcs_world import WorldConfig, ForkPCSWorld, N_ACTIONS, _ridge_r2
# Reuse the emergence battery's proven, tested machinery verbatim (import, do not reinvent).
from research.runners._fork_pcs_emergence_derisk import (
    _ridge_weights, _r2_with_floors, _beats_floors, replay_untrained, _quantile_bin_edges,
    _host, _f, FLOOR_MARGIN, PLACE_SI_STABILITY_THRESH,
)
# ATTRIBUTION (not just measurement): the soundness step measures a trained-core vs untrained-reservoir
# treatment/control pair; attributable_to asks whose the difference is, rather than letting the raw ratio
# stand (gap#5 banked both arms one key apart for weeks — the subtraction is the finding).
from tools.lab import attributable_to

# ── pre-registered bars (fixed BEFORE the multi-seed run) ────────────────────
VALENCE_ABS_BAR = FLOOR_MARGIN          # population presence: held-out R^2 must be >= this AND beat each of
#                                          the 3 floors (untrained / raw-V1 / shuffle) by FLOOR_MARGIN.
CYCLIC_SHIFT_DRAWS = 100                # cyclic-shift null draws for the per-unit significance test
VALENCE_SI_N_BINS = 5                  # quantile bins the valence label is discretized into for tuning
STAB_TRAINED_OVER_UNTRAINED = 2.0      # trained #valence-selective units must be >= this x the untrained
#                                          reservoir's (the grew-through-training, not-relabeled control).
SEEDS_REQUIRED_FRAC = 5.0 / 6.0        # >= 5/6 seeds must pass for the aggregate GO
# The signal the GO gate is built on is the reward-PREDICTION-ERROR read-out (last_valence_rpe): the
# always-available (incl. eval), omission-testable, non-tautological one. `adv` presence is reported
# ALONGSIDE for completeness but is explicitly NOT gated on (see the tautology banner above).
GO_SIGNAL = "rpe"


# ─────────────────────────────────────────────────────────────────────────────
# FROZEN probe rollout — collect h_t, raw V1, position, and the valence read-outs per step.
# ─────────────────────────────────────────────────────────────────────────────
def _collect_probe(world, sub, n_steps, explore_eps):
    """Drive the FROZEN substrate against the world, collecting per-step traces.

    Mirrors `_fork_pcs_emergence_derisk.rollout`'s observe->act->step->learn ordering and its exact
    input capture (for the untrained replay), and ADDS the valence read-outs:
      RPE   sub.last_valence_rpe  (reward - reward-head prediction; updated inside learn() EVEN when
            frozen — verified in sim/pcs_substrate.py:611-613 — so it is a genuine per-step eval read).
      ADV   the actor-critic advantage. NOTE: sub.last_valence_adv is written ONLY on the non-frozen
            policy branch (sim/pcs_substrate.py:630-640), so under freeze the public field goes stale.
            To keep a meaningful per-step label WITHOUT unfreezing (which would change weights) or
            editing the substrate, we reconstruct it from the substrate's OWN exposed intermediate
            reads using the substrate's OWN formula, byte-identical arithmetic:
              adv = (reward + curiosity_beta * _last_lp) - V(h_t)         [substrate lines 631,636]
            where V(h_t)=sub._last_value and _last_lp=sub._last_lp are both set by act()/frozen-stable.
      VHAT  V(h_t) = sub._last_value (the value-head estimate; None if value_weight==0) — the nuisance
            regressed out in the collinearity control.
    """
    sub.freeze()
    sub.set_lesion_mask(None)
    H, POS, RAW, RPE, ADV, VHAT, REW, ATE, INSEQ = [], [], [], [], [], [], [], [], []
    a_prev = -1
    beta = float(sub.cfg.curiosity_beta)
    for _ in range(n_steps):
        d = world.drive_afferent()
        v1 = world.crop_v1feat()
        v1_host = np.asarray(v1.get() if hasattr(v1, "get") else v1, dtype=np.float32)
        d_host = np.asarray(d, dtype=np.float32)
        pos_before = world.agent
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=explore_eps)
        v_hat = sub._last_value                     # V(h_t), set by act(); None when no value head
        lp = float(sub._last_lp)                    # learning-progress (frozen-stable)
        r, info = world.step(a)
        sub.learn(r)                                # updates sub.last_valence_rpe (frozen-safe)
        rpe = sub.last_valence_rpe
        adv = ((float(r) + beta * lp) - float(v_hat)) if v_hat is not None else float("nan")
        H.append(_host(h).astype(np.float32))
        POS.append(np.asarray(pos_before, dtype=np.float32))
        RAW.append(v1_host)
        RPE.append(float(rpe) if rpe is not None else float("nan"))
        ADV.append(adv)
        VHAT.append(float(v_hat) if v_hat is not None else float("nan"))
        REW.append(float(r))
        ATE.append(1.0 if info.get("ate") else 0.0)
        INSEQ.append((v1_host, a_prev, d_host))
        a_prev = a
    return {"H": np.asarray(H), "POS": np.asarray(POS), "RAW": np.asarray(RAW),
            "RPE": np.asarray(RPE, np.float64), "ADV": np.asarray(ADV, np.float64),
            "VHAT": np.asarray(VHAT, np.float64), "REW": np.asarray(REW, np.float64),
            "ATE": np.asarray(ATE, np.float64), "INPUT_SEQ": INSEQ}


# ─────────────────────────────────────────────────────────────────────────────
# Per-unit SOUNDNESS: split-half Skaggs-SI tuning + the CORRECT (cyclic-shift) null.
# ─────────────────────────────────────────────────────────────────────────────
def _si_bits_per_activation(R, bin_idx, n_bins, counts, occ, p_i, lam, live):
    """Skaggs bits-per-activation of each unit's rectified rate over the label bins (KL(q||p) >= 0).
    Magnitude-normalized: firing MORE does not raise it, only CONCENTRATION of firing by bin does."""
    T = R.shape[0]
    B = np.zeros((n_bins, T), dtype=np.float64)
    B[bin_idx, np.arange(T)] = 1.0
    sum_R = B @ R                                    # (n_bins, U)
    lam_i = np.zeros_like(sum_R)
    lam_i[occ] = sum_R[occ] / counts[occ, None]
    lam_safe = np.where(live, lam, 1.0)
    ratio = lam_i / lam_safe[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = p_i[:, None] * ratio * np.log2(ratio)
    contrib[~np.isfinite(contrib)] = 0.0
    si = contrib.sum(axis=0)
    si[~live] = 0.0
    return si


def _valence_selectivity_metrics(H, valence, seed, n_bins=VALENCE_SI_N_BINS,
                                 n_shift=CYCLIC_SHIFT_DRAWS, stab_thresh=PLACE_SI_STABILITY_THRESH):
    """FLOOR-INDEPENDENT per-unit valence tuning, with the CYCLIC-SHIFT null the design requires.

    Same Skaggs-SI + split-half-stability formalism as `_fork_pcs_emergence_derisk._place_cell_metrics`,
    retargeted from spatial (x,y) bins to 5 quantile bins of the valence label. The ONE deliberate
    deviation from that file: the significance null is a CYCLIC SHIFT of the valence-label time series
    (np.roll by a random offset), NOT an i.i.d. permutation. Valence is temporally autocorrelated (a
    reward event depresses/raises the read-out for several steps), so an i.i.d. shuffle destroys that
    autocorrelation on the label side and INFLATES apparent significance; a cyclic shift preserves the
    label's marginal AND autocorrelation and destroys only its cross-alignment with h_t.

    A unit is 'valence-selective' iff split-half stability > stab_thresh AND its real SI exceeds its
    OWN cyclic-shift 95th percentile. Returns population summaries so trained vs untrained read side by
    side. HONESTY: a functional tuning read-out; asserts nothing about felt valence."""
    H = np.asarray(H, dtype=np.float64)
    valence = np.asarray(valence, dtype=np.float64).reshape(-1)
    if H.ndim != 2 or len(H) < 50:
        return {"note": "too few probe steps", "n_steps": int(len(H)) if H.ndim == 2 else 0}
    finite = np.isfinite(valence)
    if finite.sum() < 50:
        return {"note": "too few finite valence labels", "n_finite": int(finite.sum())}
    H = H[finite]; valence = valence[finite]
    edges, eff_bins = _quantile_bin_edges(valence, n_bins)
    if eff_bins < 2:
        return {"note": "insufficient valence spread for quantile binning",
                "n_unique_values": int(len(np.unique(valence)))}
    bin_idx = np.clip(np.digitize(valence, edges[1:-1], right=False), 0, eff_bins - 1).astype(np.int64)
    T, U = H.shape
    R = np.maximum(0.0, H)                            # non-negative rate proxy (rectified activation)
    counts = np.bincount(bin_idx, minlength=eff_bins).astype(np.float64)
    occ = counts > 0
    p_i = counts / counts.sum()
    lam = R.mean(axis=0)
    live = lam > 1e-9

    si_real = _si_bits_per_activation(R, bin_idx, eff_bins, counts, occ, p_i, lam, live)

    # CYCLIC-SHIFT null: roll the label sequence by a random offset (marginal + autocorrelation preserved;
    # only the h<->valence alignment destroyed). Occupancy/overall-mean stay invariant, only lam_i moves.
    rng = np.random.default_rng(seed + 8123)
    si_shift = np.empty((n_shift, U), dtype=np.float64)
    for s in range(n_shift):
        off = int(rng.integers(1, T))                # non-zero offset
        bi = np.roll(bin_idx, off)
        # roll preserves counts/occ/p_i exactly, so reuse them
        si_shift[s] = _si_bits_per_activation(R, bi, eff_bins, counts, occ, p_i, lam, live)
    thresh_u = np.percentile(si_shift, 95, axis=0)   # (U,) per-unit cyclic-shift 95th pctile

    # split-half stability: per-unit bin-mean rate map on first vs second temporal half, Pearson-correlate
    def _binmeans(sl):
        bi = bin_idx[sl]
        c = np.bincount(bi, minlength=eff_bins).astype(np.float64)
        Bh = np.zeros((eff_bins, len(bi))); Bh[bi, np.arange(len(bi))] = 1.0
        o = c > 0
        bm = np.full((eff_bins, U), np.nan)
        bm[o] = (Bh @ R[sl])[o] / c[o, None]
        return bm, o

    half = T // 2
    bm1, o1 = _binmeans(slice(0, half))
    bm2, o2 = _binmeans(slice(half, T))
    both = o1 & o2
    stability = np.full(U, np.nan)
    if int(both.sum()) >= 2:
        a = bm1[both]; b = bm2[both]
        am = a - a.mean(0); bm = b - b.mean(0)
        denom = np.sqrt((am * am).sum(0) * (bm * bm).sum(0)) + 1e-12
        stability = (am * bm).sum(0) / denom
        stability[~live] = np.nan

    is_sel = live & (si_real > thresh_u) & (np.nan_to_num(stability, nan=-1.0) > stab_thresh)
    n_sel = int(is_sel.sum())
    live_any = bool(live.any())
    mean_si_live = float(np.nanmean(si_real[live])) if live_any else float("nan")
    return {
        "n_steps": int(T), "n_units": int(U), "n_live_units": int(live.sum()),
        "n_bins_requested": int(n_bins), "n_bins_effective": int(eff_bins),
        "n_shift": int(n_shift), "stability_thresh": stab_thresh,
        "mean_si": _f(mean_si_live) if live_any else None,
        "max_si": _f(np.nanmax(si_real[live])) if live_any else None,
        "mean_si_cyclic_shift": _f(float(si_shift.mean())),
        "si95_cyclic_shift_pooled": _f(float(np.percentile(si_shift, 95))),
        "si_real_over_shift_ratio": _f(mean_si_live / (float(si_shift.mean()) + 1e-9)) if live_any else None,
        "mean_stability": _f(float(np.nanmean(stability))) if np.isfinite(stability).any() else None,
        "n_valence_selective_units": n_sel,
        "frac_valence_selective_units": _f(n_sel / U),
        "mean_stability_selective": _f(float(np.nanmean(stability[is_sel]))) if n_sel > 0 else None,
        "mean_si_selective": _f(float(np.nanmean(si_real[is_sel]))) if n_sel > 0 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anti-collinearity control: regress position + value out of the valence label, re-decode the residual.
# ─────────────────────────────────────────────────────────────────────────────
def _regress_out_and_decode(H, valence, POS, VHAT, seed):
    """Partial out position (x,y) and the value-head estimate V(h_t) from the valence label, then ridge-
    decode the RESIDUAL from h_t (held-out R^2). The skeptic's Trap-3 (value/position collinearity): a
    'valence' decoder can just be re-reading distance-to-food / value. If the residual (= genuine
    better/worse-THAN-EXPECTED, i.e. prediction error) still decodes above its shuffle floor, the signal
    is not merely position/value re-expressed; if it vanishes, we SAY SO. Returns the residual decode R^2,
    a shuffle floor on the residual, and a survives flag."""
    v = np.asarray(valence, dtype=np.float64).reshape(-1)
    finite = np.isfinite(v)
    Hf, vf, POSf, VHf = H[finite], v[finite], POS[finite], np.asarray(VHAT, np.float64)[finite]
    n = len(vf)
    if n < 40:
        return {"note": "too few finite labels for regress-out", "n": int(n)}
    cols = [POSf[:, 0], POSf[:, 1]]
    used = ["pos_x", "pos_y"]
    if np.isfinite(VHf).all():
        cols.append(VHf); used.append("value_head")
    Nz = np.column_stack(cols + [np.ones(n)])
    beta, *_ = np.linalg.lstsq(Nz, vf, rcond=None)
    resid = vf - Nz @ beta
    perm = np.random.default_rng(seed + 5).permutation(n)
    cut = int(0.7 * n); tr, te = perm[:cut], perm[cut:]
    r2_resid = _ridge_r2(Hf[tr], resid[tr], Hf[te], resid[te])
    sh = np.random.default_rng(seed + 9).permutation(n)
    r2_sh = _ridge_r2(Hf[tr], resid[sh][tr], Hf[te], resid[sh][te])
    survives = (not np.isnan(r2_resid)) and (r2_resid >= r2_sh + FLOOR_MARGIN) and (r2_resid >= FLOOR_MARGIN)
    return {"nuisance_regressed": used, "resid_decode_r2": _f(r2_resid), "resid_shuffle_floor": _f(r2_sh),
            "survives_regress_out": bool(survives),
            "raw_valence_var": _f(float(np.var(vf))), "resid_var": _f(float(np.var(resid)))}


# ─────────────────────────────────────────────────────────────────────────────
# Expectation-violation micro-probe: reward-omission at an anticipated-reward (eat) step.
# ─────────────────────────────────────────────────────────────────────────────
def _omission_probe(world, sub, n_steps, seed, explore_eps=0.4):
    """At steps where the trained agent EATS (an anticipated reward), randomly either DELIVER the reward
    (sub.learn(r)) or OMIT it (sub.learn(0.0)) — the omission is done ONLY inside this probe by passing
    0.0 to learn(); the world file is untouched. The substrate is FROZEN, so learn() changes no weight;
    it only sets last_valence_rpe = reward - rhat, where rhat is the reward-head's prediction at act()
    time. A trained reward-head that ANTICIPATES the eat has rhat>0, so:
        delivery:  rpe = r - rhat   (>= 0 when the reward meets/exceeds the prediction)
        omission:  rpe = 0 - rhat   (NEGATIVE = worse than expected)
    This is the dopamine-RPE expectation-violation signature (Schultz 1997), read functionally off the
    substrate's own bookkeeping. Non-eat steps always deliver r (kept faithful)."""
    sub.freeze()
    sub.set_lesion_mask(None)
    rng = np.random.default_rng(seed + 321)
    a_prev = -1
    deliv_rpe, omit_rpe, deliv_rhat, omit_rhat = [], [], [], []
    n_eats = 0
    for _ in range(n_steps):
        d = world.drive_afferent()
        v1 = world.crop_v1feat()
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=explore_eps)
        rhat = sub._last_rhat                         # reward-head prediction for THIS step (pre-outcome)
        r, info = world.step(a)
        if info.get("ate") and r > 0.0:
            n_eats += 1
            if rng.random() < 0.5:
                sub.learn(0.0)                        # OMISSION: withhold the anticipated reward
                omit_rpe.append(float(sub.last_valence_rpe))
                omit_rhat.append(float(rhat) if rhat is not None else float("nan"))
            else:
                sub.learn(float(r))                   # DELIVERY
                deliv_rpe.append(float(sub.last_valence_rpe))
                deliv_rhat.append(float(rhat) if rhat is not None else float("nan"))
        else:
            sub.learn(float(r))
        a_prev = a
    mean_om = float(np.mean(omit_rpe)) if omit_rpe else float("nan")
    mean_de = float(np.mean(deliv_rpe)) if deliv_rpe else float("nan")
    # decisive iff we saw eats on BOTH arms and omission read-out is more negative than delivery
    om_lt_de = bool(np.isfinite(mean_om) and np.isfinite(mean_de) and (mean_om < mean_de))
    return {"n_eats": int(n_eats), "n_omission": len(omit_rpe), "n_delivery": len(deliv_rpe),
            "mean_rpe_omission": _f(mean_om), "mean_rpe_delivery": _f(mean_de),
            "mean_rhat_omission": _f(float(np.mean(omit_rhat)) if omit_rhat else float("nan")),
            "mean_rhat_delivery": _f(float(np.mean(deliv_rhat)) if deliv_rhat else float("nan")),
            "omission_below_delivery": om_lt_de,
            "omission_is_negative": bool(np.isfinite(mean_om) and mean_om < 0.0)}


# ─────────────────────────────────────────────────────────────────────────────
# per-seed battery
# ─────────────────────────────────────────────────────────────────────────────
def run_seed(seed, units="rate", encoder="learned_ema", n_hidden=128, n_latent=64,
             n_train=200_000, value_weight=1.0, n_probe=None, n_omit=None,
             grid_size=18, crop_radius=2, verbose=True):
    t0 = time.time()
    # auto-scale probe/omission budgets down for small (smoke) n_train
    if n_probe is None:
        n_probe = min(6000, max(800, n_train // 2))
    if n_omit is None:
        n_omit = min(4000, max(1200, n_train // 2))

    wcfg = WorldConfig(seed=seed, grid_size=grid_size, crop_radius=crop_radius)  # base foraging world (no nav)
    world = ForkPCSWorld(wcfg)
    scfg = PCSConfig(n_hidden=n_hidden, feat_dim=wcfg.n_v1, n_latent=n_latent, n_actions=N_ACTIONS,
                     n_drive=4, tbptt_T=18, units=units, encoder=encoder, seed=seed,
                     value_weight=value_weight)
    sub = PredictiveContinualSubstrate(scfg)

    # ---- 1. TRAIN online (curiosity policy + small exploration for early coverage) ----
    sub.unfreeze()
    sub.set_lesion_mask(None)
    a_prev = -1
    max_online_loss = 0.0
    for t in range(n_train):
        d = world.drive_afferent(); v1 = world.crop_v1feat()
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=0.2)
        r, _ = world.step(a)
        sub.learn(r)
        if sub.last_pred_loss is not None:
            max_online_loss = max(max_online_loss, float(sub.last_pred_loss))
        a_prev = a

    # ---- 2. FROZEN probe rollout (higher explore for grid + valence-range coverage) ----
    pr = _collect_probe(world, sub, n_probe, explore_eps=0.4)
    H, POS, RAW, RPE, ADV, VHAT = pr["H"], pr["POS"], pr["RAW"], pr["RPE"], pr["ADV"], pr["VHAT"]
    input_seq = pr["INPUT_SEQ"]

    # untrained-core reservoir replayed on the SAME input sequence (the grew-through-training floor)
    H_un = replay_untrained(wcfg, seed, units, encoder, wcfg.n_v1, n_latent, n_hidden, input_seq)

    # ---- 3. POPULATION PRESENCE — ridge-decode EACH valence signal from h_t vs 3 floors ----
    presence = {}
    for sig, lab in (("rpe", RPE), ("adv", ADV)):
        fin = np.isfinite(lab)
        if fin.sum() < 40:
            presence[sig] = {"note": f"too few finite {sig} labels ({int(fin.sum())})"}
            continue
        d = _r2_with_floors(H[fin], H_un[fin], RAW[fin], lab[fin].reshape(-1, 1), seed)
        d["beats_floors"] = bool(_beats_floors(d, VALENCE_ABS_BAR))
        presence[sig] = {k: (_f(v) if isinstance(v, (int, float)) else v) for k, v in d.items()}

    # ---- 4. PER-UNIT SOUNDNESS (cyclic-shift null) — trained core vs untrained reservoir ----
    go_lab = RPE if GO_SIGNAL == "rpe" else ADV
    sound_trained = _valence_selectivity_metrics(H, go_lab, seed)
    sound_untrained = _valence_selectivity_metrics(H_un, go_lab, seed)      # base/untrained control (step 4)
    tr_nsel = int(sound_trained.get("n_valence_selective_units", 0) or 0)
    un_nsel = int(sound_untrained.get("n_valence_selective_units", 0) or 0)
    # ATTRIBUTION: what fraction of the trained core's valence-selective population is NOT present in the
    # untrained reservoir (i.e. GREW through self-supervised training, per the base/untrained control) —
    # (trained - untrained)/trained, the anti-relabel guard against the skeptic's Trap-2. Requiring this
    # fraction >= (1 - 1/ratio) is equivalent to trained_nsel >= ratio * untrained_nsel, but forces the
    # subtraction to be made out loud rather than trusting a bare ratio.
    sound_frac = attributable_to(f"valence-soundness s{seed}", float(tr_nsel), float(un_nsel))
    soundness_pass = bool(sound_frac is not None and tr_nsel >= 1
                          and sound_frac >= (1.0 - 1.0 / STAB_TRAINED_OVER_UNTRAINED))

    # ---- 5. ANTI-COLLINEARITY: regress position + value out of the valence label, re-decode residual ----
    regress = _regress_out_and_decode(H, go_lab, POS, VHAT, seed)

    # ---- 6. EXPECTATION-VIOLATION micro-probe (reward omission at anticipated-reward steps) ----
    omission = _omission_probe(world, sub, n_omit, seed, explore_eps=0.4)

    # ---- pre-registered per-seed GO ----
    presence_go = bool(presence.get(GO_SIGNAL, {}).get("beats_floors", False))
    omission_go = bool(omission["omission_below_delivery"])
    valence_presence_go = bool(presence_go and soundness_pass and omission_go)

    result = {
        "seed": seed, "units": units, "encoder": encoder, "n_hidden": n_hidden, "n_latent": n_latent,
        "n_train": n_train, "n_probe": n_probe, "n_omit": n_omit, "value_weight": value_weight,
        "grid_size": grid_size, "crop_radius": crop_radius, "curiosity_beta": float(sub.cfg.curiosity_beta),
        "go_signal": GO_SIGNAL, "adv_note": "reconstructed per substrate learn() formula (frozen probe; "
                                            "sub.last_valence_adv is not written under freeze)",
        "train_max_online_loss": round(max_online_loss, 4),
        "presence": presence,
        "soundness_trained": sound_trained,
        "soundness_untrained": sound_untrained,
        "soundness": {"trained_n_selective": tr_nsel, "untrained_n_selective": un_nsel,
                      "trained_mean_stability": sound_trained.get("mean_stability"),
                      "untrained_mean_stability": sound_untrained.get("mean_stability"),
                      "trained_mean_stability_selective": sound_trained.get("mean_stability_selective"),
                      "untrained_mean_stability_selective": sound_untrained.get("mean_stability_selective"),
                      "ratio_required": STAB_TRAINED_OVER_UNTRAINED,
                      "attributable_fraction": _f(sound_frac), "pass": soundness_pass},
        "regress_out": regress,
        "omission": omission,
        "presence_go": presence_go, "soundness_go": soundness_pass, "omission_go": omission_go,
        "VALENCE_PRESENCE_GO": valence_presence_go,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if verbose:
        pg = presence.get(GO_SIGNAL, {})
        print(f"[seed {seed} units={units} h={n_hidden}] VALENCE_PRESENCE_GO={valence_presence_go} "
              f"({result['elapsed_s']}s)")
        print(f"    presence[{GO_SIGNAL}] R2={pg.get('r2')} floors(un/raw/sh)="
              f"{pg.get('floor_untrained')}/{pg.get('floor_rawv1')}/{pg.get('floor_shuffle')} "
              f"beats={pg.get('beats_floors')}")
        if "adv" in presence:
            pa = presence["adv"]
            print(f"    presence[adv] R2={pa.get('r2')} beats={pa.get('beats_floors')} "
                  f"(TAUTOLOGY-flagged: adv IS the policy-gradient multiplier — reported, not gated)")
        print(f"    soundness: trained_nsel={tr_nsel} untrained_nsel={un_nsel} "
              f"stab(tr/un)={sound_trained.get('mean_stability')}/{sound_untrained.get('mean_stability')} "
              f"stab_sel(tr/un)={sound_trained.get('mean_stability_selective')}/"
              f"{sound_untrained.get('mean_stability_selective')} pass={soundness_pass}")
        print(f"    regress-out({','.join(regress.get('nuisance_regressed', []))}): "
              f"resid_R2={regress.get('resid_decode_r2')} floor={regress.get('resid_shuffle_floor')} "
              f"survives={regress.get('survives_regress_out')}")
        print(f"    omission: n_om/n_de={omission['n_omission']}/{omission['n_delivery']} "
              f"rpe_om={omission['mean_rpe_omission']} rpe_de={omission['mean_rpe_delivery']} "
              f"rhat_de={omission['mean_rhat_delivery']} om<de={omission['omission_below_delivery']} "
              f"om_neg={omission['omission_is_negative']}")
    return result


def aggregate(per_seed):
    n = len(per_seed)
    def _cnt(key):
        return int(sum(1 for r in per_seed if r.get(key)))
    n_go = _cnt("VALENCE_PRESENCE_GO")
    required = int(math.ceil(SEEDS_REQUIRED_FRAC * n))
    def _meanf(path):
        vals = []
        for r in per_seed:
            v = r
            for p in path:
                v = v.get(p, {}) if isinstance(v, dict) else None
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return _f(float(np.mean(vals))) if vals else None
    return {
        "n_seeds": n, "seeds_required": required,
        "n_presence_go": _cnt("presence_go"), "n_soundness_go": _cnt("soundness_go"),
        "n_omission_go": _cnt("omission_go"), "n_VALENCE_PRESENCE_GO": n_go,
        "VALENCE_PRESENCE_GO_AGGREGATE": bool(n_go >= required),
        "mean_presence_rpe_r2": _meanf(["presence", "rpe", "r2"]),
        "mean_presence_adv_r2": _meanf(["presence", "adv", "r2"]),
        "mean_trained_stability": _meanf(["soundness", "trained_mean_stability"]),
        "mean_untrained_stability": _meanf(["soundness", "untrained_mean_stability"]),
        "mean_trained_n_selective": _meanf(["soundness", "trained_n_selective"]),
        "mean_untrained_n_selective": _meanf(["soundness", "untrained_n_selective"]),
        "mean_regress_out_resid_r2": _meanf(["regress_out", "resid_decode_r2"]),
        "n_survives_regress_out": int(sum(1 for r in per_seed
                                          if r.get("regress_out", {}).get("survives_regress_out"))),
        "mean_rpe_omission": _meanf(["omission", "mean_rpe_omission"]),
        "mean_rpe_delivery": _meanf(["omission", "mean_rpe_delivery"]),
    }


def main():
    ap = argparse.ArgumentParser(description="AGI-fork AFFECT C1 — valence-presence probe")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--units", choices=["rate", "spike"], default="rate")
    ap.add_argument("--encoder", choices=["learned_ema", "fixed"], default="learned_ema")
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--n-latent", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--value-weight", type=float, default=1.0,
                    help="value-head weight (default 1.0 so BOTH valence signals exist: adv needs value_weight>0)")
    ap.add_argument("--n-probe", type=int, default=None, help="frozen probe steps (auto-scaled from n_train if unset)")
    ap.add_argument("--n-omit", type=int, default=None, help="omission-probe steps (auto-scaled from n_train if unset)")
    ap.add_argument("--grid-size", type=int, default=18)
    ap.add_argument("--crop-radius", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    per_seed = [run_seed(s, units=args.units, encoder=args.encoder, n_hidden=args.n_hidden,
                         n_latent=args.n_latent, n_train=args.n_train, value_weight=args.value_weight,
                         n_probe=args.n_probe, n_omit=args.n_omit, grid_size=args.grid_size,
                         crop_radius=args.crop_radius)
                for s in args.seeds]
    agg = aggregate(per_seed)
    payload = {
        "battery": "fork_pcs_valence_presence", "stage": "C1 (presence + soundness + expectation-violation)",
        "units": args.units, "encoder": args.encoder, "n_hidden": args.n_hidden, "n_latent": args.n_latent,
        "n_train": args.n_train, "value_weight": args.value_weight, "n_probe": args.n_probe,
        "n_omit": args.n_omit, "grid_size": args.grid_size, "crop_radius": args.crop_radius,
        "seeds": args.seeds,
        "honesty_note": ("FUNCTIONAL read-outs only; no phenomenal claim. This C1 probe tests PRESENCE + "
                         "SOUNDNESS + EXPECTATION-VIOLATION of the reward-prediction-error read-out, "
                         "explicitly NOT the load-bearing faculty claim. adv presence is reported but NOT "
                         "gated (tautology: adv IS the policy-gradient multiplier)."),
        "pre_registered_gate": {
            "go_signal": GO_SIGNAL, "valence_abs_bar": VALENCE_ABS_BAR, "floor_margin": FLOOR_MARGIN,
            "cyclic_shift_draws": CYCLIC_SHIFT_DRAWS, "valence_si_n_bins": VALENCE_SI_N_BINS,
            "split_half_stability_thresh": PLACE_SI_STABILITY_THRESH,
            "stability_trained_over_untrained": STAB_TRAINED_OVER_UNTRAINED,
            "seeds_required_frac": SEEDS_REQUIRED_FRAC,
            "VALENCE_PRESENCE_GO": ("(population presence of the reward-PE signal beats all 3 floors) AND "
                                    "(per-unit trained #valence-selective units >= 2x untrained via the "
                                    "cyclic-shift null) AND (mean omission RPE < mean delivery RPE), on "
                                    ">= 5/6 seeds")},
        "per_seed": per_seed, "aggregate": agg}
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
