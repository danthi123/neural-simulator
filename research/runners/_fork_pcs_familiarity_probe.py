"""_fork_pcs_familiarity_probe — AGI-fork SELF-AWARENESS arc, stage C1: the FAMILIARITY-PRESENCE probe.

The presence test for the design's SECOND faculty (self-awareness / familiarity), mirroring the affect
arc's C1 valence-presence probe (`research/runners/_fork_pcs_valence_probe.py`) and the design doc
`docs/plans/2026-09-07-fork-affect-selfaware-arc-design.md` ("DESIGN — emergent SELF-AWARENESS/familiarity").
It asks a NARROW, pre-registered question of the substrate's now-exposed, byte-identical prediction-error
read-out (`sub.last_pred_error` = the realized one-step JEPA latent prediction error, set live in observe()):

  Does the substrate's own prediction error DISCRIMINATE FAMILIAR (in-distribution) from NOVEL
  (out-of-distribution) inputs — a familiarity AUROC well above 0.5 — and CRUCIALLY does the TRAINED
  substrate discriminate where the UNTRAINED reservoir does NOT (the base control that makes the signal
  genuinely trained-EMERGENT, not a trivial property of the dynamics)?

⭐ WHY THIS SHOULD EMERGE WHERE VALENCE DID NOT (the affect arc's characterized negative). A TRAINED
substrate predicts FAMILIAR next-views well = LOW JEPA prediction error, and NOVEL/OOD next-views poorly =
HIGH error. An UNTRAINED reservoir predicts EVERYTHING badly = no familiar/novel distinction. So a
prediction-error-as-familiarity read-out is grown-through-training by construction (the untrained control
sits at floor), unlike valence (which the C1 valence probe found the untrained reservoir already carries).

⚠️ WHAT THIS IS NOT (honesty boundary + scope). This is the PRESENCE test (like the affect C1), NOT the
load-bearing test. It does NOT build the W_phi recurrent-feedback loop or the confidence-gated temperature
that the design's self-awareness sections 2-3/6 describe — those are the follow-on that makes familiarity
CAUSAL on behavior. Every read-out here is a FUNCTIONAL instrument reading ("my familiarity monitor reads
this as novel, so I am uncertain"), NEVER a phenomenal claim ("it feels familiar").

REUSES (import, does not modify) the proven anti-hollow machinery of
`research.runners._fork_pcs_emergence_derisk` (`_r2_with_floors`, `_beats_floors`, `_host`, `_f`,
`FLOOR_MARGIN`, `PLACE_SI_STABILITY_THRESH`) and the affect probe's per-unit selectivity + CYCLIC-SHIFT
null (`_fork_pcs_valence_probe._valence_selectivity_metrics`), applied to the graded novelty (phi) signal.

Run:
  # CPU smoke (end-to-end, 1 seed, tiny — confirms JSON has all fields, no crash; does NOT decide emergence):
  SIM_BACKEND=numpy PYTHONPATH=. python -m research.runners._fork_pcs_familiarity_probe \
      --seeds 42 --n-hidden 64 --n-train 8000 --out /tmp/_fam_smoke.json
  # full 6-seed decisive run (GPU lane — queue it; 0 agent tokens):
  SIM_BACKEND=cupy python -m research.runners._fork_pcs_familiarity_probe \
      --seeds 42 43 44 100 101 102 --units rate --n-hidden 512 --n-train 200000 \
      --out research/findings/raw/_fork_pcs_familiarity_presence_6seed.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig
from research.runners.fork_pcs_world import WorldConfig, ForkPCSWorld, N_ACTIONS
# Reuse the emergence battery's proven, tested machinery verbatim (import, do not reinvent).
from research.runners._fork_pcs_emergence_derisk import (
    _r2_with_floors, _beats_floors, _host, _f, FLOOR_MARGIN, PLACE_SI_STABILITY_THRESH,
)
# Reuse the affect probe's per-unit selectivity (Skaggs-SI + split-half + CYCLIC-SHIFT null), applied here to
# the GRADED novelty signal (phi): the same anti-hollow per-unit soundness the valence probe used.
from research.runners._fork_pcs_valence_probe import _valence_selectivity_metrics
# ATTRIBUTION (not just measurement): the emergence claim is trained >> untrained, so we ask attributable_to
# whose the discrimination is rather than letting a bare ratio stand (the affect arc's discipline).
from tools.lab import attributable_to

# ── pre-registered bars (fixed BEFORE the multi-seed run) ────────────────────
FAMILIARITY_AUROC_BAR = 0.65   # trained prediction-error must discriminate familiar/novel WELL above chance
AUROC_ATTRIB_RATIO = 2.0       # trained effect (AUROC-0.5) must be >= this x the untrained reservoir's, i.e.
#                                attributable fraction >= 1 - 1/ratio = 0.5 (the grew-through-training bar,
#                                mirroring the valence probe's STAB_TRAINED_OVER_UNTRAINED=2.0)
CYCLIC_SHIFT_DRAWS = 100       # cyclic-shift null draws for the AUROC significance test
NOVEL_BLOCK_LEN = 20           # steps per alternating familiar/novel input block (lets h settle in each regime)
PHI_SI_N_BINS = 5              # quantile bins the graded-novelty (phi) label is discretized into for per-unit tuning
SEEDS_REQUIRED_FRAC = 5.0 / 6.0

# The NOVEL set (documented in _scramble_frame): a within-frame random permutation of the V1 feature vector.


def _average_rank_auroc(labels, scores):
    """Tie-aware AUROC of `scores` discriminating the POSITIVE class (labels==1, here NOVEL) from the NEGATIVE
    (labels==0, FAMILIAR), via the Mann-Whitney-U / average-rank identity. Higher `scores` (= higher
    prediction error) predicting NOVEL gives AUROC > 0.5. Non-finite scores/labels are dropped; nan if a
    class is empty. FUNCTIONAL discrimination read-out — asserts nothing about felt familiarity."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    fin = np.isfinite(scores) & np.isfinite(labels)
    labels, scores = labels[fin], scores[fin]
    pos = labels == 1.0
    neg = labels == 0.0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=np.float64)
    j = 0
    n = len(scores)
    while j < n:                                     # average ranks within tie groups (1-based)
        k = j
        while k + 1 < n and s_sorted[k + 1] == s_sorted[j]:
            k += 1
        ranks_sorted[j:k + 1] = (j + k) / 2.0 + 1.0
        j = k + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _auroc_cyclic_shift_null(novel, phi, seed, n_shift=CYCLIC_SHIFT_DRAWS):
    """Null distribution of the familiarity AUROC under a CYCLIC SHIFT of the novel-label series (np.roll by a
    random offset): preserves the label's marginal AND its block-autocorrelation, destroying only its
    alignment with which step was actually mispredicted. The valence probe's cyclic-shift discipline applied
    to the AUROC statistic — NOT an i.i.d. shuffle, which would break the block autocorrelation and inflate
    significance. Returns (p95, mean) over the null AUROCs."""
    novel = np.asarray(novel, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    T = len(novel)
    rng = np.random.default_rng(seed + 9137)
    null = np.empty(n_shift, dtype=np.float64)
    for s in range(n_shift):
        off = int(rng.integers(1, T))                # non-zero offset
        null[s] = _average_rank_auroc(np.roll(novel, off), phi)
    return float(np.nanpercentile(null, 95)), float(np.nanmean(null))


def _scramble_frame(v1_host, rng):
    """The NOVEL / out-of-distribution manipulation: a WITHIN-FRAME random permutation of the V1 feature
    vector. Rationale (documented so the choice is principled, not arbitrary):
      * It is GENUINELY off-distribution: a real egocentric crop, passed through the fixed Gabor V1 front end,
        produces a structured pattern (which orientation/frequency fires at which retinal position). A random
        permutation of those feature entries is a pattern no real crop can produce — off the real-crop
        manifold — so a substrate that has learned real visual dynamics cannot predict its next latent.
      * It is MARGINAL-MATCHED: the permutation preserves the frame's exact activation MULTISET (identical
        L1/L2 norm, identical sparsity/histogram). So the discrimination CANNOT be a trivial magnitude/energy
        read, and — decisively — the UNTRAINED reservoir sees statistically identical inputs in both
        conditions, so any familiar/novel separation it shows is pure chance. That is the base control that
        makes a trained substrate's separation attributable to LEARNED structure.
    Alternatives considered: a held-out object TYPE (invasive world surgery; objects are rare in the crop) and
    pure Gaussian NOISE frames (different marginals -> not magnitude-matched). Within-frame permutation is the
    cleanest marginal-matched OOD set for a PRESENCE test."""
    perm = rng.permutation(len(v1_host))
    return v1_host[perm].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FROZEN familiarity probe — alternating familiar / scrambled-novel input blocks; collect phi, h_t, labels.
# ─────────────────────────────────────────────────────────────────────────────
def _collect_familiarity_probe(world, sub, n_steps, explore_eps, block_len, seed):
    """Drive the FROZEN substrate against the world, presenting alternating BLOCKS of FAMILIAR (the real
    world crop) and NOVEL (within-frame-scrambled) V1 input, and record per step:
      PHI    sub.last_pred_error  — the realized one-step JEPA latent prediction error, set inside observe()
             (byte-identical read-out) BEFORE act(); nan on the first-ever step (no prior prediction).
      NOVEL  1.0 if the fed frame was scrambled this step, else 0.0 (the discrimination label).
      H      h_t (for the supporting h_t->familiarity decode + per-unit soundness).
      RAW_FED the fed V1 (raw-V1 floor feature for the decode).
    Blocks (not per-step Bernoulli) so h settles into each regime; block boundaries carry the previous
    regime's state into the first frame of the next (a CONSERVATIVE contamination that only weakens the
    trained signal). INPUT_SEQ records the EXACT fed inputs so the untrained reservoir replays them identically.
    The world advances normally throughout (garbage actions during novel blocks just wander the agent)."""
    sub.freeze()
    sub.set_lesion_mask(None)
    rng = np.random.default_rng(seed + 4242)
    H, PHI, NOVEL, RAW_FED, POS, INSEQ = [], [], [], [], [], []
    a_prev = -1
    for t in range(n_steps):
        d = world.drive_afferent()
        v1_real = world.crop_v1feat()
        v1_real_host = np.asarray(v1_real.get() if hasattr(v1_real, "get") else v1_real, dtype=np.float32)
        is_novel = ((t // block_len) % 2) == 1        # block 0 familiar, block 1 novel, ... (50/50)
        v1_fed_host = _scramble_frame(v1_real_host, rng) if is_novel else v1_real_host
        d_host = np.asarray(d, dtype=np.float32)
        pos_before = world.agent
        h = sub.observe(v1_fed_host, a_prev, d_host)
        phi = sub.last_pred_error                     # set inside observe() for this transition
        a = sub.act(h, explore_eps=explore_eps)
        r, _info = world.step(a)
        sub.learn(r)                                  # frozen: changes no weight; keeps loop ordering identical
        H.append(_host(h).astype(np.float32))
        PHI.append(float(phi) if phi is not None else float("nan"))
        NOVEL.append(1.0 if is_novel else 0.0)
        RAW_FED.append(v1_fed_host)
        POS.append(np.asarray(pos_before, dtype=np.float32))
        INSEQ.append((v1_fed_host, a_prev, d_host))
        a_prev = a
    return {"H": np.asarray(H), "PHI": np.asarray(PHI, np.float64), "NOVEL": np.asarray(NOVEL, np.float64),
            "RAW_FED": np.asarray(RAW_FED), "POS": np.asarray(POS), "INPUT_SEQ": INSEQ}


def _replay_untrained_with_phi(seed, units, encoder, feat_dim, n_latent, n_hidden, input_seq):
    """Replay the EXACT fed input sequence (familiar + scrambled-novel, same order) through a FRESH UNTRAINED
    core, capturing its OWN per-step h_t AND prediction error. THE base control that makes the signal
    trained-EMERGENT: a random reservoir predicts everything badly, so its phi should NOT discriminate
    familiar from novel (AUROC ~ 0.5). Uses seed+777, matching the emergence battery's replay_untrained."""
    sub = PredictiveContinualSubstrate(PCSConfig(
        n_hidden=n_hidden, feat_dim=feat_dim, n_latent=n_latent, n_actions=N_ACTIONS,
        n_drive=4, units=units, encoder=encoder, seed=seed + 777))
    sub.freeze()
    H, PHI = [], []
    for (v1, ap, d) in input_seq:
        h = sub.observe(v1, ap, d)
        H.append(np.asarray(h.get() if hasattr(h, "get") else h, dtype=np.float32))
        PHI.append(float(sub.last_pred_error) if sub.last_pred_error is not None else float("nan"))
    return np.asarray(H), np.asarray(PHI, np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# per-seed battery
# ─────────────────────────────────────────────────────────────────────────────
def run_seed(seed, units="rate", encoder="learned_ema", n_hidden=128, n_latent=64,
             n_train=200_000, value_weight=0.0, n_probe=None, block_len=NOVEL_BLOCK_LEN,
             grid_size=18, crop_radius=2, verbose=True):
    t0 = time.time()
    if n_probe is None:
        n_probe = min(6000, max(1200, n_train // 2))

    wcfg = WorldConfig(seed=seed, grid_size=grid_size, crop_radius=crop_radius)  # base foraging world (no nav)
    world = ForkPCSWorld(wcfg)
    scfg = PCSConfig(n_hidden=n_hidden, feat_dim=wcfg.n_v1, n_latent=n_latent, n_actions=N_ACTIONS,
                     n_drive=4, tbptt_T=18, units=units, encoder=encoder, seed=seed, value_weight=value_weight)
    sub = PredictiveContinualSubstrate(scfg)

    # ---- 1. TRAIN online (curiosity policy + small exploration for early coverage) ----
    sub.unfreeze()
    sub.set_lesion_mask(None)
    a_prev = -1
    max_online_loss = 0.0
    for _t in range(n_train):
        d = world.drive_afferent(); v1 = world.crop_v1feat()
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=0.2)
        r, _ = world.step(a)
        sub.learn(r)
        if sub.last_pred_loss is not None:
            max_online_loss = max(max_online_loss, float(sub.last_pred_loss))
        a_prev = a

    # ---- 2. FROZEN familiarity probe (alternating familiar / scrambled-novel blocks) ----
    pr = _collect_familiarity_probe(world, sub, n_probe, explore_eps=0.4, block_len=block_len, seed=seed)
    H, PHI, NOVEL, RAW = pr["H"], pr["PHI"], pr["NOVEL"], pr["RAW_FED"]
    input_seq = pr["INPUT_SEQ"]

    # untrained-core reservoir replayed on the SAME fed inputs (the grew-through-training floor)
    H_un, PHI_un = _replay_untrained_with_phi(seed, units, encoder, wcfg.n_v1, n_latent, n_hidden, input_seq)

    # ---- 3. PRIMARY — prediction-error familiarity AUROC: trained vs untrained + cyclic-shift null ----
    auroc_tr = _average_rank_auroc(NOVEL, PHI)
    auroc_un = _average_rank_auroc(NOVEL, PHI_un)
    p95, null_mean = _auroc_cyclic_shift_null(NOVEL, PHI, seed)
    tr_eff = (auroc_tr - 0.5) if np.isfinite(auroc_tr) else float("nan")
    un_eff = (auroc_un - 0.5) if np.isfinite(auroc_un) else float("nan")
    # attributable: what fraction of the trained discrimination is NOT present in the untrained reservoir
    attr = attributable_to(f"familiarity-AUROC s{seed}", tr_eff, un_eff)
    attributable_pass = bool(attr is not None and np.isfinite(tr_eff) and tr_eff > 0.0
                             and attr >= (1.0 - 1.0 / AUROC_ATTRIB_RATIO))
    auroc_bar_pass = bool(np.isfinite(auroc_tr) and auroc_tr >= FAMILIARITY_AUROC_BAR)
    cyclic_pass = bool(np.isfinite(auroc_tr) and np.isfinite(p95) and auroc_tr > p95)
    FAMILIARITY_GO = bool(auroc_bar_pass and cyclic_pass and attributable_pass)

    # raw phi separation (diagnostic / wiring visibility)
    fam_m = (NOVEL == 0.0) & np.isfinite(PHI)
    nov_m = (NOVEL == 1.0) & np.isfinite(PHI)
    fam_m_un = (NOVEL == 0.0) & np.isfinite(PHI_un)
    nov_m_un = (NOVEL == 1.0) & np.isfinite(PHI_un)
    phi_sep = {
        "trained_phi_familiar_mean": _f(float(np.mean(PHI[fam_m])) if fam_m.any() else float("nan")),
        "trained_phi_novel_mean": _f(float(np.mean(PHI[nov_m])) if nov_m.any() else float("nan")),
        "untrained_phi_familiar_mean": _f(float(np.mean(PHI_un[fam_m_un])) if fam_m_un.any() else float("nan")),
        "untrained_phi_novel_mean": _f(float(np.mean(PHI_un[nov_m_un])) if nov_m_un.any() else float("nan")),
    }

    # ---- 4. SUPPORTING — h_t -> familiarity-label decode (held-out R^2 vs 3 floors) ----
    # NOTE: because NOVELTY is an INPUT-LEVEL manipulation, the raw-V1 floor can be strong (the scramble is
    # in the input); so this decode beating raw-V1 is a STRICT bonus ("h_t carries familiarity beyond the raw
    # input"), NOT the emergence claim. The decisive emergence test is the phi-AUROC-vs-untrained above.
    decode = _r2_with_floors(H, H_un, RAW, NOVEL.reshape(-1, 1), seed)
    decode["beats_floors"] = bool(_beats_floors(decode, FLOOR_MARGIN))
    decode = {k: (_f(v) if isinstance(v, (int, float)) else v) for k, v in decode.items()}

    # ---- 5. SUPPORTING — per-unit soundness on GRADED novelty (phi): trained vs untrained ----
    # h_t units tuned to the graded prediction-error magnitude, split-half stable AND significant vs the
    # CYCLIC-SHIFT null (reused verbatim from the valence probe). Continuous phi is the tuning label (a binary
    # label collapses the quantile bins). Attributable trained-over-untrained = the grew-through-training test.
    fin_tr = np.isfinite(PHI)
    fin_un = np.isfinite(PHI_un)
    sound_tr = _valence_selectivity_metrics(H[fin_tr], PHI[fin_tr], seed, n_bins=PHI_SI_N_BINS)
    sound_un = _valence_selectivity_metrics(H_un[fin_un], PHI_un[fin_un], seed, n_bins=PHI_SI_N_BINS)
    tr_nsel = int(sound_tr.get("n_valence_selective_units", 0) or 0)
    un_nsel = int(sound_un.get("n_valence_selective_units", 0) or 0)
    sound_frac = attributable_to(f"familiarity-soundness s{seed}", float(tr_nsel), float(un_nsel))
    soundness_pass = bool(sound_frac is not None and tr_nsel >= 1
                          and sound_frac >= (1.0 - 1.0 / AUROC_ATTRIB_RATIO))

    result = {
        "seed": seed, "units": units, "encoder": encoder, "n_hidden": n_hidden, "n_latent": n_latent,
        "n_train": n_train, "n_probe": n_probe, "block_len": block_len, "value_weight": value_weight,
        "grid_size": grid_size, "crop_radius": crop_radius, "novel_set": "within_frame_v1_permutation",
        "train_max_online_loss": round(max_online_loss, 4),
        "n_probe_familiar": int(fam_m.sum()), "n_probe_novel": int(nov_m.sum()),
        # PRIMARY
        "familiarity_auroc_trained": _f(auroc_tr), "familiarity_auroc_untrained": _f(auroc_un),
        "auroc_cyclic_shift_p95": _f(p95), "auroc_cyclic_shift_mean": _f(null_mean),
        "auroc_attributable_fraction": _f(attr),
        "auroc_bar": FAMILIARITY_AUROC_BAR, "auroc_bar_pass": auroc_bar_pass,
        "cyclic_shift_pass": cyclic_pass, "attributable_pass": attributable_pass,
        "phi_separation": phi_sep,
        "FAMILIARITY_GO": FAMILIARITY_GO,
        # SUPPORTING
        "decode": decode,
        "soundness_trained": sound_tr, "soundness_untrained": sound_un,
        "soundness": {"trained_n_selective": tr_nsel, "untrained_n_selective": un_nsel,
                      "trained_mean_stability": sound_tr.get("mean_stability"),
                      "untrained_mean_stability": sound_un.get("mean_stability"),
                      "ratio_required": AUROC_ATTRIB_RATIO,
                      "attributable_fraction": _f(sound_frac), "pass": soundness_pass},
        "elapsed_s": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"[seed {seed} units={units} h={n_hidden}] FAMILIARITY_GO={FAMILIARITY_GO} "
              f"({result['elapsed_s']}s)")
        print(f"    AUROC(phi->novel) trained={_f(auroc_tr)} untrained={_f(auroc_un)} "
              f"bar>={FAMILIARITY_AUROC_BAR}:{auroc_bar_pass} cyc_p95={_f(p95)}:{cyclic_pass} "
              f"attr={_f(attr)}:{attributable_pass}")
        print(f"    phi familiar/novel  trained={phi_sep['trained_phi_familiar_mean']}/"
              f"{phi_sep['trained_phi_novel_mean']}  untrained={phi_sep['untrained_phi_familiar_mean']}/"
              f"{phi_sep['untrained_phi_novel_mean']}")
        print(f"    [supporting] h_t->novel decode R2={decode.get('r2')} "
              f"floors(un/raw/sh)={decode.get('floor_untrained')}/{decode.get('floor_rawv1')}/"
              f"{decode.get('floor_shuffle')} beats={decode.get('beats_floors')}  "
              f"| per-unit phi-tuning nsel tr/un={tr_nsel}/{un_nsel} attr={_f(sound_frac)} pass={soundness_pass}")
    return result


def aggregate(per_seed):
    n = len(per_seed)

    def _cnt(key):
        return int(sum(1 for r in per_seed if r.get(key)))

    def _meanf(key):
        vals = [float(r[key]) for r in per_seed
                if isinstance(r.get(key), (int, float)) and np.isfinite(r.get(key))]
        return _f(float(np.mean(vals))) if vals else None

    n_go = _cnt("FAMILIARITY_GO")
    required = int(math.ceil(SEEDS_REQUIRED_FRAC * n))
    return {
        "n_seeds": n, "seeds_required": required,
        "n_auroc_bar_pass": _cnt("auroc_bar_pass"), "n_cyclic_shift_pass": _cnt("cyclic_shift_pass"),
        "n_attributable_pass": _cnt("attributable_pass"), "n_FAMILIARITY_GO": n_go,
        "FAMILIARITY_GO_AGGREGATE": bool(n_go >= required),
        "mean_familiarity_auroc_trained": _meanf("familiarity_auroc_trained"),
        "mean_familiarity_auroc_untrained": _meanf("familiarity_auroc_untrained"),
        "mean_auroc_cyclic_shift_p95": _meanf("auroc_cyclic_shift_p95"),
        "mean_auroc_attributable_fraction": _meanf("auroc_attributable_fraction"),
        "n_decode_beats_floors": int(sum(1 for r in per_seed if r.get("decode", {}).get("beats_floors"))),
        "n_soundness_pass": int(sum(1 for r in per_seed if r.get("soundness", {}).get("pass"))),
    }


def main():
    ap = argparse.ArgumentParser(description="AGI-fork SELF-AWARENESS C1 — familiarity-presence probe")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--units", choices=["rate", "spike"], default="rate")
    ap.add_argument("--encoder", choices=["learned_ema", "fixed"], default="learned_ema")
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--n-latent", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--value-weight", type=float, default=0.0,
                    help="value-head weight (default 0.0=OFF; the familiarity signal needs only the JEPA head)")
    ap.add_argument("--n-probe", type=int, default=None, help="frozen probe steps (auto-scaled from n_train if unset)")
    ap.add_argument("--block-len", type=int, default=NOVEL_BLOCK_LEN, help="steps per familiar/novel input block")
    ap.add_argument("--grid-size", type=int, default=18)
    ap.add_argument("--crop-radius", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    per_seed = [run_seed(s, units=args.units, encoder=args.encoder, n_hidden=args.n_hidden,
                         n_latent=args.n_latent, n_train=args.n_train, value_weight=args.value_weight,
                         n_probe=args.n_probe, block_len=args.block_len, grid_size=args.grid_size,
                         crop_radius=args.crop_radius)
                for s in args.seeds]
    agg = aggregate(per_seed)
    payload = {
        "battery": "fork_pcs_familiarity_presence",
        "stage": "C1 (presence + soundness: does prediction-error discriminate familiar/novel, trained>>untrained)",
        "units": args.units, "encoder": args.encoder, "n_hidden": args.n_hidden, "n_latent": args.n_latent,
        "n_train": args.n_train, "value_weight": args.value_weight, "n_probe": args.n_probe,
        "block_len": args.block_len, "grid_size": args.grid_size, "crop_radius": args.crop_radius,
        "seeds": args.seeds, "novel_set": "within_frame_v1_permutation",
        "honesty_note": ("FUNCTIONAL read-outs only; no phenomenal claim. This C1 probe tests PRESENCE of a "
                         "trained-EMERGENT familiarity signal (prediction-error discriminates familiar/novel "
                         "where the untrained reservoir does not), NOT the load-bearing faculty claim (the "
                         "W_phi feedback + confidence-gated behavior is the follow-on)."),
        "pre_registered_gate": {
            "familiarity_auroc_bar": FAMILIARITY_AUROC_BAR, "auroc_attrib_ratio": AUROC_ATTRIB_RATIO,
            "cyclic_shift_draws": CYCLIC_SHIFT_DRAWS, "novel_block_len": NOVEL_BLOCK_LEN,
            "seeds_required_frac": SEEDS_REQUIRED_FRAC,
            "FAMILIARITY_GO": ("(trained prediction-error AUROC for novel-vs-familiar >= 0.65) AND (> its "
                               "cyclic-shift 95th pctile) AND (trained effect >= 2x the untrained reservoir's "
                               "== attributable fraction >= 0.5), on >= 5/6 seeds")},
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
