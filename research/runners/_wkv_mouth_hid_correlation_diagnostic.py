"""PHASE 0 -- cheap diagnostic gate for the mouth read-SNR HID-DECORRELATION lever (#80 continued, 2026-08-27).

READ FIRST (do not re-derive): `research/findings/2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`
-- the word-pool ENSEMBLE read lever (`--sub-pop`) is INERT BY CONSTRUCTION: the P word-pool "clones" for one word
are DETERMINISTIC CONDUCTANCE REPLICAS of ONE shared noisy hidden population (same presynaptic wiring, no per-
member noise since the pools never spike and OU noise enters as CURRENT not conductance) -> summing P of them
cancels the pooling gain exactly, zero SNR averaging. The adversarially-ranked next lever (per the design panel
this file answers to) is RECURRENT-INHIBITION ACTIVE DECORRELATION of the HIDDEN population itself (a fast E->I->E
circuit sited on hid/hidinh, `--hidfb`, NOT yet built by this file) -- but that lever only has something to do if
the hidden population's noise is genuinely CORRELATED across the neurons a downstream read pools over. This file
answers that ONE question cheaply, before any hidfb mechanism is written.

Also read `research/findings/2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research.md` (a different
subsystem -- the cortex whitening arc -- but the GENERAL result transfers): off-diagonal cross-neuron decorrelation
mathematically requires a genuine cross-neuron recurrent circuit (no per-feature-local mechanism can do it), the
recommended biological form is a Dale's-law INHIBITORY-INTERNEURON population (King, Zylberberg, DeWeese, J.
Neurosci. 2013) -- exactly the E->I->E shape `--hidfb` would take -- and FULL whitening OVER-WHITENS / collapses
(the project's own full-ZCA run: +0.307 -> -0.012), so any eventual Phase-1 mechanism must stay LOW-RANK /
regularized, never a full decorrelator. That constrains Phase 1's design; it does not change this file's question.

THE QUESTION THIS FILE ANSWERS: is the hidden population's (hid+hidinh) TRIAL-TO-TRIAL noise correlated across the
neurons that feed one word's read (the "clones" the ensemble finding named), or is it already close to
independent? If already independent (rho <= ~0.05-0.10, "like BDSP's 0.03"), a recurrent-inhibition decorrelation
circuit has NOTHING TO DECORRELATE and the whole hidfb lever is MOOT -- the read-SNR gap is dominated by something
else (divisive normalization, a predictive prior, or a still-different mechanism), and that is the honest,
valuable, redirecting negative to file. If correlated (rho >= ~0.15), Phase 1 (the hidfb build) is warranted.

METHOD (a MEASUREMENT, not a build -- additive, NO sim/ edit, reuse-by-import of BatchedSubstrateReadout's EXISTING
`_build_bridge`/`_wire` from `_wkv_mouth_readout_eprop_batched_substrate_derisk.py`; this file adds ZERO new
bridge/wiring code):
  1. Drive ALL B block-diagonal copies (normally B independent DATA POSITIONS) with the IDENTICAL fixed feature
     vector -- repurposing the B copies as B independent trial-repeats of one input. Their OU noise streams are
     per-neuron-independent BY DEFAULT (`sim/bridge.py::_draw_ou_noise_samples`: one global `cp.random.randn(n)`
     draw per step over ALL n neurons in the pool, giving every neuron -- including the same-role neuron in a
     different block -- its own iid N(0,1) increment; no per-region/per-neuron shared-stream opt-in is enabled
     here), so block-to-block IS a valid independent-trial axis, not just a coincidence of layout.
  2. Run `--n-repeats` further repeated read_window passes on top (fresh accumulated OU evolution each pass --
     `_reset()` clears membrane/recovery/firing/conductance state but NOT the persistent OU AR(1) current; this
     matches EXACTLY how the production forward's repeated `batch_margin` calls already behave across gradient
     steps during training, not a diagnostic-only quirk). Total trial count = n_repeats * B per neuron.
  3. For each of the F feature groups, gather the 2*Hp "clones" (Hp hid + Hp hidinh neurons sharing that feature's
     drive) and compute: (a) mean pairwise Pearson correlation rho across trials; (b) pooling gain
     CV(single)/CV(sum-of-clones) vs the ideal sqrt(2*Hp); (c) Var(sum)/[2*Hp*Var(single)] (1.0 = independent,
     2*Hp = full common-mode).

WHAT IS ACTUALLY MEASURED (the "integrated conductance" proxy, and why it is the right one): hid/hidinh neurons
receive ONLY the external drive current (`internal_density=0.0`, zero incoming synapses in this bridge) -- so
their OWN `cp_conductance_g_e`/`g_i` is identically 0 always; there is no such thing as "the conductance ON a hid
neuron" to read. What matters for a downstream reader (wpool, or a future hidfb interneuron) is the conductance a
hid/hidinh neuron INDUCES in whatever it synapses onto, which for a linear conductance-based synapse (an
EPSC/IPSC kernel triggered per presynaptic spike) is proportional to that neuron's own integrated SPIKE COUNT over
the read window. So this file reads each hid/hidinh neuron's spike count (`cp_firing_states` summed post-settle)
as the "per-trial integrated conductance it contributes downstream" -- the standard proxy for exactly this
shared-vs-independent-drive question, and the only signal these neurons actually carry.

VOCAB TRUNCATION (`--v-diag`, a legitimate COST-ONLY simplification, NOT an operating-point change): the reused
`_wire()` wires ALL Hn=F*Hp hid/hidinh neurons DENSELY onto V*P word-pool neurons -- at the real V=1000 that is
~2M edges PER BLOCK, expensive for a numpy/CPU diagnostic that never reads wpool. wpool has NO feedback path to
hid/hidinh (internal_density=0, no recurrent edges anywhere in this bridge) -- its SIZE cannot affect the hidden-
population correlation being measured, only the (here-irrelevant) wiring cost. This file therefore builds a VOCAB-
TRUNCATED view of a real checkpoint (same D, same REAL head_w/head_b rows for the words kept, V shrunk) fed to the
unmodified BatchedSubstrateReadout class. ou_std / hid_gain / sub_read_window / settle_frac / uniform_thresh /
sub_hid_pop are all held at the real production operating point (they are, in fact, already the class's defaults).

PHASE-0 GATE (owner-specified): rho_baseline <= ~0.05-0.10 -> HONEST NEGATIVE, STOP (file the negative, do not
build Phase 1 / the hidfb lever). rho_baseline >= ~0.15 -> PROCEED to Phase 1.

Run in the FOREGROUND (this is a cheap diagnostic; do not background it):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_hid_correlation_diagnostic --seed 42
If numpy is impractically slow, fall back to a short cupy run (does NOT contend with the in-flight dendritic
GPU run in any load-bearing way -- this is seconds-to-minutes, not a training job):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_hid_correlation_diagnostic --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from sim.backend import to_host, get_backend  # noqa: E402
from tools.lab import void_if  # noqa: E402

from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import BatchedSubstrateReadout  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _native  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _host_feat  # noqa: E402


# ======================================================================================================================
# A vocab-truncated VIEW of a real checkpoint: keeps D (feature dim) and REAL head_w/head_b rows for the first
# `v_diag` words, so hid/hidinh's OWN dynamics (driven purely by external current -- never by wpool, see module
# docstring) are UNCHANGED from the real operating point; only V shrinks, cutting the (irrelevant-to-this-
# diagnostic) hid->wpool wiring cost from O(V*Hn) to O(v_diag*Hn). unk_idx is dropped (-1) since truncation may
# have cut the real <unk> row; this only affects wpool's tonic-bias wiring (never read here).
# ======================================================================================================================
class _VocabTruncatedRO:
    def __init__(self, ro, v_diag):
        self.V = int(v_diag)
        self.D = int(ro.D)
        self.head_w = np.asarray(ro.head_w[:v_diag], dtype=np.float64).copy()
        self.head_b = np.asarray(ro.head_b[:v_diag], dtype=np.float64).copy()
        self.unk_idx = -1


def _fixed_feature(ro, seed, n_tokens=6):
    """A single REAL, representative signed feature vector [D] -- the state after a short random-token walk
    through the real checkpoint's own dynamics (`ro.advance`), read via the SAME `_host_feat` the production
    forward uses. No corpus load needed: any token id in [0, ro.V) is a valid walk step, and this diagnostic only
    needs a realistically-SCALED drive (partial hid activity, not all-silent/all-saturated), not real text content."""
    rng = np.random.default_rng(seed * 3301 + 17)
    ap = np.zeros(ro.D); an = np.zeros(ro.D)
    tid = int(rng.integers(0, ro.V))
    for _ in range(n_tokens):
        ap, an = ro.advance(ap, an, tid)
        tid = int(rng.integers(0, ro.V))
    return _host_feat(ro, ap, an, tid)


def _corr_stats(clones):
    """clones: [n_trials, K] per-trial integrated spike count for the K 'clone' neurons of one feature group.
    Returns mean pairwise rho, the pooling-gain readout, and the common-mode variance ratio (see module docstring
    for the definitions and the 1.0-vs-K reference points)."""
    K = clones.shape[1]
    std = clones.std(axis=0)
    valid = std > 1e-9
    out = {"n_valid_clones": int(valid.sum()), "k": int(K)}
    if valid.sum() < 2:
        out.update(rho=None, cv_single=None, cv_sum=None, gain_ideal=round(float(np.sqrt(K)), 4),
                   gain_actual=None, var_ratio=None)
        return out
    C = clones[:, valid]
    corr = np.corrcoef(C, rowvar=False)
    iu = np.triu_indices(corr.shape[0], k=1)
    rho = float(np.nanmean(corr[iu])) if iu[0].size else None
    per_mean = clones.mean(axis=0)
    per_std = clones.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_each = np.where(np.abs(per_mean) > 1e-9, per_std / np.abs(per_mean), np.nan)
    cv_single = float(np.nanmean(cv_each)) if np.any(~np.isnan(cv_each)) else None
    s = clones.sum(axis=1)
    s_mean = float(s.mean())
    cv_sum = float(s.std() / abs(s_mean)) if abs(s_mean) > 1e-9 else None
    gain_ideal = float(np.sqrt(K))
    gain_actual = (float(cv_single / cv_sum) if (cv_single is not None and cv_sum and cv_sum > 1e-12) else None)
    mean_var_single = float(np.mean(per_std ** 2))
    var_ratio = (float(s.var() / (K * mean_var_single)) if mean_var_single > 1e-12 else None)
    out.update(rho=(round(rho, 4) if rho is not None else None),
                cv_single=(round(cv_single, 4) if cv_single is not None else None),
                cv_sum=(round(cv_sum, 4) if cv_sum is not None else None),
                gain_ideal=round(gain_ideal, 4),
                gain_actual=(round(gain_actual, 4) if gain_actual is not None else None),
                var_ratio=(round(var_ratio, 4) if var_ratio is not None else None))
    return out


def measure_hid_correlation(s_batch, feat, n_repeats, verbose=True):
    """Drive all B blocks with the IDENTICAL feat; run n_repeats independent read_window passes; return per-
    feature-group correlation stats over the B*n_repeats trials (see module docstring, METHOD)."""
    B = s_batch.B
    F, Hp = s_batch.F, s_batch.Hp
    hid_dim = s_batch.hid_dim                                            # [Hn] feature index per hid neuron (per block)
    b = s_batch._b
    xp, _ = get_backend()
    feats_signed = np.tile(np.asarray(feat, dtype=np.float64)[None, :], (B, 1))          # [B, D] identical rows
    featF = np.concatenate([np.maximum(feats_signed, 0.0), np.maximum(-feats_signed, 0.0)], axis=1)  # [B, F]
    settle = int(s_batch.read_window * s_batch.settle_frac)

    Hn = F * Hp
    hid_counts = np.zeros((n_repeats, B, Hn), dtype=np.float64)
    hidinh_counts = np.zeros((n_repeats, B, Hn), dtype=np.float64)
    for rep in range(n_repeats):
        t0 = time.time()
        s_batch._reset()
        drive = xp.zeros(b.core_config.num_neurons, dtype=xp.float64)
        fdrive = xp.asarray(s_batch.hid_bias + s_batch.hid_gain * featF[:, hid_dim]).reshape(-1)
        drive[s_batch._hid_flat] = fdrive
        drive[s_batch._hidinh_flat] = fdrive
        b.cp_external_input_current[:] = drive.astype(b.cp_external_input_current.dtype)
        acc_hid = xp.zeros(B * Hn, dtype=xp.float64)
        acc_hidinh = xp.zeros(B * Hn, dtype=xp.float64)
        for step in range(s_batch.read_window):
            b._run_one_simulation_step()
            if step < settle:
                continue
            fs = b.cp_firing_states
            acc_hid = acc_hid + fs[s_batch._hid_flat].astype(xp.float64)
            acc_hidinh = acc_hidinh + fs[s_batch._hidinh_flat].astype(xp.float64)
        b.cp_external_input_current[:] = 0.0
        hid_counts[rep] = np.asarray(to_host(acc_hid)).reshape(B, Hn)
        hidinh_counts[rep] = np.asarray(to_host(acc_hidinh)).reshape(B, Hn)
        if verbose:
            print(f"  [repeat {rep + 1}/{n_repeats}] {time.time() - t0:.1f}s "
                  f"mean_hid_spikes/neuron={hid_counts[rep].mean():.3f}", flush=True)

    per_feature = {}
    for f in range(F):
        cols = np.where(hid_dim == f)[0]                                 # Hp indices within this feature's slice
        hid_f = hid_counts[:, :, cols].reshape(-1, Hp)                    # [n_repeats*B, Hp]
        hidinh_f = hidinh_counts[:, :, cols].reshape(-1, Hp)
        clones = np.concatenate([hid_f, hidinh_f], axis=1)                # [trials, 2*Hp]
        per_feature[f] = _corr_stats(clones)
    return per_feature, hid_counts, hidinh_counts


def _thr_hash(seed, ro, B, hid_pop, ou_std, read_window, hid_gain, ratio, n_bias, bias_drive_pA, settle_frac):
    s = BatchedSubstrateReadout(ro, seed, B, hid_pop=hid_pop, pop=1, ou_std=ou_std, read_window=read_window,
                                hid_gain=hid_gain, ratio=ratio, n_bias=n_bias, bias_drive_pA=bias_drive_pA,
                                settle_frac=settle_frac)
    thr = np.asarray(to_host(s._b.cp_neuron_firing_thresholds)).astype(np.float64)
    del s
    return hashlib.sha1(thr.tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=8, help="block-diagonal copies repurposed as trial-repeats")
    ap.add_argument("--n-repeats", type=int, default=25, help="further repeated read_window passes per block")
    ap.add_argument("--v-diag", type=int, default=24, help="vocab truncation for the diagnostic (cost-only, see docstring)")
    ap.add_argument("--feat-scale", type=float, default=1.0,
                    help="multiply the fixed feature vector (robustness check: does a stronger/near-saturating "
                         "drive reintroduce correlation via shared reset/refractory effects? default 1.0 = the "
                         "real, unscaled feature magnitude)")
    # the mouth's ACTUAL operating point (all already the class defaults; passed explicitly for the record)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--sub-read-window", type=int, default=120)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--rho-negative", type=float, default=0.10, help="rho <= this -> Phase-0 NEGATIVE (already decorrelated)")
    ap.add_argument("--rho-proceed", type=float, default=0.15, help="rho >= this -> Phase-0 CLEARED (proceed to Phase 1)")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_mouth_hid_correlation_diagnostic.json")
    args = ap.parse_args()

    ckpt = args.ckpt.format(seed=args.seed) if "{seed}" in args.ckpt else args.ckpt
    void_if(not Path(ckpt).exists(), f"checkpoint {ckpt} missing")
    ro_full = WKVReadout(ckpt)
    ro = _VocabTruncatedRO(ro_full, args.v_diag)

    # CLAUDE.md seed trap: build-twice determinism hash (cfg.seed, NOT actual_seed_used) before trusting the run.
    h1 = _thr_hash(args.seed, ro, args.B, args.sub_hid_pop, args.ou_std, args.sub_read_window, args.hid_gain,
                   args.ratio, args.n_bias, args.bias_drive_pA, args.settle_frac)
    h2 = _thr_hash(args.seed, ro, args.B, args.sub_hid_pop, args.ou_std, args.sub_read_window, args.hid_gain,
                   args.ratio, args.n_bias, args.bias_drive_pA, args.settle_frac)
    seeded = bool(h1 == h2)
    print(f"[seed-trap] thr hash {h1} == {h2} -> {'SEEDED' if seeded else 'NOT SEEDED'}", flush=True)

    feat = _fixed_feature(ro_full, args.seed) * args.feat_scale
    print(f"[feature] fixed feature vector norm={float(np.linalg.norm(feat)):.4g} D={ro_full.D} "
          f"(feat_scale={args.feat_scale})", flush=True)

    t0 = time.time()
    s_batch = BatchedSubstrateReadout(ro, args.seed, args.B, hid_pop=args.sub_hid_pop, pop=1, ou_std=args.ou_std,
                                      read_window=args.sub_read_window, hid_gain=args.hid_gain, ratio=args.ratio,
                                      settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA)
    build_secs = round(time.time() - t0, 1)
    print(f"[build] B={args.B} F={s_batch.F} Hp={s_batch.Hp} Hn={s_batch.F * s_batch.Hp} V_diag={ro.V} "
          f"in {build_secs}s", flush=True)

    t1 = time.time()
    per_feature, hid_counts, hidinh_counts = measure_hid_correlation(s_batch, feat, args.n_repeats)
    measure_secs = round(time.time() - t1, 1)

    rhos = [v["rho"] for v in per_feature.values() if v["rho"] is not None]
    gains = [v["gain_actual"] for v in per_feature.values() if v["gain_actual"] is not None]
    var_ratios = [v["var_ratio"] for v in per_feature.values() if v["var_ratio"] is not None]
    n_undefined = sum(1 for v in per_feature.values() if v["rho"] is None)

    summary = {
        "n_feature_groups": s_batch.F, "n_undefined_feature_groups": n_undefined,
        "n_trials_per_neuron": args.n_repeats * args.B,
        "rho_mean": (round(float(np.mean(rhos)), 4) if rhos else None),
        "rho_median": (round(float(np.median(rhos)), 4) if rhos else None),
        "rho_min": (round(float(np.min(rhos)), 4) if rhos else None),
        "rho_max": (round(float(np.max(rhos)), 4) if rhos else None),
        "gain_actual_mean": (round(float(np.mean(gains)), 4) if gains else None),
        "gain_ideal_sqrt_2Hp": round(float(np.sqrt(2 * s_batch.Hp)), 4),
        "var_ratio_mean": (round(float(np.mean(var_ratios)), 4) if var_ratios else None),
        "var_ratio_ideal_independent": 1.0,
        "var_ratio_full_common_mode": float(2 * s_batch.Hp),
    }
    rho_mean = summary["rho_mean"]
    if rho_mean is None:
        gate = "UNDEFINED"
    elif rho_mean <= args.rho_negative:
        gate = "PHASE0-NEGATIVE (already decorrelated -- STOP, do not build Phase 1)"
    elif rho_mean >= args.rho_proceed:
        gate = "PHASE0-CLEARED (proceed to Phase 1)"
    else:
        gate = "AMBIGUOUS (between the two thresholds -- read the finding, do not auto-decide)"
    summary["gate"] = gate

    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[gate] {gate}", flush=True)

    out = {
        "seed": args.seed, "seed_hash_check": {"thr_hash_1": h1, "thr_hash_2": h2, "seeded": seeded},
        "B": args.B, "n_repeats": args.n_repeats, "v_diag": args.v_diag,
        "operating_point": {"ou_std": args.ou_std, "hid_gain": args.hid_gain,
                            "sub_read_window": args.sub_read_window, "settle_frac": args.settle_frac,
                            "sub_hid_pop": args.sub_hid_pop, "ratio": args.ratio, "uniform_thresh": True},
        "feature_norm": float(np.linalg.norm(feat)), "feat_scale": args.feat_scale,
        "build_secs": build_secs, "measure_secs": measure_secs,
        "summary": summary,
        "per_feature": {str(k): v for k, v in per_feature.items()},
        "backend": os.environ.get("SIM_BACKEND", "numpy"), "argv": sys.argv,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"[done] -> {args.json} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
