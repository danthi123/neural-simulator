"""H1 (NOISE/RATE-limited) vs H2 (BIAS/decoder-direction-limited) diagnostic for the mouth read-SNR wall
(#80 continued, 2026-08-27).

READ FIRST (do not re-derive):
  - `research/findings/2026-08-27-mouth-readsnr-hid-decorrelation-PHASE0-NEG.md` -- the hidden population
    (hid+hidinh) is ALREADY ~uncorrelated (rho ~0, var_ratio ~1.0) at the mouth's real operating point, so the
    recurrent-inhibition decorrelation lever has nothing to decorrelate. That finding's own redirect names the
    open question this file answers: "the bottleneck is more likely the ABSOLUTE sample count / firing rate the
    fixed readout weights sum over ... a rate/gain lever is a promising next diagnostic."
  - `research/findings/2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md` -- the word-pool
    ENSEMBLE lever is inert by construction (deterministic conductance replicas, zero independent noise to
    average); the dendritic (Urbanczik-Senn) lever is already in flight on the GPU -- NOT duplicated here.
  - `research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py` -- the production pipeline this file
    explains: `sub_learned_recov_mean` ~0.34-0.37, `sub_copied_recov_mean` ~0.98, `weight_cosine_mean` ~0.135-0.136
    (`research/findings/raw/_wkv_readout_eprop_batched_substrate_6seed.json`,
    `..._marginclean_6seed.json`) -- the learned decoder's direction is nearly orthogonal to the copied target
    while the copied decoder (same read, same everything, only the weight DIRECTION differs) reads almost
    perfectly. That is already suggestive; this file makes it decisive by SWEEPING spike count independently of
    decoder identity.

THE QUESTION: is the ~0.34-0.37 plateau NOISE-LIMITED (H1: the hidden population's read window integrates too few
spikes, so a fixed-weight linear read is swamped by sampling variance -- SNR of a rate code ~ sqrt(spike count) --
FIX = a rate/gain mechanism, raise spikes/window) or BIAS-LIMITED (H2: the learned weight direction W_hat is just
wrong relative to the target head_w -- more spikes cannot fix a systematically wrong linear combination -- FIX =
a better decoder-LEARNING rule, not a rate/gain lever)?

METHOD (additive, NO sim/ edit, reuse-by-import of `BatchedSubstrateReadout`'s UNMODIFIED `_build_bridge`/`_wire`/
`batch_margin`/`set_weights`/`_calibrate_gain`/`_learn_substrate_batched`/`_thr_hash`/`_wcos` -- this file adds only
a NEW instrumented read loop, `_read_margin_and_spikes`, that measures spikes/neuron ALONGSIDE the margin the
unmodified `batch_margin` already computes -- the same "duplicate the read loop for instrumentation, wiring
untouched" precedent `_wkv_mouth_hid_correlation_diagnostic.py` already established):

  1. TRAIN a genuine (REDUCED-SCALE, but REAL V=1000 vocabulary) learned W_hat via the UNMODIFIED production
     substrate-forward rule (`_learn_substrate_batched`, mode=main, forward=substrate -- 0 host matmul on the
     forward, exactly the production path). This is NOT a decisive 6-seed replication (a small B / few epochs
     budget, chosen to stay CPU-numpy-tractable) -- it is a REAL, honestly-noisy learned decoder whose OWN
     `weight_cosine` is measured and reported, and cross-checked against the cited production 0.135-0.136 numbers.
  2. Hold BOTH W_hat (learned) and `hw` = `ro.head_w` (copied, the perfect-direction reference) FIXED for the rest
     of the run. Sweep THREE knobs that change the hidden population's firing WITHOUT touching either decoder:
     `hid_gain` (drive amplitude), `read_window` (integration time), and `ou_std` (noise magnitude, via
     `bridge.py`'s precomputed `ou_noise_std` scalar). All three are HOST-SIDE attributes/scalars read FRESH on
     every call/step -- `hid_gain`/`read_window` at the top of `batch_margin` each call, `ou_noise_std` at every
     `_run_one_simulation_step` (`bridge.py:8113/8330/9436`, precomputed once from `cfg.ou_std_current_pA` at
     `bridge.py:3836-3838` and NEVER re-read from cfg after that) -- so mutating them in place changes the read's
     operating point with NO WIRING REBUILD (verified empirically: a probe run built once and read at 3 different
     `read_window`/`hid_gain` values produced 3 different spike counts from the SAME CSR).
  3. At each operating point, for BOTH weight sets, measure (SAME fresh feature batch, SAME operating point --
     matched spike count BY CONSTRUCTION): (a) recov_argmax (identical formula to `_eval_substrate`:
     mean(mass_read)/mean(mass_ax) over the eval positions); (b) the hid+hidinh population's ACTUAL mean
     spikes/neuron over the post-settle read window (the `_wkv_mouth_hid_correlation_diagnostic.py` proxy: hid/
     hidinh receive ONLY external current -- `internal_density=0.0` -- so a neuron's own integrated spike count
     IS the "conductance it contributes downstream", the correct SNR-relevant sample-count quantity).

VERDICT (owner pre-registered):
  H1 (NOISE/RATE-LIMITED) if the LEARNED head's recov rises with measured spike count (a positive, non-trivial
      recov-vs-spike-count correlation) AND the COPIED head is ALSO poor at the SAME low-spike-count baseline
      (both heads noise-starved -- the copied direction is not enough to rescue a starved read).
  H2 (BIAS/DIRECTION-LIMITED) if the LEARNED head's recov stays FLAT across the whole spike-count range (rising
      spike count buys nothing) AND the COPIED head is ALREADY high at the SAME low-spike-count baseline where
      the learned head is stuck (perfect direction reads fine with few spikes; only a wrong direction cannot).
  Anything else -> AMBIGUOUS, report the numbers plainly, do not force a verdict.

Backend: numpy (CPU), per the task's cost-routing preference. Measured on this machine: ~15s to build B=8 @ real
V=1000; ~2.8s per `batch_margin`-equivalent call at read_window=120 -- fully CPU-tractable at the small B this
diagnostic uses (a MEASUREMENT run, not the decisive production scale).

Run (single seed, ~7 min on this machine -- run per-seed to stay under a foreground timeout):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_readout_rate_vs_bias_diagnostic \
    --seeds 42 --json research/findings/raw/_wkv_mouth_readout_rate_vs_bias_diagnostic_seed42.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from tools.lab import lever, void_if, undefined_if_empty, project_cost  # noqa: E402

from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import (  # noqa: E402
    BatchedSubstrateReadout, _calibrate_gain, _learn_substrate_batched, _thr_hash, _wcos,
)
from research.runners._wkv_mouth_readout_eprop_learn_derisk import (  # noqa: E402
    _host_feat, _positions, _positions_sub,
)
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _native, _load_eval  # noqa: E402


# ======================================================================================================================
# Instrumented read: IDENTICAL to `BatchedSubstrateReadout.batch_margin` (same drive construction, same settle/
# integrate loop, same margin formula) PLUS an accumulation of the hid+hidinh population's post-settle spike count
# -- the ONE thing `batch_margin` does not report. `_build_bridge`/`_wire`/`set_weights` are reused UNMODIFIED.
# ======================================================================================================================
def _read_margin_and_spikes(s_batch, feats_signed, silence_bias=False):
    from sim.backend import to_host, get_backend
    b = s_batch._b
    xp, _ = get_backend()
    nb = feats_signed.shape[0]
    assert nb == s_batch.B, "expects a full block (%d), got %d" % (s_batch.B, nb)
    featF = np.concatenate([np.maximum(feats_signed, 0.0), np.maximum(-feats_signed, 0.0)], axis=1)
    s_batch._reset()
    drive = xp.zeros(b.core_config.num_neurons, dtype=xp.float64)
    fdrive = xp.asarray(s_batch.hid_bias + s_batch.hid_gain * featF[:, s_batch.hid_dim]).reshape(-1)
    drive[s_batch._hid_flat] = fdrive
    drive[s_batch._hidinh_flat] = fdrive
    if s_batch.use_bias_pop and not silence_bias:
        be = xp.asarray(s_batch.bias_e_all.reshape(-1)); bi = xp.asarray(s_batch.bias_i_all.reshape(-1))
        drive[be] = s_batch.bias_drive_pA; drive[bi] = s_batch.bias_drive_pA
    if s_batch.floor_pA:
        drive[s_batch._wpool_flat] += s_batch.floor_pA
    b.cp_external_input_current[:] = drive.astype(b.cp_external_input_current.dtype)
    settle = int(s_batch.read_window * s_batch.settle_frac)
    ge_sum = xp.zeros(s_batch.B * s_batch.V, dtype=xp.float64)
    gi_sum = xp.zeros(s_batch.B * s_batch.V, dtype=xp.float64)
    spike_sum = 0.0
    n_hidpop = int(s_batch._hid_flat.shape[0] + s_batch._hidinh_flat.shape[0])   # B * 2*Hn
    n_acc = 0
    for step in range(s_batch.read_window):
        b._run_one_simulation_step()
        if step < settle:
            continue
        fs = b.cp_firing_states
        spike_sum += float(np.asarray(to_host(fs[s_batch._hid_flat])).sum()
                            + np.asarray(to_host(fs[s_batch._hidinh_flat])).sum())
        ge = b.cp_conductance_g_e[s_batch._wpool_flat].astype(xp.float64).reshape(s_batch.B * s_batch.V,
                                                                                   s_batch.P).sum(axis=1)
        gi = b.cp_conductance_g_i[s_batch._wpool_flat].astype(xp.float64).reshape(s_batch.B * s_batch.V,
                                                                                   s_batch.P).sum(axis=1)
        ge_sum += ge; gi_sum += gi; n_acc += 1
    b.cp_external_input_current[:] = 0.0
    n_acc = max(1, n_acc)
    margin = (s_batch.df_e * (ge_sum / n_acc) + s_batch.df_i * (gi_sum / n_acc))
    margin = np.asarray(to_host(margin)).reshape(s_batch.B, s_batch.V)
    mean_spikes_per_neuron_window = spike_sum / max(1, n_hidpop)     # spikes/neuron over the POST-SETTLE window
    return margin, mean_spikes_per_neuron_window


def _ou_noise_std_for(b, ou_std_pA):
    """Recompute bridge.py's precomputed `ou_noise_std` scalar (bridge.py:3836-3838) for a NEW ou_std_pA, without
    touching sim/ -- this is the exact host-side formula the bridge itself used at build time; mutating the
    result back onto `b.ou_noise_std` changes noise magnitude for every subsequent step (verified: read fresh at
    bridge.py:8113/8330/9436, never re-derived from cfg after init)."""
    cfg = b.core_config
    dt_sec = cfg.dt_ms / 1000.0
    tau_sec = cfg.ou_tau_ms / 1000.0
    return float(ou_std_pA * np.sqrt((1.0 - np.exp(-2.0 * dt_sec / tau_sec)) / 2.0))


def _recov_over_eval(s_batch, W, feats, Ys, PFs, unk, silence_bias=False):
    """recov_argmax / argmax_agree (IDENTICAL formula to `_eval_substrate`) + the mean spikes/neuron/window
    actually measured at s_batch's CURRENT (hid_gain, read_window, ou_noise_std) operating point, chunked into
    s_batch.B-sized batches (the last partial chunk is padded by repeating its first row; padded rows are excluded
    from the accumulated recov metrics but DO contribute real neural activity to the spike-count reading)."""
    B = s_batch.B
    N = feats.shape[0]
    s_batch.set_weights(W)
    mass_read = 0.0; mass_ax = 0.0; agree = 0.0; n_tot = 0; spikes_tot = 0.0; n_chunks = 0
    i = 0
    while i < N:
        j = min(i + B, N)
        chunk_feats = feats[i:j]
        pad = B - chunk_feats.shape[0]
        if pad > 0:
            chunk_feats = np.concatenate([chunk_feats, np.tile(chunk_feats[:1], (pad, 1))], axis=0)
        margin, spikes = _read_margin_and_spikes(s_batch, chunk_feats, silence_bias=silence_bias)
        n_valid = j - i
        for k in range(n_valid):
            row = margin[k].copy()
            if unk >= 0:
                row[unk] = -1e30
            win = -1 if float(row.max() - row.min()) <= 1e-9 else int(np.argmax(row))
            mass_read += (PFs[i + k][win] if win >= 0 else 0.0)
            mass_ax += PFs[i + k][Ys[i + k]]
            agree += float(win == Ys[i + k])
        n_tot += n_valid
        spikes_tot += spikes
        n_chunks += 1
        i = j
    n_tot = max(1, n_tot)
    return dict(recov_argmax=round((mass_read / n_tot) / max(1e-9, mass_ax / n_tot), 4),
                argmax_agree=round(agree / n_tot, 4),
                mean_spikes_per_neuron_window=round(spikes_tot / max(1, n_chunks), 4))


def _build_ops(args):
    """The sweep grid: baseline (production demo defaults) + 3 knobs, each varied independently, baseline value
    skipped in each sweep list to avoid a duplicate op."""
    base = dict(hid_gain=args.hid_gain, read_window=args.read_window, ou_std=args.ou_std)
    ops = [dict(name="baseline", **base)]
    for rw in args.rw_sweep:
        if rw == base["read_window"]:
            continue
        ops.append(dict(name="rw%d" % rw, hid_gain=base["hid_gain"], read_window=rw, ou_std=base["ou_std"]))
    for g in args.gain_sweep:
        if g == base["hid_gain"]:
            continue
        ops.append(dict(name="gain%d" % int(g), hid_gain=g, read_window=base["read_window"], ou_std=base["ou_std"]))
    for o in args.ou_sweep:
        if o == base["ou_std"]:
            continue
        ops.append(dict(name="ou%d" % int(o), hid_gain=base["hid_gain"], read_window=base["read_window"], ou_std=o))
    return ops


def run_seed(seed, ro, args):
    t_seed = time.time()
    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(args.frac_train * len(usable))
    train_ids, eval_ids = usable[:cut], usable[cut:]
    H, Y, _ = _positions(ro, train_ids, args.warmup, args.n_train_pos)
    sub_tuples, Ys, PFs = _positions_sub(ro, eval_ids, args.warmup, args.n_eval_demo)
    void_if(len(H) < args.batch or len(sub_tuples) < args.batch, "insufficient train/eval positions")
    feats_eval = np.asarray([_host_feat(ro, ap, an, tid) for (ap, an, tid) in sub_tuples])   # [N, D]

    hw = ro.head_w.copy()
    head_b = ro.head_b.astype(np.float64)
    unk = ro.unk_idx

    # -- build ONCE at the production demo operating point; hid_gain/read_window/ou_std are all mutated IN PLACE
    #    for the sweep below (see module docstring point 2 -- verified no rebuild is needed for any of the three). --
    s_batch = BatchedSubstrateReadout(ro, seed, args.batch, hid_pop=args.sub_hid_pop, pop=args.sub_pop,
                                      ou_std=args.ou_std, read_window=args.read_window, hid_gain=args.hid_gain,
                                      ratio=args.ratio, settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA)
    build_secs = round(time.time() - t_seed, 1)

    # CLAUDE.md seed trap: build-twice determinism hash (cfg.seed, NOT actual_seed_used).
    h1 = _thr_hash(seed, ro, args.sub_hid_pop, args.sub_pop, args.ou_std, args.read_window, args.hid_gain,
                   args.ratio, args.n_bias, args.bias_drive_pA)
    h2 = _thr_hash(seed, ro, args.sub_hid_pop, args.sub_pop, args.ou_std, args.read_window, args.hid_gain,
                   args.ratio, args.n_bias, args.bias_drive_pA)
    seeded = bool(h1 == h2)
    print("[seed-trap seed %d] thr hash %s == %s -> %s" % (seed, h1, h2, "SEEDED" if seeded else "NOT SEEDED"),
          flush=True)

    # -- TRAIN a genuine (reduced-scale, REAL V=1000) learned W_hat via the UNMODIFIED production substrate-
    #    forward rule (mode=main, forward=substrate, 0 host matmul on the forward). NOT a decisive replication --
    #    a small B / few-epoch budget kept CPU-numpy-tractable; the achieved weight_cosine is measured and reported
    #    (cross-checked against the cited production 0.135-0.136 6-seed numbers in the finding, not assumed). --
    t0 = time.time()
    s_batch.read_window = args.train_read_window                     # the TRAINING forward's OWN op point (cheaper)
    gain, gain_corr = _calibrate_gain(s_batch, ro, H[:args.batch], seed)
    train_args = argparse.Namespace(epochs=args.train_epochs, lr=args.train_lr,
                                    weight_decay=args.train_weight_decay, w_target=args.train_w_target,
                                    zero_init=False, forward="substrate", eval_every_epochs=0)
    W_learned, n_grad, n_mm = _learn_substrate_batched(seed, ro, s_batch, H, Y, train_args, gain, head_b, "main")
    train_secs = round(time.time() - t0, 1)
    wcos_learned = _wcos(W_learned, hw)
    print("[train seed %d] n_grad=%d host_matmul_on_forward=%d gain=%.5g wcos_vs_copied=%s (%ss)"
          % (seed, n_grad, n_mm, gain, wcos_learned, train_secs), flush=True)

    # -- SWEEP: W_learned and hw held FIXED for the rest of the run. Vary hid_gain / read_window / ou_std -- knobs
    #    that change the hidden population's firing WITHOUT touching either decoder -- and measure BOTH recov and
    #    the ACTUAL spike count achieved, for BOTH weight sets, at the SAME operating point each time (matched
    #    spike count by construction). Demo regime: bias-pop ACTIVE (silence_bias=False), matching how
    #    sub_learned/sub_copied are measured in production (`_eval_substrate` default). --
    ops = _build_ops(args)
    rows = []
    t1 = time.time()
    for op in ops:
        s_batch.hid_gain = float(op["hid_gain"])
        s_batch.read_window = int(op["read_window"])
        s_batch._b.ou_noise_std = _ou_noise_std_for(s_batch._b, float(op["ou_std"]))
        r_learned = _recov_over_eval(s_batch, W_learned, feats_eval, Ys, PFs, unk)
        r_copied = _recov_over_eval(s_batch, hw, feats_eval, Ys, PFs, unk)
        row = {"op": op["name"], "hid_gain": op["hid_gain"], "read_window": op["read_window"],
               "ou_std": op["ou_std"],
               "learned_recov": r_learned["recov_argmax"], "learned_agree": r_learned["argmax_agree"],
               "learned_spikes_per_neuron_window": r_learned["mean_spikes_per_neuron_window"],
               "copied_recov": r_copied["recov_argmax"], "copied_agree": r_copied["argmax_agree"],
               "copied_spikes_per_neuron_window": r_copied["mean_spikes_per_neuron_window"]}
        rows.append(row)
        print("[sweep seed %d %-8s] gain=%6.1f rw=%4d ou=%5.1f | spikes/neuron learned=%.3f copied=%.3f "
              "| recov learned=%.4f copied=%.4f"
              % (seed, op["name"], op["hid_gain"], op["read_window"], op["ou_std"],
                 row["learned_spikes_per_neuron_window"], row["copied_spikes_per_neuron_window"],
                 row["learned_recov"], row["copied_recov"]), flush=True)
    sweep_secs = round(time.time() - t1, 1)

    spikes_learned = np.array([r["learned_spikes_per_neuron_window"] for r in rows])
    spikes_copied = np.array([r["copied_spikes_per_neuron_window"] for r in rows])
    recov_learned = np.array([r["learned_recov"] for r in rows])
    recov_copied = np.array([r["copied_recov"] for r in rows])

    def _safe_corr(x, y):
        if x.std() < 1e-9 or y.std() < 1e-9:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    corr_learned = _safe_corr(spikes_learned, recov_learned)
    corr_copied = _safe_corr(spikes_copied, recov_copied)
    base_row = rows[0]
    m = {
        "seed": seed, "build_secs": build_secs, "train_secs": train_secs, "sweep_secs": sweep_secs,
        "seed_hash_check": {"thr_hash_1": h1, "thr_hash_2": h2, "seeded": seeded},
        "n_grad_steps": n_grad, "host_matmul_on_forward": n_mm, "forward_is_substrate": bool(n_mm == 0),
        "weight_cosine_learned_vs_copied": wcos_learned,
        "gain_calib": round(float(gain), 6), "gain_calib_corr": gain_corr,
        "baseline_op": base_row,
        "recov_spike_corr_learned": (round(corr_learned, 4) if corr_learned is not None else None),
        "recov_spike_corr_copied": (round(corr_copied, 4) if corr_copied is not None else None),
        "recov_learned_range": [round(float(recov_learned.min()), 4), round(float(recov_learned.max()), 4)],
        "recov_copied_range": [round(float(recov_copied.min()), 4), round(float(recov_copied.max()), 4)],
        "spikes_learned_range": [round(float(spikes_learned.min()), 3), round(float(spikes_learned.max()), 3)],
        "spikes_copied_range": [round(float(spikes_copied.min()), 3), round(float(spikes_copied.max()), 3)],
        "rows": rows,
    }
    lever("recov_vs_spike_count_learned_seed%d" % seed,
          before=round(float(recov_learned[np.argmin(spikes_learned)]), 4),
          after=round(float(recov_learned[np.argmax(spikes_learned)]), 4),
          required=False, continuous=corr_learned)
    del s_batch
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=8)                    # B block-diagonal copies (build+train+eval)
    ap.add_argument("--n-train-pos", type=int, default=640)
    ap.add_argument("--n-eval-demo", type=int, default=24)
    # training (a REAL but reduced-scale learned decoder; NOT a decisive replication -- see module docstring)
    ap.add_argument("--train-epochs", type=int, default=1)
    ap.add_argument("--train-lr", type=float, default=0.5)
    ap.add_argument("--train-weight-decay", type=float, default=8e-4)
    ap.add_argument("--train-w-target", type=float, default=40.0)
    ap.add_argument("--train-read-window", type=int, default=60)
    # production demo operating point (the sweep's "baseline" / matched-low-spike-count anchor)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-pop", type=int, default=1)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    # sweep grids (spike-count-changing knobs; the value equal to the baseline is auto-skipped in each list)
    ap.add_argument("--rw-sweep", type=str, default="60,150,300,480")
    ap.add_argument("--gain-sweep", type=str, default="30,60,120,240")
    ap.add_argument("--ou-sweep", type=str, default="10,40,80,160")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wkv_mouth_readout_rate_vs_bias_diagnostic.json")
    args = ap.parse_args()
    args.rw_sweep = [int(x) for x in args.rw_sweep.split(",") if x.strip()]
    args.gain_sweep = [float(x) for x in args.gain_sweep.split(",") if x.strip()]
    args.ou_sweep = [float(x) for x in args.ou_sweep.split(",") if x.strip()]

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    t_all = time.time()
    for si, seed in enumerate(seeds):
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print("[skip] seed %d: checkpoint %s missing" % (seed, ckpt), flush=True)
            continue
        ro = WKVReadout(ckpt)
        m = run_seed(seed, ro, args)
        results.append(m)
        project_cost("rate-vs-bias diagnostic", si + 1, len(seeds), time.time() - t_all, warn_hours=1.0)

    undefined_if_empty("rate_vs_bias_seeds", len(results), len(results), len(seeds))
    summary = {}
    if results:
        wcos_vals = [r["weight_cosine_learned_vs_copied"] for r in results]
        corr_l = [r["recov_spike_corr_learned"] for r in results if r["recov_spike_corr_learned"] is not None]
        corr_c = [r["recov_spike_corr_copied"] for r in results if r["recov_spike_corr_copied"] is not None]
        base_learned = [r["baseline_op"]["learned_recov"] for r in results]
        base_copied = [r["baseline_op"]["copied_recov"] for r in results]
        base_spikes_l = [r["baseline_op"]["learned_spikes_per_neuron_window"] for r in results]
        base_spikes_c = [r["baseline_op"]["copied_spikes_per_neuron_window"] for r in results]
        summary = {
            "n_seeds": len(results),
            "weight_cosine_mean": round(float(np.mean(wcos_vals)), 4),
            "recov_spike_corr_learned_mean": (round(float(np.mean(corr_l)), 4) if corr_l else None),
            "recov_spike_corr_copied_mean": (round(float(np.mean(corr_c)), 4) if corr_c else None),
            "baseline_learned_recov_mean": round(float(np.mean(base_learned)), 4),
            "baseline_copied_recov_mean": round(float(np.mean(base_copied)), 4),
            "baseline_spikes_per_neuron_window_learned_mean": round(float(np.mean(base_spikes_l)), 3),
            "baseline_spikes_per_neuron_window_copied_mean": round(float(np.mean(base_spikes_c)), 3),
            "recov_learned_range_over_all_seeds": [
                round(float(min(r["recov_learned_range"][0] for r in results)), 4),
                round(float(max(r["recov_learned_range"][1] for r in results)), 4)],
            "recov_copied_range_over_all_seeds": [
                round(float(min(r["recov_copied_range"][0] for r in results)), 4),
                round(float(max(r["recov_copied_range"][1] for r in results)), 4)],
        }
        # VERDICT (owner pre-registered, module docstring): H1 needs a rising learned-recov-vs-spike-count trend
        # AND a copied head that is ALSO poor at the matched low-spike baseline. H2 needs a FLAT learned trend AND
        # a copied head that is ALREADY high at that same baseline.
        copied_high_at_baseline = summary["baseline_copied_recov_mean"] >= 0.75
        learned_low_at_baseline = summary["baseline_learned_recov_mean"] <= 0.55
        cl = summary["recov_spike_corr_learned_mean"]
        learned_flat = (cl is None or abs(cl) < 0.4)
        if copied_high_at_baseline and learned_low_at_baseline and learned_flat:
            verdict = "H2_BIAS_LIMITED"
        elif (cl is not None and cl > 0.4 and not copied_high_at_baseline):
            verdict = "H1_NOISE_RATE_LIMITED"
        else:
            verdict = "AMBIGUOUS"
        summary["verdict"] = verdict
        print("\n[SUMMARY] %s" % json.dumps(summary, indent=2), flush=True)
        print("[VERDICT] %s" % verdict, flush=True)

    out = {"results": _native(results), "summary": _native(summary), "seeds": seeds,
           "backend": os.environ.get("SIM_BACKEND", "numpy"),
           "elapsed_s": round(time.time() - t_all, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print("[done] %d rows -> %s (%.0fs)" % (len(results), args.json, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
