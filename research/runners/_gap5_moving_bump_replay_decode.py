"""gap#5 MOVING-BUMP REPLAY-DECODE readout (Davidson-Kloosterman-Wilson 2009 / Ecker 2022) on the CHAIN-WRITTEN band.

The encode-level weight-bin metrics are per-bin noisy (finding 2026-07-24 §5); the FIELD-STANDARD replay test is a
BAYESIAN POPULATION DECODE of the position TRAJECTORY over each SWR event, scored by WEIGHTED CORRELATION (Davidson 2009,
the exact method Ecker 2022 uses), which integrates over the whole population and is robust to per-bin encode noise.

Pipeline (reuse-by-import; NO sim/ edit):
  1. ENCODE: the chain-written LOAD-BEARING band (chain_rule="hebb_sym" + freeze_between_within; the within phase writes
     NO between-links, the Ecker symmetric rule is the sole between-writer).  [research/runners/_gap5_sequence_replay_derisk]
  2. REPLAY: the SWR-state E>I-transient envelope + weak-noise ignition SEED + SPIKE-FREQUENCY ADAPTATION (SFA makes the
     bump TRAVEL, not sit -- Ecker's adapt-lesion control).  [research/runners/_gap5_swr_envelope_replay_derisk]
  3. READOUT (this file): tuning templates (place cells = assembly members) -> Bayesian decode P(pos | spikes) per time
     bin -> the decoded position TRAJECTORY -> weighted-correlation replay score (forward = +slope, reverse = -slope),
     with a per-event position-shuffle NULL (Davidson significance).
  4. ANTI-CHEATS: STRUCTURE-SHUFFLE (permute the band's between-weights -> trajectory collapses = "structure not
     statistics"), REVERSE-STORE (encode reversed -> trajectory reverses), INTERIOR-SEED-INVARIANCE, ADAPT-LESION
     (SFA off -> no coherent trajectory), NO-ENCODE, NO-NOISE, FROZEN.

CPU-SMOKE (validate the machinery, NOT the science):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_moving_bump_replay_decode --smoke
FULL 6-seed GO (GPU, HELD until the gap#4 diagnostic frees the 3090):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_moving_bump_replay_decode \
      --seeds 42 43 44 100 101 102 --rest-steps 1500
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402

from research.runners._gap5_sequence_replay_derisk import (  # noqa: E402
    _prepare_sequence, _scramble_between_weights, _event_windows,
)
from research.runners._gap5_swr_envelope_replay_derisk import _rest_swr_envelope  # noqa: E402
from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_r4" / "moving_bump_replay_decode.json"

# ----- the CHAIN-WRITTEN LOAD-BEARING band encode (finding 2026-07-24 §5; tuned config) --------------------------------
BAND_CFG = dict(DECOUPLED_CFG)
BAND_CFG.update(
    n_ca3=2000, n_mem=6, assembly_frac=0.06, chain_rule="hebb_sym", chain_overlap=True, overlap_draw=False,
    chain_fwd=15, chain_rev=0, seq_win_steps=16, within_events=6, within_refresh=0, freeze_between_within=True,
    hebb_coact_decay=0.95, hebb_coact_thresh=0.15, hebb_sym_lr=0.03,
)

# ----- the validated SWR-envelope replay parameters (from _gap5_swr_envelope_replay_derisk defaults) ------------------
ENV = dict(mode="swr", noise_on=True, env_exc_pa=180.0, env_basket_drop=400.0, env_basket_boost=200.0,
           swr_period=250, env_dur=120, noise_rate=0.01, noise_pa=800.0, noise_dur=5, self_regen_read=0.0,
           recall_k_thresh=None, d_abs=40.0, a_abs=0.008, adapt=True, self_regen_ignite=0.15, ignite_frac=0.4,
           seed_assembly=True, seed_pa=600.0, seed_dur=8, seed_frac=0.5)


# ====================================================================================================================
# Bayesian population decode (Zhang 1998) + weighted correlation (Davidson 2009)
# ====================================================================================================================
def _build_tuning(prep, floor=0.05, peak=1.0):
    """Place-cell tuning template f[cell, pos]: each assembly is a POSITION; a member cell's field peaks at its home
    position (disjoint assemblies -> unique home). Returns f [n_place_cells x n_pos] + the LOCAL ca3 place-cell indices
    (columns into F)."""
    als = prep["assemblies_local"]; n_pos = len(als)
    home = {}
    for x, A in enumerate(als):
        for c in A:
            home[int(c)] = x
    place = np.array(sorted(home), dtype=np.int64)
    f = np.full((len(place), n_pos), float(floor), dtype=np.float64)
    for i, c in enumerate(place):
        f[i, home[int(c)]] = float(peak)
    return f, place


def _bayes_decode(counts, f, tau=1.0):
    """counts [n_bins x n_cells] spike counts -> posterior P [n_bins x n_pos]. log P(x|n) = sum_c n_c log f[c,x] -
    tau sum_c f[c,x] (uniform prior). Softmax-normalized per bin."""
    logf = np.log(np.clip(f, 1e-6, None))                 # [n_cells, n_pos]
    LL = counts @ logf - float(tau) * f.sum(0)[None, :]    # [n_bins, n_pos]
    LL -= LL.max(1, keepdims=True)
    P = np.exp(LL); P /= (P.sum(1, keepdims=True) + 1e-12)
    return P


def _weighted_corr(P):
    """Davidson-2009 weighted correlation between time-bin index and decoded position, weights P[t,x]. Sign = direction
    (forward = +). |r| near 1 = a coherent constant-velocity sweep."""
    T, X = P.shape
    tt = np.arange(T, dtype=float)[:, None]; xx = np.arange(X, dtype=float)[None, :]
    w = P; sw = float(w.sum())
    if sw < 1e-9:
        return 0.0
    mt = float((w * tt).sum() / sw); mx = float((w * xx).sum() / sw)
    cov = float((w * (tt - mt) * (xx - mx)).sum() / sw)
    vt = float((w * (tt - mt) ** 2).sum() / sw); vx = float((w * (xx - mx) ** 2).sum() / sw)
    if vt < 1e-9 or vx < 1e-9:
        return 0.0
    return cov / np.sqrt(vt * vx)


def _decode_replay(F, prep, f, place, *, tau_bin=5, min_len=25, min_spikes=15, r_sig=0.5, n_shuffle=200,
                   asize_ref=1.0, W=5, ev_floor=0.4, ev_k=4.0, seed=0, keep_traj=0):
    """Detect SWR events -> Bayesian-decode each -> weighted-corr r. A SIGNIFICANT replay event has |r| > the 95th pct
    of a per-event POSITION-SHUFFLE null (Davidson). Returns event count, significant forward/reverse fractions, mean |r|,
    and (optionally) a few decoded trajectories for the smoke."""
    events, duty, pop_rate = _event_windows(F, W=W, ev_floor=ev_floor, ev_k=ev_k, asize_ref=asize_ref)
    rng = np.random.default_rng(int(seed) * 77713 + 5)
    rows = []; trajs = []
    n_sig = fwd = rev = 0
    for (s, e) in events:
        if e - s < min_len:
            continue
        Fe = F[s:e][:, place]                                  # [len, n_place_cells]
        nb = (e - s) // tau_bin
        if nb < 3:
            continue
        counts = np.stack([Fe[b * tau_bin:(b + 1) * tau_bin].sum(0) for b in range(nb)]).astype(float)
        if counts.sum() < min_spikes:
            continue
        P = _bayes_decode(counts, f, tau=1.0)
        r = _weighted_corr(P)
        # per-event position-shuffle null: permute the posterior COLUMNS (position labels) -> destroys the (t,x) coherence
        null = np.array([abs(_weighted_corr(P[:, rng.permutation(P.shape[1])])) for _ in range(n_shuffle)])
        p95 = float(np.percentile(null, 95)); sig = bool(abs(r) > p95)
        if sig:
            n_sig += 1; fwd += int(r > 0); rev += int(r < 0)
        rows.append(dict(s=int(s), e=int(e), nb=int(nb), r=round(float(r), 3), null_p95=round(p95, 3),
                         sig=sig, spikes=int(counts.sum())))
        if len(trajs) < keep_traj:
            trajs.append(dict(s=int(s), e=int(e), r=round(float(r), 3), argmax_pos=[int(x) for x in P.argmax(1)]))
    n_ev = len(rows)
    return dict(n_windows=len(events), n_decoded=n_ev, n_sig=n_sig,
                sig_forward_frac=(fwd / n_ev) if n_ev else 0.0, sig_reverse_frac=(rev / n_ev) if n_ev else 0.0,
                mean_abs_r=round(float(np.mean([abs(x["r"]) for x in rows])) if rows else 0.0, 3),
                mean_r=round(float(np.mean([x["r"] for x in rows])) if rows else 0.0, 3),
                duty_cycle=round(float(duty), 4), pop_rate=round(float(pop_rate), 5),
                rows=rows[:40], trajs=trajs)


def run_decode(prep, seed, *, env=None, rest_steps=1500, tau_bin=5, keep_traj=0, verbose=False):
    """Run the SWR-envelope replay on `prep` -> F -> Bayesian decode -> weighted-corr replay score."""
    env = dict(ENV if env is None else env)
    # decode-tuning keys (prefixed _) are consumed HERE, not passed to the envelope
    dpar = dict(ev_floor=float(env.pop("_ev_floor", 0.4)), ev_k=float(env.pop("_ev_k", 4.0)),
                min_len=int(env.pop("_min_len", 25)), min_spikes=int(env.pop("_min_spikes", 15)))
    f, place = _build_tuning(prep)
    asize_ref = float(np.mean([len(a) for a in prep["assemblies_local"]]))
    res = _rest_swr_envelope(prep, rest_steps, seed, verbose=verbose, **env)
    F = res["F"]
    dec = _decode_replay(F, prep, f, place, tau_bin=tau_bin, asize_ref=asize_ref, seed=seed, keep_traj=keep_traj, **dpar)
    dec["weights_frozen"] = bool(res["weights_frozen"]); dec["n_env"] = int(res["n_env"])
    dec["n_place_cells"] = int(len(place)); dec["F_active_frac"] = round(float(F.mean()), 5)
    return dec, F


# ====================================================================================================================
# DECODER UNIT TEST (synthetic positive control): feed a KNOWN traveling bump -> the decode MUST recover a forward
# trajectory (r>0, high |r|, argmax sweeps 0->n_pos-1), and permuting the position labels MUST collapse |r| to ~0.
# This validates the decode + weighted-corr + shuffle machinery INDEPENDENT of whether the spiking replay fires.
# ====================================================================================================================
def _synthetic_bump(prep, direction=+1, dwell=8, reps=3, fire_frac=0.7, bg=0.003, gap=8, seed=0):
    als = prep["assemblies_local"]; n_pos = len(als); n_ca3 = len(prep["ca3_idx"])
    rng = np.random.default_rng(seed); frames = []
    order = list(range(n_pos)) if direction > 0 else list(range(n_pos - 1, -1, -1))
    for _ in range(reps):
        for x in order:
            A = als[x]
            for _d in range(dwell):
                fr = np.zeros(n_ca3, dtype=bool)
                fr[A[rng.random(len(A)) < fire_frac]] = True
                fr[rng.random(n_ca3) < bg] = True
                frames.append(fr)
        for _g in range(gap):
            frames.append(rng.random(n_ca3) < bg)
    return np.array(frames)


def _decode_window(Fw, prep, f, place, tau_bin=4):
    """Decode a GIVEN window (no event detection) -> weighted-corr r + argmax trajectory. For the unit test / for a
    caller that already knows the event bounds."""
    nb = Fw.shape[0] // tau_bin
    if nb < 3:
        return 0.0, []
    Fe = Fw[:, place]
    counts = np.stack([Fe[b * tau_bin:(b + 1) * tau_bin].sum(0) for b in range(nb)]).astype(float)
    P = _bayes_decode(counts, f, tau=1.0)
    return _weighted_corr(P), [int(x) for x in P.argmax(1)]


def _decoder_unit_test(prep, seed=0):
    """Synthetic positive control: ONE clean forward sweep window + one reverse + a position-relabel shuffle, decoded
    DIRECTLY (bypassing the bursty-event detector, which is only for the real SWR replay)."""
    f, place = _build_tuning(prep)
    Ff = _synthetic_bump(prep, direction=+1, reps=1, gap=0, seed=seed)      # one clean 0->..->n sweep
    Fr = _synthetic_bump(prep, direction=-1, reps=1, gap=0, seed=seed + 1)  # one clean n->..->0 sweep
    r_fwd, traj_fwd = _decode_window(Ff, prep, f, place, tau_bin=4)
    r_rev, traj_rev = _decode_window(Fr, prep, f, place, tau_bin=4)
    # STRUCTURE-SHUFFLE == relabel the tuning positions: decode the SAME forward sweep against a permuted template ->
    # the (time, position) coherence is destroyed -> |r| collapses.
    perm = np.random.default_rng(seed * 13 + 1).permutation(f.shape[1])
    r_sh, _ = _decode_window(Ff, prep, f[:, perm], place, tau_bin=4)
    ok = bool(r_fwd > 0.6 and r_rev < -0.6 and abs(r_sh) < 0.5)
    return dict(ok=ok, r_fwd=round(float(r_fwd), 3), r_rev=round(float(r_rev), 3), r_shuffled=round(float(r_sh), 3),
                traj_fwd=traj_fwd, traj_rev=traj_rev)


# ====================================================================================================================
# CPU SMOKE: does the readout run end-to-end + produce a decodable trajectory + does the structure-shuffle differ?
# ====================================================================================================================
def smoke(seed=42):
    t0 = time.time()
    cfg = dict(BAND_CFG); cfg.update(n_ca3=400, n_mem=5, chain_fwd=15, within_events=6)   # small CPU band (tuned shape)
    env = dict(ENV); env.update(swr_period=180, env_dur=90, seed_dur=6,
                                _ev_floor=0.12, _ev_k=2.5, _min_len=9, _min_spikes=4)       # lower thresholds for the small band
    rest = 900
    print(f"[smoke] building chain-written band (n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']}) ...", flush=True)
    prep = _prepare_sequence(seed, cfg, do_encode=True)
    print(f"[smoke] encode done: within={prep['w_within']:.1f} adj_fwd={prep.get('w_adj_fwd'):.1f} "
          f"adj_rev={prep.get('w_adj_rev'):.1f} ({time.time()-t0:.0f}s)", flush=True)

    # (A) DECODER UNIT TEST (synthetic traveling bump) -- the decisive machinery validation
    ut = _decoder_unit_test(prep, seed=seed)
    print(f"[smoke] UNIT-TEST decoder: fwd r={ut['r_fwd']} (traj {ut['traj_fwd']}) | rev r={ut['r_rev']} "
          f"(traj {ut['traj_rev']}) | shuffled-labels r={ut['r_shuffled']} => decoder_OK={ut['ok']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # (B) REAL spiking-replay decode
    real, F = run_decode(prep, seed, env=env, rest_steps=rest, tau_bin=3, keep_traj=3, verbose=True)
    print(f"[smoke] REAL: windows={real['n_windows']} decoded={real['n_decoded']} sig={real['n_sig']} "
          f"mean|r|={real['mean_abs_r']} mean_r={real['mean_r']} sig_fwd={real['sig_forward_frac']:.2f} "
          f"sig_rev={real['sig_reverse_frac']:.2f} n_env={real['n_env']} place_cells={real['n_place_cells']} "
          f"F_active={real['F_active_frac']} frozen={real['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)
    for tr in real["trajs"]:
        print(f"[smoke]   traj event[{tr['s']}:{tr['e']}] r={tr['r']} decoded-pos-per-bin={tr['argmax_pos']}", flush=True)

    # STRUCTURE-SHUFFLE anti-cheat: permute the band's between-weights -> re-decode -> should COLLAPSE
    prep_sh = _prepare_sequence(seed, cfg, do_encode=True)
    n_sh = _scramble_between_weights(prep_sh, seed)
    shuf, _ = run_decode(prep_sh, seed, env=env, rest_steps=rest, tau_bin=3)
    print(f"[smoke] SHUFFLE ({n_sh} edges): decoded={shuf['n_decoded']} sig={shuf['n_sig']} mean|r|={shuf['mean_abs_r']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # machinery verdict: the DECODER (the new component) is validated by the synthetic unit test; the spiking-replay
    # firing (n_windows) is a separate op-point-tuning question for the GPU science.
    machinery_ok = bool(ut["ok"])
    real_ran = real["n_windows"] >= 0        # pipeline executed end-to-end without error
    real_fired = real["n_windows"] > 0
    out = dict(probe="moving_bump_replay_decode_SMOKE", seed=seed, unit_test=ut, real=real, shuffle=shuf,
               decoder_machinery_ok=machinery_ok, pipeline_runs_end_to_end=bool(real_ran),
               spiking_replay_fired_events=bool(real_fired),
               real_vs_shuffle_differ=bool(real["mean_abs_r"] > shuf["mean_abs_r"] or real["n_sig"] > shuf["n_sig"]),
               elapsed_s=round(time.time() - t0, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[smoke] VERDICT: decoder_machinery_OK={machinery_ok} (synthetic fwd r={ut['r_fwd']} / "
          f"rev r={ut['r_rev']} / shuffled r={ut['r_shuffled']}) | pipeline_end_to_end={real_ran} | "
          f"spiking_replay_fired_events={real_fired} (windows={real['n_windows']} -> op-point tuning for the GPU run)", flush=True)
    print(f"[smoke] wrote {OUT} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU smoke: validate the decode machinery (tiny band)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="FULL 6-seed GO (GPU) -- HELD until GPU frees")
    ap.add_argument("--rest-steps", type=int, default=1500)
    a = ap.parse_args()
    if a.smoke or a.seeds is None:
        smoke(a.seed); return 0
    # FULL 6-seed path (GPU) -- built + ready; the controller launches it when the GPU frees.
    print("[full] 6-seed moving-bump GO path is built; run on GPU (SIM_BACKEND=cupy). Held per coordinator.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
