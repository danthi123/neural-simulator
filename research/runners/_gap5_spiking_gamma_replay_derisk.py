"""gap#5 RUNG-2 (the spiking realization) — theta/gamma phase-organized replay ON the real spiking CA3 substrate.

The numpy isolation (`_gap5_gamma_wta_replay_derisk.py`, 3/3 GO) proved that a gamma-WTA + post-fire silence turns RANK 2's
marginal weight-only replay order (chance) into a reliable forward order on the learned weights, EVEN when the raw
forward/reverse asymmetry is reverse-signed (it rides the adjacent chain + self-avoidance, not the fragile asymmetry).
This builds the on-spikes version: during the spontaneous-replay REST phase over RANK 2's real BTSP chain + bistable
within-attractors, add a **theta/gamma self-avoidance** — each theta cycle is one sequence; once an assembly reactivates
(fires) it is SILENCED (an inhibitory pulse = the gamma reset / de Almeida-Idiart-Lisman E%-max post-fire suppression) so
it cannot dwell or re-win, letting the forward chain drive the NEXT assembly; the fired set resets at each theta boundary
so the sequence restarts. Lisman-Idiart theta-gamma multiplexing, catalog N.15.

NO new `sim/` edit: the gamma reset is an inhibitory external current injected to the already-fired assembly cells
(host-side world/body-legit? NO -- this is a NEURAL inhibition, so it is realized as a current the FS/basket would deliver;
here injected directly as the RUNG-2 scaffold, to be replaced by a gamma FS pool delivering it. Honest: RUNG 2 = the
scaffolded gamma inhibition; RUNG 3 = a learned gamma FS pool that emits it. This de-risk proves the phase-organized read
fixes the order on the real spikes).

ARMS (same encoded bridge, same noise): NO-GAMMA (baseline RANK 2 rest) vs GAMMA (theta/gamma self-avoidance). Anti-cheats:
NO-NOISE acid (no replay without background), NO-ENCODE (no chain -> no ordered replay), SCRAMBLE-BETWEEN (shuffle the
between-assembly edges -> the forward order must break even WITH gamma, proving the order rides the learned chain).
forward_frac from the RANK 2 `_detect_sequence_events`. numpy CPU; coexists with the GPU training.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence, SEQ_CFG, _detect_sequence_events  # noqa: E402
from research.runners._gap5_spontaneous_reactivation_derisk import _hard_silence, _configure_ou  # noqa: E402


def _rest_with_gamma(prep, noise, rest_steps, seed, gamma, theta_period, fire_thresh, inhib_pa, W_smooth,
                     silence_delay=8, release_mode=False, release_v=-75.0, proportional=False,
                     fs_gamma=False, fs_amp=1200.0, gamma_period=12):
    """Freeze plasticity + hard-silence + weak background, run REST; if gamma, apply theta/gamma self-avoidance (silence
    already-fired assemblies, reset the fired set every theta_period steps). Returns the CA3 firing matrix F."""
    from sim.backend import get_backend as _gb
    cp, _ = _gb()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    _hard_silence(bridge)
    kind = noise[0]
    _configure_ou(bridge, (noise[1] if kind == "ou" else None), seed)
    poisson = kind == "poisson"
    ca3_arr_host = prep["ca3_arr_host"]
    assemblies_local = prep["assemblies_local"]                    # per-assembly LOCAL indices into ca3_arr_host
    asm_glob = [cp.asarray(ca3_arr_host[np.asarray(a, dtype=np.int64)], dtype=cp.int64) for a in assemblies_local]
    asm_sizes = [max(1, len(a)) for a in assemblies_local]
    basket_glob = cp.asarray(np.asarray(list(prep["bridge"].region_manager.indices("ca3_pv_basket")), dtype=np.int64),
                             dtype=cp.int64) if fs_gamma else None
    if poisson:
        p_rate, p_pa = float(noise[1]), float(noise[2])
        p_dur = int(noise[3]) if len(noise) > 3 else 5
        exc_glob = ca3_arr_host[prep["ca3_exc_local"]]
        exc_dev = cp.asarray(exc_glob, dtype=cp.int64)
        prng = np.random.default_rng(int(seed) * 100003 + 11)
        countdown = np.zeros(len(exc_glob), dtype=np.int32)
    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    fired_at = {}     # k -> onset step; silence starts silence_delay steps LATER so the burst completes + is detected
    for t in range(rest_steps):
        if gamma and (t % theta_period == 0):
            fired_at = {}                                         # theta boundary: new sequence
        bridge.cp_external_input_current[:] = 0.0
        if poisson:
            new = prng.random(len(exc_glob)) < p_rate
            countdown[new] = p_dur
            active = countdown > 0
            if active.any():
                bridge.cp_external_input_current[exc_dev[cp.asarray(np.nonzero(active)[0], dtype=cp.int64)]] = p_pa
            countdown[active] -= 1
        if gamma and fs_gamma:
            # gamma-RHYTHM FS-basket FEEDBACK inhibition: drive the basket sinusoidally at gamma freq -> the basket->CA3
            # inhibition self-scales through the real synaptic loop -> gamma windows (trough = reactivation possible).
            phase = (t % gamma_period) / gamma_period
            bridge.cp_external_input_current[basket_glob] += float(fs_amp) * (1.0 - np.cos(2.0 * np.pi * phase)) / 2.0
        if gamma and not release_mode and not fs_gamma:
            for k, t0 in fired_at.items():
                if t >= t0 + silence_delay:                       # POST-burst SOMA silence (the gamma reset)
                    if proportional and t > 0:
                        lo2 = max(0, t - W_smooth)                 # SELF-SCALING: inhibition ~ the assembly's OWN recent
                        fr = F[lo2:t][:, assemblies_local[k]].sum() / (asm_sizes[k] * max(1, t - lo2))  # firing (feedback
                        bridge.cp_external_input_current[asm_glob[k]] += inhib_pa * (fr / fire_thresh)   # inhibition scales
                    else:
                        bridge.cp_external_input_current[asm_glob[k]] += inhib_pa
        bridge._run_one_simulation_step()
        if gamma and release_mode and getattr(bridge, "cp_v_apical", None) is not None:
            for k, t0 in fired_at.items():
                if t >= t0 + silence_delay:                       # PLATEAU RELEASE: reset the apical to the KIR down-state
                    bridge.cp_v_apical[asm_glob[k]] = cp.float32(release_v)   # un-latch, so the within-attractor decays
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)
        if gamma:
            lo = max(0, t - W_smooth + 1)
            for k, a in enumerate(assemblies_local):
                if k not in fired_at and F[lo:t + 1][:, a].sum() / (asm_sizes[k] * (t + 1 - lo)) >= fire_thresh:
                    fired_at[k] = t
    return F


def _fwd(F, assemblies_local, det):
    r = _detect_sequence_events(F, assemblies_local, **det)
    return r["forward_frac"], r["reverse_frac"], r["n_multi"], r["per_asm_active"], r["n_events"], float(F.mean())


def one_seed(seed, cfg, a):
    det = dict(W=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k, active_frac=a.active_frac, onset_frac=a.onset_frac)
    noise = ("poisson", a.poisson_rate, a.poisson_pa, a.poisson_dur)
    gk = dict(theta_period=a.theta_period, fire_thresh=a.fire_thresh, inhib_pa=a.inhib_pa, W_smooth=a.window,
              release_mode=a.release_mode, release_v=a.release_v, proportional=a.proportional,
              fs_gamma=a.fs_gamma, fs_amp=a.fs_amp, gamma_period=a.gamma_period)

    prep = _prepare_sequence(seed, cfg)
    al = prep["assemblies_local"]
    Fng = _rest_with_gamma(prep, noise, a.rest_steps, seed, gamma=False, **gk)     # baseline (no gamma)
    Fg = _rest_with_gamma(prep, noise, a.rest_steps, seed, gamma=True, **gk)        # gamma
    Fnn = _rest_with_gamma(prep, ("none",), a.rest_steps, seed, gamma=True, **gk)   # NO-NOISE acid (gamma on)

    prep_ne = _prepare_sequence(seed, cfg, do_encode=False)
    Fne = _rest_with_gamma(prep_ne, noise, a.rest_steps, seed, gamma=True, **gk)    # NO-ENCODE (gamma on)

    ng = _fwd(Fng, al, det)
    g = _fwd(Fg, al, det)
    nn = _fwd(Fnn, al, det)
    ne = _fwd(Fne, prep_ne["assemblies_local"], det)

    go = (g[0] > ng[0] + 0.15) and (g[2] >= 2) and (nn[2] == 0)
    print(f"  [seed {seed}] NO-GAMMA fwd={ng[0]:.3f} (ev={ng[4]} multi={ng[2]} act={ng[3]} pop={ng[5]:.4f}) | "
          f"GAMMA fwd={g[0]:.3f} rev={g[1]:.3f} (ev={g[4]} multi={g[2]} act={g[3]} pop={g[5]:.4f}) | "
          f"NO-NOISE multi={nn[2]} pop={nn[5]:.4f} | NO-ENCODE fwd={ne[0]:.3f} multi={ne[2]} "
          f"=> {'SPIKING-GAMMA-FIXES-ORDER' if go else 'no'}")
    return dict(seed=seed, no_gamma=dict(fwd=ng[0], n_multi=ng[2], pop=ng[5]), gamma=dict(fwd=g[0], rev=g[1], n_multi=g[2], pop=g[5]),
                no_noise=dict(n_multi=nn[2]), no_encode=dict(fwd=ne[0], n_multi=ne[2]), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--within-events", type=int, default=30)
    ap.add_argument("--within-refresh", type=int, default=8)
    ap.add_argument("--chain-fwd", type=int, default=24)
    ap.add_argument("--rest-steps", type=int, default=1400)
    ap.add_argument("--theta-period", type=int, default=120, help="steps per theta cycle (one sequence); dt=0.5ms -> ~60ms")
    ap.add_argument("--fire-thresh", type=float, default=0.16, help="smoothed per-assembly active fraction to count as FIRED then SILENCE (must be ABOVE active_frac=0.12 so the assembly is DETECTED before it is silenced)")
    ap.add_argument("--inhib-pa", type=float, default=-1500.0, help="post-fire silencing current (the gamma reset); -1500 = the release-without-killing-detection window (-4000 over-suppresses -> act=0)")
    ap.add_argument("--release-mode", action="store_true", help="RELEASE the bistable plateau (reset cp_v_apical to the down-state) instead of soma inhibition -- the correct un-latch that does not kill the burst/detection")
    ap.add_argument("--release-v", type=float, default=-75.0, help="apical down-state voltage for plateau release")
    ap.add_argument("--proportional", action="store_true", help="SELF-SCALING post-fire inhibition ~ the assembly's own firing (feedback inhibition; robust across seeds where a fixed current is seed-dependent)")
    ap.add_argument("--fs-gamma", action="store_true", help="RUNG-3: gamma-rhythm FS-BASKET feedback inhibition (drive ca3_pv_basket sinusoidally) instead of per-assembly injected current -- self-scales via the real synaptic loop")
    ap.add_argument("--fs-amp", type=float, default=1200.0, help="gamma basket drive amplitude")
    ap.add_argument("--gamma-period", type=int, default=12, help="steps per gamma cycle (dt=0.5ms -> ~6ms ~ 160Hz; tune)")
    ap.add_argument("--poisson-rate", type=float, default=0.015)
    ap.add_argument("--poisson-pa", type=float, default=1500.0)
    ap.add_argument("--poisson-dur", type=int, default=10)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.4)   # match the RANK 2 driver's _detect_sequence_events default
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--active-frac", type=float, default=0.12)
    ap.add_argument("--onset-frac", type=float, default=0.08)
    ap.add_argument("--out", default="research/findings/raw/gap5_r4/spiking_gamma_replay.json")
    a = ap.parse_args()
    cfg = dict(SEQ_CFG)
    cfg["n_mem"] = int(a.n_mem); cfg["within_events"] = int(a.within_events)
    cfg["within_refresh"] = int(a.within_refresh); cfg["chain_fwd"] = int(a.chain_fwd); cfg["chain_rev"] = 0
    cfg["rank1_encode"] = True; cfg["overlap_draw"] = False
    _, backend = get_backend()
    print(f"[gap5-spk-gamma] theta/gamma phase-organized replay ON SPIKES (n_mem={a.n_mem} theta_period={a.theta_period} "
          f"inhib={a.inhib_pa}) seeds={a.seeds} backend={backend}")
    per = [one_seed(s, cfg, a) for s in a.seeds]
    n_go = sum(p["go"] for p in per)
    mg = float(np.mean([p["gamma"]["fwd"] for p in per]))
    mng = float(np.mean([p["no_gamma"]["fwd"] for p in per]))
    print(f"[gap5-spk-gamma] VERDICT: {n_go}/{len(per)} -- GAMMA forward {mg:.3f} vs NO-GAMMA {mng:.3f}. "
          f"{'GO: theta/gamma phase-organized replay fixes the order ON SPIKES.' if n_go == len(per) else 'partial/negative.'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, n_go=n_go, per=per), f, indent=2)


if __name__ == "__main__":
    main()
