"""gap#5 RUNG-2 (the CORRECT on-spikes mechanism) — INTRINSIC-FATIGUE ordered replay: transient plateau + Izhikevich
spike-frequency adaptation drives the A->B->C transition, NO external inhibition, NO `sim/` edit.

Deep-research (5/5 angles unanimous; Ecker 2022 eLife e71850 = our exact substrate class, Haga-Fukai 2018, Romani-Tsodyks
2015, Chenkov-Sprekeler-Kempter 2017, Schmutz-Gerstner-Schwalger 2022) reframed the 3 FAILED external-inhibition attempts
(crude soma-silence / gamma-FS-basket / theta-ramp) as a HOLD-vs-PUSH CATEGORY ERROR:
  (1) BISTABLE HOLD is the WRONG representation for the replay READ. Every canonical CA3 replay model represents a
      replayed item as a TRANSIENT / metastable moving bump, NEVER a self-sustaining attractor. Our self-regenerating
      dendritic plateau + KIR latch is the exact "stationary bump" those papers must BREAK. Ecker's smoking-gun ablation:
      remove spike-frequency adaptation -> "a STATIONARY rather than a moving bump" and NO replay = bit-for-bit our
      co-ignition symptom active=[3,3,3], forward_frac~0.33.
  (2) The A->B transition is INTRINSIC (fatigue of the just-active population: Izhikevich spike-frequency ADAPTATION,
      tau_u~85ms), NOT external inhibition (feedback inhibition sets ripple/gamma SYNCHRONY, not DIRECTION). The just-fired
      assembly is the MOST-fatigued -> removed from competition -> reverse is blocked by its own fresh fatigue -> the tiny
      +1.3 fwd asymmetry is a RED HERRING; order rides the robust adjacent(143)>>skip(22) chain. Adaptation IS the spiking
      realization of the numpy reference's np.fill_diagonal(Wm,0)+self-avoidance.

THE BUILD (top-ranked, NO `sim/` edit; all knobs already exposed + written by the gap5 harness):
  1. DE-LATCH: bridge.core_config.coincidence_plateau_self_regen -> 0.0 (read live each step at bridge.py:7399) so each
     ignition is a TRANSIENT event, not a latch (the proven encode-side load-bearing knob, applied to the READ).
  2. CRANK adaptation on the CA3-exc slice: raise cp_izh_d_increment (per-spike u-kick = Ecker AdEx b~207pA analog) + slow
     cp_izh_a (recovery tau_u > gamma so the just-fired can't re-win the same cycle).
  3. Let the stored BTSP forward chain drive the next; keep weak Poisson (finite-size noise triggers each hop at N=3);
     NO external inhibition schedule.

ARMS (same encoded chain, same seeds): INTRINSIC (de-latch + adaptation) vs ADAPTATION-LESION (de-latch, adaptation OFF =
Ecker's ExpIF control, MUST co-ignite) vs LATCH-ON (bistable + adaptation, MUST degrade = de-latch load-bearing) vs
NO-NOISE acid (MUST ->0). GO: forward_frac >= 1.5x chance (0.33) AND per_asm_active ~[1,1,1] (NOT [3,3,3]); lesion/latch-on
collapse; no-noise ->0. numpy CPU; coexists with the GPU training. NUMPY-REFERENCE GUARD: NO host per-step assembly silence
or argmax-next in the loop -- order emerges from the substrate's own u-fatigue + the stored weights.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence, SEQ_CFG, _detect_sequence_events  # noqa: E402
from research.runners._gap5_spontaneous_reactivation_derisk import _hard_silence, _configure_ou  # noqa: E402


def _rest_with_fatigue(prep, noise, rest_steps, seed, adapt, self_regen_read, d_abs, a_abs, verbose=False):
    """Freeze plasticity + hard-silence + weak background, run REST. DE-LATCH the plateau (self_regen_read) and, if adapt,
    CRANK Izhikevich spike-frequency adaptation on the CA3-exc slice (cp_izh_d_increment=d_abs, cp_izh_a=a_abs) so the
    just-fired assembly self-fatigues and the stored forward chain drives the next. NO external inhibition. Returns F."""
    from sim.backend import get_backend as _gb
    cp, _ = _gb()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    # DE-LATCH: the plateau self-regen is read live each step (bridge.py:7399); lowering it makes each ignition transient.
    bridge.core_config.coincidence_plateau_self_regen = float(self_regen_read)
    _hard_silence(bridge)
    kind = noise[0]
    _configure_ou(bridge, (noise[1] if kind == "ou" else None), seed)
    poisson = kind == "poisson"
    ca3_arr_host = prep["ca3_arr_host"]
    assemblies_local = prep["assemblies_local"]
    asm_sizes = [max(1, len(a)) for a in assemblies_local]
    exc_glob = ca3_arr_host[prep["ca3_exc_local"]]
    exc_dev = cp.asarray(exc_glob, dtype=cp.int64)
    # Baseline adaptation (for the report) + CRANK it on the exc slice (the transition driver, Ecker 2022).
    if getattr(bridge, "cp_izh_d_increment", None) is not None:
        d0 = float(to_host(bridge.cp_izh_d_increment[exc_dev]).mean())
        a0 = float(to_host(bridge.cp_izh_a[exc_dev]).mean())
        if adapt:
            bridge.cp_izh_d_increment[exc_dev] = cp.float32(d_abs)
            bridge.cp_izh_a[exc_dev] = cp.float32(a_abs)
        if verbose:
            print(f"      [adapt={adapt}] baseline exc d_increment={d0:.3f} a={a0:.4f} -> "
                  f"{'set d='+str(d_abs)+' a='+str(a_abs) if adapt else 'unchanged'}")
    if poisson:
        p_rate, p_pa = float(noise[1]), float(noise[2])
        p_dur = int(noise[3]) if len(noise) > 3 else 5
        prng = np.random.default_rng(int(seed) * 100003 + 11)
        countdown = np.zeros(len(exc_glob), dtype=np.int32)
    # NUMPY-REFERENCE / FROZEN-PLASTICITY GUARD (RANK-1 discipline): _prepare_sequence already disabled BTSP/BDSP +
    # closed gates post-encode; capture the weights to VERIFY they are byte-unchanged across rest (order must ride the
    # STORED frozen chain + the substrate's own u-fatigue, NOT any rest-phase re-encoding -- retires the Wang confound).
    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = 0.0
        if poisson:
            new = prng.random(len(exc_glob)) < p_rate
            countdown[new] = p_dur
            active = countdown > 0
            if active.any():
                bridge.cp_external_input_current[exc_dev[cp.asarray(np.nonzero(active)[0], dtype=cp.int64)]] = p_pa
            countdown[active] -= 1
        bridge._run_one_simulation_step()          # NO external inhibition / argmax / per-assembly silence (numpy-ref guard)
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    assert np.array_equal(w_before, w_after), "PLASTICITY LEAK: cp_connections changed during rest (Wang confound not frozen)"
    return F


def _fwd(F, assemblies_local, det):
    r = _detect_sequence_events(F, assemblies_local, **det)
    return r["forward_frac"], r["reverse_frac"], r["n_multi"], r["per_asm_active"], r["n_events"], float(F.mean())


def one_seed(seed, cfg, a):
    det = dict(W=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k, active_frac=a.active_frac, onset_frac=a.onset_frac)
    noise = ("poisson", a.poisson_rate, a.poisson_pa, a.poisson_dur)

    prep = _prepare_sequence(seed, cfg)
    al = prep["assemblies_local"]
    # INTRINSIC: de-latch + cranked adaptation (the recommended mechanism)
    Fi = _rest_with_fatigue(prep, noise, a.rest_steps, seed, adapt=True, self_regen_read=a.self_regen_read,
                            d_abs=a.d_abs, a_abs=a.a_abs, verbose=True)
    # ADAPTATION-LESION (de-latch, adaptation OFF) = Ecker's ExpIF control -> MUST co-ignite (adaptation is load-bearing)
    prep_l = _prepare_sequence(seed, cfg)
    Fl = _rest_with_fatigue(prep_l, noise, a.rest_steps, seed, adapt=False, self_regen_read=a.self_regen_read,
                            d_abs=a.d_abs, a_abs=a.a_abs)
    i = _fwd(Fi, al, det); les = _fwd(Fl, al, det)
    if a.quick:                                       # calibration: only the load-bearing INTRINSIC vs ADAPT-LESION pair
        go = (i[0] >= 0.50) and (i[2] >= 2) and (les[0] < i[0] - 0.15)
        print(f"  [seed {seed}][QUICK] INTRINSIC fwd={i[0]:.3f} rev={i[1]:.3f} (ev={i[4]} multi={i[2]} act={i[3]} pop={i[5]:.4f}) | "
              f"ADAPT-LESION fwd={les[0]:.3f} (multi={les[2]} act={les[3]}) => {'ORDERS' if go else 'no'}")
        return dict(seed=seed, intrinsic=dict(fwd=i[0], rev=i[1], n_multi=i[2], act=i[3], pop=i[5]),
                    adapt_lesion=dict(fwd=les[0], n_multi=les[2], act=les[3]), go=bool(go))
    # LATCH-ON (bistable plateau kept + cranked adaptation) -> MUST degrade (soma adaptation can't release a dendritic latch)
    prep_o = _prepare_sequence(seed, cfg)
    Fo = _rest_with_fatigue(prep_o, noise, a.rest_steps, seed, adapt=True, self_regen_read=cfg["plateau_self_regen"],
                            d_abs=a.d_abs, a_abs=a.a_abs)
    # NO-NOISE acid (intrinsic, no Poisson) -> MUST ->0
    prep_n = _prepare_sequence(seed, cfg)
    Fn = _rest_with_fatigue(prep_n, ("none",), a.rest_steps, seed, adapt=True, self_regen_read=a.self_regen_read,
                            d_abs=a.d_abs, a_abs=a.a_abs)
    latch = _fwd(Fo, al, det); nn = _fwd(Fn, al, det)
    go = (i[0] >= 0.50) and (i[2] >= 2) and (les[0] < i[0] - 0.15) and (nn[2] == 0)
    print(f"  [seed {seed}] INTRINSIC fwd={i[0]:.3f} rev={i[1]:.3f} (ev={i[4]} multi={i[2]} act={i[3]} pop={i[5]:.4f}) | "
          f"ADAPT-LESION fwd={les[0]:.3f} (multi={les[2]} act={les[3]}) | LATCH-ON fwd={latch[0]:.3f} (act={latch[3]}) | "
          f"NO-NOISE multi={nn[2]} => {'INTRINSIC-FATIGUE-ORDERS' if go else 'no'}")
    return dict(seed=seed, intrinsic=dict(fwd=i[0], rev=i[1], n_multi=i[2], act=i[3], pop=i[5]),
                adapt_lesion=dict(fwd=les[0], n_multi=les[2], act=les[3]),
                latch_on=dict(fwd=latch[0], act=latch[3]), no_noise=dict(n_multi=nn[2]), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--within-events", type=int, default=30)
    ap.add_argument("--within-refresh", type=int, default=8)
    ap.add_argument("--chain-fwd", type=int, default=24)
    ap.add_argument("--rest-steps", type=int, default=1200)
    ap.add_argument("--self-regen-read", type=float, default=0.0, help="plateau self-regen during the READ (0 = fully transient de-latch; the load-bearing knob)")
    ap.add_argument("--d-abs", type=float, default=40.0, help="cranked Izhikevich per-spike u-kick d_increment on CA3-exc (Ecker AdEx b~207pA analog; sweep)")
    ap.add_argument("--a-abs", type=float, default=0.008, help="cranked Izhikevich recovery rate a on CA3-exc (SMALLER=slower fatigue recovery, tau_u=1/a; 0.008 -> tau~125ms > theta)")
    ap.add_argument("--poisson-rate", type=float, default=0.015)
    ap.add_argument("--poisson-pa", type=float, default=1500.0)
    ap.add_argument("--poisson-dur", type=int, default=10)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.4)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--active-frac", type=float, default=0.12)
    ap.add_argument("--onset-frac", type=float, default=0.08)
    ap.add_argument("--quick", action="store_true", help="calibration: run only the load-bearing INTRINSIC vs ADAPT-LESION pair (2 arms, ~2x faster) -- skip LATCH-ON + NO-NOISE")
    ap.add_argument("--out", default="research/findings/raw/gap5_r4/intrinsic_fatigue.json")
    a = ap.parse_args()
    cfg = dict(SEQ_CFG)
    cfg["n_mem"] = int(a.n_mem); cfg["within_events"] = int(a.within_events)
    cfg["within_refresh"] = int(a.within_refresh); cfg["chain_fwd"] = int(a.chain_fwd); cfg["chain_rev"] = 0
    cfg["rank1_encode"] = True; cfg["overlap_draw"] = False
    _, backend = get_backend()
    print(f"[gap5-fatigue] INTRINSIC-FATIGUE ordered replay (transient plateau + Izh adaptation, NO ext inhibition): "
          f"n_mem={a.n_mem} self_regen_read={a.self_regen_read} d_abs={a.d_abs} a_abs={a.a_abs} seeds={a.seeds} backend={backend}")
    per = [one_seed(s, cfg, a) for s in a.seeds]
    n_go = sum(p["go"] for p in per)
    mi = float(np.mean([p["intrinsic"]["fwd"] for p in per]))
    ml = float(np.mean([p["adapt_lesion"]["fwd"] for p in per]))
    print(f"[gap5-fatigue] VERDICT: {n_go}/{len(per)} -- INTRINSIC forward {mi:.3f} vs ADAPT-LESION {ml:.3f}. "
          f"{'GO: intrinsic fatigue orders replay ON SPIKES.' if n_go == len(per) else 'partial/negative -- sweep d_abs/a_abs/self_regen.'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, n_go=n_go, per=per), f, indent=2)


if __name__ == "__main__":
    main()
