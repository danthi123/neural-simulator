"""OPTION (a) — the PHASE-encoding handoff de-risk for the on-substrate spiking whitening (2026-06-06).

THE SETUP (from the 2026-06-06 graded-LGN BOUNDARY). The validated rate/algorithm-level whitening
(`research/findings/2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md`; the regularized local rule
ΔM∝<a aT>-I-lambda*M, λ=0.01, K=300) produces concept codes that COMPOSE the conversational benchmark at 100%
(6/6). Realizing that whitening ON the spiking substrate as a GRADED pre-spike pairwise lateral genuinely
pairwise-decorrelates (coh 0.47->0.187) BUT does NOT compose: it stays at the RAW floor (~66.7%). The DECISIVE
isolation control there localized the cause precisely: drive the KNOWN-100%-composing whitened code INTO the
spiking LGN membrane, read it back through the RATE read-out a = clip((v-v_rest)/scale, 0, 1), and compose ->
it drops 100% -> 72%. The rectifying/saturating RATE read-out destroys the gentle SIGNED whitened structure (the
on-substrate face of the 2026-06-05 opponency wall: a rate code cannot carry a small signed difference faithfully).

THE HYPOTHESIS (option a, this runner). The failure is the RATE read-out SPECIFICALLY. Spike PHASE (timing)
carries signed precision through the threshold where spike RATE cannot — and the FHRR composer already speaks
phase. So read the whitened code out in PHASE instead of RATE and test whether composition is preserved.

THE DECISIVE CHEAP-FIRST ISOLATION (mirrors the boundary's, swapping the read channel RATE->PHASE on a
KNOWN-100%-composing code):
  - KNOWN          : the rate-model's learned whitened code Y (K=300, λ=0.01) fed straight to the composer -> ~100%
                     (sanity: Y itself composes).
  - RATE read-out  : Y -> ON/OFF drive -> graded-LGN Izhikevich membrane settles (M=0, NO lateral) ->
                     a=clip((v-v_rest)/scale,0,1) -> recombine a[:K]-a[K:] -> composer. == the boundary's 72%.
  - PHASE direct   : Y[i,k] -> phase phi=(Y_norm+1)/2 -> phasor exp(i 2pi phi) KICK into a bridge
                     resonate-and-fire neuron -> rf_resonate_steps (the RF "membrane settling") ->
                     rf_read_phases -> invert phi back to Y_read -> composer. The CLEAN read-channel swap (RATE
                     -> the RF magnitude-invariant PHASE channel), no Izhikevich membrane between.
  - PHASE thru-mem : Y -> ON/OFF drive -> graded-LGN Izhikevich membrane settles (M=0, IDENTICAL to RATE) ->
                     read each neuron's spike LATENCY in the window as a PHASE (early spike = high drive = high
                     phase) -> recombine -> composer. The STRICT apples-to-apples: holds the membrane settling
                     IDENTICAL to the RATE control and swaps ONLY the read channel (clip-rate -> latency-phase).

THE DECISIVE COMPARISON: the SAME membrane-degraded code that gave 72% through the RATE read-out — does the
PHASE read-out recover it toward ~100%?

CONTROLS that bracket every run (or the setup is invalid): RAW grounded ~66.7% floor; CONCEPT-whiten ~100% target.
The RATE read-out's 72% is the number PHASE must beat to count as progress; ~100% is resolution.

GUARDS every run (false-positive catchers): the RF phasor codes are alive (the read-back phases are not
silent/degenerate/all-zero); multi-seed (>=3 to conclude, the project's 6-seed bar before a GO); gate on
COMPOSITION, never coherence (coherence misled this arc three separate times). A great composition with a
SILENT/DEGENERATE phase read-out is the false positive to catch — check guards first.

Reuse-by-import; NO sim/ edits. The RF phase substrate already EXISTS (rf_kick/rf_read_phases/rf_resonate_steps,
NeuronModel.RESONATE_AND_FIRE) — phase-encoding a vector is reachable through it. The graded-LGN membrane is the
shipped cp_graded_lateral mechanism (used here with the lateral OFF, M=0, purely to reproduce the membrane the
RATE control degraded through).

  REALOBJ_CIFAR=data/cifar10/cifar-10-batches-py/data_batch_1 \
      python -m research.runners.phase_handoff_decorrelation_compose --seeds 42 43 44 --out out.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.bridge import SimulationBridge
from sim.backend import get_backend, to_host

from research.runners.unified_agent_realobject_grounded import build_realobject_features, run_seed
from research.runners.unified_agent_visual_grounded import _decorrelate
from research.runners.unified_agent_benchmark import build_vocab, aggregate
from research.runners._visual_grounding_probe import _v1_matrix
# reuse the boundary's KNOWN-composing learned whitening + the rate-readout pipeline VERBATIM (apples-to-apples)
from research.runners.graded_lgn_decorrelation_compose import (
    build_graded_lgn_bridge, make_projection, project_drive, read_codes, _recombine, coherence)

import importlib.util as _ilu
import os as _os
_AW = _ilu.spec_from_file_location(
    "_A_whitening_compose_gate",
    _os.path.join(_os.path.dirname(__file__), "..", "findings", "raw", "_A_whitening_compose_gate.py"))
_aw = _ilu.module_from_spec(_AW); _AW.loader.exec_module(_aw)
learned_whiten = _aw.learned_whiten


# ----------------------------------------------------------------------------------------------------
# Composition gate (the SAME run_seed pipeline the whole arc uses: codes -> complex random projection ->
# phase angles -> NestedCompositionAgent -> the full capability benchmark %). The codes fed in are the ONLY
# thing that varies between conditions; the composer is identical. Gate on this %, NEVER coherence.
# ----------------------------------------------------------------------------------------------------
def compose(codes, seeds, tokens, nouns, verbs, adjs):
    d = codes.shape[1]
    seed_res = [run_seed(s, codes, d, tokens, nouns, verbs, adjs, decorrelate=False) for s in seeds]
    _, gok, gtot = aggregate(seed_res)
    return gok, gtot


# ----------------------------------------------------------------------------------------------------
# The PHASE channel: a bridge of resonate-and-fire neurons. Encode a real value y in [-1,1] as a PHASE
# phi(y) = (y+1)/2 in [0,1), kick the phasor exp(i 2pi phi), RESONATE (the RF "membrane settling"), and read the
# phase back (magnitude-invariant). The phase read-out CANNOT saturate the signed structure the way clip-rate
# does — that is the whole hypothesis. We read PER-NEURON (one RF neuron per code dim) so it is a pure read-channel
# swap of the SAME K-dim code the RATE control reads.
# ----------------------------------------------------------------------------------------------------
def _build_rf_bridge(n, seed, period):
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    b.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return b


def _y_to_phase(Y):
    """Per-ROW peak-normalize Y to [-1,1] then map to a phase in [0,1). The whitened code is signed and roughly
    balanced; mapping y -> (y+1)/2 puts 0 at phase 0.5 and the sign in the two phase half-circles (the signed
    structure that the RATE clip destroys lives in the phase here). Per-row scale is benign — the composer's
    complex projection is scale-equivariant up to a global magnitude, and PHASE is magnitude-invariant anyway."""
    s = np.abs(Y).max(axis=1, keepdims=True) + 1e-12
    yn = Y / s                                   # per-row in [-1,1]
    return (yn + 1.0) * 0.5, s                    # phase in [0,1), and the scale to invert


def _phase_to_y(phi, s):
    """Invert _y_to_phase: phase in [0,1) -> y in [-1,1] -> * scale. Phases come back in [0,1) from rf_read_phases."""
    yn = phi * 2.0 - 1.0
    return yn * s


def phase_direct_codes(Y, seed, period, resonate_steps):
    """The CLEAN read-channel swap: encode each whitened scalar Y[i,k] as a phase, kick the phasor into RF neuron k,
    resonate (the RF settling), read the phase back, invert to Y_read. NO Izhikevich membrane between — this
    isolates the READ CHANNEL (RATE clip vs the RF magnitude-invariant PHASE). Returns (Y_read[N,K], guards)."""
    N, K = Y.shape
    phi, scale = _y_to_phase(Y)
    b = _build_rf_bridge(K, seed, period)
    Y_read = np.zeros((N, K))
    phi_read = np.zeros((N, K))
    for i in range(N):
        kick = np.exp(2j * np.pi * phi[i])           # the per-dim phasor (unit magnitude -> info in PHASE)
        b.rf_kick(kick, period=period, lam=0.0)
        b.rf_resonate_steps(period + 8)
        pr = np.asarray(b.rf_read_phases())
        phi_read[i] = pr
        Y_read[i] = _phase_to_y(pr, scale[i, 0])
    # GUARD: the read-back phases must be ALIVE — not all 0 (a neuron that never crosses reads phase 0), not
    # degenerate (all identical). phase_spread = mean over concepts of the std of read phases across the K dims.
    spread = float(np.mean(np.std(phi_read, axis=1)))
    n_zero = int((phi_read == 0.0).sum())
    frac_zero = float(n_zero) / float(N * K)            # fraction of the N*K phase reads that are exactly 0
    # fidelity of the round-trip: how well the recovered phase tracks the input phase (1 = perfect).
    rt = float(np.mean([np.corrcoef(phi[i], phi_read[i])[0, 1] for i in range(N)]))
    guards = {"phase_spread": spread, "n_zero_phase": n_zero, "frac_zero_phase": frac_zero,
              "roundtrip_phase_corr": rt,
              "read_coh_mean": coherence(Y_read)[0], "read_coh_max": coherence(Y_read)[1]}
    return Y_read, guards


def phase_thrumem_codes(Y, seed, K, gain_pA, act_scale, drive_scale, window, settle, proj_seed,
                        period, resonate_steps, signed=True):
    """The STRICT apples-to-apples: drive the KNOWN code Y into the graded-LGN Izhikevich membrane (M=0, lateral OFF
    — IDENTICAL to the RATE control's membrane) but read each neuron's spike LATENCY in the window as a PHASE,
    then route that recovered code through the RF phase channel (so the read-OUT to the composer is phase, not the
    saturating clip). This swaps ONLY the read channel while holding the membrane degradation identical to the RATE
    control. Returns (Y_read[N,K], y_rate[N,K] the rate readout from the SAME pass, guards).

    Latency->phase: a neuron that spikes EARLY in the window was driven HARD (high value); never-spiked = lowest.
    We map first-spike step in [0,window) to a value, sign via the ON/OFF recombination, then re-encode through RF.
    """
    P = make_projection(Y.shape[1], K, proj_seed)
    # Note: the membrane is driven by the PROJECTED code (the rate-model's x = feats @ P), exactly as the RATE
    # control — we project the KNOWN whitened Y the same way the boundary projected it.
    drives = project_drive(Y, P, signed=signed)            # [N, K] or [N, 2K] in [0,1]
    n_it = 2 * K if signed else K
    b = build_graded_lgn_bridge(seed, n_it, lam=0.0, lr=0.0, gain_pA=0.0, act_scale=act_scale)  # M stays 0 (no lateral)
    lgn = np.asarray(b.region_manager.indices("lgn"))

    cp, _ = get_backend()
    a_rate_raw = np.zeros((Y.shape[0], n_it))
    lat_raw = np.zeros((Y.shape[0], n_it))                 # first-spike latency in [0, window]; window = never fired
    for i in range(Y.shape[0]):
        ext = cp.zeros(b.cp_external_input_current.shape[0], dtype=cp.float32)
        ext[cp.asarray(lgn, dtype=cp.int64)] = cp.asarray(drives[i] * drive_scale, dtype=cp.float32)
        b.cp_external_input_current[:] = ext
        for _ in range(settle):
            b._run_one_simulation_step()
        a_acc = np.zeros(n_it)
        first = np.full(n_it, window, dtype=float)          # default: never-fired = window (latest possible)
        for t in range(window):
            b._run_one_simulation_step()
            a_acc += np.asarray(to_host(b._graded_lateral_activity())).astype(float)
            fired = np.asarray(to_host(b.cp_firing_states)).astype(bool)[lgn]
            newly = fired & (first == window)
            first[newly] = t
        a_rate_raw[i] = a_acc / window
        lat_raw[i] = first
        b.cp_external_input_current[:] = 0.0
        for _ in range(settle):
            b._run_one_simulation_step()

    # latency -> phase value in [0,1): early spike (small first) -> high value. window (never fired) -> 0.
    lat_val_raw = 1.0 - lat_raw / float(window)             # in [0,1]; 1 = fired at t=0, 0 = never fired
    # recombine the ON/OFF latency code to a signed K-dim value (the same recombination the RATE readout uses)
    y_lat = np.stack([_recombine(lat_val_raw[i], K, signed) for i in range(Y.shape[0])])
    y_rate = np.stack([_recombine(a_rate_raw[i], K, signed) for i in range(Y.shape[0])])

    # Now route the LATENCY-recovered signed code through the RF phase channel (the read-OUT to the composer is
    # PHASE, the magnitude-invariant channel), exactly mirroring phase_direct but on the membrane-derived code.
    Y_read, ph_guards = phase_direct_codes(y_lat, seed, period, resonate_steps)

    sp_active = (lat_raw < window).sum(1)
    guards = {
        "mean_spike_active": float(sp_active.mean()), "min_spike_active": int(sp_active.min()),
        "n_silent_latency": int((sp_active == 0).sum()),
        "latency_coh_mean": coherence(y_lat)[0], "rate_coh_mean": coherence(y_rate)[0],
        **{f"phase_{k}": v for k, v in ph_guards.items()},
    }
    return Y_read, y_rate, guards


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--K", type=int, default=300, help="whitening subspace / LGN-pool dimension (<= N=320)")
    ap.add_argument("--lam", type=float, default=0.01, help="the rate-model's -lambda*M (KNOWN-100% whitening)")
    ap.add_argument("--gain", type=float, default=0.0, help="graded_lateral_gain_pA — 0 = NO lateral (M=0 pass-through membrane)")
    ap.add_argument("--act-scale", type=float, default=15.0, help="graded_lateral_act_scale (mV->a normalizer; the RATE clip)")
    ap.add_argument("--drive-scale", type=float, default=1400.0, help="pA scale for the projected LGN drive")
    ap.add_argument("--window", type=int, default=40, help="read-window steps (rate accumulation + latency)")
    ap.add_argument("--settle", type=int, default=15, help="settle steps before the read window")
    ap.add_argument("--period", type=int, default=200, help="RF phasor period (one cycle = T resonate steps)")
    ap.add_argument("--resonate-steps", type=int, default=0, help="(unused; period+8 is used internally)")
    ap.add_argument("--skip-thrumem", action="store_true", help="skip the strict through-membrane phase variant (faster)")
    ap.add_argument("--bench-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="research/findings/raw/_phase_handoff_compose.json")
    args = ap.parse_args()
    bench_seeds = args.bench_seeds if args.bench_seeds is not None else args.seeds

    nouns, verbs, adjs = build_vocab()
    W, _ = _v1_matrix()
    feats, dim, tokens, src = build_realobject_features(nouns, verbs, adjs, W, seed=args.seeds[0])
    rawm, rawx = coherence(feats)
    print(f"=== OPTION (a) PHASE-handoff COMPOSITION gate | grounding={src} | {len(tokens)} concepts | K={args.K} "
          f"| lam(-lambdaM)={args.lam} ===", flush=True)
    print(f"  raw feature coherence: mean {rawm:.3f}, max {rawx:.3f}", flush=True)

    out = {"source": src, "K": args.K, "lam": args.lam, "gain": args.gain, "act_scale": args.act_scale,
           "drive_scale": args.drive_scale, "window": args.window, "settle": args.settle, "period": args.period,
           "seeds": args.seeds, "bench_seeds": bench_seeds, "raw_coherence": [rawm, rawx]}

    # ---- CONTROLS (bracket every result; if these are off, the harness is broken) ----
    raw_ok, raw_tot = compose(feats, bench_seeds, tokens, nouns, verbs, adjs)
    out["RAW"] = [raw_ok, raw_tot]
    print(f"  {'RAW grounded (floor control)':<46} {raw_ok}/{raw_tot} = {raw_ok/raw_tot*100:.1f}%   (expect ~66.7%)",
          flush=True)
    cw_ok, cw_tot = compose(_decorrelate(feats), bench_seeds, tokens, nouns, verbs, adjs)
    out["CONCEPT-whiten"] = [cw_ok, cw_tot]
    print(f"  {'CONCEPT-whiten (100% target control)':<46} {cw_ok}/{cw_tot} = {cw_ok/cw_tot*100:.1f}%   (expect ~100%)",
          flush=True)
    harness_ok = (raw_ok / raw_tot < 0.85) and (cw_ok / cw_tot > 0.9)
    print(f"  harness sanity: {'OK (controls bracket as expected)' if harness_ok else 'BROKEN - distrust below'}",
          flush=True)

    # ---- per seed: the KNOWN learned whitening, then RATE vs PHASE read-out of THE SAME code (heavy; SEQUENTIAL) ----
    out["SEEDS"] = {}
    for s in args.seeds:
        # the rate-model's KNOWN-100%-composing whitened code (λ=0.01, K-subspace). Sanity: it composes.
        Y, mratio, blew = learned_whiten(feats, args.K, s, lam=args.lam)
        kn_ok, kn_tot = compose(Y, bench_seeds, tokens, nouns, verbs, adjs)
        print(f"\n  [seed={s}] KNOWN (rate-model learned whitening, lam={args.lam}): "
              f"{kn_ok}/{kn_tot} = {kn_ok/kn_tot*100:.1f}%  (M-ratio {mratio:.2f}, blew={blew})", flush=True)

        # RATE read-out (reproduce the boundary's 72%): Y -> ON/OFF drive -> graded-LGN membrane (M=0) -> clip rate.
        n_it = 2 * args.K
        b = build_graded_lgn_bridge(s, n_it, lam=0.0, lr=0.0, gain_pA=0.0, act_scale=args.act_scale)
        lgn = np.asarray(b.region_manager.indices("lgn"))
        P = make_projection(Y.shape[1], args.K, s)
        drives = project_drive(Y, P, signed=True)
        a_rate_raw = np.zeros((len(feats), n_it))
        for i in range(len(feats)):
            a_rate_raw[i], _ = read_codes(b, lgn, drives[i], args.drive_scale, args.window, args.settle)
        y_rate = np.stack([_recombine(a_rate_raw[i], args.K, True) for i in range(len(feats))])
        rate_ok, rate_tot = compose(y_rate, bench_seeds, tokens, nouns, verbs, adjs)
        rate_coh = coherence(y_rate)[0]
        print(f"            RATE read-out (boundary control): {rate_ok}/{rate_tot} = {rate_ok/rate_tot*100:.1f}%  "
              f"(coh {rate_coh:.3f})  [expect ~72%]", flush=True)

        # PHASE read-out DIRECT (the clean channel swap).
        Y_pd, gpd = phase_direct_codes(Y, s, args.period, args.resonate_steps)
        # DEGENERATE = the phase channel is dead/unfaithful: low round-trip fidelity, collapsed spread, or most
        # reads stuck at phase 0 (neurons never crossing). (A perfect round-trip with spread>0 is ALIVE — the K=80
        # smoke's false alarm came from a wrong 0.5*n_it threshold; frac_zero_phase fixes it.)
        pd_degen = (gpd["phase_spread"] < 1e-3 or gpd["roundtrip_phase_corr"] < 0.9
                    or gpd["frac_zero_phase"] > 0.5)
        pd_ok, pd_tot = compose(Y_pd, bench_seeds, tokens, nouns, verbs, adjs)
        print(f"            PHASE read-out DIRECT (channel swap): {pd_ok}/{pd_tot} = {pd_ok/pd_tot*100:.1f}%  "
              f"(roundtrip_corr {gpd['roundtrip_phase_corr']:.3f}, spread {gpd['phase_spread']:.3f}, "
              f"coh {gpd['read_coh_mean']:.3f}){'  ** DEGENERATE PHASE' if pd_degen else ''}", flush=True)

        entry = {"known_compose": [kn_ok, kn_tot], "m_ratio": mratio, "blew_up": bool(blew),
                 "rate_compose": [rate_ok, rate_tot], "rate_coh": rate_coh,
                 "phase_direct_compose": [pd_ok, pd_tot], "phase_direct_guards": gpd,
                 "phase_direct_degenerate": bool(pd_degen)}

        # PHASE read-out THROUGH the membrane (strict apples-to-apples), unless skipped.
        if not args.skip_thrumem:
            Y_pm, y_rate_pm, gpm = phase_thrumem_codes(
                Y, s, args.K, args.gain, args.act_scale, args.drive_scale, args.window, args.settle,
                proj_seed=s, period=args.period, resonate_steps=args.resonate_steps, signed=True)
            pm_degen = (gpm["n_silent_latency"] > 0 or gpm["phase_roundtrip_phase_corr"] < 0.5)
            pm_ok, pm_tot = compose(Y_pm, bench_seeds, tokens, nouns, verbs, adjs)
            print(f"            PHASE read-out THRU-MEMBRANE (strict): {pm_ok}/{pm_tot} = {pm_ok/pm_tot*100:.1f}%  "
                  f"(latency_coh {gpm['latency_coh_mean']:.3f}, n_silent {gpm['n_silent_latency']}/{len(feats)})"
                  f"{'  ** DEGENERATE' if pm_degen else ''}", flush=True)
            entry["phase_thrumem_compose"] = [pm_ok, pm_tot]
            entry["phase_thrumem_guards"] = gpm
            entry["phase_thrumem_degenerate"] = bool(pm_degen)

        out["SEEDS"][str(s)] = entry

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)

    # ---- VERDICT (gate on the PHASE DIRECT read-out — the clean channel swap; thru-mem as the strict corroborator) ----
    raw_rate = raw_ok / raw_tot
    cw_rate = cw_ok / cw_tot
    kn_rates = [v["known_compose"][0] / v["known_compose"][1] for v in out["SEEDS"].values()]
    rt_rates = [v["rate_compose"][0] / v["rate_compose"][1] for v in out["SEEDS"].values()]
    pd_rates = [v["phase_direct_compose"][0] / v["phase_direct_compose"][1] for v in out["SEEDS"].values()]
    pd_degen = any(v["phase_direct_degenerate"] for v in out["SEEDS"].values())
    pm_rates = [v["phase_thrumem_compose"][0] / v["phase_thrumem_compose"][1]
                for v in out["SEEDS"].values() if "phase_thrumem_compose" in v]
    print("\n" + "=" * 88, flush=True)
    print(f"VERDICT (OPTION a — PHASE-handoff, {len(args.seeds)} seed(s)):", flush=True)
    print(f"  RAW floor {raw_rate*100:.1f}%  |  CONCEPT-whiten target {cw_rate*100:.1f}%  |  rate-model KNOWN "
          f"{np.mean(kn_rates)*100:.1f}%", flush=True)
    print(f"  RATE read-out (the boundary's 72%): {['%.1f%%' % (r*100) for r in rt_rates]}  "
          f"(mean {np.mean(rt_rates)*100:.1f}%)  <- the number PHASE must beat", flush=True)
    print(f"  PHASE DIRECT read-out: {['%.1f%%' % (r*100) for r in pd_rates]}  (mean {np.mean(pd_rates)*100:.1f}%)",
          flush=True)
    if pm_rates:
        print(f"  PHASE THRU-MEMBRANE (strict): {['%.1f%%' % (r*100) for r in pm_rates]}  "
              f"(mean {np.mean(pm_rates)*100:.1f}%)", flush=True)
    if not harness_ok:
        verdict = "INVALID - controls did not bracket (harness broken)"
    elif pd_degen:
        verdict = "FALSE-POSITIVE RISK - a seed had a degenerate/silent PHASE read-out; see guards"
    elif np.mean(pd_rates) >= 0.95 and min(pd_rates) >= 0.90:
        verdict = ("GO - the PHASE read-out RECOVERS the known composing code (>=95% mean, >=90% min); "
                   "the boundary WAS the rate read-out")
    elif np.mean(pd_rates) > np.mean(rt_rates) + 0.1:
        verdict = (f"PARTIAL - PHASE beats the RATE read-out (phase {np.mean(pd_rates)*100:.1f}% vs rate "
                   f"{np.mean(rt_rates)*100:.1f}%) but below the 100% target")
    else:
        verdict = (f"BOUNDARY - PHASE does NOT beat the RATE read-out (phase {np.mean(pd_rates)*100:.1f}% vs rate "
                   f"{np.mean(rt_rates)*100:.1f}%); phase does not recover the gentle whitened structure")
    print(f"  => {verdict}", flush=True)
    print("=" * 88, flush=True)


if __name__ == "__main__":
    main()
