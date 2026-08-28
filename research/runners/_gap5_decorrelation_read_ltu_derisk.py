"""gap#5 learn-through-use (2026-08-27) -- test a DECORRELATION READ on the graded weak-cue recall floor.

The reverse-edge-heterosynaptic-depression NO-GO relocated the LTU residual to "the substrate READ-SIDE
noise floor" (4/6 seeds' weak-cue depth insensitive to reverse-edge suppression driven far past full
elimination). Biological precedent (retina/LGN decorrelate their output; Pitkow & Meister 2012; Ruda et al.
2019 -- ignoring correlated activity fails population codes): a decorrelating interneuron layer removes the
shared common-mode so differential structure survives the read.

MECHANISM (additive, default-OFF, byte-identical at lambda=0): during an SWR replay period ALL assemblies
ride a shared sharp-wave common-mode. Before onset detection, subtract (or divide by) the CROSS-ASSEMBLY
common-mode m_t = mean over assemblies of their size-normalised smoothed activity, from each assembly's
trace: a_dec = a - lam*m_t (subtractive lateral inhibition) or a/(sigma+lam*m_t) (divisive normalisation).
This is a READ-only change on the SAME captured firing matrix -- no write, no new plasticity, no sim/ edit.

Does the decorrelated read LIFT the weak-cue recall floor: weak-cue depth_frac AFTER consolidation > BEFORE
with headroom, lesion-attributable (NO-SEED replay null), on >= bar seeds? GO if yes on the decorrelated read
where the plain graded read gave 2/6.

Reuse-by-import ONLY (NO sim/ edit): build_store/encode/rest_and_replay/measure_band/_load_weights/_smooth
from _gap5_ecker_adex_ca3_stdp_band_derisk; consolidate_by_btsp_replay_delayed/measure_band_from from
_gap5_ecker_replay_learn_through_use_derisk; the ESTABLISHED graded config + thresholds from
_gap5_graded_recall_learn_through_use_derisk. SIM_BACKEND=numpy, 500 neurons, CPU, host-RAM-safe.

  Verify byte-identity at lam=0 + smoke (1 seed):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_decorrelation_read_ltu_derisk --seeds 42
  6-seed decisive:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_decorrelation_read_ltu_derisk \
        --seeds 42 43 44 100 101 102 --decorr-lambda 1.0 --decorr-mode sub
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import to_host, get_backend  # noqa: E402
from research.runners._gap5_ecker_adex_ca3_stdp_band_derisk import (  # noqa: E402
    build_store, encode, rest_and_replay, measure_band, _load_weights, _smooth,
)
from research.runners._gap5_ecker_replay_learn_through_use_derisk import (  # noqa: E402
    consolidate_by_btsp_replay_delayed, measure_band_from,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "decorrelation_read_ltu.json"


def _score_periods_graded_decorr(F, assemblies_local, env_seed_log, swr_period, *, W, active_frac, onset_frac,
                                 decorr_lambda=0.0, decorr_mode="sub", decorr_sigma=0.05):
    """IDENTICAL to _gap5_graded_recall_learn_through_use_derisk._score_periods_graded, EXCEPT: within each SWR
    period the per-assembly size-normalised smoothed activity a_kk(t) has the CROSS-ASSEMBLY common-mode
    m(t)=mean_kk a_kk(t) removed before onset/active detection. decorr_lambda=0 -> byte-identical to the plain
    graded read (the common-mode is computed but never applied)."""
    T, _ = F.shape
    n_mem = len(assemblies_local)
    asizes = [max(1, len(a)) for a in assemblies_local]
    n_periods = min(len(env_seed_log), T // swr_period)
    per_asm_active = [0] * n_mem
    depths, depth_fracs, taus = [], [], []
    n_multi = fwd = rev = seed_first = 0
    chance_terms = []
    for n in range(n_periods):
        k = int(env_seed_log[n])
        s0, s1 = n * swr_period, (n + 1) * swr_period
        Fw = F[s0:s1]
        # first pass: all assemblies' size-normalised smoothed activity traces (the population)
        acts = [_smooth(Fw[:, A].sum(1), W) / asizes[kk] for kk, A in enumerate(assemblies_local)]
        if decorr_lambda > 0.0 and acts:
            common = np.mean(np.stack(acts, axis=0), axis=0)                 # m(t): the shared sharp-wave common-mode
            if decorr_mode == "div":
                acts = [a / (decorr_sigma + decorr_lambda * common) for a in acts]
            else:                                                            # subtractive lateral inhibition
                acts = [np.maximum(a - decorr_lambda * common, 0.0) for a in acts]
        active = []
        for kk, a_t in enumerate(acts):
            if a_t.size and float(a_t.max()) >= active_frac:
                per_asm_active[kk] += 1
                cross = np.nonzero(a_t >= onset_frac)[0]
                onset = float(cross[0]) if cross.size else float(np.argmax(a_t))
                active.append((kk, onset + 1e-3 * float(np.argmax(a_t))))
        order = [kk for kk, _ in sorted(active, key=lambda kv: kv[1])]
        depth = 0; expect = k
        for idx in order:
            if idx == expect:
                depth += 1; expect += 1
            else:
                break
        max_possible = max(1, n_mem - k)
        depths.append(depth); depth_fracs.append(depth / max_possible)
        if len(active) >= 2:
            n_multi += 1
            chance_terms.append(1.0 / math.factorial(len(active)))
            if order[0] == k:
                seed_first += 1
            is_fwd = (order[0] == k) and all(order[i + 1] == order[i] + 1 for i in range(len(order) - 1))
            is_rev = (order[0] == k) and all(order[i + 1] == order[i] - 1 for i in range(len(order) - 1))
            if is_fwd:
                fwd += 1
            if is_rev:
                rev += 1
            m = len(order); conc = disc = 0
            for i in range(m):
                for j in range(i + 1, m):
                    if order[i] < order[j]:
                        conc += 1
                    else:
                        disc += 1
            taus.append((conc - disc) / (m * (m - 1) / 2))
    return dict(
        n_events=n_periods, n_multi=n_multi,
        forward_frac=(fwd / n_multi) if n_multi else 0.0, reverse_frac=(rev / n_multi) if n_multi else 0.0,
        depth_mean=float(np.mean(depths)) if depths else 0.0,
        depth_frac_mean=float(np.mean(depth_fracs)) if depth_fracs else 0.0,
        tau_mean=float(np.mean(taus)) if taus else 0.0, n_tau_events=len(taus),
        per_asm_active=per_asm_active, chance_forward=float(np.mean(chance_terms)) if chance_terms else 0.0,
        seed_first_frac=(seed_first / n_multi) if n_multi else 0.0,
    )


def _read_decorr(bkw, seed, w_host, a, *, cue_pa, cue_frac, swr_period, rest_steps, tag):
    s = build_store(seed, **bkw)
    _load_weights(s, w_host)
    r = rest_and_replay(s, rest_steps, seed, swr_period=swr_period, cue_pa=cue_pa,
                        cue_steps=a.cue_steps, cue_frac=cue_frac, seed_on=True)
    sc = _score_periods_graded_decorr(r["F"], s["asm_local"], r["env_seed_log"], swr_period,
                                      W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac,
                                      decorr_lambda=a.decorr_lambda, decorr_mode=a.decorr_mode,
                                      decorr_sigma=a.decorr_sigma)
    return dict(forward=sc["forward_frac"], depth_frac=sc["depth_frac_mean"], depth_mean=sc["depth_mean"],
                tau=sc["tau_mean"], n_multi=sc["n_multi"], n_tau_events=sc["n_tau_events"], tag=tag)


def _bkw(a):
    return dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)


def byte_identity_check(seed, a):
    """At decorr_lambda=0 the decorrelated read MUST equal the plain graded read exactly (the common-mode is
    computed but never applied). Compares depth_frac on a known-good store between lam=0 and the plain scorer."""
    from research.runners._gap5_graded_recall_learn_through_use_derisk import _score_periods_graded
    bkw = _bkw(a)
    st = build_store(seed, **bkw)
    encode(st, seed, n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
           cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    w = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    s = build_store(seed, **bkw); _load_weights(s, w)
    r = rest_and_replay(s, a.rest_steps, seed, swr_period=a.swr_period, cue_pa=a.cue_pa,
                        cue_steps=a.cue_steps, cue_frac=a.cue_frac, seed_on=True)
    plain = _score_periods_graded(r["F"], s["asm_local"], r["env_seed_log"], a.swr_period,
                                  W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    dec0 = _score_periods_graded_decorr(r["F"], s["asm_local"], r["env_seed_log"], a.swr_period,
                                        W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac,
                                        decorr_lambda=0.0)
    ok = (abs(plain["depth_frac_mean"] - dec0["depth_frac_mean"]) < 1e-12
          and abs(plain["tau_mean"] - dec0["tau_mean"]) < 1e-12)
    print(f"[byte-identity lam=0] plain depth_frac={plain['depth_frac_mean']:.6f} dec0={dec0['depth_frac_mean']:.6f} "
          f"-> {'IDENTICAL' if ok else 'DIFFERS'}", flush=True)
    return ok


def one_seed(seed, a):
    t0 = time.time(); out = {"seed": seed}
    bkw = _bkw(a)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    cons_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac, dt=a.dt)
    weak_pa = a.cue_pa * a.weak_cue_mult

    st = build_store(seed, **bkw); encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    out["band_before"] = measure_band(st)

    rd_weak_before = _read_decorr(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=a.swr_period, rest_steps=a.rest_steps, tag="weak_before")

    st_c = build_store(seed, **bkw); _load_weights(st_c, w_learned)
    overlap_kw = dict(W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    cons = consolidate_by_btsp_replay_delayed(st_c, a.consol_steps, seed, seed_on=True,
                                              elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                              eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                              delay_steps=a.fwd_delay_steps, overlap_kw=overlap_kw, **cons_kw)
    w_consol = cons["w_after"]
    out["consolidate"] = dict(dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"])

    rd_weak_after = _read_decorr(bkw, seed, w_consol, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                 swr_period=a.swr_period, rest_steps=a.rest_steps, tag="weak_after")

    st_n = build_store(seed, **bkw); _load_weights(st_n, w_learned)
    cons_ns = consolidate_by_btsp_replay_delayed(st_n, a.consol_steps, seed, seed_on=False,
                                                 elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                                 eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                                 delay_steps=a.fwd_delay_steps, **cons_kw)
    rd_weak_noseed = _read_decorr(bkw, seed, cons_ns["w_after"], a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=a.swr_period, rest_steps=a.rest_steps, tag="weak_noseed")

    dw_fwd, dw_rev, dw_ns = cons["dw_fwd"], cons["dw_rev"], cons_ns["dw_fwd"]
    directional = (dw_fwd - dw_rev) >= a.dw_min
    headroom = rd_weak_before["depth_frac"] <= a.headroom_max
    depth_gain = (rd_weak_after["depth_frac"] - rd_weak_before["depth_frac"]) >= a.depth_gain_min
    tau_gain = (rd_weak_after["tau"] - rd_weak_before["tau"]) >= a.tau_gain_min
    recall_gain = bool(depth_gain or tau_gain)
    lesion_ok = (abs(dw_ns) <= a.noseed_max_frac * max(abs(dw_fwd), 1e-6)
                 and rd_weak_noseed["depth_frac"] <= rd_weak_before["depth_frac"] + a.depth_gain_min)
    seed_go = bool(directional and headroom and recall_gain and lesion_ok)
    out["reads"] = dict(weak_before=rd_weak_before, weak_after=rd_weak_after, weak_noseed=rd_weak_noseed)
    out["checks"] = dict(directional=directional, headroom=headroom, depth_gain=depth_gain, tau_gain=tau_gain,
                         recall_gain=recall_gain, lesion_ok=lesion_ok,
                         weak_depth_before=round(rd_weak_before["depth_frac"], 4),
                         weak_depth_after=round(rd_weak_after["depth_frac"], 4),
                         weak_depth_noseed=round(rd_weak_noseed["depth_frac"], 4),
                         weak_tau_before=round(rd_weak_before["tau"], 4),
                         weak_tau_after=round(rd_weak_after["tau"], 4))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'} depth {rd_weak_before['depth_frac']:.3f}->"
          f"{rd_weak_after['depth_frac']:.3f} (noseed {rd_weak_noseed['depth_frac']:.3f}) tau "
          f"{rd_weak_before['tau']:.3f}->{rd_weak_after['tau']:.3f} dw_fwd={dw_fwd:.1f} dw_rev={dw_rev:.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--decorr-lambda", type=float, default=1.0, help="0 = byte-identical plain read")
    ap.add_argument("--decorr-mode", choices=["sub", "div"], default="sub")
    ap.add_argument("--decorr-sigma", type=float, default=0.05)
    ap.add_argument("--decorr-lambda-scan", type=str, default="", help="comma list; if set, seed[0] scan only")
    ap.add_argument("--skip-byte-check", action="store_true")
    # match the graded-recall NO-GO cfg exactly
    ap.add_argument("--n-mem", type=int, default=6); ap.add_argument("--asm-size", type=int, default=80)
    ap.add_argument("--within-density", type=float, default=0.5)
    ap.add_argument("--rest-steps", type=int, default=9000); ap.add_argument("--consol-steps", type=int, default=6500)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--w-within", type=float, default=60.0); ap.add_argument("--between-init", type=float, default=15.0)
    ap.add_argument("--b-override", type=float, default=120.0); ap.add_argument("--stdp-w-max", type=float, default=900.0)
    ap.add_argument("--stdp-a-plus", type=float, default=0.05); ap.add_argument("--stdp-a-minus", type=float, default=0.06)
    ap.add_argument("--stdp-tau", type=float, default=20.0)
    ap.add_argument("--btsp-elig-tau", type=float, default=80.0); ap.add_argument("--btsp-plat-tau", type=float, default=1.0)
    ap.add_argument("--btsp-eta", type=float, default=0.001); ap.add_argument("--btsp-w-max", type=float, default=900.0)
    ap.add_argument("--fwd-delay-steps", type=int, default=90)
    ap.add_argument("--n-laps", type=int, default=14); ap.add_argument("--enc-step", type=int, default=80)
    ap.add_argument("--enc-dwell", type=int, default=40); ap.add_argument("--enc-gap", type=int, default=600)
    ap.add_argument("--enc-cue-pa", type=float, default=9000.0); ap.add_argument("--enc-cue-frac", type=float, default=0.6)
    ap.add_argument("--swr-period", type=int, default=650); ap.add_argument("--cue-pa", type=float, default=9000.0)
    ap.add_argument("--cue-steps", type=int, default=40); ap.add_argument("--cue-frac", type=float, default=0.6)
    ap.add_argument("--weak-cue-mult", type=float, default=0.5); ap.add_argument("--weak-cue-frac", type=float, default=0.35)
    ap.add_argument("--ou-sigma", type=float, default=40.0)
    ap.add_argument("--window", type=int, default=30); ap.add_argument("--active-frac", type=float, default=0.10)
    ap.add_argument("--onset-frac", type=float, default=0.06)
    ap.add_argument("--dw-min", type=float, default=5.0); ap.add_argument("--headroom-max", type=float, default=0.90)
    ap.add_argument("--depth-gain-min", type=float, default=0.05); ap.add_argument("--tau-gain-min", type=float, default=0.05)
    ap.add_argument("--noseed-max-frac", type=float, default=0.20)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    print(f"[decorr-ltu] decorr_mode={a.decorr_mode} lambda={a.decorr_lambda} sigma={a.decorr_sigma} "
          f"seeds={a.seeds} backend={backend}", flush=True)

    byte_ok = None
    if not a.skip_byte_check:
        byte_ok = byte_identity_check(a.seeds[0], a)

    # optional single-seed lambda scan
    scan = None
    if a.decorr_lambda_scan:
        scan = []
        lams = [float(x) for x in a.decorr_lambda_scan.split(",")]
        for lam in lams:
            a.decorr_lambda = lam
            r = one_seed(a.seeds[0], a)
            scan.append(dict(lam=lam, before=r["reads"]["weak_before"]["depth_frac"],
                             after=r["reads"]["weak_after"]["depth_frac"],
                             gain=r["reads"]["weak_after"]["depth_frac"] - r["reads"]["weak_before"]["depth_frac"],
                             tau_gain=r["reads"]["weak_after"]["tau"] - r["reads"]["weak_before"]["tau"]))
            print(f"  [scan lam={lam}] gain={scan[-1]['gain']:+.3f} tau_gain={scan[-1]['tau_gain']:+.3f}", flush=True)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    go = False; verdict = ""; decided = None
    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        bar = 5 if len(per) >= 6 else max(1, (len(per) + 1) // 2)
        go = n_go >= bar
        mb = float(np.mean([p["reads"]["weak_before"]["depth_frac"] for p in per]))
        ma = float(np.mean([p["reads"]["weak_after"]["depth_frac"] for p in per]))
        mns = float(np.mean([p["reads"]["weak_noseed"]["depth_frac"] for p in per]))
        n_directional = sum(1 for p in per if p["checks"]["directional"])
        n_headroom = sum(1 for p in per if p["checks"]["headroom"])
        n_lesion = sum(1 for p in per if p["checks"]["lesion_ok"])
        verdict = (f"DECORRELATION-READ-LTU {'GO' if go else 'NO-GO'} {n_go}/{len(per)} (mode={a.decorr_mode} "
                   f"lam={a.decorr_lambda}) -- weak-cue depth_frac {mb:.3f}->{ma:.3f} (noseed {mns:.3f}), "
                   f"byte-identical-off={byte_ok}.")
        # PRECONDITIONS are VALIDITY conditions (the test could show a gain); the recall GAIN itself is the DECISION,
        # never a precondition -- a failed outcome under valid preconditions is a real NO-GO, not UNDEFINED.
        v = Verdict("Does a cross-assembly common-mode DECORRELATION read lift the weak-cue recall floor?")
        v.require("byte-identical at lambda=0 (the read change is additive)", bool(byte_ok is None or byte_ok),
                  expect=True)
        v.require("the write is DIRECTIONAL (dw_fwd > dw_rev + dw_min) on >= bar seeds", n_directional,
                  expect=lambda x, b=bar: x >= b)
        v.require("weak-cue depth_frac BEFORE has headroom (not at ceiling) on >= bar seeds", n_headroom,
                  expect=lambda x, b=bar: x >= b)
        v.control("seeded vs NO-SEED weak depth (lesion-the-replay) -- must DIFFER", treatment=ma, control=mns,
                  min_separation=0.0)
        decided = v.decide(go=go, verbose=False)
        attributable_to("decorr-read weak-cue depth gain (seeded vs NO-SEED)", ma - mb, mns - mb)
    else:
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = dict(probe="gap5_decorrelation_read_ltu", decorr_mode=a.decorr_mode, decorr_lambda=a.decorr_lambda,
                   byte_identical_off=byte_ok, seeds=a.seeds, cfg=vars(a), verdict=verdict, GO=go,
                   decided=decided, preconditions=(decided.get("preconditions") if decided else []),
                   scan=scan, per_seed=per, elapsed_seconds=round(time.time() - t0, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[decorr-ltu] {verdict}\n[decorr-ltu] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
