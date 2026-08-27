"""gap#5 learn-through-use, GRADED-RECALL-INSTRUMENT variant (2026-08-27): does the already-established
DIRECTIONAL write (BTSP-eligibility + forward-edge conduction delay, 6/6 GO on
[[2026-08-27-conduction-delay-directional-replay-learn-through-use-PARTIAL]]) produce a recall GAIN once the
recall READ is fixed?

CONTEXT -- the wall this runner surpasses is the INSTRUMENT, not the write:
  * conduction-delay PARTIAL: a 9ms forward-edge axonal delay separates the SWR volleys (overlap 0.58->0.28)
    and flips the BTSP write NET-DIRECTIONAL 6/6 (dw_fwd > dw_rev every seed). But the learn-through-use GO bar
    stayed at 1/6 because weak-cue forward_frac read ~1.0 BEFORE consolidation -- no headroom to show a gain.
  * Braun-2022 gap-coding NO-GO: DECISIVELY diagnosed WHY -- `forward_frac` (forward-ordered events /
    multi-assembly events) is BINARY-ORDER: "it only drops on ORDER ERRORS or reverse intrusions; a truncated
    forward run (A->B->C then stop) still scores forward." A clean band makes almost no order errors, so
    forward_frac reads ~1.0 at EVERY non-degenerate op-point regardless of how well the volleys separate or how
    DEEP the completion runs. The residual the PARTIAL called "recall at ceiling" is the INSTRUMENT, not a
    genuine substrate ceiling.

THIS RUNNER: builds a GRADED recall instrument with real dynamic range (`_score_periods_graded`, candidates
(a)+(b) from the task brief) and VERIFIES it is graded on a known-good store (`verify_instrument`, a cue-strength
sweep) BEFORE spending it on the decisive question, then re-runs the DECISIVE learn-through-use test with the
SAME established directional write, reading recall with the new instrument instead of the old one.

  DEPTH (candidate a, SEQUENCE-COMPLETION DEPTH): per SWR event seeded by assembly k, the length of the
    unbroken PREFIX match between the observed onset order and the canonical chain k,k+1,k+2,... -- a count
    0..(n_mem-k), not a 0/1. A truncated run now reads LESS than a full completion.
  TAU (candidate b, cross-check / trajectory fidelity): a Kendall-tau-style pairwise concordance over the
    ACTIVATED subset (+1 exact forward order, -1 exact reverse, 0 = chance ordering). Catches a different
    failure mode than DEPTH (a locally-scrambled-but-globally-forward trajectory).
  Read-period DECOUPLING (candidate c) is available via --read-swr-period (defaults to --swr-period, i.e. the
  SAME long separation regime the write needs) -- the decisive run uses the SAME regime as the write by default
  so any instrument-driven gain cannot be attributed to an easier read op-point; --read-swr-period lets a
  shorter regime be swept separately if the default regime under-samples events.

Reuse-by-import (NO sim/ edit, NO new mechanism -- only a new READOUT on the same captured firing matrix):
  build_store / encode / rest_and_replay / measure_band / _load_weights / _smooth (transitively) from
  _gap5_ecker_adex_ca3_stdp_band_derisk; consolidate_by_btsp_replay_delayed / measure_band_from from
  _gap5_ecker_replay_learn_through_use_derisk (the ESTABLISHED 6/6-directional write, byte-identical reuse).

  Verify:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_graded_recall_learn_through_use_derisk \
               --seeds 42 --verify-only
  6-seed:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_graded_recall_learn_through_use_derisk \
               --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
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

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "graded_recall_learn_through_use.json"
VERIFY_OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "graded_recall_instrument_verify.json"


# ----------------------------------------------------------------------------------------------------------------------
# THE GRADED INSTRUMENT (2026-08-27). See module docstring for DEPTH / TAU definitions. Reuses the identical
# smoother/thresholds/onset detection as `_score_periods` (one instrument, same detection stage) -- ONLY the
# aggregation from "activated subset in onset order" -> a scalar changes.
# ----------------------------------------------------------------------------------------------------------------------
def _score_periods_graded(F, assemblies_local, env_seed_log, swr_period, *, W, active_frac, onset_frac):
    T, _ = F.shape
    n_mem = len(assemblies_local)
    asizes = [max(1, len(a)) for a in assemblies_local]
    n_periods = min(len(env_seed_log), T // swr_period)
    per_asm_active = [0] * n_mem
    depths = []
    depth_fracs = []
    taus = []
    n_multi = fwd = rev = seed_first = 0
    chance_terms = []
    for n in range(n_periods):
        k = int(env_seed_log[n])
        s0, s1 = n * swr_period, (n + 1) * swr_period
        Fw = F[s0:s1]
        active = []
        for kk, A in enumerate(assemblies_local):
            a_t = _smooth(Fw[:, A].sum(1), W) / asizes[kk]
            if a_t.size and float(a_t.max()) >= active_frac:
                per_asm_active[kk] += 1
                cross = np.nonzero(a_t >= onset_frac)[0]
                onset = float(cross[0]) if cross.size else float(np.argmax(a_t))
                active.append((kk, onset + 1e-3 * float(np.argmax(a_t))))
        order = [kk for kk, _ in sorted(active, key=lambda kv: kv[1])]
        # DEPTH: run-on prefix match from the seed k, over EVERY period (a failure-to-ignite -> depth 0). This
        # is the metric that gives a truncated A->B->C (then stop) run LESS credit than a full completion --
        # the exact axis forward_frac cannot see (it only checks internal CONSISTENCY of whatever fired).
        depth = 0
        expect = k
        for idx in order:
            if idx == expect:
                depth += 1
                expect += 1
            else:
                break
        max_possible = max(1, n_mem - k)
        depths.append(depth)
        depth_fracs.append(depth / max_possible)
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
            # TAU: pairwise concordance over the activated subset (independent of WHERE the run starts/derails).
            m = len(order)
            conc = disc = 0
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


def _read_graded(bkw, seed, w_host, a, *, cue_pa, cue_frac, swr_period, rest_steps, tag):
    """Fresh store, load weights, replay READ (frozen), score with the GRADED instrument."""
    s = build_store(seed, **bkw)
    _load_weights(s, w_host)
    r = rest_and_replay(s, rest_steps, seed, swr_period=swr_period, cue_pa=cue_pa,
                        cue_steps=a.cue_steps, cue_frac=cue_frac, seed_on=True)
    sc = _score_periods_graded(r["F"], s["asm_local"], r["env_seed_log"], swr_period,
                               W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    band = measure_band(s)
    return dict(forward=sc["forward_frac"], reverse=sc["reverse_frac"], chance=max(sc["chance_forward"], 1e-6),
                n_multi=sc["n_multi"], depth_mean=sc["depth_mean"], depth_frac=sc["depth_frac_mean"],
                tau=sc["tau_mean"], n_tau_events=sc["n_tau_events"], per_asm_active=sc["per_asm_active"],
                frozen=r["weights_frozen"], band=band, tag=tag)


# ----------------------------------------------------------------------------------------------------------------------
# INSTRUMENT VALIDATION (task requirement: "verify the instrument first"). Encode ONE known-good store, sweep cue
# STRENGTH at the DECISIVE read regime (same swr_period the write needs -- no cherry-picked easy op-point), and
# confirm depth_frac/tau read INTERMEDIATE values while legacy forward_frac stays pinned near ceiling.
# ----------------------------------------------------------------------------------------------------------------------
def verify_instrument(seed, a):
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    st = build_store(seed, **bkw)
    encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    band = measure_band(st)
    print(f"[verify-instrument] seed={seed} band adj_fwd={band['adj_fwd']:.1f} adj_rev={band['adj_rev']:.1f} "
          f"(known-good STDP-encoded store)", flush=True)
    rows = []
    for mult in a.verify_cue_mults:
        rd = _read_graded(bkw, seed, w_learned, a, cue_pa=a.cue_pa * mult, cue_frac=a.cue_frac,
                          swr_period=a.swr_period, rest_steps=a.rest_steps, tag=f"mult{mult}")
        rows.append(dict(cue_mult=mult, forward=rd["forward"], reverse=rd["reverse"], chance=rd["chance"],
                         n_multi=rd["n_multi"], depth_mean=rd["depth_mean"], depth_frac=rd["depth_frac"],
                         tau=rd["tau"], n_tau_events=rd["n_tau_events"]))
        print(f"  [verify] cue_mult={mult:.2f}: LEGACY forward_frac={rd['forward']:.3f} (chance {rd['chance']:.3f}, "
              f"n_multi={rd['n_multi']}) | GRADED depth_frac={rd['depth_frac']:.3f} depth_mean={rd['depth_mean']:.2f} "
              f"tau={rd['tau']:.3f} (n_tau_events={rd['n_tau_events']})", flush=True)
    fwd_vals = [r["forward"] for r in rows]
    df_vals = [r["depth_frac"] for r in rows]
    tau_vals = [r["tau"] for r in rows]
    df_range = max(df_vals) - min(df_vals)
    tau_range = max(tau_vals) - min(tau_vals)
    fwd_range = max(fwd_vals) - min(fwd_vals)
    n_intermediate = sum(1 for v in df_vals if 0.05 < v < 0.95)
    graded = bool(df_range >= a.verify_min_range and n_intermediate >= 2)
    legacy_flat = bool(fwd_range <= 0.20)
    print(f"[verify-instrument] depth_frac range={df_range:.3f} (min={min(df_vals):.3f} max={max(df_vals):.3f}, "
          f"{n_intermediate}/{len(df_vals)} levels intermediate) | tau range={tau_range:.3f} | LEGACY "
          f"forward_frac range={fwd_range:.3f} (min={min(fwd_vals):.3f} max={max(fwd_vals):.3f})", flush=True)
    print(f"[verify-instrument] => INSTRUMENT IS GRADED: {graded}  (legacy stayed near-flat/ceiling: {legacy_flat})",
          flush=True)
    return dict(seed=seed, band=band, rows=rows, graded=graded, legacy_flat=legacy_flat,
               depth_frac_range=df_range, tau_range=tau_range, forward_frac_range=fwd_range,
               n_intermediate=n_intermediate)


# ----------------------------------------------------------------------------------------------------------------------
# THE DECISIVE PER-SEED TEST: identical BUILD/ENCODE/CONSOLIDATE(seeded+lesion) to the established
# conduction-delay PARTIAL (SAME write function, SAME hyperparameters by default) -- ONLY the recall READ
# instrument changes (graded depth/tau instead of binary forward_frac).
# ----------------------------------------------------------------------------------------------------------------------
def one_seed(seed, a):
    t0 = time.time()
    out = {"seed": seed}
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    cons_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac, dt=a.dt)
    weak_pa = a.cue_pa * a.weak_cue_mult
    read_period = a.read_swr_period if a.read_swr_period > 0 else a.swr_period

    # 1. BUILD + ENCODE the memory (moderate band; headroom below stdp_w_max) -- identical to the PARTIAL.
    st = build_store(seed, **bkw)
    encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    band_before = measure_band(st)
    out["band_before"] = band_before
    print(f"  [seed {seed}] ENCODE: band fwd={band_before['adj_fwd']:.1f} rev={band_before['adj_rev']:.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 2. READ BEFORE (GRADED, full + weak cue) at the DECISIVE read regime
    rd_full_before = _read_graded(bkw, seed, w_learned, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="full_before")
    rd_weak_before = _read_graded(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="weak_before")
    print(f"  [seed {seed}] BEFORE: full depth_frac={rd_full_before['depth_frac']:.3f} "
          f"(legacy fwd={rd_full_before['forward']:.3f}) | weak depth_frac={rd_weak_before['depth_frac']:.3f} "
          f"tau={rd_weak_before['tau']:.3f} (legacy fwd={rd_weak_before['forward']:.3f}) n_multi="
          f"{rd_weak_before['n_multi']} ({time.time()-t0:.0f}s)", flush=True)

    # 3. CONSOLIDATE-BY-REPLAY (seeded): the ESTABLISHED directional write (BTSP-eligibility + forward
    #    conduction delay -- the SAME function, SAME hyperparameters as the 6/6-directional PARTIAL).
    overlap_kw = dict(W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    st_c = build_store(seed, **bkw)
    _load_weights(st_c, w_learned)
    cons = consolidate_by_btsp_replay_delayed(st_c, a.consol_steps, seed, seed_on=True,
                                              elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                              eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                              delay_steps=a.fwd_delay_steps, overlap_kw=overlap_kw, **cons_kw)
    w_consol = cons["w_after"]
    out["consolidate"] = dict(dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"],
                              volley_overlap=cons.get("volley_overlap"), changed=cons["changed"])
    print(f"  [seed {seed}] CONSOLIDATE(seeded): dw_fwd={cons['dw_fwd']:.2f} dw_rev={cons['dw_rev']:.2f} "
          f"volley_overlap={cons.get('volley_overlap')} ({time.time()-t0:.0f}s)", flush=True)

    # 4. READ AFTER (GRADED, full + weak cue)
    rd_full_after = _read_graded(bkw, seed, w_consol, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac,
                                 swr_period=read_period, rest_steps=a.rest_steps, tag="full_after")
    rd_weak_after = _read_graded(bkw, seed, w_consol, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                 swr_period=read_period, rest_steps=a.rest_steps, tag="weak_after")
    band_after = measure_band_from(w_consol, st_c)
    out["band_after"] = band_after
    print(f"  [seed {seed}] AFTER: full depth_frac={rd_full_after['depth_frac']:.3f} | weak depth_frac="
          f"{rd_weak_after['depth_frac']:.3f} tau={rd_weak_after['tau']:.3f} (legacy fwd={rd_weak_after['forward']:.3f}) "
          f"band fwd={band_after['adj_fwd']:.1f} rev={band_after['adj_rev']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    # 5. LESION-THE-REPLAY: NO-SEED consolidation (identical write path, seed_on=False -> no ignition -> null)
    st_n = build_store(seed, **bkw)
    _load_weights(st_n, w_learned)
    cons_ns = consolidate_by_btsp_replay_delayed(st_n, a.consol_steps, seed, seed_on=False,
                                                 elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                                 eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                                 delay_steps=a.fwd_delay_steps, **cons_kw)
    w_noseed = cons_ns["w_after"]
    rd_weak_noseed = _read_graded(bkw, seed, w_noseed, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="weak_noseed")
    out["no_seed"] = dict(dw_fwd=cons_ns["dw_fwd"], dw_rev=cons_ns["dw_rev"],
                          weak_depth_frac=rd_weak_noseed["depth_frac"], weak_tau=rd_weak_noseed["tau"],
                          weak_forward_legacy=rd_weak_noseed["forward"])
    print(f"  [seed {seed}] NO-SEED(lesion-replay): dw_fwd={cons_ns['dw_fwd']:.3f} weak depth_frac="
          f"{rd_weak_noseed['depth_frac']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    out["reads"] = dict(full_before=rd_full_before, weak_before=rd_weak_before,
                        full_after=rd_full_after, weak_after=rd_weak_after)

    # ============ PER-SEED VERDICT (GRADED-INSTRUMENT GO bar; verify, don't assert) ============
    dw_fwd = cons["dw_fwd"]; dw_rev = cons["dw_rev"]; dw_ns = cons_ns["dw_fwd"]
    directional = ((dw_fwd - dw_rev) >= a.dw_min)
    headroom = (rd_weak_before["depth_frac"] <= a.headroom_max)
    depth_gain = ((rd_weak_after["depth_frac"] - rd_weak_before["depth_frac"]) >= a.depth_gain_min)
    tau_gain = ((rd_weak_after["tau"] - rd_weak_before["tau"]) >= a.tau_gain_min)
    recall_gain = bool(depth_gain or tau_gain)
    lesion_controlled = (abs(dw_ns) <= a.noseed_max_frac * max(abs(dw_fwd), 1e-6)
                         and (rd_weak_noseed["depth_frac"] <= rd_weak_before["depth_frac"] + a.depth_gain_min))
    seed_go = bool(directional and headroom and recall_gain and lesion_controlled)
    out["checks"] = dict(directional=directional, headroom=headroom, depth_gain=depth_gain, tau_gain=tau_gain,
                         recall_gain=recall_gain, lesion_controlled=lesion_controlled,
                         dw_fwd=round(dw_fwd, 3), dw_rev=round(dw_rev, 3), dw_noseed=round(dw_ns, 3),
                         weak_depth_frac_before=round(rd_weak_before["depth_frac"], 3),
                         weak_depth_frac_after=round(rd_weak_after["depth_frac"], 3),
                         weak_tau_before=round(rd_weak_before["tau"], 3),
                         weak_tau_after=round(rd_weak_after["tau"], 3),
                         weak_depth_frac_noseed=round(rd_weak_noseed["depth_frac"], 3))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=6)
    ap.add_argument("--asm-size", type=int, default=80)
    ap.add_argument("--within-density", type=float, default=0.5)
    ap.add_argument("--rest-steps", type=int, default=9000)
    ap.add_argument("--consol-steps", type=int, default=6500)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--w-within", type=float, default=60.0)
    ap.add_argument("--between-init", type=float, default=15.0)
    ap.add_argument("--b-override", type=float, default=120.0)
    ap.add_argument("--stdp-w-max", type=float, default=900.0)
    ap.add_argument("--stdp-a-plus", type=float, default=0.05)
    ap.add_argument("--stdp-a-minus", type=float, default=0.06)
    ap.add_argument("--stdp-tau", type=float, default=20.0)
    # the ESTABLISHED directional write (matches the conduction-delay PARTIAL's decisive 6-seed cfg exactly)
    ap.add_argument("--btsp-elig-tau", type=float, default=80.0)
    ap.add_argument("--btsp-plat-tau", type=float, default=1.0)
    ap.add_argument("--btsp-eta", type=float, default=0.001)
    ap.add_argument("--btsp-w-max", type=float, default=900.0)
    ap.add_argument("--fwd-delay-steps", type=int, default=90)
    # ENCODE
    ap.add_argument("--n-laps", type=int, default=14)
    ap.add_argument("--enc-step", type=int, default=80)
    ap.add_argument("--enc-dwell", type=int, default=40)
    ap.add_argument("--enc-gap", type=int, default=600)
    ap.add_argument("--enc-cue-pa", type=float, default=9000.0)
    ap.add_argument("--enc-cue-frac", type=float, default=0.6)
    # SWR replay / prefix seed (write side)
    ap.add_argument("--swr-period", type=int, default=650)
    ap.add_argument("--cue-pa", type=float, default=9000.0)
    ap.add_argument("--cue-steps", type=int, default=40)
    ap.add_argument("--cue-frac", type=float, default=0.6)
    ap.add_argument("--weak-cue-mult", type=float, default=0.5)
    ap.add_argument("--weak-cue-frac", type=float, default=0.35)
    ap.add_argument("--ou-sigma", type=float, default=40.0)
    # READ decoupling (candidate c): 0 = use --swr-period (decisive regime, default); >0 = a separate read period
    ap.add_argument("--read-swr-period", type=int, default=0)
    # detection
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--active-frac", type=float, default=0.10)
    ap.add_argument("--onset-frac", type=float, default=0.06)
    # GO thresholds (GRADED-instrument bar)
    ap.add_argument("--dw-min", type=float, default=5.0)
    ap.add_argument("--headroom-max", type=float, default=0.90, help="weak-cue depth_frac BEFORE must be <= this "
                    "(NOT at ceiling) for the headroom precondition to hold")
    ap.add_argument("--depth-gain-min", type=float, default=0.05, help="min weak-cue depth_frac gain (after-before)")
    ap.add_argument("--tau-gain-min", type=float, default=0.05, help="min weak-cue tau gain (after-before)")
    ap.add_argument("--noseed-max-frac", type=float, default=0.20)
    # instrument verification
    ap.add_argument("--verify-only", action="store_true", help="run ONLY the instrument-validation sweep, skip "
                    "the decisive 6-seed test")
    ap.add_argument("--skip-verify", action="store_true", help="skip the instrument-validation sweep (decisive "
                    "test only; use only if verify_instrument was already run+recorded this session)")
    ap.add_argument("--verify-cue-mults", type=float, nargs="+", default=[1.0, 0.85, 0.7, 0.5, 0.35, 0.2])
    ap.add_argument("--verify-min-range", type=float, default=0.15, help="min depth_frac range across the cue-"
                    "strength sweep required to call the instrument GRADED")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify-out", default=str(VERIFY_OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    print(f"[graded-ltu] Ecker AdEx CA3 GRADED-RECALL learn-through-use | write=btsp+delay elig_tau={a.btsp_elig_tau} "
          f"plat_tau={a.btsp_plat_tau} eta={a.btsp_eta} fwd_delay={a.fwd_delay_steps}steps({a.fwd_delay_steps*a.dt:.1f}"
          f"ms) | n_mem={a.n_mem} asm={a.asm_size} | encode {a.n_laps}laps | swr={a.swr_period} read_swr="
          f"{a.read_swr_period or a.swr_period} cue={a.cue_pa}@{a.cue_frac} weak={a.cue_pa*a.weak_cue_mult}@"
          f"{a.weak_cue_frac} | rest={a.rest_steps} consol={a.consol_steps} dt={a.dt} seeds={a.seeds} "
          f"backend={backend}", flush=True)

    verify = None
    if not a.skip_verify:
        verify = verify_instrument(a.seeds[0], a)
        Path(a.verify_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.verify_out).write_text(json.dumps(dict(seeds=a.seeds, cfg=vars(a), **verify), indent=2, default=str))
        print(f"[graded-ltu] wrote {a.verify_out}", flush=True)
        if not verify["graded"]:
            print("[graded-ltu] ⛔ INSTRUMENT VALIDATION FAILED -- depth_frac did not show real dynamic range on "
                  "the known-good store. Refusing to spend it on the decisive test (an instrument you cannot show "
                  "is graded is not graded).", flush=True)
            return 1
    if a.verify_only:
        return 0 if (verify is None or verify["graded"]) else 1

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_go = sum(1 for p in per if p.get("seed_go"))
        bar = max(1, (len(per) + 1) // 2) if len(per) < 6 else 5
        go = n_go >= bar
        mdwf = float(np.mean([p["consolidate"]["dw_fwd"] for p in per]))
        mdwr = float(np.mean([p["consolidate"]["dw_rev"] for p in per]))
        mdwns = float(np.mean([p["no_seed"]["dw_fwd"] for p in per]))
        mwdf_b = float(np.mean([p["reads"]["weak_before"]["depth_frac"] for p in per]))
        mwdf_a = float(np.mean([p["reads"]["weak_after"]["depth_frac"] for p in per]))
        mwdf_ns = float(np.mean([p["no_seed"]["weak_depth_frac"] for p in per]))
        mwtau_b = float(np.mean([p["reads"]["weak_before"]["tau"] for p in per]))
        mwtau_a = float(np.mean([p["reads"]["weak_after"]["tau"] for p in per]))
        mfull_dfb = float(np.mean([p["reads"]["full_before"]["depth_frac"] for p in per]))
        mfull_dfa = float(np.mean([p["reads"]["full_after"]["depth_frac"] for p in per]))
        mwfwd_legacy_b = float(np.mean([p["reads"]["weak_before"]["forward"] for p in per]))
        mwfwd_legacy_a = float(np.mean([p["reads"]["weak_after"]["forward"] for p in per]))
        n_headroom = sum(1 for p in per if p["checks"]["headroom"])
        n_directional = sum(1 for p in per if p["checks"]["directional"])
        n_lesion_ok = sum(1 for p in per if p["checks"]["lesion_controlled"])
        if go:
            verdict = (f"GRADED-RECALL-INSTRUMENT GO {n_go}/{len(per)} -- the ESTABLISHED directional write "
                       f"(dw_fwd {mdwf:.1f} vs dw_rev {mdwr:.1f}, {n_directional}/{len(per)} directional) now "
                       f"produces a MEASURABLE weak-cue recall GAIN once the read is graded: depth_frac "
                       f"{mwdf_b:.3f}->{mwdf_a:.3f} (tau {mwtau_b:.3f}->{mwtau_a:.3f}), headroom held "
                       f"{n_headroom}/{len(per)} (before <= {a.headroom_max}), LESION-THE-REPLAY null "
                       f"{n_lesion_ok}/{len(per)} (dw_fwd_noseed {mdwns:.2f}~0, weak depth_frac_noseed "
                       f"{mwdf_ns:.3f}~before). Legacy forward_frac stayed near-ceiling throughout "
                       f"({mwfwd_legacy_b:.3f}->{mwfwd_legacy_a:.3f}) -- the PARTIAL's 'no headroom' was the OLD "
                       f"metric, not the substrate. => converts the conduction-delay PARTIAL to GO with NO new "
                       f"store.")
        else:
            verdict = (f"GRADED-RECALL-INSTRUMENT NO-GO {n_go}/{len(per)} -- the write stays directional "
                       f"({n_directional}/{len(per)}, dw_fwd {mdwf:.1f} vs dw_rev {mdwr:.1f}) and headroom holds "
                       f"({n_headroom}/{len(per)} weak-cue depth_frac_before <= {a.headroom_max}: {mwdf_b:.3f}), "
                       f"but even the GRADED instrument does not show a durable weak-cue recall gain: depth_frac "
                       f"{mwdf_b:.3f}->{mwdf_a:.3f} (tau {mwtau_b:.3f}->{mwtau_a:.3f}), lesion-null "
                       f"{n_lesion_ok}/{len(per)}. Legacy forward_frac {mwfwd_legacy_b:.3f}->{mwfwd_legacy_a:.3f}. "
                       f"=> this is a GENUINE negative (not the metric artifact): the directional write does not "
                       f"durably deepen weak-cue recall on this substrate even with headroom available.")
        v = Verdict("Ecker AdEx CA3: does the ESTABLISHED directional (BTSP+delay) replay-driven write produce a "
                    "GRADED weak-cue recall GAIN (headroom + lesion-controlled)?")
        v.require("the GRADED instrument itself reads graded on a known-good store (pre-flight verify_instrument)",
                  bool(verify is None or verify["graded"]), expect=True)
        v.require("weak-cue depth_frac BEFORE has headroom (not at ceiling) on >= bar seeds", n_headroom,
                  expect=lambda x, b=bar: x >= b)
        v.require("the write is DIRECTIONAL (dw_fwd > dw_rev + dw_min) on >= bar seeds", n_directional,
                  expect=lambda x, b=bar: x >= b)
        v.control("LESION-THE-REPLAY: seeded forward-deepening vs NO-SEED forward-deepening -- must DIFFER",
                  treatment=mdwf, control=mdwns, min_separation=0.0)
        v.disabled("within-assembly recurrence + assembly identity; ONLY the inter-assembly SEQUENCE band is "
                   "plastic (same scope as the conduction-delay PARTIAL)",
                   why="scope: reuses the established write unmodified -- only the READ instrument is new")
        decided = v.decide(go=go, verbose=False)
        attributable_to("weak-cue depth_frac gain (seeded replay vs NO-SEED lesion-the-replay)",
                        mwdf_a - mwdf_b, mwdf_ns - mwdf_b)
        summary_extra = dict(GO=go, n_go=n_go, status=decided.get("status"),
                             dw_fwd=mdwf, dw_rev=mdwr, dw_fwd_noseed=mdwns,
                             weak_depth_frac_before=mwdf_b, weak_depth_frac_after=mwdf_a,
                             weak_depth_frac_noseed=mwdf_ns, weak_tau_before=mwtau_b, weak_tau_after=mwtau_a,
                             full_depth_frac_before=mfull_dfb, full_depth_frac_after=mfull_dfa,
                             weak_forward_legacy_before=mwfwd_legacy_b, weak_forward_legacy_after=mwfwd_legacy_a,
                             n_headroom=n_headroom, n_directional=n_directional, n_lesion_ok=n_lesion_ok,
                             instrument_verify=verify, preconditions=decided.get("preconditions", []),
                             decided=decided)
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_go=0, instrument_verify=verify)

    summary = {"probe": "gap5_graded_recall_learn_through_use",
               "mechanism": "GRADED (non-binary-order) recall instrument -- sequence-completion DEPTH + Kendall-"
                            "tau trajectory fidelity -- applied to the established BTSP+forward-conduction-delay "
                            "directional replay write",
               "seeds": a.seeds, "n_mem": a.n_mem, "asm_size": a.asm_size,
               "cfg": vars(a),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[graded-ltu] VERDICT: {verdict}\n[graded-ltu] wrote {a.out}\n" + "=" * 120,
          flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
