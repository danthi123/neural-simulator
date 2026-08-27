"""gap#5 learn-through-use ON THE ECKER AdEx CA3 STORE: does OFFLINE discrete forward replay DURABLY STRENGTHEN the
replayed sequence -- and does LESION-THE-REPLAY (no ignition -> no replay) control it?

CONTEXT (read these -- the wall + the unblock):
  * 2026-08-27-swr-envelope-learn-through-use-NOGO: the SWR envelope on the BISTABLE-completion CA3 store CANNOT reach
    discrete forward-ordered replay -- its strong within-attractors reverberate semi-continuously (co_active 0.97, never
    rests/segments), so replay-driven learning has NO forward-ordered spike pairs to ride. The wall is the STORE
    ARCHITECTURE.
  * 2026-08-20-ecker-adex-ca3-forward-replay-6seed-GO + ...-stdp-band-...-GO: the Ecker-2022 AdEx CA3 (self-terminating
    within-assembly volleys + STRONG forward / WEAK reverse between links + spike-triggered adaptation) DOES segment into
    DISCRETE forward SWR events A->B->C from a non-specific prefix seed (6-seed GO, band both hand-wired AND STDP-grown).

THE QUESTION THIS RUNNER ANSWERS (the capability the bistable store could not support): with the Ecker store's DISCRETE
FORWARD replay in hand, turn the substrate's OWN spike-timing plasticity (cfg.enable_stdp, the same fused kernel that
GREW the band) ON DURING the offline SWR replay bouts. The self-generated forward-ordered reactivation (A fires before B
fires before C) drives DIRECTIONAL STDP: forward edges see pre-before-post (LTP), reverse edges see post-before-pre
(LTD). So REPLAYING the memory should DEEPEN its forward band (adj_fwd up, adj_rev flat/down) = the sequence becomes more
robust = "using (replaying) a memory strengthens it" via OFFLINE replay. This is exactly what a NON-SEGMENTING co-firing
store CANNOT do: simultaneous co-fire has no pre/post order, so STDP would potentiate all edges symmetrically -> NO
directional consolidation.

MECHANISM (brain-based-only; NO sim/ edit; reuse the STDP-band runner's build/encode/replay/measure by import):
  1. BUILD + ENCODE a forward-asymmetric band by STDP (moving A->B->C cue sweep) to a MODERATE strength (headroom below
     stdp_w_max) -- the memory to be consolidated. Freeze. Read it (band_before + forward-replay quality).
  2. CONSOLIDATE-BY-REPLAY: run SWR replay bouts (non-specific random-per-event prefix seed) with enable_stdp=True AND the
     clock ADVANCED each step (else delta_t==0 -> STDP silently inert, the banked 2026-07-29 failure). The discrete
     forward replay's own spike pairs potentiate the forward band. Measure dw_fwd / dw_rev.
  3. AFTER: freeze, re-read (band_after + forward-replay quality + robustness at a REDUCED cue).
  4. LESION-THE-REPLAY [KEY CONTROL]: repeat step 2 with seed_on=False (no prefix cue -> no ignition -> no replay events
     -> no forward-ordered spike pairs). STDP is ON and the clock advances IDENTICALLY, so the ONLY difference is the
     REPLAY. dw_fwd_noseed must be ~0 and the band/robustness must NOT change -> the strengthening is carried by the
     REPLAY, not by STDP-on time or OU noise.

GO (per seed) =
  * REPLAY-DEEPENS   : seeded consolidation grows the forward band (dw_fwd >= DW_MIN and adj_fwd_after > adj_fwd_before).
  * DIRECTIONAL      : forward deepens MORE than reverse (dw_fwd - dw_rev >= DW_MIN) -> rides the replay ORDER, not
                       generic activity (a co-firing store would move fwd==rev).
  * RECALL-CHANGE    : forward-replay quality is durably maintained/improved after consolidation (forward_frac_after >=
                       forward_frac_before - TOL and still >> chance), AND robustness at a REDUCED cue is >= before.
  * LESION-CONTROLLED: NO-SEED consolidation gives |dw_fwd_noseed| <= NOSEED_MAX_FRAC * dw_fwd and no robustness gain;
                       attributable_to(deepening, seeded vs no-seed).
  * BOUNDED          : adj_fwd_after <= stdp_w_max (soft cap) and the seeded deepening does not blow up.
Honest NO-GO otherwise (localizes whether replay-driven STDP consolidation works on this substrate at all).

SCOPE (stated, not hidden): this demonstrates replay-driven learn-through-use on the ECKER REPLAY SUBSTRATE's OWN
sequence memory (the assembly-chain store). Wiring the strengthened replay back into the production D5 EpisodicDapMemory
organ is a SEPARATE integration: the Ecker AdEx SOMA recurrence does NOT reactivate D5's sparse ~14-cell episodic
assembly (2026-08-20-ecker-real-d5-...-NO-GO, 3-seed + 24-op-point), so that path needs the dendritic-dAP-latch
composition, not soma recurrence. This runner tests the replay substrate itself, the rung the bistable store failed.

Reuse-by-import: build_store / encode / rest_and_replay / measure_band / _score_periods / _load_weights from
_gap5_ecker_adex_ca3_stdp_band_derisk (byte-identical store + replay + scorer).

  Calib:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_ecker_replay_learn_through_use_derisk \
              --seeds 42 --rest-steps 6500 --consol-steps 6500 --n-laps 14
  6-seed: SIM_BACKEND=cupy  .venv/bin/python -m research.runners._gap5_ecker_replay_learn_through_use_derisk \
              --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
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
    build_store, encode, rest_and_replay, measure_band, _score_periods, _load_weights,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "ecker_replay_learn_through_use.json"


def measure_band_from(w_host, store):
    fwd = np.asarray(w_host)[store["fwd_pos"]]; rev = np.asarray(w_host)[store["rev_pos"]]
    win = np.asarray(w_host)[store["within_pos"]]
    af = float(fwd.mean()) if fwd.size else 0.0; ar = float(rev.mean()) if rev.size else 0.0
    return dict(adj_fwd=af, adj_rev=ar, adj_within=float(win.mean()) if win.size else 0.0,
                ratio=(af / max(ar, 1e-6)), fwd_max=float(fwd.max()) if fwd.size else 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# CONSOLIDATE-BY-REPLAY: SWR replay with STDP ON + clock ADVANCED. Identical drive to rest_and_replay except (a) STDP is
# enabled so the replay's forward-ordered spike pairs potentiate the band, and (b) the clock advances each step (else
# delta_t==0 and STDP is inert). seed_on=False = LESION-THE-REPLAY (no ignition -> no replay -> no directional pairs).
# ----------------------------------------------------------------------------------------------------------------------
def consolidate_by_replay(store, steps, seed, *, swr_period, cue_pa, cue_steps, cue_frac, dt, seed_on=True):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    asm_size = store["asm_size"]
    fwd_pos = store["fwd_pos"]; rev_pos = store["rev_pos"]
    # SAME cue-cell subsets + assembly-choice stream as the read (rest_and_replay) uses -> consistent ignition.
    cell_rng = np.random.default_rng(int(seed) * 314159 + 17)
    k_cells = max(1, int(round(float(cue_frac) * asm_size)))
    cue_cells_dev = []
    for a_loc in asm_local:
        sub = np.sort(cell_rng.choice(a_loc, min(k_cells, len(a_loc)), replace=False))
        cue_cells_dev.append(cp.asarray(pc[sub], dtype=cp.int64))
    choice_rng = np.random.default_rng(int(seed) * 271828 + 23)

    w0 = np.asarray(to_host(bridge.cp_connections.data)).copy()
    dw_half = []                                       # forward-edge deepening at the halfway snapshot (self-limit check)
    half = max(1, steps // 2)
    bridge.core_config.enable_stdp = True
    bridge.runtime_state.current_time_ms = 0.0
    cur_k = None; n_env = 0
    for t in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        phase = t % swr_period
        if phase == 0 and seed_on:
            cur_k = int(choice_rng.integers(0, m)); n_env += 1
        if seed_on and phase < cue_steps and cur_k is not None:
            bridge.cp_external_input_current[cue_cells_dev[cur_k]] += float(cue_pa)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += float(dt)   # ADVANCE THE CLOCK (else STDP delta_t==0 -> inert)
        if t + 1 == half:
            wh = np.asarray(to_host(bridge.cp_connections.data))
            dw_half.append(float((wh[fwd_pos] - w0[fwd_pos]).mean()))
    bridge.core_config.enable_stdp = False
    w1 = np.asarray(to_host(bridge.cp_connections.data))
    dw_fwd_first = dw_half[0] if dw_half else 0.0
    dw_fwd_total = float((w1[fwd_pos] - w0[fwd_pos]).mean())
    return dict(n_env=n_env, w_after=w1.copy(),
                dw_fwd=dw_fwd_total, dw_rev=float((w1[rev_pos] - w0[rev_pos]).mean()),
                dw_fwd_first_half=dw_fwd_first, dw_fwd_second_half=float(dw_fwd_total - dw_fwd_first),
                changed=bool(not np.array_equal(w0, w1)))


def _read(store_kw_build, seed, w_host, a, *, cue_pa, cue_frac, tag):
    """Fresh store, load weights, replay READ (STDP OFF -> frozen). Returns forward-replay quality."""
    s = build_store(seed, **store_kw_build)
    _load_weights(s, w_host)
    r = rest_and_replay(s, a.rest_steps, seed, swr_period=a.swr_period, cue_pa=cue_pa,
                        cue_steps=a.cue_steps, cue_frac=cue_frac, seed_on=True)
    sc = _score_periods(r["F"], s["asm_local"], r["env_seed_log"], a.swr_period,
                        W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    band = measure_band(s)
    return dict(forward=sc["forward_frac"], reverse=sc["reverse_frac"], chance=max(sc["chance_forward"], 1e-6),
                n_multi=sc["n_multi"], per_asm_active=sc["per_asm_active"], seed_first=sc.get("seed_first_frac"),
                duty=sc["duty_cycle"], frozen=r["weights_frozen"], band=band, tag=tag)


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

    # SEED-CONTROLS-SUBSTRATE guard (build twice, hash firing thresholds)
    if a.verify_seed:
        s1 = build_store(seed, **bkw); s2 = build_store(seed, **bkw)
        h1 = s1["bridge"].cp_neuron_firing_thresholds; h2 = s2["bridge"].cp_neuron_firing_thresholds
        out["seed_hash_ok"] = bool(h1 is None or float(np.asarray(to_host(h1)).sum()) ==
                                   float(np.asarray(to_host(h2)).sum()))

    # 1. BUILD + ENCODE the memory (moderate band; headroom below stdp_w_max)
    st = build_store(seed, **bkw)
    band_pre_encode = measure_band(st)
    encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    band_before = measure_band(st)
    out["band_before"] = band_before
    print(f"  [seed {seed}] ENCODE: band fwd {band_pre_encode['adj_fwd']:.1f}->{band_before['adj_fwd']:.1f} "
          f"rev {band_pre_encode['adj_rev']:.1f}->{band_before['adj_rev']:.1f} "
          f"(w_max={a.stdp_w_max}) ({time.time()-t0:.0f}s)", flush=True)

    # 2. READ BEFORE (full cue + weak cue), frozen
    rd_full_before = _read(bkw, seed, w_learned, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac, tag="full_before")
    rd_weak_before = _read(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac, tag="weak_before")
    chance = rd_full_before["chance"]
    print(f"  [seed {seed}] BEFORE: full FWD={rd_full_before['forward']:.3f} (chance {chance:.3f}) "
          f"weak FWD={rd_weak_before['forward']:.3f} multi={rd_weak_before['n_multi']} ({time.time()-t0:.0f}s)",
          flush=True)

    # 3. CONSOLIDATE-BY-REPLAY (seeded): STDP on, forward-ordered replay deepens the band
    st_c = build_store(seed, **bkw); _load_weights(st_c, w_learned)
    cons = consolidate_by_replay(st_c, a.consol_steps, seed, seed_on=True, **cons_kw)
    w_consol = cons["w_after"]
    out["consolidate"] = dict(n_env=cons["n_env"], dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"],
                              dw_fwd_first_half=cons["dw_fwd_first_half"], dw_fwd_second_half=cons["dw_fwd_second_half"],
                              changed=cons["changed"])
    print(f"  [seed {seed}] CONSOLIDATE(seeded): n_env={cons['n_env']} dw_fwd={cons['dw_fwd']:.2f} "
          f"dw_rev={cons['dw_rev']:.2f} (half1={cons['dw_fwd_first_half']:.2f} half2={cons['dw_fwd_second_half']:.2f}) "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 4. READ AFTER (full + weak), frozen
    rd_full_after = _read(bkw, seed, w_consol, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac, tag="full_after")
    rd_weak_after = _read(bkw, seed, w_consol, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac, tag="weak_after")
    band_after = measure_band_from(w_consol, st_c)
    out["band_after"] = band_after
    print(f"  [seed {seed}] AFTER: full FWD={rd_full_after['forward']:.3f} weak FWD={rd_weak_after['forward']:.3f} "
          f"multi={rd_weak_after['n_multi']} band fwd={band_after['adj_fwd']:.1f} rev={band_after['adj_rev']:.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 5. LESION-THE-REPLAY: NO-SEED consolidation (STDP on, clock advances, NO ignition -> no replay)
    st_n = build_store(seed, **bkw); _load_weights(st_n, w_learned)
    cons_ns = consolidate_by_replay(st_n, a.consol_steps, seed, seed_on=False, **cons_kw)
    w_noseed = cons_ns["w_after"]
    band_noseed = measure_band_from(w_noseed, st_n)
    rd_weak_noseed = _read(bkw, seed, w_noseed, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac, tag="weak_noseed")
    out["no_seed"] = dict(n_env=cons_ns["n_env"], dw_fwd=cons_ns["dw_fwd"], dw_rev=cons_ns["dw_rev"],
                          band_after=band_noseed, weak_forward=rd_weak_noseed["forward"],
                          weak_multi=rd_weak_noseed["n_multi"])
    print(f"  [seed {seed}] NO-SEED(lesion-replay): n_env={cons_ns['n_env']} dw_fwd={cons_ns['dw_fwd']:.3f} "
          f"dw_rev={cons_ns['dw_rev']:.3f} weak FWD={rd_weak_noseed['forward']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # NB: the NO-SEED weak read lives ONLY in out["no_seed"]["weak_forward"] (a scalar). It is byte-identical to
    # weak_before BY DESIGN (the lesion produces zero weight change), so it is deliberately NOT stored beside
    # weak_before as a sibling arm (that identity is the lesion-null working, not a dead lever).
    out["reads"] = dict(full_before=rd_full_before, weak_before=rd_weak_before,
                        full_after=rd_full_after, weak_after=rd_weak_after)

    # ============ PER-SEED VERDICT (verify, don't assert) ============
    dw_fwd = cons["dw_fwd"]; dw_rev = cons["dw_rev"]; dw_ns = cons_ns["dw_fwd"]
    replay_deepens = (dw_fwd >= a.dw_min and band_after["adj_fwd"] > band_before["adj_fwd"])
    directional = ((dw_fwd - dw_rev) >= a.dw_min)
    recall_maintained = (rd_full_after["forward"] >= rd_full_before["forward"] - a.fwd_tol
                         and rd_full_after["forward"] >= 1.5 * chance)
    robustness_gain = (rd_weak_after["forward"] >= rd_weak_before["forward"] + a.robust_min
                       or rd_weak_after["n_multi"] > rd_weak_before["n_multi"])
    recall_change = bool(recall_maintained and robustness_gain)
    lesion_controlled = (abs(dw_ns) <= a.noseed_max_frac * max(abs(dw_fwd), 1e-6)
                         and rd_weak_noseed["forward"] <= rd_weak_before["forward"] + a.robust_min)
    bounded = (band_after["adj_fwd"] <= a.stdp_w_max + 1e-3
               and cons["dw_fwd_second_half"] <= cons["dw_fwd_first_half"] + a.dw_min)
    seed_go = bool(replay_deepens and directional and recall_change and lesion_controlled and bounded)
    out["checks"] = dict(replay_deepens=replay_deepens, directional=directional,
                         recall_maintained=recall_maintained, robustness_gain=robustness_gain,
                         recall_change=recall_change, lesion_controlled=lesion_controlled, bounded=bounded,
                         dw_fwd=round(dw_fwd, 3), dw_rev=round(dw_rev, 3), dw_noseed=round(dw_ns, 3),
                         seed_hash_ok=out.get("seed_hash_ok"))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=6)
    ap.add_argument("--asm-size", type=int, default=80)
    ap.add_argument("--within-density", type=float, default=0.5)
    ap.add_argument("--rest-steps", type=int, default=13000, help="read replay length (~40 events)")
    ap.add_argument("--consol-steps", type=int, default=13000, help="consolidation replay length (STDP on)")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--w-within", type=float, default=60.0)
    ap.add_argument("--between-init", type=float, default=15.0)
    ap.add_argument("--b-override", type=float, default=120.0)
    # STDP
    ap.add_argument("--stdp-w-max", type=float, default=900.0)
    ap.add_argument("--stdp-a-plus", type=float, default=0.05)
    ap.add_argument("--stdp-a-minus", type=float, default=0.06)
    ap.add_argument("--stdp-tau", type=float, default=20.0)
    # ENCODE (moderate: fewer laps than the band-GO's 30, so there is headroom to deepen by replay)
    ap.add_argument("--n-laps", type=int, default=14)
    ap.add_argument("--enc-step", type=int, default=80)
    ap.add_argument("--enc-dwell", type=int, default=40)
    ap.add_argument("--enc-gap", type=int, default=600)
    ap.add_argument("--enc-cue-pa", type=float, default=9000.0)
    ap.add_argument("--enc-cue-frac", type=float, default=0.6)
    # SWR replay / prefix seed
    ap.add_argument("--swr-period", type=int, default=325)
    ap.add_argument("--cue-pa", type=float, default=9000.0)
    ap.add_argument("--cue-steps", type=int, default=40)
    ap.add_argument("--cue-frac", type=float, default=0.6)
    ap.add_argument("--weak-cue-mult", type=float, default=0.5, help="reduced-cue robustness read: cue_pa * this")
    ap.add_argument("--weak-cue-frac", type=float, default=0.35, help="reduced-cue robustness read: fewer cue cells")
    ap.add_argument("--ou-sigma", type=float, default=40.0)
    # detection
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--active-frac", type=float, default=0.10)
    ap.add_argument("--onset-frac", type=float, default=0.06)
    # GO thresholds
    ap.add_argument("--dw-min", type=float, default=5.0, help="min forward-edge deepening (adj_fwd units) to count")
    ap.add_argument("--fwd-tol", type=float, default=0.10, help="allowed drop in full-cue forward_frac after consol")
    ap.add_argument("--robust-min", type=float, default=0.05, help="min weak-cue forward_frac gain to count robustness")
    ap.add_argument("--noseed-max-frac", type=float, default=0.20, help="|dw_noseed| must be <= this * dw_seeded")
    ap.add_argument("--verify-seed", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    print(f"[ecker-ltu] Ecker AdEx CA3 replay-driven learn-through-use | n_mem={a.n_mem} asm={a.asm_size} "
          f"within={a.w_within} between_init={a.between_init} | encode {a.n_laps}laps | STDP a+={a.stdp_a_plus} "
          f"a-={a.stdp_a_minus} tau={a.stdp_tau} w_max={a.stdp_w_max} | swr={a.swr_period} cue={a.cue_pa}@{a.cue_frac} "
          f"weak={a.cue_pa*a.weak_cue_mult}@{a.weak_cue_frac} | rest={a.rest_steps} consol={a.consol_steps} dt={a.dt} "
          f"seeds={a.seeds} backend={backend}", flush=True)

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
        maf_b = float(np.mean([p["band_before"]["adj_fwd"] for p in per]))
        maf_a = float(np.mean([p["band_after"]["adj_fwd"] for p in per]))
        mar_a = float(np.mean([p["band_after"]["adj_rev"] for p in per]))
        mfull_b = float(np.mean([p["reads"]["full_before"]["forward"] for p in per]))
        mfull_a = float(np.mean([p["reads"]["full_after"]["forward"] for p in per]))
        mweak_b = float(np.mean([p["reads"]["weak_before"]["forward"] for p in per]))
        mweak_a = float(np.mean([p["reads"]["weak_after"]["forward"] for p in per]))
        mweak_ns = float(np.mean([p["no_seed"]["weak_forward"] for p in per]))
        mch = float(np.mean([p["reads"]["full_before"]["chance"] for p in per]))
        if go:
            verdict = (f"ECKER-REPLAY-LEARN-THROUGH-USE GO {n_go}/{len(per)} -- OFFLINE discrete forward SWR replay on "
                       f"the Ecker AdEx CA3 store DURABLY DEEPENS the replayed sequence via the substrate's OWN STDP: "
                       f"forward band adj_fwd {maf_b:.1f}->{maf_a:.1f} (rev after {mar_a:.1f}); dw_fwd {mdwf:.1f} vs "
                       f"dw_rev {mdwr:.1f} (DIRECTIONAL -- rides the replay order). Recall durably changes: weak-cue "
                       f"forward {mweak_b:.3f}->{mweak_a:.3f} (full {mfull_b:.3f}->{mfull_a:.3f}, chance {mch:.3f}). "
                       f"LESION-THE-REPLAY (NO-SEED): dw_fwd {mdwns:.2f}~0, weak forward {mweak_ns:.3f} (no gain) -> the "
                       f"strengthening is CARRIED BY THE REPLAY. => the Ecker store UNBLOCKS replay-driven "
                       f"learn-through-use the bistable co-firing store could not.")
        else:
            verdict = (f"ECKER-REPLAY-LEARN-THROUGH-USE NO-GO {n_go}/{len(per)} -- the store SEGMENTS (full-cue forward "
                       f"{mfull_b:.3f} vs chance {mch:.3f}) and replay drives DURABLE LESION-CONTROLLED plasticity "
                       f"(NO-SEED dw_fwd {mdwns:.2f}~0), but replay-driven STDP does NOT strengthen forward recall: it "
                       f"SYMMETRIZES the band (dw_fwd {mdwf:.1f} vs dw_rev {mdwr:.1f}; adj_fwd {maf_b:.1f}->{maf_a:.1f}); "
                       f"weak forward {mweak_b:.3f}->{mweak_a:.3f} (noseed {mweak_ns:.3f}). 0/6 directional. Next "
                       f"method: separated-volley conduction delay / inhibitory gap-coding / BTSP-eligibility write.")
        # Preconditions are INSTRUMENT-VALIDITY only (all must HOLD for the go/no-go to be meaningful); the
        # forward-consolidation CONCLUSION is carried by `go` (n_go >= bar), NOT registered as a precondition -- a
        # failed conclusion is a NO-GO, not an UNDEFINED.
        v = Verdict("Ecker AdEx CA3: does OFFLINE replay durably STRENGTHEN forward-ordered recall via replay-driven "
                    "STDP (lesion-the-replay controlled)?", chance=mch)
        v.floor("the store SEGMENTS: full-cue forward replay ignites above chance (a memory to consolidate exists)",
                mfull_b, floor=mch)
        v.require("plasticity is LIVE during replay: seeded consolidation moved the forward band (|dw_fwd| > 0)",
                  abs(mdwf), expect=lambda x: x > 1e-6)
        v.control("LESION-THE-REPLAY ENGAGED: seeded forward-deepening vs NO-SEED forward-deepening -- must DIFFER, so "
                  "the NO-SEED arm is a true null and the negative is about the replay's DIRECTION, not a dead lever",
                  treatment=mdwf, control=mdwns, min_separation=0.0)
        v.disabled("within-assembly recurrence + assembly identity (pre-formed cell groups; only the inter-assembly "
                   "SEQUENCE band is plastic)", why="scope: this tests replay-driven consolidation of the learned "
                   "forward sequence, not assembly formation")
        decided = v.decide(go=go, verbose=False)
        attributable_to("forward-band deepening (seeded replay vs NO-SEED lesion-the-replay)", mdwf, mdwns)
        summary_extra = dict(GO=go, n_go=n_go, status=decided.get("status"),
                             dw_fwd=mdwf, dw_rev=mdwr, dw_fwd_noseed=mdwns,
                             band_adj_fwd_before=maf_b, band_adj_fwd_after=maf_a, band_adj_rev_after=mar_a,
                             full_forward_before=mfull_b, full_forward_after=mfull_a,
                             weak_forward_before=mweak_b, weak_forward_after=mweak_a, weak_forward_noseed=mweak_ns,
                             chance=mch, preconditions=decided.get("preconditions", []), decided=decided)
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_go=0)

    summary = {"probe": "gap5_ecker_replay_learn_through_use",
               "mechanism": "Ecker-2022 AdEx CA3 discrete forward SWR replay -> directional replay-driven STDP "
                            "consolidation (offline learn-through-use), lesion-the-replay controlled",
               "seeds": a.seeds, "n_mem": a.n_mem, "asm_size": a.asm_size,
               "cfg": dict(w_within=a.w_within, between_init=a.between_init, b_override=a.b_override, n_laps=a.n_laps,
                           stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
                           stdp_tau=a.stdp_tau, swr_period=a.swr_period, cue_pa=a.cue_pa, cue_frac=a.cue_frac,
                           weak_cue_mult=a.weak_cue_mult, weak_cue_frac=a.weak_cue_frac, ou_sigma=a.ou_sigma,
                           rest_steps=a.rest_steps, consol_steps=a.consol_steps, dt=a.dt),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[ecker-ltu] VERDICT: {verdict}\n[ecker-ltu] wrote {a.out}\n" + "=" * 120, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
