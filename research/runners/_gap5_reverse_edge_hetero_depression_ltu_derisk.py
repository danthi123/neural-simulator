"""gap#5 learn-through-use, REVERSE-EDGE HETEROSYNAPTIC-DEPRESSION variant (2026-08-27): the graded-recall NO-GO
([[2026-08-27-graded-recall-instrument-learn-through-use-NOGO]]) pinned the root cause of the recall-depth REGRESSION
as PURE-POTENTIATION BTSP -- the established directional write (BTSP-eligibility x forward-edge conduction delay) keeps
dw_fwd > dw_rev on 6/6 seeds, but the reverse edges still DEEPEN in absolute terms (dw_rev ~84% of dw_fwd, nothing
depresses them), and that residual reverse growth is large enough to intrude on weak-cue completion and TERMINATE the
forward prefix earlier than before training (weak-cue depth_frac 0.586->0.497, 5/6 seeds).

THIS RUNNER tests the named next mechanism: heterosynaptic DEPRESSION on the reverse edges, keyed to the SAME-step
forward potentiation ("the winner-forward edge suppresses the loser-reverse edge") -- Chistiakova-Volgushev
heterosynaptic plasticity / Miller-MacKay competitive normalization: biology runs a depressive process ALONGSIDE
potentiation that a pure-potentiation BTSP write omits (the "what else does the real system run alongside this, that
we replaced with a constant?" wall-reframe). A 2025 PLoS Comput Biol model (Kim & Kim, "Selective inhibition in CA3:
a mechanism for stable pattern completion through heterosynaptic plasticity", PMID 40623085) independently grounds
heterosynaptic plasticity as the competition mechanism that stabilizes CA3 pattern completion against interference
from competing engrams -- the same failure mode this runner targets (reverse-band interference degrading completion
depth).

FIRST ATTEMPT (recorded, not the final design): the substrate ALREADY HAS a general per-postsynaptic-neuron
heterosynaptic-competition kernel, `sim.kernels.fused_btsp_hetero_update` (gap#4<->gap#5 unification, 2026-07-18/20,
wired behind `cfg.btsp_hetero_dep`, previously exercised at hetero_dep=0.2 by place-field/dendritic-subunit runners):
dw_ij = eta*IS_i*[Etilde_j*(w_max-w_ij) - lam_dep*(1-Etilde_j)*(w_ij-w_min)] -- depresses each plateauing post
neuron's LOW-eligibility inputs. A single-seed scan (lam_dep in 0.05..1.5) showed this does NOT selectively suppress
reverse edges on THIS substrate/timing regime: the reverse/forward weight ratio INCREASED with lam_dep (0.894 at
lam_dep=0 -> 0.90-1.04 as lam_dep rose) because Ecker AdEx post neurons re-fire repeatedly through a burst, and by
the LATER re-firings the FORWARD partner's own eligibility has also decayed -- so the per-postsynaptic-neuron
competition depresses forward and reverse comparably (sometimes forward MORE). Banked as a genuine negative for that
specific implementation; NOT reused below.

THE MECHANISM ACTUALLY TESTED: a targeted BAND-LEVEL heterosynaptic-LTD term, applied ONLY to reverse edges, keyed to
the reverse edge's OWN "post-fires-before-pre" mismatch relative to the SAME-step forward drive:

    w_fwd_new = fused_btsp_update(w_fwd, e_fwd, p_fwd, eta, w_min, w_max)                       [UNCHANGED math]
    pot_fwd_signal = eta * mean(e_fwd * p_fwd * (w_max - w_fwd))     [this step's mean forward LTP drive, a scalar]
    w_rev_pot = fused_btsp_update(w_rev, e_rev, p_rev, eta, w_min, w_max)                        [UNCHANGED math]
    w_rev_new = clip(w_rev_pot - lam_dep * pot_fwd_signal * (w_rev - w_min), w_min, w_max)   [NEW: heterosynaptic LTD]

The forward path is LITERALLY the established `fused_btsp_update` call, untouched. The reverse path computes the
SAME established potentiation, then subtracts a depression proportional to (a) the reverse edge's own current weight
above floor and (b) how strongly the forward band potentiated THIS SAME STEP -- so reverse depression scales with
"how much is the forward edge winning right now", matching the task's own framing directly. lam_dep=0.0 (default)
makes the subtracted term IDENTICALLY 0.0*x=0.0 (IEEE754-exact) for every reverse edge -> w_rev_new reduces to
EXACTLY the same `fused_btsp_update` call as the established write (same slice, same formula, no reordering) ->
VERIFIED byte-identical (exact SHA-256 match) below, unlike the first-attempt kernel (which reordered the SAME
formula and left ~2e-3 float32 rounding noise on weights ~O(200-900), i.e. not a real behavioral difference but not
an EXACT hash match either -- this redesign closes that gap).

Reuse-by-import (NO sim/ edit; only a NEW consolidate function built from the EXISTING kernel + the graded
instrument already built for this exact question):
  build_store / encode / rest_and_replay / measure_band / _load_weights / _smooth <- _gap5_ecker_adex_ca3_stdp_band_derisk
  measure_band_from <- _gap5_ecker_replay_learn_through_use_derisk (established write, for the byte-identical-off check)
  consolidate_by_btsp_replay_delayed <- _gap5_ecker_replay_learn_through_use_derisk (established write, ditto)
  _score_periods_graded / _read_graded / verify_instrument <- _gap5_graded_recall_learn_through_use_derisk (the
    PROVEN-graded depth+tau instrument, unmodified -- only the WRITE changes in this runner)
  fused_btsp_update <- sim.kernels (the established pure-potentiation kernel, called TWICE per step -- fwd and rev
    slices separately -- with the depression term added only to the reverse slice's result)

  Byte-identical-off check: SIM_BACKEND=numpy .venv/bin/python -m
      research.runners._gap5_reverse_edge_hetero_depression_ltu_derisk --byte-identical-check --seeds 42
  Lam-dep scan (cheap, seed 42): SIM_BACKEND=numpy .venv/bin/python -m
      research.runners._gap5_reverse_edge_hetero_depression_ltu_derisk --scan-lam-dep --seeds 42
  6-seed decisive: SIM_BACKEND=numpy .venv/bin/python -m
      research.runners._gap5_reverse_edge_hetero_depression_ltu_derisk --lam-dep <chosen> \
          --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import hashlib
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
from sim.kernels import fused_btsp_update  # noqa: E402  -- REUSE the established pure-potentiation kernel unmodified
from research.runners._gap5_ecker_adex_ca3_stdp_band_derisk import (  # noqa: E402
    build_store, encode, rest_and_replay, measure_band, _load_weights, _smooth,
)
from research.runners._gap5_ecker_replay_learn_through_use_derisk import (  # noqa: E402
    measure_band_from, consolidate_by_btsp_replay_delayed,
)
from research.runners._gap5_graded_recall_learn_through_use_derisk import (  # noqa: E402
    _score_periods_graded, _read_graded, verify_instrument,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "reverse_edge_hetero_depression_ltu.json"
SCAN_OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "reverse_edge_hetero_depression_lamdep_scan.json"


# ----------------------------------------------------------------------------------------------------------------------
# THE NEW WRITE: identical to consolidate_by_btsp_replay_delayed (conduction-delay-separated SWR replay, same
# eligibility/plateau traces, same decoupled-forward-drive delay line, same fused_btsp_update potentiation math on
# BOTH bands) PLUS one addition: after computing this step's forward and reverse potentiation with the UNCHANGED
# established kernel, subtract a heterosynaptic-LTD term from the reverse edges ONLY, proportional to (a) the
# reverse edge's own weight above floor and (b) how strongly the forward band potentiated this SAME step. Everything
# else (delay_steps, elig_tau_ms, plat_tau_ms, eta, swr_period, cue timing) stays identical to the NO-GO's write
# hyperparameters -- lam_dep is the ONLY new lever, additive and default-OFF (0.0).
# ----------------------------------------------------------------------------------------------------------------------
def consolidate_by_hetero_replay_delayed(store, steps, seed, *, swr_period, cue_pa, cue_steps, cue_frac, dt,
                                         seed_on=True, elig_tau_ms, plat_tau_ms, eta, w_min, w_max, delay_steps,
                                         lam_dep=0.0, overlap_kw=None):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    asm_size = store["asm_size"]; n_pc = len(pc)
    fwd_pos = store["fwd_pos"]; rev_pos = store["rev_pos"]; n_fwd = int(fwd_pos.size)
    row, col = store["pre_post"]
    bet_pos = np.concatenate([fwd_pos, rev_pos])
    bet_pos_dev = cp.asarray(bet_pos.astype(np.int64))
    fwd_pos_dev = cp.asarray(fwd_pos.astype(np.int64)); rev_pos_dev = cp.asarray(rev_pos.astype(np.int64))
    row_bet = cp.asarray(row[bet_pos].astype(np.int64)); col_bet = cp.asarray(col[bet_pos].astype(np.int64))
    row_fwd = cp.asarray(row[fwd_pos].astype(np.int64)); col_fwd = cp.asarray(col[fwd_pos].astype(np.int64))
    row_rev_e = row_bet[n_fwd:]; col_rev_e = col_bet[n_fwd:]  # gather indices for the reverse SLICE of w_bet
    prop = float(getattr(bridge.core_config, "propagation_strength", 0.05))
    nN = int(bridge.cp_firing_states.size)

    cell_rng = np.random.default_rng(int(seed) * 314159 + 17)
    k_cells = max(1, int(round(float(cue_frac) * asm_size)))
    cue_cells_dev = []
    for a_loc in asm_local:
        sub = np.sort(cell_rng.choice(a_loc, min(k_cells, len(a_loc)), replace=False))
        cue_cells_dev.append(cp.asarray(pc[sub], dtype=cp.int64))
    choice_rng = np.random.default_rng(int(seed) * 271828 + 23)

    e_pre = cp.zeros(nN, dtype=cp.float32); p_post = cp.zeros(nN, dtype=cp.float32)
    decay_e = cp.float32(np.exp(-dt / max(elig_tau_ms, 1e-9))); decay_p = cp.float32(np.exp(-dt / max(plat_tau_ms, 1e-9)))
    eta_d = cp.float32(eta); wmin_d = cp.float32(w_min); wmax_d = cp.float32(w_max)
    lam_d = cp.float32(lam_dep)

    w0 = np.asarray(to_host(bridge.cp_connections.data)).copy()
    w_bet = bridge.cp_connections.data[bet_pos_dev].copy()
    bridge.cp_connections.data[fwd_pos_dev] = cp.float32(0.0)
    D = int(max(0, delay_steps))
    buf = cp.zeros((max(D, 1), nN), dtype=cp.float32) if D > 0 else None
    ptr = 0

    F = np.zeros((steps, n_pc), dtype=bool); env_seed_log = []
    bridge.core_config.enable_stdp = False
    bridge.runtime_state.current_time_ms = 0.0
    cur_k = None; n_env = 0
    for t in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        phase = t % swr_period
        if phase == 0 and seed_on:
            cur_k = int(choice_rng.integers(0, m)); env_seed_log.append(cur_k); n_env += 1
        if seed_on and phase < cue_steps and cur_k is not None:
            bridge.cp_external_input_current[cue_cells_dev[cur_k]] += float(cue_pa)
        prev_f = bridge.cp_prev_firing_states.astype(cp.float32)
        fwd_g = cp.zeros(nN, dtype=cp.float32)
        w_fwd_now = w_bet[:n_fwd]
        cp.add.at(fwd_g, col_fwd, w_fwd_now * prev_f[row_fwd] * cp.float32(prop))
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += float(dt)
        if D <= 0:
            bridge.cp_conductance_g_e += fwd_g
        else:
            bridge.cp_conductance_g_e += buf[ptr]
            buf[ptr] = fwd_g; ptr = (ptr + 1) % D
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[pc].astype(bool)
        fired = bridge.cp_firing_states.astype(cp.float32)
        e_pre = cp.maximum(e_pre * decay_e, fired); p_post = cp.maximum(p_post * decay_p, fired)
        # THE ONLY DIFF from consolidate_by_btsp_replay_delayed: forward stays the LITERAL established
        # fused_btsp_update call (untouched); the reverse slice gets the SAME established call PLUS a
        # heterosynaptic-LTD subtraction keyed to this step's forward drive. lam_dep=0.0 -> the subtracted term is
        # 0.0*x=0.0 (IEEE754-exact) for every reverse edge -> w_rev_new IS (bit-for-bit) the established
        # fused_btsp_update result -- verified byte-identical below (--byte-identical-check).
        e_fwd = e_pre[row_fwd]; p_fwd = p_post[col_fwd]
        e_rev = e_pre[row_rev_e]; p_rev = p_post[col_rev_e]
        w_fwd_now = w_bet[:n_fwd]; w_rev_now = w_bet[n_fwd:]
        w_fwd_new = fused_btsp_update(w_fwd_now, e_fwd, p_fwd, eta_d, wmin_d, wmax_d)          # UNCHANGED math
        pot_fwd_signal = eta_d * cp.mean(e_fwd * p_fwd * (wmax_d - w_fwd_now))                  # this-step forward drive (scalar)
        w_rev_pot = fused_btsp_update(w_rev_now, e_rev, p_rev, eta_d, wmin_d, wmax_d)           # UNCHANGED math
        dep_rev = lam_d * pot_fwd_signal * (w_rev_now - wmin_d)                                 # NEW: heterosynaptic LTD
        w_rev_new = cp.clip(w_rev_pot - dep_rev, wmin_d, wmax_d)
        w_bet = cp.concatenate([w_fwd_new, w_rev_new])
        bridge.cp_connections.data[rev_pos_dev] = w_bet[n_fwd:]
    w1 = np.asarray(to_host(bridge.cp_connections.data)).copy()
    w_bet_h = np.asarray(to_host(w_bet))
    w1[fwd_pos] = w_bet_h[:n_fwd]; w1[rev_pos] = w_bet_h[n_fwd:]
    ov = None
    if overlap_kw is not None and seed_on:
        from research.runners._gap5_ecker_replay_learn_through_use_derisk import _volley_overlap
        ov = _volley_overlap(F, asm_local, env_seed_log, swr_period, **overlap_kw)
    return dict(n_env=n_env, w_after=w1.copy(),
                dw_fwd=float((w1[fwd_pos] - w0[fwd_pos]).mean()), dw_rev=float((w1[rev_pos] - w0[rev_pos]).mean()),
                dw_fwd_first_half=0.0, dw_fwd_second_half=0.0, volley_overlap=ov,
                changed=bool(not np.array_equal(w0, w1)))


# ----------------------------------------------------------------------------------------------------------------------
# BYTE-IDENTICAL-OFF CHECK (asserted IN THE DATA, per docs/TERMS.md -- a hash/exact compare, not read-the-code).
# Runs the ESTABLISHED write (fused_btsp_update, the NO-GO's exact function) and the NEW write with lam_dep=0.0 on the
# SAME encoded store/seed and compares w_after exactly (hash + max-abs-diff).
# ----------------------------------------------------------------------------------------------------------------------
def byte_identical_check(seed, a):
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    cons_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac, dt=a.dt)

    st_a = build_store(seed, **bkw); encode(st_a, seed, **enc_kw)
    w_learned = np.asarray(to_host(st_a["bridge"].cp_connections.data)).copy()

    st_old = build_store(seed, **bkw); _load_weights(st_old, w_learned)
    old = consolidate_by_btsp_replay_delayed(st_old, a.consol_steps, seed, seed_on=True,
                                             elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                             eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                             delay_steps=a.fwd_delay_steps, **cons_kw)
    st_new = build_store(seed, **bkw); _load_weights(st_new, w_learned)
    new = consolidate_by_hetero_replay_delayed(st_new, a.consol_steps, seed, seed_on=True,
                                               elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                               eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                               delay_steps=a.fwd_delay_steps, lam_dep=0.0, **cons_kw)
    w_old = np.asarray(old["w_after"]); w_new = np.asarray(new["w_after"])
    h_old = hashlib.sha256(w_old.tobytes()).hexdigest(); h_new = hashlib.sha256(w_new.tobytes()).hexdigest()
    maxdiff = float(np.max(np.abs(w_old - w_new)))
    exact = bool(h_old == h_new)
    print(f"[byte-identical-check] seed={seed} sha256_old={h_old[:16]} sha256_new={h_new[:16]} "
          f"EXACT_HASH_MATCH={exact} max_abs_diff={maxdiff:.3e} dw_fwd_old={old['dw_fwd']:.4f} "
          f"dw_fwd_new={new['dw_fwd']:.4f} dw_rev_old={old['dw_rev']:.4f} dw_rev_new={new['dw_rev']:.4f}", flush=True)
    return dict(seed=seed, sha256_old=h_old, sha256_new=h_new, exact_hash_match=exact, max_abs_diff=maxdiff,
               dw_fwd_old=old["dw_fwd"], dw_fwd_new=new["dw_fwd"], dw_rev_old=old["dw_rev"], dw_rev_new=new["dw_rev"])


# ----------------------------------------------------------------------------------------------------------------------
# CHEAP LAM_DEP SCAN (single seed, decisive read regime): pick the smallest lam_dep that (1) keeps the write
# directional, (2) LOWERS the dw_rev/dw_fwd ratio below the NO-GO's ~0.84, (3) shows a weak-cue depth_frac gain --
# WITHOUT retuning the established write hyperparameters (fwd_delay_steps/elig_tau/plat_tau/swr_period stay fixed).
# ----------------------------------------------------------------------------------------------------------------------
def scan_lam_dep(seed, a, mults):
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    cons_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac, dt=a.dt)
    weak_pa = a.cue_pa * a.weak_cue_mult
    read_period = a.read_swr_period if a.read_swr_period > 0 else a.swr_period

    st = build_store(seed, **bkw); encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    rd_weak_before = _read_graded(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="weak_before")
    rows = []
    for mult in mults:
        st_c = build_store(seed, **bkw); _load_weights(st_c, w_learned)
        cons = consolidate_by_hetero_replay_delayed(st_c, a.consol_steps, seed, seed_on=True,
                                                     elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                                     eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                                     delay_steps=a.fwd_delay_steps, lam_dep=mult, **cons_kw)
        rd_weak_after = _read_graded(bkw, seed, cons["w_after"], a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                     swr_period=read_period, rest_steps=a.rest_steps, tag=f"weak_after_lam{mult}")
        ratio = cons["dw_rev"] / max(cons["dw_fwd"], 1e-6)
        gain = rd_weak_after["depth_frac"] - rd_weak_before["depth_frac"]
        rows.append(dict(lam_dep=mult, dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"], rev_fwd_ratio=ratio,
                         weak_depth_frac_before=rd_weak_before["depth_frac"],
                         weak_depth_frac_after=rd_weak_after["depth_frac"], depth_gain=gain,
                         weak_tau_before=rd_weak_before["tau"], weak_tau_after=rd_weak_after["tau"]))
        print(f"  [scan] lam_dep={mult:.3f}: dw_fwd={cons['dw_fwd']:.2f} dw_rev={cons['dw_rev']:.2f} "
              f"ratio={ratio:.3f} weak depth_frac {rd_weak_before['depth_frac']:.3f}->"
              f"{rd_weak_after['depth_frac']:.3f} (gain {gain:+.3f})", flush=True)
    return dict(seed=seed, weak_depth_frac_before=rd_weak_before["depth_frac"], rows=rows)


# ----------------------------------------------------------------------------------------------------------------------
# DECISIVE PER-SEED TEST: identical to the graded-recall NO-GO's one_seed() (same BUILD/ENCODE/READ/LESION scaffold,
# same GO bar structure) with ONE change: the seeded consolidation calls consolidate_by_hetero_replay_delayed
# (lam_dep=a.lam_dep) instead of consolidate_by_btsp_replay_delayed. The NO-SEED lesion control uses the SAME hetero
# write function (seed_on=False -> no ignition -> no eligibility/plateau events -> depression term never fires either,
# same null-control logic as the pure-potentiation write).
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

    st = build_store(seed, **bkw)
    encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    band_before = measure_band(st)
    out["band_before"] = band_before
    print(f"  [seed {seed}] ENCODE: band fwd={band_before['adj_fwd']:.1f} rev={band_before['adj_rev']:.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    rd_full_before = _read_graded(bkw, seed, w_learned, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="full_before")
    rd_weak_before = _read_graded(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="weak_before")
    print(f"  [seed {seed}] BEFORE: full depth_frac={rd_full_before['depth_frac']:.3f} | weak depth_frac="
          f"{rd_weak_before['depth_frac']:.3f} tau={rd_weak_before['tau']:.3f} n_multi={rd_weak_before['n_multi']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    overlap_kw = dict(W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    st_c = build_store(seed, **bkw)
    _load_weights(st_c, w_learned)
    cons = consolidate_by_hetero_replay_delayed(st_c, a.consol_steps, seed, seed_on=True,
                                                elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                                eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                                delay_steps=a.fwd_delay_steps, lam_dep=a.lam_dep,
                                                overlap_kw=overlap_kw, **cons_kw)
    w_consol = cons["w_after"]
    rev_fwd_ratio = cons["dw_rev"] / max(cons["dw_fwd"], 1e-6)
    out["consolidate"] = dict(dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"], rev_fwd_ratio=rev_fwd_ratio,
                              volley_overlap=cons.get("volley_overlap"), changed=cons["changed"])
    print(f"  [seed {seed}] CONSOLIDATE(hetero, lam_dep={a.lam_dep}): dw_fwd={cons['dw_fwd']:.2f} "
          f"dw_rev={cons['dw_rev']:.2f} rev/fwd={rev_fwd_ratio:.3f} volley_overlap={cons.get('volley_overlap')} "
          f"({time.time()-t0:.0f}s)", flush=True)

    rd_full_after = _read_graded(bkw, seed, w_consol, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac,
                                 swr_period=read_period, rest_steps=a.rest_steps, tag="full_after")
    rd_weak_after = _read_graded(bkw, seed, w_consol, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                 swr_period=read_period, rest_steps=a.rest_steps, tag="weak_after")
    band_after = measure_band_from(w_consol, st_c)
    out["band_after"] = band_after
    print(f"  [seed {seed}] AFTER: full depth_frac={rd_full_after['depth_frac']:.3f} | weak depth_frac="
          f"{rd_weak_after['depth_frac']:.3f} tau={rd_weak_after['tau']:.3f} band fwd={band_after['adj_fwd']:.1f} "
          f"rev={band_after['adj_rev']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    st_n = build_store(seed, **bkw)
    _load_weights(st_n, w_learned)
    cons_ns = consolidate_by_hetero_replay_delayed(st_n, a.consol_steps, seed, seed_on=False,
                                                   elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                                   eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                                   delay_steps=a.fwd_delay_steps, lam_dep=a.lam_dep, **cons_kw)
    w_noseed = cons_ns["w_after"]
    rd_weak_noseed = _read_graded(bkw, seed, w_noseed, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="weak_noseed")
    out["no_seed"] = dict(dw_fwd=cons_ns["dw_fwd"], dw_rev=cons_ns["dw_rev"],
                          weak_depth_frac=rd_weak_noseed["depth_frac"], weak_tau=rd_weak_noseed["tau"])
    print(f"  [seed {seed}] NO-SEED(lesion-replay): dw_fwd={cons_ns['dw_fwd']:.3f} weak depth_frac="
          f"{rd_weak_noseed['depth_frac']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    out["reads"] = dict(full_before=rd_full_before, weak_before=rd_weak_before,
                        full_after=rd_full_after, weak_after=rd_weak_after)

    # ============ PER-SEED VERDICT ============
    dw_fwd = cons["dw_fwd"]; dw_rev = cons["dw_rev"]; dw_ns = cons_ns["dw_fwd"]
    directional = ((dw_fwd - dw_rev) >= a.dw_min)
    rev_suppressed = (rev_fwd_ratio <= a.rev_ratio_max)
    headroom = (rd_weak_before["depth_frac"] <= a.headroom_max)
    depth_gain = ((rd_weak_after["depth_frac"] - rd_weak_before["depth_frac"]) >= a.depth_gain_min)
    tau_gain = ((rd_weak_after["tau"] - rd_weak_before["tau"]) >= a.tau_gain_min)
    recall_gain = bool(depth_gain or tau_gain)
    lesion_controlled = (abs(dw_ns) <= a.noseed_max_frac * max(abs(dw_fwd), 1e-6)
                         and (rd_weak_noseed["depth_frac"] <= rd_weak_before["depth_frac"] + a.depth_gain_min))
    seed_go = bool(directional and rev_suppressed and headroom and recall_gain and lesion_controlled)
    out["checks"] = dict(directional=directional, rev_suppressed=rev_suppressed, headroom=headroom,
                         depth_gain=depth_gain, tau_gain=tau_gain, recall_gain=recall_gain,
                         lesion_controlled=lesion_controlled, dw_fwd=round(dw_fwd, 3), dw_rev=round(dw_rev, 3),
                         rev_fwd_ratio=round(rev_fwd_ratio, 3), dw_noseed=round(dw_ns, 3),
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
    # the ESTABLISHED directional write (IDENTICAL to the graded-recall NO-GO's decisive 6-seed cfg -- unchanged)
    ap.add_argument("--btsp-elig-tau", type=float, default=80.0)
    ap.add_argument("--btsp-plat-tau", type=float, default=1.0)
    ap.add_argument("--btsp-eta", type=float, default=0.001)
    ap.add_argument("--btsp-w-max", type=float, default=900.0)
    ap.add_argument("--fwd-delay-steps", type=int, default=90)
    # NEW: the heterosynaptic-depression coefficient (0.0 = OFF = the established write, byte-identical)
    ap.add_argument("--lam-dep", type=float, default=0.0)
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
    ap.add_argument("--read-swr-period", type=int, default=0)
    # detection
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--active-frac", type=float, default=0.10)
    ap.add_argument("--onset-frac", type=float, default=0.06)
    # GO thresholds
    ap.add_argument("--dw-min", type=float, default=5.0)
    ap.add_argument("--rev-ratio-max", type=float, default=0.80, help="dw_rev/dw_fwd must be <= this (below the "
                    "NO-GO's ~0.84 pure-potentiation ratio) for rev_suppressed to hold")
    ap.add_argument("--headroom-max", type=float, default=0.90)
    ap.add_argument("--depth-gain-min", type=float, default=0.05)
    ap.add_argument("--tau-gain-min", type=float, default=0.05)
    ap.add_argument("--noseed-max-frac", type=float, default=0.20)
    # instrument verification (reused unmodified from the graded-recall runner)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verify-cue-mults", type=float, nargs="+", default=[1.0, 0.85, 0.7, 0.5, 0.35, 0.2])
    ap.add_argument("--verify-min-range", type=float, default=0.15)
    # modes
    ap.add_argument("--byte-identical-check", action="store_true", help="run ONLY the byte-identical-off "
                    "hash comparison (lam_dep=0.0 new-write vs the established write), skip everything else")
    ap.add_argument("--scan-lam-dep", action="store_true", help="run ONLY a single-seed lam_dep scan, skip the "
                    "decisive multi-seed test")
    ap.add_argument("--scan-mults", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.5])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--scan-out", default=str(SCAN_OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    print(f"[hetero-ltu] Ecker AdEx CA3 REVERSE-EDGE HETEROSYNAPTIC-DEPRESSION learn-through-use | "
          f"write=btsp_hetero+delay elig_tau={a.btsp_elig_tau} plat_tau={a.btsp_plat_tau} eta={a.btsp_eta} "
          f"fwd_delay={a.fwd_delay_steps}steps lam_dep={a.lam_dep} | n_mem={a.n_mem} asm={a.asm_size} | "
          f"swr={a.swr_period} seeds={a.seeds} backend={backend}", flush=True)

    if a.byte_identical_check:
        rows = [byte_identical_check(s, a) for s in a.seeds]
        all_exact = all(r["exact_hash_match"] for r in rows)
        print(f"[hetero-ltu] BYTE-IDENTICAL-OFF: {'CONFIRMED (exact hash match)' if all_exact else 'NOT exact -- see max_abs_diff'} "
              f"on {sum(r['exact_hash_match'] for r in rows)}/{len(rows)} seeds", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(str(a.out) + ".byte_identical_check.json").write_text(json.dumps(rows, indent=2, default=str))
        return 0 if all_exact else 1

    if a.scan_lam_dep:
        result = scan_lam_dep(a.seeds[0], a, a.scan_mults)
        Path(a.scan_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.scan_out).write_text(json.dumps(dict(seeds=a.seeds, cfg=vars(a), **result), indent=2, default=str))
        print(f"[hetero-ltu] wrote {a.scan_out}", flush=True)
        return 0

    verify = None
    if not a.skip_verify:
        verify = verify_instrument(a.seeds[0], a)
        if not verify["graded"]:
            print("[hetero-ltu] instrument validation failed (unexpected -- reused unmodified from the graded-"
                  "recall runner); aborting.", flush=True)
            return 1

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
        mratio = float(np.mean([p["consolidate"]["rev_fwd_ratio"] for p in per]))
        mdwns = float(np.mean([p["no_seed"]["dw_fwd"] for p in per]))
        mwdf_b = float(np.mean([p["reads"]["weak_before"]["depth_frac"] for p in per]))
        mwdf_a = float(np.mean([p["reads"]["weak_after"]["depth_frac"] for p in per]))
        mwdf_ns = float(np.mean([p["no_seed"]["weak_depth_frac"] for p in per]))
        mwtau_b = float(np.mean([p["reads"]["weak_before"]["tau"] for p in per]))
        mwtau_a = float(np.mean([p["reads"]["weak_after"]["tau"] for p in per]))
        n_headroom = sum(1 for p in per if p["checks"]["headroom"])
        n_directional = sum(1 for p in per if p["checks"]["directional"])
        n_rev_suppressed = sum(1 for p in per if p["checks"]["rev_suppressed"])
        n_lesion_ok = sum(1 for p in per if p["checks"]["lesion_controlled"])
        if go:
            verdict = (f"REVERSE-EDGE HETEROSYNAPTIC-DEPRESSION GO {n_go}/{len(per)} (lam_dep={a.lam_dep}) -- the "
                       f"write stays directional (dw_fwd {mdwf:.1f} vs dw_rev {mdwr:.1f}, {n_directional}/{len(per)}) "
                       f"AND the reverse/forward ratio drops to {mratio:.3f} ({n_rev_suppressed}/{len(per)} below "
                       f"the NO-GO's ~0.84 pure-potentiation ratio) AND weak-cue recall now GAINS: depth_frac "
                       f"{mwdf_b:.3f}->{mwdf_a:.3f} (tau {mwtau_b:.3f}->{mwtau_a:.3f}), headroom held "
                       f"{n_headroom}/{len(per)}, lesion-null {n_lesion_ok}/{len(per)} (dw_fwd_noseed {mdwns:.2f}~0). "
                       f"=> converts the graded-recall NO-GO to GO via heterosynaptic depression on reverse edges.")
        else:
            verdict = (f"REVERSE-EDGE HETEROSYNAPTIC-DEPRESSION NO-GO {n_go}/{len(per)} (lam_dep={a.lam_dep}) -- "
                       f"directional {n_directional}/{len(per)} (dw_fwd {mdwf:.1f} vs dw_rev {mdwr:.1f}, ratio "
                       f"{mratio:.3f}, rev_suppressed {n_rev_suppressed}/{len(per)}), headroom {n_headroom}/{len(per)} "
                       f"(weak depth_frac_before {mwdf_b:.3f}), but recall depth_frac {mwdf_b:.3f}->{mwdf_a:.3f} "
                       f"(tau {mwtau_b:.3f}->{mwtau_a:.3f}) does not clear the gain bar on enough seeds; lesion-null "
                       f"{n_lesion_ok}/{len(per)}. => a GENUINE negative for this lam_dep on this substrate.")
        v = Verdict("Ecker AdEx CA3: does heterosynaptic depression on reverse edges (fused_btsp_hetero_update, "
                    "lam_dep>0) restore a weak-cue GRADED recall GAIN that pure-potentiation BTSP could not?")
        v.require("the GRADED instrument reads graded on a known-good store (pre-flight verify_instrument, reused)",
                  bool(verify is None or verify["graded"]), expect=True)
        v.require("weak-cue depth_frac BEFORE has headroom (not at ceiling) on >= bar seeds", n_headroom,
                  expect=lambda x, b=bar: x >= b)
        v.require("the write is DIRECTIONAL (dw_fwd > dw_rev + dw_min) on >= bar seeds", n_directional,
                  expect=lambda x, b=bar: x >= b)
        v.require("the reverse/forward ratio is SUPPRESSED below the NO-GO's ~0.84 pure-potentiation ratio on "
                  ">= bar seeds", n_rev_suppressed, expect=lambda x, b=bar: x >= b)
        v.control("LESION-THE-REPLAY: seeded forward-deepening vs NO-SEED forward-deepening -- must DIFFER",
                  treatment=mdwf, control=mdwns, min_separation=0.0)
        v.disabled("within-assembly recurrence + assembly identity; ONLY the inter-assembly SEQUENCE band is "
                   "plastic (same scope as the established write)",
                   why="scope: reuses the established write's scaffold unmodified -- only the per-edge UPDATE "
                       "kernel (fused_btsp_hetero_update, an already-existing sim/kernels.py function) changes")
        decided = v.decide(go=go, verbose=False)
        attributable_to("weak-cue depth_frac gain (seeded hetero-replay vs NO-SEED lesion-the-replay)",
                        mwdf_a - mwdf_b, mwdf_ns - mwdf_b)
        summary_extra = dict(GO=go, n_go=n_go, status=decided.get("status"), lam_dep=a.lam_dep,
                             dw_fwd=mdwf, dw_rev=mdwr, rev_fwd_ratio=mratio, dw_fwd_noseed=mdwns,
                             weak_depth_frac_before=mwdf_b, weak_depth_frac_after=mwdf_a,
                             weak_depth_frac_noseed=mwdf_ns, weak_tau_before=mwtau_b, weak_tau_after=mwtau_a,
                             n_headroom=n_headroom, n_directional=n_directional, n_rev_suppressed=n_rev_suppressed,
                             n_lesion_ok=n_lesion_ok, instrument_verify=verify,
                             preconditions=decided.get("preconditions", []), decided=decided)
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_go=0, instrument_verify=verify)

    summary = {"probe": "gap5_reverse_edge_hetero_depression_ltu",
               "mechanism": "heterosynaptic DEPRESSION / competitive normalization on reverse edges "
                            "(sim.kernels.fused_btsp_hetero_update, an EXISTING kernel) applied to the established "
                            "BTSP+forward-conduction-delay directional replay write",
               "seeds": a.seeds, "n_mem": a.n_mem, "asm_size": a.asm_size,
               "cfg": vars(a),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[hetero-ltu] VERDICT: {verdict}\n[hetero-ltu] wrote {a.out}\n" + "=" * 120,
          flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
