"""gap#5 READOUT — the SWR-STATE E/I-TRANSIENT ENVELOPE replay (Option 1 diagnostic + Option 2 build).

2026-07-24 (research gate research/findings/2026-07-24-gap5-SWR-state-readout-research-gate-the-missing-biology.md).
The 5-method readout boundary was a brain-STATE error: every prior readout drove a THETA-paced basket-disinhibition
SWEEP, but offline sequence replay is a SHARP-WAVE-RIPPLE (SWR) phenomenon -- a DIFFERENT interneuron regime. The
cited, point-neuron-realizable ignition mechanism (Buzsaki Rhythms L7076/L14400-14574; Ecker 2022 eLife e71850):
recurrent E transiently OUTRUNS feedback-I -> a 3-5x gain, ~100ms envelope, self-terminating when I catches up.

This runner reads the frozen DECOUPLED forward-asymmetric store (within ~206 + adj_fwd ~38 / adj_rev ~5, 6/6-GO WEIGHT
store) as a self-organized SWR replay. NO `sim/` edit (all external current + gate toggles + config knobs; reuse-by-
import of the store builder, the RANK-1 rest/silence/OU/poisson helpers, the ordered-replay diagnostic, and the
between-edge weight-lesion controls).

OPTION 1 (--option1): the ~20-min config diagnostic. On the decoupled store, cue assembly-0 at the COMPLETION's igniting
op-point (sustained recall_drive~700 for ~150 steps + self_regen_read=0.15 re-latch + recall_k_thresh~110) and read
per_asm_active. Disambiguates: does assembly-0 IGNITE at the completion op-point (=> ignition IS achievable; the wall
was op-point/state, not a substrate wall)? Prediction: [0] ignites but likely [1,0,0]/[3,3,3] (no forward hand-off).

OPTION 2 (default): the real build. On the frozen decoupled store -- (1) REST in the SWR state: bistable silent
down-state (self_regen_read) + `_hard_silence` reset + WEAK NON-SPECIFIC noise (RANK-1 Poisson/OU) as the self-organized
ignition SEED (NOT a targeted detonator, NOT theta pacing); (2) impose a TRANSIENT E>I ENVELOPE (~100-200ms: transiently
DROP `ca3_pv_basket` feedback-I AND add broad weak excitation to CA3-exc so recurrent E outruns I -> the most-excitable
assembly ignites and its gain-amplified forward-asymmetric links carry A->B->C; reuse run_swr_replay_phase's ~100/50ms
envelope timing); (3) SELF-TERMINATE: re-raise the basket + SFA (`d_abs`/`a_abs`) -> burst ends -> rest; (4) order
within the envelope by TIMING (SFA post-fire self-avoidance over the real chain; NEVER STD/fatigue the chain weights).

GO GATE (verify, don't assert -- the runner PRINTS its verdict; the caller reads THAT line): per_asm_active ~[1,1,1]
(NOT [3,3,3] co-fire, NOT [0,0,0] no-ignition) AND forward_frac >= 1.5x chance & forward > reverse AND the net RESTS
silent between discrete events (low duty cycle), >=5/6 seeds on the full run.
Anti-cheats (each WIRED AND INVOKED; a control written-but-never-called is the silent-failure mode):
  (1) NO-SWR (constant E/I, no transient) -> collapse.                    [the transient gain is load-bearing]
  (2) SHUFFLED-STORE (permute between-edge multiset) -> order collapses.  [the learned chain carries the ORDER]
  (3) REVERSE-ASYM-LESION (symmetrize between-edges -> adj_fwd==adj_rev)  [the forward WEIGHT ASYMMETRY is load-bearing;
       -> forward direction destroyed.                                     covers the "reverse" control]
  (4) PERMUTED-ASSEMBLY (re-score with random assembly labels) -> chance. [the order is real, not a labeling artifact]
  (5) NO-NOISE acid (envelope on, noise off) -> no specific forward events.[retires the self-sustaining/deterministic
       envelope-detonator confound: the noise is the SEED, the envelope is the GAIN]
  (6) NO-ENCODE (fresh bridge, store skipped, same envelope+noise) -> no specific events. [retires the noise artifact]
  (7) ADAPT-LESION (d_abs->0, no SFA) -> [3,3,3] co-fire (Ecker control).  [SFA self-avoidance drives the hand-off]
  (8) FROZEN-plasticity byte-hash (cp_connections.data unchanged every arm).[order rides the frozen chain, not re-encode]
  (9) NUMPY-REFERENCE GUARD (NO host per-step per-assembly silence / argmax in the loop -- the order EMERGES from the
       substrate's own envelope + weights + SFA; verified by construction: the rest loop only injects external current).

CPU-smoke (proves it RUNS + all arms live + a verdict; NOT a GO claim -- the store completes at n_ca3=2000):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_swr_envelope_replay_derisk \
      --seeds 42 --n-ca3 800 --rest-steps 500 --swr-period 200 --env-dur 100
Full run (GPU; the 3090): SIM_BACKEND=cupy nice -n 10 .venv/bin/python -m research.runners._gap5_swr_envelope_replay_derisk \
      --seeds 42 43 44 100 101 102 --n-ca3 2000 --rest-steps 1500
Option 1 diagnostic: SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_swr_envelope_replay_derisk \
      --option1 --seeds 42 --n-ca3 2000
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

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

from sim.backend import get_backend, to_host  # noqa: E402
# the DECOUPLED forward-asymmetric encode (6/6-GO WEIGHT store) + its config + the ordered-replay diagnostic + the
# between-edge weight-lesion controls (reuse-by-import; NO `sim/` edit anywhere)
from research.runners._gap5_sequence_replay_derisk import (  # noqa: E402
    _prepare_sequence, _detect_sequence_events, _scramble_between_weights, _symmetrize_between_weights,
    _event_windows, _smooth,
)
from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG  # noqa: E402
# the RANK-1 rest building blocks (freeze/silence/OU) reused verbatim + the vectorized ca3->ca3 CSR extractor (used to
# recompute the FORWARD-LINK flat indices for the transient SWR gain -- same classification as _prepare_sequence)
from research.runners._gap5_spontaneous_reactivation_derisk import (  # noqa: E402
    _hard_silence, _configure_ou, _extract_ca3ca3_vec,
)

OUT = _REPO / "research" / "findings" / "raw" / "gap5_r4" / "swr_envelope_replay.json"


# ----------------------------------------------------------------------------------------------------------------------
# shared setup: freeze plasticity, hard-silence (verify dendritic reset), configure SFA on CA3-exc, resolve region index
# handles. Returns a dict of device handles used by both the Option-1 cued readout and the Option-2 envelope readout.
# ----------------------------------------------------------------------------------------------------------------------
def _setup_read(prep, seed, *, self_regen_read, recall_k_thresh, d_abs, a_abs, adapt):
    cp, _ = get_backend()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    assert bridge.core_config.enable_hebbian_learning is False
    bridge.core_config.coincidence_plateau_self_regen = float(self_regen_read)
    if recall_k_thresh is not None:
        bridge.core_config.coincidence_k_threshold = float(recall_k_thresh)

    _hard_silence(bridge)
    apical_max = None; n_latched = 0
    if getattr(bridge, "cp_v_apical", None) is not None:
        va = np.asarray(to_host(bridge.cp_v_apical))[prep["ca3_arr_host"][prep["assembly_local"]]]
        apical_max = float(np.max(va)); n_latched = int((va > float(DECOUPLED_CFG["plateau_v_hold"])).sum())

    _configure_ou(bridge, None, seed)   # OU off by default (Option-2 poisson path re-enables per-step manually)

    ca3_arr_host = prep["ca3_arr_host"]
    exc_glob = ca3_arr_host[prep["ca3_exc_local"]]
    exc_dev = cp.asarray(exc_glob, dtype=cp.int64)
    rm = bridge.region_manager
    basket_glob = None; basket_n = 0
    try:
        _b = np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)
        basket_glob = cp.asarray(_b, dtype=cp.int64); basket_n = int(len(_b))
    except Exception:
        basket_glob = None

    # crank the spike-frequency adaptation that turns co-firing into a forward sweep (Ecker 2022). adapt=False = the
    # ADAPT-LESION control. Izhikevich: per-neuron u-kick cp_izh_d_increment + recovery a on CA3-exc. AdEx: GLOBAL
    # w-adaptation cfg.adex_b (spike-triggered) + cfg.adex_a (subthreshold) -- no per-neuron array, so the lesion zeros
    # them globally (still a valid adapt-lesion: removes ALL w-adaptation). adapt=True keeps the band's built AdEx regime.
    _adex_read = getattr(bridge, "cp_adex_w", None) is not None
    if _adex_read:
        _cfg = bridge.core_config
        if not adapt:                                  # ADAPT-LESION: remove AdEx w-adaptation entirely
            _cfg.adex_b = 0.0; _cfg.adex_a = 0.0
        # adapt=True -> leave cfg.adex_b / adex_a / adex_tau_w as built (the Ecker RS-pyramidal regime), unless a runner
        # explicitly re-tuned them via the build's adex_params. (d_abs/a_abs are Izhikevich-scale and not applied to AdEx.)
    elif adapt and getattr(bridge, "cp_izh_d_increment", None) is not None:
        bridge.cp_izh_d_increment[exc_dev] = cp.float32(d_abs)
        bridge.cp_izh_a[exc_dev] = cp.float32(a_abs)

    return dict(bridge=bridge, cp=cp, ca3_arr_host=ca3_arr_host, exc_glob=exc_glob, exc_dev=exc_dev,
                basket_glob=basket_glob, basket_n=basket_n, apical_rest_max=apical_max, apical_n_latched=n_latched)


# ----------------------------------------------------------------------------------------------------------------------
# OPTION 1 (diagnostic): cue assembly-0 at the COMPLETION's igniting op-point (sustained recall_drive onto assembly-0 +
# bistable re-latch self_regen_read + recall_k_thresh) and read per_asm_active. Purpose = disambiguation only.
# ----------------------------------------------------------------------------------------------------------------------
def _option1_cued_ignition(prep, seed, *, recall_drive, recall_steps, self_regen_read, recall_k_thresh,
                           det_frac, tail_steps, det):
    h = _setup_read(prep, seed, self_regen_read=self_regen_read, recall_k_thresh=recall_k_thresh,
                    d_abs=0.0, a_abs=0.0, adapt=False)   # NO SFA -- a pure "does it ignite/latch" test
    bridge, cp = h["bridge"], h["cp"]
    ca3_arr_host = h["ca3_arr_host"]
    a0_loc = prep["assemblies_local"][0]
    drng = np.random.default_rng(int(seed) * 4242 + 1)
    k = max(1, int(round(det_frac * len(a0_loc))))
    sel_loc = np.sort(drng.choice(a0_loc, min(k, len(a0_loc)), replace=False))
    cue_dev = cp.asarray(ca3_arr_host[sel_loc], dtype=cp.int64)

    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    T = int(recall_steps) + int(tail_steps)
    F = np.zeros((T, len(ca3_arr_host)), dtype=bool)
    for t in range(T):
        bridge.cp_external_input_current[:] = 0.0
        if t < recall_steps:
            bridge.cp_external_input_current[cue_dev] += float(recall_drive)   # SUSTAINED cue onto assembly-0
        bridge._run_one_simulation_step()
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    frozen = bool(np.array_equal(w_before, w_after))
    seq = _detect_sequence_events(F, prep["assemblies_local"], **det)
    # per-assembly mean active fraction over the whole window (an ignition read independent of event windowing)
    per_asm_frac = [float(F[:, prep["assemblies_local"][k2]].mean()) for k2 in range(len(prep["assemblies_local"]))]
    return dict(per_asm_active=seq["per_asm_active"], per_asm_peak=seq["per_asm_peak"], per_asm_frac=per_asm_frac,
                n_events=seq["n_events"], n_multi=seq["n_multi"], forward_frac=seq["forward_frac"],
                reverse_frac=seq["reverse_frac"], duty_cycle=seq["duty_cycle"], pop_rate=seq["pop_rate"],
                weights_frozen=frozen, apical_rest_max=h["apical_rest_max"], apical_n_latched=h["apical_n_latched"],
                k_cue=int(len(sel_loc)))


# ----------------------------------------------------------------------------------------------------------------------
# FORWARD-LINK GAIN (2026-08-20 build): the missing SWR ingredient. Buzsaki Rhythms L14452 -- during the SWR envelope the
# transient E>I imbalance GAIN-AMPLIFIES the learned forward-asymmetric links (a "three- to fivefold gain") so the ignited
# assembly's now-strong forward links carry A->B->C, then it restores. The frozen decoupled store has adj_fwd~=38, but the
# ordered-handoff band is ~114-190 (Section B of the 2026-07-24 finding) and NO prior knob lifted it there -- so the
# ignited assembly ignited in isolation and never handed off (baseline: per_asm_active[1,1,1] but forward_frac 0).
#
# This helper recomputes the FLAT CSR-data indices of the between-assembly recurrent edges, split by DIRECTION/DISTANCE,
# for a TRANSIENT multiplicative gain within the envelope (cached on prep). It reuses _prepare_sequence's EXACT edge
# classification (coincidence-masked ca3->ca3 + asm_of_local positional labels). CRITICAL anti-cheat property: position
# only SELECTS a scope; the AMOUNT is a multiplicative gain on the STORED weight (w -> w*g). So a symmetrized store
# (REVERSE-ASYM-LESION, adj_fwd==adj_rev) or shuffled store carries NO direction even under the gain -> forward collapses.
# The default scope "adjacent" is distance-1 BOTH directions (position-SYMMETRIC) -> the gain cannot manufacture direction
# from a symmetrized store; the learned WEIGHT asymmetry is the sole source of order (exactly the lesion the gate demands).
# ----------------------------------------------------------------------------------------------------------------------
def _between_edge_classes(prep):
    if prep.get("_edge_classes") is not None:
        return prep["_edge_classes"]
    cp, _ = get_backend()
    bridge = prep["bridge"]
    ca3_idx = prep["ca3_idx"]
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    asm_of_local = np.full(len(ca3_idx), -1, dtype=np.int64)
    for m, a in enumerate(prep["assemblies"]):
        asm_of_local[np.asarray([ca3_pos[int(g)] for g in a], dtype=np.int64)] = m
    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_vec(bridge, ca3_idx, to_host)
    a_pre = asm_of_local[pre_l_h]; a_post = asm_of_local[post_l_h]
    between = (a_pre >= 0) & (a_post >= 0) & (a_pre != a_post)
    adj_fwd = between & (a_post == a_pre + 1)
    adj_rev = between & (a_post == a_pre - 1)

    def _dev(mask):
        f = flat_h[mask].astype(np.int64)
        return cp.asarray(f, dtype=cp.int64) if f.size else None
    classes = dict(
        adjacent=_dev(adj_fwd | adj_rev),            # distance-1 BOTH directions (position-symmetric -> lesion-safe)
        between=_dev(between),                        # all between edges (position-symmetric multiset -> lesion-safe)
        forward=_dev(between & (a_post > a_pre)),     # forward-position ONLY (diagnostic; position-ASYMMETRIC -> the
                                                      #   REVERSE-ASYM-LESION gate WILL catch it if it fabricates order)
        adj_fwd=_dev(adj_fwd), adj_rev=_dev(adj_rev),
        n_adj_fwd=int(adj_fwd.sum()), n_adj_rev=int(adj_rev.sum()), n_between=int(between.sum()))
    prep["_edge_classes"] = classes
    return classes


# ----------------------------------------------------------------------------------------------------------------------
# OPTION 2 (the build): the SWR-state E/I-transient envelope readout. mode in {"swr","no_swr"}. noise as the self-
# organized ignition SEED; the envelope as the transient E>I GAIN; SFA for the forward hand-off + self-termination.
# NO host per-step per-assembly silence / argmax (numpy-reference guard: the loop only injects external current).
# fwd_gain: the TRANSIENT forward-link gain (w->w*fwd_gain on the fwd_gain_scope edges, ONLY within the envelope window;
#   restored to baseline in the inter-event rest AND -- guaranteed -- after the loop, so the byte-hash is unchanged).
# ----------------------------------------------------------------------------------------------------------------------
def _rest_swr_envelope(prep, rest_steps, seed, *, mode, noise_on, env_exc_pa, env_basket_drop, env_basket_boost,
                       swr_period, env_dur, noise_rate, noise_pa, noise_dur, self_regen_read, recall_k_thresh,
                       d_abs, a_abs, adapt, self_regen_ignite=None, ignite_frac=0.4, env_exc_ramp=False,
                       fwd_gain=1.0, fwd_gain_scope="adjacent", env_exc_ramp_floor=0.5,
                       seed_assembly=False, seed_pa=0.0, seed_dur=0, seed_frac=0.5, fixed_seed_asm=None,
                       verbose=False):
    # MECHANISM #3 (per-envelope single-assembly SEED = Diba-Buzsaki SWR initiation): when seed_assembly, each envelope a
    # RANDOM assembly k (NOT always 0) gets a WEAK sub-detonator pulse (seed_pa to seed_frac of its cells) for the first
    # seed_dur steps -> k ignites FIRST; the LEARNED forward links + SFA must then carry k->k+1->k+2 (forward FROM the
    # seed). The random k is the LOAD-BEARING anti-cheat: forward order that only appears for k=0 = host-imposed = a cheat.
    # env_seed_log records k per envelope so the scorer measures forward-FROM-seed (not absolute 0->1->2).
    # MECHANISM #1 (latch-then-release): when self_regen_ignite is not None, the plateau self-regen is HIGH (a bistable
    # LATCH) for the first ignite_frac of each envelope so a WEAK noise seed SELECTIVELY ignites+latches ONE assembly
    # (RANK-1: weak-noise+latch = selective single-assembly ignition), then DROPS to self_regen_read (transient) for the
    # rest of the window so the latched bump DE-latches and hands off forward via the links + SFA. Default None = the
    # constant-self_regen behavior (byte-identical). env_exc_ramp: ramp env_exc 0->peak across the window (the depolarizing
    # RAMP, mechanism #2 -- most-excitable assembly crosses threshold FIRST -> sequences instead of co-fires).
    h = _setup_read(prep, seed, self_regen_read=self_regen_read, recall_k_thresh=recall_k_thresh,
                    d_abs=d_abs, a_abs=a_abs, adapt=adapt)
    bridge, cp = h["bridge"], h["cp"]
    _ignite_steps = int(round(float(ignite_frac) * float(env_dur)))
    ca3_arr_host = h["ca3_arr_host"]
    exc_dev, basket_glob = h["exc_dev"], h["basket_glob"]
    exc_glob = h["exc_glob"]

    duty = float(env_dur) / float(swr_period)
    # NO-SWR control = constant E/I: deliver the time-AVERAGED exc drive + a constant mid basket, no transient window.
    const_exc = env_exc_pa * duty
    const_basket = env_basket_boost * (1.0 - duty) - env_basket_drop * duty

    # poisson non-specific noise setup (CA3-EXC-targeted; deterministic host RNG; == RANK-1 _rest_and_detect)
    prng = np.random.default_rng(int(seed) * 100003 + 11)
    countdown = np.zeros(len(exc_glob), dtype=np.int32)

    # MECHANISM #3 seed setup: precompute a FIXED sub-sample (seed_frac) of each assembly's cells (device indices); a
    # SEPARATE per-envelope RNG picks which assembly k to seed each envelope (so the cell subsets stay fixed).
    seed_cells_dev = None; env_seed_log = []
    if seed_assembly:
        _als = prep["assemblies_local"]
        _cell_rng = np.random.default_rng(int(seed) * 314159 + 17)
        seed_cells_dev = []
        for a_loc in _als:
            _k = max(1, int(round(float(seed_frac) * len(a_loc))))
            _sub = np.sort(_cell_rng.choice(a_loc, min(_k, len(a_loc)), replace=False))
            seed_cells_dev.append(cp.asarray(ca3_arr_host[_sub], dtype=cp.int64))
        _choice_rng = np.random.default_rng(int(seed) * 271828 + 23)   # per-envelope RANDOM assembly choice
        _cur_seed_k = None

    if verbose:
        print(f"      [swr mode={mode} noise={noise_on} env_exc={env_exc_pa} basket_drop={env_basket_drop} "
              f"basket_boost={env_basket_boost} period={swr_period} env_dur={env_dur} self_regen={self_regen_read} "
              f"k_thresh={recall_k_thresh} adapt={adapt} basket_n={h['basket_n']}]", flush=True)

    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()

    # -- FORWARD-LINK GAIN setup (transient SWR weight amplification) -- captured AFTER w_before so the restore below is
    # byte-exact. _amp_base holds the scope's baseline stored weights (on device); _amp_mult tracks the currently-applied
    # multiplier so we write conn.data only at envelope boundaries (~2*n_env writes, not per step). In no_swr mode the
    # gain is applied CONSTANTLY at its duty-averaged value (same amplification budget, NO transient window) -- so if
    # NO-SWR collapses, the TRANSIENT (not merely the average gain) is load-bearing.
    _amp_idx = None; _amp_base = None; _amp_mult = 1.0; _amp_base_mean = None
    fwd_gain = float(fwd_gain)
    if abs(fwd_gain - 1.0) > 1e-9:
        _cls = _between_edge_classes(prep)
        _amp_idx = _cls.get(fwd_gain_scope)
        if _amp_idx is not None:
            _amp_base = bridge.cp_connections.data[_amp_idx].copy()
            _amp_base_mean = float(np.asarray(to_host(_amp_base)).mean())
    _const_gain = 1.0 + (fwd_gain - 1.0) * duty       # no_swr constant (duty-averaged) multiplier

    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    n_env = 0
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = 0.0
        # -- SWR envelope (the transient E>I gain) --
        if mode == "swr":
            phase_in = t % swr_period
            in_env = phase_in < env_dur
            if in_env:
                # latch-then-release schedule (mechanism #1): HIGH self-regen (latch) for the first _ignite_steps ->
                # selective ignition from a weak seed; then self_regen_read (release) -> de-latch + forward hand-off.
                if self_regen_ignite is not None:
                    bridge.core_config.coincidence_plateau_self_regen = (
                        float(self_regen_ignite) if phase_in < _ignite_steps else float(self_regen_read))
                # env_exc: flat, or a floor->peak ramp across the window (mechanism #2). BUGFIX 2026-08-20: the old ramp
                # went 0->peak, so env-exc was subthreshold for most of the window and KILLED ignition (every ramp config
                # -> [0,0,0]). Ramp now from env_exc_ramp_floor*peak (default 0.5) -> peak: still "most-excitable first"
                # (a rising depolarization) but never subthreshold. (Default env_exc_ramp off -> flat, byte-identical.)
                if env_exc_ramp:
                    _frac = float(phase_in + 1) / float(env_dur)
                    _ee = float(env_exc_pa) * (env_exc_ramp_floor + (1.0 - env_exc_ramp_floor) * _frac)
                else:
                    _ee = float(env_exc_pa)
                if _ee != 0.0:
                    bridge.cp_external_input_current[exc_dev] += _ee                     # broad weak exc: raise E
                if basket_glob is not None and env_basket_drop != 0.0:
                    bridge.cp_external_input_current[basket_glob] += -float(env_basket_drop)  # DROP feedback-I: I can't track E
                # MECHANISM #3: per-envelope single-assembly seed (random k, weak sub-detonator pulse for the first seed_dur)
                if seed_assembly:
                    if phase_in == 0:
                        # DIRECTIONAL ONSET CUE (2026-08-20): a fixed_seed_asm (e.g. 0 = the chain-start A) makes the
                        # onset prefix cue target the SAME assembly every envelope -> A ignites first -> its gain-amplified
                        # forward links carry A->B->C (forward mode). fixed_seed_asm=None keeps the RANDOM per-envelope
                        # choice (the rigorous forward-FROM-SEED anti-cheat: order that survives a random start rides the
                        # WEIGHT asymmetry, not the cue). The cue targets A's CELLS (a recall prefix), never the output order.
                        _cur_seed_k = (int(fixed_seed_asm) if fixed_seed_asm is not None
                                       else int(_choice_rng.integers(0, len(seed_cells_dev))))
                        env_seed_log.append(_cur_seed_k)
                    if phase_in < seed_dur and _cur_seed_k is not None and seed_pa != 0.0:
                        bridge.cp_external_input_current[seed_cells_dev[_cur_seed_k]] += float(seed_pa)
                if phase_in == 0:
                    n_env += 1
            else:
                if self_regen_ignite is not None:
                    bridge.core_config.coincidence_plateau_self_regen = float(self_regen_read)   # rest = release value
                if basket_glob is not None and env_basket_boost != 0.0:
                    bridge.cp_external_input_current[basket_glob] += float(env_basket_boost)  # RE-RAISE I: self-terminate -> rest
        elif mode == "no_swr":
            bridge.cp_external_input_current[exc_dev] += float(const_exc)                 # constant E, no transient window
            if basket_glob is not None:
                bridge.cp_external_input_current[basket_glob] += float(const_basket)
        # -- non-specific noise (the self-organized ignition SEED; a sparse suprathreshold volley to random CA3-exc) --
        if noise_on:
            new = prng.random(len(exc_glob)) < noise_rate
            countdown[new] = noise_dur
            active = countdown > 0
            if active.any():
                bridge.cp_external_input_current[exc_dev[cp.asarray(np.nonzero(active)[0], dtype=cp.int64)]] += float(noise_pa)
            countdown[active] -= 1
        # -- FORWARD-LINK GAIN (transient synaptic gain = the SWR E>I imbalance amplifying the learned forward chain).
        # Multiplicative w->w*g on the STORED weights of the scope edges. swr: g during the env window, 1.0 in the rest.
        # no_swr: the constant duty-averaged g throughout (no window). Written ONLY when the multiplier changes.
        if _amp_idx is not None:
            _want = (fwd_gain if (t % swr_period) < env_dur else 1.0) if mode == "swr" else _const_gain
            if _want != _amp_mult:
                bridge.cp_connections.data[_amp_idx] = _amp_base * _want
                _amp_mult = _want
        bridge._run_one_simulation_step()      # NO external per-assembly silence / argmax (numpy-reference guard)
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)

    # GUARANTEED transient restore: return the scope edges to their EXACT baseline before the byte-hash, regardless of
    # whether the loop ended inside or outside an envelope window (anti-cheat #8 FROZEN-plasticity must stay byte-exact).
    if _amp_idx is not None:
        bridge.cp_connections.data[_amp_idx] = _amp_base
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    frozen = bool(np.array_equal(w_before, w_after))
    amp = dict(fwd_gain=fwd_gain, scope=fwd_gain_scope,
               n_edges=(int(_amp_idx.size) if _amp_idx is not None else 0),
               base_mean=_amp_base_mean,
               peak_mean=(None if _amp_base_mean is None else _amp_base_mean * fwd_gain),
               const_gain=(_const_gain if mode == "no_swr" else None))
    return dict(F=F, weights_frozen=frozen, apical_rest_max=h["apical_rest_max"],
                apical_n_latched=h["apical_n_latched"], n_env=n_env, basket_n=h["basket_n"],
                env_seed_log=env_seed_log, amp=amp)


def _score_forward_from_seed(F, assemblies_local, env_seed_log, swr_period, W=5, ev_floor=0.4, ev_k=4.0,
                             active_frac=0.12, onset_frac=0.08, min_ev_len=4):
    """MECHANISM #3 scorer: forward-FROM-SEED. Each event is mapped to the envelope that contains its START (env_idx =
    s // swr_period) -> the seeded assembly k = env_seed_log[env_idx]. A multi-assembly event is FORWARD-from-seed if,
    ordering its active assemblies by onset, the seeded k has the EARLIEST onset AND the onset-sorted index sequence is
    strictly INCREASING and CONTIGUOUS from k (k, k+1, k+2, ...); REVERSE-from-seed if strictly DECREASING from k
    (k, k-1, ...). Because the store's only directional links are forward (adj_fwd 38 / adj_rev 5), a genuine cascade
    riding the learned links produces k->k+1->k+2 REGARDLESS of which k was seeded -- so forward>>reverse here, while it
    was seeded at RANDOM k, decisively rules out 'order = where I injected'. Also returns the per-seed-position breakdown."""
    import math
    T, nca3 = F.shape
    n_mem = len(assemblies_local)
    asizes = [max(1, len(a)) for a in assemblies_local]
    asize_ref = float(np.mean(asizes))
    events, duty, pop_rate = _event_windows(F, W=W, ev_floor=ev_floor, ev_k=ev_k, asize_ref=asize_ref)
    per_asm_active = [0] * n_mem
    n_multi = fwd = rev = seed_first = 0
    chance_terms = []
    by_seedpos = {k: {"multi": 0, "fwd": 0, "rev": 0} for k in range(n_mem)}
    for (s, e) in events:
        if e - s < min_ev_len:
            continue
        env_idx = s // swr_period
        if env_idx >= len(env_seed_log):
            continue
        k = int(env_seed_log[env_idx])
        active = []
        for kk, A in enumerate(assemblies_local):
            a_t = _smooth(F[s:e][:, A].sum(1), W) / asizes[kk]
            if a_t.size and float(a_t.max()) >= active_frac:
                per_asm_active[kk] += 1
                cross = np.nonzero(a_t >= onset_frac)[0]
                onset = float(cross[0]) if cross.size else float(np.argmax(a_t))
                active.append((kk, onset + 1e-3 * float(np.argmax(a_t))))
        if len(active) < 2:
            continue
        n_multi += 1
        chance_terms.append(1.0 / math.factorial(len(active)))
        order = [kk for kk, _ in sorted(active, key=lambda kv: kv[1])]     # assembly indices sorted by onset
        by_seedpos[k]["multi"] += 1
        if order[0] == k:
            seed_first += 1
        is_fwd = (order[0] == k) and all(order[i + 1] == order[i] + 1 for i in range(len(order) - 1))
        is_rev = (order[0] == k) and all(order[i + 1] == order[i] - 1 for i in range(len(order) - 1))
        if is_fwd:
            fwd += 1; by_seedpos[k]["fwd"] += 1
        if is_rev:
            rev += 1; by_seedpos[k]["rev"] += 1
    return dict(n_events=len(events), n_multi=n_multi, duty_cycle=duty, pop_rate=pop_rate,
                per_asm_active=per_asm_active,
                forward_from_seed=(fwd / n_multi) if n_multi else 0.0,
                reverse_from_seed=(rev / n_multi) if n_multi else 0.0,
                seed_first_frac=(seed_first / n_multi) if n_multi else 0.0,
                chance=float(np.mean(chance_terms)) if chance_terms else 0.0,
                by_seedpos={k: dict(v) for k, v in by_seedpos.items()},
                n_seed_envs=len(env_seed_log))


def _seed_scored_to_std(sc, n_mem):
    """Remap a _score_forward_from_seed result onto the _detect_sequence_events key contract (forward_frac=forward-FROM-
    seed, reverse_frac=reverse-from-seed, chance_forward=per-event 1/n! chance) so the same verdict logic scores both the
    absolute-forward (no onset cue) and the forward-from-seed (onset cue) regimes. n_full/mean_tau/per_asm_peak are not
    defined for the seed scorer -> filled with neutral defaults (unused by the onset-cue verdict path)."""
    return dict(n_events=sc["n_events"], n_multi=sc["n_multi"], n_full=0,
                forward_frac=sc["forward_from_seed"], reverse_frac=sc["reverse_from_seed"],
                forward_frac_full=0.0, reverse_frac_full=0.0, mean_tau=0.0,
                chance_forward=sc["chance"], duty_cycle=sc["duty_cycle"], pop_rate=sc["pop_rate"],
                per_asm_active=sc["per_asm_active"], per_asm_peak=[0.0] * n_mem,
                seed_first_frac=sc["seed_first_frac"], by_seedpos=sc["by_seedpos"], n_seed_envs=sc["n_seed_envs"])


def _score_arm(r, assemblies_local, det, swr_period, onset_on):
    """Score one readout arm. With the DIRECTIONAL ONSET CUE on, forward = forward-FROM-SEED (each event scored relative
    to the assembly the cue seeded that envelope, from env_seed_log) so a fixed-A cue reads A->B->C AND a random-start cue
    reads k->k+1->k+2 -- both riding the learned forward links. With the cue off, forward = absolute 0->1->2."""
    F = r["F"]
    if onset_on:
        sc = _score_forward_from_seed(F, assemblies_local, r["env_seed_log"], swr_period,
                                      W=det["W"], ev_floor=det["ev_floor"], ev_k=det["ev_k"],
                                      active_frac=det["active_frac"], onset_frac=det["onset_frac"])
        return _seed_scored_to_std(sc, len(assemblies_local))
    return _detect_sequence_events(F, assemblies_local, **det)


def _permuted_assembly_score(F, assemblies_local, seed, det, env_seed_log=None, swr_period=None, onset_on=False):
    """PERMUTED-ASSEMBLY anti-cheat: re-score the SAME rest firing with a random permutation of the assembly LABELS ->
    forward_frac collapses to chance (the ordered structure is real, not a labeling artifact). Deterministic per seed.
    In the onset-cue regime the SAME env_seed_log is kept while the labels are permuted, so the seeded position k now
    points to a physically-unrelated assembly -> forward-from-seed contiguity breaks -> chance (the intended control)."""
    perm = np.random.default_rng(int(seed) * 5150 + 3).permutation(len(assemblies_local))
    relabeled = [assemblies_local[i] for i in perm]
    if onset_on:
        sc = _score_forward_from_seed(F, relabeled, env_seed_log, swr_period, W=det["W"], ev_floor=det["ev_floor"],
                                      ev_k=det["ev_k"], active_frac=det["active_frac"], onset_frac=det["onset_frac"])
        return _seed_scored_to_std(sc, len(relabeled))
    return _detect_sequence_events(F, relabeled, **det)


def _weight_diag(prep):
    return dict(w_within=prep["w_within"], w_adj_fwd=prep.get("w_adj_fwd"), w_adj_rev=prep.get("w_adj_rev"),
                ratio_adj=(float(prep.get("w_adj_fwd", 0.0)) / max(abs(float(prep.get("w_adj_rev", 0.0))), 1e-6)),
                n_between_fwd=prep.get("n_between_fwd"), n_between_rev=prep.get("n_between_rev"),
                assembly_sizes=[int(len(a)) for a in prep["assemblies"]])


def one_seed(seed, cfg, a):
    t0 = time.time()
    out = {"seed": seed}
    det = dict(W=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k, active_frac=a.active_frac, onset_frac=a.onset_frac)
    env_kw = dict(env_exc_pa=a.env_exc_pa, env_basket_drop=a.env_basket_drop, env_basket_boost=a.env_basket_boost,
                  swr_period=a.swr_period, env_dur=a.env_dur, noise_rate=a.noise_rate, noise_pa=a.noise_pa,
                  noise_dur=a.noise_dur, self_regen_read=a.self_regen_read, recall_k_thresh=a.recall_k_thresh,
                  d_abs=a.d_abs, a_abs=a.a_abs, self_regen_ignite=a.self_regen_ignite, ignite_frac=a.ignite_frac,
                  env_exc_ramp=a.env_exc_ramp, env_exc_ramp_floor=a.env_exc_ramp_floor,
                  fwd_gain=a.fwd_gain, fwd_gain_scope=a.fwd_gain_scope,
                  seed_assembly=a.seed_assembly, seed_pa=a.onset_cue_pa, seed_dur=a.onset_cue_dur,
                  seed_frac=a.onset_cue_frac, fixed_seed_asm=a.fixed_seed_asm)
    onset_on = a.onset_on

    # -- BUILD the DECOUPLED forward-asymmetric store (reused frozen across all readout arms) --
    prep = _prepare_sequence(seed, cfg, do_encode=True)
    out["encode_decoupled"] = _weight_diag(prep)
    al = prep["assemblies_local"]
    print(f"  [seed {seed}] DECOUPLED store: within={prep['w_within']:.1f} adj_fwd={prep['w_adj_fwd']:.2f} "
          f"adj_rev={prep['w_adj_rev']:.2f} ratio={out['encode_decoupled']['ratio_adj']:.2f}x "
          f"(n_fwd={prep['n_between_fwd']} n_rev={prep['n_between_rev']}) ({time.time()-t0:.0f}s)", flush=True)

    # ================= OPTION 1 diagnostic (cued completion op-point) =================
    if a.option1:
        o1 = _option1_cued_ignition(prep, seed, recall_drive=a.recall_drive, recall_steps=a.recall_steps,
                                    self_regen_read=a.o1_self_regen, recall_k_thresh=a.o1_k_thresh,
                                    det_frac=a.o1_det_frac, tail_steps=a.o1_tail, det=det)
        out["option1"] = o1
        print(f"  [seed {seed}] OPTION1 (cue asm0 @ drive={a.recall_drive} steps={a.recall_steps} "
              f"self_regen={a.o1_self_regen} k_thresh={a.o1_k_thresh}): per_asm_active={o1['per_asm_active']} "
              f"per_asm_frac={[round(x,3) for x in o1['per_asm_frac']]} n_multi={o1['n_multi']} "
              f"FWD={o1['forward_frac']:.3f} duty={o1['duty_cycle']:.3f} frozen={o1['weights_frozen']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # Option-1 is a diagnostic only: report + (optionally) still run Option 2 below unless --option1-only.
        if a.option1_only:
            out["seed_go"] = None
            return out

    # ================= OPTION 2 GO arm (SWR envelope) =================
    r_go = _rest_swr_envelope(prep, a.rest_steps, seed, mode="swr", noise_on=True, adapt=True, verbose=True, **env_kw)
    s_go = _score_arm(r_go, al, det, a.swr_period, onset_on)
    out["go"] = {**{k: s_go[k] for k in ("n_events", "n_multi", "n_full", "forward_frac", "reverse_frac", "mean_tau",
                                         "chance_forward", "duty_cycle", "pop_rate", "per_asm_active", "per_asm_peak")},
                 "weights_frozen": r_go["weights_frozen"], "apical_rest_max": r_go["apical_rest_max"],
                 "apical_n_latched": r_go["apical_n_latched"], "n_env": r_go["n_env"], "basket_n": r_go["basket_n"],
                 "amp": r_go["amp"]}
    if onset_on:
        out["go"]["onset_cue"] = dict(asm=a.onset_cue_asm, pa=a.onset_cue_pa, dur=a.onset_cue_dur,
                                      frac=a.onset_cue_frac, fixed=a.fixed_seed_asm)
        out["go"]["seed_first_frac"] = s_go.get("seed_first_frac")
        out["go"]["by_seedpos"] = s_go.get("by_seedpos")
        out["go"]["n_seed_envs"] = s_go.get("n_seed_envs")
        print(f"  [seed {seed}] ONSET-CUE: asm={a.onset_cue_asm} (fixed={a.fixed_seed_asm}) pa={a.onset_cue_pa} "
              f"dur={a.onset_cue_dur} frac={a.onset_cue_frac} -> forward=forward-FROM-SEED; "
              f"seed_first_frac={s_go.get('seed_first_frac')} by_seedpos={s_go.get('by_seedpos')}", flush=True)
    chance = max(s_go["chance_forward"], 1e-6)
    _amp = r_go["amp"]
    print(f"  [seed {seed}] FWD-GAIN: scope={_amp['scope']} gain={_amp['fwd_gain']}x on {_amp['n_edges']} edges "
          f"(adj_fwd {_amp['base_mean'] if _amp['base_mean'] is None else round(_amp['base_mean'],1)}"
          f"->{_amp['peak_mean'] if _amp['peak_mean'] is None else round(_amp['peak_mean'],1)} within the envelope; "
          f"restored after)", flush=True)
    print(f"  [seed {seed}] GO (SWR envelope): ev={s_go['n_events']} multi={s_go['n_multi']} full={s_go['n_full']} "
          f"FWD={s_go['forward_frac']:.3f} REV={s_go['reverse_frac']:.3f} chance={chance:.3f} tau={s_go['mean_tau']:+.3f} "
          f"duty={s_go['duty_cycle']:.3f} act={s_go['per_asm_active']} pop={s_go['pop_rate']:.4f} n_env={r_go['n_env']} "
          f"frozen={r_go['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEAT 1: NO-SWR (constant E/I, no transient) -> collapse --
    r_ns = _rest_swr_envelope(prep, a.rest_steps, seed, mode="no_swr", noise_on=True, adapt=True, **env_kw)
    s_ns = _score_arm(r_ns, al, det, a.swr_period, onset_on)
    out["no_swr"] = dict(n_multi=s_ns["n_multi"], forward_frac=s_ns["forward_frac"], reverse_frac=s_ns["reverse_frac"],
                         duty_cycle=s_ns["duty_cycle"], per_asm_active=s_ns["per_asm_active"], pop_rate=s_ns["pop_rate"],
                         weights_frozen=r_ns["weights_frozen"])
    print(f"  [seed {seed}] NO-SWR (constant E/I): multi={s_ns['n_multi']} FWD={s_ns['forward_frac']:.3f} "
          f"duty={s_ns['duty_cycle']:.3f} act={s_ns['per_asm_active']} pop={s_ns['pop_rate']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEAT 5: NO-NOISE acid (envelope on, noise off) -> no specific forward events --
    r_nn = _rest_swr_envelope(prep, a.rest_steps, seed, mode="swr", noise_on=False, adapt=True, **env_kw)
    s_nn = _score_arm(r_nn, al, det, a.swr_period, onset_on)
    out["no_noise"] = dict(n_multi=s_nn["n_multi"], forward_frac=s_nn["forward_frac"], duty_cycle=s_nn["duty_cycle"],
                           per_asm_active=s_nn["per_asm_active"], pop_rate=s_nn["pop_rate"],
                           weights_frozen=r_nn["weights_frozen"])
    print(f"  [seed {seed}] NO-NOISE (acid): multi={s_nn['n_multi']} FWD={s_nn['forward_frac']:.3f} "
          f"duty={s_nn['duty_cycle']:.3f} act={s_nn['per_asm_active']} pop={s_nn['pop_rate']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEAT 7: ADAPT-LESION (d_abs->0, no SFA) -> [3,3,3] co-fire (Ecker control) --
    r_al = _rest_swr_envelope(prep, a.rest_steps, seed, mode="swr", noise_on=True, adapt=False, **env_kw)
    s_al = _score_arm(r_al, al, det, a.swr_period, onset_on)
    out["adapt_lesion"] = dict(n_multi=s_al["n_multi"], forward_frac=s_al["forward_frac"], duty_cycle=s_al["duty_cycle"],
                               per_asm_active=s_al["per_asm_active"], pop_rate=s_al["pop_rate"],
                               weights_frozen=r_al["weights_frozen"])
    print(f"  [seed {seed}] ADAPT-LESION (d_abs=0): multi={s_al['n_multi']} FWD={s_al['forward_frac']:.3f} "
          f"duty={s_al['duty_cycle']:.3f} act={s_al['per_asm_active']} pop={s_al['pop_rate']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEAT 4: PERMUTED-ASSEMBLY (re-score GO firing with random assembly labels) -> chance --
    s_pa = _permuted_assembly_score(r_go["F"], al, seed, det, env_seed_log=r_go["env_seed_log"],
                                    swr_period=a.swr_period, onset_on=onset_on)
    out["permuted_assembly"] = dict(n_multi=s_pa["n_multi"], forward_frac=s_pa["forward_frac"],
                                    reverse_frac=s_pa["reverse_frac"], per_asm_active=s_pa["per_asm_active"])
    print(f"  [seed {seed}] PERMUTED-ASSEMBLY (relabel): multi={s_pa['n_multi']} FWD={s_pa['forward_frac']:.3f} "
          f"REV={s_pa['reverse_frac']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEAT 6: NO-ENCODE (fresh bridge, store skipped, same envelope+noise) -> no specific events --
    prep_ne = _prepare_sequence(seed, cfg, do_encode=False)
    r_ne = _rest_swr_envelope(prep_ne, a.rest_steps, seed, mode="swr", noise_on=True, adapt=True, **env_kw)
    s_ne = _score_arm(r_ne, prep_ne["assemblies_local"], det, a.swr_period, onset_on)
    out["no_encode"] = dict(n_multi=s_ne["n_multi"], forward_frac=s_ne["forward_frac"], w_within=prep_ne["w_within"],
                            per_asm_active=s_ne["per_asm_active"], pop_rate=s_ne["pop_rate"],
                            weights_frozen=r_ne["weights_frozen"])
    print(f"  [seed {seed}] NO-ENCODE: multi={s_ne['n_multi']} FWD={s_ne['forward_frac']:.3f} "
          f"w_within={prep_ne['w_within']:.2f} act={s_ne['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEAT 2: SHUFFLED-STORE (fresh encode + permute between-edge multiset) -> order collapses --
    prep_sc = _prepare_sequence(seed, cfg, do_encode=True)
    n_sc = _scramble_between_weights(prep_sc, seed)
    r_sc = _rest_swr_envelope(prep_sc, a.rest_steps, seed, mode="swr", noise_on=True, adapt=True, **env_kw)
    s_sc = _score_arm(r_sc, prep_sc["assemblies_local"], det, a.swr_period, onset_on)
    out["shuffled_store"] = dict(n_between_shuffled=n_sc, n_multi=s_sc["n_multi"], forward_frac=s_sc["forward_frac"],
                                 reverse_frac=s_sc["reverse_frac"], per_asm_active=s_sc["per_asm_active"],
                                 weights_frozen=r_sc["weights_frozen"])
    print(f"  [seed {seed}] SHUFFLED-STORE ({n_sc} edges): multi={s_sc['n_multi']} FWD={s_sc['forward_frac']:.3f} "
          f"REV={s_sc['reverse_frac']:.3f} act={s_sc['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEAT 3: REVERSE-ASYM-LESION (fresh encode + symmetrize between-edges) -> forward direction destroyed --
    prep_sym = _prepare_sequence(seed, cfg, do_encode=True)
    n_sym = _symmetrize_between_weights(prep_sym)
    r_sym = _rest_swr_envelope(prep_sym, a.rest_steps, seed, mode="swr", noise_on=True, adapt=True, **env_kw)
    s_sym = _score_arm(r_sym, prep_sym["assemblies_local"], det, a.swr_period, onset_on)
    out["reverse_asym_lesion"] = dict(n_between_symmetrized=n_sym, n_multi=s_sym["n_multi"],
                                      forward_frac=s_sym["forward_frac"], reverse_frac=s_sym["reverse_frac"],
                                      per_asm_active=s_sym["per_asm_active"], weights_frozen=r_sym["weights_frozen"])
    print(f"  [seed {seed}] REVERSE-ASYM-LESION ({n_sym} edges): multi={s_sym['n_multi']} "
          f"FWD={s_sym['forward_frac']:.3f} REV={s_sym['reverse_frac']:.3f} act={s_sym['per_asm_active']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ================= PER-SEED VERDICT (verify, don't assert) =================
    fwd = s_go["forward_frac"]; rev = s_go["reverse_frac"]
    pa = s_go["per_asm_active"]; nev = max(s_go["n_multi"], 1)
    forward_ordered = (fwd >= 1.5 * chance and fwd > rev and s_go["n_multi"] >= 2)
    # discreteness: per_asm_active leaning ~[1,1,1] (not [3,3,3] co-ignition) AND the net rests silent between events.
    ignites = (min(pa) >= 1)                                          # every assembly participates (not [0,0,0]/[x,0,0])
    not_cofire = all(x <= 1.6 * nev for x in pa)                      # not [3,3,3] co-fire every event
    discrete = (s_go["duty_cycle"] <= 0.45 and not_cofire)
    # anti-cheats (each must retire its confound)
    def _collapsed(s):  # forward collapses to <= max(0.67*fwd, 1.5*chance) OR no multi events
        return (s["forward_frac"] <= max(0.67 * fwd, 1.5 * chance)) or (s["n_multi"] == 0)
    no_swr_collapses = _collapsed(s_ns)
    shuffled_collapses = _collapsed(s_sc)
    reverse_lesion_collapses = _collapsed(s_sym)
    permuted_chance = (s_pa["forward_frac"] <= max(0.67 * fwd, 1.5 * chance)) or (s_pa["n_multi"] == 0)
    noise_acid = _collapsed(s_nn)                                     # NO-NOISE must NOT give clean forward replay
    # NO-ENCODE SELECTIVITY GATE (load-bearing, coordinator 2026-07-24): the ignition must REQUIRE the learned attractor.
    # With NO store (weights=0.5), the SAME envelope+noise must NOT ignite the labeled cells -> otherwise the drive is a
    # broad DETONATOR (fires cells regardless of the store) and the "forward replay" is an artifact of the drive, not the
    # chain. Require the no-encode run to stay near-silent (max per-assembly participation <= 1) AND not forward-above-chance.
    noencode_selective = (max(s_ne["per_asm_active"]) <= 1) and (
        (s_ne["n_multi"] == 0) or (s_ne["forward_frac"] <= 1.5 * chance))
    adapt_lesion_cofire = (max(s_al["per_asm_active"]) >= 2.0 * nev) or (s_al["duty_cycle"] >= 0.6) or _collapsed(s_al)
    frozen_ok = bool(r_go["weights_frozen"] and r_ns["weights_frozen"] and r_nn["weights_frozen"]
                     and r_al["weights_frozen"] and r_ne["weights_frozen"] and r_sc["weights_frozen"]
                     and r_sym["weights_frozen"])
    dendrite_reset_ok = (r_go["apical_rest_max"] is None
                         or r_go["apical_rest_max"] <= float(DECOUPLED_CFG["plateau_v_hold"]) + 1e-3)

    seed_go = bool(forward_ordered and ignites and discrete and no_swr_collapses and shuffled_collapses
                   and reverse_lesion_collapses and permuted_chance and noise_acid and noencode_selective
                   and adapt_lesion_cofire and frozen_ok and dendrite_reset_ok)
    out["checks"] = dict(forward_ordered=forward_ordered, ignites=ignites, discrete=discrete,
                         no_swr_collapses=no_swr_collapses, shuffled_collapses=shuffled_collapses,
                         reverse_lesion_collapses=reverse_lesion_collapses, permuted_chance=permuted_chance,
                         noise_acid=noise_acid, noencode_selective=noencode_selective,
                         noencode_ignite=list(s_ne["per_asm_active"]),
                         adapt_lesion_cofire=adapt_lesion_cofire, frozen_ok=frozen_ok,
                         dendrite_reset_ok=dendrite_reset_ok)
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def _seq_detect(F, assemblies_local, det):
    return _detect_sequence_events(F, assemblies_local, **det)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000, help="the decoupled store only completes at 2000")
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--rest-steps", type=int, default=1500)
    # SWR envelope (the transient E>I gain)
    ap.add_argument("--swr-period", type=int, default=250, help="steps per SWR cycle (env window + inter-event rest); dt=0.5ms")
    ap.add_argument("--env-dur", type=int, default=120, help="E>I window length within each cycle (~60ms @ dt=0.5)")
    ap.add_argument("--env-exc-pa", type=float, default=180.0, help="broad weak excitation onto ALL CA3-exc during the window (raise E; sub-threshold alone)")
    ap.add_argument("--env-basket-drop", type=float, default=400.0, help="suppress the ca3_pv_basket during the window (drop feedback-I so I can't track E)")
    ap.add_argument("--env-basket-boost", type=float, default=200.0, help="drive the basket ABOVE baseline in the inter-event rest (re-raise I -> self-terminate -> rest)")
    # non-specific noise (the self-organized ignition SEED)
    ap.add_argument("--noise-rate", type=float, default=0.01, help="fraction of CA3-EXC cells NEWLY triggered per step")
    ap.add_argument("--noise-pa", type=float, default=800.0, help="per-pulse pA of the non-specific volley")
    ap.add_argument("--noise-dur", type=int, default=5, help="pulse duration (steps) each triggered CA3-EXC cell is driven")
    # readout substrate
    ap.add_argument("--self-regen-read", type=float, default=0.0, help="plateau self-regen during the READ (0 = transient de-latch -> discrete + able to hand off)")
    ap.add_argument("--self-regen-ignite", type=float, default=None, help="MECHANISM #1 latch-then-release: HIGH self-regen (e.g. 0.15) for the first ignite-frac of each envelope (selective ignition), then drops to self-regen-read (de-latch + forward hand-off). None = constant self-regen-read (byte-identical).")
    ap.add_argument("--ignite-frac", type=float, default=0.4, help="fraction of the envelope window held at self-regen-ignite (latch phase) before release")
    ap.add_argument("--env-exc-ramp", action="store_true", help="MECHANISM #2: ramp env-exc-pa floor->peak across the envelope (most-excitable assembly crosses threshold FIRST -> sequences)")
    ap.add_argument("--env-exc-ramp-floor", type=float, default=0.5, help="MECHANISM #2 ramp start as a fraction of peak (BUGFIX: was implicitly 0 -> subthreshold -> killed ignition). 0.5 = ramp 0.5*peak -> peak.")
    # FORWARD-LINK GAIN (the missing SWR ingredient, 2026-08-20): transiently amplify the learned forward-asymmetric links
    # INTO the ~114-190 handoff band DURING the envelope so the ignited assembly hands off A->B->C, then restore (transient).
    ap.add_argument("--fwd-gain", type=float, default=1.0, help="TRANSIENT multiplicative gain (w->w*g) on the forward-link scope WITHIN the envelope (Buzsaki's 3-5x SWR E>I gain). 1.0 = off (byte-identical). adj_fwd~=38 -> band ~114-190 needs g~3-5.")
    ap.add_argument("--fwd-gain-scope", choices=["adjacent", "between", "forward"], default="adjacent",
                    help="which between-edges the gain multiplies. 'adjacent'=distance-1 BOTH directions (position-symmetric -> lesion-safe; the learned WEIGHT asymmetry carries order); 'between'=all between edges; 'forward'=forward-position only (DIAGNOSTIC/position-asymmetric -> the reverse-asym-lesion gate will flag it if it fabricates order).")
    # DIRECTIONAL ONSET CUE (2026-08-20): a biological PREFIX cue at each SWR envelope onset that biases WHICH assembly
    # ignites first, so the gain-amplified forward links carry the replay FORWARD (the residual after fwd-gain was: ordered
    # but REVERSE, because non-specific noise let the last-seeded assembly dominate). The cue targets the chosen assembly's
    # CELLS (a recall prefix) -- it is NOT a host reordering of the output; forward is then scored forward-FROM-SEED.
    ap.add_argument("--onset-cue-asm", type=str, default=None,
                    help="assembly to cue at each envelope onset: '0' = the chain-start A (fixed prefix -> forward A->B->C); "
                         "an int index for a fixed prefix; 'rand' = a RANDOM assembly per envelope (the rigorous "
                         "forward-FROM-SEED anti-cheat: order that survives a random start rides the WEIGHT asymmetry, not "
                         "the cue -> the REVERSE-ASYM-LESION MUST then collapse it). None = off (non-specific ignition only).")
    ap.add_argument("--onset-cue-pa", type=float, default=0.0, help="onset prefix cue amplitude (pA) onto the cued assembly's cells")
    ap.add_argument("--onset-cue-dur", type=int, default=0, help="onset prefix cue duration (steps) from each envelope onset")
    ap.add_argument("--onset-cue-frac", type=float, default=0.5, help="fraction of the cued assembly's cells the prefix drives")
    ap.add_argument("--recall-k-thresh", type=float, default=None, help="coincidence_k_threshold at read (None = the DECOUPLED store's own, 40)")
    ap.add_argument("--d-abs", type=float, default=40.0, help="Izhikevich per-spike u-kick on CA3-exc (SFA self-avoidance)")
    ap.add_argument("--a-abs", type=float, default=0.008, help="Izhikevich recovery rate a on CA3-exc")
    # detection
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.4)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--active-frac", type=float, default=0.12)
    ap.add_argument("--onset-frac", type=float, default=0.08)
    # OPTION 1 diagnostic (cued completion op-point)
    ap.add_argument("--option1", action="store_true", help="ALSO run the Option-1 cued-ignition diagnostic")
    ap.add_argument("--option1-only", action="store_true", help="run ONLY the Option-1 diagnostic (skip the Option-2 build)")
    ap.add_argument("--recall-drive", type=float, default=700.0, help="Option-1 sustained cue amplitude onto assembly-0 (pA)")
    ap.add_argument("--recall-steps", type=int, default=150, help="Option-1 sustained cue duration (steps)")
    ap.add_argument("--o1-self-regen", type=float, default=0.15, help="Option-1 plateau self-regen (0.15 = bistable re-latch)")
    ap.add_argument("--o1-k-thresh", type=float, default=110.0, help="Option-1 coincidence_k_threshold (completion op-point)")
    ap.add_argument("--o1-det-frac", type=float, default=1.0, help="Option-1 fraction of assembly-0 driven by the cue")
    ap.add_argument("--o1-tail", type=int, default=200, help="Option-1 steps after the cue (observe hand-off / self-terminate)")
    # store knobs (default = the 6/6-GO DECOUPLED store)
    ap.add_argument("--sel-inhib-spare", type=float, default=None,
                    help="readout: basket->member synapse weight. DECOUPLED default 0.0 (members spared). Set >0 (e.g. 20) "
                         "so the SWR basket DROP disinhibits the assembly members (more faithful E/I transient).")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.option1_only:
        a.option1 = True

    # -- parse the DIRECTIONAL ONSET CUE mode --
    if a.onset_cue_asm is None or str(a.onset_cue_asm).strip() == "":
        a.onset_on = False; a.seed_assembly = False; a.fixed_seed_asm = None
    elif str(a.onset_cue_asm).strip().lower() in ("rand", "random", "-1"):
        a.onset_on = True; a.seed_assembly = True; a.fixed_seed_asm = None
    else:
        a.onset_on = True; a.seed_assembly = True; a.fixed_seed_asm = int(a.onset_cue_asm)
    if a.onset_on and (a.onset_cue_pa == 0.0 or a.onset_cue_dur <= 0):
        print(f"[gap5-swr-envelope] WARNING: --onset-cue-asm set but onset-cue-pa={a.onset_cue_pa} "
              f"onset-cue-dur={a.onset_cue_dur} -> the cue injects nothing (no directional bias).", flush=True)

    cfg = dict(DECOUPLED_CFG)
    cfg["n_ca3"] = int(a.n_ca3); cfg["n_mem"] = int(a.n_mem)
    if a.sel_inhib_spare is not None:
        cfg["sel_inhib_spare"] = float(a.sel_inhib_spare)
    if a.recall_k_thresh is None:
        a.recall_k_thresh = float(cfg["recall_k_thresh"])   # the DECOUPLED store's own recall threshold (40)

    _, backend = get_backend()
    print(f"[gap5-swr-envelope] SWR-STATE E/I-TRANSIENT ENVELOPE readout on the DECOUPLED forward-asymmetric store | "
          f"n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']} rest_steps={a.rest_steps} | swr_period={a.swr_period} "
          f"env_dur={a.env_dur} env_exc={a.env_exc_pa} basket_drop={a.env_basket_drop} basket_boost={a.env_basket_boost} "
          f"| noise_rate={a.noise_rate} noise_pa={a.noise_pa} noise_dur={a.noise_dur} | self_regen={a.self_regen_read} "
          f"k_thresh={a.recall_k_thresh} d_abs={a.d_abs} a_abs={a.a_abs} sel_inhib_spare={cfg['sel_inhib_spare']} | "
          f"fwd_gain={a.fwd_gain}x scope={a.fwd_gain_scope} | onset_cue_asm={a.onset_cue_asm} "
          f"(pa={a.onset_cue_pa} dur={a.onset_cue_dur} frac={a.onset_cue_frac}) | "
          f"option1={a.option1} seeds={a.seeds} backend={backend}", flush=True)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, cfg, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per and not a.option1_only:
        n_go = sum(1 for p in per if p.get("seed_go"))
        go = n_go >= max(1, (len(per) + 1) // 2)      # smoke gate; the FULL-RUN GO bar is >=5/6
        mf = float(np.mean([p["go"]["forward_frac"] for p in per]))
        mr = float(np.mean([p["go"]["reverse_frac"] for p in per]))
        mch = float(np.mean([p["go"]["chance_forward"] for p in per]))
        mns = float(np.mean([p["no_swr"]["forward_frac"] for p in per]))
        mnn = float(np.mean([p["no_noise"]["forward_frac"] for p in per]))
        msc = float(np.mean([p["shuffled_store"]["forward_frac"] for p in per]))
        msym = float(np.mean([p["reverse_asym_lesion"]["forward_frac"] for p in per]))
        mduty = float(np.mean([p["go"]["duty_cycle"] for p in per]))
        pa_mean = [float(np.mean([p["go"]["per_asm_active"][k] for p in per])) for k in range(a.n_mem)]
        if go:
            verdict = (f"SWR-ENVELOPE GO {n_go}/{len(per)} -- the SWR-state E/I-transient envelope reads the stored "
                       f"forward-asymmetric chain as DISCRETE forward-ordered bursts (forward_frac {mf:.3f} vs reverse "
                       f"{mr:.3f} vs chance {mch:.3f}; per_asm_active~{[round(x,1) for x in pa_mean]}, duty {mduty:.3f}); "
                       f"NO-SWR collapses ({mns:.3f}), NO-NOISE collapses ({mnn:.3f}), SHUFFLED-STORE ({msc:.3f}) and "
                       f"REVERSE-ASYM-LESION ({msym:.3f}) collapse. => the SWR-state readout surpasses the theta-sweep "
                       f"boundary; run the full 6-seed GPU confirm (bar >=5/6).")
        else:
            verdict = (f"SWR-ENVELOPE PARTIAL/NEGATIVE {n_go}/{len(per)} -- forward_frac {mf:.3f} vs reverse {mr:.3f} "
                       f"vs chance {mch:.3f}; per_asm_active~{[round(x,1) for x in pa_mean]} duty {mduty:.3f}; NO-SWR "
                       f"{mns:.3f} NO-NOISE {mnn:.3f} SHUFFLED {msc:.3f} REVERSE-ASYM {msym:.3f}. Per THE LAW: the "
                       f"envelope depth x duration x noise-sigma is the TUNING band -- over-drive -> [3,3,3] co-fire, "
                       f"under-drive -> [0,0,0] no-ignition. Tune env_exc_pa / env_basket_drop / swr_period / env_dur / "
                       f"noise_pa / self_regen_read (and try --sel-inhib-spare 20 so the basket drop reaches members). "
                       f"A partial on the SWR-envelope rung is a real, honestly-reported result.")
        summary_extra = dict(GO=go, n_go=n_go)
    elif a.option1_only and per:
        go = False; n_go = 0
        pa = [p["option1"]["per_asm_active"] for p in per]
        verdict = (f"OPTION-1 DIAGNOSTIC (cued completion op-point) -- per_asm_active {pa}. "
                   f"Read: [x,0,0] = ignites but NO forward hand-off (residual = ignition-compatible-with-handoff); "
                   f"[3,3,3] = co-fire (over-latched); [0,0,0] = no ignition even at the completion op-point (deeper "
                   f"finding). => build/tune Option 2 accordingly.")
        summary_extra = dict(GO=None, n_go=0)
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_go=0)

    summary = {"probe": "gap5_swr_envelope_replay", "mechanism": "SWR-state E/I-transient envelope readout",
               "seeds": a.seeds, "n_ca3": cfg["n_ca3"], "n_mem": cfg["n_mem"], "rest_steps": a.rest_steps,
               "env_cfg": dict(swr_period=a.swr_period, env_dur=a.env_dur, env_exc_pa=a.env_exc_pa,
                               env_basket_drop=a.env_basket_drop, env_basket_boost=a.env_basket_boost,
                               noise_rate=a.noise_rate, noise_pa=a.noise_pa, noise_dur=a.noise_dur,
                               self_regen_read=a.self_regen_read, recall_k_thresh=a.recall_k_thresh,
                               d_abs=a.d_abs, a_abs=a.a_abs, sel_inhib_spare=cfg["sel_inhib_spare"],
                               fwd_gain=a.fwd_gain, fwd_gain_scope=a.fwd_gain_scope,
                               onset_cue_asm=a.onset_cue_asm, onset_cue_pa=a.onset_cue_pa,
                               onset_cue_dur=a.onset_cue_dur, onset_cue_frac=a.onset_cue_frac,
                               fixed_seed_asm=a.fixed_seed_asm,
                               env_exc_ramp=a.env_exc_ramp, env_exc_ramp_floor=a.env_exc_ramp_floor),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[gap5-swr-envelope] VERDICT: {verdict}\n[gap5-swr-envelope] wrote {a.out}\n"
          + "=" * 120, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
