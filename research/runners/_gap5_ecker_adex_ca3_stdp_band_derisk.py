"""gap#5 SWR forward-replay -- the EMERGENCE variant: make the forward-asymmetric band LEARN itself by STDP instead of
being HAND-WIRED (the #1 residual flagged by the 6-seed GO `_gap5_ecker_adex_ca3_replay_derisk.py`).

WHAT THE GO RUNNER DOES (the scaffold this removes):
  It installs w_fwd=800, w_rev=15 directly (inject_explicit_wiring) -- a hand-wired 53x forward/reverse asymmetry -- and
  proves the discrete forward SWR replay RIDES that asymmetry (REVERSE-ASYM-LESION collapses forward to chance). Per the
  project EMERGENCE BAR, an installed band is a SCAFFOLD; the band should GROW from experience.

WHAT THIS RUNNER DOES (the emergent version):
  1. START the between-assembly links WEAK + SYMMETRIC (forward == reverse == `between_init`, e.g. 15) and PLASTIC.
     Within-assembly recurrence (w_within=60) is FIXED -- the assemblies are pre-formed cell groups; only the
     inter-assembly SEQUENCE order is what must be learned (declared as scope in the verdict).
  2. ENCODING PHASE: repeatedly sweep a moving external cue across the assemblies in the STORED ORDER A->B->C->...->F
     (onset spacing `enc_step`, dwell `enc_dwell`, inter-lap gap `enc_gap` chosen so lap_len*dt > the STDP window so
     cross-lap pairings are skipped). The engine's spike-timing-dependent plasticity (`cfg.enable_stdp`, the SAME fused
     kernel every runner uses) then POTENTIATES the forward edges A->B (pre-before-post, delta_t>0 -> LTP) and DEPRESSES
     the reverse edges B->A (post-before-pre, delta_t<0 -> LTD). The band's forward asymmetry is MEASURED before vs after
     (adj_fwd / adj_rev) -- it must GROW from ~symmetric toward a forward handoff band.
       * ONLY the between edges are plastic (per-synapse cp_synapse_plastic_mask: within-group injected plastic=False,
         between-group injected plastic=True -> `any_fixed` builds the mask -> STDP freezes within, learns between).
       * CRITICAL: `_run_one_simulation_step()` does NOT advance the clock, so the encode loop advances
         `runtime_state.current_time_ms += dt` EACH step -- otherwise every spike shares one timestamp, delta_t==0, and
         STDP is silently INERT (the banked 2026-07-29 silent-failure). Verified: adj_fwd must move.
  3. FREEZE plasticity (enable_stdp=False), then REPLAY exactly as the GO runner: rest + a NON-SPECIFIC random-per-event
     prefix cue, forward-FROM-SEED scoring, and the full anti-cheat battery.

THE GATE (same decisive test, now with a LEARNED band):
  - Discrete forward replay: per_asm_active~[1..], forward_frac >= 1.5x chance, forward > reverse, rests silent.
  - REVERSE-ASYM-LESION (symmetrize the LEARNED between-edges) -> forward collapses to chance. [rides the LEARNED asym]
  - NO-ENCODE (KEY for emergence): skip the encoding phase -> band stays symmetric/weak -> NO forward replay. [proves the
    forward order came from LEARNING, not any residual hand-wiring -- there is none to fall back on]
  - SHUFFLED-STORE, PERMUTED-ASSEMBLY, NO-SEED-silent, FROZEN-during-replay byte-hash, ADAPT-LESION (honest report).

DISCIPLINE: the band is LEARNED by a spiking STDP rule -- the encode drive + STDP produce the asymmetry; the weights are
NEVER host-assigned to a forward-biased value. (The reverse-asym / shuffle CONTROLS do edit the LEARNED weights -- that is
a lesion of the learned band, the same as the GO runner's build-time lesions, applied AFTER learning.)

Reuse-by-import: `_smooth` (the box smoother the sequence/SWR scorers share). Region framework + the committed
ADEX_ECKER_CA3_PC preset + engine STDP. NO `sim/` edit. GPU-preferred.

  Calib:  SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_ecker_adex_ca3_stdp_band_derisk --seeds 42 --calibrate
  Smoke:  SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_ecker_adex_ca3_stdp_band_derisk --seeds 42
  6-seed: SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_ecker_adex_ca3_stdp_band_derisk \
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

from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402
from sim.enums import NeuronModel, NeuronType  # noqa: E402
from sim.backend import to_host, get_backend  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import _smooth  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "ecker_adex_ca3_stdp_band.json"


# ----------------------------------------------------------------------------------------------------------------------
# BUILD the store: within-assembly recurrence FIXED (frozen), between-assembly edges (fwd AND rev) start SYMMETRIC + weak
# and PLASTIC. No hand-wired asymmetry. STDP params are set on cfg but enable_stdp starts False (flipped on per-phase).
# ----------------------------------------------------------------------------------------------------------------------
def build_store(seed, *, m_asm, asm_size, w_within, between_init, within_density, b_override, a_override,
                ou_sigma, dt, stdp_w_max, stdp_a_plus, stdp_a_minus, stdp_tau):
    cp, _ = get_backend()
    n_pc = m_asm * asm_size + 20
    regions = [BrainRegion(name="pc", n_neurons=n_pc, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)   # SET cfg.seed (NOT actual_seed_used -- the no-op field)
    cfg.dt_ms = float(dt); cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.default_neuron_type_adex = NeuronType.ADEX_ECKER_CA3_PC.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    for f in ("enable_homeostasis", "enable_hebbian_learning", "enable_structural_plasticity",
              "enable_parameter_heterogeneity", "enable_inhibitory_stdp", "enable_reward_modulation"):
        setattr(cfg, f, False)
    # STDP config. enable_stdp MUST be True THROUGH _initialize_simulation_data so cp_last_spike_time is ALLOCATED
    # (bridge.py:2713 gates the allocation on it); we flip it OFF right after init and the encode phase flips it back on.
    # (A None cp_last_spike_time makes the STDP path silently skip -> the band never learns.)
    cfg.enable_stdp = True
    cfg.stdp_a_plus = float(stdp_a_plus); cfg.stdp_a_minus = float(stdp_a_minus)
    cfg.stdp_tau_plus_ms = float(stdp_tau); cfg.stdp_tau_minus_ms = float(stdp_tau)
    cfg.stdp_w_min = 0.0; cfg.stdp_w_max = float(stdp_w_max)
    cfg.enable_ou_process = ou_sigma > 0; cfg.ou_noise_sigma_pa = float(ou_sigma)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    b.core_config.enable_stdp = False   # cp_last_spike_time now allocated; STDP stays OFF until encode() turns it on
    if b_override is not None:
        b.core_config.adex_b = float(b_override)
    if a_override is not None:
        b.core_config.adex_a = float(a_override)
    pc = np.asarray(b.region_manager.indices("pc"), int)
    asm_local = [np.arange(i * asm_size, (i + 1) * asm_size) for i in range(m_asm)]

    rng = np.random.default_rng(seed * 17 + 3)
    # Two injected GROUPS so the plastic mask is built: within (FROZEN) + between (PLASTIC, symmetric weak).
    win_pre, win_post, win_w = [], [], []
    bet_pre, bet_post, bet_w = [], [], []

    def blk(dst_pre, dst_post, src, dst, weight, dens):
        if weight <= 0:
            return
        for s in src:
            d = dst[dst != s]
            if dens < 1.0:
                d = d[rng.random(len(d)) < dens]
            if len(d) == 0:
                continue
            dst_pre.append(np.full(len(d), pc[s])); dst_post.append(pc[d]); weight_arr = np.full(len(d), float(weight))
            (win_w if dst_pre is win_pre else bet_w).append(weight_arr)

    for i in range(m_asm):
        blk(win_pre, win_post, asm_local[i], asm_local[i], w_within, within_density)
        if i + 1 < m_asm:
            blk(bet_pre, bet_post, asm_local[i], asm_local[i + 1], between_init, within_density)   # forward edges
        if i - 1 >= 0:
            blk(bet_pre, bet_post, asm_local[i], asm_local[i - 1], between_init, within_density)   # reverse edges

    def cat(a):
        return np.concatenate(a) if a else np.zeros(0, int)

    wiring = {
        "within": {"pre_indices": cat(win_pre).astype(int).tolist(), "post_indices": cat(win_post).astype(int).tolist(),
                   "initial_weights": cat(win_w).astype(float).tolist(), "plastic": False, "conn_type": "ff"},
        "between": {"pre_indices": cat(bet_pre).astype(int).tolist(), "post_indices": cat(bet_post).astype(int).tolist(),
                    "initial_weights": cat(bet_w).astype(float).tolist(), "plastic": True, "conn_type": "ff"},
    }
    b.inject_explicit_wiring(wiring)

    # Position masks into cp_connections.data (COO order == CSR .data order): forward / reverse / within edges.
    coo = b.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row), int); col = np.asarray(to_host(coo.col), int)
    asm_of = np.full(b.core_config.num_neurons, -1, int)
    for i, al in enumerate(asm_local):
        asm_of[pc[al]] = i
    a_pre = asm_of[row]; a_post = asm_of[col]
    valid = a_pre >= 0
    fwd_pos = np.nonzero(valid & (a_post == a_pre + 1))[0]
    rev_pos = np.nonzero(valid & (a_post == a_pre - 1))[0]
    within_pos = np.nonzero(valid & (a_post == a_pre))[0]

    # SANITY: alignment + plastic-mask correctness (loud failure if the COO/.data order assumption is wrong).
    data0 = np.asarray(to_host(b.cp_connections.data))
    assert fwd_pos.size and rev_pos.size, "no forward/reverse between edges found -- topology/label bug"
    assert np.allclose(data0[fwd_pos], between_init) and np.allclose(data0[rev_pos], between_init), \
        "between-edge init != between_init at fwd/rev positions -- COO/.data misalignment"
    assert np.allclose(data0[within_pos], w_within), "within-edge weight mismatch -- COO/.data misalignment"
    if b.cp_synapse_plastic_mask is not None:
        pm = np.asarray(to_host(b.cp_synapse_plastic_mask)).astype(bool)
        assert pm[fwd_pos].all() and pm[rev_pos].all() and (not pm[within_pos].any()), \
            "plastic mask wrong: between must be plastic, within must be frozen"
    else:
        raise AssertionError("cp_synapse_plastic_mask is None -- within(plastic=False) should have built it")

    return dict(bridge=b, cp=cp, pc=pc, asm_local=asm_local, m_asm=m_asm, asm_size=asm_size,
                fwd_pos=fwd_pos, rev_pos=rev_pos, within_pos=within_pos,
                n_between=int(fwd_pos.size + rev_pos.size), n_fwd=int(fwd_pos.size), n_rev=int(rev_pos.size),
                pre_post=(row, col))


def measure_band(store):
    d = np.asarray(to_host(store["bridge"].cp_connections.data))
    fwd = d[store["fwd_pos"]]; rev = d[store["rev_pos"]]; win = d[store["within_pos"]]
    af = float(fwd.mean()) if fwd.size else 0.0
    ar = float(rev.mean()) if rev.size else 0.0
    return dict(adj_fwd=af, adj_rev=ar, adj_within=float(win.mean()) if win.size else 0.0,
                ratio=(af / max(ar, 1e-6)), fwd_max=float(fwd.max()) if fwd.size else 0.0,
                fwd_min=float(fwd.min()) if fwd.size else 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# ENCODING PHASE: STDP on, clock ADVANCED each step, moving cue sweeps A->B->...->F for n_laps. The forward pairing
# (pre-before-post) potentiates forward edges; the reverse pairing (post-before-pre) depresses reverse edges.
# ----------------------------------------------------------------------------------------------------------------------
def encode(store, seed, *, n_laps, enc_step, enc_dwell, enc_gap, cue_pa, cue_frac, dt):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    asm_size = store["asm_size"]
    cell_rng = np.random.default_rng(int(seed) * 777 + 5)
    k_cells = max(1, int(round(float(cue_frac) * asm_size)))
    cue_cells_dev = []
    for a_loc in asm_local:
        sub = np.sort(cell_rng.choice(a_loc, min(k_cells, len(a_loc)), replace=False))
        cue_cells_dev.append(cp.asarray(pc[sub], dtype=cp.int64))
    bridge.core_config.enable_stdp = True
    bridge.runtime_state.current_time_ms = 0.0
    n_steps = 0
    for _lap in range(n_laps):
        for k in range(m):
            for s in range(enc_step):
                bridge.cp_external_input_current[:] = 0.0
                if s < enc_dwell:
                    bridge.cp_external_input_current[cue_cells_dev[k]] += float(cue_pa)
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_ms += float(dt)   # ADVANCE THE CLOCK (else STDP is inert)
                n_steps += 1
        for _s in range(enc_gap):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_ms += float(dt)
            n_steps += 1
    bridge.core_config.enable_stdp = False
    return dict(n_steps=n_steps, lap_len_ms=(m * enc_step + enc_gap) * dt)


# ----------------------------------------------------------------------------------------------------------------------
# REST + non-specific-seeded discrete SWR replay (STDP OFF -> weights frozen). Injects ONLY external current
# (NUMPY-REFERENCE guard). seed_on=False = NO-SEED control.
# ----------------------------------------------------------------------------------------------------------------------
def rest_and_replay(store, rest_steps, seed, *, swr_period, cue_pa, cue_steps, cue_frac, seed_on=True):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    n_pc = len(pc)
    cell_rng = np.random.default_rng(int(seed) * 314159 + 17)
    asm_size = store["asm_size"]
    k_cells = max(1, int(round(float(cue_frac) * asm_size)))
    cue_cells_dev = []
    for a_loc in asm_local:
        sub = np.sort(cell_rng.choice(a_loc, min(k_cells, len(a_loc)), replace=False))
        cue_cells_dev.append(cp.asarray(pc[sub], dtype=cp.int64))
    choice_rng = np.random.default_rng(int(seed) * 271828 + 23)

    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    F = np.zeros((rest_steps, n_pc), dtype=bool)
    env_seed_log = []
    cur_k = None; n_env = 0
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = 0.0
        phase = t % swr_period
        if phase == 0 and seed_on:
            cur_k = int(choice_rng.integers(0, m))
            env_seed_log.append(cur_k); n_env += 1
        if seed_on and phase < cue_steps and cur_k is not None:
            bridge.cp_external_input_current[cue_cells_dev[cur_k]] += float(cue_pa)
        bridge._run_one_simulation_step()
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[pc].astype(bool)
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    frozen = bool(np.array_equal(w_before, w_after))
    return dict(F=F, env_seed_log=env_seed_log, n_env=n_env, weights_frozen=frozen)


def _score_periods(F, assemblies_local, env_seed_log, swr_period, *, W, active_frac, onset_frac):
    """PERIOD-LOCKED forward-FROM-SEED scoring (identical contract to the GO runner)."""
    import math
    T, _ = F.shape
    n_mem = len(assemblies_local)
    asizes = [max(1, len(a)) for a in assemblies_local]
    n_periods = min(len(env_seed_log), T // swr_period)
    per_asm_active = [0] * n_mem
    n_multi = fwd = rev = seed_first = 0
    chance_terms = []
    by_seedpos = {k: {"multi": 0, "fwd": 0, "rev": 0} for k in range(n_mem)}
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
        if len(active) < 2:
            continue
        n_multi += 1
        chance_terms.append(1.0 / math.factorial(len(active)))
        order = [kk for kk, _ in sorted(active, key=lambda kv: kv[1])]
        by_seedpos[k]["multi"] += 1
        if order[0] == k:
            seed_first += 1
        is_fwd = (order[0] == k) and all(order[i + 1] == order[i] + 1 for i in range(len(order) - 1))
        is_rev = (order[0] == k) and all(order[i + 1] == order[i] - 1 for i in range(len(order) - 1))
        if is_fwd:
            fwd += 1; by_seedpos[k]["fwd"] += 1
        if is_rev:
            rev += 1; by_seedpos[k]["rev"] += 1
    pop = F.mean(1)
    return dict(n_events=n_periods, n_multi=n_multi, forward_frac=(fwd / n_multi) if n_multi else 0.0,
                reverse_frac=(rev / n_multi) if n_multi else 0.0, per_asm_active=per_asm_active,
                chance_forward=float(np.mean(chance_terms)) if chance_terms else 0.0,
                seed_first_frac=(seed_first / n_multi) if n_multi else 0.0,
                duty_cycle=float((pop > 0.01).mean()), pop_rate=float(pop.mean()),
                by_seedpos={k: dict(v) for k, v in by_seedpos.items()})


def _permuted_assembly(store, r, seed, det, swr_period):
    al = store["asm_local"]
    perm = np.random.default_rng(int(seed) * 5150 + 3).permutation(len(al))
    relabeled = [al[i] for i in perm]
    return _score_periods(r["F"], relabeled, r["env_seed_log"], swr_period, W=det["W"],
                          active_frac=det["active_frac"], onset_frac=det["onset_frac"])


def _load_weights(store, w_host):
    cp = store["cp"]
    store["bridge"].cp_connections.data[:] = cp.asarray(np.asarray(w_host, dtype=np.float32))


def _symmetrize(w_host, fwd_pos, rev_pos):
    """REVERSE-ASYM-LESION on the LEARNED band: set fwd==rev==count-weighted mean (direction destroyed, budget kept)."""
    w = np.asarray(w_host, dtype=np.float32).copy()
    tot = float(w[fwd_pos].sum() + w[rev_pos].sum()); n = int(fwd_pos.size + rev_pos.size)
    m = tot / max(n, 1)
    w[fwd_pos] = m; w[rev_pos] = m
    return w


def _shuffle_between(w_host, fwd_pos, rev_pos, seed):
    w = np.asarray(w_host, dtype=np.float32).copy()
    bpos = np.concatenate([fwd_pos, rev_pos])
    vals = w[bpos].copy()
    np.random.default_rng(int(seed) * 29 + 7).shuffle(vals)
    w[bpos] = vals
    return w


# ----------------------------------------------------------------------------------------------------------------------
def _fresh_build(seed, a, **over):
    kw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
              within_density=a.within_density, b_override=a.b_override, a_override=None,
              ou_sigma=a.ou_sigma, dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus,
              stdp_a_minus=a.stdp_a_minus, stdp_tau=a.stdp_tau)
    kw.update(over)
    return build_store(seed, **kw)


def one_seed(seed, a):
    t0 = time.time()
    out = {"seed": seed}
    det = dict(W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    seed_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)

    # ============ 1. BUILD + ENCODE: the band LEARNS its forward asymmetry by STDP ============
    st_enc = _fresh_build(seed, a)
    band_before = measure_band(st_enc)
    enc = encode(st_enc, seed, **enc_kw)
    band_after = measure_band(st_enc)
    w_learned = np.asarray(to_host(st_enc["bridge"].cp_connections.data)).copy()
    fwd_pos = st_enc["fwd_pos"]; rev_pos = st_enc["rev_pos"]
    out["band_before"] = band_before; out["band_after"] = band_after
    out["encode"] = dict(n_steps=enc["n_steps"], lap_len_ms=enc["lap_len_ms"], n_laps=a.n_laps)
    print(f"  [seed {seed}] BAND before: fwd={band_before['adj_fwd']:.2f} rev={band_before['adj_rev']:.2f} "
          f"within={band_before['adj_within']:.1f}  ->  after ENCODE ({a.n_laps} laps, {enc['n_steps']} steps): "
          f"fwd={band_after['adj_fwd']:.2f} rev={band_after['adj_rev']:.2f} ratio={band_after['ratio']:.1f}x "
          f"fwd_max={band_after['fwd_max']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    # ============ 2. GO REPLAY from the LEARNED band (fresh store, frozen weights) ============
    st_go = _fresh_build(seed, a); _load_weights(st_go, w_learned)
    r_go = rest_and_replay(st_go, a.rest_steps, seed, **seed_kw)
    s_go = _score_periods(r_go["F"], st_go["asm_local"], r_go["env_seed_log"], a.swr_period, **det)
    chance = max(s_go["chance_forward"], 1e-6)
    fwd = s_go["forward_frac"]; rev = s_go["reverse_frac"]; pa = s_go["per_asm_active"]; nmulti = s_go["n_multi"]
    out["go"] = dict(n_events=s_go["n_events"], n_multi=nmulti, forward_frac=fwd, reverse_frac=rev, chance=chance,
                     seed_first_frac=s_go.get("seed_first_frac"), duty_cycle=s_go["duty_cycle"],
                     pop_rate=s_go["pop_rate"], per_asm_active=pa, n_env=r_go["n_env"],
                     weights_frozen=r_go["weights_frozen"], by_seedpos=s_go.get("by_seedpos"))
    print(f"  [seed {seed}] GO(learned): ev={s_go['n_events']} multi={nmulti} FWD={fwd:.3f} REV={rev:.3f} "
          f"chance={chance:.3f} seed_first={s_go.get('seed_first_frac')} act={pa} duty={s_go['duty_cycle']:.3f} "
          f"n_env={r_go['n_env']} frozen={r_go['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)

    if a.calibrate:
        seed_go = (fwd >= 1.5 * chance and fwd > rev and min(pa) >= 1)
        out["seed_go"] = bool(seed_go)
        out["checks"] = dict(calibrate=True, band_grew=band_after["adj_fwd"] > 2 * band_before["adj_fwd"],
                             forward_ordered=bool(fwd >= 1.5 * chance and fwd > rev))
        print(f"  [seed {seed}] CALIBRATE => band fwd {band_before['adj_fwd']:.1f}->{band_after['adj_fwd']:.1f} "
              f"rev {band_before['adj_rev']:.1f}->{band_after['adj_rev']:.1f} | replay FWD={fwd:.3f} vs chance "
              f"{chance:.3f} ({'ok' if seed_go else 'no'})", flush=True)
        return out

    # ============ 3. ANTI-CHEAT BATTERY ============
    def replay_arm(w_host, tag, seed_on=True, **over):
        s = _fresh_build(seed, a, **over)
        if w_host is not None:
            _load_weights(s, w_host)
        rr = rest_and_replay(s, a.rest_steps, seed, seed_on=seed_on, **seed_kw)
        sco = _score_periods(rr["F"], s["asm_local"], rr["env_seed_log"], a.swr_period, **det)
        print(f"  [seed {seed}] {tag}: multi={sco['n_multi']} FWD={sco['forward_frac']:.3f} "
              f"REV={sco['reverse_frac']:.3f} act={sco['per_asm_active']} duty={sco['duty_cycle']:.3f} "
              f"pop={sco['pop_rate']:.5f} frozen={rr['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)
        return rr, sco

    # REVERSE-ASYM-LESION: symmetrize the LEARNED band -> forward MUST collapse [KEY: rides the LEARNED asymmetry]
    r_sym, s_sym = replay_arm(_symmetrize(w_learned, fwd_pos, rev_pos), "REVERSE-ASYM-LESION (symmetrize learned)")
    # NO-ENCODE: DO NOT load the learned band -> stays symmetric/weak -> NO forward replay [KEY: proves emergence]
    r_ne, s_ne = replay_arm(None, "NO-ENCODE (skip STDP -> symmetric weak band)")
    # SHUFFLED-STORE: permute the learned between weights
    r_sc, s_sc = replay_arm(_shuffle_between(w_learned, fwd_pos, rev_pos, seed), "SHUFFLED-STORE (permute learned band)")
    # NO-SEED: learned band, no prefix cues -> silent
    r_ns, s_ns = replay_arm(w_learned, "NO-SEED (no prefix cues)", seed_on=False)
    # ADAPT-LESION: learned band, a=b=0 -> honest report
    r_al, s_al = replay_arm(w_learned, "ADAPT-LESION (a=0,b=0)", b_override=0.0, a_override=0.0)
    # PERMUTED-ASSEMBLY: re-score GO firing with random labels
    s_pa = _permuted_assembly(st_go, r_go, seed, det, a.swr_period)
    print(f"  [seed {seed}] PERMUTED-ASSEMBLY: multi={s_pa['n_multi']} FWD={s_pa['forward_frac']:.3f} "
          f"REV={s_pa['reverse_frac']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    out["reverse_asym_lesion"] = dict(n_multi=s_sym["n_multi"], forward_frac=s_sym["forward_frac"],
                                      reverse_frac=s_sym["reverse_frac"], per_asm_active=s_sym["per_asm_active"],
                                      weights_frozen=r_sym["weights_frozen"])
    out["no_encode"] = dict(n_multi=s_ne["n_multi"], forward_frac=s_ne["forward_frac"], reverse_frac=s_ne["reverse_frac"],
                            per_asm_active=s_ne["per_asm_active"], pop_rate=s_ne["pop_rate"],
                            weights_frozen=r_ne["weights_frozen"])
    out["shuffled_store"] = dict(n_between=st_enc["n_between"], n_multi=s_sc["n_multi"],
                                 forward_frac=s_sc["forward_frac"], reverse_frac=s_sc["reverse_frac"],
                                 per_asm_active=s_sc["per_asm_active"], weights_frozen=r_sc["weights_frozen"])
    out["no_seed"] = dict(n_multi=s_ns["n_multi"], forward_frac=s_ns["forward_frac"], pop_rate=s_ns["pop_rate"],
                          per_asm_active=s_ns["per_asm_active"])
    out["adapt_lesion"] = dict(n_multi=s_al["n_multi"], forward_frac=s_al["forward_frac"], duty_cycle=s_al["duty_cycle"],
                               per_asm_active=s_al["per_asm_active"])
    out["permuted_assembly"] = dict(n_multi=s_pa["n_multi"], forward_frac=s_pa["forward_frac"],
                                    reverse_frac=s_pa["reverse_frac"])

    # ================= PER-SEED VERDICT (verify, don't assert) =================
    def _collapsed(s):
        return (s["forward_frac"] <= max(0.5 * fwd, 1.5 * chance)) or (s["n_multi"] == 0)
    band_grew = (band_after["adj_fwd"] > 2.0 * max(band_before["adj_fwd"], 1e-6)
                 and band_after["adj_fwd"] > 2.0 * max(band_after["adj_rev"], 1e-6))
    forward_ordered = (fwd >= 1.5 * chance and fwd > rev and nmulti >= a.min_multi)
    ignites = (min(pa) >= 1)
    discrete = (s_go["duty_cycle"] <= a.max_duty)
    reverse_lesion_collapses = _collapsed(s_sym)
    no_encode_collapses = _collapsed(s_ne)
    shuffled_collapses = _collapsed(s_sc)
    permuted_chance = _collapsed(s_pa)
    no_seed_silent = (s_ns["n_multi"] == 0) or (s_ns["forward_frac"] <= 1.5 * chance)
    frozen_ok = bool(r_go["weights_frozen"] and r_sym["weights_frozen"] and r_ne["weights_frozen"]
                     and r_sc["weights_frozen"] and r_ns["weights_frozen"] and r_al["weights_frozen"])
    seed_go = bool(band_grew and forward_ordered and ignites and discrete and reverse_lesion_collapses
                   and no_encode_collapses and shuffled_collapses and permuted_chance and no_seed_silent and frozen_ok)
    out["checks"] = dict(band_grew=band_grew, forward_ordered=forward_ordered, ignites=ignites, discrete=discrete,
                         reverse_lesion_collapses=reverse_lesion_collapses, no_encode_collapses=no_encode_collapses,
                         shuffled_collapses=shuffled_collapses, permuted_chance=permuted_chance,
                         no_seed_silent=no_seed_silent, frozen_ok=frozen_ok,
                         adapt_lesion_fwd=round(s_al["forward_frac"], 3))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=6)
    ap.add_argument("--asm-size", type=int, default=80)
    ap.add_argument("--within-density", type=float, default=0.5)
    ap.add_argument("--rest-steps", type=int, default=13000)
    ap.add_argument("--dt", type=float, default=0.1)
    # store weights: within FIXED, between START symmetric+weak+PLASTIC (NO hand-wired asymmetry)
    ap.add_argument("--w-within", type=float, default=60.0)
    ap.add_argument("--between-init", type=float, default=15.0, help="symmetric weak init for BOTH fwd+rev between edges")
    ap.add_argument("--b-override", type=float, default=120.0)
    # STDP (the learning rule that GROWS the band)
    ap.add_argument("--stdp-w-max", type=float, default=900.0, help="soft/hard cap; forward asymptotes toward this")
    ap.add_argument("--stdp-a-plus", type=float, default=0.05)
    ap.add_argument("--stdp-a-minus", type=float, default=0.06)
    ap.add_argument("--stdp-tau", type=float, default=20.0)
    # ENCODING sweep A->B->...->F
    ap.add_argument("--n-laps", type=int, default=30, help="repetitions of the A->B->C stored sequence (band saturates "
                    "~15-20 laps at the default STDP gains; 30 for margin)")
    ap.add_argument("--enc-step", type=int, default=80, help="steps per assembly slot (onset-to-onset; *dt = fwd lag ms)")
    ap.add_argument("--enc-dwell", type=int, default=40, help="steps of active cue within a slot")
    ap.add_argument("--enc-gap", type=int, default=600, help="silent steps after each lap (lap_len*dt must exceed 5*tau)")
    ap.add_argument("--enc-cue-pa", type=float, default=9000.0)
    ap.add_argument("--enc-cue-frac", type=float, default=0.6)
    # SWR replay / non-specific prefix seed
    ap.add_argument("--swr-period", type=int, default=325)
    ap.add_argument("--cue-pa", type=float, default=9000.0)
    ap.add_argument("--cue-steps", type=int, default=40)
    ap.add_argument("--cue-frac", type=float, default=0.6)
    ap.add_argument("--ou-sigma", type=float, default=40.0)
    # detection
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--active-frac", type=float, default=0.10)
    ap.add_argument("--onset-frac", type=float, default=0.06)
    ap.add_argument("--min-multi", type=int, default=6)
    ap.add_argument("--max-duty", type=float, default=0.35)
    ap.add_argument("--calibrate", action="store_true", help="build+encode+GO-replay only (locate the STDP regime)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    print(f"[gap5-stdp-band] Ecker AdEx CA3 STDP-LEARNED band | n_mem={a.n_mem} asm={a.asm_size} within={a.w_within} "
          f"between_init={a.between_init} | STDP a+={a.stdp_a_plus} a-={a.stdp_a_minus} tau={a.stdp_tau} "
          f"w_max={a.stdp_w_max} | encode {a.n_laps}laps step={a.enc_step} dwell={a.enc_dwell} gap={a.enc_gap} "
          f"cue={a.enc_cue_pa}@{a.enc_cue_frac} | swr={a.swr_period} rest={a.rest_steps} dt={a.dt} seeds={a.seeds} "
          f"backend={backend}", flush=True)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per and not a.calibrate:
        n_go = sum(1 for p in per if p.get("seed_go"))
        bar = max(1, (len(per) + 1) // 2) if len(per) < 6 else 5
        go = n_go >= bar
        mf = float(np.mean([p["go"]["forward_frac"] for p in per]))
        mr = float(np.mean([p["go"]["reverse_frac"] for p in per]))
        mch = float(np.mean([p["go"]["chance"] for p in per]))
        msym = float(np.mean([p["reverse_asym_lesion"]["forward_frac"] for p in per]))
        mne = float(np.mean([p["no_encode"]["forward_frac"] for p in per]))
        msc = float(np.mean([p["shuffled_store"]["forward_frac"] for p in per]))
        mduty = float(np.mean([p["go"]["duty_cycle"] for p in per]))
        maf_b = float(np.mean([p["band_before"]["adj_fwd"] for p in per]))
        maf_a = float(np.mean([p["band_after"]["adj_fwd"] for p in per]))
        mar_a = float(np.mean([p["band_after"]["adj_rev"] for p in per]))
        pa_mean = [float(np.mean([p["go"]["per_asm_active"][k] for p in per])) for k in range(a.n_mem)]
        fwd_vec = [round(p["go"]["forward_frac"], 3) for p in per]
        if go:
            verdict = (f"STDP-BAND GO {n_go}/{len(per)} -- the forward-asymmetric band was LEARNED by STDP (adj_fwd "
                       f"{maf_b:.1f}->{maf_a:.1f}, adj_rev after {mar_a:.1f}) from an A->B->C encoding sweep, then the "
                       f"frozen learned band replays DISCRETE forward SWR events from a NON-SPECIFIC prefix seed: "
                       f"forward-FROM-SEED {mf:.3f} vs reverse {mr:.3f} vs chance {mch:.3f}; "
                       f"per_asm_active~{[round(x,1) for x in pa_mean]} duty {mduty:.3f}. The order RIDES THE LEARNED "
                       f"asymmetry: REVERSE-ASYM-LESION -> {msym:.3f}; and it EMERGED from learning: NO-ENCODE -> "
                       f"{mne:.3f}, SHUFFLED -> {msc:.3f}. forward-frac/seed={fwd_vec}. => the SWR forward-replay band "
                       f"is now EMERGENT (STDP-grown), not hand-wired. "
                       f"{'Run the full 6-seed confirm (bar >=5/6).' if len(per) < 6 else ''}")
        else:
            verdict = (f"STDP-BAND PARTIAL/NEGATIVE {n_go}/{len(per)} -- band adj_fwd {maf_b:.1f}->{maf_a:.1f} rev "
                       f"{mar_a:.1f}; replay forward {mf:.3f} vs reverse {mr:.3f} vs chance {mch:.3f}; "
                       f"REVERSE-ASYM {msym:.3f} NO-ENCODE {mne:.3f} SHUFFLED {msc:.3f}. forward-frac/seed={fwd_vec}. "
                       f"Check per-seed checks (KEY gates: band_grew, reverse_lesion_collapses, no_encode_collapses).")
        v = Verdict("Ecker AdEx CA3: STDP-LEARNED forward band -> discrete forward SWR replay that rides the LEARNED "
                    "asymmetry (emergent, not hand-wired)", chance=mch)
        v.require("forward events ignite + discrete + forward-ordered on >= bar of the seeds", n_go,
                  expect=lambda n, b=bar: n >= b)
        v.reaches("STDP GREW the forward band (mean adj_fwd before vs after encoding)", before=maf_b, after=maf_a)
        v.floor("mean forward-from-seed exceeds chance", mf, floor=mch)
        v.require("reverse replay is absent (mean reverse_frac < 0.05)", mr, expect=lambda r: r < 0.05)
        v.control("NO-ENCODE [EMERGENCE]: learned-band forward vs no-encode (symmetric weak) forward -- must DIFFER, "
                  "proving the order came from LEARNING not any residual wiring", treatment=mf, control=mne,
                  min_separation=0.0)
        v.control("REVERSE-ASYM-LESION: learned-band forward vs symmetrized-learned-band forward (order rides the "
                  "LEARNED asymmetry, not the cue)", treatment=mf, control=msym, min_separation=0.0)
        v.control("SHUFFLED-STORE: learned-band forward vs shuffled-learned-band forward", treatment=mf, control=msc,
                  min_separation=0.0)
        v.disabled("within-assembly recurrence (the assemblies are pre-formed cell groups; ONLY the inter-assembly "
                   "SEQUENCE order is learned by STDP -- within edges injected plastic=False/frozen)",
                   why="scope: this arc removes the hand-wired BETWEEN-assembly asymmetry residual; within-assembly "
                       "structure is a separate (assembly-formation) question")
        v.disabled("OU-only spontaneous ignition (a non-specific random-per-event prefix cue is required); STDP is "
                   "frozen during replay (the reverse-asym-lesion + no-encode controls test the frozen learned band)",
                   why="isolation: ignition uses a non-specific prefix; the order must ride the LEARNED frozen band")
        decided = v.decide(go=go, verbose=False)
        attributable_to("forward SWR replay ordering (LEARNED band vs NO-ENCODE control)", mf, mne)
        attributable_to("forward SWR replay ordering (LEARNED band vs reverse-asym-lesioned learned band)", mf, msym)
        summary_extra = dict(GO=go, n_go=n_go, status=decided.get("status"),
                             band_adj_fwd_before=maf_b, band_adj_fwd_after=maf_a, band_adj_rev_after=mar_a,
                             forward=mf, reverse=mr, chance=mch, no_encode_forward=mne, reverse_asym_forward=msym,
                             preconditions=decided.get("preconditions", []), decided=decided)
    elif err is None and per and a.calibrate:
        go = False; n_go = 0
        verdict = "CALIBRATE -- see per-seed band before/after + GO replay forward vs chance (no full battery run)"
        summary_extra = dict(GO=False, n_go=0, calibrate=True)
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_go=0)

    summary = {"probe": "gap5_ecker_adex_ca3_stdp_band",
               "mechanism": "STDP-LEARNED forward-asymmetric band (A->B->C encoding sweep) -> discrete forward SWR replay",
               "seeds": a.seeds, "n_mem": a.n_mem, "asm_size": a.asm_size, "rest_steps": a.rest_steps,
               "cfg": dict(w_within=a.w_within, between_init=a.between_init, b_override=a.b_override,
                           within_density=a.within_density, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus,
                           stdp_a_minus=a.stdp_a_minus, stdp_tau=a.stdp_tau, n_laps=a.n_laps, enc_step=a.enc_step,
                           enc_dwell=a.enc_dwell, enc_gap=a.enc_gap, enc_cue_pa=a.enc_cue_pa, enc_cue_frac=a.enc_cue_frac,
                           swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac,
                           ou_sigma=a.ou_sigma, dt=a.dt),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[gap5-stdp-band] VERDICT: {verdict}\n[gap5-stdp-band] wrote {a.out}\n"
          + "=" * 120, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
