"""gap#5 SWR forward-replay wall -> the named-fix BUILD: an Ecker-2022-style AdEx CA3 with ASSEMBLY-structured banded
recurrence that produces DISCRETE SWR events with genuine forward A->B->C population hand-off RIDING THE WEIGHT ASYMMETRY.

WHY THIS MODEL (vs the two banked negatives):
  1. The DECOUPLED BISTABLE STORE (`_gap5_swr_envelope_replay_derisk.py`, 2026-08-20 UPDATE-2) FAILED the store-lesion
     anti-cheats: its strong within-attractors (~200) reverberate semi-continuously (never SEGMENT into discrete events)
     and its forward weight-asymmetry (adj_fwd 38 vs adj_rev 5) is NOT load-bearing on order -- forward SURVIVED the
     REVERSE-ASYM-LESION. Bistable *completion* is antithetical to a moving-bump *hand-off*.
  2. The Ecker Gaussian near-diagonal CONTINUOUS-TRACK model (`_gap5_ecker_recurrent_replay.py`, 2026-07-25) decodes a
     traveling bump via Bayesian population decode, but its "wave" is a ~1-cell creeping single-fire front (F_active
     ~0.0004, ~0.4 cells/step) -- it produces NO assembly-level POPULATION bursts, so per_asm_active ~ [0,0,0] on the
     assembly scorer this arc requires. (It also self-sustains one traversal, not discrete re-igniting events.)

THIS MODEL (calibrated single-seed-42 first, scratchpad/ecker_calib{3,4}.py):
  - PC = ECKER_CA3_PC AdEx point neurons (committed additive preset), M disjoint block ASSEMBLIES of asm_size cells.
  - MODERATE within-assembly recurrence (w_within): a BRIEF, SELF-TERMINATING population volley per assembly (NOT the
    bistable ~200 latch -- that never segments; NOT single-fire -- that never reaches population level).
  - STRONG forward links A_i->A_{i+1} (w_fwd): the ignited assembly's volley recruits the NEXT assembly as a POPULATION.
  - WEAK reverse links A_i->A_{i-1} (w_rev << w_fwd): the load-bearing WEIGHT ASYMMETRY. Reverse-seeding does NOT cascade
    backward (calib C4-B); symmetrizing (REVERSE-ASYM-LESION) makes a middle seed spread BOTH ways -> forward collapses.
  - Adaptation (ECKER neg-a + spike-triggered b) + AdEx spike-reset refractoriness SELF-TERMINATE each assembly's volley
    -> the bump moves forward and the whole cascade dies -> the net RESTS SILENT between events (calib C4-D: gap=0.0000).
  - IGNITION = a NON-SPECIFIC minimal prefix cue: each SWR period a RANDOM assembly k gets a brief prefix pulse (the seed
    that starts the sequence FROM position k). Because k is RANDOM per event, forward-FROM-SEED >> reverse-FROM-SEED can
    ONLY come from the encoded forward-weight asymmetry -> the REVERSE-ASYM-LESION is the decisive arbiter.

SCORING (reuse-by-import; the order EMERGES from the substrate -- NO host per-step argmax/silence in the loop):
  forward-FROM-SEED (`_score_forward_from_seed`, `_seed_scored_to_std` from _gap5_swr_envelope_replay_derisk) + the
  event detector / order stats (`_detect_sequence_events`, `_event_windows`, `_smooth` from _gap5_sequence_replay_derisk).

ANTI-CHEAT BATTERY (each REBUILT as its own store; the result IS the controls):
  REVERSE-ASYM-LESION (symmetrize between-edges w_rev:=w_fwd:=mean) -> forward MUST COLLAPSE   [KEY GATE]
  SHUFFLED-STORE (permute the between-edge weights) -> order collapses
  PERMUTED-ASSEMBLY (re-score with random assembly labels) -> chance
  NO-BAND (w_within=w_fwd=w_rev=0) -> seeded assembly fires alone, no cascade -> forward collapses
  NO-NOISE / NO-SEED (no prefix cues) -> no events
  ADAPT-LESION (a=0,b=0) -> HONEST report (may co-fire / may be inert -- the mechanism finding says refractoriness also
     terminates; the verdict does NOT hard-gate on it, it is reported)
  FROZEN-plasticity byte-hash (plasticity off -> weights byte-identical every arm)
  NUMPY-REFERENCE GUARD (the rest loop only injects external current -> order is not host-imposed)

Single-seed 42 first (locate the regime); GO iff forward-from-seed rides the asymmetry AND collapses under
REVERSE-ASYM-LESION -> then the 6-seed GO gate (42/43/44/100/101/102, bar >=5/6).
NO `sim/` edit (region framework + inject_explicit_wiring + the committed ECKER_CA3_PC preset only). GPU-preferred.

  Smoke:  SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_ecker_adex_ca3_replay_derisk --seeds 42
  6-seed: SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_ecker_adex_ca3_replay_derisk \
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
# scoring contract (reuse-by-import; NO sim/ edit) -- the same box-smoother the sequence/SWR-envelope scorers use, so
# the per-assembly activity trace + onset detection are byte-consistent with the banked-negative harness they replace.
from research.runners._gap5_sequence_replay_derisk import _smooth  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "ecker_adex_ca3_replay.json"


# ----------------------------------------------------------------------------------------------------------------------
# BUILD the Ecker AdEx CA3 assembly store. `lesion` in {None, "symmetrize", "shuffle", "no_band"}.
# ----------------------------------------------------------------------------------------------------------------------
def build_store(seed, *, m_asm, asm_size, w_within, w_fwd, w_rev, within_density, b_override, a_override,
                n_pvbc, w_pc_pvbc, w_pvbc_pc, ou_sigma, dt, lesion=None):
    cp, _ = get_backend()
    n_pc = m_asm * asm_size + 20
    regions = [BrainRegion(name="pc", n_neurons=n_pc, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    if n_pvbc > 0:
        regions.append(BrainRegion(name="pvbc", n_neurons=n_pvbc, exc_fraction=0.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)   # SET cfg.seed (NOT actual_seed_used -- the no-op field)
    cfg.dt_ms = float(dt); cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.default_neuron_type_adex = NeuronType.ADEX_ECKER_CA3_PC.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    for f in ("enable_homeostasis", "enable_stdp", "enable_hebbian_learning", "enable_structural_plasticity",
              "enable_parameter_heterogeneity"):
        setattr(cfg, f, False)
    cfg.enable_ou_process = ou_sigma > 0; cfg.ou_noise_sigma_pa = float(ou_sigma)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    if b_override is not None:
        b.core_config.adex_b = float(b_override)
    if a_override is not None:
        b.core_config.adex_a = float(a_override)
    pc = np.asarray(b.region_manager.indices("pc"), int)
    pv = np.asarray(b.region_manager.indices("pvbc"), int) if n_pvbc > 0 else np.zeros(0, int)
    asm_local = [np.arange(i * asm_size, (i + 1) * asm_size) for i in range(m_asm)]

    # effective weights under the lesion
    if lesion == "no_band":
        _wi = _wf = _wr = 0.0
    elif lesion == "symmetrize":
        # ASYM-LESION: forward==reverse (both = the between-edge mean, count-weighted) -> direction destroyed, budget kept
        n_fwd = m_asm - 1; n_rev = m_asm - 1
        _mean_between = (n_fwd * w_fwd + n_rev * w_rev) / max(n_fwd + n_rev, 1)
        _wi, _wf, _wr = w_within, _mean_between, _mean_between
    else:
        _wi, _wf, _wr = w_within, w_fwd, w_rev

    rng = np.random.default_rng(seed * 17 + 3)
    pre, post, w = [], [], []
    between_edge_ids = []   # positional ids of between edges appended, for the SHUFFLE lesion
    _eid = [0]

    def blk(src, dst, weight, dens, is_between):
        if weight <= 0:
            return
        for s in src:
            d = dst[dst != s]
            if dens < 1.0:
                d = d[rng.random(len(d)) < dens]
            if len(d) == 0:
                continue
            pre.append(np.full(len(d), pc[s])); post.append(pc[d]); w.append(np.full(len(d), float(weight)))
            if is_between:
                between_edge_ids.append(np.arange(_eid[0], _eid[0] + len(d)))
            _eid[0] += len(d)

    for i in range(m_asm):
        blk(asm_local[i], asm_local[i], _wi, within_density, False)
        if i + 1 < m_asm:
            blk(asm_local[i], asm_local[i + 1], _wf, within_density, True)
        if i - 1 >= 0:
            blk(asm_local[i], asm_local[i - 1], _wr, within_density, True)
    if n_pvbc > 0 and w_pc_pvbc > 0 and w_pvbc_pc > 0:
        allpc = np.arange(m_asm * asm_size)
        for k in range(n_pvbc):
            pre.append(pc[allpc]); post.append(np.full(len(allpc), pv[k])); w.append(np.full(len(allpc), float(w_pc_pvbc)))
            pre.append(np.full(len(allpc), pv[k])); post.append(pc[allpc]); w.append(np.full(len(allpc), float(w_pvbc_pc)))

    if not pre:
        pre_a = np.zeros(0, int); post_a = np.zeros(0, int); w_a = np.zeros(0, float)
    else:
        pre_a = np.concatenate(pre); post_a = np.concatenate(post); w_a = np.concatenate(w)

    # SHUFFLE lesion: permute the between-edge weights AMONG THEMSELVES (same multiset, forward/reverse pairing destroyed)
    n_between = 0
    if lesion == "shuffle" and between_edge_ids:
        bids = np.concatenate(between_edge_ids)
        n_between = int(len(bids))
        vals = w_a[bids].copy()
        np.random.default_rng(seed * 29 + 7).shuffle(vals)
        w_a[bids] = vals

    b.inject_explicit_wiring({"rec": {"pre_indices": pre_a.astype(int).tolist(), "post_indices": post_a.astype(int).tolist(),
                                      "initial_weights": w_a.astype(float).tolist(), "plastic": False, "conn_type": "ff"}})
    n_between_total = int(sum(len(x) for x in between_edge_ids))
    return dict(bridge=b, cp=cp, pc=pc, pv=pv, asm_local=asm_local, m_asm=m_asm, asm_size=asm_size,
                n_pvbc=int(n_pvbc), n_between=n_between_total, n_between_shuffled=n_between,
                eff_w=(_wi, _wf, _wr), lesion=lesion)


# ----------------------------------------------------------------------------------------------------------------------
# REST + non-specific-seeded discrete SWR events. Each swr_period, at phase 0, a RANDOM assembly k gets a brief prefix
# cue (cue_pa to cue_frac of its cells for cue_steps). env_seed_log[event] = k. The loop injects ONLY external current
# (NUMPY-REFERENCE GUARD -- no host per-step per-assembly silence/argmax; the cascade order EMERGES from the weights).
# seed_on=False = NO-SEED control (no prefix cues -> no ignition).
# ----------------------------------------------------------------------------------------------------------------------
def rest_and_replay(store, rest_steps, seed, *, swr_period, cue_pa, cue_steps, cue_frac, seed_on=True, fixed_asm=None):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    n_pc = len(pc)
    # precompute a FIXED cue-cell subset per assembly (device indices); a per-event RNG picks which assembly k to seed
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
            cur_k = int(fixed_asm) if fixed_asm is not None else int(choice_rng.integers(0, m))
            env_seed_log.append(cur_k); n_env += 1
        if seed_on and phase < cue_steps and cur_k is not None:
            bridge.cp_external_input_current[cue_cells_dev[cur_k]] += float(cue_pa)   # non-specific prefix seed
        bridge._run_one_simulation_step()
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[pc].astype(bool)
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    frozen = bool(np.array_equal(w_before, w_after))
    return dict(F=F, env_seed_log=env_seed_log, n_env=n_env, weights_frozen=frozen)


def _score_periods(F, assemblies_local, env_seed_log, swr_period, *, W, active_frac, onset_frac, min_ev_len=6):
    """PERIOD-LOCKED forward-FROM-SEED scoring (robust; no fragile event-detector->envelope mapping). Each SWR period n
    holds EXACTLY one non-specific seed k=env_seed_log[n] at phase 0 and (at most) one cascade. We slice the firing to
    that period window, find the assemblies that reach `active_frac` (peak of the W-summed per-assembly firing / size),
    order them by ONSET (first crossing of onset_frac), and classify the event:
      FORWARD-from-seed  = seeded k has the EARLIEST onset AND the onset-sorted indices are strictly INCREASING and
                           CONTIGUOUS from k (k, k+1, k+2, ...).
      REVERSE-from-seed  = k earliest AND strictly DECREASING contiguous from k (k, k-1, ...).
    Because k is RANDOM per period, forward>>reverse can only come from the encoded forward-weight asymmetry. Returns the
    standard key contract + seed_first_frac + by_seedpos. A period with <2 active assemblies is not a multi event."""
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


def _score(store, r, det, swr_period):
    """Period-locked forward-FROM-SEED scoring on the discrete SWR events."""
    sc = _score_periods(r["F"], store["asm_local"], r["env_seed_log"], swr_period, W=det["W"],
                        active_frac=det["active_frac"], onset_frac=det["onset_frac"])
    return sc, sc


def _permuted_assembly(store, r, seed, det, swr_period):
    """PERMUTED-ASSEMBLY: re-score the SAME firing with random assembly labels -> forward-from-seed contiguity breaks
    (the seeded position k now points to a physically-unrelated assembly)."""
    al = store["asm_local"]
    perm = np.random.default_rng(int(seed) * 5150 + 3).permutation(len(al))
    relabeled = [al[i] for i in perm]
    return _score_periods(r["F"], relabeled, r["env_seed_log"], swr_period, W=det["W"],
                          active_frac=det["active_frac"], onset_frac=det["onset_frac"])


def _weights_diag(store):
    wi, wf, wr = store["eff_w"]
    return dict(w_within=wi, w_fwd=wf, w_rev=wr, ratio=(wf / max(wr, 1e-6)), n_between=store["n_between"],
                m_asm=store["m_asm"], asm_size=store["asm_size"], n_pvbc=store["n_pvbc"])


# ----------------------------------------------------------------------------------------------------------------------
def one_seed(seed, a):
    t0 = time.time()
    out = {"seed": seed}
    det = dict(W=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k, active_frac=a.active_frac, onset_frac=a.onset_frac)
    build_kw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, w_fwd=a.w_fwd, w_rev=a.w_rev,
                    within_density=a.within_density, b_override=a.b_override, a_override=None,
                    n_pvbc=a.n_pvbc, w_pc_pvbc=a.w_pc_pvbc, w_pvbc_pc=a.w_pvbc_pc, ou_sigma=a.ou_sigma, dt=a.dt)
    seed_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac)

    # SEED-CONTROL determinism guard: build twice at this seed, hash firing thresholds -> prove cfg.seed controls neurons
    if a.verify_seed:
        s1 = build_store(seed, lesion=None, **build_kw)
        h1 = float(np.asarray(to_host(s1["bridge"].cp_neuron_firing_thresholds)).sum()) \
            if getattr(s1["bridge"], "cp_neuron_firing_thresholds", None) is not None else None
        s2 = build_store(seed, lesion=None, **build_kw)
        h2 = float(np.asarray(to_host(s2["bridge"].cp_neuron_firing_thresholds)).sum()) \
            if getattr(s2["bridge"], "cp_neuron_firing_thresholds", None) is not None else None
        out["seed_hash_ok"] = (h1 is None) or (h1 == h2)

    # -- REAL: forward-asymmetric store, non-specific random-per-event seeding --
    st = build_store(seed, lesion=None, **build_kw)
    out["store"] = _weights_diag(st)
    r = rest_and_replay(st, a.rest_steps, seed, **seed_kw)
    s_go, sc_go = _score(st, r, det, a.swr_period)
    chance = max(s_go["chance_forward"], 1e-6)
    fwd = s_go["forward_frac"]; rev = s_go["reverse_frac"]; pa = s_go["per_asm_active"]; nmulti = s_go["n_multi"]
    out["go"] = dict(n_events=s_go["n_events"], n_multi=nmulti, forward_frac=fwd, reverse_frac=rev,
                     chance=chance, seed_first_frac=s_go.get("seed_first_frac"), duty_cycle=s_go["duty_cycle"],
                     pop_rate=s_go["pop_rate"], per_asm_active=pa, n_env=r["n_env"], weights_frozen=r["weights_frozen"],
                     by_seedpos=sc_go.get("by_seedpos"))
    print(f"  [seed {seed}] store: within={st['eff_w'][0]} fwd={st['eff_w'][1]} rev={st['eff_w'][2]} "
          f"ratio={out['store']['ratio']:.1f}x n_between={st['n_between']} pvbc={st['n_pvbc']}", flush=True)
    print(f"  [seed {seed}] GO: ev={s_go['n_events']} multi={nmulti} FWD(from-seed)={fwd:.3f} REV={rev:.3f} "
          f"chance={chance:.3f} seed_first={s_go.get('seed_first_frac')} act={pa} duty={s_go['duty_cycle']:.3f} "
          f"n_env={r['n_env']} frozen={r['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEAT arms (each REBUILT) --
    def arm(lesion, tag):
        s = build_store(seed, lesion=lesion, **build_kw)
        rr = rest_and_replay(s, a.rest_steps, seed, **seed_kw)
        sco, _ = _score(s, rr, det, a.swr_period)
        print(f"  [seed {seed}] {tag}: multi={sco['n_multi']} FWD={sco['forward_frac']:.3f} REV={sco['reverse_frac']:.3f} "
              f"act={sco['per_asm_active']} duty={sco['duty_cycle']:.3f} frozen={rr['weights_frozen']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return s, rr, sco

    _, r_sym, s_sym = arm("symmetrize", "REVERSE-ASYM-LESION (symmetrize)")
    _, r_sc, s_scr = arm("shuffle", "SHUFFLED-STORE")
    _, r_nb, s_nb = arm("no_band", "NO-BAND")

    # -- NO-SEED (no prefix cues) -> no events --
    st_ns = build_store(seed, lesion=None, **build_kw)
    r_nsd = rest_and_replay(st_ns, a.rest_steps, seed, seed_on=False, **seed_kw)
    s_nsd, _ = _score(st_ns, r_nsd, det, a.swr_period)
    print(f"  [seed {seed}] NO-SEED: multi={s_nsd['n_multi']} FWD={s_nsd['forward_frac']:.3f} "
          f"act={s_nsd['per_asm_active']} pop={s_nsd['pop_rate']:.5f} ({time.time()-t0:.0f}s)", flush=True)

    # -- ADAPT-LESION (a=0,b=0) -> honest report --
    st_al = build_store(seed, lesion=None, **{**build_kw, "b_override": 0.0, "a_override": 0.0})
    r_al = rest_and_replay(st_al, a.rest_steps, seed, **seed_kw)
    s_al, _ = _score(st_al, r_al, det, a.swr_period)
    print(f"  [seed {seed}] ADAPT-LESION (a=0,b=0): multi={s_al['n_multi']} FWD={s_al['forward_frac']:.3f} "
          f"act={s_al['per_asm_active']} duty={s_al['duty_cycle']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # -- PERMUTED-ASSEMBLY (re-score GO firing with random labels) --
    s_pa = _permuted_assembly(st, r, seed, det, a.swr_period)
    print(f"  [seed {seed}] PERMUTED-ASSEMBLY: multi={s_pa['n_multi']} FWD={s_pa['forward_frac']:.3f} "
          f"REV={s_pa['reverse_frac']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    out["reverse_asym_lesion"] = dict(n_multi=s_sym["n_multi"], forward_frac=s_sym["forward_frac"],
                                      reverse_frac=s_sym["reverse_frac"], per_asm_active=s_sym["per_asm_active"],
                                      weights_frozen=r_sym["weights_frozen"])
    out["shuffled_store"] = dict(n_between=st["n_between"], n_multi=s_scr["n_multi"], forward_frac=s_scr["forward_frac"],
                                 reverse_frac=s_scr["reverse_frac"], per_asm_active=s_scr["per_asm_active"],
                                 weights_frozen=r_sc["weights_frozen"])
    out["no_band"] = dict(n_multi=s_nb["n_multi"], forward_frac=s_nb["forward_frac"],
                          per_asm_active=s_nb["per_asm_active"], weights_frozen=r_nb["weights_frozen"])
    out["no_seed"] = dict(n_multi=s_nsd["n_multi"], forward_frac=s_nsd["forward_frac"], pop_rate=s_nsd["pop_rate"],
                          per_asm_active=s_nsd["per_asm_active"])
    out["adapt_lesion"] = dict(n_multi=s_al["n_multi"], forward_frac=s_al["forward_frac"], duty_cycle=s_al["duty_cycle"],
                               per_asm_active=s_al["per_asm_active"])
    out["permuted_assembly"] = dict(n_multi=s_pa["n_multi"], forward_frac=s_pa["forward_frac"],
                                    reverse_frac=s_pa["reverse_frac"])

    # ================= PER-SEED VERDICT (verify, don't assert) =================
    def _collapsed(s):
        return (s["forward_frac"] <= max(0.5 * fwd, 1.5 * chance)) or (s["n_multi"] == 0)
    forward_ordered = (fwd >= 1.5 * chance and fwd > rev and nmulti >= a.min_multi)
    ignites = (min(pa) >= 1)
    # DISCRETENESS = the net rests silent between events (low duty) AND ignition REQUIRES the seed (NO-SEED is silent).
    # A high forward_frac already implies the assemblies fire SEQUENTIALLY (ordered onsets), NOT co-firing -- so the
    # discreteness gate is the duty cycle + the NO-SEED silence, not an aggregated per-assembly co-fire count (which
    # conflates cross-event activation counts with within-event co-firing). ADAPT-LESION (reported) shows the contrast:
    # without adaptation the duty rises / the order degrades (self-termination is adaptation-assisted here).
    discrete = (s_go["duty_cycle"] <= a.max_duty)
    reverse_lesion_collapses = _collapsed(s_sym)
    shuffled_collapses = _collapsed(s_scr)
    no_band_collapses = _collapsed(s_nb)
    permuted_chance = _collapsed(s_pa)
    no_seed_silent = (s_nsd["n_multi"] == 0) or (s_nsd["forward_frac"] <= 1.5 * chance)
    frozen_ok = bool(r["weights_frozen"] and r_sym["weights_frozen"] and r_sc["weights_frozen"]
                     and r_nb["weights_frozen"] and r_al["weights_frozen"] and r_nsd["weights_frozen"])
    seed_go = bool(forward_ordered and ignites and discrete and reverse_lesion_collapses and shuffled_collapses
                   and no_band_collapses and permuted_chance and no_seed_silent and frozen_ok)
    out["checks"] = dict(forward_ordered=forward_ordered, ignites=ignites, discrete=discrete,
                         reverse_lesion_collapses=reverse_lesion_collapses, shuffled_collapses=shuffled_collapses,
                         no_band_collapses=no_band_collapses, permuted_chance=permuted_chance,
                         no_seed_silent=no_seed_silent, frozen_ok=frozen_ok,
                         adapt_lesion_fwd=round(s_al["forward_frac"], 3),
                         seed_hash_ok=out.get("seed_hash_ok"))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=6, help="number of assemblies (chain length)")
    ap.add_argument("--asm-size", type=int, default=80)
    ap.add_argument("--within-density", type=float, default=0.5)
    ap.add_argument("--rest-steps", type=int, default=13000, help="~40 SWR events at swr_period=325 (robust statistics)")
    ap.add_argument("--dt", type=float, default=0.1, help="Ecker AdEx needs dt<=0.1 (stiff exp term blows up at 0.5)")
    # weights (calibrated seed-42; scratchpad/ecker_calib4.py)
    ap.add_argument("--w-within", type=float, default=60.0)
    ap.add_argument("--w-fwd", type=float, default=800.0)
    ap.add_argument("--w-rev", type=float, default=15.0)
    ap.add_argument("--b-override", type=float, default=120.0, help="spike-triggered adaptation (ECKER default 206.84)")
    # SWR events / non-specific prefix seed
    ap.add_argument("--swr-period", type=int, default=325, help="steps per SWR cycle (event + inter-event silence)")
    ap.add_argument("--cue-pa", type=float, default=9000.0, help="prefix-seed amplitude onto the random assembly")
    ap.add_argument("--cue-steps", type=int, default=40)
    ap.add_argument("--cue-frac", type=float, default=0.6, help="fraction of the seeded assembly's cells the prefix drives")
    ap.add_argument("--ou-sigma", type=float, default=40.0)
    # optional PVBC (honest control; the cascade works without it -- adaptation+refractoriness give discreteness)
    ap.add_argument("--n-pvbc", type=int, default=0)
    ap.add_argument("--w-pc-pvbc", type=float, default=150.0)
    ap.add_argument("--w-pvbc-pc", type=float, default=4.0)
    # detection -- W must BRIDGE the inter-volley gaps of a sequential cascade so the whole A->B->C cascade is ONE
    # multi-assembly event (not split into per-assembly single-assembly events). The assembly onsets stay resolvable
    # because onset = first crossing of onset_frac on the per-assembly smoothed trace (still monotonic across the chain).
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--ev-floor", type=float, default=0.15)
    ap.add_argument("--ev-k", type=float, default=3.0)
    ap.add_argument("--active-frac", type=float, default=0.10)
    ap.add_argument("--onset-frac", type=float, default=0.06)
    ap.add_argument("--min-multi", type=int, default=6, help="min multi-assembly events required for forward_ordered")
    ap.add_argument("--max-duty", type=float, default=0.35, help="discreteness: max fraction of steps with pop activity")
    ap.add_argument("--verify-seed", action="store_true", help="build twice + hash thresholds (seed-controls-substrate)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    print(f"[gap5-ecker-adex] Ecker AdEx CA3 ASSEMBLY replay | n_mem={a.n_mem} asm_size={a.asm_size} "
          f"within={a.w_within} fwd={a.w_fwd} rev={a.w_rev} b={a.b_override} | swr_period={a.swr_period} "
          f"cue={a.cue_pa}x{a.cue_steps}@{a.cue_frac} ou={a.ou_sigma} pvbc={a.n_pvbc} | rest_steps={a.rest_steps} "
          f"dt={a.dt} | seeds={a.seeds} backend={backend}", flush=True)

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
        mf = float(np.mean([p["go"]["forward_frac"] for p in per]))
        mr = float(np.mean([p["go"]["reverse_frac"] for p in per]))
        mch = float(np.mean([p["go"]["chance"] for p in per]))
        msym = float(np.mean([p["reverse_asym_lesion"]["forward_frac"] for p in per]))
        mscr = float(np.mean([p["shuffled_store"]["forward_frac"] for p in per]))
        mnb = float(np.mean([p["no_band"]["forward_frac"] for p in per]))
        mduty = float(np.mean([p["go"]["duty_cycle"] for p in per]))
        pa_mean = [float(np.mean([p["go"]["per_asm_active"][k] for p in per])) for k in range(a.n_mem)]
        fwd_vec = [round(p["go"]["forward_frac"], 3) for p in per]
        if go:
            verdict = (f"ECKER-ADEX-CA3 GO {n_go}/{len(per)} -- the Ecker AdEx CA3 assembly store replays DISCRETE "
                       f"forward SWR events A->B->C from a NON-SPECIFIC random-per-event prefix seed: forward-FROM-SEED "
                       f"{mf:.3f} vs reverse {mr:.3f} vs chance {mch:.3f}; per_asm_active~{[round(x,1) for x in pa_mean]} "
                       f"duty {mduty:.3f}. The order RIDES THE WEIGHT ASYMMETRY: REVERSE-ASYM-LESION collapses forward to "
                       f"{msym:.3f}, SHUFFLED-STORE to {mscr:.3f}, NO-BAND to {mnb:.3f}. forward-frac/seed={fwd_vec}. "
                       f"=> closes the gap#5 SWR forward-replay wall the bistable store could not (it FAILED the "
                       f"reverse-asym-lesion). {'Run the full 6-seed confirm (bar >=5/6).' if len(per) < 6 else ''}")
        else:
            verdict = (f"ECKER-ADEX-CA3 PARTIAL/NEGATIVE {n_go}/{len(per)} -- forward-from-seed {mf:.3f} vs reverse "
                       f"{mr:.3f} vs chance {mch:.3f}; per_asm_active~{[round(x,1) for x in pa_mean]} duty {mduty:.3f}; "
                       f"REVERSE-ASYM-LESION {msym:.3f} SHUFFLED {mscr:.3f} NO-BAND {mnb:.3f}. forward-frac/seed={fwd_vec}. "
                       f"Check per-seed checks for the failing predicate (the KEY gate is reverse_lesion_collapses).")
        # Earn a top-level verdict that TRAVELS with its preconditions (tools.verdict) + an attribution call
        # (tools.lab): the go/no-go must carry what earned it, and the forward ordering must be shown to belong to
        # the encoded weight asymmetry (real store) rather than the prefix cue (reverse-asym-lesioned store).
        v = Verdict("Ecker AdEx CA3: discrete forward SWR replay that RIDES the encoded weight asymmetry", chance=mch)
        v.require("forward events ignite + discrete + forward-ordered on >= bar of the seeds", n_go,
                  expect=lambda n, b=bar: n >= b)
        v.floor("mean forward-from-seed exceeds chance", mf, floor=mch)
        v.require("reverse replay is absent (mean reverse_frac < 0.05)", mr, expect=lambda r: r < 0.05)
        v.control("REVERSE-ASYM-LESION: real-store forward vs symmetrized-store forward (must DIFFER -> the order "
                  "rides the weight asymmetry, not the cue)", treatment=mf, control=msym, min_separation=0.0)
        v.control("SHUFFLED-STORE: real-store forward vs shuffled-store forward", treatment=mf, control=mscr,
                  min_separation=0.0)
        v.control("NO-BAND: real-store forward vs no-band forward", treatment=mf, control=mnb, min_separation=0.0)
        for proc in ("STDP/Hebbian plasticity (weights frozen during replay)", "OU-only spontaneous ignition "
                     "(a prefix cue is required)"):
            v.disabled(proc, why="isolation: the band is hand-wired + frozen and ignition uses a non-specific "
                                 "random-per-event prefix; the order still must ride the frozen asymmetry (the "
                                 "reverse-asym-lesion control tests exactly that)")
        decided = v.decide(go=go, verbose=False)
        attributable_to("forward SWR replay ordering (real store forward vs reverse-asym-lesioned store forward)",
                        mf, msym)
        summary_extra = dict(GO=go, n_go=n_go, status=decided.get("status"),
                             preconditions=decided.get("preconditions", []), decided=decided)
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_go=0)

    summary = {"probe": "gap5_ecker_adex_ca3_replay",
               "mechanism": "Ecker-2022 AdEx CA3 assembly banded recurrence -> discrete forward SWR replay",
               "seeds": a.seeds, "n_mem": a.n_mem, "asm_size": a.asm_size, "rest_steps": a.rest_steps,
               "cfg": dict(w_within=a.w_within, w_fwd=a.w_fwd, w_rev=a.w_rev, b_override=a.b_override,
                           within_density=a.within_density, swr_period=a.swr_period, cue_pa=a.cue_pa,
                           cue_steps=a.cue_steps, cue_frac=a.cue_frac, ou_sigma=a.ou_sigma, n_pvbc=a.n_pvbc, dt=a.dt),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[gap5-ecker-adex] VERDICT: {verdict}\n[gap5-ecker-adex] wrote {a.out}\n"
          + "=" * 120, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
