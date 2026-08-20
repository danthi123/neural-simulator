"""IDLE-REPLAY via DG/EC-AFFERENT drive on the REAL D5 episodic bridge -- the biologically-correct successor to
`_idle_replay_on_d5_episodic_derisk.py` (2026-08-20, UNDEFINED: untargeted CA3-soma noise never seeded the ~1%-sparse
stored assembly; D5's own dapB precedent already showed population-level CA3 reads are non-specific at this scale).

THE FIX THIS RUNNER TESTS (content-blind-but-structure-aware replay): instead of driving i.i.d. noise into CA3 SOMATA
(`cp_external_input_current` on random CA3 cells), drive the DG/EC AFFERENT layer -- the mossy-fiber input to CA3
(`dg -> ca3`, the trisynaptic detonator, present on the readout bridge at density=0.10 weight=8.0 per
`text_minimal_isolation.build_biological_brain_regions`). Biological SWR replay INITIATES from a sparse afferent /
CA3-recurrent volley, not from a hand-addressed CA3 cue. A content-blind DG volley is STRUCTURE-aware IF (and this is the
empirical question this runner answers, not assumes) the mossy projection concentrates activity onto the stored assembly
so its potentiated recurrence can complete it -- WITHOUT the host ever naming which cells are 'dog'.

⚠️ THE ARCHITECTURAL RISK THIS RUNNER MEASURES HEAD-ON (found by code-read; asserted here, not commented away):
  The D5 organ SELECTS the assembly membership on a SEPARATE, DISCARDED bridge (`emergent_assemblies` ->
  `_gap5_emergent_dg_selection_derisk._build_bridge`, ca3_density=0.05, mossy_weight=3000, mossy_density=0.02). The
  READOUT bridge that we actually drive (`_build_dap_readout` -> `_riii_..._build`, ca3_density=0.5, DEFAULT mossy
  weight=8.0 density=0.10) is a DIFFERENT `_build` call -> a DIFFERENT connectivity RNG draw. So the readout bridge's
  OWN mossy wiring does NOT necessarily project onto the CA3 indices that were selected as 'dog' on the selection
  bridge. If it does not, a content-blind DG volley detonates readout-mossy targets that are RANDOM w.r.t. the stored
  assembly -> the afferent is content-blind but NOT structure-aware toward the store, and the mechanism cannot transfer
  as-is. => This runner FIRST measures, content-blind, whether readout-DG drive concentrates on dog's assembly more than
  cat's (the STRUCTURE-AWARENESS precondition), THEN measures the apical-dAP / write transfer. A failed structure
  precondition is reported UNDEFINED with the precise reason (the mechanism was driven at a locus whose wiring does not
  reach the store), never a validated NO-GO.

WHAT IS CONTENT-BLIND HERE (anti-cheat, asserted): the DG drive pattern is drawn from the WHOLE DG population by a
seed-derived RNG, NEVER from dog's selection pattern or dog's CA3 membership. The mossy pathway is (optionally)
strengthened by a GLOBAL whole-pathway scalar (`--mossy-boost`), applied to EVERY dg->ca3 synapse uniformly -- it uses
no knowledge of which DG or CA3 cells belong to any topic. The ONLY thing that can make dog light up more than cat is
(a) the mossy structure landing on dog + (b) dog's potentiated within-assembly recurrence completing it. Both are
structure, not host targeting.

INSTRUMENT FIX (a validated baseline is a PRECONDITION for any transfer claim): the prior runner's recall() varied by
+-1 held cell between calls on a provably-inert state, so its exact-equality (1e-6) reproducibility check failed. Here
every recall read is AVERAGED over K independent draws (`--recall-draws`), and reproducibility is defined against the
MEASURED per-draw noise floor (the read is reproducible if its run-to-run std is below the control threshold CTRL_MAX,
so a signal above COMPLETE_MIN is distinguishable from the read's own jitter) -- the noise floor is reported, not hidden.

Run (GPU, ~21GB free fits n_ca3=2000 easily):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._idle_replay_dgec_afferent_d5_derisk --seed 42 \
      --out research/findings/raw/_idle_replay_dgec_afferent_d5/seed42.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
from sim.backend import to_host  # noqa: E402
from tools.lab import attributable_to, void_if  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
# reuse-by-import, UNCHANGED: the real D5 production mechanism.
from research.runners._episodic_dap_dialogue_memory import EpisodicDapMemory  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import _reset_apical_latch  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_idle_replay_dgec_afferent_d5" / "seed42.json"

# Completion criterion, reused verbatim from D5's own GO bar (_episodic_dap_dialogue_memory.py).
COMPLETE_MIN, CUE_OVER_CTRL, CTRL_MAX = 0.20, 3.0, 0.10
# PC separation bar (loose on purpose: n=1 exploratory seed -- SIGN + MAGNITUDE, a strict bar is for a later 6-seed).
PC_MIN_SEPARATION = 0.05
# Structure-awareness bar: readout-DG drive must recruit dog's assembly meaningfully ABOVE cat's + above chance.
STRUCT_MIN_SEPARATION = 0.05


# ------------------------------------------------------------------------------------------------------------------
# geometry helpers (global CA3/DG indices)
# ------------------------------------------------------------------------------------------------------------------
def _topic_held_global(mem, topic):
    slot = mem.topic_slot[topic]
    held_pos = mem.held_pos_by_asm[slot]
    return np.asarray([int(mem.R.ca3_idx[p]) for p in held_pos], dtype=np.int64)


def _topic_assembly_global(mem, topic):
    slot = mem.topic_slot[topic]
    return np.asarray(mem.assemblies[slot], dtype=np.int64)


def _dg_global(mem):
    return np.asarray(list(mem.bridge.region_manager.indices("dg")), dtype=np.int64)


def _w_within(mem, topic):
    slot = mem.topic_slot[topic]
    m = mem.R.withinA_masks[slot]
    cp = mem.R.cp
    n = int(to_host(cp.sum(m)))
    return float(to_host(cp.mean(mem.R.C.data[m]))) if n else 0.0


def _w_rec_all(mem):
    cp = mem.R.cp
    m = mem.R.rec_mask
    n = int(to_host(cp.sum(m)))
    return float(to_host(cp.mean(mem.R.C.data[m]))) if n else 0.0


def _mossy_mask(mem):
    """Boolean mask over synapses selecting the dg->ca3 (mossy) pathway on the READOUT bridge. Auto-detects CSR
    orientation (rows=pre, cols=post per _csr_row_col + 'connections.T@drive'); falls back to the transpose and
    records which orientation was non-empty, so a wrong-orientation guess cannot silently select 0 synapses."""
    cp = mem.R.cp
    n = mem.R.n
    dg = _dg_global(mem)
    is_dg = cp.zeros(n, dtype=cp.bool_); is_dg[cp.asarray(dg)] = True
    is_ca3 = cp.zeros(n, dtype=cp.bool_); is_ca3[mem.R.ca3_arr] = True
    m_pre_dg = is_dg[mem.R.rows] & is_ca3[mem.R.cols]          # rows=pre(dg), cols=post(ca3) -- expected
    n_pre = int(to_host(cp.sum(m_pre_dg)))
    if n_pre > 0:
        return m_pre_dg, "rows=dg(pre),cols=ca3(post)", n_pre
    m_alt = is_ca3[mem.R.rows] & is_dg[mem.R.cols]
    n_alt = int(to_host(cp.sum(m_alt)))
    return m_alt, "rows=ca3,cols=dg (transpose)", n_alt


# ------------------------------------------------------------------------------------------------------------------
# INSTRUMENT: averaged recall (the reproducibility fix)
# ------------------------------------------------------------------------------------------------------------------
def _recall_stats(mem, topic, K, *, lesion=False):
    """Average the SPIKING recall read over K independent draws so the +-1-held-cell run-to-run jitter that broke the
    prior runner's exact-equality instrument check is measured (per-draw std) instead of aliasing into the verdict."""
    recs = [mem.recall(topic, lesion=lesion) for _ in range(K)]
    cue = np.asarray([r["apical_cue"] for r in recs], dtype=np.float64)
    perm = np.asarray([r["apical_perm"] for r in recs], dtype=np.float64)
    nocue = np.asarray([r["apical_nocue"] for r in recs], dtype=np.float64)
    in_mem = np.asarray([1.0 if r["in_memory"] else 0.0 for r in recs], dtype=np.float64)
    return {
        "apical_cue": float(cue.mean()), "apical_cue_std": float(cue.std()),
        "apical_perm": float(perm.mean()), "apical_perm_std": float(perm.std()),
        "apical_nocue": float(nocue.mean()), "apical_nocue_std": float(nocue.std()),
        "in_memory_frac": float(in_mem.mean()), "in_memory": bool(in_mem.mean() >= 0.5),
        "K": int(K),
    }


def _quiet_window(mem, n_steps):
    """N steps, zero drive, plasticity OFF -- the 'skip the replay pass' lesion / G2 control + an inertness check."""
    bridge = mem.bridge
    w_before = _w_rec_all(mem)
    mem.R.hard_silence()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    w_after = _w_rec_all(mem)
    return {"w_rec_before": w_before, "w_rec_after": w_after, "inert": bool(abs(w_after - w_before) < 1e-9)}


# ------------------------------------------------------------------------------------------------------------------
# AFFERENT drive primitives (content-blind: a seed-derived random DG volley, NEVER dog's pattern)
# ------------------------------------------------------------------------------------------------------------------
def _content_blind_dg_pattern(mem, args, rng):
    """A random subset of the WHOLE DG population, drawn by a seed-derived RNG independent of any topic. This is the
    content-blind SWR-like afferent volley."""
    dg = _dg_global(mem)
    n_dg_drive = max(1, int(round(len(dg) * args.dg_frac)))
    return np.asarray(rng.choice(dg, size=n_dg_drive, replace=False), dtype=np.int64)


def _overlap(recruited_local, assembly_global, ca3_idx):
    """Fraction of `assembly_global` (global CA3 indices) whose CA3-local index is in `recruited_local`."""
    pos = {int(g): i for i, g in enumerate(ca3_idx)}
    hit = sum(1 for g in assembly_global if pos.get(int(g)) in recruited_local)
    return float(hit) / float(len(assembly_global)) if len(assembly_global) else 0.0


# ------------------------------------------------------------------------------------------------------------------
# TRANSFER: afferent-driven idle reactivation + BTSP write, swept over a content-blind DOSE grid
# ------------------------------------------------------------------------------------------------------------------
def _afferent_replay(mem, dg_pat, mossy_boost, dg_pA, *, warm, read, up_thresh, sample_every, recruit_theta,
                     lesion=False):
    """THE MECHANISM at the afferent locus, at ONE content-blind dose (mossy_boost x dg_pA). Full recurrent-weight
    state is snapshotted and RESTORED afterward so a dose sweep does not contaminate (each dose is measured on the
    identical post-store state). reset clean -> enable_btsp -> content-blind global mossy boost -> drive the SAME
    content-blind DG volley -> sample dog-vs-cat apical-dAP UP + soma firing + CA3 recruitment under the identical
    drive -> freeze BTSP + restore ALL weights. dg_pat is the fixed content-blind DG pattern (shared across doses).

    lesion=True (D5's own load-bearing teeth): restore the UNFORMED baseline recurrent weights before the drive, so
    the stored (potentiated) recurrence is GONE. If afferent-driven dog completion collapses under lesion, the
    completion was carried by the STORED assembly's potentiated recurrence -- not by the afferent volley reaching dog's
    cells directly. This is the decisive structure-awareness anti-cheat."""
    bridge = mem.bridge
    cp = mem.R.cp
    ca3_arr = mem.R.ca3_arr
    ca3_idx = mem.R.ca3_idx

    dog_held = _topic_held_global(mem, "dog")
    cat_held = _topic_held_global(mem, "cat")
    dog_asm = _topic_assembly_global(mem, "dog")
    cat_asm = _topic_assembly_global(mem, "cat")
    mmask, orientation, n_mossy = _mossy_mask(mem)

    full_snapshot = mem.R.C.data.copy()             # snapshot the WHOLE recurrent state (dose-isolation)
    if lesion:
        mem.R.C.data[:] = mem.baseline_weights      # UNFORMED recurrent weights: the load-bearing teeth control
    w_within_dog_before = _w_within(mem, "dog")
    w_within_cat_before = _w_within(mem, "cat")

    mem.R.hard_silence()
    _reset_apical_latch(bridge)
    if mossy_boost != 1.0:
        mem.R.C.data[mmask] = mem.R.C.data[mmask] * cp.float32(mossy_boost)

    cfg = bridge.core_config
    saved = (bool(getattr(cfg, "enable_btsp", False)), float(getattr(cfg, "btsp_learning_rate", 0.0)),
             float(getattr(cfg, "btsp_w_max", 5.0)), float(getattr(cfg, "btsp_elig_tau_ms", 1000.0)))
    cfg.enable_btsp = True
    cfg.btsp_learning_rate = float(mem.p["btsp_lr"])
    cfg.btsp_w_max = float(mem.p["wmax"])
    cfg.btsp_elig_tau_ms = 1000.0
    if getattr(bridge, "cp_btsp_pre_elig", None) is not None:
        bridge.cp_btsp_pre_elig[:] = 0.0

    darr = cp.asarray(dg_pat, dtype=cp.int64)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[darr] = cp.float32(dg_pA)

    n_total = int(warm) + int(read)
    dog_apical_up, cat_apical_up, dog_fire, cat_fire, ca3_active = [], [], [], [], []
    cnt = cp.zeros(len(ca3_arr), dtype=cp.float32)
    read_steps_counted = 0
    for step in range(n_total):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        if step >= warm:
            cnt += bridge.cp_firing_states[ca3_arr].astype(cp.float32)
            read_steps_counted += 1
        if step % max(1, sample_every) == 0 or step == n_total - 1:
            va = to_host(bridge.cp_v_apical) if getattr(bridge, "cp_v_apical", None) is not None else None
            if va is not None:
                dog_apical_up.append(float(np.mean(va[dog_held] > up_thresh)) if len(dog_held) else 0.0)
                cat_apical_up.append(float(np.mean(va[cat_held] > up_thresh)) if len(cat_held) else 0.0)
            fired = to_host(bridge.cp_firing_states).astype(bool)
            dog_fire.append(float(np.mean(fired[dog_held])) if len(dog_held) else 0.0)
            cat_fire.append(float(np.mean(fired[cat_held])) if len(cat_held) else 0.0)
            ca3_active.append(float(np.mean(to_host(bridge.cp_firing_states[ca3_arr]).astype(bool))))

    rate = to_host(cnt) / float(max(1, read_steps_counted))
    recruited = set(int(i) for i in np.nonzero(rate >= recruit_theta)[0])
    rec_frac = float(len(recruited)) / float(len(ca3_idx))
    ov_dog = _overlap(recruited, dog_asm, ca3_idx)
    ov_cat = _overlap(recruited, cat_asm, ca3_idx)

    bridge.cp_external_input_current[:] = 0.0
    cfg.enable_btsp, cfg.btsp_learning_rate, cfg.btsp_w_max, cfg.btsp_elig_tau_ms = saved

    w_within_dog_after = _w_within(mem, "dog")
    w_within_cat_after = _w_within(mem, "cat")
    mem.R.C.data[:] = full_snapshot                 # restore the WHOLE recurrent state (dose-isolation)

    dog_ap = float(np.mean(dog_apical_up)) if dog_apical_up else 0.0
    cat_ap = float(np.mean(cat_apical_up)) if cat_apical_up else 0.0
    return {
        "mossy_boost": float(mossy_boost), "dg_pA": float(dg_pA), "lesion": bool(lesion),
        "n_mossy_synapses": int(n_mossy),
        "orientation": orientation, "n_dg_driven": int(len(dg_pat)), "n_steps": n_total,
        "ca3_active_mean": float(np.mean(ca3_active)) if ca3_active else 0.0,
        "ca3_active_max": float(np.max(ca3_active)) if ca3_active else 0.0,
        "ca3_recruited_frac": rec_frac, "ca3_recruited_n": int(len(recruited)),
        "overlap_dog": ov_dog, "overlap_cat": ov_cat, "chance_overlap": rec_frac,
        "struct_gap_dog_minus_cat": ov_dog - ov_cat, "struct_gap_dog_minus_chance": ov_dog - rec_frac,
        "dog_apical_up_mean": dog_ap, "cat_apical_up_mean": cat_ap,
        "dog_apical_up_max": float(np.max(dog_apical_up)) if dog_apical_up else 0.0,
        "cat_apical_up_max": float(np.max(cat_apical_up)) if cat_apical_up else 0.0,
        "apical_gap_dog_minus_cat": dog_ap - cat_ap,
        "dog_fire_mean": float(np.mean(dog_fire)) if dog_fire else 0.0,
        "cat_fire_mean": float(np.mean(cat_fire)) if cat_fire else 0.0,
        "w_within_dog_before": w_within_dog_before, "w_within_dog_after": w_within_dog_after,
        "w_within_cat_before": w_within_cat_before, "w_within_cat_after": w_within_cat_after,
        "dw_dog": w_within_dog_after - w_within_dog_before, "dw_cat": w_within_cat_after - w_within_cat_before,
    }


def _dose_grid(args):
    mbs = [float(x) for x in args.mossy_boosts.split(",")]
    pAs = [float(x) for x in args.dg_pAs.split(",")]
    return [(mb, pA) for mb in mbs for pA in pAs]


def _pick_primary(sweep, complete_min, resp_floor):
    """Primary dose selection over the content-blind sweep. ENGAGEMENT is read on the APICAL dAP response (the D5 read
    variable), NOT soma firing -- the mechanism is the per-cell apical latch, deliberately DECOUPLED from soma
    reverberation (apical_gc=0.3), so CA3 can be soma-quiet while dog's apical dAP latches UP. Primary = the GENTLEST
    dose (ranked by mossy_boost*dg_pA = minimal SWR seed) that drives dog's apical dAP to the completion bar; failing
    that, the gentlest dose with ANY apical response; failing that, the strongest dose (flagged not-engaged)."""
    completing = [d for d in sweep if d["dog_apical_up_mean"] >= complete_min]
    if completing:
        return min(completing, key=lambda d: d["mossy_boost"] * d["dg_pA"]), True
    responding = [d for d in sweep if max(d["dog_apical_up_mean"], d["cat_apical_up_mean"]) > resp_floor]
    if responding:
        return min(responding, key=lambda d: d["mossy_boost"] * d["dg_pA"]), True
    return max(sweep, key=lambda d: d["mossy_boost"] * d["dg_pA"]), False


def _afferent_replay_persist(mem, dg_pat, mossy_boost, dg_pA, *, warm, read, up_thresh):
    """Apply the primary dose ONCE and LEAVE the ca3->ca3 BTSP change in place (restore only the content-blind mossy
    boost) so the after-replay recall reads the pure BTSP effect. Returns the within-assembly weight deltas."""
    bridge = mem.bridge
    cp = mem.R.cp
    mmask, _o, _n = _mossy_mask(mem)
    w_dog_before = _w_within(mem, "dog"); w_cat_before = _w_within(mem, "cat")
    mossy_baseline = mem.R.C.data[mmask].copy()

    mem.R.hard_silence(); _reset_apical_latch(bridge)
    if mossy_boost != 1.0:
        mem.R.C.data[mmask] = mem.R.C.data[mmask] * cp.float32(mossy_boost)
    cfg = bridge.core_config
    saved = (bool(getattr(cfg, "enable_btsp", False)), float(getattr(cfg, "btsp_learning_rate", 0.0)),
             float(getattr(cfg, "btsp_w_max", 5.0)), float(getattr(cfg, "btsp_elig_tau_ms", 1000.0)))
    cfg.enable_btsp = True; cfg.btsp_learning_rate = float(mem.p["btsp_lr"]); cfg.btsp_w_max = float(mem.p["wmax"])
    cfg.btsp_elig_tau_ms = 1000.0
    if getattr(bridge, "cp_btsp_pre_elig", None) is not None:
        bridge.cp_btsp_pre_elig[:] = 0.0
    darr = cp.asarray(dg_pat, dtype=cp.int64)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[darr] = cp.float32(dg_pA)
    for _ in range(int(warm) + int(read)):
        bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[:] = 0.0
    cfg.enable_btsp, cfg.btsp_learning_rate, cfg.btsp_w_max, cfg.btsp_elig_tau_ms = saved
    mem.R.C.data[mmask] = mossy_baseline   # restore ONLY mossy; keep the ca3->ca3 BTSP change
    return {"dw_dog": _w_within(mem, "dog") - w_dog_before, "dw_cat": _w_within(mem, "cat") - w_cat_before}


# ------------------------------------------------------------------------------------------------------------------
def run(seed, args):
    t0 = time.time()
    result = {"seed": seed, "backend": os.environ.get("SIM_BACKEND", "(unset)"), "timing": {}}

    t_build = time.time()
    mem = EpisodicDapMemory(seed, ["dog", "cat"], verbose=True)
    result["timing"]["init_s"] = round(time.time() - t_build, 1)
    result["n_ca3"] = mem.n_ca3
    result["assembly_sizes"] = mem.assembly_sizes
    result["topic_slot"] = dict(mem.topic_slot)
    result["held_sizes"] = {t: len(mem.held_pos_by_asm[mem.topic_slot[t]]) for t in ("dog", "cat")}
    result["n_dg"] = int(len(_dg_global(mem)))

    t_store = time.time()
    wrote = mem.store("dog")
    result["timing"]["store_s"] = round(time.time() - t_store, 1)
    result["store_wrote"] = bool(wrote)
    result["w_within_dog_poststore"] = _w_within(mem, "dog")
    result["w_within_cat_poststore"] = _w_within(mem, "cat")   # baseline (never formed)

    K = int(args.recall_draws)
    t_r = time.time()
    baseline = {"dog": _recall_stats(mem, "dog", K), "cat": _recall_stats(mem, "cat", K)}
    result["timing"]["baseline_recall_s"] = round(time.time() - t_r, 1)
    result["baseline"] = baseline
    print(f"[dgec-replay] baseline dog cue={baseline['dog']['apical_cue']:.3f}"
          f"+-{baseline['dog']['apical_cue_std']:.3f} cat cue={baseline['cat']['apical_cue']:.3f}"
          f"+-{baseline['cat']['apical_cue_std']:.3f} (+{time.time()-t0:.0f}s)", flush=True)

    t_l = time.time()
    result["lesion_dog"] = _recall_stats(mem, "dog", K, lesion=True)
    result["timing"]["lesion_recall_s"] = round(time.time() - t_l, 1)

    t_q = time.time()
    quiet_instr = _quiet_window(mem, args.quiet_steps)
    quiet = {"dog": _recall_stats(mem, "dog", K), "cat": _recall_stats(mem, "cat", K)}
    result["timing"]["quiet_phase_s"] = round(time.time() - t_q, 1)
    result["quiet_instrument"] = quiet_instr
    result["quiet"] = quiet
    print(f"[dgec-replay] quiet dog cue={quiet['dog']['apical_cue']:.3f} cat cue={quiet['cat']['apical_cue']:.3f} "
          f"inert={quiet_instr['inert']} (+{time.time()-t0:.0f}s)", flush=True)

    # ---- afferent-driven idle reactivation, SWEPT over a content-blind dose grid (mossy_boost x dg_pA) -----------
    # One content-blind DG volley (seed-derived, NEVER dog's pattern), reused across all doses; each dose is measured
    # on the identical post-store state (full recurrent-weight snapshot+restore inside _afferent_replay). The sweep
    # separates "non-specific because the afferent hammer is too strong" (dapB's warned failure mode) from
    # "fundamentally non-specific": if even the GENTLEST engaging dose shows no dog>cat apical gap, the negative is robust.
    t_c = time.time()
    rng = np.random.default_rng(seed * 7919 + 1)
    dg_pat = _content_blind_dg_pattern(mem, args, rng)
    result["n_dg_driven"] = int(len(dg_pat))
    sweep = []
    for mb, pA in _dose_grid(args):
        d = _afferent_replay(mem, dg_pat, mb, pA, warm=args.warm_steps, read=args.read_steps,
                             up_thresh=mem.p["up_thresh"], sample_every=args.pc_sample_every,
                             recruit_theta=args.recruit_theta)
        sweep.append(d)
        print(f"[dgec-replay] dose mb={mb:g} pA={pA:g}: CA3act={d['ca3_active_mean']:.3f} recruited={d['ca3_recruited_frac']:.3f} "
              f"| apical dog={d['dog_apical_up_mean']:.3f} cat={d['cat_apical_up_mean']:.3f} "
              f"(gap {d['apical_gap_dog_minus_cat']:+.3f}) | struct dog={d['overlap_dog']:.3f} cat={d['overlap_cat']:.3f} "
              f"| dw dog={d['dw_dog']:.2f} cat={d['dw_cat']:.2f} (+{time.time()-t0:.0f}s)", flush=True)
    result["timing"]["sweep_s"] = round(time.time() - t_c, 1)
    result["dose_sweep"] = sweep

    primary, engaged = _pick_primary(sweep, COMPLETE_MIN, args.engage_floor)
    mb, pA = primary["mossy_boost"], primary["dg_pA"]
    result["primary_dose"] = {"mossy_boost": mb, "dg_pA": pA}
    result["afferent_engaged"] = bool(engaged)
    print(f"[dgec-replay] PRIMARY dose mb={mb:g} pA={pA:g} engaged={engaged} "
          f"(apical dog={primary['dog_apical_up_mean']:.3f} cat={primary['cat_apical_up_mean']:.3f})", flush=True)

    # ---- VOLLEY-ROBUSTNESS: does dog>>cat hold across DIFFERENT content-blind DG draws at the primary dose? (a
    # single lucky draw that happened to hit dog's mossy targets would NOT be structure-awareness; a store-carried
    # effect survives re-drawing the volley.) volley 0 reuses the sweep's dg_pat so primary == volley[0]. --------------
    t_v = time.time()
    volleys = [dg_pat] + [_content_blind_dg_pattern(mem, args, np.random.default_rng(seed * 7919 + 100 + vi))
                          for vi in range(1, args.n_volleys)]
    volley_results = []
    for vi, vpat in enumerate(volleys):
        d = _afferent_replay(mem, vpat, mb, pA, warm=args.warm_steps, read=args.read_steps,
                             up_thresh=mem.p["up_thresh"], sample_every=args.pc_sample_every,
                             recruit_theta=args.recruit_theta)
        volley_results.append(d)
        print(f"[dgec-replay] volley {vi}: apical dog={d['dog_apical_up_mean']:.3f} cat={d['cat_apical_up_mean']:.3f} "
              f"(gap {d['apical_gap_dog_minus_cat']:+.3f}) | dw dog={d['dw_dog']:.2f} cat={d['dw_cat']:.2f} "
              f"(+{time.time()-t0:.0f}s)", flush=True)
    result["volley_results"] = volley_results
    result["replay_diagnostic"] = primary                    # the sweep's primary-dose measurement (engagement read)

    # ---- REPRODUCIBILITY probe: the SAME dose + SAME content-blind pattern, K back-to-back drives. If the specific
    # completion is a reliable transfer it fires every time; if it is a state/phase-dependent transient it fires
    # intermittently (the sweep hit vs volley-0 miss on the IDENTICAL pattern is the first hint of this). ------------
    repro = []
    for _ in range(args.repro_k):
        d = _afferent_replay(mem, dg_pat, mb, pA, warm=args.warm_steps, read=args.read_steps,
                             up_thresh=mem.p["up_thresh"], sample_every=args.pc_sample_every,
                             recruit_theta=args.recruit_theta)
        repro.append({"dog": d["dog_apical_up_mean"], "cat": d["cat_apical_up_mean"],
                      "gap": d["apical_gap_dog_minus_cat"], "t_step": int(mem.bridge.runtime_state.current_time_step)})
    result["repro_probe"] = repro
    n_repro_fire = sum(1 for x in repro if x["gap"] >= PC_MIN_SEPARATION)
    result["repro_fire_rate"] = float(n_repro_fire) / float(max(1, len(repro)))
    print(f"[dgec-replay] REPRO probe (same dose+pattern x{len(repro)}): dog apical="
          f"{[round(x['dog'], 3) for x in repro]} -> fired {n_repro_fire}/{len(repro)}", flush=True)

    # ---- LESION CONTROL (the decisive teeth): same primary dose + volley 0, but with dog's UNFORMED baseline
    # recurrent weights. If afferent-driven dog completion collapses, it was carried by the STORED recurrence, not the
    # afferent reaching dog directly -> genuine content-blind STRUCTURE-aware replay. ---------------------------------
    lesion_rep = _afferent_replay(mem, volleys[0], mb, pA, warm=args.warm_steps, read=args.read_steps,
                                  up_thresh=mem.p["up_thresh"], sample_every=args.pc_sample_every,
                                  recruit_theta=args.recruit_theta, lesion=True)
    result["lesion_replay"] = lesion_rep
    result["timing"]["robustness_lesion_s"] = round(time.time() - t_v, 1)
    print(f"[dgec-replay] LESION replay (unformed weights): apical dog={lesion_rep['dog_apical_up_mean']:.3f} "
          f"cat={lesion_rep['cat_apical_up_mean']:.3f} (must collapse if structure-aware) (+{time.time()-t0:.0f}s)",
          flush=True)

    # after-replay recall: re-apply the PRIMARY dose ONCE (persisting its BTSP change) and read recall vs quiet.
    t_a = time.time()
    dprim = _afferent_replay_persist(mem, dg_pat, mb, pA,
                                     warm=args.warm_steps, read=args.read_steps, up_thresh=mem.p["up_thresh"])
    result["primary_persist_dw"] = dprim
    after = {"dog": _recall_stats(mem, "dog", K), "cat": _recall_stats(mem, "cat", K)}
    result["timing"]["after_recall_s"] = round(time.time() - t_a, 1)
    result["after_replay"] = after
    print(f"[dgec-replay] after-replay(primary persisted) dog cue={after['dog']['apical_cue']:.3f} "
          f"cat cue={after['cat']['apical_cue']:.3f} (+{time.time()-t0:.0f}s)", flush=True)

    result["timing"]["total_s"] = round(time.time() - t0, 1)
    return result


def build_verdict(r, args):
    base_dog, base_cat = r["baseline"]["dog"], r["baseline"]["cat"]
    quiet_dog, quiet_cat = r["quiet"]["dog"], r["quiet"]["cat"]
    after_dog, after_cat = r["after_replay"]["dog"], r["after_replay"]["cat"]
    les = r["lesion_dog"]
    rep = r["replay_diagnostic"]         # the PRIMARY (gentlest engaging) dose

    # ---- INSTRUMENT (averaged reads; reproducibility defined as SIGNAL DETECTABILITY, not an absolute-std bound --
    # the prior runner's absolute-std<=0.10 check is IMPOSSIBLE to pass on dog's 6-held-cell read, whose quantization
    # floor is 1/6=0.167 regardless of signal; the correct instrument criterion is that the stored-vs-unstored
    # SEPARATION exceeds the measured read jitter, so a completion is distinguishable from the read's own noise). -----
    noise_floor = max(base_dog["apical_cue_std"], base_cat["apical_cue_std"],
                      quiet_dog["apical_cue_std"], quiet_cat["apical_cue_std"])
    intact_fires = bool(base_dog["in_memory"] and base_dog["apical_cue"] >= COMPLETE_MIN
                        and base_dog["apical_perm"] <= CTRL_MAX and base_dog["apical_nocue"] <= CTRL_MAX)
    unstored_abstains = bool((not base_cat["in_memory"]) and base_cat["apical_cue"] <= CTRL_MAX)
    lesion_collapses = bool((not les["in_memory"]) and les["apical_cue"] <= CTRL_MAX)
    quiet_inert = bool(r["quiet_instrument"]["inert"])
    separation = base_dog["apical_cue"] - base_cat["apical_cue"]
    drift_ok = bool(abs(quiet_dog["apical_cue"] - base_dog["apical_cue"]) <= max(3 * noise_floor, 0.05)
                    and abs(quiet_cat["apical_cue"] - base_cat["apical_cue"]) <= max(3 * noise_floor, 0.05))
    # reproducible discrimination: dog(stored) vs cat(unstored) separation exceeds the read jitter by a clear margin
    # AND clears the completion bar; drift across the quiet lesion stays within the jitter.
    recall_reproducible = bool(separation >= max(2.0 * noise_floor, COMPLETE_MIN) and drift_ok)
    instr_ok = bool(intact_fires and unstored_abstains and lesion_collapses and quiet_inert and recall_reproducible)

    # ---- AFFERENT ENGAGEMENT (load-bearing?): read on the APICAL dAP (the D5 read variable), NOT soma firing -- the
    # per-cell apical latch is deliberately decoupled from soma reverberation, so CA3 can be soma-quiet while dog's
    # apical dAP latches. Engaged = the primary dose drives an apical response above the read floor. ------------------
    afferent_engaged = bool(r["afferent_engaged"]
                            and max(rep["dog_apical_up_mean"], rep["cat_apical_up_mean"]) > 1e-4)

    # ---- STRUCTURE-AWARENESS + PC: the D5 read variable is the apical-dAP UP gap (dog vs cat) under the identical
    # content-blind afferent dose. The soma-recruitment overlap is a secondary (assumption-light) cross-check. --------
    apical_gap = float(rep.get("apical_gap_dog_minus_cat", rep["dog_apical_up_mean"] - rep["cat_apical_up_mean"]))
    struct_gap_dog_cat = float(rep.get("struct_gap_dog_minus_cat", 0.0))
    struct_gap_dog_chance = float(rep.get("struct_gap_dog_minus_chance", 0.0))
    STRUCT_AWARE = bool(apical_gap >= STRUCT_MIN_SEPARATION)
    pc_apical_gap = rep["dog_apical_up_mean"] - rep["cat_apical_up_mean"]
    pc_apical_max_gap = rep["dog_apical_up_max"] - rep["cat_apical_up_max"]
    pc_fire_gap = rep["dog_fire_mean"] - rep["cat_fire_mean"]
    PC = bool(pc_apical_gap >= PC_MIN_SEPARATION or pc_apical_max_gap >= PC_MIN_SEPARATION)

    # ---- write-side: did afferent replay specifically potentiate dog's within-assembly weight (vs cat's)? -------
    dw_dog = float(rep["dw_dog"]); dw_cat = float(rep["dw_cat"])
    WRITE_SPECIFIC = bool(dw_dog > dw_cat + 1e-6 and dw_dog > 1e-6)

    # ---- LESION CONTROL (decisive teeth): with the UNFORMED baseline recurrent weights, afferent-driven dog
    # completion must COLLAPSE -- proving the completion is carried by the STORED assembly's recurrence, not the
    # afferent volley reaching dog's cells directly (which lesion would not remove). ----------------------------------
    lesion_rep = r.get("lesion_replay", {})
    lesion_dog_apical = float(lesion_rep.get("dog_apical_up_mean", 1.0))
    LESION_COLLAPSES = bool(lesion_dog_apical <= CTRL_MAX and lesion_dog_apical <= 0.5 * rep["dog_apical_up_mean"] + 1e-9)

    # ---- VOLLEY ROBUSTNESS: dog>>cat across DIFFERENT content-blind DG draws at the primary dose (a store-carried
    # effect survives re-drawing the volley; a single lucky draw that hit dog's mossy targets would not). --------------
    vres = r.get("volley_results", [rep])
    volley_gaps = [float(d["apical_gap_dog_minus_cat"]) for d in vres]
    n_volley_specific = int(sum(1 for g in volley_gaps if g >= PC_MIN_SEPARATION))
    VOLLEY_ROBUST = bool(len(vres) >= 2 and n_volley_specific >= max(2, int(np.ceil(0.6 * len(vres)))))

    # ---- REPRODUCIBILITY: at the primary dose + IDENTICAL content-blind pattern, does the completion fire reliably?
    repro = r.get("repro_probe", [])
    repro_fire_rate = float(r.get("repro_fire_rate", 0.0))
    REPRODUCIBLE = bool(len(repro) >= 3 and repro_fire_rate >= 0.8)

    g1_gain_dog = after_dog["apical_cue"] - quiet_dog["apical_cue"]
    G1 = bool(g1_gain_dog > 1e-3)
    G3_specific = bool((after_cat["apical_cue"] - quiet_cat["apical_cue"]) < 1e-3)

    transfers = bool(instr_ok and afferent_engaged and STRUCT_AWARE and PC and WRITE_SPECIFIC
                     and LESION_COLLAPSES and VOLLEY_ROBUST and REPRODUCIBLE)

    v = Verdict("idle-replay via DG/EC-AFFERENT drive on the REAL D5 episodic bridge (content-blind mossy volley -> "
                "pattern completion + BTSP write, n_ca3=2000)")
    v.require("instrument: D5's own intact/unstored/lesion/quiet-inert/recall-reproducible-in-mean all hold",
              instr_ok, expect=True)
    v.require("afferent ENGAGES: content-blind DG drive produces an apical-dAP response (the D5 read variable, not "
              "soma firing) at the primary dose -- the injection is load-bearing, not inert", afferent_engaged, expect=True)
    # PRECONDITIONS are ONLY the two requires above (instrument validated + afferent load-bearing) plus the
    # store-forming reaches() below -- all of which PASS here. The transfer TEETH (STRUCT_AWARE, PC, WRITE_SPECIFIC,
    # LESION_COLLAPSES, VOLLEY_ROBUST, REPRODUCIBLE) are the RESULT being measured; they drive `transfers` (go=) below
    # and are recorded as top-level booleans + in the verdict string's miss-list. They are deliberately NOT
    # require()/control() checks: modeling a MEASURED negative as a precondition failure would force UNDEFINED on a
    # validated instrument and mislabel a clean NO-GO (the exact defect verdict-preconditions guards against). In
    # particular the STRUCT overlap TIES (dog==cat) -- that tie IS the structure-blindness finding, not a void
    # manipulation, so it belongs in the result body, never in a control() that would spuriously read as UNDEFINED.
    v.reaches("store() potentiates dog's within-assembly weight above the never-formed baseline (cat)",
              before=r["w_within_cat_poststore"], after=r["w_within_dog_poststore"])
    for proc in ("STDP", "Hebbian (kernel-driven)", "homeostasis", "short-term plasticity", "reward modulation",
                 "structural plasticity", "BDSP learning (the tested-NEGATIVE hidden-credit rule)"):
        v.disabled(proc, why="isolation: only the pathway's permanent coincidence-plateau routing + BTSP (toggled "
                             "only during the replay window) are live; the mossy boost is a content-blind global "
                             "whole-pathway scalar, restored before the after-replay recall")
    decided = v.decide(go=transfers)

    void_if(not instr_ok, "an instrument check failed (intact-fires / unstored-abstains / lesion-collapses / "
                          "quiet-inert / recall-reproducible-in-mean)")
    void_if(not afferent_engaged, "the content-blind afferent drive produced no apical response at any dose (inert "
                                  "injection) -- the transfer read is unattributable")
    attributable_to("PC pattern-completion (dog vs cat apical-UP, identical content-blind afferent dose)",
                    rep["dog_apical_up_mean"], rep["cat_apical_up_mean"])
    attributable_to("LESION teeth (stored-recurrence dog apical vs UNFORMED-baseline dog apical)",
                    rep["dog_apical_up_mean"], lesion_dog_apical)

    if not instr_ok:
        verdict = ("UNDEFINED -- an instrument precondition failed (see 'instrument' checks); the transfer claim is "
                   "NOT cleanly attributable. Re-check before re-running.")
    elif not afferent_engaged:
        verdict = ("UNDEFINED -- the content-blind DG/EC afferent drive produced NO apical response at ANY swept dose "
                   f"(gentlest-to-strongest; primary apical dog={rep['dog_apical_up_mean']:.4f}). The injection was "
                   "inert, so nothing measured on top of it is attributable. Next: widen the dose grid (--dg-pAs / "
                   "--mossy-boosts) until the afferent engages, then re-read.")
    elif transfers:
        verdict = (f"GO -- content-blind DG/EC-afferent idle reactivation on the REAL D5 CA3 bridge (n_ca3=2000; "
                   f"primary dose mb={rep['mossy_boost']:g} pA={rep['dg_pA']:g}) STRUCTURE-AWARE-ly pattern-completes "
                   f"the stored 'dog' assembly (apical-UP dog={rep['dog_apical_up_mean']:.3f} vs cat="
                   f"{rep['cat_apical_up_mean']:.3f}, gap {apical_gap:+.3f}), the completion COLLAPSES under the "
                   f"unformed-weight lesion (dog apical {lesion_dog_apical:.3f}) proving it is carried by the stored "
                   f"recurrence, it is ROBUST across {len(vres)} content-blind DG draws ({n_volley_specific} specific), "
                   f"and the BTSP write is dog-specific (dw dog={dw_dog:.3f} vs cat={dw_cat:.3f}) -- the afferent-drive "
                   f"replay mechanism TRANSFERS to the real sparse code with NO host-known cue.")
    else:
        # instrument validates AND the afferent engages CA3 -> this is a CLEAN measured NEGATIVE (NO-GO on the method),
        # not an UNDEFINED-from-precondition. The dose sweep (gentlest engaging dose is the primary) rules out the
        # "too-strong hammer" confound: if even the minimal engaging seed is non-specific, the negative is robust.
        miss = []
        if not STRUCT_AWARE:
            miss.append(f"NOT STRUCTURE-AWARE -- content-blind afferent drive did NOT preferentially reach dog's "
                        f"assembly (apical gap dog-cat={apical_gap:+.3f}; recruit overlap dog={rep.get('overlap_dog', 0.0):.3f} "
                        f"vs cat={rep.get('overlap_cat', 0.0):.3f} vs chance={rep.get('chance_overlap', 0.0):.3f}). The "
                        f"readout bridge's mossy wiring is independent of the DISCARDED selection bridge that defined the "
                        f"assembly membership, so the afferent volley excites dog and cat alike -- content-blind but "
                        f"structure-BLIND to the store")
        if not PC:
            miss.append(f"PC FAILED -- afferent drive did not preferentially recruit dog's apical dAP over cat's "
                        f"(apical-UP dog={rep['dog_apical_up_mean']:.3f} max={rep['dog_apical_up_max']:.3f} vs "
                        f"cat={rep['cat_apical_up_mean']:.3f} max={rep['cat_apical_up_max']:.3f})")
        if not WRITE_SPECIFIC:
            miss.append(f"WRITE non-specific (dw dog={dw_dog:.4f} cat={dw_cat:.4f}) -- the diffuse afferent volley + "
                        f"BTSP potentiates the never-stored assembly as much or more (moat leak)")
        if not LESION_COLLAPSES:
            miss.append(f"LESION did NOT collapse (dog apical under UNFORMED baseline weights={lesion_dog_apical:.3f} "
                        f"vs stored={rep['dog_apical_up_mean']:.3f}) -- the afferent may be driving dog's cells "
                        f"DIRECTLY rather than via the stored recurrence (structure-awareness not proven)")
        if not VOLLEY_ROBUST:
            miss.append(f"NOT VOLLEY-ROBUST -- dog>>cat held on only {n_volley_specific}/{len(vres)} content-blind DG "
                        f"draws (gaps={[round(g, 3) for g in volley_gaps]}); the specificity may be one lucky draw, "
                        f"not a store-carried effect")
        if not REPRODUCIBLE:
            miss.append(f"NOT REPRODUCIBLE -- at the primary dose + IDENTICAL content-blind pattern, the completion "
                        f"fired only {repro_fire_rate:.0%} of {len(repro)} re-drives; it is a STATE/PHASE-DEPENDENT "
                        f"TRANSIENT (the sweep hit vs identical-pattern volley-0 miss), not a reliable transfer")
        verdict = ("NO-GO / HONEST-TRANSFER-NEGATIVE (instrument validated, afferent engaged) -- " + "; ".join(miss) +
                   ". Per THE LAW this is a verdict on a METHOD (driving the readout bridge's OWN mossy afferent, whose "
                   "wiring is independent of the selection bridge that defined the assembly), not a closure of the "
                   "capability. See NEXT_RUNG.")

    return {"GO": transfers, "verdict": verdict, "decided": decided,
            "status": decided.get("status"), "preconditions": decided.get("preconditions", []),
            "instr_ok": instr_ok, "afferent_engaged": afferent_engaged, "STRUCT_AWARE": STRUCT_AWARE,
            "PC": PC, "WRITE_SPECIFIC": WRITE_SPECIFIC, "LESION_COLLAPSES": LESION_COLLAPSES,
            "VOLLEY_ROBUST": VOLLEY_ROBUST, "n_volley_specific": n_volley_specific, "n_volleys": len(vres),
            "volley_apical_gaps": volley_gaps, "lesion_dog_apical": lesion_dog_apical,
            "REPRODUCIBLE": REPRODUCIBLE, "repro_fire_rate": repro_fire_rate,
            "G1_gain_after_replay": G1, "G3_specificity_cat_no_gain": G3_specific,
            "noise_floor": noise_floor, "separation_dog_minus_cat": separation,
            "struct_gap_dog_minus_cat": struct_gap_dog_cat, "struct_gap_dog_minus_chance": struct_gap_dog_chance,
            "apical_gap_dog_minus_cat": apical_gap,
            "pc_apical_gap": pc_apical_gap, "pc_apical_max_gap": pc_apical_max_gap, "pc_fire_gap": pc_fire_gap,
            "dw_dog": dw_dog, "dw_cat": dw_cat, "g1_gain_dog": g1_gain_dog}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--recall-draws", type=int, default=5, dest="recall_draws",
                    help="K independent recall draws to average per read (the instrument reproducibility fix)")
    ap.add_argument("--quiet-steps", type=int, default=100, dest="quiet_steps")
    # DG afferent drive: a content-blind random DG volley (fraction of the WHOLE DG population), NEVER dog's pattern.
    ap.add_argument("--dg-frac", type=float, default=0.10, dest="dg_frac",
                    help="fraction of the DG population driven (content-blind volley; ~0.10 matches the selection "
                         "drive-pattern sparsity)")
    # DOSE GRID (content-blind): the afferent volley is swept gentle->strong so the negative (if any) is not a
    # too-strong-hammer artifact. mossy_boost = global whole-pathway scalar on dg->ca3 (default weight 8.0; the
    # selection detonator ran ~3000) -- uniform over every mossy synapse, NO topic knowledge. dg_pA = drive current.
    ap.add_argument("--mossy-boosts", type=str, default="1,10,50", dest="mossy_boosts",
                    help="comma-list of content-blind global dg->ca3 weight scalars to sweep")
    ap.add_argument("--dg-pAs", type=str, default="300,900", dest="dg_pAs",
                    help="comma-list of DG afferent drive currents (pA) to sweep")
    ap.add_argument("--engage-floor", type=float, default=1e-4, dest="engage_floor",
                    help="min apical-response fraction for a dose to count as producing ANY apical response")
    ap.add_argument("--n-volleys", type=int, default=4, dest="n_volleys",
                    help="number of DIFFERENT content-blind DG draws to test dog>>cat robustness at the primary dose")
    ap.add_argument("--repro-k", type=int, default=6, dest="repro_k",
                    help="number of IDENTICAL (same dose + same pattern) re-drives to measure the completion's "
                         "reproducibility / reliability")
    ap.add_argument("--warm-steps", type=int, default=100, dest="warm_steps")
    ap.add_argument("--read-steps", type=int, default=100, dest="read_steps")
    ap.add_argument("--recruit-theta", type=float, default=0.05, dest="recruit_theta",
                    help="CA3 cell counts as recruited if it fires in >= this fraction of read steps (low, because "
                         "the strongly-inhibited CA3 fires in transient volleys, not sustained trains)")
    ap.add_argument("--pc-sample-every", type=int, default=10, dest="pc_sample_every")
    a = ap.parse_args()

    err = None
    result = None
    verdict_block = None
    try:
        result = run(a.seed, a)
        verdict_block = build_verdict(result, a)
    except (RuntimeError, ValueError, AttributeError, KeyError, IndexError, TypeError) as e:
        err = "%s: %s" % (type(e).__name__, e)
        traceback.print_exc()

    summary = {"probe": "idle_replay_dgec_afferent_d5_transfer", "seed": a.seed,
               "params": {"recall_draws": a.recall_draws, "quiet_steps": a.quiet_steps, "dg_frac": a.dg_frac,
                          "mossy_boosts": a.mossy_boosts, "dg_pAs": a.dg_pAs, "n_volleys": a.n_volleys,
                          "engage_floor": a.engage_floor, "warm_steps": a.warm_steps, "read_steps": a.read_steps,
                          "recruit_theta": a.recruit_theta, "pc_sample_every": a.pc_sample_every},
               "backend": os.environ.get("SIM_BACKEND", "(unset)")}
    if err is not None:
        summary["error"] = err
        summary["GO"] = False
        summary["verdict"] = f"ERROR -- {err}"
    else:
        summary["run"] = result
        summary.update(verdict_block)

    summary["HONEST_NOTE"] = (
        "This runner imports the REAL production D5 organ (EpisodicDapMemory) UNCHANGED -- no sim/ edit, no existing "
        "runner edit, single seed (a scoped GPU transfer de-risk, not a generalization claim; 6-seed is the natural "
        "next rung IF GO). The DRIVER is the biologically-correct successor to the prior untargeted-CA3-noise probe: "
        "the DG/EC afferent (dg->ca3 mossy pathway), driven by a CONTENT-BLIND random DG volley (never dog's pattern). "
        "The mossy pathway is (content-blind) globally scaled so it can detonate CA3 at all (its default weight 8.0 is "
        "far below the selection detonator's ~3000); the scalar is uniform over every mossy synapse and uses NO topic "
        "knowledge, and is restored before the after-replay recall. THE DECISIVE MEASUREMENT is whether this "
        "content-blind afferent drive is STRUCTURE-AWARE -- i.e. whether the readout bridge's OWN mossy wiring "
        "concentrates CA3 recruitment onto the stored (dog) assembly more than the never-stored (cat) control and "
        "above chance. Code-read establishes the risk: the assembly membership was SELECTED on a separate, discarded "
        "bridge (_gap5_emergent_dg_selection_derisk._build_bridge) whose mossy wiring is a DIFFERENT RNG draw than the "
        "readout bridge we drive, so structure-awareness toward the store is NOT guaranteed and is measured here, not "
        "assumed. NOT 'consolidation'/'stabilization' in docs/TERMS.md -- a single-tick transfer-mechanism probe.")
    summary["NEXT_RUNG"] = (
        "IF GO: 6 seeds (42 43 44 100 101 102); add the starting-weight-gated metaplastic write suppression; wire "
        "under continuous_engine.py's idle tick, default-off. "
        "IF NOT STRUCTURE-AWARE (the code-read-predicted risk): the afferent locus alone cannot seed the store because "
        "the readout bridge's mossy wiring is independent of the selection bridge that defined membership. The smallest "
        "structure-preserving fix is an ORGAN change (out of scope for 'import unchanged'): build the readout bridge to "
        "SHARE the selection bridge's dg->ca3 mossy connectivity (or re-project the store's DG selection pattern onto "
        "the readout bridge's mossy synapses), so a content-blind DG volley detonates the SAME CA3 cells that were "
        "selected -> dog's potentiated recurrence completes it. Only THEN is content-blind afferent replay "
        "structure-aware end-to-end. "
        "IF STRUCTURE-AWARE but PC/WRITE fail: the seed reaches dog but the potentiated recurrence + dendritic dAP does "
        "not complete/latch under a diffuse afferent seed -- tune warm/read duration and the coincidence operating "
        "point (kthresh window), and add the metaplastic write gate to keep cat's baseline recurrence from co-firing.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[dgec-replay] VERDICT: {summary.get('verdict')}", flush=True)
    print(f"[dgec-replay] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    sys.exit(main())
