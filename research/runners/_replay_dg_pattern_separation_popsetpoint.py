"""Stabilize the DG k-WTA with a POPULATION-activity set-point (board #78).

Attacks the SAME residual as board #43/#71 and #73, now mapped by TWO negatives:

  * #71 bridge (`2026-08-19-replay-separator-bridge-...-kWTA-stability-residual.md`)
    FIXED the Izhikevich rebound (shunting inhibition) + a Hebbian write-runaway
    (transmission-gated write); single-memory recall is at ceiling. RESIDUAL: the
    k-WTA does not hold a SPARSE code for BOTH similar memories -- one engram
    collapses near-dense (150-200 of 200 granules), SUBSUMES the other, and the
    dense memory's answer wins BOTH probes (anti-symmetric signature; both_win 0/6).
  * #73 homeostat (`2026-08-19-kwta-stability-homeostat-NOGO.md`) built a per-CELL
    firing-RATE set-point (adaptive threshold). It FAILED and made it WORSE: a
    per-cell rate set-point is ANTI-SPARSE (its fixed point is "every drivable cell
    fires at target" = the densest code); it RECRUITS silenced cells, defeats the
    basket. Wrong locus (per-cell) + wrong sign (recruits, not selects).

THE NAMED, UNTRIED MECHANISM (this file): a POPULATION-level set-point. A fast
controller holds TOTAL dentate granule activity near a target sparsity k by adapting
the INHIBITORY pool's GAIN -- NOT per-cell thresholds. When too many granules fire,
total-activity feedback RAISES the basket gain so the divisive (shunting) competition
SHARPENS and only the top-k best-driven granules survive; when too few fire, it
relaxes. This is the OPPOSITE of the per-cell homeostat: it SELECTS at the population
level (the drive gradient decides WHICH granules win; the set-point decides HOW MANY).

Biology: divisive normalization / population gain control -- Carandini & Heeger (2012)
Nat Rev Neurosci 13:51-62 (normalization as a canonical neural computation: response /
(sigma + gain * pool)); the DG feedback PV-basket (dg_fs, all-to-all dg->fs->dg with
shunting reversal ~ECl) IS this normalizer, and its ~1-2% sparsity is a k-of-N
competition held by a set-point on TOTAL activity (Marr 1971; O'Reilly-McClelland 1994;
Leutgeb 2007; Bakker 2008), not a per-cell rate (Turrigiano 2011, the wrong tool #73
banked).

IMPLEMENTATION -- NO sim/ edit; reuse the #71 runner by import (its shunting reversal +
transmission-gated write + all measurement/probe/scramble/direct-readout machinery).
The population set-point is a fast proportional-integral controller on the DG active
COUNT vs a target k. Its output is an adaptive DEPOLARIZING current injected into the
dg_fs BASKET pool: too many granules fire -> more basket drive -> more basket spikes ->
more divisive (fixed-weight, shunting) inhibition onto granules -> the marginal ones
fall below threshold while the top-k survive. We inject to the basket INPUT (not scale
its OUTPUT gate), so the base runner's competition LESION (separator_off: fs->dg
transmission gate = 0) automatically neutralizes the controller -> the null stays a
true null with no per-condition bookkeeping. The controller is wrapped around
``bridge._run_one_simulation_step`` so EVERY base function (engram / consolidate /
probe / direct-readout / scramble) gets it, exactly like #73 swapped ``build_bridge``.

LESION arm (dissociation): ``pop_setpoint=False`` -> the wrapper is a no-op -> the
circuit is byte-identical to the #71 fixed-gain basket -> dense-collapse + both_win 0/6
RETURN (reproduces the #71/#73 baseline). Deterministic (cfg.seed).

ANTI-CHEATS (design first, from the bar the two negatives defined):
  1. Two SIMILAR memories BOTH discriminable -- both_win 6/6. Per-memory selectivity
     for BOTH reported.
  2. Total DG activity bounded near k across BOTH memories -- engram sizes per memory
     per seed stay near the set-point (not ~200). No dense-collapse under ON.
  3. The population set-point is LOAD-BEARING: LESION -> dense-collapse + the
     asymmetric failure RETURN (reproduce #71/#73). Dissociation.
  4. No regression: single-memory recall stays +1.00; DISSIMILAR pairs both-win;
     scramble-teach inverts. 6 seeds (42/43/44/100/101/102), deterministic (cfg.seed).

Run:
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_popsetpoint \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/kwta_popsetpoint/popsetpoint_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._replay_dg_pattern_separation_bridge as base  # noqa: E402
from research.runners._replay_dg_pattern_separation_bridge import (  # noqa: E402
    BridgeConfig,
    SEEDS,
    DG_WRITE_GATE,
    DG_COMPETITION_GATE,
    ANSWER_INHIBITION_GATE,
    DG_ANSWER_TX_GATE,
)
from research.runners._replay_dg_pattern_separation_gate import (  # noqa: E402
    _perforant_edges,
)
from research.runners._replay_cortical_consolidation_gate import (  # noqa: E402
    _all_to_all,
)
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


@dataclass(frozen=True)
class PConfig(BridgeConfig):
    """#71 BridgeConfig + the POPULATION-activity set-point knobs (additive)."""
    # Master switch. False -> the wrapper is a no-op -> build is byte-identical to the
    # #71 runner (the LESION).
    pop_setpoint: bool = True
    # Target number of active granules (the population set-point). n_dg=200; the working
    # LIF regime + answer_assembly=16 put the useful engram near ~20-30 (10-15% active),
    # well under the dense_engram_frac=0.60 (120-cell) failure line.
    pop_k_target: float = 18.0
    # EMA smoothing of the instantaneous DG active COUNT (tau ~ 1/alpha steps). Smooths
    # the feedback so the controller does not chatter with the 2 ms basket delay.
    pop_ema_alpha: float = 0.4
    # Proportional gain: pA of extra basket drive per granule of EXCESS activity.
    pop_kp: float = 45.0
    # Integral gain + clamp: removes steady-state offset (fixed-weight basket has finite
    # authority) without windup.
    pop_ki: float = 6.0
    pop_integ_max: float = 800.0
    # Clamp on the injected basket drive (pA). The basket is IZH2007_FS; a few hundred
    # to ~2000 pA spans quiescent->fast-spiking.
    pop_drive_max: float = 2500.0


def _install_pop_controller(bridge, cfg: PConfig, regions):
    """Wrap ``bridge._run_one_simulation_step`` with the population-activity set-point.

    The controlled quantity is CUMULATIVE RECRUITMENT: the number of DISTINCT granules
    that have fired since the event began (this is what the dense-collapse instrument
    measures -- an engram is the set of cells that fire at least once during the event).
    Diagnosed by the arc's probes: instantaneous DG activity is already sparse (~15/step)
    at a reasonable operating point, yet the winner set ROTATES across the 37-step window
    so the CUMULATIVE engram runs dense. A set-point on instantaneous total activity thus
    never engages (activity is below k every step); the load-bearing population variable
    is cumulative recruitment. The controller ramps a depolarizing drive into the dg_fs
    basket (raising the divisive shunting inhibition) as cumulative recruitment approaches
    and exceeds k, blocking LATE recruits so the early, strongest-driven winners lock in.
    Per-event reset on the silent settle gap (>=2 consecutive silent steps). A no-op when
    ``cfg.pop_setpoint`` is False (the LESION -> #71 baseline)."""
    from sim.backend import to_host

    dg_idx = regions["dg"]
    fs_idx = regions["dg_fs"]
    n_dg = int(dg_idx.size)
    orig_step = bridge._run_one_simulation_step

    # controller state; ``ever`` = cumulative distinct-fired mask for the current event
    state = {"ever": np.zeros(n_dg, dtype=bool), "integ": 0.0, "drive": 0.0, "silent": 0}
    bridge._pop_state = state
    bridge._pop_enabled = bool(cfg.pop_setpoint)

    k = float(cfg.pop_k_target)
    kp = float(cfg.pop_kp)
    ki = float(cfg.pop_ki)
    integ_max = float(cfg.pop_integ_max)
    drive_max = float(cfg.pop_drive_max)

    if not cfg.pop_setpoint:
        return  # LESION: leave the original step untouched (byte-identical to #71)

    def wrapped_step():
        # (1) apply the drive computed from the PREVIOUS step to the basket INPUT.
        #     Base already set dg_fs external current to 0 this step; we ADD.
        d = state["drive"]
        if d > 0.0:
            bridge.cp_external_input_current[fs_idx] += np.float32(d)
        # (2) run the real substrate step
        orig_step()
        # (3) read DG firing, update cumulative recruitment + PI controller
        fired = np.asarray(to_host(bridge.cp_firing_states[dg_idx])).astype(bool)
        n_active = int(fired.sum())
        if n_active == 0:
            state["silent"] += 1
            if state["silent"] >= 2:          # event boundary -> reset for the next event
                state["ever"][:] = False
                state["integ"] = 0.0
                state["drive"] = 0.0
            return
        state["silent"] = 0
        state["ever"] |= fired
        nfired = int(state["ever"].sum())     # cumulative distinct granules this event
        err = nfired - k
        integ = state["integ"] + max(0.0, err)
        if integ > integ_max:
            integ = integ_max
        drive = kp * max(0.0, err) + ki * integ
        if drive < 0.0:
            drive = 0.0
        elif drive > drive_max:
            drive = drive_max
        state["integ"] = integ
        state["drive"] = drive

    bridge._run_one_simulation_step = wrapped_step


def build_bridge_popsetpoint(seed: int, cfg: PConfig):
    """Copy of ``base.build_bridge`` (keeps the shunting reversal + transmission-gated
    write) plus the population-activity set-point controller wrapped around the step.

    When ``cfg.pop_setpoint`` is False the controller is a no-op and this is
    byte-identical to ``base.build_bridge`` (the #71 residual). That False path IS the
    lesion control."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel, NeuronType
    from sim.regions import BrainRegion, RegionPathway

    exc = dict(exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
               weight_jitter=0.0, plastic_internal=False,
               izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    inh = dict(exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
               weight_jitter=0.0, plastic_internal=False,
               izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name)

    dg_kwargs = dict(exc, syn_reversal_potential_i_override=float(cfg.dg_inh_reversal_mV))
    ans_kwargs = dict(exc, syn_reversal_potential_i_override=float(cfg.answer_inh_reversal_mV))

    core = CoreSimConfig()
    core.enable_brain_region_framework = True
    core.brain_regions = [
        BrainRegion("input", cfg.n_input, **exc),
        BrainRegion("dg", cfg.n_dg, **dg_kwargs),
        BrainRegion("dg_fs", cfg.n_dg_fs, **inh),
        BrainRegion("answer", cfg.n_answer, **ans_kwargs),
        BrainRegion("answer_fs", cfg.n_answer_fs, **inh),
    ]
    core.region_pathways = [RegionPathway("input", "dg", density=0.01, weight_mean=0.01, plastic=False)]
    core.num_neurons = 0
    core.connections_per_neuron = 0
    core.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core.neural_profile_name = "GENERIC_UNSTRUCTURED"
    core.dt_ms = 1.0
    core.seed = core.heterogeneity_seed = core.ou_seed = int(seed)
    core.enable_stdp = False
    core.enable_hebbian_learning = True
    core.hebbian_rate_window = True
    core.hebbian_learning_rate = float(cfg.wake_learning_rate)
    core.hebbian_max_weight = float(cfg.hebbian_max_weight)
    core.hebbian_min_weight = 0.0
    core.hebbian_weight_decay = 0.0
    core.hebbian_coactivity_decay = float(cfg.hebbian_coactivity_decay)
    core.hebbian_coactivity_thresh = float(cfg.hebbian_coactivity_thresh)
    core.enable_reward_modulation = False
    core.enable_homeostasis = False
    core.enable_short_term_plasticity = False
    core.enable_structural_plasticity = False
    core.enable_ou_process = False
    core.ou_std_current_pA = 0.0
    core.fast_spike_reset = True
    core.propagation_strength = float(cfg.propagation_strength)
    core.max_synaptic_delay_ms = float(cfg.max_synaptic_delay_ms)

    runtime = RuntimeState()
    runtime.actual_seed_used = int(seed)
    bridge = SimulationBridge(core_config=core, viz_config=VisualizationConfig(),
                              runtime_state=runtime, gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(core.max_synaptic_delay_ms / core.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    regions = {name: np.asarray(bridge.region_manager.indices(name), dtype=np.int64)
               for name in ("input", "dg", "dg_fs", "answer", "answer_fs")}

    pp_pre, pp_post = _perforant_edges(seed, cfg, regions["input"], regions["dg"])
    i2fs_pre, i2fs_post = _all_to_all(regions["input"], regions["dg_fs"])
    d2fs_pre, d2fs_post = _all_to_all(regions["dg"], regions["dg_fs"])
    fs2d_pre, fs2d_post = _all_to_all(regions["dg_fs"], regions["dg"])
    a2fs_pre, a2fs_post = _all_to_all(regions["answer"], regions["answer_fs"])
    fs2a_pre, fs2a_post = _all_to_all(regions["answer_fs"], regions["answer"])
    dga_pre, dga_post = _all_to_all(regions["dg"], regions["answer"])

    def group(pre, post, weight, *, plastic, conn_type, plasticity_gate=None, transmission_gate=None):
        row = {"pre_indices": pre.tolist(), "post_indices": post.tolist(),
               "initial_weights": np.full(pre.size, weight, dtype=np.float32),
               "plastic": bool(plastic), "conn_type": conn_type, "count": int(pre.size)}
        if plasticity_gate:
            row["plasticity_gate"] = plasticity_gate
        if transmission_gate:
            row["transmission_gate"] = transmission_gate
        return row

    wiring = {
        "input_to_dg": group(pp_pre, pp_post, cfg.input_to_dg_weight, plastic=False, conn_type="E_TO_E"),
        "input_to_fs": group(i2fs_pre, i2fs_post, cfg.input_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "dg_to_fs": group(d2fs_pre, d2fs_post, cfg.dg_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "fs_to_dg": group(fs2d_pre, fs2d_post, cfg.fs_to_dg_weight, plastic=False,
                          conn_type="I_TO_E", transmission_gate=DG_COMPETITION_GATE),
        "answer_to_fs": group(a2fs_pre, a2fs_post, cfg.answer_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "fs_to_answer": group(fs2a_pre, fs2a_post, cfg.fs_to_answer_weight, plastic=False,
                              conn_type="I_TO_E", transmission_gate=ANSWER_INHIBITION_GATE),
        "dg_to_answer": group(dga_pre, dga_post, cfg.dg_answer_init_weight, plastic=True,
                              conn_type="E_TO_E", plasticity_gate=DG_WRITE_GATE,
                              transmission_gate=DG_ANSWER_TX_GATE),
    }
    inh_idx = np.concatenate([regions["dg_fs"], regions["answer_fs"]]).tolist()
    bridge.inject_explicit_wiring(wiring, output_inhibitory_indices=inh_idx)
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)

    # THE MECHANISM: install the population-activity set-point controller (no-op when off)
    _install_pop_controller(bridge, cfg, regions)

    handles = {"regions": regions, "bridge_identity": id(bridge),
               "pop_setpoint": bool(cfg.pop_setpoint),
               "wiring_counts": {k: v["count"] for k, v in wiring.items()}}
    return bridge, handles


def _run_arm(seeds, cfg):
    """Run the full #71 pipeline (via base.run) with build_bridge swapped to the
    pop-setpoint build. cfg.pop_setpoint selects ON vs LESION."""
    base.build_bridge = build_bridge_popsetpoint
    try:
        return base.run(seeds, cfg)
    finally:
        base.build_bridge = _ORIG_BUILD


_ORIG_BUILD = base.build_bridge


def _seed_row(on_seed, off_seed):
    """Assemble the dissociation row for one seed from the ON and LESION payloads."""
    on = on_seed["conditions"]["similar_separator_on"]
    off = off_seed["conditions"]["similar_separator_on"]
    return {
        "seed": on_seed["seed"],
        # anti-cheat 1: both similar memories discriminable
        "both_win_on": on["both_win"],
        "both_win_lesion": off["both_win"],
        "per_memory_selectivity_on": on["per_memory_selectivity"],
        "per_memory_selectivity_lesion": off["per_memory_selectivity"],
        "mean_selectivity_on": on["mean_selectivity"],
        "mean_selectivity_lesion": off["mean_selectivity"],
        # anti-cheat 2: engram sizes bounded near k (ON) vs dense-collapse (LESION)
        "dg_sizes_on": (on["dg_separation"]["dg_size_m0"], on["dg_separation"]["dg_size_m1"]),
        "dg_sizes_lesion": (off["dg_separation"]["dg_size_m0"], off["dg_separation"]["dg_size_m1"]),
        "dense_collapse_on": on["dense_engram_collapse"],
        "dense_collapse_lesion": off["dense_engram_collapse"],
        "dg_jaccard_on": on["dg_separation"]["dg_jaccard"],
        "dg_jaccard_lesion": off["dg_separation"]["dg_jaccard"],
        # anti-cheat 3: no regression (from the ON arm)
        "single_selectivity_on": on_seed["summary"]["single_selectivity"],
        "single_scramble_on": on_seed["summary"]["single_scramble_selectivity"],
        "dissimilar_both_win_on": on_seed["summary"]["dissimilar_both_win"],
    }


def relocalization_sweep(seeds, cfg_base):
    """RE-LOCALIZATION evidence (this arc's decisive finding): both_win is independent
    of DG engram density/size/separation. Sweep the perforant DRIVE (input_to_dg_weight)
    across a feed-forward-inhibition regime (dg_to_fs low, shunting reversal above vr)
    that yields NON-dense, SYMMETRIC engrams whose size scales cleanly with drive. If
    both_win were a DG-sparsity problem, some drive level would pass. It does not: the
    anti-symmetric readout signature persists at every size. Set-point OFF (this probes
    the DG code the read receives, not the controller)."""
    from dataclasses import replace as _replace
    rows = []
    for w in (40.0, 50.0, 60.0, 70.0):
        cfg = _replace(cfg_base, pop_setpoint=False, dg_inh_reversal_mV=-58.0,
                       dg_to_fs_weight=5.0, input_to_dg_weight=float(w))
        for s in seeds:
            on = base.run_condition(int(s), "similar_separator_on", cfg)
            ds = on["dg_separation"]
            rows.append({
                "input_to_dg_weight": w, "seed": int(s),
                "both_win": bool(on["both_win"]),
                "dg_sizes": (ds["dg_size_m0"], ds["dg_size_m1"]),
                "dg_jaccard": ds["dg_jaccard"],
                "dense_collapse": bool(on["dense_engram_collapse"]),
                "per_memory_selectivity": on["per_memory_selectivity"],
            })
    both_win_any = int(sum(1 for r in rows if r["both_win"]))
    return {
        "note": "both_win vs perforant drive (DG density); set-point OFF, FF-inhibition regime",
        "both_win_count_over_all_cells": both_win_any,
        "n_cells": len(rows),
        "rows": rows,
    }


def run(seeds, cfg_on, relocalize=True):
    started = time.time()
    cfg_lesion = PConfig(**{**cfg_on.__dict__, "pop_setpoint": False})

    on_payload = _run_arm(seeds, cfg_on)
    lesion_payload = _run_arm(seeds, cfg_lesion)

    by_seed_on = {r["seed"]: r for r in on_payload["per_seed"]}
    by_seed_off = {r["seed"]: r for r in lesion_payload["per_seed"]}
    rows = [_seed_row(by_seed_on[int(s)], by_seed_off[int(s)]) for s in seeds]
    n = len(rows)

    def cnt(fn):
        return int(sum(1 for r in rows if fn(r)))

    both_win_on = cnt(lambda r: r["both_win_on"])
    both_win_lesion = cnt(lambda r: r["both_win_lesion"])
    dense_on = cnt(lambda r: r["dense_collapse_on"])
    dense_lesion = cnt(lambda r: r["dense_collapse_lesion"])
    single_ok = cnt(lambda r: r["single_selectivity_on"] >= 0.30)
    scramble_ok = cnt(lambda r: r["single_scramble_on"] <= r["single_selectivity_on"] - 0.30)
    dissim_ok = cnt(lambda r: r["dissimilar_both_win_on"])
    # engram sizes bounded near k (both memories under the dense line) on ON
    max_size_line = float(cfg_on.dense_engram_frac) * float(cfg_on.n_dg)
    sizes_bounded = cnt(lambda r: max(r["dg_sizes_on"]) < max_size_line)

    checks = {
        # AC1: both similar memories discriminable, all 6 seeds (the bar the two negatives set).
        "both_similar_discriminable_6of6": both_win_on == n,
        # AC2: pop set-point holds sparsity -- no dense-collapse, sizes bounded, under ON.
        "no_dense_collapse_on": dense_on == 0,
        "sizes_bounded_near_k": sizes_bounded == n,
        # AC3 dissociation: LESION reproduces the #71/#73 failure (dense-collapse returns
        # AND both_win fails).
        "lesion_reproduces_residual": (dense_lesion >= 1) and (both_win_lesion < n),
        # AC4: no regression.
        "single_recall_ceiling": single_ok == n,
        "scramble_inverts": scramble_ok == n,
        "dissimilar_both_win": dissim_ok == n,
    }
    go = (checks["both_similar_discriminable_6of6"]
          and checks["no_dense_collapse_on"]
          and checks["sizes_bounded_near_k"]
          and checks["lesion_reproduces_residual"]
          and checks["single_recall_ceiling"]
          and checks["scramble_inverts"]
          and checks["dissimilar_both_win"])

    # ATTRIBUTION: does the set-point REDUCE dense-collapse? (ON count minus LESION
    # count; a NEGATIVE fraction is the hoped-for GO -- the set-point removed collapses
    # the lesion has.)
    dense_attrib = attributable_to("dense-collapse ON vs LESION", dense_on, dense_lesion)

    # VERDICT: the preconditions travel with the artifact so the gate can enforce that
    # what earned the verdict is present.
    v = Verdict("DG k-WTA stability via a POPULATION-activity set-point (adaptive basket gain)")
    v.require("pipeline intact: single-memory recall at ceiling (6/6)", single_ok == n, expect=True)
    v.require("read rides the learned mapping: scramble inverts (6/6)", scramble_ok == n, expect=True)
    v.require("baseline is the #71/#73 residual: lesion dense-collapses", dense_lesion >= 1, expect=True)
    v.require("baseline is the #71/#73 residual: lesion both_win fails", both_win_lesion < n, expect=True)
    v.control("set-point changes the DG code (manipulation landed): ON vs LESION dense-collapse",
              treatment=float(n - dense_on), control=float(n - dense_lesion), min_separation=0.0)
    v.disabled("STDP / reward / short-term & structural plasticity / OU noise / per-cell homeostat",
               why="isolation inherited from the #71 runner; the separator + Hebbian write are the only live plasticity")
    decided = v.decide(go=go)

    relocalization = relocalization_sweep(seeds, cfg_on) if relocalize else None

    return {
        "gate": "replay_dg_pattern_separation_popsetpoint",
        "mechanism": "population-activity set-point: PI controller on total DG active count -> adaptive divisive basket (dg_fs) gain",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        "status": decided["status"],
        "verdict": decided,
        "preconditions": decided["preconditions"],
        "dense_collapse_attributable_to_setpoint": dense_attrib,
        "relocalization": relocalization,
        "checks": checks,
        "popsetpoint_config": {
            "k_target": cfg_on.pop_k_target,
            "ema_alpha": cfg_on.pop_ema_alpha,
            "kp": cfg_on.pop_kp,
            "ki": cfg_on.pop_ki,
            "integ_max": cfg_on.pop_integ_max,
            "drive_max": cfg_on.pop_drive_max,
            "dense_engram_frac": cfg_on.dense_engram_frac,
        },
        "pooled": {
            "both_win_on_count": both_win_on,
            "both_win_lesion_count": both_win_lesion,
            "dense_collapse_on_count": dense_on,
            "dense_collapse_lesion_count": dense_lesion,
            "sizes_bounded_near_k_count": sizes_bounded,
            "single_recall_ok_count": single_ok,
            "scramble_inverts_count": scramble_ok,
            "dissimilar_both_win_count": dissim_ok,
            "mean_selectivity_on": float(np.mean([r["mean_selectivity_on"] for r in rows])),
            "mean_selectivity_lesion": float(np.mean([r["mean_selectivity_lesion"] for r in rows])),
        },
        "per_seed": rows,
        "on_arm_full": on_payload,
        "lesion_arm_full": lesion_payload,
        "scaffolds": on_payload.get("scaffolds", []) + [
            "population set-point CONTROLLER (PI on total DG active count -> injected basket drive) is host; "
            "the SELECTION (which granules survive the divisive shunting basket) is on-substrate",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k-target", type=float, default=None)
    ap.add_argument("--ema-alpha", type=float, default=None)
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--ki", type=float, default=None)
    ap.add_argument("--integ-max", type=float, default=None)
    ap.add_argument("--drive-max", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.smoke:
        sm = base.smoke_config()
        overrides.update(sm.__dict__)
    for name, val in (("pop_k_target", args.k_target),
                      ("pop_ema_alpha", args.ema_alpha),
                      ("pop_kp", args.kp),
                      ("pop_ki", args.ki),
                      ("pop_integ_max", args.integ_max),
                      ("pop_drive_max", args.drive_max)):
        if val is not None:
            overrides[name] = val
    cfg = PConfig(**overrides)

    print(f"[kwta-popsetpoint] backend={os.environ.get('SIM_BACKEND','default')} "
          f"seeds={args.seeds} smoke={args.smoke} "
          f"k={cfg.pop_k_target} alpha={cfg.pop_ema_alpha} kp={cfg.pop_kp} ki={cfg.pop_ki} "
          f"integ_max={cfg.pop_integ_max} drive_max={cfg.pop_drive_max}", flush=True)
    payload = run(args.seeds, cfg)
    for r in payload["per_seed"]:
        pm_on = r["per_memory_selectivity_on"]
        pm_le = r["per_memory_selectivity_lesion"]
        print(f"  seed {r['seed']}: "
              f"ON both_win={r['both_win_on']} sizes={r['dg_sizes_on']} dense={r['dense_collapse_on']} "
              f"m0={pm_on['m0']:+.2f} m1={pm_on['m1']:+.2f} | "
              f"LESION both_win={r['both_win_lesion']} sizes={r['dg_sizes_lesion']} dense={r['dense_collapse_lesion']} "
              f"m0={pm_le['m0']:+.2f} m1={pm_le['m1']:+.2f} | "
              f"single={r['single_selectivity_on']:+.2f} scr={r['single_scramble_on']:+.2f} "
              f"dissim_bw={r['dissimilar_both_win_on']}", flush=True)
    print(f"  STATUS: {payload['status']}", flush=True)
    print(f"  checks: {json.dumps(payload['checks'])}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled'])}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
