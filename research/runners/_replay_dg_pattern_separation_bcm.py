"""Close the memory-separator readout residual with a BCM / SELECTIVITY-GATED dg->answer
WRITE (board #90). This is a WRITE-GATING mechanism, a DIFFERENT CLASS from the
per-granule OUTPUT-transform family that #73/#78/#readout banked as exhausted.

WHERE THE RESIDUAL SITS (established on main; verified before building, do NOT re-derive):
  * #78 pop-setpoint made the DG code symmetric+sparse, both_win STILL 0/6.
  * The competitive/heterosynaptic WRITE (`_replay_dg_pattern_separation_readout.py`,
    committed NO-GO 98cd33bb) RE-LOCALIZED the residual to the WEIGHTS: the collision is a
    near-perfect ANTI-SYMMETRY (m0-margin ~= -m1-margin) whose memory-PRIVATE granules are
    UNDER-WRITTEN (private_m1 weight == baseline on every seed). No per-granule OUTPUT
    transform can create the missing discriminative signal. Named next lever (this file):
    a selectivity/novelty-gated write (BCM-like) that AMPLIFIES private granules /
    SUPPRESSES shared granules.

WHY private_m1 is under-written (MEASURED here on the substrate, deepens #readout's
localization): the private_m1 granules fire in ZERO of the 14 m1 replay events during the
#78 interleaved consolidation. The #78 population set-point controller's cumulative-
recruitment integrator WINDS UP and SATURATES (ever ~= 53 >> k=18, injected basket
drive pinned at drive_max=2500) after the FIRST memory's engram is recruited, and never
resets across the interleaved schedule -- so every later event's marginal recruits are
blocked. The strongly + always-firing SHARED (and first-memory-private) granules dominate
the write; the discriminative granules never fire, hence never potentiate. In ISOLATION
(a single memory, fresh drain) the private granules fire FULLY -- the selectivity signal
is recoverable there. (This is the CLAUDE.md wall-reframe: the companion process #78 added
to hold sparsity is a clamp that dominates + suppresses exactly the granules whose absence
is the defect.)

THE MECHANISM (this file), stacked on the #78 population set-point (unchanged):
  1. Run the #78 base consolidation (the on-substrate coincidence write) -- byte-identical
     to the committed baseline. Shared granules get their (soon-suppressed) excess.
  2. Read per-granule per-memory ISOLATED reactivation firing on-substrate (drive each
     memory's INPUT alone, fresh drain -- private granules fire here). This recovers the
     memory-selectivity the interleaved write destroyed.
  3. BCM sliding-threshold selectivity gate (Bienenstock-Cooper-Munro 1982): a granule's
     threshold theta_g = <activity across memories>; a granule that fires ABOVE its own
     threshold for memory m (selective for m) is POTENTIATED toward a_m; a granule firing
     equally for all memories (shared, non-selective) is SUPPRESSED (heterosynaptic LTD).
     Applied as a runner-side transform on the plastic dg->answer weight vector (the same
     vector the #readout renorm used; the SELECTION + reactivation firing + the read stay
     on-substrate spiking; NO sim/ edit).

LOAD-BEARING QUANTITY, measured BEFORE both_win (the method the arc requires): does the
gate raise private_m1_to_correct ABOVE private_m1_to_wrong (write the discriminative signal
into the private-granule weights) and break the anti-symmetry (m0-margin + m1-margin off 0,
both positive)? The weight-space block reports LESION vs ON private_m1(corr,wrong) + margins
per seed. Only then is both_win a fair test.

LESION arm (dissociation): ``bcm_gate=False`` -> the transform is never applied ->
consolidation is the unmodified #78 write -> byte-identical to the committed NO-GO baseline
(both_win 0/6 RETURN). Deterministic (cfg.seed).

ANTI-CHEATS (design first):
  1. Two SIMILAR memories BOTH discriminable -- both_win 6/6 (the bar).
  2. The gate is LOAD-BEARING: LESION (bcm off) reproduces the residual (both_win < 6,
     anti-symmetric, private_m1 at baseline). Byte-identical dissociation.
  3. No regression: single-memory recall at ceiling; scramble-teach inverts; dissimilar
     both-win. 6 seeds (42/43/44/100/101/102), deterministic (cfg.seed).
  4. Mechanism moves the RIGHT quantity: private_m1_to_correct > private_m1_to_wrong ON vs
     == LESION, and both margins positive ON, measured BEFORE both_win is believed.

Run:
    OMP_NUM_THREADS=4 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_bcm \
        --seeds 42 43 44 100 101 102 --bcm-gate \
        --out research/findings/raw/sep_readout/bcm_6seed.json
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
    SEEDS,
    DG_WRITE_GATE,
    DG_COMPETITION_GATE,
    ANSWER_INHIBITION_GATE,
    DG_ANSWER_TX_GATE,
    _drain,
    _dg_engram,
)
from research.runners._replay_dg_pattern_separation_gate import (  # noqa: E402
    _answer_assemblies,
    _input_patterns,
    _jaccard,
)
from research.runners._replay_dg_pattern_separation_popsetpoint import (  # noqa: E402
    PConfig,
    build_bridge_popsetpoint,
)
from research.runners._replay_dg_pattern_separation_readout import (  # noqa: E402
    _weight_margins,
)
from research.runners._replay_cortical_consolidation_gate import _zero_current  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


@dataclass(frozen=True)
class BConfig(PConfig):
    """#78 PConfig (population set-point ON) + the BCM selectivity-WRITE knobs."""
    # Master switch. False -> the transform is never applied -> consolidation is the
    # unmodified #78 write -> byte-identical to the committed NO-GO baseline (the LESION).
    bcm_gate: bool = False
    # BCM sliding-threshold scale: theta_g = bcm_theta_scale * mean_m rate[g,m]. 1.0 = the
    # granule's own mean activity (classic BCM). >1 sharpens selectivity (fewer winners).
    bcm_theta_scale: float = 1.0
    # Minimum selectivity to count a granule as a memory-private WINNER (fraction of the
    # granule's drive concentrated on one memory). s = 1 - other_rate/top_rate.
    bcm_sel_thresh: float = 0.5
    # Heterosynaptic-LTD exponent on NON-selective granules: base excess *= s ** gamma
    # (s=1 private kept, s=0 shared washed out). Higher = harder suppression of shared mass.
    bcm_supp_gamma: float = 2.0
    # Potentiation weight (dg->answer units) written into a selective granule's answer
    # column, scaled by its selectivity. Comparable to the mean shared per-synapse excess.
    bcm_gain: float = 50.0


_ORIG_CONSOLIDATE = base._consolidate
_ORIG_BUILD = base.build_bridge


def _isolated_reactivation_rate(bridge, cfg, regions, memories):
    """Per-granule per-memory ISOLATED input reactivation rate (on-substrate spiking).
    Drive each memory's INPUT alone (fresh drain, competition on, plasticity + teacher
    off) and read DG firing. On a FRESH (pre-consolidation) substrate the memory-PRIVATE
    granules fire here -- this recovers the memory-selectivity of the fixed perforant
    wiring. (Measured on a twin: the #78 interleaved consolidation drives a persistent
    dominant-engram state in which the subordinate memory's input reactivates the WRONG
    engram, so this signal CANNOT be read off the post-consolidation substrate.) Returns
    rate[g, m] in [0,1] (spikes / window)."""
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 0.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    dg = regions["dg"]
    n_dg = int(dg.size)
    window = int(cfg.replay_on_steps + cfg.replay_settle_steps)
    names = list(memories.keys())
    rate = np.zeros((n_dg, len(names)), dtype=np.float64)
    pop_state = getattr(bridge, "_pop_state", None)
    for mi, name in enumerate(names):
        _drain(bridge)
        if pop_state is not None:
            pop_state["ever"][:] = False
            pop_state["integ"] = 0.0
            pop_state["drive"] = 0.0
            pop_state["silent"] = 0
        inp = memories[name]["input"]
        counts = np.zeros(n_dg, dtype=np.float64)
        for step in range(window):
            _zero_current(bridge)
            if step < cfg.replay_on_steps:
                bridge.cp_external_input_current[inp] = cfg.replay_input_drive_pA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            counts += np.asarray(to_host(bridge.cp_firing_states[dg]), dtype=np.float64)
        rate[:, mi] = counts / float(window)
    _zero_current(bridge)
    return rate, names


def _twin_selectivity(cfg, seed, regions, memories):
    """Measure the per-granule memory-selectivity on a FRESH twin bridge (same seed/cfg,
    so identical fixed perforant wiring + region indices), where the memory-private
    granules still reactivate. Returns rate[g, m] aligned to ``regions['dg']`` order.
    Deterministic; does not perturb the main bridge's consolidation."""
    twin, th = build_bridge_popsetpoint(int(seed), cfg)
    tw_regions = th["regions"]
    tw_inputs = _input_patterns(int(seed), cfg, "similar")
    tw_answers = _answer_assemblies(int(seed), cfg, tw_regions["answer"])
    tw_mems = {name: {"input": tw_inputs[name], "answer": tw_answers[name]} for name in memories}
    rate, names = _isolated_reactivation_rate(twin, cfg, tw_regions, tw_mems)
    return rate, names


def _bcm_selectivity_write(bridge, cfg, regions, memories, seed):
    """BCM sliding-threshold selectivity gate on the plastic dg->answer weights.

    theta_g = bcm_theta_scale * mean_m rate[g,m] (selectivity measured on a fresh twin);
    a granule selective for memory m (rate[g,m] > theta_g, drive concentrated on one
    memory) is POTENTIATED toward a_m; a non-selective (shared) granule is SUPPRESSED
    (base excess * s**gamma, heterosynaptic LTD). Suppression is applied to the base
    excess FIRST, then the selective potentiation is ADDED (so a private granule's write
    is never self-suppressed). Operates on the same (n_dg, n_answer) weight block the
    runner reads via the DG_WRITE_GATE indices."""
    from sim.backend import to_host

    idx = bridge._plasticity_gate_indices_gpu[DG_WRITE_GATE]
    w = np.asarray(to_host(bridge.cp_connections.data[idx]), dtype=np.float64)
    n_dg = int(cfg.n_dg)
    n_ans = int(cfg.n_answer)
    w0 = float(cfg.dg_answer_init_weight)
    if w.size != n_dg * n_ans:      # only the clean all-to-all block is handled
        return None
    E = np.maximum(w - w0, 0.0).reshape(n_dg, n_ans)

    # selectivity from a FRESH twin (private granules reactivate there; the post-consolidation
    # main substrate reactivates the wrong engram, so its selectivity is unreadable).
    rate, names = _twin_selectivity(cfg, seed, regions, memories)   # (n_dg, n_mem), dg order

    ans_sorted = np.sort(regions["answer"])
    col = {int(g): i for i, g in enumerate(ans_sorted)}
    a_cols = {name: np.asarray([col[int(x)] for x in memories[name]["answer"]], dtype=np.int64)
              for name in names}
    dg_sorted = np.sort(regions["dg"])
    g2r = {int(g): i for i, g in enumerate(dg_sorted)}
    dg_global = regions["dg"]  # rate rows are in regions["dg"] order

    top = rate.max(axis=1)
    if rate.shape[1] > 1:
        second = np.sort(rate, axis=1)[:, -2]
    else:
        second = np.zeros(n_dg)
    win_mem = rate.argmax(axis=1)
    theta = float(cfg.bcm_theta_scale) * rate.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sel = np.where(top > 1e-9, 1.0 - second / top, 0.0)   # 1=private, 0=shared
    sel = np.clip(sel, 0.0, 1.0)

    gamma = float(cfg.bcm_supp_gamma)
    gain = float(cfg.bcm_gain)
    thr = float(cfg.bcm_sel_thresh)

    # map per-granule (rate-row) quantities onto the weight-matrix rows (sorted dg)
    supp = np.ones(n_dg, dtype=np.float64)     # per weight-row suppression factor
    pot_rows = []                              # (weight-row, answer-cols, amount)
    n_winners = 0
    for ri in range(dg_global.size):
        g = int(dg_global[ri])
        wr = g2r[g]
        s = sel[ri]
        supp[wr] = s ** gamma                  # heterosynaptic LTD on non-selective
        if top[ri] > 1e-9 and s >= thr and rate[ri, win_mem[ri]] > theta[ri]:
            m = names[int(win_mem[ri])]
            pot_rows.append((wr, a_cols[m], gain * s))
            n_winners += 1

    E_new = E * supp[:, None]                  # (1) suppress the base-written shared mass
    for wr, cols, amt in pot_rows:             # (2) THEN add the selective private write
        E_new[wr, cols] += amt
    new_w = (w0 + E_new).reshape(-1)
    bridge.cp_connections.data[idx] = new_w.astype(bridge.cp_connections.data.dtype)
    return {
        "n_selective_winners": int(n_winners),
        "mean_selectivity": float(sel.mean()),
        "mean_supp_factor": float(supp.mean()),
    }


def _consolidate_bcm(bridge, cfg, regions, memories, competition, seed):
    """#78 base consolidation (on-substrate coincidence write) + the BCM selectivity gate.
    When ``cfg.bcm_gate`` is False this delegates to the UNMODIFIED #78 consolidate
    (byte-identical LESION)."""
    _ORIG_CONSOLIDATE(bridge, cfg, regions, memories, competition, seed)
    if getattr(cfg, "bcm_gate", False) and competition:
        _bcm_selectivity_write(bridge, cfg, regions, memories, seed)
        # restore the WRITE/READ gate state the base consolidate leaves (READ enabled)
        bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
        bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)


def _run_arm(seeds, cfg):
    base.build_bridge = build_bridge_popsetpoint
    base._consolidate = _consolidate_bcm
    try:
        return base.run(seeds, cfg)
    finally:
        base.build_bridge = _ORIG_BUILD
        base._consolidate = _ORIG_CONSOLIDATE


# ---------------------------------------------------------------------------------------
# WEIGHT-SPACE ANALYSIS (the mechanism-level evidence, measured BEFORE trusting both_win).
# Reads the FINAL dg->answer matrix for the LESION (#78 baseline) and the ON (BCM) write and
# reports private_m1(corr,wrong) + the two answer-vote margins for each, so the finding shows
# DIRECTLY whether the BCM gate raises the private write off baseline and breaks anti-symmetry.
# ---------------------------------------------------------------------------------------

def _final_matrix(bridge, cfg):
    from sim.backend import to_host
    idx = bridge._plasticity_gate_indices_gpu[DG_WRITE_GATE]
    w = np.asarray(to_host(bridge.cp_connections.data[idx]), dtype=np.float64)
    return w.reshape(int(cfg.n_dg), int(cfg.n_answer))


def _read_reactivation(bridge, cfg, regions, inp):
    """Probe-style READ reactivation: drive INPUT only (competition on), read the cumulative
    DG engram the READ actually recruits. Compared to the pre-consolidation engrams this
    exposes whether the post-consolidation read reactivates the memory's OWN engram or the
    dominant one -- the decisive dissociation (weights can separate, but the read reactivates
    the wrong engram)."""
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _drain(bridge)
    dg = regions["dg"]
    counts = np.zeros(dg.size, dtype=np.float64)
    for _ in range(cfg.probe_steps):
        _zero_current(bridge)
        bridge.cp_external_input_current[inp] = cfg.probe_input_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += np.asarray(to_host(bridge.cp_firing_states[dg]), dtype=np.float64)
    _zero_current(bridge)
    return dg[counts > 0]


def weight_space_analysis(seeds, cfg_on):
    rows = []
    for s in seeds:
        s = int(s)
        cfg_l = BConfig(**{**cfg_on.__dict__, "bcm_gate": False})
        # ----- LESION (#78 baseline) -----
        bl, hl = build_bridge_popsetpoint(s, cfg_l)
        regions = hl["regions"]
        inputs = _input_patterns(s, cfg_l, "similar")
        answers = _answer_assemblies(s, cfg_l, regions["answer"])
        mems = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
                "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
        eng0, _ = _dg_engram(bl, cfg_l, regions, inputs["m0"], True)
        eng1, _ = _dg_engram(bl, cfg_l, regions, inputs["m1"], True)
        _consolidate_bcm(bl, cfg_l, regions, mems, True, s)
        W_l = _final_matrix(bl, cfg_l)
        m_l = _weight_margins(W_l, cfg_l, eng0, eng1, answers["m0"], answers["m1"], regions)
        # ----- ON (BCM selectivity write) -----
        bo, ho = build_bridge_popsetpoint(s, cfg_on)
        regions_o = ho["regions"]
        inputs_o = _input_patterns(s, cfg_on, "similar")
        answers_o = _answer_assemblies(s, cfg_on, regions_o["answer"])
        mems_o = {"m0": {"input": inputs_o["m0"], "answer": answers_o["m0"]},
                  "m1": {"input": inputs_o["m1"], "answer": answers_o["m1"]}}
        eng0o, _ = _dg_engram(bo, cfg_on, regions_o, inputs_o["m0"], True)
        eng1o, _ = _dg_engram(bo, cfg_on, regions_o, inputs_o["m1"], True)
        _consolidate_bcm(bo, cfg_on, regions_o, mems_o, True, s)
        W_o = _final_matrix(bo, cfg_on)
        m_o = _weight_margins(W_o, cfg_on, eng0o, eng1o, answers_o["m0"], answers_o["m1"], regions_o)
        # READ-time reactivation collapse: does each memory's input reactivate its OWN
        # engram after consolidation, or the dominant one? (dissociation vs the weights)
        r0 = _read_reactivation(bo, cfg_on, regions_o, inputs_o["m0"])
        r1 = _read_reactivation(bo, cfg_on, regions_o, inputs_o["m1"])
        read_react = {
            "m0_to_eng0": _jaccard(r0, eng0o), "m0_to_eng1": _jaccard(r0, eng1o),
            "m1_to_eng0": _jaccard(r1, eng0o), "m1_to_eng1": _jaccard(r1, eng1o),
            "m1_reactivates_own": bool(_jaccard(r1, eng1o) > _jaccard(r1, eng0o)),
            "m0_reactivates_own": bool(_jaccard(r0, eng0o) > _jaccard(r0, eng1o)),
        }
        rows.append({
            "seed": s,
            "dg_jaccard": _jaccard(eng0, eng1),
            "read_reactivation": read_react,
            "lesion": {k: m_l[k] for k in ("m0", "m1", "both_positive", "antisymmetry_m0_plus_m1",
                                           "private_m1_to_correct", "private_m1_to_wrong",
                                           "private_m0_to_correct", "private_m0_to_wrong",
                                           "shared_to_a0", "shared_to_a1", "n_private_m1", "n_shared")},
            "on": {k: m_o[k] for k in ("m0", "m1", "both_positive", "antisymmetry_m0_plus_m1",
                                       "private_m1_to_correct", "private_m1_to_wrong",
                                       "private_m0_to_correct", "private_m0_to_wrong",
                                       "shared_to_a0", "shared_to_a1", "n_private_m1", "n_shared")},
        })
    both_pos_on = int(sum(1 for r in rows if r["on"]["both_positive"]))
    both_pos_lesion = int(sum(1 for r in rows if r["lesion"]["both_positive"]))
    # did the gate raise private_m1 discriminative weight off baseline? (corr > wrong ON)
    priv_written_on = int(sum(1 for r in rows
                              if r["on"]["private_m1_to_correct"] > r["on"]["private_m1_to_wrong"] + 1e-6))
    priv_written_lesion = int(sum(1 for r in rows
                                  if r["lesion"]["private_m1_to_correct"] > r["lesion"]["private_m1_to_wrong"] + 1e-6))
    # READ-time dissociation: how many seeds have BOTH memories reactivate their OWN engram
    # from their own input? (if the weights separate but this is < n, the residual is the
    # read reactivation, not the write.)
    both_reactivate_own = int(sum(1 for r in rows
                                  if r["read_reactivation"]["m0_reactivates_own"]
                                  and r["read_reactivation"]["m1_reactivates_own"]))
    return {
        "note": "weight-space dg->answer margins + private/shared decomposition + read-time reactivation, LESION vs ON",
        "n_seeds_both_positive_on": both_pos_on,
        "n_seeds_both_positive_lesion": both_pos_lesion,
        "n_seeds_private_m1_written_on": priv_written_on,
        "n_seeds_private_m1_written_lesion": priv_written_lesion,
        "n_seeds_both_reactivate_own_on": both_reactivate_own,
        "rows": rows,
    }


def _seed_row(on_seed, off_seed):
    on = on_seed["conditions"]["similar_separator_on"]
    off = off_seed["conditions"]["similar_separator_on"]

    def _direct(row):
        d = row.get("direct_readout")
        if not d:
            return None
        return {"m0": d["m0_engram_selectivity"], "m1": d["m1_engram_selectivity"],
                "both_positive": bool(d["m0_engram_selectivity"] > 0 and d["m1_engram_selectivity"] > 0)}

    return {
        "seed": on_seed["seed"],
        "both_win_on": on["both_win"],
        "both_win_lesion": off["both_win"],
        "per_memory_selectivity_on": on["per_memory_selectivity"],
        "per_memory_selectivity_lesion": off["per_memory_selectivity"],
        "mean_selectivity_on": on["mean_selectivity"],
        "mean_selectivity_lesion": off["mean_selectivity"],
        "direct_readout_on": _direct(on),
        "direct_readout_lesion": _direct(off),
        "dg_sizes_on": (on["dg_separation"]["dg_size_m0"], on["dg_separation"]["dg_size_m1"]),
        "dg_jaccard_on": on["dg_separation"]["dg_jaccard"],
        "single_selectivity_on": on_seed["summary"]["single_selectivity"],
        "single_scramble_on": on_seed["summary"]["single_scramble_selectivity"],
        "dissimilar_both_win_on": on_seed["summary"]["dissimilar_both_win"],
    }


def run(seeds, cfg_on, weight_analysis=True):
    started = time.time()
    cfg_lesion = BConfig(**{**cfg_on.__dict__, "bcm_gate": False})

    on_payload = _run_arm(seeds, cfg_on)
    lesion_payload = _run_arm(seeds, cfg_lesion)
    wsa = weight_space_analysis(seeds, cfg_on) if weight_analysis else None

    by_seed_on = {r["seed"]: r for r in on_payload["per_seed"]}
    by_seed_off = {r["seed"]: r for r in lesion_payload["per_seed"]}
    rows = [_seed_row(by_seed_on[int(s)], by_seed_off[int(s)]) for s in seeds]
    n = len(rows)

    def cnt(fn):
        return int(sum(1 for r in rows if fn(r)))

    both_win_on = cnt(lambda r: r["both_win_on"])
    both_win_lesion = cnt(lambda r: r["both_win_lesion"])
    direct_both_on = cnt(lambda r: r["direct_readout_on"] and r["direct_readout_on"]["both_positive"])
    direct_both_lesion = cnt(lambda r: r["direct_readout_lesion"] and r["direct_readout_lesion"]["both_positive"])
    single_ok = cnt(lambda r: r["single_selectivity_on"] >= 0.30)
    scramble_ok = cnt(lambda r: r["single_scramble_on"] <= r["single_selectivity_on"] - 0.30)
    dissim_ok = cnt(lambda r: r["dissimilar_both_win_on"])

    # did the gate demonstrably alter the learned map / write the private granules?
    manip_landed = False
    priv_written_on = None
    if wsa:
        priv_written_on = int(wsa["n_seeds_private_m1_written_on"])
        for r in wsa["rows"]:
            if (abs(r["on"]["m0"] - r["lesion"]["m0"]) > 0.02
                    or abs(r["on"]["m1"] - r["lesion"]["m1"]) > 0.02
                    or r["on"]["private_m1_to_correct"] > r["lesion"]["private_m1_to_correct"] + 1e-6):
                manip_landed = True
                break

    checks = {
        "both_similar_discriminable_6of6": both_win_on == n,
        "lesion_reproduces_residual": both_win_lesion < n,
        "bcm_write_lands": bool(manip_landed),
        "private_m1_written_on_all": (priv_written_on == n) if priv_written_on is not None else None,
        "single_recall_ceiling": single_ok == n,
        "scramble_inverts": scramble_ok == n,
        "dissimilar_both_win": dissim_ok == n,
    }
    go = (checks["both_similar_discriminable_6of6"]
          and checks["lesion_reproduces_residual"]
          and checks["single_recall_ceiling"]
          and checks["scramble_inverts"]
          and checks["dissimilar_both_win"])

    both_win_attrib = attributable_to("both_win ON vs LESION", both_win_on, both_win_lesion)

    v = Verdict("memory-separator readout residual via a BCM SELECTIVITY-GATED dg->answer WRITE")
    v.require("pipeline intact: single-memory recall at ceiling (6/6)", single_ok == n, expect=True)
    v.require("read rides the learned mapping: scramble inverts (6/6)", scramble_ok == n, expect=True)
    v.require("baseline is the #78 residual: lesion both_win fails", both_win_lesion < n, expect=True)
    v.require("manipulation landed: the BCM write demonstrably alters the learned map",
              bool(manip_landed), expect=True)
    v.disabled("STDP / reward / short-term & structural plasticity / OU noise",
               why="isolation inherited from the #71 runner; separator + Hebbian write + this BCM gate are the only live plasticity")
    decided = v.decide(go=go)

    return {
        "gate": "replay_dg_pattern_separation_bcm",
        "mechanism": "BCM selectivity-gated dg->answer WRITE: isolated-reactivation selectivity -> potentiate memory-private granules toward their answer, heterosynaptic-LTD suppress shared (on top of the #78 population set-point)",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        "status": decided["status"],
        "verdict": decided,
        "preconditions": decided["preconditions"],
        "both_win_attributable_to_bcm": both_win_attrib,
        "weight_space_analysis": wsa,
        "checks": checks,
        "bcm_config": {
            "bcm_gate": cfg_on.bcm_gate,
            "theta_scale": cfg_on.bcm_theta_scale,
            "sel_thresh": cfg_on.bcm_sel_thresh,
            "supp_gamma": cfg_on.bcm_supp_gamma,
            "gain": cfg_on.bcm_gain,
            "pop_setpoint": cfg_on.pop_setpoint,
            "pop_k_target": cfg_on.pop_k_target,
        },
        "pooled": {
            "both_win_on_count": both_win_on,
            "both_win_lesion_count": both_win_lesion,
            "direct_both_positive_on_count": direct_both_on,
            "direct_both_positive_lesion_count": direct_both_lesion,
            "private_m1_written_on_count": priv_written_on,
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
            "BCM selectivity gate applied as a runner-side transform on the plastic dg->answer weights "
            "(potentiation of memory-private granules toward their answer + heterosynaptic LTD of shared); "
            "the ISOLATED reactivation firing (selectivity signal) and the read stay on-substrate spiking",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bcm-gate", dest="bcm_gate", action="store_true",
                    help="enable the BCM selectivity write (default OFF = #78 baseline)")
    ap.add_argument("--theta-scale", type=float, default=None)
    ap.add_argument("--sel-thresh", type=float, default=None)
    ap.add_argument("--supp-gamma", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--k-target", type=float, default=None)
    ap.add_argument("--no-weight-analysis", dest="weight_analysis", action="store_false")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    overrides = {"bcm_gate": bool(args.bcm_gate)}
    if args.smoke:
        sm = base.smoke_config()
        overrides.update(sm.__dict__)
        overrides["bcm_gate"] = bool(args.bcm_gate)
    for name, val in (("bcm_theta_scale", args.theta_scale),
                      ("bcm_sel_thresh", args.sel_thresh),
                      ("bcm_supp_gamma", args.supp_gamma),
                      ("bcm_gain", args.gain),
                      ("pop_k_target", args.k_target)):
        if val is not None:
            overrides[name] = val
    cfg = BConfig(**overrides)

    print(f"[sep-bcm] backend={os.environ.get('SIM_BACKEND','default')} "
          f"seeds={args.seeds} smoke={args.smoke} "
          f"bcm_gate={cfg.bcm_gate} theta={cfg.bcm_theta_scale} sel_thresh={cfg.bcm_sel_thresh} "
          f"gamma={cfg.bcm_supp_gamma} gain={cfg.bcm_gain} pop_k={cfg.pop_k_target}", flush=True)
    payload = run(args.seeds, cfg, weight_analysis=args.weight_analysis)
    for r in payload["per_seed"]:
        pm = r["per_memory_selectivity_on"]
        d_on = r["direct_readout_on"]
        d_on_s = f"({d_on['m0']:+.2f},{d_on['m1']:+.2f})" if d_on else "NA"
        print(f"  seed {r['seed']}: "
              f"ON both_win={r['both_win_on']} m0={pm['m0']:+.2f} m1={pm['m1']:+.2f} "
              f"direct_map ON={d_on_s} | LESION both_win={r['both_win_lesion']} | "
              f"single={r['single_selectivity_on']:+.2f} scr={r['single_scramble_on']:+.2f} "
              f"dissim_bw={r['dissimilar_both_win_on']} sizes={r['dg_sizes_on']} dgJ={r['dg_jaccard_on']:.2f}",
              flush=True)
    wsa = payload.get("weight_space_analysis")
    if wsa:
        print(f"  WEIGHT-SPACE both-positive ON={wsa['n_seeds_both_positive_on']}/{len(wsa['rows'])} "
              f"LESION={wsa['n_seeds_both_positive_lesion']}/{len(wsa['rows'])} | "
              f"private_m1 written ON={wsa['n_seeds_private_m1_written_on']}/{len(wsa['rows'])} "
              f"LESION={wsa['n_seeds_private_m1_written_lesion']}/{len(wsa['rows'])} | "
              f"READ both-reactivate-own ON={wsa['n_seeds_both_reactivate_own_on']}/{len(wsa['rows'])}", flush=True)
        for r in wsa["rows"]:
            lo, on, rr = r["lesion"], r["on"], r["read_reactivation"]
            print(f"    seed {r['seed']}: dgJ={r['dg_jaccard']:.2f} | "
                  f"LESION m=({lo['m0']:+.2f},{lo['m1']:+.2f}) antisym={lo['antisymmetry_m0_plus_m1']:+.2f} "
                  f"privM1(c={lo['private_m1_to_correct']:.0f},w={lo['private_m1_to_wrong']:.0f}) | "
                  f"ON m=({on['m0']:+.2f},{on['m1']:+.2f}) antisym={on['antisymmetry_m0_plus_m1']:+.2f} "
                  f"privM1(c={on['private_m1_to_correct']:.0f},w={on['private_m1_to_wrong']:.0f}) "
                  f"shared(a0={on['shared_to_a0']:.0f},a1={on['shared_to_a1']:.0f}) | "
                  f"READ m1->J(eng0={rr['m1_to_eng0']:.2f},eng1={rr['m1_to_eng1']:.2f})", flush=True)
    print(f"  STATUS: {payload['status']}", flush=True)
    print(f"  checks: {json.dumps(payload['checks'])}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled'])}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
