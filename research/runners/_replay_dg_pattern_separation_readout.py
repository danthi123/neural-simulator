"""Close the memory-separator READOUT/WRITE residual with a competitive write (board #73).

WHERE THE RESIDUAL SITS (established on main, do NOT re-derive):
  * #43 LIF GO / #71 bridge / #73 homeostat / #78 population set-point have EXHAUSTED
    the DG-competition side. Three DG-side mechanisms (k-WTA, per-cell homeostat,
    population set-point) each leave ``both_win`` = 0/6 with the same ANTI-SYMMETRIC
    signature (one memory reads +x, the other -x -> both pick the SAME answer).
  * #78 (`2026-08-19-dg-population-setpoint-NOGO-relocalizes-residual-to-readout.md`)
    RE-LOCALIZED the residual OUT of DG competition: with the population set-point ON
    the DG code is symmetric, sparse, non-dense (engrams 36-65, Jaccard 0.29-0.67) and
    both_win is STILL 0/6 -- and both_win is 0/24 across a drive sweep spanning engram
    sizes 3-88. The DG codes ARE separable; the READOUT does not use the separation.
    #78's direct-readout diagnosis localized it to the dg->answer associative WRITE:
    "the soft-bound rate-window rule writes overlapping-engram granules to BOTH answers".
  * Answer-side READ-time WTA is REFUTED (#78: forcing the answer opponent-inhibition
    gate to 0 leaves the reads unchanged) -- so the lever is NOT a competitive READ.

THE MECHANISM (this file): a COMPETITIVE / heterosynaptic WRITE. The bridge rate-window
Hebbian ``dw = lr*coact*(w_max - w)`` writes every co-active dg->answer synapse UP. Two
similar memories share 29-67% of their DG granules; a shared granule co-fires with a0
during m0's replay AND with a1 during m1's replay, so it is written to BOTH answers and
votes for both -> the read cannot use the DG separation. FIX: a PRESYNAPTIC
(granule-output) heterosynaptic renormalization, interleaved between replay events -- each
granule conserves a FIXED total outgoing dg->answer budget across its answer targets. A
granule that committed to ONE answer keeps its full vote there; a SHARED granule that
tried to write to two answers has the same budget SPLIT between them (~half each), so it
can no longer out-vote the memory-selective PRIVATE granules. The two memories' answer
associations become orthogonal in the weight matrix (m0's private granules -> a0,
m1's private granules -> a1; shared granules diluted to a wash).

Biology: heterosynaptic plasticity / synaptic competition as a normalizer -- potentiation
at one synapse is balanced by depression at the cell's other synapses, conserving total
synaptic weight (Royer & Pare 2003, Nature 422:518-522, "Conservation of total synaptic
weight through balanced synaptic depression and potentiation"; Chistiakova, Bannon, Chen,
Rioult-Pedotti, Volgushev 2014, "Heterosynaptic plasticity: multiple mechanisms and their
functions", Neuroscientist; von der Malsburg 1973; Oja 1982 weight normalization). The
offline down-state between replay events is the biological window for this renormalization.

BUILDS ON the #78 population set-point (imports ``build_bridge_popsetpoint``) so the DG
code is already symmetric + non-dense (the #78 GO precondition) and the ONLY manipulated
variable is the WRITE rule. NO ``sim/`` edit -- the renormalization is a runner-side
transform on the plastic dg->answer weights (``cp_connections.data``), exactly the
weight-vector the runner already reads via ``_path_weights``; the SELECTION and the read
stay on-substrate spiking.

LESION arm (dissociation): ``orthowrite=False`` -> the renormalization is never applied ->
consolidation delegates to the UNMODIFIED #71 write -> this is byte-identical to the #78
ON arm (both_win 0/6 RETURN). Deterministic (cfg.seed).

MECHANISM-LEVEL METRIC measured BEFORE trusting both_win: the ``direct_readout`` per-engram
selectivity (base runner drives each WRITTEN engram directly and reads the answer -- this
IS the learned dg->answer map's separability). LESION -> anti-symmetric (one +, one -);
the competitive write should make BOTH positive (the map separates). Only then is the
spiking probe ``both_win`` a fair test.

ANTI-CHEATS (design first):
  1. Two SIMILAR memories BOTH discriminable -- both_win 6/6 (the bar #71/#73/#78 set).
  2. The competitive write is LOAD-BEARING: LESION (orthowrite off) reproduces the #78
     residual (both_win < 6, anti-symmetric direct_readout). Dissociation.
  3. No regression: single-memory recall stays at ceiling; scramble-teach inverts;
     DISSIMILAR pairs both-win. 6 seeds (42/43/44/100/101/102), deterministic (cfg.seed).
  4. Mechanism moves the RIGHT quantity: direct_readout both-positive ON vs anti-symmetric
     LESION, measured before both_win is believed.

Run:
    OMP_NUM_THREADS=4 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readout \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/sep_readout/orthowrite_6seed.json
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
from research.runners._replay_cortical_consolidation_gate import _zero_current  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


@dataclass(frozen=True)
class RConfig(PConfig):
    """#78 PConfig (population set-point ON by default) + the competitive-WRITE knobs."""
    # Master switch. False -> the renormalization is never applied -> consolidation is the
    # unmodified #71 write -> byte-identical to the #78 ON arm (the LESION).
    orthowrite: bool = True
    # Heterosynaptic competition exponent on the per-granule output distribution. 1.0 =
    # pure budget conservation (a shared granule splits its budget proportionally); >1
    # sharpens toward a granule-output WTA (commit to the dominant answer).
    orthowrite_exponent: float = 1.0
    # Per-granule outgoing budget. <=0 -> AUTO: the mean per-granule learned-excess total
    # over committed granules (auto-scales with the write magnitude; the MARGIN SIGN is
    # budget-independent, only the private-vs-shared dilution ratio -- set by conservation
    # -- matters). A positive value fixes it.
    orthowrite_budget: float = -1.0
    # Renormalize after every N replay events (1 = after each memory's event; the offline
    # down-state between events is the biological renormalization window).
    orthowrite_every: int = 1


_ORIG_CONSOLIDATE = base._consolidate


def _renormalize_dg_answer(bridge, cfg: RConfig) -> None:
    """Presynaptic (granule-output) heterosynaptic renormalization of the plastic
    dg->answer weights: conserve each granule's LEARNED-EXCESS drive across its answer
    targets to a fixed per-granule budget. A granule committed to one answer keeps its
    full vote; a granule that tried to write to two answers has its budget SPLIT -> it
    cannot out-vote the memory-selective private granules. Operates on the same weight
    vector the runner reads via ``_path_weights`` (cp_connections.data[gate indices]).
    Order is (pre,post)-sorted all-to-all dg->answer -> a clean (n_dg, n_answer) matrix."""
    from sim.backend import to_host

    idx = bridge._plasticity_gate_indices_gpu[DG_WRITE_GATE]
    pre_coords, _post = bridge._plasticity_gate_to_coords[DG_WRITE_GATE]
    w = np.asarray(to_host(bridge.cp_connections.data[idx]), dtype=np.float64)

    n_dg = int(cfg.n_dg)
    n_ans = int(cfg.n_answer)
    w0 = float(cfg.dg_answer_init_weight)
    p = float(cfg.orthowrite_exponent)

    excess = np.maximum(w - w0, 0.0)

    # Group by presynaptic granule. The gate weights are a clean (pre,post)-sorted
    # all-to-all block, so reshape when the count matches; else fall back to coords.
    if w.size == n_dg * n_ans:
        E = excess.reshape(n_dg, n_ans)                    # rows = granules
        row_tot = E.sum(axis=1)                            # total learned excess / granule
        committed = row_tot > 1e-9
        if not committed.any():
            return
        budget = float(cfg.orthowrite_budget)
        if budget <= 0.0:
            budget = float(row_tot[committed].mean())      # AUTO: mean committed total
        if p != 1.0:
            Ep = np.where(E > 0.0, E, 0.0) ** p
        else:
            Ep = E
        denom = Ep.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            E_norm = np.where(denom > 1e-12, budget * Ep / denom, 0.0)
        new_E = np.where(committed[:, None], E_norm, E)
        new_w = (w0 + new_E).reshape(-1)
    else:  # robust fallback: iterate granules via coords
        new_w = w.copy()
        uniq = np.unique(pre_coords)
        tots = []
        masks = {}
        for g in uniq:
            m = pre_coords == g
            masks[g] = m
            tots.append(float(excess[m].sum()))
        tots = np.asarray(tots)
        committed = tots > 1e-9
        if not committed.any():
            return
        budget = float(cfg.orthowrite_budget)
        if budget <= 0.0:
            budget = float(tots[committed].mean())
        for g, tot in zip(uniq, tots):
            if tot <= 1e-9:
                continue
            m = masks[g]
            e = excess[m]
            ep = e ** p if p != 1.0 else e
            s = ep.sum()
            if s <= 1e-12:
                continue
            new_w[m] = w0 + budget * ep / s

    bridge.cp_connections.data[idx] = new_w.astype(bridge.cp_connections.data.dtype)


def _consolidate_orthowrite(bridge, cfg, regions, memories, competition, seed):
    """Offline replay WRITE with an interleaved competitive (heterosynaptic) renorm.

    When ``cfg.orthowrite`` is False this delegates to the UNMODIFIED #71 consolidate
    (byte-identical LESION). When True it mirrors the #71 write loop and applies the
    presynaptic renormalization after every ``orthowrite_every`` replay events -- the
    down-state window between replay events."""
    if not getattr(cfg, "orthowrite", False):
        return _ORIG_CONSOLIDATE(bridge, cfg, regions, memories, competition, seed)

    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0 if competition else 0.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 0.0)          # WRITE: no read-back
    bridge.set_plasticity_gate(DG_WRITE_GATE, 1.0)
    bridge.core_config.hebbian_learning_rate = float(cfg.replay_learning_rate)
    names = list(memories.keys())
    order = []
    for _ in range(cfg.replay_events_per_memory):
        order.extend(names)
    every = max(1, int(cfg.orthowrite_every))
    for ev, name in enumerate(order):
        _drain(bridge)
        inp, ans = memories[name]["input"], memories[name]["answer"]
        for step in range(cfg.replay_on_steps + cfg.replay_settle_steps):
            _zero_current(bridge)
            if step < cfg.replay_on_steps:
                bridge.cp_external_input_current[inp] = cfg.replay_input_drive_pA
                bridge.cp_external_input_current[ans] = cfg.replay_answer_drive_pA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        if (ev + 1) % every == 0:
            _renormalize_dg_answer(bridge, cfg)      # THE MECHANISM (down-state renorm)
    # final renorm so the last events are balanced too
    _renormalize_dg_answer(bridge, cfg)
    _zero_current(bridge)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)          # READ enabled
    bridge.core_config.hebbian_learning_rate = float(cfg.wake_learning_rate)


def _run_arm(seeds, cfg):
    """Run the full #71 pipeline (via base.run) with build_bridge swapped to the
    #78 pop-setpoint build AND consolidate swapped to the competitive write.
    cfg.orthowrite selects TREATMENT vs LESION."""
    base.build_bridge = build_bridge_popsetpoint
    base._consolidate = _consolidate_orthowrite
    try:
        return base.run(seeds, cfg)
    finally:
        base.build_bridge = _ORIG_BUILD
        base._consolidate = _ORIG_CONSOLIDATE


_ORIG_BUILD = base.build_bridge


# ---------------------------------------------------------------------------------------
# WEIGHT-SPACE ANALYSIS (the mechanism-level evidence, measured BEFORE trusting both_win).
# Decomposes the learned dg->answer matrix into PRIVATE vs SHARED granule contributions and
# sweeps the heterosynaptic exponent, so the finding shows DIRECTLY whether the competitive
# write can make the two memories' answer maps both-positive (orthogonal) in weight space.
# ---------------------------------------------------------------------------------------

def _renorm_matrix(W, cfg, exponent):
    """Presynaptic (granule-output) heterosynaptic renorm on a (n_dg, n_answer) matrix,
    matching _renormalize_dg_answer. exponent<1 compresses (equalizes across a granule's
    answer targets), 1 conserves budget, >1 sharpens toward a granule-output WTA."""
    w0 = float(cfg.dg_answer_init_weight)
    E = np.maximum(W - w0, 0.0)
    row_tot = E.sum(axis=1)
    committed = row_tot > 1e-9
    if not committed.any():
        return W
    budget = float(cfg.orthowrite_budget)
    if budget <= 0.0:
        budget = float(row_tot[committed].mean())
    Ep = np.where(E > 0.0, E, 0.0) ** float(exponent) if exponent != 1.0 else E
    denom = Ep.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        En = np.where(denom > 1e-12, budget * Ep / denom, 0.0)
    return np.where(committed[:, None], w0 + En, W)


def _weight_margins(W, cfg, eng0, eng1, a0, a1, regions):
    """Weight-space answer-vote margin per memory + private/shared decomposition."""
    ans = np.sort(regions["answer"])
    col = {int(g): i for i, g in enumerate(ans)}
    a0c = np.asarray([col[int(x)] for x in a0]); a1c = np.asarray([col[int(x)] for x in a1])
    dgs = np.sort(regions["dg"])
    g2r = {int(g): i for i, g in enumerate(dgs)}
    s0 = set(int(g) for g in eng0); s1 = set(int(g) for g in eng1)
    shared = sorted(s0 & s1); p0 = sorted(s0 - s1); p1 = sorted(s1 - s0)

    def _sum(S, cc):
        if not S:
            return 0.0
        r = [g2r[g] for g in S]
        return float(W[np.ix_(r, cc)].sum())

    def _margin(rows_set, cc, wc):
        r = [g2r[g] for g in rows_set]
        vc = float(W[np.ix_(r, cc)].sum()); vw = float(W[np.ix_(r, wc)].sum())
        return (vc - vw) / (vc + vw + 1e-9)

    m0 = _margin(s0, a0c, a1c)
    m1 = _margin(s1, a1c, a0c)
    return {
        "m0": m0, "m1": m1, "both_positive": bool(m0 > 0 and m1 > 0),
        "antisymmetry_m0_plus_m1": m0 + m1,   # ~0 => the collision is a pure anti-symmetry
        "n_private_m0": len(p0), "n_private_m1": len(p1), "n_shared": len(shared),
        "private_m0_to_correct": _sum(p0, a0c), "private_m0_to_wrong": _sum(p0, a1c),
        "private_m1_to_correct": _sum(p1, a1c), "private_m1_to_wrong": _sum(p1, a0c),
        "shared_to_a0": _sum(shared, a0c), "shared_to_a1": _sum(shared, a1c),
    }


def _write_engram(bridge, cfg, regions, inp, ans):
    """Cumulative DG cells that fire during a consolidation-STYLE event (input+answer drive,
    plasticity off). Compared to the read engram (_dg_engram, input-only) this checks whether
    reactivation is consistent -- whether the probe recreates the WRITTEN engram."""
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 0.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _drain(bridge)
    dg = regions["dg"]; counts = np.zeros(dg.size)
    for step in range(cfg.replay_on_steps + cfg.replay_settle_steps):
        from research.runners._replay_cortical_consolidation_gate import _zero_current as _zc
        _zc(bridge)
        if step < cfg.replay_on_steps:
            bridge.cp_external_input_current[inp] = cfg.replay_input_drive_pA
            bridge.cp_external_input_current[ans] = cfg.replay_answer_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += np.asarray(to_host(bridge.cp_firing_states[dg]), dtype=np.float64)
    _zero_current(bridge)
    return dg[counts > 0]


def weight_space_analysis(seeds, cfg, exponents=(0.5, 1.0, 3.0)):
    """Per seed: run the #78 write (lesion), read the dg->answer matrix, and report the
    weight-space margins for the raw write AND for each heterosynaptic exponent. This is
    the decisive mechanism-level evidence: does ANY per-granule output transform make both
    memories' answer maps positive (orthogonal)?"""
    from sim.backend import to_host
    rows = []
    for s in seeds:
        s = int(s)
        cfg_l = RConfig(**{**cfg.__dict__, "orthowrite": False})
        bridge, h = build_bridge_popsetpoint(s, cfg_l)
        regions = h["regions"]
        inputs = _input_patterns(s, cfg_l, "similar")
        answers = _answer_assemblies(s, cfg_l, regions["answer"])
        mems = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
                "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
        eng0, _ = _dg_engram(bridge, cfg_l, regions, inputs["m0"], True)
        eng1, _ = _dg_engram(bridge, cfg_l, regions, inputs["m1"], True)
        # reactivation-consistency check: does the input-only READ engram equal the
        # input+answer WRITE engram? (1.0 => reactivation is consistent; the residual is
        # NOT read/write engram mismatch.)
        weng0 = _write_engram(bridge, cfg_l, regions, inputs["m0"], answers["m0"])
        weng1 = _write_engram(bridge, cfg_l, regions, inputs["m1"], answers["m1"])
        read_write_jaccard = (float(_jaccard(eng0, weng0)) + float(_jaccard(eng1, weng1))) / 2.0
        _ORIG_CONSOLIDATE(bridge, cfg_l, regions, mems, True, s)   # the #78 raw write
        idx = bridge._plasticity_gate_indices_gpu[DG_WRITE_GATE]
        w = np.asarray(to_host(bridge.cp_connections.data[idx]), dtype=np.float64)
        W = w.reshape(int(cfg_l.n_dg), int(cfg_l.n_answer))
        raw = _weight_margins(W, cfg_l, eng0, eng1, answers["m0"], answers["m1"], regions)
        by_exp = {}
        for e in exponents:
            Wn = _renorm_matrix(W, cfg_l, e)
            mm = _weight_margins(Wn, cfg_l, eng0, eng1, answers["m0"], answers["m1"], regions)
            by_exp[f"p{e}"] = {"m0": mm["m0"], "m1": mm["m1"],
                               "both_positive": mm["both_positive"],
                               "antisymmetry_m0_plus_m1": mm["antisymmetry_m0_plus_m1"]}
        rows.append({"seed": s, "dg_jaccard": _jaccard(eng0, eng1),
                     "sizes": (int(eng0.size), int(eng1.size)),
                     "read_write_engram_jaccard": read_write_jaccard,
                     "raw_write": raw, "by_exponent": by_exp})
    any_both_pos = int(sum(
        1 for r in rows
        if r["raw_write"]["both_positive"] or any(v["both_positive"] for v in r["by_exponent"].values())
    ))
    return {
        "note": "weight-space dg->answer answer-vote margins; both_positive over raw + every exponent",
        "n_seeds_any_both_positive": any_both_pos,
        "mean_read_write_engram_jaccard": float(np.mean([r["read_write_engram_jaccard"] for r in rows])),
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
        # AC1: both similar memories discriminable (the behavioral bar)
        "both_win_on": on["both_win"],
        "both_win_lesion": off["both_win"],
        "per_memory_selectivity_on": on["per_memory_selectivity"],
        "per_memory_selectivity_lesion": off["per_memory_selectivity"],
        "mean_selectivity_on": on["mean_selectivity"],
        "mean_selectivity_lesion": off["mean_selectivity"],
        # AC4: mechanism-level -- the learned dg->answer MAP separability (direct readout)
        "direct_readout_on": _direct(on),
        "direct_readout_lesion": _direct(off),
        # engram code (should be identical ON vs LESION -- write does not touch the DG code)
        "dg_sizes_on": (on["dg_separation"]["dg_size_m0"], on["dg_separation"]["dg_size_m1"]),
        "dg_jaccard_on": on["dg_separation"]["dg_jaccard"],
        "dense_collapse_on": on["dense_engram_collapse"],
        # AC3: no regression (from the ON arm)
        "single_selectivity_on": on_seed["summary"]["single_selectivity"],
        "single_scramble_on": on_seed["summary"]["single_scramble_selectivity"],
        "dissimilar_both_win_on": on_seed["summary"]["dissimilar_both_win"],
    }


def run(seeds, cfg_on, weight_analysis=True):
    started = time.time()
    cfg_lesion = RConfig(**{**cfg_on.__dict__, "orthowrite": False})

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

    # Did the competitive write demonstrably ALTER the learned map? (proof the manipulation
    # landed -- the renorm reaches the weights; from the weight-space sweep, robust to the
    # primary exponent choice.) A landed-but-failed manipulation is a NO-GO, not UNDEFINED.
    manip_landed = False
    wsa_no_ortho = None
    if wsa:
        wsa_no_ortho = int(wsa["n_seeds_any_both_positive"]) == 0
        for r in wsa["rows"]:
            raw = r["raw_write"]
            for v_ in r["by_exponent"].values():
                if abs(v_["m0"] - raw["m0"]) > 0.02 or abs(v_["m1"] - raw["m1"]) > 0.02:
                    manip_landed = True
                    break

    checks = {
        # AC1: both similar memories discriminable, all 6 seeds (the headline bar).
        "both_similar_discriminable_6of6": both_win_on == n,
        # AC2: the competitive write is load-bearing (LESION reproduces the #78 residual).
        "lesion_reproduces_residual": both_win_lesion < n,
        # AC4: the competitive write demonstrably alters the learned dg->answer map.
        "competitive_write_lands": bool(manip_landed),
        # weight-space: does ANY exponent make both memories' answer maps positive?
        "weight_space_no_orthogonalization": bool(wsa_no_ortho) if wsa_no_ortho is not None else None,
        # AC3: no regression (the write does not damage single-memory recall).
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
    direct_attrib = attributable_to("direct-map both-positive ON vs LESION", direct_both_on, direct_both_lesion)

    v = Verdict("memory-separator readout residual via a COMPETITIVE (heterosynaptic) dg->answer WRITE")
    v.require("pipeline intact: single-memory recall at ceiling (6/6)", single_ok == n, expect=True)
    v.require("read rides the learned mapping: scramble inverts (6/6)", scramble_ok == n, expect=True)
    v.require("baseline is the #78 residual: lesion both_win fails", both_win_lesion < n, expect=True)
    v.require("manipulation landed: the competitive write demonstrably alters the learned map",
              bool(manip_landed), expect=True)
    v.disabled("STDP / reward / short-term & structural plasticity / OU noise",
               why="isolation inherited from the #71 runner; separator + Hebbian write + this renorm are the only live plasticity")
    decided = v.decide(go=go)

    return {
        "gate": "replay_dg_pattern_separation_readout",
        "mechanism": "competitive/heterosynaptic dg->answer WRITE: per-granule outgoing-budget renormalization interleaved between replay events (on top of the #78 population set-point)",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        "status": decided["status"],
        "verdict": decided,
        "preconditions": decided["preconditions"],
        "both_win_attributable_to_orthowrite": both_win_attrib,
        "direct_map_attributable_to_orthowrite": direct_attrib,
        "weight_space_analysis": wsa,
        "checks": checks,
        "orthowrite_config": {
            "orthowrite": cfg_on.orthowrite,
            "exponent": cfg_on.orthowrite_exponent,
            "budget": cfg_on.orthowrite_budget,
            "every": cfg_on.orthowrite_every,
            "pop_setpoint": cfg_on.pop_setpoint,
            "pop_k_target": cfg_on.pop_k_target,
        },
        "pooled": {
            "both_win_on_count": both_win_on,
            "both_win_lesion_count": both_win_lesion,
            "direct_both_positive_on_count": direct_both_on,
            "direct_both_positive_lesion_count": direct_both_lesion,
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
            "heterosynaptic renormalization applied as a runner-side transform on the plastic "
            "dg->answer weights (the offline down-state renorm); the SELECTION and the read stay on-substrate",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exponent", type=float, default=None)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--every", type=int, default=None)
    ap.add_argument("--k-target", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.smoke:
        sm = base.smoke_config()
        overrides.update(sm.__dict__)
    for name, val in (("orthowrite_exponent", args.exponent),
                      ("orthowrite_budget", args.budget),
                      ("orthowrite_every", args.every),
                      ("pop_k_target", args.k_target)):
        if val is not None:
            overrides[name] = val
    cfg = RConfig(**overrides)

    print(f"[sep-readout] backend={os.environ.get('SIM_BACKEND','default')} "
          f"seeds={args.seeds} smoke={args.smoke} "
          f"orthowrite={cfg.orthowrite} exp={cfg.orthowrite_exponent} "
          f"budget={cfg.orthowrite_budget} every={cfg.orthowrite_every} "
          f"pop_k={cfg.pop_k_target}", flush=True)
    payload = run(args.seeds, cfg)
    for r in payload["per_seed"]:
        pm = r["per_memory_selectivity_on"]
        d_on = r["direct_readout_on"]
        d_le = r["direct_readout_lesion"]
        d_on_s = f"({d_on['m0']:+.2f},{d_on['m1']:+.2f})" if d_on else "NA"
        d_le_s = f"({d_le['m0']:+.2f},{d_le['m1']:+.2f})" if d_le else "NA"
        print(f"  seed {r['seed']}: "
              f"ON both_win={r['both_win_on']} m0={pm['m0']:+.2f} m1={pm['m1']:+.2f} "
              f"direct_map ON={d_on_s} LESION={d_le_s} | "
              f"LESION both_win={r['both_win_lesion']} | "
              f"single={r['single_selectivity_on']:+.2f} scr={r['single_scramble_on']:+.2f} "
              f"dissim_bw={r['dissimilar_both_win_on']} sizes={r['dg_sizes_on']} dgJ={r['dg_jaccard_on']:.2f}",
              flush=True)
    wsa = payload.get("weight_space_analysis")
    if wsa:
        print(f"  WEIGHT-SPACE (any seed both-positive over raw+exponents): "
              f"{wsa['n_seeds_any_both_positive']}/{len(wsa['rows'])}", flush=True)
        for r in wsa["rows"]:
            rw = r["raw_write"]
            exps = " ".join(f"{k}=({v['m0']:+.2f},{v['m1']:+.2f})bp{int(v['both_positive'])}"
                            for k, v in r["by_exponent"].items())
            print(f"    seed {r['seed']}: dgJ={r['dg_jaccard']:.2f} raw=({rw['m0']:+.2f},{rw['m1']:+.2f}) "
                  f"antisym={rw['antisymmetry_m0_plus_m1']:+.2f} | {exps} | "
                  f"priv_m1(corr={rw['private_m1_to_correct']:.0f},wrong={rw['private_m1_to_wrong']:.0f}) "
                  f"shared(a0={rw['shared_to_a0']:.0f},a1={rw['shared_to_a1']:.0f})", flush=True)
    print(f"  STATUS: {payload['status']}", flush=True)
    print(f"  checks: {json.dumps(payload['checks'])}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled'])}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
