"""gap#5 RANK 3 (imagination) — FULL PHASE-GATED SPIKING recombination replay at a shared hub, reading the POTENTIATED
per-synapse SUBSET (not the extracted mean transition matrix).

WHY THIS RUNNER EXISTS (the named next method).
`_gap5_gamma_recombination_derisk.py` applied gamma-WTA timing to the shared-hub topology (A->B->C + X->B->Y, B shared)
over the EXTRACTED MEAN between-assembly transition matrix `_extract_W` -> 0/6, at the 2/3 geometric chance
(finding 2026-08-01-gap5-RANK3-gamma-organized-recombination-extracted-matrix-proxy-sits-at-chance.md). The cause was
LOCATED: at the hub the mean B->{C,Y} learned-successor weight is INDISTINGUISHABLE from the unlearned out-edges
B->{X,A} (ratio ~1.14) because the BTSP coincidence encode SATURATES at the hub and the MEAN averages the potentiated
SUBSET away. The named next method: a full SPIKING read whose postsynaptic threshold rides the actual potentiated-synapse
subset -- if that subset targets SPECIFIC successor cells (concentration), a spiking k-WTA read fires those cells while
the diffuse unlearned out-edges stay sub-threshold, recovering the discrimination the mean lost.

NOTE ON THE FILENAME: the finding named the vehicle `_gap5_spiking_gamma_replay_derisk.py`, but that file ALREADY EXISTS
(the RANK 2 forward-ORDER spiking runner). This is a DISTINCT file for the RANK 3 shared-hub RECOMBINATION on spikes, so
the RANK 2 runner is not clobbered.

WHAT IT DOES (NO `sim/` edit; reuse-by-import).
  1. Build+encode the shared-hub substrate with `_prepare_sequence(..., chain_edges=SHARED_EDGES)` -- the SAME encode the
     matrix runner used (so the ONLY change is the READ: spiking-per-cell instead of mean-matrix argmax).
  2. INSTRUMENT (the finding's mandated diagnostic -- "the instrument is part of the emulation"): from the REAL
     cp_connections, compute the per-post-cell summed input each candidate successor receives FROM B's cells. Report the
     MEAN (== the proxy, reproduces ratio~1.14) AND the TOP-K / MAX per-cell input (the concentration a spiking threshold
     reads). If learned/unlearned TOP-K ratio >> 1 while the mean ratio ~= 1, the discrimination IS in the potentiated
     subset (the mean lost it, the spiking read can recover it). If TOP-K ratio ~= 1 too, the ENCODE is non-selective.
     This answers the task's "diagnose WHY" if the spiking read also sits at chance.
  3. FULL SPIKING CUED GAMMA REPLAY: cue a predecessor (A or X) with a strong pulse; each theta cycle is one walk. The
     cue ignites A; post-fire self-avoidance silences A (the gamma reset); A->B potentiated synapses ignite B; B is
     silenced; B->{C,Y} potentiated synapses drive the successors and the substrate's own feedback inhibition + an
     optional de Almeida-Idiart-Lisman E%-max theta-ramp make it a per-cycle WINNER-TAKE-ALL; weak background OU noise
     breaks the C-vs-Y tie stochastically across cycles -> the walk traverses A->B->C (stored) OR A->B->Y (RECOMBINED) on
     different cycles. The B-EXIT = the first successor to fire after B. Classify stored / recomb / other over many cycles.

METRICS: reachB_frac (walk reaches the hub), learned_exit_frac (B exits to a LEARNED successor C/Y vs an unlearned one --
the discrimination the mean lost; chance = 2/3), recomb_frac (of learned exits, the OTHER chain's successor -- a genuine
branch samples both, 0<recomb<1), co_ignite_frac (both successors fired -- the co-ignition boundary the WTA must break).

ANTI-CHEATS (the RANK 3 gate's mandated suite):
  - NO-SHARED (A->B->C, X->D->Y; B!=D): X never reaches B -> recomb must vanish (~0).
  - NO-ENCODE (init weights, no chain): learned_exit collapses to chance (no learned successors to fire).
  - SCRAMBLE (shuffle the between-assembly edge weights): the learned B->{C,Y} structure gone -> learned_exit collapses.
  - NO-NOISE (ou_sigma=0): deterministic -> no C-vs-Y sampling (degenerate branch); the stochastic branching REQUIRES noise.

GO: MAIN learned_exit_frac > 2/3 + margin (spiking read discriminates learned successors) AND recomb_frac in (lo,hi)
(genuine both-path branch) AND reachB high; NO-SHARED recomb ~0; NO-ENCODE + SCRAMBLE learned_exit collapse to chance.

Backend: SIM_BACKEND=numpy default (deterministic, matches the sibling spiking runners; the encode is small). Override
SIM_BACKEND=cupy for speed under concurrent load (the recombination is SAMPLED, so byte-determinism is not required; the
6-seed aggregate is robust to per-run nondeterminism as long as cfg.seed is set).

SMOKE:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_spiking_gamma_recombination_derisk \
            --seeds 42 --n-cycles 60 --smoke
FULL:   SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_spiking_gamma_recombination_derisk \
            --seeds 42 43 44 100 101 102 --n-cycles 200 \
            --out research/findings/raw/gap5_r4/spk_gamma_recomb_6seed.json
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")   # deterministic default; the encode is small
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import (  # noqa: E402
    SEQ_CFG, _prepare_sequence, _scramble_between_weights,
)
from research.runners._gap5_spontaneous_reactivation_derisk import (  # noqa: E402
    _extract_ca3ca3_vec, _hard_silence, _configure_ou,
)

# 5 assemblies A=0 B=1(shared) C=2 X=3 Y=4; two stored chains A->B->C, X->B->Y (B shared).
SHARED_EDGES = [(0, 1), (1, 2), (3, 1), (1, 4)]
# NO-SHARED control: a 6th assembly D=5 replaces B in the second chain -> X->D->Y, NO branch node at B.
NOSHARE_EDGES = [(0, 1), (1, 2), (3, 5), (5, 4)]
B_IDX = 1
PREDS = (0, 3)                  # predecessors A, X
STORED_SUCC = {0: 2, 3: 4}      # A->C, X->Y  (stored whole)
RECOMB_SUCC = {0: 4, 3: 2}      # A->Y, X->C  (novel recombination)
SUCC_SET = (2, 4)               # learned successors C, Y
UNLEARNED_OUT = (0, 3)          # B's unlearned out-edges (to A, X)


def _make_cfg(a, edges, n_mem):
    cfg = dict(SEQ_CFG)
    cfg["n_ca3"] = int(a.n_ca3)
    cfg["n_mem"] = int(n_mem)
    cfg["within_events"] = int(a.within_events)
    cfg["within_refresh"] = int(a.within_refresh)
    cfg["chain_fwd"] = int(a.chain_fwd)
    cfg["chain_rev"] = 0
    cfg["chain_edges"] = edges
    cfg["rank1_encode"] = True
    cfg["overlap_draw"] = False
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
# INSTRUMENT: the per-cell / per-synapse CONCENTRATION diagnostic (the finding's "the instrument is part of the emulation")
# ----------------------------------------------------------------------------------------------------------------------
def _b_out_per_cell_input(prep):
    """Per-CA3-cell summed synaptic weight FROM the hub B's cells, on the REAL cp_connections (the potentiated subset,
    NOT the mean). Returns inp_full[local_ca3_pos] = sum over b in B of W[b -> this cell]."""
    bridge = prep["bridge"]
    ca3_idx = list(bridge.region_manager.indices("ca3"))
    flat_h, pre_l, post_l = _extract_ca3ca3_vec(bridge, ca3_idx, to_host)
    d = np.asarray(to_host(bridge.cp_connections.data))
    asm_of = np.full(len(ca3_idx), -1, dtype=np.int64)
    for m, al in enumerate(prep["assemblies_local"]):
        asm_of[np.asarray(al, dtype=np.int64)] = m
    mask_preB = asm_of[pre_l] == B_IDX
    inp_full = np.zeros(len(ca3_idx), dtype=np.float64)
    np.add.at(inp_full, post_l[mask_preB], d[flat_h[mask_preB]])
    return inp_full


def _conc_stats(inp_full, prep, k_frac):
    """For each candidate successor Z, the concentration of B's drive onto Z's cells: mean (== the mean-matrix proxy),
    top-k mean, and max. Then the LEARNED (C,Y) vs UNLEARNED (A,X) discrimination at each read level. If mean_ratio ~= 1
    but topk_ratio >> 1, the potentiated SUBSET carries the signal the mean lost -> a spiking threshold read can recover it."""
    al = prep["assemblies_local"]

    def _z(z):
        v = inp_full[al[z]]
        v = np.sort(v)[::-1]
        k = max(1, int(np.ceil(k_frac * len(v))))
        return dict(mean=float(v.mean()), topk=float(v[:k].mean()), mx=float(v[0]))

    per = {z: _z(z) for z in (0, 2, 3, 4)}   # A, C, X, Y  (B's candidate out-targets)
    learned_mean = float(np.mean([per[2]["mean"], per[4]["mean"]]))
    unlearn_mean = float(np.mean([per[0]["mean"], per[3]["mean"]]))
    learned_topk = float(np.mean([per[2]["topk"], per[4]["topk"]]))
    unlearn_topk = float(np.mean([per[0]["topk"], per[3]["topk"]]))
    learned_mx = float(np.mean([per[2]["mx"], per[4]["mx"]]))
    unlearn_mx = float(np.mean([per[0]["mx"], per[3]["mx"]]))
    return dict(
        per_z=per,
        mean_learned=learned_mean, mean_unlearned=unlearn_mean,
        mean_ratio=(learned_mean / unlearn_mean if unlearn_mean else float("inf")),
        topk_learned=learned_topk, topk_unlearned=unlearn_topk,
        topk_ratio=(learned_topk / unlearn_topk if unlearn_topk else float("inf")),
        max_learned=learned_mx, max_unlearned=unlearn_mx,
        max_ratio=(learned_mx / unlearn_mx if unlearn_mx else float("inf")),
    )


# ----------------------------------------------------------------------------------------------------------------------
# FULL SPIKING CUED GAMMA REPLAY
# ----------------------------------------------------------------------------------------------------------------------
def _one_theta_cycle(bridge, cp, cue_idx, asm_glob, asm_sizes, assemblies_local, ca3_arr_host, exc_dev_all, a):
    """One theta cycle = one cued walk. Cue the predecessor for cue_steps; each step apply the E%-max theta-ramp global
    inhibition (moving threshold, high at onset), post-fire self-avoidance on already-fired assemblies, and let the
    substrate's own feedback inhibition + background OU noise resolve the per-cycle WTA. Returns {asm_idx: onset_step}."""
    n_asm = len(assemblies_local)
    fired_at = {}
    onset = {}
    afrac = np.zeros((a.theta_period, n_asm), dtype=np.float64)
    cue_glob = asm_glob[cue_idx]
    for t in range(a.theta_period):
        bridge.cp_external_input_current[:] = 0.0
        if t < a.cue_steps:
            bridge.cp_external_input_current[cue_glob] += a.cue_pa
        if a.ramp_hi > 0.0:
            # de Almeida-Idiart-Lisman E%-max moving threshold: global CA3-exc inhibition high at theta onset, ramps
            # down -> the most-excited (concentrated-input) successor crosses FIRST, one per cycle (the WTA the mean lacked).
            tp = t / a.theta_period
            ramp = a.ramp_hi - (a.ramp_hi - a.ramp_lo) * tp
            bridge.cp_external_input_current[exc_dev_all] += -float(ramp)
        for k, t0 in fired_at.items():
            if t >= t0 + a.silence_delay:                       # gamma reset: silence the already-fired assembly (self-avoid)
                bridge.cp_external_input_current[asm_glob[k]] += a.inhib_pa
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host]
        for k, alk in enumerate(assemblies_local):
            afrac[t, k] = fs[alk].sum() / asm_sizes[k]
        lo = max(0, t - a.window + 1)
        for k in range(n_asm):
            # windowed SUM of per-step active fractions (== _cue_and_measure's _smooth-convolve/size convention: spikes
            # per cell over the last `window` steps). Using .sum() not .mean() -- the reactivation here is sparse/weak
            # (peak ~0.06-0.20 spikes/cell/window), so a per-step MEAN is ~window x too small and never crosses threshold.
            if k not in fired_at and afrac[lo:t + 1, k].sum() >= a.fire_thresh:
                fired_at[k] = t
                onset[k] = t
    return onset


def _b_exit(onset, cue_idx):
    """The single first assembly to fire AFTER B (excluding the cue and B). None = B not reached; -1 = reached B, no exit."""
    if B_IDX not in onset:
        return None
    tB = onset[B_IDX]
    cands = sorted((onset[k], k) for k in onset if k not in (B_IDX, cue_idx) and onset[k] > tB)
    return cands[0][1] if cands else -1


def _cued_gamma_replay(prep, a, seed, ou_sigma, n_cycles):
    """Run n_cycles theta cycles per predecessor (A, X); collect + classify each B-exit. A short hard-silence between
    cycles gives a clean theta trough (un-latches the bistable within-attractors) while the OU stream keeps advancing
    (-> different noise each cycle -> stochastic C-vs-Y sampling)."""
    cp, _ = get_backend()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    _hard_silence(bridge, settle=20)
    _configure_ou(bridge, (ou_sigma if ou_sigma > 0 else None), seed)
    ca3_arr_host = prep["ca3_arr_host"]
    assemblies_local = prep["assemblies_local"]
    asm_glob = [cp.asarray(ca3_arr_host[np.asarray(al, dtype=np.int64)], dtype=cp.int64) for al in assemblies_local]
    asm_sizes = [max(1, len(al)) for al in assemblies_local]
    exc_dev_all = cp.asarray(ca3_arr_host[prep["ca3_exc_local"]], dtype=cp.int64)

    tally = {c: dict(stored=0, recomb=0, other=0, reachB=0, no_exit=0, co_ignite=0, total=0) for c in PREDS}
    for cue in PREDS:
        for _ in range(n_cycles):
            tally[cue]["total"] += 1
            onset = _one_theta_cycle(bridge, cp, cue, asm_glob, asm_sizes, assemblies_local,
                                     ca3_arr_host, exc_dev_all, a)
            # co-ignition instrument: did BOTH successors fire this cycle?
            if all(z in onset for z in SUCC_SET):
                tally[cue]["co_ignite"] += 1
            ex = _b_exit(onset, cue)
            if ex is None:
                continue
            tally[cue]["reachB"] += 1
            if ex == -1:
                tally[cue]["no_exit"] += 1
            elif ex == STORED_SUCC[cue]:
                tally[cue]["stored"] += 1
            elif ex == RECOMB_SUCC[cue]:
                tally[cue]["recomb"] += 1
            else:
                tally[cue]["other"] += 1
            _hard_silence(bridge, settle=a.intercycle_silence)   # theta trough (OU stream continues across cycles)

    # aggregate over both predecessors
    stored = sum(tally[c]["stored"] for c in PREDS)
    recomb = sum(tally[c]["recomb"] for c in PREDS)
    other = sum(tally[c]["other"] for c in PREDS)
    reachB = sum(tally[c]["reachB"] for c in PREDS)
    no_exit = sum(tally[c]["no_exit"] for c in PREDS)
    co_ig = sum(tally[c]["co_ignite"] for c in PREDS)
    total = sum(tally[c]["total"] for c in PREDS)
    learned = stored + recomb
    exits = learned + other                                     # B-reached cycles that produced an exit
    return dict(
        per_cue=tally, stored=stored, recomb=recomb, other=other, reachB=reachB, no_exit=no_exit,
        co_ignite=co_ig, total=total,
        reachB_frac=reachB / max(1, total),
        learned_exit_frac=learned / max(1, exits),
        recomb_frac=(recomb / learned) if learned else 0.0,
        co_ignite_frac=co_ig / max(1, reachB),
    )


def one_seed(seed, a):
    _, backend = get_backend()
    ncy = a.n_cycles

    # ---- MAIN: shared-hub topology; INSTRUMENT then full spiking cued gamma replay ----
    prep = _prepare_sequence(seed, _make_cfg(a, SHARED_EDGES, 5))
    inp_full = _b_out_per_cell_input(prep)
    conc = _conc_stats(inp_full, prep, a.k_frac)
    main = _cued_gamma_replay(prep, a, seed, a.ou_sigma, ncy)
    nonoise = _cued_gamma_replay(prep, a, seed, 0.0, max(20, ncy // 3))     # NO-NOISE acid (deterministic; fewer cycles)

    # ---- SCRAMBLE: shuffle between-assembly edges -> the learned B->{C,Y} structure destroyed ----
    prep_sc = _prepare_sequence(seed, _make_cfg(a, SHARED_EDGES, 5))
    _scramble_between_weights(prep_sc, seed)
    scram = _cued_gamma_replay(prep_sc, a, seed, a.ou_sigma, max(30, ncy // 2))

    # ---- NO-ENCODE: init weights, no chain -> no learned successors ----
    prep_ne = _prepare_sequence(seed, _make_cfg(a, SHARED_EDGES, 5), do_encode=False)
    noenc = _cued_gamma_replay(prep_ne, a, seed, a.ou_sigma, max(30, ncy // 2))

    # ---- NO-SHARED: A->B->C, X->D->Y (B!=D) -> X never reaches B -> recomb must vanish ----
    prep_ns = _prepare_sequence(seed, _make_cfg(a, NOSHARE_EDGES, 6))
    noshare = _cued_gamma_replay(prep_ns, a, seed, a.ou_sigma, ncy)

    chance = 2.0 / 3.0
    branches = (a.recomb_lo < main["recomb_frac"] < a.recomb_hi)
    discriminates = main["learned_exit_frac"] >= chance + a.learned_margin
    reaches = main["reachB_frac"] >= a.reach_thr
    noshare_clean = noshare["recomb_frac"] <= a.control_recomb_max
    noenc_collapse = noenc["learned_exit_frac"] <= chance + a.collapse_margin
    scram_collapse = scram["learned_exit_frac"] <= chance + a.collapse_margin
    go = bool(branches and discriminates and reaches and noshare_clean and noenc_collapse and scram_collapse)

    print(f"  [seed {seed}] INSTRUMENT B-out: mean L/U={conc['mean_learned']:.1f}/{conc['mean_unlearned']:.1f} "
          f"(ratio {conc['mean_ratio']:.2f}) | topk L/U={conc['topk_learned']:.1f}/{conc['topk_unlearned']:.1f} "
          f"(ratio {conc['topk_ratio']:.2f}) | max ratio {conc['max_ratio']:.2f}")
    print(f"  [seed {seed}] MAIN reachB={main['reachB_frac']:.3f} learned_exit={main['learned_exit_frac']:.3f} "
          f"(chance {chance:.3f}) recomb={main['recomb_frac']:.3f} co_ignite={main['co_ignite_frac']:.3f} "
          f"(stored={main['stored']} recomb={main['recomb']} other={main['other']}) | NO-SHARED recomb={noshare['recomb_frac']:.3f} "
          f"| NO-ENCODE learned={noenc['learned_exit_frac']:.3f} | SCRAMBLE learned={scram['learned_exit_frac']:.3f} "
          f"| NO-NOISE recomb={nonoise['recomb_frac']:.3f} => {'SPK-RECOMB-GO' if go else 'no (chance=0.667)'}")
    return dict(seed=seed, backend=backend, conc=conc, main=main, nonoise=nonoise, scramble=scram,
                noenc=noenc, noshare=noshare, go=go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000)
    ap.add_argument("--within-events", type=int, default=30)
    ap.add_argument("--within-refresh", type=int, default=8)
    ap.add_argument("--chain-fwd", type=int, default=24)
    ap.add_argument("--n-cycles", type=int, default=200, help="theta cycles per predecessor (each cycle = one cued walk)")
    # --- theta/gamma cued-walk timing ---
    ap.add_argument("--theta-period", type=int, default=400, help="steps per theta cycle (one walk: cue + propagate)")
    ap.add_argument("--cue-steps", type=int, default=150, help="steps the predecessor cue is injected at cycle onset")
    ap.add_argument("--cue-pa", type=float, default=1000.0, help="predecessor cue drive (RANK 1 completion cue ~700-1000 pA)")
    ap.add_argument("--fire-thresh", type=float, default=0.06, help="windowed-SUM per-assembly active fraction to count as FIRED (the reactivation here is sparse/weak, peak ~0.06-0.20 spikes/cell/window)")
    ap.add_argument("--inhib-pa", type=float, default=-800.0, help="post-fire self-avoidance current (the gamma reset)")
    ap.add_argument("--silence-delay", type=int, default=6, help="steps after onset before an assembly is silenced (let the burst be detected)")
    ap.add_argument("--ramp-hi", type=float, default=0.0, help="E%-max theta-ramp inhibition at cycle onset (0 = off; rely on the substrate's own feedback WTA)")
    ap.add_argument("--ramp-lo", type=float, default=0.0, help="E%-max theta-ramp inhibition at cycle end")
    ap.add_argument("--ou-sigma", type=float, default=40.0, help="weak background OU noise std (pA) -- breaks the C-vs-Y tie for stochastic sampling")
    ap.add_argument("--intercycle-silence", type=int, default=6, help="hard-silence steps between cycles (theta trough)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--k-frac", type=float, default=0.15, help="top-k fraction of successor cells for the concentration instrument")
    # --- GO thresholds ---
    ap.add_argument("--learned-margin", type=float, default=0.10, help="MAIN learned_exit must exceed chance(0.667)+margin")
    ap.add_argument("--collapse-margin", type=float, default=0.05, help="NO-ENCODE/SCRAMBLE learned_exit must be <= chance+margin")
    ap.add_argument("--recomb-lo", type=float, default=0.10)
    ap.add_argument("--recomb-hi", type=float, default=0.90)
    ap.add_argument("--reach-thr", type=float, default=0.50)
    ap.add_argument("--control-recomb-max", type=float, default=0.10)
    ap.add_argument("--smoke", action="store_true", help="reduce cycles for a fast smoke")
    ap.add_argument("--out", default="research/findings/raw/gap5_r4/spk_gamma_recomb.json")
    a = ap.parse_args()
    if a.smoke and a.n_cycles > 60:
        a.n_cycles = 60
    _, backend = get_backend()
    print(f"[gap5-spk-recomb] RANK3 FULL-SPIKING gamma-gated recombination A->B->C + X->B->Y (B shared), "
          f"cue={a.cue_pa}pA x{a.cue_steps} theta={a.theta_period} ou={a.ou_sigma} ramp={a.ramp_hi} "
          f"n_cycles={a.n_cycles} seeds={a.seeds} backend={backend}")
    per = [one_seed(s, a) for s in a.seeds]
    n_go = sum(p["go"] for p in per)
    mL = float(np.mean([p["main"]["learned_exit_frac"] for p in per]))
    mR = float(np.mean([p["main"]["recomb_frac"] for p in per]))
    mTKR = float(np.mean([p["conc"]["topk_ratio"] for p in per]))
    mMR = float(np.mean([p["conc"]["mean_ratio"] for p in per]))
    print(f"[gap5-spk-recomb] VERDICT: {n_go}/{len(per)} seeds -- spiking read learned_exit {mL:.3f} (chance 0.667), "
          f"recomb {mR:.3f} | INSTRUMENT topk_ratio {mTKR:.2f} vs mean_ratio {mMR:.2f}. "
          f"{'GO: the spiking per-cell read recovers the discrimination the mean lost.' if n_go == len(per) else 'partial/negative -- read the INSTRUMENT ratios to diagnose (encode-nonselective vs read-loses-it).'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, n_go=n_go, args=vars(a), per=per), f, indent=2)


if __name__ == "__main__":
    main()
