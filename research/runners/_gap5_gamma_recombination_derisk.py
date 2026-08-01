"""gap#5 RANK 3 — imaginative/recombinative replay via GAMMA-WTA phase-organized sampling (NO sim/ edit).

The RANK 3 research gate (finding `2026-07-22-gap5-RANK3-imagination-recombinative-replay-research-gate.md`) characterised
a BOUNDARY: BOTH direct-composition methods for novel recombination at a shared branch node (A->B->C + X->B->Y) FAILED
-- spontaneous replay reactivates WITHOUT ordered transitions, and cue-driven replay BROADLY CO-IGNITES the connected
component instead of traversing one branch. The gate's named NEXT METHOD was **theta/gamma phase-organized replay**:
separate the assemblies IN TIME so the shared node B cannot co-ignite both successors, and SAMPLE one successor per gamma
cycle. That mechanism -- gamma-WTA + post-fire silence -- is ALREADY VALIDATED for RANK 2 forward order
(`_gap5_gamma_wta_replay_derisk.py`, 3/3 GO, `2026-07-22-gap5-gamma-WTA-timing-fixes-replay-order-cheap-GO.md`). This
runner applies it to the shared-node topology -- the deferred RANK 3 next-arc, now cheap because the timing primitive works.

Method (reuses the proven primitives, no new sim/):
  1. Encode the shared-node topology (SHARED_EDGES, 5 assemblies, B=1 shared) with the RANK 1 (bistable within) + RANK 2
     (forward BTSP chain via additive `chain_edges`) recipe; extract the 5x5 between-assembly transition matrix W with the
     SAME `_extract_W` that produced the RANK 2 gamma GO.
  2. Gamma-organized replay: start at a predecessor (A=0 or X=3); each gamma cycle drive[j] = W[cur][j] + noise, the
     winner fires, post-fire silence removes it. The SHARED node B has TWO learned successors (C, Y); which one wins is
     SAMPLED by the noise -> the replay traverses A->B->C (stored) OR A->B->Y (RECOMBINED) on different trials. Cueing X
     symmetrically gives X->B->Y (stored) or X->B->C (recombined).
  3. recomb_frac = P(B exits to the OTHER chain's successor | B reached, exit is a learned successor). A genuine branch
     samples BOTH successors (0 < recomb_frac < 1); a degenerate one is stuck on one.

Anti-cheats (the RANK 3 gate's mandated suite):
  - NO-SHARED (A->B->C + X->D->Y, B!=D): B has ONE successor and X never reaches B -> recomb must vanish (~0).
  - CONSISTENCY: B must exit ONLY to a learned successor (C or Y), never a random assembly -- imagination is CONSISTENT
    recombination, not fantasy. Report learned_exit_frac; it must stay high in MAIN and COLLAPSE under NO-ENCODE/SCRAMBLE.
  - NO-ENCODE: W at init, no learned edges -> exits are not to learned successors (consistency collapses).
  - SCRAMBLE (per-seed off-diagonal shuffle): destroys the learned B->{C,Y} structure -> exits stop landing on the true
    successors (consistency collapses); load-bearing that the recombination rides LEARNED weights, not the topology alone.
  - NO-NOISE (noise=0): deterministic argmax -> B takes its single strongest successor every trial -> NO sampling
    (branch is degenerate). Reported: the STOCHASTIC generative branching requires the noise.

numpy-deterministic (the transition-ORDER metric is GPU-non-deterministic -- RANK 2 lesson). SIM_BACKEND=numpy default.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")   # deterministic; the encode is small
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json

import numpy as np

from sim.backend import get_backend  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import SEQ_CFG, _prepare_sequence  # noqa: E402
from research.runners._gap5_gamma_wta_replay_derisk import _extract_W  # noqa: E402

# 5 assemblies A=0 B=1(shared) C=2 X=3 Y=4; two stored chains A->B->C, X->B->Y (B shared).
SHARED_EDGES = [(0, 1), (1, 2), (3, 1), (1, 4)]
# NO-SHARED control: a 6th assembly D=5 replaces B in the second chain -> X->D->Y, NO branch node.
NOSHARE_EDGES = [(0, 1), (1, 2), (3, 5), (5, 4)]
B_IDX = 1
PREDS = (0, 3)                 # A, X
STORED_SUCC = {0: 2, 3: 4}     # A->C, X->Y  (stored whole)
RECOMB_SUCC = {0: 4, 3: 2}     # A->Y, X->C  (novel recombination)
SUCC_SET = (2, 4)              # learned successors C, Y


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


def _gamma_walk_exit(W, cue, rng, noise, n_read):
    """One gamma-organized replay from predecessor `cue`. Post-fire silence (self-avoidance) + gamma-advance.
    Returns (reached_B, exit_assembly): the assembly that fires immediately AFTER B, or (False, -1) if B not reached."""
    Wm = W.copy()
    np.fill_diagonal(Wm, 0.0)   # the within-attractor is the HOLD; the gamma cycle drives the PUSH
    fired = [cue]
    cur = cue
    for _ in range(n_read):
        drive = Wm[cur] + rng.normal(0.0, noise, W.shape[0])
        drive[cur] = -1e18
        for f in fired:
            drive[f] = -1e18    # post-fire silence: an already-fired assembly cannot re-win
        nxt = int(np.argmax(drive))
        if cur == B_IDX:
            return True, nxt    # the step immediately after B is the branch decision
        fired.append(nxt)
        cur = nxt
    return False, -1


def _branch_stats(W, seed, noise, n_trials, n_read=4, scramble=False):
    """Sample gamma walks from BOTH predecessors; classify each B-exit as stored / recomb / other (unlearned)."""
    off = ~np.eye(W.shape[0], dtype=bool)
    rng = np.random.default_rng(int(seed) * 7919 + (100000 if scramble else 0))
    Wt = W
    if scramble:
        Wt = W.copy()
        vals = Wt[off].copy()
        rng.shuffle(vals)
        Wt[off] = vals
    n_stored = n_recomb = n_other = n_reachB = n_total = 0
    # per-cue exit-to-successor tallies (for branch degeneracy diagnostics)
    per_cue = {c: {"stored": 0, "recomb": 0, "other": 0} for c in PREDS}
    for cue in PREDS:
        for _ in range(n_trials):
            n_total += 1
            reached, exit_a = _gamma_walk_exit(Wt, cue, rng, noise, n_read)
            if not reached:
                continue
            n_reachB += 1
            if exit_a == STORED_SUCC[cue]:
                n_stored += 1; per_cue[cue]["stored"] += 1
            elif exit_a == RECOMB_SUCC[cue]:
                n_recomb += 1; per_cue[cue]["recomb"] += 1
            else:
                n_other += 1; per_cue[cue]["other"] += 1
    learned = n_stored + n_recomb
    return dict(
        n_stored=n_stored, n_recomb=n_recomb, n_other=n_other, n_reachB=n_reachB, n_total=n_total,
        reachB_frac=n_reachB / max(1, n_total),
        learned_exit_frac=learned / max(1, n_reachB),
        recomb_frac=(n_recomb / learned) if learned else 0.0,
        per_cue=per_cue,
    )


def _diff_stats(W):
    """Load-bearing DIAGNOSTIC: at the shared hub B, are the LEARNED successors (C,Y) differentiated from the UNLEARNED
    out-edges (X,A)? If learned ~= unlearned, the extracted mean transition matrix carries NO successor signal for the
    gamma argmax to ride -> the metric can only read the geometric chance (2 of 3 remaining candidates are successors)."""
    learned = float(np.mean([W[B_IDX, 2], W[B_IDX, 4]]))     # B->C, B->Y (the two stored successors)
    unlearned = float(np.mean([W[B_IDX, 3], W[B_IDX, 0]]))   # B->X, B->A (no learned out-edge from B)
    return dict(w_learned_succ=learned, w_unlearned_out=unlearned,
                diff=learned - unlearned, ratio=(learned / unlearned if unlearned else float("inf")))


def one_seed(seed, a):
    _, backend = get_backend()
    # --- MAIN: shared-node topology ---
    prep = _prepare_sequence(seed, _make_cfg(a, SHARED_EDGES, 5))
    W = _extract_W(prep, 5)
    diff = _diff_stats(W)
    main = _branch_stats(W, seed, a.noise, a.n_trials, a.n_read)
    nonoise = _branch_stats(W, seed, 0.0, a.n_trials, a.n_read)          # NO-NOISE acid
    scram = _branch_stats(W, seed, a.noise, a.n_trials, a.n_read, scramble=True)  # SCRAMBLE

    # --- NO-ENCODE control (init weights, no learned chain) ---
    prep_ne = _prepare_sequence(seed, _make_cfg(a, SHARED_EDGES, 5), do_encode=False)
    Wne = _extract_W(prep_ne, 5)
    noenc = _branch_stats(Wne, seed, a.noise, a.n_trials, a.n_read)

    # --- NO-SHARED control (A->B->C, X->D->Y; B!=D) : score on the 6x6 W, recomb must vanish ---
    prep_ns = _prepare_sequence(seed, _make_cfg(a, NOSHARE_EDGES, 6))
    Wns = _extract_W(prep_ns, 6)
    noshare = _branch_stats(Wns, seed, a.noise, a.n_trials, a.n_read)

    # GO: B is a GENUINE branch (samples BOTH successors) with CONSISTENT learned exits, and every control is clean.
    branches = (a.recomb_lo < main["recomb_frac"] < a.recomb_hi)      # samples both stored AND recombined
    consistent = main["learned_exit_frac"] >= a.consistency_thr        # exits land on learned successors
    reachesB = main["reachB_frac"] >= a.reach_thr                      # replay reliably reaches the branch node
    noshare_clean = noshare["recomb_frac"] <= a.control_recomb_max     # no branch -> no recombination
    noenc_collapse = noenc["learned_exit_frac"] <= a.collapse_thr      # no learning -> exits not to learned succ
    scram_collapse = scram["learned_exit_frac"] <= a.collapse_thr      # scrambled -> structure gone
    go = bool(branches and consistent and reachesB and noshare_clean and noenc_collapse and scram_collapse)

    wB_C, wB_Y = float(W[B_IDX, 2]), float(W[B_IDX, 4])
    print(f"  [seed {seed}] MAIN recomb_frac={main['recomb_frac']:.3f} (stored={main['n_stored']} recomb={main['n_recomb']} "
          f"other={main['n_other']}) learned_exit={main['learned_exit_frac']:.3f} reachB={main['reachB_frac']:.3f} "
          f"| DIFF learned_succ={diff['w_learned_succ']:.1f} vs unlearned_out={diff['w_unlearned_out']:.1f} "
          f"(ratio {diff['ratio']:.2f}) | NO-SHARED recomb={noshare['recomb_frac']:.3f} "
          f"| NO-NOISE recomb={nonoise['recomb_frac']:.3f} | NO-ENCODE learned={noenc['learned_exit_frac']:.3f} "
          f"| SCRAMBLE learned={scram['learned_exit_frac']:.3f} => {'RECOMB-GO' if go else 'no (chance=0.667)'}")
    return dict(seed=seed, backend=backend, main=main, nonoise=nonoise, scramble=scram, noenc=noenc, noshare=noshare,
                wB_C=wB_C, wB_Y=wB_Y, diff=diff, go=go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000)
    ap.add_argument("--within-events", type=int, default=30)
    ap.add_argument("--within-refresh", type=int, default=8)
    ap.add_argument("--chain-fwd", type=int, default=24)
    ap.add_argument("--noise", type=float, default=8.0, help="gamma-cycle drive noise std (comparable to the learned asym)")
    ap.add_argument("--n-trials", type=int, default=400, help="gamma replays per predecessor")
    ap.add_argument("--n-read", type=int, default=4, help="gamma cycles per replay (enough to reach B and read its exit)")
    ap.add_argument("--recomb-lo", type=float, default=0.05)
    ap.add_argument("--recomb-hi", type=float, default=0.95)
    ap.add_argument("--consistency-thr", type=float, default=0.90)
    ap.add_argument("--reach-thr", type=float, default=0.90)
    ap.add_argument("--control-recomb-max", type=float, default=0.02)
    ap.add_argument("--collapse-thr", type=float, default=0.60)
    ap.add_argument("--out", default="research/findings/raw/gap5_r4/gamma_recombination.json")
    a = ap.parse_args()
    _, backend = get_backend()
    print(f"[gap5-gamma-recomb] RANK3 gamma-WTA phase-organized recombination A->B->C + X->B->Y (B shared), "
          f"noise={a.noise} trials={a.n_trials} seeds={a.seeds} backend={backend}")
    per = [one_seed(s, a) for s in a.seeds]
    n_go = sum(p["go"] for p in per)
    mR = float(np.mean([p["main"]["recomb_frac"] for p in per]))
    mL = float(np.mean([p["main"]["learned_exit_frac"] for p in per]))
    mNS = float(np.mean([p["noshare"]["recomb_frac"] for p in per]))
    print(f"[gap5-gamma-recomb] VERDICT: {n_go}/{len(per)} seeds -- gamma-organized replay generates NOVEL recombination "
          f"(recomb_frac {mR:.3f}, learned_exit {mL:.3f}) with NO-SHARED clean ({mNS:.3f}) + NO-ENCODE/SCRAMBLE collapse. "
          f"{'GO: phase-organized replay imagines consistent recombinations.' if n_go == len(per) else 'partial/negative.'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, n_go=n_go, noise=a.noise, per=per), f, indent=2)


if __name__ == "__main__":
    main()
