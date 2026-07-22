"""gap#5 RANK 3 — imaginative/generative replay: NOVEL recombination at a SHARED branch node.

Mechanism (research gate 2026-07-22-gap5-RANK3-imagination-recombinative-replay-research-gate.md): imagination =
novel-but-consistent recombination of LEARNED transitions at a shared state (Olafsdottir/Gupta shortcut replay;
Ecker-2022 CA3 model). Store TWO overlapping chains sharing a middle assembly B: A->B->C (assemblies 0->1->2) and
X->B->Y (3->1->4). During REST under weak non-specific noise (frozen plasticity, no cue), the CA3 branch point B can be
entered from either predecessor and exit to EITHER learned successor -> the replay sometimes traverses the NOVEL
recombination A->B->Y or X->B->C, a path never stored as a whole. That is generative/imaginative replay.

Composes the WORKING RANK 1 primitive (bistable within-attractor -> spontaneous reactivation, --rank1-encode) + the RANK
2 primitive (forward BTSP chain, now via the additive `chain_edges` branch topology). NO `sim/` edit. The order/transition
metric is GPU-non-deterministic (RANK 2 lesson) -> run on numpy (SIM_BACKEND=numpy) for reproducible claims.

Detection: for each detected replay event, per-assembly smoothed onset time; a TRANSITION triplet is pred(A=0/X=3)
-> B(1) -> succ(C=2/Y=4) with onset(pred) < onset(B) < onset(succ). WITHIN-chain = the stored successor (A->C, X->Y);
CROSS-chain = the recombination (A->Y, X->C). recomb_frac = cross / (within+cross).

Anti-cheats: NO-SHARED-NODE (A->B->C + X->D->Y, B!=D -> no branch -> cross must vanish); NO-NOISE acid; NO-ENCODE;
LEARNED-SUCCESS consistency (B exits only to a learned successor, never a random assembly).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")   # deterministic default for the order metric; override with SIM_BACKEND=cupy

import argparse
import json

import numpy as np

from sim.backend import get_backend  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import (  # noqa: E402
    SEQ_CFG, _prepare_sequence, _smooth, _event_windows,
)
from research.runners._gap5_spontaneous_reactivation_derisk import _rest_and_detect, _noise_label  # noqa: E402

# 5 assemblies A=0 B=1(shared) C=2 X=3 Y=4; the two stored chains:
SHARED_EDGES = [(0, 1), (1, 2), (3, 1), (1, 4)]      # A->B->C, X->B->Y (B shared)
# NO-SHARED control: a 6th assembly D=5 replaces B in the second chain -> X->D->Y, no branch node.
NOSHARE_EDGES = [(0, 1), (1, 2), (3, 5), (5, 4)]     # A->B->C, X->D->Y (B!=D)
PRED = (0, 3)                 # predecessors A, X
SUCC = (2, 4)                 # successors C, Y
STORED_SUCC = {0: 2, 3: 4}   # A->C, X->Y (stored)
RECOMB_SUCC = {0: 4, 3: 2}   # A->Y, X->C (novel recombination)
B_IDX = 1


def _onsets(F, assemblies_local, s, e, W, onset_frac):
    """{asm_idx: onset_step} for assemblies whose smoothed active-fraction crosses onset_frac in window [s,e)."""
    out = {}
    for k, A in enumerate(assemblies_local):
        if len(A) == 0:
            continue
        a_t = _smooth(F[s:e][:, A].sum(1).astype(float), W) / max(1, len(A))
        if a_t.size and float(a_t.max()) >= onset_frac:
            cross = np.nonzero(a_t >= onset_frac)[0]
            if cross.size:
                out[k] = int(cross[0])
    return out


def _count_transitions(F, assemblies_local, W, ev_floor, ev_k, onset_frac, min_len):
    """Count pred->B->succ transition triplets across replay events; classify within-chain vs cross-chain (recombination)
    and whether B exits to a LEARNED successor (any of SUCC) at all."""
    asize_ref = float(np.mean([max(1, len(a)) for a in assemblies_local]))
    events, _, _ = _event_windows(F, W=W, ev_floor=ev_floor, ev_k=ev_k, asize_ref=asize_ref)
    within = cross = b_active = b_to_learned = 0
    per_asm = [0] * len(assemblies_local)   # per-assembly activation count across events (diagnostic)
    for (s, e) in events:
        if e - s < min_len:
            continue
        ons = _onsets(F, assemblies_local, s, e, W, onset_frac)
        for k in ons:
            per_asm[k] += 1
        if B_IDX not in ons:
            continue
        b_active += 1
        tB = ons[B_IDX]
        preds = [p for p in PRED if p in ons and ons[p] < tB]
        succs = [c for c in SUCC if c in ons and ons[c] > tB]
        if succs:
            b_to_learned += 1
        for p in preds:
            for c in succs:
                if STORED_SUCC[p] == c:
                    within += 1
                else:
                    cross += 1
    tot = within + cross
    return dict(events=len(events), b_active=b_active, b_to_learned=b_to_learned, per_asm=per_asm,
                within=within, cross=cross, recomb_frac=(cross / tot if tot else 0.0), n_transitions=tot)


def _make_cfg(a, edges, n_mem):
    cfg = dict(SEQ_CFG)
    cfg["n_ca3"] = int(a.n_ca3)
    cfg["n_mem"] = int(n_mem)
    cfg["within_events"] = int(a.within_events)
    cfg["chain_fwd"] = int(a.chain_fwd)
    cfg["chain_rev"] = 0
    cfg["chain_edges"] = edges
    cfg["rank1_encode"] = True        # the RANK 2 within-reactivation fix (proven)
    cfg["within_refresh"] = int(a.within_refresh)   # restore within-attractor after the chain (RANK 2 recipe)
    cfg["overlap_draw"] = False
    return cfg


def one_seed(seed, a):
    xp, backend = get_backend()
    noise = ("poisson", a.poisson_rate, a.poisson_pa, a.poisson_dur)
    det = dict(W=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k, onset_frac=a.onset_frac, min_len=a.min_ev_len)

    # --- MAIN: shared-node topology ---
    prep = _prepare_sequence(seed, _make_cfg(a, SHARED_EDGES, n_mem=5))
    _, F = _rest_and_detect(prep, noise, a.rest_steps, seed, W=a.window, ev_floor=0.5, ev_k=a.ev_k, min_frac=0.30)
    main = _count_transitions(F, prep["assemblies_local"], **det)
    if getattr(a, "main_only", False):
        print(f"  [seed {seed}] DIAGNOSE MAIN: events={main['events']} per_asm(A,B,C,X,Y)={main['per_asm']} "
              f"b_active={main['b_active']} within={main['within']} cross={main['cross']} w_within(prep)={prep.get('w_within', 0):.1f}")
        return dict(seed=seed, backend=backend, main=main, go=False)

    # NO-NOISE acid (no spontaneous recombination without background)
    _, Fnn = _rest_and_detect(prep, ("none",), a.rest_steps, seed, W=a.window, ev_floor=0.5, ev_k=a.ev_k, min_frac=0.30)
    nonoise = _count_transitions(Fnn, prep["assemblies_local"], **det)

    # NO-ENCODE (weights at init -> no chain, no recombination)
    prep_ne = _prepare_sequence(seed, _make_cfg(a, SHARED_EDGES, n_mem=5), do_encode=False)
    _, Fne = _rest_and_detect(prep_ne, noise, a.rest_steps, seed, W=a.window, ev_floor=0.5, ev_k=a.ev_k, min_frac=0.30)
    noenc = _count_transitions(Fne, prep_ne["assemblies_local"], **det)

    # NO-SHARED-NODE control (A->B->C, X->D->Y; B!=D -> no branch -> cross must vanish)
    prep_ns = _prepare_sequence(seed, _make_cfg(a, NOSHARE_EDGES, n_mem=6))
    _, Fns = _rest_and_detect(prep_ns, noise, a.rest_steps, seed, W=a.window, ev_floor=0.5, ev_k=a.ev_k, min_frac=0.30)
    noshare = _count_transitions(Fns, prep_ns["assemblies_local"][:5], **det)  # score on A,B,C,X,Y only

    go = (main["cross"] >= 1 and main["recomb_frac"] > 0.0 and nonoise["n_transitions"] == 0
          and noenc["n_transitions"] == 0 and noshare["cross"] == 0)
    print(f"  [seed {seed}] MAIN: within={main['within']} CROSS(recomb)={main['cross']} "
          f"recomb_frac={main['recomb_frac']:.3f} b_active={main['b_active']} b_to_learned={main['b_to_learned']} "
          f"| NO-NOISE trans={nonoise['n_transitions']} NO-ENCODE trans={noenc['n_transitions']} "
          f"NO-SHARED cross={noshare['cross']} within={noshare['within']} => {'RECOMB' if go else 'no'}")
    return dict(seed=seed, backend=backend, main=main, nonoise=nonoise, noenc=noenc, noshare=noshare, go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000)
    ap.add_argument("--within-events", type=int, default=30)
    ap.add_argument("--within-refresh", type=int, default=8)
    ap.add_argument("--chain-fwd", type=int, default=24)
    ap.add_argument("--poisson-rate", type=float, default=0.015)
    ap.add_argument("--poisson-pa", type=float, default=1500.0)
    ap.add_argument("--poisson-dur", type=int, default=10)
    ap.add_argument("--rest-steps", type=int, default=1400)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.5)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--onset-frac", type=float, default=0.10)
    ap.add_argument("--min-ev-len", type=int, default=4)
    ap.add_argument("--main-only", action="store_true", help="FAST diagnostic: run MAIN only + report per-assembly activation counts (skip anti-cheats)")
    ap.add_argument("--out", default="research/findings/raw/gap5_r4/rank3_recombination.json")
    a = ap.parse_args()
    _, backend = get_backend()
    print(f"[gap5-recomb] RANK3 shared-node A->B->C + X->B->Y (B shared) noise={_noise_label(('poisson', a.poisson_rate, a.poisson_pa, a.poisson_dur))} "
          f"rest_steps={a.rest_steps} seeds={a.seeds} backend={backend}")
    per = [one_seed(s, a) for s in a.seeds]
    n_go = sum(p["go"] for p in per)
    print(f"[gap5-recomb] VERDICT: {n_go}/{len(per)} seeds show NOVEL recombination (cross-chain replay at the shared "
          f"branch node) with NO-NOISE/NO-ENCODE/NO-SHARED all clean")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, n_go=n_go, per=per), f, indent=2)


if __name__ == "__main__":
    main()
