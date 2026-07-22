"""gap#5 TIMING de-risk (cheapest-first, NO sim/ edit) — does a gamma-WTA + post-fire silence turn RANK 2's marginal
weight-only replay order into a RELIABLE forward order?

Motivation (2026-07-22 deep-research, finding `2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS...`):
RANK 2's forward-replay is only 4/6 (mean 0.806) because ONE pool of recurrent weights must both HOLD each assembly and
PUSH to the next -> a strong within-encode adds reverse links that swamp the marginal forward asymmetry. The biological
fix (Lisman-Idiart theta-gamma, de Almeida-Idiart-Lisman E%-max WTA) is a per-gamma-cycle winner-take-all whose reset
SILENCES the just-fired assembly (self-avoidance), DECOUPLING hold from push: the forward chain then only has to make A
slightly more excited than B, and self-avoidance forbids going backward to an already-fired assembly.

This isolates that claim in MINUTES on the REAL learned weights, before any bridge build:
1. Run RANK 2's proven encode (n_mem, --rank1-encode + within-refresh) -> extract the n_mem x n_mem BETWEEN-assembly
   transition weight matrix W[i][j] (mean learned weight of edges from assembly i to assembly j).
2. Replay = a gamma-cycle sequence of winners starting from assembly 0. Each cycle the drive to j = W[current][j] + noise;
   the winner fires; a gamma reset then silences the current assembly (it cannot immediately re-win).
   - ARM A (WEIGHT-ONLY, the RANK 2 analogue): no self-avoidance -> a later cycle can pick an already-fired assembly
     (a backward transition), so the marginal asymmetry + noise gives an unreliable order.
   - ARM B (GAMMA-WTA + POST-FIRE SILENCE): self-avoidance -> already-fired assemblies are removed, so backward is
     forbidden and the forward chain reliably wins each remaining slot.
3. Anti-cheats: SCRAMBLE (shuffle W within the off-diagonal -> ARM B's DIRECTION must collapse to chance, proving the
   forward bias comes from the learned weights, not from self-avoidance imposing an arbitrary permutation); and the
   ARM B >> ARM A contrast is the load-bearing single variable (self-avoidance on/off, everything else identical).

forward_frac = fraction of adjacent replay steps that go i -> i+1 (a full A->B->C = 1.000). chance for n_mem=3 from a
fixed start with self-avoidance = 0.5 (two possible orders).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU, deterministic; the encode is small

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence, SEQ_CFG  # noqa: E402
from research.runners._gap5_spontaneous_reactivation_derisk import _extract_ca3ca3_vec  # noqa: E402


def _extract_W(prep, n_mem):
    """n_mem x n_mem mean between-assembly transition weight matrix from the encoded bridge."""
    bridge = prep["bridge"]
    ca3_idx = list(bridge.region_manager.indices("ca3"))
    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_vec(bridge, ca3_idx, to_host)
    asm_of_local = np.full(len(ca3_idx), -1, dtype=np.int64)
    for m, a in enumerate(prep["assemblies_local"]):
        asm_of_local[np.asarray(a, dtype=np.int64)] = m
    a_pre = asm_of_local[pre_l_h]
    a_post = asm_of_local[post_l_h]
    d = np.asarray(to_host(bridge.cp_connections.data))
    W = np.zeros((n_mem, n_mem), dtype=np.float64)
    for i in range(n_mem):
        for j in range(n_mem):
            mask = (a_pre == i) & (a_post == j)
            W[i, j] = float(d[flat_h[mask]].mean()) if mask.any() else 0.0
    return W


def _replay(W, n_mem, self_avoid, rng, noise, start=0):
    """One gamma-organized replay from `start`. Returns the ordered list of fired assemblies."""
    Wm = W.copy()
    np.fill_diagonal(Wm, 0.0)   # the within-attractor is the HOLD, not the transition; the gamma cycle drives the PUSH
    fired = [start]
    cur = start
    for _ in range(n_mem - 1):
        drive = Wm[cur] + rng.normal(0.0, noise, n_mem)
        drive[cur] = -1e18                       # gamma always advances (the current is reset out of THIS slot)
        if self_avoid:
            for f in fired:
                drive[f] = -1e18                 # post-fire silence: an already-fired assembly cannot re-win
        nxt = int(np.argmax(drive))
        fired.append(nxt)
        cur = nxt
    return fired


def _forward_frac(order):
    steps = len(order) - 1
    fwd = sum(1 for k in range(steps) if order[k + 1] == order[k] + 1)
    return fwd / max(1, steps)


def _arm_stats(W, n_mem, self_avoid, seed, noise, n_trials, scramble=False):
    """scramble=True: shuffle the OFF-diagonal entries INDEPENDENTLY PER TRIAL (a valid control -- destroys the learned
    directionality on every trial, so the order can only come from self-avoidance imposing an arbitrary permutation)."""
    rng = np.random.default_rng(int(seed) * 7919 + (1 if self_avoid else 0) + (100000 if scramble else 0))
    off = ~np.eye(n_mem, dtype=bool)
    fracs = []
    for _ in range(n_trials):
        Wt = W
        if scramble:
            Wt = W.copy()
            vals = Wt[off].copy()
            rng.shuffle(vals)
            Wt[off] = vals
        fracs.append(_forward_frac(_replay(Wt, n_mem, self_avoid, rng, noise)))
    fr = np.asarray(fracs)
    return dict(mean=float(fr.mean()), full_fwd_rate=float((fr >= 0.999).mean()))


def one_seed(seed, cfg, n_mem, noise, n_trials):
    prep = _prepare_sequence(seed, cfg)
    W = _extract_W(prep, n_mem)
    prep_ne = _prepare_sequence(seed, cfg, do_encode=False)   # NO-ENCODE: baseline weights, no chain
    Wne = _extract_W(prep_ne, n_mem)

    A = _arm_stats(W, n_mem, False, seed, noise, n_trials)                    # weight-only (no self-avoidance)
    B = _arm_stats(W, n_mem, True, seed, noise, n_trials)                     # gamma-WTA + post-fire silence
    Bsc = _arm_stats(W, n_mem, True, seed, noise, n_trials, scramble=True)    # SCRAMBLE (per-trial) on ARM B
    Bne = _arm_stats(Wne, n_mem, True, seed, noise, n_trials)                 # NO-ENCODE on ARM B (no learned chain)

    # GO: gamma-WTA reliably forward AND that forwardness collapses to ~chance under BOTH per-trial scramble AND no-encode
    go = (B["mean"] > A["mean"] + 0.15) and (B["full_fwd_rate"] >= 0.90) \
        and (B["mean"] > Bsc["mean"] + 0.2) and (B["mean"] > Bne["mean"] + 0.2)
    wdiag = float(np.mean(np.diag(W)))
    adj_fwd = float(np.mean([W[i, i + 1] for i in range(n_mem - 1)]))
    adj_rev = float(np.mean([W[i + 1, i] for i in range(n_mem - 1)]))
    skip = float(np.mean([W[i, j] for i in range(n_mem) for j in range(n_mem) if j > i + 1])) if n_mem > 2 else 0.0
    print(f"  [seed {seed}] W: within~{wdiag:.1f} adj_fwd={adj_fwd:.1f} adj_rev={adj_rev:.1f} skip_fwd={skip:.1f} asym={adj_fwd - adj_rev:+.2f} "
          f"| ARM_A(weight-only)={A['mean']:.3f} | ARM_B(gamma-WTA+silence)={B['mean']:.3f} full={B['full_fwd_rate']:.3f} "
          f"| SCRAMBLE(per-trial)={Bsc['mean']:.3f} NO-ENCODE={Bne['mean']:.3f} => {'GAMMA-FIXES-ORDER' if go else 'no'}")
    return dict(seed=seed, W=W.tolist(), armA=A, armB=B, scramble=Bsc, noencode=Bne, go=bool(go),
                asym=adj_fwd - adj_rev, wdiag=wdiag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--within-events", type=int, default=30)
    ap.add_argument("--within-refresh", type=int, default=8)
    ap.add_argument("--chain-fwd", type=int, default=24)
    ap.add_argument("--noise", type=float, default=8.0, help="gamma-cycle drive noise std (comparable to the ~5 pA asym)")
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--out", default="research/findings/raw/gap5_r4/gamma_wta_replay.json")
    a = ap.parse_args()
    cfg = dict(SEQ_CFG)
    cfg["n_mem"] = int(a.n_mem); cfg["within_events"] = int(a.within_events)
    cfg["within_refresh"] = int(a.within_refresh); cfg["chain_fwd"] = int(a.chain_fwd); cfg["chain_rev"] = 0
    cfg["rank1_encode"] = True; cfg["overlap_draw"] = False
    _, backend = get_backend()
    print(f"[gap5-gamma] gamma-WTA+post-fire-silence over RANK 2's learned W (n_mem={a.n_mem}, noise={a.noise}, "
          f"trials={a.n_trials}) seeds={a.seeds} backend={backend}")
    per = [one_seed(s, cfg, a.n_mem, a.noise, a.n_trials) for s in a.seeds]
    n_go = sum(p["go"] for p in per)
    mA = float(np.mean([p["armA"]["mean"] for p in per]))
    mB = float(np.mean([p["armB"]["mean"] for p in per]))
    mS = float(np.mean([p["scramble"]["mean"] for p in per]))
    print(f"[gap5-gamma] VERDICT: {n_go}/{len(per)} seeds -- gamma-WTA+silence forward {mB:.3f} vs weight-only {mA:.3f} "
          f"vs SCRAMBLE {mS:.3f}. {'GO: phase-timing fixes the replay order on the learned weights.' if n_go == len(per) else 'partial/negative.'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, n_go=n_go, per=per), f, indent=2)


if __name__ == "__main__":
    main()
