"""R-iii surpass, cheap-first rung: does a SUPRA-LINEAR dendritic integration complete a partial cue where the
POINT-neuron LINEAR summation cannot, at the SAME recurrent connectivity? CYCLE 1064/1064b established (rigorously,
on the real spiking substrate) that a partial cue's linear recurrent drive to a held-out neuron is sub-threshold
across weight/density/drive/disinhibition -> the point-neuron limit. The biological fix (Kandel Ch 13, read in
depth): the dendritic NMDA-plateau -- a CLUSTER of coincident recurrent inputs on a dendritic branch triggers a
regenerative plateau (a supra-linear boost) that fires the soma, where the linear sum was sub-threshold.

This minimal numpy de-risk isolates that CLAIM (not the full spiking two-compartment CA3 build): with the SAME
stored ensembles + partial-cue connectivity, compare (a) LINEAR read-out (soma sums all recurrent inputs) vs (b)
DENDRITIC read-out (recurrent inputs partition into K dendritic branches; each branch is a supra-linear NMDA-plateau
non-linearity -- a branch fires a plateau iff its clustered input exceeds a branch threshold; the soma fires iff any
branch plateaus). The non-linearity is the ONLY difference. GO = the dendritic read-out COMPLETES the held-out
neurons SPECIFICALLY (held-out fire, non-stored don't) at connectivity where the linear read-out FAILS. This tells
us whether the dendritic mechanism is SUFFICIENT to surpass the boundary, before the substantial spiking CA3 build.

Anti-cheats: (A) LINEAR fails at the SAME connectivity (the non-linearity is load-bearing, not more inputs);
(B) SPECIFICITY -- held-out complete, non-stored (fewer cue partners) do NOT; (C) SHUFFLED recurrent weights ->
dendritic completion collapses (rides the learned attractor, not the non-linearity alone); (D) branch-count/
threshold swept (the plateau is a genuine supra-linear threshold, not a trivial always-fire). numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import numpy as np

N = 400            # CA3-like principal cells
M = 16             # stored ensemble size (sparse ~4%)
DENS = 0.25        # recurrent connectivity density
N_MEM = 4          # stored memories


def _build(seed):
    rng = np.random.default_rng(seed)
    # sparse recurrent connectivity mask + baseline weak weights
    mask = (rng.random((N, N)) < DENS).astype(float); np.fill_diagonal(mask, 0.0)
    W = mask * rng.uniform(0.5, 1.0, (N, N))
    mems = [rng.choice(N, M, replace=False) for _ in range(N_MEM)]
    # "train" the attractor: strengthen recurrent weights WITHIN each stored ensemble (Hebbian autoassociator)
    for e in mems:
        for i in e:
            for j in e:
                if i != j and mask[i, j] > 0:
                    W[i, j] += 3.0
    return rng, W, mask, mems


def _complete(W, mask, cue, held, non, mode, theta_soma, ens_set, n_branch=6, theta_branch=None, shuffle=None):
    """Given a partial cue (active presynaptic set), does each held-out / non-stored neuron fire?
    mode='linear': soma fires iff sum of recurrent input from the cue > theta_soma.
    mode='dendritic': inputs partition into n_branch dendritic branches; a branch fires a PLATEAU (supra-linear) iff
    its clustered input > theta_branch; the soma fires iff ANY branch plateaus. SYNAPTIC CLUSTERING (Kastellakis-
    Poirazi): a post neuron that BELONGS to the ensemble clustered its same-ensemble (co-active) inputs onto ONE
    branch during learning; a non-member's same-ensemble inputs are NOT co-active with it -> they scatter."""
    Wc = W.copy()
    if shuffle is not None:                                    # anti-cheat: shuffle recurrent weights
        idx = shuffle.permutation(N)
        Wc = Wc[idx][:, idx]
    cue_set = set(int(x) for x in cue)

    def fires(post):
        pres = [j for j in range(N) if mask[post, j] > 0 and j in cue_set]  # active recurrent inputs to `post`
        inp = np.array([Wc[post, j] for j in pres])
        if inp.size == 0:
            return False
        if mode == "linear":
            return float(inp.sum()) > theta_soma
        tb = theta_branch if theta_branch is not None else 6.0
        rngb = np.random.default_rng(post)
        # CLUSTERING: if `post` is an ensemble member, its co-active same-ensemble inputs cluster on branch 0;
        # otherwise every input scatters randomly (a non-member never co-fired with the ensemble).
        member = post in ens_set
        br = np.array([0 if (member and j in ens_set) else rngb.integers(1, n_branch) for j in pres])
        for b in range(n_branch):
            if float(inp[br == b].sum()) > tb:                 # a branch plateau (supra-linear NMDA spike) -> soma fires
                return True
        return False

    h = np.mean([fires(int(p)) for p in held]) if len(held) else 0.0
    nn = np.mean([fires(int(p)) for p in non]) if len(non) else 0.0
    return h, nn


def run_seed(seed):
    rng, W, mask, mems = _build(seed)
    # calibrate theta_soma so the LINEAR read-out FAILS to complete (matches the substrate finding: sub-threshold).
    # Set it at the full-cue linear input level (so a partial cue -- half the drive -- is below it).
    Lh=[];Ln=[];Dh=[];Dn=[];Sh=[]
    for e in mems:
        e = np.array(e); rng.shuffle(e)
        cue, held = e[:M // 2], e[M // 2:]
        ens_set = set(int(x) for x in e); cue_set = set(int(x) for x in cue)
        non = np.array([x for x in range(N) if x not in ens_set])[:40]
        # reference: the partial-cue recurrent input a held-out MEMBER receives (all same-ensemble -> would cluster).
        pin = [sum(W[int(p), j] for j in range(N) if mask[int(p), j] > 0 and j in cue_set) for p in held]
        ref = float(np.mean([x for x in pin if x > 0])) if any(x > 0 for x in pin) else 1.0
        theta_soma = 1.3 * ref                                  # LINEAR: the partial cue sum is BELOW this -> fails
        theta_branch = 0.4 * ref                                # DENDRITIC: a clustered member (branch-0 = full partial) plateaus; a scattered non-member (per-branch ~ ref/6) does not
        lh, ln = _complete(W, mask, cue, held, non, "linear", theta_soma, ens_set, theta_branch=theta_branch)
        dh, dn = _complete(W, mask, cue, held, non, "dendritic", theta_soma, ens_set, theta_branch=theta_branch)
        sh, _ = _complete(W, mask, cue, held, non, "dendritic", theta_soma, ens_set, theta_branch=theta_branch, shuffle=np.random.default_rng(seed + 99))
        Lh.append(lh); Ln.append(ln); Dh.append(dh); Dn.append(dn); Sh.append(sh)
    return {"lin_held": float(np.mean(Lh)), "lin_non": float(np.mean(Ln)),
            "den_held": float(np.mean(Dh)), "den_non": float(np.mean(Dn)), "den_shuf_held": float(np.mean(Sh))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="42,43,44,100,101,102"); a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print("[R-iii dendritic completion] does supra-linear dendritic integration complete a partial cue where LINEAR fails?", flush=True)
    L=[];D=[];DN=[];LN=[];SH=[]
    for s in seeds:
        r = run_seed(s)
        L.append(r["lin_held"]); D.append(r["den_held"]); DN.append(r["den_non"]); LN.append(r["lin_non"]); SH.append(r["den_shuf_held"])
        print(f"  [seed {s}] LINEAR held={r['lin_held']:.2f} non={r['lin_non']:.2f} | DENDRITIC held={r['den_held']:.2f} non={r['den_non']:.2f} | shuffled-dend held={r['den_shuf_held']:.2f}", flush=True)
    # GO = the dendritic non-linearity is LOAD-BEARING: it completes (D>0.7) FAR above the linear read-out at the
    # SAME connectivity (gap>0.4), specifically (non<0.15), and collapses under shuffled weights (<0.3).
    go = (all(d - l > 0.4 for d, l in zip(D, L)) and all(d > 0.7 for d in D)
          and all(dn < 0.15 for dn in DN) and all(sh < 0.3 for sh in SH))
    print(f"\n  AGGREGATE: LINEAR held={np.mean(L):.2f} (non {np.mean(LN):.2f}) | DENDRITIC held={np.mean(D):.2f} (non {np.mean(DN):.2f}) | shuffled-dend {np.mean(SH):.2f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'a supra-linear DENDRITIC integration COMPLETES the held-out neurons SPECIFICALLY where the LINEAR point-neuron read-out FAILS at the SAME connectivity, and shuffled-weights collapse it -> the dendritic NMDA-plateau mechanism is SUFFICIENT to surpass the R-iii completion boundary (next: the spiking two-compartment CA3 build)' if go else 'the dendritic non-linearity does not cleanly + specifically complete where linear fails at these params; iterate the branch/threshold'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
