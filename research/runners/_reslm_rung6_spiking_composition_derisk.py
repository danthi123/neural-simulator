"""RUNG 6 — the FULL SPIKING composition: the validated two-gate SPIKING D3 register (`SpikingPopGateRegister`, a
persistent slow-NMDA held slot on a real `SimulationBridge`) supplies the resumed-protagonist who-state to a reservoir
GENERATOR's read-out, so the generator predicts the next-clause subject across a discourse POP where the fading reservoir
alone cannot. This upgrades the cheap-first Rung-6 probe (a minimal structural register) to the DEPLOYED spiking register
+ real discourse statistics (`make_discourse`). NO `sim/` edit (reuse-by-import).

The D3 arc already validated that the spiking register RESUMES the earlier protagonist across a pop (RESUME_spiking vs
RESUME_poplesion). Rung 6 tests the GENERATION value: does its spiking who-state, fed to a reservoir read-out, let the
GENERATOR predict the resumed referent — the register carrying what the reservoir fades. Arms:
  - REGISTER (reservoir state ++ the spiking register's who_agent() after the pop) -> predicts the resumed referent
  - RESERVOIR-ONLY (fading) -> chance (the resumed protagonist is distal)
  - POP-LESION (the register's own pop_lesion=True -> the held slot is not restored) -> collapses
  - SHUFFLE (the who-state permuted across trials) -> collapses

GO: register pop-clause referent-accuracy > reservoir-only + margin AND > pop-lesion + margin AND > shuffle + margin.
Compute note: the spiking register trains a delta + builds a SimulationBridge + runs spiking per clause -> a real job;
`--smoke` runs a tiny seed-42 wiring check; the 6-seed validation is the heavy background run.

Run (smoke): python -m research.runners._reslm_rung6_spiking_composition_derisk --smoke
Run (full):  python -m research.runners._reslm_rung6_spiking_composition_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse
import json
import numpy as np

from research.runners._d3_event_popgate_spiking_agent_derisk import SpikingPopGateRegister
from research.runners._d3_event_selfsup_pair_agent_derisk import make_discourse, COREF_W, PROMOTE_W, CONNECTIVES
from research.runners._d3_event_popgate_agent_derisk import _truth

N_POOL = 200
N_RES_IN = 16


def _reservoir(seed):
    rng = np.random.default_rng(seed * 7 + 1)
    Win = rng.standard_normal((N_POOL, N_RES_IN)) * 0.6
    W = rng.standard_normal((N_POOL, N_POOL))
    W *= 0.95 / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9)
    return Win, W


def _tok_code(word, ecode, w2i):
    i = w2i.setdefault(word, len(w2i) % ecode.shape[0])
    return ecode[i]


def _drive_reservoir(clauses, Win, W, ecode, w2i):
    """Reservoir state after each clause's SURFACE tokens (a pronoun on a pop is just its generic token -- the reservoir
    never sees the resumed referent's name)."""
    x = np.zeros(N_POOL)
    for clause in clauses:
        for tok in clause.split():
            x = np.tanh(W @ x + Win @ _tok_code(tok, ecode, w2i))
    return x.copy()


def _drive_register(reg, clauses):
    """Drive the spiking register clause-by-clause (connective-prefixed clause = boundary; pronoun subject = pop);
    return the register's who_agent() after the LAST clause (the resumed protagonist after a pop)."""
    reg.reset()
    for clause in clauses:
        toks = clause.split()
        if toks and toks[0] in CONNECTIVES:
            reg.mark_boundary()
            toks = toks[1:]
        if len(toks) >= 3:
            reg.observe(toks[0], toks[2])                    # subject, object (verb between)
        elif len(toks) == 2:
            reg.observe(toks[0], toks[1])
    return reg.who_agent()


def _onehot(name, refs):
    z = np.zeros(len(refs)); z[refs.index(name)] = 1.0; return z


def _fit(X, Y, l2=1.0):
    return np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ Y)


def _collect(seed, n_disc, pop_lesion):
    """Gather pop-discourses (a return-to-earlier-protagonist), the reservoir state + the spiking register who-state at
    the pop, and the ground-truth resumed referent."""
    rng = np.random.RandomState(seed)                    # make_discourse uses legacy .randint
    refs = [f"e{i}" for i in range(6)]
    reg = SpikingPopGateRegister(refs, seed=seed, pop_lesion=pop_lesion)
    Win, W = _reservoir(seed)
    ecode = np.random.default_rng(seed * 3 + 5).standard_normal((32, N_RES_IN))
    w2i = {}
    rows = []
    tried = 0
    while len(rows) < n_disc and tried < n_disc * 80:
        tried += 1
        clauses, tn, tb = make_discourse(rng, refs)
        tr = _truth(clauses, refs)                           # per-clause (agent, patient, is_pop)
        if len(tr) < 2 or not tr[-1][2]:                     # the last clause must be a POP (return + pronoun)
            continue
        resumed, pre_pop = tr[-1][0], tr[-2][0]
        if resumed == pre_pop:                               # the pop must actually change the agent back
            continue
        s = _drive_reservoir(clauses, Win, W, ecode, w2i)
        who = _drive_register(reg, clauses)
        rows.append((s, _onehot(who, refs), _onehot(refs[resumed], refs)))
    return rows, refs


def run(seed, n_disc):
    rows, refs = _collect(seed, n_disc, pop_lesion=False)
    les_rows, _ = _collect(seed, max(20, n_disc // 3), pop_lesion=True)
    if len(rows) < 12:
        print(f"[rung6-spk seed={seed}] too few pop discourses ({len(rows)}) -- inconclusive", flush=True)
        return {"seed": seed, "n": len(rows), "GO": False}
    S = np.array([r[0] for r in rows]); WHO = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows])
    ntr = int(0.7 * len(S))

    def acc(Xtr, Xte, ytr, yte):
        Wro = _fit(Xtr, ytr); return float(np.mean((Xte @ Wro).argmax(1) == yte.argmax(1)))
    yv = Y.argmax(1)
    reg_acc = acc(np.hstack([S, WHO])[:ntr], np.hstack([S, WHO])[ntr:], Y[:ntr], Y[ntr:])
    res_acc = acc(S[:ntr], S[ntr:], Y[:ntr], Y[ntr:])
    sh = WHO.copy(); sh = sh[np.random.default_rng(seed * 5).permutation(len(sh))]
    shu_acc = acc(np.hstack([S, sh])[:ntr], np.hstack([S, sh])[ntr:], Y[:ntr], Y[ntr:])
    # pop-lesion: the register's held slot is not restored -> its who-state is wrong -> augmented read-out gains nothing
    Sl = np.array([r[0] for r in les_rows]); Wl = np.array([r[1] for r in les_rows]); Yl = np.array([r[2] for r in les_rows])
    nl = int(0.7 * len(Sl))
    les_acc = acc(np.hstack([Sl, Wl])[:nl], np.hstack([Sl, Wl])[nl:], Yl[:nl], Yl[nl:]) if len(Sl) >= 12 else 0.0
    go = bool(reg_acc > res_acc + 0.15 and reg_acc > shu_acc + 0.15 and reg_acc > les_acc + 0.10)
    print(f"[rung6-spk seed={seed}] register={reg_acc:.2f} reservoir={res_acc:.2f} shuffle={shu_acc:.2f} "
          f"pop_lesion={les_acc:.2f} (n_pop={len(rows)}) -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "register": reg_acc, "reservoir": res_acc, "shuffle": shu_acc, "pop_lesion": les_acc,
            "n": len(rows), "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-disc", type=int, default=120)
    ap.add_argument("--smoke", action="store_true", help="tiny seed-42 wiring check")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.smoke:
        run(42, 24)
        return
    res = [run(s, a.n_disc) for s in a.seeds]
    print(f"[rung6-spk] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
