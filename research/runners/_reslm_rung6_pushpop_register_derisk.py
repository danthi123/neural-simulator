"""RUNG 6 (cheap-first) — the emergent generator's next-clause SUBJECT prediction across a discourse POP needs a TWO-GATE
push/pop who-register, which a single latch (Rung 2) cannot do. The distinctive new capability over Rung 2: after a
discourse pop (a return connective + pronoun, Grosz-Sidner attentional stack), the correct next subject is the EARLIER
protagonist — which a PUSH gate stored and a POP gate restores, but which a single most-recent latch OVERWROTE. This
probes whether adding the two-gate register's who-state as a read-out feature lets the reservoir generator predict the
resumed protagonist, where the reservoir alone (fading) and a single latch (holds the interloper) both fail.

Self-contained cheap-first (minimal ESN reservoir + a minimal two-gate push/pop who-register + a synthetic push/pop
discourse) — the CORE hypothesis before wiring the full validated D3 register + reslm reservoir. NO `sim/` edit.

DISCOURSE (per trial): INTRODUCE A (A acts) -> [PUSH: connective + a NEW named subject B; B acts for k clauses] ->
[POP: a return connective + a pronoun -> the subject RESUMES to A] -> predict the subject of the clause AFTER the pop.
The correct answer is A (the resumed, earlier protagonist). Arms:
  - REGISTER (reservoir state ++ two-gate push/pop who-state) -> should predict A
  - LATCH-ONLY (reservoir ++ single most-recent-subject latch) -> predicts B (the interloper) = the Rung-2 ceiling
  - RESERVOIR-ONLY (fading) -> chance
  - SHUFFLE-REGISTER (anti-cheat: register who-state shuffled across trials) -> collapses

GO (6-seed 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): register post-pop subject-accuracy > latch-only + > reservoir-
only + > shuffle, each by a margin, on >=5/6 both sets. (Latch-only being WRONG post-pop is the point — it is the Rung-2
single-latch ceiling the two-gate register surpasses.)

Run: python -m research.runners._reslm_rung6_pushpop_register_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np

N_ENT = 6                    # named entities (the "subjects" vocabulary)
PRONOUN = N_ENT              # a generic PRONOUN surface token (drives the reservoir but does NOT name the referent)
N_POOL = 200                 # reservoir size
N_TRIALS = 240               # discourses per seed
PUSH_K = 8                   # clauses the interloper B holds before the pop -> A is DISTAL (reservoir fades it)


def _ent_code(rng):
    """Fixed random code per entity + one for the PRONOUN surface token (the emergent generator's token embedding)."""
    return rng.standard_normal((N_ENT + 1, 16))


def _reservoir(seed):
    rng = np.random.default_rng(seed * 7 + 1)
    Win = rng.standard_normal((N_POOL, 16)) * 0.6
    W = rng.standard_normal((N_POOL, N_POOL))
    W *= 0.95 / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9)   # spectral radius 0.95 (echo-state)
    return Win, W


def _run_discourse(ecode, Win, W, clauses):
    """Drive the reservoir along the clause SURFACE tokens; return per-clause reservoir states + the two-gate who-state
    + the single-latch who-state. clauses = ordered (surface_token, gate_kind, named_subj) -- surface_token is the code
    that DRIVES the reservoir (a PRONOUN on a pop, so the reservoir never directly sees the resumed referent)."""
    x = np.zeros(N_POOL)
    states, reg_who, latch_who = [], [], []
    held = -1; current = -1; latch = -1
    for (surface, kind, named) in clauses:
        x = np.tanh(W @ x + Win @ ecode[surface])            # driven by the SURFACE token (pronoun on a pop)
        if kind == "push":                                   # boundary + NEW named subject -> push current into held
            held = current; current = named; latch = named
        elif kind == "pop":                                  # return marker + PRONOUN -> restore the held protagonist
            current = held if held >= 0 else current         # (latch is NOT updated -- pop names no one)
        else:                                                # introduce / continue
            current = named; latch = named
        states.append(x.copy())
        reg_who.append(current)                              # two-gate: resumes the earlier protagonist after a pop
        latch_who.append(latch)                              # single latch: the most-recent NAMED subject (the interloper)
    return np.array(states), np.array(reg_who), np.array(latch_who)


def _build_trial(rng):
    """A push/pop discourse; returns (clauses, post-pop clause index, the correct referent A). The pop clause's SURFACE
    is a PRONOUN -> the reservoir does not directly encode A; only the register's pop gate restores it."""
    A, B = rng.choice(N_ENT, 2, replace=False)
    seq = [(int(A), "introduce", int(A))]                    # A introduced
    seq.append((int(B), "push", int(B)))                     # push: a new named subject B enters
    for _ in range(PUSH_K):
        seq.append((int(B), "continue", int(B)))             # B holds the floor (A goes DISTAL)
    seq.append((PRONOUN, "pop", -1))                         # pop: PRONOUN surface -> resume A (unnamed at the surface)
    post = len(seq) - 1                                      # predict the referent of the pronoun = the resumed A
    return seq, post, int(A)


def _onehot(v, n=N_ENT):
    z = np.zeros(n); z[v] = 1.0; return z


def _fit_readout(X, Y, l2=1.0):
    A = X.T @ X + l2 * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def _arm_acc(seed, arm, ecode, Win, W, rng):
    trials = [_build_trial(rng) for _ in range(N_TRIALS)]
    feats, targs, post_idx = [], [], []
    who_col = []
    for (seq, post, A) in trials:
        states, reg_who, latch_who = _run_discourse(ecode, Win, W, seq)
        s = states[post]
        if arm == "reservoir":
            f = s
        elif arm == "register":
            f = np.concatenate([s, _onehot(reg_who[post])])
        elif arm == "latch":
            f = np.concatenate([s, _onehot(latch_who[post])])
        elif arm == "shuffle":
            f = np.concatenate([s, _onehot(reg_who[post])])   # who-state shuffled below (across trials)
        feats.append(f); targs.append(_onehot(A)); post_idx.append(post)
        who_col.append(reg_who[post])
    X = np.array(feats); Y = np.array(targs)
    if arm == "shuffle":                                     # anti-cheat: permute the who-feature across trials
        who = X[:, N_POOL:].copy()
        X[:, N_POOL:] = who[rng.permutation(len(who))]
    ntr = int(0.7 * len(X))
    Wro = _fit_readout(X[:ntr], Y[:ntr])
    pred = (X[ntr:] @ Wro).argmax(1)
    true = Y[ntr:].argmax(1)
    return float(np.mean(pred == true))


def run(seed):
    rng = np.random.default_rng(seed)
    ecode = _ent_code(np.random.default_rng(seed * 3 + 5))
    Win, W = _reservoir(seed)
    accs = {arm: _arm_acc(seed, arm, ecode, Win, W, np.random.default_rng(seed * 11 + {"register": 1, "latch": 2, "reservoir": 3, "shuffle": 4}[arm]))
            for arm in ("register", "latch", "reservoir", "shuffle")}
    reg = accs["register"]
    go = bool(reg > accs["latch"] + 0.15 and reg > accs["reservoir"] + 0.15 and reg > accs["shuffle"] + 0.15)
    print(f"[rung6 seed={seed}] register={reg:.2f} latch={accs['latch']:.2f} reservoir={accs['reservoir']:.2f} "
          f"shuffle={accs['shuffle']:.2f} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, **accs, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    print(f"[rung6] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
