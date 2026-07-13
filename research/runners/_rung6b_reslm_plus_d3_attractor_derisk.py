"""RUNG 6b of the open-generation ladder -- UNBOUNDED discourse-referent tracking: the reslm generator reads each clause
(its emergent front-end), and the VALIDATED D3 discrete-attractor holds the referent against the reservoir's fade, so
tracking generalizes to held-out-DEEPER narratives where the reslm reservoir ALONE fades (Rung 6 showed it fades 1.0->0.83
by gap 6).

WHY (Rung 6 -> 6b): Rung 6 (`2026-07-13-RUNG6-...`) showed the reslm's OWN reservoir carries the discourse referent at
SHORT range but FADES with distance -- exactly what a fixed reservoir must do. The D3 finding
(`2026-07-09-D3-language-reference-tracking-GO.md`) established the fix: a DISCRETE ATTRACTOR (re-discretizes each clause
-> no drift -> generalizes to any depth). Rung 6b COMPOSES them: the reslm reads each clause into a per-clause reservoir
STATE (the emergent, learned front-end -- the "dense per-step observation" the D3 finding said learning the referent-delta
needs), and the discrete-attractor integrates those per-clause states into an UNBOUNDED-tracked holder. Emergent front-end
+ validated unbounded-tracker; no composer scaffold.

THE TASK (a possession-transfer discourse, TOKEN-encoded): reuse `make_reference_tracking_task`'s holder-transition logic
(holder=b iff holder==a on a transfer, else no-op distractor; the last clause forced no-op so markov/last-named reveal
nothing) -> per narrative a STATE (holder per clause) + SEQ (pair a*K+b). Token-encode each clause as "subj_a gives
subj_b" over K=6 of the 16 A->W subjects; run the reslm over the whole narrative once; the per-clause reservoir STATE (the
state after each clause's last token) becomes the attractor's per-clause input X. Train on SHORT narratives (lens 1-3),
test on held-out-DEEPER (lens 6-8).

THE GATE (>=3, ideally 6 seeds; chance holder 1/6=0.167):
  (a) reslm-states -> DISCRETE-ATTRACTOR holder-track at test_DEEPER >> the RESERVOIR-ALONE read (a ridge read-out on the
      FINAL reslm state -> final holder, which fades at depth like Rung 6) -> the attractor HOLDS where the reservoir fades.
  (b) markov/last-object + retention floors near chance (the task needs the composed history, not a shortcut).
  (c) ATTRACTOR-LESION control: replace the reslm states with ZEROS -> the attractor cannot track (collapses) -> the reslm
      evidence is load-bearing.
GO: attractor state_deeper > 0.6 AND > reservoir_alone_deeper + 0.15 AND floors < 0.35 AND lesion < 0.35. numpy-CPU
(single run, contention-safe). Reuse-by-import; NO `sim/` edit.

Run:
  SIM_BACKEND=numpy python -m research.runners._rung6b_reslm_plus_d3_attractor_derisk --seed 42
  SIM_BACKEND=numpy python -m research.runners._rung6b_reslm_plus_d3_attractor_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import time

import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, ReservoirStates
from research.runners._emerge67_neural_spell_wirein_derisk import _AW_SUBJECTS
from research.runners._d3_reference_tracking_derisk import make_reference_tracking_task
from research.runners._d3_group_composition_derisk import discrete_attractor_rnn

_K = 6
_GIVES = "gives"
_SUBJ = _AW_SUBJECTS[:_K]                                  # 6 subjects from the A->W set
_VOCAB = list(_SUBJ) + [_GIVES]


def _reslm_states_for_task(res, vocab, split, n_pool):
    """Replace the abstract task split's X with the reslm's per-CLAUSE reservoir STATE. THE DIVISION OF LABOR: the reslm
    is the LOCAL per-clause ENCODER (it reads THIS clause's 'subj_a gives subj_b' -> a reservoir code of the (a,b)
    transfer), and the discrete-attractor does the GLOBAL cross-clause tracking. So each clause is a SEPARATE reslm wash
    (per_token_states washes at the start) -> X[n,t] encodes ONLY clause t (not a blur of 1..t, which is what made the
    running-cumulative whole-narrative read collapse the attractor to chance)."""
    X, Y, L, SEQ, STATE = split
    N, Lmax = SEQ.shape
    Xr = np.zeros((N, Lmax, n_pool), np.float32)
    structured = (n_pool == 2 * res.n)                         # STRUCTURED 2-slot read: reslm(a) ++ reslm(b), separable
    for n in range(N):
        for t in range(int(L[n])):
            a = int(SEQ[n, t]) // _K
            b = int(SEQ[n, t]) % _K
            if structured:                                     # a and b in SEPARATE halves (mirrors the D3 clean code)
                ca = res.per_token_states(vocab.encode_seq([_SUBJ[a]]))[-1]
                cb = res.per_token_states(vocab.encode_seq([_SUBJ[b]]))[-1]
                Xr[n, t] = np.concatenate([np.asarray(ca), np.asarray(cb)])
            else:                                              # NATURAL blurred per-clause code (subj_a gives subj_b)
                S = res.per_token_states(vocab.encode_seq([_SUBJ[a], _GIVES, _SUBJ[b]]))
                Xr[n, t] = np.asarray(S[-1])
    return (Xr, Y, L, SEQ, STATE)


def _reservoir_alone(train, test_deeper, n_cls, l2=1.0):
    """Baseline: ridge read-out on the FINAL reslm state -> final holder. Fades at depth (the Rung-6 fading-memory)."""
    def feat_y(split):
        Xr, Y, L, SEQ, STATE = split
        F = np.asarray([Xr[n, int(L[n]) - 1] for n in range(len(L))])
        y = np.asarray([STATE[n, int(L[n]) - 1] for n in range(len(L))])
        return F, y
    Ftr, ytr = feat_y(train); Fte, yte = feat_y(test_deeper)
    m, s = Ftr.mean(0), Ftr.std(0) + 1e-6
    Xa = np.concatenate([(Ftr - m) / s, np.ones((len(Ftr), 1))], 1)
    W = np.linalg.solve(Xa.T @ Xa + l2 * np.eye(Xa.shape[1]), Xa.T @ np.eye(n_cls)[ytr])
    Xte = np.concatenate([(Fte - m) / s, np.ones((len(Fte), 1))], 1)
    return float(np.mean(np.argmax(Xte @ W, 1) == yte))


def run(seed, n_hid=160, epochs=80, n_per_len=800, structured=False):
    rng = np.random.RandomState(seed)
    vocab = Vocab(list(_VOCAB))
    res = ReservoirStates(in_dim=vocab.size, seed=seed, n=160)
    n_pool = 2 * res.n if structured else res.n                # structured 2-slot read doubles the feature dim
    abs_task = make_reference_tracking_task(seed, K=_K, n_per_len=n_per_len)
    # rebuild the task with reslm-state X (train + test_deeper), keep K/ident/color
    tr = _reslm_states_for_task(res, vocab, abs_task["train"], n_pool)
    td = _reslm_states_for_task(res, vocab, abs_task["test_deeper"], n_pool)
    task = {"train": tr, "test_same": tr, "test_deeper": td,
            "K": _K, "ident": abs_task["ident"], "n_pool": n_pool, "color": abs_task["color"],
            "p_transfer": abs_task["p_transfer"]}
    att = discrete_attractor_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs)
    attr_deeper = att["state_deeper"]
    res_alone = _reservoir_alone(tr, td, _K)
    # LESION: zero the reslm states -> the attractor has no evidence
    trz = (np.zeros_like(tr[0]), tr[1], tr[2], tr[3], tr[4])
    tdz = (np.zeros_like(td[0]), td[1], td[2], td[3], td[4])
    lz = discrete_attractor_rnn({**task, "train": trz, "test_same": trz, "test_deeper": tdz}, seed=seed,
                                n_hid=n_hid, epochs=epochs)["state_deeper"]
    # floors: markov = predict the last clause's mentioned obj b; retention = predict ident
    def floors(split):
        Xr, Y, L, SEQ, STATE = split
        yt = np.asarray([STATE[n, int(L[n]) - 1] for n in range(len(L))])
        mk = np.asarray([int(SEQ[n, int(L[n]) - 1]) % _K for n in range(len(L))])
        rt = np.full(len(L), abs_task["ident"])
        return float(np.mean(mk == yt)), float(np.mean(rt == yt))
    mk, rt = floors(td)
    go = (attr_deeper > 0.6) and (attr_deeper > res_alone + 0.15) and (mk < 0.35) and (rt < 0.35) and (lz < 0.35)
    print(f"[rung6b seed={seed}] attractor_deeper={attr_deeper:.3f}  reservoir_alone={res_alone:.3f}  "
          f"lesion={lz:.3f} | markov={mk:.3f} retention={rt:.3f} chance=0.167 -> {'GO' if go else 'no'}")
    return dict(seed=seed, attractor_deeper=round(attr_deeper, 3), reservoir_alone=round(res_alone, 3),
                lesion=round(lz, 3), markov=round(mk, 3), retention=round(rt, 3), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--structured", action="store_true", help="2-slot read reslm(a)++reslm(b) (diagnostic: separable a/b)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    t0 = time.time()
    results = [run(s, structured=a.structured) for s in seeds]
    if len(results) > 1:
        gos = sum(1 for r in results if r["go"]); print(f"[rung6b] {gos}/{len(results)} seeds GO")
    if a.out:
        json.dump(dict(results=results, elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
