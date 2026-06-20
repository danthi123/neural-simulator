"""Phase C — TASK 1 PROBE: measure the OneBrainComposer cleanup membrane (the OP RESULT the S5 seam must couple
on-substrate) for a present vs absent block, BEFORE building the option-(a) on-bridge projection.

The S5 question (`2026-06-19-tier2-phaseC-integrated-loop-design.md` §2.3 / §6): the cleanup result is RF state
`Re(c)` on `cp_membrane_potential_v` (a GRADED matched-filter score per role per word). Phase B read it to host
(`block_cleanup_scores`) and applied `scores_to_drive` (threshold at 0.5*peak) in numpy. Option (a) asks: can a
FIXED on-bridge projection convert that graded score into the SAME decoded-word-line drive without a host read,
cleanly enough that the gated-match stays decisive?

This probe quantifies the input the projection must convert:
  - the per-word cleanup scores for the agent role + the action role, for a present block (block 0) and an absent
    block read against the wrong codebook position;
  - the PEAK, the RUNNER-UP, the ratio, and whether the absolute magnitude is far above an Izhikevich threshold
    (so a fixed threshold can't separate winner from runner-up -- the structural reason the discrimination is
    RELATIVE, the heart of option (a) vs (b)).

  SIM_BACKEND=cupy python -u -m research.runners._phaseC_task1_cleanup_probe --seed 42 --dim 64
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from sim.backend import to_host, is_gpu_backend
from research.runners.one_brain_composer import OneBrainComposer

FACTS = [("dog", "go", "north"), ("cat", "run", "river")]
VOCAB = ["cat", "dog", "fox", "go", "north", "river", "run", "see", "tree", "bird", "sun", "moon"]


def block_cleanup_scores_full(c, block_idx):
    """Reconstruct + unbind + cleanup ONE block (the validated _read_block op), and return the per-word cleanup
    membrane scores for ALL main roles (a list of V-vectors) -- the raw scores _read_block argmaxes over and the
    SAME arrays Phase B's `block_cleanup_scores` reads (agent + action). The cleanup membrane is Re(c)."""
    comp, b, D, Pd, V = c.comp, c.b, c.D, c.period, c.V
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    trig = c.store_base + block_idx * c.block
    kick = np.zeros(c.n_total, dtype=np.complex128)
    kick[trig] = 1.0
    b.rf_set_complex_weights(c.store_conns)
    b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
    b.rf_resonate_steps(Pd + 8)
    unbind = []
    for ri, role in enumerate(c.bind_roles):
        zc = np.conj(comp._to_phasor(comp.roles[role]))
        unbind += [(c.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unbind)
    b.rf_resonate_steps(Pd + 8)
    clean = []
    for ri, role in enumerate(c.main_roles):
        for j in range(V):
            cc = np.conj(comp._to_phasor(comp.concepts[c.words[j]]))
            clean += [(c.c_base + ri * V + j, c.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    return [mem[c.c_base + ri * V:c.c_base + (ri + 1) * V].copy() for ri in range(c.n_main)]


def describe(scores, label, words):
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    order = np.argsort(s)[::-1]
    peak = float(s[order[0]])
    runner = float(s[order[1]]) if s.size > 1 else 0.0
    return dict(label=label, peak=peak, runner_up=runner,
                peak_word=words[int(order[0])], runner_word=words[int(order[1])] if s.size > 1 else None,
                ratio=(peak / runner if runner > 0 else float("inf")),
                raw_min=float(s.min()), raw_max=float(s.max()),
                top5=[(words[int(i)], round(float(s[i]), 1)) for i in order[:5]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default="research/findings/raw/_phaseC_task1_cleanup_probe.json")
    args = ap.parse_args()

    c = OneBrainComposer(seed=args.seed, D=args.dim, vocab=VOCAB, k_max=8,
                         enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in FACTS:
        c.store(a, x, p)
    words = c.words
    # block 0 = (dog, go, north) -> agent should peak at 'dog', action at 'go'
    sc0 = block_cleanup_scores_full(c, 0)
    sc1 = block_cleanup_scores_full(c, 1)   # (cat, run, river)
    rep = dict(seed=args.seed, dim=args.dim, n_words=len(words), words=words,
               block0_agent=describe(sc0[0], "block0/agent (truth=dog)", words),
               block0_action=describe(sc0[1], "block0/action (truth=go)", words),
               block1_agent=describe(sc1[0], "block1/agent (truth=cat)", words),
               block1_action=describe(sc1[1], "block1/action (truth=run)", words))
    # the structural question: is the peak far above an Izhikevich threshold (~30mV) so a fixed threshold can't
    # separate winner from runner-up? (peak AND runner-up both >> 30 -> the discrimination is RELATIVE.)
    rep["both_suprathreshold_vs_izh_30mV"] = bool(rep["block0_agent"]["runner_up"] > 30.0)
    rep["gpu"] = is_gpu_backend()

    print(json.dumps(rep, indent=2, default=str), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
