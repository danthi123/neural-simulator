"""ROADMAP PHASE 2 (the real "one brain"), STEP 3a -- fold the CLEANUP matched-filter onto the SAME persistent
bridge as the store+query (steps 1+2), so the full chain bind->bundle->unbind->CLEANUP runs register->register with
NO host round-trips. The cleanup's stage-1 matched filter is an RF complex-synapse matvec (the SAME op as unbind):
concept-score neurons read the recovered Q register via conj(codebook) synapses, and their membrane (re) = the match
score (= the numpy cosine score, per `_spiking_cleanup`). The argmax over those on-bridge scores is the answer.

This replaces the numpy `comp._cleanup` stage-1 cosine with an on-bridge matvec reading Q directly (no `rf_read_phases`
of Q to numpy). Registers (D each + V concept neurons): a_in[0], v_in[1], a_bound[2], v_bound[3], C[4], Q[5], then V
concept-score neurons at [6D : 6D+V]. Windows: bind -> bundle -> query(C->Q, conj agent_role) -> cleanup(Q->concept
via conj(code_k), one matvec) -> read concept membranes -> argmax.

GATE (3 seeds x 2 D): the on-bridge cleanup argmax == the numpy cleanup of the same Q AND == the true filler, 100%
of facts (agent role). Anti-cheat: a WRONG-codebook cleanup (shuffle the concept->code map) must miss. Reuse-by-
import (RFPhasorComposer + _build_rf_bridge). GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_cleanup_onbridge_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import to_host  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge  # noqa: E402

AGENTS = ["dog", "cat", "bird", "river", "apple"]
ACTIONS = ["go", "come", "look", "stop", "swim"]
VOCAB = AGENTS + ACTIONS


def _store_query_cleanup(b, comp, agent_w, action_w, period, shuffle_codebook=False):
    """Full chain on ONE bridge: bind->bundle->store C->unbind agent to Q->matched-filter Q to V concept neurons.
    Returns (onbridge_word, numpy_word) for the AGENT query."""
    D = comp.D; V = len(VOCAB)
    za_role = comp._to_phasor(comp.roles["agent"]); zv_role = comp._to_phasor(comp.roles["action"])
    za_fill = comp._to_phasor(comp.concepts[agent_w]); zv_fill = comp._to_phasor(comp.concepts[action_w])
    bind_a = [(2 * D + k, 0 * D + k, complex(za_role[k])) for k in range(D)]
    bind_v = [(3 * D + k, 1 * D + k, complex(zv_role[k])) for k in range(D)]
    bundle = ([(4 * D + k, 2 * D + k, 1.0) for k in range(D)] + [(4 * D + k, 3 * D + k, 1.0) for k in range(D)])
    qa = [(5 * D + k, 4 * D + k, complex(np.conj(za_role[k]))) for k in range(D)]            # C -> Q (unbind agent)
    # cleanup matched filter: concept neuron 6D+j reads Q[5D+d] via conj(code_j[d]); winner read via the TRUE VOCAB
    # order. Anti-cheat (shuffle_codebook): RANDOM codes (not the real concept codes) -> no real match -> chance
    # (a permutation would be self-undoing: the matched filter finds the true code wherever it sits).
    rng = np.random.default_rng(999)
    clean = []
    for j in range(V):
        cc = (np.conj(np.exp(2j * np.pi * rng.uniform(0, 1, D))) if shuffle_codebook
              else np.conj(comp._to_phasor(comp.concepts[VOCAB[j]])))
        clean += [(6 * D + j, 5 * D + d, complex(cc[d])) for d in range(D)]
    kick = np.zeros(6 * D + V, dtype=np.complex128)
    kick[0 * D:1 * D] = za_fill; kick[1 * D:2 * D] = zv_fill
    b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0     # reset work registers
    b.rf_set_complex_weights(bind_a + bind_v); b.rf_kick(kick, period=period, lam=0.0); b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(bundle); b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(qa); b.rf_resonate_steps(period + 8)
    # numpy reference cleanup of Q (read Q out -- only for the comparison; the on-bridge path does NOT use it)
    q_phases = np.asarray(b.rf_read_phases())[5 * D:6 * D]
    numpy_word = comp._cleanup(q_phases, VOCAB)
    # on-bridge cleanup: one matched-filter matvec step Q -> concept; the concept membrane (re) = the match score
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    scores = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)[6 * D:6 * D + V]
    onbridge_word = VOCAB[int(np.argmax(scores))]
    return onbridge_word, numpy_word


def run_seed(seed, D):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    b = _build_rf_bridge(6 * D + len(VOCAB), seed)
    pairs = list(zip(AGENTS, ACTIONS))
    ok_self = ok_np = 0
    for ag, ac in pairs:
        ob, npw = _store_query_cleanup(b, comp, ag, ac, comp.period)
        ok_self += int(ob == ag); ok_np += int(ob == npw)
    # anti-cheat: wrong-codebook cleanup on the first fact (should miss the true agent)
    ag0, ac0 = pairs[0]
    shuf, _ = _store_query_cleanup(b, comp, ag0, ac0, comp.period, shuffle_codebook=True)
    n = len(pairs)
    row = {"seed": seed, "D": D, "self": ok_self / n, "vs_numpy": ok_np / n, "shuffle_hit": int(shuf == ag0)}
    print(f"  [seed {seed} D={D}] on-bridge cleanup self={ok_self/n:.2f} | ==numpy {ok_np/n:.2f} | "
          f"shuffle_hit={row['shuffle_hit']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_cleanup_onbridge.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print("[one-brain on-bridge cleanup de-risk] does the matched-filter cleanup fold onto the persistent bridge "
          "(concept neurons read Q, membrane=score) == the numpy cleanup?\n", flush=True)
    rows = [run_seed(s, D) for s in seeds for D in dims]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    self_acc, vs_np, shuf = m("self"), m("vs_numpy"), m("shuffle_hit")
    n_go = sum(int(r["self"] >= 0.99 and r["vs_numpy"] >= 0.99) for r in rows)
    go = (n_go == len(rows)) and (shuf <= 0.34)
    print(f"\n{'='*96}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D): on-bridge cleanup self {self_acc:.3f} | ==numpy {vs_np:.3f} | "
          f"shuffle_hit {shuf:.2f} | self&==numpy>=0.99: {n_go}/{len(rows)}", flush=True)
    if go:
        print(f"  GO: the cleanup matched-filter folds onto the persistent bridge (concept neurons read Q directly, "
              f"membrane=score, argmax=answer) == the numpy cleanup 100%, wrong-codebook misses. ==> bind->bundle->"
              f"unbind->cleanup all run register->register on ONE bridge, no host round-trip. Next: the spiking WTA "
              f"selection (izh co-resident) + the moat + the parser front-end.", flush=True)
    else:
        print(f"  BOUNDARY: on-bridge cleanup {self_acc:.3f}/==numpy {vs_np:.3f} (shuffle {shuf:.2f}) -- the matvec "
              f"score read or the chain coherence needs tuning.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*96}", flush=True)
    out = {"verdict": "GO" if go else "BOUNDARY", "seeds": seeds, "dims": dims, "self": self_acc,
           "vs_numpy": vs_np, "shuffle_hit": shuf, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
