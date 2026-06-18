"""ROADMAP PHASE 2 (the real "one brain"), STEP 3b -- the no-confab MOAT on the persistent bridge. The cleanup
matched-filter's PEAK score (the max concept-neuron membrane, step 3a) IS a neural familiarity signal: a stored,
correctly-cued fact makes the recovered Q strongly match one concept (HIGH peak); an unbound/unstored query makes Q
noise that matches nothing (LOW peak). So a threshold on the on-bridge peak score = abstain-vs-answer -- the moat,
read off the substrate, no host equality check.

Builds on step 3a (`2026-06-18-one-brain-cleanup-onbridge-GO.md`): same persistent bridge + cleanup, but the de-risk
compares the peak score for a BOUND role (the fact has it -> answer) vs an UNBOUND role (the fact lacks it -> Q is
cross-talk noise -> abstain). GATE (3 seeds x 2 D): bound peak >> unbound peak with a clean separating threshold
(every bound > thr, every unbound < thr) AND the bound query answers correctly AND the unbound query abstains. The
threshold is the MIDPOINT of the bound/unbound peak distributions (measured, not tuned-to-pass).
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_moat_onbridge_derisk --seeds 42,43,44
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


def _store_query(b, comp, agent_w, action_w, query_role, period):
    """Store fact (agent,action) on ONE bridge, query `query_role` (bound: agent/action; unbound: patient), run the
    cleanup matched-filter, return (answer_word, peak_score)."""
    D = comp.D; V = len(VOCAB)
    za = comp._to_phasor(comp.roles["agent"]); zv = comp._to_phasor(comp.roles["action"])
    zq = comp._to_phasor(comp.roles[query_role])
    fa = comp._to_phasor(comp.concepts[agent_w]); fv = comp._to_phasor(comp.concepts[action_w])
    bind_a = [(2 * D + k, 0 * D + k, complex(za[k])) for k in range(D)]
    bind_v = [(3 * D + k, 1 * D + k, complex(zv[k])) for k in range(D)]
    bundle = ([(4 * D + k, 2 * D + k, 1.0) for k in range(D)] + [(4 * D + k, 3 * D + k, 1.0) for k in range(D)])
    qx = [(5 * D + k, 4 * D + k, complex(np.conj(zq[k]))) for k in range(D)]
    clean = []
    for j in range(V):
        cc = np.conj(comp._to_phasor(comp.concepts[VOCAB[j]]))
        clean += [(6 * D + j, 5 * D + d, complex(cc[d])) for d in range(D)]
    kick = np.zeros(6 * D + V, dtype=np.complex128); kick[:D] = fa; kick[D:2 * D] = fv
    b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
    b.rf_set_complex_weights(bind_a + bind_v); b.rf_kick(kick, period=period, lam=0.0); b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(bundle); b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(qx); b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
    scores = np.maximum(np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)[6 * D:6 * D + V], 0.0)
    return VOCAB[int(np.argmax(scores))], float(scores.max())


def run_seed(seed, D):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    b = _build_rf_bridge(6 * D + len(VOCAB), seed)
    bound_peaks, unbound_peaks, ans_ok = [], [], 0
    for ag, ac in zip(AGENTS, ACTIONS):
        w_a, p_a = _store_query(b, comp, ag, ac, "agent", comp.period)      # BOUND
        w_v, p_v = _store_query(b, comp, ag, ac, "action", comp.period)     # BOUND
        _, p_p = _store_query(b, comp, ag, ac, "patient", comp.period)      # UNBOUND -> should abstain
        bound_peaks += [p_a, p_v]; unbound_peaks += [p_p]
        ans_ok += int(w_a == ag) + int(w_v == ac)
    bmin, umax = min(bound_peaks), max(unbound_peaks)
    thr = 0.5 * (np.mean(bound_peaks) + np.mean(unbound_peaks))             # measured midpoint, NOT tuned
    sep = int(bmin > umax)                                                   # clean separation?
    bound_above = sum(int(p > thr) for p in bound_peaks) / len(bound_peaks)
    unbound_below = sum(int(p < thr) for p in unbound_peaks) / len(unbound_peaks)
    row = {"seed": seed, "D": D, "answer_acc": ans_ok / (2 * len(AGENTS)), "bound_min": bmin, "unbound_max": umax,
           "sep": sep, "bound_above_thr": bound_above, "unbound_below_thr": unbound_below}
    print(f"  [seed {seed} D={D}] answer={row['answer_acc']:.2f} | bound_peak_min={bmin:.2f} unbound_peak_max={umax:.2f}"
          f" | clean_sep={sep} | bound>thr={bound_above:.2f} unbound<thr={unbound_below:.2f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44"); ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_moat_onbridge.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print("[one-brain moat de-risk] does the on-bridge cleanup PEAK score gate abstain-vs-answer (bound>>unbound)?\n",
          flush=True)
    rows = [run_seed(s, D) for s in seeds for D in dims]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    ans, ba, ub = m("answer_acc"), m("bound_above_thr"), m("unbound_below_thr")
    n_sep = sum(r["sep"] for r in rows)
    go = (ans >= 0.99) and (ba >= 0.99) and (ub >= 0.99) and (n_sep == len(rows))
    print(f"\n{'='*96}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D): answer {ans:.3f} | bound>thr {ba:.3f} | unbound<thr {ub:.3f} | "
          f"clean-separation {n_sep}/{len(rows)}", flush=True)
    if go:
        print(f"  GO: the on-bridge cleanup PEAK score IS the moat -- bound roles score high (answer), unbound roles "
              f"score low (abstain), clean separation every seed/D, answer 100%. ==> the no-confab moat reads off "
              f"the substrate (a neural familiarity signal), no host equality check. Next: the spiking WTA + the "
              f"parser front-end to close the full who/what turn on one bridge.", flush=True)
    else:
        print(f"  BOUNDARY: peak-score moat not clean (answer {ans:.3f}, bound>thr {ba:.3f}, unbound<thr {ub:.3f}, "
              f"sep {n_sep}/{len(rows)}) -- the bound/unbound peak gap is seed-fragile; the familiarity gate needs "
              f"the validated Bogacz-Brown contrast, not a raw peak threshold.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*96}", flush=True)
    out = {"verdict": "GO" if go else "BOUNDARY", "seeds": seeds, "dims": dims, "answer": ans, "bound_above": ba,
           "unbound_below": ub, "clean_sep": n_sep, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
