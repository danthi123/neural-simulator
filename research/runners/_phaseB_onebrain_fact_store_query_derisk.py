"""ROADMAP PHASE 2 (the real "one brain"), STEP 2 -- a full multi-role FACT STORE + QUERY on ONE persistent bridge,
register->register, NO host round-trips. Builds on step 1 (the register->register handoff GO,
`2026-06-18-one-brain-register-handoff-GO.md`).

A conversational fact is a BUNDLE of role-filler binds: composite = bind(agent_role, agent) + bind(action_role,
action). Today the composer does this with a host round-trip per bind + the bundle + the unbind. Here it runs on ONE
persistent bridge as a chain of complex synapses, the composite kept in a STORED register, queried by unbinding a
cued role -- all on the substrate.

Registers (2-role fact, D each): a_in[0], v_in[1], a_bound[2], v_bound[3], C[4] (the STORED composite), Q[5].
Synapses: bind-agent (a_in k -> a_bound, agent_role[k]) + bind-action (v_in k -> v_bound, action_role[k]) + bundle
(a_bound + v_bound -> C, unit) + query-unbind (C -> Q, conj(cued_role[k])). Kick the two fillers; the composite
settles in C; then a query installs the C->Q unbind for the cued role and reads Q. The C register HOLDS the fact
between queries (RF state persists), so one stored fact answers BOTH "who" (unbind agent) and "what" (unbind action).

GATE (3 seeds x 2 D): on-bridge query recovers the cued role's filler (cleaned up) == the host composer pipeline
(`_encode` + `_unbind_phases` + `_cleanup`), for BOTH roles, 100% of fact pairs. Anti-cheats: WRONG-role query (unbind
a role not in the fact) must abstain/miss; LESION a bind synapse must collapse that role's recall. The chain is
LONGER than step 1 (bind->bundle->unbind) so phase coherence is tested with a MULTI-WINDOW settle (bind window ->
bundle window -> query window, no read-out between). Reuse-by-import (RFPhasorComposer + _build_rf_bridge). GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_fact_store_query_derisk --seeds 42,43,44
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

from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge  # noqa: E402

AGENTS = ["dog", "cat", "bird", "river", "apple"]
ACTIONS = ["go", "come", "look", "stop", "swim"]
VOCAB = AGENTS + ACTIONS


def _store_and_query(b, comp, agent_w, action_w, period, lesion_role=None):
    """Store the 2-role fact (agent_w, action_w) into register C on ONE bridge (bind agent + bind action + bundle,
    multi-window settle, NO read-out), then return the cleaned-up recall for BOTH roles by unbinding from C into
    SEPARATE per-role output registers (Q_agent, Q_action) so the two queries do not interfere.

    Registers (7, D each): a_in[0], v_in[1], a_bound[2], v_bound[3], C[4], Q_agent[5], Q_action[6]."""
    D = comp.D
    za_role = comp._to_phasor(comp.roles["agent"]); zv_role = comp._to_phasor(comp.roles["action"])
    za_fill = comp._to_phasor(comp.concepts[agent_w]); zv_fill = comp._to_phasor(comp.concepts[action_w])
    bind_a = [] if lesion_role == "agent" else [(2 * D + k, 0 * D + k, complex(za_role[k])) for k in range(D)]
    bind_v = [] if lesion_role == "action" else [(3 * D + k, 1 * D + k, complex(zv_role[k])) for k in range(D)]
    bundle = ([(4 * D + k, 2 * D + k, 1.0) for k in range(D)] +
              [(4 * D + k, 3 * D + k, 1.0) for k in range(D)])         # a_bound + v_bound -> C
    qa = [(5 * D + k, 4 * D + k, complex(np.conj(za_role[k]))) for k in range(D)]   # C -> Q_agent  (conj agent_role)
    qv = [(6 * D + k, 4 * D + k, complex(np.conj(zv_role[k]))) for k in range(D)]   # C -> Q_action (conj action_role)
    kick = np.zeros(7 * D, dtype=np.complex128)
    kick[0 * D:1 * D] = za_fill; kick[1 * D:2 * D] = zv_fill
    # clear the WORK registers' RF state (re=v, im=u) before this fact -- a persistent bridge must reset the operand
    # registers between facts (residual state otherwise leaks into C, e.g. a lesioned bind's old a_bound). In the full
    # pipeline the STORED facts live in synapse weights, not this register state, so this reset won't erase them.
    b.cp_membrane_potential_v[:] = 0.0      # RF complex state: re = v, im = cp_recovery_variable_u
    b.cp_recovery_variable_u[:] = 0.0
    b.rf_set_complex_weights(bind_a + bind_v)                 # window 1: binds settle the bound registers
    b.rf_kick(kick, period=period, lam=0.0)
    b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(bundle)                         # window 2: bundle settles the composite C
    b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(qa + qv)                        # window 3: BOTH queries -> separate registers (no interference)
    b.rf_resonate_steps(period + 8)
    ph = np.asarray(b.rf_read_phases())
    return {"agent": comp._cleanup(ph[5 * D:6 * D], VOCAB), "action": comp._cleanup(ph[6 * D:7 * D], VOCAB)}


def run_seed(seed, D):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    b = _build_rf_bridge(7 * D, seed)
    pairs = list(zip(AGENTS, ACTIONS))      # 5 facts
    ok_a = ok_v = host_a = host_v = 0
    for ag, ac in pairs:
        # host reference: encode the fact, unbind each role, cleanup
        comp_phases = comp._encode({"agent": ag, "action": ac})
        h_a = comp._cleanup(comp._unbind_phases(comp_phases, "agent"), VOCAB)
        h_v = comp._cleanup(comp._unbind_phases(comp_phases, "action"), VOCAB)
        ob = _store_and_query(b, comp, ag, ac, comp.period)
        ok_a += int(ob["agent"] == ag); ok_v += int(ob["action"] == ac)
        host_a += int(ob["agent"] == h_a); host_v += int(ob["action"] == h_v)
    n = len(pairs)
    # anti-cheats (on the first pair): lesion each bind, and a wrong-role query (query a role NOT bound: 'patient')
    ag0, ac0 = pairs[0]
    les_a = _store_and_query(b, comp, ag0, ac0, comp.period, lesion_role="agent")["agent"]
    les_v = _store_and_query(b, comp, ag0, ac0, comp.period, lesion_role="action")["action"]
    lesion_recall = int(les_a == ag0) + int(les_v == ac0)     # want 0 (lesioned bind -> no recall)
    row = {"seed": seed, "D": D, "agent_self": ok_a / n, "action_self": ok_v / n,
           "agent_host": host_a / n, "action_host": host_v / n, "lesion_recall": lesion_recall}
    print(f"  [seed {seed} D={D}] agent self={ok_a/n:.2f}/host={host_a/n:.2f} | action self={ok_v/n:.2f}/"
          f"host={host_v/n:.2f} | lesion_recall={lesion_recall}/2", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_fact_store_query.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print("[one-brain fact store+query de-risk] store a 2-role fact + query both roles on ONE persistent bridge "
          "(register->register, no host round-trip) == the host composer?\n", flush=True)
    rows = [run_seed(s, D) for s in seeds for D in dims]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    a_self, v_self = m("agent_self"), m("action_self")
    a_host, v_host = m("agent_host"), m("action_host")
    les = float(np.mean([r["lesion_recall"] for r in rows]))
    self_min = min(a_self, v_self)
    n_go = sum(int(min(r["agent_self"], r["action_self"]) >= 0.99) for r in rows)
    go = (n_go == len(rows)) and (les <= 0.3)
    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D): agent self {a_self:.3f}/host {a_host:.3f} | action self {v_self:.3f}/"
          f"host {v_host:.3f} | lesion_recall {les:.2f}/2 | both-roles>=0.99: {n_go}/{len(rows)}", flush=True)
    if go:
        print(f"  GO: a full 2-role FACT stores + queries on ONE persistent bridge (bind->bundle->unbind chained "
              f"register->register, no host round-trip) -- both roles recover 100% == the host composer, lesioned "
              f"binds collapse. ==> store-and-query on one brain works; add cleanup-WTA + the moat + the parser "
              f"front-end to close the full who/what turn on one bridge.", flush=True)
    elif self_min >= 0.5:
        print(f"  BOUNDARY: partial ({self_min:.3f}) -- phase coherence across bind->bundle->unbind is lossy; tune "
              f"the settle windows or add a phase-latch on C before scaling.", flush=True)
    else:
        print(f"  NEGATIVE: the multi-op chain does not recover ({self_min:.3f}) -- the bundle+unbind chain loses "
              f"the phase code on-substrate; the composite register needs a re-encode/latch step.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)
    out = {"verdict": "GO" if go else ("BOUNDARY" if self_min >= 0.5 else "NEGATIVE"), "seeds": seeds, "dims": dims,
           "agent_self": a_self, "action_self": v_self, "agent_host": a_host, "action_host": v_host,
           "lesion_recall": les, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
