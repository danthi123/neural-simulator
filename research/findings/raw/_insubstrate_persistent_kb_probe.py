"""In-substrate spiking PERSISTENT knowledge base across SESSIONS: the artificial-life / continual-
learning dimension. An agent stores SVO facts (separate spiking bound structures), PERSISTS them to
disk, reloads in a fresh "session", ADDS new facts, and answers questions across the ACCUMULATED KB --
with NO forgetting of earlier facts (separate-fact storage means a new fact cannot disturb prior ones).

Composes validated pieces: the spiking bind (role x filler) + relational query + the project's premise
(continual learning without catastrophic forgetting). The bound vectors are self-contained rate codes;
roles + concept codes regenerate deterministically from --seed, so a reload is exact.

Protocol: SESSION 1 stores facts {A,B}, saves the bound vectors. SESSION 2 (fresh bridge) loads them,
ADDS fact C, then answers a question about EACH of A, B (persisted), C (new). FROZEN: all three answered
(session-1 facts SURVIVE + session-2 fact works) -> RESOLVES. GPU/CuPy; reuse-by-import; no protected
module modification. Only float arrays are persisted (np.savez, no pickle).
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
KB_PATH = "research/findings/raw/_persistent_kb_seed%d.npz"


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_concepts(seed):
    d = np.load(CACHE % seed)
    ws = [k[5:] for k in d.files if k.startswith("obs__")]
    return ws, {w: _center(d["obs__" + w].mean(axis=0)) for w in ws}


def setup(seed, proj_dim, xp):
    words, concepts = load_concepts(seed)
    rng = np.random.default_rng(seed)
    if proj_dim and proj_dim > 0:
        Pm = rng.standard_normal((concepts[words[0]].shape[0], proj_dim)) / np.sqrt(concepts[words[0]].shape[0])
        concepts = {w: _center(concepts[w] @ Pm) for w in words}
    D = concepts[words[0]].shape[0]
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}   # deterministic from seed
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bridge, idx = P.build(seed, D, xp)
    return words, concepts, roles, bridge, idx, D


def query_who(bridge, idx, bounds, action, patient, roles, concepts, words, D, xp):
    for b in bounds:
        if (RM.unbind_spiking(bridge, idx, b, "action", roles, concepts, words, D, xp) == action and
                RM.unbind_spiking(bridge, idx, b, "patient", roles, concepts, words, D, xp) == patient):
            return RM.unbind_spiking(bridge, idx, b, "agent", roles, concepts, words, D, xp)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    a = ap.parse_args()
    if not os.path.exists(CACHE % a.seed):
        print("CANNOT-CONCLUDE (no cache)"); return
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    xp, backend = get_backend()
    print(f"=== in-substrate spiking PERSISTENT KB across sessions (backend={backend}, seed={a.seed}) ===",
          flush=True)

    # ----- SESSION 1: store facts A, B; persist the bound vectors -----
    words, concepts, roles, bridge, idx, D = setup(a.seed, a.proj_dim, xp)
    rng = np.random.default_rng(a.seed + 7)
    pk = rng.choice(len(words), 9, replace=False)
    A = {"agent": words[pk[0]], "action": words[pk[1]], "patient": words[pk[2]]}
    B = {"agent": words[pk[3]], "action": words[pk[4]], "patient": words[pk[5]]}
    C = {"agent": words[pk[6]], "action": words[pk[7]], "patient": words[pk[8]]}
    bA = RM.bind_fact_spiking(bridge, idx, A, concepts, roles, D, xp)
    bB = RM.bind_fact_spiking(bridge, idx, B, concepts, roles, D, xp)
    np.savez(KB_PATH % a.seed, on_0=bA[0], off_0=bA[1], on_1=bB[0], off_1=bB[1])  # float arrays only
    print(f"  SESSION 1: stored 2 facts -> persisted to {KB_PATH % a.seed}", flush=True)
    print(f"    A = {A['agent']} {A['action']} {A['patient']}", flush=True)
    print(f"    B = {B['agent']} {B['action']} {B['patient']}", flush=True)
    del bridge

    # ----- SESSION 2: FRESH bridge; reload; add fact C; query A, B, C -----
    words, concepts, roles, bridge, idx, D = setup(a.seed, a.proj_dim, xp)
    d = np.load(KB_PATH % a.seed)
    bounds = [(d["on_0"], d["off_0"]), (d["on_1"], d["off_1"])]      # persisted facts
    print(f"  SESSION 2 (fresh bridge): reloaded {len(bounds)} facts; adding 1 new fact C", flush=True)
    bC = RM.bind_fact_spiking(bridge, idx, C, concepts, roles, D, xp)
    bounds.append(bC)
    print(f"    C = {C['agent']} {C['action']} {C['patient']}  (new this session)", flush=True)

    ans_A = query_who(bridge, idx, bounds, A["action"], A["patient"], roles, concepts, words, D, xp)
    ans_B = query_who(bridge, idx, bounds, B["action"], B["patient"], roles, concepts, words, D, xp)
    ans_C = query_who(bridge, idx, bounds, C["action"], C["patient"], roles, concepts, words, D, xp)
    okA, okB, okC = ans_A == A["agent"], ans_B == B["agent"], ans_C == C["agent"]
    print(f"    Q 'who {A['action']} {A['patient']}?' (session-1 fact) -> {ans_A}  [{'OK' if okA else 'MISS'}]",
          flush=True)
    print(f"    Q 'who {B['action']} {B['patient']}?' (session-1 fact) -> {ans_B}  [{'OK' if okB else 'MISS'}]",
          flush=True)
    print(f"    Q 'who {C['action']} {C['patient']}?' (session-2 fact) -> {ans_C}  [{'OK' if okC else 'MISS'}]",
          flush=True)
    if okA and okB and okC:
        print("VERDICT: RESOLVES -- a PERSISTENT spiking knowledge base: facts stored in session 1 SURVIVE "
              "a reload in a fresh session (no forgetting), and a new fact added in session 2 also answers "
              "-- continual accumulation across sessions, the artificial-life premise.", flush=True)
    else:
        print(f"VERDICT: persistence/recall {int(okA)+int(okB)+int(okC)}/3 -- inspect save/reload.", flush=True)


if __name__ == "__main__":
    main()
