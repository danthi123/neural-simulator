"""Conversation at 64 words: the spiking relational fact-memory + wh-QA using SPARSE-DISTRIBUTED
concept codes (the G.20 front-end's validated 100%-per-bridge tier: 64 concepts/bridge). Grounds the
honest scaling answer -- the bind/COMPOSITION is vocabulary-robust; the real ceiling is the recognition
front-end, which is clean at 64/bridge. So a 64-word conversation is feasible; this demonstrates it.

Stores K SVO facts over a 64-word vocab (sparse K-of-N codes), answers wh-questions (who/what-object/
what-action), checks an unknown-question control. FROZEN: QA >= 0.80 multi-seed at V=64 -> RESOLVES
(4x the 16-word tier). GPU/CuPy; reuse-by-import; no protected-module modification.
"""
from __future__ import annotations
import argparse
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend


def gen_sparse(V, N, Kp, rng):
    """V sparse K-of-N concept codes, mean-centered (near-orthogonal -> easy cleanup, the G.20 regime)."""
    C = {}
    for i in range(V):
        v = np.zeros(N); v[rng.choice(N, Kp, replace=False)] = 1.0
        v = v - v.mean(); v /= np.linalg.norm(v) + 1e-12
        C[f"w{i:02d}"] = v
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--n-dim", type=int, default=2000)
    ap.add_argument("--k-sparse", type=int, default=100)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--n-facts", type=int, default=2)
    a = ap.parse_args()
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    xp, backend = get_backend()
    rng = np.random.default_rng(a.seed)
    concepts = gen_sparse(a.vocab, a.n_dim, a.k_sparse, rng)
    words = list(concepts.keys()); D = a.n_dim
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    print(f"=== conversation at {a.vocab} words (sparse codes, backend={backend}, seed={a.seed}, "
          f"D={D}, 8D={8*D} neurons) ===", flush=True)
    bridge, idx = P.build(a.seed, D, xp)

    def q(bounds, given, query_role):
        for b in bounds:
            if all(RM.unbind_spiking(bridge, idx, b, r, roles, concepts, words, D, xp) == w
                   for r, w in given.items()):
                return RM.unbind_spiking(bridge, idx, b, query_role, roles, concepts, words, D, xp)
        return None

    qa_ok = ctrl_ok = tot = 0
    for _ in range(a.n_trials):
        pk = rng.choice(len(words), 3 * a.n_facts, replace=False)
        facts = [{"agent": words[pk[3*f]], "action": words[pk[3*f+1]], "patient": words[pk[3*f+2]]}
                 for f in range(a.n_facts)]
        bounds = [RM.bind_fact_spiking(bridge, idx, fc, concepts, roles, D, xp) for fc in facts]
        f = facts[rng.integers(a.n_facts)]
        who = q(bounds, {"action": f["action"], "patient": f["patient"]}, "agent")
        wob = q(bounds, {"agent": f["agent"], "action": f["action"]}, "patient")
        wac = q(bounds, {"agent": f["agent"], "patient": f["patient"]}, "action")
        qa_ok += int(who == f["agent"] and wob == f["patient"] and wac == f["action"])
        used = set(w for fc in facts for w in fc.values())
        spare = [w for w in words if w not in used]
        ctrl_ok += int(q(bounds, {"action": spare[0], "patient": spare[1]}, "agent") is None)
        tot += 1
    print(f"  QA at V={a.vocab} (who/what-obj/what-act all correct): {qa_ok/tot:.3f}  "
          f"unknown control: {ctrl_ok/tot:.3f}  (chance {1.0/a.vocab:.4f})", flush=True)
    if qa_ok / tot >= 0.80:
        print(f"VERDICT: RESOLVES -- wh-question answering works at {a.vocab} words (sparse codes) "
              f"in-substrate. The bind/composition handles the larger vocabulary; the front-end (clean at "
              f"64/bridge) is the real ceiling.", flush=True)
    else:
        print(f"VERDICT: QA {qa_ok/tot:.2f} at V={a.vocab}.", flush=True)


if __name__ == "__main__":
    main()
