"""Cheap-first (CPU, numpy algebra): can a HIERARCHICAL code give 320 DISTINCT composable concept codes
WITHOUT retraining the 5 shared-pattern bridges?

The 320-tier's 5 bridges share seed-42 sparse patterns, so bridgeA-i and bridgeB-i have near-identical codes
(the documented duplicate problem) -> a global 320-way spiking cleanup fails. Proposed fix, no retrain:
bind each concept with its bridge's ROLE vector -> concept_code = bridge_role (Hadamard) within_code. Since
the 5 bridge-roles are near-orthogonal, (roleA*p_i) . (roleB*p_i) = (roleA.roleB)*|p_i|^2 ~ 0, so cross-
bridge same-index concepts become DISTINCT. This is the cheap-first: (1) do the hierarchical codes stay
distinct (low between-cos)? (2) does the relational bind/QA compose + abstain over all 320?

numpy algebra (the qa64 ceiling), no GPU, no spiking -- a viability gate before any spiking build.
Run: python -m research.findings.raw._hierarchical320_cheap_probe
"""
from __future__ import annotations
import numpy as np

D = 2000; N_BRIDGES = 5; PER_BRIDGE = 64; V = N_BRIDGES * PER_BRIDGE  # 320
PATTERN_SIZE = 100
ROLES = ["agent", "action", "patient"]


def _mc(v):
    v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-12)


def main():
    rng = np.random.default_rng(42)
    # 64 shared within-bridge sparse patterns (same seed across bridges = the duplicate cause)
    within = []
    wrng = np.random.RandomState(42 * 17 + 19)
    for _ in range(PER_BRIDGE):
        p = np.zeros(D); p[wrng.choice(D, PATTERN_SIZE, replace=False)] = 1.0
        within.append(_mc(p))
    # 5 near-orthogonal bridge-role vectors (distributed +-1)
    bridge_roles = [rng.choice([-1.0, 1.0], size=D) / np.sqrt(D) for _ in range(N_BRIDGES)]

    # FLAT codes (current: cross-bridge duplicates) vs HIERARCHICAL codes (bridge_role (Hadamard) within)
    words, flat, hier = [], {}, {}
    for b in range(N_BRIDGES):
        for i in range(PER_BRIDGE):
            w = f"b{b}_w{i:02d}"; words.append(w)
            flat[w] = within[i]
            hier[w] = _mc(bridge_roles[b] * within[i])

    def maxcos(codes):
        M = np.stack([codes[w] for w in words]); G = M @ M.T
        off = ~np.eye(V, dtype=bool); return float(G[off].mean()), float(G[off].max())

    fm, fx = maxcos(flat); hm, hx = maxcos(hier)
    print(f"=== hierarchical-320 cheap-first (numpy) ===", flush=True)
    print(f"  FLAT codes:         between-cos mean {fm:.3f}  MAX {fx:.3f}  (max~1.0 = cross-bridge duplicates)",
          flush=True)
    print(f"  HIERARCHICAL codes: between-cos mean {hm:.3f}  MAX {hx:.3f}  (max<<1.0 = distinct -> composable)",
          flush=True)

    # relational bind/QA over the 320 HIERARCHICAL codes (numpy algebra): store SVO facts, query, abstain
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}

    def cleanup(vec):
        return words[int(np.argmax([float(vec @ hier[w]) for w in words]))]

    def bind_fact(f):
        return sum(roles[r] * hier[f[r]] for r in ROLES)  # Hadamard bind + superpose

    def unbind(bound, r):
        return cleanup(roles[r] * bound)   # unbind by re-binding the role, then cleanup

    qa_ok = ctrl_ok = tot = 0
    for _ in range(40):
        pk = rng.choice(V, 6, replace=False)
        facts = [{"agent": words[pk[3*f]], "action": words[pk[3*f+1]], "patient": words[pk[3*f+2]]}
                 for f in range(2)]
        bounds = [bind_fact(fc) for fc in facts]
        f = facts[rng.integers(2)]

        def q(given, qr):
            for b in bounds:
                if all(unbind(b, r) == w for r, w in given.items()):
                    return unbind(b, qr)
            return None
        who = q({"action": f["action"], "patient": f["patient"]}, "agent")
        qa_ok += int(who == f["agent"])
        used = set(w for fc in facts for w in fc.values()); spare = [w for w in words if w not in used]
        ctrl_ok += int(q({"action": spare[0], "patient": spare[1]}, "agent") is None)
        tot += 1
    qa, ctrl = qa_ok / tot, ctrl_ok / tot
    print(f"  HIERARCHICAL 320-way relational QA (who) = {qa:.3f}   abstention = {ctrl:.3f}  (chance {1/V:.4f})",
          flush=True)
    if hx < 0.5 and qa >= 0.80 and ctrl >= 0.80:
        print("VERDICT: VIABLE -- hierarchical bridge-role binding makes 320 codes distinct AND composable in "
              "the algebra, NO retrain. Worth a spiking build (bind the bridge-role in-substrate).", flush=True)
    elif hx < 0.5:
        print(f"VERDICT: codes DISTINCT (max {hx:.2f}) but QA {qa:.2f} -- the 2-level nesting may hit the "
              "multi-hop SNR wall; characterize before building.", flush=True)
    else:
        print(f"VERDICT: hierarchical codes still overlap (max {hx:.2f}) -- the role binding did not separate "
              "them; retrain with distinct seeds is the path.", flush=True)


if __name__ == "__main__":
    main()
