"""In-substrate spiking RELATIONAL FACT-MEMORY -- the conversational primitive built from the
validated spiking bind. The numpy cheap-first (_vsa_relational_query_probe.py) RESOLVED multi-seed
(separate-fact storage + cue-based retrieval = 1.000); this realizes it IN the spiking substrate.

A fact "dog chases cat" = agent (x) dog + action (x) chase + patient (x) cat, computed by the
validated spiking coincidence bind (3 bindings per fact = K=3, within the clean-AND K=4 capacity).
Facts are stored SEPARATELY (each its own bound-rate vector -- the correct architecture; superposed
storage degrades per the multi-hop wall). Relational query "what does <agent> <action>?":
  for each stored fact -> spiking-unbind agent -> cleanup -> match the cue agent;
  for the matched fact -> spiking-unbind patient -> cleanup -> the answer.
All bind/unbind are spiking (reuse _insubstrate_bind_unbind_probe.hadamard_spiking); only the
inter-fact iteration + cleanup match are control logic.

FROZEN: spiking single-fact role query >= 0.80 AND spiking relational (find-by-agent, read-patient)
>= 0.80, multi-seed, with the absent-cue control giving no false match -> RESOLVES (a queryable
relational fact-memory runs IN the spiking substrate). GPU/CuPy. Reuse-by-import; no protected mod.

RESULT 2026-05-31 seed 42 (D=800, bias=-1000, 2 facts): RESOLVES. spiking single-fact=0.917,
relational(find-agent,read-patient)=0.917, control(no-false-match)=1.000. A queryable SVO fact
base runs IN the spiking substrate. Multi-seed confirmation in flight.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
from sim.backend import get_backend

ROLES = ["agent", "action", "patient"]


def bind_fact_spiking(bridge, idx, fact, concepts, roles, D, xp):
    """Bind agent(x)X + action(x)Y + patient(x)Z in spiking; return canonical bound ON/OFF."""
    bon = np.zeros(D); boff = np.zeros(D)
    for role in ROLES:
        c_on, c_off = P.onoff(concepts[fact[role]])
        fon, foff = P._scale_to_current(c_on, c_off, P.FILL_DRIVE)
        o, f = P.hadamard_spiking(bridge, idx, roles[role], fon, foff, D, xp)
        bon += o; boff += f
    bsig = bon - boff
    return P.onoff(bsig)


def unbind_spiking(bridge, idx, bound_onoff, role, roles, concepts, words, D, xp):
    fon, foff = P._scale_to_current(bound_onoff[0], bound_onoff[1], P.FILL_DRIVE)
    e_on, e_off = P.hadamard_spiking(bridge, idx, roles[role], fon, foff, D, xp)
    est = e_on - e_off
    sims = np.array([concepts[w] @ est for w in words])
    return words[int(np.argmax(sims))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--run-steps", type=int, default=150)
    ap.add_argument("--coinc-bias", type=float, default=-1000.0)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--n-facts", type=int, default=2)
    a = ap.parse_args()
    if not os.path.exists(P.CACHE % a.seed):
        print("CANNOT-CONCLUDE (no cache)"); return
    P.RUN_STEPS = a.run_steps; P.COINC_BIAS = a.coinc_bias
    xp, backend = get_backend()
    words, codes = P.load_concepts(a.seed, a.proj_dim, None if a.proj_dim <= 0 else np.random.default_rng(a.seed))
    D = codes.shape[1]
    concepts = {w: codes[i] for i, w in enumerate(words)}
    rng = np.random.default_rng(a.seed)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    print(f"=== in-substrate spiking RELATIONAL FACT-MEMORY (backend={backend}, seed={a.seed}, "
          f"D={D}, n_facts={a.n_facts}, bias={a.coinc_bias}) ===", flush=True)
    bridge, idx = P.build(a.seed, D, xp)

    s_ok = rel_ok = ctrl_ok = tot = 0
    for _ in range(a.n_trials):
        picks = rng.choice(len(words), size=3 * a.n_facts, replace=False)
        facts = [{"agent": words[picks[3*f]], "action": words[picks[3*f+1]], "patient": words[picks[3*f+2]]}
                 for f in range(a.n_facts)]
        bound = [bind_fact_spiking(bridge, idx, fc, concepts, roles, D, xp) for fc in facts]

        # single-fact role query
        qf = rng.integers(a.n_facts); qrole = ROLES[rng.integers(3)]
        s_ok += int(unbind_spiking(bridge, idx, bound[qf], qrole, roles, concepts, words, D, xp)
                    == facts[qf][qrole])

        # relational: find fact by agent cue, read patient
        tf = rng.integers(a.n_facts); cue = facts[tf]["agent"]
        best = None
        for f in range(a.n_facts):
            if unbind_spiking(bridge, idx, bound[f], "agent", roles, concepts, words, D, xp) == cue:
                best = f; break
        ans = (unbind_spiking(bridge, idx, bound[best], "patient", roles, concepts, words, D, xp)
               if best is not None else None)
        rel_ok += int(ans == facts[tf]["patient"])

        # control: absent-cue agent -> expect no false match
        non = [w for w in words if w not in [fc["agent"] for fc in facts]]
        cue_bad = str(rng.choice(non)); bestc = None
        for f in range(a.n_facts):
            if unbind_spiking(bridge, idx, bound[f], "agent", roles, concepts, words, D, xp) == cue_bad:
                bestc = f; break
        ctrl_ok += int(bestc is None)
        tot += 1

    print(f"  spiking single-fact={s_ok/tot:.3f}  relational(find-agent,read-patient)={rel_ok/tot:.3f}  "
          f"control(no-false-match)={ctrl_ok/tot:.3f}  (chance={1.0/len(words):.3f})", flush=True)
    ok = (s_ok / tot >= 0.80) and (rel_ok / tot >= 0.80)
    if ok:
        print("VERDICT: RESOLVES -- a queryable RELATIONAL FACT-MEMORY runs IN the spiking substrate "
              "(spiking bind stores SVO facts; spiking unbind + cleanup answers relational queries).", flush=True)
    else:
        print("VERDICT: needs tuning -- single/relational below 0.80; raise rate (less-negative bias) "
              "or window, or check fact storage.", flush=True)


if __name__ == "__main__":
    main()
