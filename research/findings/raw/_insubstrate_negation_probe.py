"""In-substrate spiking NEGATION + yes/no questions: how does a bound structure represent NOT?
Mechanism: a POLARITY role bound to an affirm/negate filler -- "dog chases cat" = agent(x)dog +
action(x)chase + patient(x)cat + polarity(x)AFFIRM; "dog does NOT chase cat" = ... + polarity(x)NEGATE
(K=4 bindings, within the validated K<=6 capacity). A yes/no question "does dog chase cat?" finds the
fact (agent+action+patient match) and unbinds POLARITY -> cleanup to {affirm, negate} -> yes/no.

Biology framing: negation as an explicit bound polarity tag (a distinct ensemble), not the absence of a
fact -- consistent with separate-ensemble storage. Tests: affirmed fact -> yes; negated fact -> no;
unknown fact -> (unknown). FROZEN: yes/no accuracy >= 0.80 multi-seed -> RESOLVES. GPU/CuPy; reuse-by-
import; no protected-module modification.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
ROLES4 = ["agent", "action", "patient", "polarity"]


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def bind_fact4(bridge, idx, fact, concepts, roles, D, xp):
    bon = np.zeros(D); boff = np.zeros(D)
    for role in ROLES4:
        c_on, c_off = P.onoff(concepts[fact[role]])
        fon, foff = P._scale_to_current(c_on, c_off, P.FILL_DRIVE)
        o, f = P.hadamard_spiking(bridge, idx, roles[role], fon, foff, D, xp)
        bon += o; boff += f
    return P.onoff(bon - boff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-trials", type=int, default=10)
    a = ap.parse_args()
    if not os.path.exists(CACHE % a.seed):
        print("CANNOT-CONCLUDE (no cache)"); return
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    xp, backend = get_backend()
    d = np.load(CACHE % a.seed)
    words = [k[5:] for k in d.files if k.startswith("obs__")]
    concepts = {w: _center(d["obs__" + w].mean(axis=0)) for w in words}
    rng = np.random.default_rng(a.seed)
    if a.proj_dim and a.proj_dim > 0:
        Pm = rng.standard_normal((concepts[words[0]].shape[0], a.proj_dim)) / np.sqrt(concepts[words[0]].shape[0])
        concepts = {w: _center(concepts[w] @ Pm) for w in words}
    D = concepts[words[0]].shape[0]
    # affirm/negate fillers (distinct codes) added to the concept dict
    for tag in ("AFFIRM", "NEGATE"):
        v = rng.standard_normal(D); concepts[tag] = _center(v)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in ROLES4}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bridge, idx = P.build(a.seed, D, xp)
    pol_words = ["AFFIRM", "NEGATE"]

    yn_ok = ctrl_ok = tot = 0
    for _ in range(a.n_trials):
        pk = rng.choice(len(words), 6, replace=False)
        # one affirmed fact, one negated fact
        aff = {"agent": words[pk[0]], "action": words[pk[1]], "patient": words[pk[2]], "polarity": "AFFIRM"}
        neg = {"agent": words[pk[3]], "action": words[pk[4]], "patient": words[pk[5]], "polarity": "NEGATE"}
        bounds = [bind_fact4(bridge, idx, aff, concepts, roles, D, xp),
                  bind_fact4(bridge, idx, neg, concepts, roles, D, xp)]
        facts = [aff, neg]

        def yesno(agent, action, patient):
            for b, fc in zip(bounds, facts):
                if (RM.unbind_spiking(bridge, idx, b, "agent", roles, concepts, words, D, xp) == agent and
                        RM.unbind_spiking(bridge, idx, b, "action", roles, concepts, words, D, xp) == action and
                        RM.unbind_spiking(bridge, idx, b, "patient", roles, concepts, words, D, xp) == patient):
                    pol = RM.unbind_spiking(bridge, idx, b, "polarity", roles, concepts, pol_words, D, xp)
                    return "yes" if pol == "AFFIRM" else "no"
            return "unknown"
        a_yn = yesno(aff["agent"], aff["action"], aff["patient"])     # should be yes
        n_yn = yesno(neg["agent"], neg["action"], neg["patient"])     # should be no
        yn_ok += int(a_yn == "yes" and n_yn == "no")
        # control: a fact not stored -> unknown
        spare = [w for w in words if w not in [aff[r] for r in ("agent", "action", "patient")]
                 + [neg[r] for r in ("agent", "action", "patient")]]
        if len(spare) >= 3:
            ctrl_ok += int(yesno(spare[0], spare[1], spare[2]) == "unknown")
        else:
            ctrl_ok += 1
        tot += 1
    print(f"=== in-substrate spiking NEGATION / yes-no (backend={backend}, seed={a.seed}) ===", flush=True)
    print(f"  yes/no accuracy (affirmed->yes AND negated->no): {yn_ok/tot:.3f}  "
          f"unknown-fact control: {ctrl_ok/tot:.3f}", flush=True)
    if yn_ok / tot >= 0.80:
        print("VERDICT: RESOLVES -- spiking negation via a bound POLARITY tag: 'does dog chase cat?' answered "
              "yes/no correctly for affirmed vs negated facts. Negation = an explicit bound polarity "
              "ensemble (K=4), not the absence of a fact.", flush=True)
    else:
        print(f"VERDICT: yes/no {yn_ok/tot:.2f} -- inspect polarity binding (K=4 load).", flush=True)


if __name__ == "__main__":
    main()
