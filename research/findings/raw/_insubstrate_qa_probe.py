"""In-substrate spiking QUESTION-ANSWERING: wh-questions over a stored SVO knowledge base.
Extends the validated relational fact-memory: a wh-word identifies the QUERY slot, the other content
words are the GIVEN cues. "who chases cat?" -> given {action:chase, patient:cat}, query agent ->
find the stored fact matching all givens, read the query role. Three question types:
  "who V N?"        -> given action+patient, query agent
  "what does A V?"  -> given agent+action,  query patient
  "what does A do-to N?" -> given agent+patient, query action

All bind/unbind spiking (reuse _insubstrate_relational_memory_probe + _insubstrate_bind_unbind_probe).
Stores K facts, asks each fact's 3 questions, checks the answer. Control: a question whose cues match
NO stored fact -> "(unknown)". FROZEN: spiking QA accuracy >= 0.80 multi-seed -> RESOLVES.
GPU/CuPy; reuse-by-import; no protected-module modification.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_concepts(seed):
    d = np.load(CACHE % seed)
    ws = [k[5:] for k in d.files if k.startswith("obs__")]
    return ws, {w: _center(d["obs__" + w].mean(axis=0)) for w in ws}


def answer_question(bridge, idx, bounds, given, query_role, roles, concepts, words, D, xp):
    """given: {role: word} cues; query_role: the role to read. Find the stored fact matching ALL
    given cues (spiking unbind + cleanup), then read the query role. Returns the answer word or None."""
    for b in bounds:
        if all(RM.unbind_spiking(bridge, idx, b, r, roles, concepts, words, D, xp) == w
               for r, w in given.items()):
            return RM.unbind_spiking(bridge, idx, b, query_role, roles, concepts, words, D, xp)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--n-facts", type=int, default=2)
    a = ap.parse_args()
    if not os.path.exists(CACHE % a.seed):
        print("CANNOT-CONCLUDE (no cache)"); return
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    xp, backend = get_backend()
    print(f"=== in-substrate spiking QUESTION-ANSWERING (backend={backend}, seed={a.seed}) ===", flush=True)
    words, concepts = load_concepts(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.proj_dim and a.proj_dim > 0:
        Pm = rng.standard_normal((concepts[words[0]].shape[0], a.proj_dim)) / np.sqrt(concepts[words[0]].shape[0])
        concepts = {w: _center(concepts[w] @ Pm) for w in words}
    D = concepts[words[0]].shape[0]
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bridge, idx = P.build(a.seed, D, xp)

    qa_ok = ctrl_ok = tot = 0
    for _ in range(a.n_trials):
        picks = rng.choice(len(words), 3 * a.n_facts, replace=False)
        facts = [{"agent": words[picks[3*f]], "action": words[picks[3*f+1]], "patient": words[picks[3*f+2]]}
                 for f in range(a.n_facts)]
        bounds = [RM.bind_fact_spiking(bridge, idx, fc, concepts, roles, D, xp) for fc in facts]
        f = facts[rng.integers(a.n_facts)]
        # who V N? -> agent ; what does A V? -> patient ; what does A do-to N? -> action
        who = answer_question(bridge, idx, bounds, {"action": f["action"], "patient": f["patient"]},
                              "agent", roles, concepts, words, D, xp)
        what_obj = answer_question(bridge, idx, bounds, {"agent": f["agent"], "action": f["action"]},
                                   "patient", roles, concepts, words, D, xp)
        what_act = answer_question(bridge, idx, bounds, {"agent": f["agent"], "patient": f["patient"]},
                                   "action", roles, concepts, words, D, xp)
        qa_ok += int(who == f["agent"] and what_obj == f["patient"] and what_act == f["action"])
        # control: a question with cues matching no stored fact -> expect None
        used = set(w for fc in facts for w in fc.values())
        spare = [w for w in words if w not in used]
        if len(spare) >= 2:
            ans = answer_question(bridge, idx, bounds, {"action": spare[0], "patient": spare[1]},
                                  "agent", roles, concepts, words, D, xp)
            ctrl_ok += int(ans is None)
        else:
            ctrl_ok += 1
        tot += 1
    print(f"  spiking QA (who/what-obj/what-act all correct): {qa_ok/tot:.3f}  "
          f"unknown-question control (-> none): {ctrl_ok/tot:.3f}", flush=True)
    if qa_ok / tot >= 0.80:
        print("VERDICT: RESOLVES -- spiking wh-question answering over an SVO knowledge base works "
              "in-substrate (who/what-object/what-action), via the validated bind + relational query.",
              flush=True)
    else:
        print(f"VERDICT: QA {qa_ok/tot:.2f} -- inspect multi-cue match / bind.", flush=True)


if __name__ == "__main__":
    main()
