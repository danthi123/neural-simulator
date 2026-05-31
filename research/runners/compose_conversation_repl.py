"""Interactive CONVERSATIONAL AGENT REPL -- talk to the spiking brain-analogue agent.

Type statements to teach it facts, ask wh-questions or yes/no questions, and it answers -- every
operation is spiking compositional binding on real substrate concept codes. Its knowledge persists
to disk between runs (continual accumulation across sessions, the artificial-life premise).

Built on the validated 2026-05-31 composition arc (bind/unbind multi-seed + adversarial CLEAR;
relational fact-memory; wh-QA; persistent KB; negation polarity tag).

Grammar (16-word vocab: north/east/south/west/apple/river/dog/cat/go/come/stop/look/big/small/hot/cold):
  STATEMENT      "<agent> <action> <patient>"           e.g. "dog go north"   -> stored
  NEGATED        "<agent> not <action> <patient>"        e.g. "cat not come south"
  WHO-QUESTION   "who <action> <patient>?"               e.g. "who go north?"  -> agent
  WHAT-OBJECT    "what does <agent> <action>?"           e.g. "what does dog go?" -> patient
  YES/NO         "does <agent> <action> <patient>?"      e.g. "does dog go north?" -> yes/no
  COMMANDS       :facts  (list KB)   :save   :quit

Run:  python -m research.runners.compose_conversation_repl
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
import research.findings.raw._insubstrate_negation_probe as NEG
from sim.backend import get_backend

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
KB_STORE = "research/findings/raw/_repl_kb_seed%d.npz"


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--script", type=str, default=None, help="newline-or-;-separated lines (non-interactive)")
    a = ap.parse_args()
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
    for tag in ("AFFIRM", "NEGATE"):
        concepts[tag] = _center(rng.standard_normal(D))
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in NEG.ROLES4}     # agent/action/patient/polarity
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bridge, idx = P.build(a.seed, D, xp)
    KB = []  # (label, bound)

    def store(agent, action, patient, polarity="AFFIRM"):
        fact = {"agent": agent, "action": action, "patient": patient, "polarity": polarity}
        KB.append((f"{agent} {'' if polarity=='AFFIRM' else 'not '}{action} {patient}",
                   NEG.bind_fact4(bridge, idx, fact, concepts, roles, D, xp)))

    # reload persisted KB if present (float arrays only)
    if os.path.exists(KB_STORE % a.seed):
        kd = np.load(KB_STORE % a.seed)
        n = int(kd["n"])
        for i in range(n):
            KB.append((str(kd[f"lbl_{i}"]), (kd[f"on_{i}"], kd[f"off_{i}"])))
        print(f"[loaded {n} facts from a previous session]")

    def find(agent=None, action=None, patient=None):
        for lbl, b in KB:
            ok = True
            if agent is not None:
                ok = ok and RM.unbind_spiking(bridge, idx, b, "agent", roles, concepts, words, D, xp) == agent
            if action is not None:
                ok = ok and RM.unbind_spiking(bridge, idx, b, "action", roles, concepts, words, D, xp) == action
            if patient is not None:
                ok = ok and RM.unbind_spiking(bridge, idx, b, "patient", roles, concepts, words, D, xp) == patient
            if ok:
                return b
        return None

    def respond(line):
        t = line.strip().lower().rstrip("?").split()
        if not t:
            return ""
        if t[0] == ":facts":
            return "  " + "\n  ".join(lbl for lbl, _ in KB) if KB else "  (no facts yet)"
        if t[0] == ":save":
            save = {"n": len(KB)}
            for i, (lbl, b) in enumerate(KB):
                save[f"lbl_{i}"] = lbl; save[f"on_{i}"] = b[0]; save[f"off_{i}"] = b[1]
            np.savez(KB_STORE % a.seed, **save)
            return f"  [saved {len(KB)} facts]"
        if t[0] == ":quit":
            return None
        if t[0] == "who" and len(t) >= 3:
            b = find(action=t[1], patient=t[2])
            return f"  {RM.unbind_spiking(bridge, idx, b, 'agent', roles, concepts, words, D, xp)}" if b is not None else "  (unknown)"
        if t[0] == "what" and "does" in t and len(t) >= 4:
            b = find(agent=t[2], action=t[3])
            return f"  {RM.unbind_spiking(bridge, idx, b, 'patient', roles, concepts, words, D, xp)}" if b is not None else "  (unknown)"
        if t[0] == "does" and len(t) >= 4:
            b = find(agent=t[1], action=t[2], patient=t[3])
            if b is None:
                return "  (unknown)"
            pol = RM.unbind_spiking(bridge, idx, b, "polarity", roles, concepts, ["AFFIRM", "NEGATE"], D, xp)
            return "  yes" if pol == "AFFIRM" else "  no"
        if "not" in t and len(t) >= 4:
            i = t.index("not"); store(t[0], t[i+1], t[i+2], "NEGATE"); return "  [stored, negated]"
        if len(t) >= 3:
            store(t[0], t[1], t[2]); return "  [stored]"
        return "  (didn't understand -- try 'dog go north' or 'who go north?')"

    print(f"=== Spiking conversational agent REPL (seed {a.seed}, vocab {len(words)}) ===")
    print(f"vocab: {words}")
    print("teach facts ('dog go north'), ask ('who go north?', 'does dog go north?'); :facts :save :quit\n")
    if a.script is not None:
        for line in a.script.replace(";", "\n").split("\n"):
            line = line.strip()
            if not line:
                continue
            print(f"> {line}")
            r = respond(line)
            if r is None:
                break
            print(r)
    else:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            r = respond(line)
            if r is None:
                break
            print(r)


if __name__ == "__main__":
    main()
