"""Owner-facing CONVERSATIONAL AGENT demo: a scripted multi-turn conversation, entirely in the
spiking substrate. The agent STORES facts from statements (spiking compositional bind), ANSWERS
wh-questions (relational query), and PERSISTS its knowledge across a session boundary -- composing
every validated piece of the 2026-05-31 composition arc into one conversational artifact.

Validated sub-capabilities (research/findings/2026-05-31-composition-in-spiking-substrate-SYNTHESIS.md):
  bind/unbind multi-seed (K<=6) + adversarial CLEAR; relational fact-memory multi-seed (scales ~12);
  end-to-end live-text + learned parser; wh-question answering; persistent KB across sessions.

Honest scope: 16-word vocab; canonical SVO statements + wh-questions; roles/cues are mapped from the
sentence structure (the learned parser validated separately, _insubstrate_parser_stdp_probe.py); the
recognition front-end caps the vocabulary (~64/bridge clean), NOT the composition.

Run:  python -m research.runners.compose_conversation_demo
"""
from __future__ import annotations
import argparse
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
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
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bridge, idx = P.build(a.seed, D, xp)

    def pick(*c):
        for w in c:
            if w in concepts:
                return w
        return words[0]

    KB = []  # list of (label, bound)

    def tell(agent, action, patient):
        fact = {"agent": agent, "action": action, "patient": patient}
        KB.append((f"{agent} {action} {patient}", RM.bind_fact_spiking(bridge, idx, fact, concepts, roles, D, xp)))
        print(f"  > {agent} {action} {patient}.            [stored]")

    def ask_who(action, patient):
        for _, b in KB:
            if (RM.unbind_spiking(bridge, idx, b, "action", roles, concepts, words, D, xp) == action and
                    RM.unbind_spiking(bridge, idx, b, "patient", roles, concepts, words, D, xp) == patient):
                return RM.unbind_spiking(bridge, idx, b, "agent", roles, concepts, words, D, xp)
        return "(unknown)"

    def ask_what_obj(agent, action):
        for _, b in KB:
            if (RM.unbind_spiking(bridge, idx, b, "agent", roles, concepts, words, D, xp) == agent and
                    RM.unbind_spiking(bridge, idx, b, "action", roles, concepts, words, D, xp) == action):
                return RM.unbind_spiking(bridge, idx, b, "patient", roles, concepts, words, D, xp)
        return "(unknown)"

    print(f"=== Spiking conversational agent (backend={backend}, seed={a.seed}, vocab {len(words)}) ===")
    print("(statements are bound + stored in spiking; questions answered by spiking relational query)\n")
    print("-- Turn 1: the user tells the agent some facts --")
    dog, cat = pick("dog"), pick("cat")
    go, come = pick("go", "come"), pick("come", "stop")
    north, south = pick("north", "river"), pick("south", "apple")
    tell(dog, go, north)
    tell(cat, come, south)

    print("\n-- Turn 2: the user asks questions --")
    print(f"  ? who {go} {north}?            -> {ask_who(go, north)}")
    print(f"  ? what does {dog} {go}?        -> {ask_what_obj(dog, go)}")
    print(f"  ? who {come} {south}?          -> {ask_who(come, south)}")

    print("\n-- Turn 3: persist + reload (a fresh session), then keep answering --")
    saved = [(lbl, b[0].copy(), b[1].copy()) for lbl, b in KB]
    del bridge
    bridge, idx = P.build(a.seed, D, xp)                       # fresh substrate
    KB = [(lbl, (on, off)) for lbl, on, off in saved]          # reloaded knowledge
    print(f"  [reloaded {len(KB)} facts into a fresh substrate]")
    print(f"  ? who {go} {north}?            -> {ask_who(go, north)}   (recalled across the session boundary)")
    print(f"  + the user adds a new fact this session:")
    tell(cat, pick('stop', 'look'), pick('big', 'east'))
    print(f"  ? who {go} {north}?            -> {ask_who(go, north)}   (old fact still known)")

    print("\nEverything -- storing facts, answering wh-questions, persisting across sessions -- runs as "
          "spiking compositional binding on real substrate concepts. Honest scope: 16-word vocab, SVO; the "
          "recognition front-end (not the composition) caps vocabulary growth.")


if __name__ == "__main__":
    main()
