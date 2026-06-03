"""320-concept conversational agent: comprehend -> decide-what-to-say (content-selection Control) ->
produce, with STORAGE and PRODUCTION on the validated 320-concept spiking substrate (spiking coincidence
bind / unbind) -- the project's scale frontier (15x the 22-word toy integration).

This grounds the integration arc's dialogue-planning agent in the validated 320 substrate:
  - COMPREHEND: simple SVO parse (numpy; the faithful Hebbian conjunctive parser is the remaining piece).
  - STORE: each SVO fact bound into the spiking substrate (RM.bind_fact_spiking, spiking coincidence).
  - DECIDE-WHAT-TO-SAY: the content-selection Control (validated this session) over an association graph
    built from the agent's OWN stored facts (a 320-word graph).
  - PRODUCE: each elaborated fact recovered by spiking UNBIND + cleanup over all 320 concepts
    (RM.unbind_spiking) -> rendered as a sentence.

So three of the four pieces are FAITHFUL SPIKING (storage, dialogue planning, production); only the parse
is numpy. Reuses the validated 320 machinery (_insubstrate_bind_unbind_probe + _insubstrate_relational_
memory_probe + the 320 distinct flat codes cache) + ContentSelectionController. Needs the cache (produced
by _insubstrate_flatdistinct320_test) + GPU. No protected-module change.

  python -m research.runners.integrated_conversation_320
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend
from research.runners.content_selection import build_association_graph, ContentSelectionController

CACHE = "research/findings/raw/_flatdist320_codes.npz"


class Conversation320Agent:
    """A conversational agent over 320 concepts: store facts (spiking bind), answer/elaborate via the
    content-selection Control, produce sentences by spiking unbind."""

    def __init__(self, seed=42, run_steps=150, coinc_bias=-500.0):
        if not os.path.exists(CACHE):
            raise FileNotFoundError(f"{CACHE} missing -- run _insubstrate_flatdistinct320_test first.")
        self.xp, _ = get_backend()
        d = np.load(CACHE)
        self.words = [str(w) for w in d["_words"]]
        self.bank_of = {str(w): str(b) for w, b in zip(d["_words"], d["_banks"])}
        self.codes = {w: np.asarray(d[w], dtype=np.float64) for w in self.words}
        self.D = self.codes[self.words[0]].shape[0]
        P.RUN_STEPS = run_steps
        P.COINC_BIAS = coinc_bias                       # validated higher-rate operating point
        rng = np.random.default_rng(seed)
        self.roles = {r: rng.choice([-1.0, 1.0], size=self.D) for r in RM.ROLES}
        self.roles = {r: v / np.linalg.norm(v) for r, v in self.roles.items()}
        self.bridge, self.bidx = P.build(seed, self.D, self.xp)
        self.kb = []                                    # [{"fact": {role: word}, "bound": onoff}]
        self.focus = None
        self.ctrl = None
        self._elaborated = set()

    # --- STORE (spiking bind) ---
    def learn(self, fact):
        bound = RM.bind_fact_spiking(self.bridge, self.bidx, fact, self.codes, self.roles, self.D, self.xp)
        self.kb.append({"fact": fact, "bound": bound})
        return fact

    # --- PRODUCE (spiking unbind + cleanup over all 320) ---
    def _produce(self, entry):
        return " ".join(
            RM.unbind_spiking(self.bridge, self.bidx, entry["bound"], r, self.roles,
                              self.codes, self.words, self.D, self.xp)
            for r in RM.ROLES)

    # --- the agent's KB-derived association graph (320-word) ---
    def _kb_graph(self):
        pairs = []
        for e in self.kb:
            ws = [e["fact"][r] for r in RM.ROLES]
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    pairs.append(f"{ws[i]}_{ws[j]}")
        return build_association_graph(pairs) if pairs else {}

    def _facts_about(self, word):
        return [e for e in self.kb if word in e["fact"].values()]

    # --- DECIDE WHAT TO SAY (content-selection Control) + produce ---
    def set_topic(self, topic):
        self.focus = topic
        self._elaborated = set()
        self.ctrl = ContentSelectionController(self._kb_graph())

    def elaborate(self):
        if self.focus is None:
            return "(no topic yet)"
        if not self._facts_about(self.focus):
            return f"(i don't know about {self.focus})"
        for _ in range(len(self.words)):
            pick = self.ctrl.turn([self.focus])
            if pick is None:
                break
            for e in self.kb:
                vals = set(e["fact"].values())
                if self.focus in vals and pick in vals and id(e) not in self._elaborated:
                    self._elaborated.add(id(e))
                    return self._produce(e)             # PRODUCE via spiking unbind
        return f"(that's all i know about {self.focus})"

    # --- COMPREHEND + dispatch ---
    def hear(self, text):
        toks = (text or "").strip().rstrip("?").split()
        if not toks:
            return "(i didn't understand)"
        if toks[0] in ("more", "and") and self.focus is not None:
            return self.elaborate()
        content = [t for t in toks if t in self.codes]
        if len(content) >= 3:                            # STATEMENT (SVO) -> store via spiking bind
            self.learn({"agent": content[0], "action": content[1], "patient": content[2]})
            return f"ok -- i learned: {content[0]} {content[1]} {content[2]}"
        if len(content) == 1:                            # TOPIC -> elaborate via Control + spiking produce
            self.set_topic(content[0])
            return self.elaborate()
        return "(i didn't understand)"


def _pick(words, bank_of, bank, n, rng):
    cands = [w for w in words if bank_of[w] == bank]
    return [str(w) for w in rng.choice(cands, size=min(n, len(cands)), replace=False)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-steps", type=int, default=150)
    args = ap.parse_args()
    try:
        agent = Conversation320Agent(seed=args.seed, run_steps=args.run_steps)
    except FileNotFoundError as e:
        print(f"CANNOT-RUN: {e}", flush=True)
        return
    print(f"=== 320-concept conversational agent (V={len(agent.words)}, D={agent.D}) ===", flush=True)
    print("    storage = spiking bind | dialogue planning = content-selection Control | "
          "production = spiking unbind\n", flush=True)

    rng = np.random.default_rng(args.seed)
    nouns = _pick(agent.words, agent.bank_of, "noun", 3, rng)
    verbs = _pick(agent.words, agent.bank_of, "verb", 3, rng)
    adjs = _pick(agent.words, agent.bank_of, "adj", 3, rng)
    # two facts sharing the first noun, so elaborating that noun produces BOTH (the new dialogue-planning)
    facts = [
        f"{nouns[0]} {verbs[0]} {adjs[0]}",
        f"{nouns[0]} {verbs[1]} {adjs[1]}",
        f"{nouns[1]} {verbs[2]} {adjs[2]}",
    ]
    script = facts + [nouns[0], "more", "more", nouns[1]]
    for u in script:
        print(f"  user : {u}", flush=True)
        print(f"  agent: {agent.hear(u)}", flush=True)
    print("\n  -> a conversational agent over 320 concepts: it STORES facts in the spiking substrate,", flush=True)
    print("     ELABORATES a topic via the content-selection Control (walking its associative memory),", flush=True)
    print("     and PRODUCES each fact by spiking unbind -- dialogue planning + storage + production all", flush=True)
    print("     faithful spiking, at 15x the toy-integration scale.", flush=True)


if __name__ == "__main__":
    main()
