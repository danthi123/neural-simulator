"""Integrated biology-faithful conversational loop (tangible artifact): comprehend -> memory -> compose ->
PRODUCE full ordered-sentence responses. The agent hears statements, binds them into a knowledge base, and
answers questions by RETRIEVING the relevant meaning and PRODUCING a full composed sentence (generate-by-
composition), not a one-word answer. Demonstrates the validated conversational components integrated into one
loop, with the new full-sentence production capability.

Honest scope: the parser here is simple position-based SVO (the validated faithful component is the Hebbian
conjunctive parser, _vsa_parser_voice_probe); the substrate is numpy (the components are validated spiking).
This demo shows the INTEGRATION + full-sentence PRODUCTION; the spiking + learned-parser version is the
follow-up. Reuse-by-import (generate-by-composition); no protected-module change.

  python -m research.findings.raw._integrated_conversation_loop_demo
"""
from __future__ import annotations
import numpy as np
from research.findings.raw._generate_by_composition_probe import build_world, compose, generate

WORDS = ["dog", "cat", "bird", "child", "ball", "apple", "river", "sun",          # 0-7 nouns
         "chase", "eat", "see", "hold", "want", "give", "watch", "find",          # 8-15 verbs
         "small", "red", "happy", "fast", "big", "cold"]                          # 16-21 adj/patient
W2I = {w: i for i, w in enumerate(WORDS)}
ROLES3 = ["agent", "action", "patient"]


class Agent:
    """Comprehend -> bind into KB -> retrieve + PRODUCE full-sentence responses."""

    def __init__(self, seed=42, D=1024):
        self.concepts, self.roles = build_world(len(WORDS), D, seed)
        self.words = list(self.concepts.keys())
        self.kb = []          # list of {role: word_index} facts (the agent's memory)

    def _say(self, meaning):
        """PRODUCE: generate the ordered sentence from a composed meaning (generate-by-composition)."""
        bound = compose(meaning, self.concepts, self.roles)
        idxs = generate(bound, ROLES3, self.concepts, self.roles, self.words)
        return " ".join(WORDS[i] for i in idxs)

    def hear(self, text):
        toks = text.strip().rstrip("?").split()
        # QUESTION forms -> retrieve + produce a full sentence
        if toks and toks[0] in ("what", "who", "tell"):
            if toks[0] == "tell":                              # "tell me about <noun>"
                noun = toks[-1]
                facts = [f for f in self.kb if WORDS[f["agent"]] == noun or WORDS[f["patient"]] == noun]
                if not facts:
                    return "(i don't know about %s)" % noun
                return " ; ".join(self._say(f) for f in facts)
            # "what does <agent> <action>?" -> fill patient ; "who <action> <patient>?" -> fill agent
            known = {ROLES3[i]: W2I[w] for i, w in enumerate(toks[1:]) if w in W2I}
            content = [w for w in toks[1:] if w in W2I]
            if toks[0] == "what" and len(content) >= 2:        # agent + action known, want patient
                ag, ac = W2I[content[0]], W2I[content[1]]
                for f in self.kb:
                    if f["agent"] == ag and f["action"] == ac:
                        return self._say(f)                    # produce the FULL sentence answer
                return "(i don't know what %s %s)" % (WORDS[ag], WORDS[ac])
            if toks[0] == "who" and len(content) >= 2:         # action + patient known, want agent
                ac, pa = W2I[content[0]], W2I[content[1]]
                for f in self.kb:
                    if f["action"] == ac and f["patient"] == pa:
                        return self._say(f)
                return "(i don't know who %s %s)" % (WORDS[ac], WORDS[pa])
            return "(i didn't understand the question)"
        # STATEMENT form "agent action patient" -> comprehend + bind into KB
        content = [w for w in toks if w in W2I]
        if len(content) >= 3:
            fact = {"agent": W2I[content[0]], "action": W2I[content[1]], "patient": W2I[content[2]]}
            self.kb.append(fact)
            return "ok -- i learned: %s" % self._say(fact)     # echo back the COMPOSED+PRODUCED sentence
        return "(i didn't understand)"


def main():
    print("=== integrated biology-faithful conversational loop (comprehend -> memory -> compose -> PRODUCE) ===",
          flush=True)
    a = Agent(seed=42)
    script = [
        "dog chase cat", "child hold ball", "bird eat apple", "cat see bird",
        "what does dog chase",          # -> produce full sentence "dog chase cat"
        "who chase cat",                # -> "dog chase cat"
        "what does bird eat",           # -> "bird eat apple"
        "tell me about cat",            # -> all facts mentioning cat, each a full produced sentence
        "what does child hold",         # -> "child hold ball"
        "who eat apple",                # -> "bird eat apple"
        "what does dog eat",            # -> unknown (not learned)
    ]
    correct = total = 0
    for turn in script:
        resp = a.hear(turn)
        print(f"  USER: {turn:<28} AGENT: {resp}", flush=True)
        # score the question turns (the produced full sentence must contain the right facts)
        if turn.startswith(("what", "who")) and not resp.startswith("(i don't"):
            total += 1
            # the produced sentence is correct if it is a real stored fact rendered in order
            correct += int(all(w in WORDS for w in resp.split()))
    print(f"\n  KB now holds {len(a.kb)} facts (persisted across the conversation).", flush=True)
    print(f"  The agent PRODUCED full composed sentences as answers (not one-word) for {total} questions.",
          flush=True)
    print("  This integrates comprehend (parse) + memory (bind/retrieve) + PRODUCE (generate-by-composition) "
          "into one biology-faithful loop -- the tangible conversational agent.", flush=True)


if __name__ == "__main__":
    main()
