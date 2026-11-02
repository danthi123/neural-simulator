"""The brain: cortex + hippocampus + self, one continuous life saved to a directory."""
from __future__ import annotations

import json
import os
import time

import torch

from brain.cortex import Cortex
from brain.hippocampus import Episode, Hippocampus

# Scaffold (see SCAFFOLDS.md): a neutral "birth" context that sets the conversational register. Names and facts are
# NOT here; they live in memory.
BIRTH = "System: A friendly conversation between a young digital brain and a person.\n\n"


# Scaffold (SCAFFOLDS.md): which turns are experiences worth storing. Questions and requests are not stored.
_REQUEST = ("tell me", "can you", "could you", "please", "let's", "what", "how", "why", "who", "where", "when", "do you")


def _is_telling(text: str) -> bool:
    t = text.strip().lower()
    return not t.endswith("?") and not t.startswith(_REQUEST)


class Brain:
    def __init__(self, home: str, name: str = "Nova", friend: str = "Dan", device: str = "cpu", cortex: Cortex = None):
        self.home = home
        os.makedirs(home, exist_ok=True)
        self.cortex = cortex or Cortex(device=device)
        self.hippocampus = Hippocampus()
        _, self.base = self.cortex.read(BIRTH, None)
        self.state = [x.clone() for x in self.base]
        self.meta = {"name": name, "friend": friend, "born": time.time(), "turns": 0}
        if not self.load():
            self._grow_self()
            self.save()

    # ---- self ----
    def _grow_self(self):
        n, f = self.meta["name"], self.meta["friend"]
        # Self-knowledge is stored as lived exchanges (question and own answer), so the matching question cues it.
        for q, ans in ((f"What's your name?", f"My name is {n}."), (f"Who are you?", f"I'm {n}, a young digital brain."),
                       (f"Do you remember my name?", f"Of course, you're {f}, my friend.")):
            self.hippocampus.store(self.cortex, self.base, f"{q}\n\nAssistant: {ans}", speaker="User", salience=3.0,
                                   permanent=True)

    # ---- one turn ----
    def hear(self, text: str):
        """Process one user turn; returns (reply, info). Memory returns as cortical state, never as text."""
        text = text.strip()
        info = {"recalled": None, "similarity": None, "stored": False}
        ep, sim = self.hippocampus.recall(self.cortex, self.base, text, question=text.endswith("?"))
        info["similarity"] = round(sim, 3)
        # Recall is a transient reactivation for THIS reply; it does not accumulate in the ongoing state.
        active = self.state
        if ep is not None:
            active = self.hippocampus.reinstate(self.cortex, self.state, ep)
            info["recalled"] = ep.text
        logits, st = self.cortex.read(f"User: {text}\n\nAssistant:", active)
        reply, _ = self.cortex.speak(logits, st)
        _, self.state = self.cortex.read(f"User: {text}\n\nAssistant: {reply}\n\n", self.state)
        if self.hippocampus.enabled and _is_telling(text):
            self.hippocampus.store(self.cortex, self.base, text, speaker="User")
            info["stored"] = True
        self.meta["turns"] += 1
        return reply, info

    # ---- persistence ----
    def save(self):
        eps = [{k: getattr(e, k) for k in ("text", "delta", "key", "strength", "created", "last_used", "uses",
                                             "permanent")} for e in self.hippocampus.episodes]
        torch.save({"episodes": eps, "state": self.state}, os.path.join(self.home, "memory.pt"))
        with open(os.path.join(self.home, "meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=1)

    def load(self) -> bool:
        p = os.path.join(self.home, "memory.pt")
        if not os.path.exists(p):
            return False
        blob = torch.load(p, weights_only=False)
        self.hippocampus.episodes = [Episode(**e) for e in blob["episodes"]]
        # A restart is a new day: working state starts from the base (plus, later, what sleep consolidated); the
        # previous day's live state is NOT restored (that would be working memory surviving sleep).
        with open(os.path.join(self.home, "meta.json")) as fh:
            self.meta = json.load(fh)
        return True
