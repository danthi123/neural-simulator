"""Dialogue-state working memory: recent (concept, role) ring;
resolves pronoun/elliptical follow-ups to the last subject. Pure."""
from __future__ import annotations
from collections import deque

_PRONOUNS = {"it", "its", "that", "they", "them", "this"}

class DialogueState:
    def __init__(self, maxlen: int = 8):
        self._ring = deque(maxlen=maxlen)
    def push(self, concept: str, role: str) -> None:
        self._ring.append((concept, role))
    def recent(self): return list(self._ring)
    def last_subject(self):
        for c, r in reversed(self._ring):
            if r == "SUBJ": return c
        return None
    def resolve(self, token: str):
        if token.lower() in _PRONOUNS: return self.last_subject()
        return None
