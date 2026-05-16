"""Conversation -> concept-sequence log records (Stage-2 replay fuel).
Pure record builder; the agent appends these as JSONL."""
from __future__ import annotations
import re

def _concepts_in(text: str):
    return [w for w in re.findall(r"[a-z]+", text.lower())]

def make_record(turn, user, intent, retrieved, abstained, response):
    q = _concepts_in(user)
    query_concept = q[-1] if q else ""
    seq = [query_concept] if query_concept else []
    if not abstained and retrieved:
        seq.append(retrieved[0][0])
    return {
        "turn": int(turn), "user": user, "intent": intent,
        "retrieved": [[c, float(r)] for c, r in retrieved],
        "abstained": bool(abstained), "response": response,
        "concept_sequence": seq,
    }
