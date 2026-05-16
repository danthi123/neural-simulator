"""Productive concept-grammar: grammatical strings from retrieved
concepts. Pure (no bridge). Slots SUBJ/REL/OBJ/ATTR/POLARITY/QTY."""
from __future__ import annotations

def _cap(s: str) -> str: return s[:1].upper() + s[1:] if s else s

def render(intent: str, fillers: dict) -> str:
    f = fillers
    def has(*ks): return all(k in f and f[k] not in (None, "", []) for k in ks)
    try:
        if intent == "assoc" and has("SUBJ", "OBJ"):
            return f"{_cap(f['SUBJ'])} is associated with {f['OBJ']}."
        if intent == "attr" and has("SUBJ", "ATTR"):
            neg = " not" if f.get("POLARITY") == "neg" else ""
            return f"{_cap(f['SUBJ'])} is{neg} {f['ATTR']}."
        if intent == "yesno_yes" and has("SUBJ", "ATTR"):
            return f"Yes, {f['SUBJ']} is {f['ATTR']}."
        if intent == "yesno_no" and has("SUBJ", "ATTR"):
            return f"No, I haven't learned that {f['SUBJ']} is {f['ATTR']}."
        if intent == "list" and has("SUBJ", "OBJ"):
            objs = f["OBJ"] if isinstance(f["OBJ"], list) else [f["OBJ"]]
            if len(objs) == 1: joined = objs[0]
            elif len(objs) == 2: joined = f"{objs[0]} and {objs[1]}"
            else: joined = ", ".join(objs[:-1]) + f", and {objs[-1]}"
            return f"{_cap(f['SUBJ'])} is associated with {joined}."
    except Exception:
        pass
    subj = f.get("SUBJ", "that")
    return f"I don't know about {subj} yet."
