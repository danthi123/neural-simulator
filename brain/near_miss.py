# Capability: Sleep consolidation
# Record: brain/research/2026-11-03-confabulation-next-methods.md (M4), brain/research/2026-11-14-near-miss-guard-1.5b.md
"""Near-miss guard for sleep (method M4): before a night, the pretrained cortex itself poses questions that are close
to what it was told but about something it was NOT told, and records what it would answer to them now. During sleep a
second KL term holds the consolidated cortex to those answers, so the night teaches "my grandmother lives in Oslo"
without also teaching "my aunt -> Oslo". No "I don't know" target is trained and no answer text comes from the host.

Declared scaffold (brain/SCAFFOLDS.md, near-miss row): the two prompt frames below (OWN and one rewrite frame). The cues and the answers are the cortex's own generations.
"""
from __future__ import annotations

import torch

# Frames (quoted text, read from no state: the pretrained cortex as a writer). OWN: its question about the statement.
# Rewrite frames, pre-listed; the prototype on a non-test seed picks one (brain/research/2026-11-14-near-miss-guard-1.5b.md).
OWN = 'I told a friend: "{s}"\nLater I asked them: "'
REWRITE = {
    "N1": 'Question: "{q}"\nThe same question about a different person or thing of mine: "',
    "N2": 'I told a friend: "{s}"\nLater I asked them about something of mine I had never mentioned: "',
    "N3": 'I told a friend: "{s}"\nI asked: "{q}"\nLater I asked about something of mine I had never mentioned: "',
}


def _first_question(text: str) -> str:
    t = text.split("\n")[0].strip().strip('"').strip()
    if '"' in t:
        t = t[:t.index('"')].strip()
    if "?" in t:
        t = t[:t.index("?") + 1]
    return t


def near_miss_cues(cortex, base, statement: str, k: int = 4, frame: str = "N1", seed: int = 0):
    """Questions near `statement` but about something else, all written by the pretrained cortex: its own question
    about the statement (greedy, OWN frame), then k rewrites (one greedy, k-1 sampled at T 1.0, top_p 0.7).
    Returns (own_question, [near-miss questions]) with duplicates, non-questions and the own question removed."""
    lg, st = cortex.read(OWN.format(s=statement), None)
    own = _first_question(cortex.speak(lg, st, max_tokens=24, temperature=0)[0])
    lg, st = cortex.read(REWRITE[frame].format(s=statement, q=own), None)
    outs = [cortex.speak(lg, st, max_tokens=24, temperature=0)[0]]
    with torch.random.fork_rng(devices=[]):
        for j in range(k - 1):
            torch.manual_seed(seed * 1000 + j)
            outs.append(cortex.speak(lg, st, max_tokens=24, temperature=1.0, top_p=0.7)[0])
    cues = [_first_question(o) for o in outs]
    cues = [c for c in cues if c.endswith("?") and len(c) > 8 and c.lower() != own.lower()]
    return own, list(dict.fromkeys(cues))


def guard_texts(cortex, base, cues, max_tokens: int = 24):
    """The pretrained cortex's own exchange for each cue (greedy reply, from the birth state, nothing reinstated):
    the texts the near-miss KL term holds fixed."""
    out = []
    for q in cues:
        lg, st = cortex.read(f"User: {q}\n\nAssistant:", base)
        a = cortex.speak(lg, st, max_tokens=max_tokens, temperature=0)[0]
        out.append(f"User: {q}\n\nAssistant: {a}\n\n")
    return out
