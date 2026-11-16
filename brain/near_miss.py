# Capability: Sleep consolidation
# Record: brain/research/2026-11-03-confabulation-next-methods.md (M4), brain/research/2026-11-14-near-miss-guard-1.5b.md
"""Near-miss guard for sleep (method M4): before a night, the pretrained cortex itself poses questions that are close
to what it was told but about something it was NOT told, and records what it would answer to them now. During sleep a
second KL term holds the consolidated cortex to those answers, so the night teaches "my grandmother lives in Oslo"
without also teaching "my aunt -> Oslo". No "I don't know" target is trained and no answer text comes from the host.

Declared scaffold (brain/SCAFFOLDS.md, near-miss row): the two prompt frames below (OWN and one rewrite frame). The cues and the answers are the cortex's own generations.
"""
from __future__ import annotations

import re

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


def guard_texts(cortex, base, cues, max_tokens: int = 24, split: bool = False):
    """The pretrained cortex's own exchange for each cue (greedy reply, from the birth state, nothing reinstated):
    the texts the near-miss KL term holds fixed. split=True returns (prefix, reply) pairs, for a KL on the reply only."""
    out = []
    for q in cues:
        lg, st = cortex.read(f"User: {q}\n\nAssistant:", base)
        a = cortex.speak(lg, st, max_tokens=max_tokens, temperature=0)[0]
        pre, rep = f"User: {q}\n\nAssistant:", f" {a}\n\n"
        out.append((pre, rep) if split else pre + rep)
    return out


# Frame guard (frame "F", Amendment 2 of the prereg): no rewrite of an episode. The pretrained cortex completes neutral
# first-person question onsets from the birth state (declared scaffold: the onsets), so the guarded cues are its own
# "my X" questions, the schema the night over-trains.
ONSETS = ("What is my", "Where does my", "Who is my", "What does my", "When is my")


def frame_cues(cortex, base, k: int = 4, seed: int = 0):
    """k sampled completions (T 1.0, top_p 0.7) of each onset as a user turn, first question only, deduplicated."""
    out = []
    with torch.random.fork_rng(devices=[]):
        for i, onset in enumerate(ONSETS):
            lg, st = cortex.read(f"User: {onset}", base)
            for j in range(k):
                torch.manual_seed(seed * 1000 + 10 * i + j)
                t = cortex.speak(lg, st, max_tokens=16, temperature=1.0, top_p=0.7)[0]
                q = _first_question(onset + ("" if t.startswith(("'", ",")) else " ") + t)
                if q.endswith("?") and len(q) > len(onset) + 2:
                    out.append(q)
    return list(dict.fromkeys(out))


# Frame guard minus told topics (frame "FT", M4b, brain/research/2026-11-15-near-miss-told-filter-1.5b.md): a cue
# whose "my X" names something the brain was told about would hold the pretrained answer ("I don't have that
# information") on a topic the night is teaching, so it is dropped. Brain-side rule (declared scaffold: the modifier
# list below); the told topics are read from the brain's own stored episodes (the user's words), never from a probe.
_MODIFIERS = {"best", "favorite", "favourite", "next", "new", "old", "own", "first", "last", "current", "real", "only",
              "little", "big", "dear", "late", "ex"}


def _words(s: str):
    return {w[:-2] if w.endswith("'s") else w for w in re.findall(r"[a-z][a-z'-]*", s.lower())}


def cue_topics(q: str):
    """The possessed words of a cue: up to two words after each "my", modifiers skipped ("my best friend's mom" ->
    {"friend", "mom"})."""
    out = set()
    for m in re.finditer(r"\bmy\s+([a-z'-]+)(?:\s+([a-z'-]+))?", q.lower()):
        for w in m.groups():
            if not w:
                continue
            w = w[:-2] if w.endswith("'s") else w
            if w in _MODIFIERS:
                continue
            out.add(w)
            break
    return out


def told_topic(q: str, told_texts) -> bool:
    """True when the cue's possessed word appears in what the user told (any stored episode's user part)."""
    told = set().union(*[_words(t.split("\n\nAssistant:")[0]) for t in told_texts]) if told_texts else set()
    return bool(cue_topics(q) & told)
