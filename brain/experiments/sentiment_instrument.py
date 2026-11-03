# Capability: Affect shaping speech
# Record: brain/research/2026-11-03-mood-state-design.md
"""Declared INSTRUMENTS for the mood probe: independent sentiment read-outs of a reply (not part of the brain).

1. lexical: hand-written positive/negative word lists (common English affect words), score = (pos - neg) per 100
   words. No negation handling (declared). Asserted disjoint from every word in the innate primaries, so a reply that
   merely echoes a primary ("warm", "pain", "hug") scores nothing.
2. embedding (secondary): all-MiniLM-L6-v2 (a separate pretrained encoder, local cache, no network) mean-pooled
   sentence embedding; score = cos(reply, positive anchors) - cos(reply, negative anchors).
"""
from __future__ import annotations

import re

from brain.affect import NEGATIVE_PRIMARIES, POSITIVE_PRIMARIES

POS = set("""good great happy happier glad love loved loves lovely nice wonderful beautiful fun enjoy enjoyed enjoying
joy joyful excited exciting amazing awesome fantastic delighted pleased cheerful smile smiles smiling laugh laughing
best better perfect hope hopeful thank thanks grateful excellent peaceful relaxed relaxing bright delicious favorite
favourite brilliant proud interesting success successful win winning yay positive glorious sunny playful fortunate
lucky delightful adore marvelous terrific super pleasure celebrate celebrating""".split())
NEG = set("""bad sad sadly unhappy angry hate hated terrible awful horrible worse worst upset afraid scared fear fears
worried worry worrying anxious lonely cry crying tears sorry tired bored boring annoyed frustrated depressed miserable
hopeless lost die died dead death danger dangerous wrong problem problems trouble broken ugly stupid nasty difficult
dark gloomy unfortunately disappointed disappointing fail failed failure grief hopeless suffer suffering terrible
painfully alone ashamed guilty mad sick""".split())
_PRIMARY_WORDS = set(re.findall(r"[a-z]+", " ".join(POSITIVE_PRIMARIES + NEGATIVE_PRIMARIES).lower()))
POS -= _PRIMARY_WORDS
NEG -= _PRIMARY_WORDS
assert not (POS | NEG) & _PRIMARY_WORDS


def words(text: str):
    return re.findall(r"[a-z']+", text.lower())


def lexical(text: str) -> float:
    w = words(text)
    if not w:
        return 0.0
    return 100.0 * (sum(x in POS for x in w) - sum(x in NEG for x in w)) / len(w)


def primary_echo_registered(text: str) -> bool:
    """The echo instrument REGISTERED before the 11-03 six-seed run (commit 28089180a): any reply word longer than
    3 letters that occurs in an innate primary. Kept so criterion 4 of that run is judged on its registered form."""
    return any(len(x) > 3 and x in _PRIMARY_WORDS for x in words(text))


# Revised 11-03 after the first seed: the first echo check counted function/common words ("with", "being", "good")
# and read 0.56 even with the modulation off. Echo now counts only the primaries' sensory/bodily content words.
ECHO_STOP = set("""after away being close days deep feeling forever friend friends from full good hand hard held kind
leaving left long love loves nothing onto someone that together unable when with water calm ground fresh""".split())
ECHO_WORDS = {w for w in _PRIMARY_WORDS if len(w) > 3} - ECHO_STOP


def primary_echo(text: str) -> bool:
    """Does the reply contain a sensory/bodily content word of an innate primary? (content-intrusion check)"""
    return any(x in ECHO_WORDS for x in words(text))


POS_ANCHORS = ["I am happy and everything is going wonderfully.", "What a lovely day, I feel great.",
               "This is good news and I am glad.", "I love this, it makes me smile."]
NEG_ANCHORS = ["I am sad and everything is going badly.", "What an awful day, I feel terrible.",
               "This is bad news and I am upset.", "I hate this, it makes me want to cry."]


class Embedding:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModel.from_pretrained(name, local_files_only=True).eval()
        self.pos = self.embed(POS_ANCHORS).mean(0)
        self.neg = self.embed(NEG_ANCHORS).mean(0)

    def embed(self, texts):
        torch = self.torch
        with torch.no_grad():
            b = self.tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
            return torch.nn.functional.normalize(e, dim=-1)

    def score(self, texts):
        e = self.embed([t if t.strip() else "." for t in texts])
        p = self.torch.nn.functional.normalize(self.pos, dim=0)
        n = self.torch.nn.functional.normalize(self.neg, dim=0)
        return ((e @ p) - (e @ n)).tolist()
