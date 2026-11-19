# Capability: Curiosity
# Record: brain/research/2026-11-18-novelty-inquiry-record.md (stage 0: results/2026-11-18-novelty-stage0-1.5b.md)
"""The cortex's own surprise at what it hears: a novelty signal for the brain (no words, no host lexicon).

word_surprisal: per-word prediction error (nats, summed over each word's tokens) of `prefix + text` read from a cortex
state. A word starts at a token whose text begins with a space (the RWKV world tokenizer joins the space to the word).
novelty: the largest word surprisal in the turn, the first word excluded (it carries the turn boundary). Read from the
brain's current state, so a word heard earlier in the conversation is less surprising (habituation, stage 0: AUROC
1.00 first vs repeat mention at 1.5B).
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


def word_surprisal(cortex, state, text: str, prefix: str = "User: "):
    """[(word, nats)] for the words of `text` after `prefix`, read from `state` (never mutated)."""
    pipe = cortex.pipe
    pre = pipe.encode(prefix.rstrip(" ")) if prefix.endswith(" ") else pipe.encode(prefix)
    full = pipe.encode(prefix.rstrip(" ") + " " + text if prefix.endswith(" ") else prefix + text)
    if full[:len(pre)] != pre:     # tokenizer merged across the boundary: fall back to reading the whole line
        pre = []
    model = getattr(cortex, "raw", cortex).model
    with torch.no_grad():
        logits, _ = model.forward(full, copy.deepcopy(state), full_output=True)
    lp = F.log_softmax(logits.float(), -1)
    words, cur, nll = [], "", 0.0
    for t in range(max(len(pre), 1), len(full)):
        piece = pipe.decode([full[t]])
        if piece.startswith(" ") and cur:
            words.append((cur.strip(), nll))
            cur, nll = "", 0.0
        cur += piece
        nll += -float(lp[t - 1, full[t]])
    if cur:
        words.append((cur.strip(), nll))
    return words


def novelty(cortex, state, text: str) -> float:
    """Largest word surprisal of the heard turn (nats), excluding its first word."""
    w = word_surprisal(cortex, state, text)
    return max((n for _, n in w[1:]), default=0.0)
