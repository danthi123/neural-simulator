# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md
"""Strict recall scorer for the live-brain probe.

A reply counts as a correct answer only if (1) the key word appears as a word (a plural/suffix of the key is fine),
and (2) the reply is not first-person: a reply that speaks as "I/my/me/mine" is the user's own told statement echoed
back (e.g. "My cat is called Pixel."), not an answer addressed to the user. A key word inside the question itself never counts.
"""
import re

FIRST_PERSON = {"i", "i'm", "im", "i've", "ive", "i'll", "i'd", "my", "me", "mine", "myself"}


def _words(text: str) -> list:
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


def has_key(reply: str, key: str) -> bool:
    """Key present as a word start (peanut matches peanuts); not inside another word."""
    return re.search(r"\b" + re.escape(key.lower()), reply.lower()) is not None


def is_first_person(reply: str) -> bool:
    return any(w in FIRST_PERSON for w in _words(reply))


def score(reply: str, key: str, question: str = "") -> dict:
    """lenient = key substring present (the old scorer); hit = strict; echo = key present but first-person reply."""
    lenient = key.lower() in reply.lower()
    present = has_key(reply, key) and not has_key(question, key)
    echo = present and is_first_person(reply)
    return {"lenient": lenient, "hit": present and not echo, "echo": echo}
