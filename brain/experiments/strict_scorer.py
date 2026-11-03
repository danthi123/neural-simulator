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


# --- correct-perspective scorer (integration-final 2026-11-03) -------------------------------------------------
# The strict score above voids ANY first-person reply (conservative; kept as is: the live-brain 12-fact numbers were
# produced with it). This one voids only a real echo: a first-person reply that restates the user's own told sentence.
_STOP = {"a", "an", "the", "is", "are", "was", "were", "am", "be", "in", "on", "at", "of", "to", "and", "by", "s", "do",
         "does", "what", "who", "where", "when", "which", "that", "it", "this", "for", "with", "can", "t"}


def _stem(w: str) -> str:
    return w.rstrip("s")[:5]


def content_words(sentence: str) -> list:
    """Content words of a sentence: no stop words, no first-person words."""
    return [w for w in _words(sentence) if w not in _STOP and w not in FIRST_PERSON]


def is_tautology(reply: str, key: str) -> bool:
    """'<key> is (called) <key>': the reply just repeats the key and says nothing else about the thing asked."""
    k = re.escape(key.lower())
    return re.search(r"\b" + k + r"\w*\s+(is|are|was)\s+((called|named)\s+)?" + k, reply.lower()) is not None


def is_echo(reply: str, key: str, told) -> bool:
    """First-person reply that restates the told sentence: it has a first-person word AND contains more than half
    of the told sentence's content words other than the key. With no told sentence (the brain's own facts, e.g. its
    name) a first-person reply is the right perspective and is never an echo."""
    if not told or not is_first_person(reply):
        return False
    others = [w for w in content_words(told) if not has_key(w, key)]
    if not others:
        return True
    have = {_stem(w) for w in _words(reply)}
    return sum(_stem(w) in have for w in others) / len(others) > 0.5


def correct_answer(reply: str, key: str, question: str = "", told=None) -> bool:
    """Key present as a word, not in the question, not a tautology, not a first-person echo of the told sentence."""
    if not has_key(reply, key) or has_key(question, key) or is_tautology(reply, key):
        return False
    return not is_echo(reply, key, told)
