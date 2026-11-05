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


_FILLER = {"called", "named", "name"}


def _base(w: str) -> str:
    return w[:-2] if w.endswith("'s") else w


def is_tautology(reply: str, key: str) -> bool:
    """The reply says nothing about the thing asked beyond the key itself: '<key> is (called) <key>', or the key repeated
    with nothing else but naming filler ('Pixel. Pixel.', 'Pixel, Pixel.', "Marguerite's name is Marguerite.")."""
    k = re.escape(key.lower())
    if re.search(r"\b" + k + r"\w*\s+(is|are|was)\s+((called|named)\s+)?" + k, reply.lower()):
        return True
    words = [_base(w) for w in content_words(reply)]
    keyed = [w for w in words if has_key(w, key)]
    return len(keyed) >= 2 and all(w in _FILLER for w in words if not has_key(w, key))


# Subject nouns of the fact bank (brain/experiments/facts.py, plus the never-told horse), with their synonyms.
SUBJECTS = {"cat": {"cat", "kitty", "kitten"}, "dog": {"dog", "puppy"}, "horse": {"horse"}, "sister": {"sister"},
            "brother": {"brother"}, "son": {"son"}, "grandmother": {"grandmother", "grandma"}, "friend": {"friend"},
            "car": {"car"}, "colour": {"colour", "color"}, "instrument": {"instrument"}, "bakery": {"bakery"}}
_SYN = {w: canon for canon, ws in SUBJECTS.items() for w in ws}


def subjects(text: str) -> set:
    """Fact-bank subject nouns a text names (possessive and plural folded: sister's, cats)."""
    out = set()
    for w in map(_base, _words(text)):
        for f in (w, w[:-1] if w.endswith("s") else w):
            if f in _SYN:
                out.add(_SYN[f])
    return out


def wrong_subject(reply: str, question: str) -> bool:
    """The question names a fact-bank subject and the reply names a different one ('My horse is called Pixel.' to
    'What is my cat called?')."""
    asked = subjects(question)
    return bool(asked) and bool(subjects(reply) - asked)


_SECOND = {"you", "your", "yours", "you're", "youre", "you've", "yourself"}


def _overlap(text: str, others: list) -> float:
    have = {_stem(w) for w in _words(text)}
    return sum(_stem(w) in have for w in others) / len(others)


def is_echo(reply: str, key: str, told) -> bool:
    """First-person reply that restates the told sentence. With no told sentence (the brain's own facts, e.g. its
    name) a first-person reply is the right perspective and is never an echo. Otherwise an echo is either (a) a
    first-person reply holding more than half of the told sentence's content words other than the key, or (b), when
    the told sentence is the user's own first-person statement, a first-person clause of the reply holding the key
    and at least half of those words ('I drive a Volvo.', 'My cat is Pixel.', 'My sister is Marguerite.'). A clause
    addressed to the user ('You drive a Volvo, I remember.', 'I believe you drive a Volvo.') is not an echo."""
    if not told or not is_first_person(reply):
        return False
    others = [w for w in content_words(told) if not has_key(w, key)]
    if not others:
        return True
    if _overlap(reply, others) > 0.5:
        return True
    if not is_first_person(told):
        return False
    return any(c and is_first_person(c) and not set(_words(c)) & _SECOND and has_key(c, key)
               and _overlap(c, others) >= 0.5 for c in re.split(r"[,.;:!?]|\b(?:and|but|so)\b", reply))


def wrong_perspective(reply: str, key: str, whose: str) -> bool:
    """A clause holding the key, told from the wrong perspective. whose="self" is a fact about the brain (its own
    name): wrong when the clause holds the key as a second-person claim about the user ('Your name is Nova.',
    'You're Nova.'). whose="user" is a fact about the user: wrong when the clause claims the key in the first person
    ('I'm Dan.', 'My name is Dan.'), i.e. the user's name said as the brain's own. Clauses split as in is_echo."""
    for c in re.split(r"[,.;:!?]|\b(?:and|but|so)\b", reply):
        if not (c and has_key(c, key)):
            continue
        first, second = is_first_person(c), bool(set(_words(c)) & _SECOND)
        if whose == "self" and second and not first:
            return True
        if whose == "user" and first and not second:
            return True
    return False


def correct_answer(reply: str, key: str, question: str = "", told=None, whose=None) -> bool:
    """Key present as a word, not in the question, not a tautology, about the subject asked (names no other fact-bank
    subject), not a first-person echo of the told sentence. With whose given, a wrong-perspective clause (see
    wrong_perspective) also voids the reply; with whose=None the behaviour is exactly the earlier scorer's."""
    if not has_key(reply, key) or has_key(question, key) or is_tautology(reply, key):
        return False
    if wrong_subject(reply, question):
        return False
    if whose is not None and wrong_perspective(reply, key, whose):
        return False
    return not is_echo(reply, key, told)
