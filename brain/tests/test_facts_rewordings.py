import re

from brain.experiments.facts import FACTS, PARA
from brain.experiments.facts_rewordings import REWORDINGS

STOP = set("a an the is are was were am be my i me of in on at to do does did what whats which who whos where when how "
           "any there can cant it its that this for and or by with as so s".split())


def _words(text):
    return re.findall(r"[a-z]+", text.lower().replace("'", ""))


def _grams(seq):
    return {tuple(seq[i:i + 3]) for i in range(len(seq) - 2)}


def _content(text):
    return [w for w in _words(text) if w not in STOP]


def test_shape():
    assert len(REWORDINGS) == len(FACTS)
    for pair in REWORDINGS:
        assert len(pair) == 2 and pair[0] != pair[1]


def test_keep_key_word():
    for (s, _, k), rw in zip(FACTS, REWORDINGS):
        for r in rw:
            assert k in r.lower(), (k, r)
            assert r != s


def test_no_para_question_wording():
    for q in PARA:
        for r in (x for rw in REWORDINGS for x in rw):
            assert not _grams(_content(q)) & _grams(_content(r)), (q, r)
            assert not _grams(_words(q)) & _grams(_words(r)), (q, r)
