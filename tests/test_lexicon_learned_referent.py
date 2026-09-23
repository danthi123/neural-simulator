"""Guards for the learned open-vocab referent detector wiring into the D6 multi-referent WM organ (default OFF).

No corpus needed: the learned lexicon is replaced by a fake with the same `is_referent` interface.
"""
import os
import re

import pytest

import research.runners.d6_multiref_wm_production_organ as D6
from research.runners import lexicon_learned_referent as LR

SENTS = [
    "the wolf watches the owl",
    "the fox and the wolf walked in",
    "Mary gave the monkey a banana and the cookie",
    "who are we talking about?",
    "what are you keeping in mind",
    "the dog and the cat and the bird and the fish and the horse and the cow",
    "",
]


def _hand_only_reference(text, max_refs=D6.R_MAX):
    """Verbatim copy of the pre-2026-09-23 `extract_referents` body (the byte-identity oracle)."""
    raw = re.compile(r"[A-Za-z']+").findall(text or "")
    refs = []
    for i, w in enumerate(raw):
        lw = w.lower()
        is_lex = lw in D6._REFERENT_NOUNS
        is_proper = (len(w) > 1 and w[0].isupper() and i > 0 and lw not in D6._STOP)
        if (is_lex or is_proper) and lw not in D6._PRONOUNS:
            if lw not in refs:
                refs.append(lw)
        if len(refs) >= max_refs:
            break
    return refs[:min(max_refs, D6._BINDER_K)]


class _Fake:
    def __init__(self, nouns, lesioned=False):
        self.nouns = set(nouns)
        self.lesioned = lesioned

    def is_referent(self, w):
        return (not self.lesioned) and w in self.nouns


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("BRAIN_LEARNED_REFERENT_LEXICON", raising=False)
    monkeypatch.delenv("BRAIN_LEARNED_REFERENT_LESION", raising=False)


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("BRAIN_MULTIREF_ON_ABSTAIN", raising=False)
    assert D6.learned_referent_enabled() is False
    assert D6.learned_referent_lesioned() is False
    assert D6.multiref_on_abstain_enabled() is False


@pytest.mark.parametrize("s", SENTS)
def test_off_is_byte_identical_to_hand_path(s):
    assert D6.extract_referents(s) == _hand_only_reference(s)


def test_learned_extends_scope_and_lesion_reverts():
    fake = _Fake({"owl", "monkey", "banana", "cookie", "mind"})
    assert D6.extract_referents("the wolf watches the owl", referent_lexicon=fake) == ["wolf", "owl"]
    # hold-query vocabulary never becomes a referent, even if the learner calls it a noun
    assert D6.extract_referents("what are you keeping in mind", referent_lexicon=fake) == []
    les = _Fake(fake.nouns, lesioned=True)
    for s in SENTS:
        assert D6.extract_referents(s, referent_lexicon=les) == _hand_only_reference(s)


def test_hand_noun_seeds_match_organ_table_common_nouns():
    names = {"john", "mary", "alice", "bob", "sam", "tom", "anna", "lucy"}
    assert set(LR.HAND_NOUN_SEEDS) == set(D6._REFERENT_NOUNS) - names
    assert not set(LR.NONNOUN_SEEDS) & set(D6._REFERENT_NOUNS)
