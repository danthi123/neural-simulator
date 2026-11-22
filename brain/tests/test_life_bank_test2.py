# Capability: Lifelong learning
# Record: brain/research/2026-11-22-fresh-test-split-design.md
# Expected-fail: none (mechanical checks of TEST split v2 content; skipped while the v2 files do not exist)
"""Mechanical validator of brain/eval/life_bank_test2.py and brain/eval/life_cues_test2.py (TEST split v2, design
section 4 and the host-side wording check of section 3). CPU only, no model.

BLIND-WRITER RULE: a writer who must not see v1 TEST content reads these assertion messages. Every message names only
the offending v2 id, the offending v2 word or n-gram (the clashing token itself) and the rule. v1 TEST text, v1 keys
and v1 items are never printed beyond that clashing token. Do not add messages that echo v1 or DEV text.

The tests skip (importorskip) while the v2 files are absent. Set TEST2_BANK_MODULE / TEST2_CUES_MODULE to point the
checks at another module (used only to dry-run the validator itself).
"""
import ast
import dataclasses as dc
import importlib
import os
import re
import types
from collections import Counter

import pytest

import brain.eval.life_bank as DEV
import brain.eval.life_bank_test as T1
import brain.eval.life_cues as LC
import brain.eval.life_heldout as HELD
from brain.brain import _is_telling
from brain.experiments.strict_scorer import _base

BANK_MOD = os.environ.get("TEST2_BANK_MODULE", "brain.eval.life_bank_test2")
CUES_MOD = os.environ.get("TEST2_CUES_MODULE", "brain.eval.life_cues_test2")
HEADER = ("# Capability: Lifelong learning", "# Record: brain/research/2026-11-22-fresh-test-split-design.md",
          "# Expected-fail:")
SLOTS3 = ("thing", "name", "place")
ALLOWED_SLOTS = ("name", "place", "thing", "job", "event-detail")
INCIDENT_END = "What would you check first?"
# the incident `new` question ends with this fixed phrase in every split; it is excluded from the n-gram scans
FIXED_PHRASES = re.compile(re.escape(INCIDENT_END), re.I)
ADJ = frozenset("favourite favorite first last old new best usual own current".split())


@pytest.fixture(scope="module")
def B():
    return pytest.importorskip(BANK_MOD)


@pytest.fixture(scope="module")
def CU():
    return pytest.importorskip(CUES_MOD)


# ---------------------------------------------------------------- helpers (no v1 text is ever returned to a message)
def norm(text):
    return (text or "").lower().replace("’", "'").replace("‘", "'")


def toks(text):
    """Word tokens with a trailing 's folded off (strict_scorer._base), so "Pixel's" matches "pixel"."""
    return [_base(w) for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", norm(text))]


def raw_toks(text):
    return re.findall(r"[a-z0-9']+", norm(text))


def occurs(key, token_lists):
    """True if the (possibly multi-word) key occurs as whole words in any of the token lists."""
    kt = toks(key)
    if not kt:
        return False
    n = len(kt)
    for tl in token_lists:
        if n == 1:
            if kt[0] in tl:
                return True
        else:
            for i in range(len(tl) - n + 1):
                if tl[i:i + n] == kt:
                    return True
    return False


def has(word, text):
    return occurs(word, [toks(text)])


def ngrams(text, n):
    tl = raw_toks(FIXED_PHRASES.sub(" ", text))
    return {tuple(tl[i:i + n]) for i in range(len(tl) - n + 1)}


def item_texts(it):
    out = [it.telling, it.direct, it.reworded, it.partial, it.gist_q]
    return [t for t in out if t]


def inc_leaves(B, skip=()):
    out = []
    for v in B.INCIDENTS.values():
        for i in v:
            for f, val in i.items():
                if f in skip:
                    continue
                out.extend(val.values() if isinstance(val, dict) else [val])
    return [s for s in out if isinstance(s, str)]


def never_ladder(B):
    return list(getattr(B, "NEVER_LADDER", []))


def split_texts(B, with_h=True, with_opt=True):
    """[(label, text)] of every free-text string of a bank. label is an id only meaningful for the v2 bank."""
    out = []
    for it in B.H + B.AB + B.E + B.F + B.D:
        if it.cls == "H" and not with_h:
            continue
        out += [(it.id, t) for t in item_texts(it)]
    out += [(n[0], n[1]) for n in B.NEVER]
    out += [(n[0], n[1]) for n in never_ladder(B)]
    out += [("small_talk", s) for s in B.SMALL_TALK] + [("greeting", s) for s in B.GREETINGS]
    out += [("open", o[1]) for o in B.OPEN]
    out += [("incident", s) for s in inc_leaves(B, skip=() if with_opt else ("opt",))]
    return out


def bank_keys(B):
    """K(B) exactly as written in design section 4."""
    return ({_base(k) for x in B.ALL.values() for k in x.keys + x.gist_keys}
            | {_base(k) for n in B.NEVER for k in n[2]})


def h_keys(B):
    return {_base(k) for h in B.H for k in h.keys}


def non_h_key_set(B):
    """Keys (item keys, gist keys, never lure keys) of a bank that are not H keys, as folded tokens strings."""
    hk = h_keys(B)
    ks = {_base(k) for x in B.ALL.values() if x.cls != "H" for k in x.keys + x.gist_keys}
    ks |= {_base(k) for n in B.NEVER for k in n[2]}
    return ks - hk


def other_banks():
    return (DEV, T1)


def all_item_keys(*banks):
    return {_base(k) for b in banks for x in b.ALL.values() for k in x.keys + x.gist_keys}


def subject_nouns(text):
    """Possessed head nouns: the token after 'my'/'our' (skipping adjectives such as 'favourite'), 's folded."""
    tl = toks(text)
    out = set()
    for i, w in enumerate(tl[:-1]):
        if w in ("my", "our"):
            j = i + 1
            while j < len(tl) - 1 and tl[j] in ADJ:
                j += 1
            out.add(tl[j])
    return out


def is_content(w):
    return bool(LC.content(w))


def content_bigrams(text):
    tl = re.findall(r"[a-z]+", norm(text))
    return {(a, b) for a, b in zip(tl, tl[1:]) if is_content(a) and is_content(b)}


def tellings(B, items=None):
    items = items if items is not None else B.H + B.AB + B.E + B.F + B.D
    return [(i.id, i.telling) for i in items if i.telling]


def non_h(B):
    return B.AB + B.E + B.F + B.D


def seq(prefix, n):
    return [f"{prefix}{i:02d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------- files, counts, shape
def test_0_header_lines_on_v2_files(B, CU):
    for mod in (B, CU):
        lines = open(mod.__file__).read().splitlines()[:3]
        bad = [h for h, l in zip(HEADER, lines) if not l.startswith(h)]
        assert not bad and len(lines) == 3, f"{os.path.basename(mod.__file__)}: header line(s) missing or wrong: {bad}"


def test_1_counts_and_ids(B):
    assert [len(x) for x in (B.H, B.AB, B.E, B.F, B.D)] == [6, 48, 20, 60, 18], "class counts must be H 6, AB 48, E 20, F 60, D 18"
    assert [x.id for x in B.AB] == seq("ab", 48), "AB ids must be ab01..ab48 in order"
    assert [x.id for x in B.E] == seq("e", 20), "E ids must be e01..e20 in order"
    assert [x.id for x in B.F] == seq("f", 60), "F ids must be f01..f60 in order"
    assert [x.id for x in B.D] == seq("d", 18), "D ids must be d01..d18 in order"
    assert len(B.NEVER) == 12 and len(B.SMALL_TALK) == 30 and len(B.GREETINGS) == 8 and len(B.OPEN) == 12, \
        "NEVER 12, SMALL_TALK 30, GREETINGS 8, OPEN 12 required"
    ids = [x.id for x in B.H + B.AB + B.E + B.F + B.D]
    assert not [i for i, c in Counter(ids).items() if c > 1], "duplicate item ids"
    assert list(B.ALL) == ids and all(B.ALL[x.id] is x for x in B.H + B.AB + B.E + B.F + B.D), \
        "ALL must be {x.id: x for x in H + AB + E + F + D}"


def test_1b_split_stats(B):
    n_inc = sum(len(v) for v in B.INCIDENTS.values())
    want = {"H": 6, "AB": 48, "E": 20, "F": 60, "D": 18, "never": 12, "small_talk": 30, "open": 12, "incidents": 16,
            "incident_tellings": 32, "items": 6 + 48 + 20 + 60 + 18 + 12 + 30 + 12 + 16}
    assert n_inc == 16 and B.split_stats() == want, "split_stats() must return the same keys and formula as the other banks"


def test_2_h_identical_to_v1(B):
    assert [dc.asdict(x) for x in B.H] == [dc.asdict(x) for x in T1.H], "H must be identical to the shared H block (copy it from the DEV bank)"


def test_3_item_fields(B):
    bad = []
    for x in B.H + B.AB + B.E + B.F + B.D:
        for f in ("direct", "reworded"):
            if not getattr(x, f).endswith("?"):
                bad.append((x.id, f, "must end with ?"))
        if not x.partial.endswith("..."):
            bad.append((x.id, "partial", "must end with ..."))
        if x.slot not in ALLOWED_SLOTS:
            bad.append((x.id, "slot", "not an allowed slot"))
        if x.cls != "H" and not x.keys:
            bad.append((x.id, "keys", "empty"))
    assert not bad, bad


def test_3b_e_items(B):
    bad = []
    for x in B.E:
        if not (x.context and x.gist_q and x.gist_keys):
            bad.append((x.id, "needs context, gist_q and gist_keys"))
        if x.slot != "event-detail":
            bad.append((x.id, "E slot must be event-detail"))
        for f in ("detail_slot", "gist_slot"):
            if x.extra.get(f) not in SLOTS3:
                bad.append((x.id, f"extra[{f!r}] must be one of thing/name/place"))
        if not x.gist_q.endswith("?"):
            bad.append((x.id, "gist_q must end with ?"))
    assert not bad, bad


def test_4_slot_counts(B):
    f = Counter(x.slot for x in B.F)
    assert f == Counter({"thing": 38, "name": 11, "place": 11}), f"F slots must be thing 38, name 11, place 11 (got {dict(f)}); no job"
    assert {x.slot for x in B.D} == {"name"} and len(B.D) == 18, "every D item must have slot name"
    bad = [x.id for x in B.F + B.D if x.slot == "job"]
    assert not bad, f"job slot is not allowed on F or D (ids {bad})"


def test_4b_d_frames(B):
    hid = {h.id: h for h in B.H}
    bad = [x.id for x in B.D if x.dup_of not in hid or hid[x.dup_of].telling is None or hid[x.dup_of].slot != "name"]
    assert not bad, f"D dup_of must be a name-slot H item with a telling (pixel or marguerite): ids {bad}"
    assert {x.dup_of for x in B.D} == {"pixel", "marguerite"}, "D must use both name-slot H frames (pixel and marguerite)"


def test_5_never_shapes(B):
    bad = []
    ids = [n[0] for n in B.NEVER]
    if len(set(ids)) != 12:
        bad.append("NEVER ids must be 12 distinct ids")
    for n in B.NEVER:
        if not (isinstance(n, tuple) and len(n) == 3 and isinstance(n[1], str) and isinstance(n[2], tuple)):
            bad.append((n[0], "NEVER entries are tuples (id, text, lure_keys-tuple)"))
            continue
        if not n[1].endswith("?"):
            bad.append((n[0], "must end with ?"))
    assert not bad, bad
    key_pool = {_base(k) for x in B.ALL.values() for k in x.keys}
    bad = [(n[0], k) for n in B.NEVER for k in n[2] if _base(k) not in key_pool]
    assert not bad, f"lure key is not a key of any v2 item: {bad}"
    assert isinstance(B.NEVER_SLOT, dict) and set(B.NEVER_SLOT) == set(ids), "NEVER_SLOT keys must be exactly the NEVER ids"
    assert all(v in SLOTS3 for v in B.NEVER_SLOT.values()), "NEVER_SLOT values must be thing, name or place"
    assert len({n[1] for n in B.NEVER}) == 12, "NEVER texts must be distinct"


def test_5b_never_ladder_shape(B):
    nl = never_ladder(B)
    assert len(nl) == 48, "NEVER_LADDER must have 48 tuples (id, text, slot)"
    bad = []
    for n in nl:
        if not (isinstance(n, tuple) and len(n) == 3 and isinstance(n[0], str) and isinstance(n[1], str)):
            bad.append((n[0] if n else "?", "tuple (id, text, slot)"))
            continue
        if n[2] not in SLOTS3:
            bad.append((n[0], "slot must be thing, name or place (no job)"))
        if not n[1].endswith("?"):
            bad.append((n[0], "must end with ?"))
    assert not bad, bad
    assert Counter(n[2] for n in nl) == Counter({"thing": 22, "name": 14, "place": 12}), "NEVER_LADDER slots: thing 22, name 14, place 12"
    lid = [n[0] for n in nl]
    assert len(set(lid)) == 48, "NEVER_LADDER ids must be distinct"
    assert not set(lid) & ({n[0] for n in B.NEVER} | set(B.ALL)), "NEVER_LADDER ids must differ from NEVER and item ids"
    texts = [norm(n[1]) for n in nl]
    assert len(set(texts)) == 48, "NEVER_LADDER texts must be distinct"
    assert not set(texts) & {norm(n[1]) for n in B.NEVER}, "NEVER_LADDER texts must differ from NEVER texts"
    dup = [n[0] for n in nl if norm(n[1]) in {norm(x[1]) for b in other_banks() for x in b.NEVER}]
    dup += [n[0] for n in B.NEVER if norm(n[1]) in {norm(x[1]) for b in other_banks() for x in b.NEVER}]
    assert not dup, f"question text already used by another split: ids {dup}"
    carriers = Counter(x.slot for x in B.F + B.D + B.AB)
    bad = [s for s in {n[2] for n in nl} if carriers[s] < 3]
    assert not bad, f"slot {bad}: fewer than 3 v2 F/D/AB items carry it"


def test_5c_never_topics(B):
    qs = {n[1] for n in B.NEVER}
    assert isinstance(B.NEVER_TOPICS, dict) and set(B.NEVER_TOPICS) == qs, "NEVER_TOPICS keys must be exactly the NEVER question texts"
    keys = all_item_keys(B, DEV, T1)
    bad = []
    for q, tp in B.NEVER_TOPICS.items():
        if not (isinstance(tp, str) and tp.strip()):
            bad.append((q, "topic must be a non-empty string"))
            continue
        for k in keys:
            if occurs(k, [toks(tp)]):
                bad.append((q, f"topic contains the key word {k!r}"))
    assert not bad, bad


def test_6_incidents(B):
    assert set(B.INCIDENTS) == {"sysadmin", "home repair"}, "INCIDENTS domains must be sysadmin and home repair"
    bad = []
    for d, lst in B.INCIDENTS.items():
        if len(lst) != 8:
            bad.append((d, "needs 8 incidents"))
        for i in lst:
            for f in ("X", "Y", "new", "sub"):
                if not i.get(f):
                    bad.append((d, i.get("sub"), "missing " + f))
            if not i.get("new", "").endswith(INCIDENT_END):
                bad.append((d, i.get("sub"), f"new must end with: {INCIDENT_END}"))
            if not (isinstance(i.get("opt"), dict) and set(i["opt"]) == {"X", "Y"}):
                bad.append((d, i.get("sub"), "opt must be a dict with X and Y"))
    assert not bad, bad
    assert B.INCIDENT_DAYS == T1.INCIDENT_DAYS == DEV.INCIDENT_DAYS, "INCIDENT_DAYS must stay unchanged"


def test_6b_open_and_smalltalk(B):
    pat = lambda b: Counter((o[0], o[2] is None) for o in b.OPEN)
    assert all(isinstance(o, tuple) and len(o) == 3 for o in B.OPEN), "OPEN entries are 3-tuples (id, prompt, hint or None)"
    assert pat(B) == pat(T1), "OPEN needs 2 each of story (with a hint), feeling, advice, opinion, knowledge, chat (hint None)"
    bad = [o[0] for o in B.OPEN if _is_telling(o[1])]
    assert not bad, f"OPEN prompts must be requests or questions, not tellings: ids {bad}"
    assert len({norm(s) for s in B.SMALL_TALK}) == 30 and len({norm(s) for s in B.GREETINGS}) == 8, "SMALL_TALK and GREETINGS must be distinct"


# ---------------------------------------------------------------- tellings and keys
def test_7_tellings_are_tellings(B):
    bad = []
    for tid, t in tellings(B) + [("inc:" + i["sub"], i[w]) for v in B.INCIDENTS.values() for i in v for w in ("X", "Y")]:
        if not _is_telling(t):
            bad.append((tid, "starts like a request or is not a telling"))
        if t.rstrip().endswith("?"):
            bad.append((tid, "a telling must not end with ?"))
    assert not bad, bad


def test_7b_first_four_words_unique(B):
    seen, bad = {}, []
    for tid, t in tellings(B) + [("inc-X:" + i["sub"], i["X"]) for v in B.INCIDENTS.values() for i in v]:
        k = tuple(w.lower().strip(".,;") for w in t.split()[:4])
        if k in seen:
            bad.append((tid, "first four words repeat another telling", " ".join(k)))
        seen[k] = tid
    assert not bad, bad


def test_7c_no_ai_words(B):
    bad = []
    for tid, t in tellings(B) + [("inc", t) for t in inc_leaves(B, skip=("opt", "new", "sub"))]:
        for w in ("ai", "assistant", "brain", "robot"):
            if has(w, t):
                bad.append((tid, w, "AI word"))
    assert not bad, bad


def test_8_keys_lowercase_and_in_own_telling(B):
    bad = []
    for it in non_h(B):
        for k in it.keys + it.gist_keys:
            if k != k.lower():
                bad.append((it.id, k, "key not lowercase"))
            if not has(k, it.telling):
                bad.append((it.id, k, "key is not a whole word of its own telling"))
    for h in B.H:
        if h.telling is not None and not any(has(k, h.telling) for k in h.keys):
            bad.append((h.id, "no key in own telling"))
    assert not bad, bad


def test_8b_keys_not_in_other_v2_texts(B):
    others = [("small_talk", s) for s in B.SMALL_TALK] + [("greeting", s) for s in B.GREETINGS] \
        + [("open", o[1]) for o in B.OPEN] + [("incident", s) for s in inc_leaves(B)]
    questions = [(n[0], n[1]) for n in B.NEVER] + [(n[0], n[1]) for n in never_ladder(B)]
    bad = []
    for it in non_h(B):
        for k in it.keys + it.gist_keys:
            for oid, ot in tellings(B):
                if oid != it.id and has(k, ot):
                    bad.append((it.id, k, f"key occurs in the telling of {oid}"))
            for n, ot in others:
                if has(k, ot):
                    bad.append((it.id, k, f"key occurs in a v2 {n} text"))
            for qid, qt in questions:
                if has(k, qt):
                    bad.append((it.id, k, f"key occurs in never question {qid}"))
    assert not bad, bad


# ---------------------------------------------------------------- key disjointness and word / n-gram clashes
def test_9_key_set_disjointness_exact_rule(B):
    k2, ko, hk = bank_keys(B), bank_keys(DEV) | bank_keys(T1), h_keys(B)
    extra = sorted((k2 & ko) - hk)
    missing = sorted(hk - (k2 & ko))
    owners = {}
    for x in B.ALL.values():
        for k in x.keys + x.gist_keys:
            owners.setdefault(_base(k), []).append(x.id)
    for n in B.NEVER:
        for k in n[2]:
            owners.setdefault(_base(k), []).append(n[0])
    assert not extra, "v2 key(s) also used as a key in another split (rule K(T2) & (K(DEV) | K(T1)) == H keys): " \
        + str([(k, owners.get(k)) for k in extra])
    assert not missing, f"H keys must all be present in K(T2): {missing}"
    assert k2 & ko == hk


def test_9b_v2_keys_not_words_in_dev_or_v1_text(B):
    v1_tok = [toks(t) for b in other_banks() for _, t in split_texts(b)]
    owners = {}
    for x in non_h(B):
        for k in x.keys + x.gist_keys:
            owners.setdefault(_base(k), []).append(x.id)
    for n in B.NEVER:
        for k in n[2]:
            if _base(k) not in h_keys(B):
                owners.setdefault(_base(k), []).append(n[0])
    bad = [(i, k) for k, ids in owners.items() if k not in h_keys(B) and occurs(k, v1_tok) for i in ids]
    assert not bad, "v2 key occurs as a word in DEV or v1 text (ids, key): " + str(sorted(bad))


def test_9c_other_keys_not_words_in_v2_text(B):
    other_keys = set()
    for b in other_banks():
        other_keys |= non_h_key_set(b)
    other_keys -= h_keys(B)
    v2 = [(lab, toks(t)) for lab, t in split_texts(B, with_h=False)]
    toklists = [tl for _, tl in v2]
    bad = []
    for k in sorted(other_keys):
        if not occurs(k, toklists):
            continue
        for lab, tl in v2:
            if occurs(k, [tl]):
                bad.append((lab, k))
    assert not bad, "v2 text contains a word that is a key in DEV or v1 (v2 id or text kind, the word): " + str(sorted(set(bad)))


def test_10_no_5gram_shared_with_dev_or_v1(B):
    other = set()
    for b in other_banks():
        for lab, t in split_texts(b, with_opt=False):
            other |= ngrams(t, 5)
    bad = []
    for lab, t in split_texts(B, with_h=False, with_opt=False):
        for g in sorted(ngrams(t, 5) & other):
            bad.append((lab, " ".join(g)))
    assert not bad, "v2 text shares a 5-word run with DEV or v1 text (v2 id or text kind, the 5 words): " + str(bad)


def test_10b_no_4gram_with_heldout(B):
    held = set()
    for t in HELD.HELDOUT:
        held |= ngrams(t, 4)
    bad = [(lab, " ".join(g)) for lab, t in split_texts(B, with_h=False) for g in sorted(ngrams(t, 4) & held)]
    assert not bad, "v2 text shares a 4-word run with the language-guard held-out set (v2 id, the 4 words): " + str(bad)


# ---------------------------------------------------------------- never-question wording rules (design section 3)
def _never_questions(B):
    return [(n[0], n[1]) for n in B.NEVER] + [(n[0], n[1]) for n in never_ladder(B)]


def test_11_never_questions_contain_no_key(B):
    keys = all_item_keys(B, DEV, T1) | {_base(k) for b in (B, DEV, T1) for n in b.NEVER for k in n[2]}
    bad = []
    for qid, q in _never_questions(B):
        ql = toks(q)
        for k in sorted(keys):
            if occurs(k, [ql]):
                bad.append((qid, k))
    assert not bad, "never question contains an item key (key or gist key of a v2, v1 or DEV item) (v2 id, the word): " + str(bad)


def test_11b_never_subject_not_subject_of_v2_item(B):
    subj = {}
    for it in B.H + B.AB + B.E + B.F + B.D:
        for t in (it.telling, it.direct, it.gist_q):
            for s in subject_nouns(t or ""):
                subj.setdefault(s, it.id)
    bad = []
    for qid, q in _never_questions(B):
        for s in subject_nouns(q):
            if s in subj:
                bad.append((qid, s, f"is the subject of v2 item {subj[s]}"))
    assert not bad, "never question subject noun (the word after 'my', adjectives skipped) must not be the subject of any v2 item: " + str(bad)


# words that head many questions or tellings and carry no item-specific subject (generic verbs, pronouns, kinds)
GENERIC_HEADS = frozenset("i we you on is has have do get got give take put use like love live play print printed "
                          "go eat buy shade colour town city man woman person friend name day".split()) - {"print", "printed"}
QWORDS = ("who", "which", "what", "where")
QSKIP = frozenset("is was are were do did does should could would can will i my our the a an of kind sort type brand "
                  "make model".split())
# near-synonym families: a never question whose head is in a family with a v2 subject asks about the same thing
SUBJECT_FAMILIES = (
    frozenset("post postman postwoman parcel letter deliver courier newspaper".split()),
    frozenset("tea coffee cuppa milk".split()),
    frozenset("plumber plumb boiler pipe".split()),
    frozenset("pub bar inn".split()),
    frozenset("teach teacher headmaster headmistress school".split()),
    frozenset("beach seaside sea pier".split()),
    frozenset("watch wristwatch".split()),
    frozenset("umbrella brolly".split()),
    frozenset("loft attic".split()),
    frozenset("pet vet".split()),
)


def _stem(w):
    for suf in ("ing", "ed", "es", "s", "er"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def question_heads(text):
    """The head word of a who/which/what/where clause (the first word after it that is not an auxiliary, article,
    'kind of'-type word or adjective: 'Which plumber...' -> plumber, 'Who delivers...' -> delivers) and the main verb
    of 'do/did/does I <verb>'."""
    tl = toks(text)
    out = set()
    for i, w in enumerate(tl):
        if w in QWORDS:
            j = i + 1
            while j < len(tl) and (tl[j] in QSKIP or tl[j] in ADJ):
                j += 1
            if j < len(tl):
                out.add(tl[j])
        if w in ("do", "did", "does") and i + 2 < len(tl) and tl[i + 1] == "i":
            out.add(tl[i + 2])
    return out


def name_bearers(text):
    """Role nouns that carry a told name in a telling: '<noun> named/called X' and 'the <noun>, Name,'."""
    raw = re.findall(r"[A-Za-z']+|,", text or "")
    out = set()
    for i, w in enumerate(raw):
        if w.lower() in ("named", "called") and i > 0 and raw[i - 1] != ",":
            out.add(_base(raw[i - 1].lower()))
        if w == "," and 0 < i < len(raw) - 1 and raw[i + 1][:1].isupper() and raw[i - 1][:1].islower():
            out.add(_base(raw[i - 1].lower()))
    return out


GENERIC_STEMS = frozenset(_stem(w) for w in GENERIC_HEADS)


def _family(w):
    st = _stem(w)
    return {_stem(x) for f in SUBJECT_FAMILIES if st in {_stem(y) for y in f} for x in f} | {st}


def test_11d_never_head_not_subject_of_v2_item(B):
    """The head noun or verb of a never question (who/which/what X, do I <verb>), and its possessed noun, must not be
    the subject, question head or name-bearing role of a v2 item, also via a compound ('wristwatch' holds 'watch') or a
    near-synonym family ('parcels' ~ postman): such a question would score a carried-over real memory as chance."""
    subj = {}
    for it in B.H + B.AB + B.E + B.F + B.D:
        ws = set()
        for t in (it.telling, it.direct, it.reworded, it.gist_q):
            ws |= subject_nouns(t or "")
        for t in (it.direct, it.reworded, it.gist_q):
            ws |= question_heads(t or "")
        ws |= name_bearers(it.telling)
        for w in ws:
            if _stem(w) not in GENERIC_STEMS:
                subj.setdefault(_stem(w), it.id)
    bad = []
    for qid, q in _never_questions(B):
        for h in sorted(w for w in question_heads(q) | subject_nouns(q) if _stem(w) not in GENERIC_STEMS):
            for f in sorted(_family(h)):
                if f in subj:
                    bad.append((qid, h, f"~{f}, a subject of v2 item {subj[f]}"))
            for s in sorted(subj):
                if len(s) >= 5 and s != _stem(h) and s in h:
                    bad.append((qid, h, f"compound holding {s!r}, a subject of v2 item {subj[s]}"))
    assert not bad, "never question head word is a v2 item subject (never id, head word, why): " + str(bad)


def test_11d_check_bites():
    """The head check catches the clashes it was written for (a never question about a told item's subject)."""
    b = type("Bk", (), {})()
    told = [types.SimpleNamespace(id="f05", telling="My postman is a cheerful man named Alphonse.", direct="Who delivers my letters?",
                                  reworded="Who brings my post?", gist_q=None),
            types.SimpleNamespace(id="e11", telling="The pipe burst and the plumber, Dermot, came.", direct="Who came?",
                                  reworded="Who fixed the pipe?", gist_q=None),
            types.SimpleNamespace(id="ab17", telling="My club meets at a pub called the Marmion.", direct="Where does my club meet?",
                                  reworded="Which pub hosts my club?", gist_q=None),
            types.SimpleNamespace(id="ab29", telling="My watch is a Seiko.", direct="What make is my watch?",
                                  reworded="Which watch do I wear?", gist_q=None)]
    b.H, b.AB, b.E, b.F, b.D = [], [told[2], told[3]], [told[1]], [told[0]], []
    b.NEVER = []
    b.NEVER_LADDER = [("x1", "Who delivers my parcels?", "name"), ("x2", "Which plumber fixed my boiler?", "name"),
                      ("x3", "Which pub do I drink in on Fridays?", "place"),
                      ("x4", "What does my wristwatch strap look like?", "thing"),
                      ("x5", "What is stitched on my apron?", "thing")]
    with pytest.raises(AssertionError) as e:
        test_11d_never_head_not_subject_of_v2_item(b)
    msg = str(e.value)
    assert all(f"'{x}'" in msg for x in ("x1", "x2", "x3", "x4")) and "'x5'" not in msg


def test_11c_never_content_overlap(B):
    texts = [(lab, t) for lab, t in split_texts(B) if lab not in {n[0] for n in B.NEVER} | {n[0] for n in never_ladder(B)}]
    bad = []
    for qid, q in _never_questions(B):
        qc, qb = LC.content(q), content_bigrams(q)
        for lab, t in texts:
            shared = qc & LC.content(t)
            if len(shared) > 1:
                bad.append((qid, lab, "more than 1 content word shared: " + ",".join(sorted(shared))))
            gb = qb & content_bigrams(t)
            if gb:
                bad.append((qid, lab, "shares a content bigram: " + " ".join(sorted(gb)[0])))
    assert not bad, "never question vs v2 text (never id, v2 id or text kind, rule): " + str(bad)


# ---------------------------------------------------------------- life_cues_test2.py
@pytest.fixture(scope="module")
def cues(CU):
    assert isinstance(getattr(CU, "TEST2", None), dict), "life_cues_test2.py must define the dict TEST2"
    return CU.TEST2


def test_12_cue_file_shape_and_ids(B, cues, CU):
    assert set(cues) == {x.id for x in B.E + B.F}, "TEST2 keys must be exactly the E and F item ids"
    bad = [i for i, v in cues.items() if not (isinstance(v, tuple) and len(v) == 2 and all(isinstance(s, str) for s in v))]
    assert not bad, f"TEST2 values must be (C2, C3) tuples of two strings: ids {bad}"
    src = open(CU.__file__).read()
    tree = ast.parse(src)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = " ".join((getattr(n, "module", "") or "") + " " + " ".join(a.name for a in n.names) for n in imports)
    assert "life_cues" not in names, "life_cues_test2.py is data only: it must not import life_cues or life_cues_split (the shim imports it)"


def test_12b_cue_texts(B, cues):
    allkeys = all_item_keys(B) | {_base(k) for n in B.NEVER for k in n[2]}
    bad = []
    for iid, (c2, c3) in cues.items():
        it = B.ALL[iid]
        for nm, t in (("C2", c2), ("C3", c3)):
            if not t.endswith(LC.TAIL):
                bad.append((iid, nm, f"must end with {LC.TAIL!r}"))
            if "\n" in t:
                bad.append((iid, nm, "single line only"))
            for k in it.keys:
                if has(k, t):
                    bad.append((iid, nm, f"contains the item's own key {k!r}"))
            for k in sorted(allkeys - {_base(k) for k in it.keys}):
                if occurs(k, [toks(t)]):
                    bad.append((iid, nm, f"contains the key {k!r} of another v2 item"))
        if LC.content(c2) == LC.content(c3):
            bad.append((iid, "C3", "must describe a different situation than C2"))
    assert not bad, bad


def test_12c_cue_overlap_rule(B, cues):
    bad = []
    for iid, (c2, c3) in cues.items():
        tel = LC.content(B.ALL[iid].telling)
        o2, o3 = len(LC.content(c2) & tel), len(LC.content(c3) & tel)
        o4 = max(len(LC.content(p) & tel) for p in LC.C4_PROMPTS)
        if abs(o2 - o3) > 1:
            bad.append((iid, f"|overlap(C2) - overlap(C3)| must be <= 1 (got {o2} and {o3})"))
        if o2 < 1:
            bad.append((iid, "C2 must share at least 1 content word with the telling"))
        if o4 != 0:
            bad.append((iid, "the telling must not contain a content word of the generic C4 prompts"))
    assert not bad, bad


def test_12d_cue_clashes_with_other_splits(B, cues):
    other_keys = set()
    for b in other_banks():
        other_keys |= non_h_key_set(b)
    other_keys -= h_keys(B)
    other_cues = set()
    for d in (LC.DEV, LC.TEST):
        for pair in d.values():
            for t in pair:
                other_cues |= ngrams(t, 5)
    held = set()
    for t in HELD.HELDOUT:
        held |= ngrams(t, 4)
    bad = []
    for iid, pair in cues.items():
        for nm, t in zip(("C2", "C3"), pair):
            tl = toks(t)
            for k in sorted(other_keys):
                if occurs(k, [tl]):
                    bad.append((iid, nm, f"contains the word {k!r}, a key in DEV or v1"))
            for g in sorted(ngrams(t, 5) & other_cues):
                bad.append((iid, nm, "5-word run shared with DEV or v1 cue text: " + " ".join(g)))
            for g in sorted(ngrams(t, 4) & held):
                bad.append((iid, nm, "4-word run shared with the held-out set: " + " ".join(g)))
    assert not bad, bad


def test_12e_life_cues_wiring_if_present(B, cues):
    from brain.eval import life_cues_split as CS
    assert "test2" not in LC.CUES, "life_cues.py stays frozen (registered sha256): TEST v2 goes through life_cues_split"
    assert CS.cues("test2") is cues, "life_cues_split.cues('test2') must be life_cues_test2.TEST2"
    ov = CS.overlaps("test2")
    assert set(ov) == set(cues)
