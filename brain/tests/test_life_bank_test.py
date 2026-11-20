# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (mechanical checks of the TEST split content)
"""Mechanical validator of brain/eval/life_bank_test.py (TEST split). CPU only, no model."""
import re
from collections import Counter

import brain.eval.life_bank as DEV
import brain.eval.life_bank_test as T
from brain.brain import _is_telling

ITEMS = T.H + T.AB + T.E + T.F + T.D
NON_H = T.AB + T.E + T.F + T.D
GREET = list(T.GREETINGS)


def inc_texts(which=("X", "Y", "new")):
    return [i[w] for v in T.INCIDENTS.values() for i in v for w in which]


def tellings(items=ITEMS):
    return [(i.id, i.telling) for i in items if i.telling]


def has(word, text):
    return re.search(r"\b" + re.escape(word) + r"\b", text, re.I) is not None


def open_texts():
    return [o if isinstance(o, str) else " ".join(map(str, o)) for o in T.OPEN]


def test_1_tellings_pass_is_telling():
    bad = [(i, t) for i, t in tellings() if not _is_telling(t)]
    bad += [("inc", t) for t in inc_texts(("X", "Y")) if not _is_telling(t)]
    assert not bad, bad


def test_2a_keys_lowercase_and_in_own_telling():
    bad = []
    for it in ITEMS:
        if it.telling is None:
            continue
        if it.cls == "H":   # H is the shared SESSION_1 anchor (identical to DEV); its extra keys are accepted answer forms
            if not any(has(k, it.telling) for k in it.keys):
                bad.append((it.id, it.keys, "no key in own telling"))
            continue
        for k in it.keys + it.gist_keys:
            if k != k.lower():
                bad.append((it.id, k, "not lowercase"))
            if not has(k, it.telling):
                bad.append((it.id, k, "not whole word in own telling"))
    assert not bad, bad


def test_2b_keys_not_in_other_texts():
    others_common = [("small_talk", s) for s in T.SMALL_TALK] + [("greeting", s) for s in GREET] \
        + [("open", s) for s in open_texts()] + [("incident", s) for s in inc_texts()]
    bad = []
    for it in ITEMS:
        if it.telling is None:
            continue
        for k in it.keys + it.gist_keys:
            for oid, ot in tellings():
                if oid != it.id and has(k, ot):
                    bad.append((it.id, k, "in telling of " + oid))
            for n, ot in others_common:
                if has(k, ot):
                    bad.append((it.id, k, "in " + n + ": " + ot))
    assert not bad, bad


def test_2c_non_h_disjoint_from_dev():
    dev_items = DEV.H + DEV.AB + DEV.E + DEV.F + DEV.D
    dev_keys = {k.lower() for x in dev_items for k in x.keys + x.gist_keys}
    dev_tell = [x.telling for x in dev_items if x.telling]
    bad = []
    for it in NON_H:
        for k in it.keys + it.gist_keys:
            if k.lower() in dev_keys:
                bad.append((it.id, k, "is a DEV key"))
            for dt in dev_tell:
                if has(k, dt):
                    bad.append((it.id, k, "in DEV telling: " + dt))
    assert not bad, bad


def test_3_ids_and_endings():
    ids = [x.id for x in ITEMS]
    assert not [i for i, c in Counter(ids).items() if c > 1]
    bad = []
    for x in ITEMS:
        for f in ("direct", "reworded"):
            if not getattr(x, f).endswith("?"):
                bad.append((x.id, f, getattr(x, f)))
        if not x.partial.endswith("..."):
            bad.append((x.id, "partial", x.partial))
    assert not bad, bad


def test_4_counts():
    assert len(T.H) == 6
    import dataclasses as dc
    assert [dc.asdict(x) for x in T.H] == [dc.asdict(x) for x in DEV.H]
    assert len(T.AB) == 48
    assert len(T.E) == 20
    bad = [x.id for x in T.E if not (x.context and x.gist_q and x.gist_keys)]
    assert not bad, bad
    assert len(T.F) == 60
    assert len(T.D) == 18
    hid = {h.id for h in T.H}
    assert not [x.id for x in T.D if x.dup_of not in hid]
    assert len(T.NEVER) == 12
    assert len(T.SMALL_TALK) == 30
    assert len(T.GREETINGS) == 8
    assert len(T.OPEN) == 12
    assert set(T.INCIDENTS) == {"sysadmin", "home repair"}
    bad = []
    for d, lst in T.INCIDENTS.items():
        if len(lst) != 8:
            bad.append((d, "count", len(lst)))
        for i in lst:
            for f in ("X", "Y", "new", "sub"):
                if not i.get(f):
                    bad.append((d, i.get("sub"), "missing " + f))
            if not i.get("new", "").endswith("What would you check first?"):
                bad.append((d, i.get("sub"), "new ending"))
    assert not bad, bad


def test_4b_d_uses_all_four_h_frames():
    # the four H frames with a told sentence are pixel, admin, marguerite, volvo
    frames = {x.dup_of for x in T.D}
    assert frames == {"pixel", "admin", "marguerite", "volvo"}, frames


def test_5_first_four_words_unique():
    seen, bad = {}, []
    texts = tellings() + [("inc-X:" + i["sub"], i["X"]) for v in T.INCIDENTS.values() for i in v]
    for tid, t in texts:
        k = tuple(w.lower().strip(".,;") for w in t.split()[:4])
        if k in seen:
            bad.append((tid, seen[k], " ".join(k)))
        seen[k] = tid
    assert not bad, bad


def test_6_no_ai_words():
    bad = []
    for tid, t in tellings() + [("inc", t) for t in inc_texts(("X", "Y"))]:
        for w in ("ai", "assistant", "brain", "robot"):
            if has(w, t):
                bad.append((tid, w, t))
    assert not bad, bad
