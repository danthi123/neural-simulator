"""What the learned referent lexicon changes in the load-bearing battery's parse (language lane E, 2026-09-24).

WHY THIS EXISTS. `BRAIN_LEARNED_REFERENT_LEXICON` (default OFF) passed its own 6-seed route test
(research/findings/2026-09-24-language-learned-referent-production-route-GO-6seed.md), but the B2b pre-battery review
measured, with the seed-42 production lexicon, that switching it on changes the D6 referent parse
(`d6_multiref_wm_production_organ.extract_referents`) on 60 of the 112 battery probe turns: question and closed-class
words ('what', 'who', 'before', 'most', 'today') are admitted as referents, and on the false-belief turn `tom_fb` the
name 'anne' falls out of the referent cap. No artifact was written. This runner is that measurement as a committed,
re-runnable instrument, and the scoring rule a fix must meet.

WHAT IT MEASURES (per lexicon-training seed). For every turn in `onebrain_regression_battery._TURN_BY_LABEL` (the 112
probe turns the load-bearing battery can run, including the row-module EXTRA_TURNS such as `tom_fb`):
  off = extract_referents(text)                       # BRAIN_LEARNED_REFERENT_LEXICON unset: the hand-table path
  on  = extract_referents(text, referent_lexicon=lex) # the learned lexicon consulted for words the hand table lacks
and the per-word decision the lexicon returned for every word it was asked about.

THE ADJUDICATION RULE (`adjudicate`, pure, unit-tested). A turn MATCHES iff
  (a) every word the flag ADDS (in `on`, not in `off`) is a genuinely new noun by the independent ground truth
      (`is_gt_noun`: spaCy dominant POS == NOUN in research/fixtures/lexicon_referent_pos_gt.json, the TinyStories
      fixture the referent GO already uses; for a word it lacks, research/findings/raw/_corpus_pos_map.json; a word in
      neither map -- closed-class words, proper names -- is NOT a genuine noun), AND
  (b) every word the flag DROPS (in `off`, not in `on`) was displaced by the referent cap alone: `on` is full
      (len == cap) and (a) holds. The lexicon path only ADDS candidates, so a drop can only come from the cap; the rule
      exempts it only when every word that took the capacity is a genuine noun (WM capacity is a real limit; a
      closed-class word occupying it is not), AND
  (c) the words present in both lists keep their relative order (order-of-mention is the role marker).
Everything else is a MISMATCH. The ground truth is type-level (dominant POS of the word form), so a noun form used as
a verb in context ('Sally leaves the room') is credited as a noun: noun-hood, not referent-hood, the lexicon's own
declared residual.

Usage (numpy backend; the corpus is untracked -- pass the full 19,971,040-byte file explicitly, its sha256 is recorded):
  bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy python -u -m \
      research.runners._lexicon_closed_class_parse_diag --seed 7 --corpus /path/to/tinystories.txt \
      --json research/findings/raw/_lexicon_closed_class/diag_frame_s7.json
  python -m research.runners._lexicon_closed_class_parse_diag --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FIXTURE = os.path.join(_REPO, "research", "fixtures", "lexicon_referent_pos_gt.json")
CORPUS_POS_MAP = os.path.join(_REPO, "research", "findings", "raw", "_corpus_pos_map.json")
_FLAG_ENV = ("BRAIN_LEARNED_REFERENT_LEXICON", "BRAIN_LEARNED_REFERENT_LESION")


# ── the independent ground truth ───────────────────────────────────────────────────────────────────────────────
def load_gt_noun():
    """is_gt_noun(word) from the two committed spaCy dominant-POS maps (TinyStories fixture first)."""
    fx = json.load(open(FIXTURE))["pos"]
    cm = json.load(open(CORPUS_POS_MAP))

    def is_gt_noun(w: str) -> bool:
        p = fx.get(w)
        if p is not None:
            return p == "NOUN"
        e = cm.get(w)
        return bool(e) and e.get("pos") == "NOUN"

    def gt_pos(w: str):
        p = fx.get(w)
        if p is not None:
            return p
        e = cm.get(w)
        return e.get("pos") if e else None

    return is_gt_noun, gt_pos


# ── the adjudication rule (pure) ─────────────────────────────────────────────────────────────────────────────
def adjudicate(off, on, cap, is_noun):
    """See the module docstring, rule (a)-(c). Returns a dict; `match` is the verdict for this turn."""
    off, on = list(off), list(on)
    admitted = [w for w in on if w not in off]
    dropped = [w for w in off if w not in on]
    bad_admits = [w for w in admitted if not is_noun(w)]
    cap_full = len(on) >= cap
    unexplained_drops = [] if (not dropped or (cap_full and not bad_admits)) else list(dropped)
    order_ok = [w for w in on if w in off] == [w for w in off if w in on]
    return {"changed": on != off, "admitted": admitted, "dropped": dropped, "bad_admits": bad_admits,
            "unexplained_drops": unexplained_drops, "cap_full": cap_full, "order_ok": order_ok,
            "match": (not bad_admits) and (not unexplained_drops) and order_ok}


def _sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest(), os.path.getsize(path)
    except OSError:
        return None, None


def _clean_env():
    for k in _FLAG_ENV:
        os.environ.pop(k, None)


def run(seed: int, corpus: str, lesion=None):
    """Build the deployment lexicon at `seed` on `corpus` (the variant `get_lexicon()` selects from the process env),
    then parse every battery probe turn with and without it."""
    from research.runners import lexicon_spiking_frame_category as L
    from research.runners import d6_multiref_wm_production_organ as D6
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL

    t0 = time.time()
    _clean_env()                      # the OFF arm is the production default: the flag must be unset in-process
    L._LEXICON = None
    lex = L.get_lexicon(seed=seed, corpus_path=corpus)
    variant = getattr(lex, "variant", "frame")
    if lesion:
        lex.set_lesion(lesion)
    is_noun, gt_pos = load_gt_noun()
    cap = min(D6.R_MAX, D6._BINDER_K)

    turns, queried = [], {}
    for label, t in _TURN_BY_LABEL.items():
        text, session = t[1], t[2]
        off = D6.extract_referents(text)
        on = D6.extract_referents(text, referent_lexicon=lex)
        adj = adjudicate(off, on, cap, is_noun)
        turns.append({"label": label, "session": session, "text": text, "off": off, "on": on, **adj})
        for w in D6._WORD_RE.findall(text or ""):
            lw = w.lower()
            if (lw in D6._REFERENT_NOUNS or lw in D6._STOP or lw in D6._PRONOUNS or lw in D6._HOLD_QUERY_WORDS
                    or lw in queried):
                continue
            dec, rn, rx = lex.decide(lw)
            queried[lw] = {"decision": dec[0], "rate_cn": None if rn is None else float(rn[0]),
                           "rate_cx": None if rx is None else float(rx[0]),
                           "heard": int(len(lex.env.pos.get(lw, ()))), "gt_pos": gt_pos(lw),
                           "gt_noun": is_noun(lw)}
    _clean_env()

    changed = [r for r in turns if r["changed"]]
    mism = [r for r in turns if not r["match"]]
    offending = sorted({w for r in turns for w in r["bad_admits"]})
    new_nouns = sorted({w for r in turns for w in r["admitted"] if is_noun(w)})
    parse_blob = json.dumps([(r["label"], r["off"], r["on"]) for r in turns], sort_keys=True).encode()
    out = {
        "seed": seed, "variant": variant, "lesion": lesion, "cap": cap,
        "n_turns": len(turns), "n_changed": len(changed), "n_mismatch": len(mism),
        "mismatch_labels": [r["label"] for r in mism],
        "offending_words": offending,
        "dropped_words": sorted({w for r in turns for w in r["dropped"]}),
        "unexplained_drops": sorted({w for r in turns for w in r["unexplained_drops"]}),
        "new_gt_nouns_recovered": new_nouns,
        "sessions_changed": sorted({r["session"] for r in changed}),
        "sessions_mismatch": sorted({r["session"] for r in mism}),
        "admitted_by_decision": sorted(w for w, q in queried.items() if q["decision"] is True),
        "parse_sha256": hashlib.sha256(parse_blob).hexdigest(),
        "decisions_sha256": hashlib.sha256(json.dumps(queried, sort_keys=True).encode()).hexdigest(),
        "turns": turns, "queried": queried,
    }
    out["corpus_path"] = corpus
    out["corpus_sha256"], out["corpus_bytes"] = _sha256(corpus)
    out["fixture_sha256"], _ = _sha256(FIXTURE)
    out["corpus_pos_map_sha256"], _ = _sha256(CORPUS_POS_MAP)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _selftest():
    nouns = {"owl", "apple", "marble", "basket", "room", "box", "leaves"}
    isn = nouns.__contains__
    # a genuinely new noun, no drop -> match
    assert adjudicate(["wolf"], ["wolf", "owl"], 5, isn)["match"]
    # a closed-class admission -> mismatch
    r = adjudicate(["dog", "bird"], ["dog", "when", "bird"], 5, isn)
    assert not r["match"] and r["bad_admits"] == ["when"]
    # cap displacement by genuine nouns only -> explained -> match
    r = adjudicate(["sally", "anne", "box"], ["marble", "basket", "sally", "leaves", "room"], 5, isn)
    assert r["match"] and r["dropped"] == ["anne", "box"] and not r["unexplained_drops"]
    # cap displacement with a closed-class word in the capacity -> both counted
    r = adjudicate(["sally", "anne"], ["marble", "sally", "when", "room", "basket"], 5, isn)
    assert not r["match"] and r["bad_admits"] == ["when"] and r["unexplained_drops"] == ["anne"]
    # a drop with the cap NOT full cannot be explained by capacity
    r = adjudicate(["sally", "anne"], ["sally", "owl"], 5, isn)
    assert not r["match"] and r["unexplained_drops"] == ["anne"]
    # order change of shared words -> mismatch
    assert not adjudicate(["a1", "b1"], ["b1", "a1"], 5, isn)["match"]
    # identical -> match, unchanged
    r = adjudicate(["dog"], ["dog"], 5, isn)
    assert r["match"] and not r["changed"]
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--lesion", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    r = run(a.seed, corpus, lesion=a.lesion or None)
    print(f"[seed {r['seed']} variant={r['variant']} lesion={r['lesion']}] changed {r['n_changed']}/{r['n_turns']} "
          f"mismatch {r['n_mismatch']} offending={r['offending_words']} unexplained_drops={r['unexplained_drops']} "
          f"new_nouns={len(r['new_gt_nouns_recovered'])} ({r['elapsed_s']}s)", flush=True)
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(r, open(dst, "w"), indent=1, default=str)
        print("wrote", dst)


if __name__ == "__main__":
    main()
