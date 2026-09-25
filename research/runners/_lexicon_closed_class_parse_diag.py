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

THE GROUND TRUTH (`load_gt`, three-valued, type-level; an instrument, never read by any mechanism):
  NON     the word is in a standard closed-class inventory (research/fixtures/closed_class_inventory_nltk_english.json,
          the NLTK English stopword list, 198 words, copied verbatim), OR its spaCy dominant POS is VERB/ADJ;
  NOUN    otherwise, its spaCy dominant POS is NOUN (research/fixtures/lexicon_referent_pos_gt.json, the TinyStories
          fixture the referent GO already uses; for a word it lacks, research/findings/raw/_corpus_pos_map.json);
  UNKNOWN in no map and not in the inventory (proper names, rare words, 'east'). The POS maps cover only open-class
          words, so absence from them is not evidence of being closed-class.
  REVISION (before any mechanism run): the first committed version (ebabbe1a6) was two-valued, "not NOUN in a map
  => not a noun". Its seed-7 baseline counted 'east' ("the sun rises in the east", a noun use) as a non-noun only
  because the maps omit it. The third value and the inventory replaced that; nothing else changed.

THE ADJUDICATION RULE (`adjudicate`, pure, unit-tested). A turn MATCHES iff
  (a) no word the flag ADDS (in `on`, not in `off`) is NON (an added UNKNOWN word is listed in `unknown_admits`,
      not counted), AND
  (b) every word the flag DROPS (in `off`, not in `on`) was displaced by the referent cap alone: `on` is full
      (len == cap) and (a) holds. The lexicon path only ADDS candidates, so a drop can only come from the cap; the rule
      exempts it only when no NON word took the capacity (WM capacity is a real limit; a closed-class word occupying
      it is not), AND
  (c) the words present in both lists keep their relative order (order-of-mention is the role marker).
Everything else is a MISMATCH. The ground truth is type-level (dominant POS of the word form), so a noun form used as
a verb in context ('Sally leaves the room') is credited as a noun: noun-hood, not referent-hood, the lexicon's own
declared residual.

Usage (numpy backend; the corpus is untracked -- pass the full 19,971,040-byte file explicitly, its sha256 is recorded):
  bash tools/mem_ok.sh 2 && bash tools/memcap.sh 2 -- env SIM_BACKEND=numpy python -u -m \
      research.runners._lexicon_closed_class_parse_diag --seed 7 --corpus /path/to/tinystories.txt \
      --json research/findings/raw/_lexicon_closed_class/diag_frame_s7_gt3.json
  # the evaluation form (junction variant, intact + coincidence-lesion arms, one lexicon build per seed):
  BRAIN_LEARNED_REFERENT_JUNCTION=1 ... --seed <S> --lesions none,coincidence --json <dir>/junction_s<S>.json
  python -m research.runners._lexicon_closed_class_parse_diag --score <dir> [--route <route runner out dir>]
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
CLOSED_CLASS = os.path.join(_REPO, "research", "fixtures", "closed_class_inventory_nltk_english.json")
_FLAG_ENV = ("BRAIN_LEARNED_REFERENT_LEXICON", "BRAIN_LEARNED_REFERENT_LESION")


# ── the independent ground truth ───────────────────────────────────────────────────────────────────────────────
NOUN, NON, UNKNOWN = "NOUN", "NON", "UNKNOWN"


def load_gt():
    """(gt_class, gt_pos): gt_class(word) -> NOUN / NON / UNKNOWN (see the module docstring)."""
    fx = json.load(open(FIXTURE))["pos"]
    cm = json.load(open(CORPUS_POS_MAP))
    closed = set(json.load(open(CLOSED_CLASS))["words"])

    def gt_pos(w: str):
        p = fx.get(w)
        if p is not None:
            return p
        e = cm.get(w)
        return e.get("pos") if e else None

    def gt_class(w: str) -> str:
        if w in closed:
            return NON
        p = gt_pos(w)
        if p == "NOUN":
            return NOUN
        if p in ("VERB", "ADJ"):
            return NON
        return UNKNOWN

    return gt_class, gt_pos


# ── the adjudication rule (pure) ─────────────────────────────────────────────────────────────────────────────
def adjudicate(off, on, cap, gt_class):
    """See the module docstring, rule (a)-(c). `gt_class(word)` -> NOUN / NON / UNKNOWN. Returns a dict; `match` is
    the verdict for this turn."""
    off, on = list(off), list(on)
    admitted = [w for w in on if w not in off]
    dropped = [w for w in off if w not in on]
    bad_admits = [w for w in admitted if gt_class(w) == NON]
    unknown_admits = [w for w in admitted if gt_class(w) == UNKNOWN]
    cap_full = len(on) >= cap
    unexplained_drops = [] if (not dropped or (cap_full and not bad_admits)) else list(dropped)
    order_ok = [w for w in on if w in off] == [w for w in off if w in on]
    return {"changed": on != off, "admitted": admitted, "dropped": dropped, "bad_admits": bad_admits,
            "unknown_admits": unknown_admits, "unexplained_drops": unexplained_drops, "cap_full": cap_full,
            "order_ok": order_ok, "match": (not bad_admits) and (not unexplained_drops) and order_ok}


def _sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest(), os.path.getsize(path)
    except OSError:
        return None, None


def _clean_env():
    for k in _FLAG_ENV:
        os.environ.pop(k, None)


def _parse_arm(lex, arm, gt_class, gt_pos):
    """Parse every battery probe turn with and without `lex` (its current lesion state) and adjudicate."""
    from research.runners import d6_multiref_wm_production_organ as D6
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL

    cap = min(D6.R_MAX, D6._BINDER_K)
    turns, queried = [], {}
    for label, t in _TURN_BY_LABEL.items():
        text, session = t[1], t[2]
        off = D6.extract_referents(text)
        on = D6.extract_referents(text, referent_lexicon=lex)
        adj = adjudicate(off, on, cap, gt_class)
        turns.append({"label": label, "session": session, "text": text, "off": off, "on": on, **adj})
        for w in D6._WORD_RE.findall(text or ""):
            lw = w.lower()
            if (lw in D6._REFERENT_NOUNS or lw in D6._STOP or lw in D6._PRONOUNS or lw in D6._HOLD_QUERY_WORDS
                    or lw in queried):
                continue
            dec, rn, rx = lex.decide(lw)
            queried[lw] = {"decision": dec[0], "rate_cn": None if rn is None else float(rn[0]),
                           "rate_cx": None if rx is None else float(rx[0]),
                           "heard": int(len(lex.env.pos.get(lw, ())))}
    changed = [r for r in turns if r["changed"]]
    mism = [r for r in turns if not r["match"]]
    parse_blob = json.dumps([(r["label"], r["off"], r["on"]) for r in turns], sort_keys=True).encode()
    # the hash covers the lexicon's own outputs only (not the ground-truth labels), so it can pin byte-identity
    dec_blob = json.dumps(queried, sort_keys=True).encode()
    for w, q in queried.items():
        q["gt_pos"], q["gt_class"] = gt_pos(w), gt_class(w)
    tom = next((r for r in turns if r["label"] == "tom_fb"), None)
    return {
        "arm": arm, "cap": cap,
        "n_turns": len(turns), "n_changed": len(changed), "n_mismatch": len(mism),
        "mismatch_labels": [r["label"] for r in mism],
        "offending_words": sorted({w for r in turns for w in r["bad_admits"]}),
        "dropped_words": sorted({w for r in turns for w in r["dropped"]}),
        "unexplained_drops": sorted({w for r in turns for w in r["unexplained_drops"]}),
        "unknown_admits": sorted({w for r in turns for w in r["unknown_admits"]}),
        "new_gt_nouns_recovered": sorted({w for r in turns for w in r["admitted"] if gt_class(w) == NOUN}),
        "sessions_changed": sorted({r["session"] for r in changed}),
        "sessions_mismatch": sorted({r["session"] for r in mism}),
        "admitted_by_decision": sorted(w for w, q in queried.items() if q["decision"] is True),
        "tom_fb_on": None if tom is None else tom["on"],
        "tom_fb_anne_kept": None if tom is None else ("anne" in tom["on"]),
        "parse_sha256": hashlib.sha256(parse_blob).hexdigest(),
        "decisions_sha256": hashlib.sha256(dec_blob).hexdigest(),
        "turns": turns, "queried": queried,
    }


def run(seed: int, corpus: str, lesions=(None,)):
    """Build the deployment lexicon ONCE at `seed` on `corpus` (the variant `get_lexicon()` selects from the process
    env), then parse the battery turns once per arm: None = intact, else a lesion kind applied (and verified by the
    lexicon's own weight-hash check in `decide()`) for that arm only."""
    from research.runners import lexicon_spiking_frame_category as L

    t0 = time.time()
    _clean_env()                      # the OFF arm is the production default: the flag must be unset in-process
    L._LEXICON = None
    lex = L.get_lexicon(seed=seed, corpus_path=corpus)
    t_build = round(time.time() - t0, 1)
    gt_class, gt_pos = load_gt()
    arms = {}
    for lesion in lesions:
        name = lesion or "intact"
        lex.set_lesion(lesion)
        arms[name] = _parse_arm(lex, name, gt_class, gt_pos)
    lex.set_lesion(None)
    _clean_env()
    out = {"seed": seed, "variant": getattr(lex, "variant", "frame"),
           "junction_flag": os.environ.get("BRAIN_LEARNED_REFERENT_JUNCTION"),
           "arms": arms, "build_train_s": t_build}
    # headline copy of the intact arm (the first arm) for readability
    first = arms[next(iter(arms))]
    for k in ("n_turns", "n_changed", "n_mismatch", "offending_words", "unknown_admits", "parse_sha256",
              "decisions_sha256"):
        out[k] = first[k]
    out["corpus_path"] = corpus
    out["corpus_sha256"], out["corpus_bytes"] = _sha256(corpus)
    out["fixture_sha256"], _ = _sha256(FIXTURE)
    out["corpus_pos_map_sha256"], _ = _sha256(CORPUS_POS_MAP)
    out["closed_class_sha256"], _ = _sha256(CLOSED_CLASS)
    try:
        import resource
        out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # noqa: BLE001
        out["peak_rss_mb"] = None
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


EVAL_SEEDS = [42, 43, 44, 100, 101, 102]
PRODUCTION_SEED = 42


def score(src, route_dir=None):
    """The pre-registered G2/G3 verdict (research/findings/2026-09-24-lexicon-closed-class-frame-junction-
    PREREGISTRATION.md), plus G1 read from the route runner's own verdict.json in `route_dir` when given.
      G2: intact n_mismatch == 0 at seed 42 AND at >= 5 of the 6 seeds.
      G3: on >= 5 of 6 seeds the `coincidence` arm's n_mismatch exceeds the intact arm's (the lever moves).
    INCOMPLETE if a seed or an arm is missing or a run is not the junction variant; MIXED-INPUT if the corpus or
    ground-truth hashes differ between seeds (or between these runs and the route runs)."""
    import glob
    from tools.lab import lever
    per = {}
    for p in sorted(glob.glob(os.path.join(src, "*.json"))):
        if p.endswith(".prov.json") or os.path.basename(p) == "verdict.json":
            continue
        d = json.load(open(p))
        if "arms" in d:
            per[int(d["seed"])] = d
    M = {s: per[s] for s in EVAL_SEEDS if s in per}
    complete = (len(M) == len(EVAL_SEEDS)
                and all(d.get("variant") == "junction" and {"intact", "coincidence"} <= set(d["arms"])
                        for d in M.values()))
    rows = {}
    for s, d in M.items():
        a_i, a_l = d["arms"].get("intact"), d["arms"].get("coincidence")
        g2 = a_i is not None and a_i["n_mismatch"] == 0
        moved = False
        if a_i is not None and a_l is not None:
            moved = bool(lever(f"seed {s}: coincidence lesion -> parse mismatches", a_i["n_mismatch"],
                               a_l["n_mismatch"], required=False)) and a_l["n_mismatch"] > a_i["n_mismatch"]
        rows[s] = {"g2_pass": g2, "g3_lever_moved": moved,
                   "intact_mismatch": None if a_i is None else a_i["n_mismatch"],
                   "lesion_mismatch": None if a_l is None else a_l["n_mismatch"],
                   "offending": None if a_i is None else a_i["offending_words"],
                   "unknown_admits": None if a_i is None else a_i["unknown_admits"],
                   "new_nouns": None if a_i is None else len(a_i["new_gt_nouns_recovered"]),
                   "tom_fb_anne_kept": None if a_i is None else a_i["tom_fb_anne_kept"]}
    n_g2 = sum(r["g2_pass"] for r in rows.values())
    n_g3 = sum(r["g3_lever_moved"] for r in rows.values())
    g2 = bool(rows.get(PRODUCTION_SEED, {}).get("g2_pass")) and n_g2 >= 5
    g3 = n_g3 >= 5
    inputs = sorted({(d.get("corpus_sha256"), d.get("fixture_sha256"), d.get("corpus_pos_map_sha256"),
                      d.get("closed_class_sha256")) for d in M.values()}, key=str)
    one_input = len(inputs) == 1 and None not in inputs[0]
    g1 = None
    if route_dir:
        vpath = os.path.join(route_dir, "verdict.json")
        if os.path.exists(vpath):
            rv = json.load(open(vpath))
            g1 = rv.get("verdict") == "GO"
            route_corpus = {tuple(i)[1] for i in rv.get("inputs", [])}
            if one_input and route_corpus != {inputs[0][0]}:
                one_input = False
    verdict = ("INCOMPLETE" if (not complete or (route_dir is not None and g1 is None))
               else "MIXED-INPUT" if not one_input
               else ("GO" if (g2 and g3 and (g1 if route_dir else False)) else
                     ("G2+G3 PASS, G1 NOT SCORED" if (g2 and g3 and not route_dir) else "NO-GO")))
    return {"verdict": verdict, "complete_6seed": complete, "one_input": one_input,
            "G1_route_go": g1, "G2_parse_match": (n_g2, g2), "G3_conjunction_lesion": (n_g3, g3),
            "per_seed": rows, "inputs": [list(i) for i in inputs]}


def _selftest():
    nouns = {"owl", "apple", "marble", "basket", "room", "box", "leaves"}
    closed = {"when", "most", "the", "where"}
    gc = lambda w: NOUN if w in nouns else (NON if w in closed else UNKNOWN)  # noqa: E731
    assert adjudicate(["wolf"], ["wolf", "owl"], 5, gc)["match"]
    r = adjudicate(["dog", "bird"], ["dog", "when", "bird"], 5, gc)
    assert not r["match"] and r["bad_admits"] == ["when"]
    r = adjudicate(["dog"], ["dog", "east"], 5, gc)
    assert r["match"] and r["unknown_admits"] == ["east"]
    r = adjudicate(["sally", "anne", "box"], ["marble", "basket", "sally", "leaves", "room"], 5, gc)
    assert r["match"] and r["dropped"] == ["anne", "box"] and not r["unexplained_drops"]
    r = adjudicate(["sally", "anne"], ["marble", "sally", "when", "room", "basket"], 5, gc)
    assert not r["match"] and r["bad_admits"] == ["when"] and r["unexplained_drops"] == ["anne"]
    r = adjudicate(["sally", "anne"], ["sally", "owl"], 5, gc)
    assert not r["match"] and r["unexplained_drops"] == ["anne"]
    assert not adjudicate(["a1", "b1"], ["b1", "a1"], 5, gc)["match"]
    r = adjudicate(["dog"], ["dog"], 5, gc)
    assert r["match"] and not r["changed"]
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--lesions", default="none", help="comma list of arms: none (intact) and/or lesion kinds")
    ap.add_argument("--json", default="")
    ap.add_argument("--score", default="")
    ap.add_argument("--route", default="", help="with --score: the route runner's output dir (its verdict.json)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if a.score:
        out = score(a.score, a.route or None)
        print(json.dumps(out, indent=1, default=str))
        json.dump(out, open(os.path.join(a.score, "verdict.json"), "w"), indent=1, default=str)
        return
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    lesions = tuple(None if x.strip() in ("none", "") else x.strip() for x in a.lesions.split(","))
    r = run(a.seed, corpus, lesions=lesions)
    for name, arm in r["arms"].items():
        print(f"[seed {r['seed']} variant={r['variant']} arm={name}] changed {arm['n_changed']}/{arm['n_turns']} "
              f"mismatch {arm['n_mismatch']} offending={arm['offending_words']} unknown={arm['unknown_admits']} "
              f"unexplained_drops={arm['unexplained_drops']} new_nouns={len(arm['new_gt_nouns_recovered'])} "
              f"anne_kept={arm['tom_fb_anne_kept']}", flush=True)
    print(f"build+train {r['build_train_s']}s, total {r['elapsed_s']}s, peak RSS {r['peak_rss_mb']} MB", flush=True)
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(r, open(dst, "w"), indent=1, default=str)
        print("wrote", dst)


if __name__ == "__main__":
    main()
