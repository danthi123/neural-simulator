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
  # AMENDMENT 3 (elemental partial-match edge; arms intact + the two drive-REMOVING G3' lesions):
  BRAIN_LEARNED_REFERENT_JUNCTION=1 BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL=1 ... --seed <S> \
      --lesions none,elemental,conjunctive --json <dir>/junction_elemental_s<S>.json
  (score() detects variant `junction_elemental` and scores G2, G3', G4, and G1 from --route.) Every queried word
  also records `drive` (the afferent drive into each pool per edge; the junction lexicon's `drive_of`), and every
  arm its `mean_afferent_drive` -- G3''s drive condition.
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
TOKEN_FIXTURE = os.path.join(_REPO, "research", "fixtures", "lexicon_referent_pos_gt_tokenlevel.json")
_FLAG_ENV = ("BRAIN_LEARNED_REFERENT_LEXICON", "BRAIN_LEARNED_REFERENT_LESION")

# AMENDMENT 2 (review issue G2-3): a small, explicit override for words the NLTK stopword list and the dominant-POS
# maps both miss AND the Penn Treebank convention itself tags NN (see
# `_lexicon_closed_class_token_pos_fixture.py`'s `_INDEFINITE_PRONOUN_OVERRIDE` docstring). Applied to BOTH the
# type-level and token-level classifiers below so the two instruments agree on this word class.
INDEFINITE_PRONOUNS = {
    "something", "someone", "somebody", "anything", "anyone", "anybody",
    "everything", "everyone", "everybody", "nothing", "noone", "nobody",
}


# ── the independent ground truth ───────────────────────────────────────────────────────────────────────────────
NOUN, NON, UNKNOWN = "NOUN", "NON", "UNKNOWN"


def load_gt():
    """(gt_class, gt_pos): gt_class(word) -> NOUN / NON / UNKNOWN, TYPE-level (one dominant-POS reading per word
    form, see the module docstring). Used for the cross-turn `queried` report (AMENDMENT 2 keeps this scope) and as
    the per-turn adjudication FALLBACK for any word the token-level fixture does not cover."""
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
        if w in closed or w in INDEFINITE_PRONOUNS:
            return NON
        p = gt_pos(w)
        if p == "NOUN":
            return NOUN
        if p in ("VERB", "ADJ"):
            return NON
        return UNKNOWN

    return gt_class, gt_pos


def load_token_gt():
    """label -> {word_lower: NOUN/NON/UNKNOWN}, TOKEN-level (AMENDMENT 2, review issue G2-3): each battery turn's
    OWN words tagged IN THAT SENTENCE by `_lexicon_closed_class_token_pos_fixture.py` (nltk averaged-perceptron,
    Penn Treebank tagset, built once and committed -- no runtime nltk dependency; see that module's docstring).
    Fixes 'leaves' (VBZ in "Sally leaves the room", not the type-level dominant NOUN), 'today' (whatever its
    CONTEXTUAL tag is, not the fixture's blanket NOUN), and every modal/wh-word/pronoun/adverb the NLTK-198
    stopword list and the NOUN/VERB/ADJ-only POS maps miss (falls to UNKNOWN there; resolves directly here).
    HONEST RESIDUAL: the tagger is itself imperfect (verified: 'east' tags RB, an adverb reading, in "the sun
    rises in the east" -- a noun use the tagger gets wrong). This is a second, independently-imperfect instrument,
    not a superseding one."""
    d = json.load(open(TOKEN_FIXTURE))
    return {label: pw for label, pw in d["turns"].items()}


def make_turn_gt_class(label: str, token_gt: dict, gt_class_fallback):
    """A gt_class(word) closure SCOPED TO ONE TURN: the token-level tag for that turn's own occurrence of `word`
    when available, else the type-level fallback (AMENDMENT 2)."""
    per_word = token_gt.get(label) or {}

    def gt_class_turn(w: str) -> str:
        info = per_word.get(w)
        if info is not None:
            return info["class"]
        return gt_class_fallback(w)

    return gt_class_turn


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


def _parse_arm(lex, arm, gt_class, gt_pos, token_gt=None):
    """Parse every battery probe turn with and without `lex` (its current lesion state) and adjudicate.

    AMENDMENT 2 (review issues G2-2, G2-3): adjudication uses the TOKEN-level ground truth for each turn's own
    words when `token_gt` is given (falls back to the type-level `gt_class`), and every queried word carries its
    CN-CX rate margin plus a NON-word silence flag (silent = both pools below `L.MIN_RATE`, a failure to decide,
    not a margin abstain)."""
    from research.runners import d6_multiref_wm_production_organ as D6
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL
    from research.runners import lexicon_spiking_frame_category as L

    cap = min(D6.R_MAX, D6._BINDER_K)
    turns, queried = [], {}
    for label, t in _TURN_BY_LABEL.items():
        text, session = t[1], t[2]
        off = D6.extract_referents(text)
        on = D6.extract_referents(text, referent_lexicon=lex)
        gt_class_turn = make_turn_gt_class(label, token_gt, gt_class) if token_gt is not None else gt_class
        adj = adjudicate(off, on, cap, gt_class_turn)
        turns.append({"label": label, "session": session, "text": text, "off": off, "on": on, **adj})
        for w in D6._WORD_RE.findall(text or ""):
            lw = w.lower()
            if (lw in D6._REFERENT_NOUNS or lw in D6._STOP or lw in D6._PRONOUNS or lw in D6._HOLD_QUERY_WORDS
                    or lw in queried):
                continue
            dec, rn, rx = lex.decide(lw)
            rn0 = None if rn is None else float(rn[0])
            rx0 = None if rx is None else float(rx[0])
            queried[lw] = {"decision": dec[0], "rate_cn": rn0, "rate_cx": rx0,
                           "margin": None if (rn0 is None or rx0 is None) else rn0 - rx0,
                           "heard": int(len(lex.env.pos.get(lw, ())))}
            # AMENDMENT 3 instrument (junction lexicon only; v2 has no drive_of): the afferent drive into each pool.
            # Not part of the decisions hash below (that pins the 4 original fields only).
            drive_of = getattr(lex, "drive_of", None)
            queried[lw]["drive"] = drive_of(lw) if drive_of is not None else None
    changed = [r for r in turns if r["changed"]]
    mism = [r for r in turns if not r["match"]]
    parse_blob = json.dumps([(r["label"], r["off"], r["on"]) for r in turns], sort_keys=True).encode()
    # the hash covers the lexicon's own outputs only (not the ground-truth labels), so it can pin byte-identity.
    # AMENDMENT 2: hashed over the ORIGINAL 4 fields only (decision/rate_cn/rate_cx/heard) -- "margin" added below
    # is a pure re-derivation of rate_cn-rate_cx, not new decision information, and must not perturb this pin.
    dec_blob = json.dumps({w: {k: q[k] for k in ("decision", "rate_cn", "rate_cx", "heard")}
                           for w, q in queried.items()}, sort_keys=True).encode()
    for w, q in queried.items():
        q["gt_pos"], q["gt_class"] = gt_pos(w), gt_class(w)
        q["silent"] = bool(q["rate_cn"] is not None and q["rate_cx"] is not None
                           and q["rate_cn"] < L.MIN_RATE and q["rate_cx"] < L.MIN_RATE)
    tom = next((r for r in turns if r["label"] == "tom_fb"), None)
    # AMENDMENT 2 G3 (review issue G3-1): per-word margins for every ADMITTED word (not just bad_admits), and a
    # near-boundary flag (margin within 1.5x DEAD_MARGIN of the abstain threshold) -- separates "the conjunction
    # decided this word cleanly" from "this word rode in on general drive, barely clearing the margin".
    admitted_margins = []
    for r in turns:
        for w in r["admitted"]:
            q = queried.get(w)
            if q is None or q["margin"] is None:
                continue
            admitted_margins.append({"label": r["label"], "word": w, "rate_cn": q["rate_cn"], "rate_cx": q["rate_cx"],
                                     "margin": q["margin"], "gt_class": q["gt_class"],
                                     "near_boundary": bool(abs(q["margin"]) < 1.5 * L.DEAD_MARGIN)})
    # AMENDMENT 2 G2-2 (review issue G2-2, no-pass-by-abstaining): NON-ground-truth words the lexicon was ASKED
    # ABOUT that leave BOTH pools silent -- a failure to decide, distinct from a margin abstain (both pools active
    # but too close to call). Reported per arm; `score()` compares this seed's junction reading against v2's own
    # reading at the SAME seed (never a cross-seed or absolute bar chosen after the fact).
    non_queried = {w: q for w, q in queried.items() if q["gt_class"] == NON and q["heard"] > 0}
    silent_non = sorted(w for w, q in non_queried.items() if q["silent"])
    # AMENDMENT 3 (G3' drive condition): the arm mean of each afferent-drive component over the queried words that
    # were presented (None for a lexicon without the instrument, e.g. v2).
    drives = [q["drive"] for q in queried.values() if q.get("drive")]
    mean_drive = ({k: float(sum(d[k] for d in drives) / len(drives)) for k in drives[0]} if drives else None)
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
        "admitted_margins": admitted_margins,
        "n_non_heard": len(non_queried), "silent_non_words": silent_non,
        "silent_non_fraction": (len(silent_non) / len(non_queried)) if non_queried else None,
        "mean_afferent_drive": mean_drive, "n_drive_words": len(drives),
        "tom_fb_on": None if tom is None else tom["on"],
        "tom_fb_anne_kept": None if tom is None else ("anne" in tom["on"]),
        "parse_sha256": hashlib.sha256(parse_blob).hexdigest(),
        "decisions_sha256": hashlib.sha256(dec_blob).hexdigest(),
        "turns": turns, "queried": queried,
    }


def _git_sha():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


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
    token_gt = load_token_gt()
    arms = {}
    for lesion in lesions:
        name = lesion or "intact"
        lex.set_lesion(lesion)
        arms[name] = _parse_arm(lex, name, gt_class, gt_pos, token_gt=token_gt)
    lex.set_lesion(None)
    _clean_env()
    out = {"seed": seed, "variant": getattr(lex, "variant", "frame"),
           "junction_flag": os.environ.get("BRAIN_LEARNED_REFERENT_JUNCTION"),
           "elemental_flag": os.environ.get("BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL"),
           "arms": arms, "build_train_s": t_build, "git_sha": _git_sha()}
    # AMENDMENT 2 (review issue #6): record the junction variant's own frozen constants on every run, so score()
    # can refuse to pool seeds that ran under different constants/code as MIXED-INPUT (below) instead of silently
    # averaging them.
    if out["variant"] == "junction":
        from research.runners import lexicon_frame_junction as J
        out["constants"] = {"W_J": J.W_J, "I_TONIC_J": J.I_TONIC_J, "T_ON_J": J.T_ON_J,
                            "DRIVE_MATCH_S": J.DRIVE_MATCH_S, "OR_LESION_FACTOR": J.OR_LESION_FACTOR,
                            "OR_MATCH_FACTOR": getattr(J, "OR_MATCH_FACTOR", None),
                            "stp_enabled": getattr(J, "STP_ENABLED", None)}
    elif out["variant"] == "junction_elemental":
        # AMENDMENT 3: a separate branch, so the round-2 junction artifacts' constants blob is unchanged.
        from research.runners import lexicon_frame_junction as J
        out["constants"] = {"W_J": J.W_J, "I_TONIC_J": J.I_TONIC_J, "T_ON_J": J.T_ON_J,
                            "DRIVE_MATCH_S": J.DRIVE_MATCH_S, "stp_enabled": J.STP_ENABLED, "elemental": True,
                            "W_INIT_E": J.W_INIT_E, "ETA_E": J.ETA_E, "OJA_BETA_E": J.OJA_BETA_E,
                            "ELEMENTAL_JITTER_SEED": J.ELEMENTAL_JITTER_SEED}
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
    out["token_fixture_sha256"], _ = _sha256(TOKEN_FIXTURE)
    try:
        import resource
        out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # noqa: BLE001
        out["peak_rss_mb"] = None
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


EVAL_SEEDS = [42, 43, 44, 100, 101, 102]
PRODUCTION_SEED = 42
G4_SILENT_NON_MAX = 0.30          # AMENDMENT 2's G4 bar, unchanged by AMENDMENT 3
ELEMENTAL_ARMS = ("intact", "elemental", "conjunctive")


def g3prime_seed(arms):
    """AMENDMENT 3's G3' at one seed, from the parse arms `intact`, `elemental`, `conjunctive` (pure; the dev script
    and score() share it). G3'a: the `conjunctive` lesion (no junction can fire) raises the battery mismatch count.
    G3'b: the `elemental` lesion raises the silent-NON fraction. Each lesion must also not RAISE the arm-mean total
    afferent drive (it removes an edge); if either drive condition fails the seed is VOID (a bug, not a result)."""
    from tools.lab import lever
    a_i, a_e, a_c = arms["intact"], arms["elemental"], arms["conjunctive"]

    def _total(a):
        d = a.get("mean_afferent_drive")
        return None if not d else d.get("total")
    t_i, t_e, t_c = _total(a_i), _total(a_e), _total(a_c)
    drive_ok_c = t_i is not None and t_c is not None and t_c <= t_i + 1e-9
    drive_ok_e = t_i is not None and t_e is not None and t_e <= t_i + 1e-9
    moved_a = bool(lever("G3'a: conjunctive lesion -> battery parse mismatches", a_i["n_mismatch"],
                         a_c["n_mismatch"], required=False)) and a_c["n_mismatch"] > a_i["n_mismatch"]
    s_i, s_e = a_i.get("silent_non_fraction"), a_e.get("silent_non_fraction")
    moved_b = (s_i is not None and s_e is not None
               and bool(lever("G3'b: elemental lesion -> silent-NON fraction", s_i, s_e, required=False))
               and s_e > s_i)
    void = not (drive_ok_c and drive_ok_e)
    return {"g3a_conjunctive_raises_mismatch": moved_a, "g3b_elemental_raises_silent_non": moved_b,
            "drive_total": {"intact": t_i, "elemental": t_e, "conjunctive": t_c},
            "drive_not_raised": {"conjunctive": drive_ok_c, "elemental": drive_ok_e},
            "void": void, "g3prime_pass": bool(moved_a and moved_b and not void),
            "mismatch": {"intact": a_i["n_mismatch"], "elemental": a_e["n_mismatch"],
                         "conjunctive": a_c["n_mismatch"]},
            "silent_non_fraction": {"intact": s_i, "elemental": s_e, "conjunctive": a_c.get("silent_non_fraction")}}


def _score_elemental(M, route_dir):
    """AMENDMENT 3 scoring of the `junction_elemental` variant: G2 (unchanged), G3' (replaces G3), G4 (AMENDMENT 2,
    every seed), G1 from the route runner's own verdict when `route_dir` is given."""
    complete = (len(M) == len(EVAL_SEEDS)
                and all(d.get("variant") == "junction_elemental" and set(ELEMENTAL_ARMS) <= set(d["arms"])
                        for d in M.values()))
    rows = {}
    for s, d in M.items():
        arms = d["arms"]
        a_i = arms.get("intact")
        row = {"g2_pass": a_i is not None and a_i["n_mismatch"] == 0,
               "intact_mismatch": None if a_i is None else a_i["n_mismatch"],
               "offending": None if a_i is None else a_i["offending_words"],
               "silent_non_fraction": None if a_i is None else a_i.get("silent_non_fraction"),
               "tom_fb_anne_kept": None if a_i is None else a_i["tom_fb_anne_kept"]}
        sf = row["silent_non_fraction"]
        row["g4_pass"] = sf is not None and sf <= G4_SILENT_NON_MAX
        row["g3prime"] = g3prime_seed(arms) if set(ELEMENTAL_ARMS) <= set(arms) else None
        row["g3prime_pass"] = bool(row["g3prime"] and row["g3prime"]["g3prime_pass"])
        rows[s] = row
    n_g2 = sum(r["g2_pass"] for r in rows.values())
    n_g3 = sum(r["g3prime_pass"] for r in rows.values())
    n_g4 = sum(r["g4_pass"] for r in rows.values())
    g2 = bool(rows.get(PRODUCTION_SEED, {}).get("g2_pass")) and n_g2 >= 5
    g3 = n_g3 >= 5
    g4 = n_g4 == len(EVAL_SEEDS)
    return complete, rows, (n_g2, g2), (n_g3, g3), (n_g4, g4)


def _score_with_elemental(M, variants, route_dir):
    """score() for AMENDMENT 3's `junction_elemental` runs: the same provenance / MIXED-INPUT / G1 logic as the
    junction path, G2 + G3' + G4 as pre-registered. A mix of variants across seeds is MIXED-INPUT."""
    complete, rows, g2t, g3t, g4t = _score_elemental(M, route_dir)
    inputs_required = sorted({(d.get("corpus_sha256"), d.get("fixture_sha256"), d.get("corpus_pos_map_sha256"),
                              d.get("closed_class_sha256")) for d in M.values()}, key=str)
    inputs = sorted({(d.get("corpus_sha256"), d.get("fixture_sha256"), d.get("corpus_pos_map_sha256"),
                      d.get("closed_class_sha256"), d.get("token_fixture_sha256"),
                      json.dumps(d.get("constants"), sort_keys=True) if d.get("constants") else None,
                      d.get("git_sha")) for d in M.values()}, key=str)
    one_input = (len(variants) == 1 and len(inputs_required) == 1 and None not in inputs_required[0]
                 and len(inputs) == 1)
    g1 = None
    if route_dir:
        vpath = os.path.join(route_dir, "verdict.json")
        if os.path.exists(vpath):
            rv = json.load(open(vpath))
            g1 = rv.get("verdict") == "GO"
            route_corpus = {tuple(i)[1] for i in rv.get("inputs", [])}
            if one_input and route_corpus != {inputs[0][0]}:
                one_input = False
    passed = g2t[1] and g3t[1] and g4t[1]
    verdict = ("INCOMPLETE" if (not complete or (route_dir is not None and g1 is None))
               else "MIXED-INPUT" if not one_input
               else ("GO" if (passed and (g1 if route_dir else False)) else
                     ("G2+G3'+G4 PASS, G1 NOT SCORED" if (passed and not route_dir) else "NO-GO")))
    return {"verdict": verdict, "variant": "junction_elemental", "complete_6seed": complete, "one_input": one_input,
            "G1_route_go": g1, "G2_parse_match": g2t, "G3prime_dissociation": g3t, "G4_silent_non": g4t,
            "per_seed": rows, "inputs": [list(i) for i in inputs]}


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
    variants = {d.get("variant") for d in M.values()}
    if "junction_elemental" in variants:
        return _score_with_elemental(M, variants, route_dir)
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
    # AMENDMENT 2 (review issue #6): REQUIRED provenance (corpus/fixture/POS-map/closed-class hashes) must always be
    # present and identical, as before. OPTIONAL provenance added here (token-fixture hash, junction constants, git
    # SHA) does not need to be POPULATED on every artifact (older runs predate these fields), but if it disagrees
    # across seeds -- different constants because an amendment landed mid-run, or a different git SHA -- that is
    # still MIXED-INPUT, never silently pooled as homogeneous.
    inputs_required = sorted({(d.get("corpus_sha256"), d.get("fixture_sha256"), d.get("corpus_pos_map_sha256"),
                              d.get("closed_class_sha256")) for d in M.values()}, key=str)
    inputs = sorted({(d.get("corpus_sha256"), d.get("fixture_sha256"), d.get("corpus_pos_map_sha256"),
                      d.get("closed_class_sha256"), d.get("token_fixture_sha256"),
                      json.dumps(d.get("constants"), sort_keys=True) if d.get("constants") else None,
                      d.get("git_sha")) for d in M.values()}, key=str)
    one_input = len(inputs_required) == 1 and None not in inputs_required[0] and len(inputs) == 1
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
