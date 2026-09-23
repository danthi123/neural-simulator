"""LEXICON de-risk — does the D6 multi-referent WM's REFERENT category EMERGE for OPEN vocabulary from real-corpus
distributional frames, spiking-realized, and is it LOAD-BEARING on the organ's reply? (language lane E, 2026-09-23)

WHAT IS TESTED: `research/runners/lexicon_learned_referent.py` (positional-frame PPMI graph + label-spreading from a
small seed set + a spiking two-pool WTA decision) replacing the D6 organ's hand 48-noun `_REFERENT_NOUNS` scope.
Ground truth = an INDEPENDENT tagger (spaCy dominant POS, `research/fixtures/lexicon_referent_pos_gt.json`, built by
`_lexicon_build_pos_gt_fixture.py`) over the learner's own vocabulary. HELD-OUT = every GT word NOT in any hand list
(the D6 table, HAND_NOUN_SEEDS, NONNOUN_SEEDS) — the learner is never given a label for any held-out word.
Cross-validated seeding: k=12 seeds per class drawn per-seed from the hand lists (rng = seed).

PRE-REGISTERED GO GATE (written BEFORE any 6-seed result; seeds 42 43 44 100 101 102):
  G1 learned (frame graph) held-out BALANCED accuracy, NOUN vs {VERB,ADJ}, abstain counted WRONG:
     mean >= 0.80 AND every seed >= 0.75
  G2 SHUFFLED-graph control (edge weights permuted, corpus structure destroyed): mean <= 0.60
  G3 learned - max(shuffled, frequency-only, seed-label-permuted) >= 0.15 (mean over seeds)
  G4 SPIKING decision agrees with the offline sign on >= 0.95 of held-out words, AND spiking balanced accuracy
     mean >= 0.80
  G5 LESION (zeroed spiking drive): held-out abstain rate == 1.0 on every seed
  G6 DETERMINISM: an independent rebuild at the same seed reproduces the offline scores + spiking decisions exactly
  G7 ORGAN (reply-level, the load-bearing test): MultiReferentWMOrgan on "the A and the B walked in" -> "who are we
     talking about?" with A,B held-out GT nouns absent from the hand table (the "owl" case):
       learned read-out recovers BOTH referents on >= 0.80 of trials (mean);
       hand-table baseline in-scope <= 0.10 and learned+LESION in-scope <= 0.10 (the difference vanishes);
       FALSE-POSITIVE control "we V1 and V2 all day" (V1,V2 held-out GT verbs/adjectives): learned in-scope <= 0.20
ATTRIBUTION (reported, not gated): the TOPICAL window-4 PPMI graph (the learned-animacy mechanism, no position) —
if positional frames carry noun-hood it should be below the frame graph.

Pool-friendly: self-contained (corpus path + committed fixture), numpy CPU, no production brain build.
    SIM_BACKEND=numpy OMP_NUM_THREADS=6 python -u -m research.runners._lexicon_learned_referent_derisk \
        --seeds 42 43 --corpus data/corpus/tinystories.txt \
        --json research/findings/raw/_lexicon_learned_referent/lexicon_referent_s42_43.json
    python -m research.runners._lexicon_learned_referent_derisk --score research/findings/raw/_lexicon_learned_referent
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._comprehension_learned_animacy_cue_derisk import load_tokens, build_vocab  # noqa: E402
from research.runners import lexicon_learned_referent as LR  # noqa: E402
from tools.lab import lever, undefined_if_empty, void_if  # noqa: E402

FIXTURE = os.path.join(_REPO, "research", "fixtures", "lexicon_referent_pos_gt.json")
SEEDS6 = [42, 43, 44, 100, 101, 102]
K_SEED = 12
GATE = dict(g1_mean=0.80, g1_min=0.75, g2_max=0.60, g3_gap=0.15, g4_agree=0.95, g4_bacc=0.80, g5_abstain=1.0,
            g7_recover=0.80, g7_base_max=0.10, g7_lesion_max=0.10, g7_fp_max=0.20)


def _bacc(pred: dict, gt: dict):
    """Balanced accuracy of NOUN(True) vs non-noun(False); abstain (None) counts wrong. None if a class is empty."""
    pos = [w for w in gt if gt[w]]
    neg = [w for w in gt if not gt[w]]
    if not pos or not neg:
        return None
    tpr = sum(1 for w in pos if pred.get(w) is True) / len(pos)
    tnr = sum(1 for w in neg if pred.get(w) is False) / len(neg)
    return 0.5 * (tpr + tnr)


def _held_out(vocab, pos_gt):
    from research.runners.d6_multiref_wm_production_organ import _REFERENT_NOUNS
    excluded = set(_REFERENT_NOUNS) | set(LR.HAND_NOUN_SEEDS) | set(LR.NONNOUN_SEEDS)
    return {w: (pos_gt[w] == "NOUN") for w in vocab if w in pos_gt and w not in excluded}


def _freq_only(seed_words, seed_lab, held, freq):
    """Frequency-only control: best single log-frequency threshold+direction fit on the SEED words only."""
    xs = [(math.log(freq[w] + 1), lab) for w, lab in zip(seed_words, seed_lab)]
    best = (-1, 0.0, 1)
    for thr in sorted({x for x, _ in xs}):
        for d in (1, -1):
            acc = sum(1 for x, lab in xs if ((d * (x - thr)) >= 0) == lab) / len(xs)
            if acc > best[0]:
                best = (acc, thr, d)
    _, thr, d = best
    return {w: bool(d * (math.log(freq[w] + 1) - thr) >= 0) for w in held}


def _organ_arm(seed, lex, noun_pool, nonnoun_pool, n_pairs):
    """Reply-level load-bearing test on the REAL D6 organ (fresh organ per trial: no carried-over binder state)."""
    from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan
    rng = np.random.default_rng(seed + 991)
    q = "who are we talking about?"
    out = {"trials": [], "fp_trials": []}
    for t in range(n_pairs):
        a, b = [str(x) for x in rng.choice(noun_pool, 2, replace=False)]
        sent = f"the {a} and the {b} walked in"
        row = {"a": a, "b": b}
        for cond in ("hand", "learned", "lesion"):
            org = MultiReferentWMOrgan(seed=seed)
            if cond == "hand":
                org.referent_lexicon = _NullLexicon()
            else:
                lex.set_lesion(cond == "lesion")
                org.referent_lexicon = lex
            j1 = org.judge(sent)
            j2 = org.judge(q)
            readout = (j2 or {}).get("readout")
            rec = set(((j2 or {}).get("recovered") or {}).values())
            row[cond] = {"in_scope": j1 is not None, "readout": readout,
                         "recovered_both": bool(j2 is not None and {a, b} <= rec)}
        lex.set_lesion(False)
        row["reply_differs_learned_vs_hand"] = row["learned"]["readout"] != row["hand"]["readout"]
        row["reply_differs_lesion_vs_hand"] = row["lesion"]["readout"] != row["hand"]["readout"]
        out["trials"].append(row)
    for t in range(n_pairs):
        v1, v2 = [str(x) for x in rng.choice(nonnoun_pool, 2, replace=False)]
        sent = f"we {v1} and {v2} all day"
        org = MultiReferentWMOrgan(seed=seed)
        org.referent_lexicon = lex
        j1 = org.judge(sent)
        out["fp_trials"].append({"v1": v1, "v2": v2, "in_scope": j1 is not None,
                                 "refs": (j1 or {}).get("input_order")})
    tr = out["trials"]
    n = len(tr)
    out["learned_recover_rate"] = sum(r["learned"]["recovered_both"] for r in tr) / n
    out["learned_in_scope_rate"] = sum(r["learned"]["in_scope"] for r in tr) / n
    out["hand_in_scope_rate"] = sum(r["hand"]["in_scope"] for r in tr) / n
    out["lesion_in_scope_rate"] = sum(r["lesion"]["in_scope"] for r in tr) / n
    out["reply_differs_learned_vs_hand_rate"] = sum(r["reply_differs_learned_vs_hand"] for r in tr) / n
    out["reply_differs_lesion_vs_hand_rate"] = sum(r["reply_differs_lesion_vs_hand"] for r in tr) / n
    out["fp_in_scope_rate"] = sum(r["in_scope"] for r in out["fp_trials"]) / len(out["fp_trials"])
    return out


class _NullLexicon:
    """Hand-table-only baseline injected explicitly (never reads the env flag)."""

    def is_referent(self, word):
        return False


def run_seed(seed, tokens, vocab, freq, pos_gt, W_frame, W_topic, n_pairs, spiking_n):
    t0 = time.time()
    held = _held_out(vocab, pos_gt)
    common = dict(seed=seed, k_seed=K_SEED, cv_seed=seed, tokens=tokens, vocab=vocab)
    arms = {
        "learned": LR.LearnedReferentLexicon(W=W_frame, **common),
        "shuffled": LR.LearnedReferentLexicon(W=W_frame, shuffle_control=True, **common),
        "label_permuted": LR.LearnedReferentLexicon(W=W_frame, label_permute=True, **common),
        "topical": LR.LearnedReferentLexicon(W=W_topic, **common),
    }
    res = {"seed": seed, "n_held": len(held), "n_held_noun": sum(held.values()),
           "seed_words": arms["learned"].seed_words}
    for name, lex in arms.items():
        pred = {w: lex.offline_sign(w) for w in held}
        res[f"bacc_{name}"] = _bacc(pred, held)
    lab = [w in LR.HAND_NOUN_SEEDS for w in arms["learned"].seed_words]
    res["bacc_freq_only"] = _bacc(_freq_only(arms["learned"].seed_words, lab, held, freq), held)
    lever("shuffle-graph", res["bacc_learned"], res["bacc_shuffled"], required=True)

    # spiking realization on a deterministic held-out subsample (or all if spiking_n <= 0)
    lex = arms["learned"]
    words = sorted(held)
    if spiking_n > 0 and len(words) > spiking_n:
        words = sorted(np.random.default_rng(seed + 5).choice(words, spiking_n, replace=False).tolist())
    sp = {w: lex.classify(w) for w in words}
    off = {w: lex.offline_sign(w) for w in words}
    agree = [sp[w] == off[w] for w in words if off[w] is not None]
    res["spiking_n"] = len(words)
    res["spiking_agree"] = undefined_if_empty("spiking-vs-offline agreement", len(agree),
                                              (sum(agree) / len(agree)) if agree else None, len(words))
    res["bacc_spiking"] = _bacc(sp, {w: held[w] for w in words})
    lex.set_lesion(True)
    les = {w: lex.classify(w) for w in words}
    lex.set_lesion(False)
    res["lesion_abstain_rate"] = sum(1 for w in words if les[w] is None) / max(len(words), 1)
    lever("spiking-drive lesion", res["bacc_spiking"], _bacc(les, {w: held[w] for w in words}), required=True)

    # determinism: independent rebuild (new bridge) at the same seed
    lex2 = LR.LearnedReferentLexicon(W=W_frame, **common)
    det_words = words[:100]
    h1 = hashlib.sha256(json.dumps([[w, lex.scores.get(w), lex.classify(w)] for w in det_words]).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps([[w, lex2.scores.get(w), lex2.classify(w)] for w in det_words]).encode()).hexdigest()
    res["deterministic"] = h1 == h2

    # organ (reply-level) arm
    noun_pool = sorted(w for w in held if held[w] and lex.offline_sign(w) is not None)
    nonnoun_pool = sorted(w for w in held if not held[w] and lex.offline_sign(w) is not None)
    # the organ arm draws from GT nouns (NOT from what the learner calls nouns) -> its recover rate is a real
    # recall measurement, not a tautology
    res["organ"] = _organ_arm(seed, lex, noun_pool, nonnoun_pool, n_pairs)
    res["elapsed_s"] = round(time.time() - t0, 1)
    void_if(not res["deterministic"], "rebuild at the same seed differs — substrate not seeded")
    return res


def score(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        rows.extend(d["per_seed"])
    by = {r["seed"]: r for r in rows}
    seeds = sorted(by)
    R = [by[s] for s in seeds]

    def m(key):
        v = [r[key] for r in R if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    def mo(key):
        v = [r["organ"][key] for r in R]
        return float(np.mean(v))

    g = GATE
    learned = m("bacc_learned")
    ctrl = max(m("bacc_shuffled"), m("bacc_freq_only"), m("bacc_label_permuted"))
    checks = {
        "G1_learned_mean": (learned, learned >= g["g1_mean"]),
        "G1_learned_min": (min(r["bacc_learned"] for r in R), min(r["bacc_learned"] for r in R) >= g["g1_min"]),
        "G2_shuffled_mean": (m("bacc_shuffled"), m("bacc_shuffled") <= g["g2_max"]),
        "G3_gap_vs_best_control": (learned - ctrl, learned - ctrl >= g["g3_gap"]),
        "G4_spiking_agree_min": (min(r["spiking_agree"] for r in R), min(r["spiking_agree"] for r in R) >= g["g4_agree"]),
        "G4_spiking_bacc_mean": (m("bacc_spiking"), m("bacc_spiking") >= g["g4_bacc"]),
        "G5_lesion_abstain_min": (min(r["lesion_abstain_rate"] for r in R),
                                  min(r["lesion_abstain_rate"] for r in R) >= g["g5_abstain"]),
        "G6_deterministic_all": (all(r["deterministic"] for r in R), all(r["deterministic"] for r in R)),
        "G7_learned_recover_mean": (mo("learned_recover_rate"), mo("learned_recover_rate") >= g["g7_recover"]),
        "G7_hand_in_scope_mean": (mo("hand_in_scope_rate"), mo("hand_in_scope_rate") <= g["g7_base_max"]),
        "G7_lesion_in_scope_mean": (mo("lesion_in_scope_rate"), mo("lesion_in_scope_rate") <= g["g7_lesion_max"]),
        "G7_fp_in_scope_mean": (mo("fp_in_scope_rate"), mo("fp_in_scope_rate") <= g["g7_fp_max"]),
    }
    complete = seeds == SEEDS6
    go = complete and all(ok for _, ok in checks.values())
    out = {"seeds": seeds, "complete_6seed": complete, "gate": GATE,
           "checks": {k: {"value": v, "pass": bool(ok)} for k, (v, ok) in checks.items()},
           "attribution_topical_bacc_mean": m("bacc_topical"),
           "means": {k: m(k) for k in ("bacc_learned", "bacc_shuffled", "bacc_label_permuted", "bacc_freq_only",
                                       "bacc_topical", "bacc_spiking", "spiking_agree", "lesion_abstain_rate")},
           "per_seed_bacc_learned": {s: by[s]["bacc_learned"] for s in seeds},
           "verdict": "GO" if go else ("INCOMPLETE" if not complete else "NO-GO")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS6)
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--max-chars", type=int, default=8_000_000)
    ap.add_argument("--top-v", type=int, default=2000)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--spiking-n", type=int, default=400)
    ap.add_argument("--json", default="")
    ap.add_argument("--score", default="", help="dir (or glob) of per-seed JSONs -> aggregate verdict")
    a = ap.parse_args()
    if a.score:
        paths = sorted(glob.glob(os.path.join(a.score, "lexicon_referent_s*.json")) if os.path.isdir(a.score)
                       else glob.glob(a.score))
        out = score(paths)
        print(json.dumps(out, indent=1))
        dst = os.path.join(a.score if os.path.isdir(a.score) else os.path.dirname(a.score), "verdict.json")
        json.dump(out, open(dst, "w"), indent=1)
        return
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    tokens = load_tokens(corpus, a.max_chars)
    vocab, freq = build_vocab(tokens, a.top_v)
    fx = json.load(open(FIXTURE))
    if fx.get("top_v") != a.top_v or fx.get("max_chars") != a.max_chars:
        raise SystemExit("fixture was built for a different vocab config — rebuild it")
    pos_gt = fx["pos"]
    cover = sum(1 for w in vocab if w in pos_gt)
    if cover < 0.8 * fx["n"]:
        raise SystemExit(f"corpus vocab covers only {cover}/{fx['n']} fixture words — WRONG CORPUS (degenerate)")
    t0 = time.time()
    W_frame = LR.build_frame_graph(tokens, vocab)
    W_topic = LR.build_topical_graph(tokens, vocab)
    print(f"[lexicon-referent] tokens={len(tokens)} vocab={len(vocab)} fixture_cover={cover} "
          f"graphs {time.time() - t0:.1f}s", flush=True)
    per = []
    for s in a.seeds:
        r = run_seed(s, tokens, vocab, freq, pos_gt, W_frame, W_topic, a.n_pairs, a.spiking_n)
        print(f"[seed {s}] learned={r['bacc_learned']:.3f} shuf={r['bacc_shuffled']:.3f} "
              f"perm={r['bacc_label_permuted']:.3f} freq={r['bacc_freq_only']:.3f} topical={r['bacc_topical']:.3f} "
              f"spk={r['bacc_spiking']:.3f} agree={r['spiking_agree']} lesion_abst={r['lesion_abstain_rate']:.2f} "
              f"det={r['deterministic']} organ recover={r['organ']['learned_recover_rate']:.2f} "
              f"hand={r['organ']['hand_in_scope_rate']:.2f} lesion={r['organ']['lesion_in_scope_rate']:.2f} "
              f"fp={r['organ']['fp_in_scope_rate']:.2f} ({r['elapsed_s']}s)", flush=True)
        per.append(r)
    out = {"runner": "_lexicon_learned_referent_derisk", "corpus": os.path.basename(corpus),
           "tokens": len(tokens), "vocab": len(vocab), "k_seed": K_SEED, "gate": GATE, "per_seed": per}
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(out, open(dst, "w"), indent=1)
        print("wrote", dst)


if __name__ == "__main__":
    main()
