"""DEV calibration of the learned affect vocabulary's READ constants on seed 7 (NOT an evaluation seed).

Input: one or more seed-7 weight files (one per training variant), trained on the SAME heard stream as the
evaluation seeds. Every read goes through the production reader (`load_reader`). The rule below was written before
the dev weights were read; it uses only DEV items: the even crc32 half of the independent lexicon, the seed-7
cross-validation held-out seed words, and the FACT_DEV sentences. The EVAL half and FACT_EVAL are never read here.

RULE. For each (variant, G in GRID_G, MIN_RATE in GRID_MIN_RATE):
  R_REF := the median rate margin |r+ - r-| over the decided dev words (lexicon dev half + held-out seeds), divided
           by 0.5 (so a median decided word reads |valence| 0.5, the middle of the organ's graded range).
  admissible iff  dev-negative wrong-sign share <= 0.10  AND  FACT_DEV share with |appraisal| <= 0.25 >= 0.95.
  choose the admissible (variant, G, MIN_RATE) with the highest dev-negative recall; ties -> the variant listed
  first, then the lower G, then the lower MIN_RATE.
AMENDED 2026-09-24 ~07:20 EDT, after the first dev table (three variants read at G = 500 only, best dev-negative
recall 0.126): G, the synaptic gain from learned excess to weight, sets the lowest threshold the read can have (the
pool's rheobase), and MIN_RATE can only raise it. Because the learned drive cancels in the training increment, G acts
almost only at read time, so the dev weights are re-read over GRID_G. The chosen G is then used to TRAIN the evaluation
seeds, and one confirmation dev run is trained at that G before the pre-registration is committed. No admissible pair, or a best recall < 0.27, -> the design is predicted to fail and no
  evaluation seed is staged.
"""
from __future__ import annotations

import json
import os
import sys
import zlib

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import affect_learned_vocabulary as A  # noqa: E402
from research.runners import _affect_learned_vocabulary_derisk as D  # noqa: E402
from research.runners._affect_distributional_tag_derisk import WARRINER, STOP  # noqa: E402

GRID_MIN_RATE = (0.002, 0.005, 0.01, 0.02, 0.03)
GRID_G = (500.0, 1000.0, 2000.0, 4000.0)
DEV_SEED = 7


def _sentence_valence(text, reader):
    """appraise_text's flag-on arithmetic with this reader (valence only)."""
    from research.runners import affect_production_organ as AO
    vals = []
    for w in (t.lower() for t in AO._WORD_RE.findall(text or "")):
        if w in STOP:
            continue
        if w not in WARRINER:
            v = reader.read(w)
            if v != 0.0:
                vals.append(v)
            continue
        v9, a9 = WARRINER[w]
        if abs(v9 - 5.0) < AO._STRONG_MARGIN:
            continue
        v9, _ = AO._get_learned_valence().get(w, (v9, a9)) if AO.dr2_enabled() else (v9, a9)
        vals.append((v9 - 5.0) / 4.0)
    return float(np.mean(vals)) if vals else 0.0


def evaluate_variant(path):
    lex = D.load_lexicon()
    _, held = A.seed_split(DEV_SEED)
    rd = A.load_reader(path, DEV_SEED)
    dev_neg = sorted(w for w, v in lex.items() if v < 0 and not D.eval_half(w))
    dev_pos = sorted(w for w, v in lex.items() if v > 0 and not D.eval_half(w))
    words = sorted(set(dev_neg) | set(dev_pos) | set(held))
    rows = []
    for g, mr in [(g, mr) for g in GRID_G for mr in GRID_MIN_RATE]:
        rd.g = g
        rd._cache.clear()
        rates = {w: rd.read_rates(w) for w in words}
        A.MIN_RATE = mr
        marg = [abs(float(r[0][0] - r[1][0])) for r in rates.values() if r is not None and max(r[0][0], r[1][0]) >= mr]
        r_ref = (float(np.median(marg)) / 0.5) if marg else A.R_REF
        rd.r_ref = r_ref
        rd._cache.clear()
        val = {w: rd.read(w) for w in words}
        f = lambda ws, c: float(np.mean([c(val[w]) for w in ws])) if ws else None  # noqa: E731
        fact = [_sentence_valence(t, rd) for t in D.FACT_DEV]
        rows.append({"g": g, "min_rate": mr, "r_ref": r_ref, "dev_neg_recall": f(dev_neg, lambda v: v < 0),
                     "dev_neg_wrong": f(dev_neg, lambda v: v > 0), "dev_pos_recall": f(dev_pos, lambda v: v > 0),
                     "dev_pos_wrong": f(dev_pos, lambda v: v < 0),
                     "dev_contrast_D": f(dev_neg, lambda v: v < 0) - f(dev_pos, lambda v: v < 0),
                     "held_seed_decided": sum(val[w] != 0 for w in held),
                     "held_seed_sign_acc": (float(np.mean([np.sign(val[w]) == np.sign(held[w]) for w in held
                                                           if val[w] != 0])) if any(val[w] != 0 for w in held) else None),
                     "fact_dev_within": float(np.mean([abs(v) <= 0.25 for v in fact])),
                     "fact_dev_max_abs": float(np.max(np.abs(fact))),
                     "named": {w: rd.read(w) for w in D.NAMED}})
    return rows


def main():
    variants = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
    out = {"rule": __doc__, "variants": {}}
    best = None
    for name, path in variants:
        rows = evaluate_variant(path)
        out["variants"][name] = {"path": path, "rows": rows}
        for r in rows:
            ok = r["dev_neg_wrong"] <= 0.10 and r["fact_dev_within"] >= 0.95
            r["admissible"] = ok
            if ok and (best is None or r["dev_neg_recall"] > best[2]["dev_neg_recall"]):
                best = (name, path, r)
            print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                    if k != "named"}), flush=True)
            print("   named", {k: round(v, 3) for k, v in r["named"].items()})
    if best is None or best[2]["dev_neg_recall"] < 0.27:
        out["choice"] = None
        print("CHOICE: NONE (design predicted to fail)")
    else:
        out["choice"] = {"variant": best[0], "path": best[1], "g": best[2]["g"], "min_rate": best[2]["min_rate"],
                         "r_ref": best[2]["r_ref"], "dev_neg_recall": best[2]["dev_neg_recall"]}
        print("CHOICE", out["choice"])
    dst = os.path.join(_REPO, "research", "findings", "raw", "_affect_learned_vocab", "dev_s7", "calibration.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
