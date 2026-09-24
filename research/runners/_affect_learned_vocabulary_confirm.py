"""DEV confirmation (seed 7): read a weight file trained at the chosen constants with the FROZEN read constants of the
module (G from the file, MIN_RATE, V_MIN, R_REF) and apply the calibration rule's admissibility test. Seed 7 only."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import affect_learned_vocabulary as A  # noqa: E402
from research.runners import _affect_learned_vocabulary_derisk as D  # noqa: E402
from research.runners._affect_learned_vocabulary_calibrate import _sentence_valence, DEV_SEED  # noqa: E402


def main(path):
    lex = D.load_lexicon()
    _, held = A.seed_split(DEV_SEED)
    rd = A.load_reader(path, DEV_SEED)
    dev_neg = sorted(w for w, v in lex.items() if v < 0 and not D.eval_half(w))
    dev_pos = sorted(w for w, v in lex.items() if v > 0 and not D.eval_half(w))
    val = {w: rd.read(w) for w in sorted(set(dev_neg) | set(dev_pos) | set(held))}
    f = lambda ws, c: float(np.mean([c(val[w]) for w in ws]))  # noqa: E731
    fact = [_sentence_valence(t, rd) for t in D.FACT_DEV]
    out = {"weights": path, "g": rd.g, "min_rate": A.MIN_RATE, "v_min": A.V_MIN, "r_ref": rd.r_ref,
           "dev_neg_recall": f(dev_neg, lambda v: v < 0), "dev_neg_wrong": f(dev_neg, lambda v: v > 0),
           "dev_pos_recall": f(dev_pos, lambda v: v > 0), "dev_pos_wrong": f(dev_pos, lambda v: v < 0),
           "dev_contrast_D": f(dev_neg, lambda v: v < 0) - f(dev_pos, lambda v: v < 0),
           "held_seed_decided": sum(val[w] != 0 for w in held),
           "held_seed_sign_acc": (float(np.mean([np.sign(val[w]) == np.sign(held[w]) for w in held if val[w] != 0]))
                                  if any(val[w] != 0 for w in held) else None),
           "fact_dev_within": float(np.mean([abs(v) <= 0.25 for v in fact])),
           "fact_dev": [[t, v] for t, v in zip(D.FACT_DEV, fact) if v != 0.0],
           "named": {w: rd.read(w) for w in D.NAMED}}
    out["admissible"] = out["dev_neg_wrong"] <= 0.10 and out["fact_dev_within"] >= 0.95
    out["passes_rule"] = out["admissible"] and out["dev_neg_recall"] >= 0.27
    dst = os.path.join(_REPO, "research", "findings", "raw", "_affect_learned_vocab", "dev_s7", "confirmation.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("fact_dev",)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
