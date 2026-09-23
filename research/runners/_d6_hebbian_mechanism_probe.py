"""D6 mechanism probe: the Hebbian-written fact block vs the direct composite copy on a small OneBrainComposer.

Records, per seed: the learned-vs-direct phase error per block, the per-role cleanup margins for both write paths,
the frozen in-conversation block's |w| and recall, and the engram read (held / readout) of built vs frozen blocks.
numpy, 12-word vocab, D=128 (~1 min). Usage:
  .venv/bin/python -m research.runners._d6_hebbian_mechanism_probe --seeds 42 --json research/findings/raw/_d6_learn_through_use/mechanism_s42.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

VOCAB = sorted({"dog", "chase", "cat", "eat", "fish", "wolf", "hunt", "deer", "fox", "berry", "bird", "worm"})


def _build(store, freeze, seed):
    from research.runners.one_brain_composer import OneBrainComposer
    os.environ["BRAIN_D6_HEBBIAN_STORE"] = store
    os.environ["BRAIN_D6_HEBBIAN_FREEZE"] = freeze
    return OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, vocab_headroom=2)


def _block(c, i):
    return np.array([complex(w) for (_p, _q, w) in c.store_conns[i * c.D:(i + 1) * c.D]])


def _margins(c, i):
    return {r: {"word": v[0], "margin": round(float(v[2]), 4)} for r, v in c._block_role_scores(i).items()}


def probe(seed):
    from research.runners import d6_hebbian_store as d6
    d = _build("0", "0", seed); d.hear("dog chase cat"); d.hear("wolf hunt deer")
    h = _build("1", "0", seed); h.hear("dog chase cat"); h.hear("wolf hunt deer")
    f = _build("1", "1", seed); f.hear("dog chase cat")
    with d6.conversation_write(f):
        f.hear("wolf hunt deer")
    out = {"seed": seed, "blocks": []}
    for i in range(2):
        dphi = np.angle(_block(h, i) * np.conj(_block(d, i)))
        out["blocks"].append({"block": i, "phase_err_mean_rad": round(float(np.mean(np.abs(dphi))), 5),
                              "phase_err_max_rad": round(float(np.max(np.abs(dphi))), 5),
                              "margins_direct": _margins(d, i), "margins_hebbian": _margins(h, i)})
    out["recall"] = {"direct": [d.query_patient("dog", "chase"), d.query_patient("wolf", "hunt")],
                     "hebbian": [h.query_patient("dog", "chase"), h.query_patient("wolf", "hunt")],
                     "frozen_conv": [f.query_patient("dog", "chase"), f.query_patient("wolf", "hunt")]}
    out["frozen_block_max_abs_w"] = float(np.max(np.abs(_block(f, 1))))
    out["engram"] = {"frozen_built_block": d6.engram_held(f, 0), "frozen_conv_block": d6.engram_held(f, 1)}
    out["hebbian_encode_diag"] = h._d6_last_encode
    for k in ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_HEBBIAN_FREEZE"):
        os.environ.pop(k, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    res = {"runner": "research.runners._d6_hebbian_mechanism_probe", "per_seed": [probe(s) for s in a.seeds]}
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(res, open(a.json, "w"), indent=2, default=str)
    print(json.dumps(res, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
