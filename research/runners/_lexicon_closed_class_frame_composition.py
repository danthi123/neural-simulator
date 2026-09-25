"""DESIGN EVIDENCE for AMENDMENT 3 of the frame-junction pre-registration
(research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md): how many heard occurrences
the junction layer can see at all (host analysis of the environment; no circuit, no learning, no decision).

A junction J(a, b) receives only FR(-1, a) and FR(+1, b), so it can fire only on an occurrence whose left AND right
neighbours are both among the C most-heard context words (a COMPLETE frame). An occurrence with only one frequent
neighbour (left-only / right-only) or none drives no junction: under the strict AND it is invisible to the category
pools. This runner counts, for the sampled K occurrences the lexicon actually presents (`FrameEnvironment.
occurrences(word, K_OCC, seed)`, the pause-token environment the junction variant hears), how each occurrence splits
into complete / left-only / right-only / none:
  * over the whole teacher curriculum (38 + 37 seed words), per seed;
  * for every heard NON-ground-truth word the round-2 dev check found SILENT in both pools (the G4 failure set,
    read from the committed round-2 junction artifacts), plus 'might' (the round-2 G2 failure) and the words
    AMENDMENT 3 names as its G2 risk ('most', 'crazy', 'wonderful', 'amazing'), each with its frequent left/right
    neighbours per occurrence kind.
It is a count over the environment, not the spiking circuit; it predicts nothing about decisions and no constant is
chosen from it.

    bash tools/mem_ok.sh 2 && bash tools/memcap.sh 2 -- env SIM_BACKEND=numpy .venv/bin/python -u -m \
        research.runners._lexicon_closed_class_frame_composition --corpus data/corpus/tinystories.txt \
        --json research/findings/raw/_lexicon_closed_class/frame_composition_s7_s42.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

ROUND2 = {7: "research/findings/raw/_lexicon_closed_class/dev_s7_amendment2/junction_s7.json",
          42: "research/findings/raw/_lexicon_closed_class/dev_s42_amendment2/junction_s42.json"}


def composition(env, word, k, seed, with_context=False):
    """Counter over {'complete', 'left_only', 'right_only', 'none'} for the K presented occurrences of `word`
    (with_context: also the frequent left / right neighbours per occurrence kind, most common first)."""
    from research.runners.lexicon_frame_junction import _OFF_L, _OFF_R, PAUSE_TOKEN
    C = env.C
    occ = env.occurrences(word, k, seed)
    if occ is None:
        return None
    c = collections.Counter()
    ctx = collections.defaultdict(collections.Counter)
    name = lambda i: "<pause>" if env.ctx[i] == PAUSE_TOKEN else env.ctx[i]  # noqa: E731
    for feats in occ:
        lf = [f - _OFF_L * C for f in feats if _OFF_L * C <= f < (_OFF_L + 1) * C]
        rf = [f - _OFF_R * C for f in feats if _OFF_R * C <= f < (_OFF_R + 1) * C]
        kind = "complete" if (lf and rf) else "left_only" if lf else "right_only" if rf else "none"
        c[kind] += 1
        for i in lf:
            ctx[kind + "_left"][name(i)] += 1
        for i in rf:
            ctx[kind + "_right"][name(i)] += 1
    out = {k2: int(c.get(k2, 0)) for k2 in ("complete", "left_only", "right_only", "none")}
    if with_context:
        out["context"] = {key: cnt.most_common() for key, cnt in sorted(ctx.items())}
    return out


def main():
    from research.runners import lexicon_spiking_frame_category as L
    from research.runners.lexicon_frame_junction import load_tokens_with_pause
    from research.runners._comprehension_learned_animacy_cue_derisk import build_vocab
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    tokens = load_tokens_with_pause(corpus, 8_000_000)              # exactly what get_lexicon()'s junction branch hears
    vocab, _ = build_vocab(tokens, 2000)
    env = L.FrameEnvironment(tokens, vocab + [w for w in L.HAND_NOUN_SEEDS + L.NONNOUN_SEEDS if w not in vocab])
    words, labels = L.seed_curriculum(env)
    out = {"corpus_path": corpus, "k_occ": L.K_OCC, "ctx_c": env.C, "per_seed": {}}
    with open(corpus, "rb") as f:
        out["corpus_sha256"] = hashlib.sha256(f.read()).hexdigest()
    for seed, path in ROUND2.items():
        tot = collections.Counter()
        for w in words:
            for k2, v in composition(env, w, L.K_OCC, seed).items():
                tot[k2] += v
        n = sum(tot.values())
        r2 = json.load(open(os.path.join(_REPO, path)))["arms"]["intact"]
        # + round 2's G2 failure ('might') and the words v2 admitted that AMENDMENT 3 names as its G2 risk
        probe = sorted(set(r2["silent_non_words"]) | {"might", "most", "crazy", "wonderful", "amazing"})
        out["per_seed"][str(seed)] = {
            "curriculum": {**{k2: int(v) for k2, v in tot.items()}, "n": int(n),
                           "one_sided_fraction": (tot["left_only"] + tot["right_only"]) / n,
                           "complete_fraction": tot["complete"] / n},
            "round2_artifact": path,
            "round2_silent_non_words": r2["silent_non_words"],
            "words": {w: composition(env, w, L.K_OCC, seed, with_context=True) for w in probe}}
        cur = out["per_seed"][str(seed)]["curriculum"]
        print(f"seed {seed} curriculum {cur}", flush=True)
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(out, open(dst, "w"), indent=1)
        print("wrote", dst)


if __name__ == "__main__":
    main()
