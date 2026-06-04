"""(v) MULTI-MODAL grounding: ground the VISUAL concepts via the real V1 Gabor bank AND the abstract concepts via
the project's word encoder, in ONE decorrelated codebook, then run the unified-agent benchmark.

#4 grounded the visual subset (V1 Gabor → concept codes); abstract words (verbs, function words) have no canonical
image. The honest target is MULTI-MODAL: vision grounds visual concepts, language grounds the rest. Here: nouns →
real V1 Gabor responses (`sim/visual_cortex.py`); verbs + adjectives → the word encoder
(`sim.text_embeddings.vocab_to_drive_pattern`, the grounded word-cue level already validated). The two modalities
are block-padded into one feature space, decorrelated (ventral-hierarchy stand-in), projected to phases, and fed as
the agent's `external_codes`. Question: does a MIXED-modality codebook still match constructed parity, or does the
cross-modal coherence structure break composition? Reported raw vs decorrelated (mirroring #4). Reuse-by-import.

  python -m research.runners.unified_agent_multimodal_grounded --seeds 42 43
"""
from __future__ import annotations
import argparse
import json
import numpy as np

import research.runners.unified_agent_visual_grounded as uvg
from research.runners.unified_agent_visual_grounded import _v1_codes_for_tokens, _decorrelate
from research.runners._visual_grounding_probe import _v1_matrix
from research.runners.unified_agent_benchmark import (
    build_vocab, ALL_FACTS, CATEGORIES, WHO_QUERIES, ABSTAIN_QUERIES, aggregate)
from research.runners.nested_composition_agent import NestedCompositionAgent
from sim.text_embeddings import vocab_to_drive_pattern

D = 2048
V1_DIM = 8 * 4 * 16 * 16     # 8192
WORD_DIM = 2048


def build_multimodal_features(nouns, verbs, adjs):
    """nouns -> real V1 Gabor responses (visual); verbs+adjs -> word-encoder patterns (abstract). Block-padded into
    one (N, V1_DIM+WORD_DIM) feature matrix so the two modalities occupy disjoint feature blocks."""
    uvg.STIMULUS_MODE = "tiled"
    W, _ = _v1_matrix()
    tokens = nouns + verbs + adjs
    v1 = _v1_codes_for_tokens(nouns, W)                              # (n_nouns, V1_DIM), unit-normalized
    abstract = verbs + adjs
    word = np.stack([vocab_to_drive_pattern(t, n_neurons=WORD_DIM, sparsity=0.1) for t in abstract]).astype(np.float64)
    word = word / (np.linalg.norm(word, axis=1, keepdims=True) + 1e-12)
    dim = V1_DIM + WORD_DIM
    feats = np.zeros((len(tokens), dim), dtype=np.float64)
    feats[:len(nouns), :V1_DIM] = v1                                 # nouns in the visual block
    feats[len(nouns):, V1_DIM:] = word                              # verbs+adjs in the word block
    return feats, dim, tokens


def run_seed_mm(seed, feats, dim, tokens, nouns, verbs, adjs, decorrelate=True):
    f = _decorrelate(feats) if decorrelate else feats
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((D, dim)) + 1j * rng.standard_normal((D, dim))
    Z = proj @ f.T
    ext = {t: np.angle(Z[:, i]) for i, t in enumerate(tokens)}      # phase angles (agent's external_codes format)
    agent = NestedCompositionAgent(nouns, verbs, adjs, D=D, seed=seed, external_codes=ext)
    for ag, ac, pa in ALL_FACTS:
        agent.learn(ag, ac, pa)
    res = {}
    for name, facts in CATEGORIES:
        ok = sum(int(agent.query_patient(ag, ac) == (pa if isinstance(pa, str) else agent._render_filler(pa)))
                 for ag, ac, pa in facts)
        res[name] = [ok, len(facts)]
    res["who-query"] = [sum(int(agent.query_agent(ac, pn) == w) for ac, pn, w in WHO_QUERIES), len(WHO_QUERIES)]
    res["abstain"] = [sum(int(agent.query_patient(ag, ac) is None) for ag, ac in ABSTAIN_QUERIES), len(ABSTAIN_QUERIES)]
    return {"seed": seed, "categories": res}


def _report(label, seed_results):
    agg, gok, gtot = aggregate(seed_results)
    print(f"\n  --- {label} (multi-seed) ---", flush=True)
    for c, (ok, tot, rate) in agg.items():
        print(f"    {c:<16} {ok:>3}/{tot:<3} = {rate*100:5.1f}%", flush=True)
    print(f"    OVERALL: {gok}/{gtot} = {gok/gtot*100:.1f}%", flush=True)
    return agg, gok, gtot


def main():
    ap = argparse.ArgumentParser(description="Multi-modal grounding: nouns via V1, verbs+adjs via word encoder.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nouns, verbs, adjs = build_vocab()
    print(f"=== MULTI-MODAL grounding | nouns->V1 Gabor (visual), verbs+adjs->word encoder (abstract) | "
          f"{len(nouns)+len(verbs)+len(adjs)} concepts | D={D} | seeds={args.seeds} ===", flush=True)
    feats, dim, tokens = build_multimodal_features(nouns, verbs, adjs)
    print(f"  combined feature dim {dim} (V1 {V1_DIM} + word {WORD_DIM}); nouns visual, verbs+adjs abstract", flush=True)

    raw = [run_seed_mm(s, feats, dim, tokens, nouns, verbs, adjs, decorrelate=False) for s in args.seeds]
    dec = [run_seed_mm(s, feats, dim, tokens, nouns, verbs, adjs, decorrelate=True) for s in args.seeds]
    agg_raw, _, _ = _report("RAW mixed-modality (no decorrelation)", raw)
    agg_dec, gok, gtot = _report("DECORRELATED mixed-modality (ventral-hierarchy stand-in)", dec)
    print("\n  (numpy agent on single-modality V1-grounded + decorrelate = 92.3% constructed parity, #4)", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"D": D, "seeds": args.seeds, "raw": raw, "decorrelated": dec,
                       "agg_raw": agg_raw, "agg_dec": agg_dec}, f, indent=2)
        print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
