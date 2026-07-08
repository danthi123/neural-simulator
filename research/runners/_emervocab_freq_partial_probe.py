"""Decisive H1/H2 probe for the K=1024 breadth freq-confound.

The 6-seed TinyStories breadth run showed: at K=1024 the emergent learned co-occurrence
margin (+0.090) MATCHES the host-PPMI batch ceiling (+0.093) but a rank-1 frequency-only
baseline (+0.111) beats BOTH. The runner's auto-gate stamped that "MECHANISM gap". Two
hypotheses:

  H1 (probe/data property, mechanism fine): the 8-category probe is frequency-stratified
     at K=1024, so frequency is a strong MARGINAL predictor -- but co-occurrence still
     carries category structure INDEPENDENT of frequency (the raw margin just looks
     freq-dominated).
  H2 (mechanism capacity limit): co-occurrence adds NOTHING beyond frequency at K=1024
     -> a real large-vocab code-capacity gap needing hierarchical/sparse codes.

Decisive single-variable test = PARTIAL CORRELATION. For every pair (i,j) of probe words:
   x = cosine(code_i, code_j)         (the code's similarity)
   y = 1[same category]               (the target structure)
   z = cosine(freqonly_i, freqonly_j) (the FREQUENCY-driven similarity -- the confound)
partial_corr(x, y | z) = does the code-similarity predict same-category AFTER removing the
frequency-similarity effect? H1 => partial_corr(learned) > 0 (and host > 0); H2 => ~0.

Reuse-by-import from _emergent_vocab_breadth_scale_derisk. numpy-only, offline.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._emergent_vocab_breadth_scale_derisk import (
    TAXONOMY_8x8, discover_vocab, learn_stream_codes, batch_count_matrix,
    ppmi_matrix, build_probe, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners._emergent_vocab_breadth_scale_derisk import DEFAULT_CORPUS
from research.runners.corpus_stream import load_token_stream_multi


def _load_stories(path):
    # use the runner's EXACT loader so the probe measures on the identical stories
    return load_token_stream_multi(path, max_stories=None)


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _partial_corr(x, y, z):
    """partial_corr(x,y|z) = (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))."""
    rxy, rxz, ryz = _pearson(x, y), _pearson(x, z), _pearson(y, z)
    denom = np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / denom), rxy, rxz, ryz


def _pair_sims(codes, rows):
    """Upper-triangular pairwise cosine over the probe rows; returns (sims, same_flag_getter-ready idx pairs)."""
    C = codes[rows]
    n = C.shape[0]
    nrm = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    U = C / nrm
    S = U @ U.T
    iu = np.triu_indices(n, k=1)
    return S[iu], iu


def run(path, Ks, seeds):
    stories = _load_stories(path)
    print(f"[freq-partial probe] corpus={path} stories={len(stories)} "
          f"tokens={sum(len(s) for s in stories)}", flush=True)
    out = {"corpus": path, "per_K": {}}
    for K in Ks:
        vocab, gfreq = discover_vocab(stories, K)
        target_set = set(vocab)
        hubs = []
        for w, _ in gfreq.most_common():
            if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
                continue
            hubs.append(w)
            if len(hubs) >= N_HUB:
                break
        probe_rows, probe_labels, probe_words, n_cat, per_cat = build_probe(vocab, TAXONOMY_8x8)
        if probe_rows.size < 4:
            print(f"  [K={K}] probe too thin ({probe_rows.size}w) -- skip", flush=True)
            continue
        # same-category flag over the upper-triangular probe pairs
        lab = probe_labels
        n = lab.size
        iu = np.triu_indices(n, k=1)
        same = (lab[iu[0]] == lab[iu[1]]).astype(float)

        recs = []
        for seed in seeds:
            learned, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
            fo, _ = learn_stream_codes(seed, stories, vocab, hubs, freq_only=True, window=WINDOW)
            C = batch_count_matrix(stories, vocab, hubs, window=WINDOW)
            host = ppmi_matrix(C, 0.75)

            s_learned, _ = _pair_sims(learned, probe_rows)
            s_fo, _ = _pair_sims(fo, probe_rows)
            s_host, _ = _pair_sims(host, probe_rows)

            pc_learned, rxy_l, rxz_l, _ = _partial_corr(s_learned, same, s_fo)
            pc_host, rxy_h, rxz_h, _ = _partial_corr(s_host, same, s_fo)
            # raw correlation of freq-only with same-category (the marginal confound strength)
            r_fo = _pearson(s_fo, same)
            recs.append({
                "seed": seed,
                "raw_r_learned_vs_samecat": rxy_l,
                "raw_r_host_vs_samecat": rxy_h,
                "raw_r_freqonly_vs_samecat": r_fo,
                "corr_learned_with_freqonly": rxz_l,
                "PARTIAL_learned_given_freq": pc_learned,
                "PARTIAL_host_given_freq": pc_host,
            })
        # aggregate
        def mean(k): return float(np.mean([r[k] for r in recs]))
        agg = {k: mean(k) for k in recs[0] if k != "seed"}
        agg["n_probe_words"] = int(probe_rows.size)
        agg["n_categories"] = int(n_cat)
        out["per_K"][K] = {"agg": agg, "per_seed": recs}
        print(f"  [K={K}] probe={probe_rows.size}w/{n_cat}cat  "
              f"raw r(learned,samecat)={agg['raw_r_learned_vs_samecat']:+.3f}  "
              f"raw r(freqonly,samecat)={agg['raw_r_freqonly_vs_samecat']:+.3f}  ||  "
              f"PARTIAL(learned|freq)={agg['PARTIAL_learned_given_freq']:+.3f}  "
              f"PARTIAL(host|freq)={agg['PARTIAL_host_given_freq']:+.3f}", flush=True)
    # verdict
    print("\n  VERDICT:", flush=True)
    for K, d in out["per_K"].items():
        pl = d["agg"]["PARTIAL_learned_given_freq"]
        ph = d["agg"]["PARTIAL_host_given_freq"]
        tag = ("H1 (co-occurrence carries category signal BEYOND frequency -> mechanism fine, "
               "the raw freq-dominance is a probe property)" if pl > 0.05 else
               "H2 (co-occurrence adds ~nothing beyond frequency -> a real large-vocab capacity limit)")
        print(f"    K={K}: PARTIAL(learned|freq)={pl:+.3f}, PARTIAL(host|freq)={ph:+.3f} -> {tag}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default=DEFAULT_CORPUS)
    ap.add_argument("--vocab-sizes", default="256,1024")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Ks = [int(x) for x in a.vocab_sizes.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    out = run(a.corpus_path, Ks, seeds)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
