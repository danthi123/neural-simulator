"""Build the EVAL-ONLY ground-truth fixture for the learned-referent lexicon de-risk (host curriculum/eval prep).

Restricts the committed spaCy dominant-POS map (`research/findings/raw/_corpus_pos_map.json`, built by
`_corpus_pos_map.py` over TinyStories+SimpleWiki with an independent tagger) to the learner's own TinyStories
vocabulary, and writes it to `research/fixtures/lexicon_referent_pos_gt.json` (research/fixtures/ is provisioned to
the mini-PC pool; findings/raw/ is not). The LEARNER never reads this file — only the de-risk's scorer does, as the
held-out truth. Deterministic.

    python -m research.runners._lexicon_build_pos_gt_fixture --corpus data/corpus/tinystories.txt
"""
import argparse
import json
import os

from research.runners._comprehension_learned_animacy_cue_derisk import load_tokens, build_vocab

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--max-chars", type=int, default=8_000_000)
    ap.add_argument("--top-v", type=int, default=2000)
    ap.add_argument("--pos-map", default=os.path.join(_REPO, "research/findings/raw/_corpus_pos_map.json"))
    ap.add_argument("--out", default=os.path.join(_REPO, "research/fixtures/lexicon_referent_pos_gt.json"))
    a = ap.parse_args()
    toks = load_tokens(a.corpus, a.max_chars)
    vocab, _ = build_vocab(toks, a.top_v)
    pm = json.load(open(a.pos_map))
    gt = {w: pm[w]["pos"] for w in vocab if w in pm and pm[w].get("pos") in ("NOUN", "VERB", "ADJ")}
    out = {"source": "research/findings/raw/_corpus_pos_map.json (spaCy en_core_web_sm dominant POS)",
           "corpus": os.path.relpath(a.corpus, _REPO) if a.corpus.startswith(_REPO) else os.path.basename(a.corpus),
           "max_chars": a.max_chars, "top_v": a.top_v, "n": len(gt),
           # no simulation runs here (a dictionary lookup over a committed tagger map); recorded for the device gate
           "backend": "numpy", "device": "cpu",
           "counts": {p: sum(1 for v in gt.values() if v == p) for p in ("NOUN", "VERB", "ADJ")},
           "pos": dict(sorted(gt.items()))}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=0, sort_keys=False)
    print(json.dumps({k: out[k] for k in ("n", "counts")}))


if __name__ == "__main__":
    main()
