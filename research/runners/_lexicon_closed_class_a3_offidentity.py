"""INTEGRITY SMOKE 1 of AMENDMENT 3 (research/findings/2026-09-24-lexicon-closed-class-frame-junction-
PREREGISTRATION.md): with `BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL` unset, the post-change junction lexicon is the
AMENDMENT 2 junction lexicon exactly -- asserted in data, not inferred from reading the code.

The pre-change `lexicon_frame_junction.py` is loaded straight from git (`--pre-sha`, default the round-2 merge
db8db2a5b) into a private module; the post-change one is the working tree's. Both build a junction lexicon at the
same seed on the same pause-token environment and must give identical:
  * connection data / indices / indptr hashes right after the build;
  * learned-edge weight hash and connection-data hash after a SHORT training (the curriculum's first 2 nouns and
    last 2 non-nouns, 1 epoch -- the full 8-epoch 75-word training costs ~1.6 h per lexicon on numpy, declared);
  * decide() outputs (decision, CN rate, CX rate) on a fixed probe list;
  * the learned_edge lesion's homeostatic-settle weights (AMENDMENT 2's R4 path, which AMENDMENT 3 edits).
Not evidence for any gate. Exit 1 on any difference.

    bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy .venv/bin/python -u -m \
        research.runners._lexicon_closed_class_a3_offidentity --seed 7 --corpus data/corpus/tinystories.txt \
        --json research/findings/raw/_lexicon_closed_class/a3_offidentity_s7.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

PROBES = ("owl", "most", "might", "dog", "when", "wonderful")
_REL = "research/runners/lexicon_frame_junction.py"


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def load_pre_change(sha: str):
    src = subprocess.check_output(["git", "show", f"{sha}:{_REL}"], cwd=_REPO, text=True)
    mod = types.ModuleType("pre_change_lexicon_frame_junction")
    mod.__file__ = os.path.join(_REPO, _REL)
    exec(compile(src, f"{sha}:{_REL}", "exec"), mod.__dict__)
    return mod, hashlib.sha256(src.encode()).hexdigest()


def fingerprint(M, seed, env, words, labels):
    from sim.backend import to_host
    t0 = time.time()
    lex = M.FrameJunctionLexicon(seed, env, epochs=1)
    b = lex.b.cp_connections
    out = {"variant": lex.variant, "elemental_attr": getattr(lex, "elemental", None),
           "build": {"data": _h(to_host(b.data)), "indices": _h(to_host(b.indices)), "indptr": _h(to_host(b.indptr))}}
    lex.train(words, labels[:, None])
    out["train"] = {"W": _h(lex.W), "data": _h(to_host(lex.b.cp_connections.data))}
    dec = {}
    for w in PROBES:
        d, rn, rx = lex.decide(w)
        dec[w] = [d[0], None if rn is None else float(rn[0]), None if rx is None else float(rx[0])]
    out["decide"] = dec
    lex.set_lesion("learned_edge")
    out["learned_edge_settle"] = {"W_lesion_settled": _h(lex.W_lesion_settled),
                                  "data": _h(to_host(lex.b.cp_connections.data))}
    lex.set_lesion(None)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--pre-sha", default="db8db2a5b")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    os.environ.pop("BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL", None)       # the flag under test: UNSET
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    from research.runners import lexicon_spiking_frame_category as L
    from research.runners import lexicon_frame_junction as NEW
    from research.runners._comprehension_learned_animacy_cue_derisk import build_vocab
    OLD, old_src_sha = load_pre_change(a.pre_sha)
    tokens = NEW.load_tokens_with_pause(corpus, 8_000_000)
    assert tokens == OLD.load_tokens_with_pause(corpus, 8_000_000), "the pause-token environment differs"
    vocab, _ = build_vocab(tokens, 2000)
    env = L.FrameEnvironment(tokens, vocab + [w for w in L.HAND_NOUN_SEEDS + L.NONNOUN_SEEDS if w not in vocab])
    words, labels = L.seed_curriculum(env)
    sel = list(range(2)) + list(range(len(words) - 2, len(words)))
    w_sel, l_sel = [words[i] for i in sel], labels[sel]
    fp_old = fingerprint(OLD, a.seed, env, w_sel, l_sel)
    print("pre-change ", json.dumps(fp_old), flush=True)
    fp_new = fingerprint(NEW, a.seed, env, w_sel, l_sel)
    print("post-change", json.dumps(fp_new), flush=True)
    keys = ("build", "train", "decide", "learned_edge_settle")
    same = {k: fp_old[k] == fp_new[k] for k in keys}
    out = {"seed": a.seed, "pre_sha": a.pre_sha, "pre_change_source_sha256": old_src_sha,
           "corpus_path": corpus, "train_words": w_sel, "probes": list(PROBES),
           "pre_change": fp_old, "post_change": fp_new, "identical": same,
           "byte_identical_flag_off": bool(all(same.values()) and fp_new["variant"] == "junction"
                                           and fp_new["elemental_attr"] is False)}
    print("identical:", same, "->", out["byte_identical_flag_off"], flush=True)
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(out, open(dst, "w"), indent=1)
        print("wrote", dst)
    sys.exit(0 if out["byte_identical_flag_off"] else 1)


if __name__ == "__main__":
    main()
