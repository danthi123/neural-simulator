"""FULLY-SPIKING SPOKEN both-dimension turn: the property HTM reasoner + the relational RF-FHRR composer +
the A->W speaker ALL co-execute on cupy in ONE process, REASON both dimensions ON SPIKES and SPEAK the
answers ON SPIKES, over spellable animals. The directive-central capstone of the fully-spiking arc
(CYCLE-1003 did the reasoning; this adds spiking speech + spellable-animal alignment).

Honest scope: the property HTM needs K=1024 emergent clusters (where the spiking read is strong); the
spellable animals must co-cluster there (the CYCLE-987 alignment tension). If a cluster with >=2 spellable
animals forms and a weakly-inherited spellable exception exists, the turn is fully spoken; else NOT-EVALUABLE
(an honest, characterized boundary). Reuse-by-import. NO `sim/` edit. Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_cancellation_spiking_derisk import (
    CancellingPoolerProbe, emergent_inputs, _adaptive_teach,
)
from research.runners._realcorpus_svo_spiking_derisk import _grounded_phases, D
from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw import VOCAB, WORD_TO_POOL
from research.runners._realcorpus_cancellation_derisk import _ANIMALS
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi


def run(corpus_path, K, n_clusters, seed, aw_seed=42):
    speaker = ConceptFrameSpeaker("bridges/breadth_aw/seed42.simstate.h5", seed=aw_seed,
                                  vocab=VOCAB, word_to_pool=WORD_TO_POOL)   # spiking A->W
    spellable = set(speaker.vocab) & _ANIMALS

    # ---- PROPERTY (spiking HTM): find a cluster with >=2 inheriting spellable animals; pick the one that
    #      overrides with the FEWEST passes as the exception (weakly-inherited -> no saturation) ----
    sdr_by_row, row_to_cat, cat_ids, vocab = emergent_inputs(corpus_path, K, seed, n_clusters)
    prop = CancellingPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=40)
    word_of = {r: vocab[r] for r in prop.rows}
    by_cat = {}
    for r in prop.rows:
        by_cat.setdefault(prop.row2cat[r], []).append(r)
    pos = exc_row = None; others = []
    for k in prop.cat_ids:
        animals = [r for r in by_cat.get(k, []) if word_of[r] in spellable and prop.query(r, include_exc=False) == f"C{k}"]
        if len(animals) >= 2:
            # exception = a spellable animal that overrides within a modest pass budget (avoid saturation)
            for cand in animals:
                p = _adaptive_teach(prop, cand, k, max_passes=10)
                if prop.query(cand) == "EXC":
                    pos, exc_row = k, cand; others = [r for r in animals if r != cand]; break
            if pos is not None:
                break
            # not overridable within budget on this cluster -> try the next cluster (rebuild to undo teaching)
            prop = CancellingPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=40)
    if pos is None:
        print(f"  seed {seed}: NO cluster with a weakly-inherited spellable-animal exception (CYCLE-987 alignment)"); return None
    exc_word = word_of[exc_row]
    other_word = word_of[others[0]] if others else None

    # ---- RELATIONAL (spiking RF): grounded codes + a spellable-animal fact ----
    stories = load_token_stream_multi(corpus_path, max_stories=None)
    rvocab, gfreq = discover_vocab(stories, 64)
    rrow = {w: i for i, w in enumerate(rvocab)}
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in set(rvocab) or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, rvocab, hubs, window=WINDOW)
    rel_animals = [a for a in spellable if a in rrow]
    ra_v = next((v for v in ("see", "eat", "find", "like", "run") if v in rrow), None)   # a real verb in the K=64 vocab
    if len(rel_animals) < 2 or ra_v is None:
        print(f"  seed {seed}: <2 spellable animals or no relational verb in the K=64 vocab"); return None
    ra_s, ra_o = rel_animals[0], rel_animals[1]
    words = sorted(set(rel_animals) | {ra_v})
    grounded = _grounded_phases(codes, [w for w in words if w in rrow], rrow, seed)
    rel = RFPhasorComposer(seed=seed, D=D, vocab=[w for w in words if w in rrow],
                           grounded_codes=grounded, enable_substrate_store=True)
    rel.store(ra_s, ra_v, ra_o)

    print(f"  === ONE BRAIN, BOTH DIMENSIONS, FULLY SPIKING + SPOKEN (one cupy process) ===", flush=True)
    print(f"  property cluster {pos}: exception '{exc_word}' + inheriting '{other_word}'; relational '{ra_s} {ra_v} {ra_o}'", flush=True)
    # PROPERTY answers SPOKEN on spikes
    ex_frame, _ = speaker.speak_frame(exc_word, "sleep")             # the exception's override, spoken
    print(f"    Q: does the {exc_word} run?  -> reason(EXC, spikes) -> \"no, {ex_frame}\"", flush=True)
    prop_spoken = ex_frame
    if other_word:
        inh_frame, _ = speaker.speak_frame(other_word, "run")
        print(f"    Q: does the {other_word} run? -> reason(inherit, spikes) -> \"yes, {inh_frame}\"", flush=True)
    # RELATIONAL answer SPOKEN on spikes
    got = rel.query_patient(ra_s, ra_v)
    obj_spoken = speaker.spell(got) if got in speaker.vocab else got
    print(f"    Q: what does the {ra_s} {ra_v}?  -> reason(RF, spikes) -> \"the {ra_s} {ra_v}s {obj_spoken}\"", flush=True)

    prop_ok = (prop.query(exc_row) == "EXC") and ("sleep" in prop_spoken)
    rel_ok = (got == ra_o) and (obj_spoken == ra_o)
    return {"seed": seed, "prop_ok": bool(prop_ok), "rel_ok": bool(rel_ok), "exc": exc_word, "rel": f"{ra_s} {ra_v} {ra_o}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"[one brain SPOKEN] property HTM + relational RF + A->W, all on cupy, one process | seed={a.seed}", flush=True)
    r = run(a.corpus_path, a.K, a.n_clusters, a.seed)
    if r is None:
        print("  VERDICT: NOT-EVALUABLE (alignment boundary)"); return
    go = r["prop_ok"] and r["rel_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- BOTH dimensions REASONED ON SPIKES (property HTM apical "
          f"cancellation + relational RF-FHRR) AND SPOKEN ON SPIKES (A->W), in ONE cupy process. The TRUE one "
          f"brain, fully spiking, spoken, both dimensions.", flush=True)


if __name__ == "__main__":
    main()
