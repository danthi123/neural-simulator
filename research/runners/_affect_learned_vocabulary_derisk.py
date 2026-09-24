"""AFFECT LEARNED VOCABULARY de-risk — do Hebbian word->valence synapses, learned from heard text with the innate
WARRINER seeds as the unconditioned stimulus, let the appraisal perceive HELD-OUT negative words, without colouring
neutral facts, and does lesioning the learned synapses return the old 0.0 reads? (lane A · Affect, 2026-09-24)

Mechanism: `research/runners/affect_learned_vocabulary.py` (see its docstring). Pre-registration:
`research/findings/2026-09-24-affect-learned-vocabulary-PREREG.md` (gates, thresholds, seeds, commands).

Per seed (42 43 44 100 101 102): one heard stream (fineweb-edu, first CORPUS_CHARS characters), R = 1 + N_SHUF
replicas trained on it at once (replica 0 = true seed valences; replicas 1..N_SHUF = seed valences permuted across
the innate words). Every read in the gates goes through the PRODUCTION reader (`load_reader`, the same code path
`appraise_text` uses under BRAIN_AFFECT_LEARNED_VOCAB=1), one reader per replica.

Run (local split, seed 42):
  SIM_BACKEND=numpy python -u -m research.runners._affect_learned_vocabulary_derisk --part train --seed 42 --replicas 0
  SIM_BACKEND=numpy python -u -m research.runners._affect_learned_vocabulary_derisk --part train --seed 42 --replicas 1-8
  SIM_BACKEND=numpy python -u -m research.runners._affect_learned_vocabulary_derisk --part eval --seed 42
Run (one process): --part all --seed S --replicas 0-8
Score: python -m research.runners._affect_learned_vocabulary_derisk --score research/findings/raw/_affect_learned_vocab/run1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import zlib

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import affect_learned_vocabulary as A  # noqa: E402
from research.runners._affect_distributional_tag_derisk import WARRINER  # noqa: E402

SEEDS6 = [42, 43, 44, 100, 101, 102]
N_SHUF = 8
CORPUS_CHARS = 1_000_000_000
OUT_DIR = os.path.join(_REPO, "research", "findings", "raw", "_affect_learned_vocab", "run1")
LEX_PATH = os.path.join(_REPO, "research", "runners", "_lbf_affect_tone_indep_lexicon.json")
NAMED = ["sadness", "saddened", "unhappy", "sorrow", "melancholy", "loss", "loneliness", "decay", "alone"]

# ── the pre-registered gate (see the PREREG for the reasoning and the failing outcome of each) ───────────────────
GATE = dict(
    g1_recall=0.20,        # 0.75 x the seed-7 DEV recall 0.295, rounded down to 0.05 (fixed before any eval seed)
    g1_wrong_max=0.15,     # DEV wrong-sign share 0.084, with margin
    g1_min_seeds=5,
    g2_contrast=0.15,      # 0.75 x the DEV contrast 0.241, rounded down to 0.05
    g2_shuf_mean_max=0.05,
    g2_min_seeds=5,
    g3_abs_max=0.25,       # the affect ladder's dead zone at |appraisal| <= 0.25
    g3_min_frac=0.95,
    g3_min_seeds=6,
)

# EVAL neutral fact sentences (G3). Written before any run; none contains a strong WARRINER word (checked).
FACT_EVAL = [
    "What is the capital of France?",
    "Who was Enrico Fermi?",
    "Tell me about Wolfgang Amadeus Mozart.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Paris is the largest city in France.",
    "Germany borders France, Poland and Austria.",
    "The Pacific is the largest ocean on the planet.",
    "John Adams was the second president of the United States.",
    "John von Neumann worked on computer architecture and game theory.",
    "The human heart has four chambers.",
    "Photosynthesis converts carbon dioxide and water into glucose.",
    "The Roman Empire built roads across Europe.",
    "Mount Everest is the highest mountain above sea level.",
    "The printing press was invented in the fifteenth century.",
    "Copper conducts electricity and heat.",
    "The treaty was signed in 1648 in the city of Osnabrueck.",
    "Japan is an island country in East Asia.",
    "The library opens at nine in the morning.",
    "An atom consists of a nucleus and electrons.",
    "Beethoven composed nine symphonies.",
    "The Amazon river flows through Brazil.",
    "Which planet has the most moons?",
    "How many bones are in the adult human body?",
    "The committee meets every Tuesday afternoon.",
    "Isaac Newton described the laws of motion.",
    "The train leaves the station at half past six.",
    "Spanish is spoken in most of South America.",
    "The museum holds paintings from the eighteenth century.",
    "Glaciers form where snow accumulates over many years.",
    "The bridge was completed in 1937.",
    "Marie Curie studied radioactivity.",
    "The recipe calls for flour, sugar and two eggs.",
    "Iron rusts when exposed to oxygen and moisture.",
    "The census counts the population every ten years.",
    "Chess is played on a board of sixty four squares.",
    "The Nile is a river in northeastern Africa.",
    "What year did the Berlin Wall come down?",
    "Where is the Great Barrier Reef located?",
    "The company reported its quarterly earnings on Monday.",
    "Leonardo da Vinci painted the Mona Lisa.",
]
# DEV neutral sentences (used in the seed-7 calibration only; not a gate input).
FACT_DEV = [
    "What does the cat eat?",
    "The city council approved the new budget.",
    "Einstein published the theory of relativity.",
    "The river runs past the old mill.",
    "The school year begins in September.",
    "Tell me about the history of Rome.",
    "The computer stores data on a disk.",
    "The farmer planted wheat in the spring.",
    "Mercury is the smallest planet in the solar system.",
    "The orchestra rehearsed for the concert.",
    "The ship sailed from Lisbon to Brazil.",
    "Who wrote the novel Moby Dick?",
    "The engine uses diesel fuel.",
    "The students read the chapter on cells.",
    "The capital of Italy is Rome.",
    "Bread is made from flour, water and yeast.",
    "The election was held in November.",
    "The telescope observed a distant galaxy.",
    "The map shows the northern provinces.",
    "The museum is closed on Sundays.",
]


def load_lexicon():
    with open(LEX_PATH, encoding="utf-8") as fh:
        words = json.load(fh)["words"]
    over = set(words) & set(WARRINER)
    if over:
        raise RuntimeError(f"independent lexicon overlaps WARRINER: {sorted(over)[:5]}")
    return {w: int(v) for w, v in words.items()}


def eval_half(word: str) -> bool:
    """The EVAL half of the independent lexicon (crc32 odd); the even half is the DEV half used on seed 7."""
    return zlib.crc32(word.encode("utf-8")) % 2 == 1


def corpus_spec(path=None, chars=CORPUS_CHARS):
    return ((path or os.path.join("data", "corpus", "fineweb_edu.txt"), int(chars)),)


def parse_replicas(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def eval_items(seed):
    """Every word any gate or report reads (the shuffled replicas are saved for these words only)."""
    _, held = A.seed_split(seed)
    return sorted(set(load_lexicon()) | set(NAMED) | set(held) | set(WARRINER))


def weights_path(out_dir, seed, rid, scratch_dir):
    if rid == 0:
        return os.path.join(out_dir, f"weights_s{seed}.npz")
    return os.path.join(scratch_dir, f"weights_s{seed}_r{rid}.npz")


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ── training ─────────────────────────────────────────────────────────────────────────────────────────────────────
def part_train(seed, rids, corpus, out_dir, scratch_dir, log=print):
    t0 = time.time()
    st = A.HeardStream(corpus=corpus)
    innate, held = A.seed_split(seed)
    log(f"[train s{seed} r{rids}] V={st.V} tokens={st.n_tok} innate={len(innate)} held={len(held)} "
        f"load {time.time() - t0:.0f}s")
    lav = A.LearnedAffectVocabulary(seed, st.vocab, innate, replica_ids=rids)
    x_ref = lav.calibrate_afferent()
    rows = st.presentations()
    t1 = time.time()
    lav.train(rows, log_every=1_000_000, log=log)
    train_s = time.time() - t1
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)
    meta = []
    for k, rid in enumerate(rids):
        p = weights_path(out_dir, seed, rid, scratch_dir)
        u = lav.u[k]
        lav.save(p, replica=k, meta={"seed": seed, "replica_id": rid, "corpus": list(map(list, corpus))},
                 only_words=None if rid == 0 else eval_items(seed))
        meta.append({"replica_id": rid, "path": os.path.relpath(p, _REPO) if p.startswith(_REPO) else p,
                     "sha256": _sha_file(p), "floor_share": float(np.mean(u <= 0.0)),
                     "n_nonzero_words": int(np.sum(np.abs(u).sum(axis=1) > 0)),
                     "theta_final": lav.theta[k].tolist()})
    info = {"seed": seed, "replica_ids": rids, "V": st.V, "n_tokens": st.n_tok, "n_presentations": int(len(rows)),
            "x_ref": x_ref, "train_seconds": round(train_s, 1), "per_presentation_ms": 1000 * train_s / len(rows),
            "vocab_sha": hashlib.sha256("\n".join(st.vocab).encode()).hexdigest(),
            "innate_words": sorted(innate), "held_seed_words": sorted(held), "replicas": meta,
            "corpus": [list(c) for c in corpus],
            "constants": {k: getattr(A, k) for k in ("MIN_HEARD", "CHUNK", "SEED_MARGIN", "SEED_FRAC", "N_CAT",
                                                      "N_FSI", "T_CS", "T_ON", "T_READ", "I_AFF", "W_US", "TAU_THETA", "TAU_SCALE", "WARMUP", "ETA_MIN",
                                                      "N0", "G", "READ_GAIN", "R_REF", "MIN_RATE", "V_MIN", "U_DEP", "TAU_REC")}}
    tag = "-".join(map(str, rids))
    with open(os.path.join(out_dir if 0 in rids else scratch_dir, f"train_s{seed}_r{tag}.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    log(f"[train s{seed} r{rids}] done in {time.time() - t0:.0f}s ({info['per_presentation_ms']:.2f} ms/presentation)")
    return info


# ── evaluation (every read through the production reader) ──────────────────────────────────────────────────────
def _appraise(text):
    from research.runners import affect_production_organ as AO
    return AO.appraise_text(text)


def _core(d):
    return {k: d[k] for k in ("valence", "arousal", "n_hits", "words")}


def part_eval(seed, out_dir, scratch_dir, n_shuf=N_SHUF, log=print):
    lex = load_lexicon()
    innate, held = A.seed_split(seed)
    readers = {}
    for rid in range(0, n_shuf + 1):
        p = weights_path(out_dir, seed, rid, scratch_dir)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        readers[rid] = A.load_reader(p, seed)
    r0 = readers[0]
    heard = set(r0.vocab)
    neg_eval = sorted(w for w, v in lex.items() if v < 0 and eval_half(w))
    pos_eval = sorted(w for w, v in lex.items() if v > 0 and eval_half(w))
    # a word with no learned synapse (never heard >= MIN_HEARD or all-zero) is still a legitimate test item: it reads 0
    words = sorted(set(neg_eval) | set(pos_eval) | set(NAMED) | set(held)
                   | {w for w, (v, a) in WARRINER.items() if abs(v - 5) < 2})
    reads = {rid: {w: rd.read(w) for w in words} for rid, rd in readers.items()}

    def stats(rd):
        def frac(ws, cond):
            return float(np.mean([cond(rd[w]) for w in ws])) if ws else None
        return {"neg_recall": frac(neg_eval, lambda v: v < 0), "neg_wrong": frac(neg_eval, lambda v: v > 0),
                "pos_recall": frac(pos_eval, lambda v: v > 0), "pos_wrong": frac(pos_eval, lambda v: v < 0),
                "neg_on_pos": frac(pos_eval, lambda v: v < 0),
                "contrast_D": frac(neg_eval, lambda v: v < 0) - frac(pos_eval, lambda v: v < 0)}

    per_rep = {rid: stats(reads[rid]) for rid in readers}
    hs = [w for w in sorted(held)]
    hs_dec = [w for w in hs if reads[0][w] != 0]
    held_seed = {"n": len(hs), "n_decided": len(hs_dec),
                 "sign_acc_decided": (float(np.mean([np.sign(reads[0][w]) == np.sign(held[w]) for w in hs_dec]))
                                      if hs_dec else None),
                 "reads": {w: [held[w], reads[0][w]] for w in hs}}
    neutral62 = sorted(w for w, (v, a) in WARRINER.items() if abs(v - 5) < 2)
    neutral_diag = {"n": len(neutral62), "nonzero_frac": float(np.mean([reads[0][w] != 0 for w in neutral62])),
                    "reads": {w: reads[0][w] for w in neutral62 if reads[0][w] != 0}}
    # sentence level through appraise_text: flag off / intact / lesion
    os.environ["BRAIN_CHAT_SEED"] = str(seed)
    os.environ["BRAIN_AFFECT_LEARNED_VOCAB_PATH"] = weights_path(out_dir, seed, 0, scratch_dir)
    A._READER.clear()
    probes = {"fact_eval": FACT_EVAL, "fact_dev": FACT_DEV,
              "neg_word_probe": [f"It was {w}." for w in neg_eval],
              "named_probe": [f"It was {w}." for w in NAMED]}
    sent = {}
    for arm, env in (("off", {}), ("intact", {"BRAIN_AFFECT_LEARNED_VOCAB": "1"}),
                     ("lesion", {"BRAIN_AFFECT_LEARNED_VOCAB": "1", "BRAIN_AFFECT_LEARNED_VOCAB_LESION": "1"})):
        for k in ("BRAIN_AFFECT_LEARNED_VOCAB", "BRAIN_AFFECT_LEARNED_VOCAB_LESION"):
            os.environ.pop(k, None)
        os.environ.update(env)
        sent[arm] = {name: [_appraise(t) for t in texts] for name, texts in probes.items()}
    for k in ("BRAIN_AFFECT_LEARNED_VOCAB", "BRAIN_AFFECT_LEARNED_VOCAB_LESION"):
        os.environ.pop(k, None)
    reader_loaded = A.get_reader(seed) is not None
    fact = sent["intact"]["fact_eval"]
    g3 = {"off_all_zero": all(d["valence"] == 0.0 and d["n_hits"] == 0 for d in sent["off"]["fact_eval"]),
          "frac_within": float(np.mean([abs(d["valence"]) <= GATE["g3_abs_max"] for d in fact])),
          "mean_abs": float(np.mean([abs(d["valence"]) for d in fact])),
          "max_abs": float(np.max([abs(d["valence"]) for d in fact])),
          "flagged": [[t, d["valence"], d.get("learned_words")] for t, d in zip(FACT_EVAL, fact) if d["n_hits"]]}
    lesion_identical = all(_core(a) == _core(b) for name in probes
                           for a, b in zip(sent["lesion"][name], sent["off"][name]))
    lesion_words_zero = all(v == 0.0 for v in _lesion_reads(readers[0], neg_eval))
    neg_probe_intact = float(np.mean([d["valence"] < 0 for d in sent["intact"]["neg_word_probe"]]))
    neg_probe_lesion = float(np.mean([d["valence"] < 0 for d in sent["lesion"]["neg_word_probe"]]))
    from tools.lab import attributable_to
    attr = attributable_to("share of eval-negative probes read negative: learned synapses vs learned_edge lesion",
                           neg_probe_intact, neg_probe_lesion)
    shas = set()
    for d_ in (out_dir, scratch_dir):
        for tp in glob.glob(os.path.join(d_, f"train_s{seed}_r*.json")):
            with open(tp) as fh:
                shas.add(json.load(fh).get("vocab_sha"))
    out = {"seed": seed, "n_shuf": n_shuf, "reader_loaded": reader_loaded, "vocab_shas": sorted(shas),
           "n_neg_eval": len(neg_eval), "n_pos_eval": len(pos_eval),
           "n_neg_eval_heard": sum(w in heard for w in neg_eval),
           "replica_stats": {str(k): v for k, v in per_rep.items()},
           "true": per_rep[0],
           "shuffled": {"contrast_D": [per_rep[r]["contrast_D"] for r in range(1, n_shuf + 1)],
                        "neg_recall": [per_rep[r]["neg_recall"] for r in range(1, n_shuf + 1)]},
           "held_seed": held_seed, "neutral62_diag": neutral_diag, "g3": g3,
           "g4": {"lesion_appraisal_identical_to_off": lesion_identical, "lesion_neg_words_read_zero": lesion_words_zero,
                  "neg_probe_negative_intact": neg_probe_intact, "neg_probe_negative_lesion": neg_probe_lesion,
                  "attributable_to_learned_edge": attr},
           "named": {w: reads[0][w] for w in NAMED},
           "named_sentences": {t: [d["valence"], d.get("learned_words")]
                               for t, d in zip(probes["named_probe"], sent["intact"]["named_probe"])},
           "fact_dev_intact": [[t, d["valence"], d.get("learned_words")] for t, d in zip(FACT_DEV, sent["intact"]["fact_dev"])],
           "neg_eval_reads": {w: reads[0][w] for w in neg_eval},
           "pos_eval_reads": {w: reads[0][w] for w in pos_eval},
           "weights_sha256": _sha_file(weights_path(out_dir, seed, 0, scratch_dir))}
    return out


def _lesion_reads(reader, words):
    reader.set_lesion("learned_edge")
    try:
        return [reader.read(w) for w in words]
    finally:
        reader.set_lesion(None)


# ── scoring (6 seeds) ───────────────────────────────────────────────────────────────────────────────────────────
def score(run_dir):
    rows = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "eval_s*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        rows[d["seed"]] = d
    missing = [s for s in SEEDS6 if s not in rows]
    pre = []
    pre.append(("all 6 seeds present", not missing))
    shas = {sh for d in rows.values() for sh in d.get("vocab_shas", [])}
    pre.append(("one heard vocabulary on every seed and replica block", len(shas) == 1))
    for s, d in rows.items():
        pre.append((f"s{s} reader loaded", bool(d["reader_loaded"])))
        pre.append((f"s{s} fact sentences read 0 with the flag off", bool(d["g3"]["off_all_zero"])))
        pre.append((f"s{s} >= 20 heard eval negatives", d["n_neg_eval_heard"] >= 20))
        pre.append((f"s{s} INTEGRITY lesion appraisal == off", bool(d["g4"]["lesion_appraisal_identical_to_off"])))
        pre.append((f"s{s} INTEGRITY lesion reads 0", bool(d["g4"]["lesion_neg_words_read_zero"])))
    per = {}
    for s, d in rows.items():
        t = d["true"]
        sh = d["shuffled"]["contrast_D"]
        g1 = (t["neg_recall"] >= GATE["g1_recall"]) and (t["neg_wrong"] <= GATE["g1_wrong_max"])
        g2 = (t["contrast_D"] >= GATE["g2_contrast"]) and (t["contrast_D"] > max(sh)) \
            and (float(np.mean(sh)) <= GATE["g2_shuf_mean_max"])
        g3 = d["g3"]["frac_within"] >= GATE["g3_min_frac"]
        g4 = d["g4"]["neg_probe_negative_lesion"] == 0.0 and d["g4"]["neg_probe_negative_intact"] >= GATE["g1_recall"]
        per[s] = {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "neg_recall": t["neg_recall"], "neg_wrong": t["neg_wrong"],
                  "contrast_D": t["contrast_D"], "shuf_D_max": max(sh), "shuf_D_mean": float(np.mean(sh)),
                  "g3_frac_within": d["g3"]["frac_within"], "g3_max_abs": d["g3"]["max_abs"]}
    n = lambda k: sum(1 for v in per.values() if v[k])  # noqa: E731
    gates = {"G1": n("g1") >= GATE["g1_min_seeds"], "G2": n("g2") >= GATE["g2_min_seeds"],
             "G3": n("g3") >= GATE["g3_min_seeds"], "G4": n("g4") >= 6}
    pre_ok = all(ok for _, ok in pre)
    verdict = "UNDEFINED" if not pre_ok else ("GO" if all(gates.values()) else "NO-GO")
    return {"verdict": verdict, "gate": GATE, "preconditions": [{"name": k, "ok": v} for k, v in pre],
            "gates": gates, "per_seed": {str(k): v for k, v in per.items()}, "missing_seeds": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("train", "eval", "all"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--replicas", default="0-%d" % N_SHUF)
    ap.add_argument("--corpus-path", default=None)
    ap.add_argument("--corpus-chars", type=int, default=CORPUS_CHARS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--scratch-dir", default=None)
    ap.add_argument("--score", default=None)
    ap.add_argument("--n-shuf", type=int, default=N_SHUF)
    a = ap.parse_args()
    if a.score:
        v = score(a.score)
        with open(os.path.join(a.score, "verdict.json"), "w") as fh:
            json.dump(v, fh, indent=1)
        print(json.dumps({k: v[k] for k in ("verdict", "gates", "missing_seeds")}, indent=1))
        return
    scratch = a.scratch_dir or os.path.join(a.out_dir, "shuf")
    corpus = corpus_spec(a.corpus_path, a.corpus_chars)
    if a.part in ("train", "all"):
        part_train(a.seed, parse_replicas(a.replicas), corpus, a.out_dir, scratch, log=lambda m: print(m, flush=True))
    if a.part in ("eval", "all"):
        out = part_eval(a.seed, a.out_dir, scratch, n_shuf=a.n_shuf, log=lambda m: print(m, flush=True))
        p = os.path.join(a.out_dir, f"eval_s{a.seed}.json")
        with open(p, "w") as fh:
            json.dump(out, fh, indent=1)
        print(json.dumps({"seed": a.seed, "true": out["true"], "shuffled": out["shuffled"], "g3": {
            k: out["g3"][k] for k in ("frac_within", "mean_abs", "max_abs")}, "g4": out["g4"],
            "named": out["named"]}, indent=1))


if __name__ == "__main__":
    main()
