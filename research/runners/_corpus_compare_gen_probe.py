"""CPU numpy CORPUS-COMPARISON generalization probe -- the cheap de-risk specified in
`research/findings/raw/_corpus_richness_gen_lever_scoping.md` section (c).

THE QUESTION: does a RICHER corpus (Simple-English-Wikipedia sample, `data/corpus/wikitext.txt`) raise the
INTRINSIC numpy category-structure ceiling above TinyStories, at MATCHED corpus size + MATCHED a-priori labels?
This isolates the corpus's structure ceiling from the spiking-substrate + scale confounds, with ZERO GPU.

WHY THIS IS THE GATE (scoping (c) + the C0 control `_curriculum_gen_C0_substrate_vs_scale.json`): the C0 control
already proved the numpy substrate carries the structure (numpy-320 Pearson +0.215, ratio 18x chance) and the
spiking read-out then loses ~0.145 of it. The corpus-content hypothesis has been REFUTED THREE TIMES
(yardstick-swap 0.153->0.167; content-vocab filter ->0.125; the C0 decomposition: scale ~0.30 + spiking ~0.145).
So a richer corpus can only matter as an INTRINSIC numpy-ceiling lever -- and the apples-to-apples way to test
that, free, is to run the VALIDATED numpy exact-count pipeline on a SIZE-MATCHED sample of each corpus, scoring
the SAME a-priori taxonomy labels. If wikitext numpy-Pearson does NOT beat TinyStories numpy-Pearson, the corpus
is NOT the lever and a multi-hour GPU run is saved.

REUSE-BY-IMPORT (the C0 template, byte-for-byte where possible): the pipeline body is `run_seed_numpy_exact`
from `_curriculum_gen_C0_substrate_vs_scale.py` (M[a,b]+=1 exact count over a WM window=2, then
`double_center(log1p(M*100))`, then `heldout_generalization` + `_pearson_vs_Strue`). The taxonomies are
TAXONOMY_8x8 (64 words / 8 cat -- the +0.513 anchor) and TAXONOMY_40x8 (320 words / 40 cat -- the +0.215
anchor). N_HUB=500 (the validated value; each corpus uses ITS OWN top-500 frequent context words as hubs).

THE THREE DESIGN POINTS that make this apples-to-apples (all documented; NO `sim/` edit, NO GPU, numpy host):
  (1) SIZE-MATCH at the token-stream level. wikitext (~620K `[a-z]+` tokens) is the smaller corpus, so BOTH
      corpora are truncated to the common token budget (= min of the two token counts). This isolates corpus
      QUALITY (within-category co-occurrence density) from corpus SIZE (a bigger corpus trivially has more
      f>=20 words). [scoping anti-cheat 8 + (c) step 1.]
  (2) RE-CHUNK both flattened token streams into equal-length PSEUDO-STORIES (PSEUDO_STORY_LEN tokens each), so
      the WM-window semantics are IDENTICAL across corpora and neither gets a doc-boundary advantage. wikitext
      has 0 `<|endoftext|>` delimiters (it would otherwise stream as one giant document); re-chunking both the
      same way removes that confound. The window=2 co-occurrence is window-LOCAL, so the only effect of a chunk
      boundary is that 2 pairs straddling it are dropped -- identical treatment for both corpora.
  (3) INTERSECTION VOCAB. The scored words = the taxonomy words present in BOTH size-matched corpora at
      freq >= MIN_FREQ (20). The a-priori category labels (cat_ids) stay the SAME taxonomy categories; only the
      surviving words are scored (apples-to-apples). If the intersection is small (< MIN_WORDS_POWERED words OR
      < MIN_CATS_POWERED non-empty categories) the probe is flagged UNDER-POWERED loudly in the verdict.

ANTI-CHEATS (scoping (d) -- enforced + reported):
  - S_true is the INDEPENDENT a-priori taxonomy block matrix, NEVER corpus-derived (`run_seed_numpy_exact`
    builds it from `cat_ids` only).
  - Size-matched (isolate quality, not size).
  - Pearson(cos, S_true) is the chance-INDEPENDENT tell (gen + ratio_vs_chance reported alongside).
  - DERANGEMENT control per corpus: shuffle the category labels -> Pearson must collapse to ~0 (proves the
    number reflects real structure, not an artifact). Run on BOTH corpora at BOTH taxonomy scales.

DETERMINISM: the exact-count pipeline is deterministic given (corpus, vocab, hubs, story_order). The only
RNG is the seeded story permutation in `run_seed_numpy_exact` (deterministic per seed) and the seeded
label-derangement. We run SEEDS=(42,43,44) for the main arms (story-order is the lone randomness; reporting the
spread documents it is tiny) and a single seed-42 derangement (it only needs to show collapse).

Run:  SIM_BACKEND=numpy python -u -m research.runners._corpus_compare_gen_probe
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse the VALIDATED numpy exact-count pipeline body VERBATIM (the C0 template) + the gen metrics + taxonomies.
from research.runners._curriculum_gen_C0_substrate_vs_scale import run_seed_numpy_exact  # noqa: E402
from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim,
    _pearson_vs_Strue,
    heldout_generalization,
)
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
)
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8  # noqa: E402
from research.runners.corpus_stream import iter_stories  # noqa: E402

SEEDS = (42, 43, 44)
N_HUB = 500
MIN_FREQ = 20                 # the intersection frequency floor (scoping (c) step 2)
PSEUDO_STORY_LEN = 400        # re-chunk granularity (design point 2); window-local co-occ is chunk-size-robust
MIN_WORDS_POWERED = 40        # < this many surviving words -> flag UNDER-POWERED (scoping (c) step 2)
MIN_CATS_POWERED = 6          # < this many non-empty categories -> flag UNDER-POWERED

CORPORA = {
    "tinystories": os.path.join(_REPO, "data", "corpus", "tinystories.txt"),
    "wikitext":    os.path.join(_REPO, "data", "corpus", "wikitext.txt"),
}


def flat_tokens(path):
    """Flatten a corpus to ONE ordered token list via the project tokenizer (re.findall r'[a-z]+', through the
    streaming iter_stories so the raw file is never read whole). Concatenates across any document delimiters --
    we re-chunk uniformly afterwards (design point 2), so the corpus's OWN doc structure is intentionally not
    used (wikitext has none anyway)."""
    toks = []
    for story_toks in iter_stories(path, skip_empty=False):
        toks.extend(story_toks)
    return toks


def chunk_into_pseudo_stories(tokens, story_len=PSEUDO_STORY_LEN):
    """Split a flat token list into equal-length pseudo-stories (the LAST may be shorter). Identical treatment
    for every corpus so the WM-window behavior in run_seed_numpy_exact is matched."""
    return [tokens[i:i + story_len] for i in range(0, len(tokens), story_len)]


def intersection_vocab(taxonomy, freq_by_corpus, min_freq=MIN_FREQ):
    """Keep only taxonomy words present in ALL corpora at freq >= min_freq. Returns (vocab, cat_ids, cat_names,
    report). cat_ids keep the ORIGINAL taxonomy category ids (a-priori labels unchanged; only words are dropped).
    A category that loses ALL its words is removed from the scored set (its id would have no members)."""
    full_vocab, full_cat_ids, full_cat_names = taxonomy_to_vocab_categories(taxonomy)
    kept_words, kept_cat_ids = [], []
    dropped = []
    for w, cid in zip(full_vocab, full_cat_ids):
        present_all = all(freq_by_corpus[c].get(w, 0) >= min_freq for c in freq_by_corpus)
        if present_all:
            kept_words.append(w)
            kept_cat_ids.append(int(cid))
        else:
            dropped.append(w)
    # Re-index the surviving category ids to a dense 0..K-1 (so chance = 1/#non-empty-cats is correct) while
    # preserving which words share a category (the a-priori grouping is untouched -- only empty cats removed).
    uniq = sorted(set(kept_cat_ids))
    remap = {old: new for new, old in enumerate(uniq)}
    dense_cat_ids = np.asarray([remap[c] for c in kept_cat_ids], dtype=int)
    surviving_cat_names = [full_cat_names[old] for old in uniq]
    # per-category surviving counts (for the under-power report)
    per_cat = Counter(dense_cat_ids.tolist())
    report = {
        "n_words_full": len(full_vocab),
        "n_words_kept": len(kept_words),
        "n_words_dropped": len(dropped),
        "n_categories_full": len(full_cat_names),
        "n_categories_nonempty": len(surviving_cat_names),
        "min_freq": min_freq,
        "surviving_categories": surviving_cat_names,
        "per_category_surviving_count": {surviving_cat_names[c]: int(per_cat[c]) for c in range(len(surviving_cat_names))},
        "dropped_words": sorted(dropped),
    }
    return kept_words, dense_cat_ids, surviving_cat_names, report


def derangement_pearson(seed, stories, vocab, cat_ids, n_hub):
    """Anti-cheat: shuffle the a-priori category labels, re-score Pearson(cos, S_true). The CODES are unchanged
    (same corpus co-occurrence); only S_true is deranged -> Pearson must collapse to ~0. We reuse the exact-count
    code build inline (cannot call run_seed_numpy_exact because it builds S_true from the TRUE cat_ids)."""
    # Build the code EXACTLY as run_seed_numpy_exact does (so the only difference is the deranged labels).
    from research.runners._phaseB_online_stream_cortex_derisk import WINDOW, EMA_ALPHA, double_center
    from research.runners.option_c_stageB_fair_test import STOPLIST
    rng = np.random.RandomState(seed)
    targets = list(vocab)
    target_set = set(targets)
    Nt = len(targets)
    gfreq = Counter()
    for toks in stories:
        gfreq.update(toks)
    hubs = [w for w, _ in gfreq.most_common() if w not in STOPLIST and w not in target_set][:n_hub]
    hub_idx = {w: i for i, w in enumerate(hubs)}
    keep = target_set | set(hubs)
    tgt_row = {w: i for i, w in enumerate(targets)}
    M = np.zeros((Nt, len(hubs)), dtype=np.float64)
    story_order = rng.permutation(len(stories))
    for si in story_order:
        kept = [t for t in stories[si] if t in keep]
        for c in range(len(kept)):
            w = kept[c]
            lo, hi = max(0, c - WINDOW), min(len(kept), c + WINDOW + 1)
            ctx = set(kept[lo:hi]) - {w}
            if w in target_set:
                for u in ctx:
                    if u in hub_idx:
                        M[tgt_row[w], hub_idx[u]] += 1.0
    code = double_center(np.log1p(M * 100.0))
    cos = _cos_sim(code)
    der = rng.permutation(np.asarray(cat_ids))
    S_der = (der[:, None] == der[None, :]).astype(np.float64)
    return float(_pearson_vs_Strue(cos, S_der))


def run_corpus_arm(corpus_name, stories, vocab, cat_ids, cat_names, n_hub):
    chance = 1.0 / len(cat_names)
    print(f"\n  [{corpus_name}] {len(vocab)} concepts x {len(cat_names)} non-empty categories | n_hub={n_hub} "
          f"| chance {chance:.4f}", flush=True)
    rows = [run_seed_numpy_exact(s, stories, vocab, cat_ids, n_hub) for s in SEEDS]
    for r in rows:
        print(f"      seed {r['seed']}: gen {r['gen']:.3f} (ch {r['chance']:.4f}, ratio {r['ratio_vs_chance']:.2f}x) "
              f"| Pearson(cos,S_true) {r['pearson_cos_Strue']:+.4f} | {r['n_updates']} exact-count updates",
              flush=True)
    der = derangement_pearson(42, stories, vocab, cat_ids, n_hub)
    print(f"      derangement(seed42): Pearson {der:+.4f}  (anti-cheat: must be ~0)", flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    def sd(k):
        return float(np.std([r[k] for r in rows]))
    summ = {
        "corpus": corpus_name,
        "n_concepts": len(vocab),
        "n_categories": len(cat_names),
        "n_hub": n_hub,
        "chance": chance,
        "gen_mean": m("gen"),
        "gen_std": sd("gen"),
        "ratio_vs_chance_mean": m("ratio_vs_chance"),
        "pearson_cos_Strue_mean": m("pearson_cos_Strue"),
        "pearson_cos_Strue_std": sd("pearson_cos_Strue"),
        "derangement_pearson_seed42": der,
        "per_seed": rows,
    }
    print(f"    MEAN ({len(SEEDS)} seeds): gen {summ['gen_mean']:.3f}+-{summ['gen_std']:.3f} "
          f"(ratio {summ['ratio_vs_chance_mean']:.2f}x) | "
          f"Pearson {summ['pearson_cos_Strue_mean']:+.4f}+-{summ['pearson_cos_Strue_std']:.4f}", flush=True)
    return summ


def run_taxonomy(label, taxonomy, stories_by_corpus, freq_by_corpus):
    """Run BOTH corpora on one taxonomy at the intersection vocab + report the comparison."""
    vocab, cat_ids, cat_names, isect = intersection_vocab(taxonomy, freq_by_corpus)
    print("\n" + "=" * 104, flush=True)
    print(f"  TAXONOMY: {label}  -- intersection vocab @ freq>={MIN_FREQ} in BOTH size-matched corpora", flush=True)
    print("=" * 104, flush=True)
    print(f"  intersection: {isect['n_words_kept']}/{isect['n_words_full']} words survive; "
          f"{isect['n_categories_nonempty']}/{isect['n_categories_full']} categories non-empty", flush=True)
    underpowered = (isect["n_words_kept"] < MIN_WORDS_POWERED) or (isect["n_categories_nonempty"] < MIN_CATS_POWERED)
    if underpowered:
        print(f"  *** UNDER-POWERED FLAG: only {isect['n_words_kept']} words / "
              f"{isect['n_categories_nonempty']} categories survive the intersection "
              f"(thresholds: >={MIN_WORDS_POWERED} words, >={MIN_CATS_POWERED} categories). "
              f"Treat the comparison with caution. ***", flush=True)
    if isect["n_words_kept"] < 8 or isect["n_categories_nonempty"] < 2:
        print("  intersection too small to score (need >=2 categories, >=8 words). SKIPPING this taxonomy.",
              flush=True)
        return {"label": label, "intersection": isect, "underpowered": True, "skipped": True}

    arms = {}
    for cname in ("tinystories", "wikitext"):
        arms[cname] = run_corpus_arm(cname, stories_by_corpus[cname], vocab, cat_ids, cat_names, N_HUB)

    ts_p = arms["tinystories"]["pearson_cos_Strue_mean"]
    wk_p = arms["wikitext"]["pearson_cos_Strue_mean"]
    delta = wk_p - ts_p
    print("\n  ---- COMPARISON (the ONLY varied axis is the corpus) ----", flush=True)
    print(f"    TinyStories Pearson {ts_p:+.4f}   vs   wikitext Pearson {wk_p:+.4f}   "
          f"=>  delta (wiki - tiny) = {delta:+.4f}", flush=True)
    material = abs(delta) >= 0.03   # "material" lift threshold (a few % of the validated +0.40-0.51 band)
    if delta >= 0.03:
        call = "wikitext MATERIALLY HIGHER -- corpus richness IS a category-structure lever on this sample."
    elif delta <= -0.03:
        call = "TinyStories HIGHER -- the thin wiki sample does NOT lift the numpy ceiling here."
    else:
        call = "ROUGHLY EQUAL (|delta| < 0.03) -- no demonstrated corpus lift on this sample."
    print(f"    => {call}", flush=True)
    return {
        "label": label,
        "intersection": isect,
        "underpowered": bool(underpowered),
        "skipped": False,
        "tinystories": arms["tinystories"],
        "wikitext": arms["wikitext"],
        "tinystories_pearson": ts_p,
        "wikitext_pearson": wk_p,
        "delta_wiki_minus_tiny": delta,
        "material_lift": bool(material and delta > 0),
        "call": call,
    }


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print("=" * 104, flush=True)
    print("  CORPUS-COMPARISON generalization probe (CPU numpy exact-count) -- does a richer corpus raise the", flush=True)
    print("  INTRINSIC numpy category-structure ceiling above TinyStories, at MATCHED size + MATCHED labels?", flush=True)
    print("=" * 104, flush=True)

    # 1) FLATTEN both corpora to ordered token lists (project tokenizer).
    flat = {}
    for name, path in CORPORA.items():
        flat[name] = flat_tokens(path)
        print(f"  loaded {name:11s}: {len(flat[name]):>9,} tokens (re[a-z]+)  [{path}]", flush=True)

    # 2) SIZE-MATCH: truncate BOTH to the common token budget (= the smaller corpus's token count).
    budget = min(len(flat[name]) for name in flat)
    print(f"\n  SIZE-MATCH budget = {budget:,} tokens (min of the two) -- both corpora truncated to this.", flush=True)
    matched = {name: flat[name][:budget] for name in flat}

    # 3) RE-CHUNK both into equal-length pseudo-stories (identical WM-window treatment; design point 2).
    stories_by_corpus = {name: chunk_into_pseudo_stories(matched[name]) for name in matched}
    for name in stories_by_corpus:
        n_st = len(stories_by_corpus[name])
        print(f"    {name:11s}: {n_st:,} pseudo-stories of <= {PSEUDO_STORY_LEN} tokens", flush=True)

    # 4) per-corpus frequency over the SIZE-MATCHED tokens (for the intersection vocab).
    freq_by_corpus = {name: Counter(matched[name]) for name in matched}

    # 5) run BOTH taxonomies (64/8 and 320/40) on the intersection vocab.
    res64 = run_taxonomy("TAXONOMY_8x8 (64 words / 8 cat -- the +0.513 anchor)", TAXONOMY_8x8,
                         stories_by_corpus, freq_by_corpus)
    res320 = run_taxonomy("TAXONOMY_40x8 (320 words / 40 cat -- the +0.215 anchor)", TAXONOMY_40x8,
                          stories_by_corpus, freq_by_corpus)

    # ---- TOP-LEVEL VERDICT ----
    print("\n" + "=" * 104, flush=True)
    print("  VERDICT", flush=True)
    print("=" * 104, flush=True)
    lifts = []
    for res in (res64, res320):
        if res.get("skipped"):
            print(f"  [{res['label']}] SKIPPED (intersection too small).", flush=True)
            continue
        d = res["delta_wiki_minus_tiny"]
        up = "  [UNDER-POWERED]" if res["underpowered"] else ""
        print(f"  [{res['label']}]{up}", flush=True)
        print(f"    intersection {res['intersection']['n_words_kept']} words / "
              f"{res['intersection']['n_categories_nonempty']} cats | "
              f"TinyStories {res['tinystories_pearson']:+.4f}  vs  wikitext {res['wikitext_pearson']:+.4f}  "
              f"(delta {d:+.4f})", flush=True)
        print(f"    derangement: tiny {res['tinystories']['derangement_pearson_seed42']:+.4f}, "
              f"wiki {res['wikitext']['derangement_pearson_seed42']:+.4f}  (both must be ~0)", flush=True)
        lifts.append(res["material_lift"] and not res["underpowered"])

    any_clean_lift = any(lifts)
    if any_clean_lift:
        top = ("VERDICT: a RICHER corpus IS a real category-structure lever -- wikitext numpy-Pearson beats "
               "TinyStories at matched size + matched a-priori labels on at least one (non-under-powered) "
               "taxonomy scale. RECOMMEND proceeding to download the FULL Simple-English-Wikipedia dump "
               "(~23.9M words; the thin local sample already shows the signal, the full dump should be stronger).")
    else:
        # distinguish "wiki <= tiny" from "under-powered / inconclusive"
        powered_results = [r for r in (res64, res320) if not r.get("skipped") and not r["underpowered"]]
        if not powered_results:
            top = ("VERDICT: INCONCLUSIVE / UNDER-POWERED -- the intersection vocab is too thin to power a "
                   "clean comparison at the not-under-powered threshold. On THIS local sample the corpus is "
                   "NOT a demonstrated generalization lever. CAVEAT: the full Simple-Wiki dump (~15x larger, "
                   "deeper Zipf) would survive a much larger intersection and may differ; do NOT over-claim "
                   "either way from this thin sample.")
        else:
            top = ("VERDICT: on THIS thin local Simple-Wiki sample the corpus is NOT a demonstrated "
                   "generalization lever (wikitext numpy-Pearson <= TinyStories at matched size + matched "
                   "labels). CAVEAT: the full Simple-Wiki dump (~15x larger, deeper Zipf) may differ, and the "
                   "intersection here is thin. This is consistent with the 3x-refuted corpus-content hypothesis "
                   "(scale + spiking read-out, not corpus flatness, cap the 320 gen number).")
    print("\n  " + top, flush=True)
    print(f"\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)

    out = {
        "experiment": "corpus_compare_gen_probe",
        "question": ("does a richer corpus (Simple-Wiki sample) raise the INTRINSIC numpy category-structure "
                     "ceiling above TinyStories, at MATCHED size + MATCHED a-priori labels?"),
        "substrate": "pure-numpy-exact-count (M[a,b]+=1, corr(M,C)=1.0 by construction) -- ZERO GPU/bridge",
        "pipeline": "double_center(log1p(M*100)) + heldout_generalization + Pearson(cos,S_true) (C0 template, verbatim body)",
        "seeds": list(SEEDS),
        "n_hub": N_HUB,
        "min_freq_intersection": MIN_FREQ,
        "pseudo_story_len": PSEUDO_STORY_LEN,
        "underpower_thresholds": {"min_words": MIN_WORDS_POWERED, "min_categories": MIN_CATS_POWERED},
        "size_match_budget_tokens": int(budget),
        "raw_token_counts": {name: int(len(flat[name])) for name in flat},
        "n_pseudo_stories": {name: int(len(stories_by_corpus[name])) for name in stories_by_corpus},
        "taxonomy_8x8": {k: v for k, v in res64.items() if k != "intersection"},
        "taxonomy_8x8_intersection": res64.get("intersection"),
        "taxonomy_40x8": {k: v for k, v in res320.items() if k != "intersection"},
        "taxonomy_40x8_intersection": res320.get("intersection"),
        "any_clean_material_lift": bool(any_clean_lift),
        "verdict": top,
        "provenance_priors": {
            "validated_numpy_64_full_tinystories": {"gen": 0.91, "pearson": 0.513, "chance": 0.125},
            "numpy_320_full_tinystories": {"gen": 0.453, "pearson": 0.215, "chance": 0.025,
                                           "source": "_curriculum_gen_C0_substrate_vs_scale.json"},
            "spiking_bridge_320": {"gen": 0.15, "pearson": 0.07,
                                   "source": "_curriculum_step1_320_real_corpus_seed42.json"},
            "refuted_corpus_content_hypotheses": ["yardstick-swap (0.153->0.167)",
                                                   "content-vocab filter (->0.125)",
                                                   "C0 decomposition: scale ~0.30 + spiking ~0.145 (Pearson)"],
            "note": ("THESE priors used the FULL corpus / FULL taxonomy. The numbers in THIS probe are on a "
                     "SIZE-MATCHED ~620K-token sample + the BOTH-corpora INTERSECTION vocab, so they are NOT "
                     "directly comparable to the full-corpus priors -- they are an apples-to-apples corpus "
                     "vs corpus comparison at matched size, which is the question (c) asks."),
        },
    }
    path = os.path.join(_REPO, "research", "findings", "raw", "_corpus_compare_gen_probe.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}\n", flush=True)


if __name__ == "__main__":
    main()
