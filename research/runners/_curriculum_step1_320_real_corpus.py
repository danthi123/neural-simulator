"""FOUNDATIONAL CURRICULUM — STEP 1: learn 320 concepts from the REAL TinyStories corpus with a
CORPUS-FREQUENCY-DERIVED curriculum, on a SINGLE bridge, and validate the bars + MEASURE VRAM/wall-clock.

Per the scoping `research/findings/raw/_foundational_curriculum_scaling_scoping.md` §5 Step 1:
    "Stream a REAL TinyStories shard to learn the ~320 most-frequent content words on ONE bridge (reproduce the
     validated tier, but with concepts/vocab DERIVED FROM THE CORPUS BY FREQUENCY, not the hand-curated
     taxonomy). GO bars (3 seeds): who/what recall >= 0.95; moat 0 false-accepts; generalization (held-out
     category) >= 0.80 with derangement-control collapse; MEASURE VRAM + wall-clock. This proves the
     corpus-derived (not hand-curated) pipeline works at the validated scale."

THE ENGINEERING PIECE (scoping §3 piece 2, "a corpus-derived curriculum"): replace the hardcoded syllabus with a
HIGH-FREQUENCY-FIRST order derived from the streamed corpus. `derive_curriculum_from_corpus` streams the corpus,
counts word frequency, and returns the top-N CONTENT words (a word that is a member of the INDEPENDENT
`g20_vocab_spec_2048` semantic taxonomy AND not a stopword/hub) ranked HIGH-FREQUENCY-FIRST. Age-of-acquisition
tracks frequency, so high-frequency-first IS the developmental order. Each chosen word keeps its independent
taxonomy category label -> the generalization reference S_true is the a-priori category structure (NEVER
corpus-derived; the load-bearing correctness property from option_c_real_cooccurrence_derisk).

THE SUBSTRATE (reuse-by-import, NO `sim/` edit): the validated on-bridge stream cortex
(`_phaseB_onbridge_stream_cortex_derisk.build_stream_bridge`): a hub (context) region + a target (concept) region
on ONE `SimulationBridge`, a fully-connected hub->target plastic pathway that LEARNS the co-occurrence M[Nt,n_hub]
by rate-Hebbian coincidence as the brain HEARS the corpus window-by-window. The read-out is the population
block-mean + log-double-centre (the validated normalization) -> a {word: phases[D]} grounded-code dict.

THE BARS (3 seeds):
  - recall >= 0.95 (who/what): the production `RFPhasorComposer` on the stream-LEARNED grounded codes; store SVO
    facts drawn from the corpus-learned vocab (noun-agent, verb, noun-patient); query who/what.
  - moat 0 false-accepts: query never-stored facts -> MUST abstain (return None). A single confident answer is a
    HARD STOP (the no-confab moat is NEVER weakened).
  - generalization >= 0.80 + category-derangement collapses: `heldout_generalization(code, cat_ids)` on the
    learned codes (a held-out concept lands in its correct category) vs SHUFFLED labels (must collapse to ~chance).
  - frozen-brain control: plasticity OFF -> the bridge hears but learns no codes -> competence (corr(M,C),
    generalization, recall) must NOT rise = the codes are LEARNED, not smuggled.
  - MEASURED VRAM + wall-clock: the calibrated per-bridge rate from a REAL corpus.

HONEST: if a bar misses on the real (noisier) corpus vs the curated 64-word baseline, report the EXACT value +
the cause (real-corpus noise vs curated). The first real-corpus 320-concept data point is a FINDING, not a
failure.

VOCAB-FILTER FIX (2026-06-25, per `research/findings/raw/_curriculum_gen_miss_REAL_scoping.md`): the gen miss
(0.153) was DIAGNOSED to the VOCAB SELECTION -- frequency-ranking over the FULL g20 taxonomy put ~48%
distributionally-FLAT adjective/function/emotion words at the TOP (they co-occur with everything -> near-uniform
codes that homogenize the entity codes too). The fix (`--vocab-filter content`, the DEFAULT) frequency-ranks
WITHIN co-occurrence-COHERENT CONTENT categories only (entities + verbs); the flat words stay context HUBS.
`--vocab-filter all` = the old freq-top-N (provenance); `--vocab-filter curated` = the validated TAXONOMY_40x8
positive control.

GPU (`SIM_BACKEND=cupy`). Run (the FIX, 3 seeds + the apples-to-apples coherent gen reference):
    SIM_BACKEND=cupy python -u -m research.runners._curriculum_step1_320_real_corpus \
        --seeds 42,43,44 --n-concepts 320 --vocab-filter content --gen-reference coherent \
        --out research/findings/raw/_curriculum_step1_320_real_corpus.json
The #3 positive control (should reproduce the validated ~0.91 gen on the curated content vocab):
    SIM_BACKEND=cupy python -u -m research.runners._curriculum_step1_320_real_corpus \
        --seeds 42,43,44 --n-concepts 320 --vocab-filter curated --gen-reference coherent \
        --out research/findings/raw/_curriculum_step1_320_curated_control.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

os.environ.setdefault("SIM_BACKEND", "cupy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host, is_gpu_backend  # noqa: E402
from research.runners.corpus_stream import (  # noqa: E402
    iter_stories, load_token_stream, default_corpus_path,
    iter_stories_multi, load_token_stream_multi, normalize_corpus_paths,
)
from research.runners._phaseB_onbridge_stream_cortex_derisk import build_stream_bridge  # noqa: E402
from research.runners.option_c_stageB_fair_test import STOPLIST  # noqa: E402
from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
import research.runners.g20_vocab_spec_2048 as VOCAB_SPEC  # noqa: E402

WINDOW = 2


def double_center(X):
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()


def _vram_used_mb():
    """Resident GPU memory used by THIS process (allocated, not just the pool), via cupy memGetInfo. None on CPU."""
    if not is_gpu_backend():
        return None
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return round((total - free) / (1024.0 * 1024.0), 1)
    except Exception:
        return None


def _pool_used_mb():
    try:
        from sim.backend import get_memory_pool_used_mb
        v = get_memory_pool_used_mb()
        return round(float(v), 1) if v is not None else None
    except Exception:
        return None


# ============================================================================================================
# THE ENGINEERING PIECE — derive the curriculum from the corpus by frequency (high-freq-first = developmental).
# ============================================================================================================

def _category_map():
    """word -> independent a-priori category label, from g20_vocab_spec_2048 (32 hand-curated semantic clusters of
    64 mutually-similar concepts). This taxonomy is INDEPENDENT of the corpus (the load-bearing correctness
    property): S_true (the generalization reference) is the category-block matrix over THIS taxonomy, NEVER
    corpus-derived.

    NOTE (the gen-reference fix, 2026-06-25): this `g20_vocab_spec_2048` taxonomy is a *SHARDING* spec (32x64 for
    splitting concepts across bridges), NOT a co-occurrence-CLUSTERABLE reference. ~33% of its categories are
    adjective/function clusters (texture/color/size adjectives + abstract/spatial/time/quantity/discourse function
    words) whose members modify DIFFERENT nouns / scatter across contexts and so do NOT share co-occurrence context
    with EACH OTHER -- a distributional (Hebbian co-occurrence) cortex provably cannot cluster them, yet this
    reference demands it (the Step-1 gen 0.153 / Pearson 0.07 miss; see
    `research/findings/raw/_curriculum_gen_miss_scoping.md`). Use `_coherent_category_map` for the gen metric;
    this map is retained as the `--gen-reference sharding` provenance number (reported alongside, never hidden)."""
    word2cat = {}
    for cat, words in VOCAB_SPEC.ALL_CLUSTERS_2048.items():
        for w in words:
            word2cat.setdefault(w, cat)   # clusters are globally unique (asserted at spec import); first wins
    return word2cat


# The CO-OCCURRENCE-COHERENT g20 domains: the concrete ENTITY + VERB clusters a distributional cortex CAN recover
# (members share story context -- animals appear together as pets, verbs share agents/scenes). The adjective +
# function clusters are DELIBERATELY EXCLUDED (color/size/texture adjectives modify different nouns; abstract/
# spatial/time/quantity/discourse words scatter) -- the scoping verified these are NOT co-occurrence-clusterable.
# Matches the domains the VALIDATED stream cortex scored generalization 0.91 against (stream_taxonomy_320).
COHERENT_G20_DOMAINS = frozenset({
    "mammals", "birds", "fish_reptiles", "insects",                       # animals
    "fruits", "vegetables", "prepared_foods", "drinks",                   # food/drink
    "body_parts", "kinship_people", "emotion_states",                    # body / people / feelings
    "weather_nature", "plants_trees",                                     # nature
    "furniture", "buildings", "clothing", "hand_tools", "machines",      # built environment / objects
    "land_vehicles", "air_water_vehicles",                               # vehicles
    "motion_verbs", "manipulation_verbs", "perception_verbs", "communication_verbs",  # verbs
})


# The TARGET-VOCAB CONTENT g20 domains (the 2026-06-25 vocab-filter fix, `--vocab-filter content`): the strict
# ENTITY + VERB domains a distributional cortex CAN cluster -- COHERENT_G20_DOMAINS minus emotion_states (the
# prompt's exclusion list is exactly adjectives [texture/size/color] + function [abstract_relations/spatial/time/
# quantity/question_discourse] + EMOTION; emotion words, though they co-occur, are excluded from the TARGET set per
# the owner directive and remain context HUBS like the rest). DISTINCT from COHERENT_G20_DOMAINS (the GEN
# reference, which keeps emotion as a coherent gen category) so the two concerns stay independent + auditable:
# this set picks WHICH words become TARGET concepts; the coherent map picks WHICH words are gen-scored.
CONTENT_G20_DOMAINS = COHERENT_G20_DOMAINS - frozenset({"emotion_states"})


def _coherent_category_map():
    """word -> INDEPENDENT a-priori CO-OCCURRENCE-COHERENT category label = the gen reference a distributional
    cortex can actually recover. Two independent a-priori sources, merged (NEVER corpus-derived):

      1. the VALIDATED `stream_taxonomy_320.TAXONOMY_40x8` (40x8 hand-curated semantic categories, freq>=50, the
         exact reference the stream cortex scored generalization 0.91 against -- CYCLE 94/96). FIRST priority.
      2. the COHERENT g20 entity/verb domains (`COHERENT_G20_DOMAINS`) for content words not in (1), prefixed
         `g20_<domain>` so the two sources stay distinct + auditable.

    Words in the INCOHERENT g20 categories (adjectives + function words) get NO coherent label (returned absent) --
    a co-occurrence cortex cannot cluster them, so they are not gen-usable (reported transparently as coverage).
    Returns {word: coherent_category_name} (a word absent from the dict has no coherent home)."""
    from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
    word2cat = {}
    # (1) the validated coherent taxonomy first (its categories are the apples-to-apples 0.91 reference)
    for cat, words in TAXONOMY_40x8.items():
        for w in words:
            word2cat.setdefault(w, cat)
    # (2) coherent g20 entity/verb domains for the remaining content words (distinct `g20_` prefix)
    g20 = _category_map()
    for w, cat in g20.items():
        if w not in word2cat and cat in COHERENT_G20_DOMAINS:
            word2cat[w] = "g20_" + cat
    return word2cat


def _coherent_labels_for_vocab(vocab, min_members=2):
    """Build the gen labels for `vocab` against the coherent a-priori map. A word is GEN-USABLE iff it has a
    coherent category AND that category has >= `min_members` of the vocab in it (heldout-nearest needs >=2).
    Returns (usable_idx [k], usable_labels [k] contiguous int ids, coverage_report dict). The labels are a-priori
    (independent of the corpus); only WHICH words are scored is restricted to the gen-usable coherent subset."""
    cmap = _coherent_category_map()
    cat_of = [cmap.get(w) for w in vocab]
    counts = Counter(c for c in cat_of if c is not None)
    usable_idx, usable_cats = [], []
    for i, c in enumerate(cat_of):
        if c is not None and counts[c] >= min_members:
            usable_idx.append(i)
            usable_cats.append(c)
    present = sorted(set(usable_cats))
    cat_to_id = {c: j for j, c in enumerate(present)}
    usable_labels = np.asarray([cat_to_id[c] for c in usable_cats], dtype=int)
    n_in_tax = sum(1 for c in cat_of if c is not None and not c.startswith("g20_"))
    n_in_g20 = sum(1 for c in cat_of if c is not None and c.startswith("g20_"))
    report = {
        "n_vocab": len(vocab),
        "n_with_coherent_category": int(sum(c is not None for c in cat_of)),
        "n_gen_usable": len(usable_idx),
        "coverage_frac_with_category": round(sum(c is not None for c in cat_of) / max(len(vocab), 1), 3),
        "coverage_frac_gen_usable": round(len(usable_idx) / max(len(vocab), 1), 3),
        "n_coherent_categories_used": len(present),
        "n_from_validated_taxonomy_40x8": n_in_tax,
        "n_from_coherent_g20_domains": n_in_g20,
        "n_incoherent_excluded": int(sum(c is None for c in cat_of)),
        "per_category_count": dict(counts.most_common()),
    }
    return usable_idx, usable_labels, report


def _is_content_word(word, word2cat):
    """The a-priori CONTENT-word predicate (the 2026-06-25 vocab-filter fix). A word is a CONTENT word iff its
    INDEPENDENT g20 category is in `CONTENT_G20_DOMAINS` -- i.e. it is an ENTITY (animals, food, body, people,
    nature, objects, vehicles, ...) or a VERB (motion/manipulation/perception/communication). The
    distributionally-FLAT g20 categories the prompt enumerates -- ADJECTIVES (texture_material_adj /
    size_shape_adj / color_adj), FUNCTION words (abstract_relations / spatial_words / time_words /
    quantity_number_words / question_discourse), and EMOTION (emotion_states) -- are EXCLUDED from the TARGET
    set. They remain available as context HUBS (the n_hub dimension), exactly as the validated recipe
    (stream_taxonomy_320 docstring :22-23 "ABSTRACT / FUNCTION words ... deliberately EXCLUDED") does. The filter
    is a-priori (CONTENT_G20_DOMAINS is fixed before the run), NOT tuned on the gen score (anti-cheat #2)."""
    return word2cat.get(word) in CONTENT_G20_DOMAINS


def derive_curriculum_from_corpus(corpus_path, n_concepts, verbose=True, vocab_filter="content"):
    """Stream the corpus, count word frequency, and return the top-`n_concepts` words ranked
    HIGH-FREQUENCY-FIRST (the developmental order) under the chosen `vocab_filter`. Returns:
        vocab     : list[str], the curriculum in developmental order (most-frequent first)
        cat_ids   : np.ndarray[int], the INDEPENDENT g20 category id per word (for the sharding gen reference)
        cat_names : list[str], the g20 category names (cat_ids index into this)
        freqs     : list[int], the corpus frequency per word (developmental order)
        report    : dict, the frequency-derivation report (coverage, range, per-category spread)

    THREE FILTERS (the 2026-06-25 fix; declared a-priori, never tuned on gen -- anti-cheat #2):

      - "content" (DEFAULT = the FIX, per `_curriculum_gen_miss_REAL_scoping.md` option #1): frequency-rank
        but KEEP ONLY words whose coherent category (`_coherent_category_map`) is a CONTENT category --
        ENTITIES + VERBS. The distributionally-FLAT adjective/function/emotion words (which a distributional
        Hebbian cortex provably cannot cluster, and which -- being the MOST frequent words -- homogenized the
        entity codes too: the gen 0.153 miss) are EXCLUDED from the TARGET set; they stay HUBS (context).
        Fills to `n_concepts`, or FEWER if the content-coverage of the corpus caps it (reported honestly).

      - "all" (PROVENANCE): the ORIGINAL top-N over the FULL g20 taxonomy (the gen 0.153 vocab -- 48% flat
        adjective/function words). Retained verbatim so the report can show the original number alongside.

      - "curated" (the #3 CONTROL): use the VALIDATED `stream_taxonomy_320.TAXONOMY_40x8` words directly
        (40x8 hand-curated co-occurrence-coherent CONTENT words, freq>=50 -- the EXACT vocab the stream cortex
        scored generalization ~0.91 against), frequency-ranked. NOT corpus-derived (curated meanings); the
        positive control that proves the pipeline reproduces the validated number when given the validated
        content-word vocab. cat_ids are mapped to the g20 taxonomy where a word has a g20 home (else a synthetic
        `curated_<TAXONOMY_40x8_category>` label so every curated word keeps an independent category for the
        sharding reference; the gen `coherent` reference reads TAXONOMY_40x8 natively).
    """
    word2cat = _category_map()
    coherent_map = _coherent_category_map()   # only for the gen-coverage diagnostic below (which CHOSEN words
    #                                            have a coherent GEN home), NOT the content filter (g20 domains).

    # COMBINED-CORPUS support (ADDITIVE; a single str path is byte-identical to the old single-file iter_stories):
    # `corpus_path` may be a single path, a comma/os.pathsep-separated string, or a list -> aggregate frequency
    # across the UNION (so the derived vocab is from all corpora) via iter_stories_multi.
    corpus_paths = normalize_corpus_paths(corpus_path)
    gfreq = Counter()
    n_stories = 0
    n_tokens = 0
    for toks in iter_stories_multi(corpus_paths):
        n_stories += 1
        n_tokens += len(toks)
        gfreq.update(toks)

    if vocab_filter == "curated":
        # the VALIDATED TAXONOMY_40x8 content vocab (the #3 control); frequency-rank for the developmental order.
        from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
        curated_cat_of = {}
        for cat, words in TAXONOMY_40x8.items():
            for w in words:
                curated_cat_of.setdefault(w, cat)
        candidates = list(curated_cat_of)
        # g20 home where it exists (keeps the sharding reference comparable); else a synthetic curated_<cat> label.
        def _cat_for(w):
            return word2cat.get(w) or ("curated_" + curated_cat_of[w])
    else:
        # frequency-rank over the g20 taxonomy members; "content" additionally requires a coherent (content) home.
        candidates = list(word2cat)
        def _cat_for(w):
            return word2cat[w]

    pool = [(w, gfreq[w]) for w in candidates if w not in STOPLIST and gfreq.get(w, 0) > 0]
    if vocab_filter == "content":
        pool = [(w, f) for (w, f) in pool if _is_content_word(w, word2cat)]
    pool.sort(key=lambda x: (-x[1], x[0]))   # freq desc, name asc for a deterministic tie-break
    chosen = pool[:n_concepts]

    n_content_present = sum(1 for (w, _) in pool)   # how many of the (already-filtered) pool exist in-corpus

    # the independent category structure of the chosen vocab (drop the empty categories so cat ids are contiguous)
    chosen_words = [w for w, _ in chosen]
    chosen_cats = [_cat_for(w) for w in chosen_words]
    present_cats = sorted(set(chosen_cats))
    cat_to_id = {c: i for i, c in enumerate(present_cats)}
    cat_ids = np.asarray([cat_to_id[c] for c in chosen_cats], dtype=int)
    freqs = [f for _, f in chosen]

    cat_count = Counter(chosen_cats)
    # a word is gen-usable iff its category has >= 2 members in the chosen vocab (heldout needs another member)
    n_gen_usable = sum(1 for c in chosen_cats if cat_count[c] >= 2)
    # how many of the chosen words are CONTENT (entity/verb g20 domain) -- 100% by construction for "content".
    n_chosen_content = sum(1 for w in chosen_words if _is_content_word(w, word2cat))
    # how many chosen words have a coherent GEN home (TAXONOMY_40x8 + coherent g20) -- the gen-scorable subset.
    n_chosen_coherent_gen = sum(1 for w in chosen_words if coherent_map.get(w) is not None)

    report = {
        "vocab_filter": vocab_filter,
        "corpus": "+".join(os.path.basename(p) for p in corpus_paths),
        "corpus_paths": list(corpus_paths),
        "n_corpora": len(corpus_paths),
        "n_stories": n_stories,
        "n_tokens": n_tokens,
        "n_unique_types": len(gfreq),
        "n_candidate_pool_present": n_content_present,
        "n_concepts_requested": n_concepts,
        "n_concepts_chosen": len(chosen_words),
        "capped_by_coverage": bool(len(chosen_words) < n_concepts),
        "freq_range": [int(chosen[0][1]), int(chosen[-1][1])] if chosen else [0, 0],
        "n_categories_covered": len(present_cats),
        "n_categories_total": len(VOCAB_SPEC.ALL_CLUSTERS_2048),
        "categories_missing": sorted(set(VOCAB_SPEC.ALL_CLUSTERS_2048) - set(present_cats)),
        "per_category_count": dict(cat_count.most_common()),
        "n_gen_usable_words": n_gen_usable,
        "n_chosen_content_words": n_chosen_content,
        "frac_chosen_content": round(n_chosen_content / max(len(chosen_words), 1), 3),
        "n_chosen_coherent_gen": n_chosen_coherent_gen,
        "top25": [(w, int(f)) for w, f in chosen[:25]],
        "tail10": [(w, int(f)) for w, f in chosen[-10:]],
    }
    if verbose:
        print(f"  [curriculum] corpus={report['corpus']} | {n_stories} stories, {n_tokens} tokens, "
              f"{len(gfreq)} types | vocab-filter={vocab_filter}", flush=True)
        _cap = " (CAPPED by content-coverage)" if report["capped_by_coverage"] else ""
        print(f"  [curriculum] {len(chosen_words)} words (high-freq-first){_cap}, freq {report['freq_range'][0]}"
              f"->{report['freq_range'][1]} | {len(present_cats)}/{len(VOCAB_SPEC.ALL_CLUSTERS_2048)} categories | "
              f"{n_gen_usable}/{len(chosen_words)} gen-usable (cat has >=2 members) | "
              f"{n_chosen_content}/{len(chosen_words)} content", flush=True)
        print(f"  [curriculum] developmental order (top 20): "
              f"{[w for w,_ in chosen[:20]]}", flush=True)
    return chosen_words, cat_ids, present_cats, freqs, report


# ============================================================================================================
# THE SUBSTRATE — the on-bridge stream cortex (reuse build_stream_bridge), hear the corpus, read learned codes.
# ============================================================================================================

class StreamCortexBridge:
    """A single persistent GPU bridge that LEARNS concept codes by hearing the corpus window-by-window (online
    rate-Hebbian co-occurrence) -- the validated on-bridge stream cortex, parameterized for the 320-concept tier."""

    def __init__(self, vocab, cat_ids, seed, stories, n_hub=300, n_per=16, hub_scale=250.0, tgt_scale=1200.0,
                 window_steps=2, D=128, plasticity_on=True, verbose=True):
        self.vocab = list(vocab)
        self.Nt = len(self.vocab)
        self.cat_ids = np.asarray(cat_ids, dtype=int)
        self.tgt_row = {w: i for i, w in enumerate(self.vocab)}
        self.target_set = set(self.vocab)
        self.seed = int(seed)
        self.n_hub, self.n_per = int(n_hub), int(n_per)
        self.hub_scale, self.tgt_scale = float(hub_scale), float(tgt_scale)
        self.window_steps = int(window_steps)
        self.D = int(D)
        self.plasticity_on = bool(plasticity_on)
        self.verbose = verbose
        self.stories = stories

        # hubs = the top-n_hub frequent CONTEXT words (a brain knows its common words; the fixed context dimension
        # keeps M LINEAR in vocab -- the scoping's decisive feasibility fact). Exclude stopwords + the targets.
        gfreq = Counter()
        for toks in self.stories:
            gfreq.update(toks)
        self.hubs = [w for w, _ in gfreq.most_common()
                     if w not in STOPLIST and w not in self.target_set][:self.n_hub]
        self.hub_idx = {w: i for i, w in enumerate(self.hubs)}
        self.keep = self.target_set | set(self.hubs)

        t0 = time.time()
        self.bridge, self.hub_region, self.tgt_region = build_stream_bridge(self.Nt, self.n_hub, self.n_per, seed)
        if not self.plasticity_on:
            # FROZEN-BRAIN anti-cheat: gate the stream cortex's Hebbian learning OFF (hears but learns no codes).
            self.bridge.core_config.enable_hebbian_learning = False
        self.build_s = time.time() - t0
        self.xp = self.bridge._cp if hasattr(self.bridge, "_cp") else None
        self.n_hub_neurons = self.n_hub * self.n_per
        self.n_tgt_neurons = self.Nt * self.n_per

        # fixed random complex projection: learned hub-co-occurrence row (length n_hub) -> phasor[D]. proj: (D, n_hub)
        rng = np.random.RandomState(seed * 7 + 3)
        self.proj = (rng.randn(self.D, self.n_hub) + 1j * rng.randn(self.D, self.n_hub)) / np.sqrt(self.n_hub)

        # host reference count of EXACTLY the windows the bridge sees (the learning-fidelity reference ONLY; it does
        # NOT drive the bridge -- the bridge learns M in its synapses from the co-activation).
        self.C_stream = np.zeros((self.Nt, self.n_hub), dtype=np.float64)
        self.total_windows = 0

    def _present_window(self, tgt_ids, hub_ids):
        hub_full = np.zeros(self.n_hub_neurons, np.float32)
        tgt_full = np.zeros(self.n_tgt_neurons, np.float32)
        for h in hub_ids:
            hub_full[h * self.n_per:(h + 1) * self.n_per] = self.hub_scale
        for t in tgt_ids:
            tgt_full[t * self.n_per:(t + 1) * self.n_per] = self.tgt_scale
        b = self.bridge
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[self.hub_region] = self.xp.asarray(hub_full) if self.xp is not None else hub_full
        b.cp_external_input_current[self.tgt_region] = self.xp.asarray(tgt_full) if self.xp is not None else tgt_full
        for _ in range(self.window_steps):
            b._run_one_simulation_step()

    def hear_corpus(self, max_windows, story_seed):
        """Stream corpus windows; co-activate each window's target + context-hub populations; the bridge's Hebbian
        synapses accumulate the co-occurrence. Returns the number of windows streamed. NO precomputed co-occurrence
        in the drive -- only who co-occurs in each window."""
        rng = np.random.RandomState(story_seed)
        story_order = rng.permutation(len(self.stories))
        n_win = 0
        _learn_t0 = time.time()
        for si in story_order:
            if n_win >= max_windows:
                break
            kept = [t for t in self.stories[si] if t in self.keep]
            for c in range(len(kept)):
                lo, hi = max(0, c - WINDOW), min(len(kept), c + WINDOW + 1)
                win = kept[lo:hi]
                tgt_ids = [self.tgt_row[w] for w in win if w in self.target_set]
                hub_ids = [self.hub_idx[w] for w in win if w in self.hub_idx]
                if tgt_ids and hub_ids:
                    self._present_window(tgt_ids, hub_ids)
                    for t in tgt_ids:
                        for h in hub_ids:
                            self.C_stream[t, h] += 1.0
                    n_win += 1
                    if (n_win % 5000) == 0:
                        print(f"[learn-progress] {n_win}/{max_windows} windows  "
                              f"{time.time()-_learn_t0:.0f}s", flush=True)
                    if n_win >= max_windows:
                        break
        self.bridge.cp_external_input_current[:] = 0.0
        self.total_windows += n_win
        return n_win

    def read_codes(self):
        """Read the stream-learned codes from the bridge synapses -> (M[Nt,n_hub], code[Nt,n_hub], grounded dict).
        M = population block-mean of the learned hub->target weights; code = log-double-centre; grounded phasor =
        angle(proj @ code_row) in [0,1)^D."""
        W = np.asarray(to_host(self.bridge.cp_connections.todense())).astype(np.float64)
        blk = W[np.ix_(self.hub_region, self.tgt_region)].reshape(
            self.n_hub, self.n_per, self.Nt, self.n_per).mean(axis=(1, 3))
        M = blk.T                                       # (Nt, n_hub) stream-learned co-occurrence
        code = double_center(np.log1p(M * 100.0))       # (Nt, n_hub) normalized code
        grounded = {}
        for w, i in self.tgt_row.items():
            z = self.proj @ code[i].astype(np.complex128)
            grounded[w] = (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)
        return M, code, grounded

    def learning_fidelity(self):
        M, _, _ = self.read_codes()
        if M.std() <= 0 or self.C_stream.std() <= 0:
            return 0.0
        return float(np.corrcoef(M.flatten(), self.C_stream.flatten())[0, 1])

    def close(self):
        try:
            self.bridge = None
            if self.xp is not None:
                self.xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


# ============================================================================================================
# THE BARS.
# ============================================================================================================

def _make_svo_facts(vocab, cat_ids, cat_names, n_facts, seed):
    """Construct sensible SVO facts from the corpus-LEARNED vocab: a noun agent, a verb, a noun patient, all in
    the learned 320. (noun cluster = an entity category; verb cluster = a *_verbs category.) Deterministic per
    seed. Returns (facts, absent_what, absent_who):
      facts        : list[(agent, action, patient)] to store + recall
      absent_what  : list[(agent, action)] NEVER stored (moat: what_does must abstain)
      absent_who   : list[(action, patient)] NEVER stored (moat: who_does must abstain)
    """
    rng = np.random.RandomState(seed * 131 + 5)
    name_of = {i: c for i, c in enumerate(cat_names)}
    verb_cats = {i for i, c in name_of.items() if c.endswith("_verbs")}
    # entity (noun) categories: everything that is neither a verb cluster nor an adjective/abstract/function cluster
    non_entity_suffix = ("_verbs", "_adj")
    non_entity_names = {"abstract_relations", "spatial_words", "time_words", "quantity_number_words",
                        "question_discourse", "emotion_states"}
    nouns = [w for w, ci in zip(vocab, cat_ids)
             if not name_of[ci].endswith(non_entity_suffix) and name_of[ci] not in non_entity_names]
    verbs = [w for w, ci in zip(vocab, cat_ids) if ci in verb_cats]
    if len(nouns) < 4 or len(verbs) < 2:
        # fall back to any words if the frequency-derived vocab is verb/noun-thin (still a valid recall test)
        nouns = list(vocab)
        verbs = list(vocab)

    facts, seen_pairs = [], set()
    attempts = 0
    while len(facts) < n_facts and attempts < n_facts * 50:
        attempts += 1
        a = nouns[rng.randint(len(nouns))]
        v = verbs[rng.randint(len(verbs))]
        p = nouns[rng.randint(len(nouns))]
        if a == p:
            continue
        if (a, v) in seen_pairs or (v, p) in seen_pairs:   # keep who/what cues unambiguous (one patient per a,v)
            continue
        facts.append((a, v, p))
        seen_pairs.add((a, v))
        seen_pairs.add((v, p))

    # absent cues: (agent, action) and (action, patient) combos that were NEVER stored -> the moat must abstain.
    stored_av = {(a, v) for a, v, _ in facts}
    stored_vp = {(v, p) for _, v, p in facts}
    absent_what, absent_who = [], []
    tries = 0
    while (len(absent_what) < len(facts) or len(absent_who) < len(facts)) and tries < n_facts * 80:
        tries += 1
        a = nouns[rng.randint(len(nouns))]
        v = verbs[rng.randint(len(verbs))]
        p = nouns[rng.randint(len(nouns))]
        if len(absent_what) < len(facts) and (a, v) not in stored_av and (a, v) not in {x[:2] for x in absent_what}:
            absent_what.append((a, v))
        if len(absent_who) < len(facts) and (v, p) not in stored_vp and (v, p) not in {x for x in absent_who}:
            absent_who.append((v, p))
    return facts, absent_what, absent_who


def measure_recall_and_moat(grounded, vocab, cat_ids, cat_names, seed, n_facts, D):
    """recall (who/what) + the no-confab moat (0 false-accepts) on the production RFPhasorComposer, using the
    stream-LEARNED grounded codes. Returns a dict."""
    from research.runners.rf_phasor_composer import RFPhasorComposer

    facts, absent_what, absent_who = _make_svo_facts(vocab, cat_ids, cat_names, n_facts, seed)
    used = sorted({w for f in facts for w in f})
    comp = RFPhasorComposer(seed=seed, D=D, vocab=sorted(set(vocab)), grounded_codes=grounded)
    for a, v, p in facts:
        comp.store(a, v, p)

    # recall: every stored fact retrieves (who AND what)
    recall_ok, recall_tot = 0, 0
    for a, v, p in facts:
        if comp.query_patient(a, v) == p:
            recall_ok += 1
        recall_tot += 1
        if comp.query_agent(v, p) == a:
            recall_ok += 1
        recall_tot += 1
    recall = recall_ok / max(recall_tot, 1)

    # moat: every NEVER-stored cue must abstain (return None). A single confident answer = a moat breach.
    false_accept, abstain_tot, breaches = 0, 0, []
    for a, v in absent_what:
        abstain_tot += 1
        ans = comp.query_patient(a, v)
        if ans is not None:
            false_accept += 1
            breaches.append(f"query_patient({a},{v}) -> {ans!r} (should abstain)")
    for v, p in absent_who:
        abstain_tot += 1
        ans = comp.query_agent(v, p)
        if ans is not None:
            false_accept += 1
            breaches.append(f"query_agent({v},{p}) -> {ans!r} (should abstain)")
    abstain = 1.0 - false_accept / max(abstain_tot, 1)
    mean_sim = float(np.mean([_phase_cos(grounded[a], grounded[b], D)
                              for i, a in enumerate(used) for b in used[i + 1:]])) if len(used) > 1 else 0.0
    return {
        "recall": recall, "recall_correct": recall_ok, "recall_total": recall_tot,
        "abstain": abstain, "false_accept": false_accept, "abstain_total": abstain_tot,
        "moat_breaches": breaches[:8],
        "n_facts": len(facts), "n_used_concepts": len(used), "mean_grounded_phase_cos": mean_sim,
    }


def _phase_cos(pa, pb, D):
    za = np.exp(2j * np.pi * np.asarray(pa, dtype=float))
    zb = np.exp(2j * np.pi * np.asarray(pb, dtype=float))
    return float(np.real(np.vdot(za, zb)) / D)


def _gen_on_labels(code, labels, seed):
    """Held-out nearest-category gen + derangement control + Pearson(cos, S_true) on a code matrix + a-priori
    `labels` (the shared metric, byte-identical to the validated arc: heldout_generalization / _cos_sim /
    _pearson_vs_Strue). `code`/`labels` may be a row-SUBSET (the gen-usable coherent words)."""
    labels = np.asarray(labels, dtype=int)
    if code.shape[0] < 2 or len(np.unique(labels)) < 2:
        return {"generalization": 0.0, "chance": 0.0, "ratio_vs_chance": 0.0,
                "derangement_generalization": 0.0, "derangement_collapses": False, "pearson_vs_Strue": 0.0,
                "n_scored": int(code.shape[0]), "n_categories": int(len(np.unique(labels)))}
    gen, chance = heldout_generalization(code, labels)
    rng = np.random.RandomState(seed * 99 + 1)
    perm = rng.permutation(labels)
    gen_perm, _ = heldout_generalization(code, perm)
    S_true = (labels[:, None] == labels[None, :]).astype(np.float64)
    pearson = _pearson_vs_Strue(_cos_sim(code), S_true)
    return {"generalization": gen, "chance": chance, "ratio_vs_chance": gen / max(chance, 1e-9),
            "derangement_generalization": gen_perm,
            "derangement_collapses": bool(gen_perm <= max(chance, 1e-9) + 0.05 or gen_perm < 0.5 * gen),
            "pearson_vs_Strue": pearson,
            "n_scored": int(code.shape[0]), "n_categories": int(len(np.unique(labels)))}


def measure_generalization(code, cat_ids, seed, vocab=None, gen_reference="coherent"):
    """Held-out category generalization on the learned code (a concept lands in its correct category by mean
    cosine to the OTHER members) + the category-DERANGEMENT control + the structure-recovery Pearson(cos, S_true).

    Two REFERENCES (the 2026-06-25 gen-reference fix; both reported, neither hidden -- anti-cheat #5):
      - "sharding"  : the FULL g20_vocab_spec_2048 sharding taxonomy (`cat_ids`) -- the ORIGINAL Step-1 reference
                      (gen 0.153). A *SHARDING* spec, NOT co-occurrence-clusterable; ~33% adjective/function
                      categories a distributional cortex cannot recover. Retained for provenance.
      - "coherent"  : the INDEPENDENT co-occurrence-COHERENT reference (validated TAXONOMY_40x8 + coherent g20
                      entity/verb domains), scored over the GEN-USABLE coherent subset of `vocab` -- the
                      apples-to-apples reference the validated stream cortex scored 0.91 against.

    Returns the dict for the SELECTED `gen_reference` as the top-level numbers, PLUS `both` = {sharding:..,
    coherent:..} (each with its coverage) so the report shows BOTH. `vocab` is required for "coherent"."""
    cat_ids = np.asarray(cat_ids, dtype=int)

    # (1) sharding reference: the full-vocab g20 labels (the original 0.153 number; provenance).
    sharding = _gen_on_labels(code, cat_ids, seed)
    sharding["reference"] = "sharding_g20_vocab_spec_2048"
    sharding["coverage"] = {"n_scored": int(code.shape[0]), "coverage_frac_gen_usable": 1.0,
                            "note": "full vocab; ~33% categories are co-occurrence-INCOHERENT adjective/function "
                                    "clusters a distributional cortex cannot recover"}

    # (2) coherent reference: gen-usable coherent subset of the vocab (the apples-to-apples 0.91 reference).
    coherent = None
    if vocab is not None:
        usable_idx, usable_labels, cov = _coherent_labels_for_vocab(vocab)
        coh = _gen_on_labels(code[np.asarray(usable_idx, dtype=int)] if usable_idx else code[:0],
                             usable_labels, seed)
        coh["reference"] = "coherent_taxonomy_40x8_plus_coherent_g20"
        coh["coverage"] = cov
        coherent = coh

    selected = coherent if (gen_reference == "coherent" and coherent is not None) else sharding
    out = dict(selected)
    out["gen_reference"] = gen_reference
    out["both"] = {"sharding": sharding, "coherent": coherent}
    return out


# ============================================================================================================
# PER-SEED + DRIVER.
# ============================================================================================================

def run_seed(seed, stories, vocab, cat_ids, cat_names, a):
    print(f"\n{'='*100}\n  STEP 1 — SEED {seed}  ({len(vocab)} corpus-frequency concepts on ONE bridge)\n{'='*100}",
          flush=True)
    vram0 = _vram_used_mb()

    # ---- LEARN: hear the corpus on a persistent bridge (plasticity ON) ----
    cx = StreamCortexBridge(vocab, cat_ids, seed, stories, n_hub=a.n_hub, n_per=a.n_per,
                            window_steps=a.window_steps, D=a.D, plasticity_on=True, verbose=True)
    t_learn = time.time()
    n_win = cx.hear_corpus(a.max_windows, story_seed=seed)
    learn_s = time.time() - t_learn
    vram_peak = _vram_used_mb()
    pool_mb = _pool_used_mb()
    M, code, grounded = cx.read_codes()
    corr_mc = cx.learning_fidelity()
    n_pool_neurons = cx.n_hub_neurons + cx.n_tgt_neurons
    print(f"  [learn] {n_win} windows in {learn_s:.0f}s | {n_pool_neurons} neurons "
          f"(hub {cx.n_hub_neurons} + tgt {cx.n_tgt_neurons}) | corr(M,C)={corr_mc:+.3f} | "
          f"VRAM resident {vram_peak} MB (pool {pool_mb} MB)", flush=True)

    # ---- SAVE the trained codes (the first-chat brain artifact the composer/DiscursiveTurn loads) ----
    if getattr(a, "save_codes", None):
        _sc = a.save_codes if len(getattr(a, "seeds", [seed])) <= 1 else f"{a.save_codes}_seed{seed}"
        os.makedirs(os.path.dirname(_sc) or ".", exist_ok=True)
        G = np.array([grounded[w] for w in vocab], dtype=np.float64)   # (Nt, D) phasors, vocab order
        np.savez(_sc, vocab=np.array(vocab, dtype=object), grounded=G,
                 cat_ids=np.asarray(cat_ids), cat_names=np.array(cat_names, dtype=object),
                 code=code, M=M, seed=int(seed), n_concepts=int(len(vocab)), D=int(a.D))
        print(f"  [save-codes] wrote {_sc}.npz  ({len(vocab)} concepts, D={G.shape[1]}, "
              f"corr(M,C)={corr_mc:+.3f})", flush=True)

    # ---- BARS on the LEARNED codes ----
    rm = measure_recall_and_moat(grounded, vocab, cat_ids, cat_names, seed, a.n_facts, a.D)
    gen = measure_generalization(code, cat_ids, seed, vocab=vocab, gen_reference=a.gen_reference)
    print(f"  [recall/moat] recall {rm['recall']:.3f} ({rm['recall_correct']}/{rm['recall_total']}) | "
          f"abstain {rm['abstain']:.3f} (false-accept {rm['false_accept']}/{rm['abstain_total']}) | "
          f"{rm['n_facts']} facts", flush=True)
    _shc, _coc = gen["both"]["sharding"], gen["both"]["coherent"]
    print(f"  [generalization] gen-reference={a.gen_reference} -> SELECTED {gen['generalization']:.3f} "
          f"({gen['ratio_vs_chance']:.1f}x chance {gen['chance']:.3f}) | derangement "
          f"{gen['derangement_generalization']:.3f} (collapses={gen['derangement_collapses']}) | "
          f"Pearson {gen['pearson_vs_Strue']:+.3f}", flush=True)
    if _coc is not None:
        print(f"  [generalization] BOTH refs: sharding(g20) {_shc['generalization']:.3f} (full 320, "
              f"Pearson {_shc['pearson_vs_Strue']:+.3f}) | coherent {_coc['generalization']:.3f} "
              f"(scored {_coc['coverage']['n_gen_usable']}/{len(vocab)} = "
              f"{_coc['coverage']['coverage_frac_gen_usable']*100:.0f}% gen-usable, "
              f"{_coc['coverage']['n_coherent_categories_used']} cats, Pearson "
              f"{_coc['pearson_vs_Strue']:+.3f})", flush=True)
    cx.close()

    # ---- FROZEN-BRAIN control: plasticity OFF -> hears but learns no codes -> competence must NOT rise ----
    frozen = None
    if not a.no_frozen:
        cxf = StreamCortexBridge(vocab, cat_ids, seed, stories, n_hub=a.n_hub, n_per=a.n_per,
                                 window_steps=a.window_steps, D=a.D, plasticity_on=False, verbose=False)
        cxf.hear_corpus(a.max_windows, story_seed=seed)
        Mf, codef, groundedf = cxf.read_codes()
        corr_mc_f = cxf.learning_fidelity()
        genf = measure_generalization(codef, cat_ids, seed, vocab=vocab, gen_reference=a.gen_reference)
        rmf = measure_recall_and_moat(groundedf, vocab, cat_ids, cat_names, seed, a.n_facts, a.D)
        cxf.close()
        # The frozen brain HEARS but learns no codes -> its competence must NOT rise above the learned brain. The
        # LOAD-BEARING signals (robust to scale): (i) corr(M,C) ~ 0 (no co-occurrence learned in the synapses);
        # (ii) recall well below the bar; (iii) its generalization does NOT exceed the learned brain's. (A loose
        # absolute "<= chance" check is fragile at small scale where chance is high; the relative
        # frozen-doesn't-beat-learned + corr~0 + recall-collapse is the honest provenance contrast.)
        frozen_flat = bool(abs(corr_mc_f) < 0.15
                           and rmf["recall"] < 0.5
                           and genf["generalization"] <= max(gen["generalization"], genf["chance"]) + 1e-9)
        frozen = {"corr_MC": corr_mc_f, "generalization": genf["generalization"], "chance": genf["chance"],
                  "recall": rmf["recall"], "frozen_competence_flat": frozen_flat,
                  "learned_corr_MC": corr_mc, "learned_generalization": gen["generalization"],
                  "learned_recall": rm["recall"]}
        print(f"  [frozen-brain] corr(M,C)={corr_mc_f:+.3f} | gen {genf['generalization']:.3f} | "
              f"recall {rmf['recall']:.3f} -> competence-flat={frozen_flat}  (vs learned: corr {corr_mc:+.3f}, "
              f"gen {gen['generalization']:.3f}, recall {rm['recall']:.3f})", flush=True)

    return {
        "seed": seed,
        "n_concepts": len(vocab),
        "n_windows": n_win,
        "n_pool_neurons": n_pool_neurons,
        "learn_seconds": round(learn_s, 1),
        "build_seconds": round(cx.build_s, 2),
        "vram_resident_mb_before": vram0,
        "vram_resident_mb_peak": vram_peak,
        "pool_used_mb": pool_mb,
        "corr_MC": corr_mc,
        "recall": rm["recall"], "recall_detail": rm,
        "moat_false_accepts": rm["false_accept"],
        "generalization": gen["generalization"], "generalization_detail": gen,
        # explicit provenance: BOTH gen numbers, never hidden (anti-cheat #5).
        "generalization_sharding": gen["both"]["sharding"]["generalization"],
        "generalization_coherent": (gen["both"]["coherent"]["generalization"]
                                    if gen["both"]["coherent"] is not None else None),
        "frozen": frozen,
    }


def decide(per_seed, a):
    seeds = list(per_seed.keys())
    rec = [per_seed[s]["recall"] for s in seeds]
    fa = [per_seed[s]["moat_false_accepts"] for s in seeds]
    g = [per_seed[s]["generalization"] for s in seeds]                       # the SELECTED reference (gen-gating)
    g_shard = [per_seed[s]["generalization_sharding"] for s in seeds]        # provenance (the original 0.153)
    g_coh = [per_seed[s]["generalization_coherent"] for s in seeds]          # the coherent (fix) number
    der = [per_seed[s]["generalization_detail"]["derangement_collapses"] for s in seeds]
    frozen_flat = [per_seed[s]["frozen"]["frozen_competence_flat"] for s in seeds
                   if per_seed[s]["frozen"] is not None]
    corr = [per_seed[s]["corr_MC"] for s in seeds]

    recall_ok = all(r >= a.recall_bar for r in rec)
    moat_ok = all(f == 0 for f in fa)                # the no-confab moat is NEVER weakened
    gen_ok = all(x >= a.gen_bar for x in g)
    der_ok = all(der)
    frozen_ok = (len(frozen_flat) == 0) or all(frozen_flat)

    go = bool(recall_ok and moat_ok and gen_ok and der_ok and frozen_ok)

    # per-bar pass/miss for an honest report (the first real-corpus 320 data point)
    bars = {
        "recall_bar": a.recall_bar, "recall_per_seed": rec, "recall_pass": recall_ok,
        "moat_false_accepts_per_seed": fa, "moat_pass_0FA": moat_ok,
        "gen_bar": a.gen_bar, "gen_reference": a.gen_reference,
        "generalization_per_seed": g, "generalization_pass": gen_ok,
        # BOTH references reported alongside (anti-cheat #5: the swap is a re-measure, NOT hiding the 0.153):
        "generalization_sharding_per_seed": g_shard,
        "generalization_coherent_per_seed": g_coh,
        "derangement_collapses_per_seed": der, "derangement_pass": der_ok,
        "frozen_competence_flat_per_seed": frozen_flat, "frozen_pass": frozen_ok,
        "corr_MC_per_seed": corr,
    }
    return go, bars


def main():
    p = argparse.ArgumentParser(description="Foundational curriculum Step 1: 320 concepts from the REAL "
                                            "TinyStories corpus, frequency-derived curriculum, single bridge.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-concepts", type=int, default=320, help="curriculum size (the validated 320 tier)")
    p.add_argument("--n-hub", type=int, default=300, help="stream-cortex hub (context-word) count")
    p.add_argument("--n-per", type=int, default=16, help="neurons per concept (population code)")
    p.add_argument("--window-steps", type=int, default=2, help="bridge steps per stream window")
    p.add_argument("--max-windows", type=int, default=150000,
                   help="stream-window budget (the validated 320 rate: ~150K windows). Caps wall-clock.")
    p.add_argument("--D", type=int, default=128, help="composer phasor dimension")
    p.add_argument("--n-facts", type=int, default=24, help="SVO facts to store for the recall/moat bars")
    p.add_argument("--recall-bar", type=float, default=0.95)
    p.add_argument("--gen-bar", type=float, default=0.80)
    p.add_argument("--vocab-filter", choices=["content", "all", "curated"], default="content",
                   help="WHICH words become the TARGET concepts (the 2026-06-25 fix per "
                        "_curriculum_gen_miss_REAL_scoping.md option #1). 'content' (DEFAULT = the FIX): "
                        "frequency-rank but keep ONLY co-occurrence-COHERENT CONTENT words (entities + verbs); "
                        "the distributionally-flat adjective/function/emotion words (the gen-0.153 cause -- "
                        "they homogenize the entity codes) are EXCLUDED from the target set, remaining context "
                        "HUBS. 'all': the ORIGINAL freq-top-N over the FULL g20 taxonomy (the gen-0.153 vocab, "
                        "~half flat words) -- PROVENANCE. 'curated': the VALIDATED stream_taxonomy_320.TAXONOMY_40x8 "
                        "content vocab directly (the #3 positive CONTROL -- should reproduce the ~0.91 gen).")
    p.add_argument("--gen-reference", choices=["coherent", "sharding"], default="coherent",
                   help="generalization reference. 'coherent' (default, the FIX): the INDEPENDENT co-occurrence-"
                        "COHERENT taxonomy (validated TAXONOMY_40x8 + coherent g20 entity/verb domains), scored "
                        "over the gen-usable coherent subset -- the apples-to-apples reference the validated "
                        "stream cortex scored 0.91 against. 'sharding': the ORIGINAL full g20_vocab_spec_2048 "
                        "sharding taxonomy (gen 0.153). BOTH are always reported regardless of this choice.")
    p.add_argument("--no-frozen", action="store_true", help="skip the frozen-brain control (debug only)")
    p.add_argument("--corpus-path", default=None,
                   help="path to a SINGLE plain-text corpus shard; default = data/corpus/tinystories.txt. "
                        "(Byte-identical legacy single-corpus path; for the COMBINED corpus use --corpus-paths.)")
    p.add_argument("--corpus-paths", default=None,
                   help="COMBINED corpus: comma-separated list of plain-text corpus shards whose token-frequency "
                        "+ co-occurrence are aggregated across the UNION (so the derived vocab spans all corpora) "
                        "-- the Rung-1 knowledge-scaling path (TinyStories for clean codes + Wikipedia for "
                        "breadth, per _knowledge_scaling_first_chat_scoping.md). Each file is split on its OWN "
                        "<|endoftext|> delimiter (a file with none streams as one document). Default = the single "
                        "TinyStories corpus (--corpus-path / data/corpus/tinystories.txt) => byte-identical to the "
                        "legacy single-corpus run. Takes precedence over --corpus-path when given.")
    p.add_argument("--out", default="research/findings/raw/_curriculum_step1_320_real_corpus.json")
    p.add_argument("--save-codes", default=None,
                   help="path (no ext) to save the trained grounded codes (.npz: vocab+grounded+code+M) "
                        "as the first-chat brain artifact the composer/DiscursiveTurn loads")
    a = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    # Resolve the corpus path(s): --corpus-paths (COMBINED, comma-sep) takes precedence; else the single
    # --corpus-path (default TinyStories). normalize_corpus_paths makes a single bare path byte-identical to the
    # legacy single-corpus run. (A combined-corpus run is the additive Rung-1 path.)
    if a.corpus_paths:
        corpus_paths = normalize_corpus_paths(a.corpus_paths)
    else:
        corpus_paths = [a.corpus_path or default_corpus_path()]
    for cp in corpus_paths:
        if not os.path.exists(cp):
            print(f"[ERROR] corpus not found: {cp}", flush=True)
            sys.exit(2)
    # `corpus_path` retained as the value passed to the curriculum deriver (str when single -> byte-identical;
    # list when combined). The deriver + loaders accept either (normalize_corpus_paths internally).
    corpus_path = corpus_paths[0] if len(corpus_paths) == 1 else corpus_paths

    print("=" * 100, flush=True)
    print("[FOUNDATIONAL CURRICULUM — STEP 1: 320 concepts from the REAL TinyStories corpus]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND')}  seeds={seeds}  n_concepts={a.n_concepts}  "
          f"n_hub={a.n_hub}  n_per={a.n_per}  max_windows={a.max_windows}  D={a.D}  "
          f"vocab_filter={a.vocab_filter}", flush=True)
    print(f"  corpus={'+'.join(corpus_paths) if len(corpus_paths) > 1 else corpus_paths[0]} "
          f"({len(corpus_paths)} corpus file(s))", flush=True)
    print("  bars: recall>=%.2f (who/what) | moat 0 false-accepts | generalization>=%.2f (gen-reference=%s; "
          "BOTH sharding+coherent reported) + derangement collapse | frozen-brain control | MEASURE VRAM + "
          "wall-clock" % (a.recall_bar, a.gen_bar, a.gen_reference), flush=True)
    print("=" * 100, flush=True)

    t0 = time.time()
    # derive the curriculum ONCE (corpus frequency is seed-independent); the story stream is loaded once + reused.
    # load_token_stream_multi over a single-element list is byte-identical to the legacy load_token_stream.
    stories = load_token_stream_multi(corpus_paths)
    vocab, cat_ids, cat_names, freqs, curr_report = derive_curriculum_from_corpus(
        corpus_path, a.n_concepts, vocab_filter=a.vocab_filter)

    # the COHERENT gen-reference coverage of the frequency-derived vocab (seed-independent; reported transparently).
    _coh_usable_idx, _coh_labels, coherent_coverage = _coherent_labels_for_vocab(vocab)
    print(f"  [gen-reference] coherent map covers {coherent_coverage['n_with_coherent_category']}/{len(vocab)} "
          f"({coherent_coverage['coverage_frac_with_category']*100:.0f}%) of the frequency-vocab with a coherent "
          f"category; {coherent_coverage['n_gen_usable']}/{len(vocab)} "
          f"({coherent_coverage['coverage_frac_gen_usable']*100:.0f}%) gen-usable (cat>=2 members) across "
          f"{coherent_coverage['n_coherent_categories_used']} categories "
          f"({coherent_coverage['n_from_validated_taxonomy_40x8']} from validated TAXONOMY_40x8 + "
          f"{coherent_coverage['n_from_coherent_g20_domains']} from coherent g20 domains); "
          f"{coherent_coverage['n_incoherent_excluded']} incoherent (adjective/function) words excluded.", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, stories, vocab, cat_ids, cat_names, a)

    go, bars = decide(per_seed, a)

    # the measured calibrated per-bridge rate (the number that firms up the full-training ETA)
    learn_secs = [per_seed[str(s)]["learn_seconds"] for s in seeds]
    n_wins = [per_seed[str(s)]["n_windows"] for s in seeds]
    vram_peaks = [per_seed[str(s)]["vram_resident_mb_peak"] for s in seeds
                  if per_seed[str(s)]["vram_resident_mb_peak"]]
    mean_learn_s = float(np.mean(learn_secs))
    mean_win = float(np.mean(n_wins))
    windows_per_s = mean_win / max(mean_learn_s, 1e-9)
    calibrated = {
        "mean_learn_seconds_per_bridge": round(mean_learn_s, 1),
        "mean_windows_per_bridge": round(mean_win, 0),
        "windows_per_second": round(windows_per_s, 1),
        "seconds_per_320_concepts": round(mean_learn_s, 1),
        "vram_resident_mb_peak_max": (max(vram_peaks) if vram_peaks else None),
        "vram_under_24gb": (max(vram_peaks) < 24000 if vram_peaks else None),
    }

    _g_shard = bars["generalization_sharding_per_seed"]
    _g_coh = [x for x in bars["generalization_coherent_per_seed"] if x is not None]
    _both_str = (f"[BOTH references: sharding(g20, the original 0.153) "
                 f"{min(_g_shard):.3f}-{max(_g_shard):.3f}; coherent(fix) "
                 f"{(min(_g_coh) if _g_coh else 0):.3f}-{(max(_g_coh) if _g_coh else 0):.3f} over "
                 f"{coherent_coverage['n_gen_usable']}/{len(vocab)} gen-usable "
                 f"({coherent_coverage['coverage_frac_gen_usable']*100:.0f}%) words]")
    if go:
        verdict = (
            f"STEP 1 GO -> STEP 2 — the CORPUS-FREQUENCY-DERIVED 320-concept curriculum learns on ONE bridge from "
            f"the REAL TinyStories corpus and passes every bar (3 seeds): recall {min(bars['recall_per_seed']):.2f}"
            f"-{max(bars['recall_per_seed']):.2f} >= {a.recall_bar}, moat 0 false-accepts, generalization "
            f"({a.gen_reference} reference) {min(bars['generalization_per_seed']):.2f}"
            f"-{max(bars['generalization_per_seed']):.2f} >= {a.gen_bar} with derangement collapse, frozen-brain "
            f"control holds. {_both_str}. Calibrated per-bridge rate: "
            f"{calibrated['mean_learn_seconds_per_bridge']}s for 320 concepts "
            f"({calibrated['windows_per_second']} win/s), VRAM {calibrated['vram_resident_mb_peak_max']} MB resident "
            f"(<24 GB). The corpus-derived pipeline works at the validated scale -> the decisive Step-2 4-bridge "
            f"1,280-concept de-risk is unblocked."
        )
    else:
        misses = []
        if not bars["recall_pass"]:
            misses.append(f"recall {bars['recall_per_seed']} (bar {a.recall_bar})")
        if not bars["moat_pass_0FA"]:
            misses.append(f"MOAT false-accepts {bars['moat_false_accepts_per_seed']} (must be 0)")
        if not bars["generalization_pass"]:
            misses.append(f"generalization ({a.gen_reference}) "
                          f"{[round(x,3) for x in bars['generalization_per_seed']]} (bar {a.gen_bar})")
        if not bars["derangement_pass"]:
            misses.append(f"derangement did NOT collapse {bars['derangement_collapses_per_seed']}")
        if not bars["frozen_pass"]:
            misses.append(f"frozen-brain competence NOT flat {bars['frozen_competence_flat_per_seed']}")
        verdict = (
            f"STEP 1 PARTIAL/MISS (first REAL-corpus 320 data point = a FINDING, not a failure) — miss: "
            f"{'; '.join(misses)}. {_both_str}. Calibrated per-bridge rate: "
            f"{calibrated['mean_learn_seconds_per_bridge']}s/320 concepts ({calibrated['windows_per_second']} "
            f"win/s), VRAM {calibrated['vram_resident_mb_peak_max']} MB. NOTE the gen-reference fix (2026-06-25): "
            f"the original 0.153 was the g20 SHARDING taxonomy (~33% co-occurrence-INCOHERENT adjective/function "
            f"categories); the 'coherent' reference (validated TAXONOMY_40x8 + coherent g20 entity/verb domains) "
            f"is the apples-to-apples reference the stream cortex scored 0.91 against. HONEST CAVEAT: only "
            f"{coherent_coverage['coverage_frac_gen_usable']*100:.0f}% of the frequency-derived vocab is "
            f"coherent-clusterable (TinyStories' most-frequent content words are ~1/3 adjective/function words a "
            f"distributional cortex cannot cluster) -- gen is scored over that gen-usable subset."
        )

    res = {
        "step": 1,
        "go": go,
        "verdict": verdict,
        "backend": os.environ.get("SIM_BACKEND"),
        "seeds": seeds,
        "config": {"n_concepts": a.n_concepts, "n_hub": a.n_hub, "n_per": a.n_per,
                   "window_steps": a.window_steps, "max_windows": a.max_windows, "D": a.D,
                   "n_facts": a.n_facts, "recall_bar": a.recall_bar, "gen_bar": a.gen_bar,
                   "gen_reference": a.gen_reference, "vocab_filter": a.vocab_filter,
                   "corpus_paths": list(corpus_paths), "n_corpora": len(corpus_paths)},
        "curriculum": curr_report,
        "curriculum_developmental_order": vocab,
        "curriculum_category_ids": cat_ids.tolist(),
        "curriculum_category_names": cat_names,
        "curriculum_frequencies": freqs,
        "coherent_gen_reference_coverage": coherent_coverage,
        "bars": bars,
        "calibrated_per_bridge_rate": calibrated,
        "per_seed": per_seed,
        "notes": [
            "ENGINEERING PIECE: the curriculum is DERIVED from corpus frequency (top-N content words, high-freq "
            "first = developmental order), NOT the hardcoded syllabus. CONTENT word = a member of the independent "
            "g20_vocab_spec_2048 semantic taxonomy that is not a stopword (function-word hubs are the context "
            "dimension, not targets).",
            "VOCAB-FILTER FIX (2026-06-25, per research/findings/raw/_curriculum_gen_miss_REAL_scoping.md option "
            "#1, vocab_filter=%r): the original Step-1 gen 0.153 was DECISIVELY diagnosed to the VOCAB SELECTION "
            "-- frequency-ranking over the FULL g20 taxonomy put ~48%% distributionally-FLAT adjective/function/"
            "emotion words at the TOP of the 320 (hey/very/big/happy/there... co-occur with everything -> "
            "near-uniform codes that ALSO homogenize the genuine entity codes; the SAME pipeline+corpus+"
            "normalization scored 0.91 on the curated CONTENT words). The FIX ('content', the default): "
            "frequency-rank but keep ONLY co-occurrence-COHERENT CONTENT words (entities + verbs per "
            "_coherent_category_map / COHERENT_G20_DOMAINS); the flat adjective/function/emotion words are "
            "EXCLUDED from the TARGET set and remain context HUBS (n_hub unchanged). 'all' = the old freq-top-N "
            "(provenance, gen 0.153); 'curated' = the validated TAXONOMY_40x8 positive control (~0.91). The "
            "filter is a-priori (the coherent map is fixed BEFORE the run, never tuned on the gen score -- "
            "anti-cheat #2)." % a.vocab_filter,
            "INDEPENDENCE (load-bearing): the generalization reference S_true is the a-priori taxonomy "
            "category-block matrix, NEVER corpus-derived. The corpus only sets WHICH words (by frequency) + their "
            "co-occurrence codes; the category labels come from the independent taxonomy.",
            "SUBSTRATE: the validated on-bridge stream cortex (build_stream_bridge) -- ONE bridge, hub+target "
            "regions, rate-Hebbian co-occurrence learned in the synapses as the brain hears the corpus "
            "window-by-window. Reuse-by-import, NO sim/ edit.",
            "BARS: recall/moat on the production RFPhasorComposer with the stream-LEARNED grounded codes; "
            "generalization via heldout-category cosine + derangement control; frozen-brain (plasticity off) "
            "competence-flat control = the codes are LEARNED, not smuggled.",
            "GEN-REFERENCE FIX (2026-06-25, per research/findings/raw/_curriculum_gen_miss_scoping.md): the "
            "original Step-1 gen 0.153 scored against the g20_vocab_spec_2048 SHARDING taxonomy, ~33% of whose "
            "categories are adjective/function clusters (texture/color/size adjectives + abstract/spatial/time/"
            "quantity/discourse words) whose members modify DIFFERENT nouns / scatter -> a distributional Hebbian "
            "co-occurrence cortex provably cannot cluster them (Pearson 0.07 vs the validated +0.41-0.52). The "
            "'coherent' reference (default) = the INDEPENDENT, a-priori, co-occurrence-COHERENT taxonomy "
            "(validated stream_taxonomy_320.TAXONOMY_40x8 + coherent g20 entity/verb domains), the apples-to-apples "
            "reference the validated stream cortex scored 0.91 against. BOTH numbers are reported "
            "(generalization_sharding / generalization_coherent) -- the swap is a correctly-scoped re-measure, "
            "NOT hiding the 0.153 (anti-cheat #5).",
            "GEN-REFERENCE INDEPENDENCE: the 'coherent' labels are a-priori (TAXONOMY_40x8 is a hand-curated "
            "semantic taxonomy asserted distinct from the corpus; the coherent g20 domains are the spec's "
            "entity/verb clusters) -- NEVER corpus-derived. Only WHICH words are scored is restricted to the "
            "gen-usable coherent subset (a category with >=2 members of the vocab). The frequency-derived "
            "REAL-corpus vocab is KEPT intact (the north-star); the incoherent adjective/function words are simply "
            "not gen-usable (a co-occurrence cortex has no category for them) and are reported as coverage.",
            "HONEST COVERAGE CAVEAT: only ~%d%% of the frequency-derived 320 is coherent-clusterable -- "
            "TinyStories' most-frequent content words are ~1/3 adjectives + function words a distributional cortex "
            "cannot cluster by category. The >=90%% coverage one might hope for is UNREACHABLE over the frequency "
            "vocab while keeping the real-corpus words; gen is scored over the gen-usable coherent subset and the "
            "coverage is reported. (Raising coverage would require swapping to the CURATED TAXONOMY_40x8 vocab -- "
            "which the prompt explicitly forbids: real-corpus learning is the north-star.)"
            % round(coherent_coverage["coverage_frac_gen_usable"] * 100),
        ],
    }
    res["wall_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    print(f"  VERDICT: {res['verdict']}", flush=True)
    print(f"  [saved] {a.out}  (wall {res['wall_seconds']}s)\n{'='*100}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
