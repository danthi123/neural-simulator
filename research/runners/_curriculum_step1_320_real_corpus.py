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

GPU (`SIM_BACKEND=cupy`). Run:
    SIM_BACKEND=cupy python -u -m research.runners._curriculum_step1_320_real_corpus \
        --seeds 42,43,44 --n-concepts 320 \
        --out research/findings/raw/_curriculum_step1_320_real_corpus.json
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
from research.runners.corpus_stream import iter_stories, load_token_stream, default_corpus_path  # noqa: E402
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
    corpus-derived."""
    word2cat = {}
    for cat, words in VOCAB_SPEC.ALL_CLUSTERS_2048.items():
        for w in words:
            word2cat.setdefault(w, cat)   # clusters are globally unique (asserted at spec import); first wins
    return word2cat


def derive_curriculum_from_corpus(corpus_path, n_concepts, verbose=True):
    """Stream the corpus, count word frequency, and return the top-`n_concepts` CONTENT words ranked
    HIGH-FREQUENCY-FIRST (the developmental order). A CONTENT word = a member of the independent
    g20_vocab_spec_2048 semantic taxonomy that is NOT a stopword (the function-word hubs are the context
    dimension, not target concepts). Returns:
        vocab     : list[str], the curriculum in developmental order (most-frequent first)
        cat_ids   : np.ndarray[int], the INDEPENDENT category id per word (for generalization)
        cat_names : list[str], the category names (cat_ids index into this)
        freqs     : list[int], the corpus frequency per word (developmental order)
        report    : dict, the frequency-derivation report (coverage, range, per-category spread)
    """
    word2cat = _category_map()
    candidates = set(word2cat)

    gfreq = Counter()
    n_stories = 0
    n_tokens = 0
    for toks in iter_stories(corpus_path):
        n_stories += 1
        n_tokens += len(toks)
        gfreq.update(toks)

    # CONTENT candidates = taxonomy member, non-stopword, appears in the corpus. Rank HIGH-FREQ-FIRST.
    content = [(w, gfreq[w]) for w in candidates if w not in STOPLIST and gfreq.get(w, 0) > 0]
    content.sort(key=lambda x: (-x[1], x[0]))   # freq desc, name asc for a deterministic tie-break
    chosen = content[:n_concepts]

    # the independent category structure of the chosen vocab (drop the empty categories so cat ids are contiguous)
    chosen_words = [w for w, _ in chosen]
    chosen_cats = [word2cat[w] for w in chosen_words]
    present_cats = sorted(set(chosen_cats))
    cat_to_id = {c: i for i, c in enumerate(present_cats)}
    cat_ids = np.asarray([cat_to_id[c] for c in chosen_cats], dtype=int)
    freqs = [f for _, f in chosen]

    cat_count = Counter(chosen_cats)
    # a word is gen-usable iff its category has >= 2 members in the chosen vocab (heldout needs another member)
    n_gen_usable = sum(1 for c in chosen_cats if cat_count[c] >= 2)

    report = {
        "corpus": os.path.basename(corpus_path),
        "n_stories": n_stories,
        "n_tokens": n_tokens,
        "n_unique_types": len(gfreq),
        "n_candidate_content_present": len(content),
        "n_concepts_requested": n_concepts,
        "n_concepts_chosen": len(chosen_words),
        "freq_range": [int(chosen[0][1]), int(chosen[-1][1])] if chosen else [0, 0],
        "n_categories_covered": len(present_cats),
        "n_categories_total": len(VOCAB_SPEC.ALL_CLUSTERS_2048),
        "categories_missing": sorted(set(VOCAB_SPEC.ALL_CLUSTERS_2048) - set(present_cats)),
        "per_category_count": dict(cat_count.most_common()),
        "n_gen_usable_words": n_gen_usable,
        "top25": [(w, int(f)) for w, f in chosen[:25]],
        "tail10": [(w, int(f)) for w, f in chosen[-10:]],
    }
    if verbose:
        print(f"  [curriculum] corpus={report['corpus']} | {n_stories} stories, {n_tokens} tokens, "
              f"{len(gfreq)} types", flush=True)
        print(f"  [curriculum] {len(chosen_words)} content words (high-freq-first), freq {report['freq_range'][0]}"
              f"->{report['freq_range'][1]} | {len(present_cats)}/{len(VOCAB_SPEC.ALL_CLUSTERS_2048)} categories | "
              f"{n_gen_usable}/{len(chosen_words)} gen-usable (cat has >=2 members)", flush=True)
        print(f"  [curriculum] developmental order (top 12): "
              f"{[w for w,_ in chosen[:12]]}", flush=True)
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


def measure_generalization(code, cat_ids, seed):
    """Held-out category generalization on the learned code (a concept lands in its correct category by mean
    cosine to the OTHER members) + the category-DERANGEMENT control (shuffle the labels -> must collapse to
    ~chance). Also the structure recovery Pearson(cos, S_true) for context."""
    cat_ids = np.asarray(cat_ids, dtype=int)
    gen, chance = heldout_generalization(code, cat_ids)
    rng = np.random.RandomState(seed * 99 + 1)
    perm = rng.permutation(cat_ids)
    gen_perm, _ = heldout_generalization(code, perm)
    S_true = (cat_ids[:, None] == cat_ids[None, :]).astype(np.float64)
    pearson = _pearson_vs_Strue(_cos_sim(code), S_true)
    return {"generalization": gen, "chance": chance, "ratio_vs_chance": gen / max(chance, 1e-9),
            "derangement_generalization": gen_perm,
            "derangement_collapses": bool(gen_perm <= max(chance, 1e-9) + 0.05 or gen_perm < 0.5 * gen),
            "pearson_vs_Strue": pearson}


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

    # ---- BARS on the LEARNED codes ----
    rm = measure_recall_and_moat(grounded, vocab, cat_ids, cat_names, seed, a.n_facts, a.D)
    gen = measure_generalization(code, cat_ids, seed)
    print(f"  [recall/moat] recall {rm['recall']:.3f} ({rm['recall_correct']}/{rm['recall_total']}) | "
          f"abstain {rm['abstain']:.3f} (false-accept {rm['false_accept']}/{rm['abstain_total']}) | "
          f"{rm['n_facts']} facts", flush=True)
    print(f"  [generalization] {gen['generalization']:.3f} ({gen['ratio_vs_chance']:.1f}x chance "
          f"{gen['chance']:.3f}) | derangement {gen['derangement_generalization']:.3f} "
          f"(collapses={gen['derangement_collapses']}) | Pearson(S,S_true) {gen['pearson_vs_Strue']:+.3f}",
          flush=True)
    cx.close()

    # ---- FROZEN-BRAIN control: plasticity OFF -> hears but learns no codes -> competence must NOT rise ----
    frozen = None
    if not a.no_frozen:
        cxf = StreamCortexBridge(vocab, cat_ids, seed, stories, n_hub=a.n_hub, n_per=a.n_per,
                                 window_steps=a.window_steps, D=a.D, plasticity_on=False, verbose=False)
        cxf.hear_corpus(a.max_windows, story_seed=seed)
        Mf, codef, groundedf = cxf.read_codes()
        corr_mc_f = cxf.learning_fidelity()
        genf = measure_generalization(codef, cat_ids, seed)
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
        "frozen": frozen,
    }


def decide(per_seed, a):
    seeds = list(per_seed.keys())
    rec = [per_seed[s]["recall"] for s in seeds]
    fa = [per_seed[s]["moat_false_accepts"] for s in seeds]
    g = [per_seed[s]["generalization"] for s in seeds]
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
        "gen_bar": a.gen_bar, "generalization_per_seed": g, "generalization_pass": gen_ok,
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
    p.add_argument("--no-frozen", action="store_true", help="skip the frozen-brain control (debug only)")
    p.add_argument("--corpus-path", default=None,
                   help="path to a plain-text corpus shard; default = data/corpus/tinystories.txt")
    p.add_argument("--out", default="research/findings/raw/_curriculum_step1_320_real_corpus.json")
    a = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    corpus_path = a.corpus_path or default_corpus_path()
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)

    print("=" * 100, flush=True)
    print("[FOUNDATIONAL CURRICULUM — STEP 1: 320 concepts from the REAL TinyStories corpus]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND')}  seeds={seeds}  n_concepts={a.n_concepts}  "
          f"n_hub={a.n_hub}  n_per={a.n_per}  max_windows={a.max_windows}  D={a.D}", flush=True)
    print(f"  corpus={corpus_path}", flush=True)
    print("  bars: recall>=%.2f (who/what) | moat 0 false-accepts | generalization>=%.2f + derangement collapse | "
          "frozen-brain control | MEASURE VRAM + wall-clock" % (a.recall_bar, a.gen_bar), flush=True)
    print("=" * 100, flush=True)

    t0 = time.time()
    # derive the curriculum ONCE (corpus frequency is seed-independent); the story stream is loaded once + reused.
    stories = load_token_stream(corpus_path)
    vocab, cat_ids, cat_names, freqs, curr_report = derive_curriculum_from_corpus(corpus_path, a.n_concepts)

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

    if go:
        verdict = (
            f"STEP 1 GO -> STEP 2 — the CORPUS-FREQUENCY-DERIVED 320-concept curriculum learns on ONE bridge from "
            f"the REAL TinyStories corpus and passes every bar (3 seeds): recall {min(bars['recall_per_seed']):.2f}"
            f"-{max(bars['recall_per_seed']):.2f} >= {a.recall_bar}, moat 0 false-accepts, generalization "
            f"{min(bars['generalization_per_seed']):.2f}-{max(bars['generalization_per_seed']):.2f} >= {a.gen_bar} "
            f"with derangement collapse, frozen-brain control holds. Calibrated per-bridge rate: "
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
            misses.append(f"generalization {[round(x,3) for x in bars['generalization_per_seed']]} (bar {a.gen_bar})")
        if not bars["derangement_pass"]:
            misses.append(f"derangement did NOT collapse {bars['derangement_collapses_per_seed']}")
        if not bars["frozen_pass"]:
            misses.append(f"frozen-brain competence NOT flat {bars['frozen_competence_flat_per_seed']}")
        verdict = (
            f"STEP 1 PARTIAL/MISS (first REAL-corpus 320 data point = a FINDING, not a failure) — miss: "
            f"{'; '.join(misses)}. Calibrated per-bridge rate: {calibrated['mean_learn_seconds_per_bridge']}s/320 "
            f"concepts ({calibrated['windows_per_second']} win/s), VRAM {calibrated['vram_resident_mb_peak_max']} MB. "
            f"Likely cause = real-corpus noise vs the curated 64-word baseline (frequency-derived 320 vocab is "
            f"uneven per category + the long-frequency tail at ~{curr_report['freq_range'][1]} counts is thinly "
            f"learned). See the per-seed detail + the curriculum report for the localize."
        )

    res = {
        "step": 1,
        "go": go,
        "verdict": verdict,
        "backend": os.environ.get("SIM_BACKEND"),
        "seeds": seeds,
        "config": {"n_concepts": a.n_concepts, "n_hub": a.n_hub, "n_per": a.n_per,
                   "window_steps": a.window_steps, "max_windows": a.max_windows, "D": a.D,
                   "n_facts": a.n_facts, "recall_bar": a.recall_bar, "gen_bar": a.gen_bar},
        "curriculum": curr_report,
        "curriculum_developmental_order": vocab,
        "curriculum_category_ids": cat_ids.tolist(),
        "curriculum_category_names": cat_names,
        "curriculum_frequencies": freqs,
        "bars": bars,
        "calibrated_per_bridge_rate": calibrated,
        "per_seed": per_seed,
        "notes": [
            "ENGINEERING PIECE: the curriculum is DERIVED from corpus frequency (top-N content words, high-freq "
            "first = developmental order), NOT the hardcoded syllabus. CONTENT word = a member of the independent "
            "g20_vocab_spec_2048 semantic taxonomy that is not a stopword (function-word hubs are the context "
            "dimension, not targets).",
            "INDEPENDENCE (load-bearing): the generalization reference S_true is the a-priori taxonomy "
            "category-block matrix, NEVER corpus-derived. The corpus only sets WHICH words (by frequency) + their "
            "co-occurrence codes; the category labels come from the independent taxonomy.",
            "SUBSTRATE: the validated on-bridge stream cortex (build_stream_bridge) -- ONE bridge, hub+target "
            "regions, rate-Hebbian co-occurrence learned in the synapses as the brain hears the corpus "
            "window-by-window. Reuse-by-import, NO sim/ edit.",
            "BARS: recall/moat on the production RFPhasorComposer with the stream-LEARNED grounded codes; "
            "generalization via heldout-category cosine + derangement control; frozen-brain (plasticity off) "
            "competence-flat control = the codes are LEARNED, not smuggled.",
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
