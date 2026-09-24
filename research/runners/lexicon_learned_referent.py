"""LEXICON v1 — a corpus-LEARNED open-vocabulary REFERENT (noun-category) SCORE, host-computed, spike-RELAYED (2026-09-23).

⛔ RELABELLED 2026-09-23 after adversarial review (see `_lexicon_learned_referent_derisk.py` AMENDMENT A1): the
"two-pool WTA" below is a RELAY, not a winner-take-all. The pools of `_gap3_spiking_feature_compat_derisk._build`
are uncoupled (internal_density 0, one weight-0 pathway) and `classify()` drives exactly one of them with a fixed
current chosen by `np.sign(host label-spread score)`. The category is computed by host label-spreading; the spikes
only carry its sign. The SPIKING decision (coupled WTA, graded drive through Hebbian-learned frame->category
synapses) is v2: `research/runners/lexicon_spiking_frame_category.py`, which is what the production flag now uses.
The seed curriculum is 38 hand nouns (not "40" as first written) + 37 non-nouns.

WHY (language lane E, lexicon): the D6 multi-referent working-memory organ (`d6_multiref_wm_production_organ.py`)
decides WHICH tokens of an utterance are discourse referents with a hand-typed 48-noun table (`_REFERENT_NOUNS`,
its own declared "vocab-ceiling residual"). A referent off the table is invisible to the buffer: the load-bearing
battery's `held` turn ("the wolf watches the owl") reads `wm-binding-advanced` NOT-EXERCISED because "owl" is not
in the table, so the turn names only ONE referent and the organ goes out of scope. The same table feeds the
activity-silent WM organ (`activity_silent_wm_production_organ._extract_refs`). This is the same vocab-ceiling class
the comprehension organs had for ANIMACY / VERB_SELECTS, which were converted to corpus-learned cues (2026-08-26/27).

MECHANISM (developmental biology of word-category learning): infants and children assign novel words to the NOUN
category from their DISTRIBUTIONAL FRAMES — the immediately-adjacent function words ("the __ is", "a __ and") —
before they know the words' meanings (Redington, Chater & Finch 1998, Cognitive Science 22:425; Mintz 2003
"frequent frames", Cognition 90:91; Gerken et al.: determiner-noun co-occurrence in infancy). The learner here is
that mechanism, not a POS tagger:
  1. each frequent content word gets a POSITIONAL-CONTEXT vector: counts of which of the top-C most frequent
     tokens (function words included — they ARE the frames) occur at offsets -2, -1, +1, +2;
  2. PPMI-weight it; cosine-similarity kNN graph between words (words used in the same frames are linked);
  3. Zhou label-spreading from a SMALL seed set: the D6 hand table's own common nouns (+1) and a small list of
     obvious verbs/adjectives (-1). The hand table becomes the initial-education SEED, not the SCOPE — every other
     word's category is inferred from real English usage (TinyStories), never labelled.

SPIKE RELAY (NOT a spiking decision — see the relabel above): the per-word continuous category score and its sign
are the offline label-propagation scaffold, and the sign IS the decision. It is relayed through two UNCOUPLED pools
on a real `SimulationBridge` (`_gap3_spiking_feature_compat_derisk._build`): the sign drives ONE pool (REF = pool A,
NON-REF = pool B) with a fixed current, the pools run `steps` ticks, and the pool that fired is read back. A word off
the learned graph gets no drive -> tie -> ABSTAIN (not a referent: the no-confab default). LESION
(`set_lesion(True)`) zeroes both pools' drive -> every word abstains -> the organ's referent scope reverts exactly
to the hand table (byte-identical to the flag being off, for every word).

HONEST RESIDUALS (declared, not hidden):
  * The category SCORE is offline label-spreading (host numpy), exactly the learned-animacy precedent; only the
    decision is spiking. A fully-spiking distributional category learner (Hebbian frame->category synapses) is the
    next rung.
  * NOUN-hood is learned, not REFERENT-hood: abstract nouns ("time", "idea") and plural/verb-ambiguous forms
    ("watches") can be admitted as referents. Concrete-vs-abstract needs grounding (the multimodal ATL hub).
  * The seed labels are a small hand list (the initial-education curriculum): 38 common nouns from the old table
    (`HAND_NOUN_SEEDS`; the table's 8 proper names are excluded) + 37 `NONNOUN_SEEDS` below.
  * Corpus is TinyStories (child-directed-like register), capped at `max_chars`.

Run (numpy CPU, ~1 min to build the graph; each spiking read ~25 steps on an 80-neuron bridge):
    SIM_BACKEND=numpy python -m research.runners.lexicon_learned_referent --smoke
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._comprehension_learned_animacy_cue_derisk import (  # noqa: E402
    load_tokens,
    build_vocab,
    label_spread,
    shuffle_graph,
    _STOP as _CONTENT_STOP,
)

_DEFAULT_CORPUS = os.path.join(_REPO, "data", "corpus", "tinystories.txt")

# The D6 organ's hand table, COMMON NOUNS only (the proper names are handled by D6's own capitalization rule and are
# not a distributional category). Kept as a literal copy (not an import) so this module has no import-time
# dependency on the organ; `tests`/the de-risk assert it stays equal to the organ's table minus the names.
HAND_NOUN_SEEDS = [
    "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "pig", "mouse", "rabbit", "fox", "wolf", "bear", "lion",
    "man", "woman", "boy", "girl", "child", "baby", "king", "queen", "doctor", "teacher", "farmer", "friend",
    "car", "ball", "book", "tree", "house", "box", "cup", "table", "chair", "door", "key", "phone",
]
# Obvious non-noun content words (verbs in several inflections + adjectives + adverbs). The initial-education
# negative seed set. Chosen as unambiguous-in-use words, NOT tuned against the eval set (the eval set excludes them).
NONNOUN_SEEDS = [
    "run", "ran", "jump", "jumped", "play", "played", "eat", "ate", "sleep", "slept", "sing", "sang", "walk",
    "walked", "find", "found", "give", "gave", "open", "opened", "happy", "sad", "big", "small", "little", "red",
    "blue", "funny", "scared", "angry", "quickly", "slowly", "loudly", "together", "never", "always", "again",
]


def build_frame_graph(tokens, vocab, top_c=150, offsets=(-2, -1, 1, 2), knn=10):
    """Positional-context (frame) similarity graph over `vocab` (Redington et al. 1998). Returns a symmetric
    non-negative weight matrix W (V x V), kNN-sparsified cosine similarity of PPMI-weighted context vectors."""
    cnt = collections.Counter(tokens)
    ctx = [w for w, _ in cnt.most_common(top_c)]
    cidx = {w: i for i, w in enumerate(ctx)}
    vidx = {w: i for i, w in enumerate(vocab)}
    V, C, P = len(vocab), len(ctx), len(offsets)
    M = np.zeros((V, P * C), dtype=np.float64)
    n = len(tokens)
    for i, t in enumerate(tokens):
        vi = vidx.get(t)
        if vi is None:
            continue
        for pj, off in enumerate(offsets):
            j = i + off
            if 0 <= j < n:
                ci = cidx.get(tokens[j])
                if ci is not None:
                    M[vi, pj * C + ci] += 1.0
    total = M.sum()
    row = M.sum(1, keepdims=True)
    col = M.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * total) / (row @ col))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[pmi < 0] = 0.0
    norm = np.linalg.norm(pmi, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    X = pmi / norm
    S = X @ X.T
    np.fill_diagonal(S, 0.0)
    S[S < 0] = 0.0
    W = np.zeros_like(S)
    k = min(knn, max(V - 1, 1))
    nbr = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(V), k)
    W[rows, nbr.ravel()] = S[rows, nbr.ravel()]
    W = np.maximum(W, W.T)
    return W


def build_topical_graph(tokens, vocab, window=4):
    """ATTRIBUTION CONTROL: the learned-animacy mechanism's TOPICAL window co-occurrence PPMI graph (no position).
    If positional frames carry noun-hood, this graph should do worse than `build_frame_graph`."""
    from research.runners._comprehension_learned_animacy_cue_derisk import cooccur_ppmi
    return cooccur_ppmi(tokens, vocab, window=window)


class LearnedReferentLexicon:
    """v1 open-vocabulary referent (noun-category) detector: offline frame-graph label-spreading SCORE whose sign is
    RELAYED through one of two uncoupled spiking pools (host-computed category, spike-relayed). See module docstring.

    DEPLOYMENT (`k_seed=None`): seeds with ALL `HAND_NOUN_SEEDS` / `NONNOUN_SEEDS` in the corpus vocab.
    CROSS-VALIDATION (`k_seed=int`): subsamples k seeds per class with rng(`cv_seed`); `label_permute=True` shuffles
    which seed gets which label (anti-cheat); `graph_kind` in {"frame","topical"}; `shuffle_control` permutes the
    graph's edge weights (destroys the corpus structure)."""

    def __init__(self, seed: int = 42, corpus_path: str | None = None, max_chars: int = 8_000_000,
                 top_v: int = 2000, n_feat: int = 40, drive: float = 500.0, steps: int = 25,
                 k_seed: int | None = None, cv_seed: int | None = None, shuffle_control: bool = False,
                 label_permute: bool = False, graph_kind: str = "frame", tokens=None, vocab=None, W=None):
        self.seed = int(seed)
        self.drive = float(drive)
        self.steps = int(steps)
        self.n_feat = int(n_feat)
        self._lesioned = False
        self._cache: dict = {}
        if tokens is None:
            tokens = load_tokens(corpus_path or _DEFAULT_CORPUS, max_chars)
        if vocab is None:
            vocab, _ = build_vocab(tokens, top_v)
            for w in HAND_NOUN_SEEDS + NONNOUN_SEEDS:        # seed words must be on the graph if the corpus has them
                if w not in vocab and w in set(tokens[:200000]):
                    vocab.append(w)
        if W is None:
            W = build_frame_graph(tokens, vocab) if graph_kind == "frame" else build_topical_graph(tokens, vocab)
        rng_key = cv_seed if cv_seed is not None else seed
        if shuffle_control:
            W = shuffle_graph(W, np.random.default_rng(rng_key))
        idx = {w: i for i, w in enumerate(vocab)}
        pos_pool = [w for w in HAND_NOUN_SEEDS if w in idx]
        neg_pool = [w for w in NONNOUN_SEEDS if w in idx]
        if k_seed is not None:
            rng = np.random.default_rng(rng_key)
            pos_pool = list(rng.permutation(pos_pool))[:k_seed]
            neg_pool = list(rng.permutation(neg_pool))[:k_seed]
        seeds = [(w, +1.0) for w in pos_pool] + [(w, -1.0) for w in neg_pool]
        if label_permute:
            labs = np.array([s for _, s in seeds])
            np.random.default_rng(rng_key + 1).shuffle(labs)
            seeds = [(w, float(s)) for (w, _), s in zip(seeds, labs)]
        y = np.zeros(len(vocab))
        for w, s in seeds:
            y[idx[w]] = s
        f = label_spread(W, y)
        self.scores = {w: float(f[i]) for w, i in idx.items()}
        self.vocab = vocab
        self.seed_words = [w for w, _ in seeds]
        self.W = W
        self._b = None

    # --- spiking decision -------------------------------------------------------------------------------------
    def _bridge(self):
        if self._b is None:
            from research.runners._gap3_spiking_feature_compat_derisk import _build as _build_feature_pools
            self._b = _build_feature_pools(self.seed, n_feat=self.n_feat)
            self._n = self._b.core_config.num_neurons
            # pool A (F_anim slot) = REFERENT, pool B (F_inanim slot) = NON-REFERENT: the two-pool bridge is reused
            # verbatim; only the ROLE of each pool differs.
            self._pa = np.asarray(list(self._b.region_manager.indices("F_anim")), int)
            self._pb = np.asarray(list(self._b.region_manager.indices("F_inanim")), int)
        return self._b

    def _read_margin(self, sign_val: float) -> float:
        from sim.backend import to_host, from_host
        b = self._bridge()
        if getattr(b, "cp_izh_c_reset", None) is not None:
            b.cp_membrane_potential_v[:] = b.cp_izh_c_reset
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        for a in ("cp_conductance_g_e", "cp_conductance_g_i"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = 0.0
        mult = 0.0 if self._lesioned else 1.0
        cur = np.zeros(self._n)
        cur[self._pa] += self.drive * max(sign_val, 0.0) * mult
        cur[self._pb] += self.drive * max(-sign_val, 0.0) * mult
        dev = from_host(cur.astype(np.float64))
        ra = rb = 0.0
        for _ in range(self.steps):
            b.cp_external_input_current[:] = dev
            b._run_one_simulation_step()
            fs = np.asarray(to_host(b.cp_firing_states))
            ra += float(fs[self._pa].mean())
            rb += float(fs[self._pb].mean())
        return ra - rb

    def set_lesion(self, on: bool = True) -> None:
        self._lesioned = bool(on)

    def offline_sign(self, word: str):
        s = self.scores.get(word)
        if s is None or abs(s) < 1e-12:
            return None
        return bool(s > 0)

    def classify(self, word: str):
        """True (referent) / False (non-referent) / None (abstain: off-graph or lesioned). Decision = the host sign, relayed."""
        key = (word, self._lesioned)
        if key in self._cache:
            return self._cache[key]
        s = self.scores.get(word)
        if s is None or abs(s) < 1e-12:
            out = None
        else:
            m = self._read_margin(float(np.sign(s)))
            out = None if m == 0.0 else bool(m > 0)
        self._cache[key] = out
        return out

    def is_referent(self, word: str) -> bool:
        return self.classify(word) is True


_LEXICON: LearnedReferentLexicon | None = None


def get_lexicon(seed: int = 42) -> LearnedReferentLexicon:
    """The process-shared DEPLOYMENT lexicon (built once, lazily)."""
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = LearnedReferentLexicon(seed=seed)
    return _LEXICON


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corpus", default=None)
    a = ap.parse_args()
    lex = LearnedReferentLexicon(seed=42, corpus_path=a.corpus)
    for w in ("owl", "monkey", "garden", "cookie", "watches", "jumped", "beautiful", "slowly", "the"):
        print(w, "offline", lex.offline_sign(w), "spiking", lex.classify(w))
