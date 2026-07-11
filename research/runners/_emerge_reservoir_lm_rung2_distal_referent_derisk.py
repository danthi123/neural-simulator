"""EMERGE-RESERVOIR-LM RUNG 2 -- condition the Rung-1 on-bridge spiking reservoir next-token generator on a DISTAL REFERENT
held in a single-slot WORKING-MEMORY / TOPIC buffer, so a referent introduced FAR back (beyond the reservoir's fading-memory
horizon) still determines a later continuation token.

WHY THIS IS THE RIGHT RUNG. Rung-1 (`_emerge_reservoir_lm_derisk`) showed a FIXED on-bridge spiking reservoir (the EMERGE-82
`OnBridgeLSM`) + a SHALLOW one-step-delta read-out is an emergent, no-BPTT next-token LM that BEATS the bigram via fading
memory. But a reservoir has FADING memory (echo-state/LSM discipline): a cue introduced far enough back is forgotten. Real
discourse needs a NON-fading store -- a topic/working-memory slot (Grosz-Sidner attentional focus; the prefrontal WM latch;
EMERGE-85's theta-gamma multiplex, but here a GENERAL single-slot referent latch, not the mirror-pair number math). RUNG 2
adds ONE gated slot that LATCHES the discourse TOPIC when a boundary marker is seen and holds it unfading; the read-out is
conditioned on the CONCATENATED [reservoir-state | topic-buffer] feature. The claim: at a filler distance D* BEYOND the
reservoir's horizon (where the reservoir alone is at chance on the distal-referent-dependent token), the buffer RESTORES the
dependency -- a load-bearing WM slot, not a reservoir the buffer merely decorates.

THE TASK (a controlled discourse-referent grammar; content from the EMERGE-62 inventories). Each document is ONE continuous
token sequence (one reservoir wash unit):
  * a boundary marker `DOC`, then `the <TOPIC> can <v> .` -- the TOPIC subject is introduced in the unique position right
    after DOC (so it is identifiable ONLY by POSITION, not by presence).
  * then D filler clauses `the <subj> <v> .`, each about a DIFFERENT same-category subject (SAME animal category as the
    topic). The same-category distractors are ESSENTIAL: they drive the same reservoir subject-dimensions (agreement-
    attraction interference that erodes the topic in the fading memory) AND they put MULTIPLE subjects in the prefix so a
    bag-of-prefix cannot just read the lone subject. Filler subjects are drawn from the full category pool (>= n_distractors
    distinct others guaranteed), so at large D the subject-multiset is ~topic-invariant and a bag is uninformative.
  * a continuation `it goes to the <DEN> .` whose DEPENDENT TOKEN <DEN> is determined by the topic via a per-seed FROZEN
    bijection HOME: subject -> a UNIQUE object (its "den"), which is NEVER mentioned in the intro or any filler -> a
    deterministic single correct target -> clean top-1. The subject is NOT re-named in the continuation ("it" is the
    anaphor), so the dependent token is fixed ONLY by the distally-introduced topic. We SCORE ONLY that dependent position.

THE TOPIC BUFFER (`TopicBuffer`). A single gated slot that latches the referent identity when the `DOC` boundary marker is
seen (the next token IS the topic) and exposes a fixed-length one-hot of the held topic token for every subsequent position
(zeros before any latch). The latch TIMING is triggered by DOC; the SLOT-SCRAMBLE control latches a RANDOM subject's
identity per doc (same dimensionality, wrong content) -- the ordered-slot/scramble idea of EMERGE-85's WMBuffer, but a
GENERAL single-slot token-identity encode (not the 2-valued mirror-pair math). COUPLING: concatenate the buffer one-hot to
the Rung-1 per-token reservoir state before the SAME one-step-local-delta read-out; STANDARDIZE THE RESERVOIR SUB-BLOCK ONLY
(the buffer one-hot is left as-is, mean 0 / std 1 on those dims).

ARMS (like-for-like: same docs, same delta-rule read-out training, only the FEATURE / TARGET differs; every arm scored at
the dependent position):
  * reservoir-ONLY (NO buffer) -- the KEY control (the fading memory).
  * reservoir + buffer.
  * buffer SLOT-SCRAMBLE (latch a random subject per doc) -> must collapse to ~reservoir-only (buffer content useless).
  * REFERENT-SHUFFLE (train the read-out with a DERANGED topic->den mapping) -> must collapse to chance (the buffer holds
    the true topic but the learned map is broken).
  * BAG-of-prefix (delta read-out over unordered prefix token counts) -> must be <= chance (topic not readable from counts).
  * BIGRAM (P(next|prev) at the dependent position) -> must be <= chance (topic not readable from the previous token).

THE CRUX -- a HORIZON SWEEP FIRST (`--horizon-sweep`): sweep the filler distance D and report the reservoir-ONLY dependent
accuracy at each D. The reservoir is known to hold a distal cue for many fillers (EMERGE-81 >= 16; EMERGE-83 primacy), so we
MUST find a D* where reservoir-only dep_acc falls to ~chance (with the same-category distractors doing the interference). The
GO experiment then runs AT D* (or beyond). If reservoir-only NEVER falls to chance even at the largest D, that is an HONEST
finding (the buffer is not load-bearing at this scale) -- reported with named levers (bigger D, more distractors, a leakier/
smaller reservoir, the per_window state feature).

GO GATE (at D* beyond the reservoir horizon, 6-seed): reservoir+buffer dep_acc > reservoir-only dep_acc by MARGIN, AND the
reservoir is near chance at D* (buffer load-bearing), AND slot-scramble ~= reservoir-only AND referent-shuffle ~= chance AND
bag <= chance AND bigram <= chance AND the region is genuinely active. Numbers are HONEST -- do NOT force GO.

HONEST SCOPE. A controlled discourse-referent grammar (a closed template domain), not open prose (R4). The reservoir + input
projection are fixed-random; only the shallow output read-out is learned (the ESN/LSM discipline); the topic buffer is a
functional single-slot latch (the SPIKING theta-gamma / NMDA-WM realization is the follow-on rung). Reuse-by-import (Rung-1
ReservoirStates/Vocab/train_readout/_softmax + EMERGE-82 OnBridgeLSM + EMERGE-61 wash-out + EMERGE-62 inventories); NO `sim/`
edit; NO edit to any existing runner.

Run:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SIM_BACKEND=numpy \
    python -u -m research.runners._emerge_reservoir_lm_rung2_distal_referent_derisk \
        --seeds 42 --n-docs 200 --vocab 24 --epochs 8 --horizon-sweep
  (6-seed sweep -- see the returned command.)
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse-by-import: the Rung-1 on-bridge spiking reservoir + per-token read (ReservoirStates), the closed token Vocab, the
# one-step next-token delta-rule read-out (train_readout -- FEATURE-AGNOSTIC, so it takes the concatenated feature
# unchanged), the softmax, the reservoir-feature standardization, the bigram fit; and the EMERGE-62 content inventories.
import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge_reservoir_lm_derisk import (  # noqa: E402
    ReservoirStates, Vocab, train_readout, _softmax, _standardize_fit, fit_bigram, _N_POOL,
)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge_reservoir_lm_rung2_distal_referent.json"

DOC = "DOC"                 # the discourse boundary marker (opens the buffer WRITE gate: the next token is the topic)
READ_CUE = "it"            # the anaphor that opens the buffer READ gate (Grosz-Sidner pop / the EMERGE-D3 read gate)
PERIOD = "."               # clause delimiter (a token; NOT a wash unit here -- the whole DOC is one wash unit)

# GO thresholds (accuracy scale, at the dependent position).
MARGIN = 0.15              # reservoir+buffer must beat reservoir-only dep_acc by at least this (buffer load-bearing)
COLLAPSE_EPS = 0.08        # a collapsed control (scramble/referent/bag/bigram) must be within this of its target
CHANCE_TOL = 0.12          # reservoir-only "near chance" at D* (the buffer is doing the work, i.e. beyond the horizon)
ACTIVE_MIN = 1e-4          # mean reservoir spike-rate/neuron/step above this = genuinely active


# ---------------------------------------------------------------------------------------------------------------------
# THE CONTROLLED DISCOURSE-REFERENT DOMAIN (content from the EMERGE-62 inventories). A CLOSED domain -> a COMPLETE vocab
# (every subject / den / structural token guaranteed present; NO <unk> for a critical token -- else distinct topics would
# collapse to <unk> and the referent dependency would be unreadable). `--vocab` sizes the category (# subjects K) so the
# closed vocab is ~that size; chance at the dependent position = 1/K (K unique dens).
# ---------------------------------------------------------------------------------------------------------------------
_STRUCTURAL = [DOC, "the", "can", "goes", "to", "it", "its", PERIOD]  # fixed structural tokens ("its" = the unique
                                                                     # pre-den cue so the reservoir can predict the DEN
                                                                     # class; the topic then fixes WHICH den)
_N_VERB = 4                                                       # intro "can <v>" + filler "<v>"


def build_domain(vocab_target):
    """Slice the EMERGE-62 inventories into a closed domain: K subjects (topics), K dens (HOME targets, DISJOINT from the
    filler-clause objects -- fillers here are intransitive so there are no filler objects to collide), n_verb verbs. K is
    sized so the complete closed vocab is ~vocab_target. Returns (subjects, dens, verbs, all_tokens)."""
    overhead = len(_STRUCTURAL) + _N_VERB
    K = max(4, (int(vocab_target) - overhead) // 2)
    K = min(K, len(m62._SUBJECTS), len(m62._OBJECTS))            # bounded by the inventory sizes
    subjects = list(m62._SUBJECTS[:K])
    dens = list(m62._OBJECTS[:K])                                # HOME target dens (bijection targets), never in fillers
    verbs = list(m62._VERBS[:_N_VERB])
    all_tokens = sorted(set(_STRUCTURAL) | set(subjects) | set(dens) | set(verbs))
    return subjects, dens, verbs, all_tokens


def _home_bijection(seed, subjects, dens):
    """A per-seed FROZEN bijection subject -> UNIQUE den (the topic's 'den'). Randomized per seed; fixed within a run."""
    rng = np.random.default_rng(seed * 991 + 7)
    perm = list(dens)
    rng.shuffle(perm)
    return {s: perm[i] for i, s in enumerate(subjects)}


def _deranged_home(seed, subjects, home):
    """A DERANGEMENT of the topic->den mapping (no fixed point) for the REFERENT-SHUFFLE control: topic t is trained to
    predict home[other] (a different subject's den). Every topic's mapping is broken -> at test (true HOME) -> collapse."""
    rng = np.random.default_rng(seed * 733 + 13)
    sperm = list(subjects)
    rng.shuffle(sperm)
    for i in range(len(subjects)):                               # fix any fixed point by a local swap -> guaranteed derangement
        if sperm[i] == subjects[i]:
            j = (i + 1) % len(subjects)
            sperm[i], sperm[j] = sperm[j], sperm[i]
    return {subjects[i]: home[sperm[i]] for i in range(len(subjects))}


def build_discourse_docs(seed, n_docs, D, n_distractors, subjects, home, verbs):
    """Generate n_docs discourse documents at filler distance D. Each = (tokens, dep_t, topic, den). The topic is right
    after DOC; D same-category filler clauses follow (>= n_distractors distinct OTHER subjects guaranteed, the rest drawn
    from the full pool so the subject-multiset is ~topic-invariant); the continuation's dependent token = home[topic],
    which appears NOWHERE else in the doc. dep_t indexes the token whose NEXT token is the den (always 'the')."""
    rng = np.random.default_rng(seed * 100003 + D * 17 + 1)
    docs = []
    K = len(subjects)
    for _ in range(n_docs):
        t = str(subjects[rng.integers(K)])
        others = [s for s in subjects if s != t]
        rng.shuffle(others)
        guaranteed = others[:min(n_distractors, D, len(others))]  # distinct distractors guaranteed to appear (interference)
        fill = list(guaranteed)
        while len(fill) < D:
            fill.append(str(subjects[rng.integers(K)]))          # uniform over the FULL pool (incl topic) -> balanced
        rng.shuffle(fill)
        fill = fill[:D]
        # BAG-CLEAN compensation: the intro names the topic once, which would make the topic the (slightly) most-frequent
        # subject and readable from unordered counts. Remove one topic occurrence from the fillers (replacing it with a
        # random OTHER subject) so the topic's TOTAL count (intro + fillers) matches the distractors' -> the subject
        # multiset is ~topic-invariant -> a bag-of-prefix cannot read the topic from counts.
        if t in fill and others:
            fill[fill.index(t)] = str(others[rng.integers(len(others))])
        # intro: the boundary marker DOC DIRECTLY introduces the TOPIC (the unique position right after DOC), so the
        # buffer's WRITE gate (latch the token after DOC) captures the topic subject -- NOT an intervening determiner.
        tokens = [DOC, t, "can", str(verbs[rng.integers(len(verbs))]), PERIOD]
        for s in fill:
            tokens += ["the", s, str(verbs[rng.integers(len(verbs))]), PERIOD]
        den = home[t]
        # continuation: the anaphor "it" (opens the buffer READ gate) ... "its <DEN>" -- the unique pre-den token "its"
        # marks "a den comes next" (so the reservoir predicts the den CLASS; the topic fixes WHICH den). DEN appears
        # NOWHERE else in the doc; the subject is NOT re-named.
        tokens += [READ_CUE, "goes", "to", "its", den, PERIOD]
        den_index = len(tokens) - 2                               # den is second-to-last (before the final PERIOD)
        dep_t = den_index - 1                                     # the token before den ('its'); target = ids[dep_t+1]
        docs.append((tokens, dep_t, t, den))
    return docs


# ---------------------------------------------------------------------------------------------------------------------
# THE TOPIC BUFFER -- a single gated slot latched by the DOC boundary marker.
# ---------------------------------------------------------------------------------------------------------------------
class TopicBuffer:
    """A single gated WM slot with a TWO-GATE (write/read) discourse-register discipline (the EMERGE-D3 push/pop pattern;
    Grosz-Sidner attentional stack; O'Reilly-Frank PBWM separate input/output gates):
      * WRITE gate -- opened by the `DOC` boundary marker: the NEXT token is latched as the held topic (unfading).
      * READ  gate -- opened by the anaphor `READ_CUE` ('it'): the slot is EXPOSED (its one-hot output) only from the
        anaphor onward. Before the read gate opens (during the D fillers), the exposed feature is ZEROS.
    The read gate is what makes a LINEAR delta read-out learnable here: the buffer one-hot is exposed ONLY across the
    (deterministic) continuation clause, NOT across the D random-subject fillers -- so the topic->den weight is not
    suppressed by filler positions whose target is never the den. `features(ids, latched_id=...)` returns a per-position
    list of V-dim one-hots; `latched_id` overrides WHAT is latched (SLOT-SCRAMBLE = a random subject, same gate timing) --
    a general single-slot token-identity encode (the EMERGE-85 ordered-slot/scramble idea, not the mirror-pair number math)."""

    def __init__(self, vocab, gain=1.0):
        self.vocab = vocab
        self.V = vocab.size
        self.doc_id = vocab.id(DOC)
        self.read_id = vocab.id(READ_CUE)
        self.gain = float(gain)                                  # read-out drive strength of the WM slot (a fixed
                                                                 # architectural constant, NOT label-tuned): a single
                                                                 # one-hot dim must compete with ~600 reservoir dims in
                                                                 # the linear softmax, so the slot drives with `gain`.

    def features(self, ids, latched_id=None):
        """latched_id=None -> latch the ACTUAL token seen at the topic position (the true topic). latched_id=<id> ->
        latch that fixed identity instead (slot-scramble). The slot is EXPOSED only after the READ gate opens."""
        feats = []
        held = None
        expect = False
        read_open = False
        for tok in ids:
            if expect:
                held = tok if latched_id is None else latched_id  # WRITE: latch (true topic, or scrambled identity)
                expect = False
            if tok == self.doc_id:
                expect = True                                     # the NEXT token is the topic
            if tok == self.read_id:
                read_open = True                                  # READ gate opens at the anaphor (the continuation)
            f = np.zeros(self.V, np.float64)
            if held is not None and read_open:
                f[held] = self.gain                               # exposed (drive `gain`) ONLY once the read gate is open
            feats.append(f)
        return feats


# ---------------------------------------------------------------------------------------------------------------------
# Reservoir caching (the expensive bridge forward pass) -- run ONCE per (seed, D); every arm reuses it.
# ---------------------------------------------------------------------------------------------------------------------
def _reservoir_feature(res, U, feature):
    """One reservoir pass (per_window) -> the per-token reservoir feature. running_cumulative == the cumulative MEAN of
    the per_window rates (Rung-1's final_state is exactly counts/steps = cummean of the windows), so ONE per_window pass
    yields BOTH. `feature`:
      * 'both' (DEFAULT) -> concat([per_window | running_cumulative]) -- per_window LOCALIZES the current token (so the
        read-out knows it is at the den-prediction step, essential for a long prefix) AND running_cumulative carries the
        integrated fading MEMORY of the distal topic (so reservoir-only shows the horizon: high at small D, chance at large D).
      * 'per_window' -> the current-token window rate only (localizer, ~no distal memory).
      * 'running_cumulative' -> the cumulative-mean rate only (memory, ~no localization at a long prefix)."""
    pw = np.asarray(res.per_token_states(U, feature="per_window"))          # (L, n_pool)
    if len(pw) == 0:
        return []
    cum = np.cumsum(pw, axis=0) / np.arange(1, len(pw) + 1)[:, None]        # cumulative mean == running_cumulative
    if feature == "per_window":
        S = pw
    elif feature == "running_cumulative":
        S = cum
    else:                                                                   # 'both' (default)
        S = np.concatenate([pw, cum], axis=1)
    return [S[t].copy() for t in range(len(S))]


def _res_cache(res, vocab, docs, feature):
    """Run the fixed reservoir over each doc ONCE. Returns list of (S, ids, dep_t, topic, den) where S[t] is the reservoir
    per-token feature and ids = vocab ids for the doc tokens."""
    out = []
    for (tokens, dep_t, topic, den) in docs:
        S = _reservoir_feature(res, vocab.encode_seq(tokens), feature)
        out.append((S, vocab.ids(tokens), dep_t, topic, den))
    return out


def _standardize_res_block(cache, n_res):
    """Standardize the RESERVOIR sub-block only; leave the buffer one-hot dims as-is (mean 0 / std 1). `cache` = list of
    (states, ids, dep_t) with states[t] = concat([reservoir(n_res) | buffer(V)])."""
    xs = [np.asarray(s)[:, :n_res] for (s, _i, _d) in cache if len(s) > 0]
    allx = np.concatenate(xs, axis=0)
    rmean = allx.mean(0)
    rstd = allx.std(0) + 1e-6
    n_buf = len(cache[0][0][0]) - n_res
    mean = np.concatenate([rmean, np.zeros(n_buf)])
    std = np.concatenate([rstd, np.ones(n_buf)])
    return mean, std


# ---------------------------------------------------------------------------------------------------------------------
# The DEPENDENT-POSITION evaluation (score ONLY the referent-dependent token). `train_readout` (Rung-1) trains over ALL
# positions unchanged (feature-agnostic; it just wants (states, ids) pairs) -- the topic->den mapping is learned from the
# dependent positions in training; eval scores ONLY the dependent position.
# ---------------------------------------------------------------------------------------------------------------------
def _pairs(cache3):
    """(states, ids, dep_t) -> (states, ids) for train_readout (which trains over all positions, feature-agnostic)."""
    return [(s, i) for (s, i, _d) in cache3]


def eval_dep(W, mean, std, cache3, V):
    """Cross-entropy + top-1 accuracy at ONLY the dependent position (ids[dep_t+1] = the den) for each doc."""
    tot = 0.0
    hit = 0
    n = 0
    for (states, ids, dep_t) in cache3:
        x = np.concatenate([(states[dep_t] - mean) / std, [1.0]])
        p = _softmax(W @ x)
        tgt = ids[dep_t + 1]
        tot += -math.log(max(p[tgt], 1e-12))
        hit += int(np.argmax(p) == tgt)
        n += 1
    return tot / max(1, n), hit / max(1, n), n


def _attach_reservoir_only(res_cache):
    """Arm feature = reservoir state only (no buffer). -> (S, ids, dep_t)."""
    return [(S, ids, dep_t) for (S, ids, dep_t, _t, _d) in res_cache]


def _attach_buffer(res_cache, buffer, vocab, subjects, scramble_rng=None):
    """Arm feature = concat([reservoir | topic-buffer one-hot]). scramble_rng!=None -> latch a RANDOM subject per doc
    (slot-scramble). -> (concat_states, ids, dep_t)."""
    out = []
    subj_ids = [vocab.id(s) for s in subjects]
    for (S, ids, dep_t, _t, _d) in res_cache:
        latched = None
        if scramble_rng is not None:
            latched = int(subj_ids[scramble_rng.integers(len(subj_ids))])  # a random (wrong) subject identity
        B = buffer.features(ids, latched_id=latched)
        concat = [np.concatenate([S[t], B[t]]) for t in range(len(ids))]
        out.append((concat, ids, dep_t))
    return out


def _bag_cache(res_cache, V):
    """Arm feature = normalized UNORDERED prefix token-count vector (no reservoir, no order). -> (bag_states, ids, dep_t)."""
    out = []
    for (_S, ids, dep_t, _t, _d) in res_cache:
        bags = []
        counts = np.zeros(V, np.float64)
        for t in range(len(ids)):
            counts[ids[t]] += 1.0
            bags.append((counts / (t + 1)).copy())
        out.append((bags, ids, dep_t))
    return out


def _bigram_dep(P_bi, res_cache):
    """Bigram P(next|prev) scored at the dependent position (prev is always 'its' -> uniform over dens -> chance)."""
    tot = 0.0
    hit = 0
    n = 0
    for (_S, ids, dep_t, _t, _d) in res_cache:
        prev = ids[dep_t]
        tgt = ids[dep_t + 1]
        p = P_bi[prev]
        tot += -math.log(max(p[tgt], 1e-12))
        hit += int(np.argmax(p) == tgt)
        n += 1
    return tot / max(1, n), hit / max(1, n), n


# ---------------------------------------------------------------------------------------------------------------------
# Run ALL arms at ONE filler distance D (the reservoir pass is shared across arms).
# ---------------------------------------------------------------------------------------------------------------------
def _run_at_distance(seed, args, vocab, subjects, home, verbs, res, buffer, D, chance):
    V = vocab.size
    feature = args.state_feature

    docs = build_discourse_docs(seed, args.n_docs, D, args.n_distractors, subjects, home, verbs)
    n_tr = int(len(docs) * 0.8)
    train_docs = docs[:n_tr][:args.max_train]
    eval_docs = docs[n_tr:][:args.max_eval]

    rc_tr = _res_cache(res, vocab, train_docs, feature)          # EXPENSIVE (bridge forward), once
    rc_ev = _res_cache(res, vocab, eval_docs, feature)
    n_res = len(rc_tr[0][0][0])                                  # reservoir feature dim (2*n_pool for 'both')
    mean_rate = float(np.mean([np.mean(S) for (S, *_r) in rc_ev if len(S)])) if rc_ev else 0.0

    lr = args.lr
    epochs = args.epochs
    wd = args.weight_decay
    ls = args.label_smoothing

    # ARM 1: reservoir-only (the key control -- the fading memory)
    ro_tr = _attach_reservoir_only(rc_tr)
    ro_ev = _attach_reservoir_only(rc_ev)
    ro_mean, ro_std = _standardize_fit(_pairs(ro_tr))
    W_ro = train_readout(_pairs(ro_tr), V, epochs, lr, np.random.default_rng(seed * 13 + 1), ro_mean, ro_std, wd=wd, ls=ls)
    ro_ce, ro_acc, _ = eval_dep(W_ro, ro_mean, ro_std, ro_ev, V)

    # ARM 2: reservoir + buffer (standardize the reservoir sub-block only)
    bf_tr = _attach_buffer(rc_tr, buffer, vocab, subjects)
    bf_ev = _attach_buffer(rc_ev, buffer, vocab, subjects)
    bf_mean, bf_std = _standardize_res_block(bf_tr, n_res)
    W_bf = train_readout(_pairs(bf_tr), V, epochs, lr, np.random.default_rng(seed * 17 + 1), bf_mean, bf_std, wd=wd, ls=ls)
    bf_ce, bf_acc, _ = eval_dep(W_bf, bf_mean, bf_std, bf_ev, V)

    # ARM 3: buffer SLOT-SCRAMBLE (latch a random subject per doc; same latch timing/dim) -> ~reservoir-only
    sc_tr = _attach_buffer(rc_tr, buffer, vocab, subjects, scramble_rng=np.random.default_rng(seed * 811 + D * 3 + 1))
    sc_ev = _attach_buffer(rc_ev, buffer, vocab, subjects, scramble_rng=np.random.default_rng(seed * 811 + D * 3 + 2))
    sc_mean, sc_std = _standardize_res_block(sc_tr, n_res)
    W_sc = train_readout(_pairs(sc_tr), V, epochs, lr, np.random.default_rng(seed * 19 + 1), sc_mean, sc_std, wd=wd, ls=ls)
    sc_ce, sc_acc, _ = eval_dep(W_sc, sc_mean, sc_std, sc_ev, V)

    # ARM 4: REFERENT-SHUFFLE (train the buffer read-out with a DERANGED topic->den mapping; eval on the TRUE test) -> chance
    dhome = _deranged_home(seed, subjects, home)
    rf_tr = []
    for (concat, ids, dep_t), (_S, _ids2, _dep2, topic, _den) in zip(bf_tr, rc_tr):
        ids2 = list(ids)
        ids2[dep_t + 1] = vocab.id(dhome[topic])                 # break the mapping IN TRAINING (deranged den target)
        rf_tr.append((concat, ids2, dep_t))
    W_rf = train_readout(_pairs(rf_tr), V, epochs, lr, np.random.default_rng(seed * 23 + 1), bf_mean, bf_std, wd=wd, ls=ls)
    rf_ce, rf_acc, _ = eval_dep(W_rf, bf_mean, bf_std, bf_ev, V)  # eval on the TRUE mapping test cache

    # ARM 5: BAG-of-prefix (unordered counts; no reservoir/order) -> <= chance
    bag_tr = _bag_cache(rc_tr, V)
    bag_ev = _bag_cache(rc_ev, V)
    bag_mean, bag_std = _standardize_fit(_pairs(bag_tr))
    W_bag = train_readout(_pairs(bag_tr), V, epochs, lr, np.random.default_rng(seed * 29 + 1), bag_mean, bag_std, wd=wd, ls=ls)
    bag_ce, bag_acc, _ = eval_dep(W_bag, bag_mean, bag_std, bag_ev, V)

    # ARM 6: BIGRAM at the dependent position -> <= chance
    P_bi = fit_bigram([ids for (_S, ids, *_r) in rc_tr], V)
    bi_ce, bi_acc, _ = _bigram_dep(P_bi, rc_ev)

    return {
        "D": int(D), "doc_len": len(train_docs[0][0]) if train_docs else 0,
        "n_train": len(train_docs), "n_eval": len(eval_docs), "mean_rate": mean_rate, "chance": chance,
        "reservoir_only_dep_acc": ro_acc, "reservoir_only_dep_ce": ro_ce,
        "buffer_dep_acc": bf_acc, "buffer_dep_ce": bf_ce,
        "slot_scramble_dep_acc": sc_acc, "slot_scramble_dep_ce": sc_ce,
        "referent_shuffle_dep_acc": rf_acc, "referent_shuffle_dep_ce": rf_ce,
        "bag_dep_acc": bag_acc, "bag_dep_ce": bag_ce,
        "bigram_dep_acc": bi_acc, "bigram_dep_ce": bi_ce,
    }


def _gate(mrow):
    """The GO gate at a distance (evaluate at D* beyond the reservoir horizon)."""
    ch = mrow["chance"]
    ro = mrow["reservoir_only_dep_acc"]
    bf = mrow["buffer_dep_acc"]
    checks = {
        "buffer_helps": bool(bf >= ro + MARGIN),
        "reservoir_near_chance": bool(ro <= ch + CHANCE_TOL),
        "scramble_collapses": bool(mrow["slot_scramble_dep_acc"] <= ro + COLLAPSE_EPS),
        "referent_collapses": bool(mrow["referent_shuffle_dep_acc"] <= ch + COLLAPSE_EPS),
        "bag_le_chance": bool(mrow["bag_dep_acc"] <= ch + COLLAPSE_EPS),
        "bigram_le_chance": bool(mrow["bigram_dep_acc"] <= ch + COLLAPSE_EPS),
        "active": bool(mrow["mean_rate"] > ACTIVE_MIN),
    }
    return bool(all(checks.values())), checks


# ---------------------------------------------------------------------------------------------------------------------
# The de-risk (one seed): a horizon sweep (optional) + the full arm comparison + the GO gate at D*.
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_one(seed, args):
    subjects, dens, verbs, all_tokens = build_domain(args.vocab)
    vocab = Vocab(all_tokens)                                    # a COMPLETE closed vocab (all critical tokens guaranteed)
    K = len(subjects)
    chance = 1.0 / K
    home = _home_bijection(seed, subjects, dens)
    res = ReservoirStates(vocab.size, seed=seed, n=args.n_pool)
    buffer = TopicBuffer(vocab, gain=args.buffer_gain)

    horizon = None
    if args.horizon_sweep:
        horizon = []
        for D in args.sweep_distances:
            row = _run_at_distance(seed, args, vocab, subjects, home, verbs, res, buffer, D, chance)
            horizon.append(row)
            print(f"    D={D:>3d} (len {row['doc_len']:>3d})  res-only {row['reservoir_only_dep_acc']:.3f}  |  "
                  f"buffer {row['buffer_dep_acc']:.3f}  |  scramble {row['slot_scramble_dep_acc']:.3f}  |  "
                  f"referent {row['referent_shuffle_dep_acc']:.3f}  |  bag {row['bag_dep_acc']:.3f}  |  bigram "
                  f"{row['bigram_dep_acc']:.3f}   (chance {chance:.3f}, rate {row['mean_rate']:.4f})", flush=True)
        dstar_row = horizon[-1]                                  # D* = the largest swept distance (beyond the horizon)
    else:
        dstar_row = _run_at_distance(seed, args, vocab, subjects, home, verbs, res, buffer, args.filler_distance, chance)

    go, checks = _gate(dstar_row)
    # the CRUX: did the reservoir-only fall to ~chance at ANY swept distance (or at D*)?
    if horizon is not None:
        res_fell = any(r["reservoir_only_dep_acc"] <= chance + CHANCE_TOL for r in horizon)
        res_min = float(min(r["reservoir_only_dep_acc"] for r in horizon))
    else:
        res_fell = dstar_row["reservoir_only_dep_acc"] <= chance + CHANCE_TOL
        res_min = dstar_row["reservoir_only_dep_acc"]

    return {
        "seed": seed, "K_subjects": K, "vocab_size": vocab.size, "chance": chance, "n_pool": res.n,
        "state_feature": args.state_feature, "dstar": dstar_row["D"],
        "reservoir_fell_to_chance": bool(res_fell), "reservoir_min_dep_acc": res_min,
        "dstar_row": dstar_row, "checks": checks, "seed_go": bool(go),
        "horizon": horizon,
    }


def _derisk(seeds, args):
    print(f"EMERGE-RESERVOIR-LM RUNG 2: a DISTAL-REFERENT TOPIC buffer conditions the on-bridge spiking reservoir "
          f"next-token generator; does the buffer restore the topic->den dependency BEYOND the reservoir's fading-memory "
          f"horizon? {len(seeds)}-seed; state-feature={args.state_feature}; "
          f"{'HORIZON-SWEEP ' + str(args.sweep_distances) if args.horizon_sweep else 'single D=' + str(args.filler_distance)}",
          flush=True)
    t0 = time.time()
    err = None
    per = []
    try:
        for s in seeds:
            print(f"  [seed {s}]", flush=True)
            d = _derisk_one(s, args)
            per.append(d)
            r = d["dstar_row"]
            print(f"    => D*={d['dstar']}: reservoir-only {r['reservoir_only_dep_acc']:.3f} vs buffer "
                  f"{r['buffer_dep_acc']:.3f} (chance {d['chance']:.3f})  reservoir_fell_to_chance="
                  f"{d['reservoir_fell_to_chance']}  seed_go={d['seed_go']}  {d['checks']}", flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        def m(field):
            return float(np.mean([d["dstar_row"][field] for d in per]))
        chance = per[0]["chance"]
        ro = m("reservoir_only_dep_acc")
        bf = m("buffer_dep_acc")
        sc = m("slot_scramble_dep_acc")
        rf = m("referent_shuffle_dep_acc")
        bag = m("bag_dep_acc")
        bi = m("bigram_dep_acc")
        rate = m("mean_rate")
        res_fell = all(d["reservoir_fell_to_chance"] for d in per)

        agg_row = {"reservoir_only_dep_acc": ro, "buffer_dep_acc": bf, "slot_scramble_dep_acc": sc,
                   "referent_shuffle_dep_acc": rf, "bag_dep_acc": bag, "bigram_dep_acc": bi,
                   "mean_rate": rate, "chance": chance}
        go, checks = _gate(agg_row)
        go = bool(go and res_fell)

        if go:
            verdict = (
                f"GO -- a single-slot DISTAL-REFERENT TOPIC BUFFER restores a discourse dependency BEYOND the on-bridge "
                f"spiking reservoir's fading-memory horizon. At D*={per[0]['dstar']} filler clauses (beyond the horizon: "
                f"reservoir-only dep_acc {ro:.3f} ~ chance {chance:.3f}), conditioning the Rung-1 reservoir next-token "
                f"read-out on the CONCATENATED [reservoir | topic-buffer] feature lifts the referent-dependent token to "
                f"{bf:.3f} (+{bf - ro:.3f} over reservoir-only, MARGIN {MARGIN}) -- the buffer is LOAD-BEARING, not a "
                f"decoration. Every control COLLAPSES: buffer SLOT-SCRAMBLE (latch a random subject, same slot) "
                f"{sc:.3f} ~ reservoir-only (the held CONTENT is load-bearing); REFERENT-SHUFFLE (deranged topic->den map "
                f"in training) {rf:.3f} ~ chance (the learned mapping is broken); BAG-of-prefix {bag:.3f} <= chance (the "
                f"topic is not readable from unordered counts -- the same-category distractors mask it); BIGRAM {bi:.3f} "
                f"<= chance (not readable from the previous token). The reservoir is genuinely active ({rate:.4f} spikes/"
                f"neuron/step). {len(seeds)} seeds. ==> the ESN/LSM reservoir (fading memory) + a NON-fading single-slot "
                f"WM latch (Grosz-Sidner attentional focus / a prefrontal topic register) together carry a distal "
                f"discourse referent a reservoir alone cannot. HONEST SCOPE: a controlled discourse-referent grammar "
                f"(closed template domain), NOT open prose (R4); the buffer is a functional single-slot latch (the "
                f"spiking theta-gamma / NMDA-WM realization is the follow-on rung). Reuse-by-import; NO sim/ edit; NO edit "
                f"to any existing runner.")
        else:
            miss = []
            if not res_fell:
                miss.append(f"the reservoir-only did NOT fall to ~chance at the swept distances (min reservoir dep_acc "
                            f"{min(d['reservoir_min_dep_acc'] for d in per):.3f} vs chance {chance:.3f}) -- the reservoir "
                            f"still carries the referent, so the buffer is NOT yet load-bearing at this scale. Named levers: "
                            f"a larger filler distance D, more/other distractors (--n-distractors), a leakier/smaller "
                            f"reservoir (--n-pool), or the recency-emphasizing per_window state feature (--state-feature "
                            f"per_window). This is an HONEST finding, not a forced GO")
            if not checks["buffer_helps"]:
                miss.append(f"the buffer did not beat reservoir-only by {MARGIN} (buffer {bf:.3f} vs reservoir {ro:.3f})")
            if not checks["reservoir_near_chance"]:
                miss.append(f"reservoir-only at D* is not near chance ({ro:.3f} vs chance {chance:.3f})")
            if not checks["scramble_collapses"]:
                miss.append(f"slot-scramble did not collapse ({sc:.3f} vs reservoir {ro:.3f})")
            if not checks["referent_collapses"]:
                miss.append(f"referent-shuffle did not collapse to chance ({rf:.3f} vs chance {chance:.3f})")
            if not checks["bag_le_chance"]:
                miss.append(f"bag-of-prefix beats chance ({bag:.3f} vs chance {chance:.3f}) -- the topic leaks into counts")
            if not checks["bigram_le_chance"]:
                miss.append(f"bigram beats chance ({bi:.3f} vs chance {chance:.3f})")
            if not checks["active"]:
                miss.append(f"the reservoir is nearly SILENT ({rate:.4f} spikes/neuron/step)")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". An HONEST negative is the deliverable: this de-risk is "
                       "designed to REFUTE the buffer's usefulness if the reservoir already carries the referent. Do NOT "
                       "force GO.")
    else:
        go = False
        verdict = f"ERROR -- {err}"
        ro = bf = sc = rf = bag = bi = rate = None
        agg_row = None
        res_fell = False

    summary = {
        "probe": "emerge_reservoir_lm_rung2_distal_referent", "verdict": verdict,
        "go": bool(go) if err is None else False,
        "state_feature": args.state_feature,
        "mechanism": ("condition the Rung-1 on-bridge spiking reservoir (EMERGE-82 OnBridgeLSM: a recurrent Izhikevich "
                      "BrainRegion on a real SimulationBridge; the per-token running_cumulative spike-rate read) + its "
                      "one-step next-token delta read-out on the CONCATENATED [reservoir-state | topic-buffer one-hot] "
                      "feature. The TopicBuffer is a single gated WM slot that latches the discourse topic when the DOC "
                      "boundary marker is seen (the next token) and holds it unfading. Standardize the reservoir sub-block "
                      "only; leave the buffer one-hot as-is. Reuse-by-import; NO sim/ edit; NO edit to any existing runner."),
        "task": ("a controlled discourse-referent grammar: DOC the <TOPIC> can <v> . ; D same-category filler clauses ; "
                 "it goes to the <DEN=HOME[TOPIC]> . -- score ONLY the dependent DEN token, which is fixed only by the "
                 "distally-introduced topic (never mentioned in intro/fillers). At D* BEYOND the reservoir's fading-memory "
                 "horizon: does the topic buffer restore the dependency (buffer > reservoir-only by MARGIN), while "
                 "slot-scramble ~ reservoir-only, referent-shuffle ~ chance, bag <= chance, bigram <= chance; reservoir "
                 "genuinely active; 6-seed"),
        "crux": ("the HORIZON SWEEP must find a D* where reservoir-only dep_acc falls to ~chance (the reservoir forgets "
                 "the distal topic under same-category interference); the GO experiment runs AT/beyond D*. If reservoir-"
                 "only never falls to chance, the buffer is not load-bearing at this scale (an honest negative)."),
        "thresholds": {"margin": MARGIN, "collapse_eps": COLLAPSE_EPS, "chance_tol": CHANCE_TOL, "active_min": ACTIVE_MIN},
        "params": {"n_pool": args.n_pool, "vocab": args.vocab, "epochs": args.epochs, "lr": args.lr,
                   "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
                   "n_docs": args.n_docs, "filler_distance": args.filler_distance, "n_distractors": args.n_distractors,
                   "max_train": args.max_train, "max_eval": args.max_eval, "state_feature": args.state_feature,
                   "horizon_sweep": bool(args.horizon_sweep), "sweep_distances": args.sweep_distances},
        "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "reservoir_fell_to_chance": bool(res_fell) if err is None else None,
        "aggregate": None if err is not None else {
            "dstar": per[0]["dstar"], "chance": per[0]["chance"],
            "reservoir_only_dep_acc": ro, "buffer_dep_acc": bf, "slot_scramble_dep_acc": sc,
            "referent_shuffle_dep_acc": rf, "bag_dep_acc": bag, "bigram_dep_acc": bi, "mean_rate": rate,
            "checks": None if agg_row is None else _gate(agg_row)[1],
        },
        "per_seed": per,
        "HONEST_NOTE": ("RUNG 2 of the emergent-generation ladder: a single-slot distal-referent working-memory buffer "
                        "conditions the Rung-1 on-bridge spiking reservoir LM so a topic introduced beyond the reservoir's "
                        "fading-memory horizon still determines a later token. GO = at D* beyond the horizon (reservoir-"
                        "only ~ chance), the buffer restores the dependency AND every control collapses (slot-scramble ~ "
                        "reservoir-only, referent-shuffle ~ chance, bag/bigram <= chance). This de-risk is designed to "
                        "REFUTE the buffer if the reservoir already carries the referent -- an honest negative (reservoir "
                        "never falls to chance) is the deliverable, with named levers (bigger D, more distractors, leakier/"
                        "smaller reservoir, per_window feature). Controlled discourse-referent grammar (closed domain), NOT "
                        "open prose (R4). The buffer is a functional single-slot latch (spiking theta-gamma/NMDA-WM = the "
                        "follow-on rung). Reuse-by-import; NO sim/ edit; NO edit to any existing runner."),
    }
    out_path = Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[rung2-distal-referent] VERDICT: {verdict}", flush=True)
    print(f"[rung2-distal-referent] wrote {out_path}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-docs", type=int, default=1200, help="documents generated per distance (split 80/20, then capped)")
    ap.add_argument("--vocab", type=int, default=40, help="target closed-vocab size -> sizes K subjects/dens; chance=1/K")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02,
                    help="delta-rule lr (a 1-dim buffer one-hot must compete with ~600 standardized reservoir dims in the "
                         "linear softmax, so the read-out needs enough training to grow the buffer->den weight)")
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=_N_POOL, help="reservoir region size (smaller = leakier, a named lever)")
    ap.add_argument("--state-feature", choices=["both", "running_cumulative", "per_window"], default="both",
                    help="reservoir feature: 'both' (per_window localizer + running_cumulative memory, DEFAULT) / "
                         "'per_window' / 'running_cumulative'")
    ap.add_argument("--filler-distance", type=int, default=24, help="single-D mode: the D* to run the GO gate at")
    ap.add_argument("--n-distractors", type=int, default=3, help="distinct same-category distractor subjects guaranteed per doc")
    ap.add_argument("--buffer-gain", type=float, default=4.0,
                    help="WM-slot read-out drive strength (a fixed architectural constant, NOT label-tuned): the single "
                         "one-hot dim must compete with ~600 reservoir dims in the linear softmax")
    ap.add_argument("--max-train", type=int, default=120, help="cap on train docs pushed through the reservoir (per distance)")
    ap.add_argument("--max-eval", type=int, default=40, help="cap on eval docs pushed through the reservoir (per distance)")
    ap.add_argument("--horizon-sweep", action="store_true",
                    help="sweep --sweep-distances, print reservoir-only dep_acc (the crux) + the full arm table at each D; "
                         "gate at the largest D (D* beyond the horizon)")
    ap.add_argument("--sweep-distances", type=int, nargs="+", default=[2, 8, 16, 32],
                    help="filler distances for the horizon sweep")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args()
    return _derisk(a.seeds, a)


if __name__ == "__main__":
    raise SystemExit(main())
