"""EMERGE-RESERVOIR-LM -- a fully-EMERGENT, learned-from-experience, ON-BRIDGE, NO-BPTT autoregressive NEXT-TOKEN
language generator: a FIXED spiking reservoir (the EMERGE-82 OnBridgeLSM -- a recurrent Izhikevich BrainRegion on a real
`SimulationBridge`) + a SHALLOW linear-softmax read-out trained ONLINE by the ONE-STEP next-token delta rule (Widrow-Hoff
on a clean error; NO backprop-through-time, NO weight transport). This is the primary emergence experiment: language
generation that learns from raw token experience, on the project's own spiking substrate, with a local output-layer rule.

WHY THIS IS THE RIGHT (EMERGENT, NO-BPTT) SHAPE. The reservoir/echo-state (Jaeger 2001) + liquid-state-machine (Maass
2002) discipline: a fixed random recurrent pool projects the input history into a high-dimensional fading-memory state;
ONLY a shallow read-out is trained. Because the *target* of a language model IS the next observed token, the output-layer
error `target - softmax(W_out @ state)` is a CLEAN, LOCAL error (the next token is directly observed -- no credit must be
propagated back through time or through the recurrence). So the read-out is learnable by the biologically-plausible
one-step delta rule -- exactly the pair a brain has (a fixed recurrent cortex + a locally-trained output projection),
replacing BPTT with self-supervised next-token prediction. The reservoir recurrence + the input projection are FIXED-RANDOM
(never trained) -- the ESN/LSM invariant.

THE SUBSTRATE (reuse-by-import; NO `sim/` edit; NO edit to any existing runner). The reservoir IS the EMERGE-82
`OnBridgeLSM`: one recurrent Izhikevich `BrainRegion` (internal_density -> fixed-random recurrent conductance synapses),
driven per token through the bridge's real `cp_external_input_current` + `_run_one_simulation_step` (conductance synapses
`g_syn*(V-E)`, the actual neuron model), read from the region's real `cp_firing_states`, washed to its post-init snapshot
(EMERGE-61 mechanism) before each sentence so every sentence is an independent read. We subclass it (`ReservoirStates`) to
expose the reservoir feature AFTER EACH TOKEN of a sequence (not just the final cumulative state), so the read-out can
predict every next token. The corpus is `m62.build_stream` (the controlled EMERGE SVO+function-word stream); we build a
small closed TOKEN vocabulary (top-V) and drive the reservoir with a ONE-HOT over that vocab (so the reservoir sees the
full token identity, competing fairly with the bigram).

THE PER-TOKEN STATE FEATURE (documented choice). Two options: (a) the per-token WINDOW spike-count (spikes in token t's
`_T_STEP` window), or (b) the RUNNING-CUMULATIVE normalized feature (mean spike-rate over the whole prefix 0..t). We
DEFAULT to (b) running-cumulative (`--state-feature running_cumulative`) -- it is the OnBridgeLSM's OWN validated read
(its `final_state` returns exactly the cumulative-normalized pool rate), and it integrates the WHOLE prefix (the fading
memory is in the recurrent dynamics; the cumulative read is a stable low-variance summary of it), which is what lets a
next-token predictor exploit higher-order context (e.g. "does not" -> a bare verb) that a bigram cannot. (a) per_window is
selectable (`--state-feature per_window`) as the more recency-emphasizing alternative. The choice is a module flag so the
controller can compare.

THE READ-OUT (shallow, local, no-BPTT). `W_out` (V x n_reservoir+bias), init ZEROS. For each position t in each training
sentence: `p = softmax(W_out @ [S[t], 1])`, `target = onehot(id[t+1])`, `e = target - p`, `W_out += lr * outer(e, [S[t],1])`
(standardized features). Iterate epochs over the training sentences. The reservoir states are FIXED, so they are computed
ONCE (the expensive bridge forward pass) and CACHED; the many delta-rule epochs run over the cached states (cheap).

DE-RISK (the GO gate = the reservoir read-out BEATS the bigram on held-out cross-entropy, AND every anti-cheat collapses):
  * HELD-OUT split (disjoint index partition from train; assert 0 index-overlap) -> mean next-token CROSS-ENTROPY + top-1 ACC.
  * BIGRAM baseline (add-1 smoothed P(next|prev) on the SAME train split -> the real comparator; the reservoir carries
    higher-order context via fading memory, so it should beat it). TRIGRAM = an upper reference/ceiling. UNIGRAM = a floor.
  * ANTI-CHEAT A (shuffled reservoir STATE): permute S[t] across positions within each held-out sentence -> destroys the
    temporal alignment -> CE collapses to ~bigram/unigram (NOT below bigram).
  * ANTI-CHEAT B (permuted-corpus): shuffle token order within TRAIN sentences, recompute reservoir states, train a fresh
    read-out -> no learnable sequential structure -> must NOT beat the bigram on the real held-out.
  * ANTI-CHEAT C (frozen read-out): W_out left at init (zeros) -> CE == log(V) (chance).
  * ANTI-CHEAT D (silenced reservoir): drive OFF (OnBridgeLSM `silence=True`) -> states carry no input -> read-out
    collapses to ~unigram.
  GO = reservoir_CE < bigram_CE - MARGIN AND shuffled_state_CE >= bigram_CE - EPS AND permuted_corpus_CE >= bigram_CE - EPS
       AND frozen_CE ~ log(V) AND silenced_CE >= bigram_CE - EPS AND the region is genuinely active.
  Numbers are honest -- if the reservoir does NOT beat the bigram at this scale, it is REPORTED as a BOUNDARY (an honest
  negative is the deliverable; the named next levers are more reservoir / more epochs / the per_window feature).

AUTOREGRESSIVE ROLLOUT (qualitative, NOT gated): from a seed token, feed the read-out's argmax back as the next reservoir
input, generate ~8 tokens; print a few sample rollouts for eyeballing grammaticality.

HONEST SCOPE. A TOKEN-level LM over the BOUNDED controlled EMERGE stream (a closed template grammar), not open prose (R4).
The reservoir + input projection are fixed-random; only the shallow output read-out is learned (the ESN/LSM discipline).
Reuse-by-import (EMERGE-82 OnBridgeLSM + EMERGE-61 wash-out + EMERGE-62 stream); NO `sim/` edit; NO edit to any existing
runner.

Run:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SIM_BACKEND=numpy \
    python -u -m research.runners._emerge_reservoir_lm_derisk --seeds 42 --n-sentences 1500 --vocab 24 --epochs 8
  (6-seed sweep -- see the tail of this file / the returned command.)
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
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse-by-import: the EMERGE-82 on-bridge spiking reservoir (OnBridgeLSM + constants) + the EMERGE-61 wash-out
# (`_restore_state`, re-exported by emerge82) + the EMERGE-62 controlled stream.
import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge82_onbridge_lsm_derisk import (  # noqa: E402
    OnBridgeLSM, _N_POOL, _T_STEP, _BIAS, _restore_state,
    # additive (adversarial controls only): the reservoir-build constants + the wash-out snapshot helper, reused to
    # build the NON-RECURRENT (zeroed-recurrent-weight) memoryless-projection control bridge. Default run never touches these.
    _snapshot_state, _INTERNAL_DENSITY, _IN_SCALE,
)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge_reservoir_lm.json"

MARGIN = 0.05   # the reservoir must beat the bigram held-out CE by at least this many nats
EPS = 0.03      # a collapsed control must be within this (or above) the bigram CE (i.e. NOT beat it)
FROZEN_TOL = 0.05  # frozen (zeros) read-out CE must be within this of log(V) (chance)
ACTIVE_MIN = 1e-4  # mean reservoir spike-rate/neuron/step above this = genuinely active


# ---------------------------------------------------------------------------------------------------------------------
# A small closed TOKEN vocabulary built from the stream (top-V by frequency; the rest -> <unk>). A one-hot over this vocab
# drives the reservoir (so it sees the full token identity, competing fairly with the bigram).
# ---------------------------------------------------------------------------------------------------------------------
class Vocab:
    def __init__(self, words):
        self.i2w = list(words) + ["<unk>"]
        self.w2i = {w: i for i, w in enumerate(self.i2w)}
        self.unk = len(words)
        self.size = len(self.i2w)

    @classmethod
    def build(cls, sents, V):
        c = Counter(w for s in sents for w in s)
        keep = [w for w, _ in c.most_common(max(1, V - 1))]
        return cls(keep)

    def id(self, w):
        return self.w2i.get(w, self.unk)

    def word(self, i):
        return self.i2w[i]

    def ids(self, s):
        return [self.id(w) for w in s]

    def onehot(self, w):
        v = np.zeros(self.size, np.float64)
        v[self.id(w)] = 1.0
        return v

    def encode_seq(self, s):
        return np.asarray([self.onehot(w) for w in s]) if s else np.zeros((0, self.size))


# ---------------------------------------------------------------------------------------------------------------------
# The FIXED spiking reservoir with a PER-TOKEN read. Subclasses the EMERGE-82 OnBridgeLSM (its fixed recurrent Izhikevich
# region + fixed-random input projection + wash-out) and exposes the reservoir feature AFTER EACH token of a sequence.
# ---------------------------------------------------------------------------------------------------------------------
class ReservoirStates(OnBridgeLSM):
    """OnBridgeLSM + a per-token read: `per_token_states(U)` washes the bridge, drives the region per token via
    cp_external_input_current, runs the real step loop, and returns S[t] = the reservoir feature after token t.
    feature='running_cumulative' -> mean spike-rate over the whole prefix 0..t (OnBridgeLSM's own validated read);
    feature='per_window' -> spike-rate in token t's `_T_STEP` window (the recency-emphasizing alternative)."""

    def per_token_states(self, U, silence=False, feature="running_cumulative"):
        from sim.backend import to_host
        b = self.bridge
        _restore_state(b, self._snap)                         # wash to post-init -> independent read per sentence
        counts = np.zeros(self.n, np.float64)                 # cumulative pool spike-counts over the sequence
        S = []
        steps = 0
        for t in range(len(U)):
            drive = np.zeros(self.n) if silence else (self.W_in @ U[t] + _BIAS)
            cur = np.zeros(self._num, np.float32)
            cur[self.res_idx] = drive.astype(np.float32)
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[self.res_idx] = (self._xp.asarray(cur[self.res_idx])
                                                         if self._xp is not None else cur[self.res_idx])
            win = np.zeros(self.n, np.float64)
            for _ in range(_T_STEP):
                b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)[self.res_idx]
                counts += fs
                win += fs
                steps += 1
            if feature == "per_window":
                S.append((win / _T_STEP).copy())
            else:                                             # running_cumulative
                S.append((counts / max(1, steps)).copy())
        b.cp_external_input_current[:] = 0.0
        return S

    def rollout(self, vocab, W, mean, std, seed_token, n_gen, feature="running_cumulative"):
        """Autoregressive greedy rollout: wash, then feed each argmax token back as the next reservoir input."""
        from sim.backend import to_host
        b = self.bridge
        _restore_state(b, self._snap)
        counts = np.zeros(self.n, np.float64)
        steps = 0
        cur = seed_token
        toks = [cur]
        for _ in range(n_gen):
            u = vocab.onehot(cur)
            drive = self.W_in @ u + _BIAS
            cur_full = np.zeros(self._num, np.float32)
            cur_full[self.res_idx] = drive.astype(np.float32)
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[self.res_idx] = (self._xp.asarray(cur_full[self.res_idx])
                                                         if self._xp is not None else cur_full[self.res_idx])
            win = np.zeros(self.n, np.float64)
            for _ in range(_T_STEP):
                b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)[self.res_idx]
                counts += fs
                win += fs
                steps += 1
            feat = (win / _T_STEP) if feature == "per_window" else (counts / max(1, steps))
            x = np.concatenate([(feat - mean) / std, [1.0]])
            cur = vocab.word(int(np.argmax(W @ x)))
            toks.append(cur)
        b.cp_external_input_current[:] = 0.0
        return toks


# ---------------------------------------------------------------------------------------------------------------------
# The shallow linear-softmax read-out trained by the ONE-STEP next-token delta rule (Widrow-Hoff on a clean error). NO
# BPTT: the reservoir states are fixed + cached, only W_out is learned over positions independently.
# ---------------------------------------------------------------------------------------------------------------------
def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _standardize_fit(cache):
    xs = [np.asarray(states) for states, ids in cache if len(states) > 0]
    if not xs:
        return np.zeros(1), np.ones(1)
    allx = np.concatenate(xs, axis=0)
    mean = allx.mean(0)
    std = allx.std(0) + 1e-6
    return mean, std


def train_readout(cache, V, epochs, lr, rng, mean, std, wd=0.0, ls=0.0, polyak=True):
    """Online one-step next-token delta rule over cached (states, ids). W_out init ZEROS (so frozen == chance).
    The update is `W += lr * (outer(target - p, x) - wd*W)`, where `target` is a label-smoothed one-hot of the NEXT
    token. `wd` (L2 weight decay -- biologically a synaptic-homeostasis decay) + `ls` (label smoothing -- the target
    spike distribution is not a perfect delta) keep the shallow read-out CALIBRATED (they cap the softmax confidence so
    held-out cross-entropy is not dominated by a heavy tail of confidently-wrong predictions). Both keep the rule LOCAL +
    one-step -- NO backprop-through-time, NO weight transport.

    `polyak=True` (DEFAULT -- byte-identical to the original) applies Polyak-Ruppert tail averaging over the last half of
    the epochs. `polyak=False` (the --vanilla control) returns the RAW final-epoch weights -- the pure one-step delta rule
    with NO tail-averaging, so a caller can confirm the reservoir's win is not a calibration artifact."""
    n_feat = len(mean) + 1
    W = np.zeros((V, n_feat))
    idx = list(range(len(cache)))
    burn = epochs // 2                                        # Polyak-Ruppert tail averaging: average W over the last
    W_sum = np.zeros_like(W)                                  # half of the epochs -> reduces SGD endpoint noise (a slow
    n_avg = 0                                                 # consolidation read of the fast local-delta weights; the
    for ep in range(epochs):                                 # UPDATES stay the one-step local delta rule).
        rng.shuffle(idx)
        for si in idx:
            states, ids = cache[si]
            for t in range(len(ids) - 1):
                x = np.concatenate([(states[t] - mean) / std, [1.0]])
                p = _softmax(W @ x)
                tgt = np.full(V, ls / V)                      # label-smoothed target
                tgt[ids[t + 1]] += 1.0 - ls
                W += lr * (np.outer(tgt - p, x) - wd * W)     # clean next-token error + L2 decay (local, one-step)
        if polyak and ep >= burn:
            W_sum += W
            n_avg += 1
    if not polyak:
        return W                                              # --vanilla: raw one-step delta, no tail-averaging
    return W_sum / n_avg if n_avg > 0 else W


def eval_ce(W, mean, std, cache, V):
    tot = 0.0
    hit = 0
    n = 0
    for states, ids in cache:
        for t in range(len(ids) - 1):
            x = np.concatenate([(states[t] - mean) / std, [1.0]])
            p = _softmax(W @ x)
            tgt = ids[t + 1]
            tot += -math.log(max(p[tgt], 1e-12))
            hit += int(np.argmax(p) == tgt)
            n += 1
    return tot / max(1, n), hit / max(1, n), n


# ---------------------------------------------------------------------------------------------------------------------
# n-gram baselines (trained on the SAME capped train split as the reservoir -> a fair CE comparison).
# ---------------------------------------------------------------------------------------------------------------------
def fit_bigram(id_sents, V):
    c = np.ones((V, V))                                       # add-1 smoothing
    for ids in id_sents:
        for a, b in zip(ids, ids[1:]):
            c[a, b] += 1.0
    return c / c.sum(1, keepdims=True)


def bigram_ce(P, id_sents):
    tot = 0.0
    hit = 0
    n = 0
    for ids in id_sents:
        for a, b in zip(ids, ids[1:]):
            tot += -math.log(max(P[a, b], 1e-12))
            hit += int(np.argmax(P[a]) == b)
            n += 1
    return tot / max(1, n), hit / max(1, n), n


def fit_trigram(id_sents, V):
    from collections import defaultdict
    c = defaultdict(lambda: np.ones(V))                      # add-1
    for ids in id_sents:
        for i in range(len(ids) - 2):
            c[(ids[i], ids[i + 1])][ids[i + 2]] += 1.0
    return c


def trigram_ce(ctx, P_bi, id_sents):
    tot = 0.0
    hit = 0
    n = 0
    for ids in id_sents:
        for t in range(len(ids) - 1):
            tgt = ids[t + 1]
            if t >= 1 and (ids[t - 1], ids[t]) in ctx:
                row = ctx[(ids[t - 1], ids[t])]
                p = row / row.sum()
            else:
                p = P_bi[ids[t]]                              # backoff to bigram at t==0 / unseen context
            tot += -math.log(max(p[tgt], 1e-12))
            hit += int(np.argmax(p) == tgt)
            n += 1
    return tot / max(1, n), hit / max(1, n), n


def fit_fourgram(id_sents, V):
    """ADDITIVE (--controls only): order-3 Markov (4-gram) context table. c[(w_{t-2},w_{t-1},w_t)][w_{t+1}] add-1 counts.
    A stronger n-gram CEILING than the trigram; if a 4-gram beats the reservoir, the reservoir is merely ~n-gram."""
    from collections import defaultdict
    c = defaultdict(lambda: np.ones(V))                      # add-1
    for ids in id_sents:
        for i in range(len(ids) - 3):
            c[(ids[i], ids[i + 1], ids[i + 2])][ids[i + 3]] += 1.0
    return c


def fourgram_ce(ctx4, ctx3, P_bi, id_sents):
    """4-gram held-out CE with backoff: 3-token context -> trigram 2-token context -> bigram (at t<2 / unseen context)."""
    tot = 0.0
    hit = 0
    n = 0
    for ids in id_sents:
        for t in range(len(ids) - 1):
            tgt = ids[t + 1]
            if t >= 2 and (ids[t - 2], ids[t - 1], ids[t]) in ctx4:
                row = ctx4[(ids[t - 2], ids[t - 1], ids[t])]
                p = row / row.sum()
            elif t >= 1 and (ids[t - 1], ids[t]) in ctx3:                 # backoff to trigram context
                row = ctx3[(ids[t - 1], ids[t])]
                p = row / row.sum()
            else:
                p = P_bi[ids[t]]                                          # backoff to bigram
            tot += -math.log(max(p[tgt], 1e-12))
            hit += int(np.argmax(p) == tgt)
            n += 1
    return tot / max(1, n), hit / max(1, n), n


def fit_unigram(id_sents, V):
    c = np.ones(V)
    for ids in id_sents:
        for a in ids:
            c[a] += 1.0
    return c / c.sum()


def unigram_ce(P, id_sents):
    tot = 0.0
    n = 0
    for ids in id_sents:
        for t in range(len(ids) - 1):
            tot += -math.log(max(P[ids[t + 1]], 1e-12))
            n += 1
    return tot / max(1, n), n


# ---------------------------------------------------------------------------------------------------------------------
# corpus helpers.
# ---------------------------------------------------------------------------------------------------------------------
def _split_sentences(stream):
    """Split the flat token stream (m62.build_stream) into sentences on the '.' delimiter (period dropped)."""
    sents = []
    cur = []
    for tok in stream:
        if tok == m62.SENT_PERIOD:
            if cur:
                sents.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        sents.append(cur)
    return sents


def _shuffle_states(states, ids, rng):
    """Anti-cheat A: permute the STATE order within a sentence, keep the target ids -> destroy temporal alignment."""
    order = list(range(len(states)))
    rng.shuffle(order)
    return [states[i] for i in order], ids


def _shuffle_tokens(s, rng):
    """Anti-cheat B: shuffle the token order within a sentence -> no learnable sequential structure."""
    s2 = list(s)
    rng.shuffle(s2)
    return s2


def _cache(res, vocab, sents, silence=False, feature="running_cumulative"):
    """Run the (fixed) reservoir over each sentence ONCE -> list of (per-token states, token-id list)."""
    out = []
    for s in sents:
        states = res.per_token_states(vocab.encode_seq(s), silence=silence, feature=feature)
        out.append((states, vocab.ids(s)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# ADVERSARIAL CONTROLS (--controls only; all ADDITIVE -- the default run never touches any of this).
# ---------------------------------------------------------------------------------------------------------------------
def _bag_cache(cache, V):
    """CONTROL 1 -- BAG-OF-PREFIX feature (the key confound). Build a cache in the SAME (states, ids) format as the
    reservoir cache, but with states[t] = the normalized COUNT VECTOR of the token ids in the prefix 0..t (a V-dim
    UNORDERED bag -- NO reservoir, NO order, includes token t exactly as the reservoir S[t] does). Reuses the token-id
    lists already in `cache`, so the SAME positions are trained/evaluated. Trained + evaluated by the SAME delta-rule
    read-out (train_readout/eval_ce) with the SAME epochs/lr/calibration -> only the FEATURE differs. INTERPRETATION: if
    the bag ALSO beats the bigram, then 'beating the bigram' only needs >1 prior token (a bag gives that free) and the
    reservoir's recurrent DYNAMICS are not the load-bearing thing; the reservoir must beat the BAG to claim they are."""
    out = []
    for states, ids in cache:
        bags = []
        counts = np.zeros(V, np.float64)
        for t in range(len(ids)):
            counts[ids[t]] += 1.0
            bags.append((counts / (t + 1)).copy())
        out.append((bags, ids))
    return out


def _build_nonrec_bridge(seed, n_pool, in_dim, dt=0.5):
    """CONTROL 2 -- build the SAME reservoir bridge as EMERGE-82 `_build_reservoir_bridge`, but with the recurrent
    synapses ZEROED: the same internal_density connectivity STRUCTURE (so the bridge builds -- an internal_density=0.0
    region has ZERO synapses total and trips a bridge-init bug), but exc_weight_mean=inh_weight_mean=weight_jitter=0.0 so
    every recurrent synapse carries EXACTLY zero conductance -> no recurrent current -> the fixed-random LSM recurrence is
    functionally REMOVED (the task's sanctioned 'zero the recurrent synapses' variant). Everything else (input projection
    W_in seed/scale, tonic bias, Izhikevich neurons, seed) is identical -> a memoryless feed-forward fixed-random
    projection. Local copy (NO edit to the EMERGE-82 runner, NO sim/ edit); only invoked under --controls."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="reservoir", n_neurons=n_pool, exc_fraction=0.8, internal_density=_INTERNAL_DENSITY,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = []
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    rt = RuntimeState()
    rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    res_idx = np.asarray(b.region_manager.indices("reservoir"))
    rng = np.random.default_rng(seed * 7919 + 3)              # SAME W_in seed formula as EMERGE-82 (fair projection)
    W_in = (rng.random((len(res_idx), in_dim)) * 2 - 1) * _IN_SCALE
    snap = _snapshot_state(b)
    return b, res_idx, W_in, snap


class NonRecurrentReservoirStates(ReservoirStates):
    """CONTROL 2 -- the NON-RECURRENT, MEMORYLESS projection of the CURRENT token only. Same bridge/neurons/input as the
    reservoir but with the recurrent synapses ZEROED (weight 0 -> no recurrent current). To be a genuinely memoryless
    read (no fading memory from EITHER recurrence OR single-neuron adaptation), `per_token_states` (a) WASHES the bridge
    to its post-init snapshot before EACH token, and (b) reads the per-token WINDOW spike-rate -> S[t] is a pure function
    of token t's drive. So the delta read-out learns P(next | current-token-spikes) = a learned bigram-equivalent.
    INTERPRETATION: a memoryless projection should NOT beat the bigram; if it does, something is leaking. The recurrent
    reservoir must beat THIS to claim its temporal (fading) memory matters."""

    def __init__(self, in_dim, seed, n=_N_POOL, dt=0.5):
        # deliberately do NOT call super().__init__ (that builds the recurrent bridge); build the density=0 bridge.
        self.n = n
        self.bridge, self.res_idx, self.W_in, self._snap = _build_nonrec_bridge(seed, n, in_dim, dt=dt)
        from sim.backend import get_backend
        self._xp, _ = get_backend()
        self._num = int(self.bridge.core_config.num_neurons)
        self._last_mean_spikes = 0.0

    def per_token_states(self, U, silence=False, feature="per_window"):
        from sim.backend import to_host
        b = self.bridge
        S = []
        for t in range(len(U)):
            _restore_state(b, self._snap)                     # wash BEFORE EACH token -> zero cross-token carryover
            drive = np.zeros(self.n) if silence else (self.W_in @ U[t] + _BIAS)
            cur = np.zeros(self._num, np.float32)
            cur[self.res_idx] = drive.astype(np.float32)
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[self.res_idx] = (self._xp.asarray(cur[self.res_idx])
                                                         if self._xp is not None else cur[self.res_idx])
            win = np.zeros(self.n, np.float64)
            for _ in range(_T_STEP):
                b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)[self.res_idx]
                win += fs
            S.append((win / _T_STEP).copy())                  # per-window read (ignores `feature` -> always memoryless)
        b.cp_external_input_current[:] = 0.0
        return S


def _run_controls(seed, args, vocab, V, res, feat, tr, ev, tr_cache, ev_cache, tr_ids, ev_ids, P_bi, ctx,
                  mean, std, res_ce, bi_ce):
    """Run the 4 adversarial controls (bag-of-prefix, non-recurrent projection, vanilla read-out, 4-gram) like-for-like
    against the reservoir + bigram, and return the extra JSON fields. Only called when --controls is set."""
    # (1) BAG-OF-PREFIX: same delta read-out, SAME calibration/epochs/lr, feature = bag-of-prefix counts (no reservoir).
    bag_tr = _bag_cache(tr_cache, V)
    bag_ev = _bag_cache(ev_cache, V)
    bmean, bstd = _standardize_fit(bag_tr)
    W_bag = train_readout(bag_tr, V, args.epochs, args.lr, np.random.default_rng(seed * 31 + 1), bmean, bstd,
                          wd=args.weight_decay, ls=args.label_smoothing)
    bag_ce, bag_acc, _ = eval_ce(W_bag, bmean, bstd, bag_ev, V)

    # (2) NON-RECURRENT memoryless projection: zeroed recurrent weights, per-token reset + per-window read; same read-out.
    nonrec = NonRecurrentReservoirStates(V, seed=seed, n=args.n_pool)
    nr_tr = _cache(nonrec, vocab, tr)
    nr_ev = _cache(nonrec, vocab, ev)
    nonrec_rate = float(np.mean([np.mean(s) for s, _ in nr_ev if len(s)])) if nr_ev else 0.0
    nmean, nstd = _standardize_fit(nr_tr)
    W_nr = train_readout(nr_tr, V, args.epochs, args.lr, np.random.default_rng(seed * 37 + 1), nmean, nstd,
                         wd=args.weight_decay, ls=args.label_smoothing)
    nonrec_ce, nonrec_acc, _ = eval_ce(W_nr, nmean, nstd, nr_ev, V)

    # (3) VANILLA reservoir read-out: SAME reservoir cache/feature, calibration OFF (wd=0, ls=0, no Polyak averaging).
    W_van = train_readout(tr_cache, V, args.epochs, args.lr, np.random.default_rng(seed * 13 + 1), mean, std,
                          wd=0.0, ls=0.0, polyak=False)
    van_ce, van_acc, _ = eval_ce(W_van, mean, std, ev_cache, V)

    # (4) 4-GRAM ceiling (backoff to trigram/bigram), trained on the SAME train ids, held-out CE.
    ctx4 = fit_fourgram(tr_ids, V)
    four_ce, four_acc, _ = fourgram_ce(ctx4, ctx, P_bi, ev_ids)

    return {
        "bag_ce": bag_ce, "bag_acc": bag_acc,
        "nonrec_ce": nonrec_ce, "nonrec_acc": nonrec_acc, "nonrec_mean_rate": nonrec_rate,
        "vanilla_reservoir_ce": van_ce, "vanilla_reservoir_acc": van_acc,
        "fourgram_ce": four_ce, "fourgram_acc": four_acc,
        # per-seed control verdict (the reservoir must beat the bag AND the non-recurrent control by MARGIN; vanilla must
        # still beat the bigram; beating the 4-gram is a strengthener). Booleans mirror the aggregate verdict.
        "controls_verdict": {
            "reservoir_beats_bag": bool(res_ce < bag_ce - MARGIN),
            "reservoir_beats_nonrec": bool(res_ce < nonrec_ce - MARGIN),
            "vanilla_still_beats_bigram": bool(van_ce < bi_ce),
            "reservoir_beats_4gram": bool(res_ce < four_ce),
        },
    }


# ---------------------------------------------------------------------------------------------------------------------
# THE DE-RISK (one seed).
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_one(seed, args):
    stream = m62.build_stream(seed, n_sentences=args.n_sentences)
    sents = _split_sentences(stream)
    n = len(sents)
    n_tr = int(n * 0.8)
    train_all, eval_all = sents[:n_tr], sents[n_tr:]
    assert set(range(n_tr)).isdisjoint(range(n_tr, n)), "train/held-out index partition must be disjoint"

    # cap the sentences pushed through the reservoir (the bridge forward pass is the bottleneck); the bigram/trigram/
    # unigram are trained on the SAME capped train split so the CE comparison is fair.
    tr = train_all[:args.max_train_sents]
    ev = eval_all[:args.max_eval_sents]
    ctr = train_all[:args.max_ctrl_sents]
    cev = eval_all[:args.max_ctrl_eval]

    vocab = Vocab.build(tr, V=args.vocab)
    V = vocab.size
    tr_ids = [vocab.ids(s) for s in tr]
    ev_ids = [vocab.ids(s) for s in ev]

    # honest report of held-out string overlap with train (the template grammar makes string overlap expected; this is a
    # next-token-prediction test on fresh sampled instances, NOT a memorization test -- both models see the same held-out).
    train_strs = {" ".join(s) for s in tr}
    ev_str_overlap = float(np.mean([(" ".join(s) in train_strs) for s in ev])) if ev else 0.0

    res = ReservoirStates(V, seed=seed, n=args.n_pool)
    feat = args.state_feature

    # cache the FIXED reservoir states ONCE (the expensive bridge forward pass); the delta-rule epochs run over the cache.
    tr_cache = _cache(res, vocab, tr, feature=feat)
    ev_cache = _cache(res, vocab, ev, feature=feat)
    mean_rate = float(np.mean([np.mean(states) for states, _ in ev_cache if states])) if ev_cache else 0.0

    mean, std = _standardize_fit(tr_cache)
    # --vanilla flips OFF the calibration on the MAIN read-out (raw one-step delta only). Default (not --vanilla) keeps
    # wd=args.weight_decay, ls=args.label_smoothing, polyak=True -> BYTE-IDENTICAL to the original run.
    _van = bool(getattr(args, "vanilla", False))
    _main_wd = 0.0 if _van else args.weight_decay
    _main_ls = 0.0 if _van else args.label_smoothing
    W_main = train_readout(tr_cache, V, args.epochs, args.lr, np.random.default_rng(seed * 13 + 1), mean, std,
                           wd=_main_wd, ls=_main_ls, polyak=not _van)
    res_ce, res_acc, _n = eval_ce(W_main, mean, std, ev_cache, V)

    # n-gram baselines
    P_bi = fit_bigram(tr_ids, V)
    bi_ce, bi_acc, _ = bigram_ce(P_bi, ev_ids)
    ctx = fit_trigram(tr_ids, V)
    tri_ce, tri_acc, _ = trigram_ce(ctx, P_bi, ev_ids)
    P_uni = fit_unigram(tr_ids, V)
    uni_ce, _ = unigram_ce(P_uni, ev_ids)
    chance_ce = math.log(V)

    # ANTI-CHEAT A: shuffled reservoir state (permute S[t] within each held-out sentence)
    shufA = [_shuffle_states(states, ids, np.random.default_rng(seed * 17 + i)) for i, (states, ids) in enumerate(ev_cache)]
    shufA_ce, shufA_acc, _ = eval_ce(W_main, mean, std, shufA, V)

    # ANTI-CHEAT C: frozen (untrained, zeros) read-out -> CE == log(V)
    froz_ce, froz_acc, _ = eval_ce(np.zeros_like(W_main), mean, std, ev_cache, V)

    # ANTI-CHEAT B: permuted-corpus (shuffle train token order, recompute states, fresh read-out) -> eval on real held-out
    perm_sents = [_shuffle_tokens(s, np.random.default_rng(seed * 19 + i)) for i, s in enumerate(ctr)]
    perm_cache = _cache(res, vocab, perm_sents, feature=feat)
    pmean, pstd = _standardize_fit(perm_cache)
    W_perm = train_readout(perm_cache, V, args.epochs, args.lr, np.random.default_rng(seed * 23 + 1), pmean, pstd,
                           wd=args.weight_decay, ls=args.label_smoothing)
    permB_ce, permB_acc, _ = eval_ce(W_perm, pmean, pstd, ev_cache, V)

    # ANTI-CHEAT D: silenced reservoir (drive OFF) -> states carry no input -> read-out collapses to ~unigram
    sil_tr = _cache(res, vocab, ctr, silence=True, feature=feat)
    sil_ev = _cache(res, vocab, cev, silence=True, feature=feat)
    smean, sstd = _standardize_fit(sil_tr)
    W_sil = train_readout(sil_tr, V, args.epochs, args.lr, np.random.default_rng(seed * 29 + 1), smean, sstd,
                          wd=args.weight_decay, ls=args.label_smoothing)
    silD_ce, silD_acc, _ = eval_ce(W_sil, smean, sstd, sil_ev, V)

    # autoregressive rollouts (qualitative)
    rolls = []
    for st in ["the", "a", "it"]:
        if st in vocab.w2i:
            rolls.append((st, res.rollout(vocab, W_main, mean, std, st, args.rollout_len, feature=feat)))

    # per-seed GO
    beats = res_ce < bi_ce - MARGIN
    shufA_ok = shufA_ce >= bi_ce - EPS
    permB_ok = permB_ce >= bi_ce - EPS
    frozen_ok = abs(froz_ce - chance_ce) <= FROZEN_TOL
    silD_ok = silD_ce >= bi_ce - EPS
    active = mean_rate > ACTIVE_MIN
    seed_go = bool(beats and shufA_ok and permB_ok and frozen_ok and silD_ok and active)

    result = {
        "seed": seed, "V": V, "n_pool": res.n, "state_feature": feat,
        "n_sentences": n, "n_train_used": len(tr), "n_eval_used": len(ev),
        "heldout_string_overlap_train": ev_str_overlap, "mean_rate_per_neuron_step": mean_rate,
        "reservoir_ce": res_ce, "reservoir_acc": res_acc,
        "bigram_ce": bi_ce, "bigram_acc": bi_acc,
        "trigram_ce": tri_ce, "trigram_acc": tri_acc,
        "unigram_ce": uni_ce, "chance_ce": chance_ce,
        "shuffled_state_ce": shufA_ce, "permuted_corpus_ce": permB_ce, "frozen_ce": froz_ce, "silenced_ce": silD_ce,
        "shuffled_state_acc": shufA_acc, "permuted_corpus_acc": permB_acc, "silenced_acc": silD_acc,
        "seed_go": seed_go,
        "checks": {"beats_bigram": bool(beats), "shuffled_collapses": bool(shufA_ok),
                   "permuted_corpus_collapses": bool(permB_ok), "frozen_chance": bool(frozen_ok),
                   "silenced_collapses": bool(silD_ok), "active": bool(active)},
        "rollouts": [{"seed_token": st, "tokens": tk} for st, tk in rolls],
    }
    # ADDITIVE: adversarial controls (bag-of-prefix / non-recurrent projection / vanilla read-out / 4-gram). Gated behind
    # --controls -> the default run's returned dict (and JSON) is byte-unchanged.
    if getattr(args, "controls", False):
        result.update(_run_controls(seed, args, vocab, V, res, feat, tr, ev, tr_cache, ev_cache, tr_ids, ev_ids,
                                    P_bi, ctx, mean, std, res_ce, bi_ce))
    return result


def _print_seed(d):
    print(f"  [seed {d['seed']}] V={d['V']} feat={d['state_feature']} rate {d['mean_rate_per_neuron_step']:.4f}  ||  "
          f"RESERVOIR CE {d['reservoir_ce']:.3f} (acc {d['reservoir_acc']:.3f})  vs  bigram {d['bigram_ce']:.3f} "
          f"(acc {d['bigram_acc']:.3f})  [trigram {d['trigram_ce']:.3f} | unigram {d['unigram_ce']:.3f} | "
          f"chance {d['chance_ce']:.3f}]", flush=True)
    print(f"          anti-cheats -> shuffled-state {d['shuffled_state_ce']:.3f} | permuted-corpus "
          f"{d['permuted_corpus_ce']:.3f} | frozen {d['frozen_ce']:.3f} | silenced {d['silenced_ce']:.3f}  ||  "
          f"seed_go {d['seed_go']}  {d['checks']}", flush=True)
    if "bag_ce" in d:                                          # --controls: the adversarial control baselines
        print(f"          CONTROLS -> bag-of-prefix {d['bag_ce']:.3f} (acc {d['bag_acc']:.3f}) | non-recurrent "
              f"{d['nonrec_ce']:.3f} (acc {d['nonrec_acc']:.3f}, rate {d['nonrec_mean_rate']:.4f}) | vanilla-reservoir "
              f"{d['vanilla_reservoir_ce']:.3f} | 4-gram {d['fourgram_ce']:.3f}  ||  {d['controls_verdict']}", flush=True)
    for r in d["rollouts"]:
        print(f"          rollout '{r['seed_token']}' -> {' '.join(r['tokens'])}", flush=True)


def _derisk(seeds, args):
    print(f"EMERGE-RESERVOIR-LM de-risk: a fixed on-bridge spiking reservoir + a shallow one-step-delta read-out (NO "
          f"BPTT) as an autoregressive next-token LM; reservoir_CE < bigram_CE + every anti-cheat collapses; "
          f"{len(seeds)}-seed; state-feature={args.state_feature}", flush=True)
    t0 = time.time()
    err = None
    per = []
    controls_agg = None                                       # ADDITIVE: aggregate control block (--controls only)
    try:
        for s in seeds:
            d = _derisk_one(s, args)
            per.append(d)
            _print_seed(d)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        res_ce, bi_ce, tri_ce, uni_ce = m("reservoir_ce"), m("bigram_ce"), m("trigram_ce"), m("unigram_ce")
        res_acc, bi_acc = m("reservoir_acc"), m("bigram_acc")
        shufA, permB, froz, silD = m("shuffled_state_ce"), m("permuted_corpus_ce"), m("frozen_ce"), m("silenced_ce")
        rate = m("mean_rate_per_neuron_step")
        chance_ce = per[0]["chance_ce"]

        beats = res_ce < bi_ce - MARGIN
        shufA_ok = shufA >= bi_ce - EPS
        permB_ok = permB >= bi_ce - EPS
        frozen_ok = abs(froz - chance_ce) <= FROZEN_TOL
        silD_ok = silD >= bi_ce - EPS
        active = rate > ACTIVE_MIN
        go = bool(beats and shufA_ok and permB_ok and frozen_ok and silD_ok and active)

        # ADDITIVE: aggregate the adversarial controls (--controls only). The main GO gate above is UNCHANGED -- the
        # controls are reported alongside as extra fields + a separate controls_verdict (they do NOT alter `go`).
        if getattr(args, "controls", False) and per and "bag_ce" in per[0]:
            bag_ce, bag_acc = m("bag_ce"), m("bag_acc")
            nonrec_ce, nonrec_acc, nonrec_rate = m("nonrec_ce"), m("nonrec_acc"), m("nonrec_mean_rate")
            van_ce, van_acc = m("vanilla_reservoir_ce"), m("vanilla_reservoir_acc")
            four_ce, four_acc = m("fourgram_ce"), m("fourgram_acc")
            controls_verdict = {
                "reservoir_beats_bag": bool(res_ce < bag_ce - MARGIN),
                "reservoir_beats_nonrec": bool(res_ce < nonrec_ce - MARGIN),
                "vanilla_still_beats_bigram": bool(van_ce < bi_ce),
                "reservoir_beats_4gram": bool(res_ce < four_ce),
            }
            controls_agg = {
                "reservoir_ce": res_ce, "bigram_ce": bi_ce, "trigram_ce": tri_ce, "fourgram_ce": four_ce,
                "bag_ce": bag_ce, "bag_acc": bag_acc,
                "nonrec_ce": nonrec_ce, "nonrec_acc": nonrec_acc, "nonrec_mean_rate": nonrec_rate,
                "vanilla_reservoir_ce": van_ce, "vanilla_reservoir_acc": van_acc, "fourgram_acc": four_acc,
                "controls_verdict": controls_verdict,
            }

        if go:
            verdict = (
                f"GO -- a FIXED on-bridge spiking reservoir (EMERGE-82 OnBridgeLSM: a recurrent Izhikevich BrainRegion on "
                f"a real SimulationBridge, genuinely active at {rate:.4f} spikes/neuron/step) + a SHALLOW linear-softmax "
                f"read-out trained ONLINE by the ONE-STEP next-token delta rule (NO BPTT, NO weight transport -- the next "
                f"token IS the clean local target) is an autoregressive TOKEN-level LANGUAGE MODEL that BEATS the bigram on "
                f"held-out next-token cross-entropy: reservoir CE {res_ce:.3f} (acc {res_acc:.3f}) < bigram {bi_ce:.3f} "
                f"(acc {bi_acc:.3f}) by {bi_ce - res_ce:.3f} nats [trigram ceiling {tri_ce:.3f}, unigram floor {uni_ce:.3f}, "
                f"chance {chance_ce:.3f}] -- the reservoir's fading-memory recurrence carries higher-order context a bigram "
                f"cannot. Every anti-cheat COLLAPSES: shuffled-state (permute S[t]) CE {shufA:.3f} >= bigram (temporal "
                f"alignment load-bearing); permuted-corpus (scramble train word-order, fresh read-out) CE {permB:.3f} >= "
                f"bigram (no learnable structure); frozen (untrained) CE {froz:.3f} == chance; silenced (drive OFF) CE "
                f"{silD:.3f} >= bigram (the read is from the region's driven SPIKES). {len(seeds)} seeds, state-feature="
                f"{args.state_feature}. ==> emergent, learned-from-experience, on-bridge, no-BPTT language generation with "
                f"a local output-layer rule -- the ESN/LSM discipline (fixed recurrent cortex + locally-trained read-out) "
                f"on the project's own spiking substrate. HONEST SCOPE: a TOKEN-level LM over the BOUNDED controlled EMERGE "
                f"stream (a closed template grammar), NOT open prose (R4). Reuse-by-import (EMERGE-82 OnBridgeLSM + "
                f"EMERGE-61 wash-out + EMERGE-62 stream); NO sim/ edit; NO edit to any existing runner.")
        else:
            miss = []
            if not beats:
                miss.append(f"the reservoir does NOT beat the bigram (reservoir CE {res_ce:.3f} vs bigram {bi_ce:.3f}, "
                            f"need < {bi_ce - MARGIN:.3f}) -- the named next levers are more reservoir (n_pool) / more "
                            f"epochs / the per_window state feature")
            if not active:
                miss.append(f"the reservoir is nearly SILENT ({rate:.4f} spikes/neuron/step) -- the operating point "
                            f"(input/recurrent weights/bias) needs tuning")
            if not shufA_ok:
                miss.append(f"shuffled-state control did not collapse (CE {shufA:.3f} < bigram {bi_ce:.3f})")
            if not permB_ok:
                miss.append(f"permuted-corpus control did not collapse (CE {permB:.3f} < bigram {bi_ce:.3f})")
            if not frozen_ok:
                miss.append(f"frozen read-out CE {froz:.3f} != chance {chance_ce:.3f}")
            if not silD_ok:
                miss.append(f"silenced control did not collapse (CE {silD:.3f} < bigram {bi_ce:.3f})")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". An HONEST negative is the deliverable: if the reservoir does "
                       "not beat the bigram at this scale, the named next single-variable levers are more reservoir "
                       "(--n-pool), more epochs (--epochs), or the recency-emphasizing per_window state feature "
                       "(--state-feature per_window). Do NOT force GO.")
    else:
        go = False
        verdict = f"ERROR -- {err}"
        res_ce = bi_ce = tri_ce = uni_ce = res_acc = bi_acc = shufA = permB = froz = silD = rate = None

    summary = {
        "probe": "emerge_reservoir_lm", "verdict": verdict, "go": bool(go) if err is None else False,
        "state_feature": args.state_feature,
        "state_feature_choice": ("DEFAULT running_cumulative -- the OnBridgeLSM's own validated read (its final_state "
                                 "returns the cumulative-normalized pool rate); it integrates the WHOLE prefix (the "
                                 "fading memory lives in the recurrent dynamics; the cumulative read is a stable "
                                 "low-variance summary), which lets the next-token predictor exploit higher-order "
                                 "context a bigram cannot. per_window (spikes in token t's window) is the "
                                 "recency-emphasizing selectable alternative."),
        "mechanism": ("a FIXED on-bridge spiking reservoir (EMERGE-82 OnBridgeLSM: a recurrent Izhikevich BrainRegion on "
                      "a real SimulationBridge; internal_density recurrent conductance synapses; input one-hot over a "
                      "closed token vocab drives cp_external_input_current; the read feature = the region's cp_firing_"
                      "states spike-counts per token; EMERGE-61 wash-out between sentences) + a SHALLOW linear-softmax "
                      "read-out trained ONLINE by the ONE-STEP next-token delta rule (Widrow-Hoff on the clean, locally-"
                      "observed next-token error -- NO backprop-through-time, NO weight transport; reservoir + input "
                      "projection are fixed-random, only the output read-out is learned). Reuse-by-import; NO sim/ edit; "
                      "NO edit to any existing runner."),
        "task": ("autoregressive TOKEN-level next-token language model over the controlled EMERGE stream: does the fixed "
                 "reservoir + local-delta read-out BEAT the bigram on held-out next-token cross-entropy, while every "
                 "anti-cheat (shuffled-state, permuted-corpus, frozen read-out, silenced reservoir) collapses; the "
                 "region genuinely active; 6-seed; bridge on numpy/cupy"),
        "thresholds": {"margin_nats": MARGIN, "eps_nats": EPS, "frozen_tol": FROZEN_TOL, "active_min": ACTIVE_MIN},
        "params": {"n_pool": args.n_pool, "vocab": args.vocab, "epochs": args.epochs, "lr": args.lr,
                   "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
                   "n_sentences": args.n_sentences, "max_train_sents": args.max_train_sents,
                   "max_eval_sents": args.max_eval_sents, "max_ctrl_sents": args.max_ctrl_sents,
                   "max_ctrl_eval": args.max_ctrl_eval, "rollout_len": args.rollout_len},
        "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err is not None else {
            "reservoir_ce": res_ce, "reservoir_acc": res_acc, "bigram_ce": bi_ce, "bigram_acc": bi_acc,
            "trigram_ce": tri_ce, "unigram_ce": uni_ce, "chance_ce": per[0]["chance_ce"] if per else None,
            "shuffled_state_ce": shufA, "permuted_corpus_ce": permB, "frozen_ce": froz, "silenced_ce": silD,
            "mean_rate_per_neuron_step": rate,
        },
        "per_seed": per,
        "HONEST_NOTE": ("The PRIMARY emergence experiment: emergent, learned-from-experience, on-bridge, NO-BPTT "
                        "autoregressive next-token language generation with a LOCAL output-layer delta rule -- the "
                        "ESN/LSM discipline (a fixed random recurrent spiking cortex + a locally-trained output read-out) "
                        "on the project's own SimulationBridge. GO = the reservoir read-out beats the bigram on held-out "
                        "cross-entropy AND every input-destruction anti-cheat collapses (temporal alignment, corpus "
                        "word-order, trained read-out, and driven spikes all load-bearing). A TOKEN-level LM over the "
                        "BOUNDED controlled EMERGE template grammar (string overlap between held-out and train is EXPECTED "
                        "in a template grammar and is reported per seed; both models see the same held-out, so the CE "
                        "comparison is fair; this is a next-token-prediction test, not a memorization test), NOT open prose "
                        "(R4). Reuse-by-import; NO sim/ edit; NO edit to any existing runner."),
    }
    # ADDITIVE: only emit the `controls` field when --controls is set -> the default JSON is byte-unchanged.
    if getattr(args, "controls", False):
        summary["controls"] = controls_agg
    out_path = Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge-reservoir-lm] VERDICT: {verdict}", flush=True)
    if controls_agg is not None:
        c = controls_agg
        print(f"[emerge-reservoir-lm] CONTROLS (aggregate): reservoir_CE {c['reservoir_ce']:.3f} | bag_CE "
              f"{c['bag_ce']:.3f} | nonrec_CE {c['nonrec_ce']:.3f} | vanilla_reservoir_CE {c['vanilla_reservoir_ce']:.3f} "
              f"|| bigram_CE {c['bigram_ce']:.3f} | trigram_CE {c['trigram_ce']:.3f} | fourgram_CE {c['fourgram_ce']:.3f}",
              flush=True)
        print(f"[emerge-reservoir-lm] controls_verdict: {c['controls_verdict']}", flush=True)
        print("  INTERPRETATION: the reservoir's recurrent DYNAMICS are load-bearing ONLY if reservoir_CE beats BOTH the "
              "bag-of-prefix AND the non-recurrent projection (and vanilla still beats the bigram). If the bag/non-rec "
              "also beat the bigram OR the reservoir does not beat them, the win is NOT carried by the dynamics.", flush=True)
    print(f"[emerge-reservoir-lm] wrote {out_path}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-sentences", type=int, default=4000)
    ap.add_argument("--vocab", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.005,
                    help="delta-rule learning rate (0.05 diverges the softmax SGD -> 0.005 is the stable default)")
    ap.add_argument("--weight-decay", type=float, default=0.001, help="L2 decay on the read-out (calibration)")
    ap.add_argument("--label-smoothing", type=float, default=0.05, help="target label smoothing (calibration)")
    ap.add_argument("--n-pool", type=int, default=_N_POOL)
    ap.add_argument("--state-feature", choices=["running_cumulative", "per_window"], default="running_cumulative")
    ap.add_argument("--max-train-sents", type=int, default=1500, help="cap on train sentences pushed through the reservoir")
    ap.add_argument("--max-eval-sents", type=int, default=300, help="cap on held-out sentences pushed through the reservoir")
    ap.add_argument("--max-ctrl-sents", type=int, default=300, help="cap on train sentences for the permuted/silenced controls")
    ap.add_argument("--max-ctrl-eval", type=int, default=120, help="cap on held-out sentences for the silenced control")
    ap.add_argument("--rollout-len", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--demo", action="store_true", help="single-seed verbose run (uses seeds[0])")
    ap.add_argument("--derisk", action="store_true", help="(compat) run the seed sweep -- the default action anyway")
    ap.add_argument("--controls", action="store_true",
                    help="ADVERSARIAL controls (additive; default OFF -> byte-identical run). Runs, alongside the normal "
                         "run: (1) a BAG-OF-PREFIX read-out (same delta rule, feature = unordered prefix token counts, no "
                         "reservoir), (2) a NON-RECURRENT memoryless projection (zeroed recurrent weights, per-token reset "
                         "+ per-window read), (3) a VANILLA reservoir read-out (calibration off), (4) a 4-gram ceiling; adds "
                         "bag_ce/nonrec_ce/vanilla_reservoir_ce/fourgram_ce + a controls_verdict to the JSON. The reservoir "
                         "dynamics are load-bearing ONLY if reservoir_CE beats BOTH the bag AND the non-recurrent control.")
    ap.add_argument("--vanilla", action="store_true",
                    help="disable the MAIN read-out calibration (Polyak tail-averaging + weight-decay + label-smoothing) "
                         "-> the RAW one-step delta rule only. Confirms the reservoir's win is not a calibration artifact. "
                         "Non-default (opt-in); the default run is byte-identical.")
    a = ap.parse_args()
    if a.demo:
        return _derisk([a.seeds[0]], a)
    return _derisk(a.seeds, a)


if __name__ == "__main__":
    raise SystemExit(main())
