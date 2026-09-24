"""AFFECT LEARNED VOCABULARY — word -> valence synapses learned by a local Hebbian rule from heard language, read by a
spiking opponent pair (V+ / V-), feeding `affect_production_organ.appraise_text` behind a DEFAULT-OFF flag
(lane A · Affect, 2026-09-24, branch research/affect-learned-vocabulary).

WHY. The content-locked tone-selection probe (research/findings/2026-09-24-affect-tone-selection-content-locked-PREREG.md,
AMENDMENT 2) found that the appraisal only hears the 180-word WARRINER list: "sadness", "sorrow", "melancholy", "loss",
"alone" read 0.0. The organ cannot perceive most negative wording, in the held mood or in a candidate reply.

WHAT THIS IS (every step between the heard word and the valence read is neurons / synapses; declared exceptions below):
  * ENVIRONMENT (host, legitimate): the heard corpus, streamed in consecutive CHUNK-token presentations. Each content
    word present drives ITS lexical afferent; each innate seed word present also drives ITS innate (US) afferent.
  * INNATE EDGE (fixed, declared scaffold): US afferent of seed word s -> V+ (if its norm valence > 0) or V- (< 0),
    weight W_US * |v_s|. The seeds are the strong WARRINER words (|v-5| >= 2) that the organ already hears; per brain
    seed a cross-validation split keeps SEED_FRAC of them innate and holds the rest out (their sign is a test).
  * LEARNED EDGE (plastic): lexical afferent of word i -> V+ and -> V-. Starts at zero (no drive).
  * COMPETITION: V+ and V- each drive their own FS interneuron pool, which inhibits the OTHER pool (reciprocal lateral
    inhibition; the motif of lexicon_spiking_frame_category / _affect_marker_wta_derisk).
  * LEARNING (synapse-local, evaluative conditioning; De Houwer, Thomas & Baeyens 2001, Psychol Bull 127:853). Each
    presentation has two phases, like delay conditioning: the heard words alone for T_CS steps (the pools' CS-evoked
    response, which carries the LEARNED drive), then the words plus the innate US afferents for T_ON steps. For every
    lexical afferent i that fired and each pool P:
        inc_P = y2_P - y1_P                      (the pool's US-evoked increment over its CS-alone rate)
        e_P   = (inc_P - theta_P) / rms_P
        u_iP <- u_iP + x_i * (e_P - u_iP) * max(1/(n_i + N0), ETA_MIN),     n_i <- n_i + x_i
    x_i = the afferent's spike rate normalised to its drive rate (~1 when heard). theta_P = the pool's SLIDING
    THRESHOLD, a slow trace of its own increment (BCM; Bienenstock, Cooper & Munro 1982; the metaplastic sliding
    threshold of Abraham & Bear 1996), time constant TAU_THETA presentations. rms_P = the running RMS of the pool's
    excess (multiplicative SYNAPTIC SCALING; Turrigiano et al. 1998), time constant TAU_SCALE: it puts the quiet V-
    pool and the busier V+ pool on the same footing. The weight is w_iP = G * max(u_iP, 0) (an excitatory conductance
    cannot be negative; the share of synapses at the floor is reported). The rule is an INSTAR (Grossberg): the weight
    tracks the mean scaled excess, conditional on the word being heard. Plasticity opens after WARMUP presentations
    (the running statistics settle first).
    WHY the increment (the stability companion): with a plain post-rate Hebbian rule every learned word drives the
    pools in every chunk it is heard, and the loop runs away (measured on a synthetic stream: one pool came to fire in
    every presentation and every word learned its sign). The learned drive is present in both phases, so it cancels in
    the increment; once the CS drive saturates a pool the US adds less, so learning slows as the prediction grows (a
    Rescorla-Wagner-like saturation, Rescorla & Wagner 1972).
    WHY the sliding threshold (CLAUDE.md "what else does the real system run alongside this"): the prior learned-gate
    attempts read valence off whole-corpus co-occurrence, and the register confound
    (research/findings/2026-09-05-affect-learned-gate-retry-register-confound-BOUNDARY.md) made every word of a warm
    register read warm. A threshold that tracks the pool's RECENT increment subtracts the register of the passage being
    heard, so only words whose own contexts are MORE (or less) affective than their passage learn.
    WHY the use-dependent rate (synaptic maturation; the cascade model of Fusi, Drew & Abbott 2005): a young synapse
    changes fast and a mature one slowly, so w converges to the mean excess; N0 is the initial maturity. It shrinks
    the weights of RARELY heard words toward zero, which keeps a few chance co-occurrences from crossing the pool
    threshold. ETA_MIN keeps a mature synapse slowly forgetting.
    US DEPRESSION (optional, U_DEP > 0): each US synapse depresses by U_DEP when its seed is heard and recovers with
    TAU_REC presentations (short-term depression, Tsodyks & Markram 1997): very frequent generic evaluatives ("good")
    then teach less than rare specific ones.
    The rule is applied runner-side to the bridge's own synapse entries (as in lexicon_spiking_frame_category: the
    engine's generic Hebbian path clips every synapse to hebbian_max_weight). The US supplies the teaching signal
    through fixed innate synapses, so the learning is "innate-US-driven", not "self-organized" (docs/TERMS.md); the
    seed valences are host-given norms.
  * READ-OUT (instrument): a word is presented ALONE (its lexical afferent only) for T_READ steps; the host reads the
    two pool rates. A word whose learned weights do not bring the pools over threshold produces no spikes and reads
    exactly 0 — the pool's firing threshold is the salience gate. valence = clip((r+ - r-) / R_REF, -1, 1). The winner
    is decided by the spiking competition; the reading of the rates is a host read-out ("spiking with a host
    read-out", never "fully spiking").

EXECUTION CONVENIENCE (declared): the lexical layer is executed as CHUNK "slot" afferent neurons. Before each
presentation the learned synapses of the words heard in it are copied into the slot neurons' synapses. This is the
same circuit as one afferent per vocabulary word because a silent afferent delivers no current; what differs is the
per-neuron heterogeneity of the afferent (all slot neurons are the same few neurons). A full localist layer of ~30k
afferents costs ~2.7 ms/step on numpy; the slot layer makes a multi-million-token stream affordable.

REPLICAS: one stream feeds R independent opponent circuits. Replica 0 has the true seed valences; replicas 1..R-1
have the seed valences PERMUTED across seed words (the shuffled-label control, a distribution, not one shuffle). All
replicas hear the same words at the same time.

LESIONS: "learned_edge" zeroes every learned synapse (u := 0); the innate edge and the circuit stay.

HONEST RESIDUALS (declared): seed valences are host norms (WARRINER); the corpus chunking and the sliding-threshold
time constant are host protocol; the read of the rates and the valence scaling are a host read-out; the slot execution
above; per-presentation learning is applied runner-side from spike RATES (not an STDP kernel).

Smoke: SIM_BACKEND=numpy python -m research.runners.affect_learned_vocabulary --smoke
"""
from __future__ import annotations

import collections
import hashlib
import os
import re
import sys
import zlib

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._affect_distributional_tag_derisk import WARRINER, STOP  # noqa: E402

_WORD_RE = re.compile(r"[a-z]+")

# ── protocol / circuit constants. Calibrated on DEV seed 7 only (NOT an evaluation seed); see the PREREG. ──────────
CORPUS = (("data/corpus/fineweb_edu.txt", 1_000_000_000),)   # the heard stream (first 1e9 characters)
MIN_HEARD = 20            # a word heard fewer times has no lexical afferent (reads 0)
CHUNK = 13                # tokens per presentation (a sentence-scale window)
SEED_MARGIN = 2.0         # |v-5| >= this -> an innate seed (the organ's own strong-affect set)
SEED_FRAC = 0.8           # per brain seed: fraction of seeds kept innate; the rest are held-out tests
N_CAT = 20                # excitatory neurons per valence pool
N_FSI = 10                # FS interneurons per pool's cross-inhibition sub-pool
T_CS = 6                  # steps of the CS-alone phase of a presentation
T_ON = 12                 # steps of the CS+US phase of a presentation
T_READ = 30               # steps per read presentation
I_AFF = 4000.0            # afferent drive while its word is heard (pA)
W_US = 600.0              # innate US weight scale (x |v_s|)
TO_FSI_W = 70.0
CROSS_W = 22.0
TAU_THETA = 80.0          # sliding-threshold time constant (presentations, ~1000 tokens)
N0 = 50.0                 # initial synapse maturity (pseudo-count)
G = 500.0                 # weight per unit of learned (RMS-scaled) excess (bridge weight units)
R_REF = 0.10              # read-out scale: rate margin (spikes/step/neuron) that maps to |valence| = 1
MIN_RATE = 0.002          # below this in BOTH pools = no read (0)
U_DEP = 0.0               # US synapse utilisation per heard presentation (short-term depression); 0 = off
TAU_REC = 100.0           # US resource recovery time constant (presentations)
TAU_SCALE = 2000.0        # synaptic-scaling time constant (presentations); 0 = off (raw rate units)
WARMUP = 5000             # presentations before plasticity opens (theta / scaling / depression settle first)
ETA_MIN = 1e-4            # the floor of the use-dependent learning rate (a mature synapse still forgets slowly)

LESION_KINDS = ("learned_edge",)


def _stable_int(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


# ── the heard corpus (ENVIRONMENT) ────────────────────────────────────────────────────────────────────────────────
def _chunks(corpus, block=8_000_000):
    for path, max_chars in corpus:
        p = path if os.path.isabs(path) else os.path.join(_REPO, path)
        with open(p, encoding="utf-8", errors="ignore") as fh:
            left = int(max_chars)
            while left > 0:
                s = fh.read(min(block, left))
                if not s:
                    break
                left -= len(s)
                yield [t for t in _WORD_RE.findall(s.lower()) if t != "endoftext"]


class HeardStream:
    """Vocabulary (content words heard >= MIN_HEARD) and the token-id stream (-1 = not a content word)."""

    def __init__(self, corpus=CORPUS, min_heard: int = MIN_HEARD):
        cnt = collections.Counter()
        for c in _chunks(corpus):
            cnt.update(c)
        self.vocab = sorted(w for w, c in cnt.items() if c >= min_heard and w not in STOP and len(w) >= 3)
        self.vid = {w: i for i, w in enumerate(self.vocab)}
        self.count = np.array([cnt[w] for w in self.vocab], dtype=np.int64)
        total = int(sum(cnt.values()))
        del cnt
        self.ids = np.empty(total, dtype=np.int32)          # preallocated: one copy of the stream in memory
        k = 0
        for c in _chunks(corpus):
            a = np.fromiter((self.vid.get(t, -1) for t in c), dtype=np.int32, count=len(c))
            self.ids[k:k + len(a)] = a
            k += len(a)
        if k != total:
            raise RuntimeError(f"stream length changed between passes ({k} != {total})")
        self.n_tok = int(total)
        self.corpus = tuple(corpus)

    @property
    def V(self):
        return len(self.vocab)

    def presentations(self, chunk: int = CHUNK):
        n = self.n_tok // chunk
        return self.ids[: n * chunk].reshape(n, chunk)


def seed_split(seed: int, seed_margin: float = SEED_MARGIN, frac: float = SEED_FRAC):
    """(innate {word: signed valence in [-1,1]}, held-out {word: signed valence}) — per-seed CV split."""
    strong = {w: (v - 5.0) / 4.0 for w, (v, a) in WARRINER.items() if abs(v - 5.0) >= seed_margin}
    ws = sorted(strong)
    perm = list(np.random.default_rng(seed).permutation(ws))
    k = int(round(frac * len(ws)))
    return {w: strong[w] for w in perm[:k]}, {w: strong[w] for w in perm[k:]}


def shuffled_valences(innate: dict, seed: int, r: int) -> dict:
    """Replica r >= 1: the innate valences permuted across the innate words (sign AND magnitude)."""
    ws = sorted(innate)
    vals = np.array([innate[w] for w in ws])
    p = np.random.default_rng(seed * 1000 + 17 * r + 1).permutation(len(ws))
    return {w: float(vals[p[i]]) for i, w in enumerate(ws)}


# ── the circuit ──────────────────────────────────────────────────────────────────────────────────────────────────
def _region(name, n, *, exc_fraction, neuron_type):
    from sim.regions import BrainRegion
    return BrainRegion(name=name, n_neurons=int(n), exc_fraction=float(exc_fraction), internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                       izh_neuron_type=neuron_type.name, enable_homeostasis=False)


def build_circuit(seed: int, n_slot: int, n_us: int, n_replicas: int):
    """SL (lexical slots) + US (innate seed afferents) -> R x {VP, VN} with reciprocal FSI lateral inhibition."""
    from sim import CoreSimConfig, GPUConfig, RuntimeState, SimulationBridge, VisualizationConfig
    from sim.enums import NeuronType
    from sim.regions import RegionPathway
    rs, fs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL, NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = False
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    regions = [_region("SL", n_slot, exc_fraction=1.0, neuron_type=rs)]
    if n_us:
        regions.append(_region("US", n_us, exc_fraction=1.0, neuron_type=rs))
    pathways = []
    for r in range(n_replicas):
        for p in ("VP", "VN"):
            regions.append(_region(f"{p}{r}", N_CAT, exc_fraction=1.0, neuron_type=rs))
        for p in ("IP", "IN"):
            regions.append(_region(f"{p}{r}", N_FSI, exc_fraction=0.0, neuron_type=fs))
        for p in ("VP", "VN"):
            # weight 1.0 placeholders; the real weights are installed per presentation (0 until learned)
            pathways.append(RegionPathway(from_region="SL", to_region=f"{p}{r}", density=1.0, weight_mean=1.0,
                                          weight_jitter=0.0, plastic=False))
            if n_us:
                pathways.append(RegionPathway(from_region="US", to_region=f"{p}{r}", density=1.0, weight_mean=1.0,
                                              weight_jitter=0.0, plastic=False))
        pathways.append(RegionPathway(from_region=f"VP{r}", to_region=f"IP{r}", density=1.0, weight_mean=TO_FSI_W,
                                      weight_jitter=0.05, plastic=False))
        pathways.append(RegionPathway(from_region=f"VN{r}", to_region=f"IN{r}", density=1.0, weight_mean=TO_FSI_W,
                                      weight_jitter=0.05, plastic=False))
        pathways.append(RegionPathway(from_region=f"IP{r}", to_region=f"VN{r}", density=1.0, weight_mean=CROSS_W,
                                      weight_jitter=0.05, plastic=False, receptor="gaba_a"))
        pathways.append(RegionPathway(from_region=f"IN{r}", to_region=f"VP{r}", density=1.0, weight_mean=CROSS_W,
                                      weight_jitter=0.05, plastic=False, receptor="gaba_a"))
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b


def _synapse_slots(b, pre_idx, post_groups):
    """S (len(pre) x total_post) of cp_connections.data indices for pre -> each post neuron (CSR row = pre)."""
    from sim.backend import to_host
    indptr = np.asarray(to_host(b.cp_connections.indptr))
    indices = np.asarray(to_host(b.cp_connections.indices))
    n = int(b.core_config.num_neurons)
    col_slot = np.full(n, -1, dtype=np.int64)
    flat = np.concatenate(post_groups)
    col_slot[flat] = np.arange(len(flat))
    S = np.full((len(pre_idx), len(flat)), -1, dtype=np.int64)
    for a, p in enumerate(pre_idx):
        ks = np.arange(indptr[p], indptr[p + 1])
        sl = col_slot[indices[ks]]
        ok = sl >= 0
        S[a, sl[ok]] = ks[ok]
    if (S < 0).any():
        raise RuntimeError("pathway is not all-to-all (missing synapses)")
    return S


class LearnedAffectVocabulary:
    """See the module docstring. `train(stream)` then `read(word)`; `u` has shape (R, V, 2) [V+, V-]."""

    def __init__(self, seed: int, vocab, innate: dict, n_replicas: int = 1, *, replica_ids=None, n_slot: int = CHUNK, g=G,
                 n0=N0, tau_theta=TAU_THETA, r_ref=R_REF, w_us=W_US, t_on=T_ON, t_read=T_READ, i_aff=I_AFF,
                 u_dep=U_DEP, tau_rec=TAU_REC, tau_scale=TAU_SCALE, warmup=WARMUP, eta_min=ETA_MIN, t_cs=T_CS):
        from sim.backend import to_host
        self.replica_ids = list(range(int(n_replicas))) if replica_ids is None else [int(r) for r in replica_ids]
        self.seed, self.R = int(seed), len(self.replica_ids)
        self.vocab = list(vocab)
        self.vid = {w: i for i, w in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.g, self.n0, self.tau, self.r_ref, self.w_us = float(g), float(n0), float(tau_theta), float(r_ref), float(w_us)
        self.t_on, self.t_read, self.i_aff = int(t_on), int(t_read), float(i_aff)
        self.innate = dict(innate)
        self.us_words = sorted(w for w in self.innate if w in self.vid)
        self.us_of_vid = np.full(self.V, -1, dtype=np.int64)
        for k, w in enumerate(self.us_words):
            self.us_of_vid[self.vid[w]] = k
        self.n_slot = int(n_slot)
        self.b = build_circuit(self.seed, self.n_slot, len(self.us_words), self.R)
        rm = self.b.region_manager
        ix = lambda name: np.asarray(list(rm.indices(name)), dtype=np.int64)  # noqa: E731
        self.sl = ix("SL")
        self.us = ix("US") if self.us_words else np.zeros(0, dtype=np.int64)
        self.vp = [ix(f"VP{r}") for r in range(self.R)]
        self.vn = [ix(f"VN{r}") for r in range(self.R)]
        self.n = int(self.b.core_config.num_neurons)
        post = []
        for r in range(self.R):
            post += [self.vp[r], self.vn[r]]
        self.S_sl = _synapse_slots(self.b, self.sl, post).reshape(self.n_slot, self.R, 2, N_CAT)
        # innate edge (fixed): replica 0 true valences, replicas >= 1 permuted
        # replica id 0 = the true seed valences; id >= 1 = a permutation (the shuffled-label control)
        self.valences = [self.innate if rid == 0 else shuffled_valences(self.innate, self.seed, rid)
                         for rid in self.replica_ids]
        data = np.asarray(to_host(self.b.cp_connections.data)).astype(np.float64).copy()
        self.S_us = None
        if self.us_words:
            self.S_us = _synapse_slots(self.b, self.us, post).reshape(len(self.us_words), self.R, 2, N_CAT)
            self.us_base = np.zeros((len(self.us_words), self.R, 2))
            for r in range(self.R):
                for k, w in enumerate(self.us_words):
                    v = self.valences[r][w]
                    self.us_base[k, r, 0] = self.w_us * v if v > 0 else 0.0
                    self.us_base[k, r, 1] = self.w_us * (-v) if v < 0 else 0.0
            data[self.S_us] = self.us_base[:, :, :, None]
        # US short-term depression (habituation): available resources per US afferent (Tsodyks & Markram 1997)
        self.us_res = np.ones(len(self.us_words))
        self.u_dep, self.tau_rec = float(u_dep), float(tau_rec)
        data[self.S_sl] = 0.0
        self._data = data
        self._push()
        self.u = np.zeros((self.R, self.V, 2))
        self.nuse = np.zeros(self.V)
        self.theta = np.zeros((self.R, 2))
        # synaptic scaling: per pool running mean-square of the excess (the learning signal is divided by its RMS)
        self.tau_scale = float(tau_scale)
        self.warmup, self.eta_min, self.t_cs = int(warmup), float(eta_min), int(t_cs)
        self.msq = np.full((self.R, 2), 1e-4)
        self.lesion = None
        self._cache = {}
        self.n_presented = 0

    # ── bridge I/O ───────────────────────────────────────────────────────────────────────────────────────────────
    def _push(self):
        from sim.backend import from_host
        self.b.cp_connections.data = from_host(self._data.astype(np.float32))

    def weights_of(self, wids):
        """(len(wids), R, 2) installed weights: G * max(u, 0), or 0 under the learned_edge lesion."""
        if self.lesion == "learned_edge":
            return np.zeros((len(wids), self.R, 2))
        return self.g * np.maximum(self.u[:, wids, :], 0.0).transpose(1, 0, 2)

    def _install_slots(self, wids):
        W = self.weights_of(wids)                                  # (k, R, 2)
        self._data[self.S_sl] = 0.0
        k = len(wids)
        if k:
            self._data[self.S_sl[:k]] = W[:, :, :, None]
        self._push()

    def set_lesion(self, kind):
        if kind not in (None,) + LESION_KINDS:
            raise ValueError(kind)
        self.lesion = kind
        self._cache.clear()

    def reset_state(self):
        b = self.b
        if getattr(b, "cp_izh_c_reset", None) is not None:
            b.cp_membrane_potential_v[:] = b.cp_izh_c_reset
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        for a in ("cp_firing_states", "cp_prev_firing_states"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = False
        for a in ("cp_conductance_g_e", "cp_conductance_g_i"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = 0.0

    def _run(self, wids, us_k, steps):
        """Drive slots 0..len(wids)-1 and US afferents us_k for `steps`; return per-neuron spike counts."""
        from sim.backend import to_host, from_host
        if self.S_us is not None and len(us_k):
            # depressed US efficacy for the afferents heard in this presentation
            self._data[self.S_us[us_k]] = (self.us_base[us_k] * self.us_res[us_k, None, None])[:, :, :, None]
        self._install_slots(wids)
        cur = np.zeros(self.n, dtype=np.float32)
        cur[self.sl[: len(wids)]] = self.i_aff
        if len(us_k):
            cur[self.us[us_k]] = self.i_aff
        dev = from_host(cur)
        counts = np.zeros(self.n)
        b = self.b
        for _ in range(int(steps)):
            b.cp_external_input_current[:] = dev
            b._run_one_simulation_step()
            counts += np.asarray(to_host(b.cp_firing_states), dtype=np.float64)
        b.cp_external_input_current[:] = 0.0
        return counts

    def _pool_rates(self, counts, steps):
        yp = np.array([counts[self.vp[r]].mean() / steps for r in range(self.R)])
        yn = np.array([counts[self.vn[r]].mean() / steps for r in range(self.R)])
        return yp, yn

    # ── learning ───────────────────────────────────────────────────────────────────────────────────────────────
    def present_and_learn(self, row):
        """One heard presentation (a row of token ids, -1 = no afferent), in two phases (delay conditioning): the
        heard words alone for T_CS steps (the pools' CS-evoked response, which carries the LEARNED drive), then the
        words plus the innate US afferents for T_ON steps. The learning signal is the US-evoked INCREMENT of each
        pool's rate over its CS-alone rate, minus that increment's sliding threshold, in per-pool RMS units:
            e_P = ((y2_P - y1_P) - theta_P) / rms_P,   u_iP <- u_iP + x_i * (e_P - u_iP) * max(1/(n_i+N0), ETA_MIN)
        The learned drive appears in BOTH phases, so it cancels in the increment: a word cannot potentiate itself
        through its own learned drive (the positive-feedback runaway a plain post-rate Hebbian rule has here). Once
        the CS drive saturates a pool, the US adds less, so learning slows as the prediction grows (Rescorla-Wagner-
        like saturation). Returns the increment (R, 2)."""
        wids = np.unique(row[row >= 0])[: self.n_slot]
        us_k = self.us_of_vid[wids]
        us_k = us_k[us_k >= 0]
        c1 = self._run(wids, np.zeros(0, dtype=np.int64), self.t_cs)
        c2 = self._run(wids, us_k, self.t_on)
        y1 = np.stack(self._pool_rates(c1, self.t_cs), axis=1)      # (R, 2) CS alone
        y2 = np.stack(self._pool_rates(c2, self.t_on), axis=1)      # (R, 2) CS + US
        inc = y2 - y1
        if len(wids) and self.n_presented >= self.warmup:
            x = np.minimum(1.0, (c1 + c2)[self.sl[: len(wids)]] / (self.t_cs + self.t_on) / self._x_ref)   # (k,)
            ex = inc - self.theta                                    # (R, 2) excess over the sliding threshold
            if self.tau_scale > 0:
                ex = ex / np.sqrt(self.msq)                          # homeostatic scaling (per-pool RMS units)
            eta = x * np.maximum(1.0 / (self.nuse[wids] + self.n0), self.eta_min)   # (k,)
            du = eta[None, :, None] * (ex[:, None, :] - self.u[:, wids, :])
            self.u[:, wids, :] += du
            self.nuse[wids] += x
        if self.tau_scale > 0:
            self.msq += ((inc - self.theta) ** 2 - self.msq) / self.tau_scale
        self.theta += (inc - self.theta) / self.tau
        if self.S_us is not None:
            # use-dependent depression of the US synapses heard now, then recovery of all (one presentation)
            self.us_res[us_k] *= (1.0 - self.u_dep)
            self.us_res = 1.0 - (1.0 - self.us_res) * self._rec
        self.n_presented += 1
        return inc

    def calibrate_afferent(self):
        """The afferent's own spike rate when driven (normalises x to ~1). Read once, on a silent circuit."""
        saved = self.lesion
        self.lesion = "learned_edge"
        self.reset_state()
        c = self._run(np.array([0]), np.zeros(0, dtype=np.int64), 200)
        self.lesion = saved
        self._x_ref = max(1e-6, float(c[self.sl[0]]) / 200.0)
        self.reset_state()
        return self._x_ref

    def train(self, rows, log_every=0, log=print):
        self._rec = float(np.exp(-1.0 / self.tau_rec)) if self.tau_rec > 0 else 0.0
        if not hasattr(self, "_x_ref"):
            self.calibrate_afferent()
        saved, self.lesion = self.lesion, None
        self.reset_state()
        for k, row in enumerate(rows):
            self.present_and_learn(row)
            if log_every and (k + 1) % log_every == 0:
                log(f"  presented {k + 1}/{len(rows)} theta={np.round(self.theta[0], 4).tolist()}")
        self.lesion = saved
        self._cache.clear()

    # ── read-out ───────────────────────────────────────────────────────────────────────────────────────────────
    def read_rates(self, word: str):
        """(rP[R], rN[R]) when `word` is heard alone for T_READ steps; None if the word has no afferent."""
        w = (word or "").lower()
        key = (w, self.lesion)
        if key in self._cache:
            return self._cache[key]
        i = self.vid.get(w)
        if i is None:
            out = None
        else:
            self.reset_state()
            counts = self._run(np.array([i]), np.zeros(0, dtype=np.int64), self.t_read)
            out = self._pool_rates(counts, self.t_read)
            self.reset_state()
        self._cache[key] = out
        return out

    def read(self, word: str, replica: int = 0) -> float:
        """Signed valence in [-1, 1]; exactly 0.0 when neither pool reaches MIN_RATE (or the word has no afferent)."""
        rr = self.read_rates(word)
        if rr is None:
            return 0.0
        rp, rn = float(rr[0][replica]), float(rr[1][replica])
        if max(rp, rn) < MIN_RATE:
            return 0.0
        return float(np.clip((rp - rn) / self.r_ref, -1.0, 1.0))

    def read_all(self, word: str):
        rr = self.read_rates(word)
        if rr is None:
            return [0.0] * self.R
        return [self.read(word, r) for r in range(self.R)]

    # ── persistence (the learned synapses ARE the brain's learned state) ─────────────────────────────────────
    def state_hash(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.u).tobytes()).hexdigest()

    def save(self, path: str, replica: int = 0, meta=None, only_words=None):
        """Save one replica (by POSITION in replica_ids) — its learned synapses, use counts and the circuit calibration.
        only_words: keep just these words (the shuffled-control replicas are only ever read on the evaluation items)."""
        nz = np.abs(self.u[replica]).sum(axis=1) > 0
        if only_words is not None:
            sel = np.zeros(self.V, dtype=bool)
            sel[[self.vid[w] for w in only_words if w in self.vid]] = True
            nz &= sel
        keep = np.flatnonzero(nz)
        np.savez_compressed(path, words=np.array([self.vocab[i] for i in keep]), u=self.u[replica][keep],
                            nuse=self.nuse[keep], innate_words=np.array(sorted(self.innate)),
                            innate_vals=np.array([self.innate[w] for w in sorted(self.innate)]),
                            x_ref=np.array([getattr(self, "_x_ref", 1.0)]),
                            replica_id=np.array([self.replica_ids[replica]]), g=np.array([self.g]),
                            meta=np.array([repr(meta or {})]))


# ── production reader (appraise_text, flag BRAIN_AFFECT_LEARNED_VOCAB) ──────────────────────────────────────────────
def learned_vocab_enabled() -> bool:
    """DEFAULT-OFF. `BRAIN_AFFECT_LEARNED_VOCAB` in {1,true,yes,on} -> appraise_text consults the learned vocabulary
    for words OUTSIDE the WARRINER list."""
    return os.environ.get("BRAIN_AFFECT_LEARNED_VOCAB", "").strip().lower() in ("1", "true", "yes", "on")


def learned_vocab_lesioned() -> bool:
    return os.environ.get("BRAIN_AFFECT_LEARNED_VOCAB_LESION", "").strip().lower() in ("1", "true", "yes", "on")


DEFAULT_WEIGHTS_DIR = os.path.join(_REPO, "research", "findings", "raw", "_affect_learned_vocab", "run1")
_READER = {}


def load_reader(path: str, seed: int):
    """Rebuild a replica-0 read circuit from saved learned synapses."""
    z = np.load(path, allow_pickle=False)
    words = [str(w) for w in z["words"]]
    lav = LearnedAffectVocabulary(seed, words, innate={}, n_replicas=1, n_slot=1)
    lav.u[0] = z["u"]
    lav.nuse[:] = z["nuse"]
    lav.g = float(z["g"][0])                     # the synaptic gain the weights were learned with
    lav._x_ref = float(z["x_ref"][0])
    return lav


def get_reader(seed: int = 42):
    """The process-shared reader for `seed` (weights from BRAIN_AFFECT_LEARNED_VOCAB_PATH or the default run dir).
    Missing weights -> None (appraise_text then behaves as flag-off; the caller records it)."""
    key = int(seed)
    if key not in _READER:
        path = os.environ.get("BRAIN_AFFECT_LEARNED_VOCAB_PATH") or os.path.join(
            DEFAULT_WEIGHTS_DIR, f"weights_s{key}.npz")
        _READER[key] = load_reader(path, key) if os.path.exists(path) else None
    r = _READER[key]
    if r is not None:
        r.set_lesion("learned_edge" if learned_vocab_lesioned() else None)
    return r


def learned_valence(word: str, seed: int = 42) -> float:
    r = get_reader(seed)
    return 0.0 if r is None else r.read(word)


if __name__ == "__main__":
    import argparse
    import time
    os.environ.setdefault("SIM_BACKEND", "numpy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    st = HeardStream(corpus=(("data/corpus/simplewiki.txt", 3_000_000),))
    innate, held = seed_split(a.seed)
    lav = LearnedAffectVocabulary(a.seed, st.vocab, innate, n_replicas=9)
    rows = st.presentations()
    print(f"V={st.V} presentations={len(rows)} x_ref={lav.calibrate_afferent():.3f} build {time.time() - t0:.1f}s")
    t1 = time.time()
    lav.train(rows[:20000], log_every=5000)
    print(f"train {time.time() - t1:.1f}s ({(time.time() - t1) / 20000 * 1000:.2f} ms/presentation)")
    for w in ("sad", "war", "death", "sadness", "sorrow", "cat", "city", "physicist"):
        print(w, lav.read_all(w), lav.read_rates(w))
    # ATTRIBUTION: how much of "war"'s V- read is carried by the LEARNED synapses (control = learned_edge lesion)?
    from tools.lab import attributable_to
    rr = lav.read_rates("war")
    lav.set_lesion("learned_edge")
    rr0 = lav.read_rates("war")
    lav.set_lesion(None)
    if rr is not None and rr0 is not None:
        attributable_to("war V- rate: learned vs lesioned synapses", float(rr[1][0]), float(rr0[1][0]))
