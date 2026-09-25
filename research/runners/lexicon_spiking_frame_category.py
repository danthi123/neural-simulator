"""LEXICON v2 — a referent (noun-category) decision computed by a COUPLED spiking winner-take-all whose graded drive
arrives through HEBBIAN-LEARNED frame->category synapses (language lane E, 2026-09-23).

WHY THIS REPLACES v1 (`lexicon_learned_referent.LearnedReferentLexicon`). The adversarial review of v1 (workflow
wf_d8f8c85b, review of `research/language-lane-next`) found that v1's "spiking two-pool WTA" was a RELAY: the two pools
of `_gap3_spiking_feature_compat_derisk._build` are uncoupled (internal_density 0, one weight-0 pathway), and
`classify()` computed `np.sign(host label-spread score)` and drove exactly ONE pool with a fixed current. The category
was computed by host label-spreading; the spikes only carried its sign. v1 is kept, relabelled "host-computed
category, spike-relayed", as the baseline its artifacts measured.

WHAT v2 IS (every step between the heard corpus and the decision is neurons/synapses):
  * ENVIRONMENT (host, legitimate): the corpus is the child's heard language. To present a word, the host samples K
    of its real occurrences and, per occurrence, activates the FRAME AFFERENTS that the heard neighbours excite:
    one afferent neuron per (offset, context-token) pair, offsets -2,-1,+1,+2 over the CTX_C most frequent tokens
    (function words included; they ARE the frames of Mintz 2003). Each occurrence drives its <=4 afferents for
    T_ON steps. No counting, PPMI, similarity graph or score is computed by the host.
  * SYNAPSES (learned): FR -> CN (referent pool) and FR -> CX (non-referent pool), all-to-all, starting from a
    uniform-with-jitter W_INIT. They are the ONLY learned edge.
  * COMPETITION (coupled): CN and CX each drive their own fast-spiking interneuron pool (IN, IX), which inhibits the
    OTHER pool (reciprocal lateral inhibition; Grossberg 1973; the motif of `_affect_marker_wta_derisk` and the BG
    SPEAK/SILENT selector). The winner is decided by the circuit: the pool whose graded synaptic drive is larger
    suppresses the other.
  * LEARNING (Hebbian, synapse-local, Oja 1982): during the teacher phase, the seed word's category pool receives a
    TEACHER current (the social environment naming the category: the curriculum = the same small seed lists v1 used,
    38 hand nouns + 37 non-nouns; cross-validation subsamples 12 per class). After each word presentation every
    FR->pool synapse updates   dw_ij = ETA * (x_i * y_j - OJA_BETA * y_j^2 * w_ij)   with x_i, y_j the pre/post SPIKE
    RATES the bridge produced over that presentation. Oja's multiplicative decay is the companion normalization
    process; there is NO clamp/bound (the CLAUDE.md "proxy dominates" lesson). The rule is applied runner-side to
    the bridge's own `cp_connections.data` entries for these synapses (the engine's generic Hebbian path clips every
    synapse to hebbian_max_weight, which would crush the inhibitory weights; so the rule is executed here, on these
    synapses only). Because the teacher supplies the post factor, the learning is "host-SUPERVISED / teacher-driven"
    (docs/TERMS.md), NOT "self-organized".
  * READ-OUT (instrument): the host reads which pool fired more over the presentation (and abstains if neither pool
    fires or the margin is within DEAD_MARGIN). The COMPETITION that decides it is spiking; the reading of the winner
    is a read-out, so the TERMS.md wording is "spiking with a host read-out", never "fully spiking".

LESIONS (each hits ONE named edge; `set_lesion(kind)`; verified to still hold at measurement via a weight hash):
  "learned_edge"  FR->CN/CX weights restored to W_INIT (the pre-learning synapses): the learned content is removed,
                  the circuit and its drive stay.            <- the lesion of the claimed learned component
  "competition"   IN->CX and IX->CN inhibitory weights zeroed: the pools no longer compete.
  "afferent_zero" FR->CN/CX weights zeroed: no drive at all -> abstain (an INTEGRITY smoke; passes by construction).

REPLICAS (`n_replicas`): one frame region FR feeds R independent category circuits (CN_r, CX_r, IN_r, IX_r). All
replicas hear the same words at the same time; each has its own teacher labels, own learned synapses and own decision.
Used for the >=1000-permutation null (replica r trained with permuted seed labels). Deployment uses R=1.

HONEST RESIDUALS (declared):
  * Supervised seed curriculum (teacher labels on 12+12 or 38+37 hand words); no self-organized category discovery.
  * The frame code is one afferent per (offset, token): a localist input code (sensory rendering of heard neighbours).
  * NOUN-hood, not REFERENT-hood (abstract nouns and noun/verb-ambiguous forms can be admitted).
  * Per-presentation state reset (membrane, recovery, conductances) = a washout convenience between words.
  * The winner is READ by the host from pool rates (read-out instrument).

Smoke: SIM_BACKEND=numpy python -m research.runners.lexicon_spiking_frame_category --smoke
"""
from __future__ import annotations

import collections
import os
import sys
import zlib

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.lexicon_learned_referent import HAND_NOUN_SEEDS, NONNOUN_SEEDS  # noqa: E402

_DEFAULT_CORPUS = os.path.join(_REPO, "data", "corpus", "tinystories.txt")

# ── circuit / protocol constants (calibrated on DEV seed 7, which is NOT in the 6-seed evaluation set) ──────────
CTX_C = 100                     # context tokens with a frame afferent per offset
OFFSETS = (-2, -1, 1, 2)
N_CAT = 20                      # excitatory neurons per category pool
N_FSI = 10                      # fast-spiking interneurons per pool's cross-inhibition sub-pool
K_OCC = 32                      # heard occurrences per word presentation
T_ON = 10                       # steps each occurrence drives its afferents
I_FRAME = 1200.0                # afferent drive while its frame is heard (pA)
W_INIT = 40.0                   # initial FR->category weight (uniform, jittered)
W_JITTER = 0.1
TO_FSI_W = 70.0                 # pool -> own FSI
CROSS_W = 22.0                  # FSI -> opposite pool (reciprocal lateral inhibition)
TEACHER_I = 400.0               # teacher current on the labelled pool during the teacher phase (pA)
ETA = 1000.0                    # Hebbian rate
OJA_BETA = 0.004                # Oja normalisation strength (sets the weight scale; w* = <xy>/(beta <y^2>))
EPOCHS = 8                      # passes over the seed curriculum
DEAD_MARGIN = 0.002             # rate margin the winner must clear (spikes/step/neuron)
MIN_RATE = 0.002                # below this in BOTH pools = no decision (abstain)

MIN_HEARD = 3                   # a word heard fewer times than this is never presented (abstain)

LESION_KINDS =("learned_edge", "competition", "afferent_zero")


def _stable_int(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


class FrameEnvironment:
    """The ENVIRONMENT half: the heard corpus. Maps a word to the frame afferents its real occurrences activate."""

    def __init__(self, tokens, vocab=(), ctx_c: int = CTX_C, offsets=OFFSETS, min_count: int = MIN_HEARD):
        cnt = collections.Counter(tokens)
        self.ctx = [w for w, _ in cnt.most_common(ctx_c)]
        cidx = {w: i for i, w in enumerate(self.ctx)}
        self.C, self.offsets = len(self.ctx), tuple(offsets)
        self.n_frame = self.C * len(self.offsets)
        self.tok_ctx = np.array([cidx.get(t, -1) for t in tokens], dtype=np.int32)
        # OPEN vocabulary: every word type heard >= min_count times can be presented (plus any explicit vocab word);
        # a word heard fewer times has no presentation -> the circuit is never driven -> abstain.
        vset = set(vocab) | {w for w, c in cnt.items() if c >= int(min_count)}
        pos = collections.defaultdict(list)
        for i, t in enumerate(tokens):
            if t in vset:
                pos[t].append(i)
        self.pos = {w: np.asarray(p, dtype=np.int64) for w, p in pos.items()}
        self.n_tok = len(tokens)

    def occurrences(self, word: str, k: int, seed: int):
        """K sampled occurrences (deterministic in (word, seed)); each = the list of active frame-afferent ids."""
        p = self.pos.get(word)
        if p is None or len(p) == 0:
            return None
        rng = np.random.default_rng((_stable_int(word) * 1000003 + int(seed)) & 0x7FFFFFFF)
        pick = rng.choice(p, size=k, replace=len(p) < k)
        occ = []
        for i in pick:
            feats = []
            for pj, off in enumerate(self.offsets):
                j = int(i) + off
                if 0 <= j < self.n_tok:
                    c = int(self.tok_ctx[j])
                    if c >= 0:
                        feats.append(pj * self.C + c)
            occ.append(feats)
        return occ


def _region(name, n, *, exc_fraction, neuron_type):
    from sim.regions import BrainRegion
    return BrainRegion(name=name, n_neurons=int(n), exc_fraction=float(exc_fraction), internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                       izh_neuron_type=neuron_type.name, enable_homeostasis=False)


def build_circuit(seed: int, n_frame: int, n_replicas: int = 1):
    """FR (n_frame afferents) -> R x {CN, CX} category pools with reciprocal FSI lateral inhibition."""
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
    regions = [_region("FR", n_frame, exc_fraction=1.0, neuron_type=rs)]
    pathways = []
    for r in range(n_replicas):
        for p in ("CN", "CX"):
            regions.append(_region(f"{p}{r}", N_CAT, exc_fraction=1.0, neuron_type=rs))
        for p in ("IN", "IX"):
            regions.append(_region(f"{p}{r}", N_FSI, exc_fraction=0.0, neuron_type=fs))
        for p in ("CN", "CX"):
            pathways.append(RegionPathway(from_region="FR", to_region=f"{p}{r}", density=1.0, weight_mean=W_INIT,
                                          weight_jitter=W_JITTER, plastic=False))
        pathways.append(RegionPathway(from_region=f"CN{r}", to_region=f"IN{r}", density=1.0, weight_mean=TO_FSI_W,
                                      weight_jitter=0.05, plastic=False))
        pathways.append(RegionPathway(from_region=f"CX{r}", to_region=f"IX{r}", density=1.0, weight_mean=TO_FSI_W,
                                      weight_jitter=0.05, plastic=False))
        pathways.append(RegionPathway(from_region=f"IN{r}", to_region=f"CX{r}", density=1.0, weight_mean=CROSS_W,
                                      weight_jitter=0.05, plastic=False, receptor="gaba_a"))
        pathways.append(RegionPathway(from_region=f"IX{r}", to_region=f"CN{r}", density=1.0, weight_mean=CROSS_W,
                                      weight_jitter=0.05, plastic=False, receptor="gaba_a"))
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b


def _synapse_slots(b, pre_idx, post_groups):
    """Return S (len(pre) x total_post) of cp_connections.data indices for pre -> each post neuron in post_groups
    (a list of index arrays, concatenated in order). CSR: row = pre, col = post."""
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
        raise RuntimeError("frame->category pathway is not all-to-all (missing synapses)")
    return S


class SpikingFrameCategoryLexicon:
    """See module docstring. `train(labels_per_replica)` then `classify(word)` / `decide(word)`."""

    variant = "frame"   # get_lexicon() rebuilds the singleton when the requested variant differs

    def __init__(self, seed: int, env: FrameEnvironment, n_replicas: int = 1, *, eta=ETA, oja_beta=OJA_BETA,
                 teacher_i=TEACHER_I, i_frame=I_FRAME, k_occ=K_OCC, t_on=T_ON, epochs=EPOCHS):
        from sim.backend import to_host
        self.seed, self.env, self.R = int(seed), env, int(n_replicas)
        self.eta, self.beta, self.teacher_i, self.i_frame = float(eta), float(oja_beta), float(teacher_i), float(i_frame)
        self.k_occ, self.t_on, self.epochs = int(k_occ), int(t_on), int(epochs)
        self.b = build_circuit(self.seed, env.n_frame, self.R)
        rm = self.b.region_manager
        self.fr = np.asarray(list(rm.indices("FR")), dtype=np.int64)
        self.cn = [np.asarray(list(rm.indices(f"CN{r}")), dtype=np.int64) for r in range(self.R)]
        self.cx = [np.asarray(list(rm.indices(f"CX{r}")), dtype=np.int64) for r in range(self.R)]
        self.inn = [np.asarray(list(rm.indices(f"IN{r}")), dtype=np.int64) for r in range(self.R)]
        self.ixx = [np.asarray(list(rm.indices(f"IX{r}")), dtype=np.int64) for r in range(self.R)]
        self.n = int(self.b.core_config.num_neurons)
        # learned edge: FR -> [CN0, CX0, CN1, CX1, ...]
        self.post_groups = []
        for r in range(self.R):
            self.post_groups += [self.cn[r], self.cx[r]]
        self.S = _synapse_slots(self.b, self.fr, self.post_groups)                   # (n_frame, R*2*N_CAT)
        # competition edge: IN_r -> CX_r and IX_r -> CN_r
        inh_rows = []
        for r in range(self.R):
            inh_rows.append(_synapse_slots(self.b, self.inn[r], [self.cx[r]]).ravel())
            inh_rows.append(_synapse_slots(self.b, self.ixx[r], [self.cn[r]]).ravel())
        self.S_inh = np.concatenate(inh_rows)
        data = np.asarray(to_host(self.b.cp_connections.data))
        self.W_init = data[self.S].astype(np.float64).copy()
        self.W = self.W_init.copy()
        self.inh_init = data[self.S_inh].astype(np.float64).copy()
        self.lesion = None
        self._cache = {}
        self._install()

    # ── weights on the bridge ────────────────────────────────────────────────────────────────────────────────
    def _install(self):
        from sim.backend import to_host, from_host
        data = np.asarray(to_host(self.b.cp_connections.data)).copy()
        W = self.W
        if self.lesion == "learned_edge":
            W = self.W_init
        elif self.lesion == "afferent_zero":
            W = np.zeros_like(self.W)
        data[self.S] = W
        data[self.S_inh] = 0.0 if self.lesion == "competition" else self.inh_init
        self.b.cp_connections.data = from_host(data.astype(np.float32))
        self._w_hash = self.weight_hash()

    def weight_hash(self) -> str:
        import hashlib
        from sim.backend import to_host
        return hashlib.sha256(np.asarray(to_host(self.b.cp_connections.data)).tobytes()).hexdigest()

    def set_lesion(self, kind):
        if kind not in (None,) + LESION_KINDS:
            raise ValueError(kind)
        if kind != self.lesion:
            self.lesion = kind
            self._install()

    # ── one presentation ─────────────────────────────────────────────────────────────────────────────────────
    def _reset_state(self):
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

    def present(self, word: str, teacher=None):
        """Present `word` (K heard occurrences). teacher: None, or a length-R array of +1 (CN) / -1 (CX) / 0.
        Returns (counts over all neurons, n_steps) or (None, 0) if the word was never heard."""
        from sim.backend import to_host, from_host
        occ = self.env.occurrences(word, self.k_occ, self.seed)
        if occ is None:
            return None, 0
        self._reset_state()
        base = np.zeros(self.n, dtype=np.float64)
        if teacher is not None:
            for r in range(self.R):
                if teacher[r] > 0:
                    base[self.cn[r]] = self.teacher_i
                elif teacher[r] < 0:
                    base[self.cx[r]] = self.teacher_i
        counts = np.zeros(self.n, dtype=np.float64)
        steps = 0
        b = self.b
        for feats in occ:
            cur = base.copy()
            if feats:
                cur[self.fr[np.asarray(feats, dtype=np.int64)]] += self.i_frame
            dev = from_host(cur.astype(np.float32))
            for _ in range(self.t_on):
                b.cp_external_input_current[:] = dev
                b._run_one_simulation_step()
                counts += np.asarray(to_host(b.cp_firing_states), dtype=np.float64)
                steps += 1
        b.cp_external_input_current[:] = 0.0
        return counts, steps

    def _rates(self, counts, steps):
        rn = np.array([counts[self.cn[r]].mean() / steps for r in range(self.R)])
        rx = np.array([counts[self.cx[r]].mean() / steps for r in range(self.R)])
        return rn, rx

    # ── learning (Hebbian, synapse-local, Oja) ───────────────────────────────────────────────────────────────
    def hebbian_update(self, counts, steps):
        x = counts[self.fr] / steps                                        # pre rates   (n_frame,)
        y = np.concatenate([counts[g] for g in self.post_groups]) / steps  # post rates  (R*2*N_CAT,)
        self.W += self.eta * (np.outer(x, y) - self.beta * (y * y)[None, :] * self.W)

    def train(self, words, labels, order_seed=None):
        """words: list; labels: array (len(words), R) of +1/-1 teacher labels per replica."""
        labels = np.asarray(labels, dtype=np.float64).reshape(len(words), self.R)
        rng = np.random.default_rng(self.seed if order_seed is None else order_seed)
        saved, self.lesion = self.lesion, None
        self._install()
        for _ in range(self.epochs):
            for i in rng.permutation(len(words)):
                counts, steps = self.present(words[i], teacher=labels[i])
                if counts is None:
                    continue
                self.hebbian_update(counts, steps)
                self._install()
        self.lesion = saved
        self._install()
        self._cache.clear()

    def self_train(self, words, epochs: int = 1, order_seed=None):
        """UNSUPERVISED phase (competitive Hebbian; Rumelhart & Zipser 1985): unlabelled heard words, NO teacher.
        The post factor is whatever the coupled WTA itself produced, so here neither factor is host-supplied."""
        rng = np.random.default_rng((self.seed * 7 + 3) if order_seed is None else order_seed)
        saved, self.lesion = self.lesion, None
        self._install()
        for _ in range(int(epochs)):
            for i in rng.permutation(len(words)):
                counts, steps = self.present(words[i], teacher=None)
                if counts is None:
                    continue
                self.hebbian_update(counts, steps)
                self._install()
        self.lesion = saved
        self._install()
        self._cache.clear()

    def reset_learning(self):
        self.W = self.W_init.copy()
        self._install()
        self._cache.clear()

    # ── decision ─────────────────────────────────────────────────────────────────────────────────────────────
    def decide(self, word: str):
        """Per replica: (decision list of True/False/None, rn, rx, graded_drive_margin). Plasticity is OFF here; the
        installed weights are hash-checked after the read (a lesion/weight must still hold at measurement)."""
        key = (word, self.lesion)
        if key in self._cache:
            return self._cache[key]
        h0 = self._w_hash
        counts, steps = self.present(word)
        if counts is None:
            out = ([None] * self.R, None, None)
        else:
            rn, rx = self._rates(counts, steps)
            dec = []
            for r in range(self.R):
                if max(rn[r], rx[r]) < MIN_RATE or abs(rn[r] - rx[r]) <= DEAD_MARGIN:
                    dec.append(None)
                else:
                    dec.append(bool(rn[r] > rx[r]))
            out = (dec, rn, rx)
        if self.weight_hash() != h0:
            raise RuntimeError("weights changed during a read — a lesion/weight did not hold at measurement")
        self._cache[key] = out
        return out

    def graded_drive(self, word: str):
        """The synaptic drive margin (CN minus CX) a word's heard frames deliver through the installed weights —
        diagnostic only (not used by the decision)."""
        occ = self.env.occurrences(word, self.k_occ, self.seed)
        if occ is None:
            return None
        x = np.zeros(self.env.n_frame)
        for f in occ:
            x[f] += 1
        data_w = {None: self.W, "learned_edge": self.W_init, "afferent_zero": 0 * self.W}.get(self.lesion, self.W)
        d = x @ data_w                                                    # (R*2*N_CAT,)
        d = d.reshape(self.R, 2, N_CAT).mean(axis=2)
        return d[:, 0] - d[:, 1]

    def classify(self, word: str, replica: int = 0):
        return self.decide(word)[0][replica]

    def is_referent(self, word: str) -> bool:
        return self.classify(word) is True


def seed_curriculum(env: FrameEnvironment, k_seed=None, cv_seed=None):
    """(words, labels) from the hand seed lists present in the corpus; k_seed subsamples per class."""
    pos = [w for w in HAND_NOUN_SEEDS if w in env.pos]
    neg = [w for w in NONNOUN_SEEDS if w in env.pos]
    if k_seed is not None:
        rng = np.random.default_rng(cv_seed)
        pos = list(rng.permutation(pos))[:k_seed]
        neg = list(rng.permutation(neg))[:k_seed]
    words = [str(w) for w in pos] + [str(w) for w in neg]
    labels = np.array([1.0] * len(pos) + [-1.0] * len(neg))
    return words, labels


_LEXICON = None


def _junction_requested() -> bool:
    """`BRAIN_LEARNED_REFERENT_JUNCTION` in {1,true,yes,on} -> the frame-junction variant
    (`lexicon_frame_junction.FrameJunctionLexicon`; pre-registration
    research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md). DEFAULT OFF: unset -> the
    v2 lexicon below, and the junction module is never imported."""
    v = os.environ.get("BRAIN_LEARNED_REFERENT_JUNCTION")
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def get_lexicon(seed: int = 42, corpus_path=None, max_chars: int = 8_000_000, top_v: int = 2000):
    """The process-shared DEPLOYMENT lexicon (built + trained once, lazily): all 38 + 37 seeds, R=1. The variant
    follows `BRAIN_LEARNED_REFERENT_JUNCTION` (default: v2, this module's class)."""
    global _LEXICON
    want = "junction" if _junction_requested() else "frame"
    if _LEXICON is None or getattr(_LEXICON, "variant", "frame") != want:
        from research.runners._comprehension_learned_animacy_cue_derisk import load_tokens, build_vocab
        if want == "junction":
            # AMENDMENT 2 mechanism C: the junction variant hears an explicit sentence-boundary PAUSE token instead
            # of the shared tokenizer's silent punctuation-stripping (see lexicon_frame_junction.py's module
            # docstring). v2 (want == "frame") is UNTOUCHED -- it still calls the shared `load_tokens` below, so its
            # default-OFF byte-identity is unaffected by this branch existing.
            from research.runners.lexicon_frame_junction import load_tokens_with_pause
            tokens = load_tokens_with_pause(corpus_path or _DEFAULT_CORPUS, max_chars)
        else:
            tokens = load_tokens(corpus_path or _DEFAULT_CORPUS, max_chars)
        vocab, _ = build_vocab(tokens, top_v)
        env = FrameEnvironment(tokens, vocab + [w for w in HAND_NOUN_SEEDS + NONNOUN_SEEDS if w not in vocab])
        if want == "junction":
            from research.runners.lexicon_frame_junction import FrameJunctionLexicon
            lex = FrameJunctionLexicon(seed, env, n_replicas=1)
        else:
            lex = SpikingFrameCategoryLexicon(seed, env, n_replicas=1)
        words, labels = seed_curriculum(env)
        lex.train(words, labels[:, None])
        _LEXICON = lex
    return _LEXICON


if __name__ == "__main__":
    import argparse
    import time
    os.environ.setdefault("SIM_BACKEND", "numpy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    lex = get_lexicon(seed=a.seed)
    print(f"built+trained in {time.time() - t0:.1f}s; n={lex.n}")
    for w in ("owl", "monkey", "garden", "cookie", "watches", "jumped", "beautiful", "slowly", "wolf"):
        print(w, lex.decide(w), lex.graded_drive(w))
    # ATTRIBUTION: how much of the intact CN-CX rate margin for "owl" is carried by the LEARNED synapses (control =
    # the same circuit with FR->category restored to its pre-learning weights)?
    from tools.lab import attributable_to
    _, rn, rx = lex.decide("owl")
    lex.set_lesion("learned_edge")
    _, rn0, rx0 = lex.decide("owl")
    lex.set_lesion(None)
    attributable_to("owl CN-CX margin: learned vs pre-learning synapses", float(rn[0] - rx[0]), float(rn0[0] - rx0[0]))
